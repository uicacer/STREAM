"""
Cloud model catalog and API key validation endpoints.

This module provides two key capabilities:

1. API KEY VALIDATION:
   When a user enters an API key in the STREAM settings panel, we need to
   verify it works before they try to send a chat message. The validation
   endpoint makes a minimal test call to the provider and returns whether
   the key is valid.

2. DYNAMIC MODEL CATALOG (OpenRouter):
   OpenRouter provides 500+ models through a single API. Instead of
   hardcoding all of them in STREAM's config, we fetch the catalog
   dynamically from OpenRouter's /api/v1/models endpoint.

   The catalog is cached on the backend for 1 hour to avoid hammering
   OpenRouter's API on every settings panel open. The frontend adds its
   own 5-minute cache on top of this.

WHY PROXY THROUGH THE BACKEND?
-------------------------------
The frontend can't call OpenRouter's API directly because of CORS
(Cross-Origin Resource Sharing). Browsers block requests to different
domains unless the server explicitly allows it. By proxying through
our backend, we:
  1. Avoid CORS issues entirely
  2. Cache results server-side (saves bandwidth for all users)
  3. Can filter/categorize models before sending to the frontend
  4. Don't expose the user's API key in browser network logs to
     third-party domains
"""

import logging
import time

import httpx
import litellm
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# =============================================================================
# API KEY VALIDATION
# =============================================================================


class ValidateKeyRequest(BaseModel):
    """Request body for API key validation.

    The frontend sends this when the user enters or changes an API key
    in the settings panel. We validate the key before the user tries
    to use it for an actual chat.
    """

    provider: str = Field(
        ...,
        description=("Which provider to validate: 'openrouter', 'anthropic', or 'openai'"),
    )
    api_key: str = Field(
        ...,
        description="The API key to validate",
    )


# Map provider names to the litellm model string used for the test call.
# We use the cheapest/fastest model from each provider to minimize cost
# and latency during validation.
_VALIDATION_MODELS = {
    "openrouter": "openrouter/openai/gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "openai": "gpt-4o-mini",
}


@router.post("/health/validate-key")
async def validate_api_key(request: ValidateKeyRequest):
    """
    Validate a user-provided API key by making a minimal test call.

    This endpoint:
    1. Takes a provider name and API key
    2. Makes a 1-token completion call to the cheapest model
    3. Returns whether the key is valid

    The test call uses max_tokens=1 and a trivial prompt to minimize
    cost (fractions of a cent) and latency (~200-500ms).

    Returns:
        {"valid": true}  — key works
        {"valid": false, "error": "Invalid API key"}  — key is invalid
        {"valid": false, "error": "Rate limited"}  — key works but rate limited

    Why not just try the key on the first real chat?
    Because a failed chat is a bad user experience — they type a long
    question, wait, then get an error. Validating upfront catches the
    problem immediately with clear feedback.
    """
    if request.provider not in _VALIDATION_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {request.provider}. "
            f"Valid: {list(_VALIDATION_MODELS.keys())}",
        )

    test_model = _VALIDATION_MODELS[request.provider]

    try:
        # Make a minimal test call — 1 token, trivial prompt.
        # litellm.completion() is synchronous but fast for 1 token.
        # We pass the user's key directly via api_key parameter.
        litellm.completion(
            model=test_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            api_key=request.api_key,
        )
        return {"valid": True}

    except litellm.AuthenticationError:
        return {"valid": False, "error": "Invalid API key"}

    except litellm.RateLimitError:
        # Rate limited means the key IS valid — just throttled.
        # This happens with free-tier OpenRouter keys (20 req/min).
        return {"valid": True, "warning": "Key is valid but currently rate-limited"}

    except litellm.BadRequestError as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["billing", "credit", "quota", "exceeded"]):
            return {"valid": False, "error": "Account has no credits or billing issue"}
        return {"valid": False, "error": f"Bad request: {e}"}

    except Exception as e:
        logger.warning(f"Key validation failed for {request.provider}: {e}")
        return {"valid": False, "error": f"Validation failed: {e}"}


# =============================================================================
# DYNAMIC MODEL CATALOG (OpenRouter)
# =============================================================================
#
# OpenRouter's /api/v1/models endpoint returns ALL available models with
# their pricing, context length, and capabilities. We cache this and
# serve it to the frontend in a structured format.

# Cache for the OpenRouter model catalog.
# We store the response and a timestamp so we can serve stale data
# while fetching fresh data in the background.
_catalog_cache: dict = {
    "data": None,
    "fetched_at": 0,
}
_CATALOG_TTL_SECONDS = 3600  # 1 hour


@router.get("/models/catalog")
async def get_model_catalog(
    openrouter_api_key: str | None = None,
):
    """
    Fetch and return the OpenRouter model catalog.

    This endpoint:
    1. Checks the 1-hour cache first
    2. If stale, fetches from OpenRouter's /api/v1/models
    3. Categorizes models (recommended, free, vision, etc.)
    4. Returns structured data for the frontend model browser

    Query Parameters:
        openrouter_api_key: Optional. Not required for catalog browsing
                           (OpenRouter's model list is public), but
                           including it shows user-specific model access.

    Returns:
        {
            "models": [...],        # Full model list
            "recommended": [...],   # Curated picks for STREAM
            "free": [...],          # Free-tier models
            "categories": {         # Models grouped by provider
                "Anthropic": [...],
                "OpenAI": [...],
                ...
            }
        }

    The frontend calls this when the user expands the "Browse All Models"
    section in the settings panel. Results are cached on the frontend
    for 5 minutes to avoid repeated requests during a session.
    """
    now = time.time()

    # Return cached data if fresh enough
    if _catalog_cache["data"] and (now - _catalog_cache["fetched_at"]) < _CATALOG_TTL_SECONDS:
        return _catalog_cache["data"]

    # Fetch from OpenRouter
    try:
        headers = {}
        if openrouter_api_key:
            headers["Authorization"] = f"Bearer {openrouter_api_key}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenRouter API returned {response.status_code}",
            )

        raw_models = response.json().get("data", [])

    except httpx.TimeoutException as e:
        # If we have stale cached data, return it rather than failing
        if _catalog_cache["data"]:
            logger.warning("OpenRouter catalog fetch timed out, returning stale cache")
            return _catalog_cache["data"]
        raise HTTPException(
            status_code=504,
            detail="OpenRouter API timed out",
        ) from e

    except Exception as e:
        if _catalog_cache["data"]:
            logger.warning(f"OpenRouter catalog fetch failed ({e}), returning stale cache")
            return _catalog_cache["data"]
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch OpenRouter catalog: {e}",
        ) from e

    # Process and categorize models.
    # We extract only the fields the frontend needs to keep the
    # response payload small (OpenRouter returns ~500 models).
    models = []
    free_models = []
    recommended_ids = {
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-exp",
        "meta-llama/llama-3.1-70b-instruct",
    }
    recommended = []
    categories: dict[str, list] = {}

    for m in raw_models:
        model_id = m.get("id", "")
        pricing = m.get("pricing", {})

        # Skip models without pricing info (usually deprecated or preview)
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        if prompt_price is None or completion_price is None:
            continue

        # Extract provider from model ID (e.g., "anthropic/claude-sonnet-4" → "Anthropic")
        provider = model_id.split("/")[0].title() if "/" in model_id else "Unknown"

        # Determine capabilities from the model's architecture info.
        # OpenRouter provides modality data in two fields:
        #   - input_modalities: ["text", "image", "file"] (list of supported inputs)
        #   - output_modalities: ["text"] (list of supported outputs)
        #   - modality: "text+image->text" (legacy string summary)
        arch = m.get("architecture", {})
        if not isinstance(arch, dict):
            arch = {}
        modality_input = arch.get("input_modalities", [])
        if not isinstance(modality_input, list):
            modality_input = []
        modality_output = arch.get("output_modalities", [])
        if not isinstance(modality_output, list):
            modality_output = []
        supports_vision = "image" in modality_input

        # Check if this is a free model
        is_free = (float(prompt_price) == 0 and float(completion_price) == 0) or ":free" in model_id

        model_entry = {
            "id": model_id,
            "name": m.get("name", model_id),
            "provider": provider,
            "description": m.get("description", ""),
            "context_length": m.get("context_length", 0),
            "pricing": {
                "prompt": float(prompt_price),
                "completion": float(completion_price),
                "prompt_display": f"${float(prompt_price) * 1_000_000:.2f}/1M",
                "completion_display": f"${float(completion_price) * 1_000_000:.2f}/1M",
            },
            "is_free": is_free,
            "supports_vision": supports_vision,
            "modality_input": modality_input,
            "modality_output": modality_output,
            "top_provider": m.get("top_provider", {}),
        }

        models.append(model_entry)

        if is_free:
            free_models.append(model_entry)

        if model_id in recommended_ids:
            recommended.append(model_entry)

        if provider not in categories:
            categories[provider] = []
        categories[provider].append(model_entry)

    # Sort: recommended first by our preferred order, then alphabetically
    recommended_order = list(recommended_ids)
    recommended.sort(
        key=lambda x: (recommended_order.index(x["id"]) if x["id"] in recommended_order else 999)
    )

    # Sort free models by name
    free_models.sort(key=lambda x: x["name"])

    # Sort all models by provider then name
    models.sort(key=lambda x: (x["provider"], x["name"]))

    result = {
        "models": models,
        "recommended": recommended,
        "free": free_models,
        "categories": categories,
        "total_count": len(models),
        "free_count": len(free_models),
    }

    # Update cache
    _catalog_cache["data"] = result
    _catalog_cache["fetched_at"] = now

    logger.info(
        f"Fetched OpenRouter catalog: {len(models)} models "
        f"({len(free_models)} free, {len(recommended)} recommended)"
    )

    return result
