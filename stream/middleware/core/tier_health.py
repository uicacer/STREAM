"""
Tier health checking and availability tracking.

This module manages the health status of all AI service tiers:
- LOCAL (Ollama)
- LAKESHORE (vLLM)
- CLOUD (LiteLLM → Anthropic/OpenAI)
"""

import logging
import os
import time
from datetime import datetime, timedelta

import httpx
import litellm

import stream.proxy.app as _proxy_app
from stream.middleware.config import (
    CLOUD_PROVIDERS,
    DEFAULT_MODELS,
    HEALTH_CHECK_TTL,
    LAKESHORE_HEALTH_TIMEOUT,
    LAKESHORE_PROXY_URL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODELS,
    STREAM_MODE,
)
from stream.middleware.core.globus_auth import is_authenticated as globus_is_authenticated

# In desktop mode, we call litellm as a Python library (no HTTP gateway).
# _resolve_model() translates STREAM's internal model names into the kwargs
# that litellm.completion() needs. The two naming systems are different because:
#
#   STREAM names    → Human-readable tier identifiers used throughout the app
#                     e.g., "cloud-claude", "cloud-gpt", "local-llama"
#
#   litellm kwargs  → The actual provider model IDs + connection details that
#                     litellm needs to make API calls to the right provider
#                     e.g., model="claude-sonnet-4-20250514" (Anthropic API)
#                           model="gpt-4-turbo" (OpenAI API)
#                           model="ollama/llama3.2:3b", api_base="http://localhost:11434"
#
# The mapping is defined in stream/gateway/litellm_config.yaml and loaded once
# at import time by litellm_direct.py.
from stream.middleware.core.litellm_direct import _resolve_model

logger = logging.getLogger(__name__)

# Track health status (module-level state)
# error_type can be: None, "auth", "connection", "timeout", "unknown"
_tier_health = {
    "local": {"available": False, "last_check": None, "error": None, "error_type": None},
    "lakeshore": {"available": False, "last_check": None, "error": None, "error_type": None},
    "cloud": {"available": False, "last_check": None, "error": None, "error_type": None},
}

# Tracks the user's currently selected cloud provider.
# At startup this is None → check_all_tiers() uses the default from config.
# Once the frontend sends a health poll or chat request with cloud_provider,
# this gets updated. After that, ALL cloud health checks (including the
# background monitor) test the user's ACTUAL selection, not the default.
_active_cloud_provider: str | None = None


def set_active_cloud_provider(provider: str | None):
    """Update the active cloud provider (called when frontend sends cloud_provider)."""
    global _active_cloud_provider
    if provider:
        _active_cloud_provider = provider


def mark_tier_unavailable(
    tier: str,
    error: str,
    lakeshore_model: str | None = None,
) -> None:
    """
    Mark a tier as unavailable after a real query failure.

    Called by the streaming code when a request to a tier fails at runtime.
    The next on-demand health check (e.g., user changes tier or model in
    settings) will re-check and restore the indicator if the tier recovers.

    For Lakeshore, each model runs on a separate vLLM port on the HPC cluster
    (e.g., qwen-1.5b on :8000, coder-1.5b on :8001, qwen-32b on :8004).
    A single model being down (e.g., 32B not running) should NOT mark
    the entire Lakeshore tier red — other models may still work.
    So we use per-model cache keys: "lakeshore:lakeshore-qwen-32b".

    Args:
        tier: The tier to mark ("local", "lakeshore", or "cloud")
        error: Error message describing the failure
        lakeshore_model: For lakeshore tier, the specific model that failed.
                        When provided, only that model is marked unavailable.
                        When None, the entire tier is marked unavailable.
    """
    # Build cache key: use per-model key for lakeshore so each vLLM model
    # is tracked independently (same pattern as cloud:{provider} and local:{model})
    cache_key = tier
    if tier == "lakeshore" and lakeshore_model:
        cache_key = f"lakeshore:{lakeshore_model}"

    _tier_health[cache_key] = {
        "available": False,
        "error": error,
        "error_type": "connection",
        "last_check": datetime.now(),
    }
    logger.warning(f"[Health] Marked {cache_key} as unavailable: {error}")


def check_tier_health(
    tier: str,
    cloud_provider: str | None = None,
    local_model: str | None = None,
    lakeshore_model: str | None = None,
) -> tuple[bool, str | None]:
    """
    Check if a specific tier is available.

    This is the SINGLE health check function for all tiers and all callers
    (startup, background monitor, frontend polling). Having one function
    eliminates duplication and ensures bug fixes apply everywhere.

    Args:
        tier: The tier to check ("local", "lakeshore", or "cloud")
        cloud_provider: For cloud tier, test a specific provider (e.g., "cloud-gpt")
                       instead of the default. Each provider has its own API key
                       and billing — Claude being down shouldn't block GPT.
        local_model: For local tier, test a specific model (e.g., "local-llama-quality")
                    instead of the default. Verifies the model is installed in Ollama.
        lakeshore_model: For lakeshore tier, the specific model to test
                        (e.g., "lakeshore-qwen-32b"). We can't ping vLLM ports
                        directly (they're behind Globus on the HPC), so this is
                        used to check if a previous inference attempt failed for
                        this specific model.

    Returns:
        Tuple of (is_available, error_message). error_message is None if available.
    """
    # Use per-tier model override or fall back to default
    if tier == "cloud" and cloud_provider:
        model = cloud_provider
    elif tier == "local" and local_model:
        model = local_model
    else:
        model = DEFAULT_MODELS.get(tier)
    if not model:
        return False, f"No model configured for tier {tier}"

    try:
        # LOCAL: Check Ollama directly
        if tier == "local":
            with httpx.Client(timeout=5.0) as client:
                # Get the Ollama model name for this tier
                ollama_model = OLLAMA_MODELS.get(model)
                if not ollama_model:
                    return False, f"No Ollama model mapping for {model}"

                # Use /api/show to verify the model exists AND is usable.
                # This is more reliable than /api/tags + name matching because:
                #   1. Ollama resolves aliases itself (same logic as /api/chat)
                #   2. No prefix matching needed — if /api/show succeeds, inference will too
                #   3. Catches corrupted/incomplete models that /api/tags still lists
                response = client.post(
                    f"{OLLAMA_BASE_URL}/api/show",
                    json={"name": ollama_model},
                )
                if response.status_code != 200:
                    return False, f"Model {ollama_model} not installed in Ollama"

                return True, None

        # LAKESHORE: Two-level health check.
        #
        # Level 1 (always runs, cheap ~100ms):
        #   Verify Globus auth + proxy reachable. If this fails, ALL lakeshore
        #   models are unavailable — no point testing individual ones.
        #
        # Level 2 (only when lakeshore_model is specified, ~5-15s):
        #   Send a real 1-token inference through Globus to the specific model's
        #   vLLM port. This confirms the model is actually running on the HPC.
        #
        # Why we need Level 2:
        #   Each model runs as a separate vLLM instance on a different port
        #   (e.g., qwen-1.5b on :8000, qwen-32b on :8004). The base check
        #   (Level 1) only confirms "Globus is reachable" — it can't tell us
        #   if a specific vLLM port is up. Without Level 2, switching to a
        #   model that isn't running would still show a green health indicator.
        #
        # When does Level 2 run?
        #   - When the frontend sends lakeshore_model in the health poll
        #     (happens when user changes model selection in settings)
        #   - Results are cached with a per-model key ("lakeshore:lakeshore-qwen-32b")
        #   - Subsequent polls use the cached result until TTL expires
        #   - So the slow check only happens once per model change, not every 30s
        #
        elif tier == "lakeshore":
            # === LEVEL 1: Base infrastructure check ===
            if STREAM_MODE == "desktop":
                # Desktop mode: proxy is embedded in the same server process.
                # Check the Globus client directly instead of HTTP (which would
                # be the server calling itself on port 5000 — blocks on startup
                # when the server isn't accepting connections yet).
                gc = _proxy_app.globus_client
                if not gc or not gc.is_available():
                    return False, "Globus Compute not configured or unavailable"
                if not globus_is_authenticated():
                    return False, "Globus Compute authentication required"

                # === LEVEL 2: Per-model vLLM port check (desktop mode) ===
                # If a specific model was requested, send a real 1-token
                # inference through Globus to verify the vLLM instance on
                # that port is actually running and can generate output.
                if lakeshore_model:
                    logger.info(
                        f"[Health] Running Level 2 check for {lakeshore_model} "
                        f"(1-token inference via Globus, timeout={LAKESHORE_HEALTH_TIMEOUT}s)"
                    )
                    return gc.check_model_health(
                        model=lakeshore_model,
                        timeout=LAKESHORE_HEALTH_TIMEOUT,
                    )

                # No specific model requested — base check passed, tier is healthy
                return True, None

            # Server mode: proxy runs as a separate container, check via HTTP
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"{LAKESHORE_PROXY_URL}/health")
                    if response.status_code != 200:
                        return (
                            False,
                            f"Lakeshore proxy not responding (HTTP {response.status_code})",
                        )
                    health_data = response.json()
                    if health_data.get("status") != "healthy":
                        return False, "Lakeshore proxy unhealthy"
                    if health_data.get("globus_authenticated") is False:
                        return False, "Globus Compute authentication required"
            except httpx.ConnectError:
                return False, "Cannot connect to Lakeshore proxy. Is the proxy service running?"
            except Exception as e:
                return False, f"Lakeshore proxy error: {str(e)}"

            # === LEVEL 2: Per-model check (server mode) ===
            # In server mode, the Globus client lives in the proxy container.
            # Send a minimal inference request through the proxy to test the
            # specific model's vLLM port.
            if lakeshore_model:
                try:
                    logger.info(
                        f"[Health] Running Level 2 check for {lakeshore_model} "
                        f"via proxy (timeout={LAKESHORE_HEALTH_TIMEOUT}s)"
                    )
                    with httpx.Client(timeout=float(LAKESHORE_HEALTH_TIMEOUT)) as client:
                        # Send a 1-token inference through the proxy.
                        # The proxy routes this to Globus → HPC → vLLM on the
                        # model's port. If the port isn't serving, we get an error.
                        response = client.post(
                            f"{LAKESHORE_PROXY_URL}/v1/chat/completions",
                            json={
                                "model": lakeshore_model,
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 1,
                                "temperature": 0.0,
                            },
                        )
                        if response.status_code == 200:
                            return True, None
                        else:
                            error_msg = (
                                response.text[:150]
                                if response.text
                                else f"HTTP {response.status_code}"
                            )
                            return False, f"Model not responding: {error_msg}"
                except httpx.TimeoutException:
                    return (
                        False,
                        f"Model not responding (timed out after {LAKESHORE_HEALTH_TIMEOUT}s)",
                    )
                except Exception as e:
                    return False, f"Model health check error: {str(e)[:150]}"

            return True, None

        # CLOUD: Verify the cloud provider is reachable and the API key is valid.
        # Makes a minimal 1-token test call. Works for ANY provider litellm supports
        # (Anthropic, OpenAI, DeepSeek, GLM, etc.) — no hardcoded provider names.
        elif tier == "cloud":
            if STREAM_MODE == "desktop":
                # Desktop mode: call litellm as a library (no HTTP gateway).
                # First, check that the required API key env var is actually set.
                # CLOUD_PROVIDERS maps each model to its env var:
                #   "cloud-claude" → "ANTHROPIC_API_KEY"
                #   "cloud-gpt"   → "OPENAI_API_KEY"
                # This lookup is generic — any new provider just needs an entry
                # in CLOUD_PROVIDERS with its env_key.
                provider_info = CLOUD_PROVIDERS.get(model, {})
                env_key = provider_info.get("env_key", "")
                if env_key and not os.environ.get(env_key):
                    return False, f"[AUTH] Missing API key: {env_key} not set"

                # API key exists — make a real 1-token test call to verify
                # it's valid (not expired, not over quota, etc.).
                # Cost: <$0.001 per check (1 input token + 1 output token).
                # Retry once for transient network hiccups.
                for attempt in range(2):
                    try:
                        kwargs = _resolve_model(model)
                        litellm.completion(
                            messages=[{"role": "user", "content": "hi"}],
                            max_tokens=1,
                            temperature=0.0,
                            **kwargs,
                        )
                        return True, None
                    except litellm.AuthenticationError as e:
                        # Auth errors are permanent — retrying won't help
                        return False, f"[AUTH] Invalid API key: {str(e)[:100]}"
                    except litellm.RateLimitError:
                        # Rate-limited = key IS valid, just throttled temporarily
                        if attempt < 1:
                            time.sleep(1)
                            continue
                        return True, None
                    except litellm.BadRequestError as e:
                        # BadRequestError can mean two very different things:
                        #   1. Billing/credit issues (e.g. Anthropic returns 400
                        #      for expired credits instead of 401) → NOT valid
                        #   2. Actual bad request (wrong params) → key IS valid
                        # Check for billing keywords to distinguish the two cases.
                        error_lower = str(e).lower()
                        billing_keywords = [
                            "credit",
                            "balance",
                            "billing",
                            "subscription",
                            "payment",
                            "expired",
                            "quota",
                            "plan",
                        ]
                        if any(kw in error_lower for kw in billing_keywords):
                            return False, f"[AUTH] {str(e)[:100]}"
                        # Not a billing issue — key works fine
                        return True, None
                    except Exception as e:
                        # Transient errors (network blip, DNS) → retry once
                        if attempt < 1:
                            time.sleep(1)
                            continue
                        return False, f"Cloud provider error: {str(e)[:100]}"

            else:
                # Server mode: test through the LiteLLM HTTP gateway (port 4000).
                # The gateway runs as a Docker container and handles model routing.
                for attempt in range(2):
                    try:
                        with httpx.Client(timeout=10.0) as client:
                            response = client.post(
                                f"{LITELLM_BASE_URL}/v1/chat/completions",
                                json={
                                    "model": model,
                                    "messages": [{"role": "user", "content": "test"}],
                                    "max_tokens": 1,
                                    "temperature": 0.0,
                                },
                                headers={
                                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                                    "Content-Type": "application/json",
                                },
                            )

                            # 200 = provider responded successfully → tier is healthy
                            if response.status_code == 200:
                                return True, None

                            # Non-200: parse error to give a useful message
                            error_msg = ""
                            try:
                                error_data = response.json()
                                error_msg = str(
                                    error_data.get("detail", "")
                                    or error_data.get("error", {}).get("message", "")
                                    or error_data
                                )
                            except Exception:
                                error_msg = response.text[:200] if response.text else ""

                            # Check for auth/billing errors (provider-agnostic keywords)
                            error_lower = error_msg.lower()
                            auth_keywords = [
                                "credit",
                                "billing",
                                "balance",
                                "subscription",
                                "expired",
                                "api key",
                                "unauthorized",
                                "forbidden",
                                "quota",
                                "payment",
                            ]
                            if any(kw in error_lower for kw in auth_keywords):
                                return False, f"[AUTH] {error_msg}"

                            return (
                                False,
                                f"HTTP {response.status_code}: {error_msg[:100]}"
                                if error_msg
                                else f"HTTP {response.status_code}",
                            )

                    except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                        if attempt < 1:  # Not last attempt (range(2) → 0, 1)
                            time.sleep(2)
                            continue
                        return False, f"Connection failed after 2 attempts: {str(e)}"

        return False, "Unknown tier"

    except httpx.TimeoutException:
        return False, "Timeout"
    except httpx.ConnectError:
        return False, "Connection refused"
    except Exception as e:
        return False, str(e)


def _determine_error_type(error: str | None) -> str | None:
    """Determine error type from error message."""
    if error is None:
        return None
    if error.startswith("[AUTH]"):
        return "auth"
    if "timeout" in error.lower():
        return "timeout"
    if "connection" in error.lower() or "connect" in error.lower():
        return "connection"
    return "unknown"


def update_tier_health(tier: str):
    """
    Update health status for a tier (used by startup and on-demand checks).

    For cloud tier: uses the user's active cloud provider if set.
    At startup (before any frontend request), _active_cloud_provider is None
    so it falls back to the default. Once the user's selection arrives via
    a health poll or chat request, all subsequent checks use their provider.
    """
    # For cloud tier, use the user's active provider (not always the default).
    cloud_provider = _active_cloud_provider if tier == "cloud" else None
    is_available, error = check_tier_health(tier, cloud_provider=cloud_provider)
    error_type = _determine_error_type(error)

    # Use provider-specific cache key when we have a specific provider,
    # so it matches what is_tier_available() and the health endpoint look up.
    cache_key = tier
    if tier == "cloud" and cloud_provider:
        cache_key = f"cloud:{cloud_provider}"

    _tier_health[cache_key] = {
        "available": is_available,
        "last_check": datetime.now(),
        "error": error,
        "error_type": error_type,
    }

    model = cloud_provider or DEFAULT_MODELS.get(tier, "unknown")

    if is_available:
        logger.info(f"  ✓ {tier.upper():12s} {model}")
    else:
        logger.warning(f"  {tier.upper():12s} {model} — {error}")


def is_tier_available(
    tier: str,
    ttl: int = HEALTH_CHECK_TTL,
    cloud_provider: str | None = None,
    local_model: str | None = None,
    lakeshore_model: str | None = None,
) -> bool:
    """
    Check if tier is available (with caching).

    Uses cached status if available and fresh (within TTL).
    If cache expired, runs a fresh check_tier_health() call.

    Args:
        tier: The tier to check ("local", "lakeshore", or "cloud")
        ttl: Cache TTL in seconds. Use HEALTH_CHECK_TTL (6 min) for internal routing,
             or QUICK_CHECK_TTL (30 sec) for frontend polling.
        cloud_provider: For cloud tier, the specific provider to test (e.g., "cloud-gpt").
                       If None, uses DEFAULT_MODELS["cloud"].
        local_model: For local tier, the specific model to test (e.g., "local-llama-quality").
                    If None, uses DEFAULT_MODELS["local"].
        lakeshore_model: For lakeshore tier, the specific model to test
                        (e.g., "lakeshore-qwen-32b"). Each Lakeshore model runs on
                        a separate vLLM port, so we track availability per-model.
                        If None, uses the base "lakeshore" cache key.

    Per-model cache keys:
    ---------------------
    Each tier supports per-model health tracking:
      - Cloud:     "cloud:{provider}"      e.g., "cloud:cloud-gpt"
      - Local:     "local:{model}"         e.g., "local:local-llama-quality"
      - Lakeshore: "lakeshore:{model}"     e.g., "lakeshore:lakeshore-qwen-32b"

    This ensures that one model being down doesn't incorrectly mark the
    entire tier as unavailable. For example, if the 32B model isn't running
    on Lakeshore but the 1.5B is, selecting 32B should show red while
    selecting 1.5B should show green.
    """
    # Use model-specific cache keys so each model is tracked independently.
    # This allows Claude to be "unhealthy" while GPT is "healthy",
    # local-llama to be installed while local-llama-quality is not,
    # and lakeshore-qwen-1.5b to be running while lakeshore-qwen-32b is not.
    cache_key = tier
    if tier == "cloud" and cloud_provider:
        cache_key = f"cloud:{cloud_provider}"
    elif tier == "local" and local_model:
        cache_key = f"local:{local_model}"
    elif tier == "lakeshore" and lakeshore_model:
        cache_key = f"lakeshore:{lakeshore_model}"

    status = _tier_health.get(cache_key)

    # If cache is valid (within TTL), use it
    if (
        status is not None
        and status["last_check"] is not None
        and datetime.now() - status["last_check"] <= timedelta(seconds=ttl)
    ):
        return status["available"]

    # Cache expired — run a fresh health check
    logger.debug(f"Cache expired for {cache_key} (TTL={ttl}s), running health check")

    # Pass model overrides so the check tests the right model
    is_available, error = check_tier_health(
        tier,
        cloud_provider=cloud_provider,
        local_model=local_model,
        lakeshore_model=lakeshore_model,
    )
    error_type = _determine_error_type(error)

    # Preserve auth errors even if a subsequent check times out.
    # Auth errors require user action (fix API key) — we don't want
    # a transient timeout to mask the real issue.
    previous_status = _tier_health.get(cache_key)
    if (
        previous_status
        and previous_status.get("error_type") == "auth"
        and error_type in ("timeout", "connection")
    ):
        logger.debug(f"Preserving auth error for {cache_key} (new check got {error_type})")
        error = previous_status.get("error")
        error_type = "auth"

    _tier_health[cache_key] = {
        "available": is_available,
        "last_check": datetime.now(),
        "error": error,
        "error_type": error_type,
    }

    return is_available


def get_available_tiers() -> list[str]:
    """Get list of currently available tiers (uses user's active cloud provider)."""
    return [
        tier
        for tier in ["local", "lakeshore", "cloud"]
        if is_tier_available(
            tier, cloud_provider=_active_cloud_provider if tier == "cloud" else None
        )
    ]


def get_tier_error(
    tier: str,
    cloud_provider: str | None = None,
    lakeshore_model: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Get the error details for a tier.

    Args:
        tier: The tier to get error for ("local", "lakeshore", or "cloud")
        cloud_provider: For cloud tier, the specific provider (e.g., "cloud-gpt").
                       This must match what was passed to is_tier_available(),
                       otherwise we'll look up the wrong cache entry.
        lakeshore_model: For lakeshore tier, the specific model (e.g., "lakeshore-qwen-32b").
                        Same logic as cloud_provider — must match is_tier_available()
                        so we look up the correct per-model cache entry.

    Why model-specific keys matter:
    -------------------------------
    The health cache stores per-model errors separately:
    - "cloud" key has errors from default provider (Claude)
    - "cloud:cloud-gpt" key has errors from GPT
    - "lakeshore" key has errors from base tier check
    - "lakeshore:lakeshore-qwen-32b" key has errors from the 32B model check

    If we test the 32B model but look up the "lakeshore" key, we'd get the
    base tier status (which may be "healthy") even though 32B is down.

    Returns:
        Tuple of (error_message, error_type) where error_type is one of:
        - "auth": API key invalid or subscription expired
        - "connection": Cannot connect to service
        - "timeout": Service timed out
        - "unknown": Other error
        - None: No error (tier is available)
    """
    # Use the same cache key logic as is_tier_available
    cache_key = tier
    if tier == "cloud" and cloud_provider:
        cache_key = f"cloud:{cloud_provider}"
    elif tier == "lakeshore" and lakeshore_model:
        cache_key = f"lakeshore:{lakeshore_model}"

    status = _tier_health.get(cache_key)
    if status is None:
        return None, None
    return status.get("error"), status.get("error_type")


def check_all_tiers():
    """Check health of all tiers (run on startup)"""
    for tier in ["local", "lakeshore", "cloud"]:
        update_tier_health(tier)

    available = get_available_tiers()
    if not available:
        logger.warning("No tiers available! Check Ollama, Globus, and API keys.")
    else:
        logger.info(f"{len(available)}/3 tiers ready")
