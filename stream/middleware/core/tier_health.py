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
    LAKESHORE_PROXY_URL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODELS,
    STREAM_MODE,
)

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


def check_tier_health(
    tier: str,
    cloud_provider: str | None = None,
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

    Returns:
        Tuple of (is_available, error_message). error_message is None if available.
    """
    # For cloud tier, use the specified provider or fall back to default
    model = cloud_provider if (tier == "cloud" and cloud_provider) else DEFAULT_MODELS.get(tier)
    if not model:
        return False, f"No model configured for tier {tier}"

    try:
        # LOCAL: Check Ollama directly
        if tier == "local":
            with httpx.Client(timeout=5.0) as client:
                # OLLAMA_BASE_URL resolves to:
                #   Docker mode:  http://ollama:11434  (container DNS)
                #   Desktop mode: http://localhost:11434  (native)
                response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if response.status_code != 200:
                    return False, f"Ollama not responding (HTTP {response.status_code})"

                # Verify the specific model exists
                data = response.json()
                installed_models = [m["name"] for m in data.get("models", [])]

                # Get the Ollama model name for this tier
                ollama_model = OLLAMA_MODELS.get(model)
                if not ollama_model:
                    return False, f"No Ollama model mapping for {model}"

                if ollama_model not in installed_models:
                    return False, f"Model {ollama_model} not installed in Ollama"

                return True, None

        # LAKESHORE: Check proxy health and (in server mode) test inference
        elif tier == "lakeshore":
            if STREAM_MODE == "desktop":
                # Desktop mode: proxy is embedded in the same server process.
                # Check the Globus client directly instead of HTTP (which would
                # be the server calling itself on port 5000 — blocks on startup
                # when the server isn't accepting connections yet).
                gc = _proxy_app.globus_client
                if not gc or not gc.is_available():
                    return False, "Globus Compute not configured or unavailable"
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
                    # Check for Globus auth errors
                    if health_data.get("globus_authenticated") is False:
                        return False, "Globus Compute authentication required"
            except httpx.ConnectError:
                return False, "Cannot connect to Lakeshore proxy. Is the proxy service running?"
            except Exception as e:
                return False, f"Lakeshore proxy error: {str(e)}"

            # Server mode: test ACTUAL inference through the LiteLLM HTTP gateway.
            # LiteLLM handles model name translation and routing.
            for attempt in range(2):
                try:
                    with httpx.Client(timeout=20.0) as client:  # HPC can be slow
                        response = client.post(
                            f"{LITELLM_BASE_URL}/v1/chat/completions",
                            json={
                                "model": model,  # LiteLLM translates this
                                "messages": [{"role": "user", "content": "ping"}],
                                "max_tokens": 1,  # Minimal response
                                "stream": False,
                            },
                            headers={
                                "Authorization": f"Bearer {LITELLM_API_KEY}",
                                "Content-Type": "application/json",
                            },
                        )

                        if response.status_code == 200:
                            return True, None
                        else:
                            # Parse error to get useful info
                            try:
                                error_data = response.json()
                                error_msg = str(
                                    error_data.get(
                                        "detail",
                                        error_data.get("error", f"HTTP {response.status_code}"),
                                    )
                                )
                                # Check for common HPC issues
                                if "ManagerLost" in error_msg:
                                    return False, "HPC workers crashed (ManagerLost)"
                                if "authentication" in error_msg.lower():
                                    return (
                                        False,
                                        f"Globus Compute authentication required: {error_msg[:80]}",
                                    )
                                return False, f"Inference failed: {error_msg[:100]}"
                            except Exception:
                                return False, f"HTTP {response.status_code}"

                except httpx.TimeoutException:
                    if attempt < 1:
                        time.sleep(2)
                        continue
                    return False, "Inference timeout (HPC workers may be unavailable)"
                except Exception as e:
                    return False, f"Inference error: {str(e)}"

            return False, "Inference failed after retries"

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
    Update health status for a tier (used by background monitor and startup).

    For cloud tier: uses the user's active cloud provider after startup.
    At startup (before any frontend request), _active_cloud_provider is None
    so it falls back to the default. Once the user's selection arrives via
    a health poll or chat request, all subsequent checks use their provider.
    """
    # For cloud tier, use the user's active provider (not always the default).
    # This ensures the background monitor tests GPT if the user selected GPT.
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

    status = "✅" if is_available else "❌"
    model = cloud_provider or DEFAULT_MODELS.get(tier, "unknown")

    if is_available:
        print(f"{status} {tier.upper()} ({model}) is available")
    else:
        print(f"{status} {tier.upper()} ({model}) is UNAVAILABLE: {error}")


def is_tier_available(
    tier: str,
    ttl: int = HEALTH_CHECK_TTL,
    cloud_provider: str | None = None,
) -> bool:
    """
    Check if tier is available (with caching).

    Uses cached status if available and fresh (within TTL).
    If cache expired, runs a fresh check_tier_health() call.
    The background monitor also keeps the cache fresh every 5 minutes.

    Args:
        tier: The tier to check ("local", "lakeshore", or "cloud")
        ttl: Cache TTL in seconds. Use HEALTH_CHECK_TTL (6 min) for internal routing,
             or QUICK_CHECK_TTL (30 sec) for frontend polling.
        cloud_provider: For cloud tier, the specific provider to test (e.g., "cloud-gpt").
                       If None, uses DEFAULT_MODELS["cloud"].

    Why cloud_provider matters:
    ---------------------------
    The health check tests ONE cloud model to determine if Cloud tier is available.
    Without this parameter, it always tests the default provider (e.g., Claude).

    Problem scenario:
    1. Claude has billing issues → health check fails → Cloud marked unavailable
    2. User switches to GPT in settings (which works fine)
    3. User selects Cloud tier → still fails because cache says "unavailable"
    4. User is stuck, can't use Cloud even though GPT works

    Solution:
    By passing cloud_provider, we test the ACTUAL provider the user selected.
    Each provider gets its own cache entry, so Claude being down doesn't block GPT.
    """
    # For cloud tier with a specific provider, use provider-specific cache key
    # This allows Claude to be "unhealthy" while GPT is "healthy"
    cache_key = tier
    if tier == "cloud" and cloud_provider:
        cache_key = f"cloud:{cloud_provider}"

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

    # Pass cloud_provider so the check tests the right provider model
    is_available, error = check_tier_health(tier, cloud_provider=cloud_provider)
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
) -> tuple[str | None, str | None]:
    """
    Get the error details for a tier.

    Args:
        tier: The tier to get error for ("local", "lakeshore", or "cloud")
        cloud_provider: For cloud tier, the specific provider (e.g., "cloud-gpt").
                       This must match what was passed to is_tier_available(),
                       otherwise we'll look up the wrong cache entry.

    Why cloud_provider matters:
    ---------------------------
    The health cache stores cloud provider errors separately:
    - "cloud" key has errors from default provider (Claude)
    - "cloud:cloud-gpt" key has errors from GPT

    If we test GPT but look up "cloud" key, we get Claude's old error!
    This causes confusing messages like "AnthropicException" when GPT was tested.

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

    status = _tier_health.get(cache_key)
    if status is None:
        return None, None
    return status.get("error"), status.get("error_type")


def check_all_tiers():
    """Check health of all tiers (run on startup)"""
    print("\n🔍 Checking health of all AI tiers...")
    print("=" * 60)

    for tier in ["local", "lakeshore", "cloud"]:
        update_tier_health(tier)

    print("=" * 60)

    available = get_available_tiers()
    if not available:
        print("❌ WARNING: NO AI TIERS ARE AVAILABLE!")
        print("   Check that Docker services are running:")
        print("   - Ollama (local)")
        print("   - LiteLLM gateway")
        print("   - Cloud API keys configured")
    else:
        print(f"✅ {len(available)}/3 tiers available: {', '.join(available).upper()}")
    print()
