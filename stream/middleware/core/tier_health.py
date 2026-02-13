"""
Tier health checking and availability tracking.

This module manages the health status of all AI service tiers:
- LOCAL (Ollama)
- LAKESHORE (vLLM)
- CLOUD (LiteLLM → Anthropic/OpenAI)
"""

import logging
import time  # FIX: Add missing import (F821)
from datetime import datetime, timedelta

import httpx

from stream.middleware.config import (
    DEFAULT_MODELS,
    HEALTH_CHECK_TTL,
    LAKESHORE_PROXY_URL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    OLLAMA_MODELS,
    OLLAMA_PORT,
)

logger = logging.getLogger(__name__)

# Track health status (module-level state)
# error_type can be: None, "auth", "connection", "timeout", "unknown"
_tier_health = {
    "local": {"available": False, "last_check": None, "error": None, "error_type": None},
    "lakeshore": {"available": False, "last_check": None, "error": None, "error_type": None},
    "cloud": {"available": False, "last_check": None, "error": None, "error_type": None},
}


def check_tier_health(tier: str) -> tuple[bool, str | None]:
    """Check if a specific tier is available"""
    model = DEFAULT_MODELS.get(tier)
    if not model:
        return False, f"No model configured for tier {tier}"

    try:
        # LOCAL: Check Ollama directly
        if tier == "local":
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"http://ollama:{OLLAMA_PORT}/api/tags")
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

        # LAKESHORE: Test ACTUAL inference capability through LiteLLM
        # Going through LiteLLM ensures model name translation works correctly
        # (LiteLLM maps "lakeshore-qwen" → actual vLLM model name)
        elif tier == "lakeshore":
            # First, quick check if proxy is even running
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

            # Proxy is running - now test ACTUAL inference through LiteLLM
            # LiteLLM handles model name translation and routing
            for attempt in range(2):
                try:
                    with httpx.Client(timeout=30.0) as client:  # Longer timeout for HPC
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

        # CLOUD: Test through LiteLLM WITH RETRY
        elif tier == "cloud":
            # Try 2 times with delays (LiteLLM might be starting)
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

                        if response.status_code == 200:
                            data = response.json()
                            actual_model = data.get("model", "").lower()
                            if "claude" in actual_model or "gpt" in actual_model:
                                return True, None
                            else:
                                return False, f"Unexpected model: {actual_model}"
                        else:
                            # Try to parse error message from response
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

                            error_lower = error_msg.lower()

                            # Debug: Print what we got from Cloud health check
                            print(
                                f"🔍 CLOUD HEALTH CHECK (retry): status={response.status_code}, error={error_msg[:200]}"
                            )

                            # Detect auth/billing errors - these can come as 400, 401, 402, or 403
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

                            keywords_found = [kw for kw in auth_keywords if kw in error_lower]
                            print(f"🔍 CLOUD AUTH CHECK (retry): keywords_found={keywords_found}")

                            if keywords_found:
                                print(f"❌ CLOUD AUTH ERROR DETECTED (retry): {error_msg[:100]}")
                                return False, f"[AUTH] {error_msg}"

                            return (
                                False,
                                f"HTTP {response.status_code}: {error_msg[:100]}"
                                if error_msg
                                else f"HTTP {response.status_code}",
                            )

                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    if attempt < 2:  # Not last attempt
                        time.sleep(2)  # Wait 2 seconds before retry
                        continue
                    else:
                        return False, f"Connection failed after 2 attempts: {str(e)}"

        return False, "Unknown tier"

    except httpx.TimeoutException:
        return False, "Timeout"
    except httpx.ConnectError:
        return False, "Connection refused"
    except Exception as e:
        return False, str(e)


def _quick_health_check(
    tier: str,
    cloud_provider: str | None = None,
) -> tuple[bool, str | None]:
    """
    Quick single-attempt health check (no retries) for on-demand fallback.

    This function is called when the cache expires and the background monitor
    hasn't updated yet. It's designed to be FAST to avoid blocking user requests.

    Args:
        tier: The tier to check ("local", "lakeshore", or "cloud")
        cloud_provider: For cloud tier, the specific provider to test (e.g., "cloud-gpt").
                       If None, uses DEFAULT_MODELS["cloud"].

    Why no retries here:
    - Retries are slow (2s delay between attempts = 4-6s total wait time)
    - If a service is truly down, retrying won't help
    - The background monitor does proper retries every 5 minutes
    - The routing system will fallback to another tier if this check fails
    - We prioritize user experience (fast response) over accuracy
    """
    # For cloud tier, use the user-specified provider if provided
    # This allows testing GPT even if Claude (the default) has auth issues
    if tier == "cloud" and cloud_provider:
        model = cloud_provider
        print(f"🔍 QUICK HEALTH: Using user-selected cloud provider: {model}")
    else:
        model = DEFAULT_MODELS.get(tier)
        print(f"🔍 QUICK HEALTH: Using default model for {tier}: {model}")
    if not model:
        return False, f"No model configured for tier {tier}"

    try:
        # LOCAL: Check Ollama directly
        if tier == "local":
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://ollama:{OLLAMA_PORT}/api/tags")
                if response.status_code != 200:
                    return False, f"Ollama not responding (HTTP {response.status_code})"

                data = response.json()
                installed_models = [m["name"] for m in data.get("models", [])]
                ollama_model = OLLAMA_MODELS.get(model)

                if not ollama_model:
                    return False, f"No Ollama model mapping for {model}"
                if ollama_model not in installed_models:
                    return False, f"Model {ollama_model} not installed in Ollama"

                return True, None

        # LAKESHORE: Quick check - proxy health + actual inference test
        # We need to test actual inference because proxy can be healthy while HPC workers are down
        elif tier == "lakeshore":
            # First check if proxy is running
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
                return False, "Cannot connect to Lakeshore proxy"

            # Proxy is healthy - now test actual inference (single attempt, shorter timeout)
            try:
                with httpx.Client(timeout=15.0) as client:
                    response = client.post(
                        f"{LITELLM_BASE_URL}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
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
                        try:
                            error_data = response.json()
                            error_msg = str(
                                error_data.get(
                                    "detail",
                                    error_data.get("error", f"HTTP {response.status_code}"),
                                )
                            )
                            if "ManagerLost" in error_msg:
                                return False, "HPC workers unavailable"
                            return False, f"Inference failed: {error_msg[:80]}"
                        except Exception:
                            return False, f"HTTP {response.status_code}"
            except httpx.TimeoutException:
                return False, "HPC inference timeout"

        # CLOUD: Single attempt through LiteLLM
        # Use longer timeout (15s) to ensure we get actual error response (not just timeout)
        # Auth errors can take a few seconds to come back from Anthropic
        elif tier == "cloud":
            with httpx.Client(timeout=15.0) as client:
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
                if response.status_code == 200:
                    data = response.json()
                    actual_model = data.get("model", "").lower()
                    if "claude" in actual_model or "gpt" in actual_model:
                        return True, None
                    else:
                        return False, f"Unexpected model: {actual_model}"
                else:
                    # Try to parse error message from response
                    error_msg = ""
                    try:
                        error_data = response.json()
                        # LiteLLM wraps errors in "detail" or nested "error.message"
                        error_msg = str(
                            error_data.get("detail", "")
                            or error_data.get("error", {}).get("message", "")
                            or error_data
                        )
                    except Exception:
                        error_msg = response.text[:200] if response.text else ""

                    error_lower = error_msg.lower()

                    # Debug: Print what we got from Cloud health check
                    print(
                        f"🔍 CLOUD HEALTH CHECK: status={response.status_code}, error={error_msg[:200]}"
                    )

                    # Detect auth/billing errors - these can come as 400, 401, 402, or 403
                    # Anthropic returns 400 with "credit balance is too low"
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

                    keywords_found = [kw for kw in auth_keywords if kw in error_lower]
                    print(f"🔍 CLOUD AUTH CHECK: keywords_found={keywords_found}")

                    if keywords_found:
                        print(f"❌ CLOUD AUTH ERROR DETECTED: {error_msg[:100]}")
                        return False, f"[AUTH] {error_msg}"

                    return (
                        False,
                        f"HTTP {response.status_code}: {error_msg[:100]}"
                        if error_msg
                        else f"HTTP {response.status_code}",
                    )

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
    """Update health status for a tier (used by background monitor with retries)"""
    is_available, error = check_tier_health(tier)
    error_type = _determine_error_type(error)

    _tier_health[tier] = {
        "available": is_available,
        "last_check": datetime.now(),
        "error": error,
        "error_type": error_type,
    }

    status = "✅" if is_available else "❌"
    model = DEFAULT_MODELS.get(tier, "unknown")

    # Display tier status
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
    If cache expired, uses quick single-attempt check (no retries) to avoid blocking.
    The background monitor handles proper retries and keeps cache fresh.

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

    # Debug: Show what cache key we're checking
    print(f"🔍 HEALTH CHECK: tier={tier}, cloud_provider={cloud_provider}, cache_key={cache_key}")

    status = _tier_health.get(cache_key)

    # If cache is valid (within TTL), use it
    if (
        status is not None
        and status["last_check"] is not None
        and datetime.now() - status["last_check"] <= timedelta(seconds=ttl)
    ):
        return status["available"]

    # Cache expired - do quick single-attempt check (no retries)
    # This is a fallback in case background monitor hasn't run yet
    logger.debug(f"Cache expired for {cache_key} (TTL={ttl}s), doing quick health check")

    # Pass cloud_provider to health check so it tests the right model
    is_available, error = _quick_health_check(tier, cloud_provider=cloud_provider)
    error_type = _determine_error_type(error)

    # IMPORTANT: Preserve auth errors even if quick check times out
    # Auth errors require user action (fix API key), so we don't want
    # a timeout to mask the real issue
    previous_status = _tier_health.get(cache_key)
    if (
        previous_status
        and previous_status.get("error_type") == "auth"
        and error_type in ("timeout", "connection")
    ):
        # Keep the previous auth error - don't overwrite with timeout
        logger.debug(f"Preserving auth error for {cache_key} (quick check got {error_type})")
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
    """Get list of currently available tiers"""
    return [tier for tier in ["local", "lakeshore", "cloud"] if is_tier_available(tier)]


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
