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
_tier_health = {
    "local": {"available": False, "last_check": None, "error": None},
    "lakeshore": {"available": False, "last_check": None, "error": None},
    "cloud": {"available": False, "last_check": None, "error": None},
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

        # LAKESHORE: Check proxy service WITH RETRY
        # The proxy handles routing via Globus Compute or SSH port forwarding
        elif tier == "lakeshore":
            # Retry 3 times with delays (proxy might be starting up)
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=10.0) as client:
                        # Check if the Lakeshore proxy is healthy
                        response = client.get(f"{LAKESHORE_PROXY_URL}/health")

                        if response.status_code == 200:
                            # Proxy is healthy
                            # The proxy's health endpoint tells us if it's configured properly
                            health_data = response.json()
                            if health_data.get("status") == "healthy":
                                return True, None
                            else:
                                return False, "Lakeshore proxy unhealthy"
                        else:
                            return (
                                False,
                                f"Lakeshore proxy not responding (HTTP {response.status_code})",
                            )

                except httpx.ConnectError:
                    if attempt < 2:  # Don't sleep on last attempt
                        time.sleep(2)  # Wait 2 seconds before retrying
                        continue
                    return False, "Cannot connect to Lakeshore proxy. Is the proxy service running?"
                except Exception as e:
                    return False, f"Lakeshore proxy error: {str(e)}"

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
                            return False, f"HTTP {response.status_code}"

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


def _quick_health_check(tier: str) -> tuple[bool, str | None]:
    """
    Quick single-attempt health check (no retries) for on-demand fallback.

    This function is called when the cache expires and the background monitor
    hasn't updated yet. It's designed to be FAST to avoid blocking user requests.

    Why no retries here:
    - Retries are slow (2s delay between attempts = 4-6s total wait time)
    - If a service is truly down, retrying won't help
    - The background monitor does proper retries every 5 minutes
    - The routing system will fallback to another tier if this check fails
    - We prioritize user experience (fast response) over accuracy
    """
    model = DEFAULT_MODELS.get(tier)
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

        # LAKESHORE: Single attempt to proxy
        elif tier == "lakeshore":
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{LAKESHORE_PROXY_URL}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        return True, None
                    else:
                        return False, "Lakeshore proxy unhealthy"
                else:
                    return False, f"Lakeshore proxy not responding (HTTP {response.status_code})"

        # CLOUD: Single attempt through LiteLLM
        elif tier == "cloud":
            with httpx.Client(timeout=5.0) as client:
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
                    return False, f"HTTP {response.status_code}"

        return False, "Unknown tier"

    except httpx.TimeoutException:
        return False, "Timeout"
    except httpx.ConnectError:
        return False, "Connection refused"
    except Exception as e:
        return False, str(e)


def update_tier_health(tier: str):
    """Update health status for a tier (used by background monitor with retries)"""
    is_available, error = check_tier_health(tier)

    _tier_health[tier] = {"available": is_available, "last_check": datetime.now(), "error": error}

    status = "✅" if is_available else "❌"
    model = DEFAULT_MODELS.get(tier, "unknown")

    # Display tier status
    if is_available:
        print(f"{status} {tier.upper()} ({model}) is available")
    else:
        print(f"{status} {tier.upper()} ({model}) is UNAVAILABLE: {error}")


def is_tier_available(tier: str) -> bool:
    """
    Check if tier is available (with caching).

    Uses cached status if available and fresh.
    If cache expired, uses quick single-attempt check (no retries) to avoid blocking.
    The background monitor handles proper retries and keeps cache fresh.
    """
    status = _tier_health.get(tier)

    # If cache is valid, use it
    if (
        status is not None
        and status["last_check"] is not None
        and datetime.now() - status["last_check"] <= timedelta(seconds=HEALTH_CHECK_TTL)
    ):
        return status["available"]

    # Cache expired - do quick single-attempt check (no retries)
    # This is a fallback in case background monitor hasn't run yet
    logger.debug(f"Cache expired for {tier}, doing quick health check")
    is_available, error = _quick_health_check(tier)

    _tier_health[tier] = {
        "available": is_available,
        "last_check": datetime.now(),
        "error": error,
    }

    return is_available


def get_available_tiers() -> list[str]:
    """Get list of currently available tiers"""
    return [tier for tier in ["local", "lakeshore", "cloud"] if is_tier_available(tier)]


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
