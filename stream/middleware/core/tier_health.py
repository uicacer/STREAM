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
    LAKESHORE_VLLM_ENDPOINT,
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

        # LAKESHORE: Check vLLM
        elif tier == "lakeshore":
            if not LAKESHORE_VLLM_ENDPOINT:
                return False, "No Lakeshore endpoint configured in .env"

            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{LAKESHORE_VLLM_ENDPOINT}/v1/models",
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    return True, None
                else:
                    return False, f"vLLM not responding (HTTP {response.status_code})"

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


def update_tier_health(tier: str):
    """Update health status for a tier"""
    is_available, error = check_tier_health(tier)

    _tier_health[tier] = {"available": is_available, "last_check": datetime.now(), "error": error}

    status = "✅" if is_available else "❌"
    model = DEFAULT_MODELS.get(tier, "unknown")

    if is_available:
        print(f"{status} {tier.upper()} ({model}) is available")
    else:
        print(f"{status} {tier.upper()} ({model}) is UNAVAILABLE: {error}")


def is_tier_available(tier: str) -> bool:
    """Check if tier is available (with caching)"""
    status = _tier_health.get(tier)

    if (
        status is None
        or status["last_check"] is None
        or datetime.now() - status["last_check"] > timedelta(seconds=HEALTH_CHECK_TTL)
    ):
        update_tier_health(tier)
        status = _tier_health.get(tier)

    return status["available"]


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
