"""
Warm ping for AI tiers (SERVER MODE ONLY).

In server mode, sends a small test request to Local (Ollama) and Cloud
on startup to pre-load models into memory and reduce first-request latency.

In DESKTOP mode, warm pings are skipped entirely because the judge warmup
in lifecycle.py already pre-loads the local Ollama model into memory.
Lakeshore is always skipped (too expensive — requires a Globus Compute job).

WHY WARM PING?
--------------
Without warm ping (server mode):
  User sends "hi" → Ollama loads model (5-10s) → Response (slow!)

With warm ping:
  Startup → Ollama pre-loads model → User sends "hi" → Response (fast!)
"""

import asyncio
import logging
import time
from datetime import datetime

import httpx

from stream.middleware.config import (
    DEFAULT_MODELS,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODELS,
)
from stream.middleware.core.tier_health import _determine_error_type, _tier_health

logger = logging.getLogger(__name__)

# Warm ping test prompt (minimal tokens)
WARM_PING_PROMPT = "Say hi"
WARM_PING_MAX_TOKENS = 5
WARM_PING_TIMEOUT = 30.0  # Longer timeout for model loading


async def warm_ping_local() -> tuple[bool, float, str | None]:
    """
    Send a warm ping to the local Ollama tier.

    This pre-loads the model into GPU/RAM memory.
    First request is slow (model loading), subsequent requests are fast.

    Returns:
        Tuple of (success, latency_ms, error_message)
    """
    model = DEFAULT_MODELS.get("local")
    ollama_model = OLLAMA_MODELS.get(model)

    if not ollama_model:
        return False, 0, f"No Ollama model mapping for {model}"

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=WARM_PING_TIMEOUT) as client:
            # OLD: f"http://ollama:{OLLAMA_PORT}/api/generate"
            # "ollama" was a Docker-only hostname. Now we use OLLAMA_BASE_URL
            # from config, which points to localhost outside Docker.
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": WARM_PING_PROMPT,
                    "stream": False,
                    "options": {"num_predict": WARM_PING_MAX_TOKENS},
                },
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                return True, latency_ms, None
            else:
                return False, latency_ms, f"HTTP {response.status_code}"

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, "Timeout"
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, str(e)


async def warm_ping_lakeshore() -> tuple[bool, float, str | None]:
    """
    Send a warm ping to the Lakeshore tier through LiteLLM.

    This tests ACTUAL inference capability, not just proxy health.
    Going through LiteLLM ensures model name translation works correctly.

    Returns:
        Tuple of (success, latency_ms, error_message)
    """
    model = DEFAULT_MODELS.get("lakeshore")
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=WARM_PING_TIMEOUT) as client:
            # Route through LiteLLM for model name translation
            response = await client.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,  # LiteLLM translates this
                    "messages": [{"role": "user", "content": WARM_PING_PROMPT}],
                    "max_tokens": WARM_PING_MAX_TOKENS,
                    "stream": False,
                },
                headers={
                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                return True, latency_ms, None
            else:
                # Parse error message if available
                try:
                    error_data = response.json()
                    error_msg = str(
                        error_data.get(
                            "detail", error_data.get("error", f"HTTP {response.status_code}")
                        )
                    )
                except Exception:
                    error_msg = f"HTTP {response.status_code}"
                return False, latency_ms, error_msg

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, "Timeout"
    except httpx.ConnectError:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, "Connection refused (LiteLLM not running?)"
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, str(e)


async def warm_ping_cloud() -> tuple[bool, float, str | None]:
    """
    Send a warm ping to the cloud tier.

    This tests the LiteLLM → Cloud API connection.
    Usually fast since cloud APIs are always warm.

    Returns:
        Tuple of (success, latency_ms, error_message)
    """
    model = DEFAULT_MODELS.get("cloud")
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=WARM_PING_TIMEOUT) as client:
            response = await client.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": WARM_PING_PROMPT}],
                    "max_tokens": WARM_PING_MAX_TOKENS,
                },
                headers={
                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                return True, latency_ms, None
            else:
                return False, latency_ms, f"HTTP {response.status_code}"

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, "Timeout"
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, str(e)


async def warm_up_all_tiers():
    """
    Warm up tiers in parallel (SERVER MODE ONLY).

    Called from lifecycle.py only when STREAM_MODE != "desktop".
    In desktop mode, the judge warmup already pre-loads the local Ollama model.

    Sends a small "Say hi" request to Local and Cloud to:
    - Pre-load models (Ollama loads model into GPU on first inference)
    - Test real availability (proxy "healthy" ≠ inference working)
    - Establish connections early

    Lakeshore is always skipped — its warm ping would submit a real Globus
    Compute job to the HPC, which is expensive and wasteful at scale.
    """
    # Warm up LOCAL (Ollama) and CLOUD. Lakeshore is always skipped because
    # its warm ping submits a real inference job through Globus Compute to the HPC,
    # which is expensive (~10-30s) and wasteful if the user never uses Lakeshore.
    warm_pings = [warm_ping_local(), warm_ping_cloud()]
    tier_names = ["local", "cloud"]

    results = await asyncio.gather(*warm_pings, return_exceptions=True)

    for tier, result in zip(tier_names, results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"  {tier.upper():12s} {result}")
            _update_tier_health(tier, False, str(result))
        else:
            success, latency_ms, error = result

            if success:
                logger.info(f"  ✓ {tier.upper():12s} warm ({latency_ms:.0f}ms)")
                _update_tier_health(tier, True, None)
            else:
                logger.warning(f"  {tier.upper():12s} {error}")
                _update_tier_health(tier, False, error)


def _update_tier_health(tier: str, available: bool, error: str | None):
    """Update tier health status after warm ping."""
    _tier_health[tier] = {
        "available": available,
        "last_check": datetime.now(),
        "error": error,
        "error_type": _determine_error_type(error),
    }
