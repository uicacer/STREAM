# =============================================================================
# STREAM Middleware - Health Check Routes
# =============================================================================

import logging
from datetime import UTC, datetime

from fastapi import APIRouter

from stream.middleware.config import (
    CLOUD_PROVIDERS,
    DEFAULT_CLOUD_PROVIDER,
    QUICK_CHECK_TTL,
    SERVICE_VERSION,
)
from stream.middleware.core.globus_auth import is_authenticated as globus_is_authenticated
from stream.middleware.core.tier_health import (
    _tier_health,
    check_all_tiers,
    get_available_tiers,
    is_tier_available,
    set_active_cloud_provider,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check - is the service running?
    """
    return {
        "status": "healthy",
        "service": "STREAM Middleware",
        "version": SERVICE_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/health/tiers")
async def get_tier_health(
    cloud_provider: str | None = None,
    local_model: str | None = None,
    lakeshore_model: str | None = None,
):
    """
    Get current health status of all AI tiers.

    Returns availability, last check time, and any error messages.
    Frontend polls this every 30 seconds to show tier status indicators.

    Args:
        cloud_provider: Optional. Checks health for this specific cloud provider.
        local_model: Optional. Checks that this specific Ollama model is installed.
        lakeshore_model: Optional. When provided, does a real 1-token inference
                        test through Globus to verify this specific model's vLLM
                        instance is running. Without this, only base Globus auth
                        is checked (which would show green even if the model
                        isn't running). Results are cached per-model.

    NOTE: This calls is_tier_available() which does a FRESH check if the
    cached status is stale. This ensures the frontend gets up-to-date info.
    For Lakeshore with a specific model, the first check may take ~5-15s
    (Globus round-trip), but subsequent polls use the cached result.
    """
    # Remember the user's cloud provider selection so the background monitor
    # and get_available_tiers() test the RIGHT provider (not the default).
    if cloud_provider:
        set_active_cloud_provider(cloud_provider)

    tiers = {}
    for tier_name in ["local", "lakeshore", "cloud"]:
        try:
            # Per-tier model overrides for health checks.
            # Each tier can test a specific model independently:
            #   - Cloud: test user's selected provider (Claude vs GPT)
            #   - Local: test user's selected Ollama model
            #   - Lakeshore: test user's selected vLLM model (new!)
            tier_cloud_provider = cloud_provider if tier_name == "cloud" else None
            tier_local_model = local_model if tier_name == "local" else None
            tier_lakeshore_model = lakeshore_model if tier_name == "lakeshore" else None

            # Use shorter TTL (30 sec) for frontend polling to show near real-time status
            is_available = is_tier_available(
                tier_name,
                ttl=QUICK_CHECK_TTL,
                cloud_provider=tier_cloud_provider,
                local_model=tier_local_model,
                lakeshore_model=tier_lakeshore_model,
            )

            # Build the cache key to get the correct status.
            # Must match the key used by is_tier_available() so we read
            # back the result that was just cached.
            cache_key = tier_name
            if tier_name == "cloud" and cloud_provider:
                cache_key = f"cloud:{cloud_provider}"
            elif tier_name == "local" and local_model:
                cache_key = f"local:{local_model}"
            elif tier_name == "lakeshore" and lakeshore_model:
                cache_key = f"lakeshore:{lakeshore_model}"

            # Now get the updated status from cache (which was just refreshed if stale)
            status = _tier_health.get(cache_key, {})

            # Safely get last_check - handle both datetime and string values
            last_check = status.get("last_check")
            if last_check is not None:
                # If it's a datetime, convert to ISO string
                if hasattr(last_check, "isoformat"):
                    last_check = last_check.isoformat()
                # If it's already a string, use as-is
                elif not isinstance(last_check, str):
                    last_check = str(last_check)

            tiers[tier_name] = {
                "available": is_available,
                "error": status.get("error"),
                "error_type": status.get(
                    "error_type"
                ),  # "auth", "connection", "timeout", or "unknown"
                "last_check": last_check,
            }
        except Exception as e:
            # Log the error and provide a safe fallback
            logger.error(f"Error checking health for {tier_name}: {e}", exc_info=True)
            tiers[tier_name] = {
                "available": False,
                "error": f"Health check failed: {str(e)}",
                "error_type": "unknown",
                "last_check": None,
            }

    # Add Lakeshore-specific auth status
    try:
        tiers["lakeshore"]["authenticated"] = globus_is_authenticated()
        logger.debug(f"Globus authenticated: {tiers['lakeshore']['authenticated']}")
    except Exception as e:
        logger.warning(f"Failed to check Globus auth status: {e}")
        tiers["lakeshore"]["authenticated"] = None

    # Get available tiers (with defensive error handling)
    try:
        available = get_available_tiers()
    except Exception as e:
        logger.error(f"Error getting available tiers: {e}", exc_info=True)
        available = []

    return {
        "tiers": tiers,
        "available_tiers": available,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.post("/health/tiers/refresh")
async def refresh_tier_health():
    """
    Force refresh health check for all tiers.

    Use this to immediately check tier availability without waiting
    for the background monitor. Useful after restarting services.
    """
    check_all_tiers()

    return {
        "status": "refreshed",
        "available_tiers": get_available_tiers(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/health/cloud-providers")
async def get_cloud_providers():
    """
    Get available cloud providers for the settings dropdown.

    Users can switch cloud providers if:
    - Their current provider's subscription expired
    - They prefer a different model (Claude vs GPT)
    - One provider is having issues

    Returns:
        - providers: Dict of available cloud providers with metadata
        - current: Currently selected provider ID
    """
    return {
        "providers": CLOUD_PROVIDERS,
        "current": DEFAULT_CLOUD_PROVIDER,
        "timestamp": datetime.now(UTC).isoformat(),
    }
