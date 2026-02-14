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
async def get_tier_health(cloud_provider: str | None = None):
    """
    Get current health status of all AI tiers.

    Returns availability, last check time, and any error messages.
    Frontend polls this every 30 seconds to show tier status indicators.

    Args:
        cloud_provider: Optional. If provided, checks health for this specific
                       cloud provider (e.g., "cloud-gpt") instead of the default.
                       This allows the UI to show correct status when user
                       switches providers (e.g., Claude has billing issues but GPT works).

    NOTE: This calls is_tier_available() which does a FRESH check if the
    cached status is stale. This ensures the frontend gets up-to-date info.
    """
    print(f"🔍 HEALTH ENDPOINT: called with cloud_provider={cloud_provider}")
    tiers = {}
    for tier_name in ["local", "lakeshore", "cloud"]:
        try:
            # For cloud tier, use the user's selected provider if specified
            tier_cloud_provider = cloud_provider if tier_name == "cloud" else None

            # Use shorter TTL (30 sec) for frontend polling to show near real-time status
            is_available = is_tier_available(
                tier_name,
                ttl=QUICK_CHECK_TTL,
                cloud_provider=tier_cloud_provider,
            )

            # Build the cache key to get the correct status
            cache_key = tier_name
            if tier_name == "cloud" and cloud_provider:
                cache_key = f"cloud:{cloud_provider}"

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
            print(f"✅ HEALTH: {tier_name} -> available={is_available}")
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
