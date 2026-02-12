"""
STREAM Middleware - Configuration API
=====================================

This endpoint exposes backend configuration to the frontend.

PRINCIPLE: Single Source of Truth
- All model/routing logic defaults are defined HERE in the backend
- Frontend fetches config on startup, doesn't define its own defaults
- No risk of frontend/backend config drift
"""

import logging

from fastapi import APIRouter

from stream.middleware.config import (
    DEFAULT_JUDGE_STRATEGY,
    DEFAULT_MODELS,
    JUDGE_STRATEGIES,
    SERVICE_VERSION,
    TIERS,
)
from stream.middleware.core.tier_health import get_available_tiers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/config")
async def get_config():
    """
    Get application configuration for the frontend.

    This endpoint provides:
    - Available tiers and their status
    - Default settings (judge strategy, temperature, etc.)
    - Model information

    The frontend should call this on startup and use these values
    as defaults instead of hardcoding them.
    """
    return {
        # Version info
        "version": SERVICE_VERSION,
        # Tier configuration
        "tiers": {
            "available": get_available_tiers(),
            "all": list(TIERS.keys()),
            "default": "auto",
            "info": TIERS,
        },
        # Judge configuration
        "judge": {
            "strategies": list(JUDGE_STRATEGIES.keys()),
            "default": DEFAULT_JUDGE_STRATEGY,
            "info": {
                name: {
                    "description": config.get("description", ""),
                    "timeout": config.get("timeout", 5.0),
                }
                for name, config in JUDGE_STRATEGIES.items()
            },
        },
        # Model configuration
        "models": {
            "default_by_tier": DEFAULT_MODELS,
        },
        # Default settings for the chat interface
        "defaults": {
            "tier": "auto",
            "judgeStrategy": DEFAULT_JUDGE_STRATEGY,
            "temperature": 0.7,
        },
    }


@router.get("/config/tiers")
async def get_tiers_config():
    """
    Get tier-specific configuration and health status.

    Useful for displaying tier options in the UI with real-time availability.
    """
    available = get_available_tiers()

    return {
        "tiers": [
            {
                "id": tier_id,
                "name": tier_info["name"],
                "description": tier_info["description"],
                "available": tier_id in available,
                "model": DEFAULT_MODELS.get(tier_id),
            }
            for tier_id, tier_info in TIERS.items()
        ],
        "default": "auto",
    }
