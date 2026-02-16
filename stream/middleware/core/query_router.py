"""
Intelligent query routing with automatic tier fallback.

This module determines which AI tier to use for a query based on:
1. User preference (if specified)
2. Query complexity (from complexity_judge)
3. Tier availability (from tier_health)
"""

import logging

from stream.middleware.config import DEFAULT_MODELS, LLM_JUDGE_ENABLED
from stream.middleware.core.complexity_judge import (
    judge_complexity_with_keywords,
    judge_complexity_with_llm,
)
from stream.middleware.core.tier_health import (
    get_tier_error,
    is_tier_available,
)

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised when a tier has an authentication/billing error."""

    def __init__(self, tier: str, message: str):
        self.tier = tier
        self.message = message
        super().__init__(f"{tier} auth error: {message}")


def get_tier_with_fallback(
    preferred_tier: str,
    complexity: str,
    cloud_provider: str | None = None,
) -> tuple[str, str, list[str]]:
    """
    Get tier with intelligent fallback.

    Args:
        preferred_tier: The tier to try first
        complexity: Query complexity (low/medium/high)
        cloud_provider: For cloud tier, the specific provider (e.g., "cloud-gpt")
                       Passed to health check so we test the right provider.

    Returns:
        Tuple of (tier, message, unavailable_tiers)

    Raises:
        AuthError: If a tier has an auth/billing error (user must fix it, no fallback)
    """
    # Define fallback chain based on complexity
    if complexity == "low":
        fallback_chain = ["local", "lakeshore", "cloud"]
    elif complexity == "medium":
        fallback_chain = ["lakeshore", "cloud", "local"]
    else:  # high
        fallback_chain = ["cloud", "lakeshore", "local"]

    # Ensure preferred tier is first
    if preferred_tier in fallback_chain:
        fallback_chain.remove(preferred_tier)
        fallback_chain.insert(0, preferred_tier)

    # Try each tier in order, tracking which were unavailable
    unavailable_tiers = []

    for tier in fallback_chain:
        # For cloud tier, pass cloud_provider so we check the RIGHT provider's health
        # (e.g., GPT might be healthy while Claude has billing issues)
        tier_cloud_provider = cloud_provider if tier == "cloud" else None

        if is_tier_available(tier, cloud_provider=tier_cloud_provider):
            if tier == preferred_tier:
                return tier, f"{complexity.upper()} → {tier.upper()}", []
            else:
                # Build message showing all unavailable tiers
                unavail_str = " and ".join(t.title() for t in unavailable_tiers)
                return (
                    tier,
                    f"{complexity.upper()} → {unavail_str} unavailable, using {tier.title()}",
                    unavailable_tiers,
                )
        else:
            # Check if this is an auth error - DON'T fall back, show error to user
            # IMPORTANT: Pass cloud_provider to get the CORRECT error from cache.
            # Without it, we'd look up "cloud" key and get Claude's error even when GPT was tested!
            error_msg, error_type = get_tier_error(tier, cloud_provider=tier_cloud_provider)
            if error_type == "auth":
                raise AuthError(tier, error_msg)
            unavailable_tiers.append(tier)

    return None, "All AI services unavailable", unavailable_tiers


class RoutingResult:
    """Result of tier routing decision, including fallback information."""

    def __init__(
        self,
        tier: str,
        complexity: str,
        preferred_tier: str | None = None,
        fallback_used: bool = False,
        unavailable_tiers: list[str] | None = None,
        auth_error_info: dict | None = None,
    ):
        self.tier = tier
        self.complexity = complexity
        self.preferred_tier = preferred_tier or tier
        self.fallback_used = fallback_used
        # original_tier is set when fallback occurred
        self.original_tier = preferred_tier if fallback_used else None
        # All tiers that were tried but unavailable
        self.unavailable_tiers = unavailable_tiers or []
        # Auth error info if a tier had billing/auth issue: {tier, message}
        self.auth_error_info = auth_error_info


def get_tier_for_query(
    query: str,
    user_preference: str = "auto",
    cloud_provider: str | None = None,
) -> RoutingResult:
    """
    Determine which tier to use based on LLM judge + keyword fallback + health checks.

    Returns a RoutingResult with:
    - tier: The actual tier to use
    - complexity: Query complexity (low/medium/high)
    - preferred_tier: The tier that would have been used if available
    - fallback_used: Whether a fallback occurred
    - original_tier: The tier that was unavailable (if fallback occurred)
    """
    # If user explicitly chose a tier, respect it strictly (no silent fallback)
    if user_preference in ["local", "lakeshore", "cloud"]:
        # For Cloud tier, pass the user's selected cloud_provider to health check
        # so we test the ACTUAL provider they want (e.g., GPT), not the default (Claude)
        if is_tier_available(user_preference, cloud_provider=cloud_provider):
            return RoutingResult(
                tier=user_preference,
                complexity="user_override",
                preferred_tier=user_preference,
                fallback_used=False,
            )
        else:
            # User explicitly selected this tier - don't silently fallback
            # Raise an error so the user knows their selection couldn't be honored
            # IMPORTANT: Pass cloud_provider to get error for the ACTUAL provider tested
            error_msg, error_type = get_tier_error(user_preference, cloud_provider=cloud_provider)

            # Provide specific error messages based on error type
            if error_type == "auth":
                raise Exception(
                    "Cloud API authentication failed. Your API key may be invalid or your "
                    "subscription may have expired. Please check your API key configuration."
                )
            elif error_msg:
                raise Exception(
                    f"{user_preference.upper()} tier is currently unavailable: {error_msg}"
                )
            else:
                raise Exception(
                    f"{user_preference.upper()} tier is currently unavailable. "
                    f"Please try again or select a different tier."
                )

    # Try LLM judge first (if enabled)
    complexity = None
    method = "unknown"

    if LLM_JUDGE_ENABLED:
        complexity, error, _cost, _tokens = judge_complexity_with_llm(query)
        if complexity:
            method = "LLM judge"
        else:
            logger.warning(f"LLM judge failed ({error}), falling back to keywords")

    # Fallback to keyword-based if LLM failed or disabled
    if complexity is None:
        complexity, matched_keyword = judge_complexity_with_keywords(query)
        method = (
            f"keyword matching ('{matched_keyword}')" if matched_keyword else "default (medium)"
        )

    logger.debug(method)

    # Map complexity to preferred tier
    if complexity == "low":
        preferred_tier = "local"
    elif complexity == "medium":
        preferred_tier = "lakeshore"
    else:  # high
        preferred_tier = "cloud"

    # Get tier with intelligent fallback (raises AuthError if auth issue)
    # Pass cloud_provider so Auto mode checks the user's selected provider (not default Claude)
    tier, fallback_reason, unavailable_tiers = get_tier_with_fallback(
        preferred_tier, complexity, cloud_provider=cloud_provider
    )

    # If no tier available, raise error
    if tier is None:
        logger.error(f"Routing failed: {fallback_reason}")
        raise Exception("All AI services are currently unavailable. Please try again later.")

    fallback_used = tier != preferred_tier

    return RoutingResult(
        tier=tier,
        complexity=complexity,
        preferred_tier=preferred_tier,
        fallback_used=fallback_used,
        unavailable_tiers=unavailable_tiers,
    )


def get_model_for_tier(tier: str, cloud_provider: str | None = None) -> str:
    """
    Get model name for a tier.

    Args:
        tier: The tier to get model for (local, lakeshore, cloud)
        cloud_provider: Optional cloud provider override for cloud tier
                       (cloud-claude, cloud-gpt, cloud-gpt-cheap)

    Returns:
        Model name to use with LiteLLM
    """
    # For cloud tier, allow user to select specific provider
    if tier == "cloud" and cloud_provider:
        return cloud_provider
    return DEFAULT_MODELS.get(tier, DEFAULT_MODELS["local"])
