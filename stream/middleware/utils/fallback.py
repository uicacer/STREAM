"""
Tier fallback logic for automatic failover.

When a tier fails (local Ollama down, campus GPU unavailable, cloud API error),
this module determines which tier to try next. The fallback order depends on
query complexity:

- Simple queries: Try cheap tiers first (Lakeshore → Cloud → Local)
- Complex queries: Try capable tiers first (Cloud → Lakeshore → Local)

This ensures:
1. Cost optimization (use free resources when possible)
2. High availability (automatic failover to working tiers)
3. Appropriate routing (complex queries go to powerful models)
"""

import logging

from stream.middleware.core.tier_health import is_tier_available

logger = logging.getLogger(__name__)


def get_fallback_tier(complexity: str, already_tried: list[str]) -> str | None:
    """
    Determine the next tier to attempt after a failure.

    Implements intelligent fallback chains based on query complexity:

    Low/Medium Complexity:
        Priority: Cost optimization
        Chain: lakeshore (free) → cloud (reliable) → local (last resort)
        Rationale: Simple queries work fine on free campus GPU

    High Complexity:
        Priority: Capability
        Chain: cloud (most capable) → lakeshore (decent) → local (weakest)
        Rationale: Complex queries need powerful models

    Args:
        complexity: Query complexity level ("low", "medium", or "high")
        already_tried: List of tiers that have already failed
                      (prevents infinite retry loops)

    Returns:
        Next tier name to try ("local", "lakeshore", or "cloud"),
        or None if no fallbacks remain

    Example:
        >>> # Simple query, first attempt failed
        >>> get_fallback_tier("medium", ["lakeshore"])
        "cloud"  # Try cloud next

        >>> # Complex query, first attempt failed
        >>> get_fallback_tier("high", ["cloud"])
        "lakeshore"  # Try campus GPU next

        >>> # All tiers exhausted
        >>> get_fallback_tier("low", ["lakeshore", "cloud", "local"])
        None  # No more fallbacks available

    Note:
        This function also checks tier availability via is_tier_available()
        to avoid attempting tiers that are known to be down.
    """

    # Define fallback chains based on complexity
    if complexity in ("low", "medium"):
        # For simple queries: Prioritize cost (free tiers first)
        fallback_chain = ["lakeshore", "cloud", "local"]
        logger.debug(f"Using cost-optimized fallback chain for {complexity} complexity")
    else:
        # For complex queries: Prioritize capability (powerful models first)
        fallback_chain = ["cloud", "lakeshore", "local"]
        logger.debug(f"Using capability-optimized fallback chain for {complexity} complexity")

    # Try each tier in the chain
    for tier in fallback_chain:
        # Skip tiers we've already tried (prevents infinite loops)
        if tier in already_tried:
            logger.debug(f"Skipping {tier} (already tried)")
            continue

        # Check if tier is currently available
        if not is_tier_available(tier):
            logger.debug(f"Skipping {tier} (not available)")
            continue

        # Found a viable fallback
        logger.info(f"Selected fallback tier: {tier}")
        return tier

    # No viable fallbacks remaining
    logger.warning(
        f"No fallback tiers available. Already tried: {already_tried}",
        extra={"already_tried": already_tried, "complexity": complexity},
    )
    return None


def get_fallback_reason(error: Exception) -> str:
    """
    Determine human-readable reason for fallback.

    Args:
        error: Exception that triggered the fallback

    Returns:
        User-friendly error description

    Example:
        >>> get_fallback_reason(TimeoutError())
        "Request timeout"
        >>> get_fallback_reason(ConnectionError())
        "Connection failed"
    """
    error_str = str(error).lower()

    if "timeout" in error_str:
        return "Request timeout"
    elif "connection" in error_str:
        return "Connection failed"
    elif "unavailable" in error_str:
        return "Service unavailable"
    elif "500" in error_str:
        return "Internal server error"
    else:
        return "Service error"
