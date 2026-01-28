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
    get_available_tiers,
    is_tier_available,
)

logger = logging.getLogger(__name__)


def get_tier_with_fallback(preferred_tier: str, complexity: str) -> tuple[str, str]:
    """Get tier with intelligent fallback"""
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

    # Try each tier in order
    for tier in fallback_chain:
        if is_tier_available(tier):
            if tier == preferred_tier:
                return tier, f"{complexity.upper()} → {tier.upper()}"
            else:
                return (
                    tier,
                    f"{complexity.upper()} → {preferred_tier.upper()} unavailable, using {tier.upper()}",
                )

    # No tiers available!
    return None, "All AI services unavailable"


def get_tier_for_query(query: str, user_preference: str = "auto") -> str:
    """
    Determine which tier to use based on LLM judge + keyword fallback + health checks
    """
    # If user explicitly chose a tier, respect it (but check health)
    if user_preference in ["local", "lakeshore", "cloud"]:
        if is_tier_available(user_preference):
            print(f"🔍 ROUTING: User override → {user_preference.upper()}")
            return user_preference
        else:
            print(f"⚠️  ROUTING: User selected {user_preference.upper()} but it's unavailable")
            # Continue to auto-routing with fallback

    # Try LLM judge first (if enabled)
    complexity = None
    method = "unknown"

    if LLM_JUDGE_ENABLED:
        complexity = judge_complexity_with_llm(query)
        if complexity:
            method = "LLM judge"
        else:
            print("⚠️ ROUTING: LLM judge failed, falling back to keywords")

    # Fallback to keyword-based if LLM failed or disabled
    if complexity is None:
        complexity = judge_complexity_with_keywords(query)
        method = "keyword matching"

    # Map complexity to preferred tier
    if complexity == "low":
        preferred_tier = "local"
    elif complexity == "medium":
        preferred_tier = "lakeshore"
    else:  # high
        preferred_tier = "cloud"

    # Get tier with intelligent fallback
    tier, fallback_reason = get_tier_with_fallback(preferred_tier, complexity)

    # If no tier available, raise error
    if tier is None:
        print(f"❌ ROUTING FAILED: {fallback_reason}")
        print(f"   Available tiers: {get_available_tiers()}")
        raise Exception("All AI services are currently unavailable. Please try again later.")

    # Debug logging
    print(f"🔍 SMART ROUTING ({method}):")
    print(f"   Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
    print(f"   Complexity: {complexity.upper()}")
    print(f"   Decision: {fallback_reason}")

    return tier


def get_model_for_tier(tier: str) -> str:
    """Get model name for a tier"""
    return DEFAULT_MODELS.get(tier, DEFAULT_MODELS["local"])
