"""
Intelligent query routing with automatic tier fallback.

This module determines which AI tier to use for a query based on:
1. User preference (if specified)
2. Query complexity (from complexity_judge)
3. Tier availability (from tier_health)
4. Budget-aware adaptive threshold θ (optional)
"""

import logging
import threading
import time

from stream.middleware.config import (
    DEFAULT_MODELS,
    DEFAULT_VISION_MODELS,
    LLM_JUDGE_ENABLED,
    ROUTING_BUDGET_PERIOD,
    ROUTING_BUDGET_USD,
    ROUTING_THETA_BASE,
)
from stream.middleware.core.complexity_judge import (
    judge_complexity_with_keywords,
    judge_complexity_with_llm,
)
from stream.middleware.core.tier_health import (
    get_tier_error,
    is_tier_available,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


class _BudgetTracker:
    """
    Thread-safe tracker for cloud spend within a rolling period.

    Implements θ_effective = max(θ_base, cumulative_spend / budget).
    Resets automatically at the start of each period (daily / weekly / monthly).
    """

    _PERIOD_SECONDS = {
        "daily": 86_400,
        "weekly": 7 * 86_400,
        "monthly": 30 * 86_400,
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._spend_usd = 0.0
        self._period_start = time.time()

    def _reset_if_needed(self) -> None:
        period_s = self._PERIOD_SECONDS.get(ROUTING_BUDGET_PERIOD, 30 * 86_400)
        if time.time() - self._period_start >= period_s:
            self._spend_usd = 0.0
            self._period_start = time.time()
            logger.info("Budget tracker: period reset (spend → $0.00)")

    def record_cloud_query(self, cost_usd: float = 0.0) -> None:
        """Call after a query is routed to cloud. Pass estimated cost if known."""
        with self._lock:
            self._reset_if_needed()
            self._spend_usd += cost_usd

    def effective_theta(self) -> float:
        """
        Return θ_effective for the current moment.

        If ROUTING_BUDGET_USD == 0 (disabled), always returns ROUTING_THETA_BASE.
        Otherwise: max(θ_base, spend / budget), capped at 0.999 so that even at
        100% budget depletion the most genuinely HIGH queries still reach cloud.
        """
        if ROUTING_BUDGET_USD <= 0:
            return ROUTING_THETA_BASE
        with self._lock:
            self._reset_if_needed()
            ratio = self._spend_usd / ROUTING_BUDGET_USD
        return min(max(ROUTING_THETA_BASE, ratio), 0.999)

    @property
    def spend_usd(self) -> float:
        with self._lock:
            self._reset_if_needed()
            return self._spend_usd

    @property
    def budget_usd(self) -> float:
        return ROUTING_BUDGET_USD


# Module-level singleton — shared across all requests
budget_tracker = _BudgetTracker()


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

    Only Level 1 health checks are used during routing (fast, no GPU jobs).
    lakeshore_model is intentionally NOT accepted here — passing it would
    trigger a Level 2 check (1-token inference via Globus Compute, ~10-30s)
    that blocks the user's query. Cloud provider IS passed because cloud
    Level 2 checks are fast (~1s HTTP call).

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
        # During routing, we only check Level 1 health (fast, no GPU jobs):
        #   - Local: is Ollama reachable?
        #   - Lakeshore: is Globus authenticated and configured?
        #   - Cloud: is the API key valid?
        #
        # We intentionally DON'T pass lakeshore_model here because that
        # triggers a Level 2 check (1-token inference via Globus Compute,
        # ~10-30s) which blocks the user's query. If Lakeshore is configured
        # and authenticated, we trust it and try the inference directly.
        # If it fails, the streaming/batch fallback handles it gracefully.
        #
        # Cloud provider IS passed because cloud Level 2 checks are fast
        # (~1s HTTP call) and catch real issues like expired API keys.
        tier_cloud_provider = cloud_provider if tier == "cloud" else None

        if is_tier_available(
            tier,
            cloud_provider=tier_cloud_provider,
        ):
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
            # Check if this is an auth error - DON'T fall back, show error to user.
            # Only pass cloud_provider (fast check). Lakeshore_model is NOT passed
            # for the same reason as above — it would trigger a Level 2 GPU job.
            error_msg, error_type = get_tier_error(
                tier,
                cloud_provider=tier_cloud_provider,
            )
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
    lakeshore_model: str | None = None,
) -> RoutingResult:
    """
    Determine which tier to use based on LLM judge + keyword fallback + health checks.

    Args:
        query: The user's query text
        user_preference: "auto", "local", "lakeshore", or "cloud"
        cloud_provider: For cloud tier, the specific provider (e.g., "cloud-gpt")
        lakeshore_model: For lakeshore tier, the specific model (e.g., "lakeshore-qwen-vl-72b")

    Returns a RoutingResult with:
    - tier: The actual tier to use
    - complexity: Query complexity (low/medium/high)
    - preferred_tier: The tier that would have been used if available
    - fallback_used: Whether a fallback occurred
    - original_tier: The tier that was unavailable (if fallback occurred)
    """
    # If user explicitly chose a tier, respect it strictly (no silent fallback)
    if user_preference in ["local", "lakeshore", "cloud"]:
        # Only Level 1 health check during routing (fast, no GPU jobs).
        # For Lakeshore, we DON'T pass lakeshore_model — that triggers a
        # Level 2 check (1-token inference via Globus Compute, ~10-30s)
        # which blocks the user's query. Instead, trust Globus auth status
        # and try the inference directly. If it fails, streaming/batch
        # fallback handles it gracefully.
        # Cloud provider IS passed because cloud checks are fast (~1s).
        if is_tier_available(
            user_preference,
            cloud_provider=cloud_provider if user_preference == "cloud" else None,
        ):
            return RoutingResult(
                tier=user_preference,
                complexity="user_override",
                preferred_tier=user_preference,
                fallback_used=False,
            )
        else:
            # User explicitly selected this tier - don't silently fallback
            # Raise an error so the user knows their selection couldn't be honored.
            # Same rule: no lakeshore_model to avoid Level 2 GPU jobs.
            error_msg, error_type = get_tier_error(
                user_preference,
                cloud_provider=cloud_provider if user_preference == "cloud" else None,
            )

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
    _soft_p_high: float | None = None  # P(HIGH) from classifier, if available

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

    # ------------------------------------------------------------------
    # Budget-aware adaptive threshold θ (only affects HIGH→cloud routing)
    #
    # When budget tracking is enabled (ROUTING_BUDGET_USD > 0), we apply
    #   θ_effective = max(θ_base, cumulative_spend / budget)
    # to the classifier's P(HIGH) soft score.  If P(HIGH) < θ_effective,
    # the query is downgraded from HIGH to MEDIUM (HPC) instead of cloud.
    #
    # This only fires when:
    #   (a) the query was classified HIGH (would otherwise go to cloud)
    #   (b) budget tracking is active
    #   (c) we can get a soft score from the ModernBERT classifier
    #
    # Queries that hit keyword or LLM judge paths skip the θ gate and are
    # routed normally — budget still rises over time, affecting future θ.
    # ------------------------------------------------------------------
    if complexity == "high" and ROUTING_BUDGET_USD > 0:
        theta_eff = budget_tracker.effective_theta()
        if theta_eff > ROUTING_THETA_BASE:
            # Try to get P(HIGH) from the classifier for the θ gate
            try:
                from stream.middleware.core.complexity_judge import judge_complexity_with_classifier

                _, scores, _ = judge_complexity_with_classifier(query)
                _soft_p_high = scores.get("high", 1.0)
                if _soft_p_high < theta_eff:
                    logger.info(
                        f"Budget θ gate: P(HIGH)={_soft_p_high:.3f} < θ_eff={theta_eff:.3f} "
                        f"(spend=${budget_tracker.spend_usd:.2f}/${ROUTING_BUDGET_USD:.2f}) "
                        f"→ downgrading HIGH→MEDIUM"
                    )
                    complexity = "medium"
            except Exception as e:
                logger.debug(f"Budget θ gate: classifier unavailable ({e}), routing HIGH normally")

    # Map complexity to preferred tier
    if complexity == "low":
        preferred_tier = "local"
    elif complexity == "medium":
        preferred_tier = "lakeshore"
    else:  # high
        preferred_tier = "cloud"

    # Get tier with intelligent fallback (raises AuthError if auth issue).
    # cloud_provider is passed so cloud health checks test the right API key.
    # lakeshore_model is NOT passed — only Level 1 checks run during routing.
    tier, fallback_reason, unavailable_tiers = get_tier_with_fallback(
        preferred_tier,
        complexity,
        cloud_provider=cloud_provider,
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


def get_model_for_tier(
    tier: str,
    cloud_provider: str | None = None,
    local_model: str | None = None,
    lakeshore_model: str | None = None,
    has_images: bool = False,
) -> str:
    """
    Get model name for a tier, with modality-aware selection.

    MODALITY-AWARE ROUTING LOGIC:
    When the user's query contains images (has_images=True), this function
    needs to ensure a vision-capable model is selected. The behavior depends
    on how the user made their selection:

    1. User specified an explicit model (e.g., local_model="local-llama"):
       → Return it as-is. The caller (chat.py) will check if it's vision-capable
         and raise an error if not. We don't silently override explicit choices.

    2. User selected a tier but NOT a specific model:
       → If images are present, return the default VISION model for that tier.
       → If no images, return the default TEXT model for that tier.

    3. AUTO mode (no tier or model specified):
       → Same as #2 — the tier was already determined by the router.

    Args:
        tier: The tier to get model for (local, lakeshore, cloud)
        cloud_provider: Optional cloud provider override for cloud tier
        local_model: Optional model override for local tier
        lakeshore_model: Optional model override for lakeshore tier
        has_images: Whether the query contains images

    Returns:
        Model name to use for inference
    """
    # CASE 1: User explicitly specified a model — return it as-is.
    # The caller (chat.py Step 4b) handles the modality validation and
    # returns a clear error if the model can't process images.
    if tier == "local" and local_model:
        return local_model
    if tier == "lakeshore" and lakeshore_model:
        return lakeshore_model
    if tier == "cloud" and cloud_provider:
        return cloud_provider

    # CASE 2 & 3: No explicit model — STREAM picks the best model.
    # If images are present, use the default vision model for this tier.
    # This is NOT a silent override — the user only chose the tier (or AUTO),
    # so picking the right model within that tier is STREAM's job.
    if has_images:
        vision_model = DEFAULT_VISION_MODELS.get(tier)
        if vision_model:
            logger.info(
                f"Selecting vision model '{vision_model}' for tier '{tier}' "
                f"(query contains images)"
            )
            return vision_model

    # No images or no vision model available — use the text default
    return DEFAULT_MODELS.get(tier, DEFAULT_MODELS["local"])
