"""
Query complexity classification using LLM and keyword fallback.

This module determines whether a query is LOW, MEDIUM, or HIGH complexity
to route it to the appropriate AI tier.

Supports multiple judge strategies:
- ollama-1b: Fastest local, less accurate, free
- ollama-3b: Balanced accuracy, free (default)
- haiku: Fastest & most accurate, ~$1 per 5,000 judgments
"""

import hashlib
import logging
import time
from dataclasses import dataclass

import httpx

from stream.middleware.config import (
    COMPLEXITY_KEYWORDS,
    DEFAULT_JUDGE_STRATEGY,
    JUDGE_CACHE_TTL,
    JUDGE_PROMPT,
    JUDGE_STRATEGIES,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LLM_JUDGE_ENABLED,
)

logger = logging.getLogger(__name__)

# Judge cache (module-level state)
_judge_cache = {}


@dataclass
class JudgmentResult:
    """Result of complexity judgment with metadata."""

    complexity: str  # "low", "medium", "high"
    method: str  # "llm", "keyword_fallback", "default_fallback"
    strategy_used: str | None = None  # e.g., "ollama-3b", "haiku"
    fallback_reason: str | None = None  # Why fallback was used


def _get_cache_key(query: str) -> str:
    """Generate cache key from query"""
    return hashlib.md5(query.lower().encode()).hexdigest()


def get_cached_judgment(query: str) -> str | None:
    """Get cached judgment if exists and not expired"""
    key = _get_cache_key(query)
    if key in _judge_cache:
        judgment, timestamp = _judge_cache[key]
        if time.time() - timestamp < JUDGE_CACHE_TTL:
            return judgment
        else:
            # Expired, remove
            del _judge_cache[key]
    return None


def _cache_judgment(query: str, judgment: str):
    """Cache a judgment"""
    key = _get_cache_key(query)
    _judge_cache[key] = (judgment, time.time())


def judge_complexity_with_llm(query: str, strategy: str = None) -> tuple[str | None, str | None]:
    """
    Use an LLM to judge query complexity.

    Args:
        query: User's question
        strategy: Judge strategy to use ("ollama-1b", "ollama-3b", "haiku")

    Returns:
        Tuple of (complexity, error_reason) where complexity is "low", "medium", "high", or None if failed
    """
    strategy = strategy or DEFAULT_JUDGE_STRATEGY

    # Validate strategy
    if strategy not in JUDGE_STRATEGIES:
        logger.warning(f"Unknown judge strategy '{strategy}', using default")
        strategy = DEFAULT_JUDGE_STRATEGY

    strategy_config = JUDGE_STRATEGIES[strategy]
    model = strategy_config["model"]
    timeout = strategy_config["timeout"]

    # Check cache first (cache is strategy-agnostic for same queries)
    cached = get_cached_judgment(query)
    if cached:
        logger.debug("🔍 Using cached complexity result")
        return cached, None

    # Build judge prompt
    prompt = JUDGE_PROMPT.format(query=query)

    try:
        # Call LiteLLM with the selected judge model
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,  # Just need one word
                    "temperature": 0.0,  # Deterministic
                },
                headers={
                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )

        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}"
            print(f"⚠️ JUDGE [{strategy}]: Failed with status {response.status_code}")
            return None, error_msg

        # Parse response
        data = response.json()
        judgment_text = data["choices"][0]["message"]["content"].strip().upper()

        # Extract LOW/MEDIUM/HIGH from response
        if "LOW" in judgment_text:
            judgment = "low"
        elif "MEDIUM" in judgment_text:
            judgment = "medium"
        elif "HIGH" in judgment_text:
            judgment = "high"
        else:
            error_msg = f"Unexpected response: {judgment_text}"
            print(f"⚠️ JUDGE [{strategy}]: {error_msg}")
            return None, error_msg

        # Cache the judgment
        _cache_judgment(query, judgment)

        print(f"🔍 JUDGE [{strategy}]: LLM classified as → {judgment.upper()}")
        return judgment, None

    except httpx.TimeoutException:
        error_msg = f"Timeout after {timeout}s"
        print(f"⚠️ JUDGE [{strategy}]: {error_msg}")
        return None, error_msg

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ JUDGE [{strategy}]: Error: {error_msg}")
        return None, error_msg


def judge_complexity_with_keywords(query: str) -> tuple[str, str | None]:
    """
    Fallback: Use keyword matching to judge complexity.

    Returns:
        Tuple of (complexity, matched_keyword) where matched_keyword is None if default was used
    """
    query_lower = query.lower()

    # Check LOW keywords FIRST (most specific patterns)
    for kw in COMPLEXITY_KEYWORDS["low"]:
        if kw in query_lower:
            logger.debug(f"Matched LOW keyword: '{kw}'")
            print(f"🔍 JUDGE [keywords]: Matched '{kw}' → LOW")
            return "low", kw

    # Then check MEDIUM
    for kw in COMPLEXITY_KEYWORDS["medium"]:
        if kw in query_lower:
            logger.debug(f"Matched MEDIUM keyword: '{kw}'")
            print(f"🔍 JUDGE [keywords]: Matched '{kw}' → MEDIUM")
            return "medium", kw

    # Then check HIGH
    for kw in COMPLEXITY_KEYWORDS["high"]:
        if kw in query_lower:
            logger.debug(f"Matched HIGH keyword: '{kw}'")
            print(f"🔍 JUDGE [keywords]: Matched '{kw}' → HIGH")
            return "high", kw

    # Default: medium (safer to overestimate)
    logger.debug("No keywords matched, defaulting to MEDIUM")
    print("🔍 JUDGE [default]: No keywords matched → MEDIUM")
    return "medium", None


def judge_complexity(query: str, strategy: str = None) -> JudgmentResult:
    """
    Judge query complexity using LLM with keyword fallback.

    Args:
        query: User's question
        strategy: Judge strategy ("ollama-1b", "ollama-3b", "haiku"). Default: ollama-3b

    Returns:
        JudgmentResult with complexity, method used, and fallback info if applicable
    """
    strategy = strategy or DEFAULT_JUDGE_STRATEGY

    if LLM_JUDGE_ENABLED:
        complexity, error = judge_complexity_with_llm(query, strategy)
        if complexity:
            return JudgmentResult(
                complexity=complexity,
                method="llm",
                strategy_used=strategy,
                fallback_reason=None,
            )

        # LLM failed - try keyword fallback
        logger.warning(f"LLM judge ({strategy}) failed: {error}. Falling back to keywords.")
        keyword_complexity, matched_keyword = judge_complexity_with_keywords(query)

        if matched_keyword:
            return JudgmentResult(
                complexity=keyword_complexity,
                method="keyword_fallback",
                strategy_used=strategy,
                fallback_reason=f"LLM judge failed ({error}), used keyword matching",
            )
        else:
            # No keywords matched - defaulted to MEDIUM
            return JudgmentResult(
                complexity=keyword_complexity,
                method="default_fallback",
                strategy_used=strategy,
                fallback_reason=f"LLM judge failed ({error}) and no keywords matched",
            )

    # LLM judge disabled - use keywords
    keyword_complexity, matched_keyword = judge_complexity_with_keywords(query)
    if matched_keyword:
        return JudgmentResult(
            complexity=keyword_complexity,
            method="keyword_fallback",
            strategy_used=None,
            fallback_reason="LLM judge disabled",
        )
    else:
        return JudgmentResult(
            complexity=keyword_complexity,
            method="default_fallback",
            strategy_used=None,
            fallback_reason="LLM judge disabled and no keywords matched",
        )


# Legacy function for backwards compatibility
def judge_complexity_simple(query: str, strategy: str = None) -> str:
    """
    Simple wrapper that returns just the complexity string.
    Use judge_complexity() for full metadata.
    """
    result = judge_complexity(query, strategy)
    return result.complexity
