"""
Query complexity classification using LLM and keyword fallback.

This module determines whether a query is LOW, MEDIUM, or HIGH complexity
to route it to the appropriate AI tier.
"""

import hashlib
import logging
import time

import httpx

from stream.middleware.config import (
    COMPLEXITY_KEYWORDS,
    JUDGE_CACHE_TTL,
    JUDGE_MODEL,
    JUDGE_PROMPT,
    JUDGE_TIMEOUT,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LLM_JUDGE_ENABLED,
)

logger = logging.getLogger(__name__)

# Judge cache (module-level state)
_judge_cache = {}


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


def judge_complexity_with_llm(query: str) -> str | None:
    """
    Use a lightweight LLM to judge query complexity

    Args:
        query: User's question

    Returns:
        "low", "medium", "high", or None if failed
    """
    # Check cache first
    cached = get_cached_judgment(query)
    if cached:
        print(f"🔍 JUDGE: Using cached result → {cached.upper()}")
        return cached

    # Build judge prompt
    prompt = JUDGE_PROMPT.format(query=query)

    try:
        # Call LiteLLM with judge model
        with httpx.Client(timeout=JUDGE_TIMEOUT) as client:
            response = client.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": JUDGE_MODEL,
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
            print(f"⚠️ JUDGE: Failed with status {response.status_code}")
            return None

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
            print(f"⚠️ JUDGE: Unexpected response: {judgment_text}")
            return None

        # Cache the judgment
        _cache_judgment(query, judgment)

        print(f"🔍 JUDGE: LLM classified as → {judgment.upper()}")
        return judgment

    except httpx.TimeoutException:
        print(f"⚠️ JUDGE: Timeout after {JUDGE_TIMEOUT}s")
        return None

    except Exception as e:
        print(f"⚠️ JUDGE: Error: {str(e)}")
        return None


def judge_complexity_with_keywords(query: str) -> str:
    """Fallback: Use keyword matching to judge complexity."""
    query_lower = query.lower()

    # Check LOW keywords FIRST (most specific patterns)
    # "what is" is more specific than just "is"
    for kw in COMPLEXITY_KEYWORDS["low"]:
        if kw in query_lower:
            logger.debug(f"Matched LOW keyword: '{kw}'")
            return "low"

    # Then check MEDIUM
    for kw in COMPLEXITY_KEYWORDS["medium"]:
        if kw in query_lower:
            logger.debug(f"Matched MEDIUM keyword: '{kw}'")
            return "medium"

    # Then check HIGH
    for kw in COMPLEXITY_KEYWORDS["high"]:
        if kw in query_lower:
            logger.debug(f"Matched HIGH keyword: '{kw}'")
            return "high"

    # Default: medium (safer to overestimate)
    logger.debug("No keywords matched, defaulting to MEDIUM")
    return "medium"


def judge_complexity(query: str) -> str:
    """
    Judge query complexity using LLM with keyword fallback.

    Returns:
        "low", "medium", or "high"
    """
    if LLM_JUDGE_ENABLED:
        complexity = judge_complexity_with_llm(query)
        if complexity:
            return complexity
        logger.warning("LLM judge failed, falling back to keywords")

    return judge_complexity_with_keywords(query)
