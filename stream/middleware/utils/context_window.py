"""
Context window validation utilities.

This module provides functions to validate that conversations stay within
model context limits. Context limits define how many tokens a model can
process in a single request.

Purpose: Answer "Does this fit in the model's context window?"
"""

import logging

from stream.middleware.config import MODEL_CONTEXT_LIMITS

logger = logging.getLogger(__name__)


def get_max_input_tokens(model: str) -> int:
    """
    Get maximum input tokens allowed for a model.

    This calculates: total_context - reserve_output

    Args:
        model: Model identifier (e.g., "local-llama", "cloud-claude")

    Returns:
        Maximum input tokens allowed for this model

    Example:
        >>> get_max_input_tokens("local-llama")
        1748  # 2048 total - 300 reserved for output

    Note:
        If model not found in config, assumes large context (196K)
        for forward compatibility with future models.
    """
    config = MODEL_CONTEXT_LIMITS.get(model)
    if config:
        return config["total"] - config["reserve_output"]

    # Default: assume large context (Claude-sized) for unknown models
    logger.warning(
        f"No context limit configured for {model}, assuming 196K", extra={"model": model}
    )
    return 196000


def check_context_limit(estimated_tokens: int, model: str, correlation_id: str) -> tuple[bool, int]:
    """
    Check if estimated tokens exceed a model's context window limit.

    This function prevents sending requests that would fail or get truncated
    due to context window constraints. Different models have different limits:
    - Local (Llama 3.2 3B): ~2,000 tokens
    - Lakeshore (vLLM): ~8,000 tokens
    - Cloud (Claude Sonnet 4): ~200,000 tokens

    Args:
        estimated_tokens: Number of tokens in the conversation history
        model: Model identifier (e.g., "local-llama")
        correlation_id: Request ID for logging/debugging

    Returns:
        Tuple of (within_limit: bool, max_allowed: int)
        - within_limit: True if conversation fits, False if too large
        - max_allowed: Maximum input tokens allowed for this model

    Example:
        >>> check_context_limit(5000, "local-llama", "req-123")
        (False, 1748)  # 5000 > 1748, exceeds limit

        >>> check_context_limit(1000, "local-llama", "req-123")
        (True, 1748)  # 1000 < 1748, within limit

    Note:
        If no limit is configured for a model, we assume it's within limits
        (cloud models typically have very large contexts).
    """
    # Look up model configuration
    model_config = MODEL_CONTEXT_LIMITS.get(model)

    if not model_config:
        # No limit defined - assume model can handle it
        # This is typical for cloud models (GPT-4, Claude) with large contexts
        logger.debug(f"[{correlation_id}] No context limit defined for {model}, allowing request")
        return True, float("inf")

    # Calculate maximum allowed input tokens
    # Reserve some tokens for the model's output
    total_context = model_config["total"]
    reserved_for_output = model_config["reserve_output"]
    max_input = total_context - reserved_for_output

    # Check if within limits
    if estimated_tokens > max_input:
        logger.error(
            f"[{correlation_id}] Context exceeded: {estimated_tokens} tokens > {max_input} limit for {model}",
            extra={
                "correlation_id": correlation_id,
                "estimated_tokens": estimated_tokens,
                "max_input": max_input,
                "model": model,
            },
        )
        return False, max_input

    # Within limits - good to go
    logger.debug(f"[{correlation_id}] Context OK: {estimated_tokens} tokens < {max_input} limit")
    return True, max_input


def get_tier_context_limits() -> dict:
    """
    Get context limits organized by tier for frontend display.

    This provides tier-level summaries (not model-specific) for UI purposes.
    Uses the default model for each tier to determine limits.

    Returns:
        Dictionary with tier limits and notes

    Example:
        >>> limits = get_tier_context_limits()
        >>> limits["local"]["max_input"]
        1748
        >>> limits["cloud"]["total"]
        200000

    Use Cases:
        - Display limits in UI before user submits query
        - Help users understand tier capabilities
        - Guide users to select appropriate tier
        - Show warnings when approaching limits
    """
    return {
        "local": {
            "max_input": get_max_input_tokens("local-llama"),
            "total": MODEL_CONTEXT_LIMITS["local-llama"]["total"],
            "note": "Ollama default (models capable of 128k if reconfigured)",
        },
        "lakeshore": {
            "max_input": get_max_input_tokens("lakeshore-qwen"),
            "total": MODEL_CONTEXT_LIMITS["lakeshore-qwen"]["total"],
            "note": "vLLM conservative (models capable of 128k)",
        },
        "cloud": {
            "max_input": get_max_input_tokens("cloud-claude"),
            "total": MODEL_CONTEXT_LIMITS["cloud-claude"]["total"],
            "note": "Claude Sonnet 4 maximum",
        },
    }
