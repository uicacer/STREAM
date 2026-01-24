"""
Token estimation and context window validation utilities.

This module provides functions to estimate token counts for messages and validate
that conversations stay within model context limits. These are critical for
preventing crashes and ensuring smooth user experience.

Token estimation uses the rule of thumb: 1 token ≈ 4 characters
This is approximate but sufficient for pre-flight checks.
"""

import logging

from stream.middleware.config import MODEL_CONTEXT_LIMITS

logger = logging.getLogger(__name__)


def estimate_tokens(messages: list[dict]) -> int:
    """
    Estimate total token count for a list of chat messages.

    Uses a simple heuristic: divide total character count by 4.
    This approximation works reasonably well across most models and is
    computationally cheap (no actual tokenization required).

    Args:
        messages: List of message dictionaries, each containing:
                 - role: str (user/assistant/system)
                 - content: str (message text)

    Returns:
        Estimated total token count across all messages.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello world"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> estimate_tokens(messages)
        6  # ~24 characters / 4 = 6 tokens

    Note:
        This is intentionally conservative (may overestimate slightly)
        to provide a safety margin for context window checks.
    """
    total_chars = 0

    for message in messages:
        # Extract content, default to empty string if missing
        content = message.get("content", "")

        # Convert to string (handles non-string content gracefully)
        total_chars += len(str(content))

    # Apply the 1 token ≈ 4 characters rule
    estimated_tokens = total_chars // 4

    return estimated_tokens


def check_context_limit(estimated_tokens: int, model: str, correlation_id: str) -> tuple[bool, int]:
    """
    Check if estimated tokens exceed a model's context window limit.

    This function prevents sending requests that would fail or get truncated
    due to context window constraints. Different models have different limits:
    - GPT-4: 128,000 tokens
    - Llama 3.2 (3B): 8,000 tokens
    - etc.

    Args:
        estimated_tokens: Number of tokens in the conversation history
        model: Model identifier (e.g., "llama3.2:3b", "gpt-4")
        correlation_id: Request ID for logging/debugging

    Returns:
        Tuple of (within_limit: bool, max_allowed: int)
        - within_limit: True if conversation fits, False if too large
        - max_allowed: Maximum input tokens allowed for this model

    Example:
        >>> check_context_limit(5000, "llama3.2:3b", "req-123")
        (True, 6000)  # 5000 < 6000, within limit

        >>> check_context_limit(10000, "llama3.2:3b", "req-123")
        (False, 6000)  # 10000 > 6000, exceeds limit

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
