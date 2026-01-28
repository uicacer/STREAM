"""
Token estimation utilities.

This module provides functions to estimate token counts from messages or text.
Uses the rule of thumb: 1 token ≈ 4 characters

Purpose: Answer "How many tokens is this?"
"""

import logging

logger = logging.getLogger(__name__)


def estimate_tokens(messages: list[dict]) -> int:
    """
    Estimate total token count for a list of chat messages.

    Uses the rule of thumb: 1 token ≈ 4 characters
    This approximation works reasonably well across most models and is
    computationally cheap (no actual tokenization required).

    Args:
        messages: List of message dictionaries, each containing:
                 - role: str (user/assistant/system)
                 - content: str (message text)

    Returns:
        Estimated total token count across all messages

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


def estimate_tokens_from_text(text: str) -> int:
    """
    Estimate token count from raw text.

    Uses the rule of thumb: 1 token ≈ 4 characters

    Args:
        text: Raw text string

    Returns:
        Estimated token count

    Example:
        >>> estimate_tokens_from_text("Hello world")
        2  # 11 chars // 4 = 2.75 ≈ 2

    Use Case:
        - Estimate output tokens when LLM doesn't provide usage
        - Quick token checks without full message structure
    """
    return len(text) // 4
