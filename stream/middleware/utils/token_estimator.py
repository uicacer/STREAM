"""
Token estimation utilities.

This module provides functions to estimate token counts from messages or text.
Uses the rule of thumb: 1 token ≈ 4 characters for text content.

For multimodal content (images), a fixed estimate of 765 tokens per image
is used, based on OpenAI's convention for "auto" detail level images.

Purpose: Answer "How many tokens is this?"

WHY ACCURATE IMAGE TOKEN ESTIMATION MATTERS:
=============================================
Before multimodal support, this module used len(str(content)) to estimate
tokens. This worked fine for text, but when content is a list of blocks
(OpenAI vision format), str() would convert the entire list — including
base64 image data — to a string and count its characters.

A single base64-encoded image is ~500,000 characters (for a 375 KB JPEG).
At 4 chars/token, that would estimate ~125,000 tokens — far more than the
actual ~765 tokens the model uses to process the image. This would cause
EVERY image query to be rejected as "context too long."

The fix: count text characters normally, but use a fixed 765-token estimate
per image. This matches how vision models actually tokenize images.
"""

import logging

logger = logging.getLogger(__name__)

# How many tokens a single image consumes in the model's context window.
# OpenAI uses 765 tokens for "auto" detail level (the most common setting).
# This is a conservative estimate that works across providers:
#   - OpenAI: 85 tokens (low detail) to 1,105 tokens (high detail)
#   - Anthropic: ~1,000 tokens per image
#   - vLLM: varies by model, but 765 is a reasonable middle ground
#
# Using 765 ensures we don't reject queries that would actually fit,
# while still leaving enough safety margin for context window checks.
TOKENS_PER_IMAGE = 765


def estimate_tokens(messages: list[dict]) -> int:
    """
    Estimate total token count for a list of chat messages.

    Handles both text-only and multimodal (image) messages:
      - Text content: 1 token ≈ 4 characters (standard approximation)
      - Image content: 765 tokens per image (OpenAI convention)

    Args:
        messages: List of message dictionaries, each containing:
                 - role: str (user/assistant/system)
                 - content: str | list[dict] (text or multimodal blocks)

    Returns:
        Estimated total token count across all messages

    Examples:
        Text-only:
        >>> messages = [{"role": "user", "content": "Hello world"}]
        >>> estimate_tokens(messages)
        2  # 11 chars // 4 = 2

        Multimodal (text + 1 image):
        >>> messages = [{"role": "user", "content": [
        ...     {"type": "text", "text": "What is this?"},
        ...     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ... ]}]
        >>> estimate_tokens(messages)
        768  # 3 text tokens + 765 image tokens
    """
    total_chars = 0
    image_count = 0

    for message in messages:
        content = message.get("content", "")

        if isinstance(content, str):
            # Simple text message: count characters normally
            total_chars += len(content)
        elif isinstance(content, list):
            # Multimodal message: process each content block separately
            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    # Text block: count its characters for token estimation
                    total_chars += len(block.get("text", ""))
                elif block_type == "image_url":
                    # Image block: use fixed token estimate.
                    # We intentionally do NOT count the base64 string length —
                    # that would massively overestimate (a 500KB image = 125K
                    # "tokens" by character count, but only ~765 actual tokens).
                    image_count += 1

    # Calculate total: text tokens + image tokens
    text_tokens = total_chars // 4
    image_tokens = image_count * TOKENS_PER_IMAGE

    return text_tokens + image_tokens


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
