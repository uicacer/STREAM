"""
Context window limit utilities.

This module provides helper functions for working with model context limits.
Context limits define how many tokens a model can process in a single request.
"""

import logging

from stream.middleware.config import MODEL_CONTEXT_LIMITS

logger = logging.getLogger(__name__)


def get_max_input_tokens(model: str) -> int:
    """
    Get maximum input tokens allowed for a model.

    This calculates: total_context - reserve_output

    Args:
        model: Model identifier (e.g., "gpt-4", "llama3.2:3b")

    Returns:
        int: Maximum input tokens allowed

    Example:
        >>> get_max_input_tokens("local-llama")
        1748  # 2048 total - 300 reserved

    Note:
        If model not found in config, assumes large context (196K)
        for forward compatibility with future models.
    """
    config = MODEL_CONTEXT_LIMITS.get(model)
    if config:
        return config["total"] - config["reserve_output"]

    # Default: assume large context (Claude-sized)
    logger.warning(
        f"No context limit configured for {model}, assuming 196K", extra={"model": model}
    )
    return 196000


def get_tier_context_limits() -> dict:
    """
    Get context limits organized by tier for frontend display.

    This provides tier-level summaries (not model-specific) for UI purposes.
    Uses the default model for each tier to determine limits.

    Returns:
        dict: Tier-level context limits with notes

    Example:
        >>> limits = get_tier_context_limits()
        >>> limits["local"]["max_input"]
        1748

    Use Case:
        - Display limits in UI before user submits query
        - Help users understand tier capabilities
        - Guide users to select appropriate tier
    """
    return {
        "local": {
            "max_input": get_max_input_tokens("local-llama"),
            "total": MODEL_CONTEXT_LIMITS["local-llama"]["total"],
            "note": "Ollama default (models capable of 128k if reconfigured)",
        },
        "lakeshore": {
            "max_input": get_max_input_tokens("lakeshore-llama"),
            "total": MODEL_CONTEXT_LIMITS["lakeshore-llama"]["total"],
            "note": "vLLM conservative (models capable of 128k)",
        },
        "cloud": {
            "max_input": get_max_input_tokens("cloud-claude"),
            "total": MODEL_CONTEXT_LIMITS["cloud-claude"]["total"],
            "note": "Claude Sonnet 4 maximum",
        },
    }
