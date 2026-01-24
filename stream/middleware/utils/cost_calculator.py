"""
Cost calculation utilities for tracking AI model usage expenses.

This module provides functions to calculate the cost of LLM API calls based on
token usage. Different models have different pricing structures:
- Cloud models (GPT-4, Claude): Pay per token
- Local/Campus models (Ollama, vLLM): Free (already paid for hardware)

Pricing is configured centrally in config.py as the single source of truth.
"""

import logging

from stream.middleware.config import MODEL_COSTS

logger = logging.getLogger(__name__)


def calculate_query_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate total cost for a single LLM query.

    Cost formula:
        Total = (input_tokens × input_price) + (output_tokens × output_price)

    Different models have different pricing:
        - GPT-4: $0.03/1K input tokens, $0.06/1K output tokens
        - Claude Sonnet: $0.003/1K input, $0.015/1K output
        - Local models: $0.00 (already paid for hardware)

    Args:
        model: Model identifier (e.g., "gpt-4", "llama3.2:3b")
        input_tokens: Number of tokens in the prompt/conversation history
        output_tokens: Number of tokens in the model's response

    Returns:
        Total cost in USD as a float

    Example:
        >>> # GPT-4 query: 1000 input tokens, 500 output tokens
        >>> calculate_query_cost("gpt-4", 1000, 500)
        0.06  # $0.03 (input) + $0.03 (output) = $0.06

        >>> # Local model: always free
        >>> calculate_query_cost("llama3.2:3b", 1000, 500)
        0.0

    Note:
        Returns 0.0 if model pricing is not configured (graceful degradation).
        This allows the system to continue functioning even without cost data.
    """

    # Check if we have pricing data for this model
    if model not in MODEL_COSTS:
        logger.warning(f"⚠️ No cost data configured for model: {model}", extra={"model": model})
        return 0.0

    # Get pricing structure for this model
    pricing = MODEL_COSTS[model]

    # Extract per-token prices
    # These are typically very small numbers (e.g., 0.00003 = $0.03 per 1000 tokens)
    input_price_per_token = pricing["input"]
    output_price_per_token = pricing["output"]

    # Calculate costs
    input_cost = input_tokens * input_price_per_token
    output_cost = output_tokens * output_price_per_token
    total_cost = input_cost + output_cost

    # Log for debugging (optional, can be removed in production)
    logger.debug(
        f"Cost calculation: {model} - {input_tokens} input + {output_tokens} output = ${total_cost:.6f}",
        extra={
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
        },
    )

    return total_cost


def format_cost(cost: float) -> str:
    """
    Format cost for user display.

    Args:
        cost: Cost in USD

    Returns:
        Formatted string (e.g., "$0.06", "$0.00001", "$12.34")

    Example:
        >>> format_cost(0.06)
        '$0.06'
        >>> format_cost(0.000012)
        '$0.000012'
    """
    # For very small costs, use more decimal places
    if cost < 0.01:
        return f"${cost:.6f}"
    else:
        return f"${cost:.2f}"
