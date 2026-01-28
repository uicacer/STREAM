"""
Calculate query costs based on token usage.

Single source of truth: Reads pricing from LiteLLM config via cost_reader.
"""

import logging

from stream.middleware.utils.cost_reader import get_model_cost

logger = logging.getLogger(__name__)


def calculate_query_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for a query.

    Reads pricing from LiteLLM config (single source of truth).

    Args:
        model: Model identifier (e.g., "cloud-claude")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        float: Total cost in USD

    Example:
        >>> calculate_query_cost("cloud-claude", 100, 200)
        0.0033  # $0.000003 * 100 + $0.000015 * 200
    """
    # Get pricing from LiteLLM config
    costs = get_model_cost(model)

    # Calculate
    input_cost = input_tokens * costs["input"]
    output_cost = output_tokens * costs["output"]
    total_cost = input_cost + output_cost

    logger.debug(
        f"Cost calculation: {model} | "
        f"in={input_tokens}×${costs['input']:.8f} + out={output_tokens}×${costs['output']:.8f} = ${total_cost:.8f}"
    )

    return total_cost
