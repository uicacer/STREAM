"""
Calculate query costs based on token usage.

Single source of truth: Reads pricing from LiteLLM config via cost_reader.
"""

import logging

from stream.middleware.utils.cost_reader import get_model_cost

logger = logging.getLogger(__name__)


def calculate_query_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """
    Calculate cost for a query, accounting for prompt caching discounts.

    Cached reads are billed at 0.1× the input rate (Anthropic) or 0.5× (OpenAI).
    Cache creation is billed at 1.25× the input rate (Anthropic).
    We use Anthropic's multipliers as the default since that's the primary cloud tier.

    Args:
        model: Model identifier (e.g., "cloud-claude")
        input_tokens: Total prompt tokens (including any cached subset)
        output_tokens: Number of output tokens
        cache_read_tokens: Tokens served from KV cache (subset of input_tokens)
        cache_creation_tokens: Tokens written to KV cache (subset of input_tokens)

    Returns:
        float: Total cost in USD
    """
    costs = get_model_cost(model)
    input_price = costs["input"]

    # Tokens that were neither cached reads nor cache writes are billed at full price.
    regular_tokens = input_tokens - cache_read_tokens - cache_creation_tokens

    input_cost = (
        regular_tokens * input_price
        + cache_read_tokens * input_price * 0.1
        + cache_creation_tokens * input_price * 1.25
    )
    output_cost = output_tokens * costs["output"]
    total_cost = input_cost + output_cost

    logger.debug(
        f"Cost calculation: {model} | "
        f"regular={regular_tokens}×${input_price:.8f} "
        f"cache_read={cache_read_tokens}×${input_price * 0.1:.8f} "
        f"cache_write={cache_creation_tokens}×${input_price * 1.25:.8f} "
        f"out={output_tokens}×${costs['output']:.8f} = ${total_cost:.8f}"
    )

    return total_cost
