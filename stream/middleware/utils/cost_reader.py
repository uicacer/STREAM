"""
Read model pricing from LiteLLM configuration.

This ensures a single source of truth for costs.
Middleware uses this for real-time calculation.
Frontend uses this via API for display.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Cache pricing (loaded once at startup)
_MODEL_PRICING = None


def load_model_pricing() -> dict:
    """
    Load model pricing from LiteLLM config.

    Returns:
        dict: Model pricing {model_name: {input: X, output: Y}}

    Example:
        >>> pricing = load_model_pricing()
        >>> pricing["cloud-claude"]
        {'input': 0.000003, 'output': 0.000015}
    """
    global _MODEL_PRICING

    if _MODEL_PRICING is not None:
        return _MODEL_PRICING

    try:
        # Path to LiteLLM config
        config_path = Path(__file__).parent.parent.parent / "gateway" / "litellm_config.yaml"

        if not config_path.exists():
            logger.error(f"❌ LiteLLM config not found at: {config_path}")
            return {}

        # Read and parse YAML
        with open(config_path) as f:
            litellm_config = yaml.safe_load(f)

        # Extract pricing from model_list
        pricing = {}
        for model_def in litellm_config.get("model_list", []):
            model_name = model_def.get("model_name")
            model_info = model_def.get("model_info", {})

            if model_name:
                pricing[model_name] = {
                    "input": model_info.get("input_cost_per_token", 0.0),
                    "output": model_info.get("output_cost_per_token", 0.0),
                }

        _MODEL_PRICING = pricing
        logger.info(f"✅ Loaded pricing for {len(pricing)} models from LiteLLM config")
        return pricing

    except Exception as e:
        logger.error(f"❌ Failed to load pricing from LiteLLM config: {e}", exc_info=True)
        return {}


def get_model_cost(model: str) -> dict:
    """
    Get cost rates for a specific model.

    Args:
        model: Model identifier (e.g., "cloud-claude", "local-llama")

    Returns:
        dict: {"input": X, "output": Y} or {"input": 0, "output": 0} if not found

    Example:
        >>> costs = get_model_cost("cloud-claude")
        >>> costs["input"]
        0.000003
    """
    pricing = load_model_pricing()
    return pricing.get(model, {"input": 0.0, "output": 0.0})


# def get_all_model_costs() -> dict:
#     """
#     Get pricing for all models.

#     Returns:
#         dict: All model pricing

#     Example:
#         >>> all_costs = get_all_model_costs()
#         >>> all_costs.keys()
#         dict_keys(['local-llama', 'cloud-claude', ...])
#     """
#     return load_model_pricing()
