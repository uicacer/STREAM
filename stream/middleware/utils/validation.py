"""
Configuration validation utilities.

This module validates that configuration is consistent across files.
"""

import logging
from importlib.resources import files

import yaml

from stream.middleware.config import MODEL_COSTS

LITELLM_CONFIG = files("stream.gateway").joinpath("litellm_config.yaml")

logger = logging.getLogger(__name__)


def validate_costs_match_litellm():
    """
    Validate that MODEL_COSTS matches gateway/litellm_config.yaml

    This ensures single source of truth is maintained.
    """

    try:
        # Read config using importlib.resources (works in packaged apps!)
        config_text = LITELLM_CONFIG.read_text()
        litellm_config = yaml.safe_load(config_text)

    except FileNotFoundError:
        print("⚠️  WARNING: litellm_config.yaml not found, skipping cost validation")
        return
    except Exception as e:
        print(f"⚠️  WARNING: Could not read config: {e}")
        return

    try:
        mismatches = []

        for model_def in litellm_config.get("model_list", []):
            model_name = model_def.get("model_name")
            model_info = model_def.get("model_info", {})

            if model_name in MODEL_COSTS:
                litellm_input = model_info.get("input_cost_per_token", 0.0)
                litellm_output = model_info.get("output_cost_per_token", 0.0)

                stream_input = MODEL_COSTS[model_name]["input"]
                stream_output = MODEL_COSTS[model_name]["output"]

                if litellm_input != stream_input or litellm_output != stream_output:
                    mismatches.append(
                        f"  {model_name}: LiteLLM({litellm_input}/{litellm_output}) != "
                        f"STREAM({stream_input}/{stream_output})"
                    )

        if mismatches:
            print("\n❌ COST MISMATCH DETECTED!")
            print("   Costs in litellm_config.yaml don't match middleware/config.py:")
            for mismatch in mismatches:
                print(mismatch)
            print("\n   💡 Update costs in BOTH files to match.")
            print(
                "   Single source of truth: middleware/config.py -> sync to litellm_config.yaml\n"
            )
        else:
            print("✅ Cost validation passed: middleware/config.py matches litellm_config.yaml")

    except Exception as e:
        print(f"⚠️  WARNING: Could not validate costs: {e}")


# def validate_environment_variables():
#     """Validate that all required environment variables are set."""

#     required_vars = [
#         "MIDDLEWARE_HOST",
#         "MIDDLEWARE_PORT",
#         "OLLAMA_PORT",
#         "LITELLM_BASE_URL",
#         "LITELLM_MASTER_KEY",
#         "LAKESHORE_VLLM_ENDPOINT",
#     ]

#     missing_vars = [var for var in required_vars if var not in os.environ]

#     if missing_vars:
#         print("⚠️  WARNING: Missing required environment variables:")
#         for var in missing_vars:
#             print(f"  - {var}")
#         return False

#     return True
