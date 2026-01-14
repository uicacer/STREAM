# =============================================================================
# STREAM - Configuration
# =============================================================================

"""
STREAM Backend Configuration

Central configuration for:
- LiteLLM gateway connection settings
- Model definitions (local/Ollama, cloud/Anthropic, lakeshore/vLLM)
- Routing rules and complexity thresholds
- Streamlit UI settings and feature flags
- Example queries and development settings
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# LITELLM GATEWAY SETTINGS
# =============================================================================

LITELLM_BASE_URL = "http://localhost:4000"
LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODELS = {
    "local": {
        "tiny": "local-llama-tiny",
        "default": "local-llama",
        "quality": "local-llama-quality",
    },
    "cloud": {"claude": "cloud-claude", "gpt": "cloud-gpt", "gpt_cheap": "cloud-gpt-cheap"},
    "lakeshore": {"llama": "lakeshore-llama"},
}

DEFAULT_MODELS = {
    "local": MODELS["local"]["default"],
    "cloud": MODELS["cloud"]["claude"],
    "lakeshore": MODELS["lakeshore"]["llama"],
}

# =============================================================================
# ROUTING RULES
# =============================================================================

MAX_TOKENS = 2000

TOKEN_LIMITS = {
    "local_max_input": 2000,
    "local_max_output": 1000,
}

COST_THRESHOLDS = {
    "daily_limit": 10.00,
    "query_warning": 0.50,
}


# =============================================================================
# STREAMLIT UI SETTINGS
# =============================================================================

APP_TITLE = "STREAM"
APP_SUBTITLE = "Smart Tiered Routing Engine for AI Models"
APP_ICON = "🌊"

UI_CONFIG = {
    "show_tier_badge": True,
    "show_cost_estimate": True,
    "enable_streaming": True,
    "max_history": 50,
}

CHAT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 2000,
    "stream": True,
}

# =============================================================================
# FEATURE FLAGS
# =============================================================================

FEATURES = {
    "cost_tracking": True,  # Track costs per query and session
    # TODO: Future features
    # "slurm_tools": False,    # SLURM cluster management tools
    # "web_search": False,     # Web search integration
    # "file_upload": False,    # Document upload/analysis
    # "user_auth": False,      # User authentication
}

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"

# =============================================================================
# EXAMPLE QUERIES
# =============================================================================

EXAMPLE_QUERIES = [
    "What is Python?",
    "Explain how neural networks work",
    "Write a Python function to calculate fibonacci numbers",
    "Compare React and Vue.js frameworks",
]
