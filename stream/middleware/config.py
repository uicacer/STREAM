"""
STREAM Middleware - Configuration

This module contains ONLY configuration data and constants.
Business logic has been moved to appropriate modules:
- Health checks → core.tier_health
- Complexity judging → core.complexity_judge
- Query routing → core.query_router
- Validation → utils.validation
"""

import os

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SERVICE METADATA
# =============================================================================

SERVICE_NAME = "STREAM Middleware"
SERVICE_VERSION = "1.0.0"
SERVICE_DESCRIPTION = "Smart Tiered Routing Engine for AI Models"

# =============================================================================
# SERVICE CONFIGURATION
# =============================================================================

MIDDLEWARE_HOST = os.getenv("MIDDLEWARE_HOST")
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
RELOAD = os.getenv("RELOAD", "false").lower() == "true"

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# CORS
# =============================================================================

CORS_ORIGINS = os.getenv("CORS_ORIGINS").split(",")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# =============================================================================
# EXTERNAL SERVICES
# =============================================================================

OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY")

# Lakeshore connection configuration
# Two modes: SSH port forwarding (legacy) or Globus Compute (preferred)
LAKESHORE_VLLM_ENDPOINT = os.getenv(
    "LAKESHORE_VLLM_ENDPOINT"
)  # SSH port forward URL (e.g., http://host.docker.internal:8000)

# Lakeshore proxy service configuration (configurable host and port)
LAKESHORE_PROXY_HOST = os.getenv("LAKESHORE_PROXY_HOST", "lakeshore-proxy")
LAKESHORE_PROXY_PORT = int(os.getenv("LAKESHORE_PROXY_PORT", "8001"))
LAKESHORE_PROXY_URL = f"http://{LAKESHORE_PROXY_HOST}:{LAKESHORE_PROXY_PORT}"

USE_GLOBUS_COMPUTE = (
    os.getenv("USE_GLOBUS_COMPUTE", "true").lower() == "true"
)  # Enable Globus Compute mode
GLOBUS_COMPUTE_ENDPOINT_ID = os.getenv(
    "GLOBUS_COMPUTE_ENDPOINT_ID"
)  # Globus endpoint ID for Lakeshore
VLLM_SERVER_URL = os.getenv(
    "VLLM_SERVER_URL", "http://ga-001:8000"
)  # vLLM URL on Lakeshore (for Globus remote execution)

# =============================================================================
# HEALTH CHECKS
# =============================================================================

HEALTH_CHECK_TTL = 360  # 6 minutes - slightly longer than background monitor interval (5 min)
HEALTH_CHECK_TIMEOUT = 5.0

# =============================================================================
# JUDGE CONFIGURATION
# =============================================================================

# Judge strategy options (user can select in UI)
JUDGE_STRATEGIES = {
    "ollama-1b": {
        "model": "local-llama-tiny",
        "name": "Ollama 1b",
        "description": "Fastest local, less accurate, free",
        "icon": "⚡",
        "timeout": 30,
    },
    "ollama-3b": {
        "model": "local-llama",
        "name": "Ollama 3b",
        "description": "Balanced accuracy, free",
        "icon": "🎯",
        "timeout": 60,
    },
    "haiku": {
        "model": "cloud-haiku",
        "name": "Claude Haiku",
        "description": "Fastest & most accurate, ~$1 per 5,000 judgments",
        "icon": "🚀",
        "timeout": 15,
    },
}

# Default judge strategy
DEFAULT_JUDGE_STRATEGY = "ollama-3b"

# Legacy config (for backwards compatibility)
JUDGE_MODEL = JUDGE_STRATEGIES[DEFAULT_JUDGE_STRATEGY]["model"]
JUDGE_TIMEOUT = JUDGE_STRATEGIES[DEFAULT_JUDGE_STRATEGY]["timeout"]
LLM_JUDGE_ENABLED = True
JUDGE_CACHE_TTL = 3600

# =============================================================================
# ROUTING
# =============================================================================

TIERS = {
    "local": {"name": "Local Ollama", "description": "Free local inference"},
    "lakeshore": {"name": "Campus vLLM", "description": "UIC Lakeshore GPU cluster"},
    "cloud": {"name": "Cloud APIs", "description": "Claude, GPT, etc."},
}

DEFAULT_MODELS = {"local": "local-llama", "lakeshore": "lakeshore-qwen", "cloud": "cloud-claude"}


# =============================================================================
# OLLAMA MODELS
# =============================================================================

OLLAMA_MODELS = {
    "local-llama-tiny": "llama3.2:1b",
    "local-llama": "llama3.2:3b",
}

# =============================================================================
# CONTEXT LIMITS
# =============================================================================

MODEL_CONTEXT_LIMITS = {
    # Llama 3.2 models support up to 128K context, but we limit to 8K for performance
    # Higher context = more memory usage and slower inference on local machines
    "local-llama-tiny": {"total": 8192, "reserve_output": 1024},
    "local-llama": {"total": 8192, "reserve_output": 1024},
    "local-llama-quality": {"total": 8192, "reserve_output": 1024},
    "lakeshore-qwen": {"total": 8192, "reserve_output": 500},
    "cloud-claude": {"total": 200000, "reserve_output": 4000},
    "cloud-gpt": {"total": 128000, "reserve_output": 4000},
    "cloud-gpt-cheap": {"total": 16385, "reserve_output": 1000},
}

# =============================================================================
# JUDGE PROMPT
# =============================================================================

JUDGE_PROMPT = """

You are a query complexity classifier. Analyze the following user query and classify its complexity level.

Classification Guidelines:

LOW complexity (route to local model):
- Simple greetings (hi, hello, thanks)
- Basic factual questions (what is X?, who is Y?)
- Simple definitions: "What is Python?", "Define recursion"
- One-word or very short answers
- No reasoning required

MEDIUM complexity (route to campus GPU):
- Explanations that require understanding
- Basic coding tasks (write a function, create a script)
- Step-by-step tutorials
- Comparisons of 2-3 items
- Calculations or problem-solving
- Moderate technical questions

HIGH complexity (route to cloud):
- Deep analysis or critique
- Complex comparisons (multiple factors)
- Advanced coding (optimization, debugging, architecture)
- Research-level questions
- Multi-step reasoning
- Creative writing (essays, stories)
- Production/enterprise considerations

Respond with ONLY ONE WORD: LOW, MEDIUM, or HIGH

User Query: {query}

Complexity:

"""

# =============================================================================
# FALLBACK KEYWORDS (used if LLM judge fails)
# =============================================================================

COMPLEXITY_KEYWORDS = {
    "high": [
        "analyze",
        "compare",
        "evaluate",
        "critique",
        "assess",
        "synthesize",
        "justify",
        "argue",
        "prove",
        "derive",
        "detailed",
        "comprehensive",
        "in-depth",
        "thorough",
        "research",
        "investigate",
        "optimize",
        "debug",
        "architecture",
        "design pattern",
        "best practices",
    ],
    "medium": [
        "explain",
        "describe",
        "how does",
        "why does",
        "write",
        "create",
        "generate",
        "build",
        "code",
        "function",
        "calculate",
        "solve",
        "determine",
    ],
    "low": ["what is", "who is", "define", "list", "hello", "hi", "hey", "thanks"],
}


# # =============================================================================
# # POLICY CONFIGURATION (Future)
# # =============================================================================

# DEFAULT_QUOTAS = {
#     "undergraduate": {
#         "daily_requests": 100,
#         "monthly_cost": 10.00,
#         "allowed_tiers": ["local", "lakeshore"],
#     },
#     "graduate": {
#         "daily_requests": 500,
#         "monthly_cost": 50.00,
#         "allowed_tiers": ["local", "lakeshore", "cloud"],
#     },
#     "faculty": {
#         "daily_requests": 1000,
#         "monthly_cost": 200.00,
#         "allowed_tiers": ["local", "lakeshore", "cloud"],
#     },
# }
