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
LAKESHORE_VLLM_ENDPOINT = os.getenv("LAKESHORE_VLLM_ENDPOINT")

# =============================================================================
# HEALTH CHECKS
# =============================================================================

HEALTH_CHECK_TTL = 3600  # Recheck every hour
HEALTH_CHECK_TIMEOUT = 5.0

# =============================================================================
# JUDGE CONFIGURATION
# =============================================================================

JUDGE_MODEL = "local-llama"
JUDGE_TIMEOUT = 60
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

DEFAULT_MODELS = {"local": "local-llama", "lakeshore": "lakeshore-llama", "cloud": "cloud-claude"}

# =============================================================================
# MODEL COSTS (per token)
# =============================================================================

MODEL_COSTS = {
    "local-llama-tiny": {"input": 0.0, "output": 0.0},
    "local-llama": {"input": 0.0, "output": 0.0},
    "local-llama-quality": {"input": 0.0, "output": 0.0},
    "cloud-claude": {"input": 0.000003, "output": 0.000015},
    "cloud-gpt": {"input": 0.00001, "output": 0.00003},
    "cloud-gpt-cheap": {"input": 0.0000005, "output": 0.0000015},
    "lakeshore-llama": {"input": 0.0000005, "output": 0.0000005},
}

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
    "local-llama-tiny": {"total": 2048, "reserve_output": 300},
    "local-llama": {"total": 2048, "reserve_output": 300},
    "local-llama-quality": {"total": 8192, "reserve_output": 500},
    "lakeshore-llama": {"total": 8192, "reserve_output": 500},
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
