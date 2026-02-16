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
# MODE DETECTION
# =============================================================================
# "server" = Docker/cloud deployment (default, backwards-compatible)
# "desktop" = Native desktop app (PyWebView, SQLite, direct litellm calls)
STREAM_MODE = os.getenv("STREAM_MODE", "server")

# =============================================================================
# SERVICE METADATA
# =============================================================================

SERVICE_NAME = "STREAM Middleware"
SERVICE_VERSION = "1.0.0"
SERVICE_DESCRIPTION = "Smart Tiered Routing Engine for AI Models"

# =============================================================================
# SERVICE CONFIGURATION
# =============================================================================

MIDDLEWARE_HOST = os.getenv("MIDDLEWARE_HOST", "127.0.0.1")
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT", "5000"))
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

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5000").split(",")
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# =============================================================================
# EXTERNAL SERVICES
# =============================================================================

# In Docker, OLLAMA_HOST="ollama" (Docker DNS name for the Ollama container).
# Outside Docker (desktop mode), Ollama runs natively on localhost.
# Default "localhost" works for desktop; .env overrides it to "ollama" for Docker.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
# Centralized URL so we don't hardcode "http://ollama:11434" in multiple files.
# tier_health.py and warm_ping.py use this instead of building their own URLs.
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000")
LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY", "")

# Lakeshore connection configuration
# Two modes: SSH port forwarding (legacy) or Globus Compute (preferred)
LAKESHORE_VLLM_ENDPOINT = os.getenv(
    "LAKESHORE_VLLM_ENDPOINT"
)  # SSH port forward URL (e.g., http://host.docker.internal:8000)

# Lakeshore proxy service configuration (configurable host and port)
# In Docker mode, the proxy runs as a separate container on port 8001.
# In desktop mode, the proxy routes are mounted at /lakeshore on the main app,
# so the URL includes a path prefix instead of a different port.
_default_proxy_host = "127.0.0.1" if STREAM_MODE == "desktop" else "lakeshore-proxy"
LAKESHORE_PROXY_HOST = os.getenv("LAKESHORE_PROXY_HOST", _default_proxy_host)
LAKESHORE_PROXY_PORT = int(os.getenv("LAKESHORE_PROXY_PORT", "8001"))
# Allow full URL override via env var. Desktop mode sets this to
# "http://127.0.0.1:5000/lakeshore" so requests go to the embedded router.
LAKESHORE_PROXY_URL = os.getenv(
    "LAKESHORE_PROXY_URL",
    f"http://{LAKESHORE_PROXY_HOST}:{LAKESHORE_PROXY_PORT}",
)

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

# TTL = "Time To Live" - how long cached data is considered fresh/valid.
# After TTL expires, the next request triggers a fresh health check.
#
# We use two different TTLs because internal routing and frontend display
# have different freshness requirements:
#
# HEALTH_CHECK_TTL (6 min): Used internally when routing requests to tiers.
#   - Longer TTL reduces server load from many concurrent API requests
#   - Background monitor refreshes status every 5 minutes anyway
#   - Stale data is acceptable here since routing has fallback logic
#
# QUICK_CHECK_TTL (30 sec): Used by frontend polling to show tier status dots.
#   - Matches the frontend poll interval (30 seconds)
#   - Users expect to see tier changes reflected quickly in the UI
#   - Quick checks are lightweight (single attempt, short timeout)
#
HEALTH_CHECK_TTL = 360  # 6 minutes - for internal routing decisions
QUICK_CHECK_TTL = 30  # 30 seconds - for frontend status display
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

# =============================================================================
# CLOUD PROVIDERS
# =============================================================================
# Available cloud providers that users can choose from.
# Each provider maps to a model_name in litellm_config.yaml
#
# Users can switch providers in settings if:
# - Their current provider's subscription expired
# - They prefer a different model
# - One provider is having issues
#
CLOUD_PROVIDERS = {
    "cloud-claude": {
        "name": "Claude Sonnet 4",
        "provider": "Anthropic",
        "description": "Best for complex reasoning and coding",
        "env_key": "ANTHROPIC_API_KEY",  # Required env var
    },
    "cloud-gpt": {
        "name": "GPT-4 Turbo",
        "provider": "OpenAI",
        "description": "Strong general-purpose model",
        "env_key": "OPENAI_API_KEY",
    },
    "cloud-gpt-cheap": {
        "name": "GPT-3.5 Turbo",
        "provider": "OpenAI",
        "description": "Fast and affordable",
        "env_key": "OPENAI_API_KEY",
    },
}

# Default cloud provider (can be overridden by user in settings)
DEFAULT_CLOUD_PROVIDER = os.getenv("DEFAULT_CLOUD_PROVIDER", "cloud-claude")

DEFAULT_MODELS = {
    "local": "local-llama",
    "lakeshore": "lakeshore-qwen",
    "cloud": DEFAULT_CLOUD_PROVIDER,  # Now configurable!
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
#
# How context windows work:
# -------------------------
# LLMs have a fixed "context window" - the total number of tokens they can
# process in a single request. This window is SHARED between:
#
#   INPUT (what you send)     +    OUTPUT (what the model generates)
#   ─────────────────────          ───────────────────────────────────
#   • System prompt                • The model's response
#   • Conversation history         • Can be cut off mid-sentence if
#   • User's current message         no room left!
#
# Example with 4096-token model:
#   ┌─────────────────────────────────────────────────────────────┐
#   │                    4096 token context window                │
#   ├───────────────────────────────────┬─────────────────────────┤
#   │  INPUT: 3584 tokens (max)         │  OUTPUT: 512 reserved   │
#   │  (conversation + query)           │  (model's response)     │
#   └───────────────────────────────────┴─────────────────────────┘
#
# Why reserve_output is needed:
# -----------------------------
# If we send 4000 tokens to a 4096-token model, it only has 96 tokens
# left to respond - that's about 2 sentences! The response would be
# truncated mid-thought.
#
# By reserving tokens for output, we ensure:
#   max_input_tokens = total - reserve_output
#
# This is calculated in context_window.py:get_max_input_tokens()
#
# reserve_output guidelines:
# --------------------------
# • 512 tokens  ≈ 1-2 paragraphs (good for simple Q&A)
# • 1000 tokens ≈ half a page
# • 2048 tokens ≈ 1 page (good for explanations)
# • 4000 tokens ≈ 2 pages (good for detailed responses)
#
MODEL_CONTEXT_LIMITS = {
    # Local: 4K limit for faster CPU inference
    # max_input = 4096 - 512 = 3584 tokens (~14KB of text)
    "local-llama-tiny": {"total": 4096, "reserve_output": 512},
    # Llama 3.2:3b supports 128K context natively. 32K is a practical limit
    # for desktop — large enough for extended conversations, small enough for
    # fast Apple Silicon GPU inference. (~2GB model leaves plenty of VRAM.)
    "local-llama": {"total": 32768, "reserve_output": 2048},
    # Uncomment below to test context-limit-exceeded error dialog:
    # "local-llama": {"total": 500, "reserve_output": 100},
    "local-llama-quality": {"total": 4096, "reserve_output": 512},
    # Lakeshore: 32K (runs on campus GPU, Qwen supports 32K natively)
    # max_input = 32768 - 2048 = 30720 tokens (~120KB of text)
    "lakeshore-qwen": {"total": 32768, "reserve_output": 2048},
    # Cloud: Full native context limits
    # max_input = 200000 - 4000 = 196000 tokens (~780KB of text)
    "cloud-claude": {"total": 200000, "reserve_output": 4000},
    "cloud-gpt": {"total": 128000, "reserve_output": 4000},
    "cloud-gpt-cheap": {"total": 16385, "reserve_output": 1000},
}

# =============================================================================
# TIMEOUT WARNINGS
# =============================================================================
# Warn users when response takes too long (in seconds)
# These thresholds trigger a warning message in the UI
TIER_TIMEOUT_WARNING = {
    "local": 30,  # Warn after 30s (CPU inference is slow)
    "lakeshore": 60,  # Warn after 60s (HPC queue/network delays)
    "cloud": 15,  # Warn after 15s (should be fast)
}

# =============================================================================
# JUDGE PROMPT
# =============================================================================

JUDGE_PROMPT = """
You are a query complexity classifier for an AI routing system used by students and researchers across ALL fields (science, engineering, humanities, business, healthcare, etc.).

Classification Guidelines:

LOW complexity (simple, factual - route to local):
- Greetings and thanks (hi, hello, thank you)
- Simple definitions: "What is photosynthesis?", "Define GDP", "What is Python?"
- Single factual lookups: "Who invented the telephone?", "What year did X happen?"
- Yes/no questions with obvious answers
- One-word or very short answers expected
- No reasoning or explanation needed

MEDIUM complexity (explanations, moderate analysis - route to campus GPU):
- "Explain how X works" (single concept)
- Compare 2-3 things: "Compare Python and JavaScript"
- Step-by-step instructions or tutorials
- Basic calculations or problem-solving
- Summarize a concept or article
- Write a single function or short code snippet
- Moderate technical questions with straightforward answers

HIGH complexity (deep analysis, design, research - route to cloud):
- System design or architecture (any domain: software, business, scientific)
- Multi-factor analysis or trade-off evaluation
- Research-level questions requiring domain expertise
- Design patterns, frameworks, methodologies
- Security, scalability, optimization, or performance considerations
- Multi-step reasoning across multiple concepts or domains
- Policy analysis, strategic planning, decision frameworks
- Scientific experiment design or research methodology
- Complex debugging, troubleshooting, or root cause analysis
- Creative works requiring extensive planning (essays, stories, reports)
- Anything requiring synthesis of multiple concepts or domains
- Questions with "design", "architect", "analyze trade-offs", "evaluate", "comprehensive"

Respond with ONLY ONE WORD: LOW, MEDIUM, or HIGH

User Query: {query}

Complexity:
"""

# =============================================================================
# FALLBACK KEYWORDS (used if LLM judge fails)
# =============================================================================

COMPLEXITY_KEYWORDS = {
    "high": [
        # Analysis & evaluation
        "analyze",
        "evaluate",
        "critique",
        "assess",
        "synthesize",
        "trade-off",
        "trade off",
        # Design & architecture
        "design",
        "architect",
        "architecture",
        "framework",
        "methodology",
        "strategy",
        "scalability",
        "microservices",
        "distributed",
        # Research & depth
        "research",
        "investigate",
        "comprehensive",
        "in-depth",
        "thorough",
        "detailed analysis",
        # Technical complexity
        "optimize",
        "performance",
        "security",
        "debug",
        "troubleshoot",
        "root cause",
        # Multi-domain
        "policy analysis",
        "strategic planning",
        "experiment design",
        "real-time",
        "conflict resolution",
        "version control",
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
        "compare",
        "summarize",
        "tutorial",
        "step by step",
    ],
    "low": ["what is", "who is", "define", "list", "hello", "hi", "hey", "thanks", "thank you"],
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
