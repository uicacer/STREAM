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

# Maximum payload size for Globus Compute tasks (in bytes).
#
# Globus Compute enforces a 10 MB limit on task submissions:
#   "The current data limit is set to 10MB on task submissions, which
#    applies to both individual functions as well as batch submissions."
# Reference: https://globus-compute.readthedocs.io/en/stable/limits.html
#
# We set our internal limit to 8 MB to leave headroom for:
#   - Function bytecode serialization overhead (~50-100 KB)
#   - dill serialization framing (~10-20 KB)
#   - Safety margin for edge cases
#
# This primarily matters for multimodal queries: a single base64-encoded
# image can be 1-5 MB. The frontend compresses images (max 1024px, JPEG 85%)
# to keep them under ~500 KB, but multiple images can still exceed the limit.
GLOBUS_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

# Maximum total image data per message for Lakeshore (in bytes).
# Within the 8 MB total payload budget, we reserve ~2 MB for conversation
# text, function serialization, and framing. This leaves 6 MB for images
# in the current user message. The frontend warns users when attached
# images exceed this threshold and suggests Local or Cloud tiers instead.
GLOBUS_MAX_IMAGE_BYTES = 6 * 1024 * 1024  # 6 MB

# =============================================================================
# WEB SEARCH (Internet Connectivity for LLM Queries)
# =============================================================================
# When enabled, STREAM searches the web for the user's query BEFORE sending
# it to the LLM. Search results are injected as a system message so the LLM
# can reference current information in its response.
#
# This is a "Retrieval-Augmented Generation" (RAG) approach using live web
# search instead of a static vector database. It works with ALL models across
# ALL tiers because it injects results as plain text context — no tool calling
# or function calling support required from the model.
#
# Two providers are supported:
#   - DuckDuckGo: Free, no API key, works out of the box (default)
#   - Tavily: AI-optimized results, requires API key, better quality

# How many search results to include in the context.
# More results = more context for the LLM, but also more tokens consumed.
# 5 results is a good balance: enough variety without overwhelming the context.
WEB_SEARCH_MAX_RESULTS = 5

# Maximum characters of extracted content to include per search result.
# DuckDuckGo returns snippets (~200 chars), but URL fetching can return
# entire pages. We truncate to avoid blowing the model's context window.
# 4000 chars ≈ 1000 tokens, so 5 results × 4000 chars ≈ 5000 tokens.
WEB_SEARCH_MAX_CONTENT_LENGTH = 4000

# Timeout for web search and URL fetch operations (in seconds).
# Web searches should be fast, but we need to account for slow networks,
# rate limiting, and complex queries. 10 seconds is generous enough to
# avoid timeout errors while not blocking the user for too long.
WEB_SEARCH_TIMEOUT = 10

USE_GLOBUS_COMPUTE = (
    os.getenv("USE_GLOBUS_COMPUTE", "true").lower() == "true"
)  # Enable Globus Compute mode
GLOBUS_COMPUTE_ENDPOINT_ID = os.getenv(
    "GLOBUS_COMPUTE_ENDPOINT_ID"
)  # Globus endpoint ID for Lakeshore
VLLM_SERVER_URL = os.getenv(
    "VLLM_SERVER_URL", "http://ga-002:8000"
)  # vLLM URL on Lakeshore (for Globus remote execution)

# =============================================================================
# WEBSOCKET RELAY (True Token Streaming from Lakeshore)
# =============================================================================
# When set, enables REAL token streaming from Lakeshore via a WebSocket relay.
# Without this, Lakeshore uses "fake streaming" — we wait for the full response
# from Globus Compute, then split it into word-by-word chunks to simulate typing.
#
# With the relay, tokens flow directly from vLLM → relay → browser as the GPU
# generates them. The relay is a lightweight forwarding server (see stream/relay/).
#
# URL formats:
#   Development (ngrok):  wss://abc123.ngrok-free.app  (or https:// — auto-converted)
#   Production:           wss://relay.your-domain.com
#
# Both the producer (Lakeshore) and consumer (your app) use the same URL.
# You can paste the ngrok HTTPS URL directly — it's auto-converted to wss://.
_raw_relay_url = os.getenv("RELAY_URL", "")
if _raw_relay_url.startswith("https://"):
    _raw_relay_url = "wss://" + _raw_relay_url[8:]
elif _raw_relay_url.startswith("http://"):
    _raw_relay_url = "ws://" + _raw_relay_url[7:]
RELAY_URL = _raw_relay_url

# =============================================================================
# LAKESHORE MODELS
# =============================================================================
# Each Lakeshore model runs as a separate vLLM instance on a different port.
# The Globus Compute client uses this mapping to route to the correct vLLM URL.
#
# GPU: Each model gets a 3g.40gb MIG slice (39.5 GiB usable VRAM) on Lakeshore.
#
# CURRENT DEMO CONFIG: Using Qwen 1.5B for fast responses.
# The Globus Compute round-trip adds ~5s overhead, so a small model (~100+ tok/s)
# gives a much better demo experience than 32B models (~15 tok/s).
# Only 1 SLURM job allowed per user (QOSMaxGRESPerUser), so all model keys
# point to the same 1.5B instance on port 8000.
#
# PRODUCTION CONFIG (commented out below): 32B AWQ models, each on its own port.
# To switch, uncomment the production entries, comment out the demo entries,
# and get the HPC admin to increase the per-user GPU limit.
# vLLM flags for 32B: --enforce-eager --max-model-len 16384 --quantization awq
# See scripts/vllm-*-32b.sh for SLURM launch scripts.
#
# hf_name: The HuggingFace model ID that vLLM loads. This MUST match the model
# name passed to `vllm serve` in the SLURM script, because vLLM's OpenAI-
# compatible API uses this as the model identifier in chat completion requests.
LAKESHORE_MODELS = {
    # --- 1.5B models (fast demo) ---
    # Each model runs as a separate vLLM instance on its own port.
    # See scripts/vllm-*-1.5b.sh for the SLURM launch scripts.
    # Only 1 SLURM job allowed per user (QOSMaxGRESPerUser), so for the demo
    # only one model will actually be running. The others will show as unavailable.
    #
    # Port assignments (matching SLURM scripts):
    #   8000 = Qwen 2.5 1.5B (general purpose)
    #   8001 = Qwen 2.5 Coder 1.5B (coding specialist)
    #   8002 = DeepSeek R1 Distill 1.5B (deep reasoning)
    #   8003 = Qwen 2.5 1.5B stand-in for QwQ (no official 1.5B QwQ exists)
    #   8004 = Qwen 2.5 32B AWQ (high quality, needs --gpu-memory-utilization 0.75)
    "lakeshore-qwen-1.5b": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "port": 8000,
        "description": "General purpose (1.5B, fast demo)",
    },
    "lakeshore-coder-1.5b": {
        "hf_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "port": 8001,
        "description": "Coding specialist (1.5B, fast demo)",
    },
    "lakeshore-deepseek-r1": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "port": 8002,
        "description": "Deep reasoning (1.5B, fast demo)",
    },
    "lakeshore-qwq": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "port": 8003,
        "description": "Reasoning (1.5B, fast demo)",
    },
    # --- H100 models (ghi2-002 in batch_gpu2 partition) ---
    # These run on the full H100 NVL GPU (96 GiB VRAM), NOT on ga-002 (A100 MIG).
    # The "host" field overrides the default VLLM_SERVER_URL host for these models.
    # Only one can run at a time (same port on same node).
    #
    # 32B FP16: Full precision, no quantization. Fast (~40-60 tok/s).
    #   See scripts/vllm-qwen-32b-fp16.sh for SLURM launch script.
    "lakeshore-qwen-32b-fp16": {
        "hf_name": "Qwen/Qwen2.5-32B-Instruct",
        "host": "ghi2-002",
        "port": 8000,
        "description": "General purpose (32B FP16, high quality)",
    },
    # 72B AWQ: Flagship quality. Requires CUDA driver 545+ for fast Marlin kernels.
    #   See scripts/vllm-qwen-72b.sh for SLURM launch script.
    "lakeshore-qwen-72b": {
        "hf_name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "host": "ghi2-002",
        "port": 8000,
        "description": "General purpose (72B AWQ, flagship quality)",
    },
    # 72B VL AWQ: Vision-Language flagship. Handles both text and image queries.
    #   See scripts/vllm-qwen-vl-72b.sh for SLURM launch script.
    "lakeshore-qwen-vl-72b": {
        "hf_name": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "host": "ghi2-002",
        "port": 8000,
        "description": "Vision + Text (72B AWQ, multimodal flagship)",
        "multimodal": True,
    },
    # --- 32B AWQ model (high quality, runs alongside 1.5B models) ---
    # Requires its own 3g.40gb MIG slice. Uses CUDA graphs (no --enforce-eager)
    # with --gpu-memory-utilization 0.75 to leave room for CUDA graphs + sampler.
    # Context limited to 8K tokens due to reduced KV cache budget.
    # See scripts/vllm-qwen-32b.sh for SLURM launch script.
    "lakeshore-qwen-32b": {
        "hf_name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "port": 8004,
        "description": "General purpose (32B AWQ, high quality)",
    },
    # --- Production config: 32B AWQ models (1 per MIG slice, 1 per port) ---
    # Requires multiple SLURM jobs or increased QOS GPU limit.
    # "lakeshore-qwen-32b": {
    #     "hf_name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    #     "port": 8000,
    #     "description": "General purpose (32B, high quality)",
    # },
    # "lakeshore-coder-32b": {
    #     "hf_name": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    #     "port": 8001,
    #     "description": "Coding specialist",
    # },
    # "lakeshore-deepseek-r1": {
    #     # Community AWQ — official BF16 is 64 GiB, won't fit on 40GB MIG
    #     "hf_name": "casperhansen/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    #     "port": 8002,
    #     "description": "Deep reasoning (R1 chain-of-thought)",
    # },
    # "lakeshore-qwq": {
    #     "hf_name": "Qwen/QwQ-32B-AWQ",
    #     "port": 8003,
    #     "description": "Reasoning (Qwen o1-style)",
    # },
    # Legacy model — kept for backwards compatibility during testing.
    "lakeshore-qwen": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "port": 8000,
        "description": "Qwen 2.5 1.5B (legacy)",
    },
}


def get_lakeshore_vllm_url(model: str) -> str:
    """Get the vLLM URL on Lakeshore for a given model name.

    Each model can run on a different node. If the model entry has a "host"
    field, use that host directly. Otherwise, fall back to the default host
    from VLLM_SERVER_URL.

    Examples:
        lakeshore-qwen-1.5b (no host) → http://ga-002:8000  (from VLLM_SERVER_URL)
        lakeshore-qwen-72b  (host=ghi2-002) → http://ghi2-002:8000
    """
    model_info = LAKESHORE_MODELS.get(model)
    if not model_info:
        # Fall back to the default VLLM_SERVER_URL for unknown models
        return VLLM_SERVER_URL

    # If the model specifies its own host (e.g., on a different GPU node),
    # use it directly instead of the default from VLLM_SERVER_URL.
    if "host" in model_info:
        return f"http://{model_info['host']}:{model_info['port']}"

    # Otherwise, use the default host from VLLM_SERVER_URL with the model's port.
    # Extract host from VLLM_SERVER_URL (e.g., "http://ga-002:8000" → "http://ga-002")
    base_url = VLLM_SERVER_URL.rsplit(":", 1)[0]
    return f"{base_url}:{model_info['port']}"


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

# Timeout for Lakeshore per-model health checks (seconds).
# This is used when the user selects a specific Lakeshore model and we need to
# verify it's actually running by sending a 1-token inference through Globus.
#
# How the timeout works:
#   - Model NOT running: Globus submits to HPC, remote `requests.post()` gets
#     ConnectionRefused immediately → result comes back in ~5-6s (Globus round-trip).
#   - Model running & fast (1.5B): 1-token generation is <1s → result in ~5-7s.
#   - Model running but slow (32B): 1-token might take 5-10s → result in ~10-15s.
#   - Globus/network issues: times out at this limit.
#
# 20s is long enough for even slow models, short enough to not block the UI.
# (The full inference timeout is 240s — way too long for a health check.)
LAKESHORE_HEALTH_TIMEOUT = int(os.getenv("LAKESHORE_HEALTH_TIMEOUT", "20"))

# =============================================================================
# JUDGE CONFIGURATION
# =============================================================================

# Judge strategy options (user can select in UI)
#
# These control how STREAM classifies query complexity (LOW/MEDIUM/HIGH).
# The judge runs BEFORE the main inference to decide which tier to use.
#
# NOTE: "ollama-1b" was removed because we removed llama3.2:1b from local
# models to save disk space. The 3B model provides better accuracy anyway.
JUDGE_STRATEGIES = {
    "ollama-3b": {
        "model": "local-llama",
        "name": "Ollama 3b",
        "description": "Balanced accuracy, free",
        "icon": "🎯",
        "timeout": 60,
    },
    "gemma-vision": {
        "model": "local-vision",
        "name": "Gemma Vision 4B",
        "description": "Vision-capable judge, can analyze images, free",
        "icon": "👁️",
        "timeout": 60,
        # This flag tells the complexity judge to pass full multimodal
        # content (including images) to the judge model, instead of
        # extracting text only. Useful when the image itself affects
        # complexity (e.g., a simple photo vs. a complex medical scan).
        "vision": True,
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

# Legacy config (for backwards compatibility with older code paths)
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
#
# STREAM supports two methods for accessing cloud models:
#
# 1. OPENROUTER (aggregator — recommended for most users):
#    - One API key gives access to 500+ models from all major providers.
#    - User enters their own key in the STREAM settings panel.
#    - The key is stored in the browser (localStorage), never on the server.
#    - LiteLLM natively supports the `openrouter/` prefix for routing.
#    - Free models available (rate-limited, no payment required).
#
# 2. DIRECT PROVIDER KEYS (advanced users):
#    - User enters their own Anthropic or OpenAI API key.
#    - Bypasses the aggregator for potentially lower latency.
#    - Each provider billed separately.
#
# KEY SOURCE:
# -----------
# The "key_source" field tells the system where to find the API key:
#   - "user": Key comes from the user's browser (sent in each request body).
#             Used for OpenRouter and user-provided direct keys.
#   - "env":  Key comes from server environment variable (legacy/admin mode).
#             Used when ANTHROPIC_API_KEY or OPENAI_API_KEY is set in .env.
#
# When a user provides their own key, it takes priority over env vars.
# This allows STREAM to work in both personal (user keys) and shared
# (admin-managed env vars) deployment modes.
#
CLOUD_PROVIDERS = {
    # -----------------------------------------------------------------
    # OpenRouter models (one API key for all — aggregator)
    # -----------------------------------------------------------------
    # These use the `openrouter/` prefix in LiteLLM, which routes to
    # https://openrouter.ai/api/v1 automatically. The user provides
    # a single OPENROUTER_API_KEY that unlocks all models.
    #
    # CURATED FRONTIER SET — the best 1-2 models from each major provider.
    # Updated to reflect actual frontier models, not older/cheaper variants.
    "cloud-or-claude": {
        "name": "Claude Sonnet 4",
        "provider": "OpenRouter",
        "description": "Anthropic's best balance of capability, speed, and cost",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["multimodal", "reasoning"],
    },
    "cloud-or-gpt4o": {
        "name": "GPT-4o",
        "provider": "OpenRouter",
        "description": "OpenAI's flagship multimodal model",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["multimodal"],
    },
    "cloud-or-gemini-pro": {
        "name": "Gemini 2.5 Pro",
        "provider": "OpenRouter",
        "description": "Google's most capable model with 1M context",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["multimodal", "reasoning"],
    },
    "cloud-or-gemini-flash": {
        "name": "Gemini 2.5 Flash",
        "provider": "OpenRouter",
        "description": "Google's fast model — 1M context, very low cost",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["multimodal"],
    },
    "cloud-or-o3-mini": {
        "name": "o3-mini",
        "provider": "OpenRouter",
        "description": "OpenAI's reasoning specialist — great for math & code",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["reasoning"],
    },
    "cloud-or-deepseek-r1": {
        "name": "DeepSeek R1",
        "provider": "OpenRouter",
        "description": "Top reasoning model at a fraction of the cost",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["reasoning"],
    },
    "cloud-or-llama-maverick": {
        "name": "Llama 4 Maverick",
        "provider": "OpenRouter",
        "description": "Meta's best open-source model — 1M context, multimodal",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["multimodal"],
    },
    "cloud-or-deepseek-v3": {
        "name": "DeepSeek V3",
        "provider": "OpenRouter",
        "description": "Powerful and extremely affordable general-purpose model",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "budget",
        "tags": [],
    },
    "cloud-or-glm5": {
        "name": "GLM-5",
        "provider": "OpenRouter",
        "description": "Z.ai's frontier model — near-Opus capability at a fraction of the cost",
        "env_key": "OPENROUTER_API_KEY",
        "key_source": "user",
        "category": "recommended",
        "tags": ["reasoning"],
    },
    # -----------------------------------------------------------------
    # Direct provider models (user provides their own provider key)
    # -----------------------------------------------------------------
    # These call the provider API directly (no aggregator hop).
    # Lower latency than OpenRouter but requires separate keys.
    "cloud-claude": {
        "name": "Claude Sonnet 4",
        "provider": "Anthropic",
        "description": "Direct Anthropic API — lowest latency",
        "env_key": "ANTHROPIC_API_KEY",
        "key_source": "user",
        "category": "direct",
    },
    "cloud-gpt": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "description": "Direct OpenAI API — lowest latency",
        "env_key": "OPENAI_API_KEY",
        "key_source": "user",
        "category": "direct",
    },
    "cloud-gpt-cheap": {
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "description": "Direct OpenAI API — fast and affordable",
        "env_key": "OPENAI_API_KEY",
        "key_source": "user",
        "category": "direct",
    },
}

# Map cloud provider IDs to the env var key name that holds the API key.
# Used when the user provides keys via the request body — we need to know
# which user-provided key field corresponds to which provider.
CLOUD_PROVIDER_KEY_MAPPING = {
    "OPENROUTER_API_KEY": "openrouter_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "OPENAI_API_KEY": "openai_api_key",
}

# Default cloud provider (can be overridden by user in settings)
DEFAULT_CLOUD_PROVIDER = os.getenv("DEFAULT_CLOUD_PROVIDER", "cloud-claude")

DEFAULT_MODELS = {
    "local": "local-llama",
    "lakeshore": "lakeshore-qwen-1.5b",
    "cloud": DEFAULT_CLOUD_PROVIDER,  # Now configurable!
}

# Default vision models for each tier.
# When the router detects images and needs to pick a vision-capable model,
# it uses this mapping to find the right model for the selected tier.
# This is only used when the user selected AUTO or a tier without
# specifying a model — if the user explicitly chose a model, we respect it.
DEFAULT_VISION_MODELS = {
    "local": "local-vision",
    "lakeshore": "lakeshore-qwen-vl-72b",
    "cloud": DEFAULT_CLOUD_PROVIDER,  # All cloud models support vision
}


# =============================================================================
# OLLAMA MODELS
# =============================================================================

OLLAMA_MODELS = {
    # Text-only model: good for general queries without images.
    # Llama 3.2 3B is a balanced choice — fast enough for local inference,
    # capable enough for most text tasks, and fits comfortably in ~4 GB RAM.
    "local-llama": "llama3.2:3b",
    # Vision model: handles both text AND image queries.
    # Gemma 3 4B is Google's open-source multimodal model based on the
    # Gemini 2.0 architecture. It can process images (describe, analyze,
    # extract text, etc.) while still being small enough for local inference.
    # Uses ~6 GB RAM, fits within Docker Ollama's 8 GB allocation.
    "local-vision": "gemma3:4b",
}


# =============================================================================
# VISION-CAPABLE MODELS
# =============================================================================
# This set tells the router which models can process images.
# When a user sends an image, the router uses this to:
#   1. AUTO mode: pick a vision-capable model automatically
#   2. Explicit model: reject text-only models with a helpful error
#
# If you add a new vision model (local, lakeshore, or cloud), add it here.
VISION_CAPABLE_MODELS = {
    # Local: Gemma 3 4B (multimodal, handles text + images)
    "local-vision",
    # Lakeshore: Qwen2.5-VL-72B (vision-language model on H100)
    "lakeshore-qwen-vl-72b",
    # Cloud (direct): Claude Sonnet 4, GPT-4o, and GPT-4o Mini all support vision
    "cloud-claude",
    "cloud-gpt",
    "cloud-gpt-cheap",
    # Cloud (OpenRouter): frontier models with vision/multimodal support
    "cloud-or-claude",
    "cloud-or-gpt4o",
    "cloud-or-gemini-pro",
    "cloud-or-gemini-flash",
    "cloud-or-llama-maverick",
}
# Note: Dynamically-selected OpenRouter models may also support vision.
# The model catalog API includes modality info, and the frontend marks
# vision-capable models. For static routing, only the curated set above
# is checked. Dynamic models are assumed vision-capable if the user
# explicitly selected them for an image query.

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
    # Llama 3.2:3b supports 128K context natively. 32K is a practical limit
    # for desktop — large enough for extended conversations, small enough for
    # fast Apple Silicon GPU inference. (~2GB model leaves plenty of VRAM.)
    "local-llama": {"total": 32768, "reserve_output": 2048},
    # Uncomment below to test context-limit-exceeded error dialog:
    # "local-llama": {"total": 500, "reserve_output": 100},
    # Gemma 3 4B supports 128K context, but images consume significant context.
    # Each image uses ~765 tokens, so we use 32K as a practical limit.
    "local-vision": {"total": 32768, "reserve_output": 2048},
    # Lakeshore: 32K total context (vLLM --max-model-len=32768).
    # Demo uses 1.5B model which fits easily with 32K context on 40GB MIG.
    # For 32B production models, reduce to 16384 (--enforce-eager needed, less VRAM).
    "lakeshore-qwen-1.5b": {"total": 32768, "reserve_output": 2048},
    "lakeshore-coder-1.5b": {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwen-32b-fp16": {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwen-72b": {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwen-32b": {"total": 8192, "reserve_output": 1024},
    "lakeshore-deepseek-r1": {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwq": {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwen-vl-72b": {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwen": {"total": 32768, "reserve_output": 2048},  # Legacy
    # Cloud (direct): Full native context limits
    # max_input = 200000 - 4000 = 196000 tokens (~780KB of text)
    "cloud-claude": {"total": 200000, "reserve_output": 4000},
    "cloud-haiku": {"total": 200000, "reserve_output": 4000},
    "cloud-gpt": {"total": 128000, "reserve_output": 4000},
    "cloud-gpt-cheap": {"total": 128000, "reserve_output": 4000},
    # Cloud (OpenRouter): context limits match the underlying provider models.
    # OpenRouter passes through to the actual provider without adding limits.
    "cloud-or-claude": {"total": 200000, "reserve_output": 4000},
    "cloud-or-gpt4o": {"total": 128000, "reserve_output": 4000},
    "cloud-or-gemini-pro": {"total": 1000000, "reserve_output": 8000},
    "cloud-or-gemini-flash": {"total": 1000000, "reserve_output": 8000},
    "cloud-or-o3-mini": {"total": 200000, "reserve_output": 4000},
    "cloud-or-deepseek-r1": {"total": 64000, "reserve_output": 4000},
    "cloud-or-llama-maverick": {"total": 1000000, "reserve_output": 8000},
    "cloud-or-deepseek-v3": {"total": 128000, "reserve_output": 4000},
    "cloud-or-glm5": {"total": 200000, "reserve_output": 4000},
}

# Default context limits for dynamically-selected OpenRouter models.
# When a user picks a model from the catalog that isn't in the static
# list above, we use these safe defaults. The catalog API provides the
# actual context_length, and the frontend can pass it — but as a safety
# net, we default to 128K (the most common limit for modern models).
DEFAULT_CLOUD_CONTEXT_LIMIT = {"total": 128000, "reserve_output": 4000}

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


# =============================================================================
# REASONING MODEL DETECTION
# =============================================================================
# Models that produce reasoning/thinking content when asked.
# Used by both desktop mode (litellm_direct.py) and server mode
# (litellm_client.py) to pass `reasoning_effort` to litellm.

REASONING_MODEL_PATTERNS = [
    "claude-sonnet-4",
    "claude-opus",
    "claude-4",
    "o1",
    "o3",
    "o4",
    "deepseek-r1",
    "deepseek/deepseek-r1",
]


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model supports extended thinking / reasoning output."""
    lower = model_name.lower()
    return any(p in lower for p in REASONING_MODEL_PATTERNS)


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
