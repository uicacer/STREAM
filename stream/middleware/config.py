# =============================================================================
# STREAM Middleware - Configuration (LLM-AS-A-JUDGE ROUTING)
# =============================================================================
# Intelligent LLM-based complexity detection with keyword fallback
# =============================================================================

import hashlib
import os
import time
from datetime import datetime, timedelta
from importlib.resources import files

import httpx
import yaml
from dotenv import load_dotenv

LITELLM_CONFIG = files("stream.gateway").joinpath("litellm_config.yaml")

load_dotenv()

# =============================================================================
# SERVICE CONFIGURATION
# =============================================================================

MIDDLEWARE_HOST = os.getenv("MIDDLEWARE_HOST", "0.0.0.0")
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT", "5000"))

SERVICE_NAME = "STREAM Middleware"
SERVICE_VERSION = "1.0"
SERVICE_DESCRIPTION = "AI Middleware Hub - LLM Judge Routing + Policy + Telemetry"

# Health check cache (recheck every 3600 seconds; 1 hour)
HEALTH_CHECK_TTL = 3600  # seconds

# Used in health checks to avoid hanging in middleware/routes/health.py
HEALTH_CHECK_TIMEOUT = 2.0  # seconds

# =============================================================================
# LITELLM GATEWAY SETTINGS
# =============================================================================

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY")
LAKESHORE_VLLM_ENDPOINT = os.getenv("LAKESHORE_VLLM_ENDPOINT")


# =============================================================================
# JUDGE LLM CONFIGURATION
# =============================================================================

# Which model to use as the judge (must be fast and local)
JUDGE_MODEL = "local-llama"  # Llama 3.2 1B - perfect for classification
# Other options could be "local-llama" (3B) or "cloud-gpt-cheap" if local not available

# Timeout for judge (fallback to keywords if exceeded)
JUDGE_TIMEOUT = 60  # 60 seconds max

# Enable/disable LLM judge (can turn off for debugging)
LLM_JUDGE_ENABLED = True

# Cache judge decisions for this many seconds (avoid repeat judgments)
# TTL stands for Time To Live. It refers to how long a cached judge
# decision will remain valid before it expires.
JUDGE_CACHE_TTL = 3600  # 1 hour

# =============================================================================
# ROUTING CONFIGURATION
# =============================================================================

# Tier definitions
TIERS = {
    "local": {"name": "Local Ollama", "description": "Free local inference"},
    "lakeshore": {"name": "Campus vLLM", "description": "UIC Lakeshore GPU cluster"},
    "cloud": {"name": "Cloud APIs", "description": "Claude, GPT, etc."},
}

# Default models per tier
DEFAULT_MODELS = {"local": "local-llama", "lakeshore": "lakeshore-llama", "cloud": "cloud-claude"}


# =============================================================================
# MODEL NAME MAPPING
# =============================================================================

# Model costs (per token) - UPDATE QUARTERLY
MODEL_COSTS = {
    "local-llama-tiny": {"input": 0.0, "output": 0.0},
    "local-llama": {"input": 0.0, "output": 0.0},
    "local-llama-quality": {"input": 0.0, "output": 0.0},
    "cloud-claude": {"input": 0.000003, "output": 0.000015},  # $3/$15 per 1M tokens
    "cloud-gpt": {"input": 0.00001, "output": 0.00003},  # $10/$30 per 1M tokens
    "cloud-gpt-cheap": {"input": 0.0000005, "output": 0.0000015},  # $0.50/$1.50 per 1M tokens
    "lakeshore-llama": {
        "input": 0.0000005,
        "output": 0.0000005,
    },  # $0.50/$0.50 per 1M tokens (campus rate)
}


# =============================================================================
# OLLAMA MODEL DEFINITIONS (Single Source of Truth). However, make sure it is
# consistent with the models defined in gateway/litellm_config.yaml
# =============================================================================

OLLAMA_MODELS = {
    "local-llama-tiny": "llama3.2:1b",
    "local-llama": "llama3.2:3b",
    "local-llama-quality": "llama3.1:8b",
}


# =============================================================================
# CONTEXT WINDOW LIMITS
# =============================================================================

# Model context limits (reserve space for output)
# Used in middleware/routes/chat.py and backend/src/core/chat_handler.py
# reserve_output is the number of tokens to reserve for the model's response
MODEL_CONTEXT_LIMITS = {
    "lakeshore-llama": {"total": 8192, "reserve_output": 500},
    "local-llama": {"total": 2048, "reserve_output": 300},  # ← Ollama default
    "local-llama-tiny": {"total": 2048, "reserve_output": 300},
    "local-llama-quality": {"total": 8192, "reserve_output": 500},
}


# Helper to get max input tokens for a model
def get_max_input_tokens(model: str) -> int:
    """Get maximum input tokens allowed for a model"""
    config = MODEL_CONTEXT_LIMITS.get(model)
    if config:
        return config["total"] - config["reserve_output"]
    return 200000  # Default: assume large context (cloud)


# Helper to get limits by tier
def get_tier_context_limits() -> dict:
    """Get context limits organized by tier for frontend"""
    return {
        "local": get_max_input_tokens("local-llama"),
        "lakeshore": get_max_input_tokens("lakeshore-llama"),
        "cloud": get_max_input_tokens("cloud-claude"),
    }


# =============================================================================
# COMPREHENSIVE HEALTH CHECK SYSTEM
# =============================================================================

# Track health status
_tier_health = {
    "local": {"available": False, "last_check": None, "error": None},
    "lakeshore": {"available": False, "last_check": None, "error": None},
    "cloud": {"available": False, "last_check": None, "error": None},
}


def check_tier_health(tier: str) -> tuple[bool, str | None]:
    """Check if a specific tier is available"""
    model = DEFAULT_MODELS.get(tier)
    if not model:
        return False, f"No model configured for tier {tier}"

    try:
        # LOCAL: Check Ollama directly
        if tier == "local":
            with httpx.Client(timeout=10.0) as client:
                response = client.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    return False, f"Ollama not responding (HTTP {response.status_code})"

                # Verify the specific model exists
                data = response.json()
                installed_models = [m["name"] for m in data.get("models", [])]

                # Get the Ollama model name for this tier
                ollama_model = OLLAMA_MODELS.get(model)
                if not ollama_model:
                    return False, f"No Ollama model mapping for {model}"

                if ollama_model not in installed_models:
                    return False, f"Model {ollama_model} not installed in Ollama"

                return True, None

        # LAKESHORE: Check vLLM
        elif tier == "lakeshore":
            if not LAKESHORE_VLLM_ENDPOINT:
                return False, "No Lakeshore endpoint configured in .env"

            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{LAKESHORE_VLLM_ENDPOINT}/v1/models",
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    return True, None
                else:
                    return False, f"vLLM not responding (HTTP {response.status_code})"

        # CLOUD: Test through LiteLLM WITH RETRY
        elif tier == "cloud":
            # Try 2 times with delays (LiteLLM might be starting)
            for attempt in range(2):
                try:
                    with httpx.Client(timeout=10.0) as client:
                        response = client.post(
                            f"{LITELLM_BASE_URL}/v1/chat/completions",
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": "test"}],
                                "max_tokens": 1,
                                "temperature": 0.0,
                            },
                            headers={
                                "Authorization": f"Bearer {LITELLM_API_KEY}",
                                "Content-Type": "application/json",
                            },
                        )

                        if response.status_code == 200:
                            data = response.json()
                            actual_model = data.get("model", "").lower()
                            if "claude" in actual_model or "gpt" in actual_model:
                                return True, None
                            else:
                                return False, f"Unexpected model: {actual_model}"
                        else:
                            return False, f"HTTP {response.status_code}"

                except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    if attempt < 2:  # Not last attempt
                        time.sleep(2)  # Wait 2 seconds before retry
                        continue
                    else:
                        return False, f"Connection failed after 2 attempts: {str(e)}"

        return False, "Unknown tier"

    except httpx.TimeoutException:
        return False, "Timeout"
    except httpx.ConnectError:
        return False, "Connection refused"
    except Exception as e:
        return False, str(e)


def update_tier_health(tier: str):
    """Update health status for a tier"""
    is_available, error = check_tier_health(tier)

    _tier_health[tier] = {"available": is_available, "last_check": datetime.now(), "error": error}

    status = "✅" if is_available else "❌"
    model = DEFAULT_MODELS.get(tier, "unknown")

    if is_available:
        print(f"{status} {tier.upper()} ({model}) is available")
    else:
        print(f"{status} {tier.upper()} ({model}) is UNAVAILABLE: {error}")


def is_tier_available(tier: str) -> bool:
    """Check if tier is available (with caching)"""
    status = _tier_health.get(tier)

    if (
        status is None
        or status["last_check"] is None
        or datetime.now() - status["last_check"] > timedelta(seconds=HEALTH_CHECK_TTL)
    ):
        update_tier_health(tier)
        status = _tier_health.get(tier)

    return status["available"]


def get_available_tiers() -> list[str]:
    """Get list of currently available tiers"""
    return [tier for tier in ["local", "lakeshore", "cloud"] if is_tier_available(tier)]


def check_all_tiers():
    """Check health of all tiers (run on startup)"""
    print("\n🔍 Checking health of all AI tiers...")
    print("=" * 60)

    for tier in ["local", "lakeshore", "cloud"]:
        update_tier_health(tier)

    print("=" * 60)

    available = get_available_tiers()
    if not available:
        print("❌ WARNING: NO AI TIERS ARE AVAILABLE!")
        print("   Check that Docker services are running:")
        print("   - Ollama (local)")
        print("   - LiteLLM gateway")
        print("   - Cloud API keys configured")
    else:
        print(f"✅ {len(available)}/3 tiers available: {', '.join(available).upper()}")
    print()


def get_tier_with_fallback(preferred_tier: str, complexity: str) -> tuple[str, str]:
    """Get tier with intelligent fallback"""
    # Define fallback chain based on complexity
    if complexity == "low":
        fallback_chain = ["local", "lakeshore", "cloud"]
    elif complexity == "medium":
        fallback_chain = ["lakeshore", "cloud", "local"]
    else:  # high
        fallback_chain = ["cloud", "lakeshore", "local"]

    # Ensure preferred tier is first
    if preferred_tier in fallback_chain:
        fallback_chain.remove(preferred_tier)
        fallback_chain.insert(0, preferred_tier)

    # Try each tier in order
    for tier in fallback_chain:
        if is_tier_available(tier):
            if tier == preferred_tier:
                return tier, f"{complexity.upper()} → {tier.upper()}"
            else:
                return (
                    tier,
                    f"{complexity.upper()} → {preferred_tier.upper()} unavailable, using {tier.upper()}",
                )

    # No tiers available!
    return None, "All AI services unavailable"


# =============================================================================
# LLM JUDGE PROMPT
# =============================================================================

JUDGE_PROMPT = """You are a query complexity classifier. Analyze the following user query and classify its complexity level.

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

Complexity:"""

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

# =============================================================================
# JUDGE CACHE (in-memory, simple dict)
# =============================================================================

_judge_cache = {}


def _get_cache_key(query: str) -> str:
    """Generate cache key from query"""
    return hashlib.md5(query.lower().encode()).hexdigest()


def _get_cached_judgment(query: str) -> str | None:
    """Get cached judgment if exists and not expired"""
    key = _get_cache_key(query)
    if key in _judge_cache:
        judgment, timestamp = _judge_cache[key]
        if time.time() - timestamp < JUDGE_CACHE_TTL:
            return judgment
        else:
            # Expired, remove
            del _judge_cache[key]
    return None


def _cache_judgment(query: str, judgment: str):
    """Cache a judgment"""
    key = _get_cache_key(query)
    _judge_cache[key] = (judgment, time.time())


# =============================================================================
# LLM JUDGE IMPLEMENTATION
# =============================================================================


def judge_complexity_with_llm(query: str) -> str | None:
    """
    Use a lightweight LLM to judge query complexity

    Args:
        query: User's question

    Returns:
        "low", "medium", "high", or None if failed
    """
    # Check cache first
    cached = _get_cached_judgment(query)
    if cached:
        print(f"🔍 JUDGE: Using cached result → {cached.upper()}")
        return cached

    # Build judge prompt
    prompt = JUDGE_PROMPT.format(query=query)

    try:
        # Call LiteLLM with judge model
        with httpx.Client(timeout=JUDGE_TIMEOUT) as client:
            response = client.post(
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json={
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,  # Just need one word
                    "temperature": 0.0,  # Deterministic
                },
                headers={
                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )

        if response.status_code != 200:
            print(f"⚠️ JUDGE: Failed with status {response.status_code}")
            return None

        # Parse response
        data = response.json()
        judgment_text = data["choices"][0]["message"]["content"].strip().upper()

        # Extract LOW/MEDIUM/HIGH from response
        if "LOW" in judgment_text:
            judgment = "low"
        elif "MEDIUM" in judgment_text:
            judgment = "medium"
        elif "HIGH" in judgment_text:
            judgment = "high"
        else:
            print(f"⚠️ JUDGE: Unexpected response: {judgment_text}")
            return None

        # Cache the judgment
        _cache_judgment(query, judgment)

        print(f"🔍 JUDGE: LLM classified as → {judgment.upper()}")
        return judgment

    except httpx.TimeoutException:
        print(f"⚠️ JUDGE: Timeout after {JUDGE_TIMEOUT}s")
        return None

    except Exception as e:
        print(f"⚠️ JUDGE: Error: {str(e)}")
        return None


def judge_complexity_with_keywords(query: str) -> str:
    """
    Fallback: Use keyword matching to judge complexity

    Args:
        query: User's question

    Returns:
        "low", "medium", or "high"
    """
    query_lower = query.lower()

    # Check high complexity keywords
    if any(kw in query_lower for kw in COMPLEXITY_KEYWORDS["high"]):
        return "high"

    # Check medium complexity keywords
    if any(kw in query_lower for kw in COMPLEXITY_KEYWORDS["medium"]):
        return "medium"

    # Check low complexity keywords
    if any(kw in query_lower for kw in COMPLEXITY_KEYWORDS["low"]):
        return "low"

    # Default: medium (safer to overestimate than underestimate)
    return "medium"


def get_tier_for_query(query: str, user_preference: str = "auto") -> str:
    """
    Determine which tier to use based on LLM judge + keyword fallback + health checks
    """
    # If user explicitly chose a tier, respect it (but check health)
    if user_preference in ["local", "lakeshore", "cloud"]:
        if is_tier_available(user_preference):
            print(f"🔍 ROUTING: User override → {user_preference.upper()}")
            return user_preference
        else:
            print(f"⚠️  ROUTING: User selected {user_preference.upper()} but it's unavailable")
            # Continue to auto-routing with fallback

    # Try LLM judge first (if enabled)
    complexity = None
    method = "unknown"

    if LLM_JUDGE_ENABLED:
        complexity = judge_complexity_with_llm(query)
        if complexity:
            method = "LLM judge"
        else:
            print("⚠️ ROUTING: LLM judge failed, falling back to keywords")

    # Fallback to keyword-based if LLM failed or disabled
    if complexity is None:
        complexity = judge_complexity_with_keywords(query)
        method = "keyword matching"

    # Map complexity to preferred tier
    if complexity == "low":
        preferred_tier = "local"
    elif complexity == "medium":
        preferred_tier = "lakeshore"
    else:  # high
        preferred_tier = "cloud"

    # Get tier with intelligent fallback
    tier, fallback_reason = get_tier_with_fallback(preferred_tier, complexity)

    # If no tier available, raise error
    if tier is None:
        print(f"❌ ROUTING FAILED: {fallback_reason}")
        print(f"   Available tiers: {get_available_tiers()}")
        raise Exception("All AI services are currently unavailable. Please try again later.")

    # Debug logging
    print(f"🔍 SMART ROUTING ({method}):")
    print(f"   Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
    print(f"   Complexity: {complexity.upper()}")
    print(f"   Decision: {fallback_reason}")

    return tier


def get_model_for_tier(tier: str) -> str:
    """Get model name for a tier"""
    return DEFAULT_MODELS.get(tier, DEFAULT_MODELS["local"])


def get_routing_reason(query: str, user_preference: str, tier: str) -> str:
    """Get human-readable routing reason"""
    if user_preference != "auto":
        return f"User selected {tier} tier"

    # Get complexity from cache or recalculate
    cached = _get_cached_judgment(query)
    if cached:
        complexity = cached
        source = "(cached)"
    elif LLM_JUDGE_ENABLED:
        complexity = judge_complexity_with_llm(query)
        source = "(LLM)"
    else:
        complexity = judge_complexity_with_keywords(query)
        source = "(keywords)"

    if not complexity:
        return f"Routed to {tier.upper()}"

    # Show preferred tier vs actual tier
    if complexity == "low":
        preferred = "local"
    elif complexity == "medium":
        preferred = "lakeshore"
    else:
        preferred = "cloud"

    if tier == preferred:
        return f"LLM judge: {complexity.upper()} complexity {source} → {tier.upper()}"
    else:
        return f"LLM judge: {complexity.upper()} complexity {source} → {preferred.upper()} unavailable, using {tier.upper()}"


# =============================================================================
# POLICY CONFIGURATION (Future)
# =============================================================================

DEFAULT_QUOTAS = {
    "undergraduate": {
        "daily_requests": 100,
        "monthly_cost": 10.00,
        "allowed_tiers": ["local", "lakeshore"],
    },
    "graduate": {
        "daily_requests": 500,
        "monthly_cost": 50.00,
        "allowed_tiers": ["local", "lakeshore", "cloud"],
    },
    "faculty": {
        "daily_requests": 1000,
        "monthly_cost": 200.00,
        "allowed_tiers": ["local", "lakeshore", "cloud"],
    },
}

DEFAULT_RATE_LIMIT = 100

# =============================================================================
# AUTHENTICATION (Future)
# =============================================================================

JWT_ENABLED = False
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = "RS256"
JWKS_URL = os.getenv("JWKS_URL", "")

# =============================================================================
# OBSERVABILITY (Future)
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

SPLUNK_ENABLED = False
SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL", "")
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN", "")

# =============================================================================
# CORS CONFIGURATION
# =============================================================================

CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
RELOAD = DEBUG


# =============================================================================
# COST VALIDATION
# =============================================================================


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
