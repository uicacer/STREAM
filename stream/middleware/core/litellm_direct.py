"""
Direct LiteLLM library calls for desktop mode.

WHY THIS FILE EXISTS:
---------------------
In Docker/server mode, chat requests go through the LiteLLM HTTP server on port 4000:

    Your code  →  HTTP POST  →  LiteLLM server (:4000)  →  Cloud API

In desktop mode, there's no separate LiteLLM server running. Instead, we call
the litellm Python library directly — a function call in the same process:

    Your code  →  litellm.acompletion()  →  Cloud API

Same library, two usage patterns:
    Server mode  = litellm as a "restaurant" (you send HTTP orders to port 4000)
    Desktop mode = litellm as a "cookbook"   (you cook directly with it in-process)

The output format (SSE lines) is identical either way, so the rest of the
streaming pipeline (streaming.py → chat.py → frontend) works unchanged.

HOW MODEL NAME TRANSLATION WORKS:
----------------------------------
STREAM uses friendly model names like "cloud-claude" and "local-llama".
The LiteLLM server translates these using litellm_config.yaml:
    "cloud-claude" → "claude-sonnet-4-20250514"
    "local-llama"  → "ollama/llama3.2:3b"

Since there's no server in desktop mode, we load the same YAML file
and do the translation ourselves (see _load_model_map below).
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import litellm
import yaml
from fastapi import HTTPException

import stream.proxy.app as _proxy_app
from stream.middleware.config import (
    LAKESHORE_MODELS,
    LAKESHORE_PROXY_URL,
    MODEL_CONTEXT_LIMITS,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

# Reduce litellm's verbose startup logging in desktop mode.
# Without this, litellm prints debug info about every provider it loads.
litellm.suppress_debug_info = True


# =============================================================================
# MODEL NAME MAPPING
# =============================================================================
#
# This dict maps friendly names → actual provider model names + connection info.
# It's built once at import time from litellm_config.yaml.
#
# Example entries after loading:
#   "cloud-claude"    → {model: "claude-sonnet-4-20250514", api_base: None}
#   "local-llama"     → {model: "ollama/llama3.2:3b", api_base: "http://ollama:11434"}
#   "lakeshore-qwen"  → {model: "openai/Qwen/Qwen2.5-1.5B-Instruct", api_base: "http://lakeshore-proxy:8001/v1"}
#

_MODEL_MAP: dict[str, dict] = {}


def _load_model_map():
    """
    Parse litellm_config.yaml and build a lookup table.

    The YAML lives at stream/gateway/litellm_config.yaml and defines all
    available models with their provider names and connection details.
    This is the same file the LiteLLM Docker server reads — we're just
    reading it ourselves instead of relying on the server.
    """
    global _MODEL_MAP

    # Navigate from this file to the YAML:
    #   stream/middleware/core/litellm_direct.py  (this file)
    #   → stream/middleware/core/                 (.parent)
    #   → stream/middleware/                      (.parent.parent)
    #   → stream/                                 (.parent.parent.parent)
    #   → stream/gateway/litellm_config.yaml
    config_path = Path(__file__).resolve().parent.parent.parent / "gateway" / "litellm_config.yaml"

    if not config_path.exists():
        logger.warning(f"litellm_config.yaml not found at {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for entry in config.get("model_list", []):
        friendly_name = entry["model_name"]
        params = entry.get("litellm_params", {})

        _MODEL_MAP[friendly_name] = {
            "model": params.get("model"),  # e.g., "claude-sonnet-4-20250514"
            "api_base": params.get("api_base"),  # e.g., "http://ollama:11434" (Docker URL)
            "api_key": params.get("api_key"),  # e.g., "os.environ/ANTHROPIC_API_KEY" or "dummy"
        }

    logger.info(f"Loaded {len(_MODEL_MAP)} model mappings for desktop mode")


# Load mapping once at import time (runs when this module is first imported)
_load_model_map()


def _resolve_model(friendly_name: str) -> dict:
    """
    Translate a friendly model name into kwargs for litellm.acompletion().

    Does two key things:

    1. MODEL NAME TRANSLATION:
       "cloud-claude" → "claude-sonnet-4-20250514"
       litellm auto-detects the provider from the model name. For example,
       "claude-*" routes to Anthropic, "gpt-*" routes to OpenAI,
       "ollama/*" routes to Ollama.

    2. API BASE URL FIXING:
       The YAML has Docker-only URLs (e.g., "http://ollama:11434") that don't
       work outside Docker. We replace them with localhost equivalents:
         "http://ollama:11434"            → OLLAMA_BASE_URL (http://localhost:11434)
         "http://lakeshore-proxy:8001/v1" → f"{LAKESHORE_PROXY_URL}/v1"
         None (cloud models)              → litellm uses provider defaults automatically

    3. API KEY HANDLING:
       For cloud models, the YAML says "os.environ/ANTHROPIC_API_KEY" — that's
       LiteLLM server syntax. The litellm library reads env vars automatically,
       so we don't need to pass API keys for cloud models.
       For Lakeshore (vLLM proxy), the key is "dummy" — we pass it explicitly
       since vLLM requires an api_key header even if it doesn't validate it.

    Returns:
        Dict of kwargs to pass to litellm.acompletion() or litellm.completion()
    """
    if friendly_name not in _MODEL_MAP:
        raise ValueError(
            f"Unknown model: {friendly_name}. " f"Available: {list(_MODEL_MAP.keys())}"
        )

    entry = _MODEL_MAP[friendly_name]
    kwargs = {"model": entry["model"]}

    # Fix Docker-specific api_base URLs for desktop mode.
    # These URLs contain Docker service names ("ollama", "lakeshore-proxy")
    # that only resolve inside Docker's virtual network.
    api_base = entry.get("api_base")
    if api_base:
        if "ollama" in api_base:
            # Docker: http://ollama:11434  →  Desktop: http://localhost:11434
            kwargs["api_base"] = OLLAMA_BASE_URL
        elif "lakeshore" in api_base:
            # Docker: http://lakeshore-proxy:8001/v1  →  Desktop: http://127.0.0.1:8001/v1
            # We preserve the /v1 path suffix — vLLM expects it.
            # Split on the port number to extract the path after it.
            path_suffix = api_base.split("8001")[-1]  # Gets "/v1" from the URL
            kwargs["api_base"] = f"{LAKESHORE_PROXY_URL}{path_suffix}"
        else:
            kwargs["api_base"] = api_base

    # Handle API keys.
    # "os.environ/KEY_NAME" is LiteLLM server syntax — the library reads env
    # vars automatically so we skip those. We only pass literal keys like "dummy".
    api_key = entry.get("api_key")
    if api_key and not api_key.startswith("os.environ"):
        kwargs["api_key"] = api_key

    return kwargs


# =============================================================================
# STREAMING CHAT COMPLETION
# =============================================================================
# Used by litellm_client.py when STREAM_MODE == "desktop".
# This replaces the HTTP call to the LiteLLM server with a direct library call.


async def _forward_lakeshore(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
) -> AsyncGenerator[str, None]:
    """
    Call the Lakeshore Globus Compute client directly (desktop mode only).

    WHY NOT USE litellm.acompletion() FOR LAKESHORE?
    -------------------------------------------------
    In desktop mode, litellm.acompletion() for lakeshore would make an HTTP
    POST to http://127.0.0.1:5000/lakeshore/v1/chat/completions — which is
    the SAME server we're running on. This "self-connection" is problematic:

      litellm (our process) → HTTP POST → FastAPI (same process)
                                            ↓
                                    Proxy handler → Globus Compute
                                            ↓
                                    Response flows back through HTTP
                                            ↓
                              litellm reads the response

    The server is both the client AND the server for the same request.
    This can cause deadlocks, timeouts, and empty responses because the
    single-worker event loop must handle both sides simultaneously.

    SOLUTION: Skip HTTP entirely. Call the Globus Compute client directly
    (it's already loaded in the same process), then convert the response
    to the same SSE format that streaming.py expects.

      forward_direct → globus_client.submit_inference() → Globus Compute
                            ↓
                    vLLM response (JSON) → convert to SSE chunks
                            ↓
                    streaming.py processes normally

    The response format is identical either way — streaming.py doesn't
    know or care whether the data came from HTTP or a direct call.
    """
    gc = _proxy_app.globus_client
    if not gc or not gc.is_available():
        raise HTTPException(status_code=503, detail="Globus Compute not configured")

    # Resolve the HuggingFace model name for logging.
    model_info = LAKESHORE_MODELS.get(model)
    hf_name = model_info["hf_name"] if model_info else model

    logger.info(
        f"[{correlation_id}] Lakeshore direct call: {model} → {hf_name}",
        extra={"correlation_id": correlation_id, "model": model},
    )

    # Call Globus Compute directly — no HTTP, no self-connection.
    # submit_inference handles vLLM URL routing and HF name resolution
    # internally using LAKESHORE_MODELS config.
    #
    # max_tokens = how many tokens the model is allowed to generate.
    # We read this from MODEL_CONTEXT_LIMITS (defined in config.py) where
    # each model has a "reserve_output" field — that's the number of tokens
    # reserved for the model's response.
    model_limits = MODEL_CONTEXT_LIMITS.get(model, {})
    max_tokens = model_limits.get("reserve_output", 2048)

    result = await gc.submit_inference(
        messages=messages,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
    )

    # Log the raw result for debugging (truncated to avoid flooding logs)
    result_preview = str(result)[:500] if result else "None"
    logger.info(
        f"[{correlation_id}] Lakeshore raw result: {result_preview}",
        extra={"correlation_id": correlation_id},
    )

    # Check for errors from Globus/vLLM
    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error", "Unknown Lakeshore error")
        error_type = result.get("error_type", "")
        if error_type == "AuthenticationError":
            raise HTTPException(status_code=401, detail=error_msg)
        raise HTTPException(status_code=503, detail=f"Lakeshore inference failed: {error_msg}")

    # Convert the complete vLLM response to SSE chunks.
    # vLLM returns a standard OpenAI chat completion response:
    #   {"choices": [{"message": {"content": "the response text"}}], "usage": {...}}
    #
    # We split the content into word-by-word SSE events — the same format
    # that litellm streaming would produce. This way streaming.py handles
    # it identically to any other tier.
    choices = result.get("choices", []) if isinstance(result, dict) else []
    if not choices:
        logger.warning(
            f"[{correlation_id}] Lakeshore returned no choices. Result type: {type(result).__name__}",
            extra={"correlation_id": correlation_id},
        )
        yield "data: [DONE]"
        return

    content = choices[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})

    logger.info(
        f"[{correlation_id}] Lakeshore content length: {len(content)}, usage: {usage}",
        extra={"correlation_id": correlation_id},
    )

    # Yield content word by word as SSE delta chunks.
    # We add a small delay between chunks to simulate streaming — the same
    # approach as _convert_json_to_sse_stream() in proxy/app.py (server mode).
    # Without this delay, all words yield in a single event loop tick and
    # FastAPI sends them as one block — the frontend sees the whole response
    # appear at once instead of progressively.
    words_per_chunk = 2  # Match server mode: 2 words per chunk
    delay_between_chunks = 0.05  # 50ms — comfortable reading pace

    if content:
        words = content.split(" ")
        for i in range(0, len(words), words_per_chunk):
            word_group = words[i : i + words_per_chunk]
            text = " ".join(word_group) if i == 0 else " " + " ".join(word_group)
            chunk = {
                "choices": [{"index": 0, "delta": {"content": text}}],
            }
            yield f"data: {json.dumps(chunk)}"
            await asyncio.sleep(delay_between_chunks)

    # Yield usage info in the final chunk (streaming.py reads this for cost)
    if usage:
        final_chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        }
        yield f"data: {json.dumps(final_chunk)}"

    yield "data: [DONE]"


async def forward_direct(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
) -> AsyncGenerator[str, None]:
    """
    Call litellm library directly and stream the response as SSE lines.

    This is the desktop-mode replacement for forward_to_litellm() in
    litellm_client.py. It produces the EXACT SAME SSE output format:

        "data: {"choices": [{"delta": {"content": "Hello"}}]}"
        "data: {"choices": [{"delta": {"content": " world"}}]}"
        "data: [DONE]"

    Because the format is identical, streaming.py (which consumes these lines)
    works without any changes. It doesn't know or care whether the lines came
    from an HTTP server or a direct library call.

    For Lakeshore tier: uses _forward_lakeshore() which calls Globus Compute
    directly instead of going through HTTP (see that function's docstring).

    For Local and Cloud tiers: uses litellm.acompletion() which calls the
    provider API directly (Ollama for local, Anthropic/OpenAI for cloud).

    Args:
        model: Friendly model name (e.g., "cloud-claude", "local-llama")
               Gets translated to actual provider model name internally.
        messages: Conversation history (list of {role, content} dicts)
        temperature: 0.0 = deterministic, 2.0 = creative
        correlation_id: Unique request ID for log tracing

    Yields:
        SSE-formatted lines (same format as LiteLLM HTTP server)
    """
    # Lakeshore: call Globus Compute directly (no HTTP self-connection)
    if model.startswith("lakeshore"):
        async for line in _forward_lakeshore(model, messages, temperature, correlation_id):
            yield line
        return

    # Cloud and Local: call litellm library directly
    kwargs = _resolve_model(model)
    kwargs.update(
        {
            "messages": messages,
            "temperature": temperature,
            "stream": True,  # Enable streaming (returns async generator of chunks)
        }
    )

    logger.debug(
        f"[{correlation_id}] Direct litellm call: {model} → {kwargs['model']}",
        extra={"correlation_id": correlation_id, "model": model},
    )

    try:
        # litellm.acompletion() = async version of litellm.completion()
        # With stream=True, it returns an async generator (CustomStreamWrapper)
        # that yields ModelResponse chunks as the LLM generates tokens.
        response = await litellm.acompletion(**kwargs)

        # Each chunk is a ModelResponse object with the same structure as
        # OpenAI's streaming format. We convert to dict → JSON → SSE line.
        #
        # Example chunk after model_dump():
        # {
        #   "choices": [{"delta": {"content": "Hello"}, "finish_reason": null}],
        #   "usage": null  (or {"prompt_tokens": X, "completion_tokens": Y} in last chunk)
        # }
        async for chunk in response:
            # model_dump() converts the Pydantic model to a plain dict.
            # exclude_none=True removes null fields for cleaner output.
            chunk_dict = chunk.model_dump(exclude_none=True)
            yield f"data: {json.dumps(chunk_dict)}"

        # Signal end-of-stream. This is the SSE convention from OpenAI's API.
        # streaming.py checks for this to know the response is complete.
        yield "data: [DONE]"

    except litellm.AuthenticationError as e:
        # API key is invalid or subscription expired.
        # litellm raises this for 401/403 responses from cloud providers.
        raise HTTPException(
            status_code=401,
            detail={
                "error_type": "auth_subscription",
                "message": (
                    "Cloud provider authentication failed. "
                    "Your API key may be invalid or your subscription may have expired."
                ),
                "raw_error": str(e),
                "provider": "cloud",
            },
        ) from e

    except litellm.RateLimitError as e:
        # Too many requests to the provider (429 response).
        raise HTTPException(
            status_code=429,
            detail={
                "error_type": "rate_limit",
                "message": (
                    "Rate limit exceeded. "
                    "Please wait a moment or switch to a different provider."
                ),
                "raw_error": str(e),
                "provider": "cloud",
            },
        ) from e

    except litellm.APIConnectionError as e:
        # Can't reach the provider — Ollama not running, network down, etc.
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to AI provider: {str(e)}",
        ) from e

    except Exception as e:
        # Catch-all for unexpected litellm errors
        raise HTTPException(
            status_code=502,
            detail=f"LiteLLM direct call failed: {str(e)}",
        ) from e


# =============================================================================
# NON-STREAMING JUDGE CALL
# =============================================================================
# Used by complexity_judge.py when STREAM_MODE == "desktop".
# The judge doesn't need streaming — it asks "is this LOW, MEDIUM, or HIGH?"
# and gets a one-word answer.


def judge_direct(
    model: str,
    prompt: str,
    timeout: float,
) -> dict:
    """
    Synchronous litellm call for the complexity judge (desktop mode).

    The complexity judge classifies queries as LOW/MEDIUM/HIGH to route them
    to the right AI tier. It sends a short prompt and expects a one-word answer.
    No streaming needed — just a simple request/response.

    Returns a dict matching the OpenAI response format that complexity_judge.py
    already knows how to parse:
        {
            "choices": [{"message": {"content": "LOW"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 1}
        }

    Args:
        model: Friendly model name (e.g., "local-llama", "cloud-haiku")
        prompt: The judge prompt with the user's query embedded
        timeout: Max seconds to wait for the judge's response
    """
    # Translate friendly name → actual litellm kwargs
    kwargs = _resolve_model(model)
    kwargs.update(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,  # Just need one word: LOW, MEDIUM, or HIGH
            "temperature": 0.0,  # Deterministic — same query gives same result
            "timeout": timeout,
        }
    )

    # litellm.completion() is the synchronous version (vs acompletion for async).
    # This matches the complexity judge which is also synchronous.
    response = litellm.completion(**kwargs)

    # Convert Pydantic ModelResponse → plain dict so the existing parsing
    # code in complexity_judge.py works unchanged (it does data["choices"][0]...).
    return response.model_dump()
