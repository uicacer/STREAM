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
    RELAY_URL,
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

    Two modes depending on whether RELAY_URL is configured:

    1. WITH RELAY (true streaming):
       Submits the job to Globus, then connects to the relay as a consumer.
       Tokens flow in real-time:  Lakeshore GPU → relay → here → frontend

    2. WITHOUT RELAY (fake streaming — original behavior):
       Waits for the full response via Globus, then splits it into word-by-word
       chunks with delays to simulate a typing effect.

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
    """
    gc = _proxy_app.globus_client
    if not gc or not gc.is_available():
        raise HTTPException(status_code=503, detail="Globus Compute not configured")

    # Resolve the HuggingFace model name for logging.
    model_info = LAKESHORE_MODELS.get(model)
    hf_name = model_info["hf_name"] if model_info else model

    logger.info(
        f"[{correlation_id}] Lakeshore direct call: {model} → {hf_name}"
        f"{' (STREAMING via relay)' if RELAY_URL else ' (batch mode)'}",
        extra={"correlation_id": correlation_id, "model": model},
    )

    # =====================================================================
    # PATH 1: TRUE STREAMING via WebSocket relay
    # =====================================================================
    # When RELAY_URL is configured, we use the relay for real-time token
    # streaming. Tokens appear in the browser as the GPU generates them.
    # If the relay connection fails (tunnel died, relay crashed), we fall
    # back to PATH 2 (batch mode) so the user still gets a response.
    if RELAY_URL:
        try:
            async for line in _forward_lakeshore_streaming(
                gc, model, messages, temperature, correlation_id
            ):
                yield line
            return
        except Exception as e:
            error_str = str(e)
            if "did not receive a valid HTTP response" in error_str:
                reason = "relay not reachable"
            elif "Connect call failed" in error_str or "ConnectionRefused" in error_str:
                reason = "relay server not running"
            elif "timed out" in error_str.lower():
                reason = "connection timed out"
            else:
                reason = error_str
            logger.warning(
                f"Falling back to BATCH MODE ({reason}). "
                f"Response will arrive all at once instead of streaming.",
                extra={"correlation_id": correlation_id},
            )
            # Fall through to PATH 2 below

    # =====================================================================
    # PATH 2: BATCH MODE (fallback when relay is unavailable)
    # =====================================================================
    # Waits for the full response via Globus Compute's control plane,
    # then returns the complete response as a single SSE burst.
    # This happens when:
    #   - RELAY_URL is not configured (relay not set up)
    #   - The relay connection failed (tunnel died, relay crashed)

    model_limits = MODEL_CONTEXT_LIMITS.get(model, {})
    max_tokens = model_limits.get("reserve_output", 2048)

    result = await gc.submit_inference(
        messages=messages,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
    )

    # Log the raw result at DEBUG level (truncated to avoid flooding logs)
    result_preview = str(result)[:500] if result else "None"
    logger.debug(
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

    # Yield the complete response as a single SSE chunk (no artificial delays).
    # For real token-by-token streaming, configure RELAY_URL (PATH 1 above).
    if content:
        chunk = {
            "choices": [{"index": 0, "delta": {"content": content}}],
        }
        yield f"data: {json.dumps(chunk)}"

    # FAKE STREAMING (commented out — kept in case we want to re-enable it).
    # This splits the batch response into word-by-word chunks with delays
    # to simulate a typing effect. Replaced by the single burst above.
    # words_per_chunk = 2
    # delay_between_chunks = 0.05  # 50ms — comfortable reading pace
    # if content:
    #     words = content.split(" ")
    #     for i in range(0, len(words), words_per_chunk):
    #         word_group = words[i : i + words_per_chunk]
    #         text = " ".join(word_group) if i == 0 else " " + " ".join(word_group)
    #         chunk = {
    #             "choices": [{"index": 0, "delta": {"content": text}}],
    #         }
    #         yield f"data: {json.dumps(chunk)}"
    #         await asyncio.sleep(delay_between_chunks)

    # Yield usage info in the final chunk (streaming.py reads this for cost)
    if usage:
        final_chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        }
        yield f"data: {json.dumps(final_chunk)}"

    yield "data: [DONE]"


async def _check_relay_reachable(relay_url: str, timeout: float = 3.0) -> bool:
    """
    Quick check if the WebSocket relay is reachable.

    Connects to the relay's /health endpoint and reads the response.
    Returns True if the relay responds, False otherwise.

    This prevents submitting expensive Globus Compute jobs when the relay
    is down (SSH tunnel expired, relay server stopped, etc.).
    """
    from websockets.asyncio.client import connect as ws_connect

    try:
        async with ws_connect(f"{relay_url}/health", open_timeout=timeout) as ws:
            await asyncio.wait_for(ws.recv(), timeout=timeout)
            return True
    except Exception as e:
        logger.warning(
            f"Relay health check failed: {type(e).__name__}: {e} " f"(url={relay_url}/health)"
        )
        return False


async def _forward_lakeshore_streaming(
    gc,
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
) -> AsyncGenerator[str, None]:
    """
    TRUE streaming from Lakeshore via the WebSocket relay.

    How it works:
      1. Check relay connectivity (prevents wasted Globus jobs)
      2. Submit the streaming job to Globus Compute (fast — just sends the job)
      3. Connect to the relay as a CONSUMER on the returned channel_id
      4. Receive tokens in real-time as the GPU generates them on Lakeshore
      5. Convert each token to SSE format and yield it to the streaming pipeline

    The data flow:
      Lakeshore GPU → vLLM (stream=True) → relay PRODUCER → relay → CONSUMER (us)
                                                                       ↓
      Frontend ← streaming.py ← SSE chunks ← this function ←─────────┘

    The SSE output format is identical to fake streaming — streaming.py
    doesn't know or care whether the tokens came from a relay or were
    split from a batch response.
    """
    from websockets.asyncio.client import connect as ws_connect

    model_limits = MODEL_CONTEXT_LIMITS.get(model, {})
    max_tokens = model_limits.get("reserve_output", 2048)

    # Step 1: Check relay connectivity BEFORE submitting the Globus job.
    # Without this check, we'd submit a Globus job (which runs on HPC for
    # ~10-30s), then discover the relay is down when we try to connect as
    # consumer. That wastes a Globus job and delays the user by 10+ seconds.
    if not await _check_relay_reachable(RELAY_URL):
        raise ConnectionError("Relay not reachable — did not receive a valid HTTP response")

    # Step 2: Submit the streaming job to Globus Compute.
    # This returns immediately with a channel_id. The actual inference
    # hasn't started yet — Globus needs a few seconds to route the job
    # to Lakeshore.
    result = await gc.submit_streaming_inference(
        messages=messages,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        relay_url=RELAY_URL,
    )

    # Check for submission errors (auth, config, etc.)
    if "error" in result:
        error_msg = result.get("error", "Unknown error")
        error_type = result.get("error_type", "")
        if error_type == "AuthenticationError":
            raise HTTPException(status_code=401, detail=error_msg)
        raise HTTPException(status_code=503, detail=f"Lakeshore streaming failed: {error_msg}")

    channel_id = result["channel_id"]
    logger.info(
        f"[{correlation_id}] Connecting to relay as consumer "
        f"(channel={channel_id[:8]}, relay={RELAY_URL})",
        extra={"correlation_id": correlation_id},
    )

    # Step 3: Connect to the relay as a CONSUMER.
    # We connect immediately after submitting the job. The relay will hold
    # our connection until the producer (Lakeshore) connects and starts
    # sending tokens. If the producer sent tokens before we connected,
    # the relay buffered them and flushes them to us now.
    try:
        async with ws_connect(f"{RELAY_URL}/consume/{channel_id}") as ws:
            # Step 4: Receive tokens and convert to SSE format.
            # Each message from the relay is a JSON object:
            #   {"type": "token", "content": "Hello"}  — a generated token
            #   {"type": "done", "usage": {...}}        — stream complete
            #   {"type": "error", "message": "..."}     — something went wrong
            async for msg_str in ws:
                msg = json.loads(msg_str)

                if msg["type"] == "token":
                    # Convert to the same SSE delta format that streaming.py expects.
                    # This is identical to what litellm streaming produces.
                    chunk = {
                        "choices": [{"index": 0, "delta": {"content": msg["content"]}}],
                    }
                    yield f"data: {json.dumps(chunk)}"

                elif msg["type"] == "done":
                    # Stream complete. Include usage stats if available.
                    usage = msg.get("usage", {})
                    if usage:
                        final_chunk = {
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": usage,
                        }
                        yield f"data: {json.dumps(final_chunk)}"
                    yield "data: [DONE]"
                    break

                elif msg["type"] == "error":
                    error_msg = msg.get("message", "Unknown streaming error")
                    logger.error(
                        f"[{correlation_id}] Relay error: {error_msg}",
                        extra={"correlation_id": correlation_id},
                    )
                    # Don't break — the producer will send "done" after the error

    except Exception as e:
        error_str = str(e)

        # Translate cryptic WebSocket errors into actionable messages
        if "did not receive a valid HTTP response" in error_str:
            cause = "SSH tunnel expired"
            fix = "ssh -4 -R 80:localhost:8765 nokey@localhost.run  then update RELAY_URL in .env"
        elif "Connect call failed" in error_str or "ConnectionRefused" in error_str:
            cause = "Relay server not running"
            fix = "python -m stream.relay.server"
        elif "timed out" in error_str.lower():
            cause = "Connection timed out"
            fix = "Check that the relay server and SSH tunnel are both running"
        else:
            cause = "Unexpected error"
            fix = error_str

        logger.error(
            f"\n{'=' * 60}\n"
            f"  RELAY CONNECTION FAILED\n"
            f"  Cause: {cause}\n"
            f"  Fix:   {fix}\n"
            f"  Raw:   {error_str}\n"
            f"{'=' * 60}",
            extra={"correlation_id": correlation_id},
        )
        # Re-raise so _forward_lakeshore() can catch it and fall back to batch mode
        raise


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
