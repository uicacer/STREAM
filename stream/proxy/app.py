"""
Lakeshore vLLM Proxy Service — Routes requests via Globus Compute or SSH.

DUAL-USE DESIGN:
----------------
This module serves two roles:

1. STANDALONE SERVICE (Docker mode):
   Runs as its own container on port 8001. The middleware forwards Lakeshore
   requests to this separate service via HTTP.
   → Start with: python -m stream.proxy.app

2. EMBEDDED ROUTER (Desktop mode):
   The middleware imports `router` from this module and mounts it at /lakeshore
   on the main FastAPI app. No separate process needed — everything runs in
   one server on port 5000.

WHY AN APIRouter:
-----------------
FastAPI's APIRouter lets you define routes that can be included in ANY app.
Think of it as a "plug-in" — define routes once, then plug them into either
a standalone app (Docker) or the middleware app (desktop). Same routes,
same code, different hosting.

The standalone `app` at the bottom is just a thin wrapper:
  app = FastAPI()
  app.include_router(router)  ← same router used by desktop mode
"""

import asyncio
import json
import logging
import os

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from globus_sdk import GlobusAPIError

from stream.middleware.core.globus_compute_client import GlobusComputeClient

# =========================================================================
# Configuration — read from environment variables
# =========================================================================
# These settings control how the proxy connects to the Lakeshore HPC cluster.
# In Docker, they come from docker-compose.yml. In desktop, from config.py.
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

USE_GLOBUS_COMPUTE = os.getenv("USE_GLOBUS_COMPUTE", "true").lower() == "true"
GLOBUS_COMPUTE_ENDPOINT_ID = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://ga-001:8000")
LAKESHORE_VLLM_ENDPOINT = os.getenv("LAKESHORE_VLLM_ENDPOINT", "http://host.docker.internal:8000")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =========================================================================
# Initialize Globus Compute client (module-level, runs once at import time)
# =========================================================================
# This client handles authentication and job submission to UIC's HPC cluster.
# It's initialized at module level so both standalone and embedded modes share it.
globus_client = None
if USE_GLOBUS_COMPUTE and GLOBUS_COMPUTE_ENDPOINT_ID:
    try:
        globus_client = GlobusComputeClient()  # Reads from env vars
        logger.info("Globus Compute client loaded")
    except Exception as e:
        logger.error(f"Failed to initialize Globus Compute client: {e}")


# =========================================================================
# APIRouter — the actual route definitions
# =========================================================================
# These routes can be included in any FastAPI app:
#   - Standalone proxy: app.include_router(router)
#   - Desktop middleware: app.include_router(router, prefix="/lakeshore")
router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Lakeshore vLLM Proxy",
        "mode": "globus_compute" if USE_GLOBUS_COMPUTE else "ssh",
        "globus_configured": bool(globus_client and globus_client.is_available())
        if USE_GLOBUS_COMPUTE
        else False,
    }


@router.post("/reload-auth")
async def reload_authentication():
    """
    Reload Globus credentials from disk.

    This endpoint should be called after the user authenticates on the host machine.
    It forces the proxy to re-read the credentials from ~/.globus_compute/storage.db.

    Returns:
        JSON with success status and message
    """
    if not USE_GLOBUS_COMPUTE or not globus_client:
        return {"success": False, "message": "Globus Compute not configured"}

    try:
        success, message = globus_client.reload_credentials()
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Failed to reload credentials: {e}")
        return {"success": False, "message": f"Failed to reload: {str(e)}"}


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    model = body.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    # =========================================================================
    # MAX_TOKENS AND CONTEXT WINDOW - LAKESHORE TIER ONLY
    # =========================================================================
    # This proxy handles ONLY Lakeshore tier requests. Cloud and Local tiers
    # go through LiteLLM directly and don't have this constraint (Cloud has
    # 200K+ context, Local depends on the Ollama model configuration).
    #
    # CONTEXT WINDOW BASICS:
    # LLMs have a fixed "context window" - the total tokens they can process.
    # For Qwen2.5-1.5B on Lakeshore, the context window is 8192 tokens.
    #
    # The constraint: input_tokens + output_tokens <= context_window
    #   - input_tokens = your messages (system prompt + conversation history)
    #   - output_tokens = the model's response (controlled by max_tokens)
    #
    # TOKEN TO WORD CONVERSION (rough estimate):
    #   - 1 token ≈ 0.75 words (or ~4 characters)
    #   - 1000 tokens ≈ 750 words
    #   - 8192 tokens ≈ 6,100 words total context
    #
    # WHY 1024 TOKENS (15%) IS A GOOD DEFAULT FOR OUTPUT:
    # =========================================================================
    # In a chat application, conversation history grows over time, but individual
    # responses are typically short (100-500 tokens for most answers).
    #
    # With 8192 context and max_tokens=1024:
    #   - ~7000 tokens for conversation history (85%) ≈ 5,250 words of chat
    #   - ~1000 tokens for model response (15%) ≈ 750 words per response
    #
    # This allows:
    #   - Long conversations with many back-and-forth messages
    #   - Sufficient response length for detailed answers (750 words is plenty)
    #   - Maximizes available space for conversation context
    #   - Avoids "max_tokens too large" errors as history grows
    #
    # If a user explicitly requests a larger max_tokens, we use their value.
    # vLLM will return an error if input + max_tokens > context_window.
    # =========================================================================
    max_tokens = body.get("max_tokens", 1024)

    logger.info(
        f"Proxy request: model={model}, messages={len(messages)}, stream={stream}, mode={'globus' if USE_GLOBUS_COMPUTE else 'ssh'}"
    )

    if USE_GLOBUS_COMPUTE:
        return await _route_via_globus_compute(model, messages, temperature, max_tokens, stream)
    else:
        return await _route_via_ssh(model, messages, temperature, max_tokens, stream)


async def _route_via_globus_compute(model, messages, temperature, max_tokens, stream):
    if not globus_client or not globus_client.is_available():
        raise HTTPException(status_code=503, detail="Globus Compute not configured")

    if stream:
        logger.warning(
            "Streaming not yet supported via Globus Compute, converting non-streaming response to SSE format"
        )

    try:
        logger.info(f"Submitting to Globus endpoint: {GLOBUS_COMPUTE_ENDPOINT_ID}")
        result = await globus_client.submit_inference(
            messages=messages, temperature=temperature, max_tokens=max_tokens, model=model
        )

        if "error" in result:
            error_msg = result.get("error", "Unknown error")
            error_type = result.get("error_type", "UnknownError")

            # Use HTTP 401 for authentication errors, 503 for other service errors
            if error_type == "AuthenticationError":
                raise HTTPException(
                    status_code=401, detail=f"Globus Compute authentication required: {error_msg}"
                )
            else:
                raise HTTPException(
                    status_code=503, detail=f"Lakeshore inference failed: {error_msg}"
                )

        logger.info("Globus Compute inference successful")

        if stream:
            return _convert_json_to_sse_stream(result)
        return result

    except HTTPException:
        raise
    except GlobusAPIError as e:
        # Handle Globus API errors specifically
        if e.http_status in (401, 403):
            raise HTTPException(
                status_code=401, detail=f"Globus authentication required: {str(e)}"
            ) from e
        else:
            raise HTTPException(status_code=503, detail=f"Globus API error: {str(e)}") from e
    except Exception as e:
        logger.error(f"Globus Compute routing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}") from e


async def _route_via_ssh(model, messages, temperature, max_tokens, stream):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    target_url = f"{LAKESHORE_VLLM_ENDPOINT}/v1/chat/completions"
    logger.info(f"Forwarding to SSH endpoint: {target_url}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                async with client.stream("POST", target_url, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"vLLM error: {error_text.decode()}",
                        )

                    async def stream_generator():
                        async for line in response.aiter_lines():
                            if line.strip():
                                yield line + "\n"

                    return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                response = await client.post(target_url, json=payload)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code, detail=f"vLLM error: {response.text}"
                    )
                logger.info("SSH forwarding successful")
                return response.json()

    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Lakeshore via SSH. Is the tunnel running? Error: {str(e)}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="vLLM request timeout") from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}") from e


def _convert_json_to_sse_stream(json_response: dict):
    """
    Convert a complete chat.completion response to a simulated streaming response.

    WHY SIMULATE STREAMING FOR GLOBUS COMPUTE?
    =========================================================================
    Globus Compute is a Function-as-a-Service (FaaS) system:
    - You submit a function → it runs remotely → returns complete result
    - There's NO way to get partial results while the function is running
    - This is fundamentally different from Local/Cloud tiers which support true streaming

    To provide a consistent user experience across all tiers, we simulate streaming:
    - Split the complete response into small chunks (groups of words)
    - Yield each chunk as a separate SSE event
    - Add tiny delays between chunks to create natural "typing" effect

    The result: Users see text appearing progressively, just like Local/Cloud tiers,
    even though we already have the complete response.
    =========================================================================
    """
    # Configuration for simulated streaming
    # Speed calculation: words_per_chunk / delay = words per second
    # Current: 2 words / 0.05s = ~40 words/second (comfortable reading pace)
    # For reference: average reading speed is ~250 words/minute = ~4 words/second
    words_per_chunk = 2  # Fewer words per chunk = smoother appearance
    delay_between_chunks = 0.05  # 50ms between chunks (comfortable pace)

    async def sse_generator():
        choices = json_response.get("choices", [])
        if not choices:
            yield "data: [DONE]\n\n"
            return

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        role = message.get("role", "assistant")

        # Common fields for all chunks
        chunk_base = {
            "id": json_response.get("id", ""),
            "object": "chat.completion.chunk",
            "created": json_response.get("created", 0),
            "model": json_response.get("model", ""),
        }

        # Chunk 1: Send the role (assistant)
        if role:
            chunk = {
                **chunk_base,
                "choices": [
                    {"index": 0, "delta": {"role": role, "content": ""}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Chunk 2+: Send content in small word groups to simulate streaming
        # =====================================================================
        # Instead of sending all content at once, we:
        # 1. Split into words
        # 2. Group into small chunks (e.g., 3 words at a time)
        # 3. Yield each chunk with a small delay
        # This creates a smooth "typing" effect for the user
        # =====================================================================
        if content:
            words = content.split(" ")

            for i in range(0, len(words), words_per_chunk):
                # Get the next group of words
                word_group = words[i : i + words_per_chunk]

                # Add space before words (except for the first chunk)
                text_chunk = " ".join(word_group) if i == 0 else " " + " ".join(word_group)

                chunk = {
                    **chunk_base,
                    "choices": [
                        {"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Small delay to create natural streaming effect
                # Without this, all chunks would arrive instantly (defeating the purpose)
                await asyncio.sleep(delay_between_chunks)

        # Final chunk: Signal completion
        chunk = {
            **chunk_base,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}
            ],
            "usage": json_response.get("usage", {}),
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# =========================================================================
# Standalone FastAPI app (used in Docker mode only)
# =========================================================================
# This wraps the router in a full FastAPI app. In Docker, this module is
# the entry point: uvicorn stream.proxy.app:app
# In desktop mode, only `router` is imported — this `app` object is ignored.
app = FastAPI(title="Lakeshore vLLM Proxy")
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Log startup info when running as a standalone service (Docker mode)."""
    logger.info("=" * 60)
    logger.info("Lakeshore vLLM Proxy Starting")
    logger.info("=" * 60)
    logger.info(f"Mode: {'Globus Compute' if USE_GLOBUS_COMPUTE else 'SSH Port Forward'}")
    if USE_GLOBUS_COMPUTE and globus_client:
        logger.info(f"Globus Endpoint: {GLOBUS_COMPUTE_ENDPOINT_ID}")
        logger.info(f"vLLM Server URL: {VLLM_SERVER_URL}")
    logger.info(f"Listening on: {PROXY_HOST}:{PROXY_PORT}")
    logger.info("=" * 60)


def main():
    import uvicorn

    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
