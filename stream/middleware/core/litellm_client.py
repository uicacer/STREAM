"""
LiteLLM Gateway HTTP client.

This module is the ONLY place that communicates with the LiteLLM gateway.
All HTTP requests to LiteLLM flow through this module, providing:
1. Centralized error handling
2. Consistent logging
3. Connection pooling (via httpx.AsyncClient)
4. Timeout management

LiteLLM acts as a unified gateway to multiple LLM providers:
- Local: Ollama models
- Campus: vLLM on GPU cluster
- Cloud: OpenAI, Anthropic, etc.
"""

import logging
from collections.abc import AsyncGenerator

import httpx
from fastapi import HTTPException

from stream.middleware.config import LITELLM_API_KEY, LITELLM_BASE_URL, STREAM_MODE
from stream.middleware.core.litellm_direct import forward_direct

logger = logging.getLogger(__name__)

# =============================================================================
# HTTP TIMEOUT CONFIGURATION
# =============================================================================
#
# Why granular timeouts prevent the "hang" problem:
# -------------------------------------------------
# With a single timeout (timeout=120), httpx waits 120s for the ENTIRE request.
# If cloud provider hiccups mid-stream, user sees UI freeze for 2 minutes!
#
# With granular timeouts, we detect problems in seconds:
#
#   Normal streaming: chunks arrive every 100-500ms
#   If 10s pass with no chunk → something's wrong → timeout → user can retry
#
# Quick glossary:
# ---------------
# • TCP connection: The network "pipe" between your computer and the server.
#                   Must be opened before any data can flow.
#
# • Connection pool: Reuses open connections instead of creating new ones
#                    for each request. Faster, less resource-intensive.
#
# Timeout values:
# ---------------
# • connect (10s):  Time to open connection to server. Usually <1s.
# • read (10s):     Max gap between chunks during streaming. KEY TIMEOUT!
#                   Detects mid-stream hangs from provider issues.
# • write (30s):    Time to send request (long conversations need more).
# • pool (10s):     Time to get connection from pool. Should be instant.
#
# NOTE: For user feedback BEFORE timeout (e.g., "taking a while..."),
#       see the chunk gap warning in streaming.py
#
REQUEST_TIMEOUT = httpx.Timeout(
    connect=10.0,  # 10s to establish connection
    read=10.0,  # 10s max gap between chunks (key for detecting hangs!)
    write=30.0,  # 30s to send request body
    pool=10.0,  # 10s to get pooled connection
)


async def forward_to_litellm(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
    user_api_keys: dict[str, str] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Forward a chat completion request to LiteLLM and stream the response.

    This function:
    1. Constructs the LiteLLM API payload
    2. Opens a streaming HTTP connection
    3. Yields SSE-formatted lines as they arrive
    4. Handles connection errors and timeouts

    The streaming approach (vs. waiting for complete response) provides:
    - Better user experience (see response as it's generated)
    - Lower memory usage (no need to buffer entire response)
    - Ability to cancel mid-generation

    Args:
        model: Model identifier (e.g., "gpt-4", "llama3.2:3b")
               LiteLLM routes this to the appropriate backend
        messages: List of message dicts with role and content
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
        correlation_id: Unique request ID for tracing through logs
        user_api_keys: Optional dict mapping env var names to user-provided
                       API key values. Example:
                       {"OPENROUTER_API_KEY": "sk-or-v1-abc123"}
                       When present, these override environment variables.

    Yields:
        SSE-formatted lines from LiteLLM, e.g.:
        "data: {"choices": [{"delta": {"content": "Hello"}}]}"
        "data: [DONE]"

    Raises:
        HTTPException: On connection errors, timeouts, or non-200 responses

    Note:
        This is an async generator - it yields lines as they arrive from LiteLLM,
        rather than buffering the entire response in memory.
    """
    # -------------------------------------------------------------------------
    # DESKTOP MODE: Call litellm library directly (skip HTTP server)
    # -------------------------------------------------------------------------
    # In server/Docker mode, we send an HTTP request to the LiteLLM server
    # running on port 4000. In desktop mode, there's no server — we call
    # the litellm Python library directly. Same library, different usage:
    #   Server:  HTTP POST → LiteLLM server (:4000) → Cloud API
    #   Desktop: litellm.acompletion() → Cloud API (no HTTP hop)
    #
    # The output format (SSE lines) is identical, so streaming.py doesn't
    # know or care which path was used.
    if STREAM_MODE == "desktop":
        async for line in forward_direct(
            model,
            messages,
            temperature,
            correlation_id,
            user_api_keys=user_api_keys,
        ):
            yield line
        return

    # -------------------------------------------------------------------------
    # SERVER MODE: Forward via HTTP to LiteLLM server (existing behavior below)
    # -------------------------------------------------------------------------

    # Construct the request payload
    # This matches OpenAI's API format, which LiteLLM expects
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    # Enable reasoning/thinking for direct provider calls only.
    # OpenRouter-proxied models (cloud-or-* mapped to openrouter/*) don't support
    # reasoning_effort through litellm — litellm raises UnsupportedParamsError.
    # In server mode, cloud models go through OpenRouter, so we skip this.
    from stream.middleware.config import is_reasoning_model

    is_cloud_model = model.startswith("cloud")
    if not is_cloud_model and is_reasoning_model(model):
        payload["reasoning_effort"] = "low"
        logger.info(
            f"[{correlation_id}] Enabling reasoning (effort=low) for {model}",
            extra={"correlation_id": correlation_id},
        )

    # Prepare HTTP headers
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
        "X-Correlation-ID": correlation_id,  # For distributed tracing
    }

    # Construct full URL to LiteLLM gateway
    url = f"{LITELLM_BASE_URL}/v1/chat/completions"

    logger.debug(
        f"[{correlation_id}] Forwarding request to LiteLLM: {url}",
        extra={
            "correlation_id": correlation_id,
            "model": model,
            "message_count": len(messages),
        },
    )

    try:
        # FIX: Combine nested async with statements (SIM117)
        async with (
            httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client,
            client.stream("POST", url, json=payload, headers=headers) as response,
        ):
            # Check for HTTP errors (4xx, 5xx)
            if response.status_code != 200:
                # Read error details from response body
                error_text = await response.aread()  # aread means "asynchronous read"
                error_message = error_text.decode("utf-8")
                error_lower = error_message.lower()

                logger.error(
                    f"[{correlation_id}] LiteLLM error: {response.status_code} - {error_message}",
                    extra={
                        "correlation_id": correlation_id,
                        "status_code": response.status_code,
                        "error": error_message,
                    },
                )

                # -------------------------------------------------------------------------
                # CLASSIFY ERROR TYPE for better user messages
                # -------------------------------------------------------------------------
                # Cloud providers return different error codes/messages:
                # - 401: Invalid API key
                # - 403: Forbidden (subscription expired, no access)
                # - 429: Rate limit exceeded
                # - 400: Bad request (often contains "credit", "billing", "quota")
                #
                # We detect these to show CLEAR messages instead of raw API errors.

                error_type = "unknown"
                user_message = error_message

                # Authentication / Subscription errors
                if response.status_code in [401, 403] or any(
                    keyword in error_lower
                    for keyword in [
                        "authentication",
                        "invalid api key",
                        "api key",
                        "unauthorized",
                        "forbidden",
                        "credit",
                        "billing",
                        "subscription",
                        "expired",
                        "quota exceeded",
                        "insufficient_quota",
                        "account",
                    ]
                ):
                    error_type = "auth_subscription"
                    user_message = (
                        "Cloud provider authentication failed. "
                        "Your API key may be invalid or your subscription may have expired. "
                        "Please check your API key in .env or try a different cloud provider."
                    )

                # Rate limiting
                elif response.status_code == 429 or "rate limit" in error_lower:
                    error_type = "rate_limit"
                    user_message = (
                        "Cloud provider rate limit exceeded. "
                        "Please wait a moment and try again, or switch to a different provider."
                    )

                # Model not found / not available
                elif "model" in error_lower and (
                    "not found" in error_lower or "does not exist" in error_lower
                ):
                    error_type = "model_not_found"
                    user_message = (
                        "The selected cloud model is not available. "
                        "Please try a different cloud provider in settings."
                    )

                raise HTTPException(
                    status_code=response.status_code,
                    detail={
                        "error_type": error_type,
                        "message": user_message,
                        "raw_error": error_message,  # Keep raw for debugging
                        "provider": "cloud",  # Will be enhanced later with actual provider
                    },
                )

            # Stream lines from LiteLLM
            # aiter_lines() yields each line as it arrives (not buffered)
            async for line in response.aiter_lines():
                # Filter out empty lines (SSE protocol allows them)
                if line.strip():
                    yield line  # Forward to streaming.py

    except httpx.TimeoutException as e:
        # Request took longer than REQUEST_TIMEOUT seconds
        logger.error(
            f"[{correlation_id}] LiteLLM request timeout after {REQUEST_TIMEOUT}s",
            extra={"correlation_id": correlation_id},
        )
        # FIX: Add exception chaining (B904)
        raise HTTPException(
            status_code=504,  # Gateway Timeout
            detail=f"Gateway timeout - request took longer than {REQUEST_TIMEOUT}s",
        ) from e

    except httpx.ConnectError as e:
        # Cannot establish connection to LiteLLM (service down or unreachable)
        logger.error(
            f"[{correlation_id}] Cannot connect to LiteLLM at {url}: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "url": url,
                "error": str(e),
            },
        )
        # Add exception chaining (B904)
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="LiteLLM gateway unavailable - service may be down",
        ) from e

    except HTTPException:
        # Re-raise HTTPExceptions (already formatted)
        raise

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(
            f"[{correlation_id}] Unexpected error communicating with LiteLLM: {str(e)}",
            exc_info=True,  # Include full traceback in logs
            extra={"correlation_id": correlation_id},
        )
        # Add exception chaining (B904)
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail=f"Unexpected error: {str(e)}",
        ) from e
