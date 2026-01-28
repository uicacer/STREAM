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

from stream.middleware.config import LITELLM_API_KEY, LITELLM_BASE_URL

logger = logging.getLogger(__name__)

# HTTP timeout in seconds
# 120s allows for slow model inference on large contexts
REQUEST_TIMEOUT = 120.0


async def forward_to_litellm(
    model: str, messages: list[dict], temperature: float, correlation_id: str
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

    Yields:
        SSE-formatted lines from LiteLLM, e.g.:
        "data: {"choices": [{"delta": {"content": "Hello"}}]}"
        "data: [DONE]"

    Raises:
        HTTPException: On connection errors, timeouts, or non-200 responses

    Example:
        >>> async for line in forward_to_litellm("gpt-4", messages, 0.7, "req-123"):
        ...     print(line)
        data: {"choices": [{"delta": {"content": "The"}}]}
        data: {"choices": [{"delta": {"content": " answer"}}]}
        data: [DONE]

    Note:
        This is an async generator - it yields lines as they arrive from LiteLLM,
        rather than buffering the entire response in memory.
    """
    # Construct the request payload
    # This matches OpenAI's API format, which LiteLLM expects
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,  # Critical: enables SSE streaming
    }

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
                error_text = await response.aread()
                error_message = error_text.decode("utf-8")

                logger.error(
                    f"[{correlation_id}] LiteLLM error: {response.status_code} - {error_message}",
                    extra={
                        "correlation_id": correlation_id,
                        "status_code": response.status_code,
                        "error": error_message,
                    },
                )

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"LiteLLM error: {error_message}",
                )

            # Stream lines from LiteLLM
            # aiter_lines() yields each line as it arrives (not buffered)
            async for line in response.aiter_lines():
                # Filter out empty lines (SSE protocol allows them)
                if line.strip():
                    yield line

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
        # FIX: Add exception chaining (B904)
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
