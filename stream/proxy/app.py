"""
Lakeshore vLLM Proxy Service - Routes requests via Globus Compute or SSH
"""

import json
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from stream.middleware.core.globus_compute_client import GlobusComputeClient

# Configuration
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

USE_GLOBUS_COMPUTE = os.getenv("USE_GLOBUS_COMPUTE", "true").lower() == "true"
GLOBUS_COMPUTE_ENDPOINT_ID = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://ga-001:8000")
LAKESHORE_VLLM_ENDPOINT = os.getenv("LAKESHORE_VLLM_ENDPOINT", "http://host.docker.internal:8000")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Lakeshore vLLM Proxy")

# Initialize Globus client
globus_client = None
if USE_GLOBUS_COMPUTE and GLOBUS_COMPUTE_ENDPOINT_ID:
    try:
        globus_client = GlobusComputeClient()  # Reads from env vars
        logger.info("✓ Globus Compute client loaded")
    except Exception as e:
        logger.error(f"Failed to initialize Globus Compute client: {e}")


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Lakeshore vLLM Proxy Starting")
    logger.info("=" * 60)
    logger.info(f"Mode: {'Globus Compute' if USE_GLOBUS_COMPUTE else 'SSH Port Forward'}")
    if USE_GLOBUS_COMPUTE and globus_client:
        logger.info(f"Globus Endpoint: {GLOBUS_COMPUTE_ENDPOINT_ID}")
        logger.info(f"vLLM Server URL: {VLLM_SERVER_URL}")
    logger.info(f"Listening on: {PROXY_HOST}:{PROXY_PORT}")
    logger.info("=" * 60)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Lakeshore vLLM Proxy",
        "mode": "globus_compute" if USE_GLOBUS_COMPUTE else "ssh",
        "globus_configured": bool(globus_client and globus_client.is_available())
        if USE_GLOBUS_COMPUTE
        else False,
    }


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    model = body.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 512)

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
            raise HTTPException(
                status_code=503, detail=f"Lakeshore inference failed: {result['error']}"
            )

        logger.info("Globus Compute inference successful")

        if stream:
            return _convert_json_to_sse_stream(result)
        return result

    except HTTPException:
        raise
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
    """Convert chat.completion (message format) to chat.completion.chunk (delta format) SSE stream"""

    async def sse_generator():
        choices = json_response.get("choices", [])
        if not choices:
            yield "data: [DONE]\n\n"
            return

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        role = message.get("role", "assistant")

        # Chunk 1: Role
        if role:
            chunk = {
                "id": json_response.get("id", ""),
                "object": "chat.completion.chunk",
                "created": json_response.get("created", 0),
                "model": json_response.get("model", ""),
                "choices": [
                    {"index": 0, "delta": {"role": role, "content": ""}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Chunk 2: Content
        if content:
            chunk = {
                "id": json_response.get("id", ""),
                "object": "chat.completion.chunk",
                "created": json_response.get("created", 0),
                "model": json_response.get("model", ""),
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Chunk 3: Finish
        chunk = {
            "id": json_response.get("id", ""),
            "object": "chat.completion.chunk",
            "created": json_response.get("created", 0),
            "model": json_response.get("model", ""),
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}
            ],
            "usage": json_response.get("usage", {}),
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


def main():
    import uvicorn

    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
