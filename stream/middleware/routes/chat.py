# =============================================================================
# STREAM Middleware - Chat Routes (Real Costs)
# =============================================================================
# Chat completion endpoints - main AI interaction
# =============================================================================

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from psycopg2 import pool
from pydantic import BaseModel, Field

from stream.middleware.config import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LLM_JUDGE_ENABLED,
    MODEL_CONTEXT_LIMITS,
    MODEL_COSTS,
    TIERS,
    get_model_for_tier,
    get_routing_reason,
    get_tier_for_query,
    is_tier_available,
    judge_complexity_with_keywords,
    judge_complexity_with_llm,
)

logger = logging.getLogger(__name__)

# Validate that MODEL_COSTS is the single source of truth
assert MODEL_COSTS, "MODEL_COSTS must be defined in config.py"
logger.info(f"✅ Cost configuration loaded: {len(MODEL_COSTS)} models defined")

# =============================================================================
# DATABASE CONNECTION POOL
# =============================================================================
try:
    # Validate required env vars
    required_vars = {
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
    }

    missing = [k for k, v in required_vars.items() if v is None]
    if missing:
        raise ValueError(f"Missing: {', '.join(missing)}")

    db_pool = pool.SimpleConnectionPool(
        1,
        5,
        host=required_vars["POSTGRES_HOST"],
        port=int(required_vars["POSTGRES_PORT"]),
        database=required_vars["POSTGRES_DB"],
        user=required_vars["POSTGRES_USER"],
        password=required_vars["POSTGRES_PASSWORD"],
    )
    logger.info("✅ Database connection pool created")

except ValueError as e:
    logger.critical(f"❌ CONFIG ERROR: {e} - Set in .env file")
    logger.warning("⚠️  Cost tracking disabled")
    db_pool = None

except Exception as e:
    logger.critical(f"❌ DB connection failed: {e}")
    logger.warning("⚠️  Cost tracking disabled")
    db_pool = None


# =============================================================================
# TIER FALLBACK HELPER
# =============================================================================


def get_fallback_tier(failed_tier: str, complexity: str, already_tried: list[str]) -> str | None:
    """
    Get next tier to try when current tier fails

    Args:
        failed_tier: The tier that just failed
        complexity: Query complexity (low/medium/high)
        already_tried: List of tiers already attempted

    Returns:
        Next tier to try, or None if no fallbacks available
    """

    # Define fallback chains based on complexity
    if complexity == "low" or complexity == "medium":
        fallback_chain = ["lakeshore", "cloud", "local"]
    else:  # high
        fallback_chain = ["cloud", "lakeshore", "local"]

    # Try each tier in order (excluding already tried)
    for tier in fallback_chain:
        if tier not in already_tried and is_tier_available(tier):
            return tier

    return None


# =============================================================================
# API ROUTER
# =============================================================================

router = APIRouter()

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class Message(BaseModel):
    """Chat message"""

    role: str = Field(..., description="Role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)"""

    model: str = Field(default="auto", description="Model or tier to use")
    messages: list[Message] = Field(..., description="Conversation messages")
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool | None = Field(default=False)
    user: str | None = Field(default=None, description="User ID (future)")


class ChatCompletionResponse(BaseModel):
    """Chat completion response"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int] | None = None

    # STREAM-specific metadata
    stream_metadata: dict[str, Any] | None = None


def calculate_query_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost using config-based pricing"""

    if model not in MODEL_COSTS:
        logger.warning(f"⚠️ No cost data for model: {model}")
        return 0.0

    costs = MODEL_COSTS[model]
    input_cost = input_tokens * costs["input"]
    output_cost = output_tokens * costs["output"]

    return input_cost + output_cost


async def create_streaming_response(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
    tier: str,
    user_query: str = "",  # ← ADD THIS
    complexity: str = "medium",  # ← ADD THIS
):
    """Create async generator for streaming SSE responses with automatic fallback"""

    input_tokens = 0
    output_tokens = 0
    tiers_tried = [tier]
    current_tier = tier
    current_model = model

    # **SEND METADATA AS VERY FIRST EVENT**
    metadata_event = {
        "stream_metadata": {
            "tier": current_tier,
            "model": current_model,
            "correlation_id": correlation_id,
        }
    }
    yield f"data: {json.dumps(metadata_event)}\n\n"

    max_retries = 3  # Try up to 3 tiers

    for attempt in range(max_retries):
        try:
            # Get the async generator from forward_to_litellm
            litellm_stream = await forward_to_litellm(
                model=current_model,
                messages=messages,
                temperature=temperature,
                stream=True,
                correlation_id=correlation_id,
            )

            # Forward all chunks
            async for line in litellm_stream:
                yield f"{line}\n\n"

                # Parse for token tracking
                if line.startswith("data: "):
                    try:
                        data_str = line[6:]
                        if data_str != "[DONE]":
                            data = json.loads(data_str)
                            if "usage" in data:
                                input_tokens = data["usage"].get("prompt_tokens", 0)
                                output_tokens = data["usage"].get("completion_tokens", 0)
                    except json.JSONDecodeError:
                        pass

            # SUCCESS! Send final cost event
            if input_tokens > 0 or output_tokens > 0:
                cost = calculate_query_cost(current_model, input_tokens, output_tokens)

                cost_event = {
                    "stream_metadata": {
                        "tier": current_tier,
                        "model": current_model,
                        "fallback_used": len(tiers_tried) > 1,
                        "tiers_tried": tiers_tried,
                        "cost": {
                            "total": cost,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                    }
                }
                yield f"data: {json.dumps(cost_event)}\n\n"

                if len(tiers_tried) > 1:
                    logger.warning(
                        f"[{correlation_id}] Fallback successful: {tiers_tried[0]} → {current_tier}"
                    )

                logger.info(
                    f"[{correlation_id}] Stream completed: cost=${cost:.6f}, tokens={input_tokens + output_tokens}",
                    extra={
                        "correlation_id": correlation_id,
                        "tier": current_tier,
                        "cost": cost,
                        "tokens": input_tokens + output_tokens,
                    },
                )

            # Success - exit retry loop
            return

        except HTTPException as e:
            # Check if it's a connection/availability error
            error_str = str(e.detail).lower()
            is_tier_failure = any(
                [
                    "connection" in error_str,
                    "unavailable" in error_str,
                    "500" in str(e.status_code),
                    "503" in str(e.status_code),
                    "504" in str(e.status_code),
                ]
            )

            if is_tier_failure and attempt < max_retries - 1:
                # Try fallback tier
                logger.warning(f"[{correlation_id}] {current_tier.upper()} failed: {e.detail}")

                # Get next tier to try
                fallback_tier = get_fallback_tier(current_tier, complexity, tiers_tried)

                if fallback_tier:
                    logger.info(f"[{correlation_id}] Attempting fallback: {fallback_tier.upper()}")

                    # Update for next attempt
                    current_tier = fallback_tier
                    current_model = get_model_for_tier(fallback_tier)
                    tiers_tried.append(fallback_tier)

                    # Send fallback notification to client
                    fallback_event = {
                        "stream_metadata": {
                            "fallback": True,
                            "original_tier": tier,
                            "current_tier": current_tier,
                            "model": current_model,
                            "reason": "Connection error",
                        }
                    }
                    yield f"data: {json.dumps(fallback_event)}\n\n"

                    # Continue to next attempt
                    continue
                else:
                    # No more fallbacks available
                    logger.error(
                        f"[{correlation_id}] No fallback tiers available",
                        extra={"correlation_id": correlation_id},
                    )
                    yield f'data: {{"error": "All AI tiers unavailable. Tried: {", ".join(tiers_tried)}"}}\n\n'
                    return
            else:
                # Not a tier failure or last retry - propagate error
                logger.error(
                    f"[{correlation_id}] Streaming error: {str(e)}",
                    exc_info=True,
                    extra={"correlation_id": correlation_id},
                )
                yield f'data: {{"error": "{str(e)}"}}\n\n'
                return

        except Exception as e:
            logger.error(
                f"[{correlation_id}] Unexpected streaming error: {str(e)}",
                exc_info=True,
                extra={"correlation_id": correlation_id},
            )
            yield f'data: {{"error": "{str(e)}"}}\n\n'
            return


# =============================================================================
# CHAT COMPLETION ENDPOINT
# =============================================================================
@router.post("/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, request: Request):
    """Chat completion endpoint with streaming support and automatic fallback"""

    correlation_id = request.state.correlation_id

    # Extract user query
    user_messages = [msg for msg in request_body.messages if msg.role == "user"]
    user_query = user_messages[-1].content if user_messages else ""

    # Get complexity (needed for intelligent fallback)
    if LLM_JUDGE_ENABLED:
        complexity = judge_complexity_with_llm(user_query)
        if not complexity:
            complexity = judge_complexity_with_keywords(user_query)
    else:
        complexity = judge_complexity_with_keywords(user_query)

    # Determine routing
    tier = get_tier_for_query(user_query, request_body.model)
    model = get_model_for_tier(tier)

    # Prepare message list
    messages = [msg.model_dump() for msg in request_body.messages]

    # CONTEXT WINDOW CHECK
    # Estimate input tokens
    def estimate_tokens(msgs):
        return sum(len(str(m.get("content", ""))) // 4 for m in msgs)

    estimated_input = estimate_tokens(messages)

    # Check if we'll exceed context for this tier
    if tier in ["lakeshore", "local"]:
        model_config = MODEL_CONTEXT_LIMITS.get(model)
        if model_config:
            max_input = model_config["total"] - model_config["reserve_output"]

            if estimated_input > max_input:
                # DON'T truncate or reroute - return helpful error!
                logger.error(
                    f"[{correlation_id}] Context window exceeded: "
                    f"{estimated_input} tokens > {max_input} limit for {model}"
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "context_too_long",
                        "message": f"Conversation history ({estimated_input} tokens) exceeds {model} limit ({max_input} tokens)",
                        "suggestion": "Please start a new conversation or use Cloud tier for longer contexts",
                        "estimated_tokens": estimated_input,
                        "model_limit": max_input,
                    },
                ) from None

    # Log routing decision
    logger.info(
        f"[{correlation_id}] Routing: tier={tier}, model={model}, complexity={complexity}, stream={request_body.stream}",
        extra={
            "correlation_id": correlation_id,
            "tier": tier,
            "model": model,
            "complexity": complexity,
            "stream": request_body.stream,
        },
    )

    try:
        if request_body.stream:
            # Streaming response WITH FALLBACK
            return StreamingResponse(
                create_streaming_response(
                    model=model,
                    messages=messages,
                    temperature=request_body.temperature,
                    correlation_id=correlation_id,
                    tier=tier,
                    user_query=user_query,  # ← ADD THIS
                    complexity=complexity,  # ← ADD THIS
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        else:
            # Non-streaming response
            litellm_response = await forward_to_litellm(
                model=model,
                messages=messages,
                temperature=request_body.temperature,
                stream=False,
                correlation_id=correlation_id,
            )

            # Calculate cost
            usage = litellm_response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            actual_cost = calculate_query_cost(model, input_tokens, output_tokens)

            # Add metadata
            if isinstance(litellm_response, dict):
                litellm_response["stream_metadata"] = {
                    "tier": tier,
                    "tier_name": TIERS[tier]["name"],
                    "routing_reason": get_routing_reason(user_query, request_body.model, tier),
                    "correlation_id": correlation_id,
                    "cost": {
                        "total": actual_cost,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "input_cost_per_token": MODEL_COSTS.get(model, {}).get("input", 0.0),
                        "output_cost_per_token": MODEL_COSTS.get(model, {}).get("output", 0.0),
                    },
                }

            logger.info(
                f"[{correlation_id}] Request completed: cost=${actual_cost:.6f}, tokens={total_tokens}",
                extra={
                    "correlation_id": correlation_id,
                    "tier": tier,
                    "status": "success",
                    "cost": actual_cost,
                    "tokens": total_tokens,
                },
            )

            return litellm_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[{correlation_id}] Error processing request: {str(e)}",
            exc_info=True,
            extra={"correlation_id": correlation_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        ) from e  # Exception chaining


# =============================================================================
# COST ENDPOINTS
# =============================================================================


@router.get("/costs/models")
async def get_model_costs_endpoint():
    """
    Get cost information (now from config, not LiteLLM API)
    """
    return {
        "success": True,
        "costs": MODEL_COSTS,
        "source": "config.py",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/costs/summary")
async def get_cost_summary(days: int = 7):
    """
    Get cost summary from LiteLLM database

    Args:
        days: Number of days to look back (default 7)

    ✅ FIXED: Proper error handling with status codes
    """

    conn = None

    try:
        if not db_pool:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database pool not available",
            ) from None

        conn = db_pool.getconn()
        cur = conn.cursor()

        # Query last N days
        start_date = datetime.now() - timedelta(days=days)

        cur.execute(
            """
            SELECT
                model,
                COUNT(*) as requests,
                SUM(spend) as total_cost,
                SUM(prompt_tokens) as input_tokens,
                SUM(completion_tokens) as output_tokens
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" >= %s
            GROUP BY model
            ORDER BY total_cost DESC
        """,
            (start_date,),
        )

        results = cur.fetchall()

        # Format response
        summary = {
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "models": [],
        }

        total_cost = 0.0
        total_requests = 0

        for row in results:
            model_data = {
                "model": row[0],
                "requests": row[1],
                "cost": float(row[2]) if row[2] else 0.0,
                "input_tokens": row[3] or 0,
                "output_tokens": row[4] or 0,
                "avg_cost_per_request": (float(row[2]) / row[1]) if row[1] > 0 and row[2] else 0.0,
            }
            summary["models"].append(model_data)
            total_cost += model_data["cost"]
            total_requests += model_data["requests"]

        summary["total_cost"] = total_cost
        summary["total_requests"] = total_requests
        summary["avg_cost_per_request"] = total_cost / total_requests if total_requests > 0 else 0.0

        cur.close()

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error in cost_summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}"
        ) from e  # Exception chaining

    finally:
        if conn:
            db_pool.putconn(conn)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def forward_to_litellm(
    model: str, messages: list[dict], temperature: float, stream: bool, correlation_id: str
):
    """
    Forward request to LiteLLM gateway with streaming support

    Returns:
        - If stream=False: Dict (JSON response)
        - If stream=True: Returns async generator function
    """
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": stream}

    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
        "X-Correlation-ID": correlation_id,
    }

    url = f"{LITELLM_BASE_URL}/v1/chat/completions"

    logger.debug(
        f"[{correlation_id}] Forwarding to LiteLLM: {url}, stream={stream}",
        extra={"correlation_id": correlation_id},
    )

    if not stream:
        # ========== NON-STREAMING MODE ==========
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)

                logger.debug(
                    f"[{correlation_id}] LiteLLM response: status={response.status_code}",
                    extra={"correlation_id": correlation_id},
                )

                if response.status_code != 200:
                    logger.error(
                        f"[{correlation_id}] LiteLLM error: {response.status_code} - {response.text}",
                        extra={"correlation_id": correlation_id},
                    )
                    raise HTTPException(
                        status_code=response.status_code, detail=f"LiteLLM error: {response.text}"
                    ) from None

                try:
                    return response.json()
                except Exception as json_err:
                    logger.error(
                        f"[{correlation_id}] Failed to parse LiteLLM response",
                        extra={"correlation_id": correlation_id},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Invalid JSON response from LiteLLM: {str(json_err)}",
                    ) from json_err

        except httpx.TimeoutException:
            logger.error(
                f"[{correlation_id}] LiteLLM timeout", extra={"correlation_id": correlation_id}
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gateway timeout - request took too long",
            ) from None

        except httpx.ConnectError:
            logger.error(
                f"[{correlation_id}] Cannot connect to LiteLLM at {url}",
                extra={"correlation_id": correlation_id},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LiteLLM gateway unavailable",
            ) from None
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"[{correlation_id}] Unexpected error: {str(e)}",
                exc_info=True,
                extra={"correlation_id": correlation_id},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {str(e)}",
            ) from e  # Exception chaining

    else:
        # ========== STREAMING MODE - Return async generator ==========
        async def stream_lines():
            try:
                async with (
                    httpx.AsyncClient(timeout=120.0) as client,
                    client.stream("POST", url, json=payload, headers=headers) as response,
                ):
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(
                            f"[{correlation_id}] LiteLLM error: {response.status_code}",
                            extra={"correlation_id": correlation_id},
                        )
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"LiteLLM error: {error_text.decode()}",
                        ) from None

                    # Yield each line from the stream
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield line

            except httpx.TimeoutException:
                logger.error(
                    f"[{correlation_id}] LiteLLM timeout", extra={"correlation_id": correlation_id}
                )
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Gateway timeout"
                ) from None
            except httpx.ConnectError:
                logger.error(
                    f"[{correlation_id}] Cannot connect to LiteLLM at {url}",
                    extra={"correlation_id": correlation_id},
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="LiteLLM gateway unavailable",
                ) from None
            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    f"[{correlation_id}] Unexpected error: {str(e)}",
                    exc_info=True,
                    extra={"correlation_id": correlation_id},
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected error: {str(e)}",
                ) from e  # Exception chaining

        return stream_lines()


@router.get("/context/limits")
async def get_context_limits():
    """
    Get context window limits for all tiers

    Returns limits organized by tier for UI display
    """
    from stream.middleware.config import get_tier_context_limits

    return {
        "success": True,
        "limits": get_tier_context_limits(),
        "timestamp": datetime.now(UTC).isoformat(),
    }
