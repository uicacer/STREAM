"""
Streaming response orchestration with automatic tier fallback.

This is the core streaming logic for STREAM. It manages:
1. Server-Sent Events (SSE) formatting
2. Automatic tier fallback on failures
3. Token usage tracking
4. Real-time cost calculation
5. Client notifications (metadata, fallback events, costs)

The streaming approach provides a ChatGPT-like experience where users see
the response being generated token-by-token in real-time.

Architecture:
    User → chat.py (endpoint) → streaming.py (orchestration) → litellm_client.py → LiteLLM
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import HTTPException

from stream.middleware.config import TIER_TIMEOUT_WARNING
from stream.middleware.core.litellm_client import forward_to_litellm
from stream.middleware.core.metrics import MetricsTracker
from stream.middleware.core.query_router import get_model_for_tier
from stream.middleware.utils.cost_calculator import calculate_query_cost
from stream.middleware.utils.fallback import get_fallback_reason, get_fallback_tier
from stream.middleware.utils.token_estimator import estimate_tokens, estimate_tokens_from_text

logger = logging.getLogger(__name__)

# Maximum number of tier fallback attempts
# This prevents infinite retry loops while still giving reasonable coverage
MAX_FALLBACK_ATTEMPTS = 3

# Chunk gap warning threshold (seconds)
# If no chunk arrives within this time, warn the user that something's slow
CHUNK_GAP_WARNING_SECONDS = 5.0


async def stream_with_gap_warnings(
    generator: AsyncGenerator[str, None],
    tier: str,
    correlation_id: str,
) -> AsyncGenerator[str, None]:
    """
    Wrap a streaming generator to detect and warn about long gaps between chunks.

    Why this matters for UX:
    ------------------------
    When streaming pauses (cloud provider hiccup, rate limiting, network issue),
    users see the UI freeze with no feedback. This feels broken.

    With gap warnings, after 5 seconds of no data, we send:
    ⚠️ "Cloud is taking longer than usual, please wait..."

    This tells the user "we know it's slow, we're working on it" - building trust.

    How it works:
    -------------
    1. Wait for next chunk with a 5-second timeout
    2. If chunk arrives → yield it, reset timer
    3. If timeout → send warning event, then continue waiting
    4. The underlying httpx 10s read timeout is the hard limit

    Args:
        generator: The original streaming generator (from forward_to_litellm)
        tier: Current tier name (for warning message)
        correlation_id: Request ID for logging

    Yields:
        Original chunks from generator, plus warning events during long gaps
    """
    warning_sent = False
    async_gen = generator.__aiter__()

    while True:
        try:
            # Wait for next chunk, but timeout after 5 seconds
            chunk = await asyncio.wait_for(
                async_gen.__anext__(),
                timeout=CHUNK_GAP_WARNING_SECONDS,
            )
            # Reset warning flag when we get data (gap ended)
            if warning_sent:
                warning_sent = False
            yield chunk

        except TimeoutError:
            # No chunk for 5 seconds - send warning (once per gap)
            if not warning_sent:
                warning_sent = True
                warning_event = {
                    "stream_metadata": {
                        "warning": "slow_stream",
                        "tier": tier,
                        "message": f"{tier.title()} is taking longer than usual, please wait...",
                    }
                }
                # Yield in same format as LiteLLM chunks (main loop adds \n\n)
                yield f"data: {json.dumps(warning_event)}"
                logger.warning(
                    f"[{correlation_id}] Chunk gap warning: no data for {CHUNK_GAP_WARNING_SECONDS}s on {tier}",
                    extra={"correlation_id": correlation_id, "tier": tier},
                )
            # Continue waiting - the httpx 10s timeout will eventually catch true hangs

        except StopAsyncIteration:
            # Generator exhausted - we're done
            break


async def create_streaming_response(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
    tier: str,
    user_query: str = "",
    complexity: str = "medium",
    judge_fallback_info: dict | None = None,
    routing_fallback_info: dict | None = None,
    judge_cost: float = 0.0,
) -> AsyncGenerator[str, None]:
    """
    Create a streaming Server-Sent Events (SSE) response with metrics tracking and automatic fallback.

    This is the main streaming orchestrator. It:
    1. Sends initial metadata to the client (which tier/model being used)
    2. Streams the LLM response token-by-token
    3. Tracks token usage for cost calculation
    4. Handles failures by automatically trying fallback tiers
    5. Sends final cost summary when complete

    SSE Format:
        Every event is formatted as:
        "data: <JSON>\n\n"

        The double newline is required by SSE specification.
        Browsers parse this automatically via EventSource API.

    Args:
        model: Initial model to try (e.g., "llama3.2:3b")
        messages: Conversation history (list of role/content dicts)
        temperature: Sampling temperature (0.0-2.0)
        correlation_id: Unique request ID for logging and tracing
        tier: Initial tier to try ("local", "lakeshore", or "cloud")
        user_query: The user's query text (for complexity re-evaluation)
        complexity: Query complexity ("low", "medium", or "high")

    Yields:
        SSE-formatted strings containing:
        - Initial metadata (tier, model, correlation_id)
        - Content chunks (the actual AI response, token by token)
        - Fallback notifications (if tier switching occurs)
        - Final cost summary (tokens used, total cost)
        - Error messages (if all tiers fail)

    Example Flow (Successful):
        yield "data: {"stream_metadata": {"tier": "local", "model": "llama3.2:3b"}}\n\n"
        yield "data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n"
        yield "data: {"choices": [{"delta": {"content": " world"}}]}\n\n"
        yield "data: {"stream_metadata": {"cost": {"total": 0.0}}}\n\n"

    Example Flow (With Fallback):
        yield "data: {"stream_metadata": {"tier": "local", ...}}\n\n"
        [local fails]
        yield "data: {"stream_metadata": {"fallback": true, "current_tier": "cloud"}}\n\n"
        yield "data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n"
        ...

    Note:
        This function implements exponential backoff implicitly through the
        fallback chain (local → lakeshore → cloud), where each tier is
        progressively more reliable.
    """

    # Initialize metrics tracker
    tracker = MetricsTracker()
    tracker.start_request(correlation_id, tier, model, complexity)

    # Initialize tracking variables
    input_tokens = 0  # Tokens in the prompt/conversation
    output_tokens = 0  # Tokens in the model's response
    output_text = []  # Track generated text for estimation
    tiers_tried = [tier]  # Track which tiers we've attempted
    current_tier = tier  # The tier we're currently trying
    current_model = model  # The model we're currently trying

    # =========================================================================
    # STEP 1: Send Initial Metadata
    # =========================================================================
    # Send metadata FIRST so the client knows what's happening before
    # the actual response starts streaming
    metadata_event = {
        "stream_metadata": {
            "tier": current_tier,
            "model": current_model,
            "correlation_id": correlation_id,
            "complexity": complexity,  # Actual query complexity for UI display
        }
    }

    # Include judge fallback info if LLM judge failed
    if judge_fallback_info:
        metadata_event["stream_metadata"]["judge_fallback"] = judge_fallback_info

    # -------------------------------------------------------------------------
    # PRE-ROUTING FALLBACK vs RUNTIME FALLBACK
    # -------------------------------------------------------------------------
    # There are TWO types of fallback in STREAM:
    #
    # 1. PRE-ROUTING FALLBACK (handled here via routing_fallback_info):
    #    - Happens BEFORE any API call is made
    #    - The query_router checks tier health status and finds the preferred
    #      tier is unavailable (e.g., Lakeshore health check failed)
    #    - Router silently selects the next tier in the fallback chain
    #    - Example: User in auto mode, medium query → should go to Lakeshore
    #               but Lakeshore is marked unhealthy → routes to Cloud instead
    #    - The streaming code receives the final tier (Cloud) but needs to
    #      notify the user that their preferred tier was skipped
    #
    # 2. RUNTIME FALLBACK (handled in STEP 2 below):
    #    - Happens DURING the API call attempt
    #    - The tier was selected and we tried to call it, but it failed
    #      (connection refused, timeout, HTTP error, etc.)
    #    - We catch the exception and try the next tier in the fallback chain
    #    - Example: Lakeshore was healthy at routing time but the actual
    #               request to vLLM timed out → fall back to Cloud
    #    - A separate fallback event is yielded to notify the client in real-time
    #
    # Both types should show the same user-facing message:
    # "Lakeshore unavailable — using Cloud instead"
    # -------------------------------------------------------------------------
    if routing_fallback_info:
        metadata_event["stream_metadata"]["fallback"] = True
        metadata_event["stream_metadata"]["original_tier"] = routing_fallback_info["original_tier"]
        metadata_event["stream_metadata"]["fallback_used"] = True
        metadata_event["stream_metadata"]["unavailable_tiers"] = routing_fallback_info.get(
            "unavailable_tiers", []
        )

    # Format as SSE: "data: <JSON>\n\n"
    yield f"data: {json.dumps(metadata_event)}\n\n"

    logger.info(
        f"[{correlation_id}] Starting stream: tier={current_tier}, model={current_model}",
        extra={
            "correlation_id": correlation_id,
            "tier": current_tier,
            "model": current_model,
        },
    )

    # =========================================================================
    # STEP 2: Retry Loop with Automatic Fallback
    # =========================================================================
    # Track time for timeout warnings
    stream_start_time = time.perf_counter()
    timeout_warning_sent = False

    for attempt in range(MAX_FALLBACK_ATTEMPTS):
        try:
            first_chunk = True  # Track if this is the first chunk for TTFT
            logger.debug(
                f"[{correlation_id}] Attempt {attempt + 1}/{MAX_FALLBACK_ATTEMPTS} on {current_tier}",
                extra={"correlation_id": correlation_id, "attempt": attempt + 1},
            )

            # ---------------------------------------------------------------------
            # Stream from LiteLLM (with gap warnings)
            # ---------------------------------------------------------------------
            # All tiers (local, lakeshore, cloud) route through LiteLLM gateway
            # For Lakeshore: LiteLLM → Lakeshore Proxy → Globus Compute OR SSH
            # For Local: LiteLLM → Ollama
            # For Cloud: LiteLLM → Anthropic/OpenAI APIs
            #
            # stream_with_gap_warnings wraps the generator to detect long pauses
            # and warn the user in real-time (after 5s of no data)
            raw_stream = forward_to_litellm(
                model=current_model,
                messages=messages,
                temperature=temperature,
                correlation_id=correlation_id,
            )
            async for line in stream_with_gap_warnings(raw_stream, current_tier, correlation_id):
                # Record TTFT on first chunk
                if first_chunk:
                    tracker.record_first_token()
                    first_chunk = False

                # ---------------------------------------------------------------------
                # TIMEOUT WARNING: Check if response is taking too long
                # ---------------------------------------------------------------------
                # Send a warning event (once) if elapsed time exceeds tier threshold
                if not timeout_warning_sent:
                    elapsed_seconds = time.perf_counter() - stream_start_time
                    timeout_threshold = TIER_TIMEOUT_WARNING.get(current_tier, 30)

                    if elapsed_seconds >= timeout_threshold:
                        timeout_warning_sent = True
                        warning_event = {
                            "stream_metadata": {
                                "warning": "timeout",
                                "tier": current_tier,
                                "elapsed_seconds": round(elapsed_seconds, 1),
                                "message": f"{current_tier.title()} is taking longer than usual ({int(elapsed_seconds)}s)",
                            }
                        }
                        yield f"data: {json.dumps(warning_event)}\n\n"
                        logger.warning(
                            f"[{correlation_id}] Timeout warning: {current_tier} took {elapsed_seconds:.1f}s",
                            extra={
                                "correlation_id": correlation_id,
                                "tier": current_tier,
                                "elapsed_seconds": elapsed_seconds,
                            },
                        )

                # ---------------------------------------------------------------------
                # IMPORTANT: Intercept [DONE] marker - don't forward it yet!
                # We need to send cost metadata BEFORE [DONE] so SDK can process it.
                # ---------------------------------------------------------------------
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        # Don't yield [DONE] yet - we'll send it after cost metadata
                        continue

                # Forward the line to the client immediately
                # This provides real-time streaming (no buffering)
                yield f"{line}\n\n"  # Add extra newline for SSE format

                # ---------------------------------------------------------------------
                # Parse Token Usage (for cost tracking)
                # ---------------------------------------------------------------------
                # LiteLLM sends token usage in the stream (OpenAI format)
                # Format: data: {"usage": {"prompt_tokens": X, "completion_tokens": Y}}
                if line.startswith("data: "):
                    try:
                        # Extract JSON payload (skip "data: " prefix and trim whitespace)
                        data_str = line[6:].strip()

                        # [DONE] already handled above
                        if data_str == "[DONE]":
                            continue

                        # Skip empty
                        if not data_str:
                            continue

                        # Parse JSON
                        data = json.loads(data_str)

                        # Method 1: Look for usage in standard OpenAI format
                        if "usage" in data and data["usage"]:
                            usage = data["usage"]
                            if usage.get("prompt_tokens") is not None:
                                input_tokens = usage.get("prompt_tokens", 0)
                                output_tokens = usage.get("completion_tokens", 0)
                                logger.debug(f"[{correlation_id}] Tokens from usage object")
                                # Found tokens - don't collect text!

                        # Method 2: Some providers put tokens at top level (not nested in "usage")
                        elif "prompt_tokens" in data:
                            input_tokens = data.get("prompt_tokens", 0)
                            output_tokens = data.get("completion_tokens", 0)
                            logger.debug(f"[{correlation_id}] Tokens from top-level")
                            # Found tokens - don't collect text!

                        # Method 3: Collect text ONLY if we didn't get tokens (Extract content for token estimation)
                        else:
                            # Only collect if we haven't found usage yet
                            if "choices" in data:
                                for choice in data["choices"]:
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            output_text.append(content)

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        logger.warning(
                            f"[{correlation_id}] Error parsing usage: {e}",
                            extra={"correlation_id": correlation_id},
                        )

            # ---------------------------------------------------------------------
            # SUCCESS! Stream completed without errors. Now, calculate cost with fallback estimation
            # ---------------------------------------------------------------------
            # If we didn't get token counts from stream, estimate them
            if output_tokens == 0 and output_text:
                full_output = "".join(output_text)
                output_tokens = estimate_tokens_from_text(full_output)

                logger.info(
                    f"[{correlation_id}] Estimated output tokens from text: {output_tokens}",
                    extra={"correlation_id": correlation_id, "output_tokens": output_tokens},
                )

            if input_tokens == 0:
                input_tokens = estimate_tokens(messages)

                logger.info(
                    f"[{correlation_id}] Estimated input tokens: {input_tokens}",
                    extra={"correlation_id": correlation_id, "input_tokens": input_tokens},
                )

            # Calculate cost only once (now using centralized cost_reader!)
            inference_cost = calculate_query_cost(current_model, input_tokens, output_tokens)
            total_cost = inference_cost + judge_cost  # Include judge cost in total
            tracker.record_tokens(input_tokens, output_tokens)
            tracker.record_completion(total_cost)

            # Send final cost summary if we have token counts
            if input_tokens > 0 or output_tokens > 0:
                # Determine if ANY fallback occurred:
                # - Runtime fallback: len(tiers_tried) > 1 (tier failed during API call)
                # - Pre-routing fallback: routing_fallback_info is not None (tier was unavailable before trying)
                runtime_fallback = len(tiers_tried) > 1
                prerouting_fallback = routing_fallback_info is not None
                any_fallback = runtime_fallback or prerouting_fallback

                cost_event = {
                    "stream_metadata": {
                        "tier": current_tier,
                        "model": current_model,
                        "complexity": complexity,  # Actual query complexity
                        "fallback_used": any_fallback,
                        "tiers_tried": tiers_tried,  # Which tiers did we try?
                        "cost": {
                            "total": total_cost,  # Total includes inference + judge
                            "inference_cost": inference_cost,  # LLM response cost
                            "judge_cost": judge_cost,  # Complexity judge cost (Haiku, etc.)
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                    }
                }

                # Include original_tier in final event for pre-routing fallback
                # This ensures the frontend can display the fallback warning after streaming
                if prerouting_fallback:
                    cost_event["stream_metadata"]["original_tier"] = routing_fallback_info[
                        "original_tier"
                    ]

                yield f"data: {json.dumps(cost_event)}\n\n"

            # Send [DONE] AFTER cost metadata so SDK can process cost first
            yield "data: [DONE]\n\n"

            # Log success metrics
            if len(tiers_tried) > 1:
                logger.warning(
                    f"[{correlation_id}] Fallback successful: {tiers_tried[0]} → {current_tier}",
                    extra={
                        "correlation_id": correlation_id,
                        "tiers_tried": tiers_tried,
                    },
                )

            judge_cost_str = f" (incl. ${judge_cost:.6f} judge)" if judge_cost > 0 else ""
            logger.info(
                f"[{correlation_id}] Stream completed successfully: "
                f"cost=${total_cost:.6f}{judge_cost_str}, tokens={input_tokens + output_tokens}",
                extra={
                    "correlation_id": correlation_id,
                    "tier": current_tier,
                    "total_cost": total_cost,
                    "inference_cost": inference_cost,
                    "judge_cost": judge_cost,
                    "total_tokens": input_tokens + output_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

            # Exit retry loop - we succeeded!
            return

        except HTTPException as e:
            # ---------------------------------------------------------------------
            # FAILURE - Determine if we should try fallback
            # ---------------------------------------------------------------------

            # Check if this is an authentication error (needs user action)
            error_str = str(e.detail).lower()
            is_auth_error = any(
                keyword in error_str
                for keyword in ["authentication", "401", "auth required", "globus"]
            )

            # Check if this is a tier availability issue (vs. user error)
            # NOTE: Auth errors are NOT tier failures - they need user action
            # so we don't fallback, we prompt the user to authenticate
            is_tier_failure = any(
                keyword in error_str
                for keyword in ["connection", "unavailable", "timeout", "500", "503", "504"]
            )

            # Should we attempt fallback?
            # Yes if: (1) it's a tier failure AND (2) we haven't exhausted retries
            if is_tier_failure and attempt < MAX_FALLBACK_ATTEMPTS - 1:
                logger.warning(
                    f"[{correlation_id}] {current_tier.upper()} tier failed: {e.detail}",
                    extra={
                        "correlation_id": correlation_id,
                        "failed_tier": current_tier,
                        "error": str(e.detail),
                    },
                )

                # Get next tier to try
                fallback_tier = get_fallback_tier(complexity, tiers_tried)

                if fallback_tier:
                    # Found a viable fallback tier
                    logger.info(
                        f"[{correlation_id}] Attempting fallback to: {fallback_tier.upper()}",
                        extra={
                            "correlation_id": correlation_id,
                            "fallback_tier": fallback_tier,
                        },
                    )

                    # Record fallback in metrics
                    tracker.record_fallback(fallback_tier)

                    # Update for next attempt
                    current_tier = fallback_tier
                    current_model = get_model_for_tier(fallback_tier)
                    tiers_tried.append(fallback_tier)

                    # Reset timeout tracking for new tier
                    stream_start_time = time.perf_counter()
                    timeout_warning_sent = False

                    # Notify client of fallback
                    # This shows in the UI: "Local unavailable, trying cloud..."
                    fallback_event = {
                        "stream_metadata": {
                            "fallback": True,
                            "original_tier": tier,
                            "current_tier": current_tier,
                            "model": current_model,
                            "complexity": complexity,  # Preserve original complexity
                            "reason": get_fallback_reason(e),
                        }
                    }
                    yield f"data: {json.dumps(fallback_event)}\n\n"

                    # Continue to next attempt in the retry loop
                    continue

                else:
                    # No more fallback tiers available
                    logger.error(
                        f"[{correlation_id}] No fallback tiers available",
                        extra={
                            "correlation_id": correlation_id,
                            "tiers_tried": tiers_tried,
                        },
                    )

                    # Record error
                    tracker.record_error(f"All tiers unavailable: {', '.join(tiers_tried)}")

                    # Send error to client
                    error_message = f"All AI tiers unavailable. Tried: {', '.join(tiers_tried)}"
                    yield f'data: {{"error": "{error_message}"}}\n\n'
                    return

            else:
                # Either not a tier failure (e.g., bad request)
                # OR we've exhausted all retry attempts
                logger.error(
                    f"[{correlation_id}] Streaming error (no fallback): {str(e)}",
                    exc_info=True,
                    extra={
                        "correlation_id": correlation_id,
                        "is_tier_failure": is_tier_failure,
                        "is_auth_error": is_auth_error,
                        "attempt": attempt,
                    },
                )

                # Record error
                tracker.record_error(str(e.detail))

                # Send error to client with auth_required flag if applicable
                error_response = {
                    "error": str(e.detail),
                    "tier": current_tier,
                    "model": current_model,
                }
                if is_auth_error:
                    error_response["auth_required"] = True

                yield f"data: {json.dumps(error_response)}\n\n"
                return

        except Exception as e:
            # ---------------------------------------------------------------------
            # UNEXPECTED ERROR (not HTTPException)
            # ---------------------------------------------------------------------
            # This should rarely happen - indicates a bug or unexpected condition
            logger.error(
                f"[{correlation_id}] Unexpected streaming error: {str(e)}",
                exc_info=True,  # Include full traceback for debugging
                extra={"correlation_id": correlation_id},
            )

            # Record unexpected error
            tracker.record_error(f"Unexpected: {str(e)}")

            # Send generic error to client
            yield 'data: {"error": "Internal server error"}\n\n'
            return
