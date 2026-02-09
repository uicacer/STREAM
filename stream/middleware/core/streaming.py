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

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import HTTPException

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


async def create_streaming_response(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
    tier: str,
    user_query: str = "",
    complexity: str = "medium",
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
        }
    }

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
    for attempt in range(MAX_FALLBACK_ATTEMPTS):
        try:
            first_chunk = True  # Track if this is the first chunk for TTFT
            logger.debug(
                f"[{correlation_id}] Attempt {attempt + 1}/{MAX_FALLBACK_ATTEMPTS} on {current_tier}",
                extra={"correlation_id": correlation_id, "attempt": attempt + 1},
            )

            # ---------------------------------------------------------------------
            # Stream from LiteLLM
            # ---------------------------------------------------------------------
            # All tiers (local, lakeshore, cloud) route through LiteLLM gateway
            # For Lakeshore: LiteLLM → Lakeshore Proxy → Globus Compute OR SSH
            # For Local: LiteLLM → Ollama
            # For Cloud: LiteLLM → Anthropic/OpenAI APIs
            #
            # forward_to_litellm is an async generator that yields SSE lines
            # we use 'async for' here because we want to process each line as it arrives
            async for line in forward_to_litellm(
                model=current_model,
                messages=messages,
                temperature=temperature,
                correlation_id=correlation_id,
            ):
                # Record TTFT on first chunk
                if first_chunk:
                    tracker.record_first_token()
                    first_chunk = False

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
            cost = calculate_query_cost(current_model, input_tokens, output_tokens)
            tracker.record_tokens(input_tokens, output_tokens)
            tracker.record_completion(cost)

            # Send final cost summary if we have token counts
            if input_tokens > 0 or output_tokens > 0:
                cost_event = {
                    "stream_metadata": {
                        "tier": current_tier,
                        "model": current_model,
                        "fallback_used": len(tiers_tried) > 1,  # Did we use fallback?
                        "tiers_tried": tiers_tried,  # Which tiers did we try?
                        "cost": {
                            "total": cost,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                    }
                }

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

            logger.info(
                f"[{correlation_id}] Stream completed successfully: "
                f"cost=${cost:.6f}, tokens={input_tokens + output_tokens}",
                extra={
                    "correlation_id": correlation_id,
                    "tier": current_tier,
                    "cost": cost,
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

                    # Notify client of fallback
                    # This shows in the UI: "Local unavailable, trying cloud..."
                    fallback_event = {
                        "stream_metadata": {
                            "fallback": True,
                            "original_tier": tier,
                            "current_tier": current_tier,
                            "model": current_model,
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
