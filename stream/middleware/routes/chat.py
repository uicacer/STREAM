"""
Main chat completion endpoint.

This module defines the primary API endpoint for chat interactions.
It handles:
1. Request validation (via Pydantic models)
2. Query complexity analysis
3. Tier routing decisions
4. Context window validation
5. Streaming response orchestration

The endpoint is OpenAI-compatible, meaning any client that works with
OpenAI's API will work with STREAM.

API Documentation:
    POST /chat/completions
    - Request: OpenAI-compatible chat completion request
    - Response: Server-Sent Events (SSE) stream

    GET /context/limits
    - Response: Context window limits for each tier
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stream.middleware.config import (
    DEFAULT_JUDGE_STRATEGY,
    DEFAULT_VISION_MODELS,
    JUDGE_STRATEGIES,
    VISION_CAPABLE_MODELS,
)
from stream.middleware.core.complexity_judge import judge_complexity
from stream.middleware.core.query_router import AuthError, get_model_for_tier, get_tier_for_query
from stream.middleware.core.streaming import create_streaming_response
from stream.middleware.core.tier_health import set_active_cloud_provider
from stream.middleware.utils.context_window import check_context_limit
from stream.middleware.utils.multimodal import extract_text_content, has_images
from stream.middleware.utils.token_estimator import estimate_tokens

# Configure module logger
logger = logging.getLogger(__name__)

# Create FastAPI router
# This will be registered in app.py with a prefix (e.g., /api/v1)
router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS (Pydantic)
# =============================================================================


class Message(BaseModel):
    """
    A single message in a conversation.

    OpenAI-compatible format with role and content.

    MULTIMODAL SUPPORT:
    The "content" field can be either:
      1. A plain string (text-only messages) — the traditional format
      2. A list of content blocks (multimodal messages) — the OpenAI vision format

    Text-only example:
        {"role": "user", "content": "Hello!"}

    Multimodal example (text + image):
        {"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
        ]}

    WHY str | list[dict]:
    The original STREAM only supported text (content: str). To add image
    support without breaking existing clients, we accept BOTH formats.
    A plain string request still works exactly as before. A list request
    enables multimodal content. This is the same approach OpenAI uses —
    their API accepts both formats for backwards compatibility.

    Attributes:
        role: Message sender ("user", "assistant", or "system")
        content: Message text (string) or list of content blocks (multimodal)
    """

    role: str = Field(
        ...,  # Required field (... means no default)
        description="Role: user, assistant, or system",
    )
    content: str | list[dict] = Field(
        ...,  # Required field
        description=(
            "Message content. String for text-only, or list of content blocks "
            "for multimodal (OpenAI vision format)."
        ),
    )


class ChatCompletionRequest(BaseModel):
    """
    Chat completion request body.

    This matches OpenAI's API format for compatibility.
    Any OpenAI client library can use STREAM without modification.

    Attributes:
        model: Model or tier to use ("auto" for intelligent routing)
        messages: Conversation history (list of Message objects)
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
        user: Optional user ID (reserved for future authentication)

    Example:
        >>> request = ChatCompletionRequest(
        ...     model="auto",
        ...     messages=[{"role": "user", "content": "Hi"}],
        ...     temperature=0.7
        ... )

    Validation:
        - temperature must be between 0.0 and 2.0
        - messages list cannot be empty
        - Each message must have role and content
    """

    model: str = Field(
        default="auto", description="Model or tier to use (auto = intelligent routing)"
    )

    messages: list[Message] = Field(
        ...,  # Required
        description="Conversation messages",
        min_length=1,  # At least one message required
    )

    temperature: float | None = Field(
        default=0.7,
        ge=0.0,  # Greater than or equal to 0.0
        le=1.0,  # Less than or equal to 1.0
        description="Sampling temperature (0.0-1.0)",
    )

    user: str | None = Field(
        default=None, description="User ID (for future rate limiting/authentication)"
    )

    judge_strategy: str | None = Field(
        default=None,
        description="Judge strategy for complexity analysis (ollama-3b, gemma-vision, haiku). Default: ollama-3b",
    )

    local_model: str | None = Field(
        default=None,
        description="Model to use for local tier (local-llama, local-vision). Default: local-llama",
    )

    lakeshore_model: str | None = Field(
        default=None,
        description="Model to use for lakeshore tier (lakeshore-qwen). Default: lakeshore-qwen",
    )

    cloud_provider: str | None = Field(
        default=None,
        description="Cloud provider to use when tier is 'cloud' (cloud-claude, cloud-gpt, cloud-gpt-cheap). Default: cloud-claude",
    )


# =============================================================================
# MAIN CHAT ENDPOINT
# =============================================================================


@router.post("/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, request: Request):
    """
    Main chat completion endpoint with streaming and automatic fallback.

    This endpoint:
    1. Extracts the user's query from the conversation history
    2. Analyzes query complexity (simple vs. complex)
    3. Routes to the appropriate tier (local/campus/cloud)
    4. Validates context window limits
    5. Streams the response with automatic tier fallback

    The endpoint is OpenAI-compatible, meaning you can use it as a drop-in
    replacement for OpenAI's API:

        openai.api_base = "http://localhost:5000"
        response = openai.ChatCompletion.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello!"}]
        )

    Args:
        request_body: Validated request body (Pydantic handles validation)
        request: FastAPI request object (contains middleware-injected state)

    Returns:
        StreamingResponse: Server-Sent Events (SSE) stream containing:
        - Initial metadata (tier, model)
        - Content chunks (the AI response, token by token)
        - Cost summary (tokens used, total cost)

    Raises:
        HTTPException 400: Invalid request (context too long, bad parameters)
        HTTPException 500: Internal server error

    Example Request:
        POST /chat/completions
        {
          "model": "auto",
          "messages": [
            {"role": "user", "content": "Explain quantum computing"}
          ],
          "temperature": 0.7
        }

    There are 3 roles in chat messages:

    1. user: The human asking questions (visible to everyone)
    Example: {"role": "user", "content": "What is Python?"}

    2. assistant: The AI's responses (visible to everyone)
    Example: {"role": "assistant", "content": "Python is a programming language..."}

    3. system: Instructions/configuration for the AI (HIDDEN from user, only AI sees it)
    Example: {"role": "system", "content": "You are a helpful coding tutor."}

    Key Difference:
    - user & assistant = The visible conversation (back and forth)
    - system = Hidden instructions that shape how the assistant behaves

    Example Response (SSE stream):
        data: {"stream_metadata": {"tier": "lakeshore", "model": "llama3.2:3b"}}

        data: {"choices": [{"delta": {"content": "Quantum"}}]}

        data: {"choices": [{"delta": {"content": " computing"}}]}

        data: {"stream_metadata": {"cost": {"total": 0.0}}}

    Note:
        The correlation_id is injected by middleware for request tracing.
        It allows us to follow a single request through all log files.
    """

    # =========================================================================
    # STEP 1: Extract Correlation ID (for logging/tracing)
    # =========================================================================
    # This is injected by middleware in app.py
    # Format: UUID or timestamp-based identifier
    correlation_id = request.state.correlation_id

    # =========================================================================
    # STEP 2: Extract User Query (with multimodal support)
    # =========================================================================
    # Get the most recent user message (last message with role="user").
    #
    # MULTIMODAL HANDLING:
    # The user's content might be a string ("Hello") or a list of content
    # blocks ([{"type": "text", "text": "..."}, {"type": "image_url", ...}]).
    # For the complexity judge, we only need the TEXT portion — the judge
    # can't process images (unless using the gemma-vision strategy).
    # extract_text_content() handles both formats transparently.
    user_messages = [msg for msg in request_body.messages if msg.role == "user"]

    if user_messages:
        # Extract just the text for the complexity judge.
        # If content is a string, this returns it as-is.
        # If content is a list of blocks, this extracts and joins text blocks.
        user_query = extract_text_content(user_messages[-1].content)
    else:
        # Edge case: No user messages (shouldn't happen due to validation)
        user_query = ""
        logger.warning(
            f"[{correlation_id}] No user messages in conversation",
            extra={"correlation_id": correlation_id},
        )

    # =========================================================================
    # STEP 2b: Detect Images in the Conversation
    # =========================================================================
    # Check if ANY message in the conversation contains images.
    # This affects:
    #   1. ROUTING: Must select a vision-capable model
    #   2. TOKEN ESTIMATION: Images consume ~765 tokens each
    #   3. PAYLOAD SIZE: Base64 images can be large (Globus has 10 MB limit)
    #
    # We convert messages to dicts first so has_images() can inspect them.
    messages_as_dicts = [msg.model_dump() for msg in request_body.messages]
    query_has_images = has_images(messages_as_dicts)

    # =========================================================================
    # STEP 3: Analyze Query Complexity
    # =========================================================================
    # Complexity determines routing and fallback priority:
    # - Low: Simple factual queries ("What's 2+2?")
    # - Medium: Moderate tasks ("Summarize this article")
    # - High: Complex reasoning ("Design a database schema for...")
    #
    # OPTIMIZATION: Skip complexity judgment if user explicitly selected a tier
    # (not "auto"). This saves 500-2000ms per request.

    user_selected_tier = request_body.model in ["local", "lakeshore", "cloud"]
    judge_strategy = request_body.judge_strategy or DEFAULT_JUDGE_STRATEGY
    judge_fallback_info = None  # Track fallback for UI notification
    judge_cost = 0.0  # Track judge cost (for paid judges like Haiku)

    # Validate judge strategy
    if judge_strategy not in JUDGE_STRATEGIES:
        logger.warning(
            f"[{correlation_id}] Unknown judge strategy '{judge_strategy}', using default"
        )
        judge_strategy = DEFAULT_JUDGE_STRATEGY

    if user_selected_tier:
        # User explicitly chose a tier - skip the slow LLM judge
        complexity = "user_override"
        logger.debug(
            f"[{correlation_id}] Skipping complexity judge (user selected tier: {request_body.model})",
            extra={"correlation_id": correlation_id},
        )
    else:
        # Use the judge_complexity function with selected strategy.
        # We pass query_has_images so the judge can adjust its default
        # complexity for image queries (MEDIUM instead of the usual default).
        judgment_result = judge_complexity(
            user_query, judge_strategy, query_has_images=query_has_images
        )
        complexity = judgment_result.complexity
        judge_cost = judgment_result.judge_cost  # Capture judge cost

        # Track fallback info for UI notification
        if judgment_result.method in ["keyword_fallback", "default_fallback"]:
            judge_fallback_info = {
                "method": judgment_result.method,
                "reason": judgment_result.fallback_reason,
                "strategy_attempted": judgment_result.strategy_used,
            }
            logger.warning(
                f"[{correlation_id}] Judge fallback: {judgment_result.fallback_reason}",
                extra={"correlation_id": correlation_id},
            )

    logger.debug(
        f"[{correlation_id}] Query complexity: {complexity}",
        extra={"correlation_id": correlation_id, "complexity": complexity},
    )

    # =========================================================================
    # STEP 4: Route to Tier
    # =========================================================================
    # Determine which tier to use based on:
    # - User's model preference (from request_body.model)
    # - Query complexity
    # - Tier availability
    try:
        # Remember the user's cloud provider so background monitor tests it too
        set_active_cloud_provider(request_body.cloud_provider)

        routing_result = get_tier_for_query(
            user_query,
            request_body.model,
            cloud_provider=request_body.cloud_provider,
            lakeshore_model=request_body.lakeshore_model,
        )
    except AuthError as e:
        # Auth error on a tier - don't fallback, show error to user
        logger.error(
            f"[{correlation_id}] Auth error on {e.tier}: {e.message}",
            extra={"correlation_id": correlation_id, "tier": e.tier},
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error_type": "auth_subscription",
                "message": "Cloud API authentication failed. Your API key may be invalid or your subscription may have expired.",
                "raw_error": e.message,
                "provider": e.tier,
            },
        ) from e

    tier = routing_result.tier
    model = get_model_for_tier(
        tier,
        cloud_provider=request_body.cloud_provider,
        local_model=request_body.local_model,
        lakeshore_model=request_body.lakeshore_model,
        has_images=query_has_images,
    )

    # =========================================================================
    # STEP 4b: Validate Model Supports Modality
    # =========================================================================
    # If the query contains images but the selected model is text-only:
    #   - AUTO mode → auto-switch to the vision model for that tier
    #   - Explicit tier/model → respect user choice, return helpful error
    if query_has_images and model not in VISION_CAPABLE_MODELS:
        if request_body.model == "auto":
            # Auto mode: STREAM is in control, switch to vision model
            vision_model = DEFAULT_VISION_MODELS.get(tier)
            if vision_model:
                logger.info(
                    f"[{correlation_id}] Auto-switching from '{model}' to "
                    f"vision model '{vision_model}' (query contains images)",
                    extra={
                        "correlation_id": correlation_id,
                        "model": model,
                        "vision_model": vision_model,
                    },
                )
                model = vision_model
            else:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_type": "model_not_multimodal",
                        "message": (f"No vision-capable model is available for the '{tier}' tier."),
                    },
                )
        else:
            # User explicitly selected a tier/model — respect their choice
            logger.warning(
                f"[{correlation_id}] Image query sent to text-only model '{model}'",
                extra={"correlation_id": correlation_id, "model": model},
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error_type": "model_not_multimodal",
                    "message": (
                        "The selected model is text-only and can't process images. "
                        "Switch to a vision model or use Auto mode to let STREAM pick one for you."
                    ),
                },
            )

    # Track routing fallback info for UI notification
    routing_fallback_info = None
    if routing_result.fallback_used:
        routing_fallback_info = {
            "fallback_used": True,
            "original_tier": routing_result.original_tier,
            "actual_tier": routing_result.tier,
            "unavailable_tiers": routing_result.unavailable_tiers,
        }

    logger.debug(
        f"[{correlation_id}] Routing decision: tier={tier}, model={model}",
        extra={
            "correlation_id": correlation_id,
            "tier": tier,
            "model": model,
        },
    )

    # =========================================================================
    # STEP 5: Prepare Messages
    # =========================================================================
    # Use the dict conversion we already computed in Step 2b.
    # These dicts preserve the original content format (string or list),
    # which flows through to Ollama, vLLM, and LiteLLM unchanged.
    # All three inference engines accept the OpenAI vision format natively.
    messages = messages_as_dicts

    # =========================================================================
    # STEP 6: Validate Context Window
    # =========================================================================
    # Check if conversation fits in the model's context window
    # This prevents crashes and truncation

    estimated_input = estimate_tokens(messages)
    within_limit, max_allowed = check_context_limit(estimated_input, model, correlation_id)

    if not within_limit:
        # Conversation is too long for this model
        logger.error(
            f"[{correlation_id}] Context window exceeded: "
            f"{estimated_input} tokens > {max_allowed} limit for {model}",
            extra={
                "correlation_id": correlation_id,
                "estimated_tokens": estimated_input,
                "max_allowed": max_allowed,
                "model": model,
            },
        )

        # Return helpful error to user
        raise HTTPException(
            status_code=400,  # Bad Request
            detail={
                "error": "context_too_long",
                "message": (
                    f"Conversation history ({estimated_input} tokens) exceeds "
                    f"{model} limit ({max_allowed} tokens)"
                ),
                "suggestion": "Please start a new conversation or use Cloud tier for longer contexts",
                "estimated_tokens": estimated_input,
                "model_limit": max_allowed,
            },
        )

    # =========================================================================
    # STEP 7: Log Routing Decision
    # =========================================================================
    logger.info(
        f"[{correlation_id}] Routing: tier={tier}, model={model}, complexity={complexity}",
        extra={
            "correlation_id": correlation_id,
            "tier": tier,
            "model": model,
            "complexity": complexity,
            "estimated_tokens": estimated_input,
        },
    )

    # =========================================================================
    # STEP 8: Stream Response
    # =========================================================================
    try:
        # Create streaming response using core/streaming.py
        return StreamingResponse(
            create_streaming_response(
                model=model,
                messages=messages,
                temperature=request_body.temperature,
                correlation_id=correlation_id,
                tier=tier,
                user_query=user_query,
                complexity=complexity,
                judge_fallback_info=judge_fallback_info,
                routing_fallback_info=routing_fallback_info,
                judge_cost=judge_cost,
            ),
            media_type="text/event-stream",  # SSE content type
            headers={
                # Disable caching (streaming must be live)
                "Cache-Control": "no-cache",
                # Disable Nginx buffering (for real-time streaming)
                "X-Accel-Buffering": "no",
                # Keep connection alive during streaming
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        # Re-raise HTTPExceptions (already formatted)
        raise

    except Exception as e:
        # Catch unexpected errors
        logger.error(
            f"[{correlation_id}] Error processing request: {str(e)}",
            exc_info=True,  # Include full traceback
            extra={"correlation_id": correlation_id},
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        ) from e
        # What this "from e" does is preserve the original exception context
        # This means that when the HTTPException is raised, it retains information about the original error
        # Without "from e" the original traceback would be lost, making debugging harder


# # =============================================================================
# # CONTEXT LIMITS ENDPOINT
# # =============================================================================


# @router.get("/context/limits")
# async def get_context_limits():
#     """
#     Get context window limits for all tiers.

#     This endpoint provides information about token limits for each tier,
#     which is useful for:
#     - UI warnings ("You're approaching the limit")
#     - Client-side truncation
#     - Tier selection

#     Returns:
#         Dictionary with limits organized by tier:
#         {
#           "success": true,
#           "limits": {
#             "local": {"total": 8000, "reserve_output": 2000},
#             "lakeshore": {"total": 8000, "reserve_output": 2000},
#             "cloud": {"total": 128000, "reserve_output": 4000}
#           },
#           "timestamp": "2026-01-23T12:34:56.789Z"
#         }

#     Example:
#         GET /context/limits

#         Response:
#         {
#           "success": true,
#           "limits": {...},
#           "timestamp": "2026-01-23T12:34:56.789Z"
#         }
#     """

#     return {
#         "success": True,
#         "limits": get_tier_context_limits(),
#         "timestamp": datetime.now(UTC).isoformat(),
#     }
