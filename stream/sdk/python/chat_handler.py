# =============================================================================
# STREAM - Chat Handler (SDK Client)
# =============================================================================
# Simplified client for frontends to interact with STREAM middleware
#
# RESPONSIBILITIES:
#   - Send requests to middleware
#   - Manage local conversation history (for UI display)
#   - Handle streaming responses
#   - Track session statistics
#   - Estimate costs for UI display (using rates from middleware)
#
# MIDDLEWARE HANDLES:
#   - Context management and trimming
#   - Token counting (for routing decisions)
#   - Routing decisions
#   - Cost calculations (authoritative)
#   - Policy enforcement
# =============================================================================

import json
import logging
import os
import time
import uuid
from collections.abc import Generator

import httpx

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# ERROR TYPES (Framework-agnostic constants)
# =============================================================================


class ErrorType:
    """Standardized error types for STREAM SDK"""

    AUTHENTICATION = "authentication"
    CONTEXT_TOO_LONG = "context_too_long"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    UNKNOWN = "unknown"


# =============================================================================
# CONFIGURATION
# =============================================================================

MIDDLEWARE_BASE_URL = os.getenv("MIDDLEWARE_URL", "http://localhost:5000")
REQUEST_TIMEOUT = 120.0

# =============================================================================
# CHAT HANDLER CLASS
# =============================================================================


class ChatHandler:
    """
    Simplified client for STREAM middleware

    Handles conversation history and communication with middleware.
    All intelligent routing, cost calculation, and context management
    is delegated to the middleware.
    """

    def __init__(self):
        """Initialize chat handler"""
        self.conversation_history: list[dict] = []
        self.total_cost = 0.0
        self.query_count = 0
        self._last_stream_metadata = {}

        # HTTP client for middleware communication
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True)

    def chat(
        self,
        user_message: str,
        user_preference: str = "auto",
        stream: bool = False,
        temperature: float = 0.7,
    ) -> dict:
        """
        Send a chat request to middleware

        Args:
            user_message: User's question/message
            user_preference: "auto", "local", "lakeshore", or "cloud"
            stream: Enable streaming responses
            temperature: Response creativity (0.0-1.0)

        Returns:
            Dictionary with success, response, tier, model, cost, duration, error
        """
        start_time = time.time()

        # Build full message history (middleware will handle context limits)
        messages = self.conversation_history + [{"role": "user", "content": user_message}]

        payload = {
            "model": user_preference,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        try:
            if stream:
                return self._handle_streaming_request(payload, user_message, start_time)
            else:
                return self._handle_non_streaming_request(payload, user_message, start_time)

        except httpx.ConnectError:
            return self._create_error_response(
                "Cannot connect to middleware. Is it running?",
                start_time,
                error_type=ErrorType.CONNECTION_ERROR,
            )
        except httpx.TimeoutException:
            return self._create_error_response(
                f"Request timeout after {REQUEST_TIMEOUT} seconds",
                start_time,
                error_type=ErrorType.TIMEOUT,
            )
        except Exception as e:
            logger.exception("Unexpected error in chat()")
            return self._create_error_response(
                f"Unexpected error: {str(e)}", start_time, error_type=ErrorType.UNKNOWN
            )

    def _handle_streaming_request(
        self, payload: dict, user_message: str, start_time: float
    ) -> dict:
        """Handle streaming chat request"""
        correlation_id = str(uuid.uuid4())

        generator = self._create_stream_generator(payload, user_message, correlation_id)

        return {
            "success": True,
            "response": generator,
            "tier": "streaming",
            "model": "streaming",
            "routing_reason": "",
            "cost": 0.0,
            "duration": time.time() - start_time,
            "stream": True,
            "error": None,
            "correlation_id": correlation_id,
        }

    def _create_stream_generator(
        self, payload: dict, user_message: str, correlation_id: str
    ) -> Generator:
        """Create generator for streaming response"""
        full_text = ""
        tier = "unknown"
        model = "unknown"
        complexity = "unknown"
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        fallback_used = False
        original_tier = None
        auth_error_detected = False
        error_message = None

        with self.client.stream(
            "POST",
            f"{MIDDLEWARE_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Correlation-ID": correlation_id,
            },
        ) as response:
            if response.status_code != 200:
                # Detect authentication error
                if response.status_code == 401:
                    auth_error_detected = True
                    # Try to extract the detailed error message
                    try:
                        error_data = response.json()
                        error_message = error_data.get("detail", "Authentication required")
                    except Exception:
                        error_message = "Globus Compute authentication required"
                    # Don't yield anything - we'll set metadata instead
                else:
                    error_message = f"Middleware returned {response.status_code}"

                # Set error metadata and stop
                self._last_stream_metadata = {
                    "auth_required": auth_error_detected,
                    "error_type": ErrorType.AUTHENTICATION
                    if auth_error_detected
                    else ErrorType.UNKNOWN,
                    "error": error_message,
                    "tier": "error",
                    "model": "none",
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
                return

            for line in response.iter_lines():
                logger.debug(f"Received SSE line: {line}")  # DEBUG: See what we're receiving

                if not line.strip() or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    logger.debug(f"Parsed SSE data: {data}")  # DEBUG: See parsed data

                    # ========== CHECK FOR ERRORS IN SSE STREAM ==========
                    # LiteLLM sends errors as JSON in the stream, not as HTTP status
                    if "error" in data:
                        logger.debug("ERROR FIELD DETECTED in SSE data!")  # DEBUG
                        # Handle both string and dict error formats
                        error_field = data.get("error")

                        # If error is a string (LiteLLM format), use it directly
                        if isinstance(error_field, str):
                            error_message = error_field
                        # If error is a dict (OpenAI format), extract message
                        elif isinstance(error_field, dict):
                            error_message = error_field.get("message", str(error_field))
                        else:
                            error_message = str(error_field)

                        # Check if server explicitly marked this as auth error
                        # OR detect from error message content
                        is_auth_error = data.get("auth_required", False) or (
                            "AuthenticationError" in error_message
                            or "authentication required" in error_message.lower()
                            or "globus" in error_message.lower()
                            or "401" in error_message
                        )

                        # Get tier/model from error response if available
                        error_tier = data.get("tier", tier if tier != "unknown" else "error")
                        error_model = data.get("model", model if model != "unknown" else "none")

                        # Set error metadata and stop streaming
                        self._last_stream_metadata = {
                            "auth_required": is_auth_error,
                            "error_type": ErrorType.AUTHENTICATION
                            if is_auth_error
                            else ErrorType.UNKNOWN,
                            "error": error_message,
                            "tier": error_tier,
                            "model": error_model,
                            "cost": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                        logger.warning(
                            f"Error detected in stream: {error_message} (auth_required={is_auth_error})"
                        )
                        return  # Stop streaming
                    # ========== END ERROR CHECK ==========

                    # Extract metadata from middleware
                    if "stream_metadata" in data:
                        metadata = data["stream_metadata"]

                        # Check for fallback
                        if metadata.get("fallback"):
                            fallback_used = True
                            original_tier = metadata.get("original_tier")
                            tier = metadata.get("current_tier", tier)
                            logger.warning(f"⚠️ Fallback detected: {original_tier} → {tier}")

                        # Update tier/model/complexity
                        tier = metadata.get("tier", tier)
                        model = metadata.get("model", model)
                        complexity = metadata.get("complexity", complexity)

                        # Update metadata immediately so UI can access tier info
                        # during streaming (before generator completes)
                        self._last_stream_metadata.update(
                            {
                                "tier": tier,
                                "model": model,
                                "complexity": complexity,
                                "fallback_used": fallback_used,
                                "original_tier": original_tier,
                            }
                        )

                        # Extract cost
                        if "cost" in metadata:
                            cost = metadata["cost"].get("total", 0.0)
                            input_tokens = metadata["cost"].get("input_tokens", 0)
                            output_tokens = metadata["cost"].get("output_tokens", 0)
                            logger.info(
                                f"💰 Cost extracted from stream_metadata: ${cost:.6f} "
                                f"(in={input_tokens}, out={output_tokens})"
                            )

                    # Extract content
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            yield content  # The actual response. For example: # "Python", " is", " a", ...

                    # Extract usage (fallback if not in metadata)
                    if "usage" in data and input_tokens == 0:
                        input_tokens = data["usage"].get("prompt_tokens", 0)
                        output_tokens = data["usage"].get("completion_tokens", 0)

                except json.JSONDecodeError:
                    # JSON parsing failed - check if this is an error message
                    logger.debug(f"JSON decode error for line: {data_str[:200]}")

                    # Check if the raw line contains error indicators
                    if '"error"' in data_str and (
                        "AuthenticationError" in data_str
                        or "authentication required" in data_str.lower()
                        or "401" in data_str
                    ):
                        # Authentication error in malformed JSON
                        logger.warning("Authentication error detected in malformed JSON")
                        self._last_stream_metadata = {
                            "auth_required": True,
                            "error_type": ErrorType.AUTHENTICATION,
                            "error": "Authentication required (Globus Compute)",
                            "tier": "error",
                            "model": "none",
                            "cost": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                        return  # Stop streaming
                    elif '"error"' in data_str:
                        # Generic error in malformed JSON
                        logger.warning("Error detected in malformed JSON")
                        self._last_stream_metadata = {
                            "auth_required": False,
                            "error_type": ErrorType.UNKNOWN,
                            "error": "Error occurred during inference",
                            "tier": "error",
                            "model": "none",
                            "cost": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                        return  # Stop streaming

                    # Not an error, just skip this line
                    continue

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": full_text})
        self.query_count += 1
        self.total_cost += cost

        # Store metadata for later retrieval
        self._last_stream_metadata = {
            "tier": tier,
            "model": model,
            "complexity": complexity,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "fallback_used": fallback_used,
            "original_tier": original_tier,
            "auth_required": False,  # No auth error if we got here
            "error_type": None,
            "error": None,
        }

        logger.info(
            f"Stream completed - tier={tier}, model={model}, "
            f"cost=${cost:.6f}, tokens={input_tokens + output_tokens}, "
            f"fallback_used={fallback_used}"
        )

    def _handle_non_streaming_request(
        self, payload: dict, user_message: str, start_time: float
    ) -> dict:
        """Handle non-streaming chat request"""
        response = self.client.post(
            f"{MIDDLEWARE_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            # Check if it's an authentication error
            if response.status_code == 401:
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", "Authentication required")
                    return self._create_error_response(
                        error_detail,
                        start_time,
                        error_type=ErrorType.AUTHENTICATION,
                        auth_required=True,
                    )
                except Exception:
                    return self._create_error_response(
                        "Globus Compute authentication required",
                        start_time,
                        error_type=ErrorType.AUTHENTICATION,
                        auth_required=True,
                    )
            else:
                return self._create_error_response(
                    f"Middleware error {response.status_code}: {response.text}", start_time
                )

        response_data = response.json()

        # Extract data from middleware response
        metadata = response_data.get("stream_metadata", {})
        tier = metadata.get("tier", "unknown")
        model = response_data.get("model", "unknown")
        assistant_message = response_data["choices"][0]["message"]["content"]
        cost = metadata.get("cost", {}).get("total", 0.0)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # Update statistics
        self.total_cost += cost
        self.query_count += 1

        return {
            "success": True,
            "response": assistant_message,
            "tier": tier,
            "model": model,
            "routing_reason": metadata.get("routing_reason", ""),
            "cost": cost,
            "duration": time.time() - start_time,
            "stream": False,
            "error": None,
            "correlation_id": metadata.get("correlation_id", ""),
        }

    def _create_error_response(
        self,
        error_message: str,
        start_time: float,
        error_type: str = ErrorType.UNKNOWN,
        auth_required: bool = False,
    ) -> dict:
        """Create standardized error response with structured error information"""
        return {
            "success": False,
            "response": None,
            "tier": "error",
            "model": "none",
            "routing_reason": "",
            "cost": 0.0,
            "duration": time.time() - start_time,
            "stream": False,
            "error": error_message,
            "error_type": error_type,
            "auth_required": auth_required,
        }

    def get_last_stream_metadata(self) -> dict:
        """Get metadata from most recent streaming request"""
        return self._last_stream_metadata.copy()

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_stats(self) -> dict:
        """Get session statistics"""
        return {
            "total_queries": self.query_count,
            "total_cost": self.total_cost,
            "avg_cost_per_query": (
                self.total_cost / self.query_count if self.query_count > 0 else 0
            ),
            "history_length": len(self.conversation_history),
        }

    def export_history(self) -> list[dict]:
        """Export conversation history"""
        return self.conversation_history.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_chat(query: str, tier: str = "auto") -> str:
    """
    Quick one-off chat (no history tracking)

    Args:
        query: Question to ask
        tier: "auto", "local", "lakeshore", or "cloud"

    Returns:
        Response text or error message
    """
    handler = ChatHandler()
    result = handler.chat(query, user_preference=tier, stream=False)

    if result["success"]:
        return result["response"]
    else:
        return f"Error: {result['error']}"


def test_middleware_connection() -> bool:
    """
    Test if middleware is reachable

    Returns:
        True if middleware is responding, False otherwise
    """
    print("🔍 Testing middleware connection...")
    print(f"   URL: {MIDDLEWARE_BASE_URL}")
    print()

    try:
        client = httpx.Client(timeout=5.0)
        response = client.get(f"{MIDDLEWARE_BASE_URL}/health")

        if response.status_code == 200:
            print("   ✅ Middleware is responding")
            data = response.json()
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"   ❌ Middleware returned {response.status_code}")
            return False

    except httpx.ConnectError:
        print(f"   ❌ Cannot connect to {MIDDLEWARE_BASE_URL}")
        print("   Is middleware running on port 5000?")
        return False

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  STREAM SDK Chat Handler - Test")
    print("=" * 70)
    print()

    # Test middleware connection
    if not test_middleware_connection():
        print()
        print("❌ Middleware not available. Start it with:")
        print("   docker-compose up -d middleware")
        exit(1)

    print()
    print("=" * 70)
    print("  Testing Chat Functionality")
    print("=" * 70)
    print()

    # Test queries
    handler = ChatHandler()

    test_queries = [
        ("Hi, how are you?", "auto"),
        ("Explain quantum computing", "auto"),
    ]

    for query, preference in test_queries:
        print(f"💬 User ({preference}): {query}")
        result = handler.chat(query, user_preference=preference, stream=False)

        if result["success"]:
            response_preview = result["response"][:100]
            print(f"🤖 {result['tier'].upper()} ({result['model']}): {response_preview}...")
            print(f"   Routing: {result['routing_reason']}")
            print(f"   Duration: {result['duration']:.2f}s | Cost: ${result['cost']:.6f}")
        else:
            print(f"❌ Error: {result['error']}")
        print()

    # Show stats
    print("=" * 70)
    print("  Session Statistics")
    print("=" * 70)
    stats = handler.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
