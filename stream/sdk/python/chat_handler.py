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
# CONFIGURATION
# =============================================================================

MIDDLEWARE_BASE_URL = os.getenv("MIDDLEWARE_URL")
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
        self._cost_rates = {}  # Cost rates from middleware (for UI display only)

        # HTTP client for middleware communication
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True)

        # Load cost rates from middleware (for UI cost estimates only)
        self._load_cost_rates()

    def _load_cost_rates(self):
        """
        Load cost rates from middleware (for UI display only)

        Note: This is NOT used for routing decisions (middleware handles that).
        It's only used to show estimated costs in the UI when middleware
        doesn't return them (e.g., during streaming before completion).
        """
        try:
            response = self.client.get(f"{MIDDLEWARE_BASE_URL}/v1/costs/models", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                self._cost_rates = data.get("costs", {})
                logger.debug(f"Loaded cost rates for {len(self._cost_rates)} models")
            else:
                logger.warning(f"⚠️ Failed to load cost rates: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"⚠️ Failed to load cost rates: {e}")

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for UI display only

        IMPORTANT: This is NOT used for routing decisions!
        Middleware calculates authoritative costs.

        This method only exists to show estimated costs in the UI
        when middleware doesn't provide them (e.g., during streaming).

        Args:
            model: Model identifier (e.g., "cloud-claude")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in dollars (for display purposes only)
        """
        if model not in self._cost_rates:
            return 0.0

        rates = self._cost_rates[model]
        input_cost = input_tokens * rates.get("input", 0.0)
        output_cost = output_tokens * rates.get("output", 0.0)

        return input_cost + output_cost

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
                "Cannot connect to middleware. Is it running?", start_time
            )
        except httpx.TimeoutException:
            return self._create_error_response(
                f"Request timeout after {REQUEST_TIMEOUT} seconds", start_time
            )
        except Exception as e:
            logger.exception("Unexpected error in chat()")
            return self._create_error_response(f"Unexpected error: {str(e)}", start_time)

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
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        fallback_used = False
        original_tier = None

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
                yield f"Error: Middleware returned {response.status_code}"
                return

            for line in response.iter_lines():
                if not line.strip() or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Extract metadata from middleware
                    if "stream_metadata" in data:
                        metadata = data["stream_metadata"]

                        # Check for fallback
                        if metadata.get("fallback"):
                            fallback_used = True
                            original_tier = metadata.get("original_tier")
                            tier = metadata.get("current_tier", tier)
                            logger.warning(f"⚠️ Fallback detected: {original_tier} → {tier}")

                        # Update tier/model
                        tier = metadata.get("tier", tier)
                        model = metadata.get("model", model)

                        # Extract cost
                        if "cost" in metadata:
                            cost = metadata["cost"].get("total", 0.0)
                            input_tokens = metadata["cost"].get("input_tokens", 0)
                            output_tokens = metadata["cost"].get("output_tokens", 0)

                    # Extract content
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            yield content

                    # Extract usage (fallback if not in metadata)
                    if "usage" in data and input_tokens == 0:
                        input_tokens = data["usage"].get("prompt_tokens", 0)
                        output_tokens = data["usage"].get("completion_tokens", 0)

                except json.JSONDecodeError:
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
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "fallback_used": fallback_used,
            "original_tier": original_tier,
        }

        logger.info(
            f"Stream completed - tier={tier}, model={model}, "
            f"cost=${cost:.6f}, tokens={input_tokens + output_tokens}"
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

    def _create_error_response(self, error_message: str, start_time: float) -> dict:
        """Create standardized error response"""
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
