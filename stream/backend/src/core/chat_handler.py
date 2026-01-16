# =============================================================================
# STREAM - Chat Handler (Middleware Version)
# =============================================================================

import json
import logging
import os
import time
import uuid
from collections.abc import Generator

import httpx
import tiktoken

from stream.middleware.config import DEFAULT_MODELS, get_max_input_tokens, get_tier_context_limits

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MIDDLEWARE_BASE_URL = os.getenv("MIDDLEWARE_URL")
REQUEST_TIMEOUT = 120.0  # This is for both streaming and non-streaming requests

# =============================================================================
# CHAT HANDLER CLASS
# =============================================================================


class ChatHandler:
    """
    Simplified chat handler that calls STREAM middleware

    RESPONSIBILITIES:
        Send requests to middleware
        Manage conversation history
        Handle streaming responses
        Track session statistics

    NOT RESPONSIBLE FOR (middleware handles these):
        Routing decisions (which tier to use)
        Cost calculations
        Policy enforcement
        Authentication
    """

    def __init__(self):
        """Initialize chat handler"""
        self.conversation_history: list[dict] = []
        self.total_cost = 0.0
        self.query_count = 0
        self._last_stream_metadata = {}
        self._cost_rates = {}  # Middleware handles cost calculations
        self._context_limits = {}  # Middleware handles context limits
        # This client will be used for all requests to middleware
        # it is configured with a long timeout and follows redirects
        # redirects means that if the final destination is slow, the client will wait.
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True)

        # Fetch cost rates from middleware
        self._load_cost_rates()

        # Fetch context limits from middleware
        self._load_context_limits()

    def _load_context_limits(self):
        """Load context limits from middleware (single source of truth)"""
        try:
            response = self.client.get(f"{MIDDLEWARE_BASE_URL}/v1/context/limits", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                self._context_limits = data.get("limits", {})
                logger.debug(f"Loaded context limits: {self._context_limits}")
            else:
                logger.warning(f"⚠️ Failed to load context limits: HTTP {response.status_code}")
                # Use config helper function
                self._context_limits = get_tier_context_limits()
        except Exception as e:
            logger.error(f"⚠️ Failed to load context limits: {e}")
            # Use config helper function
            self._context_limits = get_tier_context_limits()

    def get_context_limits(self) -> dict:
        """Get context limits for all tiers"""
        return self._context_limits.copy()

    # TOKEN COUNTING & HISTORY MANAGEMENT
    def estimate_token_count(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Accurate token counting using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(
                f"⚠️ Token counting failed: {e}. Using fallback estimate of 4 char per token."
            )
            # Fallback to character-based estimate
            return len(text) // 4

    def trim_history_to_fit(
        self, messages: list[dict], max_input_tokens: int, reserve_for_response: int = 500
    ) -> list[dict]:
        """
        Trim conversation history to fit within context window

        Strategy:
        - Reserve tokens for the response
        - Keep messages from newest to oldest
        - Always keep the last user message
        - Drop oldest messages when limit exceeded

        Args:
            messages: Full conversation history
            max_input_tokens: Maximum tokens allowed for input
            reserve_for_response: Tokens to reserve for model's response

        Returns:
            Trimmed message list that fits in context window
        """
        if not messages:
            return messages

        # Calculate available budget
        available_tokens = max_input_tokens - reserve_for_response

        # Always keep the last message (current user query)
        if len(messages) == 0:
            return messages

        last_message = messages[-1]
        current_tokens = self.estimate_token_count(last_message["content"])

        # If even the last message is too big, truncate it
        if current_tokens > available_tokens:
            logger.warning(
                f"⚠️ Last message ({current_tokens} tokens) exceeds "
                f"available budget ({available_tokens} tokens). Truncating."
            )
            # Keep ~80% of available budget for user message
            max_chars = int(available_tokens * 4 * 0.8)
            truncated_content = last_message["content"][:max_chars] + "...[truncated]"
            return [{"role": last_message["role"], "content": truncated_content}]

        # Build trimmed history from newest to oldest
        trimmed = [last_message]
        remaining_tokens = available_tokens - current_tokens

        # Add messages from newest to oldest (excluding last, which we already added)
        for msg in reversed(messages[:-1]):
            msg_tokens = self.estimate_token_count(msg["content"])

            if msg_tokens <= remaining_tokens:
                trimmed.insert(0, msg)  # Add at beginning to maintain order
                remaining_tokens -= msg_tokens
            else:
                # Can't fit any more messages
                logger.debug(
                    f"📊 Context trimmed: keeping {len(trimmed)}/{len(messages)} messages "
                    f"({available_tokens - remaining_tokens}/{available_tokens} tokens used)"
                )
                break

        return trimmed

    def chat(
        self,
        user_message: str,
        user_preference: str = "auto",
        stream: bool = False,
        temperature: float = 0.7,
    ) -> dict:
        """
        Send a chat request to middleware and get response

        Args:
            user_message: What the user typed
            user_preference: "auto", "local", "lakeshore", or "cloud"
            stream: Enable streaming (words appear as they're generated)
            temperature: How creative (0=deterministic, 1=creative)

        Returns:
            Dictionary with success, response, tier, model, cost, duration, error
        """
        start_time = time.time()

        full_messages = self.conversation_history + [{"role": "user", "content": user_message}]

        # ========== CONTEXT MANAGEMENT ==========
        # Determine which tier will be used (for context limit)
        # We need to know this BEFORE sending to middleware

        messages = full_messages  # Default: use full history

        try:
            # Quick check: estimate which tier will be selected
            # (This is a simplified version - middleware does the real routing)
            # Use conservative estimate (smallest tier)
            estimated_tier = "local" if user_preference == "auto" else user_preference

            # Get context limit for this tier
            tier_model = DEFAULT_MODELS.get(estimated_tier, "local-llama")
            max_input_tokens = get_max_input_tokens(tier_model)

            # Trim history to fit
            messages = self.trim_history_to_fit(
                full_messages,  # Pass the full history
                max_input_tokens=max_input_tokens,
                reserve_for_response=500,
            )

            # Calculate total tokens AFTER trimming
            total_tokens = sum(self.estimate_token_count(m["content"]) for m in messages)

            logger.info(
                f"📊 Context: {len(messages)}/{len(full_messages)} messages, "
                f"~{total_tokens}/{max_input_tokens} tokens (tier={estimated_tier})"
            )

        except Exception as e:
            logger.error(f"⚠️ Context trimming FAILED: {e}", exc_info=True)
            # Still try with full history (middleware will reject if too big)
        # ========== END CONTEXT MANAGEMENT ==========

        payload = {
            "model": user_preference,
            "messages": messages,
            "temperature": temperature,
            # "max_tokens": MAX_TOKENS,
            "stream": stream,
        }

        try:
            if stream:
                return self._handle_streaming_request(payload, user_message, start_time)
            else:
                return self._handle_non_streaming_request(payload, user_message, start_time)

        except httpx.ConnectError:
            return self._create_error_response(
                "Cannot connect to middleware. Is it running on port 5000?", start_time
            )
        except httpx.TimeoutException:
            return self._create_error_response(
                f"Middleware timeout after {REQUEST_TIMEOUT} seconds", start_time
            )
        except Exception as e:
            return self._create_error_response(f"Unexpected error: {str(e)}", start_time)

    def _load_cost_rates(self):
        """Load cost rates from middleware (single source of truth)"""
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
        Estimate cost using rates from middleware

        Args:
            model: Model identifier (e.g., "cloud-claude")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in dollars
        """
        if model not in self._cost_rates:
            return 0.0

        rates = self._cost_rates[model]
        input_cost = input_tokens * rates.get("input", 0.0)
        output_cost = output_tokens * rates.get("output", 0.0)

        return input_cost + output_cost

    def _handle_streaming_request(
        self, payload: dict, user_message: str, start_time: float
    ) -> dict:
        """Handle streaming chat request"""

        # Generate correlation ID for tracking
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
                if not line.strip():
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        break

                    try:
                        # ========== PARSE JSON FIRST ==========
                        data = json.loads(data_str)  # ← This MUST come first!

                        # **PRIORITY 1: Extract metadata from middleware**
                        if "stream_metadata" in data:
                            metadata = data["stream_metadata"]
                            logger.info(f"📦 Metadata received: {metadata}")

                            # ========== CHECK FOR FALLBACK ==========
                            if metadata.get("fallback"):
                                fallback_used = True
                                original_tier = metadata.get("original_tier")
                                tier = metadata.get("current_tier", tier)
                                logger.info(f"⚠️ Fallback: {original_tier} → {tier}")
                                model = metadata.get("model", model)
                                logger.warning(f"⚠️ FALLBACK DETECTED: {original_tier} → {tier}")
                            # ========== END FALLBACK CHECK ==========

                            # Always update tier/model from metadata
                            tier = metadata.get("tier", tier)
                            model = metadata.get("model", model)

                            # Extract cost if present
                            if "cost" in metadata:
                                cost = metadata["cost"].get("total", 0.0)
                                input_tokens = metadata["cost"].get("input_tokens", 0)
                                output_tokens = metadata["cost"].get("output_tokens", 0)

                        # **PRIORITY 2: Extract content**
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                yield content

                        # **PRIORITY 3: Extract usage (fallback)**
                        if "usage" in data and input_tokens == 0:
                            input_tokens = data["usage"].get("prompt_tokens", 0)
                            output_tokens = data["usage"].get("completion_tokens", 0)

                    except json.JSONDecodeError:
                        continue

        # Update history after stream completes
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": full_text})
        self.query_count += 1

        # Store metadata
        self.total_cost += cost

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

        if fallback_used:
            logger.warning(
                f"Stream completed with fallback: {original_tier} → {tier}, "
                f"cost={cost}, tokens={input_tokens + output_tokens}"
            )
        else:
            logger.debug(
                f"Stream completed - tier={tier}, model={model}, "
                f"cost={cost}, tokens={input_tokens + output_tokens}"
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
                f"Middleware returned {response.status_code}: {response.text}", start_time
            )

        response_data = response.json()

        # Extract metadata
        metadata = response_data.get("stream_metadata", {})
        tier = metadata.get("tier", "unknown")
        model = response_data.get("model", "unknown")

        # Extract AI response
        assistant_message = response_data["choices"][0]["message"]["content"]

        # Extract cost
        cost = metadata.get("cost", {}).get("total", 0.0)

        # Calculate duration
        duration = time.time() - start_time

        # Update history
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
            "duration": duration,
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
    """Quick one-off chat (no history)"""
    handler = ChatHandler()
    result = handler.chat(query, user_preference=tier, stream=False)

    if result["success"]:
        return result["response"]
    else:
        return f"Error: {result['error']}"


def test_middleware_connection() -> bool:
    """Test if middleware is reachable"""
    print("🔍 Testing middleware connection...")
    print(f"   Middleware URL: {MIDDLEWARE_BASE_URL}")
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
        print("   Is middleware running? Start it with: cd middleware && python app.py")
        return False

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  STREAM Chat Handler - Test (Middleware Version)")
    print("=" * 70)
    print()

    # Test middleware connection
    if not test_middleware_connection():
        print()
        print("❌ Middleware not available. Start it first:")
        print("   cd middleware && python app.py")
        exit(1)

    print()
    print("=" * 70)
    print("  Testing Chat Functionality")
    print("=" * 70)
    print()

    # Test queries
    handler = ChatHandler()

    test_queries = [
        ("Hi", "auto"),
        ("Explain quantum computing in detail", "auto"),
    ]

    for query, preference in test_queries:
        print(f"💬 User ({preference}): {query}")
        result = handler.chat(query, user_preference=preference, stream=False)

        if result["success"]:
            print(f"🤖 {result['tier'].upper()} ({result['model']}): {result['response'][:100]}...")
            print(f"   Routing: {result['routing_reason']}")
            print(f"   Duration: {result['duration']:.2f}s | Cost: ${result['cost']:.6f}")
            if result.get("correlation_id"):
                print(f"   Correlation ID: {result['correlation_id']}")
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
