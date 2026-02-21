"""
Globus Compute client for submitting inference tasks to Lakeshore HPC cluster.

This module provides:
1. Remote function execution on Lakeshore via Globus Compute
2. vLLM inference through Globus Compute endpoints
3. Fallback handling for connection failures

Architecture:
    Middleware → GlobusComputeClient → Globus Endpoint (on Lakeshore) → vLLM Server

This avoids the need for SSH port forwarding by using Globus Compute's
managed networking infrastructure.
"""

import asyncio
import logging
import os
import time
import warnings
from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.errors.error_types import DeserializationError, TaskExecutionFailed
from globus_compute_sdk.serialize import AllCodeStrategies, ComputeSerializer
from globus_sdk import GlobusAPIError
from globus_sdk.login_flows.command_line_login_flow_manager import CommandLineLoginFlowEOFError

from stream.middleware.config import (
    GLOBUS_MAX_PAYLOAD_BYTES,
    LAKESHORE_MODELS,
    MODEL_CONTEXT_LIMITS,
    get_lakeshore_vllm_url,
)
from stream.middleware.utils.multimodal import strip_old_images

logger = logging.getLogger(__name__)

# Suppress the Globus SDK's noisy "Environment differences detected" warning.
# This fires when the local Python version (e.g., 3.12.12) differs slightly from
# the endpoint workers (e.g., 3.12.3). Minor version differences are harmless —
# serialization works fine across patch versions with the same dill version.
warnings.filterwarnings(
    "ignore", message=r"[\s\S]*Environment differences detected", category=UserWarning
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Globus Compute endpoint ID for Lakeshore cluster
# This should be set in your .env file as GLOBUS_COMPUTE_ENDPOINT_ID
GLOBUS_ENDPOINT_ID = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")

# vLLM server URL *on Lakeshore* (not localhost)
# This is the URL that the remote function will use to access vLLM
# Example: "http://ga-001:8000" or "http://localhost:8000" if vLLM runs on the compute node
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://ga-001:8000")

# Timeout for Globus Compute tasks (seconds)
GLOBUS_TASK_TIMEOUT = int(os.getenv("GLOBUS_TASK_TIMEOUT", "240"))


# =============================================================================
# REMOTE FUNCTION (Executes on Lakeshore)
# =============================================================================
#
# This function is serialized and sent to the Lakeshore HPC endpoint via Globus
# Compute. It must be completely self-contained because it executes in an isolated
# Python environment on the remote machine. All imports (like `requests`) MUST be
# inside the function body — module-level imports don't exist on the endpoint.
#
# WHY exec() FROM A SOURCE STRING:
# PyInstaller bundles .pyc bytecode, not .py source files. That bytecode contains
# references to PyInstaller's internal import system (pyimod02_importers). When
# Globus Compute serializes a function for the remote endpoint, it captures the
# bytecode. The endpoint doesn't have PyInstaller, so deserialization fails.
#
# By defining the function from a source string via exec() at runtime, Python's
# standard compiler produces clean bytecode with no PyInstaller references.
#
# Previous attempts that failed:
# 1. CombinedCode strategy → inspect.getsource() fails (no .py files in bundle)
# 2. AllCodeStrategies with normal def → dill by-reference → "No module named 'stream'"
# 3. __module__ = '__main__' → dill by-value → "No module named 'pyimod02_importers'"
# 4. exec() from source string → clean bytecode, works everywhere ✓

_REMOTE_FN_SOURCE = """\
def remote_vllm_inference(vllm_url, model, messages, temperature, max_tokens, stream=False):
    try:
        import requests
        endpoint = f"{vllm_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=180)
            if response.status_code >= 400:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text
                return {
                    "error": f"{response.status_code} Error: {error_body}",
                    "error_type": "HTTPError",
                    "status_code": response.status_code,
                    "response_body": error_body,
                    "request_payload": payload,
                }
            return response.json()
        except requests.exceptions.RequestException as e:
            error_response = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_response = e.response.json()
                except Exception:
                    error_response = e.response.text if hasattr(e.response, "text") else str(e.response)
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "status_code": getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None,
                "response_body": error_response,
            }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "error_type": type(e).__name__,
        }
"""

# Compile and execute the source string to produce a function with clean bytecode.
# The filename "<remote_vllm_inference>" appears in tracebacks for debugging.
_ns = {}
exec(compile(_REMOTE_FN_SOURCE, "<remote_vllm_inference>", "exec"), _ns)
remote_vllm_inference = _ns["remote_vllm_inference"]


# =============================================================================
# REMOTE STREAMING FUNCTION (Executes on Lakeshore — streams tokens via relay)
# =============================================================================
#
# This is the streaming counterpart of remote_vllm_inference. Instead of
# collecting the full response and returning it via Globus Compute's result
# channel (which is batch-only), this function streams ALL data — tokens,
# usage stats, errors — through the WebSocket relay in real-time:
#
#   1. Connects to the WebSocket relay as a PRODUCER
#   2. Calls vLLM with stream=True (so vLLM returns tokens one at a time)
#   3. Reads each token from vLLM's SSE stream
#   4. Forwards it through the relay to the waiting consumer
#   5. Sends final "done" message with usage stats through the relay
#
# Everything the consumer needs flows through the RELAY (data plane).
# The Globus Compute return value is just a technical requirement — every
# Globus function must return something, but we don't wait for it or use it.
# The consumer reads tokens from the relay and is done before Globus even
# delivers the return value.
#
# Same exec() pattern as above — see comments on _REMOTE_FN_SOURCE for why.

_REMOTE_STREAMING_FN_SOURCE = """\
def remote_vllm_streaming(vllm_url, model, messages, temperature, max_tokens, relay_url, channel_id):
    import json
    import requests
    from websockets.sync.client import connect as ws_connect

    ws = None

    try:
        # ---- Step 1: Connect to the relay as a PRODUCER ----
        # The relay is a public WebSocket server that both sides connect to.
        # We're the PRODUCER — we'll send tokens. The consumer (STREAM's proxy
        # or litellm_direct) is already connected and waiting on the other end.
        ws_url = f"{relay_url}/produce/{channel_id}"
        ws = ws_connect(ws_url)

        # ---- Step 2: Make a STREAMING request to vLLM ----
        # stream=True tells vLLM to return tokens as Server-Sent Events (SSE)
        # as the GPU generates them, instead of waiting for the full response.
        #
        # requests stream=True tells the requests library to not download the
        # full response body immediately, but to give us an iterator we can
        # read line by line.
        response = requests.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=180,
        )

        if response.status_code >= 400:
            error_msg = f"vLLM HTTP {response.status_code}"
            try:
                error_msg += f": {response.text[:300]}"
            except Exception:
                pass
            ws.send(json.dumps({"type": "error", "message": error_msg}))
            ws.send(json.dumps({"type": "done"}))
            return {"error": error_msg}

        # ---- Step 3: Read SSE chunks from vLLM and forward via relay ----
        # vLLM's SSE format (one event per line):
        #   data: {"choices":[{"delta":{"content":"Hello"}}]}
        #   data: {"choices":[{"delta":{"content":" world"}}]}
        #   data: [DONE]
        #
        # We parse each line, extract the token text from delta.content,
        # and send it through the WebSocket relay to the consumer.
        usage = {}
        tokens_sent = 0
        for line in response.iter_lines(decode_unicode=True):
            # Skip empty lines (SSE uses blank lines as event separators)
            if not line or not line.startswith("data: "):
                continue

            # Strip the "data: " prefix to get the JSON payload
            payload = line[6:]

            # "[DONE]" signals the end of the SSE stream
            if payload.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")

            # Forward the token to the relay (and thus to the consumer)
            if content:
                ws.send(json.dumps({"type": "token", "content": content}))
                tokens_sent += 1

            # Capture usage stats from the final chunk (vLLM includes them
            # in the last SSE event with finish_reason="stop")
            if chunk.get("usage"):
                usage = chunk["usage"]

        # ---- Step 4: Signal completion through the relay ----
        # Everything the consumer needs is sent here: the "done" signal
        # plus usage stats (prompt_tokens, completion_tokens, total_tokens).
        # The consumer reads this and knows the stream is complete.
        ws.send(json.dumps({"type": "done", "usage": usage}))

    except Exception as e:
        # Best-effort: try to notify the consumer about the error via relay
        if ws:
            try:
                ws.send(json.dumps({"type": "error", "message": str(e)}))
                ws.send(json.dumps({"type": "done"}))
            except Exception:
                pass
        return {"error": f"{type(e).__name__}: {e}"}

    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    # Globus Compute requires a return value, but the consumer doesn't use it.
    # All data (tokens, usage, errors) was already sent through the relay.
    return {"ok": True, "tokens_sent": tokens_sent}
"""

_ns2 = {}
exec(compile(_REMOTE_STREAMING_FN_SOURCE, "<remote_vllm_streaming>", "exec"), _ns2)
remote_vllm_streaming = _ns2["remote_vllm_streaming"]


# =============================================================================
# GLOBUS COMPUTE CLIENT CLASS
# =============================================================================


class GlobusComputeClient:
    """
    Client for submitting vLLM inference tasks to Lakeshore via Globus Compute.

    This client:
    - Submits inference jobs to the remote Globus endpoint
    - Waits for results with configurable timeout
    - Handles errors and provides fallback information

    Usage:
        client = GlobusComputeClient()

        if client.is_available():
            result = await client.submit_inference(
                messages=[{"role": "user", "content": "Hello!"}],
                temperature=0.7,
                max_tokens=100
            )
    """

    def __init__(self):
        """Initialize the Globus Compute client."""
        self.endpoint_id = GLOBUS_ENDPOINT_ID
        self.vllm_url = VLLM_SERVER_URL
        self._executor = None
        self._globus_app = None  # Will be initialized lazily for authentication

        # Log configuration
        if self.endpoint_id:
            logger.info(
                f"Globus Compute client initialized: endpoint={self.endpoint_id}, "
                f"vllm_url={self.vllm_url}"
            )
        else:
            logger.warning(
                "Globus Compute endpoint ID not configured. "
                "Set GLOBUS_COMPUTE_ENDPOINT_ID in .env to enable Globus Compute mode."
            )

    def is_available(self) -> bool:
        """
        Check if Globus Compute is configured and available.

        Returns:
            True if endpoint ID is configured, False otherwise
        """
        return self.endpoint_id is not None and self.endpoint_id.strip() != ""

    # =========================================================================
    # PERSISTENT EXECUTOR
    # =========================================================================
    #
    # WHAT IS THE EXECUTOR?
    # ---------------------
    # The Globus Compute Executor is the main way to run functions on remote
    # HPC clusters. It works like Python's concurrent.futures.Executor but
    # for remote machines:
    #
    #   Local (concurrent.futures):   executor.submit(fn, args) → runs fn on a thread/process
    #   Remote (Globus Compute):      executor.submit(fn, args) → runs fn on Lakeshore HPC
    #
    # Under the hood, the Executor:
    #   1. Serializes your function + arguments into bytes (using dill/pickle)
    #   2. Sends the bytes to Globus cloud via AMQP (a messaging protocol)
    #   3. Globus cloud routes the task to the Lakeshore endpoint
    #   4. The endpoint deserializes and runs the function on a GPU node
    #   5. The result comes back through the same AMQP connection
    #   6. executor.submit() returns a Future — a "ticket" you can check later
    #
    # WHY DO WE NEED IT?
    # ------------------
    # We can't call vLLM on Lakeshore directly — it's behind the university
    # firewall. Globus Compute provides secure, managed access to HPC resources
    # without needing SSH tunnels or VPN. The Executor is the SDK's interface
    # for submitting work to those resources.
    #
    # WHY KEEP IT PERSISTENT?
    # -----------------------
    # Creating a new Executor per request is expensive (~1-2 seconds) because
    # it must:
    #   1. Open a TCP connection to Globus cloud
    #   2. Perform the AMQP handshake (authentication, channel setup)
    #   3. Register as a task submitter
    #
    # By keeping one Executor alive across requests, the AMQP connection stays
    # open. Subsequent requests skip steps 1-3 and go straight to submitting
    # the task (~100ms instead of ~1500ms).
    #
    # WHAT IF THE CONNECTION DROPS?
    # -----------------------------
    # AMQP connections can die from network glitches, Globus service restarts,
    # or token expiry. When that happens, we:
    #   1. Detect the error during submit or result retrieval
    #   2. Close the broken Executor (_reset_executor)
    #   3. Create a fresh one (_get_executor)
    #   4. Retry the request once
    # If the retry also fails, we return the error to the user.

    def _get_executor(self) -> Executor:
        """
        Get or create a persistent Globus Compute Executor.

        On the first call, creates a new Executor and establishes the AMQP
        connection to Globus cloud. On subsequent calls, returns the same
        Executor — reusing the existing connection.

        The serializer is configured once with AllCodeStrategies, which tries
        multiple serialization methods to handle Python version mismatches
        between the local machine and the Lakeshore endpoint.

        Returns:
            A ready-to-use Executor instance
        """
        if self._executor is None:
            logger.info("Creating persistent Globus Compute Executor...")
            self._executor = Executor(endpoint_id=self.endpoint_id)
            # AllCodeStrategies tries multiple serialization methods to find one
            # that works. This is more robust than the default CombinedCode
            # strategy when there's a Python version mismatch between the local
            # machine and the endpoint (e.g., local 3.12.12 vs endpoint 3.12.3).
            self._executor.serializer = ComputeSerializer(strategy_code=AllCodeStrategies())
            logger.info("Persistent Executor created (AMQP connection established)")
        return self._executor

    def _reset_executor(self):
        """
        Close the current Executor and clear it so the next call to
        _get_executor() creates a fresh one.

        Called when:
        - The AMQP connection drops (network issue, Globus restart)
        - Authentication tokens expire mid-session
        - Any unexpected error during task submission

        shutdown(wait=False) tells the Executor to close immediately without
        waiting for pending tasks to finish. cancel_futures=True cancels any
        tasks that haven't completed yet.
        """
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                # Best effort — the connection might already be dead
                logger.debug(f"Error during Executor shutdown (expected if connection died): {e}")
            self._executor = None
            logger.info("Executor reset — will reconnect on next request")

    def shutdown(self):
        """
        Clean up the persistent Executor when the app is shutting down.

        This should be called during app exit (e.g., FastAPI shutdown event)
        to properly close the AMQP connection and release resources.
        Without this, the connection would be abandoned and the OS would
        eventually clean it up, but it's better to close it properly.
        """
        logger.info("Shutting down Globus Compute client...")
        self._reset_executor()
        logger.info("Globus Compute client shut down")

    def _get_globus_app(self, force_refresh: bool = False):
        """
        Get the Globus app instance for authentication.

        Args:
            force_refresh: If True, forces creation of a fresh app instance
                          to pick up newly saved credentials.

        Returns:
            GlobusApp instance
        """
        if force_refresh:
            # Clear SDK's internal singleton cache before getting fresh instance
            try:
                from globus_compute_sdk.sdk.auth import globus_app as globus_app_module

                if hasattr(globus_app_module, "_globus_app"):
                    globus_app_module._globus_app = None
                if hasattr(globus_app_module, "GLOBUS_APP"):
                    globus_app_module.GLOBUS_APP = None
            except Exception:
                pass  # Best effort - some SDK versions have different internals

            self._globus_app = None

        if self._globus_app is None:
            from globus_compute_sdk.sdk.auth.globus_app import get_globus_app

            self._globus_app = get_globus_app()
            logger.debug(f"GlobusApp initialized (force_refresh={force_refresh})")

        return self._globus_app

    def reload_credentials(self) -> tuple[bool, str]:
        """
        Force reload of Globus credentials from disk.

        This should be called after the user authenticates on the host machine
        to pick up the newly saved credentials.

        The Globus SDK uses a singleton pattern for get_globus_app(), so we need
        to clear its internal cache to force re-reading credentials from disk.

        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info("Reloading Globus credentials...")

            # Clear our cached reference
            self._globus_app = None

            # Clear the Globus SDK's internal singleton cache
            # The SDK caches the GlobusApp instance in the module
            try:
                from globus_compute_sdk.sdk.auth import globus_app as globus_app_module

                # Clear the module-level cache that get_globus_app() uses
                if hasattr(globus_app_module, "_globus_app"):
                    globus_app_module._globus_app = None
                    logger.debug("Cleared globus_app_module._globus_app")

                # Some SDK versions use different cache names
                if hasattr(globus_app_module, "GLOBUS_APP"):
                    globus_app_module.GLOBUS_APP = None
                    logger.debug("Cleared globus_app_module.GLOBUS_APP")

            except Exception as e:
                logger.debug(f"Could not clear SDK cache (this is often OK): {e}")

            # Get fresh app instance - this will re-read credentials from disk
            app = self._get_globus_app(force_refresh=True)

            # Check if we're now authenticated
            if app.login_required():
                logger.warning("Still not authenticated after reload")
                return False, "Credentials not found. Please authenticate first."
            else:
                logger.info("Credentials reloaded successfully")
                return True, "Credentials reloaded successfully"

        except Exception as e:
            logger.error(f"Failed to reload credentials: {e}")
            return False, f"Failed to reload credentials: {str(e)}"

    def ensure_authenticated(self, force_refresh: bool = False) -> tuple[bool, str | None]:
        """
        Check if user is authenticated with Globus Compute.

        Args:
            force_refresh: If True, forces re-reading credentials from disk.
                          Use this after user authenticates on host machine.

        Returns:
            Tuple of (is_authenticated, auth_url_or_error)
            - If authenticated: (True, None)
            - If auth required: (False, auth_instructions)
        """
        try:
            # Get app instance, optionally forcing a fresh check
            app = self._get_globus_app(force_refresh=force_refresh)

            if app.login_required():
                logger.warning("Globus Compute authentication required")

                # We're running in Docker - can't open browser automatically
                # Return instructions for the user to authenticate via the Streamlit frontend
                auth_message = (
                    "Globus Compute authentication required. "
                    "Please authenticate by running this command on your host machine:\n\n"
                    "  python3 -m globus_compute_sdk.sdk.login_manager.manager\n\n"
                    "Or visit: https://app.globus.org/\n\n"
                    "After authenticating, retry your request."
                )

                logger.error(auth_message)
                return False, auth_message
            else:
                logger.debug("Already authenticated with Globus Compute")
                return True, None

        except Exception as e:
            logger.error(f"Authentication check failed: {e}")
            return False, f"Authentication check failed: {str(e)}"

    def _estimate_payload_size(self, messages: list[dict]) -> int:
        """
        Estimate the serialized payload size for a Globus Compute task.

        WHY THIS MATTERS:
        Globus Compute enforces a 10 MB limit on task submissions. When the
        user sends images, the base64-encoded image data is included in the
        messages list, which gets serialized and sent to the Globus service.

        Reference: https://globus-compute.readthedocs.io/en/stable/limits.html

        A single uncompressed image can be 5-10 MB in base64. Even after frontend
        compression (max 1024px, JPEG 85%), each image is typically 300-700 KB.
        Multiple images can easily exceed the limit.

        HOW WE ESTIMATE:
        We can't cheaply compute the exact serialized size (that would require
        actually serializing with dill). Instead, we estimate by measuring the
        JSON representation of the messages, which is a good approximation since
        the base64 strings are the dominant size component.

        Args:
            messages: Chat messages that would be sent to the remote function.

        Returns:
            Estimated payload size in bytes.
        """
        import json

        # json.dumps gives a close approximation of the serialized size.
        # The actual dill-serialized payload includes function bytecode and
        # other arguments, but messages (especially base64 images) dominate.
        try:
            return len(json.dumps(messages).encode("utf-8"))
        except (TypeError, ValueError):
            # If messages can't be JSON-serialized, fall back to str() length.
            # This is a rough estimate but better than nothing.
            return len(str(messages).encode("utf-8"))

    async def submit_inference(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        _retry: bool = False,  # Internal flag to prevent infinite retry loops
    ) -> dict[str, Any]:
        """
        Submit an inference task to Lakeshore via Globus Compute.

        This method:
        1. Validates payload size (base64 images can be large)
        2. Creates a Globus Compute executor
        3. Submits the remote_vllm_inference function to execute on Lakeshore
        4. Waits for the result with timeout
        5. Returns the vLLM response or error information

        Args:
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate. If None, reads from
                        MODEL_CONTEXT_LIMITS["lakeshore-qwen"]["reserve_output"]
                        in config.py (single source of truth for all context
                        window settings).
            model: Model identifier on vLLM server

        Returns:
            Dictionary containing either:
            - Success: vLLM response in OpenAI format
            - Error: Error information with "error" key

        Raises:
            Exception: If Globus Compute is not available or submission fails
        """
        # If max_tokens was not passed by the caller, read the default from
        # MODEL_CONTEXT_LIMITS in config.py. The "reserve_output" field is how
        # many tokens are reserved for the model's response — the same value
        # used by context_window.py and litellm_direct.py.
        if max_tokens is None:
            lakeshore_limits = MODEL_CONTEXT_LIMITS.get(model, {})
            max_tokens = lakeshore_limits.get("reserve_output", 2048)

        if not self.is_available():
            raise RuntimeError(
                "Globus Compute not configured. Set GLOBUS_COMPUTE_ENDPOINT_ID in .env"
            )

        # =====================================================================
        # STRIP OLD IMAGES (keep only latest user message's images)
        # =====================================================================
        # Long conversations with multiple image messages can easily exceed
        # the 8 MB Globus payload limit. We strip images from older messages
        # before serialization. The model's previous text responses about
        # those images remain in the history for context.
        messages = strip_old_images(messages)

        # =====================================================================
        # PAYLOAD SIZE VALIDATION (for multimodal messages with images)
        # =====================================================================
        # Globus Compute enforces a 10 MB task payload limit (see limits.html).
        # Our safety limit is 8 MB (GLOBUS_MAX_PAYLOAD_BYTES). We check BEFORE
        # submitting to give the user a clear error instead of a cryptic
        # TASK_PAYLOAD_TOO_LARGE failure from the Globus API.
        estimated_size = self._estimate_payload_size(messages)
        if estimated_size > GLOBUS_MAX_PAYLOAD_BYTES:
            size_mb = estimated_size / (1024 * 1024)
            limit_mb = GLOBUS_MAX_PAYLOAD_BYTES / (1024 * 1024)
            logger.error(
                f"Payload too large for Globus Compute: {size_mb:.1f} MB > {limit_mb:.0f} MB limit"
            )
            return {
                "error": (
                    f"Image payload too large for Lakeshore ({size_mb:.1f} MB). "
                    f"Globus Compute has a {limit_mb:.0f} MB limit. "
                    "Try reducing image size/quality or using fewer images. "
                    "Alternatively, use the Cloud tier which has no size limit."
                ),
                "error_type": "payload_too_large",
            }

        # Log if this is a retry attempt
        if _retry:
            logger.debug("This is a retry attempt after authentication")

        # Ensure authentication before attempting submission
        is_authenticated, auth_message = self.ensure_authenticated()
        if not is_authenticated:
            logger.error("Authentication required")
            return {
                "error": auth_message or "Globus Compute authentication required",
                "error_type": "AuthenticationError",
                "auth_required": True,  # Special flag for frontend to detect
            }

        logger.info(f"Submitting inference task to Globus endpoint {self.endpoint_id}")

        # Track timing for each step so we can see where latency comes from.
        # These logs help identify the bottleneck:
        #   - get_executor: AMQP connection setup (should be ~0 if reusing)
        #   - submit: serialization + AMQP send
        #   - wait: Globus routing + remote execution + result return
        t_start = time.perf_counter()

        try:
            # Get the persistent Executor (creates one if first call, reuses
            # the existing AMQP connection on subsequent calls).
            gce = self._get_executor()

            t_executor = time.perf_counter()

            # Resolve the vLLM URL for this specific model.
            # Each Lakeshore model runs on a different port (e.g., 8000, 8001, ...).
            vllm_url = get_lakeshore_vllm_url(model)

            # Resolve the HuggingFace model name that vLLM expects.
            # STREAM uses internal names like "lakeshore-qwen-1.5b", but the vLLM
            # instance is loaded with the HF name (e.g., "Qwen/Qwen2.5-32B-Instruct-AWQ").
            model_info = LAKESHORE_MODELS.get(model)
            hf_model = model_info["hf_name"] if model_info else model

            # Submit the function to execute remotely on Lakeshore.
            # gce.submit() serializes remote_vllm_inference + its arguments,
            # sends them to Globus cloud via AMQP, which routes them to the
            # Lakeshore HPC endpoint. The function runs on a GPU node there.
            future = gce.submit(
                remote_vllm_inference,
                vllm_url,
                hf_model,
                messages,
                temperature,
                max_tokens,
                False,  # stream=False (streaming not yet implemented)
            )

            t_submit = time.perf_counter()

            logger.debug("Waiting for Globus Compute task to complete...")

            # HOW LAKESHORE INFERENCE WORKS (both server and desktop mode):
            # ============================================================
            #
            # 1. gce.submit() sends the inference function to UIC's Lakeshore
            #    HPC cluster via Globus Compute (a remote execution service).
            #    The function runs on a GPU node at Lakeshore, calls vLLM,
            #    and returns the result.
            #
            # 2. gce.submit() returns a concurrent.futures.Future — a "ticket"
            #    for the result. The result isn't ready yet (the GPU is working).
            #
            # 3. future.result() blocks until the GPU finishes and sends back
            #    the result. This can take 5-30+ seconds depending on the query.
            #
            # WHY asyncio.to_thread():
            # ------------------------
            # future.result() is a BLOCKING call — it freezes the calling thread.
            # In an async server, that freezes the event loop, which prevents the
            # server from handling ANY other requests (health polls, the litellm
            # self-connection in desktop mode, etc.).
            #
            # asyncio.to_thread() moves the blocking wait to a background thread,
            # keeping the event loop free. This is safe for both modes:
            #   - Server mode:  proxy container stays responsive during Globus wait
            #   - Desktop mode: main server handles the litellm self-connection
            #                   while Globus processes on Lakeshore
            #
            result = await asyncio.to_thread(future.result, timeout=GLOBUS_TASK_TIMEOUT)

            t_result = time.perf_counter()

            # Log timing breakdown so we can see exactly where time is spent.
            # Example output:
            #   "Lakeshore timing: executor=0.01s, submit=0.45s, wait=3.21s, total=3.67s"
            logger.info(
                f"Lakeshore timing: "
                f"executor={t_executor - t_start:.2f}s, "
                f"submit={t_submit - t_executor:.2f}s, "
                f"wait={t_result - t_submit:.2f}s, "
                f"total={t_result - t_start:.2f}s"
            )

            # Check if the remote function returned an error
            if isinstance(result, dict) and "error" in result:
                logger.error(f"vLLM inference failed on Lakeshore: {result['error']}")
                return result

            logger.info("Globus Compute task completed successfully")
            return result

        except TimeoutError:
            logger.error(f"Globus Compute task timeout after {GLOBUS_TASK_TIMEOUT}s")
            return {
                "error": f"Task timeout after {GLOBUS_TASK_TIMEOUT}s",
                "error_type": "TimeoutError",
            }

        except GlobusAPIError as e:
            # Check if this is an authentication error
            if e.http_status in (401, 403):
                logger.warning(f"Authentication error detected (HTTP {e.http_status})")
                # Auth tokens expired — reset executor so next request gets
                # a fresh AMQP connection with refreshed tokens.
                self._reset_executor()
                auth_message = (
                    "Your Globus Compute session has expired. "
                    "Please authenticate by running this command on your host machine:\n\n"
                    "  python3 -m globus_compute_sdk.sdk.login_manager.manager\n\n"
                    "Or visit: https://app.globus.org/\n\n"
                    "After authenticating, retry your request."
                )
                return {
                    "error": auth_message,
                    "error_type": "AuthenticationError",
                    "auth_required": True,
                }
            else:
                # Not an auth error, log and return error
                logger.error(f"Globus API error (HTTP {e.http_status}): {str(e)}")
                return {
                    "error": str(e),
                    "error_type": "GlobusAPIError",
                }

        except CommandLineLoginFlowEOFError:
            # This happens when the Globus SDK tries to re-authenticate in a non-interactive
            # environment (Docker container). It means tokens are invalid or expired.
            logger.warning("Globus SDK tried to re-authenticate in non-interactive mode")
            self._reset_executor()
            return {
                "error": "Globus Compute authentication required. Please authenticate.",
                "error_type": "AuthenticationError",
                "auth_required": True,
            }

        except (DeserializationError, TaskExecutionFailed) as e:
            # TaskExecutionFailed wraps different remote errors:
            #   - ManagerLost: HPC compute node crashed or SLURM job expired
            #   - DeserializationError: Python version mismatch between local/remote
            #   - Other remote execution failures
            #
            # We inspect the error message to give the user an accurate diagnosis.
            error_str = str(e)
            error_lower = error_str.lower()

            if "managerlost" in error_lower or "loss of manager" in error_lower:
                # The HPC compute node's worker manager crashed. Common causes:
                # - SLURM job timed out and was killed
                # - Node ran out of memory
                # - Endpoint was restarted
                # Extract just the final error line (skip the huge remote traceback)
                last_line = error_str.strip().split("\n")[-1].strip().rstrip("-")
                logger.error(f"Lakeshore compute node lost: {last_line.strip()}")
                return {
                    "error": (
                        "Lakeshore HPC compute node is unavailable "
                        "(worker manager lost on the cluster). "
                        "The SLURM job may have expired or the node crashed. "
                        "Check: ssh lakeshore 'squeue -u $USER' to verify jobs are running."
                    ),
                    "error_type": "ManagerLost",
                }

            # Actual deserialization error (Python version mismatch)
            logger.error(
                f"Globus result deserialization failed: {e}",
                exc_info=True,
            )
            return {
                "error": (
                    "Lakeshore processed the request but the result couldn't be decoded. "
                    "This usually means the Python version or globus_compute_sdk version "
                    "on your machine doesn't match the Lakeshore endpoint. "
                    "Try: uv pip install --upgrade globus-compute-sdk"
                ),
                "error_type": "DeserializationError",
            }

        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Globus Compute task failed: {str(e)}", exc_info=True)

            # Check if this is an authentication-related error
            # "unable to open database file" = token database doesn't exist (need to authenticate)
            # "login_required" = SDK detected tokens are missing/expired
            # "eof" errors = SDK trying to prompt for auth in non-interactive mode
            if (
                "unable to open database file" in error_str
                or "login_required" in error_str
                or "eof" in error_str
                and "authorization" in error_str
            ):
                logger.warning("Authentication required - triggering auth flow")
                self._reset_executor()
                return {
                    "error": "Globus Compute authentication required. Please authenticate.",
                    "error_type": "AuthenticationError",
                    "auth_required": True,
                }

            # -----------------------------------------------------------------
            # STALE CONNECTION RETRY
            # -----------------------------------------------------------------
            # If the AMQP connection died (network glitch, Globus service
            # restart, idle timeout), the Executor may throw various errors
            # (ConnectionError, OSError, AMQP errors, etc.). We can't
            # enumerate them all, so on ANY unexpected error that's not auth
            # or deserialization, we:
            #   1. Reset the Executor (close dead connection)
            #   2. Retry once with a fresh Executor
            #   3. If the retry fails too, return the error
            #
            # The _retry flag prevents infinite retry loops — if this IS
            # already a retry, we just return the error.
            if not _retry:
                logger.warning(
                    f"Unexpected error ({type(e).__name__}), resetting Executor and retrying..."
                )
                self._reset_executor()
                return await self.submit_inference(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    _retry=True,
                )

            return {
                "error": str(e),
                "error_type": type(e).__name__,
            }

    # =========================================================================
    # STREAMING INFERENCE (via WebSocket relay)
    # =========================================================================
    #
    # Unlike submit_inference() which waits for the complete result via Globus,
    # submit_streaming_inference() just SUBMITS the job and returns immediately.
    # The actual tokens flow through the WebSocket relay — the consumer connects
    # to the relay and receives tokens in real-time as the GPU generates them.
    #
    # The caller's workflow:
    #   1. result = await client.submit_streaming_inference(...)
    #   2. channel_id = result["channel_id"]
    #   3. Connect to relay as consumer: wss://relay-url/consume/{channel_id}
    #   4. Receive tokens in real-time from the relay
    #
    # Meanwhile on Lakeshore (the Globus job):
    #   1. The remote function runs on a GPU compute node (e.g., a-001)
    #   2. It makes an OUTBOUND WebSocket connection to the relay's public URL
    #      (e.g., wss://abc.ngrok-free.app/produce/{channel_id})
    #      This works because the compute node can make outbound HTTPS connections
    #      (we verified this in test_compute_node_connectivity.py)
    #   3. It calls vLLM with stream=True to get tokens one at a time
    #   4. It forwards each token through the WebSocket relay to the consumer
    #   5. All data (tokens, usage stats, errors) flows through the relay
    #
    # Two channels carry information back:
    #   DATA PLANE (relay):   tokens + usage stats + done signal → used by consumer
    #   CONTROL PLANE (Globus): job status (ok/error + token count) → confirmation
    #
    # We don't wait for the Globus result — the relay's "done" message tells
    # the consumer everything it needs. Globus delivers the job status later,
    # which could be checked for monitoring/debugging if needed.

    async def submit_streaming_inference(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str = "",
        relay_url: str = "",
    ) -> dict[str, Any]:
        """
        Submit a streaming inference task to Lakeshore via Globus Compute.

        This is the streaming counterpart of submit_inference(). Instead of
        waiting for the full result, it:
          1. Generates a unique channel_id (UUID)
          2. Submits the remote_vllm_streaming function to Globus
          3. Returns the channel_id immediately

        The caller then connects to the relay as a consumer using the
        channel_id to receive tokens in real-time.

        Args:
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: STREAM model key (e.g., "lakeshore-qwen-1.5b", "lakeshore-qwen-32b").
                   Resolved to the HuggingFace model name internally.
            relay_url: WebSocket URL of the relay server (e.g., wss://abc.ngrok-free.app)

        Returns:
            {"channel_id": "uuid-string"} on success
            {"error": "message", ...} on failure
        """
        import uuid

        if max_tokens is None:
            lakeshore_limits = MODEL_CONTEXT_LIMITS.get(model, {})
            max_tokens = lakeshore_limits.get("reserve_output", 2048)

        if not self.is_available():
            return {"error": "Globus Compute not configured", "error_type": "ConfigError"}

        if not relay_url:
            return {"error": "RELAY_URL not configured", "error_type": "ConfigError"}

        # Strip old images (same as submit_inference)
        messages = strip_old_images(messages)

        # Payload size check (same as submit_inference — images can be large)
        estimated_size = self._estimate_payload_size(messages)
        if estimated_size > GLOBUS_MAX_PAYLOAD_BYTES:
            size_mb = estimated_size / (1024 * 1024)
            limit_mb = GLOBUS_MAX_PAYLOAD_BYTES / (1024 * 1024)
            return {
                "error": (
                    f"Image payload too large for Lakeshore ({size_mb:.1f} MB). "
                    f"Globus Compute has a {limit_mb:.0f} MB limit. "
                    "Try reducing image size/quality or using fewer images."
                ),
                "error_type": "payload_too_large",
            }

        # Ensure authentication before attempting submission
        is_authenticated, auth_message = self.ensure_authenticated()
        if not is_authenticated:
            return {
                "error": auth_message or "Globus Compute authentication required",
                "error_type": "AuthenticationError",
                "auth_required": True,
            }

        # Generate a unique channel ID for this streaming session.
        # Both the producer (Lakeshore) and consumer (our app) use this
        # to connect to the same relay channel.
        channel_id = str(uuid.uuid4())

        try:
            gce = self._get_executor()

            # Resolve the vLLM URL for this model (each model runs on its own port)
            vllm_url = get_lakeshore_vllm_url(model)

            # Resolve the HuggingFace model name that vLLM expects in API calls
            model_info = LAKESHORE_MODELS.get(model)
            hf_model = model_info["hf_name"] if model_info else model

            logger.info(
                f"Submitting STREAMING inference to Globus endpoint {self.endpoint_id} "
                f"(model={model} → {hf_model}, channel={channel_id[:8]}, relay={relay_url})"
            )

            # Submit the streaming function — this is fast (~100ms).
            # It serializes the function + args and sends them to Globus via AMQP.
            # The function will run on Lakeshore, connect OUTBOUND to the relay,
            # and stream tokens through the WebSocket connection.
            #
            # We don't call future.result() here. The consumer reads tokens from
            # the relay in real-time. Globus delivers the job status (ok/error)
            # later via the Future, but we don't need to wait for it.
            gce.submit(
                remote_vllm_streaming,
                vllm_url,
                hf_model,
                messages,
                temperature,
                max_tokens,
                relay_url,
                channel_id,
            )

            logger.info(f"Streaming job submitted (channel={channel_id[:8]})")

            return {"channel_id": channel_id}

        except Exception as e:
            logger.error(f"Failed to submit streaming inference: {e}", exc_info=True)

            error_str = str(e).lower()
            if "unable to open database file" in error_str or "login_required" in error_str:
                self._reset_executor()
                return {
                    "error": "Globus Compute authentication required.",
                    "error_type": "AuthenticationError",
                    "auth_required": True,
                }

            return {
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def check_model_health(self, model: str, timeout: int = 20) -> tuple[bool, str | None]:
        """
        SYNCHRONOUS per-model health check: sends a 1-token inference to a
        specific Lakeshore model through Globus Compute, with a short timeout.

        This is called by tier_health.check_tier_health() when the user selects
        a specific Lakeshore model. It verifies that the model's vLLM instance
        is actually running and can generate output.

        Why sync (not async)?
        ---------------------
        check_tier_health() is a sync function called from both:
          - Async context: health route handler (via is_tier_available)
          - Sync context: query_router (via is_tier_available)
        Making this async would require two code paths. Since the existing
        health checks already block (httpx sync calls for local/cloud), adding
        another blocking call is consistent with the codebase pattern.

        How the timeout works:
        ----------------------
        There are two timeout layers:
          1. Remote function: requests.post(timeout=180) on the HPC — this is
             the vLLM request timeout inside remote_vllm_inference.
          2. Client side: future.result(timeout=20) — this is OUR timeout.

        Scenario: Model NOT running
          → remote requests.post() gets ConnectionRefused immediately (~0s)
          → Globus returns the error dict in ~5-6s (AMQP round-trip)
          → We get result in ~6s, well within our 20s timeout.

        Scenario: Model running, fast (1.5B)
          → 1-token inference takes <1s on GPU
          → Globus round-trip ~5s
          → We get result in ~6s.

        Scenario: Model running, slow (32B)
          → 1-token inference takes 5-10s on GPU
          → Globus round-trip ~5s
          → We get result in ~10-15s. Still within 20s.

        Scenario: Globus itself is broken
          → future.result(timeout=20) raises TimeoutError at 20s.
          → We return (False, "timeout message").

        Note: If the remote function is still running on HPC when we timeout,
        it continues running there but we ignore its result. This is a 1-token
        request so it wastes negligible resources.

        Args:
            model: STREAM model key (e.g., "lakeshore-qwen-32b").
                   Used to look up the correct vLLM port and HuggingFace name.
            timeout: Max seconds to wait for the Globus round-trip.
                    Default 20s — long enough for slow models, short enough
                    to not block the UI.

        Returns:
            (is_healthy, error_message)
            - (True, None) if the model responded successfully
            - (False, "error description") if the model is down or unreachable
        """
        if not self.is_available():
            return False, "Globus Compute not configured"

        # Check authentication first — no point submitting if not authed.
        # ensure_authenticated() is a cheap local check (no network call).
        is_authenticated, auth_message = self.ensure_authenticated()
        if not is_authenticated:
            return False, auth_message or "Globus Compute authentication required"

        try:
            # Get the persistent Executor (reuses existing AMQP connection)
            gce = self._get_executor()

            # Resolve model-specific vLLM URL (each model runs on a different port).
            # Example: "lakeshore-qwen-32b" → "http://ga-002:8004"
            vllm_url = get_lakeshore_vllm_url(model)

            # Resolve the HuggingFace model name that vLLM expects in API calls.
            # STREAM uses internal names (e.g., "lakeshore-qwen-1.5b"), but vLLM
            # was started with the HF name (e.g., "Qwen/Qwen2.5-1.5B-Instruct").
            model_info = LAKESHORE_MODELS.get(model)
            if not model_info:
                return False, f"Unknown model: {model}"
            hf_model = model_info["hf_name"]

            logger.info(
                f"[Health] Checking model {model} → {hf_model} at {vllm_url} "
                f"(timeout={timeout}s)"
            )

            t_start = time.perf_counter()

            # Submit minimal 1-token inference to the remote HPC node.
            # This goes through the full Globus Compute path:
            #   local → AMQP → Globus cloud → Lakeshore endpoint → vLLM on port
            # The prompt "hi" and max_tokens=1 is the cheapest possible inference.
            future = gce.submit(
                remote_vllm_inference,
                vllm_url,
                hf_model,
                [{"role": "user", "content": "hi"}],
                0.0,  # temperature (deterministic — don't waste randomness)
                1,  # max_tokens (just 1 token — we only care if it responds)
                False,  # stream=False
            )

            # Block until the result arrives or timeout fires.
            # This is the key difference from submit_inference() which uses
            # asyncio.to_thread(). Here we block directly because
            # check_tier_health() is a sync function. The timeout ensures
            # we don't hang for 240s if something goes wrong.
            result = future.result(timeout=timeout)

            elapsed = time.perf_counter() - t_start
            logger.info(f"[Health] Model {model} check completed in {elapsed:.1f}s")

            # Check if the remote function returned an error dict.
            # ConnectionRefused → {"error": "...", "error_type": "ConnectionError"}
            # HTTP 4xx/5xx → {"error": "...", "error_type": "HTTPError"}
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "Unknown error")
                logger.warning(f"[Health] Model {model} is NOT available: {error_msg}")
                return False, f"Model not responding: {error_msg[:150]}"

            # Success — the model generated at least 1 token
            logger.info(f"[Health] Model {model} is available")
            return True, None

        except TimeoutError:
            logger.warning(f"[Health] Model {model} health check timed out after {timeout}s")
            return False, f"Model not responding (timed out after {timeout}s)"

        except Exception as e:
            # Unexpected error (AMQP connection issue, serialization failure, etc.)
            # Don't reset the executor here — let the next real inference handle that.
            logger.error(
                f"[Health] Model {model} health check failed: {e}",
                exc_info=True,
            )
            return False, f"Health check error: {str(e)[:150]}"

    async def health_check(self) -> tuple[bool, str | None]:
        """
        Perform a health check by submitting a simple test inference.

        Returns:
            Tuple of (is_healthy, error_message)
            - is_healthy: True if health check passed
            - error_message: None if healthy, error description otherwise
        """
        if not self.is_available():
            return False, "Globus Compute not configured"

        try:
            # Submit a minimal test query
            result = await self.submit_inference(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.0,
                max_tokens=5,
            )

            # Check if result indicates an error
            if "error" in result:
                return False, result["error"]

            # Success!
            return True, None

        except Exception as e:
            return False, str(e)


# =============================================================================
# MODULE-LEVEL CLIENT INSTANCE
# =============================================================================

# Global client instance (initialized once)
# This avoids creating multiple executors
globus_client = GlobusComputeClient()
