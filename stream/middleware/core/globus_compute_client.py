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
from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.errors.error_types import DeserializationError, TaskExecutionFailed
from globus_compute_sdk.serialize import AllCodeStrategies, ComputeSerializer
from globus_sdk import GlobusAPIError
from globus_sdk.login_flows.command_line_login_flow_manager import CommandLineLoginFlowEOFError

from stream.middleware.config import LAKESHORE_MODELS, MODEL_CONTEXT_LIMITS, get_lakeshore_vllm_url

logger = logging.getLogger(__name__)

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
            logger.info("🔄 Reloading Globus credentials...")

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
                logger.info("✅ Credentials reloaded successfully!")
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
                logger.warning("🔐 Globus Compute authentication required")

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
        1. Creates a Globus Compute executor
        2. Submits the remote_vllm_inference function to execute on Lakeshore
        3. Waits for the result with timeout
        4. Returns the vLLM response or error information

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
            # TaskExecutionFailed wraps DeserializationError when the SDK can't
            # decode the result from the endpoint. Most common cause: Python
            # version mismatch (e.g., local 3.12.12 vs endpoint 3.12.3).
            # Switching to AllCodeStrategies usually fixes this.
            # Don't reset the executor — the connection is fine, just the
            # serialization format is wrong.
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
