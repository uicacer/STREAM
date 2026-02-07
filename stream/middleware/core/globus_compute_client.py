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

import logging
import os
from typing import Any

from globus_compute_sdk import Executor
from globus_compute_sdk.serialize import CombinedCode, ComputeSerializer

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
GLOBUS_TASK_TIMEOUT = int(os.getenv("GLOBUS_TASK_TIMEOUT", "120"))


# =============================================================================
# REMOTE FUNCTION (Executes on Lakeshore)
# =============================================================================


def remote_vllm_inference(
    vllm_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Remote function that executes on Lakeshore via Globus Compute.

    This function is serialized and sent to the Globus endpoint, where it:
    1. Makes an HTTP request to the local vLLM server on Lakeshore
    2. Returns the inference result back to the middleware

    IMPORTANT: This function must be self-contained - it cannot reference
    external variables or imports from outside the function body.

    Args:
        vllm_url: URL of vLLM server (e.g., "http://ga-001:8000")
        model: Model identifier (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        messages: Chat messages in OpenAI format
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response (currently only non-streaming supported)

    Returns:
        Dictionary containing the vLLM response in OpenAI format

    Raises:
        Exception: If vLLM request fails
    """
    # Import inside the function so it's available when executed remotely
    import requests

    # Construct the full endpoint URL
    endpoint = f"{vllm_url}/v1/chat/completions"

    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,  # Currently only supporting non-streaming
    }

    # Make the HTTP request to vLLM
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=60,  # Timeout for the HTTP request itself
        )
        response.raise_for_status()  # Raise exception for 4xx/5xx

        return response.json()

    except requests.exceptions.RequestException as e:
        # Return error information that can be handled by middleware
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "status_code": getattr(e.response, "status_code", None)
            if hasattr(e, "response")
            else None,
        }


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

    async def submit_inference(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
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
            max_tokens: Maximum tokens to generate
            model: Model identifier on vLLM server

        Returns:
            Dictionary containing either:
            - Success: vLLM response in OpenAI format
            - Error: Error information with "error" key

        Raises:
            Exception: If Globus Compute is not available or submission fails
        """
        if not self.is_available():
            raise RuntimeError(
                "Globus Compute not configured. Set GLOBUS_COMPUTE_ENDPOINT_ID in .env"
            )

        logger.info(f"Submitting inference task to Globus endpoint {self.endpoint_id}")

        try:
            # Create executor and submit task
            # Using context manager ensures proper cleanup
            with Executor(endpoint_id=self.endpoint_id) as gce:
                # Use CombinedCode serialization to avoid module import issues
                # This serializes the function code directly instead of pickling
                gce.serializer = ComputeSerializer(strategy_code=CombinedCode())

                # Submit the function to execute remotely
                future = gce.submit(
                    remote_vllm_inference,
                    self.vllm_url,
                    model,
                    messages,
                    temperature,
                    max_tokens,
                    False,  # stream=False (streaming not yet implemented)
                )

                logger.debug("Waiting for Globus Compute task to complete...")

                # Wait for result with timeout
                result = future.result(timeout=GLOBUS_TASK_TIMEOUT)

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

        except Exception as e:
            logger.error(f"Globus Compute task failed: {str(e)}", exc_info=True)
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
