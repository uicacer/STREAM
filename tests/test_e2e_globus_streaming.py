"""
End-to-End Test: Real Globus Compute → WebSocket Relay → Consumer

=============================================================================
WHAT THIS TEST DOES
=============================================================================
This test runs the REAL production flow through Globus Compute:

  Your machine                       Lakeshore HPC
  ──────────                         ──────────────
  1. Submit streaming job ──AMQP──→  Globus endpoint receives it
                                     ↓
  2. Consumer connects to relay      Remote function starts on GPU node
     (wss://tunnel/consume/UUID)     ↓
                                     3. Remote function connects to relay
                                        (wss://tunnel/produce/UUID)
                                     ↓
                                     4. Calls vLLM with stream=True
                                     ↓
  5. Consumer receives tokens  ←───  Tokens flow through relay in real-time
     in real-time via WebSocket      ↓
                                     6. Sends "done" + usage stats
  7. Consumer gets done signal  ←──  ↓
                                     7. Globus returns job status (not used)

Unlike test_relay_integration.py (which uses a fake vLLM), this test goes
through the REAL infrastructure: Globus Compute → Lakeshore → vLLM GPU.

=============================================================================
PREREQUISITES
=============================================================================
  1. Relay server running:      python -m stream.relay.server
  2. Tunnel active:             ssh -R 80:localhost:8765 nokey@localhost.run
  3. RELAY_URL set in .env:     RELAY_URL=https://<tunnel-url>
  4. Globus authenticated:      globus-compute-endpoint is running on Lakeshore
  5. vLLM running on Lakeshore: at least one model must be serving

=============================================================================
HOW TO RUN
=============================================================================
    python tests/test_e2e_globus_streaming.py
"""

import asyncio
import json
import os
import sys
import time

# Add project root to path so we can import STREAM modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env so RELAY_URL and other configs are available
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from websockets.asyncio.client import connect as async_connect  # noqa: E402

from stream.middleware.config import RELAY_URL  # noqa: E402
from stream.middleware.core.globus_compute_client import globus_client  # noqa: E402


async def test_e2e_streaming():
    """
    End-to-end test: submit a real streaming job via Globus Compute,
    consume tokens from the relay, and verify the output.
    """
    print(f"  RELAY_URL: {RELAY_URL}")
    print(f"  Globus available: {globus_client.is_available()}")
    print(f"  Endpoint ID: {globus_client.endpoint_id}")
    print(f"  vLLM URL: {globus_client.vllm_url}")
    print()

    if not RELAY_URL:
        print("  ERROR: RELAY_URL not configured in .env")
        print("  Set it to your tunnel URL (e.g., https://abc123.lhr.life)")
        return False

    if not globus_client.is_available():
        print("  ERROR: Globus Compute not configured")
        print("  Set GLOBUS_COMPUTE_ENDPOINT_ID in .env")
        return False

    # ---- Step 1: Submit the streaming job via Globus Compute ----
    print("  [1/4] Submitting streaming job via Globus Compute...")
    t_start = time.perf_counter()

    result = await globus_client.submit_streaming_inference(
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        temperature=0.7,
        max_tokens=50,
        model="lakeshore-qwen-vl-72b",  # Uses the default Lakeshore model
        relay_url=RELAY_URL,
    )

    t_submit = time.perf_counter()
    print(f"        Submit took: {t_submit - t_start:.2f}s")

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        if result.get("auth_required"):
            print("  → Please authenticate with Globus Compute first")
        return False

    channel_id = result["channel_id"]
    print(f"        Channel ID: {channel_id[:8]}...")
    print()

    # ---- Step 2: Connect to relay as consumer ----
    print("  [2/4] Connecting to relay as consumer...")
    consume_url = f"{RELAY_URL}/consume/{channel_id}"
    print(f"        URL: {consume_url[:60]}...")

    tokens = []
    usage = {}
    error_msg = None
    t_first_token = None

    try:
        async with async_connect(consume_url) as ws:
            print("        Connected! Waiting for tokens...")
            print()
            print("  [3/4] Receiving tokens:")
            print("        ", end="", flush=True)

            async for msg_str in ws:
                msg = json.loads(msg_str)

                if msg["type"] == "token":
                    token = msg["content"]
                    tokens.append(token)
                    print(token, end="", flush=True)

                    if t_first_token is None:
                        t_first_token = time.perf_counter()

                elif msg["type"] == "done":
                    usage = msg.get("usage", {})
                    break

                elif msg["type"] == "error":
                    error_msg = msg.get("message", "Unknown error")
                    print(f"\n        ERROR from Lakeshore: {error_msg}")
                    # Don't break — wait for "done" which follows error

    except Exception as e:
        print(f"\n  ERROR connecting to relay: {e}")
        return False

    t_done = time.perf_counter()
    print()  # newline after tokens
    print()

    # ---- Step 3: Display results ----
    print("  [4/4] Results:")
    full_text = "".join(tokens)
    print(f"        Full response: {repr(full_text)}")
    print(f"        Tokens received: {len(tokens)}")
    print(f"        Usage: {usage}")
    print()

    # Timing breakdown
    print("  Timing:")
    print(f"        Submit → first token: {(t_first_token or t_done) - t_start:.2f}s")
    if t_first_token:
        print(f"        First token → done:   {t_done - t_first_token:.2f}s")
    print(f"        Total:                {t_done - t_start:.2f}s")
    print()

    # ---- Step 4: Verify ----
    if error_msg:
        print(f"  FAIL: Remote function reported error: {error_msg}")
        return False

    if len(tokens) == 0:
        print("  FAIL: No tokens received")
        return False

    if not usage:
        print("  WARNING: No usage stats received (stream may have been interrupted)")

    print(f"  PASS: Received {len(tokens)} tokens via real Globus Compute + relay")
    return True


if __name__ == "__main__":
    print("=" * 65)
    print("  STREAM WebSocket Relay — End-to-End Globus Compute Test")
    print("  (real Lakeshore GPU inference via Globus → relay → consumer)")
    print("=" * 65)
    print()

    success = asyncio.run(test_e2e_streaming())

    print()
    if success:
        print("  " + "=" * 40)
        print("  TEST PASSED — Real streaming works!")
        print("  " + "=" * 40)
    else:
        print("  " + "=" * 40)
        print("  TEST FAILED")
        print("  " + "=" * 40)

    sys.exit(0 if success else 1)
