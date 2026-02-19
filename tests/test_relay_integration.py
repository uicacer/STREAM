"""
Integration Test: Simulates the full STREAM ↔ Lakeshore token streaming flow.

=============================================================================
WHAT THIS TEST DOES
=============================================================================
This test simulates the EXACT data flow of the WebSocket relay in production,
but entirely locally (no Globus, no Lakeshore, no ngrok needed):

    Simulated vLLM  →  remote_vllm_streaming logic  →  Relay  →  Consumer  →  SSE chunks

Specifically:
  1. Starts a local relay server
  2. Starts a fake vLLM HTTP server that returns streaming SSE responses
  3. Runs the remote_vllm_streaming function (the one that executes on Lakeshore)
     against the fake vLLM, streaming tokens through the relay
  4. A consumer connects to the relay and converts messages to SSE format
  5. Verifies the SSE output matches what the frontend expects

This catches integration issues between ALL the pieces:
  - The remote function's SSE parsing (reading vLLM's format)
  - The relay's message forwarding
  - The consumer's SSE generation (writing frontend's format)
  - The relay protocol (token/done/error messages)

=============================================================================
HOW TO RUN
=============================================================================
    python tests/test_relay_integration.py
"""

import asyncio
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websockets.asyncio.client import connect as async_connect
from websockets.asyncio.server import serve

from stream.middleware.core.globus_compute_client import remote_vllm_streaming
from stream.relay.server import handle_connection

# =============================================================================
# FAKE vLLM SERVER
# =============================================================================
# This simulates vLLM's /v1/chat/completions endpoint with stream=True.
# It returns SSE events in the exact format vLLM uses.

FAKE_VLLM_RESPONSE_TOKENS = [
    "Hello",
    " from",
    " Lakeshore",
    "!",
    " How",
    " can",
    " I",
    " help",
    "?",
]


class FakeVLLMHandler(BaseHTTPRequestHandler):
    """Simulates vLLM's streaming chat completions endpoint."""

    def do_POST(self):
        # Read the request body to verify it's well-formed
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request = json.loads(body)

        if not request.get("stream"):
            # Non-streaming response (not used in this test, but for completeness)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "choices": [{"message": {"content": "Hello from fake vLLM"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            self.wfile.write(json.dumps(response).encode())
            return

        # Streaming response — send SSE events like real vLLM
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        model = request.get("model", "test-model")

        # Send each token as a separate SSE event
        for token in FAKE_VLLM_RESPONSE_TOKENS:
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # Final chunk with usage stats and finish_reason
        final_chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
        self.wfile.flush()

        # End-of-stream marker
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        """Suppress HTTP server logs to keep test output clean."""
        pass


def start_fake_vllm():
    """Start a fake vLLM server on a random port. Returns (server, port)."""
    server = HTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


# =============================================================================
# TEST: Full Integration Flow
# =============================================================================


async def test_full_integration():
    """
    Simulate the complete production flow locally:
      Fake vLLM → remote_vllm_streaming → Relay → Consumer → SSE output
    """
    # ---- Start infrastructure ----
    # 1. Start the relay server
    relay_server = await serve(handle_connection, "127.0.0.1", 0)
    relay_port = relay_server.sockets[0].getsockname()[1]
    relay_url = f"ws://127.0.0.1:{relay_port}"

    # 2. Start the fake vLLM server
    vllm_server, vllm_port = start_fake_vllm()
    vllm_url = f"http://127.0.0.1:{vllm_port}"

    # 3. Generate a channel ID (like submit_streaming_inference does)
    channel_id = str(uuid.uuid4())

    print(f"  Relay:     {relay_url}")
    print(f"  Fake vLLM: {vllm_url}")
    print(f"  Channel:   {channel_id[:8]}...")
    print()

    received_sse = []

    # ---- Consumer (simulates _forward_lakeshore_streaming) ----
    async def consumer():
        async with async_connect(f"{relay_url}/consume/{channel_id}") as ws:
            async for msg_str in ws:
                msg = json.loads(msg_str)

                if msg["type"] == "token":
                    chunk = {
                        "choices": [{"index": 0, "delta": {"content": msg["content"]}}],
                    }
                    received_sse.append(f"data: {json.dumps(chunk)}")

                elif msg["type"] == "done":
                    usage = msg.get("usage", {})
                    if usage:
                        final_chunk = {
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": usage,
                        }
                        received_sse.append(f"data: {json.dumps(final_chunk)}")
                    received_sse.append("data: [DONE]")
                    break

                elif msg["type"] == "error":
                    received_sse.append(f"ERROR: {msg.get('message')}")

    # ---- Producer (simulates remote_vllm_streaming on Lakeshore) ----
    def producer():
        """Run the actual remote function against fake vLLM."""
        # This is the EXACT function that runs on Lakeshore.
        # We're testing it locally against our fake vLLM server.
        result = remote_vllm_streaming(
            vllm_url=vllm_url,
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            relay_url=relay_url,
            channel_id=channel_id,
        )
        return result

    # ---- Run both concurrently ----
    # Consumer connects first (as in production), producer follows
    consumer_task = asyncio.create_task(consumer())

    # Small delay so consumer connects first
    await asyncio.sleep(0.1)

    # Run the producer in a thread (it uses sync WebSocket client)
    producer_result = await asyncio.to_thread(producer)

    # Wait for consumer to finish
    await asyncio.wait_for(consumer_task, timeout=10)

    # ---- Clean up ----
    relay_server.close()
    await relay_server.wait_closed()
    vllm_server.shutdown()

    # ---- Verify results ----
    print("  Producer result (via Globus control plane):")
    print(f"    {producer_result}")
    print()

    print("  SSE chunks received by consumer:")
    for sse_line in received_sse:
        print(f"    {sse_line[:100]}")
    print()

    # Check producer completed successfully
    assert producer_result.get("ok") is True, f"Producer failed: {producer_result}"
    assert producer_result.get("tokens_sent") == len(
        FAKE_VLLM_RESPONSE_TOKENS
    ), f"Expected {len(FAKE_VLLM_RESPONSE_TOKENS)} tokens, got {producer_result.get('tokens_sent')}"

    # Check consumer received all tokens + done
    # Expected: one SSE line per token + one final chunk with usage + [DONE]
    token_lines = [line for line in received_sse if '"delta"' in line and '"content"' in line]
    assert len(token_lines) == len(
        FAKE_VLLM_RESPONSE_TOKENS
    ), f"Expected {len(FAKE_VLLM_RESPONSE_TOKENS)} token SSE lines, got {len(token_lines)}"

    # Verify the tokens are correct and in order
    for i, token in enumerate(FAKE_VLLM_RESPONSE_TOKENS):
        chunk_data = json.loads(token_lines[i][6:])  # strip "data: " prefix
        actual_content = chunk_data["choices"][0]["delta"]["content"]
        assert (
            actual_content == token
        ), f"Token {i}: expected {repr(token)}, got {repr(actual_content)}"

    # Check that usage stats came through
    usage_lines = [line for line in received_sse if '"usage"' in line]
    assert len(usage_lines) == 1, f"Expected 1 usage line, got {len(usage_lines)}"
    usage_data = json.loads(usage_lines[0][6:])
    assert usage_data["usage"]["total_tokens"] == 21

    # Check [DONE] marker
    assert received_sse[-1] == "data: [DONE]"

    # Reconstruct the full text from tokens
    full_text = "".join(FAKE_VLLM_RESPONSE_TOKENS)
    print(f"  Full text: {repr(full_text)}")
    print()

    return True


# =============================================================================
# TEST RUNNER
# =============================================================================


async def run_all_tests():
    tests = [
        ("Full integration (vLLM → relay → SSE)", test_full_integration),
    ]

    passed = 0
    failed = 0

    for i, (name, test_fn) in enumerate(tests, 1):
        label = f"[{i}/{len(tests)}] {name}"
        try:
            await test_fn()
            print(f"  {label} {'.' * (45 - len(label))} PASS")
            passed += 1
        except Exception as e:
            print(f"  {label} {'.' * (45 - len(label))} FAIL")
            print(f"         Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    if failed == 0:
        print(f"  All {passed} tests passed!")
    else:
        print(f"  {passed} passed, {failed} FAILED")

    return failed == 0


if __name__ == "__main__":
    print("=" * 65)
    print("  STREAM WebSocket Relay — Integration Tests")
    print("  (simulates full Lakeshore → relay → consumer flow)")
    print("=" * 65)
    print()

    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
