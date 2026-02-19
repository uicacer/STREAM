"""
Local End-to-End Test for the WebSocket Relay Server.

=============================================================================
WHAT THIS TEST DOES
=============================================================================
This test verifies the relay server works LOCALLY before we deploy anything
to Lakeshore. It simulates the full data flow:

    Fake Producer  →  Relay Server  →  Fake Consumer

The "fake producer" pretends to be Lakeshore, sending token messages.
The "fake consumer" pretends to be the STREAM proxy (or litellm_direct),
receiving those tokens.

If this test passes, we know:
  1. The relay server starts and accepts connections
  2. Producers can connect and send messages
  3. Consumers can connect and receive messages
  4. Messages flow from producer → relay → consumer correctly
  5. Buffering works (producer sends before consumer connects)
  6. Channel cleanup works after both sides disconnect
  7. The health endpoint works

=============================================================================
HOW TO RUN
=============================================================================
From the project root:

    python tests/test_relay_local.py

No external dependencies needed beyond `websockets` (already installed).
The test starts its own relay server on a random port, so no port conflicts.

=============================================================================
WHAT SUCCESS LOOKS LIKE
=============================================================================
    [1/5] Basic message flow ............. PASS
    [2/5] Buffered messages .............. PASS
    [3/5] Health endpoint ................ PASS
    [4/5] Multiple channels .............. PASS
    [5/5] Channel cleanup ................ PASS

    All 5 tests passed!
"""

import asyncio
import json

# ---------------------------------------------------------------------------
# We need to import the relay server's start function.
# sys.path manipulation ensures this works when run from the project root.
# ---------------------------------------------------------------------------
import os
import sys
import uuid

from websockets.asyncio.client import connect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stream.relay.server import channels, handle_connection

# =============================================================================
# HELPER: Start a relay server on a random port for testing
# =============================================================================
# We use port 0, which tells the OS to pick any available port.
# This avoids conflicts if the real relay is already running on 8765.


async def start_test_relay():
    """
    Start a relay server on a random available port.

    Returns:
        (server, port) — the server object and the port it's listening on.

    Why port 0?
        When you bind to port 0, the OS kernel picks a random available port.
        This is standard practice for tests — it means you can run tests even
        if port 8765 (or any other fixed port) is already in use.
    """
    from websockets.asyncio.server import serve

    server = await serve(handle_connection, "127.0.0.1", 0)
    # The server is now listening. Get the actual port the OS assigned.
    port = server.sockets[0].getsockname()[1]
    return server, port


# =============================================================================
# TEST 1: Basic Message Flow
# =============================================================================
# The simplest test: producer sends tokens, consumer receives them.
#
# Timeline:
#   1. Consumer connects first (this is the normal order in production —
#      STREAM's proxy/litellm_direct connects before Lakeshore starts generating)
#   2. Producer connects
#   3. Producer sends 3 token messages + 1 done message
#   4. Consumer receives all 4 messages in order
#   5. Both disconnect


async def test_basic_flow(port):
    """Test that tokens flow from producer → relay → consumer."""
    channel_id = str(uuid.uuid4())
    received = []

    # --- Consumer task ---
    # Connects first, waits for messages from the producer.
    async def consumer():
        async with connect(f"ws://127.0.0.1:{port}/consume/{channel_id}") as ws:
            # Receive messages until the producer sends "done"
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                received.append(data)
                if data.get("type") == "done":
                    break

    # --- Producer task ---
    # Connects second, sends tokens like vLLM would.
    async def producer():
        # Small delay so consumer connects first (simulates real timing)
        await asyncio.sleep(0.1)
        async with connect(f"ws://127.0.0.1:{port}/produce/{channel_id}") as ws:
            # Simulate vLLM generating 3 tokens
            await ws.send(json.dumps({"type": "token", "content": "Hello"}))
            await ws.send(json.dumps({"type": "token", "content": " world"}))
            await ws.send(json.dumps({"type": "token", "content": "!"}))
            # Signal generation is complete
            await ws.send(json.dumps({"type": "done"}))

    # Run both tasks concurrently (like they would in production)
    await asyncio.gather(consumer(), producer())

    # --- Verify ---
    assert len(received) == 4, f"Expected 4 messages, got {len(received)}"
    assert received[0] == {"type": "token", "content": "Hello"}
    assert received[1] == {"type": "token", "content": " world"}
    assert received[2] == {"type": "token", "content": "!"}
    assert received[3] == {"type": "done"}

    return True


# =============================================================================
# TEST 2: Buffered Messages
# =============================================================================
# Tests the edge case where the producer connects FIRST and sends tokens
# before the consumer is ready. The relay should buffer these messages
# and deliver them when the consumer connects.
#
# Why this matters:
#   In production, the consumer (proxy/litellm_direct) usually connects first.
#   But there's a race condition: if Globus routes the job very quickly,
#   the Lakeshore function might start producing tokens before the consumer
#   has connected. The relay must not lose these tokens.


async def test_buffered_messages(port):
    """Test that tokens sent before consumer connects are buffered."""
    channel_id = str(uuid.uuid4())
    received = []

    # --- Producer sends FIRST (before consumer connects) ---
    async with connect(f"ws://127.0.0.1:{port}/produce/{channel_id}") as producer_ws:
        await producer_ws.send(json.dumps({"type": "token", "content": "buffered1"}))
        await producer_ws.send(json.dumps({"type": "token", "content": "buffered2"}))
        await producer_ws.send(json.dumps({"type": "done"}))

    # Small delay to ensure producer messages are buffered
    await asyncio.sleep(0.1)

    # --- Consumer connects AFTER producer has already sent and disconnected ---
    async with connect(f"ws://127.0.0.1:{port}/consume/{channel_id}") as consumer_ws:
        # The relay should flush all buffered messages immediately
        while True:
            try:
                msg = await asyncio.wait_for(consumer_ws.recv(), timeout=2)
                data = json.loads(msg)
                received.append(data)
                if data.get("type") == "done":
                    break
            except TimeoutError:
                break

    # --- Verify ---
    assert len(received) == 3, f"Expected 3 buffered messages, got {len(received)}"
    assert received[0] == {"type": "token", "content": "buffered1"}
    assert received[1] == {"type": "token", "content": "buffered2"}
    assert received[2] == {"type": "done"}

    return True


# =============================================================================
# TEST 3: Health Endpoint
# =============================================================================
# The /health endpoint lets monitoring tools (and humans) check if the
# relay is running. It returns a JSON object with the server status and
# number of active channels.


async def test_health_endpoint(port):
    """Test the /health WebSocket endpoint."""
    async with connect(f"ws://127.0.0.1:{port}/health") as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(msg)

    # --- Verify ---
    assert data["status"] == "healthy", f"Expected 'healthy', got {data['status']}"
    assert "active_channels" in data, "Missing 'active_channels' field"
    assert "timestamp" in data, "Missing 'timestamp' field"

    return True


# =============================================================================
# TEST 4: Multiple Channels
# =============================================================================
# Tests that multiple independent conversations can run simultaneously.
# Each channel_id is a separate stream — tokens from one channel must NOT
# leak into another.
#
# Why this matters:
#   In production, multiple users might be chatting with Lakeshore at the
#   same time. Each user gets their own channel_id, and the relay must
#   keep them completely isolated.


async def test_multiple_channels(port):
    """Test that multiple channels work independently."""
    channel_a = str(uuid.uuid4())
    channel_b = str(uuid.uuid4())
    received_a = []
    received_b = []

    async def run_channel(channel_id, received_list, content_prefix):
        """Run a complete producer/consumer exchange on one channel."""

        async def consumer():
            async with connect(f"ws://127.0.0.1:{port}/consume/{channel_id}") as ws:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(msg)
                    received_list.append(data)
                    if data.get("type") == "done":
                        break

        async def producer():
            await asyncio.sleep(0.1)
            async with connect(f"ws://127.0.0.1:{port}/produce/{channel_id}") as ws:
                await ws.send(json.dumps({"type": "token", "content": f"{content_prefix}_token1"}))
                await ws.send(json.dumps({"type": "token", "content": f"{content_prefix}_token2"}))
                await ws.send(json.dumps({"type": "done"}))

        await asyncio.gather(consumer(), producer())

    # Run two channels simultaneously
    await asyncio.gather(
        run_channel(channel_a, received_a, "A"),
        run_channel(channel_b, received_b, "B"),
    )

    # --- Verify channel A got only A's tokens ---
    assert len(received_a) == 3, f"Channel A: expected 3 messages, got {len(received_a)}"
    assert received_a[0]["content"] == "A_token1"
    assert received_a[1]["content"] == "A_token2"

    # --- Verify channel B got only B's tokens ---
    assert len(received_b) == 3, f"Channel B: expected 3 messages, got {len(received_b)}"
    assert received_b[0]["content"] == "B_token1"
    assert received_b[1]["content"] == "B_token2"

    return True


# =============================================================================
# TEST 5: Channel Cleanup
# =============================================================================
# After both producer and consumer disconnect, the channel should be
# removed from the registry. This prevents memory leaks.


async def test_channel_cleanup(port):
    """Test that channels are cleaned up after both sides disconnect."""
    channel_id = str(uuid.uuid4())

    # Run a complete exchange
    async def consumer():
        async with connect(f"ws://127.0.0.1:{port}/consume/{channel_id}") as ws:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                if data.get("type") == "done":
                    break

    async def producer():
        await asyncio.sleep(0.1)
        async with connect(f"ws://127.0.0.1:{port}/produce/{channel_id}") as ws:
            await ws.send(json.dumps({"type": "done"}))

    await asyncio.gather(consumer(), producer())

    # Give the relay a moment to clean up
    await asyncio.sleep(0.2)

    # --- Verify the channel was removed ---
    assert (
        channel_id not in channels
    ), f"Channel {channel_id[:8]} still in registry after both sides disconnected"

    return True


# =============================================================================
# TEST RUNNER
# =============================================================================


async def run_all_tests():
    """Start the relay and run all tests."""
    # Start the relay on a random port
    server, port = await start_test_relay()
    print(f"Relay server started on ws://127.0.0.1:{port}")
    print()

    tests = [
        ("Basic message flow", test_basic_flow),
        ("Buffered messages", test_buffered_messages),
        ("Health endpoint", test_health_endpoint),
        ("Multiple channels", test_multiple_channels),
        ("Channel cleanup", test_channel_cleanup),
    ]

    passed = 0
    failed = 0

    for i, (name, test_fn) in enumerate(tests, 1):
        label = f"[{i}/{len(tests)}] {name}"
        try:
            await test_fn(port)
            print(f"  {label} {'.' * (40 - len(label))} PASS")
            passed += 1
        except Exception as e:
            print(f"  {label} {'.' * (40 - len(label))} FAIL")
            print(f"         Error: {e}")
            failed += 1

    print()
    if failed == 0:
        print(f"  All {passed} tests passed!")
    else:
        print(f"  {passed} passed, {failed} FAILED")

    # Clean up
    server.close()
    await server.wait_closed()

    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("  STREAM WebSocket Relay — Local End-to-End Tests")
    print("=" * 60)
    print()

    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
