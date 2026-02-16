# WebSocket Relay Implementation Guide

**Purpose:** Step-by-step implementation plan for adding true token streaming from Lakeshore HPC to STREAM via a WebSocket side-channel relay. This document is written to be educational and self-contained — every decision is explained.

---

## Table of Contents

1. [Confirmation: Globus Compute Has No Native Streaming](#1-confirmation-globus-compute-has-no-native-streaming)
2. [Architecture Overview](#2-architecture-overview)
3. [What We're Building (Component Map)](#3-what-were-building-component-map)
4. [Component 1: The Relay Server](#4-component-1-the-relay-server)
5. [Component 2: The Streaming Remote Function](#5-component-2-the-streaming-remote-function)
6. [Component 3: Client-Side Relay Consumer](#6-component-3-client-side-relay-consumer)
7. [Component 4: Orchestration (Connecting It All)](#7-component-4-orchestration-connecting-it-all)
8. [Integration Into STREAM's Existing Pipeline](#8-integration-into-streams-existing-pipeline)
9. [The Demo Setup](#9-the-demo-setup)
10. [Error Handling and Edge Cases](#10-error-handling-and-edge-cases)
11. [Testing Plan](#11-testing-plan)
12. [File Change Summary](#12-file-change-summary)

---

## 1. Confirmation: Globus Compute Has No Native Streaming

Before building a side-channel relay, we thoroughly investigated whether Globus Compute has any native support for streaming intermediate results. **It does not.** Here is the evidence:

### SDK API Search (globus-compute-sdk v4.5.0)

We searched the entire installed SDK package for any streaming-related APIs:

| Search Term | Results |
|---|---|
| `stream` | **Zero matches** |
| `intermediate` | **Zero matches** |
| `partial` | **Zero matches** |
| `progress` | **Zero matches** |
| `callback` | **Zero matches** |
| `publish` | **Zero matches in SDK** (only found in `globus_compute_common` for internal task dispatch) |

### Executor Public API (Complete)

The `Executor` class has exactly these public methods — none support streaming:

| Method | Purpose |
|--------|---------|
| `submit(fn, *args, **kwargs)` | Submit a function, returns `ComputeFuture` |
| `submit_to_registered_function(function_id, ...)` | Execute pre-registered function |
| `register_function(fn, ...)` | Pre-register a function |
| `register_source_code(source, function_name, ...)` | Register raw source code |
| `reload_tasks(task_group_id)` | Reattach to previous tasks |
| `shutdown(wait, cancel_futures)` | Clean up resources |
| `get_worker_hardware_details()` | Get hardware info |

### ComputeFuture (Complete)

`ComputeFuture` extends Python's `concurrent.futures.Future` with exactly two additions:
- `task_id` — the UUID of the task
- `_metadata` — internal metadata dict

No progress hooks, no streaming iterators, no callback registration.

### Result Architecture

The result handling is strictly **one result per task**:
- Each `Result` message contains one `task_id` and one `data` blob
- The `_ResultWatcher` matches results to futures by `task_id` and calls `fut.set_result()` **exactly once**
- Python's `Future.set_result()` can only be called once (second call raises `InvalidStateError`)
- There is no mechanism to send multiple results for a single task

### GitHub Issues

| Issue | Title | Status | Relevant? |
|-------|-------|--------|-----------|
| [#410](https://github.com/globus/globus-compute/issues/410) | Streaming results with WebSockets | **Closed** | About delivering **final** results via AMQP instead of polling. NOT intermediate results. |
| [#509](https://github.com/globus/globus-compute/issues/509) | Stream status information from endpoint to client | **Open** (since 2021, unimplemented) | About task lifecycle status ("running", "waiting"), NOT function output. |
| [#301](https://github.com/globus/globus-compute/issues/301) | Async Version of SDK | **Open** (since 2020, 1 comment) | Brief mention of async generators. No implementation. |

### Why Modifying the SDK Is Impractical

Even if we wanted to add streaming ourselves, the architecture makes it very difficult:

1. **Process boundary:** Remote functions execute in isolated **worker processes**. The AMQP connection lives in the **endpoint daemon process**. These are separate OS processes with a serialization boundary — the worker has no way to publish messages to the daemon's AMQP connection.

2. **Would require 4 changes:**
   - IPC channel between worker and daemon
   - New message type in the Globus Compute protocol
   - New client-side API (`stream_results()`)
   - Changes to AMQP queue topology

3. **SDK update fragility:** Any of these internals could change in a Globus SDK update, breaking our custom streaming.

**Conclusion:** A side-channel relay is not just a shortcut — it's the only practical approach for a single developer.

### What STREAM's Own Code Already Says

Our codebase already documents this limitation in [`stream/proxy/app.py:292-306`](stream/proxy/app.py#L292-L306):

```python
# WHY SIMULATE STREAMING FOR GLOBUS COMPUTE?
# =========================================================================
# Globus Compute is a Function-as-a-Service (FaaS) system:
# - You submit a function → it runs remotely → returns complete result
# - There's NO way to get partial results while the function is running
# - This is fundamentally different from Local/Cloud tiers which support true streaming
```

---

## 2. Architecture Overview

### Current Architecture (Batch)

```
User types message
        │
        ▼
   STREAM client (React UI)
        │
        ▼  HTTP POST /chat
   STREAM middleware (FastAPI)
        │
        ▼  forward_to_litellm() or forward_direct()
   Globus Compute Executor
        │
        ▼  AMQP: serialize function + args → Globus cloud → Lakeshore endpoint
   Remote function on Lakeshore GPU
        │
        ▼  POST to vLLM (stream=False) — waits for COMPLETE response
   vLLM generates all tokens
        │
        ▼  Returns complete JSON
   Remote function returns complete result via AMQP
        │
        ▼  result appears ~5 seconds later
   STREAM middleware receives complete response
        │
        ▼  _convert_json_to_sse_stream() or word-by-word splitting
   Simulated streaming (fake: splits complete response into word chunks)
        │
        ▼  SSE events with 50ms delays
   React UI displays text progressively (but it's all pre-generated)
```

### New Architecture (WebSocket Relay)

```
User types message
        │
        ▼
   STREAM client (React UI)
        │
        ▼  HTTP POST /chat
   STREAM middleware (FastAPI)
        │
        ├──────────────────────────────────────────────┐
        │  CONTROL PLANE                               │  DATA PLANE
        │                                              │
        ▼  Globus Compute submit                       ▼  WebSocket connect to relay
   Executor.submit(streaming_fn, ..., task_id)    ws://relay:8765 + send task_id
        │                                              │
        ▼  AMQP routing (~1-2s)                        │  (waiting for tokens)
   Remote function starts on Lakeshore GPU             │
        │                                              │
        ├──── WebSocket connect to relay ──────────────┤
        │     ws://relay:8765 + send task_id           │
        │                                              │
        ▼  POST to vLLM (stream=True)                  │
   vLLM generates token₁                              │
        │                                              │
        ▼  ws.send(token₁) → relay → client receives ─┤──→ yield SSE event
   vLLM generates token₂                              │
        │                                              │
        ▼  ws.send(token₂) → relay → client receives ─┤──→ yield SSE event
   ...                                                 │
   vLLM generates token_N                              │
        │                                              │
        ▼  ws.send(token_N) → relay → client receives ─┤──→ yield SSE event
        │                                              │
        ▼  ws.send("[DONE]") → relay → client receives ┤──→ yield "data: [DONE]"
        │                                              │
        ▼  function returns {"status": "streamed"}     │
   Globus Compute resolves Future (we ignore it)       │
```

The key insight: **Globus Compute handles the control plane** (authentication, job submission, getting through the firewall) while the **WebSocket relay handles the data plane** (real-time token delivery). They work in parallel, not sequentially.

---

## 3. What We're Building (Component Map)

Four components need to be built or modified:

| Component | Where It Runs | What It Does | New or Modified? |
|-----------|--------------|--------------|-----------------|
| **Relay Server** | Public server (cloud VM or Lakeshore with open port) | Forwards WebSocket messages between two clients with the same task_id | **New file** |
| **Streaming Remote Function** | Lakeshore GPU node (via Globus Compute) | Calls vLLM with `stream=True`, forwards tokens to relay via WebSocket | **Modified:** `globus_compute_client.py` |
| **Relay Consumer** | STREAM middleware (user's machine) | Connects to relay, receives tokens, yields them as SSE events | **New function** in `litellm_direct.py` or `globus_compute_client.py` |
| **Orchestration** | STREAM middleware | Coordinates Globus submit + relay connection, integrates with existing SSE pipeline | **Modified:** `litellm_direct.py`, `proxy/app.py` |

### Files That Change

| File | Change |
|------|--------|
| `relay_server.py` (new) | The relay server — standalone script, deployed separately |
| `stream/middleware/core/globus_compute_client.py` | Add `_REMOTE_FN_STREAMING_SOURCE`, add `submit_inference_streaming()` |
| `stream/middleware/core/litellm_direct.py` | Modify `_forward_lakeshore()` to use relay streaming |
| `stream/proxy/app.py` | Modify `_route_via_globus_compute()` for server mode streaming |
| `stream/middleware/config.py` | Add `RELAY_URL` config variable |
| `stream/desktop/config.py` | Add `RELAY_URL` desktop default |
| `.env` | Add `RELAY_URL` |

---

## 4. Component 1: The Relay Server

### What It Does

The relay server is a standalone WebSocket server that acts as a message broker between two parties:
1. The **Lakeshore worker** (producer — sends tokens)
2. The **STREAM client** (consumer — receives tokens)

Both connect to the relay by sending a **task_id** as their first message. The relay groups connections by task_id and forwards all subsequent messages from one to all others in the same group.

### Why It's So Simple

The relay doesn't understand what it's forwarding. It doesn't parse JSON, validate tokens, or do any AI-related processing. It's a generic message forwarder — like a conference call bridge. You join a "room" (identified by task_id) and anything anyone says in the room is heard by everyone else.

This simplicity is deliberate:
- Fewer lines of code = fewer bugs
- No state to persist = no database, no disk
- No computation = negligible CPU usage
- Protocol-agnostic = could forward anything, not just AI tokens

### The Code

```python
#!/usr/bin/env python3
"""
WebSocket Relay Server for STREAM Lakeshore Token Streaming.

This server acts as a message bridge between a Lakeshore GPU worker
(which generates AI tokens) and a STREAM client (which displays them).

Protocol:
  1. Client connects via WebSocket
  2. First message is the task_id (UUID string)
  3. All subsequent messages are forwarded to other clients with the same task_id
  4. When a connection closes, it's removed from its channel

Deployment:
  python relay_server.py                    # Default: 0.0.0.0:8765
  python relay_server.py --port 9000        # Custom port
  RELAY_PORT=9000 python relay_server.py    # Via environment variable
"""

import argparse
import asyncio
import logging
import os
import time
from collections import defaultdict

import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("relay")

# Active channels: task_id → set of connected WebSocket clients
# defaultdict(set) means: if a task_id isn't in the dict yet, create an empty set
channels: dict[str, set] = defaultdict(set)

# Track channel creation time for cleanup
channel_created: dict[str, float] = {}

# Maximum age for a channel (seconds) — prevents leaked channels from accumulating
MAX_CHANNEL_AGE = 600  # 10 minutes (generous for any LLM response)

# Maximum connections per channel (should be exactly 2: producer + consumer)
MAX_CONNECTIONS_PER_CHANNEL = 5  # Allow some headroom for reconnects


async def handler(websocket):
    """
    Handle a single WebSocket connection.

    The first message identifies which task this connection belongs to.
    All subsequent messages are forwarded to other connections in the same channel.
    """
    task_id = None

    try:
        # Step 1: Read the task_id (first message)
        task_id = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        logger.info(f"Connection joined channel: {task_id[:8]}...")

        # Enforce connection limit per channel
        if len(channels[task_id]) >= MAX_CONNECTIONS_PER_CHANNEL:
            logger.warning(f"Channel {task_id[:8]}... full, rejecting connection")
            await websocket.close(4001, "Channel full")
            return

        # Join the channel
        channels[task_id].add(websocket)
        if task_id not in channel_created:
            channel_created[task_id] = time.time()

        # Step 2: Forward all subsequent messages to other connections in this channel
        async for message in websocket:
            # Get all other connections in the same channel
            peers = channels.get(task_id, set())
            dead_peers = set()

            for peer in peers:
                if peer != websocket:
                    try:
                        await peer.send(message)
                    except websockets.ConnectionClosed:
                        dead_peers.add(peer)

            # Clean up any dead connections we discovered
            for dead in dead_peers:
                channels[task_id].discard(dead)

    except asyncio.TimeoutError:
        logger.warning("Connection timed out waiting for task_id")
    except websockets.ConnectionClosed:
        pass  # Normal disconnection
    except Exception as e:
        logger.error(f"Handler error: {e}")
    finally:
        # Remove this connection from its channel
        if task_id and task_id in channels:
            channels[task_id].discard(websocket)
            # Clean up empty channels
            if not channels[task_id]:
                del channels[task_id]
                channel_created.pop(task_id, None)
                logger.info(f"Channel {task_id[:8]}... closed (empty)")


async def cleanup_stale_channels():
    """
    Periodically remove channels that have been open too long.

    This prevents memory leaks from connections that were never properly closed
    (e.g., if the Lakeshore worker crashed mid-stream and the client disconnected).
    """
    while True:
        await asyncio.sleep(60)  # Check every minute
        now = time.time()
        stale = [
            tid for tid, created in channel_created.items()
            if now - created > MAX_CHANNEL_AGE
        ]
        for tid in stale:
            # Close all connections in the stale channel
            for ws in list(channels.get(tid, set())):
                try:
                    await ws.close(4002, "Channel expired")
                except Exception:
                    pass
            channels.pop(tid, None)
            channel_created.pop(tid, None)
            logger.info(f"Cleaned up stale channel: {tid[:8]}...")


async def main(host: str, port: int):
    """Start the relay server."""
    logger.info(f"STREAM WebSocket Relay starting on {host}:{port}")

    # Start the cleanup task in the background
    asyncio.create_task(cleanup_stale_channels())

    # Start the WebSocket server
    async with websockets.serve(
        handler,
        host,
        port,
        # Ping every 30 seconds to detect dead connections
        ping_interval=30,
        ping_timeout=10,
        # Maximum message size (1MB — plenty for SSE chunks which are tiny)
        max_size=1_048_576,
    ):
        logger.info(f"Relay server listening on ws://{host}:{port}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STREAM WebSocket Relay Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RELAY_PORT", "8765")),
        help="Listen port (default: 8765)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
```

### Dependencies

The relay server needs only one package beyond the standard library:

```
pip install websockets
```

That's it. No FastAPI, no STREAM codebase, no Globus SDK. The relay is completely standalone.

---

## 5. Component 2: The Streaming Remote Function

### What Changes

Currently, the remote function ([`globus_compute_client.py:73-120`](stream/middleware/core/globus_compute_client.py#L73-L120)) calls vLLM with `stream=False` and returns the complete response as one JSON object. We need a second remote function that:

1. Calls vLLM with `stream=True` (vLLM returns Server-Sent Events)
2. Opens a WebSocket connection to the relay
3. Forwards each SSE chunk through the relay as it arrives
4. Returns a summary when done (Globus still needs a return value)

### Why a Separate Function (Not Modifying the Existing One)

The existing `remote_vllm_inference` function works reliably for the current batch approach. We don't want to risk breaking it. Instead, we add a new `remote_vllm_inference_streaming` function alongside it. The orchestration layer chooses which one to use based on whether the relay is configured.

This also makes the demo easy: you can toggle between batch and streaming by setting or unsetting `RELAY_URL` in the `.env` file.

### The Code

This gets added to `globus_compute_client.py` after the existing `_REMOTE_FN_SOURCE`:

```python
_REMOTE_FN_STREAMING_SOURCE = """\
def remote_vllm_inference_streaming(vllm_url, model, messages, temperature,
                                     max_tokens, task_id, relay_url):
    \"\"\"
    Runs on Lakeshore via Globus Compute.
    Calls vLLM with streaming and forwards tokens through the WebSocket relay.

    Unlike remote_vllm_inference (which returns the complete response),
    this function streams tokens in real-time through a side-channel WebSocket.
    The Globus Compute return value is just a summary — the actual content
    was already delivered to the client through the relay.

    Args:
        vllm_url: vLLM server URL on Lakeshore (e.g., "http://ga-001:8000")
        model: Model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        messages: Chat messages in OpenAI format
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        task_id: UUID that identifies this request — used to pair
                 this producer with the STREAM client consumer on the relay
        relay_url: WebSocket URL of the relay server (e.g., "ws://relay.example.com:8765")
    \"\"\"
    try:
        import json
        import requests

        # =====================================================================
        # STEP 1: Connect to the WebSocket relay
        # =====================================================================
        # We use the `websocket-client` library (synchronous WebSocket client).
        # It must be installed on the Lakeshore endpoint's Python environment.
        #
        # Why synchronous? Globus Compute worker functions run in regular
        # (non-async) Python. Using asyncio here would require creating an
        # event loop inside the worker, which adds complexity for no benefit
        # since we're doing sequential I/O anyway.
        try:
            import websocket as ws_client
            ws = ws_client.create_connection(relay_url, timeout=10)
            ws.send(task_id)  # Join the channel for this task
            relay_connected = True
        except Exception as e:
            # If relay connection fails, fall back to batch mode
            # (return complete response like the non-streaming function)
            relay_connected = False
            fallback_reason = str(e)

        # =====================================================================
        # STEP 2: Call vLLM with streaming enabled
        # =====================================================================
        endpoint = f"{vllm_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": relay_connected,  # Only stream if relay is connected
        }

        response = requests.post(endpoint, json=payload, stream=relay_connected, timeout=120)

        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text
            error_result = {
                "error": f"{response.status_code} Error: {error_body}",
                "error_type": "HTTPError",
                "status_code": response.status_code,
            }
            if relay_connected:
                # Send error through relay so client knows what happened
                ws.send(json.dumps({"error": error_result}))
                ws.close()
            return error_result

        # =====================================================================
        # STEP 3: Forward SSE chunks through the relay (or return batch)
        # =====================================================================
        if not relay_connected:
            # Relay connection failed — return the complete response
            # (same behavior as the non-streaming function)
            result = response.json()
            result["_relay_fallback"] = fallback_reason
            return result

        # Streaming mode: forward each SSE line through the WebSocket relay
        token_count = 0
        accumulated_usage = {}

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode("utf-8")

            if line_text.startswith("data: [DONE]"):
                # Send the DONE signal through the relay
                ws.send("data: [DONE]")
                break

            if line_text.startswith("data: "):
                # Forward the SSE chunk as-is through the relay
                ws.send(line_text)
                token_count += 1

                # Try to capture usage from the chunk (usually in the last chunk)
                try:
                    chunk = json.loads(line_text[6:])
                    if "usage" in chunk and chunk["usage"]:
                        accumulated_usage = chunk["usage"]
                except (json.JSONDecodeError, KeyError):
                    pass

        ws.close()

        # =====================================================================
        # STEP 4: Return summary through Globus Compute
        # =====================================================================
        # The actual content was already streamed through the relay.
        # This return value is just metadata for logging/verification.
        return {
            "status": "streamed",
            "tokens_sent": token_count,
            "usage": accumulated_usage,
            "relay_url": relay_url,
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "error_type": type(e).__name__,
        }
"""

# Compile the streaming function the same way as the batch function
_ns_streaming = {}
exec(compile(_REMOTE_FN_STREAMING_SOURCE, "<remote_vllm_inference_streaming>", "exec"), _ns_streaming)
remote_vllm_inference_streaming = _ns_streaming["remote_vllm_inference_streaming"]
```

### Lakeshore Dependency: `websocket-client`

The streaming remote function uses the `websocket-client` library to connect to the relay. This package must be installed in the Python environment on Lakeshore's endpoint.

To install it:

```bash
ssh lakeshore.uic.edu
pip install websocket-client    # or: conda install websocket-client
```

This is a pure Python package with no compiled dependencies — it should install cleanly on any Lakeshore node.

**Important:** The package name on PyPI is `websocket-client`, but you import it as `websocket`. Don't confuse it with the `websockets` package (async, used by the relay server) — they're different libraries:
- `websockets` — async, used by the relay server
- `websocket-client` — sync, used by the remote function on Lakeshore

---

## 6. Component 3: Client-Side Relay Consumer

### What It Does

While Globus Compute handles the job submission, the STREAM middleware simultaneously connects to the relay and listens for tokens. As tokens arrive from Lakeshore through the relay, the consumer yields them as SSE lines — the exact same format that `streaming.py` already processes.

### Where It Goes

This function is added to `litellm_direct.py` (for desktop mode) and a parallel version in `proxy/app.py` (for server/Docker mode). Both produce the same SSE output format.

### The Code (Desktop Mode — `litellm_direct.py`)

```python
async def _forward_lakeshore_streaming(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
    relay_url: str,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Lakeshore through the WebSocket relay.

    This function does two things in parallel:
    1. Submits the streaming remote function via Globus Compute (control plane)
    2. Connects to the relay and yields tokens as they arrive (data plane)

    The function yields SSE-formatted lines — the exact same format as
    _forward_lakeshore() (batch mode) and litellm streaming. The downstream
    pipeline (streaming.py) doesn't know the difference.

    Args:
        model: Friendly model name (e.g., "lakeshore-qwen")
        messages: Chat messages in OpenAI format
        temperature: Sampling temperature
        correlation_id: Unique request ID for log tracing
        relay_url: WebSocket URL of the relay server
    """
    import uuid
    import websockets

    gc = _proxy_app.globus_client
    if not gc or not gc.is_available():
        raise HTTPException(status_code=503, detail="Globus Compute not configured")

    # Generate a unique task_id for this request.
    # This ID is shared between the Globus Compute submission and the relay
    # connection — it's how the relay knows which producer goes with which consumer.
    #
    # We use our own UUID rather than Globus Compute's task_id because we need
    # the ID BEFORE submitting the task (to pass it as an argument to the remote
    # function), but Globus doesn't assign the task_id until after submission.
    relay_task_id = str(uuid.uuid4())

    # Get the vLLM model name (strip the "openai/" prefix that litellm needs)
    entry = _MODEL_MAP.get(model, {})
    vllm_model = entry.get("model", "").replace("openai/", "")

    model_limits = MODEL_CONTEXT_LIMITS.get(model, {})
    max_tokens = model_limits.get("reserve_output", 2048)

    logger.info(
        f"[{correlation_id}] Lakeshore streaming via relay: {model} → {vllm_model}",
        extra={"correlation_id": correlation_id, "model": model},
    )

    # =========================================================================
    # STEP 1: Submit the streaming function via Globus Compute
    # =========================================================================
    # This is async (non-blocking) — we don't await the result because the
    # actual response comes through the relay, not through Globus.
    #
    # We use asyncio.to_thread because gc._get_executor().submit() is blocking
    # (it serializes the function and sends it over AMQP).
    from stream.middleware.core.globus_compute_client import (
        remote_vllm_inference_streaming,
    )

    def _submit():
        """Submit the streaming function (runs in a thread)."""
        executor = gc._get_executor()
        future = executor.submit(
            remote_vllm_inference_streaming,
            gc.vllm_url,
            vllm_model,
            messages,
            temperature,
            max_tokens,
            relay_task_id,
            relay_url,
        )
        return future

    # Start the submission in a background thread (non-blocking)
    submit_task = asyncio.get_event_loop().run_in_executor(None, _submit)

    # =========================================================================
    # STEP 2: Connect to the relay and receive tokens
    # =========================================================================
    # We connect to the relay immediately — even before Globus has routed
    # the task to Lakeshore. The relay connection will just wait silently
    # until the remote function connects and starts sending tokens.
    #
    # This means the relay connection is established in parallel with the
    # Globus AMQP routing, saving ~1-2 seconds of sequential wait time.
    try:
        async with websockets.connect(
            relay_url,
            open_timeout=15,       # Allow time for relay to be reachable
            close_timeout=5,
            ping_interval=20,      # Keep connection alive
            ping_timeout=10,
        ) as ws:
            # Join the channel for our task
            await ws.send(relay_task_id)

            logger.info(
                f"[{correlation_id}] Connected to relay, waiting for tokens...",
                extra={"correlation_id": correlation_id},
            )

            # Receive tokens as they arrive from Lakeshore
            async for message in ws:
                # Check for DONE signal
                if message.strip() == "data: [DONE]":
                    yield "data: [DONE]"
                    break

                # Check for error from remote function
                if message.startswith("{"):
                    try:
                        data = json.loads(message)
                        if "error" in data:
                            logger.error(
                                f"[{correlation_id}] Error from Lakeshore: {data['error']}",
                                extra={"correlation_id": correlation_id},
                            )
                            raise HTTPException(
                                status_code=503,
                                detail=f"Lakeshore inference failed: {data['error']}",
                            )
                    except json.JSONDecodeError:
                        pass

                # Forward the SSE line as-is
                # The message is already in "data: {json}" format from vLLM
                yield message

    except websockets.ConnectionClosed as e:
        logger.warning(
            f"[{correlation_id}] Relay connection closed: {e}",
            extra={"correlation_id": correlation_id},
        )
    except OSError as e:
        # Relay unreachable — fall back to batch mode
        logger.warning(
            f"[{correlation_id}] Relay unreachable ({e}), falling back to batch mode",
            extra={"correlation_id": correlation_id},
        )
        # Fall back to the existing batch approach
        async for line in _forward_lakeshore(model, messages, temperature, correlation_id):
            yield line
        return

    # =========================================================================
    # STEP 3: Check the Globus Compute result (for error handling)
    # =========================================================================
    # The actual content was already streamed through the relay.
    # We check the Globus return value to see if there were any errors
    # that the relay connection might have missed.
    try:
        future = await submit_task  # Get the Future from the submission
        # Wait briefly for the result (should already be done since streaming finished)
        result = await asyncio.to_thread(future.result, timeout=10)
        if isinstance(result, dict) and "error" in result:
            logger.error(
                f"[{correlation_id}] Globus result had error: {result['error']}",
                extra={"correlation_id": correlation_id},
            )
        elif isinstance(result, dict) and result.get("status") == "streamed":
            logger.info(
                f"[{correlation_id}] Streaming completed: {result.get('tokens_sent', '?')} tokens",
                extra={"correlation_id": correlation_id},
            )
    except Exception as e:
        # The streaming already completed successfully through the relay,
        # so a Globus error here is just logged, not raised.
        logger.warning(
            f"[{correlation_id}] Globus result check failed (streaming was OK): {e}",
            extra={"correlation_id": correlation_id},
        )
```

### How the Existing `_forward_lakeshore()` Changes

The existing function becomes the fallback. We add a check at the top of the Lakeshore routing:

```python
async def _forward_lakeshore(
    model: str,
    messages: list[dict],
    temperature: float,
    correlation_id: str,
) -> AsyncGenerator[str, None]:
    """
    Call the Lakeshore Globus Compute client (desktop mode).
    If RELAY_URL is configured, uses true streaming through the relay.
    Otherwise, falls back to batch mode (current behavior).
    """
    from stream.middleware.config import RELAY_URL

    # If relay is configured, use true streaming
    if RELAY_URL:
        async for line in _forward_lakeshore_streaming(
            model, messages, temperature, correlation_id, RELAY_URL
        ):
            yield line
        return

    # Otherwise, existing batch mode (unchanged)
    gc = _proxy_app.globus_client
    # ... rest of existing code ...
```

---

## 7. Component 4: Orchestration (Connecting It All)

### The Timing Dance

The most subtle part of this implementation is getting the timing right. Here's what happens in order:

```
Time 0.000s  Client sends chat message to STREAM middleware
         │
Time 0.001s  Middleware generates relay_task_id (UUID)
         │
         ├──── PARALLEL ────────────────────────────────────────────┐
         │                                                          │
Time 0.002s  [Thread 1] Start Globus submit          [Main] Connect to relay
         │   Serialize function + args                  ws.send(relay_task_id)
         │   Send over AMQP                             Now waiting for messages...
         │
Time ~1.5s   AMQP routing through Globus cloud            (still waiting)
         │
Time ~2.0s   Task arrives at Lakeshore endpoint            (still waiting)
         │   Worker deserializes function
         │   Function starts executing
         │
Time ~2.1s   Function connects to relay                    (still waiting)
         │   ws.send(relay_task_id)  ← joins same channel
         │
Time ~2.2s   Function calls vLLM with stream=True          (still waiting)
         │   vLLM begins processing prompt
         │
Time ~3.0s   vLLM generates first token
         │   Function sends token through relay ──────────→ Client receives first token!
         │                                                   yield "data: {chunk}"
Time ~3.05s  vLLM generates second token
         │   Function sends token through relay ──────────→ Client receives second token
         │                                                   yield "data: {chunk}"
         │   ... (tokens arrive every ~50ms) ...
         │
Time ~5.0s   vLLM generates last token
         │   Function sends "data: [DONE]" ───────────────→ Client receives DONE
         │   Function closes WebSocket                       Stream complete
         │   Function returns {"status": "streamed"}
         │
Time ~5.1s   Globus resolves the Future
         │   Middleware checks result (just for error logging)
         │
         └──────────────────────────────────────────────────────────┘
```

**Key detail:** The relay connection is established at time 0.002s, **in parallel** with the Globus submission. The client doesn't wait for Globus to finish routing before connecting to the relay. This means as soon as the remote function starts sending tokens (~3s), the client is already listening.

### Config Changes

**`stream/middleware/config.py`:**

```python
# WebSocket Relay URL for Lakeshore true streaming.
# When set, Lakeshore responses stream through this relay in real-time
# instead of waiting for the complete response via Globus Compute.
# When empty/None, falls back to batch mode (current behavior).
# Example: "ws://your-relay-server.com:8765"
RELAY_URL = os.getenv("RELAY_URL", "")
```

**`stream/desktop/config.py`** — add to `apply_desktop_defaults()`:

```python
# WebSocket relay URL (optional — enables true Lakeshore streaming).
# In desktop mode, we use setdefault so the user's .env value takes priority.
# If not set, Lakeshore uses batch mode (current behavior).
os.environ.setdefault("RELAY_URL", "")
```

**`.env`:**

```bash
# WebSocket relay for Lakeshore true streaming (optional)
# Set this to your relay server URL to enable token streaming from Lakeshore
# RELAY_URL=ws://your-relay-server.com:8765
```

---

## 8. Integration Into STREAM's Existing Pipeline

### How the SSE Pipeline Currently Works

The streaming pipeline is a chain of async generators, each yielding SSE-formatted lines:

```
forward_to_litellm() or forward_direct()
    │
    │ yields: "data: {"choices": [{"delta": {"content": "Hello"}}]}"
    │ yields: "data: {"choices": [{"delta": {"content": " world"}}]}"
    │ yields: "data: [DONE]"
    │
    ▼
stream_with_gap_warnings()    ← adds warnings if no chunk arrives within 5 seconds
    │
    ▼
create_streaming_response()   ← adds metadata, cost tracking, fallback handling
    │
    ▼
FastAPI StreamingResponse     ← sends SSE events to the React frontend via HTTP
    │
    ▼
React ChatContainer.tsx       ← displays tokens in the chat bubble
```

### What the Relay Consumer Must Produce

The relay consumer must yield **exactly the same SSE format** as the existing batch approach. This means:

```
"data: {"choices": [{"index": 0, "delta": {"content": "Hello"}}]}"
"data: {"choices": [{"index": 0, "delta": {"content": " world"}}]}"
"data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {...}}"
"data: [DONE]"
```

This format comes directly from vLLM's streaming output. Since the remote function forwards vLLM's SSE lines unchanged through the relay, and the consumer yields them as-is, the format is automatically correct. No translation needed.

### What Doesn't Change

The following files require **zero modifications**:

| File | Why It Stays The Same |
|------|----------------------|
| `streaming.py` | Consumes SSE lines regardless of source (batch/streaming). Already handles gap warnings for Lakeshore. |
| `chat.py` | Calls `create_streaming_response()` which calls `forward_to_litellm()`. No change needed. |
| `litellm_client.py` | Dispatches to `forward_direct()` in desktop mode. No change needed. |
| `query_router.py` | Routes queries to tiers. Tier selection is unaffected by streaming method. |
| React frontend | Parses SSE events from the `/chat` endpoint. Same format, no change needed. |

The entire change is contained in the "bottom" of the pipeline — how we get tokens from Lakeshore. Everything above (orchestration, cost tracking, UI) stays the same.

---

## 9. The Demo Setup

### What You Need

1. **A cloud VM** (DigitalOcean, AWS Lightsail, etc.) — ~$5/month
   - OR: Any machine with a public IP that Lakeshore can reach
   - OR: Ask ACER to open a port on Lakeshore (see `LAKESHORE_CONNECTIVITY.md`)

2. **Python 3.10+** on the VM (for the relay server)

3. **`websocket-client` package** installed on Lakeshore's endpoint environment

4. **`websockets` package** installed in STREAM's environment (already in most Python installs)

### Step-by-Step Setup

**Step 1: Deploy the relay server**

```bash
# On the cloud VM:
pip install websockets
python relay_server.py --port 8765

# Verify it's running:
# From your laptop:
python -c "
import asyncio, websockets
async def test():
    async with websockets.connect('ws://YOUR_VM_IP:8765') as ws:
        await ws.send('test-channel')
        print('Connected successfully!')
asyncio.run(test())
"
```

**Step 2: Install `websocket-client` on Lakeshore**

```bash
ssh lakeshore.uic.edu
pip install websocket-client

# Verify it can reach the relay:
python -c "
import websocket
ws = websocket.create_connection('ws://YOUR_VM_IP:8765')
ws.send('test-channel')
ws.send('hello from lakeshore')
ws.close()
print('Lakeshore can reach relay!')
"
```

**Step 3: Configure STREAM**

Add to your `.env`:

```bash
RELAY_URL=ws://YOUR_VM_IP:8765
```

**Step 4: Add the code changes** (Components 2, 3, 4 described above)

**Step 5: Run the demo**

```bash
# Start STREAM desktop app
python -m stream.desktop.main

# Send a message through the UI
# Watch the terminal output — you should see:
#   "Lakeshore streaming via relay: lakeshore-qwen → Qwen/Qwen2.5-1.5B-Instruct"
#   "Connected to relay, waiting for tokens..."
#   "Streaming completed: 147 tokens"
```

### Demo Comparison

To demonstrate the difference to ACER:

1. **Without relay** (current behavior): Set `RELAY_URL=` (empty). Send a message. Note the ~5 second blank screen before text appears.

2. **With relay**: Set `RELAY_URL=ws://YOUR_VM_IP:8765`. Send the same message. Note text starts appearing at ~3 seconds and flows progressively.

Record both with a screen recording tool for a side-by-side comparison.

---

## 10. Error Handling and Edge Cases

### What Can Go Wrong

| Scenario | What Happens | How We Handle It |
|----------|-------------|-----------------|
| Relay server is down | Client can't connect to relay | Fall back to batch mode (existing `_forward_lakeshore()`) |
| Lakeshore worker can't reach relay | Worker can't open WebSocket | Worker falls back to batch mode (returns complete response via Globus) |
| Relay connection drops mid-stream | Client stops receiving tokens | Partial response shown; Globus result checked for full content |
| vLLM errors out | Worker receives HTTP error from vLLM | Error sent through relay; client raises HTTPException |
| Globus submit fails | Task never reaches Lakeshore | Relay connection times out; falls back or raises error |
| Task_id mismatch | Producer and consumer in different channels | Neither receives messages; consumer times out |
| Network latency spike | Tokens delayed through relay | streaming.py's gap warning kicks in after 5 seconds |

### The Critical Fallback

The most important design decision: **if anything goes wrong with the relay, fall back to batch mode.** The batch approach works reliably today. The relay is an optimization, not a replacement. If the relay server is down, unreachable, or misbehaving, STREAM should still work exactly as it does now — just without the streaming improvement.

This is implemented at two levels:

1. **Client side:** If `websockets.connect()` raises `OSError` (relay unreachable), yield from the existing `_forward_lakeshore()` instead.

2. **Worker side:** If `websocket.create_connection()` fails, set `stream=False` on the vLLM call and return the complete response through Globus (same as current behavior).

---

## 11. Testing Plan

### Unit Tests

| Test | What It Verifies |
|------|-----------------|
| Relay: two clients in same channel | Messages from client A arrive at client B |
| Relay: two clients in different channels | Messages don't leak between channels |
| Relay: client disconnects | Channel cleaned up, no memory leak |
| Relay: stale channel cleanup | Channels older than MAX_CHANNEL_AGE are removed |
| Remote function: relay connected | Tokens forwarded, returns `{"status": "streamed"}` |
| Remote function: relay unreachable | Falls back to batch, returns complete response |
| Consumer: receives tokens | Yields SSE lines in correct format |
| Consumer: relay unreachable | Falls back to `_forward_lakeshore()` batch mode |
| Consumer: relay drops mid-stream | Partial response handled gracefully |

### Integration Test

1. Start the relay server locally (`python relay_server.py`)
2. Run STREAM with `RELAY_URL=ws://127.0.0.1:8765`
3. Send a Lakeshore query through the UI
4. Verify tokens appear progressively (not all at once)
5. Check logs for: "Streaming completed: N tokens"

### Comparison Test

1. Send the same query with `RELAY_URL=` (batch mode) — record time-to-first-token
2. Send the same query with `RELAY_URL=ws://...` (streaming mode) — record time-to-first-token
3. Compare: streaming should show first token ~2 seconds earlier

---

## 12. File Change Summary

| File | Type | Lines Changed (Estimated) | Description |
|------|------|--------------------------|-------------|
| `relay_server.py` | **New** | ~140 | Standalone relay server |
| `stream/middleware/core/globus_compute_client.py` | Modified | ~80 added | Add `_REMOTE_FN_STREAMING_SOURCE` and `remote_vllm_inference_streaming` |
| `stream/middleware/core/litellm_direct.py` | Modified | ~100 added | Add `_forward_lakeshore_streaming()`, modify `_forward_lakeshore()` to check RELAY_URL |
| `stream/proxy/app.py` | Modified | ~80 added | Add streaming route for server/Docker mode |
| `stream/middleware/config.py` | Modified | ~3 added | Add `RELAY_URL` config |
| `stream/desktop/config.py` | Modified | ~2 added | Add `RELAY_URL` setdefault |
| `.env` | Modified | ~2 added | Add `RELAY_URL` variable |

**Total new code:** ~400 lines (including the relay server)
**Total modified code:** ~10 lines changed in existing functions (adding RELAY_URL checks)

The vast majority of existing code is untouched. The change is additive — it adds a new streaming path alongside the existing batch path, with the batch path serving as an automatic fallback.
