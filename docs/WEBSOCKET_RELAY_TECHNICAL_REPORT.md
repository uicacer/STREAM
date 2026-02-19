# WebSocket Relay for Real-Time Token Streaming from HPC

## A Technical Report on Enabling True Token Streaming Through Globus Compute

---

## Table of Contents

1. [The Problem: Why Globus Compute Cannot Stream Tokens](#1-the-problem)
2. [Alternative Approaches Considered](#2-alternatives-considered)
3. [Our Solution: The WebSocket Relay Pattern](#3-our-solution)
4. [Architecture: Control Plane vs Data Plane](#4-architecture)
5. [How WebSockets Work (A Primer)](#5-websocket-primer)
6. [The Relay Server](#6-relay-server)
7. [The Producer: Remote Function on Lakeshore](#7-the-producer)
8. [The Consumer: Middleware Receiving Tokens](#8-the-consumer)
9. [Message Protocol](#9-message-protocol)
10. [The Complete Data Flow (Step by Step)](#10-complete-data-flow)
11. [Fallback Mechanism: Relay → Batch Mode](#11-fallback-mechanism)
12. [Security Considerations](#12-security)
13. [Network Topology and Firewall Traversal](#13-network-topology)
14. [Performance Characteristics](#14-performance)
15. [Testing Strategy](#15-testing)
16. [How to Run and Demo](#16-how-to-run)
17. [Production Deployment](#17-production)
18. [Code File Reference](#18-code-reference)
19. [Frequently Asked Questions](#19-faq)
20. [References](#20-references)

---

## 1. The Problem: Why Globus Compute Cannot Stream Tokens {#1-the-problem}

### What Globus Compute does

Globus Compute (formerly funcX) is a federated function-as-a-service platform that enables remote execution of Python functions on HPC resources [1]. It provides:

- **Authentication** via Globus Auth (OAuth2 federated identity)
- **Job submission** via AMQP message queues (serialized function + arguments)
- **Result retrieval** via polling or futures (the complete return value)

The execution model is fundamentally **batch-oriented**:

```
Submit function(args) → Wait → Get complete result
```

### What we need for a ChatGPT-like experience

Large Language Models (LLMs) generate text one token at a time. Modern chat interfaces display tokens as they are generated, providing a "typing" effect that:

1. **Reduces perceived latency** — the user sees output immediately instead of waiting 5-30 seconds for the full response
2. **Enables early cancellation** — if the model is going off-track, the user can stop it
3. **Provides feedback** — the user knows the system is working, not frozen

This requires **streaming**: delivering tokens from the GPU to the browser as they are generated.

### Why Globus Compute cannot stream natively

Globus Compute's architecture makes streaming impossible for three fundamental reasons:

1. **The return value IS the result.** Globus Compute serializes the function's return value and delivers it as a single payload via AMQP. There is no mechanism to deliver partial results while the function is still running.

2. **The communication channel is AMQP (message queue), not a persistent connection.** The SDK polls for results or waits on a Future. There is no bidirectional stream between the caller and the remote function.

3. **The function runs in a sandboxed worker process** on the HPC compute node. It has no direct network path back to the caller — all communication goes through Globus's cloud infrastructure (serialized via AMQP to `compute.amqps.globus.org`).

We verified this by searching the Globus Compute SDK source code, API documentation, and GitHub issues. As of 2026, there is no streaming API, no `yield`-based result delivery, and no planned support for streaming in the roadmap [2].

### What STREAM did before the relay (batch mode with fake streaming)

Before implementing the relay, STREAM used a workaround:

1. Submit the vLLM inference to Globus Compute with `stream=False`
2. Wait for the complete response (5-30 seconds, depending on response length)
3. Split the response into word groups and yield them with artificial delays

```python
# Fake streaming (simplified)
words = complete_response.split(" ")
for i in range(0, len(words), 2):
    yield words[i:i+2]
    await asyncio.sleep(0.05)  # 50ms delay between chunks
```

This created a "typing" illusion, but the user still waited the full inference time before seeing any output. The experience was:

```
User sends query → [10 second wait, blank screen] → Text appears word-by-word (fake)
```

With real streaming, the experience becomes:

```
User sends query → [2-3 second Globus routing] → Tokens appear as GPU generates them (real)
```

---

## 2. Alternative Approaches Considered {#2-alternatives-considered}

Before settling on the WebSocket relay, we evaluated several alternatives:

### Alternative 1: SSH Port Forwarding (Direct vLLM Access)

**Idea:** Create an SSH tunnel from the user's machine directly to the vLLM server on the HPC compute node, bypassing Globus entirely for inference.

```
User's machine → SSH tunnel → Lakeshore compute node → vLLM (port 8000)
```

**Why we rejected it:**
- Requires the user to have SSH access to Lakeshore — not all STREAM users have HPC accounts
- SSH tunnels are fragile and user-hostile (users must manage tunnels manually)
- Bypasses Globus authentication entirely, creating security concerns
- The compute node IP changes when jobs are rescheduled (it could be `ga-001` one day and `ga-003` the next)
- Does not work for the Docker/server deployment mode

### Alternative 2: HTTP Long Polling

**Idea:** The remote function writes tokens to a shared file or database on Lakeshore. The middleware polls this file/database repeatedly to retrieve new tokens.

```
Lakeshore: vLLM → write tokens to /tmp/tokens_{job_id}.jsonl
Middleware: while True: poll for new lines in that file (via Globus Transfer or HTTP)
```

**Why we rejected it:**
- Lakeshore compute nodes have no inbound HTTP server (no way to poll from outside)
- Globus Transfer is designed for bulk file transfers, not real-time streaming
- Polling introduces latency (the interval between polls becomes the minimum token delivery time)
- Writing to shared filesystems (NFS) from GPU nodes adds I/O overhead
- Requires cleanup logic for temporary files

### Alternative 3: gRPC Streaming Through Globus

**Idea:** Use gRPC bidirectional streaming to send tokens through Globus's infrastructure.

**Why we rejected it:**
- Globus Compute does not expose a gRPC interface
- The Globus AMQP transport layer only supports serialized function calls and return values
- Would require modifications to Globus Compute itself (not feasible for an application-level project)

### Alternative 4: Redis Pub/Sub

**Idea:** Run a Redis instance accessible to both Lakeshore and the middleware. The remote function publishes tokens to a Redis channel; the middleware subscribes.

```
Lakeshore: vLLM → Redis PUBLISH → channel:{job_id}
Middleware: Redis SUBSCRIBE → channel:{job_id} → SSE to browser
```

**Why we rejected it:**
- Requires a Redis server accessible from both the HPC compute node and the middleware
- Lakeshore compute nodes are behind the UIC campus firewall — outbound access to arbitrary Redis servers is not guaranteed
- Adds an infrastructure dependency (Redis must be maintained, monitored, backed up)
- Redis Pub/Sub is fire-and-forget — messages published before the subscriber connects are lost (no buffering)

### Alternative 5: WebSocket Relay (Selected)

**Idea:** Run a lightweight WebSocket server on a publicly accessible host. Both the HPC compute node (producer) and the middleware (consumer) make outbound WebSocket connections to this relay. The relay simply forwards messages from producer to consumer.

**Why this works:**
- **Both sides connect outbound** — no inbound connections needed on either side, so firewalls are not an obstacle
- **HPC compute nodes can make outbound HTTPS/WSS connections** — this is how Globus Compute itself works (AMQP over TLS to `compute.amqps.globus.org` on port 443)
- **The relay is stateless and lightweight** — it just forwards bytes, using ~10 MB RAM
- **WebSocket is a standard protocol** — well-supported libraries exist for both synchronous (Lakeshore) and asynchronous (middleware) Python
- **Built-in buffering** — if the producer sends tokens before the consumer connects, the relay buffers them
- **No infrastructure dependencies** — a single Python process, no database, no message broker

We verified that Lakeshore compute nodes can make outbound WebSocket connections by running a diagnostic test through Globus Compute (see `tests/test_compute_node_connectivity.py`). The test confirmed:
- Outbound TCP on ports 80 and 443: **working**
- DNS resolution: **working**
- HTTPS requests: **working**
- The `websockets` library: **installed** (or installable via `pip install websockets`)

---

## 3. Our Solution: The WebSocket Relay Pattern {#3-our-solution}

The WebSocket relay pattern separates job submission from data delivery:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE                                │
│                     (Globus Compute — existing)                      │
│                                                                      │
│  User's App  ──AMQP──→  Globus Cloud  ──AMQP──→  Lakeshore HPC     │
│              (submit job)              (deliver job)                  │
│                                                                      │
│  What it does:                                                       │
│  • Authentication (OAuth2 tokens)                                    │
│  • Job serialization (function + arguments)                          │
│  • Job routing (to the correct HPC endpoint)                         │
│  • Job execution (launches the function on a compute worker)         │
│  • Status reporting (success/failure)                                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                          DATA PLANE                                  │
│                   (WebSocket Relay — new)                             │
│                                                                      │
│  Lakeshore HPC  ──WSS──→  Relay Server  ──WSS──→  User's App        │
│   (PRODUCER)           (message forwarder)         (CONSUMER)        │
│                                                                      │
│  What it does:                                                       │
│  • Real-time token delivery (as GPU generates them)                  │
│  • Lightweight message forwarding (no computation)                   │
│  • Firewall traversal (both sides connect outbound)                  │
│  • Message buffering (handles timing mismatches)                     │
└──────────────────────────────────────────────────────────────────────┘
```

The control plane handles the heavy lifting of authentication, serialization, and job management. The data plane handles the lightweight but latency-sensitive task of delivering tokens in real-time. This separation means we leverage Globus Compute's security and job management without being limited by its batch-only data delivery.

This pattern is analogous to how video conferencing works:
- **Signaling** (who's calling whom, codec negotiation) goes through a signaling server
- **Media data** (video/audio) goes through TURN/STUN servers or peer-to-peer

In our case:
- **Signaling** (job submission, authentication) goes through Globus Compute
- **Token data** goes through the WebSocket relay

---

## 4. Architecture: Control Plane vs Data Plane {#4-architecture}

### Before the relay (batch mode)

```
User sends "How do I submit a GPU job?"

User's Browser → Middleware → Globus Compute (AMQP) → Lakeshore
                                                       │
                                                  vLLM generates
                                                  full response
                                                  (5-15 seconds)
                                                       │
User's Browser ← Middleware ← Globus Compute (AMQP) ←─┘
                                    │
                              [wait for ALL tokens]
                              [then deliver at once]
```

**Time to first visible token**: 5-15 seconds (full inference time + Globus round-trip)

### After the relay (streaming mode)

```
User sends "How do I submit a GPU job?"

CONTROL PLANE (job submission):
User's Browser → Middleware → Globus Compute (AMQP) → Lakeshore
                                                       │
                                                  vLLM starts generating
                                                  with stream=True
                                                       │
DATA PLANE (token delivery):                           │
                                                       ↓
                                              token₁ ("How") ──WSS──→ Relay ──WSS──→ Middleware → Browser
                                              token₂ (" to")  ──WSS──→ Relay ──WSS──→ Middleware → Browser
                                              token₃ (" submit") ──→ Relay ──→ Middleware → Browser
                                              ...
                                              [DONE]   ──WSS──→ Relay ──WSS──→ Middleware → Browser
```

**Time to first visible token**: 2-4 seconds (Globus routing + vLLM first-token latency)

The user sees output appear in real-time while the GPU is still generating subsequent tokens.

---

## 5. How WebSockets Work (A Primer) {#5-websocket-primer}

### HTTP vs WebSocket

**HTTP** (HyperText Transfer Protocol) is request-response: the client sends a request, the server sends one response, and the connection is done. Every new interaction requires a new request. HTTP is like sending letters — you send one, wait for a reply, send another.

**WebSocket** is a persistent, bidirectional communication channel. After an initial HTTP handshake (the "upgrade" request), the connection stays open. Both sides can send messages at any time without requesting or responding. WebSocket is like a phone call — once connected, both sides can talk freely.

### The WebSocket handshake

```
Client → Server:
  GET /produce/abc123 HTTP/1.1
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==

Server → Client:
  HTTP/1.1 101 Switching Protocols
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

[Connection upgraded — now both sides send WebSocket frames]
```

After the 101 response, the HTTP connection is "upgraded" to a WebSocket connection. From this point on, both sides communicate using WebSocket frames (small binary headers + payload). The TCP connection remains open until explicitly closed.

### Why WebSocket for our relay

1. **Low latency**: No HTTP overhead per message. Each token is delivered in a single WebSocket frame (~10 bytes overhead vs ~200+ bytes for an HTTP request).
2. **Server push**: The relay can push tokens to the consumer as they arrive — no polling needed.
3. **Standard protocol**: Supported by all browsers, proxies, load balancers, and tunnel services (ngrok, cloudflared, localhost.run).
4. **TLS support**: `wss://` provides encryption (WebSocket over TLS), same as `https://`.
5. **Library support**: Python has excellent WebSocket libraries — `websockets` for async, `websocket-client` for sync.

---

## 6. The Relay Server {#6-relay-server}

**Source file:** `stream/relay/server.py` (~435 lines)

### What it does

The relay server is a standalone WebSocket server that:
1. Accepts connections from **producers** (Lakeshore compute nodes) and **consumers** (STREAM middleware)
2. Matches them by **channel ID** (a UUID generated per chat request)
3. **Forwards messages** from producer to consumer in real-time
4. **Buffers messages** if the producer sends tokens before the consumer connects
5. **Cleans up** channels when both sides disconnect

### How channels work

A "channel" is a logical pairing of one producer and one consumer, identified by a UUID:

```python
channels = {
    "4427025d-...": {
        "producer": <WebSocket connection from Lakeshore>,
        "consumer": <WebSocket connection from middleware>,
        "buffer": [],  # Messages queued before consumer connected
    }
}
```

### URL-based routing

The relay uses URL paths to determine roles:

| Path | Role | Who connects |
|------|------|-------------|
| `/produce/{channel_id}` | Producer | Lakeshore remote function |
| `/consume/{channel_id}` | Consumer | STREAM middleware |
| `/health` | Health check | Monitoring tools |

### Message buffering

In practice, the consumer (middleware) connects to the relay immediately after submitting the Globus job. The producer (Lakeshore) connects a few seconds later, after Globus routes the job to the compute node. So the consumer is typically waiting when the first token arrives.

However, in edge cases (network hiccup, slow middleware), the producer might connect first. The relay handles this by buffering:

```
Timeline:
  0s: Consumer submits Globus job + connects to relay as consumer
  2s: Globus delivers job to Lakeshore
  3s: Producer connects to relay
  3s: Producer sends token₁ → relay forwards to consumer (consumer is ready)
  3s: Producer sends token₂ → relay forwards to consumer
  ...

Edge case (consumer connects late):
  0s: Middleware submits Globus job
  2s: Globus delivers job to Lakeshore
  3s: Producer connects to relay
  3s: Producer sends token₁ → relay BUFFERS (no consumer yet)
  3s: Producer sends token₂ → relay BUFFERS
  4s: Consumer connects → relay FLUSHES buffer (token₁, token₂)
  4s: Producer sends token₃ → relay forwards directly
```

### Channel cleanup

When both the producer and consumer disconnect, the channel is removed from the registry to prevent memory leaks. However, if there are buffered messages (producer disconnected but consumer hasn't connected yet), the channel is kept alive so the consumer can receive those messages:

```python
def _maybe_cleanup_channel(channel_id):
    channel = channels.get(channel_id)
    if channel and channel["producer"] is None and channel["consumer"] is None:
        if channel["buffer"]:
            # Keep alive — consumer needs these messages
            return
        del channels[channel_id]
```

### Resource usage

The relay is extremely lightweight:
- **Memory**: ~10 MB base + ~1 KB per active channel
- **CPU**: Near-zero (just forwarding bytes between sockets)
- **Bandwidth**: Proportional to token generation rate (~100 bytes/token × ~50 tokens/second = ~5 KB/s per active stream)
- **Connections**: Two WebSocket connections per active stream (one producer, one consumer)

A single relay instance can comfortably handle hundreds of concurrent streams.

---

## 7. The Producer: Remote Function on Lakeshore {#7-the-producer}

**Source file:** `stream/middleware/core/globus_compute_client.py` (the `remote_vllm_streaming` function, lines 158-277)

### What it does

The producer is a Python function that:
1. Runs on a Lakeshore HPC compute node (launched by Globus Compute)
2. Connects to the relay server as a producer via WebSocket
3. Makes a streaming request to the local vLLM server (`stream=True`)
4. Reads tokens from vLLM's SSE response one by one
5. Forwards each token through the WebSocket to the relay
6. Sends a "done" message when generation is complete

### Why it's defined as a string (exec/compile pattern)

The remote function is defined as a **string** and compiled at import time:

```python
_REMOTE_STREAMING_FN_SOURCE = """
def remote_vllm_streaming(vllm_url, model, messages, ...):
    import json
    import requests
    from websockets.sync.client import connect as ws_connect
    ...
"""

_ns2 = {}
exec(compile(_REMOTE_STREAMING_FN_SOURCE, "<remote_vllm_streaming>", "exec"), _ns2)
remote_vllm_streaming = _ns2["remote_vllm_streaming"]
```

This is necessary because Globus Compute serializes the function's bytecode and sends it to the HPC worker. When STREAM is packaged with PyInstaller (desktop mode), the bytecode format can differ between the local Python version and the HPC worker's Python version. Using `exec(compile(...))` generates clean bytecode that avoids serialization mismatches. The Globus SDK's `AllCodeStrategies` serializer handles this pattern correctly [3].

### Synchronous WebSocket on Lakeshore

The remote function uses `websockets.sync.client` (synchronous WebSocket) rather than the async version. This is because:

1. Globus Compute workers run functions synchronously (no event loop)
2. The function needs to read from vLLM (HTTP streaming) and write to the relay (WebSocket) in a simple sequential loop
3. Synchronous code is easier to debug on HPC systems where logging is limited

### The vLLM streaming protocol

vLLM implements the OpenAI-compatible SSE streaming protocol:

```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}

data: [DONE]
```

The remote function parses each `data:` line, extracts the `delta.content` field, and forwards it as a relay message:

```python
for line in response.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data: "):
        continue
    payload = line[6:]
    if payload.strip() == "[DONE]":
        break
    chunk = json.loads(payload)
    content = chunk["choices"][0].get("delta", {}).get("content")
    if content:
        ws.send(json.dumps({"type": "token", "content": content}))
```

---

## 8. The Consumer: Middleware Receiving Tokens {#8-the-consumer}

**Source file:** `stream/middleware/core/litellm_direct.py` (the `_forward_lakeshore_streaming` function, lines 352-458)

### What it does

The consumer is an async generator function in the STREAM middleware that:
1. Connects to the relay server as a consumer via async WebSocket
2. Receives token messages in real-time
3. Converts each token to SSE format (identical to what litellm produces)
4. Yields SSE lines to the streaming pipeline (`streaming.py`)
5. The streaming pipeline delivers them to the React frontend via HTTP SSE

### How it integrates with the existing streaming pipeline

STREAM's streaming pipeline is designed around SSE lines:

```
litellm_client.py → streaming.py → chat.py → React frontend
                         │
               Processes SSE lines:
               "data: {"choices":[{"delta":{"content":"Hello"}}]}"
```

The relay consumer produces **the exact same SSE format**. This means `streaming.py`, `chat.py`, and the React frontend work without any changes — they don't know or care whether the tokens came from a relay or from litellm directly:

```python
async for msg_str in ws:
    msg = json.loads(msg_str)
    if msg["type"] == "token":
        chunk = {"choices": [{"index": 0, "delta": {"content": msg["content"]}}]}
        yield f"data: {json.dumps(chunk)}"
    elif msg["type"] == "done":
        yield "data: [DONE]"
        break
```

### Relay pre-check and fallback

Before submitting an expensive Globus Compute job, the middleware checks if the relay is reachable by connecting to its `/health` WebSocket endpoint. This prevents wasting HPC resources when the relay is down (tunnel expired, server stopped):

```python
if RELAY_URL:
    try:
        async for line in _forward_lakeshore_streaming(...):
            # Inside _forward_lakeshore_streaming:
            # 1. Check relay /health endpoint (fast WebSocket ping)
            # 2. Only if reachable: submit Globus job + connect as consumer
            yield line
        return
    except Exception as e:
        logger.warning(f"Falling back to BATCH MODE ({reason}).")
        # Fall through to batch mode

# Batch mode: wait for complete response via Globus (no relay needed)
result = await gc.submit_inference(...)
```

This ensures the user always gets a response, even if the relay is unavailable — and no HPC jobs are wasted on relay failures.

---

## 9. Message Protocol {#9-message-protocol}

All messages sent through the relay are JSON strings with a `type` field:

### Producer → Relay → Consumer

| Message Type | Format | Description |
|---|---|---|
| Token | `{"type": "token", "content": "Hello"}` | A chunk of generated text (one or more tokens from vLLM) |
| Done | `{"type": "done", "usage": {...}}` | Generation complete. Usage contains token counts. |
| Error | `{"type": "error", "message": "..."}` | Something went wrong on Lakeshore |

### Example complete stream

```json
{"type": "token", "content": "Hello"}
{"type": "token", "content": "!"}
{"type": "token", "content": " How"}
{"type": "token", "content": " can"}
{"type": "token", "content": " I"}
{"type": "token", "content": " assist"}
{"type": "token", "content": " you"}
{"type": "token", "content": " today"}
{"type": "token", "content": "?"}
{"type": "done", "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}}
```

The relay does not interpret these messages — it forwards them verbatim. This makes the relay protocol-agnostic: if we change the message format in the future, the relay code doesn't need to change.

---

## 10. The Complete Data Flow (Step by Step) {#10-complete-data-flow}

Here is every step that happens from the moment the user presses Enter to the moment they see the first token:

### Step 1: User sends a message (Browser → Middleware)
The React frontend sends an HTTP POST to `/v1/chat/completions` with the conversation history.

### Step 2: Middleware routes to Lakeshore (query_router.py)
The complexity judge classifies the query. A "medium" query is routed to Lakeshore. The router performs a Level 1 health check only (is Globus authenticated? — takes ~100ms, no GPU job).

### Step 3: Middleware checks relay health (litellm_direct.py)
Before submitting any Globus job, the middleware performs a quick WebSocket health check against the relay's `/health` endpoint. This takes ~50-200ms. If the relay is unreachable (tunnel expired, server stopped), the middleware skips streaming entirely and goes straight to batch mode — no expensive HPC job is wasted.

### Step 4: Middleware generates a channel ID and submits the streaming job (litellm_direct.py → globus_compute_client.py)
A UUID is generated (e.g., `4427025d-8f3a-4b2c-9d1e-...`) as the channel ID. The `remote_vllm_streaming` function is serialized along with its arguments (vLLM URL, model name, messages, temperature, max_tokens, relay URL, channel ID) and sent to Globus via AMQP. This takes ~100-500ms.

### Step 5: Middleware connects to relay as consumer (litellm_direct.py)
Immediately after submitting the job, the middleware opens an async WebSocket connection to `wss://relay-url/consume/{channel_id}`. This happens in parallel with Globus routing the job. The relay creates the channel and registers the consumer.

### Step 6: Globus routes the job to Lakeshore (~1-3 seconds)
Globus Compute delivers the serialized function to the Lakeshore endpoint worker via AMQP. The worker deserializes the function and begins executing it.

### Step 7: Remote function connects to relay as producer
The `remote_vllm_streaming` function, now running on a Lakeshore compute node (e.g., `ga-001`), opens a synchronous WebSocket connection to `wss://relay-url/produce/{channel_id}`. The relay registers the producer and pairs it with the consumer.

### Step 8: Remote function calls vLLM with `stream=True`
An HTTP POST is made to the local vLLM server (e.g., `http://ga-001:8000/v1/chat/completions`) with `stream: true`. vLLM begins generating tokens on the GPU.

### Step 9: Tokens flow through the relay (real-time)
As vLLM generates each token, it sends an SSE event. The remote function parses it, extracts the text, and sends it through the WebSocket to the relay. The relay forwards it to the consumer. The consumer converts it to SSE format and yields it to the streaming pipeline.

### Step 10: Tokens appear in the browser (real-time)
The streaming pipeline (`streaming.py`) forwards each SSE line to the HTTP response. The React frontend's `EventSource`-like reader parses each line and appends the token to the chat message. The user sees text appearing word by word.

### Step 11: Stream completes
vLLM sends `[DONE]`. The remote function sends `{"type": "done", "usage": {...}}` through the relay. The consumer yields `data: [DONE]`. The streaming pipeline sends the final cost metadata and closes the SSE stream. Both sides disconnect from the relay, and the channel is cleaned up.

### Timing breakdown (typical)

| Step | Duration | Cumulative |
|------|----------|------------|
| Query routing + complexity judge | ~800ms | 0.8s |
| Relay health check | ~50-200ms | ~1.0s |
| Globus job submission (AMQP) | ~500ms | 1.5s |
| Globus routing to Lakeshore | ~1-3s | 2.5-4.5s |
| vLLM first token generation | ~200-500ms | 2.7-5.0s |
| **User sees first token** | | **~3-5s** |
| Subsequent tokens | ~20-50ms each | continuous |
| Full response (500 tokens) | ~10s total | ~13-15s |

Compare with batch mode: the user would see nothing for 10-15 seconds, then the entire response at once.

---

## 11. Fallback Mechanism: Relay → Batch Mode {#11-fallback-mechanism}

The relay is an enhancement, not a requirement. If the relay is unavailable, STREAM falls back gracefully:

### Layer 1: Relay pre-check → Batch (before Globus submission)

In `_forward_lakeshore_streaming()` (`litellm_direct.py`), the first thing the streaming path does is check if the relay is reachable by connecting to its `/health` WebSocket endpoint. If the relay is down, a `ConnectionError` is raised *before* any Globus job is submitted, and the caller falls back to batch mode immediately:

```python
# Inside _forward_lakeshore_streaming():
if not await _check_relay_reachable(RELAY_URL):
    raise ConnectionError("Relay not reachable")

# Only now submit the expensive Globus Compute job
result = await gc.submit_streaming_inference(...)

# Caller in _forward_lakeshore():
if RELAY_URL:
    try:
        async for line in _forward_lakeshore_streaming(...):
            yield line
        return
    except Exception:
        # Relay unreachable — fall through to batch mode (no HPC job wasted)
        pass

result = await gc.submit_inference(...)  # Batch mode
```

This is a critical optimization: without the pre-check, a relay failure would waste a Globus Compute job (which consumes HPC allocation time) before discovering the relay is down.

### Layer 2: Lakeshore → Other Tier (across tiers)

In `streaming.py`, if the entire Lakeshore inference fails (both relay and batch), the runtime fallback kicks in:

1. `mark_tier_unavailable("lakeshore", error)` — turns the health dot red
2. `get_fallback_tier(complexity, tiers_tried)` — picks the next tier (e.g., Cloud)
3. Sends a fallback SSE event to the frontend
4. Retries inference on the fallback tier

### Layer 3: Frontend health dot update

When the frontend receives a fallback SSE event, it calls `markTierFailed(tier)` on the health store, turning the Lakeshore dot red immediately without waiting for the next health poll.

---

## 12. Security Considerations {#12-security}

### Channel ID as access token

The channel ID is a UUID v4 — 122 bits of randomness. Only someone who knows the exact channel ID can connect to a channel as producer or consumer. This provides equivalent security to a bearer token for the ~10-30 second lifetime of a streaming session.

### What the relay can see

The relay forwards raw messages without inspection. In transit, the relay can see the token content (the generated text). For sensitive deployments:

1. **Use TLS (wss://)** to encrypt traffic between all parties and the relay
2. **Deploy the relay on trusted infrastructure** (e.g., within UIC's network)
3. **End-to-end encryption** could be added by encrypting messages with a shared key exchanged via the Globus control plane (not yet implemented)

### What the relay cannot do

- Cannot execute code on Lakeshore or the user's machine
- Cannot access files, databases, or credentials
- Cannot modify messages (the consumer would detect JSON corruption)
- Stores no data — channels are in-memory and cleaned up immediately after use

### One producer, one consumer per channel

The relay enforces that each channel has at most one producer and one consumer. If a second connection attempts to claim an already-occupied role, it receives error code 4001 and is disconnected. This prevents channel hijacking.

---

## 13. Network Topology and Firewall Traversal {#13-network-topology}

### The firewall problem

```
┌───────────────────────────┐     ┌─────────────────────────────────┐
│  User's Machine           │     │  Lakeshore HPC (UIC campus)     │
│                           │     │                                  │
│  ┌─────────────────────┐  │     │  ┌──────────────────────────┐   │
│  │ STREAM Middleware    │  │     │  │ Compute Node (ga-001)    │   │
│  │                      │  │     │  │                          │   │
│  │ Can make OUTBOUND    │  │     │  │ Can make OUTBOUND        │   │
│  │ connections ✓        │  │     │  │ connections ✓            │   │
│  │                      │  │     │  │                          │   │
│  │ Cannot accept        │  │     │  │ Cannot accept            │   │
│  │ INBOUND connections ✗│  │     │  │ INBOUND connections ✗    │   │
│  └─────────────────────┘  │     │  └──────────────────────────┘   │
│                           │     │                                  │
│  NAT / Home Router        │     │  Campus Firewall                 │
└───────────────────────────┘     └─────────────────────────────────┘
```

Neither the user's machine nor the Lakeshore compute node can accept inbound connections. This rules out any solution where one side acts as a server.

### How the relay solves it

```
┌─────────────────┐              ┌─────────────────┐
│  User's Machine │              │  Lakeshore HPC  │
│  (CONSUMER)     │              │  (PRODUCER)     │
│                 │              │                 │
│  OUTBOUND ──────┼──WSS──┐     │  OUTBOUND ──────┼──WSS──┐
└─────────────────┘       │     └─────────────────┘       │
                          ↓                               ↓
                   ┌──────────────────────────────┐
                   │  Relay Server                 │
                   │  (public IP / ngrok tunnel)   │
                   │                               │
                   │  Accepts INBOUND connections  │
                   │  from both sides ✓            │
                   └──────────────────────────────┘
```

Both sides make **outbound** connections to the relay. The relay is the only component that needs to accept inbound connections, so it runs on a publicly accessible server (or behind an ngrok/localhost.run tunnel during development).

This is the same principle used by TURN servers in WebRTC video calls, MQTT brokers in IoT systems, and Globus Compute itself (both the user's SDK and the HPC endpoint connect outbound to `compute.amqps.globus.org`).

---

## 14. Performance Characteristics {#14-performance}

### Latency added by the relay

The relay adds one network hop in each direction:

```
Without relay:  Lakeshore → [Globus AMQP cloud] → User's machine
With relay:     Lakeshore → [Relay server] → User's machine
```

The relay hop adds approximately **1-5ms** per token (measured as the time from the relay receiving a message to forwarding it). This is negligible compared to the ~20-50ms between tokens from vLLM.

### Total token delivery latency

| Component | Latency |
|---|---|
| vLLM token generation | ~20-50ms per token |
| WebSocket send (Lakeshore → relay) | ~5-15ms (depends on relay location) |
| Relay forwarding | ~1-5ms |
| WebSocket send (relay → middleware) | ~5-15ms |
| SSE delivery (middleware → browser) | ~1ms (localhost) |
| **Total per-token latency** | **~30-85ms** |

This is well within the threshold for a smooth "typing" experience (humans perceive text appearing smoothly at up to ~200ms intervals).

### First token latency comparison

| Mode | Time to first visible token |
|---|---|
| **Relay streaming** | ~3-5s (Globus routing + vLLM first token) |
| **Batch mode** | ~8-15s (full inference time + Globus round-trip) |
| **Cloud tier (direct)** | ~1-2s (direct API call, no Globus) |

---

## 15. Testing Strategy {#15-testing}

We employ a three-layer testing pyramid:

### Layer 1: Unit Tests — `tests/test_relay_local.py`

**What it tests:** The relay server in isolation, with no external dependencies.

**How it works:** Starts a real relay server on a random port (OS-assigned, port 0) and connects test producers and consumers via `websockets.asyncio.client`.

**Test cases:**

| Test | What it verifies |
|---|---|
| Basic message flow | Producer sends 3 tokens + done → consumer receives all 4 in order |
| Message buffering | Producer sends tokens BEFORE consumer connects → messages are buffered and delivered when consumer connects |
| Health endpoint | `/health` returns status, active channel count, and timestamp |
| Multiple channels | Two independent channels run simultaneously with no cross-talk |
| Channel cleanup | Channels are removed from the registry after both sides disconnect |

**Run command:**
```bash
python -m pytest tests/test_relay_local.py -v
```

### Layer 2: Integration Tests — `tests/test_relay_integration.py`

**What it tests:** The complete data flow from vLLM response → remote function → relay → consumer → SSE output, using a **fake vLLM server** (no GPU required).

**How it works:**
1. Starts a local HTTP server that simulates vLLM's streaming SSE endpoint (produces 9 predefined tokens: "Hello from Lakeshore! How can I help?")
2. Starts a local relay server
3. Runs the **actual `remote_vllm_streaming` function** (the same code that runs on Lakeshore) against the fake vLLM server
4. Connects a consumer to the relay
5. Verifies that the SSE output matches what the frontend expects

**What it catches:**
- SSE format correctness (the frontend's parser is strict)
- Token ordering (tokens must arrive in generation order)
- Usage statistics propagation (prompt_tokens, completion_tokens)
- The `[DONE]` marker is present

**Run command:**
```bash
python -m pytest tests/test_relay_integration.py -v
```

### Layer 3: End-to-End Test — `tests/test_e2e_globus_streaming.py`

**What it tests:** The complete production path: your machine → Globus Compute → Lakeshore HPC → vLLM GPU → relay → consumer.

**Prerequisites:**
- Relay server running (`python -m stream.relay.server`)
- Tunnel active (e.g., `ssh -4 -R 80:localhost:8765 nokey@localhost.run`)
- `RELAY_URL` set in `.env` to the tunnel URL
- Globus Compute authenticated (`globus-compute-endpoint configure`)
- vLLM running on Lakeshore

**What it does:**
1. Submits a real streaming inference job via `globus_client.submit_streaming_inference()`
2. Connects to the relay as a consumer
3. Receives tokens in real-time, printing them as they arrive
4. Reports timing: submit-to-first-token, first-token-to-done, total duration
5. Verifies tokens were received and usage stats are present

**Run command:**
```bash
python -m pytest tests/test_e2e_globus_streaming.py -v -s
```

The `-s` flag disables output capture so you can see tokens arriving in real-time during the test.

### Diagnostic Test — `tests/test_compute_node_connectivity.py`

**What it tests:** Whether the Lakeshore compute node environment supports the WebSocket relay approach.

**What it checks:**
- Outbound TCP connectivity (ports 80, 443)
- DNS resolution
- HTTPS request capability
- Whether the `websockets` library is installed
- Whether vLLM is reachable from the compute node

This test was used during development to verify feasibility before writing the relay implementation.

---

## 16. How to Run and Demo {#16-how-to-run}

### Development Setup (3 terminals)

**Terminal 1: Start the relay server**
```bash
cd /path/to/STREAM
python -m stream.relay.server
```

You should see:
```
Starting WebSocket relay on ws://0.0.0.0:8765
  Producer URL: ws://<host>:8765/produce/{channel_id}
  Consumer URL: ws://<host>:8765/consume/{channel_id}
```

**Terminal 2: Start a public tunnel**

The relay needs to be reachable from Lakeshore. Use one of:

```bash
# Option A: localhost.run (free, no signup, force IPv4)
ssh -4 -R 80:localhost:8765 nokey@localhost.run

# Option B: ngrok (free tier, requires signup)
ngrok http 8765
```

Copy the HTTPS URL from the tunnel output (e.g., `https://b277bb105f0a5e.lhr.life`).

Update your `.env`:
```
RELAY_URL=https://b277bb105f0a5e.lhr.life
```

> **Important:** The tunnel URL changes every time you restart the tunnel. In production, use a permanent server with a stable domain.

**Terminal 3: Start STREAM**
```bash
# Desktop dev mode
python -m stream.desktop.main --dev

# Or with separate frontend
python -m stream.desktop.main --dev &
cd frontends/react && npm run dev:vite
```

### Verifying it works

Send a message in STREAM that routes to Lakeshore (e.g., "How do I submit a GPU job?"). Watch the backend logs for:

```
Lakeshore direct call: lakeshore-qwen-1.5b → ... (STREAMING via relay)
Connecting to relay as consumer (channel=4427025d, relay=wss://...)
```

If you see `(STREAMING via relay)`, the relay is being used. If you see `(batch mode)`, the relay connection failed and it fell back.

### Demoing the difference (relay vs batch)

To show the audience the difference between real-time streaming and batch mode:

**Step 1: Show relay streaming (real-time)**
1. Ensure the relay and tunnel are running
2. Send a Lakeshore query
3. The audience sees tokens appearing word-by-word as the GPU generates them

**Step 2: Show batch mode (all-at-once)**
1. Stop the relay server (Ctrl+C in Terminal 1)
2. Send another Lakeshore query
3. The audience sees a long pause (5-15 seconds), then the entire response appears at once
4. The backend logs show: `Relay streaming failed, falling back to batch mode`

**Step 3: Show automatic recovery**
1. Restart the relay server
2. Send another query
3. Streaming resumes automatically — no app restart needed

The visual contrast is dramatic and self-explanatory. The audience immediately understands why real-time streaming matters.

### Running the tests for a demo

```bash
# Quick unit tests (no infrastructure needed, ~2 seconds)
python -m pytest tests/test_relay_local.py -v

# Integration test with fake vLLM (~5 seconds)
python -m pytest tests/test_relay_integration.py -v

# Real end-to-end through Globus + GPU (~30 seconds, requires full setup)
python -m pytest tests/test_e2e_globus_streaming.py -v -s
```

---

## 17. Production Deployment {#17-production}

### Current setup (development)

```
Relay: localhost:8765 → ngrok/localhost.run tunnel → public URL
```

This works for development and demos but has limitations:
- Tunnel URL changes on every restart
- Free tunnels have bandwidth limits
- Adds an extra hop (localhost → tunnel service → internet)

### Production setup (recommended)

```
Relay: relay.stream.example.com:443 (TLS via nginx/caddy)
```

**Option 1: UIC campus VM** (ideal)
- Deploy the relay on a VM behind ACER's reverse proxy
- Data stays within UIC's network
- Lakeshore compute nodes have fast, low-latency access

**Option 2: Small cloud VM**
- DigitalOcean/Hetzner droplet ($5/month)
- Add TLS with Caddy (automatic Let's Encrypt)
- Simple, independent of campus infrastructure

**Deployment steps:**
1. Install Python 3.11+ and the `websockets` library on the server
2. Copy `stream/relay/server.py` to the server
3. Run: `python server.py --host 0.0.0.0 --port 8765`
4. Configure a reverse proxy (nginx/Caddy) with TLS termination
5. Set `RELAY_URL=wss://relay.stream.example.com` in STREAM's `.env`

**Systemd service file** (for automatic startup):
```ini
[Unit]
Description=STREAM WebSocket Relay
After=network.target

[Service]
Type=simple
User=stream
ExecStart=/usr/bin/python3 /opt/stream-relay/server.py --host 0.0.0.0 --port 8765
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 18. Code File Reference {#18-code-reference}

### Core implementation files

| File | Lines | Purpose |
|---|---|---|
| `stream/relay/server.py` | ~435 | WebSocket relay server (standalone process) |
| `stream/relay/__init__.py` | 0 | Package marker |
| `stream/middleware/core/globus_compute_client.py` | ~1000 | Contains `remote_vllm_streaming` (producer function) and `submit_streaming_inference()` method |
| `stream/middleware/core/litellm_direct.py` | ~638 | Contains `_forward_lakeshore_streaming()` (consumer) and `_forward_lakeshore()` (orchestrator with fallback) |

### Configuration files

| File | Relevant section | Purpose |
|---|---|---|
| `.env` | `RELAY_URL=` | Relay URL (set per deployment) |
| `.env.example` | `RELAY_URL=` | Template with setup instructions |
| `stream/middleware/config.py` | `RELAY_URL` | Loads and auto-converts `https://` → `wss://` |

### Supporting files (modified to support relay)

| File | Change | Purpose |
|---|---|---|
| `stream/middleware/core/streaming.py` | Runtime fallback | Catches inference failures, falls back to next tier, marks tiers unavailable |
| `stream/middleware/core/tier_health.py` | `mark_tier_unavailable()` | Updates health status when inference fails |
| `stream/middleware/core/query_router.py` | Level 1 only for routing | Avoids expensive Lakeshore health checks during query routing |
| `frontends/react/src/stores/healthStore.ts` | `markTierFailed()` | Instantly updates health dot to red on fallback events |
| `frontends/react/src/components/chat/ChatContainer.tsx` | Fallback handler | Calls `markTierFailed()` when SSE reports a tier failure |

### Test files

| File | Type | Infrastructure required |
|---|---|---|
| `tests/test_relay_local.py` | Unit tests (5 tests) | None (self-contained) |
| `tests/test_relay_integration.py` | Integration test | None (uses fake vLLM) |
| `tests/test_e2e_globus_streaming.py` | End-to-end test | Relay + tunnel + Globus + Lakeshore + vLLM |
| `tests/test_compute_node_connectivity.py` | Diagnostic | Globus + Lakeshore endpoint |

---

## 19. Frequently Asked Questions {#19-faq}

### Q: Why not use Server-Sent Events (SSE) instead of WebSockets?

SSE is HTTP-based and unidirectional (server → client only). While the relay's data flow is also unidirectional, SSE has a critical limitation: it requires the server to initiate the stream. In our case, the Lakeshore compute node (the "server" side) is behind a firewall and cannot accept inbound HTTP connections. WebSocket works because both sides make outbound connections to the relay.

### Q: Why not use gRPC streaming?

gRPC would work technically, but it adds complexity: Protocol Buffer schemas, code generation, and a heavier runtime. WebSocket is simpler (JSON over a persistent connection), has universal library support in Python (both sync and async), and works through HTTP proxies and tunnel services without modification.

### Q: Can the relay handle thousands of concurrent users?

Yes. Each active stream uses two WebSocket connections and ~1 KB of memory. A single relay instance on a modest server can handle thousands of concurrent streams. For horizontal scaling, a load balancer can distribute channels across multiple relay instances (each channel is independent and stateless).

### Q: What happens if the relay server crashes mid-stream?

The consumer (middleware) loses the WebSocket connection, catches the exception, and falls back to batch mode for the current request. The user sees a slight delay but still gets their response. The next request will automatically retry the relay if it's back up.

### Q: Does the relay store any user data?

No. Messages exist in memory only during forwarding (or in the buffer until the consumer connects). When both sides disconnect, the channel and all its messages are deleted. There is no database, no disk storage, no logging of message content.

### Q: Can we use the relay for other HPC streaming use cases?

Yes. The relay is generic — it forwards any JSON messages between a producer and consumer matched by channel ID. It could be used for streaming results from any Globus Compute function, not just LLM inference. For example: real-time training metrics, simulation progress updates, or log streaming from HPC jobs.

### Q: Why does the tunnel URL change every time?

Free tunnel services (ngrok, localhost.run) assign random subdomains per session. This is a development convenience, not a production concern. In production, the relay runs on a server with a fixed domain name (e.g., `relay.stream.example.com`), so the URL never changes.

### Q: What is the `websockets` library and why do we use it?

`websockets` is a mature, well-maintained Python library for WebSocket communication [4]. We chose it because:
- It supports both async (`websockets.asyncio`) and sync (`websockets.sync`) APIs
- It's already available on Lakeshore's Globus Compute workers
- It handles the WebSocket protocol (framing, ping/pong, close handshake) correctly
- It's lightweight (~100 KB) with no external dependencies

---

## 20. References {#20-references}

[1] Chard, R., et al. "funcX: A Federated Function Serving Fabric for Science." *Proceedings of the 29th International Symposium on High-Performance Parallel and Distributed Computing (HPDC)*, 2020. https://doi.org/10.1145/3369583.3392683

[2] Globus Compute SDK Documentation and GitHub Repository. https://github.com/funcx-faas/funcX — No streaming API exists as of 2026. The SDK's `Executor.submit()` returns a `Future` that resolves to the complete return value.

[3] Globus Compute SDK Serialization — `AllCodeStrategies` serializer documentation. The `exec(compile(...))` pattern ensures clean bytecode that serializes correctly across different Python installations.

[4] Aymeric Augustin. "websockets — A library for building WebSocket servers and clients in Python." https://websockets.readthedocs.io/

[5] Fette, I. and Melnikov, A. "The WebSocket Protocol." RFC 6455, IETF, 2011. https://tools.ietf.org/html/rfc6455

[6] STREAM Project Repository. `stream/relay/server.py`, `stream/middleware/core/globus_compute_client.py`, `stream/middleware/core/litellm_direct.py`.

---

*This document was written for the PEARC conference paper on STREAM. It covers the design, implementation, and testing of the WebSocket relay system that enables real-time token streaming from HPC-hosted LLMs through the Globus Compute framework.*
