# STREAM–Lakeshore Connectivity: Technical Report

**Purpose:** This report documents how the STREAM middleware connects to UIC's Lakeshore HPC cluster for GPU-accelerated AI inference, covering both server (Docker) and desktop (PyInstaller) deployment modes. This report is intended to support the STREAM paper submission to the PEARC conference.

---

## Table of Contents

1. [Overview](#1-overview)
2. [What is Lakeshore?](#2-what-is-lakeshore)
3. [What is Globus Compute?](#3-what-is-globus-compute)
4. [The Globus Compute Connection in Detail](#4-the-globus-compute-connection-in-detail)
5. [Server Mode (Docker) Architecture](#5-server-mode-docker-architecture)
6. [Desktop Mode (PyInstaller) Architecture](#6-desktop-mode-pyinstaller-architecture)
7. [The Remote Function: How Code Executes on Lakeshore](#7-the-remote-function-how-code-executes-on-lakeshore)
8. [Authentication Flow](#8-authentication-flow)
9. [Context Window Management](#9-context-window-management)
10. [Streaming and Response Delivery](#10-streaming-and-response-delivery)
11. [Performance Analysis and Latency Breakdown](#11-performance-analysis-and-latency-breakdown)
12. [Persistent Executor Optimization](#12-persistent-executor-optimization)
13. [Error Handling and Resilience](#13-error-handling-and-resilience)
14. [Code Reference Map](#14-code-reference-map)
15. [Future Work: Hybrid Approaches to Reduce Lakeshore Latency](#15-future-work-hybrid-approaches-to-reduce-lakeshore-latency)

---

## 1. Overview

STREAM (Smart Tiered Routing Engine for AI Models) is a middleware system that routes AI queries to one of three tiers based on complexity:

| Tier | Backend | Use Case | Cost |
|------|---------|----------|------|
| **Local** | Ollama (Llama 3.2:3b on Apple Silicon) | Simple queries (greetings, definitions) | Free |
| **Lakeshore** | vLLM (Qwen 2.5-1.5B on UIC HPC GPUs) | Medium queries (explanations, comparisons) | Free (university GPU) |
| **Cloud** | Claude Sonnet 4 / GPT-4 Turbo | Complex queries (design, analysis, research) | Pay-per-token |

The Lakeshore tier is the most architecturally interesting because it connects a user's laptop to a remote HPC cluster behind a university firewall. This connection works through **Globus Compute**, a Function-as-a-Service (FaaS) platform for research computing.

STREAM operates in two modes:
- **Server mode (Docker):** Five containerized microservices communicating over a Docker virtual network.
- **Desktop mode (PyInstaller):** A single-process native macOS/Windows app with everything embedded.

Both modes use the same Globus Compute client code to reach Lakeshore, but the request path to that client differs significantly.

---

## 2. What is Lakeshore?

Lakeshore is UIC's HPC cluster operated by Academic Computing and Communications Center (ACCC). For STREAM, it provides:

- **GPU node:** `ga-001` with NVIDIA A100 (MIG 3g.40gb partition)
- **Model server:** vLLM serving `Qwen/Qwen2.5-1.5B-Instruct`
- **vLLM configuration:** Started via SLURM job script with `--max-model-len 32768` (32K token context window)
- **Access:** Behind UIC's campus firewall — not directly reachable from the internet

The vLLM server runs as a SLURM job on Lakeshore and exposes an OpenAI-compatible REST API at `http://ga-001:8000`. This API is **only accessible from within the Lakeshore cluster network** — not from a user's laptop, Docker container, or the public internet. This is why we need Globus Compute.

---

## 3. What is Globus Compute?

Globus Compute (formerly funcX) is a **Function-as-a-Service (FaaS)** platform for research computing, operated by the Globus team at the University of Chicago and Argonne National Laboratory. It enables remote function execution on HPC resources without requiring SSH access, VPN, or firewall modifications.

### How FaaS Works (Conceptual)

In traditional HPC access, you SSH into a login node, submit a SLURM batch job, and retrieve results later. Globus Compute replaces this with a programmatic API:

```
Traditional HPC:    SSH → login node → sbatch job.sh → wait → scp results
Globus Compute:     Python SDK → Globus cloud → endpoint on HPC → result returned
```

You write a regular Python function, submit it via the SDK, and it executes on a remote HPC endpoint. The endpoint is a persistent daemon running on the HPC cluster that receives functions, executes them, and returns results — all managed by the Globus infrastructure.

### Key Components

| Component | What It Is | Where It Runs |
|-----------|-----------|---------------|
| **Globus Compute SDK** | Python library (`globus-compute-sdk`) | User's machine (STREAM) |
| **Globus Cloud** | Central coordination service | AWS (managed by Globus) |
| **Globus Compute Endpoint** | Daemon that executes functions | Lakeshore HPC cluster |
| **AMQP Connection** | Message queue protocol for task transport | Between SDK ↔ Cloud ↔ Endpoint |

The relevant code for initializing the Globus Compute client is in [`stream/middleware/core/globus_compute_client.py`](stream/middleware/core/globus_compute_client.py).

---

## 4. The Globus Compute Connection in Detail

This section explains exactly what happens when STREAM sends a query to Lakeshore via Globus Compute, step by step.

### 4.1 The Executor: STREAM's Connection to Globus

The `Executor` class from `globus_compute_sdk` is the primary interface for submitting work to remote HPC endpoints. It works similarly to Python's `concurrent.futures.Executor`, but instead of running functions on local threads or processes, it runs them on remote machines.

When a new `Executor` is created ([`globus_compute_client.py:249`](stream/middleware/core/globus_compute_client.py#L249)):

```python
self._executor = Executor(endpoint_id=self.endpoint_id)
```

The following happens under the hood:

1. **TCP Connection:** The SDK opens a TCP socket to `compute.amqps.globus.org:443` (the Globus cloud AMQP broker, hosted on AWS at `18.205.84.179`).

2. **TLS Handshake:** The connection is encrypted using TLS (Transport Layer Security), the same encryption that HTTPS uses.

3. **AMQP Handshake:** AMQP (Advanced Message Queuing Protocol) is a messaging protocol designed for reliable message delivery. The SDK authenticates with the broker using Globus OAuth2 tokens and opens an AMQP channel.

4. **Queue Registration:** The SDK registers as a task producer for the specified endpoint ID. The broker sets up a result queue so the endpoint can send results back to this specific client.

This entire process takes approximately **0.5–1.5 seconds** on a typical network connection.

### 4.2 Task Submission: Serialization and Transport

When `gce.submit()` is called ([`globus_compute_client.py:502-510`](stream/middleware/core/globus_compute_client.py#L502-L510)):

```python
future = gce.submit(
    remote_vllm_inference,   # The function to run remotely
    self.vllm_url,           # "http://ga-001:8000"
    model,                   # "Qwen/Qwen2.5-1.5B-Instruct"
    messages,                # The chat conversation
    temperature,             # 0.7
    max_tokens,              # 2048 (from MODEL_CONTEXT_LIMITS)
    False,                   # stream=False
)
```

The following sequence occurs:

1. **Serialization:** The SDK serializes the function (`remote_vllm_inference`) and all its arguments into a byte stream using `dill` (an extended version of Python's `pickle` that can serialize functions, closures, and lambdas). The serializer is configured with `AllCodeStrategies` ([`globus_compute_client.py:254-256`](stream/middleware/core/globus_compute_client.py#L254-L256)) which tries multiple serialization strategies to handle Python version mismatches between the user's machine and the endpoint.

2. **AMQP Publish:** The serialized bytes are published to the Globus AMQP broker as a message, tagged with the target endpoint ID. This is like dropping a letter in a mailbox — the broker handles routing.

3. **Future Creation:** The SDK returns a `concurrent.futures.Future` object immediately. This is a "ticket" — the task hasn't completed yet, but you can check it later. The `submit()` call itself is fast (tens of milliseconds).

### 4.3 Task Routing: Globus Cloud to Lakeshore

Once the AMQP broker receives the serialized task:

1. **Cloud-Side Routing:** The Globus cloud service identifies the target endpoint by its UUID (`GLOBUS_COMPUTE_ENDPOINT_ID`) and routes the message to the appropriate AMQP queue for that endpoint.

2. **Endpoint Pickup:** The Globus Compute endpoint daemon running on Lakeshore (started via `globus-compute-endpoint start`) continuously polls its AMQP queue. When it sees a new task message, it:
   - Downloads the serialized bytes
   - Deserializes the function and arguments
   - Spawns a worker process (or reuses one from its pool)
   - Executes the function

3. **Remote Execution:** The deserialized `remote_vllm_inference` function runs on a Lakeshore compute node. Because the function runs **inside** the Lakeshore network, it can directly reach the vLLM server at `http://ga-001:8000`.

### 4.4 Result Return: Lakeshore Back to STREAM

After the remote function completes:

1. **Result Serialization:** The endpoint serializes the function's return value (the vLLM JSON response) using the same `dill` strategy.

2. **AMQP Return:** The serialized result is published to the client's result queue on the AMQP broker.

3. **Client Receipt:** The SDK's background thread picks up the result from the queue and stores it in the `Future` object.

4. **`future.result()` Unblocks:** The blocking call to `future.result(timeout=120)` returns the deserialized result — the vLLM response as a Python dictionary.

In STREAM, we call `future.result()` in a background thread using `asyncio.to_thread()` ([`globus_compute_client.py:543`](stream/middleware/core/globus_compute_client.py#L543)) so the main event loop stays free:

```python
result = await asyncio.to_thread(future.result, timeout=GLOBUS_TASK_TIMEOUT)
```

### 4.5 Complete Round-Trip Diagram

```
User's Machine (STREAM)                 AWS (Globus Cloud)              Lakeshore HPC
========================               ===================             ================

1. Executor.__init__()
   → TCP + TLS + AMQP handshake ----→ AMQP Broker receives
                                       client registration

2. gce.submit(fn, args)
   → dill.serialize(fn + args)
   → AMQP publish --------------------→ Broker routes to endpoint
                                         queue ----------------------→ Endpoint daemon
                                                                       picks up task

3. Future returned immediately                                         Deserializes fn
   (task still in transit)                                             Executes fn:
                                                                         → HTTP POST to
                                                                           ga-001:8000/v1
                                                                           /chat/completions
                                                                         → vLLM generates
                                                                           response

4. future.result() blocks...                                           Serializes result
                                       Broker receives ←-------------- AMQP publish result
                                       result message
   SDK picks up result ←-------------- Broker forwards to
   future.result() returns               client queue

5. Result processed by STREAM
   → Converted to SSE chunks
   → Streamed to React frontend
```

The measured latency breakdown for this round-trip (from production timing in `submit_inference()`):

| Step | First Request | Subsequent Requests |
|------|--------------|-------------------|
| Executor creation (AMQP connect) | ~0.28s | ~0.00s (reused) |
| Task submission (serialize + send) | ~0.56s | ~0.00s |
| Wait for result (round-trip) | ~4.95s | ~5.18s |
| **Total** | **~5.79s** | **~5.18s** |

The dominant cost (~5 seconds) is the Globus Compute round-trip: serialization → cloud routing → endpoint scheduling → vLLM inference → result return. This is inherent to the FaaS architecture and cannot be reduced from the client side.

---

## 5. Server Mode (Docker) Architecture

In server mode, STREAM runs as five Docker containers communicating over an isolated virtual network (`stream-network`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Docker Network (stream-network)                     │
│                                                                             │
│  ┌──────────┐     ┌──────────┐     ┌───────────────┐     ┌──────────────┐  │
│  │ React UI │────→│Middleware │────→│  LiteLLM GW   │────→│   Ollama     │  │
│  │ (host)   │     │  :5000   │     │    :4000      │     │   :11434     │  │
│  └──────────┘     └────┬─────┘     └───────┬───────┘     └──────────────┘  │
│                        │                   │                                │
│                        │                   │  ┌──────────────────────────┐  │
│                        │                   └─→│  Lakeshore Proxy :8001  │  │
│                        │                      │  (Globus Compute)       │  │
│                        │                      └───────────┬──────────────┘  │
│                        │                                  │                 │
│                   ┌────┴─────┐                            │                 │
│                   │PostgreSQL│                   ┌────────┴────────┐        │
│                   │  :5432   │                   │ Globus Cloud    │        │
│                   └──────────┘                   │  (AMQP broker)  │        │
│                                                  └────────┬────────┘        │
└─────────────────────────────────────────────────────────────┼───────────────┘
                                                              │
                                                     ┌────────┴────────┐
                                                     │  Lakeshore HPC  │
                                                     │  vLLM :8000     │
                                                     └─────────────────┘
```

### Request Flow (Server Mode)

The code path for a Lakeshore request in server mode:

1. **React frontend** sends `POST /v1/chat/completions` to the middleware on port 5000.

2. **Middleware** ([`routes/chat.py:141`](stream/middleware/routes/chat.py#L141)) receives the request, judges complexity, and routes to the Lakeshore tier.

3. **Streaming orchestrator** ([`core/streaming.py:213`](stream/middleware/core/streaming.py#L213)) calls `forward_to_litellm()` with model `"lakeshore-qwen"`.

4. **LiteLLM HTTP client** ([`core/litellm_client.py:69`](stream/middleware/core/litellm_client.py#L69)) sends an HTTP POST to the LiteLLM gateway server on port 4000.

5. **LiteLLM gateway** (running in its own container) reads `litellm_config.yaml` ([`gateway/litellm_config.yaml:97-106`](stream/gateway/litellm_config.yaml#L97-L106)), sees `lakeshore-qwen` maps to `api_base: http://lakeshore-proxy:8001/v1`, and forwards the request via HTTP to the Lakeshore proxy container.

6. **Lakeshore Proxy** ([`proxy/app.py:117`](stream/proxy/app.py#L117)) receives the OpenAI-compatible request and routes it through `_route_via_globus_compute()`.

7. **Globus Compute Client** ([`core/globus_compute_client.py:421`](stream/middleware/core/globus_compute_client.py#L421)) serializes and submits the inference function to Lakeshore.

8. The remote function executes on Lakeshore, calls vLLM at `http://ga-001:8000`, and returns the result.

9. The result flows back: Proxy → LiteLLM → Middleware → React UI (as SSE chunks).

### Key Docker Configuration

The proxy container mounts the user's Globus credentials from the host machine ([`docker-compose.yml:85`](docker-compose.yml#L85)):

```yaml
volumes:
  - ${HOME}/.globus_compute:/root/.globus_compute:rw
```

This allows the containerized proxy to authenticate with Globus Compute using tokens stored on the host machine, without requiring the user to authenticate inside the container.

---

## 6. Desktop Mode (PyInstaller) Architecture

In desktop mode, everything runs in a single process — no Docker, no separate servers, no inter-container HTTP:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Single Process (STREAM.app)                       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  FastAPI Server (:5000)                       │   │
│  │                                                              │   │
│  │  /v1/chat/completions  ←  React UI (served as static files)  │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  streaming.py → litellm_client.py                            │   │
│  │                      │                                       │   │
│  │         ┌────────────┼────────────┐                          │   │
│  │         ▼            ▼            ▼                          │   │
│  │    litellm lib   litellm lib  _forward_lakeshore()          │   │
│  │   (→ Ollama)    (→ Cloud API)       │                        │   │
│  │                                     ▼                        │   │
│  │                          globus_compute_client.py            │   │
│  │                          submit_inference()                  │   │
│  │                                     │                        │   │
│  └─────────────────────────────────────┼────────────────────────┘   │
│                                        │                            │
│  ┌────────────┐                        │                            │
│  │ PyWebView  │                        │                            │
│  │ (native    │               ┌────────┴────────┐                   │
│  │  window)   │               │ Globus Cloud    │                   │
│  └────────────┘               │  (AMQP broker)  │                   │
│                               └────────┬────────┘                   │
└────────────────────────────────────────┼────────────────────────────┘
                                         │
                                ┌────────┴────────┐
                                │  Lakeshore HPC  │
                                │  vLLM :8000     │
                                └─────────────────┘
```

### Request Flow (Desktop Mode)

The desktop mode path differs significantly from server mode:

1. **React frontend** (embedded in the FastAPI app as static files, served from the same port 5000) sends `POST /v1/chat/completions`.

2. **Middleware** ([`routes/chat.py:141`](stream/middleware/routes/chat.py#L141)) — same as server mode.

3. **Streaming orchestrator** ([`core/streaming.py:213`](stream/middleware/core/streaming.py#L213)) calls `forward_to_litellm()`.

4. **LiteLLM client** ([`core/litellm_client.py:123`](stream/middleware/core/litellm_client.py#L123)) detects `STREAM_MODE == "desktop"` and calls `forward_direct()` instead of making an HTTP request to a LiteLLM server (which doesn't exist in desktop mode).

5. **`forward_direct()`** ([`core/litellm_direct.py:323`](stream/middleware/core/litellm_direct.py#L323)) detects `model.startswith("lakeshore")` and calls `_forward_lakeshore()` instead of `litellm.acompletion()`.

6. **`_forward_lakeshore()`** ([`core/litellm_direct.py:182`](stream/middleware/core/litellm_direct.py#L182)) calls the Globus Compute client directly — a Python function call within the same process. No HTTP involved.

7. **`submit_inference()`** ([`core/globus_compute_client.py:421`](stream/middleware/core/globus_compute_client.py#L421)) submits the task to Lakeshore via Globus Compute (same as server mode from this point forward).

### Why Not Use LiteLLM for Lakeshore in Desktop Mode?

In server mode, LiteLLM forwards the Lakeshore request to `http://lakeshore-proxy:8001/v1` — a separate container. In desktop mode, the proxy routes are mounted into the same FastAPI app at `/lakeshore` ([`app.py:352-353`](stream/middleware/app.py#L352-L353)):

```python
if os.environ.get("STREAM_MODE") == "desktop":
    app.include_router(lakeshore_router, prefix="/lakeshore", tags=["Lakeshore"])
```

If we used `litellm.acompletion()` for Lakeshore in desktop mode, it would make an HTTP POST to `http://127.0.0.1:5000/lakeshore/v1/chat/completions` — which is **the same server we're running on**. This "self-connection" is documented in [`litellm_direct.py:192-217`](stream/middleware/core/litellm_direct.py#L192-L217):

```
litellm (our process) → HTTP POST → FastAPI (same process)
                                      ↓
                              Proxy handler → Globus Compute
```

The single-worker event loop must handle both sides of this HTTP request simultaneously, which can cause deadlocks and empty responses. The solution is to bypass HTTP entirely and call `globus_client.submit_inference()` directly as a Python function call.

### Desktop Startup Sequence

The desktop app startup is orchestrated in [`desktop/main.py`](stream/desktop/main.py):

1. **Apply desktop defaults** ([`desktop/config.py:33`](stream/desktop/config.py#L33)): Force-set environment variables (e.g., `STREAM_MODE=desktop`, `LAKESHORE_PROXY_URL=http://127.0.0.1:5000/lakeshore`) BEFORE any middleware modules are imported.

2. **First-run setup** if `~/.stream/` doesn't exist.

3. **Start Ollama** (local model server).

4. **Start FastAPI** in a background daemon thread ([`desktop/main.py:359`](stream/desktop/main.py#L359)).

5. **Poll `/health`** until the server is ready ([`desktop/main.py:253`](stream/desktop/main.py#L253)).

6. **Open PyWebView window** ([`desktop/main.py:402`](stream/desktop/main.py#L402)) — a native OS window rendering the React UI.

7. On window close, run cleanup (including `globus_client.shutdown()` via [`lifecycle.py:96-101`](stream/middleware/core/lifecycle.py#L96-L101)).

---

## 7. The Remote Function: How Code Executes on Lakeshore

The function that runs on Lakeshore is `remote_vllm_inference`, defined in [`globus_compute_client.py:73-120`](stream/middleware/core/globus_compute_client.py#L73-L120). It is a self-contained function that:

1. Imports `requests` (must be inside the function because the remote environment doesn't have STREAM's modules)
2. Sends an OpenAI-compatible HTTP POST to the vLLM server on Lakeshore's internal network
3. Returns the JSON response (or an error dictionary)

### The PyInstaller Serialization Problem

This function is defined via `exec()` from a source string rather than a normal `def` statement. This is documented in [`globus_compute_client.py:58-72`](stream/middleware/core/globus_compute_client.py#L58-L72):

**The problem:** PyInstaller bundles Python modules as `.pyc` bytecode, not `.py` source files. This bytecode contains references to PyInstaller's internal import system (`pyimod02_importers`). When Globus Compute serializes a function for the remote endpoint, it captures the bytecode. The Lakeshore endpoint doesn't have PyInstaller, so deserialization fails with `ModuleNotFoundError: No module named 'pyimod02_importers'`.

**The solution:** Define the function from a source string using `exec()` at runtime:

```python
_REMOTE_FN_SOURCE = """\
def remote_vllm_inference(vllm_url, model, messages, temperature, max_tokens, stream=False):
    # ... function body ...
"""

_ns = {}
exec(compile(_REMOTE_FN_SOURCE, "<remote_vllm_inference>", "exec"), _ns)
remote_vllm_inference = _ns["remote_vllm_inference"]
```

Python's standard compiler produces clean bytecode from the source string at runtime — bytecode with no PyInstaller references. This works in both development (normal Python) and production (PyInstaller bundle).

**Previous failed attempts** (documented in the source):
1. `CombinedCode` strategy → `inspect.getsource()` fails (no `.py` files in bundle)
2. `AllCodeStrategies` with normal `def` → `dill` by-reference → `"No module named 'stream'"`
3. `__module__ = '__main__'` → `dill` by-value → `"No module named 'pyimod02_importers'"`
4. `exec()` from source string → clean bytecode, works everywhere

### Serialization Strategy

The Executor's serializer is configured with `AllCodeStrategies` ([`globus_compute_client.py:254-256`](stream/middleware/core/globus_compute_client.py#L254-L256)):

```python
self._executor.serializer = ComputeSerializer(
    strategy_code=AllCodeStrategies()
)
```

`AllCodeStrategies` tries multiple serialization methods (dill by-value, dill by-reference, cloudpickle, etc.) to find one that produces bytecode compatible with the remote endpoint's Python version. This handles cases where the user's machine runs Python 3.12.12 but the Lakeshore endpoint runs Python 3.12.3.

---

## 8. Authentication Flow

Globus Compute uses OAuth2 for authentication. The flow differs between server and desktop modes.

### How Globus Authentication Works

1. **First time:** The user runs `globus-compute-endpoint configure` and `globus-compute-endpoint start` on Lakeshore, which triggers an OAuth2 browser flow. The resulting tokens are stored in `~/.globus_compute/storage.db` (a SQLite database on the user's machine or the server host).

2. **On each request:** The Globus Compute SDK reads tokens from `storage.db` and includes them in AMQP authentication. If tokens are expired, the SDK attempts to refresh them automatically.

3. **On token expiry:** If refresh fails, the SDK raises `GlobusAPIError` (HTTP 401/403) or `CommandLineLoginFlowEOFError` (in non-interactive environments like Docker).

### Server Mode Authentication

In Docker, the proxy container mounts `~/.globus_compute` from the host ([`docker-compose.yml:85`](docker-compose.yml#L85)):

```yaml
volumes:
  - ${HOME}/.globus_compute:/root/.globus_compute:rw
```

The user authenticates on the host machine once, and the Docker container reads those credentials. A `/reload-auth` endpoint ([`proxy/app.py:95`](stream/proxy/app.py#L95)) allows re-reading credentials after the user re-authenticates.

### Desktop Mode Authentication

The desktop app reads credentials directly from `~/.globus_compute/storage.db` on the user's machine. The `ensure_authenticated()` method ([`globus_compute_client.py:381`](stream/middleware/core/globus_compute_client.py#L381)) checks if tokens are valid before each submission.

If authentication fails, STREAM returns a structured error with `auth_required: True` ([`globus_compute_client.py:477-480`](stream/middleware/core/globus_compute_client.py#L477-L480)):

```python
return {
    "error": auth_message,
    "error_type": "AuthenticationError",
    "auth_required": True,
}
```

The React frontend detects this flag and displays authentication instructions to the user.

### Credential Reload

The `reload_credentials()` method ([`globus_compute_client.py:329`](stream/middleware/core/globus_compute_client.py#L329)) clears the Globus SDK's internal singleton cache and forces a fresh read from `storage.db`. This is needed because the SDK caches the `GlobusApp` instance at the module level — without explicitly clearing it, newly saved credentials would be ignored.

---

## 9. Context Window Management

Context window management for Lakeshore is centralized in [`config.py:267-286`](stream/middleware/config.py#L267-L286) as a single source of truth:

```python
MODEL_CONTEXT_LIMITS = {
    "lakeshore-qwen": {"total": 32768, "reserve_output": 2048},
    # ... other models ...
}
```

- **`total` (32768):** Must match vLLM's `--max-model-len` on Lakeshore.
- **`reserve_output` (2048):** Reserved for the model's response. Used as the `max_tokens` parameter.

This configuration is consumed in three places:

1. **`globus_compute_client.py`** ([line 459-461](stream/middleware/core/globus_compute_client.py#L459-L461)): If `max_tokens` is not passed to `submit_inference()`, reads `reserve_output` from config.
2. **`litellm_direct.py`** ([line 253-254](stream/middleware/core/litellm_direct.py#L253-L254)): Desktop mode reads `reserve_output` as `max_tokens` for the direct Globus call.
3. **`proxy/app.py`** ([line 174-176](stream/proxy/app.py#L174-L176)): Server mode reads `reserve_output` as the default `max_tokens` in the proxy endpoint.

Additionally, [`context_window.py`](stream/middleware/utils/context_window.py) uses `MODEL_CONTEXT_LIMITS` to validate that a conversation fits within the model's context window before sending it to any tier.

---

## 10. Streaming and Response Delivery

### The Streaming Challenge with Globus Compute

Unlike Local and Cloud tiers which support true token-by-token streaming, Globus Compute is inherently **non-streaming** — you submit a function and get the complete result back. There is no way to get partial results while the function is running on the remote endpoint.

### Simulated Streaming (Server Mode)

In server mode, the Lakeshore proxy converts the complete JSON response into simulated SSE events ([`proxy/app.py:288-385`](stream/proxy/app.py#L288-L385)):

1. Split the response text into 2-word chunks
2. Yield each chunk as a separate SSE `data:` event
3. Add 50ms delays between chunks to create a natural "typing" effect

This provides a consistent user experience across all tiers — the text appears progressively.

### Direct SSE Conversion (Desktop Mode)

In desktop mode, `_forward_lakeshore()` ([`litellm_direct.py:302-320`](stream/middleware/core/litellm_direct.py#L302-L320)) converts the complete vLLM response into word-by-word SSE delta chunks:

```python
words = content.split(" ")
for i, word in enumerate(words):
    text = word if i == 0 else f" {word}"
    chunk = {"choices": [{"index": 0, "delta": {"content": text}}]}
    yield f"data: {json.dumps(chunk)}"
```

These chunks are in the same format that `litellm.acompletion()` would produce for streaming, so the downstream pipeline ([`streaming.py`](stream/middleware/core/streaming.py)) processes them identically regardless of tier.

### The asyncio.wait() Fix for Lakeshore

The streaming pipeline includes a gap warning system ([`streaming.py:88-210`](stream/middleware/core/streaming.py#L88-L210)) that warns users when no data arrives for 5 seconds. The original implementation used `asyncio.wait_for()`, which **cancels** the underlying operation on timeout. For Lakeshore, where the first chunk takes 5-15 seconds (Globus round-trip), this would cancel the request before any data arrived.

The fix uses `asyncio.wait()` instead ([`streaming.py:169-172`](stream/middleware/core/streaming.py#L169-L172)):

```python
done, _ = await asyncio.wait(
    {next_task},
    timeout=CHUNK_GAP_WARNING_SECONDS,
)
```

The key difference:
- `asyncio.wait_for(task, timeout=5)`: **CANCELS** the task after 5 seconds
- `asyncio.wait({task}, timeout=5)`: Returns empty set, task **KEEPS RUNNING**

This allows the streaming pipeline to send "Lakeshore is taking longer than usual, please wait..." to the user while the Globus Compute task continues running undisturbed.

---

## 11. Performance Analysis and Latency Breakdown

### Timing Instrumentation

STREAM instruments the Lakeshore round-trip at three measurement points ([`globus_compute_client.py:489-556`](stream/middleware/core/globus_compute_client.py#L489-L556)):

```python
t_start = time.perf_counter()
gce = self._get_executor()           # Step 1: Get/create AMQP connection
t_executor = time.perf_counter()
future = gce.submit(...)             # Step 2: Serialize and send task
t_submit = time.perf_counter()
result = await asyncio.to_thread(    # Step 3: Wait for result
    future.result, timeout=GLOBUS_TASK_TIMEOUT
)
t_result = time.perf_counter()
```

### Measured Results

From production logs on a campus network:

| Metric | Request 1 (cold) | Request 2 (warm) |
|--------|-----------------|------------------|
| Executor creation | 0.28s | 0.00s |
| Task submission | 0.56s | 0.00s |
| Wait for result | 4.95s | 5.18s |
| **Total** | **5.79s** | **5.18s** |

### Latency Breakdown

The ~5 second wait time breaks down approximately as:

| Component | Estimated Time | Notes |
|-----------|---------------|-------|
| Client → AMQP broker | ~50ms | TCP/TLS to AWS |
| Broker → Lakeshore endpoint | ~100ms | Inter-network routing |
| Endpoint task scheduling | ~200ms | Worker pool dispatch |
| Function deserialization | ~100ms | dill unpickling |
| vLLM inference | ~2-3s | GPU generation (varies with prompt length) |
| Result serialization | ~50ms | dill pickling |
| Result return path | ~150ms | Lakeshore → broker → client |
| SDK overhead | ~200ms | Future resolution, callbacks |

The dominant factor is **vLLM inference time** (~2-3s for typical queries) plus **network round-trip overhead** (~1-2s for the multi-hop path through Globus cloud).

---

## 12. Persistent Executor Optimization

### The Problem

Initially, STREAM created a new `Executor` for every request:

```python
with Executor(endpoint_id=self.endpoint_id) as gce:
    future = gce.submit(...)
    result = future.result()
```

Each `Executor.__init__()` opens a new TCP+TLS+AMQP connection to the Globus broker, costing 0.5-1.5 seconds per request.

### The Solution

STREAM now maintains a persistent Executor across requests ([`globus_compute_client.py:232-281`](stream/middleware/core/globus_compute_client.py#L232-L281)):

```python
def _get_executor(self) -> Executor:
    if self._executor is None:
        self._executor = Executor(endpoint_id=self.endpoint_id)
        self._executor.serializer = ComputeSerializer(strategy_code=AllCodeStrategies())
    return self._executor
```

On the first request, this creates the Executor and establishes the AMQP connection. On subsequent requests, it returns the same Executor — reusing the existing connection.

### Stale Connection Handling

AMQP connections can drop from network glitches, Globus service restarts, or token expiry. When this happens, STREAM detects the error and retries once with a fresh connection ([`globus_compute_client.py:654-679`](stream/middleware/core/globus_compute_client.py#L654-L679)):

```python
if not _retry:
    self._reset_executor()
    return await self.submit_inference(
        messages=messages, temperature=temperature,
        max_tokens=max_tokens, model=model,
        _retry=True,  # Prevent infinite retry loops
    )
```

The `_retry` flag prevents infinite loops — if the retry also fails, the error is returned to the user.

### Lifecycle Management

The persistent Executor is properly shut down during app exit ([`lifecycle.py:92-101`](stream/middleware/core/lifecycle.py#L92-L101)):

```python
async def shutdown():
    try:
        import stream.proxy.app as _proxy_app
        if _proxy_app.globus_client is not None:
            _proxy_app.globus_client.shutdown()
    except Exception as e:
        logger.debug(f"Globus client shutdown (best effort): {e}")
```

The `shutdown()` method ([`globus_compute_client.py:283-294`](stream/middleware/core/globus_compute_client.py#L283-L294)) calls `self._executor.shutdown(wait=False, cancel_futures=True)` to properly close the AMQP connection.

### Measured Impact

| Metric | Without Persistent Executor | With Persistent Executor |
|--------|---------------------------|------------------------|
| First request total | ~5.8s | ~5.8s (same — must connect) |
| Second request total | ~5.8s | ~5.2s (saved ~0.6s) |
| Executor creation (2nd+) | ~0.5-1.5s | ~0.00s |

The persistent Executor saves approximately 0.6 seconds per request after the first, by eliminating the AMQP connection setup overhead.

---

## 13. Error Handling and Resilience

STREAM implements comprehensive error handling for Globus Compute failures ([`globus_compute_client.py:566-684`](stream/middleware/core/globus_compute_client.py#L566-L684)):

| Error Type | Detection | Action |
|-----------|-----------|--------|
| **Task timeout** | `TimeoutError` after 120s | Return error, no retry |
| **Auth expired** | `GlobusAPIError` HTTP 401/403 | Reset executor, prompt re-auth |
| **Non-interactive auth** | `CommandLineLoginFlowEOFError` | Reset executor, prompt re-auth |
| **Deserialization failure** | `DeserializationError` / `TaskExecutionFailed` | Return error with upgrade suggestion (don't reset — connection is fine) |
| **Database missing** | `"unable to open database file"` in error | Prompt authentication |
| **Stale AMQP connection** | Any other unexpected error | Reset executor, retry once |

### Tier Fallback

If Lakeshore fails entirely, STREAM's streaming pipeline ([`streaming.py:595-734`](stream/middleware/core/streaming.py#L595-L734)) automatically falls back to the next available tier:

- Medium complexity: Lakeshore → Cloud → Local
- Low complexity: Local → Lakeshore → Cloud
- High complexity: Cloud → Lakeshore → Local

The fallback is transparent to the user — the UI shows "Lakeshore unavailable — using Cloud instead" and continues with the response.

---

## 14. Code Reference Map

### Core Lakeshore Files

| File | Purpose | Key Functions |
|------|---------|--------------|
| [`stream/middleware/core/globus_compute_client.py`](stream/middleware/core/globus_compute_client.py) | Globus Compute SDK integration | `submit_inference()`, `_get_executor()`, `_reset_executor()`, `shutdown()` |
| [`stream/proxy/app.py`](stream/proxy/app.py) | Lakeshore proxy (standalone + embedded routes) | `proxy_chat_completions()`, `_route_via_globus_compute()`, `_convert_json_to_sse_stream()` |
| [`stream/middleware/core/litellm_direct.py`](stream/middleware/core/litellm_direct.py) | Desktop-mode direct calls (bypasses HTTP) | `_forward_lakeshore()`, `forward_direct()` |
| [`stream/middleware/core/litellm_client.py`](stream/middleware/core/litellm_client.py) | LiteLLM HTTP client (mode dispatcher) | `forward_to_litellm()` |

### Configuration Files

| File | Purpose |
|------|---------|
| [`stream/middleware/config.py`](stream/middleware/config.py) | Central config: `MODEL_CONTEXT_LIMITS`, `LAKESHORE_PROXY_URL`, `USE_GLOBUS_COMPUTE` |
| [`stream/desktop/config.py`](stream/desktop/config.py) | Desktop env vars: `LAKESHORE_PROXY_URL=http://127.0.0.1:5000/lakeshore` |
| [`stream/gateway/litellm_config.yaml`](stream/gateway/litellm_config.yaml) | Model mappings: `lakeshore-qwen → openai/Qwen/Qwen2.5-1.5B-Instruct` |
| [`docker-compose.yml`](docker-compose.yml) | Container definitions: proxy on port 8001, credential volume mount |

### Supporting Files

| File | Purpose |
|------|---------|
| [`stream/middleware/core/streaming.py`](stream/middleware/core/streaming.py) | SSE orchestration, gap warnings, `asyncio.wait()` fix |
| [`stream/middleware/core/lifecycle.py`](stream/middleware/core/lifecycle.py) | Startup/shutdown: Globus Executor cleanup |
| [`stream/middleware/utils/context_window.py`](stream/middleware/utils/context_window.py) | Context window validation using `MODEL_CONTEXT_LIMITS` |
| [`stream/middleware/routes/chat.py`](stream/middleware/routes/chat.py) | Main chat endpoint: complexity routing, tier selection |
| [`stream/middleware/core/query_router.py`](stream/middleware/core/query_router.py) | Tier routing with fallback chains |
| [`stream/middleware/core/tier_health.py`](stream/middleware/core/tier_health.py) | Health checks for all tiers including Lakeshore proxy |
| [`stream/desktop/main.py`](stream/desktop/main.py) | Desktop app entry point: startup sequence, PyWebView |

---

## 15. Future Work: Hybrid Approaches to Reduce Lakeshore Latency

The current Globus Compute approach adds approximately 2–3 seconds of overhead per request on top of the actual vLLM inference time (~2–3 seconds). This section explores two hybrid approaches that could significantly reduce this latency while retaining the security and firewall-transparency benefits of Globus Compute.

### 15.1 The Latency Problem Recap

To understand where improvement is possible, recall the measured latency breakdown:

```
Total latency: ~5 seconds
├── Globus overhead: ~2-3 seconds
│   ├── AMQP publish to broker: ~50ms
│   ├── Broker → Lakeshore routing: ~100ms
│   ├── Endpoint task scheduling: ~200ms
│   ├── Function deserialization: ~100ms
│   ├── Result serialization: ~50ms
│   ├── Result return (Lakeshore → broker → client): ~150ms
│   └── SDK overhead (callbacks, Future resolution): ~200ms
│
└── Actual vLLM inference: ~2-3 seconds
    └── GPU processes the query and generates tokens
```

If we could send requests **directly** to vLLM (bypassing the Globus round-trip), latency would drop to just the 2–3 seconds of inference. The challenge is that vLLM runs behind UIC's campus firewall — it's not reachable from the public internet. Globus Compute solves this by routing through its cloud infrastructure, but that routing is what adds the overhead.

The key insight is: **Globus Compute is excellent at solving the authentication and firewall problem, but it doesn't have to be the transport layer for every single request.** We can use Globus for the hard part (establishing access) and then switch to a faster channel for the data-intensive part (streaming inference).

### 15.2 Approach 1: Globus-Assisted SSH Tunnel (Control Plane / Data Plane Separation)

#### The Core Idea

This approach separates the connection into two layers, borrowing a concept from network engineering called **control plane / data plane separation**:

- **Control plane** (Globus Compute): Handles authentication, discovers where vLLM is running, and manages the connection lifecycle. Runs infrequently — once per session.
- **Data plane** (SSH tunnel): Carries the actual inference requests at full speed. Runs for every request.

Think of it like air travel: the control plane is the booking system (security checks, authentication, gate assignment), and the data plane is the actual airplane (fast transport). You go through security once, then fly multiple times.

#### How SSH Tunneling Works

SSH (Secure Shell) is a protocol for secure remote access. Beyond running commands on remote machines, SSH can create **tunnels** — encrypted passages that forward network traffic between machines. There are two types relevant here:

**Local port forwarding** (what we'd use):

```
Your machine                    Lakeshore login node              GPU node
============                    ====================              ========
                    SSH tunnel
localhost:8000 ←================→ login.lakeshore.uic.edu ------→ ga-001:8000
     ↑               (encrypted)                                      ↑
     │                                                                │
  STREAM sends                                              vLLM listens here
  requests here                                             (OpenAI-compatible API)
```

The command to create this tunnel:
```bash
ssh -L 8000:ga-001:8000 user@lakeshore.uic.edu -N -f
```

Breakdown:
- `-L 8000:ga-001:8000`: Forward local port 8000 to `ga-001:8000` through the SSH connection
- `user@lakeshore.uic.edu`: SSH into the Lakeshore login node (which CAN reach `ga-001`)
- `-N`: Don't run any remote command (just tunnel)
- `-f`: Run in background

Once this tunnel is established, `http://localhost:8000/v1/chat/completions` goes directly to vLLM — no Globus overhead, no AMQP routing. The latency drops to **just the inference time** plus SSH encryption overhead (negligible, ~5ms).

#### Where Globus Compute Fits In

The problem with SSH tunnels is that they require knowing:
1. Which GPU node is vLLM running on (could change if the SLURM job restarts)
2. What port it's listening on
3. Whether the SLURM job is still active

Globus Compute can answer all of these questions by running a **discovery function** on Lakeshore:

```python
def discover_vllm_endpoint():
    """
    Runs on Lakeshore via Globus Compute.
    Discovers where vLLM is currently running.
    """
    import subprocess

    # Check if the vLLM SLURM job is running
    result = subprocess.run(
        ["squeue", "-u", "stream", "-n", "vllm-server", "--format=%N,%T"],
        capture_output=True, text=True
    )

    # Parse the output to find the node and status
    # e.g., "ga-001,RUNNING"
    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        return {"status": "not_running", "node": None, "port": None}

    node, status = lines[1].split(",")

    return {
        "status": status.lower(),
        "node": node,           # e.g., "ga-001"
        "port": 8000,           # vLLM's configured port
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_model_len": 32768,
    }
```

This function runs through the same Globus Compute infrastructure we already have — it solves the firewall problem for this one-time discovery call. The ~5-second Globus overhead is acceptable here because it only happens once.

#### The Complete Hybrid Flow

```
SESSION STARTUP (once, ~5 seconds):
═══════════════════════════════════

1. STREAM starts up
2. submit_inference(discover_vllm_endpoint)  ←── Globus Compute (through firewall)
3. Result: {"node": "ga-001", "port": 8000, "status": "running"}
4. Establish SSH tunnel:
     ssh -L 8000:ga-001:8000 user@lakeshore.uic.edu -N -f
5. Verify tunnel: HTTP GET http://localhost:8000/health
6. Store tunnel info: {"local_port": 8000, "pid": 12345}

EVERY SUBSEQUENT REQUEST (~2-3 seconds):
════════════════════════════════════════

1. User asks a question (complexity=MEDIUM → Lakeshore)
2. HTTP POST http://localhost:8000/v1/chat/completions  ←── Direct through tunnel
3. vLLM generates response on GPU
4. Response streams back through SSH tunnel
5. Total latency: ~2-3s (inference only!)

TUNNEL RECOVERY (if tunnel drops):
══════════════════════════════════

1. Request to localhost:8000 fails (ConnectionRefused)
2. Re-run discovery via Globus Compute (~5s)
3. Re-establish SSH tunnel
4. Retry the request
```

#### What This Approach Enables

Beyond lower latency, a direct connection to vLLM unlocks **true token-by-token streaming**. Currently, Globus Compute returns the complete response at once (because FaaS is batch-oriented). With an SSH tunnel, we can pass `"stream": true` to vLLM and receive tokens as they're generated:

```
Current (Globus Compute, batch):
  [====== 5s wait ======][all tokens arrive at once]

With SSH tunnel (true streaming):
  [== 2s first token ==][token][token][token][token]...
```

This makes the Lakeshore tier feel as responsive as the Cloud tier.

#### Prerequisites and Limitations

| Requirement | Details |
|-------------|---------|
| **SSH key access** | User needs passwordless SSH access to Lakeshore login nodes. The same user who set up the Globus Compute endpoint likely already has this. |
| **Network access to login nodes** | SSH to `lakeshore.uic.edu` must be reachable. Works from campus network and VPN. Does NOT work from arbitrary public networks (coffee shop, home without VPN). |
| **`paramiko` or system SSH** | The desktop app would need to create SSH tunnels programmatically. Python's `paramiko` library or `subprocess` with the system `ssh` command could do this. |
| **SLURM job persistence** | vLLM must be running as a persistent SLURM job. If the job ends, the tunnel becomes useless. Globus discovery detects this. |

#### When This Approach Works vs. Doesn't

| Scenario | Works? | Why |
|----------|--------|-----|
| Student on campus Wi-Fi | Yes | Direct network path to login nodes |
| Researcher on VPN | Yes | VPN extends campus network to their location |
| Faculty in office (wired) | Yes | On campus network |
| Student at home (no VPN) | No | Login nodes not reachable from public internet |
| Docker deployment on campus server | Yes | Server has campus network access |
| Docker deployment on AWS/cloud | No | Would need VPN or Globus Compute (current approach) |

This limitation is why the hybrid approach should be an **additional mode**, not a replacement. STREAM would try the SSH tunnel first (fast path) and fall back to pure Globus Compute (slow but universal) if the tunnel can't be established.

#### Implementation Sketch

A new routing mode in the existing architecture:

```
_route_via_globus_compute    →  Current: every request through Globus (~5s)
_route_via_ssh               →  Current: manual SSH tunnel (user sets up)
_route_via_globus_ssh_hybrid →  New: Globus discovers, SSH carries data (~2-3s)
```

The proxy code ([`proxy/app.py`](stream/proxy/app.py)) already has `_route_via_globus_compute()` and `_route_via_ssh()`. The hybrid would combine them:

```python
async def _route_via_globus_ssh_hybrid(model, messages, temperature, max_tokens, stream):
    """
    Use Globus Compute for discovery and auth, SSH tunnel for data transport.

    First request: discover vLLM via Globus, establish SSH tunnel
    Subsequent requests: forward directly through tunnel
    If tunnel drops: re-discover and re-establish
    """
    # Check if we have an active tunnel
    if not _ssh_tunnel_active():
        # Phase 1: Discovery via Globus Compute
        vllm_info = await globus_client.submit_inference(discover_vllm_endpoint)

        if vllm_info["status"] != "running":
            raise HTTPException(503, "vLLM not running on Lakeshore")

        # Phase 2: Establish SSH tunnel
        _establish_ssh_tunnel(
            remote_node=vllm_info["node"],
            remote_port=vllm_info["port"],
            local_port=8000,
        )

    # Phase 3: Forward through tunnel (fast path)
    return await _route_via_ssh(model, messages, temperature, max_tokens, stream)
```

### 15.3 Approach 2: AMQP-Based Token Streaming (Streaming Through the Existing Channel)

#### The Core Idea

The Globus Compute Executor already maintains a persistent AMQP connection between the user's machine and the Lakeshore endpoint (through the Globus cloud broker). This connection is bidirectional — the endpoint can send messages back to the client. Currently, we use this channel to send **one big message** (the complete vLLM response). What if instead we sent **many small messages** (one per token)?

This approach doesn't bypass Globus Compute — it uses the existing infrastructure more efficiently. The AMQP channel becomes a real-time streaming pipe.

#### How AMQP Works (Background)

AMQP (Advanced Message Queuing Protocol) is a messaging protocol used by systems like RabbitMQ and, in this case, Globus Compute's infrastructure. Understanding a few AMQP concepts helps explain this approach:

**Message broker:** A server that receives messages from producers and delivers them to consumers. Globus runs this broker at `compute.amqps.globus.org`. Think of it as a post office — you drop off mail (task), and it delivers it to the recipient (endpoint).

**Queues:** Named mailboxes where messages wait for consumers to pick them up. Each Globus Compute endpoint has a task queue (incoming work) and each client has a result queue (outgoing results).

**Publish/Subscribe:** A pattern where a producer publishes messages to a topic, and any number of consumers subscribed to that topic receive copies. This is different from request/response — it's more like a radio broadcast.

The key property of AMQP for our purposes: **messages are delivered as soon as they arrive at the broker**. There's no batching, no waiting for the full response. If the endpoint publishes 100 small messages, the client receives 100 small messages, each arriving as soon as the broker processes it. The per-message latency through the broker is typically **10-50 milliseconds**.

#### Current Flow vs. Streaming Flow

**Current (batch response):**

```
Lakeshore endpoint                    AMQP Broker (AWS)              STREAM client
==================                    ================               =============

Receives task
Deserializes function
Calls vLLM (non-streaming)
Waits 2-3 seconds...
Gets complete response (all tokens)
Serializes entire response
Publishes ONE message ────────────→ Routes to client queue ────────→ Receives result
                                                                     All tokens at once
                                                                     after ~5 seconds
```

**Proposed (streaming through AMQP):**

```
Lakeshore endpoint                    AMQP Broker (AWS)              STREAM client
==================                    ================               =============

Receives task
Deserializes function
Calls vLLM (streaming=True)
  token₁ generated ──────────────→ Routes immediately ────────────→ Displays token₁
  (50ms later)                                                       (~3s after submit)
  token₂ generated ──────────────→ Routes immediately ────────────→ Displays token₂
  (50ms later)                                                       (~50ms later)
  token₃ generated ──────────────→ Routes immediately ────────────→ Displays token₃
  ...                                                                ...
  token_N generated ──────────────→ Routes immediately ────────────→ Displays token_N
  [DONE] ────────────────────────→ Routes immediately ────────────→ Stream complete
```

The first token still takes ~3 seconds (Globus routing + vLLM processing the prompt). But after that, each subsequent token arrives within ~50ms of being generated — the user sees text appearing progressively, just like the Cloud tier.

#### Why This Is Better Than Simulated Streaming

Currently, STREAM simulates streaming for Lakeshore ([`proxy/app.py:288-385`](stream/proxy/app.py#L288-L385)) by splitting the complete response into word groups and yielding them with 50ms delays. This creates the visual appearance of streaming, but:

1. **The user still waits ~5 seconds before seeing anything.** The entire response must complete on the GPU before any text appears.
2. **The simulated speed doesn't match real generation speed.** We artificially pace the output at 40 words/second regardless of how fast the GPU actually generated them.
3. **There's no way to cancel mid-generation.** By the time we're "streaming" the response, the GPU has already finished all the work.

With true AMQP streaming, the first token appears after ~3 seconds (instead of ~5), and subsequent tokens arrive at the GPU's actual generation speed. The user can also potentially cancel the request and stop the GPU mid-generation.

#### What the Remote Function Would Look Like

Instead of calling vLLM with `stream=False` and returning the complete response, the remote function would call vLLM with `stream=True` and publish each token as a separate message:

```python
def remote_vllm_inference_streaming(vllm_url, model, messages, temperature,
                                     max_tokens, result_queue_id):
    """
    Runs on Lakeshore. Streams tokens back through AMQP.

    Instead of returning one big result, this function publishes
    each token as a separate AMQP message to the client's queue.
    """
    import requests
    import json
    # hypothetical: a Globus-provided function to publish to the client's queue
    from globus_compute_endpoint.streaming import publish_intermediate_result

    endpoint = f"{vllm_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,  # Tell vLLM to stream!
    }

    # vLLM returns a streaming response (Server-Sent Events)
    response = requests.post(endpoint, json=payload, stream=True, timeout=60)

    accumulated_usage = {}

    for line in response.iter_lines():
        if not line:
            continue

        line_text = line.decode("utf-8")

        if line_text.startswith("data: [DONE]"):
            # Final message: send usage stats
            publish_intermediate_result(result_queue_id, {
                "type": "done",
                "usage": accumulated_usage,
            })
            break

        if line_text.startswith("data: "):
            chunk = json.loads(line_text[6:])

            # Extract the token text from the SSE chunk
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    # Publish this token immediately through AMQP
                    # The client receives it ~50ms later
                    publish_intermediate_result(result_queue_id, {
                        "type": "token",
                        "content": content,
                    })

            # Capture usage if present (usually in the last chunk)
            if "usage" in chunk:
                accumulated_usage = chunk["usage"]

    # Return a summary (the standard Globus Compute return value)
    return {"status": "streamed", "usage": accumulated_usage}
```

On the client side, STREAM would subscribe to the result queue and yield each token as an SSE event:

```python
async def _forward_lakeshore_streaming(model, messages, temperature, correlation_id):
    """
    Hypothetical: receive streamed tokens from Lakeshore through AMQP.
    """
    # Subscribe to intermediate results from the Globus Compute task
    async for message in globus_client.stream_results(task_id):
        if message["type"] == "token":
            chunk = {
                "choices": [{"index": 0, "delta": {"content": message["content"]}}],
            }
            yield f"data: {json.dumps(chunk)}"

        elif message["type"] == "done":
            final = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": message.get("usage", {}),
            }
            yield f"data: {json.dumps(final)}"
            yield "data: [DONE]"
            return
```

#### The Technical Challenge: Globus Compute SDK Limitations

The main obstacle is that the Globus Compute SDK does not currently expose an API for publishing intermediate results from within a running function. The SDK's model is:

```
submit(function, args) → Future → result (one shot)
```

There is no built-in way for the function to send partial results while it's still running. Implementing AMQP streaming would require one of the following:

**Option A: Custom Globus Compute endpoint extension.**

The Globus Compute endpoint is open-source. One could extend it to support a `publish_intermediate_result()` function that running tasks can call. This would publish messages to the client's result queue using the endpoint's existing AMQP connection. The SDK on the client side would need a corresponding `stream_results()` method.

This is the cleanest approach but requires changes to both the endpoint software (running on Lakeshore) and the client SDK (running in STREAM). It could be proposed as a feature to the Globus Compute team.

**Option B: Side-channel AMQP connection.**

Instead of using Globus Compute's internal AMQP infrastructure, set up a separate AMQP broker (like RabbitMQ on a cloud server) that both the Lakeshore endpoint and the STREAM client can reach. The remote function publishes tokens to this broker, and the client consumes them.

```
Lakeshore GPU node                  External RabbitMQ              STREAM client
==================                  =================              =============

Remote function runs
vLLM generates token₁
Publish to queue ──────────────────→ Queue stores ──────────────→ Client receives
vLLM generates token₂
Publish to queue ──────────────────→ Queue stores ──────────────→ Client receives
...
```

This is simpler to implement (no changes to Globus SDK) but requires deploying and maintaining an external message broker. It also introduces a new network dependency.

**Option C: WebSocket relay.**

The remote function opens a WebSocket connection to a relay server (could be the same cloud server), and the STREAM client also connects. Tokens flow through the relay in real-time.

This is similar to Option B but uses WebSocket instead of AMQP. It's more familiar to web developers but has the same dependency on an external relay server.

#### Comparison: Current vs. Hybrid SSH vs. AMQP Streaming

| Metric | Current (Globus Batch) | Hybrid SSH Tunnel | AMQP Streaming |
|--------|----------------------|-------------------|----------------|
| **First token latency** | ~5 seconds | ~2-3 seconds | ~3 seconds |
| **Per-token latency** | N/A (all at once) | Real-time (~5ms) | Near-real-time (~50ms) |
| **True streaming** | No (simulated) | Yes | Yes |
| **Works off-campus** | Yes | No (needs SSH access) | Yes |
| **Additional infrastructure** | None | SSH keys configured | Custom endpoint or external broker |
| **Implementation complexity** | Current (done) | Medium | High |
| **Cancel mid-generation** | No | Yes | Yes |
| **Firewall transparent** | Yes | Partial (needs SSH) | Yes |

#### Research Contribution Potential

The AMQP streaming approach addresses a gap in the Globus Compute ecosystem: **interactive FaaS workloads**. Most Globus Compute use cases are batch-oriented (submit a job, get a result minutes later). AI inference is fundamentally different — users expect real-time, token-by-token responses. Proposing and implementing a streaming extension to Globus Compute could benefit not just STREAM but any research application that needs interactive access to HPC resources:

- Real-time data visualization from running simulations
- Interactive scientific computing notebooks connected to HPC backends
- Streaming sensor data processing on GPU clusters
- Any application where the user is waiting for progressive results

This could be presented as a collaboration opportunity with the Globus team at the University of Chicago and Argonne National Laboratory.

### 15.4 Recommended Roadmap

Based on feasibility and impact, the recommended implementation order is:

| Phase | Approach | Effort | Impact |
|-------|----------|--------|--------|
| **Phase 1** | Globus-assisted SSH tunnel | Medium | High (2-3s → direct vLLM) |
| **Phase 2** | AMQP streaming (side-channel broker) | Medium-High | Medium (true streaming, universal) |
| **Phase 3** | AMQP streaming (native Globus extension) | High (requires Globus collaboration) | Very High (ecosystem-wide benefit) |

Phase 1 delivers the largest immediate latency improvement for campus users. Phase 2 brings true streaming to all users regardless of network location. Phase 3 is a research contribution that could benefit the broader Globus Compute community.

---

## Summary

STREAM's Lakeshore connectivity demonstrates how a research middleware system can bridge the gap between user-facing desktop/web applications and HPC resources behind campus firewalls. The key architectural decisions are:

1. **Globus Compute as the transport layer** — eliminates the need for SSH tunnels, VPN, or firewall modifications while providing secure, authenticated access to HPC GPUs.

2. **Dual-mode architecture** — the same `GlobusComputeClient` code runs in both Docker (via the proxy container) and desktop (via direct function calls), with the mode difference handled at the routing layer.

3. **Persistent AMQP connections** — reduce per-request overhead from ~1.5s to ~0s by reusing the Executor across requests.

4. **`exec()`-based remote functions** — solve the PyInstaller serialization incompatibility that would otherwise prevent desktop deployment.

5. **Simulated streaming** — converts Globus Compute's batch-style responses into SSE streams, providing a consistent ChatGPT-like experience across all tiers.

6. **Centralized context window configuration** — `MODEL_CONTEXT_LIMITS` in `config.py` serves as the single source of truth for token limits across proxy, desktop direct calls, and context validation.

The irreducible ~5-second latency for Lakeshore requests is inherent to the Globus Compute FaaS architecture (multi-hop AMQP routing through cloud infrastructure) and represents the trade-off for secure, firewall-transparent HPC access without SSH or VPN infrastructure.
