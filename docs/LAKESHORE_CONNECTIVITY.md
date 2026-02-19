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
14. [Per-Model Health Checks](#14-per-model-health-checks)
15. [Code Reference Map](#15-code-reference-map)
16. [Future Work: Hybrid Approaches to Reduce Lakeshore Latency](#16-future-work-hybrid-approaches-to-reduce-lakeshore-latency)

---

## 1. Overview

STREAM (Smart Tiered Routing Engine for AI Models) is a middleware system that routes AI queries to one of three tiers based on complexity:

| Tier | Backend | Use Case | Cost |
|------|---------|----------|------|
| **Local** | Ollama (Llama 3.2:3b on Apple Silicon) | Simple queries (greetings, definitions) | Free |
| **Lakeshore** | vLLM (multiple models on UIC HPC GPUs) | Medium queries (explanations, comparisons) | Free (university GPU) |
| **Cloud** | Claude Sonnet 4 / GPT-4 Turbo | Complex queries (design, analysis, research) | Pay-per-token |

The Lakeshore tier is the most architecturally interesting because it connects a user's laptop to a remote HPC cluster behind a university firewall. This connection works through **Globus Compute**, a Function-as-a-Service (FaaS) platform for research computing.

STREAM operates in two modes:
- **Server mode (Docker):** Five containerized microservices communicating over a Docker virtual network.
- **Desktop mode (PyInstaller):** A single-process native macOS/Windows app with everything embedded.

Both modes use the same Globus Compute client code to reach Lakeshore, but the request path to that client differs significantly.

---

## 2. What is Lakeshore?

Lakeshore is UIC's HPC cluster operated by ACER. For STREAM, it provides:

- **GPU node:** `ga-002` with NVIDIA A100 (MIG 3g.40gb partitions — one per model)
- **Model servers:** Five vLLM instances, each serving a different model on its own port
- **Access:** Behind UIC's campus firewall — not directly reachable from the internet

### Multi-Model Architecture

STREAM runs five models on Lakeshore, each as a separate vLLM instance on a dedicated MIG slice (3g.40gb, 39.5 GiB usable VRAM):

| Model Key | HuggingFace Model | Port | Context Window | Description |
|-----------|------------------|------|---------------|-------------|
| `lakeshore-qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 8000 | 32K tokens | General purpose (fast) |
| `lakeshore-coder-1.5b` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 8001 | 32K tokens | Coding specialist |
| `lakeshore-deepseek-r1` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 8002 | 32K tokens | Deep reasoning |
| `lakeshore-qwq` | `Qwen/Qwen2.5-1.5B-Instruct` | 8003 | 32K tokens | Reasoning |
| `lakeshore-qwen-32b` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 8004 | 8K tokens | High quality (32B AWQ) |

Each vLLM instance is started via a SLURM job script and exposes an OpenAI-compatible REST API at `http://ga-002:<port>`. The model configuration is centralized in [`config.py`](stream/middleware/config.py) as `LAKESHORE_MODELS`, which maps each model key to its HuggingFace name, port, and description. The `get_lakeshore_vllm_url()` function dynamically constructs the correct URL by extracting the host from `VLLM_SERVER_URL` (default `http://ga-002:8000`) and replacing the port based on the selected model.

These APIs are **only accessible from within the Lakeshore cluster network** — not from a user's laptop, Docker container, or the public internet. This is why we need Globus Compute.

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
    vllm_url,                # e.g., "http://ga-002:8004" (port varies per model)
    hf_model,                # e.g., "Qwen/Qwen2.5-32B-Instruct-AWQ"
    messages,                # The chat conversation
    temperature,             # 0.7
    max_tokens,              # From MODEL_CONTEXT_LIMITS (varies per model)
    False,                   # stream=False
)
```

The `vllm_url` is dynamically constructed from the model key. For example, `lakeshore-qwen-32b` maps to port 8004, so the URL becomes `http://ga-002:8004`. The HuggingFace model name is resolved from `LAKESHORE_MODELS` in `config.py`.

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

3. **Remote Execution:** The deserialized `remote_vllm_inference` function runs on a Lakeshore compute node. Because the function runs **inside** the Lakeshore network, it can directly reach the vLLM servers at `http://ga-002:<port>` (port determined by the selected model).

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
                                                                           ga-002:<port>/v1
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
                                                     │  vLLM :8000-    │
                                                     │        :8004    │
                                                     └─────────────────┘
```

### Request Flow (Server Mode)

The code path for a Lakeshore request in server mode:

1. **React frontend** sends `POST /v1/chat/completions` to the middleware on port 5000, including the selected `lakeshore_model` (e.g., `lakeshore-qwen-32b`).

2. **Middleware** ([`routes/chat.py:141`](stream/middleware/routes/chat.py#L141)) receives the request, judges complexity, and routes to the Lakeshore tier.

3. **Streaming orchestrator** ([`core/streaming.py:213`](stream/middleware/core/streaming.py#L213)) calls `forward_to_litellm()` with the selected Lakeshore model name.

4. **LiteLLM HTTP client** ([`core/litellm_client.py:69`](stream/middleware/core/litellm_client.py#L69)) sends an HTTP POST to the LiteLLM gateway server on port 4000.

5. **LiteLLM gateway** (running in its own container) reads `litellm_config.yaml` ([`gateway/litellm_config.yaml`](stream/gateway/litellm_config.yaml)), sees the Lakeshore model maps to `api_base: http://lakeshore-proxy:8001/v1`, and forwards the request via HTTP to the Lakeshore proxy container.

6. **Lakeshore Proxy** ([`proxy/app.py:117`](stream/proxy/app.py#L117)) receives the OpenAI-compatible request and routes it through `_route_via_globus_compute()`.

7. **Globus Compute Client** ([`core/globus_compute_client.py:421`](stream/middleware/core/globus_compute_client.py#L421)) resolves the model key to the correct vLLM URL (e.g., `lakeshore-qwen-32b` → `http://ga-002:8004`) and HuggingFace model name (e.g., `Qwen/Qwen2.5-32B-Instruct-AWQ`), then serializes and submits the inference function to Lakeshore.

8. The remote function executes on Lakeshore, calls the appropriate vLLM instance at `http://ga-002:<port>`, and returns the result.

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
                                │  vLLM :8000-    │
                                │        :8004    │
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

Context window management for Lakeshore is centralized in [`config.py`](stream/middleware/config.py) as a single source of truth:

```python
MODEL_CONTEXT_LIMITS = {
    "lakeshore-qwen-1.5b":  {"total": 32768, "reserve_output": 2048},
    "lakeshore-coder-1.5b": {"total": 32768, "reserve_output": 2048},
    "lakeshore-deepseek-r1": {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwq":        {"total": 32768, "reserve_output": 2048},
    "lakeshore-qwen-32b":   {"total": 8192,  "reserve_output": 1024},
    "lakeshore-qwen":       {"total": 32768, "reserve_output": 2048},  # Legacy alias
    # ... local and cloud models ...
}
```

- **`total`:** Must match vLLM's `--max-model-len` on Lakeshore. The 1.5B models use 32K tokens; the 32B AWQ model is limited to 8K due to KV cache memory constraints on the 40GB MIG slice.
- **`reserve_output`:** Reserved for the model's response. Used as the `max_tokens` parameter. The 32B model reserves 1024 tokens (shorter responses trade for more input context).

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

## 14. Per-Model Health Checks

With five models running on separate vLLM instances, STREAM needs to verify that each individual model is operational — not just that Globus Compute authentication works. A model's vLLM instance may be down (SLURM job ended, GPU error, OOM) while others remain healthy.

### The Problem with Authentication-Only Health Checks

Previously, the Lakeshore health check only verified Globus authentication status. If the user's Globus tokens were valid, the tier showed as "available" — even if the actual vLLM instance serving the selected model had crashed. This led to confusing failures: the health indicator was green, but inference requests failed with connection errors.

### Real Inference Health Checks

STREAM now performs **real 1-token inference tests** through the full Globus Compute path to verify each model is actually running. The `check_model_health()` method ([`globus_compute_client.py`](stream/middleware/core/globus_compute_client.py)) sends a minimal request through Globus Compute to the specific model's vLLM instance:

```python
# Minimal test: send "hi", request 1 token, deterministic (temperature=0)
messages = [{"role": "user", "content": "hi"}]
result = submit_inference(messages, temperature=0.0, max_tokens=1, model=model_key)
```

This test traverses the entire inference path: AMQP → Globus Cloud → Lakeshore endpoint → vLLM on the correct port → response back. If any part of the chain is broken for that specific model, the health check catches it.

### Per-Model Cache Keys

Health check results are cached to avoid hammering Globus Compute on every frontend poll. The cache uses per-model keys ([`routes/health.py`](stream/middleware/routes/health.py)):

```python
# Without model parameter: only checks Globus auth (fast, but incomplete)
cache_key = "lakeshore"

# With model parameter: real inference test for that specific model
cache_key = "lakeshore:lakeshore-qwen-32b"
```

The frontend passes the currently selected Lakeshore model in the health check request (`GET /health/tiers?lakeshore_model=lakeshore-qwen-32b`), so the health indicator reflects whether **that specific model** is available.

### Timeout Configuration

Per-model health checks use a 20-second timeout (configurable via `LAKESHORE_HEALTH_TIMEOUT`). This is generous because the full Globus Compute round-trip for even a 1-token request takes ~5 seconds, and the 32B model's initial prompt processing can add several more seconds on a cold start.

---

## 15. Code Reference Map

### Core Lakeshore Files

| File | Purpose | Key Functions |
|------|---------|--------------|
| [`stream/middleware/core/globus_compute_client.py`](stream/middleware/core/globus_compute_client.py) | Globus Compute SDK integration | `submit_inference()`, `check_model_health()`, `_get_executor()`, `_reset_executor()`, `shutdown()` |
| [`stream/proxy/app.py`](stream/proxy/app.py) | Lakeshore proxy (standalone + embedded routes) | `proxy_chat_completions()`, `_route_via_globus_compute()`, `_convert_json_to_sse_stream()` |
| [`stream/middleware/core/litellm_direct.py`](stream/middleware/core/litellm_direct.py) | Desktop-mode direct calls (bypasses HTTP) | `_forward_lakeshore()`, `forward_direct()` |
| [`stream/middleware/core/litellm_client.py`](stream/middleware/core/litellm_client.py) | LiteLLM HTTP client (mode dispatcher) | `forward_to_litellm()` |

### Configuration Files

| File | Purpose |
|------|---------|
| [`stream/middleware/config.py`](stream/middleware/config.py) | Central config: `LAKESHORE_MODELS`, `MODEL_CONTEXT_LIMITS`, `LAKESHORE_PROXY_URL`, `get_lakeshore_vllm_url()` |
| [`stream/desktop/config.py`](stream/desktop/config.py) | Desktop env vars: `LAKESHORE_PROXY_URL=http://127.0.0.1:5000/lakeshore` |
| [`stream/gateway/litellm_config.yaml`](stream/gateway/litellm_config.yaml) | Model mappings: Lakeshore models → `openai/<HF_name>` with proxy base URL |
| [`docker-compose.yml`](docker-compose.yml) | Container definitions: proxy on port 8001, credential volume mount |

### Supporting Files

| File | Purpose |
|------|---------|
| [`stream/middleware/core/streaming.py`](stream/middleware/core/streaming.py) | SSE orchestration, gap warnings, `asyncio.wait()` fix |
| [`stream/middleware/core/lifecycle.py`](stream/middleware/core/lifecycle.py) | Startup/shutdown: Globus Executor cleanup |
| [`stream/middleware/utils/context_window.py`](stream/middleware/utils/context_window.py) | Context window validation using `MODEL_CONTEXT_LIMITS` |
| [`stream/middleware/routes/chat.py`](stream/middleware/routes/chat.py) | Main chat endpoint: complexity routing, tier selection |
| [`stream/middleware/core/query_router.py`](stream/middleware/core/query_router.py) | Tier routing with fallback chains |
| [`stream/middleware/routes/health.py`](stream/middleware/routes/health.py) | Health check endpoints: per-model Lakeshore health via 1-token inference |
| [`stream/middleware/core/tier_health.py`](stream/middleware/core/tier_health.py) | Health checks for all tiers including Lakeshore proxy |
| [`stream/desktop/main.py`](stream/desktop/main.py) | Desktop app entry point: startup sequence, PyWebView |

---

## 16. Future Work: Hybrid Approaches to Reduce Lakeshore Latency

The current Globus Compute approach adds approximately 2–3 seconds of overhead per request on top of the actual vLLM inference time (~2–3 seconds). This section explores two hybrid approaches that could significantly reduce this latency while retaining the security and firewall-transparency benefits of Globus Compute.

### 16.1 The Latency Problem Recap

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

### 16.2 Approach 1: SSH Tunnel with Globus Compute as Universal Fallback

#### The Two Authentication Systems

STREAM currently uses two independent systems to reach Lakeshore, each with its own authentication:

**SSH authentication (key-based):**
When you SSH to Lakeshore, your machine presents its private key (e.g., `~/.ssh/id_ed25519`) and the server checks if the corresponding public key exists in `~/.ssh/authorized_keys` on Lakeshore. If it matches, you're in — no password prompt, no visible authentication step. The handshake happens transparently in milliseconds. This is how users who already have SSH access to Lakeshore experience it: it "just works."

**Globus authentication (OAuth2-based):**
Globus Compute uses OAuth2 through CILogon — a federated identity service for research. The user logs in with their university credentials in a browser, and Globus stores tokens in `~/.globus_compute/storage.db`. The Globus Compute endpoint on Lakeshore verifies these tokens. This is a different identity system — having an SSH key doesn't give you Globus access, and having Globus tokens doesn't give you SSH access.

The key insight: **these are orthogonal systems solving the same problem (proving you're authorized) through different mechanisms**. For users who already have SSH keys configured for Lakeshore — which is common for researchers and HPC users — the SSH path is simpler, faster, and already works.

#### Why SSH Tunnels Are Faster

An SSH tunnel creates a direct, encrypted pipe from your machine to the vLLM server on Lakeshore. The tunnel goes through a Lakeshore login node (which has network access to the GPU nodes):

```
Your machine                    Lakeshore login node              GPU node
============                    ====================              ========
                    SSH tunnel
localhost:8000 ←================→ login.lakeshore.uic.edu ------→ ga-002:8000
     ↑               (encrypted)                                      ↑
     │                                                                │
  STREAM sends                                              vLLM listens here
  requests here                                             (OpenAI-compatible API)
```

The command to create this tunnel:
```bash
ssh -L 8000:ga-002:8000 user@lakeshore.uic.edu -N -f
```

Breakdown:
- `-L 8000:ga-002:8000`: Forward local port 8000 to `ga-002:8000` through the SSH connection
- `user@lakeshore.uic.edu`: SSH into the Lakeshore login node (which CAN reach `ga-002`)
- `-N`: Don't run any remote command (just tunnel)
- `-f`: Run in background

Once this tunnel is established, `http://localhost:8000/v1/chat/completions` goes directly to vLLM — no Globus overhead, no AMQP routing, no serialization. The latency drops to **just the inference time** (~2-3 seconds) plus SSH encryption overhead (negligible, ~5ms).

STREAM already supports this path. The `_route_via_ssh` function in [`proxy/app.py:238`](stream/proxy/app.py#L238) forwards requests directly to a vLLM endpoint URL. Setting `USE_GLOBUS_COMPUTE=false` and `LAKESHORE_VLLM_ENDPOINT=http://localhost:8000` in `.env` activates this mode.

#### What SSH Tunnels Enable Over Globus Compute

Beyond lower latency, a direct connection to vLLM unlocks **true token-by-token streaming**. Globus Compute is inherently batch-oriented (submit function → get complete result). With an SSH tunnel, we can pass `"stream": true` to vLLM and receive tokens as they're generated:

```
Current (Globus Compute, batch):
  [====== 5s wait ======][all tokens arrive at once]

With SSH tunnel (true streaming):
  [== 2s first token ==][token][token][token][token]...
```

This makes the Lakeshore tier feel as responsive as the Cloud tier — the user sees text appearing progressively instead of waiting for the entire response.

#### Where SSH Fails and Globus Compute Is Needed

SSH tunnels require two things that Globus Compute does not:

1. **SSH key access to Lakeshore login nodes.** The user must have a private key on their machine (e.g., `~/.ssh/id_ed25519`) with the corresponding public key in `~/.ssh/authorized_keys` on Lakeshore. Not all STREAM users will have this — particularly students in a classroom setting who interact with Lakeshore only through STREAM and have never SSH'd into an HPC cluster.

2. **Network path to login nodes.** The user must be on a network that can reach Lakeshore's login nodes — typically the campus network or a VPN. From a coffee shop, home network without VPN, or a cloud server, SSH to `lakeshore.uic.edu` will not connect.

Globus Compute solves both problems: it uses OAuth2 (university SSO, no SSH keys needed) and routes through the Globus cloud infrastructure (works from any internet connection, regardless of firewall). This is what makes it universal — but that universality comes at the cost of ~2-3 seconds of overhead per request.

| Scenario | SSH Tunnel | Globus Compute |
|----------|-----------|----------------|
| Researcher on campus with SSH keys | **Best choice** (~2-3s) | Works but slower (~5s) |
| Student on campus, no SSH keys | Cannot use | **Only option** (~5s) |
| Researcher at home with VPN + SSH keys | Works (~2-3s) | Works (~5s) |
| Student at home, no VPN | Cannot use | **Only option** (~5s) |
| Docker on campus server | Works (~2-3s) | Works (~5s) |
| Docker on AWS/cloud | Cannot use | **Only option** (~5s) |

#### The Recommended Architecture: SSH-First with Globus Fallback

Rather than choosing one transport, STREAM should try SSH first (fast path) and fall back to Globus Compute (universal path) when SSH is unavailable:

```
STARTUP:
════════

1. Try to establish SSH tunnel to Lakeshore
   ssh -L 8000:ga-002:8000 user@lakeshore.uic.edu -N -f

2a. SUCCESS → Use SSH tunnel for all requests (~2-3s per request)
    Set LAKESHORE_VLLM_ENDPOINT=http://localhost:8000

2b. FAILURE (no keys, no network) → Fall back to Globus Compute (~5s per request)
    Use existing submit_inference() path

RUNTIME:
════════

If SSH tunnel drops mid-session:
  1. Detect failure (ConnectionRefused on localhost:8000)
  2. Attempt to re-establish tunnel
  3. If re-establishment fails → switch to Globus Compute for remaining session
```

This gives users with SSH access the fastest possible experience, while ensuring STREAM works for everyone through Globus Compute regardless of their network or key setup.

#### Optional: Globus Compute for Dynamic Discovery

If the vLLM SLURM job moves to a different node (e.g., job restarts after a node failure), the SSH tunnel target would be wrong. For deployments where the node and port are not fixed, Globus Compute can serve as a **discovery mechanism** — running a quick function on Lakeshore to query `squeue` and find where vLLM is currently running:

```python
def discover_vllm_endpoint():
    """
    Runs on Lakeshore via Globus Compute.
    Discovers where vLLM is currently running.
    """
    import subprocess

    result = subprocess.run(
        ["squeue", "-u", "stream", "-n", "vllm-server", "--format=%N,%T"],
        capture_output=True, text=True
    )

    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        return {"status": "not_running", "node": None, "port": None}

    node, status = lines[1].split(",")

    return {
        "status": status.lower(),
        "node": node,           # e.g., "ga-002"
        "port": 8000,           # vLLM's configured port
    }
```

This costs ~5 seconds (one Globus round-trip) but only needs to run once per session, or when the tunnel breaks. For fixed deployments where the node is known in advance (as is currently the case with `ga-002`), this discovery step is unnecessary — the tunnel targets can be configured statically (one tunnel per port, or a range).

#### Implementation in the Existing Codebase

The proxy code ([`proxy/app.py`](stream/proxy/app.py)) already has both paths:

```
_route_via_globus_compute  →  Current: every request through Globus (~5s)
_route_via_ssh             →  Current: forwards to LAKESHORE_VLLM_ENDPOINT
```

The SSH-first architecture would add a startup-time tunnel manager and a routing preference:

```python
async def _route_lakeshore(model, messages, temperature, max_tokens, stream):
    """
    Route to Lakeshore using the fastest available transport.
    SSH tunnel if available, Globus Compute as universal fallback.
    """
    if _ssh_tunnel_active():
        # Fast path: direct to vLLM through SSH (~2-3s)
        return await _route_via_ssh(model, messages, temperature, max_tokens, stream)
    else:
        # Universal fallback: through Globus Compute (~5s)
        return await _route_via_globus_compute(
            model, messages, temperature, max_tokens, stream
        )
```

### 16.3 Approach 2: AMQP-Based Token Streaming (Streaming Through the Existing Channel)

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

#### The Technical Challenge: Why Modifying Globus Compute Internals Is Impractical

The idea of streaming tokens through Globus Compute's own AMQP infrastructure is architecturally elegant but practically infeasible for a single-developer project. Here's why.

The Globus Compute SDK's model is strictly one-shot:

```
submit(function, args) → Future → result (one shot)
```

There is no built-in API for publishing intermediate results from within a running function. The SDK does not expose `publish_intermediate_result()` or `stream_results()` — a function runs to completion and returns one value. Implementing streaming *through* Globus would require modifying both the endpoint software (running on Lakeshore) and the client SDK (running in STREAM).

More fundamentally, the Globus Compute endpoint runs your function in an isolated **worker process**, but the AMQP connection lives in the **endpoint daemon process**. These are separate OS processes. The worker has no direct access to the daemon's AMQP connection — there is a serialization boundary between them. Bridging this would require:

1. An IPC (inter-process communication) channel between worker and daemon
2. A new message type in the Globus Compute protocol for intermediate results
3. A new client-side API to receive these intermediate results
4. Changes to the AMQP queue topology (currently one result message per task)

This is not a weekend project — it's a months-long effort that requires deep knowledge of the Globus Compute internals, and any implementation would **break when Globus updates their SDK** because it depends on undocumented internal architecture. The Globus team could reasonably change how workers communicate with the daemon, how results are serialized, or how AMQP queues are structured — and any of those changes would break a custom streaming extension.

This approach makes sense as a **feature request to the Globus team** (and as a research contribution to the Globus ecosystem), but not as something to build and maintain independently.

#### The Practical Solution: WebSocket Side-Channel Relay

Instead of fighting Globus Compute's architecture, we work *alongside* it. The insight is that we can **separate the control plane from the data plane**:

- **Control plane (Globus Compute):** Handles authentication, job submission, and task lifecycle — the things Globus is good at.
- **Data plane (WebSocket relay):** Handles real-time token streaming — the thing Globus isn't designed for.

Globus Compute still submits the function to Lakeshore. But instead of having the function return the complete response through Globus, the function opens a **WebSocket connection to an external relay server** and streams tokens through that side channel. The STREAM client also connects to the same relay server and receives tokens in real-time.

```
                         ┌─────────────────────┐
                         │   WebSocket Relay    │
                         │  (public server or   │
                         │   cloud VM, ~$5/mo)  │
                         └──────┬────────┬──────┘
                         outbound│        │outbound
                        connection│      connection
                                 │        │
┌────────────────────┐           │        │           ┌──────────────────┐
│   Lakeshore GPU    │───────────┘        └───────────│  STREAM client   │
│                    │  tokens flow →→→→→→→→→          │  (user's laptop) │
│  vLLM generates    │                                │                  │
│  token by token    │                                │  displays tokens │
│                    │                                │  progressively   │
└────────────────────┘                                └──────────────────┘
         ↑                                                     │
         │              ┌─────────────────────┐                │
         └──────────────│   Globus Compute    │────────────────┘
            job submit  │   (control plane)   │  submit + poll
            via AMQP    │                     │  via AMQP
                        └─────────────────────┘
```

Both Lakeshore and the user's machine make **outbound** connections to the relay — neither needs to accept incoming connections. This is critical because it means:

- **The relay works through firewalls.** Lakeshore's firewall blocks inbound connections but allows outbound. The user's home NAT/firewall also blocks inbound. Since both sides connect *out* to the relay, no firewall rules need to change.
- **No SSH keys required.** The relay is a public WebSocket endpoint. Authentication is handled by Globus (the task ID acts as a shared secret — only the submitter and the worker know it).
- **Works from anywhere.** Campus, home, coffee shop — as long as the user has internet, the relay works.

#### Why Can't the Relay Be on the User's Machine?

A natural question: why deploy a separate relay server? Why not run it on the user's own laptop?

The problem is **NAT (Network Address Translation) and firewalls**. When you're at home, your laptop sits behind your router. Your laptop can make outbound connections to the internet (that's how browsing works), but **nothing on the internet can connect inbound to your laptop** — your router doesn't know which device to forward the connection to. This is the same fundamental problem that Globus Compute was built to solve.

For the relay to work, **both sides** (Lakeshore and the user) must be able to connect to it. If the relay is on the user's laptop:

```
Lakeshore → tries to connect to user's laptop → BLOCKED by user's NAT/firewall ✗
```

If the relay is on a public server:

```
Lakeshore → connects outbound to relay server → OK ✓
User      → connects outbound to relay server → OK ✓
```

This is the same reason video calls use TURN servers, and why Globus itself routes through AWS — when both endpoints are behind firewalls, you need a publicly-reachable intermediary.

The relay server itself is tiny (under 50 lines of code, uses negligible CPU/bandwidth). It could run on:
- A $5/month cloud VM (DigitalOcean, AWS Lightsail)
- A free-tier cloud function (AWS Lambda, Google Cloud Run)
- Any existing server with a public IP (e.g., a UIC department server)

#### Hosting the Relay on Lakeshore: Detailed Analysis

The ideal deployment for the WebSocket relay is on Lakeshore itself — no external resources, no monthly costs, and all data stays within UIC's network. However, this requires ACER's cooperation on two fronts: opening a firewall port and approving a lightweight process on cluster infrastructure. This section details the technical requirements, security implications, and deployment options to support that conversation.

##### The Firewall Problem

The reason STREAM uses Globus Compute in the first place is that Lakeshore's firewall blocks inbound connections from the internet on all ports except a few explicitly allowed ones (like SSH on port 22). This same firewall would block a WebSocket relay.

Lakeshore's firewall sits between the internet and the cluster. It has rules that control which incoming connections are allowed:

```
Lakeshore firewall rules (simplified):
───────────────────────────────────────
Port 22 (SSH)     → ALLOW inbound     (this is how users ssh to lakeshore.uic.edu)
Port 443 (HTTPS)  → ALLOW inbound     (if any web services are exposed)
Port 8765         → no rule → BLOCKED (our hypothetical relay port)
Port 8000         → no rule → BLOCKED (vLLM's port on the GPU node)
Everything else   → BLOCKED by default
```

If we run the WebSocket relay on Lakeshore and a user's STREAM app tries to connect:

```
User's laptop:  websocket.connect("ws://lakeshore.uic.edu:8765")

Network path:
  User's laptop  →  internet  →  UIC campus network  →  Lakeshore firewall  →  DROPPED
                                                          (port 8765 not allowed)
```

The firewall doesn't inspect what the traffic is doing — it doesn't know the difference between "streaming AI tokens" and "anything else." It simply drops all incoming TCP connections on ports it hasn't been told to allow. This is standard HPC security practice: minimize the attack surface by only exposing what's strictly necessary.

**What we'd need from ACER:** Open one TCP port (e.g., 8765) for inbound WebSocket connections. This is the same type of request as asking for a web service to be externally accessible — ACER has likely handled similar requests for other research projects that need external connectivity (web dashboards, Jupyter notebooks, API endpoints, etc.).

##### The Login Node Problem

HPC clusters have two types of nodes with very different roles:

**Login nodes** (`lakeshore.uic.edu`) are shared machines where users:
- Log in via SSH
- Edit files, compile code
- Submit SLURM jobs to compute nodes
- Transfer data

Login nodes are **shared resources** — dozens or hundreds of users are logged in simultaneously. Running persistent services (daemons, servers, long-running processes) on login nodes is against standard HPC usage policies because:
- A misbehaving service could consume CPU/memory and slow down the login experience for everyone
- Login nodes aren't designed for high availability (they get rebooted for maintenance)
- If every researcher ran a service on the login node, it would become unusable

So the naive approach — "start the relay on the login node and leave it running" — is not appropriate. You should not be running persistent services on the login node.

**Compute nodes** (`ga-002`, etc.) are where actual work runs, managed by SLURM:
- Users submit jobs, SLURM allocates resources
- Jobs run in isolation with dedicated CPU/memory/GPU
- When the job finishes, the resources are released

The GPU node `ga-002` where vLLM runs is a compute node. But compute nodes are only reachable from within Lakeshore's internal network — they don't have public IP addresses and the firewall doesn't expose them. So running the relay on a compute node has the same firewall problem, plus the relay would die when the SLURM job ends.

##### Deployment Options for ACER Discussion

There are several ways ACER could accommodate the relay, listed from most to least ideal:

**Option 1: Dedicated lightweight service node or VM (best option)**

ACER could provision a small VM or container on Lakeshore's infrastructure specifically for the relay. This is the cleanest approach:

- The relay runs as a managed service, separate from login and compute nodes
- It doesn't interfere with other users or with job scheduling
- ACER can control resource limits, monitor it, and restart it if needed
- The firewall rule points to this specific VM, not to a login node

Many HPC centers already have "service nodes" or "gateway nodes" for exactly this kind of purpose — hosting web dashboards (like Open OnDemand), Jupyter hubs, or API endpoints that need external access. The relay would fit naturally alongside these.

**What to ask ACER:** "Could we get a small VM or service endpoint on Lakeshore's infrastructure to run a lightweight WebSocket relay for our research project? It needs one open TCP port for external WebSocket connections and access to Lakeshore's internal network to communicate with GPU compute nodes."

**Why not a SLURM job?** A natural thought is to run the relay as a SLURM job on a compute node. This doesn't work because compute nodes don't have public IP addresses — they're only reachable from within Lakeshore's internal network. The user's laptop can't connect to a compute node any more than it can connect directly to vLLM. ACER would still need port forwarding or a reverse proxy, and now there's an additional problem: SLURM might assign the relay to a different node each time the job restarts, so the forwarding target keeps changing. SLURM jobs also have time limits, causing periodic downtime. A relay needs a stable, always-on home — not a job scheduler.

**Option 2: Run on a login node with ACER's explicit approval**

If ACER doesn't have service node infrastructure, they might approve running the relay directly on a login node as an exception. The relay's resource usage is genuinely negligible:

- **CPU:** Near zero. The relay just forwards WebSocket messages — it doesn't parse, transform, or compute anything. It's doing the same work as an `ssh` connection, which is already running on the login node for every user.
- **Memory:** Under 10MB. The relay holds open WebSocket connections (a few KB each) and a dictionary mapping task IDs to connections. Even with 100 concurrent users, it would use less memory than a single `vim` session.
- **Bandwidth:** Minimal. AI responses are text — a typical 500-token response is ~2KB. Even 100 concurrent requests would generate less traffic than a single file transfer.
- **Disk:** Zero. The relay is stateless — it doesn't log, doesn't write files, doesn't store anything.

For context, the Globus Compute endpoint daemon itself is a persistent process that runs on Lakeshore continuously, maintaining AMQP connections and managing worker processes. The relay would use far fewer resources than the endpoint daemon. If ACER already permits the Globus endpoint daemon, the relay is a much lighter burden.

**What to ask ACER:** "Our WebSocket relay uses under 10MB of memory and near-zero CPU — it just forwards text messages between two WebSocket connections. Could we run it as a lightweight daemon on a login node, similar to how the Globus Compute endpoint runs? Or is there a service node / gateway we should use instead?"

**Option 3: Run behind ACER's existing reverse proxy**

Many HPC centers run a reverse proxy (like NGINX or Apache) on their externally-facing infrastructure to expose web services (Jupyter, Open OnDemand, Grafana dashboards). If Lakeshore has such infrastructure, the relay could run on any internal node and be exposed through the existing reverse proxy.

```
Internet → ACER's NGINX (port 443, already open) → proxy pass to relay (internal node:8765)
```

This is elegant because it reuses existing infrastructure — no new firewall rules needed. The relay would be accessible at something like `wss://lakeshore.uic.edu/stream-relay/` through the existing HTTPS port. ACER might prefer this because it doesn't increase the firewall's attack surface at all.

**What to ask ACER:** "Does Lakeshore have a reverse proxy or web gateway (like Open OnDemand or an NGINX frontend)? If so, could we route WebSocket traffic through it to an internal relay process?"

##### Security Analysis: What the Open Port Exposes

ACER's primary concern will be: "Does opening this port create a security risk for Lakeshore?" Here's a thorough analysis.

**What the relay does:**

The relay is a message forwarder. Its complete behavior is:
1. Accept incoming WebSocket connections
2. The first message on each connection is a task ID (a UUID string)
3. Group connections by task ID
4. Forward messages from one connection to all other connections with the same task ID
5. Clean up when connections close

That's it. The relay does not:
- Execute any code received from connections
- Access the filesystem (no reads, no writes, no directory listings)
- Authenticate to Lakeshore services (no SSH, no SLURM, no sudo)
- Have elevated privileges (runs as a regular user process)
- Store any data (stateless — when a connection closes, it's gone)
- Spawn subprocesses or run shell commands

**Comparison to SSH (port 22), which is already open:**

| Capability | SSH (port 22) | WebSocket relay (port 8765) |
|-----------|--------------|---------------------------|
| Execute arbitrary commands | Yes (full shell) | No |
| Read/write files | Yes (any file user can access) | No |
| Transfer files | Yes (scp, sftp) | No |
| Tunnel to internal services | Yes (port forwarding) | No |
| Escalate privileges | Possible (if sudo configured) | No |
| Attack surface | Large (complex protocol, auth) | Tiny (forwards bytes) |

The relay exposes **far less** than SSH. If an attacker connects to the relay, the worst they can do is:
- Connect and send messages to a channel (but without a valid task ID, no one is listening)
- Try to guess task IDs to eavesdrop on AI responses (mitigated by using UUIDs — 128 bits of randomness, infeasible to guess)
- Open many connections to consume memory (mitigated by connection limits, which can be set in the relay code)

**Potential concerns and mitigations:**

| Concern | Risk Level | Mitigation |
|---------|-----------|------------|
| **Unauthorized eavesdropping** (attacker reads AI responses) | Low | Task IDs are UUIDs (128-bit random). Can add short-lived auth tokens for extra security. |
| **DoS via connection flooding** | Low | Set max connections per IP (e.g., 10). The relay's memory footprint per connection is ~1KB, so even 10,000 connections would use ~10MB. |
| **Code execution through relay** | None | The relay forwards bytes. It doesn't interpret, compile, or execute anything it receives. There is no `eval()`, no `exec()`, no shell access. |
| **Lateral movement** (attacker uses relay to access other Lakeshore services) | None | The relay is a WebSocket forwarder. It doesn't make connections to other services. An attacker connecting to the relay cannot use it to reach SLURM, SSH, databases, or any other internal service. |
| **Data exfiltration** | None | The relay doesn't access the filesystem or any Lakeshore data. It only sees the AI tokens that flow through it, which are generated by the model in response to user prompts — not stored research data. |
| **Privilege escalation** | None | The relay runs as a regular user (no root, no sudo). Even if the relay process were somehow compromised, the attacker would only have the permissions of that user account. |

**TLS encryption (recommended):**

For production deployment, the relay should use TLS (`wss://` instead of `ws://`). This encrypts all traffic between the user and the relay, preventing anyone on the network path from reading the AI responses. If ACER's reverse proxy option (Option 3 above) is available, TLS comes for free — the existing HTTPS certificate handles it.

If running standalone, ACER could provide a TLS certificate for the relay, or the relay could use a Let's Encrypt certificate if it has a DNS name.

##### Summary: What to Discuss With ACER

Here's a concise summary of the request and its justification:

**The request:** We need a way for external users to receive real-time WebSocket messages from a lightweight relay process running on Lakeshore's infrastructure. This requires either (a) one open TCP port, (b) a reverse proxy route, or (c) a small service VM.

**Why:** STREAM uses Globus Compute to submit AI inference jobs to Lakeshore's GPUs. Currently, the entire response must complete before any text is sent back to the user (~5 seconds of blank screen). With a WebSocket relay, tokens can stream to the user as the GPU generates them — the same experience as ChatGPT. This requires a persistent relay process that both the GPU node (internal) and the user's laptop (external) can connect to.

**Resource requirements:**
- CPU: Near zero (just forwarding text messages)
- Memory: Under 10MB (even with many concurrent users)
- Bandwidth: Negligible (text tokens, ~2KB per response)
- Disk: Zero (stateless, no logging)

**Security posture:**
- The relay is a pure message forwarder — it doesn't execute code, access files, or connect to other services
- It exposes far less than SSH (port 22), which is already open
- Connection authentication via UUID task IDs (128-bit random, infeasible to guess)
- Can add TLS encryption and auth tokens for additional security
- If hosted on Lakeshore, all data stays within UIC's network — no tokens leave university infrastructure

**Preferred deployment (in order):**
1. Service VM or dedicated endpoint (cleanest separation)
2. Route through existing reverse proxy/web gateway (no new firewall rules)
3. Login node daemon with explicit approval (lightest resource usage, but login nodes aren't meant for services)
4. External cloud VM (fallback if ACER can't accommodate on-cluster hosting — see prototype demo below)

#### How Globus Compute's Role Changes With the Relay

To summarize how the two approaches differ at a high level:

**Current approach (Globus Compute alone):**

1. Globus Compute submits the job to Lakeshore
2. The remote function calls vLLM and waits for the **complete** response (all tokens)
3. The function returns the full response through Globus Compute's AMQP channel
4. Globus delivers the complete result back to STREAM — all tokens arrive at once
5. STREAM then *simulates* streaming by splitting the response into word chunks with artificial delays

The user sees nothing for ~5 seconds, then text starts appearing — but it's fake streaming. The GPU finished its work long ago; we're just replaying the result slowly.

**WebSocket relay approach:**

1. Globus Compute submits the job to Lakeshore — same as before
2. The remote function calls vLLM with `stream=True` and forwards each token through the **WebSocket relay** as it's generated
3. The user's STREAM app receives tokens through the relay in real-time — Globus is not involved in this data flow
4. Globus Compute eventually gets back a small summary (`{"status": "streamed"}`) just to close out the task

Globus Compute's role changes from **carrying the full payload** to **starting the job**. The heavy lifting (token delivery) moves to the relay. Globus still handles the hard parts — authentication, getting through the firewall, launching the function on the right GPU node — but it no longer carries the actual response data.

#### Why Is This Faster? Time-to-First-Token vs. Total Time

An important distinction: the WebSocket relay approach does **not** reduce the total time to generate a response. If vLLM takes 3 seconds of compute to produce 200 tokens, it still takes 3 seconds either way. What changes is **when the user sees the first token**.

**Current approach (batch):**

```
Time:  0s          1s          2s          3s          4s          5s
       │           │           │           │           │           │
       ├── Globus submit ──────┤           │           │           │
       │   (AMQP routing)      │           │           │           │
       │                       ├── vLLM generates all tokens ─────┤
       │                       │   (user sees NOTHING)            │
       │                       │                                  ├── Globus return ──→ ALL tokens
       │                       │                                  │   arrive at once
       │                       │                                  │
User:  [waiting...             waiting...              waiting... │ sees everything]
                                                                  ↑
                                                          first token at ~5s
```

**WebSocket relay approach (streaming):**

```
Time:  0s          1s          2s          3s          3.1s   3.2s  ...  5s
       │           │           │           │           │      │          │
       ├── Globus submit ──────┤           │           │      │          │
       │   (AMQP routing)      │           │           │      │          │
       │                       ├── vLLM generates ─────┼──────┼──────────┤
       │                       │   token₁ ─── relay ───→      │          │
       │                       │   token₂ ──── relay ──────────→         │
       │                       │   ...                                   │
       │                       │   token_N ──── relay ───────────────────→
       │                       │                                         │
User:  [waiting...             waiting...  │ sees token₁! tokens keep flowing...]
                                           ↑
                                   first token at ~3s
```

The total wall-clock time is similar (~5s). But the **perceived** experience is dramatically different:

| Metric | Current (batch) | WebSocket relay |
|--------|----------------|-----------------|
| **Time to first token** | ~5 seconds | ~3 seconds |
| **User sees progress** | No (blank screen for 5s) | Yes (text appears at 3s) |
| **True streaming** | No (simulated after all tokens arrive) | Yes (real tokens from GPU) |
| **Can cancel mid-generation** | No (GPU already finished) | Yes (stop the function) |
| **Feels like ChatGPT** | Somewhat (simulated) | Yes (real progressive display) |

This matters for user experience. Research on perceived latency shows that users tolerate longer total waits when they see progressive feedback. A 5-second blank screen feels much slower than 3 seconds of waiting followed by 2 seconds of text appearing. This is the same reason ChatGPT streams tokens — the total generation time is the same, but the perceived responsiveness is much better.

#### Concrete Implementation: The WebSocket Relay

The relay server is remarkably simple — under 50 lines of Python:

```python
# relay_server.py — deploy on any public server
import asyncio
import websockets
import json

# Active channels: task_id → set of connected WebSocket clients
channels = {}

async def handler(ws):
    """
    Handle a WebSocket connection from either a Lakeshore worker or a STREAM client.

    Protocol:
    1. First message is the task_id (shared secret from Globus Compute)
    2. All subsequent messages are forwarded to all other clients in the same channel
    """
    # First message identifies which task this connection belongs to
    task_id = await ws.recv()

    # Join the channel for this task
    channels.setdefault(task_id, set()).add(ws)

    try:
        async for message in ws:
            # Forward to all other clients in this channel
            for client in channels.get(task_id, set()):
                if client != ws:
                    await client.send(message)
    finally:
        # Clean up when connection closes
        channels.get(task_id, set()).discard(ws)
        if not channels.get(task_id):
            del channels[task_id]

# Start the relay server
asyncio.run(websockets.serve(handler, "0.0.0.0", 8765))
```

The remote function (running on Lakeshore) streams tokens to the relay:

```python
def remote_vllm_inference_streaming(vllm_url, model, messages, temperature,
                                     max_tokens, task_id, relay_url):
    """
    Runs on Lakeshore via Globus Compute.
    Calls vLLM with streaming enabled and forwards each token
    to the WebSocket relay for real-time delivery to the client.
    """
    import requests
    import json
    import websocket  # websocket-client library (synchronous)

    # Connect to the relay and identify ourselves with the task_id
    ws = websocket.create_connection(relay_url)
    ws.send(task_id)

    # Call vLLM with stream=True — it returns Server-Sent Events
    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,  # This is the key difference from current approach
        },
        stream=True,
        timeout=60,
    )

    # Forward each SSE event through the relay
    for line in response.iter_lines():
        if not line:
            continue
        line_text = line.decode("utf-8")
        if line_text.startswith("data: "):
            ws.send(line_text)  # Forward the SSE chunk as-is

    ws.close()
    return {"status": "streamed"}
```

On the client side, STREAM connects to the same relay and yields tokens as SSE events:

```python
async def _forward_lakeshore_streaming(task_id, relay_url):
    """
    Connect to the WebSocket relay and yield SSE chunks as they arrive
    from Lakeshore. Each chunk is a real token from vLLM's streaming output.
    """
    import websockets

    async with websockets.connect(relay_url) as ws:
        # Join the channel for our task
        await ws.send(task_id)

        # Receive tokens as they arrive from Lakeshore
        async for message in ws:
            yield message  # This is an SSE "data: {...}" line
```

The orchestration would work as follows:

1. **Submit via Globus Compute** (control plane): `executor.submit(remote_vllm_inference_streaming, ...)` — this takes ~1-2 seconds for AMQP routing.
2. **Immediately connect to relay** (data plane): While Globus routes the task, the client opens a WebSocket to the relay using the same `task_id`.
3. **Tokens flow through relay**: Once the remote function starts on Lakeshore, it connects to the relay and begins streaming. The client receives tokens within ~50ms of generation.
4. **Globus returns summary**: When the function completes, Globus returns `{"status": "streamed"}` through the normal channel. STREAM uses this to confirm the task finished (or to detect errors if the relay connection dropped).

#### Comparison: Current vs. Hybrid SSH vs. WebSocket Relay

| Metric | Current (Globus Batch) | Hybrid SSH Tunnel | WebSocket Relay |
|--------|----------------------|-------------------|-----------------|
| **First token latency** | ~5 seconds | ~2-3 seconds | ~3 seconds |
| **Per-token latency** | N/A (all at once) | Real-time (~5ms) | Near-real-time (~50ms) |
| **True streaming** | No (simulated) | Yes | Yes |
| **Works off-campus** | Yes | No (needs SSH access) | Yes |
| **Additional infrastructure** | None | SSH keys configured | Relay server (~$5/mo) |
| **Implementation complexity** | Current (done) | Medium | Medium |
| **Cancel mid-generation** | No | Yes | Yes |
| **Firewall transparent** | Yes | Partial (needs SSH) | Yes |
| **No user setup required** | Yes | No (SSH keys) | Yes |

#### Research Contribution Potential

The WebSocket relay approach addresses a gap in the Globus Compute ecosystem: **interactive FaaS workloads**. Most Globus Compute use cases are batch-oriented (submit a job, get a result minutes later). AI inference is fundamentally different — users expect real-time, token-by-token responses.

The pattern of separating the control plane (Globus Compute for auth/submission) from the data plane (WebSocket relay for streaming) is generalizable to any research application that needs interactive access to HPC resources:

- Real-time data visualization from running simulations
- Interactive scientific computing notebooks connected to HPC backends
- Streaming sensor data processing on GPU clusters
- Any application where the user is waiting for progressive results

The control plane / data plane separation could also be proposed as a design pattern to the Globus team at the University of Chicago and Argonne National Laboratory, potentially motivating native streaming support in future SDK versions.

### 16.4 Recommended Roadmap

Based on feasibility, user impact, and the goal of seamless UX:

| Phase | Approach | Effort | Impact | Target Users |
|-------|----------|--------|--------|--------------|
| **Phase 1** | WebSocket relay (side-channel streaming) | Medium (~1 developer) | High (true streaming, 3s first token) | All users, any network |
| **Phase 2** | SSH tunnel fast-path (optional optimization) | Low-Medium | Medium (2s first token for SSH users) | Campus users with SSH keys |
| **Phase 3** | Propose native Globus streaming extension | Collaborative (Globus team) | Very High (ecosystem-wide benefit) | All Globus Compute users |

**Phase 1** is the priority. It delivers true streaming to all users regardless of network location, requires no user setup (no SSH keys, no VPN), and can be built by a single developer. The relay server is under 50 lines of code and costs ~$5/month to host.

**Phase 2** is an optional optimization for users who already have SSH access to Lakeshore. It reduces first-token latency from ~3s to ~2s by bypassing Globus entirely for the data path. This is a nice-to-have but not essential — the relay already provides a good experience for everyone.

**Phase 3** is a research contribution. If the Globus team added native streaming support to their SDK, the relay server would become unnecessary — tokens could flow through Globus's own AMQP infrastructure. This is the long-term ideal, but it requires collaboration with the Globus team and shouldn't block Phase 1.

---

## Summary

STREAM's Lakeshore connectivity demonstrates how a research middleware system can bridge the gap between user-facing desktop/web applications and HPC resources behind campus firewalls. The key architectural decisions are:

1. **Globus Compute as the transport layer** — eliminates the need for SSH tunnels, VPN, or firewall modifications while providing secure, authenticated access to HPC GPUs.

2. **Dual-mode architecture** — the same `GlobusComputeClient` code runs in both Docker (via the proxy container) and desktop (via direct function calls), with the mode difference handled at the routing layer.

3. **Persistent AMQP connections** — reduce per-request overhead from ~1.5s to ~0s by reusing the Executor across requests.

4. **`exec()`-based remote functions** — solve the PyInstaller serialization incompatibility that would otherwise prevent desktop deployment.

5. **Multi-model architecture** — five models on separate vLLM instances (ports 8000–8004) with per-model URL resolution, context limits, and health checks, all configured through `LAKESHORE_MODELS` in `config.py`.

6. **Per-model health checks** — real 1-token inference tests through Globus Compute verify each model's vLLM instance is operational, not just that authentication works.

7. **Simulated streaming** — converts Globus Compute's batch-style responses into SSE streams, providing a consistent ChatGPT-like experience across all tiers.

8. **Centralized configuration** — `LAKESHORE_MODELS` and `MODEL_CONTEXT_LIMITS` in `config.py` serve as the single source of truth for model mappings, port assignments, and token limits across proxy, desktop direct calls, and context validation.

The irreducible ~5-second latency for Lakeshore requests is inherent to the Globus Compute FaaS architecture (multi-hop AMQP routing through cloud infrastructure) and represents the trade-off for secure, firewall-transparent HPC access without SSH or VPN infrastructure. The proposed WebSocket relay approach (Section 16) offers a practical path to true streaming with ~3-second first-token latency while preserving the seamless, zero-setup user experience that Globus Compute provides.
