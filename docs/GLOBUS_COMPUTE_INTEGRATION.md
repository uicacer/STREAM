# STREAM + Globus Compute Integration: A Technical Deep Dive
---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What is Globus Compute?](#2-what-is-globus-compute)
3. [Why Globus Compute for STREAM?](#3-why-globus-compute-for-stream)
4. [Architecture Overview](#4-architecture-overview)
5. [OAuth 2.0 Authentication Deep Dive](#5-oauth-20-authentication-deep-dive)
6. [Zero-Friction Browser Authentication Implementation](#6-zero-friction-browser-authentication-implementation)
7. [Request Flow: From User Query to HPC Response](#7-request-flow-from-user-query-to-hpc-response)
8. [Code Walkthrough](#8-code-walkthrough)
9. [Performance & Latency Analysis](#9-performance--latency-analysis)
10. [Security Considerations](#10-security-considerations)
11. [Troubleshooting Guide](#11-troubleshooting-guide)
12. [Q&A](#12-conference-qa-preparation)

---

## 1. Executive Summary

STREAM (Smart Tiered Routing for Efficient AI at Marquette) is a multi-tier AI gateway that intelligently routes user queries to the most appropriate computational resource:

- **Local Tier**: Ollama running on local Docker (free, fast, limited capability)
- **Lakeshore Tier**: Campus HPC GPU cluster via Globus Compute (low cost, powerful)
- **Cloud Tier**: Commercial APIs like Claude/GPT (paid, most capable)

The **Globus Compute integration** enables STREAM to leverage Marquette's Lakeshore HPC cluster for AI inference without requiring users to:
- SSH into the cluster
- Write SLURM job scripts
- Manage job queues
- Understand HPC infrastructure

Instead, users simply type a question in a chat interface, and STREAM handles everything else.

---

## 2. What is Globus Compute?

### 2.1 The Problem Globus Compute Solves

Traditional HPC access requires:
```
User → SSH → Login Node → Write SLURM Script → Submit Job → Wait → Check Output → Download Results
```

This is complex, time-consuming, and requires HPC expertise.

### 2.2 Globus Compute's Solution

Globus Compute (formerly funcX) is a **Function-as-a-Service (FaaS) platform** for research computing. It allows you to:

1. **Define a Python function** on your local machine
2. **Submit it for execution** on a remote HPC cluster
3. **Get results back** as if it ran locally

```
User → API Call → Globus Compute → HPC Cluster → Results → User
```

### 2.3 Key Components

| Component | Description |
|-----------|-------------|
| **Globus Compute Endpoint** | A daemon running on the HPC cluster that receives and executes tasks |
| **Globus Compute SDK** | Python library for submitting tasks to endpoints |
| **Globus Auth** | OAuth 2.0 identity provider for authentication |
| **funcX Web Service** | Cloud service that routes tasks between clients and endpoints |

### 2.4 How It Works (Simplified)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Your Code     │────▶│  Globus Compute  │────▶│   HPC Endpoint      │
│   (Client)      │◀────│  Cloud Service   │◀────│   (Lakeshore)       │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
      │                        │                         │
      │ Submit function        │ Route to endpoint       │ Execute on GPU
      │ + arguments            │ Queue management        │ Run vLLM inference
      │                        │ Result delivery         │
```

---

## 3. Why Globus Compute for STREAM?

### 3.1 The Challenge

Marquette's Lakeshore HPC cluster has NVIDIA GPUs capable of running large language models via vLLM. However:

- vLLM runs on compute nodes, not login nodes
- Compute nodes are behind a firewall (no direct external access)
- Traditional access requires VPN + SSH + job submission

### 3.2 The Solution

Globus Compute acts as a **secure bridge**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INTERNET                                      │
│                                                                       │
│  ┌─────────────┐         ┌─────────────────────┐                     │
│  │   STREAM    │◀───────▶│  Globus Compute     │                     │
│  │   (Docker)  │  HTTPS  │  Cloud Service      │                     │
│  └─────────────┘         └──────────┬──────────┘                     │
│                                     │                                 │
└─────────────────────────────────────│─────────────────────────────────┘
                                      │ Outbound HTTPS
                                      │ (initiated by endpoint)
┌─────────────────────────────────────│─────────────────────────────────┐
│  LAKESHORE HPC CLUSTER              │                                 │
│  (Behind Firewall)                  ▼                                 │
│                          ┌─────────────────────┐                     │
│                          │  Globus Compute     │                     │
│                          │  Endpoint Daemon    │                     │
│                          └──────────┬──────────┘                     │
│                                     │                                 │
│                          ┌──────────▼──────────┐                     │
│                          │   vLLM Server       │                     │
│                          │   (GPU Node)        │                     │
│                          └─────────────────────┘                     │
└───────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The endpoint initiates an *outbound* connection to Globus Compute's cloud service. This works even behind firewalls because:
- Outbound HTTPS (port 443) is typically allowed
- The connection is persistent (WebSocket-like)
- No inbound ports need to be opened

### 3.3 Benefits

| Benefit | Explanation |
|---------|-------------|
| **No VPN Required** | Globus Compute handles secure routing |
| **No SSH Required** | API-based interaction, not shell access |
| **No SLURM Knowledge** | The endpoint manager handles job scheduling |
| **Firewall Friendly** | Works through NAT and corporate firewalls |
| **Persistent Auth** | One-time browser login, tokens cached locally |

---

## 4. Architecture Overview

### 4.1 STREAM's Three-Tier Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              STREAM SYSTEM                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐    │
│  │    Streamlit    │───▶│    Middleware   │───▶│    Tier Routing     │    │
│  │    Frontend     │    │    (FastAPI)    │    │    Decision Logic   │    │
│  └─────────────────┘    └─────────────────┘    └──────────┬──────────┘    │
│                                                           │               │
│                         ┌─────────────────────────────────┼───────────────┤
│                         │                                 │               │
│                         ▼                                 ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐    │
│  │   LOCAL TIER    │    │  LAKESHORE TIER │    │    CLOUD TIER       │    │
│  │   (Ollama)      │    │  (Globus Proxy) │    │    (LiteLLM)        │    │
│  │   Port 11434    │    │   Port 8001     │    │    Port 4000        │    │
│  └─────────────────┘    └────────┬────────┘    └─────────────────────┘    │
│                                  │                                        │
└──────────────────────────────────│────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Globus Compute Service     │
                    │   (globus-compute.org)       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Lakeshore HPC Endpoint     │
                    │   + vLLM GPU Inference       │
                    └──────────────────────────────┘
```

### 4.2 Docker Services

STREAM runs as a set of Docker containers:

| Service | Port | Purpose |
|---------|------|---------|
| `streamlit-frontend` | 8501 | Web chat interface |
| `stream-middleware` | 5000 | API gateway, routing logic |
| `lakeshore-proxy` | 8001 | Globus Compute client |
| `ollama` | 11434 | Local LLM inference |
| `litellm` | 4000 | Cloud API gateway |

### 4.3 The Lakeshore Proxy

The `lakeshore-proxy` service is a FastAPI application that:
1. Receives inference requests from the middleware
2. Authenticates with Globus Compute (using cached tokens)
3. Submits tasks to the Lakeshore endpoint
4. Returns results to the middleware

**File**: `stream/proxy/app.py`

---

## 5. OAuth 2.0 Authentication Deep Dive

### 5.1 What is OAuth 2.0?

OAuth 2.0 is an **authorization framework** that allows applications to obtain limited access to user accounts on external services. Instead of sharing passwords, users grant tokens that can be revoked.

### 5.2 OAuth 2.0 Terminology

| Term | Meaning in STREAM Context |
|------|---------------------------|
| **Resource Owner** | You (the user) |
| **Client** | STREAM application |
| **Authorization Server** | Globus Auth (auth.globus.org) |
| **Resource Server** | Globus Compute API |
| **Access Token** | Short-lived credential for API calls |
| **Refresh Token** | Long-lived credential to get new access tokens |
| **Scope** | Permissions requested (e.g., "run functions on endpoints") |

### 5.3 The OAuth 2.0 Authorization Code Flow

This is the flow used by Globus Compute:

```
┌──────────┐                               ┌──────────────┐                    ┌──────────────┐
│          │                               │              │                    │              │
│   User   │                               │    STREAM    │                    │  Globus Auth │
│          │                               │              │                    │              │
└────┬─────┘                               └──────┬───────┘                    └──────┬───────┘
     │                                            │                                   │
     │  1. User triggers action needing auth      │                                   │
     │  ────────────────────────────────────────▶ │                                   │
     │                                            │                                   │
     │                                            │  2. Generate auth URL with:       │
     │                                            │     - client_id                   │
     │                                            │     - redirect_uri (localhost)    │
     │                                            │     - scope (compute permissions) │
     │                                            │     - state (CSRF protection)     │
     │                                            │  ──────────────────────────────▶  │
     │                                            │                                   │
     │  3. Browser opens to Globus login page     │                                   │
     │  ◀──────────────────────────────────────── │                                   │
     │                                            │                                   │
     │  4. User logs in with Globus credentials   │                                   │
     │  ──────────────────────────────────────────────────────────────────────────▶  │
     │                                            │                                   │
     │  5. User approves requested permissions    │                                   │
     │  ──────────────────────────────────────────────────────────────────────────▶  │
     │                                            │                                   │
     │  6. Globus redirects to localhost with     │                                   │
     │     authorization code                     │                                   │
     │  ◀──────────────────────────────────────────────────────────────────────────  │
     │                                            │                                   │
     │  7. Local server captures the code         │                                   │
     │  ────────────────────────────────────────▶ │                                   │
     │                                            │                                   │
     │                                            │  8. Exchange code for tokens      │
     │                                            │  ──────────────────────────────▶  │
     │                                            │                                   │
     │                                            │  9. Receive access + refresh      │
     │                                            │     tokens                        │
     │                                            │  ◀──────────────────────────────  │
     │                                            │                                   │
     │                                            │ 10. Store tokens in               │
     │                                            │     ~/.globus_compute/storage.db  │
     │                                            │                                   │
     │ 11. Authentication complete!               │                                   │
     │  ◀──────────────────────────────────────── │                                   │
```

### 5.4 Globus-Specific Scopes

Globus Compute requires multiple scopes for full functionality:

```python
# Required scopes for Globus Compute
SCOPES = [
    "openid",                                    # Basic identity
    "profile",                                   # User profile info
    "email",                                     # Email address
    "urn:globus:auth:scope:compute.api.globus.org:all",  # Compute API access
    "urn:globus:auth:scope:funcx.globus.org:all",        # Legacy funcX access
]
```

### 5.5 Token Storage

Tokens are stored in an SQLite database:

```
~/.globus_compute/storage.db
```

This file contains:
- Access tokens (expire in ~24 hours)
- Refresh tokens (long-lived, used to get new access tokens)
- Token metadata (scopes, expiration times)

---

## 6. Zero-Friction Browser Authentication Implementation

### 6.1 The Problem with Traditional CLI Auth

The default Globus Compute authentication requires:
1. User runs a command
2. A URL is printed to the terminal
3. User copies URL to browser
4. User logs in
5. User copies authorization code from browser
6. User pastes code back into terminal

This is error-prone and tedious.

### 6.2 Our Solution: LocalServerLoginFlowManager

The Globus SDK provides `LocalServerLoginFlowManager` which:
1. Starts a temporary HTTP server on localhost
2. Opens the browser automatically
3. Receives the OAuth callback automatically
4. Exchanges the code for tokens automatically

**No copy-pasting required!**

### 6.3 Implementation Details

**File**: `stream/middleware/core/globus_auth.py`

```python
from globus_compute_sdk.sdk.auth.globus_app import get_globus_app
from globus_sdk.login_flows import LocalServerLoginFlowManager

def authenticate_with_browser_callback() -> Tuple[bool, str]:
    """
    Zero-friction OAuth authentication using automatic browser callback.
    """
    # Get the shared GlobusApp instance
    app = get_globus_app()

    # Replace the default CommandLineLoginFlowManager with LocalServerLoginFlowManager
    app._login_flow_manager = LocalServerLoginFlowManager(
        app._login_client,
        request_refresh_tokens=True  # Important: enables persistent auth
    )

    # Creating a Client triggers the login flow if needed
    # The SDK handles everything: browser opening, local server, token exchange
    client = Client(app=app)

    return True, "Authentication successful!"
```

### 6.4 How LocalServerLoginFlowManager Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. Start local HTTP server on random high port (e.g., 54321)              │
│                                                                             │
│  2. Generate authorization URL:                                             │
│     https://auth.globus.org/v2/oauth2/authorize?                           │
│       client_id=...&                                                        │
│       redirect_uri=http://localhost:54321/callback&                         │
│       scope=...&                                                            │
│       state=random_csrf_token                                               │
│                                                                             │
│  3. Open browser to authorization URL                                       │
│                                                                             │
│  4. User logs in at auth.globus.org                                        │
│                                                                             │
│  5. Globus redirects browser to:                                            │
│     http://localhost:54321/callback?code=AUTH_CODE&state=random_csrf_token │
│                                                                             │
│  6. Local server receives the request, extracts AUTH_CODE                   │
│                                                                             │
│  7. SDK exchanges AUTH_CODE for access + refresh tokens                     │
│                                                                             │
│  8. Tokens saved to ~/.globus_compute/storage.db                           │
│                                                                             │
│  9. Local server shuts down                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Integration with Streamlit UI

When a user's request routes to Lakeshore and authentication is needed:

**File**: `frontends/streamlit/streamlit_app.py`

```python
# Detect auth error in streaming response
if stream_meta.get("auth_required"):
    # Set up auth flow state
    st.session_state.auth_flow_step = "vpn_warning"
    st.session_state.auth_pending_message = user_message
    st.rerun()

# In the auth flow handler:
if st.session_state.auth_flow_step == "authenticating":
    with st.spinner("Authenticating with Globus Compute..."):
        success, message = authenticate_globus_compute()

    if success:
        # Retry the original question
        st.session_state.pending_query = user_message
        st.rerun()
```

---

## 7. Request Flow: From User Query to HPC Response

### 7.1 Complete Request Flow

Let's trace a request from the user typing a question to receiving a response from Lakeshore:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: User Input                                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  User types: "Explain the Transformer architecture in deep learning"      │
│                                                                            │
│  Streamlit captures input via st.chat_input()                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: SDK Sends to Middleware                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ChatHandler.chat() makes HTTP POST to middleware:                         │
│                                                                            │
│  POST http://middleware:5000/v1/chat/completions                           │
│  {                                                                         │
│    "model": "auto",                                                        │
│    "messages": [{"role": "user", "content": "Explain the Transformer..."}],│
│    "temperature": 0.7                                                      │
│  }                                                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Middleware Analyzes Query Complexity                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  complexity_judge.py analyzes the query:                                   │
│  - Keywords detected: "architecture", "deep learning", "explain"           │
│  - Complexity: HIGH (requires detailed technical explanation)              │
│                                                                            │
│  query_router.py determines tier:                                          │
│  - HIGH complexity → Prefer LAKESHORE or CLOUD                             │
│  - Check tier health → LAKESHORE available                                 │
│  - Decision: Route to LAKESHORE                                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Middleware Forwards to Lakeshore Proxy                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  streaming.py sends request to lakeshore-proxy:                            │
│                                                                            │
│  POST http://lakeshore-proxy:8001/v1/chat/completions                      │
│  {                                                                         │
│    "model": "Qwen/Qwen2.5-1.5B-Instruct",                                  │
│    "messages": [...],                                                      │
│    "max_tokens": 2048                                                      │
│  }                                                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Proxy Submits to Globus Compute                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  globus_compute_client.py:                                                 │
│                                                                            │
│  1. Check authentication status                                            │
│  2. Create Executor with endpoint_id                                       │
│  3. Define inference function:                                             │
│                                                                            │
│     def run_vllm_inference(messages, temperature, max_tokens, model):      │
│         # This runs ON THE HPC CLUSTER                                     │
│         response = requests.post(                                          │
│             "http://localhost:8000/v1/chat/completions",  # vLLM server    │
│             json={"model": model, "messages": messages, ...}               │
│         )                                                                  │
│         return response.json()                                             │
│                                                                            │
│  4. Submit function to Globus Compute:                                     │
│     future = executor.submit(run_vllm_inference, messages, temp, ...)      │
│                                                                            │
│  5. Wait for result:                                                       │
│     result = future.result(timeout=120)                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Globus Compute Routes to Endpoint                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Globus Compute Cloud Service:                                             │
│                                                                            │
│  1. Receives task submission                                               │
│  2. Looks up endpoint by UUID                                              │
│  3. Serializes function + arguments (using dill/pickle)                    │
│  4. Queues task for endpoint                                               │
│  5. Sends task to endpoint over persistent connection                      │
│                                                                            │
│  Lakeshore Endpoint:                                                       │
│                                                                            │
│  1. Receives serialized task                                               │
│  2. Deserializes function + arguments                                      │
│  3. Executes function in worker process                                    │
│  4. Sends result back to cloud service                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: vLLM Generates Response                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  On Lakeshore GPU Node:                                                    │
│                                                                            │
│  1. vLLM server receives HTTP request                                      │
│  2. Tokenizes input messages                                               │
│  3. Runs transformer inference on GPU                                      │
│  4. Generates tokens autoregressively                                      │
│  5. Returns response:                                                      │
│                                                                            │
│  {                                                                         │
│    "choices": [{                                                           │
│      "message": {                                                          │
│        "content": "The Transformer architecture, introduced in the         │
│                    paper 'Attention Is All You Need' (2017), is..."        │
│      }                                                                     │
│    }]                                                                      │
│  }                                                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: Response Flows Back                                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  vLLM → Globus Endpoint → Globus Cloud → Proxy → Middleware → Streamlit   │
│                                                                            │
│  Each layer adds metadata:                                                 │
│  - Proxy: execution time, endpoint status                                  │
│  - Middleware: tier used, cost estimate, correlation ID                    │
│  - Streamlit: display formatting, routing info badge                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: User Sees Response                                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 🏫 LAKESHORE · vLLM                     ⏱️ 3.45s    💰 Low Cost │       │
│  │                                                                 │       │
│  │ The Transformer architecture, introduced in the landmark paper  │       │
│  │ "Attention Is All You Need" by Vaswani et al. (2017), is a     │       │
│  │ neural network architecture that revolutionized NLP...          │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Code Walkthrough

### 8.1 Key Files and Their Purposes

```
STREAM/
├── frontends/streamlit/
│   └── streamlit_app.py          # Web UI, auth flow UI, chat interface
│
├── stream/
│   ├── middleware/
│   │   ├── core/
│   │   │   ├── globus_auth.py    # Zero-friction OAuth implementation
│   │   │   ├── globus_compute_client.py  # Task submission to Globus
│   │   │   ├── query_router.py   # Tier selection logic
│   │   │   ├── tier_health.py    # Health checks for each tier
│   │   │   └── streaming.py      # SSE streaming response handler
│   │   │
│   │   └── routes/
│   │       └── chat.py           # /chat/completions endpoint
│   │
│   ├── proxy/
│   │   └── app.py                # Lakeshore proxy FastAPI service
│   │
│   └── sdk/python/
│       └── chat_handler.py       # Python SDK for frontends
│
└── docker-compose.yml            # Service orchestration
```

### 8.2 globus_auth.py - Zero-Friction Authentication

```python
"""
Key components of the authentication module:
"""

from globus_compute_sdk.sdk.client import Client
from globus_compute_sdk.sdk.auth.globus_app import get_globus_app
from globus_sdk.login_flows import LocalServerLoginFlowManager

def authenticate_with_browser_callback() -> Tuple[bool, str]:
    """
    The magic happens here:

    1. get_globus_app() returns the shared GlobusApp instance
       - This is a singleton that manages authentication state
       - It's shared across all Globus SDK components

    2. We replace the login_flow_manager:
       - Default: CommandLineLoginFlowManager (prints URL, asks for code)
       - Ours: LocalServerLoginFlowManager (opens browser, captures callback)

    3. Creating a Client triggers login if needed:
       - Client checks if tokens exist and are valid
       - If not, it calls the login_flow_manager
       - Our manager opens browser and handles OAuth automatically
    """
    app = get_globus_app()

    app._login_flow_manager = LocalServerLoginFlowManager(
        app._login_client,
        request_refresh_tokens=True  # Critical for persistent auth!
    )

    client = Client(app=app)

    return True, "Authentication successful!"


def is_authenticated() -> bool:
    """
    Quick check without triggering login flow.

    app.login_required() checks:
    - Do tokens exist?
    - Are they expired?
    - Do they have all required scopes?

    Returns True if any of these fail.
    """
    app = get_globus_app()
    return not app.login_required()
```

### 8.3 globus_compute_client.py - Task Submission

```python
"""
Key components of the Globus Compute client:
"""

from globus_compute_sdk import Executor

class GlobusComputeClient:
    def __init__(self):
        self.endpoint_id = GLOBUS_ENDPOINT_ID
        self.vllm_url = VLLM_SERVER_URL
        self._executor = None

    async def submit_inference(self, messages, temperature, max_tokens, model):
        """
        Submit an inference task to the HPC endpoint.

        The function we submit runs ON THE REMOTE MACHINE:
        - It has access to the vLLM server running on the GPU node
        - It can import libraries installed on the endpoint
        - It cannot access our local machine
        """

        # Define the function that will run on the HPC cluster
        def run_vllm_inference(messages, temperature, max_tokens, model, vllm_url):
            """This entire function is serialized and sent to the endpoint."""
            import requests

            response = requests.post(
                f"{vllm_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120
            )
            return response.json()

        # Submit to Globus Compute
        with Executor(endpoint_id=self.endpoint_id) as executor:
            future = executor.submit(
                run_vllm_inference,
                messages,
                temperature,
                max_tokens,
                model,
                self.vllm_url
            )

            # Wait for result (with timeout)
            result = future.result(timeout=120)

        return result
```

### 8.4 query_router.py - Intelligent Routing

```python
"""
Routing logic determines which tier handles each query.
"""

def get_tier_for_query(query: str, user_preference: str = "auto") -> str:
    """
    Routing decision tree:

    1. User explicit selection (not "auto"):
       - If tier available → use it
       - If tier unavailable → raise error (don't silently fallback)

    2. Auto mode:
       a. Analyze query complexity (LLM judge or keyword matching)
       b. Map complexity to preferred tier:
          - LOW → local (fast, free)
          - MEDIUM → lakeshore (balanced)
          - HIGH → cloud (most capable)
       c. Apply fallback if preferred tier unavailable
    """

    # Explicit user selection - respect it strictly
    if user_preference in ["local", "lakeshore", "cloud"]:
        if is_tier_available(user_preference):
            return user_preference
        else:
            raise Exception(f"{user_preference.upper()} tier is unavailable")

    # Auto mode - analyze and route
    complexity = judge_complexity_with_keywords(query)

    tier_map = {"low": "local", "medium": "lakeshore", "high": "cloud"}
    preferred_tier = tier_map[complexity]

    # Get available tier with fallback
    tier, reason = get_tier_with_fallback(preferred_tier, complexity)

    return tier
```

### 8.5 Docker Compose - Service Orchestration

```yaml
# Key parts of docker-compose.yml

services:
  lakeshore-proxy:
    build: ./stream/proxy
    ports:
      - "8001:8001"
    environment:
      - GLOBUS_COMPUTE_ENDPOINT_ID=${GLOBUS_COMPUTE_ENDPOINT_ID}
      - VLLM_SERVER_URL=${VLLM_SERVER_URL}
    volumes:
      # Mount Globus credentials from host
      # This allows the container to use tokens from browser auth
      - ${HOME}/.globus_compute:/root/.globus_compute:rw
```

---

## 9. Performance & Latency Analysis

Understanding latency is crucial when designing systems that span multiple computational tiers. This section breaks down where time is spent in each tier and explains the trade-offs.

### 9.1 Latency Comparison by Tier

| Tier | Typical Latency | Range | Primary Factor |
|------|-----------------|-------|----------------|
| **Local** | 0.5-2s | 0.3-5s | Model size, CPU/GPU |
| **Lakeshore** | 3-10s | 2-15s | Globus Compute overhead |
| **Cloud** | 1-3s | 0.5-5s | Network + API processing |

### 9.2 Lakeshore Latency Breakdown

The 3-10 second latency for Lakeshore tier is primarily due to Globus Compute's function-as-a-service architecture:

```
User Request
    ↓ ~50ms      ← STREAM Middleware processing
STREAM Middleware
    ↓ ~50ms      ← LiteLLM routing
LiteLLM Gateway
    ↓ ~50ms      ← Proxy forwarding
Lakeshore Proxy
    ↓ ~500-1000ms ← Task serialization + API submission
Globus Compute API (Cloud)
    ↓ ~500-1000ms ← Task routing to endpoint
HPC Endpoint (Lakeshore)
    ↓ ~1-3s       ← vLLM inference (actual model work)
Response back through all layers
    ↓ ~500-1000ms ← Result retrieval + deserialization
User sees response
─────────────────
Total: 3-7 seconds typical
```

**Key Insight**: The majority of latency (~60-70%) comes from Globus Compute's task submission and retrieval overhead, not the actual model inference.

### 9.3 Why Globus Compute Has This Overhead

Globus Compute is optimized for **reliability and security**, not low-latency interactive use:

| Design Choice | Benefit | Latency Cost |
|---------------|---------|--------------|
| Centralized task routing | Works through firewalls | +500ms |
| Task serialization | Language-agnostic execution | +100ms |
| Result persistence | Fault tolerance | +200ms |
| OAuth token validation | Secure authentication | +100ms |
| Endpoint polling | No inbound ports needed | +variable |

### 9.4 The Trade-off: Accessibility vs. Speed

| Approach | Latency | Requirements |
|----------|---------|--------------|
| **Globus Compute (current)** | 3-10s | Browser only, no VPN |
| **Direct SSH tunnel** | 1-2s | VPN + SSH + port forwarding |
| **Direct API** | 0.5-1s | Firewall rules, public IP |

STREAM chose Globus Compute because:
1. **Zero infrastructure burden** - Users don't need VPN or SSH
2. **Works from anywhere** - Coffee shop, home, conference WiFi
3. **Secure by default** - No firewall holes, OAuth authentication
4. **Maintained by Globus** - We don't manage endpoint security

### 9.5 Optimization Strategies

To minimize perceived latency while using Globus Compute:

#### 1. Smart Routing (Implemented)
Route simple queries to Local tier (sub-second) and reserve Lakeshore for complex queries that benefit from larger models.

```python
# Example routing logic
if complexity == "low":
    return "local"   # 0.5-2s
elif complexity == "high":
    return "lakeshore"  # 3-10s, but better quality
```

#### 2. User Feedback (Implemented)
Show engaging progress messages so users don't feel the wait:
- "🏔️ Connecting to Marquette's Lakeshore HPC..."
- "💡 Did you know? vLLM serves models with continuous batching!"

#### 3. Future Optimizations (Potential)
- **Persistent connections**: Keep endpoint "warm" with keepalive tasks
- **Batching**: Group multiple requests to amortize overhead
- **Caching**: Cache responses for identical queries
- **Hybrid mode**: Offer SSH tunnel option for power users

### 9.6 When Lakeshore Latency is Worth It

Despite the overhead, Lakeshore tier is valuable when:

| Scenario | Why Lakeshore |
|----------|---------------|
| Complex reasoning tasks | Larger model = better quality |
| Cost-sensitive users | Free vs. $0.01-0.10 per query |
| Research workloads | Access to institution's GPUs |
| Privacy requirements | Data stays on campus network |

### 9.7 Latency Monitoring

STREAM tracks latency metrics for optimization:

```python
# In MetricsTracker
tracker.record_first_token()   # Time to first token (TTFT)
tracker.record_completion()     # Total request time

# Logs show:
# [correlation_id] Stream completed: cost=$0.00, duration=5.2s
```

---

## 10. Security Considerations

### 10.1 Token Security

| Aspect | Implementation |
|--------|----------------|
| Storage | SQLite database with file permissions (0600) |
| Location | `~/.globus_compute/storage.db` |
| Encryption | Tokens encrypted at rest by Globus SDK |
| Expiration | Access tokens: ~24 hours, auto-refreshed |

### 10.2 Network Security

| Layer | Protection |
|-------|------------|
| STREAM ↔ Globus | HTTPS/TLS 1.3 |
| Globus ↔ Endpoint | HTTPS/TLS, authenticated connection |
| Endpoint ↔ vLLM | Localhost only (127.0.0.1) |

### 10.3 Authorization Model

```
User → Globus Auth → Access Token → Globus Compute API → Endpoint
         │                │
         │                └── Token contains:
         │                    - User identity
         │                    - Granted scopes
         │                    - Expiration time
         │
         └── Identity linked to:
             - Institutional login (SSO)
             - Globus ID
             - ORCID
```

### 10.4 What Users CAN'T Do

- Access other users' tasks
- Execute arbitrary code on endpoints they don't own
- Bypass endpoint access controls
- Access the HPC filesystem directly

---

## 11. Troubleshooting Guide

### 11.1 Authentication Issues

**Symptom**: "Authentication required" error
```
Solution:
1. Delete cached tokens: rm -rf ~/.globus_compute
2. Restart STREAM services: docker-compose restart
3. Try again - browser should open for fresh login
```

**Symptom**: Browser doesn't open (SSH/headless environment)
```
Solution:
1. Run authentication on a machine with a browser:
   python -c "from globus_compute_sdk import Client; Client()"
2. Copy ~/.globus_compute/storage.db to your server
3. Restart services
```

**Symptom**: Browser shows "ERR_SOCKET_NOT_CONNECTED" or "localhost refused to connect" after Globus login
```
Cause: VPN interference with localhost OAuth callback

When you authenticate, the LocalServerLoginFlowManager starts a temporary HTTP server
on localhost (e.g., http://127.0.0.1:56390). After you approve access in your browser,
Globus redirects back to this localhost URL. However, some VPNs intercept ALL browser
traffic - including localhost requests - and route them through the VPN tunnel.

Since the callback server is running on your actual localhost (not through the VPN),
the browser can't connect because its "localhost" traffic is being routed elsewhere.

Solution:
1. Disconnect your VPN before clicking "Authenticate Now"
2. Complete the Globus authentication in your browser
3. Once you see "Authentication successful", reconnect your VPN
4. Your tokens are now cached - VPN won't interfere with normal usage

Note: This is a one-time issue. Once authenticated, tokens are cached locally and
the VPN won't affect subsequent Lakeshore requests (which use HTTPS to Globus APIs).

Affected VPNs: This commonly affects VPNs that use full-tunnel mode or have
"split tunneling" disabled. Corporate VPNs are more likely to cause this issue.
```

### 11.2 Endpoint Issues

**Symptom**: "Endpoint offline" error
```
Diagnosis:
1. Check endpoint status:
   globus-compute-endpoint status [endpoint-name]

2. If stopped, start it:
   globus-compute-endpoint start [endpoint-name]

3. Check logs:
   tail -f ~/.globus_compute/[endpoint-name]/endpoint.log
```

**Symptom**: Tasks timeout
```
Causes:
- vLLM server not running on the endpoint
- GPU not allocated to the endpoint
- Network issues between endpoint and Globus service

Diagnosis:
1. SSH to HPC cluster
2. Check vLLM status: curl http://localhost:8000/health
3. Check GPU: nvidia-smi
```

### 11.3 Docker Issues

**Symptom**: Proxy can't authenticate
```
Check volume mount:
docker exec lakeshore-proxy ls -la /root/.globus_compute/

Expected: storage.db file present

If missing:
1. Run auth on host first
2. Check docker-compose.yml volume mapping
3. Ensure ${HOME} is set correctly
```

---

## 12. Q&A

### Q: "How do you access the HPC cluster without VPN or SSH?"

**Answer**: We use Globus Compute, a function-as-a-service platform designed for research computing. The key insight is that the Globus Compute endpoint running on the HPC cluster initiates an *outbound* connection to Globus's cloud service. Since outbound HTTPS is typically allowed through firewalls, we can submit tasks and receive results without any inbound ports or VPN connections. The user authenticates once via OAuth, and then all communication is routed through Globus's secure infrastructure.

### Q: "How does the authentication work?"

**Answer**: We implement OAuth 2.0 Authorization Code flow with a twist. Instead of requiring users to copy-paste authorization codes, we use the Globus SDK's LocalServerLoginFlowManager which starts a temporary HTTP server on localhost. When the user logs in through their browser, Globus redirects back to this local server, which automatically captures the authorization code and exchanges it for tokens. The tokens are cached locally, so authentication is a one-time setup that persists across sessions.

### Q: "How do you handle the latency of remote execution?"

**Answer**: We implement intelligent query routing. Simple queries go to our local Ollama instance for sub-second responses. Complex queries that benefit from larger models are routed to Lakeshore, where the latency is typically 3-10 seconds including network overhead. For maximum capability needs, we route to cloud APIs. The routing decision is made automatically based on query complexity analysis, though users can override it.

### Q: "Is this secure?"

**Answer**: Yes, security is multi-layered. Authentication uses Globus Auth, which supports institutional SSO and multi-factor authentication. All network traffic is encrypted with TLS. The endpoint only executes tasks from authenticated users with appropriate permissions. The vLLM server only listens on localhost within the HPC cluster, so it's not directly accessible from the network. Tokens are encrypted at rest and automatically refreshed.

### Q: "Can this scale to many users?"

**Answer**: Absolutely. Globus Compute is designed for research infrastructure scale. The cloud service handles task routing and queuing. The endpoint can be configured with multiple workers to handle concurrent tasks. For higher throughput, we can deploy multiple endpoints or use Globus Compute's support for SLURM job submission for larger tasks.

### Q: "What happens if the HPC cluster is down?"

**Answer**: STREAM implements automatic fallback. If the Lakeshore tier is unavailable (detected via health checks), queries automatically route to the cloud tier. Users see a notification that fallback occurred. They can also explicitly select a tier if they prefer to wait for Lakeshore rather than incur cloud costs.

### Q: "Why does authentication fail when I'm on VPN?"

**Answer**: This is a localhost routing issue, not a STREAM bug. During OAuth authentication, we start a temporary HTTP server on localhost (e.g., 127.0.0.1:56390) to receive the callback from Globus Auth. Some VPNs, especially those in full-tunnel mode, intercept ALL browser traffic including requests to localhost and route them through the VPN tunnel. Since our callback server is listening on your actual localhost (not accessible through the VPN tunnel), the browser can't connect. The solution is simple: disconnect your VPN before authenticating, complete the OAuth flow, then reconnect. This is a one-time issue since tokens are cached locally.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **FaaS** | Function-as-a-Service: Execute functions on remote infrastructure |
| **OAuth 2.0** | Authorization framework for granting limited access |
| **SSE** | Server-Sent Events: Protocol for streaming responses |
| **vLLM** | High-performance LLM inference engine |
| **HPC** | High-Performance Computing: Cluster of powerful computers |
| **Endpoint** | Globus Compute daemon that receives and executes tasks |
| **Refresh Token** | Long-lived credential for obtaining new access tokens |

---

## Appendix B: Environment Variables

```bash
# Required for Lakeshore integration
GLOBUS_COMPUTE_ENDPOINT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
VLLM_SERVER_URL=http://localhost:8000

# Optional configuration
LOG_LEVEL=INFO
HEALTH_CHECK_TTL=60
```

---

*This document is part of the STREAM project. For updates, see the project repository.*
