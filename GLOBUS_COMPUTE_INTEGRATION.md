# Globus Compute Integration for Lakeshore (HTTP Proxy Architecture)

This document explains the **HTTP proxy architecture** for Lakeshore connectivity in STREAM.

## Overview

STREAM uses a **lightweight HTTP proxy** that sits between LiteLLM and Lakeshore, enabling communication via either **Globus Compute** or **SSH port forwarding** while keeping LiteLLM as the unified gateway.

### Architecture

```
User → Frontend → Middleware → LiteLLM Gateway → Lakeshore Proxy → {Globus Compute OR SSH} → Lakeshore vLLM
```

**Key Components:**
1. **LiteLLM**: Unified API gateway (handles cost tracking, token counting, rate limiting)
2. **Lakeshore Proxy**: HTTP service that routes requests via Globus Compute or SSH
3. **Globus Compute** (preferred): Remote job submission to Lakeshore cluster
4. **SSH Port Forward** (fallback): Direct HTTP tunneling

### Why This Architecture?

**Benefits:**
- ✅ **LiteLLM stays in the loop**: Cost tracking, token counting work for all tiers
- ✅ **Single code path**: All tiers route through LiteLLM (no special cases)
- ✅ **Easy mode switching**: Change `USE_GLOBUS_COMPUTE` in `.env` to switch modes
- ✅ **Transparent**: Middleware doesn't need to know about transport mechanisms
- ✅ **Cleaner separation**: Proxy handles transport, LiteLLM handles API gateway logic

---

## How It Works

### Request Flow

1. **User Query** → Streamlit frontend
2. **Middleware** determines complexity → routes to Lakeshore tier
3. **Middleware** → LiteLLM with model `lakeshore-qwen`
4. **LiteLLM** → HTTP request to `http://lakeshore-proxy:8001/v1/chat/completions`
5. **Proxy checks** `USE_GLOBUS_COMPUTE` environment variable:

   **If Globus Compute mode (`USE_GLOBUS_COMPUTE=true`):**
   ```
   Proxy → Globus Compute SDK → Submit job to endpoint
   → Remote function executes on Lakeshore compute node
   → Makes HTTP request to vLLM (http://ga-001:8000)
   → Returns result → Proxy → LiteLLM → Middleware → Frontend
   ```

   **If SSH mode (`USE_GLOBUS_COMPUTE=false`):**
   ```
   Proxy → HTTP request to localhost:8000 (SSH tunnel)
   → SSH forwards to Lakeshore vLLM
   → Returns result → Proxy → LiteLLM → Middleware → Frontend
   ```

### Proxy Service Details

**Location**: `stream/proxy/app.py`

**Key Functions:**
- `/health`: Health check endpoint
- `/v1/chat/completions`: Main proxy endpoint (OpenAI API compatible)
- `_route_via_globus_compute()`: Handles Globus Compute routing
- `_route_via_ssh()`: Handles SSH port forwarding

**Docker Service**: `lakeshore-proxy`
- Port: 8001 (internal Docker network)
- Reads same environment variables as middleware
- Lightweight FastAPI server (~50 MB)

---

## Configuration

### Globus Compute Mode (Recommended)

**1. Set environment variables in `.env`:**
```bash
# Enable Globus Compute
USE_GLOBUS_COMPUTE=true

# Your Globus endpoint ID (get from: globus-compute-endpoint list)
GLOBUS_COMPUTE_ENDPOINT_ID=8d978809-eec4-413d-bbd4-b099e488100a

# vLLM URL *on Lakeshore* (not localhost!)
VLLM_SERVER_URL=http://ga-001:8000

# Task timeout (optional)
GLOBUS_TASK_TIMEOUT=120
```

**2. Start STREAM:**
```bash
docker-compose up -d
```

**3. Verify:**
```bash
# Check proxy is healthy
curl http://localhost:8001/health

# Should return:
# {
#   "status": "healthy",
#   "service": "Lakeshore vLLM Proxy",
#   "mode": "globus_compute",
#   "globus_configured": true
# }
```

**No SSH tunnel needed!** ✅

### SSH Port Forwarding Mode (Fallback)

**1. Set environment variables in `.env`:**
```bash
# Disable Globus Compute
USE_GLOBUS_COMPUTE=false

# SSH tunnel endpoint
LAKESHORE_VLLM_ENDPOINT=http://host.docker.internal:8000
```

**2. Start SSH tunnel:**
```bash
ssh -L 8000:ga-001:8000 nassar@lakeshore.acer.uic.edu -N
```

**3. Start STREAM:**
```bash
docker-compose up -d
```

**4. Verify:**
```bash
curl http://localhost:8001/health

# Should return:
# {
#   "status": "healthy",
#   "service": "Lakeshore vLLM Proxy",
#   "mode": "ssh_port_forward",
#   "globus_configured": null
# }
```

---

## Files Created/Modified

### New Files

1. **[stream/proxy/__init__.py](stream/proxy/__init__.py)** - Proxy package init
2. **[stream/proxy/app.py](stream/proxy/app.py)** - Proxy FastAPI server (main logic)
3. **[Dockerfile.proxy](Dockerfile.proxy)** - Docker image for proxy service
4. **[stream/middleware/core/globus_compute_client.py](stream/middleware/core/globus_compute_client.py)** - Globus Compute SDK wrapper (used by proxy)

### Modified Files

1. **[docker-compose.yml](docker-compose.yml)**
   - Added `lakeshore-proxy` service
   - LiteLLM depends on proxy

2. **[stream/gateway/litellm_config.yaml](stream/gateway/litellm_config.yaml)**
   - Changed Lakeshore `api_base` from `http://host.docker.internal:8000/v1` to `http://lakeshore-proxy:8001/v1`

3. **[stream/middleware/core/streaming.py](stream/middleware/core/streaming.py)**
   - **Simplified**: Removed Lakeshore bypass logic
   - All tiers now route through LiteLLM uniformly

4. **[stream/middleware/core/tier_health.py](stream/middleware/core/tier_health.py)**
   - **Simplified**: Lakeshore health check now just checks proxy service
   - Removed Globus/SSH dual-mode logic

### Deleted Files

1. **`stream/middleware/core/lakeshore_router.py`** - No longer needed (proxy handles routing)

---

## Comparison: Before vs. After

### Before (Bypass Architecture)

```
Middleware → Check tier
  ├─ If Lakeshore → Route through lakeshore_router (bypasses LiteLLM)
  │                  ├─ If Globus Compute → globus_compute_client
  │                  └─ If SSH → forward_to_litellm
  └─ Other tiers → forward_to_litellm → LiteLLM
```

**Issues:**
- ❌ Lakeshore requests bypassed LiteLLM (lost cost tracking)
- ❌ Two code paths (complex)
- ❌ Special case logic in streaming.py

### After (Proxy Architecture)

```
Middleware → LiteLLM (for ALL tiers)
  ├─ Local → LiteLLM → Ollama
  ├─ Lakeshore → LiteLLM → Proxy → {Globus OR SSH} → vLLM
  └─ Cloud → LiteLLM → Anthropic/OpenAI
```

**Benefits:**
- ✅ All requests through LiteLLM (unified gateway)
- ✅ Single code path (simpler)
- ✅ No special cases in middleware
- ✅ Proxy handles transport complexity

---

## Testing

### Test Proxy Directly

**Globus Compute mode:**
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hi!"}],
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

**SSH mode:**
```bash
# Same command, proxy automatically routes via SSH if USE_GLOBUS_COMPUTE=false
```

### Test Through STREAM

1. Start STREAM: `docker-compose up -d`
2. Open Streamlit: `http://localhost:8501`
3. Send a medium-complexity query (routes to Lakeshore)
4. Check logs:
   ```bash
   docker logs stream-lakeshore-proxy -f
   docker logs stream-litellm -f
   docker logs stream-middleware -f
   ```

### Verify Globus Compute

```bash
# Test Globus Compute directly
python tests/test_vllm_via_globus_compute.py
```

---

## Troubleshooting

### Proxy Not Starting

**Check logs:**
```bash
docker logs stream-lakeshore-proxy
```

**Common issues:**
- Missing `GLOBUS_COMPUTE_ENDPOINT_ID` (if Globus mode)
- Import error for `globus-compute-sdk` (check `pyproject.toml`)

### "Globus Compute not configured"

- Set `GLOBUS_COMPUTE_ENDPOINT_ID` in `.env`
- Verify endpoint is active: `globus-compute-endpoint status <id>`

### "Cannot connect to Lakeshore proxy"

- Check proxy service is running: `docker ps | grep proxy`
- Check health: `curl http://localhost:8001/health`
- Restart: `docker-compose restart lakeshore-proxy`

### SSH Mode Not Working

- Verify SSH tunnel is running: `ps aux | grep "ssh -L 8000"`
- Test tunnel: `curl http://localhost:8000/v1/models`
- Check `LAKESHORE_VLLM_ENDPOINT` in `.env`

### LiteLLM Can't Reach Proxy

- Check Docker network: `docker network inspect stream_network`
- Verify proxy is in network: `docker inspect stream-lakeshore-proxy`
- Check LiteLLM config: `cat stream/gateway/litellm_config.yaml | grep api_base`

---

## Why `VLLM_SERVER_URL` is Needed

**Question**: "Why do we need `VLLM_SERVER_URL=http://ga-001:8000` if Globus Compute submits directly to Lakeshore?"

**Answer**: The Globus Compute function **executes ON Lakeshore**, not on your laptop!

**The Flow:**
```
Your Laptop (Proxy)  →  Globus Compute  →  Lakeshore Compute Node
                                         ↓
                                  Remote function runs HERE
                                         ↓
                                  Makes HTTP request to vLLM
                                         ↓
                                  http://ga-001:8000 ✓
```

**Two different perspectives:**

| Location | Needs to connect to | URL |
|----------|-------------------|-----|
| **Your laptop** (SSH mode) | Lakeshore via tunnel | `http://host.docker.internal:8000` |
| **Lakeshore compute node** (Globus mode) | vLLM server on Lakeshore | `http://ga-001:8000` |

The remote function (`remote_vllm_inference`) is **serialized and sent to Lakeshore** where it executes. It needs the URL that works **from within Lakeshore's network**.

**Analogy**: It's like telling someone in a different building to "go to room 301" - you need to tell them the room number **in their building**, not yours!

---

## Summary

The HTTP proxy architecture provides a **clean, maintainable** solution for Lakeshore connectivity:

1. **LiteLLM** remains the unified API gateway
2. **Proxy** handles transport mechanism (Globus vs SSH)
3. **Middleware** stays simple (no special cases)
4. **Easy mode switching** via environment variables
5. **Full LiteLLM features** (cost tracking, token counting)

**Best Practice:**
- Use **Globus Compute** for production (no SSH tunnel needed)
- Use **SSH mode** for development/testing (easier setup)
