# STREAM Middleware

FastAPI service that provides authentication, policy enforcement, and telemetry for STREAM.

## Architecture

```
Streamlit UI (Port 8501)
       ↓
Middleware (Port 5000) ← YOU ARE HERE
       ↓
LiteLLM Gateway (Port 4000)
       ↓
Backends (Ollama, Claude, GPT, vLLM)
```

## Features

### Current (v0.1.0)
- ✅ Intelligent tier routing (local/lakeshore/cloud)
- ✅ OpenAI-compatible API
- ✅ Correlation ID tracking
- ✅ Health checks
- ✅ CORS enabled
- ✅ Request logging

### Coming Soon
- ⏳ JWT authentication (Shibboleth OIDC)
- ⏳ Policy enforcement (quotas, ABAC)
- ⏳ Splunk telemetry
- ⏳ MCP tool gateway
- ⏳ Rate limiting (Redis)
- ⏳ Cost tracking (PostgreSQL)

## Quick Start

### 1. Install Dependencies

```bash
cd middleware
pip install -r requirements.txt
```

### 2. Configure Environment

Middleware uses the same `.env` file as the rest of STREAM:

```bash
# From STREAM root
bash scripts/sync-env.sh
```

### 3. Start Middleware

```bash
# Development mode (auto-reload)
python app.py

# Or with uvicorn directly
uvicorn app:app --reload --port 5000
```

### 4. Test Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Service info
curl http://localhost:5000/

# Chat completion (make sure LiteLLM is running)
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## API Endpoints

### Health & Status

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health (checks LiteLLM)
- `GET /health/ready` - Readiness probe (Kubernetes)
- `GET /health/live` - Liveness probe (Kubernetes)
- `GET /` - Service information

### Chat

- `POST /v1/chat/completions` - OpenAI-compatible chat endpoint

### Documentation

- `GET /docs` - Swagger UI (development only)
- `GET /redoc` - ReDoc UI (development only)

## Configuration

Key settings in `config.py`:

```python
MIDDLEWARE_PORT = 5000          # Service port
LITELLM_BASE_URL = "http://localhost:4000"  # LiteLLM endpoint
DEBUG = True/False              # Enable debug features
```

## Request Flow

```
1. Request arrives → CORS middleware
2. Correlation ID assigned
3. Route to /v1/chat/completions
4. Determine tier (local/lakeshore/cloud)
5. Forward to LiteLLM with selected model
6. Add STREAM metadata to response
7. Return to client
```

## Correlation IDs

Every request gets a unique correlation ID that flows through all logs:

```
[abc-123] POST /v1/chat/completions
[abc-123] Routing: tier=local, model=local-llama
[abc-123] Request completed (2.3s)
```

This enables end-to-end tracing in production.

## Development

### Project Structure

```
middleware/
├── app.py              # Main FastAPI app
├── config.py           # Configuration
├── routes/
│   ├── chat.py        # Chat endpoints
│   └── health.py      # Health checks
└── requirements.txt   # Dependencies
```

### Running Tests

```bash
# TODO: Add pytest tests
pytest
```

### Adding New Routes

1. Create file in `routes/`
2. Import in `app.py`
3. Include router: `app.include_router(new_router)`

## Troubleshooting

**Middleware won't start:**
- Check LiteLLM is running: `docker-compose ps`
- Check port 5000 is free: `lsof -i :5000`

**CORS errors:**
- Add your UI URL to `CORS_ORIGINS` in config

**Can't reach LiteLLM:**
- Verify LiteLLM URL: `curl http://localhost:4000/health`
- Check Docker network: `docker network ls`

## Next Steps

1. ✅ Test basic routing works
2. Add JWT authentication
3. Add policy enforcement
4. Add Splunk logging
5. Add tool gateway

---

**Version:** 0.1.0
**Status:** Alpha - Basic routing only
