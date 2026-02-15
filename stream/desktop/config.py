"""
Desktop environment configuration.

WHY THIS FILE EXISTS:
---------------------
When STREAM runs in Docker, environment variables come from docker-compose.yml
and the .env file. In desktop mode there's no Docker, so we must set these
environment variables ourselves BEFORE any other module reads them.

This is the FIRST thing that runs when the desktop app starts (called from
main.py). It ensures stream/middleware/config.py sees the right values when
it loads.

HOW os.environ.setdefault() WORKS:
-----------------------------------
    os.environ.setdefault("KEY", "value")

This means: "Set KEY to value ONLY if KEY is not already set."
If the user has a .env file or exported a variable in their shell,
their value takes priority. We never overwrite what the user chose.
"""

import os


def apply_desktop_defaults():
    """
    Set environment variables needed for desktop mode.

    Called ONCE at the very start of main.py, BEFORE importing the middleware.
    This ensures stream/middleware/config.py picks up desktop values instead
    of Docker defaults.
    """

    # -------------------------------------------------------------------------
    # STREAM_MODE — the master switch
    # -------------------------------------------------------------------------
    # stream/middleware/config.py reads this to decide between two code paths:
    #   "server"  → PostgreSQL, LiteLLM HTTP server, Docker DNS names
    #   "desktop" → SQLite, direct LiteLLM library calls, localhost
    os.environ.setdefault("STREAM_MODE", "desktop")

    # -------------------------------------------------------------------------
    # FastAPI server address
    # -------------------------------------------------------------------------
    # "127.0.0.1" means "only accept connections from this machine."
    # This is safer than "0.0.0.0" (which listens on ALL network interfaces
    # and would let other devices on your network connect to your app).
    # In Docker, "0.0.0.0" is needed so containers can talk to each other.
    # On a desktop, we only need the app talking to itself.
    os.environ.setdefault("MIDDLEWARE_HOST", "127.0.0.1")
    os.environ.setdefault("MIDDLEWARE_PORT", "5000")

    # -------------------------------------------------------------------------
    # Ollama — the local AI model runner
    # -------------------------------------------------------------------------
    # In Docker, Ollama runs in a container named "ollama", so its hostname
    # is "ollama" (Docker's built-in DNS resolves container names).
    # On desktop, Ollama runs natively on the same machine → "localhost".
    os.environ.setdefault("OLLAMA_HOST", "localhost")
    os.environ.setdefault("OLLAMA_PORT", "11434")

    # -------------------------------------------------------------------------
    # LiteLLM — AI model gateway
    # -------------------------------------------------------------------------
    # In Docker, LiteLLM runs as a separate HTTP server on port 4000.
    # In desktop mode, we call LiteLLM as a Python library (litellm_direct.py),
    # so this URL is never actually used. But stream/middleware/config.py would
    # break if it were empty, so we give it a harmless placeholder.
    os.environ.setdefault("LITELLM_BASE_URL", "http://127.0.0.1:4000")
    os.environ.setdefault("LITELLM_MASTER_KEY", "not-needed-in-desktop-mode")

    # -------------------------------------------------------------------------
    # CORS — Cross-Origin Resource Sharing
    # -------------------------------------------------------------------------
    # In desktop mode, the React UI and API are served from the SAME origin
    # (both at http://127.0.0.1:5000), so the browser won't block requests.
    # We set it anyway as a safety net in case someone opens it in a browser.
    os.environ.setdefault("CORS_ORIGINS", "http://127.0.0.1:5000")

    # -------------------------------------------------------------------------
    # Lakeshore proxy
    # -------------------------------------------------------------------------
    # In Docker, the Lakeshore proxy runs in its own container ("lakeshore-proxy")
    # on port 8001. In desktop mode, there's no separate container — instead,
    # the proxy routes are mounted directly into our main FastAPI app at the
    # /lakeshore path prefix. So we set the full URL to point to ourselves:
    #   http://127.0.0.1:5000/lakeshore
    # This way, when tier_health.py calls LAKESHORE_PROXY_URL/health, it hits
    #   http://127.0.0.1:5000/lakeshore/health → our embedded proxy routes.
    middleware_port = os.environ.get("MIDDLEWARE_PORT", "5000")
    os.environ.setdefault("LAKESHORE_PROXY_HOST", "127.0.0.1")
    os.environ.setdefault("LAKESHORE_PROXY_PORT", middleware_port)
    os.environ.setdefault(
        "LAKESHORE_PROXY_URL",
        f"http://127.0.0.1:{middleware_port}/lakeshore",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    # "human" = readable text logs (good for desktop debugging).
    # "json"  = structured JSON logs (good for log aggregation tools like
    #           Grafana/Loki, but unreadable by humans).
    os.environ.setdefault("LOG_FORMAT", "human")
    os.environ.setdefault("LOG_LEVEL", "INFO")

    # -------------------------------------------------------------------------
    # Disable server-only features
    # -------------------------------------------------------------------------
    # DEBUG=true shows /docs and /redoc endpoints — not needed for desktop users.
    # RELOAD=true makes uvicorn watch for file changes — only useful during dev.
    os.environ.setdefault("DEBUG", "false")
    os.environ.setdefault("RELOAD", "false")
