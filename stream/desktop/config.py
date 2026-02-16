"""
Desktop environment configuration.

WHY THIS FILE EXISTS:
---------------------
When STREAM runs in Docker, environment variables come from docker-compose.yml
and the .env file. In desktop mode there's no Docker, so we must set these
environment variables ourselves BEFORE any other module reads them.

This is the FIRST thing that runs when the desktop app starts (called from
main.py). It ensures stream/middleware/config.py sees the right values when
it loads. Server/Docker mode never imports this module.

TWO KINDS OF ENV VARS:
----------------------
1. API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
   → Loaded from .env files. User values take priority. We never overwrite.

2. Networking/infrastructure (MIDDLEWARE_HOST, OLLAMA_HOST, etc.)
   → FORCE-SET to desktop values. The .env file has Docker hostnames like
     "middleware" and "ollama" (Docker container DNS names). These don't
     exist on a desktop machine — we must always use "127.0.0.1"/"localhost".
     Using setdefault() here would be wrong because load_dotenv() already
     set the Docker values, and setdefault() would leave them in place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def apply_desktop_defaults():
    """
    Set environment variables needed for desktop mode.

    Called ONCE at the very start of main.py, BEFORE importing the middleware.
    This ensures stream/middleware/config.py picks up desktop values instead
    of Docker defaults.
    """

    # -------------------------------------------------------------------------
    # LOAD .env FILES — API keys and user configuration
    # -------------------------------------------------------------------------
    # In Docker, environment variables come from docker-compose.yml.
    # In desktop mode, we need to find and load .env files ourselves.
    #
    # The problem: Python's load_dotenv() searches from the current working
    # directory upward. When the user launches STREAM.app by double-clicking,
    # the working directory is their home folder (NOT the project root), so
    # load_dotenv() can't find the .env file. API keys never get loaded.
    #
    # The fix: explicitly load .env from two known locations:
    #   1. ~/.stream/.env — The permanent config location for the bundled app.
    #      Users put their API keys here and they work regardless of how the
    #      app is launched (double-click, terminal, Spotlight, etc.).
    #   2. <project_root>/.env — For development mode when running from source.
    #      This is the same .env Docker uses, so developers don't need to
    #      maintain two copies during development.
    #
    # load_dotenv(override=False) means: only set a variable if it's not
    # already in the environment. So if both files define OPENAI_API_KEY,
    # the first one loaded wins (user config takes priority over dev config).

    # Priority 1: User's permanent config directory
    user_env = Path.home() / ".stream" / ".env"
    if user_env.exists():
        load_dotenv(user_env, override=False)

    # Priority 2: Project root .env (for development mode)
    # In development, this file's path is: <project_root>/stream/desktop/config.py
    # So project root is 3 directories up from this file.
    # In the PyInstaller bundle, this path points inside the app bundle where
    # there's no .env file — the if-check safely skips it.
    project_env = Path(__file__).resolve().parent.parent.parent / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=False)

    # =====================================================================
    # FORCE-SET: Networking/infrastructure variables
    # =====================================================================
    # These MUST be desktop values. The .env file may contain Docker
    # hostnames ("middleware", "ollama", "lakeshore-proxy") that don't
    # resolve on a desktop machine. We use direct assignment (not
    # setdefault) to guarantee the correct values regardless of .env.

    # STREAM_MODE — the master switch that config.py uses to choose
    # between server code paths (PostgreSQL, HTTP gateway) and desktop
    # code paths (SQLite, direct library calls).
    os.environ["STREAM_MODE"] = "desktop"

    # FastAPI server address — "127.0.0.1" = only accept local connections.
    # In Docker this is "0.0.0.0" (all interfaces), but on desktop we only
    # need the app talking to itself.
    os.environ["MIDDLEWARE_HOST"] = "127.0.0.1"
    os.environ["MIDDLEWARE_PORT"] = "5000"

    # Ollama — in Docker it's a container named "ollama". On desktop it
    # runs natively on the same machine → localhost.
    os.environ["OLLAMA_HOST"] = "localhost"
    os.environ["OLLAMA_PORT"] = "11434"

    # LiteLLM — in Docker it's an HTTP server on port 4000. In desktop mode
    # we call LiteLLM as a Python library (litellm_direct.py), so this URL
    # is never used. But config.py would break if it were empty.
    os.environ["LITELLM_BASE_URL"] = "http://127.0.0.1:4000"
    os.environ["LITELLM_MASTER_KEY"] = "not-needed-in-desktop-mode"

    # CORS — in desktop mode, React UI and API share the same origin
    # (both at http://127.0.0.1:5000), so the browser won't block requests.
    os.environ["CORS_ORIGINS"] = "http://127.0.0.1:5000"

    # Lakeshore proxy — in Docker it's a separate container on port 8001.
    # In desktop mode, proxy routes are mounted into our FastAPI app at
    # /lakeshore. So we point LAKESHORE_PROXY_URL to ourselves.
    os.environ["LAKESHORE_PROXY_HOST"] = "127.0.0.1"
    os.environ["LAKESHORE_PROXY_PORT"] = "5000"
    os.environ["LAKESHORE_PROXY_URL"] = "http://127.0.0.1:5000/lakeshore"

    # =====================================================================
    # SETDEFAULT: Variables where user/env values should take priority
    # =====================================================================
    # These don't have Docker-specific values that would break desktop mode.
    # setdefault() means: "only set if not already defined" — so if the user
    # exported a value in their shell, it wins.

    # "human" = readable text logs; "json" = structured logs for aggregation
    os.environ.setdefault("LOG_FORMAT", "human")
    os.environ.setdefault("LOG_LEVEL", "INFO")

    # Server-only features — not needed for desktop users
    os.environ.setdefault("DEBUG", "false")
    os.environ.setdefault("RELOAD", "false")
