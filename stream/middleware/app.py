"""
STREAM Middleware - Main Application

FastAPI service for intelligent AI model routing.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from rich.console import Console
from rich.panel import Panel

# find_react_dist locates the pre-built React UI (frontends/react/dist/).
# Returns None if no build exists — safe to call in server mode too.
# mount_static_files mounts the React dist/ folder onto FastAPI (desktop mode).
from stream.desktop.static_files import find_react_dist, mount_static_files
from stream.middleware.config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ORIGINS,
    DEBUG,
    LOG_FORMAT,
    LOG_LEVEL,
    MIDDLEWARE_HOST,
    MIDDLEWARE_PORT,
    RELOAD,
    SERVICE_DESCRIPTION,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from stream.middleware.core import lifecycle
from stream.middleware.routes.auth import router as auth_router
from stream.middleware.routes.chat import router as chat_router
from stream.middleware.routes.config import router as config_router
from stream.middleware.routes.costs import router as costs_router
from stream.middleware.routes.documents import router as documents_router
from stream.middleware.routes.health import router as health_router
from stream.middleware.routes.models import router as models_router
from stream.middleware.utils.logging_config import configure_logging
from stream.proxy.app import (
    router as lakeshore_router,  # Lakeshore proxy routes (mounted at /lakeshore in desktop mode)
)

console = Console()

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Configure logging (JSON for production, human-readable for development)
LOG_FORMAT_TYPE = os.getenv("LOG_FORMAT", "json")  # "json" or "human"
configure_logging(LOG_LEVEL, LOG_FORMAT, LOG_FORMAT_TYPE)

logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN
# =============================================================================


# async context manager
# Purpose: Manages asynchronous resource setup (e.g., connecting) and
# teardown (e.g., disconnecting) in non-blocking code.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Manages async resource setup and teardown.

    Startup tasks (before yield):
    1. Initialize database connection pool
    2. Load model pricing from LiteLLM config
    3. Pull Ollama models if missing
    4. Start background tier health checks

    Shutdown tasks (after yield):
    1. Stop background health checks
    2. Close database connections

    The "yield" statement means:
    - "Pause here and let the application run"
    - "When application stops, continue to cleanup code"

    Args:
    app: FastAPI app instance (required by FastAPI, not used here)

    """
    await lifecycle.startup()
    yield  # yield here means "pause and wait for the application to finish"
    await lifecycle.shutdown()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description=SERVICE_DESCRIPTION,
    debug=DEBUG,  # This means debug mode is enabled
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    lifespan=lifespan,
)

# =============================================================================
# MIDDLEWARE
# =============================================================================

"""
app.add_middleware():
---------------------
Registers a middleware class (CORSMiddleware).

What is CORS?
-------------
CORS = Cross-Origin Resource Sharing

Problem: Browsers block requests from different origins by default
Example:
  Frontend: http://localhost:8501 (Streamlit)
  Middleware: http://localhost:5000 (FastAPI)
  Browser: "Different ports = different origins, BLOCKED!" ❌

Solution: CORS headers tell browser "this cross-origin request is allowed"

CORS Parameters:
----------------
- allow_origins: Which domains can make requests
  ["http://localhost:8501"] = Only Streamlit can call our API
  ["*"] = Anyone can call (insecure, avoid in production)

- allow_credentials: Allow cookies/auth headers
  True = Browser can send credentials (needed for auth)
  False = No credentials allowed

- allow_methods: Which HTTP methods allowed
  ["*"] = All methods (GET, POST, PUT, DELETE, etc.)
  ["GET", "POST"] = Only these methods

- allow_headers: Which headers allowed
  ["*"] = All headers
  ["Content-Type"] = Only this header

How it works:
-------------
Browser sends "preflight" request (OPTIONS):
  OPTIONS /v1/chat/completions
  Origin: http://localhost:8501

CORSMiddleware responds:
  Access-Control-Allow-Origin: http://localhost:8501
  Access-Control-Allow-Methods: *
  Access-Control-Allow-Headers: *

Browser: "OK, this cross-origin request is safe" ✓
Then browser sends actual request (POST)
"""

# IMPORTANT: CORS settings should be reviewed before deploying to production
# In particular, review the allow_origins setting to avoid security issues.
# NOTE: Using ["*"] is not recommended for production.
# Consider using a specific list of allowed origins.
# Example:
# CORS_ORIGINS = [
#     "http://localhost:8501",
#     "https://your-frontend-domain.com",
# ]

# NOTE: middleware runs automatically for every request. No explicit function call
# needed for it to run

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Which domains can call our API
    allow_credentials=CORS_ALLOW_CREDENTIALS,  # Allow cookies/auth
    allow_methods=CORS_ALLOW_METHODS,  # Which HTTP methods are allowed
    allow_headers=CORS_ALLOW_HEADERS,  # Which headers are allowed
)


# Custom middleware to add correlation ID
# This function is NEVER called explicitly!
# FastAPI calls it automatically because of the @app.middleware decorator
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """
    Add correlation ID to every request.

    This middleware:
    1. Generates or extracts correlation ID (UUID)
    2. Stores in request.state for use by route handlers
    3. Logs request start
    4. Calls next handler
    5. Adds correlation ID to response headers
    6. Logs request completion with duration

    Args:
        request: Incoming HTTP request
        call_next: Function to call next middleware/handler

    Returns:
        HTTP response (possibly modified by this middleware)


    Notes:
    1) This middleware is applied to all incoming requests.
    2) call_next is a function that takes the request as input and returns the response.
    """
    # Extract correlation ID from header or generate new UUID
    # uuid library generates random UUIDs that are unique across space and time
    # By space and time we mean across different machines and at different times
    # So no 2 UUIDs are the same wherever they are generated
    # uuid stands for Universally Unique Identifier
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

    # Store in request.state (accessible in all route handlers)
    request.state.correlation_id = correlation_id
    request.state.start_time = datetime.now(UTC)

    logger.info(
        f"[{correlation_id}] {request.method} {request.url.path}",
        extra={"correlation_id": correlation_id},
    )

    # Call next middleware or route handler
    # This is where the actual request processing happens
    response = await call_next(request)

    # Add correlation ID to response headers
    # Allows client to track their request
    response.headers["X-Correlation-ID"] = correlation_id

    # Calculate request duration. This is needed for logging
    duration = (datetime.now(UTC) - request.state.start_time).total_seconds()

    # Log request completion with status and duration
    logger.info(
        f"[{correlation_id}] {response.status_code} ({duration:.3f}s)",
        extra={
            "correlation_id": correlation_id,
            "status_code": response.status_code,
            "duration_seconds": duration,
        },
    )

    return response


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Catches any exception not handled by route handlers.
    Logs the error with full traceback and returns formatted response.

    In production (DEBUG=False):
    - Hides error details from user (security)
    - Returns generic message

    In development (DEBUG=True):
    - Shows full error message
    - Helps with debugging

    Args:
        request: Current request (contains correlation_id in request.state)
        exc: The exception that was raised

    Returns:
        JSONResponse with error details and 500 status code

    Example:
        Route handler:
            def chat():
                raise ValueError("Invalid model")

        This handler catches it:
            → Logs: [abc-123] Unhandled exception: Invalid model
            → Returns: {"error": "Internal server error", ...}
    """

    # Get correlation ID from request state (added by middleware)
    # If not found (shouldn't happen), use "unknown"
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    # Log error with full traceback
    # exc_info=True includes full stack trace for debugging
    logger.error(
        f"[{correlation_id}] Unhandled exception: {str(exc)}",
        exc_info=True,  # Include full traceback
        extra={"correlation_id": correlation_id},
    )

    # Return formatted error response
    return JSONResponse(
        status_code=500,  # Internal Server Error
        content={
            "error": "Internal server error",
            "message": str(exc) if DEBUG else "An error occurred",  # Hide details in production
            "correlation_id": correlation_id,  # Allow user to reference in support
        },
    )


# =============================================================================
# ROUTERS
# =============================================================================

app.include_router(health_router, tags=["Health"])
app.include_router(auth_router, prefix="/v1", tags=["Auth"])
app.include_router(config_router, prefix="/v1", tags=["Config"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])
app.include_router(costs_router, prefix="/v1", tags=["Costs"])
app.include_router(documents_router, prefix="/v1", tags=["Documents"])
app.include_router(models_router, prefix="/v1", tags=["Models"])


# =============================================================================
# LAKESHORE PROXY ROUTES (Desktop Mode Only)
# =============================================================================
# In Docker, the Lakeshore proxy runs as a separate container on port 8001.
# In desktop mode, there's no Docker — so we embed the proxy routes directly
# into this FastAPI app at the /lakeshore prefix.
#
# This means requests like:
#   GET  /lakeshore/health              → proxy health check
#   POST /lakeshore/reload-auth         → reload Globus credentials
#   POST /lakeshore/v1/chat/completions → forward to Lakeshore HPC
#
# The LAKESHORE_PROXY_URL config is set to http://127.0.0.1:5000/lakeshore
# in desktop mode, so all existing code (tier_health, auth, litellm_direct)
# routes to these embedded endpoints automatically.
if os.environ.get("STREAM_MODE") == "desktop":
    app.include_router(lakeshore_router, prefix="/lakeshore", tags=["Lakeshore"])


# =============================================================================
# STATIC FILE SERVING (Desktop Mode Only)
# =============================================================================
# In desktop mode, there's no separate Vite dev server to serve the React UI.
# So we mount the pre-built React files (frontends/react/dist/) directly onto
# FastAPI. This lets one server handle BOTH the API and the UI.
#
# IMPORTANT: This MUST come AFTER all API routers are registered above.
# The static file mounter adds a "catch-all" route that returns index.html
# for any URL that doesn't match an existing route. If we mounted it BEFORE
# the API routes, it would intercept /v1/chat/completions, /health, etc.
# By registering it last, API routes get matched first, and only truly
# unrecognized URLs (like /settings, /about) fall through to the React UI.
# We check os.environ directly instead of the imported STREAM_MODE variable.
# This is more reliable because os.environ always reflects the actual current
# state, whereas the imported variable depends on module import ordering.
if os.environ.get("STREAM_MODE") == "desktop":
    mount_static_files(app)


# =============================================================================
# ROOT
# =============================================================================


@app.get("/")
async def root():
    """
    Root endpoint — serves different content based on the mode.

    We check os.environ at REQUEST TIME (not import time) because the
    environment variable is always correct, regardless of module import order.
    Module-level checks like `if STREAM_MODE == "desktop":` can fail if
    config.py gets imported before apply_desktop_defaults() runs.
    Checking os.environ directly at request time avoids that problem entirely.

    Desktop mode: returns the React UI (index.html)
    Server mode:  returns JSON service info (useful for health checks)
    """
    # Check the actual environment variable right now, not a cached import
    if os.environ.get("STREAM_MODE") == "desktop":
        # Serve the React frontend's index.html
        dist_path = find_react_dist()
        if dist_path:
            index_html = dist_path / "index.html"
            response = HTMLResponse(index_html.read_text())
            response.headers["Cache-Control"] = "no-store"
            return response

    # Server mode (or desktop mode with no React build): return JSON status
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/v1/chat/completions",
            "costs": "/v1/costs/models",
            "docs": "/docs" if DEBUG else "disabled",
        },
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Entry point for local development.

    Starts uvicorn server with auto-reload enabled.

    Usage:
        python -m stream.middleware.app

    Or:
        uvicorn stream.middleware.app:app --reload
    """
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{SERVICE_NAME}[/bold cyan] [cyan]v{SERVICE_VERSION}[/cyan]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()

    # Start uvicorn server
    uvicorn.run(
        "stream.middleware.app:app",  # Python path to app
        host=MIDDLEWARE_HOST,
        port=MIDDLEWARE_PORT,
        reload=RELOAD,
        log_level=False,  # LOG_LEVEL.lower(),
    )


# If this file is run directly (not imported)
if __name__ == "__main__":
    main()
