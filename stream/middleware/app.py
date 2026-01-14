# =============================================================================
# STREAM Middleware - Main Application
# =============================================================================
# FastAPI service that sits between UI and LiteLLM gateway
# Provides: Authentication, Policy, Telemetry, Tool Routing
# =============================================================================

from stream.middleware.utils.visuals import PreImportSpinner

# Start the spinner
spinner = PreImportSpinner()
spinner.start()

import io
import logging
import time
import uuid
from contextlib import asynccontextmanager, redirect_stdout
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Rich library for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from stream.backend.src.core.ollama_manager import OllamaModelManager
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
    OLLAMA_MODELS,
    RELOAD,
    SERVICE_DESCRIPTION,
    SERVICE_NAME,
    SERVICE_VERSION,
    check_all_tiers,
    judge_complexity_with_llm,
    validate_costs_match_litellm,
)

# Import routes (these are separate files with specific endpoints)
from stream.middleware.routes import chat, health

# Stop the spinner - imports are done!
spinner.stop()

# Initialize Rich console
console = Console()

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Configure Python's logging system to output logs with timestamps and levels
# This helps us debug issues by seeing what happened when

logging.basicConfig(
    level=LOG_LEVEL,  # INFO, DEBUG, WARNING, ERROR
    format=LOG_FORMAT,  # Timestamp - Logger name - Level - Message
)
logger = logging.getLogger(__name__)  # Create logger for this file


# =============================================================================
# LIFESPAN EVENT HANDLER
# =============================================================================
# Modern way to handle startup and shutdown events in FastAPI
# This replaces the deprecated @app.on_event("startup") approach
#
# Why we need this:
# - Run setup code when server starts (connect to databases, etc.)
# - Run cleanup code when server stops (close connections, etc.)
# - "yield" separates startup from shutdown logic


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # ==========================================================================
    # STARTUP
    # ==========================================================================
    logger.info(f"🚀 {SERVICE_NAME} v{SERVICE_VERSION} starting up...")
    logger.info(f"📊 Debug mode: {DEBUG}")
    logger.info(f"🔗 CORS origins: {CORS_ORIGINS}")
    logger.info("✅ Middleware ready!")

    yield

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================
    logger.info(f"👋 {SERVICE_NAME} shutting down...")

    # Close database connection pool
    if chat.db_pool:
        chat.db_pool.closeall()
        logger.info("✅ Database connection pool closed")


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================
# Create the main FastAPI application instance
# This is the core object that handles all HTTP requests

app = FastAPI(
    title=SERVICE_NAME,  # Shows in API docs
    version=SERVICE_VERSION,  # Version number
    description=SERVICE_DESCRIPTION,  # API description
    debug=DEBUG,  # Enable detailed error messages in dev
    docs_url="/docs" if DEBUG else None,  # Swagger UI (only in development)
    redoc_url="/redoc" if DEBUG else None,  # ReDoc UI (only in development)
    lifespan=lifespan,  # ← Connect the lifespan handler we defined above
)

# =============================================================================
# CORS MIDDLEWARE
# =============================================================================
# CORS = Cross-Origin Resource Sharing
#
# WHY WE NEED THIS:
# Web browsers block requests from one domain to another for security.
# Example: Streamlit runs on localhost:8501, middleware on localhost:5000
# Without CORS, browser would block the request!
#
# WHAT THIS DOES:
# Tells browsers "it's okay for these origins to make requests to me"

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Which websites can call this API
    # Example: ["http://localhost:8501"] = Streamlit can call us
    allow_credentials=CORS_ALLOW_CREDENTIALS,  # Allow cookies/auth headers
    allow_methods=CORS_ALLOW_METHODS,  # Allow GET, POST, etc.
    allow_headers=CORS_ALLOW_HEADERS,  # Allow any headers
)

# =============================================================================
# CORRELATION ID MIDDLEWARE
# =============================================================================
# WHAT IS A CORRELATION ID?
# A unique ID for each request that flows through all systems.
#
# WHY WE NEED IT:
# When debugging errors, we can search logs for one ID and see the complete
# journey of a request through: UI → Middleware → LiteLLM → Backends
#
# EXAMPLE:
# User reports error → Search logs for correlation_id="abc-123"
# See: [abc-123] Request received → Routed to cloud → Error at LiteLLM
# Now we know exactly where it failed!


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """
    Add correlation ID to every request for tracing

    HOW IT WORKS:
    1. Request comes in
    2. We generate or extract a correlation ID
    3. Attach it to request.state (available throughout request handling)
    4. Process the request (call_next does this)
    5. Add correlation ID to response headers
    6. Return response to client
    """
    # Generate unique ID or use one from request headers
    # UUID4 = Universally Unique Identifier (practically impossible to collide)
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

    # Attach to request state so other functions can access it
    request.state.correlation_id = correlation_id
    request.state.start_time = datetime.now(UTC)  # For timing requests

    # Log the incoming request
    logger.info(
        f"[{correlation_id}] {request.method} {request.url.path}",
        extra={"correlation_id": correlation_id},  # Structured logging
    )

    # Process the request (call the actual endpoint)
    # This is where your route handlers (chat.py, health.py) run
    response = await call_next(request)

    # Add correlation ID to response headers
    # Client can see this ID and use it when reporting issues
    response.headers["X-Correlation-ID"] = correlation_id

    # Calculate how long the request took
    duration = (datetime.now(UTC) - request.state.start_time).total_seconds()

    # Log the response
    logger.info(
        f"[{correlation_id}] Response: {response.status_code} ({duration:.3f}s)",
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
# WHAT IS AN EXCEPTION HANDLER?
# A "safety net" that catches any errors that weren't handled elsewhere.
#
# WHY WE NEED IT:
# Without this, if code crashes, users see ugly Python tracebacks.
# With this, users get clean JSON error messages.


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors

    WHEN THIS RUNS:
    - Code throws an exception
    - Exception isn't caught anywhere else
    - This function catches it and returns a nice error message

    WHAT IT DOES:
    - Logs the error with full traceback (for developers)
    - Returns clean JSON error to user
    - Includes correlation ID so we can find it in logs
    """
    # Get correlation ID (might not exist if error was very early)
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    # Log the error with full stack trace
    logger.error(
        f"[{correlation_id}] Unhandled exception: {str(exc)}",
        exc_info=True,  # Include full traceback in logs
        extra={"correlation_id": correlation_id},
    )

    # Return clean error response to user
    return JSONResponse(
        status_code=500,  # Internal Server Error
        content={
            "error": "Internal server error",
            # Show detailed message only in debug mode (security!)
            "message": str(exc) if DEBUG else "An error occurred",
            "correlation_id": correlation_id,  # User can report this ID
        },
    )


# =============================================================================
# INCLUDE ROUTERS
# =============================================================================
# WHAT ARE ROUTERS?
# Routers are separate files containing related endpoints.
# Instead of putting all endpoints in app.py, we organize them:
# - health.py = health check endpoints
# - chat.py = AI chat endpoints
#
# WHY THIS IS BETTER:
# - app.py stays clean and readable
# - Easy to find code (all health checks in health.py)
# - Different people can work on different routers

# Health check endpoints (/health, /health/detailed, etc.)
app.include_router(
    health.router,  # Import router from routes/health.py
    tags=["Health"],  # Group in API docs under "Health"
)

# Chat endpoints (/v1/chat/completions)
app.include_router(
    chat.router,  # Import router from routes/chat.py
    prefix="/v1",  # All routes get /v1 prefix
    tags=["Chat"],  # Group in API docs under "Chat"
)

# =============================================================================
# ROOT ENDPOINT
# =============================================================================
# The "/" endpoint - what users see when they visit http://localhost:5000


@app.get("/")
async def root():
    """
    Root endpoint - service info

    WHEN TO USE:
    - Check if service is running: curl http://localhost:5000
    - See what endpoints are available
    - Get version information

    RETURNS:
    JSON with service name, version, status, available endpoints
    """
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/v1/chat/completions",
            "docs": "/docs" if DEBUG else "disabled",
        },
    }


# =============================================================================
# MAIN (for running directly)
# =============================================================================
# This code only runs when you execute: python app.py
# It does NOT run when imported by other files
def main():
    """Entry point for CLI command"""

    # Print banner
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{SERVICE_NAME}[/bold cyan] [cyan]v{SERVICE_VERSION}[/cyan]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()

    # Step 1: Check/download Ollama models
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Checking required Ollama models..."),
        console=console,
        transient=True,
    ) as progress:
        # task = progress.add_task("check_models", total=None)
        manager = OllamaModelManager()
        missing_models = []
        for _, ollama_model in OLLAMA_MODELS.items():
            if not manager.is_model_available(ollama_model):
                missing_models.append(ollama_model)

    if not missing_models:
        console.print("   ✅ [green]All Ollama models available[/green]")
    else:
        console.print(
            f"   ⚠️  [yellow]{len(missing_models)} model(s) need to be downloaded[/yellow]"
        )
        for model in missing_models:
            success = manager.ensure_model(model, auto_download=False)
            if not success:
                console.print(f"   ❌ [red]Skipped {model}[/red]")

    console.print()

    # Step 2: Health checks
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Running startup health checks..."),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("health_check", total=None)
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            check_all_tiers()

    console.print(output_buffer.getvalue(), end="")

    # Step 3: Warm up judge
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Warming up LLM judge (first load may take 30-60s)..."),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("warmup", total=None)
        start = time.time()
        result = judge_complexity_with_llm("warmup test")
        elapsed = time.time() - start

    if result:
        console.print(f"   ✅ [green]Judge model ready in {elapsed:.1f}s[/green]")
    else:
        console.print(f"   ⚠️  [yellow]Judge warmup failed in {elapsed:.1f}s[/yellow]")

    # Step 4: Validate costs
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Validating cost configurations..."),
        console=console,
        transient=True,
    ) as progress:
        # task = progress.add_task("validate_costs", total=None)
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            validate_costs_match_litellm()

    console.print(output_buffer.getvalue(), end="")

    # Start server
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]🚀 Starting server on {MIDDLEWARE_HOST}:{MIDDLEWARE_PORT}[/bold green]",
            border_style="green",
            padding=(0, 2),
        )
    )
    console.print()

    uvicorn.run(
        "stream.middleware.app:app",
        host=MIDDLEWARE_HOST,
        port=MIDDLEWARE_PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()

# =============================================================================
# HOW REQUESTS FLOW THROUGH THIS FILE
# =============================================================================
"""
1. REQUEST ARRIVES
   ↓
2. CORS MIDDLEWARE checks if origin is allowed
   ↓
3. CORRELATION ID MIDDLEWARE adds tracking ID
   ↓
4. ROUTER matches URL to endpoint
   - / → root()
   - /health → health.router
   - /v1/chat/completions → chat.router
   ↓
5. ENDPOINT HANDLER processes request
   ↓
6. CORRELATION ID MIDDLEWARE adds ID to response
   ↓
7. RESPONSE SENT to client

IF ERROR OCCURS:
   ↓
EXCEPTION HANDLER catches it
   ↓
Returns clean error message
"""

# =============================================================================
# LEARNING RESOURCES
# =============================================================================
"""
FastAPI basics:
https://fastapi.tiangolo.com/tutorial/

Middleware:
https://fastapi.tiangolo.com/advanced/middleware/

Lifespan events:
https://fastapi.tiangolo.com/advanced/events/

CORS explained:
https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS

Logging best practices:
https://docs.python.org/3/howto/logging.html
"""
