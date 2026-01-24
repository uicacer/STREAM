"""
STREAM Middleware - Main Application

FastAPI service for intelligent AI model routing.
"""

from stream.middleware.utils.visuals import PreImportSpinner

spinner = PreImportSpinner()
spinner.start()

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.console import Console
from rich.panel import Panel

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
from stream.middleware.routes.chat import router as chat_router
from stream.middleware.routes.costs import router as costs_router
from stream.middleware.routes.health import router as health_router
from stream.middleware.utils.logging_config import configure_logging

spinner.stop()

console = Console()

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Configure logging (JSON for production, human-readable for development)
LOG_FORMAT_TYPE = os.getenv("LOG_FORMAT", "json")  # "json" or "human"
configure_logging(LOG_LEVEL, LOG_FORMAT, LOG_FORMAT_TYPE)  # ← ONE LINE!

logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await lifecycle.startup()
    yield
    await lifecycle.shutdown()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description=SERVICE_DESCRIPTION,
    debug=DEBUG,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    lifespan=lifespan,
)

# =============================================================================
# MIDDLEWARE
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Add correlation ID to every request."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    request.state.start_time = datetime.now(UTC)

    logger.info(
        f"[{correlation_id}] {request.method} {request.url.path}",
        extra={"correlation_id": correlation_id},
    )

    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id

    duration = (datetime.now(UTC) - request.state.start_time).total_seconds()
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
    """Global exception handler."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.error(
        f"[{correlation_id}] Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={"correlation_id": correlation_id},
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if DEBUG else "An error occurred",
            "correlation_id": correlation_id,
        },
    )


# =============================================================================
# ROUTERS
# =============================================================================

app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])
app.include_router(costs_router, prefix="/v1", tags=["Costs"])


# =============================================================================
# ROOT
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
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
    """Entry point for local development."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{SERVICE_NAME}[/bold cyan] [cyan]v{SERVICE_VERSION}[/cyan]",
            border_style="cyan",
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
