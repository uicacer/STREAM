# =============================================================================
# STREAM Middleware - Health Check Routes
# =============================================================================
# Health check and status endpoints
# =============================================================================

import logging
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse  # Import for proper status codes

from stream.middleware.config import (
    HEALTH_CHECK_TIMEOUT,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    SERVICE_VERSION,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================


@router.get("/health")
async def health_check():
    """
    Basic health check - is the service running?
    """
    return {
        "status": "healthy",
        "service": "STREAM Middleware",
        "version": SERVICE_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check - check all dependencies"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": SERVICE_VERSION,
        "dependencies": {},
    }

    # Check LiteLLM
    litellm_healthy = await check_litellm_health()
    health_status["dependencies"]["litellm"] = {
        "status": "healthy" if litellm_healthy else "unhealthy",
        "url": LITELLM_BASE_URL,
    }

    # Check Database
    db_healthy = await check_database_health()
    health_status["dependencies"]["database"] = {
        "status": "healthy" if db_healthy else "unavailable"
    }

    # Overall status
    if not litellm_healthy:
        health_status["status"] = "degraded"

    if not db_healthy:
        health_status["status"] = "degraded" if litellm_healthy else "unhealthy"

    return health_status


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness check"""
    litellm_ready = await check_litellm_health()

    if not litellm_ready:
        # Change status code for unhealthy
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "reason": "LiteLLM gateway unavailable",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    # Return dict (FastAPI handles it)
    return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check - is the process alive?
    Used by Kubernetes/orchestration systems
    """
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def check_database_health() -> bool:
    """
    Check if PostgreSQL database is healthy

    Verifies connection to LiteLLM's cost tracking database.
    Returns False if database is not configured.
    """

    # Import here to avoid potential circular dependency
    # (health.py and chat.py are both route modules in the same package)
    from stream.middleware.routes.chat import db_pool

    if not db_pool:
        return False

    conn = None  # Initialize
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False
    finally:
        # Always return connection
        if conn:
            db_pool.putconn(conn)


async def check_litellm_health() -> bool:
    """Check if LiteLLM gateway is healthy"""
    try:
        headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{LITELLM_BASE_URL}/health", headers=headers)
            return response.status_code == status.HTTP_200_OK
    except httpx.TimeoutException:
        logger.warning("LiteLLM health check timeout")
        return False
    except httpx.ConnectError:
        logger.warning("LiteLLM health check connection failed")
        return False
    except Exception as e:
        logger.error(f"LiteLLM health check error: {e}")
        return False
