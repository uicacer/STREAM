"""
Application lifecycle management.

This module handles startup and shutdown logic for STREAM middleware.
Extracted from app.py to keep the main application file clean.
"""

import logging
import sys

from stream.middleware.config import (
    CORS_ORIGINS,
    DEBUG,
    OLLAMA_MODELS,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from stream.middleware.core.complexity_judge import judge_complexity_with_llm
from stream.middleware.core.database import close_database_pool, initialize_database_pool
from stream.middleware.core.health_monitor import health_monitor
from stream.middleware.core.ollama_manager import OllamaModelManager
from stream.middleware.core.tier_health import check_all_tiers
from stream.middleware.core.warm_ping import warm_up_all_tiers

logger = logging.getLogger(__name__)


async def startup():
    """
    Application startup logic.

    This function runs once when the server starts.
    It performs all initialization tasks:
    1. Log service info
    2. Check/download Ollama models
    3. Initialize database
    4. Run health checks
    5. Warm up LLM judge
    6. Validate configuration
    """
    logger.info(f"🚀 {SERVICE_NAME} v{SERVICE_VERSION} starting up...")
    logger.info(f"📊 Debug mode: {DEBUG}")
    logger.info(f"🔗 CORS origins: {CORS_ORIGINS}")

    # Step 1: Check Ollama models
    # In desktop mode, the interactive download prompt runs in main.py
    # BEFORE the server starts (so it doesn't block the startup lifecycle).
    # Here we only log warnings for missing models (useful in Docker/server mode
    # where models should already be pre-downloaded).
    logger.info("🔍 Checking required Ollama models...")
    await _check_ollama_models()

    # Step 2: Initialize database
    logger.info("🔍 Initializing database connection pool...")
    initialize_database_pool()

    # Step 3: Health checks
    logger.info("🔍 Running startup health checks...")
    check_all_tiers()

    # Step 4: Start background health monitor
    logger.info("🔍 Starting background health monitor...")
    health_monitor.start()

    # Step 5: Warm up judge (optional)
    logger.info("🔍 Warming up LLM judge...")
    judge_complexity_with_llm("warmup test")

    # Step 6: Warm ping all tiers
    # This sends a small test request to each tier to:
    # - Pre-load models into memory (especially Ollama)
    # - Detect actual availability (not just proxy health)
    # - Establish connections early
    logger.info("🔍 Warming up inference tiers...")
    await warm_up_all_tiers()

    logger.info("✅ Middleware ready!")


async def shutdown():
    """
    Application shutdown logic.

    This function runs once when the server stops.
    It performs cleanup tasks:
    1. Close Globus Compute Executor (AMQP connection)
    2. Stop background health monitor
    3. Close database connections
    """
    logger.info(f"👋 {SERVICE_NAME} shutting down...")

    # Close the persistent Globus Compute Executor.
    # This properly closes the AMQP connection to Globus cloud instead of
    # letting it die when the process exits.
    try:
        import stream.proxy.app as _proxy_app

        if _proxy_app.globus_client is not None:
            _proxy_app.globus_client.shutdown()
    except Exception as e:
        logger.debug(f"Globus client shutdown (best effort): {e}")

    # Stop background health monitor
    health_monitor.stop()

    # Close database pool
    close_database_pool()

    logger.info("✅ Shutdown complete")


async def _check_ollama_models():
    """
    Log warnings for missing Ollama models (non-blocking).

    This runs inside the FastAPI startup lifecycle, so it must NEVER block
    on user input — that would prevent the server from accepting requests
    and cause the health-check timeout in desktop mode.

    Interactive downloads are handled earlier in main.py (desktop mode) or
    by the user manually (Docker mode).
    """
    manager = OllamaModelManager()
    missing_models = []

    for _, ollama_model in OLLAMA_MODELS.items():
        if not manager.is_model_available(ollama_model):
            missing_models.append(ollama_model)
            logger.warning(f"⚠️  Model {ollama_model} not found")

    if not missing_models:
        logger.info("✅ All Ollama models available")
    else:
        logger.warning(
            f"⚠️  {len(missing_models)} model(s) not found — LOCAL tier may be unavailable"
        )
        if not sys.stdin.isatty():
            logger.warning("📋 To download missing models:")
            for model in missing_models:
                logger.warning(f"   docker exec -it stream-ollama ollama pull {model}")
