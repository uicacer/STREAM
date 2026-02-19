"""
Application lifecycle management.

This module handles startup and shutdown logic for STREAM middleware.
Extracted from app.py to keep the main application file clean.
"""

import logging
import sys
import time

from stream.middleware.config import (
    DEBUG,
    OLLAMA_MODELS,
    SERVICE_NAME,
    SERVICE_VERSION,
    STREAM_MODE,
)
from stream.middleware.core.complexity_judge import judge_complexity_with_llm
from stream.middleware.core.database import close_database_pool, initialize_database_pool
from stream.middleware.core.ollama_manager import OllamaModelManager
from stream.middleware.core.tier_health import check_all_tiers
from stream.middleware.core.warm_ping import warm_up_all_tiers

logger = logging.getLogger(__name__)


async def startup():
    """
    Application startup logic.

    This function runs once when the server starts.
    It performs all initialization tasks:
    1. Check Ollama models
    2. Initialize database
    3. Run health checks (Local, Lakeshore, Cloud)
    4. Warm up LLM judge (also pre-loads local Ollama model into memory)
    5. Warm ping tiers (server mode only)
    """
    total_start = time.perf_counter()

    logger.info(f"{SERVICE_NAME} v{SERVICE_VERSION} | mode={STREAM_MODE} | debug={DEBUG}")

    # Step 1: Check Ollama models
    logger.info("Checking Ollama models...")
    await _check_ollama_models()

    # Step 2: Initialize database
    logger.info("Initializing database...")
    initialize_database_pool()

    # Step 3: Health checks for all tiers.
    # - Local: checks if Ollama is running and model is installed (no inference)
    # - Lakeshore: checks Globus auth status only (no GPU jobs)
    # - Cloud: makes a real 1-token API call to verify the API key is valid
    logger.info("Checking tier health...")
    check_all_tiers()

    # Step 4: Warm up the LLM judge by running a test classification.
    # This also pre-loads the local Ollama model (llama3.2:3b) into GPU memory,
    # so the user's first message gets a fast response instead of waiting for
    # Ollama to load the model cold (~5-10s).
    logger.info("Warming up LLM judge (pre-loads local model)...")
    judge_complexity_with_llm("warmup test")

    # Step 5: Warm ping tiers (server mode only).
    # In desktop mode this is skipped — the judge warmup above already pre-loaded
    # the local Ollama model, and cloud/lakeshore don't need warm pings.
    # In server mode, the judge runs through the LiteLLM HTTP gateway which
    # doesn't pre-load the Ollama model directly, so we still warm ping local.
    if STREAM_MODE != "desktop":
        logger.info("Warming up inference tiers...")
        await warm_up_all_tiers()

    elapsed = time.perf_counter() - total_start
    logger.info(f"Ready in {elapsed:.1f}s")


async def shutdown():
    """
    Application shutdown logic.

    This function runs once when the server stops.
    It performs cleanup tasks:
    1. Close Globus Compute Executor (AMQP connection)
    2. Close database connections
    """
    logger.info(f"{SERVICE_NAME} shutting down...")

    try:
        import stream.proxy.app as _proxy_app

        if _proxy_app.globus_client is not None:
            _proxy_app.globus_client.shutdown()
    except Exception as e:
        logger.debug(f"Globus client shutdown (best effort): {e}")

    close_database_pool()
    logger.info("Shutdown complete")


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

    if not missing_models:
        logger.info("All Ollama models available")
    else:
        for model in missing_models:
            logger.warning(f"Model {model} not found")
        logger.warning(f"{len(missing_models)} model(s) missing — LOCAL tier may be unavailable")
        if not sys.stdin.isatty():
            logger.warning("To download: docker exec -it stream-ollama ollama pull <model>")
