"""
Application lifecycle management.

This module handles startup and shutdown logic for STREAM middleware.
Extracted from app.py to keep the main application file clean.
"""

import logging
import sys

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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

logger = logging.getLogger(__name__)
console = Console()


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

    # Step 1: Check/download Ollama models
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

    logger.info("✅ Middleware ready!")


async def shutdown():
    """
    Application shutdown logic.

    This function runs once when the server stops.
    It performs cleanup tasks:
    1. Close database connections
    2. Log shutdown message
    """
    logger.info(f"👋 {SERVICE_NAME} shutting down...")

    # Stop background health monitor
    health_monitor.stop()

    # Close database pool
    close_database_pool()

    logger.info("✅ Shutdown complete")


async def _check_ollama_models():
    """
    Check if required Ollama models are available.

    In Docker (non-interactive):
        - Logs warnings if models missing
        - Shows instructions for manual download
        - Continues startup (doesn't block)

    In local dev (interactive):
        - Prompts user to download missing models
        - Downloads if user confirms
        - Continues regardless
    """
    manager = OllamaModelManager()
    missing_models = []

    for _, ollama_model in OLLAMA_MODELS.items():
        if not manager.is_model_available(ollama_model):
            missing_models.append(ollama_model)
            logger.warning(f"⚠️  Model {ollama_model} not found")

            size_estimate = manager.get_model_size_estimate(ollama_model)
            logger.warning(f"   Estimated size: {size_estimate}")

            # Check if running in interactive mode (TTY) or Docker
            if sys.stdin.isatty():
                # Interactive mode: ask user to download
                console.print(f"\nModel [bold]{ollama_model}[/bold] not found.")
                console.print(f"Size: [bold]{size_estimate}[/bold]")
                response = (
                    console.input(
                        "Download now? ([bold green]y[/bold green]/[bold red]n[/bold red]): "
                    )
                    .strip()
                    .lower()
                )

                if response in ["y", "yes"]:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        TimeElapsedColumn(),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Downloading [bold]{ollama_model}[/bold]...", start=False
                        )
                        progress.start_task(task)

                        success = manager.pull_model(ollama_model, show_progress=False)

                        if success:
                            progress.update(
                                task,
                                description=f"[green]✓ {ollama_model} downloaded[/green]",
                            )
                            logger.info(f"✅ Downloaded {ollama_model}")
                        else:
                            progress.update(
                                task,
                                description=f"[red]✗ Failed to download {ollama_model}[/red]",
                            )
                            logger.error(f"❌ Failed to download {ollama_model}")
                else:
                    logger.warning(f"   Skipping {ollama_model}")
            else:
                # Docker mode: show instructions
                logger.warning("⚠️  Running in Docker - models should be pre-downloaded")
                logger.warning(
                    f"   To download: docker exec -it stream-ollama ollama pull {ollama_model}"
                )

    if not missing_models:
        logger.info("✅ All Ollama models available")
    else:
        logger.warning(f"⚠️  {len(missing_models)} model(s) not found")
        if not sys.stdin.isatty():
            logger.warning("=" * 70)
            logger.warning("📋 To download missing models:")
            for model in missing_models:
                logger.warning(f"   docker exec -it stream-ollama ollama pull {model}")
            logger.warning("=" * 70)
            logger.warning("⚠️  LOCAL tier unavailable until models downloaded")
