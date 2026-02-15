"""
Database connection management.

This module provides centralized database access for STREAM.
All database operations should go through this module to ensure:
- Efficient connection reuse
- Proper connection lifecycle management
- Consistent error handling

MODE SWITCHING:
--------------
This module routes to the right database backend based on STREAM_MODE:

    Server/Docker mode (STREAM_MODE="server"):
        Uses PostgreSQL — a full database server running in a Docker container.
        The LiteLLM server writes cost data to PostgreSQL automatically.
        We read from it for the /costs/summary endpoint.

    Desktop mode (STREAM_MODE="desktop"):
        Uses SQLite — a file-based database at ~/.stream/data/costs.db.
        No server process needed. We write cost data ourselves (from streaming.py)
        since there's no LiteLLM server to do it.

The rest of the app calls the same functions (initialize_database_pool,
is_database_available, etc.) regardless of which backend is active.
"""

import logging
import os

from stream.middleware.config import STREAM_MODE
from stream.middleware.core.database_sqlite import (
    close_sqlite,
    initialize_sqlite,
    is_sqlite_available,
)

logger = logging.getLogger(__name__)

# Global connection pool (PostgreSQL only — SQLite manages its own connection)
_db_pool = None


def initialize_database_pool():
    """
    Initialize the database.

    Called ONCE at application startup (from lifecycle.py).
    Routes to SQLite or PostgreSQL based on STREAM_MODE.

    Returns:
        bool: True if successful, False if failed
    """
    # Desktop mode: use SQLite (file-based, no server needed)
    if STREAM_MODE == "desktop":
        return initialize_sqlite()

    # Server mode: use PostgreSQL (Docker container)
    return _initialize_postgres()


def _initialize_postgres():
    """
    Initialize the PostgreSQL connection pool (server/Docker mode only).

    PostgreSQL runs as a separate Docker container. We connect to it using
    credentials from environment variables (set in .env / docker-compose).
    """
    global _db_pool

    try:
        # Import psycopg2 only in server mode — it's not needed for desktop
        # and may not be installed if the user only has SQLite dependencies.
        from psycopg2 import pool

        # Load database credentials from environment variables
        required_vars = {
            "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
            "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
            "POSTGRES_DB": os.getenv("POSTGRES_DB"),
            "POSTGRES_USER": os.getenv("POSTGRES_USER"),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
        }

        # Validate all required variables are present
        missing = [k for k, v in required_vars.items() if v is None]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        # Create connection pool.
        # A pool reuses database connections instead of opening a new one per request.
        # This is much faster and uses fewer resources.
        _db_pool = pool.SimpleConnectionPool(
            minconn=1,  # 5 in production
            maxconn=5,  # 20 in production
            host=required_vars["POSTGRES_HOST"],
            port=int(required_vars["POSTGRES_PORT"]),
            database=required_vars["POSTGRES_DB"],
            user=required_vars["POSTGRES_USER"],
            password=required_vars["POSTGRES_PASSWORD"],
        )

        logger.info(
            "✅ Database connection pool initialized",
            extra={
                "host": required_vars["POSTGRES_HOST"],
                "database": required_vars["POSTGRES_DB"],
                "min_connections": 1,
                "max_connections": 5,
            },
        )

        return True

    except ValueError as e:
        logger.critical(f"❌ Database configuration error: {e}")
        logger.warning("⚠️  Cost tracking will be disabled")
        _db_pool = None
        return False

    except Exception as e:
        logger.critical(f"❌ Database connection failed: {e}", exc_info=True)
        logger.warning("⚠️  Cost tracking will be disabled")
        _db_pool = None
        return False


def get_database_pool():
    """
    Get the database connection pool (PostgreSQL only).

    In desktop mode, this returns None — use database_sqlite functions instead.
    costs.py checks STREAM_MODE to decide which path to take.

    Returns:
        SimpleConnectionPool or None
    """
    return _db_pool


def is_database_available() -> bool:
    """
    Check if the database is available (works for both modes).

    Returns:
        bool: True if the database is ready to use
    """
    if STREAM_MODE == "desktop":
        return is_sqlite_available()
    return _db_pool is not None


def close_database_pool():
    """
    Close all database connections.

    Called ONCE at application shutdown (from lifecycle.py).
    Routes to the right backend based on STREAM_MODE.
    """
    global _db_pool

    if STREAM_MODE == "desktop":
        close_sqlite()
        return

    if _db_pool:
        _db_pool.closeall()
        logger.info("✅ Database connection pool closed")
        _db_pool = None
    else:
        logger.debug("Database pool already closed or never initialized")
