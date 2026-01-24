"""
Database connection management.

This module provides centralized database connection pooling for STREAM.
All database access should go through this module to ensure:
- Efficient connection reuse
- Proper connection lifecycle management
- Consistent error handling
"""

import logging
import os

from psycopg2 import pool

logger = logging.getLogger(__name__)

# Global connection pool (initialized at startup)
_db_pool = None


def initialize_database_pool():
    """
    Initialize the PostgreSQL connection pool.

    This should be called ONCE at application startup.
    Creates a connection pool that can be shared across all modules.

    Returns:
        bool: True if successful, False if failed

    Raises:
        ValueError: If required environment variables are missing
    """
    global _db_pool

    try:
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

        # Create connection pool
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
    Get the database connection pool.

    Returns:
        SimpleConnectionPool or None: The pool if available, None if not initialized

    Example:
        >>> pool = get_database_pool()
        >>> if pool:
        ...     conn = pool.getconn()
        ...     # Use connection
        ...     pool.putconn(conn)
    """
    return _db_pool


def is_database_available() -> bool:
    """
    Check if database is available.

    Returns:
        bool: True if pool is initialized, False otherwise
    """
    return _db_pool is not None


def close_database_pool():
    """
    Close all database connections.

    This should be called ONCE at application shutdown.
    Closes all connections in the pool gracefully.
    """
    global _db_pool

    if _db_pool:
        _db_pool.closeall()
        logger.info("✅ Database connection pool closed")
        _db_pool = None
    else:
        logger.debug("Database pool already closed or never initialized")
