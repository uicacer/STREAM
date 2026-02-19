"""
SQLite database backend for desktop mode.

WHY SQLITE INSTEAD OF POSTGRESQL:
----------------------------------
PostgreSQL is a powerful database SERVER — it runs as a separate process,
needs configuration (user, password, port), and takes ~100MB of disk.
It's great for multi-user servers, but overkill for a single-user desktop app.

SQLite is a file-based database built into Python's standard library:
    PostgreSQL: Your App → TCP connection → PostgreSQL server → disk
    SQLite:     Your App → direct function call → disk file

No server, no network, no configuration. Just a file at ~/.stream/data/costs.db.

WHAT THIS MODULE PROVIDES:
--------------------------
The same interface as database.py (initialize, get_pool, is_available, close)
plus a log_cost() function for recording chat costs. In server mode, the
LiteLLM server handles cost logging automatically to PostgreSQL. In desktop
mode, there's no LiteLLM server, so we log costs ourselves here.

THREAD SAFETY:
--------------
FastAPI runs request handlers concurrently. SQLite supports this with:
1. check_same_thread=False — allows multiple threads to share one connection
2. WAL (Write-Ahead Logging) mode — allows concurrent reads while writing
   (default SQLite blocks ALL reads during a write; WAL fixes this)
"""

import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level SQLite connection (shared across the app)
_conn: sqlite3.Connection | None = None

# Path to the SQLite database file
_db_path: Path | None = None


def initialize_sqlite() -> bool:
    """
    Create the SQLite database at ~/.stream/data/costs.db.

    Creates the directory structure and costs table if they don't exist.
    Called once at application startup (same timing as PostgreSQL pool init).

    Returns:
        True if successful, False if failed
    """
    global _conn, _db_path

    try:
        # Create the user data directory (~/.stream/data/)
        # parents=True creates all intermediate directories
        # exist_ok=True doesn't error if directory already exists
        _db_path = Path.home() / ".stream" / "data" / "costs.db"
        _db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open SQLite database (creates the file if it doesn't exist)
        # check_same_thread=False: allow FastAPI's async threads to share this connection
        _conn = sqlite3.connect(str(_db_path), check_same_thread=False)

        # Enable WAL (Write-Ahead Logging) mode for better concurrent access.
        # Without WAL, SQLite locks the entire database during writes, blocking reads.
        # With WAL, readers and writers can operate simultaneously.
        _conn.execute("PRAGMA journal_mode=WAL")

        # Return query results as sqlite3.Row objects (access columns by name)
        # Without this, results are plain tuples and you'd have to remember column order.
        _conn.row_factory = sqlite3.Row

        # Create the costs table if it doesn't exist.
        # This schema mirrors what costs.py expects (modeled after LiteLLM_SpendLogs):
        #   model          — which AI model was used (e.g., "cloud-claude")
        #   spend          — total cost in USD for this request
        #   prompt_tokens  — number of input tokens (the conversation sent to the model)
        #   completion_tokens — number of output tokens (the model's response)
        #   start_time     — when the request was made
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS spend_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                spend REAL DEFAULT 0.0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _conn.commit()

        logger.info(f"SQLite database initialized at {_db_path}")
        return True

    except Exception as e:
        logger.error(f"SQLite initialization failed: {e}", exc_info=True)
        logger.warning("Cost tracking will be disabled")
        _conn = None
        return False


def log_cost(
    model: str,
    spend: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """
    Record a cost entry after a chat completion finishes.

    In server/Docker mode, the LiteLLM server automatically logs costs to
    PostgreSQL. In desktop mode, there's no LiteLLM server, so we call
    this function ourselves from streaming.py after each request completes.

    Args:
        model: The model that was used (e.g., "cloud-claude", "local-llama")
        spend: Total cost in USD (0.0 for free models like Ollama)
        prompt_tokens: Number of input tokens sent to the model
        completion_tokens: Number of output tokens the model generated
    """
    if _conn is None:
        return

    try:
        _conn.execute(
            """
            INSERT INTO spend_logs (model, spend, prompt_tokens, completion_tokens, start_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            # SQLite uses ? for parameter placeholders (PostgreSQL uses %s)
            (model, spend, prompt_tokens, completion_tokens, datetime.now(UTC).isoformat()),
        )
        _conn.commit()
    except Exception as e:
        # Don't crash the app if cost logging fails — it's not critical
        logger.error(f"Failed to log cost: {e}")


def get_cost_summary(days: int) -> dict:
    """
    Query cost summary for the last N days.

    Returns the same dict structure that costs.py builds from PostgreSQL,
    so the API response format is identical regardless of database backend.

    Args:
        days: Number of days to look back (e.g., 7 for last week)

    Returns:
        Dict with period info, per-model breakdown, and totals
    """
    if _conn is None:
        return {
            "period_days": days,
            "models": [],
            "total_cost": 0.0,
            "total_requests": 0,
        }

    start_date = datetime.now(UTC) - timedelta(days=days)

    cursor = _conn.execute(
        """
        SELECT
            model,
            COUNT(*) as requests,
            SUM(spend) as total_cost,
            SUM(prompt_tokens) as input_tokens,
            SUM(completion_tokens) as output_tokens
        FROM spend_logs
        WHERE start_time >= ?
        GROUP BY model
        ORDER BY total_cost DESC
        """,
        # SQLite uses ? instead of PostgreSQL's %s for parameter placeholders
        (start_date.isoformat(),),
    )

    results = cursor.fetchall()

    summary = {
        "period_days": days,
        "start_date": start_date.isoformat(),
        "end_date": datetime.now(UTC).isoformat(),
        "models": [],
        "total_cost": 0.0,
        "total_requests": 0,
    }

    for row in results:
        model_name = row["model"]
        request_count = row["requests"]
        model_cost = float(row["total_cost"]) if row["total_cost"] else 0.0
        input_tokens = row["input_tokens"] or 0
        output_tokens = row["output_tokens"] or 0
        avg_cost = model_cost / request_count if request_count > 0 else 0.0

        summary["models"].append(
            {
                "model": model_name,
                "requests": request_count,
                "cost": model_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "avg_cost_per_request": avg_cost,
            }
        )
        summary["total_cost"] += model_cost
        summary["total_requests"] += request_count

    summary["avg_cost_per_request"] = (
        summary["total_cost"] / summary["total_requests"] if summary["total_requests"] > 0 else 0.0
    )

    return summary


def is_sqlite_available() -> bool:
    """Check if SQLite database is initialized and ready."""
    return _conn is not None


def close_sqlite() -> None:
    """Close the SQLite connection. Called once at application shutdown."""
    global _conn

    if _conn:
        _conn.close()
        logger.info(f"SQLite database closed ({_db_path})")
        _conn = None
