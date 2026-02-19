"""
Logging configuration for STREAM middleware.

Two output channels (both active simultaneously):

  1. CONSOLE (human-readable):
     HH:MM:SS module       message
     HH:MM:SS module       ⚠ message         (warnings)
     HH:MM:SS module       ✖ message         (errors)

  2. FILE (structured JSON for Splunk/ELK/Datadog):
     {"timestamp": "2026-02-19T14:06:26.123Z", "level": "WARNING", "module": "litellm_direct", ...}

     Written to ~/.stream/logs/stream.log with automatic rotation (5 MB × 3 files).

Third-party noise (LiteLLM, Pydantic, Pika) is suppressed in both channels.
"""

import json
import logging
import logging.handlers
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

# =========================================================================
# STANDARD LOGRECORD ATTRIBUTES (used by JsonFileFormatter to skip them)
# =========================================================================

_STANDARD_ATTRS = {
    "name",
    "msg",
    "args",
    "created",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "thread",
    "threadName",
    "exc_info",
    "exc_text",
    "stack_info",
    "taskName",
    "asctime",
    "color_message",
}


# =========================================================================
# HUMAN-READABLE FORMATTER (console)
# =========================================================================


class CleanFormatter(logging.Formatter):
    """
    Clean, readable log formatter for the console.

    Produces output like:
      11:01:22 lifecycle       STREAM Middleware starting up...
      11:01:24 litellm_direct  ⚠ Relay streaming failed...
      11:01:25 streaming       ✖ Inference failed for lakeshore

    Design decisions:
    - Time only (no date) — you're watching logs in real-time
    - Short module name — just the filename, not the full path
    - Level prefix only for WARNING (⚠) and ERROR (✖) — INFO is the default
    - No extras duplication — correlation_id is already in the message text
    """

    # Level prefixes — only WARNING and above get a visual marker
    LEVEL_PREFIX = {
        logging.WARNING: "⚠ ",
        logging.ERROR: "✖ ",
        logging.CRITICAL: "✖ ",
    }

    def format(self, record):
        # Time: just HH:MM:SS (no date, no milliseconds)
        time_str = self.formatTime(record, "%H:%M:%S")

        # Module: short name only (e.g., "lifecycle" not "stream.middleware.core.lifecycle")
        module = record.name.rsplit(".", 1)[-1] if record.name else "root"

        # Level prefix: only for warnings and errors
        prefix = self.LEVEL_PREFIX.get(record.levelno, "")

        # Message
        msg = record.getMessage()

        # Build the line — no extras (they're already in the message)
        line = f"{time_str} {module:20s} {prefix}{msg}"

        # Append exception info if present
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text

        return line


# =========================================================================
# JSON FORMATTER (file — for Splunk/ELK/Datadog)
# =========================================================================


class JsonFileFormatter(logging.Formatter):
    """
    Structured JSON log formatter for log aggregation platforms.

    Produces one JSON object per line (NDJSON / JSON Lines format):

      {"timestamp": "2026-02-19T14:06:26.123Z", "level": "WARNING",
       "module": "litellm_direct", "logger": "stream.middleware.core.litellm_direct",
       "message": "Falling back to BATCH MODE (relay not reachable).",
       "correlation_id": "06de0394-..."}

    Any extra fields passed via logger.info("msg", extra={...}) are included
    as top-level keys in the JSON object. This makes them searchable/filterable
    in Splunk, Kibana, Datadog, etc.

    No external dependencies — uses only stdlib json.
    """

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "module": record.module,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include any extra fields (correlation_id, model, tier, etc.)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in log_entry:
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


# =========================================================================
# MAIN CONFIGURATION
# =========================================================================


def configure_logging(log_level: str, log_format: str, log_format_type: str = "json"):
    """
    Configure application-wide logging with two handlers:

    1. Console handler — human-readable (CleanFormatter)
    2. File handler — structured JSON (JsonFileFormatter) at ~/.stream/logs/stream.log

    The file handler uses RotatingFileHandler:
    - Max 5 MB per file
    - Keeps 3 backup files (stream.log, stream.log.1, stream.log.2, stream.log.3)
    - Total max disk usage: ~20 MB

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Standard log format string (unused, kept for API compat)
        log_format_type: Format type ("json" or "human")
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    if log_format_type == "json":
        # =====================================================================
        # DOCKER/SERVER MODE: JSON to stdout only (container logs are captured
        # by Docker/Kubernetes and shipped to the log platform directly).
        # =====================================================================
        try:
            from pythonjsonlogger import jsonlogger

            handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                rename_fields={"asctime": "@timestamp", "levelname": "level", "name": "logger"},
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            _suppress_noisy_loggers()
            return logger

        except ImportError:
            print("python-json-logger not installed, falling back to human format", file=sys.stderr)
            # Fall through to human format

    # =========================================================================
    # DESKTOP/DEV MODE: Human console + JSON file
    # =========================================================================

    # Handler 1: Clean console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CleanFormatter())
    logger.addHandler(console_handler)

    # Handler 2: Structured JSON log file
    _add_json_file_handler(logger, log_level)

    _suppress_noisy_loggers()
    return logger


def _add_json_file_handler(logger: logging.Logger, log_level: str):
    """
    Add a RotatingFileHandler that writes structured JSON to ~/.stream/logs/stream.log.

    This runs best-effort — if the directory can't be created or the file can't
    be opened, we skip silently. The console handler is always available.
    """
    try:
        log_dir = Path.home() / ".stream" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "stream.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,  # Keep 3 rotated files
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JsonFileFormatter())
        logger.addHandler(file_handler)
    except Exception:
        # Don't let file logging failure break the app
        pass


def _suppress_noisy_loggers():
    """
    Suppress verbose third-party loggers that clutter the output.

    These loggers produce INFO-level noise on every request:
    - LiteLLM: "completion() model=..." on every call
    - Pika: AMQP connection details (Globus Compute uses this internally)
    - Globus SDK: task submission details
    - httpcore/httpx: connection pool management

    We keep WARNING+ so actual errors still surface.
    """
    for noisy_logger in [
        "LiteLLM",
        "globus_compute_sdk",
        "pika",  # AMQP connection to Globus (extremely verbose)
        "pika.adapters",
        "pika.adapters.utils",
        "pika.adapters.utils.connection_workflow",
        "pika.adapters.utils.io_services_utils",
        "httpcore",
        "httpx",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Suppress Pydantic serialization warnings (litellm version mismatch)
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
