"""
Logging configuration for STREAM middleware.

This module provides centralized logging setup with support for:
- JSON structured logging (production)
- Human-readable logging with extras (development)
- Automatic format selection based on environment
"""

import logging
import sys


class ExtraFormatter(logging.Formatter):
    """Formatter that appends extra fields to log messages."""

    # Standard logging attributes (exclude from extras)
    STANDARD_ATTRS = {
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
        "taskName",  # Python 3.12+
    }

    def format(self, record):
        """Format log record with extra fields appended."""
        # Format base message
        base_msg = super().format(record)

        # Extract extra fields
        extras = {k: v for k, v in record.__dict__.items() if k not in self.STANDARD_ATTRS}

        # Append extras if present
        if extras:
            # Format extras as key=value pairs
            extra_parts = []
            for k, v in extras.items():
                # Handle different types
                # FIX: Use modern isinstance syntax (UP038)
                if isinstance(v, list | dict):
                    extra_parts.append(f"{k}={v}")
                elif isinstance(v, float):
                    extra_parts.append(f"{k}={v:.2f}")
                elif v is None:
                    extra_parts.append(f"{k}=None")
                else:
                    extra_parts.append(f"{k}={v}")

            extra_str = " | " + " ".join(extra_parts)
            return base_msg + extra_str

        return base_msg


def configure_logging(log_level: str, log_format: str, log_format_type: str = "json"):
    """
    Configure application-wide logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Standard log format string
        log_format_type: Format type ("json" or "human")

    Returns:
        logging.Logger: Configured root logger

    Example:
        >>> from stream.middleware.utils.logging_config import configure_logging
        >>> logger = configure_logging("INFO", "%(message)s", "json")
        >>> logger.info("Application started")
    """

    if log_format_type == "json":
        # =====================================================================
        # PRODUCTION: JSON Structured Logging
        # =====================================================================
        try:
            from pythonjsonlogger import jsonlogger

            handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                rename_fields={"asctime": "@timestamp", "levelname": "level", "name": "logger"},
            )
            handler.setFormatter(formatter)

            # Configure root logger
            logger = logging.getLogger()
            logger.setLevel(log_level)
            logger.handlers.clear()  # Remove default handlers
            logger.addHandler(handler)

            print("✅ JSON logging enabled (production mode)", file=sys.stderr)
            return logger

        except ImportError:
            print(
                "⚠️  python-json-logger not installed, falling back to human format", file=sys.stderr
            )
            print("   Install: pip install python-json-logger", file=sys.stderr)
            # Fall through to human format

    # =========================================================================
    # DEVELOPMENT: Human-Readable Logging with Extras
    # =========================================================================
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ExtraFormatter(log_format))

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()  # Remove default handlers
    logger.addHandler(handler)

    print("✅ Human-readable logging enabled (development mode)", file=sys.stderr)
    return logger
