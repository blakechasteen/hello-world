"""
Structured logging for Portal components.

Provides JSON-formatted logs suitable for production monitoring.
"""

import json
import logging
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "component"):
            log_data["component"] = record.component
        if hasattr(record, "node_id"):
            log_data["node_id"] = record.node_id
        if hasattr(record, "job_id"):
            log_data["job_id"] = record.job_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    component: str | None = None
) -> None:
    """
    Configure logging for a Portal component.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting (True for production)
        component: Component name to include in logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    root_logger.addHandler(handler)


def get_logger(name: str, component: str | None = None) -> logging.Logger:
    """
    Get a logger with optional component context.

    Args:
        name: Logger name (usually __name__)
        component: Component name (portal, node, shuttle)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Create adapter to add component context
    if component:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.component = component
            return record

        logging.setLogRecordFactory(record_factory)

    return logger
