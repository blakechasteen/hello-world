"""
BossPig Logging Configuration

Provides structured logging with multiple output formats and log levels.

Created: 2025-11-22
Status: Week 11 Production Hardening
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# ============================================================================
# JSON Formatter
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for easy parsing and analysis.

    Output format:
    {
      "timestamp": "2025-11-22T10:30:45.123456Z",
      "level": "INFO",
      "logger": "bosspig.detector",
      "message": "Analysis complete",
      "extra": {...}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "detector"):
            log_data["detector"] = record.detector
        if hasattr(record, "document_words"):
            log_data["document_words"] = record.document_words
        if hasattr(record, "findings_count"):
            log_data["findings_count"] = record.findings_count
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "categories"):
            log_data["categories"] = record.categories

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_type: str = "text",
    enable_console: bool = True
) -> logging.Logger:
    """
    Set up BossPig logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (enables file logging)
        format_type: "text" or "json"
        enable_console: Whether to log to console (stdout)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(level="DEBUG", log_file=Path("bosspig.log"))
        >>> logger.info("Analysis started")
    """
    # Get root logger for bosspig
    logger = logging.getLogger("bosspig")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Choose formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        # Text formatter with timestamp, level, logger name, message
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler (stdout)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (with rotation)
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler (10 MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document", extra={"document_words": 1523})
    """
    return logging.getLogger(f"bosspig.{name}")


# ============================================================================
# Logging Helpers
# ============================================================================

def log_analysis_start(
    logger: logging.Logger,
    document_path: Optional[str] = None,
    document_size: int = 0
):
    """Log analysis start event."""
    logger.info(
        "Analysis started",
        extra={
            "document_path": document_path,
            "document_words": document_size
        }
    )


def log_analysis_complete(
    logger: logging.Logger,
    findings_count: int,
    duration_ms: float,
    categories: list
):
    """Log analysis complete event."""
    logger.info(
        f"Analysis complete: {findings_count} findings in {duration_ms:.1f}ms",
        extra={
            "findings_count": findings_count,
            "duration_ms": duration_ms,
            "categories": categories
        }
    )


def log_detector_run(
    logger: logging.Logger,
    detector_name: str,
    findings_count: int,
    duration_ms: float
):
    """Log individual detector run."""
    logger.debug(
        f"{detector_name}: {findings_count} findings in {duration_ms:.1f}ms",
        extra={
            "detector": detector_name,
            "findings_count": findings_count,
            "duration_ms": duration_ms
        }
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[dict] = None
):
    """Log error with context."""
    logger.error(
        f"Error: {str(error)}",
        exc_info=True,
        extra=context or {}
    )


# ============================================================================
# Performance Logging
# ============================================================================

class PerformanceLogger:
    """
    Context manager for logging performance metrics.

    Example:
        >>> with PerformanceLogger(logger, "jargon_detection"):
        ...     # detect jargon
        ...     findings = detector.detect_jargon(text)
        # Automatically logs duration
    """

    def __init__(self, logger: logging.Logger, operation: str):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance
            operation: Operation name
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.utcnow()
        self.logger.debug(f"{self.operation} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration."""
        duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000

        if exc_type is None:
            self.logger.debug(
                f"{self.operation} completed in {duration:.1f}ms",
                extra={"operation": self.operation, "duration_ms": duration}
            )
        else:
            self.logger.error(
                f"{self.operation} failed after {duration:.1f}ms",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={"operation": self.operation, "duration_ms": duration}
            )

        # Don't suppress exceptions
        return False


if __name__ == "__main__":
    # Example usage
    import time

    # Setup logging with JSON format
    logger = setup_logging(
        level="DEBUG",
        log_file=Path("bosspig.log"),
        format_type="json"
    )

    # Log some events
    log_analysis_start(logger, document_path="test.md", document_size=1523)

    # Use performance logger
    with PerformanceLogger(logger, "jargon_detection"):
        time.sleep(0.01)  # Simulate detection work

    # Log completion
    log_analysis_complete(
        logger,
        findings_count=12,
        duration_ms=85.5,
        categories=["jargon", "vague_commitments"]
    )

    print("\nLogging configured. Check bosspig.log for JSON-formatted logs.")
