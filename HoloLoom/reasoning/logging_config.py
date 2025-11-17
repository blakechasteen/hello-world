#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured JSON Logging for Reasoning Engine
==============================================
Machine-readable JSON logs for production monitoring.

Phase 3: Polish

Author: Claude Code
Date: 2025-11-17
"""

import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


# ============================================================================
# JSON Formatter
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for machine readability.

    Output format:
    {
        "timestamp": "2025-11-17T10:30:45.123Z",
        "level": "INFO",
        "logger": "HoloLoom.reasoning.engine",
        "message": "Reasoning complete",
        "module": "engine",
        "function": "reason",
        "line": 245,
        "thread": "MainThread",
        "extra": {
            "mode": "standard",
            "confidence": 0.85,
            "duration_ms": 150.5
        }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields (custom attributes)
        extra = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info"
            ]:
                extra[key] = value

        if extra:
            log_data["extra"] = extra

        return json.dumps(log_data)


# ============================================================================
# Structured Logger
# ============================================================================

class StructuredLogger:
    """
    Wrapper for structured logging with automatic context.

    Usage:
        logger = StructuredLogger("my.module")

        logger.info("Operation complete", extra={
            "operation": "reasoning",
            "duration_ms": 150.5,
            "confidence": 0.85
        })
    """

    def __init__(self, name: str, use_json: bool = False):
        self.logger = logging.getLogger(name)
        self.use_json = use_json

    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Internal logging method."""
        # Merge extra dict with kwargs
        if extra:
            kwargs.update(extra)

        # Log with extra fields
        self.logger.log(level, message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        if exc_info:
            kwargs['exc_info'] = True
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        if exc_info:
            kwargs['exc_info'] = True
        self._log(logging.CRITICAL, message, **kwargs)


# ============================================================================
# Logger Configuration
# ============================================================================

def setup_json_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
):
    """
    Setup JSON logging for the reasoning engine.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logs
        console: Whether to log to console (default: True)

    Usage:
        # Setup JSON logging to file and console
        setup_json_logging(level="INFO", log_file="./logs/reasoning.json")

        # Use standard logging
        logger = logging.getLogger(__name__)
        logger.info("Test message", extra={"key": "value"})
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = JSONFormatter()

    # Console handler (if requested)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.info("JSON logging configured", extra={
        "level": level,
        "log_file": log_file,
        "console": console
    })


def setup_human_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
):
    """
    Setup human-readable logging (standard format).

    Args:
        level: Logging level
        log_file: Optional file path
        console: Whether to log to console

    Format:
        2025-11-17 10:30:45,123 INFO [engine.py:245] Reasoning complete (mode=standard, confidence=0.85)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# ============================================================================
# Context Manager for Scoped Logging
# ============================================================================

class LogContext:
    """
    Context manager for scoped logging with automatic field injection.

    Usage:
        with LogContext(mode="standard", operation="reasoning"):
            logger.info("Starting")  # Automatically includes mode and operation
            # ... do work ...
            logger.info("Complete")  # Also includes mode and operation
    """

    def __init__(self, **context_fields):
        self.context_fields = context_fields
        self.old_factory = None

    def __enter__(self):
        # Save old log record factory
        self.old_factory = logging.getLogRecordFactory()

        # Create new factory that injects context
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context_fields.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old factory
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


# ============================================================================
# Convenience Functions
# ============================================================================

def log_reasoning_operation(
    logger: logging.Logger,
    mode: str,
    duration_ms: float,
    confidence: float,
    chain_length: int,
    success: bool = True
):
    """
    Log reasoning operation with structured fields.

    Args:
        logger: Logger instance
        mode: Reasoning mode
        duration_ms: Duration in milliseconds
        confidence: Confidence score
        chain_length: Number of reasoning steps
        success: Whether operation succeeded
    """
    level = logging.INFO if success else logging.ERROR
    message = "Reasoning complete" if success else "Reasoning failed"

    logger.log(level, message, extra={
        "mode": mode,
        "duration_ms": duration_ms,
        "confidence": confidence,
        "chain_length": chain_length,
        "success": success
    })


def log_resource_usage(
    logger: logging.Logger,
    memory_mb: float,
    active_operations: int,
    peak_operations: int
):
    """Log resource usage statistics."""
    logger.info("Resource usage", extra={
        "memory_mb": memory_mb,
        "active_operations": active_operations,
        "peak_operations": peak_operations
    })


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup JSON logging
    setup_json_logging(level="INFO", log_file="./logs/example.json")

    # Get logger
    logger = logging.getLogger(__name__)

    # Log with structured fields
    logger.info("Operation starting", extra={
        "operation": "reasoning",
        "mode": "standard"
    })

    # Use context for automatic field injection
    with LogContext(operation="reasoning", mode="standard"):
        logger.info("Processing query")
        logger.info("Query processed", extra={"duration_ms": 150.5})

    # Log operation result
    log_reasoning_operation(
        logger,
        mode="standard",
        duration_ms=150.5,
        confidence=0.85,
        chain_length=5,
        success=True
    )

    print("\nExample logs written to ./logs/example.json")
