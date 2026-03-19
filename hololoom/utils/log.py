"""
HoloLoom Logging Utility
=========================
Consistent logger factory for the HoloLoom package.

Usage:
    from hololoom.utils.log import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__).
        level: Default log level if no handler is configured.

    Returns:
        Configured logging.Logger instance.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    # Only configure root handler once to avoid duplicate output
    if not _CONFIGURED:
        root = logging.getLogger("hololoom")
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            root.addHandler(handler)
            root.setLevel(level)
        _CONFIGURED = True

    return logger
