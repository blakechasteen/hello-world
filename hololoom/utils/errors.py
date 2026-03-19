"""
HoloLoom Exception Hierarchy
==============================
Central base exceptions for the HoloLoom system.

Domain modules define their own subclasses (PipelineError, ShuttleError, etc.)
but all should ultimately inherit from HoloLoomError so callers can catch
broadly when needed.
"""



class HoloLoomError(Exception):
    """Base exception for all HoloLoom errors.

    Attributes:
        message: Human-readable error description.
        details: Optional structured context for logging/debugging.
    """

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigError(HoloLoomError):
    """Invalid or missing configuration."""
    pass


class BackendError(HoloLoomError):
    """A storage or compute backend is unavailable or failed."""
    pass


class TimeoutError(HoloLoomError):
    """An operation exceeded its time budget."""
    pass
