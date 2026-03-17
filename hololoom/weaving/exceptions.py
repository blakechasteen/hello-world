"""
Weaving Exceptions
==================

Domain-specific exceptions for the weaving pipeline.
Built on top of the generic pipeline exception hierarchy.
"""

from hololoom.pipeline.exceptions import PipelineError, StageExecutionError


class WeavingError(PipelineError):
    """Base exception for weaving-specific errors."""
    pass


class StageSkippedError(WeavingError):
    """Raised when a required stage was skipped unexpectedly."""
    pass


class ConvergenceError(WeavingError):
    """Raised when the convergence engine fails to produce a result."""
    pass


class SafetyBlockedError(WeavingError):
    """Raised when safety guardrails block execution."""

    def __init__(self, stage_name: str, reason: str):
        self.stage_name = stage_name
        self.reason = reason
        super().__init__(
            f"Safety blocked stage '{stage_name}': {reason}",
            details={"stage": stage_name, "reason": reason},
        )


__all__ = [
    'WeavingError',
    'StageSkippedError',
    'ConvergenceError',
    'SafetyBlockedError',
    # Re-export from pipeline for convenience
    'PipelineError',
    'StageExecutionError',
]
