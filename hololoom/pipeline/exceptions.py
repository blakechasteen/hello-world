"""
Pipeline Exceptions
===================

Typed exception hierarchy for the pipeline framework.
Generic — not specific to weaving or any domain.
"""

from hololoom.utils.errors import HoloLoomError


class PipelineError(HoloLoomError):
    """Base exception for all pipeline errors."""
    pass


class CyclicDependencyError(PipelineError):
    """Raised when stage dependencies form a cycle."""
    pass


class WriteConflictError(PipelineError):
    """Raised when concurrent stages at the same level write the same field."""
    pass


class StageExecutionError(PipelineError):
    """Raised when a stage fails during execution."""

    def __init__(self, stage_name: str, cause: Exception):
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(
            f"Stage '{stage_name}' failed: {cause}",
            details={"stage": stage_name, "cause_type": type(cause).__name__},
        )


class UndeclaredReadError(PipelineError):
    """Raised by ReadWriteTracker when a stage reads undeclared fields."""
    pass


class UndeclaredWriteError(PipelineError):
    """Raised by ReadWriteTracker when a stage writes undeclared fields."""
    pass
