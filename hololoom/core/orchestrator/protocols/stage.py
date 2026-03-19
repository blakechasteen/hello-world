#!/usr/bin/env python3
"""
Stage Executor Protocols and Base Classes
==========================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes (Days 6-10)

This module defines:
- StageExecutorProtocol: Runtime-checkable protocol for all stage executors
- BaseStageExecutor: ABC with timing, logging, and lifecycle helpers

Philosophy:
    "Executors wrap pure functions, protocols enable polymorphism"
    Each executor encapsulates dependencies in __init__,
    delegates to Phase 1 pure functions in execute().

Benefits:
    1. Encapsulation: Dependencies stored in instance, not passed every call
    2. Polymorphism: Swap implementations via protocol
    3. Lifecycle: Constructor for setup, execute for work
    4. Composition: Pipeline chains executors

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


# ============================================================================
# Stage Executor Protocol
# ============================================================================

@runtime_checkable
class StageExecutorProtocol(Protocol):
    """
    Protocol defining the interface for all stage executors.

    Every stage executor must:
    - Have a unique stage_id (0-9 for standard weaving stages)
    - Have a human-readable stage_name
    - Implement async execute() that transforms context

    The @runtime_checkable decorator enables isinstance() checks:
        >>> isinstance(my_executor, StageExecutorProtocol)
        True

    Example Implementation:
        >>> class MyExecutor:
        ...     stage_id = 1
        ...     stage_name = "My Stage"
        ...
        ...     async def execute(self, ctx: WeavingContext) -> WeavingContext:
        ...         # Transform context
        ...         return ctx
    """

    stage_id: int
    """Unique identifier for this stage (0-9 for standard stages)."""

    stage_name: str
    """Human-readable name for logging and tracing."""

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        """
        Execute this stage and return updated context.

        Args:
            ctx: WeavingContext with accumulated pipeline state

        Returns:
            WeavingContext with this stage's contributions added

        Note:
            Implementations should:
            - Read only what they need from ctx
            - Write only their outputs to ctx
            - Handle errors gracefully (add to ctx.errors)
            - Record timing (add to ctx.stage_timings)
        """
        ...


# ============================================================================
# Base Stage Executor (ABC)
# ============================================================================

class BaseStageExecutor(ABC):
    """
    Abstract base class with common functionality for all stage executors.

    Provides:
    - Timing helpers (_start_timing, _record_timing)
    - Logging helpers (_log_stage_start, _log_stage_complete)
    - Optional stage event emission for monitoring

    Subclasses must:
    - Set stage_id and stage_name class attributes
    - Implement execute() method

    Example:
        >>> class PatternSelectionExecutor(BaseStageExecutor):
        ...     stage_id = 1
        ...     stage_name = "Loom Command"
        ...
        ...     def __init__(self, loom_command, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.loom_command = loom_command
        ...
        ...     async def execute(self, ctx: WeavingContext) -> WeavingContext:
        ...         self._log_stage_start()
        ...         start = self._start_timing()
        ...         # ... do work ...
        ...         self._record_timing(ctx, 'pattern_selection', start)
        ...         self._log_stage_complete(duration)
        ...         return ctx
    """

    # Subclasses must override these
    stage_id: int = 0
    stage_name: str = "Unknown"

    def __init__(
        self,
        logger: logging.Logger | None = None,
        emit_stage_event: Callable[[int, str, float], None] | None = None
    ):
        """
        Initialize base executor with optional logging and monitoring.

        Args:
            logger: Logger instance for stage logging. If None, creates default.
            emit_stage_event: Optional callback for stage event monitoring.
                Signature: (stage_id, stage_name, duration_ms) -> None
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._emit_stage_event = emit_stage_event or (lambda *args: None)

    @abstractmethod
    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        """
        Execute this stage and return updated context.

        Subclasses must implement this method to perform stage-specific work.

        Args:
            ctx: WeavingContext with accumulated pipeline state

        Returns:
            WeavingContext with this stage's contributions added
        """
        ...

    # ========================================================================
    # Timing Helpers
    # ========================================================================

    def _start_timing(self) -> float:
        """
        Start timing a stage operation.

        Returns:
            High-resolution timestamp for duration calculation
        """
        return time.perf_counter()

    def _record_timing(
        self,
        ctx: WeavingContext,
        stage_key: str,
        start: float
    ) -> float:
        """
        Record timing and emit stage event.

        Args:
            ctx: WeavingContext to record timing in
            stage_key: Key for stage_timings dict (e.g., 'pattern_selection')
            start: Start timestamp from _start_timing()

        Returns:
            Duration in milliseconds
        """
        duration_ms = (time.perf_counter() - start) * 1000
        ctx.stage_timings[stage_key] = duration_ms
        self._emit_stage_event(self.stage_id, self.stage_name, duration_ms)
        return duration_ms

    # ========================================================================
    # Logging Helpers
    # ========================================================================

    def _log_stage_start(self) -> None:
        """Log stage start at DEBUG level."""
        self.logger.debug(
            f"[Stage {self.stage_id}] {self.stage_name} starting..."
        )

    def _log_stage_complete(self, duration_ms: float) -> None:
        """
        Log stage completion at DEBUG level.

        Args:
            duration_ms: Stage duration in milliseconds
        """
        self.logger.debug(
            f"[Stage {self.stage_id}] {self.stage_name} completed in {duration_ms:.1f}ms"
        )

    def _log_stage_error(self, error: str) -> None:
        """
        Log stage error at ERROR level.

        Args:
            error: Error message to log
        """
        self.logger.error(
            f"[Stage {self.stage_id}] {self.stage_name} error: {error}"
        )

    def _log_stage_warning(self, warning: str) -> None:
        """
        Log stage warning at WARNING level.

        Args:
            warning: Warning message to log
        """
        self.logger.warning(
            f"[Stage {self.stage_id}] {self.stage_name} warning: {warning}"
        )

    # ========================================================================
    # Context Helpers
    # ========================================================================

    def _add_error(self, ctx: WeavingContext, error: str) -> None:
        """
        Add error to context and log it.

        Args:
            ctx: WeavingContext to add error to
            error: Error message
        """
        ctx.add_error(error)
        self._log_stage_error(error)

    def _add_warning(self, ctx: WeavingContext, warning: str) -> None:
        """
        Add warning to context and log it.

        Args:
            ctx: WeavingContext to add warning to
            warning: Warning message
        """
        ctx.add_warning(warning)
        self._log_stage_warning(warning)


# ============================================================================
# Gate Executor (ABC for pre-pipeline gates)
# ============================================================================

class GateExecutor(BaseStageExecutor):
    """
    Abstract base class for pipeline gates that run before the weaving stages.

    Gates evaluate a query and either:
    - Return None to pass through to the next gate / full pipeline
    - Return a result (e.g., Spacetime) to short-circuit the pipeline

    Subclasses must:
    - Set stage_id, stage_name, and timing_key class attributes
    - Implement evaluate() method

    The execute() method delegates to evaluate(), handling timing automatically.

    Example:
        >>> class MyCacheGate(GateExecutor):
        ...     stage_id = -1
        ...     stage_name = "Cache Check"
        ...     timing_key = "cache_check"
        ...
        ...     async def evaluate(self, ctx: WeavingContext) -> Optional[Any]:
        ...         cached = self.cache.get(ctx.current_query_text)
        ...         return cached  # None = miss (continue), value = hit (short-circuit)
    """

    timing_key: str = "gate"

    @abstractmethod
    async def evaluate(self, ctx: WeavingContext) -> Any | None:
        """
        Evaluate whether this gate short-circuits the pipeline.

        Args:
            ctx: WeavingContext with accumulated pipeline state

        Returns:
            None to continue the pipeline, or a result to short-circuit
        """
        ...

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        """
        Execute the gate by delegating to evaluate() with timing.

        If evaluate() returns a non-None result, it is stored as
        ctx.gate_result for the orchestrator to handle.
        """
        self._log_stage_start()
        start = self._start_timing()

        result = await self.evaluate(ctx)
        if result is not None:
            ctx.gate_result = result

        duration_ms = self._record_timing(ctx, self.timing_key, start)
        self._log_stage_complete(duration_ms)
        return ctx


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'StageExecutorProtocol',
    'BaseStageExecutor',
    'GateExecutor',
]
