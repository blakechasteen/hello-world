# HoloLoom/ralph/engine.py
"""
Ralph Loop Engine - Main Orchestrator.

The core iteration loop that embodies the Ralph philosophy:
"Iteration beats perfection."

Each iteration:
1. Load state from persistent storage
2. Execute work (weaving, reasoning, etc.)
3. Save progress
4. Check completion / context usage
5. Either continue or trigger reset

Created: 2026-01-28
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncIterator, Callable, Awaitable
import asyncio
import hashlib
import logging
import time
import traceback

from .config import RalphConfig, ResetStrategy
from .state import (
    RalphState,
    StateCheckpoint,
    IterationRecord,
    TaskProgress,
    CheckpointType,
    save_state,
    load_state,
    create_handoff_summary,
)


logger = logging.getLogger(__name__)


class IterationStatus(Enum):
    """Status of a single iteration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    RESET_TRIGGERED = "reset_triggered"


class LoopStatus(Enum):
    """Status of the overall loop."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    RESET_PENDING = "reset_pending"


@dataclass
class RalphResult:
    """Result of a complete Ralph loop execution."""

    loop_id: str
    task: str
    status: LoopStatus
    total_iterations: int
    total_resets: int
    completion_percent: float
    final_state: RalphState
    handoff_summary: str
    duration_ms: float
    artifacts: List[str]  # Files created/modified
    errors: List[str]

    def is_complete(self) -> bool:
        return self.status == LoopStatus.COMPLETED

    def is_failed(self) -> bool:
        return self.status == LoopStatus.FAILED

    def needs_reset(self) -> bool:
        return self.status == LoopStatus.RESET_PENDING


@dataclass
class IterationResult:
    """Result of a single iteration."""

    iteration_number: int
    status: IterationStatus
    duration_ms: float
    actions_taken: List[str]
    files_modified: List[str]
    confidence: float
    summary: str
    error: Optional[str] = None
    should_continue: bool = True
    reset_triggered: bool = False
    context_usage_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RalphIteration:
    """
    Represents a single iteration in the Ralph loop.

    Provides context for executing work and tracking results.
    """

    engine: "RalphEngine"
    iteration_number: int
    state: RalphState
    started_at: datetime = field(default_factory=datetime.utcnow)

    _result: Optional[IterationResult] = field(default=None, repr=False)
    _actions: List[str] = field(default_factory=list, repr=False)
    _files: List[str] = field(default_factory=list, repr=False)

    def log_action(self, action: str):
        """Log an action taken during this iteration."""
        self._actions.append(action)

    def log_file(self, filepath: str):
        """Log a file that was modified."""
        self._files.append(filepath)

    async def execute(
        self,
        work_fn: Optional[Callable[["RalphIteration"], Awaitable[Dict[str, Any]]]] = None,
    ) -> IterationResult:
        """
        Execute the iteration's work.

        If work_fn is provided, it will be called with this iteration.
        Otherwise, uses the engine's default work function.
        """
        start_time = time.perf_counter()

        try:
            # Get work function
            fn = work_fn or self.engine._work_fn
            if fn is None:
                raise ValueError("No work function provided")

            # Execute with timeout if configured
            timeout = self.engine.config.iteration_timeout
            if timeout > 0:
                result_data = await asyncio.wait_for(fn(self), timeout=timeout)
            else:
                result_data = await fn(self)

            duration_ms = (time.perf_counter() - start_time) * 1000

            self._result = IterationResult(
                iteration_number=self.iteration_number,
                status=IterationStatus.SUCCESS,
                duration_ms=duration_ms,
                actions_taken=self._actions.copy(),
                files_modified=self._files.copy(),
                confidence=result_data.get("confidence", 0.5),
                summary=result_data.get("summary", ""),
                should_continue=result_data.get("should_continue", True),
                reset_triggered=result_data.get("reset_triggered", False),
                context_usage_percent=result_data.get("context_usage_percent", 0.0),
                metadata=result_data.get("metadata", {}),
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._result = IterationResult(
                iteration_number=self.iteration_number,
                status=IterationStatus.TIMEOUT,
                duration_ms=duration_ms,
                actions_taken=self._actions.copy(),
                files_modified=self._files.copy(),
                confidence=0.0,
                summary="Iteration timed out",
                error=f"Timeout after {timeout}s",
                should_continue=True,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            self._result = IterationResult(
                iteration_number=self.iteration_number,
                status=IterationStatus.ERROR,
                duration_ms=duration_ms,
                actions_taken=self._actions.copy(),
                files_modified=self._files.copy(),
                confidence=0.0,
                summary=f"Error: {error_msg}",
                error=f"{error_msg}\n{traceback.format_exc()}",
                should_continue=self.engine.config.continue_on_error,
            )
            logger.error(f"Iteration {self.iteration_number} failed: {error_msg}")

        return self._result

    @property
    def result(self) -> Optional[IterationResult]:
        return self._result


class RalphEngine:
    """
    The Ralph Loop Engine.

    Orchestrates iterative execution with state persistence, context monitoring,
    and automatic reset handling.

    Usage:
        engine = RalphEngine(config)

        # As async iterator
        async for iteration in engine.loop(task="Build feature X"):
            result = await iteration.execute(my_work_function)
            if result.is_complete:
                break

        # Or with convenience method
        result = await engine.run(task="Build feature X", work_fn=my_work)
    """

    def __init__(
        self,
        config: Optional[RalphConfig] = None,
        work_fn: Optional[Callable[["RalphIteration"], Awaitable[Dict[str, Any]]]] = None,
    ):
        self.config = config or RalphConfig.default()
        self._work_fn = work_fn

        # State
        self._state: Optional[RalphState] = None
        self._status: LoopStatus = LoopStatus.PENDING
        self._current_iteration: int = 0
        self._errors: List[str] = []

        # Hooks (will be populated from hooks module)
        self._pre_iteration_hooks: List[Callable] = []
        self._post_iteration_hooks: List[Callable] = []
        self._on_reset_hooks: List[Callable] = []

        # Context monitor (will be set up if auto_monitor enabled)
        self._context_monitor = None

        # Setup logging
        if self.config.verbose:
            logging.basicConfig(level=logging.DEBUG)

    @property
    def state(self) -> Optional[RalphState]:
        return self._state

    @property
    def status(self) -> LoopStatus:
        return self._status

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    def add_pre_iteration_hook(self, hook: Callable):
        """Add a hook to run before each iteration."""
        self._pre_iteration_hooks.append(hook)

    def add_post_iteration_hook(self, hook: Callable):
        """Add a hook to run after each iteration."""
        self._post_iteration_hooks.append(hook)

    def add_on_reset_hook(self, hook: Callable):
        """Add a hook to run when context reset is triggered."""
        self._on_reset_hooks.append(hook)

    async def _initialize_state(self, task: str, template: str) -> RalphState:
        """Initialize or load existing state."""
        state_path = self.config.state_path

        # Try to load existing state
        existing_state = await load_state(state_path)
        if existing_state and existing_state.task == task and existing_state.status != "completed":
            logger.info(f"Resuming loop {existing_state.loop_id} from iteration {existing_state.current_iteration}")
            return existing_state

        # Create new state
        loop_id = hashlib.sha256(f"{task}:{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]

        return RalphState(
            loop_id=loop_id,
            task=task,
            template=template,
            created_at=datetime.utcnow().isoformat(),
            status="running",
        )

    async def _run_pre_hooks(self, iteration: RalphIteration):
        """Run pre-iteration hooks."""
        for hook in self._pre_iteration_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(iteration)
                else:
                    hook(iteration)
            except Exception as e:
                logger.warning(f"Pre-iteration hook failed: {e}")

    async def _run_post_hooks(self, iteration: RalphIteration, result: IterationResult):
        """Run post-iteration hooks."""
        for hook in self._post_iteration_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(iteration, result)
                else:
                    hook(iteration, result)
            except Exception as e:
                logger.warning(f"Post-iteration hook failed: {e}")

    async def _run_reset_hooks(self, state: RalphState, reason: str):
        """Run on-reset hooks."""
        for hook in self._on_reset_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(state, reason)
                else:
                    hook(state, reason)
            except Exception as e:
                logger.warning(f"On-reset hook failed: {e}")

    async def _save_iteration(self, result: IterationResult):
        """Save state after iteration."""
        if not self.config.auto_save:
            return

        record = IterationRecord(
            iteration_number=result.iteration_number,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=result.duration_ms,
            status=result.status.value,
            actions_taken=result.actions_taken,
            files_modified=result.files_modified,
            confidence=result.confidence,
            error_message=result.error,
            summary=result.summary,
        )
        self._state.add_iteration(record)

        await save_state(
            state=self._state,
            path=self.config.state_path,
            checkpoint_type=CheckpointType.ITERATION,
            context_usage_percent=result.context_usage_percent,
            create_git_commit=self.config.git_commits,
        )

    async def _check_context_threshold(self, result: IterationResult) -> Optional[str]:
        """Check if context usage exceeds threshold."""
        usage = result.context_usage_percent
        thresholds = self.config.context_thresholds

        if usage >= thresholds.critical_percent:
            return f"Critical context usage ({usage:.1%})"
        elif usage >= thresholds.reset_percent:
            return f"Context usage exceeded reset threshold ({usage:.1%})"

        return None

    async def _trigger_reset(self, reason: str) -> StateCheckpoint:
        """Trigger a context reset."""
        logger.info(f"Triggering context reset: {reason}")

        # Run reset hooks
        await self._run_reset_hooks(self._state, reason)

        # Record reset
        self._state.record_reset(reason)
        self._state.status = "reset_pending"

        # Create reset checkpoint
        checkpoint = await save_state(
            state=self._state,
            path=self.config.state_path,
            checkpoint_type=CheckpointType.RESET,
            reason=reason,
            create_git_commit=self.config.git_commits,
        )

        self._status = LoopStatus.RESET_PENDING
        return checkpoint

    async def loop(
        self,
        task: str,
        template: str = "iterative_refinement",
        resume: bool = True,
    ) -> AsyncIterator[RalphIteration]:
        """
        Create an async iterator over loop iterations.

        Usage:
            async for iteration in engine.loop(task="Build X"):
                result = await iteration.execute()
                if not result.should_continue:
                    break
        """
        # Initialize state
        self._state = await self._initialize_state(task, template)
        self._current_iteration = self._state.current_iteration
        self._status = LoopStatus.RUNNING

        while True:
            # Check iteration limit
            if self.config.max_iterations > 0 and self._current_iteration >= self.config.max_iterations:
                logger.info(f"Max iterations ({self.config.max_iterations}) reached")
                break

            # Check consecutive errors
            if self._state.consecutive_errors >= self.config.max_consecutive_errors:
                logger.error(f"Max consecutive errors ({self.config.max_consecutive_errors}) reached")
                self._status = LoopStatus.FAILED
                break

            # Create iteration
            self._current_iteration += 1
            iteration = RalphIteration(
                engine=self,
                iteration_number=self._current_iteration,
                state=self._state,
            )

            # Run pre-hooks
            await self._run_pre_hooks(iteration)

            # Yield iteration for execution
            yield iteration

            # Get result (should have been set by execute())
            result = iteration.result
            if result is None:
                logger.warning(f"Iteration {self._current_iteration} returned no result")
                continue

            # Run post-hooks
            await self._run_post_hooks(iteration, result)

            # Save iteration
            await self._save_iteration(result)

            # Check for completion
            if not result.should_continue:
                if result.status == IterationStatus.SUCCESS:
                    self._status = LoopStatus.COMPLETED
                    self._state.status = "completed"
                else:
                    self._status = LoopStatus.FAILED
                    self._state.status = "failed"
                break

            # Check for reset trigger
            if result.reset_triggered:
                await self._trigger_reset("Explicit reset requested")
                break

            # Check context threshold
            reset_reason = await self._check_context_threshold(result)
            if reset_reason:
                await self._trigger_reset(reset_reason)
                break

            # Delay between iterations
            if self.config.iteration_delay > 0:
                await asyncio.sleep(self.config.iteration_delay)

        # Save final state
        await save_state(
            state=self._state,
            path=self.config.state_path,
            checkpoint_type=CheckpointType.COMPLETE if self._status == LoopStatus.COMPLETED else CheckpointType.ITERATION,
            create_git_commit=self.config.git_commits,
        )

    async def run(
        self,
        task: str,
        work_fn: Callable[["RalphIteration"], Awaitable[Dict[str, Any]]],
        template: str = "iterative_refinement",
    ) -> RalphResult:
        """
        Run the complete loop with the given work function.

        Simpler API for common use cases.
        """
        self._work_fn = work_fn
        start_time = time.perf_counter()

        async for iteration in self.loop(task=task, template=template):
            await iteration.execute()

        duration_ms = (time.perf_counter() - start_time) * 1000

        return RalphResult(
            loop_id=self._state.loop_id,
            task=task,
            status=self._status,
            total_iterations=self._current_iteration,
            total_resets=self._state.reset_count,
            completion_percent=self._state.completion_percent,
            final_state=self._state,
            handoff_summary=create_handoff_summary(self._state),
            duration_ms=duration_ms,
            artifacts=self._state.files_modified,
            errors=self._errors,
        )

    async def pause(self):
        """Pause the loop execution."""
        if self._status == LoopStatus.RUNNING:
            self._status = LoopStatus.PAUSED
            self._state.status = "paused"
            await save_state(
                state=self._state,
                path=self.config.state_path,
                checkpoint_type=CheckpointType.MANUAL,
                reason="User paused",
            )

    async def resume(self):
        """Resume the loop execution."""
        if self._status == LoopStatus.PAUSED:
            self._status = LoopStatus.RUNNING
            self._state.status = "running"

    def get_handoff_summary(self) -> str:
        """Get the current handoff summary for context transfer."""
        if self._state is None:
            return "No active Ralph loop"
        return create_handoff_summary(self._state)

    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        if self._state is None:
            return {"status": "no_active_loop"}

        return {
            "loop_id": self._state.loop_id,
            "task": self._state.task,
            "status": self._status.value,
            "iteration": self._current_iteration,
            "resets": self._state.reset_count,
            "completion_percent": self._state.completion_percent,
            "consecutive_errors": self._state.consecutive_errors,
            "last_update": self._state.updated_at,
        }
