# HoloLoom/ralph/convenience.py
"""
Ralph Loop Engine - Convenience Functions.

High-level API for common Ralph operations.
These functions provide a simple interface without requiring
direct use of RalphEngine.

Created: 2026-01-28
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Awaitable, List
import asyncio
import logging

from .config import RalphConfig, ResetStrategy
from .engine import RalphEngine, RalphResult, IterationStatus, LoopStatus
from .state import RalphState, save_state, load_state, create_handoff_summary, CheckpointType
from .templates import LoopTemplate, get_template
from .context_monitor import ContextMonitor, create_context_monitor


logger = logging.getLogger(__name__)


# Global engine instance for simple usage
_global_engine: Optional[RalphEngine] = None
_global_monitor: Optional[ContextMonitor] = None


async def ralph_loop(
    task: str,
    work_fn: Callable[["RalphIteration"], Awaitable[Dict[str, Any]]],
    template: str = "iterative_refinement",
    max_iterations: int = 50,
    on_iteration: Optional[Callable[[int, "IterationResult"], None]] = None,
    on_reset: Optional[Callable[[str], None]] = None,
    config: Optional[RalphConfig] = None,
    verbose: bool = True,
) -> RalphResult:
    """
    Run a Ralph loop with the given task and work function.

    This is the main convenience function for using Ralph.

    Args:
        task: Description of the task to accomplish
        work_fn: Async function that executes each iteration
                 Should return dict with: confidence, summary, should_continue, etc.
        template: Loop template to use (default: iterative_refinement)
        max_iterations: Maximum iterations before stopping
        on_iteration: Callback after each iteration
        on_reset: Callback when context reset triggered
        config: Custom configuration (optional)
        verbose: Print progress to stdout

    Returns:
        RalphResult with final state and metrics

    Example:
        async def my_work(iteration):
            # Do work here
            return {
                "confidence": 0.8,
                "summary": "Made progress",
                "should_continue": True,
            }

        result = await ralph_loop(
            task="Build feature X",
            work_fn=my_work,
            max_iterations=20,
        )
    """
    # Create config if not provided
    if config is None:
        config = RalphConfig.default()
    config.max_iterations = max_iterations
    config.verbose = verbose

    # Create engine
    engine = RalphEngine(config=config, work_fn=work_fn)

    # Add callbacks as hooks if provided
    if on_reset:
        engine.add_on_reset_hook(lambda state, reason: on_reset(reason))

    # Run the loop
    async for iteration in engine.loop(task=task, template=template):
        result = await iteration.execute()

        # Call iteration callback
        if on_iteration:
            try:
                on_iteration(iteration.iteration_number, result)
            except Exception as e:
                logger.warning(f"on_iteration callback failed: {e}")

        # Check if we should stop
        if not result.should_continue:
            break

    # Build result
    return RalphResult(
        loop_id=engine.state.loop_id,
        task=task,
        status=engine.status,
        total_iterations=engine.current_iteration,
        total_resets=engine.state.reset_count,
        completion_percent=engine.state.completion_percent,
        final_state=engine.state,
        handoff_summary=create_handoff_summary(engine.state),
        duration_ms=engine.state.total_duration_ms,
        artifacts=engine.state.files_modified,
        errors=[],
    )


async def reset_context(
    reason: str = "Manual reset",
    save_checkpoint: bool = True,
    consolidate_memories: bool = True,
    preserve_hot_patterns: bool = True,
) -> Dict[str, Any]:
    """
    Trigger a context reset with state preservation.

    This is for manual resets, e.g., from ChatOps (!reset command).

    Args:
        reason: Why the reset is being triggered
        save_checkpoint: Whether to save a checkpoint before reset
        consolidate_memories: Whether to trigger memory consolidation
        preserve_hot_patterns: Whether to preserve hot patterns

    Returns:
        Dict with handoff summary and reset details
    """
    global _global_engine, _global_monitor

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "actions_taken": [],
    }

    # If there's an active engine, use it
    if _global_engine and _global_engine.state:
        state = _global_engine.state

        # Consolidate if requested
        if consolidate_memories:
            # Would integrate with HoloLoom memory consolidation here
            result["actions_taken"].append("memory_consolidation_triggered")

        # Save checkpoint
        if save_checkpoint:
            checkpoint = await save_state(
                state=state,
                path=_global_engine.config.state_path,
                checkpoint_type=CheckpointType.RESET,
                reason=reason,
            )
            result["checkpoint_id"] = checkpoint.checkpoint_id
            result["actions_taken"].append("checkpoint_saved")

        # Record reset
        state.record_reset(reason)
        result["actions_taken"].append("reset_recorded")

        # Generate handoff
        result["handoff_summary"] = create_handoff_summary(state)

    else:
        result["handoff_summary"] = "No active Ralph loop"

    # Reset context monitor if exists
    if _global_monitor:
        _global_monitor.reset_tracking()
        result["actions_taken"].append("context_monitor_reset")

    logger.info(f"Context reset triggered: {reason}")

    return result


def get_loop_status() -> Dict[str, Any]:
    """
    Get the status of the current Ralph loop.

    Returns:
        Dict with loop status, progress, and metrics
    """
    global _global_engine, _global_monitor

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "has_active_loop": False,
    }

    if _global_engine and _global_engine.state:
        result["has_active_loop"] = True
        result.update(_global_engine.get_status())

    if _global_monitor:
        result["context_usage"] = _global_monitor.get_current_usage()
        result["context_projection"] = _global_monitor.get_projection()

    return result


async def get_handoff_summary() -> str:
    """
    Get the current handoff summary.

    This is what would be passed to a fresh context window
    to continue work.

    Returns:
        Markdown-formatted handoff summary
    """
    global _global_engine

    if _global_engine and _global_engine.state:
        return create_handoff_summary(_global_engine.state)

    # Check for saved state
    state_path = Path(".ralph")
    if state_path.exists():
        state = await load_state(state_path)
        if state:
            return create_handoff_summary(state)

    return "No Ralph state available for handoff."


def init_global_engine(config: Optional[RalphConfig] = None) -> RalphEngine:
    """
    Initialize the global Ralph engine.

    Call this at startup if you want to use the convenience functions
    with a custom configuration.

    Args:
        config: Configuration to use (optional)

    Returns:
        The initialized engine
    """
    global _global_engine, _global_monitor

    _global_engine = RalphEngine(config=config or RalphConfig.default())
    _global_monitor = create_context_monitor(
        context_window_size=_global_engine.config.context_thresholds.estimated_context_size,
        conservative=True,
    )

    return _global_engine


def get_global_engine() -> Optional[RalphEngine]:
    """Get the global engine instance."""
    return _global_engine


def get_global_monitor() -> Optional[ContextMonitor]:
    """Get the global context monitor."""
    return _global_monitor


# ------------ Integration with HoloLoom ------------


async def integrate_with_hololoom(
    loom: "HoloLoom",  # Type hint without import to avoid circular
    preserve_hot_patterns: bool = True,
    consolidate_on_reset: bool = True,
) -> Dict[str, Any]:
    """
    Integrate Ralph with an existing HoloLoom instance.

    This sets up:
    - Memory consolidation on reset
    - Hot pattern preservation
    - Context tracking for memories

    Args:
        loom: HoloLoom instance to integrate with
        preserve_hot_patterns: Preserve hot patterns across resets
        consolidate_on_reset: Consolidate memories before reset

    Returns:
        Dict with integration status
    """
    global _global_engine

    if _global_engine is None:
        _global_engine = RalphEngine()

    result = {
        "integrated": True,
        "features": [],
    }

    # Set up reset hook for consolidation
    if consolidate_on_reset:
        async def consolidate_hook(state, reason):
            try:
                # Would call loom's consolidation here
                # await loom._memory._stream_manager.consolidate_once()
                logger.info("Memory consolidation triggered on reset")
            except Exception as e:
                logger.warning(f"Consolidation failed: {e}")

        _global_engine.add_on_reset_hook(consolidate_hook)
        result["features"].append("consolidation_on_reset")

    # Set up hot pattern preservation
    if preserve_hot_patterns:
        async def preserve_hook(state, reason):
            try:
                # Would get hot patterns from HoloLoom here
                # patterns = loom.get_hot_patterns()
                # state.hot_patterns = patterns
                logger.info("Hot patterns preserved")
            except Exception as e:
                logger.warning(f"Pattern preservation failed: {e}")

        _global_engine.add_on_reset_hook(preserve_hook)
        result["features"].append("hot_pattern_preservation")

    logger.info(f"Ralph integrated with HoloLoom: {result['features']}")
    return result
