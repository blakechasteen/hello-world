# HoloLoom/ralph/hooks.py
"""
Ralph Loop Engine - Hook System.

Hooks allow extending Ralph behavior at key points:
- Pre-iteration: Before each iteration starts
- Post-iteration: After each iteration completes
- On-reset: When context reset is triggered

Created: 2026-01-28
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable, TYPE_CHECKING
import asyncio
import logging

if TYPE_CHECKING:
    from .engine import RalphIteration, IterationResult
    from .state import RalphState


logger = logging.getLogger(__name__)


class RalphHook(ABC):
    """Base class for Ralph hooks."""

    name: str = "base_hook"
    description: str = ""
    enabled: bool = True

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute the hook. Returns optional data to pass forward."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"


class PreIterationHook(RalphHook):
    """Hook that runs before each iteration."""

    @abstractmethod
    async def execute(
        self,
        iteration: "RalphIteration",
    ) -> Optional[Dict[str, Any]]:
        """
        Execute before iteration starts.

        Args:
            iteration: The iteration about to run

        Returns:
            Optional dict of data to inject into iteration context
        """
        pass


class PostIterationHook(RalphHook):
    """Hook that runs after each iteration."""

    @abstractmethod
    async def execute(
        self,
        iteration: "RalphIteration",
        result: "IterationResult",
    ) -> Optional[Dict[str, Any]]:
        """
        Execute after iteration completes.

        Args:
            iteration: The iteration that ran
            result: The iteration's result

        Returns:
            Optional dict of data (e.g., for metrics collection)
        """
        pass


class OnResetHook(RalphHook):
    """Hook that runs when context reset is triggered."""

    @abstractmethod
    async def execute(
        self,
        state: "RalphState",
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute when context reset is triggered.

        This is the last chance to do cleanup before context window resets.

        Args:
            state: Current loop state
            reason: Why reset was triggered

        Returns:
            Optional dict of data to include in handoff
        """
        pass


# ------------ Built-in Hooks ------------


class StateCheckpointHook(PostIterationHook):
    """Creates checkpoints at regular intervals."""

    name = "state_checkpoint"
    description = "Creates state checkpoints every N iterations"

    def __init__(self, interval: int = 5):
        self.interval = interval

    async def execute(
        self,
        iteration: "RalphIteration",
        result: "IterationResult",
    ) -> Optional[Dict[str, Any]]:
        if iteration.iteration_number % self.interval == 0:
            logger.debug(f"Checkpoint hook: iteration {iteration.iteration_number}")
            return {"checkpoint_created": True, "iteration": iteration.iteration_number}
        return None


class MemoryConsolidationHook(PostIterationHook):
    """Triggers memory consolidation at regular intervals."""

    name = "memory_consolidation"
    description = "Consolidates HoloLoom memories every N iterations"

    def __init__(self, interval: int = 5, consolidator=None):
        self.interval = interval
        self.consolidator = consolidator

    async def execute(
        self,
        iteration: "RalphIteration",
        result: "IterationResult",
    ) -> Optional[Dict[str, Any]]:
        if iteration.iteration_number % self.interval == 0:
            if self.consolidator:
                try:
                    stats = await self.consolidator.consolidate_once()
                    logger.debug(f"Memory consolidation: {stats}")
                    return {"consolidation_stats": stats}
                except Exception as e:
                    logger.warning(f"Memory consolidation failed: {e}")
            else:
                logger.debug("Memory consolidation hook: no consolidator configured")
        return None


class ProgressLoggingHook(PostIterationHook):
    """Logs progress after each iteration."""

    name = "progress_logging"
    description = "Logs iteration progress to stdout/file"

    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose

    async def execute(
        self,
        iteration: "RalphIteration",
        result: "IterationResult",
    ) -> Optional[Dict[str, Any]]:
        msg = (
            f"[Ralph] Iteration {iteration.iteration_number}: "
            f"{result.status.value} ({result.duration_ms:.0f}ms) "
            f"confidence={result.confidence:.2f}"
        )

        if self.verbose:
            print(msg)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{datetime.utcnow().isoformat()} {msg}\n")

        logger.info(msg)
        return None


class HotPatternPreservationHook(OnResetHook):
    """Preserves hot patterns across context resets."""

    name = "hot_pattern_preservation"
    description = "Saves hot patterns to state before reset"

    def __init__(self, hot_tracker=None, max_patterns: int = 20):
        self.hot_tracker = hot_tracker
        self.max_patterns = max_patterns

    async def execute(
        self,
        state: "RalphState",
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        if self.hot_tracker:
            try:
                hot_patterns = self.hot_tracker.get_hot_patterns(limit=self.max_patterns)
                state.hot_patterns = [str(p) for p in hot_patterns]
                logger.debug(f"Preserved {len(hot_patterns)} hot patterns")
                return {"patterns_preserved": len(hot_patterns)}
            except Exception as e:
                logger.warning(f"Hot pattern preservation failed: {e}")
        return None


class ContextUsageTrackingHook(PreIterationHook):
    """Tracks context window usage before each iteration."""

    name = "context_usage_tracking"
    description = "Estimates context usage before iteration"

    def __init__(self, estimator=None):
        self.estimator = estimator
        self._last_estimate = 0.0

    async def execute(
        self,
        iteration: "RalphIteration",
    ) -> Optional[Dict[str, Any]]:
        if self.estimator:
            try:
                usage = await self.estimator.estimate_usage()
                self._last_estimate = usage
                return {"context_usage_percent": usage}
            except Exception as e:
                logger.warning(f"Context usage estimation failed: {e}")
        return None


class GitCommitHook(PostIterationHook):
    """Creates git commits after iterations with file changes."""

    name = "git_commit"
    description = "Creates git commits for file changes"

    def __init__(self, commit_template: str = "ralph: iteration {iteration} - {summary}"):
        self.commit_template = commit_template

    async def execute(
        self,
        iteration: "RalphIteration",
        result: "IterationResult",
    ) -> Optional[Dict[str, Any]]:
        if result.files_modified:
            import subprocess

            try:
                # Stage modified files
                for f in result.files_modified:
                    subprocess.run(["git", "add", f], capture_output=True)

                # Create commit
                message = self.commit_template.format(
                    iteration=iteration.iteration_number,
                    summary=result.summary[:50] if result.summary else "progress",
                )
                subprocess.run(["git", "commit", "-m", message], capture_output=True)

                logger.debug(f"Git commit created: {message}")
                return {"git_commit": message}
            except Exception as e:
                logger.warning(f"Git commit failed: {e}")
        return None


class MetricsCollectionHook(PostIterationHook):
    """Collects metrics for monitoring/dashboards."""

    name = "metrics_collection"
    description = "Collects metrics for Prometheus/monitoring"

    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []

    async def execute(
        self,
        iteration: "RalphIteration",
        result: "IterationResult",
    ) -> Optional[Dict[str, Any]]:
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": iteration.iteration_number,
            "status": result.status.value,
            "duration_ms": result.duration_ms,
            "confidence": result.confidence,
            "actions_count": len(result.actions_taken),
            "files_count": len(result.files_modified),
        }
        self.metrics.append(metric)
        return metric

    def get_metrics(self) -> List[Dict[str, Any]]:
        return self.metrics.copy()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.metrics:
            return ""

        lines = [
            "# HELP ralph_iteration_duration_ms Duration of Ralph iterations",
            "# TYPE ralph_iteration_duration_ms histogram",
        ]

        for m in self.metrics[-100:]:  # Last 100 iterations
            lines.append(
                f'ralph_iteration_duration_ms{{iteration="{m["iteration"]}"}} {m["duration_ms"]}'
            )

        return "\n".join(lines)


# ------------ Hook Registry ------------


@dataclass
class HookRegistry:
    """
    Registry for managing Ralph hooks.

    Hooks are organized by type (pre, post, reset) and can be
    enabled/disabled dynamically.
    """

    pre_iteration: List[PreIterationHook] = field(default_factory=list)
    post_iteration: List[PostIterationHook] = field(default_factory=list)
    on_reset: List[OnResetHook] = field(default_factory=list)

    def register_pre(self, hook: PreIterationHook):
        """Register a pre-iteration hook."""
        self.pre_iteration.append(hook)

    def register_post(self, hook: PostIterationHook):
        """Register a post-iteration hook."""
        self.post_iteration.append(hook)

    def register_reset(self, hook: OnResetHook):
        """Register an on-reset hook."""
        self.on_reset.append(hook)

    def get_enabled_pre(self) -> List[PreIterationHook]:
        """Get all enabled pre-iteration hooks."""
        return [h for h in self.pre_iteration if h.enabled]

    def get_enabled_post(self) -> List[PostIterationHook]:
        """Get all enabled post-iteration hooks."""
        return [h for h in self.post_iteration if h.enabled]

    def get_enabled_reset(self) -> List[OnResetHook]:
        """Get all enabled on-reset hooks."""
        return [h for h in self.on_reset if h.enabled]

    def enable(self, hook_name: str):
        """Enable a hook by name."""
        for hook in self.pre_iteration + self.post_iteration + self.on_reset:
            if hook.name == hook_name:
                hook.enabled = True
                return True
        return False

    def disable(self, hook_name: str):
        """Disable a hook by name."""
        for hook in self.pre_iteration + self.post_iteration + self.on_reset:
            if hook.name == hook_name:
                hook.enabled = False
                return True
        return False

    def list_hooks(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all registered hooks."""
        return {
            "pre_iteration": [
                {"name": h.name, "description": h.description, "enabled": h.enabled}
                for h in self.pre_iteration
            ],
            "post_iteration": [
                {"name": h.name, "description": h.description, "enabled": h.enabled}
                for h in self.post_iteration
            ],
            "on_reset": [
                {"name": h.name, "description": h.description, "enabled": h.enabled}
                for h in self.on_reset
            ],
        }


def create_default_registry(
    checkpoint_interval: int = 5,
    consolidation_interval: int = 5,
    verbose: bool = True,
    log_file: Optional[str] = None,
) -> HookRegistry:
    """Create a registry with default hooks configured."""
    registry = HookRegistry()

    # Pre-iteration hooks
    registry.register_pre(ContextUsageTrackingHook())

    # Post-iteration hooks
    registry.register_post(StateCheckpointHook(interval=checkpoint_interval))
    registry.register_post(MemoryConsolidationHook(interval=consolidation_interval))
    registry.register_post(ProgressLoggingHook(verbose=verbose, log_file=log_file))
    registry.register_post(MetricsCollectionHook())

    # On-reset hooks
    registry.register_reset(HotPatternPreservationHook())

    return registry
