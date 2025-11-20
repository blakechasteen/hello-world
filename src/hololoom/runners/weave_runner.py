#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weave Runner
============

Orchestrator that runs the Shuttle and notifies patterns after each query.

This is the entrypoint for "agent-like" behavior:
- Core (Shuttle, Warp, Yarn) stays focused on single weaves
- Patterns handle continual learning, logging, adaptation
- Runner wires them together without coupling

Design Philosophy:
- Core remains dumb & predictable (single weave semantics)
- Patterns are observers (no control over core logic)
- Multiple patterns can compose
- Clean separation of concerns

Created: 2025-01-20
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Protocol, Any
import logging

from hololoom.core.types import WeaveResult
from hololoom.patterns.base import LearningPattern, LifecyclePattern


logger = logging.getLogger(__name__)


class SupportsIntersect(Protocol):
    """
    Minimal interface for a Shuttle-like object.

    Any object that can produce a WeaveResult from a query + warp results
    can be used with WeaveRunner.
    """

    def intersect(self, query: str, warp_results: Any) -> WeaveResult:
        """
        Perform a single weave operation.

        Args:
            query: User query
            warp_results: Results from warp search

        Returns:
            WeaveResult with context blocks and metadata
        """
        ...


@dataclass
class WeaveRunner:
    """
    Orchestrator that runs the Shuttle and notifies patterns after each query.

    This provides the "agent loop" without modifying core:
    1. Shuttle performs single weave (stays clean)
    2. WeaveResult is produced
    3. Patterns observe and learn (optional)

    Example:
        from hololoom.runners import WeaveRunner
        from hololoom.patterns.nested_learning import (
            NestedLearningPattern,
            NoOpBanditModule,
            NoOpGraphModule,
            NoOpPolicyModule
        )

        # Create shuttle (however you currently do it)
        shuttle = Shuttle(warp=warp, yarn=yarn)

        # Create pattern
        pattern = NestedLearningPattern(
            bandit_module=NoOpBanditModule(),
            graph_module=NoOpGraphModule(),
            policy_module=NoOpPolicyModule()
        )

        # Create runner
        runner = WeaveRunner(shuttle=shuttle, patterns=[pattern])

        # Run queries
        warp_results = warp.search("Why is my build failing?", top_k=5)
        result = runner.run_query("Why is my build failing?", warp_results)

        # Patterns automatically notified after each query
    """

    shuttle: SupportsIntersect
    """Shuttle-like object that performs single weaves."""

    patterns: List[LearningPattern] = field(default_factory=list)
    """List of patterns to notify after each weave."""

    enable_lifecycle: bool = True
    """Whether to call lifecycle hooks (on_startup, on_shutdown)."""

    # Internal state
    _started: bool = field(default=False, init=False, repr=False)
    _query_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialize runner and optionally call pattern lifecycle hooks."""
        if self.enable_lifecycle:
            self._call_lifecycle_hooks("on_startup")
        self._started = True

    def run_query(self, query: str, warp_results: Any) -> WeaveResult:
        """
        Run one weave, then dispatch result to all patterns.

        Args:
            query: User query
            warp_results: Results from warp search (format depends on implementation)

        Returns:
            WeaveResult from shuttle

        Notes:
            - Patterns are notified even if they raise exceptions
            - Pattern exceptions are logged but don't stop execution
            - Query count is incremented regardless of success/failure
        """
        if not self._started:
            logger.warning("WeaveRunner not started - call startup() first")

        # Perform single weave (core logic)
        logger.debug(f"Running query #{self._query_count + 1}: {query[:50]}...")
        result: WeaveResult = self.shuttle.intersect(query=query, warp_results=warp_results)

        # Notify patterns (learning logic)
        self._notify_patterns(result)

        # Update state
        self._query_count += 1

        return result

    def _notify_patterns(self, result: WeaveResult) -> None:
        """
        Notify all patterns about completed weave.

        Args:
            result: WeaveResult to dispatch
        """
        for i, pattern in enumerate(self.patterns):
            try:
                pattern.on_episode_end(result)
            except Exception as e:
                # Patterns should not crash the runner
                logger.error(
                    f"Pattern #{i} ({type(pattern).__name__}) raised exception: {e}",
                    exc_info=True
                )

    def _call_lifecycle_hooks(self, hook_name: str) -> None:
        """
        Call lifecycle hooks on patterns that support them.

        Args:
            hook_name: 'on_startup' or 'on_shutdown'
        """
        for i, pattern in enumerate(self.patterns):
            if isinstance(pattern, LifecyclePattern):
                try:
                    hook = getattr(pattern, hook_name, None)
                    if hook:
                        hook()
                except Exception as e:
                    logger.error(
                        f"Pattern #{i} lifecycle hook '{hook_name}' raised exception: {e}",
                        exc_info=True
                    )

    def startup(self) -> None:
        """
        Explicitly call startup lifecycle hooks.

        Useful if enable_lifecycle=False in constructor.
        """
        if not self._started:
            self._call_lifecycle_hooks("on_startup")
            self._started = True

    def shutdown(self) -> None:
        """
        Call shutdown lifecycle hooks and clean up.

        Should be called when done with runner.
        """
        if self._started:
            self._call_lifecycle_hooks("on_shutdown")
            self._started = False

    def get_stats(self) -> dict:
        """
        Get runner statistics.

        Returns:
            Dictionary with:
            - query_count: Total queries run
            - pattern_count: Number of patterns attached
        """
        return {
            "query_count": self._query_count,
            "pattern_count": len(self.patterns),
        }

    def __enter__(self):
        """Context manager entry."""
        if not self._started:
            self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


__all__ = ['WeaveRunner', 'SupportsIntersect']
