#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Patterns Base Interface
=================================

Protocol for pluggable learning patterns that observe weave results
and optionally update long-horizon behavior.

Design Philosophy:
- Patterns are OBSERVERS, not controllers
- Core remains usable without any patterns
- Patterns should be side-effectful but not required for correctness
- Multiple patterns can compose

Created: 2025-01-20
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from hololoom.core.types import WeaveResult


@runtime_checkable
class LearningPattern(Protocol):
    """
    Pluggable pattern that observes each weave result and optionally
    updates long-horizon behavior (policies, heuristics, memory, etc.).

    Patterns enable continual learning without baking logic into core modules.

    Example patterns:
    - Nested Learning: Multi-timescale learning (fast/medium/slow horizons)
    - Thompson Sampling: Bandit updates from rewards
    - Reflective Learning: Pattern mining from history
    - Meta-Learning: Policy selection refinement

    Usage:
        class MyPattern:
            def on_episode_end(self, result: WeaveResult) -> None:
                # Log result, update stats, adjust policies, etc.
                pass

        runner = WeaveRunner(shuttle=shuttle, patterns=[MyPattern()])
        result = runner.run_query("my query", warp_results)
    """

    def on_episode_end(self, result: WeaveResult) -> None:
        """
        Called after each Shuttle.intersect() call.

        Implementations may:
        - Log results for analysis
        - Update statistics/counters
        - Adjust bandit priors
        - Refine graph heuristics
        - Update policy selection logic
        - Store traces for offline learning

        Args:
            result: The WeaveResult from the just-completed weave

        Notes:
            - This method should NOT raise exceptions
            - Performance should be O(1) or very fast
            - Heavy computation should be deferred to background threads
            - No return value (side-effect only)
        """
        ...


@runtime_checkable
class LifecyclePattern(Protocol):
    """
    Extended pattern interface with lifecycle hooks.

    Optional protocol for patterns that need initialization/cleanup.
    """

    def on_startup(self) -> None:
        """
        Called once when runner initializes.

        Use for:
        - Loading checkpoints
        - Initializing buffers
        - Setting up background threads
        """
        ...

    def on_shutdown(self) -> None:
        """
        Called once when runner shuts down.

        Use for:
        - Saving checkpoints
        - Flushing buffers
        - Cleaning up resources
        """
        ...

    def on_episode_end(self, result: WeaveResult) -> None:
        """Same as LearningPattern.on_episode_end"""
        ...


__all__ = ['LearningPattern', 'LifecyclePattern']
