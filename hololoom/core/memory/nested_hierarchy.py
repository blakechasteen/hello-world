"""
Nested Learning Hierarchy (5 Levels)

Manages the complete nested learning system with 5 optimization levels:
1. Ultra-Fast (1-10ms)   - Sub-query routing, token gating
2. Fast (100-200ms)      - Thompson Sampling, tool selection (Phase 5)
3. Medium (1-5s)         - Step selection, retrieval strategies
4. Slow (60s)            - Background learning (Phase 5)
5. Very Slow (∞)         - Architectural evolution

Each level has:
- Own update frequency
- Own context window
- Own associative memory
- Own learning rate

Philosophy:
"Different problems require different timescales. Fast decisions enable
fluid reasoning. Slow decisions enable deep learning."
"""

import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time
import torch

from hololoom.core.memory.nested_ultra_fast import UltraFastOptimizer, UltraFastDecision


class OptimizationLevel(Enum):
    """5 nested optimization levels"""
    ULTRA_FAST = "ultra_fast"  # 1-10ms per update
    FAST = "fast"               # 100-200ms per update
    MEDIUM = "medium"           # 1-5s per update
    SLOW = "slow"               # 60s per update
    VERY_SLOW = "very_slow"     # ∞ per update (manual)


@dataclass
class LevelConfig:
    """Configuration for an optimization level"""
    name: OptimizationLevel
    update_frequency_ms: float
    context_window_size: int
    learning_rate: float
    enabled: bool = True


@dataclass
class HierarchyState:
    """Current state of nested learning hierarchy"""
    current_level: OptimizationLevel
    levels_active: Dict[OptimizationLevel, bool]
    last_update_times: Dict[OptimizationLevel, float]
    total_updates: Dict[OptimizationLevel, int]
    average_latencies: Dict[OptimizationLevel, float]


class NestedLearningHierarchy:
    """
    Manages 5-level nested learning hierarchy.

    Coordinates updates across all optimization levels, ensuring each level
    updates at its appropriate frequency without interfering with others.

    Architecture:
        Level 1 (Ultra-Fast) → Updates every sub-query
        Level 2 (Fast)       → Updates every query (existing Phase 5)
        Level 3 (Medium)     → Updates every outcome
        Level 4 (Slow)       → Updates every 60s (background)
        Level 5 (Very Slow)  → Updates manually (architectural changes)

    Example Usage:
        >>> hierarchy = NestedLearningHierarchy()
        >>> await hierarchy.initialize()
        >>>
        >>> # Ultra-fast decision
        >>> decision = await hierarchy.ultra_fast_decision(
        ...     "Should I verify this?",
        ...     decision_type=DecisionType.VERIFY_CLAIM
        ... )
        >>>
        >>> # Background learning happens automatically
        >>> await asyncio.sleep(60)  # Slow level updates automatically
    """

    def __init__(
        self,
        enable_ultra_fast: bool = True,
        enable_fast: bool = True,
        enable_medium: bool = True,
        enable_slow: bool = True,
        enable_very_slow: bool = False  # Opt-in (experimental)
    ):
        # Level configurations
        self.configs = {
            OptimizationLevel.ULTRA_FAST: LevelConfig(
                name=OptimizationLevel.ULTRA_FAST,
                update_frequency_ms=5.0,  # Every 5ms
                context_window_size=1,     # Only current context
                learning_rate=0.1,         # Fast learning
                enabled=enable_ultra_fast
            ),
            OptimizationLevel.FAST: LevelConfig(
                name=OptimizationLevel.FAST,
                update_frequency_ms=150.0,  # Every query
                context_window_size=10,     # Last 10 queries
                learning_rate=0.01,         # Moderate learning
                enabled=enable_fast
            ),
            OptimizationLevel.MEDIUM: LevelConfig(
                name=OptimizationLevel.MEDIUM,
                update_frequency_ms=2000.0,  # Every 2s
                context_window_size=100,     # Last 100 queries
                learning_rate=0.001,         # Slow learning
                enabled=enable_medium
            ),
            OptimizationLevel.SLOW: LevelConfig(
                name=OptimizationLevel.SLOW,
                update_frequency_ms=60000.0,  # Every 60s
                context_window_size=1000,     # Last 1000 queries
                learning_rate=0.0001,         # Very slow learning
                enabled=enable_slow
            ),
            OptimizationLevel.VERY_SLOW: LevelConfig(
                name=OptimizationLevel.VERY_SLOW,
                update_frequency_ms=float('inf'),  # Manual only
                context_window_size=10000,         # All history
                learning_rate=0.00001,             # Extremely slow
                enabled=enable_very_slow
            )
        }

        # Optimizers
        self.ultra_fast_optimizer = UltraFastOptimizer() if enable_ultra_fast else None

        # Phase 5 optimizers (integrated later)
        self.fast_optimizer = None      # Thompson Sampling (Phase 5)
        self.medium_optimizer = None    # Step selection (to be implemented)
        self.slow_optimizer = None      # Background learning (Phase 5)
        self.very_slow_optimizer = None # Architectural evolution (to be implemented)

        # State tracking
        self.state = HierarchyState(
            current_level=OptimizationLevel.ULTRA_FAST,
            levels_active={level: config.enabled for level, config in self.configs.items()},
            last_update_times={level: 0.0 for level in OptimizationLevel},
            total_updates={level: 0 for level in OptimizationLevel},
            average_latencies={level: 0.0 for level in OptimizationLevel}
        )

        # Background task
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def initialize(self):
        """Initialize hierarchy and start background tasks"""
        # Start slow optimizer background task
        if self.configs[OptimizationLevel.SLOW].enabled:
            self._background_task = asyncio.create_task(
                self._background_learning_loop()
            )

        print("NestedLearningHierarchy initialized")
        print(self.summary())

    async def shutdown(self):
        """Shutdown hierarchy and cleanup"""
        self._shutdown = True

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        print("NestedLearningHierarchy shutdown complete")

    async def ultra_fast_decision(
        self,
        text: str,
        parent_query: str = "",
        working_memory_state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> UltraFastDecision:
        """
        Make ultra-fast decision (<10ms).

        Args:
            text: Query or sub-query text
            parent_query: Parent query for context
            working_memory_state: Current feature state
            **kwargs: Additional decision-specific parameters

        Returns:
            UltraFastDecision
        """
        if not self.configs[OptimizationLevel.ULTRA_FAST].enabled:
            raise RuntimeError("Ultra-fast level is disabled")

        start_time = time.perf_counter()

        # Default working memory if not provided
        if working_memory_state is None:
            working_memory_state = torch.randn(1, 244)

        # Route decision
        decision = await self.ultra_fast_optimizer.route_subquery(
            text=text,
            parent_query=parent_query,
            working_memory_state=working_memory_state,
            **kwargs
        )

        # Update statistics
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._update_level_stats(OptimizationLevel.ULTRA_FAST, latency_ms)

        return decision

    async def fast_update(self, query_outcome: Dict[str, Any]):
        """
        Fast level update (Thompson Sampling, tool selection).

        This integrates with Phase 5's existing Thompson Sampling.

        Args:
            query_outcome: Query result with confidence, tool used, etc.
        """
        if not self.configs[OptimizationLevel.FAST].enabled:
            return

        start_time = time.perf_counter()

        # Phase 5 Thompson Sampling update
        # (Integrated later with existing FullLearningEngine)
        # For now, placeholder
        await asyncio.sleep(0.001)  # Simulate update

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._update_level_stats(OptimizationLevel.FAST, latency_ms)

    async def medium_update(self, query_outcome: Dict[str, Any]):
        """
        Medium level update (step selection, retrieval strategies).

        This will be implemented in Sub-Phase 6.2.

        Args:
            query_outcome: Query result with steps used, retrieval method, etc.
        """
        if not self.configs[OptimizationLevel.MEDIUM].enabled:
            return

        start_time = time.perf_counter()

        # Placeholder for step selection policy update
        await asyncio.sleep(0.001)

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._update_level_stats(OptimizationLevel.MEDIUM, latency_ms)

    async def _background_learning_loop(self):
        """
        Slow level background learning loop (runs every 60s).

        This integrates with Phase 5's background learning thread.
        """
        while not self._shutdown:
            try:
                await asyncio.sleep(60.0)  # Every 60s

                if not self.configs[OptimizationLevel.SLOW].enabled:
                    continue

                start_time = time.perf_counter()

                # Phase 5 background learning
                # (Integrated later with existing FullLearningEngine)
                await self._slow_update()

                latency_ms = (time.perf_counter() - start_time) * 1000
                self._update_level_stats(OptimizationLevel.SLOW, latency_ms)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Background learning error: {e}")

    async def _slow_update(self):
        """Slow level update (background learning)"""
        # Placeholder - will integrate Phase 5's background learning
        pass

    def _update_level_stats(self, level: OptimizationLevel, latency_ms: float):
        """Update statistics for a level"""
        self.state.last_update_times[level] = time.time()
        self.state.total_updates[level] += 1

        # Update average latency (running average)
        n = self.state.total_updates[level]
        old_avg = self.state.average_latencies[level]
        self.state.average_latencies[level] = (old_avg * (n - 1) + latency_ms) / n

    def get_statistics(self) -> Dict[str, Any]:
        """Get hierarchy statistics"""
        return {
            'state': {
                'current_level': self.state.current_level.value,
                'levels_active': {k.value: v for k, v in self.state.levels_active.items()},
            },
            'updates': {
                level.value: count
                for level, count in self.state.total_updates.items()
            },
            'latencies': {
                level.value: f"{latency:.3f}ms"
                for level, latency in self.state.average_latencies.items()
            },
            'ultra_fast_details': (
                self.ultra_fast_optimizer.get_statistics()
                if self.ultra_fast_optimizer else None
            )
        }

    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            "Nested Learning Hierarchy",
            "=" * 50,
            ""
        ]

        for level in OptimizationLevel:
            config = self.configs[level]
            status = "ENABLED" if config.enabled else "DISABLED"
            updates = self.state.total_updates[level]
            avg_latency = self.state.average_latencies[level]

            lines.append(f"Level {level.value.upper()}: {status}")
            lines.append(f"  Update Frequency: {config.update_frequency_ms}ms")
            lines.append(f"  Context Window: {config.context_window_size}")
            lines.append(f"  Learning Rate: {config.learning_rate}")
            lines.append(f"  Total Updates: {updates}")
            if updates > 0:
                lines.append(f"  Average Latency: {avg_latency:.3f}ms")
            lines.append("")

        # Ultra-fast details
        if self.ultra_fast_optimizer:
            lines.append("Ultra-Fast Optimizer Details:")
            lines.append("-" * 50)
            lines.append(self.ultra_fast_optimizer.summary())

        return "\n".join(lines)


# Context manager support
class NestedLearningContext:
    """
    Context manager for nested learning hierarchy.

    Usage:
        async with NestedLearningContext() as hierarchy:
            decision = await hierarchy.ultra_fast_decision("Route this query")
    """

    def __init__(self, **kwargs):
        self.hierarchy = NestedLearningHierarchy(**kwargs)

    async def __aenter__(self):
        await self.hierarchy.initialize()
        return self.hierarchy

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.hierarchy.shutdown()
        return False
