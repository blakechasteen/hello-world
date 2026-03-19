"""
Dream Orchestrator - Collective Consolidation for the Weave House
==================================================================

Orchestrates collective dreaming for multiple Looms, enabling shared learning
while preserving perspective diversity through controlled insight bleeding.

Philosophy:
"Math bleeds freely (universal structure), patterns bleed conservatively
(preserve diversity), corrections always propagate (shared learning),
discoveries require validation (prevent hallucination spread)."

Bleed Rates:
- Math (30%): Warp Space insights are universal structure
- Patterns (20%): Preserve diversity, some sharing
- Corrections (100%): Errors should propagate immediately
- Discoveries (High-conf only): Novel findings if validated

Anti-Groupthink Mechanisms:
1. Low pattern bleed (20%) preserves perspective diversity
2. Independent priors - each Loom has own Thompson distributions
3. Diversity reward - Looms rewarded for unique correct contributions
4. Explicit disagreement - tensions surfaced, not averaged away

Date: December 2025
Author: HoloLoom Architecture Team
"""

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hololoom.loom.protocol import (
    DreamInsight,
    InsightType,
    Loom,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Dream Cycle Statistics
# ============================================================================

@dataclass
class DreamCycleStats:
    """Statistics from one dream cycle."""

    cycle_number: int
    timestamp: datetime

    # Collection stats
    total_insights_collected: int = 0
    insights_by_loom: dict[str, int] = field(default_factory=dict)
    insights_by_type: dict[str, int] = field(default_factory=dict)

    # Distribution stats
    math_shared: int = 0
    patterns_shared: int = 0
    corrections_shared: int = 0
    discoveries_shared: int = 0

    # Integration stats
    insights_distributed: int = 0
    successful_integrations: int = 0

    # Duration
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "total_insights_collected": self.total_insights_collected,
            "insights_by_loom": self.insights_by_loom,
            "insights_by_type": self.insights_by_type,
            "math_shared": self.math_shared,
            "patterns_shared": self.patterns_shared,
            "corrections_shared": self.corrections_shared,
            "discoveries_shared": self.discoveries_shared,
            "insights_distributed": self.insights_distributed,
            "successful_integrations": self.successful_integrations,
            "duration_ms": self.duration_ms,
        }


@dataclass
class DreamOrchestratorMetrics:
    """Cumulative metrics across all dream cycles."""

    total_cycles: int = 0
    total_insights_processed: int = 0
    total_insights_shared: int = 0

    # By type
    math_insights_shared: int = 0
    pattern_insights_shared: int = 0
    correction_insights_shared: int = 0
    discovery_insights_shared: int = 0

    # Bleed effectiveness
    avg_math_bleed_rate: float = 0.0
    avg_pattern_bleed_rate: float = 0.0

    # Timing
    avg_cycle_duration_ms: float = 0.0

    # Recent history
    recent_cycles: list[DreamCycleStats] = field(default_factory=list)

    def add_cycle(self, stats: DreamCycleStats):
        """Add cycle stats and update cumulative metrics."""
        self.total_cycles += 1
        self.total_insights_processed += stats.total_insights_collected
        self.total_insights_shared += (
            stats.math_shared +
            stats.patterns_shared +
            stats.corrections_shared +
            stats.discoveries_shared
        )

        self.math_insights_shared += stats.math_shared
        self.pattern_insights_shared += stats.patterns_shared
        self.correction_insights_shared += stats.corrections_shared
        self.discovery_insights_shared += stats.discoveries_shared

        # Update averages
        self.avg_cycle_duration_ms = (
            (self.avg_cycle_duration_ms * (self.total_cycles - 1) + stats.duration_ms)
            / self.total_cycles
        )

        # Keep recent cycles (last 10)
        self.recent_cycles.append(stats)
        if len(self.recent_cycles) > 10:
            self.recent_cycles.pop(0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_cycles": self.total_cycles,
            "total_insights_processed": self.total_insights_processed,
            "total_insights_shared": self.total_insights_shared,
            "math_insights_shared": self.math_insights_shared,
            "pattern_insights_shared": self.pattern_insights_shared,
            "correction_insights_shared": self.correction_insights_shared,
            "discovery_insights_shared": self.discovery_insights_shared,
            "avg_cycle_duration_ms": self.avg_cycle_duration_ms,
            "recent_cycles": [c.to_dict() for c in self.recent_cycles],
        }


# ============================================================================
# Dream Orchestrator
# ============================================================================

class DreamOrchestrator:
    """
    Orchestrates collective dreaming for the WeaveHouse.

    During dreaming, insights flow between looms based on type-specific
    bleed rates. This enables collective learning while preserving the
    unique perspectives of each loom.

    Bleed Philosophy:
    - Math insights (30%): Universal mathematical truths benefit all
    - Pattern insights (20%): Some sharing, but preserve diversity
    - Correction insights (100%): Errors must propagate immediately
    - Discovery insights (high-conf): Only validated discoveries share

    Example:
        orchestrator = DreamOrchestrator(
            looms=[recall_loom, reason_loom, reach_loom],
            consolidation_interval=3600,  # 1 hour
            math_bleed_rate=0.3,
            pattern_bleed_rate=0.2
        )

        await orchestrator.start_dreaming()
        # ... application runs ...
        await orchestrator.stop_dreaming()
    """

    def __init__(
        self,
        looms: list[Loom],
        consolidation_interval: int = 3600,
        math_bleed_rate: float = 0.3,
        pattern_bleed_rate: float = 0.2,
        discovery_confidence_threshold: float = 0.8,
        max_insights_per_cycle: int = 100,
        on_cycle_complete: Callable[[DreamCycleStats], None] | None = None,
    ):
        """
        Initialize the Dream Orchestrator.

        Args:
            looms: List of Loom instances to coordinate
            consolidation_interval: Seconds between dream cycles (default: 1 hour)
            math_bleed_rate: Fraction of math insights to share (default: 0.3)
            pattern_bleed_rate: Fraction of pattern insights to share (default: 0.2)
            discovery_confidence_threshold: Min confidence for discoveries (default: 0.8)
            max_insights_per_cycle: Cap on insights per cycle (default: 100)
            on_cycle_complete: Optional callback after each cycle
        """
        self.looms = looms
        self.consolidation_interval = consolidation_interval
        self.math_bleed_rate = math_bleed_rate
        self.pattern_bleed_rate = pattern_bleed_rate
        self.discovery_confidence_threshold = discovery_confidence_threshold
        self.max_insights_per_cycle = max_insights_per_cycle
        self.on_cycle_complete = on_cycle_complete

        # State
        self._running = False
        self._task: asyncio.Task | None = None
        self._cycle_count = 0
        self._metrics = DreamOrchestratorMetrics()

        # Loom name lookup
        self._loom_names: dict[str, Loom] = {
            loom.perspective: loom for loom in looms
        }

        logger.info(
            f"DreamOrchestrator initialized with {len(looms)} looms, "
            f"interval={consolidation_interval}s, "
            f"math_bleed={math_bleed_rate}, pattern_bleed={pattern_bleed_rate}"
        )

    # ========== Lifecycle ==========

    async def start_dreaming(self):
        """Start background dreaming loop."""
        if self._running:
            logger.warning("DreamOrchestrator already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._dreaming_loop())
        logger.info("DreamOrchestrator started dreaming loop")

    async def stop_dreaming(self):
        """Stop background dreaming loop gracefully."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("DreamOrchestrator stopped dreaming loop")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_dreaming()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_dreaming()
        return False

    # ========== Main Loop ==========

    async def _dreaming_loop(self):
        """Background dreaming loop."""
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval)

                if self._running:  # Check again after sleep
                    await self._dream_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dream cycle: {e}")
                # Continue running despite errors

    async def dream_once(self) -> DreamCycleStats:
        """
        Run a single dream cycle (for manual triggering).

        Returns:
            Statistics from this cycle
        """
        return await self._dream_cycle()

    async def _dream_cycle(self) -> DreamCycleStats:
        """
        Execute one cycle of collective dreaming.

        Steps:
        1. Collect shareable insights from all looms
        2. Filter by shareability and type
        3. Apply bleed rates to determine what to share
        4. Distribute insights to other looms
        5. Log cycle statistics

        Returns:
            Statistics from this cycle
        """
        start_time = datetime.now()
        self._cycle_count += 1

        stats = DreamCycleStats(
            cycle_number=self._cycle_count,
            timestamp=start_time,
        )

        logger.debug(f"Starting dream cycle {self._cycle_count}")

        # Step 1: Collect insights from all looms
        all_insights = await self._collect_insights(stats)

        # Step 2: Filter by type
        math_insights = [i for i in all_insights if i.insight_type == InsightType.MATH]
        pattern_insights = [i for i in all_insights if i.insight_type == InsightType.PATTERN]
        correction_insights = [i for i in all_insights if i.insight_type == InsightType.CORRECTION]
        discovery_insights = [i for i in all_insights if i.insight_type == InsightType.DISCOVERY]

        # Step 3: Apply bleed rates
        math_to_share = self._sample_insights(math_insights, self.math_bleed_rate)
        patterns_to_share = self._sample_insights(pattern_insights, self.pattern_bleed_rate)
        corrections_to_share = correction_insights  # Always share (100%)
        discoveries_to_share = [
            d for d in discovery_insights
            if d.confidence >= self.discovery_confidence_threshold
        ]

        # Update stats
        stats.math_shared = len(math_to_share)
        stats.patterns_shared = len(patterns_to_share)
        stats.corrections_shared = len(corrections_to_share)
        stats.discoveries_shared = len(discoveries_to_share)

        # Step 4: Combine for distribution
        insights_to_distribute = (
            math_to_share +
            patterns_to_share +
            corrections_to_share +
            discoveries_to_share
        )

        # Cap at maximum
        if len(insights_to_distribute) > self.max_insights_per_cycle:
            # Prioritize: corrections > discoveries > math > patterns
            sorted_insights = (
                corrections_to_share +
                discoveries_to_share +
                math_to_share +
                patterns_to_share
            )
            insights_to_distribute = sorted_insights[:self.max_insights_per_cycle]

        # Step 5: Distribute to looms
        stats.insights_distributed = len(insights_to_distribute)
        stats.successful_integrations = await self._distribute_insights(insights_to_distribute)

        # Step 6: Calculate duration
        end_time = datetime.now()
        stats.duration_ms = (end_time - start_time).total_seconds() * 1000

        # Update metrics
        self._metrics.add_cycle(stats)

        # Callback
        if self.on_cycle_complete:
            try:
                self.on_cycle_complete(stats)
            except Exception as e:
                logger.warning(f"Error in cycle callback: {e}")

        logger.info(
            f"Dream cycle {self._cycle_count} complete: "
            f"collected={stats.total_insights_collected}, "
            f"shared={stats.insights_distributed}, "
            f"integrated={stats.successful_integrations}, "
            f"duration={stats.duration_ms:.1f}ms"
        )

        return stats

    # ========== Collection ==========

    async def _collect_insights(self, stats: DreamCycleStats) -> list[DreamInsight]:
        """
        Collect shareable insights from all looms.

        Args:
            stats: Stats object to update

        Returns:
            List of all collected insights
        """
        all_insights = []

        for loom in self.looms:
            try:
                insights = await loom.dream_share()

                # Filter to shareable only
                shareable = [i for i in insights if i.shareable]
                all_insights.extend(shareable)

                # Update stats
                stats.insights_by_loom[loom.perspective] = len(shareable)

                for insight in shareable:
                    type_name = insight.insight_type.value
                    stats.insights_by_type[type_name] = (
                        stats.insights_by_type.get(type_name, 0) + 1
                    )

            except Exception as e:
                logger.warning(f"Error collecting from {loom.perspective}: {e}")

        stats.total_insights_collected = len(all_insights)
        return all_insights

    # ========== Sampling ==========

    def _sample_insights(
        self,
        insights: list[DreamInsight],
        rate: float
    ) -> list[DreamInsight]:
        """
        Sample insights based on bleed rate and confidence.

        Higher confidence insights are more likely to be selected.

        Args:
            insights: List of insights to sample from
            rate: Bleed rate (0-1)

        Returns:
            Sampled insights
        """
        if not insights:
            return []

        # Calculate how many to sample
        n_to_sample = max(1, int(len(insights) * rate))
        n_to_sample = min(n_to_sample, len(insights))

        # Weight by confidence
        weights = [i.confidence for i in insights]
        total_weight = sum(weights)

        if total_weight == 0:
            # Uniform sampling if no confidence
            return random.sample(insights, n_to_sample)

        # Weighted sampling without replacement
        sampled = []
        remaining = list(zip(insights, weights))

        for _ in range(n_to_sample):
            if not remaining:
                break

            # Normalize weights
            total = sum(w for _, w in remaining)
            if total == 0:
                break

            # Random selection
            r = random.random() * total
            cumsum = 0

            for i, (insight, weight) in enumerate(remaining):
                cumsum += weight
                if cumsum >= r:
                    sampled.append(insight)
                    remaining.pop(i)
                    break

        return sampled

    # ========== Distribution ==========

    async def _distribute_insights(
        self,
        insights: list[DreamInsight]
    ) -> int:
        """
        Distribute insights to looms (excluding source loom).

        Args:
            insights: Insights to distribute

        Returns:
            Count of successful integrations
        """
        successful = 0

        for loom in self.looms:
            # Get insights NOT from this loom
            for_this_loom = [
                i for i in insights
                if i.source_loom != loom.perspective
            ]

            for insight in for_this_loom:
                try:
                    await loom.dream_receive(insight)
                    successful += 1
                except Exception as e:
                    logger.debug(
                        f"Failed to deliver insight to {loom.perspective}: {e}"
                    )

        return successful

    # ========== Metrics ==========

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "consolidation_interval": self.consolidation_interval,
            "math_bleed_rate": self.math_bleed_rate,
            "pattern_bleed_rate": self.pattern_bleed_rate,
            "looms": [l.perspective for l in self.looms],
            "metrics": self._metrics.to_dict(),
        }

    def reset_metrics(self):
        """Reset metrics to initial state."""
        self._metrics = DreamOrchestratorMetrics()
        self._cycle_count = 0


# ============================================================================
# Factory Function
# ============================================================================

def create_dream_orchestrator(
    looms: list[Loom],
    consolidation_interval: int = 3600,
    math_bleed_rate: float = 0.3,
    pattern_bleed_rate: float = 0.2,
    **kwargs
) -> DreamOrchestrator:
    """
    Factory function to create a DreamOrchestrator.

    Args:
        looms: List of Loom instances to coordinate
        consolidation_interval: Seconds between cycles (default: 1 hour)
        math_bleed_rate: Math insight bleed rate (default: 0.3)
        pattern_bleed_rate: Pattern insight bleed rate (default: 0.2)
        **kwargs: Additional arguments for DreamOrchestrator

    Returns:
        Configured DreamOrchestrator instance
    """
    return DreamOrchestrator(
        looms=looms,
        consolidation_interval=consolidation_interval,
        math_bleed_rate=math_bleed_rate,
        pattern_bleed_rate=pattern_bleed_rate,
        **kwargs
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'DreamOrchestrator',
    'DreamCycleStats',
    'DreamOrchestratorMetrics',
    'create_dream_orchestrator',
]
