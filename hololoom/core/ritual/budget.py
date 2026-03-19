"""
Context Budget — managed resource allocation for ritual execution.
===================================================================

Treats context windows as a scarce resource (the GSD insight):
  - Allocate tokens to named partitions
  - Track pressure (0.0 = plenty, 1.0 = exhausted)
  - Auto-switch complexity modes under pressure
  - Reclaim unused tokens via context manager

The Loom Scheduler calls check_admission() before dispatching a ritual.
Critical-priority rituals are never denied.

Backward compat: TokenBudget still works as a simple single-partition budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .types import RitualBudget, TokenBudget

logger = logging.getLogger(__name__)


def check_admission(
    ritual_budget: RitualBudget,
    agent_budget: TokenBudget | Any,
) -> bool:
    """
    Returns True if the agent has enough token budget to run the ritual.
    Critical-priority rituals are never denied.

    Works with both TokenBudget (simple) and ContextBudget (partitioned).
    """
    if ritual_budget.priority_class == "critical":
        return True

    if isinstance(agent_budget, ContextBudget):
        return agent_budget.remaining("ritual") >= ritual_budget.max_tokens

    # TokenBudget fallback
    return agent_budget.remaining >= ritual_budget.max_tokens


# ============================================================================
# Context Budget Protocol + Implementation
# ============================================================================


# Pressure thresholds for automatic mode switching
_PRESSURE_FAST = 0.7
_PRESSURE_BARE = 0.9

# Mode names matching the weaving complexity spectrum
MODE_RESEARCH = "research"
MODE_FUSED = "fused"
MODE_FAST = "fast"
MODE_BARE = "bare"


@dataclass
class ContextAllocation:
    """
    A token allocation from a named partition.

    Use as an async context manager to auto-release unused tokens:

        async with budget.allocate("ritual", 300) as alloc:
            alloc.used = 150
        # 150 tokens released back to partition
    """
    partition: str
    allocated: int
    used: int = 0
    _budget: ContextBudget | None = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)

    async def __aenter__(self) -> ContextAllocation:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def release(self) -> int:
        """Release unused tokens back to the budget. Returns tokens released."""
        if self._released or self._budget is None:
            return 0
        self._released = True
        unused = max(0, self.allocated - self.used)
        if unused > 0:
            self._budget._release_to_partition(self.partition, unused)
        return unused

    @property
    def remaining(self) -> int:
        return max(0, self.allocated - self.used)


@dataclass
class ContextBudget:
    """
    Partitioned context budget with pressure tracking and mode switching.

    Partitions are named buckets (e.g., "ritual", "memory", "generation").
    Total budget is shared across all partitions. Pressure is global.

    Mode auto-switches:
      pressure < 0.7 → research/fused (caller decides)
      pressure 0.7-0.9 → fast
      pressure > 0.9 → bare
    """
    total: int
    _used_by_partition: dict[str, int] = field(default_factory=dict)
    _base_mode: str = MODE_FUSED

    @property
    def used(self) -> int:
        return sum(self._used_by_partition.values())

    def remaining(self, partition: str | None = None) -> int:
        """Remaining tokens in a partition, or total remaining if no partition."""
        if partition is None:
            return self.total - self.used
        # Partition remaining = total remaining (partitions share the pool)
        return self.total - self.used

    def allocate(self, partition: str, tokens: int) -> ContextAllocation:
        """
        Allocate tokens from a partition. Returns a ContextAllocation
        that can be used as an async context manager.
        """
        available = self.remaining()
        actual = min(tokens, max(0, available))
        self._used_by_partition[partition] = (
            self._used_by_partition.get(partition, 0) + actual
        )
        return ContextAllocation(
            partition=partition,
            allocated=actual,
            _budget=self,
        )

    def _release_to_partition(self, partition: str, tokens: int) -> None:
        """Release tokens back to a partition (called by ContextAllocation)."""
        current = self._used_by_partition.get(partition, 0)
        self._used_by_partition[partition] = max(0, current - tokens)

    def pressure(self) -> float:
        """0.0 = plenty of budget remaining, 1.0 = exhausted."""
        if self.total <= 0:
            return 1.0
        return min(1.0, self.used / self.total)

    def mode(self) -> str:
        """
        Current complexity mode based on pressure.
        Auto-switches to simpler modes as budget depletes.
        """
        p = self.pressure()
        if p >= _PRESSURE_BARE:
            return MODE_BARE
        if p >= _PRESSURE_FAST:
            return MODE_FAST
        return self._base_mode

    def partition_usage(self) -> dict[str, int]:
        """Get usage breakdown by partition."""
        return dict(self._used_by_partition)
