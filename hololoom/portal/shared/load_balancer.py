"""
Load Balancer - Intelligent node selection for job dispatch.

Strategies:
- ROUND_ROBIN: Rotate through nodes sequentially
- LEAST_LOADED: Pick node with fewest current_jobs
- WEIGHTED: Score by (cpu_cores * weight + memory * weight) / current_jobs
- CAPABILITY: Match job requirements (GPU, memory) to node capabilities
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .types import NodeRecord


class LoadBalanceStrategy(str, Enum):
    """Node selection strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    CAPABILITY = "capability"


@dataclass
class SelectionResult:
    """Result of node selection with metadata."""
    node: NodeRecord | None
    strategy_used: LoadBalanceStrategy
    reason: str
    candidates_considered: int
    selection_time_ms: float = 0.0


@dataclass
class LoadBalancerStats:
    """Metrics for monitoring."""
    selections_total: int = 0
    selections_by_strategy: dict[LoadBalanceStrategy, int] = field(default_factory=dict)
    fallbacks: int = 0  # Times fell back to another strategy


class LoadBalancer:
    """
    Intelligent node selection with multiple strategies.

    Thread-safe for concurrent job dispatching.
    """

    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED,
        cpu_weight: float = 1.0,
        memory_weight: float = 0.5,
    ):
        """
        Initialize load balancer.

        Args:
            strategy: Default selection strategy
            cpu_weight: Weight for CPU cores in WEIGHTED strategy
            memory_weight: Weight for memory in WEIGHTED strategy
        """
        self.strategy = strategy
        self.cpu_weight = cpu_weight
        self.memory_weight = memory_weight

        # Round-robin state
        self._rr_index: int = 0

        # Stats
        self.stats = LoadBalancerStats()

    def select(
        self,
        nodes: list[NodeRecord],
        job_requirements: dict[str, Any] | None = None,
    ) -> SelectionResult:
        """
        Select best node for job execution.

        Args:
            nodes: Available online nodes
            job_requirements: Optional requirements (gpu, min_memory_mb, etc.)

        Returns:
            SelectionResult with chosen node and metadata
        """
        start = time.perf_counter()

        if not nodes:
            return SelectionResult(
                node=None,
                strategy_used=self.strategy,
                reason="No nodes available",
                candidates_considered=0,
                selection_time_ms=0.0
            )

        # Filter by requirements first if capability matching
        candidates = self._filter_by_requirements(nodes, job_requirements)

        if not candidates:
            # Fallback to all nodes if no capability match
            candidates = nodes
            self.stats.fallbacks += 1

        # Apply strategy
        selected = self._apply_strategy(candidates)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Update stats
        self.stats.selections_total += 1
        self.stats.selections_by_strategy[self.strategy] = \
            self.stats.selections_by_strategy.get(self.strategy, 0) + 1

        return SelectionResult(
            node=selected,
            strategy_used=self.strategy,
            reason=self._get_reason(selected, candidates),
            candidates_considered=len(candidates),
            selection_time_ms=elapsed_ms
        )

    def _filter_by_requirements(
        self,
        nodes: list[NodeRecord],
        requirements: dict[str, Any] | None
    ) -> list[NodeRecord]:
        """Filter nodes by job requirements."""
        if not requirements:
            return nodes

        filtered = []
        for node in nodes:
            caps = node.capabilities

            # GPU requirement
            if requirements.get("gpu") and not caps.gpu:
                continue

            # Minimum memory
            min_mem = requirements.get("min_memory_mb", 0)
            if caps.memory_mb < min_mem:
                continue

            # Minimum CPU cores
            min_cpu = requirements.get("min_cpu_cores", 0)
            if caps.cpu_cores < min_cpu:
                continue

            filtered.append(node)

        return filtered

    def _apply_strategy(self, nodes: list[NodeRecord]) -> NodeRecord | None:
        """Apply selection strategy to candidate nodes."""
        if not nodes:
            return None

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(nodes)

        elif self.strategy == LoadBalanceStrategy.LEAST_LOADED:
            return self._least_loaded(nodes)

        elif self.strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted(nodes)

        elif self.strategy == LoadBalanceStrategy.CAPABILITY:
            # Capability filtering already done, use least loaded among matches
            return self._least_loaded(nodes)

        # Fallback
        return nodes[0]

    def _round_robin(self, nodes: list[NodeRecord]) -> NodeRecord:
        """Rotate through nodes sequentially."""
        idx = self._rr_index % len(nodes)
        self._rr_index += 1
        return nodes[idx]

    def _least_loaded(self, nodes: list[NodeRecord]) -> NodeRecord:
        """Pick node with fewest current jobs."""
        return min(nodes, key=lambda n: n.current_jobs)

    def _weighted(self, nodes: list[NodeRecord]) -> NodeRecord:
        """Score nodes by capacity, penalized by load."""
        def score(node: NodeRecord) -> float:
            caps = node.capabilities
            capacity = (
                caps.cpu_cores * self.cpu_weight +
                caps.memory_mb / 1000 * self.memory_weight
            )
            # Penalize by current load (avoid div by zero)
            load_factor = 1.0 / (node.current_jobs + 1)
            return capacity * load_factor

        return max(nodes, key=score)

    def _get_reason(self, node: NodeRecord | None, candidates: list[NodeRecord]) -> str:
        """Generate human-readable selection reason."""
        if not node:
            return "No suitable node found"

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return f"Round-robin selection (index {self._rr_index - 1})"

        elif self.strategy == LoadBalanceStrategy.LEAST_LOADED:
            return f"Least loaded ({node.current_jobs} jobs)"

        elif self.strategy == LoadBalanceStrategy.WEIGHTED:
            return f"Highest weighted score ({node.capabilities.cpu_cores} cores, {node.capabilities.memory_mb}MB)"

        elif self.strategy == LoadBalanceStrategy.CAPABILITY:
            return f"Capability match ({node.capabilities.cpu_cores} cores, GPU={node.capabilities.gpu})"

        return "Selected"

    def get_stats(self) -> dict[str, Any]:
        """Get metrics for monitoring."""
        return {
            "selections_total": self.stats.selections_total,
            "selections_by_strategy": {
                s.value: c for s, c in self.stats.selections_by_strategy.items()
            },
            "fallbacks": self.stats.fallbacks,
            "current_strategy": self.strategy.value,
        }

    def reset_stats(self) -> None:
        """Reset metrics."""
        self.stats = LoadBalancerStats()
        self._rr_index = 0
