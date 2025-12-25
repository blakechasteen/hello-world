"""
Eggroll Observability - Prometheus Metrics.

Provides Prometheus-compatible metrics for distributed evolutionary training.

Metrics:
- eggroll_epoch_duration_seconds: Histogram of epoch durations
- eggroll_fitness_score: Gauge of fitness scores by worker
- eggroll_workers_active: Gauge of active worker count
- eggroll_selection_pressure: Gauge of selection pressure
- eggroll_diversity_score: Gauge of population diversity
- eggroll_rust_operations_total: Counter of Rust-accelerated operations
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from HoloLoom.telemetry.metrics import (
    get_registry,
    counter,
    gauge,
    histogram,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  EGGROLL METRICS
# ═══════════════════════════════════════════════════════════════════════════════


# Epoch duration histogram (seconds)
eggroll_epoch_duration = histogram(
    name="eggroll_epoch_duration_seconds",
    description="Duration of evolutionary epochs in seconds",
    labels=["population_id", "complexity"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Fitness score gauge (by worker)
eggroll_fitness_score = gauge(
    name="eggroll_fitness_score",
    description="Current fitness score by worker",
    labels=["worker_id", "population_id"],
)

# Active workers gauge
eggroll_workers_active = gauge(
    name="eggroll_workers_active",
    description="Number of active workers in the population",
    labels=["population_id"],
)

# Selection pressure gauge
eggroll_selection_pressure = gauge(
    name="eggroll_selection_pressure",
    description="Selection pressure (fitness variance / mean)",
    labels=["population_id"],
)

# Population diversity gauge
eggroll_diversity_score = gauge(
    name="eggroll_diversity_score",
    description="Population diversity score from clustering",
    labels=["population_id"],
)

# Rust operations counter
eggroll_rust_ops = counter(
    name="eggroll_rust_operations_total",
    description="Total Rust-accelerated operations",
    labels=["operation", "backend"],
)

# Evolution events counter
eggroll_evolution_events = counter(
    name="eggroll_evolution_events_total",
    description="Evolution events (mutations, crossovers, selections)",
    labels=["event_type", "population_id"],
)

# Best fitness gauge
eggroll_best_fitness = gauge(
    name="eggroll_best_fitness",
    description="Best fitness score in population",
    labels=["population_id"],
)

# Embedding similarity gauge
eggroll_embedding_similarity = gauge(
    name="eggroll_embedding_similarity",
    description="Average embedding similarity to target",
    labels=["population_id"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  EGGROLL METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EggrollMetricsCollector:
    """
    Collector for Eggroll distributed evolution metrics.

    Provides convenient methods for recording metrics during training.

    Usage:
        collector = EggrollMetricsCollector(population_id="pop_001")

        with collector.time_epoch(complexity="FULL"):
            # Run epoch
            pass

        collector.record_fitness(worker_id="worker_0", fitness=0.85)
        collector.record_diversity(diversity_score=0.72)
    """

    population_id: str = "default"

    # Internal tracking
    _epoch_count: int = field(default=0, init=False)
    _fitness_history: List[float] = field(default_factory=list, init=False)

    @contextmanager
    def time_epoch(self, complexity: str = "FULL"):
        """Context manager to time epoch duration."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            eggroll_epoch_duration.observe(
                duration,
                labels={"population_id": self.population_id, "complexity": complexity}
            )
            self._epoch_count += 1

    def record_fitness(
        self,
        worker_id: str,
        fitness: float,
    ) -> None:
        """Record fitness score for a worker."""
        eggroll_fitness_score.set(
            fitness,
            labels={"worker_id": worker_id, "population_id": self.population_id}
        )
        self._fitness_history.append(fitness)

    def record_workers_active(self, count: int) -> None:
        """Record number of active workers."""
        eggroll_workers_active.set(
            count,
            labels={"population_id": self.population_id}
        )

    def record_selection_pressure(self, pressure: float) -> None:
        """Record selection pressure (fitness variance / mean)."""
        eggroll_selection_pressure.set(
            pressure,
            labels={"population_id": self.population_id}
        )

    def record_diversity(self, diversity_score: float) -> None:
        """Record population diversity score."""
        eggroll_diversity_score.set(
            diversity_score,
            labels={"population_id": self.population_id}
        )

    def record_best_fitness(self, fitness: float) -> None:
        """Record best fitness in population."""
        eggroll_best_fitness.set(
            fitness,
            labels={"population_id": self.population_id}
        )

    def record_embedding_similarity(self, similarity: float) -> None:
        """Record average embedding similarity to target."""
        eggroll_embedding_similarity.set(
            similarity,
            labels={"population_id": self.population_id}
        )

    def record_rust_operation(self, operation: str, backend: str = "rust") -> None:
        """Record a Rust-accelerated operation."""
        eggroll_rust_ops.inc(
            labels={"operation": operation, "backend": backend}
        )

    def record_evolution_event(self, event_type: str) -> None:
        """Record an evolution event (mutation, crossover, selection)."""
        eggroll_evolution_events.inc(
            labels={"event_type": event_type, "population_id": self.population_id}
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        return {
            "population_id": self.population_id,
            "epoch_count": self._epoch_count,
            "fitness_samples": len(self._fitness_history),
            "avg_fitness": (
                sum(self._fitness_history) / len(self._fitness_history)
                if self._fitness_history else 0.0
            ),
            "best_fitness": max(self._fitness_history) if self._fitness_history else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_eggroll_collector(population_id: str = "default") -> EggrollMetricsCollector:
    """Create an Eggroll metrics collector."""
    return EggrollMetricsCollector(population_id=population_id)


# Global collector for convenience
_default_collector: Optional[EggrollMetricsCollector] = None


def get_eggroll_collector() -> EggrollMetricsCollector:
    """Get the global Eggroll metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = create_eggroll_collector()
    return _default_collector


def set_eggroll_collector(collector: EggrollMetricsCollector) -> None:
    """Set the global Eggroll metrics collector."""
    global _default_collector
    _default_collector = collector
