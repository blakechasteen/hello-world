"""
Monitoring: Performance and Health Metrics

Part 5: Production Hardening - Day 22

Provides comprehensive monitoring for the Context Department:
- Query performance metrics (QPS, latency, errors)
- Learning system metrics (calibration, strategy updates)
- Resource usage metrics (memory, CPU)
- Prometheus-compatible metrics export
- Human-readable summaries
"""

import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available - resource monitoring will use fallback", ImportWarning)


# ============================================================================
# Metric Types
# ============================================================================

class MetricType(Enum):
    """Type of metric for categorization"""
    COUNTER = "counter"        # Monotonically increasing
    GAUGE = "gauge"            # Can go up or down
    HISTOGRAM = "histogram"    # Distribution of values
    SUMMARY = "summary"        # Percentiles over time


# ============================================================================
# Latency Histogram
# ============================================================================

@dataclass
class LatencyHistogram:
    """Histogram of latency measurements"""
    measurements: list[float] = field(default_factory=list)
    max_size: int = 10000  # Keep last 10k measurements

    def add(self, latency_ms: float):
        """Add latency measurement"""
        self.measurements.append(latency_ms)

        # Keep only recent measurements
        if len(self.measurements) > self.max_size:
            self.measurements = self.measurements[-self.max_size:]

    def get_percentile(self, percentile: float) -> float:
        """
        Get latency percentile

        Args:
            percentile: Percentile (0-100)

        Returns:
            Latency at percentile (ms)
        """
        if not self.measurements:
            return 0.0

        import statistics
        return statistics.quantiles(self.measurements, n=100)[int(percentile) - 1]

    def get_percentiles(self) -> dict[str, float]:
        """Get common percentiles"""
        if not self.measurements:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        # Need at least 2 measurements for quantiles
        if len(self.measurements) < 2:
            value = self.measurements[0]
            return {"p50": value, "p90": value, "p95": value, "p99": value}

        import statistics
        quantiles = statistics.quantiles(self.measurements, n=100)

        return {
            "p50": quantiles[49],  # Median
            "p90": quantiles[89],
            "p95": quantiles[94],
            "p99": quantiles[98]
        }

    def get_mean(self) -> float:
        """Get mean latency"""
        if not self.measurements:
            return 0.0

        import statistics
        return statistics.mean(self.measurements)

    def get_stddev(self) -> float:
        """Get standard deviation"""
        if not self.measurements or len(self.measurements) < 2:
            return 0.0

        import statistics
        return statistics.stdev(self.measurements)


# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """
    Monitor query performance metrics

    Tracks:
    - Queries per second (QPS)
    - Latency distribution
    - Error rates by type
    - Cache hit rates
    - Fallback usage rates
    """

    def __init__(self):
        """Initialize performance monitor"""
        self.start_time = time.time()
        self.query_count = 0
        self.latency_histogram = LatencyHistogram()
        self.error_counts = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallback_count = 0

    def record_query(
        self,
        latency_ms: float,
        error: str | None = None,
        cache_hit: bool = False,
        fallback_used: bool = False
    ):
        """
        Record query execution

        Args:
            latency_ms: Query latency in milliseconds
            error: Error type if query failed (None if successful)
            cache_hit: Whether cache was hit
            fallback_used: Whether fallback was used
        """
        self.query_count += 1
        self.latency_histogram.add(latency_ms)

        if error:
            self.error_counts[error] += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if fallback_used:
            self.fallback_count += 1

    def get_qps(self) -> float:
        """Get queries per second"""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.query_count / elapsed

    def get_error_rate(self) -> float:
        """Get overall error rate (0.0 - 1.0)"""
        if self.query_count == 0:
            return 0.0
        total_errors = sum(self.error_counts.values())
        return total_errors / self.query_count

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate (0.0 - 1.0)"""
        total_cache = self.cache_hits + self.cache_misses
        if total_cache == 0:
            return 0.0
        return self.cache_hits / total_cache

    def get_fallback_rate(self) -> float:
        """Get fallback usage rate (0.0 - 1.0)"""
        if self.query_count == 0:
            return 0.0
        return self.fallback_count / self.query_count

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all performance metrics

        Returns:
            Dictionary with all metrics
        """
        percentiles = self.latency_histogram.get_percentiles()

        return {
            # Query metrics
            "query_count": self.query_count,
            "qps": self.get_qps(),

            # Latency metrics
            "latency_mean": self.latency_histogram.get_mean(),
            "latency_stddev": self.latency_histogram.get_stddev(),
            "latency_p50": percentiles["p50"],
            "latency_p90": percentiles["p90"],
            "latency_p95": percentiles["p95"],
            "latency_p99": percentiles["p99"],

            # Error metrics
            "error_rate": self.get_error_rate(),
            "error_counts": dict(self.error_counts),

            # Cache metrics
            "cache_hit_rate": self.get_cache_hit_rate(),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,

            # Fallback metrics
            "fallback_rate": self.get_fallback_rate(),
            "fallback_count": self.fallback_count,

            # Time window
            "window_seconds": time.time() - self.start_time
        }

    def reset(self):
        """Reset all metrics"""
        self.start_time = time.time()
        self.query_count = 0
        self.latency_histogram = LatencyHistogram()
        self.error_counts.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallback_count = 0


# ============================================================================
# Resource Monitor
# ============================================================================

class ResourceMonitor:
    """
    Monitor resource usage (memory, CPU)

    Uses psutil for system metrics (fallback if unavailable)
    """

    def __init__(self):
        """Initialize resource monitor"""
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        if self.process:
            return self.process.memory_info().rss / (1024 * 1024)
        else:
            # Fallback: return 0 (not available)
            return 0.0

    def get_cpu_percent(self) -> float:
        """Get CPU usage percentage (0-100)"""
        if self.process:
            return self.process.cpu_percent(interval=0.1)
        else:
            # Fallback: return 0 (not available)
            return 0.0

    def get_metrics(self) -> dict[str, float]:
        """
        Get all resource metrics

        Returns:
            Dictionary with resource metrics
        """
        return {
            "memory_mb": self.get_memory_mb(),
            "cpu_percent": self.get_cpu_percent()
        }


# ============================================================================
# Learning Metrics Monitor
# ============================================================================

class LearningMetricsMonitor:
    """
    Monitor learning system metrics

    Tracks:
    - Calibration ECE over time
    - Strategy update frequency
    - Weight adjustment magnitude
    """

    def __init__(self):
        """Initialize learning metrics monitor"""
        self.calibration_ece_history = []
        self.strategy_update_count = 0
        self.last_strategy_update = None
        self.weight_adjustments = []

    def record_calibration(self, ece: float):
        """Record calibration ECE"""
        self.calibration_ece_history.append({
            "ece": ece,
            "timestamp": time.time()
        })

        # Keep last 1000 observations
        if len(self.calibration_ece_history) > 1000:
            self.calibration_ece_history = self.calibration_ece_history[-1000:]

    def record_strategy_update(self, weight_changes: dict[str, float]):
        """
        Record strategy update

        Args:
            weight_changes: Dictionary of backend -> weight change
        """
        self.strategy_update_count += 1
        self.last_strategy_update = time.time()

        # Track magnitude of weight changes
        for backend, change in weight_changes.items():
            self.weight_adjustments.append({
                "backend": backend,
                "change": change,
                "timestamp": time.time()
            })

        # Keep last 1000 adjustments
        if len(self.weight_adjustments) > 1000:
            self.weight_adjustments = self.weight_adjustments[-1000:]

    def get_current_ece(self) -> float | None:
        """Get most recent calibration ECE"""
        if not self.calibration_ece_history:
            return None
        return self.calibration_ece_history[-1]["ece"]

    def get_mean_weight_adjustment(self) -> float:
        """Get mean absolute weight adjustment"""
        if not self.weight_adjustments:
            return 0.0

        import statistics
        changes = [abs(adj["change"]) for adj in self.weight_adjustments]
        return statistics.mean(changes)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all learning metrics

        Returns:
            Dictionary with learning metrics
        """
        return {
            "calibration_ece": self.get_current_ece(),
            "calibration_observations": len(self.calibration_ece_history),
            "strategy_update_count": self.strategy_update_count,
            "last_strategy_update": self.last_strategy_update,
            "mean_weight_adjustment": self.get_mean_weight_adjustment(),
            "total_weight_adjustments": len(self.weight_adjustments)
        }


# ============================================================================
# Unified System Monitor
# ============================================================================

class SystemMonitor:
    """
    Unified monitoring for entire context system

    Combines:
    - Performance monitoring (queries, latency, errors)
    - Resource monitoring (memory, CPU)
    - Learning monitoring (calibration, strategy)
    """

    def __init__(self):
        """Initialize system monitor"""
        self.performance = PerformanceMonitor()
        self.resources = ResourceMonitor()
        self.learning = LearningMetricsMonitor()

    def get_all_metrics(self) -> dict[str, Any]:
        """
        Get all system metrics

        Returns:
            Dictionary with all metrics organized by category
        """
        return {
            "performance": self.performance.get_metrics(),
            "resources": self.resources.get_metrics(),
            "learning": self.learning.get_metrics(),
            "timestamp": time.time()
        }

    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus text format

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Performance metrics
        perf = self.performance.get_metrics()
        lines.append("# HELP context_queries_total Total number of queries processed")
        lines.append("# TYPE context_queries_total counter")
        lines.append(f"context_queries_total {perf['query_count']}")

        lines.append("# HELP context_qps Queries per second")
        lines.append("# TYPE context_qps gauge")
        lines.append(f"context_qps {perf['qps']:.2f}")

        lines.append("# HELP context_latency_seconds Query latency percentiles")
        lines.append("# TYPE context_latency_seconds summary")
        lines.append(f"context_latency_seconds{{quantile=\"0.5\"}} {perf['latency_p50']/1000:.6f}")
        lines.append(f"context_latency_seconds{{quantile=\"0.9\"}} {perf['latency_p90']/1000:.6f}")
        lines.append(f"context_latency_seconds{{quantile=\"0.95\"}} {perf['latency_p95']/1000:.6f}")
        lines.append(f"context_latency_seconds{{quantile=\"0.99\"}} {perf['latency_p99']/1000:.6f}")

        lines.append("# HELP context_error_rate Query error rate")
        lines.append("# TYPE context_error_rate gauge")
        lines.append(f"context_error_rate {perf['error_rate']:.4f}")

        lines.append("# HELP context_cache_hit_rate Cache hit rate")
        lines.append("# TYPE context_cache_hit_rate gauge")
        lines.append(f"context_cache_hit_rate {perf['cache_hit_rate']:.4f}")

        # Resource metrics
        res = self.resources.get_metrics()
        lines.append("# HELP context_memory_mb Memory usage in MB")
        lines.append("# TYPE context_memory_mb gauge")
        lines.append(f"context_memory_mb {res['memory_mb']:.2f}")

        lines.append("# HELP context_cpu_percent CPU usage percentage")
        lines.append("# TYPE context_cpu_percent gauge")
        lines.append(f"context_cpu_percent {res['cpu_percent']:.2f}")

        # Learning metrics
        learn = self.learning.get_metrics()
        if learn["calibration_ece"] is not None:
            lines.append("# HELP context_calibration_ece Calibration ECE")
            lines.append("# TYPE context_calibration_ece gauge")
            lines.append(f"context_calibration_ece {learn['calibration_ece']:.4f}")

        lines.append("# HELP context_strategy_updates_total Strategy update count")
        lines.append("# TYPE context_strategy_updates_total counter")
        lines.append(f"context_strategy_updates_total {learn['strategy_update_count']}")

        return "\n".join(lines) + "\n"

    def get_summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Formatted summary string
        """
        metrics = self.get_all_metrics()
        perf = metrics["performance"]
        res = metrics["resources"]
        learn = metrics["learning"]

        lines = []
        lines.append("=" * 80)
        lines.append("System Monitor Summary")
        lines.append("=" * 80)

        lines.append("\nPerformance:")
        lines.append(f"  Queries: {perf['query_count']} ({perf['qps']:.2f} QPS)")
        lines.append(f"  Latency: p50={perf['latency_p50']:.1f}ms, p95={perf['latency_p95']:.1f}ms, p99={perf['latency_p99']:.1f}ms")
        lines.append(f"  Error rate: {perf['error_rate']*100:.2f}%")
        lines.append(f"  Cache hit rate: {perf['cache_hit_rate']*100:.1f}%")
        lines.append(f"  Fallback rate: {perf['fallback_rate']*100:.2f}%")

        lines.append("\nResources:")
        lines.append(f"  Memory: {res['memory_mb']:.1f} MB")
        lines.append(f"  CPU: {res['cpu_percent']:.1f}%")

        lines.append("\nLearning:")
        if learn["calibration_ece"] is not None:
            lines.append(f"  Calibration ECE: {learn['calibration_ece']:.4f}")
        else:
            lines.append("  Calibration: Not yet calibrated")
        lines.append(f"  Strategy updates: {learn['strategy_update_count']}")
        lines.append(f"  Mean weight adjustment: {learn['mean_weight_adjustment']:.3f}")

        lines.append("")
        return "\n".join(lines)


# ============================================================================
# Factory Function
# ============================================================================

def create_system_monitor() -> SystemMonitor:
    """
    Create system monitor

    Returns:
        Configured SystemMonitor instance
    """
    return SystemMonitor()
