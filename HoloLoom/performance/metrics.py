#!/usr/bin/env python3
"""
Performance Metrics - Real-time system metrics collection
=========================================================
Tracks system-level metrics (CPU, memory, throughput) and
application-level metrics (query latency, cache hit rate, etc.).

Usage:
    metrics = MetricsCollector()
    metrics.record_latency("query_processing", 245.5)
    metrics.record_throughput("queries_per_second", 12)

    print(metrics.get_summary())
"""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

# Optional: psutil for system metrics
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class Metric:
    """Single metric measurement."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Real-time metrics collection with time-window aggregation.

    Features:
    - Latency tracking (p50, p95, p99)
    - Throughput tracking (ops/sec)
    - System resource monitoring
    - Time-windowed aggregation
    - Thread-safe
    """

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent samples to keep per metric
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

        # System metrics
        self.system_metrics_enabled = True
        self._last_system_check = 0
        self._system_cache = {}

    def record_latency(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a latency measurement.

        Args:
            name: Metric name (e.g., "query_processing")
            duration_ms: Duration in milliseconds
            tags: Optional tags for categorization
        """
        with self.lock:
            metric = Metric(
                name=f"{name}_latency",
                value=duration_ms,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics[metric.name].append(metric)

    def record_throughput(self, name: str, count: int = 1):
        """
        Record throughput (operations count).

        Args:
            name: Metric name (e.g., "queries_processed")
            count: Number of operations
        """
        with self.lock:
            self.counters[name] += count

    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a gauge value (point-in-time measurement).

        Args:
            name: Metric name (e.g., "active_connections")
            value: Current value
            tags: Optional tags
        """
        with self.lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        with self.lock:
            self.counters[name] += value

    def get_latency_stats(self, name: str) -> Dict[str, float]:
        """
        Get latency statistics (p50, p95, p99).

        Args:
            name: Metric name

        Returns:
            Dict with p50, p95, p99, mean, min, max
        """
        with self.lock:
            metric_name = f"{name}_latency"
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return {}

            values = [m.value for m in self.metrics[metric_name]]
            sorted_values = sorted(values)
            n = len(sorted_values)

            return {
                "count": n,
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": sorted_values[int(0.95 * n)] if n > 0 else 0,
                "p99": sorted_values[int(0.99 * n)] if n > 0 else 0,
                "min": min(values),
                "max": max(values)
            }

    def get_throughput(self, name: str, window_seconds: float = 60.0) -> float:
        """
        Calculate throughput over a time window.

        Args:
            name: Metric name
            window_seconds: Time window in seconds

        Returns:
            Operations per second
        """
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return 0.0

            now = time.time()
            recent = [m for m in self.metrics[name] if now - m.timestamp <= window_seconds]

            if not recent or window_seconds == 0:
                return 0.0

            return len(recent) / window_seconds

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self.lock:
            return self.counters.get(name, 0)

    def get_system_metrics(self) -> Dict[str, float]:
        """
        Get current system resource usage.

        Returns:
            Dict with cpu_percent, memory_percent, memory_mb
        """
        if not HAS_PSUTIL:
            return {
                "cpu_percent": 0.0,
                "memory_mb": 0.0,
                "memory_percent": 0.0,
                "error": "psutil not available"
            }

        # Cache system metrics (expensive to compute)
        now = time.time()
        if now - self._last_system_check < 1.0:  # 1 second cache
            return self._system_cache

        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()

            self._system_cache = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / (1024 * 1024),
                "memory_percent": process.memory_percent()
            }
            self._last_system_check = now

        except Exception as e:
            self._system_cache = {
                "cpu_percent": 0.0,
                "memory_mb": 0.0,
                "memory_percent": 0.0,
                "error": str(e)
            }

        return self._system_cache

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dict with all metrics, latencies, counters, system stats
        """
        summary = {
            "timestamp": time.time(),
            "latencies": {},
            "counters": dict(self.counters),
            "system": self.get_system_metrics()
        }

        # Aggregate latency metrics
        for name in self.metrics.keys():
            if name.endswith("_latency"):
                base_name = name.replace("_latency", "")
                summary["latencies"][base_name] = self.get_latency_stats(base_name)

        return summary

    def reset(self):
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metric string
        """
        lines = []

        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE hololoom_{name} counter")
            lines.append(f"hololoom_{name} {value}")

        # Latencies
        for name in self.metrics.keys():
            if name.endswith("_latency"):
                base_name = name.replace("_latency", "")
                stats = self.get_latency_stats(base_name)

                if stats:
                    lines.append(f"# TYPE hololoom_{base_name}_latency_seconds summary")
                    lines.append(f"hololoom_{base_name}_latency_seconds{{quantile=\"0.5\"}} {stats['median']/1000}")
                    lines.append(f"hololoom_{base_name}_latency_seconds{{quantile=\"0.95\"}} {stats['p95']/1000}")
                    lines.append(f"hololoom_{base_name}_latency_seconds{{quantile=\"0.99\"}} {stats['p99']/1000}")
                    lines.append(f"hololoom_{base_name}_latency_seconds_count {stats['count']}")

        # System metrics
        system = self.get_system_metrics()
        lines.append(f"# TYPE hololoom_cpu_percent gauge")
        lines.append(f"hololoom_cpu_percent {system['cpu_percent']}")
        lines.append(f"# TYPE hololoom_memory_mb gauge")
        lines.append(f"hololoom_memory_mb {system['memory_mb']}")

        return "\n".join(lines)


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_global_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics
