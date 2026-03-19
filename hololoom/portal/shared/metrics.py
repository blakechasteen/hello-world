"""
Prometheus-compatible Metrics for Portal System.

Provides Counter, Gauge, Histogram metric types with thread-safe updates
and Prometheus text format export.

Usage:
    from hololoom.portal.shared.metrics import metrics

    # Increment counters
    metrics.jobs_submitted.inc()
    metrics.jobs_completed.inc()
    metrics.jobs_failed.inc()

    # Set gauges
    metrics.nodes_online.set(5)
    metrics.queue_depth.set(10)

    # Observe histograms
    metrics.job_duration.observe(150.0)  # milliseconds

    # Export for Prometheus scraping
    text = metrics.export_prometheus()

    # Get as dictionary
    data = metrics.to_dict()
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counter:
    """
    Counter metric - monotonically increasing value.

    Counters are cumulative metrics that only go up (or reset to zero on restart).
    Use for: total requests, errors, jobs completed, etc.
    """

    name: str
    help: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: float = 1.0) -> None:
        """Increment counter by amount (default: 1)."""
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset to zero (use sparingly - counters should generally not reset)."""
        with self._lock:
            self._value = 0.0


@dataclass
class Gauge:
    """
    Gauge metric - value that can go up or down.

    Gauges represent a value that can arbitrarily increase or decrease.
    Use for: queue depth, active connections, memory usage, etc.
    """

    name: str
    help: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float) -> None:
        """Set gauge to specific value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge by amount."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge by amount."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value


@dataclass
class Histogram:
    """
    Histogram metric - samples observations into configurable buckets.

    Histograms track the distribution of values (like latencies, sizes).
    Provides count, sum, and bucket counts.
    Use for: request latencies, response sizes, job durations, etc.
    """

    name: str
    help: str
    buckets: list[float] = field(default_factory=lambda: [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    labels: dict[str, str] = field(default_factory=dict)
    _counts: dict[float, int] = field(default_factory=dict, repr=False)
    _sum: float = field(default=0.0, repr=False)
    _count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        """Initialize bucket counts."""
        self._counts = dict.fromkeys(self.buckets, 0)
        self._counts[float("inf")] = 0  # +Inf bucket

    def observe(self, value: float) -> None:
        """
        Record an observed value.

        Increments all buckets where value <= bucket.
        """
        with self._lock:
            self._sum += value
            self._count += 1

            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[bucket] += 1

            # Always increment +Inf bucket
            self._counts[float("inf")] += 1

    @property
    def sum(self) -> float:
        """Get sum of all observations."""
        with self._lock:
            return self._sum

    @property
    def count(self) -> int:
        """Get count of observations."""
        with self._lock:
            return self._count

    def get_bucket_counts(self) -> dict[float, int]:
        """Get cumulative bucket counts."""
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self._counts = dict.fromkeys(self.buckets, 0)
            self._counts[float("inf")] = 0
            self._sum = 0.0
            self._count = 0


class LabeledCounter:
    """
    Counter with dynamic labels.

    Allows creating counter instances with different label values.
    """

    def __init__(self, name: str, help: str, label_names: list[str]):
        self.name = name
        self.help = help
        self.label_names = label_names
        self._counters: dict[tuple, Counter] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> Counter:
        """
        Get counter for specific label values.

        Example:
            jobs_by_status.labels(status="completed", module="analyzer").inc()
        """
        key = tuple(kwargs.get(name, "") for name in self.label_names)

        with self._lock:
            if key not in self._counters:
                label_dict = {name: kwargs.get(name, "") for name in self.label_names}
                self._counters[key] = Counter(
                    name=self.name,
                    help=self.help,
                    labels=label_dict,
                )
            return self._counters[key]

    def get_all(self) -> list[Counter]:
        """Get all counter instances."""
        with self._lock:
            return list(self._counters.values())


class LabeledGauge:
    """
    Gauge with dynamic labels.

    Allows creating gauge instances with different label values.
    """

    def __init__(self, name: str, help: str, label_names: list[str]):
        self.name = name
        self.help = help
        self.label_names = label_names
        self._gauges: dict[tuple, Gauge] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> Gauge:
        """Get gauge for specific label values."""
        key = tuple(kwargs.get(name, "") for name in self.label_names)

        with self._lock:
            if key not in self._gauges:
                label_dict = {name: kwargs.get(name, "") for name in self.label_names}
                self._gauges[key] = Gauge(
                    name=self.name,
                    help=self.help,
                    labels=label_dict,
                )
            return self._gauges[key]

    def get_all(self) -> list[Gauge]:
        """Get all gauge instances."""
        with self._lock:
            return list(self._gauges.values())


class MetricsRegistry:
    """
    Central registry for all Portal metrics.

    Provides a single point of access for all system metrics and
    handles Prometheus text format export.
    """

    def __init__(self):
        # Job lifecycle metrics
        self.jobs_submitted = Counter(
            "portal_jobs_submitted_total",
            "Total number of jobs submitted to the system"
        )
        self.jobs_completed = Counter(
            "portal_jobs_completed_total",
            "Total number of jobs completed successfully"
        )
        self.jobs_failed = Counter(
            "portal_jobs_failed_total",
            "Total number of jobs that failed"
        )
        self.jobs_queued = Counter(
            "portal_jobs_queued_total",
            "Total number of jobs added to queue (when nodes busy)"
        )
        self.jobs_cancelled = Counter(
            "portal_jobs_cancelled_total",
            "Total number of jobs cancelled by user"
        )

        # Job timing
        self.job_duration = Histogram(
            "portal_job_duration_ms",
            "Job execution duration in milliseconds",
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
        )

        # Node metrics
        self.nodes_online = Gauge(
            "portal_nodes_online",
            "Number of nodes currently online and healthy"
        )
        self.nodes_total = Gauge(
            "portal_nodes_total",
            "Total number of registered nodes"
        )

        # Per-node metrics (labeled)
        self.node_cpu = LabeledGauge(
            "portal_node_cpu_percent",
            "Node CPU utilization percentage",
            ["node_id"],
        )
        self.node_memory = LabeledGauge(
            "portal_node_memory_percent",
            "Node memory utilization percentage",
            ["node_id"],
        )
        self.node_jobs_running = LabeledGauge(
            "portal_node_jobs_running",
            "Number of jobs currently running on node",
            ["node_id"],
        )

        # Queue metrics
        self.queue_depth = Gauge(
            "portal_queue_depth",
            "Current number of jobs waiting in queue"
        )
        self.queue_wait_time = Histogram(
            "portal_queue_wait_time_ms",
            "Time jobs spend waiting in queue (milliseconds)",
            buckets=[100, 500, 1000, 2500, 5000, 10000, 30000, 60000],
        )

        # Scheduler metrics
        self.scheduler_selections = Counter(
            "portal_scheduler_selections_total",
            "Total number of node selection operations"
        )
        self.scheduler_fallbacks = Counter(
            "portal_scheduler_fallbacks_total",
            "Number of times scheduler fell back to less optimal strategy"
        )

        # Module metrics
        self.modules_loaded = Gauge(
            "portal_modules_loaded",
            "Number of WASM modules currently loaded"
        )
        self.module_load_errors = Counter(
            "portal_module_load_errors_total",
            "Total number of module loading errors"
        )

        # Per-module job counters
        self.jobs_by_module = LabeledCounter(
            "portal_jobs_by_module_total",
            "Jobs submitted by module ID",
            ["module_id"],
        )

        # Health metrics
        self.health_checks = Counter(
            "portal_health_checks_total",
            "Total number of health check operations"
        )
        self.health_check_failures = Counter(
            "portal_health_check_failures_total",
            "Number of failed health checks"
        )

        # Timestamp of last update
        self._last_update = time.time()

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus text format."""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_pairs) + "}"

    def _export_counter(self, counter: Counter) -> list[str]:
        """Export a counter in Prometheus text format."""
        lines = []
        lines.append(f"# HELP {counter.name} {counter.help}")
        lines.append(f"# TYPE {counter.name} counter")
        labels = self._format_labels(counter.labels)
        lines.append(f"{counter.name}{labels} {counter.value}")
        return lines

    def _export_gauge(self, gauge: Gauge) -> list[str]:
        """Export a gauge in Prometheus text format."""
        lines = []
        lines.append(f"# HELP {gauge.name} {gauge.help}")
        lines.append(f"# TYPE {gauge.name} gauge")
        labels = self._format_labels(gauge.labels)
        lines.append(f"{gauge.name}{labels} {gauge.value}")
        return lines

    def _export_histogram(self, histogram: Histogram) -> list[str]:
        """Export a histogram in Prometheus text format."""
        lines = []
        lines.append(f"# HELP {histogram.name} {histogram.help}")
        lines.append(f"# TYPE {histogram.name} histogram")

        base_labels = histogram.labels
        bucket_counts = histogram.get_bucket_counts()

        for bucket, count in sorted(bucket_counts.items()):
            if bucket == float("inf"):
                le_value = "+Inf"
            else:
                le_value = str(bucket)

            labels = {**base_labels, "le": le_value}
            label_str = self._format_labels(labels)
            lines.append(f"{histogram.name}_bucket{label_str} {count}")

        # Sum and count
        lines.append(f"{histogram.name}_sum {histogram.sum}")
        lines.append(f"{histogram.name}_count {histogram.count}")

        return lines

    def _export_labeled_counter(self, labeled_counter: LabeledCounter) -> list[str]:
        """Export a labeled counter in Prometheus text format."""
        counters = labeled_counter.get_all()
        if not counters:
            return []

        lines = []
        lines.append(f"# HELP {labeled_counter.name} {labeled_counter.help}")
        lines.append(f"# TYPE {labeled_counter.name} counter")

        for counter in counters:
            labels = self._format_labels(counter.labels)
            lines.append(f"{counter.name}{labels} {counter.value}")

        return lines

    def _export_labeled_gauge(self, labeled_gauge: LabeledGauge) -> list[str]:
        """Export a labeled gauge in Prometheus text format."""
        gauges = labeled_gauge.get_all()
        if not gauges:
            return []

        lines = []
        lines.append(f"# HELP {labeled_gauge.name} {labeled_gauge.help}")
        lines.append(f"# TYPE {labeled_gauge.name} gauge")

        for gauge in gauges:
            labels = self._format_labels(gauge.labels)
            lines.append(f"{gauge.name}{labels} {gauge.value}")

        return lines

    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns string suitable for /metrics endpoint.
        """
        lines = []

        # Simple counters
        lines.extend(self._export_counter(self.jobs_submitted))
        lines.extend(self._export_counter(self.jobs_completed))
        lines.extend(self._export_counter(self.jobs_failed))
        lines.extend(self._export_counter(self.jobs_queued))
        lines.extend(self._export_counter(self.jobs_cancelled))
        lines.extend(self._export_counter(self.scheduler_selections))
        lines.extend(self._export_counter(self.scheduler_fallbacks))
        lines.extend(self._export_counter(self.module_load_errors))
        lines.extend(self._export_counter(self.health_checks))
        lines.extend(self._export_counter(self.health_check_failures))

        # Simple gauges
        lines.extend(self._export_gauge(self.nodes_online))
        lines.extend(self._export_gauge(self.nodes_total))
        lines.extend(self._export_gauge(self.queue_depth))
        lines.extend(self._export_gauge(self.modules_loaded))

        # Histograms
        lines.extend(self._export_histogram(self.job_duration))
        lines.extend(self._export_histogram(self.queue_wait_time))

        # Labeled metrics
        lines.extend(self._export_labeled_gauge(self.node_cpu))
        lines.extend(self._export_labeled_gauge(self.node_memory))
        lines.extend(self._export_labeled_gauge(self.node_jobs_running))
        lines.extend(self._export_labeled_counter(self.jobs_by_module))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Export metrics as dictionary (for JSON APIs).

        Returns nested dict with all metric values.
        """
        return {
            "jobs": {
                "submitted": self.jobs_submitted.value,
                "completed": self.jobs_completed.value,
                "failed": self.jobs_failed.value,
                "queued": self.jobs_queued.value,
                "cancelled": self.jobs_cancelled.value,
                "duration": {
                    "sum": self.job_duration.sum,
                    "count": self.job_duration.count,
                    "avg_ms": (
                        self.job_duration.sum / self.job_duration.count
                        if self.job_duration.count > 0 else 0.0
                    ),
                },
            },
            "nodes": {
                "online": self.nodes_online.value,
                "total": self.nodes_total.value,
            },
            "queue": {
                "depth": self.queue_depth.value,
                "wait_time": {
                    "sum": self.queue_wait_time.sum,
                    "count": self.queue_wait_time.count,
                    "avg_ms": (
                        self.queue_wait_time.sum / self.queue_wait_time.count
                        if self.queue_wait_time.count > 0 else 0.0
                    ),
                },
            },
            "scheduler": {
                "selections": self.scheduler_selections.value,
                "fallbacks": self.scheduler_fallbacks.value,
            },
            "modules": {
                "loaded": self.modules_loaded.value,
                "load_errors": self.module_load_errors.value,
            },
            "health": {
                "checks": self.health_checks.value,
                "failures": self.health_check_failures.value,
            },
        }

    def reset_all(self) -> None:
        """
        Reset all metrics.

        Use for testing or when restarting metric collection.
        """
        self.jobs_submitted.reset()
        self.jobs_completed.reset()
        self.jobs_failed.reset()
        self.jobs_queued.reset()
        self.jobs_cancelled.reset()
        self.job_duration.reset()
        self.nodes_online.set(0)
        self.nodes_total.set(0)
        self.queue_depth.set(0)
        self.queue_wait_time.reset()
        self.scheduler_selections.reset()
        self.scheduler_fallbacks.reset()
        self.modules_loaded.set(0)
        self.module_load_errors.reset()
        self.health_checks.reset()
        self.health_check_failures.reset()


# Global metrics registry singleton
metrics = MetricsRegistry()
