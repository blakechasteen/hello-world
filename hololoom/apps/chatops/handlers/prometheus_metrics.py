"""
Prometheus Metrics for HoloLoom ChatOps Job Execution
======================================================

Provides observability metrics for the async job execution system:
- Job throughput (jobs/sec)
- Latency percentiles (p50, p95, p99)
- Job queue depth
- Job status distribution
- Error rates

Usage:
    from hololoom.apps.chatops.handlers.prometheus_metrics import (
        JobMetricsCollector,
        create_metrics_router
    )

    # Create collector
    collector = JobMetricsCollector()

    # Record job lifecycle events
    collector.job_submitted(job_id, query)
    collector.job_started(job_id)
    collector.job_completed(job_id, duration_ms, confidence)
    collector.job_failed(job_id, error)
    collector.job_cancelled(job_id)

    # Add FastAPI router for /metrics endpoint
    app.include_router(create_metrics_router(collector))
"""

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    help_text: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class LatencyHistogram:
    """Histogram for latency tracking with percentile calculation."""
    buckets: list[float] = field(default_factory=lambda: [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    observations: list[float] = field(default_factory=list)
    bucket_counts: dict[float, int] = field(default_factory=dict)
    sum_total: float = 0.0
    count: int = 0
    max_observations: int = 10000

    def __post_init__(self):
        # Initialize bucket counts
        for bucket in self.buckets:
            self.bucket_counts[bucket] = 0
        self.bucket_counts[float('inf')] = 0

    def observe(self, value: float):
        """Record an observation."""
        self.observations.append(value)
        self.sum_total += value
        self.count += 1

        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
        self.bucket_counts[float('inf')] += 1

        # Trim old observations to prevent memory growth
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100)."""
        if not self.observations:
            return 0.0
        sorted_obs = sorted(self.observations)
        idx = int(len(sorted_obs) * p / 100)
        idx = min(idx, len(sorted_obs) - 1)
        return sorted_obs[idx]

    def mean(self) -> float:
        """Calculate mean latency."""
        if not self.observations:
            return 0.0
        return statistics.mean(self.observations)


@dataclass
class JobMetrics:
    """Metrics for a single job."""
    job_id: str
    submitted_at: float
    started_at: float | None = None
    completed_at: float | None = None
    status: str = "pending"
    duration_ms: float | None = None
    confidence: float | None = None
    error: str | None = None
    query_length: int = 0
    tool_used: str | None = None


class JobMetricsCollector:
    """
    Collects and exposes Prometheus-format metrics for job execution.

    Metrics exported:
    - hololoom_jobs_total{status}: Counter of jobs by status
    - hololoom_jobs_pending: Gauge of pending jobs
    - hololoom_jobs_running: Gauge of running jobs
    - hololoom_job_duration_seconds_bucket: Histogram of job durations
    - hololoom_job_duration_seconds_sum: Sum of job durations
    - hololoom_job_duration_seconds_count: Count of completed jobs
    - hololoom_job_confidence: Summary of job confidence scores
    - hololoom_job_queue_depth: Current queue depth
    - hololoom_jobs_per_minute: Rolling throughput
    """

    def __init__(self, window_size_minutes: int = 5):
        self._lock = threading.RLock()  # RLock allows re-entrant locking (fixes deadlock in get_summary)
        self._jobs: dict[str, JobMetrics] = {}
        self._completed_jobs: deque = deque(maxlen=10000)  # Rolling window

        # Counters
        self._jobs_total: dict[str, int] = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }

        # Gauges
        self._jobs_pending = 0
        self._jobs_running = 0

        # Histograms
        self._duration_histogram = LatencyHistogram(
            buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]
        )
        self._confidence_histogram = LatencyHistogram(
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        # Rolling throughput tracking
        self._window_size_minutes = window_size_minutes
        self._job_timestamps: deque = deque()

        # Error tracking
        self._error_counts: dict[str, int] = {}

        # Tool usage tracking
        self._tool_counts: dict[str, int] = {}

        logger.info("JobMetricsCollector initialized")

    def job_submitted(self, job_id: str, query: str, room_id: str = "", user_id: str = ""):
        """Record job submission."""
        with self._lock:
            self._jobs[job_id] = JobMetrics(
                job_id=job_id,
                submitted_at=time.time(),
                status="pending",
                query_length=len(query)
            )
            self._jobs_total["pending"] += 1
            self._jobs_pending += 1
            self._job_timestamps.append(time.time())

            # Clean old timestamps
            cutoff = time.time() - (self._window_size_minutes * 60)
            while self._job_timestamps and self._job_timestamps[0] < cutoff:
                self._job_timestamps.popleft()

    def job_started(self, job_id: str):
        """Record job start."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.started_at = time.time()
                job.status = "running"
                self._jobs_pending -= 1
                self._jobs_running += 1
                self._jobs_total["running"] += 1

    def job_completed(
        self,
        job_id: str,
        duration_ms: float,
        confidence: float,
        tool_used: str = ""
    ):
        """Record job completion."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.completed_at = time.time()
                job.status = "completed"
                job.duration_ms = duration_ms
                job.confidence = confidence
                job.tool_used = tool_used

                self._jobs_running -= 1
                self._jobs_total["completed"] += 1

                # Record in histograms
                self._duration_histogram.observe(duration_ms)
                self._confidence_histogram.observe(confidence)

                # Track tool usage
                if tool_used:
                    self._tool_counts[tool_used] = self._tool_counts.get(tool_used, 0) + 1

                # Move to completed queue
                self._completed_jobs.append(job)
                del self._jobs[job_id]

    def job_failed(self, job_id: str, error: str):
        """Record job failure."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.completed_at = time.time()
                job.status = "failed"
                job.error = error

                if job.started_at:
                    job.duration_ms = (job.completed_at - job.started_at) * 1000
                    self._duration_histogram.observe(job.duration_ms)

                self._jobs_running = max(0, self._jobs_running - 1)
                self._jobs_total["failed"] += 1

                # Track error type
                error_type = error.split(":")[0] if error else "unknown"
                self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

                # Move to completed queue
                self._completed_jobs.append(job)
                del self._jobs[job_id]

    def job_cancelled(self, job_id: str):
        """Record job cancellation."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.completed_at = time.time()
                job.status = "cancelled"

                if job.started_at:
                    job.duration_ms = (job.completed_at - job.started_at) * 1000

                # Adjust gauges based on previous state
                if job.status == "pending":
                    self._jobs_pending = max(0, self._jobs_pending - 1)
                else:
                    self._jobs_running = max(0, self._jobs_running - 1)

                self._jobs_total["cancelled"] += 1

                # Move to completed queue
                self._completed_jobs.append(job)
                del self._jobs[job_id]

    def get_jobs_per_minute(self) -> float:
        """Calculate rolling jobs per minute."""
        with self._lock:
            # Clean old timestamps
            cutoff = time.time() - (self._window_size_minutes * 60)
            while self._job_timestamps and self._job_timestamps[0] < cutoff:
                self._job_timestamps.popleft()

            count = len(self._job_timestamps)
            return count / self._window_size_minutes if self._window_size_minutes > 0 else 0

    def get_metrics(self) -> list[MetricValue]:
        """Get all metrics in Prometheus format."""
        metrics = []

        with self._lock:
            # Job totals by status
            for status, count in self._jobs_total.items():
                metrics.append(MetricValue(
                    name="hololoom_jobs_total",
                    value=count,
                    labels={"status": status},
                    metric_type=MetricType.COUNTER,
                    help_text="Total number of jobs by status"
                ))

            # Current queue gauges
            metrics.append(MetricValue(
                name="hololoom_jobs_pending",
                value=self._jobs_pending,
                metric_type=MetricType.GAUGE,
                help_text="Number of pending jobs"
            ))
            metrics.append(MetricValue(
                name="hololoom_jobs_running",
                value=self._jobs_running,
                metric_type=MetricType.GAUGE,
                help_text="Number of running jobs"
            ))
            metrics.append(MetricValue(
                name="hololoom_job_queue_depth",
                value=self._jobs_pending + self._jobs_running,
                metric_type=MetricType.GAUGE,
                help_text="Total jobs in queue (pending + running)"
            ))

            # Duration histogram buckets
            for bucket, count in self._duration_histogram.bucket_counts.items():
                bucket_label = "+Inf" if bucket == float('inf') else str(bucket)
                metrics.append(MetricValue(
                    name="hololoom_job_duration_ms_bucket",
                    value=count,
                    labels={"le": bucket_label},
                    metric_type=MetricType.HISTOGRAM,
                    help_text="Job duration histogram (milliseconds)"
                ))

            metrics.append(MetricValue(
                name="hololoom_job_duration_ms_sum",
                value=self._duration_histogram.sum_total,
                metric_type=MetricType.HISTOGRAM,
                help_text="Sum of job durations"
            ))
            metrics.append(MetricValue(
                name="hololoom_job_duration_ms_count",
                value=self._duration_histogram.count,
                metric_type=MetricType.HISTOGRAM,
                help_text="Count of completed jobs"
            ))

            # Latency percentiles
            for p in [50, 90, 95, 99]:
                metrics.append(MetricValue(
                    name="hololoom_job_duration_ms",
                    value=self._duration_histogram.percentile(p),
                    labels={"quantile": f"0.{p}" if p < 100 else "1.0"},
                    metric_type=MetricType.SUMMARY,
                    help_text="Job duration percentiles"
                ))

            # Confidence histogram
            for bucket, count in self._confidence_histogram.bucket_counts.items():
                bucket_label = "+Inf" if bucket == float('inf') else str(bucket)
                metrics.append(MetricValue(
                    name="hololoom_job_confidence_bucket",
                    value=count,
                    labels={"le": bucket_label},
                    metric_type=MetricType.HISTOGRAM,
                    help_text="Job confidence histogram"
                ))

            # Throughput
            metrics.append(MetricValue(
                name="hololoom_jobs_per_minute",
                value=self.get_jobs_per_minute(),
                metric_type=MetricType.GAUGE,
                help_text="Rolling jobs per minute"
            ))

            # Error counts
            for error_type, count in self._error_counts.items():
                metrics.append(MetricValue(
                    name="hololoom_job_errors_total",
                    value=count,
                    labels={"error_type": error_type},
                    metric_type=MetricType.COUNTER,
                    help_text="Total job errors by type"
                ))

            # Tool usage
            for tool, count in self._tool_counts.items():
                metrics.append(MetricValue(
                    name="hololoom_tool_usage_total",
                    value=count,
                    labels={"tool": tool},
                    metric_type=MetricType.COUNTER,
                    help_text="Tool usage count"
                ))

        return metrics

    def format_prometheus(self) -> str:
        """Format metrics as Prometheus text exposition format."""
        lines = []
        metrics = self.get_metrics()

        # Group by metric name
        by_name: dict[str, list[MetricValue]] = {}
        for m in metrics:
            if m.name not in by_name:
                by_name[m.name] = []
            by_name[m.name].append(m)

        for name, metric_list in sorted(by_name.items()):
            if metric_list:
                first = metric_list[0]
                lines.append(f"# HELP {name} {first.help_text}")
                lines.append(f"# TYPE {name} {first.metric_type.value}")

                for m in metric_list:
                    if m.labels:
                        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(m.labels.items()))
                        lines.append(f"{name}{{{label_str}}} {m.value}")
                    else:
                        lines.append(f"{name} {m.value}")

                lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary dict for JSON export."""
        with self._lock:
            return {
                "jobs_total": dict(self._jobs_total),
                "jobs_pending": self._jobs_pending,
                "jobs_running": self._jobs_running,
                "queue_depth": self._jobs_pending + self._jobs_running,
                "jobs_per_minute": self.get_jobs_per_minute(),
                "latency_ms": {
                    "p50": self._duration_histogram.percentile(50),
                    "p90": self._duration_histogram.percentile(90),
                    "p95": self._duration_histogram.percentile(95),
                    "p99": self._duration_histogram.percentile(99),
                    "mean": self._duration_histogram.mean()
                },
                "confidence": {
                    "p50": self._confidence_histogram.percentile(50),
                    "p90": self._confidence_histogram.percentile(90),
                    "mean": self._confidence_histogram.mean()
                },
                "error_counts": dict(self._error_counts),
                "tool_usage": dict(self._tool_counts),
                "total_completed": self._duration_histogram.count
            }


def create_metrics_router(collector: JobMetricsCollector):
    """
    Create FastAPI router for Prometheus metrics endpoint.

    Returns:
        APIRouter with /metrics endpoint
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError:
        logger.warning("FastAPI not available, metrics router disabled")
        return None

    router = APIRouter(tags=["metrics"])

    @router.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return collector.format_prometheus()

    @router.get("/metrics/json")
    async def metrics_json():
        """JSON metrics endpoint for dashboards."""
        return JSONResponse(collector.get_summary())

    return router


# Global collector instance (can be replaced)
_global_collector: JobMetricsCollector | None = None


def get_metrics_collector() -> JobMetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = JobMetricsCollector()
    return _global_collector


def set_metrics_collector(collector: JobMetricsCollector):
    """Set the global metrics collector."""
    global _global_collector
    _global_collector = collector
