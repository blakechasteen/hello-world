"""
CARTS Sandbox Metrics Collection
=================================

Prometheus metrics for sandbox deployment monitoring.

Metrics exported:
- carts_sandbox_jobs_total: Total jobs by status
- carts_sandbox_execution_seconds: Execution time histogram
- carts_sandbox_resource_usage: Resource usage gauges
- carts_sandbox_violations_total: Sandbox violations counter

Author: CARTS Team
Date: 2025-12-09
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger("carts-deploy.metrics")


@dataclass
class JobMetrics:
    """Metrics for a single job execution."""
    job_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, succeeded, failed, timeout
    exit_code: int = 0
    execution_time_seconds: float = 0.0
    cpu_seconds: float = 0.0
    memory_mb_max: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    violations: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and exports Prometheus metrics for CARTS sandbox."""

    def __init__(self, namespace: str = "carts"):
        self.namespace = namespace
        self.jobs: Dict[str, JobMetrics] = {}
        self.counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.gauges: Dict[str, float] = {}

    def start_job(self, job_id: str, labels: Dict[str, str] = None) -> None:
        """Record job start."""
        self.jobs[job_id] = JobMetrics(
            job_id=job_id,
            start_time=time.time(),
            labels=labels or {},
        )
        self.counters["jobs_started"]["total"] += 1
        logger.debug(f"Job started: {job_id}")

    def end_job(
        self,
        job_id: str,
        status: str,
        exit_code: int = 0,
        resource_usage: Dict[str, Any] = None,
    ) -> None:
        """Record job completion."""
        if job_id not in self.jobs:
            logger.warning(f"Unknown job: {job_id}")
            return

        job = self.jobs[job_id]
        job.end_time = time.time()
        job.status = status
        job.exit_code = exit_code
        job.execution_time_seconds = job.end_time - job.start_time

        # Update resource metrics
        if resource_usage:
            job.cpu_seconds = resource_usage.get("cpu_seconds", 0)
            job.memory_mb_max = resource_usage.get("memory_mb_max", 0)
            job.network_bytes_sent = resource_usage.get("network_tx", 0)
            job.network_bytes_received = resource_usage.get("network_rx", 0)

        # Update counters
        self.counters["jobs_completed"][status] += 1
        self.histograms["execution_seconds"].append(job.execution_time_seconds)

        logger.debug(f"Job ended: {job_id} ({status})")

    def record_violation(self, job_id: str, violation_type: str) -> None:
        """Record a sandbox violation."""
        if job_id in self.jobs:
            self.jobs[job_id].violations.append(violation_type)
        self.counters["violations"][violation_type] += 1
        logger.warning(f"Violation recorded: {job_id} - {violation_type}")

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self.gauges[name] = value

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Jobs counter
        lines.append(f"# HELP {self.namespace}_sandbox_jobs_total Total sandbox jobs")
        lines.append(f"# TYPE {self.namespace}_sandbox_jobs_total counter")
        for status, count in self.counters["jobs_completed"].items():
            lines.append(f'{self.namespace}_sandbox_jobs_total{{status="{status}"}} {count}')

        # Violations counter
        lines.append(f"# HELP {self.namespace}_sandbox_violations_total Sandbox violations")
        lines.append(f"# TYPE {self.namespace}_sandbox_violations_total counter")
        for vtype, count in self.counters["violations"].items():
            lines.append(f'{self.namespace}_sandbox_violations_total{{type="{vtype}"}} {count}')

        # Execution time histogram
        if self.histograms["execution_seconds"]:
            values = sorted(self.histograms["execution_seconds"])
            lines.append(f"# HELP {self.namespace}_sandbox_execution_seconds Execution time")
            lines.append(f"# TYPE {self.namespace}_sandbox_execution_seconds histogram")

            buckets = [0.1, 0.5, 1, 5, 10, 30, 60, 120, 300]
            cumulative = 0
            for bucket in buckets:
                count = sum(1 for v in values if v <= bucket)
                lines.append(f'{self.namespace}_sandbox_execution_seconds_bucket{{le="{bucket}"}} {count}')
            lines.append(f'{self.namespace}_sandbox_execution_seconds_bucket{{le="+Inf"}} {len(values)}')
            lines.append(f'{self.namespace}_sandbox_execution_seconds_sum {sum(values):.3f}')
            lines.append(f'{self.namespace}_sandbox_execution_seconds_count {len(values)}')

        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# HELP {self.namespace}_sandbox_{name} {name}")
            lines.append(f"# TYPE {self.namespace}_sandbox_{name} gauge")
            lines.append(f"{self.namespace}_sandbox_{name} {value}")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON."""
        execution_times = self.histograms["execution_seconds"]

        return {
            "jobs": {
                "total": sum(self.counters["jobs_completed"].values()),
                "by_status": dict(self.counters["jobs_completed"]),
            },
            "violations": {
                "total": sum(self.counters["violations"].values()),
                "by_type": dict(self.counters["violations"]),
            },
            "execution_time": {
                "count": len(execution_times),
                "sum_seconds": sum(execution_times) if execution_times else 0,
                "avg_seconds": sum(execution_times) / len(execution_times) if execution_times else 0,
                "min_seconds": min(execution_times) if execution_times else 0,
                "max_seconds": max(execution_times) if execution_times else 0,
            },
            "gauges": dict(self.gauges),
            "active_jobs": len([j for j in self.jobs.values() if j.status == "running"]),
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        data = self.export_json()
        lines = [
            "=== CARTS Sandbox Metrics ===",
            f"Total Jobs: {data['jobs']['total']}",
            f"  Succeeded: {data['jobs']['by_status'].get('succeeded', 0)}",
            f"  Failed: {data['jobs']['by_status'].get('failed', 0)}",
            f"  Timeout: {data['jobs']['by_status'].get('timeout', 0)}",
            f"Violations: {data['violations']['total']}",
            f"Avg Execution Time: {data['execution_time']['avg_seconds']:.2f}s",
            f"Active Jobs: {data['active_jobs']}",
        ]
        return "\n".join(lines)


# Global metrics instance
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics


def start_job(job_id: str, labels: Dict[str, str] = None) -> None:
    """Record job start on global collector."""
    _metrics.start_job(job_id, labels)


def end_job(
    job_id: str,
    status: str,
    exit_code: int = 0,
    resource_usage: Dict[str, Any] = None,
) -> None:
    """Record job end on global collector."""
    _metrics.end_job(job_id, status, exit_code, resource_usage)


def record_violation(job_id: str, violation_type: str) -> None:
    """Record violation on global collector."""
    _metrics.record_violation(job_id, violation_type)


def export_prometheus() -> str:
    """Export Prometheus metrics."""
    return _metrics.export_prometheus()
