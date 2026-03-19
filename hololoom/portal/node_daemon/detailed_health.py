"""
Detailed Health Diagnostics - Enhanced node health reporting.

Provides comprehensive health metrics beyond simple HTTP 200 checks:
- CPU and memory trend tracking
- Memory leak detection (monotonic increase)
- Job queue depth and completion stats
- Rolling window analysis for trend detection
- Module loading error tracking

Usage:
    monitor = HealthMonitor(sample_interval=30.0, max_samples=12)

    # Record periodic samples
    monitor.record_sample(job_queue_depth=5)

    # Record job completions
    monitor.record_job_completion(duration_ms=150.0)

    # Get detailed health report
    health = monitor.get_detailed_health(
        jobs_running=3,
        module_count=10,
        max_concurrent=10
    )
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HealthSample:
    """Point-in-time health sample."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: int
    job_queue_depth: int


@dataclass
class DetailedHealth:
    """
    Comprehensive node health status.

    Goes beyond simple "is it responding?" to provide:
    - Resource utilization metrics
    - Job throughput statistics
    - Trend data for capacity planning
    - Derived insights (memory leaks, available slots)
    """
    # Overall status
    healthy: bool
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)

    # Current resource metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: int = 0
    memory_available_mb: int = 0

    # Job metrics
    job_queue_depth: int = 0
    jobs_running: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    avg_job_duration_ms: float = 0.0

    # System info
    uptime_seconds: float = 0.0
    module_count: int = 0
    module_load_errors: list[str] = field(default_factory=list)

    # Trend data (last 5 minutes, 12 samples @ 30s by default)
    memory_trend: list[float] = field(default_factory=list)
    cpu_trend: list[float] = field(default_factory=list)

    # Derived insights
    memory_leak_suspected: bool = False
    high_cpu_sustained: bool = False
    estimated_available_slots: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "healthy": self.healthy,
            "degraded": self.degraded,
            "degradation_reasons": self.degradation_reasons,
            "resources": {
                "cpu_percent": round(self.cpu_percent, 1),
                "memory_percent": round(self.memory_percent, 1),
                "memory_mb": self.memory_mb,
                "memory_available_mb": self.memory_available_mb,
            },
            "jobs": {
                "queue_depth": self.job_queue_depth,
                "running": self.jobs_running,
                "completed": self.jobs_completed,
                "failed": self.jobs_failed,
                "avg_duration_ms": round(self.avg_job_duration_ms, 2),
            },
            "system": {
                "uptime_seconds": round(self.uptime_seconds, 0),
                "module_count": self.module_count,
                "module_load_errors": self.module_load_errors[-5:],  # Last 5
            },
            "trends": {
                "memory": [round(m, 1) for m in self.memory_trend[-10:]],
                "cpu": [round(c, 1) for c in self.cpu_trend[-10:]],
            },
            "insights": {
                "memory_leak_suspected": self.memory_leak_suspected,
                "high_cpu_sustained": self.high_cpu_sustained,
                "estimated_available_slots": self.estimated_available_slots,
            },
        }


class HealthMonitor:
    """
    Tracks node health over time for trend analysis.

    Periodically samples system metrics and maintains a rolling window
    for trend detection. Detects issues like:
    - Memory leaks (monotonic memory increase)
    - Sustained high CPU usage
    - Growing job queue depth

    Args:
        sample_interval: Seconds between samples (default: 30s)
        max_samples: Number of samples to retain (default: 12 = 6 minutes)
        memory_leak_threshold: Consecutive increases to trigger warning (default: 5)
        high_cpu_threshold: CPU % considered high (default: 80%)
        high_memory_threshold: Memory % considered high (default: 85%)
    """

    def __init__(
        self,
        sample_interval: float = 30.0,
        max_samples: int = 12,
        memory_leak_threshold: int = 5,
        high_cpu_threshold: float = 80.0,
        high_memory_threshold: float = 85.0,
    ):
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.memory_leak_threshold = memory_leak_threshold
        self.high_cpu_threshold = high_cpu_threshold
        self.high_memory_threshold = high_memory_threshold

        self._samples: deque[HealthSample] = deque(maxlen=max_samples)
        self._start_time = time.time()
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._job_durations: deque[float] = deque(maxlen=100)
        self._module_errors: list[str] = []

    def record_sample(
        self,
        job_queue_depth: int,
        cpu_percent: float | None = None,
        memory_percent: float | None = None,
        memory_mb: int | None = None,
    ) -> None:
        """
        Record a health sample.

        If cpu/memory metrics not provided, attempts to get from psutil
        if available, otherwise uses 0.

        Args:
            job_queue_depth: Current number of jobs in queue
            cpu_percent: CPU utilization (optional, auto-detected)
            memory_percent: Memory utilization (optional, auto-detected)
            memory_mb: Memory used in MB (optional, auto-detected)
        """
        # Try to get system metrics from psutil if not provided
        if cpu_percent is None or memory_percent is None or memory_mb is None:
            try:
                import psutil
                if cpu_percent is None:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                if memory_percent is None or memory_mb is None:
                    mem = psutil.virtual_memory()
                    if memory_percent is None:
                        memory_percent = mem.percent
                    if memory_mb is None:
                        memory_mb = int(mem.used / 1024 / 1024)
            except ImportError:
                # psutil not available, use defaults
                if cpu_percent is None:
                    cpu_percent = 0.0
                if memory_percent is None:
                    memory_percent = 0.0
                if memory_mb is None:
                    memory_mb = 0

        sample = HealthSample(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            job_queue_depth=job_queue_depth,
        )
        self._samples.append(sample)

    def record_job_completion(self, duration_ms: float, success: bool = True) -> None:
        """
        Record a completed job's duration.

        Args:
            duration_ms: Job execution time in milliseconds
            success: Whether job completed successfully
        """
        if success:
            self._jobs_completed += 1
        else:
            self._jobs_failed += 1
        self._job_durations.append(duration_ms)

    def record_module_error(self, error: str) -> None:
        """
        Record a module loading error.

        Args:
            error: Error message
        """
        self._module_errors.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {error}")
        if len(self._module_errors) > 50:
            self._module_errors = self._module_errors[-50:]

    def _detect_memory_leak(self) -> bool:
        """
        Detect potential memory leak (monotonic increase over threshold samples).

        Returns:
            True if memory leak is suspected
        """
        if len(self._samples) < self.memory_leak_threshold:
            return False

        # Check last N samples for monotonic increase
        recent = list(self._samples)[-self.memory_leak_threshold:]
        for i in range(1, len(recent)):
            if recent[i].memory_percent <= recent[i-1].memory_percent:
                return False  # Not monotonically increasing

        return True

    def _detect_high_cpu_sustained(self) -> bool:
        """
        Detect sustained high CPU usage.

        Returns:
            True if CPU has been high for multiple samples
        """
        if len(self._samples) < 3:
            return False

        # Check if last 3 samples all have high CPU
        recent = list(self._samples)[-3:]
        return all(s.cpu_percent > self.high_cpu_threshold for s in recent)

    def _get_memory_info(self) -> tuple[float, int, int]:
        """Get current memory info from psutil or defaults."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.percent, int(mem.used / 1024 / 1024), int(mem.available / 1024 / 1024)
        except ImportError:
            # Use last sample if available
            if self._samples:
                last = self._samples[-1]
                return last.memory_percent, last.memory_mb, 0
            return 0.0, 0, 0

    def _get_cpu_percent(self) -> float:
        """Get current CPU percent from psutil or defaults."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            if self._samples:
                return self._samples[-1].cpu_percent
            return 0.0

    def get_detailed_health(
        self,
        jobs_running: int = 0,
        module_count: int = 0,
        max_concurrent: int = 10,
    ) -> DetailedHealth:
        """
        Generate comprehensive health report.

        Args:
            jobs_running: Number of currently running jobs
            module_count: Number of loaded modules
            max_concurrent: Maximum concurrent jobs allowed

        Returns:
            DetailedHealth with all metrics and insights
        """
        # Use recorded sample metrics if available, fallback to live psutil
        if self._samples:
            # Use most recent sample for threshold checks (this is what was recorded)
            last_sample = self._samples[-1]
            memory_percent = last_sample.memory_percent
            memory_mb = last_sample.memory_mb
            cpu_percent = last_sample.cpu_percent
            # Still get live available memory if possible
            _, _, memory_available_mb = self._get_memory_info()
        else:
            # No samples yet, get live metrics
            memory_percent, memory_mb, memory_available_mb = self._get_memory_info()
            cpu_percent = self._get_cpu_percent()

        # Extract trends
        memory_trend = [s.memory_percent for s in self._samples]
        cpu_trend = [s.cpu_percent for s in self._samples]

        # Detect issues
        memory_leak_suspected = self._detect_memory_leak()
        high_cpu_sustained = self._detect_high_cpu_sustained()

        # Calculate derived metrics
        estimated_available_slots = max(0, max_concurrent - jobs_running)
        avg_duration = (
            sum(self._job_durations) / len(self._job_durations)
            if self._job_durations else 0.0
        )

        # Determine health status
        degradation_reasons: list[str] = []

        if memory_percent > 90:
            degradation_reasons.append("Critical memory usage (>90%)")
        elif memory_percent > self.high_memory_threshold:
            degradation_reasons.append(f"High memory usage (>{self.high_memory_threshold}%)")

        if memory_leak_suspected:
            degradation_reasons.append("Potential memory leak detected")

        if high_cpu_sustained:
            degradation_reasons.append("Sustained high CPU usage")

        if self._samples and self._samples[-1].job_queue_depth > 10:
            degradation_reasons.append("High job queue depth (>10)")

        # Recent module errors?
        recent_errors = [e for e in self._module_errors if time.time() - float(e.split(':')[0].replace('-', '').replace(' ', '')) < 300]
        if len(recent_errors) > 3:
            degradation_reasons.append(f"Recent module load errors ({len(recent_errors)})")

        healthy = len(degradation_reasons) == 0 and memory_percent < 90
        degraded = len(degradation_reasons) > 0 and memory_percent < 90

        return DetailedHealth(
            healthy=healthy,
            degraded=degraded,
            degradation_reasons=degradation_reasons,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            memory_available_mb=memory_available_mb,
            job_queue_depth=self._samples[-1].job_queue_depth if self._samples else 0,
            jobs_running=jobs_running,
            jobs_completed=self._jobs_completed,
            jobs_failed=self._jobs_failed,
            avg_job_duration_ms=avg_duration,
            uptime_seconds=time.time() - self._start_time,
            module_count=module_count,
            module_load_errors=self._module_errors[-5:],
            memory_trend=memory_trend,
            cpu_trend=cpu_trend,
            memory_leak_suspected=memory_leak_suspected,
            high_cpu_sustained=high_cpu_sustained,
            estimated_available_slots=estimated_available_slots,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics for diagnostics."""
        return {
            "sample_count": len(self._samples),
            "max_samples": self.max_samples,
            "sample_interval": self.sample_interval,
            "uptime_seconds": round(time.time() - self._start_time, 0),
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "module_errors_count": len(self._module_errors),
        }

    def reset(self) -> None:
        """Reset all tracked data (useful for testing)."""
        self._samples.clear()
        self._start_time = time.time()
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._job_durations.clear()
        self._module_errors.clear()
