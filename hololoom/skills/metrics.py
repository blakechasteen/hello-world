"""Skill Metrics - Performance tracking and monitoring"""

import time
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime

from HoloLoom.skills.base import SkillStatus


@dataclass
class SkillMetrics:
    """Metrics for a single skill."""
    skill_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    timeout_executions: int = 0

    # Timing metrics (in milliseconds)
    total_time_ms: float = 0.0
    min_time_ms: Optional[float] = None
    max_time_ms: Optional[float] = None
    execution_times: List[float] = field(default_factory=list)

    # Operation breakdown
    operation_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    operation_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    # Timestamps
    first_execution: Optional[float] = None
    last_execution: Optional[float] = None

    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        if self.total_executions == 0:
            return 0.0
        return self.total_time_ms / self.total_executions

    @property
    def median_time_ms(self) -> float:
        """Median execution time."""
        if not self.execution_times:
            return 0.0
        return statistics.median(self.execution_times)

    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0.0-1.0)."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.cache_hits / total_cache_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "skill_name": self.skill_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "timeout_executions": self.timeout_executions,
            "success_rate": self.success_rate,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "median_time_ms": self.median_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "operation_counts": dict(self.operation_counts),
            "first_execution": self.first_execution,
            "last_execution": self.last_execution
        }


@dataclass
class TimeSeriesMetrics:
    """Time series metrics tracking."""
    window_size: int = 100  # Keep last N measurements

    # Recent execution times
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    # Recent success/failure counts
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_execution(self, time_ms: float, success: bool):
        """Add execution to time series."""
        self.recent_times.append(time_ms)
        if success:
            self.recent_successes.append(1)
            self.recent_failures.append(0)
        else:
            self.recent_successes.append(0)
            self.recent_failures.append(1)

    @property
    def recent_avg_time(self) -> float:
        """Average time in recent window."""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)

    @property
    def recent_success_rate(self) -> float:
        """Success rate in recent window."""
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)

    @property
    def is_degrading(self) -> bool:
        """Check if performance is degrading."""
        if len(self.recent_times) < self.window_size:
            return False

        # Compare first half vs second half of window
        mid = len(self.recent_times) // 2
        first_half = list(self.recent_times)[:mid]
        second_half = list(self.recent_times)[mid:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # Degrading if second half is >20% slower
        return avg_second > avg_first * 1.2


class SkillMetricsTracker:
    """
    Central metrics tracker for all skills.

    Features:
    - Per-skill metrics tracking
    - Time series analysis
    - Anomaly detection
    - Performance reporting
    """

    def __init__(self, enable_time_series: bool = True):
        """
        Initialize metrics tracker.

        Args:
            enable_time_series: Whether to track time series metrics
        """
        self.enable_time_series = enable_time_series

        # Per-skill metrics
        self._skill_metrics: Dict[str, SkillMetrics] = {}

        # Time series metrics (if enabled)
        self._time_series: Dict[str, TimeSeriesMetrics] = {}

        # Global metrics
        self._global_start_time = time.time()

    def record_execution(
        self,
        skill_name: str,
        operation: str,
        status: SkillStatus,
        execution_time_ms: float,
        cache_hit: bool = False
    ):
        """
        Record a skill execution.

        Args:
            skill_name: Name of skill
            operation: Operation executed
            status: Execution status
            execution_time_ms: Execution time
            cache_hit: Whether result was cached
        """
        # Get or create skill metrics
        if skill_name not in self._skill_metrics:
            self._skill_metrics[skill_name] = SkillMetrics(skill_name=skill_name)

        metrics = self._skill_metrics[skill_name]

        # Update counters
        metrics.total_executions += 1

        if status == SkillStatus.SUCCESS:
            metrics.successful_executions += 1
        elif status == SkillStatus.TIMEOUT:
            metrics.timeout_executions += 1
        else:
            metrics.failed_executions += 1

        # Update timing
        metrics.total_time_ms += execution_time_ms
        metrics.execution_times.append(execution_time_ms)

        if metrics.min_time_ms is None or execution_time_ms < metrics.min_time_ms:
            metrics.min_time_ms = execution_time_ms

        if metrics.max_time_ms is None or execution_time_ms > metrics.max_time_ms:
            metrics.max_time_ms = execution_time_ms

        # Update operation breakdown
        metrics.operation_counts[operation] += 1
        metrics.operation_times[operation].append(execution_time_ms)

        # Update cache metrics
        if cache_hit:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1

        # Update timestamps
        current_time = time.time()
        if metrics.first_execution is None:
            metrics.first_execution = current_time
        metrics.last_execution = current_time

        # Update time series if enabled
        if self.enable_time_series:
            if skill_name not in self._time_series:
                self._time_series[skill_name] = TimeSeriesMetrics()

            self._time_series[skill_name].add_execution(
                execution_time_ms,
                status == SkillStatus.SUCCESS
            )

    def get_skill_metrics(self, skill_name: str) -> Optional[SkillMetrics]:
        """
        Get metrics for a specific skill.

        Args:
            skill_name: Name of skill

        Returns:
            SkillMetrics or None if skill not found
        """
        return self._skill_metrics.get(skill_name)

    def get_all_metrics(self) -> Dict[str, SkillMetrics]:
        """
        Get metrics for all skills.

        Returns:
            Dict of skill_name -> SkillMetrics
        """
        return self._skill_metrics.copy()

    def get_top_skills(self, n: int = 10, by: str = "executions") -> List[SkillMetrics]:
        """
        Get top N skills.

        Args:
            n: Number of skills to return
            by: Sort key ("executions", "time", "success_rate")

        Returns:
            List of SkillMetrics sorted by key
        """
        skills = list(self._skill_metrics.values())

        if by == "executions":
            skills.sort(key=lambda m: m.total_executions, reverse=True)
        elif by == "time":
            skills.sort(key=lambda m: m.total_time_ms, reverse=True)
        elif by == "success_rate":
            skills.sort(key=lambda m: m.success_rate, reverse=True)
        else:
            raise ValueError(f"Unknown sort key: {by}")

        return skills[:n]

    def get_slow_skills(self, threshold_ms: float = 1000.0) -> List[SkillMetrics]:
        """
        Get skills with high average execution time.

        Args:
            threshold_ms: Threshold in milliseconds

        Returns:
            List of slow skills
        """
        slow_skills = [
            metrics for metrics in self._skill_metrics.values()
            if metrics.avg_time_ms > threshold_ms
        ]

        slow_skills.sort(key=lambda m: m.avg_time_ms, reverse=True)
        return slow_skills

    def get_failing_skills(self, min_failure_rate: float = 0.1) -> List[SkillMetrics]:
        """
        Get skills with high failure rate.

        Args:
            min_failure_rate: Minimum failure rate (0.0-1.0)

        Returns:
            List of failing skills
        """
        failing_skills = [
            metrics for metrics in self._skill_metrics.values()
            if (1.0 - metrics.success_rate) >= min_failure_rate
        ]

        failing_skills.sort(key=lambda m: m.success_rate)
        return failing_skills

    def get_degrading_skills(self) -> List[str]:
        """
        Get skills with degrading performance.

        Returns:
            List of skill names showing degradation
        """
        if not self.enable_time_series:
            return []

        degrading = [
            skill_name for skill_name, ts in self._time_series.items()
            if ts.is_degrading
        ]

        return degrading

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall metrics summary.

        Returns:
            Dict with summary metrics
        """
        if not self._skill_metrics:
            return {
                "total_skills": 0,
                "total_executions": 0,
                "uptime_seconds": time.time() - self._global_start_time
            }

        all_metrics = list(self._skill_metrics.values())

        total_executions = sum(m.total_executions for m in all_metrics)
        total_successes = sum(m.successful_executions for m in all_metrics)
        total_failures = sum(m.failed_executions for m in all_metrics)
        total_timeouts = sum(m.timeout_executions for m in all_metrics)

        total_time = sum(m.total_time_ms for m in all_metrics)
        avg_time = total_time / total_executions if total_executions > 0 else 0.0

        total_cache_hits = sum(m.cache_hits for m in all_metrics)
        total_cache_misses = sum(m.cache_misses for m in all_metrics)
        total_cache_requests = total_cache_hits + total_cache_misses
        cache_hit_rate = total_cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0

        return {
            "total_skills": len(self._skill_metrics),
            "total_executions": total_executions,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "total_timeouts": total_timeouts,
            "success_rate": total_successes / total_executions if total_executions > 0 else 0.0,
            "total_time_ms": total_time,
            "avg_time_ms": avg_time,
            "cache_hit_rate": cache_hit_rate,
            "uptime_seconds": time.time() - self._global_start_time
        }

    def export_json(self) -> Dict[str, Any]:
        """
        Export all metrics as JSON-serializable dict.

        Returns:
            Complete metrics export
        """
        return {
            "summary": self.get_summary(),
            "skills": {
                name: metrics.to_dict()
                for name, metrics in self._skill_metrics.items()
            },
            "timestamp": datetime.now().isoformat()
        }

    def reset(self):
        """Reset all metrics."""
        self._skill_metrics.clear()
        self._time_series.clear()
        self._global_start_time = time.time()


# Global metrics tracker instance
_global_tracker = SkillMetricsTracker()


def get_tracker() -> SkillMetricsTracker:
    """Get global metrics tracker."""
    return _global_tracker


def record_execution(
    skill_name: str,
    operation: str,
    status: SkillStatus,
    execution_time_ms: float,
    cache_hit: bool = False
):
    """Record execution in global tracker."""
    _global_tracker.record_execution(
        skill_name,
        operation,
        status,
        execution_time_ms,
        cache_hit
    )


__all__ = [
    "SkillMetrics",
    "TimeSeriesMetrics",
    "SkillMetricsTracker",
    "get_tracker",
    "record_execution"
]
