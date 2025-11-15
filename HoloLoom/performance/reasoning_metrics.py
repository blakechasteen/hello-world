#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Engine Metrics - Prometheus-Style Monitoring
=======================================================

Track reasoning engine performance and behavior for monitoring and optimization.

Author: Claude Code
Date: 2025-11-15
Phase: 4 (Visualization)
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import threading


# ============================================================================
# Metric Types
# ============================================================================

class MetricType(Enum):
    """Prometheus-style metric types."""
    COUNTER = "counter"      # Monotonically increasing
    GAUGE = "gauge"          # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"      # Like histogram but with quantiles


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class HistogramBucket:
    """Single histogram bucket."""
    upper_bound: float
    count: int = 0


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Reasoning Metrics Registry
# ============================================================================

class ReasoningMetrics:
    """
    Prometheus-style metrics for reasoning engine.

    Tracks:
    - Mode distribution (FAST/STANDARD/DEEP)
    - Confidence scores (avg, min, max, distribution)
    - Duration (latency distribution)
    - Escalations (FAST → STANDARD → DEEP)
    - Verification failures
    - Step counts
    - Backtracking frequency
    """

    def __init__(self):
        """Initialize metrics registry."""
        self._lock = threading.Lock()

        # Counters
        self._reasoning_total = defaultdict(int)  # By mode
        self._escalations_total = defaultdict(int)  # By from_mode → to_mode
        self._verification_failures_total = 0
        self._critical_steps_total = 0

        # Gauges
        self._active_reasoning = 0

        # Histograms
        self._duration_buckets = self._create_duration_buckets()
        self._confidence_buckets = self._create_confidence_buckets()
        self._step_count_buckets = self._create_step_count_buckets()

        # Time series (last N values)
        self._recent_confidences = []
        self._recent_durations = []
        self._max_recent = 1000

    def _create_duration_buckets(self) -> List[HistogramBucket]:
        """Create buckets for duration histogram (ms)."""
        bounds = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, float('inf')]
        return [HistogramBucket(upper_bound=b) for b in bounds]

    def _create_confidence_buckets(self) -> List[HistogramBucket]:
        """Create buckets for confidence histogram."""
        bounds = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0, float('inf')]
        return [HistogramBucket(upper_bound=b) for b in bounds]

    def _create_step_count_buckets(self) -> List[HistogramBucket]:
        """Create buckets for step count histogram."""
        bounds = [1, 3, 5, 7, 10, 15, float('inf')]
        return [HistogramBucket(upper_bound=b) for b in bounds]

    # ========================================================================
    # Recording Methods
    # ========================================================================

    def record_reasoning(
        self,
        mode: str,
        duration_ms: float,
        confidence: float,
        step_count: int,
        verification_passed: bool,
        critical_steps: int = 0,
        escalated_from: Optional[str] = None
    ):
        """
        Record a reasoning operation.

        Args:
            mode: Reasoning mode (fast/standard/deep)
            duration_ms: Duration in milliseconds
            confidence: Overall confidence [0.0, 1.0]
            step_count: Number of reasoning steps
            verification_passed: Whether verification passed
            critical_steps: Number of critical steps (confidence < 0.5)
            escalated_from: If escalated, the original mode
        """
        with self._lock:
            # Counters
            self._reasoning_total[mode.lower()] += 1

            if escalated_from:
                key = f"{escalated_from.lower()}_to_{mode.lower()}"
                self._escalations_total[key] += 1

            if not verification_passed:
                self._verification_failures_total += 1

            self._critical_steps_total += critical_steps

            # Histograms
            self._record_histogram(self._duration_buckets, duration_ms)
            self._record_histogram(self._confidence_buckets, confidence)
            self._record_histogram(self._step_count_buckets, step_count)

            # Time series
            self._recent_confidences.append(confidence)
            self._recent_durations.append(duration_ms)

            # Trim if needed
            if len(self._recent_confidences) > self._max_recent:
                self._recent_confidences = self._recent_confidences[-self._max_recent:]
                self._recent_durations = self._recent_durations[-self._max_recent:]

    def _record_histogram(self, buckets: List[HistogramBucket], value: float):
        """Record value in histogram buckets."""
        for bucket in buckets:
            if value <= bucket.upper_bound:
                bucket.count += 1

    def start_reasoning(self):
        """Increment active reasoning gauge."""
        with self._lock:
            self._active_reasoning += 1

    def end_reasoning(self):
        """Decrement active reasoning gauge."""
        with self._lock:
            self._active_reasoning -= 1

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_mode_distribution(self) -> Dict[str, int]:
        """Get count of reasoning operations by mode."""
        with self._lock:
            return dict(self._reasoning_total)

    def get_escalation_stats(self) -> Dict[str, int]:
        """Get escalation counts."""
        with self._lock:
            return dict(self._escalations_total)

    def get_verification_failure_rate(self) -> float:
        """Get verification failure rate."""
        with self._lock:
            total = sum(self._reasoning_total.values())
            if total == 0:
                return 0.0
            return self._verification_failures_total / total

    def get_duration_percentiles(self) -> Dict[str, float]:
        """Get duration percentiles (p50, p95, p99)."""
        with self._lock:
            if not self._recent_durations:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}

            sorted_durations = sorted(self._recent_durations)
            n = len(sorted_durations)

            return {
                "p50": sorted_durations[int(n * 0.50)],
                "p95": sorted_durations[int(n * 0.95)],
                "p99": sorted_durations[int(n * 0.99)] if n > 100 else sorted_durations[-1],
                "avg": sum(sorted_durations) / n
            }

    def get_confidence_percentiles(self) -> Dict[str, float]:
        """Get confidence percentiles."""
        with self._lock:
            if not self._recent_confidences:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}

            sorted_conf = sorted(self._recent_confidences)
            n = len(sorted_conf)

            return {
                "p50": sorted_conf[int(n * 0.50)],
                "p95": sorted_conf[int(n * 0.95)],
                "p99": sorted_conf[int(n * 0.99)] if n > 100 else sorted_conf[-1],
                "avg": sum(sorted_conf) / n
            }

    def get_duration_histogram(self) -> List[Dict[str, Any]]:
        """Get duration histogram data."""
        with self._lock:
            return [
                {
                    "upper_bound": bucket.upper_bound,
                    "count": bucket.count,
                    "cumulative": sum(b.count for b in self._duration_buckets[:i+1])
                }
                for i, bucket in enumerate(self._duration_buckets)
            ]

    def get_confidence_histogram(self) -> List[Dict[str, Any]]:
        """Get confidence histogram data."""
        with self._lock:
            return [
                {
                    "upper_bound": bucket.upper_bound,
                    "count": bucket.count,
                    "cumulative": sum(b.count for b in self._confidence_buckets[:i+1])
                }
                for i, bucket in enumerate(self._confidence_buckets)
            ]

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics."""
        with self._lock:
            total = sum(self._reasoning_total.values())

            return {
                "total_operations": total,
                "active_reasoning": self._active_reasoning,
                "mode_distribution": dict(self._reasoning_total),
                "mode_percentages": {
                    mode: (count / total * 100 if total > 0 else 0)
                    for mode, count in self._reasoning_total.items()
                },
                "escalations": dict(self._escalations_total),
                "verification_failures": self._verification_failures_total,
                "verification_failure_rate": self.get_verification_failure_rate(),
                "critical_steps_total": self._critical_steps_total,
                "duration_stats": self.get_duration_percentiles(),
                "confidence_stats": self.get_confidence_percentiles(),
            }

    def get_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        lines = []

        # Mode distribution counter
        lines.append("# HELP reasoning_operations_total Total reasoning operations by mode")
        lines.append("# TYPE reasoning_operations_total counter")
        for mode, count in self.get_mode_distribution().items():
            lines.append(f'reasoning_operations_total{{mode="{mode}"}} {count}')

        # Escalations counter
        lines.append("")
        lines.append("# HELP reasoning_escalations_total Reasoning mode escalations")
        lines.append("# TYPE reasoning_escalations_total counter")
        for key, count in self.get_escalation_stats().items():
            lines.append(f'reasoning_escalations_total{{transition="{key}"}} {count}')

        # Verification failures
        lines.append("")
        lines.append("# HELP reasoning_verification_failures_total Verification failures")
        lines.append("# TYPE reasoning_verification_failures_total counter")
        lines.append(f"reasoning_verification_failures_total {self._verification_failures_total}")

        # Active reasoning gauge
        lines.append("")
        lines.append("# HELP reasoning_active Active reasoning operations")
        lines.append("# TYPE reasoning_active gauge")
        lines.append(f"reasoning_active {self._active_reasoning}")

        # Duration histogram
        lines.append("")
        lines.append("# HELP reasoning_duration_ms Reasoning duration in milliseconds")
        lines.append("# TYPE reasoning_duration_ms histogram")
        for bucket_data in self.get_duration_histogram():
            if bucket_data['upper_bound'] == float('inf'):
                le = "+Inf"
            else:
                le = str(bucket_data['upper_bound'])
            lines.append(f'reasoning_duration_ms_bucket{{le="{le}"}} {bucket_data["cumulative"]}')

        # Confidence histogram
        lines.append("")
        lines.append("# HELP reasoning_confidence Reasoning confidence score")
        lines.append("# TYPE reasoning_confidence histogram")
        for bucket_data in self.get_confidence_histogram():
            if bucket_data['upper_bound'] == float('inf'):
                le = "+Inf"
            else:
                le = str(bucket_data['upper_bound'])
            lines.append(f'reasoning_confidence_bucket{{le="{le}"}} {bucket_data["cumulative"]}')

        return "\n".join(lines)

    def reset(self):
        """Reset all metrics (for testing)."""
        with self._lock:
            self._reasoning_total.clear()
            self._escalations_total.clear()
            self._verification_failures_total = 0
            self._critical_steps_total = 0
            self._active_reasoning = 0

            # Reset histograms
            self._duration_buckets = self._create_duration_buckets()
            self._confidence_buckets = self._create_confidence_buckets()
            self._step_count_buckets = self._create_step_count_buckets()

            # Clear time series
            self._recent_confidences.clear()
            self._recent_durations.clear()


# ============================================================================
# Global Registry
# ============================================================================

# Global metrics instance
_global_metrics: Optional[ReasoningMetrics] = None


def get_reasoning_metrics() -> ReasoningMetrics:
    """
    Get global reasoning metrics instance.

    Returns:
        Global ReasoningMetrics instance
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = ReasoningMetrics()
    return _global_metrics


def reset_reasoning_metrics():
    """Reset global metrics (for testing)."""
    global _global_metrics
    if _global_metrics is not None:
        _global_metrics.reset()


# ============================================================================
# Context Manager for Automatic Tracking
# ============================================================================

class track_reasoning:
    """
    Context manager for automatic reasoning metrics tracking.

    Usage:
        >>> with track_reasoning(mode="standard") as tracker:
        ...     result = await engine.reason(query, features, context)
        ...     tracker.set_result(result)
    """

    def __init__(
        self,
        mode: str,
        metrics: Optional[ReasoningMetrics] = None,
        escalated_from: Optional[str] = None
    ):
        """
        Initialize tracker.

        Args:
            mode: Reasoning mode being tracked
            metrics: Metrics instance (uses global if None)
            escalated_from: If escalated, the original mode
        """
        self.mode = mode
        self.metrics = metrics or get_reasoning_metrics()
        self.escalated_from = escalated_from
        self.start_time = None
        self.result = None

    def __enter__(self):
        """Enter context - start tracking."""
        self.start_time = time.time()
        self.metrics.start_reasoning()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - record metrics."""
        duration_ms = (time.time() - self.start_time) * 1000
        self.metrics.end_reasoning()

        if exc_type is None and self.result is not None:
            # Extract metrics from result
            self.metrics.record_reasoning(
                mode=self.mode,
                duration_ms=duration_ms,
                confidence=self.result.total_confidence,
                step_count=len(self.result.chain),
                verification_passed=self.result.metadata.get('verification_passed', True),
                critical_steps=sum(
                    1 for step in self.result.chain
                    if step.confidence < 0.5
                ),
                escalated_from=self.escalated_from
            )

        return False  # Don't suppress exceptions

    def set_result(self, result: Any):
        """Set reasoning result for metrics extraction."""
        self.result = result
