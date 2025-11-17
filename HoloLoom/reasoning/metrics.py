#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Tracking for Reasoning Engine
=======================================
Prometheus-style metrics for monitoring and observability.

Phase 2: Production Hardening

Author: Claude Code
Date: 2025-11-17
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Types
# ============================================================================

@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    help: str
    value: int = 0
    labels: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def inc(self, amount: int = 1, **label_values):
        """Increment counter."""
        if label_values:
            label_key = "_".join(f"{k}={v}" for k, v in sorted(label_values.items()))
            self.labels[label_key] += amount
        else:
            self.value += amount

    def get(self, **label_values) -> int:
        """Get counter value."""
        if label_values:
            label_key = "_".join(f"{k}={v}" for k, v in sorted(label_values.items()))
            return self.labels.get(label_key, 0)
        return self.value


@dataclass
class Histogram:
    """Simple histogram metric."""
    name: str
    help: str
    buckets: List[float]
    observations: List[float] = field(default_factory=list)
    sum: float = 0.0
    count: int = 0

    def observe(self, value: float):
        """Record observation."""
        self.observations.append(value)
        self.sum += value
        self.count += 1

    def get_bucket_counts(self) -> Dict[float, int]:
        """Get histogram bucket counts."""
        counts = {bucket: 0 for bucket in self.buckets}
        counts[float('inf')] = 0

        for obs in self.observations:
            for bucket in self.buckets:
                if obs <= bucket:
                    counts[bucket] += 1
            if obs > self.buckets[-1]:
                counts[float('inf')] += 1

        return counts

    def get_percentile(self, p: float) -> float:
        """Get percentile value (0-100)."""
        if not self.observations:
            return 0.0

        sorted_obs = sorted(self.observations)
        index = int(len(sorted_obs) * (p / 100.0))
        index = min(index, len(sorted_obs) - 1)
        return sorted_obs[index]

    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.observations:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "count": self.count,
            "sum": self.sum,
            "avg": self.sum / self.count,
            "min": min(self.observations),
            "max": max(self.observations),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
        }


# ============================================================================
# Metrics Registry
# ============================================================================

class MetricsRegistry:
    """
    Registry for reasoning metrics.

    Prometheus-style metrics without requiring prometheus_client dependency.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self.logger = logging.getLogger(f"{__name__}.MetricsRegistry")

    def counter(self, name: str, help: str) -> Counter:
        """Get or create counter metric."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name=name, help=help)
            return self._counters[name]

    def histogram(
        self,
        name: str,
        help: str,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Get or create histogram metric."""
        if buckets is None:
            # Default buckets for milliseconds (1ms to 10s)
            buckets = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name=name,
                    help=help,
                    buckets=buckets
                )
            return self._histograms[name]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            metrics = {}

            # Counters
            for name, counter in self._counters.items():
                metrics[name] = {
                    "type": "counter",
                    "help": counter.help,
                    "value": counter.value,
                    "labels": dict(counter.labels)
                }

            # Histograms
            for name, histogram in self._histograms.items():
                metrics[name] = {
                    "type": "histogram",
                    "help": histogram.help,
                    "stats": histogram.get_stats(),
                    "buckets": histogram.get_bucket_counts()
                }

            return metrics

    def reset(self):
        """Reset all metrics (for testing)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Global registry instance
_global_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get global metrics registry."""
    return _global_registry


# ============================================================================
# Reasoning-Specific Metrics
# ============================================================================

class ReasoningMetrics:
    """
    High-level metrics tracking for reasoning operations.

    Tracks:
    - Total operations (by mode)
    - Duration distribution
    - Confidence distribution
    - Error rates
    - Escalation rates (mode upgrades)
    """

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or get_registry()

        # Define metrics
        self.operations_total = self.registry.counter(
            "reasoning_operations_total",
            "Total reasoning operations by mode"
        )

        self.duration_ms = self.registry.histogram(
            "reasoning_duration_ms",
            "Reasoning operation duration in milliseconds",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        )

        self.confidence = self.registry.histogram(
            "reasoning_confidence",
            "Reasoning confidence scores",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        self.chain_length = self.registry.histogram(
            "reasoning_chain_length",
            "Reasoning chain step count",
            buckets=[1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
        )

        self.errors_total = self.registry.counter(
            "reasoning_errors_total",
            "Total reasoning errors by type"
        )

        self.escalations_total = self.registry.counter(
            "reasoning_escalations_total",
            "Total mode escalations (FAST→STANDARD, STANDARD→DEEP)"
        )

        self.verification_failures_total = self.registry.counter(
            "reasoning_verification_failures_total",
            "Total verification failures"
        )

        self.logger = logging.getLogger(f"{__name__}.ReasoningMetrics")

    def track_operation(
        self,
        mode: str,
        duration_ms: float,
        confidence: float,
        chain_length: int,
        success: bool = True,
        escalated: bool = False,
        verification_passed: bool = True
    ):
        """Track a reasoning operation."""
        # Increment operation counter
        self.operations_total.inc(mode=mode)

        # Record duration
        self.duration_ms.observe(duration_ms)

        # Record confidence
        self.confidence.observe(confidence)

        # Record chain length
        self.chain_length.observe(chain_length)

        # Track errors
        if not success:
            self.errors_total.inc(mode=mode)

        # Track escalations
        if escalated:
            self.escalations_total.inc(from_mode=mode)

        # Track verification failures
        if not verification_passed:
            self.verification_failures_total.inc()

        self.logger.debug(
            f"Tracked: mode={mode}, duration={duration_ms:.1f}ms, "
            f"confidence={confidence:.2f}, steps={chain_length}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        all_metrics = self.registry.get_all_metrics()

        return {
            "total_operations": self.operations_total.value,
            "mode_distribution": dict(self.operations_total.labels),
            "duration_stats": self.duration_ms.get_stats(),
            "confidence_stats": self.confidence.get_stats(),
            "chain_length_stats": self.chain_length.get_stats(),
            "total_errors": self.errors_total.value,
            "total_escalations": self.escalations_total.value,
            "total_verification_failures": self.verification_failures_total.value,
            "all_metrics": all_metrics
        }


# ============================================================================
# Context Manager for Automatic Tracking
# ============================================================================

class track_reasoning:
    """
    Context manager for automatic reasoning metrics tracking.

    Usage:
        with track_reasoning(mode="standard") as tracker:
            # Do reasoning
            result = await engine.reason(...)
            tracker.set_result(result)
    """

    def __init__(
        self,
        mode: str,
        metrics: Optional[ReasoningMetrics] = None
    ):
        self.mode = mode
        self.metrics = metrics or ReasoningMetrics()
        self.start_time = None
        self.result = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000

        if self.result:
            self.metrics.track_operation(
                mode=self.mode,
                duration_ms=duration_ms,
                confidence=self.result.total_confidence,
                chain_length=len(self.result.chain),
                success=exc_type is None
            )
        else:
            # Track failed operation
            self.metrics.track_operation(
                mode=self.mode,
                duration_ms=duration_ms,
                confidence=0.0,
                chain_length=0,
                success=False
            )

        # Don't suppress exceptions
        return False

    def set_result(self, result):
        """Set reasoning result for tracking."""
        self.result = result


# ============================================================================
# Convenience Functions
# ============================================================================

def get_reasoning_metrics() -> ReasoningMetrics:
    """Get global reasoning metrics instance."""
    return ReasoningMetrics()


def reset_metrics():
    """Reset all metrics (for testing)."""
    get_registry().reset()
