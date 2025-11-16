#!/usr/bin/env python3
"""
Performance Analysis from Distributed Traces
=============================================

Analyzes traces to identify bottlenecks and performance patterns.

Features:
- Bottleneck detection (operations >50% of parent)
- P50/P95/P99 latency calculation
- Operation frequency analysis
- Critical path identification
- Slowest operations report
- Time-series analysis

Usage:
    from HoloLoom.tracing import PerformanceAnalyzer, analyze_trace

    # Analyze single trace
    bottlenecks = detect_bottlenecks(trace)

    # Analyze multiple traces
    analyzer = PerformanceAnalyzer()
    analyzer.add_traces(traces)
    report = analyzer.generate_report()

Created: 2025-11-16
"""

import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Span:
    """Simplified span representation for analysis."""

    span_id: str
    parent_id: Optional[str]
    operation_name: str
    start_time: float  # Unix timestamp
    end_time: float
    duration: float  # milliseconds
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    status: str = "OK"  # OK, ERROR
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000 if self.duration < 100 else self.duration


@dataclass
class Trace:
    """Trace containing multiple spans."""

    trace_id: str
    spans: List[Span]
    start_time: float
    end_time: float
    duration: float
    root_span: Optional[Span] = None

    def __post_init__(self):
        """Find root span."""
        for span in self.spans:
            if span.parent_id is None:
                self.root_span = span
                break

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_children(self, span_id: str) -> List[Span]:
        """Get child spans of a span."""
        return [s for s in self.spans if s.parent_id == span_id]


@dataclass
class Bottleneck:
    """Identified bottleneck in trace."""

    operation: str
    span_id: str
    duration_ms: float
    percentage_of_parent: float
    parent_operation: str
    recommendation: str


@dataclass
class OperationStats:
    """Statistics for an operation."""

    operation_name: str
    count: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    error_count: int
    error_rate: float


# ============================================================================
# Bottleneck Detection
# ============================================================================

def detect_bottlenecks(
    trace: Trace,
    threshold: float = 0.5,
) -> List[Bottleneck]:
    """
    Detect bottlenecks in a trace.

    A bottleneck is defined as a span that takes more than `threshold`
    of its parent span's duration.

    Args:
        trace: Trace to analyze
        threshold: Percentage threshold (0.0-1.0)

    Returns:
        List of identified bottlenecks

    Example:
        >>> bottlenecks = detect_bottlenecks(trace, threshold=0.5)
        >>> for bn in bottlenecks:
        ...     print(f"{bn.operation}: {bn.duration_ms:.1f}ms ({bn.percentage_of_parent:.0%})")
    """
    bottlenecks = []

    for span in trace.spans:
        if span.parent_id is None:
            continue  # Skip root span

        parent = trace.get_span(span.parent_id)
        if parent is None:
            continue

        # Calculate percentage
        if parent.duration_ms > 0:
            percentage = span.duration_ms / parent.duration_ms

            if percentage > threshold:
                # Generate recommendation
                if percentage > 0.8:
                    recommendation = f"Critical: {span.operation_name} dominates {parent.operation_name}"
                elif percentage > 0.6:
                    recommendation = f"High impact: Consider optimizing {span.operation_name}"
                else:
                    recommendation = f"Moderate impact: {span.operation_name} could be optimized"

                bottlenecks.append(
                    Bottleneck(
                        operation=span.operation_name,
                        span_id=span.span_id,
                        duration_ms=span.duration_ms,
                        percentage_of_parent=percentage,
                        parent_operation=parent.operation_name,
                        recommendation=recommendation,
                    )
                )

    # Sort by impact (duration * percentage)
    bottlenecks.sort(
        key=lambda b: b.duration_ms * b.percentage_of_parent,
        reverse=True,
    )

    return bottlenecks


def find_critical_path(trace: Trace) -> List[Span]:
    """
    Find the critical path (longest chain) in a trace.

    Args:
        trace: Trace to analyze

    Returns:
        List of spans forming the critical path

    Example:
        >>> path = find_critical_path(trace)
        >>> total_ms = sum(s.duration_ms for s in path)
        >>> print(f"Critical path: {total_ms:.1f}ms")
    """
    if not trace.root_span:
        return []

    def get_path_duration(span: Span) -> Tuple[float, List[Span]]:
        """Recursively find longest path from this span."""
        children = trace.get_children(span.span_id)

        if not children:
            # Leaf node
            return span.duration_ms, [span]

        # Find longest child path
        max_duration = 0
        max_path = []

        for child in children:
            child_duration, child_path = get_path_duration(child)
            if child_duration > max_duration:
                max_duration = child_duration
                max_path = child_path

        return span.duration_ms + max_duration, [span] + max_path

    _, path = get_path_duration(trace.root_span)
    return path


# ============================================================================
# Performance Analyzer
# ============================================================================

class PerformanceAnalyzer:
    """
    Analyzes multiple traces to identify performance patterns.

    Provides:
    - Operation statistics (avg, p50, p95, p99)
    - Bottleneck detection across traces
    - Error rate tracking
    - Frequency analysis
    - Time-series trends
    """

    def __init__(self):
        """Initialize analyzer."""
        self.traces: List[Trace] = []
        self.operation_durations: Dict[str, List[float]] = defaultdict(list)
        self.operation_errors: Dict[str, int] = defaultdict(int)
        self.operation_counts: Dict[str, int] = defaultdict(int)

    def add_trace(self, trace: Trace):
        """Add a trace to analyze."""
        self.traces.append(trace)

        # Collect operation stats
        for span in trace.spans:
            self.operation_durations[span.operation_name].append(span.duration_ms)
            self.operation_counts[span.operation_name] += 1

            if span.status == "ERROR":
                self.operation_errors[span.operation_name] += 1

    def add_traces(self, traces: List[Trace]):
        """Add multiple traces."""
        for trace in traces:
            self.add_trace(trace)

    def get_operation_stats(self, operation_name: str) -> Optional[OperationStats]:
        """
        Get statistics for a specific operation.

        Args:
            operation_name: Name of operation

        Returns:
            OperationStats or None if operation not found
        """
        if operation_name not in self.operation_durations:
            return None

        durations = self.operation_durations[operation_name]
        count = self.operation_counts[operation_name]
        error_count = self.operation_errors[operation_name]

        return OperationStats(
            operation_name=operation_name,
            count=count,
            total_duration_ms=sum(durations),
            avg_duration_ms=statistics.mean(durations),
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            p50_duration_ms=np.percentile(durations, 50),
            p95_duration_ms=np.percentile(durations, 95),
            p99_duration_ms=np.percentile(durations, 99),
            error_count=error_count,
            error_rate=error_count / count if count > 0 else 0.0,
        )

    def get_all_operation_stats(self) -> List[OperationStats]:
        """Get statistics for all operations."""
        stats = []
        for operation in self.operation_durations.keys():
            op_stats = self.get_operation_stats(operation)
            if op_stats:
                stats.append(op_stats)

        # Sort by total duration
        stats.sort(key=lambda s: s.total_duration_ms, reverse=True)
        return stats

    def get_slowest_operations(self, limit: int = 10) -> List[OperationStats]:
        """
        Get slowest operations by P95 latency.

        Args:
            limit: Number of operations to return

        Returns:
            List of slowest operations
        """
        stats = self.get_all_operation_stats()
        stats.sort(key=lambda s: s.p95_duration_ms, reverse=True)
        return stats[:limit]

    def get_most_frequent_operations(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequently called operations.

        Args:
            limit: Number of operations to return

        Returns:
            List of (operation_name, count) tuples
        """
        return Counter(self.operation_counts).most_common(limit)

    def get_highest_error_rate_operations(self, limit: int = 10) -> List[OperationStats]:
        """
        Get operations with highest error rates.

        Args:
            limit: Number of operations to return

        Returns:
            List of operations with errors
        """
        stats = [s for s in self.get_all_operation_stats() if s.error_count > 0]
        stats.sort(key=lambda s: s.error_rate, reverse=True)
        return stats[:limit]

    def detect_all_bottlenecks(self, threshold: float = 0.5) -> List[Bottleneck]:
        """
        Detect bottlenecks across all traces.

        Args:
            threshold: Bottleneck threshold

        Returns:
            List of all bottlenecks
        """
        all_bottlenecks = []

        for trace in self.traces:
            bottlenecks = detect_bottlenecks(trace, threshold)
            all_bottlenecks.extend(bottlenecks)

        # Sort by impact
        all_bottlenecks.sort(
            key=lambda b: b.duration_ms * b.percentage_of_parent,
            reverse=True,
        )

        return all_bottlenecks

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary with performance insights

        Example:
            >>> analyzer = PerformanceAnalyzer()
            >>> analyzer.add_traces(traces)
            >>> report = analyzer.generate_report()
            >>> print(report["summary"]["total_traces"])
        """
        report = {
            "summary": {
                "total_traces": len(self.traces),
                "total_operations": sum(self.operation_counts.values()),
                "unique_operations": len(self.operation_counts),
                "total_errors": sum(self.operation_errors.values()),
            },
            "slowest_operations": [
                {
                    "operation": s.operation_name,
                    "count": s.count,
                    "p50_ms": round(s.p50_duration_ms, 2),
                    "p95_ms": round(s.p95_duration_ms, 2),
                    "p99_ms": round(s.p99_duration_ms, 2),
                }
                for s in self.get_slowest_operations(10)
            ],
            "most_frequent_operations": [
                {"operation": op, "count": count}
                for op, count in self.get_most_frequent_operations(10)
            ],
            "highest_error_rates": [
                {
                    "operation": s.operation_name,
                    "error_count": s.error_count,
                    "error_rate": round(s.error_rate * 100, 2),
                }
                for s in self.get_highest_error_rate_operations(10)
            ],
            "bottlenecks": [
                {
                    "operation": b.operation,
                    "duration_ms": round(b.duration_ms, 2),
                    "percentage": round(b.percentage_of_parent * 100, 1),
                    "parent": b.parent_operation,
                    "recommendation": b.recommendation,
                }
                for b in self.detect_all_bottlenecks()[:10]
            ],
        }

        return report

    def get_time_series(
        self,
        operation_name: str,
        bucket_size: timedelta = timedelta(minutes=5),
    ) -> Dict[datetime, List[float]]:
        """
        Get time-series data for an operation.

        Args:
            operation_name: Operation to analyze
            bucket_size: Time bucket size

        Returns:
            Dictionary mapping timestamp to list of durations
        """
        time_series = defaultdict(list)

        for trace in self.traces:
            for span in trace.spans:
                if span.operation_name == operation_name:
                    # Bucket by time
                    timestamp = datetime.fromtimestamp(span.start_time)
                    bucket = timestamp.replace(
                        minute=(timestamp.minute // bucket_size.seconds) * bucket_size.seconds,
                        second=0,
                        microsecond=0,
                    )
                    time_series[bucket].append(span.duration_ms)

        return dict(time_series)


# ============================================================================
# Utility Functions
# ============================================================================

def analyze_trace(trace: Trace) -> Dict[str, Any]:
    """
    Perform quick analysis of a single trace.

    Args:
        trace: Trace to analyze

    Returns:
        Dictionary with analysis results

    Example:
        >>> analysis = analyze_trace(trace)
        >>> print(f"Duration: {analysis['total_duration_ms']:.1f}ms")
        >>> print(f"Bottlenecks: {len(analysis['bottlenecks'])}")
    """
    bottlenecks = detect_bottlenecks(trace)
    critical_path = find_critical_path(trace)

    # Operation frequency
    operation_counts = Counter(s.operation_name for s in trace.spans)

    return {
        "trace_id": trace.trace_id,
        "total_duration_ms": trace.duration,
        "span_count": len(trace.spans),
        "bottlenecks": [
            {
                "operation": b.operation,
                "duration_ms": b.duration_ms,
                "percentage": b.percentage_of_parent * 100,
            }
            for b in bottlenecks
        ],
        "critical_path": [s.operation_name for s in critical_path],
        "critical_path_duration_ms": sum(s.duration_ms for s in critical_path),
        "operation_frequency": dict(operation_counts.most_common(10)),
        "error_count": sum(1 for s in trace.spans if s.status == "ERROR"),
    }


def compare_traces(trace1: Trace, trace2: Trace) -> Dict[str, Any]:
    """
    Compare two traces for performance differences.

    Args:
        trace1: First trace
        trace2: Second trace

    Returns:
        Dictionary with comparison results

    Example:
        >>> comparison = compare_traces(baseline_trace, current_trace)
        >>> if comparison["duration_change_percent"] > 10:
        ...     print("Performance regression detected!")
    """
    analysis1 = analyze_trace(trace1)
    analysis2 = analyze_trace(trace2)

    duration_change = analysis2["total_duration_ms"] - analysis1["total_duration_ms"]
    duration_change_percent = (
        (duration_change / analysis1["total_duration_ms"]) * 100
        if analysis1["total_duration_ms"] > 0
        else 0
    )

    return {
        "trace1": analysis1,
        "trace2": analysis2,
        "duration_change_ms": duration_change,
        "duration_change_percent": duration_change_percent,
        "span_count_change": analysis2["span_count"] - analysis1["span_count"],
        "bottleneck_count_change": len(analysis2["bottlenecks"]) - len(analysis1["bottlenecks"]),
        "regression": duration_change_percent > 10,  # >10% slower
    }
