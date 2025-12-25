"""
Memory and Latency Profilers for Dark Trace Operations

Provides precise measurement of memory usage and operation latency
for production deployment and optimization.

Author: HoloLoom Team
Created: December 2025
"""

import time
import sys
import gc
import tracemalloc
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import deque
import statistics
import threading

import numpy as np

T = TypeVar('T')


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    timestamp: float
    current_bytes: int
    peak_bytes: int
    allocated_blocks: int
    label: str = ""

    @property
    def current_mb(self) -> float:
        return self.current_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    def __str__(self) -> str:
        return (
            f"{self.label}: current={self.current_mb:.2f}MB, "
            f"peak={self.peak_mb:.2f}MB, blocks={self.allocated_blocks}"
        )


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    operation: str
    duration_ns: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1_000

    def __str__(self) -> str:
        return f"{self.operation}: {self.duration_ms:.3f}ms"


@dataclass
class LatencyStatistics:
    """Statistical summary of latency measurements."""
    operation: str
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

    def __str__(self) -> str:
        return (
            f"{self.operation} (n={self.count}):\n"
            f"  mean={self.mean_ms:.3f}ms ± {self.std_ms:.3f}ms\n"
            f"  min={self.min_ms:.3f}ms, max={self.max_ms:.3f}ms\n"
            f"  p50={self.p50_ms:.3f}ms, p90={self.p90_ms:.3f}ms, "
            f"p95={self.p95_ms:.3f}ms, p99={self.p99_ms:.3f}ms"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


class MemoryProfiler:
    """
    Profile memory usage during Dark Trace operations.

    Uses tracemalloc for precise memory tracking.
    Thread-safe for concurrent profiling.
    """

    def __init__(self, track_top_allocations: int = 10):
        self.track_top_allocations = track_top_allocations
        self.snapshots: List[MemorySnapshot] = []
        self._lock = threading.Lock()
        self._tracing_started = False

    def start(self) -> None:
        """Start memory tracing."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracing_started = True

    def stop(self) -> None:
        """Stop memory tracing."""
        if self._tracing_started and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._tracing_started = False

    def snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        if not tracemalloc.is_tracing():
            self.start()

        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()

        mem_snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_bytes=current,
            peak_bytes=peak,
            allocated_blocks=len(snapshot.traces),
            label=label
        )

        with self._lock:
            self.snapshots.append(mem_snapshot)

        return mem_snapshot

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()

    def get_top_allocations(self, limit: Optional[int] = None) -> List[Tuple[str, int]]:
        """Get top memory allocations by size."""
        if not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        limit = limit or self.track_top_allocations
        result = []

        for stat in top_stats[:limit]:
            result.append((str(stat.traceback), stat.size))

        return result

    def get_memory_delta(self) -> Optional[int]:
        """Get memory change since first snapshot."""
        with self._lock:
            if len(self.snapshots) < 2:
                return None
            return self.snapshots[-1].current_bytes - self.snapshots[0].current_bytes

    @contextmanager
    def track(self, label: str = "operation"):
        """Context manager to track memory for a code block."""
        self.start()
        self.reset_peak()
        gc.collect()  # Clean up before measurement

        before = self.snapshot(f"{label}_start")

        try:
            yield before
        finally:
            gc.collect()  # Clean up after measurement
            after = self.snapshot(f"{label}_end")

            delta = after.current_bytes - before.current_bytes
            peak_during = after.peak_bytes

    def clear(self) -> None:
        """Clear all snapshots."""
        with self._lock:
            self.snapshots.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of memory profiling."""
        with self._lock:
            if not self.snapshots:
                return {"snapshots": 0}

            current_values = [s.current_bytes for s in self.snapshots]
            peak_values = [s.peak_bytes for s in self.snapshots]

            return {
                "snapshots": len(self.snapshots),
                "current_min_mb": min(current_values) / (1024 * 1024),
                "current_max_mb": max(current_values) / (1024 * 1024),
                "peak_max_mb": max(peak_values) / (1024 * 1024),
                "delta_mb": (current_values[-1] - current_values[0]) / (1024 * 1024),
            }


class LatencyProfiler:
    """
    Profile operation latency with statistical analysis.

    Features:
    - Nanosecond precision timing
    - Rolling statistics (mean, std, percentiles)
    - Operation categorization
    - Thread-safe
    """

    def __init__(self, max_samples_per_operation: int = 1000):
        self.max_samples = max_samples_per_operation
        self._measurements: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, operation: str, duration_ns: int, **metadata) -> LatencyMeasurement:
        """Record a latency measurement."""
        measurement = LatencyMeasurement(
            operation=operation,
            duration_ns=duration_ns,
            timestamp=time.time(),
            metadata=metadata
        )

        with self._lock:
            if operation not in self._measurements:
                self._measurements[operation] = deque(maxlen=self.max_samples)
            self._measurements[operation].append(measurement)

        return measurement

    @contextmanager
    def measure(self, operation: str, **metadata):
        """Context manager to measure operation latency."""
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            self.record(operation, end - start, **metadata)

    def time_function(self, operation: Optional[str] = None):
        """Decorator to time a function."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            op_name = operation or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start = time.perf_counter_ns()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end = time.perf_counter_ns()
                    self.record(op_name, end - start)

            return wrapper
        return decorator

    def get_statistics(self, operation: str) -> Optional[LatencyStatistics]:
        """Get statistics for an operation."""
        with self._lock:
            if operation not in self._measurements:
                return None

            measurements = list(self._measurements[operation])

        if not measurements:
            return None

        durations_ms = [m.duration_ms for m in measurements]

        return LatencyStatistics(
            operation=operation,
            count=len(durations_ms),
            mean_ms=statistics.mean(durations_ms),
            std_ms=statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0,
            min_ms=min(durations_ms),
            max_ms=max(durations_ms),
            p50_ms=np.percentile(durations_ms, 50),
            p90_ms=np.percentile(durations_ms, 90),
            p95_ms=np.percentile(durations_ms, 95),
            p99_ms=np.percentile(durations_ms, 99),
        )

    def get_all_statistics(self) -> Dict[str, LatencyStatistics]:
        """Get statistics for all operations."""
        with self._lock:
            operations = list(self._measurements.keys())

        return {op: self.get_statistics(op) for op in operations}

    def get_operations(self) -> List[str]:
        """Get list of recorded operations."""
        with self._lock:
            return list(self._measurements.keys())

    def clear(self, operation: Optional[str] = None) -> None:
        """Clear measurements."""
        with self._lock:
            if operation:
                if operation in self._measurements:
                    self._measurements[operation].clear()
            else:
                self._measurements.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiled operations."""
        all_stats = self.get_all_statistics()

        return {
            "operations": len(all_stats),
            "total_measurements": sum(s.count for s in all_stats.values()),
            "by_operation": {
                op: stats.to_dict() for op, stats in all_stats.items()
            }
        }


# Convenience functions

def profile_memory(label: str = "operation"):
    """Decorator to profile memory usage of a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            profiler = MemoryProfiler()
            with profiler.track(label):
                result = func(*args, **kwargs)

            summary = profiler.get_summary()
            print(f"Memory [{label}]: delta={summary.get('delta_mb', 0):.2f}MB, "
                  f"peak={summary.get('peak_max_mb', 0):.2f}MB")

            return result
        return wrapper
    return decorator


def profile_latency(operation: Optional[str] = None, print_result: bool = True):
    """Decorator to profile latency of a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start = time.perf_counter_ns()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end = time.perf_counter_ns()
                duration_ms = (end - start) / 1_000_000
                if print_result:
                    print(f"Latency [{op_name}]: {duration_ms:.3f}ms")

        return wrapper
    return decorator


class CombinedProfiler:
    """
    Combined memory and latency profiler for comprehensive analysis.
    """

    def __init__(self):
        self.memory = MemoryProfiler()
        self.latency = LatencyProfiler()

    @contextmanager
    def profile(self, operation: str):
        """Profile both memory and latency for an operation."""
        self.memory.start()
        self.memory.reset_peak()
        gc.collect()

        mem_before = self.memory.snapshot(f"{operation}_start")
        start = time.perf_counter_ns()

        try:
            yield
        finally:
            end = time.perf_counter_ns()
            gc.collect()
            mem_after = self.memory.snapshot(f"{operation}_end")

            # Record latency
            self.latency.record(operation, end - start)

            # Calculate memory delta
            delta_bytes = mem_after.current_bytes - mem_before.current_bytes
            peak_bytes = mem_after.peak_bytes

    def get_summary(self) -> Dict[str, Any]:
        """Get combined summary."""
        return {
            "memory": self.memory.get_summary(),
            "latency": self.latency.get_summary(),
        }

    def clear(self) -> None:
        """Clear all profiling data."""
        self.memory.clear()
        self.latency.clear()
