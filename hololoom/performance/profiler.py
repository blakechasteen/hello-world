#!/usr/bin/env python3
"""
Performance Profiler - Detailed timing and resource tracking
============================================================
Tracks execution time, memory usage, and component-level metrics
for the HoloLoom weaving cycle.

Usage:
    async with Profiler("query_processing") as prof:
        result = await process_query()
        prof.record_metric("tokens_processed", 1024)

    print(prof.summary())
"""

import time
import asyncio
import logging
import functools
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict

# Optional: psutil for memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not available - memory tracking disabled")

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """Single profiling measurement."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: int = 0
    memory_end: Optional[int] = None
    memory_delta: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def finish(self):
        """Complete the profile entry."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                self.memory_end = process.memory_info().rss
                self.memory_delta = self.memory_end - self.memory_start
            except Exception:
                pass


class Profiler:
    """
    Performance profiler with nested context support.

    Features:
    - Hierarchical timing (nested contexts)
    - Memory tracking
    - Custom metrics
    - Aggregation across multiple runs
    """

    def __init__(self, name: str, parent: Optional['Profiler'] = None):
        self.name = name
        self.parent = parent
        self.entry: Optional[ProfileEntry] = None
        self.children: List[ProfileEntry] = []
        self.custom_metrics: Dict[str, Any] = {}

    async def __aenter__(self):
        """Start profiling (async context)."""
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling (async context)."""
        self.stop()
        return False

    def __enter__(self):
        """Start profiling (sync context)."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling (sync context)."""
        self.stop()
        return False

    def start(self):
        """Begin profiling."""
        self.entry = ProfileEntry(
            name=self.name,
            start_time=time.time()
        )
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                self.entry.memory_start = process.memory_info().rss
            except Exception:
                pass

    def stop(self):
        """End profiling."""
        if self.entry:
            self.entry.finish()
            self.entry.metrics.update(self.custom_metrics)

            # Add to parent's children
            if self.parent and self.parent.entry:
                self.parent.children.append(self.entry)

    def record_metric(self, name: str, value: Any):
        """Record a custom metric."""
        self.custom_metrics[name] = value

    def child(self, name: str) -> 'Profiler':
        """Create a child profiler."""
        return Profiler(name=name, parent=self)

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.entry:
            return {}

        summary = {
            "name": self.name,
            "duration_ms": self.entry.duration * 1000 if self.entry.duration else 0,
            "memory_mb": self.entry.memory_delta / (1024 * 1024) if self.entry.memory_delta else 0,
            "metrics": self.entry.metrics,
        }

        if self.children:
            summary["children"] = [
                {
                    "name": child.name,
                    "duration_ms": child.duration * 1000 if child.duration else 0,
                    "memory_mb": child.memory_delta / (1024 * 1024) if child.memory_delta else 0,
                    "metrics": child.metrics
                }
                for child in self.children
            ]

        return summary

    def log_summary(self, level: int = logging.INFO):
        """Log performance summary."""
        summary = self.summary()
        if not summary:
            return

        logger.log(level, f"Profile: {summary['name']}")
        logger.log(level, f"  Duration: {summary['duration_ms']:.2f}ms")
        logger.log(level, f"  Memory: {summary['memory_mb']:.2f}MB")

        if summary.get("metrics"):
            logger.log(level, f"  Metrics: {summary['metrics']}")

        if summary.get("children"):
            for child in summary["children"]:
                logger.log(level, f"  └─ {child['name']}: {child['duration_ms']:.2f}ms")


class ProfilerRegistry:
    """
    Global registry for aggregating profiling data across runs.

    Usage:
        registry = ProfilerRegistry()

        # Record multiple runs
        for query in queries:
            async with Profiler("process") as prof:
                await process(query)
            registry.record(prof)

        # Get aggregated stats
        print(registry.aggregate_stats())
    """

    def __init__(self):
        self.profiles: List[ProfileEntry] = []
        self.metrics_by_name: Dict[str, List[float]] = defaultdict(list)

    def record(self, profiler: Profiler):
        """Record a completed profiler run."""
        if profiler.entry:
            self.profiles.append(profiler.entry)

            # Aggregate metrics by name
            if profiler.entry.duration:
                self.metrics_by_name[f"{profiler.name}_duration_ms"].append(
                    profiler.entry.duration * 1000
                )
            if profiler.entry.memory_delta:
                self.metrics_by_name[f"{profiler.name}_memory_mb"].append(
                    profiler.entry.memory_delta / (1024 * 1024)
                )

            # Record children
            for child in profiler.children:
                if child.duration:
                    self.metrics_by_name[f"{child.name}_duration_ms"].append(
                        child.duration * 1000
                    )

    def aggregate_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregate statistics (mean, p50, p95, p99)."""
        import statistics

        stats = {}
        for name, values in self.metrics_by_name.items():
            if not values:
                continue

            sorted_values = sorted(values)
            n = len(sorted_values)

            stats[name] = {
                "count": n,
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": sorted_values[int(0.95 * n)] if n > 0 else 0,
                "p99": sorted_values[int(0.99 * n)] if n > 0 else 0,
                "min": min(values),
                "max": max(values)
            }

        return stats

    def clear(self):
        """Clear all recorded profiles."""
        self.profiles.clear()
        self.metrics_by_name.clear()


def profile_async(name: Optional[str] = None):
    """
    Decorator for profiling async functions.

    Usage:
        @profile_async("my_function")
        async def my_function():
            ...
    """
    def decorator(func: Callable):
        func_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with Profiler(func_name) as prof:
                result = await func(*args, **kwargs)
            prof.log_summary()
            return result

        return wrapper
    return decorator


def profile_sync(name: Optional[str] = None):
    """
    Decorator for profiling sync functions.

    Usage:
        @profile_sync("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable):
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Profiler(func_name) as prof:
                result = func(*args, **kwargs)
            prof.log_summary()
            return result

        return wrapper
    return decorator


# Global registry instance
global_registry = ProfilerRegistry()


def get_global_registry() -> ProfilerRegistry:
    """Get the global profiler registry."""
    return global_registry
