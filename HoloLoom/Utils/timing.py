"""
Timing Context Manager - Eliminates timing code duplication.

This module provides a reusable context manager for timing operations,
reducing ~50 LOC of duplicated timing code in weaving_orchestrator.py.

Design Pattern:
- Context manager for automatic timing
- Automatic result storage in trace object
- Support for nested timings
- Optional callbacks for real-time monitoring

Usage:
    from HoloLoom.Utils.timing import TimingContext

    # Simple timing
    async with TimingContext("operation_name") as timer:
        await do_something()
    print(f"Duration: {timer.duration_ms}ms")

    # With trace storage
    trace = WeavingTrace()
    async with TimingContext("stage_1", trace=trace):
        await process_stage_1()
    print(trace.stage_durations)  # {"stage_1": 123.45}
"""

import time
from typing import Optional, Callable, Dict, Any
from contextlib import asynccontextmanager, contextmanager
import logging


class TimingContext:
    """
    Context manager for timing operations.

    Automatically captures start/end times and calculates duration.
    Optionally stores results in a trace object.

    Example:
        async with TimingContext("my_operation") as timer:
            await expensive_operation()

        print(f"Took {timer.duration_ms:.2f}ms")
    """

    def __init__(
        self,
        name: str,
        trace: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize timing context.

        Args:
            name: Operation name
            trace: Optional trace object with stage_durations dict
            logger: Optional logger for debug output
            callback: Optional callback(name, duration_ms)
        """
        self.name = name
        self.trace = trace
        self.logger = logger
        self.callback = callback

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0

    def __enter__(self):
        """Start timing (sync context manager)."""
        self.start_time = time.time()
        if self.logger:
            self.logger.debug(f"Starting: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing (sync context manager)."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

        # Store in trace
        if self.trace and hasattr(self.trace, 'stage_durations'):
            if not isinstance(self.trace.stage_durations, dict):
                self.trace.stage_durations = {}
            self.trace.stage_durations[self.name] = self.duration_ms

        # Log
        if self.logger:
            self.logger.debug(f"Completed: {self.name} ({self.duration_ms:.2f}ms)")

        # Callback
        if self.callback:
            self.callback(self.name, self.duration_ms)

        return False  # Don't suppress exceptions

    async def __aenter__(self):
        """Start timing (async context manager)."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End timing (async context manager)."""
        return self.__exit__(exc_type, exc_val, exc_tb)


class TimingStats:
    """
    Accumulator for timing statistics.

    Tracks min/max/avg/total for named operations.

    Example:
        stats = TimingStats()

        async with TimingContext("op1", callback=stats.record):
            await do_work()

        async with TimingContext("op1", callback=stats.record):
            await do_work()

        print(stats.summary())
        # op1: count=2, min=100ms, max=150ms, avg=125ms, total=250ms
    """

    def __init__(self):
        """Initialize timing stats."""
        self._stats: Dict[str, Dict[str, Any]] = {}

    def record(self, name: str, duration_ms: float):
        """
        Record a timing measurement.

        Args:
            name: Operation name
            duration_ms: Duration in milliseconds
        """
        if name not in self._stats:
            self._stats[name] = {
                "count": 0,
                "total_ms": 0.0,
                "min_ms": float('inf'),
                "max_ms": 0.0
            }

        stats = self._stats[name]
        stats["count"] += 1
        stats["total_ms"] += duration_ms
        stats["min_ms"] = min(stats["min_ms"], duration_ms)
        stats["max_ms"] = max(stats["max_ms"], duration_ms)

    def get_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get stats for a named operation.

        Args:
            name: Operation name

        Returns:
            Dict with count, min, max, avg, total or None
        """
        if name not in self._stats:
            return None

        stats = self._stats[name]
        return {
            "count": stats["count"],
            "min_ms": stats["min_ms"],
            "max_ms": stats["max_ms"],
            "avg_ms": stats["total_ms"] / stats["count"],
            "total_ms": stats["total_ms"]
        }

    def summary(self) -> str:
        """
        Generate human-readable summary.

        Returns:
            Formatted summary string
        """
        lines = ["Timing Statistics:"]
        for name in sorted(self._stats.keys()):
            stats = self.get_stats(name)
            lines.append(
                f"  {name}: count={stats['count']}, "
                f"min={stats['min_ms']:.2f}ms, "
                f"max={stats['max_ms']:.2f}ms, "
                f"avg={stats['avg_ms']:.2f}ms, "
                f"total={stats['total_ms']:.2f}ms"
            )
        return "\n".join(lines)

    def reset(self):
        """Reset all statistics."""
        self._stats.clear()


# Convenience functions
@contextmanager
def time_operation(
    name: str,
    trace: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Sync context manager for timing operations.

    Args:
        name: Operation name
        trace: Optional trace object
        logger: Optional logger

    Example:
        with time_operation("my_op", trace=trace):
            do_work()
    """
    ctx = TimingContext(name, trace=trace, logger=logger)
    try:
        yield ctx.__enter__()
    finally:
        ctx.__exit__(None, None, None)


@asynccontextmanager
async def time_operation_async(
    name: str,
    trace: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Async context manager for timing operations.

    Args:
        name: Operation name
        trace: Optional trace object
        logger: Optional logger

    Example:
        async with time_operation_async("my_op", trace=trace):
            await do_async_work()
    """
    ctx = TimingContext(name, trace=trace, logger=logger)
    try:
        yield await ctx.__aenter__()
    finally:
        await ctx.__aexit__(None, None, None)


# Example usage
if __name__ == "__main__":
    import asyncio

    async def example():
        """Example usage of timing utilities."""
        # Simple timing
        async with TimingContext("operation_1") as timer:
            await asyncio.sleep(0.1)
        print(f"Operation 1 took {timer.duration_ms:.2f}ms")

        # With stats tracking
        stats = TimingStats()
        for i in range(3):
            async with TimingContext(f"batch_{i}", callback=stats.record):
                await asyncio.sleep(0.05 * (i + 1))

        print(stats.summary())

        # With trace object
        class MockTrace:
            def __init__(self):
                self.stage_durations = {}

        trace = MockTrace()
        async with TimingContext("stage_1", trace=trace):
            await asyncio.sleep(0.1)
        async with TimingContext("stage_2", trace=trace):
            await asyncio.sleep(0.05)

        print(f"Trace durations: {trace.stage_durations}")

    asyncio.run(example())
