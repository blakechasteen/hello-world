#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Profiling for Reasoning Engine
============================================
Profiling utilities for identifying bottlenecks.

Phase 2: Production Hardening

Author: Claude Code
Date: 2025-11-17
"""

import cProfile
import pstats
import io
import time
import logging
import functools
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Profiling Results
# ============================================================================

@dataclass
class ProfileResult:
    """Results from a profiling run."""
    function_name: str
    duration_ms: float
    call_count: int = 1
    memory_delta_mb: float = 0.0
    cprofile_stats: Optional[pstats.Stats] = None
    component_times: Dict[str, float] = field(default_factory=dict)

    def get_cprofile_summary(self, top_n: int = 20) -> str:
        """Get cProfile summary as string."""
        if not self.cprofile_stats:
            return "No cProfile stats available"

        stream = io.StringIO()
        stats = self.cprofile_stats
        stats.stream = stream
        stats.sort_stats('cumulative')
        stats.print_stats(top_n)
        return stream.getvalue()

    def print_summary(self):
        """Print profiling summary."""
        print(f"\n{'='*70}")
        print(f"Profile: {self.function_name}")
        print(f"{'='*70}")
        print(f"Duration: {self.duration_ms:.2f}ms")
        print(f"Calls: {self.call_count}")

        if self.memory_delta_mb != 0.0:
            print(f"Memory delta: {self.memory_delta_mb:+.2f}MB")

        if self.component_times:
            print(f"\nComponent Times:")
            for component, duration in sorted(
                self.component_times.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percent = (duration / self.duration_ms) * 100 if self.duration_ms > 0 else 0
                print(f"  {component:30s}: {duration:8.2f}ms ({percent:5.1f}%)")

        if self.cprofile_stats:
            print(f"\nTop Functions (by cumulative time):")
            print(self.get_cprofile_summary(top_n=10))


# ============================================================================
# Time Profiler (Simple)
# ============================================================================

def time_profile(func: Callable) -> Callable:
    """
    Decorator for simple time profiling.

    Usage:
        @time_profile
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration_ms = (time.time() - start) * 1000
            logger.info(f"[TIME] {func.__name__}: {duration_ms:.2f}ms")

    return wrapper


def time_profile_async(func: Callable) -> Callable:
    """
    Decorator for simple async time profiling.

    Usage:
        @time_profile_async
        async def my_async_function():
            pass
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration_ms = (time.time() - start) * 1000
            logger.info(f"[TIME] {func.__name__}: {duration_ms:.2f}ms")

    return wrapper


# ============================================================================
# cProfile Profiler
# ============================================================================

def cprofile_profile(func: Callable) -> Callable:
    """
    Decorator for cProfile profiling.

    Usage:
        @cprofile_profile
        def my_function():
            pass

    Results are logged and saved to ./profiles/ directory.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()

            # Create stats
            stats = pstats.Stats(profiler)

            # Save to file
            profiles_dir = Path("./profiles")
            profiles_dir.mkdir(exist_ok=True)
            profile_path = profiles_dir / f"{func.__name__}_{int(time.time())}.prof"

            stats.dump_stats(str(profile_path))
            logger.info(f"[CPROFILE] Saved to {profile_path}")

            # Log summary
            stream = io.StringIO()
            stats.stream = stream
            stats.sort_stats('cumulative')
            stats.print_stats(10)
            logger.debug(f"[CPROFILE] {func.__name__}:\n{stream.getvalue()}")

    return wrapper


# ============================================================================
# Memory Profiler
# ============================================================================

try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Memory profiling disabled.")


def memory_profile(func: Callable) -> Callable:
    """
    Decorator for memory profiling.

    Usage:
        @memory_profile
        def my_function():
            pass
    """
    if not PSUTIL_AVAILABLE:
        # No-op if psutil not available
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            delta = mem_after - mem_before

            logger.info(
                f"[MEMORY] {func.__name__}: "
                f"{mem_before:.1f}MB → {mem_after:.1f}MB (Δ{delta:+.1f}MB)"
            )

    return wrapper


# ============================================================================
# Full Profiler (Time + Memory + cProfile)
# ============================================================================

def full_profile(func: Callable) -> Callable:
    """
    Decorator for comprehensive profiling (time + memory + cProfile).

    Usage:
        @full_profile
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Memory tracking
        process = None
        mem_before = 0.0
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024

        # Time + cProfile tracking
        start = time.time()
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            duration_ms = (time.time() - start) * 1000

            # Memory after
            mem_after = 0.0
            delta = 0.0
            if process:
                mem_after = process.memory_info().rss / 1024 / 1024
                delta = mem_after - mem_before

            # Create profile result
            stats = pstats.Stats(profiler)
            profile_result = ProfileResult(
                function_name=func.__name__,
                duration_ms=duration_ms,
                memory_delta_mb=delta,
                cprofile_stats=stats
            )

            # Save cProfile stats
            profiles_dir = Path("./profiles")
            profiles_dir.mkdir(exist_ok=True)
            profile_path = profiles_dir / f"{func.__name__}_{int(time.time())}.prof"
            stats.dump_stats(str(profile_path))

            # Log summary
            logger.info(
                f"[FULL PROFILE] {func.__name__}: "
                f"{duration_ms:.2f}ms, Δ{delta:+.1f}MB, saved to {profile_path}"
            )

    return wrapper


# ============================================================================
# Component Timer (for manual timing)
# ============================================================================

class ComponentTimer:
    """
    Manual timer for profiling components within a function.

    Usage:
        timer = ComponentTimer()

        with timer.time("component_a"):
            # Do work
            pass

        with timer.time("component_b"):
            # Do more work
            pass

        timer.print_summary()
    """

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.times: Dict[str, float] = {}
        self.current_component: Optional[str] = None
        self.current_start: float = 0.0

    def time(self, component_name: str):
        """Context manager for timing a component."""
        class TimerContext:
            def __init__(self, timer, name):
                self.timer = timer
                self.name = name

            def __enter__(self):
                self.timer.current_component = self.name
                self.timer.current_start = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration_ms = (time.time() - self.timer.current_start) * 1000
                if self.name in self.timer.times:
                    self.timer.times[self.name] += duration_ms
                else:
                    self.timer.times[self.name] = duration_ms
                self.timer.current_component = None

        return TimerContext(self, component_name)

    def print_summary(self):
        """Print timing summary."""
        total = sum(self.times.values())

        print(f"\n{'='*60}")
        print(f"Component Timings: {self.name}")
        print(f"{'='*60}")

        for component, duration in sorted(
            self.times.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percent = (duration / total) * 100 if total > 0 else 0
            print(f"{component:40s}: {duration:8.2f}ms ({percent:5.1f}%)")

        print(f"{'='*60}")
        print(f"{'Total':40s}: {total:8.2f}ms (100.0%)")

    def get_summary(self) -> Dict[str, float]:
        """Get timing summary as dictionary."""
        return dict(self.times)


# ============================================================================
# Async Profiling Support
# ============================================================================

def full_profile_async(func: Callable) -> Callable:
    """
    Decorator for comprehensive async profiling.

    Usage:
        @full_profile_async
        async def my_async_function():
            pass
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Memory tracking
        process = None
        mem_before = 0.0
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024

        # Time tracking
        start = time.time()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration_ms = (time.time() - start) * 1000

            # Memory after
            mem_after = 0.0
            delta = 0.0
            if process:
                mem_after = process.memory_info().rss / 1024 / 1024
                delta = mem_after - mem_before

            # Log summary
            logger.info(
                f"[ASYNC PROFILE] {func.__name__}: "
                f"{duration_ms:.2f}ms, Δ{delta:+.1f}MB"
            )

    return wrapper
