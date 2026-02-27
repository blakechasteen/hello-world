#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Optimizer for ChatOps
==================================
Response time optimization with caching, async, and profiling.

Features:
- Response caching (LRU + TTL)
- Async pipeline optimization
- Query deduplication
- Lazy loading
- Performance profiling
- Resource monitoring

Usage:
    optimizer = PerformanceOptimizer()

    # Wrap slow function
    @optimizer.cache(ttl=3600)
    async def expensive_operation(query):
        return await process_query(query)

    # Profile execution
    with optimizer.profile("query_processing"):
        result = await process_query(query)
"""

import logging
import asyncio
import time
import functools
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import psutil

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Implementation
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    value: Any
    created_at: datetime
    ttl_seconds: int
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class LRUCache:
    """
    LRU cache with TTL support.

    Features:
    - Least Recently Used eviction
    - Time-to-live expiration
    - Hit/miss statistics
    - Size limits
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if miss/expired
        """
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check expiration
        if entry.is_expired:
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.hits += 1
        self.hits += 1

        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        # Remove if exists
        if key in self.cache:
            del self.cache[key]

        # Evict if full
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest

        # Add entry
        entry = CacheEntry(
            value=value,
            created_at=datetime.now(),
            ttl_seconds=ttl if ttl is not None else self.default_ttl
        )
        self.cache[key] = entry

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


# ============================================================================
# Performance Profiler
# ============================================================================

@dataclass
class ProfileEntry:
    """Profile timing entry."""
    name: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Performance profiling tracker.

    Tracks execution times and provides statistics.
    """

    def __init__(self):
        """Initialize profiler."""
        self.entries: List[ProfileEntry] = []
        self.active_timers: Dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing an operation."""
        self.active_timers[name] = time.time()

    def stop(self, name: str, metadata: Optional[Dict] = None) -> float:
        """
        Stop timing an operation.

        Args:
            name: Operation name
            metadata: Optional metadata

        Returns:
            Duration in milliseconds
        """
        if name not in self.active_timers:
            logger.warning(f"Timer not started: {name}")
            return 0.0

        start_time = self.active_timers.pop(name)
        duration_ms = (time.time() - start_time) * 1000

        entry = ProfileEntry(
            name=name,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        self.entries.append(entry)

        return duration_ms

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get profiling statistics.

        Args:
            name: Optional operation name filter

        Returns:
            Statistics dictionary
        """
        if name:
            entries = [e for e in self.entries if e.name == name]
        else:
            entries = self.entries

        if not entries:
            return {}

        durations = [e.duration_ms for e in entries]

        return {
            "count": len(entries),
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "recent_ms": entries[-1].duration_ms if entries else 0
        }

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        self.active_timers.clear()


# ============================================================================
# Performance Optimizer
# ============================================================================

class PerformanceOptimizer:
    """
    Main performance optimization manager.

    Features:
    - Response caching
    - Query deduplication
    - Performance profiling
    - Resource monitoring

    Usage:
        optimizer = PerformanceOptimizer()

        # Cache decorator
        @optimizer.cache(ttl=3600)
        async def slow_function(arg):
            return await expensive_operation(arg)

        # Profile context
        with optimizer.profile("operation"):
            result = await do_work()

        # Get statistics
        stats = optimizer.get_statistics()
    """

    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        enable_profiling: bool = True
    ):
        """
        Initialize performance optimizer.

        Args:
            cache_size: Maximum cache entries
            cache_ttl: Default cache TTL in seconds
            enable_profiling: Enable performance profiling
        """
        self.cache = LRUCache(max_size=cache_size, default_ttl=cache_ttl)
        self.profiler = PerformanceProfiler() if enable_profiling else None

        # Query deduplication (prevent concurrent duplicate requests)
        self.in_flight: Dict[str, asyncio.Future] = {}

        logger.info("PerformanceOptimizer initialized")

    # ========================================================================
    # Caching Decorators
    # ========================================================================

    def cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate cache key from function and arguments.

        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Simple implementation - could be enhanced
        key_parts = [func_name]

        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])

        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")

        return ":".join(key_parts)

    def cache(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """
        Cache decorator for async functions.

        Args:
            ttl: Cache TTL override
            key_func: Optional custom key generation function

        Usage:
            @optimizer.cache(ttl=1800)
            async def expensive_query(text):
                return await process(text)
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self.cache_key(func.__name__, args, kwargs)

                # Check cache
                cached_value = self.cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_value

                # Execute function
                logger.debug(f"Cache miss: {cache_key}")
                result = await func(*args, **kwargs)

                # Store in cache
                self.cache.set(cache_key, result, ttl=ttl)

                return result

            return wrapper
        return decorator

    # ========================================================================
    # Query Deduplication
    # ========================================================================

    async def deduplicate(self, key: str, func: Callable, *args, **kwargs) -> Any:
        """
        Deduplicate concurrent requests.

        If multiple requests for the same key are in flight,
        only execute once and share result.

        Args:
            key: Deduplication key
            func: Async function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result
        """
        # Check if already in flight
        if key in self.in_flight:
            logger.debug(f"Deduplicating request: {key}")
            return await self.in_flight[key]

        # Create future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.in_flight[key] = future

        try:
            # Execute
            result = await func(*args, **kwargs)
            future.set_result(result)
            return result

        except Exception as e:
            future.set_exception(e)
            raise

        finally:
            # Remove from in-flight
            if key in self.in_flight:
                del self.in_flight[key]

    # ========================================================================
    # Profiling
    # ========================================================================

    class ProfileContext:
        """Context manager for profiling."""

        def __init__(self, profiler: PerformanceProfiler, name: str):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            self.profiler.start(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = self.profiler.stop(self.name)
            logger.debug(f"Profile {self.name}: {duration:.2f}ms")

    def profile(self, name: str):
        """
        Profile context manager.

        Usage:
            with optimizer.profile("operation"):
                result = await do_work()
        """
        if not self.profiler:
            # No-op if profiling disabled
            return self._NoOpContext()

        return self.ProfileContext(self.profiler, name)

    class _NoOpContext:
        """No-op context when profiling disabled."""
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    # ========================================================================
    # Resource Monitoring
    # ========================================================================

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            process = psutil.Process()

            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
            }
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return {}

    # ========================================================================
    # Statistics & Monitoring
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "cache": self.cache.get_stats(),
            "in_flight_requests": len(self.in_flight),
            "resources": self.get_resource_usage()
        }

        if self.profiler:
            stats["profiling"] = {
                "total_operations": len(self.profiler.entries),
                "operations": {}
            }

            # Get stats for each operation
            operation_names = set(e.name for e in self.profiler.entries)
            for name in operation_names:
                stats["profiling"]["operations"][name] = self.profiler.get_stats(name)

        return stats

    def cleanup(self) -> None:
        """Cleanup expired cache entries and old profiling data."""
        # Cleanup cache
        expired_count = self.cache.cleanup_expired()
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired cache entries")

        # Cleanup old profiling entries (keep last 1000)
        if self.profiler and len(self.profiler.entries) > 1000:
            self.profiler.entries = self.profiler.entries[-1000:]


# ============================================================================
# Optimization Utilities
# ============================================================================

async def batch_process(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Process items in batches with concurrency control.

    Args:
        items: Items to process
        process_func: Async function to process each item
        batch_size: Batch size
        max_concurrent: Maximum concurrent batches

    Returns:
        List of results
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(item):
        async with semaphore:
            return await process_func(item)

    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            process_with_semaphore(item)
            for item in batch
        ])
        results.extend(batch_results)

    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Performance Optimizer Demo")
    print("="*80)
    print()

    # Create optimizer
    optimizer = PerformanceOptimizer(
        cache_size=100,
        cache_ttl=60,
        enable_profiling=True
    )

    # Demo: Cached function
    @optimizer.cache(ttl=30)
    async def expensive_operation(query: str) -> str:
        await asyncio.sleep(0.1)  # Simulate slow operation
        return f"Result for: {query}"

    # Demo: Profile and test
    async def demo():
        print("1. Testing Cache:")

        # First call - cache miss
        with optimizer.profile("query_1"):
            result1 = await expensive_operation("test query")
        print(f"  First call: {result1}")

        # Second call - cache hit
        with optimizer.profile("query_2"):
            result2 = await expensive_operation("test query")
        print(f"  Second call (cached): {result2}")
        print()

        # Show cache stats
        print("2. Cache Statistics:")
        cache_stats = optimizer.cache.get_stats()
        print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"  Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
        print()

        # Show profiling stats
        print("3. Profiling Statistics:")
        stats = optimizer.get_statistics()
        for op_name, op_stats in stats["profiling"]["operations"].items():
            print(f"  {op_name}:")
            print(f"    Count: {op_stats['count']}")
            print(f"    Avg: {op_stats['avg_ms']:.2f}ms")
            print(f"    Min: {op_stats['min_ms']:.2f}ms")
            print(f"    Max: {op_stats['max_ms']:.2f}ms")
        print()

        # Show resource usage
        print("4. Resource Usage:")
        resources = stats["resources"]
        print(f"  Memory: {resources['memory_mb']:.1f}MB")
        print(f"  CPU: {resources['cpu_percent']:.1f}%")
        print(f"  Threads: {resources['threads']}")
        print()

    asyncio.run(demo())

    print("✓ Demo complete!")
