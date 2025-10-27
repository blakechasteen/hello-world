"""
HoloLoom Performance Optimizations
===================================
Caching, profiling, benchmarking, metrics, and optimization utilities.

Components:
- QueryCache: LRU cache with TTL for query results
- Profiler: Hierarchical timing and memory tracking
- ProfilerRegistry: Aggregate profiling data across runs
- MetricsCollector: Real-time system and application metrics
- RoutingProfiler: Specialized profiling for routing decisions
- Benchmark: Performance testing suite for orchestrators
"""

from .cache import QueryCache, CacheEntry
from .profiler import Profiler, ProfilerRegistry, profile_async, profile_sync, get_global_registry
from .metrics import MetricsCollector, Metric, get_global_metrics
from .routing_profiler import RoutingProfiler, get_profiler, reset_profiler
from .routing_benchmarks import RoutingBenchmark, BenchmarkResult

__all__ = [
    # Caching
    "QueryCache",
    "CacheEntry",

    # General Profiling
    "Profiler",
    "ProfilerRegistry",
    "profile_async",
    "profile_sync",
    "get_global_registry",

    # Metrics
    "MetricsCollector",
    "Metric",
    "get_global_metrics",

    # Routing-Specific Profiling
    "RoutingProfiler",
    "get_profiler",
    "reset_profiler",

    # Benchmarking
    "RoutingBenchmark",
    "BenchmarkResult",
]
