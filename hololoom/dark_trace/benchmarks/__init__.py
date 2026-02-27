"""
Dark Trace Performance Benchmarks

Production-grade benchmarking and profiling for interpretability operations.

Components:
- MemoryProfiler: Track memory usage during operations
- LatencyProfiler: Measure operation latency with percentiles
- BenchmarkSuite: Run comprehensive benchmark suites
- BenchmarkReporter: Generate reports in multiple formats

Author: HoloLoom Team
Created: December 2025
"""

from .profiler import (
    MemoryProfiler,
    LatencyProfiler,
    profile_memory,
    profile_latency,
)

from .suite import (
    BenchmarkSuite,
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark_suite,
)

from .reporter import (
    BenchmarkReporter,
    generate_report,
    export_prometheus_metrics,
)

__all__ = [
    # Profilers
    "MemoryProfiler",
    "LatencyProfiler",
    "profile_memory",
    "profile_latency",
    # Suite
    "BenchmarkSuite",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark_suite",
    # Reporter
    "BenchmarkReporter",
    "generate_report",
    "export_prometheus_metrics",
]
