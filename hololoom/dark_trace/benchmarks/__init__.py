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
    LatencyProfiler,
    MemoryProfiler,
    profile_latency,
    profile_memory,
)
from .reporter import (
    BenchmarkReporter,
    export_prometheus_metrics,
    generate_report,
)
from .suite import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    run_benchmark_suite,
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
