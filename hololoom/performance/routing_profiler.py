"""
Routing Performance Profiler
============================

Monitors and optimizes routing system performance.

Features:
- Latency tracking per backend/pattern
- Cache hit rate monitoring
- Decision timing breakdown
- Performance recommendations
- Auto-optimization suggestions
"""

import time
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

from HoloLoom.memory.routing.protocol import (
    BackendType,
    ExecutionPattern,
    RoutingDecision,
    RoutingOutcome
)


@dataclass
class LatencyMetrics:
    """Latency metrics for a component."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    samples: List[float] = field(default_factory=list)

    def record(self, latency_ms: float):
        """Record a latency sample."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.samples.append(latency_ms)

        # Keep only last 1000 samples
        if len(self.samples) > 1000:
            self.samples = self.samples[-1000:]

    @property
    def mean_ms(self) -> float:
        """Average latency."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        """Median latency."""
        if not self.samples:
            return 0.0
        return statistics.median(self.samples)

    @property
    def p95_ms(self) -> float:
        """95th percentile latency."""
        if not self.samples:
            return 0.0
        return statistics.quantiles(self.samples, n=20)[18]  # 95th percentile

    @property
    def p99_ms(self) -> float:
        """99th percentile latency."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[idx] if sorted_samples else 0.0


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    timestamp: datetime
    total_queries: int
    queries_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float
    backend_distribution: Dict[str, int]
    pattern_distribution: Dict[str, int]
    bottlenecks: List[str]


class RoutingProfiler:
    """
    Profiles routing system performance.

    Tracks:
    - Latency by backend, pattern, query type
    - Throughput (queries per second)
    - Cache effectiveness
    - Decision overhead
    - Bottleneck identification

    Usage:
        profiler = RoutingProfiler()

        # Start timing
        with profiler.time_operation("routing_decision"):
            decision = router.select_backend(...)

        # Record outcome
        profiler.record_outcome(outcome)

        # Get report
        report = profiler.generate_report()
    """

    def __init__(self):
        # Latency tracking
        self.backend_latency: Dict[BackendType, LatencyMetrics] = defaultdict(LatencyMetrics)
        self.pattern_latency: Dict[ExecutionPattern, LatencyMetrics] = defaultdict(LatencyMetrics)
        self.operation_latency: Dict[str, LatencyMetrics] = defaultdict(LatencyMetrics)

        # Counters
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()

        # Distribution tracking
        self.backend_counts: Dict[BackendType, int] = defaultdict(int)
        self.pattern_counts: Dict[ExecutionPattern, int] = defaultdict(int)

        # Snapshots for trend analysis
        self.snapshots: List[PerformanceSnapshot] = []
        self.last_snapshot_time = time.time()

    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return TimingContext(self, operation_name)

    def record_routing_decision(
        self,
        decision: RoutingDecision,
        latency_ms: float
    ):
        """Record metrics from routing decision."""
        self.query_count += 1
        self.backend_counts[decision.backend_type] += 1
        self.operation_latency['routing_decision'].record(latency_ms)

    def record_execution(
        self,
        pattern: ExecutionPattern,
        backend: BackendType,
        latency_ms: float
    ):
        """Record execution metrics."""
        self.pattern_latency[pattern].record(latency_ms)
        self.backend_latency[backend].record(latency_ms)
        self.pattern_counts[pattern] += 1

    def record_outcome(self, outcome: RoutingOutcome):
        """Record outcome metrics."""
        backend = outcome.decision.backend_type
        self.backend_latency[backend].record(outcome.latency_ms)

    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def queries_per_second(self) -> float:
        """Calculate current QPS."""
        elapsed = time.time() - self.start_time
        return self.query_count / elapsed if elapsed > 0 else 0.0

    def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Check backend latency
        for backend, metrics in self.backend_latency.items():
            if metrics.p95_ms > 1000:  # > 1s
                bottlenecks.append(
                    f"{backend.value}: p95={metrics.p95_ms:.0f}ms (slow)"
                )

        # Check pattern latency
        for pattern, metrics in self.pattern_latency.items():
            if metrics.p95_ms > 2000:  # > 2s
                bottlenecks.append(
                    f"{pattern.value}: p95={metrics.p95_ms:.0f}ms (very slow)"
                )

        # Check cache effectiveness
        if self.cache_hit_rate < 0.3:  # < 30%
            bottlenecks.append(
                f"Cache hit rate: {self.cache_hit_rate:.1%} (low)"
            )

        # Check routing overhead
        routing_metrics = self.operation_latency.get('routing_decision')
        if routing_metrics and routing_metrics.mean_ms > 100:  # > 100ms
            bottlenecks.append(
                f"Routing decision: {routing_metrics.mean_ms:.0f}ms (high overhead)"
            )

        return bottlenecks

    def generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Caching recommendations
        if self.cache_hit_rate < 0.3:
            recommendations.append(
                "⚡ Low cache hit rate - consider increasing cache size or TTL"
            )

        # Backend recommendations
        for backend, metrics in self.backend_latency.items():
            if metrics.p95_ms > 1000:
                recommendations.append(
                    f"🐌 {backend.value} is slow (p95={metrics.p95_ms:.0f}ms) - "
                    f"consider indexing or scaling"
                )

        # Pattern recommendations
        parallel_metrics = self.pattern_latency.get(ExecutionPattern.PARALLEL)
        if parallel_metrics and parallel_metrics.count > 100:
            if parallel_metrics.mean_ms > 500:
                recommendations.append(
                    "🔀 Parallel execution overhead high - consider feed-forward for simple queries"
                )

        # Routing recommendations
        routing_metrics = self.operation_latency.get('routing_decision')
        if routing_metrics and routing_metrics.mean_ms > 50:
            recommendations.append(
                "🎯 Routing decision overhead high - consider caching routing decisions"
            )

        # Load balancing recommendations
        if len(self.backend_counts) > 1:
            total = sum(self.backend_counts.values())
            max_count = max(self.backend_counts.values())
            if max_count / total > 0.7:  # One backend handling > 70%
                recommendations.append(
                    "⚖️ Backend load imbalanced - most queries to one backend"
                )

        if not recommendations:
            recommendations.append("✅ Performance looks good! No major issues detected.")

        return recommendations

    def take_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot."""
        now = datetime.now()

        # Calculate average latency across all backends
        all_samples = []
        for metrics in self.backend_latency.values():
            all_samples.extend(metrics.samples)

        avg_latency = statistics.mean(all_samples) if all_samples else 0.0
        p95_latency = statistics.quantiles(all_samples, n=20)[18] if len(all_samples) > 20 else 0.0

        snapshot = PerformanceSnapshot(
            timestamp=now,
            total_queries=self.query_count,
            queries_per_second=self.queries_per_second,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            cache_hit_rate=self.cache_hit_rate,
            backend_distribution={
                k.value: v for k, v in self.backend_counts.items()
            },
            pattern_distribution={
                k.value: v for k, v in self.pattern_counts.items()
            },
            bottlenecks=self.identify_bottlenecks()
        )

        self.snapshots.append(snapshot)
        self.last_snapshot_time = time.time()

        # Keep only last 100 snapshots
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]

        return snapshot

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'summary': {
                'total_queries': self.query_count,
                'queries_per_second': self.queries_per_second,
                'cache_hit_rate': self.cache_hit_rate,
                'uptime_seconds': time.time() - self.start_time,
            },
            'backend_performance': {
                backend.value: {
                    'count': self.backend_counts[backend],
                    'mean_ms': metrics.mean_ms,
                    'p50_ms': metrics.p50_ms,
                    'p95_ms': metrics.p95_ms,
                    'p99_ms': metrics.p99_ms,
                    'min_ms': metrics.min_ms,
                    'max_ms': metrics.max_ms,
                }
                for backend, metrics in self.backend_latency.items()
            },
            'pattern_performance': {
                pattern.value: {
                    'count': self.pattern_counts[pattern],
                    'mean_ms': metrics.mean_ms,
                    'p95_ms': metrics.p95_ms,
                }
                for pattern, metrics in self.pattern_latency.items()
            },
            'operations': {
                op: {
                    'count': metrics.count,
                    'mean_ms': metrics.mean_ms,
                    'p95_ms': metrics.p95_ms,
                }
                for op, metrics in self.operation_latency.items()
            },
            'bottlenecks': self.identify_bottlenecks(),
            'recommendations': self.generate_recommendations(),
        }

    def print_report(self):
        """Print human-readable performance report."""
        report = self.generate_report()

        print("\n" + "="*70)
        print("ROUTING PERFORMANCE REPORT")
        print("="*70 + "\n")

        # Summary
        summary = report['summary']
        print("📊 SUMMARY")
        print(f"  Total Queries: {summary['total_queries']}")
        print(f"  QPS: {summary['queries_per_second']:.2f}")
        print(f"  Cache Hit Rate: {summary['cache_hit_rate']:.1%}")
        print(f"  Uptime: {summary['uptime_seconds']:.0f}s")
        print()

        # Backend performance
        print("🗄️  BACKEND PERFORMANCE")
        for backend, metrics in report['backend_performance'].items():
            print(f"  {backend}:")
            print(f"    Queries: {metrics['count']}")
            print(f"    Mean: {metrics['mean_ms']:.0f}ms")
            print(f"    P95: {metrics['p95_ms']:.0f}ms")
            print(f"    Range: {metrics['min_ms']:.0f}-{metrics['max_ms']:.0f}ms")
        print()

        # Pattern performance
        if report['pattern_performance']:
            print("🔀 EXECUTION PATTERN PERFORMANCE")
            for pattern, metrics in report['pattern_performance'].items():
                print(f"  {pattern}:")
                print(f"    Count: {metrics['count']}")
                print(f"    Mean: {metrics['mean_ms']:.0f}ms")
                print(f"    P95: {metrics['p95_ms']:.0f}ms")
            print()

        # Bottlenecks
        if report['bottlenecks']:
            print("⚠️  BOTTLENECKS")
            for bottleneck in report['bottlenecks']:
                print(f"  • {bottleneck}")
            print()

        # Recommendations
        print("💡 RECOMMENDATIONS")
        for rec in report['recommendations']:
            print(f"  {rec}")

        print("\n" + "="*70 + "\n")


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, profiler: RoutingProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.profiler.operation_latency[self.operation_name].record(elapsed_ms)
        return False


# Global profiler instance
_global_profiler: Optional[RoutingProfiler] = None


def get_profiler() -> RoutingProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = RoutingProfiler()
    return _global_profiler


def reset_profiler():
    """Reset global profiler."""
    global _global_profiler
    _global_profiler = RoutingProfiler()
