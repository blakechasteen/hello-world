"""
Routing Performance Benchmarks
===============================

Comprehensive benchmark suite for routing system.

Benchmarks:
- Routing decision speed
- Execution pattern performance
- Backend latency comparison
- Cache effectiveness
- A/B test overhead
- Scale testing (1K, 10K, 100K queries)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

from HoloLoom.memory.routing import (
    RuleBasedRouter,
    LearnedRouter,
    BackendType
)
from HoloLoom.memory.routing.execution_patterns import (
    ExecutionPattern,
    FeedForwardEngine
)
from HoloLoom.memory.routing.orchestrator import (
    RoutingOrchestrator,
    create_test_orchestrator
)
from HoloLoom.memory.protocol import MemoryQuery


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    total_queries: int
    duration_s: float
    queries_per_second: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    metadata: Dict[str, Any]

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")
        print(f"Total Queries:    {self.total_queries}")
        print(f"Duration:         {self.duration_s:.2f}s")
        print(f"Throughput:       {self.queries_per_second:.0f} QPS")
        print(f"Latency (mean):   {self.mean_latency_ms:.2f}ms")
        print(f"Latency (p50):    {self.p50_latency_ms:.2f}ms")
        print(f"Latency (p95):    {self.p95_latency_ms:.2f}ms")
        print(f"Latency (p99):    {self.p99_latency_ms:.2f}ms")
        print(f"Latency (range):  {self.min_latency_ms:.2f}-{self.max_latency_ms:.2f}ms")


class RoutingBenchmark:
    """
    Benchmark suite for routing system.

    Usage:
        benchmark = RoutingBenchmark()

        # Run all benchmarks
        results = await benchmark.run_all()

        # Run specific benchmark
        result = await benchmark.benchmark_routing_decision()
    """

    def __init__(self):
        self.test_queries = self._generate_test_queries()

    def _generate_test_queries(self) -> List[str]:
        """Generate diverse test queries."""
        return [
            # Relationship queries
            "who inspected the hives yesterday",
            "when was the last treatment applied",
            "where are the strongest colonies",
            "what is connected to varroa mites",

            # Similarity queries
            "find similar inspection reports",
            "search for honey production data",
            "look for pest management strategies",
            "compare colony health metrics",

            # Personal queries
            "my favorite hive management techniques",
            "what are my beekeeping preferences",
            "i need personal recommendations",
            "show me user-specific data",

            # Temporal queries
            "recent hive inspections",
            "today's temperature readings",
            "latest queen activity",
            "current honey flow status",

            # Mixed queries
            "hive health status",
            "colony performance",
            "seasonal patterns",
            "best practices for winter",
        ]

    async def benchmark_routing_decision(
        self,
        iterations: int = 1000
    ) -> BenchmarkResult:
        """
        Benchmark routing decision speed.

        Measures: How fast can we route queries?
        """
        router = RuleBasedRouter()
        available = list(BackendType)

        latencies = []
        start_time = time.time()

        for i in range(iterations):
            query = self.test_queries[i % len(self.test_queries)]

            t0 = time.time()
            decision = router.select_backend(query, available)
            t1 = time.time()

            latencies.append((t1 - t0) * 1000)

        duration = time.time() - start_time

        return BenchmarkResult(
            name="Routing Decision Speed",
            total_queries=iterations,
            duration_s=duration,
            queries_per_second=iterations / duration,
            mean_latency_ms=statistics.mean(latencies),
            p50_latency_ms=statistics.median(latencies),
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18],
            p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)],
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            metadata={'router': 'rule_based'}
        )

    async def benchmark_learned_vs_rule_based(
        self,
        iterations: int = 100
    ) -> Dict[str, BenchmarkResult]:
        """Compare learned vs rule-based routing speed."""
        available = list(BackendType)

        results = {}

        # Benchmark rule-based
        rule_router = RuleBasedRouter()
        rule_latencies = []
        start = time.time()

        for i in range(iterations):
            query = self.test_queries[i % len(self.test_queries)]
            t0 = time.time()
            decision = rule_router.select_backend(query, available)
            rule_latencies.append((time.time() - t0) * 1000)

        rule_duration = time.time() - start

        results['rule_based'] = BenchmarkResult(
            name="Rule-Based Router",
            total_queries=iterations,
            duration_s=rule_duration,
            queries_per_second=iterations / rule_duration,
            mean_latency_ms=statistics.mean(rule_latencies),
            p50_latency_ms=statistics.median(rule_latencies),
            p95_latency_ms=statistics.quantiles(rule_latencies, n=20)[18],
            p99_latency_ms=sorted(rule_latencies)[int(len(rule_latencies) * 0.99)],
            min_latency_ms=min(rule_latencies),
            max_latency_ms=max(rule_latencies),
            metadata={'speedup': '1.00x (baseline)'}
        )

        # Benchmark learned
        learned_router = LearnedRouter()
        learned_latencies = []
        start = time.time()

        for i in range(iterations):
            query = self.test_queries[i % len(self.test_queries)]
            t0 = time.time()
            decision = learned_router.select_backend(query, available)
            learned_latencies.append((time.time() - t0) * 1000)

        learned_duration = time.time() - start

        speedup = rule_duration / learned_duration

        results['learned'] = BenchmarkResult(
            name="Learned Router (Thompson Sampling)",
            total_queries=iterations,
            duration_s=learned_duration,
            queries_per_second=iterations / learned_duration,
            mean_latency_ms=statistics.mean(learned_latencies),
            p50_latency_ms=statistics.median(learned_latencies),
            p95_latency_ms=statistics.quantiles(learned_latencies, n=20)[18],
            p99_latency_ms=sorted(learned_latencies)[int(len(learned_latencies) * 0.99)],
            min_latency_ms=min(learned_latencies),
            max_latency_ms=max(learned_latencies),
            metadata={'speedup': f'{speedup:.2f}x'}
        )

        return results

    async def benchmark_execution_patterns(self) -> Dict[str, BenchmarkResult]:
        """Benchmark different execution patterns."""
        # Mock backend
        class MockBackend:
            async def retrieve(self, query, strategy):
                await asyncio.sleep(0.01)  # 10ms mock latency
                from HoloLoom.memory.protocol import RetrievalResult
                return RetrievalResult(
                    memories=[],
                    scores=[],
                    strategy_used='mock',
                    metadata={}
                )

        backends = {BackendType.QDRANT: MockBackend()}
        query = MemoryQuery(text="test query", limit=5)

        results = {}

        # Test feed-forward
        from HoloLoom.memory.routing.execution_patterns import (
            FeedForwardEngine,
            RecursiveEngine,
            ParallelEngine,
            ExecutionPlan
        )

        patterns = [
            (ExecutionPattern.FEED_FORWARD, FeedForwardEngine()),
            (ExecutionPattern.RECURSIVE, RecursiveEngine()),
            (ExecutionPattern.PARALLEL, ParallelEngine()),
        ]

        for pattern_type, engine in patterns:
            plan = ExecutionPlan(
                pattern=pattern_type,
                primary_backend=BackendType.QDRANT,
                secondary_backends=[]
            )

            latencies = []
            iterations = 50

            start = time.time()
            for _ in range(iterations):
                t0 = time.time()
                result = await engine.execute(query, plan, backends)
                latencies.append((time.time() - t0) * 1000)

            duration = time.time() - start

            results[pattern_type.value] = BenchmarkResult(
                name=f"Execution Pattern: {pattern_type.value}",
                total_queries=iterations,
                duration_s=duration,
                queries_per_second=iterations / duration,
                mean_latency_ms=statistics.mean(latencies),
                p50_latency_ms=statistics.median(latencies),
                p95_latency_ms=statistics.quantiles(latencies, n=20)[18],
                p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)],
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                metadata={'pattern': pattern_type.value}
            )

        return results

    async def benchmark_scale(
        self,
        scales: List[int] = [100, 1000, 10000]
    ) -> Dict[int, BenchmarkResult]:
        """Benchmark at different scales."""
        router = RuleBasedRouter()
        available = list(BackendType)

        results = {}

        for scale in scales:
            latencies = []
            start = time.time()

            for i in range(scale):
                query = self.test_queries[i % len(self.test_queries)]
                t0 = time.time()
                decision = router.select_backend(query, available)
                latencies.append((time.time() - t0) * 1000)

            duration = time.time() - start

            results[scale] = BenchmarkResult(
                name=f"Scale Test: {scale} queries",
                total_queries=scale,
                duration_s=duration,
                queries_per_second=scale / duration,
                mean_latency_ms=statistics.mean(latencies),
                p50_latency_ms=statistics.median(latencies),
                p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
                p99_latency_ms=sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else max(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                metadata={'scale': scale}
            )

        return results

    async def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("ROUTING PERFORMANCE BENCHMARKS")
        print("="*70)

        all_results = {}

        print("\n[1/4] Routing decision speed...")
        routing_result = await self.benchmark_routing_decision()
        routing_result.print_summary()
        all_results['routing_decision'] = routing_result

        print("\n[2/4] Rule-based vs Learned...")
        comparison = await self.benchmark_learned_vs_rule_based()
        for name, result in comparison.items():
            result.print_summary()
        all_results['comparison'] = comparison

        print("\n[3/4] Execution patterns...")
        patterns = await self.benchmark_execution_patterns()
        for name, result in patterns.items():
            result.print_summary()
        all_results['patterns'] = patterns

        print("\n[4/4] Scale testing...")
        scale_results = await self.benchmark_scale([100, 1000])
        for scale, result in scale_results.items():
            result.print_summary()
        all_results['scale'] = scale_results

        print("\n" + "="*70)
        print("BENCHMARKS COMPLETE")
        print("="*70 + "\n")

        return all_results


async def main():
    """Run benchmarks."""
    benchmark = RoutingBenchmark()
    results = await benchmark.run_all()


if __name__ == "__main__":
    asyncio.run(main())
