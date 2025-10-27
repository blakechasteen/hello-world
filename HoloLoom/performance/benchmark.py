#!/usr/bin/env python3
"""
Benchmark Suite - Performance testing for HoloLoom orchestrators
================================================================
Compares different orchestrator implementations and configurations
to identify bottlenecks and optimize query processing.

Usage:
    python -m HoloLoom.performance.benchmark --mode all --queries 100
"""

import asyncio
import argparse
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.performance.profiler import Profiler, ProfilerRegistry
from HoloLoom.performance.metrics import MetricsCollector


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    mode: str
    query_count: int
    total_duration_s: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queries_per_second: float
    memory_mb: float
    success_rate: float


def create_test_queries(count: int) -> List[Query]:
    """Create test queries for benchmarking."""
    templates = [
        "What is the best way to implement {topic}?",
        "Explain {topic} in simple terms",
        "How does {topic} work?",
        "What are the benefits of {topic}?",
        "Compare {topic1} and {topic2}",
    ]

    topics = [
        "machine learning", "neural networks", "transformers",
        "knowledge graphs", "embeddings", "reinforcement learning",
        "attention mechanisms", "gradient descent", "backpropagation",
        "policy gradients", "Thompson sampling", "Bayesian optimization"
    ]

    queries = []
    for i in range(count):
        template = templates[i % len(templates)]
        if "{topic1}" in template:
            topic1 = topics[i % len(topics)]
            topic2 = topics[(i + 1) % len(topics)]
            text = template.format(topic1=topic1, topic2=topic2)
        else:
            topic = topics[i % len(topics)]
            text = template.format(topic=topic)

        queries.append(Query(text=text, timestamp=time.time()))

    return queries


def create_test_shards() -> List[MemoryShard]:
    """Create test memory shards for context."""
    return [
        MemoryShard(
            text="Machine learning is a subset of AI that learns from data.",
            source="ml_basics",
            timestamp=time.time(),
            entities=["machine learning", "AI", "data"],
            motifs=["technical", "educational"]
        ),
        MemoryShard(
            text="Neural networks are inspired by biological neurons.",
            source="nn_basics",
            timestamp=time.time(),
            entities=["neural networks", "neurons"],
            motifs=["technical", "biology"]
        ),
        MemoryShard(
            text="Transformers revolutionized NLP with attention mechanisms.",
            source="transformers",
            timestamp=time.time(),
            entities=["transformers", "NLP", "attention"],
            motifs=["technical", "modern"]
        ),
        MemoryShard(
            text="Knowledge graphs represent entities and relationships.",
            source="kg_basics",
            timestamp=time.time(),
            entities=["knowledge graphs", "entities", "relationships"],
            motifs=["technical", "data"]
        ),
        MemoryShard(
            text="Thompson Sampling balances exploration and exploitation.",
            source="bandits",
            timestamp=time.time(),
            entities=["Thompson Sampling", "exploration", "exploitation"],
            motifs=["technical", "optimization"]
        ),
    ]


async def benchmark_orchestrator(
    name: str,
    config: Config,
    queries: List[Query],
    shards: List[MemoryShard],
    metrics: MetricsCollector
) -> BenchmarkResult:
    """
    Benchmark a single orchestrator configuration.

    Args:
        name: Benchmark name
        config: Configuration to test
        queries: List of test queries
        shards: Memory shards for context
        metrics: Metrics collector

    Returns:
        BenchmarkResult with performance data
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Mode: {config.execution_mode.value}")
    print(f"Queries: {len(queries)}")
    print(f"{'='*60}\n")

    registry = ProfilerRegistry()
    latencies = []
    successes = 0

    start_time = time.time()

    async with WeavingShuttle(cfg=config, shards=shards) as shuttle:
        for i, query in enumerate(queries):
            try:
                async with Profiler(f"query_{i}") as prof:
                    spacetime = await shuttle.weave(query)
                    successes += 1

                # Record metrics
                if prof.entry and prof.entry.duration:
                    latency_ms = prof.entry.duration * 1000
                    latencies.append(latency_ms)
                    metrics.record_latency("query_processing", latency_ms)

                registry.record(prof)

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(queries)} queries")

            except Exception as e:
                print(f"  Error on query {i}: {e}")

    end_time = time.time()
    total_duration = end_time - start_time

    # Calculate statistics
    if latencies:
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        result = BenchmarkResult(
            name=name,
            mode=config.execution_mode.value,
            query_count=len(queries),
            total_duration_s=total_duration,
            mean_latency_ms=sum(latencies) / n,
            median_latency_ms=sorted_latencies[n // 2],
            p95_latency_ms=sorted_latencies[int(0.95 * n)],
            p99_latency_ms=sorted_latencies[int(0.99 * n)],
            queries_per_second=successes / total_duration if total_duration > 0 else 0,
            memory_mb=metrics.get_system_metrics()["memory_mb"],
            success_rate=successes / len(queries)
        )
    else:
        result = BenchmarkResult(
            name=name,
            mode=config.execution_mode.value,
            query_count=len(queries),
            total_duration_s=total_duration,
            mean_latency_ms=0,
            median_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            queries_per_second=0,
            memory_mb=0,
            success_rate=0
        )

    return result


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80 + "\n")

    # Header
    print(f"{'Name':<25} {'Mode':<10} {'Queries':<8} {'Mean':<10} {'P95':<10} {'QPS':<8} {'Mem(MB)':<10}")
    print("-" * 80)

    # Results
    for result in results:
        print(
            f"{result.name:<25} "
            f"{result.mode:<10} "
            f"{result.query_count:<8} "
            f"{result.mean_latency_ms:<10.2f} "
            f"{result.p95_latency_ms:<10.2f} "
            f"{result.queries_per_second:<8.2f} "
            f"{result.memory_mb:<10.2f}"
        )

    print("\n" + "="*80 + "\n")

    # Winner analysis
    if results:
        fastest = min(results, key=lambda r: r.mean_latency_ms)
        highest_qps = max(results, key=lambda r: r.queries_per_second)
        most_efficient = min(results, key=lambda r: r.memory_mb)

        print("🏆 WINNERS:")
        print(f"  Fastest (mean latency): {fastest.name} ({fastest.mean_latency_ms:.2f}ms)")
        print(f"  Highest throughput: {highest_qps.name} ({highest_qps.queries_per_second:.2f} QPS)")
        print(f"  Most memory efficient: {most_efficient.name} ({most_efficient.memory_mb:.2f}MB)")
        print()


async def run_benchmarks(query_count: int, modes: List[str]) -> List[BenchmarkResult]:
    """
    Run benchmarks for specified execution modes.

    Args:
        query_count: Number of queries to test
        modes: List of modes to benchmark (["bare", "fast", "fused"] or ["all"])

    Returns:
        List of benchmark results
    """
    # Create test data
    queries = create_test_queries(query_count)
    shards = create_test_shards()
    metrics = MetricsCollector()

    results = []

    # Mode configurations
    mode_configs = {
        "bare": Config.bare(),
        "fast": Config.fast(),
        "fused": Config.fused()
    }

    if "all" in modes:
        modes = ["bare", "fast", "fused"]

    for mode in modes:
        if mode not in mode_configs:
            print(f"Warning: Unknown mode '{mode}', skipping")
            continue

        config = mode_configs[mode]
        result = await benchmark_orchestrator(
            name=f"WeavingShuttle-{mode.upper()}",
            config=config,
            queries=queries,
            shards=shards,
            metrics=metrics
        )
        results.append(result)

    return results


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="HoloLoom Performance Benchmark Suite")
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Number of queries to benchmark (default: 50)"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["all"],
        choices=["all", "bare", "fast", "fused"],
        help="Execution modes to benchmark (default: all)"
    )

    args = parser.parse_args()

    print("\n🔬 HoloLoom Performance Benchmark Suite")
    print(f"Queries: {args.queries}")
    print(f"Modes: {', '.join(args.modes)}")

    # Run benchmarks
    results = asyncio.run(run_benchmarks(args.queries, args.modes))

    # Print results
    print_results(results)

    # Export results
    import json
    results_file = Path("benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"📊 Results exported to: {results_file}")


if __name__ == "__main__":
    main()
