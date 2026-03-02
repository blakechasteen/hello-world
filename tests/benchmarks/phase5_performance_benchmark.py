#!/usr/bin/env python3
"""
Phase 5 Performance Benchmark Suite
====================================

Comprehensive benchmarking of compositional cache performance with:
1. Cold/warm/hot path analysis
2. Cache hit rate progression
3. Memory usage tracking
4. Compositional reuse patterns
5. Production workload simulation

Usage:
    python benchmarks/phase5_performance_benchmark.py

Outputs:
    - benchmarks/results/phase5_benchmark.json (raw data)
    - benchmarks/results/phase5_benchmark.md (formatted report)
"""

import asyncio
import time
import json
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from hololoom.config import Config
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.documentation.types import MemoryShard

logging.basicConfig(level=logging.WARNING)  # Quiet for benchmarking
logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    cache_sizes: Dict[str, int]
    query_count: int
    warmup_queries: int = 5
    description: str = ""


BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="small_cache",
        cache_sizes={"parse": 100, "merge": 500},
        query_count=50,
        warmup_queries=5,
        description="Small cache (100/500) - memory-constrained environments"
    ),
    BenchmarkConfig(
        name="medium_cache",
        cache_sizes={"parse": 1000, "merge": 5000},
        query_count=100,
        warmup_queries=10,
        description="Medium cache (1000/5000) - typical development"
    ),
    BenchmarkConfig(
        name="large_cache",
        cache_sizes={"parse": 10000, "merge": 50000},
        query_count=200,
        warmup_queries=20,
        description="Large cache (10k/50k) - production (recommended)"
    ),
    BenchmarkConfig(
        name="xlarge_cache",
        cache_sizes={"parse": 100000, "merge": 500000},
        query_count=500,
        warmup_queries=50,
        description="XLarge cache (100k/500k) - high-volume production"
    ),
]


# ============================================================================
# Test Queries
# ============================================================================

def generate_test_queries() -> List[str]:
    """
    Generate realistic test queries with compositional overlap.

    Designed to have:
    - Exact repeats (100% cache hit)
    - Partial overlaps (compositional reuse)
    - Novel queries (cache miss)
    """
    base_phrases = [
        # Technical queries
        "Thompson Sampling",
        "reinforcement learning",
        "neural networks",
        "deep learning",
        "machine learning",

        # Descriptive phrases
        "the big red ball",
        "a small blue box",
        "the old wooden table",
        "a beautiful sunny day",
        "the quick brown fox",

        # Action phrases
        "jumped over the fence",
        "ran through the park",
        "walked along the beach",
        "climbed up the mountain",
        "swam across the lake",
    ]

    # Generate queries with compositional structure
    queries = []

    # Pattern 1: Direct phrases (20%)
    for phrase in base_phrases[:4]:
        queries.append(phrase)

    # Pattern 2: Questions about phrases (30%)
    question_templates = [
        "What is {}?",
        "How does {} work?",
        "Explain {} in detail",
        "Tell me about {}",
        "Define {}",
    ]
    for phrase in base_phrases[4:10]:
        for template in question_templates[:2]:
            queries.append(template.format(phrase))

    # Pattern 3: Combined phrases (30%)
    combined_templates = [
        "{} and {}",
        "{} with {}",
        "{} using {}",
        "Compare {} to {}",
        "Difference between {} and {}",
    ]
    for i in range(0, min(6, len(base_phrases) - 1)):
        for template in combined_templates[:2]:
            queries.append(template.format(base_phrases[i], base_phrases[i + 1]))

    # Pattern 4: Variations with compositional overlap (20%)
    for phrase in base_phrases[10:15]:
        queries.extend([
            f"the {phrase}",
            f"a {phrase}",
            f"{phrase} example",
            f"{phrase} tutorial",
        ])

    return queries


def create_test_shards() -> List[MemoryShard]:
    """Create realistic test memory shards."""
    return [
        MemoryShard(
            id=f"shard_{i}",
            text=text,
            episode="benchmark",
            entities=[],
            motifs=["test"]
        )
        for i, text in enumerate([
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
            "Reinforcement learning is a machine learning paradigm focused on decision making",
            "Neural networks are computing systems inspired by biological neural networks",
            "Deep learning uses multiple layers to progressively extract higher-level features",
            "Machine learning is the study of algorithms that improve through experience",
            "The big red ball bounced down the hill",
            "A small blue box sat on the shelf",
            "The old wooden table creaked under the weight",
            "A beautiful sunny day greeted the morning",
            "The quick brown fox jumped over the lazy dog",
        ])
    ]


# ============================================================================
# Benchmark Execution
# ============================================================================

@dataclass
class QueryResult:
    """Result for a single query."""
    query: str
    time_ms: float
    parse_hit: bool
    merge_hit: bool
    is_warmup: bool


@dataclass
class BenchmarkResult:
    """Results for entire benchmark."""
    config_name: str
    description: str
    cache_sizes: Dict[str, int]
    total_queries: int
    warmup_queries: int

    # Timing statistics
    mean_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    min_time_ms: float
    max_time_ms: float

    # Cache statistics
    parse_hit_rate: float
    merge_hit_rate: float
    overall_hit_rate: float

    # Speedup analysis
    cold_path_avg_ms: float
    warm_path_avg_ms: float
    hot_path_avg_ms: float
    cold_to_hot_speedup: float

    # Memory (if available)
    final_parse_cache_size: int
    final_merge_cache_size: int

    # Raw data
    query_results: List[Dict]


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    print(f"\n{'='*80}")
    print(f"Running: {config.name}")
    print(f"Description: {config.description}")
    print(f"Cache sizes: parse={config.cache_sizes['parse']}, merge={config.cache_sizes['merge']}")
    print(f"Queries: {config.query_count} + {config.warmup_queries} warmup")
    print(f"{'='*80}\n")

    # Create config
    hololoom_config = Config.fused()
    hololoom_config.enable_linguistic_gate = True
    hololoom_config.use_compositional_cache = True
    hololoom_config.linguistic_mode = "disabled"
    hololoom_config.parse_cache_size = config.cache_sizes["parse"]
    hololoom_config.merge_cache_size = config.cache_sizes["merge"]

    # Create test data
    shards = create_test_shards()
    queries = generate_test_queries()

    # Ensure we have enough queries
    while len(queries) < config.query_count + config.warmup_queries:
        queries.extend(queries)  # Repeat if needed

    queries = queries[:config.query_count + config.warmup_queries]

    # Initialize orchestrator
    shuttle = WeavingOrchestrator(cfg=hololoom_config, shards=shards)

    if not shuttle.linguistic_gate or not shuttle.linguistic_gate.compositional_cache:
        raise RuntimeError("Phase 5 not available (spaCy not installed?)")

    cache = shuttle.linguistic_gate.compositional_cache

    # Run queries
    results = []

    print("Running warmup queries...")
    for i, query in enumerate(queries[:config.warmup_queries]):
        start = time.perf_counter()
        embedding, trace = cache.get_compositional_embedding(query, return_trace=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append(QueryResult(
            query=query,
            time_ms=elapsed_ms,
            parse_hit="parse" in trace.get("hits", []),
            merge_hit=any("merge" in h for h in trace.get("hits", [])),
            is_warmup=True
        ))

        if (i + 1) % 10 == 0:
            print(f"  Warmup: {i + 1}/{config.warmup_queries}")

    print(f"\nRunning benchmark queries...")
    for i, query in enumerate(queries[config.warmup_queries:]):
        start = time.perf_counter()
        embedding, trace = cache.get_compositional_embedding(query, return_trace=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append(QueryResult(
            query=query,
            time_ms=elapsed_ms,
            parse_hit="parse" in trace.get("hits", []),
            merge_hit=any("merge" in h for h in trace.get("hits", [])),
            is_warmup=False
        ))

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{config.query_count}")

    # Analyze results (exclude warmup)
    benchmark_results = [r for r in results if not r.is_warmup]

    times = [r.time_ms for r in benchmark_results]
    parse_hits = sum(1 for r in benchmark_results if r.parse_hit)
    merge_hits = sum(1 for r in benchmark_results if r.merge_hit)

    # Categorize by cache behavior
    cold_queries = [r for r in benchmark_results if not r.parse_hit and not r.merge_hit]
    warm_queries = [r for r in benchmark_results if r.parse_hit != r.merge_hit]  # One hit
    hot_queries = [r for r in benchmark_results if r.parse_hit and r.merge_hit]

    cold_avg = statistics.mean([r.time_ms for r in cold_queries]) if cold_queries else 0
    warm_avg = statistics.mean([r.time_ms for r in warm_queries]) if warm_queries else 0
    hot_avg = statistics.mean([r.time_ms for r in hot_queries]) if hot_queries else 0

    speedup = cold_avg / hot_avg if hot_avg > 0 else 0

    # Get cache statistics
    cache_stats = cache.get_statistics()

    # Create result
    result = BenchmarkResult(
        config_name=config.name,
        description=config.description,
        cache_sizes=config.cache_sizes,
        total_queries=config.query_count,
        warmup_queries=config.warmup_queries,

        mean_time_ms=statistics.mean(times),
        median_time_ms=statistics.median(times),
        p95_time_ms=sorted(times)[int(len(times) * 0.95)],
        p99_time_ms=sorted(times)[int(len(times) * 0.99)],
        min_time_ms=min(times),
        max_time_ms=max(times),

        parse_hit_rate=cache_stats["parse_cache"]["hit_rate"],
        merge_hit_rate=cache_stats["merge_cache"]["hit_rate"],
        overall_hit_rate=cache_stats["overall_hit_rate"],

        cold_path_avg_ms=cold_avg,
        warm_path_avg_ms=warm_avg,
        hot_path_avg_ms=hot_avg,
        cold_to_hot_speedup=speedup,

        final_parse_cache_size=cache_stats["parse_cache"]["size"],
        final_merge_cache_size=cache_stats["merge_cache"]["size"],

        query_results=[asdict(r) for r in benchmark_results[:100]]  # Sample
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS: {config.name}")
    print(f"{'='*80}")
    print(f"\nTiming:")
    print(f"  Mean: {result.mean_time_ms:.3f}ms")
    print(f"  Median: {result.median_time_ms:.3f}ms")
    print(f"  P95: {result.p95_time_ms:.3f}ms")
    print(f"  P99: {result.p99_time_ms:.3f}ms")
    print(f"\nCache Performance:")
    print(f"  Parse hit rate: {result.parse_hit_rate:.1%}")
    print(f"  Merge hit rate: {result.merge_hit_rate:.1%}")
    print(f"  Overall hit rate: {result.overall_hit_rate:.1%}")
    print(f"\nSpeedup Analysis:")
    print(f"  Cold path: {result.cold_path_avg_ms:.3f}ms ({len(cold_queries)} queries)")
    print(f"  Warm path: {result.warm_path_avg_ms:.3f}ms ({len(warm_queries)} queries)")
    print(f"  Hot path: {result.hot_path_avg_ms:.3f}ms ({len(hot_queries)} queries)")
    print(f"  Cold → Hot speedup: {result.cold_to_hot_speedup:.1f}×")
    print(f"\nCache Utilization:")
    print(f"  Parse cache: {result.final_parse_cache_size}/{config.cache_sizes['parse']} ({result.final_parse_cache_size/config.cache_sizes['parse']:.1%})")
    print(f"  Merge cache: {result.final_merge_cache_size}/{config.cache_sizes['merge']} ({result.final_merge_cache_size/config.cache_sizes['merge']:.1%})")

    return result


# ============================================================================
# Report Generation
# ============================================================================

def generate_markdown_report(results: List[BenchmarkResult], output_path: Path):
    """Generate comprehensive markdown report."""
    report = []

    report.append("# Phase 5 Performance Benchmark Report")
    report.append("")
    report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # Executive summary
    report.append("## Executive Summary")
    report.append("")

    # Find best configuration
    best_speedup = max(results, key=lambda r: r.cold_to_hot_speedup)
    best_hit_rate = max(results, key=lambda r: r.overall_hit_rate)
    fastest_median = min(results, key=lambda r: r.median_time_ms)

    report.append(f"- **Best speedup**: {best_speedup.config_name} ({best_speedup.cold_to_hot_speedup:.1f}× cold→hot)")
    report.append(f"- **Best hit rate**: {best_hit_rate.config_name} ({best_hit_rate.overall_hit_rate:.1%} overall)")
    report.append(f"- **Fastest median**: {fastest_median.config_name} ({fastest_median.median_time_ms:.3f}ms)")
    report.append("")

    # Recommendations
    report.append("### Recommendations")
    report.append("")
    report.append("Based on benchmark results:")
    report.append("")

    # Analyze cache size impact
    large_config = next((r for r in results if r.config_name == "large_cache"), None)
    if large_config:
        report.append(f"**Production (recommended)**: {large_config.config_name}")
        report.append(f"- Cache sizes: parse={large_config.cache_sizes['parse']}, merge={large_config.cache_sizes['merge']}")
        report.append(f"- Expected hit rate: {large_config.overall_hit_rate:.1%}")
        report.append(f"- Expected speedup: {large_config.cold_to_hot_speedup:.1f}×")
        report.append(f"- Median latency: {large_config.median_time_ms:.3f}ms")
        report.append("")

    report.append("---")
    report.append("")

    # Detailed results
    report.append("## Detailed Results")
    report.append("")

    for result in results:
        report.append(f"### {result.config_name}")
        report.append("")
        report.append(f"**Description**: {result.description}")
        report.append("")

        report.append("#### Configuration")
        report.append("")
        report.append(f"- Parse cache size: {result.cache_sizes['parse']}")
        report.append(f"- Merge cache size: {result.cache_sizes['merge']}")
        report.append(f"- Total queries: {result.total_queries}")
        report.append(f"- Warmup queries: {result.warmup_queries}")
        report.append("")

        report.append("#### Timing Statistics")
        report.append("")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Mean | {result.mean_time_ms:.3f}ms |")
        report.append(f"| Median | {result.median_time_ms:.3f}ms |")
        report.append(f"| P95 | {result.p95_time_ms:.3f}ms |")
        report.append(f"| P99 | {result.p99_time_ms:.3f}ms |")
        report.append(f"| Min | {result.min_time_ms:.3f}ms |")
        report.append(f"| Max | {result.max_time_ms:.3f}ms |")
        report.append("")

        report.append("#### Cache Performance")
        report.append("")
        report.append(f"| Cache | Hit Rate |")
        report.append(f"|-------|----------|")
        report.append(f"| Parse | {result.parse_hit_rate:.1%} |")
        report.append(f"| Merge | {result.merge_hit_rate:.1%} |")
        report.append(f"| Overall | {result.overall_hit_rate:.1%} |")
        report.append("")

        report.append("#### Speedup Analysis")
        report.append("")
        report.append(f"| Path | Avg Time | Speedup |")
        report.append(f"|------|----------|---------|")
        report.append(f"| Cold (no cache) | {result.cold_path_avg_ms:.3f}ms | 1× (baseline) |")
        report.append(f"| Warm (partial cache) | {result.warm_path_avg_ms:.3f}ms | {result.cold_path_avg_ms/result.warm_path_avg_ms if result.warm_path_avg_ms > 0 else 0:.1f}× |")
        report.append(f"| Hot (full cache) | {result.hot_path_avg_ms:.3f}ms | **{result.cold_to_hot_speedup:.1f}×** |")
        report.append("")

        report.append("#### Cache Utilization")
        report.append("")
        report.append(f"- Parse cache: {result.final_parse_cache_size}/{result.cache_sizes['parse']} ({result.final_parse_cache_size/result.cache_sizes['parse']:.1%} full)")
        report.append(f"- Merge cache: {result.final_merge_cache_size}/{result.cache_sizes['merge']} ({result.final_merge_cache_size/result.cache_sizes['merge']:.1%} full)")
        report.append("")
        report.append("---")
        report.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report))
    print(f"\n✅ Markdown report saved to: {output_path}")


def save_json_results(results: List[BenchmarkResult], output_path: Path):
    """Save raw results as JSON."""
    data = {
        "timestamp": time.time(),
        "date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "benchmarks": [asdict(r) for r in results]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"✅ JSON results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all benchmarks."""
    print("\n" + "="*80)
    print("PHASE 5 PERFORMANCE BENCHMARK SUITE")
    print("="*80)
    print("\nThis will benchmark compositional cache performance across")
    print("multiple cache size configurations.")
    print("")

    results = []

    for config in BENCHMARK_CONFIGS:
        try:
            result = await run_benchmark(config)
            results.append(result)
        except Exception as e:
            logger.error(f"Benchmark {config.name} failed: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n❌ All benchmarks failed!")
        return

    # Generate reports
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)

    save_json_results(results, Path("benchmarks/results/phase5_benchmark.json"))
    generate_markdown_report(results, Path("benchmarks/results/phase5_benchmark.md"))

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nRan {len(results)}/{len(BENCHMARK_CONFIGS)} benchmarks successfully")
    print("\nResults:")
    print("  - benchmarks/results/phase5_benchmark.json (raw data)")
    print("  - benchmarks/results/phase5_benchmark.md (formatted report)")
    print("\n✅ All benchmarks complete!")


if __name__ == "__main__":
    asyncio.run(main())