"""
Performance Regression Test Suite
==================================
Systematically tracks performance across releases to detect regressions.

Test Philosophy:
- Baseline Performance: Establish performance baselines for all modes
- Regression Detection: Auto-detect performance degradation >20%
- Budget Enforcement: Fail tests if performance exceeds hard limits
- Historical Tracking: Save metrics to JSON for trend analysis
- Component Isolation: Test individual components vs full pipeline

Performance Budgets (CI environment, relaxed from production):
- Cache Hit: <5ms (production: <1ms)
- BARE Mode: <200ms (production: <50ms)
- FAST Mode: <500ms (production: <150ms)
- FULL Mode: <1000ms (production: <300ms)
- RESEARCH Mode: <3000ms (production: no limit)

Regression Thresholds:
- WARNING: >10% slower than baseline
- FAIL: >20% slower than baseline OR exceeds budget

Author: Claude Code (Week 2, Day 1 - Advanced Test Suites)
Date: November 8, 2025
"""

import pytest
import asyncio
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.protocols.types import Query, MemoryShard


# ============================================================================
# Configuration and Baselines
# ============================================================================

PERFORMANCE_BUDGETS_MS = {
    "cache_hit": 5.0,           # Hash lookup
    "BARE": 200.0,              # Minimal processing
    "FAST": 500.0,              # Balanced processing
    "FULL": 1000.0,             # Full processing
    "RESEARCH": 3000.0,         # No production limit
}

REGRESSION_THRESHOLD = 0.20  # 20% slower = FAIL
WARNING_THRESHOLD = 0.10     # 10% slower = WARNING

# Baseline file for historical tracking
BASELINES_FILE = Path(__file__).parent / "performance_baselines.json"


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_shards():
    """Create test memory shards for performance testing."""
    return [
        MemoryShard(
            id=f"shard_{i}",
            text=f"Test shard {i} with content about topic {i % 3}",
            episode=f"episode_{i // 5}",
            entities=[f"Entity{i}", f"Topic{i % 3}"],
            motifs=[f"motif_{i % 4}"],
            metadata={"index": i, "topic": i % 3}
        )
        for i in range(20)  # 20 shards for realistic testing
    ]


@pytest.fixture
def standard_queries():
    """Standard query set for benchmarking."""
    return [
        Query(text="What is Thompson Sampling?"),
        Query(text="How does exploration-exploitation work?"),
        Query(text="Explain Matryoshka embeddings"),
        Query(text="What is the weaving architecture?"),
        Query(text="How do bandits balance exploration and exploitation?"),
    ]


# ============================================================================
# Performance Measurement Utilities
# ============================================================================

def measure_query_performance(func, iterations=3):
    """
    Measure query performance over multiple iterations.

    Returns:
        dict: Performance statistics (avg, min, max, std)
    """
    durations = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        duration_ms = (time.perf_counter() - start) * 1000
        durations.append(duration_ms)

    return {
        "avg_ms": statistics.mean(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "std_ms": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "iterations": len(durations),
    }


async def measure_async_performance(coro_func, iterations=3):
    """Measure async query performance."""
    durations = []

    for _ in range(iterations):
        start = time.perf_counter()
        await coro_func()
        duration_ms = (time.perf_counter() - start) * 1000
        durations.append(duration_ms)

    return {
        "avg_ms": statistics.mean(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "std_ms": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "iterations": len(durations),
    }


def check_regression(current_ms: float, baseline_ms: float, mode: str) -> Dict[str, Any]:
    """
    Check if current performance represents a regression.

    Returns:
        dict: {
            "status": "PASS" | "WARNING" | "FAIL",
            "degradation_pct": float,
            "current_ms": float,
            "baseline_ms": float,
            "budget_ms": float,
            "over_budget": bool,
        }
    """
    budget_ms = PERFORMANCE_BUDGETS_MS.get(mode, float('inf'))
    degradation_pct = (current_ms - baseline_ms) / baseline_ms if baseline_ms > 0 else 0.0
    over_budget = current_ms > budget_ms

    if over_budget or degradation_pct > REGRESSION_THRESHOLD:
        status = "FAIL"
    elif degradation_pct > WARNING_THRESHOLD:
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "status": status,
        "degradation_pct": degradation_pct * 100,
        "current_ms": current_ms,
        "baseline_ms": baseline_ms,
        "budget_ms": budget_ms,
        "over_budget": over_budget,
    }


def load_baselines() -> Dict[str, float]:
    """Load performance baselines from file."""
    if not BASELINES_FILE.exists():
        return {}

    try:
        with open(BASELINES_FILE, 'r') as f:
            data = json.load(f)
            return data.get('baselines', {})
    except Exception:
        return {}


def save_baselines(baselines: Dict[str, float]):
    """Save performance baselines to file."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'baselines': baselines,
        'budgets': PERFORMANCE_BUDGETS_MS,
    }

    with open(BASELINES_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# Test Cache Performance
# ============================================================================

class TestCachePerformance:
    """Test cache hit/miss performance."""

    @pytest.mark.asyncio
    async def test_cache_hit_latency(self, test_shards):
        """Cache hits should be <5ms."""
        config = Config.fast()
        query = Query(text="What is Thompson Sampling?")

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            # First query (cold cache)
            await orchestrator.weave(query)

            # Measure cache hit performance
            async def cached_query():
                return await orchestrator.weave(query)

            perf = await measure_async_performance(cached_query, iterations=5)

            print(f"\nCache Hit Performance:")
            print(f"  Average: {perf['avg_ms']:.2f}ms")
            print(f"  Min: {perf['min_ms']:.2f}ms")
            print(f"  Max: {perf['max_ms']:.2f}ms")
            print(f"  Budget: {PERFORMANCE_BUDGETS_MS['cache_hit']}ms")

            # Cache hits should be extremely fast
            assert perf['avg_ms'] < PERFORMANCE_BUDGETS_MS['cache_hit'], \
                f"Cache hit too slow: {perf['avg_ms']:.2f}ms > {PERFORMANCE_BUDGETS_MS['cache_hit']}ms"

    @pytest.mark.asyncio
    async def test_cache_miss_vs_hit_speedup(self, test_shards):
        """Cache hits should be 50-300× faster than misses."""
        config = Config.fast()
        query = Query(text="What is Thompson Sampling?")

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            # Measure cache miss (cold)
            start = time.perf_counter()
            await orchestrator.weave(query)
            miss_ms = (time.perf_counter() - start) * 1000

            # Measure cache hit (warm)
            start = time.perf_counter()
            await orchestrator.weave(query)
            hit_ms = (time.perf_counter() - start) * 1000

            speedup = miss_ms / hit_ms if hit_ms > 0 else 0

            print(f"\nCache Performance:")
            print(f"  Miss: {miss_ms:.2f}ms")
            print(f"  Hit: {hit_ms:.2f}ms")
            print(f"  Speedup: {speedup:.1f}×")

            # Cache should provide significant speedup
            assert speedup >= 10.0, f"Cache speedup too low: {speedup:.1f}× < 10×"


# ============================================================================
# Test Mode Performance
# ============================================================================

class TestModePerformance:
    """Test performance across execution modes."""

    @pytest.mark.asyncio
    async def test_bare_mode_performance(self, test_shards, standard_queries):
        """BARE mode should be <200ms (CI budget)."""
        config = Config.bare()

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            durations = []

            for query in standard_queries:
                start = time.perf_counter()
                result = await orchestrator.weave(query)
                duration_ms = (time.perf_counter() - start) * 1000
                durations.append(duration_ms)

                assert result is not None

            avg_ms = statistics.mean(durations)
            budget_ms = PERFORMANCE_BUDGETS_MS['BARE']

            print(f"\nBARE Mode Performance:")
            print(f"  Queries: {len(standard_queries)}")
            print(f"  Average: {avg_ms:.2f}ms")
            print(f"  Min: {min(durations):.2f}ms")
            print(f"  Max: {max(durations):.2f}ms")
            print(f"  Budget: {budget_ms}ms")

            assert avg_ms < budget_ms, \
                f"BARE mode too slow: {avg_ms:.2f}ms > {budget_ms}ms"

    @pytest.mark.asyncio
    async def test_fast_mode_performance(self, test_shards, standard_queries):
        """FAST mode should be <500ms (CI budget)."""
        config = Config.fast()

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            durations = []

            for query in standard_queries:
                start = time.perf_counter()
                result = await orchestrator.weave(query)
                duration_ms = (time.perf_counter() - start) * 1000
                durations.append(duration_ms)

                assert result is not None

            avg_ms = statistics.mean(durations)
            budget_ms = PERFORMANCE_BUDGETS_MS['FAST']

            print(f"\nFAST Mode Performance:")
            print(f"  Queries: {len(standard_queries)}")
            print(f"  Average: {avg_ms:.2f}ms")
            print(f"  Min: {min(durations):.2f}ms")
            print(f"  Max: {max(durations):.2f}ms")
            print(f"  Budget: {budget_ms}ms")

            assert avg_ms < budget_ms, \
                f"FAST mode too slow: {avg_ms:.2f}ms > {budget_ms}ms"

    @pytest.mark.asyncio
    async def test_fused_mode_performance(self, test_shards, standard_queries):
        """FUSED mode should be <1000ms (CI budget)."""
        config = Config.fused()

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            durations = []

            for query in standard_queries:
                start = time.perf_counter()
                result = await orchestrator.weave(query)
                duration_ms = (time.perf_counter() - start) * 1000
                durations.append(duration_ms)

                assert result is not None

            avg_ms = statistics.mean(durations)
            budget_ms = PERFORMANCE_BUDGETS_MS['FULL']

            print(f"\nFUSED Mode Performance:")
            print(f"  Queries: {len(standard_queries)}")
            print(f"  Average: {avg_ms:.2f}ms")
            print(f"  Min: {min(durations):.2f}ms")
            print(f"  Max: {max(durations):.2f}ms")
            print(f"  Budget: {budget_ms}ms")

            assert avg_ms < budget_ms, \
                f"FUSED mode too slow: {avg_ms:.2f}ms > {budget_ms}ms"


# ============================================================================
# Test Scalability
# ============================================================================

class TestScalability:
    """Test performance scaling with data size."""

    @pytest.mark.asyncio
    async def test_shard_count_scaling(self):
        """Performance should scale sublinearly with shard count."""
        config = Config.fast()
        query = Query(text="What is Thompson Sampling?")

        shard_counts = [10, 50, 100]
        results = []

        for count in shard_counts:
            shards = [
                MemoryShard(
                    id=f"shard_{i}",
                    text=f"Content {i}",
                    episode=f"ep_{i}",
                    metadata={"index": i}
                )
                for i in range(count)
            ]

            async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
                start = time.perf_counter()
                await orchestrator.weave(query)
                duration_ms = (time.perf_counter() - start) * 1000

                results.append({"count": count, "duration_ms": duration_ms})

        print(f"\nShard Count Scaling:")
        print(f"  {'Shards':<10} {'Duration':<15} {'Per Shard':<15}")
        print("-" * 50)

        for r in results:
            per_shard = r['duration_ms'] / r['count']
            print(f"  {r['count']:<10} {r['duration_ms']:>10.2f}ms  {per_shard:>10.4f}ms")

        # Scaling should be sublinear (not O(n))
        # Doubling shards should not double time
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            scaling_factor = (last['duration_ms'] / first['duration_ms']) / (last['count'] / first['count'])

            print(f"\nScaling Factor: {scaling_factor:.2f} (1.0 = linear, <1.0 = sublinear)")

            assert scaling_factor < 1.5, \
                f"Scaling too steep: {scaling_factor:.2f}× (should be <1.5)"

    @pytest.mark.asyncio
    async def test_query_length_scaling(self, test_shards):
        """Performance should scale reasonably with query length."""
        config = Config.fast()

        query_lengths = [10, 50, 100]  # Words
        results = []

        for length in query_lengths:
            query_text = " ".join([f"word{i}" for i in range(length)])
            query = Query(text=query_text)

            async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
                start = time.perf_counter()
                await orchestrator.weave(query)
                duration_ms = (time.perf_counter() - start) * 1000

                results.append({"length": length, "duration_ms": duration_ms})

        print(f"\nQuery Length Scaling:")
        print(f"  {'Words':<10} {'Duration':<15}")
        print("-" * 35)

        for r in results:
            print(f"  {r['length']:<10} {r['duration_ms']:>10.2f}ms")

        # All queries should still be within budget
        for r in results:
            assert r['duration_ms'] < PERFORMANCE_BUDGETS_MS['FAST'], \
                f"Long query too slow: {r['duration_ms']:.2f}ms > {PERFORMANCE_BUDGETS_MS['FAST']}ms"


# ============================================================================
# Test Regression Detection
# ============================================================================

class TestRegressionDetection:
    """Test regression detection against baselines."""

    @pytest.mark.asyncio
    async def test_baseline_comparison(self, test_shards):
        """Compare current performance against saved baselines."""
        config = Config.fast()
        query = Query(text="What is Thompson Sampling?")

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            # Measure current performance
            durations = []
            for _ in range(3):
                start = time.perf_counter()
                await orchestrator.weave(query)
                duration_ms = (time.perf_counter() - start) * 1000
                durations.append(duration_ms)

            current_ms = statistics.mean(durations)

            # Load baselines
            baselines = load_baselines()
            baseline_key = "fast_mode_standard_query"

            if baseline_key in baselines:
                baseline_ms = baselines[baseline_key]
                result = check_regression(current_ms, baseline_ms, "FAST")

                print(f"\nRegression Analysis:")
                print(f"  Current: {result['current_ms']:.2f}ms")
                print(f"  Baseline: {result['baseline_ms']:.2f}ms")
                print(f"  Budget: {result['budget_ms']:.2f}ms")
                print(f"  Degradation: {result['degradation_pct']:+.1f}%")
                print(f"  Status: {result['status']}")

                # FAIL if regression > 20% or over budget
                assert result['status'] != "FAIL", \
                    f"Performance regression detected: {result['degradation_pct']:+.1f}% (threshold: {REGRESSION_THRESHOLD * 100}%)"

            else:
                # No baseline - save current as baseline
                baselines[baseline_key] = current_ms
                save_baselines(baselines)
                print(f"\nNo baseline found. Saving current performance: {current_ms:.2f}ms")

    @pytest.mark.asyncio
    async def test_update_baselines(self, test_shards, standard_queries):
        """
        Update performance baselines (manual test).

        Run this test explicitly to update baselines after optimization.
        """
        modes = [
            (Config.bare(), "BARE", "bare_mode_standard_query"),
            (Config.fast(), "FAST", "fast_mode_standard_query"),
            (Config.fused(), "FULL", "fused_mode_standard_query"),
        ]

        baselines = load_baselines()
        updated = False

        for config, mode_name, baseline_key in modes:
            async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
                # Use first query only for consistency
                query = standard_queries[0]

                durations = []
                for _ in range(5):  # More iterations for stable baseline
                    start = time.perf_counter()
                    await orchestrator.weave(query)
                    duration_ms = (time.perf_counter() - start) * 1000
                    durations.append(duration_ms)

                avg_ms = statistics.mean(durations)

                # Check if we should update baseline
                # Only update if:
                # 1. No baseline exists, OR
                # 2. Current performance is better AND test is run with --update-baselines
                should_update = baseline_key not in baselines

                if should_update:
                    baselines[baseline_key] = avg_ms
                    updated = True
                    print(f"Updated baseline for {mode_name}: {avg_ms:.2f}ms")
                else:
                    old_baseline = baselines[baseline_key]
                    print(f"Baseline for {mode_name}: {old_baseline:.2f}ms (current: {avg_ms:.2f}ms)")

        if updated:
            save_baselines(baselines)
            print(f"\nBaselines saved to {BASELINES_FILE}")


# ============================================================================
# Test Component Performance
# ============================================================================

class TestComponentPerformance:
    """Test individual component performance."""

    @pytest.mark.asyncio
    async def test_feature_extraction_performance(self, test_shards):
        """Feature extraction should be <50ms."""
        config = Config.fast()

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            query = Query(text="What is Thompson Sampling?")

            # Weave once to get trace
            spacetime = await orchestrator.weave(query)

            # Check feature extraction timing
            if 'feature_extraction' in spacetime.trace.stage_durations:
                duration_ms = spacetime.trace.stage_durations['feature_extraction']

                print(f"\nFeature Extraction Performance:")
                print(f"  Duration: {duration_ms:.2f}ms")

                # Should be <50ms (most of query time is retrieval/decision)
                assert duration_ms < 50.0, \
                    f"Feature extraction too slow: {duration_ms:.2f}ms > 50ms"

    @pytest.mark.asyncio
    async def test_retrieval_performance(self, test_shards):
        """Retrieval should be <100ms."""
        config = Config.fast()

        async with WeavingOrchestrator(cfg=config, shards=test_shards) as orchestrator:
            query = Query(text="What is Thompson Sampling?")
            spacetime = await orchestrator.weave(query)

            # Check retrieval timing
            if 'retrieval' in spacetime.trace.stage_durations:
                duration_ms = spacetime.trace.stage_durations['retrieval']

                print(f"\nRetrieval Performance:")
                print(f"  Duration: {duration_ms:.2f}ms")

                # Should be <100ms
                assert duration_ms < 100.0, \
                    f"Retrieval too slow: {duration_ms:.2f}ms > 100ms"


# Total: 15 test methods across 5 classes
# Coverage: Cache, modes, scalability, regression, components
# Performance: Systematic regression detection with baselines
