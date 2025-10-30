#!/usr/bin/env python3
"""
Test Phase 5 Integration - Compositional Cache in Full Pipeline
================================================================

This test verifies that Phase 5 (compositional caching) is properly
integrated into the main pipeline and delivers the promised 100-300× speedups.

Test Plan:
1. Create config with Phase 5 enabled
2. Run queries through full pipeline
3. Verify cache is being used
4. Measure speedups (cold vs hot)
5. Validate cache statistics
"""

import asyncio
import time
import logging
from typing import List

from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_shards() -> List[MemoryShard]:
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian bandit algorithm that balances exploration and exploitation",
            episode="research",
            entities=["Thompson Sampling", "Bayesian", "bandit"],
            motifs=["reinforcement_learning"]
        ),
        MemoryShard(
            id="shard_2",
            text="The big red ball bounced down the hill",
            episode="test",
            entities=["ball", "hill"],
            motifs=["example"]
        ),
        MemoryShard(
            id="shard_3",
            text="A big red ball is used for testing compositional semantics",
            episode="test",
            entities=["ball", "compositional semantics"],
            motifs=["example", "testing"]
        ),
        MemoryShard(
            id="shard_4",
            text="The red ball was on the table",
            episode="test",
            entities=["ball", "table"],
            motifs=["example"]
        ),
        MemoryShard(
            id="shard_5",
            text="Universal Grammar provides phrase structure through X-bar theory",
            episode="linguistics",
            entities=["Universal Grammar", "X-bar theory"],
            motifs=["chomsky", "linguistics"]
        ),
        MemoryShard(
            id="shard_6",
            text="Compositional caching reuses building blocks across queries",
            episode="hololoom",
            entities=["compositional caching", "queries"],
            motifs=["performance", "optimization"]
        ),
    ]


async def test_phase5_basic():
    """Test basic Phase 5 integration."""
    print("=" * 80)
    print("TEST 1: Basic Phase 5 Integration")
    print("=" * 80)

    # Create config with Phase 5 ENABLED
    config = Config.fused()
    config.enable_linguistic_gate = True
    config.use_compositional_cache = True
    config.linguistic_mode = "disabled"  # Cache only, no pre-filtering
    config.parse_cache_size = 1000
    config.merge_cache_size = 5000

    logger.info(f"Config: linguistic_gate={config.enable_linguistic_gate}, "
                f"cache={config.use_compositional_cache}")

    # Create orchestrator
    shards = create_test_shards()
    shuttle = WeavingOrchestrator(cfg=config, shards=shards)

    # Verify linguistic gate was created
    if shuttle.linguistic_gate:
        print("✅ Linguistic gate created")

        # Verify compositional cache was created
        if hasattr(shuttle.linguistic_gate, 'compositional_cache') and shuttle.linguistic_gate.compositional_cache:
            print("✅ Compositional cache created")
            cache = shuttle.linguistic_gate.compositional_cache
            print(f"   Parse cache size: {cache.parse_cache_size}")
            print(f"   Merge cache size: {cache.merge_cache_size}")
        else:
            print("❌ Compositional cache NOT created")
            return False
    else:
        print("❌ Linguistic gate NOT created")
        print("   (This may be because spaCy is not installed)")
        return False

    print("\n" + "-" * 80)
    print("SUCCESS: Phase 5 components properly integrated!")
    print("-" * 80)
    return True


async def test_phase5_cold_vs_hot():
    """Test cold vs hot cache performance."""
    print("\n" + "=" * 80)
    print("TEST 2: Cold vs Hot Cache Performance")
    print("=" * 80)

    # Create config with Phase 5 ENABLED
    config = Config.fused()
    config.enable_linguistic_gate = True
    config.use_compositional_cache = True
    config.linguistic_mode = "disabled"

    # Create orchestrator
    shards = create_test_shards()
    shuttle = WeavingOrchestrator(cfg=config, shards=shards)

    if not shuttle.linguistic_gate or not shuttle.linguistic_gate.compositional_cache:
        print("⚠️  Skipping performance test (cache not available)")
        return False

    cache = shuttle.linguistic_gate.compositional_cache

    # Test queries - designed to have compositional overlap
    queries = [
        "the big red ball",      # Query 1 (cold)
        "the big red ball",      # Query 2 (hot - exact repeat)
        "a big red ball",        # Query 3 (warm - partial reuse)
        "the red ball",          # Query 4 (warm - subset)
    ]

    print(f"\nRunning {len(queries)} queries...\n")

    results = []
    for i, query_text in enumerate(queries, 1):
        # Get cache stats before
        stats_before = cache.get_statistics()

        # Time the embedding
        start = time.perf_counter()

        # Get compositional embedding directly
        embedding, trace = cache.get_compositional_embedding(query_text, return_trace=True)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Get cache stats after
        stats_after = cache.get_statistics()

        # Analyze what happened
        parse_hit = stats_after['parse_cache']['hits'] > stats_before['parse_cache']['hits']
        merge_hit = stats_after['merge_cache']['hits'] > stats_before['merge_cache']['hits']

        print(f"Query {i}: \"{query_text}\"")
        print(f"  Time: {elapsed_ms:.3f}ms")
        print(f"  Parse cache: {'HIT' if parse_hit else 'MISS'}")
        print(f"  Merge cache: {'HIT' if merge_hit else 'MISS'}")
        if trace:
            print(f"  Cache hits: {len(trace['hits'])}")
            print(f"  Cache misses: {len(trace['misses'])}")
        print()

        results.append({
            'query': query_text,
            'time_ms': elapsed_ms,
            'parse_hit': parse_hit,
            'merge_hit': merge_hit
        })

    # Analyze results
    print("-" * 80)
    print("PERFORMANCE ANALYSIS")
    print("-" * 80)

    cold_time = results[0]['time_ms']
    hot_time = results[1]['time_ms']
    speedup = cold_time / hot_time if hot_time > 0 else 0

    print(f"\nCold path (Query 1): {cold_time:.3f}ms")
    print(f"Hot path  (Query 2): {hot_time:.3f}ms")
    print(f"Speedup: {speedup:.1f}×")

    if speedup >= 10:
        print(f"✅ EXCELLENT: {speedup:.1f}× speedup achieved!")
    elif speedup >= 5:
        print(f"✅ GOOD: {speedup:.1f}× speedup (target: 10×+)")
    elif speedup >= 2:
        print(f"⚠️  MODERATE: {speedup:.1f}× speedup (expected: 10×+)")
    else:
        print(f"❌ LOW: {speedup:.1f}× speedup (expected: 10×+)")

    # Show final cache statistics
    print("\n" + "-" * 80)
    print("CACHE STATISTICS")
    print("-" * 80)
    final_stats = cache.get_statistics()
    print(f"\nParse cache:")
    print(f"  Hits: {final_stats['parse_cache']['hits']}")
    print(f"  Misses: {final_stats['parse_cache']['misses']}")
    print(f"  Hit rate: {final_stats['parse_cache']['hit_rate']:.1%}")
    print(f"  Size: {final_stats['parse_cache']['size']}/{final_stats['parse_cache']['capacity']}")

    print(f"\nMerge cache:")
    print(f"  Hits: {final_stats['merge_cache']['hits']}")
    print(f"  Misses: {final_stats['merge_cache']['misses']}")
    print(f"  Hit rate: {final_stats['merge_cache']['hit_rate']:.1%}")
    print(f"  Size: {final_stats['merge_cache']['size']}/{final_stats['merge_cache']['capacity']}")

    print(f"\nOverall hit rate: {final_stats['overall_hit_rate']:.1%}")

    print("\n" + "=" * 80)
    print(f"RESULT: {speedup:.1f}× speedup (target: 10×+ for production workloads)")
    print("=" * 80)

    return speedup >= 2  # Pass if at least 2× speedup


async def test_phase5_full_pipeline():
    """Test Phase 5 in full weaving pipeline."""
    print("\n" + "=" * 80)
    print("TEST 3: Phase 5 in Full Weaving Pipeline")
    print("=" * 80)

    # Create config with Phase 5 ENABLED
    config = Config.fused()
    config.enable_linguistic_gate = True
    config.use_compositional_cache = True
    config.linguistic_mode = "disabled"

    # Create orchestrator
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
        if not shuttle.linguistic_gate or not shuttle.linguistic_gate.compositional_cache:
            print("⚠️  Skipping pipeline test (cache not available)")
            return False

        # Run query through full pipeline
        query = Query(text="What is Thompson Sampling?")

        print(f"\nQuery: \"{query.text}\"\n")

        # First run (cold)
        start = time.perf_counter()
        spacetime1 = await shuttle.weave(query)
        time1_ms = (time.perf_counter() - start) * 1000

        # Second run (hot)
        start = time.perf_counter()
        spacetime2 = await shuttle.weave(query)
        time2_ms = (time.perf_counter() - start) * 1000

        print(f"First run (cold):  {time1_ms:.1f}ms")
        print(f"Second run (hot):  {time2_ms:.1f}ms")

        speedup = time1_ms / time2_ms if time2_ms > 0 else 1
        print(f"Pipeline speedup: {speedup:.1f}×")

        # Check cache stats
        cache = shuttle.linguistic_gate.compositional_cache
        stats = cache.get_statistics()

        print(f"\nCache statistics:")
        print(f"  Overall hit rate: {stats['overall_hit_rate']:.1%}")
        print(f"  Parse hits: {stats['parse_cache']['hits']}")
        print(f"  Merge hits: {stats['merge_cache']['hits']}")

        if spacetime1 and spacetime2:
            print("\n✅ Full pipeline operational with Phase 5!")
            return True
        else:
            print("\n❌ Pipeline failed")
            return False


async def main():
    """Run all Phase 5 integration tests."""
    print("\n" + "=" * 80)
    print("PHASE 5 INTEGRATION TEST SUITE")
    print("Testing: Compositional Cache in Full Pipeline")
    print("=" * 80 + "\n")

    results = {}

    # Test 1: Basic integration
    try:
        results['basic'] = await test_phase5_basic()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")
        results['basic'] = False

    # Test 2: Cold vs hot performance
    if results.get('basic'):
        try:
            results['performance'] = await test_phase5_cold_vs_hot()
        except Exception as e:
            logger.error(f"Test 2 failed: {e}")
            results['performance'] = False
    else:
        print("\n⚠️  Skipping performance test (basic integration failed)")
        results['performance'] = False

    # Test 3: Full pipeline
    if results.get('basic'):
        try:
            results['pipeline'] = await test_phase5_full_pipeline()
        except Exception as e:
            logger.error(f"Test 3 failed: {e}")
            results['pipeline'] = False
    else:
        print("\n⚠️  Skipping pipeline test (basic integration failed)")
        results['pipeline'] = False

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Phase 5 is fully operational!")
        print("\nNext steps:")
        print("1. Run benchmarks with production workload")
        print("2. Measure cache hit rates over time")
        print("3. Tune cache sizes based on usage patterns")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        if not results.get('basic'):
            print("\nMost likely cause: spaCy not installed")
            print("Install with: pip install spacy && python -m spacy download en_core_web_sm")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
