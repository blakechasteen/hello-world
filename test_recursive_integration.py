#!/usr/bin/env python3
"""
Test Recursive Learning Integration
=====================================
Verify that recursive learning integrates correctly into WeavingOrchestrator.
"""

import asyncio
from hololoom.config import Config
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.documentation.types import Query, MemoryShard


async def test_integrated_learning():
    """Test integrated recursive learning (new API)"""
    print("\n" + "="*80)
    print("TEST 1: Integrated Recursive Learning (New API)")
    print("="*80)

    # Create sample shards
    shards = [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="docs",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["ALGORITHM", "OPTIMIZATION"]
        ),
        MemoryShard(
            id="shard_002",
            text="The algorithm balances exploration and exploitation by sampling from posterior distributions.",
            episode="docs",
            entities=["exploration", "exploitation", "posterior"],
            motifs=["ALGORITHM", "PROBABILITY"]
        ),
    ]

    # Create config with recursive learning enabled
    config = Config.fast()
    config.enable_recursive_learning = True
    config.recursive_learning_refinement_threshold = 0.75
    config.recursive_learning_max_iterations = 2
    config.enable_semantic_cache = False  # Disable for speed

    print(f"\nConfig:")
    print(f"  enable_recursive_learning: {config.enable_recursive_learning}")
    print(f"  refinement_threshold: {config.recursive_learning_refinement_threshold}")
    print(f"  max_iterations: {config.recursive_learning_max_iterations}")

    # Test with integrated API
    async with WeavingOrchestrator(cfg=config, shards=shards) as orch:
        print(f"\nProcessing queries with integrated learning...")

        # Query 1: High confidence (should learn pattern)
        query1 = Query(text="What is Thompson Sampling?")
        spacetime1 = await orch.weave(query1)

        print(f"\nQuery 1: {query1.text}")
        print(f"  Confidence: {spacetime1.confidence:.2f}")
        print(f"  Tool: {spacetime1.tool_used}")

        # Query 2: Another query (to trigger learning)
        query2 = Query(text="How does it balance exploration?")
        spacetime2 = await orch.weave(query2)

        print(f"\nQuery 2: {query2.text}")
        print(f"  Confidence: {spacetime2.confidence:.2f}")
        print(f"  Tool: {spacetime2.tool_used}")

        # Get learning statistics
        stats = orch.get_recursive_learning_stats()

        if stats:
            print(f"\n{'='*80}")
            print("LEARNING STATISTICS")
            print(f"{'='*80}")
            print(f"Queries processed: {stats['queries_processed']}")
            print(f"Average confidence: {stats['avg_confidence']:.2f}")

            print(f"\nThompson Priors:")
            for tool, priors in stats['thompson_priors'].items():
                print(f"  {tool}:")
                print(f"    Expected reward: {priors['expected_reward']:.3f}")
                print(f"    Uncertainty: {priors['uncertainty']:.3f}")

            print(f"\nPolicy Weights:")
            for adapter, weights in stats['policy_weights'].items():
                print(f"  {adapter}:")
                print(f"    Weight: {weights['weight']:.3f}")
                print(f"    Success rate: {weights['success_rate']:.2%}")
                print(f"    Total uses: {weights['total_uses']}")

            print(f"\n✅ Integrated learning working!")
        else:
            print(f"\n❌ Learning stats not available")

    print(f"\n{'='*80}")
    print("TEST 1 COMPLETE")
    print(f"{'='*80}\n")


async def test_backward_compatibility():
    """Test that orchestrator still works with learning disabled"""
    print("\n" + "="*80)
    print("TEST 2: Backward Compatibility (Learning Disabled)")
    print("="*80)

    # Create sample shards
    shards = [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach.",
            episode="docs",
            entities=["Thompson Sampling"],
            motifs=["ALGORITHM"]
        ),
    ]

    # Create config with learning DISABLED (default)
    config = Config.fast()
    config.enable_semantic_cache = False
    assert config.enable_recursive_learning == False  # Default

    print(f"\nConfig:")
    print(f"  enable_recursive_learning: {config.enable_recursive_learning}")

    # Test without learning
    async with WeavingOrchestrator(cfg=config, shards=shards) as orch:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orch.weave(query)

        print(f"\nQuery: {query.text}")
        print(f"  Confidence: {spacetime.confidence:.2f}")
        print(f"  Tool: {spacetime.tool_used}")

        # Get learning statistics (should be None)
        stats = orch.get_recursive_learning_stats()

        if stats is None:
            print(f"\n✅ Learning correctly disabled (stats=None)")
        else:
            print(f"\n❌ Unexpected: Learning stats available when disabled")

    print(f"\n{'='*80}")
    print("TEST 2 COMPLETE")
    print(f"{'='*80}\n")


async def test_performance_overhead():
    """Test performance overhead of recursive learning"""
    print("\n" + "="*80)
    print("TEST 3: Performance Overhead")
    print("="*80)

    import time

    # Create sample shards
    shards = [
        MemoryShard(
            id=f"shard_{i:03d}",
            text=f"Sample memory shard {i}",
            episode="test",
            entities=[],
            motifs=[]
        )
        for i in range(10)
    ]

    query = Query(text="Test query")

    # Test WITHOUT learning
    config_no_learning = Config.bare()
    config_no_learning.enable_recursive_learning = False
    config_no_learning.enable_semantic_cache = False

    times_no_learning = []
    async with WeavingOrchestrator(cfg=config_no_learning, shards=shards) as orch:
        for _ in range(5):
            start = time.perf_counter()
            await orch.weave(query)
            duration = (time.perf_counter() - start) * 1000
            times_no_learning.append(duration)

    avg_no_learning = sum(times_no_learning) / len(times_no_learning)

    # Test WITH learning
    config_with_learning = Config.bare()
    config_with_learning.enable_recursive_learning = True
    config_with_learning.enable_semantic_cache = False

    times_with_learning = []
    async with WeavingOrchestrator(cfg=config_with_learning, shards=shards) as orch:
        for _ in range(5):
            start = time.perf_counter()
            await orch.weave(query)
            duration = (time.perf_counter() - start) * 1000
            times_with_learning.append(duration)

    avg_with_learning = sum(times_with_learning) / len(times_with_learning)

    overhead = avg_with_learning - avg_no_learning
    overhead_pct = (overhead / avg_no_learning) * 100 if avg_no_learning > 0 else 0

    print(f"\nPerformance Results (5 queries each):")
    print(f"  WITHOUT learning: {avg_no_learning:.1f}ms avg")
    print(f"  WITH learning:    {avg_with_learning:.1f}ms avg")
    print(f"  Overhead:         {overhead:.1f}ms ({overhead_pct:.1f}%)")

    if overhead < 5.0:  # Target: <5ms overhead
        print(f"\n✅ Performance overhead within target (<5ms)")
    else:
        print(f"\n⚠️  Performance overhead higher than target (>5ms)")

    print(f"\n{'='*80}")
    print("TEST 3 COMPLETE")
    print(f"{'='*80}\n")


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("RECURSIVE LEARNING INTEGRATION TEST SUITE")
    print("="*80)

    try:
        await test_integrated_learning()
        await test_backward_compatibility()
        await test_performance_overhead()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETE")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
