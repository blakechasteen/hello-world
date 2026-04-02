#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script: Yarn Graph Integration
====================================
Verify that the WeavingOrchestrator works with KG (Yarn Graph) instances.

This script tests:
1. KG.select_threads() method
2. WeavingOrchestrator with yarn_graph parameter
3. Backward compatibility with shards parameter
4. Complete weaving cycle with KG
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
from hololoom.memory.graph import KG, KGEdge
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard
from hololoom.chrono.trigger import TemporalWindow


async def create_test_kg() -> KG:
    """Create a test KG with memory nodes."""
    kg = KG()

    # Add memory nodes
    from hololoom.memory.protocol import Memory

    memories = [
        Memory(
            id="mem_ts_def",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                 "It maintains probability distributions over each arm's reward.",
            timestamp=datetime.now() - timedelta(days=1),
            context={'entities': ['Thompson Sampling', 'Bayesian', 'bandit']},
            metadata={'episode': 'thompson_basics', 'motifs': ['definition']}
        ),
        Memory(
            id="mem_ts_algo",
            text="The Thompson Sampling algorithm samples from each arm's posterior distribution "
                 "and selects the arm with the highest sample. After observing the reward, "
                 "it updates the posterior using Bayesian inference.",
            timestamp=datetime.now() - timedelta(hours=12),
            context={'entities': ['Thompson Sampling', 'posterior', 'Bayesian']},
            metadata={'episode': 'thompson_algo', 'motifs': ['algorithm']}
        ),
        Memory(
            id="mem_ts_vs_ucb",
            text="Thompson Sampling often outperforms UCB (Upper Confidence Bound) in practice, "
                 "especially when the reward distributions are non-stationary. UCB uses deterministic "
                 "selection based on confidence bounds, while Thompson Sampling is probabilistic.",
            timestamp=datetime.now() - timedelta(hours=6),
            context={'entities': ['Thompson Sampling', 'UCB', 'bandit']},
            metadata={'episode': 'thompson_comparison', 'motifs': ['comparison']}
        ),
    ]

    # Store memories in KG (uses async store method)
    for memory in memories:
        await kg.store(memory)

    return kg


def create_test_shards() -> list:
    """Create test shards for backward compatibility testing."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling balances exploration and exploitation.",
            episode="basics",
            entities=["Thompson Sampling"],
            motifs=["exploration"]
        ),
        MemoryShard(
            id="shard_2",
            text="Bayesian inference updates posterior distributions based on observed rewards.",
            episode="basics",
            entities=["Bayesian"],
            motifs=["inference"]
        ),
    ]


async def test_kg_select_threads():
    """Test KG.select_threads() method."""
    print("=" * 80)
    print("TEST 1: KG.select_threads() Method")
    print("=" * 80)

    kg = await create_test_kg()

    # Create temporal window (last 2 days)
    temporal_window = TemporalWindow(
        start=datetime.now() - timedelta(days=2),
        end=datetime.now(),
        max_age=timedelta(days=2),
        recency_bias=0.5
    )

    # Create query
    query = Query(text="What is Thompson Sampling?")

    # Select threads
    threads = kg.select_threads(temporal_window, query)

    print(f"\nQuery: {query.text}")
    print(f"Temporal window: {temporal_window.start} to {temporal_window.end}")
    print(f"Selected {len(threads)} threads:\n")

    for i, thread in enumerate(threads, 1):
        print(f"{i}. ID: {thread.id}")
        print(f"   Text: {thread.text[:100]}...")
        print(f"   Entities: {thread.entities}")
        print(f"   Episode: {thread.episode}")
        print()

    assert len(threads) > 0, "Should select at least one thread"
    assert all(isinstance(t, MemoryShard) for t in threads), "All threads should be MemoryShards"

    print("[PASS] KG.select_threads() works correctly\n")
    return True


async def test_orchestrator_with_yarn_graph():
    """Test WeavingOrchestrator with yarn_graph parameter."""
    print("=" * 80)
    print("TEST 2: WeavingOrchestrator with yarn_graph Parameter")
    print("=" * 80)

    kg = await create_test_kg()
    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing

    # Create orchestrator with yarn_graph
    async with WeavingOrchestrator(cfg=config, yarn_graph=kg) as orchestrator:
        print(f"\nOrchestrator created with Yarn Graph")
        print(f"Yarn Graph nodes: {kg.G.number_of_nodes()}")
        print(f"Yarn Graph edges: {kg.G.number_of_edges()}")

        # Test weaving
        query = Query(text="How does Thompson Sampling work?")
        print(f"\nWeaving query: {query.text}")

        spacetime = await orchestrator.weave(query)

        print(f"\nSpacetime created:")
        print(f"  Tool used: {spacetime.metadata.get('tool_used', 'unknown')}")
        print(f"  Confidence: {spacetime.confidence:.3f}")
        print(f"  Response preview: {spacetime.response[:150]}...")

        assert spacetime is not None, "Should return Spacetime"
        assert spacetime.confidence > 0, "Should have confidence > 0"

    print("\n[PASS] WeavingOrchestrator with yarn_graph works correctly\n")
    return True


async def test_backward_compatibility():
    """Test backward compatibility with shards parameter."""
    print("=" * 80)
    print("TEST 3: Backward Compatibility with shards Parameter")
    print("=" * 80)

    shards = create_test_shards()
    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing

    # Should show deprecation warning
    print("\nCreating orchestrator with shards (should show deprecation warning)...")

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        print(f"Orchestrator created with {len(shards)} shards")

        # Test weaving
        query = Query(text="What is Bayesian inference?")
        print(f"\nWeaving query: {query.text}")

        spacetime = await orchestrator.weave(query)

        print(f"\nSpacetime created:")
        print(f"  Tool used: {spacetime.metadata.get('tool_used', 'unknown')}")
        print(f"  Confidence: {spacetime.confidence:.3f}")

        assert spacetime is not None, "Should return Spacetime"

    print("\n[PASS] Backward compatibility maintained\n")
    return True


async def test_priority_order():
    """Test that memory > yarn_graph > shards priority is respected."""
    print("=" * 80)
    print("TEST 4: Priority Order (memory > yarn_graph > shards)")
    print("=" * 80)

    kg = await create_test_kg()
    shards = create_test_shards()
    config = Config.fast()
    config.enable_safety_guardrails = False  # Disable approval for automated testing

    # Test 1: yarn_graph preferred over shards
    print("\nTest 4a: yarn_graph preferred over shards")

    async with WeavingOrchestrator(cfg=config, yarn_graph=kg, shards=shards) as orch:
        # Should use yarn_graph, not shards
        assert orch.yarn_graph is kg, "Should use KG instance, not shards"
        print("[PASS] yarn_graph takes priority over shards")

    # Test 2: shards used when no yarn_graph
    print("\nTest 4b: shards used when no yarn_graph")
    async with WeavingOrchestrator(cfg=config, shards=shards) as orch:
        # Should use YarnGraph wrapper
        assert hasattr(orch.yarn_graph, 'shards'), "Should use YarnGraph wrapper for shards"
        print("[PASS] shards used as fallback")

    print("\n[PASS] Priority order respected\n")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("YARN GRAPH INTEGRATION TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("KG.select_threads()", test_kg_select_threads),
        ("Orchestrator with yarn_graph", test_orchestrator_with_yarn_graph),
        ("Backward compatibility", test_backward_compatibility),
        ("Priority order", test_priority_order),
    ]

    results = []

    for name, test_fn in tests:
        try:
            result = await test_fn()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\nX {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, status in results:
        symbol = "PASS" if status == "PASS" else "FAIL"
        print(f"[{symbol}] {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Yarn Graph integration complete.")
        return 0
    else:
        print(f"\n[WARN] {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
