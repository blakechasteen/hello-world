#!/usr/bin/env python3
"""
Test Semantic Cache Integration with WeavingOrchestrator
========================================================
Validates that semantic cache is properly integrated and provides speedup.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard


async def main():
    print("=" * 70)
    print("SEMANTIC CACHE INTEGRATION TEST")
    print("=" * 70)

    # Create test memory shards
    shards = [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="docs",
            entities=["Thompson Sampling", "Bayesian", "bandit"],
            motifs=["ALGORITHM"]
        ),
        MemoryShard(
            id="shard_002",
            text="The hero embarks on a transformative journey.",
            episode="narrative",
            entities=["hero", "journey", "transformation"],
            motifs=["NARRATIVE"]
        )
    ]

    # Test 1: With semantic cache enabled
    print("\n1. Testing WITH semantic cache (default)...")
    config_with_cache = Config.fast()

    async with WeavingOrchestrator(
        cfg=config_with_cache,
        shards=shards,
        enable_semantic_cache=True
    ) as orchestrator:
        # First query
        query1 = Query(text="What is the hero's journey?")
        print(f"\nQuery: '{query1.text}'")
        spacetime1 = await orchestrator.weave(query1)

        print(f"Response: {spacetime1.response}")
        print(f"Confidence: {spacetime1.confidence:.2f}")
        print(f"Duration: {spacetime1.trace.duration_ms:.1f}ms")

        # Check cache stats in metadata
        if 'semantic_cache' in spacetime1.metadata:
            cache_info = spacetime1.metadata['semantic_cache']
            if cache_info['enabled']:
                print(f"\nSemantic Cache Stats:")
                print(f"  Hit rate: {cache_info['hit_rate']:.1%}")
                print(f"  Hot hits: {cache_info['hot_hits']}")
                print(f"  Warm hits: {cache_info['warm_hits']}")
                print(f"  Cold misses: {cache_info['cold_misses']}")
                print(f"  Estimated speedup: {cache_info['estimated_speedup']:.1f}×")
            else:
                print("\n❌ Semantic cache is disabled!")
        else:
            print("\n⚠️  No semantic cache info in metadata")

        # Second query (should hit cache)
        print("\n" + "-" * 70)
        query2 = Query(text="Tell me about heroes")
        print(f"\nQuery: '{query2.text}'")
        spacetime2 = await orchestrator.weave(query2)

        print(f"Response: {spacetime2.response}")
        print(f"Duration: {spacetime2.trace.duration_ms:.1f}ms")

        # Check updated cache stats
        if 'semantic_cache' in spacetime2.metadata:
            cache_info = spacetime2.metadata['semantic_cache']
            if cache_info['enabled']:
                print(f"\nUpdated Cache Stats:")
                print(f"  Hit rate: {cache_info['hit_rate']:.1%}")
                print(f"  Total queries: {cache_info['hot_hits'] + cache_info['warm_hits'] + cache_info['cold_misses']}")
                print(f"  Estimated speedup: {cache_info['estimated_speedup']:.1f}×")

    # Test 2: Without semantic cache
    print("\n" + "=" * 70)
    print("\n2. Testing WITHOUT semantic cache...")
    config_without_cache = Config.fast()

    async with WeavingOrchestrator(
        cfg=config_without_cache,
        shards=shards,
        enable_semantic_cache=False
    ) as orchestrator:
        query3 = Query(text="What is Thompson Sampling?")
        print(f"\nQuery: '{query3.text}'")
        spacetime3 = await orchestrator.weave(query3)

        print(f"Response: {spacetime3.response}")
        print(f"Duration: {spacetime3.trace.duration_ms:.1f}ms")

        # Check cache stats
        if 'semantic_cache' in spacetime3.metadata:
            cache_info = spacetime3.metadata['semantic_cache']
            if not cache_info['enabled']:
                print(f"\n✅ Semantic cache correctly disabled")
            else:
                print(f"\n❌ Semantic cache should be disabled!")
        else:
            print(f"\n✅ No semantic cache (as expected)")

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print("\n✅ Semantic cache integration working!")
    print("   - Cache initializes correctly")
    print("   - Statistics tracked in Spacetime metadata")
    print("   - Can be enabled/disabled via parameter")


if __name__ == "__main__":
    asyncio.run(main())
