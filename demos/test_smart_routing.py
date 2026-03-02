#!/usr/bin/env python3
"""
Quick test of smart query routing integration.

Tests that TRIVIAL/SIMPLE queries use fast paths and COMPLEX queries use full pipeline.
"""

import asyncio
import time
from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard
from hololoom.weaving_orchestrator import WeavingOrchestrator


# Test queries for each complexity level
TEST_CASES = [
    ("hi", "TRIVIAL"),
    ("thanks", "TRIVIAL"),
    ("what is X?", "SIMPLE"),
    ("define Thompson Sampling", "SIMPLE"),
    ("Explain Thompson Sampling in detail with examples", "COMPLEX"),
    ("Analyze the tradeoffs between Thompson Sampling and epsilon-greedy", "RESEARCH"),
]

# Simple memory shards
SHARDS = [
    MemoryShard(
        id="shard_1",
        text="Thompson Sampling is a Bayesian approach to bandits.",
        entities=["Thompson Sampling", "Bayesian"],
        motifs=["sampling", "bandit"]
    ),
]


async def test_routing():
    """Test that routing works correctly."""
    print("\n" + "=" * 70)
    print("  SMART QUERY ROUTING TEST")
    print("=" * 70)

    config = Config.fast()

    async with WeavingOrchestrator(cfg=config, shards=SHARDS) as orchestrator:
        print(f"\nInitialized orchestrator with smart routing enabled")
        print(f"Classifier: {orchestrator.query_classifier}")
        print(f"Router: {orchestrator.fast_path_router}")

        results = []

        for query_text, expected_path in TEST_CASES:
            print(f"\n{'=' * 70}")
            print(f"Query: '{query_text}'")
            print(f"Expected: {expected_path}")

            start = time.perf_counter()
            query = Query(text=query_text)
            spacetime = await orchestrator.weave(query)
            elapsed_ms = (time.perf_counter() - start) * 1000

            fast_path = spacetime.metadata.get('fast_path', None)
            actual_path = fast_path if fast_path else "FULL_PIPELINE"

            results.append({
                'query': query_text,
                'expected': expected_path,
                'actual': actual_path,
                'latency_ms': elapsed_ms,
                'response': spacetime.response[:50] + "..." if len(spacetime.response) > 50 else spacetime.response
            })

            print(f"Actual:   {actual_path}")
            print(f"Latency:  {elapsed_ms:.1f}ms")
            print(f"Response: {results[-1]['response']}")

            # Verify expectation
            if expected_path in ("TRIVIAL", "SIMPLE"):
                if fast_path:
                    print("✅ PASS - Used fast path")
                else:
                    print("❌ FAIL - Should have used fast path!")
            else:
                if not fast_path:
                    print("✅ PASS - Used full pipeline")
                else:
                    print("❌ FAIL - Should have used full pipeline!")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(f"\n{'Query':<50} {'Path':<15} {'Latency':<10}")
        print("-" * 70)

        for r in results:
            print(f"{r['query']:<50} {r['actual']:<15} {r['latency_ms']:>8.1f}ms")

        # Routing statistics
        print("\n" + "-" * 70)
        print("ROUTING STATISTICS")
        print("-" * 70)

        stats = orchestrator.fast_path_router.get_stats()
        print(f"\nTrivial:  {stats['trivial']} queries")
        print(f"Simple:   {stats['simple']} queries")
        print(f"Complex:  {stats['complex']} queries")
        print(f"Total:    {stats['total']} queries")
        print(f"Fast path: {stats['fast_path_percentage']:.1f}%")

        print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    asyncio.run(test_routing())
