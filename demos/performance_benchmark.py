#!/usr/bin/env python3
"""
Performance Benchmark - Test cache and optimization improvements
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query, MemoryShard


def create_test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(
            id="test1",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="test",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["exploration", "exploitation"],
            metadata={}
        ),
        MemoryShard(
            id="test2",
            text="Reinforcement learning agents learn by trial and error.",
            episode="test",
            entities=["reinforcement learning", "agents"],
            motifs=["learning", "trial and error"],
            metadata={}
        ),
        MemoryShard(
            id="test3",
            text="Knowledge graphs store information as entities and relationships.",
            episode="test",
            entities=["knowledge graphs", "entities", "relationships"],
            motifs=["storage", "structure"],
            metadata={}
        ),
    ]


async def benchmark_query_cache():
    """Test query caching performance."""
    print("=" * 80)
    print("BENCHMARK: Query Caching")
    print("=" * 80)

    config = Config.bare()
    shards = create_test_shards()

    async with WeavingShuttle(cfg=config, shards=shards, enable_reflection=False) as shuttle:
        query = Query(text="What is Thompson Sampling?")

        # First query - cache miss
        print("\n[Test 1] First query (cache miss)")
        start = time.time()
        result1 = await shuttle.weave(query)
        duration1 = (time.time() - start) * 1000
        print(f"  Duration: {duration1:.1f}ms")
        print(f"  Tool: {result1.tool_used}")

        # Second query - cache hit
        print("\n[Test 2] Second query (cache hit)")
        start = time.time()
        result2 = await shuttle.weave(query)
        duration2 = (time.time() - start) * 1000
        print(f"  Duration: {duration2:.1f}ms")
        print(f"  Tool: {result2.tool_used}")

        # Third query - cache hit
        print("\n[Test 3] Third query (cache hit)")
        start = time.time()
        result3 = await shuttle.weave(query)
        duration3 = (time.time() - start) * 1000
        print(f"  Duration: {duration3:.1f}ms")
        print(f"  Tool: {result3.tool_used}")

        # Results
        speedup2 = duration1 / duration2 if duration2 > 0 else 0
        speedup3 = duration1 / duration3 if duration3 > 0 else 0

        print("\n[RESULTS]")
        print(f"  Cache miss: {duration1:.1f}ms")
        print(f"  Cache hit 1: {duration2:.1f}ms (speedup: {speedup2:.1f}x)")
        print(f"  Cache hit 2: {duration3:.1f}ms (speedup: {speedup3:.1f}x)")

        # Cache stats
        stats = shuttle.query_cache.stats()
        print(f"\n[CACHE STATS]")
        print(f"  Size: {stats['size']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")


async def benchmark_repeated_queries():
    """Test performance with multiple queries."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Repeated Queries")
    print("=" * 80)

    config = Config.bare()
    shards = create_test_shards()

    async with WeavingShuttle(cfg=config, shards=shards, enable_reflection=False) as shuttle:
        queries = [
            "What is Thompson Sampling?",
            "How does reinforcement learning work?",
            "What are knowledge graphs?",
            "What is Thompson Sampling?",  # Repeat
            "How does reinforcement learning work?",  # Repeat
        ]

        total_time = 0

        for i, query_text in enumerate(queries, 1):
            query = Query(text=query_text)
            start = time.time()
            result = await shuttle.weave(query)
            duration = (time.time() - start) * 1000
            total_time += duration

            print(f"\n[Query {i}] {query_text}")
            print(f"  Duration: {duration:.1f}ms")
            print(f"  Tool: {result.tool_used}")

        avg_time = total_time / len(queries)

        print(f"\n[RESULTS]")
        print(f"  Total queries: {len(queries)}")
        print(f"  Total time: {total_time:.1f}ms")
        print(f"  Average time: {avg_time:.1f}ms")

        # Cache stats
        stats = shuttle.query_cache.stats()
        print(f"\n[CACHE STATS]")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")


async def benchmark_embedder_cache():
    """Test embedder caching performance."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Embedder Caching")
    print("=" * 80)

    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    import time

    embedder = MatryoshkaEmbeddings(sizes=[96])

    texts = [
        "Thompson Sampling",
        "Reinforcement Learning",
        "Knowledge Graphs",
        "Thompson Sampling",  # Repeat
        "Reinforcement Learning",  # Repeat
    ]

    print("\n[Test 1] Encoding 5 texts (3 unique, 2 repeats)")

    durations = []
    for i, text in enumerate(texts, 1):
        start = time.time()
        emb = embedder.encode_base([text])
        duration = (time.time() - start) * 1000
        durations.append(duration)

        is_repeat = "(repeat)" if text in texts[:i-1] else "(new)"
        print(f"  Text {i}: {duration:.1f}ms {is_repeat}")

    print(f"\n[RESULTS]")
    print(f"  First 3 (new): avg {sum(durations[:3])/3:.1f}ms")
    print(f"  Last 2 (cached): avg {sum(durations[3:])/2:.1f}ms")

    speedup = (sum(durations[:3])/3) / (sum(durations[3:])/2) if sum(durations[3:]) > 0 else float('inf')
    print(f"  Speedup: {speedup:.1f}x")


async def main():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("HoloLoom Performance Benchmark Suite")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    await benchmark_query_cache()
    await benchmark_repeated_queries()
    await benchmark_embedder_cache()

    print("\n" + "=" * 80)
    print("[OK] All benchmarks complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
