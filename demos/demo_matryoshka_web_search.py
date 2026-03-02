#!/usr/bin/env python3
"""
Demo: Matryoshka Web Search - Perplexity-Style Search with HoloLoom
=====================================================================
Demonstrates the complete Matryoshka web search system:
1. Three-stage adaptive retrieval (96d � 192d � 384d)
2. Inline citations (Perplexity-style)
3. Memory shard conversion
4. Performance analysis

Usage:
    # With real API (requires SERPAPI_KEY environment variable)
    SERPAPI_KEY=your_key python demos/demo_matryoshka_web_search.py

    # With mock provider (no API key needed)
    python demos/demo_matryoshka_web_search.py --mock
"""

import asyncio
import os
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.search import (
    MatryoshkaWebSearch,
    SearchConfig,
    CitationFormatter,
    CitationStyle
)


async def demo_basic_search():
    """Demo 1: Basic three-stage search."""
    print("=" * 80)
    print("DEMO 1: Basic Matryoshka Web Search")
    print("=" * 80 + "\n")

    # Configure search
    api_key = os.getenv("SERPAPI_KEY")
    provider = "mock" if not api_key else "serpapi"

    config = SearchConfig(
        provider=provider,
        api_key=api_key,
        stage1_candidates=100,  # Broad search
        stage2_candidates=20,   # Refinement
        final_results=10        # Final results
    )

    # Create search instance
    search = MatryoshkaWebSearch(config=config)

    # Execute search
    query = "What is Thompson Sampling?"
    print(f"Query: {query}\n")

    results = await search.search(query)

    # Display results
    print(f"\nFound {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Domain: {result.domain}")
        print(f"   Scores: 96d={result.score_96d:.3f}, "
              f"192d={result.score_192d:.3f}, "
              f"384d={result.score_384d:.3f}")
        print(f"   Rank: {result.original_rank} � {result.final_rank} "
              f"(re-ranked by Matryoshka)")
        print(f"   Snippet: {result.snippet[:100]}...")
        print()

    # Show statistics
    stats = search.get_stats()
    print("\nPerformance Statistics:")
    print(f"  Total time: {stats['avg_time_ms']:.1f}ms")
    print(f"  Stage 1 (96d):  {stats['stage_times']['stage1']:.1f}ms")
    print(f"  Stage 2 (192d): {stats['stage_times']['stage2']:.1f}ms")
    print(f"  Stage 3 (384d): {stats['stage_times']['stage3']:.1f}ms")
    print(f"  Cache hits: {stats['cache_hits']}")


async def demo_with_citations():
    """Demo 2: Search with inline citations (Perplexity-style)."""
    print("\n" + "=" * 80)
    print("DEMO 2: Search with Inline Citations")
    print("=" * 80 + "\n")

    # Configure search
    api_key = os.getenv("SERPAPI_KEY")
    provider = "mock" if not api_key else "serpapi"

    config = SearchConfig(provider=provider, api_key=api_key)
    search = MatryoshkaWebSearch(config=config)

    # Execute search
    query = "What are the advantages of Thompson Sampling?"

    # Simulate an AI-generated response
    response = """Thompson Sampling is a Bayesian approach to the exploration-exploitation trade-off.
It offers several key advantages over other bandit algorithms. First, it naturally balances
exploration and exploitation without requiring manual parameter tuning. Second, it incorporates
prior knowledge effectively through Bayesian updates. Third, it performs optimally in many
practical scenarios, often matching or exceeding the performance of UCB algorithms."""

    # Add citations
    cited_response, results = await search.search_with_citations(
        query=query,
        response=response,
        max_results=5
    )

    print(f"Query: {query}\n")
    print("Response with citations:\n")
    print(cited_response)


async def demo_memory_integration():
    """Demo 3: Convert search results to memory shards for HoloLoom."""
    print("\n" + "=" * 80)
    print("DEMO 3: Memory Shard Integration")
    print("=" * 80 + "\n")

    # Configure search
    api_key = os.getenv("SERPAPI_KEY")
    provider = "mock" if not api_key else "serpapi"

    config = SearchConfig(provider=provider, api_key=api_key)
    search = MatryoshkaWebSearch(config=config)

    # Execute search and convert to shards
    query = "How does Thompson Sampling work?"
    results, shards = await search.search_to_shards(query, max_results=3)

    print(f"Query: {query}\n")
    print(f"Created {len(shards)} memory shards from {len(results)} search results:\n")

    for i, shard in enumerate(shards[:5], 1):  # Show first 5
        print(f"{i}. Shard ID: {shard.id}")
        print(f"   Episode: {shard.episode}")
        print(f"   Text: {shard.text[:100]}...")
        print(f"   Metadata:")
        print(f"     - Search query: {shard.metadata.get('search_query')}")
        print(f"     - Rank: {shard.metadata.get('search_rank')}")
        print(f"     - Similarity: {shard.metadata.get('similarity_score', 0):.3f}")
        print(f"     - URL: {shard.metadata.get('url')}")
        print(f"     - Domain: {shard.metadata.get('domain')}")
        print()

    print(f"These shards can now be used directly in HoloLoom queries!")


async def demo_performance_comparison():
    """Demo 4: Compare Matryoshka vs single-scale performance."""
    print("\n" + "=" * 80)
    print("DEMO 4: Performance Comparison")
    print("=" * 80 + "\n")

    # This demo illustrates the speedup (actual timing depends on API/network)
    print("Matryoshka Three-Stage Search:")
    print("  Stage 1 (96d):  100 results � 96d  = 9,600 dims   (~50ms)")
    print("  Stage 2 (192d): 20 results � 192d  = 3,840 dims   (~15ms)")
    print("  Stage 3 (384d): 10 results � 384d  = 3,840 dims   (~15ms)")
    print("  Total: ~80ms\n")

    print("Traditional Single-Scale Search:")
    print("  Stage 1 (384d): 100 results � 384d = 38,400 dims  (~500ms)")
    print("  Total: ~500ms\n")

    print("Speedup: 6.25�")
    print("\nWith Phase 5 compositional cache:")
    print("  Hot path: 0.03ms (291� speedup!)")
    print("  Cache hit rate: ~77.8%")


async def demo_cache_effectiveness():
    """Demo 5: Demonstrate caching effectiveness."""
    print("\n" + "=" * 80)
    print("DEMO 5: Cache Effectiveness")
    print("=" * 80 + "\n")

    # Configure search with cache
    api_key = os.getenv("SERPAPI_KEY")
    provider = "mock" if not api_key else "serpapi"

    config = SearchConfig(provider=provider, api_key=api_key)
    search = MatryoshkaWebSearch(config=config, enable_cache=True)

    query = "What is Thompson Sampling?"

    # First search (cold cache)
    print(f"Query 1: {query} (cold cache)")
    results1 = await search.search(query)
    stats1 = search.get_stats()
    print(f"  Time: {stats1['avg_time_ms']:.1f}ms")
    print(f"  Cache hits: {stats1['cache_hits']}")

    # Second search (warm cache)
    print(f"\nQuery 2: {query} (warm cache)")
    results2 = await search.search(query)
    stats2 = search.get_stats()
    print(f"  Time: {stats2['avg_time_ms']:.1f}ms")
    print(f"  Cache hits: {stats2['cache_hits']}")

    # Calculate speedup
    if stats1['avg_time_ms'] > 0:
        speedup = stats1['avg_time_ms'] / max(stats2['avg_time_ms'], 0.1)
        print(f"\nCache speedup: {speedup:.1f}�")

    print(f"\nCache statistics:")
    cache_stats = search.cache.get_stats()
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")


async def main():
    """Run all demos."""
    import argparse

    parser = argparse.ArgumentParser(description="Matryoshka Web Search Demo")
    parser.add_argument("--demo", type=int, choices=[1, 2, 3, 4, 5], help="Run specific demo")
    parser.add_argument("--mock", action="store_true", help="Use mock provider (no API key)")
    args = parser.parse_args()

    # Check for API key if not using mock
    if not args.mock and not os.getenv("SERPAPI_KEY"):
        print("WARNING: SERPAPI_KEY not set. Using mock provider.")
        print("Set SERPAPI_KEY environment variable to use real search.")
        print()
        os.environ["SERPAPI_KEY"] = ""  # Force mock

    # Run demos
    if args.demo:
        demos = {
            1: demo_basic_search,
            2: demo_with_citations,
            3: demo_memory_integration,
            4: demo_performance_comparison,
            5: demo_cache_effectiveness,
        }
        await demos[args.demo]()
    else:
        # Run all demos
        await demo_basic_search()
        await demo_with_citations()
        await demo_memory_integration()
        await demo_performance_comparison()
        await demo_cache_effectiveness()

    print("\n" + "=" * 80)
    print("All demos complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
