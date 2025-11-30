"""
Test: Web Crawler + Matryoshka Search Integration
=================================================
Demonstrates the complete pipeline:
1. MatryoshkaWebSearch (3-stage filtering)
2. RecursiveCrawler (importance-gated exploration)
3. WebsiteSpinner (content + image extraction)
4. Citation formatting
"""

import asyncio
from HoloLoom.search.web_crawler_integration import (
    WebCrawlerSearch,
    WebCrawlerSearchConfig,
    search_and_crawl_web
)


async def test_search_only():
    """Test 1: Search only (no recursive crawl)"""
    print("=" * 80)
    print("TEST 1: SEARCH ONLY (No Recursive Crawl)")
    print("=" * 80)
    print()

    result = await search_and_crawl_web(
        query="What is Thompson Sampling?",
        enable_recursive_crawl=False,
        max_search_results=3,
        search_provider="mock"
    )

    print(f"Search Results: {len(result.search_results)}")
    print(f"Pages Crawled: {len(result.crawled_pages)}")
    print(f"Memory Shards: {len(result.shards)}")
    print(f"Search Time: {result.search_time_ms:.1f}ms")
    print(f"Total Time: {result.total_duration_ms:.1f}ms")
    print()
    print("Top Search Results:")
    for i, sr in enumerate(result.search_results, 1):
        print(f"  [{i}] {sr.title[:50]}...")
        print(f"      URL: {sr.url}")
        print(f"      Score: {sr.final_score:.3f}")
    print()
    print("-" * 80)
    print()


async def test_search_and_crawl():
    """Test 2: Search + recursive crawl"""
    print("=" * 80)
    print("TEST 2: SEARCH + RECURSIVE CRAWL")
    print("=" * 80)
    print()

    result = await search_and_crawl_web(
        query="Thompson Sampling exploration exploitation",
        seed_topic="Thompson Sampling Bayesian bandits",
        enable_recursive_crawl=True,
        max_search_results=2,
        max_crawl_depth=2,
        max_pages_per_seed=3,
        search_provider="mock"
    )

    print(f"Search Results: {len(result.search_results)}")
    print(f"Pages Crawled: {len(result.crawled_pages)}")
    print(f"Memory Shards: {len(result.shards)}")
    print(f"Search Time: {result.search_time_ms:.1f}ms")
    print(f"Crawl Time: {result.crawl_time_ms:.1f}ms")
    print(f"Extract Time: {result.extraction_time_ms:.1f}ms")
    print(f"Total Time: {result.total_duration_ms:.1f}ms")
    print()

    print("Crawled Pages by Depth:")
    depth_counts = {}
    for page in result.crawled_pages:
        depth = page['depth']
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]} pages")
    print()

    print("Sample Crawled Pages:")
    for i, page in enumerate(result.crawled_pages[:5], 1):
        print(f"  [{i}] Depth {page['depth']} | {page['title'][:40]}...")
        print(f"      URL: {page['url']}")
        print(f"      Shards: {len(page['shards'])}")
    print()
    print("-" * 80)
    print()


async def test_citations():
    """Test 3: Citation formatting"""
    print("=" * 80)
    print("TEST 3: CITATION FORMATTING")
    print("=" * 80)
    print()

    result = await search_and_crawl_web(
        query="What is Thompson Sampling?",
        enable_recursive_crawl=False,
        max_search_results=3,
        search_provider="mock"
    )

    print("Cited Response:")
    print("-" * 80)
    print(result.cited_response)
    print()
    print("-" * 80)
    print()


async def test_custom_config():
    """Test 4: Custom configuration"""
    print("=" * 80)
    print("TEST 4: CUSTOM CONFIGURATION")
    print("=" * 80)
    print()

    config = WebCrawlerSearchConfig(
        search_provider="mock",
        max_search_results=2,
        enable_recursive_crawl=True,
        max_crawl_depth=1,
        max_pages_per_seed=2,
        max_total_pages=5,
        extract_images=True,
        max_images_per_page=3,
        importance_thresholds={
            0: 0.0,   # Seeds
            1: 0.7,   # High threshold for depth 1
        },
        enable_citations=True
    )

    crawler = WebCrawlerSearch(config)

    result = await crawler.search_and_crawl(
        query="machine learning",
        seed_topic="neural networks deep learning"
    )

    print(f"Results: {len(result.search_results)} search + {len(result.crawled_pages)} crawled")
    print(f"Total Shards: {len(result.shards)}")
    print(f"Total Time: {result.total_duration_ms:.1f}ms")
    print()

    # Get statistics
    stats = crawler.get_stats()
    print("Statistics:")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Total crawls: {stats['total_crawls']}")
    print(f"  Pages crawled: {stats['total_pages_crawled']}")
    print(f"  Shards created: {stats['total_shards_created']}")
    print(f"  Avg search time: {stats['avg_search_time_ms']:.1f}ms")
    print(f"  Avg pages per crawl: {stats['avg_pages_per_crawl']:.1f}")
    print()
    print("-" * 80)
    print()


async def main():
    """Run all tests"""
    print()
    print("=" * 80)
    print(" " * 15 + "WEB CRAWLER + MATRYOSHKA SEARCH INTEGRATION")
    print("=" * 80)
    print()

    try:
        # Test 1: Search only
        await test_search_only()

        # Test 2: Search + crawl
        await test_search_and_crawl()

        # Test 3: Citations
        await test_citations()

        # Test 4: Custom config
        await test_custom_config()

        print("=" * 80)
        print("ALL TESTS COMPLETE!")
        print("=" * 80)
        print()
        print("Integration Features Demonstrated:")
        print("  [x] MatryoshkaWebSearch (3-stage filtering)")
        print("  [x] RecursiveCrawler (importance gating)")
        print("  [x] WebsiteSpinner (content + image extraction)")
        print("  [x] Citation formatting (inline + bibliography)")
        print("  [x] Configurable pipelines (search-only vs full crawl)")
        print("  [x] Statistics tracking")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
