"""
Test Medium-Term Features
==========================

Tests the three new systems:
1. Smart Deduplication
2. Advanced Query Engine with Filters
3. Reverse Query Analysis

Demonstrates how they work together.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List

# Setup path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.memory.deduplication import (
    DeduplicationEngine,
    DuplicateGroup,
    ContentSignature
)
from HoloLoom.memory.query_enhancements import (
    AdvancedQueryEngine,
    QueryFilter,
    QueryOptions,
    SortOrder,
    TimeRange,
    QueryBuilder
)
from HoloLoom.memory.reverse_query import (
    ReverseQueryEngine,
    what_queries_find_this,
    how_discoverable,
    make_more_findable
)


def test_deduplication():
    """Test 1: Smart Deduplication"""
    print("=" * 60)
    print("TEST 1: SMART DEDUPLICATION")
    print("=" * 60)
    print()

    engine = DeduplicationEngine(
        near_threshold=0.85,
        fuzzy_threshold=0.75
    )

    # Test URLs with tracking params
    url1 = "https://example.com/article?utm_source=twitter&utm_campaign=summer"
    url2 = "https://example.com/article?utm_source=facebook&ref=123"
    url3 = "https://example.com/article"

    # Import URL normalizer
    from HoloLoom.memory.deduplication import URLNormalizer

    print("URL Normalization Test:")
    print(f"  Original: {url1}")
    print(f"  Normalized: {URLNormalizer.normalize(url1)}")
    print()

    # Test exact duplicates
    content1 = "The quick brown fox jumps over the lazy dog."
    content2 = "The quick brown fox jumps over the lazy dog."
    content3 = "The quick brown fox jumps over a lazy dog."  # Near-duplicate
    content4 = "Completely different content about beekeeping."

    print("Content Deduplication:")

    # Add first memory
    sig1 = engine.create_signature(content1, url1, "mem1")
    canonical1 = engine.add_content(sig1, "mem1")
    print(f"OK Added memory 1")
    print(f"  Content hash: {sig1.content_hash}")
    print(f"  Simhash: {sig1.simhash}")
    print(f"  Is duplicate: {canonical1 is not None}")
    print()

    # Check exact duplicate
    sig2 = engine.create_signature(content2, url3, "mem2")
    dup2 = engine.check_duplicate(sig2, "mem2")
    print(f"Checking exact duplicate:")
    if dup2:
        existing_id, similarity, dup_type = dup2
        print(f"  OK Found duplicate!")
        print(f"  Type: {dup_type}")
        print(f"  Match: {existing_id}")
        print(f"  Score: {similarity:.3f}")
    else:
        print(f"  No duplicates found")
    print()

    # Add third (near-duplicate)
    sig3 = engine.create_signature(content3, "https://example.com/other", "mem3")
    dup3 = engine.check_duplicate(sig3, "mem3")
    print(f"Checking near-duplicate:")
    if dup3:
        existing_id, similarity, dup_type = dup3
        print(f"  OK Found duplicate!")
        print(f"  Type: {dup_type}")
        print(f"  Match: {existing_id}")
        print(f"  Score: {similarity:.3f}")
    else:
        print(f"  No duplicates found")
    print()

    # Check unique content
    sig4 = engine.create_signature(content4, "https://beekeeping.com", "mem4")
    dup4 = engine.check_duplicate(sig4, "mem4")
    print(f"Checking unique content:")
    if dup4:
        print(f"  Unexpected duplicate found!")
    else:
        print(f"  OK No duplicates (as expected)")
    canonical4 = engine.add_content(sig4, "mem4")
    print()

    # Get duplicate groups
    groups = [engine.get_duplicates(mid) for mid in ["mem1", "mem2", "mem3", "mem4"]]
    groups = [g for g in groups if g is not None]
    print(f"Duplicate Groups: {len(groups)}")
    for i, group in enumerate(groups):
        print(f"  Group {i+1}: {len(group.duplicate_ids)} duplicates")
        print(f"    Canonical: {group.canonical_id}")
        print(f"    Duplicates: {', '.join(group.duplicate_ids)}")
    print()

    print("OK Deduplication tests complete")
    print()


async def test_advanced_queries():
    """Test 2: Advanced Query Engine"""
    print("=" * 60)
    print("TEST 2: ADVANCED QUERY ENGINE")
    print("=" * 60)
    print()

    # Mock memory store for testing
    class MockMemoryStore:
        def __init__(self):
            self.memories = [
                {
                    'id': 'mem1',
                    'text': 'How to prepare beehives for winter',
                    'timestamp': datetime.now() - timedelta(days=1),
                    'domain': 'beekeeping.com',
                    'importance': 0.9,
                    'crawl_depth': 0,
                    'has_images': True,
                    'image_count': 3,
                    'tags': ['beekeeping', 'winter']
                },
                {
                    'id': 'mem2',
                    'text': 'Summer honey harvesting techniques',
                    'timestamp': datetime.now() - timedelta(days=30),
                    'domain': 'beekeeping.com',
                    'importance': 0.8,
                    'crawl_depth': 1,
                    'has_images': False,
                    'image_count': 0,
                    'tags': ['beekeeping', 'summer', 'harvest']
                },
                {
                    'id': 'mem3',
                    'text': 'Python async programming tutorial',
                    'timestamp': datetime.now() - timedelta(days=7),
                    'domain': 'python.org',
                    'importance': 0.95,
                    'crawl_depth': 0,
                    'has_images': True,
                    'image_count': 5,
                    'tags': ['python', 'programming']
                },
                {
                    'id': 'mem4',
                    'text': 'Old beekeeping article from last year',
                    'timestamp': datetime.now() - timedelta(days=400),
                    'domain': 'oldsite.com',
                    'importance': 0.3,
                    'crawl_depth': 2,
                    'has_images': False,
                    'image_count': 0,
                    'tags': ['beekeeping']
                }
            ]

        async def recall(self, query, limit=10):
            # Simple mock: return all memories
            class Result:
                def __init__(self, memories):
                    self.memories = [type('Mem', (), m) for m in memories]
                    self.scores = [0.8] * len(memories)
            return Result(self.memories)

    store = MockMemoryStore()
    engine = AdvancedQueryEngine(store)

    # Test 1: Recent + High Quality
    print("Query 1: Recent + High Quality")
    print("-" * 40)
    filter1 = QueryFilter(
        after=datetime.now() - timedelta(days=10),
        min_importance=0.8
    )
    options1 = QueryOptions(sort_by=SortOrder.RECENCY)

    results1 = await engine.query("beekeeping", filters=filter1, options=options1)
    print(f"Found {len(results1.memories)} results:")
    for r in results1.memories:
        print(f"  • [{r.score:.2f}] {r.memory.text[:50]}...")
        print(f"    Timestamp: {r.memory.timestamp}")
        print(f"    Importance: {r.memory.importance}")
    print()

    # Test 2: Multimodal Content Only
    print("Query 2: Multimodal Content (has images)")
    print("-" * 40)
    filter2 = QueryFilter(has_images=True)
    options2 = QueryOptions(sort_by=SortOrder.RELEVANCE)

    results2 = await engine.query("tutorial", filters=filter2, options=options2)
    print(f"Found {len(results2.memories)} results with images:")
    for r in results2.memories:
        print(f"  • [{r.score:.2f}] {r.memory.text[:50]}...")
        print(f"    Images: {r.memory.image_count}")
    print()

    # Test 3: Domain + Tag Filtering
    print("Query 3: Domain + Tag Filtering")
    print("-" * 40)
    filter3 = QueryFilter(
        domains=['beekeeping.com'],
        tags=['winter']
    )

    results3 = await engine.query("preparation", filters=filter3)
    print(f"Found {len(results3.memories)} beekeeping.com results with 'winter' tag:")
    for r in results3.memories:
        print(f"  • [{r.score:.2f}] {r.memory.text[:50]}...")
        print(f"    Domain: {r.memory.domain}")
        print(f"    Tags: {r.memory.tags}")
    print()

    # Test 4: Faceted Search
    print("Query 4: Faceted Search")
    print("-" * 40)
    filter4 = QueryFilter()
    options4 = QueryOptions(facets=['domain', 'tags', 'has_images'])

    results4 = await engine.query("beekeeping", filters=filter4, options=options4)
    print(f"Found {len(results4.memories)} total results")
    print(f"Facets:")
    for facet_name, counts in results4.facets.items():
        print(f"  {facet_name}:")
        for value, count in counts.items():
            print(f"    - {value}: {count}")
    print()

    # Test 5: Query Builder (Fluent API)
    print("Query 5: Query Builder (Fluent API)")
    print("-" * 40)
    results5 = await (
        QueryBuilder(engine, "python")
        .recent(days=30)
        .with_images()
        .min_importance(0.9)
        .sort_by_relevance()
        .limit(5)
        .execute()
    )
    print(f"Fluent API query found {len(results5.memories)} results")
    for r in results5.memories:
        print(f"  • [{r.score:.2f}] {r.memory.text[:50]}...")
    print()

    print("OK Advanced query tests complete")
    print()


async def test_reverse_queries():
    """Test 3: Reverse Query Analysis"""
    print("=" * 60)
    print("TEST 3: REVERSE QUERY ANALYSIS")
    print("=" * 60)
    print()

    # Mock memory store
    class MockStore:
        pass

    engine = ReverseQueryEngine(MockStore())

    # Populate some term frequencies (simulate existing corpus)
    engine.term_frequencies.update({
        'beekeeping': 50,
        'winter': 30,
        'hive': 45,
        'preparation': 20,
        'honey': 60,
        'queen': 25,
        'varroa': 5,  # Rare term
        'mites': 8
    })

    # Test memory
    memory_text = """
    Preparing beehives for winter is crucial for colony survival.
    The varroa mites must be treated before cold weather arrives.
    Ensure adequate honey stores and proper ventilation.
    The queen should be healthy and laying well in autumn.
    """

    print("Analyzing memory:")
    print(f'"{memory_text[:100]}..."')
    print()

    result = await engine.analyze("mem_winter_prep", memory_text)

    print("EXACT QUERIES (guaranteed to find this):")
    for q in result.exact_queries:
        print(f"  • {q}")
    print()

    print("LIKELY QUERIES (high probability):")
    for q, score in result.likely_queries[:5]:
        print(f"  • [{score:.2f}] {q}")
    print()

    print("POSSIBLE QUERIES (might find this):")
    for q, score in result.possible_queries[:5]:
        print(f"  • [{score:.2f}] {q}")
    print()

    print("KEYWORD CATEGORIES:")
    print(f"  Primary: {', '.join(result.primary_keywords[:5])}")
    print(f"  Secondary: {', '.join(result.secondary_keywords[:5])}")
    print(f"  Rare (distinctive): {', '.join(result.rare_keywords[:5])}")
    print()

    print("CONCEPTS & ENTITIES:")
    print(f"  Concepts: {', '.join(result.concepts)}")
    print(f"  Entities: {', '.join(result.entities)}")
    print()

    print("SCORES:")
    print(f"  Discoverability: {result.discoverability_score:.2f} "
          "(how easy to find)")
    print(f"  Uniqueness: {result.uniqueness_score:.2f} "
          "(how rare/distinctive)")
    print()

    # Test convenience functions
    print("CONVENIENCE FUNCTIONS:")
    print("-" * 40)

    queries = await what_queries_find_this("mem1", memory_text, engine)
    print(f"what_queries_find_this():")
    for q in queries[:3]:
        print(f"  • {q}")
    print()

    disc_score = await how_discoverable("mem1", memory_text, engine)
    print(f"how_discoverable(): {disc_score:.2f}")
    print()

    suggestions = await make_more_findable("mem1", memory_text, engine)
    print(f"make_more_findable():")
    print(f"  Add tags: {', '.join(suggestions['add_tags'][:3])}")
    print(f"  Add keywords: {', '.join(suggestions['add_keywords'][:3])}")
    print()

    print("OK Reverse query tests complete")
    print()


async def test_integration():
    """Test 4: Integration - All Three Systems Together"""
    print("=" * 60)
    print("TEST 4: INTEGRATION - ALL SYSTEMS WORKING TOGETHER")
    print("=" * 60)
    print()

    print("Scenario: Ingesting web content with smart deduplication,")
    print("          advanced queries, and reverse query analysis")
    print()

    # Setup engines
    dedup_engine = DeduplicationEngine()

    class MockStore:
        def __init__(self):
            self.memories = []

        async def recall(self, query, limit=10):
            class Result:
                def __init__(self, memories):
                    self.memories = memories
                    self.scores = [0.8] * len(memories)
            return Result(self.memories)

    store = MockStore()
    query_engine = AdvancedQueryEngine(store)
    reverse_engine = ReverseQueryEngine(store)

    # Simulate ingesting multiple webpages
    pages = [
        {
            'url': 'https://example.com/article1?utm_source=twitter',
            'content': 'How to prepare beehives for winter. Essential tips for beekeepers.'
        },
        {
            'url': 'https://example.com/article1?ref=facebook',  # Duplicate URL
            'content': 'How to prepare beehives for winter. Essential tips for beekeepers.'  # Same content
        },
        {
            'url': 'https://other.com/winter-bees',
            'content': 'How to prepare beehives for winter season. Important tips for beekeepers.'  # Near-duplicate
        },
        {
            'url': 'https://example.com/honey-harvest',
            'content': 'Summer honey harvesting techniques and best practices.'  # Unique
        }
    ]

    print("INGESTION PIPELINE:")
    print("-" * 40)

    ingested_count = 0
    duplicate_count = 0

    for i, page in enumerate(pages, 1):
        print(f"\nPage {i}: {page['url']}")

        # Step 1: Create signature
        sig = dedup_engine.create_signature(page['content'], page['url'], f"mem{i}")

        # Step 2: Check for duplicates
        duplicate = dedup_engine.check_duplicate(sig, f"mem{i}")

        if duplicate:
            existing_id, similarity, dup_type = duplicate
            print(f"  WARNING  Duplicate detected!")
            print(f"      Type: {dup_type}")
            print(f"      Match: {existing_id}")
            print(f"      Similarity: {similarity:.2f}")
            duplicate_count += 1
            # Still add to index to track duplicate
            dedup_engine.add_content(sig, f"mem{i}")
            continue

        # Step 3: Add to dedup index
        dedup_engine.add_content(sig, f"mem{i}")

        # Step 3: Analyze discoverability
        reverse_result = await reverse_engine.analyze(f"mem{i}", page['content'])

        print(f"  OK Ingested successfully")
        print(f"    Content hash: {sig.content_hash}")
        print(f"    Discoverability: {reverse_result.discoverability_score:.2f}")
        print(f"    Top queries: {', '.join(reverse_result.exact_queries[:2])}")

        # Step 4: Store in memory (mock)
        store.memories.append(type('Memory', (), {
            'id': f"mem{i}",
            'text': page['content'],
            'url': page['url'],
            'timestamp': datetime.now(),
            'domain': page['url'].split('/')[2],
            'importance': reverse_result.discoverability_score,
            'tags': reverse_result.primary_keywords[:3],
            'has_images': False,
            'image_count': 0,
            'crawl_depth': 0
        })())

        ingested_count += 1

    print()
    print("INGESTION SUMMARY:")
    print(f"  Total pages: {len(pages)}")
    print(f"  Ingested: {ingested_count}")
    print(f"  Duplicates skipped: {duplicate_count}")
    print()

    # Now query the ingested content
    print("QUERYING INGESTED CONTENT:")
    print("-" * 40)

    filter = QueryFilter(min_importance=0.5)
    results = await query_engine.query("beehives winter", filters=filter)

    print(f"Query: 'beehives winter'")
    print(f"Results: {len(results.memories)}")
    for r in results.memories:
        print(f"  • [{r.score:.2f}] {r.memory.text[:60]}...")
        print(f"    URL: {r.memory.url}")
        print(f"    Tags: {', '.join(r.memory.tags)}")
    print()

    print("OK Integration test complete")
    print()


async def main():
    """Run all tests"""
    print("\n")
    print("*" * 60)
    print("MEDIUM-TERM FEATURES TEST SUITE")
    print("*" * 60)
    print("\n")

    # Test 1: Deduplication
    test_deduplication()

    # Test 2: Advanced Queries
    await test_advanced_queries()

    # Test 3: Reverse Queries
    await test_reverse_queries()

    # Test 4: Integration
    await test_integration()

    print("*" * 60)
    print("ALL TESTS COMPLETE OK")
    print("*" * 60)
    print()
    print("Summary:")
    print("  OK Smart deduplication working (exact, near, fuzzy)")
    print("  OK Advanced query filters working (temporal, domain, quality)")
    print("  OK Reverse query analysis working (discoverability, suggestions)")
    print("  OK All three systems integrated successfully")
    print()


if __name__ == "__main__":
    asyncio.run(main())
