#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Scraping -> Memory Storage Pipeline Demo

This demonstrates the complete flow:
1. Scrape webpage with WebsiteSpinner
2. Generate MemoryShards
3. Store in unified memory (Neo4j + Qdrant)
4. Query and retrieve
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinningWheel.website import WebsiteSpinner, WebsiteSpinnerConfig
from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories, Memory


async def scrape_and_store(url: str, memory_backend: str = 'simple'):
    """
    Complete pipeline: Scrape URL → Store in memory → Query.

    Args:
        url: URL to scrape
        memory_backend: 'simple', 'neo4j', 'qdrant', or 'neo4j+qdrant'
    """

    print("=" * 80)
    print("WEB SCRAPING → MEMORY STORAGE PIPELINE")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Memory Backend: {memory_backend}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Scrape webpage
    # ========================================================================
    print("\n[STEP 1] Scraping webpage...")
    print("-" * 80)

    config = WebsiteSpinnerConfig(
        extract_images=False,  # Skip images for demo
        chunk_by='paragraph',
        chunk_size=500,
        timeout=15
    )

    spinner = WebsiteSpinner(config)

    try:
        shards = await spinner.spin({'url': url})
        print(f"✓ Generated {len(shards)} shards")

        total_chars = sum(len(s.text) for s in shards)
        total_words = sum(len(s.text.split()) for s in shards)

        print(f"  - Total content: {total_chars:,} chars, {total_words:,} words")
        print(f"  - Entities extracted: {sum(len(s.entities) for s in shards)}")

    except Exception as e:
        print(f"✗ Scraping failed: {e}")
        return

    # ========================================================================
    # STEP 2: Convert shards to Memory objects
    # ========================================================================
    print("\n[STEP 2] Converting shards to Memory objects...")
    print("-" * 80)

    memories = shards_to_memories(shards)
    print(f"✓ Converted {len(memories)} shards to Memory objects")

    # Show first memory structure
    if memories:
        mem = memories[0]
        print(f"\nSample Memory object:")
        print(f"  - ID: {mem.id}")
        print(f"  - Text: {mem.text[:100]}...")
        print(f"  - Timestamp: {mem.timestamp}")
        print(f"  - Entities: {mem.context.get('entities', [])[:5]}")
        print(f"  - Episode: {mem.context.get('episode', 'N/A')}")
        print(f"  - Metadata keys: {list(mem.metadata.keys())}")

    # ========================================================================
    # STEP 3: Create memory backend and store
    # ========================================================================
    print(f"\n[STEP 3] Storing in {memory_backend} memory...")
    print("-" * 80)

    try:
        memory = create_unified_memory(memory_backend)
        print(f"✓ Created {memory_backend} memory backend")

        # Store memories
        print(f"  Storing {len(memories)} memories...")
        stored_ids = await memory.store_batch(memories)
        print(f"✓ Stored {len(stored_ids)} memories")

        # Show stored IDs
        print(f"\nStored memory IDs:")
        for mem_id in stored_ids[:5]:
            print(f"  - {mem_id}")
        if len(stored_ids) > 5:
            print(f"  ... and {len(stored_ids) - 5} more")

    except Exception as e:
        print(f"✗ Storage failed: {e}")
        print(f"\nNote: Make sure your memory backend is configured:")
        print(f"  - Neo4j: Check connection settings in config")
        print(f"  - Qdrant: Ensure Qdrant is running")
        print(f"  - 'simple': Works without external dependencies")
        return None, None

    # ========================================================================
    # STEP 4: Query the stored memories
    # ========================================================================
    print("\n[STEP 4] Querying stored memories...")
    print("-" * 80)

    test_queries = [
        "agent skills best practices",
        "Claude API development",
        "skill authoring"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = await memory.search(query=query, limit=3)
            print(f"  Found {len(results)} results")

            for idx, result in enumerate(results, 1):
                print(f"\n  Result {idx}:")
                print(f"    ID: {result.id}")
                print(f"    Score: {getattr(result, 'score', 'N/A')}")
                print(f"    Text preview: {result.text[:100]}...")
                entities = result.context.get('entities', []) if hasattr(result, 'context') else []
                print(f"    Entities: {entities[:3]}")

        except Exception as e:
            print(f"  ✗ Query failed: {e}")

    # ========================================================================
    # STEP 5: Show memory statistics
    # ========================================================================
    print("\n[STEP 5] Memory statistics...")
    print("-" * 80)

    try:
        # Get all memories for this episode
        episode = shards[0].episode if shards else None
        if episode:
            episode_memories = await memory.retrieve_episode(episode)
            print(f"✓ Episode '{episode}' contains {len(episode_memories)} memories")

        print(f"\nMemory backend info:")
        print(f"  - Type: {type(memory).__name__}")
        print(f"  - Backend: {memory_backend}")

    except Exception as e:
        print(f"Note: Statistics not available for this backend: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"✓ Scraped {len(shards)} shards from {url}")
    print(f"✓ Stored {len(memories)} memories in {memory_backend}")
    print(f"✓ Memories are now searchable and retrievable")
    print("=" * 80)

    return memory, memories


async def demo_recursive_crawl_to_memory():
    """
    Bonus: Recursive crawl → Memory storage.
    """
    print("\n\n" + "=" * 80)
    print("BONUS: RECURSIVE CRAWL → MEMORY STORAGE")
    print("=" * 80)

    from HoloLoom.spinningWheel.recursive_crawler import crawl_recursive, CrawlConfig

    seed_url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"
    seed_topic = "Claude agent skills best practices"

    config = CrawlConfig(
        max_depth=1,  # Just seed + direct links
        max_pages=5,
        extract_images=False,
        rate_limit_seconds=1.5
    )

    print(f"Crawling from: {seed_url}")
    print(f"Topic: {seed_topic}")
    print(f"Config: depth={config.max_depth}, max_pages={config.max_pages}")

    try:
        from HoloLoom.spinningWheel.recursive_crawler import RecursiveCrawler
        crawler = RecursiveCrawler(config)
        pages = await crawler.crawl(seed_url, seed_topic)

        print(f"\n✓ Crawled {len(pages)} pages")

        # Collect all shards
        all_shards = []
        for page in pages:
            all_shards.extend(page['shards'])

        print(f"✓ Generated {len(all_shards)} total shards")

        # Store in memory
        print(f"\nStoring in simple memory backend...")
        memory = create_unified_memory('simple')
        memories = shards_to_memories(all_shards)
        stored_ids = await memory.store_batch(memories)

        print(f"✓ Stored {len(stored_ids)} memories from crawled pages")

        # Test query
        print(f"\nTest query: 'agent skills'")
        results = await memory.search(query="agent skills", limit=5)
        print(f"✓ Found {len(results)} relevant memories across multiple pages")

        for idx, result in enumerate(results, 1):
            page_url = result.metadata.get('url', 'unknown')
            print(f"  {idx}. {page_url}")
            print(f"     {result.text[:80]}...")

    except Exception as e:
        print(f"✗ Recursive crawl demo failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the demo."""

    # Test URL
    url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"

    # Run with simple memory backend (no external dependencies)
    print("\nUsing 'simple' memory backend (in-memory, no dependencies)")
    print("For production, use 'neo4j', 'qdrant', or 'neo4j+qdrant'\n")

    memory, memories = await scrape_and_store(url, memory_backend='simple')

    if memory and memories:
        # Run bonus demo
        try:
            await demo_recursive_crawl_to_memory()
        except KeyboardInterrupt:
            print("\n\nRecursive crawl demo skipped")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")