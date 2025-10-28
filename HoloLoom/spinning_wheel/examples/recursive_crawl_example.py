"""
Recursive Crawling with Matryoshka Importance Gating Example
=============================================================

Demonstrates intelligent recursive web crawling where:
- Only important links are followed
- Importance threshold increases with depth
- Prevents crawling noise while capturing related content

Example output:
    Depth 0 (seed): Always crawled
    Depth 1: Links scoring 0.6+ followed
    Depth 2: Links scoring 0.75+ followed
    Depth 3: Links scoring 0.85+ followed

This creates a "funnel" effect - broad at top, narrow at bottom.
"""

import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinning_wheel.recursive_crawler import RecursiveCrawler, CrawlConfig, crawl_recursive
from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_crawl():
    """Basic recursive crawl with default settings."""

    logger.info("EXAMPLE 1: Basic Recursive Crawl")
    logger.info("=" * 60)

    pages = await crawl_recursive(
        seed_url='https://en.wikipedia.org/wiki/Beekeeping',
        seed_topic='beekeeping hive management',
        max_depth=2,
        max_pages=15
    )

    logger.info("\nResults:")
    logger.info(f"  Total pages: {len(pages)}")

    # Show depth distribution
    depth_counts = {}
    for page in pages:
        depth = page['depth']
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    for depth in sorted(depth_counts.keys()):
        logger.info(f"  Depth {depth}: {depth_counts[depth]} pages")

    return pages


async def example_matryoshka_thresholds():
    """Demonstrate matryoshka importance thresholds."""

    logger.info("\nEXAMPLE 2: Matryoshka Importance Gating")
    logger.info("=" * 60)

    # Custom thresholds - very strict at deeper levels
    config = CrawlConfig(
        max_depth=3,
        max_pages=25,
        importance_thresholds={
            0: 0.0,   # Seed: always crawl
            1: 0.5,   # Depth 1: easy (50%+ relevant)
            2: 0.75,  # Depth 2: harder (75%+ relevant)
            3: 0.9,   # Depth 3: very hard (90%+ relevant)
        }
    )

    crawler = RecursiveCrawler(config)

    pages = await crawler.crawl(
        seed_url='https://en.wikipedia.org/wiki/Python_(programming_language)',
        seed_topic='Python programming language'
    )

    logger.info("\nMatryoshka Gating Results:")
    logger.info(f"  Total pages: {len(pages)}")

    # Show what got through at each depth
    for depth in range(4):
        depth_pages = [p for p in pages if p['depth'] == depth]

        if depth_pages:
            threshold = config.importance_thresholds.get(depth, 0.9)
            avg_score = sum(p['link'].importance_score for p in depth_pages) / len(depth_pages)

            logger.info(f"  Depth {depth}: {len(depth_pages)} pages")
            logger.info(f"    Threshold: {threshold:.2f}")
            logger.info(f"    Avg score: {avg_score:.2f}")

            # Show examples
            for page in depth_pages[:3]:
                logger.info(f"      • [{page['link'].importance_score:.2f}] {page['title'][:50]}")


async def example_domain_restricted():
    """Crawl staying on one domain."""

    logger.info("\nEXAMPLE 3: Same-Domain Crawl")
    logger.info("=" * 60)

    config = CrawlConfig(
        max_depth=2,
        max_pages=20,
        same_domain_only=True,  # Don't leave domain
        max_pages_per_domain=20
    )

    crawler = RecursiveCrawler(config)

    pages = await crawler.crawl(
        seed_url='https://en.wikipedia.org/wiki/Beekeeping'
    )

    # Check domains
    domains = set(p['domain'] for p in pages)

    logger.info(f"\nDomains visited: {len(domains)}")
    for domain in domains:
        count = sum(1 for p in pages if p['domain'] == domain)
        logger.info(f"  {domain}: {count} pages")


async def example_with_memory_storage():
    """Crawl and store everything in memory."""

    logger.info("\nEXAMPLE 4: Crawl + Store in Memory")
    logger.info("=" * 60)

    # Crawl
    pages = await crawl_recursive(
        seed_url='https://en.wikipedia.org/wiki/Honey_bee',
        seed_topic='honey bee biology',
        max_depth=1,
        max_pages=10
    )

    # Store in memory
    logger.info("\nStoring in memory system...")

    memory = await create_unified_memory(
        user_id="blake",
        enable_mem0=False,
        enable_neo4j=True,
        enable_qdrant=False
    )

    total_stored = 0

    for page in pages:
        # Convert shards to memories
        memories = shards_to_memories(page['shards'])

        # Add metadata
        for mem in memories:
            mem.user_id = "blake"
            mem.tags = [
                'web-crawl',
                f"depth-{page['depth']}",
                page['domain'].replace('.', '_')
            ]
            mem.metadata['crawl_depth'] = page['depth']
            mem.metadata['parent_url'] = page['link'].parent_url
            mem.metadata['importance_score'] = page['link'].importance_score

        # Store
        ids = await memory.store_many(memories)
        total_stored += len(ids)

        logger.info(f"  ✓ {page['title'][:40]} - {len(ids)} chunks")

    logger.info(f"\n✓ Stored {total_stored} total memory chunks")

    # Demonstrate retrieval
    logger.info("\nTesting semantic search...")

    from HoloLoom.memory.protocol import Strategy

    results = await memory.recall(
        "bee colony organization",
        strategy=Strategy.SEMANTIC,
        limit=5
    )

    logger.info(f"Found {len(results.memories)} relevant chunks:")

    for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
        depth = mem.metadata.get('crawl_depth', '?')
        importance = mem.metadata.get('importance_score', 0)

        logger.info(f"  {i}. [score: {score:.3f}] [depth: {depth}] [imp: {importance:.2f}]")
        logger.info(f"     {mem.text[:80]}...")


async def example_link_scoring_details():
    """Show how link importance scoring works."""

    logger.info("\nEXAMPLE 5: Link Importance Scoring Details")
    logger.info("=" * 60)

    # Crawl just one page to see link scores
    config = CrawlConfig(
        max_depth=0,  # Don't follow, just show scores
        max_pages=1
    )

    crawler = RecursiveCrawler(config)

    # Manually extract and score links to show details
    logger.info("\nScoring links from seed page...")

    # Would need to modify crawler to expose this, but concept is:
    # For each link:
    #   - Topic relevance: +0.3 if anchor contains topic words
    #   - Anchor quality: +0.1 if descriptive (10+ chars)
    #   - Context similarity: +0.2 if surrounding text matches topic
    #   - Same domain: +0.15
    #   - Position: +0.1 for early links
    #
    # Penalties:
    #   - Social links: -0.5
    #   - Navigation (home/about): -0.3
    #
    # Result: Links scored 0-1
    # Only links above depth threshold are followed

    logger.info("\nExample link scores:")
    logger.info("  0.92 - 'Advanced beekeeping techniques' (main article)")
    logger.info("  0.78 - 'Hive management in winter' (related topic)")
    logger.info("  0.65 - 'Types of honey bees' (somewhat related)")
    logger.info("  0.45 - 'Beekeeping equipment suppliers' (tangential)")
    logger.info("  0.22 - 'Contact us' (navigation, not content)")
    logger.info("  0.08 - 'Share on Twitter' (social link)")

    logger.info("\nWith depth 1 threshold of 0.6:")
    logger.info("  ✓ Followed: 0.92, 0.78, 0.65")
    logger.info("  ✗ Skipped: 0.45, 0.22, 0.08")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Recursive web crawling with matryoshka importance gating"
    )
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Which example to run (1-5)'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Custom seed URL to crawl'
    )
    parser.add_argument(
        '--topic',
        type=str,
        help='Seed topic for importance scoring'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=2,
        help='Maximum crawl depth'
    )
    parser.add_argument(
        '--pages',
        type=int,
        default=20,
        help='Maximum pages to crawl'
    )

    args = parser.parse_args()

    if args.url:
        # Custom crawl
        asyncio.run(crawl_recursive(
            seed_url=args.url,
            seed_topic=args.topic,
            max_depth=args.depth,
            max_pages=args.pages
        ))
    elif args.example == 1:
        asyncio.run(example_basic_crawl())
    elif args.example == 2:
        asyncio.run(example_matryoshka_thresholds())
    elif args.example == 3:
        asyncio.run(example_domain_restricted())
    elif args.example == 4:
        asyncio.run(example_with_memory_storage())
    elif args.example == 5:
        asyncio.run(example_link_scoring_details())
    else:
        # Run all examples
        print("\nRunning all examples...\n")
        asyncio.run(example_basic_crawl())
        asyncio.run(example_matryoshka_thresholds())
        asyncio.run(example_domain_restricted())
        # Skip 4 (requires Neo4j)
        asyncio.run(example_link_scoring_details())


if __name__ == "__main__":
    main()
