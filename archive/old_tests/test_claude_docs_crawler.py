#!/usr/bin/env python3
"""
Test web crawler on Claude documentation page.
Tests both single-page scraping and recursive crawling.
"""

import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinningWheel.website import WebsiteSpinner, WebsiteSpinnerConfig
from HoloLoom.spinningWheel.recursive_crawler import crawl_recursive, CrawlConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_single_page():
    """Test single page scraping."""
    logger.info("=" * 70)
    logger.info("TEST 1: Single Page Scraping")
    logger.info("=" * 70)

    url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"

    config = WebsiteSpinnerConfig(
        extract_images=True,
        max_images=5,
        chunk_by='paragraph',
        chunk_size=500,
        timeout=15
    )

    spinner = WebsiteSpinner(config)

    try:
        shards = await spinner.spin({
            'url': url,
        })

        logger.info(f"\n✓ Successfully scraped page")
        logger.info(f"  Generated {len(shards)} memory shards")

        # Show first shard details
        if shards:
            shard = shards[0]
            logger.info(f"\nFirst shard details:")
            logger.info(f"  ID: {shard.id}")
            logger.info(f"  Text length: {len(shard.text)} chars")
            logger.info(f"  Entities: {shard.entities[:5]}...")  # First 5
            logger.info(f"  Metadata keys: {list(shard.metadata.keys())}")
            logger.info(f"\nText preview (first 200 chars):")
            logger.info(f"  {shard.text[:200]}...")

        # Show content distribution
        logger.info(f"\nShard distribution:")
        for i, shard in enumerate(shards[:5]):  # First 5 shards
            logger.info(f"  Shard {i+1}: {len(shard.text)} chars, {len(shard.entities)} entities")

        if len(shards) > 5:
            logger.info(f"  ... and {len(shards) - 5} more shards")

        return shards

    except Exception as e:
        logger.error(f"✗ Failed to scrape page: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_recursive_crawl():
    """Test recursive crawling with matryoshka gating."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Recursive Crawling with Matryoshka Gating")
    logger.info("=" * 70)

    seed_url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"
    seed_topic = "Claude agent skills best practices API development"

    config = CrawlConfig(
        max_depth=2,
        max_pages=10,
        importance_thresholds={
            0: 0.0,   # Seed (always crawl)
            1: 0.6,   # Direct links (60%+ relevant)
            2: 0.75,  # Second-level (75%+ relevant)
        },
        extract_images=False,  # Skip images for faster crawling
        rate_limit_seconds=1.5,  # Be polite to Claude servers
        max_runtime_minutes=5,
        same_domain_only=True  # Stay on docs.claude.com
    )

    try:
        logger.info(f"\nCrawling configuration:")
        logger.info(f"  Seed URL: {seed_url}")
        logger.info(f"  Topic: {seed_topic}")
        logger.info(f"  Max depth: {config.max_depth}")
        logger.info(f"  Max pages: {config.max_pages}")
        logger.info(f"  Importance thresholds: {config.importance_thresholds}")

        from HoloLoom.spinningWheel.recursive_crawler import RecursiveCrawler
        crawler = RecursiveCrawler(config)
        pages = await crawler.crawl(seed_url, seed_topic)

        logger.info(f"\n✓ Crawl completed successfully")
        logger.info(f"  Total pages crawled: {len(pages)}")

        # Show depth distribution
        depth_counts = {}
        for page in pages:
            depth = page['depth']
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        logger.info(f"\nDepth distribution:")
        for depth in sorted(depth_counts.keys()):
            logger.info(f"  Depth {depth}: {depth_counts[depth]} pages")

        # Show sample pages from each depth
        logger.info(f"\nSample pages:")
        for depth in sorted(depth_counts.keys()):
            depth_pages = [p for p in pages if p['depth'] == depth]
            sample = depth_pages[:3]  # First 3 from each depth
            for page in sample:
                logger.info(f"  [Depth {depth}] {page['url']}")
                if page.get('importance_score'):
                    logger.info(f"            Importance: {page['importance_score']:.2f}")

        # Count total shards
        total_shards = sum(len(page['shards']) for page in pages)
        logger.info(f"\nTotal memory shards generated: {total_shards}")

        return pages

    except Exception as e:
        logger.error(f"✗ Crawl failed: {e}")
        import traceback
        traceback.print_exc()
        return []


async def main():
    """Run both tests."""

    # Test 1: Single page
    shards = await test_single_page()

    # Test 2: Recursive crawl (optional, takes longer)
    print("\n" + "=" * 70)
    print("Would you like to run the recursive crawl test?")
    print("This will crawl multiple pages and may take a few minutes.")
    print("(Press Ctrl+C to skip)")
    print("=" * 70)

    try:
        await asyncio.sleep(3)  # Give user time to cancel
        pages = await test_recursive_crawl()
    except KeyboardInterrupt:
        logger.info("\n\nRecursive crawl test skipped by user")

    logger.info("\n" + "=" * 70)
    logger.info("All tests completed!")
    logger.info("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
