"""
Browser History Auto-Ingest Example
====================================

Automatically ingest your browser history into HoloLoom memory.

This script:
1. Reads your Chrome/Edge/Firefox browsing history
2. Filters for meaningful visits (30+ seconds on page)
3. Scrapes page content
4. Processes through WebsiteSpinner
5. Stores in memory system

Usage:
    # Ingest last 7 days from Chrome
    python browser_history_ingest.py

    # Ingest last 30 days from all browsers
    python browser_history_ingest.py --days 30 --browser all

    # Dry run (show what would be ingested)
    python browser_history_ingest.py --dry-run
"""

import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Setup path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinning_wheel.browser_history import get_recent_history, BrowserHistoryReader
from HoloLoom.spinning_wheel.website import spin_webpage
from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def ingest_browser_history(
    days_back: int = 7,
    min_duration: int = 30,
    browser: str = 'chrome',
    dry_run: bool = False,
    user_id: str = "blake"
):
    """
    Ingest browser history into memory.

    Args:
        days_back: How many days of history to ingest
        min_duration: Minimum page visit duration (seconds)
        browser: 'chrome', 'edge', 'firefox', or 'all'
        dry_run: If True, just show what would be ingested
        user_id: User identifier for stored memories
    """
    logger.info("=" * 60)
    logger.info("Browser History Auto-Ingest")
    logger.info("=" * 60)
    logger.info(f"Browser: {browser}")
    logger.info(f"Days back: {days_back}")
    logger.info(f"Min duration: {min_duration}s")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")

    # Read history
    logger.info("Reading browser history...")
    visits = get_recent_history(
        days_back=days_back,
        min_duration=min_duration,
        browser=browser
    )

    if not visits:
        logger.warning("No browser history found!")
        return

    logger.info(f"Found {len(visits)} meaningful visits")
    logger.info("")

    # Filter out common noise patterns
    exclude_patterns = [
        'google.com/search',
        'bing.com/search',
        'duckduckgo.com',
        'facebook.com',
        'twitter.com',
        'reddit.com/r/all',
        'youtube.com/feed',
        'youtube.com/watch',  # YouTube videos are too long
        'localhost',
        '127.0.0.1',
    ]

    filtered_visits = []
    for visit in visits:
        if any(pattern in visit.url for pattern in exclude_patterns):
            continue
        filtered_visits.append(visit)

    logger.info(f"After filtering: {len(filtered_visits)} visits to ingest")
    logger.info("")

    if dry_run:
        logger.info("DRY RUN - Would ingest the following:")
        logger.info("")
        for i, visit in enumerate(filtered_visits[:20], 1):
            logger.info(f"{i}. {visit.title[:60]}")
            logger.info(f"   URL: {visit.url[:80]}")
            logger.info(f"   Duration: {visit.duration}s | Visited: {visit.timestamp.strftime('%Y-%m-%d %H:%M')}")
            logger.info("")

        if len(filtered_visits) > 20:
            logger.info(f"... and {len(filtered_visits) - 20} more")

        return

    # Initialize memory
    logger.info("Initializing memory system...")
    memory = await create_unified_memory(
        user_id=user_id,
        enable_mem0=False,  # Use in-memory for testing
        enable_neo4j=True,  # Use Neo4j if available
        enable_qdrant=False
    )
    logger.info("Memory system ready")
    logger.info("")

    # Ingest each visit
    ingested_count = 0
    failed_count = 0

    for i, visit in enumerate(filtered_visits, 1):
        logger.info(f"[{i}/{len(filtered_visits)}] Processing: {visit.url[:60]}")

        try:
            # Spin webpage
            shards = await spin_webpage(
                url=visit.url,
                title=visit.title,
                tags=['browsing-history', browser],
                visited_at=visit.timestamp,
                duration=visit.duration
            )

            if not shards:
                logger.warning(f"  ✗ No content extracted")
                failed_count += 1
                continue

            # Convert to memories
            memories = shards_to_memories(shards)

            # Add user ID
            for mem in memories:
                mem.user_id = user_id

            # Store
            ids = await memory.store_many(memories)

            logger.info(f"  ✓ Stored {len(ids)} chunks")
            ingested_count += 1

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed_count += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✓ Ingestion complete!")
    logger.info(f"  • Successful: {ingested_count}")
    logger.info(f"  • Failed: {failed_count}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest browser history into HoloLoom memory"
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='How many days of history to ingest (default: 7)'
    )
    parser.add_argument(
        '--min-duration',
        type=int,
        default=30,
        help='Minimum page visit duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--browser',
        choices=['chrome', 'edge', 'firefox', 'brave', 'all'],
        default='chrome',
        help='Which browser to read from (default: chrome)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be ingested without actually doing it'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        default='blake',
        help='User ID for stored memories (default: blake)'
    )

    args = parser.parse_args()

    asyncio.run(ingest_browser_history(
        days_back=args.days,
        min_duration=args.min_duration,
        browser=args.browser,
        dry_run=args.dry_run,
        user_id=args.user_id
    ))


if __name__ == "__main__":
    main()
