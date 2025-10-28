#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple web scraping to memory demo."""

import asyncio
import sys
from pathlib import Path

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinning_wheel.website import WebsiteSpinner, WebsiteSpinnerConfig
from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories


async def main():
    url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"

    print("=" * 80)
    print("WEB SCRAPING TO MEMORY PIPELINE")
    print("=" * 80)
    print(f"URL: {url}\n")

    # Step 1: Scrape
    print("[1] Scraping webpage...")
    config = WebsiteSpinnerConfig(extract_images=False, chunk_by='paragraph', chunk_size=500)
    spinner = WebsiteSpinner(config)
    shards = await spinner.spin({'url': url})
    print(f"    Generated {len(shards)} shards ({sum(len(s.text) for s in shards):,} chars)\n")

    # Step 2: Convert to memories
    print("[2] Converting to Memory objects...")
    memories = shards_to_memories(shards)
    print(f"    Created {len(memories)} Memory objects\n")

    # Step 3: Store in memory
    print("[3] Storing in simple memory backend...")
    memory = await create_unified_memory('simple')
    stored_ids = await memory.store_batch(memories)
    print(f"    Stored {len(stored_ids)} memories\n")

    # Step 4: Query
    print("[4] Querying stored memories...")
    for query in ["agent skills", "best practices"]:
        print(f"    Query: '{query}'")
        results = await memory.search(query=query, limit=2)
        print(f"      Found {len(results)} results")
        for r in results:
            print(f"        - {r.id}: {r.text[:60]}...")

    print("\n" + "=" * 80)
    print("COMPLETE! Scraped web content now searchable in memory.")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
