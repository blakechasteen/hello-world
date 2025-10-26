#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show the structure of MemoryShards in detail.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinningWheel.website import WebsiteSpinner, WebsiteSpinnerConfig


async def main():
    """Scrape and show shard details."""

    url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"

    config = WebsiteSpinnerConfig(
        extract_images=False,  # Skip images for simplicity
        chunk_by='paragraph',
        chunk_size=500,
        timeout=15
    )

    spinner = WebsiteSpinner(config)
    shards = await spinner.spin({'url': url})

    print("=" * 80)
    print(f"SCRAPED {len(shards)} SHARDS FROM:")
    print(f"{url}")
    print("=" * 80)

    for idx, shard in enumerate(shards, 1):
        print(f"\n{'='*80}")
        print(f"SHARD {idx}: {shard.id}")
        print(f"{'='*80}")

        print(f"\nBasic Info:")
        print(f"  ID: {shard.id}")
        print(f"  Episode: {shard.episode}")
        print(f"  Text Length: {len(shard.text)} chars, {len(shard.text.split())} words")

        print(f"\nEntities: {shard.entities}")
        print(f"Motifs: {shard.motifs}")

        print(f"\nMetadata:")
        for key, val in shard.metadata.items():
            if isinstance(val, str) and len(val) > 60:
                print(f"  {key}: {val[:60]}...")
            else:
                print(f"  {key}: {val}")

        print(f"\nText Preview (first 800 chars):")
        print("-" * 80)
        print(shard.text[:800])
        if len(shard.text) > 800:
            print(f"\n... [{len(shard.text) - 800} more characters]")
        print("-" * 80)

    # Show how these integrate with memory
    print("\n" + "=" * 80)
    print("MEMORY INTEGRATION")
    print("=" * 80)
    print("\nThese shards can be stored in:")
    print("  1. Neo4j (graph structure)")
    print("  2. Qdrant (vector embeddings)")
    print("  3. Mem0 (unified memory)")
    print("\nExample code:")
    print("""
    from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories

    # Create unified memory backend
    memory = create_unified_memory('neo4j+qdrant')

    # Convert shards to memories
    memories = shards_to_memories(shards)

    # Store in memory system
    await memory.store_batch(memories)

    # Query later
    results = await memory.search(
        query="agent skills best practices",
        limit=5
    )
    """)


if __name__ == '__main__':
    asyncio.run(main())
