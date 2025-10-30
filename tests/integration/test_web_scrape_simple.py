#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplest possible web scraping demo - just shows the data flow.
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinning_wheel.website import WebsiteSpinner, WebsiteSpinnerConfig
from HoloLoom.memory.protocol import shards_to_memories


async def main():
    """Demonstrate web scraping and conversion to memories."""

    url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"

    print("=" * 80)
    print("WEB SCRAPING DEMONSTRATION")
    print("=" * 80)
    print(f"Target: {url}\n")

    # Configure spinner
    config = WebsiteSpinnerConfig(
        extract_images=False,
        chunk_by='paragraph',
        chunk_size=500,
        timeout=15
    )

    spinner = WebsiteSpinner(config)

    # Scrape the page
    print("[Step 1] Scraping webpage...")
    shards = await spinner.spin({'url': url})

    total_chars = sum(len(s.text) for s in shards)
    total_words = sum(len(s.text.split()) for s in shards)
    total_entities = sum(len(s.entities) for s in shards)

    print(f"  ✓ Generated {len(shards)} shards")
    print(f"  ✓ Total content: {total_chars:,} characters, {total_words:,} words")
    print(f"  ✓ Extracted {total_entities} entities\n")

    # Show shard details
    print("[Step 2] Shard Details:")
    for idx, shard in enumerate(shards, 1):
        print(f"\n  Shard {idx}:")
        print(f"    ID: {shard.id}")
        print(f"    Size: {len(shard.text)} chars, {len(shard.text.split())} words")
        print(f"    Entities: {shard.entities[:5]}")
        print(f"    Episode: {shard.episode}")
        print(f"    Text preview: {shard.text[:100]}...")

    # Convert to Memory objects
    print(f"\n[Step 3] Converting to Memory objects...")
    memories = shards_to_memories(shards)
    print(f"  ✓ Created {len(memories)} Memory objects")

    print(f"\n  Memory structure:")
    if memories:
        mem = memories[0]
        print(f"    - ID: {mem.id}")
        print(f"    - Timestamp: {mem.timestamp}")
        print(f"    - Text length: {len(mem.text)} chars")
        print(f"    - Context keys: {list(mem.context.keys())}")
        print(f"    - Metadata keys: {list(mem.metadata.keys())}")

    # Show what you can do with these
    print(f"\n[Step 4] What you can do with these memories:")
    print(f"  • Store in Neo4j (graph relationships)")
    print(f"  • Store in Qdrant (vector search)")
    print(f"  • Store in Mem0 (unified memory)")
    print(f"  • Pass to HoloLoom orchestrator (decision-making)")
    print(f"  • Query and retrieve for RAG applications")

    # Demonstrate simple in-memory search
    print(f"\n[Step 5] Simple search demonstration:")
    query_terms = ["skills", "best practices", "agent"]

    for term in query_terms:
        print(f"\n  Searching for '{term}':")
        matches = [m for m in memories if term.lower() in m.text.lower()]
        print(f"    Found in {len(matches)} memory shard(s)")

        for match in matches[:2]:  # Show first 2 matches
            # Find the context around the term
            text_lower = match.text.lower()
            idx = text_lower.find(term.lower())
            if idx != -1:
                start = max(0, idx - 40)
                end = min(len(match.text), idx + len(term) + 40)
                context = match.text[start:end]
                print(f"      ...{context}...")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nYou now have {len(memories)} searchable memory objects from:")
    print(f"  {url}")
    print(f"\nThese can be stored in any memory backend and queried later.")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
