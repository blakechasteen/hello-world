"""
Simple test of text spinner → memory pipeline.

Tests the core functionality that the MCP process_text tool will use.
"""

import sys
import asyncio

# Windows UTF-8 encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


async def main():
    """Test text spinner integration."""
    print("=" * 60)
    print("Text Spinner → Memory Pipeline Test")
    print("=" * 60)
    print()

    # Sample beekeeping text
    sample_text = """
    Winter Beekeeping: Essential Practices

    Winter is a critical time for honey bees. The colony clusters together
    to maintain warmth, with the queen at the center. Worker bees vibrate
    their flight muscles to generate heat.

    Feeding: Check honey stores regularly. Each colony needs 50-60 pounds
    of honey to survive winter. If stores are low, provide sugar fondant.

    Ventilation: Moisture is the enemy. Proper ventilation prevents condensation
    that can drip onto the cluster and chill the bees.
    """

    print("Input text:")
    print(f"  Length: {len(sample_text)} characters")
    print(f"  Preview: {sample_text[:100]}...")
    print()

    # Step 1: Spin text into shards
    print("Step 1: Spinning text...")
    try:
        # Import only what we need - these modules don't trigger HoloLoom.__init__
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))

        # Direct imports from specific modules
        from HoloLoom.spinningWheel.text import spin_text
        from HoloLoom.memory.protocol import shards_to_memories, Memory
        from HoloLoom.memory.stores.in_memory_store import InMemoryStore
        from HoloLoom.memory.protocol import UnifiedMemoryInterface

        shards = await spin_text(
            text=sample_text,
            source="beekeeping_article",
            chunk_by="paragraph"
        )
        print(f"  ✓ Created {len(shards)} shards")
        print()

        # Step 2: Convert to memories
        print("Step 2: Converting shards → memories...")
        memories = shards_to_memories(shards)
        print(f"  ✓ Converted {len(memories)} Memory objects")
        print()

        # Step 3: Analyze extraction
        print("Step 3: Analyzing extracted features...")
        for i, mem in enumerate(memories, 1):
            entities = mem.context.get('entities', [])
            motifs = mem.context.get('motifs', [])
            print(f"  Chunk {i}:")
            print(f"    Text: {mem.text[:60]}...")
            print(f"    Entities: {len(entities)} - {entities[:3]}")
            print(f"    Motifs: {len(motifs)} - {motifs[:2]}")
        print()

        # Step 4: Store in memory
        print("Step 4: Storing in memory system...")
        store = InMemoryStore()
        memory_interface = UnifiedMemoryInterface(_store=store)

        # Add tags
        for mem in memories:
            mem.tags = ["beekeeping", "winter", "test"]
            mem.user_id = "blake"

        # Batch store
        ids = await memory_interface.store_many(memories)
        print(f"  ✓ Stored {len(ids)} memories")
        print()

        # Step 5: Verify retrieval
        print("Step 5: Testing retrieval...")
        from HoloLoom.memory.protocol import Strategy

        results = await memory_interface.recall(
            "how to keep bees warm",
            strategy=Strategy.SEMANTIC,
            limit=2
        )
        print(f"  Query: 'how to keep bees warm'")
        print(f"  ✓ Found {len(results.memories)} results")
        for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
            print(f"    {i}. [score: {score:.3f}] {mem.text[:80]}...")
        print()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("This demonstrates the pipeline used by MCP's process_text tool:")
        print("  1. Text → spin_text() → MemoryShards")
        print("  2. MemoryShards → shards_to_memories() → Memory objects")
        print("  3. Memory objects → store_many() → Stored in system")
        print("  4. Semantic search finds relevant chunks")

    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print()
        print("Note: This test requires HoloLoom modules to be available.")
        print("The import structure might need adjustment for your environment.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
