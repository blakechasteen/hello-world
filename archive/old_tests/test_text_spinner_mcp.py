"""
Test SpinningWheel text spinner integration with MCP server.

This tests the process_text tool that:
1. Accepts text input
2. Runs it through the text spinner (chunking + entity extraction)
3. Converts MemoryShards → Memory objects
4. Stores using store_many()
"""

import sys
import asyncio
from pathlib import Path

# Add mythRL to path to fix import issues
sys.path.insert(0, str(Path(__file__).parent))

# Windows UTF-8 encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Import modules directly by file path (avoid HoloLoom.__init__ which triggers policy imports)
import importlib.util

def import_module_by_path(module_name, file_path):
    """Import a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import protocol module directly
protocol = import_module_by_path(
    "HoloLoom.memory.protocol",
    Path(__file__).parent / "HoloLoom" / "memory" / "protocol.py"
)
create_unified_memory = protocol.create_unified_memory
Strategy = protocol.Strategy
shards_to_memories = protocol.shards_to_memories

# Import text spinner directly
text_spinner = import_module_by_path(
    "HoloLoom.spinning_wheel.text",
    Path(__file__).parent / "HoloLoom" / "spinningWheel" / "text.py"
)
spin_text = text_spinner.spin_text


async def test_text_spinner_integration():
    """Test the full text spinner → memory pipeline."""

    print("=" * 60)
    print("Testing SpinningWheel Text Spinner → MCP Integration")
    print("=" * 60)
    print()

    # Sample text to process
    sample_text = """
    Beekeeping in Winter: Essential Practices

    Winter is a critical time for honey bees. The colony clusters together
    to maintain warmth, with the queen at the center. Worker bees vibrate
    their flight muscles to generate heat, keeping the cluster temperature
    around 93°F (34°C).

    Feeding: Check honey stores regularly. Each colony needs 50-60 pounds
    of honey to survive winter. If stores are low, provide sugar fondant
    or candy boards rather than liquid syrup in cold weather.

    Ventilation: Moisture is the enemy. Proper ventilation prevents condensation
    that can drip onto the cluster and chill the bees. Use a ventilated inner
    cover or create a small upper entrance.

    Inspection: Minimize hive inspections during cold weather. Quick visual
    checks from outside are sufficient. Listen for the low hum of the cluster
    on warmer days - this confirms the colony is alive and active.

    Protection: Consider wrapping hives in black tar paper or insulation in
    extreme climates. Reduce entrance size to prevent mice from entering.
    Install entrance reducers and mouse guards before winter begins.
    """

    # Create memory system (in-memory for testing)
    print("1. Initializing memory system...")
    memory = await create_unified_memory(
        user_id="blake",
        enable_mem0=False,  # Use in-memory for quick test
        enable_neo4j=False,
        enable_qdrant=False
    )
    print("   ✓ Memory system ready")
    print()

    # Step 2: Spin text into shards
    print("2. Spinning text through SpinningWheel...")
    print(f"   Input: {len(sample_text)} characters")
    shards = await spin_text(
        text=sample_text,
        source="beekeeping_article",
        chunk_by="paragraph",  # Chunk by paragraphs
        chunk_size=500
    )
    print(f"   ✓ Created {len(shards)} shards")
    print()

    # Step 3: Convert shards → Memory objects
    print("3. Converting MemoryShards → Memory objects...")
    memories = shards_to_memories(shards)
    print(f"   ✓ Converted {len(memories)} memories")
    print()

    # Add tags to all memories
    for mem in memories:
        mem.tags = ["beekeeping", "winter", "test"]
        mem.user_id = "blake"

    # Step 4: Batch store
    print("4. Batch storing memories...")
    memory_ids = await memory.store_many(memories)
    print(f"   ✓ Stored {len(memory_ids)} memories")
    print()

    # Step 5: Analyze what was extracted
    print("5. Analysis of extracted features:")
    print()

    total_entities = 0
    total_motifs = 0
    all_entities = set()
    all_motifs = set()

    for i, mem in enumerate(memories, 1):
        entities = mem.context.get('entities', [])
        motifs = mem.context.get('motifs', [])

        total_entities += len(entities)
        total_motifs += len(motifs)
        all_entities.update(entities)
        all_motifs.update(motifs)

        print(f"   Chunk {i}:")
        print(f"   - Text: {mem.text[:80]}...")
        print(f"   - Entities: {len(entities)} ({', '.join(entities[:3])}{'...' if len(entities) > 3 else ''})")
        print(f"   - Motifs: {len(motifs)} ({', '.join(motifs[:3])}{'...' if len(motifs) > 3 else ''})")
        print(f"   - ID: {mem.id}")
        print()

    print(f"   Summary:")
    print(f"   - Total entities: {total_entities} ({len(all_entities)} unique)")
    print(f"   - Total motifs: {total_motifs} ({len(all_motifs)} unique)")
    print()

    if all_entities:
        print(f"   Unique Entities: {', '.join(sorted(all_entities))}")
    if all_motifs:
        print(f"   Unique Motifs: {', '.join(sorted(all_motifs))}")
    print()

    # Step 6: Test retrieval
    print("6. Testing semantic retrieval...")
    query = "how to keep bees warm in winter"
    results = await memory.recall(query, strategy=Strategy.SEMANTIC, limit=3)
    print(f"   Query: '{query}'")
    print(f"   ✓ Found {len(results.memories)} relevant memories")
    print()

    for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
        print(f"   Result {i} (score: {score:.3f}):")
        print(f"   {mem.text[:100]}...")
        print()

    # Step 7: Health check
    print("7. System health check...")
    health = await memory.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Backend: {health['backend']}")
    print(f"   Total memories: {health['memory_count']}")
    print()

    print("=" * 60)
    print("✓ All tests passed! Text spinner → MCP integration works!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_text_spinner_integration())
