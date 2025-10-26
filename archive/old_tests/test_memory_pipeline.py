"""
Test: Complete Memory Pipeline
==============================
Demonstrates the obvious data piping:
    Text → Spinner → MemoryShards → Memories → Store

This shows how easy it is to pipe data into the protocol-based memory system.
"""

import asyncio
import sys
from pathlib import Path

# Bypass HoloLoom __init__ by importing from specific paths
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "HoloLoom"))

# Import directly
import importlib.util

def load_module_from_path(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules directly
protocol_path = repo_root / "HoloLoom" / "memory" / "protocol.py"
store_path = repo_root / "HoloLoom" / "memory" / "stores" / "in_memory_store.py"
spinner_path = repo_root / "HoloLoom" / "spinningWheel" / "text.py"

protocol = load_module_from_path("protocol", protocol_path)
in_memory = load_module_from_path("in_memory_store", store_path)
text_spinner = load_module_from_path("text_spinner", spinner_path)

# Import what we need
Memory = protocol.Memory
UnifiedMemoryInterface = protocol.UnifiedMemoryInterface
Strategy = protocol.Strategy
shards_to_memories = protocol.shards_to_memories
pipe_text_to_memory = protocol.pipe_text_to_memory
InMemoryStore = in_memory.InMemoryStore
spin_text = text_spinner.spin_text


async def test_basic_pipeline():
    """Test basic data piping: spinner → shards → memories → store."""
    print("=" * 70)
    print("TEST 1: Basic Pipeline (Step by Step)")
    print("=" * 70)

    # Create memory interface with InMemoryStore
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)

    # Sample text
    text = """
    Hive Inspection - Jodi Colony

    Date: October 23, 2025
    Weather: Clear, 65°F

    Observations:
    - 8 frames of brood (excellent pattern)
    - Queen spotted on frame 3
    - Honey stores looking good for winter
    - No signs of disease

    Action Items:
    - Add second honey super next week
    - Monitor mite levels
    - Prepare winter insulation
    """

    print("\n1. Spinning text into shards...")
    shards = await spin_text(
        text=text,
        source='hive_inspection_notes.txt',
        chunk_by='paragraph',
        chunk_size=200
    )
    print(f"   ✓ Created {len(shards)} shards")

    print("\n2. Converting shards to memories...")
    memories = shards_to_memories(shards)
    print(f"   ✓ Converted to {len(memories)} Memory objects")

    print("\n3. Storing memories...")
    ids = await memory.store_many(memories)
    print(f"   ✓ Stored {len(ids)} memories")
    for idx, mem_id in enumerate(ids):
        print(f"      - {idx + 1}. {mem_id}")

    print("\n4. Recalling memories...")
    results = await memory.recall(
        query="winter preparation",
        strategy=Strategy.SEMANTIC,
        limit=3
    )
    print(f"   ✓ Found {results.count} relevant memories")
    for idx, (mem, score) in enumerate(zip(results.memories, results.scores)):
        print(f"\n   Memory {idx + 1} (relevance: {score:.2f}):")
        print(f"   ID: {mem.id}")
        print(f"   Text: {mem.text[:100]}...")
        print(f"   Entities: {mem.context.get('entities', [])}")

    print("\n✓ TEST 1 PASSED\n")


async def test_one_liner_pipeline():
    """Test the one-liner pipe_text_to_memory utility."""
    print("=" * 70)
    print("TEST 2: One-Liner Pipeline (Convenience Function)")
    print("=" * 70)

    # Create memory interface
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)

    # Sample text
    text = """
    YouTube Transcript Notes

    Video: Advanced Beekeeping Techniques

    Key Points:
    - Queen rearing methods
    - Swarm prevention strategies
    - Optimal hive placement
    - Seasonal management tips
    """

    print("\n1. Piping text directly into memory (one function call)...")
    ids = await pipe_text_to_memory(
        text=text,
        memory=memory,
        source='youtube_transcript',
        chunk_by='paragraph'
    )
    print(f"   ✓ Stored {len(ids)} memories in one call!")

    print("\n2. Recalling all memories...")
    results = await memory.recall(
        query="beekeeping",
        strategy=Strategy.FUSED,
        limit=10
    )
    print(f"   ✓ Found {results.count} memories")

    print("\n✓ TEST 2 PASSED\n")


async def test_shard_conversion():
    """Test Memory.from_shard() conversion."""
    print("=" * 70)
    print("TEST 3: MemoryShard → Memory Conversion")
    print("=" * 70)

    # Create memory interface
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)

    # Get shards from spinner
    text = "Simple test document for conversion demo."
    shards = await spin_text(text, source='test.txt')

    print(f"\n1. Created {len(shards)} shard(s)")
    shard = shards[0]
    print(f"   Shard ID: {shard.id}")
    print(f"   Shard text: {shard.text}")
    print(f"   Shard episode: {shard.episode}")
    print(f"   Shard entities: {shard.entities}")
    print(f"   Shard motifs: {shard.motifs}")
    print(f"   Shard metadata: {shard.metadata}")

    print("\n2. Converting to Memory...")
    mem = Memory.from_shard(shard)
    print(f"   Memory ID: {mem.id}")
    print(f"   Memory text: {mem.text}")
    print(f"   Memory timestamp: {mem.timestamp}")
    print(f"   Memory context: {mem.context}")
    print(f"   Memory metadata: {mem.metadata}")

    print("\n3. Storing Memory...")
    mem_id = await memory.store_memory(mem)
    print(f"   ✓ Stored as: {mem_id}")

    print("\n4. Retrieving Memory...")
    retrieved = await store.get_by_id(mem_id)
    print(f"   ✓ Retrieved: {retrieved.text[:50]}...")

    print("\n✓ TEST 3 PASSED\n")


async def test_health_check():
    """Test system health check."""
    print("=" * 70)
    print("TEST 4: System Health Check")
    print("=" * 70)

    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)

    # Store some data
    await pipe_text_to_memory(
        text="Sample data for health check",
        memory=memory
    )

    print("\n1. Checking system health...")
    health = await memory.health_check()

    print(f"   Store status: {health['store']['status']}")
    print(f"   Backend: {health['store']['backend']}")
    print(f"   Memory count: {health['store']['memory_count']}")
    print(f"   Navigator: {health['navigator']}")
    print(f"   Detector: {health['detector']}")

    print("\n✓ TEST 4 PASSED\n")


async def test_strategies():
    """Test different retrieval strategies."""
    print("=" * 70)
    print("TEST 5: Retrieval Strategies")
    print("=" * 70)

    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)

    # Store multiple memories
    texts = [
        "Hive inspection shows healthy brood pattern",
        "Queen bee spotted on frame 3",
        "Honey stores adequate for winter",
        "Mite count within acceptable range",
        "Added second honey super today"
    ]

    print("\n1. Storing test memories...")
    for text in texts:
        await memory.store(text, user_id="blake")
    print(f"   ✓ Stored {len(texts)} memories")

    # Test each strategy
    strategies = [
        Strategy.TEMPORAL,
        Strategy.SEMANTIC,
        Strategy.FUSED
    ]

    for strat in strategies:
        print(f"\n2. Testing {strat.value} strategy...")
        results = await memory.recall(
            query="hive queen",
            strategy=strat,
            limit=3,
            user_id="blake"
        )
        print(f"   ✓ Found {results.count} memories")
        print(f"   Strategy used: {results.strategy_used}")
        for idx, (mem, score) in enumerate(zip(results.memories, results.scores)):
            print(f"   {idx + 1}. [{score:.2f}] {mem.text[:50]}...")

    print("\n✓ TEST 5 PASSED\n")


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" MEMORY PIPELINE TESTS")
    print(" Demonstrating: Text → Spinner → Shards → Memories → Store")
    print("=" * 70 + "\n")

    try:
        await test_basic_pipeline()
        await test_one_liner_pipeline()
        await test_shard_conversion()
        await test_health_check()
        await test_strategies()

        print("=" * 70)
        print(" ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("1. Data piping is OBVIOUS and CLEAN")
        print("2. Multiple approaches: step-by-step or one-liner")
        print("3. Protocol-based design = flexible backends")
        print("4. Memory.from_shard() makes conversion trivial")
        print("5. shards_to_memories() handles batches")
        print("6. pipe_text_to_memory() does everything in one call")
        print("\nThe answer to your question: YES, it's now obvious! 🎯")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)