"""
Quick test for Neo4j + Qdrant memory integration via WeavingMemoryAdapter
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.memory.weaving_adapter import WeavingMemoryAdapter, create_weaving_memory
from HoloLoom.Documentation.types import MemoryShard, Query
from HoloLoom.config import Config, MemoryBackend


async def test_hybrid_memory():
    """Test Neo4j + Qdrant hybrid memory through adapter."""
    print("\n" + "="*70)
    print("MEMORY INTEGRATION TEST - Neo4j + Qdrant")
    print("="*70 + "\n")

    # Create config with hybrid backend
    config = Config.fused()
    config.memory_backend = MemoryBackend.NEO4J_QDRANT

    print(f"[SETUP] Backend: {config.memory_backend.value}")
    print(f"        Neo4j: {config.neo4j_uri}")
    print(f"        Qdrant: {config.qdrant_host}:{config.qdrant_port}")

    # Create test shards
    test_shards = [
        MemoryShard(
            id="test_001",
            text="Hive Jodi has 8 frames of brood and is very active",
            episode="inspection_oct_2025",
            entities=["Hive Jodi", "brood", "frames"],
            motifs=["HIVE_INSPECTION", "HEALTH"]
        ),
        MemoryShard(
            id="test_002",
            text="Winter preparations include wrapping hives and adding insulation",
            episode="planning_winter",
            entities=["winter", "insulation", "wrapping"],
            motifs=["PLANNING", "SEASONAL"]
        ),
        MemoryShard(
            id="test_003",
            text="Varroa mite treatment scheduled for Hive Delta next week",
            episode="treatment_plan",
            entities=["Varroa", "Hive Delta", "treatment"],
            motifs=["HEALTH", "TREATMENT"]
        )
    ]

    try:
        # Create adapter with backend factory (Neo4j + Qdrant)
        print("\n[1/5] Creating memory adapter...")
        adapter = WeavingMemoryAdapter.from_backend_factory(
            backend_type="hybrid",
            neo4j_config={
                'uri': config.neo4j_uri,
                'username': config.neo4j_username,
                'password': config.neo4j_password
            },
            qdrant_config={
                'host': config.qdrant_host,
                'port': config.qdrant_port
            }
        )
        print("        [OK] Adapter created")

        # Store test shards
        print(f"\n[2/5] Storing {len(test_shards)} memory shards...")
        for shard in test_shards:
            shard_id = await adapter.add_shard(shard)
            print(f"        Stored: {shard_id[:12]}... - {shard.text[:50]}...")

        print("        [OK] All shards stored")

        # Query 1: Hive inspection
        print("\n[3/5] Query 1: 'hive health'")
        query1 = Query(text="hive health")
        results1 = adapter.select_threads(None, query1)
        print(f"        Retrieved {len(results1)} shards")
        if results1:
            for i, shard in enumerate(results1[:3]):
                print(f"        [{i+1}] {shard.text[:60]}...")

        # Query 2: Winter prep
        print("\n[4/5] Query 2: 'winter preparations'")
        query2 = Query(text="winter preparations")
        results2 = adapter.select_threads(None, query2)
        print(f"        Retrieved {len(results2)} shards")
        if results2:
            for i, shard in enumerate(results2[:3]):
                print(f"        [{i+1}] {shard.text[:60]}...")

        # Get statistics
        print("\n[5/5] Backend statistics:")
        stats = adapter.get_statistics()
        for key, value in stats.items():
            print(f"        {key}: {value}")

        print("\n" + "="*70)
        print("[SUCCESS] Memory integration test complete!")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_direct_backend():
    """Test backend factory directly."""
    print("\n" + "="*70)
    print("DIRECT BACKEND TEST - Neo4j + Qdrant")
    print("="*70 + "\n")

    try:
        from HoloLoom.memory.backend_factory import create_memory_backend
        from HoloLoom.memory.protocol import Memory, MemoryQuery, Strategy
        from datetime import datetime

        config = Config.fused()
        config.memory_backend = MemoryBackend.NEO4J_QDRANT

        print("[1/3] Creating hybrid backend...")
        backend = await create_memory_backend(config)
        print("        [OK] Backend created")

        print("\n[2/3] Storing test memory...")
        test_memory = Memory(
            id="direct_test_001",
            text="Direct backend test: Queen is laying well in all hives",
            timestamp=datetime.now(),
            context={'episode': 'test', 'entities': ['queen', 'laying']},
            metadata={'test': True}
        )

        mem_id = await backend.store(test_memory, user_id="test_user")
        print(f"        Stored: {mem_id}")

        print("\n[3/3] Querying back...")
        query = MemoryQuery(
            text="queen laying",
            user_id="test_user",
            limit=5
        )

        results = await backend.recall(query, strategy=Strategy.FUSED)
        print(f"        Found {len(results.memories)} memories")
        for mem in results.memories[:3]:
            print(f"        - {mem.text[:60]}...")

        print("\n[SUCCESS] Direct backend test complete!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Direct backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nTesting HoloLoom Memory Integration")
    print("Neo4j + Qdrant hybrid backend\n")

    # Run tests
    result1 = asyncio.run(test_hybrid_memory())
    result2 = asyncio.run(test_direct_backend())

    if result1 and result2:
        print("\n" + "="*70)
        print("[ALL TESTS PASSED]")
        print("="*70 + "\n")
        sys.exit(0)
    else:
        print("\n[SOME TESTS FAILED]")
        sys.exit(1)
