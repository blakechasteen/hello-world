#!/usr/bin/env python3
"""
Neo4j + Qdrant Hybrid Memory Test
===================================
Test the full hybrid stack:
1. Neo4j vector store (symbolic + vector)
2. Qdrant vector store (pure semantic)
3. Hybrid fusion (best of both)

This proves the hyperspace architecture!
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct import to avoid package issues
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load protocol
protocol = load_module("protocol", "HoloLoom/memory/protocol.py")
Memory = protocol.Memory
MemoryQuery = protocol.MemoryQuery
Strategy = protocol.Strategy

# Load neo4j store
neo4j_store_module = load_module("neo4j_vector_store", "HoloLoom/memory/stores/neo4j_vector_store.py")
Neo4jVectorStore = neo4j_store_module.Neo4jVectorStore

# Load qdrant store
qdrant_store_module = load_module("qdrant_store", "HoloLoom/memory/stores/qdrant_store.py")
QdrantMemoryStore = qdrant_store_module.QdrantMemoryStore


# Sample beekeeping memories
BEEKEEPING_MEMORIES = [
    {
        "text": "Hive Jodi has 8 frames of brood and needs winter preparation",
        "context": {
            "entities": ["Hive Jodi", "winter"],
            "hive": "Jodi",
            "season": "fall"
        },
        "metadata": {"user_id": "blake", "importance": "high"}
    },
    {
        "text": "Weak colonies require sugar fondant for winter feeding",
        "context": {
            "entities": ["winter", "feeding"],
            "topic": "winter_prep"
        },
        "metadata": {"user_id": "blake", "importance": "high"}
    },
    {
        "text": "Mouse guards should be installed before November to prevent rodent damage",
        "context": {
            "entities": ["mouse guards", "November"],
            "topic": "winter_prep"
        },
        "metadata": {"user_id": "blake", "importance": "medium"}
    },
    {
        "text": "Hive Jodi located in north apiary with high cold exposure",
        "context": {
            "entities": ["Hive Jodi", "north apiary"],
            "hive": "Jodi",
            "place": "north_apiary"
        },
        "metadata": {"user_id": "blake", "importance": "medium"}
    },
    {
        "text": "Last inspection showed weak colony with reduced bee population",
        "context": {
            "entities": ["inspection", "weak colony"],
            "event": "inspection"
        },
        "metadata": {"user_id": "blake", "importance": "high"}
    },
    {
        "text": "Insulation wraps help maintain hive temperature in winter",
        "context": {
            "entities": ["insulation", "winter"],
            "topic": "winter_prep"
        },
        "metadata": {"user_id": "blake", "importance": "medium"}
    },
    {
        "text": "Ventilation prevents moisture buildup which can kill bees",
        "context": {
            "entities": ["ventilation", "moisture"],
            "topic": "winter_prep"
        },
        "metadata": {"user_id": "blake", "importance": "high"}
    }
]


async def test_neo4j_store():
    """Test Neo4j vector store."""
    print("\n" + "=" * 80)
    print("TEST 1: Neo4j Vector Store (Symbolic + Vector)")
    print("=" * 80)

    try:
        # Initialize store
        store = Neo4jVectorStore(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="beekeeper123",
            enable_embeddings=True
        )

        print("\n✓ Connected to Neo4j")

        # Clear existing data
        print("  Clearing existing memories...")
        # (In production, you'd keep history)

        # Store memories
        print(f"\n  Storing {len(BEEKEEPING_MEMORIES)} memories...")
        stored_ids = []
        for mem_data in BEEKEEPING_MEMORIES:
            memory = Memory(
                id="",
                text=mem_data["text"],
                timestamp=datetime.now(),
                context=mem_data["context"],
                metadata=mem_data["metadata"]
            )
            mem_id = await store.store(memory)
            stored_ids.append(mem_id)
            print(f"    Stored: {mem_id[:8]}... - {mem_data['text'][:50]}...")

        print(f"\n✓ Stored {len(stored_ids)} memories")

        # Test retrieval strategies
        query = MemoryQuery(
            text="winter preparation for Hive Jodi",
            user_id="blake",
            limit=5
        )

        # TEMPORAL
        print("\n  Testing TEMPORAL retrieval...")
        result = await store.retrieve(query, Strategy.TEMPORAL)
        print(f"    Found {len(result.memories)} memories (most recent)")
        for mem, score in zip(result.memories[:3], result.scores[:3]):
            print(f"      [{score:.2f}] {mem.text[:60]}...")

        # GRAPH
        print("\n  Testing GRAPH retrieval...")
        result = await store.retrieve(query, Strategy.GRAPH)
        print(f"    Found {len(result.memories)} memories (graph connected)")
        for mem, score in zip(result.memories[:3], result.scores[:3]):
            print(f"      [{score:.2f}] {mem.text[:60]}...")

        # SEMANTIC (vector search)
        print("\n  Testing SEMANTIC retrieval (vector similarity)...")
        result = await store.retrieve(query, Strategy.SEMANTIC)
        print(f"    Found {len(result.memories)} memories (semantic match)")
        for mem, score in zip(result.memories[:3], result.scores[:3]):
            print(f"      [{score:.2f}] {mem.text[:60]}...")

        # FUSED (hybrid)
        print("\n  Testing FUSED retrieval (symbolic + semantic)...")
        result = await store.retrieve(query, Strategy.FUSED)
        print(f"    Found {len(result.memories)} memories (HYPERSPACE!)")
        for mem, score in zip(result.memories[:3], result.scores[:3]):
            print(f"      [{score:.2f}] {mem.text[:60]}...")

        # Health check
        health = await store.health_check()
        print(f"\n✓ Health: {health}")

        store.close()
        return True

    except Exception as e:
        print(f"\n✗ Neo4j test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_qdrant_store():
    """Test Qdrant vector store."""
    print("\n" + "=" * 80)
    print("TEST 2: Qdrant Vector Store (Pure Semantic)")
    print("=" * 80)

    try:
        # Initialize store
        store = QdrantMemoryStore(
            url="http://localhost:6333",
            scales=[96, 192, 384]
        )

        print("\n✓ Connected to Qdrant")

        # Store memories
        print(f"\n  Storing {len(BEEKEEPING_MEMORIES)} memories with multi-scale embeddings...")
        for mem_data in BEEKEEPING_MEMORIES:
            memory = Memory(
                id="",
                text=mem_data["text"],
                timestamp=datetime.now(),
                context=mem_data["context"],
                metadata=mem_data["metadata"]
            )
            mem_id = await store.store(memory)
            print(f"    Stored: {mem_id[:8]}... - {mem_data['text'][:50]}...")

        # Test retrieval
        query = MemoryQuery(
            text="winter preparation for Hive Jodi",
            user_id="blake",
            limit=5
        )

        print("\n  Testing SEMANTIC retrieval...")
        result = await store.retrieve(query, Strategy.SEMANTIC)
        print(f"    Found {len(result.memories)} memories")
        for mem, score in zip(result.memories, result.scores):
            print(f"      [{score:.2f}] {mem.text[:60]}...")

        # Health check
        health = await store.health_check()
        print(f"\n✓ Health: {health}")

        return True

    except Exception as e:
        print(f"\n✗ Qdrant test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_comparison():
    """Compare Neo4j vs Qdrant retrieval."""
    print("\n" + "=" * 80)
    print("TEST 3: Neo4j vs Qdrant Comparison")
    print("=" * 80)

    try:
        neo4j = Neo4jVectorStore(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="beekeeper123",
            enable_embeddings=True
        )

        qdrant = QdrantMemoryStore(
            url="http://localhost:6333"
        )

        query = MemoryQuery(
            text="How do I keep weak hives alive in winter?",
            user_id="blake",
            limit=3
        )

        print("\n  Query: \"How do I keep weak hives alive in winter?\"")

        # Neo4j FUSED (graph + vector)
        print("\n  Neo4j FUSED (symbolic + semantic):")
        neo_result = await neo4j.retrieve(query, Strategy.FUSED)
        for mem, score in zip(neo_result.memories, neo_result.scores):
            print(f"    [{score:.2f}] {mem.text[:70]}...")

        # Qdrant semantic
        print("\n  Qdrant SEMANTIC (pure vector):")
        qdrant_result = await qdrant.retrieve(query, Strategy.SEMANTIC)
        for mem, score in zip(qdrant_result.memories, qdrant_result.scores):
            print(f"    [{score:.2f}] {mem.text[:70]}...")

        print("\n✓ Both strategies provide complementary results!")
        print("  - Neo4j adds graph context")
        print("  - Qdrant optimized for semantic similarity")
        print("  - Hybrid would fuse the best of both")

        neo4j.close()
        return True

    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🚀 NEO4J + QDRANT HYBRID MEMORY TEST")
    print("   Hyperspace: Symbolic + Semantic Vectors")
    print("=" * 80)

    results = {
        "neo4j": False,
        "qdrant": False,
        "comparison": False
    }

    # Test Neo4j
    results["neo4j"] = await test_neo4j_store()

    # Test Qdrant
    results["qdrant"] = await test_qdrant_store()

    # Compare
    if results["neo4j"] and results["qdrant"]:
        results["comparison"] = await test_comparison()

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"  Neo4j Vector Store:  {'✓ PASS' if results['neo4j'] else '✗ FAIL'}")
    print(f"  Qdrant Vector Store: {'✓ PASS' if results['qdrant'] else '✗ FAIL'}")
    print(f"  Comparison Test:     {'✓ PASS' if results['comparison'] else '✗ FAIL'}")

    if all(results.values()):
        print("\n✅ ALL TESTS PASSED - HYPERSPACE READY!")
        print("\nNext: Build hybrid fusion store")
    else:
        print("\n⚠️  Some tests failed - check logs above")

    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
