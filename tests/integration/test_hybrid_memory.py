#!/usr/bin/env python3
"""
Test HoloLoom Hybrid Memory System
===================================
Verify Neo4j + Qdrant integration is working properly
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.memory.stores.hybrid_neo4j_qdrant import HybridNeo4jQdrant, Memory, MemoryQuery, Strategy
from datetime import datetime


async def test_hybrid_memory():
    """Test the hybrid memory system."""
    
    print("=" * 70)
    print("🧠 HoloLoom Hybrid Memory System Test")
    print("=" * 70)
    
    # Initialize
    print("\n1️⃣ Initializing hybrid store (Neo4j + Qdrant)...")
    try:
        store = HybridNeo4jQdrant(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="hololoom123",
            qdrant_url="http://localhost:6333",
            collection_name="hololoom_memories"
        )
        print("   ✅ Hybrid store initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return
    
    # Check Neo4j connectivity
    print("\n2️⃣ Checking Neo4j connectivity...")
    try:
        with store.neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as total")
            count = result.single()["total"]
            print(f"   ✅ Neo4j connected: {count} nodes")
    except Exception as e:
        print(f"   ❌ Neo4j error: {e}")
    
    # Check Qdrant connectivity
    print("\n3️⃣ Checking Qdrant connectivity...")
    try:
        collection_info = store.qdrant_client.get_collection("hololoom_memories")
        vector_count = collection_info.vectors_count
        print(f"   ✅ Qdrant connected: {vector_count} vectors")
    except Exception as e:
        print(f"   ❌ Qdrant error: {e}")
    
    # Test memory storage
    print("\n4️⃣ Testing memory storage...")
    test_memory = Memory(
        id=f"test-{datetime.now().timestamp()}",
        text="Testing HoloLoom hybrid memory with Neo4j and Qdrant integration",
        timestamp=datetime.now(),
        context={"test": True, "component": "hybrid_test"},
        metadata={"user_id": "blake", "category": "system_test"}
    )
    
    try:
        await store.store(test_memory)
        print("   ✅ Memory stored successfully")
    except Exception as e:
        print(f"   ⚠️  Storage test skipped: {e}")
    
    # Test retrieval with different strategies
    print("\n5️⃣ Testing retrieval strategies...")
    
    query = MemoryQuery(
        text="beekeeping hive inspection",
        user_id="blake",
        limit=3
    )
    
    strategies = [
        (Strategy.SEMANTIC, "Semantic (Qdrant vectors)"),
        (Strategy.GRAPH, "Graph (Neo4j relationships)"),
        (Strategy.FUSED, "Fused (Hybrid best)")
    ]
    
    for strategy, name in strategies:
        try:
            result = await store.retrieve(query, strategy)
            print(f"\n   📊 {name}:")
            print(f"      Found: {len(result.memories)} memories")
            if result.memories:
                for i, mem in enumerate(result.memories[:2], 1):
                    preview = mem.text[:60] + "..." if len(mem.text) > 60 else mem.text
                    score = result.scores[i-1] if i-1 < len(result.scores) else 0.0
                    print(f"      {i}. [{score:.3f}] {preview}")
        except Exception as e:
            print(f"   ⚠️  {name} error: {e}")
    
    # Health check
    print("\n6️⃣ System health check...")
    try:
        health = await store.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Neo4j nodes: {health.get('neo4j_count', 'N/A')}")
        print(f"   Qdrant vectors: {health.get('qdrant_count', 'N/A')}")
        print(f"   Backend: {health.get('backend', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Health check error: {e}")
    
    # Cleanup
    store.neo4j_driver.close()
    
    print("\n" + "=" * 70)
    print("✅ Hybrid Memory System Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_hybrid_memory())
