#!/usr/bin/env python3
"""
Quick Status Check - HoloLoom Hybrid Memory
============================================
Run this anytime to verify your memory system is healthy
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.memory.stores.hybrid_neo4j_qdrant import HybridNeo4jQdrant


async def quick_status():
    """Quick health check."""
    print("\n" + "="*60)
    print("🧠 HoloLoom Hybrid Memory - Quick Status")
    print("="*60)
    
    try:
        store = HybridNeo4jQdrant()
        
        # Neo4j
        with store.neo4j_driver.session() as session:
            neo4j_count = session.run("MATCH (n) RETURN count(n) as total").single()["total"]
        
        # Qdrant
        collection_info = store.qdrant_client.get_collection("hololoom_memories")
        qdrant_count = collection_info.points_count
        
        # Health
        health = await store.health_check()
        
        print(f"\n✅ Status: {health['status'].upper()}")
        print(f"📊 Neo4j nodes: {neo4j_count}")
        print(f"🔍 Qdrant vectors: {qdrant_count}")
        print(f"⚡ Backend: Neo4j + Qdrant (Hybrid)")
        print(f"\n🎯 Ready for Claude Desktop MCP!")
        
        store.neo4j_driver.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check Docker containers: docker ps")
        print("   2. Verify Neo4j: docker logs hololoom-neo4j")
        print("   3. Verify Qdrant: docker logs qdrant")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(quick_status())
