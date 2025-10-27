#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Unified Memory Demo
=============================
Demonstrates dynamic memory backends with persistent storage.

Shows:
1. Static shards (backward compatibility)
2. NetworkX backend (in-memory, fast)
3. Neo4j + Qdrant backend (persistent, production)
4. Storing and querying memories dynamically
5. Lifecycle management with database connections

Requirements:
- Docker running for Neo4j + Qdrant demo
- Run: docker-compose up -d

Author: Claude Code
Date: 2025-10-26
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config, ExecutionMode, MemoryBackend
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory as BackendMemory


# ============================================================================
# Test Data
# ============================================================================

def create_test_shards():
    """Create sample memory shards for backward compatibility demo."""
    return [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="docs",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["ALGORITHM", "OPTIMIZATION"]
        ),
        MemoryShard(
            id="shard_002",
            text="The algorithm balances exploration and exploitation by sampling from posterior distributions.",
            episode="docs",
            entities=["exploration", "exploitation", "posterior"],
            motifs=["ALGORITHM", "PROBABILITY"]
        ),
        MemoryShard(
            id="shard_003",
            text="Unified memory enables persistent storage across sessions with Neo4j and Qdrant.",
            episode="system",
            entities=["unified memory", "Neo4j", "Qdrant"],
            motifs=["ARCHITECTURE", "PERSISTENCE"]
        )
    ]


def create_backend_memories():
    """Create memories for backend storage."""
    return [
        BackendMemory(
            id="mem_001",
            text="Hive Jodi has 8 frames of brood and is very active with goldenrod flow.",
            timestamp=datetime.now(),
            context={"episode": "inspection_2025_10_13", "entities": ["Hive Jodi", "brood", "goldenrod"]},
            metadata={"motifs": ["HIVE_INSPECTION", "SEASONAL"], "importance": 0.8}
        ),
        BackendMemory(
            id="mem_002",
            text="Winter preparations include wrapping hives, adding insulation, and reducing entrances.",
            timestamp=datetime.now(),
            context={"episode": "planning", "entities": ["winter", "insulation", "preparation"]},
            metadata={"motifs": ["PLANNING", "SEASONAL"], "importance": 0.9}
        ),
        BackendMemory(
            id="mem_003",
            text="Varroa mite counts were elevated in Hive Delta, treatment scheduled for next week.",
            timestamp=datetime.now(),
            context={"episode": "health_check", "entities": ["Varroa", "Hive Delta", "treatment"]},
            metadata={"motifs": ["HEALTH", "TREATMENT"], "importance": 1.0}
        ),
        BackendMemory(
            id="mem_004",
            text="Honey harvest yielded 120 pounds from 4 hives, excellent season overall.",
            timestamp=datetime.now(),
            context={"episode": "harvest_2025", "entities": ["honey", "harvest", "yield"]},
            metadata={"motifs": ["HARVEST", "SUCCESS"], "importance": 0.7}
        )
    ]


# ============================================================================
# Demo 1: Static Shards (Backward Compatibility)
# ============================================================================

async def demo_static_shards():
    """
    Demo 1: Using static shards (traditional approach).

    This demonstrates backward compatibility - existing code continues to work.
    """
    print("\n" + "="*80)
    print("DEMO 1: Static Shards (Backward Compatibility)")
    print("="*80 + "\n")

    config = Config.fast()
    shards = create_test_shards()

    print(f"[SETUP] Created {len(shards)} static shards")
    print("   Mode: In-memory (no persistence)")

    async with WeavingShuttle(cfg=config, shards=shards) as shuttle:
        query = Query(text="What is Thompson Sampling?")
        print(f"\n[QUERY] '{query.text}'")

        spacetime = await shuttle.weave(query)

        print(f"\n[RESULT]")
        print(f"   Tool: {spacetime.tool_used}")
        print(f"   Confidence: {spacetime.confidence:.2f}")
        print(f"   Context shards: {spacetime.trace.context_shards_count}")
        print(f"   Duration: {spacetime.trace.duration_ms:.1f}ms")

    print("\n[OK] Static shards demo complete")


# ============================================================================
# Demo 2: NetworkX Backend (In-Memory)
# ============================================================================

async def demo_networkx_backend():
    """
    Demo 2: Using NetworkX backend for dynamic in-memory storage.

    No Docker required - fast prototyping.
    """
    print("\n" + "="*80)
    print("DEMO 2: NetworkX Backend (Dynamic In-Memory)")
    print("="*80 + "\n")

    config = Config.fast()
    config.memory_backend = MemoryBackend.NETWORKX

    print(f"[SETUP] Backend: {config.memory_backend.value}")
    print("   Mode: Dynamic queries (no persistence)")

    # Create backend
    memory = await create_memory_backend(config)
    print(f"[OK] Backend created successfully")

    # Store some memories
    memories = create_backend_memories()
    print(f"\n[STORE] Storing {len(memories)} memories...")

    for mem in memories:
        await memory.store(mem, user_id="demo_user")

    print(f"[OK] {len(memories)} memories stored")

    # Use with WeavingShuttle
    async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
        query = Query(text="Tell me about hive inspections")
        print(f"\n[QUERY] '{query.text}'")

        spacetime = await shuttle.weave(query)

        print(f"\n[RESULT]")
        print(f"   Tool: {spacetime.tool_used}")
        print(f"   Confidence: {spacetime.confidence:.2f}")
        print(f"   Context shards: {spacetime.trace.context_shards_count}")
        print(f"   Duration: {spacetime.trace.duration_ms:.1f}ms")

        # Query again to show dynamic retrieval
        query2 = Query(text="What about winter preparations?")
        print(f"\n[QUERY] '{query2.text}'")

        spacetime2 = await shuttle.weave(query2)

        print(f"\n[RESULT]")
        print(f"   Tool: {spacetime2.tool_used}")
        print(f"   Confidence: {spacetime2.confidence:.2f}")
        print(f"   Context shards: {spacetime2.trace.context_shards_count}")

    print("\n[OK] NetworkX backend demo complete")


# ============================================================================
# Demo 3: Neo4j + Qdrant Backend (Production)
# ============================================================================

async def demo_neo4j_qdrant_backend():
    """
    Demo 3: Using Neo4j + Qdrant hybrid backend for production.

    Requires Docker: docker-compose up -d
    """
    print("\n" + "="*80)
    print("DEMO 3: Neo4j + Qdrant Backend (Production Hybrid)")
    print("="*80 + "\n")

    config = Config.fused()
    config.memory_backend = MemoryBackend.NEO4J_QDRANT

    print(f"[SETUP] Backend: {config.memory_backend.value}")
    print("   Mode: Persistent (survives restarts)")
    print(f"   Neo4j: {config.neo4j_uri}")
    print(f"   Qdrant: http://{config.qdrant_host}:{config.qdrant_port}")

    try:
        # Create backend
        memory = await create_memory_backend(config)
        print(f"\n[OK] Hybrid backend created successfully")

        # Store memories
        memories = create_backend_memories()
        print(f"\n[STORE] Storing {len(memories)} memories...")

        for mem in memories:
            await memory.store(mem, user_id="demo_user")

        print(f"[OK] {len(memories)} memories stored in Neo4j + Qdrant")

        # Use with WeavingShuttle
        async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
            # Query 1: Semantic search (Qdrant strength)
            query1 = Query(text="hive health issues")
            print(f"\n[QUERY] '{query1.text}'")

            spacetime1 = await shuttle.weave(query1)

            print(f"\n[RESULT]")
            print(f"   Tool: {spacetime1.tool_used}")
            print(f"   Confidence: {spacetime1.confidence:.2f}")
            print(f"   Context shards: {spacetime1.trace.context_shards_count}")
            print(f"   Duration: {spacetime1.trace.duration_ms:.1f}ms")

            # Query 2: Graph traversal (Neo4j strength)
            query2 = Query(text="seasonal beekeeping activities")
            print(f"\n[QUERY] '{query2.text}'")

            spacetime2 = await shuttle.weave(query2)

            print(f"\n[RESULT]")
            print(f"   Tool: {spacetime2.tool_used}")
            print(f"   Confidence: {spacetime2.confidence:.2f}")
            print(f"   Context shards: {spacetime2.trace.context_shards_count}")

            print("\n[INFO] Memories persisted - restart demo to verify!")
            print("       Data survives container restarts")

        print("\n[OK] Neo4j + Qdrant backend demo complete")

    except Exception as e:
        print(f"\n[ERROR] Failed to connect to Neo4j + Qdrant")
        print(f"        {e}")
        print("\n[FIX] Make sure Docker containers are running:")
        print("      docker-compose up -d")
        print("      docker-compose ps")


# ============================================================================
# Demo 4: Comparison (Static vs Dynamic)
# ============================================================================

async def demo_comparison():
    """
    Demo 4: Side-by-side comparison of static shards vs dynamic backend.
    """
    print("\n" + "="*80)
    print("DEMO 4: Comparison (Static Shards vs Dynamic Backend)")
    print("="*80 + "\n")

    query = Query(text="What is Thompson Sampling?")

    # Test with static shards
    print("[TEST 1] Static Shards")
    print("-" * 40)

    config1 = Config.bare()
    shards = create_test_shards()

    async with WeavingShuttle(cfg=config1, shards=shards) as shuttle1:
        import time
        start = time.time()
        spacetime1 = await shuttle1.weave(query)
        duration1 = (time.time() - start) * 1000

    print(f"   Duration: {duration1:.1f}ms")
    print(f"   Context shards: {spacetime1.trace.context_shards_count}")
    print(f"   Source: Static list")

    # Test with dynamic backend
    print("\n[TEST 2] NetworkX Backend")
    print("-" * 40)

    config2 = Config.bare()
    config2.memory_backend = MemoryBackend.NETWORKX
    memory = await create_memory_backend(config2)

    # Store shards as memories
    for shard in shards:
        mem = BackendMemory(
            id=shard.id,
            text=shard.text,
            timestamp=datetime.now(),
            context={"episode": shard.episode, "entities": shard.entities},
            metadata={"motifs": shard.motifs}
        )
        await memory.store(mem)

    async with WeavingShuttle(cfg=config2, memory=memory) as shuttle2:
        start = time.time()
        spacetime2 = await shuttle2.weave(query)
        duration2 = (time.time() - start) * 1000

    print(f"   Duration: {duration2:.1f}ms")
    print(f"   Context shards: {spacetime2.trace.context_shards_count}")
    print(f"   Source: Dynamic query")

    # Comparison
    print("\n[COMPARISON]")
    print("-" * 40)
    print(f"   Static shards: {duration1:.1f}ms")
    print(f"   Dynamic backend: {duration2:.1f}ms")
    print(f"   Overhead: {duration2 - duration1:.1f}ms")

    print("\n[OK] Comparison complete")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all unified memory demos."""
    print("\n" + "="*80)
    print("HoloLoom Unified Memory Demo")
    print("="*80)

    demos = [
        ("Static Shards", demo_static_shards),
        ("NetworkX Backend", demo_networkx_backend),
        ("Neo4j + Qdrant Backend", demo_neo4j_qdrant_backend),
        ("Comparison", demo_comparison),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n[ERROR] Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("[OK] All demos complete!")
    print("="*80 + "\n")

    print("Key Takeaways:")
    print("1. Static shards: Fast, backward compatible, no setup")
    print("2. NetworkX backend: Dynamic, in-memory, prototyping")
    print("3. Neo4j + Qdrant: Production-ready, persistent, scalable")
    print("4. Unified API: Same code works with all backends")
    print("5. Lifecycle management: Connections cleaned up automatically")

    print("\nNext Steps:")
    print("- Run with Docker: docker-compose up -d")
    print("- Explore Neo4j UI: http://localhost:7474")
    print("- Explore Qdrant: http://localhost:6333/dashboard")
    print("- Check lifecycle: All database connections closed cleanly")


if __name__ == "__main__":
    asyncio.run(main())