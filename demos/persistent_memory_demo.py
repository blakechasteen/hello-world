#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent Memory Integration Demo
===================================

Demonstrates all three memory integration options with WeavingShuttle:

Option 1: In-memory (legacy) - Fast, testing
Option 2: UnifiedMemory - Intelligent extraction
Option 3: Backend Factory (Hybrid) - Production with Neo4j + Qdrant

Shows:
- Memory persistence across sessions
- Seamless switching between backends
- ChatOps integration
- Performance comparison
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.memory.weaving_adapter import (
    WeavingMemoryAdapter,
    create_weaving_memory
)


# ============================================================================
# Demo Data
# ============================================================================

DEMO_SHARDS = [
    MemoryShard(
        id="shard_001",
        text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",
        episode="ml_docs",
        entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
        motifs=["ALGORITHM", "OPTIMIZATION"]
    ),
    MemoryShard(
        id="shard_002",
        text="The HoloLoom weaving architecture implements a complete 9-step cycle from pattern selection to Spacetime fabric.",
        episode="hololoom_docs",
        entities=["HoloLoom", "weaving", "Spacetime"],
        motifs=["ARCHITECTURE", "SYSTEM"]
    ),
    MemoryShard(
        id="shard_003",
        text="Matryoshka embeddings allow multi-scale retrieval at 96d, 192d, and 384d resolutions for flexible precision-speed tradeoffs.",
        episode="ml_docs",
        entities=["Matryoshka", "embeddings", "multi-scale"],
        motifs=["ALGORITHM", "EMBEDDINGS"]
    )
]


# ============================================================================
# Option 1: In-Memory Mode (Legacy)
# ============================================================================

async def demo_option_1_in_memory():
    """
    Option 1: In-memory mode (legacy)

    Fastest, good for testing and development.
    Memory is lost when process ends.
    """
    print("\n" + "="*80)
    print("OPTION 1: IN-MEMORY MODE (Legacy)")
    print("="*80)

    # Create config
    config = Config.fast()

    # Option 1a: Direct shards (legacy)
    print("\n[1a] Using direct shards...")
    shuttle = WeavingShuttle(
        cfg=config,
        shards=DEMO_SHARDS,
        enable_reflection=True
    )

    # Query
    query = Query(text="What is Thompson Sampling?")
    spacetime = await shuttle.weave(query)

    print(f"  Tool: {spacetime.tool_used}")
    print(f"  Confidence: {spacetime.confidence:.2%}")
    print(f"  Duration: {spacetime.trace.duration_ms:.0f}ms")
    print(f"  Shards retrieved: {spacetime.trace.context_shards_count}")

    # Option 1b: Using memory adapter (recommended)
    print("\n[1b] Using memory adapter (recommended)...")
    memory = create_weaving_memory("in_memory", shards=DEMO_SHARDS)

    shuttle2 = WeavingShuttle(
        cfg=config,
        memory=memory,
        enable_reflection=True
    )

    spacetime2 = await shuttle2.weave(query)
    print(f"  Tool: {spacetime2.tool_used}")
    print(f"  Confidence: {spacetime2.confidence:.2%}")
    print(f"  Duration: {spacetime2.trace.duration_ms:.0f}ms")

    await shuttle.close()
    await shuttle2.close()

    print("\n✅ Option 1 complete - In-memory mode works!")


# ============================================================================
# Option 2: UnifiedMemory Backend
# ============================================================================

async def demo_option_2_unified_memory():
    """
    Option 2: UnifiedMemory backend

    Intelligent memory extraction with optional Neo4j/Qdrant.
    Good balance of features and simplicity.
    """
    print("\n" + "="*80)
    print("OPTION 2: UNIFIED MEMORY BACKEND")
    print("="*80)

    # Create unified memory adapter
    print("\nInitializing UnifiedMemory...")
    try:
        memory = create_weaving_memory(
            mode="unified",
            user_id="demo_user",
            enable_neo4j=False,  # Set to True if Neo4j running
            enable_qdrant=False,  # Set to True if Qdrant running
            enable_mem0=False,    # Set to True if Mem0 configured
            enable_hofstadter=False
        )

        # Store demo data
        print("Storing demo memories...")
        for shard in DEMO_SHARDS:
            await memory.add_shard(shard)

        # Create shuttle with unified memory
        config = Config.fast()
        shuttle = WeavingShuttle(
            cfg=config,
            memory=memory,
            enable_reflection=True
        )

        # Query
        query = Query(text="Explain Matryoshka embeddings")
        spacetime = await shuttle.weave(query)

        print(f"\n  Tool: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.2%}")
        print(f"  Duration: {spacetime.trace.duration_ms:.0f}ms")
        print(f"  Memory stats: {memory.get_statistics()}")

        await shuttle.close()

        print("\n✅ Option 2 complete - UnifiedMemory backend works!")

    except Exception as e:
        print(f"\n⚠️  Option 2 skipped: {e}")
        print("   (UnifiedMemory may not be fully implemented yet)")


# ============================================================================
# Option 3: Backend Factory (Production)
# ============================================================================

async def demo_option_3_backend_factory():
    """
    Option 3: Backend factory with Neo4j + Qdrant

    Production-ready persistent storage.
    Requires Docker containers running.
    """
    print("\n" + "="*80)
    print("OPTION 3: BACKEND FACTORY (Production)")
    print("="*80)

    print("\n⚠️  This option requires Docker containers:")
    print("   - Neo4j: docker run -p 7687:7687 -p 7474:7474 neo4j:latest")
    print("   - Qdrant: docker run -p 6333:6333 qdrant/qdrant")

    try:
        # Create hybrid backend
        print("\nInitializing hybrid backend...")
        memory = create_weaving_memory(
            mode="hybrid",
            neo4j_config={
                'url': 'bolt://localhost:7687',
                'user': 'neo4j',
                'password': 'password'
            },
            qdrant_config={
                'url': 'http://localhost:6333'
            }
        )

        # Store demo data
        print("Storing demo memories in persistent storage...")
        for shard in DEMO_SHARDS:
            await memory.add_shard(shard)

        # Create shuttle
        config = Config.fused()  # Use fused mode for production
        shuttle = WeavingShuttle(
            cfg=config,
            memory=memory,
            enable_reflection=True
        )

        # Query
        query = Query(text="What is the weaving architecture?")
        spacetime = await shuttle.weave(query)

        print(f"\n  Tool: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.2%}")
        print(f"  Duration: {spacetime.trace.duration_ms:.0f}ms")
        print(f"  Memory stats: {memory.get_statistics()}")

        await shuttle.close()

        print("\n✅ Option 3 complete - Backend factory works!")
        print("   Memory persisted to Neo4j + Qdrant!")

    except Exception as e:
        print(f"\n⚠️  Option 3 skipped: {e}")
        print("   (Make sure Docker containers are running)")


# ============================================================================
# Performance Comparison
# ============================================================================

async def compare_performance():
    """
    Compare performance across all three options.
    """
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)

    import time

    query = Query(text="What is Thompson Sampling?")
    results = {}

    # Option 1: In-memory
    print("\nBenchmarking in-memory...")
    memory1 = create_weaving_memory("in_memory", shards=DEMO_SHARDS)
    shuttle1 = WeavingShuttle(cfg=Config.fast(), memory=memory1, enable_reflection=False)

    start = time.perf_counter()
    spacetime1 = await shuttle1.weave(query)
    duration1 = (time.perf_counter() - start) * 1000
    results['in_memory'] = duration1
    await shuttle1.close()

    print(f"  In-memory: {duration1:.0f}ms")

    # Option 2: UnifiedMemory
    try:
        print("\nBenchmarking UnifiedMemory...")
        memory2 = create_weaving_memory("unified", user_id="bench")
        for shard in DEMO_SHARDS:
            await memory2.add_shard(shard)

        shuttle2 = WeavingShuttle(cfg=Config.fast(), memory=memory2, enable_reflection=False)

        start = time.perf_counter()
        spacetime2 = await shuttle2.weave(query)
        duration2 = (time.perf_counter() - start) * 1000
        results['unified'] = duration2
        await shuttle2.close()

        print(f"  UnifiedMemory: {duration2:.0f}ms")
    except Exception as e:
        print(f"  UnifiedMemory: Skipped ({e})")

    # Summary
    print("\n" + "-"*80)
    print("Summary:")
    for mode, ms in results.items():
        print(f"  {mode:15s}: {ms:6.0f}ms")

    fastest = min(results.items(), key=lambda x: x[1])
    print(f"\n  Fastest: {fastest[0]} ({fastest[1]:.0f}ms)")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all demos."""
    print("\n" + "#"*80)
    print("# HoloLoom Persistent Memory Integration Demo")
    print("#"*80)

    # Run all options
    await demo_option_1_in_memory()
    await demo_option_2_unified_memory()
    await demo_option_3_backend_factory()

    # Performance comparison
    await compare_performance()

    print("\n" + "#"*80)
    print("# Demo Complete!")
    print("#"*80)
    print("\nNext steps:")
    print("  1. Start Docker containers for production mode")
    print("  2. Update ChatOps to use persistent memory")
    print("  3. Migrate existing shards to Neo4j + Qdrant")
    print("  4. Monitor performance and scale as needed")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())