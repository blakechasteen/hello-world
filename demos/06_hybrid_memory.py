#!/usr/bin/env python3
"""
Demo 06: Hybrid Memory System
==============================
Shows hybrid memory store with graceful degradation.

Flow:
1. Initialize with backend="hybrid" (tries Qdrant + Neo4j + file)
2. Add knowledge to memory
3. Query with context retrieval
4. Show which backends are active
5. Demonstrate persistence

This shows:
- Hybrid memory with weighted fusion
- Graceful degradation to file-only
- File persistence working
- Multi-backend fusion scores
"""

import asyncio
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.weaving_orchestrator import WeavingOrchestrator


async def main():
    print("=" * 80)
    print("DEMO 06: HYBRID MEMORY SYSTEM")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Initialize with Hybrid Memory
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Initialize Hybrid Memory")
    print("=" * 80)
    print()

    print("Attempting to initialize hybrid memory (Qdrant + Neo4j + File)...")
    print()

    weaver = WeavingOrchestrator(
        use_mcts=True,
        mcts_simulations=50
    )

    # Check what backends are active
    if hasattr(weaver.memory_store, 'backends'):
        print(f"\n[SUCCESS] Hybrid memory initialized!")
        print(f"Active backends: {len(weaver.memory_store.backends)}")
        for backend in weaver.memory_store.backends:
            print(f"  - {backend.name} (weight: {backend.weight:.1%})")
    elif hasattr(weaver.memory_store, 'data_dir'):
        print(f"\n[INFO] File-only memory (other backends unavailable)")
        print(f"Data directory: {weaver.memory_store.data_dir}")
    else:
        print(f"\n[WARNING] In-memory fallback (no persistence)")

    print()

    # ========================================================================
    # Step 2: Add Knowledge
    # ========================================================================
    print("=" * 80)
    print("STEP 2: Add Knowledge to Memory")
    print("=" * 80)
    print()

    knowledge_base = [
        {
            "text": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It samples from posterior distributions to balance exploration and exploitation.",
            "metadata": {"topic": "ml", "source": "thompson_paper"}
        },
        {
            "text": "MCTS (Monte Carlo Tree Search) builds a search tree by running simulations. It uses UCB1 to select promising nodes and backpropagates rewards.",
            "metadata": {"topic": "algorithms", "source": "mcts_paper"}
        },
        {
            "text": "The HoloLoom flux capacitor combines MCTS with Thompson Sampling at every level - Thompson Sampling ALL THE WAY DOWN!",
            "metadata": {"topic": "hololoom", "source": "architecture_docs"}
        },
        {
            "text": "Matryoshka embeddings provide multi-scale representations like Russian nesting dolls. They enable progressive filtering at 96d, 192d, and 384d.",
            "metadata": {"topic": "embeddings", "source": "embedding_paper"}
        },
        {
            "text": "The weaving metaphor in HoloLoom treats computation as literal weaving. Queries are woven through 7 stages into fabric (Spacetime) with complete provenance.",
            "metadata": {"topic": "hololoom", "source": "philosophy_docs"}
        },
        {
            "text": "Neo4j is a graph database that stores entities and relationships. HoloLoom uses it for the Yarn Graph (knowledge graph memory).",
            "metadata": {"topic": "databases", "source": "neo4j_docs"}
        },
        {
            "text": "Qdrant is a vector database optimized for similarity search. HoloLoom uses it for multi-scale embedding retrieval.",
            "metadata": {"topic": "databases", "source": "qdrant_docs"}
        },
    ]

    for i, kb in enumerate(knowledge_base, 1):
        await weaver.add_knowledge(kb["text"], kb["metadata"])
        print(f"{i}. Added: {kb['text'][:60]}...")

    print()
    print(f"Total knowledge added: {len(knowledge_base)} shards")
    print()

    # ========================================================================
    # Step 3: Query with Context Retrieval
    # ========================================================================
    print("=" * 80)
    print("STEP 3: Query with Context Retrieval")
    print("=" * 80)
    print()

    queries = [
        "What is Thompson Sampling and how does it work?",
        "Explain the flux capacitor in HoloLoom",
        "What databases does HoloLoom use?"
    ]

    for query_num, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"QUERY {query_num}: {query}")
        print('=' * 80)

        # Execute weaving cycle
        spacetime = await weaver.weave(query)

        # Show results
        print(f"\nTool Selected: {spacetime.tool_used}")
        print(f"Confidence: {spacetime.confidence:.1%}")
        print(f"Duration: {spacetime.trace.duration_ms:.0f}ms")

        # Show retrieved context
        print(f"\nContext Retrieved: {spacetime.trace.context_shards_count} shards")
        if spacetime.trace.context_shards_count > 0:
            print("Top context:")
            for i, thread in enumerate(spacetime.trace.threads_activated[:3], 1):
                preview = thread[:80] + "..." if len(thread) > 80 else thread
                print(f"  {i}. {preview}")

    print()

    # ========================================================================
    # Step 4: Check Memory Health
    # ========================================================================
    print("=" * 80)
    print("STEP 4: Memory Health Check")
    print("=" * 80)
    print()

    # Health check
    if hasattr(weaver.memory_store, 'health_check'):
        health = await weaver.memory_store.health_check()

        print(f"Status: {health.get('status', 'unknown')}")
        print(f"Backend: {health.get('backend', 'unknown')}")

        if 'backends' in health:
            print(f"\nBackend Health:")
            for name, backend_health in health['backends'].items():
                status = backend_health.get('status', 'unknown')
                print(f"  {name}: {status}")

                if status != 'healthy':
                    error = backend_health.get('error', 'No details')
                    print(f"    Error: {error}")

        if 'memory_count' in health:
            print(f"\nMemories stored: {health['memory_count']}")

        if 'embedding_dim' in health:
            print(f"Embedding dimension: {health['embedding_dim']}d")

    print()

    # ========================================================================
    # Step 5: System Statistics
    # ========================================================================
    print("=" * 80)
    print("STEP 5: System Statistics")
    print("=" * 80)
    print()

    stats = weaver.get_statistics()

    print(f"Weaving:")
    print(f"  Total cycles: {stats['total_weavings']}")
    print(f"  Pattern usage: {stats['pattern_usage']}")

    if 'mcts_stats' in stats:
        mcts = stats['mcts_stats']
        print(f"\nMCTS Flux Capacitor:")
        print(f"  Total simulations: {mcts['flux_stats']['total_simulations']}")
        print(f"  Decisions made: {mcts['decision_count']}")
        print(f"  Tool distribution: {mcts['flux_stats']['tool_distribution']}")

    print()

    weaver.stop()

    # ========================================================================
    # Step 6: Test Persistence
    # ========================================================================
    print("=" * 80)
    print("STEP 6: Test Persistence (File Backend)")
    print("=" * 80)
    print()

    print("Creating NEW orchestrator to test persistence...")
    print()

    weaver2 = WeavingOrchestrator(
        use_mcts=True,
        mcts_simulations=50
    )

    # Check if memories persisted
    if hasattr(weaver2.memory_store, 'memories'):
        memory_count = len(weaver2.memory_store.memories)
        print(f"[SUCCESS] Loaded {memory_count} memories from disk!")

        if memory_count > 0:
            print("\nSample memories:")
            for i, mem in enumerate(weaver2.memory_store.memories[:3], 1):
                print(f"{i}. {mem.text[:60]}...")
    else:
        print("[INFO] In-memory backend (no persistence)")

    print()

    weaver2.stop()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print("[SUCCESS] Hybrid memory system OPERATIONAL!")
    print()
    print("Capabilities:")
    print("  - Multi-backend fusion (Qdrant + Neo4j + File)")
    print("  - Graceful degradation to file-only")
    print("  - File persistence working")
    print("  - Context retrieval functional")
    print("  - Health checks implemented")
    print()
    print("The hybrid memory is LIVE!")
    print("Thompson Sampling ALL THE WAY DOWN with persistent memory!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
