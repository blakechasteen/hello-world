"""
Complete Memory System Example
===============================
Demonstrates protocol-based memory architecture with:
- Mem0: User-specific intelligent extraction
- Neo4j: Thread-based graph storage
- Qdrant: Multi-scale vector search
- Hybrid: Weighted fusion of all three

Simple Query Example:
    memory = await create_unified_memory(user_id="blake")
    
    # Store
    await memory.store("Hive Jodi needs winter prep", 
                       context={'place': 'apiary', 'time': 'evening'})
    
    # Recall
    results = await memory.recall("winter beekeeping", strategy=Strategy.FUSED)
"""

import asyncio
import logging
from datetime import datetime

# Protocol and stores
from HoloLoom.memory.protocol import (
    UnifiedMemoryInterface,
    Memory,
    Strategy,
    create_unified_memory
)

from HoloLoom.memory.stores import InMemoryStore

# Optional: Try to import production backends
try:
    from HoloLoom.memory.stores.mem0_store import Mem0MemoryStore
    _HAVE_MEM0 = True
except ImportError:
    _HAVE_MEM0 = False

try:
    from HoloLoom.memory.stores.neo4j_store import Neo4jMemoryStore
    _HAVE_NEO4J = True
except ImportError:
    _HAVE_NEO4J = False

try:
    from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore
    _HAVE_QDRANT = True
except ImportError:
    _HAVE_QDRANT = False

try:
    from HoloLoom.memory.stores.hybrid_store import HybridMemoryStore, BackendConfig
    _HAVE_HYBRID = True
except ImportError:
    _HAVE_HYBRID = False


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Simple In-Memory Store (No Dependencies)
# ============================================================================

async def example_1_in_memory():
    """
    Simplest example - pure Python, no external dependencies.
    Good for: Testing, development, learning the API.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: In-Memory Store (No External Dependencies)")
    print("="*70 + "\n")
    
    # Create store
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(store=store)
    
    print("1. Storing memories...")
    # Store some memories
    mem1 = await memory.store(
        "Hive Jodi has 8 frames of brood, very active",
        context={'place': 'apiary', 'time': 'morning'},
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem1}")
    
    mem2 = await memory.store(
        "Need to prep hives for winter - add insulation",
        context={'place': 'apiary', 'time': 'evening'},
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem2}")
    
    mem3 = await memory.store(
        "Harvested 2 gallons of honey from Hive Matriarch",
        context={'place': 'apiary', 'time': 'afternoon'},
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem3}")
    
    print("\n2. Recalling memories with different strategies...")
    
    # Semantic search
    print("\n   Strategy: SEMANTIC (text similarity)")
    results = await memory.recall("winter preparation", strategy=Strategy.SEMANTIC, user_id="blake")
    for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
        print(f"     [{i}] Score: {score:.3f} | {mem.text[:60]}...")
    
    # Temporal search
    print("\n   Strategy: TEMPORAL (most recent)")
    results = await memory.recall("hive", strategy=Strategy.TEMPORAL, user_id="blake")
    for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
        print(f"     [{i}] Score: {score:.3f} | {mem.text[:60]}...")
    
    # Fused search
    print("\n   Strategy: FUSED (combined)")
    results = await memory.recall("honey bees", strategy=Strategy.FUSED, user_id="blake")
    for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
        print(f"     [{i}] Score: {score:.3f} | {mem.text[:60]}...")
    
    print("\n3. Health check...")
    health = await memory.health_check()
    print(f"   Status: {health['store']['status']}")
    print(f"   Memory count: {health['store']['memory_count']}")
    
    print("\n✓ Example 1 complete!\n")


# ============================================================================
# Example 2: Hybrid Store (Mem0 + Neo4j + Qdrant)
# ============================================================================

async def example_2_hybrid():
    """
    Production example with all three backends.
    
    Requires:
    - pip install mem0ai
    - pip install neo4j
    - pip install qdrant-client sentence-transformers
    - Running Neo4j instance (localhost:7687)
    - Running Qdrant instance (localhost:6333)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Hybrid Store (Mem0 + Neo4j + Qdrant)")
    print("="*70 + "\n")
    
    # Check dependencies
    available = []
    if _HAVE_MEM0:
        available.append("Mem0")
    if _HAVE_NEO4J:
        available.append("Neo4j")
    if _HAVE_QDRANT:
        available.append("Qdrant")
    
    if not available:
        print("❌ No production backends available!")
        print("\nTo run this example, install:")
        print("  pip install mem0ai neo4j qdrant-client sentence-transformers")
        print("\nAnd ensure Neo4j and Qdrant are running.")
        return
    
    print(f"Available backends: {', '.join(available)}\n")
    
    # Build backend list
    backends = []
    
    if _HAVE_MEM0:
        try:
            mem0_store = Mem0MemoryStore(user_id="blake")
            backends.append(BackendConfig(
                store=mem0_store,
                weight=0.3,
                name="mem0"
            ))
            print("✓ Mem0 backend initialized")
        except Exception as e:
            print(f"⚠ Mem0 initialization failed: {e}")
    
    if _HAVE_NEO4J:
        try:
            neo4j_store = Neo4jMemoryStore(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password"
            )
            backends.append(BackendConfig(
                store=neo4j_store,
                weight=0.3,
                name="neo4j"
            ))
            print("✓ Neo4j backend initialized")
        except Exception as e:
            print(f"⚠ Neo4j initialization failed: {e}")
    
    if _HAVE_QDRANT:
        try:
            qdrant_store = QdrantMemoryStore(
                url="http://localhost:6333"
            )
            backends.append(BackendConfig(
                store=qdrant_store,
                weight=0.4,
                name="qdrant"
            ))
            print("✓ Qdrant backend initialized")
        except Exception as e:
            print(f"⚠ Qdrant initialization failed: {e}")
    
    if not backends:
        print("\n❌ No backends successfully initialized")
        return
    
    # Create hybrid store
    hybrid = HybridMemoryStore(backends=backends, fusion_method="weighted")
    memory = UnifiedMemoryInterface(store=hybrid)
    
    print(f"\n1. Storing memories across {len(backends)} backends...")
    
    mem1 = await memory.store(
        "Inspected Hive Jodi - 8 frames of brood, 3 frames honey, queen active",
        context={
            'place': 'apiary',
            'time': 'morning',
            'people': ['Blake'],
            'topics': ['beekeeping', 'inspection', 'brood']
        },
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem1}")
    
    mem2 = await memory.store(
        "Winter prep checklist: insulation, entrance reducers, mouse guards, candy boards",
        context={
            'place': 'apiary',
            'time': 'evening',
            'people': ['Blake'],
            'topics': ['beekeeping', 'winter', 'preparation']
        },
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem2}")
    
    mem3 = await memory.store(
        "Harvested 2 gallons honey from Hive Matriarch. Excellent flow this year.",
        context={
            'place': 'apiary',
            'time': 'afternoon',
            'people': ['Blake'],
            'topics': ['beekeeping', 'harvest', 'honey']
        },
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem3}")
    
    # Small delay for backends to index
    await asyncio.sleep(2)
    
    print("\n2. Hybrid recall with weighted fusion...")
    results = await memory.recall(
        "how should I prepare hives for winter?",
        strategy=Strategy.FUSED,
        user_id="blake"
    )
    
    print(f"\n   Found {len(results.memories)} memories:")
    print(f"   Fusion method: {results.metadata['fusion_method']}")
    print(f"   Backends used: {results.metadata.get('backends_used', [])}")
    
    for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
        sources = mem.metadata.get('fusion_sources', ['unknown'])
        print(f"\n   [{i}] Score: {score:.3f} | Sources: {sources}")
        print(f"       {mem.text[:80]}...")
    
    print("\n3. Health check across all backends...")
    health = await memory.health_check()
    print(f"\n   Overall status: {health['status']}")
    print(f"   Backends: {health['healthy_backends']}/{health['total_backends']} healthy")
    
    for backend_name, backend_health in health['backends'].items():
        status_emoji = "✓" if backend_health['status'] == 'healthy' else "✗"
        print(f"\n   {status_emoji} {backend_name}:")
        if backend_health['status'] == 'healthy':
            if 'memory_count' in backend_health:
                print(f"      Memory count: {backend_health['memory_count']}")
            if 'knot_count' in backend_health:
                print(f"      KNOT count: {backend_health['knot_count']}")
                print(f"      THREAD count: {backend_health['thread_count']}")
            if 'collections' in backend_health:
                for scale, stats in backend_health['collections'].items():
                    print(f"      {scale}: {stats['points']} points")
        else:
            print(f"      Error: {backend_health.get('error', 'unknown')}")
    
    print("\n✓ Example 2 complete!\n")


# ============================================================================
# Example 3: Using the Factory (Graceful Degradation)
# ============================================================================

async def example_3_factory():
    """
    Use the factory function which gracefully degrades
    based on available backends.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Factory with Graceful Degradation")
    print("="*70 + "\n")
    
    # Factory auto-detects available backends
    memory = await create_unified_memory(user_id="blake")
    
    print("1. Checking what's available...")
    health = await memory.health_check()
    print(f"   Store backend: {health['store'].get('backend', 'unknown')}")
    print(f"   Navigator: {health.get('navigator', 'not configured')}")
    print(f"   Detector: {health.get('detector', 'not configured')}")
    
    print("\n2. Storing a test memory...")
    mem_id = await memory.store(
        "Test memory from factory example",
        context={'test': True},
        user_id="blake"
    )
    print(f"   ✓ Stored: {mem_id}")
    
    print("\n3. Recalling...")
    results = await memory.recall("test", user_id="blake")
    print(f"   Found {len(results.memories)} memories")
    
    print("\n✓ Example 3 complete!\n")


# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PROTOCOL-BASED MEMORY SYSTEM - COMPLETE EXAMPLES")
    print("="*70)
    print("\nDemonstrating elegant architecture:")
    print("  ✓ Protocol-based design (swappable implementations)")
    print("  ✓ Graceful degradation (missing backends don't break)")
    print("  ✓ Async-first (non-blocking operations)")
    print("  ✓ Multi-backend fusion (Mem0 + Neo4j + Qdrant)")
    
    # Run examples
    await example_1_in_memory()
    
    try:
        await example_2_hybrid()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}", exc_info=True)
        print("\n⚠ Example 2 skipped (backends not available)")
    
    await example_3_factory()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Simple API: store() and recall() with strategies")
    print("  2. Protocol-based: Easy to test, swap implementations")
    print("  3. Multi-backend: Fuses Mem0, Neo4j, Qdrant seamlessly")
    print("  4. Graceful: Works with or without external dependencies")
    print("\nNext Steps:")
    print("  - Install backends: pip install mem0ai neo4j qdrant-client")
    print("  - Run Neo4j: docker run -p 7687:7687 neo4j")
    print("  - Run Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("  - Try Example 2 with all backends!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
