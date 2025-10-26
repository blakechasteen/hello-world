"""
Example: HoloLoom + Mem0 Integration Demo
==========================================
Demonstrates how to use the hybrid memory system with both HoloLoom
and mem0 working together.

Run from repository root:
    python HoloLoom/examples/hybrid_memory_example.py

Prerequisites:
    pip install mem0ai
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from HoloLoom.documentation.types import Query, MemoryShard, Features
    from HoloLoom.memory.cache import create_memory_manager
    from HoloLoom.memory.graph import KG
    from HoloLoom.memory.mem0_adapter import (
        HybridMemoryManager,
        Mem0Config,
        create_hybrid_memory
    )
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
except ImportError as e:
    print(f"Error importing HoloLoom modules: {e}")
    print("\nMake sure you're running from the repository root:")
    print("  python HoloLoom/examples/hybrid_memory_example.py")
    sys.exit(1)


async def main():
    """
    Demonstration of hybrid memory system.
    
    Shows:
    1. Creating memory shards with domain knowledge
    2. Initializing both HoloLoom and mem0
    3. Storing memories in both systems
    4. Retrieving with fused approach
    5. Comparing results
    """
    print("="*80)
    print("HoloLoom + Mem0 Hybrid Memory Demo")
    print("="*80 + "\n")
    
    # ========================================================================
    # Step 1: Create Sample Domain Knowledge (Beekeeping)
    # ========================================================================
    print("Step 1: Creating sample memory shards (beekeeping domain)...")
    
    shards = [
        MemoryShard(
            id="shard_001",
            text="Hive Jodi has 8 frames of brood and is very active with goldenrod flow.",
            episode="inspection_2025_10_13",
            entities=["Hive Jodi", "brood", "goldenrod"],
            motifs=["HIVE_INSPECTION", "SEASONAL"]
        ),
        MemoryShard(
            id="shard_002",
            text="Winter preparation requires 60-80 lbs of honey stores per hive.",
            episode="winter_prep_guide",
            entities=["winter", "honey stores"],
            motifs=["SEASONAL", "PREPARATION"]
        ),
        MemoryShard(
            id="shard_003",
            text="Varroa mite treatment should be done in late summer before winter bees emerge.",
            episode="varroa_management",
            entities=["varroa mite", "treatment", "winter bees"],
            motifs=["PEST_MANAGEMENT", "SEASONAL"]
        ),
        MemoryShard(
            id="shard_004",
            text="Blake prefers organic treatments for varroa mites and uses formic acid.",
            episode="user_preferences",
            entities=["Blake", "organic", "formic acid", "varroa mites"],
            motifs=["USER_PREFERENCE"]
        ),
        MemoryShard(
            id="shard_005",
            text="The apiary has 3 hives: Jodi, Aurora, and Luna.",
            episode="apiary_setup",
            entities=["Jodi", "Aurora", "Luna"],
            motifs=["FARM_STRUCTURE"]
        )
    ]
    
    print(f"  Created {len(shards)} memory shards")
    print()
    
    # ========================================================================
    # Step 2: Initialize HoloLoom Components
    # ========================================================================
    print("Step 2: Initializing HoloLoom components...")
    
    # Create embeddings
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    print(f"  ✓ Embeddings: {emb.sizes}")
    
    # Create HoloLoom memory manager
    hololoom_memory = create_memory_manager(
        shards=shards,
        emb=emb,
        root="data/demo_hybrid"
    )
    print(f"  ✓ HoloLoom memory manager initialized")
    
    # Create knowledge graph
    kg = KG()
    print(f"  ✓ Knowledge graph initialized")
    print()
    
    # ========================================================================
    # Step 3: Configure and Initialize Hybrid Memory
    # ========================================================================
    print("Step 3: Configuring hybrid memory system...")
    
    # Configure mem0 integration
    mem0_config = Mem0Config(
        enabled=True,  # Enable mem0
        extraction_enabled=True,  # Use intelligent extraction
        graph_sync_enabled=True,  # Sync entities to KG
        user_tracking_enabled=True,  # Track user-specific memories
        mem0_weight=0.3,  # 30% mem0, 70% HoloLoom
        hololoom_weight=0.7
    )
    
    print(f"  Mem0 enabled: {mem0_config.enabled}")
    print(f"  Extraction enabled: {mem0_config.extraction_enabled}")
    print(f"  Graph sync: {mem0_config.graph_sync_enabled}")
    print(f"  Fusion weights: mem0={mem0_config.mem0_weight}, hololoom={mem0_config.hololoom_weight}")
    
    try:
        # Create hybrid manager
        hybrid = create_hybrid_memory(
            hololoom_memory=hololoom_memory,
            mem0_config=mem0_config,
            kg=kg
        )
        print(f"  ✓ Hybrid memory manager created")
        print()
        
    except RuntimeError as e:
        print(f"  ⚠ Could not enable mem0: {e}")
        print(f"  Falling back to HoloLoom-only mode")
        
        # Fallback: disable mem0
        mem0_config.enabled = False
        hybrid = create_hybrid_memory(
            hololoom_memory=hololoom_memory,
            mem0_config=mem0_config,
            kg=kg
        )
        print()
    
    # ========================================================================
    # Step 4: Store Memories (with intelligent extraction)
    # ========================================================================
    print("Step 4: Storing user interaction...")
    
    query = Query(text="How should I prepare my hives for winter?")
    
    results = {
        'response': (
            "For winter preparation, ensure each hive has 60-80 lbs of honey stores. "
            "Complete varroa mite treatment in late summer before winter bees emerge. "
            "Based on your preference for organic treatments, formic acid is a good option. "
            "Check all three hives (Jodi, Aurora, Luna) individually."
        ),
        'tool': 'answer',
        'confidence': 0.92
    }
    
    features = Features(
        psi=[0.1] * 384,  # Dummy embedding
        motifs=["SEASONAL", "PREPARATION"],
        metrics={},
        metadata={'query_length': len(query.text)}
    )
    
    print(f"  Query: '{query.text}'")
    print(f"  Storing in both HoloLoom and mem0...")
    
    await hybrid.store(
        query=query,
        results=results,
        features=features,
        user_id="blake"
    )
    
    print(f"  ✓ Memory stored successfully")
    print()
    
    # ========================================================================
    # Step 5: Retrieve Memories (fused approach)
    # ========================================================================
    print("Step 5: Retrieving relevant memories...")
    
    test_query = Query(text="What organic treatments does Blake use?")
    print(f"  Test query: '{test_query.text}'")
    print()
    
    # Retrieve using hybrid system
    context = await hybrid.retrieve(
        query=test_query,
        user_id="blake",
        k=3
    )
    
    # Display results
    print(f"  Results:")
    print(f"    Total shards: {len(context.shards) if hasattr(context, 'shards') else 0}")
    print(f"    Relevance score: {context.relevance if hasattr(context, 'relevance') else 'N/A':.3f}")
    
    if hasattr(context, 'metadata'):
        print(f"    Fusion metadata:")
        for key, value in context.metadata.items():
            print(f"      {key}: {value}")
    
    print()
    print(f"  Top memories:")
    if hasattr(context, 'shards'):
        for i, shard in enumerate(context.shards[:3], 1):
            print(f"    {i}. [{shard.episode}] {shard.text[:80]}...")
    
    print()
    
    # ========================================================================
    # Step 6: Show Knowledge Graph Integration
    # ========================================================================
    print("Step 6: Knowledge graph integration...")
    
    print(f"  Graph stats:")
    stats = kg.stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    print()
    
    # Get entity context
    if 'Blake' in kg.G:
        print(f"  Entity context for 'Blake':")
        neighbors = kg.get_neighbors('Blake', direction='both', max_hops=1)
        print(f"    Direct connections: {list(neighbors)[:5]}")
    
    print()
    
    # ========================================================================
    # Step 7: User Profile
    # ========================================================================
    print("Step 7: User profile from mem0...")
    
    profile = await hybrid.get_user_profile(user_id="blake")
    
    if profile.get('available'):
        print(f"  User: {profile['user_id']}")
        print(f"  Total memories: {profile['memory_count']}")
        print(f"  Recent memories:")
        for mem in profile['memories'][:3]:
            print(f"    - {mem.get('memory', '')[:60]}...")
    else:
        print(f"  User profile not available (mem0 may be disabled)")
    
    print()
    
    # ========================================================================
    # Step 8: Comparison (HoloLoom-only vs Hybrid)
    # ========================================================================
    print("Step 8: Comparing retrieval approaches...")
    
    # HoloLoom-only retrieval
    print(f"  Testing HoloLoom-only retrieval...")
    hololoom_only_context = await hololoom_memory.retrieve(
        query=test_query,
        kg_sub=None
    )
    
    hololoom_count = len(hololoom_only_context.shards) if hasattr(hololoom_only_context, 'shards') else 0
    hybrid_count = len(context.shards) if hasattr(context, 'shards') else 0
    
    print(f"    HoloLoom-only: {hololoom_count} shards")
    print(f"    Hybrid (mem0+HoloLoom): {hybrid_count} shards")
    
    if mem0_config.enabled:
        print(f"    ✓ Hybrid system provides user-personalized results")
    else:
        print(f"    ⚠ Mem0 disabled, using HoloLoom-only")
    
    print()
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    print("Step 9: Cleanup...")
    await hybrid.shutdown()
    print(f"  ✓ Hybrid memory shutdown complete")
    
    print()
    print("="*80)
    print("Demo Complete!")
    print("="*80)
    print()
    
    print("Summary:")
    print("  ✓ Created domain-specific memory shards")
    print("  ✓ Initialized hybrid memory system")
    print("  ✓ Stored memories with intelligent extraction")
    print("  ✓ Retrieved with fused approach")
    print("  ✓ Demonstrated knowledge graph integration")
    print()
    
    if mem0_config.enabled:
        print("Next steps:")
        print("  1. Customize mem0 extraction for your domain")
        print("  2. Tune fusion weights (mem0_weight / hololoom_weight)")
        print("  3. Implement user-specific personalization")
        print("  4. Add memory decay and filtering")
    else:
        print("To enable mem0:")
        print("  1. Install: pip install mem0ai")
        print("  2. Optionally set API key for managed platform")
        print("  3. Configure Mem0Config with enabled=True")


if __name__ == "__main__":
    asyncio.run(main())
