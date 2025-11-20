"""
Demo: Beta Wave Activation Spreading for Memory Retrieval
===========================================================

This demo shows how spring dynamics enable:
1. Recall strengthening (frequently accessed = stronger springs)
2. Natural forgetting (unused = weaker springs)
3. Beta wave retrieval (activation spreading through springs)
4. Creative insights (distant but activated nodes)

The "call number" metaphor:
- Each memory has an address (node ID)
- Spring constant k = how fresh that call number is
- Strong k = easy to recall
- Weak k = faded, needs strong reminder
- Spreading = beta waves synchronizing brain regions

Author: HoloLoom Memory Team
Date: October 30, 2025
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from spring_dynamics_engine import (
    SpringDynamicsEngine,
    SpringEngineConfig,
    MemoryNode
)


async def demo_beta_wave_retrieval():
    """Demonstrate beta wave activation spreading retrieval."""

    print("=" * 70)
    print("Beta Wave Activation Spreading - Memory Retrieval Demo")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Create Memory Network
    # ========================================================================

    print("[Step 1] Creating memory network...")
    print()

    # Create engine
    config = SpringEngineConfig(
        base_decay_rate=0.05,  # Faster decay for demo
        access_boost=0.3,
        propagation_factor=0.6,
        max_propagation_hops=3
    )
    engine = SpringDynamicsEngine(config)

    # Create semantic embeddings (simplified 3D for visualization)
    # Cluster 1: Machine Learning concepts
    ml_cluster = {
        'neural_networks': np.array([1.0, 0.2, 0.1], dtype=np.float32),
        'gradient_descent': np.array([1.1, 0.3, 0.2], dtype=np.float32),
        'backpropagation': np.array([1.0, 0.4, 0.15], dtype=np.float32),
    }

    # Cluster 2: Reinforcement Learning concepts
    rl_cluster = {
        'thompson_sampling': np.array([0.1, 1.0, 0.2], dtype=np.float32),
        'bandits': np.array([0.2, 1.1, 0.3], dtype=np.float32),
        'exploration': np.array([0.15, 0.9, 0.25], dtype=np.float32),
    }

    # Cluster 3: Memory/Cognition concepts
    memory_cluster = {
        'recall': np.array([0.2, 0.3, 1.0], dtype=np.float32),
        'forgetting': np.array([0.3, 0.25, 1.1], dtype=np.float32),
        'beta_waves': np.array([0.25, 0.35, 0.9], dtype=np.float32),
    }

    # Bridge node: connects ML and RL
    bridge = {
        'ppo_algorithm': np.array([0.6, 0.7, 0.3], dtype=np.float32),
    }

    all_memories = {**ml_cluster, **rl_cluster, **memory_cluster, **bridge}

    # Add nodes to engine
    for node_id, embedding in all_memories.items():
        engine.add_node(
            node_id=node_id,
            embedding=embedding,
            content=f"Memory about {node_id.replace('_', ' ')}",
            neighbors=set(),  # Will add neighbors next
            initial_spring_constant=1.0
        )

    # Add neighbor connections (graph structure)
    # ML cluster interconnections
    engine.nodes['neural_networks'].neighbors = {'gradient_descent', 'backpropagation', 'ppo_algorithm'}
    engine.nodes['gradient_descent'].neighbors = {'neural_networks', 'backpropagation'}
    engine.nodes['backpropagation'].neighbors = {'neural_networks', 'gradient_descent'}

    # RL cluster interconnections
    engine.nodes['thompson_sampling'].neighbors = {'bandits', 'exploration', 'ppo_algorithm'}
    engine.nodes['bandits'].neighbors = {'thompson_sampling', 'exploration'}
    engine.nodes['exploration'].neighbors = {'thompson_sampling', 'bandits'}

    # Memory cluster interconnections
    engine.nodes['recall'].neighbors = {'forgetting', 'beta_waves'}
    engine.nodes['forgetting'].neighbors = {'recall', 'beta_waves'}
    engine.nodes['beta_waves'].neighbors = {'recall', 'forgetting'}

    # Bridge node connections
    engine.nodes['ppo_algorithm'].neighbors = {'neural_networks', 'thompson_sampling', 'exploration'}

    print(f"✓ Created {len(engine.nodes)} memory nodes")
    print(f"  - {len(ml_cluster)} ML concepts")
    print(f"  - {len(rl_cluster)} RL concepts")
    print(f"  - {len(memory_cluster)} Memory/Cognition concepts")
    print(f"  - 1 bridge node (PPO)")
    print()

    # ========================================================================
    # Step 2: Simulate Memory Access (Recall Strengthening)
    # ========================================================================

    print("[Step 2] Simulating memory access patterns...")
    print()

    # Start background dynamics
    await engine.start()

    # Simulate: User frequently thinks about RL concepts
    print("Simulating frequent access to RL concepts...")
    for i in range(5):
        engine.on_memory_accessed('thompson_sampling', pulse_strength=0.4)
        engine.on_memory_accessed('bandits', pulse_strength=0.3)
        await asyncio.sleep(0.2)  # Brief pause between accesses

    print("✓ Accessed thompson_sampling (5×) and bandits (5×)")
    print(f"  Spring constants after access:")
    print(f"    thompson_sampling: k = {engine.nodes['thompson_sampling'].spring_constant:.2f}")
    print(f"    bandits: k = {engine.nodes['bandits'].spring_constant:.2f}")
    print(f"    neural_networks: k = {engine.nodes['neural_networks'].spring_constant:.2f} (not accessed)")
    print()

    # Wait for background dynamics to settle
    await asyncio.sleep(0.5)

    # ========================================================================
    # Step 3: Beta Wave Retrieval
    # ========================================================================

    print("[Step 3] Retrieving memories via beta wave spreading...")
    print()

    # Query: Something related to RL exploration
    query_embedding = np.array([0.15, 0.95, 0.25], dtype=np.float32)
    print("Query embedding (close to 'exploration'): [0.15, 0.95, 0.25]")
    print()

    result = engine.retrieve_memories(
        query_embedding=query_embedding,
        top_k=10,
        seed_strength=1.0,
        activation_threshold=0.05
    )

    print(f"✓ {result}")
    print()

    print("Seed nodes (most similar to query):")
    for node_id, similarity in result.seed_nodes:
        print(f"  • {node_id}: similarity = {similarity:.3f}")
    print()

    print("Recalled memories (via activation spreading):")
    for i, (node_id, activation) in enumerate(result.recalled_memories[:10], 1):
        node = engine.nodes[node_id]
        is_seed = node_id in result.get_direct_recalls()
        is_creative = node_id in result.get_creative_insight_ids()

        marker = "🌱" if is_seed else ("💡" if is_creative else "🔗")
        print(f"  {marker} {i}. {node_id}")
        print(f"      activation = {activation:.3f}, spring_k = {node.spring_constant:.2f}")

    print()
    print("Legend:")
    print("  🌱 = Direct seed (semantically similar to query)")
    print("  🔗 = Associative recall (activated via spreading)")
    print("  💡 = Creative insight (semantically distant but activated)")
    print()

    # ========================================================================
    # Step 4: Creative Insights
    # ========================================================================

    print("[Step 4] Analyzing creative insights...")
    print()

    if result.creative_insights:
        print(f"Found {len(result.creative_insights)} creative insights:")
        for node_id, activation, insight_type in result.creative_insights:
            print(f"  💡 {node_id}")
            print(f"      activation = {activation:.3f}")
            print(f"      type = {insight_type}")
            print(f"      (semantically distant from query but activated through spreading)")
        print()
    else:
        print("No creative insights detected in this query.")
        print("(Try queries that bridge clusters for cross-domain associations)")
        print()

    # ========================================================================
    # Step 5: Demonstrate Forgetting
    # ========================================================================

    print("[Step 5] Demonstrating natural forgetting...")
    print()

    print("Simulating passage of time (24 hours)...")

    # Manually age 'neural_networks' (not accessed)
    nn_node = engine.nodes['neural_networks']
    nn_node.last_accessed = datetime.now() - timedelta(hours=24)

    # Apply forgetting decay
    engine._apply_forgetting_decay()

    print(f"✓ After 24 hours without access:")
    print(f"    neural_networks: k = {nn_node.spring_constant:.2f} (decayed)")
    print(f"    thompson_sampling: k = {engine.nodes['thompson_sampling'].spring_constant:.2f} (still strong)")
    print()

    print("Accessing neural_networks to restore it...")
    engine.on_memory_accessed('neural_networks', pulse_strength=0.5)

    print(f"✓ After access:")
    print(f"    neural_networks: k = {nn_node.spring_constant:.2f} (restored!)")
    print()

    # ========================================================================
    # Step 6: Cross-Domain Query (Bridge Activation)
    # ========================================================================

    print("[Step 6] Cross-domain query (testing bridge node)...")
    print()

    # Query near bridge node (PPO - connects ML and RL)
    bridge_query = np.array([0.6, 0.7, 0.3], dtype=np.float32)
    print("Query embedding (near PPO bridge): [0.6, 0.7, 0.3]")
    print()

    bridge_result = engine.retrieve_memories(
        query_embedding=bridge_query,
        top_k=10,
        activation_threshold=0.05
    )

    print(f"✓ {bridge_result}")
    print()

    print("Recalled memories (should span ML and RL clusters):")
    for i, (node_id, activation) in enumerate(bridge_result.recalled_memories[:8], 1):
        print(f"  {i}. {node_id}: activation = {activation:.3f}")

    print()
    print("Notice: Both ML and RL concepts recalled via bridge node!")
    print()

    # ========================================================================
    # Cleanup
    # ========================================================================

    await engine.stop()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 70)
    print("Summary: Beta Wave Retrieval Demonstrated")
    print("=" * 70)
    print()
    print("✅ Recall strengthening: Frequent access → stronger springs")
    print("✅ Natural forgetting: Unused memories → weaker springs")
    print("✅ Beta wave spreading: Activation flows through springs")
    print("✅ Creative insights: Distant nodes activated = cross-domain associations")
    print("✅ Bridge nodes: Connect clusters for creative retrieval")
    print()
    print("Key Insight:")
    print("  Spring constant k = 'freshness' of call number")
    print("  High k → easy recall, Low k → faded (needs strong reminder)")
    print("  Spreading = beta wave synchronization across brain regions")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    asyncio.run(demo_beta_wave_retrieval())
