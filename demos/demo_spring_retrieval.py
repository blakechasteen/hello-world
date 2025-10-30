"""
Spring-Based Memory Retrieval Demo
===================================
Demonstrates physics-driven spreading activation for memory retrieval.

This demo shows:
1. Building a knowledge graph with memory shards
2. Static retrieval (baseline)
3. Spring activation retrieval (multi-hop transitive)
4. Comparison of results

Run:
    python demos/demo_spring_retrieval.py

Author: HoloLoom Team
Date: 2025-10-29
"""

import asyncio
from typing import List, Dict
from dataclasses import dataclass

# HoloLoom imports
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.spring_dynamics import SpringConfig, SpringDynamics
from HoloLoom.memory.retrieval_strategies import (
    StaticRetrieval,
    SpringActivationRetrieval,
    create_retrieval_strategy
)
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config


# ============================================================================
# Demo Data: Knowledge Graph about Reinforcement Learning
# ============================================================================

def create_rl_knowledge_graph() -> tuple[KG, List[MemoryShard], Dict[str, MemoryShard]]:
    """
    Create a knowledge graph about reinforcement learning concepts.

    Returns:
        (KG, List[MemoryShard], Dict[node_id: MemoryShard])
    """
    kg = KG()

    # === Define concepts as nodes + shards ===
    concepts = [
        {"id": "thompson_sampling", "title": "Thompson Sampling",
         "content": "A probabilistic algorithm for balancing exploration and exploitation in multi-armed bandits."},

        {"id": "bandits", "title": "Multi-Armed Bandits",
         "content": "A class of reinforcement learning problems focused on resource allocation and exploration."},

        {"id": "exploration", "title": "Exploration vs Exploitation",
         "content": "The fundamental tradeoff in RL: trying new actions vs. using known good actions."},

        {"id": "bayesian_inference", "title": "Bayesian Inference",
         "content": "Statistical method for updating probabilities as new evidence is acquired."},

        {"id": "regret_bounds", "title": "Regret Bounds",
         "content": "Theoretical guarantees on the cumulative loss from suboptimal actions."},

        {"id": "ucb", "title": "Upper Confidence Bound (UCB)",
         "content": "A deterministic algorithm that uses confidence intervals to guide exploration."},

        {"id": "epsilon_greedy", "title": "Epsilon-Greedy",
         "content": "A simple exploration strategy: choose randomly with probability epsilon, otherwise greedy."},

        {"id": "prior_distribution", "title": "Prior Distribution",
         "content": "Initial beliefs about parameters before observing data, used in Bayesian methods."},

        {"id": "reward_optimization", "title": "Reward Optimization",
         "content": "The goal of finding actions that maximize cumulative reward over time."},

        {"id": "posterior_sampling", "title": "Posterior Sampling",
         "content": "Selecting actions by sampling from the posterior distribution of beliefs."},
    ]

    # Create shards and node-to-shard mapping
    shards = []
    shard_map = {}

    for concept in concepts:
        shard = MemoryShard(
            id=concept["id"],
            text=f"{concept['title']}: {concept['content']}",
            metadata={"title": concept["title"], "domain": "RL"}
        )
        shards.append(shard)
        shard_map[concept["id"]] = shard

    # === Add relationships (edges) ===
    edges = [
        # Thompson Sampling relationships
        KGEdge("thompson_sampling", "bandits", "IS_INSTANCE_OF", weight=1.0),
        KGEdge("thompson_sampling", "bayesian_inference", "USES", weight=0.9),
        KGEdge("thompson_sampling", "posterior_sampling", "IMPLEMENTS", weight=1.0),
        KGEdge("thompson_sampling", "exploration", "ADDRESSES", weight=0.8),

        # Bandit relationships
        KGEdge("bandits", "exploration", "INVOLVES", weight=1.0),
        KGEdge("bandits", "reward_optimization", "GOAL_IS", weight=0.9),
        KGEdge("bandits", "ucb", "SOLVED_BY", weight=0.7),
        KGEdge("bandits", "epsilon_greedy", "SOLVED_BY", weight=0.6),

        # Bayesian relationships
        KGEdge("bayesian_inference", "prior_distribution", "USES", weight=1.0),
        KGEdge("bayesian_inference", "posterior_sampling", "ENABLES", weight=0.9),

        # Exploration relationships
        KGEdge("exploration", "regret_bounds", "MEASURED_BY", weight=0.7),
        KGEdge("exploration", "epsilon_greedy", "IMPLEMENTED_BY", weight=0.6),

        # UCB relationships
        KGEdge("ucb", "exploration", "ADDRESSES", weight=0.8),
        KGEdge("ucb", "regret_bounds", "HAS_GUARANTEE", weight=0.9),

        # Epsilon-greedy relationships
        KGEdge("epsilon_greedy", "exploration", "SIMPLE_APPROACH_TO", weight=0.7),

        # Reward optimization
        KGEdge("reward_optimization", "regret_bounds", "EVALUATED_BY", weight=0.8),
    ]

    for edge in edges:
        kg.add_edge(edge)

    return kg, shards, shard_map


# ============================================================================
# Comparison Functions
# ============================================================================

async def compare_retrieval_strategies():
    """
    Compare static vs spring retrieval on the same query.
    """
    print("=" * 70)
    print("SPRING-BASED MEMORY RETRIEVAL DEMO")
    print("=" * 70)
    print()

    # === 1. Build Knowledge Graph ===
    print("Building knowledge graph with RL concepts...")
    kg, shards, shard_map = create_rl_knowledge_graph()
    print(f"✓ Created graph with {len(kg.G.nodes())} nodes and {len(kg.G.edges())} edges")
    print()

    # === 2. Create Retrieval Strategies ===
    print("Creating retrieval strategies...")

    # Static retrieval (baseline)
    static_retrieval = StaticRetrieval(shards=shards)

    # Spring activation retrieval
    spring_config = SpringConfig(
        stiffness=0.15,
        damping=0.85,
        decay=0.98,
        max_iterations=200,
        activation_threshold=0.1
    )
    spring_retrieval = SpringActivationRetrieval(
        graph=kg,
        shards=shards,
        shard_map=shard_map,
        spring_config=spring_config
    )

    print("✓ Static retrieval ready")
    print("✓ Spring retrieval ready")
    print()

    # === 3. Test Query ===
    query = Query(text="How does Thompson Sampling work?")

    print(f"Query: \"{query.text}\"")
    print()

    # === 4. Static Retrieval ===
    print("-" * 70)
    print("STATIC RETRIEVAL (Baseline)")
    print("-" * 70)

    static_result = await static_retrieval.retrieve_with_metadata(query, k=5)

    print(f"Strategy: {static_result.strategy}")
    print(f"Retrieved: {static_result.k_returned} shards in {static_result.retrieval_time_ms:.2f}ms")
    print(f"Confidence: avg={static_result.avg_confidence:.3f}, "
          f"min={static_result.min_confidence:.3f}, max={static_result.max_confidence:.3f}")
    print()
    print("Results:")
    for i, shard in enumerate(static_result.shards, 1):
        print(f"  {i}. {shard.metadata.get('title', 'Unknown')}")
    print()

    # === 5. Spring Activation Retrieval ===
    print("-" * 70)
    print("SPRING ACTIVATION RETRIEVAL (Multi-hop Transitive)")
    print("-" * 70)

    spring_result = await spring_retrieval.retrieve_with_metadata(query, k=10, seed_k=2)

    spring_meta = spring_result.metadata.get("spring_activation")

    print(f"Strategy: {spring_result.strategy}")
    print(f"Retrieved: {spring_result.k_returned} shards in {spring_result.retrieval_time_ms:.2f}ms")
    print(f"Confidence: avg={spring_result.avg_confidence:.3f}, "
          f"min={spring_result.min_confidence:.3f}, max={spring_result.max_confidence:.3f}")
    print()

    if spring_meta:
        print("Spring Physics:")
        print(f"  Seeds: {', '.join(spring_meta.seed_nodes)}")
        print(f"  Iterations: {spring_meta.iterations}")
        print(f"  Converged: {spring_meta.converged}")
        print(f"  Final Energy: {spring_meta.final_energy:.6f}")
        print(f"  Activated: {spring_meta.activated_count} nodes")
        print()

        print("Timing Breakdown:")
        print(f"  Embedding (seed finding): {spring_meta.embedding_time_ms:.2f}ms")
        print(f"  Spring propagation: {spring_meta.propagation_time_ms:.2f}ms")
        print(f"  Shard retrieval: {spring_meta.shard_retrieval_time_ms:.2f}ms")
        print()

    print("Results (sorted by activation):")
    for i, (shard, node_id) in enumerate(zip(spring_result.shards,
                                             spring_meta.seed_nodes if spring_meta else []), 1):
        activation = spring_meta.node_activations.get(node_id, 0.0) if spring_meta else 0.0
        title = shard.metadata.get('title', 'Unknown')
        print(f"  {i}. {title:<30} (activation: {activation:.3f})")

    print()

    # === 6. Comparison ===
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)

    # Find concepts retrieved by spring but not static
    static_ids = {shard.id for shard in static_result.shards}
    spring_ids = {shard.id for shard in spring_result.shards}

    unique_to_spring = spring_ids - static_ids

    print(f"Static retrieved: {len(static_ids)} concepts")
    print(f"Spring retrieved: {len(spring_ids)} concepts")
    print(f"Overlap: {len(static_ids & spring_ids)} concepts")
    print()

    if unique_to_spring:
        print(f"Concepts found ONLY by spring activation (transitive relationships):")
        for node_id in unique_to_spring:
            shard = shard_map[node_id]
            activation = spring_meta.node_activations.get(node_id, 0.0) if spring_meta else 0.0
            print(f"  • {shard.metadata.get('title'):<30} (activation: {activation:.3f})")
        print()
        print("These concepts have low DIRECT similarity to the query,")
        print("but are connected through MULTI-HOP relationships in the graph!")
    else:
        print("All spring results were also in static results (no transitive advantage shown)")

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


# ============================================================================
# Visual Network Display
# ============================================================================

def print_activation_network():
    """Print ASCII visualization of spring activation spreading."""
    print()
    print("=" * 70)
    print("SPRING ACTIVATION VISUALIZATION")
    print("=" * 70)
    print()
    print("Query: 'How does Thompson Sampling work?'")
    print()
    print("Activation spreading (time →):")
    print()
    print("t=0: Seeds activated")
    print("     ┌──────────────────┐")
    print("     │ Thompson Sampling│ (1.00)")
    print("     └──────────────────┘")
    print("            │")
    print("            │ (edge: USES)")
    print("            ↓")
    print("     ┌─────────────────┐")
    print("     │Bayesian Inference│ (0.00)")
    print("     └─────────────────┘")
    print()
    print("t=20: Spring forces pull neighbors")
    print("     ┌──────────────────┐")
    print("     │ Thompson Sampling│ (0.98) ← decaying")
    print("     └──────────────────┘")
    print("            │")
    print("            │ F = k × Δa")
    print("            ↓")
    print("     ┌─────────────────┐")
    print("     │Bayesian Inference│ (0.45) ← activated!")
    print("     └─────────────────┘")
    print("            │")
    print("            ↓")
    print("     ┌────────────────┐")
    print("     │Prior Distribution│ (0.12)")
    print("     └────────────────┘")
    print()
    print("t=50: System reaches equilibrium")
    print("     ┌──────────────────┐")
    print("     │ Thompson Sampling│ (0.85)")
    print("     └──────────────────┘")
    print("            │")
    print("     ┌─────────────────┐")
    print("     │Bayesian Inference│ (0.68)")
    print("     └─────────────────┘")
    print("            │")
    print("     ┌────────────────┐")
    print("     │Prior Distribution│ (0.35)")
    print("     └────────────────┘")
    print()
    print("All nodes above threshold (0.1) are retrieved!")
    print("=" * 70)
    print()


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the demo."""
    print_activation_network()
    await compare_retrieval_strategies()


if __name__ == "__main__":
    asyncio.run(main())
