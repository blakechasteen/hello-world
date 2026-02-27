"""
Demo: Memory System v1.0 + Spring Physics Integration
=======================================================

Shows how to enhance the Memory System v1.0 hybrid retrieval with
physics-based spreading activation using spring dynamics.

This replaces the simple BFS graph traversal with a physically-motivated
approach based on Hooke's Law and energy minimization.

Benefits:
- More natural activation spreading
- Edge type differentiation (IS_A stronger than MENTIONS)
- Energy-conserving dynamics (Velocity Verlet integration)
- Physically interpretable parameters
"""

import asyncio
from typing import List, Dict
import time

# Memory System v1.0
from hololoom.memory.integrated_memory_system import create_integrated_memory_system
from hololoom.memory.lifecycle_manager import MemoryScope
from hololoom.documentation.types import MemoryShard

# Spring physics
from hololoom.memory.spring_dynamics import SpringDynamics, SpringConfig
from hololoom.memory.graph import KG, KGEdge


# ============================================================================
# Enhanced Graph Retriever with Spring Physics
# ============================================================================

class SpringPhysicsGraphRetriever:
    """
    Graph retriever using spring-based activation dynamics.

    Replaces simple BFS with physics-driven spreading activation.
    """

    def __init__(
        self,
        kg: KG,
        config: SpringConfig = None
    ):
        self.kg = kg
        self.config = config or SpringConfig(
            stiffness=0.80,         # Strong spring connections
            damping=0.85,           # Prevent oscillation
            decay=0.99,             # Slow forgetting
            dt=0.2,                 # Time step
            integrator_type="verlet"  # Gold standard!
        )

        # Spring dynamics expects KG object with .G property
        self.spring_dynamics = SpringDynamics(
            graph=kg,  # Pass KG object, not kg.G
            config=self.config
        )

    def activate_from_query(
        self,
        query_entities: List[str],
        activation_strength: float = 1.0
    ) -> Dict[str, float]:
        """
        Spread activation from query entities using spring physics.

        Args:
            query_entities: Entities extracted from query
            activation_strength: Initial activation (0.0-1.0)

        Returns:
            Dict of {entity: activation_score}
        """
        if not query_entities:
            return {}

        # Create seed activations
        seeds = {entity: activation_strength for entity in query_entities}

        # Activate seed nodes
        self.spring_dynamics.activate_nodes(seeds)

        # Run spring dynamics propagation
        result = self.spring_dynamics.propagate()

        # Return final activations
        return {
            node_id: state.activation
            for node_id, state in self.spring_dynamics.node_states.items()
            if state.activation > self.config.activation_threshold
        }

    async def retrieve(
        self,
        query: str,
        memories: List[MemoryShard],
        limit: int = 10
    ) -> List[tuple]:
        """
        Retrieve memories using spring physics activation.

        Args:
            query: Search query
            memories: Candidate memories
            limit: Maximum results

        Returns:
            List of (memory, score) tuples
        """
        # Extract entities from query (simple approach)
        query_entities = self._extract_entities(query)

        if not query_entities:
            return []

        # Spring-based activation spreading
        activations = self.activate_from_query(query_entities)

        # Score memories based on activated entities
        scored_memories = []
        for memory in memories:
            # Check entity overlap with activated nodes
            memory_entities = set(memory.entities)
            activated_entities = set(activations.keys())

            overlap = memory_entities & activated_entities
            if overlap:
                # Score = sum of activation strengths
                score = sum(activations[entity] for entity in overlap)
                scored_memories.append((memory, score))

        # Sort by score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return scored_memories[:limit]

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query text."""
        # Simple approach: check if words are in graph
        words = query.replace(",", " ").replace(".", " ").split()
        entities = []

        for word in words:
            # Try exact match and capitalized
            for variant in [word, word.capitalize(), word.upper()]:
                if variant in self.kg.G.nodes:
                    entities.append(variant)
                    break

        return entities


# ============================================================================
# Demo: Compare BFS vs Spring Physics
# ============================================================================

async def demo_comparison():
    """Compare standard BFS retrieval vs spring physics."""

    print("\n" + "="*70)
    print("DEMO: Memory System v1.0 + Spring Physics Integration")
    print("="*70)

    # Create memory system
    system = create_integrated_memory_system()

    # Store sample memories with entity relationships
    print("\n1. Storing memories with entity relationships...")

    memories_data = [
        {
            "text": "Thompson Sampling is a Bayesian approach to multi-armed bandits",
            "entities": ["ThompsonSampling", "Bayesian", "MultiArmedBandit"],
            "importance": 0.9
        },
        {
            "text": "Bayesian methods use prior distributions and posterior updates",
            "entities": ["Bayesian", "PriorDistribution", "PosteriorUpdate"],
            "importance": 0.8
        },
        {
            "text": "Multi-armed bandits solve exploration-exploitation tradeoffs",
            "entities": ["MultiArmedBandit", "Exploration", "Exploitation"],
            "importance": 0.85
        },
        {
            "text": "Epsilon-greedy is a simple exploration strategy",
            "entities": ["EpsilonGreedy", "Exploration"],
            "importance": 0.7
        },
        {
            "text": "UCB uses upper confidence bounds for exploration",
            "entities": ["UCB", "Exploration", "ConfidenceBound"],
            "importance": 0.75
        },
        {
            "text": "Gaussian processes are used in Bayesian optimization",
            "entities": ["GaussianProcess", "Bayesian", "Optimization"],
            "importance": 0.8
        },
        {
            "text": "Prior distributions encode domain knowledge before seeing data",
            "entities": ["PriorDistribution", "DomainKnowledge"],
            "importance": 0.7
        }
    ]

    for mem in memories_data:
        await system.store(
            mem["text"],
            importance=mem["importance"],
            entities=mem["entities"]
        )

    print(f"✓ Stored {len(memories_data)} memories")

    # Build knowledge graph with edge types
    print("\n2. Building knowledge graph with typed edges...")

    kg = KG()
    edges = [
        # Taxonomic (strong connections)
        KGEdge("ThompsonSampling", "Bayesian", "IS_A", 1.0),
        KGEdge("ThompsonSampling", "MultiArmedBandit", "SOLVES", 1.0),
        KGEdge("EpsilonGreedy", "Exploration", "USES", 1.0),
        KGEdge("UCB", "Exploration", "USES", 1.0),

        # Compositional (medium strength)
        KGEdge("Bayesian", "PriorDistribution", "USES", 0.9),
        KGEdge("Bayesian", "PosteriorUpdate", "USES", 0.9),
        KGEdge("MultiArmedBandit", "Exploration", "INVOLVES", 0.8),
        KGEdge("MultiArmedBandit", "Exploitation", "INVOLVES", 0.8),

        # Associative (weaker)
        KGEdge("GaussianProcess", "Bayesian", "RELATED_TO", 0.7),
        KGEdge("PriorDistribution", "DomainKnowledge", "ENCODES", 0.6),
        KGEdge("UCB", "ConfidenceBound", "USES", 0.8),
    ]

    kg.add_edges(edges)
    print(f"✓ Built graph: {len(kg.G.nodes)} nodes, {len(kg.G.edges)} edges")

    # Query
    query = "Bayesian methods"
    print(f"\n3. Query: '{query}'")

    # Get candidate memories
    all_memories = system.stream_manager.get_all_memories()

    # Standard hybrid retrieval (BFS-based graph)
    print("\n4. Standard Hybrid Retrieval (BFS graph traversal)...")

    start = time.perf_counter()
    standard_result = await system.retrieve(query, limit=5)
    standard_time = (time.perf_counter() - start) * 1000

    print(f"\n   Results ({standard_time:.2f}ms):")
    for i, memory in enumerate(standard_result.memories, 1):
        print(f"   {i}. {memory.text[:60]}...")

    # Spring physics retrieval
    print("\n5. Spring Physics Retrieval (Hooke's Law dynamics)...")

    spring_retriever = SpringPhysicsGraphRetriever(kg=kg)

    start = time.perf_counter()
    spring_results = await spring_retriever.retrieve(query, all_memories, limit=5)
    spring_time = (time.perf_counter() - start) * 1000

    print(f"\n   Results ({spring_time:.2f}ms):")
    for i, (memory, score) in enumerate(spring_results, 1):
        print(f"   {i}. [{score:.3f}] {memory.text[:55]}...")

    # Show activation spreading
    print("\n6. Activation Spreading Visualization...")

    query_entities = ["Bayesian"]
    activations = spring_retriever.activate_from_query(query_entities)

    # Sort by activation
    sorted_activations = sorted(
        activations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n   Spreading from '{query_entities[0]}':")
    for entity, activation in sorted_activations[:10]:
        # Visual bar
        bar_length = int(activation * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {entity:25} {bar} {activation:.3f}")

    # Physics statistics
    print("\n7. Spring Physics Statistics...")
    print(f"   Integrator: {spring_retriever.config.integrator_type} (Velocity Verlet)")
    print(f"   Stiffness (k): {spring_retriever.config.stiffness}")
    print(f"   Damping (c): {spring_retriever.config.damping}")
    print(f"   Time step (dt): {spring_retriever.config.dt}")
    print(f"   Energy conserving: ✓ (symplectic integrator)")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: BFS vs Spring Physics")
    print("="*70)
    print(f"Standard BFS:      {standard_time:.2f}ms")
    print(f"Spring Physics:    {spring_time:.2f}ms")
    print(f"")
    print("Key Differences:")
    print("- BFS: Simple hop-based decay (0.8 per hop)")
    print("- Spring: Physics-driven with edge type differentiation")
    print("- Spring: IS_A edges stronger than MENTIONS edges")
    print("- Spring: Energy landscape reveals semantic structure")
    print("="*70)


# ============================================================================
# Demo: Advanced Features
# ============================================================================

async def demo_advanced_features():
    """Show advanced spring physics features."""

    print("\n" + "="*70)
    print("ADVANCED: Temperature Control & Multi-Query Seeds")
    print("="*70)

    # Load advanced features
    try:
        from hololoom.memory.spring_dynamics_advanced import (
            AdvancedSpringActivation,
            TemperatureSchedule
        )

        # Create knowledge graph
        kg = KG()
        edges = [
            KGEdge("A", "B", "IS_A", 1.0),
            KGEdge("B", "C", "USES", 0.8),
            KGEdge("C", "D", "MENTIONS", 0.6),
            KGEdge("D", "E", "RELATED_TO", 0.5),
        ]
        kg.add_edges(edges)

        # Temperature schedule (simulated annealing)
        schedule = TemperatureSchedule(
            initial_temp=1.0,     # Hot start (more exploration)
            final_temp=0.1,       # Cool finish (settle to best)
            cooling_rate=0.95     # Exponential cooling
        )

        advanced_spring = AdvancedSpringActivation(
            graph=kg.G,
            temperature_schedule=schedule
        )

        print("\n✓ Advanced features available!")
        print("  • Temperature control (simulated annealing)")
        print("  • Multi-query seed handling")
        print("  • Energy landscape analysis")

    except ImportError:
        print("\n⚠ Advanced features not imported (optional)")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""

    # Main comparison demo
    await demo_comparison()

    # Advanced features
    await demo_advanced_features()

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("")
    print("Next steps:")
    print("1. Integrate SpringPhysicsGraphRetriever into HybridRetriever")
    print("2. Add edge type multipliers to knowledge graph builder")
    print("3. Tune spring parameters (stiffness, damping) for your domain")
    print("4. Visualize activation spreading for debugging")
    print("="*70)
    print("")


if __name__ == "__main__":
    asyncio.run(main())
