"""
Demo: Spring Physics Memory Learning System
============================================

Shows how the spring physics system learns from usage patterns and creates
a self-improving retrieval system.

This demonstrates:
1. Initial query performance (cold start)
2. Learning from activation patterns
3. Improved performance on repeated queries
4. Edge score evolution over time
5. Persistence and recovery

Author: Blake Chasteen
Date: November 8, 2025
"""

import asyncio
from pathlib import Path

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.spring_memory_scoring import (
    SpringMemoryScorer,
    AdaptiveSpringRetriever
)
from HoloLoom.documentation.types import MemoryShard


def create_test_knowledge_graph() -> KG:
    """Create a knowledge graph about machine learning concepts."""
    kg = KG()

    edges = [
        # Thompson Sampling cluster
        KGEdge("ThompsonSampling", "Bayesian", "IS_A", 1.0),
        KGEdge("ThompsonSampling", "MultiArmedBandit", "SOLVES", 1.0),
        KGEdge("ThompsonSampling", "Exploration", "USES", 0.9),

        # Bayesian cluster
        KGEdge("Bayesian", "PriorDistribution", "USES", 0.9),
        KGEdge("Bayesian", "PosteriorUpdate", "USES", 0.9),
        KGEdge("Bayesian", "GaussianProcess", "RELATED_TO", 0.7),

        # Multi-armed bandits cluster
        KGEdge("MultiArmedBandit", "Exploration", "INVOLVES", 0.8),
        KGEdge("MultiArmedBandit", "Exploitation", "INVOLVES", 0.8),
        KGEdge("MultiArmedBandit", "UCB", "ALTERNATIVE", 0.7),
        KGEdge("MultiArmedBandit", "EpsilonGreedy", "ALTERNATIVE", 0.7),

        # Exploration strategies
        KGEdge("Exploration", "EpsilonGreedy", "STRATEGY", 0.8),
        KGEdge("Exploration", "UCB", "STRATEGY", 0.8),
        KGEdge("UCB", "ConfidenceBound", "USES", 0.8),

        # Gaussian Process connections
        KGEdge("GaussianProcess", "BayesianOptimization", "USED_IN", 0.8),
        KGEdge("GaussianProcess", "Kernel", "USES", 0.9),
    ]

    kg.add_edges(edges)
    return kg


def create_test_memories() -> list:
    """Create test memory shards."""
    return [
        MemoryShard(
            id="mem_001",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits",
            entities=["ThompsonSampling", "Bayesian", "MultiArmedBandit"],
            metadata={"importance": 0.9}
        ),
        MemoryShard(
            id="mem_002",
            text="Bayesian methods use prior distributions and posterior updates",
            entities=["Bayesian", "PriorDistribution", "PosteriorUpdate"],
            metadata={"importance": 0.8}
        ),
        MemoryShard(
            id="mem_003",
            text="Multi-armed bandits solve exploration-exploitation tradeoffs",
            entities=["MultiArmedBandit", "Exploration", "Exploitation"],
            metadata={"importance": 0.85}
        ),
        MemoryShard(
            id="mem_004",
            text="UCB uses upper confidence bounds for exploration",
            entities=["UCB", "Exploration", "ConfidenceBound"],
            metadata={"importance": 0.75}
        ),
        MemoryShard(
            id="mem_005",
            text="Gaussian processes are used in Bayesian optimization",
            entities=["GaussianProcess", "BayesianOptimization"],
            metadata={"importance": 0.8}
        ),
        MemoryShard(
            id="mem_006",
            text="Epsilon-greedy is a simple exploration strategy",
            entities=["EpsilonGreedy", "Exploration"],
            metadata={"importance": 0.7}
        ),
    ]


async def demo_learning_progression():
    """Demonstrate how the system learns over multiple queries."""

    print("\n" + "="*70)
    print("DEMO: Spring Physics Memory Learning")
    print("="*70)

    # Setup
    kg = create_test_knowledge_graph()
    memories = create_test_memories()

    # Create scorer with persistence
    persist_path = Path("./demo_memory_scores.json")
    if persist_path.exists():
        persist_path.unlink()  # Fresh start

    scorer = SpringMemoryScorer(
        persist_path=str(persist_path),
        decay_rate=0.99,
        min_score_threshold=0.01
    )

    # Create adaptive retriever
    retriever = AdaptiveSpringRetriever(kg=kg, scorer=scorer)

    print(f"\n✓ Created knowledge graph: {len(kg.G.nodes)} nodes, {len(kg.G.edges)} edges")
    print(f"✓ Created {len(memories)} test memories")
    print(f"✓ Persistence path: {persist_path}")

    # Series of queries that form a pattern
    queries = [
        "Bayesian methods",
        "Thompson Sampling",
        "Bayesian optimization",
        "exploration strategies",
        "Bayesian methods",  # Repeat
        "Thompson Sampling",  # Repeat
        "multi-armed bandits",
        "Bayesian methods",  # Repeat again
    ]

    print("\n" + "-"*70)
    print("Query Sequence (Learning in Progress)")
    print("-"*70)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] '{query}'")

        # Retrieve
        results = await retriever.retrieve(
            query=query,
            memories=memories,
            limit=3,
            learn=True  # Enable learning
        )

        print(f"  Results: {len(results)} memories")
        for memory, score in results[:2]:
            print(f"    • [{score:.3f}] {memory.text[:55]}...")

        # Show learning stats
        stats = retriever.get_learning_stats()
        print(f"  Learning: {stats['total_edges']} edges tracked, "
              f"{stats['total_activations']} total activations")

    print("\n" + "="*70)
    print("Learning Results")
    print("="*70)

    # Show top learned edges
    print("\nTop 10 Learned Edges:")
    top_edges = scorer.get_top_edges(limit=10)

    for i, edge in enumerate(top_edges, 1):
        print(f"{i:2d}. {edge.source:20} → {edge.target:20} "
              f"[score: {edge.score:.3f}, count: {edge.access_count}, "
              f"avg: {edge.avg_activation:.3f}]")

    # Show statistics
    stats = scorer.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total edges tracked: {stats['total_edges']}")
    print(f"  Average score: {stats['avg_score']:.3f}")
    print(f"  Top score: {stats['top_score']:.3f}")
    print(f"  Total activations: {stats['total_activations']}")

    # Save learned patterns
    print("\n" + "-"*70)
    print("Saving Learned Patterns")
    print("-"*70)

    retriever.save_learned_patterns()
    print(f"✓ Saved {len(scorer.edge_scores)} edge scores to {persist_path}")

    # Demonstrate persistence
    print("\n" + "-"*70)
    print("Testing Persistence (New Session)")
    print("-"*70)

    # Create new scorer, load from disk
    scorer2 = SpringMemoryScorer(persist_path=str(persist_path))
    print(f"✓ Loaded {len(scorer2.edge_scores)} edge scores from disk")

    # Show that learning persisted
    top_edges2 = scorer2.get_top_edges(limit=5)
    print("\nTop 5 Edges (from saved state):")
    for i, edge in enumerate(top_edges2, 1):
        print(f"{i}. {edge.source} → {edge.target}: {edge.score:.3f}")

    print("\n" + "="*70)
    print("✓ Demo Complete!")
    print("")
    print("Key Takeaways:")
    print("1. System learns which edges are important from query patterns")
    print("2. Frequently activated edges get higher scores")
    print("3. Scores combine frequency (log), strength (avg), and recency (decay)")
    print("4. Learning persists across sessions via JSON")
    print("5. Self-improving retrieval with zero manual tuning!")
    print("="*70)
    print("")


async def demo_edge_score_evolution():
    """Show how a single edge's score evolves over time."""

    print("\n" + "="*70)
    print("DEMO: Edge Score Evolution")
    print("="*70)

    kg = KG()
    kg.add_edges([
        KGEdge("A", "B", "IS_A", 1.0),
        KGEdge("B", "C", "USES", 0.9),
    ])

    scorer = SpringMemoryScorer()

    # Simulate activations over time
    print("\nSimulating activation pattern for edge (A → B):")
    print("")
    print("Activation | Access Count | Avg Activation | Score")
    print("-" * 60)

    activations = [0.95, 0.88, 0.92, 0.78, 0.85, 0.91, 0.83, 0.89, 0.87, 0.93]

    for i, activation in enumerate(activations, 1):
        # Record activation
        scorer.record_activation_pattern(
            activations={"A": activation, "B": activation},
            kg_edges=[("A", "B")]
        )

        edge = scorer.edge_scores[("A", "B")]
        print(f"{activation:.2f}      | {edge.access_count:12d} | {edge.avg_activation:14.3f} | {edge.score:.3f}")

    print("\nObservations:")
    print("• Access count increases linearly")
    print("• Average activation stabilizes around 0.88")
    print("• Score grows logarithmically (frequency factor)")
    print("• Score = avg_activation × log(1 + access_count) × decay_factor")


async def main():
    """Run all demos."""

    # Main learning demonstration
    await demo_learning_progression()

    # Edge evolution demonstration
    await demo_edge_score_evolution()


if __name__ == "__main__":
    asyncio.run(main())
