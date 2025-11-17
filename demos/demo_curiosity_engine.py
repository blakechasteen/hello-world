"""
Curiosity Engine Demo
=====================

Demonstrates the Curiosity Engine suggesting exploration opportunities based on
knowledge gaps, contradictions, and access patterns.

This demo shows how the engine can proactively guide user learning.

Author: HoloLoom Team
Date: 2025-11-17
"""

import asyncio
from datetime import datetime, timedelta

# Note: This demo requires HoloLoom dependencies to be installed
# Run with: PYTHONPATH=. python demos/demo_curiosity_engine.py

try:
    from HoloLoom.memory.curiosity import CuriosityEngine, CuriosityConfig
    from HoloLoom.memory.graph import KG, KGEdge
    from HoloLoom.synthesis.types import ContradictionType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure HoloLoom dependencies are installed.")
    print("Run from repository root with: PYTHONPATH=. python demos/demo_curiosity_engine.py")
    exit(1)


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def print_suggestion(index: int, suggestion):
    """Print a formatted suggestion."""
    print(f"{index}. 💡 {suggestion.concept} ({suggestion.importance:.0%} important)")
    print(f"   Type: {suggestion.type.upper()}")
    print(f"   Reason: {suggestion.reason[:120]}...")
    print(f"   Query: \"{suggestion.suggested_query}\"")
    print(f"   Benefit: {suggestion.expected_benefit[:100]}...")
    print()


async def demo_gap_based_suggestions():
    """Demonstrate gap-based exploration suggestions."""
    print_header("Demo 1: Gap-Based Suggestions")

    # Create a knowledge graph with intentional gaps
    kg = KG()

    # Scenario: User knows about ML and neural networks, but missing prerequisites
    kg.add_edges([
        KGEdge("machine learning", "supervised learning", "HAS_SUBTYPE"),
        KGEdge("machine learning", "neural networks", "USES"),
        KGEdge("neural networks", "deep learning", "IS_A"),
        KGEdge("deep learning", "transformers", "USES"),

        # Gap: Feature engineering mentioned but not explained (bridge gap)
        KGEdge("machine learning", "feature engineering", "REQUIRES"),
        KGEdge("data science", "feature engineering", "REQUIRES"),

        # Gap: Missing linear algebra (prerequisite for neural networks)
        # (intentionally absent from graph)
    ])

    # Create curiosity engine
    config = CuriosityConfig(gap_importance_threshold=0.4)
    engine = CuriosityEngine(kg=kg, config=config)

    # Get suggestions
    suggestions = await engine.suggest_exploration(limit=5)

    gap_suggestions = [s for s in suggestions if s.type == 'gap']

    print(f"Found {len(gap_suggestions)} gap-based suggestions:\n")

    for i, suggestion in enumerate(gap_suggestions, 1):
        print_suggestion(i, suggestion)

    return engine


async def demo_contradiction_suggestions():
    """Demonstrate contradiction resolution suggestions."""
    print_header("Demo 2: Contradiction Resolution Suggestions")

    kg = KG()
    engine = CuriosityEngine(kg=kg)

    # Simulate user having contradictory knowledge about Python
    now = datetime.now()

    engine.contradiction_detector.add_memory({
        'id': 'm1',
        'content': 'Python is the best language for data science, very fast and efficient',
        'timestamp': now - timedelta(days=60),
        'entities': ['python', 'data science']
    })

    engine.contradiction_detector.add_memory({
        'id': 'm2',
        'content': 'Python is terrible for data science, way too slow compared to Julia',
        'timestamp': now - timedelta(days=5),
        'entities': ['python', 'data science']
    })

    # Also add a fact update contradiction
    engine.contradiction_detector.add_memory({
        'id': 'm3',
        'content': 'Paris has a population of 2.0 million people',
        'timestamp': now - timedelta(days=90),
        'entities': ['Paris', 'population']
    })

    engine.contradiction_detector.add_memory({
        'id': 'm4',
        'content': 'Paris has a population of 2.16 million people',
        'timestamp': now - timedelta(days=10),
        'entities': ['Paris', 'population']
    })

    # Get contradiction suggestions
    suggestions = await engine.suggest_contradiction_resolution()

    print(f"Found {len(suggestions)} contradiction resolution suggestions:\n")

    for i, suggestion in enumerate(suggestions, 1):
        print_suggestion(i, suggestion)
        if 'contradiction_type' in suggestion.metadata:
            print(f"   Contradiction type: {suggestion.metadata['contradiction_type']}")
            print(f"   Temporal gap: {suggestion.metadata.get('temporal_gap_days', 0)} days\n")

    return engine


async def demo_access_pattern_suggestions():
    """Demonstrate access pattern-based suggestions (trending, related)."""
    print_header("Demo 3: Access Pattern Suggestions")

    # Create knowledge graph with related concepts
    kg = KG()
    kg.add_edges([
        KGEdge("transformers", "attention mechanism", "USES"),
        KGEdge("transformers", "self-attention", "USES"),
        KGEdge("transformers", "BERT", "INSTANCE"),
        KGEdge("transformers", "GPT", "INSTANCE"),
        KGEdge("transformers", "encoder-decoder", "HAS_ARCHITECTURE"),
        KGEdge("attention mechanism", "scaled dot-product", "USES"),
        KGEdge("BERT", "masked language modeling", "USES"),
        KGEdge("GPT", "autoregressive", "IS_A"),
    ])

    engine = CuriosityEngine(kg=kg)

    # Simulate user exploring transformers heavily (trending topic)
    now = datetime.now()
    for i in range(6):
        engine.track_query(
            f"Question about transformers architecture #{i}",
            entities=["transformers"],
            timestamp=now - timedelta(minutes=i*10)
        )

    # Get access pattern suggestions
    suggestions = await engine._suggest_from_access_patterns()

    print(f"User has been exploring 'transformers' heavily (6 recent queries)")
    print(f"\nFound {len(suggestions)} access pattern suggestions:\n")

    for i, suggestion in enumerate(suggestions[:5], 1):
        print_suggestion(i, suggestion)

    return engine


async def demo_related_concepts():
    """Demonstrate related concept discovery."""
    print_header("Demo 4: Related Concept Discovery")

    # Create a rich knowledge graph
    kg = KG()
    kg.add_edges([
        # Python ecosystem
        KGEdge("python", "numpy", "HAS_LIBRARY"),
        KGEdge("python", "pandas", "HAS_LIBRARY"),
        KGEdge("python", "scikit-learn", "HAS_LIBRARY"),
        KGEdge("python", "tensorflow", "HAS_LIBRARY"),
        KGEdge("numpy", "array operations", "PROVIDES"),
        KGEdge("pandas", "dataframes", "PROVIDES"),
        KGEdge("scikit-learn", "machine learning", "PROVIDES"),
        KGEdge("tensorflow", "deep learning", "PROVIDES"),

        # Second-hop relationships
        KGEdge("machine learning", "supervised learning", "HAS_TYPE"),
        KGEdge("deep learning", "neural networks", "USES"),
    ])

    engine = CuriosityEngine(kg=kg)

    # User is currently exploring Python
    concept = "python"
    related = await engine.suggest_related_concepts(concept, max_suggestions=5)

    print(f"User is currently exploring: '{concept}'")
    print(f"\nRelated concepts you might enjoy (1-hop and 2-hop):\n")

    for i, related_concept in enumerate(related, 1):
        print(f"{i}. {related_concept}")

    print(f"\nThese suggestions lead to natural learning paths!")

    return engine


async def demo_deep_dive_suggestions():
    """Demonstrate deep dive suggestions for partially-explored topics."""
    print_header("Demo 5: Deep Dive Suggestions")

    kg = KG()
    kg.add_edges([
        KGEdge("web development", "frontend", "HAS_PART"),
        KGEdge("web development", "backend", "HAS_PART"),
        KGEdge("frontend", "react", "USES"),
        KGEdge("react", "components", "HAS_CONCEPT"),
    ])

    engine = CuriosityEngine(kg=kg)

    # Simulate partial exploration of React (touched on but not deeply explored)
    now = datetime.now()
    engine.track_query("What is React?", entities=["react"], timestamp=now - timedelta(hours=48))
    engine.track_query("React components basics", entities=["react"], timestamp=now - timedelta(hours=24))
    engine.track_query("Quick React question", entities=["react"], timestamp=now - timedelta(hours=12))

    suggestions = await engine._suggest_deep_dives()

    print(f"User has touched on 'react' a few times but hasn't explored deeply")
    print(f"\nFound {len(suggestions)} deep dive suggestions:\n")

    for i, suggestion in enumerate(suggestions, 1):
        print_suggestion(i, suggestion)
        access_count = suggestion.metadata.get('access_count', 0)
        print(f"   Access count: {access_count} times\n")

    return engine


async def demo_comprehensive_suggestions():
    """Demonstrate comprehensive suggestion generation (all types combined)."""
    print_header("Demo 6: Comprehensive Exploration Suggestions")

    # Create a realistic knowledge graph
    kg = KG()
    kg.add_edges([
        # User knows about machine learning
        KGEdge("machine learning", "supervised learning", "HAS_TYPE"),
        KGEdge("machine learning", "neural networks", "USES"),

        # But missing prerequisites
        # (linear algebra, calculus intentionally absent)

        # Bridge gap: reinforcement learning mentioned but not explored
        KGEdge("machine learning", "reinforcement learning", "HAS_TYPE"),

        # Partial knowledge of Python
        KGEdge("programming", "python", "HAS_LANGUAGE"),
        KGEdge("python", "data science", "USED_IN"),

        # Related concepts
        KGEdge("neural networks", "backpropagation", "USES"),
        KGEdge("neural networks", "activation functions", "USES"),
        KGEdge("supervised learning", "regression", "HAS_TYPE"),
        KGEdge("supervised learning", "classification", "HAS_TYPE"),
    ])

    config = CuriosityConfig(
        max_suggestions=10,
        gap_importance_threshold=0.3,
        enable_serendipity=True,
        serendipity_probability=0.3
    )

    engine = CuriosityEngine(kg=kg, config=config)

    # Simulate user session
    now = datetime.now()

    # Heavy exploration of neural networks (trending)
    for i in range(5):
        engine.track_query(
            f"Neural networks question {i}",
            entities=["neural networks"],
            timestamp=now - timedelta(minutes=i*15)
        )

    # Partial exploration of Python (deep dive candidate)
    engine.track_query("Python basics", entities=["python"], timestamp=now - timedelta(hours=48))
    engine.track_query("Python for ML", entities=["python"], timestamp=now - timedelta(hours=24))
    engine.track_query("Python libraries", entities=["python"], timestamp=now - timedelta(hours=12))

    # Add a contradiction
    engine.contradiction_detector.add_memory({
        'id': 'm1',
        'content': 'Neural networks are easy to train',
        'timestamp': now - timedelta(days=30),
        'entities': ['neural networks']
    })
    engine.contradiction_detector.add_memory({
        'id': 'm2',
        'content': 'Neural networks are extremely difficult to train',
        'timestamp': now - timedelta(days=3),
        'entities': ['neural networks']
    })

    # Get comprehensive suggestions
    suggestions = await engine.suggest_exploration(limit=10)

    print(f"Analyzing your learning journey...\n")
    print(f"Recent activity:")
    print(f"  - Heavily exploring: neural networks (5 recent queries)")
    print(f"  - Partially explored: python (3 queries over 2 days)")
    print(f"  - Knowledge graph: {len(kg.get_all_nodes())} concepts, {len(list(kg.G.edges()))} connections")
    print(f"\n")

    print(f"Generated {len(suggestions)} personalized exploration suggestions:\n")

    for i, suggestion in enumerate(suggestions, 1):
        print_suggestion(i, suggestion)

    # Show statistics
    print(f"\n{'='*70}")
    print("Curiosity Engine Statistics")
    print(f"{'='*70}\n")

    stats = engine.get_statistics()
    print(f"Total queries tracked: {stats['total_queries_tracked']}")
    print(f"Unique entities accessed: {stats['unique_entities_accessed']}")
    print(f"\nTop trending entities:")
    for entity, count in stats['top_trending_entities'][:5]:
        print(f"  - {entity}: {count} accesses")

    return engine


async def main():
    """Run all demos."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         HoloLoom Curiosity Engine - Interactive Demo              ║")
    print("║                                                                    ║")
    print("║  Proactive exploration suggestions based on knowledge gaps,        ║")
    print("║  contradictions, and access patterns.                             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # Run demos sequentially
    await demo_gap_based_suggestions()
    await demo_contradiction_suggestions()
    await demo_access_pattern_suggestions()
    await demo_related_concepts()
    await demo_deep_dive_suggestions()
    await demo_comprehensive_suggestions()

    print_header("Demo Complete!")
    print("The Curiosity Engine can be integrated into:")
    print("  • Research Assistant chatbot (Streamlit sidebar)")
    print("  • VS Code extension (proactive learning suggestions)")
    print("  • Terminal UI (periodic exploration prompts)")
    print("  • Web dashboard (learning path visualization)")
    print("\nIt acts as a curious companion, always looking for interesting")
    print("paths to explore and connections to make.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
