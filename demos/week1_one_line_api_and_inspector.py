"""
Week 1 Demo: One-Line API + Memory Inspector UI
================================================

Demonstrates HoloLoom's new simplified API and transparency features:

1. One-Line API - Zero configuration memory operations
2. Memory Inspector - Complete retrieval transparency

This is the foundation for making HoloLoom the most impressive
persistent memory system on the planet.

Usage:
    PYTHONPATH=. python demos/week1_one_line_api_and_inspector.py

Output:
    - Console output showing one-line API usage
    - HTML file: demos/output/week1_memory_inspector.html
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hololoom_simple as loom
from HoloLoom.visualization.memory_inspector import (
    render_memory_inspector,
    InspectionResult,
    RetrievalScore
)


def demo_one_line_api():
    """Demonstrate the magical one-line API."""
    print("=" * 70)
    print("Part 1: One-Line API Demo")
    print("=" * 70)
    print()

    print("📝 Storing memories (zero configuration)...\n")

    # Just call remember() - that's it!
    loom.remember("Thompson Sampling balances exploration and exploitation")
    loom.remember("Python is great for machine learning")
    loom.remember("HoloLoom uses Matryoshka embeddings for multi-scale retrieval")
    loom.remember("The knowledge graph stores entity relationships")
    loom.remember("Neural networks can be trained with PPO reinforcement learning")
    loom.remember("TypeScript adds static typing to JavaScript")
    loom.remember("Claude Code is an AI coding assistant")
    loom.remember("Attention mechanisms are key to transformer architectures")

    print("✓ Stored 8 memories\n")

    print("🔍 Recalling memories (just ask)...\n")

    # Simple recall
    results = loom.recall("What do I know about machine learning?", limit=5)

    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['text']}")
        print(f"     Confidence: {result['confidence']:.3f}\n")

    print("📊 System metrics...\n")

    # Get statistics
    stats = loom.metrics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✨ No async/await, no configuration, just works!\n")

    return results


def demo_memory_inspector(recall_results):
    """Demonstrate Memory Inspector UI."""
    print("=" * 70)
    print("Part 2: Memory Inspector UI Demo")
    print("=" * 70)
    print()

    print("🔬 Creating detailed inspection visualization...\n")

    # Create mock inspection result with detailed scores
    scores = []
    for i, result in enumerate(recall_results):
        # Simulate component scores
        score = RetrievalScore(
            memory_id=result['id'],
            memory_text=result['text'],
            total_score=result['confidence'],
            bm25_score=0.85 - (i * 0.05),
            semantic_score=result['confidence'],
            temporal_score=0.90 - (i * 0.03),
            graph_proximity_score=0.75 - (i * 0.04),
            recency_score=1.0 - (i * 0.1),
            cache_hit=i % 2 == 0,  # Alternate cache hits/misses
            retrieval_path=[f"node_{j}" for j in range(3)],
            access_count=10 - i,
            component_weights={
                'bm25': 0.25,
                'semantic': 0.35,
                'temporal': 0.15,
                'graph_proximity': 0.15,
                'recency': 0.10
            }
        )
        scores.append(score)

    # Create inspection result
    inspection = InspectionResult(
        query="What do I know about machine learning?",
        retrieved_memories=scores,
        total_memories_searched=8,
        cache_hit_rate=0.50,
        avg_retrieval_time_ms=125.5,
        bandit_stats={
            'answer': {
                'alpha': 15.2,
                'beta': 3.8,
                'expected_reward': 0.800,
                'selections': 12
            },
            'search': {
                'alpha': 8.5,
                'beta': 5.2,
                'expected_reward': 0.620,
                'selections': 8
            },
            'retrieve': {
                'alpha': 12.0,
                'beta': 4.0,
                'expected_reward': 0.750,
                'selections': 10
            }
        },
        knowledge_graph_nodes=['ml', 'python', 'embedding', 'thompson_sampling'],
        knowledge_graph_edges=[
            ('ml', 'python', 'USES'),
            ('ml', 'embedding', 'USES'),
            ('thompson_sampling', 'ml', 'PART_OF')
        ]
    )

    # Render visualization
    html = render_memory_inspector(
        inspection,
        title="HoloLoom Memory Inspector",
        subtitle="Week 1 Demo - Complete Retrieval Transparency",
        show_graph=False,  # Simplified for demo
        show_bandit_stats=True,
        top_n=5
    )

    # Save to file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "week1_memory_inspector.html"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Saved visualization to: {output_file}\n")
    print("📊 Visualization includes:")
    print("  • Score breakdown table (BM25, semantic, temporal, graph, recency)")
    print("  • Component weight distribution")
    print("  • Cache performance metrics")
    print("  • Thompson Sampling bandit statistics")
    print("  • Complete transparency into WHY each memory was retrieved\n")

    return str(output_file)


def demo_advanced_features():
    """Show some advanced features of the one-line API."""
    print("=" * 70)
    print("Part 3: Advanced Features")
    print("=" * 70)
    print()

    print("🔍 Search with higher limit...\n")
    results = loom.search("programming languages", limit=10)
    print(f"Found {len(results)} results\n")

    print("📈 System status...\n")
    status = loom.status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print()


def main():
    """Run complete Week 1 demo."""
    print("\n" + "="*70)
    print("WEEK 1 DEMO: One-Line API + Memory Inspector")
    print("Making HoloLoom the Most Impressive Memory System on Earth")
    print("="*70 + "\n")

    try:
        # Part 1: One-line API
        recall_results = demo_one_line_api()

        # Part 2: Memory Inspector
        output_file = demo_memory_inspector(recall_results)

        # Part 3: Advanced features
        demo_advanced_features()

        # Final summary
        print("=" * 70)
        print("✅ WEEK 1 COMPLETE!")
        print("=" * 70)
        print()
        print("Achievements:")
        print("  ✓ One-line API: Zero-config memory operations")
        print("  ✓ Memory Inspector: Complete retrieval transparency")
        print()
        print("Next Steps:")
        print("  → Week 2: Benchmark suite + public results")
        print("  → Week 3: DreamEngine (memory synthesis)")
        print("  → Week 4: Personal Research Assistant demo")
        print()
        print(f"📁 Open visualization: {output_file}")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
