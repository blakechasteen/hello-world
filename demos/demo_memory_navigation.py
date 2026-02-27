"""
Demo: UnifiedMemory Navigation & Pattern Discovery

Demonstrates the new spatial navigation and pattern discovery features:
- Navigate in 4 directions: FORWARD, BACKWARD, SIDEWAYS, DEEP
- Discover 4 pattern types: LOOP, CLUSTER, RESONANCE, THREAD

Author: Claude Code
Date: 2025-11-22
"""

import asyncio
from datetime import datetime
from hololoom import hololoom
from hololoom.memory.unified import NavigationDirection
from hololoom.config import Config


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_memories(memories, title: str):
    """Print list of memories."""
    print(f"\n{title}:")
    if not memories:
        print("  (none found)")
        return

    for i, mem in enumerate(memories, 1):
        context_info = mem.context.get('direction', '') or mem.context.get('step', '')
        if context_info:
            print(f"  {i}. {mem.id} (context: {context_info})")
        else:
            print(f"  {i}. {mem.id}")


def print_patterns(patterns, title: str):
    """Print list of patterns."""
    print(f"\n{title}:")
    if not patterns:
        print("  (none found)")
        return

    for i, pattern in enumerate(patterns, 1):
        print(f"\n  {i}. {pattern.pattern_type.upper()} "
              f"(strength: {pattern.strength:.2f})")
        print(f"     {pattern.description}")
        print(f"     Memories: {', '.join(pattern.memories[:5])}"
              f"{'...' if len(pattern.memories) > 5 else ''}")


async def demo_navigation():
    """Demonstrate spatial navigation in 4 directions."""
    print_section("Demo 1: Spatial Navigation")

    print("Creating knowledge graph with connected memories...")

    async with HoloLoom() as loom:
        # Create a narrative chain
        await loom.experience("Thompson Sampling is a Bayesian exploration strategy")
        await loom.experience("It balances exploration and exploitation")
        await loom.experience("This leads to optimal decision making")
        await loom.experience("Which enables self-improving systems")

        # Create related but divergent memories
        await loom.experience("Bayesian methods use prior distributions")
        await loom.experience("Prior distributions encode domain knowledge")

        # Create a cluster of related concepts
        await loom.experience("Neural networks learn from data")
        await loom.experience("Deep learning uses multiple layers")
        await loom.experience("Backpropagation updates weights")

        print("✓ Created 9 interconnected memories\n")

        # Access underlying memory backend for navigation
        memory = loom._memory

        # Demonstrate FORWARD navigation
        print("─" * 70)
        print("Navigation Direction: FORWARD (following successors)")
        print("─" * 70)

        forward_path = memory.navigate(
            from_memory="thompson_sampling",
            direction=NavigationDirection.FORWARD,
            steps=3
        )

        print_memories(forward_path, "Forward path from 'Thompson Sampling'")

        if forward_path:
            print("\n📖 Interpretation: Following the narrative thread forward")
            print("   Each memory builds on the previous one")

        # Demonstrate BACKWARD navigation
        print("\n" + "─" * 70)
        print("Navigation Direction: BACKWARD (following predecessors)")
        print("─" * 70)

        backward_path = memory.navigate(
            from_memory="self-improving",
            direction=NavigationDirection.BACKWARD,
            steps=3
        )

        print_memories(backward_path, "Backward path from 'self-improving systems'")

        if backward_path:
            print("\n📖 Interpretation: Tracing back to foundational concepts")
            print("   Understanding what led to this conclusion")

        # Demonstrate SIDEWAYS navigation
        print("\n" + "─" * 70)
        print("Navigation Direction: SIDEWAYS (siblings & related concepts)")
        print("─" * 70)

        sideways_path = memory.navigate(
            from_memory="bayesian_methods",
            direction=NavigationDirection.SIDEWAYS,
            steps=3
        )

        print_memories(sideways_path, "Sideways from 'Bayesian methods'")

        if sideways_path:
            print("\n📖 Interpretation: Discovering related concepts at same level")
            print("   Exploring alternatives and parallel ideas")

        # Demonstrate DEEP navigation
        print("\n" + "─" * 70)
        print("Navigation Direction: DEEP (cycles & strange loops)")
        print("─" * 70)

        deep_path = memory.navigate(
            from_memory="thompson_sampling",
            direction=NavigationDirection.DEEP,
            steps=5
        )

        print_memories(deep_path, "Deep exploration from 'Thompson Sampling'")

        if deep_path:
            print("\n📖 Interpretation: Finding interconnections and feedback loops")
            print("   Revealing the holistic structure of knowledge")


async def demo_pattern_discovery():
    """Demonstrate pattern discovery (loops, clusters, threads)."""
    print_section("Demo 2: Pattern Discovery")

    print("Creating knowledge graph with various structures...")

    async with HoloLoom() as loom:
        # Create narrative thread
        await loom.experience("Research begins with a question")
        await loom.experience("Questions lead to hypotheses")
        await loom.experience("Hypotheses guide experiments")
        await loom.experience("Experiments produce data")
        await loom.experience("Data answers the question")  # Completes loop!

        # Create a cluster of tightly related concepts
        await loom.experience("Machine learning uses algorithms")
        await loom.experience("Algorithms process training data")
        await loom.experience("Training data contains patterns")
        await loom.experience("Patterns inform predictions")

        # Create another cluster
        await loom.experience("Neural networks have layers")
        await loom.experience("Layers contain neurons")
        await loom.experience("Neurons activate on features")

        print("✓ Created 12 interconnected memories\n")

        memory = loom._memory

        # Discover all patterns
        print("─" * 70)
        print("Discovering patterns in the knowledge graph...")
        print("─" * 70)

        patterns = memory.discover_patterns(
            pattern_types=["loop", "cluster", "thread"],
            min_strength=0.3
        )

        # Separate by type
        loops = [p for p in patterns if p.pattern_type == "loop"]
        clusters = [p for p in patterns if p.pattern_type == "cluster"]
        threads = [p for p in patterns if p.pattern_type == "thread"]

        print_patterns(loops, "Strange Loops (cyclical connections)")

        if loops:
            print("\n📖 Interpretation: Feedback loops show recursive relationships")
            print("   These create self-reinforcing knowledge structures")

        print_patterns(clusters, "Clusters (tightly connected groups)")

        if clusters:
            print("\n📖 Interpretation: Clusters indicate coherent topics")
            print("   Memories that belong together conceptually")

        print_patterns(threads, "Narrative Threads (causal chains)")

        if threads:
            print("\n📖 Interpretation: Threads show cause-effect relationships")
            print("   The story of how ideas connect and evolve")


async def demo_integration():
    """Demonstrate combined navigation + pattern discovery."""
    print_section("Demo 3: Integration - Navigate + Discover")

    print("Showing how navigation and pattern discovery work together...")

    async with HoloLoom() as loom:
        # Create a knowledge graph
        await loom.experience("Start exploring knowledge graphs")
        await loom.experience("Knowledge graphs store entities and relationships")
        await loom.experience("Entities can be connected in multiple ways")
        await loom.experience("Multiple connections create rich semantic networks")
        await loom.experience("Semantic networks enable intelligent retrieval")

        await loom.experience("Graph traversal algorithms navigate these networks")
        await loom.experience("Algorithms can follow different strategies")

        print("✓ Created knowledge graph\n")

        memory = loom._memory

        # Step 1: Navigate to explore
        print("Step 1: Navigate forward to explore the graph")
        print("─" * 70)

        path = memory.navigate(
            from_memory="knowledge_graphs",
            direction=NavigationDirection.FORWARD,
            steps=4
        )

        print_memories(path, "Forward exploration path")

        # Step 2: Discover patterns
        print("\n\nStep 2: Discover patterns in the entire graph")
        print("─" * 70)

        patterns = memory.discover_patterns(
            pattern_types=["loop", "cluster", "thread"],
            min_strength=0.3
        )

        print_patterns(patterns[:3], "Top 3 discovered patterns")

        # Step 3: Navigate based on patterns
        if patterns:
            print("\n\nStep 3: Navigate based on discovered patterns")
            print("─" * 70)

            # Pick first pattern and navigate from one of its memories
            pattern = patterns[0]
            if pattern.memories:
                start_memory = pattern.memories[0]

                print(f"\nNavigating from '{start_memory}' "
                      f"(found in {pattern.pattern_type} pattern)")

                path = memory.navigate(
                    from_memory=start_memory,
                    direction=NavigationDirection.SIDEWAYS,
                    steps=2
                )

                print_memories(path, "Sideways navigation from pattern")

                print("\n📖 Interpretation: Patterns guide exploration")
                print("   Use discovered structure to navigate intelligently")


async def main():
    """Run all demos."""
    print("\n" + "═" * 70)
    print(" " * 15 + "UnifiedMemory Navigation & Pattern Discovery")
    print(" " * 25 + "Interactive Demo")
    print("═" * 70)

    print("\n📚 This demo showcases HoloLoom's new spatial memory features:")
    print("   • 4 navigation directions (FORWARD/BACKWARD/SIDEWAYS/DEEP)")
    print("   • 4 pattern types (LOOP/CLUSTER/RESONANCE/THREAD)")
    print("   • Integration of navigation + pattern discovery\n")

    try:
        await demo_navigation()
        await demo_pattern_discovery()
        await demo_integration()

        print_section("Summary")

        print("✅ Navigation Capabilities:")
        print("   • FORWARD: Follow successors (what comes next)")
        print("   • BACKWARD: Follow predecessors (what came before)")
        print("   • SIDEWAYS: Find siblings (related concepts)")
        print("   • DEEP: Explore cycles (holistic connections)\n")

        print("✅ Pattern Discovery:")
        print("   • LOOP: Strange loops (recursive relationships)")
        print("   • CLUSTER: Tightly connected groups (topics)")
        print("   • THREAD: Narrative chains (causal sequences)")
        print("   • RESONANCE: Highly activated memories (hot topics)\n")

        print("🎯 Use Cases:")
        print("   • Explore knowledge graphs intuitively")
        print("   • Discover emergent structure in memories")
        print("   • Navigate semantic space efficiently")
        print("   • Understand relationships between concepts\n")

        print("═" * 70)
        print(" " * 25 + "Demo Complete!")
        print("═" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
