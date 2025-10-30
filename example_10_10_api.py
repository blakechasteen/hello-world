"""
Example: HoloLoom 10/10 API

Demonstrates the perfect, minimal, inevitable interface.

Everything is a memory operation:
- experience() → form memories
- recall() → activate memories
- reflect() → learn from feedback
"""

import sys
sys.path.insert(0, '.')

import asyncio
from HoloLoom import HoloLoom, Memory, ActivationStrategy


async def example_basic():
    """Basic usage: Experience and recall."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    print()

    # Initialize (that's it!)
    loom = HoloLoom()
    print("✓ Created HoloLoom system")
    print()

    # Experience content
    print("Experiencing content...")
    mem1 = await loom.experience("Thompson Sampling balances exploration and exploitation")
    mem2 = await loom.experience("Epsilon-Greedy is a simple bandit algorithm")
    mem3 = await loom.experience("UCB uses optimistic initialization")
    print(f"  ✓ Formed {3} memories")
    print()

    # Recall related memories
    print("Recalling: 'What are bandit algorithms?'")
    memories = await loom.recall("What are bandit algorithms?")
    print(f"  ✓ Recalled {len(memories)} memories:")
    for mem in memories:
        print(f"    - {mem.text[:50]}...")
    print()

    # System summary
    print(loom.summary())


async def example_multimodal():
    """Multimodal: Text + structured data."""
    print("=" * 70)
    print("EXAMPLE 2: Multimodal (Text + Structured Data)")
    print("=" * 70)
    print()

    loom = HoloLoom()
    print("✓ Created HoloLoom system")
    print()

    # Experience text
    print("Experiencing text...")
    mem1 = await loom.experience("Python is a high-level programming language")
    print(f"  ✓ Text memory: {mem1.id[:8]}")

    # Experience structured data
    print("Experiencing structured data...")
    mem2 = await loom.experience({
        "language": "Python",
        "paradigm": ["object-oriented", "functional", "procedural"],
        "typing": "dynamic"
    })
    print(f"  ✓ Structured memory: {mem2.id[:8]}")
    print(f"      Modality: {mem2.context.get('modality', 'text')}")
    print()

    # Recall with text query (cross-modal!)
    print("Recalling: 'Tell me about Python'")
    memories = await loom.recall("Tell me about Python")
    print(f"  ✓ Recalled {len(memories)} memories (cross-modal!)")
    for mem in memories:
        modality = mem.context.get('modality', 'text')
        print(f"    - [{modality}] {mem.text[:40]}...")
    print()


async def example_strategies():
    """Different recall strategies."""
    print("=" * 70)
    print("EXAMPLE 3: Recall Strategies")
    print("=" * 70)
    print()

    loom = HoloLoom()
    print("✓ Created HoloLoom system")
    print()

    # Build knowledge base
    print("Building knowledge base...")
    topics = [
        "Python has decorators for metaprogramming",
        "Python has generators for lazy evaluation",
        "Python has list comprehensions for concise loops",
        "JavaScript has promises for async programming",
        "JavaScript has arrow functions for brevity"
    ]

    for topic in topics:
        await loom.experience(topic)
    print(f"  ✓ Experienced {len(topics)} topics")
    print()

    # PRECISE recall (narrow, high confidence)
    print("PRECISE recall: 'Python features'")
    memories = await loom.recall("Python features", strategy=ActivationStrategy.PRECISE)
    print(f"  ✓ Recalled {len(memories)} memories (narrow search)")
    print()

    # BALANCED recall (default)
    print("BALANCED recall: 'Python features'")
    memories = await loom.recall("Python features", strategy=ActivationStrategy.BALANCED)
    print(f"  ✓ Recalled {len(memories)} memories (balanced)")
    print()

    # EXPLORATORY recall (broad)
    print("EXPLORATORY recall: 'programming'")
    memories = await loom.recall("programming", strategy=ActivationStrategy.EXPLORATORY)
    print(f"  ✓ Recalled {len(memories)} memories (broad search)")
    print()


async def example_reflection():
    """Reflection: Learning from feedback."""
    print("=" * 70)
    print("EXAMPLE 4: Reflection and Learning")
    print("=" * 70)
    print()

    loom = HoloLoom()
    print("✓ Created HoloLoom system")
    print()

    # Experience content
    print("Experiencing algorithms...")
    await loom.experience("Thompson Sampling uses Bayesian inference")
    await loom.experience("Q-Learning uses temporal difference")
    await loom.experience("SARSA is an on-policy algorithm")
    print("  ✓ Formed 3 memories")
    print()

    # Recall
    print("Recalling: 'Bayesian algorithms'")
    memories = await loom.recall("Bayesian algorithms")
    print(f"  ✓ Recalled {len(memories)} memories")
    print()

    # Reflect with positive feedback
    print("Reflecting on helpful memories...")
    await loom.reflect(memories, feedback={
        "helpful": True,
        "relevance": 0.9,
        "outcome": "answered_question"
    })
    print("  ✓ Reflection complete (system learns from feedback)")
    print()


async def example_batch():
    """Batch operations."""
    print("=" * 70)
    print("EXAMPLE 5: Batch Operations")
    print("=" * 70)
    print()

    loom = HoloLoom()
    print("✓ Created HoloLoom system")
    print()

    # Batch experience
    print("Batch experiencing...")
    contents = [
        "Sorting algorithms: bubble, merge, quick",
        "Search algorithms: binary, depth-first, breadth-first",
        "Graph algorithms: Dijkstra, A*, Bellman-Ford"
    ]

    memories = await loom.experience_batch(contents)
    print(f"  ✓ Formed {len(memories)} memories in batch")
    print()

    # Search (alias for recall)
    print("Searching: 'graph algorithms'")
    results = await loom.search("graph algorithms", limit=2)
    print(f"  ✓ Found {len(results)} results:")
    for result in results:
        print(f"    - {result.text[:50]}...")
    print()

    # Metrics
    metrics = loom.get_metrics()
    print(f"System metrics:")
    print(f"  Total memories: {metrics['n_memories']}")
    print(f"  Total connections: {metrics['n_connections']}")
    print(f"  Active memories: {metrics['n_active']}")
    print()


async def example_context_manager():
    """Async context manager."""
    print("=" * 70)
    print("EXAMPLE 6: Context Manager (Automatic Cleanup)")
    print("=" * 70)
    print()

    # Use with context manager for automatic cleanup
    async with HoloLoom() as loom:
        print("✓ Created HoloLoom system (in context)")
        print()

        memory = await loom.experience("Context managers ensure cleanup")
        memories = await loom.recall("cleanup")

        print(f"  ✓ Formed 1 memory, recalled {len(memories)}")
        print()

        # Automatic cleanup on exit
    print("✓ Context exited (automatic cleanup)")
    print()


async def run_all_examples():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                  HoloLoom 10/10 API Examples                       ║")
    print("║                                                                    ║")
    print("║  Everything is a memory operation:                                ║")
    print("║  - experience() → form memories                                   ║")
    print("║  - recall() → activate memories                                   ║")
    print("║  - reflect() → learn from feedback                                ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")

    examples = [
        ("Basic Usage", example_basic),
        ("Multimodal", example_multimodal),
        ("Recall Strategies", example_strategies),
        ("Reflection", example_reflection),
        ("Batch Operations", example_batch),
        ("Context Manager", example_context_manager),
    ]

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

        input("\nPress Enter to continue...")

    print("\n")
    print("=" * 70)
    print("All examples complete!")
    print("=" * 70)
    print()
    print("The 10/10 API:")
    print("  • Minimal (3 methods)")
    print("  • Intuitive (self-documenting)")
    print("  • Powerful (handles any modality)")
    print("  • Inevitable (of course it works this way)")
    print()


if __name__ == "__main__":
    asyncio.run(run_all_examples())
