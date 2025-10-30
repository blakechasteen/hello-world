"""
Comprehensive verification of Awareness Architecture.

Tests the core integration of semantic calculus + memory system.
Runs 5 critical tests before integration into WeavingOrchestrator.
"""

import sys
sys.path.insert(0, '.')

import asyncio
import numpy as np
from typing import List

# HoloLoom imports
from HoloLoom.memory.awareness_graph import AwarenessGraph
from HoloLoom.memory.awareness_types import (
    SemanticPerception,
    ActivationStrategy,
    ActivationBudget,
    EdgeType
)
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
import networkx as nx


async def test_perceive_remember_activate():
    """Test 1: Core perceive → remember → activate cycle."""
    print("\n[TEST 1] Core Cycle: Perceive → Remember → Activate")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # 1. Perceive input
    text = "Thompson Sampling balances exploration and exploitation in reinforcement learning."
    perception = await awareness.perceive(text)

    assert perception.position.shape == (228,), "Position should be 228D (EXTENDED_244_DIMENSIONS)"
    assert len(perception.dominant_dimensions) > 0, "Should detect dominant dimensions"
    print(f"  ✓ Perceived: {perception.dominant_dimensions[:3]}")

    # 2. Remember
    memory_id = await awareness.remember(text, perception, context={"topic": "RL"})
    assert memory_id in graph.nodes, "Memory should be stored in graph"
    assert memory_id in awareness.semantic_positions, "Should track semantic position"
    print(f"  ✓ Remembered: {memory_id[:8]}...")

    # 3. Activate
    query_perception = await awareness.perceive("What is Thompson Sampling?")
    activated = await awareness.activate(query_perception,
                                        strategy=ActivationStrategy.BALANCED)

    assert len(activated) > 0, "Should activate relevant memories"
    assert any(mem.text == text for mem in activated), "Should retrieve original memory"
    print(f"  ✓ Activated: {len(activated)} memories")

    return True


async def test_topic_shift_detection():
    """Test 2: Topic shift detection with PRECISE strategy."""
    print("\n[TEST 2] Topic Shift Detection")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Build conversation about Python
    python_queries = [
        "How do I define a function in Python?",
        "What are Python decorators?",
        "Explain list comprehensions in Python"
    ]

    for text in python_queries:
        p = await awareness.perceive(text)
        await awareness.remember(text, p)

    print(f"  → Stored {len(python_queries)} Python memories")

    # Query within same topic (should activate memories)
    perception_same = await awareness.perceive("Tell me about Python classes")
    activated_same = await awareness.activate(perception_same,
                                             strategy=ActivationStrategy.PRECISE)

    print(f"  → Same topic query activated: {len(activated_same)} memories")

    # Query with topic shift (should activate few/no memories)
    perception_shifted = await awareness.perceive("What's the weather like today?")
    activated_shifted = await awareness.activate(perception_shifted,
                                                strategy=ActivationStrategy.PRECISE)

    print(f"  → Shifted topic query activated: {len(activated_shifted)} memories")

    # Topic shift detection logic
    shift_detected = len(activated_shifted) < len(activated_same)
    assert shift_detected, "Should detect topic shift (fewer activations)"
    print(f"  ✓ Topic shift correctly detected")

    return True


async def test_context_window_budgeting():
    """Test 3: Context window budgeting adapts to constraints."""
    print("\n[TEST 3] Context Window Budgeting")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Store multiple memories
    texts = [
        "Machine learning uses algorithms to learn patterns.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "Backpropagation trains neural networks efficiently.",
        "Gradient descent optimizes model parameters.",
        "Overfitting occurs when models memorize training data.",
        "Regularization prevents overfitting in models.",
        "Cross-validation evaluates model generalization."
    ]

    for text in texts:
        p = await awareness.perceive(text)
        await awareness.remember(text, p)

    print(f"  → Stored {len(texts)} memories")

    # Query
    perception = await awareness.perceive("How does neural network training work?")

    # Test different context window sizes
    budget_small = ActivationBudget.for_context_window(2000)  # Small (~5 memories)
    activated_small = await awareness.activate(perception, budget=budget_small)

    budget_large = ActivationBudget.for_context_window(32000)  # Large (~50 memories)
    activated_large = await awareness.activate(perception, budget=budget_large)

    print(f"  → Small window (2k tokens): {len(activated_small)} memories")
    print(f"  → Large window (32k tokens): {len(activated_large)} memories")

    assert len(activated_small) <= budget_small.max_memories, "Should respect small budget"
    assert len(activated_large) >= len(activated_small), "Large budget should retrieve more"
    print(f"  ✓ Budget correctly adapts to context window")

    return True


async def test_graph_topology():
    """Test 4: Graph topology builds with typed edges."""
    print("\n[TEST 4] Graph Topology Construction")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Store sequential memories
    memory_ids = []
    for i, text in enumerate([
        "First we define the problem.",
        "Then we collect the data.",
        "Finally we train the model."
    ]):
        p = await awareness.perceive(text)
        mid = await awareness.remember(text, p)
        memory_ids.append(mid)

    # Check temporal edges
    temporal_edges = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data.get('type') == EdgeType.TEMPORAL.value
    ]

    print(f"  → Created {len(memory_ids)} memories")
    print(f"  → Found {len(temporal_edges)} temporal edges")

    assert len(temporal_edges) >= 2, "Should have temporal edges between sequential memories"

    # Check semantic edges
    semantic_edges = [
        (u, v) for u, v, data in graph.edges(data=True)
        if data.get('type') == EdgeType.SEMANTIC_RESONANCE.value
    ]

    print(f"  → Found {len(semantic_edges)} semantic resonance edges")
    print(f"  ✓ Graph topology correctly constructed")

    return True


async def test_awareness_metrics():
    """Test 5: Awareness metrics provide accurate state."""
    print("\n[TEST 5] Awareness Metrics")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Build memory
    for text in [
        "Awareness tracks semantic state.",
        "Metrics summarize system state.",
        "Policy uses metrics for decisions."
    ]:
        p = await awareness.perceive(text)
        await awareness.remember(text, p)

    # Activate
    perception = await awareness.perceive("What are awareness metrics?")
    await awareness.activate(perception, strategy=ActivationStrategy.BALANCED)

    # Get metrics
    metrics = awareness.get_metrics()

    assert metrics.current_position.shape == (64,), "Position should be 64D sample"
    assert metrics.n_memories == 3, "Should track memory count"
    assert metrics.n_connections > 0, "Should have connections"
    assert metrics.n_active > 0, "Should have active memories"
    assert 0 <= metrics.activation_density <= 1, "Density should be normalized"

    print(f"  → Memories: {metrics.n_memories}")
    print(f"  → Connections: {metrics.n_connections}")
    print(f"  → Active: {metrics.n_active}")
    print(f"  → Density: {metrics.activation_density:.2f}")
    print(f"  → Shift magnitude: {metrics.shift_magnitude:.3f}")
    print(f"  ✓ Metrics accurately reflect system state")

    return True


async def run_verification():
    """Run all verification tests."""
    print("=" * 60)
    print("AWARENESS ARCHITECTURE VERIFICATION")
    print("=" * 60)

    tests = [
        ("Core Cycle", test_perceive_remember_activate),
        ("Topic Shift Detection", test_topic_shift_detection),
        ("Context Window Budgeting", test_context_window_budgeting),
        ("Graph Topology", test_graph_topology),
        ("Awareness Metrics", test_awareness_metrics)
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = await test_fn()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready for WeavingOrchestrator integration!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_verification())
    sys.exit(exit_code)
