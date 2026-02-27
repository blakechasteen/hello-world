#!/usr/bin/env python3
"""
Awareness Architecture Verification

Standalone verification that proves:
1. Perceive → Remember → Activate cycle works
2. Topic shift detection works
3. Context window budgeting works
4. Graph topology builds correctly
5. Activation spreads through connections

Run from repo root: python verify_awareness.py
"""

import asyncio
import sys
import networkx as nx
import numpy as np

# Add repo to path
sys.path.insert(0, '.')

from HoloLoom.memory.awareness_graph import AwarenessGraph
from HoloLoom.memory.awareness_types import (
    ActivationStrategy,
    ActivationBudget,
    EdgeType
)
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus


async def test_perceive_remember_activate():
    """Test 1: Basic perceive → remember → activate cycle."""
    print("\n" + "="*80)
    print("TEST 1: Perceive → Remember → Activate Cycle")
    print("="*80)

    # Setup
    graph = nx.MultiDiGraph()
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)
    awareness = AwarenessGraph(graph, semantic, vector_store=None)

    # Test perceive
    print("\n📝 Perceiving text...")
    perception = await awareness.perceive("Python is a programming language")

    print(f"✓ Position shape: {perception.position.shape}")
    print(f"✓ Dominant dimensions: {perception.dominant_dimensions[:3]}")
    print(f"✓ Momentum: {perception.momentum:.3f}")

    assert perception.position.shape == (244,), "Position should be 244D"
    assert len(perception.dominant_dimensions) > 0, "Should have dominant dimensions"

    # Test remember
    print("\n💾 Storing memory...")
    memory_id = await awareness.remember(
        "Python is a programming language",
        perception,
        context={'test': 'example'}
    )

    print(f"✓ Memory stored: {memory_id[:8]}...")
    assert memory_id in awareness.graph.nodes, "Memory should be in graph"
    assert memory_id in awareness.semantic_positions, "Position should be indexed"

    # Add a few more memories
    for text in [
        "Python has decorators for metaprogramming",
        "Functions in Python are first-class objects"
    ]:
        p = await awareness.perceive(text)
        await awareness.remember(text, p)

    # Test activate
    print("\n🎯 Activating relevant memories...")
    query_perception = await awareness.perceive("Tell me about Python programming")

    activated = await awareness.activate(
        query_perception,
        strategy=ActivationStrategy.BALANCED
    )

    print(f"✓ Activated {len(activated)} memories")
    for mem in activated:
        print(f"   - {mem.text[:60]}...")

    assert len(activated) > 0, "Should activate at least one memory"

    print("\n✅ TEST 1 PASSED")
    return True


async def test_topic_shift_detection():
    """Test 2: Topic shift detection with PRECISE strategy."""
    print("\n" + "="*80)
    print("TEST 2: Topic Shift Detection")
    print("="*80)

    # Setup
    graph = nx.MultiDiGraph()
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)
    awareness = AwarenessGraph(graph, semantic, vector_store=None)

    # Build conversation about Python
    python_queries = [
        "How do I define a function in Python?",
        "What are Python decorators?",
        "Explain list comprehensions in Python"
    ]

    print("\n📝 Building Python conversation history...")
    for text in python_queries:
        p = await awareness.perceive(text)
        await awareness.remember(text, p)
        print(f"   ✓ {text}")

    # Query within same topic (should activate memories)
    print("\n🎯 Query within topic: 'Tell me about Python classes'")
    perception_same = await awareness.perceive("Tell me about Python classes")

    activated_same = await awareness.activate(
        perception_same,
        strategy=ActivationStrategy.PRECISE  # High precision
    )

    print(f"   ✓ Activated {len(activated_same)} memories (precise)")

    # Query with topic shift (should activate few/no memories)
    print("\n⚠️  Query with topic shift: 'What's the weather like?'")
    perception_shifted = await awareness.perceive("What's the weather like today?")

    activated_shifted = await awareness.activate(
        perception_shifted,
        strategy=ActivationStrategy.PRECISE
    )

    print(f"   ✓ Activated {len(activated_shifted)} memories (precise)")

    # Topic shift detection logic
    shift_detected = len(activated_shifted) < 2
    print(f"\n   Topic shift detected: {shift_detected}")

    assert len(activated_same) > len(activated_shifted), \
        "Same topic should activate more than shifted topic"

    print("\n✅ TEST 2 PASSED")
    return True


async def test_context_window_budgeting():
    """Test 3: Context window budgeting adapts to constraints."""
    print("\n" + "="*80)
    print("TEST 3: Context Window Budgeting")
    print("="*80)

    # Setup
    graph = nx.MultiDiGraph()
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)
    awareness = AwarenessGraph(graph, semantic, vector_store=None)

    # Add many memories
    print("\n📝 Adding 15 memories about Python...")
    for i in range(15):
        text = f"Python concept {i}: functions, classes, decorators"
        p = await awareness.perceive(text)
        await awareness.remember(text, p)

    print(f"   ✓ Stored {len(awareness.graph.nodes)} memories")

    # Query with different budgets
    query_perception = await awareness.perceive("Explain Python programming")

    # Small context (4k tokens)
    print("\n💰 Small context window (4k tokens):")
    budget_small = ActivationBudget.for_context_window(4000)
    print(f"   Max memories: {budget_small.max_memories}")
    print(f"   Semantic radius: {budget_small.semantic_radius}")

    activated_small = await awareness.activate(
        query_perception,
        budget=budget_small
    )
    print(f"   ✓ Returned {len(activated_small)} memories")

    # Large context (32k tokens)
    print("\n💰 Large context window (32k tokens):")
    budget_large = ActivationBudget.for_context_window(32000)
    print(f"   Max memories: {budget_large.max_memories}")
    print(f"   Semantic radius: {budget_large.semantic_radius}")

    activated_large = await awareness.activate(
        query_perception,
        budget=budget_large
    )
    print(f"   ✓ Returned {len(activated_large)} memories")

    assert len(activated_large) >= len(activated_small), \
        "Large context should retrieve more memories"

    assert len(activated_small) <= budget_small.max_memories, \
        "Should respect small budget limit"

    assert len(activated_large) <= budget_large.max_memories, \
        "Should respect large budget limit"

    print("\n✅ TEST 3 PASSED")
    return True


async def test_graph_topology():
    """Test 4: Graph topology builds with typed edges."""
    print("\n" + "="*80)
    print("TEST 4: Graph Topology")
    print("="*80)

    # Setup
    graph = nx.MultiDiGraph()
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)
    awareness = AwarenessGraph(graph, semantic, vector_store=None)

    # Add memories
    print("\n📝 Adding memories...")
    memory_ids = []
    for text in [
        "Python uses indentation for blocks",
        "Python has dynamic typing",
        "The weather is sunny today"  # Different topic
    ]:
        p = await awareness.perceive(text)
        mid = await awareness.remember(text, p)
        memory_ids.append(mid)
        print(f"   ✓ {text}")

    # Check graph structure
    print("\n🕸️  Analyzing graph topology...")
    print(f"   Nodes: {len(awareness.graph.nodes)}")
    print(f"   Edges: {len(awareness.graph.edges)}")

    # Count edge types
    edge_types = {}
    for u, v, data in awareness.graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"\n   Edge types:")
    for etype, count in edge_types.items():
        print(f"      {etype}: {count}")

    # Test causal edge
    print("\n🔗 Adding causal edge (query → result)...")
    awareness.add_causal_edge(
        memory_ids[0],
        memory_ids[1],
        tool='search'
    )

    # Check causal edge exists
    causal_edges = [
        (u, v, d) for u, v, d in awareness.graph.edges(data=True)
        if d.get('type') == EdgeType.CAUSAL.value
    ]
    print(f"   ✓ Causal edges: {len(causal_edges)}")

    assert len(awareness.graph.nodes) == 3, "Should have 3 nodes"
    assert len(awareness.graph.edges) > 0, "Should have edges"
    assert EdgeType.TEMPORAL.value in edge_types, "Should have temporal edges"
    assert len(causal_edges) > 0, "Should have causal edge"

    print("\n✅ TEST 4 PASSED")
    return True


async def test_awareness_metrics():
    """Test 5: Awareness metrics are accurate."""
    print("\n" + "="*80)
    print("TEST 5: Awareness Metrics")
    print("="*80)

    # Setup
    graph = nx.MultiDiGraph()
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(embedder, snapshot_interval=0.5)
    awareness = AwarenessGraph(graph, semantic, vector_store=None)

    # Add memories and activate
    print("\n📝 Building awareness...")
    for text in [
        "Python programming language",
        "Functions in Python",
        "Python decorators"
    ]:
        p = await awareness.perceive(text)
        await awareness.remember(text, p)

    # Activate to populate activation field
    query_p = await awareness.perceive("Tell me about Python")
    await awareness.activate(query_p, strategy=ActivationStrategy.BALANCED)

    # Get metrics
    print("\n📊 Getting awareness metrics...")
    metrics = awareness.get_metrics()

    print(f"   Current position: {metrics.current_position.shape}")
    print(f"   Shift magnitude: {metrics.shift_magnitude:.3f}")
    print(f"   Shift detected: {metrics.shift_detected}")
    print(f"   Total memories: {metrics.n_memories}")
    print(f"   Active memories: {metrics.n_active}")
    print(f"   Activation density: {metrics.activation_density:.3f}")
    print(f"   Trajectory length: {metrics.trajectory_length}")

    # Test feature vector conversion
    feature_vec = metrics.to_feature_vector()
    print(f"\n   Feature vector shape: {feature_vec.shape}")
    print(f"   ✓ Ready for policy consumption")

    assert metrics.current_position.shape == (64,), "Position sample should be 64D"
    assert metrics.n_memories == 3, "Should have 3 memories"
    assert metrics.n_active > 0, "Should have active memories"
    assert feature_vec.shape[0] == 70, "Feature vector should be 70D"

    print("\n✅ TEST 5 PASSED")
    return True


async def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("🧪 AWARENESS ARCHITECTURE VERIFICATION")
    print("="*80)
    print("\nRunning comprehensive verification suite...")

    tests = [
        ("Perceive → Remember → Activate", test_perceive_remember_activate),
        ("Topic Shift Detection", test_topic_shift_detection),
        ("Context Window Budgeting", test_context_window_budgeting),
        ("Graph Topology", test_graph_topology),
        ("Awareness Metrics", test_awareness_metrics)
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n❌ TEST FAILED: {e}")

    # Summary
    print("\n" + "="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if error:
            print(f"        Error: {error}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Architecture verified!")
        print("\n✅ Ready for Phase 1 integration")
        return True
    else:
        print("\n⚠️  Some tests failed - review errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)