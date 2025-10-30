"""
Awareness Graph Integration Demo

Shows elegant integration of AwarenessGraph with existing HoloLoom architecture.
This demonstrates the design BEFORE we modify WeavingOrchestrator.
"""

import asyncio
import networkx as nx
import numpy as np

from HoloLoom.memory.awareness_graph import AwarenessGraph
from HoloLoom.memory.awareness_types import ActivationStrategy, ActivationBudget
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus


async def demonstrate_awareness_graph():
    """
    Demonstrate living awareness with semantic topology.

    Shows:
    1. Perceive (semantic calculus)
    2. Remember (weave into graph)
    3. Activate (retrieve relevant)
    4. Metrics (what policy reads)
    """

    print("=" * 80)
    print("🧠 AWARENESS GRAPH DEMONSTRATION")
    print("=" * 80)
    print()

    # =========================================================================
    # Setup: Create awareness graph
    # =========================================================================

    print("🔧 Initializing awareness system...")
    print()

    # 1. Graph backend (NetworkX for demo)
    graph = nx.MultiDiGraph()

    # 2. Semantic calculus (perception)
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic_calculus = MatryoshkaSemanticCalculus(
        matryoshka_embedder=embedder,
        snapshot_interval=0.5
    )

    # 3. Awareness graph (composition!)
    awareness = AwarenessGraph(
        graph_backend=graph,
        semantic_calculus=semantic_calculus,
        vector_store=None  # No vector store for demo
    )

    print("✅ Awareness system ready")
    print()

    # =========================================================================
    # Demonstrate: Building awareness through conversation
    # =========================================================================

    conversation = [
        "I want to learn Python programming",
        "How do I define a function in Python?",
        "What are decorators in Python?",
        "Can you explain list comprehensions?",
        "Now let's talk about machine learning instead"  # Topic shift!
    ]

    print("📝 Processing conversation...")
    print()

    for i, text in enumerate(conversation, 1):
        print(f"Query {i}: {text}")
        print("-" * 80)

        # 1. PERCEIVE
        perception = await awareness.perceive(text)

        print(f"   Perception:")
        print(f"      Dominant dimensions: {', '.join(perception.dominant_dimensions[:3])}")
        print(f"      Momentum: {perception.momentum:.3f}")
        print(f"      Complexity: {perception.complexity:.3f}")

        if perception.shift_detected:
            print(f"      ⚠️  TOPIC SHIFT DETECTED (magnitude: {perception.shift_magnitude:.3f})")

        # 2. REMEMBER
        memory_id = await awareness.remember(
            text=text,
            perception=perception,
            context={'query_index': i}
        )

        print(f"   Stored as memory: {memory_id[:8]}...")

        # 3. ACTIVATE (what's relevant?)
        if i > 1:  # Need at least 2 memories
            # Standard retrieval
            context_memories = await awareness.activate(
                perception,
                strategy=ActivationStrategy.BALANCED
            )

            print(f"   Activated {len(context_memories)} relevant memories:")
            for mem in context_memories[:3]:
                print(f"      - {mem.text[:60]}...")

        # 4. METRICS (what policy would see)
        metrics = awareness.get_metrics()
        print(f"   Awareness state:")
        print(f"      Total memories: {metrics.n_memories}")
        print(f"      Active memories: {metrics.n_active}")
        print(f"      Graph resonance: {metrics.avg_resonance:.3f}")

        print()

    # =========================================================================
    # Demonstrate: Precise activation for topic shift detection
    # =========================================================================

    print("=" * 80)
    print("🎯 TOPIC SHIFT DETECTION (Precise Activation)")
    print("=" * 80)
    print()

    # Query within existing topic
    within_topic = "Tell me more about Python syntax"
    print(f"Query (within topic): {within_topic}")

    perception = await awareness.perceive(within_topic)

    # Use PRECISE strategy (high confidence only)
    precise_memories = await awareness.activate(
        perception,
        strategy=ActivationStrategy.PRECISE  # Narrow, high confidence
    )

    print(f"   Activated {len(precise_memories)} memories (precise)")
    print(f"   → Topic continuation detected")
    print()

    # Query with topic shift
    shifted_topic = "What's the weather like today?"
    print(f"Query (shifted topic): {shifted_topic}")

    perception = await awareness.perceive(shifted_topic)

    precise_memories = await awareness.activate(
        perception,
        strategy=ActivationStrategy.PRECISE
    )

    print(f"   Activated {len(precise_memories)} memories (precise)")
    if len(precise_memories) < 2:
        print(f"   → ⚠️ TOPIC SHIFT DETECTED (few strong activations)")
    print()

    # =========================================================================
    # Demonstrate: Context window budgeting
    # =========================================================================

    print("=" * 80)
    print("💰 CONTEXT WINDOW BUDGETING")
    print("=" * 80)
    print()

    # Small context window (2k tokens)
    small_budget = ActivationBudget.for_context_window(2000)
    print(f"Small context (2k tokens):")
    print(f"   Max memories: {small_budget.max_memories}")
    print(f"   Semantic radius: {small_budget.semantic_radius}")
    print(f"   Spread iterations: {small_budget.spread_iterations}")
    print()

    # Large context window (32k tokens)
    large_budget = ActivationBudget.for_context_window(32000)
    print(f"Large context (32k tokens):")
    print(f"   Max memories: {large_budget.max_memories}")
    print(f"   Semantic radius: {large_budget.semantic_radius}")
    print(f"   Spread iterations: {large_budget.spread_iterations}")
    print()

    # =========================================================================
    # Demonstrate: Graph topology
    # =========================================================================

    print("=" * 80)
    print("🕸️  AWARENESS TOPOLOGY")
    print("=" * 80)
    print()

    print(f"Graph statistics:")
    print(f"   Nodes (memories): {len(awareness.graph.nodes)}")
    print(f"   Edges (connections): {len(awareness.graph.edges)}")

    # Count edge types
    edge_types = {}
    for u, v, data in awareness.graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"   Edge types:")
    for edge_type, count in edge_types.items():
        print(f"      {edge_type}: {count}")

    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key insights:")
    print("1. Memory is immutable (content never changes)")
    print("2. Position/topology/activation are indices (recomputable)")
    print("3. One graph, typed edges (TEMPORAL | SEMANTIC | CAUSAL)")
    print("4. Activation as process (spreads dynamically)")
    print("5. Simple policy interface (memories + metrics)")
    print("6. Adaptive retrieval (strategies + budgets)")
    print()


if __name__ == "__main__":
    asyncio.run(demonstrate_awareness_graph())
