"""
Test multimodal awareness integration.

Verifies that awareness architecture works with pre-computed embeddings
from text, structured data, and multimodal fusion.

Falls forward naturally: ProcessedInput → SemanticPerception → Memory
"""

import sys
sys.path.insert(0, '.')

import asyncio
import numpy as np
import networkx as nx

from hololoom.memory.awareness_graph import AwarenessGraph
from hololoom.memory.awareness_types import ActivationStrategy
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus

from hololoom.input.protocol import ModalityType, ProcessedInput, TextFeatures, StructuredFeatures
from hololoom.input.text_processor import TextProcessor
from hololoom.input.structured_processor import StructuredDataProcessor
from hololoom.input.fusion import MultiModalFusion


async def test_text_to_awareness():
    """Test 1: Text input → awareness (baseline)."""
    print("\n[TEST 1] Text Input → Awareness (baseline)")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Perceive text (streaming semantic calculus)
    text = "Thompson Sampling balances exploration and exploitation"
    perception = await awareness.perceive(text)

    assert perception.position.shape == (228,), "Position should be 228D"
    print(f"  ✓ Perceived text: 228D position")

    # Remember
    memory_id = await awareness.remember(text, perception)
    assert memory_id in graph.nodes, "Memory should be stored"
    print(f"  ✓ Remembered: {memory_id[:8]}...")

    # Activate
    query_perception = await awareness.perceive("What is Thompson Sampling?")
    activated = await awareness.activate(query_perception, strategy=ActivationStrategy.BALANCED)

    assert len(activated) > 0, "Should activate memories"
    print(f"  ✓ Activated: {len(activated)} memories")
    print("✓ PASS: Text → Awareness")

    return True


async def test_structured_to_awareness():
    """Test 2: Structured data → awareness (multimodal)."""
    print("\n[TEST 2] Structured Data → Awareness (multimodal)")

    # Setup awareness
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Process structured data (pre-computed embedding)
    processor = StructuredDataProcessor()
    data = {
        "algorithm": "Thompson Sampling",
        "category": "reinforcement_learning",
        "metrics": {"exploration_rate": 0.3, "exploitation_rate": 0.7}
    }

    processed = await processor.process(data)

    assert processed.modality == ModalityType.STRUCTURED, "Should be structured"
    assert processed.embedding is not None, "Should have embedding"
    print(f"  ✓ Processed structured data: {processed.embedding.shape[0]}D embedding")

    # Perceive from ProcessedInput (multimodal bridge!)
    perception = await awareness.perceive(processed)

    assert perception.position.shape == (228,), "Position should be 228D"
    print(f"  ✓ Perceived structured data: 228D position (aligned from {processed.embedding.shape[0]}D)")

    # Remember with modality metadata
    memory_id = await awareness.remember(processed, perception)
    assert memory_id in graph.nodes, "Memory should be stored"

    # Check modality metadata preserved
    node_data = graph.nodes[memory_id]
    assert node_data['context']['modality'] == 'structured', "Modality should be preserved"
    print(f"  ✓ Remembered with modality: {node_data['context']['modality']}")

    print("✓ PASS: Structured Data → Awareness")

    return True


async def test_multimodal_fusion_to_awareness():
    """Test 3: Fused multimodal input → awareness."""
    print("\n[TEST 3] Multimodal Fusion → Awareness")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Create text input
    text_processor = TextProcessor()
    text_input = await text_processor.process(
        "Thompson Sampling is used in reinforcement learning for exploration-exploitation tradeoff"
    )

    # Create structured input
    struct_processor = StructuredDataProcessor()
    struct_input = await struct_processor.process({
        "algorithm": "Thompson Sampling",
        "applications": ["A/B testing", "multi-armed bandits", "RL"],
        "complexity": "O(k)"
    })

    # Fuse multimodal inputs
    fusion = MultiModalFusion()
    fused = await fusion.fuse([text_input, struct_input], strategy="attention", target_dim=512)

    assert fused.modality == ModalityType.MULTIMODAL, "Should be multimodal"
    assert fused.embedding is not None, "Should have fused embedding"
    print(f"  ✓ Fused text + structured: {fused.embedding.shape[0]}D embedding")

    # Perceive fused input
    perception = await awareness.perceive(fused)

    assert perception.position.shape == (228,), "Position should be 228D"
    print(f"  ✓ Perceived multimodal: 228D position (aligned from {fused.embedding.shape[0]}D)")

    # Remember
    memory_id = await awareness.remember(fused, perception)
    node_data = graph.nodes[memory_id]
    assert node_data['context']['modality'] == 'multimodal', "Should be multimodal"
    print(f"  ✓ Remembered multimodal memory: {memory_id[:8]}...")

    print("✓ PASS: Multimodal Fusion → Awareness")

    return True


async def test_cross_modal_activation():
    """Test 4: Cross-modal activation (text query → structured memory)."""
    print("\n[TEST 4] Cross-Modal Activation")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Store structured memories
    processor = StructuredDataProcessor()

    data_items = [
        {"algorithm": "Thompson Sampling", "type": "bayesian"},
        {"algorithm": "UCB", "type": "optimistic"},
        {"algorithm": "Epsilon-Greedy", "type": "simple"}
    ]

    for data in data_items:
        processed = await processor.process(data)
        perception = await awareness.perceive(processed)
        await awareness.remember(processed, perception)

    print(f"  ✓ Stored {len(data_items)} structured memories")

    # Query with text (cross-modal!)
    text_query = "Tell me about bayesian algorithms"
    query_perception = await awareness.perceive(text_query)

    # Activate (should find structured memories)
    activated = await awareness.activate(query_perception, strategy=ActivationStrategy.BALANCED)

    assert len(activated) > 0, "Should activate structured memories from text query"
    print(f"  ✓ Text query activated {len(activated)} structured memories")

    # Check modality mix
    modalities = [awareness.graph.nodes[mem.id]['context'].get('modality', 'text')
                  for mem in activated]
    print(f"  ✓ Activated modalities: {set(modalities)}")

    print("✓ PASS: Cross-Modal Activation")

    return True


async def test_dimension_alignment():
    """Test 5: Embedding dimension alignment (various sizes → 228D)."""
    print("\n[TEST 5] Dimension Alignment")

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    semantic = MatryoshkaSemanticCalculus(matryoshka_embedder=embedder)
    graph = nx.MultiDiGraph()
    awareness = AwarenessGraph(graph_backend=graph, semantic_calculus=semantic)

    # Test various embedding sizes
    test_sizes = [96, 192, 228, 384, 512, 768]

    for size in test_sizes:
        # Create mock ProcessedInput with specific embedding size
        mock_embedding = np.random.randn(size).astype(np.float32)
        mock_input = ProcessedInput(
            modality=ModalityType.TEXT,
            content=f"test {size}D",
            embedding=mock_embedding,
            confidence=1.0,
            features=TextFeatures(entities=[], sentiment={}, topics=[], keyphrases=[])
        )

        # Perceive (should align to 228D)
        perception = await awareness.perceive(mock_input)

        assert perception.position.shape == (228,), f"Should align {size}D → 228D"
        print(f"  ✓ Aligned {size}D → 228D")

    print("✓ PASS: Dimension Alignment")

    return True


async def run_all_tests():
    """Run all multimodal awareness tests."""
    print("=" * 70)
    print("MULTIMODAL AWARENESS INTEGRATION TESTS")
    print("=" * 70)

    tests = [
        ("Text → Awareness", test_text_to_awareness),
        ("Structured Data → Awareness", test_structured_to_awareness),
        ("Multimodal Fusion → Awareness", test_multimodal_fusion_to_awareness),
        ("Cross-Modal Activation", test_cross_modal_activation),
        ("Dimension Alignment", test_dimension_alignment),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAIL: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    else:
        print("🎉 ALL MULTIMODAL AWARENESS TESTS PASSED!")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
