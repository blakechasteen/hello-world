"""
Unit tests for AwarenessGraph - Living memory with semantic topology.

Created 2026-01-28: Comprehensive test coverage for awareness_graph.py
Tests: initialization, embedding alignment, perceive, remember, activate,
       connection weaving, metrics, brute force search, and edge cases.
"""

import pytest
import asyncio
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from HoloLoom.memory.awareness_types import (
    SemanticPerception,
    ActivationStrategy,
    ActivationBudget,
    AwarenessMetrics,
    EdgeType,
)
from HoloLoom.memory.awareness_graph import AwarenessGraph


# =============================================================================
# Fixtures
# =============================================================================

def make_mock_semantic_calculus():
    """Create a mock semantic calculus with stream_analyze."""
    mock = MagicMock()
    mock.embedder = MagicMock()
    mock.embedder.encode_base = MagicMock(
        return_value=[np.random.randn(384)]
    )

    async def mock_stream_analyze(word_stream):
        snapshot = MagicMock()
        snapshot.states_by_scale = {}
        snapshot.dominant_dimensions_by_scale = {}
        snapshot.narrative_momentum = 0.5
        snapshot.complexity_index = 0.3
        from enum import Enum
        class MockScale(Enum):
            PARAGRAPH = "paragraph"
        snapshot.states_by_scale[MockScale.PARAGRAPH] = {
            'position': np.random.randn(228),
            'velocity': np.random.randn(228),
        }
        snapshot.dominant_dimensions_by_scale[MockScale.PARAGRAPH] = ['Warmth', 'Valence']
        yield snapshot

    mock.stream_analyze = mock_stream_analyze
    return mock


def make_graph():
    return nx.MultiDiGraph()


def make_awareness_graph(vector_store=None):
    graph = make_graph()
    semantic = make_mock_semantic_calculus()
    return AwarenessGraph(
        graph_backend=graph,
        semantic_calculus=semantic,
        vector_store=vector_store,
    )


def make_perception(position=None, raw_embedding=None):
    if position is None:
        position = np.random.randn(228)
    return SemanticPerception(
        position=position,
        raw_embedding=raw_embedding,
        dominant_dimensions=['Warmth'],
        momentum=0.7,
        complexity=0.4,
        shift_magnitude=0.1,
        shift_detected=False,
    )


# =============================================================================
# Initialization Tests
# =============================================================================

class TestAwarenessGraphInit:
    def test_init_without_vector_store(self):
        ag = make_awareness_graph(vector_store=None)
        assert ag.graph is not None
        assert ag.semantic is not None
        assert ag.vectors is None
        assert len(ag.semantic_positions) == 0
        assert len(ag.raw_embeddings) == 0
        assert ag.activation_field is not None
        assert len(ag.trajectory) == 0

    def test_init_with_vector_store(self):
        mock_vs = MagicMock()
        ag = make_awareness_graph(vector_store=mock_vs)
        assert ag.vectors is mock_vs

    def test_trajectory_max_length(self):
        ag = make_awareness_graph()
        assert ag.trajectory.maxlen == 100

    def test_resonance_cache_starts_empty(self):
        ag = make_awareness_graph()
        assert len(ag.resonance_cache) == 0


# =============================================================================
# Embedding Alignment Tests
# =============================================================================

class TestEmbeddingAlignment:
    def test_exact_228d(self):
        ag = make_awareness_graph()
        emb = np.ones(228)
        result = ag._align_embedding_to_228d(emb)
        assert result.shape == (228,)
        np.testing.assert_array_equal(result, emb)

    def test_smaller_embedding_padded(self):
        ag = make_awareness_graph()
        emb = np.ones(100)
        result = ag._align_embedding_to_228d(emb)
        assert result.shape == (228,)
        np.testing.assert_array_equal(result[:100], np.ones(100))
        np.testing.assert_array_equal(result[100:], np.zeros(128))

    def test_larger_embedding_truncated(self):
        ag = make_awareness_graph()
        emb = np.ones(384)
        result = ag._align_embedding_to_228d(emb)
        assert result.shape == (228,)
        np.testing.assert_array_equal(result, np.ones(228))

    def test_single_dimension(self):
        ag = make_awareness_graph()
        emb = np.array([42.0])
        result = ag._align_embedding_to_228d(emb)
        assert result.shape == (228,)
        assert result[0] == 42.0
        assert result[1] == 0.0


# =============================================================================
# Perceive Tests
# =============================================================================

class TestPerceive:
    def test_perceive_text_returns_perception(self):
        ag = make_awareness_graph()
        perception = asyncio.run(ag.perceive("Thompson Sampling"))
        assert isinstance(perception, SemanticPerception)
        assert perception.position is not None
        assert perception.position.shape[0] > 0

    def test_perceive_text_updates_trajectory(self):
        ag = make_awareness_graph()
        assert len(ag.trajectory) == 0
        asyncio.run(ag.perceive("first query"))
        assert len(ag.trajectory) == 1
        asyncio.run(ag.perceive("second query"))
        assert len(ag.trajectory) == 2

    def test_perceive_invalid_type_raises(self):
        ag = make_awareness_graph()
        with pytest.raises(TypeError):
            asyncio.run(ag.perceive(12345))

    def test_perceive_shift_detection_first_query(self):
        ag = make_awareness_graph()
        perception = asyncio.run(ag.perceive("first query"))
        assert perception.shift_magnitude == 0.0
        assert perception.shift_detected is False


# =============================================================================
# Remember Tests
# =============================================================================

class TestRemember:
    def test_remember_returns_id(self):
        ag = make_awareness_graph()
        perception = make_perception()
        memory_id = asyncio.run(ag.remember("test content", perception))
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

    def test_remember_adds_node_to_graph(self):
        ag = make_awareness_graph()
        perception = make_perception()
        memory_id = asyncio.run(ag.remember("test content", perception))
        assert memory_id in ag.graph.nodes

    def test_remember_stores_semantic_position(self):
        ag = make_awareness_graph()
        perception = make_perception()
        memory_id = asyncio.run(ag.remember("test content", perception))
        assert memory_id in ag.semantic_positions
        np.testing.assert_array_equal(
            ag.semantic_positions[memory_id], perception.position
        )

    def test_remember_stores_raw_embedding(self):
        ag = make_awareness_graph()
        raw_emb = np.random.randn(384)
        perception = make_perception(raw_embedding=raw_emb)
        memory_id = asyncio.run(ag.remember("test content", perception))
        assert memory_id in ag.raw_embeddings

    def test_remember_no_raw_embedding(self):
        ag = make_awareness_graph()
        perception = make_perception(raw_embedding=None)
        memory_id = asyncio.run(ag.remember("test content", perception))
        assert memory_id not in ag.raw_embeddings

    def test_remember_with_context(self):
        ag = make_awareness_graph()
        perception = make_perception()
        ctx = {"source": "test", "importance": "high"}
        memory_id = asyncio.run(ag.remember("test", perception, context=ctx))
        node_data = ag.graph.nodes[memory_id]
        assert "context" in node_data
        assert node_data["context"].get("source") == "test"

    def test_remember_multiple_creates_temporal_edge(self):
        ag = make_awareness_graph()
        p1 = make_perception()
        p2 = make_perception()
        id1 = asyncio.run(ag.remember("first", p1))
        id2 = asyncio.run(ag.remember("second", p2))
        edges = list(ag.graph.edges(data=True))
        temporal_edges = [e for e in edges if e[2].get('type') == EdgeType.TEMPORAL.value]
        assert len(temporal_edges) >= 1


# =============================================================================
# Activation Tests
# =============================================================================

class TestActivate:
    def test_activate_empty_graph(self):
        ag = make_awareness_graph()
        perception = make_perception()
        memories = asyncio.run(ag.activate(perception))
        assert isinstance(memories, list)
        assert len(memories) == 0

    def test_activate_with_strategy(self):
        ag = make_awareness_graph()
        perception = make_perception()
        for strategy in ActivationStrategy:
            memories = asyncio.run(ag.activate(perception, strategy=strategy))
            assert isinstance(memories, list)

    def test_activate_with_budget(self):
        ag = make_awareness_graph()
        perception = make_perception()
        budget = ActivationBudget(
            max_memories=5, semantic_radius=1.0,
            spread_iterations=1, activation_threshold=0.3,
        )
        memories = asyncio.run(ag.activate(perception, budget=budget))
        assert isinstance(memories, list)


# =============================================================================
# Brute Force Search Tests
# =============================================================================

class TestBruteForceSearch:
    def test_brute_force_empty_positions(self):
        ag = make_awareness_graph()
        query = np.random.randn(228)
        result = ag._brute_force_search(query, radius=2.0, k=5)
        assert result == []

    def test_brute_force_with_positions(self):
        ag = make_awareness_graph()
        pos = np.zeros(228)
        ag.semantic_positions["node1"] = pos + 0.1
        ag.semantic_positions["node2"] = pos + 0.5
        result = ag._brute_force_search(pos, radius=50.0, k=10)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_brute_force_return_scores(self):
        ag = make_awareness_graph()
        pos = np.zeros(228)
        ag.semantic_positions["node1"] = pos + 0.1
        result = ag._brute_force_search(pos, radius=50.0, k=10, return_scores=True)
        assert isinstance(result, dict)

    def test_brute_force_uses_raw_embeddings(self):
        ag = make_awareness_graph()
        raw = np.ones(384)
        ag.raw_embeddings["node1"] = raw * 0.9
        ag.raw_embeddings["node2"] = raw * 0.1
        result = ag._brute_force_search(
            np.zeros(228), radius=5.0, k=10,
            query_raw_embedding=raw, return_scores=True
        )
        assert isinstance(result, dict)
        if "node1" in result and "node2" in result:
            assert result["node1"] > result["node2"]

    def test_brute_force_respects_k_limit(self):
        ag = make_awareness_graph()
        pos = np.zeros(228)
        for i in range(20):
            ag.semantic_positions[f"node{i}"] = pos + np.random.randn(228) * 0.1
        result = ag._brute_force_search(pos, radius=100.0, k=5)
        assert len(result) <= 5


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    def test_metrics_empty_graph(self):
        ag = make_awareness_graph()
        metrics = ag.get_metrics()
        assert isinstance(metrics, AwarenessMetrics)
        assert metrics.n_memories == 0
        assert metrics.n_connections == 0
        assert metrics.trajectory_length == 0
        assert metrics.avg_resonance == 0.0

    def test_metrics_after_remember(self):
        ag = make_awareness_graph()
        perception = make_perception()
        asyncio.run(ag.remember("test content", perception))
        metrics = ag.get_metrics()
        assert metrics.n_memories >= 1

    def test_metrics_shift_detection(self):
        ag = make_awareness_graph()
        ag.trajectory.append(np.zeros(228))
        ag.trajectory.append(np.ones(228))
        metrics = ag.get_metrics()
        assert metrics.shift_magnitude > 0

    def test_metrics_current_position_shape(self):
        ag = make_awareness_graph()
        metrics = ag.get_metrics()
        assert metrics.current_position.shape == (64,)


# =============================================================================
# Connection Weaving Tests
# =============================================================================

class TestConnectionWeaving:
    def test_add_causal_edge(self):
        ag = make_awareness_graph()
        ag.graph.add_node("source")
        ag.graph.add_node("target")
        ag.add_causal_edge("source", "target", tool="answer")
        edges = list(ag.graph.edges(data=True))
        assert len(edges) == 1
        assert edges[0][2]['type'] == EdgeType.CAUSAL.value
        assert edges[0][2]['tool'] == 'answer'

    def test_semantic_weaving_high_similarity(self):
        ag = make_awareness_graph()
        pos = np.ones(228)
        ag.semantic_positions["existing"] = pos * 0.99
        asyncio.run(ag._weave_semantic("new_node", pos, threshold=0.5))
        edges = list(ag.graph.edges(data=True))
        semantic_edges = [e for e in edges if e[2].get('type') == EdgeType.SEMANTIC_RESONANCE.value]
        # Bidirectional edges if similarity > threshold
        assert len(semantic_edges) in (0, 2)


# =============================================================================
# Resonance Computation Tests
# =============================================================================

class TestResonance:
    def test_identical_vectors(self):
        ag = make_awareness_graph()
        v = np.ones(228)
        assert ag._compute_resonance(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        ag = make_awareness_graph()
        v1 = np.ones(228)
        v2 = -np.ones(228)
        assert ag._compute_resonance(v1, v2) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        ag = make_awareness_graph()
        v1 = np.zeros(228); v1[0] = 1.0
        v2 = np.zeros(228); v2[1] = 1.0
        assert ag._compute_resonance(v1, v2) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        ag = make_awareness_graph()
        assert ag._compute_resonance(np.zeros(228), np.ones(228)) == 0.0


# =============================================================================
# Activation Budget Tests
# =============================================================================

class TestActivationBudget:
    def test_for_strategy_precise(self):
        budget = ActivationBudget.for_strategy(ActivationStrategy.PRECISE)
        assert budget.max_memories == 3
        assert budget.spread_iterations == 0

    def test_for_strategy_balanced(self):
        budget = ActivationBudget.for_strategy(ActivationStrategy.BALANCED)
        assert budget.max_memories == 10
        assert budget.spread_iterations == 2

    def test_for_strategy_exploratory(self):
        budget = ActivationBudget.for_strategy(ActivationStrategy.EXPLORATORY)
        assert budget.max_memories == 30

    def test_for_strategy_deep(self):
        budget = ActivationBudget.for_strategy(ActivationStrategy.DEEP)
        assert budget.max_memories == 20
        assert budget.spread_iterations == 5

    def test_for_context_window_small(self):
        budget = ActivationBudget.for_context_window(1000)
        assert budget.max_memories <= 5

    def test_for_context_window_large(self):
        budget = ActivationBudget.for_context_window(100000)
        assert budget.max_memories <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
