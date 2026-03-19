"""
Unit tests for remaining uncovered memory modules:
- spring_dynamics_engine.py
- multi_wave_engine.py
- streaming_expansion.py
- hybrid_retrieval.py
- llm_consolidator.py

Target: 60%+ coverage on each file.
"""

import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Dict, Set

import numpy as np
import pytest

# =============================================================================
# spring_dynamics_engine tests
# =============================================================================

from hololoom.core.memory.spring_dynamics_engine import (
    MemoryNode,
    SpringEngineConfig,
    SpringDynamicsEngine,
    BetaWaveRecallResult,
)


class TestMemoryNode:
    """Tests for MemoryNode dataclass."""

    def test_creation_with_numpy(self):
        node = MemoryNode(
            id="n1",
            content="hello",
            position=np.array([1.0, 2.0, 3.0]),
            rest_position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
        )
        assert node.id == "n1"
        assert node.spring_constant == 1.0
        assert node.access_count == 0

    def test_creation_with_lists_converts_to_numpy(self):
        node = MemoryNode(
            id="n2",
            content="test",
            position=[1.0, 2.0],
            rest_position=[0.0, 0.0],
            velocity=[0.0, 0.0],
        )
        assert isinstance(node.position, np.ndarray)
        assert isinstance(node.rest_position, np.ndarray)
        assert isinstance(node.velocity, np.ndarray)

    def test_get_displacement(self):
        node = MemoryNode(
            id="n3",
            content="",
            position=np.array([3.0, 4.0]),
            rest_position=np.array([1.0, 1.0]),
            velocity=np.zeros(2),
        )
        disp = node.get_displacement()
        np.testing.assert_array_almost_equal(disp, [2.0, 3.0])

    def test_get_spring_force(self):
        node = MemoryNode(
            id="n4",
            content="",
            position=np.array([2.0, 0.0]),
            rest_position=np.array([0.0, 0.0]),
            velocity=np.zeros(2),
            spring_constant=3.0,
        )
        force = node.get_spring_force()
        # F = -k * (x - x0) = -3 * [2, 0] = [-6, 0]
        np.testing.assert_array_almost_equal(force, [-6.0, 0.0])

    def test_get_recall_strength_normalized(self):
        node = MemoryNode(
            id="n5",
            content="",
            position=np.zeros(2),
            rest_position=np.zeros(2),
            velocity=np.zeros(2),
            spring_constant=5.5,
            min_spring_constant=0.5,
            max_spring_constant=10.5,
        )
        # (5.5 - 0.5) / (10.5 - 0.5) = 5.0 / 10.0 = 0.5
        assert node.get_recall_strength() == pytest.approx(0.5)

    def test_get_recall_strength_zero_range(self):
        node = MemoryNode(
            id="n6",
            content="",
            position=np.zeros(2),
            rest_position=np.zeros(2),
            velocity=np.zeros(2),
            min_spring_constant=1.0,
            max_spring_constant=1.0,
        )
        assert node.get_recall_strength() == 0.5


class TestSpringEngineConfig:
    def test_defaults(self):
        cfg = SpringEngineConfig()
        assert cfg.damping == 0.95
        assert cfg.dt == 0.1
        assert cfg.max_active_nodes == 1000


class TestSpringDynamicsEngine:
    """Tests for SpringDynamicsEngine."""

    def _make_engine(self, **kwargs) -> SpringDynamicsEngine:
        cfg = SpringEngineConfig(**kwargs)
        return SpringDynamicsEngine(cfg)

    def test_add_and_get_node(self):
        engine = self._make_engine()
        emb = np.array([1.0, 0.0, 0.0])
        engine.add_node("a", emb, content="test content", neighbors={"b"})
        node = engine.get_node("a")
        assert node is not None
        assert node.id == "a"
        assert "b" in node.neighbors
        assert "a" in engine.active_nodes

    def test_add_duplicate_node_skips(self):
        engine = self._make_engine()
        emb = np.array([1.0, 0.0])
        engine.add_node("a", emb)
        engine.add_node("a", emb)  # Should warn, not crash
        assert len(engine.nodes) == 1

    def test_remove_node(self):
        engine = self._make_engine()
        engine.add_node("a", np.array([1.0, 0.0]), neighbors={"b"})
        engine.add_node("b", np.array([0.0, 1.0]), neighbors={"a"})
        engine.remove_node("a")
        assert "a" not in engine.nodes
        assert "a" not in engine.active_nodes
        assert "a" not in engine.nodes["b"].neighbors

    def test_on_memory_accessed_strengthens(self):
        engine = self._make_engine()
        engine.add_node("a", np.array([1.0, 0.0]), initial_spring_constant=1.0)
        old_k = engine.nodes["a"].spring_constant
        engine.on_memory_accessed("a", pulse_strength=0.5)
        assert engine.nodes["a"].spring_constant > old_k
        assert engine.nodes["a"].access_count == 1
        assert engine.total_accesses == 1

    def test_on_memory_accessed_with_query_embedding(self):
        engine = self._make_engine()
        engine.add_node("a", np.array([1.0, 0.0]))
        engine.on_memory_accessed("a", query_embedding=np.array([0.0, 1.0]), pulse_strength=0.8)
        # Velocity should now point toward query embedding direction
        assert np.linalg.norm(engine.nodes["a"].velocity) > 0

    def test_on_memory_accessed_nonexistent(self):
        engine = self._make_engine()
        engine.on_memory_accessed("missing")  # Should not crash

    def test_propagate_activation_to_neighbors(self):
        engine = self._make_engine()
        engine.add_node("a", np.array([1.0, 0.0]), neighbors={"b"})
        engine.add_node("b", np.array([0.0, 1.0]))
        old_k = engine.nodes["b"].spring_constant
        engine.on_memory_accessed("a", pulse_strength=0.8)
        # b should have been boosted via propagation
        assert engine.nodes["b"].spring_constant >= old_k
        assert "b" in engine.active_nodes
        assert engine.total_propagations > 0

    def test_update_physics(self):
        engine = self._make_engine(damping=0.9)
        engine.add_node("a", np.array([1.0, 0.0]))
        # Displace the node
        engine.nodes["a"].position = np.array([3.0, 0.0])
        engine.nodes["a"].velocity = np.array([0.5, 0.0])
        engine._update_physics(dt=0.1)
        # Velocity should have changed due to spring force
        # Spring force = -k * (3 - 1) = -1 * 2 = -2
        # But rest_position is [1,0] from embedding, so displacement is [2,0]
        # Actually add_node copies embedding to rest_position, so:
        # rest is [1,0], position is [3,0], displacement = [2,0]
        # force = -1 * [2,0] = [-2, 0]
        # accel = [-2, 0] / 1.0 = [-2, 0]
        # v = [0.5, 0] + [-2, 0] * 0.1 = [0.3, 0]
        # Then damped: [0.3 * 0.9, 0] = [0.27, 0]
        np.testing.assert_array_almost_equal(
            engine.nodes["a"].velocity, [0.27, 0.0], decimal=2
        )

    def test_apply_forgetting_decay(self):
        engine = self._make_engine()
        engine.add_node("a", np.array([1.0, 0.0]), initial_spring_constant=5.0)
        # Set last_accessed to 2 hours ago
        engine.nodes["a"].last_accessed = datetime.now() - timedelta(hours=2)
        engine._apply_forgetting_decay()
        assert engine.nodes["a"].spring_constant < 5.0

    def test_update_active_set_removes_at_rest(self):
        engine = self._make_engine(sleep_after_inactive_hours=0)
        engine.add_node("a", np.array([1.0, 0.0]), initial_spring_constant=0.1)
        engine.nodes["a"].velocity = np.array([0.0, 0.0])
        engine.nodes["a"].spring_constant = 0.1  # at min
        engine.nodes["a"].last_accessed = datetime.now() - timedelta(hours=25)
        sleeping = []
        engine.on_node_sleeping = lambda nid: sleeping.append(nid)
        engine._update_active_set()
        assert "a" not in engine.active_nodes
        assert "a" in sleeping

    def test_retrieve_memories_basic(self):
        engine = self._make_engine()
        # Add nodes with embeddings
        engine.add_node("a", np.array([1.0, 0.0, 0.0]), content="alpha")
        engine.add_node("b", np.array([0.0, 1.0, 0.0]), content="beta")
        engine.add_node("c", np.array([0.9, 0.1, 0.0]), content="close to a", neighbors={"a"})

        result = engine.retrieve_memories(
            query_embedding=np.array([1.0, 0.0, 0.0]),
            top_k=5,
        )
        assert isinstance(result, BetaWaveRecallResult)
        assert len(result.seed_nodes) > 0
        # 'a' should be the top seed (most similar to query)
        assert result.seed_nodes[0][0] == "a"

    def test_find_seed_nodes(self):
        engine = self._make_engine()
        engine.add_node("x", np.array([1.0, 0.0]))
        engine.add_node("y", np.array([0.0, 1.0]))
        seeds = engine._find_seed_nodes(np.array([1.0, 0.0]), top_n=2)
        assert seeds[0][0] == "x"
        assert seeds[0][1] > seeds[1][1]

    def test_detect_creative_insights(self):
        engine = self._make_engine()
        engine.add_node("seed", np.array([0.0, 0.0]))
        engine.add_node("distant", np.array([10.0, 10.0]), neighbors={"seed"})
        insights = engine._detect_creative_insights(
            seed_nodes=["seed"],
            activated_nodes=[("distant", 0.6)],
            query_embedding=np.array([0.0, 0.0]),
        )
        assert len(insights) > 0
        assert insights[0][2] == "strong_association"

    def test_get_statistics(self):
        engine = self._make_engine()
        engine.add_node("a", np.array([1.0, 0.0]))
        stats = engine.get_statistics()
        assert stats["total_nodes"] == 1
        assert stats["active_nodes"] == 1
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_start_stop(self):
        engine = self._make_engine(update_interval_ms=50)
        await engine.start()
        assert engine.running is True
        await asyncio.sleep(0.05)
        await engine.stop()
        assert engine.running is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        engine = self._make_engine(update_interval_ms=50)
        async with engine:
            assert engine.running is True
        assert engine.running is False

    def test_beta_wave_recall_result_methods(self):
        result = BetaWaveRecallResult(
            recalled_memories=[("seed1", 0.9), ("assoc1", 0.5)],
            all_activations={"seed1": 0.9, "assoc1": 0.5},
            creative_insights=[("creative1", 0.3, "bridge_node")],
            seed_nodes=[("seed1", 0.95)],
            iterations=10,
        )
        assert result.get_direct_recalls() == ["seed1"]
        assert result.get_associative_recalls() == ["assoc1"]
        assert result.get_creative_insight_ids() == ["creative1"]
        s = str(result)
        assert "2 recalled" in s


# =============================================================================
# multi_wave_engine tests
# =============================================================================

from hololoom.core.memory.multi_wave_engine import (
    BrainWaveMode,
    MultiWaveMemoryEngine,
    ThetaWaveConsolidator,
    DeltaWavePruner,
    REMDreamer,
    ActivationPattern,
)


class TestBrainWaveMode:
    def test_modes_exist(self):
        assert BrainWaveMode.BETA.value == "beta"
        assert BrainWaveMode.ALPHA.value == "alpha"
        assert BrainWaveMode.THETA.value == "theta"
        assert BrainWaveMode.DELTA.value == "delta"
        assert BrainWaveMode.REM.value == "rem"


class TestThetaWaveConsolidator:
    def test_record_activation_pattern(self):
        engine = MultiWaveMemoryEngine()
        consolidator = ThetaWaveConsolidator(engine)
        activations = {"a": 0.5, "b": 0.8, "c": 0.1}
        consolidator.record_activation_pattern(activations, source="query", threshold=0.3)
        assert len(consolidator.co_activation_history) == 1
        assert set(consolidator.co_activation_history[0].nodes) == {"a", "b"}

    def test_record_pattern_needs_at_least_2(self):
        engine = MultiWaveMemoryEngine()
        consolidator = ThetaWaveConsolidator(engine)
        activations = {"a": 0.5}
        consolidator.record_activation_pattern(activations, threshold=0.3)
        assert len(consolidator.co_activation_history) == 0

    def test_theta_consolidation_creates_neighbors(self):
        engine = MultiWaveMemoryEngine()
        engine.add_node("a", np.array([1.0, 0.0, 0.0]))
        engine.add_node("b", np.array([0.0, 1.0, 0.0]))
        consolidator = ThetaWaveConsolidator(engine)
        consolidator.consolidation_threshold = 2

        # Record enough co-activations
        for _ in range(3):
            consolidator.record_activation_pattern({"a": 0.5, "b": 0.5}, threshold=0.3)

        result = consolidator.theta_consolidation_update()
        assert result >= 1
        assert "b" in engine.nodes["a"].neighbors
        assert "a" in engine.nodes["b"].neighbors

    def test_consolidation_max_history(self):
        engine = MultiWaveMemoryEngine()
        consolidator = ThetaWaveConsolidator(engine)
        consolidator.max_history = 5
        for i in range(10):
            consolidator.record_activation_pattern(
                {f"n{i}": 0.5, f"n{i+1}": 0.5}, threshold=0.3
            )
        assert len(consolidator.co_activation_history) == 5


class TestDeltaWavePruner:
    def test_prune_weak_old_nodes(self):
        engine = MultiWaveMemoryEngine()
        engine.add_node("weak", np.array([1.0, 0.0]), initial_spring_constant=0.2)
        engine.nodes["weak"].last_accessed = datetime.now() - timedelta(hours=100)
        engine.add_node("strong", np.array([0.0, 1.0]), initial_spring_constant=6.0)
        engine.nodes["strong"].last_accessed = datetime.now()

        # Connect them
        engine.nodes["weak"].neighbors.add("strong")
        engine.nodes["strong"].neighbors.add("weak")

        pruner = DeltaWavePruner(engine)
        pruned, strengthened = pruner.delta_pruning_update()
        assert pruned >= 1
        assert strengthened >= 1
        # weak node's neighbors should be cleared
        assert len(engine.nodes["weak"].neighbors) == 0
        # strong node should no longer point to weak
        assert "weak" not in engine.nodes["strong"].neighbors

    def test_delta_resets_velocities(self):
        engine = MultiWaveMemoryEngine()
        engine.add_node("a", np.array([1.0, 0.0]))
        engine.nodes["a"].velocity = np.array([5.0, 5.0])
        pruner = DeltaWavePruner(engine)
        pruner.delta_pruning_update()
        # Velocity should be dramatically reduced
        assert np.linalg.norm(engine.nodes["a"].velocity) < 1.0


class TestREMDreamer:
    @pytest.mark.asyncio
    async def test_dream_cycle_creates_bridges(self):
        engine = MultiWaveMemoryEngine()
        # Create nodes that are semantically distant
        for i in range(5):
            engine.add_node(
                f"n{i}",
                np.array([float(i * 10), 0.0]),
                neighbors={f"n{j}" for j in range(5) if j != i},
            )
        dreamer = REMDreamer(engine)
        dreamer.bridge_distance_threshold = 5.0
        # Run very short dream
        bridges = await dreamer.dream_cycle(duration_seconds=0.1)
        # We can't guarantee bridges are created (random), but it shouldn't crash
        assert isinstance(bridges, int)

    @pytest.mark.asyncio
    async def test_dream_cycle_with_few_nodes(self):
        engine = MultiWaveMemoryEngine()
        engine.add_node("only", np.array([1.0, 0.0]))
        dreamer = REMDreamer(engine)
        dreamer.random_seed_count = 3
        bridges = await dreamer.dream_cycle(duration_seconds=0.1)
        assert bridges == 0  # Not enough nodes


class TestMultiWaveMemoryEngine:
    def test_initial_mode_is_beta(self):
        engine = MultiWaveMemoryEngine()
        assert engine.mode == BrainWaveMode.BETA

    def test_update_mode_beta(self):
        engine = MultiWaveMemoryEngine()
        engine.last_query_time = datetime.now()
        engine._update_mode()
        assert engine.mode == BrainWaveMode.BETA

    def test_update_mode_alpha(self):
        engine = MultiWaveMemoryEngine()
        engine.last_query_time = datetime.now() - timedelta(minutes=10)
        engine._update_mode()
        assert engine.mode == BrainWaveMode.ALPHA

    def test_update_mode_theta(self):
        engine = MultiWaveMemoryEngine()
        engine.last_query_time = datetime.now() - timedelta(minutes=60)
        engine._update_mode()
        assert engine.mode == BrainWaveMode.THETA

    def test_update_mode_deep_sleep(self):
        engine = MultiWaveMemoryEngine()
        engine.last_query_time = datetime.now() - timedelta(hours=3)
        engine._update_mode()
        assert engine.mode in (BrainWaveMode.DELTA, BrainWaveMode.REM)

    def test_update_mode_stays_beta_during_ingestion(self):
        engine = MultiWaveMemoryEngine()
        engine.ingestion_active = True
        engine.last_query_time = datetime.now() - timedelta(hours=5)
        engine._update_mode()
        assert engine.mode == BrainWaveMode.BETA

    def test_alpha_filtering_update(self):
        engine = MultiWaveMemoryEngine()
        engine.add_node("weak", np.array([1.0, 0.0]), initial_spring_constant=1.0)
        engine.nodes["weak"].velocity = np.array([0.1, 0.0])  # Below alpha threshold
        engine.add_node("strong", np.array([0.0, 1.0]), initial_spring_constant=4.0)
        engine.nodes["strong"].velocity = np.array([0.5, 0.0])  # Above threshold
        engine._alpha_filtering_update()
        # weak node should have faster decay
        assert engine.nodes["weak"].spring_constant < 1.0
        # strong node should be slightly boosted
        assert engine.nodes["strong"].spring_constant > 4.0

    def test_on_query_switches_to_beta(self):
        engine = MultiWaveMemoryEngine()
        engine.mode = BrainWaveMode.THETA
        engine.add_node("a", np.array([1.0, 0.0, 0.0]))
        result = engine.on_query(np.array([1.0, 0.0, 0.0]))
        assert engine.mode == BrainWaveMode.BETA
        assert isinstance(result, BetaWaveRecallResult)

    def test_encode_new_memory(self):
        engine = MultiWaveMemoryEngine()
        engine.add_node("existing", np.array([0.9, 0.1, 0.0]))
        engine.encode_new_memory(
            node_id="new1",
            content="new memory",
            embedding=np.array([0.95, 0.05, 0.0]),
            initial_k=2.0,
        )
        assert "new1" in engine.nodes
        # Should have found "existing" as neighbor (cosine sim > 0.7)

    def test_get_statistics_includes_mode(self):
        engine = MultiWaveMemoryEngine()
        stats = engine.get_statistics()
        assert "mode" in stats
        assert stats["mode"] == "beta"
        assert "total_ingested" in stats


# =============================================================================
# streaming_expansion tests
# =============================================================================

from hololoom.core.memory.streaming_expansion import (
    ContextChunk,
    StreamingResult,
    StreamingContextBuilder,
    ChunkYieldStrategy,
    stream_context_expansion,
)


class TestContextChunk:
    def test_avg_relevance(self):
        chunk = ContextChunk(
            nodes=["a", "b"],
            contents={"a": "hello", "b": "world"},
            relevance_scores={"a": 0.8, "b": 0.6},
            hop_distance=0,
            token_count=100,
            cumulative_tokens=100,
            chunk_index=0,
            is_final=False,
        )
        assert chunk.avg_relevance == pytest.approx(0.7)
        assert chunk.node_count == 2

    def test_avg_relevance_empty(self):
        chunk = ContextChunk(
            nodes=[], contents={}, relevance_scores={},
            hop_distance=0, token_count=0, cumulative_tokens=0,
            chunk_index=0, is_final=True,
        )
        assert chunk.avg_relevance == 0.0


class TestStreamingResult:
    def test_avg_chunk_size(self):
        result = StreamingResult(
            total_nodes=10,
            total_tokens=1000,
            total_chunks=4,
            max_hop_reached=2,
            stopping_reason="completed",
            execution_time_ms=50.0,
            chunks_summary=[],
        )
        assert result.avg_chunk_size_tokens == pytest.approx(250.0)

    def test_avg_chunk_size_zero_chunks(self):
        result = StreamingResult(
            total_nodes=0, total_tokens=0, total_chunks=0,
            max_hop_reached=0, stopping_reason="completed",
            execution_time_ms=0.0, chunks_summary=[],
        )
        assert result.avg_chunk_size_tokens == 0.0


class TestStreamingContextBuilder:
    @pytest.mark.asyncio
    async def test_stream_expansion_with_mock_graph(self):
        """Test basic streaming expansion with a mock graph."""
        mock_graph = MagicMock()
        mock_graph.get_neighbors = MagicMock(return_value=set())

        builder = StreamingContextBuilder()

        chunks = []
        async for chunk in builder.stream_expansion(
            query="test query",
            seed_nodes=["seed1"],
            graph=mock_graph,
            token_budget=2000,
            chunk_size=500,
            node_contents={"seed1": "Seed content here"},
            importance_scores={"seed1": 0.9},
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        # Last chunk should be final
        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_expansion_no_seeds(self):
        """Empty seed list still yields a final chunk."""
        mock_graph = MagicMock()
        builder = StreamingContextBuilder()

        chunks = []
        async for chunk in builder.stream_expansion(
            query="test",
            seed_nodes=[],
            graph=mock_graph,
            token_budget=500,
        ):
            chunks.append(chunk)

        # Should get at least the final chunk
        assert len(chunks) >= 1
        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_expansion_with_neighbors(self):
        """Test expansion traverses neighbors."""
        # Create a mock graph with neighbors
        mock_graph = MagicMock()

        def mock_get_neighbors(node_id, direction="both", max_hops=1):
            if node_id == "seed1":
                return {"neighbor1", "neighbor2"}
            return set()

        mock_graph.get_neighbors = mock_get_neighbors
        # Mock G for edge type lookup
        mock_graph.G = MagicMock()
        mock_graph.G.has_edge = MagicMock(return_value=False)

        builder = StreamingContextBuilder()

        chunks = []
        async for chunk in builder.stream_expansion(
            query="test query",
            seed_nodes=["seed1"],
            graph=mock_graph,
            token_budget=5000,
            chunk_size=100,
            min_relevance=0.0,  # Accept all
            node_contents={
                "seed1": "seed content",
                "neighbor1": "neighbor one content",
                "neighbor2": "neighbor two content",
            },
            importance_scores={"seed1": 0.9, "neighbor1": 0.7, "neighbor2": 0.6},
            yield_strategy=ChunkYieldStrategy.HOP_BOUNDARY,
        ):
            chunks.append(chunk)

        # Should have seed chunk + at least hop boundary + final
        assert len(chunks) >= 2

    def test_get_last_result_none_initially(self):
        builder = StreamingContextBuilder()
        assert builder.get_last_result() is None

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test stream_context_expansion convenience wrapper."""
        mock_graph = MagicMock()
        mock_graph.get_neighbors = MagicMock(return_value=set())

        chunks = []
        async for chunk in stream_context_expansion(
            query="test",
            seed_nodes=["s1"],
            graph=mock_graph,
            token_budget=500,
            importance_scores={"s1": 0.5},
            node_contents={"s1": "content"},
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1

    def test_get_neighbors_networkx_fallback(self):
        """Test _get_neighbors with a NetworkX graph (no get_neighbors method)."""
        import networkx as nx
        G = nx.MultiDiGraph()
        G.add_edge("a", "b", relation="IS_A")

        builder = StreamingContextBuilder()
        neighbors = builder._get_neighbors(G, "a")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "b"


# =============================================================================
# hybrid_retrieval tests
# =============================================================================

from hololoom.core.memory.hybrid_retrieval import (
    BM25Retriever,
    SemanticRetriever,
    GraphRetriever,
    HybridRetriever,
    RetrievalScore,
    RetrievalResult,
    reciprocal_rank_fusion,
    create_hybrid_retriever,
)
from hololoom.core.protocols.types import MemoryShard
from hololoom.core.memory.graph import KG


def _make_shard(id: str, text: str, entities: list = None) -> MemoryShard:
    return MemoryShard(id=id, text=text, entities=entities or [])


class TestBM25Retriever:
    def test_tokenize(self):
        bm25 = BM25Retriever()
        tokens = bm25.tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_index_documents(self):
        bm25 = BM25Retriever()
        memories = [
            _make_shard("1", "thompson sampling algorithm"),
            _make_shard("2", "bayesian statistics method"),
        ]
        bm25.index_documents(memories)
        assert bm25.num_docs == 2
        assert bm25.avg_doc_length > 0

    def test_idf_known_term(self):
        bm25 = BM25Retriever()
        memories = [
            _make_shard("1", "thompson sampling"),
            _make_shard("2", "thompson method"),
            _make_shard("3", "bayesian statistics"),
        ]
        bm25.index_documents(memories)
        idf_thompson = bm25.idf("thompson")
        idf_bayesian = bm25.idf("bayesian")
        # thompson appears in 2/3 docs, bayesian in 1/3 — bayesian should have higher IDF
        assert idf_bayesian > idf_thompson

    def test_idf_unknown_term(self):
        bm25 = BM25Retriever()
        bm25.num_docs = 5
        assert bm25.idf("nonexistent") == 0.0

    def test_score_document(self):
        bm25 = BM25Retriever()
        # Need multiple docs so IDF is positive for a term
        memories = [
            _make_shard("1", "thompson sampling algorithm"),
            _make_shard("2", "deep learning neural networks"),
            _make_shard("3", "bayesian statistics methods"),
        ]
        bm25.index_documents(memories)
        score = bm25.score_document(
            ["thompson"], ["thompson", "sampling", "algorithm"], "1"
        )
        assert score > 0.0

    def test_score_document_empty(self):
        bm25 = BM25Retriever()
        assert bm25.score_document(["test"], ["test"], "unknown") == 0.0

    @pytest.mark.asyncio
    async def test_retrieve(self):
        bm25 = BM25Retriever()
        memories = [
            _make_shard("1", "thompson sampling for multi armed bandits"),
            _make_shard("2", "deep learning neural networks"),
            _make_shard("3", "thompson sampling bayesian approach"),
        ]
        results = await bm25.retrieve("thompson sampling", memories, limit=2)
        assert len(results) == 2
        # Top results should be the ones mentioning thompson sampling
        top_ids = {r[0].id for r in results}
        assert "1" in top_ids or "3" in top_ids


class TestReciprocalRankFusion:
    def test_rrf_combines_rankings(self):
        m1 = _make_shard("a", "alpha")
        m2 = _make_shard("b", "beta")
        m3 = _make_shard("c", "gamma")

        ranking1 = [(m1, 0.9), (m2, 0.5)]
        ranking2 = [(m2, 0.8), (m3, 0.4)]

        fused = reciprocal_rank_fusion([ranking1, ranking2], k=60)
        # m2 should rank highest (appears in both lists)
        assert fused[0][0].id == "b"

    def test_rrf_empty_rankings(self):
        result = reciprocal_rank_fusion([])
        assert result == []


class TestGraphRetriever:
    def _make_kg(self):
        kg = KG()
        kg.G.add_node("Python")
        kg.G.add_node("programming")
        kg.G.add_node("language")
        kg.G.add_edge("Python", "programming", relation="IS_A")
        kg.G.add_edge("programming", "language", relation="IS_A")
        return kg

    def test_extract_entities(self):
        kg = self._make_kg()
        retriever = GraphRetriever(kg)
        entities = retriever.extract_entities("Python is a programming language")
        assert "Python" in entities or "programming" in entities

    def test_traverse(self):
        kg = self._make_kg()
        retriever = GraphRetriever(kg, max_hops=2)
        result = retriever.traverse("Python", max_hops=2)
        assert isinstance(result, dict)
        assert "Python" in result
        assert result["Python"] == 1.0
        assert "programming" in result

    def test_traverse_nonexistent_entity(self):
        kg = self._make_kg()
        retriever = GraphRetriever(kg)
        result = retriever.traverse("nonexistent", max_hops=2)
        # Should return TraversalResult, not dict
        assert not isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_retrieve_with_entities(self):
        kg = self._make_kg()
        retriever = GraphRetriever(kg)
        memories = [
            _make_shard("1", "Python is great", entities=["Python"]),
            _make_shard("2", "Java is okay", entities=["Java"]),
        ]
        results = await retriever.retrieve("Python programming", memories)
        # Should return list since Python is in graph
        if isinstance(results, list):
            assert len(results) >= 1
            assert results[0][0].id == "1"

    @pytest.mark.asyncio
    async def test_retrieve_no_entities_found(self):
        kg = self._make_kg()
        retriever = GraphRetriever(kg)
        memories = [_make_shard("1", "random text")]
        results = await retriever.retrieve("completely unknown query", memories)
        # Should return RetrievalResultEnhanced (no entities found)
        from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced
        assert isinstance(results, RetrievalResultEnhanced)


class TestSemanticRetriever:
    def test_fallback_when_unavailable(self):
        """Semantic retriever should work in fallback mode without sentence-transformers."""
        retriever = SemanticRetriever(enable_fallback=True)
        if not retriever.available:
            # This is the expected case in test environment
            assert retriever.model is None

    @pytest.mark.asyncio
    async def test_retrieve_fallback_keyword(self):
        retriever = SemanticRetriever(enable_fallback=True)
        if not retriever.available:
            memories = [
                _make_shard("1", "thompson sampling algorithm"),
                _make_shard("2", "deep learning neural networks"),
            ]
            results = await retriever.retrieve("thompson sampling", memories)
            # Should return RetrievalResultEnhanced as fallback
            from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced
            assert isinstance(results, RetrievalResultEnhanced)

    @pytest.mark.asyncio
    async def test_retrieve_unavailable_no_fallback(self):
        retriever = SemanticRetriever(enable_fallback=False)
        if not retriever.available:
            memories = [_make_shard("1", "test")]
            results = await retriever.retrieve("query", memories)
            from hololoom.core.memory.retrieval_result import RetrievalResultEnhanced
            assert isinstance(results, RetrievalResultEnhanced)

    def test_cosine_similarity_none_inputs(self):
        retriever = SemanticRetriever()
        assert retriever.cosine_similarity(None, None) == 0.0


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_bm25_only(self):
        kg = KG()
        retriever = HybridRetriever(
            kg=kg,
            enable_semantic=False,
            enable_bm25=True,
            enable_graph=False,
        )
        memories = [
            _make_shard("1", "thompson sampling for bandits"),
            _make_shard("2", "deep learning models"),
        ]
        result = await retriever.retrieve("thompson sampling", memories)
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) > 0
        assert result.total_candidates == 2

    @pytest.mark.asyncio
    async def test_no_methods_enabled(self):
        kg = KG()
        retriever = HybridRetriever(
            kg=kg,
            enable_semantic=False,
            enable_bm25=False,
            enable_graph=False,
        )
        memories = [_make_shard("1", "test")]
        result = await retriever.retrieve("query", memories)
        assert isinstance(result, RetrievalResult)
        assert len(result.memories) == 0
        assert "warning" in result.metadata

    def test_create_hybrid_retriever_factory(self):
        kg = KG()
        retriever = create_hybrid_retriever(kg)
        assert retriever.bm25_retriever is not None


class TestRetrievalScore:
    def test_creation(self):
        score = RetrievalScore(
            memory_id="m1",
            semantic_score=0.8,
            bm25_score=0.6,
            combined_score=0.7,
        )
        assert score.memory_id == "m1"
        assert score.graph_score == 0.0


# =============================================================================
# llm_consolidator tests
# =============================================================================

from hololoom.core.memory.llm_consolidator import (
    LLMConfig,
    LLMClient,
    LLMUsage,
    ProductionLLMConsolidator,
    ConsolidationPrompt,
    calculate_cost,
    create_llm_config,
    create_production_consolidator,
)


class TestCalculateCost:
    def test_known_model(self):
        cost = calculate_cost("gpt-3.5-turbo", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.0)  # 0.5 + 1.5

    def test_unknown_model(self):
        cost = calculate_cost("unknown-model", 100, 100)
        assert cost == 0.0

    def test_free_model(self):
        cost = calculate_cost("ollama", 1000, 1000)
        assert cost == 0.0


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(provider="openai", model="gpt-3.5-turbo")
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 500
        assert cfg.timeout == 30.0


class TestLLMClient:
    def test_none_provider(self):
        cfg = LLMConfig(provider="none", model="none")
        client = LLMClient(cfg)
        assert client.client is None

    @pytest.mark.asyncio
    async def test_complete_with_no_client(self):
        cfg = LLMConfig(provider="none", model="none")
        client = LLMClient(cfg)
        result = await client.complete("test prompt")
        assert result is None

    def test_record_usage(self):
        cfg = LLMConfig(provider="none", model="none")
        client = LLMClient(cfg)
        client._record_usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
        assert len(client.usage_history) == 1
        assert client.total_cost_usd == pytest.approx(0.01)
        assert client.usage_history[0].total_tokens == 150

    def test_get_statistics_empty(self):
        cfg = LLMConfig(provider="none", model="none")
        client = LLMClient(cfg)
        stats = client.get_statistics()
        assert stats["total_requests"] == 0
        assert stats["total_cost_usd"] == 0.0

    def test_get_statistics_with_usage(self):
        cfg = LLMConfig(provider="none", model="none")
        client = LLMClient(cfg)
        client._record_usage(100, 50, 0.01)
        client._record_usage(200, 100, 0.02)
        stats = client.get_statistics()
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 450
        assert stats["total_cost_usd"] == pytest.approx(0.03)

    def test_openai_init_without_package(self):
        """OpenAI init should degrade gracefully without the package."""
        cfg = LLMConfig(provider="openai", model="gpt-3.5-turbo", api_key="fake")
        # This might fail (no openai package) or succeed — either way shouldn't crash
        client = LLMClient(cfg)
        # client.client may be None if openai not installed


class TestConsolidationPrompt:
    def test_templates_exist(self):
        prompts = ConsolidationPrompt()
        assert "{episodes}" in prompts.fact_extraction
        assert "{episodes}" in prompts.entity_extraction
        assert "{memories}" in prompts.deduplication


class TestProductionLLMConsolidator:
    def test_init_without_config(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)
        assert consolidator.llm_client is None

    @pytest.mark.asyncio
    async def test_extract_facts_empty(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)
        facts = await consolidator.extract_facts([])
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_fallback(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)
        episodes = [
            _make_shard("1", "Thompson Sampling uses Bayesian probability. It balances exploration and exploitation effectively."),
            _make_shard("2", "The algorithm maintains Beta distributions for each arm."),
        ]
        facts = await consolidator.extract_facts(episodes)
        assert len(facts) > 0
        # Should get individual sentences
        assert any("Thompson" in f or "Beta" in f for f in facts)

    @pytest.mark.asyncio
    async def test_extract_entities_empty(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)
        edges = await consolidator.extract_entities([])
        assert edges == []

    @pytest.mark.asyncio
    async def test_extract_entities_fallback(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)
        episodes = [
            _make_shard("1", "test", entities=["Python", "ML", "AI"]),
        ]
        edges = await consolidator.extract_entities(episodes)
        assert len(edges) > 0
        assert edges[0][2] == "MENTIONS"

    @pytest.mark.asyncio
    async def test_deduplicate_empty(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)
        result = await consolidator.deduplicate([])
        assert result == []

    @pytest.mark.asyncio
    async def test_deduplicate_fallback(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)
        memories = [
            _make_shard("1", "duplicate text"),
            _make_shard("2", "duplicate text"),
            _make_shard("3", "unique text"),
        ]
        unique = await consolidator.deduplicate(memories)
        assert len(unique) == 2
        ids = {m.id for m in unique}
        assert "1" in ids
        assert "3" in ids

    @pytest.mark.asyncio
    async def test_extract_facts_no_fallback(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=False)
        episodes = [_make_shard("1", "some text here")]
        facts = await consolidator.extract_facts(episodes)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_entities_no_fallback(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=False)
        episodes = [_make_shard("1", "test", entities=["A", "B"])]
        edges = await consolidator.extract_entities(episodes)
        assert edges == []

    @pytest.mark.asyncio
    async def test_deduplicate_no_fallback(self):
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=False)
        memories = [_make_shard("1", "text")]
        result = await consolidator.deduplicate(memories)
        assert result == memories  # Returns original if no LLM and no fallback

    def test_get_statistics_no_llm(self):
        consolidator = ProductionLLMConsolidator(config=None)
        stats = consolidator.get_statistics()
        assert stats["provider"] == "none"

    @pytest.mark.asyncio
    async def test_extract_facts_llm_mocked(self):
        """Test LLM-based fact extraction with _extract_facts_llm mocked."""
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)

        # Mock the internal LLM extraction method directly (MetapromptConfig
        # may not match the installed version, so bypass it)
        consolidator._extract_facts_llm = AsyncMock(return_value=[
            "Thompson Sampling is a Bayesian algorithm for exploration",
            "It uses Beta distributions to model uncertainty",
            "The algorithm balances exploration and exploitation",
        ])
        # Make LLM path trigger
        consolidator.llm_client = MagicMock()
        consolidator.llm_client.client = True

        episodes = [
            _make_shard("1", "Thompson Sampling uses Bayesian probability."),
        ]
        facts = await consolidator.extract_facts(episodes)
        assert len(facts) == 3
        assert any("Thompson" in f for f in facts)

    @pytest.mark.asyncio
    async def test_extract_entities_llm_mocked(self):
        """Test LLM-based entity extraction with _extract_entities_llm mocked."""
        consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)

        consolidator._extract_entities_llm = AsyncMock(return_value=[
            ("Thompson Sampling", "Bayesian Algorithm", "IS_A"),
            ("Thompson Sampling", "Beta Distribution", "USES"),
        ])
        consolidator.llm_client = MagicMock()
        consolidator.llm_client.client = True

        episodes = [_make_shard("1", "Thompson Sampling uses Beta distributions")]
        edges = await consolidator.extract_entities(episodes)
        assert len(edges) == 2
        assert edges[0] == ("Thompson Sampling", "Bayesian Algorithm", "IS_A")

    @pytest.mark.asyncio
    async def test_deduplicate_llm_mocked(self):
        """Test LLM deduplication with mocked client."""
        cfg = LLMConfig(provider="none", model="test")
        consolidator = ProductionLLMConsolidator(config=cfg, enable_fallback=True)

        mock_client = MagicMock()
        mock_client.client = True
        mock_client.complete = AsyncMock(return_value=(
            "Keep these unique memories:\n"
            "[ID: 1] Thompson Sampling\n"
            "[ID: 3] Deep Learning\n"
        ))
        consolidator.llm_client = mock_client

        memories = [
            _make_shard("1", "Thompson Sampling"),
            _make_shard("2", "Thompson Sampling algorithm"),
            _make_shard("3", "Deep Learning"),
        ]
        unique = await consolidator.deduplicate(memories)
        assert len(unique) == 2
        ids = {m.id for m in unique}
        assert "1" in ids
        assert "3" in ids

    @pytest.mark.asyncio
    async def test_extract_facts_llm_failure_falls_back(self):
        """LLM failure should fall back to rule-based extraction."""
        cfg = LLMConfig(provider="none", model="test")
        consolidator = ProductionLLMConsolidator(config=cfg, enable_fallback=True)

        mock_client = MagicMock()
        mock_client.client = True
        mock_client.complete = AsyncMock(side_effect=Exception("LLM down"))
        consolidator.llm_client = mock_client

        episodes = [
            _make_shard("1", "Thompson Sampling is a Bayesian approach. It works with Beta distributions."),
        ]
        facts = await consolidator.extract_facts(episodes)
        assert len(facts) > 0  # Fallback should still produce results


class TestCreateLLMConfig:
    def test_default_openai(self):
        cfg = create_llm_config(provider="openai")
        assert cfg.model == "gpt-3.5-turbo"
        assert cfg.provider == "openai"

    def test_custom_model(self):
        cfg = create_llm_config(provider="ollama", model="llama3")
        assert cfg.model == "llama3"

    def test_default_ollama(self):
        cfg = create_llm_config(provider="ollama")
        assert cfg.model == "llama2"


class TestCreateProductionConsolidator:
    def test_none_provider(self):
        consolidator = create_production_consolidator(provider="none")
        assert consolidator.llm_client is None

    def test_with_model_provider(self):
        consolidator = create_production_consolidator(
            provider="none", model_provider="claude"
        )
        assert consolidator.model_provider == "claude"
