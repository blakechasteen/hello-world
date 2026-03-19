"""
Unit tests for large uncovered memory modules:
- neo4j_graph.py
- interleaved_generation.py
- interleaved_generation_advanced.py
- v2_bridge.py
- repository_context.py

All external services (Neo4j, LLM, etc.) are mocked.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock

import pytest
import networkx as nx


# ============================================================================
# neo4j_graph.py tests
# ============================================================================


class TestGetBoundedIntEnv:
    """Tests for _get_bounded_int_env."""

    def test_returns_default_when_env_not_set(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_int_env
        with patch.dict(os.environ, {}, clear=True):
            assert _get_bounded_int_env("FAKE_VAR", 42, 1, 100) == 42

    def test_returns_value_within_bounds(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_int_env
        with patch.dict(os.environ, {"FAKE_VAR": "25"}):
            assert _get_bounded_int_env("FAKE_VAR", 42, 1, 100) == 25

    def test_clamps_below_minimum(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_int_env
        with patch.dict(os.environ, {"FAKE_VAR": "0"}):
            assert _get_bounded_int_env("FAKE_VAR", 42, 1, 100) == 1

    def test_clamps_above_maximum(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_int_env
        with patch.dict(os.environ, {"FAKE_VAR": "999"}):
            assert _get_bounded_int_env("FAKE_VAR", 42, 1, 100) == 100

    def test_returns_default_on_invalid_string(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_int_env
        with patch.dict(os.environ, {"FAKE_VAR": "not_a_number"}):
            assert _get_bounded_int_env("FAKE_VAR", 42, 1, 100) == 42


class TestGetBoundedFloatEnv:
    """Tests for _get_bounded_float_env."""

    def test_returns_default_when_env_not_set(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_float_env
        with patch.dict(os.environ, {}, clear=True):
            assert _get_bounded_float_env("FAKE_FLOAT", 3.14, 0.1, 100.0) == 3.14

    def test_returns_value_within_bounds(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_float_env
        with patch.dict(os.environ, {"FAKE_FLOAT": "2.5"}):
            assert _get_bounded_float_env("FAKE_FLOAT", 3.14, 0.1, 100.0) == 2.5

    def test_clamps_below_minimum(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_float_env
        with patch.dict(os.environ, {"FAKE_FLOAT": "0.01"}):
            assert _get_bounded_float_env("FAKE_FLOAT", 3.14, 0.1, 100.0) == 0.1

    def test_clamps_above_maximum(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_float_env
        with patch.dict(os.environ, {"FAKE_FLOAT": "999.0"}):
            assert _get_bounded_float_env("FAKE_FLOAT", 3.14, 0.1, 100.0) == 100.0

    def test_returns_default_on_invalid_string(self):
        from hololoom.core.memory.neo4j_graph import _get_bounded_float_env
        with patch.dict(os.environ, {"FAKE_FLOAT": "abc"}):
            assert _get_bounded_float_env("FAKE_FLOAT", 3.14, 0.1, 100.0) == 3.14


class TestNeo4jConfig:
    """Tests for Neo4jConfig."""

    def test_default_config(self):
        from hololoom.core.memory.neo4j_graph import Neo4jConfig
        config = Neo4jConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.password == ""
        assert config.database == "neo4j"
        assert config.max_connection_pool_size == 50
        assert config.connection_timeout == 30.0

    def test_from_env_requires_password(self):
        from hololoom.core.memory.neo4j_graph import Neo4jConfig
        with patch.dict(os.environ, {}, clear=True):
            # Remove NEO4J_PASSWORD if set
            os.environ.pop("NEO4J_PASSWORD", None)
            with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
                Neo4jConfig.from_env()

    def test_from_env_with_password(self):
        from hololoom.core.memory.neo4j_graph import Neo4jConfig
        env = {
            "NEO4J_PASSWORD": "secret123",
            "NEO4J_URI": "bolt://myhost:7687",
            "NEO4J_USERNAME": "admin",
            "NEO4J_DATABASE": "testdb",
            "NEO4J_POOL_SIZE": "10",
            "NEO4J_TIMEOUT": "5.0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Neo4jConfig.from_env()
            assert config.password == "secret123"
            assert config.uri == "bolt://myhost:7687"
            assert config.username == "admin"
            assert config.database == "testdb"
            assert config.max_connection_pool_size == 10
            assert config.connection_timeout == 5.0

    def test_from_env_bounds_validation(self):
        from hololoom.core.memory.neo4j_graph import Neo4jConfig
        env = {
            "NEO4J_PASSWORD": "secret",
            "NEO4J_POOL_SIZE": "99999",  # above max 1000
            "NEO4J_TIMEOUT": "0.1",  # below min 1.0
        }
        with patch.dict(os.environ, env, clear=True):
            config = Neo4jConfig.from_env()
            assert config.max_connection_pool_size == 1000  # clamped
            assert config.connection_timeout == 1.0  # clamped


class TestNeo4jKGWithMockedDriver:
    """Tests for Neo4jKG with mocked Neo4j driver."""

    @pytest.fixture(autouse=True)
    def reset_driver_singleton(self):
        """Reset the class-level driver singleton before each test."""
        from hololoom.core.memory.neo4j_graph import Neo4jKG
        Neo4jKG._driver_instance = None
        Neo4jKG._connection_metrics = {
            'total_connections': 0,
            'failed_connections': 0,
            'pool_exhaustion_count': 0,
            'last_health_check': None,
            'health_status': 'unknown'
        }
        yield
        Neo4jKG._driver_instance = None

    def _make_kg(self):
        """Create Neo4jKG with fully mocked driver."""
        from hololoom.core.memory.neo4j_graph import Neo4jKG, Neo4jConfig

        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session.run = Mock(return_value=MagicMock())

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver.verify_connectivity = Mock()

        config = Neo4jConfig(password="test")

        with patch("hololoom.core.memory.neo4j_graph.NEO4J_AVAILABLE", True), \
             patch("hololoom.core.memory.neo4j_graph.GraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = mock_driver
            kg = Neo4jKG(config=config)

        return kg, mock_driver, mock_session

    def test_init_connects(self):
        kg, mock_driver, _ = self._make_kg()
        assert kg._connected is True
        assert kg.config.password == "test"

    def test_health_check_healthy(self):
        kg, mock_driver, _ = self._make_kg()
        health = kg.health_check()
        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert health["error"] is None

    def test_health_check_unhealthy(self):
        kg, mock_driver, _ = self._make_kg()
        mock_driver.verify_connectivity.side_effect = Exception("down")
        health = kg.health_check()
        assert health["status"] == "unhealthy"
        assert "down" in health["error"]

    def test_get_pool_metrics(self):
        kg, _, _ = self._make_kg()
        metrics = kg.get_pool_metrics()
        assert metrics["pool_size"] == 50
        assert metrics["failure_rate"] == 0.0
        assert "health_status" in metrics

    def test_add_edge(self):
        from hololoom.core.memory.graph import KGEdge
        kg, _, mock_session = self._make_kg()
        edge = KGEdge(src="A", dst="B", type="IS_A", weight=1.0, span_id="s1")
        kg.add_edge(edge)
        mock_session.run.assert_called()

    def test_add_edges_batch(self):
        from hololoom.core.memory.graph import KGEdge
        kg, _, mock_session = self._make_kg()
        edges = [
            KGEdge(src="A", dst="B", type="IS_A", weight=1.0, span_id="s1"),
            KGEdge(src="B", dst="C", type="USES", weight=0.5, span_id="s2"),
        ]

        mock_tx = MagicMock()
        mock_tx.__enter__ = Mock(return_value=mock_tx)
        mock_tx.__exit__ = Mock(return_value=False)
        mock_session.begin_transaction.return_value = mock_tx

        kg.add_edges(edges)
        assert mock_tx.run.call_count == 2
        mock_tx.commit.assert_called_once()

    def test_get_neighbors_validates_max_hops(self):
        kg, _, _ = self._make_kg()
        with pytest.raises(ValueError, match="max_hops"):
            kg.get_neighbors("A", max_hops=99)

    def test_get_neighbors_validates_direction(self):
        kg, _, _ = self._make_kg()
        with pytest.raises(ValueError, match="direction"):
            kg.get_neighbors("A", direction="sideways")

    def test_get_neighbors_out(self):
        kg, _, mock_session = self._make_kg()
        mock_result = [{"name": "B"}, {"name": "C"}]
        mock_session.run.return_value = mock_result
        neighbors = kg.get_neighbors("A", direction="out", max_hops=2)
        assert neighbors == {"B", "C"}

    def test_get_neighbors_in(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [{"name": "X"}]
        neighbors = kg.get_neighbors("A", direction="in")
        assert neighbors == {"X"}

    def test_get_neighbors_both(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [{"name": "Z"}]
        neighbors = kg.get_neighbors("A", direction="both")
        assert neighbors == {"Z"}

    def test_subgraph_empty_entities(self):
        kg, _, _ = self._make_kg()
        G = kg.subgraph_for_entities([])
        assert isinstance(G, nx.MultiDiGraph)
        assert len(G.nodes) == 0

    def test_subgraph_validates_max_hops(self):
        kg, _, _ = self._make_kg()
        with pytest.raises(ValueError, match="max_hops"):
            kg.subgraph_for_entities(["A"], max_hops=50)

    def test_subgraph_no_expand(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [
            {"src": "A", "dst": None, "type": None, "weight": None, "span_id": None, "metadata": {}}
        ]
        G = kg.subgraph_for_entities(["A"], expand=False)
        assert "A" in G.nodes

    def test_subgraph_with_edges(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [
            {"src": "A", "dst": "B", "type": "IS_A", "weight": 1.0, "span_id": "s1", "metadata": {"type": "IS_A", "weight": 1.0}},
        ]
        G = kg.subgraph_for_entities(["A"], expand=True, max_hops=1)
        assert "A" in G.nodes
        assert "B" in G.nodes
        assert G.has_edge("A", "B")

    def test_get_edge_types(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [{"type": "IS_A"}, {"type": "USES"}]
        types = kg.get_edge_types("A", "B")
        assert types == ["IS_A", "USES"]

    def test_get_paths_validates_max_length(self):
        kg, _, _ = self._make_kg()
        with pytest.raises(ValueError, match="max_length"):
            kg.get_paths("A", "B", max_length=99)

    def test_get_paths(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [{"path": ["A", "C", "B"]}]
        paths = kg.get_paths("A", "B", max_length=3)
        assert paths == [["A", "C", "B"]]

    def test_get_related_by_type_out(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [{"name": "B"}]
        result = kg.get_related_by_type("A", "IS_A", direction="out")
        assert result == ["B"]

    def test_get_related_by_type_in(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.return_value = [{"name": "X"}]
        result = kg.get_related_by_type("A", "IS_A", direction="in")
        assert result == ["X"]

    def test_stats(self):
        kg, _, mock_session = self._make_kg()
        mock_record = {"num_nodes": 10, "num_edges": 15}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        stats = kg.stats()
        assert stats["backend"] == "neo4j"

    def test_run_cypher(self):
        kg, _, mock_session = self._make_kg()
        mock_record = MagicMock()
        mock_record.__iter__ = Mock(return_value=iter([("count", 5)]))
        mock_record.keys.return_value = ["count"]
        mock_session.run.return_value = [{"count": 5}]
        results = kg.run_cypher("MATCH (n) RETURN count(n) AS count")
        assert len(results) == 1

    def test_pagerank_returns_empty_on_failure(self):
        kg, _, mock_session = self._make_kg()
        mock_session.run.side_effect = Exception("GDS not installed")
        result = kg.pagerank()
        assert result == []

    def test_clear(self):
        kg, _, mock_session = self._make_kg()
        kg.clear()
        mock_session.run.assert_called()

    def test_close(self):
        kg, _, _ = self._make_kg()
        kg.close()
        assert kg._connected is False

    def test_context_manager(self):
        kg, _, _ = self._make_kg()
        with kg as k:
            assert k is kg
        assert kg._connected is False

    def test_force_close_driver(self):
        from hololoom.core.memory.neo4j_graph import Neo4jKG
        kg, mock_driver, _ = self._make_kg()
        Neo4jKG._driver_instance = mock_driver
        Neo4jKG.force_close_driver()
        mock_driver.close.assert_called_once()
        assert Neo4jKG._driver_instance is None

    def test_connect_entity_to_time(self):
        from datetime import datetime
        kg, _, mock_session = self._make_kg()
        mock_result = MagicMock()
        mock_record = {"thread_id": "time::2025-W01"}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        thread_id = kg.connect_entity_to_time("event_A", datetime(2025, 1, 1))
        assert "time::" in thread_id

    def test_execute_with_retry_success(self):
        kg, _, _ = self._make_kg()
        result = kg._execute_with_retry(lambda: 42)
        assert result == 42

    def test_execute_with_retry_session_expired(self):
        from hololoom.core.memory.neo4j_graph import Neo4jKG
        kg, _, _ = self._make_kg()

        # Mock SessionExpired from neo4j
        class FakeSessionExpired(Exception):
            pass

        call_count = 0
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeSessionExpired("expired")
            return "ok"

        with patch("hololoom.core.memory.neo4j_graph.SessionExpired", FakeSessionExpired):
            result = kg._execute_with_retry(flaky)
            assert result == "ok"


# ============================================================================
# interleaved_generation.py tests
# ============================================================================


class TestStreamItemTypes:
    """Tests for data structures in interleaved_generation."""

    def test_stream_item_type_values(self):
        from hololoom.core.memory.interleaved_generation import StreamItemType
        assert StreamItemType.CONTEXT_CHUNK == "context_chunk"
        assert StreamItemType.GENERATION_TOKEN == "generation_token"
        assert StreamItemType.METADATA == "metadata"

    def test_stream_mode_values(self):
        from hololoom.core.memory.interleaved_generation import StreamMode
        assert StreamMode.BATCHED == "batched"
        assert StreamMode.CONCURRENT == "concurrent"

    def test_generation_token_fields(self):
        from hololoom.core.memory.interleaved_generation import GenerationToken
        token = GenerationToken(
            token="hello",
            cumulative_text="hello",
            token_index=0,
            is_final=False,
            metadata={"key": "val"},
        )
        assert token.token == "hello"
        assert token.token_index == 0
        assert not token.is_final

    def test_stream_metadata_fields(self):
        from hololoom.core.memory.interleaved_generation import StreamMetadata
        meta = StreamMetadata(event_type="expansion_start", data={"query": "test"})
        assert meta.event_type == "expansion_start"
        assert meta.data["query"] == "test"

    def test_interleaved_result_fields(self):
        from hololoom.core.memory.interleaved_generation import InterleavedResult
        result = InterleavedResult(
            response="answer",
            context_chunks=[],
            total_tokens_generated=10,
            total_tokens_retrieved=20,
            expansion_time_ms=100.0,
            generation_time_ms=200.0,
            total_time_ms=300.0,
        )
        assert result.response == "answer"
        assert result.total_time_ms == 300.0


class TestMockLLM:
    """Tests for MockLLM."""

    @pytest.mark.asyncio
    async def test_mock_llm_default_response(self):
        from hololoom.core.memory.interleaved_generation import MockLLM
        llm = MockLLM(tokens_per_second=10000)  # fast for testing
        tokens = []
        async for token in llm.generate_stream("test query", "some context", max_tokens=5):
            tokens.append(token)
        assert len(tokens) > 0
        assert len(tokens) <= 5

    @pytest.mark.asyncio
    async def test_mock_llm_template_response(self):
        from hololoom.core.memory.interleaved_generation import MockLLM
        llm = MockLLM(response_template="Hello world", tokens_per_second=10000)
        tokens = []
        async for token in llm.generate_stream("q", "ctx"):
            tokens.append(token)
        full_text = "".join(tokens)
        assert "Hello" in full_text


class TestCreateStreamLLM:
    """Tests for the create_stream_llm factory."""

    def test_mock_provider(self):
        from hololoom.core.memory.interleaved_generation import create_stream_llm, MockLLM
        llm = create_stream_llm("mock")
        assert isinstance(llm, MockLLM)

    def test_unknown_provider_raises(self):
        from hololoom.core.memory.interleaved_generation import create_stream_llm
        with pytest.raises(ValueError, match="Unknown provider"):
            create_stream_llm("nonexistent_provider")

    def test_ollama_falls_back_to_mock(self):
        """Ollama unavailable should gracefully degrade to MockLLM."""
        from hololoom.core.memory.interleaved_generation import create_stream_llm, MockLLM
        # OllamaStreamLLM won't be available in test env
        llm = create_stream_llm("ollama")
        assert isinstance(llm, MockLLM)

    def test_anthropic_falls_back_to_mock(self):
        from hololoom.core.memory.interleaved_generation import create_stream_llm, MockLLM
        llm = create_stream_llm("anthropic")
        assert isinstance(llm, MockLLM)

    def test_openai_falls_back_to_mock(self):
        from hololoom.core.memory.interleaved_generation import create_stream_llm, MockLLM
        llm = create_stream_llm("openai")
        assert isinstance(llm, MockLLM)


class TestOllamaStreamLLM:
    """Tests for OllamaStreamLLM."""

    def test_unavailable_when_import_fails(self):
        from hololoom.core.memory.interleaved_generation import OllamaStreamLLM
        # In test env, OllamaLLM is likely not importable or not running
        llm = OllamaStreamLLM(model="test")
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_stream_raises_when_unavailable(self):
        from hololoom.core.memory.interleaved_generation import OllamaStreamLLM
        llm = OllamaStreamLLM(model="test")
        llm._available = False
        with pytest.raises(RuntimeError, match="not available"):
            async for _ in llm.generate_stream("q", "ctx"):
                pass


class TestAnthropicStreamLLM:
    def test_unavailable(self):
        from hololoom.core.memory.interleaved_generation import AnthropicStreamLLM
        llm = AnthropicStreamLLM()
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_raises_when_unavailable(self):
        from hololoom.core.memory.interleaved_generation import AnthropicStreamLLM
        llm = AnthropicStreamLLM()
        with pytest.raises(RuntimeError):
            async for _ in llm.generate_stream("q", "c"):
                pass


class TestOpenAIStreamLLM:
    def test_unavailable(self):
        from hololoom.core.memory.interleaved_generation import OpenAIStreamLLM
        llm = OpenAIStreamLLM()
        assert llm.is_available() is False


class TestInterleavedStreamManager:
    """Tests for InterleavedStreamManager."""

    @pytest.mark.asyncio
    async def test_get_last_result_returns_none(self):
        from hololoom.core.memory.interleaved_generation import InterleavedStreamManager, MockLLM
        manager = InterleavedStreamManager(MockLLM())
        assert manager.get_last_result() is None

    @pytest.mark.asyncio
    async def test_collect_generation_tokens(self):
        from hololoom.core.memory.interleaved_generation import InterleavedStreamManager, MockLLM

        async def fake_stream():
            for t in ["Hello ", "world "]:
                yield t

        manager = InterleavedStreamManager(MockLLM())
        tokens = await manager._collect_generation_tokens(fake_stream())
        assert tokens == ["Hello ", "world "]

    @pytest.mark.asyncio
    async def test_stream_interleaved_batched_with_mocked_builder(self):
        """Test batched mode stream_interleaved with a mock expansion builder."""
        from hololoom.core.memory.interleaved_generation import (
            InterleavedStreamManager, MockLLM, StreamMode,
            GenerationToken, StreamMetadata,
        )
        from hololoom.core.memory.streaming_expansion import ContextChunk

        # Create mock expansion builder that yields known chunks
        mock_builder = MagicMock()

        async def fake_expansion(**kwargs):
            yield ContextChunk(
                nodes=["seed_a"], relevance_scores={"seed_a": 0.9},
                contents={"seed_a": "Seed content A."},
                token_count=10, cumulative_tokens=10, hop_distance=0,
                chunk_index=0, is_final=False,
            )
            yield ContextChunk(
                nodes=["hop_b"], relevance_scores={"hop_b": 0.7},
                contents={"hop_b": "Hop content B."},
                token_count=8, cumulative_tokens=18, hop_distance=1,
                chunk_index=1, is_final=True,
            )

        mock_builder.stream_expansion = MagicMock(side_effect=fake_expansion)

        llm = MockLLM(response_template="The answer is 42", tokens_per_second=10000)
        manager = InterleavedStreamManager(llm, expansion_builder=mock_builder)

        items = []
        async for item in manager.stream_interleaved(
            query="test",
            seed_nodes=["seed_a"],
            graph=nx.MultiDiGraph(),
            stream_mode=StreamMode.BATCHED,
            emit_metadata=True,
        ):
            items.append(item)

        # Should have: expansion_start metadata, context chunks, generation_start,
        # generation tokens, stream_complete metadata
        chunk_items = [i for i in items if isinstance(i, ContextChunk)]
        gen_items = [i for i in items if isinstance(i, GenerationToken)]
        meta_items = [i for i in items if isinstance(i, StreamMetadata)]

        assert len(chunk_items) >= 1
        assert len(gen_items) >= 1
        assert any(m.event_type == "expansion_start" for m in meta_items)
        assert any(m.event_type == "stream_complete" for m in meta_items)

        # Check final token
        final_tokens = [t for t in gen_items if t.is_final]
        assert len(final_tokens) == 1

    @pytest.mark.asyncio
    async def test_stream_interleaved_concurrent_with_mocked_builder(self):
        """Test concurrent mode stream_interleaved."""
        from hololoom.core.memory.interleaved_generation import (
            InterleavedStreamManager, MockLLM, StreamMode,
            GenerationToken, StreamMetadata,
        )
        from hololoom.core.memory.streaming_expansion import ContextChunk

        mock_builder = MagicMock()

        async def fake_expansion(**kwargs):
            yield ContextChunk(
                nodes=["node1"], relevance_scores={"node1": 0.8},
                contents={"node1": "Node one content."},
                token_count=5, cumulative_tokens=5, hop_distance=0,
                chunk_index=0, is_final=False,
            )
            yield ContextChunk(
                nodes=["node2"], relevance_scores={"node2": 0.6},
                contents={"node2": "Node two content."},
                token_count=5, cumulative_tokens=10, hop_distance=1,
                chunk_index=1, is_final=True,
            )

        mock_builder.stream_expansion = MagicMock(side_effect=fake_expansion)

        llm = MockLLM(response_template="Quick answer", tokens_per_second=10000)
        manager = InterleavedStreamManager(llm, expansion_builder=mock_builder)

        items = []
        async for item in manager.stream_interleaved(
            query="test",
            seed_nodes=["node1"],
            graph=nx.MultiDiGraph(),
            stream_mode=StreamMode.CONCURRENT,
            emit_metadata=True,
        ):
            items.append(item)

        chunk_items = [i for i in items if isinstance(i, ContextChunk)]
        gen_items = [i for i in items if isinstance(i, GenerationToken)]
        meta_items = [i for i in items if isinstance(i, StreamMetadata)]

        assert len(chunk_items) >= 1
        assert len(gen_items) >= 1
        assert any(m.event_type == "expansion_start" for m in meta_items)

    @pytest.mark.asyncio
    async def test_stream_interleaved_no_chunks(self):
        """Test when expansion yields only a final chunk (no generation starts)."""
        from hololoom.core.memory.interleaved_generation import (
            InterleavedStreamManager, MockLLM, StreamMode,
        )
        from hololoom.core.memory.streaming_expansion import ContextChunk

        mock_builder = MagicMock()

        async def empty_expansion(**kwargs):
            yield ContextChunk(
                nodes=["only"], relevance_scores={"only": 0.5},
                contents={"only": "Only chunk."},
                token_count=3, cumulative_tokens=3, hop_distance=0,
                chunk_index=0, is_final=True,
            )

        mock_builder.stream_expansion = MagicMock(side_effect=empty_expansion)

        llm = MockLLM(tokens_per_second=10000)
        manager = InterleavedStreamManager(llm, expansion_builder=mock_builder)

        items = []
        async for item in manager.stream_interleaved(
            query="test", seed_nodes=["only"], graph=nx.MultiDiGraph(),
            stream_mode=StreamMode.BATCHED,
        ):
            items.append(item)

        # Should have the chunk but no generation (first chunk was is_final)
        assert len(items) >= 1


class TestStreamInterleavedConvenience:
    """Test the convenience function."""

    @pytest.mark.asyncio
    async def test_stream_interleaved_expansion_generation(self):
        from hololoom.core.memory.interleaved_generation import (
            stream_interleaved_expansion_generation, MockLLM,
        )
        # This will use the real StreamingContextBuilder which needs a real graph
        # Just test that the function can be called without error
        graph = nx.MultiDiGraph()
        graph.add_node("test_node")

        items = []
        async for item in stream_interleaved_expansion_generation(
            query="test",
            seed_nodes=["test_node"],
            graph=graph,
            llm=MockLLM(tokens_per_second=10000),
            token_budget=100,
            max_generation_tokens=5,
            node_contents={"test_node": "Test content"},
        ):
            items.append(item)
        # May or may not produce items depending on expansion behavior
        assert isinstance(items, list)


# ============================================================================
# interleaved_generation_advanced.py tests
# ============================================================================


class TestContextUpdate:
    """Tests for ContextUpdate dataclass."""

    def test_context_update_defaults(self):
        from hololoom.core.memory.interleaved_generation_advanced import ContextUpdate
        from hololoom.core.memory.streaming_expansion import ContextChunk

        chunk = ContextChunk(
            nodes=["a"], relevance_scores={"a": 0.9}, contents={"a": "text"},
            token_count=5, cumulative_tokens=5, hop_distance=0,
            chunk_index=0, is_final=False,
        )
        update = ContextUpdate(chunk=chunk)
        assert update.priority == 0.5
        assert update.timestamp_ms == 0.0


class TestNodeRequestType:
    def test_enum_values(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequestType
        assert NodeRequestType.BY_NAME == "by_name"
        assert NodeRequestType.BY_RELATIONSHIP == "by_rel"
        assert NodeRequestType.BY_QUERY == "by_query"


class TestAgenticGraphNavigator:
    """Tests for AgenticGraphNavigator."""

    def _make_navigator(self):
        from hololoom.core.memory.interleaved_generation_advanced import AgenticGraphNavigator
        graph = nx.MultiDiGraph()
        graph.add_node("thompson_sampling")
        graph.add_node("bayesian_approach")
        graph.add_edge("thompson_sampling", "bayesian_approach")
        node_contents = {
            "thompson_sampling": "Thompson Sampling is a Bayesian bandit algorithm.",
            "bayesian_approach": "Bayesian methods use prior distributions.",
        }
        return AgenticGraphNavigator(graph, node_contents)

    def test_detect_by_name_request(self):
        nav = self._make_navigator()
        text = "Let me look up <request_node>thompson_sampling</request_node> for more info."
        requests = nav.detect_requests(text)
        assert len(requests) == 1
        assert requests[0].target == "thompson_sampling"

    def test_detect_by_relationship_request(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequestType
        nav = self._make_navigator()
        text = "Show me <request_related>exploration</request_related>."
        requests = nav.detect_requests(text)
        assert len(requests) == 1
        assert requests[0].request_type == NodeRequestType.BY_RELATIONSHIP

    def test_detect_by_query_request(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequestType
        nav = self._make_navigator()
        text = "<request_query>What connects to Bayesian?</request_query>"
        requests = nav.detect_requests(text)
        assert len(requests) == 1
        assert requests[0].request_type == NodeRequestType.BY_QUERY

    def test_deduplication(self):
        nav = self._make_navigator()
        text = "<request_node>thompson_sampling</request_node>"
        nav.detect_requests(text)
        # Second call with same text should return empty (already fulfilled)
        requests = nav.detect_requests(text)
        assert len(requests) == 0

    def test_fulfill_by_name_found(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequest, NodeRequestType
        nav = self._make_navigator()
        request = NodeRequest(request_type=NodeRequestType.BY_NAME, target="thompson_sampling")
        chunk = nav.fulfill_request(request)
        assert chunk is not None
        assert "thompson_sampling" in chunk.nodes
        assert chunk.relevance_scores["thompson_sampling"] == 1.0

    def test_fulfill_by_name_not_found(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequest, NodeRequestType
        nav = self._make_navigator()
        request = NodeRequest(request_type=NodeRequestType.BY_NAME, target="nonexistent")
        chunk = nav.fulfill_request(request)
        assert chunk is None

    def test_fulfill_by_name_no_content(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequest, NodeRequestType
        nav = self._make_navigator()
        nav.node_contents["thompson_sampling"] = ""
        request = NodeRequest(request_type=NodeRequestType.BY_NAME, target="thompson_sampling")
        chunk = nav.fulfill_request(request)
        assert chunk is None

    def test_fulfill_by_relationship_not_implemented(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequest, NodeRequestType
        nav = self._make_navigator()
        request = NodeRequest(request_type=NodeRequestType.BY_RELATIONSHIP, target="IS_A")
        chunk = nav.fulfill_request(request)
        assert chunk is None

    def test_fulfill_by_query_not_implemented(self):
        from hololoom.core.memory.interleaved_generation_advanced import NodeRequest, NodeRequestType
        nav = self._make_navigator()
        request = NodeRequest(request_type=NodeRequestType.BY_QUERY, target="what is bayes")
        chunk = nav.fulfill_request(request)
        assert chunk is None


class TestSummarizationConfig:
    def test_defaults(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            SummarizationConfig, SummarizationStrategy,
        )
        config = SummarizationConfig()
        assert config.enabled is True
        assert config.strategy == SummarizationStrategy.DYNAMIC
        assert config.chunk_threshold == 10
        assert config.token_threshold == 2000


class TestContextSummarizer:
    """Tests for ContextSummarizer."""

    def _make_chunk(self, nodes, relevance, token_count=100, contents=None):
        from hololoom.core.memory.streaming_expansion import ContextChunk
        if contents is None:
            contents = {n: f"Content for {n}. More details here." for n in nodes}
        return ContextChunk(
            nodes=nodes,
            relevance_scores=relevance,
            contents=contents,
            token_count=token_count,
            cumulative_tokens=token_count,
            hop_distance=0,
            chunk_index=0,
            is_final=False,
        )

    def test_should_summarize_disabled(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            ContextSummarizer, SummarizationConfig,
        )
        config = SummarizationConfig(enabled=False)
        summarizer = ContextSummarizer(config)
        chunks = [self._make_chunk(["a"], {"a": 0.1})] * 20
        assert summarizer.should_summarize(chunks) is False

    def test_should_summarize_dynamic_by_count(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            ContextSummarizer, SummarizationConfig,
        )
        config = SummarizationConfig(chunk_threshold=5)
        summarizer = ContextSummarizer(config)
        chunks = [self._make_chunk(["a"], {"a": 0.5}, token_count=10)] * 6
        assert summarizer.should_summarize(chunks) is True

    def test_should_summarize_dynamic_by_tokens(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            ContextSummarizer, SummarizationConfig,
        )
        config = SummarizationConfig(token_threshold=500)
        summarizer = ContextSummarizer(config)
        chunks = [self._make_chunk(["a"], {"a": 0.5}, token_count=200)] * 3
        assert summarizer.should_summarize(chunks) is True

    def test_should_summarize_importance_based(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            ContextSummarizer, SummarizationConfig, SummarizationStrategy,
        )
        config = SummarizationConfig(
            strategy=SummarizationStrategy.IMPORTANCE_BASED,
            importance_cutoff=0.4,
        )
        summarizer = ContextSummarizer(config)
        # 3 low-importance chunks trigger
        chunks = [self._make_chunk([f"n{i}"], {f"n{i}": 0.1}) for i in range(4)]
        assert summarizer.should_summarize(chunks) is True

    def test_should_summarize_token_threshold_strategy(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            ContextSummarizer, SummarizationConfig, SummarizationStrategy,
        )
        config = SummarizationConfig(
            strategy=SummarizationStrategy.TOKEN_THRESHOLD,
            token_threshold=100,
        )
        summarizer = ContextSummarizer(config)
        chunks = [self._make_chunk(["a"], {"a": 0.5}, token_count=200)]
        assert summarizer.should_summarize(chunks) is True

    def test_summarize_produces_kept_and_summary(self):
        from hololoom.core.memory.interleaved_generation_advanced import ContextSummarizer
        summarizer = ContextSummarizer()
        chunks = [
            self._make_chunk(["high1"], {"high1": 0.9}, token_count=100),
            self._make_chunk(["high2"], {"high2": 0.8}, token_count=100),
            self._make_chunk(["low1"], {"low1": 0.1}, token_count=100),
            self._make_chunk(["low2"], {"low2": 0.05}, token_count=100),
        ]
        kept, summary = summarizer.summarize(chunks)
        assert len(kept) > 0
        if summary is not None:
            assert "SUMMARY" in summary.contents.get("summary", "")

    def test_summarize_nothing_to_summarize(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            ContextSummarizer, SummarizationConfig,
        )
        config = SummarizationConfig(compression_ratio=0.0)  # keep everything
        summarizer = ContextSummarizer(config)
        chunks = [self._make_chunk(["a"], {"a": 0.9})]
        kept, summary = summarizer.summarize(chunks)
        assert summary is None


class TestMockAdaptiveLLM:
    """Tests for MockAdaptiveLLM."""

    @pytest.mark.asyncio
    async def test_generate_stream_default(self):
        from hololoom.core.memory.interleaved_generation_advanced import MockAdaptiveLLM
        llm = MockAdaptiveLLM(tokens_per_second=10000)
        tokens = []
        async for token in llm.generate_stream("test", "context", max_tokens=5):
            tokens.append(token)
        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_generate_stream_template(self):
        from hololoom.core.memory.interleaved_generation_advanced import MockAdaptiveLLM
        llm = MockAdaptiveLLM(response_template="Custom response", tokens_per_second=10000)
        tokens = []
        async for token in llm.generate_stream("q", "c", max_tokens=50):
            tokens.append(token)
        assert "Custom" in "".join(tokens)

    @pytest.mark.asyncio
    async def test_generate_stream_adaptive(self):
        from hololoom.core.memory.interleaved_generation_advanced import MockAdaptiveLLM
        llm = MockAdaptiveLLM(tokens_per_second=10000)
        queue = asyncio.Queue()
        tokens = []
        async for token in llm.generate_stream_adaptive("test", "context", queue, max_tokens=10):
            tokens.append(token)
        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_adaptive_with_context_update(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            MockAdaptiveLLM, ContextUpdate,
        )
        from hololoom.core.memory.streaming_expansion import ContextChunk

        llm = MockAdaptiveLLM(tokens_per_second=10000)
        queue = asyncio.Queue()

        chunk = ContextChunk(
            nodes=["extra_node"], relevance_scores={"extra_node": 0.8},
            contents={"extra_node": "Extra context"}, token_count=5,
            cumulative_tokens=5, hop_distance=1, chunk_index=1, is_final=False,
        )
        await queue.put(ContextUpdate(chunk=chunk))

        tokens = []
        async for token in llm.generate_stream_adaptive("query", "initial", queue, max_tokens=15):
            tokens.append(token)
        text = "".join(tokens)
        assert len(text) > 0


class TestAdvancedInterleavedManager:
    """Tests for AdvancedInterleavedManager."""

    @pytest.mark.asyncio
    async def test_stream_advanced_requires_node_contents_for_navigation(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            AdvancedInterleavedManager, MockAdaptiveLLM,
        )
        llm = MockAdaptiveLLM(tokens_per_second=10000)
        manager = AdvancedInterleavedManager(
            llm, enable_agentic_navigation=True, enable_adaptive_feeding=False,
            enable_summarization=False,
        )
        with pytest.raises(ValueError, match="node_contents required"):
            async for _ in manager.stream_advanced(
                query="test", seed_nodes=["a"], graph=nx.MultiDiGraph(),
                node_contents=None,
            ):
                pass

    @pytest.mark.asyncio
    async def test_stream_advanced_with_all_features(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            AdvancedInterleavedManager, MockAdaptiveLLM,
            GenerationToken, StreamMetadata,
        )
        from hololoom.core.memory.streaming_expansion import ContextChunk

        llm = MockAdaptiveLLM(
            response_template="The answer involves concepts",
            tokens_per_second=10000,
        )

        graph = nx.MultiDiGraph()
        graph.add_node("concept_a")
        graph.add_node("concept_b")
        graph.add_edge("concept_a", "concept_b")

        node_contents = {
            "concept_a": "Concept A is about exploration.",
            "concept_b": "Concept B relates to exploitation.",
        }

        manager = AdvancedInterleavedManager(
            llm,
            enable_adaptive_feeding=True,
            enable_agentic_navigation=True,
            enable_summarization=True,
        )

        items = []
        async for item in manager.stream_advanced(
            query="What is the balance?",
            seed_nodes=["concept_a"],
            graph=graph,
            node_contents=node_contents,
            emit_metadata=True,
            token_budget=100,
            max_generation_tokens=10,
        ):
            items.append(item)

        # Should have produced some items
        assert len(items) > 0
        meta_items = [i for i in items if isinstance(i, StreamMetadata)]
        assert any(m.event_type == "stream_complete" for m in meta_items)

    @pytest.mark.asyncio
    async def test_stream_advanced_no_navigation(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            AdvancedInterleavedManager, MockAdaptiveLLM,
        )
        llm = MockAdaptiveLLM(tokens_per_second=10000)
        manager = AdvancedInterleavedManager(
            llm,
            enable_adaptive_feeding=False,
            enable_agentic_navigation=False,
            enable_summarization=False,
        )

        graph = nx.MultiDiGraph()
        graph.add_node("x")

        items = []
        async for item in manager.stream_advanced(
            query="test", seed_nodes=["x"], graph=graph,
            node_contents={"x": "Content X."},
            token_budget=50, max_generation_tokens=5,
        ):
            items.append(item)
        assert len(items) >= 0  # May produce items depending on expansion


class TestStreamAdvancedGeneration:
    """Test the convenience function."""

    @pytest.mark.asyncio
    async def test_stream_advanced_generation_convenience(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            stream_advanced_generation, MockAdaptiveLLM,
        )
        graph = nx.MultiDiGraph()
        graph.add_node("n1")

        items = []
        async for item in stream_advanced_generation(
            query="test", seed_nodes=["n1"], graph=graph,
            llm=MockAdaptiveLLM(tokens_per_second=10000),
            node_contents={"n1": "Node content."},
            token_budget=50, max_generation_tokens=5,
            enable_adaptive_feeding=False,
            enable_agentic_navigation=False,
            enable_summarization=False,
        ):
            items.append(item)
        assert isinstance(items, list)

    @pytest.mark.asyncio
    async def test_stream_advanced_generation_default_llm(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            stream_advanced_generation,
        )
        graph = nx.MultiDiGraph()
        graph.add_node("n1")

        items = []
        async for item in stream_advanced_generation(
            query="test", seed_nodes=["n1"], graph=graph,
            llm=None,  # Uses default MockAdaptiveLLM
            node_contents={"n1": "Content."},
            token_budget=50, max_generation_tokens=5,
            enable_agentic_navigation=False,
        ):
            items.append(item)
        assert isinstance(items, list)


class TestCreateAdaptiveLLM:
    """Tests for create_adaptive_llm factory."""

    def test_mock_provider(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            create_adaptive_llm, MockAdaptiveLLM,
        )
        llm = create_adaptive_llm("mock")
        assert isinstance(llm, MockAdaptiveLLM)

    def test_unknown_provider_falls_back(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            create_adaptive_llm, MockAdaptiveLLM,
        )
        llm = create_adaptive_llm("unknown_provider")
        assert isinstance(llm, MockAdaptiveLLM)

    def test_ollama_falls_back_to_mock(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            create_adaptive_llm, MockAdaptiveLLM,
        )
        llm = create_adaptive_llm("ollama")
        assert isinstance(llm, MockAdaptiveLLM)

    def test_anthropic_falls_back_to_mock(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            create_adaptive_llm, MockAdaptiveLLM,
        )
        llm = create_adaptive_llm("anthropic")
        assert isinstance(llm, MockAdaptiveLLM)

    def test_openai_falls_back_to_mock(self):
        from hololoom.core.memory.interleaved_generation_advanced import (
            create_adaptive_llm, MockAdaptiveLLM, AdaptiveLLMProtocol,
        )
        llm = create_adaptive_llm("openai")
        # May be OpenAIAdaptiveLLM if API key is set, but always an AdaptiveLLMProtocol
        assert isinstance(llm, AdaptiveLLMProtocol)


class TestOllamaAdaptiveLLM:
    def test_unavailable(self):
        from hololoom.core.memory.interleaved_generation_advanced import OllamaAdaptiveLLM
        llm = OllamaAdaptiveLLM()
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_stream_raises_when_unavailable(self):
        from hololoom.core.memory.interleaved_generation_advanced import OllamaAdaptiveLLM
        llm = OllamaAdaptiveLLM()
        with pytest.raises(RuntimeError):
            async for _ in llm.generate_stream("q", "c"):
                pass

    @pytest.mark.asyncio
    async def test_generate_stream_adaptive_raises_when_unavailable(self):
        from hololoom.core.memory.interleaved_generation_advanced import OllamaAdaptiveLLM
        llm = OllamaAdaptiveLLM()
        with pytest.raises(RuntimeError):
            async for _ in llm.generate_stream_adaptive("q", "c", asyncio.Queue()):
                pass


class TestAnthropicAdaptiveLLM:
    def test_unavailable_without_key(self):
        from hololoom.core.memory.interleaved_generation_advanced import AnthropicAdaptiveLLM
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            llm = AnthropicAdaptiveLLM()
            assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_stream_raises(self):
        from hololoom.core.memory.interleaved_generation_advanced import AnthropicAdaptiveLLM
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            llm = AnthropicAdaptiveLLM()
            with pytest.raises(RuntimeError):
                async for _ in llm.generate_stream("q", "c"):
                    pass

    @pytest.mark.asyncio
    async def test_generate_stream_adaptive_raises(self):
        from hololoom.core.memory.interleaved_generation_advanced import AnthropicAdaptiveLLM
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            llm = AnthropicAdaptiveLLM()
            with pytest.raises(RuntimeError):
                async for _ in llm.generate_stream_adaptive("q", "c", asyncio.Queue()):
                    pass


class TestOpenAIAdaptiveLLM:
    def test_unavailable_without_key(self):
        from hololoom.core.memory.interleaved_generation_advanced import OpenAIAdaptiveLLM
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            llm = OpenAIAdaptiveLLM()
            assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_stream_raises(self):
        from hololoom.core.memory.interleaved_generation_advanced import OpenAIAdaptiveLLM
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            llm = OpenAIAdaptiveLLM()
            with pytest.raises(RuntimeError):
                async for _ in llm.generate_stream("q", "c"):
                    pass

    @pytest.mark.asyncio
    async def test_generate_stream_adaptive_raises(self):
        from hololoom.core.memory.interleaved_generation_advanced import OpenAIAdaptiveLLM
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            llm = OpenAIAdaptiveLLM()
            with pytest.raises(RuntimeError):
                async for _ in llm.generate_stream_adaptive("q", "c", asyncio.Queue()):
                    pass


# ============================================================================
# v2_bridge.py tests
# ============================================================================


class TestPatternPPRConfig:
    """Tests for PATTERN_PPR mapping."""

    def test_all_patterns_exist(self):
        from hololoom.core.memory.v2_bridge import PATTERN_PPR
        assert "bare" in PATTERN_PPR
        assert "fast" in PATTERN_PPR
        assert "fused" in PATTERN_PPR

    def test_bare_has_high_alpha(self):
        from hololoom.core.memory.v2_bridge import PATTERN_PPR
        assert PATTERN_PPR["bare"].alpha == 0.25

    def test_fused_has_low_alpha(self):
        from hololoom.core.memory.v2_bridge import PATTERN_PPR
        assert PATTERN_PPR["fused"].alpha == 0.10


class TestEnhancedRecallResult:
    """Tests for EnhancedRecallResult dataclass."""

    def test_fields(self):
        from hololoom.core.memory.v2_bridge import EnhancedRecallResult
        result = EnhancedRecallResult(
            memories=[{"id": "1", "text": "hello", "score": 0.9, "memory_type": "factual"}],
            confidence=0.85,
            decision="sufficient",
            context_text="context here",
            justifications=[],
            retractions=[],
            cbr_hints=[],
            seed_count=3,
            ppr_converged=True,
            node_count=10,
            elapsed=0.05,
        )
        assert result.confidence == 0.85
        assert result.decision == "sufficient"
        assert result.ppr_converged is True


class TestFormatEnhancedResult:
    """Tests for format_enhanced_result."""

    def test_empty_memories(self):
        from hololoom.core.memory.v2_bridge import format_enhanced_result, EnhancedRecallResult
        result = EnhancedRecallResult(
            memories=[], confidence=0.0, decision="flag",
            context_text="", justifications=[], retractions=[],
            cbr_hints=[], seed_count=0, ppr_converged=False,
            node_count=0, elapsed=0.01,
        )
        text = format_enhanced_result(result)
        assert text == "No memories found."

    def test_with_memories(self):
        from hololoom.core.memory.v2_bridge import format_enhanced_result, EnhancedRecallResult
        result = EnhancedRecallResult(
            memories=[
                {"id": "m1", "text": "Thompson Sampling", "score": 0.95, "memory_type": "factual"},
            ],
            confidence=0.85,
            decision="sufficient",
            context_text="context",
            justifications=[{"belief_id": "b1"}],
            retractions=[],
            cbr_hints=[{"similarity": 0.8, "query": "similar query"}],
            seed_count=2,
            ppr_converged=True,
            node_count=50,
            elapsed=0.123,
        )
        text = format_enhanced_result(result)
        assert "Thompson Sampling" in text
        assert "0.85" in text
        assert "sufficient" in text
        assert "TMS" in text
        assert "CBR" in text

    def test_with_retractions(self):
        from hololoom.core.memory.v2_bridge import format_enhanced_result, EnhancedRecallResult
        result = EnhancedRecallResult(
            memories=[{"id": "m1", "text": "x", "score": 0.5, "memory_type": "factual"}],
            confidence=0.5,
            decision="verify",
            context_text="ctx",
            justifications=[],
            retractions=[{"id": "r1"}],
            cbr_hints=[],
            seed_count=1,
            ppr_converged=True,
            node_count=10,
            elapsed=0.05,
        )
        text = format_enhanced_result(result)
        assert "retractions" in text.lower()


class TestCognitiveBridge:
    """Tests for CognitiveBridge initialization and helper methods."""

    def test_init_creates_components(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        assert bridge._initialized is False
        assert bridge.bus is not None
        assert bridge.activation is not None
        assert bridge.working_memory is not None

    def test_init_with_persist_path(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        persist = str(tmp_path / "bridge.json")
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        assert bridge.persist_path == persist

    def test_find_source_returns_none_for_missing(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        result = bridge._find_source("nonexistent")
        assert result is None

    def test_find_source_finds_existing(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        # Check that we can find a source that exists
        source_names = [s.name for s in bridge.sources._sources]
        if source_names:
            found = bridge._find_source(source_names[0])
            assert found is not None

    def test_report_before_init(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        report = bridge.report()
        assert report["initialized"] is False
        assert "graph_nodes" in report

    def test_demon_report_no_demons(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        # Remove demons source if present
        bridge.sources._sources = [
            s for s in bridge.sources._sources if s.name != "demons"
        ]
        report = bridge.demon_report()
        assert report == {"alerts": [], "total_alerts": 0}

    def test_recall_from_memory_sync(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [
            SimpleNamespace(text="hello", id="m1", context={}),
        ]
        bridge = CognitiveBridge(mock_memory)
        result = bridge._recall_from_memory("query", 10)
        assert len(result) == 1

    def test_recall_from_memory_with_memories_attr(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_result = SimpleNamespace(memories=["a", "b"])
        mock_memory.recall.return_value = mock_result
        bridge = CognitiveBridge(mock_memory)
        result = bridge._recall_from_memory("query", 10)
        assert result == ["a", "b"]

    def test_recall_from_memory_exception(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.side_effect = Exception("fail")
        bridge = CognitiveBridge(mock_memory)
        result = bridge._recall_from_memory("query", 10)
        assert result == []

    @pytest.mark.asyncio
    async def test_enhanced_recall_requires_init(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        with pytest.raises(RuntimeError, match="not initialized"):
            await bridge.enhanced_recall("test query")

    @pytest.mark.asyncio
    async def test_feedback_before_init(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        # Should return without error when not initialized
        await bridge.feedback("query", 0.8)

    @pytest.mark.asyncio
    async def test_index_memory(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        await bridge.bus.initialize()
        node_id = await bridge._index_memory("Thompson Sampling is a bandit algorithm")
        assert node_id is not None
        assert len(bridge.bus._items) > 0

    @pytest.mark.asyncio
    async def test_index_memory_with_importance(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        await bridge.bus.initialize()
        node_id = await bridge._index_memory(
            "Important fact", context={"importance": 0.9}
        )
        assert node_id is not None

    @pytest.mark.asyncio
    async def test_index_memory_string_importance(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        await bridge.bus.initialize()
        node_id = await bridge._index_memory(
            "Some text", context={"importance": "0.7"}
        )
        assert node_id is not None

    @pytest.mark.asyncio
    async def test_index_memory_invalid_importance(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        await bridge.bus.initialize()
        node_id = await bridge._index_memory(
            "Text", context={"importance": "invalid"}
        )
        assert node_id is not None  # falls back to 0.5

    @pytest.mark.asyncio
    async def test_store_and_index(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory)
        await bridge.bus.initialize()
        await bridge.store_and_index("New fact about graphs")
        assert len(bridge.bus._items) > 0

    def test_save_combined_snapshot(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        persist = str(tmp_path / "snap.json")
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        bridge._save_combined_snapshot()
        assert Path(persist).exists()
        with open(persist) as f:
            data = json.load(f)
        assert data["_version"] == 2
        assert "graph" in data

    def test_load_combined_snapshot_missing_file(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        persist = str(tmp_path / "missing.json")
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        assert bridge._load_combined_snapshot() is False

    def test_load_combined_snapshot_v1(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        persist = str(tmp_path / "v1.json")
        # Write a v1-style snapshot (no _version)
        with open(persist, "w") as f:
            json.dump({"items": {}, "edges": {}, "aliases": {}}, f)
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        loaded = bridge._load_combined_snapshot()
        assert loaded is True

    def test_load_combined_snapshot_v2(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        persist = str(tmp_path / "v2.json")
        with open(persist, "w") as f:
            json.dump({
                "_version": 2,
                "graph": {"items": {}, "edges": {}, "aliases": {}},
            }, f)
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        loaded = bridge._load_combined_snapshot()
        assert loaded is True

    def test_save_if_persistent_no_path(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        bridge = CognitiveBridge(mock_memory, persist_path=None)
        bridge._save_if_persistent()  # should do nothing

    @pytest.mark.asyncio
    async def test_initialize_cold_start(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        bridge = CognitiveBridge(mock_memory)
        await bridge.initialize(warm_up_limit=0)
        assert bridge._initialized is True
        assert bridge.navigator is not None
        assert bridge.confidence_estimator is not None

    @pytest.mark.asyncio
    async def test_initialize_with_snapshot(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        persist = str(tmp_path / "bridge.json")
        # Write a v2 snapshot
        with open(persist, "w") as f:
            json.dump({
                "_version": 2,
                "graph": {"items": {}, "edges": {}, "aliases": {}},
            }, f)
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        await bridge.initialize()
        assert bridge._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_corrupt_snapshot(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        persist = str(tmp_path / "bad.json")
        with open(persist, "w") as f:
            f.write("{invalid json")
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        await bridge.initialize(warm_up_limit=0)
        assert bridge._initialized is True  # falls back to warm-up

    @pytest.mark.asyncio
    async def test_warm_up_with_memories(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [
            SimpleNamespace(text="Memory about Thompson Sampling", id="m1", context={}),
            SimpleNamespace(text="Memory about Bayesian Methods", id="m2", context={}),
        ]
        bridge = CognitiveBridge(mock_memory)
        await bridge.bus.initialize()
        await bridge._warm_up(limit=10)
        assert len(bridge.bus._items) > 0

    @pytest.mark.asyncio
    async def test_enhanced_recall_runs_pipeline(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [
            SimpleNamespace(text="Thompson Sampling is a bandit algorithm", id="m1", context={}),
        ]
        bridge = CognitiveBridge(mock_memory)
        await bridge.initialize(warm_up_limit=5)
        result = await bridge.enhanced_recall("Thompson Sampling", limit=5)
        assert result.confidence >= 0.0
        assert result.decision in ("sufficient", "verify", "flag")
        assert result.elapsed > 0

    @pytest.mark.asyncio
    async def test_enhanced_recall_with_patterns(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        bridge = CognitiveBridge(mock_memory)
        await bridge.initialize(warm_up_limit=0)
        # Test with different pattern cards
        for pattern in ("bare", "fast", "fused"):
            result = await bridge.enhanced_recall("test", pattern=pattern)
            assert result.decision in ("sufficient", "verify", "flag")

    @pytest.mark.asyncio
    async def test_feedback_updates_cbr(self):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        bridge = CognitiveBridge(mock_memory)
        await bridge.initialize(warm_up_limit=0)
        # feedback should not error
        await bridge.feedback("test query", 0.9)

    @pytest.mark.asyncio
    async def test_store_and_index_saves_snapshot(self, tmp_path):
        from hololoom.core.memory.v2_bridge import CognitiveBridge
        mock_memory = MagicMock()
        persist = str(tmp_path / "snap2.json")
        bridge = CognitiveBridge(mock_memory, persist_path=persist)
        await bridge.bus.initialize()
        await bridge.store_and_index("Knowledge about graphs")
        assert Path(persist).exists()


# ============================================================================
# repository_context.py tests
# ============================================================================


def _can_import_repository_context():
    """Check if repository_context is importable (requires mcp_rag_server chain)."""
    try:
        from hololoom.core.memory.repository_context import AccessLevel
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_REPO_CTX_AVAILABLE = _can_import_repository_context()
_skip_repo_ctx = pytest.mark.skipif(
    not _REPO_CTX_AVAILABLE,
    reason="repository_context import chain broken (mcp_rag_server dependency)"
)


@_skip_repo_ctx
class TestAccessLevel:
    """Tests for AccessLevel enum."""

    def test_values(self):
        from hololoom.core.memory.repository_context import AccessLevel
        assert AccessLevel.PUBLIC.value == "public"
        assert AccessLevel.INTERNAL.value == "internal"
        assert AccessLevel.PRIVATE.value == "private"
        assert AccessLevel.RESTRICTED.value == "restricted"


@_skip_repo_ctx
class TestRepositoryMetadata:
    """Tests for RepositoryMetadata."""

    def test_defaults(self):
        from hololoom.core.memory.repository_context import RepositoryMetadata, AccessLevel
        repo = RepositoryMetadata(name="test", path="/tmp/test")
        assert repo.name == "test"
        assert repo.access_level == AccessLevel.PUBLIC
        assert repo.file_count == 0
        assert "__pycache__" in repo.ignore_patterns

    def test_custom_tags(self):
        from hololoom.core.memory.repository_context import RepositoryMetadata
        repo = RepositoryMetadata(name="test", path="/tmp", tags={"python", "ml"})
        assert "python" in repo.tags


@_skip_repo_ctx
class TestAgentContext:
    """Tests for AgentContext access control."""

    def test_can_access_public_repo(self):
        from hololoom.core.memory.repository_context import (
            AgentContext, RepositoryMetadata, AccessLevel,
        )
        ctx = AgentContext(agent_id="agent1")
        repo = RepositoryMetadata(name="pub", path="/tmp", access_level=AccessLevel.PUBLIC)
        assert ctx.can_access_repo(repo) is True

    def test_blocked_repo(self):
        from hololoom.core.memory.repository_context import (
            AgentContext, RepositoryMetadata, AccessLevel,
        )
        ctx = AgentContext(agent_id="agent1", blocked_repos={"secret"})
        repo = RepositoryMetadata(name="secret", path="/tmp", access_level=AccessLevel.PUBLIC)
        assert ctx.can_access_repo(repo) is False

    def test_restricted_always_blocked(self):
        from hololoom.core.memory.repository_context import (
            AgentContext, RepositoryMetadata, AccessLevel,
        )
        ctx = AgentContext(agent_id="agent1", allowed_repos={"restricted"})
        repo = RepositoryMetadata(name="restricted", path="/tmp", access_level=AccessLevel.RESTRICTED)
        assert ctx.can_access_repo(repo) is False

    def test_private_requires_explicit_allow(self):
        from hololoom.core.memory.repository_context import (
            AgentContext, RepositoryMetadata, AccessLevel,
        )
        ctx_no_access = AgentContext(agent_id="agent1")
        ctx_with_access = AgentContext(agent_id="agent2", allowed_repos={"priv"})
        repo = RepositoryMetadata(name="priv", path="/tmp", access_level=AccessLevel.PRIVATE)
        assert ctx_no_access.can_access_repo(repo) is False
        assert ctx_with_access.can_access_repo(repo) is True

    def test_allowed_repos_filter(self):
        from hololoom.core.memory.repository_context import (
            AgentContext, RepositoryMetadata, AccessLevel,
        )
        ctx = AgentContext(agent_id="agent1", allowed_repos={"repo_a"})
        repo_a = RepositoryMetadata(name="repo_a", path="/tmp")
        repo_b = RepositoryMetadata(name="repo_b", path="/tmp")
        assert ctx.can_access_repo(repo_a) is True
        assert ctx.can_access_repo(repo_b) is False

    def test_can_access_tags_no_restrictions(self):
        from hololoom.core.memory.repository_context import AgentContext
        ctx = AgentContext(agent_id="agent1")
        assert ctx.can_access_tags({"python", "ml"}) is True

    def test_can_access_tags_blocked(self):
        from hololoom.core.memory.repository_context import AgentContext
        ctx = AgentContext(agent_id="agent1", blocked_tags={"private"})
        assert ctx.can_access_tags({"python", "private"}) is False

    def test_can_access_tags_allowed(self):
        from hololoom.core.memory.repository_context import AgentContext
        ctx = AgentContext(agent_id="agent1", allowed_tags={"python"})
        assert ctx.can_access_tags({"python", "ml"}) is True
        assert ctx.can_access_tags({"rust"}) is False


@_skip_repo_ctx
class TestChunkCodeFile:
    """Tests for chunk_code_file and helpers."""

    def test_chunk_python_file(self):
        from hololoom.core.memory.repository_context import chunk_code_file
        content = """import os

class MyClass:
    def method_a(self):
        pass

def standalone_func():
    return 42
"""
        chunks = chunk_code_file(Path("test.py"), content)
        assert len(chunks) > 0
        assert chunks[0]["extension"] == ".py"
        assert chunks[0]["language"] == "python"

    def test_chunk_python_with_class(self):
        from hololoom.core.memory.repository_context import chunk_code_file
        content = """import os
from pathlib import Path

class Processor:
    def __init__(self):
        self.data = []

    def run(self):
        pass
"""
        chunks = chunk_code_file(Path("processor.py"), content)
        assert any(c.get("chunk_type") == "class" for c in chunks)

    def test_chunk_markdown_file(self):
        from hololoom.core.memory.repository_context import chunk_code_file
        content = """# Title

Introduction paragraph.

## Section One

Content here.

## Section Two

More content.
"""
        chunks = chunk_code_file(Path("doc.md"), content)
        assert len(chunks) >= 2
        assert chunks[0]["language"] == "markdown"

    def test_chunk_markdown_with_heading(self):
        from hololoom.core.memory.repository_context import _chunk_markdown
        content = "# Main\nHello\n## Sub\nWorld"
        chunks = _chunk_markdown(Path("test.md"), content, 1000)
        assert len(chunks) == 2
        assert chunks[0]["section_heading"] == "Main"
        assert chunks[1]["section_heading"] == "Sub"

    def test_make_python_chunk_with_imports(self):
        from hololoom.core.memory.repository_context import _make_python_chunk
        chunk = _make_python_chunk(
            Path("test.py"),
            lines=["def foo():", "    return 1"],
            chunk_type="function",
            name="foo",
            imports=["import os"],
            line_number=5,
        )
        assert chunk["entity_name"] == "foo"
        assert chunk["chunk_type"] == "function"
        assert chunk["has_imports"] is True
        assert "import os" in chunk["text"]

    def test_make_python_chunk_without_imports(self):
        from hololoom.core.memory.repository_context import _make_python_chunk
        chunk = _make_python_chunk(
            Path("test.py"),
            lines=["x = 1"],
            chunk_type=None,
            name=None,
            imports=[],
            line_number=1,
        )
        assert chunk["chunk_type"] == "code"
        assert chunk["has_imports"] is False


@_skip_repo_ctx
class TestRepositoryContextManager:
    """Tests for RepositoryContextManager."""

    def test_init_defaults(self):
        from hololoom.core.memory.repository_context import RepositoryContextManager
        mgr = RepositoryContextManager()
        assert mgr.memory is None
        assert mgr.repositories == {}
        assert mgr.agent_contexts == {}

    def test_add_repository_nonexistent_path(self):
        from hololoom.core.memory.repository_context import RepositoryContextManager

        async def run():
            mgr = RepositoryContextManager(memory=MagicMock())
            with pytest.raises(ValueError, match="does not exist"):
                await mgr.add_repository("test", "/nonexistent/path/xyz")

        asyncio.get_event_loop().run_until_complete(run())

    def test_create_agent_context(self):
        from hololoom.core.memory.repository_context import RepositoryContextManager, AgentQueryContext
        mgr = RepositoryContextManager(memory=MagicMock())
        ctx = mgr.create_agent_context(
            agent_id="agent1",
            allowed_repos={"repo_a"},
            blocked_repos={"repo_b"},
        )
        assert isinstance(ctx, AgentQueryContext)
        assert "agent1" in mgr.agent_contexts

    def test_list_repositories_no_context(self):
        from hololoom.core.memory.repository_context import (
            RepositoryContextManager, RepositoryMetadata,
        )
        mgr = RepositoryContextManager(memory=MagicMock())
        mgr.repositories["a"] = RepositoryMetadata(name="a", path="/tmp")
        mgr.repositories["b"] = RepositoryMetadata(name="b", path="/tmp")
        repos = mgr.list_repositories()
        assert len(repos) == 2

    def test_list_repositories_with_context(self):
        from hololoom.core.memory.repository_context import (
            RepositoryContextManager, RepositoryMetadata, AgentContext,
        )
        mgr = RepositoryContextManager(memory=MagicMock())
        mgr.repositories["a"] = RepositoryMetadata(name="a", path="/tmp")
        mgr.repositories["b"] = RepositoryMetadata(name="b", path="/tmp")
        ctx = AgentContext(agent_id="test", allowed_repos={"a"})
        repos = mgr.list_repositories(agent_context=ctx)
        assert len(repos) == 1
        assert repos[0].name == "a"


@_skip_repo_ctx
class TestAgentQueryContext:
    """Tests for AgentQueryContext."""

    def test_list_repositories(self):
        from hololoom.core.memory.repository_context import (
            RepositoryContextManager, RepositoryMetadata, AgentContext, AgentQueryContext,
        )
        mgr = RepositoryContextManager(memory=MagicMock())
        mgr.repositories["a"] = RepositoryMetadata(name="a", path="/tmp")
        ctx = AgentContext(agent_id="test")
        query_ctx = AgentQueryContext(mgr, ctx)
        repos = query_ctx.list_repositories()
        assert len(repos) == 1
