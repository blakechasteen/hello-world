"""
Test Conductor Memory System Integrations
==========================================

Tests for the 4 memory system integrations in MemoryConductor:
1. HotPatternTracker - Usage-based boosting
2. AwarenessGraph - Spreading activation
3. SpringDynamics - Physics-based connectivity
4. MultiWaveEngine - Temporal propagation

Author: Claude Code
Date: 2025-12-03
"""

import pytest
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock, AsyncMock

# Import conductor and protocols (moved to hololoom/memory/symphony/ in Dec 2025)
from hololoom.core.memory.symphony.conductor import MemoryConductor
from hololoom.core.memory.symphony.protocol import (
    MemoryQuery,
    MemoryResult,
    MemoryStrategy,
    MemorySystem,
)


# ============================================================================
# Mock Classes for Testing
# ============================================================================

@dataclass
class MockUsageRecord:
    """Mock of HotPatternTracker's UsageRecord."""
    element_id: str
    element_type: str = "thread"
    access_count: int = 10
    success_count: int = 8
    avg_confidence: float = 0.85

    @property
    def heat_score(self) -> float:
        success_rate = self.success_count / max(self.access_count, 1)
        return self.access_count * success_rate * self.avg_confidence

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.access_count, 1)


class MockHotPatternTracker:
    """Mock of HotPatternTracker for testing."""

    def __init__(self, records: List[MockUsageRecord] = None):
        self._records = records or [
            MockUsageRecord("node_1", "thread", 20, 18, 0.9),
            MockUsageRecord("node_2", "shard", 15, 12, 0.85),
            MockUsageRecord("node_3", "motif", 10, 7, 0.8),
        ]

    def get_hot_patterns(
        self,
        element_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[MockUsageRecord]:
        records = self._records
        if element_type:
            records = [r for r in records if r.element_type == element_type]
        # Sort by heat score descending
        records = sorted(records, key=lambda r: r.heat_score, reverse=True)
        return records[:top_k]


@dataclass
class MockMemory:
    """Mock Memory object from AwarenessGraph."""
    id: str
    text: str
    activation: float = 0.5
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class MockSemanticPerception:
    """Mock SemanticPerception from AwarenessGraph."""
    query: str
    embedding: List[float] = None


class MockAwarenessGraph:
    """Mock of AwarenessGraph for testing."""

    def __init__(self, memories: List[MockMemory] = None):
        self._memories = memories or [
            MockMemory("aware_1", "Memory about Thompson Sampling", 0.9),
            MockMemory("aware_2", "Memory about exploration", 0.75),
            MockMemory("aware_3", "Memory about bandits", 0.6),
        ]

    def perceive(self, content: str) -> MockSemanticPerception:
        return MockSemanticPerception(query=content)

    def activate(self, perception: MockSemanticPerception) -> List[MockMemory]:
        # Return memories sorted by activation
        return sorted(self._memories, key=lambda m: m.activation, reverse=True)


class MockAsyncAwarenessGraph(MockAwarenessGraph):
    """Async version of MockAwarenessGraph."""

    async def activate(self, perception: MockSemanticPerception) -> List[MockMemory]:
        return sorted(self._memories, key=lambda m: m.activation, reverse=True)


@dataclass
class MockSpringPropagationResult:
    """Mock of SpringPropagationResult."""
    iterations: int = 25
    converged: bool = True
    final_energy: float = 0.001
    activated_nodes: List[str] = None
    node_activations: Dict[str, float] = None

    def __post_init__(self):
        if self.activated_nodes is None:
            self.activated_nodes = ["spring_1", "spring_2", "spring_3"]
        if self.node_activations is None:
            self.node_activations = {
                "spring_1": 0.95,
                "spring_2": 0.8,
                "spring_3": 0.65,
            }


class MockSpringDynamics:
    """Mock of SpringDynamics for testing."""

    def __init__(self):
        self._activations: Dict[str, float] = {}
        self._result = MockSpringPropagationResult()

    def activate_nodes(self, activations: Dict[str, float]):
        self._activations = activations.copy()

    def propagate(self) -> MockSpringPropagationResult:
        return self._result


@dataclass
class MockBetaWaveRecallResult:
    """Mock of BetaWaveRecallResult from MultiWaveEngine."""
    recalled_memories: List[tuple] = None  # (node_id, activation)
    all_activations: Dict[str, float] = None
    creative_insights: List[tuple] = None
    seed_nodes: List[tuple] = None

    def __post_init__(self):
        if self.recalled_memories is None:
            self.recalled_memories = [
                ("wave_1", 0.92),
                ("wave_2", 0.78),
                ("wave_3", 0.65),
            ]
        if self.all_activations is None:
            self.all_activations = {}
        if self.creative_insights is None:
            self.creative_insights = []
        if self.seed_nodes is None:
            self.seed_nodes = []


class MockMultiWaveEngine:
    """Mock of MultiWaveEngine for testing."""

    def __init__(self):
        self._result = MockBetaWaveRecallResult()

    def retrieve_memories(
        self,
        query_embedding,
        top_k: int = 10,
        **kwargs
    ) -> MockBetaWaveRecallResult:
        return self._result

    def on_query(self, query_embedding) -> MockBetaWaveRecallResult:
        return self._result


class MockMemoryBackend:
    """Mock memory backend for testing."""

    def __init__(self):
        self.graph = MockGraph()
        self._vectors = {}

    def embed(self, text: str):
        import numpy as np
        return np.random.randn(384)


class MockGraph:
    """Mock graph with node data."""

    def __init__(self):
        import networkx as nx
        self._graph = nx.MultiDiGraph()
        # Add some test nodes
        self._graph.add_node("node_1", content="Content of node 1")
        self._graph.add_node("node_2", text="Text of node 2")
        self._graph.add_node("node_3", data="Data of node 3")


# ============================================================================
# Tests for _get_memory_content
# ============================================================================

class TestGetMemoryContent:
    """Tests for _get_memory_content helper."""

    def test_content_from_graph_content_field(self):
        """Test content retrieval from graph 'content' field."""
        backend = MockMemoryBackend()
        conductor = MemoryConductor(backend, verbose=True)

        content = conductor._get_memory_content("node_1")
        assert content == "Content of node 1"

    def test_content_from_graph_text_field(self):
        """Test content retrieval from graph 'text' field."""
        backend = MockMemoryBackend()
        conductor = MemoryConductor(backend, verbose=True)

        content = conductor._get_memory_content("node_2")
        assert content == "Text of node 2"

    def test_content_from_graph_data_field(self):
        """Test content retrieval from graph 'data' field."""
        backend = MockMemoryBackend()
        conductor = MemoryConductor(backend, verbose=True)

        content = conductor._get_memory_content("node_3")
        assert content == "Data of node 3"

    def test_fallback_to_node_id(self):
        """Test fallback to node_id when no content found."""
        backend = MockMemoryBackend()
        conductor = MemoryConductor(backend, verbose=True)

        content = conductor._get_memory_content("unknown_node")
        assert content == "unknown_node"

    def test_no_graph_fallback(self):
        """Test fallback when backend has no graph."""
        backend = Mock(spec=[])  # Empty spec means no attributes
        conductor = MemoryConductor(backend, verbose=True)

        content = conductor._get_memory_content("any_node")
        assert content == "any_node"


# ============================================================================
# Tests for _query_hot_patterns
# ============================================================================

class TestQueryHotPatterns:
    """Tests for hot pattern integration."""

    @pytest.mark.asyncio
    async def test_hot_patterns_returns_results(self):
        """Test that hot patterns returns results when tracker provided."""
        backend = MockMemoryBackend()
        tracker = MockHotPatternTracker()

        conductor = MemoryConductor(
            backend,
            hot_tracker=tracker,
            verbose=True
        )

        query = MemoryQuery(text="Test query", k=5)
        results = await conductor._query_hot_patterns(query)

        assert len(results) == 3
        assert all(isinstance(r, MemoryResult) for r in results)
        assert all(r.source_system == MemorySystem.HOT_PATTERNS for r in results)

    @pytest.mark.asyncio
    async def test_hot_patterns_empty_without_tracker(self):
        """Test that hot patterns returns empty without tracker."""
        backend = MockMemoryBackend()

        conductor = MemoryConductor(backend, verbose=True)

        query = MemoryQuery(text="Test query", k=5)
        results = await conductor._query_hot_patterns(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_hot_patterns_heat_score_in_metadata(self):
        """Test that heat score is captured in metadata."""
        backend = MockMemoryBackend()
        tracker = MockHotPatternTracker()

        conductor = MemoryConductor(
            backend,
            hot_tracker=tracker,
            verbose=True
        )

        query = MemoryQuery(text="Test query", k=5)
        results = await conductor._query_hot_patterns(query)

        assert len(results) > 0
        # First result should have highest heat
        first = results[0]
        assert first.heat > 0
        assert "access_count" in first.metadata
        assert "success_rate" in first.metadata

    @pytest.mark.asyncio
    async def test_hot_patterns_respects_k(self):
        """Test that hot patterns respects k limit."""
        backend = MockMemoryBackend()
        records = [MockUsageRecord(f"node_{i}") for i in range(20)]
        tracker = MockHotPatternTracker(records)

        conductor = MemoryConductor(
            backend,
            hot_tracker=tracker,
            verbose=True
        )

        query = MemoryQuery(text="Test query", k=5)
        results = await conductor._query_hot_patterns(query)

        assert len(results) <= 5


# ============================================================================
# Tests for _query_awareness_graph
# ============================================================================

class TestQueryAwarenessGraph:
    """Tests for awareness graph integration."""

    @pytest.mark.asyncio
    async def test_awareness_returns_results(self):
        """Test that awareness graph returns results."""
        backend = MockMemoryBackend()
        awareness = MockAwarenessGraph()

        conductor = MemoryConductor(
            backend,
            awareness_graph=awareness,
            verbose=True
        )

        query = MemoryQuery(text="Thompson Sampling", k=5)
        results = await conductor._query_awareness_graph(query)

        assert len(results) == 3
        assert all(isinstance(r, MemoryResult) for r in results)
        assert all(r.source_system == MemorySystem.AWARENESS_GRAPH for r in results)

    @pytest.mark.asyncio
    async def test_awareness_async_activate(self):
        """Test awareness graph with async activate method."""
        backend = MockMemoryBackend()
        awareness = MockAsyncAwarenessGraph()

        conductor = MemoryConductor(
            backend,
            awareness_graph=awareness,
            verbose=True
        )

        query = MemoryQuery(text="Thompson Sampling", k=5)
        results = await conductor._query_awareness_graph(query)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_awareness_empty_without_graph(self):
        """Test awareness returns empty without graph."""
        backend = MockMemoryBackend()

        conductor = MemoryConductor(backend, verbose=True)

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_awareness_graph(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_awareness_activation_as_relevance(self):
        """Test that activation level becomes relevance."""
        backend = MockMemoryBackend()
        memories = [MockMemory("m1", "Test", activation=0.85)]
        awareness = MockAwarenessGraph(memories)

        conductor = MemoryConductor(
            backend,
            awareness_graph=awareness,
            verbose=True
        )

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_awareness_graph(query)

        assert len(results) == 1
        assert results[0].relevance == 0.85
        assert results[0].activation == 0.85


# ============================================================================
# Tests for _query_spring_dynamics
# ============================================================================

class TestQuerySpringDynamics:
    """Tests for spring dynamics integration."""

    @pytest.mark.asyncio
    async def test_spring_returns_results(self):
        """Test that spring dynamics returns results."""
        backend = MockMemoryBackend()
        spring = MockSpringDynamics()

        # Mock vector memory to provide seed nodes
        conductor = MemoryConductor(
            backend,
            spring_dynamics=spring,
            verbose=True
        )

        # Mock _query_vector_memory to return seed nodes
        conductor._query_vector_memory = AsyncMock(return_value=[
            MemoryResult("seed_1", "content", 0.9, MemorySystem.VECTOR_MEMORY),
            MemoryResult("seed_2", "content", 0.8, MemorySystem.VECTOR_MEMORY),
        ])

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_spring_dynamics(query)

        assert len(results) == 3
        assert all(isinstance(r, MemoryResult) for r in results)
        assert all(r.source_system == MemorySystem.SPRING_DYNAMICS for r in results)

    @pytest.mark.asyncio
    async def test_spring_empty_without_dynamics(self):
        """Test spring returns empty without dynamics instance."""
        backend = MockMemoryBackend()

        conductor = MemoryConductor(backend, verbose=True)

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_spring_dynamics(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_spring_metadata_includes_physics(self):
        """Test that physics metadata is included."""
        backend = MockMemoryBackend()
        spring = MockSpringDynamics()

        conductor = MemoryConductor(
            backend,
            spring_dynamics=spring,
            verbose=True
        )

        conductor._query_vector_memory = AsyncMock(return_value=[
            MemoryResult("seed_1", "content", 0.9, MemorySystem.VECTOR_MEMORY),
        ])

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_spring_dynamics(query)

        assert len(results) > 0
        assert results[0].metadata.get("physics_based") is True
        assert "converged" in results[0].metadata
        assert "iterations" in results[0].metadata


# ============================================================================
# Tests for _query_multi_wave
# ============================================================================

class TestQueryMultiWave:
    """Tests for multi-wave engine integration."""

    @pytest.mark.asyncio
    async def test_wave_returns_results(self):
        """Test that multi-wave returns results."""
        backend = MockMemoryBackend()
        wave = MockMultiWaveEngine()

        conductor = MemoryConductor(
            backend,
            multi_wave_engine=wave,
            verbose=True
        )

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_multi_wave(query)

        assert len(results) == 3
        assert all(isinstance(r, MemoryResult) for r in results)
        assert all(r.source_system == MemorySystem.MULTI_WAVE for r in results)

    @pytest.mark.asyncio
    async def test_wave_empty_without_engine(self):
        """Test wave returns empty without engine."""
        backend = MockMemoryBackend()

        conductor = MemoryConductor(backend, verbose=True)

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_multi_wave(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_wave_activations_parsed(self):
        """Test that tuple activations are parsed correctly."""
        backend = MockMemoryBackend()
        wave = MockMultiWaveEngine()

        conductor = MemoryConductor(
            backend,
            multi_wave_engine=wave,
            verbose=True
        )

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_multi_wave(query)

        # First result should have activation from tuple
        assert results[0].relevance == 0.92
        assert results[0].activation == 0.92

    @pytest.mark.asyncio
    async def test_wave_metadata_includes_insights(self):
        """Test that creative insights count is in metadata."""
        backend = MockMemoryBackend()
        wave = MockMultiWaveEngine()

        conductor = MemoryConductor(
            backend,
            multi_wave_engine=wave,
            verbose=True
        )

        query = MemoryQuery(text="Test", k=5)
        results = await conductor._query_multi_wave(query)

        assert len(results) > 0
        assert results[0].metadata.get("wave_based") is True
        assert "creative_insights" in results[0].metadata


# ============================================================================
# Integration Tests
# ============================================================================

class TestConductorWithAllSystems:
    """Integration tests with all memory systems enabled."""

    @pytest.mark.asyncio
    async def test_all_systems_integrated(self):
        """Test conductor with all 4 systems integrated."""
        backend = MockMemoryBackend()

        conductor = MemoryConductor(
            backend,
            hot_tracker=MockHotPatternTracker(),
            awareness_graph=MockAwarenessGraph(),
            spring_dynamics=MockSpringDynamics(),
            multi_wave_engine=MockMultiWaveEngine(),
            enable_hot_patterns=True,
            enable_awareness=True,
            enable_spring_dynamics=True,
            enable_multi_wave=True,
            verbose=True
        )

        # Mock vector memory for spring dynamics
        conductor._query_vector_memory = AsyncMock(return_value=[
            MemoryResult("seed", "content", 0.9, MemorySystem.VECTOR_MEMORY),
        ])

        query = MemoryQuery(text="Test all systems", k=5)

        # Test each system individually
        hot_results = await conductor._query_hot_patterns(query)
        awareness_results = await conductor._query_awareness_graph(query)
        spring_results = await conductor._query_spring_dynamics(query)
        wave_results = await conductor._query_multi_wave(query)

        # All should return results
        assert len(hot_results) > 0
        assert len(awareness_results) > 0
        assert len(spring_results) > 0
        assert len(wave_results) > 0

        # Each should have correct source system
        assert all(r.source_system == MemorySystem.HOT_PATTERNS for r in hot_results)
        assert all(r.source_system == MemorySystem.AWARENESS_GRAPH for r in awareness_results)
        assert all(r.source_system == MemorySystem.SPRING_DYNAMICS for r in spring_results)
        assert all(r.source_system == MemorySystem.MULTI_WAVE for r in wave_results)

    @pytest.mark.asyncio
    async def test_graceful_failure_handling(self):
        """Test that errors don't crash the system."""
        backend = MockMemoryBackend()

        # Create a tracker that raises an error
        bad_tracker = Mock()
        bad_tracker.get_hot_patterns = Mock(side_effect=Exception("Test error"))

        conductor = MemoryConductor(
            backend,
            hot_tracker=bad_tracker,
            verbose=True
        )

        query = MemoryQuery(text="Test", k=5)

        # Should return empty list, not raise
        results = await conductor._query_hot_patterns(query)
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
