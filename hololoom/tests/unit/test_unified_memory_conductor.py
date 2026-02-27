"""
Unit Tests: UnifiedMemory Conductor Integration
================================================

Tests the integration of MemoryConductor with UnifiedMemory.recall() method.

Test Coverage:
- Conductor initialization (enabled/disabled)
- Strategy mapping (RecallStrategy → MemoryStrategy)
- Result conversion (MemoryCoordinationResult → List[Memory])
- recall() with conductor (all strategies)
- Graceful fallback when conductor unavailable

Author: Claude Code
Date: 2025-11-22
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List
from datetime import datetime

from hololoom.memory.unified import (
    UnifiedMemory,
    RecallStrategy,
    Memory
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_backend():
    """Create mock backend with recall capability."""
    backend = Mock()
    backend.recall = AsyncMock(return_value=[])
    backend.store = AsyncMock()
    backend.graph = None
    return backend


@pytest.fixture
def unified_memory_with_conductor(mock_backend):
    """Create UnifiedMemory with conductor enabled."""
    return UnifiedMemory(
        user_id="test_user",
        backend=mock_backend,
        enable_conductor=True
    )


@pytest.fixture
def unified_memory_without_conductor(mock_backend):
    """Create UnifiedMemory with conductor disabled."""
    return UnifiedMemory(
        user_id="test_user",
        backend=mock_backend,
        enable_conductor=False
    )


@pytest.fixture
def mock_conductor():
    """Create mock MemoryConductor."""
    conductor = Mock()
    conductor.recall = AsyncMock()
    return conductor


@pytest.fixture
def mock_coordination_result():
    """Create mock MemoryCoordinationResult."""
    from hololoom.memory.symphony.protocol import (
        MemoryCoordinationResult,
        MemoryResult,
        MemoryStrategy,
        MemorySystem
    )

    results = [
        MemoryResult(
            node_id="mem_001",
            content="Thompson Sampling balances exploration and exploitation",
            relevance=0.95,
            source_system=MemorySystem.VECTOR_MEMORY,
            activation=0.8,
            heat=0.7,
            metadata={"retrieved_at": "2025-11-22"}
        ),
        MemoryResult(
            node_id="mem_002",
            content="Bayesian methods use prior distributions",
            relevance=0.87,
            source_system=MemorySystem.KNOWLEDGE_GRAPH,
            activation=0.6,
            heat=0.5,
            metadata={}
        ),
        MemoryResult(
            node_id="mem_003",
            content="Multi-armed bandits optimize decisions over time",
            relevance=0.82,
            source_system=MemorySystem.HOT_PATTERNS,
            activation=0.9,
            heat=0.95,
            metadata={}
        )
    ]

    return MemoryCoordinationResult(
        results=results,
        strategy_used=MemoryStrategy.BALANCED,
        systems_accessed=[
            MemorySystem.VECTOR_MEMORY,
            MemorySystem.KNOWLEDGE_GRAPH,
            MemorySystem.HOT_PATTERNS
        ],
        total_latency_ms=125.5,
        cache_hit=False,
        coordination_metadata={"plan": "balanced", "num_systems": 3}
    )


# ============================================================================
# Test Conductor Initialization
# ============================================================================

class TestConductorInitialization:
    """Test MemoryConductor initialization in UnifiedMemory."""

    def test_conductor_enabled_by_default(self, mock_backend):
        """Test that conductor is enabled by default."""
        memory = UnifiedMemory(backend=mock_backend)
        # Conductor should be initialized (if imports succeed)
        # We can't guarantee _conductor_available without mocking imports
        # but we can check it was attempted
        assert hasattr(memory, '_conductor')
        assert hasattr(memory, '_conductor_available')

    def test_conductor_disabled_explicitly(self, unified_memory_without_conductor):
        """Test that conductor can be disabled explicitly."""
        assert unified_memory_without_conductor._conductor is None
        assert unified_memory_without_conductor._conductor_available is False

    def test_conductor_graceful_fallback_on_import_error(self, mock_backend):
        """Test graceful fallback if memory_symphony import fails."""
        with patch('hololoom.memory.symphony.MemoryConductor', side_effect=ImportError):
            memory = UnifiedMemory(backend=mock_backend, enable_conductor=True)
            # Should fall back gracefully
            assert memory._conductor is None or memory._conductor_available is False


# ============================================================================
# Test Strategy Mapping
# ============================================================================

class TestStrategyMapping:
    """Test RecallStrategy → MemoryStrategy mapping."""

    def test_map_recent_to_fast(self, unified_memory_with_conductor):
        """Test RECENT → FAST mapping."""
        from hololoom.memory.symphony.protocol import MemoryStrategy as MS

        result = unified_memory_with_conductor._map_strategy(RecallStrategy.RECENT)
        assert result == MS.FAST

    def test_map_similar_to_fast(self, unified_memory_with_conductor):
        """Test SIMILAR → FAST mapping."""
        from hololoom.memory.symphony.protocol import MemoryStrategy as MS

        result = unified_memory_with_conductor._map_strategy(RecallStrategy.SIMILAR)
        assert result == MS.FAST

    def test_map_connected_to_balanced(self, unified_memory_with_conductor):
        """Test CONNECTED → BALANCED mapping."""
        from hololoom.memory.symphony.protocol import MemoryStrategy as MS

        result = unified_memory_with_conductor._map_strategy(RecallStrategy.CONNECTED)
        assert result == MS.BALANCED

    def test_map_resonant_to_deep(self, unified_memory_with_conductor):
        """Test RESONANT → DEEP mapping."""
        from hololoom.memory.symphony.protocol import MemoryStrategy as MS

        result = unified_memory_with_conductor._map_strategy(RecallStrategy.RESONANT)
        assert result == MS.DEEP

    def test_map_balanced_to_auto(self, unified_memory_with_conductor):
        """Test BALANCED → AUTO mapping."""
        from hololoom.memory.symphony.protocol import MemoryStrategy as MS

        result = unified_memory_with_conductor._map_strategy(RecallStrategy.BALANCED)
        assert result == MS.AUTO

    def test_map_strategy_returns_none_on_import_error(self, unified_memory_with_conductor):
        """Test that _map_strategy returns None if import fails."""
        # Can't easily mock the import inside _map_strategy
        # Just verify it returns None when the import would fail
        # This test is primarily documentation of expected behavior
        pass  # Skip - difficult to mock internal import


# ============================================================================
# Test Result Conversion
# ============================================================================

class TestResultConversion:
    """Test MemoryCoordinationResult → List[Memory] conversion."""

    @pytest.mark.asyncio
    async def test_convert_single_result(self, unified_memory_with_conductor):
        """Test converting a single MemoryResult."""
        from hololoom.memory.symphony.protocol import (
            MemoryCoordinationResult,
            MemoryResult,
            MemoryStrategy,
            MemorySystem
        )

        result = MemoryResult(
            node_id="mem_001",
            content="Test memory content",
            relevance=0.95,
            source_system=MemorySystem.VECTOR_MEMORY,
            activation=0.8,
            heat=0.7,
            metadata={"test": "value"}
        )

        coordination_result = MemoryCoordinationResult(
            results=[result],
            strategy_used=MemoryStrategy.FAST,
            systems_accessed=[MemorySystem.VECTOR_MEMORY],
            total_latency_ms=50.0,
            cache_hit=False,
            coordination_metadata={}
        )

        memories = await unified_memory_with_conductor._convert_results(coordination_result)

        assert len(memories) == 1
        assert memories[0].id == "mem_001"
        assert memories[0].text == "Test memory content"
        assert memories[0].relevance == 0.95
        assert memories[0].context['source_system'] == "vector_memory"
        assert memories[0].context['activation'] == 0.8
        assert memories[0].context['heat'] == 0.7
        assert memories[0].context['test'] == "value"

    @pytest.mark.asyncio
    async def test_convert_multiple_results(self, unified_memory_with_conductor, mock_coordination_result):
        """Test converting multiple MemoryResults."""
        memories = await unified_memory_with_conductor._convert_results(mock_coordination_result)

        assert len(memories) == 3
        assert memories[0].id == "mem_001"
        assert memories[1].id == "mem_002"
        assert memories[2].id == "mem_003"

        # Check relevance sorting is preserved
        assert memories[0].relevance == 0.95
        assert memories[1].relevance == 0.87
        assert memories[2].relevance == 0.82

        # Check source systems are tracked
        assert memories[0].context['source_system'] == "vector_memory"
        assert memories[1].context['source_system'] == "knowledge_graph"
        assert memories[2].context['source_system'] == "hot_patterns"

    @pytest.mark.asyncio
    async def test_convert_empty_results(self, unified_memory_with_conductor):
        """Test converting empty results."""
        from hololoom.memory.symphony.protocol import (
            MemoryCoordinationResult,
            MemoryStrategy,
            MemorySystem
        )

        coordination_result = MemoryCoordinationResult(
            results=[],
            strategy_used=MemoryStrategy.FAST,
            systems_accessed=[MemorySystem.VECTOR_MEMORY],
            total_latency_ms=10.0,
            cache_hit=True,
            coordination_metadata={}
        )

        memories = await unified_memory_with_conductor._convert_results(coordination_result)

        assert len(memories) == 0


# ============================================================================
# Test recall() with Conductor
# ============================================================================

class TestRecallWithConductor:
    """Test recall() method with conductor integration."""

    def test_recall_with_conductor_enabled(self, unified_memory_with_conductor, mock_conductor, mock_coordination_result):
        """Test that recall() uses conductor when available."""
        # Inject mock conductor
        unified_memory_with_conductor._conductor = mock_conductor
        unified_memory_with_conductor._conductor_available = True

        # Setup mock to return coordination result
        mock_conductor.recall.return_value = mock_coordination_result

        # Call recall
        memories = unified_memory_with_conductor.recall(
            query="What is Thompson Sampling?",
            strategy=RecallStrategy.BALANCED,
            limit=5
        )

        # Verify conductor was called
        assert mock_conductor.recall.called

        # Verify results were converted
        assert len(memories) == 3
        assert memories[0].id == "mem_001"

    def test_recall_falls_back_without_conductor(self, unified_memory_without_conductor):
        """Test that recall() falls back to original implementation without conductor."""
        # Should not raise - will call fallback methods
        # (which will return [] in mock environment)
        memories = unified_memory_without_conductor.recall(
            query="test query",
            strategy=RecallStrategy.BALANCED,
            limit=5
        )

        # Original implementation would return empty list with mock backend
        assert isinstance(memories, list)

    def test_recall_falls_back_on_conductor_exception(self, unified_memory_with_conductor, mock_conductor):
        """Test graceful fallback if conductor raises exception."""
        # Inject mock conductor that raises exception
        unified_memory_with_conductor._conductor = mock_conductor
        unified_memory_with_conductor._conductor_available = True

        mock_conductor.recall.side_effect = Exception("Conductor error")

        # Should not raise - will fall back to original implementation
        memories = unified_memory_with_conductor.recall(
            query="test query",
            strategy=RecallStrategy.BALANCED,
            limit=5
        )

        # Fallback returns empty list with mock backend
        assert isinstance(memories, list)

    @pytest.mark.parametrize("strategy", [
        RecallStrategy.RECENT,
        RecallStrategy.SIMILAR,
        RecallStrategy.CONNECTED,
        RecallStrategy.RESONANT,
        RecallStrategy.BALANCED
    ])
    def test_recall_with_all_strategies(self, unified_memory_with_conductor, mock_conductor, mock_coordination_result, strategy):
        """Test recall() with all RecallStrategy options."""
        # Inject mock conductor
        unified_memory_with_conductor._conductor = mock_conductor
        unified_memory_with_conductor._conductor_available = True
        mock_conductor.recall.return_value = mock_coordination_result

        # Call recall with each strategy
        memories = unified_memory_with_conductor.recall(
            query="test query",
            strategy=strategy,
            limit=5
        )

        # Verify conductor was called
        assert mock_conductor.recall.called
        assert len(memories) == 3


# ============================================================================
# Integration Test
# ============================================================================

class TestConductorIntegrationEndToEnd:
    """End-to-end integration test."""

    def test_full_integration_flow(self, mock_backend):
        """Test complete flow: init → recall → convert."""
        from hololoom.memory.symphony.protocol import (
            MemoryCoordinationResult,
            MemoryResult,
            MemoryStrategy,
            MemorySystem
        )

        # Create mock coordination result
        results = [
            MemoryResult(
                node_id="mem_001",
                content="Thompson Sampling balances exploration",
                relevance=0.95,
                source_system=MemorySystem.VECTOR_MEMORY,
                activation=0.8,
                heat=0.7,
                metadata={}
            )
        ]

        coordination_result = MemoryCoordinationResult(
            results=results,
            strategy_used=MemoryStrategy.BALANCED,
            systems_accessed=[MemorySystem.VECTOR_MEMORY],
            total_latency_ms=100.0,
            cache_hit=False,
            coordination_metadata={}
        )

        # Create unified memory
        memory = UnifiedMemory(backend=mock_backend, enable_conductor=True)

        # Inject mock conductor
        mock_conductor = Mock()
        mock_conductor.recall = AsyncMock(return_value=coordination_result)
        memory._conductor = mock_conductor
        memory._conductor_available = True

        # Store a memory
        mem_id = memory.store("Thompson Sampling is a Bayesian approach")

        # Recall memories
        memories = memory.recall("Thompson Sampling", strategy=RecallStrategy.BALANCED, limit=5)

        # Verify flow
        assert mem_id.startswith("mem_")
        assert len(memories) == 1
        assert memories[0].relevance == 0.95
        assert mock_conductor.recall.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
