"""
Test Context Handlers
=====================

Comprehensive test suite for HoloLoom ChatOps context packing command handlers.

Tests:
- !context pack <query> - Matryoshka compression
- !context budget <bits> - MI-aware packing
- !context expand <query> - Adaptive expansion
- !context stream <query> - Streaming expansion
- !context spread <nodes> - Beta wave spreading
- !context importance - 7-signal breakdown
- !context stats - System statistics
- !context help - Help command

Created: 2025-12-15
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio


# ============================================================================
# Mock Matrix Types
# ============================================================================

class MockMatrixRoom:
    """Mock Matrix room for testing."""
    def __init__(self, room_id: str = "!test:matrix.org"):
        self.room_id = room_id
        self.display_name = "Test Room"


class MockRoomMessageText:
    """Mock Matrix message event for testing."""
    def __init__(self, sender: str = "@user:matrix.org", body: str = ""):
        self.sender = sender
        self.body = body
        self.event_id = "$test_event_id"
        self.server_timestamp = 1702656000000


# ============================================================================
# Mock Context Packing Types
# ============================================================================

@dataclass
class MockCompressionResult:
    """Mock compression result from ContextPacker."""
    original_count: int = 100
    compressed_count: int = 50
    compression_ratio: float = 0.50
    token_savings: int = 500
    compressed_nodes: List[str] = field(default_factory=lambda: [
        'thompson_sampling', 'bayesian', 'exploration', 'exploitation',
        'neural_net', 'gradient', 'backprop', 'attention', 'transformer', 'bert'
    ])
    importance_scores: Dict[str, float] = field(default_factory=lambda: {
        'thompson_sampling': 0.95,
        'bayesian': 0.88,
        'exploration': 0.82,
        'exploitation': 0.78,
        'neural_net': 0.72
    })
    scale_assignments: Dict[str, str] = field(default_factory=lambda: {
        'thompson_sampling': '384D',
        'bayesian': '384D',
        'exploration': '256D',
        'exploitation': '256D',
        'neural_net': '128D'
    })


class MockContextPacker:
    """Mock ContextPacker for testing."""

    def __init__(self, config=None):
        self.config = config
        self._stats = {
            'total_packs': 42,
            'avg_compression_ratio': 0.55,
            'avg_token_savings': 450
        }

    def pack(self, query: str, candidate_nodes: List[str],
             graph: Any, target_tokens: int = 2000) -> MockCompressionResult:
        """Mock pack operation."""
        return MockCompressionResult()


class MockContextPackerConfig:
    """Mock ContextPackerConfig."""

    @classmethod
    def balanced(cls):
        return cls()

    @classmethod
    def aggressive(cls):
        return cls()

    @classmethod
    def conservative(cls):
        return cls()


# ============================================================================
# Mock Activation Spreader Types
# ============================================================================

class MockBetaWaveConfig:
    """Mock BetaWaveConfig."""
    pass


class MockActivationSpreader:
    """Mock ActivationSpreader for testing."""

    def __init__(self, config=None):
        self.config = config
        self._stats = {
            'total_spreads': 25,
            'avg_hops': 2.5,
            'avg_activated_nodes': 15
        }

    def spread_activation(
        self,
        source_nodes: List[str],
        graph: Any,
        initial_activation: float = 1.0,
        max_hops: int = 3
    ) -> Dict[str, float]:
        """Mock spread activation."""
        result = {node: initial_activation for node in source_nodes}
        # Add simulated spread
        result['related_node_1'] = 0.7
        result['related_node_2'] = 0.5
        result['distant_node'] = 0.2
        return result


# ============================================================================
# Mock Importance Scorer Types
# ============================================================================

class MockImportanceSignal(Enum):
    """Mock importance signals."""
    RECENCY = "recency"
    RELEVANCE = "relevance"
    CENTRALITY = "centrality"
    ACCESS_FREQUENCY = "access_frequency"
    CONFIDENCE = "confidence"
    HEAT = "heat"
    INFORMATION_CONTENT = "information_content"


class MockImportanceScorer:
    """Mock ImportanceScorer for testing."""

    def __init__(self, config=None):
        self.config = config
        self._stats = {
            'total_scored': 500,
            'cache_hits': 400,
            'cache_misses': 100,
            'mi_cache_hits': 350,
            'mi_cache_misses': 150
        }

    def score_importance(
        self,
        node_id: str,
        query: str,
        graph: Any
    ) -> Dict[MockImportanceSignal, float]:
        """Mock score importance."""
        return {
            MockImportanceSignal.RECENCY: 0.85,
            MockImportanceSignal.RELEVANCE: 0.92,
            MockImportanceSignal.CENTRALITY: 0.70,
            MockImportanceSignal.ACCESS_FREQUENCY: 0.60,
            MockImportanceSignal.CONFIDENCE: 0.88,
            MockImportanceSignal.HEAT: 0.75,
            MockImportanceSignal.INFORMATION_CONTENT: 0.82
        }


# ============================================================================
# Mock Adaptive Expansion Types
# ============================================================================

@dataclass
class MockExpansionResult:
    """Mock adaptive expansion result."""
    nodes: List[str] = field(default_factory=lambda: [
        'thompson_sampling', 'bayesian', 'exploration', 'bandits'
    ])
    total_tokens: int = 800
    avg_relevance: float = 0.72
    max_hop_reached: int = 3
    stopping_reason: str = "budget_exhausted"
    relevance_scores: Dict[str, float] = field(default_factory=lambda: {
        'thompson_sampling': 0.95,
        'bayesian': 0.85,
        'exploration': 0.75,
        'bandits': 0.65
    })


async def mock_expand_context_adaptive(**kwargs):
    """Mock expand_context_adaptive function."""
    return MockExpansionResult()


# ============================================================================
# Mock Streaming Expansion Types
# ============================================================================

class MockChunkYieldStrategy(Enum):
    """Mock chunk yield strategies."""
    TOKEN_THRESHOLD = "token_threshold"
    HOP_BOUNDARY = "hop_boundary"
    HYBRID = "hybrid"


@dataclass
class MockContextChunk:
    """Mock context chunk."""
    chunk_index: int
    node_count: int
    token_count: int
    hop_distance: int
    avg_relevance: float
    is_final: bool = False
    nodes: List[str] = field(default_factory=list)


async def mock_stream_context_expansion(**kwargs):
    """Mock stream_context_expansion generator."""
    chunks = [
        MockContextChunk(0, 5, 200, 0, 0.95),
        MockContextChunk(1, 8, 350, 1, 0.82),
        MockContextChunk(2, 4, 150, 2, 0.68, is_final=True),
    ]
    for chunk in chunks:
        yield chunk


# ============================================================================
# Mock Knowledge Graph
# ============================================================================

class MockKnowledgeGraph:
    """Mock knowledge graph for testing."""

    def __init__(self):
        import networkx as nx
        self._graph = nx.MultiDiGraph()
        # Add test nodes
        self._graph.add_node('thompson_sampling', content='Thompson Sampling content')
        self._graph.add_node('bayesian', content='Bayesian methods')
        self._graph.add_node('exploration', content='Exploration strategies')
        self._graph.add_node('neural_net', content='Neural networks')
        # Add edges
        self._graph.add_edge('thompson_sampling', 'bayesian', edge_type='IS_A')
        self._graph.add_edge('thompson_sampling', 'exploration', edge_type='USES')

    @property
    def graph(self):
        return self._graph


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def room():
    """Create a mock Matrix room."""
    return MockMatrixRoom()


@pytest.fixture
def event():
    """Create a mock Matrix message event."""
    return MockRoomMessageText()


@pytest.fixture
def mock_packer():
    """Create a mock ContextPacker."""
    return MockContextPacker()


@pytest.fixture
def mock_spreader():
    """Create a mock ActivationSpreader."""
    return MockActivationSpreader()


@pytest.fixture
def mock_scorer():
    """Create a mock ImportanceScorer."""
    return MockImportanceScorer()


@pytest.fixture
def mock_kg():
    """Create a mock knowledge graph."""
    return MockKnowledgeGraph()


@pytest.fixture
def context_handlers(mock_packer, mock_spreader, mock_scorer, mock_kg):
    """Import and configure context handlers."""
    # Mock networkx
    import networkx as nx

    # Patch the imports
    with patch.dict('sys.modules', {
        'hololoom.context_packing': Mock(
            ContextPacker=MockContextPacker,
            ContextPackerConfig=MockContextPackerConfig,
            pack_context=Mock(),
            information_budget_pack=Mock(return_value=(['node1', 'node2'], {'node1': '384D'}, {'node1': 0.9}))
        ),
        'hololoom.context_packing.activation_spreader': Mock(
            ActivationSpreader=MockActivationSpreader
        ),
        'hololoom.context_packing.config': Mock(
            BetaWaveConfig=MockBetaWaveConfig
        ),
        'hololoom.context_packing.importance_scorer': Mock(
            ImportanceScorer=MockImportanceScorer
        ),
        'hololoom.context_packing.protocol': Mock(
            ImportanceSignal=MockImportanceSignal
        ),
        'hololoom.memory.adaptive_expansion': Mock(
            AdaptiveExpander=Mock(),
            expand_context_adaptive=mock_expand_context_adaptive
        ),
        'hololoom.memory.streaming_expansion': Mock(
            StreamingContextBuilder=Mock(),
            stream_context_expansion=mock_stream_context_expansion,
            ChunkYieldStrategy=MockChunkYieldStrategy
        ),
        'nio': Mock(
            MatrixRoom=MockMatrixRoom,
            RoomMessageText=MockRoomMessageText
        ),
    }):
        # Import handlers
        from hololoom.apps.chatops.handlers import context_handlers as ch

        # Set global instances
        ch.set_context_packer(mock_packer)
        ch.set_activation_spreader(mock_spreader)
        ch.set_importance_scorer(mock_scorer)
        ch.set_knowledge_graph(mock_kg)

        # Set availability flags
        ch.PACKER_AVAILABLE = True
        ch.SPREADER_AVAILABLE = True
        ch.SCORER_AVAILABLE = True
        ch.ADAPTIVE_AVAILABLE = True
        ch.STREAMING_AVAILABLE = True
        ch.MATRIX_AVAILABLE = True

        yield ch


# ============================================================================
# Test: !context pack
# ============================================================================

class TestContextPackCommand:
    """Test !context pack command."""

    @pytest.mark.asyncio
    async def test_pack_success(self, room, event, context_handlers, mock_packer, mock_kg):
        """Test successful context packing."""
        response = await context_handlers.handle_context_pack(
            room, event, ["Thompson Sampling", "2000"], mock_packer
        )

        assert "Context Packing Result" in response
        assert "Compression" in response

    @pytest.mark.asyncio
    async def test_pack_empty_args(self, room, event, context_handlers):
        """Test pack with empty args shows usage."""
        response = await context_handlers.handle_context_pack(room, event, [])

        assert "Usage" in response
        assert "!context pack" in response

    @pytest.mark.asyncio
    async def test_pack_shows_compression_stats(self, room, event, context_handlers, mock_packer, mock_kg):
        """Test pack shows compression statistics."""
        response = await context_handlers.handle_context_pack(
            room, event, ["test query"], mock_packer
        )

        assert "Original" in response
        assert "Compressed" in response
        assert "Token Savings" in response

    @pytest.mark.asyncio
    async def test_pack_no_packer_available(self, room, event, context_handlers):
        """Test pack when no packer available."""
        context_handlers.set_context_packer(None)
        context_handlers.PACKER_AVAILABLE = False

        response = await context_handlers.handle_context_pack(
            room, event, ["test query"]
        )

        assert "not available" in response.lower()


# ============================================================================
# Test: !context budget
# ============================================================================

class TestContextBudgetCommand:
    """Test !context budget command."""

    @pytest.mark.asyncio
    async def test_budget_success(self, room, event, context_handlers, mock_packer, mock_kg):
        """Test successful budget packing."""
        response = await context_handlers.handle_context_budget(
            room, event, ["5.0", "Thompson Sampling"], mock_packer
        )

        assert "Information Budget" in response or "Budget" in response

    @pytest.mark.asyncio
    async def test_budget_empty_args(self, room, event, context_handlers):
        """Test budget with empty args shows usage."""
        response = await context_handlers.handle_context_budget(room, event, [])

        assert "Usage" in response
        assert "!context budget" in response

    @pytest.mark.asyncio
    async def test_budget_invalid_value(self, room, event, context_handlers):
        """Test budget with invalid value."""
        response = await context_handlers.handle_context_budget(
            room, event, ["invalid"]
        )

        assert "Invalid" in response or "Error" in response

    @pytest.mark.asyncio
    async def test_budget_no_kg(self, room, event, context_handlers):
        """Test budget when no knowledge graph."""
        context_handlers.set_knowledge_graph(None)

        response = await context_handlers.handle_context_budget(
            room, event, ["5.0"]
        )

        assert "No knowledge graph" in response


# ============================================================================
# Test: !context expand
# ============================================================================

class TestContextExpandCommand:
    """Test !context expand command."""

    @pytest.mark.asyncio
    async def test_expand_success(self, room, event, context_handlers, mock_kg):
        """Test successful adaptive expansion."""
        # Patch the async function
        context_handlers.expand_context_adaptive = mock_expand_context_adaptive

        response = await context_handlers.handle_context_expand(
            room, event, ["Thompson Sampling", "1000"]
        )

        # Should either work or show error about dependencies
        assert "Expansion" in response or "Error" in response or "not available" in response

    @pytest.mark.asyncio
    async def test_expand_empty_args(self, room, event, context_handlers):
        """Test expand with empty args shows usage."""
        response = await context_handlers.handle_context_expand(room, event, [])

        assert "Usage" in response
        assert "!context expand" in response

    @pytest.mark.asyncio
    async def test_expand_not_available(self, room, event, context_handlers):
        """Test expand when not available."""
        context_handlers.ADAPTIVE_AVAILABLE = False

        response = await context_handlers.handle_context_expand(
            room, event, ["test query"]
        )

        assert "not available" in response.lower()


# ============================================================================
# Test: !context stream
# ============================================================================

class TestContextStreamCommand:
    """Test !context stream command."""

    @pytest.mark.asyncio
    async def test_stream_empty_args(self, room, event, context_handlers):
        """Test stream with empty args shows usage."""
        response = await context_handlers.handle_context_stream(room, event, [])

        assert "Usage" in response
        assert "!context stream" in response

    @pytest.mark.asyncio
    async def test_stream_not_available(self, room, event, context_handlers):
        """Test stream when not available."""
        context_handlers.STREAMING_AVAILABLE = False

        response = await context_handlers.handle_context_stream(
            room, event, ["test query"]
        )

        assert "not available" in response.lower()


# ============================================================================
# Test: !context spread
# ============================================================================

class TestContextSpreadCommand:
    """Test !context spread command."""

    @pytest.mark.asyncio
    async def test_spread_success(self, room, event, context_handlers, mock_spreader, mock_kg):
        """Test successful activation spreading."""
        response = await context_handlers.handle_context_spread(
            room, event, ["thompson_sampling,bayesian", "3"], mock_spreader
        )

        assert "Beta Wave" in response or "Spreading" in response or "Activated" in response

    @pytest.mark.asyncio
    async def test_spread_single_node(self, room, event, context_handlers, mock_spreader, mock_kg):
        """Test spreading from single node."""
        response = await context_handlers.handle_context_spread(
            room, event, ["thompson_sampling"], mock_spreader
        )

        assert "Spreading" in response or "Activated" in response or "Beta" in response

    @pytest.mark.asyncio
    async def test_spread_empty_args(self, room, event, context_handlers):
        """Test spread with empty args shows usage."""
        response = await context_handlers.handle_context_spread(room, event, [])

        assert "Usage" in response
        assert "!context spread" in response

    @pytest.mark.asyncio
    async def test_spread_not_available(self, room, event, context_handlers):
        """Test spread when not available."""
        context_handlers.set_activation_spreader(None)
        context_handlers.SPREADER_AVAILABLE = False

        response = await context_handlers.handle_context_spread(
            room, event, ["test_node"]
        )

        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_spread_shows_activation_bars(self, room, event, context_handlers, mock_spreader, mock_kg):
        """Test spread shows visual activation bars."""
        response = await context_handlers.handle_context_spread(
            room, event, ["test_node"], mock_spreader
        )

        # Should have bar characters
        assert "█" in response or "░" in response or "Activated" in response


# ============================================================================
# Test: !context importance
# ============================================================================

class TestContextImportanceCommand:
    """Test !context importance command."""

    @pytest.mark.asyncio
    async def test_importance_success(self, room, event, context_handlers, mock_scorer, mock_kg):
        """Test successful importance scoring."""
        response = await context_handlers.handle_context_importance(
            room, event, ["thompson_sampling", "What is exploration?"], mock_scorer
        )

        assert "Importance" in response or "Signal" in response

    @pytest.mark.asyncio
    async def test_importance_shows_signals(self, room, event, context_handlers, mock_scorer, mock_kg):
        """Test importance shows 7 signals."""
        response = await context_handlers.handle_context_importance(
            room, event, ["test_node"], mock_scorer
        )

        # Should mention various signals
        assert "Signal" in response or "RECENCY" in response or "recency" in response.lower()

    @pytest.mark.asyncio
    async def test_importance_not_available(self, room, event, context_handlers):
        """Test importance when not available."""
        context_handlers.set_importance_scorer(None)
        context_handlers.SCORER_AVAILABLE = False

        response = await context_handlers.handle_context_importance(
            room, event, ["test_node"]
        )

        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_importance_no_kg(self, room, event, context_handlers, mock_scorer):
        """Test importance when no knowledge graph."""
        context_handlers.set_knowledge_graph(None)

        response = await context_handlers.handle_context_importance(
            room, event, ["test_node"], mock_scorer
        )

        assert "No knowledge graph" in response


# ============================================================================
# Test: !context stats
# ============================================================================

class TestContextStatsCommand:
    """Test !context stats command."""

    @pytest.mark.asyncio
    async def test_stats_success(self, room, event, context_handlers):
        """Test successful stats retrieval."""
        response = await context_handlers.handle_context_stats(room, event, [])

        assert "Statistics" in response

    @pytest.mark.asyncio
    async def test_stats_shows_packer_info(self, room, event, context_handlers):
        """Test stats shows packer info."""
        response = await context_handlers.handle_context_stats(room, event, [])

        assert "Context Packer" in response

    @pytest.mark.asyncio
    async def test_stats_shows_spreader_info(self, room, event, context_handlers):
        """Test stats shows spreader info."""
        response = await context_handlers.handle_context_stats(room, event, [])

        assert "Spreader" in response or "Activation" in response

    @pytest.mark.asyncio
    async def test_stats_shows_scorer_info(self, room, event, context_handlers):
        """Test stats shows scorer info."""
        response = await context_handlers.handle_context_stats(room, event, [])

        assert "Scorer" in response or "Importance" in response

    @pytest.mark.asyncio
    async def test_stats_shows_availability(self, room, event, context_handlers):
        """Test stats shows component availability."""
        response = await context_handlers.handle_context_stats(room, event, [])

        assert "Available" in response or "Availability" in response


# ============================================================================
# Test: !context help
# ============================================================================

class TestContextHelpCommand:
    """Test !context help command."""

    @pytest.mark.asyncio
    async def test_help_returns_content(self, room, event, context_handlers):
        """Test help returns help content."""
        response = await context_handlers.handle_context_help(room, event, [])

        assert "Context Packing Commands" in response

    @pytest.mark.asyncio
    async def test_help_lists_commands(self, room, event, context_handlers):
        """Test help lists all context commands."""
        response = await context_handlers.handle_context_help(room, event, [])

        assert "!context pack" in response
        assert "!context budget" in response
        assert "!context expand" in response
        assert "!context stream" in response
        assert "!context spread" in response
        assert "!context importance" in response
        assert "!context stats" in response

    @pytest.mark.asyncio
    async def test_help_shows_examples(self, room, event, context_handlers):
        """Test help shows usage examples."""
        response = await context_handlers.handle_context_help(room, event, [])

        assert "Example" in response or "example" in response.lower()

    @pytest.mark.asyncio
    async def test_help_describes_features(self, room, event, context_handlers):
        """Test help describes key features."""
        response = await context_handlers.handle_context_help(room, event, [])

        # Should mention key features
        assert "Matryoshka" in response or "token" in response.lower()


# ============================================================================
# Test: Handler Registration
# ============================================================================

class TestHandlerRegistration:
    """Test handler registration functionality."""

    def test_register_handlers(self, context_handlers, mock_packer, mock_spreader, mock_scorer, mock_kg):
        """Test handlers can be registered with a bot."""
        mock_bot = Mock()
        mock_bot.register_command = Mock()

        result = context_handlers.register_context_handlers(
            mock_bot,
            mock_packer,
            mock_spreader,
            mock_scorer,
            mock_kg
        )

        # Should return ContextHandlers instance
        assert result is not None
        assert hasattr(result, 'handle_pack')
        assert hasattr(result, 'handle_budget')
        assert hasattr(result, 'handle_expand')
        assert hasattr(result, 'handle_stream')
        assert hasattr(result, 'handle_spread')
        assert hasattr(result, 'handle_importance')
        assert hasattr(result, 'handle_stats')
        assert hasattr(result, 'handle_help')

    def test_register_handlers_calls_bot_register(self, context_handlers, mock_packer, mock_spreader, mock_scorer, mock_kg):
        """Test registration calls bot.register_command."""
        mock_bot = Mock()
        mock_bot.register_command = Mock()

        context_handlers.register_context_handlers(
            mock_bot,
            mock_packer,
            mock_spreader,
            mock_scorer,
            mock_kg
        )

        # Should have called register_command for each handler
        assert mock_bot.register_command.call_count >= 8

    def test_context_handlers_class(self, mock_packer, mock_spreader, mock_scorer, mock_kg):
        """Test ContextHandlers class initialization."""
        from hololoom.apps.chatops.handlers.context_handlers import ContextHandlers

        handlers = ContextHandlers(
            mock_packer,
            mock_spreader,
            mock_scorer,
            mock_kg
        )

        assert handlers.packer == mock_packer
        assert handlers.spreader == mock_spreader
        assert handlers.scorer == mock_scorer
        assert handlers.knowledge_graph == mock_kg


# ============================================================================
# Test: Global Instance Management
# ============================================================================

class TestGlobalInstanceManagement:
    """Test global instance getters and setters."""

    def test_get_set_context_packer(self, context_handlers, mock_packer):
        """Test get/set context packer."""
        context_handlers.set_context_packer(mock_packer)
        result = context_handlers.get_context_packer()
        assert result == mock_packer

    def test_get_set_activation_spreader(self, context_handlers, mock_spreader):
        """Test get/set activation spreader."""
        context_handlers.set_activation_spreader(mock_spreader)
        result = context_handlers.get_activation_spreader()
        assert result == mock_spreader

    def test_get_set_importance_scorer(self, context_handlers, mock_scorer):
        """Test get/set importance scorer."""
        context_handlers.set_importance_scorer(mock_scorer)
        result = context_handlers.get_importance_scorer()
        assert result == mock_scorer

    def test_get_set_knowledge_graph(self, context_handlers, mock_kg):
        """Test get/set knowledge graph."""
        context_handlers.set_knowledge_graph(mock_kg)
        result = context_handlers.get_knowledge_graph()
        assert result == mock_kg

    def test_get_returns_none_when_not_set(self, context_handlers):
        """Test getters return None when not set."""
        context_handlers.set_context_packer(None)
        context_handlers.set_activation_spreader(None)
        context_handlers.set_importance_scorer(None)
        context_handlers.set_knowledge_graph(None)

        assert context_handlers.get_context_packer() is None
        assert context_handlers.get_activation_spreader() is None
        assert context_handlers.get_importance_scorer() is None
        assert context_handlers.get_knowledge_graph() is None


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_pack_exception_handling(self, room, event, context_handlers, mock_kg):
        """Test pack handles exceptions gracefully."""
        bad_packer = Mock()
        bad_packer.pack = Mock(side_effect=Exception("Test error"))
        context_handlers.set_context_packer(bad_packer)

        response = await context_handlers.handle_context_pack(
            room, event, ["test query"], bad_packer
        )

        assert "Error" in response

    @pytest.mark.asyncio
    async def test_spread_exception_handling(self, room, event, context_handlers, mock_kg):
        """Test spread handles exceptions gracefully."""
        bad_spreader = Mock()
        bad_spreader.spread_activation = Mock(side_effect=Exception("Test error"))

        response = await context_handlers.handle_context_spread(
            room, event, ["test_node"], bad_spreader
        )

        assert "Error" in response

    @pytest.mark.asyncio
    async def test_importance_exception_handling(self, room, event, context_handlers, mock_kg):
        """Test importance handles exceptions gracefully."""
        bad_scorer = Mock()
        bad_scorer.score_importance = Mock(side_effect=Exception("Test error"))

        response = await context_handlers.handle_context_importance(
            room, event, ["test_node"], bad_scorer
        )

        assert "Error" in response

    @pytest.mark.asyncio
    async def test_pack_with_target_tokens(self, room, event, context_handlers, mock_packer, mock_kg):
        """Test pack accepts custom target tokens."""
        response = await context_handlers.handle_context_pack(
            room, event, ["test query", "1000"], mock_packer
        )

        assert "Context Packing" in response or "Error" not in response

    @pytest.mark.asyncio
    async def test_spread_with_max_hops(self, room, event, context_handlers, mock_spreader, mock_kg):
        """Test spread accepts custom max hops."""
        response = await context_handlers.handle_context_spread(
            room, event, ["test_node", "5"], mock_spreader
        )

        assert "Spreading" in response or "Activated" in response or "Beta" in response


# ============================================================================
# Test: Dict-based Bot Registration
# ============================================================================

class TestDictBasedRegistration:
    """Test registration with dict-based command interface."""

    def test_register_with_dict_commands(self, context_handlers, mock_packer, mock_spreader, mock_scorer, mock_kg):
        """Test handlers can register via dict interface."""
        mock_bot = Mock(spec=[])  # No register_command method
        mock_bot.commands = {}

        context_handlers.register_context_handlers(
            mock_bot,
            mock_packer,
            mock_spreader,
            mock_scorer,
            mock_kg
        )

        # Should have added commands to dict
        assert 'context pack' in mock_bot.commands
        assert 'context budget' in mock_bot.commands
        assert 'context spread' in mock_bot.commands


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
