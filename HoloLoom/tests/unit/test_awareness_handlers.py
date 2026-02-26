"""
Test Awareness Handlers
=======================

Comprehensive test suite for HoloLoom ChatOps awareness command handlers.

Tests:
- !aware metrics - Awareness graph metrics
- !aware activate <query> - Memory activation
- !aware spring <nodes> - Spring propagation
- !aware wave mode - Brain wave mode
- !aware consolidate - Theta consolidation
- !aware confidence - Meta-confidence
- !aware gaps - Knowledge gap detection
- !aware help - Help command

Created: 2025-12-15
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


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
# Mock Awareness Graph Types
# ============================================================================

@dataclass
class MockAwarenessMetrics:
    """Mock awareness metrics."""
    activation: Dict[str, Any]
    coherence: Dict[str, Any]
    shift: Dict[str, Any]
    temporal: Dict[str, Any]


@dataclass
class MockPerception:
    """Mock perception result."""
    activated_nodes: List[tuple]


class MockAwarenessGraph:
    """Mock AwarenessGraph for testing."""

    def __init__(self):
        self.activation_field = Mock()
        self.activation_field.activations = {
            'thompson_sampling': 0.9,
            'bayesian': 0.7,
            'neural_net': 0.5,
            'exploration': 0.3
        }
        self.trajectory = []
        self._metrics = {
            'activation': {
                'active_nodes': 4,
                'mean_activation': 0.6,
                'peak_activation': 0.9
            },
            'coherence': {
                'global_coherence': 0.75
            },
            'shift': {
                'detected': False,
                'magnitude': 0.1
            },
            'temporal': {
                'trajectory_length': 10
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return mock metrics."""
        return self._metrics

    def perceive(self, query: str) -> MockPerception:
        """Mock perceive method."""
        return MockPerception(activated_nodes=[
            ('thompson_sampling', 0.9),
            ('bayesian', 0.7),
            ('exploration', 0.5)
        ])

    def activate(self, query: str) -> List[tuple]:
        """Mock activate method."""
        return [
            ('thompson_sampling', 0.9),
            ('bayesian', 0.7)
        ]

    def get_meta_confidence(self) -> float:
        """Mock meta-confidence."""
        return 0.75

    def detect_knowledge_gaps(self) -> List[Dict]:
        """Mock knowledge gap detection."""
        return [
            {
                'query': 'quantum computing',
                'confidence': 0.3,
                'reason': 'Low confidence response'
            },
            {
                'query': 'advanced calculus',
                'confidence': 0.25,
                'reason': 'No relevant memories'
            }
        ]


# ============================================================================
# Mock Spring Dynamics Types
# ============================================================================

@dataclass
class MockSpringPropagationResult:
    """Mock spring propagation result."""
    activated_nodes: Dict[str, float]
    iterations: int = 50
    final_energy: float = 0.001


class MockSpringDynamics:
    """Mock SpringDynamics for testing."""

    def __init__(self):
        self._activations = {}

    def activate_nodes(self, initial: Dict[str, float]) -> None:
        """Set initial activations."""
        self._activations = initial.copy()

    def set_activations(self, initial: Dict[str, float]) -> None:
        """Alternative activation method."""
        self._activations = initial.copy()

    def propagate(self) -> MockSpringPropagationResult:
        """Mock propagation."""
        # Simulate propagation spreading activation
        activated = self._activations.copy()
        activated['related_node_1'] = 0.5
        activated['related_node_2'] = 0.3
        activated['distant_node'] = 0.1
        return MockSpringPropagationResult(
            activated_nodes=activated,
            iterations=50,
            final_energy=0.001
        )


# ============================================================================
# Mock Multi-Wave Engine Types
# ============================================================================

class MockBrainWaveMode(Enum):
    """Mock brain wave modes."""
    BETA = "beta"
    ALPHA = "alpha"
    THETA = "theta"
    DELTA = "delta"
    REM = "rem"


class MockThetaConsolidator:
    """Mock theta wave consolidator."""

    def consolidate(self) -> Dict[str, Any]:
        """Mock consolidation."""
        return {
            'pairs_analyzed': 100,
            'strengthened': 25,
            'new_links': 5
        }


class MockMultiWaveEngine:
    """Mock MultiWaveMemoryEngine for testing."""

    def __init__(self):
        self._mode = MockBrainWaveMode.BETA
        self.theta_consolidator = MockThetaConsolidator()

    def get_current_mode(self) -> MockBrainWaveMode:
        """Get current brain wave mode."""
        return self._mode

    def run_consolidation(self) -> Dict[str, Any]:
        """Run consolidation via engine."""
        return self.theta_consolidator.consolidate()


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
def mock_awareness_graph():
    """Create a mock AwarenessGraph."""
    return MockAwarenessGraph()


@pytest.fixture
def mock_spring_dynamics():
    """Create a mock SpringDynamics."""
    return MockSpringDynamics()


@pytest.fixture
def mock_wave_engine():
    """Create a mock MultiWaveMemoryEngine."""
    return MockMultiWaveEngine()


@pytest.fixture
def awareness_handlers(mock_awareness_graph, mock_spring_dynamics, mock_wave_engine):
    """Import and configure awareness handlers."""
    # Patch the imports
    with patch.dict('sys.modules', {
        'HoloLoom.memory.awareness_graph': Mock(
            AwarenessGraph=MockAwarenessGraph
        ),
        'HoloLoom.memory.awareness_types': Mock(
            AwarenessMetrics=MockAwarenessMetrics,
            ActivationStrategy=Mock(),
            ActivationBudget=Mock()
        ),
        'HoloLoom.memory.spring_dynamics': Mock(
            SpringDynamics=MockSpringDynamics,
            SpringConfig=Mock()
        ),
        'HoloLoom.memory.multi_wave_engine': Mock(
            MultiWaveMemoryEngine=MockMultiWaveEngine,
            BrainWaveMode=MockBrainWaveMode,
            ThetaWaveConsolidator=MockThetaConsolidator
        ),
        'HoloLoom.awareness.meta_awareness': Mock(
            MetaConfidence=Mock(),
            KnowledgeGapHypothesis=Mock()
        ),
        'nio': Mock(
            MatrixRoom=MockMatrixRoom,
            RoomMessageText=MockRoomMessageText
        ),
    }):
        # Import handlers
        from HoloLoom.apps.chatops.handlers import awareness_handlers as ah

        # Set global instances
        ah.set_awareness_graph(mock_awareness_graph)
        ah.set_spring_dynamics(mock_spring_dynamics)
        ah.set_wave_engine(mock_wave_engine)

        # Set availability flags
        ah.AWARENESS_AVAILABLE = True
        ah.SPRING_AVAILABLE = True
        ah.MULTIWAVE_AVAILABLE = True
        ah.META_AWARENESS_AVAILABLE = True
        ah.MATRIX_AVAILABLE = True

        yield ah


# ============================================================================
# Test: !aware metrics
# ============================================================================

class TestAwareMetricsCommand:
    """Test !aware metrics command."""

    @pytest.mark.asyncio
    async def test_metrics_success(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test successful metrics retrieval."""
        response = await awareness_handlers.handle_aware_metrics(
            room, event, "", mock_awareness_graph
        )

        assert "Awareness Metrics" in response
        assert "Activation" in response
        assert "Active Nodes" in response
        assert "Coherence" in response

    @pytest.mark.asyncio
    async def test_metrics_with_global_instance(self, room, event, awareness_handlers):
        """Test metrics using global awareness graph."""
        response = await awareness_handlers.handle_aware_metrics(room, event, "")

        assert "Awareness Metrics" in response
        assert "4" in response  # active_nodes count

    @pytest.mark.asyncio
    async def test_metrics_no_graph(self, room, event, awareness_handlers):
        """Test metrics when no graph available."""
        awareness_handlers.set_awareness_graph(None)
        awareness_handlers.AWARENESS_AVAILABLE = False

        response = await awareness_handlers.handle_aware_metrics(room, event, "")

        assert "Unavailable" in response or "Not Initialized" in response

    @pytest.mark.asyncio
    async def test_metrics_coherence_bar(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test coherence visual bar is included."""
        response = await awareness_handlers.handle_aware_metrics(
            room, event, "", mock_awareness_graph
        )

        # Should have visual bar characters
        assert "█" in response or "░" in response


# ============================================================================
# Test: !aware activate
# ============================================================================

class TestAwareActivateCommand:
    """Test !aware activate command."""

    @pytest.mark.asyncio
    async def test_activate_success(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test successful memory activation."""
        response = await awareness_handlers.handle_aware_activate(
            room, event, "Thompson Sampling", mock_awareness_graph
        )

        assert "Activated" in response
        assert "thompson_sampling" in response or "Memories" in response

    @pytest.mark.asyncio
    async def test_activate_empty_query(self, room, event, awareness_handlers):
        """Test activation with empty query shows usage."""
        response = await awareness_handlers.handle_aware_activate(room, event, "")

        assert "Usage" in response
        assert "activate" in response

    @pytest.mark.asyncio
    async def test_activate_shows_activation_bars(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test activation shows visual bars."""
        response = await awareness_handlers.handle_aware_activate(
            room, event, "test query", mock_awareness_graph
        )

        # Should have activation bar characters
        assert "▓" in response or "░" in response

    @pytest.mark.asyncio
    async def test_activate_no_graph(self, room, event, awareness_handlers):
        """Test activation when no graph available."""
        awareness_handlers.set_awareness_graph(None)

        response = await awareness_handlers.handle_aware_activate(
            room, event, "test query"
        )

        assert "Error" in response or "not initialized" in response


# ============================================================================
# Test: !aware spring
# ============================================================================

class TestAwareSpringCommand:
    """Test !aware spring command."""

    @pytest.mark.asyncio
    async def test_spring_single_node(self, room, event, awareness_handlers, mock_spring_dynamics):
        """Test spring propagation with single node."""
        response = await awareness_handlers.handle_aware_spring(
            room, event, "thompson_sampling", mock_spring_dynamics
        )

        assert "Spring Propagation" in response
        assert "thompson_sampling" in response
        assert "Activated Nodes" in response

    @pytest.mark.asyncio
    async def test_spring_multiple_nodes(self, room, event, awareness_handlers, mock_spring_dynamics):
        """Test spring propagation with multiple nodes."""
        response = await awareness_handlers.handle_aware_spring(
            room, event, "node1,node2,node3", mock_spring_dynamics
        )

        assert "Spring Propagation" in response
        assert "Seeds" in response

    @pytest.mark.asyncio
    async def test_spring_empty_args(self, room, event, awareness_handlers):
        """Test spring with empty args shows usage."""
        response = await awareness_handlers.handle_aware_spring(room, event, "")

        assert "Usage" in response
        assert "spring" in response

    @pytest.mark.asyncio
    async def test_spring_shows_iterations(self, room, event, awareness_handlers, mock_spring_dynamics):
        """Test spring shows iteration count."""
        response = await awareness_handlers.handle_aware_spring(
            room, event, "test_node", mock_spring_dynamics
        )

        assert "Iterations" in response

    @pytest.mark.asyncio
    async def test_spring_no_dynamics(self, room, event, awareness_handlers):
        """Test spring when no dynamics available."""
        awareness_handlers.set_spring_dynamics(None)
        awareness_handlers.SPRING_AVAILABLE = False

        response = await awareness_handlers.handle_aware_spring(
            room, event, "test_node"
        )

        assert "Unavailable" in response or "Not Initialized" in response


# ============================================================================
# Test: !aware wave mode
# ============================================================================

class TestAwareWaveModeCommand:
    """Test !aware wave mode command."""

    @pytest.mark.asyncio
    async def test_wave_mode_success(self, room, event, awareness_handlers, mock_wave_engine):
        """Test successful wave mode retrieval."""
        response = await awareness_handlers.handle_aware_wave_mode(
            room, event, "", mock_wave_engine
        )

        assert "Brain Wave Mode" in response
        assert "Beta" in response or "BETA" in response

    @pytest.mark.asyncio
    async def test_wave_mode_shows_frequency(self, room, event, awareness_handlers, mock_wave_engine):
        """Test wave mode shows frequency info."""
        response = await awareness_handlers.handle_aware_wave_mode(
            room, event, "", mock_wave_engine
        )

        assert "Hz" in response

    @pytest.mark.asyncio
    async def test_wave_mode_shows_transitions(self, room, event, awareness_handlers, mock_wave_engine):
        """Test wave mode shows transition info."""
        response = await awareness_handlers.handle_aware_wave_mode(
            room, event, "", mock_wave_engine
        )

        assert "Transitions" in response or "idle" in response.lower()

    @pytest.mark.asyncio
    async def test_wave_mode_no_engine(self, room, event, awareness_handlers):
        """Test wave mode when no engine available."""
        awareness_handlers.set_wave_engine(None)
        awareness_handlers.MULTIWAVE_AVAILABLE = False

        response = await awareness_handlers.handle_aware_wave_mode(room, event, "")

        assert "Unavailable" in response or "Not Initialized" in response


# ============================================================================
# Test: !aware consolidate
# ============================================================================

class TestAwareConsolidateCommand:
    """Test !aware consolidate command."""

    @pytest.mark.asyncio
    async def test_consolidate_success(self, room, event, awareness_handlers, mock_wave_engine):
        """Test successful theta consolidation."""
        response = await awareness_handlers.handle_aware_consolidate(
            room, event, "", mock_wave_engine
        )

        assert "Theta Consolidation" in response or "Consolidation" in response

    @pytest.mark.asyncio
    async def test_consolidate_shows_stats(self, room, event, awareness_handlers, mock_wave_engine):
        """Test consolidation shows statistics."""
        response = await awareness_handlers.handle_aware_consolidate(
            room, event, "", mock_wave_engine
        )

        # Should show consolidation stats
        assert "Pairs" in response or "Strengthened" in response or "Links" in response

    @pytest.mark.asyncio
    async def test_consolidate_no_engine(self, room, event, awareness_handlers):
        """Test consolidation when no engine available."""
        awareness_handlers.set_wave_engine(None)

        response = await awareness_handlers.handle_aware_consolidate(room, event, "")

        assert "Error" in response or "not initialized" in response


# ============================================================================
# Test: !aware confidence
# ============================================================================

class TestAwareConfidenceCommand:
    """Test !aware confidence command."""

    @pytest.mark.asyncio
    async def test_confidence_success(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test successful meta-confidence retrieval."""
        response = await awareness_handlers.handle_aware_confidence(
            room, event, "", mock_awareness_graph
        )

        assert "Meta-Confidence" in response or "Confidence" in response

    @pytest.mark.asyncio
    async def test_confidence_shows_level(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test confidence shows level percentage."""
        response = await awareness_handlers.handle_aware_confidence(
            room, event, "", mock_awareness_graph
        )

        assert "%" in response or "Level" in response

    @pytest.mark.asyncio
    async def test_confidence_shows_visual_bar(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test confidence shows visual bar."""
        response = await awareness_handlers.handle_aware_confidence(
            room, event, "", mock_awareness_graph
        )

        # Should have bar characters
        assert "█" in response or "░" in response

    @pytest.mark.asyncio
    async def test_confidence_shows_status(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test confidence shows status interpretation."""
        response = await awareness_handlers.handle_aware_confidence(
            room, event, "", mock_awareness_graph
        )

        # Should have status indicator
        assert "Status" in response or "✅" in response or "🟡" in response or "⚠️" in response

    @pytest.mark.asyncio
    async def test_confidence_no_graph(self, room, event, awareness_handlers):
        """Test confidence when no graph available."""
        awareness_handlers.set_awareness_graph(None)

        response = await awareness_handlers.handle_aware_confidence(room, event, "")

        assert "Error" in response or "not initialized" in response


# ============================================================================
# Test: !aware gaps
# ============================================================================

class TestAwareGapsCommand:
    """Test !aware gaps command."""

    @pytest.mark.asyncio
    async def test_gaps_success(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test successful gap detection."""
        response = await awareness_handlers.handle_aware_gaps(
            room, event, "", mock_awareness_graph
        )

        assert "Knowledge Gap" in response or "Gap" in response

    @pytest.mark.asyncio
    async def test_gaps_shows_queries(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test gaps shows problem queries."""
        response = await awareness_handlers.handle_aware_gaps(
            room, event, "", mock_awareness_graph
        )

        # Should show the gap queries from mock
        assert "quantum" in response.lower() or "calculus" in response.lower()

    @pytest.mark.asyncio
    async def test_gaps_shows_confidence(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test gaps shows confidence values."""
        response = await awareness_handlers.handle_aware_gaps(
            room, event, "", mock_awareness_graph
        )

        assert "Confidence" in response or "0." in response

    @pytest.mark.asyncio
    async def test_gaps_no_graph(self, room, event, awareness_handlers):
        """Test gaps when no graph available."""
        awareness_handlers.set_awareness_graph(None)

        response = await awareness_handlers.handle_aware_gaps(room, event, "")

        assert "Error" in response or "not initialized" in response

    @pytest.mark.asyncio
    async def test_gaps_none_detected(self, room, event, awareness_handlers, mock_awareness_graph):
        """Test gaps when none detected."""
        # Override to return empty gaps
        mock_awareness_graph.detect_knowledge_gaps = lambda: []

        response = await awareness_handlers.handle_aware_gaps(
            room, event, "", mock_awareness_graph
        )

        assert "No" in response or "good coverage" in response.lower()


# ============================================================================
# Test: !aware help
# ============================================================================

class TestAwareHelpCommand:
    """Test !aware help command."""

    @pytest.mark.asyncio
    async def test_help_returns_content(self, room, event, awareness_handlers):
        """Test help returns help content."""
        response = await awareness_handlers.handle_aware_help(room, event, "")

        assert "Awareness Commands" in response

    @pytest.mark.asyncio
    async def test_help_lists_commands(self, room, event, awareness_handlers):
        """Test help lists all awareness commands."""
        response = await awareness_handlers.handle_aware_help(room, event, "")

        assert "!aware metrics" in response
        assert "!aware activate" in response
        assert "!aware spring" in response
        assert "!aware wave" in response or "wave mode" in response
        assert "!aware consolidate" in response
        assert "!aware confidence" in response
        assert "!aware gaps" in response

    @pytest.mark.asyncio
    async def test_help_explains_modes(self, room, event, awareness_handlers):
        """Test help explains brain wave modes."""
        response = await awareness_handlers.handle_aware_help(room, event, "")

        assert "Beta" in response
        assert "Alpha" in response
        assert "Theta" in response
        assert "Delta" in response
        assert "REM" in response


# ============================================================================
# Test: Handler Registration
# ============================================================================

class TestHandlerRegistration:
    """Test handler registration functionality."""

    def test_register_handlers(self, awareness_handlers, mock_awareness_graph, mock_spring_dynamics, mock_wave_engine):
        """Test handlers can be registered with a bot."""
        mock_bot = Mock()
        mock_bot.register_command = Mock()

        result = awareness_handlers.register_awareness_handlers(
            mock_bot,
            mock_awareness_graph,
            mock_spring_dynamics,
            mock_wave_engine
        )

        # Should return AwarenessHandlers instance
        assert result is not None
        assert hasattr(result, 'handle_metrics')
        assert hasattr(result, 'handle_activate')
        assert hasattr(result, 'handle_spring')
        assert hasattr(result, 'handle_wave_mode')
        assert hasattr(result, 'handle_consolidate')
        assert hasattr(result, 'handle_confidence')
        assert hasattr(result, 'handle_gaps')
        assert hasattr(result, 'handle_help')

    def test_register_handlers_calls_bot_register(self, awareness_handlers, mock_awareness_graph, mock_spring_dynamics, mock_wave_engine):
        """Test registration calls bot.register_command."""
        mock_bot = Mock()
        mock_bot.register_command = Mock()

        awareness_handlers.register_awareness_handlers(
            mock_bot,
            mock_awareness_graph,
            mock_spring_dynamics,
            mock_wave_engine
        )

        # Should have called register_command for each handler
        assert mock_bot.register_command.call_count >= 8

    def test_awareness_handlers_class(self, mock_awareness_graph, mock_spring_dynamics, mock_wave_engine):
        """Test AwarenessHandlers class initialization."""
        from HoloLoom.apps.chatops.handlers.awareness_handlers import AwarenessHandlers

        handlers = AwarenessHandlers(
            mock_awareness_graph,
            mock_spring_dynamics,
            mock_wave_engine
        )

        assert handlers.awareness_graph == mock_awareness_graph
        assert handlers.spring_dynamics == mock_spring_dynamics
        assert handlers.wave_engine == mock_wave_engine


# ============================================================================
# Test: Global Instance Management
# ============================================================================

class TestGlobalInstanceManagement:
    """Test global instance getters and setters."""

    def test_get_set_awareness_graph(self, awareness_handlers, mock_awareness_graph):
        """Test get/set awareness graph."""
        awareness_handlers.set_awareness_graph(mock_awareness_graph)
        result = awareness_handlers.get_awareness_graph()
        assert result == mock_awareness_graph

    def test_get_set_spring_dynamics(self, awareness_handlers, mock_spring_dynamics):
        """Test get/set spring dynamics."""
        awareness_handlers.set_spring_dynamics(mock_spring_dynamics)
        result = awareness_handlers.get_spring_dynamics()
        assert result == mock_spring_dynamics

    def test_get_set_wave_engine(self, awareness_handlers, mock_wave_engine):
        """Test get/set wave engine."""
        awareness_handlers.set_wave_engine(mock_wave_engine)
        result = awareness_handlers.get_wave_engine()
        assert result == mock_wave_engine

    def test_get_returns_none_when_not_set(self, awareness_handlers):
        """Test getters return None when not set."""
        awareness_handlers.set_awareness_graph(None)
        awareness_handlers.set_spring_dynamics(None)
        awareness_handlers.set_wave_engine(None)

        assert awareness_handlers.get_awareness_graph() is None
        assert awareness_handlers.get_spring_dynamics() is None
        assert awareness_handlers.get_wave_engine() is None


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_metrics_exception_handling(self, room, event, awareness_handlers):
        """Test metrics handles exceptions gracefully."""
        # Create a mock that raises an exception
        bad_graph = Mock()
        bad_graph.get_metrics = Mock(side_effect=Exception("Test error"))

        response = await awareness_handlers.handle_aware_metrics(
            room, event, "", bad_graph
        )

        assert "Error" in response

    @pytest.mark.asyncio
    async def test_activate_exception_handling(self, room, event, awareness_handlers):
        """Test activate handles exceptions gracefully."""
        bad_graph = Mock()
        bad_graph.perceive = Mock(side_effect=Exception("Test error"))
        bad_graph.activate = Mock(side_effect=Exception("Test error"))

        response = await awareness_handlers.handle_aware_activate(
            room, event, "test query", bad_graph
        )

        assert "Error" in response

    @pytest.mark.asyncio
    async def test_spring_exception_handling(self, room, event, awareness_handlers):
        """Test spring handles exceptions gracefully."""
        bad_dynamics = Mock()
        bad_dynamics.activate_nodes = Mock()
        bad_dynamics.propagate = Mock(side_effect=Exception("Test error"))

        response = await awareness_handlers.handle_aware_spring(
            room, event, "test_node", bad_dynamics
        )

        assert "Error" in response

    @pytest.mark.asyncio
    async def test_long_node_ids_truncated(self, room, event, awareness_handlers, mock_spring_dynamics):
        """Test long node IDs are truncated in display."""
        # Create dynamics that returns long node IDs
        long_id = "very_long_node_id_that_should_be_truncated_in_the_display_output"
        mock_spring_dynamics.propagate = Mock(return_value=MockSpringPropagationResult(
            activated_nodes={long_id: 0.9},
            iterations=50,
            final_energy=0.001
        ))

        response = await awareness_handlers.handle_aware_spring(
            room, event, long_id, mock_spring_dynamics
        )

        # Should have ellipsis for long IDs
        assert "..." in response

    @pytest.mark.asyncio
    async def test_wave_mode_string_mode(self, room, event, awareness_handlers):
        """Test wave mode handles string mode values."""
        string_engine = Mock()
        string_engine.get_current_mode = Mock(return_value="BETA")

        response = await awareness_handlers.handle_aware_wave_mode(
            room, event, "", string_engine
        )

        assert "Beta" in response or "BETA" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
