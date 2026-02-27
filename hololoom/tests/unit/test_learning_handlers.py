"""Unit tests for HoloLoom learning ChatOps handlers.

Tests the complete ChatOps integration for learning commands:
- Learning statistics display
- Tool-specific Thompson Sampling priors
- Hot pattern retrieval
- Cold pattern detection
- Query refinement
- Pattern learning
- Reflection buffer metrics
- Help documentation
- Handler registration

Author: HoloLoom Team
Date: 2025-12-15
"""

import asyncio
import pytest
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from unittest.mock import Mock, AsyncMock, patch


# ============================================================================
# Mock Enums and Types
# ============================================================================

class MockRefinementStrategy(Enum):
    """Mock RefinementStrategy enum."""
    REFINE = "refine"
    CRITIQUE = "critique"
    VERIFY = "verify"
    ELEGANCE = "elegance"
    HOFSTADTER = "hofstadter"


# ============================================================================
# Mock Data Classes
# ============================================================================

@dataclass
class MockMatrixRoom:
    """Mock nio.MatrixRoom for testing."""
    room_id: str = "!test:example.com"
    name: str = "test-room"
    display_name: str = "Test Room"


@dataclass
class MockRoomMessageText:
    """Mock nio.RoomMessageText event."""
    sender: str = "@user:example.com"
    body: str = ""
    room_id: str = "!test:example.com"


@dataclass
class MockThompsonPriors:
    """Mock Thompson Sampling priors for a tool."""
    tool_name: str
    alpha: float = 1.0
    beta: float = 1.0
    expected_reward: float = 0.5
    pulls: int = 0
    successes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool': self.tool_name,
            'alpha': self.alpha,
            'beta': self.beta,
            'expected_reward': self.expected_reward,
            'pulls': self.pulls,
            'successes': self.successes
        }


@dataclass
class MockPolicyWeights:
    """Mock policy adapter weights."""
    adapter_weights: Dict[str, float] = field(default_factory=lambda: {
        'bare': 0.2,
        'fast': 0.5,
        'fused': 0.3
    })
    total_queries: int = 100


@dataclass
class MockUsageRecord:
    """Mock usage record for hot patterns."""
    element_id: str
    access_count: int = 1
    success_count: int = 1
    avg_confidence: float = 0.85
    last_access: datetime = field(default_factory=datetime.now)
    heat_score: float = 0.75

    def to_dict(self) -> Dict[str, Any]:
        return {
            'element_id': self.element_id,
            'access_count': self.access_count,
            'success_count': self.success_count,
            'avg_confidence': self.avg_confidence,
            'heat_score': self.heat_score
        }


@dataclass
class MockLearnedPattern:
    """Mock learned pattern."""
    motif: str
    tool: str
    confidence: float
    count: int
    last_seen: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'motif': self.motif,
            'tool': self.tool,
            'confidence': self.confidence,
            'count': self.count
        }


@dataclass
class MockRefinementResult:
    """Mock refinement result."""
    original_query: str
    refined_query: str
    strategy: MockRefinementStrategy
    iterations: int = 3
    quality_before: float = 0.65
    quality_after: float = 0.92

    def summary(self) -> str:
        return (
            f"Strategy: {self.strategy.value}, "
            f"Iterations: {self.iterations}, "
            f"Quality: {self.quality_before:.2f} → {self.quality_after:.2f}"
        )


@dataclass
class MockReflectionMetrics:
    """Mock reflection buffer metrics."""
    total_entries: int = 50
    success_rate: float = 0.85
    avg_confidence: float = 0.87
    tool_usage: Dict[str, int] = field(default_factory=lambda: {
        'answer': 30,
        'research': 15,
        'verify': 5
    })
    recent_patterns: List[str] = field(default_factory=lambda: [
        'question→answer pattern successful',
        'complex queries benefit from research mode'
    ])


# ============================================================================
# Mock Classes
# ============================================================================

class MockHotPatternTracker:
    """Mock HotPatternTracker for testing."""

    def __init__(self):
        self._patterns = [
            MockUsageRecord('thompson_sampling', 15, 12, 0.91, heat_score=0.89),
            MockUsageRecord('bayesian_methods', 12, 10, 0.88, heat_score=0.82),
            MockUsageRecord('exploration_strategy', 8, 6, 0.75, heat_score=0.65),
            MockUsageRecord('neural_networks', 5, 3, 0.60, heat_score=0.45),
            MockUsageRecord('old_concept', 2, 1, 0.50, heat_score=0.15),
        ]

    def get_hot_patterns(self, k: int = 10) -> List[MockUsageRecord]:
        """Get top K hot patterns by heat score."""
        sorted_patterns = sorted(
            self._patterns,
            key=lambda p: p.heat_score,
            reverse=True
        )
        return sorted_patterns[:k]

    def get_cold_patterns(self, age_hours: int = 24) -> List[MockUsageRecord]:
        """Get cold patterns (low heat score)."""
        cold = [p for p in self._patterns if p.heat_score < 0.3]
        return cold

    def get_all_patterns(self) -> List[MockUsageRecord]:
        """Get all tracked patterns."""
        return self._patterns


class MockPatternLearner:
    """Mock PatternLearner for testing."""

    def __init__(self):
        self._patterns = [
            MockLearnedPattern('question→answer', 'answer', 0.92, 45),
            MockLearnedPattern('explain→answer', 'answer', 0.88, 30),
            MockLearnedPattern('compare→research', 'research', 0.85, 20),
            MockLearnedPattern('verify→verify', 'verify', 0.90, 15),
            MockLearnedPattern('analyze→research', 'research', 0.82, 12),
        ]

    def get_patterns(self) -> List[MockLearnedPattern]:
        """Get all learned patterns."""
        return self._patterns

    def get_pattern_for_motif(self, motif: str) -> Optional[MockLearnedPattern]:
        """Get pattern for a specific motif."""
        for p in self._patterns:
            if motif in p.motif:
                return p
        return None


class MockAdvancedRefiner:
    """Mock AdvancedRefiner for testing."""

    async def refine(
        self,
        query: str,
        strategy: Optional[MockRefinementStrategy] = None,
        max_iterations: int = 3
    ) -> MockRefinementResult:
        """Refine a query using specified strategy."""
        if strategy is None:
            # Auto-select strategy based on query
            if 'verify' in query.lower():
                strategy = MockRefinementStrategy.VERIFY
            elif 'explain' in query.lower():
                strategy = MockRefinementStrategy.ELEGANCE
            else:
                strategy = MockRefinementStrategy.REFINE

        return MockRefinementResult(
            original_query=query,
            refined_query=f"[Refined] {query}",
            strategy=strategy,
            iterations=max_iterations
        )


class MockReflectionBuffer:
    """Mock ReflectionBuffer for testing."""

    def __init__(self):
        self._metrics = MockReflectionMetrics()

    def get_metrics(self) -> MockReflectionMetrics:
        """Get reflection buffer metrics."""
        return self._metrics

    @property
    def capacity(self) -> int:
        return 1000

    @property
    def current_size(self) -> int:
        return 50


class MockFullLearningEngine:
    """Mock FullLearningEngine for testing."""

    def __init__(self):
        self.hot_tracker = MockHotPatternTracker()
        self.pattern_learner = MockPatternLearner()
        self.refiner = MockAdvancedRefiner()
        self._thompson_priors = {
            'answer': MockThompsonPriors('answer', 10.5, 2.1, 0.83, 50, 42),
            'research': MockThompsonPriors('research', 5.2, 1.8, 0.74, 25, 18),
            'verify': MockThompsonPriors('verify', 3.1, 0.9, 0.78, 15, 12),
            'execute': MockThompsonPriors('execute', 2.0, 2.5, 0.44, 10, 4),
        }
        self._policy_weights = MockPolicyWeights()

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        return {
            'thompson_priors': {
                name: priors.to_dict()
                for name, priors in self._thompson_priors.items()
            },
            'policy_weights': self._policy_weights.adapter_weights,
            'total_queries': self._policy_weights.total_queries,
            'hot_pattern_count': len(self.hot_tracker.get_hot_patterns()),
            'learned_patterns': [p.to_dict() for p in self.pattern_learner.get_patterns()],
            'background_learning': True
        }

    def get_tool_stats(self, tool_name: str) -> Optional[MockThompsonPriors]:
        """Get Thompson priors for a specific tool."""
        return self._thompson_priors.get(tool_name)

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return list(self._thompson_priors.keys())


# ============================================================================
# Handler Test Fixtures
# ============================================================================

@pytest.fixture
def mock_room():
    """Create mock Matrix room."""
    return MockMatrixRoom()


@pytest.fixture
def mock_event():
    """Create mock message event."""
    return MockRoomMessageText()


@pytest.fixture
def mock_learning_engine():
    """Create mock FullLearningEngine."""
    return MockFullLearningEngine()


@pytest.fixture
def mock_hot_tracker():
    """Create mock HotPatternTracker."""
    return MockHotPatternTracker()


@pytest.fixture
def mock_reflection_buffer():
    """Create mock ReflectionBuffer."""
    return MockReflectionBuffer()


# ============================================================================
# Import handlers with mocks
# ============================================================================

@pytest.fixture
def learning_handlers(mock_learning_engine, mock_hot_tracker, mock_reflection_buffer):
    """Import and configure learning handlers."""
    with patch.dict('sys.modules', {
        'hololoom.recursive': Mock(
            FullLearningEngine=MockFullLearningEngine,
            ThompsonPriors=MockThompsonPriors,
            PolicyWeights=MockPolicyWeights,
            HotPatternTracker=MockHotPatternTracker,
            HotPatternFeedbackEngine=Mock,
            HotPatternConfig=Mock,
            UsageRecord=MockUsageRecord,
            AdvancedRefiner=MockAdvancedRefiner,
            RefinementStrategy=MockRefinementStrategy,
            RefinementResult=MockRefinementResult,
            PatternLearner=MockPatternLearner,
            LearnedPattern=MockLearnedPattern,
        ),
        'hololoom.reflection.buffer': Mock(
            ReflectionBuffer=MockReflectionBuffer,
            ReflectionMetrics=MockReflectionMetrics
        ),
    }):
        from hololoom.apps.chatops.handlers import learning_handlers as lh

        # Set global instances
        lh.set_learning_engine(mock_learning_engine)
        lh.set_hot_tracker(mock_hot_tracker)
        lh.set_reflection_buffer(mock_reflection_buffer)

        # Patch module constants
        lh.LEARNING_AVAILABLE = True
        lh.REFLECTION_AVAILABLE = True
        lh.RefinementStrategy = MockRefinementStrategy

        yield lh


# ============================================================================
# TESTS: Learn Stats Command
# ============================================================================

class TestLearnStatsCommand:
    """Test !learn stats command."""

    @pytest.mark.asyncio
    async def test_handle_learn_stats_basic(
        self, learning_handlers, mock_room, mock_event, mock_learning_engine
    ):
        """Test basic learning stats display."""
        result = await learning_handlers.handle_learn_stats(
            room=mock_room,
            event=mock_event,
            args="",
            learning_engine=mock_learning_engine
        )

        assert "Learning Statistics" in result or "Statistics" in result
        assert "Thompson" in result or "thompson" in result.lower()
        assert "Policy" in result or "policy" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_learn_stats_no_engine(
        self, learning_handlers, mock_room, mock_event
    ):
        """Test stats when learning engine not available."""
        learning_handlers.set_learning_engine(None)
        learning_handlers.LEARNING_AVAILABLE = False

        result = await learning_handlers.handle_learn_stats(
            room=mock_room,
            event=mock_event,
            args="",
            learning_engine=None
        )

        assert "not available" in result.lower() or "unavailable" in result.lower() or "not initialized" in result.lower()


class TestLearnToolCommand:
    """Test !learn tool command."""

    @pytest.mark.asyncio
    async def test_handle_learn_tool_basic(
        self, learning_handlers, mock_room, mock_event, mock_learning_engine
    ):
        """Test tool-specific stats retrieval."""
        result = await learning_handlers.handle_learn_tool(
            room=mock_room,
            event=mock_event,
            args="answer",
            learning_engine=mock_learning_engine
        )

        assert "answer" in result.lower()
        assert "alpha" in result.lower() or "α" in result

    @pytest.mark.asyncio
    async def test_handle_learn_tool_unknown(
        self, learning_handlers, mock_room, mock_event, mock_learning_engine
    ):
        """Test tool stats for unknown tool."""
        result = await learning_handlers.handle_learn_tool(
            room=mock_room,
            event=mock_event,
            args="unknown_tool",
            learning_engine=mock_learning_engine
        )

        assert "not found" in result.lower() or "unknown" in result.lower() or "Available" in result

    @pytest.mark.asyncio
    async def test_handle_learn_tool_empty_args(
        self, learning_handlers, mock_room, mock_event, mock_learning_engine
    ):
        """Test tool command with empty args shows usage."""
        result = await learning_handlers.handle_learn_tool(
            room=mock_room,
            event=mock_event,
            args="",
            learning_engine=mock_learning_engine
        )

        assert "Usage" in result or "Available" in result


class TestLearnHotCommand:
    """Test !learn hot command."""

    @pytest.mark.asyncio
    async def test_handle_learn_hot_basic(
        self, learning_handlers, mock_room, mock_event, mock_hot_tracker
    ):
        """Test hot patterns retrieval."""
        result = await learning_handlers.handle_learn_hot(
            room=mock_room,
            event=mock_event,
            args="",
            hot_tracker=mock_hot_tracker
        )

        assert "Hot Patterns" in result or "hot" in result.lower()
        # Should show the top patterns
        assert "thompson" in result.lower() or "pattern" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_learn_hot_with_limit(
        self, learning_handlers, mock_room, mock_event, mock_hot_tracker
    ):
        """Test hot patterns with custom limit."""
        result = await learning_handlers.handle_learn_hot(
            room=mock_room,
            event=mock_event,
            args="3",
            hot_tracker=mock_hot_tracker
        )

        # Should still work with limit
        assert "Hot" in result or "pattern" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_learn_hot_no_tracker(
        self, learning_handlers, mock_room, mock_event
    ):
        """Test hot patterns when tracker not available."""
        learning_handlers.set_hot_tracker(None)

        result = await learning_handlers.handle_learn_hot(
            room=mock_room,
            event=mock_event,
            args="",
            hot_tracker=None
        )

        assert "not available" in result.lower() or "unavailable" in result.lower() or "not initialized" in result.lower()


class TestLearnColdCommand:
    """Test !learn cold command."""

    @pytest.mark.asyncio
    async def test_handle_learn_cold_basic(
        self, learning_handlers, mock_room, mock_event, mock_hot_tracker
    ):
        """Test cold patterns retrieval."""
        result = await learning_handlers.handle_learn_cold(
            room=mock_room,
            event=mock_event,
            args="",
            hot_tracker=mock_hot_tracker
        )

        assert "Cold" in result or "cold" in result.lower() or "stale" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_learn_cold_with_age(
        self, learning_handlers, mock_room, mock_event, mock_hot_tracker
    ):
        """Test cold patterns with custom age threshold."""
        result = await learning_handlers.handle_learn_cold(
            room=mock_room,
            event=mock_event,
            args="48",  # 48 hours
            hot_tracker=mock_hot_tracker
        )

        assert "Cold" in result or "cold" in result.lower() or "patterns" in result.lower()


class TestLearnRefineCommand:
    """Test !learn refine command."""

    @pytest.mark.asyncio
    async def test_handle_learn_refine_basic(
        self, learning_handlers, mock_room, mock_event, mock_learning_engine
    ):
        """Test query refinement."""
        result = await learning_handlers.handle_learn_refine(
            room=mock_room,
            event=mock_event,
            args="What is Thompson Sampling?",
            learning_engine=mock_learning_engine
        )

        assert "Refine" in result or "refine" in result.lower() or "Strategy" in result

    @pytest.mark.asyncio
    async def test_handle_learn_refine_empty_args(
        self, learning_handlers, mock_room, mock_event, mock_learning_engine
    ):
        """Test refine command with empty args shows usage."""
        result = await learning_handlers.handle_learn_refine(
            room=mock_room,
            event=mock_event,
            args="",
            learning_engine=mock_learning_engine
        )

        assert "Usage" in result or "!learn refine" in result


class TestLearnPatternsCommand:
    """Test !learn patterns command."""

    @pytest.mark.asyncio
    async def test_handle_learn_patterns_basic(
        self, learning_handlers, mock_room, mock_event, mock_learning_engine
    ):
        """Test learned patterns retrieval."""
        result = await learning_handlers.handle_learn_patterns(
            room=mock_room,
            event=mock_event,
            args="",
            learning_engine=mock_learning_engine
        )

        assert "Pattern" in result or "pattern" in result.lower()
        # Should show some patterns like question→answer
        assert "→" in result or "->" in result or "answer" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_learn_patterns_no_engine(
        self, learning_handlers, mock_room, mock_event
    ):
        """Test patterns when engine not available."""
        learning_handlers.set_learning_engine(None)
        learning_handlers.LEARNING_AVAILABLE = False

        result = await learning_handlers.handle_learn_patterns(
            room=mock_room,
            event=mock_event,
            args="",
            learning_engine=None
        )

        assert "not available" in result.lower() or "unavailable" in result.lower() or "not initialized" in result.lower()


class TestLearnReflectCommand:
    """Test !learn reflect command."""

    @pytest.mark.asyncio
    async def test_handle_learn_reflect_basic(
        self, learning_handlers, mock_room, mock_event, mock_reflection_buffer
    ):
        """Test reflection buffer metrics display."""
        result = await learning_handlers.handle_learn_reflect(
            room=mock_room,
            event=mock_event,
            args="",
            reflection_buffer=mock_reflection_buffer
        )

        assert "Reflect" in result or "reflect" in result.lower() or "Metrics" in result

    @pytest.mark.asyncio
    async def test_handle_learn_reflect_no_buffer(
        self, learning_handlers, mock_room, mock_event
    ):
        """Test reflect when buffer not available."""
        learning_handlers.set_reflection_buffer(None)
        learning_handlers.REFLECTION_AVAILABLE = False

        result = await learning_handlers.handle_learn_reflect(
            room=mock_room,
            event=mock_event,
            args="",
            reflection_buffer=None
        )

        assert "not available" in result.lower() or "unavailable" in result.lower() or "not initialized" in result.lower()


class TestLearnHelpCommand:
    """Test !learn help command."""

    @pytest.mark.asyncio
    async def test_handle_learn_help(
        self, learning_handlers, mock_room, mock_event
    ):
        """Test learning help output."""
        result = await learning_handlers.handle_learn_help(
            room=mock_room,
            event=mock_event,
            args=""
        )

        assert "Learning Commands" in result or "learn" in result.lower()
        assert "!learn stats" in result
        assert "!learn tool" in result
        assert "!learn hot" in result
        assert "!learn cold" in result
        assert "!learn refine" in result
        assert "!learn patterns" in result
        assert "!learn reflect" in result


# ============================================================================
# TESTS: Handler Registration
# ============================================================================

class TestHandlerRegistration:
    """Test handler registration."""

    def test_register_learning_handlers(
        self, learning_handlers, mock_learning_engine, mock_hot_tracker, mock_reflection_buffer
    ):
        """Test handler registration function."""
        handlers = learning_handlers.register_learning_handlers(
            bot=None,  # No bot, just create handlers
            learning_engine=mock_learning_engine,
            hot_tracker=mock_hot_tracker,
            reflection_buffer=mock_reflection_buffer
        )

        assert handlers is not None
        assert handlers.learning_engine == mock_learning_engine
        assert handlers.hot_tracker == mock_hot_tracker
        assert handlers.reflection_buffer == mock_reflection_buffer

    def test_learning_handlers_class_methods(
        self, learning_handlers, mock_learning_engine, mock_hot_tracker, mock_reflection_buffer
    ):
        """Test LearningHandlers class has all methods."""
        handlers = learning_handlers.LearningHandlers(
            learning_engine=mock_learning_engine,
            hot_tracker=mock_hot_tracker,
            reflection_buffer=mock_reflection_buffer
        )

        # Verify all methods exist (class uses handle_ prefix)
        assert hasattr(handlers, 'handle_stats')
        assert hasattr(handlers, 'handle_tool')
        assert hasattr(handlers, 'handle_hot')
        assert hasattr(handlers, 'handle_cold')
        assert hasattr(handlers, 'handle_refine')
        assert hasattr(handlers, 'handle_patterns')
        assert hasattr(handlers, 'handle_reflect')
        assert hasattr(handlers, 'handle_help')

    def test_global_instance_setters(
        self, learning_handlers, mock_learning_engine, mock_hot_tracker, mock_reflection_buffer
    ):
        """Test global instance getters and setters."""
        # Test learning engine
        learning_handlers.set_learning_engine(mock_learning_engine)
        assert learning_handlers.get_learning_engine() == mock_learning_engine

        # Test hot tracker
        learning_handlers.set_hot_tracker(mock_hot_tracker)
        assert learning_handlers.get_hot_tracker() == mock_hot_tracker

        # Test reflection buffer
        learning_handlers.set_reflection_buffer(mock_reflection_buffer)
        assert learning_handlers.get_reflection_buffer() == mock_reflection_buffer

        # Test clearing
        learning_handlers.set_learning_engine(None)
        assert learning_handlers.get_learning_engine() is None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
