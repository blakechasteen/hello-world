"""
Unit tests for HoloLoom.recursive.full_learning_loop

Tests Thompson Sampling priors, policy weights, background learning, and complete learning engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from HoloLoom.recursive.full_learning_loop import (
    ThompsonPriors,
    PolicyWeights,
    LearningMetrics,
    BackgroundLearner,
    FullLearningEngine,
    weave_with_full_learning
)
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config


class TestThompsonPriors:
    """Test ThompsonPriors dataclass"""

    def test_initialization(self):
        """Test priors initialization"""
        priors = ThompsonPriors()

        assert isinstance(priors.tool_priors, dict)

    def test_default_priors(self):
        """Test default Beta distribution priors"""
        priors = ThompsonPriors()

        # Access a tool to trigger default
        default = priors.tool_priors["new_tool"]

        assert default["alpha"] == 1.0
        assert default["beta"] == 1.0

    def test_update_success(self):
        """Test updating priors after successful tool use"""
        priors = ThompsonPriors()

        initial_alpha = priors.tool_priors["answer"]["alpha"]

        priors.update_success("answer", confidence=0.85)

        assert priors.tool_priors["answer"]["alpha"] > initial_alpha
        assert priors.tool_priors["answer"]["alpha"] == initial_alpha + 0.85

    def test_update_failure(self):
        """Test updating priors after unsuccessful tool use"""
        priors = ThompsonPriors()

        initial_beta = priors.tool_priors["answer"]["beta"]

        priors.update_failure("answer", confidence=0.65)

        assert priors.tool_priors["answer"]["beta"] > initial_beta
        assert priors.tool_priors["answer"]["beta"] == initial_beta + (1.0 - 0.65)

    def test_get_expected_reward(self):
        """Test Thompson Sampling expected reward calculation"""
        priors = ThompsonPriors()

        # Add some successes
        for i in range(5):
            priors.update_success("research", confidence=0.85)

        # Add some failures
        for i in range(2):
            priors.update_failure("research", confidence=0.60)

        reward = priors.get_expected_reward("research")

        # Should be alpha / (alpha + beta)
        alpha = priors.tool_priors["research"]["alpha"]
        beta = priors.tool_priors["research"]["beta"]
        expected = alpha / (alpha + beta)

        assert reward == pytest.approx(expected, abs=0.01)

    def test_get_expected_reward_new_tool(self):
        """Test expected reward for new tool (uniform prior)"""
        priors = ThompsonPriors()

        reward = priors.get_expected_reward("new_tool")

        # Uniform prior: alpha=1, beta=1
        assert reward == 0.5

    def test_get_uncertainty(self):
        """Test uncertainty calculation (variance)"""
        priors = ThompsonPriors()

        # New tool has high uncertainty
        initial_uncertainty = priors.get_uncertainty("new_tool")

        # Add many observations
        for i in range(20):
            priors.update_success("new_tool", confidence=0.85)

        final_uncertainty = priors.get_uncertainty("new_tool")

        # Uncertainty should decrease with more observations
        assert final_uncertainty < initial_uncertainty

    def test_multiple_tools(self):
        """Test tracking multiple tools independently"""
        priors = ThompsonPriors()

        priors.update_success("answer", 0.90)
        priors.update_success("research", 0.75)
        priors.update_failure("calculate", 0.50)

        # Each tool should have different priors
        assert priors.get_expected_reward("answer") != priors.get_expected_reward("research")
        assert priors.get_expected_reward("research") != priors.get_expected_reward("calculate")


class TestPolicyWeights:
    """Test PolicyWeights dataclass"""

    def test_initialization(self):
        """Test weights initialization"""
        weights = PolicyWeights()

        assert isinstance(weights.adapter_weights, dict)
        assert isinstance(weights.adapter_successes, dict)
        assert isinstance(weights.adapter_total, dict)

    def test_update_success(self):
        """Test updating weights after success"""
        weights = PolicyWeights()

        weights.update("bare", success=True)

        assert weights.adapter_total["bare"] == 1
        assert weights.adapter_successes["bare"] == 1
        assert weights.adapter_weights["bare"] > 0.5  # Laplace smoothing

    def test_update_failure(self):
        """Test updating weights after failure"""
        weights = PolicyWeights()

        weights.update("fast", success=False)

        assert weights.adapter_total["fast"] == 1
        assert weights.adapter_successes["fast"] == 0
        assert weights.adapter_weights["fast"] < 0.5

    def test_multiple_updates(self):
        """Test multiple updates for same adapter"""
        weights = PolicyWeights()

        # 3 successes, 2 failures
        weights.update("fused", success=True)
        weights.update("fused", success=True)
        weights.update("fused", success=False)
        weights.update("fused", success=True)
        weights.update("fused", success=False)

        assert weights.adapter_total["fused"] == 5
        assert weights.adapter_successes["fused"] == 3

        # Weight should reflect success rate with smoothing
        # (3 + 1) / (5 + 2) = 4/7 ≈ 0.571
        assert weights.adapter_weights["fused"] == pytest.approx(4/7, abs=0.01)

    def test_get_weight(self):
        """Test getting weight for adapter"""
        weights = PolicyWeights()

        weights.update("bare", success=True)
        weights.update("bare", success=True)

        weight = weights.get_weight("bare")

        assert 0.0 <= weight <= 1.0

    def test_get_weight_unknown(self):
        """Test getting weight for unknown adapter"""
        weights = PolicyWeights()

        weight = weights.get_weight("unknown")

        assert weight == 1.0  # Default

    def test_get_success_rate(self):
        """Test getting success rate"""
        weights = PolicyWeights()

        weights.update("fast", success=True)
        weights.update("fast", success=True)
        weights.update("fast", success=False)

        success_rate = weights.get_success_rate("fast")

        assert success_rate == pytest.approx(2/3, abs=0.01)

    def test_get_success_rate_unknown(self):
        """Test success rate for unknown adapter"""
        weights = PolicyWeights()

        success_rate = weights.get_success_rate("unknown")

        assert success_rate == 0.5  # Unknown


class TestLearningMetrics:
    """Test LearningMetrics dataclass"""

    def test_initialization(self):
        """Test metrics initialization"""
        metrics = LearningMetrics()

        assert metrics.queries_processed == 0
        assert metrics.avg_confidence == 0.0

    def test_update_metrics(self):
        """Test updating metrics with query results"""
        metrics = LearningMetrics()

        metrics.update(confidence=0.85)

        assert metrics.queries_processed == 1
        assert metrics.avg_confidence == 0.85

    def test_update_running_average(self):
        """Test running average confidence calculation"""
        metrics = LearningMetrics()

        confidences = [0.80, 0.90, 0.70, 0.85]

        for conf in confidences:
            metrics.update(conf)

        expected_avg = sum(confidences) / len(confidences)

        assert metrics.avg_confidence == pytest.approx(expected_avg, abs=0.01)


class TestBackgroundLearner:
    """Test BackgroundLearner class"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return AsyncMock()

    @pytest.fixture
    def learner(self, mock_orchestrator):
        """Create background learner"""
        priors = ThompsonPriors()
        weights = PolicyWeights()

        return BackgroundLearner(
            orchestrator=mock_orchestrator,
            thompson_priors=priors,
            policy_weights=weights,
            update_interval=0.1  # 100ms for testing
        )

    @pytest.mark.asyncio
    async def test_initialization(self, mock_orchestrator):
        """Test learner initialization"""
        priors = ThompsonPriors()
        weights = PolicyWeights()

        learner = BackgroundLearner(
            orchestrator=mock_orchestrator,
            thompson_priors=priors,
            policy_weights=weights,
            update_interval=1.0
        )

        assert learner.orchestrator == mock_orchestrator
        assert learner.update_interval == 1.0
        assert learner.running is False

    @pytest.mark.asyncio
    async def test_start_stop(self, learner):
        """Test starting and stopping background learner"""
        assert learner.running is False

        await learner.start()

        assert learner.running is True
        assert learner.task is not None

        await learner.stop()

        assert learner.running is False

    @pytest.mark.asyncio
    async def test_record_spacetime(self, learner):
        """Test recording spacetime for learning"""
        trace = WeavingTrace(
            query_text="Test",
            tool_confidence=0.85
        )

        spacetime = Spacetime(
            query_text="Test",
            response="Response",
            confidence=0.85,
            trace=trace
        )

        initial_count = len(learner.recent_spacetimes)

        learner.record_spacetime(spacetime)

        assert len(learner.recent_spacetimes) > initial_count

    @pytest.mark.asyncio
    async def test_background_learning_cycle(self, learner):
        """Test background learning cycle runs"""
        # Add some spacetimes
        for i in range(5):
            trace = WeavingTrace(
                query_text=f"Query {i}",
                tool_selected="answer",
                policy_adapter="bare",
                tool_confidence=0.85
            )

            spacetime = Spacetime(
                query_text=f"Query {i}",
                response="Response",
                confidence=0.85,
                trace=trace
            )

            learner.record_spacetime(spacetime)

        # Start background learner
        await learner.start()

        # Wait for at least one update cycle
        await asyncio.sleep(0.2)

        # Stop learner
        await learner.stop()

        # Thompson priors should have been updated
        assert learner.thompson_priors.tool_priors["answer"]["alpha"] > 1.0


class TestFullLearningEngine:
    """Test FullLearningEngine class"""

    @pytest.fixture
    def config(self):
        """Create test config"""
        return Config.fast()

    @pytest.fixture
    def shards(self):
        """Create test shards"""
        return [
            MemoryShard(
                id="shard1",
                content="Test content",
                metadata={"tags": ["test"]}
            )
        ]

    def test_initialization(self, config, shards):
        """Test engine initialization"""
        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=True,
            learning_update_interval=60.0
        )

        assert engine.cfg == config
        assert len(engine.shards) == 1
        assert isinstance(engine.thompson_priors, ThompsonPriors)
        assert isinstance(engine.policy_weights, PolicyWeights)
        assert engine.enable_background_learning is True

    def test_initialization_no_background(self, config, shards):
        """Test initialization without background learning"""
        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        )

        assert engine.enable_background_learning is False

    @pytest.mark.asyncio
    async def test_get_learning_statistics(self, config, shards):
        """Test getting learning statistics"""
        async with FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        ) as engine:
            # Mock components to avoid full initialization
            engine.hot_pattern_engine = Mock()
            engine.hot_pattern_engine.hot_tracker = Mock()
            engine.hot_pattern_engine.hot_tracker.get_hot_patterns = Mock(return_value=[])
            engine.hot_pattern_engine.learning_engine = Mock()
            engine.hot_pattern_engine.learning_engine.pattern_learner = Mock()
            engine.hot_pattern_engine.learning_engine.pattern_learner.get_hot_patterns = Mock(
                return_value=[]
            )

            stats = engine.get_learning_statistics()

            assert "queries_processed" in stats
            assert "avg_confidence" in stats
            assert "thompson_priors" in stats
            assert "policy_weights" in stats

    def test_thompson_priors_integration(self, config, shards):
        """Test Thompson priors are tracked"""
        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        )

        # Update priors
        engine.thompson_priors.update_success("answer", 0.90)

        # Should be reflected in statistics
        expected_reward = engine.thompson_priors.get_expected_reward("answer")

        assert expected_reward > 0.5

    def test_policy_weights_integration(self, config, shards):
        """Test policy weights are tracked"""
        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        )

        # Update weights
        engine.policy_weights.update("bare", success=True)
        engine.policy_weights.update("bare", success=True)
        engine.policy_weights.update("bare", success=False)

        weight = engine.policy_weights.get_weight("bare")

        assert 0.0 < weight < 1.0

    def test_metrics_tracking(self, config, shards):
        """Test metrics are updated"""
        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        )

        # Update metrics
        engine.metrics.update(0.85)
        engine.metrics.update(0.90)

        assert engine.metrics.queries_processed == 2
        assert engine.metrics.avg_confidence == pytest.approx(0.875, abs=0.01)

    def test_save_learning_state(self, config, shards, tmp_path):
        """Test saving learning state to disk"""
        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        )

        # Add some learning data
        engine.thompson_priors.update_success("answer", 0.90)
        engine.policy_weights.update("bare", success=True)
        engine.metrics.update(0.85)

        # Save state
        save_path = str(tmp_path / "learning_state")
        engine.save_learning_state(save_path)

        # Verify files created
        assert (tmp_path / "learning_state" / "thompson_priors.json").exists()
        assert (tmp_path / "learning_state" / "policy_weights.json").exists()
        assert (tmp_path / "learning_state" / "metrics.json").exists()

    def test_get_scratchpad_history(self, config, shards):
        """Test retrieving scratchpad history"""
        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        )

        # Add entry to scratchpad
        engine.scratchpad.add_entry(
            thought="Test thought",
            action="test_action",
            observation="Test observation",
            score=0.85
        )

        history = engine.get_scratchpad_history()

        assert len(history) == 1


@pytest.mark.asyncio
async def test_weave_with_full_learning_convenience():
    """Test convenience function signature"""
    # Just test the function exists and has correct signature
    assert callable(weave_with_full_learning)


class TestThompsonSamplingUpdates:
    """Test Thompson Sampling updates from outcomes"""

    def test_successful_tool_increases_alpha(self):
        """Test successful tool use increases alpha"""
        priors = ThompsonPriors()

        initial_alpha = priors.tool_priors["answer"]["alpha"]

        priors.update_success("answer", confidence=0.85)

        assert priors.tool_priors["answer"]["alpha"] == initial_alpha + 0.85

    def test_unsuccessful_tool_increases_beta(self):
        """Test unsuccessful tool use increases beta"""
        priors = ThompsonPriors()

        initial_beta = priors.tool_priors["answer"]["beta"]

        priors.update_failure("answer", confidence=0.65)

        assert priors.tool_priors["answer"]["beta"] == initial_beta + 0.35

    def test_mixed_outcomes(self):
        """Test mixed outcomes produce reasonable priors"""
        priors = ThompsonPriors()

        # 7 successes
        for i in range(7):
            priors.update_success("research", confidence=0.85)

        # 3 failures
        for i in range(3):
            priors.update_failure("research", confidence=0.60)

        expected_reward = priors.get_expected_reward("research")

        # Should favor success (7 vs 3)
        assert expected_reward > 0.5


class TestPolicyWeightUpdates:
    """Test policy weight updates from outcomes"""

    def test_laplace_smoothing(self):
        """Test Laplace smoothing in weight calculation"""
        weights = PolicyWeights()

        # Single success
        weights.update("bare", success=True)

        # Weight = (1 + 1) / (1 + 2) = 2/3
        assert weights.adapter_weights["bare"] == pytest.approx(2/3, abs=0.01)

    def test_weights_converge_with_data(self):
        """Test weights converge to true success rate"""
        weights = PolicyWeights()

        # 80% success rate
        for i in range(80):
            weights.update("fast", success=True)
        for i in range(20):
            weights.update("fast", success=False)

        # Should converge close to 0.80
        assert weights.adapter_weights["fast"] == pytest.approx(0.80, abs=0.05)


class TestLearningIntegration:
    """Integration tests for complete learning system"""

    @pytest.mark.asyncio
    async def test_complete_learning_cycle(self):
        """Test complete learning cycle"""
        config = Config.fast()
        shards = [
            MemoryShard(id="s1", content="Test", metadata={})
        ]

        async with FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False  # Disable for testing
        ) as engine:
            # Mock hot pattern engine to avoid full initialization
            engine.hot_pattern_engine = Mock()
            engine.hot_pattern_engine.weave = AsyncMock()

            # Mock spacetime result
            trace = WeavingTrace(
                query_text="Test",
                tool_selected="answer",
                policy_adapter="bare",
                tool_confidence=0.85
            )

            spacetime = Spacetime(
                query_text="Test",
                response="Response",
                confidence=0.85,
                trace=trace
            )

            engine.hot_pattern_engine.weave.return_value = spacetime

            # Mock refiner
            engine.advanced_refiner = None

            # Process query
            result = await engine.weave(
                Query(text="Test query"),
                enable_refinement=False
            )

            # Learning should have occurred
            assert engine.metrics.queries_processed == 1
            assert engine.thompson_priors.tool_priors["answer"]["alpha"] > 1.0
            assert engine.policy_weights.adapter_total["bare"] == 1

    def test_statistics_completeness(self):
        """Test statistics include all components"""
        config = Config.fast()
        shards = []

        engine = FullLearningEngine(
            cfg=config,
            shards=shards,
            enable_background_learning=False
        )

        # Mock components
        engine.hot_pattern_engine = Mock()
        engine.hot_pattern_engine.hot_tracker = Mock()
        engine.hot_pattern_engine.hot_tracker.get_hot_patterns = Mock(return_value=[])
        engine.hot_pattern_engine.learning_engine = Mock()
        engine.hot_pattern_engine.learning_engine.pattern_learner = Mock()
        engine.hot_pattern_engine.learning_engine.pattern_learner.get_hot_patterns = Mock(
            return_value=[]
        )
        engine.advanced_refiner = Mock()
        engine.advanced_refiner.get_strategy_statistics = Mock(return_value={})

        stats = engine.get_learning_statistics()

        # Should include all major components
        assert "thompson_priors" in stats
        assert "policy_weights" in stats
        assert "refinement_strategies" in stats
        assert "background_learning" in stats
        assert "hot_patterns" in stats
        assert "learned_patterns" in stats
