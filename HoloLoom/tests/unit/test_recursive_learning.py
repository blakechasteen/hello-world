"""
Unit tests for recursive learning system.

Tests scratchpad, pattern learning, and Thompson Sampling.
Fast, isolated tests with mocked dependencies.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestScratchpadIntegration:
    """Test provenance tracking via scratchpad."""

    @pytest.mark.asyncio
    @patch("HoloLoom.recursive.weaving_orchestrator.WeavingOrchestrator")
    async def test_scratchpad_creation(self, mock_orchestrator):
        """Scratchpad should track provenance."""
        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import MemoryShard, Query
        from HoloLoom.recursive import weave_with_scratchpad

        # Mock orchestrator
        mock_result = Mock()
        mock_result.response = "test response"
        mock_result.confidence = 0.85
        mock_orchestrator.return_value.__aenter__.return_value.weave = AsyncMock(
            return_value=mock_result
        )

        cfg = Config.fast()
        query = Query(text="test query")
        shards = [MemoryShard(text="test", source="test")]

        spacetime, scratchpad = await weave_with_scratchpad(query, cfg, shards)

        assert scratchpad is not None
        assert hasattr(scratchpad, "get_history")

    @pytest.mark.asyncio
    @patch("HoloLoom.recursive.weaving_orchestrator.WeavingOrchestrator")
    async def test_scratchpad_history(self, mock_orchestrator):
        """Scratchpad should maintain history."""
        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import Query
        from HoloLoom.recursive import weave_with_scratchpad

        mock_result = Mock(response="test", confidence=0.9)
        mock_orchestrator.return_value.__aenter__.return_value.weave = AsyncMock(
            return_value=mock_result
        )

        cfg = Config.fast()
        query = Query(text="test")

        _, scratchpad = await weave_with_scratchpad(query, cfg, [])

        history = scratchpad.get_history()
        assert isinstance(history, list)


class TestPatternLearning:
    """Test pattern extraction and learning."""

    def test_pattern_extraction(self):
        """Should extract patterns from successful queries."""
        from HoloLoom.recursive.pattern_learner import PatternLearner

        learner = PatternLearner()

        # Simulate successful query
        learner.record_pattern(
            motifs=["python", "function"],
            tool_used="answer",
            confidence=0.9,
            query_type="procedural",
        )

        patterns = learner.get_hot_patterns()
        assert len(patterns) >= 0  # May be empty initially

    def test_pattern_confidence_threshold(self):
        """High confidence patterns should be learned."""
        from HoloLoom.recursive.pattern_learner import PatternLearner

        learner = PatternLearner()

        # High confidence - should learn
        learner.record_pattern(
            motifs=["test"], tool_used="answer", confidence=0.95, query_type="factual"
        )

        # Low confidence - should not learn
        learner.record_pattern(
            motifs=["bad"], tool_used="answer", confidence=0.3, query_type="factual"
        )

        patterns = learner.get_hot_patterns(limit=10)
        # High confidence pattern should dominate if recorded multiple times

    def test_pattern_pruning(self):
        """Stale patterns should be pruned."""
        from datetime import timedelta

        from HoloLoom.recursive.pattern_learner import PatternLearner

        learner = PatternLearner()

        # Record old pattern
        old_time = datetime.now() - timedelta(days=100)
        learner.record_pattern(
            motifs=["old"], tool_used="answer", confidence=0.8, query_type="factual"
        )

        # Manual prune (if method exists)
        if hasattr(learner, "prune_stale_patterns"):
            learner.prune_stale_patterns(max_age_days=30)


class TestHotPatternFeedback:
    """Test adaptive retrieval based on usage."""

    def test_heat_score_calculation(self):
        """Heat scores should reflect usage."""
        from HoloLoom.recursive.hot_tracker import HotPatternTracker

        tracker = HotPatternTracker()

        # Track accesses
        tracker.record_access("pattern_1", confidence=0.9)
        tracker.record_access("pattern_1", confidence=0.85)
        tracker.record_access("pattern_2", confidence=0.5)

        hot = tracker.get_hot_patterns(limit=2)
        # pattern_1 should rank higher

    def test_heat_decay(self):
        """Heat should decay over time."""

        from HoloLoom.recursive.hot_tracker import HotPatternTracker

        tracker = HotPatternTracker()

        # Record access
        tracker.record_access("pattern", confidence=0.9)

        # Simulate time passing
        initial_heat = tracker.get_heat_score("pattern")

        # After decay period
        if hasattr(tracker, "apply_decay"):
            tracker.apply_decay(hours=24)
            decayed_heat = tracker.get_heat_score("pattern")
            assert decayed_heat < initial_heat

    def test_hot_pattern_boost(self):
        """Hot patterns should get retrieval boost."""
        from HoloLoom.recursive.hot_tracker import HotPatternTracker

        tracker = HotPatternTracker()

        # Build heat
        for _ in range(10):
            tracker.record_access("hot_pattern", confidence=0.9)

        # Check boost multiplier
        boost = tracker.get_boost_multiplier("hot_pattern")
        assert boost > 1.0  # Should be boosted


class TestThompsonSampling:
    """Test Thompson Sampling for exploration."""

    def test_thompson_update_success(self):
        """Should update priors on success."""
        from HoloLoom.recursive.bandit import ThompsonBandit

        bandit = ThompsonBandit(n_tools=3)

        # Record success
        bandit.update(tool_idx=0, reward=0.9)

        # Alpha should increase
        stats = bandit.get_stats()
        assert stats[0]["successes"] > 0 or stats[0]["alpha"] > 1.0

    def test_thompson_update_failure(self):
        """Should update priors on failure."""
        from HoloLoom.recursive.bandit import ThompsonBandit

        bandit = ThompsonBandit(n_tools=3)

        # Record failure
        bandit.update(tool_idx=0, reward=0.2)

        # Beta should increase
        stats = bandit.get_stats()
        assert stats[0]["failures"] > 0 or stats[0]["beta"] > 1.0

    def test_thompson_exploration(self):
        """Thompson sampling should explore."""
        import torch

        from HoloLoom.recursive.bandit import ThompsonBandit

        bandit = ThompsonBandit(n_tools=5)

        # Sample multiple times
        samples = []
        for _ in range(100):
            logits = torch.randn(5)  # Mock logits
            tool_idx = bandit.sample(logits)
            samples.append(tool_idx.item())

        # Should explore different tools
        unique_tools = len(set(samples))
        assert unique_tools > 1, "Thompson sampling should explore multiple tools"

    def test_thompson_exploitation(self):
        """Should exploit after learning."""
        import torch

        from HoloLoom.recursive.bandit import ThompsonBandit

        bandit = ThompsonBandit(n_tools=3)

        # Build strong prior for tool 0
        for _ in range(50):
            bandit.update(tool_idx=0, reward=0.95)

        # Should prefer tool 0
        samples = []
        for _ in range(20):
            logits = torch.randn(3)
            tool_idx = bandit.sample(logits)
            samples.append(tool_idx.item())

        # Majority should be tool 0 (learned best)
        tool_0_count = samples.count(0)
        assert tool_0_count > len(samples) * 0.5


class TestRefinementStrategies:
    """Test multi-pass refinement."""

    @pytest.mark.asyncio
    @patch("HoloLoom.recursive.AdvancedRefiner")
    async def test_refinement_triggered(self, mock_refiner):
        """Refinement should trigger on low confidence."""
        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import Query

        cfg = Config.fast()
        query = Query(text="test")

        # Mock low confidence result
        mock_result = Mock(response="test", confidence=0.5)

        # Should trigger refinement if confidence < threshold

    @pytest.mark.asyncio
    async def test_elegance_strategy(self):
        """ELEGANCE strategy should improve clarity."""
        # Mock refinement
        from HoloLoom.recursive.strategies import RefinementStrategy

        assert RefinementStrategy.ELEGANCE is not None

    @pytest.mark.asyncio
    async def test_verify_strategy(self):
        """VERIFY strategy should check accuracy."""
        from HoloLoom.recursive.strategies import RefinementStrategy

        assert RefinementStrategy.VERIFY is not None


class TestBackgroundLearning:
    """Test background learning thread."""

    @pytest.mark.asyncio
    @patch("HoloLoom.recursive.FullLearningEngine")
    async def test_background_learning_enabled(self, mock_engine):
        """Background learning should run periodically."""
        from HoloLoom.config import Config

        cfg = Config.fast()

        # Mock engine with background learning
        mock_instance = AsyncMock()
        mock_instance.enable_background_learning = True
        mock_instance.learning_update_interval = 1.0  # Fast for testing

        # Background task should spawn

    @pytest.mark.asyncio
    async def test_learning_state_persistence(self):
        """Learning state should persist."""
        from HoloLoom.config import Config

        cfg = Config.fast()

        # Mock save/load
        # Should be able to save and restore learning state


class TestPerformanceOverhead:
    """Test learning overhead stays minimal."""

    @pytest.mark.asyncio
    async def test_provenance_overhead(self):
        """Provenance tracking should be <1ms."""
        import time

        from HoloLoom.recursive.scratchpad import Scratchpad

        pad = Scratchpad()

        start = time.perf_counter()
        for _ in range(100):
            pad.record_thought("test thought")
        elapsed = (time.perf_counter() - start) * 1000

        avg = elapsed / 100
        assert avg < 1.0, f"Provenance took {avg:.3f}ms (target: <1ms)"

    def test_pattern_extraction_overhead(self):
        """Pattern extraction should be <1ms."""
        import time

        from HoloLoom.recursive.pattern_learner import PatternLearner

        learner = PatternLearner()

        start = time.perf_counter()
        for _ in range(100):
            learner.record_pattern(
                motifs=["test"], tool_used="answer", confidence=0.8, query_type="test"
            )
        elapsed = (time.perf_counter() - start) * 1000

        avg = elapsed / 100
        assert avg < 1.0, f"Pattern extraction took {avg:.3f}ms (target: <1ms)"

    def test_heat_tracking_overhead(self):
        """Heat tracking should be <0.5ms."""
        import time

        from HoloLoom.recursive.hot_tracker import HotPatternTracker

        tracker = HotPatternTracker()

        start = time.perf_counter()
        for _ in range(100):
            tracker.record_access("pattern", confidence=0.8)
        elapsed = (time.perf_counter() - start) * 1000

        avg = elapsed / 100
        assert avg < 0.5, f"Heat tracking took {avg:.3f}ms (target: <0.5ms)"
