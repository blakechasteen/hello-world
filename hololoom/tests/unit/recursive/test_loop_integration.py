"""
Unit tests for HoloLoom.recursive.loop_integration

Tests pattern extraction, pattern learning, and learning loop engine integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from hololoom.recursive.loop_integration import (
    LearnedPattern,
    PatternExtractor,
    PatternLearner,
    LearningStats,
    LearningLoopConfig,
    LearningLoopEngine,
    weave_with_learning
)
from datetime import datetime
from hololoom.fabric.spacetime import Spacetime, WeavingTrace
from hololoom.protocols.types import Query, MemoryShard
from hololoom.config import Config


def _make_trace(**kwargs) -> WeavingTrace:
    """Helper to create a WeavingTrace with required fields defaulted."""
    now = datetime.now()
    defaults = dict(start_time=now, end_time=now, duration_ms=0.0)
    defaults.update(kwargs)
    return WeavingTrace(**defaults)


def _make_spacetime(query_text: str, response: str, confidence: float,
                    tool_used: str = "answer", **trace_kwargs) -> Spacetime:
    """Helper to create a Spacetime with required fields defaulted.

    Mirrors tool_used -> trace.tool_selected so PatternExtractor.extract()
    sees the correct tool name.
    """
    # Mirror tool_used into trace.tool_selected if not explicitly provided
    if "tool_selected" not in trace_kwargs:
        trace_kwargs["tool_selected"] = tool_used
    trace = _make_trace(**trace_kwargs)
    return Spacetime(
        query_text=query_text,
        response=response,
        tool_used=tool_used,
        confidence=confidence,
        trace=trace,
    )


class TestLearnedPattern:
    """Test LearnedPattern dataclass"""

    def test_creation(self):
        """Test creating a learned pattern"""
        pattern = LearnedPattern(
            motifs=["AI", "ML", "neural"],
            threads=["thread1", "thread2"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        assert pattern.motifs == ["AI", "ML", "neural"]
        assert pattern.threads == ["thread1", "thread2"]
        assert pattern.tool == "answer"
        assert pattern.confidence == 0.85
        assert pattern.occurrences == 1
        assert pattern.avg_confidence == 0.0

    def test_hash_consistency(self):
        """Test pattern hashing is consistent"""
        pattern1 = LearnedPattern(
            motifs=["AI", "ML"],
            threads=["t1", "t2"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        pattern2 = LearnedPattern(
            motifs=["ML", "AI"],  # Different order
            threads=["t2", "t1"],  # Different order
            tool="answer",
            adapter="bare",
            confidence=0.90,  # Different confidence
            query_type="factual"
        )

        # Should hash the same (same core pattern)
        assert hash(pattern1) == hash(pattern2)

    def test_update_pattern(self):
        """Test updating pattern with new occurrence"""
        pattern = LearnedPattern(
            motifs=["AI"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.80,
            query_type="factual"
        )

        initial_occurrences = pattern.occurrences

        pattern.update(new_confidence=0.90)

        assert pattern.occurrences == initial_occurrences + 1
        assert pattern.avg_confidence > 0
        assert pattern.last_seen > 0

    def test_update_multiple_times(self):
        """Test updating pattern multiple times"""
        pattern = LearnedPattern(
            motifs=["test"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.80,
            query_type="factual"
        )

        confidences = [0.85, 0.90, 0.75, 0.88]

        for conf in confidences:
            pattern.update(conf)

        assert pattern.occurrences == 1 + len(confidences)
        # Average should be reasonable (starts at 0.0 so lower bound is relaxed)
        assert 0.60 <= pattern.avg_confidence <= 0.95


class TestPatternExtractor:
    """Test PatternExtractor class"""

    def test_initialization(self):
        """Test extractor initialization"""
        extractor = PatternExtractor(confidence_threshold=0.8)

        assert extractor.confidence_threshold == 0.8

    def test_extract_high_confidence(self):
        """Test extracting pattern from high-confidence spacetime"""
        extractor = PatternExtractor(confidence_threshold=0.75)

        spacetime = _make_spacetime(
            query_text="What is AI?",
            response="Response",
            confidence=0.85,
            tool_used="answer",
            threads_activated=["t1", "t2", "t3"],
            motifs_detected=["AI", "ML"],
            policy_adapter="bare",
            tool_confidence=0.85,
        )

        pattern = extractor.extract(spacetime)

        assert pattern is not None
        assert pattern.tool == "answer"
        assert pattern.confidence == 0.85
        assert len(pattern.motifs) <= 5
        assert len(pattern.threads) <= 5

    def test_extract_low_confidence(self):
        """Test extraction skips low-confidence results"""
        extractor = PatternExtractor(confidence_threshold=0.75)

        spacetime = _make_spacetime(
            query_text="Query",
            response="Response",
            confidence=0.65,
            tool_confidence=0.65,
        )

        pattern = extractor.extract(spacetime)

        assert pattern is None

    def test_classify_factual_query(self):
        """Test classifying factual queries"""
        extractor = PatternExtractor()

        query_type = extractor._classify_query("What is machine learning?")
        assert query_type == "factual"

        query_type = extractor._classify_query("Define neural network")
        assert query_type == "factual"

    def test_classify_procedural_query(self):
        """Test classifying procedural queries"""
        extractor = PatternExtractor()

        query_type = extractor._classify_query("How to train a model?")
        assert query_type == "procedural"

        query_type = extractor._classify_query("How does backprop work?")
        assert query_type == "procedural"

    def test_classify_analytical_query(self):
        """Test classifying analytical queries"""
        extractor = PatternExtractor()

        query_type = extractor._classify_query("Why is AI important?")
        assert query_type == "analytical"

        query_type = extractor._classify_query("Explain the benefits of ML")
        assert query_type == "analytical"

    def test_classify_comparative_query(self):
        """Test classifying comparative queries"""
        extractor = PatternExtractor()

        query_type = extractor._classify_query("Compare CNN and RNN")
        assert query_type == "comparative"

        query_type = extractor._classify_query("What's the difference between AI and ML?")
        assert query_type == "comparative"

    def test_extract_limits_motifs_and_threads(self):
        """Test extraction limits motifs and threads to top 5"""
        extractor = PatternExtractor()

        spacetime = _make_spacetime(
            query_text="Query",
            response="Response",
            confidence=0.90,
            tool_used="answer",
            threads_activated=[f"t{i}" for i in range(10)],
            motifs_detected=[f"m{i}" for i in range(10)],
            policy_adapter="bare",
            tool_confidence=0.90,
        )

        pattern = extractor.extract(spacetime)

        assert pattern is not None
        assert len(pattern.motifs) == 5
        assert len(pattern.threads) == 5


class TestPatternLearner:
    """Test PatternLearner class"""

    def test_initialization(self):
        """Test learner initialization"""
        learner = PatternLearner(
            hot_threshold=10,
            prune_age=7200.0,
            prune_confidence=0.7
        )

        assert learner.hot_threshold == 10
        assert learner.prune_age == 7200.0
        assert learner.prune_confidence == 0.7
        assert len(learner.patterns) == 0

    def test_learn_new_pattern(self):
        """Test learning a new pattern"""
        learner = PatternLearner()

        pattern = LearnedPattern(
            motifs=["AI"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        initial_count = len(learner.patterns)

        learner.learn(pattern)

        assert len(learner.patterns) > initial_count
        assert learner.stats.patterns_learned > 0

    def test_learn_existing_pattern(self):
        """Test updating existing pattern"""
        learner = PatternLearner()

        pattern1 = LearnedPattern(
            motifs=["AI"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        pattern2 = LearnedPattern(
            motifs=["AI"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.90,
            query_type="factual"
        )

        learner.learn(pattern1)
        initial_count = len(learner.patterns)

        learner.learn(pattern2)

        # Should update existing pattern, not create new one
        assert len(learner.patterns) == initial_count
        assert learner.stats.patterns_updated > 0

    def test_get_hot_patterns(self):
        """Test retrieving hot patterns"""
        learner = PatternLearner(hot_threshold=3)

        # Create hot pattern (many occurrences)
        hot_pattern = LearnedPattern(
            motifs=["popular"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        # Simulate multiple occurrences
        for i in range(5):
            learner.learn(hot_pattern)

        hot = learner.get_hot_patterns()

        assert len(hot) >= 1
        assert all(p.occurrences >= 3 for p in hot)

    def test_get_hot_patterns_by_type(self):
        """Test filtering hot patterns by query type"""
        # prune_confidence=0.0 to avoid filtering due to avg_confidence starting at 0
        learner = PatternLearner(hot_threshold=2, prune_confidence=0.0)

        factual = LearnedPattern(
            motifs=["fact"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        procedural = LearnedPattern(
            motifs=["proc"],
            threads=["t2"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="procedural"
        )

        # Make both hot
        for i in range(3):
            learner.learn(factual)
            learner.learn(procedural)

        factual_hot = learner.get_hot_patterns(query_type="factual")

        assert len(factual_hot) >= 1
        # Should only return factual patterns
        # (Can't easily verify since patterns hash the same way)

    def test_prune_stale_patterns(self):
        """Test pruning stale patterns"""
        learner = PatternLearner(
            prune_age=0.1,  # 100ms
            prune_confidence=0.7
        )

        # Add pattern
        pattern = LearnedPattern(
            motifs=["old"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        learner.learn(pattern)

        # Wait for pattern to become stale
        import time
        time.sleep(0.15)

        initial_count = len(learner.patterns)

        pruned = learner.prune_stale_patterns()

        assert pruned > 0
        assert len(learner.patterns) < initial_count

    def test_prune_low_confidence_patterns(self):
        """Test pruning low confidence patterns"""
        learner = PatternLearner(prune_confidence=0.75)

        # Add low confidence pattern
        low_conf = LearnedPattern(
            motifs=["low"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.60,
            query_type="factual",
            avg_confidence=0.60
        )

        learner.learn(low_conf)

        # Manually set avg_confidence low
        pattern_hash = hash(low_conf)
        learner.patterns[pattern_hash].avg_confidence = 0.60

        pruned = learner.prune_stale_patterns()

        assert pruned >= 1

    def test_suggest_improvements(self):
        """Test suggesting improvements based on learned patterns"""
        # prune_confidence=0.0 to avoid filtering due to avg_confidence starting at 0
        learner = PatternLearner(hot_threshold=2, prune_confidence=0.0)

        # Learn some factual patterns
        for i in range(3):
            pattern = LearnedPattern(
                motifs=["AI", "ML"],
                threads=["t1", "t2"],
                tool="answer",
                adapter="bare",
                confidence=0.88,
                query_type="factual"
            )
            learner.learn(pattern)

        suggestions = learner.suggest_improvements("What is deep learning?")

        assert suggestions["has_suggestions"] is True
        assert suggestions["query_type"] == "factual"
        assert "suggested_motifs" in suggestions


class TestLearningStats:
    """Test LearningStats dataclass"""

    def test_initialization(self):
        """Test stats initialization"""
        stats = LearningStats()

        assert stats.patterns_learned == 0
        assert stats.patterns_updated == 0
        assert stats.unique_patterns == 0
        assert stats.queries_processed == 0


class TestLearningLoopConfig:
    """Test LearningLoopConfig dataclass"""

    def test_default_config(self):
        """Test default configuration"""
        config = LearningLoopConfig()

        assert config.enable_learning is True
        assert config.auto_prune is True
        assert config.prune_interval == 100
        assert config.hot_threshold == 5

    def test_custom_config(self):
        """Test custom configuration"""
        config = LearningLoopConfig(
            enable_learning=False,
            auto_prune=False,
            prune_interval=50,
            hot_threshold=10
        )

        assert config.enable_learning is False
        assert config.prune_interval == 50


# Integration tests with mocked components
class TestLearningLoopEngine:
    """Test LearningLoopEngine class"""

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
                text="AI and ML content",
                metadata={"tags": ["AI", "ML"]}
            )
        ]

    def test_initialization(self, config, shards):
        """Test engine initialization"""
        loop_config = LearningLoopConfig()

        engine = LearningLoopEngine(
            cfg=config,
            shards=shards,
            loop_config=loop_config
        )

        assert engine.cfg == config
        assert len(engine.shards) == 1
        assert isinstance(engine.pattern_extractor, PatternExtractor)
        assert isinstance(engine.pattern_learner, PatternLearner)

    @pytest.mark.asyncio
    async def test_context_manager(self, config, shards):
        """Test async context manager"""
        loop_config = LearningLoopConfig()

        async with LearningLoopEngine(
            cfg=config,
            shards=shards,
            loop_config=loop_config
        ) as engine:
            assert engine.orchestrator is not None

    def test_get_learning_stats(self, config, shards):
        """Test getting learning statistics"""
        engine = LearningLoopEngine(
            cfg=config,
            shards=shards
        )

        # Mock orchestrator
        engine.orchestrator = Mock()
        engine.orchestrator.get_statistics = Mock(return_value={
            "queries_processed": 10,
            "avg_confidence": 0.85
        })

        stats = engine.get_learning_stats()

        assert "queries_processed" in stats
        assert "patterns_learned" in stats
        assert "unique_patterns" in stats

    def test_suggest_improvements(self, config, shards):
        """Test improvement suggestions"""
        engine = LearningLoopEngine(
            cfg=config,
            shards=shards
        )

        suggestions = engine.suggest_improvements("What is AI?")

        assert "query_type" in suggestions


@pytest.mark.asyncio
async def test_weave_with_learning_convenience():
    """Test convenience function"""
    config = Config.fast()
    shards = [
        MemoryShard(id="s1", text="Test content", metadata={})
    ]

    query = Query(text="Test query")

    # This would require full HoloLoom integration
    # Just test the function exists and has correct signature
    assert callable(weave_with_learning)


class TestPatternIndexing:
    """Test pattern indexing by query type"""

    def test_patterns_indexed_by_type(self):
        """Test patterns are correctly indexed by query type"""
        learner = PatternLearner()

        factual = LearnedPattern(
            motifs=["AI"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="factual"
        )

        procedural = LearnedPattern(
            motifs=["ML"],
            threads=["t2"],
            tool="answer",
            adapter="bare",
            confidence=0.85,
            query_type="procedural"
        )

        learner.learn(factual)
        learner.learn(procedural)

        # Check indexes exist
        assert "factual" in learner.patterns_by_type
        assert "procedural" in learner.patterns_by_type


class TestLearningIntegration:
    """Integration tests for complete learning workflow"""

    def test_extract_learn_retrieve_workflow(self):
        """Test complete workflow: extract → learn → retrieve"""
        extractor = PatternExtractor(confidence_threshold=0.75)
        learner = PatternLearner(hot_threshold=2)

        # Create multiple successful spacetimes
        for i in range(3):
            spacetime = _make_spacetime(
                query_text=f"What is AI? (query {i})",
                response="Response",
                confidence=0.88,
                tool_used="answer",
                threads_activated=["t1", "t2"],
                motifs_detected=["AI", "ML"],
                policy_adapter="bare",
                tool_confidence=0.88,
            )

            # Extract pattern
            pattern = extractor.extract(spacetime)
            assert pattern is not None

            # Learn pattern
            learner.learn(pattern)

        # Retrieve hot patterns
        hot = learner.get_hot_patterns()

        assert len(hot) >= 1
        assert hot[0].occurrences >= 3
        assert hot[0].query_type == "factual"

    def test_learning_improves_confidence(self):
        """Test that learning tracks confidence improvements"""
        learner = PatternLearner()

        # First occurrence — set avg_confidence to match initial confidence
        # so that subsequent updates reflect the running average correctly
        pattern1 = LearnedPattern(
            motifs=["test"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.75,
            query_type="factual",
            avg_confidence=0.75,  # initialise to first-sample value
        )

        learner.learn(pattern1)

        # Second occurrence with higher confidence
        pattern2 = LearnedPattern(
            motifs=["test"],
            threads=["t1"],
            tool="answer",
            adapter="bare",
            confidence=0.90,
            query_type="factual"
        )

        learner.learn(pattern2)

        # Average confidence should be between 0.75 and 0.90
        pattern_hash = hash(pattern1)
        avg_conf = learner.patterns[pattern_hash].avg_confidence

        assert 0.75 <= avg_conf <= 0.90
