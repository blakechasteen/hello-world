"""
Comprehensive tests for HoloLoom Phase 5: Anticipatory Retrieval.

Tests cover:
- QueryTypeClassifier
- QueryPatternTracker
- FollowUpPredictor
- AnticipatoryFetcher
- SessionContext
- AnticipatoryRetrieval integration
- Edge cases and error handling
"""

import pytest
import asyncio
import time
from typing import List, Optional, Any
from unittest.mock import Mock, AsyncMock, patch

from HoloLoom.memory.anticipatory_retrieval import (
    # Enums
    QueryType,

    # Data structures
    QueryPattern,
    SessionContext,
    PredictedQuery,
    PrefetchResult,
    AnticipatoryConfig,

    # Classes
    QueryTypeClassifier,
    QueryPatternTracker,
    FollowUpPredictor,
    AnticipatoryFetcher,
    AnticipatoryRetrieval,

    # Factory functions
    create_anticipatory_retrieval,
    create_session_context,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Standard config for testing."""
    return AnticipatoryConfig(
        min_prediction_confidence=0.3,
        max_predictions=3,
        prefetch_enabled=True,
        prefetch_k=10,
        max_prefetch_total=25,
        pattern_learning_enabled=True,
        min_pattern_occurrences=3,
    )


@pytest.fixture
def classifier():
    """Query type classifier."""
    return QueryTypeClassifier()


@pytest.fixture
def pattern_tracker(config):
    """Pattern tracker with config."""
    return QueryPatternTracker(config)


@pytest.fixture
def predictor(pattern_tracker, classifier, config):
    """Follow-up predictor."""
    return FollowUpPredictor(pattern_tracker, classifier, config)


@pytest.fixture
def fetcher(predictor, config):
    """Anticipatory fetcher."""
    return AnticipatoryFetcher(predictor, config)


@pytest.fixture
def session():
    """Fresh session context."""
    return SessionContext(session_id="test_session")


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing prefetch."""
    retriever = Mock()
    retriever.retrieve_ids = AsyncMock(return_value=["mem1", "mem2", "mem3"])
    retriever.retrieve = AsyncMock(return_value=[
        (Mock(id="mem1"), 0.9),
        (Mock(id="mem2"), 0.8),
        (Mock(id="mem3"), 0.7),
    ])
    return retriever


# ============================================================================
# Test QueryTypeClassifier
# ============================================================================

class TestQueryTypeClassifier:
    """Tests for query type classification."""

    def test_definition_queries(self, classifier):
        """Test DEFINITION type classification."""
        queries = [
            "What is Thompson Sampling?",
            "Define machine learning",
            "What are neural networks?",
            "What does API mean?",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.DEFINITION

    def test_explanation_queries(self, classifier):
        """Test EXPLANATION type classification."""
        queries = [
            "Explain Thompson Sampling",
            "How does backpropagation work?",
            "Why does gradient descent converge?",
            "Tell me about deep learning",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.EXPLANATION

    def test_example_queries(self, classifier):
        """Test EXAMPLE type classification."""
        queries = [
            "Show me an example of recursion",
            "Give me an example",
            "Can you show how to use this?",
            "Example of decorators",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.EXAMPLE

    def test_comparison_queries(self, classifier):
        """Test COMPARISON type classification."""
        queries = [
            "Compare Python and JavaScript",
            "Thompson Sampling vs UCB",
            "Difference between REST and GraphQL",
            "How does React differ from Vue?",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.COMPARISON

    def test_application_queries(self, classifier):
        """Test APPLICATION type classification."""
        queries = [
            "When should I use async/await?",
            "Where can I apply reinforcement learning?",
            "How to use decorators?",
            "Use case for microservices",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.APPLICATION

    def test_implementation_queries(self, classifier):
        """Test IMPLEMENTATION type classification."""
        queries = [
            "How to implement Thompson Sampling?",
            "Code for quicksort",
            "Write a binary tree class",
            "Show me code for authentication",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.IMPLEMENTATION

    def test_tradeoffs_queries(self, classifier):
        """Test TRADEOFFS type classification."""
        queries = [
            "Tradeoffs of microservices",
            "Pros and cons of NoSQL",
            "Advantages and disadvantages of caching",
            "Strengths and weaknesses of agile",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.TRADEOFFS

    def test_alternatives_queries(self, classifier):
        """Test ALTERNATIVES type classification."""
        queries = [
            "Alternatives to Redis",
            "Other options for state management",
            "Similar to React but simpler",
            "Instead of REST, what else?",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.ALTERNATIVES

    def test_general_fallback(self, classifier):
        """Test GENERAL fallback for unmatched queries."""
        queries = [
            "Hi there",
            "Random text",
            "123",
            "",
        ]
        for q in queries:
            assert classifier.classify(q) == QueryType.GENERAL

    def test_case_insensitive(self, classifier):
        """Test case-insensitive matching."""
        assert classifier.classify("WHAT IS PYTHON?") == QueryType.DEFINITION
        assert classifier.classify("explain recursion") == QueryType.EXPLANATION
        assert classifier.classify("Show Me An Example") == QueryType.EXAMPLE

    def test_topic_extraction_definition(self, classifier):
        """Test topic extraction from DEFINITION queries."""
        topic = classifier.extract_topic(
            "What is Thompson Sampling?",
            QueryType.DEFINITION
        )
        # Topic extraction lowercases the result
        assert topic.lower() == "thompson sampling"

        topic = classifier.extract_topic(
            "What are neural networks?",
            QueryType.DEFINITION
        )
        assert topic.lower() == "neural networks"

    def test_topic_extraction_explanation(self, classifier):
        """Test topic extraction from EXPLANATION queries."""
        topic = classifier.extract_topic(
            "Explain Thompson Sampling",
            QueryType.EXPLANATION
        )
        assert topic.lower() == "thompson sampling"

        topic = classifier.extract_topic(
            "How does gradient descent work?",
            QueryType.EXPLANATION
        )
        assert topic.lower() == "gradient descent"

    def test_topic_extraction_comparison(self, classifier):
        """Test topic extraction from COMPARISON queries."""
        topic = classifier.extract_topic(
            "Compare Python and JavaScript",
            QueryType.COMPARISON
        )
        assert "Python" in topic and "JavaScript" in topic

    def test_topic_extraction_fallback(self, classifier):
        """Test fallback topic extraction."""
        topic = classifier.extract_topic(
            "Tell me more about reinforcement learning",
            QueryType.GENERAL
        )
        assert topic is not None
        assert "reinforcement" in topic or "learning" in topic

    def test_topic_extraction_empty(self, classifier):
        """Test topic extraction with empty query."""
        topic = classifier.extract_topic("", QueryType.GENERAL)
        assert topic is None or topic == ""


# ============================================================================
# Test QueryPatternTracker
# ============================================================================

class TestQueryPatternTracker:
    """Tests for pattern tracking and learning."""

    def test_record_transition(self, pattern_tracker):
        """Test recording transitions."""
        pattern_tracker.record_transition(
            QueryType.DEFINITION,
            QueryType.EXPLANATION
        )

        assert pattern_tracker.transition_counts[QueryType.DEFINITION][QueryType.EXPLANATION] == 1
        assert pattern_tracker.from_counts[QueryType.DEFINITION] == 1

    def test_multiple_transitions(self, pattern_tracker):
        """Test recording multiple transitions."""
        # Record same transition multiple times
        for _ in range(5):
            pattern_tracker.record_transition(
                QueryType.DEFINITION,
                QueryType.EXPLANATION
            )

        assert pattern_tracker.transition_counts[QueryType.DEFINITION][QueryType.EXPLANATION] == 5
        assert pattern_tracker.from_counts[QueryType.DEFINITION] == 5

    def test_topic_specific_transitions(self, pattern_tracker):
        """Test topic-specific transition tracking."""
        pattern_tracker.record_transition(
            QueryType.DEFINITION,
            QueryType.EXPLANATION,
            topic="python"
        )

        assert pattern_tracker.topic_transitions["python"][QueryType.DEFINITION][QueryType.EXPLANATION] == 1

    def test_transition_probability_default(self, pattern_tracker):
        """Test default transition probabilities."""
        # No transitions recorded, should use defaults
        prob = pattern_tracker.get_transition_probability(
            QueryType.DEFINITION,
            QueryType.EXPLANATION
        )

        # Should get default probability (0.35 for DEF->EXPL)
        assert prob == 0.35

    def test_transition_probability_learned(self, pattern_tracker):
        """Test learned transition probabilities."""
        # Record enough transitions to trust learned probability
        for _ in range(10):
            pattern_tracker.record_transition(
                QueryType.DEFINITION,
                QueryType.EXPLANATION
            )

        # Also record other transitions
        for _ in range(10):
            pattern_tracker.record_transition(
                QueryType.DEFINITION,
                QueryType.EXAMPLE
            )

        prob = pattern_tracker.get_transition_probability(
            QueryType.DEFINITION,
            QueryType.EXPLANATION
        )

        # Should be 10/20 = 0.5
        assert prob == 0.5

    def test_transition_probability_topic_blend(self, pattern_tracker):
        """Test topic-specific probability blending."""
        # Record general transitions
        for _ in range(10):
            pattern_tracker.record_transition(
                QueryType.DEFINITION,
                QueryType.EXPLANATION
            )

        # Record topic-specific transitions
        for _ in range(5):
            pattern_tracker.record_transition(
                QueryType.DEFINITION,
                QueryType.EXAMPLE,
                topic="python"
            )

        # Get topic-specific probability
        prob = pattern_tracker.get_transition_probability(
            QueryType.DEFINITION,
            QueryType.EXAMPLE,
            topic="python"
        )

        # Should blend topic (100%) with general (0%)
        # 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert prob == pytest.approx(0.7, rel=0.01)

    def test_get_likely_follow_ups(self, pattern_tracker):
        """Test getting likely follow-ups."""
        follow_ups = pattern_tracker.get_likely_follow_ups(
            QueryType.DEFINITION,
            top_k=3
        )

        assert len(follow_ups) <= 3
        assert all(isinstance(qtype, QueryType) for qtype, _ in follow_ups)
        assert all(0 <= prob <= 1 for _, prob in follow_ups)

        # Should be sorted by probability
        probs = [prob for _, prob in follow_ups]
        assert probs == sorted(probs, reverse=True)

    def test_get_likely_follow_ups_excludes_self(self, pattern_tracker):
        """Test that follow-ups exclude same type."""
        follow_ups = pattern_tracker.get_likely_follow_ups(
            QueryType.DEFINITION,
            top_k=10
        )

        # Should not include DEFINITION itself
        types = [qtype for qtype, _ in follow_ups]
        assert QueryType.DEFINITION not in types

    def test_statistics(self, pattern_tracker):
        """Test statistics collection."""
        # Record some transitions
        pattern_tracker.record_transition(QueryType.DEFINITION, QueryType.EXPLANATION)
        pattern_tracker.record_transition(QueryType.DEFINITION, QueryType.EXAMPLE)
        pattern_tracker.record_transition(QueryType.EXPLANATION, QueryType.IMPLEMENTATION, topic="python")

        stats = pattern_tracker.get_statistics()

        assert stats["total_transitions"] == 3
        assert stats["unique_from_types"] == 2
        assert stats["topics_tracked"] == 1
        assert "transition_matrix" in stats

    def test_disabled_learning(self):
        """Test with learning disabled."""
        config = AnticipatoryConfig(pattern_learning_enabled=False)
        tracker = QueryPatternTracker(config)

        # Try to record transition
        tracker.record_transition(QueryType.DEFINITION, QueryType.EXPLANATION)

        # Should not have recorded
        assert tracker.from_counts[QueryType.DEFINITION] == 0


# ============================================================================
# Test SessionContext
# ============================================================================

class TestSessionContext:
    """Tests for session context management."""

    def test_create_session(self):
        """Test session creation."""
        session = SessionContext(session_id="test123")
        assert session.session_id == "test123"
        assert len(session.query_history) == 0
        assert session.total_queries == 0
        assert session.cache_hits == 0

    def test_add_query(self, session):
        """Test adding queries to history."""
        session.add_query(
            "What is Python?",
            QueryType.DEFINITION,
            topic="python",
            confidence=0.9
        )

        assert len(session.query_history) == 1
        assert session.total_queries == 1
        assert session.current_topic == "python"
        assert "python" in session.topics

    def test_query_history_limit(self, session):
        """Test query history trimming."""
        session.max_history = 5

        # Add 10 queries
        for i in range(10):
            session.add_query(
                f"Query {i}",
                QueryType.GENERAL,
                confidence=0.5
            )

        # Should only keep last 5
        assert len(session.query_history) == 5
        assert session.query_history[0]["text"] == "Query 5"
        assert session.query_history[-1]["text"] == "Query 9"

    def test_get_last_n_queries(self, session):
        """Test getting last N queries."""
        for i in range(5):
            session.add_query(f"Query {i}", QueryType.GENERAL)

        last_3 = session.get_last_n_queries(3)
        assert len(last_3) == 3
        assert last_3[0]["text"] == "Query 2"
        assert last_3[-1]["text"] == "Query 4"

    def test_get_last_n_empty(self, session):
        """Test getting queries from empty history."""
        last_3 = session.get_last_n_queries(3)
        assert last_3 == []

    def test_check_prefetch_hit(self, session):
        """Test prefetch hit checking."""
        session.prefetched_memory_ids = {"mem1", "mem2", "mem3"}

        # Query uses 2 of 3 prefetched
        hit_rate = session.check_prefetch_hit({"mem1", "mem2", "mem4"})

        assert hit_rate == pytest.approx(2/3)
        assert session.cache_hits == 1

    def test_check_prefetch_no_overlap(self, session):
        """Test prefetch with no overlap."""
        session.prefetched_memory_ids = {"mem1", "mem2", "mem3"}

        hit_rate = session.check_prefetch_hit({"mem4", "mem5"})

        assert hit_rate == 0.0
        assert session.cache_hits == 0

    def test_check_prefetch_empty_sets(self, session):
        """Test prefetch check with empty sets."""
        hit_rate = session.check_prefetch_hit(set())
        assert hit_rate == 0.0

        session.prefetched_memory_ids = {"mem1"}
        hit_rate = session.check_prefetch_hit(set())
        assert hit_rate == 0.0

    def test_topic_tracking(self, session):
        """Test topic tracking across queries."""
        session.add_query("What is Python?", QueryType.DEFINITION, topic="python")
        session.add_query("Explain JavaScript", QueryType.EXPLANATION, topic="javascript")
        session.add_query("How does Python work?", QueryType.EXPLANATION, topic="python")

        assert session.current_topic == "python"
        assert "python" in session.topics
        assert "javascript" in session.topics


# ============================================================================
# Test FollowUpPredictor
# ============================================================================

class TestFollowUpPredictor:
    """Tests for follow-up prediction."""

    def test_predict_basic(self, predictor, session):
        """Test basic prediction."""
        predictions = predictor.predict(
            "What is Thompson Sampling?",
            session
        )

        assert isinstance(predictions, list)
        assert len(predictions) <= predictor.config.max_predictions

        for pred in predictions:
            assert isinstance(pred, PredictedQuery)
            assert isinstance(pred.query_type, QueryType)
            assert 0 <= pred.probability <= 1

    def test_predictions_sorted(self, predictor, session):
        """Test predictions are sorted by probability."""
        predictions = predictor.predict(
            "What is Python?",
            session
        )

        if len(predictions) > 1:
            probs = [p.probability for p in predictions]
            assert probs == sorted(probs, reverse=True)

    def test_predictions_with_topic(self, predictor, session):
        """Test predictions include topic."""
        predictions = predictor.predict(
            "What is Thompson Sampling?",
            session
        )

        # Should extract topic from query
        assert any(p.topic for p in predictions)

    def test_predictions_use_session_topic(self, predictor, session):
        """Test predictions use session topic as fallback."""
        session.current_topic = "machine learning"

        # Generic query without clear topic
        predictions = predictor.predict(
            "How does this work?",
            session
        )

        # Should use session topic
        for pred in predictions:
            if pred.topic:
                assert pred.topic == "machine learning"

    def test_template_generation(self, predictor, session):
        """Test query template generation."""
        predictions = predictor.predict(
            "What is recursion?",
            session
        )

        # Should generate templates with topic
        templates_with_topic = [
            p.template for p in predictions
            if p.template and "recursion" in p.template.lower()
        ]

        assert len(templates_with_topic) > 0

    def test_probability_adjustment_duplicate_type(self, predictor, session):
        """Test probability reduced for already-asked types."""
        # Add EXPLANATION query to session
        session.add_query(
            "Explain Python",
            QueryType.EXPLANATION,
            topic="python"
        )

        # Now ask DEFINITION, which typically leads to EXPLANATION
        predictions = predictor.predict(
            "What is Python?",
            session
        )

        # EXPLANATION probability should be reduced (if predicted)
        expl_preds = [p for p in predictions if p.query_type == QueryType.EXPLANATION]
        if expl_preds:
            # Should be reduced by 0.5 multiplier
            assert expl_preds[0].probability < 0.35  # Default prob for DEF->EXPL

    def test_probability_boost_topic_continuity(self, predictor, session):
        """Test probability boost with topic continuity."""
        session.current_topic = "python"

        predictions = predictor.predict(
            "What is Python?",
            session
        )

        # Predictions should have boosted probability due to topic continuity
        # (though we can't easily test the exact value without mocking)
        assert len(predictions) > 0

    def test_min_confidence_filter(self):
        """Test minimum confidence filtering."""
        config = AnticipatoryConfig(min_prediction_confidence=0.9)
        classifier = QueryTypeClassifier()
        tracker = QueryPatternTracker(config)
        predictor = FollowUpPredictor(tracker, classifier, config)
        session = SessionContext(session_id="test")

        predictions = predictor.predict("What is Python?", session)

        # With very high threshold, should get few/no predictions
        assert all(p.probability >= 0.9 for p in predictions)


# ============================================================================
# Test AnticipatoryFetcher
# ============================================================================

class TestAnticipatoryFetcher:
    """Tests for anticipatory fetching."""

    @pytest.mark.asyncio
    async def test_prefetch_basic(self, fetcher, mock_retriever, session):
        """Test basic prefetching."""
        fetcher.set_retriever(mock_retriever)

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                topic="python",
                template="How does python work?"
            )
        ]

        result = await fetcher.prefetch_for_predictions(predictions, session)

        assert isinstance(result, PrefetchResult)
        assert len(result.prefetched_memory_ids) > 0
        assert result.memories_fetched > 0
        assert result.prefetch_time_ms >= 0

    @pytest.mark.asyncio
    async def test_prefetch_updates_session(self, fetcher, mock_retriever, session):
        """Test prefetch updates session."""
        fetcher.set_retriever(mock_retriever)

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template="How does it work?"
            )
        ]

        await fetcher.prefetch_for_predictions(predictions, session)

        assert len(session.prefetched_memory_ids) > 0
        assert session.prefetch_timestamp is not None

    @pytest.mark.asyncio
    async def test_prefetch_respects_budget(self, fetcher, mock_retriever, session):
        """Test prefetch respects total budget."""
        fetcher.set_retriever(mock_retriever)
        fetcher.config.max_prefetch_total = 5

        # Create many predictions
        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template=f"Query {i}"
            )
            for i in range(10)
        ]

        result = await fetcher.prefetch_for_predictions(predictions, session)

        # Should not exceed budget
        assert result.memories_fetched <= 5

    @pytest.mark.asyncio
    async def test_prefetch_scales_by_probability(self, fetcher, mock_retriever, session):
        """Test prefetch amount scales with prediction probability."""
        fetcher.set_retriever(mock_retriever)

        # High probability prediction
        high_prob = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.9,
                template="High prob query"
            )
        ]

        result_high = await fetcher.prefetch_for_predictions(high_prob, session)

        # Reset session
        session.prefetched_memory_ids = set()

        # Low probability prediction
        low_prob = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.1,
                template="Low prob query"
            )
        ]

        result_low = await fetcher.prefetch_for_predictions(low_prob, session)

        # High probability should fetch more (or at least equal)
        # Note: Exact comparison depends on k calculation
        assert result_high.memories_fetched >= result_low.memories_fetched

    @pytest.mark.asyncio
    async def test_prefetch_disabled(self, fetcher, mock_retriever, session):
        """Test with prefetch disabled."""
        fetcher.config.prefetch_enabled = False
        fetcher.set_retriever(mock_retriever)

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template="Query"
            )
        ]

        result = await fetcher.prefetch_for_predictions(predictions, session)

        assert result.memories_fetched == 0
        assert len(result.prefetched_memory_ids) == 0

    @pytest.mark.asyncio
    async def test_prefetch_no_retriever(self, fetcher, session):
        """Test prefetch with no retriever set."""
        # Don't set retriever

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template="Query"
            )
        ]

        result = await fetcher.prefetch_for_predictions(predictions, session)

        assert result.memories_fetched == 0

    @pytest.mark.asyncio
    async def test_prefetch_no_template(self, fetcher, mock_retriever, session):
        """Test prefetch skips predictions without templates."""
        fetcher.set_retriever(mock_retriever)

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template=None  # No template
            )
        ]

        result = await fetcher.prefetch_for_predictions(predictions, session)

        assert result.memories_fetched == 0

    @pytest.mark.asyncio
    async def test_prefetch_retriever_fallback(self, fetcher, session):
        """Test fallback when retriever doesn't have retrieve_ids."""
        # Create retriever without retrieve_ids
        retriever = Mock()
        retriever.retrieve = AsyncMock(return_value=[
            (Mock(id="mem1"), 0.9),
        ])
        fetcher.set_retriever(retriever)

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template="Query"
            )
        ]

        result = await fetcher.prefetch_for_predictions(predictions, session)

        # Should have used retrieve() fallback
        assert result.memories_fetched > 0

    @pytest.mark.asyncio
    async def test_prefetch_handles_errors(self, fetcher, session):
        """Test prefetch handles retriever errors gracefully."""
        # Create retriever that raises error
        retriever = Mock()
        retriever.retrieve_ids = AsyncMock(side_effect=Exception("Retrieval error"))
        fetcher.set_retriever(retriever)

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template="Query"
            )
        ]

        # Should not raise, just log warning
        result = await fetcher.prefetch_for_predictions(predictions, session)

        assert result.memories_fetched == 0


# ============================================================================
# Test AnticipatoryRetrieval Integration
# ============================================================================

class TestAnticipatoryRetrieval:
    """Tests for main integration class."""

    def test_create_system(self):
        """Test creating anticipatory system."""
        system = AnticipatoryRetrieval()

        assert system.classifier is not None
        assert system.pattern_tracker is not None
        assert system.predictor is not None
        assert system.fetcher is not None

    def test_create_with_config(self):
        """Test creating with custom config."""
        config = AnticipatoryConfig(
            min_prediction_confidence=0.5,
            max_predictions=5
        )
        system = AnticipatoryRetrieval(config)

        assert system.config.min_prediction_confidence == 0.5
        assert system.config.max_predictions == 5

    def test_create_session(self):
        """Test session creation."""
        system = AnticipatoryRetrieval()
        session = system.create_session("user123")

        assert session.session_id == "user123"
        assert "user123" in system._sessions

    def test_get_session(self):
        """Test getting existing session."""
        system = AnticipatoryRetrieval()
        created = system.create_session("user123")

        retrieved = system.get_session("user123")
        assert retrieved is created

        # Non-existent session
        assert system.get_session("nonexistent") is None

    def test_get_or_create_session(self):
        """Test get or create session."""
        system = AnticipatoryRetrieval()

        # Create new
        session1 = system.get_or_create_session("user1")
        assert session1.session_id == "user1"

        # Get existing
        session2 = system.get_or_create_session("user1")
        assert session2 is session1

    def test_process_query(self):
        """Test query processing."""
        system = AnticipatoryRetrieval()
        session = system.create_session("test")

        predictions = system.process_query(
            session,
            "What is Thompson Sampling?"
        )

        assert isinstance(predictions, list)
        assert len(session.query_history) == 1
        assert system.total_queries == 1

    def test_process_query_records_transition(self):
        """Test query processing records transitions."""
        system = AnticipatoryRetrieval()
        session = system.create_session("test")

        # First query
        system.process_query(session, "What is Python?")

        # Second query (should record transition)
        system.process_query(session, "Explain Python")

        # Check transition was recorded
        assert system.pattern_tracker.from_counts[QueryType.DEFINITION] > 0

    @pytest.mark.asyncio
    async def test_prefetch(self, mock_retriever):
        """Test prefetch integration."""
        system = AnticipatoryRetrieval()
        system.set_retriever(mock_retriever)
        session = system.create_session("test")

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.8,
                template="How does it work?"
            )
        ]

        result = await system.prefetch(session, predictions)

        assert isinstance(result, PrefetchResult)
        assert result.memories_fetched > 0

    def test_record_prediction_hit(self):
        """Test recording prediction hits."""
        system = AnticipatoryRetrieval()
        session = system.create_session("test")

        # Add a query to session
        session.add_query("What is Python?", QueryType.DEFINITION)

        # Next query matches a predicted type (EXPLANATION is common follow-up)
        was_hit = system.record_prediction_hit(session, "Explain Python")

        # Should recognize this as a prediction hit
        assert was_hit is True
        assert system.successful_predictions > 0

    def test_statistics(self):
        """Test statistics collection."""
        system = AnticipatoryRetrieval()
        session = system.create_session("test")

        # Process some queries
        system.process_query(session, "What is Python?")
        system.process_query(session, "Explain Python")

        stats = system.get_statistics()

        assert stats["total_queries"] == 2
        assert stats["total_predictions"] >= 0
        assert stats["active_sessions"] == 1
        assert "pattern_stats" in stats

    def test_prediction_accuracy_calculation(self):
        """Test prediction accuracy calculation."""
        system = AnticipatoryRetrieval()

        system.total_predictions = 10
        system.successful_predictions = 7

        stats = system.get_statistics()

        assert stats["prediction_accuracy"] == 0.7

    def test_prediction_accuracy_zero_division(self):
        """Test accuracy calculation with zero predictions."""
        system = AnticipatoryRetrieval()

        stats = system.get_statistics()

        assert stats["prediction_accuracy"] == 0.0


# ============================================================================
# Test Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_anticipatory_retrieval(self):
        """Test factory function."""
        system = create_anticipatory_retrieval()

        assert isinstance(system, AnticipatoryRetrieval)
        assert system.config is not None

    def test_create_anticipatory_retrieval_with_config(self):
        """Test factory with config."""
        config = AnticipatoryConfig(max_predictions=10)
        system = create_anticipatory_retrieval(config)

        assert system.config.max_predictions == 10

    def test_create_anticipatory_retrieval_with_retriever(self, mock_retriever):
        """Test factory with retriever."""
        system = create_anticipatory_retrieval(retriever=mock_retriever)

        assert system.fetcher._retriever is mock_retriever

    def test_create_session_context(self):
        """Test session context factory."""
        session = create_session_context("test123")

        assert isinstance(session, SessionContext)
        assert session.session_id == "test123"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query(self, classifier):
        """Test classification of empty query."""
        qtype = classifier.classify("")
        assert qtype == QueryType.GENERAL

    def test_whitespace_query(self, classifier):
        """Test classification of whitespace query."""
        qtype = classifier.classify("   \n\t  ")
        assert qtype == QueryType.GENERAL

    def test_special_characters(self, classifier):
        """Test queries with special characters."""
        queries = [
            "What is C++?",
            "Define @decorator",
            "Explain $variable",
        ]
        for q in queries:
            qtype = classifier.classify(q)
            # Should still classify (probably as DEFINITION)
            assert qtype in QueryType

    def test_very_long_query(self, classifier):
        """Test very long query."""
        long_query = "What is " + " ".join(["word"] * 1000) + "?"
        qtype = classifier.classify(long_query)
        assert qtype == QueryType.DEFINITION

    def test_session_context_concurrent_modifications(self, session):
        """Test session handles concurrent query additions."""
        # Simulate concurrent additions
        for i in range(100):
            session.add_query(f"Query {i}", QueryType.GENERAL)

        # Should handle gracefully
        assert len(session.query_history) == min(100, session.max_history)

    def test_pattern_tracker_zero_transitions(self, pattern_tracker):
        """Test pattern tracker with no recorded transitions."""
        # Get probability without any data
        prob = pattern_tracker.get_transition_probability(
            QueryType.DEFINITION,
            QueryType.EXPLANATION
        )

        # Should fall back to default
        assert prob > 0

    @pytest.mark.asyncio
    async def test_prefetch_with_zero_probability(self, fetcher, mock_retriever, session):
        """Test prefetch with zero probability prediction."""
        fetcher.set_retriever(mock_retriever)

        predictions = [
            PredictedQuery(
                query_type=QueryType.EXPLANATION,
                probability=0.0,  # Zero probability
                template="Query"
            )
        ]

        result = await fetcher.prefetch_for_predictions(predictions, session)

        # Should fetch nothing (k < 1)
        assert result.memories_fetched == 0

    def test_session_prefetch_timestamp_expiry(self, session):
        """Test expired prefetch timestamp."""
        session.prefetched_memory_ids = {"mem1", "mem2"}
        session.prefetch_timestamp = time.time() - 400  # 400 seconds ago (> 300 TTL)

        # Check if prefetch is still valid
        # (This would be implemented in the actual retrieval logic)
        ttl = 300.0
        is_expired = (time.time() - session.prefetch_timestamp) > ttl

        assert is_expired is True

    def test_duplicate_topic_handling(self, session):
        """Test handling of duplicate topics."""
        session.add_query("What is Python?", QueryType.DEFINITION, topic="python")
        session.add_query("Explain Python", QueryType.EXPLANATION, topic="python")
        session.add_query("Show Python example", QueryType.EXAMPLE, topic="python")

        # Should only have one "python" in topics list
        assert session.topics.count("python") == 1


# ============================================================================
# Integration Test
# ============================================================================

class TestFullIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_retriever):
        """Test complete workflow from query to prefetch."""
        # Setup
        system = create_anticipatory_retrieval(retriever=mock_retriever)
        session = system.create_session("user1")

        # 1. Process first query
        query1 = "What is Thompson Sampling?"
        predictions1 = system.process_query(session, query1)

        assert len(predictions1) > 0
        assert session.total_queries == 1

        # 2. Prefetch for predictions
        prefetch1 = await system.prefetch(session, predictions1)

        assert prefetch1.memories_fetched > 0
        assert len(session.prefetched_memory_ids) > 0

        # 3. Process second query (likely follow-up)
        query2 = "Explain Thompson Sampling"
        predictions2 = system.process_query(session, query2)

        # Should have recorded transition
        assert system.pattern_tracker.from_counts[QueryType.DEFINITION] > 0

        # 4. Check prefetch hit
        hit_rate = session.check_prefetch_hit({"mem1", "mem2"})

        # Should have some overlap with prefetched
        assert hit_rate >= 0

        # 5. Get statistics
        stats = system.get_statistics()

        assert stats["total_queries"] == 2
        assert stats["active_sessions"] == 1

    @pytest.mark.asyncio
    async def test_session_continuity(self, mock_retriever):
        """Test session maintains continuity across multiple queries."""
        system = create_anticipatory_retrieval(retriever=mock_retriever)
        session = system.create_session("user1")

        # Simulate conversation about Python
        queries = [
            "What is Python?",
            "Explain Python syntax",
            "Show me a Python example",
            "How to implement loops in Python?",
        ]

        for query in queries:
            predictions = system.process_query(session, query)
            if predictions:
                await system.prefetch(session, predictions)

        # Check session state
        assert len(session.query_history) == 4
        assert "python" in session.current_topic.lower()
        assert session.total_queries == 4

    def test_pattern_learning_over_time(self):
        """Test pattern learning improves with more data."""
        system = AnticipatoryRetrieval()
        session = system.create_session("user1")

        # Simulate many queries showing pattern: DEFINITION -> EXAMPLE
        for i in range(20):
            system.process_query(session, f"What is concept{i}?")
            system.process_query(session, f"Show example of concept{i}")

        # Check that pattern was learned
        prob = system.pattern_tracker.get_transition_probability(
            QueryType.DEFINITION,
            QueryType.EXAMPLE
        )

        # Should be high since we always followed with EXAMPLE
        assert prob > 0.8


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and scalability tests."""

    def test_classification_speed(self, classifier):
        """Test classification is fast."""
        import time

        queries = [
            "What is Python?",
            "Explain machine learning",
            "Show me an example",
        ] * 100  # 300 queries

        start = time.time()
        for q in queries:
            classifier.classify(q)
        duration = time.time() - start

        # Should be very fast (< 100ms for 300 queries)
        assert duration < 0.1

    def test_large_session_history(self):
        """Test session with large history."""
        session = SessionContext(session_id="test")
        session.max_history = 1000

        # Add 1000 queries
        for i in range(1000):
            session.add_query(f"Query {i}", QueryType.GENERAL)

        # Should maintain size limit
        assert len(session.query_history) == 1000

        # Recent queries should be accessible
        recent = session.get_last_n_queries(10)
        assert len(recent) == 10

    @pytest.mark.asyncio
    async def test_concurrent_prefetch(self, mock_retriever):
        """Test concurrent prefetch operations."""
        system = create_anticipatory_retrieval(retriever=mock_retriever)

        # Create multiple sessions
        sessions = [system.create_session(f"user{i}") for i in range(10)]

        # Process queries and prefetch concurrently
        tasks = []
        for session in sessions:
            predictions = system.process_query(session, "What is Python?")
            task = system.prefetch(session, predictions)
            tasks.append(task)

        # Wait for all
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 10
        assert all(r.memories_fetched >= 0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
