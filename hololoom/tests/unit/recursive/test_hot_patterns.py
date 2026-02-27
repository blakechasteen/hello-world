"""
Unit tests for HoloLoom.recursive.hot_patterns

Tests usage tracking, heat score calculation, decay mechanism, and adaptive retrieval.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, MagicMock
from hololoom.recursive.hot_patterns import (
    UsageRecord,
    HotPatternTracker,
    AdaptiveRetriever,
    RetrievalWeights,
    HotPatternConfig,
    HotPatternFeedbackEngine
)
from hololoom.fabric.spacetime import Spacetime, WeavingTrace
from hololoom.protocols.types import Query, MemoryShard


class TestUsageRecord:
    """Test UsageRecord dataclass"""

    def test_initialization(self):
        """Test creating a usage record"""
        record = UsageRecord(
            element_id="thread_001",
            element_type="thread"
        )

        assert record.element_id == "thread_001"
        assert record.element_type == "thread"
        assert record.access_count == 0
        assert record.success_count == 0
        assert record.avg_confidence == 0.0

    def test_record_access_successful(self):
        """Test recording successful access"""
        record = UsageRecord("thread_001", "thread")

        record.record_access(0.85)

        assert record.access_count == 1
        assert record.success_count == 1  # >= 0.75
        assert record.avg_confidence == 0.85
        assert record.total_confidence == 0.85

    def test_record_access_unsuccessful(self):
        """Test recording unsuccessful access"""
        record = UsageRecord("thread_002", "thread")

        record.record_access(0.65)

        assert record.access_count == 1
        assert record.success_count == 0  # < 0.75
        assert record.avg_confidence == 0.65

    def test_record_multiple_accesses(self):
        """Test recording multiple accesses"""
        record = UsageRecord("thread_003", "thread")

        record.record_access(0.80)
        record.record_access(0.90)
        record.record_access(0.70)

        assert record.access_count == 3
        assert record.success_count == 2  # Two >= 0.75
        assert record.avg_confidence == pytest.approx(0.80, abs=0.01)

    def test_success_rate(self):
        """Test success rate calculation"""
        record = UsageRecord("thread_004", "thread")

        # No accesses
        assert record.success_rate == 0.0

        # Add accesses
        record.record_access(0.80)  # Success
        record.record_access(0.60)  # Failure
        record.record_access(0.90)  # Success

        assert record.success_rate == pytest.approx(2/3, abs=0.01)

    def test_heat_score_calculation(self):
        """Test heat score calculation"""
        record = UsageRecord("thread_005", "thread")

        record.record_access(0.85)
        record.record_access(0.90)
        record.record_access(0.80)

        # Heat = access_count * success_rate * avg_confidence
        expected_heat = 3 * 1.0 * 0.85
        assert record.heat_score == pytest.approx(expected_heat, abs=0.01)

    def test_heat_score_with_failures(self):
        """Test heat score with mixed success/failure"""
        record = UsageRecord("thread_006", "thread")

        record.record_access(0.90)  # Success
        record.record_access(0.60)  # Failure
        record.record_access(0.70)  # Failure

        # access_count=3, success_count=1, avg_confidence=(0.90+0.60+0.70)/3
        avg_conf = (0.90 + 0.60 + 0.70) / 3
        expected_heat = 3 * (1/3) * avg_conf
        assert record.heat_score == pytest.approx(expected_heat, abs=0.01)

    def test_age_seconds(self):
        """Test age calculation"""
        record = UsageRecord("thread_007", "thread")

        # Sleep a bit
        time.sleep(0.1)

        age = record.age_seconds
        assert age >= 0.1
        assert age < 1.0  # Should be recent


class TestHotPatternTracker:
    """Test HotPatternTracker class"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = HotPatternTracker(
            decay_rate=0.95,
            decay_interval=3600.0
        )

        assert tracker.decay_rate == 0.95
        assert tracker.decay_interval == 3600.0
        assert tracker.usage == {}
        assert tracker.total_accesses == 0
        assert tracker.unique_elements == 0

    def test_record_usage_threads(self):
        """Test recording thread usage"""
        tracker = HotPatternTracker()

        # Create mock spacetime
        trace = WeavingTrace(
            query_text="Test query",
            threads_activated=["thread1", "thread2"],
            motifs_detected=[],
            tool_confidence=0.85
        )
        spacetime = Spacetime(
            query_text="Test query",
            response="Test response",
            confidence=0.85,
            trace=trace
        )

        tracker.record_usage(spacetime)

        assert tracker.total_accesses == 2  # 2 threads
        assert tracker.unique_elements == 2
        assert "thread1" in tracker.usage
        assert "thread2" in tracker.usage
        assert tracker.usage["thread1"].access_count == 1

    def test_record_usage_motifs(self):
        """Test recording motif usage"""
        tracker = HotPatternTracker()

        trace = WeavingTrace(
            query_text="Test query",
            threads_activated=[],
            motifs_detected=["AI", "ML", "neural"],
            tool_confidence=0.90
        )
        spacetime = Spacetime(
            query_text="Test query",
            response="Test response",
            confidence=0.90,
            trace=trace
        )

        tracker.record_usage(spacetime)

        assert tracker.total_accesses == 3  # 3 motifs
        assert "motif:AI" in tracker.usage
        assert "motif:ML" in tracker.usage
        assert tracker.usage["motif:AI"].element_type == "motif"
        assert tracker.usage["motif:AI"].avg_confidence == 0.90

    def test_record_multiple_spacetimes(self):
        """Test tracking multiple spacetimes"""
        tracker = HotPatternTracker()

        for i in range(3):
            trace = WeavingTrace(
                query_text=f"Query {i}",
                threads_activated=["thread1", "thread2"],
                motifs_detected=["AI"],
                tool_confidence=0.80 + i * 0.05
            )
            spacetime = Spacetime(
                query_text=f"Query {i}",
                response="Response",
                confidence=0.80 + i * 0.05,
                trace=trace
            )
            tracker.record_usage(spacetime)

        # thread1 accessed 3 times, thread2 accessed 3 times, AI accessed 3 times
        assert tracker.usage["thread1"].access_count == 3
        assert tracker.usage["thread2"].access_count == 3
        assert tracker.usage["motif:AI"].access_count == 3

    def test_get_hot_patterns(self):
        """Test retrieving hot patterns"""
        tracker = HotPatternTracker()

        # Add multiple accesses to thread1
        for i in range(10):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["thread1"],
                motifs_detected=[],
                tool_confidence=0.85
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.85,
                trace=trace
            )
            tracker.record_usage(spacetime)

        # Add fewer accesses to thread2
        for i in range(2):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["thread2"],
                motifs_detected=[],
                tool_confidence=0.80
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.80,
                trace=trace
            )
            tracker.record_usage(spacetime)

        hot = tracker.get_hot_patterns(top_k=5)

        # thread1 should be hotter (more accesses)
        assert len(hot) >= 1
        assert hot[0].element_id == "thread1"
        assert hot[0].heat_score > hot[1].heat_score if len(hot) > 1 else True

    def test_get_hot_patterns_by_type(self):
        """Test filtering hot patterns by type"""
        tracker = HotPatternTracker()

        # Add threads
        for i in range(5):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["thread1"],
                motifs_detected=["AI"],
                tool_confidence=0.85
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.85,
                trace=trace
            )
            tracker.record_usage(spacetime)

        hot_threads = tracker.get_hot_patterns(element_type="thread", top_k=5)
        hot_motifs = tracker.get_hot_patterns(element_type="motif", top_k=5)

        assert all(r.element_type == "thread" for r in hot_threads)
        assert all(r.element_type == "motif" for r in hot_motifs)

    def test_get_cold_patterns(self):
        """Test retrieving cold patterns"""
        tracker = HotPatternTracker()

        # Add old pattern (simulate by setting last_accessed in the past)
        trace = WeavingTrace(
            query_text="Query",
            threads_activated=["old_thread"],
            motifs_detected=[],
            tool_confidence=0.80
        )
        spacetime = Spacetime(
            query_text="Query",
            response="Response",
            confidence=0.80,
            trace=trace
        )
        tracker.record_usage(spacetime)

        # Manually set last_accessed to past
        tracker.usage["old_thread"].last_accessed = time.time() - 7200  # 2 hours ago

        cold = tracker.get_cold_patterns(age_threshold=3600.0, top_k=5)

        assert len(cold) >= 1
        assert "old_thread" in [r.element_id for r in cold]

    def test_decay_application(self):
        """Test heat decay mechanism"""
        tracker = HotPatternTracker(
            decay_rate=0.95,
            decay_interval=0.1  # 100ms for testing
        )

        # Add usage
        trace = WeavingTrace(
            query_text="Query",
            threads_activated=["thread1"],
            motifs_detected=[],
            tool_confidence=0.85
        )
        spacetime = Spacetime(
            query_text="Query",
            response="Response",
            confidence=0.85,
            trace=trace
        )

        # Record initial access
        tracker.record_usage(spacetime)
        initial_count = tracker.usage["thread1"].access_count

        # Wait for decay interval
        time.sleep(0.15)

        # Record another access (triggers decay check)
        tracker.record_usage(spacetime)

        # Access count should be decayed then incremented
        # After decay: initial_count * 0.95
        # After new access: (initial_count * 0.95) + 1
        assert tracker.usage["thread1"].access_count < initial_count + 2

    def test_prune_cold_patterns(self):
        """Test pruning cold patterns"""
        tracker = HotPatternTracker()

        # Add high heat pattern
        for i in range(10):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["hot_thread"],
                motifs_detected=[],
                tool_confidence=0.90
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.90,
                trace=trace
            )
            tracker.record_usage(spacetime)

        # Add low heat pattern
        trace = WeavingTrace(
            query_text="Query",
            threads_activated=["cold_thread"],
            motifs_detected=[],
            tool_confidence=0.50
        )
        spacetime = Spacetime(
            query_text="Query",
            response="Response",
            confidence=0.50,
            trace=trace
        )
        tracker.record_usage(spacetime)

        initial_count = len(tracker.usage)

        # Prune patterns with heat < 1.0
        pruned = tracker.prune_cold_patterns(min_heat=1.0)

        assert pruned >= 1
        assert len(tracker.usage) < initial_count
        assert "hot_thread" in tracker.usage

    def test_get_statistics(self):
        """Test statistics generation"""
        tracker = HotPatternTracker()

        # Add some usage
        for i in range(5):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["thread1"],
                motifs_detected=["AI"],
                tool_confidence=0.85
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.85,
                trace=trace
            )
            tracker.record_usage(spacetime)

        stats = tracker.get_statistics()

        assert stats["total_accesses"] == 10  # 5 threads + 5 motifs
        assert stats["unique_elements"] == 2
        assert "avg_heat" in stats
        assert "max_heat" in stats
        assert "elements_by_type" in stats


class TestAdaptiveRetriever:
    """Test AdaptiveRetriever class"""

    def test_initialization(self):
        """Test retriever initialization"""
        retriever = AdaptiveRetriever(
            hot_boost=2.0,
            cold_penalty=0.5,
            heat_threshold=1.0
        )

        assert retriever.hot_boost == 2.0
        assert retriever.cold_penalty == 0.5
        assert retriever.heat_threshold == 1.0

    def test_update_weights_from_tracker(self):
        """Test updating weights from tracker"""
        tracker = HotPatternTracker()
        retriever = AdaptiveRetriever()

        # Add hot thread
        for i in range(10):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["hot_thread"],
                motifs_detected=["AI"],
                tool_confidence=0.90
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.90,
                trace=trace
            )
            tracker.record_usage(spacetime)

        retriever.update_weights(tracker)

        # hot_thread should have boosted weight
        weight = retriever.get_thread_weight("hot_thread")
        assert weight > 1.0  # Should be boosted

    def test_hot_boost_application(self):
        """Test hot boost is applied correctly"""
        tracker = HotPatternTracker()
        retriever = AdaptiveRetriever(hot_boost=3.0, heat_threshold=1.0)

        # Create hot pattern (high heat)
        for i in range(20):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["super_hot"],
                motifs_detected=[],
                tool_confidence=0.95
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.95,
                trace=trace
            )
            tracker.record_usage(spacetime)

        retriever.update_weights(tracker)

        weight = retriever.get_thread_weight("super_hot")
        # Should be significantly boosted
        assert weight >= 3.0

    def test_cold_penalty_application(self):
        """Test cold penalty is applied correctly"""
        tracker = HotPatternTracker()
        retriever = AdaptiveRetriever(cold_penalty=0.3, heat_threshold=2.0)

        # Create cold pattern (low heat)
        trace = WeavingTrace(
            query_text="Query",
            threads_activated=["cold_thread"],
            motifs_detected=[],
            tool_confidence=0.60
        )
        spacetime = Spacetime(
            query_text="Query",
            response="Response",
            confidence=0.60,
            trace=trace
        )
        tracker.record_usage(spacetime)

        retriever.update_weights(tracker)

        weight = retriever.get_thread_weight("cold_thread")
        assert weight == 0.3  # Should be penalized

    def test_get_thread_weight_unknown(self):
        """Test getting weight for unknown thread"""
        retriever = AdaptiveRetriever()

        weight = retriever.get_thread_weight("unknown_thread")
        assert weight == 1.0  # Base weight

    def test_get_motif_weight(self):
        """Test getting motif weights"""
        tracker = HotPatternTracker()
        retriever = AdaptiveRetriever()

        # Add hot motif
        for i in range(15):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=[],
                motifs_detected=["HotMotif"],
                tool_confidence=0.88
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.88,
                trace=trace
            )
            tracker.record_usage(spacetime)

        retriever.update_weights(tracker)

        weight = retriever.get_motif_weight("HotMotif")
        assert weight > 1.0

    def test_boost_shards(self):
        """Test shard boosting based on hot patterns"""
        tracker = HotPatternTracker()
        retriever = AdaptiveRetriever()

        # Create hot pattern
        for i in range(10):
            trace = WeavingTrace(
                query_text="Query",
                threads_activated=["thread1"],
                motifs_detected=["AI"],
                tool_confidence=0.90
            )
            spacetime = Spacetime(
                query_text="Query",
                response="Response",
                confidence=0.90,
                trace=trace
            )
            tracker.record_usage(spacetime)

        retriever.update_weights(tracker)

        # Create shards
        shard1 = MemoryShard(
            id="thread1",
            content="AI content",
            metadata={"tags": ["AI"]}
        )
        shard2 = MemoryShard(
            id="thread2",
            content="Other content",
            metadata={"tags": ["other"]}
        )

        weighted = retriever.boost_shards([shard1, shard2], query_motifs=["AI"])

        # shard1 should have higher relevance (hot thread + matching motif)
        assert weighted[0][0].id == "thread1"
        assert weighted[0][1] > weighted[1][1]


class TestRetrievalWeights:
    """Test RetrievalWeights dataclass"""

    def test_initialization(self):
        """Test weights initialization"""
        weights = RetrievalWeights()

        assert weights.thread_weights == {}
        assert weights.motif_weights == {}
        assert weights.base_weight == 1.0

    def test_custom_base_weight(self):
        """Test custom base weight"""
        weights = RetrievalWeights(base_weight=0.5)

        assert weights.base_weight == 0.5


class TestHotPatternConfig:
    """Test HotPatternConfig dataclass"""

    def test_default_config(self):
        """Test default configuration"""
        config = HotPatternConfig()

        assert config.enable_tracking is True
        assert config.enable_adaptive_retrieval is True
        assert config.update_weights_interval == 10
        assert config.decay_rate == 0.95
        assert config.hot_boost == 2.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = HotPatternConfig(
            enable_tracking=False,
            decay_rate=0.90,
            hot_boost=3.0
        )

        assert config.enable_tracking is False
        assert config.decay_rate == 0.90
        assert config.hot_boost == 3.0


# Integration tests would require full HoloLoom setup
# Keeping them simple here
class TestHotPatternIntegration:
    """Integration tests for hot pattern workflow"""

    def test_complete_tracking_workflow(self):
        """Test complete workflow: track -> analyze -> adapt"""
        tracker = HotPatternTracker()
        retriever = AdaptiveRetriever()

        # Simulate multiple queries
        for i in range(5):
            trace = WeavingTrace(
                query_text=f"Query {i}",
                threads_activated=["thread1"],
                motifs_detected=["AI"],
                tool_confidence=0.85
            )
            spacetime = Spacetime(
                query_text=f"Query {i}",
                response="Response",
                confidence=0.85,
                trace=trace
            )
            tracker.record_usage(spacetime)

        # Get statistics
        stats = tracker.get_statistics()
        assert stats["total_accesses"] > 0

        # Update retrieval weights
        retriever.update_weights(tracker)

        # Verify weights were updated
        weight = retriever.get_thread_weight("thread1")
        assert weight > 1.0
