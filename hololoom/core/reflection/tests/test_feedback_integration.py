"""
Integration tests for Reflection Learning from User Feedback.

Tests the complete feedback loop:
- FeedbackStore storage and retrieval
- Thompson Sampling integration
- Learning signal extraction
- API endpoint integration

Author: Claude Code
Date: 2025-11-20
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

from HoloLoom.core.reflection.feedback_store import FeedbackStore, FeedbackRecord, LearningSignals
from HoloLoom.core.policy.thompson_sampling import TSBandit


@pytest.fixture
async def temp_store():
    """Create temporary feedback store for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_feedback.db"

    store = FeedbackStore(db_path=str(db_path))
    await store.initialize()

    yield store

    await store.close()
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_feedback():
    """Sample feedback data for testing."""
    return [
        {
            "query": "What is reinforcement learning?",
            "response": "RL is learning through interaction...",
            "tool_used": "answer",
            "confidence": 0.92,
            "user_rating": 1.0,
            "feedback_type": "helpful"
        },
        {
            "query": "How does PPO work?",
            "response": "PPO uses clipped objective...",
            "tool_used": "answer",
            "confidence": 0.88,
            "user_rating": 0.8,
            "feedback_type": "helpful"
        },
        {
            "query": "Show me example code",
            "response": "Here's an example...",
            "tool_used": "research",
            "confidence": 0.75,
            "user_rating": 0.6,
            "feedback_type": "not_helpful"
        },
    ]


@pytest.mark.asyncio
async def test_store_and_retrieve_feedback(temp_store, sample_feedback):
    """Test basic feedback storage and retrieval."""
    # Store feedback
    feedback_ids = []
    for fb in sample_feedback:
        feedback_id = await temp_store.store_feedback(**fb, user_id="@test:local")
        feedback_ids.append(feedback_id)
        assert feedback_id is not None
        assert len(feedback_id) == 16

    # Retrieve feedback history
    records = await temp_store.get_feedback_history(limit=10)
    assert len(records) == 3

    # Check first record
    assert records[0].query_text == sample_feedback[2]["query"]  # Most recent first
    assert records[0].tool_used == sample_feedback[2]["tool_used"]
    assert records[0].user_rating == sample_feedback[2]["user_rating"]


@pytest.mark.asyncio
async def test_learning_signals_extraction(temp_store, sample_feedback):
    """Test learning signal extraction."""
    # Store feedback
    for fb in sample_feedback:
        await temp_store.store_feedback(**fb, user_id="@test:local")

    # Get learning signals
    signals = await temp_store.get_learning_signals(min_samples=1)

    # Check answer tool signals
    assert "answer" in signals
    answer_signal = signals["answer"]
    assert answer_signal.total_samples == 2
    assert answer_signal.successful_samples == 2  # Both ≥ 0.7
    assert answer_signal.failed_samples == 0
    assert 0.8 < answer_signal.avg_rating < 1.0

    # Check research tool signals
    assert "research" in signals
    research_signal = signals["research"]
    assert research_signal.total_samples == 1
    assert research_signal.successful_samples == 0  # 0.6 < 0.7
    assert research_signal.failed_samples == 1


@pytest.mark.asyncio
async def test_thompson_sampling_updates(temp_store, sample_feedback):
    """Test Thompson Sampling prior updates."""
    # Store feedback
    for fb in sample_feedback:
        feedback_id = await temp_store.store_feedback(**fb, user_id="@test:local")

        # Update Thompson priors
        alpha, beta = await temp_store.update_thompson_priors(
            tool=fb["tool_used"],
            feedback_id=feedback_id
        )

        # Check priors updated
        assert alpha >= 1.0
        assert beta >= 1.0


@pytest.mark.asyncio
async def test_thompson_bandit_integration():
    """Test TSBandit feedback integration."""
    bandit = TSBandit(n_arms=3)

    # Initial priors (uniform)
    initial_stats = bandit.get_stats()
    assert initial_stats[0]["success"] == 1.0
    assert initial_stats[0]["fail"] == 1.0

    # Update with high rating (success)
    bandit.update_from_feedback(arm=0, user_rating=0.9)
    stats = bandit.get_stats()
    assert stats[0]["success"] == 1.9
    assert stats[0]["fail"] == 1.0

    # Update with low rating (failure)
    bandit.update_from_feedback(arm=1, user_rating=0.4)
    stats = bandit.get_stats()
    assert stats[1]["success"] == 1.0
    assert stats[1]["fail"] == 1.6  # 1 + (1 - 0.4)


@pytest.mark.asyncio
async def test_bandit_state_persistence():
    """Test saving and loading bandit state."""
    temp_dir = tempfile.mkdtemp()
    state_path = Path(temp_dir) / "bandit.json"

    try:
        # Create and update bandit
        bandit = TSBandit(n_arms=3)
        bandit.update_from_feedback(0, 0.9)
        bandit.update_from_feedback(1, 0.5)

        # Save state
        bandit.save_state(str(state_path))
        assert state_path.exists()

        # Load state
        loaded_bandit = TSBandit.load_state(str(state_path))

        # Check state restored
        original_stats = bandit.get_stats()
        loaded_stats = loaded_bandit.get_stats()

        assert original_stats[0]["success"] == loaded_stats[0]["success"]
        assert original_stats[0]["fail"] == loaded_stats[0]["fail"]

    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_feedback_statistics(temp_store, sample_feedback):
    """Test comprehensive statistics."""
    # Store feedback
    for fb in sample_feedback:
        await temp_store.store_feedback(**fb, user_id="@test:local")

    # Get statistics
    stats = await temp_store.get_statistics()

    assert stats["total_feedback_count"] == 3
    assert "answer" in stats["tool_statistics"]
    assert "research" in stats["tool_statistics"]
    assert stats["tool_statistics"]["answer"]["count"] == 2
    assert stats["tool_statistics"]["research"]["count"] == 1


@pytest.mark.asyncio
async def test_feedback_filtering(temp_store):
    """Test feedback filtering by tool and user."""
    # Store feedback for different tools and users
    await temp_store.store_feedback(
        query="Q1", response="R1", tool_used="answer",
        confidence=0.9, user_rating=1.0, feedback_type="helpful",
        user_id="@alice:local"
    )
    await temp_store.store_feedback(
        query="Q2", response="R2", tool_used="research",
        confidence=0.8, user_rating=0.8, feedback_type="helpful",
        user_id="@bob:local"
    )
    await temp_store.store_feedback(
        query="Q3", response="R3", tool_used="answer",
        confidence=0.85, user_rating=0.9, feedback_type="helpful",
        user_id="@alice:local"
    )

    # Filter by tool
    answer_records = await temp_store.get_feedback_history(tool="answer")
    assert len(answer_records) == 2

    # Filter by user
    alice_records = await temp_store.get_feedback_history(user_id="@alice:local")
    assert len(alice_records) == 2

    # Filter by both
    alice_answer = await temp_store.get_feedback_history(
        tool="answer",
        user_id="@alice:local"
    )
    assert len(alice_answer) == 2


@pytest.mark.asyncio
async def test_learning_trajectory_simulation(temp_store):
    """Test learning trajectory over simulated time."""
    # Simulate 30 feedback records with improving ratings
    for i in range(30):
        rating = 0.5 + (i / 60)  # 0.5 → 1.0
        await temp_store.store_feedback(
            query=f"Query {i}",
            response=f"Response {i}",
            tool_used="answer",
            confidence=0.85,
            user_rating=rating,
            feedback_type="rating",
            user_id="@test:local"
        )

    # Get learning signals
    signals = await temp_store.get_learning_signals(tool="answer", min_samples=1)
    signal = signals["answer"]

    # Check improvement
    assert signal.total_samples == 30
    assert signal.avg_rating > 0.7  # Should be high
    assert signal.success_rate > 0.5  # Most should be successes


@pytest.mark.asyncio
async def test_min_samples_threshold(temp_store):
    """Test min_samples filtering."""
    # Store 5 feedback records
    for i in range(5):
        await temp_store.store_feedback(
            query=f"Query {i}",
            response=f"Response {i}",
            tool_used="answer",
            confidence=0.85,
            user_rating=0.9,
            feedback_type="rating",
            user_id="@test:local"
        )

    # Should return signals with min_samples=5
    signals_5 = await temp_store.get_learning_signals(min_samples=5)
    assert "answer" in signals_5

    # Should not return signals with min_samples=10
    signals_10 = await temp_store.get_learning_signals(min_samples=10)
    assert "answer" not in signals_10


@pytest.mark.asyncio
async def test_explicit_feedback_storage(temp_store):
    """Test storing explicit text feedback."""
    feedback_id = await temp_store.store_feedback(
        query="Test query",
        response="Test response",
        tool_used="answer",
        confidence=0.9,
        user_rating=1.0,
        feedback_type="explicit",
        explicit_feedback="This is excellent! Very clear explanation.",
        user_id="@test:local"
    )

    # Retrieve and check
    records = await temp_store.get_feedback_history(limit=1)
    assert len(records) == 1
    assert records[0].explicit_feedback == "This is excellent! Very clear explanation."


def test_feedback_record_serialization():
    """Test FeedbackRecord to_dict serialization."""
    record = FeedbackRecord(
        feedback_id="abc123",
        query_text="Test query",
        response_text="Test response",
        tool_used="answer",
        confidence=0.92,
        user_rating=1.0,
        feedback_type="helpful",
        user_id="@test:local"
    )

    # Serialize
    data = record.to_dict()

    assert data["feedback_id"] == "abc123"
    assert data["query_text"] == "Test query"
    assert data["user_rating"] == 1.0
    assert "timestamp" in data


def test_learning_signals_serialization():
    """Test LearningSignals to_dict serialization."""
    signal = LearningSignals(
        tool_name="answer",
        total_samples=50,
        successful_samples=42,
        failed_samples=8,
        avg_rating=0.85,
        avg_confidence=0.88,
        success_rate=0.84,
        alpha=43.0,
        beta=9.0,
        expected_reward=0.827,
        window_start=datetime.now(),
        window_end=datetime.now()
    )

    # Serialize
    data = signal.to_dict()

    assert data["tool_name"] == "answer"
    assert data["total_samples"] == 50
    assert data["success_rate"] == 0.84
    assert "window_start" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
