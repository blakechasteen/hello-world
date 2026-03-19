"""
Unit Tests for SpinnerProtocol
===============================

Tests the standardized spinner protocol including:
- SpinnerProtocol interface
- BaseSpinner implementation
- ImportanceScorer framework
- Checkpoint management
- Streaming utilities

Author: Claude Code
Date: November 2025
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import List, Any, AsyncIterator

from hololoom.protocols.types import MemoryShard
from hololoom.spinningWheel.protocol import (
    SpinnerStatus,
    SpinnerCapabilities,
    SpinResult,
    ImportanceSignals,
    ImportanceScore,
    SpinnerCheckpoint,
    SpinnerProtocol,
    BaseSpinner
)
from hololoom.spinningWheel.importance import (
    LengthScorer,
    TechnicalScorer,
    StructuralScorer,
    AuthorityScorer,
    RecencyScorer,
    EngagementScorer,
    NoiseDetector,
    ImportanceScorer,
    create_chat_scorer
)
from hololoom.spinningWheel.utils import (
    CheckpointManager,
    StreamBuffer,
    BatchProcessor,
    ShardDeduplicator,
    ErrorHandler,
    ProgressTracker,
    create_source_id
)


# ============================================================================
# Mock Spinner for Testing
# ============================================================================

class MockSpinner(BaseSpinner):
    """Mock spinner for testing BaseSpinner functionality."""

    def __init__(self, **kwargs):
        super().__init__(name="mock", **kwargs)
        self.available = True

    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=True,
            incremental=True,
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True,
            embeddings=False,
            supported_formats=['txt', 'md']
        )

    def is_available(self) -> bool:
        return self.available

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """Create mock shards."""
        if isinstance(source, str):
            shard = self._create_shard(
                id_suffix="test_001",
                text=source,
                episode="test_episode",
                entities=["Entity1", "Entity2"],
                motifs=["motif1", "motif2"],
                metadata={'test': True, 'importance_score': 0.8}
            )
            return [shard]

        return []


# ============================================================================
# Test SpinnerProtocol & BaseSpinner
# ============================================================================

@pytest.mark.asyncio
async def test_base_spinner_initialization():
    """Test BaseSpinner initialization."""
    spinner = MockSpinner(
        importance_threshold=0.5,
        checkpoint_dir="/tmp/checkpoints"
    )

    assert spinner.get_name() == "mock"
    assert spinner.importance_threshold == 0.5
    assert spinner.is_available() is True


@pytest.mark.asyncio
async def test_base_spinner_capabilities():
    """Test capabilities reporting."""
    spinner = MockSpinner()
    capabilities = spinner.get_capabilities()

    assert capabilities.basic_processing is True
    assert capabilities.streaming is True
    assert 'txt' in capabilities.supported_formats


@pytest.mark.asyncio
async def test_base_spinner_status():
    """Test status reporting."""
    spinner = MockSpinner()

    # Available
    assert spinner.get_status() == SpinnerStatus.AVAILABLE

    # Unavailable
    spinner.available = False
    assert spinner.get_status() == SpinnerStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_base_spinner_spin():
    """Test spin operation."""
    spinner = MockSpinner()
    result = await spinner.spin("Test content")

    assert result.success is True
    assert result.shard_count == 1
    assert result.shards[0].text == "Test content"
    assert result.processing_time_ms >= 0


@pytest.mark.asyncio
async def test_base_spinner_importance_filtering():
    """Test importance threshold filtering."""
    spinner = MockSpinner(importance_threshold=0.9)  # High threshold

    result = await spinner.spin("Test content")

    # Should filter out shard with importance 0.8
    assert result.shard_count == 0
    assert len(result.warnings) > 0
    assert "filtered" in result.warnings[0].lower()


@pytest.mark.asyncio
async def test_base_spinner_error_handling():
    """Test error handling."""
    class FailingSpinner(MockSpinner):
        async def _spin_impl(self, source, **kwargs):
            raise ValueError("Test error")

    spinner = FailingSpinner()
    result = await spinner.spin("Test")

    assert result.success is False
    assert result.error_message == "Test error"
    assert result.shard_count == 0


# ============================================================================
# Test ImportanceSignals & ImportanceScore
# ============================================================================

def test_importance_signals_creation():
    """Test ImportanceSignals creation."""
    signals = ImportanceSignals(
        length_score=0.8,
        technical_score=0.9,
        authority_score=0.7,
        noise_penalty=-0.2
    )

    assert signals.length_score == 0.8
    assert signals.technical_score == 0.9
    assert signals.noise_penalty == -0.2


def test_importance_signals_compute_total():
    """Test total score computation."""
    signals = ImportanceSignals(
        length_score=0.8,
        technical_score=0.9,
        structural_score=0.7,
        authority_score=0.8
    )

    total = signals.compute_total()

    # Should be weighted average
    assert 0.0 <= total <= 1.0
    assert total > 0.5  # High scores


def test_importance_signals_with_penalty():
    """Test score with noise penalty."""
    signals = ImportanceSignals(
        length_score=0.8,
        technical_score=0.8,
        noise_penalty=-0.5  # Heavy penalty
    )

    total = signals.compute_total()

    # Penalty should reduce score
    assert total < 0.6


def test_importance_signals_explain():
    """Test explanation generation."""
    signals = ImportanceSignals(
        length_score=0.9,
        technical_score=0.8,
        authority_score=0.75
    )

    explanation = signals.explain()

    assert "substantive" in explanation or "technical" in explanation or "authoritative" in explanation


def test_importance_score_validation():
    """Test ImportanceScore validation."""
    # Valid score
    score = ImportanceScore(
        score=0.75,
        signals=ImportanceSignals(),
        reason="test"
    )
    assert score.score == 0.75

    # Invalid score (should raise)
    with pytest.raises(ValueError):
        ImportanceScore(score=1.5, signals=ImportanceSignals(), reason="test")


# ============================================================================
# Test Individual Scorers
# ============================================================================

def test_length_scorer():
    """Test LengthScorer."""
    scorer = LengthScorer(min_length=50, optimal_length=300)

    # Very short
    assert scorer.score("short") < 0.3

    # Optimal
    text_optimal = "a" * 300
    assert scorer.score(text_optimal) >= 0.9

    # Very long
    text_long = "a" * 10000
    score_long = scorer.score(text_long)
    assert 0.7 <= score_long <= 1.0  # Diminishing returns


def test_technical_scorer():
    """Test TechnicalScorer."""
    scorer = TechnicalScorer()

    # Technical text
    technical_text = "Thompson Sampling is a Bayesian policy for multi-armed bandits using neural embeddings."
    score_tech = scorer.score(technical_text)

    # Non-technical text
    non_tech_text = "Hello, how are you today? The weather is nice."
    score_non_tech = scorer.score(non_tech_text)

    assert score_tech > score_non_tech
    assert score_tech > 0.3


def test_technical_scorer_custom_terms():
    """Test TechnicalScorer with custom terms."""
    scorer = TechnicalScorer()
    scorer.add_terms(["quantum", "entanglement"])

    text = "Quantum entanglement is a physical phenomenon."
    score = scorer.score(text)

    assert score > 0.2


def test_structural_scorer():
    """Test StructuralScorer."""
    scorer = StructuralScorer()

    # Well-structured text
    structured_text = """
# Header

This is a paragraph.

- Bullet point 1
- Bullet point 2

```python
code block
```
"""

    # Unstructured text
    unstructured_text = "Just a single line of text"

    score_structured = scorer.score(structured_text)
    score_unstructured = scorer.score(unstructured_text)

    assert score_structured > score_unstructured
    assert score_structured > 0.5


def test_authority_scorer():
    """Test AuthorityScorer."""
    scorer = AuthorityScorer(authority_map={"expert@": 0.9, "newbie@": 0.3})

    # High authority
    score_high = scorer.score("expert@company.com")
    # Low authority
    score_low = scorer.score("newbie@company.com")

    assert score_high > score_low
    assert score_high >= 0.9


def test_recency_scorer():
    """Test RecencyScorer."""
    scorer = RecencyScorer(half_life_days=30.0)

    # Current time
    score_now = scorer.score(time.time())
    assert score_now >= 0.99

    # 30 days ago (half-life)
    thirty_days_ago = time.time() - (30 * 86400)
    score_half = scorer.score(thirty_days_ago)
    assert 0.45 <= score_half <= 0.55  # ~0.5

    # Very old
    one_year_ago = time.time() - (365 * 86400)
    score_old = scorer.score(one_year_ago)
    assert score_old < 0.1


def test_engagement_scorer():
    """Test EngagementScorer."""
    scorer = EngagementScorer()

    # High engagement
    score_high = scorer.score(likes=50, shares=10, replies=20)
    # Low engagement
    score_low = scorer.score(likes=1, shares=0, replies=0)

    assert score_high > score_low
    assert score_high > 0.5


def test_noise_detector():
    """Test NoiseDetector."""
    detector = NoiseDetector()

    # Greeting (short)
    penalty_greeting = detector.detect("Hi")
    assert penalty_greeting < 0  # Penalty

    # Spam
    penalty_spam = detector.detect("Click here buy now!!!")
    assert penalty_spam < 0

    # Normal text
    penalty_normal = detector.detect("This is a substantive message about Thompson Sampling.")
    assert penalty_normal == 0.0  # No penalty


# ============================================================================
# Test ImportanceScorer (Integrated)
# ============================================================================

def test_importance_scorer():
    """Test full ImportanceScorer."""
    scorer = ImportanceScorer()

    text = """
# Thompson Sampling

Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
It balances exploration and exploitation using posterior sampling.

Key features:
- Bayesian framework
- Posterior sampling
- Optimal regret bounds
"""

    importance = scorer.score(
        text=text,
        source="research_paper.pdf",
        timestamp=time.time(),
        engagement={'likes': 10, 'shares': 5}
    )

    assert 0.0 <= importance.score <= 1.0
    assert importance.score > 0.5  # Should be high importance
    assert len(importance.reason) > 0


def test_importance_scorer_batch():
    """Test batch scoring."""
    scorer = ImportanceScorer()

    items = [
        {'text': 'Thompson Sampling is great', 'source': 'paper.pdf'},
        {'text': 'Hi', 'source': 'chat.txt'},
        {'text': 'A comprehensive analysis of policy optimization', 'source': 'thesis.pdf'}
    ]

    results = scorer.score_batch(items)

    assert len(results) == 3
    assert all(isinstance(r, ImportanceScore) for r in results)

    # Thesis should have highest score
    assert results[2].score > results[1].score


def test_preset_scorers():
    """Test preset scorer configurations."""
    chat_scorer = create_chat_scorer()

    # Should have chat-specific terms
    text = "Let's refactor this feature and deploy to production"
    score = chat_scorer.score(text)

    assert score.score > 0.3


# ============================================================================
# Test Checkpoint Management
# ============================================================================

def test_spinner_checkpoint_creation():
    """Test SpinnerCheckpoint creation."""
    checkpoint = SpinnerCheckpoint(
        spinner_name="git",
        source_id="repo_xyz",
        last_processed_id="commit_abc123",
        items_processed=100,
        items_total=500
    )

    assert checkpoint.spinner_name == "git"
    assert checkpoint.progress_percentage() == 20.0


def test_spinner_checkpoint_serialization():
    """Test checkpoint serialization."""
    checkpoint = SpinnerCheckpoint(
        spinner_name="git",
        source_id="repo_xyz",
        last_processed_id="commit_abc123",
        state={'branch': 'main'}
    )

    # Serialize
    data = checkpoint.to_dict()
    assert data['spinner_name'] == "git"
    assert data['state']['branch'] == 'main'

    # Deserialize
    restored = SpinnerCheckpoint.from_dict(data)
    assert restored.spinner_name == checkpoint.spinner_name
    assert restored.state == checkpoint.state


def test_checkpoint_manager(tmp_path):
    """Test CheckpointManager."""
    manager = CheckpointManager(tmp_path)

    checkpoint = SpinnerCheckpoint(
        spinner_name="test",
        source_id="source_123",
        items_processed=50
    )

    # Save
    manager.save(checkpoint)

    # Load
    loaded = manager.load("test", "source_123")
    assert loaded is not None
    assert loaded.items_processed == 50

    # List
    checkpoints = manager.list_checkpoints()
    assert len(checkpoints) == 1

    # Delete
    manager.delete("test", "source_123")
    assert manager.load("test", "source_123") is None


# ============================================================================
# Test Streaming Utilities
# ============================================================================

@pytest.mark.asyncio
async def test_stream_buffer():
    """Test StreamBuffer."""
    flushed_batches = []

    async def flush_callback(shards):
        flushed_batches.append(shards.copy())

    buffer = StreamBuffer(
        batch_size=3,
        flush_callback=flush_callback
    )

    # Add shards
    for i in range(7):
        shard = MemoryShard(
            id=f"shard_{i}",
            text=f"Shard {i}",
            episode="test",
            entities=[],
            motifs=[]
        )
        await buffer.add(shard)

    # Should have flushed 2 batches (3 + 3)
    assert len(flushed_batches) == 2
    assert len(flushed_batches[0]) == 3

    # Final flush
    await buffer.flush()
    assert len(flushed_batches) == 3  # Last batch with 1 shard


# ============================================================================
# Test Batch Processing
# ============================================================================

@pytest.mark.asyncio
async def test_batch_processor():
    """Test BatchProcessor."""
    async def double(x):
        await asyncio.sleep(0.01)  # Simulate work
        return x * 2

    processor = BatchProcessor(max_concurrent=3)
    items = [1, 2, 3, 4, 5]

    results = await processor.process(items, double)

    assert results == [2, 4, 6, 8, 10]


@pytest.mark.asyncio
async def test_batch_processor_with_errors():
    """Test BatchProcessor error handling."""
    async def flaky_processor(x):
        if x == 3:
            raise ValueError("Error on 3")
        return x * 2

    processor = BatchProcessor(max_concurrent=2, retry_on_error=False)
    items = [1, 2, 3, 4]

    results = await processor.process(items, flaky_processor)

    # Should have None for failed item
    assert results[0] == 2
    assert results[1] == 4
    assert results[2] is None  # Failed
    assert results[3] == 8


# ============================================================================
# Test Deduplication
# ============================================================================

def test_shard_deduplicator():
    """Test ShardDeduplicator."""
    deduplicator = ShardDeduplicator()

    shard1 = MemoryShard(
        id="shard_1",
        text="Unique content",
        episode="test",
        entities=[],
        motifs=[]
    )

    shard2 = MemoryShard(
        id="shard_2",
        text="Unique content",  # Same as shard1
        episode="test",
        entities=[],
        motifs=[]
    )

    shard3 = MemoryShard(
        id="shard_3",
        text="Different content",
        episode="test",
        entities=[],
        motifs=[]
    )

    # Check duplicates
    assert not deduplicator.is_duplicate(shard1)
    deduplicator.mark_seen(shard1)

    assert deduplicator.is_duplicate(shard2)  # Same text as shard1
    assert not deduplicator.is_duplicate(shard3)


def test_shard_deduplicator_filter():
    """Test filter_duplicates."""
    deduplicator = ShardDeduplicator()

    shards = [
        MemoryShard(id="1", text="A", episode="test", entities=[], motifs=[]),
        MemoryShard(id="2", text="B", episode="test", entities=[], motifs=[]),
        MemoryShard(id="3", text="A", episode="test", entities=[], motifs=[]),  # Duplicate
        MemoryShard(id="4", text="C", episode="test", entities=[], motifs=[]),
    ]

    unique = deduplicator.filter_duplicates(shards)

    assert len(unique) == 3  # A, B, C
    assert unique[0].text == "A"
    assert unique[1].text == "B"
    assert unique[2].text == "C"


# ============================================================================
# Test Error Handling
# ============================================================================

def test_error_handler():
    """Test ErrorHandler."""
    handler = ErrorHandler(log_errors=False)

    error = ValueError("Test error")
    error_shard = handler.handle(error, context="processing file X")

    assert error_shard is not None
    assert "Test error" in error_shard.text
    assert len(handler.get_errors()) == 1


def test_error_handler_critical():
    """Test critical error handling."""
    handler = ErrorHandler(raise_on_critical=True)

    error = ValueError("Critical error")

    with pytest.raises(ValueError):
        handler.handle(error, context="test", critical=True)


# ============================================================================
# Test Progress Tracking
# ============================================================================

def test_progress_tracker():
    """Test ProgressTracker."""
    tracker = ProgressTracker(total=100)

    # Initial state
    assert tracker.percentage() == 0.0

    # Increment
    tracker.increment(25)
    assert tracker.percentage() == 25.0

    # Complete
    tracker.increment(75)
    assert tracker.percentage() == 100.0


def test_progress_tracker_time_estimation():
    """Test time estimation."""
    tracker = ProgressTracker(total=100)

    # No estimate at start
    assert tracker.estimated_remaining() is None

    # Simulate progress
    tracker.increment(50)
    time.sleep(0.1)

    remaining = tracker.estimated_remaining()
    assert remaining is not None
    assert remaining > 0


def test_progress_tracker_format():
    """Test status formatting."""
    tracker = ProgressTracker(total=100)
    tracker.increment(25)
    time.sleep(0.01)

    status = tracker.format_status()

    assert "25/100" in status
    assert "25.0%" in status


# ============================================================================
# Test Utility Functions
# ============================================================================

def test_create_source_id():
    """Test source ID creation."""
    # String source
    id1 = create_source_id("/path/to/source")
    assert len(id1) == 16  # MD5 hash truncated

    # Same source = same ID
    id2 = create_source_id("/path/to/source")
    assert id1 == id2

    # Different source = different ID
    id3 = create_source_id("/different/path")
    assert id1 != id3


# ============================================================================
# Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_full_spinner_workflow(tmp_path):
    """Test complete spinner workflow."""
    # Create spinner with checkpoint
    spinner = MockSpinner(
        importance_threshold=0.5,
        checkpoint_dir=tmp_path
    )

    # Check capabilities
    capabilities = spinner.get_capabilities()
    assert capabilities.basic_processing

    # Process input
    result = await spinner.spin("Thompson Sampling is a Bayesian approach.")

    # Verify result
    assert result.success
    assert result.shard_count == 1

    # Score importance
    importance = spinner.score_importance(result.shards[0])
    assert 0.0 <= importance.score <= 1.0

    # Create checkpoint
    source_id = create_source_id("test_source")
    checkpoint = SpinnerCheckpoint(
        spinner_name=spinner.get_name(),
        source_id=source_id,
        items_processed=1,
        items_total=10
    )
    spinner.save_checkpoint(checkpoint)

    # Load checkpoint
    loaded = spinner.load_checkpoint(source_id)
    assert loaded is not None
    assert loaded.items_processed == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
