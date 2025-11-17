"""
DreamEngine Integration Tests
==============================

Tests that verify all synthesis components work together correctly.

Author: HoloLoom Team
Date: 2025-11-17
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any

from HoloLoom.synthesis.pattern_synthesis import (
    PatternSynthesizer,
    PatternSynthesisConfig,
    synthesize_patterns_from_queries
)
from HoloLoom.synthesis.contradiction_detection import (
    ContradictionDetector,
    ContradictionConfig,
    detect_contradictions_in_memories
)
from HoloLoom.synthesis.gap_identification import (
    GapIdentifier,
    GapIdentificationConfig,
    identify_knowledge_gaps
)
from HoloLoom.synthesis.background_scheduler import (
    BackgroundScheduler,
    SchedulerConfig,
    TriggerStrategy,
    run_synthesis_cycle
)
from HoloLoom.synthesis.types import (
    SynthesisType,
    ContradictionType,
    GapType
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_queries() -> List[Dict[str, Any]]:
    """Create sample queries for testing."""
    return [
        {
            'text': 'What is Thompson Sampling?',
            'confidence': 0.92,
            'timestamp': datetime(2025, 11, 1, 10, 0),
            'entities': ['Thompson Sampling'],
            'motifs': ['exploration', 'exploitation']
        },
        {
            'text': 'How does Thompson Sampling work?',
            'confidence': 0.89,
            'timestamp': datetime(2025, 11, 1, 10, 15),
            'entities': ['Thompson Sampling'],
            'motifs': ['algorithm', 'bayesian']
        },
        {
            'text': 'Thompson vs epsilon-greedy',
            'confidence': 0.91,
            'timestamp': datetime(2025, 11, 1, 10, 30),
            'entities': ['Thompson Sampling', 'epsilon-greedy'],
            'motifs': ['comparison', 'exploration']
        },
        {
            'text': 'What is deep learning?',
            'confidence': 0.87,
            'timestamp': datetime(2025, 11, 1, 11, 0),
            'entities': ['deep learning'],
            'motifs': ['neural networks']
        }
    ]


@pytest.fixture
def sample_memories() -> List[Dict[str, Any]]:
    """Create sample memories for testing."""
    return [
        {
            'id': 'm1',
            'content': 'I love Python for data science',
            'timestamp': datetime(2025, 1, 1)
        },
        {
            'id': 'm2',
            'content': 'Python is too slow for production',
            'timestamp': datetime(2025, 3, 1)
        },
        {
            'id': 'm3',
            'content': 'Paris population is 2 million',
            'timestamp': datetime(2025, 2, 1)
        },
        {
            'id': 'm4',
            'content': 'Paris population is 2.1 million',
            'timestamp': datetime(2025, 4, 1)
        }
    ]


@pytest.fixture
def sample_knowledge_graph() -> Dict[str, Any]:
    """Create sample knowledge graph for testing."""
    return {
        'nodes': [
            'machine learning',
            'neural networks',
            'deep learning',
            'python',
            'javascript'
        ],
        'edges': [
            ('deep learning', 'neural networks', 'USES'),
            ('neural networks', 'machine learning', 'IS_A'),
            ('deep learning', 'machine learning', 'IS_A')
        ],
        'node_counts': {
            'machine learning': 10,
            'neural networks': 8,
            'deep learning': 5,
            'python': 3,
            'javascript': 2
        }
    }


# ============================================================================
# Pattern Synthesis Tests
# ============================================================================

def test_pattern_synthesizer_initialization():
    """Test PatternSynthesizer initializes correctly."""
    synthesizer = PatternSynthesizer()
    assert synthesizer.config.enabled is True
    assert len(synthesizer.query_history) == 0


def test_pattern_synthesis_clustering(sample_queries):
    """Test that similar queries are clustered together."""
    synthesizer = PatternSynthesizer()

    # Add queries
    for query in sample_queries:
        synthesizer.add_query(query)

    # Synthesize patterns
    patterns = synthesizer.synthesize_patterns()

    # Should find Thompson Sampling cluster (3 related queries)
    assert len(patterns) >= 1

    # Check pattern content
    thompson_pattern = [p for p in patterns if 'Thompson Sampling' in p.content]
    assert len(thompson_pattern) > 0

    # Check metadata
    pattern = thompson_pattern[0]
    assert pattern.synthesis_type == SynthesisType.PATTERN
    assert pattern.confidence >= 0.85
    assert len(pattern.source_memories) >= 3


def test_pattern_synthesis_standalone_function(sample_queries):
    """Test standalone pattern synthesis function."""
    patterns = synthesize_patterns_from_queries(sample_queries)

    assert len(patterns) >= 1
    assert all(p.synthesis_type == SynthesisType.PATTERN for p in patterns)


def test_pattern_synthesis_window_limit():
    """Test that query history respects window size."""
    config = PatternSynthesisConfig(window_size=10)
    synthesizer = PatternSynthesizer(config)

    # Add 20 queries
    for i in range(20):
        synthesizer.add_query({
            'text': f'Query {i}',
            'confidence': 0.9,
            'timestamp': datetime.now()
        })

    # Should only keep last 10
    assert len(synthesizer.query_history) == 10


# ============================================================================
# Contradiction Detection Tests
# ============================================================================

def test_contradiction_detector_initialization():
    """Test ContradictionDetector initializes correctly."""
    detector = ContradictionDetector()
    assert detector.config.enabled is True
    assert len(detector.memory_history) == 0


def test_contradiction_detection_sentiment_reversal(sample_memories):
    """Test detection of sentiment reversals."""
    detector = ContradictionDetector()

    # Add memories with sentiment reversal
    detector.add_memory(sample_memories[0])  # "I love Python"
    detector.add_memory(sample_memories[1])  # "Python is too slow"

    contradictions = detector.detect_contradictions()

    # Should detect preference reversal
    assert len(contradictions) > 0
    assert any(c.type == ContradictionType.PREFERENCE_REVERSAL for c in contradictions)


def test_contradiction_detection_fact_update(sample_memories):
    """Test detection of fact updates."""
    detector = ContradictionDetector()

    # Add memories with fact update
    detector.add_memory(sample_memories[2])  # "Paris population is 2 million"
    detector.add_memory(sample_memories[3])  # "Paris population is 2.1 million"

    contradictions = detector.detect_contradictions()

    # Should detect fact update
    assert len(contradictions) > 0
    assert any(c.type == ContradictionType.FACT_UPDATE for c in contradictions)


def test_contradiction_detection_standalone_function(sample_memories):
    """Test standalone contradiction detection function."""
    contradictions = detect_contradictions_in_memories(sample_memories)

    assert len(contradictions) >= 2  # Sentiment reversal + fact update
    assert all(c.score >= 0.5 for c in contradictions)


def test_contradiction_detection_lookback_window():
    """Test that old memories are pruned from history."""
    config = ContradictionConfig(lookback_days=30)
    detector = ContradictionDetector(config)

    # Add old memory (outside window)
    detector.add_memory({
        'id': 'old',
        'content': 'Old memory',
        'timestamp': datetime.now() - timedelta(days=100)
    })

    # Add recent memory
    detector.add_memory({
        'id': 'recent',
        'content': 'Recent memory',
        'timestamp': datetime.now()
    })

    # Should only keep recent memory
    assert len(detector.memory_history) == 1
    assert detector.memory_history[0]['id'] == 'recent'


# ============================================================================
# Gap Identification Tests
# ============================================================================

def test_gap_identifier_initialization():
    """Test GapIdentifier initializes correctly."""
    identifier = GapIdentifier()
    assert identifier.config.enabled is True
    assert len(identifier.related_concepts) > 0


def test_gap_identification_missing_prerequisites(sample_knowledge_graph):
    """Test detection of missing prerequisites."""
    identifier = GapIdentifier()

    gaps = identifier.identify_gaps(sample_knowledge_graph)

    # Should detect missing prerequisites for deep learning
    # (knows deep learning but missing linear algebra, calculus)
    prerequisite_gaps = [g for g in gaps if g.gap_type == GapType.MISSING_PREREQUISITE]
    assert len(prerequisite_gaps) > 0


def test_gap_identification_incomplete_categories(sample_knowledge_graph):
    """Test detection of incomplete categories."""
    identifier = GapIdentifier()

    gaps = identifier.identify_gaps(sample_knowledge_graph)

    # Should detect incomplete programming knowledge
    # (knows python and javascript but missing typescript)
    category_gaps = [g for g in gaps if g.gap_type == GapType.INCOMPLETE_CATEGORY]
    assert len(category_gaps) > 0


def test_gap_identification_standalone_function(sample_knowledge_graph):
    """Test standalone gap identification function."""
    gaps = identify_knowledge_gaps(sample_knowledge_graph)

    assert len(gaps) > 0
    assert all(g.importance >= 0.5 for g in gaps)
    assert all(len(g.suggested_queries) > 0 for g in gaps)


def test_gap_identification_importance_scoring():
    """Test that gaps are sorted by importance."""
    identifier = GapIdentifier()

    graph = {
        'nodes': ['deep learning', 'python'],
        'edges': [],
        'node_counts': {'deep learning': 10, 'python': 5}
    }

    gaps = identifier.identify_gaps(graph)

    # Gaps should be sorted by importance (descending)
    for i in range(len(gaps) - 1):
        assert gaps[i].importance >= gaps[i + 1].importance


# ============================================================================
# Background Scheduler Tests
# ============================================================================

def test_background_scheduler_initialization():
    """Test BackgroundScheduler initializes correctly."""
    scheduler = BackgroundScheduler()
    assert scheduler.config.enabled is True
    assert scheduler.is_running is False
    assert scheduler.total_cycles == 0


@pytest.mark.asyncio
async def test_background_scheduler_manual_trigger(sample_knowledge_graph):
    """Test manual synthesis trigger."""
    scheduler = BackgroundScheduler()

    # Register some activity
    for _ in range(15):
        scheduler.register_memory()

    # Manually trigger synthesis
    results = await scheduler.trigger_synthesis(
        knowledge_graph=sample_knowledge_graph
    )

    assert results is not None
    assert results.cycle_duration_ms > 0
    assert scheduler.total_cycles == 1


@pytest.mark.asyncio
async def test_background_scheduler_periodic_strategy():
    """Test periodic triggering strategy."""
    config = SchedulerConfig(
        trigger_strategy=TriggerStrategy.PERIODIC,
        periodic_interval_seconds=0.5  # Fast for testing
    )
    scheduler = BackgroundScheduler(config=config)

    # Start scheduler
    await scheduler.start()

    # Wait for at least one cycle
    await asyncio.sleep(1.0)

    # Stop scheduler
    await scheduler.stop()

    # Should have run at least one cycle
    assert scheduler.total_cycles >= 1


@pytest.mark.asyncio
async def test_background_scheduler_threshold_strategy():
    """Test threshold triggering strategy."""
    config = SchedulerConfig(
        trigger_strategy=TriggerStrategy.THRESHOLD,
        threshold_new_memories=5
    )
    scheduler = BackgroundScheduler(config=config)

    # Start scheduler
    await scheduler.start()

    # Register memories to hit threshold
    for _ in range(6):
        scheduler.register_memory()

    # Wait for synthesis to trigger
    await asyncio.sleep(1.0)

    # Stop scheduler
    await scheduler.stop()

    # Should have run at least one cycle
    assert scheduler.total_cycles >= 1


@pytest.mark.asyncio
async def test_background_scheduler_statistics():
    """Test scheduler statistics."""
    scheduler = BackgroundScheduler()

    # Register activity
    scheduler.register_query()
    scheduler.register_memory()

    stats = scheduler.get_statistics()

    assert stats['is_running'] is False
    assert stats['total_cycles'] == 0
    assert stats['memories_since_last_synthesis'] == 1
    assert stats['queries_since_last_synthesis'] == 1


@pytest.mark.asyncio
async def test_run_synthesis_cycle_standalone(sample_knowledge_graph):
    """Test standalone synthesis cycle function."""
    synthesizer = PatternSynthesizer()
    detector = ContradictionDetector()
    identifier = GapIdentifier()

    results = await run_synthesis_cycle(
        synthesizer,
        detector,
        identifier,
        sample_knowledge_graph
    )

    assert results is not None
    assert results.cycle_duration_ms > 0
    assert isinstance(results.patterns_synthesized, list)
    assert isinstance(results.contradictions_detected, list)
    assert isinstance(results.gaps_identified, list)


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_dreamengine_pipeline(
    sample_queries,
    sample_memories,
    sample_knowledge_graph
):
    """Test complete DreamEngine pipeline."""
    # Create components
    synthesizer = PatternSynthesizer()
    detector = ContradictionDetector()
    identifier = GapIdentifier()

    # Add data
    for query in sample_queries:
        synthesizer.add_query(query)

    for memory in sample_memories:
        detector.add_memory(memory)

    # Run synthesis cycle
    results = await run_synthesis_cycle(
        synthesizer,
        detector,
        identifier,
        sample_knowledge_graph
    )

    # Verify results
    assert len(results.patterns_synthesized) > 0
    assert len(results.contradictions_detected) > 0
    assert len(results.gaps_identified) > 0

    # Check summary
    summary = results.summary()
    assert 'Patterns Synthesized' in summary
    assert 'Contradictions Detected' in summary
    assert 'Knowledge Gaps Identified' in summary


@pytest.mark.asyncio
async def test_scheduler_with_callback(sample_knowledge_graph):
    """Test scheduler callback mechanism."""
    callback_results = []

    def on_cycle_complete(results):
        callback_results.append(results)

    scheduler = BackgroundScheduler(
        on_cycle_complete=on_cycle_complete
    )

    # Register activity
    for _ in range(15):
        scheduler.register_memory()

    # Trigger synthesis
    await scheduler.trigger_synthesis(
        knowledge_graph=sample_knowledge_graph
    )

    # Callback should have been called
    assert len(callback_results) == 1
    assert isinstance(callback_results[0], SynthesisResults)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
