"""
Simple test runner for DreamEngine (no pytest required).

Author: HoloLoom Team
Date: 2025-11-17
"""

import asyncio
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


# Test data
def get_sample_queries():
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
        }
    ]


def get_sample_memories():
    now = datetime.now()
    return [
        {
            'id': 'm1',
            'content': 'I love Python for data science',
            'timestamp': now - timedelta(days=10)  # 10 days ago
        },
        {
            'id': 'm2',
            'content': 'Python is too slow for production',
            'timestamp': now - timedelta(days=5)  # 5 days ago
        }
    ]


def get_sample_knowledge_graph():
    return {
        'nodes': ['machine learning', 'neural networks', 'deep learning', 'python'],
        'edges': [
            ('deep learning', 'neural networks', 'USES'),
            ('neural networks', 'machine learning', 'IS_A')
        ],
        'node_counts': {
            'machine learning': 10,
            'neural networks': 8,
            'deep learning': 5
        }
    }


# Test functions
def test_pattern_synthesis():
    """Test pattern synthesis."""
    print("\n[TEST] Pattern Synthesis")

    synthesizer = PatternSynthesizer()
    queries = get_sample_queries()

    for query in queries:
        synthesizer.add_query(query)

    patterns = synthesizer.synthesize_patterns()

    assert len(patterns) >= 1, "Should synthesize at least one pattern"
    assert patterns[0].synthesis_type == SynthesisType.PATTERN
    print("  ✓ Pattern synthesis working")


def test_contradiction_detection():
    """Test contradiction detection."""
    print("\n[TEST] Contradiction Detection")

    detector = ContradictionDetector()
    memories = get_sample_memories()

    for memory in memories:
        detector.add_memory(memory)

    contradictions = detector.detect_contradictions()

    assert len(contradictions) > 0, "Should detect contradictions"
    assert contradictions[0].type == ContradictionType.PREFERENCE_REVERSAL
    print("  ✓ Contradiction detection working")


def test_gap_identification():
    """Test gap identification."""
    print("\n[TEST] Gap Identification")

    identifier = GapIdentifier()
    graph = get_sample_knowledge_graph()

    gaps = identifier.identify_gaps(graph)

    assert len(gaps) > 0, "Should identify knowledge gaps"
    assert gaps[0].importance >= 0.5
    print("  ✓ Gap identification working")


async def test_background_scheduler():
    """Test background scheduler."""
    print("\n[TEST] Background Scheduler")

    scheduler = BackgroundScheduler()

    # Register activity
    for _ in range(15):
        scheduler.register_memory()

    # Trigger synthesis
    results = await scheduler.trigger_synthesis(
        knowledge_graph=get_sample_knowledge_graph()
    )

    assert results is not None, "Should produce results"
    assert results.cycle_duration_ms > 0
    print("  ✓ Background scheduler working")


async def test_full_pipeline():
    """Test complete DreamEngine pipeline."""
    print("\n[TEST] Full Pipeline Integration")

    synthesizer = PatternSynthesizer()
    detector = ContradictionDetector()
    identifier = GapIdentifier()

    # Add data
    for query in get_sample_queries():
        synthesizer.add_query(query)

    for memory in get_sample_memories():
        detector.add_memory(memory)

    # Run synthesis
    results = await run_synthesis_cycle(
        synthesizer,
        detector,
        identifier,
        get_sample_knowledge_graph()
    )

    assert len(results.patterns_synthesized) > 0
    assert len(results.contradictions_detected) > 0
    assert len(results.gaps_identified) > 0

    print("  ✓ Full pipeline working")
    print(f"\n{results.summary()}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("DreamEngine Integration Tests")
    print("=" * 60)

    try:
        # Sync tests
        test_pattern_synthesis()
        test_contradiction_detection()
        test_gap_identification()

        # Async tests
        await test_background_scheduler()
        await test_full_pipeline()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())
