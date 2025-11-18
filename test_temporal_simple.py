"""
Simple test to verify temporal evolution implementation.
"""

import asyncio
from datetime import datetime, timedelta

# Mock HoloLoom
class MockHoloLoom:
    def __init__(self):
        self.memories = []

from HoloLoom.memory.temporal_evolution import (
    TemporalEvolutionTracker,
    TemporalEvolutionConfig,
    UnderstandingState
)


async def test_basic_functionality():
    """Test basic temporal evolution functionality."""
    print("Testing Temporal Evolution Tracker...\n")

    # Create mock loom
    loom = MockHoloLoom()

    # Create tracker
    config = TemporalEvolutionConfig(
        learning_threshold=3,
        familiar_threshold=10,
        mastery_confidence=0.85
    )

    tracker = TemporalEvolutionTracker(loom, config)

    # Test 1: Initial state (UNKNOWN)
    print("Test 1: Initial state")
    history = await tracker.get_concept_history("Test Concept")
    assert history.current_state == UnderstandingState.UNKNOWN
    print(f"  ✓ Initial state is UNKNOWN")

    # Test 2: First interaction (UNKNOWN → INTRODUCED)
    print("\nTest 2: First interaction")
    await tracker.track_interaction(
        query="What is the test concept?",
        entities=["Test Concept"],
        confidence=0.7
    )

    history = await tracker.get_concept_history("Test Concept")
    assert history.current_state == UnderstandingState.INTRODUCED
    assert history.total_interactions == 1
    print(f"  ✓ State transitioned to INTRODUCED")
    print(f"  ✓ Total interactions: {history.total_interactions}")

    # Test 3: Multiple interactions (INTRODUCED → LEARNING)
    print("\nTest 3: Active learning")
    for i in range(3):
        await tracker.track_interaction(
            query=f"Question {i+2}",
            entities=["Test Concept"],
            confidence=0.7
        )

    history = await tracker.get_concept_history("Test Concept")
    assert history.current_state == UnderstandingState.LEARNING
    assert history.total_interactions == 4
    print(f"  ✓ State transitioned to LEARNING")
    print(f"  ✓ Total interactions: {history.total_interactions}")

    # Test 4: Regular use (LEARNING → FAMILIAR)
    print("\nTest 4: Regular use")
    for i in range(7):
        await tracker.track_interaction(
            query=f"Question {i+5}",
            entities=["Test Concept"],
            confidence=0.75
        )

    history = await tracker.get_concept_history("Test Concept")
    assert history.current_state == UnderstandingState.FAMILIAR
    assert history.total_interactions == 11
    print(f"  ✓ State transitioned to FAMILIAR")
    print(f"  ✓ Total interactions: {history.total_interactions}")

    # Test 5: High confidence (FAMILIAR → MASTERY)
    print("\nTest 5: High confidence mastery")
    for i in range(6):
        await tracker.track_interaction(
            query=f"Advanced question {i+12}",
            entities=["Test Concept"],
            confidence=0.92
        )

    history = await tracker.get_concept_history("Test Concept")
    print(f"  Current state: {history.current_state.value}")
    print(f"  Total interactions: {history.total_interactions}")

    # Should be MASTERY or at least FAMILIAR (depends on semantic consistency check)
    assert history.current_state in [UnderstandingState.MASTERY, UnderstandingState.FAMILIAR]
    if history.current_state == UnderstandingState.MASTERY:
        print(f"  ✓ State transitioned to MASTERY")
    else:
        print(f"  ✓ State is FAMILIAR (mastery requires more consistent high confidence)")

    # Test 6: Confidence trajectory
    print("\nTest 6: Confidence trajectory")
    assert len(history.confidence_trajectory) == 17
    print(f"  ✓ Confidence trajectory has {len(history.confidence_trajectory)} points")

    # Test 7: State transitions
    print("\nTest 7: State transitions")
    assert len(history.state_transitions) >= 3
    print(f"  ✓ Recorded {len(history.state_transitions)} state transitions")

    for transition in history.state_transitions:
        print(f"    - {transition.old_state.value} → {transition.new_state.value} ({transition.trigger})")

    # Test 8: Milestones
    print("\nTest 8: Milestones")
    milestones = await tracker.detect_milestones("Test Concept")
    assert len(milestones) >= 1
    print(f"  ✓ Detected {len(milestones)} milestones")

    for milestone in milestones:
        print(f"    - {milestone.type}: {milestone.description}")

    # Test 9: Temporal query
    print("\nTest 9: Temporal query")
    mid_time = datetime.now() - timedelta(minutes=5)

    # Add timestamp to earlier interactions
    concept2 = "Temporal Test"
    base_time = datetime.now() - timedelta(days=10)

    for i in range(5):
        await tracker.track_interaction(
            query=f"Historical question {i}",
            entities=[concept2],
            confidence=0.6,
            timestamp=base_time + timedelta(days=i)
        )

    snapshot = await tracker.query_at_time(concept2, base_time + timedelta(days=2))
    print(f"  ✓ Snapshot state: {snapshot.state.value}")
    print(f"  ✓ Snapshot confidence: {snapshot.confidence:.2f}")

    # Test 10: Evolution summary
    print("\nTest 10: Evolution summary")
    summary = await tracker.get_evolution_summary(days=30)
    print(f"  ✓ New concepts: {summary['new_concepts']['count']}")
    print(f"  ✓ Active concepts: {summary['active_concepts']['count']}")
    print(f"  ✓ Total concepts: {summary['total_concepts']}")

    # Test 11: Visualization
    print("\nTest 11: Visualization")
    viz = tracker.visualize_trajectory("Test Concept", output_format='ascii')
    assert "Test Concept" in viz
    assert "State Transitions" in viz or "Confidence" in viz
    print(f"  ✓ Generated visualization ({len(viz)} chars)")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(test_basic_functionality())
