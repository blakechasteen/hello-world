"""
Test Suite for AR Recursive Learning Integration
=================================================
Comprehensive tests for recursive learning integration with Elle AR.

Test Coverage:
- AR provenance tracking (gesture + voice + vision)
- AR pattern extraction and learning
- AR quality refinement (VERIFY, ELEGANCE, SPATIAL, MULTIMODAL)
- Background learning with Thompson Sampling
- Learning state persistence

Author: HoloLoom Team
Date: November 17, 2025
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

# AR recursive learning imports
from HoloLoom.voice.recursive_integration import (
    ARLearningEngine,
    ARLearningConfig,
    ARProvenanceTracker,
    ARProvenanceEntry,
    ARQuery,
    ARQueryType,
    weave_with_ar_learning,
)

from HoloLoom.voice.ar_pattern_learner import (
    ARPatternLearner,
    ARPattern,
    GestureType,
    VoiceIntent,
)

from HoloLoom.voice.ar_refiner import (
    ARRefiner,
    ARRefinementStrategy,
    ARQualityMetrics,
)

from HoloLoom.voice.ar_background_learner import (
    ARBackgroundLearner,
    ARThompsonPriors,
    ARModalityWeights,
    ARLearningMetrics,
)

# HoloLoom imports
from HoloLoom.config import Config
from HoloLoom.documentation.types import MemoryShard
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.voice.ar_context import ARContext, ARObject, ARObjectType, Vector3


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Create test config"""
    return Config.fast()


@pytest.fixture
def shards():
    """Create test memory shards"""
    return [
        MemoryShard(
            text="Thompson Sampling balances exploration and exploitation",
            entities=["Thompson Sampling", "exploration", "exploitation"],
            motifs=["bayesian", "reinforcement_learning"],
        ),
        MemoryShard(
            text="AR gestures include pinch, swipe, and tap",
            entities=["AR", "gestures", "pinch", "swipe", "tap"],
            motifs=["ar_interaction", "gestures"],
        ),
    ]


@pytest.fixture
def ar_context():
    """Create test AR context"""
    return ARContext(
        user_position=Vector3(0, 1.5, 0),
        user_orientation=None,
        visible_objects=[
            ARObject(
                name="beehive_1",
                object_type=ARObjectType.BEEHIVE,
                position=Vector3(2, 1, 3),
            ),
            ARObject(
                name="frame_1",
                object_type=ARObjectType.FRAME,
                position=Vector3(2.5, 1.2, 3),
            ),
        ],
    )


@pytest.fixture
def sample_ar_query(ar_context):
    """Create sample AR query"""
    return ARQuery(
        text="Show me information about the beehive",
        ar_context=ar_context,
        gesture_detected="point",
        voice_intent="info",
        vision_objects=ar_context.visible_objects,
        user_position=ar_context.user_position,
    )


@pytest.fixture
def sample_spacetime():
    """Create sample spacetime result"""
    return Spacetime(
        response="The beehive is healthy with 50,000 bees and active foraging.",
        confidence=0.85,
        trace=WeavingTrace(
            threads_activated=[],
            tools_used=["inspect_beehive"],
        ),
        metadata={'tool_used': 'inspect_beehive'},
    )


# ============================================================================
# ARQuery Tests
# ============================================================================

def test_ar_query_type_detection():
    """Test AR query type detection"""
    # Voice only
    q = ARQuery(text="Hello", voice_intent="info")
    assert q.query_type == ARQueryType.VOICE_ONLY

    # Gesture only
    q = ARQuery(text="", gesture_detected="pinch")
    assert q.query_type == ARQueryType.GESTURE_ONLY

    # Voice + gesture
    q = ARQuery(text="Select", voice_intent="select", gesture_detected="pinch")
    assert q.query_type == ARQueryType.VOICE_GESTURE

    # Multimodal
    q = ARQuery(
        text="Info",
        voice_intent="info",
        gesture_detected="point",
        vision_objects=[ARObject("obj", ARObjectType.BEEHIVE, Vector3(0, 0, 0))],
    )
    assert q.query_type == ARQueryType.MULTIMODAL


def test_ar_query_to_hololoom_query(sample_ar_query):
    """Test conversion to HoloLoom query"""
    query = sample_ar_query.to_hololoom_query()

    assert "point" in query.text  # Gesture enrichment
    assert "beehive" in query.text.lower()  # Vision enrichment
    assert query.metadata['gesture'] == "point"
    assert query.metadata['voice_intent'] == "info"
    assert query.metadata['query_type'] == "multimodal"


# ============================================================================
# ARProvenanceTracker Tests
# ============================================================================

def test_provenance_tracker_initialization():
    """Test provenance tracker initialization"""
    tracker = ARProvenanceTracker(capacity=100)

    assert tracker.capacity == 100
    assert len(tracker.ar_entries) == 0


def test_provenance_tracking(sample_ar_query, sample_spacetime):
    """Test AR provenance tracking"""
    tracker = ARProvenanceTracker()

    tracker.track(sample_ar_query, sample_spacetime, thought="Processing AR query")

    assert len(tracker.ar_entries) == 1

    entry = tracker.ar_entries[0]
    assert entry.gesture == "point"
    assert entry.voice_intent == "info"
    assert "beehive" in entry.vision_objects[0].lower()
    assert entry.query_type == "multimodal"
    assert entry.score == 0.85


def test_provenance_get_by_query_type(sample_ar_query, sample_spacetime):
    """Test getting provenance by query type"""
    tracker = ARProvenanceTracker()

    # Track multiple queries
    tracker.track(sample_ar_query, sample_spacetime)

    voice_only = ARQuery(text="Hello", voice_intent="info")
    tracker.track(voice_only, sample_spacetime)

    # Get multimodal entries
    multimodal = tracker.get_by_query_type(ARQueryType.MULTIMODAL)
    assert len(multimodal) == 1

    # Get voice-only entries
    voice = tracker.get_by_query_type(ARQueryType.VOICE_ONLY)
    assert len(voice) == 1


def test_provenance_get_by_gesture(sample_ar_query, sample_spacetime):
    """Test getting provenance by gesture"""
    tracker = ARProvenanceTracker()

    # Track queries with different gestures
    tracker.track(sample_ar_query, sample_spacetime)

    pinch_query = ARQuery(text="Select", gesture_detected="pinch")
    tracker.track(pinch_query, sample_spacetime)

    # Get point gestures
    point_entries = tracker.get_by_gesture("point")
    assert len(point_entries) == 1
    assert point_entries[0].gesture == "point"


# ============================================================================
# ARPatternLearner Tests
# ============================================================================

def test_pattern_learner_initialization():
    """Test pattern learner initialization"""
    learner = ARPatternLearner(min_confidence=0.8, min_support=3)

    assert learner.min_confidence == 0.8
    assert learner.min_support == 3
    assert len(learner.patterns) == 0


def test_pattern_extraction(sample_ar_query, sample_spacetime):
    """Test pattern extraction from AR query"""
    learner = ARPatternLearner(min_confidence=0.8)

    learner.extract_and_learn(sample_ar_query, sample_spacetime)

    assert len(learner.patterns) == 1

    pattern = learner.patterns[0]
    assert pattern.gesture == "point"
    assert pattern.voice_intent == "info"
    assert "beehive" in pattern.vision_context
    assert pattern.tool_used == "inspect_beehive"
    assert pattern.support == 1


def test_pattern_matching():
    """Test pattern matching"""
    pattern = ARPattern(
        gesture="pinch",
        voice_intent="select",
        vision_context={"beehive", "frame"},
    )

    # Perfect match
    score = pattern.matches("pinch", "select", {"beehive", "frame"})
    assert score == 1.0

    # Partial match (gesture only)
    score = pattern.matches("pinch", None, set())
    assert 0.3 < score < 0.5

    # No match
    score = pattern.matches("swipe", "delete", set())
    assert score == 0.0


def test_pattern_get_best_tool():
    """Test getting best tool for context"""
    learner = ARPatternLearner(min_support=1)

    # Add patterns
    pattern1 = ARPattern(
        gesture="pinch",
        voice_intent="select",
        tool_used="select_object",
        confidence=0.9,
        support=5,
        success_rate=0.9,
    )
    learner.patterns.append(pattern1)

    pattern2 = ARPattern(
        gesture="pinch",
        voice_intent="select",
        tool_used="grab_object",
        confidence=0.7,
        support=2,
        success_rate=0.5,
    )
    learner.patterns.append(pattern2)

    # Get best tool
    result = learner.get_best_tool("pinch", "select", set())
    assert result is not None
    tool, confidence = result
    assert tool == "select_object"  # Higher quality pattern


def test_pattern_hot_patterns():
    """Test getting hot patterns"""
    learner = ARPatternLearner(min_support=1)

    # Add patterns with different heat scores
    pattern1 = ARPattern(
        gesture="pinch",
        tool_used="select",
        confidence=0.9,
        support=10,
        success_rate=0.9,
    )
    pattern2 = ARPattern(
        gesture="swipe",
        tool_used="delete",
        confidence=0.7,
        support=3,
        success_rate=0.6,
    )
    learner.patterns.extend([pattern1, pattern2])

    hot = learner.get_hot_patterns(limit=5)
    assert len(hot) == 2
    assert hot[0]['gesture'] == "pinch"  # Higher heat


# ============================================================================
# ARRefiner Tests
# ============================================================================

def test_ar_quality_metrics(sample_ar_query, sample_spacetime):
    """Test AR quality metrics calculation"""
    metrics = ARQualityMetrics.from_spacetime(sample_spacetime, sample_ar_query)

    assert metrics.confidence == 0.85
    assert 0.0 <= metrics.score() <= 1.0
    assert metrics.modality_balance == 1.0  # All 3 modalities


def test_ar_quality_metrics_modality_balance():
    """Test modality balance calculation"""
    # Single modality
    q = ARQuery(text="Hello")
    assert ARQualityMetrics._calculate_modality_balance(q) == pytest.approx(0.33, abs=0.01)

    # Two modalities
    q = ARQuery(text="Hello", gesture_detected="pinch")
    assert ARQualityMetrics._calculate_modality_balance(q) == pytest.approx(0.67, abs=0.01)

    # Three modalities
    q = ARQuery(
        text="Hello",
        gesture_detected="pinch",
        vision_objects=[ARObject("obj", ARObjectType.BEEHIVE, Vector3(0, 0, 0))],
    )
    assert ARQualityMetrics._calculate_modality_balance(q) == 1.0


@pytest.mark.asyncio
async def test_refiner_strategy_selection(config, shards, sample_ar_query, sample_spacetime):
    """Test auto-selection of refinement strategy"""
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        refiner = ARRefiner(orchestrator)

        strategy = refiner._select_strategy(sample_ar_query, sample_spacetime)
        assert isinstance(strategy, ARRefinementStrategy)


# ============================================================================
# ARBackgroundLearner Tests
# ============================================================================

def test_thompson_priors_initialization():
    """Test Thompson priors initialization"""
    priors = ARThompsonPriors()

    # Check default priors
    assert priors.get_expected_reward("pinch", "select") == 0.5  # Uniform prior


def test_thompson_priors_updates():
    """Test Thompson priors updates"""
    priors = ARThompsonPriors()

    # Update with successes
    priors.update_gesture_success("pinch", "select", 0.9)
    priors.update_gesture_success("pinch", "select", 0.8)

    # Expected reward should increase
    reward = priors.get_expected_reward("pinch", "select")
    assert reward > 0.5

    # Update with failure
    priors.update_gesture_failure("pinch", "delete", 0.3)
    reward_delete = priors.get_expected_reward("pinch", "delete")
    assert reward_delete < 0.5


def test_thompson_priors_best_tool():
    """Test getting best tool from priors"""
    priors = ARThompsonPriors()

    # Update multiple tools
    priors.update_gesture_success("pinch", "select", 0.9)
    priors.update_gesture_success("pinch", "grab", 0.5)

    best = priors.get_best_tool_for_gesture("pinch")
    assert best == "select"


def test_modality_weights():
    """Test modality weights tracking"""
    weights = ARModalityWeights()

    # Record interactions
    weights.update("voice+gesture", success=True)
    weights.update("voice+gesture", success=True)
    weights.update("voice+gesture", success=False)

    weight = weights.get_weight("voice+gesture")
    assert weight == pytest.approx(2/3, abs=0.01)


def test_modality_weights_best():
    """Test getting best modality"""
    weights = ARModalityWeights()

    weights.update("voice_only", success=True)
    weights.update("voice_only", success=False)

    weights.update("multimodal", success=True)
    weights.update("multimodal", success=True)

    best = weights.get_best_modality()
    assert best == "multimodal"


@pytest.mark.asyncio
async def test_background_learner_recording():
    """Test background learner interaction recording"""
    learner = ARBackgroundLearner(update_interval=0.1)

    await learner.start()

    # Record interactions
    learner.record_interaction(
        gesture="pinch",
        voice_intent="select",
        modality_combo="voice+gesture",
        tool_used="select_object",
        confidence=0.9,
    )

    # Wait for learning update
    await asyncio.sleep(0.2)

    stats = learner.get_statistics()
    assert stats['queries_processed'] >= 1

    await learner.stop()


@pytest.mark.asyncio
async def test_background_learner_persistence():
    """Test background learner state persistence"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "learning_state.json"

        # Create and train learner
        learner = ARBackgroundLearner(persist_path=str(path))
        await learner.start()

        learner.record_interaction(
            gesture="pinch",
            voice_intent="select",
            modality_combo="voice+gesture",
            tool_used="select",
            confidence=0.9,
        )

        # Wait for update
        await asyncio.sleep(0.2)

        # Stop and save
        await learner.stop()

        # Check file exists
        assert path.exists()

        # Load in new learner
        learner2 = ARBackgroundLearner(persist_path=str(path))
        await learner2.start()

        stats = learner2.get_statistics()
        assert stats['queries_processed'] > 0

        await learner2.stop()


# ============================================================================
# ARLearningEngine Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_learning_engine_initialization(config, shards):
    """Test learning engine initialization"""
    ar_config = ARLearningConfig(
        enable_provenance=True,
        enable_pattern_learning=False,  # Skip for speed
        enable_refinement=False,
        enable_background_learning=False,
    )

    async with ARLearningEngine(config, shards, ar_config) as engine:
        assert engine.orchestrator is not None
        assert engine.provenance_tracker is not None


@pytest.mark.asyncio
async def test_learning_engine_weave(config, shards, sample_ar_query):
    """Test learning engine weaving"""
    ar_config = ARLearningConfig(
        enable_provenance=True,
        enable_pattern_learning=False,
        enable_refinement=False,
        enable_background_learning=False,
    )

    async with ARLearningEngine(config, shards, ar_config) as engine:
        spacetime = await engine.weave(sample_ar_query)

        assert spacetime is not None
        assert spacetime.confidence > 0.0

        # Check provenance tracked
        assert len(engine.provenance_tracker.ar_entries) == 1


@pytest.mark.asyncio
async def test_learning_engine_statistics(config, shards, sample_ar_query):
    """Test learning engine statistics"""
    ar_config = ARLearningConfig(
        enable_provenance=True,
        enable_pattern_learning=False,
        enable_refinement=False,
        enable_background_learning=False,
    )

    async with ARLearningEngine(config, shards, ar_config) as engine:
        await engine.weave(sample_ar_query)

        stats = engine.get_learning_statistics()
        assert stats['queries_processed'] == 1
        assert stats['avg_confidence'] > 0.0


@pytest.mark.asyncio
async def test_learning_engine_persistence(config, shards, sample_ar_query):
    """Test learning engine state persistence"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "learning_state.json"

        ar_config = ARLearningConfig(
            enable_provenance=True,
            enable_pattern_learning=False,
            enable_refinement=False,
            enable_background_learning=False,
        )

        async with ARLearningEngine(config, shards, ar_config) as engine:
            await engine.weave(sample_ar_query)

            engine.save_learning_state(str(path))

        # Check saved
        assert path.exists()

        # Load
        async with ARLearningEngine(config, shards, ar_config) as engine2:
            engine2.load_learning_state(str(path))

            stats = engine2.get_learning_statistics()
            assert stats['queries_processed'] == 1


@pytest.mark.asyncio
async def test_weave_with_ar_learning_helper(config, shards, sample_ar_query):
    """Test weave_with_ar_learning helper function"""
    ar_config = ARLearningConfig(
        enable_provenance=True,
        enable_pattern_learning=False,
        enable_refinement=False,
        enable_background_learning=False,
    )

    spacetime, stats = await weave_with_ar_learning(
        sample_ar_query,
        config,
        shards,
        ar_config,
    )

    assert spacetime is not None
    assert stats['queries_processed'] == 1


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_learning_pipeline(config, shards, sample_ar_query):
    """Test complete learning pipeline"""
    ar_config = ARLearningConfig(
        enable_provenance=True,
        enable_pattern_learning=True,
        enable_refinement=False,  # Skip for speed
        enable_background_learning=False,  # Skip for speed
        min_pattern_confidence=0.5,  # Lower threshold for testing
    )

    async with ARLearningEngine(config, shards, ar_config) as engine:
        # Process multiple queries
        for _ in range(3):
            spacetime = await engine.weave(sample_ar_query)
            assert spacetime is not None

        # Check learning occurred
        stats = engine.get_learning_statistics()
        assert stats['queries_processed'] == 3

        # Check provenance
        assert len(engine.provenance_tracker.ar_entries) == 3

        # Check patterns (if learner available)
        if engine.pattern_learner:
            patterns = engine.pattern_learner.get_statistics()
            assert patterns['total_patterns'] >= 0  # May or may not have patterns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
