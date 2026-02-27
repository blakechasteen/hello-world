"""
Semantic Calculus 16 Axes Tests (Week 2 - Priority 3)
======================================================
Tests the 16 interpretable semantic axes that form the human-readable projection
of HoloLoom's 228D semantic space.

The 16 axes are divided into 4 categories:
- Affective (4 axes): Warmth, Valence, Arousal, Intensity
- Social/Interpersonal (4 axes): Formality, Directness, Power, Generosity
- Cognitive (4 axes): Certainty, Complexity, Concreteness, Familiarity
- Temporal/Dynamic (4 axes): Agency, Stability, Urgency, Completion

Test Coverage (35 tests):
- Axis initialization - 2 tests
- Axis learning (from exemplars) - 4 tests
- Projection (vector → scalar) - 3 tests
- All 16 axes semantic correctness - 16 tests
- Edge cases - 5 tests
- Integration - 5 tests

Created: 2025-11-21 (Week 2 Test Creation)
"""

import pytest
import numpy as np
from unittest.mock import Mock

from hololoom.semantic_calculus.dimensions import (
    SemanticDimension, STANDARD_DIMENSIONS
)


# ============================================================================
# Test 1-2: Axis Initialization
# ============================================================================

def test_semantic_dimension_initialization():
    """Test SemanticDimension initializes correctly."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["positive", "good"],
        negative_exemplars=["negative", "bad"]
    )

    assert dim.name == "Test"
    assert len(dim.positive_exemplars) == 2
    assert len(dim.negative_exemplars) == 2
    assert dim.axis is None  # Not learned yet


def test_standard_dimensions_count():
    """Test STANDARD_DIMENSIONS contains exactly 16 axes."""
    assert len(STANDARD_DIMENSIONS) == 16

    # Verify all axes have names and exemplars
    for dim in STANDARD_DIMENSIONS:
        assert isinstance(dim.name, str)
        assert len(dim.positive_exemplars) > 0
        assert len(dim.negative_exemplars) > 0


# ============================================================================
# Test 3-6: Axis Learning
# ============================================================================

def test_learn_axis_basic():
    """Test learn_axis() computes normalized direction vector."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["good"],
        negative_exemplars=["bad"]
    )

    # Mock embedder: "good" → [1, 0], "bad" → [0, 1]
    def mock_embed(word):
        if word == "good":
            return np.array([1.0, 0.0])
        elif word == "bad":
            return np.array([0.0, 1.0])

    axis = dim.learn_axis(mock_embed, use_batch=False)

    # Axis should be normalized difference: [1, 0] - [0, 1] = [1, -1] → normalized
    expected = np.array([1.0, -1.0]) / np.linalg.norm([1.0, -1.0])
    assert np.allclose(axis, expected)


def test_learn_axis_multiple_exemplars():
    """Test learn_axis() with multiple exemplars (centroid calculation)."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["a", "b"],
        negative_exemplars=["x", "y"]
    )

    # Mock embedder
    def mock_embed(word):
        if word == "a":
            return np.array([1.0, 0.0])
        elif word == "b":
            return np.array([1.0, 1.0])
        elif word == "x":
            return np.array([0.0, 0.0])
        elif word == "y":
            return np.array([0.0, 1.0])

    axis = dim.learn_axis(mock_embed, use_batch=False)

    # Positive centroid: ([1,0] + [1,1]) / 2 = [1, 0.5]
    # Negative centroid: ([0,0] + [0,1]) / 2 = [0, 0.5]
    # Axis: [1, 0.5] - [0, 0.5] = [1, 0] → normalized = [1, 0]

    expected = np.array([1.0, 0.0])
    assert np.allclose(axis, expected, atol=0.01)


def test_learn_axis_batch_mode():
    """Test learn_axis() with batch embedding."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["good", "great"],
        negative_exemplars=["bad", "awful"]
    )

    # Mock batch embedder
    def mock_batch_embed(words):
        result = []
        for word in words:
            if word in ["good", "great"]:
                result.append([1.0, 0.0])
            else:
                result.append([0.0, 1.0])
        return np.array(result)

    axis = dim.learn_axis(mock_batch_embed, use_batch=True)

    # Should compute axis correctly via batch processing
    assert axis is not None
    assert np.linalg.norm(axis) - 1.0 < 0.01  # Normalized


def test_learn_axis_fallback():
    """Test learn_axis() falls back to single embeddings if batch fails."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["good"],
        negative_exemplars=["bad"]
    )

    call_count = [0]

    # Mock embedder that raises TypeError on batch call
    def mock_embed(word_or_list):
        if isinstance(word_or_list, list):
            raise TypeError("Batch not supported")

        call_count[0] += 1
        if word_or_list == "good":
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])

    # Should fall back to single embeddings
    axis = dim.learn_axis(mock_embed, use_batch=True)

    assert axis is not None
    assert call_count[0] == 2  # Called twice (good + bad)


# ============================================================================
# Test 7-9: Projection
# ============================================================================

def test_project_basic():
    """Test project() computes dot product correctly."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["positive"],
        negative_exemplars=["negative"]
    )

    # Manually set axis
    dim.axis = np.array([1.0, 0.0])

    # Project vector [0.5, 0.5]
    vector = np.array([0.5, 0.5])
    projection = dim.project(vector)

    # Projection = dot([0.5, 0.5], [1, 0]) = 0.5
    assert abs(projection - 0.5) < 0.01


def test_project_positive_negative():
    """Test project() returns positive/negative values correctly."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["pos"],
        negative_exemplars=["neg"]
    )

    dim.axis = np.array([1.0, 0.0])

    # Positive direction
    pos_vec = np.array([1.0, 0.0])
    assert dim.project(pos_vec) > 0

    # Negative direction
    neg_vec = np.array([-1.0, 0.0])
    assert dim.project(neg_vec) < 0

    # Orthogonal (zero projection)
    orth_vec = np.array([0.0, 1.0])
    assert abs(dim.project(orth_vec)) < 0.01


def test_project_not_learned():
    """Test project() raises error if axis not learned yet."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["pos"],
        negative_exemplars=["neg"]
    )

    # Axis not learned (None)
    vector = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="axis not learned"):
        dim.project(vector)


# ============================================================================
# Test 10-25: All 16 Axes Semantic Correctness
# ============================================================================

def test_warmth_axis():
    """Test Warmth axis (warm/cold)."""
    warmth = [d for d in STANDARD_DIMENSIONS if d.name == "Warmth"][0]

    assert "warm" in warmth.positive_exemplars
    assert "cold" in warmth.negative_exemplars


def test_valence_axis():
    """Test Valence axis (positive/negative)."""
    valence = [d for d in STANDARD_DIMENSIONS if d.name == "Valence"][0]

    assert "positive" in valence.positive_exemplars
    assert "negative" in valence.negative_exemplars


def test_arousal_axis():
    """Test Arousal axis (excited/calm)."""
    arousal = [d for d in STANDARD_DIMENSIONS if d.name == "Arousal"][0]

    assert "excited" in arousal.positive_exemplars
    assert "calm" in arousal.negative_exemplars


def test_intensity_axis():
    """Test Intensity axis (intense/mild)."""
    intensity = [d for d in STANDARD_DIMENSIONS if d.name == "Intensity"][0]

    assert "intense" in intensity.positive_exemplars
    assert "mild" in intensity.negative_exemplars


def test_formality_axis():
    """Test Formality axis (formal/casual)."""
    formality = [d for d in STANDARD_DIMENSIONS if d.name == "Formality"][0]

    assert "formal" in formality.positive_exemplars
    assert "casual" in formality.negative_exemplars


def test_directness_axis():
    """Test Directness axis (direct/indirect)."""
    directness = [d for d in STANDARD_DIMENSIONS if d.name == "Directness"][0]

    assert "direct" in directness.positive_exemplars
    assert "indirect" in directness.negative_exemplars


def test_power_axis():
    """Test Power axis (dominant/submissive)."""
    power = [d for d in STANDARD_DIMENSIONS if d.name == "Power"][0]

    assert "dominant" in power.positive_exemplars
    assert "submissive" in power.negative_exemplars


def test_generosity_axis():
    """Test Generosity axis (generous/selfish)."""
    generosity = [d for d in STANDARD_DIMENSIONS if d.name == "Generosity"][0]

    assert "generous" in generosity.positive_exemplars
    assert "selfish" in generosity.negative_exemplars


def test_certainty_axis():
    """Test Certainty axis (certain/uncertain)."""
    certainty = [d for d in STANDARD_DIMENSIONS if d.name == "Certainty"][0]

    assert "certain" in certainty.positive_exemplars
    assert "uncertain" in certainty.negative_exemplars


def test_complexity_axis():
    """Test Complexity axis (complex/simple)."""
    complexity = [d for d in STANDARD_DIMENSIONS if d.name == "Complexity"][0]

    assert "complex" in complexity.positive_exemplars
    assert "simple" in complexity.negative_exemplars


def test_concreteness_axis():
    """Test Concreteness axis (concrete/abstract)."""
    concreteness = [d for d in STANDARD_DIMENSIONS if d.name == "Concreteness"][0]

    assert "concrete" in concreteness.positive_exemplars
    assert "abstract" in concreteness.negative_exemplars


def test_familiarity_axis():
    """Test Familiarity axis (familiar/novel)."""
    familiarity = [d for d in STANDARD_DIMENSIONS if d.name == "Familiarity"][0]

    assert "familiar" in familiarity.positive_exemplars
    assert "novel" in familiarity.negative_exemplars


def test_agency_axis():
    """Test Agency axis (active/passive)."""
    agency = [d for d in STANDARD_DIMENSIONS if d.name == "Agency"][0]

    assert "active" in agency.positive_exemplars
    assert "passive" in agency.negative_exemplars


def test_stability_axis():
    """Test Stability axis (stable/volatile)."""
    stability = [d for d in STANDARD_DIMENSIONS if d.name == "Stability"][0]

    assert "stable" in stability.positive_exemplars
    assert "volatile" in stability.negative_exemplars


def test_urgency_axis():
    """Test Urgency axis (urgent/patient)."""
    urgency = [d for d in STANDARD_DIMENSIONS if d.name == "Urgency"][0]

    assert "urgent" in urgency.positive_exemplars
    assert "patient" in urgency.negative_exemplars


def test_completion_axis():
    """Test Completion axis (complete/incomplete)."""
    completion = [d for d in STANDARD_DIMENSIONS if d.name == "Completion"][0]

    assert "complete" in completion.positive_exemplars
    assert "incomplete" in completion.negative_exemplars


# ============================================================================
# Test 26-30: Edge Cases
# ============================================================================

def test_learn_axis_near_zero_norm():
    """Test learn_axis() handles near-zero axis norm."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["same"],
        negative_exemplars=["same"]  # Identical exemplars
    )

    # Mock embedder
    def mock_embed(word):
        return np.array([1.0, 0.0])  # Same embedding for both

    # Should handle near-zero difference vector
    axis = dim.learn_axis(mock_embed, use_batch=False)

    # Axis norm should be ~1 (due to 1e-10 epsilon)
    assert np.linalg.norm(axis) > 0.99


def test_project_zero_vector():
    """Test project() with zero vector."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["pos"],
        negative_exemplars=["neg"]
    )

    dim.axis = np.array([1.0, 0.0])

    # Zero vector
    zero = np.zeros(2)
    projection = dim.project(zero)

    # Should be zero
    assert abs(projection) < 0.01


def test_project_unnormalized_axis():
    """Test project() works with unnormalized axis."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["pos"],
        negative_exemplars=["neg"]
    )

    # Unnormalized axis
    dim.axis = np.array([2.0, 0.0])  # Not unit norm

    vector = np.array([1.0, 0.0])
    projection = dim.project(vector)

    # Projection = dot([1, 0], [2, 0]) = 2.0
    assert abs(projection - 2.0) < 0.01


def test_all_axes_have_sufficient_exemplars():
    """Test all 16 axes have at least 3 exemplars each."""
    for dim in STANDARD_DIMENSIONS:
        assert len(dim.positive_exemplars) >= 3, f"{dim.name} needs ≥3 positive exemplars"
        assert len(dim.negative_exemplars) >= 3, f"{dim.name} needs ≥3 negative exemplars"


def test_axes_cover_four_categories():
    """Test 16 axes cover all 4 categories (Affective, Social, Cognitive, Temporal)."""
    names = [d.name for d in STANDARD_DIMENSIONS]

    # Affective (4)
    affective = ["Warmth", "Valence", "Arousal", "Intensity"]
    assert all(name in names for name in affective)

    # Social/Interpersonal (4)
    social = ["Formality", "Directness", "Power", "Generosity"]
    assert all(name in names for name in social)

    # Cognitive (4)
    cognitive = ["Certainty", "Complexity", "Concreteness", "Familiarity"]
    assert all(name in names for name in cognitive)

    # Temporal/Dynamic (4)
    temporal = ["Agency", "Stability", "Urgency", "Completion"]
    assert all(name in names for name in temporal)


# ============================================================================
# Test 31-35: Integration
# ============================================================================

def test_learn_all_axes():
    """Test learning all 16 axes together."""
    # Mock embedder
    def mock_embed(word):
        # Simple hash-based embedding for testing
        np.random.seed(hash(word) % 2**32)
        return np.random.randn(384)  # 384D as in HoloLoom

    # Learn all axes
    for dim in STANDARD_DIMENSIONS:
        axis = dim.learn_axis(mock_embed, use_batch=False)

        # Verify axis is normalized
        assert abs(np.linalg.norm(axis) - 1.0) < 0.01


def test_project_query_onto_all_axes():
    """Test projecting a query vector onto all 16 axes."""
    # Mock embedder
    def mock_embed(word):
        np.random.seed(hash(word) % 2**32)
        return np.random.randn(10)

    # Learn axes (smaller dim for speed)
    for dim in STANDARD_DIMENSIONS:
        dim.learn_axis(mock_embed, use_batch=False)

    # Project a test vector
    query_vector = np.random.randn(10)

    projections = {}
    for dim in STANDARD_DIMENSIONS:
        projections[dim.name] = dim.project(query_vector)

    # Should have 16 projections
    assert len(projections) == 16

    # All projections should be scalars
    assert all(isinstance(v, (int, float, np.floating)) for v in projections.values())


def test_manifold_projection():
    """Test Riemannian manifold projection (if available)."""
    dim = SemanticDimension(
        name="Test",
        positive_exemplars=["pos"],
        negative_exemplars=["neg"]
    )

    dim.axis = np.array([1.0, 0.0])

    # Mock manifold with log_map
    class MockManifold:
        def log_map(self, origin, point):
            # Simple identity for testing
            return point - origin

    manifold = MockManifold()
    vector = np.array([0.5, 0.0])

    # Should use manifold projection
    projection = dim.project(vector, manifold=manifold)

    # Should match Euclidean in this simple case
    euclidean = dim.project(vector, manifold=None)
    assert abs(projection - euclidean) < 0.01


def test_axes_orthogonality():
    """Test that axes are reasonably independent (not all parallel)."""
    # Mock embedder
    def mock_embed(word):
        np.random.seed(hash(word) % 2**32)
        return np.random.randn(100)

    # Learn axes
    axes = []
    for dim in STANDARD_DIMENSIONS:
        axis = dim.learn_axis(mock_embed, use_batch=False)
        axes.append(axis)

    # Compute pairwise dot products
    axes_matrix = np.array(axes)
    gram = axes_matrix @ axes_matrix.T

    # Diagonal should be ~1 (normalized)
    assert np.allclose(np.diag(gram), 1.0, atol=0.01)

    # Off-diagonal should be <0.9 on average (not too parallel)
    off_diag = gram[~np.eye(16, dtype=bool)]
    assert np.mean(np.abs(off_diag)) < 0.9


def test_semantic_projection_sanity():
    """Test semantic projections make intuitive sense."""
    # Mock embedder that respects semantic similarity
    def mock_embed(word):
        # Warmth dimension
        if word in ["warm", "loving", "kind"]:
            return np.array([1.0, 0.0])
        elif word in ["cold", "harsh"]:
            return np.array([-1.0, 0.0])
        else:
            return np.array([0.0, 0.0])

    warmth = [d for d in STANDARD_DIMENSIONS if d.name == "Warmth"][0]
    warmth.learn_axis(mock_embed, use_batch=False)

    # "warm" should project positively
    warm_vec = mock_embed("warm")
    assert warmth.project(warm_vec) > 0

    # "cold" should project negatively
    cold_vec = mock_embed("cold")
    assert warmth.project(cold_vec) < 0
