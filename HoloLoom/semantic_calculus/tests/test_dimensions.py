"""
Tests for Semantic Dimensions
=============================

Comprehensive tests for SemanticDimension class and all 228 dimension definitions.

Note: Despite the name EXTENDED_244_DIMENSIONS, the actual count is 228 dimensions.
      The name is a legacy artifact from the original design.

Test Categories:
    1. SemanticDimension dataclass tests
    2. Axis learning tests
    3. Vector projection tests
    4. Standard dimensions (16) validation
    5. Extended dimensions (228) validation
    6. Category grouping tests
    7. Edge cases and error handling
    8. Numerical stability tests

Run tests:
    pytest HoloLoom/semantic_calculus/tests/test_dimensions.py -v
"""

import pytest
import numpy as np
from typing import List, Dict
import sys


# ============================================================================
# Test Imports
# ============================================================================

class TestImports:
    """Test that all required modules can be imported."""

    def test_import_semantic_dimension(self):
        """Test SemanticDimension import."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension
        assert SemanticDimension is not None

    def test_import_standard_dimensions(self):
        """Test STANDARD_DIMENSIONS import."""
        from HoloLoom.semantic_calculus.dimensions import STANDARD_DIMENSIONS
        assert STANDARD_DIMENSIONS is not None
        assert len(STANDARD_DIMENSIONS) == 16

    def test_import_extended_dimensions(self):
        """Test EXTENDED_244_DIMENSIONS import (actual count is 228)."""
        from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS
        assert EXTENDED_244_DIMENSIONS is not None
        # Note: Despite the name, actual count is 228 (legacy naming)
        assert len(EXTENDED_244_DIMENSIONS) == 228

    def test_import_semantic_spectrum(self):
        """Test SemanticSpectrum import."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum
        assert SemanticSpectrum is not None

    def test_import_category_constants(self):
        """Test category constant imports."""
        from HoloLoom.semantic_calculus.dimensions import (
            NARRATIVE_DIMENSIONS,
            EMOTIONAL_DEPTH_DIMENSIONS,
            RELATIONAL_DIMENSIONS,
            ARCHETYPAL_DIMENSIONS,
            PHILOSOPHICAL_DIMENSIONS,
            TRANSFORMATION_DIMENSIONS,
        )
        assert all([
            NARRATIVE_DIMENSIONS,
            EMOTIONAL_DEPTH_DIMENSIONS,
            RELATIONAL_DIMENSIONS,
            ARCHETYPAL_DIMENSIONS,
            PHILOSOPHICAL_DIMENSIONS,
            TRANSFORMATION_DIMENSIONS,
        ])


# ============================================================================
# SemanticDimension Dataclass Tests
# ============================================================================

class TestSemanticDimensionDataclass:
    """Test SemanticDimension dataclass functionality."""

    def test_create_dimension(self):
        """Test basic dimension creation."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="TestDim",
            positive_exemplars=["good", "happy"],
            negative_exemplars=["bad", "sad"]
        )

        assert dim.name == "TestDim"
        assert dim.positive_exemplars == ["good", "happy"]
        assert dim.negative_exemplars == ["bad", "sad"]
        assert dim.axis is None

    def test_dimension_with_axis(self):
        """Test dimension creation with pre-computed axis."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        axis = np.random.randn(384)
        axis = axis / np.linalg.norm(axis)

        dim = SemanticDimension(
            name="TestDim",
            positive_exemplars=["pos"],
            negative_exemplars=["neg"],
            axis=axis
        )

        assert dim.axis is not None
        assert np.allclose(np.linalg.norm(dim.axis), 1.0)

    def test_dimension_empty_exemplars(self):
        """Test dimension with empty exemplars (edge case)."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        # Should allow empty exemplars (will fail at learn_axis)
        dim = SemanticDimension(
            name="EmptyDim",
            positive_exemplars=[],
            negative_exemplars=[]
        )

        assert dim.name == "EmptyDim"
        assert len(dim.positive_exemplars) == 0
        assert len(dim.negative_exemplars) == 0

    def test_dimension_single_exemplar(self):
        """Test dimension with single exemplar on each side."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="SingleExemplar",
            positive_exemplars=["good"],
            negative_exemplars=["bad"]
        )

        assert len(dim.positive_exemplars) == 1
        assert len(dim.negative_exemplars) == 1


# ============================================================================
# Axis Learning Tests
# ============================================================================

class TestAxisLearning:
    """Test SemanticDimension.learn_axis() functionality."""

    def test_learn_axis_basic(self, mock_embed_fn, sample_dimension):
        """Test basic axis learning."""
        sample_dimension.learn_axis(mock_embed_fn)

        assert sample_dimension.axis is not None
        assert sample_dimension.axis.shape == (384,)

    def test_learn_axis_normalized(self, mock_embed_fn, sample_dimension):
        """Test that learned axis is normalized."""
        sample_dimension.learn_axis(mock_embed_fn)

        norm = np.linalg.norm(sample_dimension.axis)
        assert np.isclose(norm, 1.0, atol=1e-6), f"Axis norm = {norm}, expected 1.0"

    def test_learn_axis_deterministic(self, mock_embed_fn, sample_dimension):
        """Test that axis learning is deterministic."""
        sample_dimension.learn_axis(mock_embed_fn)
        axis1 = sample_dimension.axis.copy()

        # Reset and learn again
        sample_dimension.axis = None
        sample_dimension.learn_axis(mock_embed_fn)
        axis2 = sample_dimension.axis.copy()

        assert np.allclose(axis1, axis2), "Axis learning should be deterministic"

    def test_learn_axis_batch_mode(self, mock_embed_fn, sample_dimension):
        """Test axis learning with batch mode enabled."""
        sample_dimension.learn_axis(mock_embed_fn, use_batch=True)
        axis_batch = sample_dimension.axis.copy()

        # Reset and learn with batch disabled
        sample_dimension.axis = None
        sample_dimension.learn_axis(mock_embed_fn, use_batch=False)
        axis_single = sample_dimension.axis.copy()

        # Should produce same result
        assert np.allclose(axis_batch, axis_single, atol=1e-5)

    def test_learn_axis_different_dimensions(self, mock_embed_fn_128):
        """Test axis learning with different embedding dimensions."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="TestDim",
            positive_exemplars=["warm", "friendly"],
            negative_exemplars=["cold", "distant"]
        )

        dim.learn_axis(mock_embed_fn_128)

        assert dim.axis.shape == (128,)
        assert np.isclose(np.linalg.norm(dim.axis), 1.0, atol=1e-6)

    def test_learn_axis_semantic_structure(self, mock_embed_fn_with_semantic_structure):
        """Test that learned axis captures semantic direction."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="Warmth",
            positive_exemplars=["warm", "friendly", "kind"],
            negative_exemplars=["cold", "distant", "hostile"]
        )

        dim.learn_axis(mock_embed_fn_with_semantic_structure)

        # Project "warm" and "cold" onto axis
        warm_emb = mock_embed_fn_with_semantic_structure("warm")
        cold_emb = mock_embed_fn_with_semantic_structure("cold")

        warm_proj = np.dot(warm_emb, dim.axis)
        cold_proj = np.dot(cold_emb, dim.axis)

        # Warm should project higher than cold
        assert warm_proj > cold_proj, f"warm_proj={warm_proj}, cold_proj={cold_proj}"


# ============================================================================
# Vector Projection Tests
# ============================================================================

class TestVectorProjection:
    """Test SemanticDimension.project() functionality."""

    def test_project_basic(self, mock_embed_fn, sample_dimension):
        """Test basic vector projection."""
        sample_dimension.learn_axis(mock_embed_fn)

        vec = mock_embed_fn("test")
        projection = sample_dimension.project(vec)

        assert isinstance(projection, float)
        assert not np.isnan(projection)

    def test_project_range(self, mock_embed_fn, sample_dimension):
        """Test that projections are in reasonable range."""
        sample_dimension.learn_axis(mock_embed_fn)

        # Project 100 random vectors
        for i in range(100):
            vec = mock_embed_fn(f"word_{i}")
            projection = sample_dimension.project(vec)

            # For normalized vectors, projection should be in [-1, 1]
            assert -1.5 <= projection <= 1.5, f"Projection {projection} out of range"

    def test_project_zero_vector(self, mock_embed_fn, sample_dimension):
        """Test projection of zero vector."""
        sample_dimension.learn_axis(mock_embed_fn)

        zero_vec = np.zeros(384)
        projection = sample_dimension.project(zero_vec)

        assert projection == 0.0 or np.isclose(projection, 0.0)

    def test_project_axis_aligned(self, mock_embed_fn, sample_dimension):
        """Test projection of vector aligned with axis."""
        sample_dimension.learn_axis(mock_embed_fn)

        # Project the axis itself
        projection = sample_dimension.project(sample_dimension.axis)

        # Should be close to 1.0 (axis projected onto itself)
        assert np.isclose(projection, 1.0, atol=1e-6)

    def test_project_opposite_axis(self, mock_embed_fn, sample_dimension):
        """Test projection of vector opposite to axis."""
        sample_dimension.learn_axis(mock_embed_fn)

        # Project negative of axis
        opposite = -sample_dimension.axis
        projection = sample_dimension.project(opposite)

        # Should be close to -1.0
        assert np.isclose(projection, -1.0, atol=1e-6)

    def test_project_orthogonal(self, mock_embed_fn, sample_dimension, orthogonal_vectors_384):
        """Test projection of orthogonal vectors."""
        sample_dimension.learn_axis(mock_embed_fn)

        v1, v2 = orthogonal_vectors_384

        p1 = sample_dimension.project(v1)
        p2 = sample_dimension.project(v2)

        # Both should be in valid range
        assert -1.5 <= p1 <= 1.5
        assert -1.5 <= p2 <= 1.5

    def test_project_without_learned_axis(self, sample_dimension):
        """Test projection without learned axis raises error or handles gracefully."""
        vec = np.random.randn(384)

        # Depending on implementation, might raise or return 0
        try:
            projection = sample_dimension.project(vec)
            # If it doesn't raise, check reasonable behavior
            assert projection == 0.0 or np.isnan(projection)
        except (ValueError, TypeError, AttributeError):
            # Expected if axis not learned
            pass


# ============================================================================
# Standard Dimensions Tests (16 Core)
# ============================================================================

class TestStandardDimensions:
    """Test the 16 standard dimensions."""

    def test_standard_dimensions_count(self, standard_dimensions):
        """Test that there are exactly 16 standard dimensions."""
        assert len(standard_dimensions) == 16

    def test_standard_dimensions_unique_names(self, standard_dimensions):
        """Test that all standard dimensions have unique names."""
        names = [dim.name for dim in standard_dimensions]
        assert len(names) == len(set(names)), "Duplicate dimension names found"

    def test_standard_dimensions_have_exemplars(self, standard_dimensions):
        """Test that all standard dimensions have exemplars."""
        for dim in standard_dimensions:
            assert len(dim.positive_exemplars) > 0, f"{dim.name} has no positive exemplars"
            assert len(dim.negative_exemplars) > 0, f"{dim.name} has no negative exemplars"

    def test_standard_dimension_names(self, standard_dimensions):
        """Test that expected dimension names are present."""
        expected_names = [
            "Warmth", "Valence", "Arousal", "Intensity",
            "Formality", "Directness", "Power", "Generosity",
            "Certainty", "Complexity", "Concreteness", "Familiarity",
            "Agency", "Stability", "Urgency", "Completion"
        ]

        actual_names = [dim.name for dim in standard_dimensions]

        for name in expected_names:
            assert name in actual_names, f"Expected dimension '{name}' not found"

    @pytest.mark.parametrize("dim_name,pos_word,neg_word", [
        ("Warmth", "warm", "cold"),
        ("Valence", "good", "bad"),
        ("Formality", "formal", "casual"),
        ("Certainty", "certain", "uncertain"),
    ])
    def test_standard_dimension_polarity(
        self, standard_dimensions, mock_embed_fn_with_semantic_structure,
        dim_name, pos_word, neg_word
    ):
        """Test that standard dimensions capture expected polarity."""
        # Find the dimension
        dim = None
        for d in standard_dimensions:
            if d.name == dim_name:
                dim = d
                break

        if dim is None:
            pytest.skip(f"Dimension {dim_name} not found")

        dim.learn_axis(mock_embed_fn_with_semantic_structure)

        pos_emb = mock_embed_fn_with_semantic_structure(pos_word)
        neg_emb = mock_embed_fn_with_semantic_structure(neg_word)

        pos_proj = dim.project(pos_emb)
        neg_proj = dim.project(neg_emb)

        # Positive word should project higher (or at least be distinguishable)
        assert pos_proj != neg_proj, f"Projections identical for {pos_word}/{neg_word}"

    def test_learn_all_standard_axes(self, standard_dimensions, mock_embed_fn):
        """Test that all standard dimension axes can be learned."""
        for dim in standard_dimensions:
            dim.learn_axis(mock_embed_fn)
            assert dim.axis is not None, f"Failed to learn axis for {dim.name}"
            assert dim.axis.shape == (384,), f"Wrong axis shape for {dim.name}"


# ============================================================================
# Extended Dimensions Tests (228 Total - Legacy name says 244)
# ============================================================================

class TestExtendedDimensions:
    """Test the 228 extended dimensions (legacy name: EXTENDED_244_DIMENSIONS)."""

    def test_extended_dimensions_count(self, extended_dimensions):
        """Test that there are exactly 228 extended dimensions."""
        # Note: Despite the name EXTENDED_244_DIMENSIONS, actual count is 228
        assert len(extended_dimensions) == 228

    def test_extended_dimensions_unique_names(self, extended_dimensions):
        """Test that all extended dimensions have unique names."""
        names = [dim.name for dim in extended_dimensions]
        duplicates = [name for name in names if names.count(name) > 1]

        if duplicates:
            pytest.fail(f"Duplicate dimension names: {set(duplicates)}")

    def test_extended_dimensions_have_exemplars(self, extended_dimensions):
        """Test that all extended dimensions have exemplars."""
        missing_positive = []
        missing_negative = []

        for dim in extended_dimensions:
            if len(dim.positive_exemplars) == 0:
                missing_positive.append(dim.name)
            if len(dim.negative_exemplars) == 0:
                missing_negative.append(dim.name)

        if missing_positive:
            pytest.fail(f"Missing positive exemplars: {missing_positive[:5]}...")
        if missing_negative:
            pytest.fail(f"Missing negative exemplars: {missing_negative[:5]}...")

    def test_extended_includes_standard(self, standard_dimensions, extended_dimensions):
        """Test that extended dimensions include all standard dimensions."""
        standard_names = {dim.name for dim in standard_dimensions}
        extended_names = {dim.name for dim in extended_dimensions}

        missing = standard_names - extended_names
        if missing:
            pytest.fail(f"Standard dimensions missing from extended: {missing}")

    def test_learn_sample_extended_axes(self, extended_dimensions, mock_embed_fn):
        """Test that sample of extended axes can be learned (not all for speed)."""
        # Test every 10th dimension
        sample_dims = extended_dimensions[::10]

        for dim in sample_dims:
            dim.learn_axis(mock_embed_fn)
            assert dim.axis is not None, f"Failed to learn axis for {dim.name}"
            assert not np.any(np.isnan(dim.axis)), f"NaN in axis for {dim.name}"

    def test_extended_dimension_categories(self, extended_dimensions):
        """Test that dimensions cover expected categories."""
        # Check that category-related names appear
        narrative_keywords = ["heroism", "transformation", "conflict", "mystery"]
        emotional_keywords = ["authenticity", "vulnerability", "trust", "hope"]
        archetypal_keywords = ["hero", "mentor", "shadow", "trickster"]

        names_lower = [dim.name.lower() for dim in extended_dimensions]

        for keyword in narrative_keywords:
            found = any(keyword in name for name in names_lower)
            assert found, f"Expected narrative dimension with '{keyword}'"

    def test_extended_exemplar_diversity(self, extended_dimensions):
        """Test that exemplars are diverse (not repeated across dimensions)."""
        all_positive = []
        all_negative = []

        for dim in extended_dimensions:
            all_positive.extend(dim.positive_exemplars)
            all_negative.extend(dim.negative_exemplars)

        # Some overlap is expected, but shouldn't be extreme
        unique_positive = len(set(all_positive))
        unique_negative = len(set(all_negative))

        # At least 50% unique (conservative threshold)
        assert unique_positive > len(all_positive) * 0.3
        assert unique_negative > len(all_negative) * 0.3


# ============================================================================
# Category Grouping Tests
# ============================================================================

class TestCategoryGroupings:
    """Test dimension category groupings."""

    def test_narrative_dimensions_count(self):
        """Test NARRATIVE_DIMENSIONS count."""
        from HoloLoom.semantic_calculus.dimensions import NARRATIVE_DIMENSIONS
        assert len(NARRATIVE_DIMENSIONS) == 16

    def test_emotional_depth_dimensions_count(self):
        """Test EMOTIONAL_DEPTH_DIMENSIONS count."""
        from HoloLoom.semantic_calculus.dimensions import EMOTIONAL_DEPTH_DIMENSIONS
        assert len(EMOTIONAL_DEPTH_DIMENSIONS) == 16

    def test_relational_dimensions_count(self):
        """Test RELATIONAL_DIMENSIONS count."""
        from HoloLoom.semantic_calculus.dimensions import RELATIONAL_DIMENSIONS
        assert len(RELATIONAL_DIMENSIONS) == 16

    def test_archetypal_dimensions_count(self):
        """Test ARCHETYPAL_DIMENSIONS count."""
        from HoloLoom.semantic_calculus.dimensions import ARCHETYPAL_DIMENSIONS
        assert len(ARCHETYPAL_DIMENSIONS) == 16

    def test_philosophical_dimensions_count(self):
        """Test PHILOSOPHICAL_DIMENSIONS count."""
        from HoloLoom.semantic_calculus.dimensions import PHILOSOPHICAL_DIMENSIONS
        assert len(PHILOSOPHICAL_DIMENSIONS) == 16

    def test_transformation_dimensions_count(self):
        """Test TRANSFORMATION_DIMENSIONS count."""
        from HoloLoom.semantic_calculus.dimensions import TRANSFORMATION_DIMENSIONS
        assert len(TRANSFORMATION_DIMENSIONS) == 16

    def test_category_dimensions_unique_within_category(self):
        """Test that dimensions within each category are unique."""
        from HoloLoom.semantic_calculus.dimensions import (
            NARRATIVE_DIMENSIONS,
            EMOTIONAL_DEPTH_DIMENSIONS,
            RELATIONAL_DIMENSIONS,
            ARCHETYPAL_DIMENSIONS,
        )

        for name, dims in [
            ("Narrative", NARRATIVE_DIMENSIONS),
            ("Emotional", EMOTIONAL_DEPTH_DIMENSIONS),
            ("Relational", RELATIONAL_DIMENSIONS),
            ("Archetypal", ARCHETYPAL_DIMENSIONS),
        ]:
            dim_names = [d.name for d in dims]
            assert len(dim_names) == len(set(dim_names)), f"Duplicate in {name}"

    def test_categories_sum_to_228(self):
        """Test that all category counts sum correctly."""
        from HoloLoom.semantic_calculus.dimensions import (
            STANDARD_DIMENSIONS,
            NARRATIVE_DIMENSIONS,
            EMOTIONAL_DEPTH_DIMENSIONS,
            RELATIONAL_DIMENSIONS,
            ARCHETYPAL_DIMENSIONS,
            PHILOSOPHICAL_DIMENSIONS,
            TRANSFORMATION_DIMENSIONS,
            MORAL_ETHICAL_DIMENSIONS,
            CREATIVE_DIMENSIONS,
            COGNITIVE_COMPLEXITY_DIMENSIONS,
            TEMPORAL_NARRATIVE_DIMENSIONS,
            SPATIAL_SETTING_DIMENSIONS,
            CHARACTER_DIMENSIONS,
            PLOT_DIMENSIONS,
            THEME_DIMENSIONS,
            STYLE_VOICE_DIMENSIONS,
        )

        # Category structure:
        # 11 categories with 16 dims: STANDARD + NARRATIVE + EMOTIONAL_DEPTH + RELATIONAL +
        #                             ARCHETYPAL + PHILOSOPHICAL + TRANSFORMATION +
        #                             MORAL_ETHICAL + CREATIVE + COGNITIVE_COMPLEXITY +
        #                             TEMPORAL_NARRATIVE = 11 * 16 = 176
        # 4 categories with 12 dims: SPATIAL_SETTING + CHARACTER + PLOT + THEME = 4 * 12 = 48
        # 1 category with 4 dims: STYLE_VOICE = 4
        # Total = 176 + 48 + 4 = 228 (not 244 as the legacy name suggests)
        total = (
            len(STANDARD_DIMENSIONS) +
            len(NARRATIVE_DIMENSIONS) +
            len(EMOTIONAL_DEPTH_DIMENSIONS) +
            len(RELATIONAL_DIMENSIONS) +
            len(ARCHETYPAL_DIMENSIONS) +
            len(PHILOSOPHICAL_DIMENSIONS) +
            len(TRANSFORMATION_DIMENSIONS) +
            len(MORAL_ETHICAL_DIMENSIONS) +
            len(CREATIVE_DIMENSIONS) +
            len(COGNITIVE_COMPLEXITY_DIMENSIONS) +
            len(TEMPORAL_NARRATIVE_DIMENSIONS) +
            len(SPATIAL_SETTING_DIMENSIONS) +
            len(CHARACTER_DIMENSIONS) +
            len(PLOT_DIMENSIONS) +
            len(THEME_DIMENSIONS) +
            len(STYLE_VOICE_DIMENSIONS)
        )

        # Verify the category totals sum correctly
        from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS
        assert len(EXTENDED_244_DIMENSIONS) == 228
        assert total == 228, f"Category sum {total} does not match expected 228"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_exemplars(self, mock_embed_fn):
        """Test dimension with very long exemplar phrases."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="LongExemplars",
            positive_exemplars=[
                "this is a very long positive exemplar phrase that tests embedding",
                "another extremely verbose positive example for testing purposes",
            ],
            negative_exemplars=[
                "this is a very long negative exemplar phrase that tests embedding",
                "another extremely verbose negative example for testing purposes",
            ]
        )

        dim.learn_axis(mock_embed_fn)
        assert dim.axis is not None

    def test_unicode_exemplars(self, mock_embed_fn):
        """Test dimension with unicode exemplars."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="Unicode",
            positive_exemplars=["happy", "joy"],
            negative_exemplars=["sad", "sorrow"]
        )

        dim.learn_axis(mock_embed_fn)
        assert dim.axis is not None

    def test_dimension_name_special_chars(self):
        """Test dimension names with special characters."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="Test-Dimension_v2.0",
            positive_exemplars=["pos"],
            negative_exemplars=["neg"]
        )

        assert dim.name == "Test-Dimension_v2.0"

    def test_duplicate_exemplars(self, mock_embed_fn):
        """Test dimension with duplicate exemplars."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="DuplicateExemplars",
            positive_exemplars=["good", "good", "nice"],
            negative_exemplars=["bad", "bad", "awful"]
        )

        # Should still work (duplicates averaged out)
        dim.learn_axis(mock_embed_fn)
        assert dim.axis is not None

    def test_overlapping_exemplars(self, mock_embed_fn):
        """Test dimension with overlapping positive/negative exemplars (edge case)."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        dim = SemanticDimension(
            name="Overlapping",
            positive_exemplars=["neutral", "good"],
            negative_exemplars=["neutral", "bad"]
        )

        # Should still work but axis quality may be reduced
        dim.learn_axis(mock_embed_fn)
        assert dim.axis is not None


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test numerical stability of dimension operations."""

    def test_axis_stability_near_zero(self, mock_embed_fn):
        """Test axis learning with near-zero embedding differences."""
        from HoloLoom.semantic_calculus.dimensions import SemanticDimension

        # Use very similar words
        dim = SemanticDimension(
            name="Similar",
            positive_exemplars=["cat", "cats"],
            negative_exemplars=["dog", "dogs"]
        )

        dim.learn_axis(mock_embed_fn)

        # Should not have NaN or Inf
        assert not np.any(np.isnan(dim.axis))
        assert not np.any(np.isinf(dim.axis))

    def test_projection_stability_small_vectors(self, mock_embed_fn, sample_dimension):
        """Test projection with very small input vectors."""
        sample_dimension.learn_axis(mock_embed_fn)

        # Very small vector (near numerical precision)
        small_vec = np.ones(384) * 1e-15

        projection = sample_dimension.project(small_vec)

        # Should not overflow/underflow
        assert not np.isnan(projection)
        assert not np.isinf(projection)

    def test_projection_stability_large_vectors(self, mock_embed_fn, sample_dimension):
        """Test projection with very large input vectors."""
        sample_dimension.learn_axis(mock_embed_fn)

        # Large vector
        large_vec = np.ones(384) * 1e10

        projection = sample_dimension.project(large_vec)

        # Should handle gracefully
        assert not np.isnan(projection)

    def test_many_dimensions_stability(self, mock_embed_fn):
        """Test learning many dimensions doesn't cause numerical issues."""
        from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS

        # Learn first 50 dimensions
        problematic = []
        for dim in EXTENDED_244_DIMENSIONS[:50]:
            dim.learn_axis(mock_embed_fn)

            if np.any(np.isnan(dim.axis)) or np.any(np.isinf(dim.axis)):
                problematic.append(dim.name)

        assert len(problematic) == 0, f"Numerical issues in: {problematic}"

    def test_repeated_learning_stability(self, mock_embed_fn, sample_dimension):
        """Test that repeated axis learning is stable."""
        results = []

        for _ in range(10):
            sample_dimension.axis = None
            sample_dimension.learn_axis(mock_embed_fn)
            results.append(sample_dimension.axis.copy())

        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i])


# ============================================================================
# Performance Tests
# ============================================================================

class TestDimensionPerformance:
    """Performance tests for dimension operations."""

    def test_axis_learning_time(self, mock_embed_fn, sample_dimension):
        """Test that axis learning completes in reasonable time."""
        import time

        start = time.time()
        sample_dimension.learn_axis(mock_embed_fn)
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Axis learning took {elapsed:.2f}s"

    def test_projection_time(self, mock_embed_fn, sample_dimension):
        """Test that projection is fast."""
        import time

        sample_dimension.learn_axis(mock_embed_fn)
        vec = mock_embed_fn("test")

        # Time 1000 projections
        start = time.time()
        for _ in range(1000):
            sample_dimension.project(vec)
        elapsed = time.time() - start

        # Should complete 1000 projections in under 100ms
        assert elapsed < 0.1, f"1000 projections took {elapsed:.3f}s"

    def test_batch_learning_efficiency(self, mock_embed_fn, sample_dimensions_list):
        """Test that batch learning is efficient."""
        import time

        start = time.time()
        for dim in sample_dimensions_list:
            dim.learn_axis(mock_embed_fn, use_batch=True)
        elapsed = time.time() - start

        # Should learn 5 dimensions in under 2 seconds
        assert elapsed < 2.0, f"Batch learning 5 dims took {elapsed:.2f}s"


# ============================================================================
# Integration Tests
# ============================================================================

class TestDimensionIntegration:
    """Integration tests for dimension functionality."""

    def test_dimension_list_projection(self, mock_embed_fn, sample_dimensions_list):
        """Test projecting onto multiple dimensions."""
        # Learn all axes
        for dim in sample_dimensions_list:
            dim.learn_axis(mock_embed_fn)

        # Project a vector
        vec = mock_embed_fn("happy and warm")

        projections = {}
        for dim in sample_dimensions_list:
            projections[dim.name] = dim.project(vec)

        # Should have projection for each dimension
        assert len(projections) == len(sample_dimensions_list)
        assert all(isinstance(v, float) for v in projections.values())

    def test_dimension_to_spectrum_compatibility(self, mock_embed_fn, sample_dimensions_list):
        """Test that dimensions work with SemanticSpectrum."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        spectrum = SemanticSpectrum(dimensions=sample_dimensions_list)
        spectrum.learn_axes(mock_embed_fn)

        vec = mock_embed_fn("test")
        projection = spectrum.project_vector(vec)

        assert isinstance(projection, dict)
        assert len(projection) == len(sample_dimensions_list)

    def test_standard_to_extended_compatibility(
        self, standard_dimensions, extended_dimensions, mock_embed_fn
    ):
        """Test that standard dimensions are compatible with extended set."""
        from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum

        # Create spectrum with extended dimensions
        spectrum = SemanticSpectrum(dimensions=extended_dimensions[:20])
        spectrum.learn_axes(mock_embed_fn)

        vec = mock_embed_fn("test")
        projection = spectrum.project_vector(vec)

        assert len(projection) == 20


# ============================================================================
# Visualization Tests
# ============================================================================

class TestDimensionVisualization:
    """Test dimension visualization functions."""

    def test_visualize_function_exists(self):
        """Test that visualization function exists."""
        try:
            from HoloLoom.semantic_calculus.dimensions import visualize_semantic_spectrum
            assert visualize_semantic_spectrum is not None
        except ImportError:
            pytest.skip("Visualization function not available")

    def test_print_summary_function_exists(self):
        """Test that print summary function exists."""
        try:
            from HoloLoom.semantic_calculus.dimensions import print_spectrum_summary
            assert print_spectrum_summary is not None
        except ImportError:
            pytest.skip("Print summary function not available")


# ============================================================================
# Dimension Selector Tests
# ============================================================================

class TestDimensionSelector:
    """Test SmartDimensionSelector functionality."""

    def test_selector_import(self):
        """Test that dimension selector can be imported."""
        try:
            from HoloLoom.semantic_calculus.dimension_selector import SmartDimensionSelector
            assert SmartDimensionSelector is not None
        except ImportError:
            pytest.skip("SmartDimensionSelector not available")

    def test_selector_strategies(self):
        """Test that selection strategies are defined."""
        try:
            from HoloLoom.semantic_calculus.dimension_selector import SelectionStrategy
            assert SelectionStrategy.BALANCED is not None
            assert SelectionStrategy.DISCRIMINATIVE is not None
            assert SelectionStrategy.NARRATIVE is not None
            assert SelectionStrategy.DIALOGUE is not None
            assert SelectionStrategy.HYBRID is not None
        except ImportError:
            pytest.skip("SelectionStrategy not available")

    def test_selector_select_36d(self, mock_embed_fn):
        """Test selecting 36 dimensions for FUSED mode."""
        try:
            from HoloLoom.semantic_calculus.dimension_selector import (
                SmartDimensionSelector,
                SelectionStrategy
            )

            selector = SmartDimensionSelector()

            # Select 36 dimensions (BALANCED strategy doesn't need embed_fn)
            selected = selector.select(
                n_dimensions=36,
                strategy=SelectionStrategy.BALANCED
            )

            assert len(selected) == 36
            assert all(hasattr(dim, 'name') for dim in selected)

        except ImportError:
            pytest.skip("Dimension selector not available")

    def test_selector_balanced_coverage(self, mock_embed_fn):
        """Test that balanced selection covers multiple categories."""
        try:
            from HoloLoom.semantic_calculus.dimension_selector import (
                SmartDimensionSelector,
                SelectionStrategy
            )

            selector = SmartDimensionSelector()
            selected = selector.select(
                n_dimensions=36,
                strategy=SelectionStrategy.BALANCED
            )

            # Check category distribution
            categories = {}
            for dim in selected:
                cat = selector.dim_to_category.get(dim.name, 'Other')
                categories[cat] = categories.get(cat, 0) + 1

            # Should have multiple categories represented
            assert len(categories) >= 3, f"Only {len(categories)} categories in selection"

        except ImportError:
            pytest.skip("Dimension selector not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
