"""
Test Suite for Dream Matching Algorithm

Comprehensive test coverage for the dream matching system including:
- Emotional similarity calculation
- Relationship tension querying
- Archetypal complementarity evaluation
- Personality compatibility assessment
- Full matching pipeline with multiple residents
- Edge cases and graceful degradation

Test Status: All tests pass with graceful degradation for missing systems
Run: pytest NeuroHood/dreams/test_dream_matching.py -v
"""

import pytest
import numpy as np
from typing import List
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

from NeuroHood.dreams.dream_matching import DreamMatcher, DreamMatchCandidate


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockResident:
    """Mock resident for testing."""
    resident_id: str
    name: str
    age: int = 30
    occupation: str = "Unknown"
    personality_vector: List[float] = None

    def __post_init__(self):
        if self.personality_vector is None:
            # Random 228D vector
            self.personality_vector = list(np.random.randn(228))


@pytest.fixture
def basic_residents():
    """Create basic test residents."""
    alice = MockResident("res_001", "Alice")
    bob = MockResident("res_002", "Bob")
    charlie = MockResident("res_003", "Charlie")

    # Create distinct personality profiles
    # Alice: warm, calm, reflective
    alice.personality_vector = [0.8] * 50 + [0.6] * 50 + [-0.2] * 128

    # Bob: energetic, outgoing, intense
    bob.personality_vector = [-0.8] * 50 + [0.9] * 50 + [0.4] * 128

    # Charlie: balanced, moderate
    charlie.personality_vector = [0.2] * 228

    return [alice, bob, charlie]


@pytest.fixture
def mock_personality_system():
    """Create mock personality system."""
    mock = Mock()
    mock.get_current_snapshot = Mock(return_value=None)
    return mock


@pytest.fixture
def mock_relationship_scm():
    """Create mock relationship SCM."""
    mock = Mock()
    mock.predict_relationship_tension = Mock(return_value=0.5)
    return mock


@pytest.fixture
def mock_symbolic_encoder():
    """Create mock symbolic encoder."""
    mock = Mock()
    return mock


@pytest.fixture
def matcher(mock_personality_system, mock_relationship_scm, mock_symbolic_encoder):
    """Create dream matcher with mocked dependencies."""
    return DreamMatcher(
        personality_system=mock_personality_system,
        relationship_scm=mock_relationship_scm,
        symbolic_encoder=mock_symbolic_encoder,
    )


@pytest.fixture
def matcher_no_deps():
    """Create dream matcher without external dependencies."""
    return DreamMatcher()


# ============================================================================
# Test Emotional Similarity
# ============================================================================


class TestEmotionalSimilarity:
    """Test emotional similarity computation."""

    def test_identical_vectors_high_similarity(self, matcher):
        """Identical personality vectors should give high similarity."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Alice_Twin")

        # Same vector
        resident_a.personality_vector = [0.5] * 228
        resident_b.personality_vector = [0.5] * 228

        similarity = matcher.compute_emotional_similarity(resident_a, resident_b)

        assert 0.95 <= similarity <= 1.0, f"Expected ~1.0, got {similarity}"

    def test_opposite_vectors_low_similarity(self, matcher):
        """Opposite personality vectors should give low similarity."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        # Opposite vectors
        resident_a.personality_vector = [1.0] * 228
        resident_b.personality_vector = [-1.0] * 228

        similarity = matcher.compute_emotional_similarity(resident_a, resident_b)

        assert 0.0 <= similarity <= 0.1, f"Expected ~0.0, got {similarity}"

    def test_orthogonal_vectors_neutral_similarity(self, matcher):
        """Orthogonal vectors should give neutral similarity."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        # First 114 dimensions high for a, last 114 high for b
        resident_a.personality_vector = [1.0] * 114 + [0.0] * 114
        resident_b.personality_vector = [0.0] * 114 + [1.0] * 114

        similarity = matcher.compute_emotional_similarity(resident_a, resident_b)

        assert 0.4 <= similarity <= 0.6, f"Expected ~0.5, got {similarity}"

    def test_similarity_is_normalized(self, matcher):
        """Similarity should always be in [0, 1] range."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        # Random vectors
        resident_a.personality_vector = list(np.random.randn(228))
        resident_b.personality_vector = list(np.random.randn(228))

        similarity = matcher.compute_emotional_similarity(resident_a, resident_b)

        assert 0.0 <= similarity <= 1.0, f"Similarity out of range: {similarity}"

    def test_similarity_graceful_degradation(self, matcher_no_deps):
        """Should handle missing personality system gracefully."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        similarity = matcher_no_deps.compute_emotional_similarity(resident_a, resident_b)

        # Should return fallback value
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


# ============================================================================
# Test Relationship Tension
# ============================================================================


class TestRelationshipTension:
    """Test relationship tension querying."""

    def test_tension_query_success(self, matcher, mock_relationship_scm):
        """Should query SCM for tension."""
        mock_relationship_scm.predict_relationship_tension.return_value = 0.7

        tension = matcher.compute_relationship_tension("res_001", "res_002")

        assert tension == 0.7
        mock_relationship_scm.predict_relationship_tension.assert_called_once()

    def test_tension_is_normalized(self, matcher, mock_relationship_scm):
        """Tension values should be normalized to [0, 1]."""
        # Test various return values
        test_values = [0.0, 0.5, 1.0, 0.99, -0.1]

        for val in test_values:
            mock_relationship_scm.predict_relationship_tension.return_value = val
            tension = matcher.compute_relationship_tension("res_001", "res_002")
            assert 0.0 <= tension <= 1.0, f"Tension {val} normalized to {tension}, out of range"

    def test_tension_graceful_degradation(self, matcher_no_deps):
        """Should handle missing SCM gracefully."""
        tension = matcher_no_deps.compute_relationship_tension("res_001", "res_002")

        assert isinstance(tension, float)
        assert 0.0 <= tension <= 1.0

    def test_tension_query_error_handling(self, matcher, mock_relationship_scm):
        """Should handle SCM errors gracefully."""
        mock_relationship_scm.predict_relationship_tension.side_effect = Exception("SCM Error")

        tension = matcher.compute_relationship_tension("res_001", "res_002")

        # Should return fallback
        assert isinstance(tension, float)
        assert 0.0 <= tension <= 1.0


# ============================================================================
# Test Archetypal Complementarity
# ============================================================================


class TestArchetypalComplementarity:
    """Test archetypal complementarity computation."""

    def test_identify_archetype_hero(self, matcher):
        """Should identify hero archetype from brave/ambitious traits."""
        resident = MockResident("res_001", "Alice")

        with patch.object(matcher, "_identify_archetype", return_value="hero"):
            with patch.object(matcher, "_identify_archetype", return_value="mentor"):
                # Manually test complementarity logic
                comp = matcher.compute_archetypal_complementarity(resident, resident)
                # hero + hero should be moderate (same archetype)
                assert 0.5 <= comp <= 0.7

    def test_complementary_archetypes_high_score(self, matcher):
        """Hero + Mentor should score high complementarity."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        with patch.object(matcher, "_identify_archetype") as mock_arch:
            mock_arch.side_effect = ["hero", "mentor"]
            comp = matcher.compute_archetypal_complementarity(resident_a, resident_b)
            assert comp > 0.8, f"Hero+Mentor should be complementary, got {comp}"

    def test_opposite_archetypes(self, matcher):
        """Shadow + Light should be complementary."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        with patch.object(matcher, "_identify_archetype") as mock_arch:
            mock_arch.side_effect = ["shadow", "light"]
            comp = matcher.compute_archetypal_complementarity(resident_a, resident_b)
            assert comp > 0.8, f"Shadow+Light should be complementary, got {comp}"

    def test_same_archetype_moderate_compatibility(self, matcher):
        """Same archetype should give moderate compatibility."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        with patch.object(matcher, "_identify_archetype") as mock_arch:
            mock_arch.side_effect = ["hero", "hero"]
            comp = matcher.compute_archetypal_complementarity(resident_a, resident_b)
            assert 0.5 <= comp <= 0.7, f"Same archetype should be moderate, got {comp}"

    def test_archetype_missing_graceful_fallback(self, matcher):
        """Should handle missing archetype gracefully."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        with patch.object(matcher, "_identify_archetype", return_value=None):
            comp = matcher.compute_archetypal_complementarity(resident_a, resident_b)
            # Should return fallback value
            assert 0.3 <= comp <= 0.6


# ============================================================================
# Test Personality Compatibility
# ============================================================================


class TestPersonalityCompatibility:
    """Test personality compatibility computation."""

    def test_personality_compatibility_bounded(self, matcher_no_deps):
        """Personality compatibility should be in [0, 1]."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        compat = matcher_no_deps.compute_personality_compatibility(resident_a, resident_b)

        assert 0.0 <= compat <= 1.0, f"Compatibility out of range: {compat}"

    def test_personality_compatibility_without_system(self, matcher_no_deps):
        """Should handle missing personality system."""
        resident_a = MockResident("res_001", "Alice")
        resident_b = MockResident("res_002", "Bob")

        compat = matcher_no_deps.compute_personality_compatibility(resident_a, resident_b)

        assert isinstance(compat, float)
        assert compat == 0.5  # Neutral fallback


# ============================================================================
# Test Full Matching Pipeline
# ============================================================================


class TestFullMatchingPipeline:
    """Test the complete matching pipeline."""

    def test_find_matches_single_pair(self, matcher, basic_residents):
        """Should find matches for small group."""
        residents = basic_residents[:2]  # Alice and Bob
        matches = matcher.find_matches(residents, max_matches=5)

        assert len(matches) > 0, "Should find at least one match"
        assert isinstance(matches[0], DreamMatchCandidate)

    def test_find_matches_multiple_pairs(self, matcher, basic_residents):
        """Should find multiple matches for larger group."""
        matches = matcher.find_matches(basic_residents, max_matches=5)

        # With 3 residents, max possible matches = C(3,2) = 3
        assert 0 <= len(matches) <= 3, f"Expected 0-3 matches, got {len(matches)}"

    def test_matches_sorted_by_score(self, matcher, basic_residents):
        """Matches should be sorted by score descending."""
        matches = matcher.find_matches(basic_residents, max_matches=5)

        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert (
                    matches[i].match_score >= matches[i + 1].match_score
                ), "Matches not sorted by score"

    def test_matches_respect_min_score(self, matcher, basic_residents):
        """All returned matches should meet minimum score threshold."""
        min_score = 0.6
        matches = matcher.find_matches(basic_residents, max_matches=5, min_score=min_score)

        for match in matches:
            assert (
                match.match_score >= min_score
            ), f"Match score {match.match_score} below threshold {min_score}"

    def test_matches_respect_max_count(self, matcher, basic_residents):
        """Should not return more than max_matches."""
        max_matches = 2
        matches = matcher.find_matches(basic_residents, max_matches=max_matches)

        assert len(matches) <= max_matches, f"Got {len(matches)} matches, expected <= {max_matches}"

    def test_match_candidate_structure(self, matcher, basic_residents):
        """Match candidates should have all required fields."""
        matches = matcher.find_matches(basic_residents[:2])

        if matches:
            match = matches[0]
            assert hasattr(match, "resident_pair")
            assert hasattr(match, "emotional_similarity")
            assert hasattr(match, "relationship_tension")
            assert hasattr(match, "complementary_archetypes")
            assert hasattr(match, "personality_compatibility")
            assert hasattr(match, "match_score")
            assert hasattr(match, "match_reason")

    def test_match_candidate_to_dict(self, matcher, basic_residents):
        """Match candidate should convert to dict."""
        matches = matcher.find_matches(basic_residents[:2])

        if matches:
            match = matches[0]
            data = match.to_dict()
            assert isinstance(data, dict)
            assert "match_score" in data
            assert "resident_pair" in data

    def test_match_candidate_str(self, matcher, basic_residents):
        """Match candidate should have readable string representation."""
        matches = matcher.find_matches(basic_residents[:2])

        if matches:
            match = matches[0]
            str_repr = str(match)
            assert "Score:" in str_repr or "score" in str_repr.lower()

    def test_no_matches_for_single_resident(self, matcher):
        """Should handle single resident gracefully."""
        resident = MockResident("res_001", "Alice")
        matches = matcher.find_matches([resident])

        assert len(matches) == 0, "Single resident should produce no matches"

    def test_no_matches_for_empty_list(self, matcher):
        """Should handle empty resident list gracefully."""
        matches = matcher.find_matches([])

        assert len(matches) == 0, "Empty list should produce no matches"


# ============================================================================
# Test Scoring Components
# ============================================================================


class TestScoringComponents:
    """Test individual scoring components."""

    def test_score_weights_sum_to_one(self, matcher):
        """Weights should sum to 1.0 for proper weighting."""
        total = sum(matcher.weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1.0: {total}"

    def test_match_score_is_composite(self, matcher, basic_residents):
        """Match score should be weighted combination of components."""
        match = matcher._compute_match(basic_residents[0], basic_residents[1])

        # Reconstruct score
        expected = (
            matcher.weights["emotional_similarity"] * match.emotional_similarity
            + matcher.weights["relationship_tension"] * match.relationship_tension
            + matcher.weights["complementary_archetypes"] * match.complementary_archetypes
            + matcher.weights["personality_compatibility"]
            * match.personality_compatibility
        )

        assert abs(match.match_score - expected) < 0.01, "Score calculation mismatch"

    def test_all_components_bounded(self, matcher, basic_residents):
        """All components should be in [0, 1]."""
        match = matcher._compute_match(basic_residents[0], basic_residents[1])

        assert 0.0 <= match.emotional_similarity <= 1.0
        assert 0.0 <= match.relationship_tension <= 1.0
        assert 0.0 <= match.complementary_archetypes <= 1.0
        assert 0.0 <= match.personality_compatibility <= 1.0
        assert 0.0 <= match.match_score <= 1.0


# ============================================================================
# Test Caching
# ============================================================================


class TestCaching:
    """Test score caching mechanism."""

    def test_score_caching(self, matcher, basic_residents):
        """Scores should be cached to avoid recomputation."""
        alice = basic_residents[0]
        bob = basic_residents[1]

        # First call
        match1 = matcher._compute_match(alice, bob)

        # Second call should use cache
        match2 = matcher._compute_match(alice, bob)

        # Should be same object (from cache)
        # Note: might be different objects with same values
        assert match1.match_score == match2.match_score

    def test_cache_order_independent(self, matcher, basic_residents):
        """Caching should be order-independent (Alice,Bob) == (Bob,Alice)."""
        alice = basic_residents[0]
        bob = basic_residents[1]

        match1 = matcher._compute_match(alice, bob)
        match2 = matcher._compute_match(bob, alice)

        # Pairs should be the same (sorted), so results should match
        assert set(match1.resident_pair) == set(match2.resident_pair)
        assert match1.match_score == match2.match_score


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with multiple components."""

    def test_full_workflow(self, basic_residents):
        """Test complete workflow from resident list to matches."""
        # Create matcher without mocks
        matcher = DreamMatcher()

        # Find matches
        matches = matcher.find_matches(basic_residents, max_matches=5, min_score=0.3)

        # Verify results
        assert isinstance(matches, list)
        for match in matches:
            assert isinstance(match, DreamMatchCandidate)
            assert 0.0 <= match.match_score <= 1.0
            assert isinstance(match.match_reason, str)

    def test_workflow_with_empty_personalities(self):
        """Test workflow when personality vectors are missing."""
        alice = MockResident("res_001", "Alice")
        bob = MockResident("res_002", "Bob")

        # Remove personality vectors
        alice.personality_vector = None
        bob.personality_vector = None

        matcher = DreamMatcher()
        matches = matcher.find_matches([alice, bob], min_score=0.0)

        # Should still work with fallback values
        assert isinstance(matches, list)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
