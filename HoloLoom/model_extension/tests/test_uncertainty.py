"""
Tests for Uncertainty as First-Class Output (Addendum A.1)

Tests the UncertaintyEnvelope, confidence tiers, and variance-aware
confidence calculation.
"""

import pytest
import math
from HoloLoom.model_extension.uncertainty import (
    UncertaintyEnvelope,
    UncertaintySource,
    ConfidenceTier,
    compute_variance_aware_confidence,
    should_refuse_to_converge,
)


class TestConfidenceTier:
    """Tests for ConfidenceTier enum."""

    def test_from_score_definitive(self):
        """Test DEFINITIVE tier (>=0.95)."""
        assert ConfidenceTier.from_score(0.95) == ConfidenceTier.DEFINITIVE
        assert ConfidenceTier.from_score(0.99) == ConfidenceTier.DEFINITIVE
        assert ConfidenceTier.from_score(1.0) == ConfidenceTier.DEFINITIVE

    def test_from_score_high(self):
        """Test HIGH tier (0.80-0.95)."""
        assert ConfidenceTier.from_score(0.80) == ConfidenceTier.HIGH
        assert ConfidenceTier.from_score(0.87) == ConfidenceTier.HIGH
        assert ConfidenceTier.from_score(0.94) == ConfidenceTier.HIGH

    def test_from_score_moderate(self):
        """Test MODERATE tier (0.60-0.80)."""
        assert ConfidenceTier.from_score(0.60) == ConfidenceTier.MODERATE
        assert ConfidenceTier.from_score(0.70) == ConfidenceTier.MODERATE
        assert ConfidenceTier.from_score(0.79) == ConfidenceTier.MODERATE

    def test_from_score_low(self):
        """Test LOW tier (0.40-0.60)."""
        assert ConfidenceTier.from_score(0.40) == ConfidenceTier.LOW
        assert ConfidenceTier.from_score(0.50) == ConfidenceTier.LOW
        assert ConfidenceTier.from_score(0.59) == ConfidenceTier.LOW

    def test_from_score_insufficient(self):
        """Test INSUFFICIENT tier (<0.40)."""
        assert ConfidenceTier.from_score(0.39) == ConfidenceTier.INSUFFICIENT
        assert ConfidenceTier.from_score(0.20) == ConfidenceTier.INSUFFICIENT
        assert ConfidenceTier.from_score(0.0) == ConfidenceTier.INSUFFICIENT

    def test_requires_verification(self):
        """Test verification requirements per tier."""
        assert not ConfidenceTier.DEFINITIVE.requires_verification
        assert not ConfidenceTier.HIGH.requires_verification
        assert not ConfidenceTier.MODERATE.requires_verification
        assert ConfidenceTier.LOW.requires_verification
        assert ConfidenceTier.INSUFFICIENT.requires_verification

    def test_output_marking(self):
        """Test human-readable output markings."""
        assert "Verified" in ConfidenceTier.DEFINITIVE.output_marking
        assert "Confident" in ConfidenceTier.HIGH.output_marking
        assert "Moderate" in ConfidenceTier.MODERATE.output_marking
        assert "Low" in ConfidenceTier.LOW.output_marking
        assert "Cannot" in ConfidenceTier.INSUFFICIENT.output_marking


class TestUncertaintySource:
    """Tests for UncertaintySource dataclass."""

    def test_valid_source(self):
        """Test creating valid uncertainty source."""
        source = UncertaintySource(
            type="retrieval_sparsity",
            contribution=0.15,
            details="Few relevant memories found"
        )
        assert source.type == "retrieval_sparsity"
        assert source.contribution == 0.15
        assert "Few" in source.details

    def test_contribution_bounds(self):
        """Test contribution must be 0.0-1.0."""
        # Valid bounds
        UncertaintySource(type="test", contribution=0.0)
        UncertaintySource(type="test", contribution=1.0)
        UncertaintySource(type="test", contribution=0.5)

        # Invalid bounds
        with pytest.raises(ValueError):
            UncertaintySource(type="test", contribution=-0.1)

        with pytest.raises(ValueError):
            UncertaintySource(type="test", contribution=1.1)


class TestUncertaintyEnvelope:
    """Tests for UncertaintyEnvelope dataclass."""

    def test_from_confidence_high(self):
        """Test creating envelope from high confidence."""
        envelope = UncertaintyEnvelope.from_confidence(0.92)

        assert envelope.point_estimate == 0.92
        assert envelope.tier == ConfidenceTier.HIGH
        assert not envelope.requires_verification
        assert envelope.credible_interval[0] <= 0.92
        assert envelope.credible_interval[1] >= 0.92

    def test_from_confidence_low(self):
        """Test creating envelope from low confidence."""
        envelope = UncertaintyEnvelope.from_confidence(0.45)

        assert envelope.point_estimate == 0.45
        assert envelope.tier == ConfidenceTier.LOW
        assert envelope.requires_verification

    def test_from_confidence_with_variance(self):
        """Test envelope with sample variance."""
        envelope = UncertaintyEnvelope.from_confidence(
            0.75,
            sample_variance=0.04
        )

        assert envelope.point_estimate == 0.75
        assert envelope.sample_variance == 0.04
        # Interval should be wider with higher variance
        width = envelope.interval_width
        assert width > 0.1  # Some meaningful interval

    def test_from_confidence_with_sources(self):
        """Test envelope with uncertainty sources."""
        sources = [
            UncertaintySource(type="retrieval", contribution=0.1),
            UncertaintySource(type="temporal", contribution=0.05),
        ]
        envelope = UncertaintyEnvelope.from_confidence(0.85, sources=sources)

        assert len(envelope.sources) == 2
        assert envelope.sources[0].type == "retrieval"
        assert envelope.sources[1].type == "temporal"

    def test_interval_width(self):
        """Test credible interval width calculation."""
        envelope = UncertaintyEnvelope(
            point_estimate=0.8,
            credible_interval=(0.7, 0.9),
            tier=ConfidenceTier.HIGH,
        )
        assert envelope.interval_width == pytest.approx(0.2)

    def test_is_high_stakes_uncertain(self):
        """Test high-stakes uncertainty detection."""
        # High confidence + no verification required = not high stakes
        envelope_high = UncertaintyEnvelope.from_confidence(0.95)
        assert not envelope_high.is_high_stakes_uncertain

        # Low confidence + verification required = high stakes
        envelope_low = UncertaintyEnvelope.from_confidence(0.35)
        assert envelope_low.is_high_stakes_uncertain

    def test_to_dict(self):
        """Test JSON serialization."""
        envelope = UncertaintyEnvelope.from_confidence(0.82)
        d = envelope.to_dict()

        assert "point_estimate" in d
        assert "credible_interval" in d
        assert "tier" in d
        assert "tier_marking" in d
        assert "requires_verification" in d
        assert d["tier"] == "high"

    def test_str_representation(self):
        """Test string representation."""
        envelope = UncertaintyEnvelope.from_confidence(0.72)
        s = str(envelope)

        assert "0.72" in s
        assert "moderate" in s

    def test_point_estimate_validation(self):
        """Test point estimate must be 0.0-1.0."""
        with pytest.raises(ValueError):
            UncertaintyEnvelope(
                point_estimate=-0.1,
                credible_interval=(0.0, 0.1),
                tier=ConfidenceTier.INSUFFICIENT,
            )

        with pytest.raises(ValueError):
            UncertaintyEnvelope(
                point_estimate=1.1,
                credible_interval=(0.9, 1.0),
                tier=ConfidenceTier.DEFINITIVE,
            )

    def test_credible_interval_validation(self):
        """Test credible interval must be valid."""
        with pytest.raises(ValueError):
            UncertaintyEnvelope(
                point_estimate=0.5,
                credible_interval=(0.6, 0.4),  # Lower > upper
                tier=ConfidenceTier.LOW,
            )


class TestVarianceAwareConfidence:
    """Tests for variance-aware confidence calculation."""

    def test_empty_samples(self):
        """Test with no samples."""
        mean, variance, uncertain = compute_variance_aware_confidence([])
        assert mean == 0.0
        assert variance == 0.0
        assert uncertain is True

    def test_single_sample(self):
        """Test with single sample."""
        mean, variance, uncertain = compute_variance_aware_confidence([0.8])
        assert mean == 0.8
        assert variance == 0.0
        assert uncertain is False

    def test_multiple_agreeing_samples(self):
        """Test with multiple agreeing samples."""
        samples = [0.85, 0.87, 0.84, 0.86]
        mean, variance, uncertain = compute_variance_aware_confidence(samples)

        assert mean == pytest.approx(0.855, rel=0.01)
        assert variance < 0.01  # Low variance
        assert uncertain is False

    def test_multiple_disagreeing_samples(self):
        """Test with multiple disagreeing samples."""
        samples = [0.3, 0.9, 0.2, 0.95]
        mean, variance, uncertain = compute_variance_aware_confidence(samples, threshold=0.10)

        assert variance > 0.10  # High variance (compared to agreeing samples < 0.01)
        assert uncertain is True

    def test_custom_threshold(self):
        """Test with custom variance threshold."""
        samples = [0.7, 0.8, 0.75]
        _, variance, uncertain = compute_variance_aware_confidence(
            samples, threshold=0.001
        )
        # With very low threshold, even small variance is uncertain
        assert uncertain is True


class TestRefuseToConverge:
    """Tests for refusal to over-converge logic."""

    def test_high_confidence_normal_stakes(self):
        """High confidence + normal stakes = no refusal."""
        should_refuse, reason = should_refuse_to_converge(0.9, stakes="normal")
        assert should_refuse is False
        assert reason is None

    def test_low_confidence_high_stakes(self):
        """Low confidence + high stakes = refuse."""
        should_refuse, reason = should_refuse_to_converge(0.5, stakes="high")
        assert should_refuse is True
        assert "Low confidence" in reason

    def test_very_low_confidence_any_stakes(self):
        """Very low confidence = always refuse."""
        should_refuse, reason = should_refuse_to_converge(0.25, stakes="low")
        assert should_refuse is True
        assert "Insufficient" in reason

    def test_moderate_confidence_normal_stakes(self):
        """Moderate confidence + normal stakes = no refusal."""
        should_refuse, reason = should_refuse_to_converge(0.6, stakes="normal")
        assert should_refuse is False

    def test_moderate_confidence_high_stakes(self):
        """Moderate confidence + high stakes = refuse."""
        should_refuse, reason = should_refuse_to_converge(0.6, stakes="high")
        assert should_refuse is True
