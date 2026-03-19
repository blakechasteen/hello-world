"""
Uncertainty as First-Class Output (Addendum A.1)

Implements explicit uncertainty representation for all HoloLoom outputs.
"The system must represent and act on uncertainty, not just hide it."

Three Uncertainty Primitives:
1. CONFIDENCE SIGNALING - Every output carries explicit confidence markers
2. VARIANCE AWARENESS - Multi-sample disagreement becomes a signal
3. REFUSAL TO OVER-CONVERGE - Low confidence + high stakes = forced verification

Confidence Tiers:
- DEFINITIVE (>=0.95): Proceed without caveat
- HIGH (0.80-0.95): Proceed with confidence
- MODERATE (0.60-0.80): Include hedging language
- LOW (0.40-0.60): Require verification
- INSUFFICIENT (<0.40): Refuse to answer definitively
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceTier(Enum):
    """Human-readable confidence categories."""
    DEFINITIVE = "definitive"      # >= 0.95
    HIGH = "high"                  # 0.80 - 0.95
    MODERATE = "moderate"          # 0.60 - 0.80
    LOW = "low"                    # 0.40 - 0.60
    INSUFFICIENT = "insufficient"  # < 0.40

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceTier":
        """Convert numeric score to confidence tier."""
        if score >= 0.95:
            return cls.DEFINITIVE
        elif score >= 0.80:
            return cls.HIGH
        elif score >= 0.60:
            return cls.MODERATE
        elif score >= 0.40:
            return cls.LOW
        else:
            return cls.INSUFFICIENT

    @property
    def requires_verification(self) -> bool:
        """Whether this tier requires additional verification."""
        return self in (ConfidenceTier.LOW, ConfidenceTier.INSUFFICIENT)

    @property
    def output_marking(self) -> str:
        """Human-readable output marking for this tier."""
        markings = {
            ConfidenceTier.DEFINITIVE: "✓ Verified",
            ConfidenceTier.HIGH: "→ Confident",
            ConfidenceTier.MODERATE: "⚠ Moderate confidence",
            ConfidenceTier.LOW: "⚠⚠ Low confidence - verify",
            ConfidenceTier.INSUFFICIENT: "✗ Cannot reliably answer",
        }
        return markings[self]


@dataclass
class UncertaintySource:
    """Attribution of uncertainty to a specific source."""
    type: str                    # e.g., "retrieval_sparsity", "source_disagreement"
    contribution: float          # How much this source contributes (0.0-1.0)
    details: str | None = None  # Human-readable explanation

    def __post_init__(self):
        if not 0.0 <= self.contribution <= 1.0:
            raise ValueError(f"contribution must be 0.0-1.0, got {self.contribution}")


@dataclass
class UncertaintyEnvelope:
    """
    First-class uncertainty representation for all HoloLoom outputs.

    Every Spacetime output includes this envelope, making uncertainty
    explicit and actionable by downstream consumers.

    Attributes:
        point_estimate: Best guess confidence (0.0-1.0)
        credible_interval: [lower, upper] bounds for confidence
        tier: Human-readable confidence category
        sources: Attribution of uncertainty to specific sources
        sample_variance: Variance across multi-sample responses (if applicable)
        requires_verification: Should downstream systems verify?

    Example:
        uncertainty = UncertaintyEnvelope(
            point_estimate=0.72,
            credible_interval=(0.65, 0.79),
            tier=ConfidenceTier.MODERATE,
            sources=[
                UncertaintySource(type="retrieval_sparsity", contribution=0.15),
                UncertaintySource(type="temporal_decay", contribution=0.08),
            ],
            sample_variance=0.12,
            requires_verification=True
        )
    """
    point_estimate: float
    credible_interval: tuple[float, float]
    tier: ConfidenceTier
    sources: list[UncertaintySource] = field(default_factory=list)
    sample_variance: float | None = None
    requires_verification: bool = False

    def __post_init__(self):
        # Validate point estimate
        if not 0.0 <= self.point_estimate <= 1.0:
            raise ValueError(f"point_estimate must be 0.0-1.0, got {self.point_estimate}")

        # Validate interval
        lower, upper = self.credible_interval
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError(f"Invalid credible_interval: {self.credible_interval}")

        # Auto-set requires_verification based on tier if not explicitly set
        if self.requires_verification is False and self.tier.requires_verification:
            object.__setattr__(self, 'requires_verification', True)

    @classmethod
    def from_confidence(
        cls,
        confidence: float,
        sources: list[UncertaintySource] | None = None,
        sample_variance: float | None = None,
    ) -> "UncertaintyEnvelope":
        """
        Create UncertaintyEnvelope from a simple confidence score.

        Automatically computes credible interval and tier.
        """
        tier = ConfidenceTier.from_score(confidence)

        # Compute credible interval based on variance or default
        if sample_variance is not None and sample_variance > 0:
            # Use variance for interval (approximately 95% CI)
            std_dev = math.sqrt(sample_variance)
            margin = 1.96 * std_dev
            lower = max(0.0, confidence - margin)
            upper = min(1.0, confidence + margin)
        else:
            # Default interval based on tier
            margin = {
                ConfidenceTier.DEFINITIVE: 0.02,
                ConfidenceTier.HIGH: 0.05,
                ConfidenceTier.MODERATE: 0.10,
                ConfidenceTier.LOW: 0.15,
                ConfidenceTier.INSUFFICIENT: 0.20,
            }.get(tier, 0.10)
            lower = max(0.0, confidence - margin)
            upper = min(1.0, confidence + margin)

        return cls(
            point_estimate=confidence,
            credible_interval=(lower, upper),
            tier=tier,
            sources=sources or [],
            sample_variance=sample_variance,
            requires_verification=tier.requires_verification,
        )

    @property
    def interval_width(self) -> float:
        """Width of the credible interval (measure of uncertainty)."""
        return self.credible_interval[1] - self.credible_interval[0]

    @property
    def is_high_stakes_uncertain(self) -> bool:
        """
        Check if this represents a high-stakes uncertain situation.
        Low confidence + high stakes = forced verification.
        """
        return self.point_estimate < 0.7 and self.requires_verification

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "point_estimate": self.point_estimate,
            "credible_interval": list(self.credible_interval),
            "tier": self.tier.value,
            "tier_marking": self.tier.output_marking,
            "sources": [
                {"type": s.type, "contribution": s.contribution, "details": s.details}
                for s in self.sources
            ],
            "sample_variance": self.sample_variance,
            "requires_verification": self.requires_verification,
            "interval_width": self.interval_width,
        }

    def __str__(self) -> str:
        return (
            f"UncertaintyEnvelope("
            f"confidence={self.point_estimate:.2f} [{self.credible_interval[0]:.2f}, {self.credible_interval[1]:.2f}], "
            f"tier={self.tier.value}, "
            f"verification_required={self.requires_verification})"
        )


def compute_variance_aware_confidence(
    samples: list[float],
    threshold: float = 0.15,
) -> tuple[float, float, bool]:
    """
    Compute variance-aware confidence from multiple samples.

    When multiple self-consistent samples disagree materially,
    that disagreement itself becomes a signal.

    Args:
        samples: List of confidence scores from multiple samples
        threshold: Variance threshold above which to flag as uncertain

    Returns:
        Tuple of (mean_confidence, variance, is_uncertain)
    """
    if not samples:
        return 0.0, 0.0, True

    if len(samples) == 1:
        return samples[0], 0.0, False

    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / len(samples)

    # High variance indicates material disagreement
    is_uncertain = variance > threshold

    return mean, variance, is_uncertain


def should_refuse_to_converge(
    confidence: float,
    stakes: str = "normal",
    confidence_threshold: float = 0.7,
) -> tuple[bool, str | None]:
    """
    Check if the system should refuse to over-converge.

    Low confidence + high stakes = forced verification.

    Args:
        confidence: Current confidence level
        stakes: "low", "normal", or "high"
        confidence_threshold: Threshold below which to require verification

    Returns:
        Tuple of (should_require_verification, reason)
    """
    if stakes == "high" and confidence < confidence_threshold:
        return True, f"Low confidence ({confidence:.2f}) on high-stakes query"

    if stakes == "normal" and confidence < 0.5:
        return True, f"Very low confidence ({confidence:.2f}) requires verification"

    if confidence < 0.3:
        return True, f"Insufficient confidence ({confidence:.2f}) to answer definitively"

    return False, None
