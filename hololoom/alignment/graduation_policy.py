"""
GraduationPolicy — decides when a shadow detector is ready to become primary.

Mirrors the SHADOW→FULL pattern from adaptive_updater.py, simplified to:
SHADOW (logging only) → READY (criteria met) → PROMOTED (shadow becomes primary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from .agreement_tracker import AgreementStats

logger = logging.getLogger(__name__)


class GraduationVerdict(Enum):
    """Result of graduation evaluation."""
    HOLD = "hold"          # Not enough data or criteria not met
    READY = "ready"        # Criteria met, eligible for promotion
    PROMOTE = "promote"    # Auto-graduate (if auto_graduate=True and READY)


@dataclass
class GraduationCriteria:
    """Configurable thresholds for graduation."""
    min_samples: int = 500
    agreement_threshold: float = 0.85     # minimum agreement rate
    max_mean_delta: float = 0.15          # maximum tolerable mean score delta
    max_stddev_delta: float = 0.20        # maximum score delta variance
    auto_graduate: bool = False           # promote automatically when READY?


class GraduationPolicy:
    """Evaluates whether a shadow detector should be promoted to primary.

    Usage:
        policy = GraduationPolicy(GraduationCriteria(min_samples=500))
        verdict = policy.evaluate(tracker.stats())
        if verdict == GraduationVerdict.PROMOTE:
            registry.promote_shadow("deception")
    """

    def __init__(self, criteria: GraduationCriteria) -> None:
        self.criteria = criteria

    def evaluate(self, stats: AgreementStats | None) -> GraduationVerdict:
        """Evaluate graduation readiness from agreement statistics."""
        if stats is None:
            logger.debug("GraduationPolicy: no stats available → HOLD")
            return GraduationVerdict.HOLD

        # Gate: minimum sample count
        if stats.total_samples < self.criteria.min_samples:
            logger.debug(
                f"GraduationPolicy: {stats.total_samples}/{self.criteria.min_samples} "
                f"samples → HOLD"
            )
            return GraduationVerdict.HOLD

        # Gate: agreement rate
        if stats.agreement_rate < self.criteria.agreement_threshold:
            logger.debug(
                f"GraduationPolicy: agreement {stats.agreement_rate:.2%} "
                f"< {self.criteria.agreement_threshold:.2%} → HOLD"
            )
            return GraduationVerdict.HOLD

        # Gate: mean delta
        if stats.mean_delta > self.criteria.max_mean_delta:
            logger.debug(
                f"GraduationPolicy: mean_delta {stats.mean_delta:.3f} "
                f"> {self.criteria.max_mean_delta:.3f} → HOLD"
            )
            return GraduationVerdict.HOLD

        # Gate: stddev delta
        if stats.stddev_delta > self.criteria.max_stddev_delta:
            logger.debug(
                f"GraduationPolicy: stddev_delta {stats.stddev_delta:.3f} "
                f"> {self.criteria.max_stddev_delta:.3f} → HOLD"
            )
            return GraduationVerdict.HOLD

        # All criteria met
        if self.criteria.auto_graduate:
            logger.info("GraduationPolicy: all criteria met + auto_graduate → PROMOTE")
            return GraduationVerdict.PROMOTE

        logger.info("GraduationPolicy: all criteria met → READY")
        return GraduationVerdict.READY

    @classmethod
    def from_config(cls, config: dict) -> GraduationPolicy:
        """Build from a config dict (e.g., from YAML)."""
        return cls(GraduationCriteria(
            min_samples=config.get("min_samples", 500),
            agreement_threshold=config.get("agreement_threshold", 0.85),
            max_mean_delta=config.get("max_mean_delta", 0.15),
            max_stddev_delta=config.get("max_stddev_delta", 0.20),
            auto_graduate=config.get("auto_graduate", False),
        ))
