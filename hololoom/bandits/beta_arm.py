"""
BetaArm — Canonical Beta-distribution arm for Thompson Sampling.

Provides a single, composable building-block that every "copy-paste
Beta prior" in the codebase can delegate to.  Conforms to the
``DiscreteSampler`` protocol from ``ts_base.py`` at the single-arm level.

Created: 2026-03-12  (Governance Phase 4)
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BetaArm:
    """
    A single Beta-distribution arm.

    Tracks alpha (successes) and beta (failures) and exposes the four
    operations every Thompson Sampling consumer needs:
    ``sample``, ``expected_value``, ``update``, ``variance``.

    This is the **canonical** implementation — all simple Beta priors
    across the codebase should compose with this rather than
    re-implementing inline.

    Examples
    --------
    >>> arm = BetaArm()
    >>> arm.update(success=True)
    >>> arm.expected_value()  # 2 / 3
    0.6666666666666666
    """

    alpha: float = 1.0
    beta: float = 1.0

    # ── core operations ──────────────────────────────────────────

    def sample(self) -> float:
        """Draw a single sample from Beta(alpha, beta)."""
        return random.betavariate(self.alpha, self.beta)

    def expected_value(self) -> float:
        """E[X] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        """
        Update the arm after an observation.

        Args:
            success: True  -> alpha += weight
                     False -> beta  += weight
            weight:  Magnitude of the update (default 1.0).
        """
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    def variance(self) -> float:
        """Var[X] = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))."""
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))

    # ── persistence helpers ──────────────────────────────────────

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> BetaArm:
        return cls(alpha=d["alpha"], beta=d["beta"])

    # ── convenience ──────────────────────────────────────────────

    def total_observations(self) -> float:
        """Number of observations above the initial (1, 1) prior."""
        return self.alpha + self.beta - 2.0

    def uncertainty(self) -> float:
        """Inverse-total heuristic: 1 / (alpha + beta). Decreases with data."""
        return 1.0 / (self.alpha + self.beta)

    def confidence(self, saturation: float = 50.0) -> float:
        """
        Confidence in [0, 1], scaling with observations.

        Saturates around ``saturation`` observations above the prior.
        """
        obs = max(0.0, self.alpha + self.beta - 2.0)
        return min(1.0, obs / saturation)

    def __repr__(self) -> str:
        return (
            f"BetaArm(alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
            f"E[X]={self.expected_value():.3f})"
        )
