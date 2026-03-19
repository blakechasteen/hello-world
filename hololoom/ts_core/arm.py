"""
Shared BanditArm — Beta-distributed arm for Thompson Sampling.

Used by:
- context/bandit.py (ThompsonBandit — named string arms)
- apps/server/model_router.py (ModelBandit — intent x model arms)

All share the same Beta(alpha, beta) update and sampling mechanics.
"""

import logging
import math
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class BanditArm:
    """
    Single arm in a Thompson Sampling bandit.

    Uses Beta distribution: Beta(alpha, beta) where alpha = successes + prior,
    beta = failures + prior. Sample from the distribution to balance
    exploration and exploitation.
    """
    name: str = ""
    alpha: float = 1.0   # Success parameter (prior = 1.0)
    beta: float = 1.0    # Failure parameter (prior = 1.0)
    total_pulls: int = 0
    total_successes: int = 0
    total_failures: int = 0

    @property
    def mean(self) -> float:
        """Expected reward: mean of Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Draw from Beta(alpha, beta)."""
        if np is not None:
            return float(np.random.beta(self.alpha, self.beta))
        # stdlib fallback
        import random
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        """
        Update arm with observation.

        Args:
            success: Whether the outcome was successful
            weight: Optional weight for the update (default 1.0)
        """
        self.total_pulls += 1
        if success:
            self.alpha += weight
            self.total_successes += 1
        else:
            self.beta += weight
            self.total_failures += 1

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        Calculate confidence interval via Monte Carlo sampling.

        Args:
            confidence: Confidence level (default: 0.95)

        Returns:
            (lower, upper) bounds
        """
        if np is not None:
            samples = np.random.beta(self.alpha, self.beta, 10000)
            lower = float(np.percentile(samples, (1 - confidence) * 100 / 2))
            upper = float(np.percentile(samples, 100 - (1 - confidence) * 100 / 2))
            return lower, upper
        # Rough approximation without numpy
        mean = self.mean
        std = math.sqrt(
            (self.alpha * self.beta)
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )
        z = 1.96 if confidence >= 0.95 else 1.645
        return max(0.0, mean - z * std), min(1.0, mean + z * std)

    def ci_width(self, confidence: float = 0.95) -> float:
        """Confidence interval width (measure of uncertainty)."""
        lower, upper = self.confidence_interval(confidence)
        return upper - lower

    def geodesic_certainty(self) -> float:
        """
        Fisher-Rao distance from uniform prior Beta(1,1) to current posterior.

        Measures accumulated evidence in the intrinsic geometry of the Beta
        manifold. Returns 0.0 for fresh arms, grows with evidence.
        Falls back to log(alpha + beta) if information geometry unavailable.
        """
        if self.alpha == 1.0 and self.beta == 1.0:
            return 0.0

        try:
            from hololoom.core.warp.math.geometry.information_geometry import (
                StatisticalManifold,
            )
            if np is None:
                raise ImportError("numpy required for geodesic")
            mfld = StatisticalManifold.beta_manifold()
            prior = np.array([1.0, 1.0])
            current = np.array([self.alpha, self.beta])
            return mfld.geodesic_distance(prior, current, n_steps=50)
        except Exception:
            # Fallback: log of total observations (crude certainty proxy)
            return math.log(self.alpha + self.beta)
