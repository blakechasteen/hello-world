"""
Bayesian Expansion Bundle - Variational Inference
=================================================

Bayesian uncertainty quantification for neural policy.

Install: pip install hololoom[bayesian]

Usage:
    from HoloLoom.config import Config
    from HoloLoom.expansions.bayesian import BayesianConfig

    config = Config.research()
    config.load_expansion(BayesianConfig(use_bayesian=True))

Date: December 2025
"""

from dataclasses import dataclass
from typing import Any, Dict

try:
    from HoloLoom.config_v2 import ExpansionBundle
except ImportError:
    class ExpansionBundle:
        def get_settings(self) -> Dict[str, Any]:
            raise NotImplementedError


@dataclass
class BayesianConfig(ExpansionBundle):
    """
    Bayesian Expansion Bundle: Variational Inference.

    Adds uncertainty quantification to the neural policy using:
    - Monte Carlo sampling for predictions
    - KL divergence regularization (ELBO)
    - Bayesian weight priors

    Trade-off: 10x inference overhead for uncertainty estimates.
    """

    # =========================================================================
    # BAYESIAN POLICY SETTINGS (4 fields)
    # =========================================================================

    use_bayesian: bool = False
    """Enable Bayesian uncertainty quantification."""

    bayesian_samples: int = 10
    """MC samples for uncertainty estimation (10x overhead)."""

    bayesian_kl_weight: float = 1.0
    """KL divergence weight in ELBO loss."""

    bayesian_prior_std: float = 1.0
    """Prior weight standard deviation (regularization strength)."""

    def get_settings(self) -> Dict[str, Any]:
        """Return dictionary of settings to merge into config."""
        return {
            "use_bayesian": self.use_bayesian,
            "bayesian_samples": self.bayesian_samples,
            "bayesian_kl_weight": self.bayesian_kl_weight,
            "bayesian_prior_std": self.bayesian_prior_std,
        }


# Convenience presets
def bayesian_default() -> BayesianConfig:
    """Bayesian policy with default settings."""
    return BayesianConfig(
        use_bayesian=True,
        bayesian_samples=10,
        bayesian_kl_weight=1.0,
        bayesian_prior_std=1.0,
    )


def bayesian_high_confidence() -> BayesianConfig:
    """Bayesian policy with more MC samples (higher confidence bounds)."""
    return BayesianConfig(
        use_bayesian=True,
        bayesian_samples=50,
        bayesian_kl_weight=1.0,
        bayesian_prior_std=1.0,
    )


def bayesian_regularized() -> BayesianConfig:
    """Bayesian policy with stronger regularization."""
    return BayesianConfig(
        use_bayesian=True,
        bayesian_samples=10,
        bayesian_kl_weight=2.0,  # Stronger KL penalty
        bayesian_prior_std=0.5,  # Tighter prior
    )


__all__ = [
    "BayesianConfig",
    "bayesian_default",
    "bayesian_high_confidence",
    "bayesian_regularized",
]
