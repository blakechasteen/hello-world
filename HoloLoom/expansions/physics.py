"""
Physics Expansion Bundle - GP Bandits + PDE Semantic Flow
=========================================================

Gaussian Process Bandits for continuous action spaces and
PDE-based semantic flow for temporal dynamics.

Install: pip install hololoom[physics]

Usage:
    from HoloLoom.config import Config
    from HoloLoom.expansions.physics import PhysicsConfig

    config = Config.research()
    config.load_expansion(PhysicsConfig(use_gp_bandits=True))

Date: December 2025
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    from HoloLoom.config import ExpansionBundle
except ImportError:
    # Fallback for testing
    class ExpansionBundle:
        def get_settings(self) -> Dict[str, Any]:
            raise NotImplementedError


@dataclass
class PhysicsConfig(ExpansionBundle):
    """
    Physics Expansion Bundle: GP Bandits + PDE Semantic Flow.

    GP Bandits (12 fields):
    - Gaussian Process regression for continuous action spaces
    - Thompson Sampling or UCB acquisition functions
    - Matern/RBF kernel options
    - Adaptive exploration parameters

    PDE Semantic Flow (7 fields):
    - Heat equation (diffusion)
    - Wave equation (oscillation)
    - Reaction-diffusion (competition)
    - Hamilton-Jacobi (optimal paths)
    """

    # =========================================================================
    # GP BANDIT SETTINGS (12 fields)
    # =========================================================================

    use_gp_bandits: bool = False
    """Enable Gaussian Process bandits for continuous action spaces."""

    gp_acquisition: str = "thompson"
    """GP acquisition function: 'thompson' or 'ucb'."""

    gp_kernel_type: str = "matern"
    """GP kernel type: 'matern' or 'rbf'."""

    gp_kernel_length_scale: float = 0.3
    """GP kernel length scale (smoothness)."""

    gp_kernel_variance: float = 1.0
    """GP kernel variance (amplitude)."""

    gp_matern_nu: float = 2.5
    """Matern kernel smoothness parameter (1.5, 2.5, or 5.0)."""

    gp_noise_variance: float = 0.01
    """GP observation noise variance."""

    gp_ucb_beta: float = 2.0
    """UCB exploration parameter (higher = more exploration)."""

    gp_ucb_adaptive_beta: bool = True
    """Use adaptive beta = sqrt(2 * log(t))."""

    gp_n_candidates_per_dim: int = 5
    """Discretization resolution per dimension."""

    gp_update_interval: int = 10
    """Retrain GP every N observations."""

    gp_warmup_samples: int = 10
    """Initial random samples before GP optimization."""

    # =========================================================================
    # PDE SEMANTIC FLOW SETTINGS (7 fields)
    # =========================================================================

    use_semantic_flow: bool = False
    """Enable PDE-based temporal evolution (research mode only - expensive!)."""

    pde_type: str = "heat"
    """
    PDE type for semantic evolution:
    - 'heat': Diffusion (smoothing, spreading activation)
    - 'wave': Oscillation (resonance, interference)
    - 'reaction_diffusion': Competition (concepts compete for attention)
    - 'hamilton_jacobi': Optimal paths (shortest semantic distances)
    """

    flow_dt: float = 0.01
    """PDE timestep (smaller = more accurate, more expensive)."""

    flow_steps: int = 10
    """Evolution steps between queries."""

    flow_reaction_type: str = "competitive"
    """Reaction type: 'logistic', 'competitive', or 'cubic'."""

    flow_diffusion_coef: float = 1.0
    """Diffusion coefficient for reaction-diffusion."""

    flow_wave_speed: float = 1.0
    """Wave speed for wave equation."""

    def get_settings(self) -> Dict[str, Any]:
        """Return dictionary of settings to merge into config."""
        return {
            # GP Bandits
            "use_gp_bandits": self.use_gp_bandits,
            "gp_acquisition": self.gp_acquisition,
            "gp_kernel_type": self.gp_kernel_type,
            "gp_kernel_length_scale": self.gp_kernel_length_scale,
            "gp_kernel_variance": self.gp_kernel_variance,
            "gp_matern_nu": self.gp_matern_nu,
            "gp_noise_variance": self.gp_noise_variance,
            "gp_ucb_beta": self.gp_ucb_beta,
            "gp_ucb_adaptive_beta": self.gp_ucb_adaptive_beta,
            "gp_n_candidates_per_dim": self.gp_n_candidates_per_dim,
            "gp_update_interval": self.gp_update_interval,
            "gp_warmup_samples": self.gp_warmup_samples,

            # PDE Semantic Flow
            "use_semantic_flow": self.use_semantic_flow,
            "pde_type": self.pde_type,
            "flow_dt": self.flow_dt,
            "flow_steps": self.flow_steps,
            "flow_reaction_type": self.flow_reaction_type,
            "flow_diffusion_coef": self.flow_diffusion_coef,
            "flow_wave_speed": self.flow_wave_speed,
        }


# Convenience presets
def gp_thompson() -> PhysicsConfig:
    """GP Bandits with Thompson Sampling (recommended)."""
    return PhysicsConfig(
        use_gp_bandits=True,
        gp_acquisition="thompson",
        gp_kernel_type="matern",
        gp_matern_nu=2.5,
    )


def gp_ucb() -> PhysicsConfig:
    """GP Bandits with UCB acquisition."""
    return PhysicsConfig(
        use_gp_bandits=True,
        gp_acquisition="ucb",
        gp_ucb_beta=2.0,
        gp_ucb_adaptive_beta=True,
    )


def semantic_heat_diffusion() -> PhysicsConfig:
    """Semantic flow with heat diffusion (smoothing)."""
    return PhysicsConfig(
        use_semantic_flow=True,
        pde_type="heat",
        flow_steps=10,
    )


def semantic_wave() -> PhysicsConfig:
    """Semantic flow with wave dynamics (resonance)."""
    return PhysicsConfig(
        use_semantic_flow=True,
        pde_type="wave",
        flow_wave_speed=1.0,
        flow_steps=20,
    )


def semantic_competition() -> PhysicsConfig:
    """Semantic flow with reaction-diffusion (competition)."""
    return PhysicsConfig(
        use_semantic_flow=True,
        pde_type="reaction_diffusion",
        flow_reaction_type="competitive",
        flow_diffusion_coef=1.0,
        flow_steps=15,
    )


__all__ = [
    "PhysicsConfig",
    "gp_thompson",
    "gp_ucb",
    "semantic_heat_diffusion",
    "semantic_wave",
    "semantic_competition",
]
