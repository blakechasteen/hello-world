"""
Base protocols and types for unified Thompson Sampling.

All TS models (discrete, linear, neural, GP) implement the ThompsonSampler
protocol, enabling seamless swapping between algorithms.
"""

from dataclasses import dataclass, field
from typing import Protocol, Any, Literal
import numpy as np
import numpy.typing as npt


@dataclass
class ThompsonSamplerConfig:
    """
    Base configuration for Thompson Samplers.

    Each sampler type has specific required fields:
    - Discrete: n_arms, alpha_prior, beta_prior
    - Linear: context_dim, n_actions, lambda_prior, sigma_noise
    - Neural: context_dim, n_actions, hidden_dims, backend, ...
    - GP: param_dim, kernel_type, length_scale, noise_std
    """
    sampler_type: Literal["discrete", "linear", "neural", "gp", "deep_kernel_gp"]
    metadata: dict[str, Any] = field(default_factory=dict)


class ThompsonSampler(Protocol):
    """
    Unified protocol for Thompson Sampling algorithms.

    All TS implementations must provide:
    - select(): Sample from posterior and return action/parameters
    - update(): Update posterior from observation
    - get_statistics(): Return diagnostics

    Different samplers have different signatures based on their domain:
    - Discrete: select() → arm_id (int), update(arm, reward)
    - Linear: select(context) → action_id, update(context, action, reward)
    - Neural: select(context, actions) → action, update(context, action, reward)
    - GP: select() → params (continuous), update(params, reward)
    """

    def select(self, *args, **kwargs) -> Any:
        """
        Sample action/parameters from posterior.

        Returns vary by sampler type:
        - Discrete: int (arm index)
        - Linear: int (action index) given context
        - Neural: Action object given context and action list
        - GP: ndarray (continuous parameters)
        """
        ...

    def update(self, *args, **kwargs) -> None:
        """
        Update posterior from observation.

        Args vary by sampler type:
        - Discrete: update(arm_id, reward)
        - Linear: update(context, action_id, reward)
        - Neural: update(observation)
        - GP: update(params, reward)
        """
        ...

    def get_statistics(self) -> dict[str, Any]:
        """
        Get sampler statistics and diagnostics.

        Returns:
            Dict with sampler-specific metrics (pulls, rewards, posterior params, etc.)
        """
        ...


@dataclass
class Observation:
    """
    Generic observation for TS updates.

    Flexible structure that works across sampler types:
    - Discrete: arm_id, reward
    - Linear/Neural: context, action_id, reward
    - GP: params, reward
    """
    reward: float
    arm_id: int | None = None
    action_id: int | None = None
    context: npt.NDArray[np.floating] | None = None
    params: npt.NDArray[np.floating] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Discrete Sampler Protocol
# ============================================================================


class DiscreteSampler(Protocol):
    """
    Protocol for discrete multi-armed bandits.

    Simple interface: select arm, observe reward, update.
    """

    def select(self) -> int:
        """Select arm index."""
        ...

    def update(self, arm: int, reward: float) -> None:
        """Update posterior for arm."""
        ...

    def get_posterior_params(self) -> dict[int, tuple[float, float]]:
        """Get posterior parameters (alpha, beta) for each arm."""
        ...


# ============================================================================
# Contextual Sampler Protocol
# ============================================================================


class ContextualSampler(Protocol):
    """
    Protocol for contextual bandits (linear, neural).

    Requires context for selection.
    """

    def select(self, context: npt.NDArray[np.floating]) -> int:
        """Select action given context."""
        ...

    def update(
        self,
        context: npt.NDArray[np.floating],
        action: int,
        reward: float,
    ) -> None:
        """Update posterior from (context, action, reward)."""
        ...


# ============================================================================
# Continuous Sampler Protocol
# ============================================================================


class ContinuousSampler(Protocol):
    """
    Protocol for continuous optimization (GP-TS).

    Samples continuous parameter vectors.
    """

    def select(self) -> npt.NDArray[np.floating]:
        """Sample parameter vector from acquisition function."""
        ...

    def update(self, params: npt.NDArray[np.floating], reward: float) -> None:
        """Update GP posterior."""
        ...

    def get_gp_params(self) -> dict[str, Any]:
        """Get GP hyperparameters and posterior."""
        ...
