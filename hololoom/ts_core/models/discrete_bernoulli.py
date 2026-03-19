"""
Discrete Bernoulli Thompson Sampling (Beta-Bernoulli MAB).

Classic multi-armed bandit with Beta priors and Bernoulli likelihoods.
Closed-form posterior updates - no neural networks needed.

Use cases:
- A/B testing (2 arms)
- Agent routing (discrete choices)
- Simple tool selection (no context)
- Baseline for contextual methods

Example:
    >>> sampler = DiscreteBernoulliTS(n_arms=5)
    >>> for t in range(1000):
    ...     arm = sampler.select()
    ...     reward = execute_arm(arm)
    ...     sampler.update(arm, reward)
    >>> stats = sampler.get_statistics()
    >>> print(stats["best_arm"])  # Arm with highest posterior mean
"""

from typing import Literal

import numpy as np


class DiscreteBernoulliTS:
    """
    Beta-Bernoulli Thompson Sampling for multi-armed bandits.

    **Model**:
    - Prior: Beta(α, β) for each arm
    - Likelihood: Bernoulli(θ) where θ ~ Beta(α, β)
    - Posterior: Beta(α + successes, β + failures)

    **Thompson Sampling**:
    1. Sample θ_a ~ Beta(α_a, β_a) for each arm a
    2. Select arm a* = argmax_a θ_a
    3. Observe reward r ∈ {0, 1}
    4. Update: α_a* ← α_a* + r, β_a* ← β_a* + (1 - r)

    Attributes:
        n_arms: Number of arms
        alpha: Success counts + prior (shape: [n_arms])
        beta: Failure counts + prior (shape: [n_arms])
        pulls: Pull counts per arm (shape: [n_arms])
        rewards: Total rewards per arm (shape: [n_arms])

    Example:
        >>> # A/B test
        >>> sampler = DiscreteBernoulliTS(n_arms=2, alpha_prior=1.0, beta_prior=1.0)
        >>>
        >>> # Select arm
        >>> arm = sampler.select()  # 0 or 1
        >>>
        >>> # Observe outcome
        >>> success = user_clicked()  # 0 or 1
        >>> sampler.update(arm, success)
        >>>
        >>> # Get statistics
        >>> stats = sampler.get_statistics()
        >>> print(f"Best arm: {stats['best_arm']}")
        >>> print(f"Posterior means: {stats['posterior_means']}")
    """

    def __init__(
        self,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize Beta-Bernoulli Thompson Sampling.

        Args:
            n_arms: Number of arms (actions)
            alpha_prior: Prior success count (uniform: 1.0, optimistic: 10.0)
            beta_prior: Prior failure count (uniform: 1.0, pessimistic: 0.1)
            seed: Random seed for reproducibility

        Raises:
            ValueError: If n_arms <= 0 or priors <= 0

        Example:
            >>> # Uniform prior (no bias)
            >>> sampler = DiscreteBernoulliTS(n_arms=5, alpha_prior=1.0, beta_prior=1.0)
            >>>
            >>> # Optimistic prior (encourages exploration)
            >>> sampler = DiscreteBernoulliTS(n_arms=5, alpha_prior=10.0, beta_prior=1.0)
        """
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError(f"Priors must be positive, got alpha={alpha_prior}, beta={beta_prior}")

        self.n_arms = n_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # Posterior parameters (Beta distribution)
        self.alpha = np.full(n_arms, alpha_prior, dtype=float)
        self.beta = np.full(n_arms, beta_prior, dtype=float)

        # Statistics
        self.pulls = np.zeros(n_arms, dtype=int)
        self.rewards = np.zeros(n_arms, dtype=float)

        # RNG
        self.rng = np.random.RandomState(seed)

    def select(self, return_diagnostics: bool = False) -> int | tuple[int, dict]:
        """
        Sample arm using Thompson Sampling.

        Algorithm:
            1. For each arm a: sample θ_a ~ Beta(α_a, β_a)
            2. Select arm a* = argmax_a θ_a

        Args:
            return_diagnostics: If True, return (arm, diagnostics)

        Returns:
            Selected arm index, or (arm, diagnostics) if return_diagnostics=True

        Example:
            >>> arm = sampler.select()
            >>> print(arm)  # 0, 1, 2, ..., n_arms-1
            >>>
            >>> # With diagnostics
            >>> arm, diag = sampler.select(return_diagnostics=True)
            >>> print(diag["sampled_thetas"])  # Sampled success probabilities
            >>> print(diag["posterior_means"])  # E[θ_a] for each arm
        """
        # Sample θ_a ~ Beta(α_a, β_a) for all arms
        sampled_thetas = self.rng.beta(self.alpha, self.beta)

        # Greedy w.r.t. sampled parameters
        selected_arm = int(sampled_thetas.argmax())

        if return_diagnostics:
            diagnostics = {
                "sampled_thetas": sampled_thetas,
                "posterior_means": self.alpha / (self.alpha + self.beta),
                "posterior_vars": self._posterior_variance(),
                "selected_arm": selected_arm,
            }
            return selected_arm, diagnostics
        else:
            return selected_arm

    def update(self, arm: int, reward: float) -> None:
        """
        Update posterior from observation.

        Bayesian update:
            α_arm ← α_arm + reward
            β_arm ← β_arm + (1 - reward)

        Args:
            arm: Arm index (0 to n_arms-1)
            reward: Observed reward (typically 0 or 1, but continuous rewards ok)

        Raises:
            ValueError: If arm index out of bounds

        Example:
            >>> sampler.update(arm=2, reward=1.0)  # Success on arm 2
            >>> sampler.update(arm=0, reward=0.0)  # Failure on arm 0
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"Arm {arm} out of bounds [0, {self.n_arms})")

        # Bayesian update
        self.alpha[arm] += reward
        self.beta[arm] += (1.0 - reward)

        # Statistics
        self.pulls[arm] += 1
        self.rewards[arm] += reward

    def get_statistics(self) -> dict:
        """
        Get sampler statistics and diagnostics.

        Returns:
            Dict with:
                - posterior_means: E[θ_a] for each arm
                - posterior_vars: Var[θ_a] for each arm
                - pulls: Pull counts per arm
                - empirical_means: Sample mean rewards
                - best_arm: Arm with highest posterior mean
                - regret_proxy: Expected regret vs. best arm

        Example:
            >>> stats = sampler.get_statistics()
            >>> print(f"Best arm: {stats['best_arm']}")
            >>> print(f"Pulls: {stats['pulls']}")
            >>> print(f"Posterior means: {stats['posterior_means']}")
        """
        posterior_means = self.alpha / (self.alpha + self.beta)
        posterior_vars = self._posterior_variance()

        # Empirical means (with Laplace smoothing)
        empirical_means = np.where(
            self.pulls > 0,
            self.rewards / self.pulls,
            0.5  # Default for unpulled arms
        )

        best_arm = int(posterior_means.argmax())
        regret_proxy = posterior_means[best_arm] - posterior_means.mean()

        return {
            "n_arms": self.n_arms,
            "total_pulls": int(self.pulls.sum()),
            "posterior_means": posterior_means,
            "posterior_vars": posterior_vars,
            "pulls": self.pulls,
            "empirical_means": empirical_means,
            "best_arm": best_arm,
            "regret_proxy": float(regret_proxy),
        }

    def get_posterior_params(self) -> dict[int, tuple[float, float]]:
        """
        Get posterior parameters (α, β) for each arm.

        Returns:
            Dict mapping arm_id → (alpha, beta)

        Example:
            >>> params = sampler.get_posterior_params()
            >>> print(params[0])  # (alpha_0, beta_0)
        """
        return {
            arm: (float(self.alpha[arm]), float(self.beta[arm]))
            for arm in range(self.n_arms)
        }

    def _posterior_variance(self) -> np.ndarray:
        """Compute Beta posterior variance: αβ / ((α+β)²(α+β+1))"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def reset(self, arm: int | None = None) -> None:
        """
        Reset posterior to prior.

        Args:
            arm: If specified, reset only this arm. Otherwise reset all.

        Example:
            >>> sampler.reset()  # Reset all arms
            >>> sampler.reset(arm=2)  # Reset only arm 2
        """
        if arm is None:
            # Reset all
            self.alpha[:] = self.alpha_prior
            self.beta[:] = self.beta_prior
            self.pulls[:] = 0
            self.rewards[:] = 0
        else:
            # Reset single arm
            if not (0 <= arm < self.n_arms):
                raise ValueError(f"Arm {arm} out of bounds")
            self.alpha[arm] = self.alpha_prior
            self.beta[arm] = self.beta_prior
            self.pulls[arm] = 0
            self.rewards[arm] = 0

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"DiscreteBernoulliTS(n_arms={self.n_arms}, "
            f"pulls={stats['total_pulls']}, "
            f"best_arm={stats['best_arm']})"
        )


# ============================================================================
# Factory functions
# ============================================================================


def create_discrete_ts(
    n_arms: int,
    prior: Literal["uniform", "optimistic", "pessimistic"] = "uniform",
    seed: int | None = None,
) -> DiscreteBernoulliTS:
    """
    Factory for discrete Thompson Sampling with preset priors.

    Args:
        n_arms: Number of arms
        prior: Prior type:
            - "uniform": α=1, β=1 (no bias)
            - "optimistic": α=10, β=1 (encourages exploration)
            - "pessimistic": α=1, β=10 (conservative)
        seed: Random seed

    Returns:
        Initialized DiscreteBernoulliTS

    Example:
        >>> # A/B test with uniform prior
        >>> sampler = create_discrete_ts(n_arms=2, prior="uniform")
        >>>
        >>> # Multi-arm bandit with optimistic exploration
        >>> sampler = create_discrete_ts(n_arms=10, prior="optimistic")
    """
    prior_params = {
        "uniform": (1.0, 1.0),
        "optimistic": (10.0, 1.0),
        "pessimistic": (1.0, 10.0),
    }

    alpha_prior, beta_prior = prior_params[prior]

    return DiscreteBernoulliTS(
        n_arms=n_arms,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        seed=seed,
    )
