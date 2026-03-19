"""
Bayesian Linear Thompson Sampling.

Contextual bandit with linear reward model and Gaussian prior.
Closed-form posterior updates (conjugate Gaussian-Gaussian).

**Model**:
    r(x, a) = x^T θ_a + ε,  ε ~ N(0, σ²)
    θ_a ~ N(μ_a, Σ_a)

**Posterior** (Gaussian):
    After observing (x, a, r):
    Σ_a^{-1} ← Σ_a^{-1} + (1/σ²) x x^T
    μ_a ← Σ_a (Σ_a^{-1} μ_a + (r/σ²) x)

Use cases:
- Linear contextual bandits (news articles, ads)
- Interpretable models (coefficients have meaning)
- Baseline for neural methods
- Fast inference (no neural network)

Example:
    >>> sampler = BayesianLinearTS(context_dim=10, n_actions=5)
    >>> for query in queries:
    ...     action = sampler.select(query.context)
    ...     reward = execute(action, query)
    ...     sampler.update(query.context, action, reward)
"""

from typing import Literal

import numpy as np
import numpy.typing as npt


class BayesianLinearTS:
    """
    Bayesian Linear Thompson Sampling for contextual bandits.

    **Algorithm** (Thompson Sampling for Linear Contextual Bandits):
    1. For each action a: sample θ_a ~ N(μ_a, Σ_a)
    2. Predict rewards: r_a = x^T θ_a
    3. Select action: a* = argmax_a r_a
    4. Observe reward r
    5. Update posterior for a* (Gaussian conjugate update)

    Attributes:
        context_dim: Context feature dimension
        n_actions: Number of actions
        mu: Posterior means (shape: [n_actions, context_dim])
        Sigma: Posterior covariances (shape: [n_actions, context_dim, context_dim])
        Sigma_inv: Inverse covariances (for efficient updates)
        pulls: Pull counts per action

    Example:
        >>> # News article recommendation
        >>> sampler = BayesianLinearTS(context_dim=50, n_actions=10)
        >>>
        >>> # User arrives with context (interests, demographics)
        >>> context = user.get_feature_vector()  # [50]
        >>>
        >>> # Select article
        >>> article = sampler.select(context)  # 0-9
        >>>
        >>> # User clicks or not
        >>> reward = 1.0 if user.clicked() else 0.0
        >>> sampler.update(context, article, reward)
    """

    def __init__(
        self,
        context_dim: int,
        n_actions: int,
        lambda_prior: float = 1.0,
        sigma_noise: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize Bayesian Linear Thompson Sampling.

        Args:
            context_dim: Context feature dimension
            n_actions: Number of actions
            lambda_prior: Prior precision (larger = tighter prior, less exploration)
            sigma_noise: Observation noise std dev
            seed: Random seed

        Raises:
            ValueError: If dimensions invalid or hyperparameters non-positive

        Example:
            >>> # Default prior (λ=1, σ=1)
            >>> sampler = BayesianLinearTS(context_dim=20, n_actions=5)
            >>>
            >>> # Tight prior (less exploration)
            >>> sampler = BayesianLinearTS(
            ...     context_dim=20,
            ...     n_actions=5,
            ...     lambda_prior=10.0,  # Stronger prior
            ...     sigma_noise=0.5     # Lower noise assumption
            ... )
        """
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        if lambda_prior <= 0:
            raise ValueError(f"lambda_prior must be positive, got {lambda_prior}")
        if sigma_noise <= 0:
            raise ValueError(f"sigma_noise must be positive, got {sigma_noise}")

        self.context_dim = context_dim
        self.n_actions = n_actions
        self.lambda_prior = lambda_prior
        self.sigma_noise = sigma_noise

        # Prior: θ_a ~ N(0, (1/λ) I)
        self.mu = np.zeros((n_actions, context_dim))
        self.Sigma = np.array([
            np.eye(context_dim) / lambda_prior
            for _ in range(n_actions)
        ])
        self.Sigma_inv = np.array([
            np.eye(context_dim) * lambda_prior
            for _ in range(n_actions)
        ])

        # Statistics
        self.pulls = np.zeros(n_actions, dtype=int)
        self.total_reward = np.zeros(n_actions)

        # RNG
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        context: npt.NDArray[np.floating],
        return_diagnostics: bool = False,
    ) -> int | tuple[int, dict]:
        """
        Select action using Thompson Sampling.

        Algorithm:
            1. For each action a: sample θ_a ~ N(μ_a, Σ_a)
            2. Compute predicted rewards: r_a = x^T θ_a
            3. Select action: a* = argmax_a r_a

        Args:
            context: Context vector, shape [context_dim]
            return_diagnostics: If True, return (action, diagnostics)

        Returns:
            Selected action index, or (action, diagnostics)

        Raises:
            ValueError: If context dimension doesn't match

        Example:
            >>> context = np.random.randn(20)
            >>> action = sampler.select(context)
            >>> print(action)  # 0-4
            >>>
            >>> # With diagnostics
            >>> action, diag = sampler.select(context, return_diagnostics=True)
            >>> print(diag["predicted_rewards"])  # Sampled predictions
            >>> print(diag["posterior_means"])  # E[x^T θ_a] for each action
        """
        if context.shape[0] != self.context_dim:
            raise ValueError(
                f"Context dimension {context.shape[0]} doesn't match {self.context_dim}"
            )

        # Sample θ_a ~ N(μ_a, Σ_a) for each action
        sampled_thetas = np.array([
            self.rng.multivariate_normal(self.mu[a], self.Sigma[a])
            for a in range(self.n_actions)
        ])

        # Predicted rewards: r_a = x^T θ_a
        predicted_rewards = sampled_thetas @ context  # [n_actions]

        # Select best action
        selected_action = int(predicted_rewards.argmax())

        if return_diagnostics:
            # Posterior mean predictions
            posterior_means = self.mu @ context  # [n_actions]

            # Posterior predictive variance
            posterior_vars = np.array([
                context @ self.Sigma[a] @ context
                for a in range(self.n_actions)
            ])

            diagnostics = {
                "predicted_rewards": predicted_rewards,
                "posterior_means": posterior_means,
                "posterior_vars": posterior_vars,
                "selected_action": selected_action,
            }
            return selected_action, diagnostics
        else:
            return selected_action

    def update(
        self,
        context: npt.NDArray[np.floating],
        action: int,
        reward: float,
    ) -> None:
        """
        Update posterior from observation.

        Gaussian conjugate update:
            Σ_a^{-1} ← Σ_a^{-1} + (1/σ²) x x^T
            μ_a ← Σ_a (Σ_a^{-1} μ_a + (r/σ²) x)

        Args:
            context: Context vector, shape [context_dim]
            action: Action index (0 to n_actions-1)
            reward: Observed reward

        Raises:
            ValueError: If context dimension or action index invalid

        Example:
            >>> context = np.random.randn(20)
            >>> action = 2
            >>> reward = 0.85
            >>> sampler.update(context, action, reward)
        """
        if context.shape[0] != self.context_dim:
            raise ValueError(
                f"Context dimension {context.shape[0]} doesn't match {self.context_dim}"
            )
        if not (0 <= action < self.n_actions):
            raise ValueError(f"Action {action} out of bounds [0, {self.n_actions})")

        # Precision update: Σ^{-1} ← Σ^{-1} + (1/σ²) x x^T
        sigma_sq = self.sigma_noise ** 2
        self.Sigma_inv[action] += (1.0 / sigma_sq) * np.outer(context, context)

        # Covariance update: Σ ← (Σ^{-1})^{-1}
        self.Sigma[action] = np.linalg.inv(self.Sigma_inv[action])

        # Mean update: μ ← Σ (Σ^{-1} μ + (r/σ²) x)
        prior_weighted = self.Sigma_inv[action] @ self.mu[action]
        obs_weighted = (reward / sigma_sq) * context
        self.mu[action] = self.Sigma[action] @ (prior_weighted + obs_weighted)

        # Statistics
        self.pulls[action] += 1
        self.total_reward[action] += reward

    def get_statistics(self) -> dict:
        """
        Get sampler statistics.

        Returns:
            Dict with:
                - pulls: Pull counts per action
                - empirical_means: Sample mean rewards
                - posterior_norms: ||μ_a|| for each action
                - best_action: Action with highest pull count
                - total_pulls: Sum of pulls

        Example:
            >>> stats = sampler.get_statistics()
            >>> print(f"Best action: {stats['best_action']}")
            >>> print(f"Pulls: {stats['pulls']}")
        """
        empirical_means = np.where(
            self.pulls > 0,
            self.total_reward / self.pulls,
            0.0
        )

        posterior_norms = np.linalg.norm(self.mu, axis=1)

        return {
            "n_actions": self.n_actions,
            "context_dim": self.context_dim,
            "total_pulls": int(self.pulls.sum()),
            "pulls": self.pulls,
            "empirical_means": empirical_means,
            "posterior_norms": posterior_norms,
            "best_action": int(self.pulls.argmax()),
        }

    def get_posterior_params(self, action: int) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Get posterior parameters (μ, Σ) for action.

        Args:
            action: Action index

        Returns:
            Tuple of (mu, Sigma) where:
                - mu: shape [context_dim]
                - Sigma: shape [context_dim, context_dim]

        Example:
            >>> mu, Sigma = sampler.get_posterior_params(action=0)
            >>> print(mu.shape)  # [20]
            >>> print(Sigma.shape)  # [20, 20]
        """
        if not (0 <= action < self.n_actions):
            raise ValueError(f"Action {action} out of bounds")

        return self.mu[action], self.Sigma[action]

    def reset(self, action: int | None = None) -> None:
        """
        Reset posterior to prior.

        Args:
            action: If specified, reset only this action. Otherwise reset all.

        Example:
            >>> sampler.reset()  # Reset all actions
            >>> sampler.reset(action=2)  # Reset only action 2
        """
        if action is None:
            # Reset all
            self.mu[:] = 0.0
            for a in range(self.n_actions):
                self.Sigma[a] = np.eye(self.context_dim) / self.lambda_prior
                self.Sigma_inv[a] = np.eye(self.context_dim) * self.lambda_prior
            self.pulls[:] = 0
            self.total_reward[:] = 0.0
        else:
            # Reset single action
            if not (0 <= action < self.n_actions):
                raise ValueError(f"Action {action} out of bounds")
            self.mu[action] = 0.0
            self.Sigma[action] = np.eye(self.context_dim) / self.lambda_prior
            self.Sigma_inv[action] = np.eye(self.context_dim) * self.lambda_prior
            self.pulls[action] = 0
            self.total_reward[action] = 0.0

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"BayesianLinearTS(context_dim={self.context_dim}, "
            f"n_actions={self.n_actions}, pulls={stats['total_pulls']}, "
            f"best_action={stats['best_action']})"
        )


# ============================================================================
# Factory
# ============================================================================


def create_bayesian_linear_ts(
    context_dim: int,
    n_actions: int,
    prior: Literal["default", "tight", "loose"] = "default",
    seed: int | None = None,
) -> BayesianLinearTS:
    """
    Factory for Bayesian Linear TS with preset priors.

    Args:
        context_dim: Context dimension
        n_actions: Number of actions
        prior: Prior type:
            - "default": λ=1, σ=1 (balanced)
            - "tight": λ=10, σ=0.5 (less exploration)
            - "loose": λ=0.1, σ=2.0 (more exploration)
        seed: Random seed

    Returns:
        Initialized BayesianLinearTS

    Example:
        >>> sampler = create_bayesian_linear_ts(
        ...     context_dim=50,
        ...     n_actions=10,
        ...     prior="default"
        ... )
    """
    prior_params = {
        "default": (1.0, 1.0),
        "tight": (10.0, 0.5),
        "loose": (0.1, 2.0),
    }

    lambda_prior, sigma_noise = prior_params[prior]

    return BayesianLinearTS(
        context_dim=context_dim,
        n_actions=n_actions,
        lambda_prior=lambda_prior,
        sigma_noise=sigma_noise,
        seed=seed,
    )
