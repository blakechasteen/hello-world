"""
Gaussian Process Thompson Sampling.

Continuous parameter optimization using GP posterior and Thompson Sampling.
For hyperparameter tuning, continuous control, and black-box optimization.

**Model**:
    f(x) ~ GP(0, k(x, x'))
    y = f(x) + ε, ε ~ N(0, σ²)

**Kernels**:
    - RBF: k(x, x') = exp(-||x - x'||² / (2l²))
    - Matern: More flexible smoothness

**Thompson Sampling for GPs**:
    1. Sample f ~ GP(μ, Σ) (posterior)
    2. Optimize: x* = argmax_x f(x)
    3. Observe y at x*
    4. Update GP posterior

Use cases:
- Hyperparameter tuning
- Continuous control
- Black-box optimization
- Bayesian optimization

Example:
    >>> sampler = GaussianProcessTS(param_dim=5, kernel="rbf")
    >>> for t in range(100):
    ...     params = sampler.select()  # Sample from acquisition
    ...     reward = evaluate(params)
    ...     sampler.update(params, reward)
    >>> best_params = sampler.get_best_params()
"""

import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class GaussianProcessTS:
    """
    Gaussian Process Thompson Sampling.

    **WARNING**: Simplified implementation for demonstration.
    For production GP-TS, use GPyTorch or scikit-learn's GP.

    This implementation:
    - Uses exact GP (no inducing points)
    - Simple acquisition (Thompson sampling via GP sample paths)
    - RBF and Matern kernels
    - Cholesky factorization for stability

    Attributes:
        param_dim: Parameter space dimension
        kernel_type: "rbf" or "matern"
        length_scale: Kernel length scale
        noise_std: Observation noise
        X: Observed parameters (shape: [n, param_dim])
        y: Observed rewards (shape: [n])

    Example:
        >>> # Tune 5D hyperparameters
        >>> sampler = GaussianProcessTS(param_dim=5, kernel="rbf", length_scale=0.5)
        >>>
        >>> # Select next parameters to evaluate
        >>> params = sampler.select()  # [5]
        >>>
        >>> # Evaluate objective
        >>> reward = objective_function(params)
        >>>
        >>> # Update GP
        >>> sampler.update(params, reward)
        >>>
        >>> # Get best found so far
        >>> best = sampler.get_best_params()
    """

    def __init__(
        self,
        param_dim: int,
        kernel: Literal["rbf", "matern"] = "rbf",
        length_scale: float = 1.0,
        noise_std: float = 0.1,
        bounds: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
    ):
        """
        Initialize GP Thompson Sampling.

        Args:
            param_dim: Parameter space dimension
            kernel: Kernel type ("rbf" or "matern")
            length_scale: Kernel length scale (controls smoothness)
            noise_std: Observation noise std dev
            bounds: Parameter bounds (low, high) - assumes same for all dims
            seed: Random seed

        Example:
            >>> sampler = GaussianProcessTS(
            ...     param_dim=10,
            ...     kernel="rbf",
            ...     length_scale=0.5,
            ...     noise_std=0.1,
            ...     bounds=(0.0, 1.0)
            ... )
        """
        self.param_dim = param_dim
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.noise_std = noise_std
        self.bounds = bounds

        # Observations
        self.X = np.empty((0, param_dim))
        self.y = np.empty(0)

        # RNG
        self.rng = np.random.RandomState(seed)

    def select(self, n_candidates: int = 100) -> npt.NDArray[np.floating]:
        """
        Select next parameters using Thompson Sampling.

        Algorithm (simplified):
            1. Sample function f ~ GP(posterior)
            2. Evaluate f on random candidates
            3. Return argmax_x f(x)

        Args:
            n_candidates: Number of random candidates to evaluate

        Returns:
            Parameter vector, shape [param_dim]

        Note:
            Production implementation should use gradient-based optimization
            or more sophisticated acquisition functions (UCB, EI).

        Example:
            >>> params = sampler.select(n_candidates=200)
            >>> print(params.shape)  # [param_dim]
        """
        if len(self.X) == 0:
            # No data yet - random exploration
            return self._sample_random()

        # Generate random candidates
        candidates = self._sample_random_batch(n_candidates)

        # Sample GP function and evaluate on candidates
        sampled_values = self._sample_gp_function(candidates)

        # Greedy w.r.t. sampled function
        best_idx = sampled_values.argmax()
        return candidates[best_idx]

    def update(self, params: npt.NDArray[np.floating], reward: float) -> None:
        """
        Update GP posterior.

        Args:
            params: Parameter vector, shape [param_dim]
            reward: Observed reward

        Example:
            >>> params = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
            >>> reward = 0.85
            >>> sampler.update(params, reward)
        """
        self.X = np.vstack([self.X, params])
        self.y = np.append(self.y, reward)

    def get_best_params(self) -> npt.NDArray[np.floating]:
        """
        Get parameters with highest observed reward.

        Returns:
            Best parameter vector, shape [param_dim]

        Example:
            >>> best = sampler.get_best_params()
            >>> print(best)  # [param_dim]
        """
        if len(self.y) == 0:
            return self._sample_random()

        best_idx = self.y.argmax()
        return self.X[best_idx]

    def get_statistics(self) -> dict:
        """Get GP statistics."""
        if len(self.y) == 0:
            return {
                "n_observations": 0,
                "best_reward": 0.0,
                "mean_reward": 0.0,
            }

        return {
            "n_observations": len(self.y),
            "best_reward": float(self.y.max()),
            "mean_reward": float(self.y.mean()),
            "std_reward": float(self.y.std()),
        }

    def _kernel(self, X1: npt.NDArray, X2: npt.NDArray) -> npt.NDArray:
        """Compute kernel matrix K(X1, X2)."""
        if self.kernel_type == "rbf":
            return self._rbf_kernel(X1, X2)
        elif self.kernel_type == "matern":
            return self._matern_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def _rbf_kernel(self, X1: npt.NDArray, X2: npt.NDArray) -> npt.NDArray:
        """RBF kernel: k(x, x') = exp(-||x - x'||² / (2l²))"""
        # Pairwise squared distances
        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1, keepdims=True).T - \
                   2 * X1 @ X2.T

        return np.exp(-sq_dists / (2 * self.length_scale ** 2))

    def _matern_kernel(self, X1: npt.NDArray, X2: npt.NDArray) -> npt.NDArray:
        """Simplified Matern-3/2 kernel."""
        dists = np.linalg.norm(
            X1[:, None, :] - X2[None, :, :],
            axis=2
        )
        sqrt3_d = np.sqrt(3) * dists / self.length_scale
        return (1 + sqrt3_d) * np.exp(-sqrt3_d)

    def _sample_gp_function(self, X_test: npt.NDArray) -> npt.NDArray:
        """Sample from GP posterior at test points."""
        n = len(self.X)

        # Kernel matrices
        K = self._kernel(self.X, self.X)
        K += self.noise_std ** 2 * np.eye(n)  # Add noise
        K_s = self._kernel(self.X, X_test)
        K_ss = self._kernel(X_test, X_test)

        # Cholesky factorization for stability
        L = np.linalg.cholesky(K)

        # Posterior mean
        alpha = cho_solve((L, True), self.y)
        mu = K_s.T @ alpha

        # Posterior covariance
        v = cho_solve((L, True), K_s)
        cov = K_ss - K_s.T @ v

        # Sample from posterior
        # Add small jitter for numerical stability
        cov += 1e-6 * np.eye(len(X_test))

        try:
            sample = self.rng.multivariate_normal(mu, cov)
        except np.linalg.LinAlgError:
            # Fallback if covariance is ill-conditioned
            sample = mu + self.rng.randn(len(X_test)) * 0.01

        return sample

    def _sample_random(self) -> npt.NDArray[np.floating]:
        """Sample random parameter vector."""
        low, high = self.bounds
        return self.rng.uniform(low, high, size=self.param_dim)

    def _sample_random_batch(self, n: int) -> npt.NDArray[np.floating]:
        """Sample batch of random parameter vectors."""
        low, high = self.bounds
        return self.rng.uniform(low, high, size=(n, self.param_dim))

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"GaussianProcessTS(param_dim={self.param_dim}, "
            f"kernel={self.kernel_type}, "
            f"n_obs={stats['n_observations']}, "
            f"best_reward={stats['best_reward']:.3f})"
        )


# ============================================================================
# Factory
# ============================================================================


def create_gp_ts(
    param_dim: int,
    kernel: Literal["rbf", "matern"] = "rbf",
    bounds: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
) -> GaussianProcessTS:
    """
    Factory for GP Thompson Sampling.

    Args:
        param_dim: Parameter space dimension
        kernel: Kernel type
        bounds: Parameter bounds
        seed: Random seed

    Returns:
        Initialized GaussianProcessTS

    Example:
        >>> sampler = create_gp_ts(param_dim=5, kernel="rbf")
    """
    return GaussianProcessTS(
        param_dim=param_dim,
        kernel=kernel,
        length_scale=1.0,
        noise_std=0.1,
        bounds=bounds,
        seed=seed,
    )
