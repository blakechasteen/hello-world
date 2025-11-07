"""
Gaussian Process Thompson Sampling and UCB
==========================================
Bayesian optimization for continuous action spaces using GP priors.

WHY GAUSSIAN PROCESSES?
- Thompson Sampling (current): Assumes Beta distributions (discrete arms)
- Reality: Tool parameters form continuous spaces
- GP-TS: Learns smooth functions over parameter spaces
- UCB-GP: Principled exploration with regret bounds

BREAKTHROUGH APPLICATIONS:
1. **Hyperparameter tuning**: Learning optimal retrieval_k, temperature, etc.
2. **Adaptive compute**: Balancing quality vs. latency
3. **Multi-objective optimization**: Pareto frontiers for accuracy/speed
4. **Transfer learning**: GP kernels encode prior knowledge

Mathematical Foundation:
-----------------------
Gaussian Process: Distribution over functions f ~ GP(μ, k)
- μ(x): Mean function (usually 0)
- k(x, x'): Kernel (covariance between inputs)

Posterior after observing data D = {(x_i, y_i)}:
    f(x) | D ~ N(μ_D(x), σ²_D(x))

Where:
    μ_D(x) = k(x, X) [K + σ²I]⁻¹ y         (predictive mean)
    σ²_D(x) = k(x, x) - k(x, X) [K + σ²I]⁻¹ k(X, x)  (predictive variance)

Thompson Sampling: Sample f ~ GP posterior, select x = argmax f(x)
UCB: Select x = argmax [μ_D(x) + β × σ_D(x)]

Author: HoloLoom Bayesian Optimization Team
Date: 2025-11-03
"""

from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings


# ============================================================================
# Kernel Functions
# ============================================================================

class KernelType(Enum):
    """Types of GP kernels."""

    RBF = "rbf"                      # Radial Basis Function (smooth)
    MATERN = "matern"                # Matérn (rougher, realistic)
    PERIODIC = "periodic"            # For periodic patterns
    LINEAR = "linear"                # For linear trends


@dataclass
class KernelConfig:
    """Configuration for GP kernel."""

    kernel_type: KernelType = KernelType.MATERN
    length_scale: float = 1.0        # How far correlations extend
    variance: float = 1.0            # Output scale
    nu: float = 2.5                  # Matérn smoothness (1.5, 2.5, inf)
    period: float = 1.0              # Period for periodic kernel


class Kernel:
    """Base class for GP kernels."""

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix K[i, j] = k(X1[i], X2[j]).

        Args:
            X1: Input points (n1 × d)
            X2: Input points (n2 × d)

        Returns:
            Kernel matrix (n1 × n2)
        """
        raise NotImplementedError


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential kernel.

    k(x, x') = σ² × exp(-||x - x'||² / (2ℓ²))

    Infinitely differentiable (very smooth). Standard choice.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Squared distances
        sq_dists = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1) - 2 * X1 @ X2.T

        # RBF kernel
        return self.variance * np.exp(-0.5 * sq_dists / self.length_scale**2)


class MaternKernel(Kernel):
    """
    Matérn kernel (more realistic than RBF).

    k(x, x') = σ² × (1 + √(2ν) r/ℓ + νr²/ℓ²) × exp(-√(2ν) r/ℓ)

    Where r = ||x - x'||

    Common values:
    - ν = 1/2: Exponential (not differentiable)
    - ν = 3/2: Once differentiable
    - ν = 5/2: Twice differentiable (recommended)
    - ν → ∞: Converges to RBF
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        nu: float = 2.5
    ):
        self.length_scale = length_scale
        self.variance = variance
        self.nu = nu

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Euclidean distances
        dists = np.sqrt(
            np.sum(X1**2, axis=1, keepdims=True) +
            np.sum(X2**2, axis=1) - 2 * X1 @ X2.T +
            1e-10  # Numerical stability
        )

        r = dists / self.length_scale

        if self.nu == 0.5:
            # Exponential kernel
            return self.variance * np.exp(-r)

        elif self.nu == 1.5:
            # Once differentiable
            sqrt3_r = np.sqrt(3) * r
            return self.variance * (1 + sqrt3_r) * np.exp(-sqrt3_r)

        elif self.nu == 2.5:
            # Twice differentiable (recommended)
            sqrt5_r = np.sqrt(5) * r
            return self.variance * (1 + sqrt5_r + 5*r**2/3) * np.exp(-sqrt5_r)

        else:
            # General formula (requires special functions)
            warnings.warn(f"Matérn ν={self.nu} not optimized, using RBF approximation")
            return self.variance * np.exp(-0.5 * dists**2 / self.length_scale**2)


def create_kernel(config: KernelConfig) -> Kernel:
    """Factory function to create kernel."""
    if config.kernel_type == KernelType.RBF:
        return RBFKernel(config.length_scale, config.variance)

    elif config.kernel_type == KernelType.MATERN:
        return MaternKernel(config.length_scale, config.variance, config.nu)

    else:
        raise ValueError(f"Kernel type {config.kernel_type} not implemented")


# ============================================================================
# Gaussian Process
# ============================================================================

@dataclass
class GaussianProcess:
    """
    Gaussian Process regression.

    Maintains posterior distribution over functions given observed data.
    Supports efficient incremental updates.
    """

    kernel: Kernel
    noise_variance: float = 0.01     # Observation noise σ²
    mean_function: Optional[Callable[[np.ndarray], float]] = None

    # Data
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Cached computations
    K_inv: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None  # K_inv @ y
    L: Optional[np.ndarray] = None      # Cholesky factor

    def __post_init__(self):
        if self.mean_function is None:
            self.mean_function = lambda x: 0.0

    def add_observation(self, x: np.ndarray, y: float):
        """
        Add new observation to GP.

        Incrementally updates posterior without full recomputation.

        Args:
            x: Input point (1D array)
            y: Observed output (scalar)
        """
        # Add to dataset
        if self.X.shape[0] == 0:
            self.X = x.reshape(1, -1)
            self.y = np.array([y])
        else:
            self.X = np.vstack([self.X, x])
            self.y = np.append(self.y, y)

        # Invalidate cache (force recomputation on next predict)
        self.K_inv = None
        self.alpha = None
        self.L = None

    def _compute_posterior_params(self):
        """Compute posterior parameters (cached)."""
        if self.alpha is not None:
            return  # Already computed

        n = len(self.y)
        if n == 0:
            return

        # Kernel matrix + noise
        K = self.kernel(self.X, self.X)
        K += self.noise_variance * np.eye(n)

        # Cholesky decomposition for numerical stability
        try:
            self.L = np.linalg.cholesky(K)
            # Solve L @ L.T @ alpha = y
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
            self.K_inv = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.eye(n)))
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            warnings.warn("Cholesky decomposition failed, using direct inversion")
            self.K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))
            self.alpha = self.K_inv @ self.y

    def predict(
        self,
        X_test: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict at test points.

        Args:
            X_test: Test input points (n_test × d)
            return_std: Whether to return standard deviation

        Returns:
            Tuple of (mean, std) or just mean
        """
        if len(self.y) == 0:
            # No data: return prior
            mean = np.array([self.mean_function(x) for x in X_test])
            if return_std:
                # Prior variance = kernel diagonal
                var = np.array([self.kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0]
                               for x in X_test])
                return mean, np.sqrt(var)
            return mean, None

        self._compute_posterior_params()

        # Predictive mean: μ(x*) = k(x*, X) @ K_inv @ y
        k_star = self.kernel(X_test, self.X)
        mean = k_star @ self.alpha

        if not return_std:
            return mean, None

        # Predictive variance: σ²(x*) = k(x*, x*) - k(x*, X) @ K_inv @ k(X, x*)
        k_star_star = np.array([
            self.kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0]
            for x in X_test
        ])

        var = k_star_star - np.sum(k_star @ self.K_inv * k_star, axis=1)
        var = np.maximum(var, 0.0)  # Numerical stability

        return mean, np.sqrt(var)

    def sample_posterior(
        self,
        X_test: np.ndarray,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Sample functions from GP posterior.

        Used for Thompson Sampling.

        Args:
            X_test: Test points
            n_samples: Number of function samples
            rng: Random number generator

        Returns:
            Samples (n_samples × n_test)
        """
        if rng is None:
            rng = np.random.default_rng()

        mean, std = self.predict(X_test, return_std=True)

        # Sample: f ~ N(μ, σ²)
        samples = mean + std * rng.standard_normal((n_samples, len(X_test)))
        return samples


# ============================================================================
# GP Thompson Sampling
# ============================================================================

@dataclass
class GPThompsonSampling:
    """
    Gaussian Process Thompson Sampling.

    Samples function from GP posterior, selects action maximizing sample.
    Provides natural exploration: uncertain regions get sampled optimistically.

    Algorithm:
    1. Fit GP to observed data
    2. Sample f ~ GP posterior
    3. Select x = argmax f(x) over candidate set
    4. Observe y, update GP
    """

    gp: GaussianProcess
    candidate_set: np.ndarray        # Discrete action set
    n_samples: int = 1               # Functions to sample (usually 1)

    def __post_init__(self):
        self.rng = np.random.default_rng()
        self.iteration = 0

    def select_action(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select next action via Thompson Sampling.

        Returns:
            Tuple of (action, metadata)
        """
        self.iteration += 1

        # Sample function from GP posterior
        samples = self.gp.sample_posterior(
            self.candidate_set,
            n_samples=self.n_samples,
            rng=self.rng
        )

        # Select action maximizing sampled function
        best_idx = np.argmax(samples[0])  # Use first sample
        action = self.candidate_set[best_idx]

        # Get posterior statistics
        mean, std = self.gp.predict(self.candidate_set, return_std=True)

        metadata = {
            'iteration': self.iteration,
            'selected_idx': int(best_idx),
            'sampled_value': float(samples[0, best_idx]),
            'posterior_mean': float(mean[best_idx]),
            'posterior_std': float(std[best_idx]),
            'n_observations': len(self.gp.y)
        }

        return action, metadata

    def update(self, action: np.ndarray, reward: float):
        """
        Update GP with observed reward.

        Args:
            action: Action taken
            reward: Observed reward
        """
        self.gp.add_observation(action, reward)


# ============================================================================
# GP Upper Confidence Bound (UCB)
# ============================================================================

@dataclass
class GPUpperConfidenceBound:
    """
    Gaussian Process Upper Confidence Bound (GP-UCB).

    Selects actions via: x = argmax [μ(x) + β × σ(x)]

    β controls exploration:
    - β = 0: Pure exploitation (greedy)
    - β → ∞: Pure exploration (random)
    - β = √(2 log(t)): Theoretical regret bound

    Advantages over Thompson Sampling:
    - Deterministic (reproducible)
    - Theoretical regret guarantees
    - Easier to tune (single β parameter)
    """

    gp: GaussianProcess
    candidate_set: np.ndarray
    beta: float = 2.0                # Exploration parameter

    # Adaptive beta scheduling
    adaptive_beta: bool = False      # Use β = √(2 log(t))?

    def __post_init__(self):
        self.iteration = 0

    def select_action(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select next action via GP-UCB.

        Returns:
            Tuple of (action, metadata)
        """
        self.iteration += 1

        # Adaptive beta schedule
        if self.adaptive_beta:
            beta = np.sqrt(2 * np.log(self.iteration + 1))
        else:
            beta = self.beta

        # Get posterior
        mean, std = self.gp.predict(self.candidate_set, return_std=True)

        # UCB: μ + β × σ
        ucb = mean + beta * std

        # Select action maximizing UCB
        best_idx = np.argmax(ucb)
        action = self.candidate_set[best_idx]

        metadata = {
            'iteration': self.iteration,
            'selected_idx': int(best_idx),
            'ucb_value': float(ucb[best_idx]),
            'posterior_mean': float(mean[best_idx]),
            'posterior_std': float(std[best_idx]),
            'beta': float(beta),
            'n_observations': len(self.gp.y)
        }

        return action, metadata

    def update(self, action: np.ndarray, reward: float):
        """
        Update GP with observed reward.

        Args:
            action: Action taken
            reward: Observed reward
        """
        self.gp.add_observation(action, reward)


# ============================================================================
# Factory Functions
# ============================================================================

def create_gp_thompson_sampling(
    candidate_set: np.ndarray,
    kernel_config: Optional[KernelConfig] = None,
    noise_variance: float = 0.01
) -> GPThompsonSampling:
    """
    Factory function for GP-TS.

    Args:
        candidate_set: Discrete action set (n_actions × d)
        kernel_config: Kernel configuration
        noise_variance: Observation noise

    Returns:
        GPThompsonSampling instance
    """
    if kernel_config is None:
        kernel_config = KernelConfig()

    kernel = create_kernel(kernel_config)
    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)

    return GPThompsonSampling(gp=gp, candidate_set=candidate_set)


def create_gp_ucb(
    candidate_set: np.ndarray,
    kernel_config: Optional[KernelConfig] = None,
    noise_variance: float = 0.01,
    beta: float = 2.0,
    adaptive_beta: bool = False
) -> GPUpperConfidenceBound:
    """
    Factory function for GP-UCB.

    Args:
        candidate_set: Discrete action set
        kernel_config: Kernel configuration
        noise_variance: Observation noise
        beta: Exploration parameter
        adaptive_beta: Use adaptive β schedule?

    Returns:
        GPUpperConfidenceBound instance
    """
    if kernel_config is None:
        kernel_config = KernelConfig()

    kernel = create_kernel(kernel_config)
    gp = GaussianProcess(kernel=kernel, noise_variance=noise_variance)

    return GPUpperConfidenceBound(
        gp=gp,
        candidate_set=candidate_set,
        beta=beta,
        adaptive_beta=adaptive_beta
    )


# ============================================================================
# Compatibility Wrappers (Legacy API)
# ============================================================================


@dataclass
class GaussianProcessBandit:
    """Backwards-compatible wrapper exposing the legacy GP bandit API."""

    kernel_config: Optional[KernelConfig] = None
    sampler_type: str = "thompson"
    noise_variance: float = 0.01
    candidate_set: Optional[np.ndarray] = None
    n_thompson_samples: int = 1

    def __post_init__(self):
        if self.kernel_config is None:
            self.kernel_config = KernelConfig()

        self.kernel = create_kernel(self.kernel_config)
        self._gp = GaussianProcess(
            kernel=self.kernel,
            noise_variance=self.noise_variance
        )

        # Observation history for tests/introspection
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []

        # Candidate set (optional, used for select_action helper)
        self._candidate_set = self._ensure_2d(self.candidate_set)
        self._rng = np.random.default_rng()

    @staticmethod
    def _ensure_2d(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if points is None:
            return None
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def add_observation(self, x: np.ndarray, y: float):
        """Add an observation to the GP posterior."""
        x = np.asarray(x, dtype=float).reshape(-1)
        self._gp.add_observation(x, float(y))
        self.X_observed.append(x)
        self.y_observed.append(float(y))

        # If we did not yet know the dimensionality, create a default grid
        if self._candidate_set is None:
            dim = x.shape[0]
            self._candidate_set = np.linspace(0, 1, 50).reshape(-1, 1)
            if dim > 1:
                self._candidate_set = np.zeros((50, dim))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and std for candidate points."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mean, std = self._gp.predict(X, return_std=True)
        return mean, std if std is not None else np.zeros(len(mean))

    def thompson_sample(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sample an action via Thompson Sampling on provided candidates."""
        if self.sampler_type != "thompson":
            raise RuntimeError("Thompson sampling requested on non-Thompson bandit")

        X = self._ensure_2d(X)
        if X is None:
            raise ValueError("candidate set required for Thompson sampling")

        mean, std = self.predict(X)
        std = np.maximum(std, 1e-8)
        samples = self._rng.normal(loc=mean, scale=std, size=mean.shape)
        idx = int(np.argmax(samples))
        return X[idx], {
            'selected_idx': idx,
            'posterior_mean': float(mean[idx]),
            'posterior_std': float(std[idx]),
            'sampled_value': float(samples[idx]),
        }

    def select_action(self, candidate_set: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convenience helper mirroring the legacy API."""
        points = self._ensure_2d(candidate_set) or self._candidate_set
        if points is None:
            raise ValueError("candidate_set must be provided for select_action")
        return self.thompson_sample(points)


@dataclass
class GaussianProcessUCB(GaussianProcessBandit):
    """Legacy-compatible GP-UCB wrapper."""

    beta: float = 2.0
    adaptive_beta: bool = False

    def __post_init__(self):
        self.sampler_type = "ucb"
        super().__post_init__()
        self._iteration = 0

    def _current_beta(self) -> float:
        if self.adaptive_beta:
            return np.sqrt(2.0 * np.log(max(2, self._iteration + 1)))
        return self.beta

    def ucb(self, X: np.ndarray) -> np.ndarray:
        """Compute UCB scores for candidate points."""
        X = self._ensure_2d(X)
        if X is None:
            raise ValueError("candidate points required for UCB computation")

        mean, std = self.predict(X)
        return mean + self._current_beta() * std

    def select_action(self, candidate_set: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        points = self._ensure_2d(candidate_set) or self._candidate_set
        if points is None:
            raise ValueError("candidate_set must be provided for select_action")

        self._iteration += 1
        scores = self.ucb(points)
        idx = int(np.argmax(scores))
        return points[idx], {
            'selected_idx': idx,
            'ucb_value': float(scores[idx]),
            'posterior_mean': float(self.predict(points)[0][idx]),
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'KernelType',
    'KernelConfig',
    'Kernel',
    'RBFKernel',
    'MaternKernel',
    'create_kernel',
    'GaussianProcess',
    'GPThompsonSampling',
    'GPUpperConfidenceBound',
    'create_gp_thompson_sampling',
    'create_gp_ucb',
    'GaussianProcessBandit',
    'GaussianProcessUCB',
]
