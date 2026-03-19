"""
Thompson Sampling for Optimal Cluster Count Discovery

Instead of trying every k and picking the best (expensive),
use Thompson Sampling to intelligently explore k values.

This is HoloLoom's "magic sauce" for clustering:
- Bayesian exploration of k values
- Learns from silhouette scores as rewards
- Converges faster than grid search
- Handles uncertainty naturally
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BetaPrior:
    """Beta distribution prior for a single k value."""
    alpha: float = 1.0  # Successes (good silhouette)
    beta: float = 1.0   # Failures (bad silhouette)

    def sample(self) -> float:
        """Sample from Beta(alpha, beta)."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: float):
        """Update prior with observed reward (0-1 scale)."""
        # Treat reward as probability of success
        self.alpha += reward
        self.beta += (1 - reward)

    @property
    def mean(self) -> float:
        """Expected value of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))


@dataclass
class ThompsonClusterSampler:
    """
    Thompson Sampling for finding optimal number of clusters.

    Instead of exhaustively trying all k values, we:
    1. Maintain Beta priors for each k
    2. Sample from priors to decide which k to try
    3. Update priors based on silhouette scores
    4. Converge toward optimal k over iterations
    """

    min_k: int = 2
    max_k: int = 10
    priors: dict[int, BetaPrior] = field(default_factory=dict)
    history: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self):
        # Initialize priors for each k
        for k in range(self.min_k, self.max_k + 1):
            if k not in self.priors:
                self.priors[k] = BetaPrior()

    def sample_k(self) -> int:
        """
        Sample next k to try using Thompson Sampling.

        Returns k with highest sampled value from its Beta prior.
        """
        samples = {k: prior.sample() for k, prior in self.priors.items()}
        return max(samples, key=samples.get)

    def update(self, k: int, silhouette: float):
        """
        Update prior for k based on observed silhouette score.

        Args:
            k: Number of clusters that was tried
            silhouette: Silhouette score (-1 to 1), normalized to (0, 1)
        """
        # Normalize silhouette from [-1, 1] to [0, 1]
        reward = (silhouette + 1) / 2

        if k in self.priors:
            self.priors[k].update(reward)
            self.history.append((k, silhouette))

    def get_best_k(self) -> tuple[int, float]:
        """
        Get the current best k based on prior means.

        Returns:
            (best_k, expected_silhouette)
        """
        means = {k: prior.mean for k, prior in self.priors.items()}
        best_k = max(means, key=means.get)

        # Convert back from normalized reward to silhouette
        expected_silhouette = means[best_k] * 2 - 1

        return best_k, expected_silhouette

    def get_confidence(self, k: int) -> float:
        """
        Get confidence in the estimate for a given k.

        Lower variance = higher confidence.
        """
        if k not in self.priors:
            return 0.0

        variance = self.priors[k].variance
        # Convert variance to confidence (inverse relationship)
        confidence = 1 / (1 + variance * 10)
        return min(confidence, 1.0)

    def should_explore_more(self, threshold: float = 0.8) -> bool:
        """
        Check if we should continue exploring k values.

        Returns True if uncertainty is still high.
        """
        best_k, _ = self.get_best_k()
        confidence = self.get_confidence(best_k)
        return confidence < threshold

    def get_statistics(self) -> dict[str, any]:
        """Get summary statistics for all k values."""
        return {
            k: {
                "mean": prior.mean,
                "variance": prior.variance,
                "alpha": prior.alpha,
                "beta": prior.beta,
                "confidence": self.get_confidence(k)
            }
            for k, prior in sorted(self.priors.items())
        }


def find_optimal_k_thompson(
    embeddings: np.ndarray,
    cluster_fn,
    min_k: int = 2,
    max_k: int = 10,
    max_iterations: int = 20,
    confidence_threshold: float = 0.8
) -> tuple[int, float, ThompsonClusterSampler]:
    """
    Find optimal k using Thompson Sampling.

    Args:
        embeddings: Data embeddings (n_samples x dim)
        cluster_fn: Function(embeddings, k) -> silhouette_score
        min_k: Minimum k to consider
        max_k: Maximum k to consider
        max_iterations: Max sampling iterations
        confidence_threshold: Stop when confidence exceeds this

    Returns:
        (optimal_k, expected_silhouette, sampler)
    """
    # Adjust max_k based on data size
    n_samples = len(embeddings)
    max_k = min(max_k, n_samples - 1)

    if max_k < min_k:
        return min_k, 0.0, None

    sampler = ThompsonClusterSampler(min_k=min_k, max_k=max_k)

    for iteration in range(max_iterations):
        # Sample k using Thompson Sampling
        k = sampler.sample_k()

        # Try clustering with this k
        silhouette = cluster_fn(embeddings, k)

        # Update prior
        sampler.update(k, silhouette)

        # Check if we're confident enough
        if not sampler.should_explore_more(confidence_threshold):
            break

    best_k, expected_score = sampler.get_best_k()
    return best_k, expected_score, sampler


def adaptive_k_search(
    embeddings: np.ndarray,
    cluster_fn,
    min_k: int = 2,
    max_k: int = 10,
    budget: int = 10
) -> tuple[int, float]:
    """
    Adaptive k search with limited budget.

    Uses Thompson Sampling with a fixed budget of evaluations,
    ideal for expensive clustering algorithms.

    Args:
        embeddings: Data embeddings
        cluster_fn: Function(embeddings, k) -> silhouette_score
        min_k, max_k: k range
        budget: Maximum number of clustering operations

    Returns:
        (optimal_k, silhouette_score)
    """
    best_k, best_score, _ = find_optimal_k_thompson(
        embeddings=embeddings,
        cluster_fn=cluster_fn,
        min_k=min_k,
        max_k=max_k,
        max_iterations=budget,
        confidence_threshold=0.95  # High threshold, rely on budget
    )

    return best_k, best_score
