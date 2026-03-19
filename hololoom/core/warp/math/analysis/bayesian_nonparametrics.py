"""
Bayesian Nonparametrics — Dirichlet Process Mixtures
=====================================================

Infinite-dimensional Bayesian models that let the data determine
model complexity. The Dirichlet Process generates distributions
over distributions with an unbounded number of components.

Core Concepts:
- Dirichlet Process: Stick-breaking construction for infinite mixtures
- Chinese Restaurant Process: Sequential partition (exchangeable)
- DP Mixture: Infinite Gaussian mixture model
- Expected Clusters: E[K] ~ α log(n) — grows logarithmically

Mathematical Foundation:
Stick-breaking: π_k = V_k Π_{j<k} (1-V_j), V_k ~ Beta(1, α)
CRP: P(cₙ = k | c_{1:n-1}) ∝ n_k for existing, α for new
Posterior: G | x₁..xₙ ~ DP(α + n, (αG₀ + Σδ_{xᵢ})/(α+n))

Applications to Warp Space:
- Automatic cluster discovery in embedding spaces
- Nonparametric priors for Thompson Sampling
- Open-ended topic modeling without fixing K

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DPMixtureResult:
    """Result of Dirichlet Process mixture model.

    Attributes:
        weights: Component weights (n_active,)
        means: Component means (n_active, n_features)
        covariances: Component covariances (n_active, n_features, n_features)
        labels: Cluster assignments (n_samples,)
        n_clusters: Number of active clusters
        log_likelihood: Final log-likelihood
        n_iter: Number of Gibbs iterations
    """
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    labels: np.ndarray
    n_clusters: int = 0
    log_likelihood: float = 0.0
    n_iter: int = 0


# ============================================================================
# DIRICHLET PROCESS (Stick-Breaking)
# ============================================================================

class DirichletProcess:
    """Dirichlet Process via stick-breaking construction.

    Generates a discrete distribution with (potentially) infinite
    support. Truncated to max_components for practical computation.

    π_k = V_k Π_{j<k} (1 - V_j)  where V_k ~ Beta(1, α)

    Example:
        >>> dp = DirichletProcess()
        >>> weights, atoms = dp.sample(alpha=1.0, n=100)
        >>> abs(weights.sum() - 1.0) < 0.01
        True
    """

    def sample(
        self,
        alpha: float,
        base_measure: Callable[[], np.ndarray] | None = None,
        n: int = 100,
        max_components: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample from a Dirichlet Process.

        Args:
            alpha: Concentration parameter (larger = more clusters)
            base_measure: Callable that samples from G₀ (default: N(0,1))
            n: Not used directly (kept for API compatibility)
            max_components: Truncation level

        Returns:
            (weights, atoms): weights sum to ~1, atoms from base measure
        """
        if base_measure is None:
            base_measure = lambda: np.random.randn(1)

        # Stick-breaking
        V = np.random.beta(1, alpha, size=max_components)
        weights = np.zeros(max_components)
        remaining = 1.0

        for k in range(max_components):
            weights[k] = V[k] * remaining
            remaining *= (1 - V[k])
            if remaining < 1e-10:
                break

        # Sample atoms from base measure
        atoms = np.array([base_measure() for _ in range(max_components)])

        return weights, atoms

    def sample_predictive(
        self,
        alpha: float,
        observations: np.ndarray,
    ) -> np.ndarray:
        """Sample from the posterior predictive distribution.

        P(x_{n+1} | x_1..x_n) = α/(α+n) G₀ + Σ (n_k/(α+n)) δ_{x_k*}

        Args:
            alpha: Concentration parameter
            observations: Previous observations (n,) or (n, d)

        Returns:
            Next sample from posterior predictive
        """
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        n = len(observations)

        if np.random.random() < alpha / (alpha + n):
            # New cluster: sample from base measure (prior)
            d = observations.shape[1]
            return np.random.randn(d)
        else:
            # Existing cluster: sample uniformly from observations
            idx = np.random.randint(n)
            return observations[idx].copy()


# ============================================================================
# CHINESE RESTAURANT PROCESS
# ============================================================================

class ChineseRestaurantProcess:
    """Chinese Restaurant Process — sequential partition generation.

    Customer n sits at:
    - existing table k with probability n_k / (n - 1 + α)
    - new table with probability α / (n - 1 + α)

    Generates exchangeable partitions equivalent to DP.

    Example:
        >>> crp = ChineseRestaurantProcess()
        >>> assignments = crp.sample(alpha=2.0, n=100)
        >>> n_tables = len(set(assignments))
        >>> # Expected: α * log(n) ≈ 2 * 4.6 ≈ 9.2
    """

    def sample(
        self,
        alpha: float,
        n: int,
    ) -> list[int]:
        """Sample a partition from the CRP.

        Args:
            alpha: Concentration parameter
            n: Number of customers

        Returns:
            List of table assignments [0-indexed]
        """
        assignments = [0]  # First customer sits at table 0
        table_counts = [1]

        for i in range(1, n):
            total = i + alpha
            probs = [c / total for c in table_counts]
            probs.append(alpha / total)

            # Sample table
            table = _categorical(probs)

            if table == len(table_counts):
                # New table
                table_counts.append(1)
            else:
                table_counts[table] += 1

            assignments.append(table)

        return assignments

    def partition_counts(
        self,
        alpha: float,
        n: int,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """Estimate distribution of number of clusters.

        Args:
            alpha: Concentration parameter
            n: Number of customers
            n_samples: Monte Carlo samples

        Returns:
            Array of cluster counts from each sample
        """
        counts = np.zeros(n_samples)
        for s in range(n_samples):
            assignments = self.sample(alpha, n)
            counts[s] = len(set(assignments))
        return counts


# ============================================================================
# DP MIXTURE MODEL (Collapsed Gibbs Sampler)
# ============================================================================

class DPMixture:
    """Dirichlet Process Gaussian Mixture Model.

    Infinite mixture model fit via collapsed Gibbs sampling.
    Automatically determines the number of clusters.

    Example:
        >>> dpmm = DPMixture()
        >>> X = np.vstack([np.random.randn(30, 2) + [3, 0],
        ...                np.random.randn(30, 2) + [-3, 0],
        ...                np.random.randn(30, 2) + [0, 3]])
        >>> result = dpmm.fit(X, alpha=1.0, max_iter=50)
        >>> result.n_clusters  # Should find ~3
    """

    def fit(
        self,
        X: np.ndarray,
        alpha: float = 1.0,
        max_iter: int = 100,
        burn_in: int = 20,
    ) -> DPMixtureResult:
        """Fit DP mixture via collapsed Gibbs sampling.

        Args:
            X: Data (n_samples, n_features)
            alpha: DP concentration parameter
            max_iter: Total Gibbs iterations
            burn_in: Iterations to discard

        Returns:
            DPMixtureResult with cluster assignments and parameters
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        # Initialize: each point in its own cluster
        labels = np.arange(n) % min(5, n)  # Start with 5 clusters
        best_labels = labels.copy()
        best_ll = -np.inf

        for iteration in range(max_iter):
            for i in range(n):
                # Remove point i from its cluster
                old_label = labels[i]

                # Compute probabilities for each existing cluster
                unique_labels = np.unique(labels)
                log_probs = []
                candidates = []

                for k in unique_labels:
                    mask = (labels == k)
                    if i == np.where(mask)[0][0] if old_label == k else False:
                        mask_without = mask.copy()
                        mask_without[i] = False
                        n_k = np.sum(mask_without)
                    else:
                        mask_without = mask.copy()
                        mask_without[i] = False
                        n_k = np.sum(mask_without)

                    if n_k == 0:
                        continue

                    # CRP prior
                    log_prior = np.log(n_k)

                    # Marginal likelihood (conjugate normal-inverse-Wishart)
                    cluster_data = X[mask_without]
                    log_lik = self._marginal_log_likelihood(X[i], cluster_data, d)

                    log_probs.append(log_prior + log_lik)
                    candidates.append(k)

                # New cluster probability
                log_prior_new = np.log(alpha)
                log_lik_new = self._marginal_log_likelihood(X[i], np.empty((0, d)), d)
                log_probs.append(log_prior_new + log_lik_new)
                new_label = max(unique_labels) + 1 if len(unique_labels) > 0 else 0
                candidates.append(new_label)

                # Sample assignment
                log_probs = np.array(log_probs)
                log_probs -= np.max(log_probs)  # Numerical stability
                probs = np.exp(log_probs)
                probs /= probs.sum()

                chosen_idx = _categorical(probs.tolist())
                labels[i] = candidates[chosen_idx]

            # Compute log-likelihood
            if iteration >= burn_in:
                ll = self._total_log_likelihood(X, labels, d)
                if ll > best_ll:
                    best_ll = ll
                    best_labels = labels.copy()

        # Extract final parameters from best labeling
        labels = best_labels
        unique = np.unique(labels)
        n_clusters = len(unique)

        # Relabel contiguously
        label_map = {old: new for new, old in enumerate(unique)}
        labels = np.array([label_map[l] for l in labels])

        weights = np.zeros(n_clusters)
        means = np.zeros((n_clusters, d))
        covariances = np.zeros((n_clusters, d, d))

        for k in range(n_clusters):
            mask = labels == k
            weights[k] = np.sum(mask) / n
            means[k] = np.mean(X[mask], axis=0)
            if np.sum(mask) > 1:
                covariances[k] = np.cov(X[mask].T) + 1e-6 * np.eye(d)
            else:
                covariances[k] = np.eye(d)

        return DPMixtureResult(
            weights=weights,
            means=means,
            covariances=covariances,
            labels=labels,
            n_clusters=n_clusters,
            log_likelihood=best_ll,
            n_iter=max_iter,
        )

    @staticmethod
    def _marginal_log_likelihood(
        x: np.ndarray,
        cluster_data: np.ndarray,
        d: int,
    ) -> float:
        """Marginal likelihood under normal-inverse-Wishart prior.

        Simplified: uses empirical mean/var with prior regularization.
        """
        n_k = len(cluster_data)

        # Prior parameters
        mu_0 = np.zeros(d)
        kappa_0 = 0.1
        nu_0 = d + 2.0
        Psi_0 = np.eye(d)

        if n_k == 0:
            # Prior predictive
            mean = mu_0
            scale = (kappa_0 + 1) / (kappa_0 * nu_0) * Psi_0
        else:
            x_bar = np.mean(cluster_data, axis=0)
            kappa_n = kappa_0 + n_k
            mu_n = (kappa_0 * mu_0 + n_k * x_bar) / kappa_n

            S = np.zeros((d, d))
            for xi in cluster_data:
                diff = xi - x_bar
                S += np.outer(diff, diff)

            diff_mean = x_bar - mu_0
            Psi_n = Psi_0 + S + kappa_0 * n_k / kappa_n * np.outer(diff_mean, diff_mean)
            nu_n = nu_0 + n_k

            mean = mu_n
            scale = (kappa_n + 1) / (kappa_n * (nu_n - d + 1)) * Psi_n

        # Multivariate t log-density (approximate with Gaussian)
        diff = x - mean
        try:
            L = np.linalg.cholesky(scale)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff)
            maha = np.sum(solved ** 2)
        except np.linalg.LinAlgError:
            log_det = np.log(max(np.linalg.det(scale), 1e-300))
            maha = diff @ np.linalg.pinv(scale) @ diff

        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    def _total_log_likelihood(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        d: int,
    ) -> float:
        """Total log-likelihood of data given clustering."""
        ll = 0.0
        for k in np.unique(labels):
            mask = labels == k
            cluster_data = X[mask]
            mean = np.mean(cluster_data, axis=0)
            if np.sum(mask) > 1:
                cov = np.cov(cluster_data.T) + 1e-6 * np.eye(d)
            else:
                cov = np.eye(d)

            for x in cluster_data:
                diff = x - mean
                try:
                    ll -= 0.5 * (diff @ np.linalg.inv(cov) @ diff +
                                 np.log(max(np.linalg.det(cov), 1e-300)))
                except np.linalg.LinAlgError:
                    ll -= 0.5 * np.sum(diff ** 2)
        return ll


# ============================================================================
# EXPECTED CLUSTERS
# ============================================================================

def expected_clusters(alpha: float, n: int) -> float:
    """Expected number of clusters under CRP/DP.

    E[K] = Σ_{i=1}^{n} α / (α + i - 1) ≈ α log(n/α + 1) for large n.

    Args:
        alpha: Concentration parameter
        n: Number of observations

    Returns:
        Expected number of clusters
    """
    return sum(alpha / (alpha + i) for i in range(n))


# ============================================================================
# UTILITIES
# ============================================================================

def _categorical(probs: list[float]) -> int:
    """Sample from categorical distribution."""
    u = np.random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if u < cumsum:
            return i
    return len(probs) - 1
