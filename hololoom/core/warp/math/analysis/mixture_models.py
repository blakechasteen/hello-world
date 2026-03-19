"""
Mixture Models — Gaussian Mixture EM and Bayesian Nonparametric Mixtures
========================================================================

Expectation-Maximization for Gaussian mixture models with full
and diagonal covariance options. Bayesian variant with Dirichlet
prior for automatic component selection.

Core Concepts:
- E-step: Compute responsibilities (posterior component membership)
- M-step: Update parameters (weights, means, covariances)
- BIC: Bayesian Information Criterion for model selection
- Dirichlet prior: Automatic component pruning

Mathematical Foundation:
E-step: γ_{nk} = π_k N(x_n | μ_k, Σ_k) / Σ_j π_j N(x_n | μ_j, Σ_j)
M-step: N_k = Σ_n γ_{nk},  μ_k = Σ_n γ_{nk} x_n / N_k
        Σ_k = Σ_n γ_{nk} (x_n - μ_k)(x_n - μ_k)ᵀ / N_k
        π_k = N_k / N
BIC: -2 ln L + k ln N

Applications to Warp Space:
- Clustering embeddings into semantic groups
- Density estimation for Thompson Sampling priors
- Detecting multi-modal distributions in warp tensions

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MixtureResult:
    """Result of mixture model fitting.

    Attributes:
        weights: Component weights π_k (n_components,)
        means: Component means μ_k (n_components, n_features)
        covariances: Component covariances Σ_k (n_components, n_features, n_features)
        labels: Hard cluster assignments (n_samples,)
        responsibilities: Soft assignments γ_{nk} (n_samples, n_components)
        log_likelihood: Final log-likelihood
        bic: Bayesian Information Criterion
        n_iter: Number of EM iterations
        converged: Whether EM converged
    """
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    labels: np.ndarray
    responsibilities: np.ndarray
    log_likelihood: float = 0.0
    bic: float = 0.0
    n_iter: int = 0
    converged: bool = True


# ============================================================================
# UTILITIES
# ============================================================================

def _log_gaussian_pdf(
    X: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Compute log N(x | mean, cov) for each sample.

    Args:
        X: Data (n_samples, n_features)
        mean: Mean vector (n_features,)
        cov: Covariance matrix (n_features, n_features)

    Returns:
        Log density values (n_samples,)
    """
    n = X.shape[1]
    diff = X - mean

    try:
        L = np.linalg.cholesky(cov)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        solved = np.linalg.solve(L, diff.T)  # (n, n_samples)
        maha = np.sum(solved ** 2, axis=0)  # Mahalanobis distance squared
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse
        log_det = np.log(max(np.linalg.det(cov), 1e-300))
        cov_inv = np.linalg.pinv(cov)
        maha = np.sum(diff @ cov_inv * diff, axis=1)

    return -0.5 * (n * np.log(2 * np.pi) + log_det + maha)


# ============================================================================
# GAUSSIAN MIXTURE MODEL
# ============================================================================

class GaussianMixture:
    """Gaussian Mixture Model with EM algorithm.

    Supports full and diagonal covariance types. Uses
    k-means++ initialization for robust convergence.

    Example:
        >>> gmm = GaussianMixture()
        >>> X = np.vstack([np.random.randn(50, 2) + [3, 0],
        ...                np.random.randn(50, 2) + [-3, 0]])
        >>> result = gmm.fit(X, n_components=2)
        >>> result.labels  # cluster assignments
    """

    def fit(
        self,
        X: np.ndarray,
        n_components: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        n_init: int = 1,
    ) -> MixtureResult:
        """Fit GMM via EM algorithm.

        Args:
            X: Data matrix (n_samples, n_features)
            n_components: Number of Gaussian components
            max_iter: Maximum EM iterations
            tol: Convergence threshold on log-likelihood change
            covariance_type: "full" or "diagonal"
            reg_covar: Regularization added to covariance diagonal
            n_init: Number of initializations (best kept)

        Returns:
            MixtureResult with fitted parameters
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        best_result = None
        best_ll = -np.inf

        for _ in range(n_init):
            result = self._fit_single(
                X, n_components, max_iter, tol, covariance_type, reg_covar
            )
            if result.log_likelihood > best_ll:
                best_ll = result.log_likelihood
                best_result = result

        return best_result

    def _fit_single(
        self,
        X: np.ndarray,
        K: int,
        max_iter: int,
        tol: float,
        cov_type: str,
        reg_covar: float,
    ) -> MixtureResult:
        """Single EM run."""
        n, d = X.shape

        # K-means++ initialization
        means = self._kmeans_init(X, K)
        covariances = np.array([np.eye(d) for _ in range(K)])
        weights = np.ones(K) / K

        prev_ll = -np.inf
        converged = False

        for iteration in range(max_iter):
            # E-step: compute responsibilities
            log_resp = np.zeros((n, K))
            for k in range(K):
                log_resp[:, k] = np.log(weights[k] + 1e-300) + \
                    _log_gaussian_pdf(X, means[k], covariances[k])

            # Log-sum-exp for normalization
            log_resp_max = np.max(log_resp, axis=1, keepdims=True)
            log_resp_norm = log_resp - log_resp_max
            resp = np.exp(log_resp_norm)
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp = resp / (resp_sum + 1e-300)

            # Log-likelihood
            ll = np.sum(log_resp_max.ravel() + np.log(resp_sum.ravel() + 1e-300))

            # Convergence check
            if abs(ll - prev_ll) < tol:
                converged = True
                break
            prev_ll = ll

            # M-step: update parameters
            Nk = resp.sum(axis=0)  # Effective count per component

            for k in range(K):
                if Nk[k] < 1e-10:
                    # Re-initialize dead component
                    means[k] = X[np.random.randint(n)]
                    covariances[k] = np.eye(d)
                    weights[k] = 1.0 / K
                    continue

                # Update mean
                means[k] = (resp[:, k:k + 1].T @ X).ravel() / Nk[k]

                # Update covariance
                diff = X - means[k]
                if cov_type == "diagonal":
                    covariances[k] = np.diag(
                        (resp[:, k] * diff.T @ diff.diagonal()) / Nk[k]
                        + reg_covar
                    )
                else:
                    covariances[k] = (resp[:, k:k + 1] * diff).T @ diff / Nk[k]
                    covariances[k] += reg_covar * np.eye(d)

                # Update weight
                weights[k] = Nk[k] / n

        labels = np.argmax(resp, axis=1)

        # BIC
        if cov_type == "diagonal":
            n_params = K * (1 + d + d) - 1
        else:
            n_params = K * (1 + d + d * (d + 1) / 2) - 1
        bic = -2 * ll + n_params * np.log(n)

        return MixtureResult(
            weights=weights,
            means=means,
            covariances=covariances,
            labels=labels,
            responsibilities=resp,
            log_likelihood=ll,
            bic=bic,
            n_iter=iteration + 1,
            converged=converged,
        )

    @staticmethod
    def _kmeans_init(X: np.ndarray, K: int) -> np.ndarray:
        """K-means++ initialization for means."""
        n = X.shape[0]
        centers = [X[np.random.randint(n)]]

        for _ in range(1, K):
            dists = np.array([
                np.min([np.sum((x - c) ** 2) for c in centers])
                for x in X
            ])
            probs = dists / (dists.sum() + 1e-300)
            idx = np.random.choice(n, p=probs)
            centers.append(X[idx])

        return np.array(centers)

    def predict(
        self,
        X: np.ndarray,
        result: MixtureResult,
    ) -> np.ndarray:
        """Predict cluster labels for new data.

        Args:
            X: New data (n_samples, n_features)
            result: Fitted MixtureResult

        Returns:
            Cluster labels (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        K = len(result.weights)

        log_resp = np.zeros((len(X), K))
        for k in range(K):
            log_resp[:, k] = np.log(result.weights[k] + 1e-300) + \
                _log_gaussian_pdf(X, result.means[k], result.covariances[k])

        return np.argmax(log_resp, axis=1)


# ============================================================================
# BAYESIAN GAUSSIAN MIXTURE (Dirichlet Process approximation)
# ============================================================================

class BayesianGaussianMixture:
    """Bayesian Gaussian Mixture with Dirichlet prior.

    Automatic component selection: starts with max_components,
    prunes components with negligible weight via the Dirichlet
    concentration parameter.

    Example:
        >>> bgmm = BayesianGaussianMixture()
        >>> result = bgmm.fit(X, max_components=10)
        >>> effective_k = np.sum(result.weights > 0.01)
    """

    def fit(
        self,
        X: np.ndarray,
        max_components: int = 10,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
    ) -> MixtureResult:
        """Fit Bayesian GMM with Dirichlet prior on weights.

        Args:
            X: Data matrix (n_samples, n_features)
            max_components: Upper bound on components
            alpha: Dirichlet concentration parameter (smaller = sparser)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            reg_covar: Covariance regularization

        Returns:
            MixtureResult (inactive components have near-zero weight)
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        K = max_components

        # Initialize with k-means++
        means = GaussianMixture._kmeans_init(X, K)
        covariances = np.array([np.eye(d) for _ in range(K)])
        weights = np.ones(K) / K

        # Dirichlet prior parameters
        alpha_k = np.ones(K) * alpha / K

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step
            log_resp = np.zeros((n, K))
            for k in range(K):
                log_resp[:, k] = np.log(weights[k] + 1e-300) + \
                    _log_gaussian_pdf(X, means[k], covariances[k])

            log_resp_max = np.max(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_resp_max)
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp = resp / (resp_sum + 1e-300)

            ll = np.sum(log_resp_max.ravel() + np.log(resp_sum.ravel() + 1e-300))

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # M-step with Dirichlet prior
            Nk = resp.sum(axis=0)

            for k in range(K):
                if Nk[k] < 1e-10:
                    means[k] = X[np.random.randint(n)]
                    covariances[k] = np.eye(d)
                    continue

                means[k] = (resp[:, k:k + 1].T @ X).ravel() / Nk[k]
                diff = X - means[k]
                covariances[k] = (resp[:, k:k + 1] * diff).T @ diff / Nk[k]
                covariances[k] += reg_covar * np.eye(d)

            # Weight update with Dirichlet prior
            weights = (Nk + alpha_k - 1)
            weights = np.maximum(weights, 0)
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum
            else:
                weights = np.ones(K) / K

        labels = np.argmax(resp, axis=1)

        # BIC
        active = np.sum(weights > 0.01)
        n_params = active * (1 + d + d * (d + 1) / 2) - 1
        bic = -2 * ll + max(n_params, 1) * np.log(n)

        return MixtureResult(
            weights=weights,
            means=means,
            covariances=covariances,
            labels=labels,
            responsibilities=resp,
            log_likelihood=ll,
            bic=bic,
            n_iter=iteration + 1,
            converged=True,
        )
