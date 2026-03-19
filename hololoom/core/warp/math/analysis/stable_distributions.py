"""
Stable Distributions & Advanced Statistical Methods
====================================================

Heavy-tailed distributions, variational inference, empirical Bayes,
canonical correlation, MDS, kernel density, rank tests, and
differential privacy mechanisms.

Core Concepts:
- Stable distributions: Generalize Gaussian to heavy tails (Levy, Cauchy)
- Variational inference: Approximate Bayesian posterior via optimization
- Empirical Bayes: Estimate prior from data via marginal likelihood
- CCA: Find maximally correlated linear projections of two datasets
- MDS: Embed distance matrix into low-dimensional Euclidean space
- Kernel density: Nonparametric density estimation
- Rank tests: Distribution-free hypothesis tests
- Differential privacy: Mathematical privacy guarantees

Mathematical Foundation:
Stable(α, β, σ, μ): characteristic function φ(t) = exp(iμt - σ^α|t|^α [1 + iβ sign(t) Φ])
ELBO = E_q[log p(x,z)] - E_q[log q(z)] ≤ log p(x)
CCA: max_a,b corr(a'X, b'Y) via SVD of Σ_XX^{-1/2} Σ_XY Σ_YY^{-1/2}

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# STABLE DISTRIBUTION
# ============================================================================

class StableDistribution:
    """
    α-stable distribution via the Chambers-Mallows-Stuck algorithm.

    The stable distribution Stable(α, β, σ, μ) is the only possible
    limit distribution for normalized sums of i.i.d. random variables.
    α ∈ (0, 2] controls tail heaviness (α=2 is Gaussian, α=1 is Cauchy).
    β ∈ [-1, 1] controls skewness.
    """

    @staticmethod
    def sample(alpha: float, beta: float, scale: float = 1.0,
               loc: float = 0.0, n: int = 1) -> np.ndarray:
        """
        Sample from Stable(α, β, σ, μ) using Chambers-Mallows-Stuck.

        Args:
            alpha: Stability parameter (0 < α ≤ 2).
            beta: Skewness parameter (-1 ≤ β ≤ 1).
            scale: Scale parameter (σ > 0).
            loc: Location parameter (μ).
            n: Number of samples.

        Returns:
            Array of n samples.
        """
        if alpha <= 0 or alpha > 2:
            raise ValueError("alpha must be in (0, 2]")
        if beta < -1 or beta > 1:
            raise ValueError("beta must be in [-1, 1]")
        if scale <= 0:
            raise ValueError("scale must be positive")

        if abs(alpha - 2.0) < 1e-10:
            # Gaussian case
            return loc + scale * np.random.normal(0, np.sqrt(2), n)

        # Chambers-Mallows-Stuck algorithm
        U = np.random.uniform(-np.pi / 2, np.pi / 2, n)
        W = np.random.exponential(1.0, n)

        if abs(alpha - 1.0) < 1e-10:
            # Cauchy-like case (α = 1)
            xi = np.pi / 2
            X = ((np.pi / 2 + beta * U) * np.tan(U) -
                 beta * np.log(np.pi / 2 * W * np.cos(U) /
                               (np.pi / 2 + beta * U + 1e-30) + 1e-30))
            X = scale * X + loc + beta * scale * 2 / np.pi * np.log(scale)
        else:
            # General case
            zeta = -beta * np.tan(np.pi * alpha / 2)
            xi = np.arctan(-zeta) / alpha

            factor1 = np.cos(U) ** (-1.0 / alpha)
            factor2 = (np.cos(U - alpha * (U + xi))) ** (1.0 / alpha)
            # Avoid numerical issues
            arg = np.clip(alpha * (U + xi), -np.pi / 2 + 1e-10,
                          np.pi / 2 - 1e-10)

            S = (np.sin(arg) /
                 (np.cos(U) ** (1.0 / alpha)) *
                 (np.cos(U - arg) / (W + 1e-30)) ** ((1 - alpha) / alpha))

            X = scale * S + loc

        return X

    @staticmethod
    def characteristic_function(alpha: float, beta: float,
                                scale: float, loc: float,
                                t: np.ndarray) -> np.ndarray:
        """
        Evaluate the characteristic function φ(t) of Stable(α, β, σ, μ).

        φ(t) = exp(iμt - σ^α|t|^α (1 - iβ sign(t) tan(πα/2)))  for α ≠ 1
        φ(t) = exp(iμt - σ|t| (1 + iβ (2/π) sign(t) ln|t|))     for α = 1
        """
        abs_t = np.abs(t) + 1e-30
        sign_t = np.sign(t)

        if abs(alpha - 1.0) < 1e-10:
            phi_factor = -scale * abs_t * (
                1 + 1j * beta * 2 / np.pi * sign_t * np.log(abs_t)
            )
        else:
            phi_factor = -(scale ** alpha) * (abs_t ** alpha) * (
                1 - 1j * beta * sign_t * np.tan(np.pi * alpha / 2)
            )

        return np.exp(1j * loc * t + phi_factor)


# ============================================================================
# VARIATIONAL INFERENCE
# ============================================================================

class VariationalInference:
    """
    Variational inference: approximate posterior q(z) ≈ p(z|x)
    by maximizing the Evidence Lower Bound (ELBO).

    ELBO(q) = E_q[log p(x, z)] - E_q[log q(z)]
            = log p(x) - KL(q || p(z|x))
    """

    @staticmethod
    def fit(log_joint: Callable[[np.ndarray, np.ndarray], float],
            q_family: str = 'gaussian',
            n_iter: int = 200,
            n_samples: int = 50,
            lr: float = 0.01,
            dim: int = 1) -> dict:
        """
        Fit variational posterior by maximizing ELBO.

        Uses black-box variational inference with reparameterization trick
        (for Gaussian q).

        Args:
            log_joint: Function (z, x) -> log p(x, z). Takes latent z and data x.
            q_family: Variational family ('gaussian').
            n_iter: Number of optimization iterations.
            n_samples: Monte Carlo samples for ELBO estimation.
            lr: Learning rate for gradient ascent.
            dim: Dimensionality of latent space.

        Returns:
            Dict with keys: mu, sigma, elbo_history.
        """
        if q_family != 'gaussian':
            raise ValueError("Only 'gaussian' q_family supported")

        # Initialize q(z) = N(mu, diag(sigma^2))
        mu = np.zeros(dim)
        log_sigma = np.zeros(dim)

        elbo_history = []
        x_dummy = np.zeros(1)  # Placeholder for observed data

        for it in range(n_iter):
            sigma = np.exp(log_sigma)

            # Reparameterization: z = mu + sigma * epsilon
            epsilon = np.random.randn(n_samples, dim)
            z_samples = mu + sigma * epsilon

            # Estimate ELBO and gradients
            elbo = 0.0
            grad_mu = np.zeros(dim)
            grad_log_sigma = np.zeros(dim)

            for s in range(n_samples):
                z = z_samples[s]
                lj = log_joint(z, x_dummy)
                lq = -0.5 * np.sum((z - mu) ** 2 / (sigma ** 2 + 1e-10)) - \
                    np.sum(log_sigma) - 0.5 * dim * np.log(2 * np.pi)

                score = lj - lq
                elbo += score

                # Score function gradients (REINFORCE with baseline)
                grad_mu += (z - mu) / (sigma ** 2 + 1e-10) * score
                grad_log_sigma += ((z - mu) ** 2 / (sigma ** 2 + 1e-10) - 1) * score

            elbo /= n_samples
            grad_mu /= n_samples
            grad_log_sigma /= n_samples

            # Gradient ascent
            mu += lr * grad_mu
            log_sigma += lr * 0.1 * grad_log_sigma  # Slower lr for variance

            elbo_history.append(float(elbo))

        return {
            'mu': mu.copy(),
            'sigma': np.exp(log_sigma).copy(),
            'elbo_history': np.array(elbo_history),
        }


# ============================================================================
# EMPIRICAL BAYES
# ============================================================================

class EmpiricalBayes:
    """
    Empirical Bayes: estimate the prior from data via marginal likelihood.

    Type-II maximum likelihood: choose prior parameters θ to maximize
    p(x | θ) = ∫ p(x | z) p(z | θ) dz.
    """

    @staticmethod
    def estimate_prior(data: np.ndarray,
                       likelihood_fn: Callable[[np.ndarray, np.ndarray], float],
                       prior_family: str = 'gaussian',
                       n_iter: int = 100,
                       lr: float = 0.01) -> dict:
        """
        Estimate prior parameters via marginal likelihood maximization.

        For Gaussian prior N(μ₀, σ₀²) with Gaussian likelihood:
        analytically tractable. For general case, uses gradient ascent.

        Args:
            data: Observed data (n_samples, dim) or (n_samples,).
            likelihood_fn: Function (data_point, z) -> log p(x_i | z).
            prior_family: 'gaussian' for Normal-Normal model.
            n_iter: Optimization iterations.
            lr: Learning rate.

        Returns:
            Dict with estimated prior parameters.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n, dim = data.shape

        if prior_family == 'gaussian':
            # Initialize prior: N(mean(data), var(data))
            mu0 = np.mean(data, axis=0)
            sigma0 = np.std(data, axis=0) + 1e-6

            for it in range(n_iter):
                # Compute marginal log-likelihood gradient numerically
                eps = 1e-5

                # Gradient w.r.t. mu0
                grad_mu = np.zeros(dim)
                for d in range(dim):
                    mu_plus = mu0.copy()
                    mu_plus[d] += eps
                    mu_minus = mu0.copy()
                    mu_minus[d] -= eps

                    ll_plus = EmpiricalBayes._marginal_ll(
                        data, likelihood_fn, mu_plus, sigma0
                    )
                    ll_minus = EmpiricalBayes._marginal_ll(
                        data, likelihood_fn, mu_minus, sigma0
                    )
                    grad_mu[d] = (ll_plus - ll_minus) / (2 * eps)

                # Gradient w.r.t. sigma0
                grad_sigma = np.zeros(dim)
                for d in range(dim):
                    s_plus = sigma0.copy()
                    s_plus[d] += eps
                    s_minus = sigma0.copy()
                    s_minus[d] = max(eps, s_minus[d] - eps)

                    ll_plus = EmpiricalBayes._marginal_ll(
                        data, likelihood_fn, mu0, s_plus
                    )
                    ll_minus = EmpiricalBayes._marginal_ll(
                        data, likelihood_fn, mu0, s_minus
                    )
                    grad_sigma[d] = (ll_plus - ll_minus) / (2 * eps)

                mu0 += lr * grad_mu
                sigma0 += lr * 0.1 * grad_sigma
                sigma0 = np.maximum(sigma0, 1e-6)

            return {
                'mu': mu0,
                'sigma': sigma0,
                'family': 'gaussian',
            }

        raise ValueError(f"Unknown prior family: {prior_family}")

    @staticmethod
    def _marginal_ll(data: np.ndarray,
                     likelihood_fn: Callable,
                     mu0: np.ndarray,
                     sigma0: np.ndarray,
                     n_mc: int = 30) -> float:
        """Estimate marginal log-likelihood via Monte Carlo."""
        n = data.shape[0]
        dim = data.shape[1]
        total = 0.0

        for i in range(n):
            # Monte Carlo estimate of p(x_i | θ) = ∫ p(x_i | z) p(z | θ) dz
            z_samples = mu0 + sigma0 * np.random.randn(n_mc, dim)
            log_likes = np.array([
                likelihood_fn(data[i], z_samples[s])
                for s in range(n_mc)
            ])
            # Log-sum-exp trick
            max_ll = np.max(log_likes)
            total += max_ll + np.log(np.mean(np.exp(log_likes - max_ll)))

        return total


# ============================================================================
# CANONICAL CORRELATION ANALYSIS
# ============================================================================

@dataclass
class CCAResult:
    """Result of Canonical Correlation Analysis."""
    correlations: np.ndarray      # Canonical correlations
    x_weights: np.ndarray         # Weights for X (p x d)
    y_weights: np.ndarray         # Weights for Y (q x d)
    x_scores: np.ndarray          # Canonical variates for X (n x d)
    y_scores: np.ndarray          # Canonical variates for Y (n x d)


class CanonicalCorrelation:
    """
    Canonical Correlation Analysis (CCA) via SVD.

    Finds linear projections of X and Y that maximize correlation:
        max_{a, b} corr(a'X, b'Y)
    """

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray) -> CCAResult:
        """
        Compute CCA between X (n x p) and Y (n x q).

        Method: SVD of Σ_XX^{-1/2} Σ_XY Σ_YY^{-1/2}.

        Args:
            X: First dataset (n x p).
            Y: Second dataset (n x q).

        Returns:
            CCAResult with canonical correlations and weights.
        """
        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError("X and Y must have same number of rows")

        # Center
        Xc = X - np.mean(X, axis=0)
        Yc = Y - np.mean(Y, axis=0)

        # Covariance matrices
        Sxx = (Xc.T @ Xc) / (n - 1) + 1e-8 * np.eye(X.shape[1])
        Syy = (Yc.T @ Yc) / (n - 1) + 1e-8 * np.eye(Y.shape[1])
        Sxy = (Xc.T @ Yc) / (n - 1)

        # Inverse square roots via eigendecomposition
        def inv_sqrt(M):
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        Sxx_inv_sqrt = inv_sqrt(Sxx)
        Syy_inv_sqrt = inv_sqrt(Syy)

        # SVD of the cross-correlation matrix
        M = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        d = min(X.shape[1], Y.shape[1])
        correlations = np.clip(s[:d], 0, 1)

        x_weights = Sxx_inv_sqrt @ U[:, :d]
        y_weights = Syy_inv_sqrt @ Vt[:d].T

        x_scores = Xc @ x_weights
        y_scores = Yc @ y_weights

        return CCAResult(
            correlations=correlations,
            x_weights=x_weights,
            y_weights=y_weights,
            x_scores=x_scores,
            y_scores=y_scores,
        )


# ============================================================================
# MULTIDIMENSIONAL SCALING
# ============================================================================

class MDS:
    """
    Classical Multidimensional Scaling (Torgerson MDS).

    Embeds a distance matrix into Euclidean space by eigendecomposition
    of the double-centered squared distance matrix.
    """

    @staticmethod
    def fit(distance_matrix: np.ndarray,
            n_components: int = 2) -> np.ndarray:
        """
        Embed distance matrix into n_components-dimensional space.

        B = -0.5 * J * D^2 * J, where J = I - (1/n)11'.
        Then X = V_k Λ_k^{1/2} where (Λ_k, V_k) are top-k eigenpairs of B.

        Args:
            distance_matrix: Symmetric (n x n) distance/dissimilarity matrix.
            n_components: Target dimensionality.

        Returns:
            Embedding coordinates (n x n_components).
        """
        n = distance_matrix.shape[0]
        D_sq = distance_matrix ** 2

        # Double centering
        row_mean = np.mean(D_sq, axis=1, keepdims=True)
        col_mean = np.mean(D_sq, axis=0, keepdims=True)
        total_mean = np.mean(D_sq)
        B = -0.5 * (D_sq - row_mean - col_mean + total_mean)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(B)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take top k components
        k = min(n_components, np.sum(eigvals > 1e-10))
        if k < n_components:
            logger.warning(
                f"Only {k} positive eigenvalues; "
                f"embedding in {k}D instead of {n_components}D"
            )

        coords = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0))

        # Pad with zeros if needed
        if k < n_components:
            coords = np.hstack([
                coords,
                np.zeros((n, n_components - k))
            ])

        return coords

    @staticmethod
    def stress(distance_matrix: np.ndarray,
               embedding: np.ndarray) -> float:
        """
        Kruskal's stress-1: measures embedding quality.

        stress = sqrt(Σ (d_ij - δ_ij)² / Σ d_ij²)
        """
        n = distance_matrix.shape[0]

        embedded_dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(embedding[i] - embedding[j])
                embedded_dists[i, j] = d
                embedded_dists[j, i] = d

        numer = np.sum((distance_matrix - embedded_dists) ** 2)
        denom = np.sum(distance_matrix ** 2)

        if denom < 1e-14:
            return 0.0
        return float(np.sqrt(numer / denom))


# ============================================================================
# KERNEL DENSITY ESTIMATION (1D)
# ============================================================================

class KernelDensity1D:
    """
    One-dimensional kernel density estimation.

    f_hat(x) = (1/nh) Σ K((x - x_i)/h)

    where K is the kernel function and h is the bandwidth.
    """

    @staticmethod
    def fit(data: np.ndarray,
            bandwidth: float | None = None) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fit a 1D kernel density estimator.

        Uses Gaussian kernel with Silverman's rule of thumb for bandwidth
        if not specified.

        Args:
            data: 1D sample array.
            bandwidth: Bandwidth h. If None, uses Silverman's rule.

        Returns:
            Callable that evaluates the density at given points.
        """
        data = np.asarray(data).ravel()
        n = len(data)

        if bandwidth is None:
            # Silverman's rule of thumb
            sigma = np.std(data, ddof=1)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            bandwidth = 0.9 * min(sigma, iqr / 1.34) * n ** (-0.2)
            bandwidth = max(bandwidth, 1e-10)

        h = bandwidth

        def density(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x).ravel()
            result = np.zeros_like(x, dtype=float)
            for xi in data:
                u = (x - xi) / h
                result += np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
            result /= (n * h)
            return result

        return density


# ============================================================================
# RANK TESTS
# ============================================================================

class RankTest:
    """
    Distribution-free rank-based hypothesis tests.

    Sign test: test median hypothesis.
    Runs test: test randomness of a binary sequence.
    """

    @staticmethod
    def sign_test(x: np.ndarray,
                  median: float = 0.0) -> tuple[int, float]:
        """
        Sign test for median.

        H0: median(X) = m0.
        Test statistic S = number of x_i > m0.
        Under H0, S ~ Binomial(n, 0.5).

        Args:
            x: Sample data.
            median: Hypothesized median.

        Returns:
            (S, p_value): Test statistic and two-sided p-value.
        """
        diffs = x - median
        n_nonzero = np.sum(diffs != 0)
        S = int(np.sum(diffs > 0))

        if n_nonzero == 0:
            return (0, 1.0)

        # Two-sided p-value via normal approximation
        # S ~ Bin(n, 0.5), mean = n/2, var = n/4
        mu = n_nonzero / 2.0
        sigma = np.sqrt(n_nonzero / 4.0)
        z = (S - mu) / (sigma + 1e-10)
        # Two-sided p-value
        p_value = 2 * (1 - _normal_cdf(abs(z)))

        return (S, float(p_value))

    @staticmethod
    def runs_test(x: np.ndarray) -> tuple[int, float]:
        """
        Wald-Wolfowitz runs test for randomness.

        A run is a maximal sequence of consecutive identical values.
        Under H0 (random sequence), the number of runs R has known
        mean and variance.

        Args:
            x: Binary sequence (or will be binarized at median).

        Returns:
            (R, p_value): Number of runs and two-sided p-value.
        """
        # Binarize at median if not already binary
        binary = (x > np.median(x)).astype(int)
        n = len(binary)

        if n < 2:
            return (1, 1.0)

        # Count runs
        R = 1
        for i in range(1, n):
            if binary[i] != binary[i - 1]:
                R += 1

        n1 = int(np.sum(binary == 1))
        n0 = n - n1

        if n0 == 0 or n1 == 0:
            return (R, 1.0)

        # Expected runs and variance under H0
        mu_R = 1 + 2 * n0 * n1 / n
        var_R = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n ** 2 * (n - 1))

        if var_R <= 0:
            return (R, 1.0)

        z = (R - mu_R) / np.sqrt(var_R)
        p_value = 2 * (1 - _normal_cdf(abs(z)))

        return (R, float(p_value))


# ============================================================================
# DIFFERENTIAL PRIVACY
# ============================================================================

class RenyiDP:
    """
    Renyi Differential Privacy (RDP).

    (α, ε)-RDP: for all neighboring datasets D, D':
        D_α(M(D) || M(D')) ≤ ε
    where D_α is the Renyi divergence of order α.

    Gaussian mechanism achieves (α, α·Δ²/(2σ²))-RDP.
    """

    @staticmethod
    def mechanism(value: float, sensitivity: float,
                  alpha: float, epsilon: float) -> float:
        """
        Apply the Gaussian mechanism with RDP guarantee.

        σ = Δ · sqrt(α / (2ε)) achieves (α, ε)-RDP.

        Args:
            value: True value to privatize.
            sensitivity: Global sensitivity Δ of the query.
            alpha: Renyi divergence order (> 1).
            epsilon: Privacy parameter.

        Returns:
            Privatized value.
        """
        if alpha <= 1:
            raise ValueError("alpha must be > 1 for RDP")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        sigma = sensitivity * np.sqrt(alpha / (2 * epsilon))
        noise = np.random.normal(0, sigma)

        return float(value + noise)

    @staticmethod
    def compose(epsilons: list[float], alpha: float) -> float:
        """
        Composition theorem for RDP: ε_total = Σ ε_i.

        RDP composes linearly, which is tighter than (ε,δ)-DP composition.
        """
        return sum(epsilons)

    @staticmethod
    def to_dp(epsilon_rdp: float, alpha: float,
              delta: float) -> float:
        """
        Convert (α, ε)-RDP to (ε, δ)-DP.

        ε_dp = ε_rdp + log(1/δ) / (α - 1)
        """
        if alpha <= 1:
            raise ValueError("alpha must be > 1")
        return epsilon_rdp + np.log(1 / delta) / (alpha - 1)


class LocalDP:
    """
    Local Differential Privacy (LDP).

    Each user perturbs their data locally before sending to the server.
    No trusted curator needed.
    """

    @staticmethod
    def randomized_response(value: bool, epsilon: float) -> bool:
        """
        Randomized response mechanism for binary data.

        With probability p = e^ε / (1 + e^ε), report truthfully.
        With probability 1 - p, report the opposite.

        Satisfies ε-LDP.

        Args:
            value: True binary value.
            epsilon: Privacy parameter.

        Returns:
            Perturbed binary value.
        """
        p = np.exp(epsilon) / (1 + np.exp(epsilon))

        if np.random.random() < p:
            return value
        else:
            return not value

    @staticmethod
    def unary_encoding(value: int, domain_size: int,
                       epsilon: float) -> np.ndarray:
        """
        Unary encoding mechanism for categorical data.

        Encode value as one-hot, then flip each bit independently.

        Args:
            value: Category index (0 to domain_size - 1).
            domain_size: Number of categories.
            epsilon: Privacy parameter.

        Returns:
            Perturbed binary vector of length domain_size.
        """
        # One-hot encode
        encoded = np.zeros(domain_size)
        encoded[value] = 1.0

        p = np.exp(epsilon / 2) / (1 + np.exp(epsilon / 2))
        q = 1 / (1 + np.exp(epsilon / 2))

        result = np.zeros(domain_size)
        for i in range(domain_size):
            if encoded[i] == 1:
                result[i] = 1 if np.random.random() < p else 0
            else:
                result[i] = 1 if np.random.random() < q else 0

        return result

    @staticmethod
    def estimate_frequency(responses: np.ndarray,
                           epsilon: float) -> np.ndarray:
        """
        Estimate true frequencies from randomized response data.

        Corrects for the bias introduced by randomization.
        """
        p = np.exp(epsilon) / (1 + np.exp(epsilon))
        observed = np.mean(responses, axis=0)
        estimated = (observed - (1 - p)) / (2 * p - 1)
        return np.clip(estimated, 0, 1)


# ============================================================================
# HELPER
# ============================================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    if x < 0:
        return 1 - _normal_cdf(-x)
    t = 1 / (1 + 0.2316419 * x)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
            t * (-1.821255978 + t * 1.330274429))))
    return 1 - d * np.exp(-0.5 * x * x) * poly
