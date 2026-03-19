"""
Copulas — Modeling Dependence Structure
========================================

Copulas separate marginal distributions from dependence structure.
By Sklar's theorem, any joint distribution can be decomposed into
marginals + a copula.

Core Concepts:
- Gaussian Copula: Dependence via correlation matrix of normal quantiles
- Clayton Copula: Lower tail dependence (simultaneous crashes)
- Frank Copula: Symmetric dependence (no tail dependence)
- Kendall's Tau: Rank correlation (copula-invariant)

Mathematical Foundation:
Sklar's: F(x₁,..,xₙ) = C(F₁(x₁),..,Fₙ(xₙ))
Gaussian: C(u,v) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)
Clayton: C(u,v) = (u⁻θ + v⁻θ - 1)^{-1/θ}
Frank: C(u,v) = -θ⁻¹ ln(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1))

Applications to Warp Space:
- Modeling joint dependencies between scoring terms
- Tail risk analysis for Thompson Sampling exploration
- Correlated memory retrieval patterns

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm as scipy_norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CopulaResult:
    """Result of copula fitting.

    Attributes:
        parameters: Copula parameters (dict)
        log_likelihood: Copula log-likelihood (not marginals)
        aic: Akaike Information Criterion
        family: Copula family name
    """
    parameters: dict
    log_likelihood: float = 0.0
    aic: float = 0.0
    family: str = ""


# ============================================================================
# UTILITIES
# ============================================================================

def _rank_transform(X: np.ndarray) -> np.ndarray:
    """Transform data to pseudo-observations (ranks / (n+1))."""
    n = X.shape[0]
    U = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        order = np.argsort(np.argsort(X[:, j]))
        U[:, j] = (order + 1) / (n + 1)
    return U


def _normal_ppf(u: np.ndarray) -> np.ndarray:
    """Inverse normal CDF (probit function)."""
    if HAS_SCIPY:
        return scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    result = np.zeros_like(u)
    for idx in np.ndindex(u.shape):
        p = u[idx]
        sign = 1.0
        if p < 0.5:
            p = 1 - p
            sign = -1.0
        t = np.sqrt(-2 * np.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        result[idx] = sign * (t - (c0 + c1 * t + c2 * t ** 2) /
                              (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3))
    return result


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


# ============================================================================
# KENDALL'S TAU
# ============================================================================

def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall's rank correlation coefficient.

    τ = (concordant - discordant) / (n choose 2)

    Args:
        x, y: Two arrays of equal length

    Returns:
        Kendall's tau in [-1, 1]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    denom = n * (n - 1) / 2
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


# ============================================================================
# GAUSSIAN COPULA
# ============================================================================

class GaussianCopula:
    """Gaussian (Normal) copula.

    Dependence is captured by the correlation matrix of the
    normal quantile transforms.

    C(u₁,..,uₙ) = Φₙ(Φ⁻¹(u₁),..,Φ⁻¹(uₙ); Σ)

    Example:
        >>> gc = GaussianCopula()
        >>> X = np.random.randn(200, 2)  # Standard normal, correlated
        >>> X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]
        >>> result = gc.fit(X)
        >>> abs(result.parameters['correlation'][0, 1] - 0.8) < 0.1
        True
    """

    def fit(self, X: np.ndarray) -> CopulaResult:
        """Fit Gaussian copula to data.

        Args:
            X: Data matrix (n_samples, n_features)

        Returns:
            CopulaResult with correlation matrix
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        # Transform to pseudo-observations
        U = _rank_transform(X)

        # Transform to normal scores
        Z = _normal_ppf(U)

        # Estimate correlation matrix
        R = np.corrcoef(Z.T)

        # Copula log-likelihood
        ll = self._log_likelihood(Z, R)

        # AIC: k = d*(d-1)/2 parameters
        n_params = d * (d - 1) // 2
        aic = -2 * ll + 2 * n_params

        return CopulaResult(
            parameters={'correlation': R},
            log_likelihood=ll,
            aic=aic,
            family='gaussian',
        )

    @staticmethod
    def _log_likelihood(Z: np.ndarray, R: np.ndarray) -> float:
        """Gaussian copula log-likelihood (copula density only)."""
        n, d = Z.shape
        try:
            R_inv = np.linalg.inv(R)
            log_det_R = np.log(max(np.linalg.det(R), 1e-300))
        except np.linalg.LinAlgError:
            return -np.inf

        ll = 0.0
        for i in range(n):
            z = Z[i]
            ll += -0.5 * (z @ (R_inv - np.eye(d)) @ z + log_det_R)

        return ll

    def sample(
        self,
        n_samples: int,
        correlation: np.ndarray,
    ) -> np.ndarray:
        """Sample from the Gaussian copula (uniform marginals).

        Args:
            n_samples: Number of samples
            correlation: Correlation matrix

        Returns:
            Samples in [0, 1]^d (n_samples, d)
        """
        R = np.asarray(correlation, dtype=float)
        d = R.shape[0]

        Z = np.random.multivariate_normal(np.zeros(d), R, size=n_samples)

        # Transform to uniform via Φ
        if HAS_SCIPY:
            U = scipy_norm.cdf(Z)
        else:
            U = np.zeros_like(Z)
            for i in range(n_samples):
                for j in range(d):
                    U[i, j] = 0.5 * (1 + np.tanh(Z[i, j] * 0.7978845608))
        return U


# ============================================================================
# CLAYTON COPULA
# ============================================================================

class ClaytonCopula:
    """Clayton copula — lower tail dependence.

    C(u, v) = (u⁻θ + v⁻θ - 1)^{-1/θ}  for θ > 0

    Lower tail dependence: λ_L = 2^{-1/θ}
    Upper tail dependence: λ_U = 0

    Good for modeling joint crashes / simultaneous failures.

    Example:
        >>> cc = ClaytonCopula()
        >>> result = cc.fit(X)
        >>> result.parameters['theta'] > 0
        True
    """

    def fit(self, X: np.ndarray) -> CopulaResult:
        """Fit Clayton copula (bivariate).

        Uses Kendall's tau: θ = 2τ / (1 - τ)

        Args:
            X: Data matrix (n_samples, 2)

        Returns:
            CopulaResult with theta parameter
        """
        X = np.asarray(X, dtype=float)
        if X.shape[1] != 2:
            raise ValueError("Clayton copula only supports bivariate data")

        U = _rank_transform(X)

        # Method of moments via Kendall's tau
        tau = kendall_tau(X[:, 0], X[:, 1])

        if tau <= 0:
            theta = 0.01  # Near independence
        else:
            theta = 2 * tau / (1 - tau)

        theta = max(theta, 0.01)  # Must be positive

        # Log-likelihood
        ll = self._log_likelihood(U, theta)
        aic = -2 * ll + 2  # 1 parameter

        return CopulaResult(
            parameters={'theta': theta, 'lower_tail_dependence': 2 ** (-1 / theta)},
            log_likelihood=ll,
            aic=aic,
            family='clayton',
        )

    @staticmethod
    def _log_likelihood(U: np.ndarray, theta: float) -> float:
        """Clayton copula log-likelihood."""
        u, v = U[:, 0], U[:, 1]
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        n = len(u)
        ll = n * np.log(1 + theta)
        ll += -(1 + theta) * np.sum(np.log(u) + np.log(v))
        ll += -(2 + 1 / theta) * np.sum(np.log(u ** (-theta) + v ** (-theta) - 1))

        return ll


# ============================================================================
# FRANK COPULA
# ============================================================================

class FrankCopula:
    """Frank copula — symmetric dependence, no tail dependence.

    C(u, v) = -θ⁻¹ ln(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1))

    Symmetric: λ_L = λ_U = 0
    Can model both positive and negative dependence.

    Example:
        >>> fc = FrankCopula()
        >>> result = fc.fit(X)
        >>> result.parameters['theta']
    """

    def fit(self, X: np.ndarray) -> CopulaResult:
        """Fit Frank copula (bivariate).

        Uses numerical inversion of Kendall's tau relationship.

        Args:
            X: Data matrix (n_samples, 2)

        Returns:
            CopulaResult with theta parameter
        """
        X = np.asarray(X, dtype=float)
        if X.shape[1] != 2:
            raise ValueError("Frank copula only supports bivariate data")

        U = _rank_transform(X)

        # Kendall's tau
        tau = kendall_tau(X[:, 0], X[:, 1])

        # Invert: τ = 1 - 4/θ [1 - D₁(θ)]  where D₁ is first Debye function
        # Use grid search for simplicity
        theta = self._invert_tau(tau)

        ll = self._log_likelihood(U, theta)
        aic = -2 * ll + 2

        return CopulaResult(
            parameters={'theta': theta},
            log_likelihood=ll,
            aic=aic,
            family='frank',
        )

    @staticmethod
    def _debye1(theta: float, n_terms: int = 100) -> float:
        """First Debye function D₁(θ) = (1/θ) ∫₀^θ t/(e^t - 1) dt."""
        if abs(theta) < 1e-10:
            return 1.0
        # Numerical integration via trapezoidal
        t = np.linspace(1e-10, abs(theta), n_terms)
        dt = t[1] - t[0]
        integrand = t / (np.exp(t) - 1 + 1e-300)
        return np.sum(integrand) * dt / abs(theta)

    def _invert_tau(self, tau: float) -> float:
        """Find theta such that Kendall's tau relationship holds."""
        # Grid search over theta
        best_theta = 1.0
        best_err = np.inf

        for theta_candidate in np.concatenate([
            np.linspace(-30, -0.1, 50),
            np.linspace(0.1, 30, 50),
        ]):
            D1 = self._debye1(theta_candidate)
            tau_pred = 1 - 4 / theta_candidate * (1 - D1)
            err = abs(tau_pred - tau)
            if err < best_err:
                best_err = err
                best_theta = theta_candidate

        return best_theta

    @staticmethod
    def _log_likelihood(U: np.ndarray, theta: float) -> float:
        """Frank copula log-likelihood."""
        u, v = U[:, 0], U[:, 1]
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        if abs(theta) < 1e-10:
            return 0.0  # Independence

        et = np.exp(-theta)
        etu = np.exp(-theta * u)
        etv = np.exp(-theta * v)

        numer = -theta * (et - 1) * np.exp(-theta * (u + v))
        denom = ((et - 1) + (etu - 1) * (etv - 1)) ** 2

        # Clamp for numerical safety
        ratio = np.clip(numer / (denom + 1e-300), 1e-300, None)

        return float(np.sum(np.log(np.abs(ratio))))
