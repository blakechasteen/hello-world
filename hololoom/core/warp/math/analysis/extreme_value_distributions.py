"""
Extreme Value Theory — Tail Risk and Score Extremes
====================================================

Distributions governing the maxima (or minima) of large samples. Essential
for understanding tail behavior in scoring systems, anomaly detection,
and risk quantification.

Core Concepts:
- Generalized Extreme Value (GEV): Unifies Gumbel, Frechet, Weibull
- Fisher-Tippett-Gnedenko theorem: Block maxima converge to GEV
- Peaks Over Threshold (POT): Exceedances follow GPD
- Pickands-Balkema-de Haan theorem: Threshold exceedances → GPD

Mathematical Foundation:
GEV CDF:  G(x) = exp(-(1 + ξ((x-μ)/σ))^{-1/ξ})   for ξ ≠ 0
           G(x) = exp(-exp(-(x-μ)/σ))                for ξ = 0 (Gumbel)

Shape parameter ξ determines tail type:
    ξ > 0  Frechet (heavy tail, e.g. financial returns)
    ξ = 0  Gumbel  (light tail, e.g. Gaussian maxima)
    ξ < 0  Weibull (bounded tail, e.g. uniform maxima)

GPD CDF:  H(x) = 1 - (1 + ξx/σ)^{-1/ξ}  for ξ ≠ 0
           H(x) = 1 - exp(-x/σ)            for ξ = 0

Applications to Warp Space:
- Tail risk in Thompson Sampling reward distributions
- Extreme score detection in additive scoring systems
- Anomaly detection via return level exceedance
- Worst-case analysis for convergence bounds

Author: HoloLoom Team
Date: 2025-10-26
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize as scipy_minimize
    from scipy.special import gamma as scipy_gamma
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("scipy not available; MLE fitting will use gradient descent fallback")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GEVParams:
    """
    Parameters of a Generalized Extreme Value distribution.

    Attributes:
        mu: Location parameter (center of the distribution)
        sigma: Scale parameter (spread, must be > 0)
        xi: Shape parameter (tail type: >0 heavy, =0 Gumbel, <0 bounded)
    """
    mu: float
    sigma: float
    xi: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Scale parameter sigma must be positive, got {self.sigma}")

    @property
    def tail_type(self) -> str:
        if abs(self.xi) < 1e-8:
            return "gumbel"
        elif self.xi > 0:
            return "frechet"
        else:
            return "weibull"


@dataclass
class EVDResult:
    """
    Result of an extreme value analysis.

    Attributes:
        params: Fitted GEV or GPD parameters
        return_level: Estimated return level for the given period
        return_period: The period used
        log_likelihood: Log-likelihood of the fit
    """
    params: GEVParams
    return_level: float
    return_period: float
    log_likelihood: float = 0.0


# ============================================================================
# GENERALIZED EXTREME VALUE DISTRIBUTION
# ============================================================================

class GeneralizedExtremeValue:
    """
    The GEV distribution — unifying Gumbel, Frechet, and Weibull.

    By the Fisher-Tippett-Gnedenko theorem, properly normalized block maxima
    of i.i.d. random variables converge in distribution to the GEV.
    """

    @staticmethod
    def cdf(x: np.ndarray, params: GEVParams) -> np.ndarray:
        """
        GEV cumulative distribution function.

        G(x) = exp(-(1 + ξ(x-μ)/σ)^{-1/ξ})   for ξ ≠ 0
        G(x) = exp(-exp(-(x-μ)/σ))             for ξ = 0
        """
        x = np.asarray(x, dtype=float)
        z = (x - params.mu) / params.sigma

        if abs(params.xi) < 1e-8:
            # Gumbel case
            return np.exp(-np.exp(-z))

        t = 1.0 + params.xi * z
        # Support: t > 0
        valid = t > 0
        result = np.zeros_like(x)
        result[valid] = np.exp(-np.power(t[valid], -1.0 / params.xi))
        # Outside support
        if params.xi > 0:
            result[~valid] = 0.0  # Below lower endpoint
        else:
            result[~valid] = 1.0  # Above upper endpoint
        return result

    @staticmethod
    def pdf(x: np.ndarray, params: GEVParams) -> np.ndarray:
        """
        GEV probability density function.

        g(x) = (1/σ) (1 + ξz)^{-1/ξ - 1} exp(-(1 + ξz)^{-1/ξ})
        where z = (x - μ) / σ
        """
        x = np.asarray(x, dtype=float)
        z = (x - params.mu) / params.sigma

        if abs(params.xi) < 1e-8:
            # Gumbel
            exp_neg_z = np.exp(-z)
            return (1.0 / params.sigma) * exp_neg_z * np.exp(-exp_neg_z)

        t = 1.0 + params.xi * z
        valid = t > 0
        result = np.zeros_like(x)

        t_valid = t[valid]
        power = -1.0 / params.xi
        result[valid] = ((1.0 / params.sigma) *
                         np.power(t_valid, power - 1.0) *
                         np.exp(-np.power(t_valid, power)))
        return result

    @staticmethod
    def quantile(p: np.ndarray, params: GEVParams) -> np.ndarray:
        """
        GEV quantile function (inverse CDF).

        x_p = μ + (σ/ξ)((-ln p)^{-ξ} - 1)    for ξ ≠ 0
        x_p = μ - σ ln(-ln p)                   for ξ = 0
        """
        p = np.asarray(p, dtype=float)
        if np.any((p <= 0) | (p >= 1)):
            raise ValueError("Probabilities must be in (0, 1)")

        if abs(params.xi) < 1e-8:
            return params.mu - params.sigma * np.log(-np.log(p))

        return params.mu + (params.sigma / params.xi) * (
            np.power(-np.log(p), -params.xi) - 1.0
        )

    @staticmethod
    def return_level(period: float, params: GEVParams) -> float:
        """
        Compute the return level for a given return period.

        The T-year return level x_T satisfies P(X > x_T) = 1/T,
        i.e., x_T = quantile(1 - 1/T).

        Args:
            period: Return period (e.g. 100 for 100-year event)
            params: GEV parameters

        Returns:
            Return level value
        """
        p = 1.0 - 1.0 / period
        return float(GeneralizedExtremeValue.quantile(np.array([p]), params)[0])

    @staticmethod
    def log_likelihood(data: np.ndarray, params: GEVParams) -> float:
        """Compute log-likelihood of data under GEV(mu, sigma, xi)."""
        data = np.asarray(data, dtype=float)
        n = len(data)
        z = (data - params.mu) / params.sigma

        if abs(params.xi) < 1e-8:
            # Gumbel
            return -n * np.log(params.sigma) - np.sum(z) - np.sum(np.exp(-z))

        t = 1.0 + params.xi * z
        if np.any(t <= 0):
            return -np.inf  # Outside support

        inv_xi = 1.0 / params.xi
        return (-n * np.log(params.sigma)
                - (1.0 + inv_xi) * np.sum(np.log(t))
                - np.sum(np.power(t, -inv_xi)))

    @staticmethod
    def fit(block_maxima: np.ndarray,
            initial_params: GEVParams | None = None) -> GEVParams:
        """
        Fit GEV to block maxima via maximum likelihood estimation.

        Args:
            block_maxima: Array of block maxima values
            initial_params: Starting parameters (estimated from data if None)

        Returns:
            Fitted GEV parameters
        """
        data = np.asarray(block_maxima, dtype=float)
        n = len(data)
        if n < 3:
            raise ValueError("Need at least 3 data points for GEV fitting")

        # Method-of-moments initial guess
        if initial_params is None:
            mean_x = np.mean(data)
            std_x = np.std(data, ddof=1)
            # Gumbel moments: mean = mu + gamma*sigma, var = pi^2*sigma^2/6
            sigma0 = std_x * np.sqrt(6) / np.pi
            mu0 = mean_x - 0.5772 * sigma0
            xi0 = 0.01  # Near Gumbel
        else:
            mu0, sigma0, xi0 = initial_params.mu, initial_params.sigma, initial_params.xi

        def neg_ll(theta):
            mu, log_sigma, xi = theta
            sigma = np.exp(log_sigma)
            params = GEVParams(mu=mu, sigma=sigma, xi=xi)
            ll = GeneralizedExtremeValue.log_likelihood(data, params)
            if np.isfinite(ll):
                return -ll
            return 1e15

        x0 = np.array([mu0, np.log(max(sigma0, 1e-6)), xi0])

        if HAS_SCIPY:
            result = scipy_minimize(neg_ll, x0, method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-8})
            mu_fit, log_sigma_fit, xi_fit = result.x
        else:
            # Simple gradient-free: coordinate descent with golden section
            mu_fit, log_sigma_fit, xi_fit = _coordinate_descent_minimize(neg_ll, x0)

        return GEVParams(mu=mu_fit, sigma=np.exp(log_sigma_fit), xi=xi_fit)

    @staticmethod
    def sample(params: GEVParams, n: int) -> np.ndarray:
        """Generate n samples from a GEV distribution via inverse CDF."""
        u = np.random.uniform(0.001, 0.999, size=n)
        return GeneralizedExtremeValue.quantile(u, params)


# ============================================================================
# GUMBEL DISTRIBUTION (xi = 0 SPECIAL CASE)
# ============================================================================

class GumbelDistribution:
    """
    Gumbel distribution — the GEV with ξ = 0.

    CDF: G(x) = exp(-exp(-(x - μ)/σ))

    Arises as the limiting distribution for maxima of distributions with
    exponential-type tails (e.g., Normal, Exponential, Gamma).
    """

    @staticmethod
    def fit(data: np.ndarray) -> GEVParams:
        """
        Fit Gumbel via method of moments.

        E[X] = μ + γσ  where γ ≈ 0.5772 (Euler-Mascheroni)
        Var[X] = π²σ²/6
        """
        data = np.asarray(data, dtype=float)
        std_x = np.std(data, ddof=1)
        sigma = std_x * np.sqrt(6) / np.pi
        mu = np.mean(data) - 0.5772156649 * sigma
        return GEVParams(mu=mu, sigma=max(sigma, 1e-10), xi=0.0)

    @staticmethod
    def cdf(x: np.ndarray, params: GEVParams) -> np.ndarray:
        """Gumbel CDF."""
        x = np.asarray(x, dtype=float)
        z = (x - params.mu) / params.sigma
        return np.exp(-np.exp(-z))

    @staticmethod
    def sample(params: GEVParams, n: int) -> np.ndarray:
        """Sample from Gumbel via inverse CDF: x = μ - σ ln(-ln(U))."""
        u = np.random.uniform(0.001, 0.999, size=n)
        return params.mu - params.sigma * np.log(-np.log(u))

    @staticmethod
    def log_likelihood(data: np.ndarray, params: GEVParams) -> float:
        """Log-likelihood under Gumbel."""
        data = np.asarray(data, dtype=float)
        n = len(data)
        z = (data - params.mu) / params.sigma
        return -n * np.log(params.sigma) - np.sum(z) - np.sum(np.exp(-z))


# ============================================================================
# PEAKS OVER THRESHOLD (GPD)
# ============================================================================

class PeaksOverThreshold:
    """
    Peaks Over Threshold — models exceedances above a high threshold
    with the Generalized Pareto Distribution.

    By Pickands-Balkema-de Haan, for sufficiently high threshold u:
        P(X - u ≤ y | X > u) ≈ H(y; σ, ξ)

    where H is the GPD CDF.
    """

    @staticmethod
    def fit_gpd(exceedances: np.ndarray,
                threshold: float) -> tuple[float, float]:
        """
        Fit GPD to threshold exceedances via maximum likelihood.

        Args:
            exceedances: Raw data values exceeding the threshold
            threshold: The threshold value

        Returns:
            (sigma, xi) — GPD scale and shape parameters
        """
        exceedances = np.asarray(exceedances, dtype=float)
        # Work with exceedances above threshold
        y = exceedances - threshold
        y = y[y > 0]
        n = len(y)

        if n < 2:
            raise ValueError("Need at least 2 exceedances for GPD fitting")

        # Method-of-moments initial guess
        mean_y = np.mean(y)
        var_y = np.var(y, ddof=1)
        # For GPD: E[Y] = σ/(1-ξ), Var[Y] = σ²/((1-ξ)²(1-2ξ))
        xi0 = 0.5 * (1.0 - mean_y ** 2 / var_y) if var_y > 0 else 0.01
        xi0 = np.clip(xi0, -0.5, 2.0)
        sigma0 = mean_y * (1.0 - xi0)
        sigma0 = max(sigma0, 1e-6)

        def neg_ll(theta):
            log_sigma, xi = theta
            sigma = np.exp(log_sigma)
            if sigma <= 0:
                return 1e15
            if abs(xi) < 1e-8:
                # Exponential case
                return n * log_sigma + np.sum(y) / sigma
            t = 1.0 + xi * y / sigma
            if np.any(t <= 0):
                return 1e15
            return n * log_sigma + (1.0 + 1.0 / xi) * np.sum(np.log(t))

        x0 = np.array([np.log(sigma0), xi0])

        if HAS_SCIPY:
            result = scipy_minimize(neg_ll, x0, method='Nelder-Mead',
                                    options={'maxiter': 5000})
            log_sigma_fit, xi_fit = result.x
        else:
            log_sigma_fit, xi_fit = _coordinate_descent_minimize(neg_ll, x0)

        return np.exp(log_sigma_fit), xi_fit

    @staticmethod
    def tail_probability(x: float, threshold: float,
                         sigma: float, xi: float,
                         exceedance_rate: float = 0.1) -> float:
        """
        Estimate P(X > x) using the GPD tail model.

        P(X > x) = P(X > u) × P(X > x | X > u)
                  = ζ_u × (1 - H((x-u); σ, ξ))

        Args:
            x: Value to compute tail probability for
            threshold: The threshold u
            sigma: GPD scale
            xi: GPD shape
            exceedance_rate: Fraction of data exceeding threshold (ζ_u)

        Returns:
            Tail probability
        """
        if x <= threshold:
            return exceedance_rate  # Rough: at least the exceedance rate

        y = x - threshold

        if abs(xi) < 1e-8:
            gpd_survival = np.exp(-y / sigma)
        else:
            t = 1.0 + xi * y / sigma
            if t <= 0:
                return 0.0
            gpd_survival = t ** (-1.0 / xi)

        return exceedance_rate * gpd_survival

    @staticmethod
    def gpd_cdf(y: np.ndarray, sigma: float, xi: float) -> np.ndarray:
        """
        GPD cumulative distribution function for exceedances y ≥ 0.

        H(y) = 1 - (1 + ξy/σ)^{-1/ξ}   for ξ ≠ 0
        H(y) = 1 - exp(-y/σ)             for ξ = 0
        """
        y = np.asarray(y, dtype=float)
        if abs(xi) < 1e-8:
            return 1.0 - np.exp(-y / sigma)

        t = 1.0 + xi * y / sigma
        valid = t > 0
        result = np.zeros_like(y)
        result[valid] = 1.0 - np.power(t[valid], -1.0 / xi)
        if xi > 0:
            result[~valid] = 0.0
        else:
            result[~valid] = 1.0
        return result

    @staticmethod
    def gpd_quantile(p: np.ndarray, sigma: float, xi: float) -> np.ndarray:
        """
        GPD quantile function.

        y_p = (σ/ξ)((1-p)^{-ξ} - 1)   for ξ ≠ 0
        y_p = -σ ln(1-p)                for ξ = 0
        """
        p = np.asarray(p, dtype=float)
        if abs(xi) < 1e-8:
            return -sigma * np.log(1.0 - p)
        return (sigma / xi) * (np.power(1.0 - p, -xi) - 1.0)


# ============================================================================
# BLOCK MAXIMA EXTRACTION
# ============================================================================

def block_maxima(series: np.ndarray, block_size: int) -> np.ndarray:
    """
    Extract block maxima from a time series.

    Divides the series into non-overlapping blocks and returns the
    maximum of each block. The last incomplete block is included
    if it has at least one element.

    Args:
        series: 1D time series
        block_size: Number of observations per block

    Returns:
        Array of block maxima
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    n_blocks = n // block_size
    remainder = n % block_size

    maxima = []
    for i in range(n_blocks):
        block = series[i * block_size:(i + 1) * block_size]
        maxima.append(np.max(block))

    if remainder > 0:
        block = series[n_blocks * block_size:]
        maxima.append(np.max(block))

    return np.array(maxima)


def mean_residual_life(data: np.ndarray,
                       thresholds: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean Residual Life plot data — helps choose the POT threshold.

    For each candidate threshold u, computes E[X - u | X > u].
    The MRL should be approximately linear in u above the true threshold
    if the GPD model is appropriate.

    Args:
        data: Observations
        thresholds: Candidate thresholds (defaults to 20 quantiles)

    Returns:
        (thresholds, mean_residual_life_values)
    """
    data = np.asarray(data, dtype=float)

    if thresholds is None:
        thresholds = np.quantile(data, np.linspace(0.5, 0.99, 20))

    mrl = []
    for u in thresholds:
        exceedances = data[data > u] - u
        if len(exceedances) > 0:
            mrl.append(np.mean(exceedances))
        else:
            mrl.append(np.nan)

    return thresholds, np.array(mrl)


# ============================================================================
# INTERNAL UTILITIES
# ============================================================================

def _coordinate_descent_minimize(func, x0, max_iter=2000,
                                 step=0.1, decay=0.999, tol=1e-10):
    """
    Simple coordinate descent for MLE when scipy is unavailable.

    Cycles through each parameter, trying ±step and accepting improvements.
    Step size decays each full cycle.
    """
    x = x0.copy().astype(float)
    f_best = func(x)
    n_params = len(x)

    for iteration in range(max_iter):
        improved = False
        for i in range(n_params):
            for direction in [step, -step]:
                x_try = x.copy()
                x_try[i] += direction
                f_try = func(x_try)
                if f_try < f_best:
                    x = x_try
                    f_best = f_try
                    improved = True

        step *= decay
        if step < tol:
            break
        if not improved and step < 1e-6:
            break

    return x
