"""
Time Series Models — ARIMA, Exponential Smoothing, Autocorrelation
==================================================================

Classical time series analysis: autoregressive models, moving average
processes, exponential smoothing families, and correlation diagnostics.

Core Concepts:
- ARIMA(p,d,q): Autoregressive Integrated Moving Average
- Exponential Smoothing: Simple, Holt's (double), Holt-Winters (triple)
- Autocorrelation Function (ACF): Serial dependence structure
- Partial ACF (PACF): Direct dependence at each lag
- Ljung-Box test: Portmanteau test for white noise

Mathematical Foundation:
ARMA(p,q): X_t = c + Σᵢ φᵢ X_{t-i} + ε_t + Σⱼ θⱼ ε_{t-j}
ARIMA(p,d,q): Apply ARMA to d-th difference of X_t

Exponential Smoothing:
  Simple:  ℓ_t = α y_t + (1-α) ℓ_{t-1}
  Holt:    ℓ_t = α y_t + (1-α)(ℓ_{t-1} + b_{t-1})
           b_t = β (ℓ_t - ℓ_{t-1}) + (1-β) b_{t-1}
  Winters: ℓ_t = α(y_t - s_{t-m}) + (1-α)(ℓ_{t-1} + b_{t-1})
           b_t = β(ℓ_t - ℓ_{t-1}) + (1-β) b_{t-1}
           s_t = γ(y_t - ℓ_t) + (1-γ) s_{t-m}

ACF: ρ(k) = Cov(X_t, X_{t+k}) / Var(X_t)
PACF: α(k) = Corr(X_t, X_{t+k} | X_{t+1}, ..., X_{t+k-1})

Applications to Warp Space:
- Forecasting convergence trajectories
- Detecting periodicity in scoring sequences
- Modeling temporal patterns in knowledge graph activity
- Confidence interval estimation for predictions

Author: HoloLoom Team
Date: 2025-10-26
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize as scipy_minimize
    from scipy.stats import chi2 as scipy_chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("scipy not available; some fitting methods will use fallbacks")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ARIMAParams:
    """
    Parameters of an ARIMA(p,d,q) model.

    Attributes:
        p: AR order (number of autoregressive terms)
        d: Differencing order
        q: MA order (number of moving average terms)
        ar_coeffs: AR coefficients φ₁, ..., φ_p
        ma_coeffs: MA coefficients θ₁, ..., θ_q
        intercept: Constant term c
        sigma2: Estimated innovation variance
    """
    p: int
    d: int
    q: int
    ar_coeffs: np.ndarray
    ma_coeffs: np.ndarray
    intercept: float = 0.0
    sigma2: float = 1.0


@dataclass
class ForecastResult:
    """
    Result of a time series forecast.

    Attributes:
        values: Point forecasts
        confidence_lower: Lower confidence bound (95%)
        confidence_upper: Upper confidence bound (95%)
        aic: Akaike Information Criterion of the fitted model
    """
    values: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    aic: float = 0.0


# ============================================================================
# ARIMA
# ============================================================================

class ARIMA:
    """
    ARIMA(p,d,q) — Autoregressive Integrated Moving Average.

    Fits via conditional least squares on the differenced series.
    """

    @staticmethod
    def difference(series: np.ndarray, d: int = 1) -> np.ndarray:
        """
        Apply d-th order differencing.

        Δ¹ y_t = y_t - y_{t-1}
        Δ² y_t = Δ¹ y_t - Δ¹ y_{t-1}
        """
        result = np.asarray(series, dtype=float).copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    @staticmethod
    def undifference(diffed: np.ndarray, original: np.ndarray, d: int = 1) -> np.ndarray:
        """
        Reverse differencing to recover the original scale.

        Uses the last d values of the original series as anchors.

        Args:
            diffed: Differenced (forecasted) values to integrate
            original: Original undifferenced series
            d: Differencing order

        Returns:
            Values in the original scale
        """
        result = diffed.copy()
        for i in range(d):
            # For each level of differencing, cumsum from the last known value
            # at that differencing level
            if d - i == 1:
                anchor = original[-1]
            else:
                temp = original.copy()
                for _ in range(d - i - 1):
                    temp = np.diff(temp)
                anchor = temp[-1]
            result = np.cumsum(np.concatenate([[anchor], result]))[1:]
        return result

    @staticmethod
    def fit(series: np.ndarray, order: tuple[int, int, int] = (1, 0, 0)
            ) -> ARIMAParams:
        """
        Fit ARIMA model via conditional least squares.

        For pure AR(p): solves Yule-Walker or OLS on lagged values.
        For ARMA(p,q): iterative conditional least squares.

        Args:
            series: Time series (1D array)
            order: (p, d, q) tuple

        Returns:
            Fitted parameters
        """
        series = np.asarray(series, dtype=float)
        p, d, q = order

        # Difference the series
        y = series.copy()
        for _ in range(d):
            y = np.diff(y)

        n = len(y)
        if n < p + q + 1:
            raise ValueError(f"Series too short ({n}) for ARIMA({p},{d},{q})")

        if q == 0:
            # Pure AR — solve via OLS
            return ARIMA._fit_ar(y, p, d)
        else:
            # ARMA — conditional least squares
            return ARIMA._fit_arma(y, p, d, q)

    @staticmethod
    def _fit_ar(y: np.ndarray, p: int, d: int) -> ARIMAParams:
        """Fit AR(p) via OLS on lagged values."""
        n = len(y)

        # Build design matrix [1, y_{t-1}, ..., y_{t-p}]
        X = np.ones((n - p, p + 1))
        for i in range(p):
            X[:, i + 1] = y[p - i - 1:n - i - 1]

        Y = y[p:]

        # OLS: θ = (X'X)^{-1} X'Y
        try:
            theta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("OLS failed; using pseudoinverse")
            theta = np.linalg.pinv(X) @ Y

        intercept = theta[0]
        ar_coeffs = theta[1:p + 1]
        residuals = Y - X @ theta
        sigma2 = np.var(residuals, ddof=p + 1)

        # AIC = n ln(σ²) + 2(p + 1)
        k = p + 1
        aic = n * np.log(max(sigma2, 1e-15)) + 2 * k

        return ARIMAParams(
            p=p, d=d, q=0,
            ar_coeffs=ar_coeffs,
            ma_coeffs=np.array([]),
            intercept=intercept,
            sigma2=max(sigma2, 1e-15)
        )

    @staticmethod
    def _fit_arma(y: np.ndarray, p: int, d: int, q: int) -> ARIMAParams:
        """Fit ARMA(p,q) via conditional least squares (iterative)."""
        n = len(y)
        start = max(p, q)

        def compute_residuals(theta):
            intercept = theta[0]
            ar = theta[1:p + 1]
            ma = theta[p + 1:p + q + 1]

            residuals = np.zeros(n)
            for t in range(start, n):
                pred = intercept
                for i in range(p):
                    if t - i - 1 >= 0:
                        pred += ar[i] * y[t - i - 1]
                for j in range(q):
                    if t - j - 1 >= 0:
                        pred += ma[j] * residuals[t - j - 1]
                residuals[t] = y[t] - pred
            return residuals[start:]

        def objective(theta):
            res = compute_residuals(theta)
            return np.sum(res ** 2)

        # Initial: AR coeffs from Yule-Walker, MA coeffs = 0
        x0 = np.zeros(1 + p + q)
        x0[0] = np.mean(y)
        if p > 0:
            # Simple Yule-Walker for initial AR
            acf_vals = _raw_acf(y, p)
            if p == 1:
                x0[1] = acf_vals[1] / max(acf_vals[0], 1e-10) if acf_vals[0] > 0 else 0.0
            else:
                # Levinson-Durbin for initial AR
                ar_init = _levinson_durbin(acf_vals, p)
                x0[1:p + 1] = ar_init

        if HAS_SCIPY:
            result = scipy_minimize(objective, x0, method='Nelder-Mead',
                                    options={'maxiter': 10000})
            theta_opt = result.x
        else:
            theta_opt = _coordinate_descent(objective, x0)

        intercept = theta_opt[0]
        ar_coeffs = theta_opt[1:p + 1]
        ma_coeffs = theta_opt[p + 1:p + q + 1]

        residuals = compute_residuals(theta_opt)
        sigma2 = np.var(residuals, ddof=0)

        return ARIMAParams(
            p=p, d=d, q=q,
            ar_coeffs=ar_coeffs,
            ma_coeffs=ma_coeffs,
            intercept=intercept,
            sigma2=max(sigma2, 1e-15)
        )

    @staticmethod
    def forecast(params: ARIMAParams, series: np.ndarray,
                 steps: int = 1) -> ForecastResult:
        """
        Produce multi-step forecasts from a fitted ARIMA model.

        Uses the fitted model to extrapolate, then undifferences if d > 0.
        Confidence intervals assume Gaussian innovations.

        Args:
            params: Fitted ARIMA parameters
            series: Original (undifferenced) series
            steps: Number of steps ahead to forecast

        Returns:
            ForecastResult with point forecasts and 95% CI
        """
        series = np.asarray(series, dtype=float)

        # Work on differenced series
        y = series.copy()
        for _ in range(params.d):
            y = np.diff(y)

        n = len(y)
        forecasts = np.zeros(steps)
        # Extend y with forecasts for autoregressive terms
        y_ext = np.concatenate([y, np.zeros(steps)])

        # Residuals from the end of the fitted series (for MA terms)
        residuals = ARIMA.residuals(params, series)
        res_ext = np.concatenate([residuals, np.zeros(steps)])

        for h in range(steps):
            t = n + h
            pred = params.intercept
            for i in range(params.p):
                if t - i - 1 >= 0:
                    pred += params.ar_coeffs[i] * y_ext[t - i - 1]
            for j in range(params.q):
                if t - j - 1 >= 0:
                    pred += params.ma_coeffs[j] * res_ext[t - j - 1]
            forecasts[h] = pred
            y_ext[t] = pred
            # Future residuals are 0 (best linear predictor)

        # Forecast variance grows with horizon
        # For AR(1): var(h) = σ² Σ_{i=0}^{h-1} φ^{2i}
        # General case: use psi weights (MA(∞) representation)
        psi = _compute_psi_weights(params, steps)
        forecast_var = np.zeros(steps)
        for h in range(steps):
            forecast_var[h] = params.sigma2 * np.sum(psi[:h + 1] ** 2)

        forecast_std = np.sqrt(np.maximum(forecast_var, 0))

        # Undifference if needed
        if params.d > 0:
            forecasts = ARIMA.undifference(forecasts, series, params.d)
            # Approximate: CI in original scale (undifferencing is linear)

        z_95 = 1.96
        lower = forecasts - z_95 * forecast_std
        upper = forecasts + z_95 * forecast_std

        # AIC
        n_total = len(series)
        k = params.p + params.q + 1  # +1 for intercept
        aic = n_total * np.log(max(params.sigma2, 1e-15)) + 2 * k

        return ForecastResult(
            values=forecasts,
            confidence_lower=lower,
            confidence_upper=upper,
            aic=aic
        )

    @staticmethod
    def residuals(params: ARIMAParams, series: np.ndarray) -> np.ndarray:
        """
        Compute residuals (one-step-ahead prediction errors) from a fitted model.

        Args:
            params: Fitted parameters
            series: Original undifferenced series

        Returns:
            Array of residuals
        """
        series = np.asarray(series, dtype=float)
        y = series.copy()
        for _ in range(params.d):
            y = np.diff(y)

        n = len(y)
        start = max(params.p, params.q)
        residuals = np.zeros(n)

        for t in range(start, n):
            pred = params.intercept
            for i in range(params.p):
                if t - i - 1 >= 0:
                    pred += params.ar_coeffs[i] * y[t - i - 1]
            for j in range(params.q):
                if t - j - 1 >= 0:
                    pred += params.ma_coeffs[j] * residuals[t - j - 1]
            residuals[t] = y[t] - pred

        return residuals


# ============================================================================
# EXPONENTIAL SMOOTHING
# ============================================================================

class ExponentialSmoothing:
    """
    Exponential smoothing family — weighted averages with exponentially
    decaying weights on past observations.
    """

    @staticmethod
    def simple(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Simple Exponential Smoothing (SES).

        ℓ_t = α y_t + (1 - α) ℓ_{t-1}

        Suitable for series with no trend or seasonality.

        Args:
            series: Time series
            alpha: Smoothing parameter in (0, 1)

        Returns:
            Smoothed values (same length as input)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        series = np.asarray(series, dtype=float)
        n = len(series)
        smoothed = np.zeros(n)
        smoothed[0] = series[0]

        for t in range(1, n):
            smoothed[t] = alpha * series[t] + (1 - alpha) * smoothed[t - 1]

        return smoothed

    @staticmethod
    def double(series: np.ndarray, alpha: float = 0.3,
               beta: float = 0.1) -> np.ndarray:
        """
        Double Exponential Smoothing (Holt's method).

        ℓ_t = α y_t + (1 - α)(ℓ_{t-1} + b_{t-1})
        b_t = β(ℓ_t - ℓ_{t-1}) + (1 - β) b_{t-1}

        Handles linear trends.

        Args:
            series: Time series
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter

        Returns:
            Smoothed values (level + trend)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        series = np.asarray(series, dtype=float)
        n = len(series)
        level = np.zeros(n)
        trend = np.zeros(n)
        smoothed = np.zeros(n)

        # Initialize
        level[0] = series[0]
        trend[0] = series[1] - series[0] if n > 1 else 0.0
        smoothed[0] = level[0]

        for t in range(1, n):
            level[t] = alpha * series[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
            smoothed[t] = level[t] + trend[t]

        return smoothed

    @staticmethod
    def triple(series: np.ndarray, alpha: float = 0.3,
               beta: float = 0.1, gamma: float = 0.1,
               period: int = 12) -> np.ndarray:
        """
        Triple Exponential Smoothing (Holt-Winters, additive seasonality).

        ℓ_t = α(y_t - s_{t-m}) + (1-α)(ℓ_{t-1} + b_{t-1})
        b_t = β(ℓ_t - ℓ_{t-1}) + (1-β) b_{t-1}
        s_t = γ(y_t - ℓ_t) + (1-γ) s_{t-m}

        Args:
            series: Time series (length should be >= 2 * period)
            alpha: Level smoothing
            beta: Trend smoothing
            gamma: Seasonal smoothing
            period: Seasonal period (m)

        Returns:
            Smoothed values
        """
        for name, val in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
            if not 0 < val < 1:
                raise ValueError(f"{name} must be in (0, 1), got {val}")

        series = np.asarray(series, dtype=float)
        n = len(series)
        if n < 2 * period:
            logger.warning(f"Series length ({n}) < 2 * period ({2 * period}); "
                           "seasonal estimates may be unreliable")

        level = np.zeros(n)
        trend = np.zeros(n)
        seasonal = np.zeros(n + period)
        smoothed = np.zeros(n)

        # Initialize: average first period for level, difference for trend
        level[0] = np.mean(series[:period])
        if n >= 2 * period:
            trend[0] = (np.mean(series[period:2 * period]) -
                        np.mean(series[:period])) / period
        else:
            trend[0] = (series[min(period, n - 1)] - series[0]) / max(period, 1)

        # Initialize seasonal from first period
        for i in range(period):
            if i < n:
                seasonal[i] = series[i] - level[0]
            else:
                seasonal[i] = 0.0

        smoothed[0] = level[0] + trend[0] + seasonal[0]

        for t in range(1, n):
            s_prev = seasonal[t - period] if t >= period else seasonal[t % period]
            level[t] = alpha * (series[t] - s_prev) + (1 - alpha) * (level[t - 1] + trend[t - 1])
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
            seasonal[t] = gamma * (series[t] - level[t]) + (1 - gamma) * s_prev
            smoothed[t] = level[t] + trend[t] + seasonal[t]

        return smoothed


# ============================================================================
# AUTOCORRELATION
# ============================================================================

class AutoCorrelation:
    """
    Autocorrelation analysis — measuring serial dependence in time series.
    """

    @staticmethod
    def acf(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
        """
        Sample autocorrelation function.

        ρ̂(k) = Σ (y_t - ȳ)(y_{t+k} - ȳ) / Σ (y_t - ȳ)²

        Args:
            series: Time series
            max_lag: Maximum lag to compute (default: min(n//2, 40))

        Returns:
            ACF values for lags 0, 1, ..., max_lag (ρ(0) = 1 by definition)
        """
        series = np.asarray(series, dtype=float)
        n = len(series)
        if max_lag is None:
            max_lag = min(n // 2, 40)

        max_lag = min(max_lag, n - 1)
        mean = np.mean(series)
        centered = series - mean
        var = np.sum(centered ** 2)

        if var < 1e-15:
            return np.ones(max_lag + 1)

        acf_vals = np.zeros(max_lag + 1)
        for k in range(max_lag + 1):
            acf_vals[k] = np.sum(centered[:n - k] * centered[k:]) / var

        return acf_vals

    @staticmethod
    def pacf(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
        """
        Sample partial autocorrelation function via Levinson-Durbin recursion.

        The PACF at lag k is the correlation between y_t and y_{t+k} after
        removing the linear effect of the intermediate lags.

        Args:
            series: Time series
            max_lag: Maximum lag (default: min(n//4, 20))

        Returns:
            PACF values for lags 0, 1, ..., max_lag (PACF(0) = 1)
        """
        series = np.asarray(series, dtype=float)
        n = len(series)
        if max_lag is None:
            max_lag = min(n // 4, 20)

        max_lag = min(max_lag, n // 2)
        acf_vals = _raw_acf(series, max_lag)

        pacf_vals = np.zeros(max_lag + 1)
        pacf_vals[0] = 1.0

        if max_lag == 0:
            return pacf_vals

        # Levinson-Durbin
        phi = np.zeros((max_lag + 1, max_lag + 1))
        phi[1, 1] = acf_vals[1] / max(acf_vals[0], 1e-15)
        pacf_vals[1] = phi[1, 1]

        for k in range(2, max_lag + 1):
            numerator = acf_vals[k]
            for j in range(1, k):
                numerator -= phi[k - 1, j] * acf_vals[k - j]

            denominator = acf_vals[0]
            for j in range(1, k):
                denominator -= phi[k - 1, j] * acf_vals[j]

            if abs(denominator) < 1e-15:
                pacf_vals[k] = 0.0
                continue

            phi[k, k] = numerator / denominator
            pacf_vals[k] = phi[k, k]

            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

        return pacf_vals

    @staticmethod
    def ljung_box(series: np.ndarray, lags: int = 10) -> tuple[float, float]:
        """
        Ljung-Box portmanteau test for autocorrelation.

        Tests H₀: the series is white noise (no autocorrelation up to lag h).

        Q = n(n+2) Σ_{k=1}^{h} ρ̂(k)² / (n-k)

        Under H₀, Q ~ χ²(h).

        Args:
            series: Time series (or residuals)
            lags: Number of lags to test

        Returns:
            (test_statistic, p_value)
        """
        series = np.asarray(series, dtype=float)
        n = len(series)
        lags = min(lags, n // 2 - 1)
        if lags < 1:
            return 0.0, 1.0

        acf_vals = AutoCorrelation.acf(series, max_lag=lags)

        Q = 0.0
        for k in range(1, lags + 1):
            Q += acf_vals[k] ** 2 / (n - k)
        Q *= n * (n + 2)

        # p-value from chi-squared distribution
        if HAS_SCIPY:
            p_value = 1.0 - scipy_chi2.cdf(Q, df=lags)
        else:
            # Rough approximation when scipy unavailable
            # Use Wilson-Hilferty normal approximation to chi-squared
            k_df = float(lags)
            z = ((Q / k_df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k_df))) / \
                np.sqrt(2.0 / (9.0 * k_df))
            # Standard normal survival function approximation
            p_value = 0.5 * (1.0 - np.tanh(z * 0.7978845608))  # erfc approx
            p_value = max(0.0, min(1.0, p_value))

        return Q, p_value


# ============================================================================
# INTERNAL UTILITIES
# ============================================================================

def _raw_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Unnormalized autocovariance (divided by n, not n-k)."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    mean = np.mean(series)
    centered = series - mean

    acf_vals = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        acf_vals[k] = np.sum(centered[:n - k] * centered[k:]) / n

    return acf_vals


def _levinson_durbin(acf: np.ndarray, p: int) -> np.ndarray:
    """
    Levinson-Durbin algorithm: solve Yule-Walker equations for AR(p) coefficients.
    """
    if p == 0:
        return np.array([])

    phi = np.zeros((p + 1, p + 1))

    if abs(acf[0]) < 1e-15:
        return np.zeros(p)

    phi[1, 1] = acf[1] / acf[0]

    for k in range(2, p + 1):
        num = acf[k]
        for j in range(1, k):
            num -= phi[k - 1, j] * acf[k - j]

        den = acf[0]
        for j in range(1, k):
            den -= phi[k - 1, j] * acf[j]

        if abs(den) < 1e-15:
            break

        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return phi[p, 1:p + 1]


def _compute_psi_weights(params: ARIMAParams, steps: int) -> np.ndarray:
    """
    Compute psi weights (MA(∞) representation) for forecast variance.

    ψ_0 = 1
    ψ_j = Σᵢ φᵢ ψ_{j-i} + θ_j  (θ_j = 0 for j > q)
    """
    psi = np.zeros(steps)
    psi[0] = 1.0

    for j in range(1, steps):
        for i in range(min(params.p, j)):
            psi[j] += params.ar_coeffs[i] * psi[j - i - 1]
        if j <= params.q:
            psi[j] += params.ma_coeffs[j - 1]

    return psi


def _coordinate_descent(func, x0, max_iter=3000, step=0.05,
                        decay=0.998, tol=1e-10):
    """Simple coordinate descent minimizer (scipy fallback)."""
    x = x0.copy().astype(float)
    f_best = func(x)

    for iteration in range(max_iter):
        improved = False
        for i in range(len(x)):
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
