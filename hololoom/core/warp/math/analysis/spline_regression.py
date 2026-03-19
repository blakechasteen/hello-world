"""
Spline Regression — LOESS, B-Splines, VAR, Regression Diagnostics
===================================================================

Flexible nonlinear regression and multivariate time series modeling.

Core Concepts:
- LOESS: Locally weighted scatterplot smoothing (local polynomial regression)
- B-Spline Regression: Piecewise polynomial basis with knot placement
- VAR: Vector autoregression for multivariate time series
- Diagnostics: Breusch-Pagan heteroscedasticity, VIF for multicollinearity

Mathematical Foundation:
LOESS: Minimize Σ w_i(x₀)(y_i - β₀ - β₁x_i)² with tricube weights
B-spline: y = Σ c_j B_j,k(x) where B_j,k are basis functions of degree k
VAR(p): Y_t = c + A₁Y_{t-1} + ... + AₚY_{t-p} + ε_t
Breusch-Pagan: nR² ~ χ²(p) under H₀ of homoscedasticity
VIF_j = 1/(1 - R²_j) where R²_j is from regressing X_j on other predictors

Applications to Warp Space:
- Smooth nonlinear trend detection in embedding trajectories
- Flexible response surfaces for scoring term calibration
- Joint modeling of correlated time series from multiple memory systems
- Diagnostic checks for linear models in convergence analysis

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
class SplineResult:
    """Result of spline regression.

    Attributes:
        coefficients: Spline basis coefficients.
        knots: Knot positions used.
        fitted: Fitted values at training points.
        basis: B-spline basis matrix (n_samples, n_basis).
        residuals: y - fitted.
        gcv: Generalized cross-validation score.
    """
    coefficients: np.ndarray
    knots: np.ndarray
    fitted: np.ndarray
    basis: np.ndarray
    residuals: np.ndarray
    gcv: float = 0.0


@dataclass
class VARResult:
    """Result of VAR model fitting.

    Attributes:
        coefficients: List of coefficient matrices [A₁, ..., Aₚ].
        intercept: Intercept vector c.
        residuals: Residual matrix (T - p, k).
        covariance: Residual covariance matrix (k, k).
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        is_stable: True if all eigenvalues of companion matrix inside unit circle.
    """
    coefficients: list[np.ndarray]
    intercept: np.ndarray
    residuals: np.ndarray
    covariance: np.ndarray
    aic: float = 0.0
    bic: float = 0.0
    is_stable: bool = True


# ============================================================================
# LOESS REGRESSION
# ============================================================================

class LOESSRegression:
    """Locally Weighted Scatterplot Smoothing (LOESS/LOWESS).

    Fits local polynomial regressions at each prediction point using
    distance-weighted neighbors. The tricube weight function ensures
    smooth transitions between local fits.

    w(u) = (1 - |u|³)³  for |u| < 1, else 0

    Example:
        >>> loess = LOESSRegression()
        >>> X = np.linspace(0, 2*np.pi, 100)
        >>> y = np.sin(X) + 0.3 * np.random.randn(100)
        >>> y_smooth = loess.fit_predict(X, y, X, span=0.3)
    """

    @staticmethod
    def _tricube(d: np.ndarray) -> np.ndarray:
        """Tricube weight function w(u) = (1 - |u|^3)^3."""
        u = np.clip(np.abs(d), 0, 1)
        return (1.0 - u ** 3) ** 3

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_new: np.ndarray,
        span: float = 0.3,
        degree: int = 1,
    ) -> np.ndarray:
        """Fit LOESS and predict at new points.

        Args:
            X: Training inputs (n,) or (n, d).
            y: Training targets (n,).
            X_new: Prediction points (m,) or (m, d).
            span: Fraction of data to use for each local fit (0, 1].
            degree: Local polynomial degree (1=linear, 2=quadratic).

        Returns:
            Predicted values at X_new (m,).
        """
        X = np.atleast_2d(X.reshape(-1, 1) if X.ndim == 1 else X)
        X_new = np.atleast_2d(X_new.reshape(-1, 1) if X_new.ndim == 1 else X_new)
        y = np.asarray(y, dtype=float)
        n = len(y)
        k = max(int(np.ceil(span * n)), degree + 1)
        k = min(k, n)

        y_pred = np.empty(len(X_new))

        for i, x0 in enumerate(X_new):
            # Distances to all training points
            dists = np.sqrt(np.sum((X - x0) ** 2, axis=1))
            # k nearest neighbors
            idx = np.argsort(dists)[:k]
            max_dist = dists[idx[-1]]
            if max_dist == 0:
                max_dist = 1.0

            # Tricube weights
            u = dists[idx] / max_dist
            w = self._tricube(u)

            # Build local design matrix
            X_local = X[idx] - x0
            if degree == 1:
                A = np.column_stack([np.ones(k), X_local])
            elif degree == 2:
                A = np.column_stack([
                    np.ones(k),
                    X_local,
                    X_local ** 2 if X_local.shape[1] == 1
                    else np.column_stack([
                        X_local[:, j] * X_local[:, l]
                        for j in range(X_local.shape[1])
                        for l in range(j, X_local.shape[1])
                    ])
                ])
            else:
                A = np.column_stack([np.ones(k), X_local])

            # Weighted least squares
            W = np.diag(w)
            try:
                AtWA = A.T @ W @ A
                AtWy = A.T @ W @ y[idx]
                beta = np.linalg.solve(AtWA + 1e-10 * np.eye(AtWA.shape[0]), AtWy)
                y_pred[i] = beta[0]  # Prediction at x0 (centered)
            except np.linalg.LinAlgError:
                y_pred[i] = np.average(y[idx], weights=w)

        return y_pred

    def bandwidth_cv(
        self, X: np.ndarray, y: np.ndarray, spans: np.ndarray | None = None
    ) -> float:
        """Select optimal span via leave-one-out cross-validation.

        Args:
            X: Training inputs.
            y: Training targets.
            spans: Candidate span values (default: 10 values in [0.1, 0.9]).

        Returns:
            Optimal span value.
        """
        if spans is None:
            spans = np.linspace(0.1, 0.9, 10)

        best_span = spans[0]
        best_cv = np.inf

        for span in spans:
            cv_error = 0.0
            n = len(y)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                pred = self.fit_predict(
                    X[mask] if X.ndim == 1 else X[mask],
                    y[mask],
                    X[i:i+1] if X.ndim == 1 else X[i:i+1],
                    span=span,
                )
                cv_error += (y[i] - pred[0]) ** 2
            cv_error /= n
            if cv_error < best_cv:
                best_cv = cv_error
                best_span = span

        return float(best_span)


# ============================================================================
# B-SPLINE REGRESSION
# ============================================================================

class SplineRegression:
    """B-Spline Regression with automatic or manual knot placement.

    Fits y = Σ c_j B_{j,k}(x) using the Cox-de Boor recursion for
    B-spline basis evaluation and least squares for coefficients.

    Example:
        >>> sr = SplineRegression()
        >>> X = np.linspace(0, 1, 200)
        >>> y = np.sin(4 * np.pi * X) + 0.2 * np.random.randn(200)
        >>> result = sr.fit(X, y, n_knots=8, degree=3)
    """

    @staticmethod
    def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
        """Evaluate B-spline basis functions via Cox-de Boor recursion.

        Args:
            x: Evaluation points (n,).
            knots: Augmented knot vector (n_knots + 2*degree + 2,).
            degree: Spline degree.

        Returns:
            Basis matrix (n, n_basis) where n_basis = len(knots) - degree - 1.
        """
        n = len(x)
        n_basis = len(knots) - degree - 1
        B = np.zeros((n, n_basis))

        # Degree 0: indicator functions
        for j in range(n_basis + degree):
            if j < len(knots) - 1:
                if j < n_basis:
                    mask = (x >= knots[j]) & (x < knots[j + 1])
                    B[:, j] = mask.astype(float)

        # Handle right endpoint
        if n_basis > 0:
            B[x == knots[-1], -1] = 1.0

        if degree == 0:
            return B

        # Recursive evaluation
        for d in range(1, degree + 1):
            n_d = len(knots) - d - 1
            B_new = np.zeros((n, min(n_d, n_basis)))
            for j in range(min(n_d, n_basis)):
                # Left term
                denom1 = knots[j + d] - knots[j]
                if denom1 > 0 and j < B.shape[1]:
                    B_new[:, j] += (x - knots[j]) / denom1 * B[:, j]
                # Right term
                denom2 = knots[j + d + 1] - knots[j + 1]
                if denom2 > 0 and j + 1 < B.shape[1]:
                    B_new[:, j] += (knots[j + d + 1] - x) / denom2 * B[:, j + 1]
            B = B_new

        return B

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_knots: int = 5,
        degree: int = 3,
        knots: np.ndarray | None = None,
        penalty: float = 0.0,
    ) -> SplineResult:
        """Fit B-spline regression.

        Args:
            X: Input values (n,).
            y: Target values (n,).
            n_knots: Number of interior knots (ignored if knots provided).
            degree: Spline degree (3 = cubic).
            knots: Manual knot positions. If None, placed at quantiles.
            penalty: Ridge penalty for regularization (0 = none).

        Returns:
            SplineResult with fitted model.
        """
        X = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = len(X)

        # Place interior knots at quantiles
        if knots is None:
            quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]
            interior_knots = np.percentile(X, quantiles)
        else:
            interior_knots = np.sort(knots)

        # Augmented knot vector with boundary knots
        x_min, x_max = X.min(), X.max()
        augmented = np.concatenate([
            np.repeat(x_min, degree + 1),
            interior_knots,
            np.repeat(x_max, degree + 1),
        ])

        # Build basis matrix
        B = self._bspline_basis(X, augmented, degree)
        n_basis = B.shape[1]

        # Solve penalized least squares: (B'B + λI)c = B'y
        BtB = B.T @ B
        if penalty > 0:
            BtB += penalty * np.eye(n_basis)
        Bty = B.T @ y

        try:
            coeffs = np.linalg.solve(BtB + 1e-12 * np.eye(n_basis), Bty)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(B, y, rcond=None)[0]

        fitted = B @ coeffs
        residuals = y - fitted

        # GCV score: ||y - f||² / (1 - tr(H)/n)²
        try:
            H = B @ np.linalg.solve(BtB + 1e-12 * np.eye(n_basis), B.T)
            tr_H = np.trace(H)
            gcv = np.sum(residuals ** 2) / (1.0 - tr_H / n) ** 2 / n
        except np.linalg.LinAlgError:
            gcv = np.mean(residuals ** 2)

        return SplineResult(
            coefficients=coeffs,
            knots=augmented,
            fitted=fitted,
            basis=B,
            residuals=residuals,
            gcv=gcv,
        )

    def predict(self, result: SplineResult, X_new: np.ndarray, degree: int = 3) -> np.ndarray:
        """Predict at new points using fitted spline.

        Args:
            result: Fitted SplineResult.
            X_new: New input values.
            degree: Spline degree (must match fit).

        Returns:
            Predicted values.
        """
        X_new = np.asarray(X_new, dtype=float).ravel()
        B_new = self._bspline_basis(X_new, result.knots, degree)
        return B_new @ result.coefficients


# ============================================================================
# VECTOR AUTOREGRESSION
# ============================================================================

class VARModel:
    """Vector Autoregression (VAR) for multivariate time series.

    Y_t = c + A₁Y_{t-1} + ... + AₚY_{t-p} + ε_t

    Estimates via OLS on stacked regressors. Supports impulse response
    functions and Granger causality testing.

    Example:
        >>> var = VARModel()
        >>> data = np.random.randn(200, 3)
        >>> result = var.fit(data, order=2)
        >>> forecast = var.forecast(result, steps=10)
    """

    def fit(self, Y: np.ndarray, order: int = 1) -> VARResult:
        """Fit VAR(p) model via OLS.

        Args:
            Y: Multivariate time series (T, k).
            order: Lag order p.

        Returns:
            VARResult with estimated parameters.
        """
        Y = np.asarray(Y, dtype=float)
        T, k = Y.shape

        if T <= order:
            raise ValueError(f"Need T > order, got T={T}, order={order}")

        # Build regressor matrix Z = [1, Y_{t-1}, ..., Y_{t-p}]
        n = T - order
        Z = np.ones((n, 1 + order * k))
        for lag in range(order):
            Z[:, 1 + lag * k: 1 + (lag + 1) * k] = Y[order - lag - 1: T - lag - 1]

        Y_dep = Y[order:]

        # OLS: B = (Z'Z)^{-1} Z'Y_dep
        ZtZ = Z.T @ Z
        try:
            B = np.linalg.solve(ZtZ + 1e-12 * np.eye(ZtZ.shape[0]), Z.T @ Y_dep)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(Z, Y_dep, rcond=None)[0]

        intercept = B[0]
        coefficients = []
        for lag in range(order):
            A_lag = B[1 + lag * k: 1 + (lag + 1) * k].T
            coefficients.append(A_lag)

        residuals = Y_dep - Z @ B
        cov = residuals.T @ residuals / n

        # Information criteria
        log_det = np.log(np.linalg.det(cov) + 1e-300)
        n_params = k * (1 + order * k)
        aic = log_det + 2.0 * n_params / n
        bic = log_det + np.log(n) * n_params / n

        # Stability: check companion matrix eigenvalues
        is_stable = self._check_stability(coefficients, k)

        return VARResult(
            coefficients=coefficients,
            intercept=intercept,
            residuals=residuals,
            covariance=cov,
            aic=aic,
            bic=bic,
            is_stable=is_stable,
        )

    def _check_stability(self, coefficients: list[np.ndarray], k: int) -> bool:
        """Check VAR stability via companion matrix eigenvalues."""
        p = len(coefficients)
        if p == 0:
            return True

        companion = np.zeros((k * p, k * p))
        for i, A in enumerate(coefficients):
            companion[:k, i * k: (i + 1) * k] = A
        if p > 1:
            companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))

        eigenvalues = np.linalg.eigvals(companion)
        return bool(np.all(np.abs(eigenvalues) < 1.0))

    def forecast(self, result: VARResult, steps: int = 10, Y_last: np.ndarray | None = None) -> np.ndarray:
        """Generate multi-step forecasts.

        Args:
            result: Fitted VARResult.
            steps: Number of steps ahead.
            Y_last: Last p observations (p, k). If None, uses residuals tail.

        Returns:
            Forecast matrix (steps, k).
        """
        p = len(result.coefficients)
        k = result.intercept.shape[0]

        if Y_last is None:
            Y_last = result.residuals[-p:]

        Y_last = np.asarray(Y_last, dtype=float)
        if Y_last.shape[0] < p:
            pad = np.zeros((p - Y_last.shape[0], k))
            Y_last = np.vstack([pad, Y_last])

        history = list(Y_last[-p:])
        forecasts = []

        for _ in range(steps):
            y_new = result.intercept.copy()
            for lag, A in enumerate(result.coefficients):
                idx = len(history) - 1 - lag
                if idx >= 0:
                    y_new = y_new + A @ history[idx]
            forecasts.append(y_new)
            history.append(y_new)

        return np.array(forecasts)

    def impulse_response(self, result: VARResult, steps: int = 20, shock_var: int = 0) -> np.ndarray:
        """Compute orthogonalized impulse response function.

        Args:
            result: Fitted VARResult.
            steps: Number of response steps.
            shock_var: Index of variable receiving unit shock.

        Returns:
            IRF matrix (steps, k) — response of each variable.
        """
        k = result.intercept.shape[0]
        p = len(result.coefficients)

        # Cholesky of covariance for orthogonalization
        try:
            L = np.linalg.cholesky(result.covariance)
        except np.linalg.LinAlgError:
            L = np.eye(k)

        # MA representation via recursion
        Phi = [np.eye(k)]
        for s in range(1, steps):
            Phi_s = np.zeros((k, k))
            for j in range(min(s, p)):
                Phi_s += result.coefficients[j] @ Phi[s - j - 1]
            Phi.append(Phi_s)

        # Orthogonalized IRF
        shock = L[:, shock_var]
        irf = np.array([Phi_s @ shock for Phi_s in Phi])
        return irf

    def granger_causality(self, Y: np.ndarray, cause: int, effect: int, order: int = 1) -> tuple[float, float]:
        """Test Granger causality from cause to effect variable.

        Uses F-test comparing restricted (without cause lags) vs
        unrestricted (full VAR) models.

        Args:
            Y: Multivariate time series (T, k).
            cause: Index of causing variable.
            effect: Index of effect variable.
            order: VAR lag order.

        Returns:
            (F-statistic, approximate p-value).
        """
        T, k = Y.shape
        n = T - order

        # Unrestricted model
        result_full = self.fit(Y, order)
        sse_u = np.sum(result_full.residuals[:, effect] ** 2)

        # Restricted model: remove cause variable lags
        Z_full = np.ones((n, 1 + order * k))
        for lag in range(order):
            Z_full[:, 1 + lag * k: 1 + (lag + 1) * k] = Y[order - lag - 1: T - lag - 1]

        # Remove cause columns
        remove_cols = [1 + lag * k + cause for lag in range(order)]
        keep_cols = [j for j in range(Z_full.shape[1]) if j not in remove_cols]
        Z_restr = Z_full[:, keep_cols]

        y_effect = Y[order:, effect]
        try:
            beta_r = np.linalg.lstsq(Z_restr, y_effect, rcond=None)[0]
            sse_r = np.sum((y_effect - Z_restr @ beta_r) ** 2)
        except np.linalg.LinAlgError:
            return 0.0, 1.0

        # F-test
        q = order  # Number of restrictions
        df = n - 1 - order * k
        if df <= 0 or sse_u <= 0:
            return 0.0, 1.0

        F = ((sse_r - sse_u) / q) / (sse_u / df)

        # Approximate p-value via F distribution CDF (chi-squared approx)
        # For exact, would need scipy.stats.f — use chi-squared approx here
        chi2 = F * q
        # Simple p-value approximation
        p_value = np.exp(-0.5 * chi2) if chi2 > 0 else 1.0

        return float(F), float(p_value)


# ============================================================================
# REGRESSION DIAGNOSTICS
# ============================================================================

class RegressionDiagnostics:
    """Diagnostic tests for regression models.

    Checks assumptions: homoscedasticity, multicollinearity,
    normality of residuals.

    Example:
        >>> diag = RegressionDiagnostics()
        >>> stat, p = diag.heteroscedasticity_test(residuals, X)
        >>> vifs = diag.vif(X)
    """

    @staticmethod
    def heteroscedasticity_test(residuals: np.ndarray, X: np.ndarray) -> tuple[float, float]:
        """Breusch-Pagan test for heteroscedasticity.

        Tests H₀: Var(ε_i) = σ² (constant) against
        H₁: Var(ε_i) = h(z_i'α) (depends on regressors).

        Procedure: Regress squared residuals on X, test nR² ~ χ²(p).

        Args:
            residuals: Regression residuals (n,).
            X: Regressor matrix (n, p).

        Returns:
            (test_statistic, approximate p-value).
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        n, p = X.shape
        e2 = residuals ** 2
        e2_norm = e2 / np.mean(e2)

        # Regress normalized squared residuals on X
        X_aug = np.column_stack([np.ones(n), X])
        try:
            beta = np.linalg.lstsq(X_aug, e2_norm, rcond=None)[0]
            fitted = X_aug @ beta
            ss_reg = np.sum((fitted - np.mean(e2_norm)) ** 2)
        except np.linalg.LinAlgError:
            return 0.0, 1.0

        stat = ss_reg / 2.0  # LM = nR² / 2 in some formulations
        # Under H₀, stat ~ χ²(p)
        # Approximate p-value
        p_value = np.exp(-0.5 * stat) if stat > 0 else 1.0

        return float(stat), float(p_value)

    @staticmethod
    def vif(X: np.ndarray) -> np.ndarray:
        """Variance Inflation Factors for multicollinearity detection.

        VIF_j = 1 / (1 - R²_j) where R²_j is the R² from regressing
        X_j on all other predictors. VIF > 10 indicates problematic
        multicollinearity.

        Args:
            X: Regressor matrix (n, p).

        Returns:
            VIF for each predictor (p,).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return np.array([1.0])

        n, p = X.shape
        vifs = np.ones(p)

        for j in range(p):
            y_j = X[:, j]
            X_rest = np.delete(X, j, axis=1)
            X_aug = np.column_stack([np.ones(n), X_rest])

            try:
                beta = np.linalg.lstsq(X_aug, y_j, rcond=None)[0]
                fitted = X_aug @ beta
                ss_res = np.sum((y_j - fitted) ** 2)
                ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            except np.linalg.LinAlgError:
                vifs[j] = np.inf

        return vifs

    @staticmethod
    def durbin_watson(residuals: np.ndarray) -> float:
        """Durbin-Watson test for autocorrelation in residuals.

        d = Σ(e_t - e_{t-1})² / Σe_t²
        d ≈ 2 indicates no autocorrelation.
        d < 2 suggests positive, d > 2 suggests negative autocorrelation.

        Args:
            residuals: Regression residuals (n,).

        Returns:
            Durbin-Watson statistic in [0, 4].
        """
        diff = np.diff(residuals)
        ss_e = np.sum(residuals ** 2)
        if ss_e == 0:
            return 2.0
        return float(np.sum(diff ** 2) / ss_e)

    @staticmethod
    def cooks_distance(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """Cook's distance for influence diagnostics.

        D_i = (e_i² / (p * MSE)) * (h_ii / (1 - h_ii)²)

        Args:
            X: Design matrix (n, p).
            residuals: Regression residuals (n,).

        Returns:
            Cook's distance for each observation (n,).
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        n, p = X.shape

        try:
            H = X @ np.linalg.solve(X.T @ X + 1e-12 * np.eye(p), X.T)
            h = np.diag(H)
        except np.linalg.LinAlgError:
            return np.zeros(n)

        mse = np.sum(residuals ** 2) / (n - p)
        if mse == 0:
            return np.zeros(n)

        D = (residuals ** 2 / (p * mse)) * (h / (1.0 - h) ** 2)
        return D
