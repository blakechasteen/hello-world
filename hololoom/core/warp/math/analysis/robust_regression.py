"""
Robust Regression — Outlier-Resistant Linear Models
====================================================

Regression methods that downweight or exclude outliers, providing
reliable estimates when data violates Gaussian assumptions.

Core Concepts:
- Huber Regression: Combines L2 (small residuals) and L1 (large residuals)
- RANSAC: Random sample consensus — fit to inliers, reject outliers
- Theil-Sen: Median of pairwise slopes — 29.3% breakdown point

Mathematical Foundation:
Huber loss: ρ(r) = r²/2 if |r| ≤ δ, else δ(|r| - δ/2)
RANSAC: argmax_S |{i : |yᵢ - xᵢᵀβ_S| < ε}| over random subsets S
Theil-Sen: β = median{(yⱼ - yᵢ)/(xⱼ - xᵢ)} for all pairs i < j

Applications to Warp Space:
- Robust trend estimation in noisy scoring term evolution
- Outlier-resistant embedding trajectory fitting
- Reliable parameter estimation for stiff systems

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RegressionResult:
    """Result of robust regression.

    Attributes:
        coefficients: Regression coefficients β (n_features,) or (n_features+1,) with intercept
        intercept: Intercept term
        residuals: Fitting residuals (n_samples,)
        inlier_mask: Boolean mask of inliers (for RANSAC)
        score: R² or pseudo-R² on inliers
        n_iterations: Iterations used (IRLS or RANSAC trials)
    """
    coefficients: np.ndarray
    intercept: float = 0.0
    residuals: np.ndarray = field(default_factory=lambda: np.empty(0))
    inlier_mask: np.ndarray | None = None
    score: float = 0.0
    n_iterations: int = 0


# ============================================================================
# HUBER REGRESSION
# ============================================================================

class HuberRegression:
    """Huber regression via iteratively reweighted least squares (IRLS).

    Uses Huber's loss function which is quadratic for small residuals
    and linear for large ones, controlled by the threshold δ.

    ρ(r) = { r²/2        if |r| ≤ δ
           { δ(|r| - δ/2) otherwise

    Example:
        >>> hr = HuberRegression()
        >>> X = np.random.randn(100, 2)
        >>> y = X @ [3, -1] + 0.5 * np.random.randn(100)
        >>> y[0] = 100  # outlier
        >>> result = hr.fit(X, y, delta=1.35)
        >>> np.allclose(result.coefficients, [3, -1], atol=0.5)
        True
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        delta: float = 1.35,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
    ) -> RegressionResult:
        """Fit Huber regression via IRLS.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            delta: Huber threshold (default 1.35 ≈ 95% efficiency at Gaussian)
            max_iter: Maximum IRLS iterations
            tol: Convergence tolerance
            fit_intercept: Whether to fit an intercept

        Returns:
            RegressionResult with robust coefficients
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if fit_intercept:
            X_aug = np.column_stack([np.ones(len(X)), X])
        else:
            X_aug = X.copy()

        n, p = X_aug.shape

        # Initial OLS estimate
        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(p)

        for iteration in range(max_iter):
            residuals = y - X_aug @ beta
            sigma = np.median(np.abs(residuals)) * 1.4826  # MAD estimate
            if sigma < 1e-10:
                sigma = 1.0

            scaled_r = residuals / sigma

            # Huber weights
            weights = np.where(
                np.abs(scaled_r) <= delta,
                1.0,
                delta / (np.abs(scaled_r) + 1e-10),
            )

            # Weighted least squares
            W = np.diag(weights)
            try:
                beta_new = np.linalg.solve(
                    X_aug.T @ W @ X_aug + 1e-10 * np.eye(p),
                    X_aug.T @ W @ y,
                )
            except np.linalg.LinAlgError:
                break

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        residuals = y - X_aug @ beta

        if fit_intercept:
            intercept = beta[0]
            coefficients = beta[1:]
        else:
            intercept = 0.0
            coefficients = beta

        # R² score
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return RegressionResult(
            coefficients=coefficients,
            intercept=intercept,
            residuals=residuals,
            score=r2,
            n_iterations=iteration + 1,
        )


# ============================================================================
# RANSAC REGRESSION
# ============================================================================

class RANSACRegression:
    """Random Sample Consensus (RANSAC) regression.

    Repeatedly samples minimal subsets, fits a model, counts inliers,
    and keeps the model with the most inliers.

    Example:
        >>> ransac = RANSACRegression()
        >>> X = np.random.randn(100, 1)
        >>> y = 2 * X.ravel() + 1 + 0.1 * np.random.randn(100)
        >>> y[:10] = 50  # outliers
        >>> result = ransac.fit(X, y, min_samples=10, residual_threshold=0.5)
        >>> np.sum(result.inlier_mask)  # ~90 inliers
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int | None = None,
        residual_threshold: float | None = None,
        max_trials: int = 100,
        fit_intercept: bool = True,
    ) -> RegressionResult:
        """Fit RANSAC regression.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            min_samples: Minimum samples for model fit (default: n_features + 1)
            residual_threshold: Inlier threshold (default: MAD of residuals)
            max_trials: Maximum RANSAC iterations
            fit_intercept: Whether to fit intercept

        Returns:
            RegressionResult with best inlier consensus
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        if fit_intercept:
            X_aug = np.column_stack([np.ones(n), X])
        else:
            X_aug = X.copy()

        p_aug = X_aug.shape[1]

        if min_samples is None:
            min_samples = p_aug

        if residual_threshold is None:
            # Use MAD of OLS residuals
            try:
                beta_ols = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                res_ols = y - X_aug @ beta_ols
                residual_threshold = np.median(np.abs(res_ols)) * 1.4826 * 3
            except np.linalg.LinAlgError:
                residual_threshold = np.std(y)

        best_n_inliers = 0
        best_beta = None
        best_inlier_mask = None

        for trial in range(max_trials):
            # Random subset
            idx = np.random.choice(n, size=min_samples, replace=False)
            X_sub = X_aug[idx]
            y_sub = y[idx]

            try:
                beta = np.linalg.lstsq(X_sub, y_sub, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            residuals = np.abs(y - X_aug @ beta)
            inlier_mask = residuals < residual_threshold
            n_inliers = np.sum(inlier_mask)

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inlier_mask = inlier_mask
                best_beta = beta

        if best_beta is None:
            logger.warning("RANSAC found no valid model")
            return RegressionResult(
                coefficients=np.zeros(p),
                n_iterations=max_trials,
                score=0.0,
            )

        # Refit on all inliers
        X_inliers = X_aug[best_inlier_mask]
        y_inliers = y[best_inlier_mask]
        try:
            best_beta = np.linalg.lstsq(X_inliers, y_inliers, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass

        residuals = y - X_aug @ best_beta

        if fit_intercept:
            intercept = best_beta[0]
            coefficients = best_beta[1:]
        else:
            intercept = 0.0
            coefficients = best_beta

        # R² on inliers
        y_in = y[best_inlier_mask]
        r_in = residuals[best_inlier_mask]
        ss_res = np.sum(r_in ** 2)
        ss_tot = np.sum((y_in - np.mean(y_in)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return RegressionResult(
            coefficients=coefficients,
            intercept=intercept,
            residuals=residuals,
            inlier_mask=best_inlier_mask,
            score=r2,
            n_iterations=max_trials,
        )


# ============================================================================
# THEIL-SEN ESTIMATOR
# ============================================================================

class TheilSenEstimator:
    """Theil-Sen robust regression (median of pairwise slopes).

    For univariate: β = median{(yⱼ - yᵢ)/(xⱼ - xᵢ)} for i < j
    For multivariate: uses subsampling with median of slopes.

    Breakdown point: 29.3% (can tolerate ~29% outliers).

    Example:
        >>> ts = TheilSenEstimator()
        >>> X = np.random.randn(100, 1)
        >>> y = 3 * X.ravel() + 2 + 0.1 * np.random.randn(100)
        >>> y[:20] = 100  # 20% outliers
        >>> result = ts.fit(X, y)
        >>> abs(result.coefficients[0] - 3) < 0.5
        True
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool = True,
        max_subpopulation: int = 10000,
    ) -> RegressionResult:
        """Fit Theil-Sen estimator.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            fit_intercept: Whether to fit intercept
            max_subpopulation: Max number of pairs to sample

        Returns:
            RegressionResult with robust slope estimate
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        if p == 1:
            coef, intercept = self._fit_univariate(
                X.ravel(), y, fit_intercept, max_subpopulation
            )
            coefficients = np.array([coef])
        else:
            coefficients, intercept = self._fit_multivariate(
                X, y, fit_intercept, max_subpopulation
            )

        if fit_intercept:
            residuals = y - X @ coefficients - intercept
        else:
            intercept = 0.0
            residuals = y - X @ coefficients

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return RegressionResult(
            coefficients=coefficients,
            intercept=intercept,
            residuals=residuals,
            score=r2,
        )

    @staticmethod
    def _fit_univariate(
        x: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        max_pairs: int,
    ) -> tuple[float, float]:
        """Univariate Theil-Sen: median of pairwise slopes."""
        n = len(x)
        n_pairs = n * (n - 1) // 2

        if n_pairs <= max_pairs:
            slopes = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j] - x[i]
                    if abs(dx) > 1e-12:
                        slopes.append((y[j] - y[i]) / dx)
        else:
            slopes = []
            for _ in range(max_pairs):
                i, j = np.random.choice(n, 2, replace=False)
                dx = x[j] - x[i]
                if abs(dx) > 1e-12:
                    slopes.append((y[j] - y[i]) / dx)

        if len(slopes) == 0:
            return 0.0, np.median(y) if fit_intercept else 0.0

        slope = float(np.median(slopes))

        if fit_intercept:
            intercept = float(np.median(y - slope * x))
        else:
            intercept = 0.0

        return slope, intercept

    @staticmethod
    def _fit_multivariate(
        X: np.ndarray,
        y: np.ndarray,
        fit_intercept: bool,
        max_samples: int,
    ) -> tuple[np.ndarray, float]:
        """Multivariate Theil-Sen via subsampled median."""
        n, p = X.shape
        n_sub = min(max_samples, n * (n - 1) // 2)

        coefficients_list = []
        for _ in range(n_sub):
            idx = np.random.choice(n, size=p + (1 if fit_intercept else 0),
                                   replace=False)
            X_sub = X[idx]
            y_sub = y[idx]

            if fit_intercept:
                X_aug = np.column_stack([np.ones(len(idx)), X_sub])
            else:
                X_aug = X_sub

            try:
                beta = np.linalg.lstsq(X_aug, y_sub, rcond=None)[0]
                coefficients_list.append(beta)
            except np.linalg.LinAlgError:
                continue

        if len(coefficients_list) == 0:
            return np.zeros(p), 0.0

        betas = np.array(coefficients_list)
        median_beta = np.median(betas, axis=0)

        if fit_intercept:
            return median_beta[1:], float(median_beta[0])
        return median_beta, 0.0
