"""
Multivariate Regression — Multiple Response Regression
=======================================================

Regression with multiple dependent variables simultaneously.

Core Concepts:
- Multivariate linear regression: Y = XB + E where Y has multiple columns
- Joint estimation captures cross-response correlations
- F-statistics test significance of each predictor across all responses

Mathematical Foundation:
B_hat = (X'X)^{-1} X'Y                    (OLS for each response, jointly)
Σ_hat = (Y - XB)'(Y - XB) / (n - p)      (Error covariance matrix)
Wilks' Λ = |E| / |E + H|                  (MANOVA test)

Applications to Warp Space:
- Joint prediction of multiple embedding dimensions
- Correlated scoring term regression
- Multi-output memory retrieval models

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
class MultivariateResult:
    """
    Result of multivariate regression.

    Attributes:
        coefficients: Regression coefficients (p x q), one column per response.
        residuals: Residual matrix (n x q).
        r_squared_per_response: R^2 for each response variable (q,).
        f_statistic: F-statistic for each predictor across all responses (p,).
        error_covariance: Estimated error covariance matrix (q x q).
        fitted: Fitted values (n x q).
        wilks_lambda: Wilks' Lambda for overall significance.
    """
    coefficients: np.ndarray
    residuals: np.ndarray
    r_squared_per_response: np.ndarray
    f_statistic: np.ndarray
    error_covariance: np.ndarray
    fitted: np.ndarray
    wilks_lambda: float


# ============================================================================
# MULTIVARIATE REGRESSION
# ============================================================================

class MultivariateRegression:
    """
    Multivariate linear regression: Y = X B + E.

    Y is (n x q) with q response variables.
    X is (n x p) with p predictor variables (including intercept if desired).
    B is (p x q) coefficient matrix.
    E is (n x q) error matrix with rows ~ N(0, Σ).
    """

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray,
            add_intercept: bool = True) -> MultivariateResult:
        """
        Fit multivariate regression via OLS.

        Args:
            X: Predictor matrix (n x p).
            Y: Response matrix (n x q).
            add_intercept: If True, prepend a column of ones to X.

        Returns:
            MultivariateResult with coefficients, residuals, diagnostics.
        """
        n = X.shape[0]
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if Y.shape[0] != n:
            raise ValueError("X and Y must have same number of rows")

        if add_intercept:
            X = np.column_stack([np.ones(n), X])

        p = X.shape[1]
        q = Y.shape[1]

        if n <= p:
            raise ValueError(f"Need n > p: n={n}, p={p}")

        # OLS: B = (X'X)^{-1} X'Y
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(p))
        B = XtX_inv @ (X.T @ Y)

        # Fitted values and residuals
        fitted = X @ B
        residuals = Y - fitted

        # Error covariance
        dof = n - p
        error_cov = (residuals.T @ residuals) / max(dof, 1)

        # R^2 per response
        ss_res = np.sum(residuals ** 2, axis=0)
        y_mean = np.mean(Y, axis=0)
        ss_tot = np.sum((Y - y_mean) ** 2, axis=0)
        r_squared = 1 - ss_res / (ss_tot + 1e-14)
        r_squared = np.clip(r_squared, 0, 1)

        # F-statistic for each predictor (testing B[j,:] = 0)
        f_stats = np.zeros(p)
        for j in range(p):
            # Partial F-test
            b_j = B[j, :]
            var_bj = XtX_inv[j, j]
            # For each response, F = b_j^2 / (var_bj * σ^2_k)
            # Average across responses
            f_vals = b_j ** 2 / (var_bj * np.diag(error_cov) + 1e-14)
            f_stats[j] = float(np.mean(f_vals))

        # Wilks' Lambda: |E| / |E + H|
        E = residuals.T @ residuals  # Error SS&CP matrix
        H = (fitted - y_mean).T @ (fitted - y_mean)  # Hypothesis SS&CP
        det_E = np.linalg.det(E + 1e-10 * np.eye(q))
        det_EH = np.linalg.det(E + H + 1e-10 * np.eye(q))
        wilks = det_E / (det_EH + 1e-14) if det_EH > 1e-14 else 1.0
        wilks = float(np.clip(wilks, 0, 1))

        return MultivariateResult(
            coefficients=B,
            residuals=residuals,
            r_squared_per_response=r_squared,
            f_statistic=f_stats,
            error_covariance=error_cov,
            fitted=fitted,
            wilks_lambda=wilks,
        )

    @staticmethod
    def predict(X: np.ndarray, result: MultivariateResult,
                add_intercept: bool = True) -> np.ndarray:
        """
        Predict using fitted multivariate regression.

        Args:
            X: New predictor data (m x p).
            result: Fitted MultivariateResult.
            add_intercept: Must match the fit() call.

        Returns:
            Predicted Y (m x q).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if add_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        return X @ result.coefficients

    @staticmethod
    def partial_r_squared(X: np.ndarray, Y: np.ndarray,
                          predictor_idx: int,
                          add_intercept: bool = True) -> np.ndarray:
        """
        Partial R^2 for a specific predictor: how much variance does it
        explain beyond the other predictors?

        Args:
            X: Full predictor matrix.
            Y: Response matrix.
            predictor_idx: Index of the predictor to evaluate (0-based,
                           before intercept insertion).
            add_intercept: Whether intercept is included.

        Returns:
            Partial R^2 per response (q,).
        """
        # Full model
        full_result = MultivariateRegression.fit(X, Y, add_intercept)
        ss_res_full = np.sum(full_result.residuals ** 2, axis=0)

        # Reduced model (drop the predictor)
        X_reduced = np.delete(X, predictor_idx, axis=1)
        reduced_result = MultivariateRegression.fit(
            X_reduced, Y, add_intercept
        )
        ss_res_reduced = np.sum(reduced_result.residuals ** 2, axis=0)

        partial = (ss_res_reduced - ss_res_full) / (ss_res_reduced + 1e-14)
        return np.clip(partial, 0, 1)
