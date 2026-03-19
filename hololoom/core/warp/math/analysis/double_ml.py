"""
Double Machine Learning — Debiased Causal Inference & Synthetic Control
========================================================================

Methods for estimating causal effects when confounders are high-dimensional
and require machine learning for nuisance parameter estimation.

Core Concepts:
- Double/Debiased ML: Chernozhukov et al. (2018) — cross-fitted residualization
- Synthetic Control: Abadie et al. (2003) — weighted donor pool for counterfactual

Mathematical Foundation:
DML: θ = E[ψ(W; θ, η₀)] = 0 where η₀ estimated by ML, cross-fitting removes bias
Neyman orthogonality: ∂_η E[ψ(W; θ₀, η)] |_{η=η₀} = 0
Synthetic control: Y₁ᵗ(0) ≈ Σ wⱼ Yⱼᵗ with wⱼ ≥ 0, Σ wⱼ = 1

Applications to Warp Space:
- Causal effect estimation of memory operations on downstream quality
- Counterfactual analysis for weaving cycle parameter changes
- Treatment effect of different embedding strategies on retrieval accuracy

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
class DMLResult:
    """Result of Double ML estimation.

    Attributes:
        ate: Average treatment effect estimate.
        se: Standard error of ATE.
        t_stat: t-statistic (ate / se).
        ci_lower: 95% confidence interval lower bound.
        ci_upper: 95% confidence interval upper bound.
        residuals_y: Residuals from outcome model.
        residuals_d: Residuals from treatment model.
    """
    ate: float
    se: float
    t_stat: float
    ci_lower: float
    ci_upper: float
    residuals_y: np.ndarray
    residuals_d: np.ndarray


@dataclass
class SyntheticResult:
    """Result of Synthetic Control method.

    Attributes:
        weights: Donor weights (n_donors,).
        counterfactual: Estimated Y(0) for treated unit in post-period.
        treatment_effect: Estimated treatment effect per period.
        pre_period_fit: RMSPE in pre-treatment period.
        donor_indices: Indices of donors with positive weight.
    """
    weights: np.ndarray
    counterfactual: np.ndarray
    treatment_effect: np.ndarray
    pre_period_fit: float
    donor_indices: np.ndarray


# ============================================================================
# DOUBLE MACHINE LEARNING
# ============================================================================

class DoubleML:
    """Double/Debiased Machine Learning for causal inference.

    Estimates the average treatment effect (ATE) of binary or continuous
    treatment D on outcome Y, controlling for high-dimensional confounders X.

    The procedure:
    1. Split data into K folds
    2. For each fold k:
       a. Estimate E[Y|X] on complement → residualize Y
       b. Estimate E[D|X] on complement → residualize D
    3. Regress Y-residuals on D-residuals → ATE

    Cross-fitting eliminates regularization bias that would persist in
    a naive plug-in estimator.

    Example:
        >>> dml = DoubleML()
        >>> Y = treatment_effect * D + X @ beta + noise
        >>> ate, se = dml.estimate(Y, D, X)
    """

    def estimate(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        model_y: Callable | None = None,
        model_d: Callable | None = None,
        n_folds: int = 5,
    ) -> tuple[float, float]:
        """Estimate ATE via cross-fitted DML.

        Args:
            Y: Outcome variable (n,).
            D: Treatment variable (n,).
            X: Confounders (n, p).
            model_y: Regression model for E[Y|X]. Callable(X_train, y_train, X_test) -> predictions.
                     Defaults to ridge regression.
            model_d: Regression model for E[D|X]. Same signature.
                     Defaults to ridge regression.
            n_folds: Number of cross-fitting folds.

        Returns:
            (ate, standard_error).
        """
        Y = np.asarray(Y, dtype=float).ravel()
        D = np.asarray(D, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(Y)

        if model_y is None:
            model_y = self._ridge_predict
        if model_d is None:
            model_d = self._ridge_predict

        # Generate fold indices
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        # Cross-fitted residuals
        V_y = np.zeros(n)
        V_d = np.zeros(n)

        for k in range(n_folds):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])

            # Residualize outcome
            V_y[test_idx] = Y[test_idx] - model_y(X[train_idx], Y[train_idx], X[test_idx])

            # Residualize treatment
            V_d[test_idx] = D[test_idx] - model_d(X[train_idx], D[train_idx], X[test_idx])

        # ATE: coefficient from regressing V_y on V_d
        denom = np.sum(V_d ** 2)
        if denom < 1e-12:
            logger.warning("Treatment residuals near zero — no variation after partialling out")
            return 0.0, np.inf

        ate = np.sum(V_d * V_y) / denom

        # Standard error (heteroscedasticity-robust)
        psi = V_d * (V_y - ate * V_d)
        se = np.sqrt(np.mean(psi ** 2) / denom ** 2 * n)

        return float(ate), float(se)

    def estimate_full(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        model_y: Callable | None = None,
        model_d: Callable | None = None,
        n_folds: int = 5,
    ) -> DMLResult:
        """Full DML estimation with confidence intervals.

        Args:
            Y, D, X, model_y, model_d, n_folds: See estimate().

        Returns:
            DMLResult with full inference.
        """
        Y = np.asarray(Y, dtype=float).ravel()
        D = np.asarray(D, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(Y)

        if model_y is None:
            model_y = self._ridge_predict
        if model_d is None:
            model_d = self._ridge_predict

        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        V_y = np.zeros(n)
        V_d = np.zeros(n)

        for k in range(n_folds):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            V_y[test_idx] = Y[test_idx] - model_y(X[train_idx], Y[train_idx], X[test_idx])
            V_d[test_idx] = D[test_idx] - model_d(X[train_idx], D[train_idx], X[test_idx])

        denom = np.sum(V_d ** 2)
        if denom < 1e-12:
            return DMLResult(0.0, np.inf, 0.0, -np.inf, np.inf, V_y, V_d)

        ate = np.sum(V_d * V_y) / denom

        psi = V_d * (V_y - ate * V_d)
        se = np.sqrt(np.mean(psi ** 2) / denom ** 2 * n)

        t_stat = ate / se if se > 0 else 0.0
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return DMLResult(
            ate=float(ate),
            se=float(se),
            t_stat=float(t_stat),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            residuals_y=V_y,
            residuals_d=V_d,
        )

    @staticmethod
    def _ridge_predict(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0
    ) -> np.ndarray:
        """Ridge regression predictor (default nuisance model).

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            alpha: Regularization strength.

        Returns:
            Predictions at X_test.
        """
        p = X_train.shape[1]
        XtX = X_train.T @ X_train + alpha * np.eye(p)
        try:
            beta = np.linalg.solve(XtX, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        return X_test @ beta


# ============================================================================
# SYNTHETIC CONTROL
# ============================================================================

class SyntheticControl:
    """Abadie Synthetic Control Method.

    Constructs a weighted combination of untreated donor units to
    approximate the counterfactual outcome for a treated unit.
    Weights are chosen to match pre-treatment outcomes.

    min_w ||Y₁ᵖʳᵉ - Y₀ᵖʳᵉ w||² subject to w ≥ 0, Σw = 1

    Example:
        >>> sc = SyntheticControl()
        >>> result = sc.fit(treated_series, donor_matrix, pre_period=50)
        >>> print(f"ATT: {result.treatment_effect.mean():.3f}")
    """

    def fit(
        self,
        treated: np.ndarray,
        donors: np.ndarray,
        pre_period: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> SyntheticResult:
        """Fit synthetic control weights.

        Args:
            treated: Outcome series for treated unit (T,).
            donors: Outcome matrix for donor units (T, J).
            pre_period: Number of pre-treatment periods.
            max_iter: Maximum iterations for weight optimization.
            tol: Convergence tolerance.

        Returns:
            SyntheticResult with weights and counterfactual.
        """
        treated = np.asarray(treated, dtype=float).ravel()
        donors = np.asarray(donors, dtype=float)
        T = len(treated)
        J = donors.shape[1]

        if pre_period >= T:
            raise ValueError("pre_period must be < total periods")
        if donors.shape[0] != T:
            raise ValueError("donors must have same number of periods as treated")

        # Pre-treatment data
        y_pre = treated[:pre_period]
        D_pre = donors[:pre_period]

        # Solve constrained quadratic program via projected gradient descent
        # min ||y_pre - D_pre @ w||^2 s.t. w >= 0, sum(w) = 1
        w = np.ones(J) / J  # Uniform initialization

        for iteration in range(max_iter):
            # Gradient: -2 * D_pre.T @ (y_pre - D_pre @ w)
            residual = y_pre - D_pre @ w
            grad = -2.0 * D_pre.T @ residual

            # Step size via backtracking line search
            step = 1.0 / (np.linalg.norm(D_pre.T @ D_pre) + 1e-8)

            # Gradient step
            w_new = w - step * grad

            # Project onto simplex (w >= 0, sum = 1)
            w_new = self._project_simplex(w_new)

            # Check convergence
            if np.linalg.norm(w_new - w) < tol:
                w = w_new
                break
            w = w_new

        # Compute counterfactual and treatment effect
        counterfactual = donors @ w
        treatment_effect = treated - counterfactual

        # Pre-period RMSPE
        pre_rmspe = np.sqrt(np.mean((treated[:pre_period] - counterfactual[:pre_period]) ** 2))

        # Donors with positive weight
        donor_idx = np.where(w > 1e-6)[0]

        return SyntheticResult(
            weights=w,
            counterfactual=counterfactual,
            treatment_effect=treatment_effect,
            pre_period_fit=float(pre_rmspe),
            donor_indices=donor_idx,
        )

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """Project vector onto probability simplex.

        Uses the algorithm from Duchi et al. (2008):
        sort, find threshold, clip and shift.

        Args:
            v: Input vector (J,).

        Returns:
            Projection onto {w: w >= 0, sum(w) = 1}.
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho = np.max(np.where(u * np.arange(1, n + 1) > cssv)[0]) + 1
        theta = cssv[rho - 1] / rho
        return np.maximum(v - theta, 0.0)

    def placebo_test(
        self,
        treated: np.ndarray,
        donors: np.ndarray,
        pre_period: int,
    ) -> tuple[np.ndarray, float]:
        """Placebo test: run synthetic control on each donor as if treated.

        Produces distribution of placebo effects for inference.

        Args:
            treated: Treated unit series.
            donors: Donor matrix.
            pre_period: Pre-treatment periods.

        Returns:
            (placebo_effects (J, T_post), p_value) where p_value is
            fraction of placebo effects larger than actual.
        """
        T = len(treated)
        J = donors.shape[1]
        T_post = T - pre_period

        # Actual effect
        result_actual = self.fit(treated, donors, pre_period)
        actual_effect = np.mean(np.abs(result_actual.treatment_effect[pre_period:]))

        # Placebo effects
        placebo_effects = np.zeros((J, T_post))
        count_larger = 0

        for j in range(J):
            # Treat donor j as treated, use remaining + actual treated as donors
            donor_j = donors[:, j]
            remaining = np.delete(donors, j, axis=1)
            donor_pool = np.column_stack([remaining, treated])

            try:
                result_j = self.fit(donor_j, donor_pool, pre_period)
                placebo_effects[j] = result_j.treatment_effect[pre_period:]
                if np.mean(np.abs(placebo_effects[j])) >= actual_effect:
                    count_larger += 1
            except (ValueError, np.linalg.LinAlgError):
                placebo_effects[j] = 0.0

        p_value = (count_larger + 1) / (J + 1)
        return placebo_effects, float(p_value)
