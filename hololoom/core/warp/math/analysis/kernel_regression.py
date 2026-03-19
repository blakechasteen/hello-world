"""
Kernel Regression — Nonparametric Function Estimation
======================================================

Nonparametric regression and density estimation using kernel methods.

Core Concepts:
- Nadaraya-Watson: weighted average with kernel weights
  ŷ(x) = Σ K_h(x - x_i) y_i / Σ K_h(x - x_i)
- Local linear regression: fits a line at each prediction point
  using kernel-weighted least squares — reduces boundary bias
- Kernel density estimation (KDE): estimate probability density
  p̂(x) = (1/nh) Σ K((x - x_i)/h)
- RKHS: Reproducing Kernel Hilbert Space for regularized estimation

Mathematical Foundation:
Kernel function K(u): symmetric, integrates to 1, controls smoothing.
  RBF: K(u) = (1/√(2π)) exp(-u²/2)
  Epanechnikov: K(u) = ¾(1 - u²) for |u| ≤ 1

Bandwidth h controls bias-variance tradeoff:
  h large → high bias, low variance (oversmoothing)
  h small → low bias, high variance (undersmoothing)
  Optimal (Silverman): h = 0.9 * min(σ, IQR/1.34) * n^{-1/5}

Representer theorem: solution in RKHS is α = (K + λI)^{-1} y
where K is the Gram matrix K_{ij} = k(x_i, x_j).

Self-contained kernel implementations (no imports from bandits/).

Applications to Warp Space:
- Smooth interpolation of sparse embedding spaces
- Nonparametric density estimation for scoring distributions
- Bandwidth-adaptive local regression for time-varying signals
- Kernel-based similarity in memory retrieval

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
class KernelRegressionResult:
    """Result of kernel regression.

    Attributes:
        predictions: Predicted values at test points, shape (n_test,).
        bandwidth: Bandwidth used for the kernel.
        effective_df: Effective degrees of freedom (trace of hat matrix).
        residuals: Residuals y_train - ŷ_train (on training set),
                   shape (n_train,).
    """
    predictions: np.ndarray
    bandwidth: float
    effective_df: float
    residuals: np.ndarray


# ============================================================================
# KERNEL FUNCTIONS
# ============================================================================

def rbf_kernel(x1: np.ndarray, x2: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Radial basis function (Gaussian) kernel.

    K(x1, x2; h) = exp(-||x1 - x2||² / (2h²))

    Args:
        x1: First set of points, shape (n1,) or (n1, d).
        x2: Second set of points, shape (n2,) or (n2, d).
        bandwidth: Kernel bandwidth h.

    Returns:
        Kernel matrix, shape (n1, n2).
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    # Compute pairwise squared distances
    sq_dists = _pairwise_sq_distances(x1, x2)

    return np.exp(-sq_dists / (2.0 * bandwidth**2))


def epanechnikov_kernel(x1: np.ndarray, x2: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Epanechnikov kernel (optimal MSE-minimizing kernel).

    K(u) = (3/4)(1 - u²) for |u| ≤ 1, else 0
    where u = ||x1 - x2|| / h

    Args:
        x1: First set of points.
        x2: Second set of points.
        bandwidth: Kernel bandwidth h.

    Returns:
        Kernel matrix, shape (n1, n2).
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    sq_dists = _pairwise_sq_distances(x1, x2)
    u_sq = sq_dists / bandwidth**2

    K = 0.75 * (1.0 - u_sq)
    K[u_sq > 1.0] = 0.0

    return K


def tricube_kernel(x1: np.ndarray, x2: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Tricube kernel (used in LOESS/LOWESS).

    K(u) = (70/81)(1 - |u|³)³ for |u| ≤ 1, else 0

    Args:
        x1: First set of points.
        x2: Second set of points.
        bandwidth: Kernel bandwidth h.

    Returns:
        Kernel matrix, shape (n1, n2).
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    dists = np.sqrt(_pairwise_sq_distances(x1, x2))
    u = dists / bandwidth

    K = (70.0 / 81.0) * (1.0 - np.abs(u)**3)**3
    K[np.abs(u) > 1.0] = 0.0

    return K


def _pairwise_sq_distances(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances.

    Args:
        x1: Shape (n1, d).
        x2: Shape (n2, d).

    Returns:
        Distance matrix, shape (n1, n2).
    """
    # ||x1 - x2||² = ||x1||² + ||x2||² - 2 x1 · x2
    sq1 = np.sum(x1**2, axis=1, keepdims=True)  # (n1, 1)
    sq2 = np.sum(x2**2, axis=1, keepdims=True)  # (n2, 1)
    cross = x1 @ x2.T  # (n1, n2)
    sq_dists = sq1 + sq2.T - 2.0 * cross
    return np.maximum(sq_dists, 0.0)


# ============================================================================
# NADARAYA-WATSON ESTIMATOR
# ============================================================================

class NadarayaWatson:
    """
    Nadaraya-Watson kernel regression estimator.

    ŷ(x) = Σ_i K_h(x - x_i) y_i / Σ_i K_h(x - x_i)

    A locally-weighted average where K_h is a kernel function with
    bandwidth h. Simple, consistent, but suffers from boundary bias.
    """

    @staticmethod
    def fit_predict(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        kernel: str = 'rbf',
        bandwidth: float | None = None
    ) -> KernelRegressionResult:
        """
        Fit and predict using Nadaraya-Watson estimator.

        Args:
            X_train: Training features, shape (n_train,) or (n_train, d).
            y_train: Training targets, shape (n_train,).
            X_test: Test features, shape (n_test,) or (n_test, d).
            kernel: Kernel type: 'rbf', 'epanechnikov', or 'tricube'.
            bandwidth: Kernel bandwidth h. If None, uses Silverman's rule.

        Returns:
            KernelRegressionResult with predictions and diagnostics.
        """
        X_train = np.atleast_2d(np.asarray(X_train, dtype=np.float64))
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        X_test = np.atleast_2d(np.asarray(X_test, dtype=np.float64))

        if X_train.shape[0] == 1 and len(y_train) > 1:
            X_train = X_train.T
        if X_test.shape[0] == 1 and X_test.shape[1] > 1:
            X_test = X_test.T

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        if bandwidth is None:
            bandwidth = KernelDensityEstimation.bandwidth_selection(X_train)

        kernel_fn = _get_kernel_fn(kernel)

        # Compute kernel weights: W[i,j] = K_h(X_test[i] - X_train[j])
        W = kernel_fn(X_test, X_train, bandwidth)  # (n_test, n_train)

        # Normalize weights
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum = np.maximum(W_sum, 1e-300)
        W_norm = W / W_sum

        # Predictions
        predictions = W_norm @ y_train

        # Effective degrees of freedom (hat matrix trace on training set)
        W_train = kernel_fn(X_train, X_train, bandwidth)
        W_train_sum = W_train.sum(axis=1, keepdims=True)
        W_train_sum = np.maximum(W_train_sum, 1e-300)
        H = W_train / W_train_sum
        effective_df = float(np.trace(H))

        # Residuals on training data
        y_hat_train = H @ y_train
        residuals = y_train - y_hat_train

        return KernelRegressionResult(
            predictions=predictions,
            bandwidth=bandwidth,
            effective_df=effective_df,
            residuals=residuals,
        )

    @staticmethod
    def cross_validate_bandwidth(
        X: np.ndarray,
        y: np.ndarray,
        bandwidths: np.ndarray,
        kernel: str = 'rbf'
    ) -> float:
        """
        Select bandwidth via leave-one-out cross-validation.

        For each bandwidth h, computes the LOO-CV score:
          CV(h) = (1/n) Σ_i (y_i - ŷ_{-i}(x_i))²

        where ŷ_{-i} is the prediction at x_i using all data except (x_i, y_i).
        Uses the hat matrix shortcut to avoid refitting n times.

        Args:
            X: Features, shape (n,) or (n, d).
            y: Targets, shape (n,).
            bandwidths: Array of candidate bandwidths to evaluate.
            kernel: Kernel type.

        Returns:
            Optimal bandwidth (float) minimizing LOO-CV error.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.shape[0] == 1 and len(y) > 1:
            X = X.T
        bandwidths = np.asarray(bandwidths, dtype=np.float64)

        n = len(y)
        kernel_fn = _get_kernel_fn(kernel)

        best_h = bandwidths[0]
        best_cv = np.inf

        for h in bandwidths:
            # Compute hat matrix H where ŷ = H y
            W = kernel_fn(X, X, h)  # (n, n)
            W_sum = W.sum(axis=1, keepdims=True)
            W_sum = np.maximum(W_sum, 1e-300)
            H = W / W_sum

            y_hat = H @ y
            residuals = y - y_hat

            # LOO-CV via hat matrix: (y_i - ŷ_{-i})² = (residual_i / (1 - H_ii))²
            diag_H = np.diag(H)
            denom = 1.0 - diag_H
            denom = np.maximum(np.abs(denom), 1e-10) * np.sign(denom + 1e-300)
            loo_residuals = residuals / denom
            cv_score = float(np.mean(loo_residuals**2))

            if cv_score < best_cv:
                best_cv = cv_score
                best_h = h

        logger.info(f"LOO-CV selected bandwidth h={best_h:.6f} (CV={best_cv:.6f})")
        return float(best_h)


# ============================================================================
# LOCAL LINEAR REGRESSION
# ============================================================================

class LocalLinearRegression:
    """
    Local linear regression (locally weighted least squares).

    At each prediction point x, fits a weighted linear model:
      min_β Σ_i K_h(x - x_i) (y_i - β_0 - β_1 (x_i - x))²

    The prediction is ŷ(x) = β_0.

    Advantages over Nadaraya-Watson:
    - Reduces boundary bias (automatic bias correction)
    - Adapts to local slope
    - Minimax optimal for function estimation
    """

    @staticmethod
    def fit_predict(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        bandwidth: float,
        kernel: str = 'rbf'
    ) -> KernelRegressionResult:
        """
        Local linear regression predictions.

        For each test point x_0, solves the weighted least squares problem:
          min_{β_0, β_1} Σ_i K_h(x_0 - x_i) [y_i - β_0 - β_1^T (x_i - x_0)]²

        Args:
            X_train: Training features, shape (n_train,) or (n_train, d).
            y_train: Training targets, shape (n_train,).
            X_test: Test features, shape (n_test,) or (n_test, d).
            bandwidth: Kernel bandwidth.
            kernel: Kernel type.

        Returns:
            KernelRegressionResult with predictions.
        """
        X_train = np.atleast_2d(np.asarray(X_train, dtype=np.float64))
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        X_test = np.atleast_2d(np.asarray(X_test, dtype=np.float64))

        if X_train.shape[0] == 1 and len(y_train) > 1:
            X_train = X_train.T
        if X_test.shape[0] == 1 and X_test.shape[1] > 1:
            X_test = X_test.T

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        d = X_train.shape[1]

        kernel_fn = _get_kernel_fn(kernel)
        predictions = np.zeros(n_test, dtype=np.float64)

        for i in range(n_test):
            x0 = X_test[i]

            # Kernel weights
            w = kernel_fn(x0.reshape(1, -1), X_train, bandwidth).ravel()  # (n_train,)

            # Design matrix: [1, (x_j - x_0)]
            diffs = X_train - x0  # (n_train, d)
            Z = np.hstack([np.ones((n_train, 1)), diffs])  # (n_train, 1+d)

            # Weighted least squares: β = (Z^T W Z)^{-1} Z^T W y
            W = np.diag(w)
            ZTWZ = Z.T @ W @ Z  # (1+d, 1+d)
            ZTWy = Z.T @ (w * y_train)  # (1+d,)

            # Regularize for stability
            reg = 1e-10 * np.eye(1 + d)
            try:
                beta = np.linalg.solve(ZTWZ + reg, ZTWy)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(ZTWZ + reg, ZTWy, rcond=None)[0]

            predictions[i] = beta[0]  # ŷ(x_0) = β_0

        # Compute effective df and residuals on training data
        residuals = np.zeros(n_train, dtype=np.float64)
        hat_diag = np.zeros(n_train, dtype=np.float64)

        for i in range(n_train):
            x0 = X_train[i]
            w = kernel_fn(x0.reshape(1, -1), X_train, bandwidth).ravel()
            diffs = X_train - x0
            Z = np.hstack([np.ones((n_train, 1)), diffs])
            W = np.diag(w)
            ZTWZ = Z.T @ W @ Z
            reg = 1e-10 * np.eye(1 + d)
            try:
                A = np.linalg.solve(ZTWZ + reg, Z.T @ W)
            except np.linalg.LinAlgError:
                A = np.linalg.lstsq(ZTWZ + reg, Z.T @ W, rcond=None)[0]

            e1 = np.zeros(1 + d)
            e1[0] = 1.0
            h_row = e1 @ A  # (n_train,)
            y_hat_i = np.dot(h_row, y_train)
            residuals[i] = y_train[i] - y_hat_i
            hat_diag[i] = h_row[i]

        effective_df = float(np.sum(hat_diag))

        return KernelRegressionResult(
            predictions=predictions,
            bandwidth=bandwidth,
            effective_df=effective_df,
            residuals=residuals,
        )


# ============================================================================
# KERNEL DENSITY ESTIMATION
# ============================================================================

class KernelDensityEstimation:
    """
    Kernel density estimation (KDE).

    Estimates the probability density function from samples:
      p̂(x) = (1 / n h) Σ_i K((x - x_i) / h)

    Nonparametric: makes no assumptions about the underlying distribution.
    """

    def __init__(self) -> None:
        self._X: np.ndarray | None = None
        self._bandwidth: float = 1.0
        self._n: int = 0

    def fit(
        self,
        X: np.ndarray,
        bandwidth: float | None = None
    ) -> 'KernelDensityEstimation':
        """
        Fit the KDE to data.

        Args:
            X: Data points, shape (n,) or (n, d).
            bandwidth: Kernel bandwidth. If None, uses Silverman's rule.

        Returns:
            self (for chaining).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T

        self._X = X
        self._n = X.shape[0]

        if bandwidth is None:
            self._bandwidth = KernelDensityEstimation.bandwidth_selection(X)
        else:
            self._bandwidth = bandwidth

        logger.info(f"KDE fitted: n={self._n}, h={self._bandwidth:.6f}")
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate log density at given points.

        p̂(x) = (1 / n h^d) Σ_i K((x - x_i) / h)
        Returns log(p̂(x)) for numerical stability.

        Args:
            X: Points to evaluate, shape (m,) or (m, d).

        Returns:
            Log density values, shape (m,).
        """
        if self._X is None:
            raise RuntimeError("Must call fit() before score_samples()")

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T

        n = self._n
        d = self._X.shape[1]
        h = self._bandwidth

        # Compute kernel values
        K = rbf_kernel(X, self._X, h)  # (m, n)

        # Density: (1/n) * (1/(2πh²)^{d/2}) * Σ_i K(...)
        # Since rbf_kernel already computes exp(-||·||²/(2h²)),
        # the normalization is (2π h²)^{-d/2}
        norm_const = (2.0 * np.pi * h**2) ** (-d / 2.0)
        density = norm_const * K.mean(axis=1)  # (m,)

        log_density = np.log(np.maximum(density, 1e-300))

        return log_density

    def density(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate density (not log) at given points.

        Args:
            X: Points to evaluate.

        Returns:
            Density values, shape (m,).
        """
        return np.exp(self.score_samples(X))

    @staticmethod
    def bandwidth_selection(
        X: np.ndarray,
        method: str = 'silverman'
    ) -> float:
        """
        Automatic bandwidth selection.

        Silverman's rule of thumb:
          h = 0.9 * min(σ, IQR/1.34) * n^{-1/5}

        where σ is the standard deviation and IQR is the interquartile range.
        This is optimal for Gaussian data but provides a reasonable starting
        point for most distributions.

        Scott's rule:
          h = σ * n^{-1/(d+4)}

        Args:
            X: Data, shape (n,) or (n, d).
            method: 'silverman' or 'scott'.

        Returns:
            Bandwidth h (float).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T

        n, d = X.shape

        if method == 'silverman':
            # Use average across dimensions for multivariate
            bandwidths = []
            for dim in range(d):
                x = X[:, dim]
                sigma = np.std(x, ddof=1)
                q75, q25 = np.percentile(x, [75, 25])
                iqr = q75 - q25
                spread = min(sigma, iqr / 1.34) if iqr > 0 else sigma
                spread = max(spread, 1e-10)
                h = 0.9 * spread * n ** (-0.2)
                bandwidths.append(h)
            return float(np.mean(bandwidths))

        elif method == 'scott':
            sigma = np.mean(np.std(X, axis=0, ddof=1))
            sigma = max(sigma, 1e-10)
            return float(sigma * n ** (-1.0 / (d + 4)))

        else:
            raise ValueError(f"Unknown method: {method}. Use 'silverman' or 'scott'.")


# ============================================================================
# REPRODUCING KERNEL HILBERT SPACE
# ============================================================================

class RKHS:
    """
    Reproducing Kernel Hilbert Space methods.

    The RKHS defined by a positive definite kernel k provides a function
    space where optimization has a finite-dimensional representation via
    the Representer Theorem.
    """

    @staticmethod
    def gram_matrix(
        X: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        bandwidth: float = 1.0
    ) -> np.ndarray:
        """
        Compute the Gram (kernel) matrix.

        K[i,j] = k(x_i, x_j)

        The Gram matrix is symmetric positive semi-definite for any
        valid kernel function.

        Args:
            X: Data points, shape (n,) or (n, d).
            kernel_fn: Kernel function k(X1, X2, h) -> matrix.
            bandwidth: Kernel bandwidth.

        Returns:
            Gram matrix K, shape (n, n).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T

        K = kernel_fn(X, X, bandwidth)
        # Symmetrize (kernel_fn should already be symmetric, but enforce)
        K = 0.5 * (K + K.T)

        return K

    @staticmethod
    def representer_theorem_solution(
        K: np.ndarray,
        y: np.ndarray,
        lam: float
    ) -> np.ndarray:
        """
        Representer Theorem solution for kernel ridge regression.

        The minimizer of:
          min_f ||y - f||² + λ ||f||²_H

        in the RKHS H has the form f(x) = Σ_i α_i k(x_i, x)
        where:
          α = (K + λI)^{-1} y

        Args:
            K: Gram matrix, shape (n, n).
            y: Target values, shape (n,).
            lam: Regularization parameter λ > 0.

        Returns:
            Coefficient vector α, shape (n,).
        """
        K = np.asarray(K, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = K.shape[0]

        A = K + lam * np.eye(n)

        try:
            alpha = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(A, y, rcond=None)[0]

        return alpha

    @staticmethod
    def predict(
        X_train: np.ndarray,
        X_test: np.ndarray,
        alpha: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        bandwidth: float = 1.0
    ) -> np.ndarray:
        """
        Predict using learned RKHS coefficients.

        f(x) = Σ_i α_i k(x_i, x)

        Args:
            X_train: Training points, shape (n_train,) or (n_train, d).
            X_test: Test points, shape (n_test,) or (n_test, d).
            alpha: Coefficients from representer_theorem_solution.
            kernel_fn: Kernel function.
            bandwidth: Kernel bandwidth.

        Returns:
            Predictions, shape (n_test,).
        """
        X_train = np.atleast_2d(np.asarray(X_train, dtype=np.float64))
        X_test = np.atleast_2d(np.asarray(X_test, dtype=np.float64))

        if X_train.shape[0] == 1 and len(alpha) > 1:
            X_train = X_train.T
        if X_test.shape[0] == 1 and X_test.shape[1] > 1:
            X_test = X_test.T

        K_test = kernel_fn(X_test, X_train, bandwidth)  # (n_test, n_train)
        return K_test @ alpha

    @staticmethod
    def effective_dimensionality(K: np.ndarray, lam: float) -> float:
        """
        Effective dimensionality of the RKHS solution.

        d_eff = trace(K (K + λI)^{-1})

        Measures how many effective parameters the model uses.
        Ranges from 0 (infinite regularization) to n (no regularization).

        Args:
            K: Gram matrix.
            lam: Regularization parameter.

        Returns:
            Effective dimensionality (float).
        """
        K = np.asarray(K, dtype=np.float64)
        n = K.shape[0]

        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 0.0)  # Ensure non-negative

        d_eff = np.sum(eigenvalues / (eigenvalues + lam))
        return float(d_eff)


# ============================================================================
# UTILITY
# ============================================================================

def _get_kernel_fn(name: str) -> Callable:
    """Look up kernel function by name."""
    kernels = {
        'rbf': rbf_kernel,
        'gaussian': rbf_kernel,
        'epanechnikov': epanechnikov_kernel,
        'tricube': tricube_kernel,
    }
    if name not in kernels:
        raise ValueError(f"Unknown kernel '{name}'. Options: {list(kernels.keys())}")
    return kernels[name]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'KernelRegressionResult',
    'rbf_kernel',
    'epanechnikov_kernel',
    'tricube_kernel',
    'NadarayaWatson',
    'LocalLinearRegression',
    'KernelDensityEstimation',
    'RKHS',
]
