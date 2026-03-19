"""
Discriminant Analysis — ST5.4
==============================

Classification and dimensionality reduction via class separation.
Fisher's LDA finds projections that maximize between-class variance
relative to within-class variance. QDA relaxes the equal covariance
assumption for nonlinear decision boundaries.

Classes:
    LDA: Fisher's Linear Discriminant Analysis
    QDA: Quadratic Discriminant Analysis

Mathematical Foundation:
    LDA criterion: max_w (w^T S_B w) / (w^T S_W w)
    where S_B = between-class scatter, S_W = within-class scatter
    Solution: eigenvectors of S_W^{-1} S_B

    QDA: per-class Gaussians N(mu_k, Sigma_k) with Bayes decision rule
    Decision boundary is quadratic when Sigma_k differ across classes

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LDAResult:
    """
    Result of Linear Discriminant Analysis.

    Attributes:
        weights: Discriminant directions, shape (n_features, n_components)
        eigenvalues: Discriminant eigenvalues (separation ratios)
        class_means: Per-class means, shape (n_classes, n_features)
        overall_mean: Grand mean, shape (n_features,)
        within_scatter: Within-class scatter matrix S_W
        between_scatter: Between-class scatter matrix S_B
        explained_ratio: Proportion of discriminant variance per component
    """
    weights: np.ndarray
    eigenvalues: np.ndarray
    class_means: np.ndarray
    overall_mean: np.ndarray
    within_scatter: np.ndarray
    between_scatter: np.ndarray
    explained_ratio: np.ndarray


class LDA:
    """
    Fisher's Linear Discriminant Analysis.

    Projects data onto directions that maximize class separation:
        w* = argmax_w (w^T S_B w) / (w^T S_W w)

    Equivalent to Bayes-optimal classification when class-conditional
    distributions are Gaussian with equal covariance.

    For K classes, produces at most K-1 discriminant components.
    """

    def __init__(self):
        self._result: LDAResult | None = None
        self._classes: np.ndarray | None = None
        self._priors: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDAResult:
        """
        Fit LDA model.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Class labels, shape (n_samples,)

        Returns:
            LDAResult with discriminant directions and scatter matrices
        """
        n, p = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        self._classes = classes

        # Compute class means and priors
        class_means = np.zeros((n_classes, p))
        class_counts = np.zeros(n_classes)
        for k, c in enumerate(classes):
            mask = y == c
            class_means[k] = X[mask].mean(axis=0)
            class_counts[k] = mask.sum()

        self._priors = class_counts / n
        overall_mean = X.mean(axis=0)

        # Within-class scatter: S_W = sum_k sum_{x in C_k} (x - mu_k)(x - mu_k)^T
        S_W = np.zeros((p, p))
        for k, c in enumerate(classes):
            X_k = X[y == c] - class_means[k]
            S_W += X_k.T @ X_k

        # Between-class scatter: S_B = sum_k n_k (mu_k - mu)(mu_k - mu)^T
        S_B = np.zeros((p, p))
        for k in range(n_classes):
            diff = (class_means[k] - overall_mean).reshape(-1, 1)
            S_B += class_counts[k] * (diff @ diff.T)

        # Solve generalized eigenvalue problem: S_B w = lambda S_W w
        # Equivalent to eigendecomposition of S_W^{-1} S_B
        S_W_reg = S_W + 1e-8 * np.eye(p)  # Regularize
        M = np.linalg.inv(S_W_reg) @ S_B

        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep at most K-1 components
        n_components = min(n_classes - 1, p)
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

        # Explained ratio
        total = eigenvalues.sum()
        explained_ratio = eigenvalues / total if total > 0 else eigenvalues

        self._result = LDAResult(
            weights=eigenvectors,
            eigenvalues=eigenvalues,
            class_means=class_means,
            overall_mean=overall_mean,
            within_scatter=S_W,
            between_scatter=S_B,
            explained_ratio=explained_ratio,
        )
        return self._result

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto discriminant directions.

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            Projected data, shape (n_samples, n_components)
        """
        if self._result is None:
            raise ValueError("Must call fit() before transform()")
        return (X - self._result.overall_mean) @ self._result.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify using minimum Mahalanobis distance in discriminant space.

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            Predicted class labels, shape (n_samples,)
        """
        if self._result is None or self._classes is None:
            raise ValueError("Must call fit() before predict()")

        X_proj = self.transform(X)
        means_proj = (self._result.class_means - self._result.overall_mean) @ self._result.weights

        # Classify by nearest centroid in projected space (+ log prior)
        predictions = np.zeros(X.shape[0], dtype=self._classes.dtype)
        for i in range(X.shape[0]):
            best_k = 0
            best_score = -np.inf
            for k in range(len(self._classes)):
                dist_sq = np.sum((X_proj[i] - means_proj[k]) ** 2)
                score = -0.5 * dist_sq + np.log(self._priors[k] + 1e-10)
                if score > best_score:
                    best_score = score
                    best_k = k
            predictions[i] = self._classes[best_k]

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))


class QDA:
    """
    Quadratic Discriminant Analysis.

    Like LDA but allows each class to have its own covariance matrix,
    producing quadratic (rather than linear) decision boundaries.

    Decision rule:
        argmax_k [ -0.5 log|Sigma_k| - 0.5 (x-mu_k)^T Sigma_k^{-1} (x-mu_k) + log pi_k ]
    """

    def __init__(self, reg_param: float = 1e-4):
        """
        Args:
            reg_param: Regularization added to covariance diagonals
        """
        self.reg_param = reg_param
        self._classes: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._covariances: list[np.ndarray] | None = None
        self._cov_invs: list[np.ndarray] | None = None
        self._log_dets: np.ndarray | None = None
        self._priors: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit QDA model (per-class mean and covariance).

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Class labels, shape (n_samples,)
        """
        n, p = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        self._classes = classes

        self._means = np.zeros((n_classes, p))
        self._covariances = []
        self._cov_invs = []
        self._log_dets = np.zeros(n_classes)
        self._priors = np.zeros(n_classes)

        for k, c in enumerate(classes):
            X_k = X[y == c]
            n_k = len(X_k)
            self._means[k] = X_k.mean(axis=0)
            self._priors[k] = n_k / n

            # Per-class covariance with regularization
            cov_k = np.cov(X_k, rowvar=False)
            if cov_k.ndim == 0:
                cov_k = np.array([[cov_k]])
            cov_k += self.reg_param * np.eye(p)

            self._covariances.append(cov_k)
            self._cov_invs.append(np.linalg.inv(cov_k))

            sign, logdet = np.linalg.slogdet(cov_k)
            self._log_dets[k] = logdet if sign > 0 else 1e10

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify using Bayes rule with per-class Gaussians.

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            Predicted class labels, shape (n_samples,)
        """
        if self._classes is None:
            raise ValueError("Must call fit() before predict()")

        n = X.shape[0]
        n_classes = len(self._classes)
        scores = np.zeros((n, n_classes))

        for k in range(n_classes):
            diff = X - self._means[k]
            mahal = np.sum(diff @ self._cov_invs[k] * diff, axis=1)
            scores[:, k] = (
                -0.5 * self._log_dets[k]
                - 0.5 * mahal
                + np.log(self._priors[k] + 1e-10)
            )

        return self._classes[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior class probabilities via Bayes rule.

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        if self._classes is None:
            raise ValueError("Must call fit() before predict_proba()")

        n = X.shape[0]
        n_classes = len(self._classes)
        log_scores = np.zeros((n, n_classes))

        for k in range(n_classes):
            diff = X - self._means[k]
            mahal = np.sum(diff @ self._cov_invs[k] * diff, axis=1)
            log_scores[:, k] = (
                -0.5 * self._log_dets[k]
                - 0.5 * mahal
                + np.log(self._priors[k] + 1e-10)
            )

        # Normalize via log-sum-exp
        max_scores = log_scores.max(axis=1, keepdims=True)
        log_probs = log_scores - max_scores - np.log(
            np.sum(np.exp(log_scores - max_scores), axis=1, keepdims=True)
        )
        return np.exp(log_probs)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X) == y))
