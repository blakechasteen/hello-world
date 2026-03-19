"""
Factor Analysis — ST5.2
========================

Latent variable model: X = Lambda * F + epsilon
where Lambda is the factor loadings matrix, F are latent factors,
and epsilon is idiosyncratic noise (diagonal covariance).

Differs from PCA: factors explain correlations, not variance.
PCA components are orthogonal directions of maximum variance;
FA factors are latent causes with unique per-variable noise.

Classes:
    FactorAnalysis: EM algorithm for maximum likelihood factor analysis

Mathematical Foundation:
    Model: X ~ N(mu, Lambda Lambda^T + Psi)
    where Psi = diag(psi_1, ..., psi_p) is the uniqueness matrix
    EM: E-step computes E[F|X], M-step updates Lambda and Psi
    Rotation: Varimax maximizes sum of variances of squared loadings

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactorResult:
    """
    Result of factor analysis.

    Attributes:
        loadings: Factor loading matrix Lambda, shape (n_features, n_factors)
        communalities: Proportion of variance explained by factors per variable
        explained_variance: Variance explained by each factor
        uniquenesses: Unique variance per variable (diagonal of Psi)
        n_iter: Number of EM iterations
        log_likelihood: Final log-likelihood
    """
    loadings: np.ndarray
    communalities: np.ndarray
    explained_variance: np.ndarray
    uniquenesses: np.ndarray
    n_iter: int = 0
    log_likelihood: float = 0.0


class FactorAnalysis:
    """
    Factor Analysis via the EM algorithm.

    Fits the model X = Lambda * F + epsilon where:
    - Lambda: p x k loading matrix
    - F ~ N(0, I): k latent factors
    - epsilon ~ N(0, Psi): idiosyncratic noise, Psi diagonal

    The EM algorithm alternates:
    - E-step: compute posterior E[F|X] and E[FF^T|X]
    - M-step: update Lambda and Psi via maximum likelihood
    """

    def fit(
        self,
        X: np.ndarray,
        n_factors: int,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> FactorResult:
        """
        Fit factor model to data.

        Args:
            X: Data matrix, shape (n_samples, n_features)
            n_factors: Number of latent factors k
            max_iter: Maximum EM iterations
            tol: Convergence tolerance on log-likelihood change

        Returns:
            FactorResult with loadings, communalities, etc.
        """
        n, p = X.shape
        k = n_factors

        # Center data
        X_centered = X - X.mean(axis=0)
        S = (X_centered.T @ X_centered) / n  # Sample covariance

        # Initialize Lambda from scaled PCA
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        Lambda = Vt[:k].T * np.sqrt(s[:k]) / np.sqrt(n)

        # Initialize Psi from residual variance
        Psi = np.diag(S) - np.sum(Lambda ** 2, axis=1)
        Psi = np.maximum(Psi, 1e-6)

        prev_ll = -np.inf

        for iteration in range(max_iter):
            Psi_inv = np.diag(1.0 / Psi)

            # E-step
            # Posterior: F|X ~ N(Beta X, I - Beta Lambda)
            # Beta = Lambda^T (Lambda Lambda^T + Psi)^{-1}
            M = Lambda.T @ Psi_inv @ Lambda + np.eye(k)
            M_inv = np.linalg.inv(M)
            Beta = M_inv @ Lambda.T @ Psi_inv

            # Expected sufficient statistics
            EF = Beta @ X_centered.T  # (k, n)
            EFF = n * M_inv + EF @ EF.T  # (k, k)

            # M-step
            # Lambda_new = (X^T E[F]^T) (E[FF^T])^{-1}
            Lambda_new = (X_centered.T @ EF.T) @ np.linalg.inv(EFF)

            # Psi_new = diag(S - Lambda_new E[F] X^T / n)
            Psi_new = np.diag(S) - np.sum(Lambda_new * (X_centered.T @ EF.T / n), axis=1)
            Psi_new = np.maximum(Psi_new, 1e-6)

            # Log-likelihood
            Sigma = Lambda_new @ Lambda_new.T + np.diag(Psi_new)
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                logdet = -1e10
            Sigma_inv = np.linalg.inv(Sigma + 1e-8 * np.eye(p))
            ll = -0.5 * n * (p * np.log(2 * np.pi) + logdet + np.trace(Sigma_inv @ S))

            Lambda = Lambda_new
            Psi = Psi_new

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        # Compute results
        communalities = np.sum(Lambda ** 2, axis=1)
        explained_variance = np.sum(Lambda ** 2, axis=0)

        return FactorResult(
            loadings=Lambda,
            communalities=communalities,
            explained_variance=explained_variance,
            uniquenesses=Psi,
            n_iter=iteration + 1,
            log_likelihood=ll,
        )

    @staticmethod
    def rotate(loadings: np.ndarray, method: str = "varimax", max_iter: int = 100) -> np.ndarray:
        """
        Rotate factor loadings for interpretability.

        Varimax: orthogonal rotation maximizing sum of variances of
        squared loadings per factor. Produces simple structure where
        each variable loads highly on few factors.

        Args:
            loadings: Loading matrix, shape (n_features, n_factors)
            method: Rotation method ("varimax" or "quartimax")
            max_iter: Maximum rotation iterations

        Returns:
            Rotated loading matrix
        """
        p, k = loadings.shape
        Lambda = loadings.copy()
        R = np.eye(k)

        for _ in range(max_iter):
            Lambda_rot = Lambda @ R
            L2 = Lambda_rot ** 2

            if method == "varimax":
                # Maximize column variance of squared loadings
                target = L2 - L2.mean(axis=0, keepdims=True)
            elif method == "quartimax":
                # Maximize row variance of squared loadings
                target = L2 - L2.mean(axis=1, keepdims=True)
            else:
                raise ValueError(f"Unknown rotation method: {method}")

            # SVD-based rotation update
            U, s, Vt = np.linalg.svd(Lambda.T @ (Lambda_rot * target))
            R_new = U @ Vt

            if np.max(np.abs(R_new - R)) < 1e-8:
                R = R_new
                break
            R = R_new

        return Lambda @ R

    @staticmethod
    def scree_plot_data(X: np.ndarray) -> np.ndarray:
        """
        Compute eigenvalues for scree plot (factor selection).

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            Eigenvalues of the correlation matrix, sorted descending
        """
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        corr = (X_std.T @ X_std) / X.shape[0]
        eigenvalues = np.linalg.eigvalsh(corr)[::-1]
        return eigenvalues
