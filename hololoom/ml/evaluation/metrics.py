"""
HoloLoom ML Training System - Evaluation Metrics

Comprehensive regression metrics with statistical utilities.
Supports both numpy-only and sklearn-enhanced calculations.

Created: 2025-12-31
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Check sklearn availability for advanced metrics
SKLEARN_AVAILABLE = False
try:
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    pass


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int | None = None,
) -> dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_features: Number of features (for adjusted R²)

    Returns:
        Dictionary of metric name to value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    n_samples = len(y_true)

    if SKLEARN_AVAILABLE:
        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        evs = float(explained_variance_score(y_true, y_pred))

        # MAPE can fail if y_true has zeros
        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            mape = None
    else:
        # Numpy-only fallback
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        evs = None
        mape = None

    rmse = float(np.sqrt(mse))

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
    }

    if evs is not None:
        metrics["explained_variance"] = evs

    if mape is not None:
        metrics["mape"] = mape

    # Adjusted R² if n_features provided
    if n_features is not None and n_samples > n_features + 1:
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        metrics["adjusted_r2"] = float(adj_r2)

    return metrics


def calculate_residual_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate residual statistics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of residual statistics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    residuals = y_true - y_pred

    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "q1": float(np.percentile(residuals, 25)),
        "q3": float(np.percentile(residuals, 75)),
        "iqr": float(np.percentile(residuals, 75) - np.percentile(residuals, 25)),
        "skewness": float(_calculate_skewness(residuals)),
        "kurtosis": float(_calculate_kurtosis(residuals)),
    }


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness (numpy-only)."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 3))


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis (numpy-only)."""
    n = len(data)
    if n < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 4) - 3)


def calculate_confidence_intervals(
    y_pred: np.ndarray,
    residuals_std: float,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions.

    Uses normal distribution approximation.

    Args:
        y_pred: Predicted values
        residuals_std: Standard deviation of residuals
        confidence_level: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    from scipy.stats import norm
    z = norm.ppf((1 + confidence_level) / 2)
    margin = z * residuals_std

    return (y_pred - margin, y_pred + margin)


@dataclass
class MetricsCalculator:
    """
    Stateful metrics calculator for incremental evaluation.

    Useful for large datasets that need batch processing.
    """

    n_samples: int = 0
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    sum_y_true: float = 0.0
    sum_y_true_squared: float = 0.0
    sum_y_pred: float = 0.0

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Update with a batch of predictions.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        self.n_samples += len(y_true)
        self.sum_squared_error += np.sum((y_true - y_pred) ** 2)
        self.sum_absolute_error += np.sum(np.abs(y_true - y_pred))
        self.sum_y_true += np.sum(y_true)
        self.sum_y_true_squared += np.sum(y_true ** 2)
        self.sum_y_pred += np.sum(y_pred)

    def compute(self) -> dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary of metrics
        """
        if self.n_samples == 0:
            return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2_score": 0.0}

        mse = self.sum_squared_error / self.n_samples
        mae = self.sum_absolute_error / self.n_samples
        rmse = np.sqrt(mse)

        # R² calculation
        mean_y = self.sum_y_true / self.n_samples
        ss_tot = self.sum_y_true_squared - self.n_samples * mean_y ** 2
        r2 = 1 - self.sum_squared_error / ss_tot if ss_tot > 0 else 0.0

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "n_samples": self.n_samples,
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self.n_samples = 0
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.sum_y_true = 0.0
        self.sum_y_true_squared = 0.0
        self.sum_y_pred = 0.0
