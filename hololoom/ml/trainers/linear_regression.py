"""
HoloLoom ML Training System - Linear Regression Trainer

sklearn-based linear regression trainer with async interface.
Supports hyperparameter tuning and feature importance extraction.

Created: 2025-12-31
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hololoom.ml.config import LinearRegressionConfig
from hololoom.ml.protocol import (
    DataSplit,
    ModelType,
    TaskType,
)
from hololoom.ml.trainers.base_trainer import BaseTrainer, SKLEARN_AVAILABLE

if SKLEARN_AVAILABLE:
    from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class LinearRegressionTrainer(BaseTrainer):
    """
    Linear Regression trainer using sklearn.

    Implements ordinary least squares linear regression with
    optional intercept fitting and positive coefficient constraints.

    Usage:
        trainer = LinearRegressionTrainer()
        result = await trainer.train(data_split)
        evaluation = await trainer.evaluate(result, data_split)
        predictions = await trainer.predict(result, X_new)
    """

    def __init__(
        self,
        config: Optional[LinearRegressionConfig] = None,
        n_workers: int = 4,
    ):
        """
        Initialize linear regression trainer.

        Args:
            config: Linear regression configuration (uses defaults if None)
            n_workers: Number of worker threads for async execution
        """
        super().__init__(n_workers=n_workers)
        self.config = config or LinearRegressionConfig()
        self._model_id_prefix = ""

    @property
    def model_type(self) -> ModelType:
        """Return LINEAR_REGRESSION model type."""
        return ModelType.LINEAR_REGRESSION

    @property
    def task_type(self) -> TaskType:
        """Return REGRESSION task type."""
        return TaskType.REGRESSION

    def _train_sync(
        self,
        data: DataSplit,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, float], Dict[str, Any]]:
        """
        Synchronous training implementation.

        Args:
            data: Data split for training
            **kwargs: Override config parameters

        Returns:
            Tuple of (model, metrics_dict, hyperparameters_used)
        """
        # Merge config with kwargs
        params = self.config.to_sklearn_params()
        params.update(kwargs)

        logger.debug(f"Training LinearRegression with params: {params}")

        # Create and train model
        model = LinearRegression(**params)
        model.fit(data.X_train, data.y_train)

        # Calculate training metrics
        y_pred_train = model.predict(data.X_train)
        mse_train = float(np.mean((data.y_train - y_pred_train) ** 2))
        r2_train = float(
            1 - (np.sum((data.y_train - y_pred_train) ** 2) /
                 np.sum((data.y_train - np.mean(data.y_train)) ** 2))
        )

        metrics = {
            "mse_train": mse_train,
            "r2_train": r2_train,
            "n_features": data.n_features,
            "n_samples": data.n_train_samples,
        }

        # Add intercept if fit
        if params.get("fit_intercept", True):
            metrics["intercept"] = float(model.intercept_)

        # Log training completion
        logger.info(
            f"LinearRegression trained: R²={r2_train:.4f}, "
            f"MSE={mse_train:.4f}, features={data.n_features}"
        )

        return model, metrics, params

    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Get hyperparameter search space for tuning.

        Linear regression has few hyperparameters.
        For regularization, consider Ridge/Lasso instead.

        Returns:
            Dictionary defining hyperparameter options
        """
        return {
            "fit_intercept": {
                "type": "bool",
                "default": True,
                "description": "Whether to calculate the intercept",
            },
            "positive": {
                "type": "bool",
                "default": False,
                "description": "Force coefficients to be positive",
            },
            "n_jobs": {
                "type": "int",
                "default": None,
                "range": [-1, None, 1, 2, 4, 8],
                "description": "Number of jobs for computation",
            },
        }

    def get_coefficients(
        self,
        model_bytes: bytes,
        feature_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Get model coefficients.

        Args:
            model_bytes: Serialized model
            feature_names: Optional feature names

        Returns:
            Dictionary mapping feature names to coefficients
        """
        model = self._deserialize_model(model_bytes)

        coef = model.coef_
        if coef.ndim > 1:
            coef = coef.flatten()

        if feature_names and len(feature_names) == len(coef):
            return dict(zip(feature_names, coef.tolist()))
        else:
            return {f"feature_{i}": float(v) for i, v in enumerate(coef)}

    def get_intercept(self, model_bytes: bytes) -> Optional[float]:
        """
        Get model intercept.

        Args:
            model_bytes: Serialized model

        Returns:
            Intercept value or None if not fit
        """
        model = self._deserialize_model(model_bytes)
        if hasattr(model, "intercept_"):
            intercept = model.intercept_
            if isinstance(intercept, np.ndarray):
                return float(intercept[0]) if len(intercept) > 0 else None
            return float(intercept)
        return None

    def get_equation(
        self,
        model_bytes: bytes,
        feature_names: Optional[list] = None,
        precision: int = 4,
    ) -> str:
        """
        Get model equation as string.

        Args:
            model_bytes: Serialized model
            feature_names: Optional feature names
            precision: Decimal precision for coefficients

        Returns:
            Equation string like "y = 2.5*x1 + -1.3*x2 + 0.5"
        """
        model = self._deserialize_model(model_bytes)

        coef = model.coef_
        if coef.ndim > 1:
            coef = coef.flatten()

        # Get feature names
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(coef))]

        # Build equation parts
        parts = []
        for name, c in zip(feature_names, coef):
            if abs(c) > 1e-10:  # Skip near-zero coefficients
                sign = "+" if c >= 0 else ""
                parts.append(f"{sign}{c:.{precision}f}*{name}")

        equation = "y = " + " ".join(parts)

        # Add intercept
        if hasattr(model, "intercept_"):
            intercept = model.intercept_
            if isinstance(intercept, np.ndarray):
                intercept = intercept[0] if len(intercept) > 0 else 0
            if abs(intercept) > 1e-10:
                sign = "+" if intercept >= 0 else ""
                equation += f" {sign}{intercept:.{precision}f}"

        return equation

    def summary(
        self,
        model_bytes: bytes,
        feature_names: Optional[list] = None,
    ) -> str:
        """
        Get human-readable model summary.

        Args:
            model_bytes: Serialized model
            feature_names: Optional feature names

        Returns:
            Multi-line summary string
        """
        model = self._deserialize_model(model_bytes)

        lines = [
            "Linear Regression Model Summary",
            "=" * 40,
        ]

        # Equation
        lines.append(f"\nEquation: {self.get_equation(model_bytes, feature_names)}")

        # Coefficients
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef.flatten()

        lines.append(f"\nCoefficients ({len(coef)} features):")

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(coef))]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        for i in sorted_idx[:10]:  # Top 10
            lines.append(f"  {feature_names[i]}: {coef[i]:.6f}")
        if len(coef) > 10:
            lines.append(f"  ... and {len(coef) - 10} more")

        # Intercept
        if hasattr(model, "intercept_"):
            intercept = model.intercept_
            if isinstance(intercept, np.ndarray):
                intercept = intercept[0] if len(intercept) > 0 else 0
            lines.append(f"\nIntercept: {intercept:.6f}")

        return "\n".join(lines)
