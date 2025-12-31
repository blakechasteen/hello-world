"""
HoloLoom ML Training System - Base Trainer

Abstract base class providing shared functionality for all trainers.
Handles sklearn availability, async execution, and common utilities.

Created: 2025-12-31
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, Optional, Callable

import numpy as np

from HoloLoom.ml.protocol import (
    ArrayLike,
    DataSplit,
    EvaluationResult,
    ModelType,
    PredictionResult,
    TaskType,
    TrainerProtocol,
    TrainingResult,
    generate_model_id,
)

logger = logging.getLogger(__name__)

# Check sklearn availability
SKLEARN_AVAILABLE = False
try:
    import sklearn
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score

    SKLEARN_AVAILABLE = True
    logger.debug(f"sklearn {sklearn.__version__} available")
except ImportError:
    logger.warning(
        "sklearn not available. Install with: pip install scikit-learn>=1.0"
    )


class BaseTrainer(ABC):
    """
    Abstract base class for ML trainers.

    Provides:
    - Async wrapper for sklearn's synchronous operations
    - Model serialization/deserialization
    - Common evaluation metrics
    - sklearn availability checking

    Subclasses must implement:
    - model_type property
    - task_type property
    - _train_sync() method
    - get_hyperparameter_space() method
    """

    # Shared thread pool for sklearn operations
    _executor: Optional[ThreadPoolExecutor] = None

    def __init__(self, n_workers: int = 4):
        """
        Initialize base trainer.

        Args:
            n_workers: Number of worker threads for async execution
        """
        if BaseTrainer._executor is None:
            BaseTrainer._executor = ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix="ml_trainer_",
            )
        self._model_id_prefix = ""

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the model type this trainer handles."""
        ...

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return the task type (regression, classification, etc.)."""
        ...

    @abstractmethod
    def _train_sync(
        self,
        data: DataSplit,
        **kwargs: Any,
    ) -> tuple:
        """
        Synchronous training implementation.

        Args:
            data: Data split for training
            **kwargs: Model-specific parameters

        Returns:
            Tuple of (model, metrics_dict, hyperparameters_used)
        """
        ...

    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """Get hyperparameter search space for tuning."""
        ...

    def _check_sklearn(self) -> None:
        """Check if sklearn is available, raise if not."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is required for ML training. "
                "Install with: pip install scikit-learn>=1.0"
            )

    async def _run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a synchronous function in thread pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        if kwargs:
            func = partial(func, **kwargs)
        return await loop.run_in_executor(self._executor, func, *args)

    def _serialize_model(self, model: Any) -> bytes:
        """
        Serialize model to bytes using pickle.

        Args:
            model: sklearn model object

        Returns:
            Pickled bytes
        """
        return pickle.dumps(model)

    def _deserialize_model(self, model_bytes: bytes) -> Any:
        """
        Deserialize model from bytes.

        Args:
            model_bytes: Pickled bytes

        Returns:
            sklearn model object
        """
        return pickle.loads(model_bytes)

    def _ensure_numpy(self, data: ArrayLike) -> np.ndarray:
        """
        Ensure data is numpy array.

        Args:
            data: Array-like data

        Returns:
            numpy ndarray
        """
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)

    async def train(
        self,
        data: DataSplit,
        **kwargs: Any,
    ) -> TrainingResult:
        """
        Train model on provided data.

        Args:
            data: Data split with training data
            **kwargs: Additional training parameters

        Returns:
            TrainingResult with trained model and metrics
        """
        self._check_sklearn()

        start_time = time.perf_counter()

        # Run training in thread pool
        model, metrics, hyperparams = await self._run_in_executor(
            self._train_sync, data, **kwargs
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Serialize model
        model_bytes = self._serialize_model(model)

        # Generate model ID
        model_id = generate_model_id(self.model_type, self._model_id_prefix)

        # Get feature importance if available
        feature_importance = self._get_feature_importance_from_model(
            model, data.feature_names
        )

        return TrainingResult(
            model_id=model_id,
            model_type=self.model_type,
            model_bytes=model_bytes,
            hyperparameters=hyperparams,
            metrics=metrics,
            feature_names=data.feature_names,
            feature_importance=feature_importance,
            data_hash=data.data_hash,
            training_duration_ms=duration_ms,
        )

    def _get_feature_importance_from_model(
        self,
        model: Any,
        feature_names: Optional[list],
    ) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained sklearn model
            feature_names: Feature names

        Returns:
            Dict of feature name to importance, or None
        """
        # Check for coefficient-based importance (linear models)
        if hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim == 1:
                importance = np.abs(coef)
            else:
                importance = np.abs(coef).mean(axis=0)

            if feature_names and len(feature_names) == len(importance):
                return dict(zip(feature_names, importance.tolist()))
            else:
                return {f"feature_{i}": float(v) for i, v in enumerate(importance)}

        # Check for feature_importances_ (tree models)
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            if feature_names and len(feature_names) == len(importance):
                return dict(zip(feature_names, importance.tolist()))
            else:
                return {f"feature_{i}": float(v) for i, v in enumerate(importance)}

        return None

    async def evaluate(
        self,
        result: TrainingResult,
        data: DataSplit,
        cv_folds: int = 5,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate trained model.

        Args:
            result: Training result containing the model
            data: Data split for evaluation
            cv_folds: Number of CV folds (0 to disable)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with metrics
        """
        self._check_sklearn()

        start_time = time.perf_counter()

        # Deserialize model
        model = self._deserialize_model(result.model_bytes)

        # Use test data if available, otherwise use training data
        X_eval = data.X_test if data.X_test is not None else data.X_train
        y_eval = data.y_test if data.y_test is not None else data.y_train

        # Run evaluation in thread pool
        metrics = await self._run_in_executor(
            self._evaluate_sync,
            model,
            data.X_train,
            data.y_train,
            X_eval,
            y_eval,
            cv_folds,
            data.n_features,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        return EvaluationResult(
            model_id=result.model_id,
            mse=metrics["mse"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            r2_score=metrics["r2_score"],
            adjusted_r2=metrics.get("adjusted_r2"),
            cv_scores=metrics.get("cv_scores"),
            cv_mean=metrics.get("cv_mean"),
            cv_std=metrics.get("cv_std"),
            residuals_stats=metrics.get("residuals_stats"),
            evaluation_duration_ms=duration_ms,
        )

    def _evaluate_sync(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        cv_folds: int,
        n_features: int,
    ) -> Dict[str, Any]:
        """
        Synchronous evaluation implementation.

        Args:
            model: Trained model
            X_train: Training features (for CV)
            y_train: Training targets (for CV)
            X_eval: Evaluation features
            y_eval: Evaluation targets
            cv_folds: Number of CV folds
            n_features: Number of features (for adjusted R²)

        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = model.predict(X_eval)

        # Calculate metrics
        mse = mean_squared_error(y_eval, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
        }

        # Adjusted R²
        n_samples = len(y_eval)
        if n_samples > n_features + 1:
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            metrics["adjusted_r2"] = float(adj_r2)

        # Cross-validation if requested
        if cv_folds > 0:
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="r2",
            )
            metrics["cv_scores"] = cv_scores.tolist()
            metrics["cv_mean"] = float(cv_scores.mean())
            metrics["cv_std"] = float(cv_scores.std())

        # Residual statistics
        residuals = y_eval - y_pred
        metrics["residuals_stats"] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
        }

        return metrics

    async def predict(
        self,
        result: TrainingResult,
        X: ArrayLike,
        **kwargs: Any,
    ) -> PredictionResult:
        """
        Make predictions with trained model.

        Args:
            result: Training result containing the model
            X: Features to predict on
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult with predictions
        """
        self._check_sklearn()

        start_time = time.perf_counter()

        # Deserialize model
        model = self._deserialize_model(result.model_bytes)

        # Ensure numpy array
        X = self._ensure_numpy(X)

        # Run prediction in thread pool
        predictions = await self._run_in_executor(model.predict, X)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResult(
            model_id=result.model_id,
            predictions=predictions,
            prediction_duration_ms=duration_ms,
        )

    def get_feature_importance(
        self,
        result: TrainingResult,
    ) -> Optional[Dict[str, float]]:
        """
        Get feature importance from training result.

        Args:
            result: Training result

        Returns:
            Feature importance dict or None
        """
        return result.feature_importance
