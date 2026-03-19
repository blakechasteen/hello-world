"""
HoloLoom ML Training System - Core Protocols and Types

This module defines the foundational types and protocols for the ML training system.
All trainers implement TrainerProtocol for consistent interface across model types.

Created: 2025-12-31
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np

# Type aliases for clarity
ArrayLike = Union[np.ndarray, list[list[float]], list[float]]
FeatureNames = list[str]


class ModelType(Enum):
    """Supported model types for training."""

    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    # Classification (future)
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"


class TaskType(Enum):
    """ML task types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class ModelStatus(Enum):
    """Model lifecycle status."""

    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class DataSplit:
    """
    Data split for training, validation, and testing.

    Attributes:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        X_test: Test features (optional)
        y_test: Test targets (optional)
        feature_names: Names of feature columns
        target_name: Name of target column
        data_hash: Hash for provenance tracking
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    feature_names: FeatureNames | None = None
    target_name: str | None = None
    data_hash: str | None = None

    def __post_init__(self):
        """Compute data hash if not provided."""
        if self.data_hash is None:
            self.data_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of training data for provenance."""
        hasher = hashlib.sha256()
        hasher.update(self.X_train.tobytes())
        hasher.update(self.y_train.tobytes())
        return hasher.hexdigest()[:16]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1

    @property
    def n_train_samples(self) -> int:
        """Number of training samples."""
        return self.X_train.shape[0]

    @property
    def n_val_samples(self) -> int | None:
        """Number of validation samples."""
        return self.X_val.shape[0] if self.X_val is not None else None

    @property
    def n_test_samples(self) -> int | None:
        """Number of test samples."""
        return self.X_test.shape[0] if self.X_test is not None else None

    def summary(self) -> dict[str, Any]:
        """Get summary of data split."""
        return {
            "n_features": self.n_features,
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
            "n_test_samples": self.n_test_samples,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "data_hash": self.data_hash,
        }


@dataclass
class TrainingResult:
    """
    Result of model training.

    Attributes:
        model_id: Unique identifier for the trained model
        model_type: Type of model trained
        model_bytes: Serialized model (pickle)
        hyperparameters: Hyperparameters used
        metrics: Training metrics (loss, iterations, etc.)
        feature_names: Names of features used
        feature_importance: Feature importance scores (if available)
        data_hash: Hash of training data for provenance
        training_duration_ms: Training time in milliseconds
        timestamp: When training completed
        status: Current model status
        metadata: Additional metadata
    """

    model_id: str
    model_type: ModelType
    model_bytes: bytes
    hyperparameters: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    feature_names: FeatureNames | None = None
    feature_importance: dict[str, float] | None = None
    data_hash: str | None = None
    training_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    status: ModelStatus = ModelStatus.TRAINED
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without model_bytes)."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "data_hash": self.data_hash,
            "training_duration_ms": self.training_duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """
    Result of model evaluation.

    Attributes:
        model_id: ID of evaluated model
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2_score: R-squared score
        adjusted_r2: Adjusted R-squared (if n_features available)
        cv_scores: Cross-validation scores (if CV performed)
        cv_mean: Mean of CV scores
        cv_std: Standard deviation of CV scores
        residuals_stats: Statistics about residuals
        evaluation_duration_ms: Evaluation time in milliseconds
        timestamp: When evaluation completed
        metadata: Additional evaluation metadata
    """

    model_id: str
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    adjusted_r2: float | None = None
    cv_scores: list[float] | None = None
    cv_mean: float | None = None
    cv_std: float | None = None
    residuals_stats: dict[str, float] | None = None
    evaluation_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2_score": self.r2_score,
            "adjusted_r2": self.adjusted_r2,
            "cv_scores": self.cv_scores,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "residuals_stats": self.residuals_stats,
            "evaluation_duration_ms": self.evaluation_duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Model: {self.model_id}",
            f"R² Score: {self.r2_score:.4f}",
            f"RMSE: {self.rmse:.4f}",
            f"MAE: {self.mae:.4f}",
        ]
        if self.cv_mean is not None:
            lines.append(f"CV Score: {self.cv_mean:.4f} ± {self.cv_std:.4f}")
        return "\n".join(lines)


@dataclass
class PredictionResult:
    """
    Result of model prediction.

    Attributes:
        model_id: ID of model used for prediction
        predictions: Predicted values
        confidence_intervals: Optional confidence intervals (lower, upper)
        prediction_duration_ms: Prediction time in milliseconds
        n_samples: Number of samples predicted
        timestamp: When prediction was made
        metadata: Additional prediction metadata
    """

    model_id: str
    predictions: np.ndarray
    confidence_intervals: tuple[np.ndarray, np.ndarray] | None = None
    prediction_duration_ms: float = 0.0
    n_samples: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set n_samples from predictions."""
        if self.n_samples == 0 and self.predictions is not None:
            self.n_samples = len(self.predictions)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "model_id": self.model_id,
            "predictions": self.predictions.tolist(),
            "prediction_duration_ms": self.prediction_duration_ms,
            "n_samples": self.n_samples,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if self.confidence_intervals is not None:
            lower, upper = self.confidence_intervals
            result["confidence_intervals"] = {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            }
        return result


@dataclass
class DataValidationIssue:
    """
    A data quality issue found during validation.

    Attributes:
        issue_type: Type of issue (missing_values, outliers, etc.)
        severity: Severity level (warning, error, critical)
        column: Affected column (if applicable)
        message: Human-readable description
        details: Additional details
        blocking: Whether this blocks training
    """

    issue_type: str
    severity: str
    column: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "column": self.column,
            "message": self.message,
            "details": self.details,
            "blocking": self.blocking,
        }


@dataclass
class DataValidationResult:
    """
    Result of data validation.

    Attributes:
        is_valid: Whether data passed validation
        issues: List of issues found
        stats: Data statistics
        recommendations: Suggested fixes
    """

    is_valid: bool
    issues: list[DataValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[DataValidationIssue]:
        """Get issues that block training."""
        return [i for i in self.issues if i.blocking]

    @property
    def warnings(self) -> list[DataValidationIssue]:
        """Get non-blocking warning issues."""
        return [i for i in self.issues if not i.blocking]

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.is_valid:
            return f"✓ Data valid ({len(self.warnings)} warnings)"
        else:
            return f"✗ Data invalid ({len(self.blocking_issues)} blocking issues)"


@runtime_checkable
class TrainerProtocol(Protocol):
    """
    Protocol for all ML trainers.

    All trainers must implement these async methods for HoloLoom integration.
    Uses run_in_executor internally for sklearn's synchronous operations.
    """

    @property
    def model_type(self) -> ModelType:
        """Return the model type this trainer handles."""
        ...

    @property
    def task_type(self) -> TaskType:
        """Return the task type (regression, classification, etc.)."""
        ...

    async def train(
        self,
        data: DataSplit,
        **kwargs: Any,
    ) -> TrainingResult:
        """
        Train model on provided data.

        Args:
            data: Data split with training (and optionally validation) data
            **kwargs: Additional training parameters

        Returns:
            TrainingResult with trained model and metrics
        """
        ...

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
            data: Data split for evaluation (uses X_test/y_test if available)
            cv_folds: Number of cross-validation folds
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with metrics
        """
        ...

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
        ...

    def get_feature_importance(
        self,
        result: TrainingResult,
    ) -> dict[str, float] | None:
        """
        Get feature importance from trained model.

        Args:
            result: Training result containing the model

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        ...

    def get_hyperparameter_space(self) -> dict[str, Any]:
        """
        Get hyperparameter search space for tuning.

        Returns:
            Dictionary defining hyperparameter ranges
        """
        ...


# Convenience type for model bytes (used in registry)
ModelBytes = bytes


def generate_model_id(model_type: ModelType, prefix: str = "") -> str:
    """
    Generate unique model ID.

    Args:
        model_type: Type of model
        prefix: Optional prefix

    Returns:
        Unique model ID like "linreg_a1b2c3d4"
    """
    import uuid

    type_prefix = {
        ModelType.LINEAR_REGRESSION: "linreg",
        ModelType.RIDGE: "ridge",
        ModelType.LASSO: "lasso",
        ModelType.ELASTIC_NET: "enet",
        ModelType.DECISION_TREE: "dtree",
        ModelType.RANDOM_FOREST: "rf",
        ModelType.GRADIENT_BOOSTING: "gbm",
        ModelType.LOGISTIC_REGRESSION: "logreg",
        ModelType.SVM: "svm",
    }.get(model_type, "model")

    short_uuid = uuid.uuid4().hex[:8]

    if prefix:
        return f"{prefix}_{type_prefix}_{short_uuid}"
    return f"{type_prefix}_{short_uuid}"
