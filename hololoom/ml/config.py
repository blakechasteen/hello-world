"""
HoloLoom ML Training System - Configuration

Configuration dataclasses for ML training, data handling, and system settings.
Provides presets for common use cases: default, fast, research.

Created: 2025-12-31
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NormalizationMethod(Enum):
    """Data normalization methods."""

    NONE = "none"
    STANDARD = "standard"  # z-score (mean=0, std=1)
    MINMAX = "minmax"  # scale to [0, 1]
    ROBUST = "robust"  # median and IQR


class MissingValueStrategy(Enum):
    """Strategies for handling missing values."""

    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    INTERPOLATE = "interpolate"


class ValidationStrictness(Enum):
    """Data validation strictness levels."""

    LOW = "low"  # Allow most issues
    MEDIUM = "medium"  # Block critical issues only
    HIGH = "high"  # Strict validation


@dataclass
class DataConfig:
    """
    Configuration for data handling.

    Attributes:
        test_size: Fraction of data for test set (0.0-1.0)
        val_size: Fraction of training data for validation (0.0-1.0)
        random_state: Random seed for reproducibility
        normalization: Normalization method to apply
        missing_value_strategy: How to handle missing values
        missing_fill_value: Value to use if strategy is CONSTANT
        categorical_encoding: How to encode categoricals ('onehot', 'label', 'target')
        max_categories: Max categories before grouping into 'other'
        shuffle: Whether to shuffle data before splitting
    """

    test_size: float = 0.2
    val_size: float = 0.0  # 0 means no separate validation set
    random_state: int | None = 42
    normalization: NormalizationMethod = NormalizationMethod.STANDARD
    missing_value_strategy: MissingValueStrategy = MissingValueStrategy.MEAN
    missing_fill_value: float | None = None
    categorical_encoding: str = "onehot"
    max_categories: int = 20
    shuffle: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "normalization": self.normalization.value,
            "missing_value_strategy": self.missing_value_strategy.value,
            "missing_fill_value": self.missing_fill_value,
            "categorical_encoding": self.categorical_encoding,
            "max_categories": self.max_categories,
            "shuffle": self.shuffle,
        }


@dataclass
class LinearRegressionConfig:
    """
    Configuration for linear regression training.

    Mirrors sklearn.linear_model.LinearRegression parameters.

    Attributes:
        fit_intercept: Whether to calculate intercept
        copy_X: Whether to copy X or overwrite
        n_jobs: Number of jobs for computation (-1 for all CPUs)
        positive: Force coefficients to be positive
    """

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: int | None = None
    positive: bool = False

    def to_sklearn_params(self) -> dict[str, Any]:
        """Convert to sklearn parameters."""
        return {
            "fit_intercept": self.fit_intercept,
            "copy_X": self.copy_X,
            "n_jobs": self.n_jobs,
            "positive": self.positive,
        }


@dataclass
class RidgeConfig:
    """
    Configuration for Ridge regression.

    Attributes:
        alpha: Regularization strength
        fit_intercept: Whether to calculate intercept
        solver: Solver to use ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga')
        max_iter: Maximum iterations for iterative solvers
        tol: Precision of solution
        random_state: Random seed
    """

    alpha: float = 1.0
    fit_intercept: bool = True
    solver: str = "auto"
    max_iter: int | None = None
    tol: float = 1e-4
    random_state: int | None = None

    def to_sklearn_params(self) -> dict[str, Any]:
        """Convert to sklearn parameters."""
        params = {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "solver": self.solver,
            "tol": self.tol,
        }
        if self.max_iter is not None:
            params["max_iter"] = self.max_iter
        if self.random_state is not None:
            params["random_state"] = self.random_state
        return params


@dataclass
class LassoConfig:
    """
    Configuration for Lasso regression.

    Attributes:
        alpha: Regularization strength
        fit_intercept: Whether to calculate intercept
        max_iter: Maximum iterations
        tol: Tolerance for optimization
        warm_start: Whether to use previous solution as initialization
        positive: Force coefficients to be positive
        selection: 'cyclic' or 'random' feature selection
        random_state: Random seed
    """

    alpha: float = 1.0
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    selection: str = "cyclic"
    random_state: int | None = None

    def to_sklearn_params(self) -> dict[str, Any]:
        """Convert to sklearn parameters."""
        params = {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "warm_start": self.warm_start,
            "positive": self.positive,
            "selection": self.selection,
        }
        if self.random_state is not None:
            params["random_state"] = self.random_state
        return params


@dataclass
class TrainingConfig:
    """
    Configuration for training process.

    Attributes:
        cv_folds: Number of cross-validation folds (0 to disable)
        cv_shuffle: Whether to shuffle before CV
        early_stopping: Enable early stopping (if supported)
        early_stopping_rounds: Rounds without improvement to stop
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        enable_hyperparameter_tuning: Enable automatic tuning
        tuning_iterations: Number of tuning iterations
        tuning_cv_folds: CV folds for tuning
        n_jobs: Number of parallel jobs
    """

    cv_folds: int = 5
    cv_shuffle: bool = True
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    verbose: int = 0
    enable_hyperparameter_tuning: bool = False
    tuning_iterations: int = 20
    tuning_cv_folds: int = 3
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cv_folds": self.cv_folds,
            "cv_shuffle": self.cv_shuffle,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose": self.verbose,
            "enable_hyperparameter_tuning": self.enable_hyperparameter_tuning,
            "tuning_iterations": self.tuning_iterations,
            "tuning_cv_folds": self.tuning_cv_folds,
            "n_jobs": self.n_jobs,
        }


@dataclass
class RegistryConfig:
    """
    Configuration for model registry.

    Attributes:
        storage_path: Base path for model storage
        compression: Whether to gzip model files
        compression_level: Gzip compression level (1-9)
        max_versions: Maximum versions to keep per model type
        index_file: Name of index file
        enable_archiving: Whether to archive instead of delete
    """

    storage_path: Path = field(default_factory=lambda: Path(".hololoom/ml/models"))
    compression: bool = True
    compression_level: int = 6
    max_versions: int = 100
    index_file: str = "index.json"
    enable_archiving: bool = True

    def __post_init__(self):
        """Ensure storage_path is a Path object."""
        if isinstance(self.storage_path, str):
            self.storage_path = Path(self.storage_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_path": str(self.storage_path),
            "compression": self.compression,
            "compression_level": self.compression_level,
            "max_versions": self.max_versions,
            "index_file": self.index_file,
            "enable_archiving": self.enable_archiving,
        }


@dataclass
class ValidationConfig:
    """
    Configuration for data validation.

    Attributes:
        strictness: Validation strictness level
        min_samples: Minimum samples required
        max_missing_ratio: Maximum allowed missing value ratio (0.0-1.0)
        outlier_threshold: Z-score threshold for outlier detection
        check_multicollinearity: Whether to check for multicollinearity
        multicollinearity_threshold: Correlation threshold
        check_constant_features: Whether to check for constant features
        check_target_distribution: Whether to check target distribution
    """

    strictness: ValidationStrictness = ValidationStrictness.MEDIUM
    min_samples: int = 10
    max_missing_ratio: float = 0.1
    outlier_threshold: float = 3.0
    check_multicollinearity: bool = True
    multicollinearity_threshold: float = 0.95
    check_constant_features: bool = True
    check_target_distribution: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strictness": self.strictness.value,
            "min_samples": self.min_samples,
            "max_missing_ratio": self.max_missing_ratio,
            "outlier_threshold": self.outlier_threshold,
            "check_multicollinearity": self.check_multicollinearity,
            "multicollinearity_threshold": self.multicollinearity_threshold,
            "check_constant_features": self.check_constant_features,
            "check_target_distribution": self.check_target_distribution,
        }


@dataclass
class MLSystemConfig:
    """
    Top-level configuration for the ML training system.

    Combines all sub-configurations with preset factories.

    Attributes:
        data: Data handling configuration
        training: Training process configuration
        linear_regression: Linear regression specific config
        ridge: Ridge regression specific config
        lasso: Lasso regression specific config
        registry: Model registry configuration
        validation: Data validation configuration
        enable_sklearn: Whether sklearn is enabled
        enable_pandas: Whether pandas is enabled
        log_level: Logging level
    """

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    linear_regression: LinearRegressionConfig = field(
        default_factory=LinearRegressionConfig
    )
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    lasso: LassoConfig = field(default_factory=LassoConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    enable_sklearn: bool = True
    enable_pandas: bool = True
    log_level: str = "INFO"

    @classmethod
    def default(cls) -> MLSystemConfig:
        """
        Create default configuration.

        Balanced settings for general use.
        """
        return cls()

    @classmethod
    def fast(cls) -> MLSystemConfig:
        """
        Create fast configuration.

        Optimized for speed: no CV, no tuning, minimal validation.
        """
        return cls(
            data=DataConfig(
                test_size=0.2,
                val_size=0.0,
                normalization=NormalizationMethod.NONE,
            ),
            training=TrainingConfig(
                cv_folds=0,
                enable_hyperparameter_tuning=False,
                verbose=0,
            ),
            validation=ValidationConfig(
                strictness=ValidationStrictness.LOW,
                check_multicollinearity=False,
            ),
        )

    @classmethod
    def research(cls) -> MLSystemConfig:
        """
        Create research configuration.

        Maximum validation, CV, and logging for research use.
        """
        return cls(
            data=DataConfig(
                test_size=0.2,
                val_size=0.1,
                random_state=42,
                normalization=NormalizationMethod.STANDARD,
            ),
            training=TrainingConfig(
                cv_folds=10,
                cv_shuffle=True,
                enable_hyperparameter_tuning=True,
                tuning_iterations=50,
                tuning_cv_folds=5,
                verbose=2,
            ),
            validation=ValidationConfig(
                strictness=ValidationStrictness.HIGH,
                check_multicollinearity=True,
                check_target_distribution=True,
            ),
            log_level="DEBUG",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "linear_regression": self.linear_regression.to_sklearn_params(),
            "ridge": self.ridge.to_sklearn_params(),
            "lasso": self.lasso.to_sklearn_params(),
            "registry": self.registry.to_dict(),
            "validation": self.validation.to_dict(),
            "enable_sklearn": self.enable_sklearn,
            "enable_pandas": self.enable_pandas,
            "log_level": self.log_level,
        }
