"""
HoloLoom ML - Linear Regression Trainer Tests

Unit tests for LinearRegressionTrainer.

Created: 2025-12-31
"""

from __future__ import annotations

import asyncio
import pickle
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from HoloLoom.ml.config import (
    LinearRegressionConfig,
    TrainingConfig,
    ValidationStrictness,
)
from HoloLoom.ml.protocol import DataSplit, ModelType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def create_linear_data(
    n_samples: int = 100,
    n_features: int = 5,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Create synthetic linear regression data."""
    np.random.seed(seed)

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # True coefficients
    true_coef = np.array([1.5, -2.0, 0.5, 1.0, -0.5])[:n_features]

    # Generate target with noise
    y = X @ true_coef + np.random.randn(n_samples) * noise

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y, feature_names


def create_data_split(
    n_samples: int = 100,
    n_features: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Create a DataSplit for testing."""
    X, y, feature_names = create_linear_data(n_samples, n_features, seed=seed)

    # Split indices
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Test: Trainer Import and Creation
# ---------------------------------------------------------------------------


class TestTrainerCreation:
    """Test trainer instantiation."""

    def test_import_trainer(self):
        """Test that trainer can be imported."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer
        assert LinearRegressionTrainer is not None

    def test_create_trainer_default_config(self):
        """Test creating trainer with default config."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        trainer = LinearRegressionTrainer()
        assert trainer is not None
        assert trainer.model_type == ModelType.LINEAR_REGRESSION

    def test_create_trainer_custom_config(self):
        """Test creating trainer with custom config."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        config = LinearRegressionConfig(
            fit_intercept=False,
            positive=True,
        )
        training_config = TrainingConfig(
            cv_folds=3,
            validation_strictness=ValidationStrictness.HIGH,
        )

        trainer = LinearRegressionTrainer(
            config=config,
            training_config=training_config,
        )

        assert trainer.config.fit_intercept is False
        assert trainer.config.positive is True
        assert trainer.training_config.cv_folds == 3


# ---------------------------------------------------------------------------
# Test: Training
# ---------------------------------------------------------------------------


class TestTraining:
    """Test model training."""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer
        return LinearRegressionTrainer()

    @pytest.fixture
    def data_split(self):
        """Create data split."""
        return create_data_split(n_samples=100, n_features=5)

    @pytest.mark.asyncio
    async def test_train_basic(self, trainer, data_split):
        """Test basic training."""
        result = await trainer.train(data_split)

        assert result is not None
        assert result.model_id is not None
        assert result.model_id.startswith("linreg_")
        assert result.model_bytes is not None
        assert len(result.model_bytes) > 0
        assert result.model_type == ModelType.LINEAR_REGRESSION

    @pytest.mark.asyncio
    async def test_train_returns_hyperparameters(self, trainer, data_split):
        """Test that training returns hyperparameters."""
        result = await trainer.train(data_split)

        assert "fit_intercept" in result.hyperparameters
        assert "n_features" in result.hyperparameters
        assert result.hyperparameters["n_features"] == 5

    @pytest.mark.asyncio
    async def test_train_returns_metrics(self, trainer, data_split):
        """Test that training returns metrics."""
        result = await trainer.train(data_split)

        # Should have training metrics
        assert result.training_metrics is not None
        assert "train_r2" in result.training_metrics

        # R² should be reasonable for clean linear data
        assert result.training_metrics["train_r2"] > 0.9

    @pytest.mark.asyncio
    async def test_train_model_serializable(self, trainer, data_split):
        """Test that trained model can be deserialized."""
        result = await trainer.train(data_split)

        # Try to load the model
        model = pickle.loads(result.model_bytes)
        assert model is not None
        assert hasattr(model, "predict")

    @pytest.mark.asyncio
    async def test_train_with_validation(self, trainer, data_split):
        """Test training includes validation metrics when val data present."""
        result = await trainer.train(data_split)

        # Should have validation metrics
        assert "val_r2" in result.training_metrics
        assert result.training_metrics["val_r2"] > 0.8


# ---------------------------------------------------------------------------
# Test: Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Test model evaluation."""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer
        return LinearRegressionTrainer()

    @pytest.fixture
    def data_split(self):
        """Create data split."""
        return create_data_split(n_samples=100, n_features=5)

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, trainer, data_split):
        """Test basic evaluation."""
        # Train first
        train_result = await trainer.train(data_split)

        # Evaluate
        eval_result = await trainer.evaluate(train_result, data_split)

        assert eval_result is not None
        assert eval_result.mse is not None
        assert eval_result.mse >= 0
        assert eval_result.r2_score is not None

    @pytest.mark.asyncio
    async def test_evaluate_metrics(self, trainer, data_split):
        """Test all evaluation metrics are present."""
        train_result = await trainer.train(data_split)
        eval_result = await trainer.evaluate(train_result, data_split)

        # Check all metrics
        assert eval_result.mse >= 0
        assert eval_result.rmse >= 0
        assert eval_result.mae >= 0
        assert -1.0 <= eval_result.r2_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_rmse_is_sqrt_mse(self, trainer, data_split):
        """Test RMSE = sqrt(MSE)."""
        train_result = await trainer.train(data_split)
        eval_result = await trainer.evaluate(train_result, data_split)

        expected_rmse = np.sqrt(eval_result.mse)
        assert abs(eval_result.rmse - expected_rmse) < 1e-6

    @pytest.mark.asyncio
    async def test_evaluate_with_cv(self, trainer, data_split):
        """Test evaluation with cross-validation."""
        train_result = await trainer.train(data_split)
        eval_result = await trainer.evaluate(
            train_result, data_split, cross_validate=True
        )

        # Should have CV scores
        assert eval_result.cv_scores is not None
        assert len(eval_result.cv_scores) > 0


# ---------------------------------------------------------------------------
# Test: Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    """Test model prediction."""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer
        return LinearRegressionTrainer()

    @pytest.fixture
    def data_split(self):
        """Create data split."""
        return create_data_split(n_samples=100, n_features=5)

    @pytest.mark.asyncio
    async def test_predict_basic(self, trainer, data_split):
        """Test basic prediction."""
        train_result = await trainer.train(data_split)

        # Predict on test data
        pred_result = await trainer.predict(train_result, data_split.X_test)

        assert pred_result is not None
        assert pred_result.predictions is not None
        assert len(pred_result.predictions) == len(data_split.X_test)

    @pytest.mark.asyncio
    async def test_predict_single_sample(self, trainer, data_split):
        """Test prediction on single sample."""
        train_result = await trainer.train(data_split)

        # Single sample
        X_single = data_split.X_test[0:1]
        pred_result = await trainer.predict(train_result, X_single)

        assert len(pred_result.predictions) == 1

    @pytest.mark.asyncio
    async def test_predictions_reasonable(self, trainer, data_split):
        """Test that predictions are reasonable (close to actual)."""
        train_result = await trainer.train(data_split)
        pred_result = await trainer.predict(train_result, data_split.X_test)

        # Calculate correlation between predictions and actual
        correlation = np.corrcoef(
            pred_result.predictions, data_split.y_test
        )[0, 1]

        # For clean linear data, correlation should be high
        assert correlation > 0.9


# ---------------------------------------------------------------------------
# Test: Feature Importance
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    """Test feature importance extraction."""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer
        return LinearRegressionTrainer()

    @pytest.fixture
    def data_split(self):
        """Create data split."""
        return create_data_split(n_samples=100, n_features=5)

    @pytest.mark.asyncio
    async def test_get_feature_importance(self, trainer, data_split):
        """Test getting feature importance."""
        train_result = await trainer.train(data_split)

        importance = await trainer.get_feature_importance(
            train_result, data_split.feature_names
        )

        assert importance is not None
        assert len(importance) == len(data_split.feature_names)

    @pytest.mark.asyncio
    async def test_feature_importance_dict_format(self, trainer, data_split):
        """Test feature importance returns dict with feature names."""
        train_result = await trainer.train(data_split)

        importance = await trainer.get_feature_importance(
            train_result, data_split.feature_names
        )

        for name in data_split.feature_names:
            assert name in importance
            assert isinstance(importance[name], float)


# ---------------------------------------------------------------------------
# Test: Configuration Variants
# ---------------------------------------------------------------------------


class TestConfigurationVariants:
    """Test different configuration options."""

    @pytest.fixture
    def data_split(self):
        """Create data split."""
        return create_data_split(n_samples=100, n_features=5)

    @pytest.mark.asyncio
    async def test_no_intercept(self, data_split):
        """Test training without intercept."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        config = LinearRegressionConfig(fit_intercept=False)
        trainer = LinearRegressionTrainer(config=config)

        result = await trainer.train(data_split)

        assert result is not None
        assert result.hyperparameters["fit_intercept"] is False

    @pytest.mark.asyncio
    async def test_positive_coefficients(self, data_split):
        """Test training with positive coefficients constraint."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        # Create data with all positive coefficients
        np.random.seed(42)
        X = np.abs(np.random.randn(100, 5))  # All positive features
        y = X @ np.array([1.0, 0.5, 0.3, 0.2, 0.1]) + np.random.randn(100) * 0.1

        data = DataSplit(
            X_train=X[:70],
            y_train=y[:70],
            X_test=X[70:],
            y_test=y[70:],
            feature_names=[f"f{i}" for i in range(5)],
        )

        config = LinearRegressionConfig(positive=True)
        trainer = LinearRegressionTrainer(config=config)

        result = await trainer.train(data)

        # Load model and check coefficients
        model = pickle.loads(result.model_bytes)
        assert all(coef >= -1e-10 for coef in model.coef_)


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer
        return LinearRegressionTrainer()

    @pytest.mark.asyncio
    async def test_single_feature(self, trainer):
        """Test with single feature."""
        X = np.random.randn(50, 1)
        y = 2.0 * X.ravel() + 1.0 + np.random.randn(50) * 0.1

        data = DataSplit(
            X_train=X[:40],
            y_train=y[:40],
            X_test=X[40:],
            y_test=y[40:],
            feature_names=["x"],
        )

        result = await trainer.train(data)
        assert result is not None

    @pytest.mark.asyncio
    async def test_many_features(self, trainer):
        """Test with many features."""
        X = np.random.randn(100, 50)
        y = X @ np.random.randn(50) + np.random.randn(100) * 0.1

        data = DataSplit(
            X_train=X[:80],
            y_train=y[:80],
            X_test=X[80:],
            y_test=y[80:],
            feature_names=[f"f{i}" for i in range(50)],
        )

        result = await trainer.train(data)
        assert result is not None

    @pytest.mark.asyncio
    async def test_small_dataset(self, trainer):
        """Test with very small dataset."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        data = DataSplit(
            X_train=X[:4],
            y_train=y[:4],
            X_test=X[4:],
            y_test=y[4:],
            feature_names=["x"],
        )

        result = await trainer.train(data)

        # Should perfectly fit y = 2x
        pred = await trainer.predict(result, data.X_test)
        assert abs(pred.predictions[0] - 10.0) < 0.1


# ---------------------------------------------------------------------------
# Test: Protocol Compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Test that trainer complies with TrainerProtocol."""

    def test_has_model_type(self):
        """Test trainer has model_type property."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        trainer = LinearRegressionTrainer()
        assert hasattr(trainer, "model_type")
        assert trainer.model_type == ModelType.LINEAR_REGRESSION

    def test_has_train_method(self):
        """Test trainer has train method."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        trainer = LinearRegressionTrainer()
        assert hasattr(trainer, "train")
        assert asyncio.iscoroutinefunction(trainer.train)

    def test_has_evaluate_method(self):
        """Test trainer has evaluate method."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        trainer = LinearRegressionTrainer()
        assert hasattr(trainer, "evaluate")
        assert asyncio.iscoroutinefunction(trainer.evaluate)

    def test_has_predict_method(self):
        """Test trainer has predict method."""
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        trainer = LinearRegressionTrainer()
        assert hasattr(trainer, "predict")
        assert asyncio.iscoroutinefunction(trainer.predict)


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
