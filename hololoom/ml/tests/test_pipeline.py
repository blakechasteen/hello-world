"""
HoloLoom ML - Data Pipeline Tests

Integration tests for data ingestion, validation, and training pipeline.

Created: 2025-12-31
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from HoloLoom.ml.config import (
    ValidationConfig,
    ValidationStrictness,
)
from HoloLoom.ml.protocol import DataSplit


# ---------------------------------------------------------------------------
# Test Data Generation
# ---------------------------------------------------------------------------


def create_csv_file(
    path: Path,
    n_samples: int = 100,
    n_features: int = 5,
    include_missing: bool = False,
    include_outliers: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a CSV file with test data."""
    np.random.seed(seed)

    # Generate data
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + np.random.randn(n_samples) * 0.1

    # Add missing values
    if include_missing:
        mask = np.random.random((n_samples, n_features)) < 0.05
        X = X.astype(float)
        X[mask] = np.nan

    # Add outliers
    if include_outliers:
        outlier_idx = np.random.choice(n_samples, size=5, replace=False)
        X[outlier_idx, 0] = X[outlier_idx, 0] * 10

    # Create column names
    columns = [f"feature_{i}" for i in range(n_features)] + ["target"]

    # Write CSV
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for i in range(n_samples):
            row = list(X[i]) + [y[i]]
            row_str = ",".join(
                str(v) if not np.isnan(v) else "" for v in row
            )
            f.write(row_str + "\n")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "columns": columns,
        "target": "target",
        "features": [f"feature_{i}" for i in range(n_features)],
    }


# ---------------------------------------------------------------------------
# Test: SpinningWheel Adapter
# ---------------------------------------------------------------------------


class TestSpinningWheelAdapter:
    """Test SpinningWheel ML adapter."""

    def test_import_adapter(self):
        """Test adapter can be imported."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter
        assert SpinningWheelMLAdapter is not None

    def test_create_adapter(self):
        """Test creating adapter."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter

        adapter = SpinningWheelMLAdapter()
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_ingest_csv(self):
        """Test ingesting CSV file."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter

        adapter = SpinningWheelMLAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            metadata = create_csv_file(csv_path, n_samples=50, n_features=3)

            result = await adapter.ingest_tabular(
                str(csv_path),
                target_column="target",
            )

            assert result is not None
            X, y, feature_names = result

            assert X.shape == (50, 3)
            assert len(y) == 50
            assert len(feature_names) == 3

    @pytest.mark.asyncio
    async def test_ingest_with_feature_selection(self):
        """Test ingesting with specific feature columns."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter

        adapter = SpinningWheelMLAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            metadata = create_csv_file(csv_path, n_samples=50, n_features=5)

            result = await adapter.ingest_tabular(
                str(csv_path),
                target_column="target",
                feature_columns=["feature_0", "feature_2"],
            )

            X, y, feature_names = result

            assert X.shape == (50, 2)
            assert feature_names == ["feature_0", "feature_2"]

    @pytest.mark.asyncio
    async def test_ingest_auto_detect_features(self):
        """Test automatic feature detection (all non-target columns)."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter

        adapter = SpinningWheelMLAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            metadata = create_csv_file(csv_path, n_samples=50, n_features=4)

            result = await adapter.ingest_tabular(
                str(csv_path),
                target_column="target",
            )

            X, y, feature_names = result

            # Should use all columns except target
            assert X.shape[1] == 4

    @pytest.mark.asyncio
    async def test_create_data_split(self):
        """Test creating data split from ingested data."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter

        adapter = SpinningWheelMLAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            create_csv_file(csv_path, n_samples=100, n_features=3)

            X, y, feature_names = await adapter.ingest_tabular(
                str(csv_path),
                target_column="target",
            )

            data_split = await adapter.create_data_split(
                X, y, feature_names,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            assert isinstance(data_split, DataSplit)
            assert len(data_split.X_train) == 70
            assert len(data_split.X_val) == 15
            assert len(data_split.X_test) == 15


# ---------------------------------------------------------------------------
# Test: DataPig Adapter
# ---------------------------------------------------------------------------


class TestDataPigAdapter:
    """Test DataPig ML adapter for data validation."""

    def test_import_adapter(self):
        """Test adapter can be imported."""
        from HoloLoom.ml.integration import DataPigMLAdapter
        assert DataPigMLAdapter is not None

    def test_create_adapter(self):
        """Test creating adapter."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()
        assert adapter is not None

    def test_create_adapter_with_config(self):
        """Test creating adapter with custom config."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        config = ValidationConfig(
            strictness=ValidationStrictness.HIGH,
            min_samples=50,
        )
        adapter = DataPigMLAdapter(config=config)
        assert adapter.config.strictness == ValidationStrictness.HIGH

    def test_validate_clean_data(self):
        """Test validating clean data."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        # Create clean data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5) + np.random.randn(100) * 0.1

        result = adapter.validate_training_data(X, y)

        assert result.is_valid
        assert len(result.issues) == 0 or all(
            not i.blocking for i in result.issues
        )

    def test_detect_missing_values(self):
        """Test detecting missing values."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        # Create data with missing values
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        X[1, 1] = np.nan
        y = np.random.randn(100)

        result = adapter.validate_training_data(X, y)

        # Should detect missing values
        missing_issues = [
            i for i in result.issues if i.issue_type == "missing_values"
        ]
        assert len(missing_issues) > 0

    def test_detect_infinite_values(self):
        """Test detecting infinite values."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        # Create data with infinite values
        X = np.random.randn(100, 5)
        X[0, 0] = np.inf
        X[1, 1] = -np.inf
        y = np.random.randn(100)

        result = adapter.validate_training_data(X, y)

        # Should detect infinite values
        inf_issues = [
            i for i in result.issues if i.issue_type == "infinite_values"
        ]
        assert len(inf_issues) > 0
        assert any(i.blocking for i in inf_issues)

    def test_detect_constant_features(self):
        """Test detecting constant features."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        # Create data with constant feature
        X = np.random.randn(100, 5)
        X[:, 2] = 1.0  # Constant feature
        y = np.random.randn(100)

        result = adapter.validate_training_data(X, y)

        # Should detect constant feature
        const_issues = [
            i for i in result.issues if i.issue_type == "constant_feature"
        ]
        assert len(const_issues) > 0

    def test_detect_constant_target(self):
        """Test detecting constant target."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        X = np.random.randn(100, 5)
        y = np.ones(100)  # Constant target

        result = adapter.validate_training_data(X, y)

        # Should detect constant target (blocking error)
        const_issues = [
            i for i in result.issues if i.issue_type == "constant_target"
        ]
        assert len(const_issues) > 0
        assert any(i.blocking for i in const_issues)
        assert not result.is_valid

    def test_detect_insufficient_samples(self):
        """Test detecting insufficient samples."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        config = ValidationConfig(min_samples=100)
        adapter = DataPigMLAdapter(config=config)

        X = np.random.randn(50, 5)  # Only 50 samples
        y = np.random.randn(50)

        result = adapter.validate_training_data(X, y)

        # Should detect insufficient samples
        sample_issues = [
            i for i in result.issues if i.issue_type == "insufficient_samples"
        ]
        assert len(sample_issues) > 0

    def test_detect_outliers(self):
        """Test detecting outliers."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        # Create data with outliers
        X = np.random.randn(100, 5)
        X[0, 0] = 100  # Extreme outlier
        y = np.random.randn(100)

        result = adapter.validate_training_data(X, y)

        # Should detect outliers
        outlier_issues = [
            i for i in result.issues if i.issue_type == "outliers"
        ]
        assert len(outlier_issues) > 0

    def test_detect_multicollinearity(self):
        """Test detecting multicollinearity."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        config = ValidationConfig(
            check_multicollinearity=True,
            multicollinearity_threshold=0.9,
        )
        adapter = DataPigMLAdapter(config=config)

        # Create highly correlated features
        X = np.random.randn(100, 5)
        X[:, 1] = X[:, 0] * 0.99 + np.random.randn(100) * 0.01  # ~0.99 correlation
        y = np.random.randn(100)

        result = adapter.validate_training_data(X, y)

        # Should detect multicollinearity
        corr_issues = [
            i for i in result.issues if i.issue_type == "multicollinearity"
        ]
        assert len(corr_issues) > 0

    def test_generate_recommendations(self):
        """Test generating recommendations."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        # Create data with issues
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan  # Missing
        X[1, 1] = 100  # Outlier
        y = np.random.randn(100)

        result = adapter.validate_training_data(X, y)

        # Should have recommendations
        assert len(result.recommendations) > 0

    def test_validate_data_split(self):
        """Test validating a DataSplit object."""
        from HoloLoom.ml.integration import DataPigMLAdapter

        adapter = DataPigMLAdapter()

        # Create DataSplit
        X = np.random.randn(100, 5)
        y = X @ np.random.randn(5)

        data_split = DataSplit(
            X_train=X[:70],
            y_train=y[:70],
            X_test=X[70:],
            y_test=y[70:],
            feature_names=[f"f{i}" for i in range(5)],
        )

        result = adapter.validate_data_split(data_split)

        assert result is not None
        assert result.is_valid


# ---------------------------------------------------------------------------
# Test: End-to-End Pipeline
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Test complete data pipeline from ingestion to training."""

    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """Test complete pipeline: ingest → validate → train → evaluate."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter, DataPigMLAdapter
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV file
            csv_path = Path(tmpdir) / "data.csv"
            create_csv_file(csv_path, n_samples=100, n_features=5)

            # 1. Ingest data
            ingest_adapter = SpinningWheelMLAdapter()
            X, y, feature_names = await ingest_adapter.ingest_tabular(
                str(csv_path),
                target_column="target",
            )

            # 2. Validate data
            validate_adapter = DataPigMLAdapter()
            validation = validate_adapter.validate_training_data(X, y, feature_names)
            assert validation.is_valid, f"Validation failed: {validation.issues}"

            # 3. Create data split
            data_split = await ingest_adapter.create_data_split(
                X, y, feature_names,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            # 4. Train model
            trainer = LinearRegressionTrainer()
            train_result = await trainer.train(data_split)
            assert train_result is not None
            assert train_result.model_bytes is not None

            # 5. Evaluate model
            eval_result = await trainer.evaluate(train_result, data_split)
            assert eval_result is not None
            assert eval_result.r2_score > 0.8  # Good fit for clean linear data

            # 6. Make predictions
            pred_result = await trainer.predict(train_result, data_split.X_test)
            assert len(pred_result.predictions) == len(data_split.X_test)

    @pytest.mark.asyncio
    async def test_pipeline_with_validation_issues(self):
        """Test pipeline handles validation issues gracefully."""
        from HoloLoom.ml.integration import SpinningWheelMLAdapter, DataPigMLAdapter
        from HoloLoom.ml.trainers import LinearRegressionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with missing values
            csv_path = Path(tmpdir) / "data.csv"
            create_csv_file(
                csv_path, n_samples=100, n_features=5, include_missing=True
            )

            # 1. Ingest data
            ingest_adapter = SpinningWheelMLAdapter()
            X, y, feature_names = await ingest_adapter.ingest_tabular(
                str(csv_path),
                target_column="target",
            )

            # 2. Validate data - should find issues
            validate_adapter = DataPigMLAdapter()
            validation = validate_adapter.validate_training_data(X, y, feature_names)

            # Should have missing value issues
            assert any(i.issue_type == "missing_values" for i in validation.issues)

            # Should have recommendations
            assert len(validation.recommendations) > 0


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
