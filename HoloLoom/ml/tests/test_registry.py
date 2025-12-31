"""
HoloLoom ML - Model Registry Tests

Tests for model persistence and versioning.

Created: 2025-12-31
"""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from HoloLoom.ml.config import RegistryConfig
from HoloLoom.ml.protocol import (
    DataSplit,
    EvaluationResult,
    ModelStatus,
    ModelType,
    TrainingResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_registry_path():
    """Create temporary registry path."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_training_result():
    """Create sample training result."""
    # Create a simple model (just coefficients)
    model = {"coef": [1.0, 2.0, 3.0], "intercept": 0.5}
    model_bytes = pickle.dumps(model)

    return TrainingResult(
        model_id="linreg_test_001",
        model_type=ModelType.LINEAR_REGRESSION,
        model_bytes=model_bytes,
        hyperparameters={
            "fit_intercept": True,
            "n_features": 3,
        },
        training_metrics={
            "train_r2": 0.95,
            "val_r2": 0.92,
        },
        provenance={
            "data_hash": "abc123",
            "created_at": datetime.now().isoformat(),
        },
    )


@pytest.fixture
def sample_evaluation_result():
    """Create sample evaluation result."""
    return EvaluationResult(
        mse=0.01,
        rmse=0.1,
        mae=0.08,
        r2_score=0.95,
        cv_scores=[0.94, 0.95, 0.96, 0.94, 0.95],
    )


# ---------------------------------------------------------------------------
# Test: Registry Import and Creation
# ---------------------------------------------------------------------------


class TestRegistryCreation:
    """Test registry instantiation."""

    def test_import_registry(self):
        """Test registry can be imported."""
        from HoloLoom.ml.registry import ModelRegistry
        assert ModelRegistry is not None

    def test_create_registry_default_path(self, temp_registry_path):
        """Test creating registry with default config."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        assert registry is not None
        assert registry.storage_path == temp_registry_path

    def test_registry_creates_directories(self, temp_registry_path):
        """Test registry creates storage directories."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        # Should create base directory
        assert temp_registry_path.exists()


# ---------------------------------------------------------------------------
# Test: Saving Models
# ---------------------------------------------------------------------------


class TestSaveModel:
    """Test model saving functionality."""

    @pytest.mark.asyncio
    async def test_save_model_basic(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test basic model saving."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        model_id = await registry.save(
            sample_training_result, sample_evaluation_result
        )

        assert model_id is not None
        assert model_id == sample_training_result.model_id

    @pytest.mark.asyncio
    async def test_save_creates_files(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test that save creates necessary files."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        model_id = await registry.save(
            sample_training_result, sample_evaluation_result
        )

        # Check model directory exists
        model_dir = temp_registry_path / "linear_regression" / model_id
        assert model_dir.exists()

        # Check files exist
        assert (model_dir / "model.pkl.gz").exists() or (model_dir / "model.pkl").exists()
        assert (model_dir / "training.json").exists()
        assert (model_dir / "evaluation.json").exists()

    @pytest.mark.asyncio
    async def test_save_updates_index(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test that save updates the index file."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        model_id = await registry.save(
            sample_training_result, sample_evaluation_result
        )

        # Check index exists and contains model
        index_path = temp_registry_path / "index.json"
        assert index_path.exists()

        with open(index_path) as f:
            index = json.load(f)

        assert model_id in index["models"]

    @pytest.mark.asyncio
    async def test_save_multiple_models(self, temp_registry_path):
        """Test saving multiple models."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        # Save multiple models
        model_ids = []
        for i in range(3):
            model = {"coef": [float(i)]}
            result = TrainingResult(
                model_id=f"linreg_test_{i:03d}",
                model_type=ModelType.LINEAR_REGRESSION,
                model_bytes=pickle.dumps(model),
                hyperparameters={"n": i},
                training_metrics={"r2": 0.9 + i * 0.01},
            )
            eval_result = EvaluationResult(
                mse=0.1 - i * 0.01, rmse=0.1, mae=0.1, r2_score=0.9 + i * 0.01
            )

            model_id = await registry.save(result, eval_result)
            model_ids.append(model_id)

        # All should be in index
        models = await registry.list_models()
        assert len(models) == 3


# ---------------------------------------------------------------------------
# Test: Loading Models
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Test model loading functionality."""

    @pytest.mark.asyncio
    async def test_load_model_basic(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test basic model loading."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        # Save first
        model_id = await registry.save(
            sample_training_result, sample_evaluation_result
        )

        # Load
        loaded = await registry.load(model_id)

        assert loaded is not None
        assert loaded.model_id == model_id
        assert loaded.model_bytes is not None

    @pytest.mark.asyncio
    async def test_load_model_bytes_match(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test that loaded model bytes match original."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        model_id = await registry.save(
            sample_training_result, sample_evaluation_result
        )
        loaded = await registry.load(model_id)

        # Deserialize both
        original_model = pickle.loads(sample_training_result.model_bytes)
        loaded_model = pickle.loads(loaded.model_bytes)

        assert original_model == loaded_model

    @pytest.mark.asyncio
    async def test_load_nonexistent_model(self, temp_registry_path):
        """Test loading nonexistent model returns None."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        loaded = await registry.load("nonexistent_model_123")

        assert loaded is None


# ---------------------------------------------------------------------------
# Test: Listing Models
# ---------------------------------------------------------------------------


class TestListModels:
    """Test model listing functionality."""

    @pytest.mark.asyncio
    async def test_list_empty_registry(self, temp_registry_path):
        """Test listing empty registry."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        models = await registry.list_models()

        assert models is not None
        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_list_models_basic(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test basic model listing."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        await registry.save(sample_training_result, sample_evaluation_result)

        models = await registry.list_models()

        assert len(models) == 1
        assert models[0]["model_id"] == sample_training_result.model_id

    @pytest.mark.asyncio
    async def test_list_models_by_type(self, temp_registry_path):
        """Test listing models filtered by type."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        # Save a linear regression model
        result = TrainingResult(
            model_id="linreg_001",
            model_type=ModelType.LINEAR_REGRESSION,
            model_bytes=pickle.dumps({"coef": [1.0]}),
            hyperparameters={},
            training_metrics={"r2": 0.9},
        )
        eval_result = EvaluationResult(mse=0.1, rmse=0.1, mae=0.1, r2_score=0.9)
        await registry.save(result, eval_result)

        # Filter by type
        models = await registry.list_models(model_type=ModelType.LINEAR_REGRESSION)
        assert len(models) == 1

        # Filter by different type
        models = await registry.list_models(model_type=ModelType.RIDGE)
        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_list_models_by_status(self, temp_registry_path):
        """Test listing models filtered by status."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        result = TrainingResult(
            model_id="linreg_001",
            model_type=ModelType.LINEAR_REGRESSION,
            model_bytes=pickle.dumps({"coef": [1.0]}),
            hyperparameters={},
            training_metrics={"r2": 0.9},
            status=ModelStatus.TRAINED,
        )
        eval_result = EvaluationResult(mse=0.1, rmse=0.1, mae=0.1, r2_score=0.9)
        await registry.save(result, eval_result)

        # Filter by trained status
        models = await registry.list_models(status=ModelStatus.TRAINED)
        assert len(models) == 1

    @pytest.mark.asyncio
    async def test_list_models_by_min_r2(self, temp_registry_path):
        """Test listing models filtered by minimum R² score."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        # Save two models with different R² scores
        for i, r2 in enumerate([0.7, 0.95]):
            result = TrainingResult(
                model_id=f"linreg_{i:03d}",
                model_type=ModelType.LINEAR_REGRESSION,
                model_bytes=pickle.dumps({"coef": [float(i)]}),
                hyperparameters={},
                training_metrics={"r2": r2},
            )
            eval_result = EvaluationResult(mse=0.1, rmse=0.1, mae=0.1, r2_score=r2)
            await registry.save(result, eval_result)

        # Filter by min R²
        models = await registry.list_models(min_r2=0.9)
        assert len(models) == 1
        assert models[0]["r2_score"] >= 0.9


# ---------------------------------------------------------------------------
# Test: Deleting Models (Archive)
# ---------------------------------------------------------------------------


class TestDeleteModel:
    """Test model deletion (archival) functionality."""

    @pytest.mark.asyncio
    async def test_delete_model_archives(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test that delete archives model rather than hard delete."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        model_id = await registry.save(
            sample_training_result, sample_evaluation_result
        )

        # Delete (archive)
        success = await registry.delete(model_id)

        assert success

        # Model should be archived, not in main list
        models = await registry.list_models()
        assert len(models) == 0

        # Archive directory should exist
        archive_dir = temp_registry_path / "archived"
        assert archive_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_model(self, temp_registry_path):
        """Test deleting nonexistent model returns False."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        success = await registry.delete("nonexistent_model")

        assert not success


# ---------------------------------------------------------------------------
# Test: Get Best Model
# ---------------------------------------------------------------------------


class TestGetBestModel:
    """Test getting best model by metric."""

    @pytest.mark.asyncio
    async def test_get_best_model_by_r2(self, temp_registry_path):
        """Test getting best model by R² score."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        # Save multiple models
        r2_scores = [0.7, 0.95, 0.85]
        for i, r2 in enumerate(r2_scores):
            result = TrainingResult(
                model_id=f"linreg_{i:03d}",
                model_type=ModelType.LINEAR_REGRESSION,
                model_bytes=pickle.dumps({"coef": [float(i)]}),
                hyperparameters={},
                training_metrics={"r2": r2},
            )
            eval_result = EvaluationResult(mse=0.1, rmse=0.1, mae=0.1, r2_score=r2)
            await registry.save(result, eval_result)

        # Get best by R²
        best = await registry.get_best_model(
            model_type=ModelType.LINEAR_REGRESSION, metric="r2_score"
        )

        assert best is not None
        assert best["r2_score"] == 0.95

    @pytest.mark.asyncio
    async def test_get_best_model_empty_registry(self, temp_registry_path):
        """Test getting best model from empty registry."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        best = await registry.get_best_model(
            model_type=ModelType.LINEAR_REGRESSION, metric="r2_score"
        )

        assert best is None


# ---------------------------------------------------------------------------
# Test: Model Describe
# ---------------------------------------------------------------------------


class TestDescribeModel:
    """Test model description functionality."""

    @pytest.mark.asyncio
    async def test_describe_model(
        self, temp_registry_path, sample_training_result, sample_evaluation_result
    ):
        """Test getting model description."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        model_id = await registry.save(
            sample_training_result, sample_evaluation_result
        )

        description = await registry.describe(model_id)

        assert description is not None
        assert description["model_id"] == model_id
        assert "hyperparameters" in description
        assert "training_metrics" in description
        assert "evaluation" in description

    @pytest.mark.asyncio
    async def test_describe_nonexistent_model(self, temp_registry_path):
        """Test describing nonexistent model."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(storage_path=temp_registry_path)
        registry = ModelRegistry(config=config)

        description = await registry.describe("nonexistent")

        assert description is None


# ---------------------------------------------------------------------------
# Test: Compression
# ---------------------------------------------------------------------------


class TestCompression:
    """Test model compression functionality."""

    @pytest.mark.asyncio
    async def test_gzip_compression(self, temp_registry_path):
        """Test that models are gzip compressed."""
        from HoloLoom.ml.registry import ModelRegistry

        config = RegistryConfig(
            storage_path=temp_registry_path,
            compression=True,
        )
        registry = ModelRegistry(config=config)

        # Create a larger model
        model = {"coef": list(range(1000)), "data": np.random.randn(1000).tolist()}
        result = TrainingResult(
            model_id="linreg_large",
            model_type=ModelType.LINEAR_REGRESSION,
            model_bytes=pickle.dumps(model),
            hyperparameters={},
            training_metrics={"r2": 0.9},
        )
        eval_result = EvaluationResult(mse=0.1, rmse=0.1, mae=0.1, r2_score=0.9)

        model_id = await registry.save(result, eval_result)

        # Check compressed file exists
        model_dir = temp_registry_path / "linear_regression" / model_id
        compressed_path = model_dir / "model.pkl.gz"

        # Either compressed or uncompressed should exist
        assert compressed_path.exists() or (model_dir / "model.pkl").exists()

        # If compressed, verify it's smaller than uncompressed
        if compressed_path.exists():
            compressed_size = compressed_path.stat().st_size
            uncompressed_size = len(result.model_bytes)
            # Compressed should be smaller for large models
            assert compressed_size < uncompressed_size


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
