"""
HoloLoom ML - Agentic Skill Tests

Tests for MLTrainerSkill agentic interface.

Created: 2025-12-31
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from HoloLoom.ml.config import RegistryConfig
from HoloLoom.ml.protocol import (
    DataSplit,
    EvaluationResult,
    ModelType,
    TrainingResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_storage_path():
    """Create temporary storage path for skill."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with test data."""
    tmpdir = tempfile.mkdtemp()
    tmppath = Path(tmpdir)

    # Create a simple CSV file
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.0, 2.0, -1.0, 0.5, -0.5]) + np.random.randn(100) * 0.1

    csv_path = tmppath / "test_data.csv"
    with open(csv_path, "w") as f:
        f.write("feature_0,feature_1,feature_2,feature_3,feature_4,target\n")
        for i in range(100):
            row = ",".join(str(v) for v in list(X[i]) + [y[i]])
            f.write(row + "\n")

    yield tmppath

    shutil.rmtree(tmpdir, ignore_errors=True)


def create_skill(storage_path: Path):
    """Create MLTrainerSkill with custom storage path."""
    from HoloLoom.ml.skills import MLTrainerSkill

    return MLTrainerSkill(registry_path=storage_path)


# ---------------------------------------------------------------------------
# Test: Skill Import and Creation
# ---------------------------------------------------------------------------


class TestSkillCreation:
    """Test skill instantiation."""

    def test_import_skill(self):
        """Test skill can be imported."""
        from HoloLoom.ml.skills import MLTrainerSkill
        assert MLTrainerSkill is not None

    def test_import_operation(self):
        """Test MLOperation enum can be imported."""
        from HoloLoom.ml.skills import MLOperation
        assert MLOperation is not None
        assert MLOperation.TRAIN.value == "train"

    def test_create_skill(self, temp_storage_path):
        """Test creating skill instance."""
        skill = create_skill(temp_storage_path)
        assert skill is not None


# ---------------------------------------------------------------------------
# Test: Natural Language Parsing
# ---------------------------------------------------------------------------


class TestNaturalLanguageParsing:
    """Test natural language command parsing."""

    def test_parse_train_command(self, temp_storage_path):
        """Test parsing train command."""
        skill = create_skill(temp_storage_path)

        text = "Train a linear regression model on data.csv to predict price"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "train"
        assert result["data_source"] == "data.csv"
        assert result["target_column"] == "price"

    def test_parse_train_with_model_type(self, temp_storage_path):
        """Test parsing train command with model type."""
        skill = create_skill(temp_storage_path)

        text = "Train a ridge regression model on sales.csv to predict revenue"
        result = skill._parse_natural_language(text)

        assert result["operation"] == "train"
        assert result["model_type"] == "ridge"

    def test_parse_evaluate_command(self, temp_storage_path):
        """Test parsing evaluate command."""
        skill = create_skill(temp_storage_path)

        text = "Evaluate model linreg_001"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "evaluate"
        assert result["model_id"] == "linreg_001"

    def test_parse_predict_command(self, temp_storage_path):
        """Test parsing predict command."""
        skill = create_skill(temp_storage_path)

        text = "Use model linreg_001 to predict on new_data.csv"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "predict"
        assert result["model_id"] == "linreg_001"
        assert result["data_source"] == "new_data.csv"

    def test_parse_list_command(self, temp_storage_path):
        """Test parsing list command."""
        skill = create_skill(temp_storage_path)

        text = "List all trained models"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "list_models"

    def test_parse_describe_command(self, temp_storage_path):
        """Test parsing describe command."""
        skill = create_skill(temp_storage_path)

        text = "Describe model linreg_001"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "describe_model"
        assert result["model_id"] == "linreg_001"

    def test_parse_delete_command(self, temp_storage_path):
        """Test parsing delete command."""
        skill = create_skill(temp_storage_path)

        text = "Delete model linreg_001"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "delete_model"
        assert result["model_id"] == "linreg_001"

    def test_parse_validate_command(self, temp_storage_path):
        """Test parsing validate command."""
        skill = create_skill(temp_storage_path)

        text = "Validate data in sales.csv targeting revenue"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "validate_data"
        assert result["data_source"] == "sales.csv"

    def test_parse_compare_command(self, temp_storage_path):
        """Test parsing compare command."""
        skill = create_skill(temp_storage_path)

        text = "Compare models linreg_001 and linreg_002"
        result = skill._parse_natural_language(text)

        assert result is not None
        assert result["operation"] == "compare_models"
        assert "linreg_001" in result["model_ids"]
        assert "linreg_002" in result["model_ids"]


# ---------------------------------------------------------------------------
# Test: Execute Operations
# ---------------------------------------------------------------------------


class TestExecuteOperations:
    """Test skill execution."""

    @pytest.mark.asyncio
    async def test_execute_train(self, temp_storage_path, temp_data_dir):
        """Test executing train operation."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        # Create SkillInput
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        input_data = SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
                "model_type": "linear_regression",
            },
        )

        result = await skill.execute(input_data)

        assert result.success
        assert "model_id" in result.data
        assert result.data["model_id"].startswith("linreg_")

    @pytest.mark.asyncio
    async def test_execute_list_empty(self, temp_storage_path):
        """Test executing list on empty registry."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        input_data = SkillInput(
            operation=MLOperation.LIST_MODELS,
            parameters={},
        )

        result = await skill.execute(input_data)

        assert result.success
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_execute_train_and_list(self, temp_storage_path, temp_data_dir):
        """Test training then listing models."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        # Train
        train_input = SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )
        train_result = await skill.execute(train_input)
        assert train_result.success

        # List
        list_input = SkillInput(
            operation=MLOperation.LIST_MODELS,
            parameters={},
        )
        list_result = await skill.execute(list_input)

        assert list_result.success
        assert list_result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_execute_train_and_evaluate(self, temp_storage_path, temp_data_dir):
        """Test training then evaluating a model."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        # Train
        train_input = SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )
        train_result = await skill.execute(train_input)
        model_id = train_result.data["model_id"]

        # Evaluate
        eval_input = SkillInput(
            operation=MLOperation.EVALUATE,
            parameters={
                "model_id": model_id,
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )
        eval_result = await skill.execute(eval_input)

        assert eval_result.success
        assert "r2_score" in eval_result.data

    @pytest.mark.asyncio
    async def test_execute_train_and_describe(self, temp_storage_path, temp_data_dir):
        """Test training then describing a model."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        # Train
        train_input = SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )
        train_result = await skill.execute(train_input)
        model_id = train_result.data["model_id"]

        # Describe
        describe_input = SkillInput(
            operation=MLOperation.DESCRIBE_MODEL,
            parameters={"model_id": model_id},
        )
        describe_result = await skill.execute(describe_input)

        assert describe_result.success
        assert "hyperparameters" in describe_result.data
        assert "training_metrics" in describe_result.data

    @pytest.mark.asyncio
    async def test_execute_validate_data(self, temp_storage_path, temp_data_dir):
        """Test validating data."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        input_data = SkillInput(
            operation=MLOperation.VALIDATE_DATA,
            parameters={
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )

        result = await skill.execute(input_data)

        assert result.success
        assert "is_valid" in result.data
        assert result.data["is_valid"] is True

    @pytest.mark.asyncio
    async def test_execute_delete_model(self, temp_storage_path, temp_data_dir):
        """Test deleting a model."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        # Train first
        train_input = SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )
        train_result = await skill.execute(train_input)
        model_id = train_result.data["model_id"]

        # Delete
        delete_input = SkillInput(
            operation=MLOperation.DELETE_MODEL,
            parameters={"model_id": model_id},
        )
        delete_result = await skill.execute(delete_input)

        assert delete_result.success

        # Verify deleted
        list_input = SkillInput(
            operation=MLOperation.LIST_MODELS,
            parameters={},
        )
        list_result = await skill.execute(list_input)
        assert list_result.data["count"] == 0


# ---------------------------------------------------------------------------
# Test: Parse and Execute
# ---------------------------------------------------------------------------


class TestParseAndExecute:
    """Test combined parsing and execution."""

    @pytest.mark.asyncio
    async def test_parse_and_execute_train(self, temp_storage_path, temp_data_dir):
        """Test natural language train command."""
        from HoloLoom.ml.skills import MLTrainerSkill

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        csv_path = temp_data_dir / "test_data.csv"
        command = f"Train a linear regression model on {csv_path} to predict target"

        result = await skill.parse_and_execute(command)

        assert result.success
        assert "model_id" in result.data

    @pytest.mark.asyncio
    async def test_parse_and_execute_list(self, temp_storage_path):
        """Test natural language list command."""
        from HoloLoom.ml.skills import MLTrainerSkill

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        result = await skill.parse_and_execute("List all models")

        assert result.success
        assert "count" in result.data

    @pytest.mark.asyncio
    async def test_parse_and_execute_invalid(self, temp_storage_path):
        """Test invalid command returns helpful message."""
        from HoloLoom.ml.skills import MLTrainerSkill

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        result = await skill.parse_and_execute("do something random")

        assert not result.success
        assert len(result.suggestions) > 0


# ---------------------------------------------------------------------------
# Test: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in skill."""

    @pytest.mark.asyncio
    async def test_train_nonexistent_file(self, temp_storage_path):
        """Test training with nonexistent file."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        input_data = SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": "/nonexistent/path/data.csv",
                "target_column": "target",
            },
        )

        result = await skill.execute(input_data)

        assert not result.success
        assert "error" in result.message.lower() or "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_evaluate_nonexistent_model(self, temp_storage_path, temp_data_dir):
        """Test evaluating nonexistent model."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        input_data = SkillInput(
            operation=MLOperation.EVALUATE,
            parameters={
                "model_id": "nonexistent_model",
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )

        result = await skill.execute(input_data)

        assert not result.success

    @pytest.mark.asyncio
    async def test_describe_nonexistent_model(self, temp_storage_path):
        """Test describing nonexistent model."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        input_data = SkillInput(
            operation=MLOperation.DESCRIBE_MODEL,
            parameters={"model_id": "nonexistent_model"},
        )

        result = await skill.execute(input_data)

        assert not result.success


# ---------------------------------------------------------------------------
# Test: Suggestions
# ---------------------------------------------------------------------------


class TestSuggestions:
    """Test skill provides helpful suggestions."""

    @pytest.mark.asyncio
    async def test_train_success_suggestions(self, temp_storage_path, temp_data_dir):
        """Test that successful train provides next step suggestions."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        input_data = SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": str(temp_data_dir / "test_data.csv"),
                "target_column": "target",
            },
        )

        result = await skill.execute(input_data)

        assert result.success
        assert len(result.suggestions) > 0
        # Should suggest evaluation as next step
        assert any("evaluate" in s.lower() for s in result.suggestions)

    @pytest.mark.asyncio
    async def test_list_empty_suggestions(self, temp_storage_path):
        """Test that empty list provides training suggestion."""
        from HoloLoom.ml.skills import MLTrainerSkill, MLOperation
        from HoloLoom.ml.skills.ml_trainer_skill import SkillInput

        skill = MLTrainerSkill(registry_path=temp_storage_path)

        input_data = SkillInput(
            operation=MLOperation.LIST_MODELS,
            parameters={},
        )

        result = await skill.execute(input_data)

        assert result.success
        assert len(result.suggestions) > 0
        # Should suggest training when no models
        assert any("train" in s.lower() for s in result.suggestions)


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
