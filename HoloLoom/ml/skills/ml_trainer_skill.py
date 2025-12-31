"""
HoloLoom ML Training System - Agentic Skill

Skill-based interface for ML training operations.
Enables natural language interaction with the ML system.

Created: 2025-12-31
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from HoloLoom.ml.config import (
    DataConfig,
    LinearRegressionConfig,
    MLSystemConfig,
    RegistryConfig,
    TrainingConfig,
    ValidationConfig,
)
from HoloLoom.ml.protocol import (
    DataSplit,
    EvaluationResult,
    ModelStatus,
    ModelType,
    PredictionResult,
    TrainingResult,
)
from HoloLoom.ml.registry import ModelRegistry

logger = logging.getLogger(__name__)


class MLOperation(str, Enum):
    """Available ML operations."""

    TRAIN = "train"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    LIST_MODELS = "list_models"
    DESCRIBE_MODEL = "describe_model"
    DELETE_MODEL = "delete_model"
    COMPARE_MODELS = "compare_models"
    VALIDATE_DATA = "validate_data"


@dataclass
class SkillInput:
    """Input for skill execution."""

    operation: MLOperation
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result from skill execution."""

    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class MLTrainerSkill:
    """
    Agentic skill for ML training operations.

    Provides natural language interface for:
    - Training models on tabular data
    - Evaluating model performance
    - Making predictions
    - Managing model lifecycle

    Usage:
        skill = MLTrainerSkill()

        # Train a model
        result = await skill.execute(SkillInput(
            operation=MLOperation.TRAIN,
            parameters={
                "data_source": "sales.csv",
                "target_column": "revenue",
                "model_type": "linear_regression"
            }
        ))

        # Natural language parsing
        result = await skill.parse_and_execute(
            "Train a linear regression model on sales.csv to predict revenue"
        )
    """

    # Trainer registry - maps model type to trainer class
    _trainer_registry: Dict[ModelType, type] = {}

    def __init__(
        self,
        config: Optional[MLSystemConfig] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize ML Trainer Skill.

        Args:
            config: ML system configuration
            registry: Model registry (creates default if None)
        """
        self.config = config or MLSystemConfig.default()
        self.registry = registry or ModelRegistry(self.config.registry)
        self._register_trainers()

    def _register_trainers(self) -> None:
        """Register available trainer classes."""
        # Import trainers lazily to avoid circular imports
        try:
            from HoloLoom.ml.trainers.linear_regression import LinearRegressionTrainer

            self._trainer_registry[ModelType.LINEAR_REGRESSION] = LinearRegressionTrainer
        except ImportError:
            logger.warning("LinearRegressionTrainer not available")

        # Future trainers can be registered here
        # self._trainer_registry[ModelType.RIDGE] = RidgeRegressionTrainer
        # self._trainer_registry[ModelType.LASSO] = LassoRegressionTrainer

    async def execute(self, input: SkillInput) -> SkillResult:
        """
        Execute an ML operation.

        Args:
            input: Skill input with operation and parameters

        Returns:
            SkillResult with outcome
        """
        operation = input.operation
        params = input.parameters

        try:
            if operation == MLOperation.TRAIN:
                return await self._train(params)
            elif operation == MLOperation.EVALUATE:
                return await self._evaluate(params)
            elif operation == MLOperation.PREDICT:
                return await self._predict(params)
            elif operation == MLOperation.LIST_MODELS:
                return await self._list_models(params)
            elif operation == MLOperation.DESCRIBE_MODEL:
                return await self._describe_model(params)
            elif operation == MLOperation.DELETE_MODEL:
                return await self._delete_model(params)
            elif operation == MLOperation.COMPARE_MODELS:
                return await self._compare_models(params)
            elif operation == MLOperation.VALIDATE_DATA:
                return await self._validate_data(params)
            else:
                return SkillResult(
                    success=False,
                    message=f"Unknown operation: {operation}",
                    suggestions=["Use one of: train, evaluate, predict, list_models, describe_model"],
                )
        except Exception as e:
            logger.exception(f"Error executing {operation}: {e}")
            return SkillResult(
                success=False,
                message=f"Error: {str(e)}",
                suggestions=self._get_error_suggestions(operation, e),
            )

    async def parse_and_execute(self, natural_language: str) -> SkillResult:
        """
        Parse natural language input and execute the appropriate operation.

        Args:
            natural_language: Natural language command

        Returns:
            SkillResult with outcome

        Examples:
            "Train a linear regression model on sales.csv to predict revenue"
            "Evaluate model linreg_001"
            "List all trained models"
            "Predict using linreg_001 on new_data.csv"
            "Delete model linreg_001"
            "Compare models linreg_001 and linreg_002"
            "Validate data in sales.csv for target revenue"
        """
        text = natural_language.lower().strip()

        # Parse operation and parameters
        operation, params = self._parse_natural_language(text)

        if operation is None:
            return SkillResult(
                success=False,
                message="Could not understand the command",
                suggestions=[
                    "Try: 'Train a linear regression model on data.csv to predict target_column'",
                    "Try: 'Evaluate model <model_id>'",
                    "Try: 'List all models'",
                    "Try: 'Predict using <model_id> on data.csv'",
                ],
            )

        return await self.execute(SkillInput(operation=operation, parameters=params))

    def _parse_natural_language(
        self, text: str
    ) -> tuple[Optional[MLOperation], Dict[str, Any]]:
        """
        Parse natural language into operation and parameters.

        Args:
            text: Natural language input (lowercase)

        Returns:
            Tuple of (operation, parameters) or (None, {}) if not understood
        """
        params: Dict[str, Any] = {}

        # TRAIN patterns
        train_patterns = [
            r"train\s+(?:a\s+)?(\w+(?:\s+\w+)?)\s+(?:model\s+)?on\s+(\S+)\s+(?:to\s+)?predict(?:ing)?\s+(\w+)",
            r"train\s+on\s+(\S+)\s+(?:to\s+)?predict(?:ing)?\s+(\w+)",
            r"create\s+(?:a\s+)?(\w+(?:\s+\w+)?)\s+model\s+(?:from|using)\s+(\S+)\s+(?:for|to predict)\s+(\w+)",
        ]

        for pattern in train_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    model_type_str, data_source, target = groups
                    params["model_type"] = self._parse_model_type(model_type_str)
                elif len(groups) == 2:
                    data_source, target = groups
                    params["model_type"] = "linear_regression"
                else:
                    continue

                params["data_source"] = data_source
                params["target_column"] = target
                return MLOperation.TRAIN, params

        # EVALUATE patterns
        eval_patterns = [
            r"evaluate\s+(?:model\s+)?(\S+)",
            r"test\s+(?:model\s+)?(\S+)",
            r"assess\s+(?:model\s+)?(\S+)",
            r"check\s+(?:model\s+)?(\S+)\s+performance",
        ]

        for pattern in eval_patterns:
            match = re.search(pattern, text)
            if match:
                params["model_id"] = match.group(1)
                return MLOperation.EVALUATE, params

        # PREDICT patterns
        predict_patterns = [
            r"predict\s+(?:using|with)\s+(\S+)\s+on\s+(\S+)",
            r"use\s+(\S+)\s+to\s+predict\s+(?:on\s+)?(\S+)",
            r"make\s+predictions?\s+(?:using|with)\s+(\S+)\s+(?:on|for)\s+(\S+)",
        ]

        for pattern in predict_patterns:
            match = re.search(pattern, text)
            if match:
                params["model_id"] = match.group(1)
                params["data_source"] = match.group(2)
                return MLOperation.PREDICT, params

        # LIST patterns
        list_patterns = [
            r"list\s+(?:all\s+)?(?:trained\s+)?models?",
            r"show\s+(?:all\s+)?(?:trained\s+)?models?",
            r"what\s+models?\s+(?:do\s+)?(?:i\s+)?have",
            r"get\s+(?:all\s+)?models?",
        ]

        for pattern in list_patterns:
            if re.search(pattern, text):
                # Check for filters
                if "linear" in text:
                    params["model_type"] = "linear_regression"
                return MLOperation.LIST_MODELS, params

        # DESCRIBE patterns
        describe_patterns = [
            r"describe\s+(?:model\s+)?(\S+)",
            r"(?:show|get)\s+(?:model\s+)?(\S+)\s+(?:details?|info)",
            r"what\s+is\s+(?:model\s+)?(\S+)",
            r"tell\s+me\s+about\s+(?:model\s+)?(\S+)",
        ]

        for pattern in describe_patterns:
            match = re.search(pattern, text)
            if match:
                params["model_id"] = match.group(1)
                return MLOperation.DESCRIBE_MODEL, params

        # DELETE patterns
        delete_patterns = [
            r"delete\s+(?:model\s+)?(\S+)",
            r"remove\s+(?:model\s+)?(\S+)",
            r"archive\s+(?:model\s+)?(\S+)",
        ]

        for pattern in delete_patterns:
            match = re.search(pattern, text)
            if match:
                params["model_id"] = match.group(1)
                return MLOperation.DELETE_MODEL, params

        # COMPARE patterns
        compare_patterns = [
            r"compare\s+(?:models?\s+)?(\S+)\s+(?:and|with|to|vs)\s+(\S+)",
        ]

        for pattern in compare_patterns:
            match = re.search(pattern, text)
            if match:
                params["model_ids"] = [match.group(1), match.group(2)]
                return MLOperation.COMPARE_MODELS, params

        # VALIDATE patterns
        validate_patterns = [
            r"validate\s+(?:data\s+)?(?:in\s+)?(\S+)\s+(?:for\s+)?(?:target\s+)?(\w+)",
            r"check\s+(?:data\s+)?(?:quality\s+)?(?:of\s+)?(\S+)",
        ]

        for pattern in validate_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                params["data_source"] = groups[0]
                if len(groups) > 1 and groups[1]:
                    params["target_column"] = groups[1]
                return MLOperation.VALIDATE_DATA, params

        return None, {}

    def _parse_model_type(self, text: str) -> str:
        """Parse model type from text."""
        text = text.lower().replace("_", " ").replace("-", " ")

        if "linear" in text and "regression" in text:
            return "linear_regression"
        elif "ridge" in text:
            return "ridge"
        elif "lasso" in text:
            return "lasso"
        elif "elastic" in text:
            return "elastic_net"
        else:
            # Default to linear regression
            return "linear_regression"

    async def _train(self, params: Dict[str, Any]) -> SkillResult:
        """Execute training operation."""
        data_source = params.get("data_source")
        target_column = params.get("target_column")
        model_type_str = params.get("model_type", "linear_regression")
        feature_columns = params.get("feature_columns")

        if not data_source:
            return SkillResult(
                success=False,
                message="Missing data source",
                suggestions=["Specify data source: 'train on data.csv to predict target'"],
            )

        if not target_column:
            return SkillResult(
                success=False,
                message="Missing target column",
                suggestions=["Specify target: 'train on data.csv to predict target_column'"],
            )

        # Parse model type
        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            return SkillResult(
                success=False,
                message=f"Unknown model type: {model_type_str}",
                suggestions=[f"Available types: {[t.value for t in ModelType]}"],
            )

        # Check trainer availability
        trainer_class = self._trainer_registry.get(model_type)
        if trainer_class is None:
            return SkillResult(
                success=False,
                message=f"No trainer available for {model_type.value}",
                suggestions=[f"Available trainers: {[t.value for t in self._trainer_registry.keys()]}"],
            )

        # Load and validate data
        try:
            from HoloLoom.ml.integration.datapig_adapter import DataPigMLAdapter
            from HoloLoom.ml.integration.spinningwheel_adapter import SpinningWheelMLAdapter

            # Ingest data
            data_adapter = SpinningWheelMLAdapter(self.config.data)
            data_split = await data_adapter.ingest_and_split(
                source=data_source,
                target_column=target_column,
                feature_columns=feature_columns,
            )

            # Validate data
            validator = DataPigMLAdapter(self.config.validation)
            validation_result = validator.validate_data_split(data_split)

            if not validation_result.is_valid:
                blocking_issues = [i for i in validation_result.issues if i.blocking]
                return SkillResult(
                    success=False,
                    message=f"Data validation failed: {len(blocking_issues)} blocking issues",
                    data={
                        "issues": [
                            {"type": i.issue_type, "message": i.message, "column": i.column}
                            for i in blocking_issues
                        ],
                        "recommendations": validation_result.recommendations,
                    },
                    suggestions=validation_result.recommendations[:3],
                )

        except ImportError as e:
            return SkillResult(
                success=False,
                message=f"Missing dependency: {e}",
                suggestions=["Install pandas: pip install pandas>=1.3"],
            )
        except Exception as e:
            return SkillResult(
                success=False,
                message=f"Error loading data: {e}",
                suggestions=["Check file path and format"],
            )

        # Train model
        try:
            trainer = trainer_class(self.config.get_model_config(model_type))
            training_result = await trainer.train(data_split)

            if training_result.status != ModelStatus.TRAINED:
                return SkillResult(
                    success=False,
                    message=f"Training failed: {training_result.metadata.get('error', 'unknown error')}",
                )

            # Evaluate on test data
            evaluation = await trainer.evaluate(training_result, data_split)

            # Save to registry
            self.registry.save(training_result, evaluation)

            return SkillResult(
                success=True,
                message=f"Successfully trained model: {training_result.model_id}",
                data={
                    "model_id": training_result.model_id,
                    "model_type": model_type.value,
                    "training_duration_ms": training_result.training_duration_ms,
                    "metrics": {
                        "r2_score": evaluation.r2_score,
                        "mse": evaluation.mse,
                        "rmse": evaluation.rmse,
                        "mae": evaluation.mae,
                    },
                    "feature_importance": training_result.feature_importance,
                    "n_features": len(data_split.feature_names or []),
                    "n_samples": len(data_split.X_train),
                },
                suggestions=[
                    f"Evaluate with: 'evaluate model {training_result.model_id}'",
                    f"Predict with: 'predict using {training_result.model_id} on new_data.csv'",
                ],
            )

        except Exception as e:
            logger.exception(f"Training error: {e}")
            return SkillResult(
                success=False,
                message=f"Training error: {e}",
            )

    async def _evaluate(self, params: Dict[str, Any]) -> SkillResult:
        """Execute evaluation operation."""
        model_id = params.get("model_id")

        if not model_id:
            return SkillResult(
                success=False,
                message="Missing model ID",
                suggestions=["Specify model: 'evaluate model linreg_001'"],
            )

        # Load evaluation from registry
        evaluation = self.registry.load_evaluation(model_id)

        if evaluation is None:
            # Try to load model and re-evaluate if data available
            training_result = self.registry.load(model_id)
            if training_result is None:
                return SkillResult(
                    success=False,
                    message=f"Model not found: {model_id}",
                    suggestions=["Use 'list models' to see available models"],
                )

            return SkillResult(
                success=True,
                message=f"Model {model_id} exists but no evaluation stored",
                data={
                    "model_id": model_id,
                    "model_type": training_result.model_type.value,
                    "status": training_result.status.value,
                    "metrics": training_result.metrics,
                },
                suggestions=["Retrain to generate evaluation metrics"],
            )

        return SkillResult(
            success=True,
            message=f"Evaluation results for {model_id}",
            data={
                "model_id": model_id,
                "r2_score": evaluation.r2_score,
                "adjusted_r2": evaluation.adjusted_r2,
                "mse": evaluation.mse,
                "rmse": evaluation.rmse,
                "mae": evaluation.mae,
                "cv_mean": evaluation.cv_mean,
                "cv_std": evaluation.cv_std,
                "residuals_stats": evaluation.residuals_stats,
            },
        )

    async def _predict(self, params: Dict[str, Any]) -> SkillResult:
        """Execute prediction operation."""
        model_id = params.get("model_id")
        data_source = params.get("data_source")
        output_path = params.get("output_path")

        if not model_id:
            return SkillResult(
                success=False,
                message="Missing model ID",
                suggestions=["Specify model: 'predict using linreg_001 on data.csv'"],
            )

        if not data_source:
            return SkillResult(
                success=False,
                message="Missing data source",
                suggestions=["Specify data: 'predict using linreg_001 on new_data.csv'"],
            )

        # Load model
        training_result = self.registry.load(model_id)
        if training_result is None:
            return SkillResult(
                success=False,
                message=f"Model not found: {model_id}",
                suggestions=["Use 'list models' to see available models"],
            )

        # Get trainer
        trainer_class = self._trainer_registry.get(training_result.model_type)
        if trainer_class is None:
            return SkillResult(
                success=False,
                message=f"No trainer for model type: {training_result.model_type.value}",
            )

        try:
            from HoloLoom.ml.integration.spinningwheel_adapter import SpinningWheelMLAdapter
            import numpy as np

            # Load data (use first column as dummy target)
            data_adapter = SpinningWheelMLAdapter(self.config.data)

            # For prediction, we need to handle data without target
            # Read raw data and extract features
            import pandas as pd

            source_path = Path(data_source)
            if source_path.suffix.lower() == ".csv":
                df = pd.read_csv(source_path)
            elif source_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(source_path)
            elif source_path.suffix.lower() == ".json":
                df = pd.read_json(source_path)
            elif source_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(source_path)
            else:
                df = pd.read_csv(source_path)

            # Use feature names from training
            feature_names = training_result.feature_names
            if feature_names:
                # Filter to only known features
                available = [f for f in feature_names if f in df.columns]
                if not available:
                    return SkillResult(
                        success=False,
                        message="No matching features found in prediction data",
                        data={"expected_features": feature_names, "available_columns": df.columns.tolist()},
                    )
                X = df[available].values.astype(np.float64)
            else:
                X = df.values.astype(np.float64)

            # Make predictions
            trainer = trainer_class(self.config.get_model_config(training_result.model_type))
            prediction_result = await trainer.predict(training_result, X)

            # Save predictions if output path specified
            if output_path:
                output_df = pd.DataFrame({"prediction": prediction_result.predictions})
                output_path = Path(output_path)
                if output_path.suffix.lower() == ".csv":
                    output_df.to_csv(output_path, index=False)
                else:
                    output_df.to_csv(output_path, index=False)

            return SkillResult(
                success=True,
                message=f"Generated {len(prediction_result.predictions)} predictions using {model_id}",
                data={
                    "model_id": model_id,
                    "n_predictions": len(prediction_result.predictions),
                    "predictions_sample": prediction_result.predictions[:10].tolist(),
                    "prediction_stats": {
                        "mean": float(np.mean(prediction_result.predictions)),
                        "std": float(np.std(prediction_result.predictions)),
                        "min": float(np.min(prediction_result.predictions)),
                        "max": float(np.max(prediction_result.predictions)),
                    },
                    "output_path": str(output_path) if output_path else None,
                },
            )

        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            return SkillResult(
                success=False,
                message=f"Prediction error: {e}",
            )

    async def _list_models(self, params: Dict[str, Any]) -> SkillResult:
        """List available models."""
        model_type_str = params.get("model_type")
        status_str = params.get("status")
        min_r2 = params.get("min_r2")
        limit = params.get("limit", 20)

        # Parse filters
        model_type = None
        if model_type_str:
            try:
                model_type = ModelType(model_type_str)
            except ValueError:
                pass

        status = None
        if status_str:
            try:
                status = ModelStatus(status_str)
            except ValueError:
                pass

        # Query registry
        models = self.registry.list_models(
            model_type=model_type,
            status=status,
            min_r2=min_r2,
            limit=limit,
        )

        if not models:
            return SkillResult(
                success=True,
                message="No models found",
                data={"models": [], "total": 0},
                suggestions=["Train a model: 'train on data.csv to predict target'"],
            )

        return SkillResult(
            success=True,
            message=f"Found {len(models)} model(s)",
            data={
                "models": [
                    {
                        "model_id": m["model_id"],
                        "model_type": m.get("model_type"),
                        "status": m.get("status"),
                        "r2_score": m.get("r2_score"),
                        "timestamp": m.get("timestamp"),
                    }
                    for m in models
                ],
                "total": len(models),
            },
        )

    async def _describe_model(self, params: Dict[str, Any]) -> SkillResult:
        """Describe a specific model."""
        model_id = params.get("model_id")

        if not model_id:
            return SkillResult(
                success=False,
                message="Missing model ID",
                suggestions=["Specify model: 'describe model linreg_001'"],
            )

        # Load model and evaluation
        training_result = self.registry.load(model_id)
        if training_result is None:
            return SkillResult(
                success=False,
                message=f"Model not found: {model_id}",
                suggestions=["Use 'list models' to see available models"],
            )

        evaluation = self.registry.load_evaluation(model_id)

        # Build description
        data = {
            "model_id": model_id,
            "model_type": training_result.model_type.value,
            "status": training_result.status.value,
            "timestamp": training_result.timestamp.isoformat(),
            "training_duration_ms": training_result.training_duration_ms,
            "hyperparameters": training_result.hyperparameters,
            "feature_names": training_result.feature_names,
            "feature_importance": training_result.feature_importance,
            "data_hash": training_result.data_hash,
        }

        if evaluation:
            data["evaluation"] = {
                "r2_score": evaluation.r2_score,
                "adjusted_r2": evaluation.adjusted_r2,
                "mse": evaluation.mse,
                "rmse": evaluation.rmse,
                "mae": evaluation.mae,
                "cv_mean": evaluation.cv_mean,
                "cv_std": evaluation.cv_std,
            }

        return SkillResult(
            success=True,
            message=f"Model: {model_id} ({training_result.model_type.value})",
            data=data,
        )

    async def _delete_model(self, params: Dict[str, Any]) -> SkillResult:
        """Delete (archive) a model."""
        model_id = params.get("model_id")

        if not model_id:
            return SkillResult(
                success=False,
                message="Missing model ID",
                suggestions=["Specify model: 'delete model linreg_001'"],
            )

        success = self.registry.delete(model_id)

        if success:
            return SkillResult(
                success=True,
                message=f"Model {model_id} archived successfully",
                data={"model_id": model_id, "archived": True},
            )
        else:
            return SkillResult(
                success=False,
                message=f"Model not found: {model_id}",
                suggestions=["Use 'list models' to see available models"],
            )

    async def _compare_models(self, params: Dict[str, Any]) -> SkillResult:
        """Compare multiple models."""
        model_ids = params.get("model_ids", [])

        if len(model_ids) < 2:
            return SkillResult(
                success=False,
                message="Need at least 2 models to compare",
                suggestions=["Try: 'compare models linreg_001 and linreg_002'"],
            )

        comparisons = []
        for model_id in model_ids:
            training_result = self.registry.load(model_id)
            evaluation = self.registry.load_evaluation(model_id)

            if training_result:
                comparisons.append({
                    "model_id": model_id,
                    "model_type": training_result.model_type.value,
                    "status": training_result.status.value,
                    "r2_score": evaluation.r2_score if evaluation else None,
                    "rmse": evaluation.rmse if evaluation else None,
                    "mae": evaluation.mae if evaluation else None,
                    "training_duration_ms": training_result.training_duration_ms,
                })
            else:
                comparisons.append({
                    "model_id": model_id,
                    "error": "Model not found",
                })

        # Determine best model by R²
        valid_models = [c for c in comparisons if c.get("r2_score") is not None]
        best_model = None
        if valid_models:
            best_model = max(valid_models, key=lambda x: x["r2_score"])

        return SkillResult(
            success=True,
            message=f"Comparison of {len(model_ids)} models",
            data={
                "comparisons": comparisons,
                "best_model": best_model["model_id"] if best_model else None,
                "best_r2": best_model["r2_score"] if best_model else None,
            },
        )

    async def _validate_data(self, params: Dict[str, Any]) -> SkillResult:
        """Validate data quality."""
        data_source = params.get("data_source")
        target_column = params.get("target_column")

        if not data_source:
            return SkillResult(
                success=False,
                message="Missing data source",
                suggestions=["Specify data: 'validate data in data.csv'"],
            )

        try:
            from HoloLoom.ml.integration.datapig_adapter import DataPigMLAdapter
            from HoloLoom.ml.integration.spinningwheel_adapter import SpinningWheelMLAdapter
            import numpy as np

            # Load data
            data_adapter = SpinningWheelMLAdapter(self.config.data)

            if target_column:
                X, y, feature_names = await data_adapter.ingest_tabular(
                    source=data_source,
                    target_column=target_column,
                )
            else:
                # No target specified, just load features
                import pandas as pd

                source_path = Path(data_source)
                if source_path.suffix.lower() == ".csv":
                    df = pd.read_csv(source_path)
                else:
                    df = pd.read_csv(source_path)

                X = df.values.astype(np.float64)
                y = np.zeros(len(df))  # Dummy target
                feature_names = df.columns.tolist()

            # Validate
            validator = DataPigMLAdapter(self.config.validation)
            result = validator.validate_training_data(X, y, feature_names)

            return SkillResult(
                success=True,
                message=f"Validation complete: {'PASS' if result.is_valid else 'FAIL'}",
                data={
                    "is_valid": result.is_valid,
                    "n_samples": result.stats.get("n_samples"),
                    "n_features": result.stats.get("n_features"),
                    "issues": [
                        {
                            "type": i.issue_type,
                            "severity": i.severity,
                            "message": i.message,
                            "column": i.column,
                            "blocking": i.blocking,
                        }
                        for i in result.issues
                    ],
                    "recommendations": result.recommendations,
                },
                suggestions=result.recommendations[:3] if not result.is_valid else [],
            )

        except Exception as e:
            logger.exception(f"Validation error: {e}")
            return SkillResult(
                success=False,
                message=f"Validation error: {e}",
            )

    def _get_error_suggestions(
        self, operation: MLOperation, error: Exception
    ) -> List[str]:
        """Get helpful suggestions based on error type."""
        error_str = str(error).lower()

        if "file not found" in error_str or "no such file" in error_str:
            return [
                "Check that the file path is correct",
                "Use absolute path if relative path doesn't work",
            ]

        if "column" in error_str and "not found" in error_str:
            return [
                "Check column names in your data file",
                "Column names are case-sensitive",
            ]

        if "sklearn" in error_str or "scikit" in error_str:
            return ["Install scikit-learn: pip install scikit-learn>=1.0"]

        if "pandas" in error_str:
            return ["Install pandas: pip install pandas>=1.3"]

        return [
            "Check your input parameters",
            f"Use 'help {operation.value}' for usage information",
        ]

    def get_help(self, operation: Optional[str] = None) -> str:
        """
        Get help text for operations.

        Args:
            operation: Specific operation or None for general help

        Returns:
            Help text
        """
        if operation is None:
            return """
ML Trainer Skill - Available Operations:

TRAIN
  Train a model on tabular data.
  Example: "Train a linear regression model on sales.csv to predict revenue"

EVALUATE
  Evaluate a trained model.
  Example: "Evaluate model linreg_001"

PREDICT
  Make predictions using a trained model.
  Example: "Predict using linreg_001 on new_data.csv"

LIST_MODELS
  List all trained models.
  Example: "List all models" or "Show linear regression models"

DESCRIBE_MODEL
  Get details about a specific model.
  Example: "Describe model linreg_001"

DELETE_MODEL
  Archive a model.
  Example: "Delete model linreg_001"

COMPARE_MODELS
  Compare multiple models.
  Example: "Compare models linreg_001 and linreg_002"

VALIDATE_DATA
  Check data quality before training.
  Example: "Validate data in sales.csv for target revenue"
"""

        operation_help = {
            "train": """
TRAIN Operation

Trains a machine learning model on tabular data.

Parameters:
  - data_source: Path to CSV, Excel, JSON, or Parquet file
  - target_column: Column to predict
  - model_type: "linear_regression" (default), "ridge", "lasso"
  - feature_columns: Optional list of feature columns

Examples:
  "Train a linear regression model on sales.csv to predict revenue"
  "Train on data.xlsx to predict price"
  "Create a ridge model from features.csv for outcome"
""",
            "evaluate": """
EVALUATE Operation

Evaluates a trained model's performance.

Parameters:
  - model_id: ID of the model to evaluate

Output includes:
  - R² score, adjusted R²
  - MSE, RMSE, MAE
  - Cross-validation scores

Examples:
  "Evaluate model linreg_001"
  "Test linreg_002"
""",
            "predict": """
PREDICT Operation

Makes predictions using a trained model.

Parameters:
  - model_id: ID of the model to use
  - data_source: Path to data file
  - output_path: Optional path to save predictions

Examples:
  "Predict using linreg_001 on new_data.csv"
  "Use linreg_002 to predict on test.csv"
""",
        }

        return operation_help.get(operation.lower(), f"No help available for: {operation}")


# Convenience function for quick skill creation
def create_ml_trainer_skill(
    config: Optional[MLSystemConfig] = None,
) -> MLTrainerSkill:
    """
    Create an ML Trainer Skill instance.

    Args:
        config: Optional configuration

    Returns:
        Configured MLTrainerSkill
    """
    return MLTrainerSkill(config=config)
