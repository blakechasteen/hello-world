"""
Model Registry for ML Routing

Manages model versions, metadata, and deployment history.

Features:
- Model versioning with semantic versions
- Metadata tracking (accuracy, training date, dataset size)
- Model promotion (dev → staging → production)
- Rollback capability

Author: HoloLoom B2B Framework
Date: November 2025
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ModelStage(Enum):
    """Model deployment stage"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Metadata for a routing model"""
    version: str
    stage: ModelStage
    created_at: float
    trained_on_examples: int

    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Training info
    training_duration_seconds: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Deployment info
    deployed_at: float | None = None
    deployed_by: str = "system"


class ModelRegistry:
    """
    Registry for ML routing models.

    Tracks all model versions with metadata and manages deployment.

    Example:
        >>> registry = ModelRegistry(registry_path="models/registry.json")
        >>>
        >>> # Register new model
        >>> metadata = ModelMetadata(
        ...     version="v1.2.0",
        ...     stage=ModelStage.DEVELOPMENT,
        ...     created_at=time.time(),
        ...     trained_on_examples=5000,
        ...     accuracy=0.92,
        ...     precision=0.90,
        ...     recall=0.94,
        ...     f1_score=0.92,
        ...     training_duration_seconds=120.5
        ... )
        >>> registry.register_model(metadata)
        >>>
        >>> # Promote to production
        >>> registry.promote_model("v1.2.0", ModelStage.PRODUCTION)
    """

    def __init__(self, registry_path: str = "models/registry.json"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.models: dict[str, ModelMetadata] = {}

        # Load existing registry
        if self.registry_path.exists():
            self._load_registry()

    def register_model(self, metadata: ModelMetadata):
        """
        Register new model in registry.

        Args:
            metadata: Model metadata
        """
        self.models[metadata.version] = metadata
        self._save_registry()

    def promote_model(
        self,
        version: str,
        new_stage: ModelStage,
        deployed_by: str = "system"
    ):
        """
        Promote model to new deployment stage.

        Args:
            version: Model version to promote
            new_stage: Target deployment stage
            deployed_by: Who is deploying (user ID or "system")
        """
        if version not in self.models:
            raise ValueError(f"Model {version} not found in registry")

        model = self.models[version]

        # Validate promotion path
        if not self._validate_promotion(model.stage, new_stage):
            raise ValueError(
                f"Invalid promotion: {model.stage.value} → {new_stage.value}"
            )

        # If promoting to production, demote current production model
        if new_stage == ModelStage.PRODUCTION:
            self._demote_current_production()

        # Update model metadata
        model.stage = new_stage
        model.deployed_at = time.time()
        model.deployed_by = deployed_by

        self._save_registry()

    def _validate_promotion(
        self,
        current_stage: ModelStage,
        new_stage: ModelStage
    ) -> bool:
        """
        Validate promotion path.

        Valid paths:
        - DEVELOPMENT → STAGING
        - STAGING → PRODUCTION
        - * → ARCHIVED

        Args:
            current_stage: Current deployment stage
            new_stage: Target deployment stage

        Returns:
            True if valid promotion, False otherwise
        """
        valid_paths = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: []
        }

        return new_stage in valid_paths[current_stage]

    def _demote_current_production(self):
        """Demote current production model to archived"""
        for version, model in self.models.items():
            if model.stage == ModelStage.PRODUCTION:
                model.stage = ModelStage.ARCHIVED

    def get_production_model(self) -> ModelMetadata | None:
        """
        Get current production model.

        Returns:
            Production model metadata or None
        """
        for model in self.models.values():
            if model.stage == ModelStage.PRODUCTION:
                return model
        return None

    def get_model(self, version: str) -> ModelMetadata | None:
        """
        Get model by version.

        Args:
            version: Model version

        Returns:
            Model metadata or None
        """
        return self.models.get(version)

    def list_models(
        self,
        stage: ModelStage | None = None
    ) -> list[ModelMetadata]:
        """
        List all models, optionally filtered by stage.

        Args:
            stage: Filter by deployment stage (None = all)

        Returns:
            List of model metadata
        """
        models = list(self.models.values())

        if stage is not None:
            models = [m for m in models if m.stage == stage]

        # Sort by created_at descending
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    def rollback_production(self) -> ModelMetadata | None:
        """
        Rollback to previous production model.

        Returns:
            New production model or None if no rollback available
        """
        # Get all archived models (sorted by deployment time)
        archived = [
            m for m in self.models.values()
            if m.stage == ModelStage.ARCHIVED and m.deployed_at is not None
        ]
        archived.sort(key=lambda m: m.deployed_at, reverse=True)

        if not archived:
            return None

        # Promote most recent archived model
        previous_model = archived[0]
        self.promote_model(
            previous_model.version,
            ModelStage.PRODUCTION,
            deployed_by="rollback"
        )

        return previous_model

    def _save_registry(self):
        """Save registry to disk"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        registry_dict = {
            version: {
                "version": model.version,
                "stage": model.stage.value,
                "created_at": model.created_at,
                "trained_on_examples": model.trained_on_examples,
                "accuracy": model.accuracy,
                "precision": model.precision,
                "recall": model.recall,
                "f1_score": model.f1_score,
                "training_duration_seconds": model.training_duration_seconds,
                "hyperparameters": model.hyperparameters,
                "deployed_at": model.deployed_at,
                "deployed_by": model.deployed_by
            }
            for version, model in self.models.items()
        }

        with open(self.registry_path, "w") as f:
            json.dump(registry_dict, f, indent=2)

    def _load_registry(self):
        """Load registry from disk"""
        with open(self.registry_path) as f:
            registry_dict = json.load(f)

        for version, model_dict in registry_dict.items():
            self.models[version] = ModelMetadata(
                version=model_dict["version"],
                stage=ModelStage(model_dict["stage"]),
                created_at=model_dict["created_at"],
                trained_on_examples=model_dict["trained_on_examples"],
                accuracy=model_dict["accuracy"],
                precision=model_dict["precision"],
                recall=model_dict["recall"],
                f1_score=model_dict["f1_score"],
                training_duration_seconds=model_dict["training_duration_seconds"],
                hyperparameters=model_dict.get("hyperparameters", {}),
                deployed_at=model_dict.get("deployed_at"),
                deployed_by=model_dict.get("deployed_by", "system")
            )
