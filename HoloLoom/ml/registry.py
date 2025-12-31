"""
HoloLoom ML Training System - Model Registry

Persistent storage for trained models with versioning, querying, and archival.
Uses gzip-compressed pickle for models and JSON for metadata.

Created: 2025-12-31
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from HoloLoom.ml.config import RegistryConfig
from HoloLoom.ml.protocol import (
    EvaluationResult,
    ModelStatus,
    ModelType,
    TrainingResult,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Persistent storage for trained ML models.

    Features:
    - Gzip-compressed model storage
    - JSON metadata for fast querying
    - Model versioning (keeps max_versions per type)
    - Archival instead of hard delete
    - Thread-safe file operations

    Storage Layout:
        .hololoom/ml/models/
        ├── index.json
        ├── linear_regression/
        │   └── linreg_001/
        │       ├── model.pkl.gz
        │       ├── training.json
        │       └── evaluation.json
        └── archived/
            └── linreg_old/
                └── ...
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        """
        Initialize model registry.

        Args:
            config: Registry configuration (uses defaults if None)
        """
        self.config = config or RegistryConfig()
        self._ensure_directories()
        self._index = self._load_index()

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        (self.config.storage_path / "archived").mkdir(exist_ok=True)

    def _index_path(self) -> Path:
        """Get path to index file."""
        return self.config.storage_path / self.config.index_file

    def _load_index(self) -> Dict[str, Any]:
        """Load index from disk or create empty."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, creating new")
        return {"version": 1, "models": {}, "archived": []}

    def _save_index(self) -> None:
        """Save index to disk."""
        index_path = self._index_path()
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    def _model_dir(self, model_id: str, model_type: ModelType) -> Path:
        """Get directory for a specific model."""
        type_dir = self.config.storage_path / model_type.value
        type_dir.mkdir(exist_ok=True)
        return type_dir / model_id

    def save(
        self,
        result: TrainingResult,
        evaluation: Optional[EvaluationResult] = None,
    ) -> str:
        """
        Save trained model to registry.

        Args:
            result: Training result with model bytes
            evaluation: Optional evaluation results

        Returns:
            Model ID
        """
        model_id = result.model_id
        model_type = result.model_type

        # Create model directory
        model_dir = self._model_dir(model_id, model_type)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model bytes (compressed)
        model_path = model_dir / "model.pkl.gz"
        if self.config.compression:
            with gzip.open(
                model_path, "wb", compresslevel=self.config.compression_level
            ) as f:
                f.write(result.model_bytes)
        else:
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                f.write(result.model_bytes)

        # Save training metadata
        training_meta = result.to_dict()
        training_path = model_dir / "training.json"
        with open(training_path, "w") as f:
            json.dump(training_meta, f, indent=2, default=str)

        # Save evaluation if provided
        if evaluation:
            eval_path = model_dir / "evaluation.json"
            with open(eval_path, "w") as f:
                json.dump(evaluation.to_dict(), f, indent=2, default=str)

        # Update index
        self._index["models"][model_id] = {
            "model_type": model_type.value,
            "status": result.status.value,
            "timestamp": result.timestamp.isoformat(),
            "r2_score": evaluation.r2_score if evaluation else None,
            "data_hash": result.data_hash,
            "path": str(model_dir),
        }
        self._save_index()

        # Enforce max versions
        self._enforce_max_versions(model_type)

        logger.info(f"Saved model {model_id} to registry")
        return model_id

    def _enforce_max_versions(self, model_type: ModelType) -> None:
        """Archive old models if exceeding max_versions."""
        type_models = [
            (mid, info)
            for mid, info in self._index["models"].items()
            if info.get("model_type") == model_type.value
        ]

        if len(type_models) > self.config.max_versions:
            # Sort by timestamp, oldest first
            sorted_models = sorted(
                type_models,
                key=lambda x: x[1].get("timestamp", ""),
            )

            # Archive oldest models
            to_archive = sorted_models[: len(type_models) - self.config.max_versions]
            for model_id, _ in to_archive:
                self.delete(model_id)

    def load(self, model_id: str) -> Optional[TrainingResult]:
        """
        Load model from registry.

        Args:
            model_id: Model identifier

        Returns:
            TrainingResult with model bytes, or None if not found
        """
        if model_id not in self._index["models"]:
            logger.warning(f"Model {model_id} not found in registry")
            return None

        info = self._index["models"][model_id]
        model_dir = Path(info["path"])

        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return None

        # Load model bytes
        model_path = model_dir / "model.pkl.gz"
        if model_path.exists():
            with gzip.open(model_path, "rb") as f:
                model_bytes = f.read()
        else:
            model_path = model_dir / "model.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
            else:
                logger.warning(f"Model file not found in {model_dir}")
                return None

        # Load training metadata
        training_path = model_dir / "training.json"
        with open(training_path, "r") as f:
            meta = json.load(f)

        return TrainingResult(
            model_id=model_id,
            model_type=ModelType(meta["model_type"]),
            model_bytes=model_bytes,
            hyperparameters=meta.get("hyperparameters", {}),
            metrics=meta.get("metrics", {}),
            feature_names=meta.get("feature_names"),
            feature_importance=meta.get("feature_importance"),
            data_hash=meta.get("data_hash"),
            training_duration_ms=meta.get("training_duration_ms", 0.0),
            timestamp=datetime.fromisoformat(meta["timestamp"]),
            status=ModelStatus(meta.get("status", "trained")),
            metadata=meta.get("metadata", {}),
        )

    def load_evaluation(self, model_id: str) -> Optional[EvaluationResult]:
        """
        Load evaluation results for a model.

        Args:
            model_id: Model identifier

        Returns:
            EvaluationResult or None if not found
        """
        if model_id not in self._index["models"]:
            return None

        info = self._index["models"][model_id]
        eval_path = Path(info["path"]) / "evaluation.json"

        if not eval_path.exists():
            return None

        with open(eval_path, "r") as f:
            meta = json.load(f)

        return EvaluationResult(
            model_id=model_id,
            mse=meta.get("mse", 0.0),
            rmse=meta.get("rmse", 0.0),
            mae=meta.get("mae", 0.0),
            r2_score=meta.get("r2_score", 0.0),
            adjusted_r2=meta.get("adjusted_r2"),
            cv_scores=meta.get("cv_scores"),
            cv_mean=meta.get("cv_mean"),
            cv_std=meta.get("cv_std"),
            residuals_stats=meta.get("residuals_stats"),
            evaluation_duration_ms=meta.get("evaluation_duration_ms", 0.0),
            timestamp=datetime.fromisoformat(meta["timestamp"]),
            metadata=meta.get("metadata", {}),
        )

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        min_r2: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query models with optional filters.

        Args:
            model_type: Filter by model type
            status: Filter by status
            min_r2: Minimum R² score
            limit: Maximum results to return

        Returns:
            List of model info dictionaries
        """
        results = []

        for model_id, info in self._index["models"].items():
            # Apply filters
            if model_type and info.get("model_type") != model_type.value:
                continue
            if status and info.get("status") != status.value:
                continue
            if min_r2 and (info.get("r2_score") or 0) < min_r2:
                continue

            results.append({
                "model_id": model_id,
                **info,
            })

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return results[:limit]

    def get_best_model(
        self,
        model_type: ModelType,
        metric: str = "r2_score",
    ) -> Optional[str]:
        """
        Get the best model by a specific metric.

        Args:
            model_type: Model type to search
            metric: Metric to optimize (currently only r2_score indexed)

        Returns:
            Model ID of best model, or None
        """
        models = self.list_models(model_type=model_type)

        if not models:
            return None

        if metric == "r2_score":
            best = max(models, key=lambda x: x.get("r2_score") or 0)
            return best["model_id"]

        # For other metrics, need to load evaluation
        best_model = None
        best_value = None

        for model_info in models:
            eval_result = self.load_evaluation(model_info["model_id"])
            if eval_result:
                value = getattr(eval_result, metric, None)
                if value is not None:
                    # Minimize MSE/RMSE/MAE, maximize R²
                    is_better = (
                        best_value is None
                        or (metric in ["mse", "rmse", "mae"] and value < best_value)
                        or (metric not in ["mse", "rmse", "mae"] and value > best_value)
                    )
                    if is_better:
                        best_model = model_info["model_id"]
                        best_value = value

        return best_model

    def delete(self, model_id: str) -> bool:
        """
        Delete (archive) a model.

        Args:
            model_id: Model to delete

        Returns:
            True if deleted, False if not found
        """
        if model_id not in self._index["models"]:
            return False

        info = self._index["models"][model_id]
        model_dir = Path(info["path"])

        if self.config.enable_archiving:
            # Move to archived directory
            archive_dir = self.config.storage_path / "archived" / model_id
            if model_dir.exists():
                shutil.move(str(model_dir), str(archive_dir))
            self._index["archived"].append({
                "model_id": model_id,
                "archived_at": datetime.now().isoformat(),
                **info,
            })
        else:
            # Hard delete
            if model_dir.exists():
                shutil.rmtree(model_dir)

        # Remove from active index
        del self._index["models"][model_id]
        self._save_index()

        logger.info(f"Deleted model {model_id}")
        return True

    def update_status(self, model_id: str, status: ModelStatus) -> bool:
        """
        Update model status.

        Args:
            model_id: Model identifier
            status: New status

        Returns:
            True if updated, False if not found
        """
        if model_id not in self._index["models"]:
            return False

        self._index["models"][model_id]["status"] = status.value
        self._save_index()

        # Also update training.json
        info = self._index["models"][model_id]
        training_path = Path(info["path"]) / "training.json"
        if training_path.exists():
            with open(training_path, "r") as f:
                meta = json.load(f)
            meta["status"] = status.value
            with open(training_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        type_counts = {}
        for info in self._index["models"].values():
            model_type = info.get("model_type", "unknown")
            type_counts[model_type] = type_counts.get(model_type, 0) + 1

        return {
            "total_models": len(self._index["models"]),
            "archived_models": len(self._index.get("archived", [])),
            "models_by_type": type_counts,
            "storage_path": str(self.config.storage_path),
        }
