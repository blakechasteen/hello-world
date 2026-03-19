"""
ML-Based Routing for Departments

Machine learning models that learn optimal department routing from usage patterns.

Exports:
- MLRouter: ML-based router using trained models
- RoutingFeatureExtractor: Extract features for ML model
- OnlineTrainingPipeline: Continuous model training
- ModelRegistry: Manage model versions

Author: HoloLoom B2B Framework
Date: November 2025
"""

from .feature_extraction import RoutingFeatureExtractor, RoutingFeatures
from .ml_router import MLRouter, RoutingPrediction
from .model_registry import ModelMetadata, ModelRegistry
from .training_pipeline import OnlineTrainingPipeline, TrainingConfig

__all__ = [
    "MLRouter",
    "RoutingPrediction",
    "RoutingFeatureExtractor",
    "RoutingFeatures",
    "OnlineTrainingPipeline",
    "TrainingConfig",
    "ModelRegistry",
    "ModelMetadata",
]
