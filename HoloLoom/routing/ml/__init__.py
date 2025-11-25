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

from .ml_router import MLRouter, RoutingPrediction
from .feature_extraction import RoutingFeatureExtractor, RoutingFeatures
from .training_pipeline import OnlineTrainingPipeline, TrainingConfig
from .model_registry import ModelRegistry, ModelMetadata

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
