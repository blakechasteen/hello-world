"""
HoloLoom ML Trainers

Model trainers implementing TrainerProtocol for different ML algorithms.

Created: 2025-12-31
"""

from hololoom.ml.trainers.base_trainer import BaseTrainer
from hololoom.ml.trainers.linear_regression import LinearRegressionTrainer

__all__ = [
    "BaseTrainer",
    "LinearRegressionTrainer",
]
