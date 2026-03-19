"""
Adaptive Learning System for Moonshot Classifier
================================================
Self-improving query classification through pattern mining and continuous validation.

Components:
- PatternMiner: Extracts patterns from production logs
- ContinuousValidator: Hourly accuracy monitoring
- AdaptiveUpdater: Safe pattern deployment with rollback
- PerformanceReporter: Weekly reports and metrics

Author: Claude Code
Date: November 13, 2025
Phase: 3 - Adaptive Learning
"""

from .adaptive_updater import (
    AdaptiveUpdater,
    DeploymentMetrics,
    DeploymentPhase,
    DeploymentResult,
    DeploymentStrategy,
    PatternVersion,
)
from .continuous_validator import (
    ContinuousValidator,
    RegressionAlert,
    ValidationQuery,
    ValidationResult,
    create_validation_set,
)
from .pattern_miner import ClassificationLog, Pattern, PatternMiner, PatternScore
from .performance_reporter import DailyReport, PerformanceReporter, WeeklyReport

__all__ = [
    'PatternMiner',
    'Pattern',
    'PatternScore',
    'ClassificationLog',
    'ContinuousValidator',
    'ValidationResult',
    'ValidationQuery',
    'RegressionAlert',
    'create_validation_set',
    'AdaptiveUpdater',
    'DeploymentStrategy',
    'DeploymentPhase',
    'DeploymentResult',
    'DeploymentMetrics',
    'PatternVersion',
    'PerformanceReporter',
    'DailyReport',
    'WeeklyReport',
]
