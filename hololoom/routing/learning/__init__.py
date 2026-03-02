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

from .pattern_miner import PatternMiner, Pattern, PatternScore, ClassificationLog
from .continuous_validator import (
    ContinuousValidator,
    ValidationResult,
    ValidationQuery,
    RegressionAlert,
    create_validation_set
)
from .adaptive_updater import (
    AdaptiveUpdater,
    DeploymentStrategy,
    DeploymentPhase,
    DeploymentResult,
    DeploymentMetrics,
    PatternVersion
)
from .performance_reporter import (
    PerformanceReporter,
    DailyReport,
    WeeklyReport
)

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
