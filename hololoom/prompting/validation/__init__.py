"""
Phase 3+: Production Validation Framework

This package provides complete infrastructure for validating UnifiedMRF improvements
in production deployments through:

1. A/B Testing - Compare traditional vs UnifiedMRF prompts
2. Data Collection - Gather queries, responses, metrics, feedback
3. Statistical Analysis - Determine statistical significance
4. Human Evaluation - Collect blind side-by-side comparisons
5. Extensible Pipeline - Composable validators with plugin architecture
6. Moonshot Swarm - Parallel validation across multiple dimensions

Created: November 2025
Status: Production ready
"""

from .ab_testing import (
    ABTestConfig,
    ABTestResults,
    ABTestRunner,
    PromptExecution,
    PromptVariant,
)

from .data_collection import (
    ProductionDataCollector,
    ProductionQuery,
    QualityMetric,
    QuerySource,
)

from .statistical_analysis import (
    SignificanceLevel,
    StatisticalAnalyzer,
    TTestResult,
    ValidationReport,
)

from .human_evaluation import (
    EvaluationCriterion,
    EvaluationPair,
    EvaluationResults,
    HumanEvaluationCollector,
    Preference,
)

from .pipeline import (
    ValidationContext,
    ValidationPhase,
    Validator,
    QualityValidator,
    LatencyValidator,
    UserSatisfactionValidator,
    StatisticalValidator,
    ValidationPipeline,
    PipelineResults,
)

from .swarm import (
    SwarmDimension,
    SwarmAgent,
    SwarmResults,
    MoonshotSwarm,
)


__all__ = [
    # A/B Testing
    "ABTestConfig",
    "ABTestResults",
    "ABTestRunner",
    "PromptExecution",
    "PromptVariant",

    # Data Collection
    "ProductionDataCollector",
    "ProductionQuery",
    "QualityMetric",
    "QuerySource",

    # Statistical Analysis
    "SignificanceLevel",
    "StatisticalAnalyzer",
    "TTestResult",
    "ValidationReport",

    # Human Evaluation
    "EvaluationCriterion",
    "EvaluationPair",
    "EvaluationResults",
    "HumanEvaluationCollector",
    "Preference",

    # Extensible Pipeline
    "ValidationContext",
    "ValidationPhase",
    "Validator",
    "QualityValidator",
    "LatencyValidator",
    "UserSatisfactionValidator",
    "StatisticalValidator",
    "ValidationPipeline",
    "PipelineResults",

    # Moonshot Swarm
    "SwarmDimension",
    "SwarmAgent",
    "SwarmResults",
    "MoonshotSwarm",
]


# Version
__version__ = "1.1.0"  # Added pipeline and swarm
