"""HoloLoom Prompt Testing Framework.

Comprehensive prompt testing with:
- Golden dataset testing (baseline comparisons)
- Mutation testing (robustness evaluation)
- Regression detection (quality monitoring)
- LLM-powered evaluation using Ollama (December 2025)
- Statistical analysis and alerting
- CI integration support

Status: Production Ready (November 2025)
Updated: December 2025 - LLMJudge integration
"""

# Protocol definitions
# Golden chain datasets (December 2025)
from hololoom.prompting.testing.golden_chains import (
    CHAIN_GOLDEN_DATASETS,
    GoldenPair,
    TestDifficulty,
    filter_by_difficulty,
    filter_by_source,
    filter_by_tags,
    get_all_golden_datasets,
    get_cases_for_evaluation_criterion,
    get_dataset_summary,
    get_difficulty_distribution,
    get_golden_dataset,
    get_total_case_count,
)
from hololoom.prompting.testing.golden_chains import (
    GoldenPair as ChainGoldenPair,  # Alias for compatibility
)

# Golden dataset management
from hololoom.prompting.testing.golden_dataset import (
    GoldenDatasetManager,
    create_golden_dataset,
)

# Metrics collection
from hololoom.prompting.testing.metrics_collector import (
    Metric,
    MetricsAggregator,
    MetricsCollector,
    MetricType,
    create_metrics_collector,
)

# Mutation testing
from hololoom.prompting.testing.mutation_testing import (
    Mutation,
    MutationTester,
    MutationType,
    PromptMutator,
    create_mutation_tester,
)
from hololoom.prompting.testing.protocol import (
    PromptTestCase,
    PromptTestConfig,
    PromptTestReport,
    PromptTestResult,
    TestStatus,
    TestType,
)

# Regression detection
from hololoom.prompting.testing.regression_testing import (
    RegressionDetector,
    create_regression_detector,
)

# Test suite
from hololoom.prompting.testing.test_suite import (
    PromptTestSuite,
    create_test_suite,
)

__all__ = [
    # Protocol
    "PromptTestCase",
    "PromptTestResult",
    "PromptTestReport",
    "PromptTestConfig",
    "TestStatus",
    "TestType",
    # Test suite
    "PromptTestSuite",
    "create_test_suite",
    # Mutation testing
    "MutationType",
    "Mutation",
    "PromptMutator",
    "MutationTester",
    "create_mutation_tester",
    # Golden dataset
    "GoldenDatasetManager",
    "create_golden_dataset",
    # Regression detection
    "RegressionDetector",
    "create_regression_detector",
    # Metrics
    "MetricsCollector",
    "MetricType",
    "Metric",
    "MetricsAggregator",
    "create_metrics_collector",
    # Golden chains
    "GoldenPair",
    "TestDifficulty",
    "CHAIN_GOLDEN_DATASETS",
    "get_golden_dataset",
    "get_all_golden_datasets",
    "filter_by_difficulty",
    "filter_by_tags",
    "get_dataset_summary",
    "filter_by_source",
    "get_cases_for_evaluation_criterion",
    "get_total_case_count",
    "get_difficulty_distribution",
    "ChainGoldenPair",
]
