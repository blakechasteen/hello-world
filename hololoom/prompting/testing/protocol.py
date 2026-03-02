"""
Protocol definitions for Prompt Testing Framework.

Defines test cases, results, protocols, and configuration for systematic prompt validation
and quality assurance across all HoloLoom systems.

Status: ✅ Production Ready (November 2025)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Any
from datetime import datetime


class TestType(Enum):
    """Types of prompt tests."""
    GOLDEN = "golden"           # Test against golden outputs
    MUTATION = "mutation"       # Test robustness to prompt mutations
    REGRESSION = "regression"   # Test for quality degradation
    QUALITY = "quality"         # Test quality metrics
    AB_COMPARISON = "ab_comparison"  # A/B test variants


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class PromptTestCase:
    """Single prompt test case with inputs and expected qualities."""
    name: str
    """Test case name for identification."""

    prompt_template: str
    """Prompt template with optional placeholders."""

    test_inputs: List[Dict[str, Any]]
    """List of input dicts for testing (replaces placeholders)."""

    expected_qualities: Dict[str, float]
    """Expected quality scores by dimension (e.g., {'accuracy': 0.9, 'clarity': 0.8})."""

    golden_outputs: Optional[List[str]] = None
    """Golden/reference outputs for comparison (optional)."""

    test_type: TestType = TestType.QUALITY
    """Type of test this case represents."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (domain, model, version, etc)."""


@dataclass
class PromptTestResult:
    """Result of executing a single prompt test."""
    test_case: PromptTestCase
    """Original test case."""

    passed: bool
    """Whether test passed all quality thresholds."""

    quality_scores: Dict[str, float]
    """Actual quality scores by dimension."""

    latency_ms: float
    """Execution time in milliseconds."""

    token_count: int
    """Tokens used in LLM response."""

    regressions_detected: List[str] = field(default_factory=list)
    """Regression issues detected (quality drops, etc)."""

    error_message: Optional[str] = None
    """Error message if test failed."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When test was executed."""

    status: TestStatus = TestStatus.PASSED
    """Test execution status."""


@dataclass
class PromptTestReport:
    """Aggregated results from test suite execution."""
    test_results: List[PromptTestResult]
    """All individual test results."""

    total_tests: int
    """Total tests executed."""

    passed_count: int
    """Number of passed tests."""

    failed_count: int
    """Number of failed tests."""

    skipped_count: int
    """Number of skipped tests."""

    overall_pass_rate: float
    """Pass rate as percentage (0.0-1.0)."""

    avg_latency_ms: float
    """Average execution latency."""

    avg_quality_score: float
    """Average quality score across all dimensions."""

    regressions: List[str] = field(default_factory=list)
    """All detected regressions."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When report was generated."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Report metadata (model versions, configs, etc)."""


@dataclass
class PromptTestConfig:
    """Configuration for prompt testing framework."""
    quality_threshold: float = 0.8
    """Minimum quality score required (0.0-1.0)."""

    regression_threshold: float = 0.05
    """Maximum quality drop allowed (0.0-1.0)."""

    max_mutations: int = 10
    """Maximum mutations to test per prompt."""

    enable_golden_tests: bool = True
    """Enable golden dataset testing."""

    enable_mutation_tests: bool = True
    """Enable prompt mutation testing."""

    enable_regression_tests: bool = True
    """Enable regression detection."""

    parallel_execution: bool = True
    """Execute tests in parallel."""

    timeout_ms: int = 5000
    """Timeout per test in milliseconds."""

    # LLM Judge Configuration (December 2025)
    use_llm_judge: bool = True
    """Use LLM-based quality evaluation instead of heuristics."""

    llm_provider: str = "ollama"
    """LLM provider for judge ('ollama', 'anthropic', 'openai')."""

    llm_model: str = "llama3.2:3b"
    """LLM model for evaluation."""

    llm_criteria: List[str] = field(default_factory=lambda: [
        "quality", "relevance", "coherence", "completeness"
    ])
    """Quality criteria to evaluate (maps to JudgeCriteria enum)."""

    llm_temperature: float = 0.3
    """LLM temperature for consistent scoring (lower = more deterministic)."""

    llm_timeout_seconds: float = 30.0
    """Timeout for LLM evaluation calls."""

    fallback_to_heuristic: bool = True
    """Fall back to heuristic scoring if LLM unavailable."""


# Protocol Definitions

class PromptTesterProtocol(Protocol):
    """Main protocol for running prompt tests."""

    async def run_test(self, test_case: PromptTestCase) -> PromptTestResult:
        """Execute a single prompt test case.

        Args:
            test_case: Test case to execute

        Returns:
            Test result with quality scores and metadata
        """
        ...


class GoldenDatasetProtocol(Protocol):
    """Protocol for managing golden dataset (ground truth)."""

    def get_test_cases(self) -> List[PromptTestCase]:
        """Retrieve all golden test cases.

        Returns:
            List of test cases from golden dataset
        """
        ...

    def add_golden_pair(self, prompt: str, expected_output: str) -> None:
        """Add new golden prompt-output pair.

        Args:
            prompt: Prompt template
            expected_output: Expected response
        """
        ...


class MutationTesterProtocol(Protocol):
    """Protocol for testing prompt robustness via mutations."""

    def mutate(self, prompt: str) -> List[str]:
        """Generate prompt mutations for robustness testing.

        Args:
            prompt: Original prompt

        Returns:
            List of mutated versions
        """
        ...

    async def test_mutations(
        self, prompt: str, baseline_quality: float
    ) -> List[PromptTestResult]:
        """Test prompt mutations and compare to baseline.

        Args:
            prompt: Original prompt
            baseline_quality: Baseline quality score to compare against

        Returns:
            Test results for all mutations
        """
        ...


class RegressionDetectorProtocol(Protocol):
    """Protocol for detecting quality regressions."""

    async def detect_regressions(
        self, current_results: List[PromptTestResult],
        baseline_results: List[PromptTestResult]
    ) -> List[str]:
        """Detect quality regressions between baseline and current results.

        Args:
            current_results: Current test results
            baseline_results: Baseline/historical results

        Returns:
            List of detected regressions (quality drops exceeding threshold)
        """
        ...


class MetricsCollectorProtocol(Protocol):
    """Protocol for collecting and exporting test metrics."""

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        ...

    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics.

        Returns:
            Dictionary of metric_name -> value
        """
        ...

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        ...
