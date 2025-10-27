#!/usr/bin/env python3
"""
Promptly A/B Testing Framework
================================
Compare prompt variants, track metrics, and determine winners.
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum


class Metric(Enum):
    """Evaluation metrics"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    QUALITY_SCORE = "quality_score"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CUSTOM = "custom"


@dataclass
class TestCase:
    """A single test case for evaluation"""
    input: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "metadata": self.metadata
        }


@dataclass
class VariantResult:
    """Results for a single variant"""
    variant_name: str
    prompt_version: int
    branch: str
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_times: List[float] = field(default_factory=list)
    tokens_used: List[int] = field(default_factory=list)
    errors: int = 0
    total_tests: int = 0

    def add_result(
        self,
        test_case: TestCase,
        output: str,
        execution_time: float,
        tokens: Optional[int] = None,
        score: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Add a test result"""
        self.test_results.append({
            "input": test_case.input,
            "output": output,
            "expected": test_case.expected_output,
            "score": score,
            "execution_time": execution_time,
            "tokens": tokens,
            "error": error
        })

        self.execution_times.append(execution_time)
        if tokens:
            self.tokens_used.append(tokens)
        if error:
            self.errors += 1
        self.total_tests += 1

    def calculate_metrics(self):
        """Calculate aggregate metrics"""
        if not self.test_results:
            return

        # Average latency
        if self.execution_times:
            self.metrics[Metric.LATENCY.value] = statistics.mean(self.execution_times)

        # Token efficiency (if available)
        if self.tokens_used:
            self.metrics[Metric.TOKEN_EFFICIENCY.value] = statistics.mean(self.tokens_used)

        # Accuracy (successful executions)
        success_rate = (self.total_tests - self.errors) / self.total_tests if self.total_tests > 0 else 0
        self.metrics[Metric.ACCURACY.value] = success_rate

        # Quality score (from evaluations)
        scores = [r["score"] for r in self.test_results if r.get("score") is not None]
        if scores:
            self.metrics[Metric.QUALITY_SCORE.value] = statistics.mean(scores)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "variant": self.variant_name,
            "version": self.prompt_version,
            "branch": self.branch,
            "total_tests": self.total_tests,
            "errors": self.errors,
            "success_rate": self.metrics.get(Metric.ACCURACY.value, 0),
            "avg_latency": self.metrics.get(Metric.LATENCY.value, 0),
            "avg_tokens": self.metrics.get(Metric.TOKEN_EFFICIENCY.value, 0),
            "quality_score": self.metrics.get(Metric.QUALITY_SCORE.value, None),
            "all_metrics": self.metrics
        }


@dataclass
class ABTest:
    """A/B test configuration and results"""
    test_name: str
    variants: List[str]  # Prompt names or branches
    test_cases: List[TestCase]
    created_at: datetime = field(default_factory=datetime.now)
    results: Dict[str, VariantResult] = field(default_factory=dict)
    winner: Optional[str] = None
    confidence: float = 0.0

    def add_variant_result(self, variant_name: str, result: VariantResult):
        """Add results for a variant"""
        self.results[variant_name] = result
        result.calculate_metrics()

    def determine_winner(
        self,
        primary_metric: Metric = Metric.QUALITY_SCORE,
        min_improvement: float = 0.05  # 5% improvement threshold
    ) -> str:
        """
        Determine the winning variant based on primary metric.

        Args:
            primary_metric: Metric to optimize for
            min_improvement: Minimum improvement to declare a winner

        Returns:
            Winner variant name
        """
        if not self.results:
            return None

        # Get metric values for all variants
        variant_scores = {}
        for variant_name, result in self.results.items():
            score = result.metrics.get(primary_metric.value)
            if score is not None:
                variant_scores[variant_name] = score

        if not variant_scores:
            return None

        # Find best variant
        if primary_metric == Metric.LATENCY:
            # Lower is better for latency
            winner = min(variant_scores.items(), key=lambda x: x[1])
        else:
            # Higher is better for other metrics
            winner = max(variant_scores.items(), key=lambda x: x[1])

        winner_name, winner_score = winner

        # Check if improvement is significant
        other_scores = [s for v, s in variant_scores.items() if v != winner_name]
        if other_scores:
            best_other = max(other_scores) if primary_metric != Metric.LATENCY else min(other_scores)

            if primary_metric == Metric.LATENCY:
                improvement = (best_other - winner_score) / best_other
            else:
                improvement = (winner_score - best_other) / best_other if best_other > 0 else 1.0

            if improvement >= min_improvement:
                self.winner = winner_name
                self.confidence = min(improvement, 1.0)
            else:
                # No clear winner
                self.winner = None
                self.confidence = 0.0
        else:
            # Only one variant
            self.winner = winner_name
            self.confidence = 1.0

        return self.winner

    def get_comparison(self) -> Dict[str, Any]:
        """Get detailed comparison of all variants"""
        comparison = {
            "test_name": self.test_name,
            "created_at": self.created_at.isoformat(),
            "variants": {},
            "winner": self.winner,
            "confidence": self.confidence
        }

        for variant_name, result in self.results.items():
            comparison["variants"][variant_name] = result.get_summary()

        return comparison

    def to_report(self) -> str:
        """Generate a human-readable report"""
        lines = [
            f"# A/B Test Report: {self.test_name}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Variants Tested: {len(self.variants)}",
            ""
        ]

        # Results for each variant
        for variant_name in self.variants:
            result = self.results.get(variant_name)
            if not result:
                continue

            summary = result.get_summary()
            lines.append(f"### {variant_name}")
            lines.append(f"- Version: {summary['version']}")
            lines.append(f"- Branch: {summary['branch']}")
            lines.append(f"- Tests Run: {summary['total_tests']}")
            lines.append(f"- Success Rate: {summary['success_rate']*100:.1f}%")
            lines.append(f"- Avg Latency: {summary['avg_latency']:.2f}s")

            if summary['avg_tokens']:
                lines.append(f"- Avg Tokens: {summary['avg_tokens']:.0f}")

            if summary['quality_score'] is not None:
                lines.append(f"- Quality Score: {summary['quality_score']:.2f}")

            lines.append("")

        # Winner
        if self.winner:
            lines.append(f"## 🏆 Winner: {self.winner}")
            lines.append(f"Confidence: {self.confidence*100:.1f}%")
        else:
            lines.append("## 🤷 No Clear Winner")
            lines.append("Results are too close to call.")

        return "\n".join(lines)


class ABTestRunner:
    """Run A/B tests with Promptly"""

    def __init__(self, promptly_instance, execution_engine):
        """
        Initialize A/B test runner.

        Args:
            promptly_instance: Instance of Promptly
            execution_engine: Instance of ExecutionEngine
        """
        self.promptly = promptly_instance
        self.engine = execution_engine

    def create_test(
        self,
        test_name: str,
        variants: List[str],
        test_cases: List[TestCase]
    ) -> ABTest:
        """Create a new A/B test"""
        return ABTest(
            test_name=test_name,
            variants=variants,
            test_cases=test_cases
        )

    def run_test(
        self,
        ab_test: ABTest,
        evaluator: Optional[Callable[[str, str], float]] = None
    ) -> ABTest:
        """
        Run an A/B test.

        Args:
            ab_test: ABTest configuration
            evaluator: Optional function(output, expected) -> score

        Returns:
            ABTest with results
        """
        print(f"Running A/B Test: {ab_test.test_name}")
        print(f"Variants: {', '.join(ab_test.variants)}")
        print(f"Test Cases: {len(ab_test.test_cases)}")
        print()

        for variant_name in ab_test.variants:
            print(f"Testing variant: {variant_name}")

            # Get prompt
            try:
                prompt_data = self.promptly.get(variant_name)
            except Exception as e:
                print(f"  ✗ Error loading variant: {e}")
                continue

            variant_result = VariantResult(
                variant_name=variant_name,
                prompt_version=prompt_data['version'],
                branch=prompt_data['branch']
            )

            # Run each test case
            for i, test_case in enumerate(ab_test.test_cases, 1):
                print(f"  Test {i}/{len(ab_test.test_cases)}...", end=" ")

                try:
                    # Format prompt with test input
                    formatted_prompt = prompt_data['content'].format(
                        input=test_case.input,
                        **test_case.metadata
                    )

                    # Execute
                    result = self.engine.execute_prompt(formatted_prompt, variant_name)

                    # Evaluate
                    score = None
                    if evaluator and test_case.expected_output and result.success:
                        score = evaluator(result.output, test_case.expected_output)

                    variant_result.add_result(
                        test_case=test_case,
                        output=result.output if result.success else "",
                        execution_time=result.execution_time,
                        tokens=result.tokens_used,
                        score=score,
                        error=result.error
                    )

                    print("✓")

                except Exception as e:
                    variant_result.add_result(
                        test_case=test_case,
                        output="",
                        execution_time=0.0,
                        error=str(e)
                    )
                    print(f"✗ {e}")

            ab_test.add_variant_result(variant_name, variant_result)
            print()

        # Determine winner
        ab_test.determine_winner()

        return ab_test


# ============================================================================
# Built-in Evaluators
# ============================================================================

def exact_match_evaluator(output: str, expected: str) -> float:
    """Exact string match (1.0 or 0.0)"""
    return 1.0 if output.strip() == expected.strip() else 0.0


def contains_evaluator(output: str, expected: str) -> float:
    """Check if output contains expected string (1.0 or 0.0)"""
    return 1.0 if expected.lower() in output.lower() else 0.0


def length_similarity_evaluator(output: str, expected: str) -> float:
    """Score based on length similarity (0.0 to 1.0)"""
    if not expected:
        return 1.0 if not output else 0.0

    len_ratio = min(len(output), len(expected)) / max(len(output), len(expected))
    return len_ratio


def word_overlap_evaluator(output: str, expected: str) -> float:
    """Score based on word overlap (0.0 to 1.0)"""
    output_words = set(output.lower().split())
    expected_words = set(expected.lower().split())

    if not expected_words:
        return 1.0 if not output_words else 0.0

    overlap = len(output_words & expected_words)
    return overlap / len(expected_words)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("A/B Testing Framework Loaded")
    print("\nAvailable Evaluators:")
    print("- exact_match_evaluator")
    print("- contains_evaluator")
    print("- length_similarity_evaluator")
    print("- word_overlap_evaluator")
    print("\nExample:")
    print("""
from promptly import Promptly
from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from ab_testing import ABTestRunner, TestCase

# Setup
p = Promptly()
config = ExecutionConfig(backend=ExecutionBackend.OLLAMA)
engine = ExecutionEngine(config)
runner = ABTestRunner(p, engine)

# Create test
test_cases = [
    TestCase(input="Hello world", expected_output="Hello"),
    TestCase(input="Goodbye", expected_output="Bye")
]

test = runner.create_test(
    "greeting_variants",
    variants=["greeter_v1", "greeter_v2"],
    test_cases=test_cases
)

# Run
result = runner.run_test(test, evaluator=contains_evaluator)
print(result.to_report())
""")
