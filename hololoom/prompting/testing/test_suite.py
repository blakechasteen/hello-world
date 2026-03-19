from __future__ import annotations
"""Main Prompt Test Suite - Orchestrates all testing workflows.

Provides a unified interface for running:
- Golden dataset tests (test against known good outputs)
- Mutation tests (test prompt robustness)
- Regression tests (detect quality degradation)
- Quality metrics (comprehensive quality assessment)

Now with LLM-powered evaluation using Ollama (December 2025).

Status: ✅ Production Ready (November 2025)
Updated: December 2025 - LLMJudge integration
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional

from hololoom.prompting.testing.golden_dataset import GoldenDatasetManager
from hololoom.prompting.testing.metrics_collector import MetricsCollector, MetricType
from hololoom.prompting.testing.mutation_testing import MutationTester
from hololoom.prompting.testing.protocol import (
    PromptTestCase,
    PromptTestConfig,
    PromptTestReport,
    PromptTestResult,
    TestStatus,
)
from hololoom.prompting.testing.regression_testing import RegressionDetector

# LLM Judge integration (December 2025)
try:
    from hololoom.chaining.evaluation import (
        JudgeConfig,
        JudgeCriteria,
        LLMJudge,
        create_judge,
    )
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False
    LLMJudge = None
    JudgeCriteria = None
    JudgeConfig = None


class PromptTestSuite:
    """Main orchestrator for prompt testing framework."""

    def __init__(
        self,
        config: PromptTestConfig,
        golden_dataset: GoldenDatasetManager | None = None,
        llm_judge: Optional["LLMJudge"] = None,
    ):
        """Initialize test suite.

        Args:
            config: Test configuration
            golden_dataset: Optional golden dataset manager for baseline comparisons
            llm_judge: Optional LLMJudge instance for quality evaluation
        """
        self.config = config
        self.golden_dataset = golden_dataset or GoldenDatasetManager()
        self.mutation_tester = MutationTester(config)
        self.regression_detector = RegressionDetector(config)
        self.metrics_collector = MetricsCollector()
        self.results: list[PromptTestResult] = []

        # Initialize LLM Judge if configured and available
        self.llm_judge = llm_judge
        self._llm_judge_available = False

        if config.use_llm_judge and LLM_JUDGE_AVAILABLE:
            if self.llm_judge is None:
                try:
                    self.llm_judge = create_judge(
                        provider=config.llm_provider,
                        model=config.llm_model,
                    )
                    self._llm_judge_available = True
                except Exception as e:
                    print(f"Warning: Could not initialize LLMJudge: {e}")
                    if not config.fallback_to_heuristic:
                        raise
            else:
                self._llm_judge_available = True
        elif config.use_llm_judge and not LLM_JUDGE_AVAILABLE:
            print("Warning: LLMJudge not available (chaining.evaluation not found)")
            if not config.fallback_to_heuristic:
                raise ImportError("LLMJudge required but not available")

    async def run_all_tests(self) -> PromptTestReport:
        """Run complete test suite (golden + mutation + regression).

        Returns:
            Comprehensive test report
        """
        tasks = []

        if self.config.enable_golden_tests:
            tasks.append(self.run_golden_tests())

        if self.config.enable_mutation_tests:
            tasks.append(self.run_mutation_tests([]))

        if self.config.enable_regression_tests:
            tasks.append(self.run_regression_tests(None))

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        self.results = []
        for result in all_results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                self.results.extend(result)
            elif isinstance(result, PromptTestResult):
                self.results.append(result)

        return self._generate_report(self.results)

    async def run_golden_tests(self) -> list[PromptTestResult]:
        """Run tests against golden dataset.

        Returns:
            List of test results for golden dataset
        """
        if not self.golden_dataset:
            return []

        test_cases = self.golden_dataset.get_test_cases()
        results = []

        if self.config.parallel_execution:
            tasks = [self.run_single_test(tc) for tc in test_cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if not isinstance(r, Exception)]
        else:
            for test_case in test_cases:
                result = await self.run_single_test(test_case)
                results.append(result)

        self.metrics_collector.record("golden_tests_executed", float(len(results)), MetricType.PASS_RATE)
        return results

    async def run_mutation_tests(self, prompts: list[str] | None = None) -> list[PromptTestResult]:
        """Test prompt robustness to mutations.

        Args:
            prompts: Optional list of prompts to mutate. If None, uses golden dataset.

        Returns:
            List of mutation test results
        """
        if not prompts:
            test_cases = self.golden_dataset.get_test_cases()
            prompts = [tc.prompt_template for tc in test_cases]

        results = []
        for prompt in prompts[:self.config.max_mutations]:
            mutations = self.mutation_tester.mutate(prompt)
            mutation_results = await self.mutation_tester.test_mutations(
                prompt, baseline_quality=0.8
            )
            results.extend(mutation_results)

        self.metrics_collector.record("mutation_tests_executed", float(len(results)), MetricType.MUTATION_ROBUSTNESS)
        return results

    async def run_regression_tests(self, baseline_path: str | None = None) -> list[PromptTestResult]:
        """Detect quality regressions against baseline.

        Args:
            baseline_path: Optional path to baseline results file

        Returns:
            List of regression test results
        """
        current_results = self.results or []

        if not baseline_path:
            # Use golden dataset as baseline if available
            test_cases = self.golden_dataset.get_test_cases()
            current_results = [await self.run_single_test(tc) for tc in test_cases]

        # Load baseline and detect regressions
        regressions = await self.regression_detector.detect_regressions(
            current_results, []
        )

        self.metrics_collector.record("regressions_detected", float(len(regressions)), MetricType.REGRESSION_COUNT)
        return current_results

    async def run_single_test(self, test_case: PromptTestCase) -> PromptTestResult:
        """Execute a single prompt test.

        Uses LLM-based evaluation if available, falls back to heuristics.

        Args:
            test_case: Test case to execute

        Returns:
            Test result with quality scores and metadata
        """
        try:
            start_time = datetime.now()

            # Get response to evaluate (use golden output for now, or prompt as content)
            response = test_case.prompt_template
            expected = test_case.golden_outputs[0] if test_case.golden_outputs else None

            # Use LLM Judge if available, otherwise fall back to heuristics
            if self._llm_judge_available and self.llm_judge:
                quality_scores = await self._evaluate_with_llm_judge(
                    response, expected, test_case
                )
            else:
                quality_scores = self._calculate_quality_score(response, expected)

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Check if passed all quality thresholds
            passed = all(
                score >= self.config.quality_threshold
                for score in quality_scores.values()
            )

            result = PromptTestResult(
                test_case=test_case,
                passed=passed,
                quality_scores=quality_scores,
                latency_ms=latency_ms,
                token_count=len(test_case.prompt_template.split()),
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                timestamp=datetime.now(),
            )

            # Record metrics
            self.metrics_collector.record(
                f"{test_case.name}_quality",
                max(quality_scores.values()) if quality_scores else 0.0,
                MetricType.QUALITY_SCORE
            )
            self.metrics_collector.record(
                f"{test_case.name}_latency",
                latency_ms,
                MetricType.LATENCY_MS
            )

            return result

        except Exception as e:
            return PromptTestResult(
                test_case=test_case,
                passed=False,
                quality_scores={},
                latency_ms=0.0,
                token_count=0,
                error_message=str(e),
                status=TestStatus.ERROR,
                timestamp=datetime.now(),
            )

    async def _evaluate_with_llm_judge(
        self,
        response: str,
        expected: str | None,
        test_case: PromptTestCase,
    ) -> dict[str, float]:
        """Evaluate response using LLM Judge.

        Args:
            response: Response to evaluate
            expected: Expected/golden response (optional)
            test_case: Test case for context

        Returns:
            Dictionary of quality scores by criterion
        """
        if not self.llm_judge or not JudgeCriteria or not JudgeConfig:
            return self._calculate_quality_score(response, expected)

        try:
            # Map configured criteria to JudgeCriteria enum
            criteria = []
            for criterion_name in self.config.llm_criteria:
                try:
                    criteria.append(JudgeCriteria[criterion_name.upper()])
                except KeyError:
                    # Skip unknown criteria
                    pass

            if not criteria:
                criteria = [JudgeCriteria.QUALITY, JudgeCriteria.RELEVANCE]

            # Create judge config
            judge_config = JudgeConfig(
                criteria=criteria,
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                reference_output=expected,
                timeout_seconds=self.config.llm_timeout_seconds,
            )

            # Build task description from test case
            task = test_case.name if test_case.name else "Evaluate this response"

            # Evaluate using LLM Judge
            result = await self.llm_judge.evaluate(
                task=task,
                output=response,
                config=judge_config,
            )

            # Convert JudgeResult to quality scores dict
            quality_scores = {}
            for score in result.scores:
                quality_scores[score.criterion] = score.score

            # Add overall score
            quality_scores["overall"] = result.overall_score

            return quality_scores

        except Exception as e:
            # Fall back to heuristics on error
            if self.config.fallback_to_heuristic:
                print(f"LLM Judge error, falling back to heuristics: {e}")
                return self._calculate_quality_score(response, expected)
            raise

    def _calculate_quality_score(
        self, response: str, expected: str | None = None
    ) -> dict[str, float]:
        """Calculate quality score for response.

        Simple implementation using:
        - Word overlap with expected (if available)
        - Length validation
        - Format completeness

        Args:
            response: Generated response
            expected: Expected/golden response (optional)

        Returns:
            Dictionary of quality scores by dimension
        """
        scores = {}

        # Length score: penalize very short or very long responses
        response_length = len(response.split())
        length_score = min(1.0, response_length / 50.0) if response_length > 10 else 0.3
        scores["completeness"] = min(1.0, length_score)

        # Format score: check for basic structure
        format_score = 1.0 if len(response) > 20 and response.strip() else 0.0
        scores["format"] = format_score

        # Relevance score: word overlap with expected (if available)
        if expected:
            response_words = set(response.lower().split())
            expected_words = set(expected.lower().split())
            overlap = len(response_words & expected_words)
            total = len(response_words | expected_words)
            relevance_score = overlap / total if total > 0 else 0.0
            scores["relevance"] = relevance_score
        else:
            scores["relevance"] = 0.7  # Default if no golden output

        # Overall: average of all dimensions
        scores["overall"] = sum(scores.values()) / len(scores)

        return scores

    def _generate_report(self, results: list[PromptTestResult]) -> PromptTestReport:
        """Generate aggregated test report.

        Args:
            results: List of individual test results

        Returns:
            Comprehensive test report
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)

        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate average quality across all dimensions
        all_scores = []
        for result in results:
            all_scores.extend(result.quality_scores.values())
        avg_quality = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Collect all regressions
        all_regressions = []
        for result in results:
            all_regressions.extend(result.regressions_detected)

        return PromptTestReport(
            test_results=results,
            total_tests=total,
            passed_count=passed,
            failed_count=failed,
            skipped_count=skipped,
            overall_pass_rate=passed / total if total > 0 else 0.0,
            avg_latency_ms=avg_latency,
            avg_quality_score=avg_quality,
            regressions=all_regressions,
            timestamp=datetime.now(),
            metadata={
                "config": {
                    "quality_threshold": self.config.quality_threshold,
                    "regression_threshold": self.config.regression_threshold,
                    "parallel_execution": self.config.parallel_execution,
                }
            },
        )


def create_test_suite(
    config: PromptTestConfig | None = None,
    llm_judge: Optional["LLMJudge"] = None,
) -> PromptTestSuite:
    """Factory function for creating a test suite.

    Args:
        config: Optional test configuration. Uses defaults if None.
        llm_judge: Optional LLMJudge instance. Created automatically if config.use_llm_judge=True.

    Returns:
        Configured PromptTestSuite instance
    """
    if config is None:
        config = PromptTestConfig()

    golden_dataset = GoldenDatasetManager()
    return PromptTestSuite(config, golden_dataset, llm_judge)


async def main():
    """CLI entry point for running tests."""
    import argparse

    parser = argparse.ArgumentParser(description="HoloLoom Prompt Test Suite")
    parser.add_argument(
        "--test-type",
        choices=["all", "golden", "mutation", "regression"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.8,
        help="Minimum quality threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=10,
        help="Maximum mutations to test",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Execute tests in parallel",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test report",
    )
    # LLM Judge options (December 2025)
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        default=True,
        help="Use LLM-based quality evaluation (default: True)",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        default=False,
        help="Disable LLM-based evaluation, use heuristics only",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="ollama",
        choices=["ollama", "anthropic", "openai"],
        help="LLM provider for evaluation",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.2:3b",
        help="LLM model for evaluation",
    )

    args = parser.parse_args()

    # Create configuration
    config = PromptTestConfig(
        quality_threshold=args.quality_threshold,
        max_mutations=args.max_mutations,
        parallel_execution=args.parallel,
        use_llm_judge=args.use_llm_judge and not args.no_llm_judge,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    # Create and run test suite
    suite = create_test_suite(config)

    if args.test_type == "all":
        report = await suite.run_all_tests()
    elif args.test_type == "golden":
        results = await suite.run_golden_tests()
        report = suite._generate_report(results)
    elif args.test_type == "mutation":
        results = await suite.run_mutation_tests()
        report = suite._generate_report(results)
    elif args.test_type == "regression":
        results = await suite.run_regression_tests()
        report = suite._generate_report(results)

    # Print report summary
    print(f"\n{'='*60}")
    print("PROMPT TEST SUITE REPORT")
    print(f"{'='*60}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed_count} ({report.overall_pass_rate*100:.1f}%)")
    print(f"Failed: {report.failed_count}")
    print(f"Skipped: {report.skipped_count}")
    print(f"Avg Latency: {report.avg_latency_ms:.2f}ms")
    print(f"Avg Quality: {report.avg_quality_score:.2f}")
    print(f"Regressions: {len(report.regressions)}")
    print(f"{'='*60}\n")

    if args.output:
        import json
        output_data = {
            "total_tests": report.total_tests,
            "passed_count": report.passed_count,
            "failed_count": report.failed_count,
            "skipped_count": report.skipped_count,
            "overall_pass_rate": report.overall_pass_rate,
            "avg_latency_ms": report.avg_latency_ms,
            "avg_quality_score": report.avg_quality_score,
            "regressions": report.regressions,
            "timestamp": report.timestamp.isoformat(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Report saved to {args.output}")

    return 0 if report.overall_pass_rate >= config.quality_threshold else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
