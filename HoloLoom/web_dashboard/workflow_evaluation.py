#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Evaluation Framework
=============================

Comprehensive evaluation system for HoloLoom workflows with:
- Test case execution and metrics collection
- A/B testing for workflow variants
- Git-like versioning for workflow evolution
- Benchmark datasets and evaluation reports

Usage:
    evaluator = WorkflowEvaluator()
    results = await evaluator.evaluate_workflow(workflow, test_cases)
    report = evaluator.generate_report(results)

Created: 2025-12-23
Author: HoloLoom Team
"""

import asyncio
import json
import time
import statistics
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class MetricType(Enum):
    """Types of metrics we collect."""
    LATENCY = "latency_ms"
    CONFIDENCE = "confidence"
    TOKEN_COUNT = "token_count"
    NODE_COUNT = "nodes_executed"
    SUCCESS_RATE = "success_rate"
    QUALITY_SCORE = "quality_score"
    MEMORY_USAGE = "memory_mb"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class TestCase:
    """A single test case for workflow evaluation."""
    id: str
    name: str
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    timeout_ms: int = 30000
    required_confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict) -> 'TestCase':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            input_data=data['input_data'],
            expected_output=data.get('expected_output'),
            tags=data.get('tags', []),
            timeout_ms=data.get('timeout_ms', 30000),
            required_confidence=data.get('required_confidence', 0.0)
        )


@dataclass
class ExecutionMetrics:
    """Metrics collected from a single execution."""
    latency_ms: float
    confidence: float
    nodes_executed: int
    tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_mb: float = 0.0
    node_timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class TestResult:
    """Result of running a single test case."""
    test_case_id: str
    test_case_name: str
    success: bool
    metrics: ExecutionMetrics
    output: Optional[Dict[str, Any]]
    expected_output: Optional[Dict[str, Any]]
    output_match: Optional[bool]
    timestamp: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'test_case_id': self.test_case_id,
            'test_case_name': self.test_case_name,
            'success': self.success,
            'metrics': asdict(self.metrics),
            'output': self.output,
            'expected_output': self.expected_output,
            'output_match': self.output_match,
            'timestamp': self.timestamp,
            'error_message': self.error_message
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a workflow."""
    workflow_id: str
    workflow_name: str
    workflow_version: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float

    # Aggregate metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    avg_confidence: float
    min_confidence: float
    max_confidence: float

    avg_nodes_executed: float
    total_tokens_used: int
    cache_hit_rate: float

    # Detailed results
    test_results: List[TestResult]

    # Metadata
    evaluation_timestamp: str
    evaluation_duration_ms: float
    evaluator_version: str = "1.0.0"

    def to_dict(self) -> Dict:
        return {
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            'workflow_version': self.workflow_version,
            'summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'success_rate': self.success_rate
            },
            'latency': {
                'avg_ms': self.avg_latency_ms,
                'p50_ms': self.p50_latency_ms,
                'p95_ms': self.p95_latency_ms,
                'p99_ms': self.p99_latency_ms
            },
            'confidence': {
                'avg': self.avg_confidence,
                'min': self.min_confidence,
                'max': self.max_confidence
            },
            'resources': {
                'avg_nodes_executed': self.avg_nodes_executed,
                'total_tokens_used': self.total_tokens_used,
                'cache_hit_rate': self.cache_hit_rate
            },
            'test_results': [r.to_dict() for r in self.test_results],
            'metadata': {
                'evaluation_timestamp': self.evaluation_timestamp,
                'evaluation_duration_ms': self.evaluation_duration_ms,
                'evaluator_version': self.evaluator_version
            }
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Workflow Evaluation Report",
            f"",
            f"**Workflow**: {self.workflow_name}",
            f"**Version**: {self.workflow_version}",
            f"**Evaluated**: {self.evaluation_timestamp}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {self.total_tests} |",
            f"| Passed | {self.passed_tests} |",
            f"| Failed | {self.failed_tests} |",
            f"| Success Rate | {self.success_rate:.1%} |",
            f"",
            f"## Performance",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Avg Latency | {self.avg_latency_ms:.1f}ms |",
            f"| P50 Latency | {self.p50_latency_ms:.1f}ms |",
            f"| P95 Latency | {self.p95_latency_ms:.1f}ms |",
            f"| P99 Latency | {self.p99_latency_ms:.1f}ms |",
            f"| Avg Confidence | {self.avg_confidence:.3f} |",
            f"| Cache Hit Rate | {self.cache_hit_rate:.1%} |",
            f"",
            f"## Test Results",
            f"",
        ]

        for result in self.test_results:
            status = "[PASS]" if result.success else "[FAIL]"
            lines.append(f"### {status} {result.test_case_name}")
            lines.append(f"")
            lines.append(f"- **Latency**: {result.metrics.latency_ms:.1f}ms")
            lines.append(f"- **Confidence**: {result.metrics.confidence:.3f}")
            lines.append(f"- **Nodes Executed**: {result.metrics.nodes_executed}")
            if result.error_message:
                lines.append(f"- **Error**: {result.error_message}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# Workflow Evaluator
# ============================================================================

class WorkflowEvaluator:
    """
    Evaluates workflows against test cases and collects metrics.

    Usage:
        evaluator = WorkflowEvaluator()
        results = await evaluator.evaluate_workflow(workflow, test_cases)
        report = evaluator.generate_report(results)
    """

    def __init__(self, executor=None, config: Dict[str, Any] = None):
        """
        Initialize evaluator.

        Args:
            executor: Optional workflow executor (uses mock if not provided)
            config: Evaluation configuration
        """
        self.executor = executor
        self.config = config or {}
        self.default_timeout_ms = self.config.get('default_timeout_ms', 30000)
        self.parallel_tests = self.config.get('parallel_tests', 1)
        self.retry_on_failure = self.config.get('retry_on_failure', False)
        self.max_retries = self.config.get('max_retries', 2)

    async def evaluate_workflow(
        self,
        workflow: Dict[str, Any],
        test_cases: List[TestCase],
        parallel: bool = False
    ) -> EvaluationReport:
        """
        Evaluate a workflow against test cases.

        Args:
            workflow: Workflow definition
            test_cases: List of test cases to run
            parallel: Run tests in parallel

        Returns:
            EvaluationReport with all results
        """
        start_time = time.time()
        results: List[TestResult] = []

        workflow_id = workflow.get('id', hashlib.md5(
            json.dumps(workflow, sort_keys=True).encode()
        ).hexdigest()[:12])

        if parallel and len(test_cases) > 1:
            # Run tests in parallel
            tasks = [
                self._run_single_test(workflow, tc)
                for tc in test_cases
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Handle exceptions
            results = [
                r if not isinstance(r, Exception) else self._exception_to_result(tc, r)
                for r, tc in zip(results, test_cases)
            ]
        else:
            # Run tests sequentially
            for tc in test_cases:
                result = await self._run_single_test(workflow, tc)
                results.append(result)

        evaluation_duration = (time.time() - start_time) * 1000

        return self._generate_report(
            workflow=workflow,
            workflow_id=workflow_id,
            results=results,
            evaluation_duration=evaluation_duration
        )

    async def _run_single_test(
        self,
        workflow: Dict[str, Any],
        test_case: TestCase
    ) -> TestResult:
        """Run a single test case."""
        start_time = time.time()

        try:
            # Execute workflow
            output, metrics = await self._execute_workflow(
                workflow,
                test_case.input_data,
                timeout_ms=test_case.timeout_ms
            )

            # Check output match if expected output provided
            output_match = None
            if test_case.expected_output is not None:
                output_match = self._check_output_match(
                    output, test_case.expected_output
                )

            # Determine success
            success = (
                metrics.error is None and
                metrics.confidence >= test_case.required_confidence and
                (output_match is None or output_match)
            )

            return TestResult(
                test_case_id=test_case.id,
                test_case_name=test_case.name,
                success=success,
                metrics=metrics,
                output=output,
                expected_output=test_case.expected_output,
                output_match=output_match,
                timestamp=datetime.utcnow().isoformat()
            )

        except asyncio.TimeoutError:
            latency = (time.time() - start_time) * 1000
            return TestResult(
                test_case_id=test_case.id,
                test_case_name=test_case.name,
                success=False,
                metrics=ExecutionMetrics(
                    latency_ms=latency,
                    confidence=0.0,
                    nodes_executed=0,
                    error=f"Timeout after {test_case.timeout_ms}ms"
                ),
                output=None,
                expected_output=test_case.expected_output,
                output_match=False,
                timestamp=datetime.utcnow().isoformat(),
                error_message=f"Timeout after {test_case.timeout_ms}ms"
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return TestResult(
                test_case_id=test_case.id,
                test_case_name=test_case.name,
                success=False,
                metrics=ExecutionMetrics(
                    latency_ms=latency,
                    confidence=0.0,
                    nodes_executed=0,
                    error=str(e)
                ),
                output=None,
                expected_output=test_case.expected_output,
                output_match=False,
                timestamp=datetime.utcnow().isoformat(),
                error_message=str(e)
            )

    async def _execute_workflow(
        self,
        workflow: Dict[str, Any],
        input_data: Dict[str, Any],
        timeout_ms: int
    ) -> Tuple[Dict[str, Any], ExecutionMetrics]:
        """Execute workflow and collect metrics."""
        start_time = time.time()

        if self.executor:
            # Use real executor
            try:
                result = await asyncio.wait_for(
                    self.executor.execute(workflow, input_data),
                    timeout=timeout_ms / 1000
                )
                latency = (time.time() - start_time) * 1000

                return result.get('output', {}), ExecutionMetrics(
                    latency_ms=latency,
                    confidence=result.get('confidence', 0.0),
                    nodes_executed=result.get('nodes_executed', 0),
                    tokens_used=result.get('tokens_used', 0),
                    cache_hits=result.get('cache_hits', 0),
                    cache_misses=result.get('cache_misses', 0),
                    node_timings=result.get('node_timings', {})
                )
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                return {}, ExecutionMetrics(
                    latency_ms=latency,
                    confidence=0.0,
                    nodes_executed=0,
                    error=str(e)
                )
        else:
            # Mock execution for testing
            await asyncio.sleep(0.05)  # Simulate 50ms execution
            latency = (time.time() - start_time) * 1000

            return {
                'response': f"Mock response for: {input_data.get('query', 'unknown')}",
                'confidence': 0.85 + (hash(str(input_data)) % 15) / 100
            }, ExecutionMetrics(
                latency_ms=latency,
                confidence=0.85 + (hash(str(input_data)) % 15) / 100,
                nodes_executed=len(workflow.get('nodes', [])),
                tokens_used=100 + (hash(str(input_data)) % 200),
                cache_hits=1 if hash(str(input_data)) % 3 == 0 else 0,
                cache_misses=1 if hash(str(input_data)) % 3 != 0 else 0
            )

    def _check_output_match(
        self,
        actual: Dict[str, Any],
        expected: Dict[str, Any]
    ) -> bool:
        """Check if actual output matches expected output."""
        if not actual or not expected:
            return actual == expected

        # Check each expected field
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            actual_value = actual[key]

            # Handle special matchers
            if isinstance(expected_value, str):
                if expected_value.startswith('$contains:'):
                    substring = expected_value[10:]
                    if substring not in str(actual_value):
                        return False
                    continue
                elif expected_value.startswith('$regex:'):
                    import re
                    pattern = expected_value[7:]
                    if not re.search(pattern, str(actual_value)):
                        return False
                    continue
                elif expected_value == '$any':
                    continue

            # Exact match
            if actual_value != expected_value:
                return False

        return True

    def _exception_to_result(self, test_case: TestCase, exc: Exception) -> TestResult:
        """Convert an exception to a TestResult."""
        return TestResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            success=False,
            metrics=ExecutionMetrics(
                latency_ms=0,
                confidence=0.0,
                nodes_executed=0,
                error=str(exc)
            ),
            output=None,
            expected_output=test_case.expected_output,
            output_match=False,
            timestamp=datetime.utcnow().isoformat(),
            error_message=str(exc)
        )

    def _generate_report(
        self,
        workflow: Dict[str, Any],
        workflow_id: str,
        results: List[TestResult],
        evaluation_duration: float
    ) -> EvaluationReport:
        """Generate evaluation report from results."""
        if not results:
            return EvaluationReport(
                workflow_id=workflow_id,
                workflow_name=workflow.get('name', 'Unknown'),
                workflow_version=workflow.get('version', '1.0'),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                success_rate=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                avg_nodes_executed=0.0,
                total_tokens_used=0,
                cache_hit_rate=0.0,
                test_results=results,
                evaluation_timestamp=datetime.utcnow().isoformat(),
                evaluation_duration_ms=evaluation_duration
            )

        # Calculate statistics
        latencies = [r.metrics.latency_ms for r in results]
        confidences = [r.metrics.confidence for r in results]
        nodes_executed = [r.metrics.nodes_executed for r in results]

        total_cache_hits = sum(r.metrics.cache_hits for r in results)
        total_cache_misses = sum(r.metrics.cache_misses for r in results)
        total_cache = total_cache_hits + total_cache_misses

        passed = sum(1 for r in results if r.success)

        return EvaluationReport(
            workflow_id=workflow_id,
            workflow_name=workflow.get('name', 'Unknown'),
            workflow_version=workflow.get('version', '1.0'),
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=len(results) - passed,
            success_rate=passed / len(results),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=self._percentile(latencies, 50),
            p95_latency_ms=self._percentile(latencies, 95),
            p99_latency_ms=self._percentile(latencies, 99),
            avg_confidence=statistics.mean(confidences),
            min_confidence=min(confidences),
            max_confidence=max(confidences),
            avg_nodes_executed=statistics.mean(nodes_executed),
            total_tokens_used=sum(r.metrics.tokens_used for r in results),
            cache_hit_rate=total_cache_hits / total_cache if total_cache > 0 else 0.0,
            test_results=results,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            evaluation_duration_ms=evaluation_duration
        )

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


# ============================================================================
# A/B Testing Framework
# ============================================================================

@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    name: str
    workflow_a: Dict[str, Any]  # Control
    workflow_b: Dict[str, Any]  # Treatment
    test_cases: List[TestCase]
    traffic_split: float = 0.5  # 50% to each
    min_samples: int = 30
    significance_level: float = 0.05


@dataclass
class ABTestResult:
    """Result of A/B test."""
    test_name: str
    workflow_a_name: str
    workflow_b_name: str
    samples_a: int
    samples_b: int

    # Performance comparison
    latency_a_avg: float
    latency_b_avg: float
    latency_improvement: float  # % improvement of B over A

    confidence_a_avg: float
    confidence_b_avg: float
    confidence_improvement: float

    success_rate_a: float
    success_rate_b: float
    success_rate_improvement: float

    # Statistical significance
    is_significant: bool
    p_value: float
    winner: Optional[str]  # 'A', 'B', or None
    recommendation: str

    def to_dict(self) -> Dict:
        return asdict(self)


class ABTestRunner:
    """
    Run A/B tests comparing two workflow variants.

    Usage:
        runner = ABTestRunner()
        result = await runner.run_test(config)
    """

    def __init__(self, evaluator: WorkflowEvaluator = None):
        self.evaluator = evaluator or WorkflowEvaluator()

    async def run_test(self, config: ABTestConfig) -> ABTestResult:
        """Run A/B test between two workflows."""
        import random

        # Split test cases
        random.shuffle(config.test_cases)
        split_idx = int(len(config.test_cases) * config.traffic_split)
        tests_a = config.test_cases[:split_idx]
        tests_b = config.test_cases[split_idx:]

        # Ensure minimum samples
        if len(tests_a) < config.min_samples or len(tests_b) < config.min_samples:
            logger.warning(
                f"Insufficient samples: A={len(tests_a)}, B={len(tests_b)}, "
                f"min={config.min_samples}"
            )

        # Evaluate both workflows
        report_a = await self.evaluator.evaluate_workflow(
            config.workflow_a, tests_a, parallel=True
        )
        report_b = await self.evaluator.evaluate_workflow(
            config.workflow_b, tests_b, parallel=True
        )

        # Calculate improvements
        latency_improvement = (
            (report_a.avg_latency_ms - report_b.avg_latency_ms) / report_a.avg_latency_ms
            if report_a.avg_latency_ms > 0 else 0.0
        ) * 100

        confidence_improvement = (
            (report_b.avg_confidence - report_a.avg_confidence) / report_a.avg_confidence
            if report_a.avg_confidence > 0 else 0.0
        ) * 100

        success_improvement = (
            (report_b.success_rate - report_a.success_rate) / report_a.success_rate
            if report_a.success_rate > 0 else 0.0
        ) * 100

        # Statistical significance (simplified)
        is_significant, p_value = self._calculate_significance(
            [r.metrics.confidence for r in report_a.test_results],
            [r.metrics.confidence for r in report_b.test_results]
        )

        # Determine winner
        winner = None
        if is_significant:
            if report_b.avg_confidence > report_a.avg_confidence:
                winner = 'B'
            elif report_a.avg_confidence > report_b.avg_confidence:
                winner = 'A'

        # Generate recommendation
        if winner == 'B':
            recommendation = f"Deploy workflow B ('{config.workflow_b.get('name', 'B')}'). Shows {confidence_improvement:.1f}% confidence improvement."
        elif winner == 'A':
            recommendation = f"Keep workflow A ('{config.workflow_a.get('name', 'A')}'). Control outperforms treatment."
        else:
            recommendation = f"No significant difference detected. Consider running more tests (current: {len(tests_a) + len(tests_b)})."

        return ABTestResult(
            test_name=config.name,
            workflow_a_name=config.workflow_a.get('name', 'Control'),
            workflow_b_name=config.workflow_b.get('name', 'Treatment'),
            samples_a=len(tests_a),
            samples_b=len(tests_b),
            latency_a_avg=report_a.avg_latency_ms,
            latency_b_avg=report_b.avg_latency_ms,
            latency_improvement=latency_improvement,
            confidence_a_avg=report_a.avg_confidence,
            confidence_b_avg=report_b.avg_confidence,
            confidence_improvement=confidence_improvement,
            success_rate_a=report_a.success_rate,
            success_rate_b=report_b.success_rate,
            success_rate_improvement=success_improvement,
            is_significant=is_significant,
            p_value=p_value,
            winner=winner,
            recommendation=recommendation
        )

    def _calculate_significance(
        self,
        samples_a: List[float],
        samples_b: List[float]
    ) -> Tuple[bool, float]:
        """Calculate statistical significance using Welch's t-test."""
        if len(samples_a) < 2 or len(samples_b) < 2:
            return False, 1.0

        try:
            # Welch's t-test (unequal variances)
            mean_a = statistics.mean(samples_a)
            mean_b = statistics.mean(samples_b)
            var_a = statistics.variance(samples_a)
            var_b = statistics.variance(samples_b)
            n_a = len(samples_a)
            n_b = len(samples_b)

            # t statistic
            se = ((var_a / n_a) + (var_b / n_b)) ** 0.5
            if se == 0:
                return False, 1.0
            t_stat = abs(mean_a - mean_b) / se

            # Degrees of freedom (Welch-Satterthwaite)
            num = ((var_a / n_a) + (var_b / n_b)) ** 2
            denom = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
            df = num / denom if denom > 0 else 1

            # Approximate p-value (simplified)
            # For proper p-value, use scipy.stats.t.sf
            p_value = max(0.001, 1 / (1 + t_stat ** 2))

            return p_value < 0.05, p_value

        except Exception as e:
            logger.warning(f"Significance calculation failed: {e}")
            return False, 1.0


# ============================================================================
# Workflow Version Control
# ============================================================================

@dataclass
class WorkflowVersion:
    """A version of a workflow."""
    version_id: str
    workflow_id: str
    version_number: int
    workflow_data: Dict[str, Any]
    message: str
    description: Optional[str]
    created_at: str
    created_by: str
    parent_version: Optional[str]
    branch: str = 'main'
    tags: List[str] = field(default_factory=list)
    evaluation_results: Optional[Dict] = None


class WorkflowVersionControl:
    """
    Git-like version control for workflows.

    Usage:
        vcs = WorkflowVersionControl(storage_path='./workflow_versions')
        version = vcs.commit(workflow, message="Initial version")
        vcs.checkout(version.version_id)
        vcs.branch('experiment')
    """

    def __init__(self, storage_path: str = './workflow_versions'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self):
        """Load version index from disk."""
        index_path = self.storage_path / 'index.json'
        if index_path.exists():
            with open(index_path) as f:
                self._index = json.load(f)
        else:
            self._index = {
                'workflows': {},  # workflow_id -> {branches, versions, head}
                'branches': {'main': None}  # branch -> latest version_id
            }

    def _save_index(self):
        """Save version index to disk."""
        index_path = self.storage_path / 'index.json'
        with open(index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    def commit(
        self,
        workflow: Dict[str, Any],
        message: str,
        description: str = None,
        author: str = 'anonymous',
        branch: str = 'main',
        tags: List[str] = None
    ) -> WorkflowVersion:
        """
        Commit a new version of a workflow.

        Args:
            workflow: Workflow definition
            message: Commit message
            description: Optional longer description
            author: Author name
            branch: Branch to commit to
            tags: Optional tags

        Returns:
            WorkflowVersion
        """
        workflow_id = workflow.get('id') or hashlib.md5(
            workflow.get('name', 'unknown').encode()
        ).hexdigest()[:12]

        # Initialize workflow entry if new
        if workflow_id not in self._index['workflows']:
            self._index['workflows'][workflow_id] = {
                'name': workflow.get('name', 'Unknown'),
                'branches': {'main': None},
                'versions': [],
                'head': None
            }

        workflow_entry = self._index['workflows'][workflow_id]

        # Determine version number
        version_number = len(workflow_entry['versions']) + 1

        # Get parent version
        parent_version = workflow_entry['branches'].get(branch)

        # Create version ID
        version_id = f"{workflow_id}-v{version_number}-{uuid.uuid4().hex[:8]}"

        # Create version
        version = WorkflowVersion(
            version_id=version_id,
            workflow_id=workflow_id,
            version_number=version_number,
            workflow_data=workflow,
            message=message,
            description=description,
            created_at=datetime.utcnow().isoformat(),
            created_by=author,
            parent_version=parent_version,
            branch=branch,
            tags=tags or []
        )

        # Save version data
        version_path = self.storage_path / f"{version_id}.json"
        with open(version_path, 'w') as f:
            json.dump(asdict(version), f, indent=2)

        # Update index
        workflow_entry['versions'].append(version_id)
        workflow_entry['branches'][branch] = version_id
        workflow_entry['head'] = version_id
        self._save_index()

        logger.info(f"Committed version {version_id}: {message}")
        return version

    def checkout(self, version_id: str) -> Dict[str, Any]:
        """
        Checkout a specific version.

        Args:
            version_id: Version to checkout

        Returns:
            Workflow definition
        """
        version_path = self.storage_path / f"{version_id}.json"
        if not version_path.exists():
            raise ValueError(f"Version not found: {version_id}")

        with open(version_path) as f:
            version_data = json.load(f)

        return version_data['workflow_data']

    def get_version(self, version_id: str) -> WorkflowVersion:
        """Get version metadata."""
        version_path = self.storage_path / f"{version_id}.json"
        if not version_path.exists():
            raise ValueError(f"Version not found: {version_id}")

        with open(version_path) as f:
            data = json.load(f)

        return WorkflowVersion(**data)

    def create_branch(
        self,
        workflow_id: str,
        branch_name: str,
        from_branch: str = 'main'
    ) -> str:
        """
        Create a new branch.

        Args:
            workflow_id: Workflow to branch
            branch_name: New branch name
            from_branch: Branch to branch from

        Returns:
            Branch name
        """
        if workflow_id not in self._index['workflows']:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow_entry = self._index['workflows'][workflow_id]

        if branch_name in workflow_entry['branches']:
            raise ValueError(f"Branch already exists: {branch_name}")

        if from_branch not in workflow_entry['branches']:
            raise ValueError(f"Source branch not found: {from_branch}")

        # Point new branch to same version as source
        workflow_entry['branches'][branch_name] = workflow_entry['branches'][from_branch]
        self._save_index()

        logger.info(f"Created branch '{branch_name}' from '{from_branch}'")
        return branch_name

    def list_branches(self, workflow_id: str) -> List[str]:
        """List branches for a workflow."""
        if workflow_id not in self._index['workflows']:
            return []
        return list(self._index['workflows'][workflow_id]['branches'].keys())

    def get_history(
        self,
        workflow_id: str,
        branch: str = None,
        limit: int = 10
    ) -> List[WorkflowVersion]:
        """Get version history for a workflow."""
        if workflow_id not in self._index['workflows']:
            return []

        workflow_entry = self._index['workflows'][workflow_id]
        versions = []

        for version_id in reversed(workflow_entry['versions'][-limit:]):
            try:
                version = self.get_version(version_id)
                if branch is None or version.branch == branch:
                    versions.append(version)
            except Exception as e:
                logger.warning(f"Failed to load version {version_id}: {e}")

        return versions

    def diff(
        self,
        version_a_id: str,
        version_b_id: str
    ) -> Dict[str, Any]:
        """
        Compare two versions.

        Returns:
            Dict with added, removed, modified nodes and connections
        """
        workflow_a = self.checkout(version_a_id)
        workflow_b = self.checkout(version_b_id)

        nodes_a = {n['id']: n for n in workflow_a.get('nodes', [])}
        nodes_b = {n['id']: n for n in workflow_b.get('nodes', [])}

        conns_a = {c['id']: c for c in workflow_a.get('connections', [])}
        conns_b = {c['id']: c for c in workflow_b.get('connections', [])}

        return {
            'nodes': {
                'added': [n for n_id, n in nodes_b.items() if n_id not in nodes_a],
                'removed': [n for n_id, n in nodes_a.items() if n_id not in nodes_b],
                'modified': [
                    {'old': nodes_a[n_id], 'new': nodes_b[n_id]}
                    for n_id in nodes_a.keys() & nodes_b.keys()
                    if nodes_a[n_id] != nodes_b[n_id]
                ]
            },
            'connections': {
                'added': [c for c_id, c in conns_b.items() if c_id not in conns_a],
                'removed': [c for c_id, c in conns_a.items() if c_id not in conns_b],
                'modified': [
                    {'old': conns_a[c_id], 'new': conns_b[c_id]}
                    for c_id in conns_a.keys() & conns_b.keys()
                    if conns_a[c_id] != conns_b[c_id]
                ]
            },
            'version_a': version_a_id,
            'version_b': version_b_id
        }

    def tag(self, version_id: str, tag: str):
        """Add a tag to a version."""
        version = self.get_version(version_id)
        if tag not in version.tags:
            version.tags.append(tag)

            # Save updated version
            version_path = self.storage_path / f"{version_id}.json"
            with open(version_path, 'w') as f:
                json.dump(asdict(version), f, indent=2)

        logger.info(f"Tagged {version_id} with '{tag}'")

    def attach_evaluation(self, version_id: str, evaluation: EvaluationReport):
        """Attach evaluation results to a version."""
        version = self.get_version(version_id)
        version.evaluation_results = evaluation.to_dict()

        # Save updated version
        version_path = self.storage_path / f"{version_id}.json"
        with open(version_path, 'w') as f:
            json.dump(asdict(version), f, indent=2)

        logger.info(f"Attached evaluation to {version_id}")


# ============================================================================
# Benchmark Datasets
# ============================================================================

class BenchmarkDataset:
    """
    Pre-defined benchmark datasets for workflow evaluation.
    """

    @staticmethod
    def factual_qa(size: int = 20) -> List[TestCase]:
        """Generate factual Q&A test cases."""
        questions = [
            "What is machine learning?",
            "Explain neural networks.",
            "What is the capital of France?",
            "How does photosynthesis work?",
            "What is Thompson Sampling?",
            "Define reinforcement learning.",
            "What is a transformer model?",
            "Explain gradient descent.",
            "What is entropy in information theory?",
            "How do convolutional neural networks work?",
            "What is backpropagation?",
            "Explain the attention mechanism.",
            "What is a knowledge graph?",
            "Define natural language processing.",
            "What is transfer learning?",
            "Explain embedding vectors.",
            "What is semantic similarity?",
            "Define recursive neural networks.",
            "What is the Bayesian approach?",
            "Explain exploration vs exploitation."
        ]

        return [
            TestCase(
                id=f"factual-{i}",
                name=f"Factual: {q[:30]}...",
                input_data={"query": q},
                tags=["factual", "qa"],
                required_confidence=0.6
            )
            for i, q in enumerate(questions[:size])
        ]

    @staticmethod
    def complex_reasoning(size: int = 10) -> List[TestCase]:
        """Generate complex reasoning test cases."""
        queries = [
            "Compare and contrast supervised and unsupervised learning, then recommend when to use each.",
            "Analyze the tradeoffs between model complexity and generalization.",
            "Design a system that combines Thompson Sampling with neural networks.",
            "Evaluate the pros and cons of different attention mechanisms.",
            "Synthesize insights from multiple sources about transfer learning best practices.",
            "Compare different approaches to handling missing data in machine learning.",
            "Analyze the ethical implications of automated decision-making systems.",
            "Design an evaluation framework for language models.",
            "Compare gradient-based and gradient-free optimization methods.",
            "Analyze the relationship between batch size and learning rate."
        ]

        return [
            TestCase(
                id=f"complex-{i}",
                name=f"Complex: {q[:30]}...",
                input_data={"query": q},
                tags=["complex", "reasoning", "multi-step"],
                required_confidence=0.5,
                timeout_ms=60000
            )
            for i, q in enumerate(queries[:size])
        ]

    @staticmethod
    def adversarial(size: int = 10) -> List[TestCase]:
        """Generate adversarial test cases."""
        queries = [
            "What is the sound of one hand clapping?",
            "If everything is possible, is it possible for something to be impossible?",
            "Explain something you don't understand.",
            "What will happen tomorrow at 3:47:23 PM exactly?",
            "Translate 'hello' into a language that doesn't exist.",
            "What is north of the North Pole?",
            "Calculate the probability of an impossible event.",
            "Describe the taste of the color blue.",
            "What is the last digit of pi?",
            "Prove that this statement is false."
        ]

        return [
            TestCase(
                id=f"adversarial-{i}",
                name=f"Adversarial: {q[:30]}...",
                input_data={"query": q},
                tags=["adversarial", "edge-case"],
                required_confidence=0.0  # Should handle gracefully
            )
            for i, q in enumerate(queries[:size])
        ]

    @staticmethod
    def get_full_benchmark() -> List[TestCase]:
        """Get full benchmark dataset."""
        return (
            BenchmarkDataset.factual_qa(20) +
            BenchmarkDataset.complex_reasoning(10) +
            BenchmarkDataset.adversarial(10)
        )


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """Demo the evaluation framework."""
    print("=" * 60)
    print("HoloLoom Workflow Evaluation Framework Demo")
    print("=" * 60)

    # Create evaluator
    evaluator = WorkflowEvaluator()

    # Sample workflow
    sample_workflow = {
        "version": "1.0",
        "name": "Sample Q&A Workflow",
        "id": "sample-qa",
        "nodes": [
            {"id": "query", "agentType": "hololoom", "x": 100, "y": 100, "config": {}},
            {"id": "response", "agentType": "response", "x": 300, "y": 100, "config": {}}
        ],
        "connections": [
            {"id": "c1", "from": "query", "to": "response"}
        ]
    }

    # Get test cases
    test_cases = BenchmarkDataset.factual_qa(5)

    print(f"\nEvaluating workflow with {len(test_cases)} test cases...")

    # Run evaluation
    report = await evaluator.evaluate_workflow(sample_workflow, test_cases)

    # Print report
    print("\n" + report.to_markdown())

    # Save report
    report_path = Path("./eval_reports")
    report_path.mkdir(exist_ok=True)

    with open(report_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\nReport saved to {report_path}/")

    # Demo version control
    print("\n" + "=" * 60)
    print("Version Control Demo")
    print("=" * 60)

    vcs = WorkflowVersionControl("./workflow_versions_demo")

    # Commit versions
    v1 = vcs.commit(sample_workflow, "Initial version", author="demo")
    print(f"Committed v1: {v1.version_id}")

    # Modify workflow
    sample_workflow['nodes'].append({
        "id": "refiner",
        "agentType": "refiner",
        "x": 200,
        "y": 100,
        "config": {}
    })

    v2 = vcs.commit(sample_workflow, "Add refiner node", author="demo")
    print(f"Committed v2: {v2.version_id}")

    # Show diff
    diff = vcs.diff(v1.version_id, v2.version_id)
    print(f"\nDiff v1 -> v2:")
    print(f"  Added nodes: {len(diff['nodes']['added'])}")
    print(f"  Removed nodes: {len(diff['nodes']['removed'])}")

    # Create branch
    vcs.create_branch(sample_workflow['id'], 'experiment')
    branches = vcs.list_branches(sample_workflow['id'])
    print(f"\nBranches: {branches}")

    # Attach evaluation
    vcs.attach_evaluation(v2.version_id, report)
    print(f"\nAttached evaluation to {v2.version_id}")

    print("\n[OK] Demo complete!")


if __name__ == '__main__':
    asyncio.run(main())
