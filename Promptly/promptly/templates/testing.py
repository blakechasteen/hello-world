#!/usr/bin/env python3
"""
Template Testing Framework - Test templates with test cases and assertions
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import re


class TestFailureError(Exception):
    """Raised when a template test fails"""
    pass


@dataclass
class TestCase:
    """Represents a single test case for a template"""

    name: str
    description: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[str] = None
    expected_contains: Optional[List[str]] = None
    expected_not_contains: Optional[List[str]] = None
    expected_pattern: Optional[str] = None
    expected_length_min: Optional[int] = None
    expected_length_max: Optional[int] = None
    custom_assertion: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_output(self, actual_output: str) -> List[str]:
        """
        Validate actual output against expectations

        Returns:
            List of error messages (empty if all assertions pass)
        """
        errors = []

        # Exact match
        if self.expected_output is not None:
            if actual_output != self.expected_output:
                errors.append(f"Expected exact output does not match")

        # Contains checks
        if self.expected_contains:
            for text in self.expected_contains:
                if text not in actual_output:
                    errors.append(f"Expected to contain: '{text}'")

        # Not contains checks
        if self.expected_not_contains:
            for text in self.expected_not_contains:
                if text in actual_output:
                    errors.append(f"Expected NOT to contain: '{text}'")

        # Pattern match
        if self.expected_pattern:
            if not re.search(self.expected_pattern, actual_output):
                errors.append(f"Expected to match pattern: {self.expected_pattern}")

        # Length checks
        actual_length = len(actual_output)

        if self.expected_length_min is not None:
            if actual_length < self.expected_length_min:
                errors.append(f"Output too short: {actual_length} < {self.expected_length_min}")

        if self.expected_length_max is not None:
            if actual_length > self.expected_length_max:
                errors.append(f"Output too long: {actual_length} > {self.expected_length_max}")

        # Custom assertion
        if self.custom_assertion:
            try:
                result = self.custom_assertion(actual_output)
                if result is False:
                    errors.append("Custom assertion failed")
                elif isinstance(result, str):
                    errors.append(f"Custom assertion: {result}")
            except Exception as e:
                errors.append(f"Custom assertion error: {e}")

        return errors


@dataclass
class TestResult:
    """Results from running a single test case"""

    test_case: TestCase
    actual_output: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{status}: {self.test_case.name}"]

        if self.test_case.description:
            lines.append(f"  Description: {self.test_case.description}")

        if not self.passed:
            lines.append(f"  Errors:")
            for error in self.errors:
                lines.append(f"    - {error}")

        lines.append(f"  Execution time: {self.execution_time_ms:.2f}ms")

        return '\n'.join(lines)


@dataclass
class TestSuite:
    """Collection of test cases for a template"""

    name: str
    template_name: str
    description: Optional[str] = None
    test_cases: List[TestCase] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_test(self, test_case: TestCase):
        """Add a test case to the suite"""
        self.test_cases.append(test_case)

    def get_test(self, name: str) -> Optional[TestCase]:
        """Get a test case by name"""
        for test in self.test_cases:
            if test.name == name:
                return test
        return None


@dataclass
class TestSuiteResult:
    """Results from running a complete test suite"""

    suite: TestSuite
    results: List[TestResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_tests(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    def __str__(self) -> str:
        lines = [
            f"\nTest Suite: {self.suite.name}",
            f"Template: {self.suite.template_name}",
            f"="* 60,
            f"Total tests: {self.total_tests}",
            f"Passed: {self.passed_tests}",
            f"Failed: {self.failed_tests}",
            f"Pass rate: {self.pass_rate:.1f}%",
            f"Total time: {self.total_time_ms:.2f}ms",
            "=" * 60,
            ""
        ]

        for result in self.results:
            lines.append(str(result))
            lines.append("")

        return '\n'.join(lines)


class TemplateTestRunner:
    """
    Test runner for template testing

    Features:
    - Run individual tests or test suites
    - Setup and teardown hooks
    - Performance tracking
    - Detailed error reporting
    """

    def __init__(self, engine):
        """
        Initialize test runner

        Args:
            engine: TemplateEngine instance for rendering templates
        """
        self.engine = engine

    def run_test(
        self,
        template_content: str,
        test_case: TestCase,
        context: Optional[Dict[str, Any]] = None
    ) -> TestResult:
        """
        Run a single test case

        Args:
            template_content: Template content to render
            test_case: Test case to run
            context: Additional context for rendering

        Returns:
            TestResult with pass/fail status and details
        """
        import time

        # Merge contexts
        variables = {**(context or {}), **test_case.variables}

        # Render template
        start_time = time.time()
        try:
            actual_output = self.engine.render_string(template_content, **variables)
        except Exception as e:
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000

            return TestResult(
                test_case=test_case,
                actual_output="",
                passed=False,
                errors=[f"Template rendering failed: {e}"],
                execution_time_ms=execution_time_ms
            )

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        # Validate output
        errors = test_case.validate_output(actual_output)
        passed = len(errors) == 0

        return TestResult(
            test_case=test_case,
            actual_output=actual_output,
            passed=passed,
            errors=errors,
            execution_time_ms=execution_time_ms
        )

    def run_suite(
        self,
        template_content: str,
        suite: TestSuite,
        context: Optional[Dict[str, Any]] = None,
        fail_fast: bool = False
    ) -> TestSuiteResult:
        """
        Run a complete test suite

        Args:
            template_content: Template content to test
            suite: Test suite to run
            context: Additional context for all tests
            fail_fast: Stop on first failure

        Returns:
            TestSuiteResult with all test results
        """
        import time

        results = []
        start_time = time.time()

        # Setup hook
        if suite.setup:
            try:
                suite.setup()
            except Exception as e:
                print(f"Warning: Setup hook failed: {e}")

        # Run each test
        for test_case in suite.test_cases:
            result = self.run_test(template_content, test_case, context)
            results.append(result)

            if fail_fast and not result.passed:
                break

        # Teardown hook
        if suite.teardown:
            try:
                suite.teardown()
            except Exception as e:
                print(f"Warning: Teardown hook failed: {e}")

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        return TestSuiteResult(
            suite=suite,
            results=results,
            total_time_ms=total_time_ms
        )

    def run_suites(
        self,
        template_content: str,
        suites: List[TestSuite],
        context: Optional[Dict[str, Any]] = None,
        fail_fast: bool = False
    ) -> List[TestSuiteResult]:
        """
        Run multiple test suites

        Args:
            template_content: Template content to test
            suites: List of test suites
            context: Additional context for all tests
            fail_fast: Stop on first suite failure

        Returns:
            List of TestSuiteResult
        """
        results = []

        for suite in suites:
            result = self.run_suite(template_content, suite, context, fail_fast)
            results.append(result)

            if fail_fast and result.failed_tests > 0:
                break

        return results


class TestSuiteBuilder:
    """Builder for creating test suites programmatically"""

    def __init__(self, name: str, template_name: str):
        self.suite = TestSuite(name=name, template_name=template_name)

    def description(self, desc: str) -> 'TestSuiteBuilder':
        """Set suite description"""
        self.suite.description = desc
        return self

    def setup(self, func: Callable) -> 'TestSuiteBuilder':
        """Set setup hook"""
        self.suite.setup = func
        return self

    def teardown(self, func: Callable) -> 'TestSuiteBuilder':
        """Set teardown hook"""
        self.suite.teardown = func
        return self

    def test(
        self,
        name: str,
        variables: Dict[str, Any],
        **expectations
    ) -> 'TestSuiteBuilder':
        """Add a test case"""
        test_case = TestCase(
            name=name,
            variables=variables,
            **expectations
        )
        self.suite.add_test(test_case)
        return self

    def build(self) -> TestSuite:
        """Build and return the test suite"""
        return self.suite


def load_test_suite_from_yaml(filepath: str) -> TestSuite:
    """Load a test suite from a YAML file"""
    import yaml

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    suite = TestSuite(
        name=data.get('name', ''),
        template_name=data.get('template', ''),
        description=data.get('description')
    )

    for test_data in data.get('tests', []):
        test_case = TestCase(
            name=test_data.get('name', ''),
            description=test_data.get('description'),
            variables=test_data.get('variables', {}),
            expected_output=test_data.get('expected_output'),
            expected_contains=test_data.get('expected_contains'),
            expected_not_contains=test_data.get('expected_not_contains'),
            expected_pattern=test_data.get('expected_pattern'),
            expected_length_min=test_data.get('expected_length_min'),
            expected_length_max=test_data.get('expected_length_max')
        )
        suite.add_test(test_case)

    return suite
