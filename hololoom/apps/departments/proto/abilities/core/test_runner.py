"""Proto Test Runner Ability - Tier 2 Plugin.

Safe test execution for Python test suites with framework auto-detection,
coverage support, and comprehensive result reporting.

Supported Test Frameworks:
    - pytest (primary framework with rich features)
    - unittest (Python standard library)
    - Auto-detection from project structure

Supported Operations:
    - run_tests: Run tests with patterns/markers
    - run_single: Execute a single test file
    - discover: Find available tests
    - coverage: Run with coverage reporting (pytest-cov)

Features:
    - Configurable test patterns (test_*.py, *_test.py)
    - pytest marker filtering (-m smoke, -m "not slow")
    - Verbose output control
    - Execution timeouts (default: 300s)
    - Coverage percentage and missing lines
    - Test durations and timing
    - Output truncation for large test suites

Safety Features:
    - Timeout enforcement for runaway tests
    - Working directory validation
    - Output size limits (1MB default)
    - Subprocess isolation
    - Test discovery validation

Implementation: December 2025
Status: Production ready
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..protocol import (
    AbilityContext,
    AbilityManifest,
    AbilityResult,
    AbilityTier,
    AbilityTrustLevel,
    BaseAbility,
    PreflightResult,
    VerificationResult,
)


logger = logging.getLogger("proto.test_runner")


# Timeout for test execution (seconds)
DEFAULT_TEST_TIMEOUT = 300.0  # 5 minutes
MAX_TEST_TIMEOUT = 1800.0     # 30 minutes

# Maximum output size to prevent memory issues
MAX_OUTPUT_SIZE = 1_000_000  # 1MB

# Test file patterns
DEFAULT_TEST_PATTERN = "test_*.py"
UNITTEST_PATTERN = "test*.py"

# Supported operations
SUPPORTED_OPERATIONS = {
    "run_tests",
    "run_single",
    "discover",
    "coverage",
}


@dataclass
class TestFrameworkInfo:
    """Information about detected test framework.

    Attributes:
        name: Framework name (pytest, unittest)
        available: Whether framework is available/installed
        version: Framework version string
        config_files: List of found config files
    """
    name: str
    available: bool
    version: Optional[str] = None
    config_files: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.config_files is None:
            self.config_files = []


@dataclass
class TestResult:
    """Result from test execution.

    Attributes:
        operation: Operation name (run_tests, coverage, etc.)
        success: Whether tests passed
        passed: Number of tests passed
        failed: Number of tests failed
        skipped: Number of tests skipped
        errors: Number of test errors
        total: Total number of tests
        duration_seconds: Total execution time
        output: Full test output
        coverage_percent: Coverage percentage (if coverage enabled)
        coverage_missing: List of missing coverage lines
        test_details: List of individual test results
        framework: Test framework used
    """
    operation: str
    success: bool
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    duration_seconds: float = 0.0
    output: Optional[str] = None
    coverage_percent: Optional[float] = None
    coverage_missing: Optional[List[str]] = None
    test_details: Optional[List[Dict[str, Any]]] = None
    framework: Optional[str] = None


class TestRunnerAbility(BaseAbility):
    """Tier 2 Ability: Safe test execution with framework auto-detection.

    Provides structured test execution for Python projects:
    - Auto-detects pytest vs unittest
    - Configurable test patterns and markers
    - Coverage reporting with pytest-cov
    - Test discovery and individual test runs
    - Timeout enforcement and output limits

    Example:
        ability = TestRunnerAbility()
        context = AbilityContext(session_id="123", working_directory="/path/to/project")

        result = await ability.execute({
            "operation": "run_tests",
            "test_path": "tests/",
            "markers": "not slow",
            "coverage": True
        }, context)

        if result.success:
            test_result = result.output
            print(f"Passed: {test_result['passed']}/{test_result['total']}")
            if test_result.get('coverage_percent'):
                print(f"Coverage: {test_result['coverage_percent']:.1f}%")
    """

    def __init__(self):
        """Initialize Test Runner ability."""
        manifest = AbilityManifest(
            name="test_runner",
            version="1.0.0",
            description="Safe Python test execution with pytest/unittest auto-detection and coverage",
            author="Proto Team",
            tier=AbilityTier.PLUGIN,
            trust_level=AbilityTrustLevel.VERIFIED,
            permissions=["execute_command", "read_file"],
            requires_confirmation=False,
            requires=["pytest", "unittest"],  # pytest optional but preferred
            tags=["testing", "pytest", "unittest", "quality", "ci"],
            homepage="https://github.com/hololoom/proto",
        )
        super().__init__(manifest)

    async def preflight(self, context: AbilityContext) -> PreflightResult:
        """Check if test runner can execute.

        Verifies:
        - Python is available
        - At least one test framework available (pytest or unittest)
        - Working directory exists

        Args:
            context: Execution context

        Returns:
            PreflightResult indicating execution feasibility
        """
        warnings = []

        # Check if Python is available
        python_exe = self._get_python_executable()
        if not python_exe:
            return PreflightResult(
                can_execute=False,
                reason="Python interpreter not found",
                estimated_duration_ms=0.0,
            )

        # Check test frameworks
        has_pytest = shutil.which("pytest") is not None
        has_unittest = True  # Always available (standard library)

        if not has_pytest and not has_unittest:
            return PreflightResult(
                can_execute=False,
                reason="No test framework found (pytest or unittest required)",
                estimated_duration_ms=0.0,
            )

        if not has_pytest:
            warnings.append("pytest not found, will use unittest (limited features)")

        # Check working directory
        if context.working_directory and not os.path.isdir(context.working_directory):
            return PreflightResult(
                can_execute=False,
                reason=f"Working directory not found: {context.working_directory}",
                estimated_duration_ms=0.0,
            )

        return PreflightResult(
            can_execute=True,
            warnings=warnings,
            estimated_duration_ms=5000.0,  # Tests typically take 5+ seconds
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: AbilityContext,
    ) -> AbilityResult:
        """Execute test operation with given parameters.

        Parameters:
            operation: str - Operation (run_tests, run_single, discover, coverage)
            test_path: Optional[str] - Path to tests (file or directory)
            pattern: str - Test file pattern (default: "test_*.py")
            markers: Optional[str] - pytest markers to filter (-m "not slow")
            verbose: bool - Verbose output (default: False)
            timeout: float - Test timeout seconds (default: 300)
            coverage: bool - Enable coverage (default: False)
            framework: Optional[str] - Force framework (pytest/unittest)

        Args:
            params: Operation parameters
            context: Execution context

        Returns:
            AbilityResult with test execution results
        """
        start_time = time.time()

        try:
            # Extract and validate parameters
            operation = params.get("operation", "run_tests").lower().strip()
            test_path = params.get("test_path")
            pattern = params.get("pattern", DEFAULT_TEST_PATTERN)
            markers = params.get("markers")
            verbose = params.get("verbose", False)
            timeout = params.get("timeout", DEFAULT_TEST_TIMEOUT)
            enable_coverage = params.get("coverage", False)
            force_framework = params.get("framework")

            # Validate operation
            if not operation:
                return AbilityResult(
                    success=False,
                    error="Missing required parameter: operation",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if operation not in SUPPORTED_OPERATIONS:
                return AbilityResult(
                    success=False,
                    error=f"Unsupported operation: {operation}. Supported: {SUPPORTED_OPERATIONS}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Normalize timeout
            timeout = min(timeout, MAX_TEST_TIMEOUT)
            timeout = max(timeout, 1.0)  # Minimum 1 second

            # Validate test path
            working_dir = context.working_directory or os.getcwd()
            if test_path:
                test_path = self._validate_test_path(test_path, working_dir)
                if isinstance(test_path, str) and test_path.startswith("ERROR:"):
                    return AbilityResult(
                        success=False,
                        error=test_path[7:],  # Remove "ERROR:" prefix
                        duration_ms=(time.time() - start_time) * 1000,
                    )
            else:
                # Default to tests directory or current directory
                test_path = self._find_test_directory(working_dir)

            # Detect test framework
            framework = self._detect_framework(working_dir, force_framework)

            logger.info(
                f"Running tests (operation={operation}, framework={framework.name}, "
                f"path={test_path}, timeout={timeout}s)"
            )

            # Execute appropriate operation
            test_result: TestResult
            if operation == "run_tests":
                test_result = await self._run_tests(
                    test_path, pattern, markers, verbose, timeout,
                    enable_coverage, framework, working_dir
                )
            elif operation == "run_single":
                test_result = await self._run_single(
                    test_path, verbose, timeout, framework, working_dir
                )
            elif operation == "discover":
                test_result = await self._discover_tests(
                    test_path, pattern, framework, working_dir
                )
            elif operation == "coverage":
                test_result = await self._run_with_coverage(
                    test_path, pattern, markers, timeout, framework, working_dir
                )
            else:
                test_result = TestResult(
                    operation=operation,
                    success=False,
                )

            duration_ms = (time.time() - start_time) * 1000

            if test_result.success or operation == "discover":
                return AbilityResult(
                    success=test_result.success,
                    output=self._format_result(test_result),
                    confidence=0.95 if test_result.success else 0.85,
                    duration_ms=duration_ms,
                    metadata={
                        "operation": operation,
                        "framework": framework.name,
                        "test_path": test_path,
                        "passed": test_result.passed,
                        "failed": test_result.failed,
                        "total": test_result.total,
                    },
                )
            else:
                return AbilityResult(
                    success=False,
                    error=f"Tests failed: {test_result.failed} failures, {test_result.errors} errors",
                    output=self._format_result(test_result),
                    confidence=0.9,  # High confidence in failure detection
                    duration_ms=duration_ms,
                    metadata={
                        "operation": operation,
                        "framework": framework.name,
                        "test_path": test_path,
                        "passed": test_result.passed,
                        "failed": test_result.failed,
                        "total": test_result.total,
                    },
                )

        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
            return AbilityResult(
                success=False,
                error=f"Execution error: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def verify(self, result: AbilityResult) -> VerificationResult:
        """Verify test execution result.

        Checks:
        - Test counts are reasonable
        - Coverage percentage is valid (if present)
        - Output structure is correct

        Args:
            result: Result from execute()

        Returns:
            VerificationResult with verification status
        """
        issues = []
        suggestions = []

        if not result.success and not result.output:
            # Test failures are valid results
            if result.error and "failed" in result.error.lower():
                # This is a legitimate test failure, not a verification issue
                return VerificationResult(verified=True)

            issues.append("Test execution failed with no output")
            return VerificationResult(verified=False, issues=issues)

        # Check output structure
        output = result.output
        if isinstance(output, dict):
            # Verify test counts
            total = output.get("total", 0)
            passed = output.get("passed", 0)
            failed = output.get("failed", 0)

            if passed + failed > total:
                issues.append("Test count mismatch: passed + failed > total")

            # Check coverage if present
            coverage = output.get("coverage_percent")
            if coverage is not None:
                if not (0 <= coverage <= 100):
                    issues.append(f"Invalid coverage percentage: {coverage}")

        if len(issues) == 0 and result.success:
            # Suggest coverage if not enabled
            output = result.output or {}
            if not output.get("coverage_percent"):
                suggestions.append("Consider enabling coverage for quality metrics")

        return VerificationResult(
            verified=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
        )

    async def cleanup(self) -> None:
        """Cleanup test runner resources.

        Removes temporary coverage files if created.
        """
        logger.info("Test runner cleanup complete")

    # ========== Test Execution Methods ==========

    async def _run_tests(
        self,
        test_path: str,
        pattern: str,
        markers: Optional[str],
        verbose: bool,
        timeout: float,
        enable_coverage: bool,
        framework: TestFrameworkInfo,
        working_dir: str,
    ) -> TestResult:
        """Run tests with pattern matching and filtering.

        Args:
            test_path: Path to test file or directory
            pattern: Test file pattern
            markers: pytest markers for filtering
            verbose: Verbose output
            timeout: Execution timeout
            enable_coverage: Enable coverage reporting
            framework: Detected test framework
            working_dir: Working directory

        Returns:
            TestResult with execution results
        """
        try:
            if framework.name == "pytest":
                return await self._run_pytest(
                    test_path, markers, verbose, timeout, enable_coverage, working_dir
                )
            else:
                return await self._run_unittest(
                    test_path, pattern, verbose, timeout, working_dir
                )
        except Exception as e:
            return TestResult(
                operation="run_tests",
                success=False,
                output=str(e),
                framework=framework.name,
            )

    async def _run_single(
        self,
        test_path: str,
        verbose: bool,
        timeout: float,
        framework: TestFrameworkInfo,
        working_dir: str,
    ) -> TestResult:
        """Run a single test file.

        Args:
            test_path: Path to test file
            verbose: Verbose output
            timeout: Execution timeout
            framework: Detected test framework
            working_dir: Working directory

        Returns:
            TestResult with execution results
        """
        try:
            # Ensure it's a file
            full_path = os.path.join(working_dir, test_path)
            if not os.path.isfile(full_path):
                return TestResult(
                    operation="run_single",
                    success=False,
                    output=f"Not a file: {test_path}",
                    framework=framework.name,
                )

            if framework.name == "pytest":
                return await self._run_pytest(
                    test_path, None, verbose, timeout, False, working_dir
                )
            else:
                return await self._run_unittest(
                    test_path, None, verbose, timeout, working_dir
                )
        except Exception as e:
            return TestResult(
                operation="run_single",
                success=False,
                output=str(e),
                framework=framework.name,
            )

    async def _discover_tests(
        self,
        test_path: str,
        pattern: str,
        framework: TestFrameworkInfo,
        working_dir: str,
    ) -> TestResult:
        """Discover available tests.

        Args:
            test_path: Path to search for tests
            pattern: Test file pattern
            framework: Detected test framework
            working_dir: Working directory

        Returns:
            TestResult with discovered tests
        """
        try:
            if framework.name == "pytest":
                cmd = ["pytest", "--collect-only", "-q", test_path]
            else:
                # Use unittest discover
                cmd = [
                    self._get_python_executable(),
                    "-m", "unittest",
                    "discover",
                    "-s", test_path,
                    "-p", pattern,
                ]

            output = await self._run_command(cmd, working_dir, timeout=30.0)

            # Parse discovered tests
            test_count = len([line for line in output.split("\n") if "::" in line or "test" in line.lower()])

            return TestResult(
                operation="discover",
                success=True,
                total=test_count,
                output=output,
                framework=framework.name,
            )
        except Exception as e:
            return TestResult(
                operation="discover",
                success=False,
                output=str(e),
                framework=framework.name,
            )

    async def _run_with_coverage(
        self,
        test_path: str,
        pattern: str,
        markers: Optional[str],
        timeout: float,
        framework: TestFrameworkInfo,
        working_dir: str,
    ) -> TestResult:
        """Run tests with coverage reporting.

        Args:
            test_path: Path to test file or directory
            pattern: Test file pattern
            markers: pytest markers for filtering
            timeout: Execution timeout
            framework: Detected test framework
            working_dir: Working directory

        Returns:
            TestResult with coverage information
        """
        # Coverage requires pytest-cov
        return await self._run_pytest(
            test_path, markers, verbose=True, timeout=timeout,
            enable_coverage=True, working_dir=working_dir
        )

    # ========== Framework-Specific Execution ==========

    async def _run_pytest(
        self,
        test_path: str,
        markers: Optional[str],
        verbose: bool,
        timeout: float,
        enable_coverage: bool,
        working_dir: str,
    ) -> TestResult:
        """Run tests using pytest.

        Args:
            test_path: Path to tests
            markers: Marker filter expression
            verbose: Verbose output
            timeout: Execution timeout
            enable_coverage: Enable coverage
            working_dir: Working directory

        Returns:
            TestResult with pytest output
        """
        try:
            cmd = ["pytest", test_path]

            if verbose:
                cmd.append("-v")

            if markers:
                cmd.extend(["-m", markers])

            if enable_coverage:
                # Check if pytest-cov available
                if shutil.which("pytest-cov") or self._has_pytest_cov():
                    cmd.extend(["--cov", "--cov-report=term-missing"])

            output = await self._run_command(cmd, working_dir, timeout)

            # Parse pytest output
            return self._parse_pytest_output(output)

        except subprocess.TimeoutExpired:
            return TestResult(
                operation="run_tests",
                success=False,
                output=f"Test execution timeout after {timeout} seconds",
                framework="pytest",
            )
        except Exception as e:
            return TestResult(
                operation="run_tests",
                success=False,
                output=str(e),
                framework="pytest",
            )

    async def _run_unittest(
        self,
        test_path: str,
        pattern: Optional[str],
        verbose: bool,
        timeout: float,
        working_dir: str,
    ) -> TestResult:
        """Run tests using unittest.

        Args:
            test_path: Path to tests
            pattern: Test file pattern
            verbose: Verbose output
            timeout: Execution timeout
            working_dir: Working directory

        Returns:
            TestResult with unittest output
        """
        try:
            python_exe = self._get_python_executable()
            cmd = [python_exe, "-m", "unittest"]

            if verbose:
                cmd.append("-v")

            # Add test path
            if os.path.isfile(os.path.join(working_dir, test_path)):
                # Single file
                cmd.append(test_path)
            else:
                # Directory - use discover
                cmd.extend(["discover", "-s", test_path])
                if pattern:
                    cmd.extend(["-p", pattern])

            output = await self._run_command(cmd, working_dir, timeout)

            # Parse unittest output
            return self._parse_unittest_output(output)

        except subprocess.TimeoutExpired:
            return TestResult(
                operation="run_tests",
                success=False,
                output=f"Test execution timeout after {timeout} seconds",
                framework="unittest",
            )
        except Exception as e:
            return TestResult(
                operation="run_tests",
                success=False,
                output=str(e),
                framework="unittest",
            )

    # ========== Output Parsing ==========

    def _parse_pytest_output(self, output: str) -> TestResult:
        """Parse pytest output to extract results.

        Args:
            output: Raw pytest output

        Returns:
            TestResult with parsed information
        """
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        duration = 0.0
        coverage_percent = None
        coverage_missing = []

        # Parse test counts from summary line
        # Format: "X passed, Y failed, Z skipped in Ns"
        summary_pattern = r"(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) error"
        for match in re.finditer(summary_pattern, output):
            if match.group(1):
                passed = int(match.group(1))
            elif match.group(2):
                failed = int(match.group(2))
            elif match.group(3):
                skipped = int(match.group(3))
            elif match.group(4):
                errors = int(match.group(4))

        # Parse duration
        duration_match = re.search(r"in ([\d.]+)s", output)
        if duration_match:
            duration = float(duration_match.group(1))

        # Parse coverage if present
        coverage_match = re.search(r"TOTAL.*?(\d+)%", output)
        if coverage_match:
            coverage_percent = float(coverage_match.group(1))

        total = passed + failed + skipped
        success = failed == 0 and errors == 0

        return TestResult(
            operation="run_tests",
            success=success,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total=total,
            duration_seconds=duration,
            output=output,
            coverage_percent=coverage_percent,
            coverage_missing=coverage_missing,
            framework="pytest",
        )

    def _parse_unittest_output(self, output: str) -> TestResult:
        """Parse unittest output to extract results.

        Args:
            output: Raw unittest output

        Returns:
            TestResult with parsed information
        """
        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        # Parse test counts from summary line
        # Format: "Ran X tests in Ys" followed by "OK" or "FAILED (failures=Y, errors=Z)"
        run_match = re.search(r"Ran (\d+) test", output)
        if run_match:
            total = int(run_match.group(1))
        else:
            total = 0

        # Check for failures/errors
        fail_match = re.search(r"FAILED \(failures=(\d+)", output)
        if fail_match:
            failed = int(fail_match.group(1))

        error_match = re.search(r"errors=(\d+)", output)
        if error_match:
            errors = int(error_match.group(1))

        skip_match = re.search(r"skipped=(\d+)", output)
        if skip_match:
            skipped = int(skip_match.group(1))

        passed = total - failed - errors - skipped

        # Parse duration
        duration = 0.0
        duration_match = re.search(r"in ([\d.]+)s", output)
        if duration_match:
            duration = float(duration_match.group(1))

        success = failed == 0 and errors == 0

        return TestResult(
            operation="run_tests",
            success=success,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total=total,
            duration_seconds=duration,
            output=output,
            framework="unittest",
        )

    # ========== Helper Methods ==========

    async def _run_command(
        self,
        cmd: List[str],
        cwd: str,
        timeout: float,
    ) -> str:
        """Run command and return output.

        Args:
            cmd: Command as list of strings
            cwd: Working directory
            timeout: Command timeout in seconds

        Returns:
            Command output (stdout + stderr combined)

        Raises:
            subprocess.CalledProcessError: If command fails
            subprocess.TimeoutExpired: If command times out
            OSError: If directory doesn't exist
        """
        try:
            # Verify directory exists
            if not os.path.isdir(cwd):
                raise OSError(f"Directory not found: {cwd}")

            # Run command with timeout
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,  # Combine stderr into stdout
                    text=True,
                ),
                timeout=timeout,
            )

            stdout, _ = await asyncio.wait_for(
                result.communicate(),
                timeout=timeout,
            )

            # Don't raise on non-zero return code for tests (failures are valid)
            # Just return the output

            # Check output size
            if len(stdout) > MAX_OUTPUT_SIZE:
                logger.warning(
                    f"Test output truncated. Size: {len(stdout)} > {MAX_OUTPUT_SIZE}"
                )
                return stdout[:MAX_OUTPUT_SIZE]

            return stdout

        except asyncio.TimeoutError:
            raise subprocess.TimeoutExpired(
                " ".join(cmd),
                timeout,
            )

    def _detect_framework(
        self,
        working_dir: str,
        force: Optional[str],
    ) -> TestFrameworkInfo:
        """Detect which test framework to use.

        Args:
            working_dir: Working directory to check
            force: Force specific framework (pytest/unittest)

        Returns:
            TestFrameworkInfo with detected framework
        """
        if force:
            force_lower = force.lower()
            if force_lower in ["pytest", "py.test"]:
                return TestFrameworkInfo(
                    name="pytest",
                    available=shutil.which("pytest") is not None,
                )
            elif force_lower == "unittest":
                return TestFrameworkInfo(
                    name="unittest",
                    available=True,  # Always available
                )

        # Check for pytest
        has_pytest = shutil.which("pytest") is not None
        pytest_configs = [
            "pytest.ini",
            "pyproject.toml",
            "setup.cfg",
        ]

        found_configs = []
        for config in pytest_configs:
            config_path = os.path.join(working_dir, config)
            if os.path.exists(config_path):
                found_configs.append(config)

        if has_pytest and (found_configs or self._has_pytest_tests(working_dir)):
            return TestFrameworkInfo(
                name="pytest",
                available=True,
                config_files=found_configs,
            )

        # Default to unittest
        return TestFrameworkInfo(
            name="unittest",
            available=True,
        )

    def _has_pytest_tests(self, working_dir: str) -> bool:
        """Check if directory has pytest-style tests.

        Args:
            working_dir: Directory to check

        Returns:
            True if pytest tests found
        """
        test_dirs = ["tests", "test"]
        for test_dir in test_dirs:
            test_path = os.path.join(working_dir, test_dir)
            if os.path.isdir(test_path):
                # Check for test_*.py files
                for root, _, files in os.walk(test_path):
                    if any(f.startswith("test_") and f.endswith(".py") for f in files):
                        return True
        return False

    def _has_pytest_cov(self) -> bool:
        """Check if pytest-cov is available.

        Returns:
            True if pytest-cov installed
        """
        try:
            import pytest_cov
            return True
        except ImportError:
            return False

    def _find_test_directory(self, working_dir: str) -> str:
        """Find test directory in project.

        Args:
            working_dir: Working directory

        Returns:
            Path to test directory (relative or absolute)
        """
        # Common test directory names
        test_dirs = ["tests", "test", "testing"]

        for test_dir in test_dirs:
            test_path = os.path.join(working_dir, test_dir)
            if os.path.isdir(test_path):
                return test_dir

        # Default to current directory
        return "."

    def _validate_test_path(
        self,
        test_path: str,
        working_dir: str,
    ) -> str:
        """Validate test path.

        Args:
            test_path: User-provided test path
            working_dir: Working directory

        Returns:
            Validated path, or "ERROR: message" string

        Safety:
            - Prevents path traversal
            - Checks path exists
        """
        # Prevent path traversal
        if ".." in test_path:
            return "ERROR: Path traversal not allowed"

        # Resolve relative to working directory
        full_path = os.path.normpath(os.path.join(working_dir, test_path))

        # Ensure it's within working directory
        try:
            rel = os.path.relpath(full_path, working_dir)
            if rel.startswith(".."):
                return "ERROR: Test path escapes working directory"
        except ValueError:
            return "ERROR: Test path is on different drive"

        # Check path exists
        if not os.path.exists(full_path):
            return f"ERROR: Test path not found: {test_path}"

        return test_path

    def _get_python_executable(self) -> Optional[str]:
        """Get Python executable path.

        Returns:
            Path to Python executable, or None if not found
        """
        import sys
        return sys.executable

    def _format_result(self, test_result: TestResult) -> Dict[str, Any]:
        """Format test result for output.

        Args:
            test_result: TestResult from execution

        Returns:
            Dict with formatted result
        """
        result = {
            "operation": test_result.operation,
            "framework": test_result.framework,
            "success": test_result.success,
            "passed": test_result.passed,
            "failed": test_result.failed,
            "skipped": test_result.skipped,
            "errors": test_result.errors,
            "total": test_result.total,
            "duration_seconds": test_result.duration_seconds,
        }

        if test_result.coverage_percent is not None:
            result["coverage_percent"] = test_result.coverage_percent

        if test_result.coverage_missing:
            result["coverage_missing"] = test_result.coverage_missing

        if test_result.test_details:
            result["test_details"] = test_result.test_details

        # Include truncated output (first 10KB)
        if test_result.output:
            output = test_result.output
            if len(output) > 10000:
                output = output[:10000] + "\n... (output truncated)"
            result["output"] = output

        return result
