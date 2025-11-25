"""
Pytest Runner Skill
==================
Automated test execution with coverage reporting and detailed analysis.

This skill wraps pytest to provide:
- Test execution with configurable options
- Coverage reporting with thresholds
- Specific test/marker execution
- Parallel execution support
- Detailed failure analysis
"""

import asyncio
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from HoloLoom.skills.base import (
    BaseSkill,
    SkillInput,
    SkillOutput,
    SkillMetadata,
    SkillCategory,
    SkillStatus
)


@dataclass
class PytestResult:
    """Structured pytest execution result."""
    passed: int
    failed: int
    skipped: int
    errors: int
    total: int
    duration_seconds: float
    coverage_percent: Optional[float] = None
    failed_tests: List[str] = None
    warnings: List[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


class PytestRunnerSkill(BaseSkill):
    """
    Pytest Runner Skill - Automated test execution with coverage.

    Operations:
    - run_all: Run all tests
    - run_with_coverage: Run tests with coverage report
    - run_specific: Run specific test file or function
    - run_marker: Run tests with specific marker
    - run_parallel: Run tests in parallel (pytest-xdist required)
    """

    def __init__(self, pytest_path: str = "pytest"):
        """
        Initialize Pytest Runner Skill.

        Args:
            pytest_path: Path to pytest executable (default: "pytest" in PATH)
        """
        super().__init__()
        self.pytest_path = pytest_path

    def get_metadata(self) -> SkillMetadata:
        """Return skill metadata."""
        return SkillMetadata(
            name="pytest_runner",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.TESTING,
            tags=["testing", "pytest", "coverage", "quality", "automation"],
            description_short="Run pytest tests with coverage and detailed reporting",
            description_long=(
                "Automated test execution skill that wraps pytest to provide "
                "comprehensive test running capabilities including coverage "
                "reporting, parallel execution, marker-based filtering, and "
                "detailed failure analysis."
            ),
            requires_code_execution=True,
            requires_filesystem_read=True,
            external_dependencies=["pytest", "pytest-cov"],
            expected_latency_ms="1000-30000ms",  # 1-30 seconds depending on test suite
            token_usage="100-500 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        """
        Validate pytest runner input.

        Args:
            skill_input: Input to validate

        Returns:
            True if valid

        Raises:
            ValueError: If input is invalid
        """
        valid_operations = [
            "run_all", "run_with_coverage", "run_specific",
            "run_marker", "run_parallel"
        ]

        if skill_input.operation not in valid_operations:
            raise ValueError(
                f"Invalid operation: {skill_input.operation}. "
                f"Must be one of: {', '.join(valid_operations)}"
            )

        # Validate operation-specific parameters
        if skill_input.operation == "run_specific":
            if "path" not in skill_input.parameters:
                raise ValueError("run_specific requires 'path' parameter")

        if skill_input.operation == "run_marker":
            if "marker" not in skill_input.parameters:
                raise ValueError("run_marker requires 'marker' parameter")

        if skill_input.operation == "run_parallel":
            if "num_workers" not in skill_input.parameters:
                raise ValueError("run_parallel requires 'num_workers' parameter")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        """
        Execute pytest runner skill.

        Args:
            skill_input: Skill input with operation and parameters

        Returns:
            SkillOutput with test results
        """
        start_time = time.time()

        try:
            # Dispatch to operation handler
            if skill_input.operation == "run_all":
                result = await self._run_all(skill_input.parameters)
            elif skill_input.operation == "run_with_coverage":
                result = await self._run_with_coverage(skill_input.parameters)
            elif skill_input.operation == "run_specific":
                result = await self._run_specific(skill_input.parameters)
            elif skill_input.operation == "run_marker":
                result = await self._run_marker(skill_input.parameters)
            elif skill_input.operation == "run_parallel":
                result = await self._run_parallel(skill_input.parameters)
            else:
                raise ValueError(f"Unknown operation: {skill_input.operation}")

            execution_time = (time.time() - start_time) * 1000

            # Determine status
            if result.failed > 0 or result.errors > 0:
                status = SkillStatus.FAILURE
                message = f"Tests failed: {result.failed} failed, {result.errors} errors"
            elif result.skipped == result.total:
                status = SkillStatus.PARTIAL
                message = f"All {result.total} tests skipped"
            else:
                status = SkillStatus.SUCCESS
                message = f"Tests passed: {result.passed}/{result.total} ({result.success_rate:.1%})"

            # Build details
            details = {
                "passed": result.passed,
                "failed": result.failed,
                "skipped": result.skipped,
                "errors": result.errors,
                "total": result.total,
                "duration_seconds": result.duration_seconds,
                "success_rate": result.success_rate
            }

            if result.coverage_percent is not None:
                details["coverage_percent"] = result.coverage_percent

            if result.failed_tests:
                details["failed_tests"] = result.failed_tests

            return SkillOutput(
                status=status,
                result=result,
                message=message,
                execution_time_ms=execution_time,
                details=details,
                warnings=result.warnings or [],
                errors=[],
                skill_name=self._metadata.name,
                skill_version=self._metadata.version
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Pytest execution failed: {e}",
                execution_time_ms=execution_time,
                errors=[str(e)],
                skill_name=self._metadata.name,
                skill_version=self._metadata.version
            )

    async def _run_pytest(
        self,
        args: List[str],
        cwd: Optional[str] = None
    ) -> PytestResult:
        """
        Run pytest with given arguments.

        Args:
            args: Pytest command-line arguments
            cwd: Working directory (default: current directory)

        Returns:
            Structured pytest result
        """
        # Build command
        cmd = [self.pytest_path] + args + [
            "--tb=short",  # Short traceback format
            "--quiet",     # Less verbose output
            "-v"           # But show test names
        ]

        # Run pytest
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )

        stdout, stderr = await proc.communicate()

        # Parse output
        output = stdout.decode('utf-8', errors='ignore')
        error_output = stderr.decode('utf-8', errors='ignore')

        # Extract results from output
        result = self._parse_pytest_output(output, error_output)

        return result

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str
    ) -> PytestResult:
        """
        Parse pytest output to extract results.

        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest

        Returns:
            Parsed pytest result
        """
        # Default values
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        duration = 0.0
        coverage = None
        failed_tests = []
        warnings = []

        # Parse test summary line (e.g., "5 passed, 2 failed in 1.23s")
        lines = stdout.split('\n')
        for line in lines:
            if ' passed' in line or ' failed' in line:
                # Extract numbers
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            passed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'failed' and i > 0:
                        try:
                            failed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'skipped' and i > 0:
                        try:
                            skipped = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'error' or part == 'errors' and i > 0:
                        try:
                            errors = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part.endswith('s') and 'in' in parts[i-1]:
                        # Duration (e.g., "in 1.23s")
                        try:
                            duration = float(part[:-1])
                        except ValueError:
                            pass

            # Extract failed test names
            if 'FAILED' in line:
                parts = line.split('FAILED')
                if len(parts) > 1:
                    test_name = parts[1].strip().split()[0]
                    failed_tests.append(test_name)

            # Extract coverage
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            coverage = float(part[:-1])
                        except ValueError:
                            pass

            # Extract warnings
            if 'warning' in line.lower():
                warnings.append(line.strip())

        total = passed + failed + skipped + errors

        return PytestResult(
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total=total,
            duration_seconds=duration,
            coverage_percent=coverage,
            failed_tests=failed_tests if failed_tests else None,
            warnings=warnings if warnings else None
        )

    async def _run_all(self, parameters: Dict[str, Any]) -> PytestResult:
        """Run all tests."""
        path = parameters.get("path", ".")
        args = [path]
        return await self._run_pytest(args)

    async def _run_with_coverage(self, parameters: Dict[str, Any]) -> PytestResult:
        """Run tests with coverage reporting."""
        path = parameters.get("path", ".")
        min_coverage = parameters.get("min_coverage", 80)

        args = [
            path,
            "--cov=HoloLoom",
            "--cov-report=term-missing",
            f"--cov-fail-under={min_coverage}"
        ]

        return await self._run_pytest(args)

    async def _run_specific(self, parameters: Dict[str, Any]) -> PytestResult:
        """Run specific test file or function."""
        path = parameters["path"]
        args = [path]
        return await self._run_pytest(args)

    async def _run_marker(self, parameters: Dict[str, Any]) -> PytestResult:
        """Run tests with specific marker."""
        marker = parameters["marker"]
        path = parameters.get("path", ".")
        args = [path, "-m", marker]
        return await self._run_pytest(args)

    async def _run_parallel(self, parameters: Dict[str, Any]) -> PytestResult:
        """Run tests in parallel using pytest-xdist."""
        num_workers = parameters["num_workers"]
        path = parameters.get("path", ".")
        args = [path, "-n", str(num_workers)]
        return await self._run_pytest(args)


# Register skill
from HoloLoom.skills.base import register_skill
register_skill(PytestRunnerSkill())
