"""
ChatOps Test Handlers - Integration for running tests via chat commands.

Provides chat-based interface to HoloLoom's test infrastructure:
- Run unit, integration, and E2E tests
- Check test status and coverage
- Run performance benchmarks
- Trigger CI pipeline checks

Created: December 2025
Author: Claude Code
"""

import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

if TYPE_CHECKING:
    from nio import AsyncClient, MatrixRoom, RoomMessageText


@dataclass
class TestResult:
    """Result from a test run."""
    success: bool
    output: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class TestRunner:
    """
    Runs pytest tests and captures results.

    Supports different test types:
    - unit: Fast isolated tests (<5s)
    - integration: Multi-component tests (<30s)
    - e2e: Full pipeline tests (<2min)
    - parametrized: N×M scenario tests
    - benchmark: Performance regression tests
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test runner with project root."""
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.test_base = self.project_root / "hololoom" / "tests"

    async def run_tests(
        self,
        test_type: str = "unit",
        verbose: bool = False,
        timeout: int = 300
    ) -> TestResult:
        """
        Run tests of specified type.

        Args:
            test_type: Type of tests (unit, integration, e2e, parametrized, all)
            verbose: Include detailed output
            timeout: Timeout in seconds

        Returns:
            TestResult with pass/fail counts and output
        """
        test_paths = self._get_test_paths(test_type)

        if not test_paths:
            return TestResult(
                success=False,
                output=f"Unknown test type: {test_type}",
                errors=[f"Valid types: unit, integration, e2e, parametrized, all"]
            )

        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            *test_paths,
            "-v" if verbose else "-q",
            "--tb=short",
            "-x"  # Stop on first failure for fast feedback
        ]

        start_time = time.time()

        try:
            # Run pytest with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
                env={**dict(__import__('os').environ), "PYTHONPATH": str(self.project_root)}
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return TestResult(
                    success=False,
                    output=f"Tests timed out after {timeout}s",
                    errors=["Timeout exceeded"]
                )

            duration_ms = (time.time() - start_time) * 1000
            output = stdout.decode() + stderr.decode()

            # Parse results from output
            passed, failed, skipped = self._parse_results(output)

            return TestResult(
                success=process.returncode == 0,
                output=output if verbose else self._summarize_output(output),
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration_ms=duration_ms,
                errors=self._extract_errors(output) if failed > 0 else []
            )

        except Exception as e:
            return TestResult(
                success=False,
                output=f"Error running tests: {str(e)}",
                errors=[str(e)]
            )

    def _get_test_paths(self, test_type: str) -> List[str]:
        """Get test paths for test type."""
        paths = {
            "unit": [str(self.test_base / "unit")],
            "integration": [str(self.test_base / "integration")],
            "e2e": [str(self.test_base / "e2e")],
            "parametrized": [str(self.test_base / "integration" / "test_parametrized_scenarios.py")],
            "benchmark": [str(self.test_base / "integration" / "test_performance_regression.py")],
            "all": [str(self.test_base)]
        }
        return paths.get(test_type, [])

    def _parse_results(self, output: str) -> Tuple[int, int, int]:
        """Parse passed/failed/skipped counts from pytest output."""
        import re

        # Look for patterns like "5 passed", "2 failed", "1 skipped"
        passed = 0
        failed = 0
        skipped = 0

        # Match summary line like "10 passed, 2 failed, 1 skipped"
        patterns = [
            (r'(\d+)\s+passed', 'passed'),
            (r'(\d+)\s+failed', 'failed'),
            (r'(\d+)\s+skipped', 'skipped'),
            (r'(\d+)\s+error', 'failed'),  # Errors count as failed
        ]

        for pattern, result_type in patterns:
            match = re.search(pattern, output)
            if match:
                count = int(match.group(1))
                if result_type == 'passed':
                    passed = count
                elif result_type == 'failed':
                    failed += count
                elif result_type == 'skipped':
                    skipped = count

        return passed, failed, skipped

    def _summarize_output(self, output: str) -> str:
        """Extract summary from pytest output."""
        lines = output.strip().split('\n')

        # Get last few lines which typically contain summary
        summary_lines = []
        for line in reversed(lines):
            if line.strip():
                summary_lines.insert(0, line)
                if 'passed' in line or 'failed' in line or 'error' in line:
                    break
            if len(summary_lines) > 5:
                break

        return '\n'.join(summary_lines) if summary_lines else output[-500:]

    def _extract_errors(self, output: str) -> List[str]:
        """Extract error messages from pytest output."""
        errors = []
        lines = output.split('\n')

        in_error = False
        current_error = []

        for line in lines:
            if 'FAILED' in line or 'ERROR' in line:
                in_error = True
                current_error = [line]
            elif in_error:
                if line.startswith('    ') or line.startswith('\t'):
                    current_error.append(line)
                else:
                    if current_error:
                        errors.append('\n'.join(current_error[:10]))  # Limit error length
                    in_error = False
                    current_error = []

        if current_error:
            errors.append('\n'.join(current_error[:10]))

        return errors[:5]  # Limit to 5 errors


class TestStatusTracker:
    """Tracks test run status and history."""

    def __init__(self):
        self.last_results: Dict[str, TestResult] = {}
        self.run_history: List[Dict[str, Any]] = []

    def record(self, test_type: str, result: TestResult):
        """Record a test run."""
        self.last_results[test_type] = result
        self.run_history.append({
            "type": test_type,
            "success": result.success,
            "passed": result.passed,
            "failed": result.failed,
            "duration_ms": result.duration_ms,
            "timestamp": time.time()
        })

        # Keep last 50 runs
        if len(self.run_history) > 50:
            self.run_history = self.run_history[-50:]

    def get_status(self) -> Dict[str, Any]:
        """Get current test status."""
        return {
            "last_results": {
                k: {
                    "success": v.success,
                    "passed": v.passed,
                    "failed": v.failed,
                    "skipped": v.skipped,
                    "duration_ms": v.duration_ms
                }
                for k, v in self.last_results.items()
            },
            "total_runs": len(self.run_history),
            "recent_success_rate": self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate success rate from recent runs."""
        if not self.run_history:
            return 0.0

        recent = self.run_history[-10:]
        successes = sum(1 for r in recent if r["success"])
        return successes / len(recent)


# Global instances
_test_runner: Optional[TestRunner] = None
_status_tracker: Optional[TestStatusTracker] = None


def get_test_runner() -> TestRunner:
    """Get or create test runner."""
    global _test_runner
    if _test_runner is None:
        _test_runner = TestRunner()
    return _test_runner


def get_status_tracker() -> TestStatusTracker:
    """Get or create status tracker."""
    global _status_tracker
    if _status_tracker is None:
        _status_tracker = TestStatusTracker()
    return _status_tracker


# =============================================================================
# Chat Command Handlers
# =============================================================================

async def handle_test_run(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    bot: Optional[Any] = None
) -> str:
    """
    Handle !test run [type] command.

    Usage:
        !test run unit        - Run unit tests
        !test run integration - Run integration tests
        !test run e2e         - Run end-to-end tests
        !test run parametrized - Run N×M scenario tests
        !test run benchmark   - Run performance benchmarks
        !test run all         - Run all tests
    """
    test_type = args[0] if args else "unit"
    verbose = "-v" in args or "--verbose" in args

    runner = get_test_runner()
    tracker = get_status_tracker()

    # Send initial message
    response = f"🧪 Running {test_type} tests..."

    # Run tests
    result = await runner.run_tests(test_type, verbose=verbose)
    tracker.record(test_type, result)

    # Format response
    if result.success:
        status = "✅ PASSED"
    else:
        status = "❌ FAILED"

    response = f"""
**Test Run: {test_type}** {status}

📊 Results:
- Passed: {result.passed}
- Failed: {result.failed}
- Skipped: {result.skipped}
- Duration: {result.duration_ms:.1f}ms

"""

    if result.errors:
        response += "**Errors:**\n```\n"
        response += "\n".join(result.errors[:3])
        response += "\n```\n"

    if verbose:
        response += f"\n**Output:**\n```\n{result.output[:1500]}\n```"

    return response


async def handle_test_status(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    bot: Optional[Any] = None
) -> str:
    """
    Handle !test status command.

    Shows status of recent test runs.
    """
    tracker = get_status_tracker()
    status = tracker.get_status()

    response = """
**🧪 Test Status**

"""

    if not status["last_results"]:
        response += "_No tests have been run yet._\n\nUse `!test run unit` to run tests."
        return response

    response += "| Type | Status | Passed | Failed | Duration |\n"
    response += "|------|--------|--------|--------|----------|\n"

    for test_type, result in status["last_results"].items():
        icon = "✅" if result["success"] else "❌"
        response += f"| {test_type} | {icon} | {result['passed']} | {result['failed']} | {result['duration_ms']:.0f}ms |\n"

    response += f"\n📈 Recent success rate: {status['recent_success_rate']:.0%}\n"
    response += f"📝 Total runs: {status['total_runs']}\n"

    return response


async def handle_test_coverage(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    bot: Optional[Any] = None
) -> str:
    """
    Handle !test coverage command.

    Shows test coverage summary (requires pytest-cov).
    """
    runner = get_test_runner()

    # Run with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(runner.test_base / "unit"),
        "--cov=HoloLoom",
        "--cov-report=term-missing",
        "-q"
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(runner.project_root),
            env={**dict(__import__('os').environ), "PYTHONPATH": str(runner.project_root)}
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=180
        )

        output = stdout.decode()

        # Extract coverage summary
        lines = output.split('\n')
        coverage_lines = []
        in_coverage = False

        for line in lines:
            if 'TOTAL' in line or 'Name' in line or '----' in line:
                in_coverage = True
            if in_coverage:
                coverage_lines.append(line)
                if 'TOTAL' in line:
                    break

        if coverage_lines:
            return f"""
**📊 Test Coverage**

```
{chr(10).join(coverage_lines)}
```

_Run `!test run unit -v` for detailed test output._
"""
        else:
            return "Could not parse coverage report. Install pytest-cov: `pip install pytest-cov`"

    except asyncio.TimeoutError:
        return "Coverage run timed out after 180s"
    except Exception as e:
        return f"Error running coverage: {str(e)}"


async def handle_test_benchmark(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    bot: Optional[Any] = None
) -> str:
    """
    Handle !test benchmark command.

    Runs performance regression tests.
    """
    runner = get_test_runner()
    tracker = get_status_tracker()

    response = "🏃 Running performance benchmarks..."

    result = await runner.run_tests("benchmark", verbose=True, timeout=600)
    tracker.record("benchmark", result)

    if result.success:
        return f"""
**⚡ Performance Benchmarks** ✅

📊 Results:
- Passed: {result.passed}
- Duration: {result.duration_ms:.1f}ms

All performance thresholds met!

```
{result.output[:1000]}
```
"""
    else:
        return f"""
**⚡ Performance Benchmarks** ❌

Some benchmarks failed regression thresholds.

```
{result.output[:1500]}
```

Check the errors above for details on which benchmarks regressed.
"""


async def handle_test_ci(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    bot: Optional[Any] = None
) -> str:
    """
    Handle !test ci command.

    Shows CI pipeline status or triggers a run.
    """
    # This would integrate with GitHub Actions API in production
    # For now, show status of what CI would run

    return """
**🔄 CI Pipeline Status**

The CI pipeline runs automatically on push to master/main/feature branches.

**Jobs:**
1. ✅ **Unit Tests** - Fast feedback (<5s)
2. ⏳ **Integration Tests** - Multi-component (<30s)
3. ⏳ **RAG Tests** - RAG system specific
4. ⏳ **Routing Tests** - Query classification
5. ⏳ **Alignment Tests** - Safety framework
6. 🔜 **E2E Tests** - Full pipeline (PR/master only)

**Workflow:** `.github/workflows/hololoom-test-suite.yml`

To run locally:
```bash
# Unit tests (fast)
pytest HoloLoom/tests/unit/ -v

# All tests
pytest HoloLoom/tests/ -v
```

_Note: Actual CI status would require GitHub API integration._
"""


async def handle_test_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    bot: Optional[Any] = None
) -> str:
    """Handle !test help command."""
    return """
**🧪 Test Commands**

| Command | Description |
|---------|-------------|
| `!test run [type]` | Run tests (unit/integration/e2e/parametrized/all) |
| `!test run unit -v` | Run unit tests with verbose output |
| `!test status` | Show status of recent test runs |
| `!test coverage` | Show test coverage report |
| `!test benchmark` | Run performance regression tests |
| `!test ci` | Show CI pipeline status |
| `!test help` | Show this help message |

**Test Types:**
- `unit` - Fast isolated tests (<5s)
- `integration` - Multi-component tests (<30s)
- `e2e` - Full pipeline tests (<2min)
- `parametrized` - N×M scenario tests
- `benchmark` - Performance regression tests
- `all` - Run all tests

**Examples:**
```
!test run unit
!test run integration -v
!test status
!test benchmark
```
"""


# =============================================================================
# Registration
# =============================================================================

def register_test_handlers(
    bot: Any,
    orchestrator: Optional[Any] = None
) -> None:
    """
    Register test command handlers with a Matrix bot.

    Args:
        bot: Matrix bot instance with register_command method
        orchestrator: Optional ChatOps orchestrator (not used for test commands)
    """
    command_map = {
        "run": lambda room, event, args: handle_test_run(room, event, args, bot),
        "status": lambda room, event, args: handle_test_status(room, event, args, bot),
        "coverage": lambda room, event, args: handle_test_coverage(room, event, args, bot),
        "benchmark": lambda room, event, args: handle_test_benchmark(room, event, args, bot),
        "ci": lambda room, event, args: handle_test_ci(room, event, args, bot),
        "help": lambda room, event, args: handle_test_help(room, event, args, bot),
    }

    for cmd, handler in command_map.items():
        bot.register_command(f"test {cmd}", handler)

    # Also register base !test command to show help
    bot.register_command("test", lambda room, event, args: handle_test_help(room, event, args, bot))


# =============================================================================
# Standalone Testing
# =============================================================================

async def _test_handlers():
    """Test the handlers directly without Matrix."""
    print("Testing ChatOps Test Handlers\n" + "="*40)

    # Mock room and event
    class MockRoom:
        room_id = "!test:example.com"

    class MockEvent:
        sender = "@user:example.com"

    room = MockRoom()
    event = MockEvent()

    # Test help
    print("\n1. Testing !test help:")
    result = await handle_test_help(room, event, [])
    print(result[:500] + "...")

    # Test status (no runs yet)
    print("\n2. Testing !test status:")
    result = await handle_test_status(room, event, [])
    print(result)

    # Test run unit (actual test run)
    print("\n3. Testing !test run unit:")
    result = await handle_test_run(room, event, ["unit"])
    print(result[:800] + "..." if len(result) > 800 else result)

    # Test status after run
    print("\n4. Testing !test status (after run):")
    result = await handle_test_status(room, event, [])
    print(result)

    print("\n" + "="*40)
    print("All handler tests completed!")


if __name__ == "__main__":
    asyncio.run(_test_handlers())


# =============================================================================
# Decorator-Based Handler Class (New ChatOps Pattern)
# =============================================================================

try:
    from hololoom.apps.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class TestHandlers:
    """
    Decorator-based ChatOps handlers for test operations.

    Usage:
        from hololoom.apps.chatops.handlers.test_handlers import TestHandlers

        handlers = TestHandlers()
        registry.register_instance(handlers)
    """

    def __init__(self):
        """Initialize test handlers."""
        self.runner = get_test_runner()
        self.tracker = get_status_tracker()

    def _parse_args(self, args: str) -> List[str]:
        """Convert args string to list."""
        return args.split() if args.strip() else []

    @HandlerRegistry.register(
        command="test run",
        description="Run tests (unit/integration/e2e/parametrized/all)",
        usage="!test run [type] [-v]",
        category=HandlerCategory.SYSTEM,
        aliases=["tr"]
    )
    async def handle_run(self, room, event, args: str) -> str:
        """Handle !test run command."""
        return await handle_test_run(room, event, self._parse_args(args))

    @HandlerRegistry.register(
        command="test status",
        description="Show status of recent test runs",
        usage="!test status",
        category=HandlerCategory.SYSTEM,
        aliases=["ts"]
    )
    async def handle_status(self, room, event, args: str) -> str:
        """Handle !test status command."""
        return await handle_test_status(room, event, self._parse_args(args))

    @HandlerRegistry.register(
        command="test coverage",
        description="Show test coverage report",
        usage="!test coverage",
        category=HandlerCategory.SYSTEM,
        aliases=["tc"]
    )
    async def handle_coverage(self, room, event, args: str) -> str:
        """Handle !test coverage command."""
        return await handle_test_coverage(room, event, self._parse_args(args))

    @HandlerRegistry.register(
        command="test benchmark",
        description="Run performance regression tests",
        usage="!test benchmark",
        category=HandlerCategory.SYSTEM,
        aliases=["tb"]
    )
    async def handle_benchmark(self, room, event, args: str) -> str:
        """Handle !test benchmark command."""
        return await handle_test_benchmark(room, event, self._parse_args(args))

    @HandlerRegistry.register(
        command="test ci",
        description="Show CI pipeline status",
        usage="!test ci",
        category=HandlerCategory.SYSTEM,
        aliases=["tci"]
    )
    async def handle_ci(self, room, event, args: str) -> str:
        """Handle !test ci command."""
        return await handle_test_ci(room, event, self._parse_args(args))

    @HandlerRegistry.register(
        command="test help",
        description="Show test command help",
        usage="!test help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    )
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !test help command."""
        return await handle_test_help(room, event, self._parse_args(args))
