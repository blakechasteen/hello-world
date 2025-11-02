"""
E2E Test Monitor Dashboard
===========================

Real-time monitoring dashboard for E2E test execution.

Features:
- Live test progress tracking
- Pass/fail/skip counts
- Test timing information
- Per-file and per-class breakdowns
- Failure details

Usage:
    # Option 1: Run with pytest plugin (automatic)
    pytest HoloLoom/tests/e2e/ -v --tb=short

    # Option 2: Standalone monitoring (manual parsing)
    python HoloLoom/tests/e2e_test_monitor.py

    # Option 3: pytest-json-report plugin
    pip install pytest-json-report
    pytest HoloLoom/tests/e2e/ --json-report --json-report-file=test_results.json
"""

import time
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Install 'rich' for live dashboard: pip install rich")


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    file: str
    status: str  # "PASSED", "FAILED", "SKIPPED", "RUNNING"
    duration_s: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class E2ETestMonitor:
    """
    Real-time E2E test monitoring dashboard.

    Tracks:
    - Total tests run/passed/failed/skipped
    - Per-file test counts
    - Per-class test counts
    - Test timing (total, average, slowest)
    - Failure details
    """

    def __init__(self):
        self.tests: List[TestResult] = []
        self.by_file: Dict[str, List[TestResult]] = defaultdict(list)
        self.by_status: Dict[str, List[TestResult]] = defaultdict(list)
        self.start_time = time.time()
        self.console = Console() if RICH_AVAILABLE else None

    def add_test(self, result: TestResult):
        """Add a test result."""
        self.tests.append(result)
        self.by_file[result.file].append(result)
        self.by_status[result.status].append(result)

    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics."""
        total = len(self.tests)
        passed = len(self.by_status.get("PASSED", []))
        failed = len(self.by_status.get("FAILED", []))
        skipped = len(self.by_status.get("SKIPPED", []))
        running = len(self.by_status.get("RUNNING", []))

        durations = [t.duration_s for t in self.tests if t.duration_s > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        total_duration = time.time() - self.start_time

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'running': running,
            'pass_rate': (passed / total * 100) if total > 0 else 0.0,
            'avg_duration_s': avg_duration,
            'total_duration_s': total_duration,
            'slowest': max(durations) if durations else 0.0,
        }

    def _create_summary_panel(self) -> Panel:
        """Create summary statistics panel."""
        from rich.text import Text

        summary = self.get_summary()

        text = Text()
        text.append("E2E Test Execution\n\n", style="bold cyan")

        # Status counts
        text.append(f"Total: ", style="bold")
        text.append(f"{summary['total']} tests\n")

        text.append(f"✅ Passed: ", style="bold green")
        text.append(f"{summary['passed']}\n")

        text.append(f"❌ Failed: ", style="bold red")
        text.append(f"{summary['failed']}\n")

        text.append(f"⏭️  Skipped: ", style="bold yellow")
        text.append(f"{summary['skipped']}\n")

        if summary['running'] > 0:
            text.append(f"🏃 Running: ", style="bold blue")
            text.append(f"{summary['running']}\n")

        text.append(f"\n")

        # Pass rate
        pass_rate = summary['pass_rate']
        color = "green" if pass_rate >= 90 else "yellow" if pass_rate >= 70 else "red"
        text.append(f"Pass Rate: ", style="bold")
        text.append(f"{pass_rate:.1f}%\n", style=color)

        # Timing
        text.append(f"\n")
        text.append(f"Avg Duration: ", style="bold")
        text.append(f"{summary['avg_duration_s']:.2f}s\n")

        text.append(f"Total Time: ", style="bold")
        text.append(f"{summary['total_duration_s']:.1f}s\n")

        text.append(f"Slowest Test: ", style="bold")
        text.append(f"{summary['slowest']:.2f}s")

        return Panel(text, title="[bold]E2E Test Summary[/bold]", border_style="cyan")

    def _create_file_table(self) -> Table:
        """Create per-file breakdown table."""
        table = Table(title="Tests by File", box=box.ROUNDED)

        table.add_column("File", style="cyan", no_wrap=False)
        table.add_column("Total", style="magenta", justify="right")
        table.add_column("✅", style="green", justify="right")
        table.add_column("❌", style="red", justify="right")
        table.add_column("⏭️", style="yellow", justify="right")
        table.add_column("Avg Time", style="blue", justify="right")

        for file, results in sorted(self.by_file.items()):
            total = len(results)
            passed = sum(1 for r in results if r.status == "PASSED")
            failed = sum(1 for r in results if r.status == "FAILED")
            skipped = sum(1 for r in results if r.status == "SKIPPED")

            durations = [r.duration_s for r in results if r.duration_s > 0]
            avg_time = sum(durations) / len(durations) if durations else 0.0

            # Shorten file path
            short_file = Path(file).name

            table.add_row(
                short_file,
                str(total),
                str(passed),
                str(failed),
                str(skipped),
                f"{avg_time:.2f}s"
            )

        return table

    def _create_failures_table(self) -> Optional[Table]:
        """Create failures table (if any failures)."""
        failures = self.by_status.get("FAILED", [])
        if not failures:
            return None

        table = Table(title="❌ Failed Tests", box=box.ROUNDED)

        table.add_column("Test", style="red", no_wrap=False)
        table.add_column("File", style="cyan", no_wrap=False)
        table.add_column("Error", style="yellow", no_wrap=False)

        for result in failures:
            short_name = result.name.split("::")[-1]
            short_file = Path(result.file).name
            error_preview = (result.error[:50] + "...") if result.error and len(result.error) > 50 else (result.error or "")

            table.add_row(short_name, short_file, error_preview)

        return table

    def _create_slowest_table(self, limit: int = 5) -> Table:
        """Create slowest tests table."""
        table = Table(title=f"🐢 Slowest {limit} Tests", box=box.ROUNDED)

        table.add_column("Test", style="cyan", no_wrap=False)
        table.add_column("Duration", style="yellow", justify="right")
        table.add_column("Status", style="magenta", justify="center")

        # Sort by duration
        slowest = sorted(
            [t for t in self.tests if t.duration_s > 0],
            key=lambda t: t.duration_s,
            reverse=True
        )[:limit]

        for result in slowest:
            short_name = result.name.split("::")[-1]
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⏭️"}.get(result.status, "?")

            table.add_row(
                short_name,
                f"{result.duration_s:.2f}s",
                status_emoji
            )

        return table

    def display(self):
        """Display the dashboard once."""
        if not RICH_AVAILABLE:
            # Fallback to text output
            summary = self.get_summary()
            print(f"\n=== E2E Test Monitor ===")
            print(f"Total: {summary['total']} tests")
            print(f"Passed: {summary['passed']} ✅")
            print(f"Failed: {summary['failed']} ❌")
            print(f"Skipped: {summary['skipped']} ⏭️")
            print(f"Pass Rate: {summary['pass_rate']:.1f}%")
            print(f"Total Time: {summary['total_duration_s']:.1f}s")
            return

        self.console.clear()
        self.console.print(Panel.fit("[bold cyan]E2E Test Monitor[/bold cyan]"))
        self.console.print()

        # Summary
        self.console.print(self._create_summary_panel())
        self.console.print()

        # Per-file breakdown
        if self.by_file:
            self.console.print(self._create_file_table())
            self.console.print()

        # Failures
        failures_table = self._create_failures_table()
        if failures_table:
            self.console.print(failures_table)
            self.console.print()

        # Slowest tests
        if self.tests:
            self.console.print(self._create_slowest_table())

    def parse_pytest_output(self, output_file: Path):
        """
        Parse pytest verbose output and populate dashboard.

        Parses lines like:
        test_file.py::TestClass::test_method PASSED [  0%]
        test_file.py::TestClass::test_method FAILED [  1%]
        """
        if not output_file.exists():
            print(f"Output file not found: {output_file}")
            return

        with open(output_file, 'r') as f:
            for line in f:
                # Match pytest output lines
                # Format: path/to/test.py::TestClass::test_method STATUS [ XX%]
                match = re.match(r'(.+?)::(.+?)\s+(PASSED|FAILED|SKIPPED)', line)
                if match:
                    file_path, test_name, status = match.groups()

                    # Try to extract timing if present
                    duration = 0.0
                    time_match = re.search(r'\[(.+?)s\]', line)
                    if time_match:
                        try:
                            duration = float(time_match.group(1))
                        except ValueError:
                            pass

                    result = TestResult(
                        name=test_name,
                        file=file_path,
                        status=status,
                        duration_s=duration
                    )
                    self.add_test(result)


def parse_json_report(json_file: Path) -> E2ETestMonitor:
    """
    Parse pytest-json-report output.

    Requires: pip install pytest-json-report
    Run with: pytest --json-report --json-report-file=results.json
    """
    monitor = E2ETestMonitor()

    if not json_file.exists():
        print(f"JSON report not found: {json_file}")
        return monitor

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Parse tests from JSON report
    for test in data.get('tests', []):
        result = TestResult(
            name=test['nodeid'],
            file=test.get('path', test['nodeid'].split('::')[0]),
            status=test['outcome'].upper(),
            duration_s=test.get('call', {}).get('duration', 0.0),
            error=test.get('call', {}).get('longrepr', None)
        )
        monitor.add_test(result)

    return monitor


def main():
    """
    Main entry point for standalone monitoring.

    Usage:
        # Generate JSON report first
        pytest HoloLoom/tests/e2e/ --json-report --json-report-file=test_results.json

        # Then display dashboard
        python HoloLoom/tests/e2e_test_monitor.py
    """
    # Check for JSON report
    json_file = Path("test_results.json")

    if json_file.exists():
        print("Loading test results from test_results.json...")
        monitor = parse_json_report(json_file)
        monitor.display()
    else:
        print("No test results found.")
        print("\nTo generate test results:")
        print("1. Install pytest-json-report: pip install pytest-json-report")
        print("2. Run tests: pytest HoloLoom/tests/e2e/ --json-report --json-report-file=test_results.json")
        print("3. View dashboard: python HoloLoom/tests/e2e_test_monitor.py")


if __name__ == "__main__":
    main()
