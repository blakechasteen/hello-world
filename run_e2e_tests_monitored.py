"""
Run E2E Tests with Monitoring Dashboard
========================================

Runs E2E tests in smaller batches to avoid Windows/PyTorch crashes,
with real-time dashboard monitoring.

Usage:
    python run_e2e_tests_monitored.py

Features:
- Runs each E2E test file individually (avoids model reload crashes)
- Displays live progress dashboard
- Generates comprehensive test report
- Handles PyTorch Windows crashes gracefully
"""

import subprocess
import json
from pathlib import Path
import time
from typing import List, Dict
import sys

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for better output: pip install rich")


# E2E test files (order optimized to run fastest first)
E2E_TEST_FILES = [
    "HoloLoom/tests/e2e/test_cache_effectiveness.py",
    "HoloLoom/tests/e2e/test_edge_cases.py",
    "HoloLoom/tests/e2e/test_error_handling.py",
    "HoloLoom/tests/e2e/test_concurrent_queries.py",
    "HoloLoom/tests/e2e/test_performance_profile.py",
    "HoloLoom/tests/e2e/test_reflection_loop.py",
    "HoloLoom/tests/e2e/test_memory_growth.py",
    "HoloLoom/tests/e2e/test_persistence.py",
    "HoloLoom/tests/e2e/test_integration_scenarios.py",
]


class TestRunner:
    """Runs E2E tests with monitoring."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.results: Dict[str, Dict] = {}
        self.start_time = time.time()

    def run_test_file(self, test_file: str) -> Dict:
        """Run a single test file and return results."""
        result = {
            'file': test_file,
            'status': 'RUNNING',
            'tests_run': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'duration_s': 0.0,
            'crashed': False,
            'error': None
        }

        start = time.time()

        try:
            # Run pytest with JSON output
            cmd = [
                sys.executable, '-m', 'pytest',
                test_file,
                '-v',
                '--tb=short',
                '--json-report',
                '--json-report-file=.temp_test_result.json',
                '--json-report-omit=log'
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )

            result['duration_s'] = time.time() - start

            # Parse JSON report if available
            json_file = Path('.temp_test_result.json')
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)

                result['tests_run'] = data['summary']['total']
                result['passed'] = data['summary'].get('passed', 0)
                result['failed'] = data['summary'].get('failed', 0)
                result['skipped'] = data['summary'].get('skipped', 0)

                # Clean up temp file
                json_file.unlink()

            # Check for crash
            if proc.returncode == 139 or 'fatal exception' in proc.stderr.lower():
                result['crashed'] = True
                result['status'] = 'CRASHED'
            elif proc.returncode == 0:
                result['status'] = 'PASSED'
            else:
                result['status'] = 'FAILED'
                result['error'] = proc.stderr[:500] if proc.stderr else None

        except subprocess.TimeoutExpired:
            result['status'] = 'TIMEOUT'
            result['error'] = 'Test file exceeded 5 minute timeout'
            result['duration_s'] = time.time() - start
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
            result['duration_s'] = time.time() - start

        return result

    def _create_summary_table(self) -> Table:
        """Create summary results table."""
        table = Table(title="E2E Test Results Summary", box=box.ROUNDED)

        table.add_column("File", style="cyan", no_wrap=False)
        table.add_column("Status", style="magenta", justify="center")
        table.add_column("Tests", style="blue", justify="right")
        table.add_column("✅", style="green", justify="right")
        table.add_column("❌", style="red", justify="right")
        table.add_column("⏭️", style="yellow", justify="right")
        table.add_column("Time", style="white", justify="right")

        for file, result in self.results.items():
            short_file = Path(file).name

            # Status emoji
            status_map = {
                'PASSED': '✅ PASS',
                'FAILED': '❌ FAIL',
                'CRASHED': '💥 CRASH',
                'TIMEOUT': '⏱️ TIMEOUT',
                'ERROR': '⚠️ ERROR',
                'RUNNING': '🏃 RUN'
            }
            status = status_map.get(result['status'], result['status'])

            table.add_row(
                short_file,
                status,
                str(result['tests_run']),
                str(result['passed']),
                str(result['failed']),
                str(result['skipped']),
                f"{result['duration_s']:.1f}s"
            )

        return table

    def _create_overall_summary(self) -> Panel:
        """Create overall summary panel."""
        from rich.text import Text

        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        total_crashed = sum(1 for r in self.results.values() if r['crashed'])
        total_duration = time.time() - self.start_time

        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0

        text = Text()
        text.append("E2E Test Execution Complete\n\n", style="bold cyan")

        text.append(f"Files Run: ", style="bold")
        text.append(f"{len(self.results)}/{len(E2E_TEST_FILES)}\n")

        text.append(f"Total Tests: ", style="bold")
        text.append(f"{total_tests}\n")

        text.append(f"✅ Passed: ", style="bold green")
        text.append(f"{total_passed}\n")

        text.append(f"❌ Failed: ", style="bold red")
        text.append(f"{total_failed}\n")

        if total_crashed > 0:
            text.append(f"💥 Crashed: ", style="bold red")
            text.append(f"{total_crashed} files (PyTorch Windows issue)\n")

        text.append(f"\n")

        # Pass rate
        color = "green" if pass_rate >= 90 else "yellow" if pass_rate >= 70 else "red"
        text.append(f"Pass Rate: ", style="bold")
        text.append(f"{pass_rate:.1f}%\n", style=color)

        text.append(f"Total Time: ", style="bold")
        text.append(f"{total_duration:.1f}s")

        return Panel(text, title="[bold]Overall Summary[/bold]", border_style="cyan")

    def run_all(self):
        """Run all E2E test files with progress tracking."""
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:

                task = progress.add_task("Running E2E tests...", total=len(E2E_TEST_FILES))

                for test_file in E2E_TEST_FILES:
                    file_name = Path(test_file).name
                    progress.update(task, description=f"Running {file_name}...")

                    result = self.run_test_file(test_file)
                    self.results[test_file] = result

                    progress.advance(task)

            # Display results
            self.console.print("\n")
            self.console.print(self._create_overall_summary())
            self.console.print("\n")
            self.console.print(self._create_summary_table())

        else:
            # Fallback without rich
            for i, test_file in enumerate(E2E_TEST_FILES, 1):
                print(f"\n[{i}/{len(E2E_TEST_FILES)}] Running {Path(test_file).name}...")

                result = self.run_test_file(test_file)
                self.results[test_file] = result

                status_emoji = {'PASSED': '✅', 'FAILED': '❌', 'CRASHED': '💥'}.get(result['status'], '❓')
                print(f"  {status_emoji} {result['status']} - {result['tests_run']} tests in {result['duration_s']:.1f}s")

            # Summary
            total_passed = sum(r['passed'] for r in self.results.values())
            total_failed = sum(r['failed'] for r in self.results.values())
            total_tests = sum(r['tests_run'] for r in self.results.values())

            print(f"\n=== Overall Summary ===")
            print(f"Total Tests: {total_tests}")
            print(f"Passed: {total_passed} ✅")
            print(f"Failed: {total_failed} ❌")


def main():
    """Main entry point."""
    # Check for pytest-json-report
    try:
        import pytest_jsonreport
    except ImportError:
        print("⚠️  Installing pytest-json-report for better reporting...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest-json-report'], check=True)

    runner = TestRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
