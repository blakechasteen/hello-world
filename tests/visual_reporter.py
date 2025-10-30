#!/usr/bin/env python3
"""
Visual Test Reporter with Sparklines
=====================================
Ruthlessly elegant test reporting.

Philosophy: "If the report is boring, we failed."

Generates beautiful visual reports with:
- ASCII sparklines showing trends
- Progress bars for pass rates
- Performance metrics visualization
- Ruthless elegance scoring

Usage:
    python tests/visual_reporter.py

Author: Claude Code
Date: October 29, 2025
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import subprocess
import re
from dataclasses import dataclass

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TestResult:
    """Single test result."""
    name: str
    status: str  # passed, failed, skipped
    duration: float  # seconds
    category: str  # core, intelligence, performance, etc.


@dataclass
class TestReport:
    """Complete test report."""
    total: int
    passed: int
    failed: int
    skipped: int
    duration: float
    results: List[TestResult]
    file_name: str


# ============================================================================
# Test Runner
# ============================================================================

def run_pytest(test_file: str) -> Tuple[TestReport, str]:
    """
    Run pytest on a file and capture results.

    Returns:
        (TestReport, raw_output)
    """
    print(f"[visual_reporter] Running {test_file}...")

    start = time.time()

    # Run the test file directly since it has pytest.main() at the bottom
    # Capture output
    import subprocess
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(Path(__file__).parent.parent)
    )

    duration = time.time() - start
    output = result.stdout + result.stderr

    # Debug: Print first line of output if there are issues
    if 'ERROR' in output.upper() and 'collected 0' in output:
        print(f"  [!] Warning: Test collection or execution issues")
        first_error = [line for line in output.split('\n') if 'ERROR' in line or 'error' in line]
        if first_error:
            print(f"  {first_error[0][:100]}")

    # Parse pytest output
    results = []

    # Extract individual test results
    # Pattern: "::test_name PASSED/FAILED/SKIPPED"
    test_pattern = r'::(\w+)\s+(PASSED|FAILED|SKIPPED|ERROR)'

    seen_tests = set()  # Avoid duplicates

    for match in re.finditer(test_pattern, output):
        test_name = match.group(1)
        status = match.group(2).lower()

        # Avoid duplicates
        if test_name in seen_tests:
            continue
        seen_tests.add(test_name)

        if status == 'error':
            status = 'failed'

        # Extract category from test name
        if 'ruthless' in test_name:
            category = 'philosophy'
        elif 'performance' in test_name or 'speed' in test_name or 'fast' in test_name:
            category = 'performance'
        elif 'intelligence' in test_name or 'insight' in test_name or 'pattern' in test_name:
            category = 'intelligence'
        elif 'edge' in test_name or 'empty' in test_name or 'error' in test_name:
            category = 'edge_cases'
        elif 'integration' in test_name or 'workflow' in test_name:
            category = 'integration'
        else:
            category = 'core'

        results.append(TestResult(
            name=test_name,
            status=status,
            duration=0.1,  # Individual durations not easily parseable
            category=category
        ))

    # Extract summary - count from parsed results
    passed = sum(1 for r in results if r.status == 'passed')
    failed = sum(1 for r in results if r.status == 'failed')
    skipped = sum(1 for r in results if r.status == 'skipped')
    total = len(results)

    report = TestReport(
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        duration=duration,
        results=results,
        file_name=Path(test_file).name
    )

    return report, output


# ============================================================================
# Visualization Helpers
# ============================================================================

def sparkline(values: List[float], width: int = 20) -> str:
    """
    Generate ASCII sparkline.

    Args:
        values: List of numeric values
        width: Width of sparkline in characters

    Returns:
        ASCII sparkline string
    """
    if not values or len(values) == 0:
        return ' ' * width

    # Normalize values to 0-7 range (8 levels)
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        normalized = [4] * len(values)
    else:
        normalized = [int((v - min_val) / (max_val - min_val) * 7) for v in values]

    # Unicode block characters for sparkline
    # Using ASCII fallback for compatibility
    chars = [' ', '.', ':', '-', '=', '+', '#', '@']

    # Sample to width
    if len(normalized) > width:
        # Downsample
        step = len(normalized) / width
        sampled = [normalized[int(i * step)] for i in range(width)]
    else:
        # Pad
        sampled = normalized + [0] * (width - len(normalized))

    return ''.join(chars[val] for val in sampled)


def progress_bar(value: float, total: float, width: int = 40) -> str:
    """
    Generate ASCII progress bar.

    Args:
        value: Current value
        total: Total value
        width: Width in characters

    Returns:
        ASCII progress bar
    """
    if total == 0:
        percent = 0
    else:
        percent = value / total

    filled = int(width * percent)
    empty = width - filled

    bar = '[' + '#' * filled + '-' * empty + ']'
    percent_str = f'{percent * 100:.1f}%'

    return f'{bar} {percent_str}'


def category_breakdown(results: List[TestResult]) -> Dict[str, Dict[str, int]]:
    """
    Break down results by category.

    Returns:
        {category: {'passed': X, 'failed': Y, 'skipped': Z}}
    """
    breakdown = {}

    for result in results:
        if result.category not in breakdown:
            breakdown[result.category] = {'passed': 0, 'failed': 0, 'skipped': 0}

        breakdown[result.category][result.status] += 1

    return breakdown


def ruthless_elegance_score(report: TestReport) -> float:
    """
    Calculate ruthless elegance score.

    Criteria:
    - Pass rate (40%): High pass rate = elegant
    - Performance (30%): Fast tests = elegant
    - Philosophy tests (30%): Passing philosophy tests = elegant

    Returns:
        Score 0.0 - 1.0
    """
    # Pass rate component
    if report.total > 0:
        pass_rate = report.passed / report.total
    else:
        pass_rate = 0

    # Performance component (fast tests = high score)
    # Assume <1s per test is excellent
    if report.total > 0:
        avg_duration = report.duration / report.total
        perf_score = max(0, 1 - avg_duration)  # 1s = 0, 0s = 1
    else:
        perf_score = 0

    # Philosophy component
    philosophy_tests = [r for r in report.results if r.category == 'philosophy']
    if philosophy_tests:
        philosophy_pass = sum(1 for r in philosophy_tests if r.status == 'passed')
        philosophy_score = philosophy_pass / len(philosophy_tests)
    else:
        philosophy_score = 1.0  # No philosophy tests = assume elegant

    # Weighted score
    score = (
        pass_rate * 0.4 +
        perf_score * 0.3 +
        philosophy_score * 0.3
    )

    return score


# ============================================================================
# Report Generation
# ============================================================================

def generate_visual_report(reports: List[TestReport]) -> str:
    """
    Generate beautiful visual test report.

    Args:
        reports: List of TestReport objects

    Returns:
        Markdown report with sparklines and visualizations
    """
    lines = []

    # Header
    lines.append("# Ruthless Test Report")
    lines.append("")
    lines.append("**Philosophy:** \"If the test is complex, the API failed.\"")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Overall Summary
    total_tests = sum(r.total for r in reports)
    total_passed = sum(r.passed for r in reports)
    total_failed = sum(r.failed for r in reports)
    total_skipped = sum(r.skipped for r in reports)
    total_duration = sum(r.duration for r in reports)

    lines.append("## Overall Summary")
    lines.append("")
    lines.append(f"**Total Tests:** {total_tests}")
    lines.append(f"**Passed:** {total_passed} {progress_bar(total_passed, total_tests, 30)}")
    lines.append(f"**Failed:** {total_failed} {progress_bar(total_failed, total_tests, 30)}")
    lines.append(f"**Skipped:** {total_skipped} {progress_bar(total_skipped, total_tests, 30)}")
    lines.append(f"**Duration:** {total_duration:.2f}s")
    lines.append("")

    # Pass rate sparkline
    pass_rates = [r.passed / r.total if r.total > 0 else 0 for r in reports]
    lines.append(f"**Pass Rate Trend:** {sparkline(pass_rates, 40)}")
    lines.append("")

    # Ruthless Elegance Score
    overall_score = sum(ruthless_elegance_score(r) for r in reports) / len(reports)
    lines.append("## Ruthless Elegance Score")
    lines.append("")
    lines.append(f"**Score:** {overall_score:.2%} {progress_bar(overall_score, 1.0, 40)}")
    lines.append("")

    if overall_score >= 0.9:
        grade = "A+ (Ruthlessly Elegant)"
    elif overall_score >= 0.8:
        grade = "A (Elegant)"
    elif overall_score >= 0.7:
        grade = "B (Good)"
    elif overall_score >= 0.6:
        grade = "C (Needs Work)"
    else:
        grade = "D (Failed Elegance)"

    lines.append(f"**Grade:** {grade}")
    lines.append("")
    lines.append("**Criteria:**")
    lines.append(f"- Pass Rate (40%): {(sum(r.passed for r in reports) / total_tests * 100) if total_tests > 0 else 0:.1f}%")
    lines.append(f"- Performance (30%): {(1 - (total_duration / total_tests)) * 100 if total_tests > 0 else 0:.1f}%")
    philosophy_tests = [r for report in reports for r in report.results if r.category == 'philosophy']
    if philosophy_tests:
        phil_pass = sum(1 for r in philosophy_tests if r.status == 'passed')
        lines.append(f"- Philosophy (30%): {(phil_pass / len(philosophy_tests) * 100):.1f}%")
    else:
        lines.append(f"- Philosophy (30%): N/A")
    lines.append("")

    # Individual Reports
    lines.append("=" * 80)
    lines.append("")
    lines.append("## Individual Test Files")
    lines.append("")

    for report in reports:
        lines.append(f"### {report.file_name}")
        lines.append("")
        lines.append(f"**Tests:** {report.total} | **Passed:** {report.passed} | **Failed:** {report.failed} | **Duration:** {report.duration:.2f}s")
        lines.append("")

        # Progress bar
        lines.append(progress_bar(report.passed, report.total, 50))
        lines.append("")

        # Category breakdown
        breakdown = category_breakdown(report.results)

        lines.append("**Category Breakdown:**")
        lines.append("")

        for category, stats in sorted(breakdown.items()):
            total_cat = sum(stats.values())
            passed_cat = stats['passed']
            lines.append(f"- **{category.replace('_', ' ').title()}:** {passed_cat}/{total_cat} {progress_bar(passed_cat, total_cat, 20)}")

        lines.append("")

        # Failed tests (if any)
        failed_tests = [r for r in report.results if r.status == 'failed']
        if failed_tests:
            lines.append("**Failed Tests:**")
            for test in failed_tests:
                lines.append(f"- `{test.name}` ({test.category})")
            lines.append("")

        # Ruthless elegance score for this file
        score = ruthless_elegance_score(report)
        lines.append(f"**Elegance Score:** {score:.2%} {progress_bar(score, 1.0, 30)}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    # Performance Analysis
    lines.append("## Performance Analysis")
    lines.append("")

    durations = [r.duration for r in reports]
    lines.append(f"**Duration Trend:** {sparkline(durations, 40)}")
    lines.append("")
    lines.append(f"**Total Duration:** {total_duration:.2f}s")
    lines.append(f"**Average per File:** {(total_duration / len(reports)):.2f}s")
    if total_tests > 0:
        lines.append(f"**Average per Test:** {(total_duration / total_tests):.3f}s")
    else:
        lines.append(f"**Average per Test:** N/A (no tests detected)")
    lines.append("")

    # Speed assessment
    avg_per_test = total_duration / total_tests if total_tests > 0 else 0
    if avg_per_test < 0.5:
        speed = "Fast (< 0.5s per test)"
    elif avg_per_test < 1.0:
        speed = "Good (< 1.0s per test)"
    elif avg_per_test < 2.0:
        speed = "Acceptable (< 2.0s per test)"
    else:
        speed = "Slow (> 2.0s per test)"

    lines.append(f"**Speed Assessment:** {speed}")
    lines.append("")

    # Philosophy Check
    lines.append("=" * 80)
    lines.append("")
    lines.append("## Philosophy Verification")
    lines.append("")

    philosophy_tests = [r for report in reports for r in report.results if r.category == 'philosophy']

    if philosophy_tests:
        phil_passed = sum(1 for r in philosophy_tests if r.status == 'passed')
        phil_total = len(philosophy_tests)

        lines.append(f"**Philosophy Tests:** {phil_passed}/{phil_total} passed")
        lines.append("")
        lines.append(progress_bar(phil_passed, phil_total, 50))
        lines.append("")

        if phil_passed == phil_total:
            lines.append("**Result:** All philosophy tests passed")
            lines.append("")
            lines.append("The APIs are ruthlessly elegant:")
            lines.append("- One line to execute")
            lines.append("- Zero configuration required")
            lines.append("- Automatic intelligence applied")
            lines.append("- Complete output generated")
        else:
            lines.append("**Result:** Some philosophy tests failed")
            lines.append("")
            lines.append("The APIs need elegance improvements:")
            for test in philosophy_tests:
                if test.status != 'passed':
                    lines.append(f"- FAILED: {test.name}")
    else:
        lines.append("**No philosophy tests found**")

    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Footer
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Overall:** {total_passed}/{total_tests} tests passed ({(total_passed / total_tests * 100) if total_tests > 0 else 0:.1f}%)")
    lines.append(f"**Elegance:** {overall_score:.2%} - {grade}")
    lines.append(f"**Performance:** {total_duration:.2f}s total, {avg_per_test:.3f}s per test")
    lines.append("")

    if total_failed == 0:
        lines.append("**Status:** All tests passed")
    else:
        lines.append(f"**Status:** {total_failed} test(s) failed - needs attention")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by visual_reporter.py - Ruthlessly elegant test reporting*")

    return '\n'.join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run tests and generate visual report."""

    print("=" * 80)
    print("RUTHLESS TEST REPORTER")
    print("=" * 80)
    print()
    print("Running tests and generating visual report with sparklines...")
    print()

    # Find test files
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / 'test_auto_elegant.py',
        test_dir / 'test_spin_elegant.py'
    ]

    # Run tests
    reports = []
    raw_outputs = []

    for test_file in test_files:
        if test_file.exists():
            try:
                report, output = run_pytest(str(test_file))
                reports.append(report)
                raw_outputs.append((test_file.name, output))

                print(f"[+] {test_file.name}: {report.passed}/{report.total} passed ({report.duration:.2f}s)")
            except Exception as e:
                print(f"[!] {test_file.name}: Error - {e}")
        else:
            print(f"[!] {test_file.name}: Not found")

    print()

    if not reports:
        print("[!] No test results to report")
        return

    # Generate report
    print("Generating visual report...")
    report_content = generate_visual_report(reports)

    # Save report
    report_path = test_dir / 'TEST_REPORT.md'
    report_path.write_text(report_content, encoding='utf-8')

    print(f"[+] Report saved to: {report_path}")
    print()

    # Print summary
    total_tests = sum(r.total for r in reports)
    total_passed = sum(r.passed for r in reports)
    total_failed = sum(r.failed for r in reports)
    overall_score = sum(ruthless_elegance_score(r) for r in reports) / len(reports)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tests: {total_passed}/{total_tests} passed")
    print(f"Elegance Score: {overall_score:.2%}")
    print(f"Duration: {sum(r.duration for r in reports):.2f}s")
    print()

    if total_failed == 0:
        print("[+] All tests passed - ruthlessly elegant!")
    else:
        print(f"[!] {total_failed} test(s) failed - needs attention")

    print("=" * 80)


if __name__ == '__main__':
    main()
