#!/usr/bin/env python3
"""
Test runner for Keep beekeeping application.

Runs comprehensive test suite and generates reports.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(verbose: bool = True, coverage: bool = False):
    """
    Run the test suite.

    Args:
        verbose: Show verbose output
        coverage: Generate coverage report
    """
    print("=" * 70)
    print("  Keep Test Suite Runner")
    print("=" * 70)

    tests_dir = Path(__file__).parent / "tests"

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=apps.keep", "--cov-report=term", "--cov-report=html"])

    # Add test directory
    cmd.append(str(tests_dir))

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)

    if coverage:
        print("\nCoverage report generated in htmlcov/index.html")

    return result.returncode


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Keep test suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-c", "--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet output")

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    return run_tests(verbose=verbose, coverage=args.coverage)


if __name__ == "__main__":
    sys.exit(main())
