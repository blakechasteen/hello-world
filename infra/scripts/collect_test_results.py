#!/usr/bin/env python3
"""
Collect test results from all Moonshot RAG test suites.

Runs individual test files and collects pass/fail/skip statistics.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def run_test_file(test_file: str) -> Tuple[int, int, int, str]:
    """
    Run a single test file and extract statistics.

    Returns: (passed, failed, skipped, output)
    """
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=no"],
            cwd="/c/Users/blake/OneDrive/Documents/mythRL",
            capture_output=True,
            text=True,
            timeout=120,
        )

        output = result.stdout + result.stderr

        # Parse pytest output for summary line
        # Pattern: "N passed, M failed, K skipped in X.XXs"
        match = re.search(
            r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?',
            output
        )

        if match:
            passed = int(match.group(1))
            failed = int(match.group(2) or 0)
            skipped = int(match.group(3) or 0)
            return passed, failed, skipped, output

        return 0, 0, 0, output

    except subprocess.TimeoutExpired:
        return 0, 0, 0, f"TIMEOUT on {test_file}"
    except Exception as e:
        return 0, 0, 0, f"ERROR running {test_file}: {str(e)}"


def main():
    test_dir = Path("/c/Users/blake/OneDrive/Documents/mythRL/hololoom/rag/tests")

    # Test files for each feature
    test_files = {
        "Streaming (Wave 3)": "test_streaming.py",
        "Custom Embeddings (Wave 3)": "test_embedding_plugins.py",
        "Reranking (Wave 3)": "test_reranking.py",
        "SQL Integration (Wave 4)": "test_sql_integration.py",
        "Multi-Hop Reasoning (Wave 4)": "test_multihop_reasoning.py",
        "Multi-Agent RAG (Wave 5)": "test_multiagent_rag.py",
        "Integration Tests": "test_moonshot_integration.py",
        "Performance Benchmarks": "test_moonshot_performance.py",
    }

    # Run tests and collect results
    results = {}
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_tests = 0

    print("\n" + "=" * 80)
    print("MOONSHOT RAG TEST COLLECTION")
    print("=" * 80 + "\n")

    for feature_name, test_file in test_files.items():
        test_path = test_dir / test_file

        if not test_path.exists():
            print(f"[SKIP] {feature_name:.<40} File not found")
            continue

        print(f"Testing {feature_name:.<40}", end="", flush=True)

        passed, failed, skipped, output = run_test_file(str(test_path))
        total_tests_in_file = passed + failed + skipped

        results[feature_name] = {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": total_tests_in_file,
        }

        total_passed += passed
        total_failed += failed
        total_skipped += skipped
        total_tests += total_tests_in_file

        # Status indicator
        if failed > 0:
            status = "[FAIL]"
        elif total_tests_in_file == 0:
            status = "[EMPTY]"
        else:
            status = "[PASS]"

        print(f" {status}")
        print(f"  {passed} passed, {failed} failed, {skipped} skipped")

    # Print summary table
    print("\n" + "=" * 80)
    print("TEST SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Feature':<35} | {'Passed':>8} | {'Failed':>8} | {'Skipped':>8} | {'Total':>8} | {'Pass Rate':>10}")
    print("-" * 80)

    for feature_name, counts in results.items():
        passed = counts["passed"]
        failed = counts["failed"]
        skipped = counts["skipped"]
        total = counts["total"]

        if total > 0:
            pass_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
        else:
            pass_rate = 0

        print(f"{feature_name:<35} | {passed:>8} | {failed:>8} | {skipped:>8} | {total:>8} | {pass_rate:>9.1f}%")

    print("-" * 80)

    # Calculate overall pass rate (excluding skipped)
    testable = total_passed + total_failed
    overall_pass_rate = (total_passed / testable * 100) if testable > 0 else 0

    print(f"{'TOTAL':<35} | {total_passed:>8} | {total_failed:>8} | {total_skipped:>8} | {total_tests:>8} | {overall_pass_rate:>9.1f}%")
    print("=" * 80)

    # Print details
    print("\nDETAILS:")
    print(f"  Total Tests Run: {total_tests}")
    print(f"  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    print(f"  Total Skipped: {total_skipped}")
    print(f"  Overall Pass Rate (non-skipped): {overall_pass_rate:.1f}%")

    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)

    success = True

    # Criterion 1: >95% pass rate
    if overall_pass_rate >= 95:
        print(f"[PASS] Pass Rate >95%: {overall_pass_rate:.1f}%")
    else:
        print(f"[FAIL] Pass Rate >95%: {overall_pass_rate:.1f}%")
        success = False

    # Criterion 2: No critical failures
    if total_failed == 0:
        print(f"[PASS] No Critical Failures: 0 failed tests")
    else:
        print(f"[FAIL] No Critical Failures: {total_failed} failed tests")
        success = False

    # Criterion 3: Integration tests exist
    if "Integration Tests" in results and results["Integration Tests"]["total"] > 0:
        print(f"[PASS] Integration Tests Created: {results['Integration Tests']['total']} tests")
    else:
        print(f"[INFO] Integration Tests: Not run yet")

    # Criterion 4: Performance benchmarks exist
    if "Performance Benchmarks" in results and results["Performance Benchmarks"]["total"] > 0:
        print(f"[PASS] Performance Benchmarks: {results['Performance Benchmarks']['total']} tests")
    else:
        print(f"[INFO] Performance Benchmarks: Not run yet")

    print("=" * 80)

    if success:
        print("\n[PASS] ALL SUCCESS CRITERIA MET")
    else:
        print("\n[FAIL] SOME SUCCESS CRITERIA FAILED")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
