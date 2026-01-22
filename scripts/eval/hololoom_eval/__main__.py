#!/usr/bin/env python3
"""
HoloLoom Evaluation Framework CLI

One-click comprehensive testing to evaluate if HoloLoom provides value.

Usage:
    python -m scripts.eval.hololoom_eval              # Standard evaluation
    python -m scripts.eval.hololoom_eval --quick      # Fast sanity check (~1 min)
    python -m scripts.eval.hololoom_eval --full       # Comprehensive (~10 min)
    python -m scripts.eval.hololoom_eval --json       # Output JSON format
    python -m scripts.eval.hololoom_eval --benchmark retrieval  # Run specific benchmark
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime

from .runner import EvaluationRunner, EvalMode


def print_banner():
    """Print startup banner."""
    print()
    print("=" * 70)
    print("  HoloLoom Evaluation Framework v1.0")
    print("  Testing: Does complexity provide value?")
    print("=" * 70)
    print()


def print_report(report, verbose: bool = False):
    """Print evaluation report in human-readable format."""

    # Overall result
    status = "✅ PASS" if report.overall_passed else "❌ FAIL"
    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {status}  (Score: {report.overall_score:.2f}/1.00)")
    print(f"{'=' * 70}\n")

    # Individual benchmarks
    print("BENCHMARK RESULTS:")
    print("-" * 70)

    for result in report.benchmarks:
        status_icon = "✅" if result.passed else "❌"
        print(f"  {status_icon} {result.name:<25} Score: {result.score:.2f}  {result.verdict}")

    # Dependencies
    print(f"\n{'DEPENDENCIES:'}")
    print("-" * 70)
    for dep, available in report.dependencies.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {dep}")

    # Summary
    print(f"\n{'SUMMARY:'}")
    print("-" * 70)
    print(f"  Mode: {report.mode}")
    print(f"  Duration: {report.duration_seconds:.1f}s")
    print(f"  Timestamp: {report.timestamp}")

    # Recommendations
    if not report.overall_passed:
        print(f"\n{'RECOMMENDATIONS:'}")
        print("-" * 70)

        missing_deps = [dep for dep, avail in report.dependencies.items() if not avail]
        if missing_deps:
            print(f"  1. Install missing dependencies:")
            print(f"     pip install {' '.join(missing_deps)}")

        failed = [r for r in report.benchmarks if not r.passed]
        if failed:
            print(f"  2. Failed benchmarks to investigate:")
            for r in failed:
                print(f"     - {r.name}: {r.verdict}")

    # Verbose details
    if verbose:
        print(f"\n{'DETAILED RESULTS:'}")
        print("-" * 70)
        for result in report.benchmarks:
            print(f"\n{result.name}:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Passed: {result.passed}")
            print(f"  Verdict: {result.verdict}")
            if result.details:
                print(f"  Details:")
                for key, value in result.details.items():
                    if isinstance(value, (dict, list)):
                        print(f"    {key}: {json.dumps(value, indent=6)[:200]}...")
                    else:
                        print(f"    {key}: {value}")

    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HoloLoom Evaluation Framework - Test if complexity provides value",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.eval.hololoom_eval              # Standard evaluation
  python -m scripts.eval.hololoom_eval --quick      # Fast sanity check (~1 min)
  python -m scripts.eval.hololoom_eval --full       # Comprehensive (~10 min)
  python -m scripts.eval.hololoom_eval --json       # JSON output
  python -m scripts.eval.hololoom_eval -v           # Verbose output

Benchmarks:
  - retrieval: Compares HoloLoom vs BM25/Vector/Hybrid baselines
  - learning:  Tests if system improves over time
  - ablation:  Measures value of each component
  - latency:   Validates performance targets
  - graph:     Tests knowledge graph value
  - cache:     Measures cache effectiveness
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick sanity check (~1 minute)"
    )
    mode_group.add_argument(
        "--full", "-f",
        action="store_true",
        help="Full comprehensive evaluation (~10 minutes)"
    )

    # Output format
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed results"
    )

    # Specific benchmark
    parser.add_argument(
        "--benchmark", "-b",
        choices=["retrieval", "learning", "ablation", "latency", "graph", "cache"],
        help="Run only a specific benchmark"
    )

    # Output file
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to file"
    )

    args = parser.parse_args()

    # Determine mode
    if args.quick:
        mode = EvalMode.QUICK
    elif args.full:
        mode = EvalMode.FULL
    else:
        mode = EvalMode.STANDARD

    # Print banner (unless JSON output)
    if not args.json:
        print_banner()
        print(f"Mode: {mode.value.upper()}")
        print(f"Started: {datetime.now().isoformat()}")
        print()

    # Create runner and execute
    runner = EvaluationRunner(mode=mode)

    try:
        if args.benchmark:
            # Run specific benchmark
            if not args.json:
                print(f"Running benchmark: {args.benchmark}")
            report = asyncio.run(runner.run_single_benchmark(args.benchmark))
        else:
            # Run all benchmarks
            if not args.json:
                print("Running all benchmarks...")
            report = asyncio.run(runner.run_all())

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Output results
    if args.json:
        output = report.to_dict()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Results also saved to {args.output}")

    # Exit code based on pass/fail
    sys.exit(0 if report.overall_passed else 1)


if __name__ == "__main__":
    main()
