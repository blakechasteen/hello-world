#!/usr/bin/env python3
"""
HoloLoom Evaluation Framework CLI v2.0

One-click comprehensive testing to evaluate if HoloLoom provides value.

Usage:
    python -m scripts.eval.hololoom_eval              # Standard evaluation
    python -m scripts.eval.hololoom_eval --quick      # Fast sanity check (~1 min)
    python -m scripts.eval.hololoom_eval --full       # Comprehensive (~10 min)
    python -m scripts.eval.hololoom_eval --json       # Output JSON format
    python -m scripts.eval.hololoom_eval --dashboard  # Generate HTML dashboard
    python -m scripts.eval.hololoom_eval --seed 42    # Reproducible run
    python -m scripts.eval.hololoom_eval --compare    # Compare with previous run
    python -m scripts.eval.hololoom_eval --triple-verify  # 3-pass stability verification
    python -m scripts.eval.hololoom_eval --benchmark retrieval  # Specific benchmark
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
    print("  HoloLoom Evaluation Framework v2.0")
    print("  Testing: Does complexity provide value?")
    print("=" * 70)
    print()


def print_report(report, verbose: bool = False):
    """Print evaluation report in human-readable format."""

    # Overall result
    status = "PASS" if report.overall_passed else "FAIL"
    print(f"\n{'=' * 70}")
    print(f"  OVERALL: {status}  (Score: {report.overall_score:.2f}/1.00)")
    print(f"{'=' * 70}\n")

    # Individual benchmarks
    print("BENCHMARK RESULTS:")
    print("-" * 70)

    for result in report.benchmarks:
        status_icon = "PASS" if result.passed else "FAIL"
        error_note = " [ERROR]" if result.error else ""
        print(f"  [{status_icon}] {result.name:<25} Score: {result.score:.2f}  {result.verdict}{error_note}")

    # Statistical summary (v2.0)
    if report.statistical_summary:
        print(f"\n{'STATISTICAL SUMMARY:'}")
        print("-" * 70)
        stats = report.statistical_summary
        if "overall" in stats:
            overall = stats["overall"]
            ci = overall.get("ci_95", {})
            print(f"  Overall Score: {overall.get('mean', 0):.3f} (95% CI: [{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}])")
            print(f"  Std Dev: {overall.get('std', 0):.3f}  |  N = {overall.get('n', 0)} benchmarks")

    # Dependencies
    print(f"\n{'DEPENDENCIES:'}")
    print("-" * 70)
    for dep, available in report.dependencies.items():
        icon = "[OK]" if available else "[--]"
        print(f"  {icon} {dep}")

    # Summary
    print(f"\n{'SUMMARY:'}")
    print("-" * 70)
    print(f"  Mode: {report.mode}")
    print(f"  Duration: {report.duration_seconds:.1f}s")
    print(f"  Timestamp: {report.timestamp}")

    # Recommendations
    if report.recommendations:
        print(f"\n{'RECOMMENDATIONS:'}")
        print("-" * 70)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

    # Verbose details
    if verbose:
        print(f"\n{'DETAILED RESULTS:'}")
        print("-" * 70)
        for result in report.benchmarks:
            print(f"\n{result.name}:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Passed: {result.passed}")
            print(f"  Verdict: {result.verdict}")
            if result.error:
                print(f"  Error: {result.error}")
            if result.details:
                print(f"  Details:")
                for key, value in result.details.items():
                    if isinstance(value, (dict, list)):
                        print(f"    {key}: {json.dumps(value, indent=6, default=str)[:200]}...")
                    else:
                        print(f"    {key}: {value}")

    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HoloLoom Evaluation Framework v2.0 - Test if complexity provides value",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.eval.hololoom_eval              # Standard evaluation
  python -m scripts.eval.hololoom_eval --quick      # Fast sanity check (~1 min)
  python -m scripts.eval.hololoom_eval --full       # Comprehensive (~10 min)
  python -m scripts.eval.hololoom_eval --json       # JSON output
  python -m scripts.eval.hololoom_eval --dashboard  # HTML dashboard report
  python -m scripts.eval.hololoom_eval --seed 42    # Reproducible run
  python -m scripts.eval.hololoom_eval --compare    # Compare with previous run
  python -m scripts.eval.hololoom_eval --triple-verify  # 3-pass stability check
  python -m scripts.eval.hololoom_eval -v           # Verbose output

Benchmarks (8 total):
  - retrieval: Compares HoloLoom vs BM25/Vector/Hybrid baselines
  - learning:  Tests if system improves over time
  - ablation:  Measures value of each component
  - latency:   Validates performance targets
  - graph:     Tests knowledge graph value
  - cache:     Measures cache effectiveness
  - stress:    Tests robustness to adversarial/edge-case queries
  - semantic:  Evaluates semantic quality beyond keyword matching
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
        choices=[
            "retrieval", "learning", "ablation", "latency",
            "graph", "cache", "stress", "semantic",
        ],
        help="Run only a specific benchmark"
    )

    # Output file
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to file"
    )

    # v2.0: Dashboard generation
    parser.add_argument(
        "--dashboard", "--report",
        action="store_true",
        dest="dashboard",
        help="Generate self-contained HTML dashboard report"
    )

    # v2.0: Reproducibility seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (e.g. --seed 42)"
    )

    # v2.0: Compare with previous run
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results with previous evaluation run"
    )

    # v2.1: Triple verification
    parser.add_argument(
        "--triple-verify",
        action="store_true",
        dest="triple_verify",
        help="Run each benchmark 3 times with different seeds to verify stability"
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
        if args.seed is not None:
            print(f"Seed: {args.seed}")
        if args.dashboard:
            print("Dashboard: enabled")
        if args.compare:
            print("Compare: enabled")
        if args.triple_verify:
            print("Triple Verify: enabled (3 passes per benchmark)")
        print(f"Started: {datetime.now().isoformat()}")
        print()

    # Create runner with v2.0+ options
    runner = EvaluationRunner(
        mode=mode,
        seed=args.seed,
        generate_dashboard=args.dashboard,
        compare_previous=args.compare,
        triple_verify=args.triple_verify,
    )

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
