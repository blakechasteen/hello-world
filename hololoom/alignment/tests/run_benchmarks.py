"""
Standalone benchmark runner - generates performance report without pytest.

This script directly measures alignment framework latencies to verify <3ms overhead claim.
"""

import sys
import time
import statistics
import tempfile
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.alignment.safety_guardrails import (
    create_guardrails,
    ActionRequest,
    ActionCategory,
)
from HoloLoom.alignment.deception_detection import (
    create_detector,
    GoalStatement,
    BehavioralProbe,
    ProbeType,
)
from HoloLoom.alignment.instrumental_convergence import (
    create_guard,
    ResourceBounds,
    ResourceType,
)
from HoloLoom.alignment.audit_trail import (
    create_audit_trail,
    DecisionType,
    OutcomeType,
)


def measure_latency(func, iterations: int = 1000) -> Dict[str, float]:
    """Measure function latency over multiple iterations."""
    latencies = []

    # Warm-up
    for _ in range(100):
        func()

    # Measurement
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies.sort()

    return {
        'median': statistics.median(latencies),
        'p95': latencies[int(0.95 * len(latencies))],
        'p99': latencies[int(0.99 * len(latencies))],
        'mean': statistics.mean(latencies),
        'std': statistics.stdev(latencies),
    }


def main():
    print("\n" + "="*70)
    print("ALIGNMENT FRAMEWORK PERFORMANCE BENCHMARKS")
    print("="*70)
    print("\nTarget: <3ms total overhead per query")
    print("Methodology: 1000 iterations per component (median latency)\n")

    # Create instances
    guardrails = create_guardrails()
    detector = create_detector()
    guard = create_guard()

    with tempfile.TemporaryDirectory() as tmpdir:
        audit = create_audit_trail(persist_path=Path(tmpdir), auto_flush=False)

        # Setup
        goal = GoalStatement(goal_id="helpful", description="Be helpful", priority=10)
        detector.goal_tracker.declare_goal(goal)

        bounds = ResourceBounds(
            resource_type=ResourceType.MEMORY,
            soft_limit=1000.0,
            hard_limit=2000.0,
            time_window_seconds=60.0
        )
        guard.set_resource_bounds(ResourceType.MEMORY, bounds)

        # Benchmark each component
        print("Running benchmarks (this may take a minute)...\n")

        # 1. SafetyGuardrails
        print("1/4 Benchmarking SafetyGuardrails...")
        request = ActionRequest(action_id="What is AI safety?", category=ActionCategory.QUERY)
        results_guardrails = measure_latency(
            lambda: guardrails.evaluate(request, text_input=request.action_id)
        )

        # 2. DeceptionDetector
        print("2/4 Benchmarking DeceptionDetector...")
        probe = BehavioralProbe(ProbeType.GOAL_ALIGNMENT, "Test alignment", "Expected")
        results_detector = measure_latency(
            lambda: detector.run_probe(probe, "Aligned response")
        )

        # 3. InstrumentalGuard
        print("3/4 Benchmarking InstrumentalGuard...")
        results_guard = measure_latency(
            lambda: guard.check_resource_usage(ResourceType.MEMORY, 500.0)
        )

        # 4. AuditTrail
        print("4/4 Benchmarking AuditTrail...")
        results_audit = measure_latency(
            lambda: audit.log_decision(
                DecisionType.SAFETY_GATE,
                OutcomeType.APPROVED,
                "Test decision"
            )
        )

        # Print results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\n{'Component':<25} {'Median':<12} {'P95':<12} {'P99':<12} {'Threshold':<12} {'Status':<10}")
        print("-" * 85)

        components = [
            ("SafetyGuardrails", results_guardrails, 0.5),
            ("DeceptionDetector", results_detector, 1.0),
            ("InstrumentalGuard", results_guard, 0.3),
            ("AuditTrail", results_audit, 0.2),
        ]

        total_median = 0.0
        all_passed = True

        for name, results, threshold in components:
            median = results['median']
            total_median += median

            status = "✅ PASS" if median <= threshold else "❌ FAIL"
            if median > threshold:
                all_passed = False

            print(f"{name:<25} {median:<12.3f} {results['p95']:<12.3f} "
                  f"{results['p99']:<12.3f} {threshold:<12.1f} {status:<10}")

        print("-" * 85)
        total_status = "✅ PASS" if total_median <= 3.0 else "❌ FAIL"
        print(f"{'TOTAL OVERHEAD':<25} {total_median:<12.3f} {'':<12} {'':<12} {'3.0':<12} {total_status:<10}")

        print("\n" + "="*70)
        if all_passed and total_median <= 3.0:
            print("✅ ALL BENCHMARKS PASSED")
            print(f"\nTotal overhead: {total_median:.3f} ms (target: <3.0 ms)")
            print(f"Headroom: {3.0 - total_median:.3f} ms ({(3.0 - total_median) / 3.0 * 100:.1f}%)")
        else:
            print("❌ SOME BENCHMARKS FAILED")
            print(f"\nTotal overhead: {total_median:.3f} ms exceeds 3.0 ms threshold")

        print("="*70 + "\n")

        # Detailed breakdown
        print("DETAILED BREAKDOWN")
        print("-" * 70)
        for name, results, _ in components:
            print(f"\n{name}:")
            print(f"  Median: {results['median']:.3f} ms")
            print(f"  Mean:   {results['mean']:.3f} ms ± {results['std']:.3f} ms")
            print(f"  P95:    {results['p95']:.3f} ms")
            print(f"  P99:    {results['p99']:.3f} ms")

        return 0 if (all_passed and total_median <= 3.0) else 1


if __name__ == "__main__":
    sys.exit(main())
