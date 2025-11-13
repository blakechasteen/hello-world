"""
Demo: Thompson Sampling (Phase 4)
==================================

🐷 Adaptive Strategy Selection Demo 🐷

Demonstrates Phase 4 features:
1. Thompson Sampling for strategy selection
2. Automatic learning from outcomes
3. Confidence calibration
4. Learning curve visualization
5. Convergence detection

Run with:
    python xterminator/demo_moonshot_phase4.py
"""

import asyncio
import random
from dataclasses import dataclass

from xterminator import (
    MoonshotOrchestrator,
    AutofixPolicy,
    FixStrategy,
    FixOutcome,
    RiskLevel,
    SelectionMode,
    create_thompson_bandit,
    create_confidence_calibrator
)


# Mock Issue for testing
@dataclass
class MockIssue:
    """Mock SlopIssue for demo"""
    category: str
    description: str
    line: int
    severity: str = "medium"


# Mock success rates for each strategy (ground truth)
GROUND_TRUTH_SUCCESS_RATES = {
    FixStrategy.AST: 0.85,       # AST is best (85% success)
    FixStrategy.TEMPLATE: 0.70,  # Template is okay (70% success)
    FixStrategy.MANUAL: 0.50     # Manual is worst (50% success)
}


def simulate_fix_outcome(strategy: FixStrategy, confidence: float) -> tuple:
    """
    Simulate fix outcome based on strategy.

    Returns:
        (success, actual_confidence)
    """
    success_rate = GROUND_TRUTH_SUCCESS_RATES[strategy]

    # Success if random draw beats (1 - success_rate)
    success = random.random() < success_rate

    # Actual confidence varies slightly from predicted
    actual_confidence = confidence + random.gauss(0, 0.05)
    actual_confidence = max(0.0, min(1.0, actual_confidence))

    return success, actual_confidence


async def demo_scenario_1_basic_thompson():
    """
    Scenario 1: Basic Thompson Sampling

    Shows how Thompson Sampling selects strategies and learns from outcomes.
    """
    print("=" * 80)
    print("         🐷 SCENARIO 1: Basic Thompson Sampling 🐷")
    print("=" * 80)
    print("Thompson Sampling learns which strategies work best...")
    print()

    # Create bandit
    bandit = create_thompson_bandit(mode=SelectionMode.THOMPSON)

    # Show initial state (all strategies equal)
    print("Initial state (uniform priors):")
    for strategy in [FixStrategy.AST, FixStrategy.TEMPLATE, FixStrategy.MANUAL]:
        arm = bandit.arms[strategy]
        print(f"  {strategy.value:8s}: α={arm.alpha:.1f}, β={arm.beta:.1f}, "
              f"E[reward]={arm.expected_reward():.3f}")
    print()

    # Simulate 30 fix attempts
    print("Simulating 30 fix attempts...")
    for i in range(30):
        # Select strategy using Thompson Sampling
        strategy = bandit.select_strategy()

        # Simulate outcome
        confidence = 0.85
        success, actual_conf = simulate_fix_outcome(strategy, confidence)

        # Update bandit
        bandit.update(
            strategy=strategy,
            success=success,
            reward=actual_conf
        )

        if (i + 1) % 10 == 0:
            print(f"  After {i+1} attempts:")
            for s in [FixStrategy.AST, FixStrategy.TEMPLATE, FixStrategy.MANUAL]:
                arm = bandit.arms[s]
                print(f"    {s.value:8s}: pulls={arm.total_pulls:2d}, "
                      f"success={arm.success_rate:.2f}, E[reward]={arm.expected_reward():.3f}")
            print()

    # Show final state
    print("Final state after 30 attempts:")
    for strategy in [FixStrategy.AST, FixStrategy.TEMPLATE, FixStrategy.MANUAL]:
        arm = bandit.arms[strategy]
        print(f"  {strategy.value:8s}: α={arm.alpha:.1f}, β={arm.beta:.1f}, "
              f"E[reward]={arm.expected_reward():.3f}, success={arm.success_rate:.2f}")

    best_strategy, best_reward = bandit.get_best_strategy()
    print(f"\n✓ Best strategy: {best_strategy.value} (expected reward: {best_reward:.3f})")
    print(f"  Ground truth: AST has highest success rate (0.85)")
    print()


async def demo_scenario_2_learning_curve():
    """
    Scenario 2: Learning Curve

    Shows how the bandit improves over time.
    """
    print("=" * 80)
    print("         🐷 SCENARIO 2: Learning Curve 🐷")
    print("=" * 80)
    print("Tracking performance improvement over time...")
    print()

    # Create bandit
    bandit = create_thompson_bandit(mode=SelectionMode.THOMPSON)

    # Simulate 100 fix attempts
    print("Simulating 100 fix attempts...")
    for i in range(100):
        strategy = bandit.select_strategy()
        confidence = 0.85
        success, actual_conf = simulate_fix_outcome(strategy, confidence)
        bandit.update(strategy=strategy, success=success, reward=actual_conf)

    # Get learning curve
    curve = bandit.get_learning_curve(window_size=20)

    # Show key points
    print(f"Total selections: {len(curve)}")
    print()

    print("Learning trajectory (every 20 attempts):")
    for i in [19, 39, 59, 79, 99]:
        if i < len(curve):
            point = curve[i]
            print(f"  Attempt {i+1:3d}: strategy={point['strategy']:8s}, "
                  f"rolling_success={point['rolling_success_rate']:.2f}")

    print()

    # Strategy distribution
    strategy_counts = {}
    for point in curve:
        s = point['strategy']
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    print("Strategy selection distribution:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(curve) * 100
        print(f"  {strategy:8s}: {count:3d} times ({pct:5.1f}%)")

    print()

    # Final performance
    final_success_rate = curve[-1]['rolling_success_rate']
    print(f"✓ Final rolling success rate (last 20): {final_success_rate:.2f}")
    print(f"  Theoretical maximum (always pick AST): 0.85")
    print()


async def demo_scenario_3_confidence_calibration():
    """
    Scenario 3: Confidence Calibration

    Shows how confidence calibration tracks predicted vs actual.
    """
    print("=" * 80)
    print("         🐷 SCENARIO 3: Confidence Calibration 🐷")
    print("=" * 80)
    print("Tracking confidence calibration...")
    print()

    # Create calibrator
    calibrator = create_confidence_calibrator(num_bins=10)

    # Simulate 100 fix attempts with varying confidence
    print("Simulating 100 fix attempts...")
    for i in range(100):
        # Random confidence
        confidence = random.uniform(0.6, 0.95)

        # Simulate outcome (higher confidence → higher success chance)
        # Add some noise to make calibration interesting
        success_prob = confidence + random.gauss(0, 0.1)
        success_prob = max(0.0, min(1.0, success_prob))
        success = random.random() < success_prob

        # Add to calibrator
        calibrator.add_sample(confidence, success)

    # Get calibration summary
    summary = calibrator.get_calibration_summary()

    print(f"Total samples: {summary['total_samples']}")
    print(f"ECE (Expected Calibration Error): {summary['ece']:.4f}")
    print(f"Calibration quality: {summary['calibration_quality']}")
    print()

    # Show calibration curve
    print("Calibration curve:")
    print("  Bin Range    | Samples | Avg Conf | Accuracy | Cal Error")
    print("  " + "-" * 60)
    for point in summary['calibration_curve']:
        print(f"  {point['bin_range']:12s} | {point['count']:7d} | "
              f"{point['confidence']:8.2f} | {point['accuracy']:8.2f} | "
              f"{point['calibration_error']:9.3f}")

    print()

    # Show overconfident/underconfident bins
    if summary['overconfident_bins']:
        print(f"⚠  Overconfident bins: {len(summary['overconfident_bins'])}")
        for bin_stats in summary['overconfident_bins']:
            print(f"   {bin_stats['bin_range']}: predicted {bin_stats['avg_confidence']:.2f}, "
                  f"actual {bin_stats['accuracy']:.2f}")

    if summary['underconfident_bins']:
        print(f"⚠  Underconfident bins: {len(summary['underconfident_bins'])}")
        for bin_stats in summary['underconfident_bins']:
            print(f"   {bin_stats['bin_range']}: predicted {bin_stats['avg_confidence']:.2f}, "
                  f"actual {bin_stats['accuracy']:.2f}")

    if not summary['overconfident_bins'] and not summary['underconfident_bins']:
        print("✓ Well calibrated across all bins!")

    print()

    # Show recommendations
    if summary['recommendations']:
        print("Recommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
        print()


async def demo_scenario_4_convergence():
    """
    Scenario 4: Convergence Detection

    Shows when Thompson Sampling converges to optimal strategy.
    """
    print("=" * 80)
    print("         🐷 SCENARIO 4: Convergence Detection 🐷")
    print("=" * 80)
    print("Detecting when Thompson Sampling converges...")
    print()

    # Create bandit
    bandit = create_thompson_bandit(mode=SelectionMode.THOMPSON)

    # Simulate until convergence (max 200 attempts)
    print("Simulating fix attempts until convergence...")
    max_attempts = 200
    converged = False
    converged_at = None

    for i in range(max_attempts):
        strategy = bandit.select_strategy()
        confidence = 0.85
        success, actual_conf = simulate_fix_outcome(strategy, confidence)
        bandit.update(strategy=strategy, success=success, reward=actual_conf)

        # Check convergence every 20 attempts (after first 100)
        if (i + 1) >= 100 and (i + 1) % 20 == 0:
            is_converged, details = bandit.detect_convergence(window_size=100, threshold=0.05)

            if is_converged and not converged:
                converged = True
                converged_at = i + 1
                print(f"✓ Converged at attempt {converged_at}!")
                print(f"  Dominant strategy: {details['dominant_strategy']}")
                print(f"  Dominant fraction: {details['dominant_fraction']:.2%}")
                print(f"  Rolling variance: {details['rolling_variance']:.4f}")
                print(f"  Recent success rate: {details['recent_success_rate']:.2%}")
                print()
                break

    if not converged:
        print(f"⚠  Did not converge after {max_attempts} attempts")
        print()

    # Show final strategy distribution
    total_selections = bandit.total_selections
    print(f"Total selections: {total_selections}")
    print("Strategy distribution:")
    for strategy in [FixStrategy.AST, FixStrategy.TEMPLATE, FixStrategy.MANUAL]:
        arm = bandit.arms[strategy]
        pct = arm.total_pulls / total_selections * 100
        print(f"  {strategy.value:8s}: {arm.total_pulls:3d} times ({pct:5.1f}%), "
              f"success={arm.success_rate:.2%}")

    print()


async def demo_scenario_5_integrated():
    """
    Scenario 5: Integrated with MoonshotOrchestrator

    Shows Thompson Sampling integrated with full pipeline (simulated).
    """
    print("=" * 80)
    print("         🐷 SCENARIO 5: Integrated Pipeline 🐷")
    print("=" * 80)
    print("Thompson Sampling integrated with MoonshotOrchestrator...")
    print()

    # Create orchestrator with Thompson Sampling enabled
    print("Creating orchestrator with Thompson Sampling...")
    orchestrator = MoonshotOrchestrator(
        policy=AutofixPolicy.balanced(),
        enable_feedback=True,
        enable_thompson_sampling=True,
        thompson_mode=SelectionMode.THOMPSON
    )

    print("✓ Orchestrator created with Thompson Sampling enabled")
    print()

    # Simulate 50 fix attempts (in real usage, would process actual issues)
    print("Simulating 50 fix attempts...")
    print("(In production, these would be real code issues)")
    print()

    for i in range(50):
        # Simulate: Thompson selects strategy, outcome is determined
        strategy = orchestrator.thompson_bandit.select_strategy()
        confidence = random.uniform(0.75, 0.95)
        success, actual_conf = simulate_fix_outcome(strategy, confidence)

        # Update bandit (in production, this happens automatically)
        orchestrator.thompson_bandit.update(
            strategy=strategy,
            success=success,
            reward=actual_conf
        )

        # Update calibration
        orchestrator.confidence_calibrator.add_sample(
            confidence=confidence,
            success=success
        )

    # Get performance summary
    print("Thompson Sampling performance:")
    perf = orchestrator.get_thompson_performance()
    for strategy_name, stats in perf['per_strategy'].items():
        print(f"  {strategy_name:8s}: pulls={stats['total_pulls']:2d}, "
              f"success={stats['success_rate']:.2%}, E[reward]={stats['expected_reward']:.3f}")

    best_strategy, best_reward = orchestrator.get_best_strategy()
    print(f"\n✓ Best strategy: {best_strategy.value} (E[reward]={best_reward:.3f})")
    print()

    # Get calibration summary
    print("Confidence calibration:")
    cal_summary = orchestrator.get_confidence_calibration()
    print(f"  ECE: {cal_summary['ece']:.4f}")
    print(f"  Quality: {cal_summary['calibration_quality']}")
    print(f"  Samples: {cal_summary['total_samples']}")
    print()

    # Get learning curve
    curve = orchestrator.get_learning_curve(window_size=20)
    if len(curve) >= 20:
        print(f"Learning trajectory:")
        print(f"  First 20: success rate = {curve[19]['rolling_success_rate']:.2%}")
        print(f"  Last 20:  success rate = {curve[-1]['rolling_success_rate']:.2%}")
        improvement = curve[-1]['rolling_success_rate'] - curve[19]['rolling_success_rate']
        print(f"  Improvement: {improvement:+.2%}")
        print()


async def main():
    """Run all demo scenarios"""
    print()
    print("=" * 80)
    print("           🐷 xTerminator Moonshot Phase 4 Demo 🐷")
    print("                Thompson Sampling + Calibration")
    print("=" * 80)
    print()

    await demo_scenario_1_basic_thompson()
    await demo_scenario_2_learning_curve()
    await demo_scenario_3_confidence_calibration()
    await demo_scenario_4_convergence()
    await demo_scenario_5_integrated()

    print("=" * 80)
    print("                      🐷 Demo Complete! 🐷")
    print("=" * 80)
    print()
    print("Phase 4 Features Demonstrated:")
    print("  ✓ Thompson Sampling bandit")
    print("  ✓ Automatic α, β updates")
    print("  ✓ Learning curve analysis")
    print("  ✓ Confidence calibration (ECE)")
    print("  ✓ Convergence detection")
    print("  ✓ Integrated pipeline")
    print()
    print("Key Insights:")
    print("  • Thompson Sampling learns optimal strategy (AST) over time")
    print("  • Success rate improves with more attempts")
    print("  • Confidence calibration detects over/underconfidence")
    print("  • System converges to ~80% selection of best strategy")
    print()
    print("Next: Phase 5 (Marketplace Quality Enforcement)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
