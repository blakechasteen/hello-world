#!/usr/bin/env python3
"""
Demo 2: Thompson Sampling Convergence

Demonstrates Thompson Sampling bandit algorithm converging over 1000 queries.
Shows how the system learns optimal backend selection (SQL, Neo4j, Qdrant) based on outcomes.

Part of: Hybrid Query Routing Architecture - Part 1 (Proof-of-Concept Demos)
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BanditArm:
    """Single backend (arm) in the bandit"""
    name: str
    alpha: float  # Success parameter
    beta: float   # Failure parameter
    true_success_rate: float  # Ground truth (hidden from bandit)


class ThompsonBandit:
    """Thompson Sampling for backend selection"""

    def __init__(self, backends: List[str], true_rates: List[float]):
        """
        Initialize Thompson Sampling bandit

        Args:
            backends: Backend names ['sql', 'neo4j', 'qdrant']
            true_rates: True success rates for each backend (ground truth)
        """
        self.arms = [
            BanditArm(
                name=name,
                alpha=1.0,  # Prior: Beta(1, 1) = uniform distribution
                beta=1.0,
                true_success_rate=rate
            )
            for name, rate in zip(backends, true_rates)
        ]
        self.history = []

    def select(self) -> int:
        """Select backend via Thompson Sampling"""
        # Sample from each arm's Beta distribution
        samples = [
            np.random.beta(arm.alpha, arm.beta)
            for arm in self.arms
        ]

        # Select arm with highest sample
        selected_arm = int(np.argmax(samples))
        return selected_arm

    def update(self, arm_idx: int, success: bool):
        """Update bandit statistics based on outcome"""
        if success:
            self.arms[arm_idx].alpha += 1.0
        else:
            self.arms[arm_idx].beta += 1.0

    def execute_query(self, arm_idx: int) -> bool:
        """Simulate query execution (using true success rate)"""
        arm = self.arms[arm_idx]
        return random.random() < arm.true_success_rate

    def get_confidence_interval(self, arm_idx: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for an arm"""
        arm = self.arms[arm_idx]

        # For Beta distribution, we use percentiles
        lower = np.percentile(np.random.beta(arm.alpha, arm.beta, 10000), (1 - confidence) * 100 / 2)
        upper = np.percentile(np.random.beta(arm.alpha, arm.beta, 10000), 100 - (1 - confidence) * 100 / 2)

        return lower, upper

    def get_stats(self) -> dict:
        """Get current statistics for all arms"""
        stats = []
        for i, arm in enumerate(self.arms):
            expected = arm.alpha / (arm.alpha + arm.beta)
            lower, upper = self.get_confidence_interval(i)
            ci_width = upper - lower

            stats.append({
                'name': arm.name,
                'alpha': arm.alpha,
                'beta': arm.beta,
                'expected_reward': expected,
                'ci_lower': lower,
                'ci_upper': upper,
                'ci_width': ci_width,
                'total_pulls': int(arm.alpha + arm.beta - 2),  # -2 for prior
                'true_rate': arm.true_success_rate
            })

        return stats


def print_header(text: str):
    """Print section header"""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def print_stats_snapshot(iteration: int, bandit: ThompsonBandit):
    """Print current bandit statistics"""
    stats = bandit.get_stats()

    print(f"\nIteration {iteration}:")
    print(f"{'Backend':<10} {'Alpha':>8} {'Beta':>8} {'Expected':>10} {'CI Width':>10} {'Pulls':>8}")
    print("-" * 80)

    for s in stats:
        print(f"{s['name']:<10} {s['alpha']:>8.1f} {s['beta']:>8.1f} "
              f"{s['expected_reward']:>10.3f} {s['ci_width']:>10.3f} {s['total_pulls']:>8d}")


def plot_convergence(history: List[dict], max_width: int = 60):
    """Plot confidence interval width convergence (ASCII art)"""
    print_header("Confidence Interval Convergence")

    # Sample every 50 iterations for visualization
    sampled = [history[i] for i in range(0, len(history), 50)]

    max_ci_width = max(h['max_ci_width'] for h in sampled)

    print(f"\n{'Iteration':<12} {'Max CI Width':<15} {'Convergence'}")
    print("-" * 80)

    for h in sampled:
        width = h['max_ci_width']
        bar_length = int((width / max_ci_width) * max_width) if max_ci_width > 0 else 0
        bar = "#" * bar_length

        # Mark convergence threshold (0.15)
        status = "[CONVERGED]" if width < 0.15 else ""

        print(f"{h['iteration']:<12d} {width:<15.3f} {bar:<{max_width}} {status}")


def main():
    """Run Thompson Sampling convergence demo"""

    print("=" * 80)
    print("DEMO 2: Thompson Sampling Convergence")
    print("=" * 80)
    print()
    print("Simulating 1000 queries with 3 backends (SQL, Neo4j, Qdrant)")
    print("Demonstrating Thompson Sampling learns optimal backend selection")
    print()

    # Ground truth: SQL is best (0.90), Neo4j is good (0.75), Qdrant is okay (0.65)
    # In production, these would be learned from actual query outcomes
    TRUE_SUCCESS_RATES = {
        'sql': 0.90,      # 90% success rate (deterministic queries)
        'neo4j': 0.75,    # 75% success rate (relationship queries)
        'qdrant': 0.65    # 65% success rate (similarity queries)
    }

    print("Ground Truth (hidden from bandit):")
    for backend, rate in TRUE_SUCCESS_RATES.items():
        print(f"  {backend.upper():<10}: {rate:.1%} success rate")
    print()

    # Initialize bandit
    bandit = ThompsonBandit(
        backends=['sql', 'neo4j', 'qdrant'],
        true_rates=[TRUE_SUCCESS_RATES['sql'], TRUE_SUCCESS_RATES['neo4j'], TRUE_SUCCESS_RATES['qdrant']]
    )

    # Simulate 1000 queries
    NUM_ITERATIONS = 1000
    SNAPSHOT_INTERVALS = [1, 50, 100, 250, 500, 750, 1000]

    history = []
    converged_at = None

    print_header("Thompson Sampling Simulation")

    for i in range(1, NUM_ITERATIONS + 1):
        # Select backend
        arm_idx = bandit.select()

        # Execute query (simulate outcome)
        success = bandit.execute_query(arm_idx)

        # Update bandit
        bandit.update(arm_idx, success)

        # Track convergence
        stats = bandit.get_stats()
        max_ci_width = max(s['ci_width'] for s in stats)

        # Get best backend's CI width (more realistic convergence criterion)
        best_backend_idx = max(range(len(stats)), key=lambda i: stats[i]['expected_reward'])
        best_ci_width = stats[best_backend_idx]['ci_width']

        history.append({
            'iteration': i,
            'selected_arm': arm_idx,
            'success': success,
            'max_ci_width': max_ci_width,
            'best_ci_width': best_ci_width
        })

        # Check for convergence (best backend's CI width < 0.10)
        if converged_at is None and best_ci_width < 0.10:
            converged_at = i

        # Print snapshots
        if i in SNAPSHOT_INTERVALS:
            print_stats_snapshot(i, bandit)

    # Plot convergence
    plot_convergence(history)

    # Final statistics
    print_header("Final Results")

    final_stats = bandit.get_stats()

    print("\nFinal Backend Statistics:")
    print(f"{'Backend':<10} {'Pulls':>8} {'Expected':>10} {'True Rate':>10} {'Error':>10}")
    print("-" * 80)

    for s in final_stats:
        error = abs(s['expected_reward'] - s['true_rate'])
        print(f"{s['name']:<10} {s['total_pulls']:>8d} {s['expected_reward']:>10.3f} "
              f"{s['true_rate']:>10.3f} {error:>10.3f}")

    print()

    # Best backend
    best_backend = max(final_stats, key=lambda x: x['expected_reward'])
    print(f"Best Backend (learned): {best_backend['name'].upper()} "
          f"(expected: {best_backend['expected_reward']:.3f})")

    actual_best = max(final_stats, key=lambda x: x['true_rate'])
    print(f"Best Backend (actual):  {actual_best['name'].upper()} "
          f"(true: {actual_best['true_rate']:.3f})")

    # Convergence summary
    print()
    print(f"Converged at iteration: {converged_at if converged_at else 'DID NOT CONVERGE'}")
    print(f"Final max CI width:     {max(s['ci_width'] for s in final_stats):.3f}")

    # Validation
    print_header("VALIDATION GATE 1.2: Thompson Sampling Convergence")

    # Check convergence of BEST backend (not all backends)
    # Explanation: Thompson Sampling correctly exploits the best arm, so other arms
    # will have wide CI's. What matters is that we're confident about the best choice.
    best_ci_converged = best_backend['ci_width'] < 0.10
    learned_best = best_backend['name'] == actual_best['name']

    print(f"\nBest Backend CI Width: {best_backend['ci_width']:.3f} (target: <0.10)")
    print(f"All Backends Max CI:   {max(s['ci_width'] for s in final_stats):.3f}")
    print()
    print("Note: Thompson Sampling exploits the best arm, so other arms have wide CI's.")
    print("      This is correct behavior - we only need confidence about the best choice.")
    print()

    if best_ci_converged and converged_at and converged_at <= 500:
        print(f"[PASS] Best backend converged at iteration {converged_at} (target: <500)")
    elif best_ci_converged and converged_at:
        print(f"[PASS] Best backend converged at iteration {converged_at}")
        print("[WARN] Slower than expected (target: <500 iterations)")
    elif best_ci_converged:
        print(f"[PASS] Best backend CI width < 0.10 (converged)")
    else:
        print(f"[FAIL] Best backend CI width {best_backend['ci_width']:.3f} > 0.10 (not converged)")

    if learned_best:
        print(f"[PASS] Correctly identified best backend ({best_backend['name'].upper()})")
    else:
        print(f"[WARN] Identified {best_backend['name'].upper()} as best, but {actual_best['name'].upper()} is optimal")

    print()

    # Alpha/Beta evolution
    print_header("Alpha/Beta Evolution (Final)")

    for s in final_stats:
        total = s['alpha'] + s['beta']
        success_rate = s['alpha'] / total if total > 0 else 0.5

        print(f"\n{s['name'].upper()}:")
        print(f"  Alpha (successes):  {s['alpha']:.1f}")
        print(f"  Beta (failures):    {s['beta']:.1f}")
        print(f"  Learned rate:       {success_rate:.3f}")
        print(f"  True rate:          {s['true_rate']:.3f}")
        print(f"  Error:              {abs(success_rate - s['true_rate']):.3f}")

    # Recommendations
    print_header("Next Steps")

    if best_ci_converged and learned_best:
        print("[READY] Thompson Sampling convergence validated")
        print("[READY] Proceed to Demo 3: SQL Schema + Mock Queries")
        print("[READY] Begin planning BackendBandit implementation (Part 4)")
    elif best_ci_converged:
        print("[WARN] Convergence achieved but may need tuning")
        print("[TODO] Investigate why best backend not identified correctly")
        print("[TODO] Consider adjusting priors or success rate definitions")
    else:
        print("[TODO] Increase iterations or adjust prior distributions")
        print("[TODO] Revisit Thompson Sampling implementation")
        print("[TODO] May need epsilon-greedy exploration for broader sampling")

    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    main()
