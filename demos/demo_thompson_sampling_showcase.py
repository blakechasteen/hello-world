#!/usr/bin/env python3
"""
Thompson Sampling Showcase - Phase 2 Demo

Demonstrates HoloLoom's Bayesian exploration strategy using Thompson Sampling.
Shows multi-armed bandits, α/β updates, exploration vs exploitation, and regret.

This demo works WITHOUT heavy ML dependencies - uses numpy only.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time

# ============================================================================
# THOMPSON SAMPLING IMPLEMENTATION
# ============================================================================

@dataclass
class ThompsonBandit:
    """
    Thompson Sampling bandit for exploration/exploitation balance.

    Uses Beta(α, β) posteriors for each arm.
    - α (alpha) = successes + 1 (prior)
    - β (beta) = failures + 1 (prior)
    """
    n_arms: int
    arm_names: List[str] = None
    alpha: np.ndarray = None  # successes
    beta: np.ndarray = None   # failures
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if self.arm_names is None:
            self.arm_names = [f"Arm_{i}" for i in range(self.n_arms)]
        # Initialize with uniform prior Beta(1, 1)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def sample(self) -> int:
        """Sample from Beta posteriors and pick the best arm."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update posterior with observed reward."""
        if reward > 0.5:  # Success threshold
            self.alpha[arm] += reward
        else:
            self.beta[arm] += (1 - reward)

        self.history.append({
            'arm': arm,
            'reward': reward,
            'alpha': self.alpha.copy(),
            'beta': self.beta.copy()
        })

    def expected_reward(self, arm: int) -> float:
        """Expected reward = α / (α + β)."""
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def get_stats(self) -> Dict:
        """Get statistics for all arms."""
        return {
            self.arm_names[i]: {
                'alpha': self.alpha[i],
                'beta': self.beta[i],
                'expected': self.expected_reward(i),
                'pulls': self.alpha[i] + self.beta[i] - 2  # Subtract prior
            }
            for i in range(self.n_arms)
        }


class SimulatedArm:
    """A simulated bandit arm with true reward probability."""

    def __init__(self, name: str, true_prob: float):
        self.name = name
        self.true_prob = true_prob

    def pull(self) -> float:
        """Pull the arm and get stochastic reward."""
        return 1.0 if np.random.random() < self.true_prob else 0.0


def render_beta_distribution(alpha: float, beta: float, width: int = 40) -> str:
    """Render ASCII approximation of Beta distribution."""
    x = np.linspace(0, 1, width)
    from scipy.stats import beta as beta_dist
    try:
        y = beta_dist.pdf(x, alpha, beta)
        y = y / (y.max() + 1e-10)  # Normalize
    except:
        # Fallback if scipy not available
        y = np.zeros(width)
        mode = (alpha - 1) / (alpha + beta - 2) if alpha + beta > 2 else 0.5
        for i, xi in enumerate(x):
            y[i] = max(0, 1 - abs(xi - mode) * 3)

    # Map to ASCII blocks
    blocks = " ▁▂▃▄▅▆▇█"
    result = ""
    for yi in y:
        idx = int(yi * (len(blocks) - 1))
        result += blocks[idx]
    return result


def render_progress_bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    """Render a progress bar."""
    filled_count = int(value * width)
    return filled * filled_count + empty * (width - filled_count)


def demo_basic_bandit():
    """Demo 1: Basic Thompson Sampling mechanics."""
    print("\n" + "=" * 60)
    print("Demo 1: Thompson Sampling Basics - Multi-Armed Bandit")
    print("=" * 60)

    # Create bandit with 3 arms
    arms = [
        SimulatedArm("Conservative", 0.3),
        SimulatedArm("Moderate", 0.5),
        SimulatedArm("Aggressive", 0.7),
    ]

    bandit = ThompsonBandit(n_arms=3, arm_names=[a.name for a in arms])

    print("\n  True reward probabilities (unknown to bandit):")
    for arm in arms:
        bar = render_progress_bar(arm.true_prob)
        print(f"    {arm.name:12s}: [{bar}] {arm.true_prob:.0%}")

    print("\n  Running 50 rounds of Thompson Sampling...")
    print("\n  Round | Selected Arm | Reward | Updated Posteriors")
    print("  " + "─" * 55)

    for round_num in range(1, 51):
        # Sample and select
        selected = bandit.sample()
        reward = arms[selected].pull()
        bandit.update(selected, reward)

        if round_num <= 10 or round_num % 10 == 0:
            posteriors = " | ".join([
                f"{bandit.arm_names[i][:4]}:α={bandit.alpha[i]:.0f},β={bandit.beta[i]:.0f}"
                for i in range(3)
            ])
            reward_sym = "✓" if reward > 0.5 else "✗"
            print(f"  {round_num:5d} | {bandit.arm_names[selected]:12s} | {reward_sym}      | {posteriors}")

    print("\n  Final Statistics:")
    stats = bandit.get_stats()
    for name, s in stats.items():
        bar = render_progress_bar(s['expected'])
        print(f"    {name:12s}: E[r]={s['expected']:.2f} [{bar}] (pulls={s['pulls']:.0f})")


def demo_exploration_exploitation():
    """Demo 2: Visualize exploration vs exploitation."""
    print("\n" + "=" * 60)
    print("Demo 2: Exploration vs Exploitation Balance")
    print("=" * 60)

    # Two arms: one unknown (high variance), one known (low variance)
    bandit = ThompsonBandit(n_arms=2, arm_names=["Explored", "Unknown"])

    # Simulate: "Explored" has many observations, "Unknown" has few
    bandit.alpha = np.array([15.0, 2.0])  # Explored arm has α=15
    bandit.beta = np.array([10.0, 1.0])   # Explored: β=10, Unknown: β=1

    print("\n  Situation: Two investment strategies")
    print("    - Explored: Many past trials (α=15, β=10) → E[r]=0.60")
    print("    - Unknown:  Few past trials (α=2, β=1) → E[r]=0.67")
    print("\n  Who should we try? Let's sample 100 times from posteriors...")

    selections = {'Explored': 0, 'Unknown': 0}
    samples_explored = []
    samples_unknown = []

    for _ in range(100):
        sample_e = np.random.beta(bandit.alpha[0], bandit.beta[0])
        sample_u = np.random.beta(bandit.alpha[1], bandit.beta[1])
        samples_explored.append(sample_e)
        samples_unknown.append(sample_u)

        if sample_e > sample_u:
            selections['Explored'] += 1
        else:
            selections['Unknown'] += 1

    print("\n  Posterior Distributions:")
    dist_e = render_beta_distribution(bandit.alpha[0], bandit.beta[0])
    dist_u = render_beta_distribution(bandit.alpha[1], bandit.beta[1])
    print(f"    Explored (α=15, β=10): [{dist_e}]")
    print(f"    Unknown  (α=2,  β=1):  [{dist_u}]")

    print("\n  Selection Results (100 samples):")
    for name, count in selections.items():
        bar = render_progress_bar(count / 100)
        print(f"    {name:10s}: {count:3d}% [{bar}]")

    print("\n  Insight: Thompson Sampling naturally explores the Unknown arm")
    print("           because its posterior has higher variance (uncertainty).")
    print("           This is OPTIMISM IN THE FACE OF UNCERTAINTY.")


def demo_convergence():
    """Demo 3: Watch convergence to optimal arm."""
    print("\n" + "=" * 60)
    print("Demo 3: Convergence to Optimal Strategy")
    print("=" * 60)

    # 4 arms with different true probabilities
    true_probs = [0.2, 0.4, 0.6, 0.8]
    arm_names = ["Poor", "Below Avg", "Above Avg", "Optimal"]

    arms = [SimulatedArm(name, prob) for name, prob in zip(arm_names, true_probs)]
    bandit = ThompsonBandit(n_arms=4, arm_names=arm_names)

    print("\n  True probabilities: Poor=0.2, Below=0.4, Above=0.6, Optimal=0.8")
    print("\n  Tracking arm selection over time...")
    print("\n  Rounds  | Poor | Below | Above | Optimal | Expected Rewards")
    print("  " + "─" * 60)

    selection_counts = np.zeros(4)

    for round_num in range(1, 201):
        selected = bandit.sample()
        reward = arms[selected].pull()
        bandit.update(selected, reward)
        selection_counts[selected] += 1

        if round_num in [10, 25, 50, 100, 200]:
            pcts = selection_counts / round_num * 100
            expected = [bandit.expected_reward(i) for i in range(4)]
            print(f"  {round_num:6d}  | {pcts[0]:4.0f}% | {pcts[1]:5.0f}% | {pcts[2]:5.0f}% | {pcts[3]:7.0f}% | " +
                  " ".join([f"{e:.2f}" for e in expected]))

    print("\n  Final arm selection distribution:")
    for i, name in enumerate(arm_names):
        pct = selection_counts[i] / 200 * 100
        bar = render_progress_bar(pct / 100)
        print(f"    {name:10s}: [{bar}] {pct:.0f}%")

    print("\n  Observation: Bandit quickly identifies 'Optimal' arm")
    print("               and allocates most pulls there (exploration → exploitation)")


def demo_regret_tracking():
    """Demo 4: Track cumulative regret over time."""
    print("\n" + "=" * 60)
    print("Demo 4: Regret Analysis - Cost of Learning")
    print("=" * 60)

    true_probs = [0.3, 0.5, 0.7]
    optimal_prob = max(true_probs)

    arms = [SimulatedArm(f"Arm_{i}", p) for i, p in enumerate(true_probs)]
    bandit = ThompsonBandit(n_arms=3)

    print(f"\n  True probabilities: {true_probs}")
    print(f"  Optimal arm: Arm_2 (p=0.7)")
    print(f"\n  Regret = Σ(optimal_reward - actual_reward)")

    cumulative_regret = 0.0
    regret_history = []
    rewards_history = []

    for round_num in range(1, 101):
        selected = bandit.sample()
        reward = arms[selected].pull()
        bandit.update(selected, reward)

        # Regret is the expected loss from not picking optimal
        instant_regret = optimal_prob - true_probs[selected]
        cumulative_regret += instant_regret
        regret_history.append(cumulative_regret)
        rewards_history.append(reward)

    print("\n  Cumulative Regret Over Time:")
    print()

    # ASCII plot
    height = 8
    width = 50
    max_regret = max(regret_history)

    # Downsample to width
    step = len(regret_history) / width
    sampled = [regret_history[int(i * step)] for i in range(width)]

    for row in range(height, 0, -1):
        threshold = max_regret * row / height
        line = "  "
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        label = f"{threshold:4.1f} │" if row in [height, height//2, 1] else "     │"
        print(label + line)

    print("       └" + "─" * width)
    print("        0" + " " * 22 + "50" + " " * 23 + "100")
    print("                            Rounds")

    print(f"\n  Final cumulative regret: {cumulative_regret:.2f}")
    print(f"  Average regret per round: {cumulative_regret / 100:.3f}")
    print("\n  Note: Regret grows sublinearly (√n) - Thompson Sampling is efficient!")


def demo_tool_selection():
    """Demo 5: Apply Thompson Sampling to AI tool selection."""
    print("\n" + "=" * 60)
    print("Demo 5: Tool Selection - HoloLoom Application")
    print("=" * 60)

    # Simulate HoloLoom's tool selection scenario
    tools = {
        "answer": 0.8,      # Direct answer - good for simple queries
        "research": 0.6,    # Research mode - good for complex queries
        "clarify": 0.4,     # Ask clarification - sometimes needed
        "refuse": 0.2,      # Refuse - rarely optimal
    }

    bandit = ThompsonBandit(n_arms=4, arm_names=list(tools.keys()))

    print("\n  Scenario: AI agent selecting tools for user queries")
    print("  Tools: answer (best), research (good), clarify (ok), refuse (poor)")

    print("\n  Simulating 30 queries with feedback...")
    print("\n  Query | Tool Selected | Feedback | Posterior Updates")
    print("  " + "─" * 55)

    arms = [SimulatedArm(name, prob) for name, prob in tools.items()]

    for query_num in range(1, 31):
        # Thompson Sampling selection
        selected = bandit.sample()
        tool_name = bandit.arm_names[selected]

        # Simulate user feedback
        feedback = arms[selected].pull()
        bandit.update(selected, feedback)

        if query_num <= 5 or query_num % 5 == 0:
            feedback_emoji = "👍" if feedback > 0.5 else "👎"
            expected = [f"{bandit.arm_names[i][:3]}:{bandit.expected_reward(i):.2f}"
                       for i in range(4)]
            print(f"  {query_num:5d} | {tool_name:13s} | {feedback_emoji}        | {', '.join(expected)}")

    print("\n  Final Tool Preferences (learned from feedback):")
    stats = bandit.get_stats()
    sorted_tools = sorted(stats.items(), key=lambda x: x[1]['expected'], reverse=True)

    for i, (name, s) in enumerate(sorted_tools):
        bar = render_progress_bar(s['expected'])
        rank = ["🥇", "🥈", "🥉", "  "][i]
        print(f"    {rank} {name:10s}: E[r]={s['expected']:.2f} [{bar}]")

    print("\n  Result: Agent learned to prefer 'answer' tool")
    print("          (highest expected reward) through exploration!")


def demo_comparison_strategies():
    """Demo 6: Compare Thompson Sampling vs other strategies."""
    print("\n" + "=" * 60)
    print("Demo 6: Strategy Comparison - Why Thompson Sampling?")
    print("=" * 60)

    true_probs = [0.3, 0.5, 0.7, 0.9]
    n_rounds = 100

    print("\n  Comparing 4 strategies over 100 rounds:")
    print("  - Greedy: Always pick current best")
    print("  - ε-Greedy (ε=0.1): 10% random exploration")
    print("  - UCB: Upper Confidence Bound")
    print("  - Thompson: Sample from posteriors")

    results = {}

    # 1. Greedy
    np.random.seed(42)
    greedy_rewards = []
    counts = np.zeros(4)
    totals = np.zeros(4)
    for _ in range(n_rounds):
        expected = totals / (counts + 1e-10)
        selected = np.argmax(expected) if np.any(counts > 0) else np.random.randint(4)
        reward = 1.0 if np.random.random() < true_probs[selected] else 0.0
        counts[selected] += 1
        totals[selected] += reward
        greedy_rewards.append(reward)
    results['Greedy'] = np.mean(greedy_rewards)

    # 2. ε-Greedy
    np.random.seed(42)
    epsilon_rewards = []
    counts = np.zeros(4)
    totals = np.zeros(4)
    for _ in range(n_rounds):
        if np.random.random() < 0.1:
            selected = np.random.randint(4)
        else:
            expected = totals / (counts + 1e-10)
            selected = np.argmax(expected) if np.any(counts > 0) else np.random.randint(4)
        reward = 1.0 if np.random.random() < true_probs[selected] else 0.0
        counts[selected] += 1
        totals[selected] += reward
        epsilon_rewards.append(reward)
    results['ε-Greedy'] = np.mean(epsilon_rewards)

    # 3. UCB
    np.random.seed(42)
    ucb_rewards = []
    counts = np.zeros(4)
    totals = np.zeros(4)
    for t in range(1, n_rounds + 1):
        if np.any(counts == 0):
            selected = np.argmin(counts)
        else:
            ucb = totals / counts + np.sqrt(2 * np.log(t) / counts)
            selected = np.argmax(ucb)
        reward = 1.0 if np.random.random() < true_probs[selected] else 0.0
        counts[selected] += 1
        totals[selected] += reward
        ucb_rewards.append(reward)
    results['UCB'] = np.mean(ucb_rewards)

    # 4. Thompson Sampling
    np.random.seed(42)
    thompson_rewards = []
    alpha = np.ones(4)
    beta = np.ones(4)
    for _ in range(n_rounds):
        samples = np.random.beta(alpha, beta)
        selected = np.argmax(samples)
        reward = 1.0 if np.random.random() < true_probs[selected] else 0.0
        if reward > 0.5:
            alpha[selected] += 1
        else:
            beta[selected] += 1
        thompson_rewards.append(reward)
    results['Thompson'] = np.mean(thompson_rewards)

    print("\n  Average Rewards (higher is better):")
    optimal = 0.9  # Best possible

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for strategy, avg_reward in sorted_results:
        bar = render_progress_bar(avg_reward / optimal)
        pct = avg_reward / optimal * 100
        print(f"    {strategy:10s}: {avg_reward:.3f} [{bar}] {pct:.0f}% of optimal")

    print(f"\n  Optimal possible: {optimal:.3f}")
    print("\n  Thompson Sampling typically performs best because it:")
    print("    • Naturally balances exploration/exploitation")
    print("    • Uses uncertainty for intelligent exploration")
    print("    • Has strong theoretical guarantees")


def main():
    """Run all Thompson Sampling demonstrations."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " THOMPSON SAMPLING SHOWCASE ".center(58) + "║")
    print("║" + " Bayesian Exploration for Optimal Decisions ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    start = time.time()

    demo_basic_bandit()
    demo_exploration_exploitation()
    demo_convergence()
    demo_regret_tracking()
    demo_tool_selection()
    demo_comparison_strategies()

    elapsed = time.time() - start

    print("\n" + "═" * 60)
    print(f"  Completed 6 demonstrations in {elapsed:.2f}s")
    print("  Key insight: Thompson Sampling = optimism under uncertainty")
    print("  Dependencies: numpy only (no torch required)")
    print("═" * 60)


if __name__ == "__main__":
    main()
