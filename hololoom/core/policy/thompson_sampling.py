"""
Thompson Sampling Bandit for Exploration/Exploitation.

Provides Bayesian bandits with multiple exploration strategies:
- Epsilon-greedy: Explore with probability epsilon
- Bayesian blend: Combine neural and bandit priors
- Pure Thompson: Use Thompson Sampling exclusively

Author: Claude Code
Date: November 7, 2025 (Elegance Track - Day 3)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

# Import BanditStrategy from shared types (no circular dependency)
from hololoom.protocols.types import BanditStrategy


class TSBandit:
    """
    Thompson Sampling bandit for exploration/exploitation.

    Maintains Beta distributions for each arm (tool choice).
    Uses Bayesian updating to balance exploration and exploitation.

    Why Thompson Sampling?
    - Optimal regret bounds
    - Natural exploration via sampling
    - Easy to interpret (success/failure counts)

    Now supports multiple exploration strategies!
    """

    def __init__(self, n_arms: int, strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY, epsilon: float = 0.1):
        """
        Initialize bandit.

        Args:
            n_arms: Number of arms (tool options)
            strategy: Which exploration strategy to use
            epsilon: Exploration rate for epsilon-greedy (default: 0.1 = 10%)
        """
        # Beta distribution parameters (successes, failures)
        self.success = np.ones(n_arms)
        self.fail = np.ones(n_arms)
        self.n_arms = n_arms
        self.strategy = strategy
        self.epsilon = epsilon

        # Track total pulls per arm
        self.pulls = np.zeros(n_arms)

    def choose(self) -> int:
        """
        Choose an arm using Thompson Sampling.

        Returns:
            Selected arm index
        """
        # Sample from Beta distributions
        samples = np.random.beta(self.success, self.fail)
        return int(np.argmax(samples))

    def get_priors(self) -> np.ndarray:
        """
        Get prior probabilities (expected values) for each arm.

        Returns:
            Array of prior probabilities [n_arms]
        """
        priors = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            # Mean of Beta distribution = α / (α + β)
            # Guard against division by zero (defensive, though Beta init prevents this)
            total = self.success[i] + self.fail[i]
            priors[i] = self.success[i] / max(total, 1e-10)
        return priors

    def select_with_strategy(
        self,
        neural_probs: np.ndarray,
        strategy: Optional[BanditStrategy] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select arm using specified strategy.

        Args:
            neural_probs: Probabilities from neural network [n_arms]
            strategy: Override default strategy (optional)

        Returns:
            Tuple of (selected_arm, debug_info)
        """
        strat = strategy or self.strategy
        debug_info = {'strategy': strat.value}

        if strat == BanditStrategy.EPSILON_GREEDY:
            # Epsilon-greedy: explore with probability epsilon
            if np.random.rand() < self.epsilon:
                # EXPLORE: Use Thompson Sampling
                arm = self.choose()
                debug_info['mode'] = 'explore'
                debug_info['exploration_rate'] = self.epsilon
            else:
                # EXPLOIT: Use neural network
                arm = int(np.argmax(neural_probs))
                debug_info['mode'] = 'exploit'

        elif strat == BanditStrategy.BAYESIAN_BLEND:
            # Bayesian blend: combine neural and bandit priors
            bandit_priors = self.get_priors()

            # Weighted combination (70% neural, 30% bandit)
            combined = 0.7 * neural_probs + 0.3 * bandit_priors
            arm = int(np.argmax(combined))

            debug_info['mode'] = 'blend'
            debug_info['neural_probs'] = neural_probs.tolist()
            debug_info['bandit_priors'] = bandit_priors.tolist()
            debug_info['combined'] = combined.tolist()

        elif strat == BanditStrategy.PURE_THOMPSON:
            # Pure Thompson: ignore neural network entirely
            arm = self.choose()
            debug_info['mode'] = 'pure_thompson'
            debug_info['sampled_values'] = np.random.beta(self.success, self.fail).tolist()

        else:
            raise ValueError(f"Unknown strategy: {strat}")

        self.pulls[arm] += 1
        debug_info['selected_arm'] = arm
        debug_info['total_pulls'] = self.pulls.tolist()

        return arm, debug_info

    def update(self, arm: int, reward: float):
        """
        Update arm statistics based on reward.

        Args:
            arm: Arm that was pulled
            reward: Observed reward (positive = success, negative = failure)
        """
        if reward > 0:
            self.success[arm] += reward
        else:
            self.fail[arm] += abs(reward)

    def get_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for all arms."""
        stats = {}
        for i in range(len(self.success)):
            # Guard against division by zero (defensive programming)
            total = self.success[i] + self.fail[i]
            mean = self.success[i] / max(total, 1e-10)
            stats[i] = {
                'mean': mean,
                'success': float(self.success[i]),
                'fail': float(self.fail[i]),
                'pulls': float(self.pulls[i])
            }
        return stats
