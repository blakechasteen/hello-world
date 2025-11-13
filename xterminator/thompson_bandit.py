"""
Thompson Sampling Bandit
========================

🐷 Phase 4: Adaptive Strategy Selection 🐷

Implements Thompson Sampling for multi-armed bandit strategy selection.
The system learns which fix strategies (AST, TEMPLATE, MANUAL) work best
and adapts over time based on outcomes.

Thompson Sampling Algorithm:
1. Maintain Beta(α, β) distribution for each strategy
2. Sample reward from each distribution
3. Select strategy with highest sampled reward
4. Update α, β based on outcome (success → α+1, failure → β+1)

Why Thompson Sampling?
- Bayesian exploration/exploitation balance
- Naturally handles uncertainty
- Converges to optimal strategy
- Simple to implement and explain

Integration with xTerminator:
- Replaces static FixStrategy selection
- Learns from FeedbackTracker outcomes
- Adapts to domain-specific success patterns
- Provides exploration in early stages, exploitation later
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random
import json
from pathlib import Path
import time

from .xterminator_types import FixStrategy


@dataclass
class BanditArm:
    """
    One arm of the multi-armed bandit (one strategy).

    Uses Beta(α, β) distribution to model success probability.
    """
    strategy: FixStrategy
    alpha: float = 1.0  # Successes + 1 (Laplace smoothing)
    beta: float = 1.0   # Failures + 1 (Laplace smoothing)
    total_pulls: int = 0
    total_successes: int = 0
    total_failures: int = 0

    # Performance tracking
    cumulative_reward: float = 0.0
    avg_confidence: float = 0.0

    def expected_reward(self) -> float:
        """
        Expected reward: E[Beta(α, β)] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """
        Sample from Beta(α, β) distribution.

        Returns sampled reward in [0, 1].
        """
        # Use Python's random.betavariate
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool, reward: float = 1.0):
        """
        Update α, β based on outcome.

        Args:
            success: True if fix succeeded
            reward: Reward value (typically confidence score)
        """
        self.total_pulls += 1

        if success:
            self.alpha += reward  # Weight by confidence
            self.total_successes += 1
            self.cumulative_reward += reward
        else:
            self.beta += (1.0 - reward)  # Weight by (1 - confidence)
            self.total_failures += 1

        # Update average confidence
        self.avg_confidence = self.cumulative_reward / max(1, self.total_pulls)

    def get_stats(self) -> Dict:
        """Get arm statistics."""
        return {
            'strategy': self.strategy.value,
            'alpha': self.alpha,
            'beta': self.beta,
            'expected_reward': self.expected_reward(),
            'total_pulls': self.total_pulls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': self.total_successes / max(1, self.total_pulls),
            'avg_confidence': self.avg_confidence,
            'cumulative_reward': self.cumulative_reward
        }


class SelectionMode(Enum):
    """How to select strategy."""
    THOMPSON = "thompson"       # Pure Thompson Sampling (default)
    EPSILON_GREEDY = "epsilon"  # ε-greedy with Thompson
    GREEDY = "greedy"           # Always pick best (no exploration)
    UNIFORM = "uniform"         # Random (for baseline)


@dataclass
class ThompsonBandit:
    """
    Thompson Sampling multi-armed bandit for strategy selection.

    Maintains Beta(α, β) distributions for each fix strategy and
    uses Thompson Sampling to balance exploration/exploitation.

    Usage:
        bandit = ThompsonBandit()

        # Select strategy
        strategy = bandit.select_strategy()

        # After fix attempt
        bandit.update(strategy, success=True, reward=0.92)

        # View performance
        print(bandit.get_performance_summary())
    """

    # Arms (one per strategy)
    arms: Dict[FixStrategy, BanditArm] = field(default_factory=dict)

    # Selection mode
    mode: SelectionMode = SelectionMode.THOMPSON
    epsilon: float = 0.1  # For epsilon-greedy mode

    # Tracking
    total_selections: int = 0
    selection_history: List[Tuple[FixStrategy, float, bool]] = field(default_factory=list)

    # Persistence
    state_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize arms for each strategy."""
        if not self.arms:
            # Initialize arms for all strategies
            for strategy in [FixStrategy.AST, FixStrategy.TEMPLATE, FixStrategy.MANUAL]:
                self.arms[strategy] = BanditArm(strategy=strategy)

    def select_strategy(
        self,
        excluded_strategies: Optional[List[FixStrategy]] = None
    ) -> FixStrategy:
        """
        Select strategy using Thompson Sampling.

        Args:
            excluded_strategies: Strategies to exclude (e.g., AST if no AST available)

        Returns:
            Selected FixStrategy
        """
        self.total_selections += 1

        # Filter available arms
        available_arms = [
            arm for arm in self.arms.values()
            if excluded_strategies is None or arm.strategy not in excluded_strategies
        ]

        if not available_arms:
            # Fallback: return AST
            return FixStrategy.AST

        # Selection based on mode
        if self.mode == SelectionMode.UNIFORM:
            # Random baseline
            selected_arm = random.choice(available_arms)

        elif self.mode == SelectionMode.GREEDY:
            # Always pick best expected reward
            selected_arm = max(available_arms, key=lambda a: a.expected_reward())

        elif self.mode == SelectionMode.EPSILON_GREEDY:
            # ε-greedy: explore with probability ε
            if random.random() < self.epsilon:
                selected_arm = random.choice(available_arms)
            else:
                selected_arm = max(available_arms, key=lambda a: a.expected_reward())

        else:  # THOMPSON (default)
            # Thompson Sampling: sample from each Beta distribution
            samples = [(arm, arm.sample()) for arm in available_arms]
            selected_arm, sampled_value = max(samples, key=lambda x: x[1])

        return selected_arm.strategy

    def update(
        self,
        strategy: FixStrategy,
        success: bool,
        reward: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """
        Update bandit based on fix outcome.

        Args:
            strategy: Strategy that was used
            success: True if fix succeeded
            reward: Reward value (typically confidence score)
            metadata: Additional metadata for tracking
        """
        if strategy not in self.arms:
            # Initialize arm if not exists
            self.arms[strategy] = BanditArm(strategy=strategy)

        # Update arm
        self.arms[strategy].update(success=success, reward=reward)

        # Track history
        self.selection_history.append((strategy, reward, success))

        # Persist if path set
        if self.state_path:
            self.save_state(self.state_path)

    def get_best_strategy(self) -> Tuple[FixStrategy, float]:
        """
        Get strategy with highest expected reward.

        Returns:
            (strategy, expected_reward)
        """
        best_arm = max(self.arms.values(), key=lambda a: a.expected_reward())
        return best_arm.strategy, best_arm.expected_reward()

    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary with:
            - per_strategy: Stats for each strategy
            - best_strategy: Strategy with highest expected reward
            - total_selections: Total number of selections
            - mode: Current selection mode
        """
        per_strategy = {
            strategy.value: arm.get_stats()
            for strategy, arm in self.arms.items()
        }

        best_strategy, best_reward = self.get_best_strategy()

        return {
            'per_strategy': per_strategy,
            'best_strategy': {
                'strategy': best_strategy.value,
                'expected_reward': best_reward
            },
            'total_selections': self.total_selections,
            'mode': self.mode.value,
            'epsilon': self.epsilon if self.mode == SelectionMode.EPSILON_GREEDY else None
        }

    def get_learning_curve(self, window_size: int = 50) -> List[Dict]:
        """
        Get learning curve showing performance improvement over time.

        Args:
            window_size: Rolling window for smoothing

        Returns:
            List of data points with:
            - selection_num: Selection number
            - strategy: Strategy selected
            - reward: Reward received
            - success: Success boolean
            - rolling_success_rate: Success rate in rolling window
        """
        curve = []

        for i, (strategy, reward, success) in enumerate(self.selection_history):
            # Calculate rolling success rate
            window_start = max(0, i - window_size)
            window = self.selection_history[window_start:i+1]
            rolling_success_rate = sum(1 for _, _, s in window if s) / len(window)

            curve.append({
                'selection_num': i + 1,
                'strategy': strategy.value,
                'reward': reward,
                'success': success,
                'rolling_success_rate': rolling_success_rate
            })

        return curve

    def detect_convergence(
        self,
        window_size: int = 100,
        threshold: float = 0.05
    ) -> Tuple[bool, Dict]:
        """
        Detect if bandit has converged to optimal strategy.

        Args:
            window_size: Window to check for convergence
            threshold: Max variance in rolling success rate to consider converged

        Returns:
            (converged, details)
        """
        if len(self.selection_history) < window_size:
            return False, {'reason': 'Not enough history', 'history_length': len(self.selection_history)}

        # Get recent window
        recent = self.selection_history[-window_size:]

        # Calculate success rate per strategy in window
        strategy_counts = {}
        strategy_successes = {}

        for strategy, reward, success in recent:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            if success:
                strategy_successes[strategy] = strategy_successes.get(strategy, 0) + 1

        # Find dominant strategy (most selected)
        if not strategy_counts:
            return False, {'reason': 'No strategies in window'}

        dominant_strategy = max(strategy_counts.keys(), key=lambda s: strategy_counts[s])
        dominant_fraction = strategy_counts[dominant_strategy] / window_size

        # Check if one strategy dominates (>80% of selections)
        if dominant_fraction < 0.8:
            return False, {
                'reason': 'No dominant strategy',
                'dominant_strategy': dominant_strategy.value,
                'dominant_fraction': dominant_fraction
            }

        # Check if success rate is stable
        rolling_rates = [
            sum(1 for _, _, s in self.selection_history[i:i+20] if s) / 20
            for i in range(len(self.selection_history) - window_size, len(self.selection_history) - 20, 10)
        ]

        if len(rolling_rates) < 2:
            return False, {'reason': 'Not enough rolling rates'}

        variance = sum((r - sum(rolling_rates) / len(rolling_rates)) ** 2 for r in rolling_rates) / len(rolling_rates)

        converged = variance < threshold

        return converged, {
            'dominant_strategy': dominant_strategy.value,
            'dominant_fraction': dominant_fraction,
            'rolling_variance': variance,
            'threshold': threshold,
            'recent_success_rate': sum(1 for _, _, s in recent if s) / len(recent)
        }

    def save_state(self, path: Path):
        """
        Save bandit state to JSON.

        Args:
            path: Path to save state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'version': '1.0',
            'timestamp': time.time(),
            'mode': self.mode.value,
            'epsilon': self.epsilon,
            'total_selections': self.total_selections,
            'arms': {
                strategy.value: {
                    'alpha': arm.alpha,
                    'beta': arm.beta,
                    'total_pulls': arm.total_pulls,
                    'total_successes': arm.total_successes,
                    'total_failures': arm.total_failures,
                    'cumulative_reward': arm.cumulative_reward,
                    'avg_confidence': arm.avg_confidence
                }
                for strategy, arm in self.arms.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: Path) -> 'ThompsonBandit':
        """
        Load bandit state from JSON.

        Args:
            path: Path to load state from

        Returns:
            Loaded ThompsonBandit
        """
        with open(path) as f:
            state = json.load(f)

        # Create bandit
        bandit = cls(
            mode=SelectionMode(state['mode']),
            epsilon=state.get('epsilon', 0.1),
            total_selections=state['total_selections'],
            state_path=path
        )

        # Restore arms
        for strategy_name, arm_data in state['arms'].items():
            strategy = FixStrategy(strategy_name)
            bandit.arms[strategy] = BanditArm(
                strategy=strategy,
                alpha=arm_data['alpha'],
                beta=arm_data['beta'],
                total_pulls=arm_data['total_pulls'],
                total_successes=arm_data['total_successes'],
                total_failures=arm_data['total_failures'],
                cumulative_reward=arm_data['cumulative_reward'],
                avg_confidence=arm_data['avg_confidence']
            )

        return bandit


def create_thompson_bandit(
    mode: SelectionMode = SelectionMode.THOMPSON,
    epsilon: float = 0.1,
    state_path: Optional[Path] = None
) -> ThompsonBandit:
    """
    Factory function to create Thompson Bandit.

    Args:
        mode: Selection mode (THOMPSON, EPSILON_GREEDY, GREEDY, UNIFORM)
        epsilon: Epsilon for epsilon-greedy mode
        state_path: Path to load/save state

    Returns:
        ThompsonBandit instance
    """
    # Load from state if exists
    if state_path and Path(state_path).exists():
        return ThompsonBandit.load_state(state_path)

    # Create new bandit
    return ThompsonBandit(
        mode=mode,
        epsilon=epsilon,
        state_path=state_path
    )
