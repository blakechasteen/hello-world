"""
Thompson Sampling Bandit for Attack Strategy Selection
======================================================

Learns which attack strategies are most effective using
Bayesian multi-armed bandit with Beta priors.

Algorithm:
- Each attack strategy is an "arm"
- Track α (successes) and β (failures) for each arm
- Sample from Beta(α, β) to select next strategy
- Update priors based on attack results

Philosophy: "Explore strategies that might work, exploit strategies that do."

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-01
"""

import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from hololoom.bandits.beta_arm import BetaArm
from .strategies import AttackStrategy


@dataclass
class BanditArm:
    """
    A single arm in the multi-armed bandit.

    Delegates Beta math to BetaArm (canonical implementation).

    Uses Beta distribution for Thompson Sampling:
    - α (alpha): Prior successes + 1
    - β (beta): Prior failures + 1
    - E[X] = α / (α + β)

    Higher α means more successful attacks found.
    """

    strategy: AttackStrategy
    _arm: BetaArm = field(default_factory=BetaArm)
    total_pulls: int = 0
    total_rewards: float = 0.0
    last_updated: Optional[str] = None

    @property
    def alpha(self) -> float:
        return self._arm.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._arm.alpha = value

    @property
    def beta(self) -> float:
        return self._arm.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._arm.beta = value

    def sample(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)."""
        return self._arm.sample()

    def update(self, success: bool, reward: float = 1.0):
        """
        Update arm based on result.

        Args:
            success: True if attack found vulnerability
            reward: Severity of vulnerability (0.0-1.0)
        """
        self.total_pulls += 1

        if success:
            self._arm.update(success=True, weight=reward)
            self.total_rewards += reward
        else:
            self._arm.update(success=False, weight=1.0)

        self.last_updated = datetime.now().isoformat()

    @property
    def expected_reward(self) -> float:
        """Expected reward (mean of Beta distribution)."""
        return self._arm.expected_value()

    @property
    def success_rate(self) -> float:
        """Empirical success rate."""
        if self.total_pulls == 0:
            return 0.0
        return self.total_rewards / self.total_pulls

    @property
    def uncertainty(self) -> float:
        """Uncertainty (variance of Beta distribution)."""
        return self._arm.variance()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy': self.strategy.value,
            'alpha': self.alpha,
            'beta': self.beta,
            'total_pulls': self.total_pulls,
            'total_rewards': self.total_rewards,
            'last_updated': self.last_updated,
            'expected_reward': self.expected_reward,
            'success_rate': self.success_rate,
            'uncertainty': self.uncertainty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BanditArm':
        """Create from dictionary."""
        return cls(
            strategy=AttackStrategy(data['strategy']),
            _arm=BetaArm(alpha=data['alpha'], beta=data['beta']),
            total_pulls=data['total_pulls'],
            total_rewards=data['total_rewards'],
            last_updated=data.get('last_updated'),
        )


@dataclass
class SelectionResult:
    """Result of a strategy selection."""

    strategy: AttackStrategy
    sampled_value: float
    all_samples: Dict[AttackStrategy, float]
    arm: BanditArm


class RedTeamBandit:
    """
    Thompson Sampling bandit for attack strategy selection.

    Maintains Beta priors for each attack strategy and learns
    which strategies are most effective at finding vulnerabilities.

    Example:
        bandit = RedTeamBandit()

        # Select strategy
        strategy = bandit.select_strategy()

        # Execute attack
        result = await executor.execute_attack(strategy, payload, context)

        # Update bandit
        bandit.update(strategy, result.bypassed, result.severity)

        # Over time, bandit learns which strategies work best
        stats = bandit.get_stats()
        print(f"Best strategy: {stats['best_strategy']}")
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        decay_rate: float = 0.0,  # Optional decay for non-stationary environments
    ):
        """
        Initialize bandit with arms for each attack strategy.

        Args:
            initial_alpha: Initial alpha for all arms
            initial_beta: Initial beta for all arms
            decay_rate: Decay rate for priors (0 = no decay)
        """
        self.arms: Dict[AttackStrategy, BanditArm] = {}
        self.decay_rate = decay_rate
        self.selection_history: List[SelectionResult] = []

        # Initialize arm for each strategy
        for strategy in AttackStrategy:
            self.arms[strategy] = BanditArm(
                strategy=strategy,
                alpha=initial_alpha,
                beta=initial_beta,
            )

    def select_strategy(self) -> AttackStrategy:
        """
        Select attack strategy using Thompson Sampling.

        Samples from Beta distribution for each arm,
        returns arm with highest sample.

        Returns:
            Selected AttackStrategy
        """
        # Apply decay if configured
        if self.decay_rate > 0:
            self._apply_decay()

        # Sample from each arm
        samples = {
            strategy: arm.sample()
            for strategy, arm in self.arms.items()
        }

        # Select strategy with highest sample
        selected = max(samples, key=samples.get)

        # Record selection
        self.selection_history.append(SelectionResult(
            strategy=selected,
            sampled_value=samples[selected],
            all_samples=samples,
            arm=self.arms[selected],
        ))

        return selected

    def select_top_k(self, k: int = 3) -> List[AttackStrategy]:
        """
        Select top-k strategies by Thompson Sampling.

        Useful for running multiple attack strategies per cycle.

        Args:
            k: Number of strategies to select

        Returns:
            List of selected strategies
        """
        samples = {
            strategy: arm.sample()
            for strategy, arm in self.arms.items()
        }

        # Sort by sample value descending
        sorted_strategies = sorted(samples, key=samples.get, reverse=True)
        return sorted_strategies[:k]

    def update(
        self,
        strategy: AttackStrategy,
        success: bool,
        severity: float = 1.0
    ):
        """
        Update bandit based on attack result.

        Args:
            strategy: Strategy that was used
            success: True if attack found vulnerability
            severity: Severity of vulnerability (0.0-1.0)
        """
        if strategy in self.arms:
            self.arms[strategy].update(success, severity if success else 0.0)

    def update_batch(self, results: List[Tuple[AttackStrategy, bool, float]]):
        """
        Update bandit with multiple results.

        Args:
            results: List of (strategy, success, severity) tuples
        """
        for strategy, success, severity in results:
            self.update(strategy, success, severity)

    def get_expected_rewards(self) -> Dict[AttackStrategy, float]:
        """Get expected rewards for all strategies."""
        return {
            strategy: arm.expected_reward
            for strategy, arm in self.arms.items()
        }

    def get_best_strategy(self) -> AttackStrategy:
        """Get strategy with highest expected reward."""
        rewards = self.get_expected_rewards()
        return max(rewards, key=rewards.get)

    def get_exploration_candidates(
        self,
        uncertainty_threshold: float = 0.1
    ) -> List[AttackStrategy]:
        """
        Get strategies with high uncertainty (worth exploring).

        Args:
            uncertainty_threshold: Minimum uncertainty to include

        Returns:
            List of high-uncertainty strategies
        """
        return [
            strategy
            for strategy, arm in self.arms.items()
            if arm.uncertainty > uncertainty_threshold
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive bandit statistics."""
        total_pulls = sum(arm.total_pulls for arm in self.arms.values())
        total_rewards = sum(arm.total_rewards for arm in self.arms.values())

        best_strategy = self.get_best_strategy()
        most_explored = max(
            self.arms.values(),
            key=lambda a: a.total_pulls
        ).strategy if self.arms else None

        return {
            'total_pulls': total_pulls,
            'total_rewards': total_rewards,
            'overall_success_rate': total_rewards / max(1, total_pulls),
            'best_strategy': best_strategy.value if best_strategy else None,
            'best_expected_reward': self.arms[best_strategy].expected_reward if best_strategy else 0,
            'most_explored': most_explored.value if most_explored else None,
            'arms': {
                strategy.value: arm.to_dict()
                for strategy, arm in self.arms.items()
            }
        }

    def get_arm_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all arms sorted by expected reward."""
        summaries = [
            {
                'strategy': arm.strategy.value,
                'expected_reward': arm.expected_reward,
                'success_rate': arm.success_rate,
                'pulls': arm.total_pulls,
                'uncertainty': arm.uncertainty,
            }
            for arm in self.arms.values()
        ]
        return sorted(summaries, key=lambda x: x['expected_reward'], reverse=True)

    def _apply_decay(self):
        """Apply decay to priors (for non-stationary environments)."""
        for arm in self.arms.values():
            # Decay towards uniform prior
            arm.alpha = 1.0 + (arm.alpha - 1.0) * (1.0 - self.decay_rate)
            arm.beta = 1.0 + (arm.beta - 1.0) * (1.0 - self.decay_rate)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Path):
        """Save bandit state to file."""
        state = {
            'decay_rate': self.decay_rate,
            'arms': {
                strategy.value: arm.to_dict()
                for strategy, arm in self.arms.items()
            },
            'saved_at': datetime.now().isoformat(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path):
        """Load bandit state from file."""
        path = Path(path)
        if not path.exists():
            return

        with open(path, 'r') as f:
            state = json.load(f)

        self.decay_rate = state.get('decay_rate', 0.0)

        for strategy_value, arm_data in state.get('arms', {}).items():
            try:
                strategy = AttackStrategy(strategy_value)
                self.arms[strategy] = BanditArm.from_dict(arm_data)
            except ValueError:
                # Unknown strategy, skip
                pass

    def reset(self, initial_alpha: float = 1.0, initial_beta: float = 1.0):
        """Reset all arms to initial state."""
        for arm in self.arms.values():
            arm.alpha = initial_alpha
            arm.beta = initial_beta
            arm.total_pulls = 0
            arm.total_rewards = 0.0
            arm.last_updated = None

        self.selection_history.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_bandit(
    initial_alpha: float = 1.0,
    initial_beta: float = 1.0,
    decay_rate: float = 0.0,
    state_path: Optional[Path] = None
) -> RedTeamBandit:
    """
    Create a RedTeamBandit, optionally loading saved state.

    Args:
        initial_alpha: Initial alpha for new bandits
        initial_beta: Initial beta for new bandits
        decay_rate: Decay rate for non-stationary environments
        state_path: Path to load saved state from

    Returns:
        Configured RedTeamBandit
    """
    bandit = RedTeamBandit(
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        decay_rate=decay_rate,
    )

    if state_path and Path(state_path).exists():
        bandit.load(state_path)

    return bandit


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'BanditArm',
    'SelectionResult',
    'RedTeamBandit',
    'create_bandit',
]
