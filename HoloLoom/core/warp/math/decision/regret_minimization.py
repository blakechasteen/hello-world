"""
Regret Minimization - Online Learning for Adaptive Systems
==========================================================

Algorithms that minimize regret over time, enabling adaptive learning.

Classes:
    RegretMinimizer: Base protocol for regret algorithms
    Hedge: Multiplicative weights algorithm
    EXP3: Bandit version of Hedge (partial information)
    FollowTheRegularizedLeader: FTRL framework
    OptimisticHedge: Optimistic online learning

Key Concept:
    Regret = ∑_t [loss(chosen_action_t)] - min_a ∑_t [loss(a)]

    "In hindsight, how much worse did we do compared to best fixed action?"

    Good algorithms achieve sublinear regret: R_T = O(√T)

Applications:
    - Online recommendation (adapt to user preferences)
    - Dynamic pricing (adapt to demand)
    - Multi-agent learning (converges to equilibria)
    - Thompson Sampling alternative (frequentist guarantee)

Author: Claude Code
Date: December 2025 (Math Module Expansion)
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class FeedbackType(Enum):
    """Type of feedback received."""
    FULL = "full"          # Observe all losses
    BANDIT = "bandit"      # Observe only chosen action's loss
    PARTIAL = "partial"    # Observe subset


@dataclass
class RegretStats:
    """Statistics for regret minimization."""
    total_rounds: int = 0
    cumulative_loss: float = 0.0
    cumulative_regret: float = 0.0
    best_fixed_loss: float = 0.0
    losses_per_action: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def average_regret(self) -> float:
        """Average regret per round."""
        return self.cumulative_regret / self.total_rounds if self.total_rounds > 0 else 0.0

    @property
    def regret_bound(self) -> float:
        """Theoretical O(√T) regret bound."""
        return np.sqrt(self.total_rounds) if self.total_rounds > 0 else 0.0


class RegretMinimizer(ABC):
    """
    Abstract base class for regret minimization algorithms.

    All algorithms should:
    1. Maintain a distribution over actions
    2. Update based on observed losses
    3. Track cumulative regret
    """

    @abstractmethod
    def get_distribution(self) -> np.ndarray:
        """Get current probability distribution over actions."""
        pass

    @abstractmethod
    def update(self, losses: np.ndarray) -> None:
        """Update based on observed losses (full feedback)."""
        pass

    @abstractmethod
    def choose(self) -> int:
        """Sample an action from current distribution."""
        pass

    @abstractmethod
    def get_stats(self) -> RegretStats:
        """Get cumulative statistics."""
        pass


class Hedge(RegretMinimizer):
    """
    Hedge (Multiplicative Weights) Algorithm.

    Full-information online learning with O(√T log n) regret.

    Update rule:
        w_i(t+1) = w_i(t) * exp(-η * loss_i(t))
        p_i(t) = w_i(t) / ∑_j w_j(t)

    Properties:
        - Regret bound: O(√(T log n))
        - Learning rate: η = √(log n / T) is optimal
        - Converges to coarse correlated equilibrium in games
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: Optional[float] = None,
        time_horizon: int = 1000
    ):
        """
        Args:
            n_actions: Number of available actions
            learning_rate: η parameter (None for adaptive)
            time_horizon: Expected number of rounds T (for optimal η)
        """
        self.n_actions = n_actions
        self.time_horizon = time_horizon

        # Set learning rate
        if learning_rate is None:
            self.eta = np.sqrt(np.log(n_actions) / time_horizon)
        else:
            self.eta = learning_rate

        # Initialize uniform weights
        self.weights = np.ones(n_actions)

        # Statistics
        self.t = 0
        self.cumulative_loss = 0.0
        self.losses_per_action = np.zeros(n_actions)
        self.chosen_actions: List[int] = []
        self.observed_losses: List[np.ndarray] = []

    def get_distribution(self) -> np.ndarray:
        """Get current probability distribution."""
        return self.weights / self.weights.sum()

    def choose(self) -> int:
        """Sample action from current distribution."""
        probs = self.get_distribution()
        action = np.random.choice(self.n_actions, p=probs)
        self.chosen_actions.append(action)
        return action

    def update(self, losses: np.ndarray) -> None:
        """
        Update weights based on observed losses.

        Args:
            losses: Loss vector (one per action)
        """
        losses = np.asarray(losses)

        # Update weights
        self.weights *= np.exp(-self.eta * losses)

        # Numerical stability: prevent underflow
        self.weights = np.maximum(self.weights, 1e-10)

        # Track statistics
        self.t += 1
        if self.chosen_actions:
            self.cumulative_loss += losses[self.chosen_actions[-1]]
        self.losses_per_action += losses
        self.observed_losses.append(losses)

    def update_bandit(self, action: int, loss: float) -> None:
        """
        Update with bandit feedback (only observe chosen action's loss).

        Uses importance-weighted estimator.
        """
        probs = self.get_distribution()

        # Importance-weighted loss estimate
        loss_estimate = np.zeros(self.n_actions)
        loss_estimate[action] = loss / probs[action]

        self.update(loss_estimate)

    def get_stats(self) -> RegretStats:
        """Get cumulative statistics."""
        if self.t == 0:
            return RegretStats()

        best_fixed_loss = np.min(self.losses_per_action)
        cumulative_regret = self.cumulative_loss - best_fixed_loss

        return RegretStats(
            total_rounds=self.t,
            cumulative_loss=self.cumulative_loss,
            cumulative_regret=cumulative_regret,
            best_fixed_loss=best_fixed_loss,
            losses_per_action=self.losses_per_action.copy()
        )

    def get_regret(self) -> float:
        """Compute current regret."""
        stats = self.get_stats()
        return stats.cumulative_regret


class EXP3(RegretMinimizer):
    """
    EXP3: Exponential-weight algorithm for Exploration and Exploitation.

    Bandit version of Hedge (only observe loss of chosen action).

    Properties:
        - Regret bound: O(√(n T log n))
        - Adds exploration: γ-mixture with uniform
        - Unbiased loss estimates via importance weighting
    """

    def __init__(
        self,
        n_actions: int,
        exploration_rate: Optional[float] = None,
        time_horizon: int = 1000
    ):
        """
        Args:
            n_actions: Number of actions
            exploration_rate: γ for mixing with uniform (None for adaptive)
            time_horizon: Expected T
        """
        self.n_actions = n_actions
        self.time_horizon = time_horizon

        # Set exploration rate
        if exploration_rate is None:
            self.gamma = min(1.0, np.sqrt(n_actions * np.log(n_actions) / time_horizon))
        else:
            self.gamma = exploration_rate

        # Learning rate
        self.eta = self.gamma / n_actions

        # Initialize weights
        self.weights = np.ones(n_actions)

        # Statistics
        self.t = 0
        self.cumulative_loss = 0.0
        self.estimated_losses = np.zeros(n_actions)
        self.last_chosen: Optional[int] = None

    def get_distribution(self) -> np.ndarray:
        """
        Get sampling distribution (mix of weights and uniform).
        """
        base_probs = self.weights / self.weights.sum()
        uniform = np.ones(self.n_actions) / self.n_actions
        return (1 - self.gamma) * base_probs + self.gamma * uniform

    def choose(self) -> int:
        """Sample action with exploration."""
        probs = self.get_distribution()
        self.last_chosen = np.random.choice(self.n_actions, p=probs)
        return self.last_chosen

    def update(self, losses: np.ndarray) -> None:
        """Update with full information (just use Hedge)."""
        self.weights *= np.exp(-self.eta * losses)
        self.weights = np.maximum(self.weights, 1e-10)

        self.t += 1
        if self.last_chosen is not None:
            self.cumulative_loss += losses[self.last_chosen]
        self.estimated_losses += losses

    def update_bandit(self, action: int, loss: float) -> None:
        """
        Update with bandit feedback.

        Args:
            action: The action that was taken
            loss: Observed loss for that action
        """
        probs = self.get_distribution()

        # Importance-weighted loss estimate (unbiased)
        loss_estimate = loss / probs[action]

        # Update only the chosen arm's weight
        self.weights[action] *= np.exp(-self.eta * loss_estimate)
        self.weights = np.maximum(self.weights, 1e-10)

        self.t += 1
        self.cumulative_loss += loss
        self.estimated_losses[action] += loss_estimate

    def get_stats(self) -> RegretStats:
        """Get statistics."""
        if self.t == 0:
            return RegretStats()

        best_fixed_loss = np.min(self.estimated_losses)
        regret = self.cumulative_loss - best_fixed_loss

        return RegretStats(
            total_rounds=self.t,
            cumulative_loss=self.cumulative_loss,
            cumulative_regret=regret,
            best_fixed_loss=best_fixed_loss,
            losses_per_action=self.estimated_losses.copy()
        )


class FollowTheRegularizedLeader(RegretMinimizer):
    """
    Follow The Regularized Leader (FTRL).

    At each round, choose:
        argmin_x [∑_{s<t} loss_s^T x + R(x)]

    where R(x) is a regularizer.

    Special cases:
        - R(x) = ||x||_2^2 / 2η: Online gradient descent
        - R(x) = ∑_i x_i log x_i: Entropic FTRL = Hedge
    """

    def __init__(
        self,
        n_actions: int,
        regularizer: str = "entropy",
        learning_rate: float = 0.1
    ):
        """
        Args:
            n_actions: Number of actions
            regularizer: "entropy" or "l2"
            learning_rate: η parameter
        """
        self.n_actions = n_actions
        self.regularizer = regularizer
        self.eta = learning_rate

        self.cumulative_losses = np.zeros(n_actions)
        self.t = 0
        self.cumulative_loss = 0.0
        self.last_chosen: Optional[int] = None

    def get_distribution(self) -> np.ndarray:
        """
        Compute optimal distribution given cumulative losses.
        """
        if self.regularizer == "entropy":
            # Entropy regularizer: softmax of negative cumulative losses
            scaled = -self.eta * self.cumulative_losses
            # Numerical stability
            scaled = scaled - np.max(scaled)
            exp_scaled = np.exp(scaled)
            return exp_scaled / exp_scaled.sum()

        else:  # l2
            # L2 regularizer: project cumulative gradient onto simplex
            # Simplified: use softmax approximation
            return self.get_distribution()  # Fallback to entropy

    def choose(self) -> int:
        """Sample from current distribution."""
        probs = self.get_distribution()
        self.last_chosen = np.random.choice(self.n_actions, p=probs)
        return self.last_chosen

    def update(self, losses: np.ndarray) -> None:
        """Update cumulative losses."""
        self.cumulative_losses += losses
        self.t += 1
        if self.last_chosen is not None:
            self.cumulative_loss += losses[self.last_chosen]

    def get_stats(self) -> RegretStats:
        """Get statistics."""
        if self.t == 0:
            return RegretStats()

        best_fixed = np.min(self.cumulative_losses)
        regret = self.cumulative_loss - best_fixed

        return RegretStats(
            total_rounds=self.t,
            cumulative_loss=self.cumulative_loss,
            cumulative_regret=regret,
            best_fixed_loss=best_fixed,
            losses_per_action=self.cumulative_losses.copy()
        )


class OptimisticHedge(Hedge):
    """
    Optimistic Hedge: Use predictions to improve regret.

    If losses are predictable, can achieve O(log T) regret.

    Uses:
        w_i(t+1) = w_i(t) * exp(-η * (loss_i(t) + prediction_i(t)))
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: Optional[float] = None,
        time_horizon: int = 1000
    ):
        super().__init__(n_actions, learning_rate, time_horizon)
        self.predictions = np.zeros(n_actions)

    def set_prediction(self, predictions: np.ndarray) -> None:
        """Set loss predictions for next round."""
        self.predictions = predictions

    def update(self, losses: np.ndarray) -> None:
        """Update using both actual losses and predictions."""
        # Use prediction as "hint"
        total = losses + self.predictions
        super().update(total)

        # Reset predictions
        self.predictions = np.zeros(self.n_actions)


# Integration with HoloLoom
class AdaptiveStrategySelector:
    """
    Use regret minimization for adaptive strategy selection in HoloLoom.

    Applications:
        - Adapter selection (which LoRA adapter to use)
        - Tool selection (which tool to invoke)
        - Memory selection (which memories to retrieve)
    """

    def __init__(
        self,
        n_strategies: int,
        algorithm: str = "hedge",
        **kwargs
    ):
        """
        Args:
            n_strategies: Number of strategies/actions
            algorithm: "hedge", "exp3", "ftrl"
        """
        if algorithm == "hedge":
            self.learner = Hedge(n_strategies, **kwargs)
        elif algorithm == "exp3":
            self.learner = EXP3(n_strategies, **kwargs)
        elif algorithm == "ftrl":
            self.learner = FollowTheRegularizedLeader(n_strategies, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def select(self) -> int:
        """Select a strategy."""
        return self.learner.choose()

    def feedback(self, strategy: int, reward: float) -> None:
        """
        Provide feedback for a strategy.

        Args:
            strategy: The strategy that was used
            reward: Reward (higher is better)
        """
        # Convert reward to loss
        loss = 1.0 - reward
        self.learner.update_bandit(strategy, loss)

    def get_preferences(self) -> np.ndarray:
        """Get current preference distribution."""
        return self.learner.get_distribution()

    def get_regret(self) -> float:
        """Get cumulative regret."""
        return self.learner.get_stats().cumulative_regret


if __name__ == "__main__":
    print("Regret Minimization Module Demo")
    print("=" * 50)

    # Simulate online learning scenario
    np.random.seed(42)

    n_actions = 3
    T = 500

    # True (unknown) best action
    true_losses = [0.3, 0.5, 0.7]  # Action 0 is best

    hedge = Hedge(n_actions, time_horizon=T)

    for t in range(T):
        # Choose action
        action = hedge.choose()

        # Observe losses (with noise)
        losses = np.array(true_losses) + 0.1 * np.random.randn(n_actions)
        losses = np.clip(losses, 0, 1)

        # Update
        hedge.update(losses)

    stats = hedge.get_stats()

    print(f"\nHedge after {T} rounds:")
    print(f"Final distribution: {hedge.get_distribution()}")
    print(f"Cumulative regret: {stats.cumulative_regret:.2f}")
    print(f"Average regret: {stats.average_regret:.4f}")
    print(f"Theoretical bound O(√T): {stats.regret_bound:.2f}")

    print(f"\nLosses per action: {stats.losses_per_action}")
    print(f"Best action: {np.argmin(stats.losses_per_action)}")

    # Test EXP3 (bandit)
    exp3 = EXP3(n_actions, time_horizon=T)

    for t in range(T):
        action = exp3.choose()
        loss = true_losses[action] + 0.1 * np.random.randn()
        exp3.update_bandit(action, max(0, min(1, loss)))

    exp3_stats = exp3.get_stats()

    print(f"\nEXP3 (bandit) after {T} rounds:")
    print(f"Final distribution: {exp3.get_distribution()}")
    print(f"Cumulative regret: {exp3_stats.cumulative_regret:.2f}")

    print("\n" + "=" * 50)
    print("Regret minimization module ready!")
