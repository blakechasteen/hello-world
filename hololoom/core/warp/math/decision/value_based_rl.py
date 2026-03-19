"""
Value-Based Reinforcement Learning — Q-Learning, SARSA, Double Q
=================================================================

Tabular RL algorithms that learn value functions from environment
interaction. Foundation for understanding deep RL methods.

Core Concepts:
- Q-Learning: Off-policy TD control (learns Q*)
- Double Q-Learning: Reduces maximization bias
- SARSA: On-policy TD control (learns Q^π)
- N-Step Returns: Multi-step bootstrap targets

Mathematical Foundation:
Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
Double Q: Q₁(s,a) ← Q₁(s,a) + α[r + γ Q₂(s', argmax_a' Q₁(s',a')) - Q₁(s,a)]
SARSA: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
N-step: G_t^{(n)} = Σ_{k=0}^{n-1} γᵏ rₜ₊ₖ + γⁿ V(sₜ₊ₙ)

Applications to Warp Space:
- Learning optimal weaving cycle policies
- Discrete action selection for memory retrieval strategies
- Foundation for Thompson Sampling policy learning

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RLResult:
    """Result of RL training.

    Attributes:
        Q: Q-table (n_states, n_actions)
        policy: Greedy policy derived from Q (n_states,)
        episode_rewards: Total reward per episode (n_episodes,)
        episode_lengths: Steps per episode (n_episodes,)
        converged: Whether Q-values stabilized
    """
    Q: np.ndarray
    policy: np.ndarray
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    converged: bool = False


# ============================================================================
# ENVIRONMENT PROTOCOL
# ============================================================================

# env_fn should return an object with:
#   reset() -> state (int)
#   step(action) -> (next_state, reward, done, info)
#   n_states: int
#   n_actions: int


def _epsilon_greedy(Q: np.ndarray, state: int, epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))


# ============================================================================
# Q-LEARNING
# ============================================================================

class QLearning:
    """Tabular Q-Learning (off-policy TD control).

    Learns the optimal Q-function Q* directly via the Bellman
    optimality equation, regardless of the behavior policy.

    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

    Example:
        >>> ql = QLearning()
        >>> result = ql.learn(env_fn, n_episodes=1000, lr=0.1,
        ...                   discount=0.99, epsilon=0.1)
        >>> result.policy  # Greedy actions per state
    """

    def learn(
        self,
        env_fn: Callable,
        n_episodes: int = 1000,
        lr: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
        max_steps: int = 1000,
        convergence_tol: float = 1e-4,
    ) -> RLResult:
        """Train Q-Learning agent.

        Args:
            env_fn: Callable that returns an environment instance
            n_episodes: Number of training episodes
            lr: Learning rate α
            discount: Discount factor γ
            epsilon: Initial exploration rate
            epsilon_decay: Multiplicative decay per episode
            min_epsilon: Minimum exploration rate
            max_steps: Maximum steps per episode
            convergence_tol: Q-value change threshold for convergence

        Returns:
            RLResult with learned Q-table and policy
        """
        env = env_fn()
        n_states = env.n_states
        n_actions = env.n_actions
        Q = np.zeros((n_states, n_actions))

        episode_rewards = np.zeros(n_episodes)
        episode_lengths = np.zeros(n_episodes, dtype=int)
        converged = False

        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            max_delta = 0.0
            eps = max(epsilon * (epsilon_decay ** ep), min_epsilon)

            for step in range(max_steps):
                action = _epsilon_greedy(Q, state, eps)

                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Q-Learning update (off-policy: uses max)
                td_target = reward + discount * np.max(Q[next_state]) * (1 - done)
                td_error = td_target - Q[state, action]
                Q[state, action] += lr * td_error

                max_delta = max(max_delta, abs(td_error))
                state = next_state

                if done:
                    break

            episode_rewards[ep] = total_reward
            episode_lengths[ep] = step + 1

            if max_delta < convergence_tol:
                converged = True

        policy = np.argmax(Q, axis=1)

        return RLResult(
            Q=Q,
            policy=policy,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            converged=converged,
        )


# ============================================================================
# DOUBLE Q-LEARNING
# ============================================================================

class DoubleQLearning:
    """Double Q-Learning — reduces maximization bias.

    Maintains two Q-tables. On each update, randomly pick which
    table to update. The other table provides the value estimate.

    This decouples action selection from value estimation,
    reducing the upward bias of standard Q-learning.

    Example:
        >>> dql = DoubleQLearning()
        >>> result = dql.learn(env_fn, n_episodes=1000)
    """

    def learn(
        self,
        env_fn: Callable,
        n_episodes: int = 1000,
        lr: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
        max_steps: int = 1000,
    ) -> RLResult:
        """Train Double Q-Learning agent.

        Args:
            env_fn: Callable returning environment
            n_episodes: Training episodes
            lr: Learning rate
            discount: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate
            min_epsilon: Minimum epsilon
            max_steps: Max steps per episode

        Returns:
            RLResult with averaged Q-table
        """
        env = env_fn()
        n_states = env.n_states
        n_actions = env.n_actions

        Q1 = np.zeros((n_states, n_actions))
        Q2 = np.zeros((n_states, n_actions))

        episode_rewards = np.zeros(n_episodes)
        episode_lengths = np.zeros(n_episodes, dtype=int)

        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            eps = max(epsilon * (epsilon_decay ** ep), min_epsilon)

            for step in range(max_steps):
                # Use combined Q for action selection
                Q_combined = Q1 + Q2
                action = _epsilon_greedy(Q_combined, state, eps)

                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                if np.random.random() < 0.5:
                    # Update Q1 using Q2 for value
                    best_a = int(np.argmax(Q1[next_state]))
                    td_target = reward + discount * Q2[next_state, best_a] * (1 - done)
                    Q1[state, action] += lr * (td_target - Q1[state, action])
                else:
                    # Update Q2 using Q1 for value
                    best_a = int(np.argmax(Q2[next_state]))
                    td_target = reward + discount * Q1[next_state, best_a] * (1 - done)
                    Q2[state, action] += lr * (td_target - Q2[state, action])

                state = next_state
                if done:
                    break

            episode_rewards[ep] = total_reward
            episode_lengths[ep] = step + 1

        Q = (Q1 + Q2) / 2
        policy = np.argmax(Q, axis=1)

        return RLResult(
            Q=Q,
            policy=policy,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )


# ============================================================================
# SARSA (On-Policy TD Control)
# ============================================================================

class SARSA:
    """SARSA — on-policy TD control.

    Unlike Q-learning, SARSA uses the actual next action (from the
    behavior policy) in the TD target. Learns Q^π, not Q*.

    Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]

    SARSA is safer than Q-learning in stochastic environments
    because it accounts for exploration in value estimates.

    Example:
        >>> sarsa = SARSA()
        >>> result = sarsa.learn(env_fn, n_episodes=1000)
    """

    def learn(
        self,
        env_fn: Callable,
        n_episodes: int = 1000,
        lr: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
        max_steps: int = 1000,
    ) -> RLResult:
        """Train SARSA agent.

        Args:
            env_fn: Callable returning environment
            n_episodes: Training episodes
            lr: Learning rate
            discount: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate
            min_epsilon: Minimum epsilon
            max_steps: Max steps per episode

        Returns:
            RLResult with learned Q-table
        """
        env = env_fn()
        n_states = env.n_states
        n_actions = env.n_actions
        Q = np.zeros((n_states, n_actions))

        episode_rewards = np.zeros(n_episodes)
        episode_lengths = np.zeros(n_episodes, dtype=int)

        for ep in range(n_episodes):
            state = env.reset()
            eps = max(epsilon * (epsilon_decay ** ep), min_epsilon)
            action = _epsilon_greedy(Q, state, eps)
            total_reward = 0.0

            for step in range(max_steps):
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # SARSA: choose next action under current policy
                next_action = _epsilon_greedy(Q, next_state, eps)

                # On-policy update
                td_target = reward + discount * Q[next_state, next_action] * (1 - done)
                Q[state, action] += lr * (td_target - Q[state, action])

                state = next_state
                action = next_action

                if done:
                    break

            episode_rewards[ep] = total_reward
            episode_lengths[ep] = step + 1

        policy = np.argmax(Q, axis=1)

        return RLResult(
            Q=Q,
            policy=policy,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )


# ============================================================================
# N-STEP RETURNS
# ============================================================================

class NStepReturn:
    """N-step return computation for TD learning.

    Bridges between one-step TD (n=1) and Monte Carlo (n=∞).
    Longer n reduces bias but increases variance.

    G_t^{(n)} = Σ_{k=0}^{n-1} γᵏ rₜ₊ₖ + γⁿ V(sₜ₊ₙ)

    Example:
        >>> nsr = NStepReturn()
        >>> rewards = np.array([1, 2, 3, 4, 5])
        >>> values = np.array([10, 11, 12, 13, 14, 15])
        >>> returns = nsr.compute(rewards, values, discount=0.99, n=3)
    """

    @staticmethod
    def compute(
        rewards: np.ndarray,
        values: np.ndarray,
        discount: float = 0.99,
        n: int = 3,
    ) -> np.ndarray:
        """Compute n-step returns.

        Args:
            rewards: Rewards r_0, r_1, ..., r_{T-1} (T,)
            values: State values V(s_0), ..., V(s_T) (T+1,)
            discount: Discount factor γ
            n: Number of steps

        Returns:
            N-step returns G_t^{(n)} for each t (T,)
        """
        rewards = np.asarray(rewards, dtype=float)
        values = np.asarray(values, dtype=float)
        T = len(rewards)

        returns = np.zeros(T)

        for t in range(T):
            G = 0.0
            for k in range(min(n, T - t)):
                G += (discount ** k) * rewards[t + k]

            # Bootstrap value at step t + n (or terminal)
            bootstrap_idx = min(t + n, T)
            G += (discount ** min(n, T - t)) * values[bootstrap_idx]

            returns[t] = G

        return returns

    @staticmethod
    def compute_lambda(
        rewards: np.ndarray,
        values: np.ndarray,
        discount: float = 0.99,
        lam: float = 0.95,
    ) -> np.ndarray:
        """Compute TD(λ) returns (exponentially weighted n-step returns).

        G_t^λ = (1-λ) Σ_{n=1}^{T-t} λ^{n-1} G_t^{(n)}

        Equivalent to GAE (Generalized Advantage Estimation) targets.

        Args:
            rewards: Rewards (T,)
            values: State values (T+1,)
            discount: Discount factor
            lam: Lambda parameter (0 = TD(0), 1 = Monte Carlo)

        Returns:
            TD(λ) returns (T,)
        """
        rewards = np.asarray(rewards, dtype=float)
        values = np.asarray(values, dtype=float)
        T = len(rewards)

        # Compute via backward recursion (efficient)
        returns = np.zeros(T)

        # TD errors
        deltas = rewards + discount * values[1:] - values[:T]

        # GAE-style backward pass
        gae = 0.0
        for t in range(T - 1, -1, -1):
            gae = deltas[t] + discount * lam * gae
            returns[t] = gae + values[t]

        return returns
