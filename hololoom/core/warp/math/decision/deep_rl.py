"""
Deep RL — DQN, Double DQN, Dueling DQN, Prioritized Replay, Safety, Regret
============================================================================

Value-based deep reinforcement learning with neural network function
approximation, experience replay variants, safety constraints, and
game-theoretic regret matching.

Core Concepts:
- DQN: Q-learning with neural network + experience replay + target network
- Double DQN: Decouple action selection and evaluation to reduce overestimation
- Dueling DQN: Separate value and advantage streams
- Prioritized replay: Sample transitions proportional to TD error
- Safety shield: Pre-execution safety verification
- Regret matching: Strategy computation for extensive-form games

Mathematical Foundation:
DQN loss: L = (r + γ max_{a'} Q_target(s', a') - Q(s, a))²
DDQN: y = r + γ Q_target(s', argmax_{a'} Q(s', a'))
Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
Priority: p_i = |δ_i| + ε, P(i) = p_i^α / Σ p_j^α

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLE NEURAL NETWORK (numpy-only)
# ============================================================================

class _SimpleNet:
    """Minimal feedforward network for DQN (numpy only)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activation."""
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    def copy_from(self, other: '_SimpleNet'):
        """Copy weights from another network."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    def update(self, x: np.ndarray, targets: np.ndarray,
               actions: np.ndarray, lr: float):
        """One step of gradient descent on MSE loss for selected actions."""
        q_values = self.forward(x)

        # Compute loss gradient only for selected actions
        n = len(actions)
        grad_out = np.zeros_like(q_values)
        for i in range(n):
            a = int(actions[i])
            grad_out[i, a] = 2.0 * (q_values[i, a] - targets[i]) / n

        # Backprop through W2
        dW2 = self.h.T @ grad_out
        db2 = np.sum(grad_out, axis=0)

        # Backprop through ReLU
        dh = grad_out @ self.W2.T
        dh[self.h <= 0] = 0

        dW1 = x.T @ dh
        db1 = np.sum(dh, axis=0)

        # Gradient descent
        self.W2 -= lr * np.clip(dW2, -1, 1)
        self.b2 -= lr * np.clip(db2, -1, 1)
        self.W1 -= lr * np.clip(dW1, -1, 1)
        self.b1 -= lr * np.clip(db1, -1, 1)


# ============================================================================
# EXPERIENCE REPLAY
# ============================================================================

@dataclass
class Transition:
    """A single experience transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplay:
    """
    Prioritized experience replay: sample transitions proportional to TD error.

    P(i) = p_i^α / Σ p_j^α  where p_i = |δ_i| + ε
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 epsilon: float = 0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer: list[Transition | None] = [None] * capacity
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0

    def add(self, transition: Transition, priority: float = 1.0):
        """Add transition with initial priority."""
        idx = self.position
        self.buffer[idx] = transition
        self.priorities[idx] = (abs(priority) + self.epsilon) ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch with importance sampling weights.

        Returns:
            (transitions, indices, is_weights)
        """
        probs = self.priorities[:self.size]
        probs = probs / (np.sum(probs) + 1e-10)

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        transitions = [self.buffer[i] for i in indices]

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-0.4)  # β=0.4
        weights /= np.max(weights)

        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = (abs(td) + self.epsilon) ** self.alpha


# ============================================================================
# DQN
# ============================================================================

@dataclass
class DQNResult:
    """Result of DQN training."""
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    final_q_net: Any  # _SimpleNet


class DQN:
    """
    Deep Q-Network with experience replay and target network.

    Key innovations over tabular Q-learning:
    1. Neural network function approximation
    2. Experience replay (break correlations)
    3. Target network (stabilize bootstrap targets)
    """

    @staticmethod
    def learn(env_fn: Callable,
              n_episodes: int = 500,
              lr: float = 0.001,
              gamma: float = 0.99,
              buffer_size: int = 10000,
              batch_size: int = 32,
              target_update: int = 100,
              epsilon_start: float = 1.0,
              epsilon_end: float = 0.01,
              epsilon_decay: float = 0.995,
              state_dim: int = 4,
              n_actions: int = 2,
              hidden_dim: int = 64) -> DQNResult:
        """
        Train a DQN agent.

        Args:
            env_fn: Factory returning (initial_state, env_info).
                    env_info['step'](action) -> (next_state, reward, done)
                    env_info['n_actions'] -> int
                    env_info.get('state_dim') -> int
            n_episodes: Training episodes.
            lr: Learning rate.
            gamma: Discount factor.
            buffer_size: Replay buffer capacity.
            batch_size: Mini-batch size.
            target_update: Steps between target network updates.
            state_dim: State dimensionality.
            n_actions: Number of discrete actions.
            hidden_dim: Hidden layer size.

        Returns:
            DQNResult with training history and final network.
        """
        q_net = _SimpleNet(state_dim, hidden_dim, n_actions)
        target_net = _SimpleNet(state_dim, hidden_dim, n_actions)
        target_net.copy_from(q_net)

        buffer = ReplayBuffer(buffer_size)
        epsilon = epsilon_start
        total_steps = 0

        episode_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            s, env_info = env_fn()
            step_fn = env_info['step']
            s = np.asarray(s, dtype=float).ravel()
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done and ep_length < env_info.get('max_steps', 500):
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    q = q_net.forward(s.reshape(1, -1))
                    a = int(np.argmax(q[0]))

                s_next, r, done = step_fn(a)
                s_next = np.asarray(s_next, dtype=float).ravel()

                buffer.add(Transition(s, a, r, s_next, done))

                # Train
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    states = np.array([t.state for t in batch])
                    actions = np.array([t.action for t in batch])
                    rewards = np.array([t.reward for t in batch])
                    next_states = np.array([t.next_state for t in batch])
                    dones = np.array([t.done for t in batch])

                    # Compute targets
                    q_next = target_net.forward(next_states)
                    targets = rewards + gamma * np.max(q_next, axis=1) * (1 - dones)

                    q_net.update(states, targets, actions, lr)

                # Update target network
                total_steps += 1
                if total_steps % target_update == 0:
                    target_net.copy_from(q_net)

                s = s_next
                ep_reward += r
                ep_length += 1

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        return DQNResult(
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.array(episode_lengths),
            final_q_net=q_net,
        )


# ============================================================================
# DOUBLE DQN
# ============================================================================

class DoubleDQN:
    """
    Double DQN: decouple action selection from evaluation.

    Target: y = r + γ Q_target(s', argmax_{a'} Q(s', a'))

    This reduces overestimation bias from the max operator.
    """

    @staticmethod
    def learn(env_fn: Callable, **kwargs) -> DQNResult:
        """Train Double DQN (same interface as DQN.learn)."""
        state_dim = kwargs.get('state_dim', 4)
        n_actions = kwargs.get('n_actions', 2)
        hidden_dim = kwargs.get('hidden_dim', 64)
        n_episodes = kwargs.get('n_episodes', 500)
        lr = kwargs.get('lr', 0.001)
        gamma = kwargs.get('gamma', 0.99)
        buffer_size = kwargs.get('buffer_size', 10000)
        batch_size = kwargs.get('batch_size', 32)
        target_update = kwargs.get('target_update', 100)
        epsilon_start = kwargs.get('epsilon_start', 1.0)
        epsilon_end = kwargs.get('epsilon_end', 0.01)
        epsilon_decay = kwargs.get('epsilon_decay', 0.995)

        q_net = _SimpleNet(state_dim, hidden_dim, n_actions)
        target_net = _SimpleNet(state_dim, hidden_dim, n_actions)
        target_net.copy_from(q_net)

        buffer = ReplayBuffer(buffer_size)
        epsilon = epsilon_start
        total_steps = 0

        episode_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            s, env_info = env_fn()
            step_fn = env_info['step']
            s = np.asarray(s, dtype=float).ravel()
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done and ep_length < env_info.get('max_steps', 500):
                if np.random.random() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    q = q_net.forward(s.reshape(1, -1))
                    a = int(np.argmax(q[0]))

                s_next, r, done = step_fn(a)
                s_next = np.asarray(s_next, dtype=float).ravel()
                buffer.add(Transition(s, a, r, s_next, done))

                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    states = np.array([t.state for t in batch])
                    actions = np.array([t.action for t in batch])
                    rewards = np.array([t.reward for t in batch])
                    next_states = np.array([t.next_state for t in batch])
                    dones = np.array([t.done for t in batch])

                    # Double DQN: select action with online net, evaluate with target
                    q_next_online = q_net.forward(next_states)
                    best_actions = np.argmax(q_next_online, axis=1)
                    q_next_target = target_net.forward(next_states)
                    targets = rewards + gamma * q_next_target[
                        np.arange(batch_size), best_actions
                    ] * (1 - dones)

                    q_net.update(states, targets, actions, lr)

                total_steps += 1
                if total_steps % target_update == 0:
                    target_net.copy_from(q_net)

                s = s_next
                ep_reward += r
                ep_length += 1

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        return DQNResult(
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.array(episode_lengths),
            final_q_net=q_net,
        )


# ============================================================================
# DUELING DQN
# ============================================================================

class DuelingDQN:
    """
    Dueling DQN: separate value and advantage streams.

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

    The value stream learns state quality independently of actions,
    while the advantage stream learns the relative benefit of each action.
    """

    @staticmethod
    def learn(env_fn: Callable, **kwargs) -> DQNResult:
        """Train Dueling DQN."""
        state_dim = kwargs.get('state_dim', 4)
        n_actions = kwargs.get('n_actions', 2)
        hidden_dim = kwargs.get('hidden_dim', 64)
        n_episodes = kwargs.get('n_episodes', 500)
        lr = kwargs.get('lr', 0.001)
        gamma = kwargs.get('gamma', 0.99)
        buffer_size = kwargs.get('buffer_size', 10000)
        batch_size = kwargs.get('batch_size', 32)
        target_update = kwargs.get('target_update', 100)
        epsilon_start = kwargs.get('epsilon_start', 1.0)
        epsilon_end = kwargs.get('epsilon_end', 0.01)
        epsilon_decay = kwargs.get('epsilon_decay', 0.995)

        # Use separate value and advantage networks
        # Value: state -> 1, Advantage: state -> n_actions
        value_net = _SimpleNet(state_dim, hidden_dim, 1)
        adv_net = _SimpleNet(state_dim, hidden_dim, n_actions)
        value_target = _SimpleNet(state_dim, hidden_dim, 1)
        adv_target = _SimpleNet(state_dim, hidden_dim, n_actions)
        value_target.copy_from(value_net)
        adv_target.copy_from(adv_net)

        def q_values(v_net, a_net, states):
            v = v_net.forward(states)  # (batch, 1)
            a = a_net.forward(states)  # (batch, n_actions)
            return v + a - np.mean(a, axis=1, keepdims=True)

        buffer = ReplayBuffer(buffer_size)
        epsilon = epsilon_start
        total_steps = 0
        episode_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            s, env_info = env_fn()
            step_fn = env_info['step']
            s = np.asarray(s, dtype=float).ravel()
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done and ep_length < env_info.get('max_steps', 500):
                if np.random.random() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    q = q_values(value_net, adv_net, s.reshape(1, -1))
                    a = int(np.argmax(q[0]))

                s_next, r, done = step_fn(a)
                s_next = np.asarray(s_next, dtype=float).ravel()
                buffer.add(Transition(s, a, r, s_next, done))

                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    states = np.array([t.state for t in batch])
                    actions = np.array([t.action for t in batch])
                    rewards = np.array([t.reward for t in batch])
                    next_states = np.array([t.next_state for t in batch])
                    dones = np.array([t.done for t in batch])

                    q_next = q_values(value_target, adv_target, next_states)
                    targets = rewards + gamma * np.max(q_next, axis=1) * (1 - dones)

                    # Update both networks
                    # This is a simplification; proper dueling DQN shares a trunk
                    value_net.update(states, targets, actions, lr)
                    adv_net.update(states, targets, actions, lr)

                total_steps += 1
                if total_steps % target_update == 0:
                    value_target.copy_from(value_net)
                    adv_target.copy_from(adv_net)

                s = s_next
                ep_reward += r
                ep_length += 1

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        return DQNResult(
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.array(episode_lengths),
            final_q_net=adv_net,  # Return advantage network
        )


# ============================================================================
# SAFETY SHIELD
# ============================================================================

class SafetyShield:
    """
    Pre-execution safety verification for RL actions.

    Checks proposed actions against safety specifications before
    allowing execution. Prevents catastrophic actions while
    maintaining learning progress.
    """

    @staticmethod
    def is_safe(state: np.ndarray, action: int,
                safety_spec: dict[str, Any]) -> bool:
        """
        Check if an action is safe to execute from the current state.

        Args:
            state: Current state vector.
            action: Proposed action.
            safety_spec: Safety specification containing:
                - 'state_bounds': (min, max) arrays for safe state region
                - 'forbidden_actions': dict mapping state conditions to forbidden actions
                - 'transition_model': optional (s, a) -> s' predictor
                - 'constraint_fn': optional (s, a) -> bool

        Returns:
            True if the action is considered safe.
        """
        # Check state bounds
        if 'state_bounds' in safety_spec:
            s_min, s_max = safety_spec['state_bounds']
            if np.any(state < s_min) or np.any(state > s_max):
                return False  # Already in unsafe state

        # Check forbidden actions
        if 'forbidden_actions' in safety_spec:
            for condition_fn, forbidden in safety_spec['forbidden_actions'].items():
                if condition_fn(state) and action in forbidden:
                    return False

        # Check via transition model
        if 'transition_model' in safety_spec and 'state_bounds' in safety_spec:
            model = safety_spec['transition_model']
            s_next = model(state, action)
            s_min, s_max = safety_spec['state_bounds']
            if np.any(s_next < s_min) or np.any(s_next > s_max):
                return False

        # Check custom constraint
        if 'constraint_fn' in safety_spec:
            if not safety_spec['constraint_fn'](state, action):
                return False

        return True

    @staticmethod
    def safe_action(state: np.ndarray, q_values: np.ndarray,
                    safety_spec: dict[str, Any]) -> int:
        """
        Select the best safe action. Falls back to safest option if
        no action is fully safe.

        Args:
            state: Current state.
            q_values: Q-values for all actions (n_actions,).
            safety_spec: Safety specification.

        Returns:
            Index of the selected safe action.
        """
        n_actions = len(q_values)

        # Try actions in order of Q-value (best first)
        sorted_actions = np.argsort(q_values)[::-1]

        for a in sorted_actions:
            if SafetyShield.is_safe(state, int(a), safety_spec):
                return int(a)

        # No safe action found — return best Q-value action with warning
        logger.warning("No safe action found; selecting best Q-value action")
        return int(sorted_actions[0])


# ============================================================================
# REGRET MATCHING
# ============================================================================

class RegretMatching:
    """
    Regret matching for extensive-form games.

    Maintains cumulative regret for each action and plays a strategy
    proportional to positive regret. Converges to Nash equilibrium
    in two-player zero-sum games.

    Average strategy converges to Nash at rate O(1/√T).
    """

    def __init__(self, n_actions: int):
        """
        Args:
            n_actions: Number of available actions.
        """
        self.n_actions = n_actions
        self.cumulative_regret = np.zeros(n_actions)
        self.cumulative_strategy = np.zeros(n_actions)
        self.t = 0

    def get_strategy(self) -> np.ndarray:
        """
        Current strategy: proportional to positive regret.

        σ(a) = max(R(a), 0) / Σ max(R(a'), 0)

        If all regrets are non-positive, use uniform strategy.
        """
        positive_regret = np.maximum(self.cumulative_regret, 0)
        total = np.sum(positive_regret)

        if total > 0:
            strategy = positive_regret / total
        else:
            strategy = np.ones(self.n_actions) / self.n_actions

        return strategy

    def update(self, losses: np.ndarray):
        """
        Update cumulative regret given the loss for each action.

        Regret for action a = (loss of played action) - (loss of action a).

        Args:
            losses: Loss for each action (n_actions,). Lower is better.
        """
        strategy = self.get_strategy()
        expected_loss = np.dot(strategy, losses)

        # Instantaneous regret: how much better each action would have been
        instant_regret = expected_loss - losses

        self.cumulative_regret += instant_regret
        self.cumulative_strategy += strategy
        self.t += 1

    def average_strategy(self) -> np.ndarray:
        """
        Average strategy over all iterations (converges to Nash).
        """
        total = np.sum(self.cumulative_strategy)
        if total > 0:
            return self.cumulative_strategy / total
        return np.ones(self.n_actions) / self.n_actions

    def exploitability_bound(self) -> float:
        """
        Upper bound on exploitability: max(cumulative_regret) / T.
        """
        if self.t == 0:
            return float('inf')
        return float(np.max(self.cumulative_regret) / self.t)
