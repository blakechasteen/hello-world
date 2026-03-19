"""
Offline Reinforcement Learning — RL4.6
=======================================

Learn policies from fixed datasets without environment interaction.
The core challenge is distribution shift: the learned policy may
visit states unseen in the data, leading to extrapolation error.

Classes:
    ConservativeQ: Conservative Q-Learning (CQL) — penalize OOD actions
    BatchConstrainedQ: BCQ — constrain actions to data support
    ImplicitQLearning: IQL — avoid querying OOD actions entirely

Key Concepts:
    - Distribution shift: policy visits states not in dataset
    - Extrapolation error: Q overestimates for unseen (s, a)
    - Conservative penalty: lower-bound Q for unseen actions
    - Support constraint: only allow actions seen in data
    - Expectile regression: avoid max over unseen actions

Applications:
    - Learning from logged HoloLoom reasoning traces
    - Policy improvement from historical weaving outcomes
    - Safe deployment without online exploration

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OfflineDataset:
    """
    Fixed dataset of transitions for offline RL.

    Attributes:
        states: State indices, shape (n,)
        actions: Action indices, shape (n,)
        rewards: Rewards, shape (n,)
        next_states: Next state indices, shape (n,)
        dones: Terminal flags, shape (n,)
    """
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    @property
    def size(self) -> int:
        return len(self.states)

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample a random mini-batch."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    @staticmethod
    def from_transitions(
        transitions: list[tuple[int, int, float, int, bool]],
    ) -> 'OfflineDataset':
        """Create dataset from list of (s, a, r, s', done) tuples."""
        s, a, r, sp, d = zip(*transitions)
        return OfflineDataset(
            states=np.array(s, dtype=int),
            actions=np.array(a, dtype=int),
            rewards=np.array(r, dtype=float),
            next_states=np.array(sp, dtype=int),
            dones=np.array(d, dtype=bool),
        )


class ConservativeQ:
    """
    Conservative Q-Learning (CQL).

    Adds a regularizer that penalizes Q-values for actions not in the
    dataset, producing a lower bound on the true Q-function:

        Q_CQL = Q_standard - alpha * (E_{a~uniform}[Q(s,a)] - E_{a~data}[Q(s,a)])

    This pushes down Q for out-of-distribution actions and pushes up
    Q for in-distribution actions, preventing overestimation.

    Reference: Kumar et al., "Conservative Q-Learning for Offline RL", NeurIPS 2020
    """

    def learn(
        self,
        dataset: OfflineDataset,
        n_states: int,
        n_actions: int,
        discount: float = 0.99,
        alpha: float = 1.0,
        n_iter: int = 1000,
        lr: float = 0.1,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Learn conservative Q-values from offline data.

        Args:
            dataset: Fixed offline dataset
            n_states: Number of states
            n_actions: Number of actions
            discount: Discount factor gamma
            alpha: CQL regularization strength (higher = more conservative)
            n_iter: Number of training iterations
            lr: Learning rate
            batch_size: Mini-batch size

        Returns:
            (Q, policy) where Q[s,a] are conservative action values
            and policy[s] is the greedy action
        """
        Q = np.zeros((n_states, n_actions))

        # Compute empirical action distribution from data
        action_dist = np.zeros((n_states, n_actions))
        for i in range(dataset.size):
            action_dist[dataset.states[i], dataset.actions[i]] += 1
        row_sums = action_dist.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        action_dist /= row_sums

        for iteration in range(n_iter):
            s, a, r, sp, done = dataset.sample_batch(
                min(batch_size, dataset.size)
            )

            for j in range(len(s)):
                # Standard Bellman target
                target = r[j] + discount * np.max(Q[sp[j]]) * (1 - done[j])

                # TD update
                Q[s[j], a[j]] += lr * (target - Q[s[j], a[j]])

            # CQL regularization: penalize OOD, reward in-distribution
            for state in range(n_states):
                if action_dist[state].sum() < 1e-10:
                    continue

                # log-sum-exp over all actions (soft maximum)
                q_logsumexp = np.log(np.sum(np.exp(Q[state] - Q[state].max())) + 1e-10) + Q[state].max()

                # Expected Q under data distribution
                q_data = np.sum(action_dist[state] * Q[state])

                # CQL penalty: pushes down OOD, pushes up in-distribution
                penalty = q_logsumexp - q_data
                Q[state] -= lr * alpha * penalty

        policy = np.argmax(Q, axis=1)
        return Q, policy


class BatchConstrainedQ:
    """
    Batch-Constrained Q-Learning (BCQ).

    Constrains the policy to only select actions within the support
    of the behavior policy (data). Actions not seen in the data are
    masked out, preventing extrapolation error entirely.

    The constraint is: pi(a|s) = 0 if P_data(a|s) < threshold.

    Reference: Fujimoto et al., "Off-Policy Deep RL without Exploration", ICML 2019
    """

    def learn(
        self,
        dataset: OfflineDataset,
        n_states: int,
        n_actions: int,
        discount: float = 0.99,
        threshold: float = 0.05,
        n_iter: int = 1000,
        lr: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Learn Q-values with batch constraint.

        Args:
            dataset: Fixed offline dataset
            n_states: Number of states
            n_actions: Number of actions
            discount: Discount factor
            threshold: Minimum data frequency to allow an action
            n_iter: Number of training iterations
            lr: Learning rate

        Returns:
            (Q, policy) where policy only selects data-supported actions
        """
        Q = np.zeros((n_states, n_actions))

        # Compute action support mask from data
        action_counts = np.zeros((n_states, n_actions))
        for i in range(dataset.size):
            action_counts[dataset.states[i], dataset.actions[i]] += 1

        total_per_state = action_counts.sum(axis=1, keepdims=True)
        total_per_state[total_per_state == 0] = 1
        action_freq = action_counts / total_per_state

        # Binary mask: actions above threshold frequency
        support_mask = action_freq >= threshold
        # Ensure at least one action per state
        for s in range(n_states):
            if not support_mask[s].any():
                if action_counts[s].sum() > 0:
                    support_mask[s, np.argmax(action_counts[s])] = True
                else:
                    support_mask[s] = True  # Allow all if no data

        for iteration in range(n_iter):
            for i in range(dataset.size):
                s = dataset.states[i]
                a = dataset.actions[i]
                r = dataset.rewards[i]
                sp = dataset.next_states[i]
                done = dataset.dones[i]

                # Constrained max: only over supported actions
                q_next = Q[sp].copy()
                q_next[~support_mask[sp]] = -np.inf
                max_q_next = np.max(q_next)
                if np.isinf(max_q_next):
                    max_q_next = 0.0

                target = r + discount * max_q_next * (1 - done)
                Q[s, a] += lr * (target - Q[s, a])

        # Constrained greedy policy
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            q_masked = Q[s].copy()
            q_masked[~support_mask[s]] = -np.inf
            policy[s] = np.argmax(q_masked)

        return Q, policy


class ImplicitQLearning:
    """
    Implicit Q-Learning (IQL).

    Avoids querying OOD actions entirely by using expectile regression
    instead of max over actions. The key insight: instead of
    V(s) = max_a Q(s, a), use V(s) = E_tau[Q(s, a)] where tau-expectile
    approximates the max as tau -> 1.

    The tau-expectile is the value v that minimizes:
        E[L_tau(Q - v)] where L_tau(u) = |tau - 1(u<0)| * u^2

    Reference: Kostrikov et al., "Offline RL with Implicit Q-Learning", ICLR 2022
    """

    def learn(
        self,
        dataset: OfflineDataset,
        n_states: int,
        n_actions: int,
        discount: float = 0.99,
        tau: float = 0.7,
        n_iter: int = 1000,
        lr: float = 0.1,
        beta: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Learn Q-values and policy via implicit Q-learning.

        Args:
            dataset: Fixed offline dataset
            n_states: Number of states
            n_actions: Number of actions
            discount: Discount factor
            tau: Expectile parameter in (0.5, 1). Higher = closer to max.
            n_iter: Number of training iterations
            lr: Learning rate
            beta: Temperature for advantage-weighted policy extraction

        Returns:
            (Q, policy) where Q avoids OOD extrapolation
        """
        Q = np.zeros((n_states, n_actions))
        V = np.zeros(n_states)

        for iteration in range(n_iter):
            # V-update: expectile regression on Q
            for s in range(n_states):
                # Collect Q(s, a) for actions seen in data at state s
                data_mask = dataset.states == s
                if not data_mask.any():
                    continue

                actions_at_s = dataset.actions[data_mask]
                q_values = Q[s, actions_at_s]

                # Expectile regression target
                # V = argmin_v E[L_tau(Q - v)]
                # Gradient: E[(tau - 1(Q < v)) * (Q - v)]
                for q_val in q_values:
                    diff = q_val - V[s]
                    weight = tau if diff > 0 else (1 - tau)
                    V[s] += lr * weight * diff

            # Q-update: standard Bellman with V as target
            for i in range(dataset.size):
                s = dataset.states[i]
                a = dataset.actions[i]
                r = dataset.rewards[i]
                sp = dataset.next_states[i]
                done = dataset.dones[i]

                target = r + discount * V[sp] * (1 - done)
                Q[s, a] += lr * (target - Q[s, a])

        # Policy extraction: advantage-weighted
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            advantages = Q[s] - V[s]
            # Softmax with temperature beta
            exp_adv = np.exp(beta * advantages - beta * advantages.max())
            probs = exp_adv / exp_adv.sum()
            policy[s] = np.argmax(probs)

        return Q, policy

    @staticmethod
    def expectile(values: np.ndarray, tau: float) -> float:
        """
        Compute the tau-expectile of a set of values.

        The tau-expectile minimizes:
            sum_i |tau - 1(v_i < m)| * (v_i - m)^2

        Args:
            values: Array of values
            tau: Expectile parameter in (0, 1)

        Returns:
            The tau-expectile
        """
        m = np.mean(values)

        for _ in range(100):
            weights = np.where(values >= m, tau, 1 - tau)
            m_new = np.sum(weights * values) / np.sum(weights)
            if abs(m_new - m) < 1e-8:
                break
            m = m_new

        return m
