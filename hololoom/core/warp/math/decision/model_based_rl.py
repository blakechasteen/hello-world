"""
Model-Based Reinforcement Learning — RL4.5
===========================================

Learn a world model (transition dynamics + reward function), then use it
for planning and policy optimization. Bridges the sample efficiency of
planning with the flexibility of learning.

Classes:
    WorldModel: Learn transition and reward functions from experience
    Dyna: Tabular Dyna-Q mixing real and simulated experience
    MCTS_RL: Monte Carlo Tree Search with learned model
    MBPolicyOptimization: Model-based policy gradient (SVG-style)

Key Concepts:
    - Model learning: fit T(s'|s,a) and R(s,a) from transitions
    - Dyna architecture: interleave real steps with planning steps
    - MCTS: tree search using learned model for rollouts
    - SVG: backpropagate through learned dynamics for policy gradients

Applications:
    - Sample-efficient exploration in HoloLoom reasoning
    - Planning with learned weaving dynamics
    - Simulated experience for strategy evaluation

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single (s, a, r, s', done) transition."""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool = False


class WorldModel:
    """
    Learn transition dynamics T(s'|s,a) and reward function R(s,a)
    from observed transitions.

    Uses maximum likelihood estimation: count-based for discrete spaces.
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        # Transition counts: N[s, a, s']
        self.transition_counts = np.zeros((n_states, n_actions, n_states))
        # Reward accumulator
        self.reward_sum = np.zeros((n_states, n_actions))
        self.reward_count = np.zeros((n_states, n_actions))

    def update(self, transition: Transition) -> None:
        """Update model with a single transition."""
        s, a, r, sp = transition.state, transition.action, transition.reward, transition.next_state
        self.transition_counts[s, a, sp] += 1
        self.reward_sum[s, a] += r
        self.reward_count[s, a] += 1

    def learn(
        self, transitions: list[Transition], n_iter: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Learn dynamics from a batch of transitions.

        Args:
            transitions: List of observed transitions
            n_iter: Number of passes over the data (for iterative refinement)

        Returns:
            (transition_fn, reward_fn) where
            transition_fn[s, a, s'] = P(s'|s, a)
            reward_fn[s, a] = E[R|s, a]
        """
        for _ in range(n_iter):
            for t in transitions:
                self.update(t)

        return self.transition_function(), self.reward_function()

    def transition_function(self) -> np.ndarray:
        """
        Maximum likelihood transition probabilities.

        Returns:
            T[s, a, s'] = P(s'|s, a), shape (n_states, n_actions, n_states)
        """
        T = np.zeros_like(self.transition_counts)
        totals = self.transition_counts.sum(axis=2, keepdims=True)
        nonzero = totals > 0
        T[nonzero[..., 0]] = (
            self.transition_counts[nonzero[..., 0]] /
            totals[nonzero][..., np.newaxis] if totals[nonzero].size > 0
            else 0
        )
        # Proper normalization
        for s in range(self.n_states):
            for a in range(self.n_actions):
                total = self.transition_counts[s, a].sum()
                if total > 0:
                    T[s, a] = self.transition_counts[s, a] / total
                else:
                    T[s, a] = 1.0 / self.n_states  # Uniform prior
        return T

    def reward_function(self) -> np.ndarray:
        """
        Estimated reward function E[R|s, a].

        Returns:
            R[s, a], shape (n_states, n_actions)
        """
        R = np.zeros_like(self.reward_sum)
        nonzero = self.reward_count > 0
        R[nonzero] = self.reward_sum[nonzero] / self.reward_count[nonzero]
        return R

    def sample_transition(self, state: int, action: int) -> tuple[int, float]:
        """
        Sample a transition from the learned model.

        Args:
            state: Current state
            action: Action taken

        Returns:
            (next_state, reward) sampled from learned distributions
        """
        T = self.transition_function()
        probs = T[state, action]

        if probs.sum() < 1e-10:
            next_state = np.random.randint(self.n_states)
        else:
            next_state = np.random.choice(self.n_states, p=probs)

        R = self.reward_function()
        reward = R[state, action]

        return next_state, reward

    def model_error(self, test_transitions: list[Transition]) -> float:
        """
        Compute model prediction error on held-out transitions.

        Returns mean negative log-likelihood of observed transitions.
        """
        T = self.transition_function()
        total_nll = 0.0
        for t in test_transitions:
            prob = T[t.state, t.action, t.next_state]
            total_nll -= np.log(max(prob, 1e-10))
        return total_nll / max(len(test_transitions), 1)


class Dyna:
    """
    Dyna-Q: interleave real experience with model-based planning.

    After each real transition:
    1. Update Q from real experience
    2. Update the world model
    3. Perform n planning steps using simulated experience from the model

    This dramatically improves sample efficiency by "replaying" and
    "hallucinating" experience through the learned model.
    """

    def learn(
        self,
        env_fn: Callable[[int, int], tuple[int, float, bool]],
        n_episodes: int = 100,
        n_planning_steps: int = 5,
        n_states: int = 25,
        n_actions: int = 4,
        discount: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        initial_state_fn: Callable[[], int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Learn Q-values and policy via Dyna-Q.

        Args:
            env_fn: Environment step function (state, action) -> (next_state, reward, done)
            n_episodes: Number of episodes
            n_planning_steps: Simulated planning steps per real step
            n_states: Number of states
            n_actions: Number of actions
            discount: Discount factor gamma
            alpha: Learning rate
            epsilon: Exploration rate for epsilon-greedy
            initial_state_fn: Returns initial state for each episode

        Returns:
            (Q, policy) where Q[s,a] are action values and
            policy[s] is the greedy action
        """
        Q = np.zeros((n_states, n_actions))
        model = WorldModel(n_states, n_actions)
        visited_sa = []  # Track visited (s, a) pairs for planning

        for episode in range(n_episodes):
            state = initial_state_fn() if initial_state_fn else 0
            done = False
            steps = 0

            while not done and steps < 1000:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[state])

                next_state, reward, done = env_fn(state, action)

                # Direct RL update
                td_target = reward + discount * np.max(Q[next_state]) * (1 - done)
                Q[state, action] += alpha * (td_target - Q[state, action])

                # Model update
                model.update(Transition(state, action, reward, next_state, done))
                visited_sa.append((state, action))

                # Planning: simulate from model
                for _ in range(n_planning_steps):
                    if not visited_sa:
                        break
                    idx = np.random.randint(len(visited_sa))
                    s_sim, a_sim = visited_sa[idx]
                    sp_sim, r_sim = model.sample_transition(s_sim, a_sim)

                    td_target_sim = r_sim + discount * np.max(Q[sp_sim])
                    Q[s_sim, a_sim] += alpha * (td_target_sim - Q[s_sim, a_sim])

                state = next_state
                steps += 1

        policy = np.argmax(Q, axis=1)
        return Q, policy


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""

    def __init__(self, state: int, parent: Optional['MCTSNode'] = None, action: int | None = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self, n_actions: int) -> bool:
        return len(self.children) == n_actions

    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        best_score = -np.inf
        best = None
        log_parent = np.log(self.visit_count + 1)
        for child in self.children.values():
            if child.visit_count == 0:
                return child
            ucb = child.value + c * np.sqrt(log_parent / child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best = child
        return best


class MCTS_RL:
    """
    Monte Carlo Tree Search with a learned world model.

    Uses the model for:
    1. Expansion: predict next state from (state, action)
    2. Simulation: rollout using model predictions
    3. Value estimation: model-based return computation

    Combines the strengths of MCTS planning with learned dynamics.
    """

    def plan(
        self,
        state: int,
        model: WorldModel,
        n_simulations: int = 100,
        n_actions: int = 4,
        discount: float = 0.99,
        max_depth: int = 20,
        c_puct: float = 1.41,
    ) -> int:
        """
        Plan the best action from current state using MCTS.

        Args:
            state: Current state
            model: Learned world model
            n_simulations: Number of MCTS simulations
            n_actions: Number of available actions
            discount: Discount factor
            max_depth: Maximum rollout depth
            c_puct: UCB exploration constant

        Returns:
            Best action to take
        """
        root = MCTSNode(state)

        for _ in range(n_simulations):
            node = root

            # Selection: traverse tree using UCB
            depth = 0
            while node.is_expanded(n_actions) and depth < max_depth:
                node = node.best_child(c_puct)
                depth += 1

            # Expansion: add a new child
            if not node.is_expanded(n_actions) and depth < max_depth:
                for a in range(n_actions):
                    if a not in node.children:
                        next_state, _ = model.sample_transition(node.state, a)
                        child = MCTSNode(next_state, parent=node, action=a)
                        node.children[a] = child
                        node = child
                        break

            # Simulation: rollout using model
            value = self._rollout(node.state, model, n_actions, discount, max_depth - depth)

            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value *= discount
                node = node.parent

        # Select most-visited action
        best_action = max(root.children.keys(),
                          key=lambda a: root.children[a].visit_count)
        return best_action

    def _rollout(
        self,
        state: int,
        model: WorldModel,
        n_actions: int,
        discount: float,
        max_steps: int,
    ) -> float:
        """Random rollout from state using model."""
        total_return = 0.0
        gamma_t = 1.0

        for _ in range(max_steps):
            action = np.random.randint(n_actions)
            next_state, reward = model.sample_transition(state, action)
            total_return += gamma_t * reward
            gamma_t *= discount
            state = next_state

        return total_return


class MBPolicyOptimization:
    """
    Model-Based Policy Optimization (SVG-style).

    Learns a differentiable world model, then optimizes the policy
    by backpropagating through the model's predicted trajectories.

    Uses a simple linear model + linear policy for tractability.
    """

    def optimize(
        self,
        model: WorldModel,
        n_states: int,
        n_actions: int,
        n_iter: int = 100,
        horizon: int = 10,
        discount: float = 0.99,
        lr: float = 0.01,
    ) -> np.ndarray:
        """
        Optimize a stochastic policy using model-based gradients.

        Policy is softmax over action preferences: pi(a|s) = softmax(theta[s, :])

        Args:
            model: Learned world model with transition and reward functions
            n_states: Number of states
            n_actions: Number of actions
            n_iter: Number of optimization iterations
            horizon: Planning horizon for return estimation
            discount: Discount factor
            lr: Learning rate for policy parameters

        Returns:
            Policy matrix pi[s, a] = probability of action a in state s
        """
        theta = np.zeros((n_states, n_actions))
        T = model.transition_function()
        R = model.reward_function()

        for iteration in range(n_iter):
            # Compute value function under current policy
            policy = self._softmax(theta)
            V = self._policy_evaluation(T, R, policy, discount, n_states, n_actions)

            # Policy gradient: d/d_theta E[sum gamma^t r_t]
            grad = np.zeros_like(theta)
            for s in range(n_states):
                for a in range(n_actions):
                    # Advantage: Q(s,a) - V(s)
                    Q_sa = R[s, a] + discount * T[s, a] @ V
                    advantage = Q_sa - V[s]

                    # Softmax gradient
                    for a2 in range(n_actions):
                        if a2 == a:
                            grad[s, a2] += policy[s, a] * (1 - policy[s, a]) * advantage
                        else:
                            grad[s, a2] -= policy[s, a] * policy[s, a2] * advantage

            theta += lr * grad

        return self._softmax(theta)

    def _softmax(self, theta: np.ndarray) -> np.ndarray:
        """Row-wise softmax."""
        exp_theta = np.exp(theta - theta.max(axis=1, keepdims=True))
        return exp_theta / exp_theta.sum(axis=1, keepdims=True)

    def _policy_evaluation(
        self,
        T: np.ndarray,
        R: np.ndarray,
        policy: np.ndarray,
        discount: float,
        n_states: int,
        n_actions: int,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Iterative policy evaluation: V^pi(s) = sum_a pi(s,a)[R(s,a) + gamma sum_s' T(s,a,s') V(s')]."""
        V = np.zeros(n_states)

        for _ in range(max_iter):
            V_new = np.zeros(n_states)
            for s in range(n_states):
                for a in range(n_actions):
                    V_new[s] += policy[s, a] * (R[s, a] + discount * T[s, a] @ V)

            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new

        return V
