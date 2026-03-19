"""
MDP Foundations — Value Iteration, Policy Iteration, TD Learning
================================================================

Core Markov Decision Process algorithms: dynamic programming
solutions and temporal-difference learning.

Core Concepts:
- MDP: (S, A, P, R, γ) tuple defining a sequential decision problem
- Value iteration: iterative Bellman updates converging to V*
- Policy iteration: alternating policy evaluation and improvement
- TD(0): Online single-step temporal-difference prediction
- TD(λ): Multi-step TD with eligibility traces

Mathematical Foundation:
Bellman optimality: V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]
Policy evaluation: V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ P(s'|s,a) V^π(s')]
TD(0) update: V(s) ← V(s) + α [r + γ V(s') - V(s)]
TD(λ): e(s) ← γλe(s) + 1; V(s) ← V(s) + α δ e(s)

Applications to Warp Space:
- Optimal memory retrieval policies
- Weaving cycle state-action optimization
- Online value estimation for streaming decisions

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
class MDPResult:
    """Result of MDP solution."""
    V: np.ndarray           # Value function (n_states,)
    policy: np.ndarray      # Optimal policy (n_states,) — action indices
    iterations: int         # Number of iterations to converge
    converged: bool         # Whether convergence criterion was met


# ============================================================================
# MARKOV DECISION PROCESS
# ============================================================================

class MDP:
    """
    Finite Markov Decision Process.

    Defined by (S, A, P, R, γ):
    - S: state space {0, 1, ..., n_states - 1}
    - A: action space {0, 1, ..., n_actions - 1}
    - P: transition probabilities P[s, a, s'] = Pr(s' | s, a)
    - R: reward function R[s, a] = expected reward for (s, a)
    - γ: discount factor
    """

    def __init__(self, S: int, A: int, P: np.ndarray,
                 R: np.ndarray, gamma: float = 0.99):
        """
        Args:
            S: Number of states.
            A: Number of actions.
            P: Transition tensor (S x A x S). P[s, a, s'] = Pr(s'|s,a).
            R: Reward matrix (S x A). R[s, a] = expected reward.
            gamma: Discount factor ∈ [0, 1).
        """
        self.n_states = S
        self.n_actions = A
        self.P = P
        self.R = R
        self.gamma = gamma

        # Validate
        assert P.shape == (S, A, S), f"P shape {P.shape} != ({S}, {A}, {S})"
        assert R.shape == (S, A), f"R shape {R.shape} != ({S}, {A})"
        assert 0 <= gamma < 1, "gamma must be in [0, 1)"

    def value_iteration(self, tol: float = 1e-6,
                        max_iter: int = 10000) -> tuple[np.ndarray, np.ndarray]:
        """
        Value iteration: find optimal value function and policy.

        V_{k+1}(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V_k(s')]

        Args:
            tol: Convergence tolerance (max |V_{k+1} - V_k|).
            max_iter: Maximum iterations.

        Returns:
            (V, pi): Optimal value function and greedy policy.
        """
        V = np.zeros(self.n_states)

        for iteration in range(max_iter):
            V_new = np.zeros(self.n_states)

            for s in range(self.n_states):
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    q_values[a] = self.R[s, a] + self.gamma * np.dot(
                        self.P[s, a], V
                    )
                V_new[s] = np.max(q_values)

            delta = np.max(np.abs(V_new - V))
            V = V_new

            if delta < tol:
                logger.info(
                    f"Value iteration converged in {iteration + 1} iterations"
                )
                pi = self._greedy_policy(V)
                return V, pi

        logger.warning(
            f"Value iteration did not converge in {max_iter} iterations "
            f"(delta={delta:.2e})"
        )
        return V, self._greedy_policy(V)

    def policy_iteration(self, tol: float = 1e-6,
                         max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Policy iteration: alternating evaluation and improvement.

        1. Policy evaluation: solve V^π = R^π + γ P^π V^π
        2. Policy improvement: π(s) = argmax_a Q^π(s, a)

        Args:
            tol: Tolerance for policy evaluation.
            max_iter: Maximum policy improvement steps.

        Returns:
            (V, pi): Optimal value function and policy.
        """
        # Start with random policy
        pi = np.random.randint(0, self.n_actions, self.n_states)

        for iteration in range(max_iter):
            # Policy evaluation
            V = self._policy_evaluation(pi, tol)

            # Policy improvement
            new_pi = self._greedy_policy(V)

            if np.array_equal(new_pi, pi):
                logger.info(
                    f"Policy iteration converged in {iteration + 1} iterations"
                )
                return V, pi

            pi = new_pi

        logger.warning("Policy iteration: max iterations reached")
        return V, pi

    def _policy_evaluation(self, pi: np.ndarray,
                           tol: float = 1e-6) -> np.ndarray:
        """Solve V^π via iterative updates or linear system."""
        S = self.n_states

        # Build transition matrix and reward vector for policy pi
        P_pi = np.zeros((S, S))
        R_pi = np.zeros(S)
        for s in range(S):
            a = pi[s]
            P_pi[s] = self.P[s, a]
            R_pi[s] = self.R[s, a]

        # Solve (I - γ P^π) V = R^π
        try:
            V = np.linalg.solve(
                np.eye(S) - self.gamma * P_pi,
                R_pi
            )
        except np.linalg.LinAlgError:
            # Fallback to iterative
            V = np.zeros(S)
            for _ in range(10000):
                V_new = R_pi + self.gamma * P_pi @ V
                if np.max(np.abs(V_new - V)) < tol:
                    break
                V = V_new

        return V

    def _greedy_policy(self, V: np.ndarray) -> np.ndarray:
        """Compute greedy policy from value function."""
        pi = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                q_values[a] = self.R[s, a] + self.gamma * np.dot(
                    self.P[s, a], V
                )
            pi[s] = np.argmax(q_values)
        return pi

    def q_values(self, V: np.ndarray) -> np.ndarray:
        """Compute Q(s,a) from V(s). Returns (S x A) matrix."""
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Q[s, a] = self.R[s, a] + self.gamma * np.dot(
                    self.P[s, a], V
                )
        return Q


# ============================================================================
# TEMPORAL DIFFERENCE LEARNING
# ============================================================================

class TemporalDifference:
    """
    Temporal-difference learning for policy evaluation.

    Model-free: learns V^π from experience without knowing P or R.
    """

    @staticmethod
    def td0(env_fn: Callable[[], tuple[int, dict]],
            policy: Callable[[int], int],
            alpha: float = 0.1,
            gamma: float = 0.99,
            n_episodes: int = 500,
            n_states: int = 0) -> np.ndarray:
        """
        TD(0) prediction: learn V^π online.

        Update: V(s) ← V(s) + α [r + γ V(s') - V(s)]

        Args:
            env_fn: Factory returning (initial_state, env_info).
                    env_info must have 'step' key: Callable[[int], (s', r, done)].
            policy: Maps state -> action.
            alpha: Learning rate.
            gamma: Discount factor.
            n_episodes: Number of training episodes.
            n_states: State space size (for V initialization).

        Returns:
            Estimated value function V (n_states,).
        """
        V = np.zeros(n_states)
        visit_count = np.zeros(n_states)

        for ep in range(n_episodes):
            s, env_info = env_fn()
            step = env_info['step']
            done = False
            max_steps = env_info.get('max_steps', 1000)
            t = 0

            while not done and t < max_steps:
                a = policy(s)
                s_next, r, done = step(a)

                # TD(0) update
                td_error = r + gamma * (V[s_next] if not done else 0) - V[s]

                # Optionally decay alpha
                visit_count[s] += 1
                lr = alpha / (1 + 0.001 * visit_count[s])
                V[s] += lr * td_error

                s = s_next
                t += 1

        return V

    @staticmethod
    def td_lambda(env_fn: Callable[[], tuple[int, dict]],
                  policy: Callable[[int], int],
                  alpha: float = 0.1,
                  lam: float = 0.8,
                  gamma: float = 0.99,
                  n_episodes: int = 500,
                  n_states: int = 0) -> np.ndarray:
        """
        TD(λ) prediction with eligibility traces.

        On each step:
            δ = r + γ V(s') - V(s)            (TD error)
            e(s) ← γ λ e(s) + 1               (trace update)
            V ← V + α δ e                      (update all states)

        λ=0: equivalent to TD(0)
        λ=1: equivalent to Monte Carlo (with online updates)

        Args:
            env_fn: Environment factory.
            policy: Policy function.
            alpha: Learning rate.
            lam: Eligibility trace decay (λ ∈ [0, 1]).
            gamma: Discount factor.
            n_episodes: Number of episodes.
            n_states: State space size.

        Returns:
            Estimated value function V.
        """
        V = np.zeros(n_states)

        for ep in range(n_episodes):
            s, env_info = env_fn()
            step = env_info['step']
            done = False
            max_steps = env_info.get('max_steps', 1000)
            t = 0

            # Eligibility traces
            e = np.zeros(n_states)

            while not done and t < max_steps:
                a = policy(s)
                s_next, r, done = step(a)

                # TD error
                td_error = r + gamma * (V[s_next] if not done else 0) - V[s]

                # Update eligibility trace (accumulating)
                e *= gamma * lam
                e[s] += 1.0

                # Update all states proportionally to eligibility
                V += alpha * td_error * e

                s = s_next
                t += 1

        return V
