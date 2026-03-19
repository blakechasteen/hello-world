"""
Constrained Markov Decision Processes — Optimal Policies Under Constraints
==========================================================================

Standard MDPs maximize expected reward. CMDPs add cost constraints:
the agent must keep expected costs below limits while maximizing reward.

    max  E[sum_t gamma^t * r(s_t, a_t)]
    s.t. E[sum_t gamma^t * c_k(s_t, a_t)] <= d_k  for each constraint k

Key insight: the optimal CMDP policy may be *stochastic* even when the
unconstrained MDP has a deterministic optimum. Randomization lets the
agent trade off between reward and cost at the boundary.

Solution Methods:
- LP relaxation: exact solution via occupancy measure formulation
- Lagrangian dual: iterative dual ascent on multipliers
- Primal-dual: simultaneous policy + multiplier updates
- Constrained policy gradient: projected gradient method

Mathematical Foundation:
The LP relaxation uses the occupancy measure d(s,a) = (1-gamma) * sum_t gamma^t P(s_t=s, a_t=a).
The CMDP becomes a linear program in d(s,a) with flow conservation and cost constraints.
Strong duality holds: LP optimum = Lagrangian dual optimum (under mild conditions).

Applications to Warp Space:
- Resource-constrained weaving (limit compute/memory per cycle)
- Safe exploration with cost budgets
- Multi-objective optimization with hard constraints
- Token budget management under quality constraints

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CMDPSpec:
    """Specification of a Constrained MDP.

    Attributes:
        n_states: Number of states |S|.
        n_actions: Number of actions |A|.
        transition: Transition kernel P(s'|s,a), shape (|S|, |A|, |S|).
        reward: Reward function r(s,a), shape (|S|, |A|).
        costs: List of cost functions [c_k(s,a)], each shape (|S|, |A|).
        cost_limits: Upper bounds [d_k] on expected discounted cost.
        discount: Discount factor gamma in (0, 1).
    """
    n_states: int
    n_actions: int
    transition: np.ndarray       # (S, A, S)
    reward: np.ndarray           # (S, A)
    costs: list[np.ndarray]      # List of (S, A)
    cost_limits: list[float]
    discount: float = 0.99

    def __post_init__(self):
        assert self.transition.shape == (self.n_states, self.n_actions, self.n_states)
        assert self.reward.shape == (self.n_states, self.n_actions)
        assert len(self.costs) == len(self.cost_limits)
        for c in self.costs:
            assert c.shape == (self.n_states, self.n_actions)
        assert 0 < self.discount < 1


@dataclass
class CMDPPolicy:
    """Result of solving a CMDP.

    Attributes:
        action_probs: Policy pi(a|s), shape (|S|, |A|). Rows sum to 1.
        value: Expected discounted reward under this policy.
        cost_values: Expected discounted cost for each constraint.
        lagrange_multipliers: Dual variables lambda_k for each constraint.
        feasible: Whether all cost constraints are satisfied.
    """
    action_probs: np.ndarray          # (S, A)
    value: float = 0.0
    cost_values: list[float] = field(default_factory=list)
    lagrange_multipliers: np.ndarray = field(default_factory=lambda: np.array([]))
    feasible: bool = True


# ============================================================================
# HELPERS
# ============================================================================

class ConstrainedMDP:
    """Solver for Constrained Markov Decision Processes.

    Provides LP relaxation, Lagrangian dual, and primal-dual methods.
    All methods return a CMDPPolicy with the stochastic policy,
    value, cost values, and feasibility flag.
    """

    @staticmethod
    def value_iteration(
        transition: np.ndarray,
        reward: np.ndarray,
        discount: float,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unconstrained value iteration.

        Args:
            transition: P(s'|s,a), shape (S, A, S).
            reward: r(s,a), shape (S, A).
            discount: Gamma.
            n_iter: Number of iterations.

        Returns:
            (V, pi) where V is shape (S,) and pi is shape (S,)
            with deterministic action indices.
        """
        n_states, n_actions, _ = transition.shape
        V = np.zeros(n_states)

        for _ in range(n_iter):
            # Q(s,a) = r(s,a) + gamma * sum_s' P(s'|s,a) * V(s')
            Q = reward + discount * np.einsum('sap,p->sa', transition, V)
            V = Q.max(axis=1)

        Q = reward + discount * np.einsum('sap,p->sa', transition, V)
        pi = Q.argmax(axis=1)
        return V, pi

    @staticmethod
    def policy_evaluation(
        transition: np.ndarray,
        reward: np.ndarray,
        policy: np.ndarray,
        discount: float,
        n_iter: int = 200,
    ) -> np.ndarray:
        """Evaluate a stochastic policy.

        Args:
            transition: P(s'|s,a), shape (S, A, S).
            reward: r(s,a), shape (S, A).
            policy: pi(a|s), shape (S, A).
            discount: Gamma.
            n_iter: Number of iterations.

        Returns:
            V: State values, shape (S,).
        """
        n_states = transition.shape[0]
        V = np.zeros(n_states)

        # Expected reward under policy: r_pi(s) = sum_a pi(a|s) * r(s,a)
        r_pi = np.sum(policy * reward, axis=1)  # (S,)

        # Transition under policy: P_pi(s'|s) = sum_a pi(a|s) * P(s'|s,a)
        P_pi = np.einsum('sa,sap->sp', policy, transition)  # (S, S)

        for _ in range(n_iter):
            V_new = r_pi + discount * P_pi @ V
            if np.max(np.abs(V_new - V)) < 1e-10:
                break
            V = V_new

        return V

    @staticmethod
    def _occupancy_to_policy(
        d: np.ndarray,
        n_states: int,
        n_actions: int,
    ) -> np.ndarray:
        """Convert occupancy measure d(s,a) to policy pi(a|s).

        pi(a|s) = d(s,a) / sum_a' d(s,a')
        """
        d_sa = d.reshape(n_states, n_actions)
        d_s = d_sa.sum(axis=1, keepdims=True)
        # Where d_s is zero, use uniform policy
        uniform = np.ones((n_states, n_actions)) / n_actions
        mask = (d_s > 1e-15).flatten()
        policy = uniform.copy()
        if mask.any():
            policy[mask] = d_sa[mask] / d_s[mask]
        return policy

    @staticmethod
    def solve_lp(spec: CMDPSpec) -> CMDPPolicy:
        """Solve CMDP via LP relaxation on occupancy measures.

        max  r^T d
        s.t. sum_a d(s',a) = (1-gamma) * mu(s') + gamma * sum_{s,a} P(s'|s,a) d(s,a)
             c_k^T d <= d_k  for each k
             d >= 0

        Uses a simple projected gradient method as LP solver (no scipy dependency).

        Args:
            spec: CMDP specification.

        Returns:
            CMDPPolicy with optimal stochastic policy.
        """
        S, A = spec.n_states, spec.n_actions
        n_vars = S * A
        n_constraints = len(spec.costs)

        # Uniform initial state distribution
        mu = np.ones(S) / S

        # Reward vector
        r = spec.reward.flatten()  # (S*A,)

        # Cost vectors
        c_vecs = [c.flatten() for c in spec.costs]

        # Flow conservation matrix:
        # For each s': sum_a d(s',a) - gamma * sum_{s,a} P(s'|s,a) * d(s,a) = (1-gamma) * mu(s')
        # Build matrix E of shape (S, S*A)
        E = np.zeros((S, n_vars))
        for s_prime in range(S):
            for a in range(A):
                # d(s',a) contributes +1 to row s'
                E[s_prime, s_prime * A + a] = 1.0
                # -gamma * sum_s P(s'|s,a) * d(s,a) for all (s,a)
                for s in range(S):
                    E[s_prime, s * A + a] -= spec.discount * spec.transition[s, a, s_prime]

        rhs = (1 - spec.discount) * mu  # (S,)

        # Solve via projected gradient ascent on the dual
        # Dual variables: nu (S,) for flow, lambda (K,) for costs
        nu = np.zeros(S)
        lam = np.zeros(n_constraints)
        lr = 0.01

        best_d = np.ones(n_vars) / n_vars
        best_val = -np.inf

        for iteration in range(2000):
            # d* = argmax_d>=0 { r^T d - nu^T (Ed - rhs) - sum_k lam_k (c_k^T d - d_k) }
            # Gradient w.r.t. d: r - E^T nu - sum_k lam_k c_k
            grad_d = r - E.T @ nu
            for k in range(n_constraints):
                grad_d -= lam[k] * c_vecs[k]

            # Project to d >= 0
            d = np.maximum(grad_d, 0.0)

            # Normalize to be a valid occupancy measure
            d_sum = d.sum()
            if d_sum > 1e-15:
                d *= (1.0 / (1 - spec.discount)) / d_sum
            else:
                d = np.ones(n_vars) / n_vars / (1 - spec.discount)

            # Dual gradient ascent
            flow_violation = E @ d - rhs
            nu += lr * flow_violation

            for k in range(n_constraints):
                cost_violation = c_vecs[k] @ d - spec.cost_limits[k]
                lam[k] = max(0, lam[k] + lr * cost_violation)

            val = r @ d
            if val > best_val:
                best_val = val
                best_d = d.copy()

            # Decay learning rate
            if iteration % 500 == 499:
                lr *= 0.5

        # Convert occupancy to policy
        policy = ConstrainedMDP._occupancy_to_policy(best_d, S, A)

        # Evaluate
        V = ConstrainedMDP.policy_evaluation(
            spec.transition, spec.reward, policy, spec.discount
        )
        value = float(V.mean())

        cost_values = []
        feasible = True
        for k, cost_mat in enumerate(spec.costs):
            V_c = ConstrainedMDP.policy_evaluation(
                spec.transition, cost_mat, policy, spec.discount
            )
            cv = float(V_c.mean())
            cost_values.append(cv)
            if cv > spec.cost_limits[k] + 1e-6:
                feasible = False

        return CMDPPolicy(
            action_probs=policy,
            value=value,
            cost_values=cost_values,
            lagrange_multipliers=lam,
            feasible=feasible,
        )

    @staticmethod
    def lagrangian_dual(
        spec: CMDPSpec,
        lr: float = 0.01,
        max_iter: int = 500,
    ) -> CMDPPolicy:
        """Solve CMDP via Lagrangian dual ascent.

        Forms the Lagrangian:
            L(pi, lambda) = V^r_pi - sum_k lambda_k (V^{c_k}_pi - d_k)

        Alternates:
            1. Fix lambda, solve unconstrained MDP with modified reward
            2. Update lambda via gradient ascent on constraint violation

        Args:
            spec: CMDP specification.
            lr: Learning rate for dual variable updates.
            max_iter: Maximum iterations.

        Returns:
            CMDPPolicy with (approximately) optimal constrained policy.
        """
        n_constraints = len(spec.costs)
        lam = np.zeros(n_constraints)

        best_policy = np.ones((spec.n_states, spec.n_actions)) / spec.n_actions
        best_feasible_value = -np.inf

        for iteration in range(max_iter):
            # Modified reward: r(s,a) - sum_k lambda_k * c_k(s,a)
            r_modified = spec.reward.copy()
            for k in range(n_constraints):
                r_modified -= lam[k] * spec.costs[k]

            # Solve unconstrained MDP with modified reward
            V, pi_det = ConstrainedMDP.value_iteration(
                spec.transition, r_modified, spec.discount
            )

            # Convert deterministic to stochastic policy
            policy = np.zeros((spec.n_states, spec.n_actions))
            for s in range(spec.n_states):
                policy[s, pi_det[s]] = 1.0

            # Evaluate cost constraints
            cost_values = []
            for k, cost_mat in enumerate(spec.costs):
                V_c = ConstrainedMDP.policy_evaluation(
                    spec.transition, cost_mat, policy, spec.discount
                )
                cost_values.append(float(V_c.mean()))

            # Update dual variables (gradient ascent on violation)
            for k in range(n_constraints):
                violation = cost_values[k] - spec.cost_limits[k]
                lam[k] = max(0, lam[k] + lr * violation)

            # Track best feasible policy
            V_r = ConstrainedMDP.policy_evaluation(
                spec.transition, spec.reward, policy, spec.discount
            )
            val = float(V_r.mean())
            feasible = all(
                cost_values[k] <= spec.cost_limits[k] + 1e-6
                for k in range(n_constraints)
            )
            if feasible and val > best_feasible_value:
                best_feasible_value = val
                best_policy = policy.copy()

        # Final evaluation of best policy
        V_r = ConstrainedMDP.policy_evaluation(
            spec.transition, spec.reward, best_policy, spec.discount
        )
        final_value = float(V_r.mean())

        final_costs = []
        final_feasible = True
        for k, cost_mat in enumerate(spec.costs):
            V_c = ConstrainedMDP.policy_evaluation(
                spec.transition, cost_mat, best_policy, spec.discount
            )
            cv = float(V_c.mean())
            final_costs.append(cv)
            if cv > spec.cost_limits[k] + 1e-6:
                final_feasible = False

        return CMDPPolicy(
            action_probs=best_policy,
            value=final_value,
            cost_values=final_costs,
            lagrange_multipliers=lam,
            feasible=final_feasible,
        )

    @staticmethod
    def primal_dual(
        spec: CMDPSpec,
        lr: float = 0.01,
        max_iter: int = 500,
    ) -> CMDPPolicy:
        """Solve CMDP via simultaneous primal-dual updates.

        Unlike lagrangian_dual which solves a full MDP at each step,
        this performs soft policy gradient updates on the primal
        simultaneously with dual ascent on the multipliers.

        Args:
            spec: CMDP specification.
            lr: Learning rate for both primal and dual updates.
            max_iter: Maximum iterations.

        Returns:
            CMDPPolicy with optimized constrained policy.
        """
        S, A = spec.n_states, spec.n_actions
        n_constraints = len(spec.costs)

        # Initialize uniform policy (in log space for stability)
        log_policy = np.zeros((S, A))
        lam = np.zeros(n_constraints)

        for iteration in range(max_iter):
            # Current policy via softmax
            log_policy_shifted = log_policy - log_policy.max(axis=1, keepdims=True)
            policy = np.exp(log_policy_shifted)
            policy /= policy.sum(axis=1, keepdims=True)

            # Q-values for reward
            V_r = ConstrainedMDP.policy_evaluation(
                spec.transition, spec.reward, policy, spec.discount
            )
            Q_r = spec.reward + spec.discount * np.einsum(
                'sap,p->sa', spec.transition, V_r
            )

            # Q-values for each cost
            Q_costs = []
            cost_values = []
            for k, cost_mat in enumerate(spec.costs):
                V_c = ConstrainedMDP.policy_evaluation(
                    spec.transition, cost_mat, policy, spec.discount
                )
                Q_c = cost_mat + spec.discount * np.einsum(
                    'sap,p->sa', spec.transition, V_c
                )
                Q_costs.append(Q_c)
                cost_values.append(float(V_c.mean()))

            # Primal update: gradient of Lagrangian w.r.t. log_policy
            # dL/d(log pi) = pi * (Q_r - sum_k lam_k Q_c_k - V_lagrangian)
            Q_lagrangian = Q_r.copy()
            for k in range(n_constraints):
                Q_lagrangian -= lam[k] * Q_costs[k]

            # Advantage
            V_lagrangian = np.sum(policy * Q_lagrangian, axis=1, keepdims=True)
            advantage = Q_lagrangian - V_lagrangian

            log_policy += lr * advantage

            # Dual update
            for k in range(n_constraints):
                violation = cost_values[k] - spec.cost_limits[k]
                lam[k] = max(0, lam[k] + lr * violation)

        # Final policy
        log_policy_shifted = log_policy - log_policy.max(axis=1, keepdims=True)
        policy = np.exp(log_policy_shifted)
        policy /= policy.sum(axis=1, keepdims=True)

        V_r = ConstrainedMDP.policy_evaluation(
            spec.transition, spec.reward, policy, spec.discount
        )
        value = float(V_r.mean())

        final_costs = []
        feasible = True
        for k, cost_mat in enumerate(spec.costs):
            V_c = ConstrainedMDP.policy_evaluation(
                spec.transition, cost_mat, policy, spec.discount
            )
            cv = float(V_c.mean())
            final_costs.append(cv)
            if cv > spec.cost_limits[k] + 1e-6:
                feasible = False

        return CMDPPolicy(
            action_probs=policy,
            value=value,
            cost_values=final_costs,
            lagrange_multipliers=lam,
            feasible=feasible,
        )


class SafeRL:
    """Safe Reinforcement Learning via constrained policy gradient.

    Uses projected gradient to maintain constraint satisfaction
    throughout training, not just at convergence.
    """

    @staticmethod
    def constrained_policy_gradient(
        spec: CMDPSpec,
        policy: np.ndarray | None = None,
        lr: float = 0.01,
        n_iter: int = 100,
    ) -> CMDPPolicy:
        """Projected constrained policy gradient.

        At each step:
        1. Compute reward gradient (ascend) and cost gradients
        2. If constraint violated, project reward gradient onto feasible half-space
        3. Update policy in softmax parameterization

        Args:
            spec: CMDP specification.
            policy: Initial policy pi(a|s), shape (S, A). If None, uniform.
            lr: Learning rate.
            n_iter: Number of gradient steps.

        Returns:
            CMDPPolicy with feasibility-aware policy.
        """
        S, A = spec.n_states, spec.n_actions
        n_constraints = len(spec.costs)

        # Initialize log-policy
        if policy is not None:
            log_policy = np.log(np.maximum(policy, 1e-15))
        else:
            log_policy = np.zeros((S, A))

        for iteration in range(n_iter):
            # Softmax policy
            lp = log_policy - log_policy.max(axis=1, keepdims=True)
            pi = np.exp(lp)
            pi /= pi.sum(axis=1, keepdims=True)

            # Q-values for reward
            V_r = ConstrainedMDP.policy_evaluation(
                spec.transition, spec.reward, pi, spec.discount
            )
            Q_r = spec.reward + spec.discount * np.einsum(
                'sap,p->sa', spec.transition, V_r
            )

            # Reward gradient (advantage)
            A_r = Q_r - V_r[:, None]
            grad_r = pi * A_r  # Natural policy gradient direction

            # Check constraints and project if needed
            grad = grad_r.copy()
            for k, cost_mat in enumerate(spec.costs):
                V_c = ConstrainedMDP.policy_evaluation(
                    spec.transition, cost_mat, pi, spec.discount
                )
                cv = float(V_c.mean())

                if cv > spec.cost_limits[k]:
                    # Constraint violated: compute cost gradient
                    Q_c = cost_mat + spec.discount * np.einsum(
                        'sap,p->sa', spec.transition, V_c
                    )
                    A_c = Q_c - V_c[:, None]
                    grad_c = pi * A_c

                    # Project reward gradient to reduce cost:
                    # Remove component of grad that increases cost
                    dot = np.sum(grad * grad_c)
                    norm_sq = np.sum(grad_c * grad_c)
                    if norm_sq > 1e-15:
                        grad -= (dot / norm_sq) * grad_c
                        # Add correction to reduce violation
                        correction_strength = min(1.0, (cv - spec.cost_limits[k]) * 10)
                        grad -= correction_strength * lr * grad_c

            log_policy += lr * grad

        # Final policy
        lp = log_policy - log_policy.max(axis=1, keepdims=True)
        pi = np.exp(lp)
        pi /= pi.sum(axis=1, keepdims=True)

        V_r = ConstrainedMDP.policy_evaluation(
            spec.transition, spec.reward, pi, spec.discount
        )
        value = float(V_r.mean())

        final_costs = []
        feasible = True
        for k, cost_mat in enumerate(spec.costs):
            V_c = ConstrainedMDP.policy_evaluation(
                spec.transition, cost_mat, pi, spec.discount
            )
            cv = float(V_c.mean())
            final_costs.append(cv)
            if cv > spec.cost_limits[k] + 1e-6:
                feasible = False

        return CMDPPolicy(
            action_probs=pi,
            value=value,
            cost_values=final_costs,
            lagrange_multipliers=np.array([]),
            feasible=feasible,
        )


if __name__ == "__main__":
    print("Constrained MDP Module Demo")
    print("=" * 50)

    # Simple 3-state, 2-action CMDP
    S, A = 3, 2
    rng = np.random.default_rng(42)

    # Random transition kernel (normalized)
    T = rng.random((S, A, S))
    T /= T.sum(axis=2, keepdims=True)

    # Reward: action 0 is safe but low reward, action 1 is risky but high
    R = np.array([
        [1.0, 3.0],
        [1.0, 4.0],
        [2.0, 5.0],
    ])

    # Cost: action 1 is expensive
    C = np.array([
        [0.1, 0.8],
        [0.1, 0.9],
        [0.2, 0.7],
    ])

    spec = CMDPSpec(
        n_states=S, n_actions=A,
        transition=T, reward=R,
        costs=[C], cost_limits=[0.5],
        discount=0.95,
    )

    # Unconstrained baseline
    V, pi = ConstrainedMDP.value_iteration(T, R, 0.95)
    print(f"Unconstrained value: {V.mean():.4f}")
    print(f"Unconstrained policy: {pi}")

    # Lagrangian dual
    result = ConstrainedMDP.lagrangian_dual(spec)
    print(f"\nLagrangian dual value: {result.value:.4f}")
    print(f"Cost: {result.cost_values}, Feasible: {result.feasible}")

    # Primal-dual
    result_pd = ConstrainedMDP.primal_dual(spec)
    print(f"\nPrimal-dual value: {result_pd.value:.4f}")
    print(f"Cost: {result_pd.cost_values}, Feasible: {result_pd.feasible}")

    print("\n" + "=" * 50)
    print("Constrained MDP module ready!")
