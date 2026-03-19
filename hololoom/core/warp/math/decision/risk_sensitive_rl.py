"""
Risk-Sensitive Reinforcement Learning — CVaR, Entropic Risk, Distributional
============================================================================

Standard RL optimizes expected return. Risk-sensitive RL optimizes risk-adjusted
objectives that account for the distribution of returns, not just the mean.

Classes:
    CVaRMDP: Conditional Value-at-Risk optimization
    EntropicRisk: Exponential utility / certainty equivalent
    DistributionalRL: Learn full return distributions (C51 algorithm)
    MeanVarianceRL: Markowitz-style mean-variance tradeoff

Key Concepts:
    - VaR(alpha): worst return in the best (1-alpha) fraction of outcomes
    - CVaR(alpha): expected return in the worst alpha fraction of outcomes
    - Entropic risk: certainty equivalent under exponential utility
    - Return distribution: full histogram of possible returns

Applications:
    - Conservative scoring (SOUS: avoid risky recipe suggestions)
    - Robust policy selection under uncertainty
    - Safety-aware planning (avoid catastrophic failures)
    - Portfolio-style multi-objective optimization

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskMetrics:
    """
    Comprehensive risk metrics for a return distribution.

    Attributes:
        cvar: Conditional Value-at-Risk (expected shortfall)
        var: Value-at-Risk (quantile threshold)
        entropic_risk: Certainty equivalent under exponential utility
        expected_return: Plain expected value
        worst_case: Minimum observed return
    """
    cvar: float
    var: float
    entropic_risk: float
    expected_return: float
    worst_case: float


def compute_risk_metrics(
    returns: np.ndarray,
    alpha: float = 0.05,
    beta: float = 1.0,
) -> RiskMetrics:
    """
    Compute all risk metrics for a sample of returns.

    Args:
        returns: Array of return samples, shape (n_samples,)
        alpha: Risk level for CVaR/VaR (fraction of worst outcomes)
        beta: Risk sensitivity for entropic risk

    Returns:
        RiskMetrics with all computed values
    """
    if len(returns) == 0:
        return RiskMetrics(
            cvar=0.0, var=0.0, entropic_risk=0.0,
            expected_return=0.0, worst_case=0.0,
        )

    sorted_returns = np.sort(returns)
    n = len(sorted_returns)

    # VaR: alpha-quantile
    var_idx = max(0, int(np.ceil(alpha * n)) - 1)
    var = float(sorted_returns[var_idx])

    # CVaR: mean of returns below VaR
    n_tail = max(1, int(np.ceil(alpha * n)))
    cvar = float(np.mean(sorted_returns[:n_tail]))

    # Entropic risk: -(1/beta) * log(E[exp(-beta * R)])
    entropic = EntropicRisk.certainty_equivalent(returns, beta)

    return RiskMetrics(
        cvar=cvar,
        var=var,
        entropic_risk=entropic,
        expected_return=float(np.mean(returns)),
        worst_case=float(np.min(returns)),
    )


class CVaRMDP:
    """
    CVaR-constrained Markov Decision Process.

    Optimizes Conditional Value-at-Risk instead of expected return:
        max_pi CVaR_alpha[ sum_t gamma^t R(s_t, a_t) ]

    CVaR_alpha = E[R | R <= VaR_alpha] — the expected return in the worst
    alpha fraction of outcomes.

    Algorithm: augmented MDP with an extra state variable tracking
    accumulated reward, then linear programming over the augmented state.

    For tractability, we discretize the return axis and solve via
    modified value iteration where the Bellman operator optimizes CVaR.
    """

    @staticmethod
    def compute_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Empirical Conditional Value-at-Risk.

        CVaR_alpha = (1/alpha) * integral_0^alpha VaR_u du
                   ≈ mean of returns in the bottom alpha fraction.

        Args:
            returns: Sample returns, shape (n,)
            alpha: Risk level (e.g., 0.05 = worst 5%)

        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        sorted_r = np.sort(returns)
        n_tail = max(1, int(np.ceil(alpha * len(sorted_r))))
        return float(np.mean(sorted_r[:n_tail]))

    @staticmethod
    def compute_var(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Empirical Value-at-Risk.

        VaR_alpha = inf{x : P(R <= x) >= alpha}

        Args:
            returns: Sample returns, shape (n,)
            alpha: Risk level

        Returns:
            VaR value (alpha-quantile of returns)
        """
        if len(returns) == 0:
            return 0.0
        return float(np.quantile(returns, alpha))

    def solve(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        discount: float,
        alpha: float = 0.05,
        n_iter: int = 100,
        n_atoms: int = 51,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        CVaR value iteration via distributional approach.

        Instead of tracking scalar values, maintain return distributions
        and optimize CVaR of those distributions.

        The approach:
        1. Use distributional RL to compute return distributions Z(s,a)
        2. For each (s,a), compute CVaR of Z(s,a)
        3. Policy = argmax_a CVaR_alpha[Z(s,a)]

        Args:
            transition: Transition probs, shape (n_states, n_actions, n_states)
            reward: Reward, shape (n_states,) or (n_states, n_actions)
            discount: Discount factor
            alpha: CVaR risk level
            n_iter: Number of iterations
            n_atoms: Atoms for distributional representation

        Returns:
            V: CVaR values per state, shape (n_states,)
            policy: Risk-sensitive policy, shape (n_states, n_actions)
        """
        n_states, n_actions, _ = transition.shape

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_actions))
        else:
            R = reward

        # Determine return range
        r_min = np.min(R)
        r_max = np.max(R)
        v_min = r_min / max(1 - discount, 1e-6)
        v_max = r_max / max(1 - discount, 1e-6)

        # Atoms for distributional representation
        atoms = np.linspace(v_min, v_max, n_atoms)
        delta_z = (v_max - v_min) / max(n_atoms - 1, 1)

        # Distribution Z(s,a) represented as probabilities over atoms
        Z = np.ones((n_states, n_actions, n_atoms)) / n_atoms

        for iteration in range(n_iter):
            Z_new = np.zeros_like(Z)

            for s in range(n_states):
                for a in range(n_actions):
                    # Target atoms: r + gamma * z_j
                    for s_next in range(n_states):
                        p_trans = transition[s, a, s_next]
                        if p_trans < 1e-10:
                            continue

                        # Pick greedy action in next state based on CVaR
                        cvar_next = np.zeros(n_actions)
                        for a_next in range(n_actions):
                            cvar_next[a_next] = self._distribution_cvar(
                                Z[s_next, a_next], atoms, alpha
                            )
                        a_star = int(np.argmax(cvar_next))

                        for j in range(n_atoms):
                            # Target atom
                            target = R[s, a] + discount * atoms[j]
                            target = np.clip(target, v_min, v_max)

                            # Project onto atom grid
                            b = (target - v_min) / max(delta_z, 1e-10)
                            l_idx = int(np.floor(b))
                            u_idx = min(l_idx + 1, n_atoms - 1)
                            l_idx = max(0, min(l_idx, n_atoms - 1))

                            u_frac = b - l_idx
                            l_frac = 1.0 - u_frac

                            prob = p_trans * Z[s_next, a_star, j]
                            Z_new[s, a, l_idx] += prob * l_frac
                            Z_new[s, a, u_idx] += prob * u_frac

            # Normalize distributions
            Z_sums = np.sum(Z_new, axis=2, keepdims=True)
            Z = np.where(Z_sums > 1e-10, Z_new / Z_sums, 1.0 / n_atoms)

        # Extract CVaR values and policy
        V = np.zeros(n_states)
        policy = np.zeros((n_states, n_actions))

        for s in range(n_states):
            cvar_values = np.array([
                self._distribution_cvar(Z[s, a], atoms, alpha)
                for a in range(n_actions)
            ])
            best_a = int(np.argmax(cvar_values))
            V[s] = cvar_values[best_a]
            policy[s, best_a] = 1.0

        return V, policy

    @staticmethod
    def _distribution_cvar(
        probs: np.ndarray, atoms: np.ndarray, alpha: float
    ) -> float:
        """
        Compute CVaR from a categorical distribution.

        Walk from the left tail, accumulate probability mass up to alpha,
        compute weighted average of atoms in that tail.
        """
        if alpha <= 0:
            return float(atoms[0])
        if alpha >= 1:
            return float(np.dot(probs, atoms))

        remaining_alpha = alpha
        cvar_sum = 0.0

        for i in range(len(atoms)):
            p = probs[i]
            if p <= 0:
                continue
            contribution = min(p, remaining_alpha)
            cvar_sum += contribution * atoms[i]
            remaining_alpha -= contribution
            if remaining_alpha <= 1e-10:
                break

        return float(cvar_sum / max(alpha, 1e-10))


class EntropicRisk:
    """
    Entropic Risk Measure / Exponential Utility.

    Uses the certainty equivalent under exponential utility:
        rho_beta(R) = -(1/beta) * log(E[exp(-beta * R)])

    - beta > 0: risk-averse (penalizes variance, prefers certain outcomes)
    - beta < 0: risk-seeking (prefers variance, gambles for high returns)
    - beta -> 0: risk-neutral (recovers expected value)

    The beauty of entropic risk: it has a clean Bellman equation.
    The risk-sensitive value function satisfies:
        V(s) = max_a [ R(s,a) - (1/beta) * log(sum_s' T(s'|s,a) * exp(-beta * gamma * V(s'))) ]
    """

    @staticmethod
    def certainty_equivalent(
        returns: np.ndarray, beta: float
    ) -> float:
        """
        Compute certainty equivalent under exponential utility.

            CE = -(1/beta) * log(E[exp(-beta * R)])

        For beta=0, returns E[R] (risk-neutral).
        Numerically stable via log-sum-exp trick.

        Args:
            returns: Sample returns
            beta: Risk sensitivity parameter

        Returns:
            Certainty equivalent value
        """
        if len(returns) == 0:
            return 0.0
        if abs(beta) < 1e-10:
            return float(np.mean(returns))

        # Log-sum-exp trick for stability
        scaled = -beta * returns
        max_scaled = np.max(scaled)
        log_expectation = max_scaled + np.log(
            np.mean(np.exp(scaled - max_scaled)) + 1e-10
        )
        return float(-log_expectation / beta)

    def solve(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        discount: float,
        beta: float = 1.0,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Risk-sensitive value iteration with exponential utility.

        Bellman equation:
            V(s) = max_a [ R(s,a) - (1/beta) * log(sum_s' T(s'|s,a) * exp(-beta*gamma*V(s'))) ]

        This naturally penalizes transitions to uncertain outcomes (when beta > 0).

        Args:
            transition: Transition probs, shape (n_states, n_actions, n_states)
            reward: Reward, shape (n_states,) or (n_states, n_actions)
            discount: Discount factor
            beta: Risk sensitivity (>0 averse, <0 seeking)
            n_iter: Number of iterations

        Returns:
            V: Risk-sensitive value function, shape (n_states,)
            policy: Risk-sensitive policy, shape (n_states, n_actions)
        """
        n_states, n_actions, _ = transition.shape

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_actions))
        else:
            R = reward

        V = np.zeros(n_states)

        for _ in range(n_iter):
            Q = np.zeros((n_states, n_actions))

            for s in range(n_states):
                for a in range(n_actions):
                    if abs(beta) < 1e-10:
                        # Risk-neutral: standard Bellman
                        Q[s, a] = R[s, a] + discount * np.dot(
                            transition[s, a], V
                        )
                    else:
                        # Risk-sensitive Bellman
                        # -(1/beta) * log(sum_s' T(s'|s,a) * exp(-beta*gamma*V(s')))
                        exponents = -beta * discount * V
                        max_exp = np.max(exponents)
                        log_term = max_exp + np.log(
                            np.dot(transition[s, a], np.exp(exponents - max_exp))
                            + 1e-10
                        )
                        Q[s, a] = R[s, a] - log_term / beta

            V_new = np.max(Q, axis=1)

            if np.max(np.abs(V_new - V)) < 1e-8:
                break
            V = V_new

        # Policy
        Q_final = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                if abs(beta) < 1e-10:
                    Q_final[s, a] = R[s, a] + discount * np.dot(
                        transition[s, a], V
                    )
                else:
                    exponents = -beta * discount * V
                    max_exp = np.max(exponents)
                    log_term = max_exp + np.log(
                        np.dot(transition[s, a], np.exp(exponents - max_exp))
                        + 1e-10
                    )
                    Q_final[s, a] = R[s, a] - log_term / beta

        policy = np.zeros((n_states, n_actions))
        best_actions = np.argmax(Q_final, axis=1)
        policy[np.arange(n_states), best_actions] = 1.0

        return V, policy


class DistributionalRL:
    """
    Distributional Reinforcement Learning — C51 Algorithm (Bellemare et al., 2017).

    Instead of learning E[R], learn the full return distribution Z(s,a).
    Represent Z as a categorical distribution over n_atoms fixed atoms.

    Why distributional:
        - Richer signal for learning (preserves multimodality)
        - Enables any risk measure as a post-hoc computation
        - Empirically more stable than scalar RL
        - Natural interface to risk-sensitive policies

    The distribution is updated via the distributional Bellman operator:
        T Z(s,a) = R(s,a) + gamma * Z(s', a*)
    where a* = argmax_a E[Z(s',a)] (or any risk measure).
    """

    @staticmethod
    def project_distribution(
        target_z: np.ndarray,
        target_probs: np.ndarray,
        v_min: float,
        v_max: float,
        n_atoms: int,
    ) -> np.ndarray:
        """
        Project a set of target atoms onto the fixed atom grid.

        Each target atom t_j with probability p_j is distributed to the
        two nearest grid atoms proportionally to distance.

        Args:
            target_z: Target atom locations, shape (n_target,)
            target_probs: Probabilities of target atoms, shape (n_target,)
            v_min: Minimum atom value
            v_max: Maximum atom value
            n_atoms: Number of atoms in the grid

        Returns:
            Projected distribution, shape (n_atoms,)
        """
        delta_z = (v_max - v_min) / max(n_atoms - 1, 1)
        projected = np.zeros(n_atoms)

        for j in range(len(target_z)):
            tz = np.clip(target_z[j], v_min, v_max)
            b = (tz - v_min) / max(delta_z, 1e-10)
            l = int(np.floor(b))
            u = min(l + 1, n_atoms - 1)
            l = max(0, min(l, n_atoms - 1))

            u_frac = b - l
            l_frac = 1.0 - u_frac

            projected[l] += target_probs[j] * l_frac
            projected[u] += target_probs[j] * u_frac

        # Normalize
        total = np.sum(projected)
        if total > 1e-10:
            projected /= total

        return projected

    def c51(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        discount: float,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        C51: Categorical Distributional RL.

        Learns Z(s,a) as a categorical distribution over n_atoms atoms
        uniformly spaced in [v_min, v_max].

        Update rule:
            1. For each (s,a), compute target distribution:
               T_z_j = r + gamma * z_j  for each atom j
            2. Project target onto atom grid
            3. Cross-entropy loss between current and projected target

        Args:
            transition: Transition probs, shape (n_states, n_actions, n_states)
            reward: Reward, shape (n_states,) or (n_states, n_actions)
            discount: Discount factor
            n_atoms: Number of atoms
            v_min: Minimum return value
            v_max: Maximum return value
            n_iter: Number of iterations

        Returns:
            Z: Return distributions, shape (n_states, n_actions, n_atoms)
            policy: Greedy policy (argmax expected value), shape (n_states, n_actions)
        """
        n_states, n_actions, _ = transition.shape

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_actions))
        else:
            R = reward

        atoms = np.linspace(v_min, v_max, n_atoms)
        delta_z = (v_max - v_min) / max(n_atoms - 1, 1)

        # Initialize uniform distributions
        Z = np.ones((n_states, n_actions, n_atoms)) / n_atoms

        for iteration in range(n_iter):
            Z_new = np.zeros_like(Z)

            for s in range(n_states):
                for a in range(n_actions):
                    for s_next in range(n_states):
                        p_trans = transition[s, a, s_next]
                        if p_trans < 1e-10:
                            continue

                        # Greedy action in next state (by expected value)
                        expected_next = np.array([
                            np.dot(Z[s_next, a_next], atoms)
                            for a_next in range(n_actions)
                        ])
                        a_star = int(np.argmax(expected_next))

                        # Target atoms
                        target_z = R[s, a] + discount * atoms
                        target_z = np.clip(target_z, v_min, v_max)

                        # Project target distribution onto atoms
                        for j in range(n_atoms):
                            tz = target_z[j]
                            b = (tz - v_min) / max(delta_z, 1e-10)
                            l = int(np.floor(b))
                            u = min(l + 1, n_atoms - 1)
                            l = max(0, min(l, n_atoms - 1))

                            u_frac = b - l
                            l_frac = 1.0 - u_frac

                            prob = p_trans * Z[s_next, a_star, j]
                            Z_new[s, a, l] += prob * l_frac
                            Z_new[s, a, u] += prob * u_frac

            # Normalize
            Z_sums = np.sum(Z_new, axis=2, keepdims=True)
            Z = np.where(Z_sums > 1e-10, Z_new / Z_sums, 1.0 / n_atoms)

        # Policy from expected values
        expected = np.sum(Z * atoms[np.newaxis, np.newaxis, :], axis=2)
        policy = np.zeros((n_states, n_actions))
        best_actions = np.argmax(expected, axis=1)
        policy[np.arange(n_states), best_actions] = 1.0

        return Z, policy


class MeanVarianceRL:
    """
    Mean-Variance Reinforcement Learning (Markowitz for MDPs).

    Optimize the mean-variance objective:
        max_pi  E[R] - lambda * Var[R]

    where lambda controls the risk-return tradeoff:
        - lambda = 0: pure expected return (risk-neutral)
        - lambda > 0: penalize variance (risk-averse)
        - lambda < 0: reward variance (risk-seeking, unusual)

    Implementation uses reward augmentation: track both E[R] and E[R^2]
    to compute Var[R] = E[R^2] - E[R]^2.

    Bellman equations for the two moments:
        V_1(s) = max_a [ R(s,a) + gamma * sum_s' T(s'|s,a) V_1(s') ]
        V_2(s) = max_a [ R(s,a)^2 + 2*gamma*R(s,a)*V_1(s')
                          + gamma^2 * sum_s' T(s'|s,a) V_2(s') ]

    Then: mean-variance value = V_1(s) - lambda * (V_2(s) - V_1(s)^2)
    """

    def solve(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        discount: float,
        lambda_var: float = 0.1,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mean-variance value iteration.

        Alternates between:
        1. Compute first moment V_1 (expected return) via standard VI
        2. Compute second moment V_2 (expected squared return) under same policy
        3. Adjust policy to optimize V_1 - lambda * (V_2 - V_1^2)

        Args:
            transition: Transition probs, shape (n_states, n_actions, n_states)
            reward: Reward, shape (n_states,) or (n_states, n_actions)
            discount: Discount factor
            lambda_var: Variance penalty coefficient
            n_iter: Number of iterations

        Returns:
            V: Mean-variance adjusted values, shape (n_states,)
            policy: Risk-adjusted policy, shape (n_states, n_actions)
        """
        n_states, n_actions, _ = transition.shape

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_actions))
        else:
            R = reward

        V1 = np.zeros(n_states)  # E[R]
        V2 = np.zeros(n_states)  # E[R^2]
        policy = np.ones((n_states, n_actions)) / n_actions

        for outer in range(n_iter):
            # Step 1: Compute V1 under current policy via value iteration
            for _ in range(20):
                Q1 = R + discount * np.einsum("ijk,k->ij", transition, V1)
                V1_new = np.sum(policy * Q1, axis=1)
                if np.max(np.abs(V1_new - V1)) < 1e-8:
                    break
                V1 = V1_new

            # Step 2: Compute V2 under current policy
            # V2(s) = E_pi[R(s,a)^2 + 2*gamma*R(s,a)*E[V1(s')]
            #          + gamma^2 * E[V2(s')]]
            for _ in range(20):
                EV1_next = np.einsum("ijk,k->ij", transition, V1)
                Q2 = (
                    R ** 2
                    + 2 * discount * R * EV1_next
                    + discount ** 2 * np.einsum("ijk,k->ij", transition, V2)
                )
                V2_new = np.sum(policy * Q2, axis=1)
                if np.max(np.abs(V2_new - V2)) < 1e-8:
                    break
                V2 = V2_new

            # Step 3: Update policy to optimize mean - lambda * variance
            # Q_mv(s,a) = Q1(s,a) - lambda * (Q2(s,a) - Q1(s,a)^2)
            Q1_full = R + discount * np.einsum("ijk,k->ij", transition, V1)
            EV1_next = np.einsum("ijk,k->ij", transition, V1)
            Q2_full = (
                R ** 2
                + 2 * discount * R * EV1_next
                + discount ** 2 * np.einsum("ijk,k->ij", transition, V2)
            )

            # Variance of Q: E[R^2] - E[R]^2
            Q_var = Q2_full - Q1_full ** 2
            Q_var = np.maximum(Q_var, 0.0)  # Numerical stability

            Q_mv = Q1_full - lambda_var * Q_var

            # Softmax policy from mean-variance Q-values (for smooth optimization)
            Q_shifted = Q_mv - np.max(Q_mv, axis=1, keepdims=True)
            exp_Q = np.exp(Q_shifted / max(0.1, 1.0 - outer / n_iter))
            policy = exp_Q / (np.sum(exp_Q, axis=1, keepdims=True) + 1e-10)

        # Final value: mean-variance adjusted
        V = V1 - lambda_var * np.maximum(V2 - V1 ** 2, 0.0)

        # Final deterministic policy
        Q1_final = R + discount * np.einsum("ijk,k->ij", transition, V1)
        EV1_final = np.einsum("ijk,k->ij", transition, V1)
        Q2_final = (
            R ** 2
            + 2 * discount * R * EV1_final
            + discount ** 2 * np.einsum("ijk,k->ij", transition, V2)
        )
        Q_var_final = np.maximum(Q2_final - Q1_final ** 2, 0.0)
        Q_mv_final = Q1_final - lambda_var * Q_var_final

        policy_final = np.zeros((n_states, n_actions))
        best_actions = np.argmax(Q_mv_final, axis=1)
        policy_final[np.arange(n_states), best_actions] = 1.0

        return V, policy_final
