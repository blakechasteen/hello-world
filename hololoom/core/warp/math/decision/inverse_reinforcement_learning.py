"""
Inverse Reinforcement Learning — Learn Reward Functions from Demonstrations
===========================================================================

Given expert demonstrations (state-action trajectories), recover the reward
function that the expert is implicitly optimizing. The recovered reward can
then be used to train new policies that generalize beyond the demonstrations.

Classes:
    MaxEntIRL: Maximum entropy IRL (Ziebart 2008)
    BayesianIRL: MCMC posterior inference over reward weights (Ramachandran 2007)
    ApprenticeshipLearning: Feature matching via LP/projection (Abbeel & Ng 2004)

Key Concepts:
    - Feature expectations: E[sum_t gamma^t * phi(s_t)] under a policy
    - MaxEnt: policy ∝ exp(reward), resolves ambiguity via maximum entropy
    - Bayesian: full posterior over reward weights, handles uncertainty
    - Apprenticeship: match expert's feature expectations, no explicit reward

Applications:
    - Learn scoring weights from user behavior (SOUS preference learning)
    - Imitation from demonstration without reward engineering
    - Preference elicitation from ranked examples
    - Transfer learning via reward portability

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Demonstration:
    """
    An expert demonstration trajectory.

    Attributes:
        states: State indices visited, shape (T,)
        actions: Actions taken, shape (T,)
        features: Optional precomputed features, shape (T, n_features)
    """
    states: np.ndarray
    actions: np.ndarray
    features: np.ndarray | None = None


@dataclass
class IRLResult:
    """
    Result of inverse reinforcement learning.

    Attributes:
        reward_weights: Learned linear reward weights, shape (n_features,)
        policy: Optimal policy under learned reward, shape (n_states, n_actions)
        log_likelihood: Log-likelihood of demonstrations under learned reward
        feature_expectations: Expected feature counts under learned policy
    """
    reward_weights: np.ndarray
    policy: np.ndarray
    log_likelihood: float
    feature_expectations: np.ndarray


class MaxEntIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning (Ziebart et al., 2008).

    Resolves the ambiguity in IRL by choosing the reward that makes the
    expert's behavior most likely under a maximum entropy model:

        P(trajectory | reward) ∝ exp(sum_t R(s_t))

    The gradient is elegantly simple:
        ∇L = μ_expert - μ_policy

    where μ are feature expectations (discounted feature counts).

    This is the workhorse IRL algorithm — clean gradients, convex
    optimization, and principled handling of stochastic experts.
    """

    @staticmethod
    def soft_value_iteration(
        reward: np.ndarray,
        transition: np.ndarray,
        discount: float,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Soft (entropy-regularized) value iteration.

        Computes the soft value function and corresponding Boltzmann policy:
            V(s) = softmax_a [R(s,a) + gamma * sum_s' T(s'|s,a) V(s')]
            pi(a|s) = exp(Q(s,a) - V(s))

        Args:
            reward: Reward per state, shape (n_states,) or (n_states, n_actions)
            transition: Transition probabilities, shape (n_states, n_actions, n_states)
            discount: Discount factor gamma
            n_iter: Number of iterations

        Returns:
            V: Soft value function, shape (n_states,)
            policy: Boltzmann policy, shape (n_states, n_actions)
        """
        n_states, n_actions, _ = transition.shape

        # Handle state-only reward by broadcasting
        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_actions))
        else:
            R = reward

        V = np.zeros(n_states)

        for _ in range(n_iter):
            # Q(s,a) = R(s,a) + gamma * sum_s' T(s'|s,a) * V(s')
            Q = R + discount * np.einsum("ijk,k->ij", transition, V)

            # V(s) = log(sum_a exp(Q(s,a)))  (log-sum-exp)
            Q_max = np.max(Q, axis=1, keepdims=True)
            V_new = np.squeeze(Q_max) + np.log(
                np.sum(np.exp(Q - Q_max), axis=1) + 1e-10
            )

            if np.max(np.abs(V_new - V)) < 1e-8:
                break
            V = V_new

        # Boltzmann policy
        Q = R + discount * np.einsum("ijk,k->ij", transition, V)
        Q_shifted = Q - np.max(Q, axis=1, keepdims=True)
        exp_Q = np.exp(Q_shifted)
        policy = exp_Q / (np.sum(exp_Q, axis=1, keepdims=True) + 1e-10)

        return V, policy

    @staticmethod
    def feature_expectations(
        policy: np.ndarray,
        features: np.ndarray,
        transition: np.ndarray,
        discount: float,
        initial: np.ndarray,
        n_steps: int = 200,
    ) -> np.ndarray:
        """
        Compute expected discounted feature counts under a policy.

            μ = E[sum_t gamma^t * phi(s_t)]

        Uses state visitation frequency computation:
            d_0 = initial distribution
            d_{t+1}(s') = gamma * sum_{s,a} d_t(s) * pi(a|s) * T(s'|s,a)
            μ = sum_t d_t^T * phi

        Args:
            policy: Policy, shape (n_states, n_actions)
            features: State features, shape (n_states, n_features)
            transition: Transition probs, shape (n_states, n_actions, n_states)
            discount: Discount factor
            initial: Initial state distribution, shape (n_states,)
            n_steps: Horizon for computing visitation frequencies

        Returns:
            Feature expectations, shape (n_features,)
        """
        n_states = features.shape[0]

        # State visitation frequencies
        d = initial.copy()
        total_d = d.copy()

        for t in range(1, n_steps):
            # d_{t+1}(s') = gamma * sum_{s,a} d_t(s) * pi(a|s) * T(s'|s,a)
            d_sa = d[:, np.newaxis] * policy  # (n_states, n_actions)
            d_next = discount * np.einsum("sa,sas'->s'", d_sa, transition)
            # Fix einsum notation for clarity
            d_next = discount * np.sum(
                d_sa[:, :, np.newaxis] * transition, axis=(0, 1)
            )
            total_d += d_next
            d = d_next

            if np.sum(d) < 1e-10:
                break

        # Feature expectations = sum of visitation-weighted features
        return total_d @ features

    def learn(
        self,
        demos: list[Demonstration],
        features: np.ndarray,
        transition: np.ndarray,
        discount: float = 0.99,
        lr: float = 0.01,
        n_iter: int = 200,
    ) -> IRLResult:
        """
        Learn reward weights via maximum entropy IRL.

        Gradient ascent on log-likelihood:
            ∇L = μ_expert - μ_learner

        where μ_expert is computed from demonstrations and μ_learner from
        the current soft-optimal policy.

        Args:
            demos: Expert demonstrations
            features: State features, shape (n_states, n_features)
            transition: Transition model, shape (n_states, n_actions, n_states)
            discount: Discount factor
            lr: Learning rate for gradient ascent
            n_iter: Maximum iterations

        Returns:
            IRLResult with learned reward weights and policy
        """
        n_states, n_features = features.shape
        n_actions = transition.shape[1]

        # Compute expert feature expectations from demonstrations
        expert_fe = np.zeros(n_features)
        total_steps = 0
        for demo in demos:
            for t, s in enumerate(demo.states):
                if demo.features is not None:
                    expert_fe += (discount ** t) * demo.features[t]
                else:
                    expert_fe += (discount ** t) * features[int(s)]
                total_steps += 1
        expert_fe /= max(len(demos), 1)

        # Initial state distribution from demonstrations
        initial = np.zeros(n_states)
        for demo in demos:
            if len(demo.states) > 0:
                initial[int(demo.states[0])] += 1.0
        initial = initial / (np.sum(initial) + 1e-10)

        # Initialize reward weights
        weights = np.zeros(n_features)
        best_ll = -np.inf
        best_weights = weights.copy()
        best_policy = None
        best_fe = None

        for iteration in range(n_iter):
            # Compute reward from weights
            reward = features @ weights  # (n_states,)

            # Solve soft MDP
            V, policy = self.soft_value_iteration(
                reward, transition, discount, n_iter=50
            )

            # Compute learner feature expectations
            learner_fe = self.feature_expectations(
                policy, features, transition, discount, initial
            )

            # Gradient = expert - learner
            grad = expert_fe - learner_fe
            weights += lr * grad

            # Log-likelihood: sum over demos of sum_t R(s_t)
            ll = 0.0
            for demo in demos:
                for t, s in enumerate(demo.states):
                    ll += (discount ** t) * reward[int(s)]
                # Subtract log partition (soft value at initial state)
                if len(demo.states) > 0:
                    ll -= V[int(demo.states[0])]
            ll /= max(len(demos), 1)

            if ll > best_ll:
                best_ll = ll
                best_weights = weights.copy()
                best_policy = policy.copy()
                best_fe = learner_fe.copy()

            # Convergence check
            if np.linalg.norm(grad) < 1e-5:
                break

        return IRLResult(
            reward_weights=best_weights,
            policy=best_policy if best_policy is not None else np.ones((n_states, n_actions)) / n_actions,
            log_likelihood=best_ll,
            feature_expectations=best_fe if best_fe is not None else np.zeros(n_features),
        )


class BayesianIRL:
    """
    Bayesian Inverse Reinforcement Learning (Ramachandran & Amir, 2007).

    Compute the posterior over reward weights given demonstrations:
        P(w | D) ∝ P(D | w) * P(w)

    where P(D | w) = product over demonstrations of P(trajectory | w).
    Uses MCMC (Metropolis-Hastings) to sample from the posterior.

    Advantages over MaxEntIRL:
        - Full posterior gives uncertainty estimates
        - Prior encodes domain knowledge
        - Handles multimodal reward functions

    Disadvantages:
        - Slower (MCMC sampling)
        - Requires tuning proposal distribution
    """

    def __init__(
        self,
        n_samples: int = 500,
        burn_in: int = 100,
        proposal_std: float = 0.3,
    ):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.proposal_std = proposal_std

    def _log_likelihood(
        self,
        weights: np.ndarray,
        demos: list[Demonstration],
        features: np.ndarray,
        transition: np.ndarray,
        discount: float,
    ) -> float:
        """
        Log-likelihood of demonstrations under reward weights.

        Uses soft value iteration to get the policy, then evaluates
        P(a_t | s_t) for each demo step.
        """
        reward = features @ weights
        n_states, n_actions, _ = transition.shape

        _, policy = MaxEntIRL.soft_value_iteration(
            reward, transition, discount, n_iter=30
        )

        ll = 0.0
        for demo in demos:
            for t in range(len(demo.states)):
                s = int(demo.states[t])
                a = int(demo.actions[t])
                ll += np.log(policy[s, a] + 1e-10)
        return ll

    def _log_prior(
        self,
        weights: np.ndarray,
        prior_mean: np.ndarray,
        prior_cov_inv: np.ndarray,
    ) -> float:
        """Log of Gaussian prior P(w)."""
        diff = weights - prior_mean
        return -0.5 * diff @ prior_cov_inv @ diff

    def learn(
        self,
        demos: list[Demonstration],
        features: np.ndarray,
        transition: np.ndarray,
        discount: float = 0.99,
        prior_mean: np.ndarray | None = None,
        prior_cov: np.ndarray | None = None,
    ) -> IRLResult:
        """
        Learn reward weights via MCMC posterior sampling.

        Metropolis-Hastings with Gaussian proposal:
            w' ~ N(w, sigma^2 * I)
            accept with probability min(1, P(w'|D) / P(w|D))

        Args:
            demos: Expert demonstrations
            features: State features, shape (n_states, n_features)
            transition: Transition model, shape (n_states, n_actions, n_states)
            discount: Discount factor
            prior_mean: Prior mean for reward weights (default: zeros)
            prior_cov: Prior covariance (default: identity)

        Returns:
            IRLResult with posterior mean weights and MAP policy
        """
        n_states, n_features = features.shape
        n_actions = transition.shape[1]

        if prior_mean is None:
            prior_mean = np.zeros(n_features)
        if prior_cov is None:
            prior_cov = np.eye(n_features)

        prior_cov_inv = np.linalg.inv(prior_cov + 1e-6 * np.eye(n_features))

        # Initialize at prior mean
        current_w = prior_mean.copy()
        current_ll = self._log_likelihood(
            current_w, demos, features, transition, discount
        )
        current_lp = self._log_prior(current_w, prior_mean, prior_cov_inv)
        current_log_post = current_ll + current_lp

        # Collect samples
        samples = []
        log_posts = []
        accepted = 0

        for i in range(self.n_samples + self.burn_in):
            # Propose
            proposed_w = current_w + self.proposal_std * np.random.randn(n_features)

            proposed_ll = self._log_likelihood(
                proposed_w, demos, features, transition, discount
            )
            proposed_lp = self._log_prior(proposed_w, prior_mean, prior_cov_inv)
            proposed_log_post = proposed_ll + proposed_lp

            # Accept/reject
            log_alpha = proposed_log_post - current_log_post
            if np.log(np.random.random() + 1e-10) < log_alpha:
                current_w = proposed_w
                current_log_post = proposed_log_post
                current_ll = proposed_ll
                accepted += 1

            if i >= self.burn_in:
                samples.append(current_w.copy())
                log_posts.append(current_log_post)

        samples = np.array(samples)
        log_posts = np.array(log_posts)

        # Posterior mean
        mean_weights = np.mean(samples, axis=0)

        # MAP estimate
        map_idx = np.argmax(log_posts)
        map_weights = samples[map_idx]

        # Policy under MAP weights
        reward = features @ map_weights
        _, policy = MaxEntIRL.soft_value_iteration(
            reward, transition, discount, n_iter=50
        )

        # Feature expectations under MAP policy
        initial = np.zeros(n_states)
        for demo in demos:
            if len(demo.states) > 0:
                initial[int(demo.states[0])] += 1.0
        initial = initial / (np.sum(initial) + 1e-10)

        fe = MaxEntIRL.feature_expectations(
            policy, features, transition, discount, initial
        )

        return IRLResult(
            reward_weights=mean_weights,
            policy=policy,
            log_likelihood=float(log_posts[map_idx]),
            feature_expectations=fe,
        )


class ApprenticeshipLearning:
    """
    Apprenticeship Learning via Inverse Reinforcement Learning (Abbeel & Ng, 2004).

    Instead of recovering the exact reward, find a policy whose feature
    expectations are close to the expert's. Uses a projection algorithm:

    1. Compute expert feature expectations μ_E
    2. Start with random policy, compute μ_0
    3. Find reward weights w that maximally separate μ_E from closest μ_i
    4. Solve the MDP with reward w, get new policy π_{i+1}
    5. Compute μ_{i+1}, project onto constraint set
    6. Repeat until ||μ_E - μ_bar|| < epsilon

    This avoids solving the IRL problem explicitly — we just need a policy
    that matches the expert's feature expectations. Simpler and with formal
    guarantees on policy quality.
    """

    @staticmethod
    def _solve_mdp(
        reward: np.ndarray,
        transition: np.ndarray,
        discount: float,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Standard value iteration (hard-max, not soft).

        Returns:
            V: Value function, shape (n_states,)
            policy: Deterministic greedy policy (one-hot), shape (n_states, n_actions)
        """
        n_states, n_actions, _ = transition.shape

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_actions))
        else:
            R = reward

        V = np.zeros(n_states)
        for _ in range(n_iter):
            Q = R + discount * np.sum(transition * V[np.newaxis, np.newaxis, :], axis=2)
            V_new = np.max(Q, axis=1)
            if np.max(np.abs(V_new - V)) < 1e-8:
                break
            V = V_new

        Q = R + discount * np.sum(transition * V[np.newaxis, np.newaxis, :], axis=2)
        policy = np.zeros((n_states, n_actions))
        best_actions = np.argmax(Q, axis=1)
        policy[np.arange(n_states), best_actions] = 1.0

        return V, policy

    def learn(
        self,
        demos: list[Demonstration],
        features: np.ndarray,
        transition: np.ndarray,
        discount: float = 0.99,
        n_iter: int = 100,
        epsilon: float = 0.01,
    ) -> IRLResult:
        """
        Learn a policy via feature expectation matching.

        Projection algorithm:
            1. Compute μ_E from demonstrations
            2. Iteratively find policies whose μ approaches μ_E
            3. Stop when ||μ_E - closest μ|| < epsilon

        Args:
            demos: Expert demonstrations
            features: State features, shape (n_states, n_features)
            transition: Transition model, shape (n_states, n_actions, n_states)
            discount: Discount factor
            n_iter: Maximum iterations
            epsilon: Convergence threshold on feature expectation distance

        Returns:
            IRLResult with best matching policy
        """
        n_states, n_features = features.shape
        n_actions = transition.shape[1]

        # Expert feature expectations
        expert_fe = np.zeros(n_features)
        for demo in demos:
            for t, s in enumerate(demo.states):
                if demo.features is not None:
                    expert_fe += (discount ** t) * demo.features[t]
                else:
                    expert_fe += (discount ** t) * features[int(s)]
        expert_fe /= max(len(demos), 1)

        # Initial state distribution
        initial = np.zeros(n_states)
        for demo in demos:
            if len(demo.states) > 0:
                initial[int(demo.states[0])] += 1.0
        initial = initial / (np.sum(initial) + 1e-10)

        # Start with random policy
        policy = np.ones((n_states, n_actions)) / n_actions
        mu = MaxEntIRL.feature_expectations(
            policy, features, transition, discount, initial
        )

        mus = [mu.copy()]
        best_weights = np.zeros(n_features)
        best_policy = policy.copy()
        best_dist = np.inf
        best_fe = mu.copy()

        # Projection-based iteration
        mu_bar = mu.copy()

        for iteration in range(n_iter):
            # Weight vector: direction from mu_bar to expert
            w = expert_fe - mu_bar
            dist = np.linalg.norm(w)

            if dist < epsilon:
                break

            if dist < best_dist:
                best_dist = dist
                best_weights = w / (dist + 1e-10)
                best_policy = policy.copy()
                best_fe = mu.copy()

            # Normalize for reward
            w_norm = w / (dist + 1e-10)

            # Solve MDP with this reward
            reward = features @ w_norm
            _, policy = self._solve_mdp(reward, transition, discount)

            # Compute feature expectations of new policy
            mu = MaxEntIRL.feature_expectations(
                policy, features, transition, discount, initial
            )
            mus.append(mu.copy())

            # Project mu_bar onto line segment between mu_bar and mu
            # to get closest point to expert_fe
            delta = mu - mu_bar
            denom = np.dot(delta, delta)
            if denom > 1e-10:
                t_proj = np.dot(expert_fe - mu_bar, delta) / denom
                t_proj = np.clip(t_proj, 0.0, 1.0)
                mu_bar = mu_bar + t_proj * delta
            else:
                mu_bar = mu.copy()

        # Final check
        final_dist = np.linalg.norm(expert_fe - mu_bar)
        if final_dist < best_dist:
            best_weights = (expert_fe - mu_bar)
            norm = np.linalg.norm(best_weights)
            if norm > 1e-10:
                best_weights /= norm
            best_policy = policy
            best_fe = mu

        # Log-likelihood approximation
        reward = features @ best_weights
        ll = 0.0
        for demo in demos:
            for t, s in enumerate(demo.states):
                ll += (discount ** t) * reward[int(s)]
        ll /= max(len(demos), 1)

        return IRLResult(
            reward_weights=best_weights,
            policy=best_policy,
            log_likelihood=ll,
            feature_expectations=best_fe,
        )
