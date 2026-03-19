"""
Policy Gradient Methods — REINFORCE, Actor-Critic, TRPO, SAC
==============================================================

Direct policy optimization methods that learn parameterized policies
via gradient ascent on expected return. Foundation for modern deep RL.

Core Concepts:
- REINFORCE: Vanilla policy gradient with baseline subtraction
- Actor-Critic: Separate policy (actor) and value (critic) networks
- TRPO: Constrained policy update via KL divergence trust region
- SAC: Maximum entropy RL with automatic temperature tuning

Mathematical Foundation:
Policy gradient: ∇J(θ) = E[Σ_t ∇log π_θ(a_t|s_t) · (R_t - b_t)]
Advantage: A(s,a) = Q(s,a) - V(s) reduces variance
TRPO: max_θ E[π_θ/π_old · A]  s.t.  D_KL(π_old || π_θ) ≤ δ
SAC: J(π) = E[Σ r_t + α H(π(·|s_t))]  (entropy-augmented return)

Applications to Warp Space:
- Policy gradient for continuous action spaces in embedding optimization
- Actor-critic for real-time scoring term weight adjustment
- Trust regions for safe policy updates in production weaving
- Maximum entropy for exploration-exploitation in memory retrieval

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PolicyGradientResult:
    """Result of policy gradient training.

    Attributes:
        policy_params: Final policy parameters.
        episode_rewards: Total reward per episode.
        episode_lengths: Steps per episode.
        loss_history: Policy loss per update.
        converged: Whether training converged.
    """
    policy_params: np.ndarray
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    loss_history: np.ndarray
    converged: bool = False


@dataclass
class ActorCriticResult:
    """Result of actor-critic training.

    Attributes:
        actor_params: Final actor (policy) parameters.
        critic_params: Final critic (value) parameters.
        episode_rewards: Total reward per episode.
        actor_loss: Actor loss history.
        critic_loss: Critic loss history.
    """
    actor_params: np.ndarray
    critic_params: np.ndarray
    episode_rewards: np.ndarray
    actor_loss: np.ndarray
    critic_loss: np.ndarray


# ============================================================================
# SIMPLE ENVIRONMENTS FOR TESTING
# ============================================================================

class TabularEnv:
    """Simple tabular environment for policy gradient testing."""

    def __init__(self, n_states: int = 10, n_actions: int = 4, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        # Transition: (s, a) -> s' (deterministic for simplicity)
        self.transitions = rng.randint(0, n_states, size=(n_states, n_actions))
        # Reward: (s, a) -> r
        self.rewards = rng.randn(n_states, n_actions) * 0.5
        self.rewards[n_states - 1, :] = 1.0  # Goal state
        self.state = 0

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        r = self.rewards[self.state, action]
        self.state = self.transitions[self.state, action]
        done = (self.state == self.n_states - 1)
        return self.state, r, done


# ============================================================================
# SOFTMAX POLICY
# ============================================================================

class SoftmaxPolicy:
    """Tabular softmax policy π(a|s) = exp(θ_{s,a}) / Σ exp(θ_{s,a'})."""

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.theta = np.zeros((n_states, n_actions))

    def probabilities(self, state: int) -> np.ndarray:
        """Action probabilities at state."""
        logits = self.theta[state] - np.max(self.theta[state])
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def sample(self, state: int) -> int:
        """Sample action from policy."""
        probs = self.probabilities(state)
        return int(np.random.choice(self.n_actions, p=probs))

    def log_prob(self, state: int, action: int) -> float:
        """Log probability of action at state."""
        probs = self.probabilities(state)
        return float(np.log(probs[action] + 1e-10))

    def log_prob_gradient(self, state: int, action: int) -> np.ndarray:
        """Gradient of log π(a|s) w.r.t. theta.

        ∇_θ log π(a|s) = e_a - π(·|s)  (score function)
        """
        probs = self.probabilities(state)
        grad = np.zeros_like(self.theta)
        grad[state] = -probs
        grad[state, action] += 1.0
        return grad

    def entropy(self, state: int) -> float:
        """Policy entropy H(π(·|s)) = -Σ π(a|s) log π(a|s)."""
        probs = self.probabilities(state)
        return float(-np.sum(probs * np.log(probs + 1e-10)))


# ============================================================================
# REINFORCE
# ============================================================================

class REINFORCE:
    """Vanilla REINFORCE (Monte Carlo policy gradient) with baseline.

    ∇J(θ) = E[Σ_t ∇log π_θ(a_t|s_t) · (G_t - b_t)]

    where G_t = Σ_{k≥t} γ^{k-t} r_k is the return-to-go and
    b_t is a baseline (running average of returns by default).

    Example:
        >>> reinforce = REINFORCE()
        >>> env = TabularEnv(n_states=10, n_actions=4)
        >>> result = reinforce.learn(env, lr=0.01, n_episodes=500)
    """

    def learn(
        self,
        env: Any,
        lr: float = 0.01,
        n_episodes: int = 500,
        gamma: float = 0.99,
        max_steps: int = 200,
        baseline: bool = True,
    ) -> PolicyGradientResult:
        """Train policy via REINFORCE.

        Args:
            env: Environment with reset() -> state, step(action) -> (state, reward, done).
            lr: Learning rate.
            n_episodes: Number of training episodes.
            gamma: Discount factor.
            max_steps: Maximum steps per episode.
            baseline: Whether to subtract running return baseline.

        Returns:
            PolicyGradientResult.
        """
        n_states = env.n_states
        n_actions = env.n_actions
        policy = SoftmaxPolicy(n_states, n_actions)

        episode_rewards = []
        episode_lengths = []
        loss_history = []
        running_return = 0.0

        for ep in range(n_episodes):
            # Collect trajectory
            states, actions, rewards = [], [], []
            state = env.reset()

            for t in range(max_steps):
                action = policy.sample(state)
                next_state, reward, done = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                if done:
                    break

            T = len(rewards)
            episode_rewards.append(sum(rewards))
            episode_lengths.append(T)

            # Compute returns-to-go
            returns = np.zeros(T)
            G = 0.0
            for t in range(T - 1, -1, -1):
                G = rewards[t] + gamma * G
                returns[t] = G

            # Update running baseline
            ep_return = returns[0] if T > 0 else 0.0
            running_return = 0.9 * running_return + 0.1 * ep_return

            # Policy gradient update
            total_loss = 0.0
            grad = np.zeros_like(policy.theta)

            for t in range(T):
                advantage = returns[t] - (running_return if baseline else 0.0)
                grad += policy.log_prob_gradient(states[t], actions[t]) * advantage
                total_loss -= policy.log_prob(states[t], actions[t]) * advantage

            policy.theta += lr * grad / max(T, 1)
            loss_history.append(total_loss / max(T, 1))

        return PolicyGradientResult(
            policy_params=policy.theta,
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.array(episode_lengths),
            loss_history=np.array(loss_history),
            converged=len(episode_rewards) > 50 and np.mean(episode_rewards[-50:]) > np.mean(episode_rewards[:50]),
        )


# ============================================================================
# ACTOR-CRITIC (A2C)
# ============================================================================

class ActorCritic:
    """Advantage Actor-Critic (A2C).

    Actor: Policy π_θ(a|s), updated with advantage estimates
    Critic: Value function V_φ(s), updated with TD errors

    Advantage: A(s,a) ≈ r + γV(s') - V(s) (one-step TD)
    Actor loss: -log π(a|s) · A(s,a)
    Critic loss: (r + γV(s') - V(s))²

    Example:
        >>> ac = ActorCritic()
        >>> env = TabularEnv(n_states=10, n_actions=4)
        >>> result = ac.learn(env, lr_actor=0.01, lr_critic=0.05, n_episodes=500)
    """

    def learn(
        self,
        env: Any,
        lr_actor: float = 0.01,
        lr_critic: float = 0.05,
        n_episodes: int = 500,
        gamma: float = 0.99,
        max_steps: int = 200,
        entropy_coeff: float = 0.01,
    ) -> ActorCriticResult:
        """Train actor-critic.

        Args:
            env: Environment.
            lr_actor: Actor learning rate.
            lr_critic: Critic learning rate.
            n_episodes: Training episodes.
            gamma: Discount factor.
            max_steps: Max steps per episode.
            entropy_coeff: Entropy bonus coefficient.

        Returns:
            ActorCriticResult.
        """
        n_states = env.n_states
        n_actions = env.n_actions
        policy = SoftmaxPolicy(n_states, n_actions)
        V = np.zeros(n_states)

        episode_rewards = []
        actor_losses = []
        critic_losses = []

        for ep in range(n_episodes):
            state = env.reset()
            ep_reward = 0.0
            ep_actor_loss = 0.0
            ep_critic_loss = 0.0
            steps = 0

            for t in range(max_steps):
                action = policy.sample(state)
                next_state, reward, done = env.step(action)
                ep_reward += reward
                steps += 1

                # TD error (advantage estimate)
                td_target = reward + (gamma * V[next_state] if not done else 0.0)
                td_error = td_target - V[state]

                # Critic update
                V[state] += lr_critic * td_error
                ep_critic_loss += td_error ** 2

                # Actor update
                advantage = td_error
                grad = policy.log_prob_gradient(state, action) * advantage
                # Entropy bonus for exploration
                H = policy.entropy(state)
                policy.theta += lr_actor * (grad.ravel().reshape(policy.theta.shape))
                policy.theta[state] += lr_actor * entropy_coeff * (1.0 / (n_actions) - policy.probabilities(state))
                ep_actor_loss -= policy.log_prob(state, action) * advantage

                state = next_state
                if done:
                    break

            episode_rewards.append(ep_reward)
            actor_losses.append(ep_actor_loss / max(steps, 1))
            critic_losses.append(ep_critic_loss / max(steps, 1))

        return ActorCriticResult(
            actor_params=policy.theta,
            critic_params=V,
            episode_rewards=np.array(episode_rewards),
            actor_loss=np.array(actor_losses),
            critic_loss=np.array(critic_losses),
        )


# ============================================================================
# TRPO (TRUST REGION POLICY OPTIMIZATION)
# ============================================================================

class TRPO:
    """Trust Region Policy Optimization.

    Constrains policy updates to a KL divergence trust region:
    max_θ E[π_θ(a|s)/π_old(a|s) · A(s,a)]
    s.t. E[D_KL(π_old(·|s) || π_θ(·|s))] ≤ δ

    Solved approximately via conjugate gradient for the natural
    gradient direction, then backtracking line search.

    Example:
        >>> trpo = TRPO()
        >>> policy = SoftmaxPolicy(10, 4)
        >>> new_params = trpo.step(policy, states, actions, advantages, delta=0.01)
    """

    def step(
        self,
        policy: SoftmaxPolicy,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        delta: float = 0.01,
        cg_iters: int = 10,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.5,
    ) -> np.ndarray:
        """Execute one TRPO update step.

        Args:
            policy: Current softmax policy.
            states: Batch of states (B,).
            actions: Batch of actions (B,).
            advantages: Advantage estimates (B,).
            delta: KL divergence constraint.
            cg_iters: Conjugate gradient iterations.
            backtrack_iters: Line search iterations.
            backtrack_coeff: Line search coefficient.

        Returns:
            Updated policy parameters.
        """
        states = np.asarray(states, dtype=int)
        actions = np.asarray(actions, dtype=int)
        advantages = np.asarray(advantages, dtype=float)
        B = len(states)

        # Normalize advantages
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Compute policy gradient
        flat_size = policy.theta.size
        g = np.zeros(flat_size)
        for i in range(B):
            grad_i = policy.log_prob_gradient(states[i], actions[i]).ravel()
            g += grad_i * advantages[i]
        g /= B

        # Compute Fisher information matrix (average outer product of score)
        F = np.zeros((flat_size, flat_size))
        for i in range(B):
            score = policy.log_prob_gradient(states[i], actions[i]).ravel()
            F += np.outer(score, score)
        F /= B
        F += 1e-5 * np.eye(flat_size)  # Damping

        # Natural gradient via conjugate gradient: solve Fx = g
        x = self._conjugate_gradient(F, g, cg_iters)

        # Step size from KL constraint: α = sqrt(2δ / x^T F x)
        xFx = x @ F @ x
        if xFx > 0:
            step_size = np.sqrt(2.0 * delta / xFx)
        else:
            step_size = 0.01

        # Backtracking line search
        old_theta = policy.theta.copy()
        old_surrogate = self._surrogate_loss(policy, states, actions, advantages, old_theta)

        for i in range(backtrack_iters):
            new_theta = old_theta + (step_size * backtrack_coeff ** i) * x.reshape(policy.theta.shape)
            policy.theta = new_theta

            new_surrogate = self._surrogate_loss(policy, states, actions, advantages, old_theta)
            kl = self._average_kl(policy, states, old_theta)

            if new_surrogate > old_surrogate and kl <= delta:
                return policy.theta

        # Revert if no improvement found
        policy.theta = old_theta
        return policy.theta

    @staticmethod
    def _conjugate_gradient(A: np.ndarray, b: np.ndarray, n_iters: int) -> np.ndarray:
        """Solve Ax = b via conjugate gradient."""
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rs_old = r @ r

        for _ in range(n_iters):
            Ap = A @ p
            alpha = rs_old / (p @ Ap + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            rs_new = r @ r
            if np.sqrt(rs_new) < 1e-10:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    @staticmethod
    def _surrogate_loss(policy, states, actions, advantages, old_theta):
        """Compute surrogate objective (importance-weighted advantage)."""
        total = 0.0
        for i in range(len(states)):
            new_lp = policy.log_prob(states[i], actions[i])
            old_probs = np.exp(old_theta[states[i]] - np.max(old_theta[states[i]]))
            old_probs /= np.sum(old_probs)
            old_lp = np.log(old_probs[actions[i]] + 1e-10)
            ratio = np.exp(new_lp - old_lp)
            total += ratio * advantages[i]
        return total / len(states)

    @staticmethod
    def _average_kl(policy, states, old_theta):
        """Average KL divergence between old and new policy."""
        total_kl = 0.0
        for s in states:
            new_probs = policy.probabilities(s)
            old_logits = old_theta[s] - np.max(old_theta[s])
            old_probs = np.exp(old_logits) / np.sum(np.exp(old_logits))
            kl = np.sum(old_probs * np.log((old_probs + 1e-10) / (new_probs + 1e-10)))
            total_kl += kl
        return total_kl / len(states)


# ============================================================================
# SAC (SOFT ACTOR-CRITIC)
# ============================================================================

class SAC:
    """Soft Actor-Critic (tabular approximation).

    Maximum entropy RL: maximize E[Σ r_t + α H(π(·|s_t))]

    The entropy bonus encourages exploration and leads to more
    robust policies. α (temperature) can be tuned automatically.

    Soft Bellman: V(s) = E[Q(s,a) - α log π(a|s)]
    Soft Q: Q(s,a) = r + γ V(s')

    Example:
        >>> sac = SAC()
        >>> env = TabularEnv(n_states=10, n_actions=4)
        >>> result = sac.learn(env, lr=0.05, n_episodes=500, alpha=0.2)
    """

    def learn(
        self,
        env: Any,
        lr: float = 0.05,
        n_episodes: int = 500,
        gamma: float = 0.99,
        alpha: float = 0.2,
        auto_alpha: bool = False,
        target_entropy: float | None = None,
        max_steps: int = 200,
        tau: float = 0.005,
    ) -> PolicyGradientResult:
        """Train via Soft Actor-Critic.

        Args:
            env: Environment.
            lr: Learning rate.
            n_episodes: Training episodes.
            gamma: Discount factor.
            alpha: Entropy temperature.
            auto_alpha: Whether to automatically tune alpha.
            target_entropy: Target entropy for auto-tuning (default: -n_actions).
            max_steps: Max steps per episode.
            tau: Soft target update rate.

        Returns:
            PolicyGradientResult.
        """
        n_s = env.n_states
        n_a = env.n_actions

        # Two Q-functions (clipped double-Q)
        Q1 = np.zeros((n_s, n_a))
        Q2 = np.zeros((n_s, n_a))
        Q1_target = Q1.copy()
        Q2_target = Q2.copy()

        policy = SoftmaxPolicy(n_s, n_a)

        if target_entropy is None:
            target_entropy = -float(n_a) * 0.5
        log_alpha = np.log(alpha)

        episode_rewards = []
        loss_history = []

        for ep in range(n_episodes):
            state = env.reset()
            ep_reward = 0.0
            ep_loss = 0.0

            for t in range(max_steps):
                action = policy.sample(state)
                next_state, reward, done = env.step(action)
                ep_reward += reward

                # Compute target
                next_probs = policy.probabilities(next_state)
                next_log_probs = np.log(next_probs + 1e-10)
                next_q = np.minimum(Q1_target[next_state], Q2_target[next_state])
                V_next = np.sum(next_probs * (next_q - alpha * next_log_probs))

                target = reward + (gamma * V_next if not done else 0.0)

                # Update Q-functions
                Q1[state, action] += lr * (target - Q1[state, action])
                Q2[state, action] += lr * (target - Q2[state, action])
                ep_loss += (target - Q1[state, action]) ** 2

                # Update policy (maximize E[Q - α log π])
                probs = policy.probabilities(state)
                q_min = np.minimum(Q1[state], Q2[state])
                # Target distribution ∝ exp(Q/α)
                target_logits = q_min / (alpha + 1e-8)
                target_logits -= np.max(target_logits)
                target_probs = np.exp(target_logits)
                target_probs /= np.sum(target_probs)

                # Move policy toward target
                policy.theta[state] += lr * (target_logits - policy.theta[state])

                # Auto-tune alpha
                if auto_alpha:
                    current_entropy = -np.sum(probs * np.log(probs + 1e-10))
                    log_alpha += lr * (current_entropy - target_entropy)
                    alpha = np.exp(log_alpha)
                    alpha = np.clip(alpha, 0.01, 10.0)

                # Soft target update
                Q1_target = tau * Q1 + (1 - tau) * Q1_target
                Q2_target = tau * Q2 + (1 - tau) * Q2_target

                state = next_state
                if done:
                    break

            episode_rewards.append(ep_reward)
            loss_history.append(ep_loss)

        return PolicyGradientResult(
            policy_params=policy.theta,
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.zeros(n_episodes),
            loss_history=np.array(loss_history),
            converged=len(episode_rewards) > 50 and np.mean(episode_rewards[-50:]) > np.mean(episode_rewards[:50]),
        )
