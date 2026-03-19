"""
Multi-Agent Reinforcement Learning — Coordination Between Multiple Evaluators
=============================================================================

Algorithms for learning in multi-agent environments where multiple decision-makers
interact simultaneously. Each agent's optimal policy depends on others' policies,
creating a coupled optimization landscape.

Classes:
    IndependentLearners: Each agent Q-learns independently (no coordination)
    CentralizedCritic: Shared value function, individual policies (CTDE)
    MeanFieldMARL: Mean-field approximation reduces N-agent to 2-agent problem
    CommProtocol: Learned message passing between agents

Key Concepts:
    - Nash equilibrium: no agent can unilaterally improve
    - Social welfare: sum of all agents' returns
    - Nash gap: how far the joint policy is from Nash equilibrium
    - Mean field: approximate effect of N-1 agents as a single distribution

Applications:
    - Multi-evaluator scoring (SOUS GPS with competing terms)
    - Distributed resource allocation
    - Cooperative planning with partial observability
    - Consensus formation in multi-agent systems

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MARLAgent:
    """
    A single agent in a multi-agent system.

    Attributes:
        id: Unique agent identifier
        n_actions: Size of the agent's action space
        policy: Probability distribution over actions (n_actions,)
        q_values: State-action values; shape depends on algorithm
        reward_history: Accumulated rewards over time
    """
    id: str
    n_actions: int
    policy: np.ndarray = field(default=None)
    q_values: np.ndarray = field(default=None)
    reward_history: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.policy is None:
            self.policy = np.ones(self.n_actions) / self.n_actions
        if self.q_values is None:
            self.q_values = np.zeros(self.n_actions)


@dataclass
class MARLResult:
    """
    Result of multi-agent learning.

    Attributes:
        joint_policy: Mapping from agent id to policy distribution
        social_welfare: Sum of all agents' expected returns
        nash_gap: Distance from Nash equilibrium (0 = perfect Nash)
        iterations: Number of learning iterations completed
    """
    joint_policy: dict[str, np.ndarray]
    social_welfare: float
    nash_gap: float
    iterations: int


class IndependentLearners:
    """
    Independent Q-learning — each agent learns as if alone.

    Each agent maintains its own Q-table and updates via standard Q-learning,
    treating other agents as part of the environment. Simple but can oscillate
    in competitive settings because the environment is nonstationary from
    each agent's perspective.

    Convergence: guaranteed in fully cooperative games with decaying lr.
    No guarantees in general-sum games — but often works in practice.
    """

    def __init__(
        self,
        agents: list[MARLAgent],
        lr: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.agents = {a.id: a for a in agents}
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self._iteration = 0

    def step(
        self,
        observations: dict[str, int],
        rewards: dict[str, float],
    ) -> dict[str, int]:
        """
        Each agent Q-learns independently and selects next action.

        Args:
            observations: Current state index per agent
            rewards: Reward received per agent from last action

        Returns:
            Actions selected by each agent (agent_id -> action_index)
        """
        actions = {}
        for aid, agent in self.agents.items():
            r = rewards.get(aid, 0.0)
            agent.reward_history.append(r)
            obs = observations.get(aid, 0)

            # Q-learning update: Q(s,a) += lr * (r + gamma * max Q(s') - Q(s,a))
            # In the bandit/stateless case, obs indexes a "state"
            # but q_values is flat (n_actions,). Extend if needed.
            if agent.q_values.ndim == 1:
                best_next = np.max(agent.q_values)
                # Find last action (use argmax of policy as proxy if no history)
                last_action = np.argmax(agent.policy)
                td_error = r + self.discount * best_next - agent.q_values[last_action]
                agent.q_values[last_action] += self.lr * td_error

            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                action = np.random.randint(agent.n_actions)
            else:
                action = int(np.argmax(agent.q_values))

            # Update policy to reflect greedy + epsilon
            agent.policy = np.full(agent.n_actions, self.epsilon / agent.n_actions)
            agent.policy[np.argmax(agent.q_values)] += 1.0 - self.epsilon
            actions[aid] = action

        self._iteration += 1
        return actions

    def step_with_states(
        self,
        states: dict[str, int],
        actions_taken: dict[str, int],
        rewards: dict[str, float],
        next_states: dict[str, int],
        n_states: int,
    ) -> dict[str, int]:
        """
        Full Q-learning with state transitions.

        Extends Q-tables to (n_states, n_actions) on first call.
        Uses the actual (s, a, r, s') tuple for updates.

        Args:
            states: Previous state per agent
            actions_taken: Actions that were actually taken
            rewards: Rewards received
            next_states: Resulting states
            n_states: Total number of states (for Q-table sizing)

        Returns:
            Next actions to take
        """
        actions = {}
        for aid, agent in self.agents.items():
            # Extend Q-table if needed
            if agent.q_values.ndim == 1:
                old_q = agent.q_values.copy()
                agent.q_values = np.zeros((n_states, agent.n_actions))
                agent.q_values[0, :] = old_q

            s = states.get(aid, 0)
            a = actions_taken.get(aid, 0)
            r = rewards.get(aid, 0.0)
            s_next = next_states.get(aid, 0)

            agent.reward_history.append(r)

            # Q-learning update
            best_next = np.max(agent.q_values[s_next])
            td_error = r + self.discount * best_next - agent.q_values[s, a]
            agent.q_values[s, a] += self.lr * td_error

            # Epsilon-greedy from next state
            if np.random.random() < self.epsilon:
                action = np.random.randint(agent.n_actions)
            else:
                action = int(np.argmax(agent.q_values[s_next]))

            actions[aid] = action

        self._iteration += 1
        return actions

    def get_joint_policy(self) -> MARLResult:
        """
        Extract the current joint policy and compute quality metrics.

        Social welfare = sum of mean rewards over last 100 steps.
        Nash gap = max improvement any single agent could achieve by deviating.
        """
        joint = {}
        total_welfare = 0.0

        for aid, agent in self.agents.items():
            # Derive policy from Q-values
            if agent.q_values.ndim == 1:
                q = agent.q_values
            else:
                # Average over states
                q = np.mean(agent.q_values, axis=0)

            # Softmax policy
            q_shifted = q - np.max(q)
            exp_q = np.exp(q_shifted)
            policy = exp_q / (np.sum(exp_q) + 1e-10)
            joint[aid] = policy
            agent.policy = policy

            # Welfare from recent rewards
            recent = agent.reward_history[-100:] if agent.reward_history else [0.0]
            total_welfare += np.mean(recent)

        # Nash gap: max potential gain from unilateral deviation
        nash_gap = self._compute_nash_gap(joint)

        return MARLResult(
            joint_policy=joint,
            social_welfare=total_welfare,
            nash_gap=nash_gap,
            iterations=self._iteration,
        )

    def _compute_nash_gap(self, joint: dict[str, np.ndarray]) -> float:
        """
        Estimate Nash gap from Q-values.

        For each agent: gap_i = max_a Q(a) - sum_a pi(a)*Q(a)
        Nash gap = max over agents.
        """
        max_gap = 0.0
        for aid, agent in self.agents.items():
            if agent.q_values.ndim == 1:
                q = agent.q_values
            else:
                q = np.mean(agent.q_values, axis=0)
            policy = joint[aid]
            expected = np.dot(policy, q)
            best = np.max(q)
            gap = best - expected
            max_gap = max(max_gap, gap)
        return max_gap


class CentralizedCritic:
    """
    Centralized Training, Decentralized Execution (CTDE).

    A single critic learns a joint value function V(s, a_1, ..., a_n) using
    all agents' observations and actions. Individual actors learn policies
    using the centralized critic's gradients. At execution time, each actor
    only needs its own observation.

    This resolves the nonstationarity issue of independent learners: the
    critic sees the full joint action, so the value function target is stable.

    The critic uses a joint Q-table of shape (n_joint_actions,) where
    joint actions are indexed by the Cartesian product of individual actions.
    """

    def __init__(
        self,
        agents: list[MARLAgent],
        lr: float = 0.01,
    ):
        self.agents = {a.id: a for a in agents}
        self.agent_ids = sorted(self.agents.keys())
        self.lr = lr
        self._iteration = 0

        # Joint action space size
        self._action_sizes = [self.agents[aid].n_actions for aid in self.agent_ids]
        self._n_joint = int(np.prod(self._action_sizes))

        # Centralized Q-values over joint actions
        self.joint_q = np.zeros(self._n_joint)

        # Per-agent policy parameters (logits)
        self._logits = {
            aid: np.zeros(agent.n_actions)
            for aid, agent in self.agents.items()
        }

    def _joint_index(self, actions: dict[str, int]) -> int:
        """Map individual actions to a joint action index."""
        idx = 0
        multiplier = 1
        for aid in reversed(self.agent_ids):
            idx += actions.get(aid, 0) * multiplier
            multiplier *= self.agents[aid].n_actions
        return idx

    def _individual_from_joint(self, joint_idx: int) -> dict[str, int]:
        """Map a joint action index back to individual actions."""
        actions = {}
        remainder = joint_idx
        for aid in reversed(self.agent_ids):
            n = self.agents[aid].n_actions
            actions[aid] = remainder % n
            remainder //= n
        return actions

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        return exp_l / (np.sum(exp_l) + 1e-10)

    def step(
        self,
        joint_obs: dict[str, int],
        rewards: dict[str, float],
    ) -> dict[str, int]:
        """
        Update centralized critic, then each agent selects independently.

        Args:
            joint_obs: State observation per agent (used for logging, future extension)
            rewards: Reward per agent (critic uses the team reward = sum)

        Returns:
            Actions selected by each agent using their individual policies
        """
        # Team reward for the critic
        team_reward = sum(rewards.values())

        # Update the critic's Q-value for the last joint action
        if self._iteration > 0:
            best_next = np.max(self.joint_q)
            last_joint = self._last_joint_idx
            td_error = team_reward + 0.99 * best_next - self.joint_q[last_joint]
            self.joint_q[last_joint] += self.lr * td_error

        # Update per-agent policies using critic gradient
        # Policy gradient: d/d_logit = advantage * (1(a=a_i) - pi(a_i))
        for aid in self.agent_ids:
            agent = self.agents[aid]
            policy = self._softmax(self._logits[aid])
            agent.reward_history.append(rewards.get(aid, 0.0))

            # Compute per-action advantage from joint Q
            advantages = np.zeros(agent.n_actions)
            for a_i in range(agent.n_actions):
                # Marginalize joint Q over other agents' policies
                q_sum = 0.0
                count = 0
                for j_idx in range(self._n_joint):
                    individual = self._individual_from_joint(j_idx)
                    if individual[aid] == a_i:
                        # Weight by other agents' policies
                        weight = 1.0
                        for other_id in self.agent_ids:
                            if other_id != aid:
                                other_policy = self._softmax(self._logits[other_id])
                                weight *= other_policy[individual[other_id]]
                        q_sum += self.joint_q[j_idx] * weight
                        count += 1
                advantages[a_i] = q_sum

            # Baseline subtraction
            baseline = np.dot(policy, advantages)
            advantages -= baseline

            # Policy gradient update on logits
            grad = advantages - policy * np.sum(advantages)
            self._logits[aid] += self.lr * grad

            agent.policy = self._softmax(self._logits[aid])

        # Each agent samples from its updated policy
        actions = {}
        for aid in self.agent_ids:
            policy = self._softmax(self._logits[aid])
            action = np.random.choice(self.agents[aid].n_actions, p=policy)
            actions[aid] = int(action)

        self._last_joint_idx = self._joint_index(actions)
        self._iteration += 1
        return actions

    def get_joint_policy(self) -> MARLResult:
        """Extract the joint policy with quality metrics."""
        joint = {aid: self._softmax(self._logits[aid]) for aid in self.agent_ids}

        total_welfare = 0.0
        for aid, agent in self.agents.items():
            recent = agent.reward_history[-100:] if agent.reward_history else [0.0]
            total_welfare += np.mean(recent)

        # Nash gap from the joint Q-table
        nash_gap = self._compute_nash_gap(joint)

        return MARLResult(
            joint_policy=joint,
            social_welfare=total_welfare,
            nash_gap=nash_gap,
            iterations=self._iteration,
        )

    def _compute_nash_gap(self, joint: dict[str, np.ndarray]) -> float:
        """
        Nash gap: max gain any agent gets from unilateral deviation.

        For each agent, compare expected value under current policy vs.
        best response given other agents' policies are fixed.
        """
        max_gap = 0.0
        for aid in self.agent_ids:
            policy = joint[aid]
            agent = self.agents[aid]

            # Expected value under current policy
            action_values = np.zeros(agent.n_actions)
            for a_i in range(agent.n_actions):
                q_sum = 0.0
                for j_idx in range(self._n_joint):
                    individual = self._individual_from_joint(j_idx)
                    if individual[aid] == a_i:
                        weight = 1.0
                        for other_id in self.agent_ids:
                            if other_id != aid:
                                weight *= joint[other_id][individual[other_id]]
                        q_sum += self.joint_q[j_idx] * weight
                action_values[a_i] = q_sum

            expected = np.dot(policy, action_values)
            best = np.max(action_values)
            max_gap = max(max_gap, best - expected)

        return max_gap


class MeanFieldMARL:
    """
    Mean-Field Multi-Agent RL.

    Instead of modeling the effect of each individual agent, approximate the
    aggregate effect of all other agents as a single mean action distribution.
    This reduces the N-agent problem to a 2-agent problem (self vs. mean field).

    Each agent i replaces the joint action a_{-i} with the mean action:
        ā = (1/(N-1)) * sum_{j != i} a_j

    Then Q_i(s, a_i, ā) approximates Q_i(s, a_1, ..., a_N).

    Complexity: O(N * |A|) per step instead of O(|A|^N).
    Convergence: to mean-field Nash equilibrium in large populations.
    """

    def __init__(
        self,
        agents: list[MARLAgent],
        lr: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.agents = {a.id: a for a in agents}
        self.agent_ids = sorted(self.agents.keys())
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self._iteration = 0

        # Per-agent Q-values: Q(a_i, mean_field_bin)
        # Discretize mean field into n_bins
        self._n_bins = 10
        self._q_tables: dict[str, np.ndarray] = {}
        for aid, agent in self.agents.items():
            self._q_tables[aid] = np.zeros((agent.n_actions, self._n_bins))

        self._last_actions: dict[str, int] = {}

    def _mean_field_bin(self, actions: dict[str, int], exclude_id: str) -> int:
        """
        Compute mean action index and discretize to a bin.

        The mean field is the average action index of all other agents,
        normalized to [0, 1] and binned.
        """
        other_actions = [
            a for aid, a in actions.items() if aid != exclude_id
        ]
        if not other_actions:
            return 0
        max_action = max(
            self.agents[aid].n_actions for aid in self.agents if aid != exclude_id
        )
        mean_action = np.mean(other_actions) / max(max_action - 1, 1)
        bin_idx = int(np.clip(mean_action * (self._n_bins - 1), 0, self._n_bins - 1))
        return bin_idx

    def step(
        self,
        observations: dict[str, int],
        rewards: dict[str, float],
    ) -> dict[str, int]:
        """
        Mean-field Q-learning step.

        Each agent updates Q(a_i, ā) and selects actions epsilon-greedily.

        Args:
            observations: Current state per agent (used for logging)
            rewards: Reward per agent

        Returns:
            Actions selected by each agent
        """
        # Update Q-values using last actions and mean field
        if self._last_actions:
            for aid, agent in self.agents.items():
                r = rewards.get(aid, 0.0)
                agent.reward_history.append(r)
                a = self._last_actions.get(aid, 0)
                mf_bin = self._mean_field_bin(self._last_actions, aid)

                # Q-learning update with mean field
                best_next = np.max(self._q_tables[aid][:, mf_bin])
                td_error = r + self.discount * best_next - self._q_tables[aid][a, mf_bin]
                self._q_tables[aid][a, mf_bin] += self.lr * td_error

        # Select actions
        actions = {}
        for aid, agent in self.agents.items():
            if np.random.random() < self.epsilon:
                action = np.random.randint(agent.n_actions)
            else:
                # Use mean field from last round as estimate
                if self._last_actions:
                    mf_bin = self._mean_field_bin(self._last_actions, aid)
                else:
                    mf_bin = self._n_bins // 2  # Assume uniform initially
                action = int(np.argmax(self._q_tables[aid][:, mf_bin]))

            actions[aid] = action

            # Update policy
            q_avg = np.mean(self._q_tables[aid], axis=1)
            q_shifted = q_avg - np.max(q_avg)
            exp_q = np.exp(q_shifted)
            agent.policy = exp_q / (np.sum(exp_q) + 1e-10)

        self._last_actions = actions.copy()
        self._iteration += 1
        return actions

    def get_joint_policy(self) -> MARLResult:
        """Extract joint policy with social welfare and Nash gap."""
        joint = {}
        total_welfare = 0.0

        for aid, agent in self.agents.items():
            q_avg = np.mean(self._q_tables[aid], axis=1)
            q_shifted = q_avg - np.max(q_avg)
            exp_q = np.exp(q_shifted)
            policy = exp_q / (np.sum(exp_q) + 1e-10)
            joint[aid] = policy

            recent = agent.reward_history[-100:] if agent.reward_history else [0.0]
            total_welfare += np.mean(recent)

        # Nash gap from mean-field Q-values
        max_gap = 0.0
        for aid, agent in self.agents.items():
            q_avg = np.mean(self._q_tables[aid], axis=1)
            expected = np.dot(joint[aid], q_avg)
            best = np.max(q_avg)
            max_gap = max(max_gap, best - expected)

        return MARLResult(
            joint_policy=joint,
            social_welfare=total_welfare,
            nash_gap=max_gap,
            iterations=self._iteration,
        )


class CommProtocol:
    """
    Learned Communication Protocol for Multi-Agent Coordination.

    Agents exchange fixed-size message vectors before selecting actions.
    Messages are learned end-to-end: each agent has an encoder (observation -> message)
    and a decoder (received_messages -> action_modifier).

    Communication rounds are sequential: after each round, agents update their
    messages based on received messages. Multiple rounds allow iterative consensus.

    Architecture:
        1. Encode: observation -> message (linear projection)
        2. Aggregate: average received messages from all other agents
        3. Decode: (observation, aggregated_message) -> action logits
        4. Repeat for n_comm_rounds
    """

    def __init__(
        self,
        agents: list[MARLAgent],
        message_size: int = 8,
        lr: float = 0.01,
        epsilon: float = 0.1,
    ):
        self.agents = {a.id: a for a in agents}
        self.agent_ids = sorted(self.agents.keys())
        self.message_size = message_size
        self.lr = lr
        self.epsilon = epsilon
        self._iteration = 0

        # Per-agent communication parameters
        # Encoder: observation (one-hot of n_actions as proxy) -> message
        # Decoder: (n_actions + message_size) -> n_actions
        self._encoders: dict[str, np.ndarray] = {}
        self._decoders: dict[str, np.ndarray] = {}
        self._obs_size: dict[str, int] = {}

        for aid, agent in self.agents.items():
            obs_size = agent.n_actions  # One-hot encoding of last action
            self._obs_size[aid] = obs_size
            self._encoders[aid] = np.random.randn(obs_size, message_size) * 0.1
            self._decoders[aid] = np.random.randn(
                obs_size + message_size, agent.n_actions
            ) * 0.1

        self._last_actions: dict[str, int] = {}
        self._last_messages: dict[str, np.ndarray] = {}

    def _encode_observation(self, aid: str, obs_idx: int) -> np.ndarray:
        """One-hot encode observation and project to message space."""
        obs = np.zeros(self._obs_size[aid])
        obs[min(obs_idx, len(obs) - 1)] = 1.0
        return np.tanh(obs @ self._encoders[aid])

    def communicate(
        self,
        agents: dict[str, int],
        round: int = 0,
    ) -> dict[str, np.ndarray]:
        """
        One round of message passing.

        Each agent encodes its observation into a message. Messages are
        broadcast to all other agents. Returns the messages.

        Args:
            agents: Current observation (state/action index) per agent
            round: Communication round number (for multi-round protocols)

        Returns:
            Messages produced by each agent (agent_id -> message vector)
        """
        messages = {}
        for aid in self.agent_ids:
            obs_idx = agents.get(aid, 0)

            if round == 0:
                # First round: encode raw observation
                msg = self._encode_observation(aid, obs_idx)
            else:
                # Subsequent rounds: encode observation + received aggregate
                agg = self._aggregate_messages(aid)
                obs = np.zeros(self._obs_size[aid])
                obs[min(obs_idx, len(obs) - 1)] = 1.0
                combined = np.concatenate([obs, agg])
                # Project combined through decoder then re-encode
                intermediate = np.tanh(combined @ self._decoders[aid])
                # Use intermediate as new observation proxy
                proxy = np.zeros(self._obs_size[aid])
                proxy[:min(len(intermediate), len(proxy))] = intermediate[
                    :min(len(intermediate), len(proxy))
                ]
                msg = np.tanh(proxy @ self._encoders[aid])

            messages[aid] = msg

        self._last_messages = messages.copy()
        return messages

    def _aggregate_messages(self, exclude_id: str) -> np.ndarray:
        """Average messages from all other agents."""
        others = [
            msg for aid, msg in self._last_messages.items() if aid != exclude_id
        ]
        if not others:
            return np.zeros(self.message_size)
        return np.mean(others, axis=0)

    def step_with_communication(
        self,
        observations: dict[str, int],
        n_comm_rounds: int = 2,
    ) -> dict[str, int]:
        """
        Full step with multiple communication rounds before action selection.

        1. Run n_comm_rounds of message passing
        2. Each agent selects action using (observation, final_aggregate)
        3. Update communication parameters using REINFORCE

        Args:
            observations: Current state per agent
            n_comm_rounds: Number of message exchange rounds

        Returns:
            Actions selected by each agent
        """
        # Communication rounds
        for r in range(n_comm_rounds):
            self.communicate(observations, round=r)

        # Action selection using observation + aggregated message
        actions = {}
        for aid in self.agent_ids:
            agent = self.agents[aid]
            obs_idx = observations.get(aid, 0)

            obs = np.zeros(self._obs_size[aid])
            obs[min(obs_idx, len(obs) - 1)] = 1.0

            agg = self._aggregate_messages(aid)
            combined = np.concatenate([obs, agg])

            # Compute action logits
            logits = combined @ self._decoders[aid]

            # Softmax policy
            logits_shifted = logits - np.max(logits)
            exp_l = np.exp(logits_shifted)
            policy = exp_l / (np.sum(exp_l) + 1e-10)

            # Epsilon-greedy
            if np.random.random() < self.epsilon:
                action = np.random.randint(agent.n_actions)
            else:
                action = np.random.choice(agent.n_actions, p=policy)

            actions[aid] = int(action)
            agent.policy = policy

        self._last_actions = actions.copy()
        self._iteration += 1
        return actions

    def update(
        self,
        rewards: dict[str, float],
        observations: dict[str, int],
    ) -> None:
        """
        Update communication parameters using REINFORCE.

        Gradient: d/d_theta [ R * log pi(a|o, m) ]

        Args:
            rewards: Rewards received after actions
            observations: Observations that were used for action selection
        """
        for aid in self.agent_ids:
            agent = self.agents[aid]
            r = rewards.get(aid, 0.0)
            agent.reward_history.append(r)

            obs_idx = observations.get(aid, 0)
            obs = np.zeros(self._obs_size[aid])
            obs[min(obs_idx, len(obs) - 1)] = 1.0
            agg = self._aggregate_messages(aid)
            combined = np.concatenate([obs, agg])

            logits = combined @ self._decoders[aid]
            logits_shifted = logits - np.max(logits)
            exp_l = np.exp(logits_shifted)
            policy = exp_l / (np.sum(exp_l) + 1e-10)

            action = self._last_actions.get(aid, 0)

            # REINFORCE gradient for decoder
            one_hot_a = np.zeros(agent.n_actions)
            one_hot_a[action] = 1.0
            grad_logits = r * (one_hot_a - policy)  # (n_actions,)
            grad_decoder = np.outer(combined, grad_logits)  # (input, n_actions)
            self._decoders[aid] += self.lr * grad_decoder

            # Update encoder via chain rule (simplified)
            # Gradient flows: reward -> logits -> aggregate -> messages -> encoder
            # Approximate: perturb encoder in direction that increases reward
            grad_agg = grad_logits @ self._decoders[aid][self._obs_size[aid]:, :].T
            # grad_agg is (message_size,), flows to encoder
            grad_encoder = np.outer(obs, grad_agg * (1 - self._last_messages.get(aid, np.zeros(self.message_size)) ** 2))
            self._encoders[aid] += self.lr * grad_encoder

    def get_joint_policy(self) -> MARLResult:
        """Extract current joint policy."""
        joint = {aid: agent.policy.copy() for aid, agent in self.agents.items()}

        total_welfare = 0.0
        for aid, agent in self.agents.items():
            recent = agent.reward_history[-100:] if agent.reward_history else [0.0]
            total_welfare += np.mean(recent)

        # Nash gap from policies
        max_gap = 0.0
        for aid, agent in self.agents.items():
            q = agent.q_values if agent.q_values.ndim == 1 else np.mean(agent.q_values, axis=0)
            if np.any(q != 0):
                expected = np.dot(agent.policy, q)
                best = np.max(q)
                max_gap = max(max_gap, best - expected)

        return MARLResult(
            joint_policy=joint,
            social_welfare=total_welfare,
            nash_gap=max_gap,
            iterations=self._iteration,
        )
