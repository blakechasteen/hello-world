"""
Hierarchical Reinforcement Learning — Options Framework and Task Decomposition
===============================================================================

Temporal abstraction for RL: instead of choosing primitive actions at every
timestep, select higher-level "options" (temporally extended actions) that
execute sub-policies over multiple steps.

Classes:
    OptionsFramework: Option-critic architecture with SMDP value iteration
    MAXQ: Recursive task decomposition with value function sharing
    FeudalRL: Manager-worker hierarchy with intrinsic motivation

Key Concepts:
    - Option: (initiation set, policy, termination condition) triple
    - SMDP: Semi-Markov Decision Process (variable-duration actions)
    - Intrinsic reward: internal signal for achieving sub-goals
    - Task graph: DAG of subtask dependencies

Applications:
    - Multi-step reasoning with composable sub-strategies
    - HoloLoom weaving cycle decomposition (9 stages as options)
    - Hierarchical scoring (GPS priority bands → per-band optimization)
    - Goal-conditioned planning with natural temporal abstraction

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Option:
    """
    A temporally extended action (Sutton, Precup & Singh, 1999).

    An option o = (I, pi, beta) consists of:
        - Initiation set I: states where the option can start
        - Policy pi: action selection within the option
        - Termination beta: probability of stopping at each state

    Attributes:
        name: Human-readable identifier
        initiation_set: Boolean mask or callable, shape (n_states,)
        policy: Action probabilities, shape (n_states, n_actions)
        termination: Termination probability per state, shape (n_states,)
        n_states: Number of states in the MDP
        n_actions: Number of primitive actions
    """
    name: str
    initiation_set: np.ndarray | Callable
    policy: np.ndarray
    termination: np.ndarray | Callable
    n_states: int
    n_actions: int

    def can_initiate(self, state: int) -> bool:
        """Check if this option can start in the given state."""
        if callable(self.initiation_set):
            return bool(self.initiation_set(state))
        return bool(self.initiation_set[state])

    def termination_prob(self, state: int) -> float:
        """Get termination probability at the given state."""
        if callable(self.termination):
            return float(self.termination(state))
        return float(self.termination[state])

    def select_action(self, state: int) -> int:
        """Sample an action from the option's policy at the given state."""
        probs = self.policy[state]
        return int(np.random.choice(self.n_actions, p=probs))


@dataclass
class HRLResult:
    """
    Result of hierarchical RL.

    Attributes:
        options: Learned or provided options
        meta_policy: Policy over options, shape (n_states, n_options)
        value: Expected return under the hierarchical policy
        n_episodes: Number of training episodes completed
    """
    options: list[Option]
    meta_policy: np.ndarray
    value: float
    n_episodes: int


class OptionsFramework:
    """
    Option-Critic Architecture with SMDP Planning.

    Learns options end-to-end: initiation sets, intra-option policies,
    and termination functions are all optimized jointly. Combines:

    1. Option-critic (Bacon et al., 2017): gradient-based option learning
    2. SMDP value iteration: planning over learned options
    3. Option models: multi-step transition/reward models for planning

    The key insight: options create a two-level MDP. The "meta" MDP selects
    options, the "intra" MDP executes primitive actions within each option.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        discount: float = 0.99,
        lr: float = 0.01,
        termination_reg: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.lr = lr
        self.termination_reg = termination_reg

    def learn_options(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        n_options: int,
        n_episodes: int = 1000,
        max_steps: int = 200,
    ) -> list[Option]:
        """
        Learn options via the option-critic algorithm.

        Each option has learnable:
            - Policy logits: shape (n_states, n_actions)
            - Termination logits: shape (n_states,)
            - Initiation set: derived from where the option has been useful

        Training loop:
            1. Select option using meta-policy (epsilon-greedy over option Q-values)
            2. Execute option until termination
            3. Update intra-option Q-values, termination gradients, meta Q-values

        Args:
            transition: Transition model, shape (n_states, n_actions, n_states)
            reward: Reward per state or (state, action), shape (n_states,) or (n_states, n_actions)
            n_options: Number of options to learn
            n_episodes: Training episodes
            max_steps: Maximum steps per episode

        Returns:
            List of learned Option objects
        """
        n_s, n_a = self.n_states, self.n_actions

        # Reward handling
        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_a))
        else:
            R = reward

        # Initialize option parameters
        # Intra-option policy logits
        policy_logits = np.random.randn(n_options, n_s, n_a) * 0.1
        # Termination logits (sigmoid applied later)
        term_logits = np.zeros((n_options, n_s))
        # Intra-option Q-values
        Q_option = np.zeros((n_options, n_s, n_a))
        # Meta Q-values: Q(s, o)
        Q_meta = np.zeros((n_s, n_options))
        # Initiation counts (for learning initiation sets)
        init_counts = np.zeros((n_options, n_s))
        init_successes = np.zeros((n_options, n_s))

        epsilon = 0.2

        for episode in range(n_episodes):
            state = np.random.randint(n_s)
            epsilon_t = max(0.01, epsilon * (1 - episode / n_episodes))

            for step in range(max_steps):
                # Select option (epsilon-greedy over meta Q-values)
                if np.random.random() < epsilon_t:
                    option_idx = np.random.randint(n_options)
                else:
                    option_idx = int(np.argmax(Q_meta[state]))

                # Execute option
                option_start = state
                option_reward = 0.0
                option_steps = 0

                while True:
                    # Intra-option action selection (softmax)
                    logits = policy_logits[option_idx, state]
                    logits_shifted = logits - np.max(logits)
                    exp_l = np.exp(logits_shifted)
                    pi = exp_l / (np.sum(exp_l) + 1e-10)

                    if np.random.random() < epsilon_t * 0.5:
                        action = np.random.randint(n_a)
                    else:
                        action = np.random.choice(n_a, p=pi)

                    # Transition
                    next_state = np.random.choice(
                        n_s, p=transition[state, action]
                    )
                    r = R[state, action]
                    option_reward += (self.discount ** option_steps) * r
                    option_steps += 1

                    # Intra-option Q-learning update
                    best_next = np.max(Q_option[option_idx, next_state])
                    td_error = r + self.discount * best_next - Q_option[option_idx, state, action]
                    Q_option[option_idx, state, action] += self.lr * td_error

                    # Policy gradient for intra-option policy
                    advantage = Q_option[option_idx, state, action] - np.dot(
                        pi, Q_option[option_idx, state]
                    )
                    one_hot = np.zeros(n_a)
                    one_hot[action] = 1.0
                    grad = advantage * (one_hot - pi)
                    policy_logits[option_idx, state] += self.lr * grad

                    state = next_state

                    # Termination check
                    term_prob = 1.0 / (1.0 + np.exp(-term_logits[option_idx, state]))

                    if np.random.random() < term_prob or option_steps >= max_steps // 4:
                        break

                # Update termination function
                # Advantage of continuing vs. terminating
                V_option = np.max(Q_option[option_idx, state])
                V_meta = np.max(Q_meta[state])
                term_advantage = V_meta - V_option + self.termination_reg

                term_prob = 1.0 / (1.0 + np.exp(-term_logits[option_idx, state]))
                term_grad = term_advantage * term_prob * (1 - term_prob)
                term_logits[option_idx, state] += self.lr * term_grad

                # Update meta Q-values (SMDP update)
                best_next_option = np.max(Q_meta[state])
                meta_td = (
                    option_reward
                    + (self.discount ** option_steps) * best_next_option
                    - Q_meta[option_start, option_idx]
                )
                Q_meta[option_start, option_idx] += self.lr * meta_td

                # Track initiation success
                init_counts[option_idx, option_start] += 1
                if option_reward > 0:
                    init_successes[option_idx, option_start] += 1

        # Build Option objects
        options = []
        for o in range(n_options):
            # Policy from logits
            logits = policy_logits[o]
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits_shifted)
            policy = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-10)

            # Termination from logits
            termination = 1.0 / (1.0 + np.exp(-term_logits[o]))

            # Initiation set: states where option has been tried and succeeded
            success_rate = np.where(
                init_counts[o] > 0,
                init_successes[o] / init_counts[o],
                0.5,  # Optimistic for untried states
            )
            initiation = success_rate > 0.2

            options.append(Option(
                name=f"option_{o}",
                initiation_set=initiation,
                policy=policy,
                termination=termination,
                n_states=n_s,
                n_actions=n_a,
            ))

        return options

    def smdp_value_iteration(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        options: list[Option],
        discount: float = 0.99,
        n_iter: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Semi-MDP value iteration over options.

        For each option o at state s, compute the multi-step model:
            - P_o(s'|s): probability of reaching s' when executing o from s
            - R_o(s): expected discounted reward from executing o starting at s
            - k_o(s): expected duration of option o from s

        Then value iteration:
            V(s) = max_o [R_o(s) + gamma^{k_o(s)} * sum_s' P_o(s'|s) * V(s')]

        Args:
            transition: Primitive transition model, shape (n_states, n_actions, n_states)
            reward: Primitive reward, shape (n_states,) or (n_states, n_actions)
            options: List of options to plan over
            discount: Discount factor
            n_iter: Value iteration sweeps

        Returns:
            V: Value function, shape (n_states,)
            meta_policy: Policy over options, shape (n_states, n_options)
        """
        n_s = self.n_states
        n_options = len(options)

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, self.n_actions))
        else:
            R = reward

        # Compute option models via simulation
        opt_trans, opt_reward, opt_duration = self._compute_option_models(
            transition, R, options, discount
        )

        # SMDP value iteration
        V = np.zeros(n_s)
        for _ in range(n_iter):
            Q_options = np.zeros((n_s, n_options))
            for o_idx, option in enumerate(options):
                for s in range(n_s):
                    if option.can_initiate(s):
                        expected_next = np.dot(opt_trans[o_idx, s], V)
                        gamma_k = discount ** opt_duration[o_idx, s]
                        Q_options[s, o_idx] = (
                            opt_reward[o_idx, s] + gamma_k * expected_next
                        )
                    else:
                        Q_options[s, o_idx] = -np.inf

            V_new = np.max(Q_options, axis=1)
            if np.max(np.abs(V_new - V)) < 1e-8:
                break
            V = V_new

        # Meta-policy (softmax over option Q-values)
        Q_shifted = Q_options - np.max(Q_options, axis=1, keepdims=True)
        Q_shifted = np.where(Q_options > -np.inf, Q_shifted, -100)
        exp_Q = np.exp(Q_shifted)
        meta_policy = exp_Q / (np.sum(exp_Q, axis=1, keepdims=True) + 1e-10)

        return V, meta_policy

    def _compute_option_models(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        options: list[Option],
        discount: float,
        n_simulations: int = 100,
        max_option_len: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute option models via Monte Carlo simulation.

        For each option o and starting state s:
            - Execute the option's policy until termination
            - Record terminal state, accumulated reward, duration
            - Average over simulations

        Returns:
            opt_trans: Multi-step transition, shape (n_options, n_states, n_states)
            opt_reward: Expected option reward, shape (n_options, n_states)
            opt_duration: Expected option duration, shape (n_options, n_states)
        """
        n_s = self.n_states
        n_options = len(options)

        opt_trans = np.zeros((n_options, n_s, n_s))
        opt_reward = np.zeros((n_options, n_s))
        opt_duration = np.ones((n_options, n_s))  # Default duration 1

        for o_idx, option in enumerate(options):
            for s in range(n_s):
                if not option.can_initiate(s):
                    opt_trans[o_idx, s, s] = 1.0  # Stay in place
                    continue

                total_reward = 0.0
                total_duration = 0.0
                terminal_counts = np.zeros(n_s)

                for _ in range(n_simulations):
                    state = s
                    ep_reward = 0.0
                    steps = 0

                    for t in range(max_option_len):
                        action = option.select_action(state)
                        next_state = np.random.choice(
                            n_s, p=transition[state, action]
                        )
                        ep_reward += (discount ** t) * reward[state, action]
                        state = next_state
                        steps += 1

                        if np.random.random() < option.termination_prob(state):
                            break

                    terminal_counts[state] += 1
                    total_reward += ep_reward
                    total_duration += steps

                opt_trans[o_idx, s] = terminal_counts / n_simulations
                opt_reward[o_idx, s] = total_reward / n_simulations
                opt_duration[o_idx, s] = total_duration / n_simulations

        return opt_trans, opt_reward, opt_duration

    @staticmethod
    def execute_option(
        option: Option,
        state: int,
        transition: np.ndarray,
        reward: np.ndarray | None = None,
        max_steps: int = 100,
    ) -> tuple[list[tuple[int, int]], int, int]:
        """
        Execute an option from a given state until termination.

        Args:
            option: The option to execute
            state: Starting state
            transition: Transition model, shape (n_states, n_actions, n_states)
            reward: Optional reward, shape (n_states,) or (n_states, n_actions)
            max_steps: Maximum execution length

        Returns:
            trajectory: List of (state, action) pairs
            final_state: State where the option terminated
            duration: Number of steps taken
        """
        trajectory = []
        n_s = transition.shape[0]

        for t in range(max_steps):
            action = option.select_action(state)
            trajectory.append((state, action))

            next_state = np.random.choice(n_s, p=transition[state, action])
            state = next_state

            if np.random.random() < option.termination_prob(state):
                break

        return trajectory, state, len(trajectory)

    @staticmethod
    def option_model(
        option: Option,
        transition: np.ndarray,
        reward: np.ndarray,
        discount: float = 0.99,
        n_iter: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the multi-step transition and reward model for an option
        analytically (when possible).

        Uses the option's policy to compute:
            T_o(s'|s) = sum_a pi(a|s) * [beta(s') * T(s'|s,a)
                         + (1-beta(s')) * sum_a' pi(a'|s') * T_o(s''|s') ...]

        Approximated via matrix iteration.

        Args:
            option: The option
            transition: Primitive transition, shape (n_states, n_actions, n_states)
            reward: Primitive reward, shape (n_states,) or (n_states, n_actions)
            discount: Discount factor
            n_iter: Iterations for convergence

        Returns:
            multi_step_trans: Option transition model, shape (n_states, n_states)
            multi_step_reward: Option reward model, shape (n_states,)
        """
        n_s, n_a, _ = transition.shape

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_a))
        else:
            R = reward

        # Policy-weighted one-step transition: T_pi(s'|s) = sum_a pi(a|s) * T(s'|s,a)
        T_pi = np.einsum("sa,sas'->ss'", option.policy, transition)

        # Policy-weighted one-step reward: R_pi(s) = sum_a pi(a|s) * R(s,a)
        R_pi = np.sum(option.policy * R, axis=1)

        # Termination probabilities
        beta = np.array([option.termination_prob(s) for s in range(n_s)])

        # Multi-step model via iteration
        # P(terminate at s' starting from s) = T_pi(s'|s) * beta(s')
        #   + sum_{s''} T_pi(s''|s) * (1-beta(s'')) * P(terminate at s' from s'')
        multi_trans = np.zeros((n_s, n_s))
        multi_reward = np.zeros(n_s)

        # Iterative computation
        # Let C(s) = (1 - beta(s)) continuation probability
        C = np.diag(1.0 - beta)

        # Accumulated transition: sum_{k=0}^{inf} (C @ T_pi)^k @ diag(beta) @ T_pi
        # = (I - C @ T_pi)^{-1} @ diag(beta) @ T_pi  (if convergent)
        CT = C @ T_pi
        power = np.eye(n_s)
        multi_trans = np.zeros((n_s, n_s))
        multi_reward = np.zeros(n_s)

        for k in range(n_iter):
            # At step k: probability of being in each state
            # Terminate at this step: power @ T_pi @ diag(beta)
            step_trans = power @ T_pi * beta[np.newaxis, :]
            step_reward = (discount ** k) * (power @ R_pi)

            multi_trans += step_trans
            multi_reward += (discount ** k) * np.sum(
                power * R_pi[np.newaxis, :], axis=1
            ) if k == 0 else np.zeros(n_s)

            if k > 0:
                multi_reward += (discount ** k) * power @ R_pi

            # Continue
            power = power @ CT

            if np.max(np.abs(power)) < 1e-10:
                break

        # Normalize transition rows
        row_sums = np.sum(multi_trans, axis=1, keepdims=True)
        multi_trans = np.where(row_sums > 1e-10, multi_trans / row_sums, 1.0 / n_s)

        return multi_trans, multi_reward


class MAXQ:
    """
    MAXQ Hierarchical Task Decomposition (Dietterich, 2000).

    Decomposes a task into a hierarchy of subtasks. Each subtask has its own
    value function. Value decomposition:

        Q(s, subtask) = V(subtask, s) + C(parent, s, subtask)

    where V is the subtask's internal value and C is the completion function
    (expected reward after the subtask finishes, under the parent's policy).

    The task graph is a DAG where leaves are primitive actions and internal
    nodes are composite subtasks that invoke children.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        discount: float = 0.99,
        lr: float = 0.1,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.lr = lr

    def decompose(
        self,
        task_graph: dict[str, list[str]],
        transition: np.ndarray,
        reward: np.ndarray,
        n_episodes: int = 500,
        max_steps: int = 100,
    ) -> dict[str, np.ndarray]:
        """
        Learn value functions for each subtask in the task graph.

        The task graph maps each composite task to its children:
            {"root": ["navigate", "pickup"],
             "navigate": ["north", "south", "east", "west"],
             "pickup": ["grasp"]}

        Leaves must be valid primitive action names or indices.

        Args:
            task_graph: DAG of subtask dependencies
            transition: Transition model, shape (n_states, n_actions, n_states)
            reward: Task reward, shape (n_states,) or (n_states, n_actions)
            n_episodes: Training episodes
            max_steps: Maximum steps per episode

        Returns:
            Value functions per subtask, {task_name: V(s)} where V shape is (n_states,)
        """
        n_s, n_a = self.n_states, self.n_actions

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_a))
        else:
            R = reward

        # Find all tasks and identify primitives
        all_tasks = set()
        for parent, children in task_graph.items():
            all_tasks.add(parent)
            all_tasks.update(children)

        composite_tasks = set(task_graph.keys())
        primitive_tasks = all_tasks - composite_tasks

        # Value functions: V(task, state)
        V = {task: np.zeros(n_s) for task in all_tasks}

        # Completion functions: C(parent, state, child)
        C = {}
        for parent, children in task_graph.items():
            C[parent] = {child: np.zeros(n_s) for child in children}

        # Map primitive task names to action indices
        primitive_to_action = {}
        for i, task in enumerate(sorted(primitive_tasks)):
            if i < n_a:
                primitive_to_action[task] = i

        def is_primitive(task: str) -> bool:
            return task not in composite_tasks

        def execute_task(
            task: str, state: int, depth: int = 0
        ) -> tuple[list[int], float, int]:
            """Recursively execute a task, returning (actions, reward, final_state)."""
            if depth > 20:
                return [], 0.0, state

            if is_primitive(task):
                action = primitive_to_action.get(task, 0)
                next_state = np.random.choice(n_s, p=transition[state, action])
                r = R[state, action]
                return [action], r, next_state

            children = task_graph[task]
            total_reward = 0.0
            all_actions = []
            current_state = state
            steps = 0

            for child in children:
                if steps >= max_steps:
                    break

                # Select child using Q = V(child, s) + C(task, s, child)
                child_values = [
                    V[c][current_state] + C[task][c][current_state]
                    for c in children
                ]
                # Epsilon-greedy over children
                if np.random.random() < 0.1:
                    selected_child = np.random.choice(children)
                else:
                    selected_child = children[np.argmax(child_values)]

                actions, r, next_state = execute_task(
                    selected_child, current_state, depth + 1
                )
                all_actions.extend(actions)
                total_reward += (self.discount ** steps) * r
                steps += len(actions)

                # Update completion function
                best_future = max(
                    V[c][next_state] + C[task][c][next_state]
                    for c in children
                )
                td_error = (
                    (self.discount ** len(actions)) * best_future
                    - C[task][selected_child][current_state]
                )
                C[task][selected_child][current_state] += self.lr * td_error

                current_state = next_state

            # Update V for this composite task
            V[task][state] += self.lr * (total_reward - V[task][state])

            return all_actions, total_reward, current_state

        # Train
        root_tasks = [t for t in task_graph if not any(
            t in children for children in task_graph.values()
        )]
        root = root_tasks[0] if root_tasks else list(task_graph.keys())[0]

        for episode in range(n_episodes):
            state = np.random.randint(n_s)
            execute_task(root, state)

        return V

    def execute(
        self,
        task: str,
        state: int,
        task_graph: dict[str, list[str]],
        transition: np.ndarray,
        reward: np.ndarray,
        V: dict[str, np.ndarray],
        max_steps: int = 100,
    ) -> tuple[list[int], float]:
        """
        Execute a learned hierarchical policy.

        Uses the learned value functions to greedily select subtasks
        at each level of the hierarchy.

        Args:
            task: Root task to execute
            state: Starting state
            task_graph: Task dependency graph
            transition: Transition model
            reward: Reward function
            V: Learned value functions from decompose()
            max_steps: Maximum steps

        Returns:
            actions: List of primitive actions taken
            total_reward: Accumulated reward
        """
        n_s, n_a = self.n_states, self.n_actions

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_a))
        else:
            R = reward

        composite_tasks = set(task_graph.keys())
        primitive_tasks = set()
        for children in task_graph.values():
            for c in children:
                if c not in composite_tasks:
                    primitive_tasks.add(c)

        primitive_to_action = {}
        for i, t in enumerate(sorted(primitive_tasks)):
            if i < n_a:
                primitive_to_action[t] = i

        actions = []
        total_reward = 0.0

        def _execute(t: str, s: int, depth: int) -> tuple[int, float]:
            nonlocal actions, total_reward
            if depth > 20 or len(actions) >= max_steps:
                return s, 0.0

            if t not in composite_tasks:
                a = primitive_to_action.get(t, 0)
                ns = np.random.choice(n_s, p=transition[s, a])
                r = R[s, a]
                actions.append(a)
                total_reward += r
                return ns, r

            children = task_graph[t]
            # Greedy child selection
            child_vals = [V[c][s] for c in children]
            best_child = children[np.argmax(child_vals)]
            return _execute(best_child, s, depth + 1)

        _execute(task, state, 0)
        return actions, total_reward


class FeudalRL:
    """
    Feudal Reinforcement Learning (Dayan & Hinton, 1993; Vezhnevets et al., 2017).

    Two-level hierarchy:
        - Manager: observes state, sets goals in a goal space
        - Worker: receives goals, takes primitive actions to achieve them

    The manager operates at a slower timescale (every k steps).
    The worker receives intrinsic reward for making progress toward the goal.

    Key insight: the manager and worker can be trained independently.
    The manager learns which goals lead to high task reward.
    The worker learns how to reach any goal (goal-conditioned policy).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        goal_dim: int = 4,
        manager_horizon: int = 5,
        discount: float = 0.99,
        lr: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.goal_dim = goal_dim
        self.manager_horizon = manager_horizon
        self.discount = discount
        self.lr = lr

        # State embedding for goal space: (n_states, goal_dim)
        self._state_embed = np.random.randn(n_states, goal_dim) * 0.3

        # Manager: state -> goal direction (logits over discretized goals)
        self._n_goal_dirs = 2 * goal_dim  # +/- along each dimension
        self._manager_q = np.zeros((n_states, self._n_goal_dirs))

        # Worker: (state, goal) -> action Q-values
        # Discretize goal to bins for tabular worker
        self._n_goal_bins = self._n_goal_dirs
        self._worker_q = np.zeros((n_states, self._n_goal_bins, n_actions))

    def _goal_direction(self, goal_idx: int) -> np.ndarray:
        """Convert goal index to a direction vector in goal space."""
        direction = np.zeros(self.goal_dim)
        dim = goal_idx // 2
        sign = 1.0 if goal_idx % 2 == 0 else -1.0
        if dim < self.goal_dim:
            direction[dim] = sign
        return direction

    def _embed_state(self, state: int) -> np.ndarray:
        """Map state to goal space."""
        return self._state_embed[state]

    def _goal_bin(self, goal: np.ndarray) -> int:
        """Discretize a goal vector to a bin index."""
        # Find the dominant dimension and sign
        abs_goal = np.abs(goal)
        if np.max(abs_goal) < 1e-10:
            return 0
        dim = int(np.argmax(abs_goal))
        sign = 0 if goal[dim] >= 0 else 1
        return min(dim * 2 + sign, self._n_goal_bins - 1)

    def manager_policy(
        self,
        state: int,
        epsilon: float = 0.1,
    ) -> np.ndarray:
        """
        Manager selects a goal direction in the learned goal space.

        Uses epsilon-greedy over manager Q-values to balance exploration.

        Args:
            state: Current state index
            epsilon: Exploration rate

        Returns:
            Goal direction vector in goal space, shape (goal_dim,)
        """
        if np.random.random() < epsilon:
            goal_idx = np.random.randint(self._n_goal_dirs)
        else:
            goal_idx = int(np.argmax(self._manager_q[state]))

        return self._goal_direction(goal_idx)

    def worker_policy(
        self,
        state: int,
        goal: np.ndarray,
        epsilon: float = 0.1,
    ) -> int:
        """
        Worker selects a primitive action to make progress toward the goal.

        Args:
            state: Current state index
            goal: Goal direction vector from manager

        Returns:
            Primitive action index
        """
        g_bin = self._goal_bin(goal)

        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._worker_q[state, g_bin]))

    def intrinsic_reward(
        self,
        state: int,
        goal: np.ndarray,
        next_state: int,
    ) -> float:
        """
        Compute intrinsic reward for the worker.

        Reward = cosine similarity between the goal direction and the
        actual transition direction in goal space.

            r_intrinsic = cos(goal, embed(s') - embed(s))

        High reward when the worker moves in the direction the manager wanted.

        Args:
            state: State before action
            goal: Goal direction from manager
            next_state: State after action

        Returns:
            Intrinsic reward in [-1, 1]
        """
        embed_s = self._embed_state(state)
        embed_s_next = self._embed_state(next_state)
        direction = embed_s_next - embed_s

        # Cosine similarity
        goal_norm = np.linalg.norm(goal)
        dir_norm = np.linalg.norm(direction)

        if goal_norm < 1e-10 or dir_norm < 1e-10:
            return 0.0

        return float(np.dot(goal, direction) / (goal_norm * dir_norm))

    def train(
        self,
        transition: np.ndarray,
        reward: np.ndarray,
        n_episodes: int = 500,
        max_steps: int = 200,
        intrinsic_weight: float = 0.5,
    ) -> HRLResult:
        """
        Train the feudal hierarchy.

        Manager updates every manager_horizon steps using extrinsic reward.
        Worker updates every step using intrinsic reward + weighted extrinsic.

        Args:
            transition: Transition model, shape (n_states, n_actions, n_states)
            reward: Extrinsic reward, shape (n_states,) or (n_states, n_actions)
            n_episodes: Training episodes
            max_steps: Max steps per episode
            intrinsic_weight: Weight of intrinsic vs extrinsic for worker

        Returns:
            HRLResult with learned options representing the manager-worker hierarchy
        """
        n_s, n_a = self.n_states, self.n_actions

        if reward.ndim == 1:
            R = np.tile(reward[:, np.newaxis], (1, n_a))
        else:
            R = reward

        for episode in range(n_episodes):
            state = np.random.randint(n_s)
            epsilon = max(0.01, 0.3 * (1 - episode / n_episodes))

            goal = self.manager_policy(state, epsilon)
            manager_start = state
            manager_reward = 0.0
            manager_steps = 0

            for step in range(max_steps):
                # Worker selects action
                action = self.worker_policy(state, goal, epsilon)
                next_state = np.random.choice(n_s, p=transition[state, action])

                # Rewards
                extrinsic = R[state, action]
                intrinsic = self.intrinsic_reward(state, goal, next_state)
                worker_reward = (
                    intrinsic_weight * intrinsic
                    + (1 - intrinsic_weight) * extrinsic
                )

                # Worker Q-learning
                g_bin = self._goal_bin(goal)
                best_next = np.max(self._worker_q[next_state, g_bin])
                td_error = (
                    worker_reward
                    + self.discount * best_next
                    - self._worker_q[state, g_bin, action]
                )
                self._worker_q[state, g_bin, action] += self.lr * td_error

                manager_reward += (self.discount ** manager_steps) * extrinsic
                manager_steps += 1
                state = next_state

                # Manager update at horizon boundary
                if manager_steps >= self.manager_horizon:
                    goal_idx = int(np.argmax(
                        [np.dot(self._goal_direction(i), goal) for i in range(self._n_goal_dirs)]
                    ))
                    best_next_manager = np.max(self._manager_q[state])
                    td_manager = (
                        manager_reward
                        + (self.discount ** manager_steps) * best_next_manager
                        - self._manager_q[manager_start, goal_idx]
                    )
                    self._manager_q[manager_start, goal_idx] += self.lr * td_manager

                    # New goal
                    goal = self.manager_policy(state, epsilon)
                    manager_start = state
                    manager_reward = 0.0
                    manager_steps = 0

            # Update state embeddings toward useful representations
            # Gradient: make embeddings reflect reward structure
            self._update_embeddings(transition, R)

        # Package results as options
        options = self._extract_options()
        meta_policy = self._manager_softmax()

        # Compute expected value
        V = np.max(self._manager_q, axis=1)
        avg_value = float(np.mean(V))

        return HRLResult(
            options=options,
            meta_policy=meta_policy,
            value=avg_value,
            n_episodes=n_episodes,
        )

    def _update_embeddings(
        self, transition: np.ndarray, reward: np.ndarray
    ) -> None:
        """
        Slowly adjust state embeddings so nearby states (in reward/transition
        similarity) have nearby embeddings.
        """
        n_s = self.n_states
        # Random pair of states
        s1, s2 = np.random.randint(n_s, size=2)

        # Reward similarity
        r_sim = -np.abs(np.mean(reward[s1]) - np.mean(reward[s2]))

        # Transition similarity (TV distance)
        t_sim = 0.0
        for a in range(self.n_actions):
            t_sim -= 0.5 * np.sum(np.abs(transition[s1, a] - transition[s2, a]))
        t_sim /= self.n_actions

        # Embedding distance
        embed_diff = self._state_embed[s1] - self._state_embed[s2]
        embed_dist = np.linalg.norm(embed_diff)

        # Pull together if similar, push apart if different
        similarity = 0.5 * r_sim + 0.5 * t_sim  # In [-1, 0]
        target_dist = -similarity  # More similar -> closer

        if embed_dist > 1e-10:
            direction = embed_diff / embed_dist
            grad = (embed_dist - target_dist) * direction
            self._state_embed[s1] -= 0.001 * grad
            self._state_embed[s2] += 0.001 * grad

    def _extract_options(self) -> list[Option]:
        """Convert the feudal hierarchy into options."""
        options = []
        for g_idx in range(self._n_goal_dirs):
            goal = self._goal_direction(g_idx)
            g_bin = self._goal_bin(goal)

            # Worker policy for this goal
            q = self._worker_q[:, g_bin, :]
            q_shifted = q - np.max(q, axis=1, keepdims=True)
            exp_q = np.exp(q_shifted)
            policy = exp_q / (np.sum(exp_q, axis=1, keepdims=True) + 1e-10)

            # Initiation: everywhere (feudal has no initiation constraints)
            initiation = np.ones(self.n_states, dtype=bool)

            # Termination: after manager_horizon steps (fixed)
            step_counter = [0]

            def make_termination(horizon):
                def term_fn(state):
                    return 1.0 / horizon  # Expected duration = horizon
                return term_fn

            termination_fn = make_termination(self.manager_horizon)

            options.append(Option(
                name=f"goal_{g_idx}",
                initiation_set=initiation,
                policy=policy,
                termination=termination_fn,
                n_states=self.n_states,
                n_actions=self.n_actions,
            ))

        return options

    def _manager_softmax(self) -> np.ndarray:
        """Softmax policy over goals from manager Q-values."""
        q = self._manager_q
        q_shifted = q - np.max(q, axis=1, keepdims=True)
        exp_q = np.exp(q_shifted)
        return exp_q / (np.sum(exp_q, axis=1, keepdims=True) + 1e-10)
