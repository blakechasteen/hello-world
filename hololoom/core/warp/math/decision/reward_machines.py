"""
Reward Machines — RL4.7
========================

Finite-state machines that specify non-Markovian reward functions.
A reward machine separates the task specification from the learning
algorithm, enabling reward decomposition and cross-product MDPs.

Classes:
    RewardMachine: Finite-state reward specification
    RMQLearning: Q-learning with reward machine decomposition
    LTLFormula: Simple LTL parser for reward machine construction

Key Concepts:
    - Reward machine: (U, u_0, delta_u, delta_r) finite automaton
    - Cross-product MDP: M x RM gives Markovian reward from non-Markovian spec
    - Q-function decomposition: separate Q per RM state
    - LTL to RM: convert temporal logic formulas to automata

Applications:
    - Specifying multi-step reasoning goals for HoloLoom
    - Sequential task completion with intermediate rewards
    - Composable reward specifications for weaving stages

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardMachine:
    """
    A reward machine: finite-state specification of reward structure.

    The RM is a tuple (U, u_0, delta_u, delta_r, F) where:
    - U: finite set of RM states
    - u_0: initial RM state
    - delta_u: U x Sigma -> U (state transition function over events)
    - delta_r: U x U -> R (reward for RM state transitions)
    - F: terminal/accepting RM states

    Events (Sigma) are high-level propositions about the environment
    state (e.g., "key_picked_up", "door_reached").

    Attributes:
        states: List of RM state labels
        initial_state: Starting RM state
        transitions: Dict mapping (rm_state, event) -> next_rm_state
        rewards: Dict mapping (rm_state, next_rm_state) -> reward
        terminal_states: Set of accepting/terminal RM states
    """
    states: list[int]
    initial_state: int = 0
    transitions: dict[tuple[int, str], int] = field(default_factory=dict)
    rewards: dict[tuple[int, int], float] = field(default_factory=dict)
    terminal_states: set[int] = field(default_factory=set)

    def step(self, rm_state: int, event: str) -> tuple[int, float]:
        """
        Transition the RM given an event.

        Args:
            rm_state: Current RM state
            event: Observed event/proposition

        Returns:
            (next_rm_state, reward) from the RM transition
        """
        key = (rm_state, event)
        if key in self.transitions:
            next_state = self.transitions[key]
            reward = self.rewards.get((rm_state, next_state), 0.0)
            return next_state, reward
        # Self-loop with zero reward if no transition defined
        return rm_state, 0.0

    def is_terminal(self, rm_state: int) -> bool:
        """Check if RM state is terminal/accepting."""
        return rm_state in self.terminal_states

    def get_events(self) -> set[str]:
        """Return all events referenced in transitions."""
        return {event for _, event in self.transitions.keys()}

    def n_states(self) -> int:
        """Number of RM states."""
        return len(self.states)

    @staticmethod
    def sequential_tasks(
        events: list[str],
        intermediate_reward: float = 1.0,
        completion_reward: float = 10.0,
    ) -> 'RewardMachine':
        """
        Create an RM for sequential task completion.

        Events must occur in order: e_0, e_1, ..., e_n.
        Each intermediate event gives intermediate_reward,
        the final event gives completion_reward.

        Args:
            events: Ordered list of events to complete
            intermediate_reward: Reward for each intermediate step
            completion_reward: Reward for final completion

        Returns:
            RewardMachine encoding the sequential task
        """
        n = len(events)
        states = list(range(n + 1))
        transitions = {}
        rewards = {}

        for i, event in enumerate(events):
            transitions[(i, event)] = i + 1
            if i < n - 1:
                rewards[(i, i + 1)] = intermediate_reward
            else:
                rewards[(i, i + 1)] = completion_reward

        return RewardMachine(
            states=states,
            initial_state=0,
            transitions=transitions,
            rewards=rewards,
            terminal_states={n},
        )

    @staticmethod
    def parallel_tasks(
        events: list[str],
        reward_per_task: float = 1.0,
        completion_bonus: float = 5.0,
    ) -> 'RewardMachine':
        """
        Create an RM for parallel (any-order) task completion.

        All events must occur, but in any order. Uses 2^n RM states
        to track which tasks are done (bitmask encoding).

        Args:
            events: List of events to complete (order-independent)
            reward_per_task: Reward for completing each individual task
            completion_bonus: Extra reward when all tasks are done

        Returns:
            RewardMachine encoding parallel tasks
        """
        n = len(events)
        n_rm_states = 2 ** n
        states = list(range(n_rm_states))
        transitions = {}
        rewards = {}

        for rm_state in range(n_rm_states):
            for i, event in enumerate(events):
                bit = 1 << i
                if not (rm_state & bit):  # Task i not yet completed
                    next_state = rm_state | bit
                    transitions[(rm_state, event)] = next_state
                    reward = reward_per_task
                    if next_state == n_rm_states - 1:
                        reward += completion_bonus
                    rewards[(rm_state, next_state)] = reward

        return RewardMachine(
            states=states,
            initial_state=0,
            transitions=transitions,
            rewards=rewards,
            terminal_states={n_rm_states - 1},
        )


class RMQLearning:
    """
    Q-learning with reward machines.

    Maintains a separate Q-function for each RM state:
        Q_u(s, a) for each u in U

    After observing a transition (s, a, s') with event e:
    1. Compute RM transition: (u, e) -> u'
    2. Get RM reward: delta_r(u, u')
    3. Update Q_u(s, a) using Q_{u'}(s', :) as the target
    4. Counterfactual updates: for ALL RM states u_i, compute
       what would have happened and update Q_{u_i} too

    The counterfactual updates provide massive sample efficiency
    gains because each real transition updates all RM states.
    """

    def learn(
        self,
        env_fn: Callable[[int, int], tuple[int, str, bool]],
        rm: RewardMachine,
        n_states: int,
        n_actions: int,
        n_episodes: int = 500,
        discount: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        initial_state_fn: Callable[[], int] | None = None,
        use_counterfactual: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Learn Q-values decomposed by RM state.

        Args:
            env_fn: (state, action) -> (next_state, event, done)
            rm: Reward machine specifying the task
            n_states: Number of environment states
            n_actions: Number of actions
            n_episodes: Number of training episodes
            discount: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
            initial_state_fn: Returns initial env state
            use_counterfactual: If True, update Q for all RM states

        Returns:
            (Q, policy) where Q[u, s, a] indexed by RM state,
            and policy[s] is greedy action under initial RM state
        """
        n_rm = rm.n_states()
        Q = np.zeros((n_rm, n_states, n_actions))

        for episode in range(n_episodes):
            state = initial_state_fn() if initial_state_fn else 0
            rm_state = rm.initial_state
            done = False
            steps = 0

            while not done and not rm.is_terminal(rm_state) and steps < 1000:
                # Epsilon-greedy using Q for current RM state
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[rm_state, state])

                next_state, event, done = env_fn(state, action)

                # RM transition
                next_rm_state, rm_reward = rm.step(rm_state, event)

                # Q-update for current RM state
                if rm.is_terminal(next_rm_state):
                    target = rm_reward
                else:
                    target = rm_reward + discount * np.max(Q[next_rm_state, next_state])
                Q[rm_state, state, action] += alpha * (target - Q[rm_state, state, action])

                # Counterfactual updates for all other RM states
                if use_counterfactual:
                    for u in rm.states:
                        if u == rm_state:
                            continue
                        u_next, u_reward = rm.step(u, event)
                        if rm.is_terminal(u_next):
                            cf_target = u_reward
                        else:
                            cf_target = u_reward + discount * np.max(Q[u_next, next_state])
                        Q[u, state, action] += alpha * (cf_target - Q[u, state, action])

                state = next_state
                rm_state = next_rm_state
                steps += 1

        # Policy under initial RM state
        policy = np.argmax(Q[rm.initial_state], axis=1)
        return Q, policy


@dataclass
class LTLFormula:
    """
    Simple Linear Temporal Logic formula for reward machine construction.

    Supported operators:
        - atomic(p): proposition p holds
        - eventually(phi): F phi (phi holds at some future time)
        - always(phi): G phi (phi holds at all future times)
        - sequence(phi, psi): phi then psi (phi before psi)
        - until(phi, psi): phi U psi (phi until psi)
    """
    operator: str
    operands: list = field(default_factory=list)

    @staticmethod
    def atomic(proposition: str) -> 'LTLFormula':
        return LTLFormula(operator="atomic", operands=[proposition])

    @staticmethod
    def eventually(formula: 'LTLFormula') -> 'LTLFormula':
        return LTLFormula(operator="eventually", operands=[formula])

    @staticmethod
    def always(formula: 'LTLFormula') -> 'LTLFormula':
        return LTLFormula(operator="always", operands=[formula])

    @staticmethod
    def sequence(*formulas: 'LTLFormula') -> 'LTLFormula':
        return LTLFormula(operator="sequence", operands=list(formulas))

    @staticmethod
    def until(phi: 'LTLFormula', psi: 'LTLFormula') -> 'LTLFormula':
        return LTLFormula(operator="until", operands=[phi, psi])

    @staticmethod
    def from_ltl(formula: 'LTLFormula') -> RewardMachine:
        """
        Convert an LTL formula to a reward machine.

        Handles simple patterns:
        - atomic(p) -> 2 states: wait for p, done
        - eventually(atomic(p)) -> same as atomic(p)
        - sequence(atomic(p), atomic(q)) -> 3 states: wait p, wait q, done
        - always(atomic(p)) -> 2 states: ok (self-loop on p), fail

        Args:
            formula: LTL formula to convert

        Returns:
            Equivalent RewardMachine
        """
        if formula.operator == "atomic":
            prop = formula.operands[0]
            return RewardMachine(
                states=[0, 1],
                initial_state=0,
                transitions={(0, prop): 1},
                rewards={(0, 1): 1.0},
                terminal_states={1},
            )

        elif formula.operator == "eventually":
            inner = formula.operands[0]
            if inner.operator == "atomic":
                prop = inner.operands[0]
                return RewardMachine(
                    states=[0, 1],
                    initial_state=0,
                    transitions={(0, prop): 1},
                    rewards={(0, 1): 1.0},
                    terminal_states={1},
                )
            # Recurse for complex inner formulas
            return LTLFormula.from_ltl(inner)

        elif formula.operator == "sequence":
            # Extract atomic propositions
            events = []
            for operand in formula.operands:
                if operand.operator == "atomic":
                    events.append(operand.operands[0])
                elif operand.operator == "eventually":
                    inner = operand.operands[0]
                    if inner.operator == "atomic":
                        events.append(inner.operands[0])
            return RewardMachine.sequential_tasks(events)

        elif formula.operator == "always":
            inner = formula.operands[0]
            if inner.operator == "atomic":
                prop = inner.operands[0]
                # "always p" = stay in state 0 as long as p holds
                # Any non-p event goes to fail state
                return RewardMachine(
                    states=[0, 1, 2],
                    initial_state=0,
                    transitions={
                        (0, prop): 0,  # p holds: stay
                        (0, f"not_{prop}"): 2,  # violation
                    },
                    rewards={
                        (0, 0): 0.1,  # Small reward for maintaining
                        (0, 2): -10.0,  # Penalty for violation
                    },
                    terminal_states={2},
                )

        elif formula.operator == "until":
            phi = formula.operands[0]
            psi = formula.operands[1]
            if phi.operator == "atomic" and psi.operator == "atomic":
                p = phi.operands[0]
                q = psi.operands[0]
                return RewardMachine(
                    states=[0, 1, 2],
                    initial_state=0,
                    transitions={
                        (0, p): 0,  # phi holds, psi not yet
                        (0, q): 1,  # psi achieved
                        (0, f"not_{p}"): 2,  # phi violated before psi
                    },
                    rewards={
                        (0, 1): 10.0,
                        (0, 2): -5.0,
                    },
                    terminal_states={1, 2},
                )

        # Fallback: trivial RM
        logger.warning(f"Unsupported LTL pattern: {formula.operator}")
        return RewardMachine(states=[0], initial_state=0)
