"""
POMDP Planning for Layer 2

Implements planning under partial observability:
- Belief state tracking (probability distributions over states)
- Bayesian belief updates after observations
- Information-gathering actions (active sensing)
- Value of information calculations
- Contingent planning (conditional plans)
- Belief space planning

Research:
- Kaelbling et al. (1998): "Planning and acting in partially observable domains"
- Cassandra et al. (1994): "Acting optimally in partially observable domains"
- Pineau et al. (2003): Point-based value iteration for POMDPs
- Silver & Veness (2010): Monte-Carlo planning in POMDPs
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple, Set
from enum import Enum
import logging
import numpy as np
from collections import defaultdict
from copy import deepcopy

# Import Layer 2 core
from HoloLoom.planning.planner import HierarchicalPlanner, Plan, Goal, Action, ActionType

logger = logging.getLogger(__name__)


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class BeliefState:
    """
    Probability distribution over possible states.

    Represents agent's belief about true world state under partial observability.
    """
    states: List[Dict[str, Any]]  # Possible states
    probabilities: np.ndarray     # P(state) for each state

    def __post_init__(self):
        """Validate and normalize probabilities."""
        # Normalize
        total = self.probabilities.sum()
        if total > 0:
            self.probabilities = self.probabilities / total
        else:
            # Uniform if all zero
            n = len(self.states)
            self.probabilities = np.ones(n) / n

    def entropy(self) -> float:
        """
        Calculate Shannon entropy of belief.

        Returns:
            Entropy in bits (0 = certain, high = uncertain)
        """
        # Avoid log(0)
        probs = self.probabilities[self.probabilities > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def most_likely_state(self) -> Dict[str, Any]:
        """Return most probable state."""
        idx = np.argmax(self.probabilities)
        return self.states[idx]

    def probability_of(self, state: Dict[str, Any]) -> float:
        """Get probability of specific state."""
        for i, s in enumerate(self.states):
            if s == state:
                return self.probabilities[i]
        return 0.0

    def __repr__(self):
        entropy = self.entropy()
        most_likely = self.most_likely_state()
        return f"BeliefState({len(self.states)} states, entropy={entropy:.2f})"


@dataclass
class ObservationAction:
    """Action that gathers information (active sensing)."""
    variable: str           # What variable to observe
    cost: float            # Cost of observation
    accuracy: float        # Observation accuracy (0-1)
    description: str = ""

    def __repr__(self):
        return f"Observe({self.variable}, accuracy={self.accuracy:.2f}, cost={self.cost:.2f})"


@dataclass
class ContingentPlan:
    """
    Conditional plan that branches based on observations.

    Tree structure:
    - Root action (could be observation or execution)
    - Branches for each possible observation
    - Recursive subplans
    """
    root_action: Action
    branches: Dict[str, 'ContingentPlan']  # observation_value -> subplan
    depth: int = 0
    expected_cost: float = 0.0
    is_terminal: bool = False  # True if leaf node (goal reached)

    def __repr__(self):
        n_branches = len(self.branches)
        return (f"ContingentPlan(action={self.root_action}, "
                f"{n_branches} branches, cost={self.expected_cost:.2f})")


# ============================================================================
# Observation Model
# ============================================================================

class ObservationModel:
    """
    Models P(observation | state, action).

    Represents sensor characteristics:
    - Accuracy: How often sensor is correct
    - Noise: Random errors
    - Bias: Systematic errors
    """

    def __init__(self, default_accuracy: float = 0.8):
        """
        Initialize observation model.

        Args:
            default_accuracy: Default observation accuracy (0-1)
        """
        self.default_accuracy = default_accuracy
        self.variable_accuracies: Dict[str, float] = {}

    def set_accuracy(self, variable: str, accuracy: float):
        """Set observation accuracy for specific variable."""
        self.variable_accuracies[variable] = accuracy
        logger.debug(f"Set observation accuracy for {variable}: {accuracy:.2f}")

    def get_accuracy(self, variable: str) -> float:
        """Get observation accuracy for variable."""
        return self.variable_accuracies.get(variable, self.default_accuracy)

    # ------------------------------------------------------------------------
    # Observation Sampling
    # ------------------------------------------------------------------------

    def sample_observation(self,
                          true_state: Dict,
                          variable: str,
                          rng: Optional[np.random.Generator] = None) -> Any:
        """
        Sample observation given true state.

        Args:
            true_state: Actual world state
            variable: Variable to observe
            rng: Random number generator

        Returns:
            Observed value (may be noisy)
        """
        if rng is None:
            rng = np.random.default_rng()

        true_value = true_state.get(variable)
        accuracy = self.get_accuracy(variable)

        # Bernoulli trial: accurate with probability = accuracy
        if rng.random() < accuracy:
            return true_value  # Correct observation
        else:
            # Noisy observation
            return self._generate_noise(true_value, rng)

    def _generate_noise(self, true_value: Any,
                       rng: np.random.Generator) -> Any:
        """Generate noisy observation."""
        # Simple noise model
        if isinstance(true_value, bool):
            return not true_value  # Flip boolean
        elif isinstance(true_value, int):
            return true_value + rng.choice([-1, 1])  # ±1
        elif isinstance(true_value, float):
            return true_value + rng.normal(0, 0.1)  # Gaussian noise
        else:
            return true_value  # Can't add noise

    # ------------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------------

    def likelihood(self,
                  observation: Any,
                  variable: str,
                  state: Dict) -> float:
        """
        Calculate P(observation | state, variable).

        Args:
            observation: Observed value
            variable: Observed variable
            state: Hypothesized state

        Returns:
            Probability of observing this value in this state
        """
        true_value = state.get(variable)
        accuracy = self.get_accuracy(variable)

        if observation == true_value:
            # Correct observation
            return accuracy
        else:
            # Incorrect observation (uniform over other values)
            # Simplified: assume binary or small discrete space
            n_values = 2  # Binary assumption
            return (1.0 - accuracy) / (n_values - 1)


# ============================================================================
# Belief Updates
# ============================================================================

class BeliefUpdater:
    """Updates belief states after observations using Bayes' rule."""

    def __init__(self, observation_model: ObservationModel):
        """Initialize belief updater."""
        self.observation_model = observation_model

    def update(self,
              prior_belief: BeliefState,
              observation: Any,
              variable: str) -> BeliefState:
        """
        Bayesian belief update after observation.

        Bayes' rule: P(state | obs) ∝ P(obs | state) × P(state)

        Args:
            prior_belief: Belief before observation
            observation: Observed value
            variable: Observed variable

        Returns:
            Updated belief (posterior)
        """
        # Calculate posterior probabilities
        posteriors = []

        for state, prior_prob in zip(prior_belief.states, prior_belief.probabilities):
            # P(obs | state)
            likelihood = self.observation_model.likelihood(observation, variable, state)

            # P(state | obs) ∝ P(obs | state) × P(state)
            posterior = likelihood * prior_prob
            posteriors.append(posterior)

        posteriors = np.array(posteriors)

        # Normalize
        total = posteriors.sum()
        if total > 0:
            posteriors = posteriors / total
        else:
            # No consistent states - uniform
            posteriors = np.ones(len(posteriors)) / len(posteriors)

        updated_belief = BeliefState(
            states=prior_belief.states,
            probabilities=posteriors
        )

        logger.debug(f"Belief update: entropy {prior_belief.entropy():.2f} → "
                    f"{updated_belief.entropy():.2f}")

        return updated_belief


# ============================================================================
# POMDP Planner
# ============================================================================

class POMDPPlanner:
    """Planner for partially observable domains."""

    def __init__(self,
                 base_planner: HierarchicalPlanner,
                 observation_model: ObservationModel,
                 belief_updater: Optional[BeliefUpdater] = None):
        """
        Initialize POMDP planner.

        Args:
            base_planner: Underlying planner (assumes full observability)
            observation_model: Observation model P(obs | state)
            belief_updater: Belief updater (created if not provided)
        """
        self.base_planner = base_planner
        self.observation_model = observation_model

        if belief_updater is None:
            belief_updater = BeliefUpdater(observation_model)
        self.belief_updater = belief_updater

        logger.info("Initialized POMDPPlanner")

    # ------------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------------

    def plan(self,
            goal: Goal,
            initial_belief: BeliefState,
            max_depth: int = 5,
            entropy_threshold: float = 1.0) -> Optional[ContingentPlan]:
        """
        Generate contingent plan under partial observability.

        Algorithm:
        1. Check if should gather information (high uncertainty)
        2. If yes, create observation action and branch
        3. Otherwise, act on most likely state
        4. Recursively plan for each branch

        Args:
            goal: Goal to achieve
            initial_belief: Initial belief state
            max_depth: Max planning depth (prevent infinite recursion)
            entropy_threshold: Gather info if entropy > threshold

        Returns:
            Contingent plan (tree structure)
        """
        logger.info(f"POMDP planning: entropy={initial_belief.entropy():.2f}, "
                   f"depth={max_depth}")

        # Base case: max depth reached
        if max_depth <= 0:
            logger.debug("Max depth reached")
            return None

        # Check if goal satisfied
        if self._check_goal_in_belief(goal, initial_belief):
            logger.info("Goal satisfied in belief")
            return ContingentPlan(
                root_action=Action(
                    action_type=ActionType.VERIFY,
                    parameters={'goal': goal.desired_state},
                    description="Verify goal"
                ),
                branches={},
                depth=0,
                expected_cost=0.0,
                is_terminal=True
            )

        # Decide: gather information or act?
        should_observe = self._should_gather_information(
            initial_belief, goal, entropy_threshold
        )

        if should_observe:
            # Information gathering
            return self._plan_with_observation(
                goal, initial_belief, max_depth, entropy_threshold
            )
        else:
            # Execute action
            return self._plan_with_action(
                goal, initial_belief, max_depth, entropy_threshold
            )

    # ------------------------------------------------------------------------
    # Information Gathering
    # ------------------------------------------------------------------------

    def _should_gather_information(self,
                                   belief: BeliefState,
                                   goal: Goal,
                                   entropy_threshold: float) -> bool:
        """
        Decide whether to gather information.

        Criteria:
        - High uncertainty (entropy > threshold)
        - Value of information > cost
        """
        # Simple heuristic: entropy-based
        return belief.entropy() > entropy_threshold

    def _plan_with_observation(self,
                              goal: Goal,
                              belief: BeliefState,
                              max_depth: int,
                              entropy_threshold: float) -> Optional[ContingentPlan]:
        """
        Create contingent plan with information gathering.

        Steps:
        1. Select variable to observe
        2. For each possible observation:
           a. Update belief
           b. Recursively plan
        3. Return contingent plan with branches
        """
        # Select observation variable
        variable = self._select_observation_variable(belief, goal)

        if not variable:
            logger.debug("No useful variable to observe")
            return self._plan_with_action(goal, belief, max_depth, entropy_threshold)

        logger.debug(f"Planning to observe: {variable}")

        # Create observation action
        obs_action = Action(
            action_type=ActionType.OBSERVE,
            parameters={'variable': variable},
            description=f"Observe {variable}"
        )

        # Generate branches for possible observations
        branches = {}
        possible_values = self._get_possible_values(belief, variable)

        for value in possible_values:
            # Simulate observation
            updated_belief = self.belief_updater.update(belief, value, variable)

            # Recursively plan for this branch
            subplan = self.plan(
                goal, updated_belief, max_depth - 1, entropy_threshold
            )

            if subplan:
                branches[str(value)] = subplan

        # Calculate expected cost
        obs_cost = 1.0  # Cost of observation
        expected_cost = obs_cost + self._calculate_expected_cost(branches, belief)

        contingent_plan = ContingentPlan(
            root_action=obs_action,
            branches=branches,
            depth=max_depth,
            expected_cost=expected_cost
        )

        logger.debug(f"Contingent plan: {len(branches)} branches, "
                    f"cost={expected_cost:.2f}")

        return contingent_plan

    def _plan_with_action(self,
                         goal: Goal,
                         belief: BeliefState,
                         max_depth: int,
                         entropy_threshold: float) -> Optional[ContingentPlan]:
        """
        Create plan based on most likely state.

        When uncertainty is low, act without gathering more information.
        """
        # Use most likely state for planning
        most_likely = belief.most_likely_state()

        logger.debug(f"Acting on most likely state: {most_likely}")

        # Plan using base planner
        plan = self.base_planner.plan(goal, most_likely)

        if not plan or not plan.actions:
            logger.warning("Base planner failed")
            return None

        # Take first action
        action = plan.actions[0]

        # Simple: no branching for execution actions
        # (In full POMDP, would model action stochasticity)
        contingent_plan = ContingentPlan(
            root_action=action,
            branches={},  # No branching
            depth=max_depth,
            expected_cost=plan.expected_cost
        )

        return contingent_plan

    # ------------------------------------------------------------------------
    # Value of Information
    # ------------------------------------------------------------------------

    def value_of_information(self,
                            belief: BeliefState,
                            variable: str,
                            goal: Goal) -> float:
        """
        Calculate expected value of observing variable.

        VOI = E[Value(belief after obs)] - Value(current belief)

        If VOI > cost, worth gathering info.

        Args:
            belief: Current belief
            variable: Variable to observe
            goal: Goal to achieve

        Returns:
            Expected value of information
        """
        # Current value (without observation)
        current_value = self._evaluate_belief(belief, goal)

        # Expected value after observation
        possible_values = self._get_possible_values(belief, variable)
        expected_value_after = 0.0

        for value in possible_values:
            # Probability of observing this value
            p_obs = self._probability_of_observation(belief, variable, value)

            # Belief after observation
            updated_belief = self.belief_updater.update(belief, value, variable)

            # Value in updated belief
            value_after = self._evaluate_belief(updated_belief, goal)

            expected_value_after += p_obs * value_after

        # VOI = expected improvement
        voi = expected_value_after - current_value

        logger.debug(f"VOI({variable}) = {voi:.3f}")

        return voi

    def _evaluate_belief(self, belief: BeliefState, goal: Goal) -> float:
        """
        Evaluate quality of belief for achieving goal.

        Simple heuristic: probability of goal being achieved.
        """
        prob_goal = 0.0

        for state, prob in zip(belief.states, belief.probabilities):
            if self._check_goal(goal, state):
                prob_goal += prob

        return prob_goal

    def _probability_of_observation(self,
                                    belief: BeliefState,
                                    variable: str,
                                    value: Any) -> float:
        """Calculate P(observation = value | belief)."""
        prob = 0.0

        for state, state_prob in zip(belief.states, belief.probabilities):
            likelihood = self.observation_model.likelihood(value, variable, state)
            prob += state_prob * likelihood

        return prob

    # ------------------------------------------------------------------------
    # Observation Selection
    # ------------------------------------------------------------------------

    def _select_observation_variable(self,
                                     belief: BeliefState,
                                     goal: Goal) -> Optional[str]:
        """
        Select best variable to observe.

        Criteria:
        - Maximizes VOI / cost ratio
        - Reduces entropy most
        - Helps achieve goal
        """
        # Get variables from states
        if not belief.states:
            return None

        variables = list(belief.states[0].keys())

        if not variables:
            return None

        # Calculate VOI for each variable
        best_var = None
        best_score = -float('inf')

        for var in variables:
            voi = self.value_of_information(belief, var, goal)
            cost = 1.0  # Assume uniform cost

            score = voi / cost

            if score > best_score:
                best_score = score
                best_var = var

        logger.debug(f"Selected variable: {best_var} (score={best_score:.3f})")

        return best_var

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def _get_possible_values(self, belief: BeliefState, variable: str) -> List[Any]:
        """Get possible values for variable from belief."""
        values = set()
        for state in belief.states:
            value = state.get(variable)
            if value is not None:
                values.add(value)
        return list(values)

    def _check_goal(self, goal: Goal, state: Dict) -> bool:
        """Check if goal satisfied in state."""
        for var, desired_value in goal.desired_state.items():
            if state.get(var) != desired_value:
                return False
        return True

    def _check_goal_in_belief(self, goal: Goal, belief: BeliefState) -> bool:
        """Check if goal definitely satisfied in belief."""
        # Conservative: only true if all probable states satisfy goal
        for state, prob in zip(belief.states, belief.probabilities):
            if prob > 0.1:  # Only check probable states
                if not self._check_goal(goal, state):
                    return False
        return True

    def _calculate_expected_cost(self,
                                branches: Dict[str, ContingentPlan],
                                belief: BeliefState) -> float:
        """Calculate expected cost of contingent plan."""
        # Simplified: average of branch costs
        if not branches:
            return 0.0

        total_cost = sum(plan.expected_cost for plan in branches.values())
        return total_cost / len(branches)

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    def execute_contingent_plan(self,
                               plan: ContingentPlan,
                               belief: BeliefState,
                               executor: Callable,
                               observer: Callable) -> Tuple[bool, BeliefState]:
        """
        Execute contingent plan with observation and action.

        Args:
            plan: Contingent plan to execute
            belief: Current belief
            executor: Function to execute actions
            observer: Function to make observations

        Returns:
            (success, updated_belief)
        """
        # Execute root action
        if plan.root_action.action_type == ActionType.OBSERVE:
            # Observation action
            variable = plan.root_action.parameters['variable']

            # Make observation
            observation = observer(variable)

            # Update belief
            updated_belief = self.belief_updater.update(belief, observation, variable)

            # Follow branch
            branch_key = str(observation)
            if branch_key in plan.branches:
                subplan = plan.branches[branch_key]
                return self.execute_contingent_plan(
                    subplan, updated_belief, executor, observer
                )
            else:
                logger.warning(f"No branch for observation: {observation}")
                return False, updated_belief

        else:
            # Execution action
            success = executor(plan.root_action)

            # No belief update for execution (simplified)
            # Full POMDP would model action stochasticity

            return success, belief


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'BeliefState',
    'ObservationAction',
    'ContingentPlan',
    'ObservationModel',
    'BeliefUpdater',
    'POMDPPlanner',
]
