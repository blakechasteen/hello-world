"""
Hierarchical Task Network (HTN) Planner with Causal Reasoning

Neurosymbolic Planning:
- Symbolic: HTN rules + causal DAG
- Neural: Can integrate learned action models (future)

Key Innovation: Uses causal knowledge to guide planning!

Example:
    Goal: recovery=1
    Causal knowledge: treatment → recovery
    Plan: [Action(intervene, treatment=1)]
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum

# Import from Layer 1 (causal reasoning)
import sys
sys.path.insert(0, '..')
from HoloLoom.causal import CausalDAG

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions in plans."""
    INTERVENE = "intervene"      # Causal intervention
    OBSERVE = "observe"          # Gather information
    WAIT = "wait"                # Temporal delay
    VERIFY = "verify"            # Check condition
    COMPOSITE = "composite"      # Contains sub-actions


@dataclass
class Action:
    """
    Executable action in a plan.

    Examples:
        Action(INTERVENE, {"treatment": 1})
        Action(WAIT, {"duration": 5})
        Action(VERIFY, {"variable": "recovery", "value": 1})
    """
    action_type: ActionType
    parameters: Dict[str, Any]
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    description: str = ""

    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"Action({self.action_type.value}, {params_str})"


@dataclass
class Plan:
    """
    Sequence of actions to achieve goal.

    Attributes:
        actions: Ordered list of actions
        goal: Original goal
        expected_cost: Estimated cost
        causal_chain: Causal reasoning used
    """
    actions: List[Action]
    goal: 'Goal'
    expected_cost: float = 0.0
    causal_chain: List[str] = field(default_factory=list)
    explanation: str = ""

    def __len__(self):
        return len(self.actions)

    def __repr__(self):
        return f"Plan({len(self.actions)} actions, cost={self.expected_cost:.1f})"


@dataclass
class Goal:
    """
    Desired state to achieve.

    Examples:
        Goal({"recovery": 1}, priority=1.0)
        Goal({"age": 30, "recovery": 1}, priority=0.8)
    """
    desired_state: Dict[str, Any]
    priority: float = 1.0
    deadline: Optional[int] = None  # Time steps
    description: str = ""

    def is_satisfied(self, current_state: Dict[str, Any]) -> bool:
        """Check if goal is satisfied in current state."""
        for var, value in self.desired_state.items():
            if current_state.get(var) != value:
                return False
        return True

    def __repr__(self):
        state_str = ", ".join(f"{k}={v}" for k, v in self.desired_state.items())
        return f"Goal({state_str})"


class HierarchicalPlanner:
    """
    HTN Planner with Causal Reasoning.

    Neurosymbolic Architecture:
    - Symbolic: HTN decomposition rules + causal DAG
    - Neural: Future - learned action models, value functions

    Usage:
        planner = HierarchicalPlanner(causal_dag)

        plan = planner.plan(
            goal=Goal({"recovery": 1}),
            current_state={"recovery": 0, "treatment": 0}
        )

        # Execute plan
        for action in plan.actions:
            execute(action)
    """

    def __init__(self, causal_dag: CausalDAG, max_depth: int = 5):
        """
        Initialize planner.

        Args:
            causal_dag: Causal knowledge from Layer 1
            max_depth: Maximum decomposition depth
        """
        self.dag = causal_dag
        self.max_depth = max_depth

        logger.info(f"Initialized planner with {len(causal_dag.nodes)} causal variables")

    def plan(
        self,
        goal: Goal,
        current_state: Dict[str, Any],
        allow_temporal: bool = True
    ) -> Optional[Plan]:
        """
        Generate plan to achieve goal.

        Strategy:
        1. Check if goal already satisfied
        2. Find causal chain from current → goal
        3. Decompose into actions
        4. Add verification steps

        Args:
            goal: Desired state
            current_state: Current values
            allow_temporal: Allow temporal planning

        Returns:
            Plan or None if impossible
        """
        logger.info(f"Planning for goal: {goal}")
        logger.info(f"Current state: {current_state}")

        # Check if already satisfied
        if goal.is_satisfied(current_state):
            logger.info("Goal already satisfied!")
            return Plan(
                actions=[],
                goal=goal,
                expected_cost=0,
                explanation="Goal already achieved"
            )

        # Find causal chain to goal
        causal_chain = self._find_causal_chain(goal, current_state)

        if not causal_chain:
            logger.warning("No causal chain found to goal")
            return None

        logger.info(f"Found causal chain: {' → '.join(causal_chain)}")

        # Decompose into actions
        actions = self._decompose_to_actions(causal_chain, goal, current_state)

        # Calculate cost
        total_cost = sum(a.cost for a in actions)

        # Build explanation
        explanation = self._generate_explanation(causal_chain, actions)

        plan = Plan(
            actions=actions,
            goal=goal,
            expected_cost=total_cost,
            causal_chain=causal_chain,
            explanation=explanation
        )

        logger.info(f"Generated plan: {plan}")
        return plan

    def _find_causal_chain(
        self,
        goal: Goal,
        current_state: Dict[str, Any]
    ) -> Optional[List[str]]:
        """
        Find causal chain from current state to goal.

        Uses causal DAG to find path:
        current variables → intermediate → goal variables

        Returns:
            List of variables in causal order, or None
        """
        # Identify what needs to change
        target_vars = list(goal.desired_state.keys())

        # Find what we can control (current state variables)
        controllable = [v for v in current_state.keys() if v in self.dag.nodes]

        if not target_vars or not controllable:
            return None

        # For each target variable, find causal parents
        chain = []
        visited = set()

        for target in target_vars:
            # Find path from controllable variables to target
            path = self._find_shortest_causal_path(controllable, target, visited)

            if path:
                # Add to chain (avoiding duplicates)
                for node in path:
                    if node not in chain:
                        chain.append(node)
                visited.add(target)

        return chain if chain else None

    def _find_shortest_causal_path(
        self,
        sources: List[str],
        target: str,
        visited: Set[str]
    ) -> Optional[List[str]]:
        """
        Find shortest directed path from any source to target.

        Uses breadth-first search on causal DAG.
        """
        if target in visited:
            return None

        # BFS from target backwards (find causes)
        queue = [(target, [target])]
        visited_in_search = {target}

        while queue:
            node, path = queue.pop(0)

            # Check if we reached a controllable source
            if node in sources:
                return list(reversed(path))

            # Explore parents (causes)
            parents = self.dag.parents(node)

            for parent in parents:
                if parent not in visited_in_search:
                    visited_in_search.add(parent)
                    queue.append((parent, path + [parent]))

        # No path found
        return None

    def _decompose_to_actions(
        self,
        causal_chain: List[str],
        goal: Goal,
        current_state: Dict[str, Any]
    ) -> List[Action]:
        """
        Decompose causal chain into executable actions.

        Strategy:
        1. For each variable in chain, create intervention if needed
        2. Add verification steps
        3. Order by causal dependencies
        """
        actions = []

        # Process chain in order
        for i, var in enumerate(causal_chain):
            # Check if this variable needs to be set
            if var in goal.desired_state:
                desired_value = goal.desired_state[var]
                current_value = current_state.get(var)

                if current_value != desired_value:
                    # Need to intervene
                    action = Action(
                        action_type=ActionType.INTERVENE,
                        parameters={"variable": var, "value": desired_value},
                        effects={var: desired_value},
                        cost=1.0,
                        description=f"Set {var} to {desired_value}"
                    )
                    actions.append(action)

            # Check if intermediate variable (not goal, but on path)
            elif i < len(causal_chain) - 1:
                # Intermediate variable - may need to set it
                # Find what value would help achieve goal
                children = self.dag.children(var)

                # If this variable causes next in chain, intervene
                next_var = causal_chain[i + 1]
                if next_var in children:
                    # Simple heuristic: set to 1 (active)
                    action = Action(
                        action_type=ActionType.INTERVENE,
                        parameters={"variable": var, "value": 1},
                        effects={var: 1},
                        cost=1.0,
                        description=f"Activate {var} (causes {next_var})"
                    )
                    actions.append(action)

        # Add verification at end
        verify_action = Action(
            action_type=ActionType.VERIFY,
            parameters={"goal": goal.desired_state},
            cost=0.1,
            description=f"Verify goal achieved: {goal}"
        )
        actions.append(verify_action)

        return actions

    def _generate_explanation(
        self,
        causal_chain: List[str],
        actions: List[Action]
    ) -> str:
        """Generate human-readable explanation of plan."""
        lines = []

        lines.append("Plan Reasoning:")
        lines.append(f"  Causal chain: {' → '.join(causal_chain)}")
        lines.append("")
        lines.append("Actions:")
        for i, action in enumerate(actions, 1):
            lines.append(f"  {i}. {action.description or action}")

        return "\n".join(lines)

    def replan(
        self,
        original_plan: Plan,
        current_state: Dict[str, Any],
        failure: Optional[str] = None
    ) -> Optional[Plan]:
        """
        Replan after execution failure.

        Args:
            original_plan: Failed plan
            current_state: Current state after failure
            failure: Description of what failed

        Returns:
            New plan or None
        """
        logger.warning(f"Replanning after failure: {failure}")

        # Try to plan again from current state
        new_plan = self.plan(original_plan.goal, current_state)

        if new_plan:
            logger.info("Generated recovery plan")
        else:
            logger.error("Cannot recover from failure")

        return new_plan

    def __repr__(self):
        return f"HierarchicalPlanner({len(self.dag.nodes)} causal vars)"
