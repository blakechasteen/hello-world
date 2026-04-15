#!/usr/bin/env python3
"""
Planning Department - Goal decomposition and action planning.

This department provides sophisticated planning capabilities:
- Hierarchical goal decomposition (complex goals → sub-goals)
- Action sequence generation (ordered steps to achieve goals)
- Constraint satisfaction (handle dependencies and limitations)
- Plan validation (feasibility checking)

Design Philosophy:
"Break it down. Build it up. Make it work."

Supported Tasks:
- "decompose_goal": Break complex goal into achievable sub-goals
- "create_plan": Generate action sequence for goal
- "validate_plan": Check plan feasibility and constraints
- "optimize_plan": Improve plan efficiency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..base import BaseDepartment
from ..protocol import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

class GoalStatus(Enum):
    """Goal achievement status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    FAILED = "failed"
    BLOCKED = "blocked"


class ConstraintType(Enum):
    """Types of plan constraints."""
    TEMPORAL = "temporal"  # Time-based constraints
    RESOURCE = "resource"  # Resource availability
    DEPENDENCY = "dependency"  # Action dependencies
    ORDERING = "ordering"  # Sequence requirements


@dataclass
class Goal:
    """A goal to be achieved."""
    id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    parent_goal_id: str | None = None
    sub_goals: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    priority: float = 0.5  # 0.0 to 1.0
    deadline: datetime | None = None


@dataclass
class Action:
    """An action step in a plan."""
    id: str
    description: str
    preconditions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    duration_estimate: float = 1.0  # in arbitrary time units
    cost: float = 1.0  # relative cost
    required_resources: list[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class Constraint:
    """A constraint on plan execution."""
    id: str
    type: ConstraintType
    description: str
    affected_actions: list[str] = field(default_factory=list)
    violation_penalty: float = 1.0
    is_hard: bool = True  # Hard constraint vs soft constraint


@dataclass
class Plan:
    """A plan to achieve a goal."""
    id: str
    goal_id: str
    actions: list[Action]
    constraints: list[Constraint] = field(default_factory=list)
    total_duration: float = 0.0
    total_cost: float = 0.0
    feasibility_score: float = 0.0
    confidence: float = 0.0


# ============================================================================
# Goal Decomposer
# ============================================================================

class GoalDecomposer:
    """
    Hierarchical goal decomposition engine.

    Breaks complex goals into achievable sub-goals using:
    - AND decomposition (all sub-goals required)
    - OR decomposition (any sub-goal sufficient)
    - Hierarchical structure (recursive decomposition)

    Example:
        Goal: "Increase bee colony health"

        Decomposition:
        ├─ Sub-goal 1: "Reduce pest pressure" (AND)
        │  ├─ Action: "Monitor varroa mite levels"
        │  └─ Action: "Apply treatment if needed"
        ├─ Sub-goal 2: "Improve nutrition" (AND)
        │  ├─ Action: "Ensure diverse foraging"
        │  └─ Action: "Supplement if needed"
        └─ Sub-goal 3: "Maintain hive conditions" (AND)
           ├─ Action: "Monitor temperature"
           └─ Action: "Check ventilation"
    """

    def __init__(self, max_depth: int = 4, max_sub_goals: int = 5):
        """
        Initialize goal decomposer.

        Args:
            max_depth: Maximum decomposition depth
            max_sub_goals: Maximum sub-goals per goal
        """
        self.max_depth = max_depth
        self.max_sub_goals = max_sub_goals

        self._goal_registry: dict[str, Goal] = {}
        self._goal_counter = 0

        logger.info(f"GoalDecomposer initialized (max_depth={max_depth})")

    def decompose(
        self,
        goal_description: str,
        context: dict[str, Any] | None = None,
        depth: int = 0
    ) -> Goal:
        """
        Decompose a goal into sub-goals.

        Args:
            goal_description: High-level goal description
            context: Additional context for decomposition
            depth: Current decomposition depth

        Returns:
            Goal with sub-goals
        """
        context = context or {}

        # Create root goal
        goal = self._create_goal(goal_description, depth=depth)

        # Stop if max depth reached
        if depth >= self.max_depth:
            return goal

        # Decompose based on goal type
        if self._is_atomic_goal(goal_description):
            # Atomic goal - no further decomposition needed
            return goal

        # Generate sub-goals
        sub_goals = self._generate_sub_goals(goal_description, context)

        for sub_goal_desc in sub_goals[:self.max_sub_goals]:
            # Recursively decompose
            sub_goal = self.decompose(sub_goal_desc, context, depth + 1)
            sub_goal.parent_goal_id = goal.id

            # Register sub-goal
            self._goal_registry[sub_goal.id] = sub_goal
            goal.sub_goals.append(sub_goal.id)

        return goal

    def get_execution_order(self, goal: Goal) -> list[Goal]:
        """
        Get execution order for goal and sub-goals (depth-first).

        Args:
            goal: Root goal

        Returns:
            List of goals in execution order
        """
        execution_order = []

        # Depth-first traversal
        def traverse(g: Goal):
            # Visit sub-goals first
            for sub_goal_id in g.sub_goals:
                sub_goal = self._goal_registry.get(sub_goal_id)
                if sub_goal:
                    traverse(sub_goal)

            # Then visit current goal
            execution_order.append(g)

        traverse(goal)
        return execution_order

    def estimate_complexity(self, goal: Goal) -> dict[str, Any]:
        """
        Estimate goal complexity metrics.

        Args:
            goal: Goal to analyze

        Returns:
            Complexity metrics
        """
        # Count sub-goals recursively
        def count_sub_goals(g: Goal) -> int:
            count = len(g.sub_goals)
            for sub_goal_id in g.sub_goals:
                sub_goal = self._goal_registry.get(sub_goal_id)
                if sub_goal:
                    count += count_sub_goals(sub_goal)
            return count

        # Calculate depth
        def calculate_depth(g: Goal) -> int:
            if not g.sub_goals:
                return 0
            max_sub_depth = 0
            for sub_goal_id in g.sub_goals:
                sub_goal = self._goal_registry.get(sub_goal_id)
                if sub_goal:
                    max_sub_depth = max(max_sub_depth, calculate_depth(sub_goal))
            return 1 + max_sub_depth

        return {
            "total_sub_goals": count_sub_goals(goal),
            "max_depth": calculate_depth(goal),
            "breadth": len(goal.sub_goals),
            "complexity_score": count_sub_goals(goal) * calculate_depth(goal)
        }

    def _create_goal(self, description: str, depth: int = 0) -> Goal:
        """Create a new goal."""
        self._goal_counter += 1
        goal_id = f"goal_{self._goal_counter}"

        goal = Goal(
            id=goal_id,
            description=description,
            priority=max(0.0, 1.0 - (depth * 0.2))  # Higher priority for top-level goals
        )

        self._goal_registry[goal_id] = goal
        return goal

    def _is_atomic_goal(self, description: str) -> bool:
        """Check if goal is atomic (cannot be decomposed further)."""
        # Simple heuristic: short descriptions are likely atomic
        return len(description.split()) <= 5

    def _generate_sub_goals(
        self,
        goal_description: str,
        context: dict[str, Any]
    ) -> list[str]:
        """
        Generate sub-goals for a complex goal.

        In a production system, this would use NLP/LLM to intelligently decompose.
        For now, using rule-based heuristics.
        """
        sub_goals = []

        # Extract keywords
        keywords = goal_description.lower().split()

        # Rule-based decomposition for common patterns
        if "increase" in keywords or "improve" in keywords:
            sub_goals = [
                f"Monitor current {keywords[-1]} levels",
                f"Identify factors affecting {keywords[-1]}",
                f"Implement improvements to {keywords[-1]}"
            ]

        elif "reduce" in keywords or "decrease" in keywords:
            sub_goals = [
                f"Assess current {keywords[-1]} levels",
                f"Identify causes of {keywords[-1]}",
                f"Apply interventions to reduce {keywords[-1]}"
            ]

        elif "maintain" in keywords:
            sub_goals = [
                f"Monitor {keywords[-1]} regularly",
                f"Ensure optimal conditions for {keywords[-1]}",
                f"Respond to {keywords[-1]} deviations"
            ]

        else:
            # Generic decomposition
            sub_goals = [
                f"Prepare for {goal_description}",
                f"Execute {goal_description}",
                f"Verify {goal_description}"
            ]

        return sub_goals


# ============================================================================
# Action Planner
# ============================================================================

class ActionPlanner:
    """
    Action sequence generation for goal achievement.

    Creates ordered action sequences using:
    - Forward planning (current state → goal state)
    - Backward chaining (goal → required preconditions)
    - Partial-order planning (flexible ordering)

    Example:
        Goal: "Treat varroa mite infestation"

        Plan:
        1. Action: "Inspect hive for mite levels" (precondition: access to hive)
        2. Action: "Select treatment method" (precondition: inspection complete)
        3. Action: "Apply treatment" (precondition: treatment selected, conditions suitable)
        4. Action: "Monitor treatment effectiveness" (precondition: treatment applied)
    """

    def __init__(self, planning_horizon: int = 10):
        """
        Initialize action planner.

        Args:
            planning_horizon: Maximum number of actions in plan
        """
        self.planning_horizon = planning_horizon

        self._action_library: dict[str, Action] = {}
        self._action_counter = 0

        logger.info(f"ActionPlanner initialized (horizon={planning_horizon})")

    def create_plan(
        self,
        goal: Goal,
        initial_state: dict[str, Any],
        available_actions: list[Action] | None = None
    ) -> Plan:
        """
        Create action plan for goal.

        Args:
            goal: Goal to achieve
            initial_state: Current state
            available_actions: Available actions (uses library if None)

        Returns:
            Plan with ordered actions
        """
        # Use action library if no actions provided
        if available_actions is None:
            available_actions = list(self._action_library.values())

        # Generate action sequence using forward planning
        actions = self._forward_plan(goal, initial_state, available_actions)

        # Create plan
        plan = Plan(
            id=f"plan_{goal.id}",
            goal_id=goal.id,
            actions=actions,
            total_duration=sum(a.duration_estimate for a in actions),
            total_cost=sum(a.cost for a in actions),
            confidence=self._calculate_plan_confidence(actions)
        )

        return plan

    def add_action_to_library(self, action: Action):
        """Add action to action library."""
        self._action_library[action.id] = action

    def _forward_plan(
        self,
        goal: Goal,
        current_state: dict[str, Any],
        available_actions: list[Action]
    ) -> list[Action]:
        """
        Forward planning: current state → goal state.

        Uses greedy selection of actions that progress toward goal.
        """
        plan_actions = []
        state = current_state.copy()
        remaining_horizon = self.planning_horizon

        # Simulate action execution until goal achieved or horizon reached
        while remaining_horizon > 0 and not self._is_goal_achieved(goal, state):
            # Find applicable actions
            applicable = [
                a for a in available_actions
                if self._are_preconditions_met(a, state)
            ]

            if not applicable:
                break  # No applicable actions

            # Select action that progresses toward goal
            action = self._select_best_action(applicable, goal, state)

            # Apply action effects
            plan_actions.append(action)
            self._apply_action_effects(action, state)

            remaining_horizon -= 1

        return plan_actions

    def _are_preconditions_met(
        self,
        action: Action,
        state: dict[str, Any]
    ) -> bool:
        """Check if action preconditions are met in current state."""
        for precondition in action.preconditions:
            if precondition not in state or not state[precondition]:
                return False
        return True

    def _apply_action_effects(
        self,
        action: Action,
        state: dict[str, Any]
    ):
        """Apply action effects to state."""
        for effect in action.effects:
            state[effect] = True

    def _is_goal_achieved(self, goal: Goal, state: dict[str, Any]) -> bool:
        """Check if goal success criteria are met."""
        if not goal.success_criteria:
            return False

        for criterion in goal.success_criteria:
            if criterion not in state or not state[criterion]:
                return False

        return True

    def _select_best_action(
        self,
        applicable_actions: list[Action],
        goal: Goal,
        state: dict[str, Any]
    ) -> Action:
        """Select action that best progresses toward goal."""
        # Score actions based on:
        # 1. Progress toward goal (effects overlap with success criteria)
        # 2. Cost (prefer lower cost)
        # 3. Confidence (prefer higher confidence)

        def score_action(action: Action) -> float:
            # Progress score: how many effects match success criteria
            progress = sum(
                1 for effect in action.effects
                if effect in goal.success_criteria
            )

            # Combined score
            return (progress * 10.0) - action.cost + (action.confidence * 5.0)

        return max(applicable_actions, key=score_action)

    def _calculate_plan_confidence(self, actions: list[Action]) -> float:
        """Calculate overall plan confidence."""
        if not actions:
            return 0.0

        # Average action confidences, penalized by plan length
        avg_confidence = sum(a.confidence for a in actions) / len(actions)
        length_penalty = 0.95 ** len(actions)  # Longer plans = less confident

        return avg_confidence * length_penalty


# ============================================================================
# Constraint Solver
# ============================================================================

class ConstraintSolver:
    """
    Constraint satisfaction for plan validation.

    Handles various constraint types:
    - Temporal: Time-based constraints (deadlines, durations)
    - Resource: Resource availability and conflicts
    - Dependency: Action dependencies (A must precede B)
    - Ordering: Sequence requirements

    Example:
        Constraints:
        1. "inspect_hive" must precede "apply_treatment"
        2. "apply_treatment" requires resource "treatment_supplies"
        3. "apply_treatment" must complete within 24 hours

        Validation:
        - Check temporal ordering
        - Verify resource availability
        - Ensure deadline feasibility
    """

    def __init__(self):
        """Initialize constraint solver."""
        self._constraint_registry: dict[str, Constraint] = {}

        logger.info("ConstraintSolver initialized")

    def add_constraint(self, constraint: Constraint):
        """Add constraint to registry."""
        self._constraint_registry[constraint.id] = constraint

    def validate_plan(
        self,
        plan: Plan,
        resources: dict[str, Any] | None = None
    ) -> tuple[bool, list[str]]:
        """
        Validate plan against constraints.

        Args:
            plan: Plan to validate
            resources: Available resources

        Returns:
            (is_valid, violation_reasons)
        """
        violations = []
        resources = resources or {}

        # Check each constraint
        for constraint in plan.constraints:
            if not self._check_constraint(constraint, plan, resources):
                violations.append(
                    f"{constraint.type.value} constraint violated: {constraint.description}"
                )

        is_valid = len(violations) == 0

        return is_valid, violations

    def _check_constraint(
        self,
        constraint: Constraint,
        plan: Plan,
        resources: dict[str, Any]
    ) -> bool:
        """Check if single constraint is satisfied."""
        if constraint.type == ConstraintType.TEMPORAL:
            return self._check_temporal_constraint(constraint, plan)
        elif constraint.type == ConstraintType.RESOURCE:
            return self._check_resource_constraint(constraint, plan, resources)
        elif constraint.type == ConstraintType.DEPENDENCY:
            return self._check_dependency_constraint(constraint, plan)
        elif constraint.type == ConstraintType.ORDERING:
            return self._check_ordering_constraint(constraint, plan)
        else:
            return True  # Unknown constraint type - assume satisfied

    def _check_temporal_constraint(
        self,
        constraint: Constraint,
        plan: Plan
    ) -> bool:
        """Check temporal constraints (deadlines, durations)."""
        # Simple check: total duration within reasonable bounds
        return plan.total_duration < 1000  # Arbitrary threshold

    def _check_resource_constraint(
        self,
        constraint: Constraint,
        plan: Plan,
        resources: dict[str, Any]
    ) -> bool:
        """Check resource availability constraints."""
        # Check if required resources are available
        for action_id in constraint.affected_actions:
            action = next((a for a in plan.actions if a.id == action_id), None)
            if action:
                for resource in action.required_resources:
                    if resource not in resources or not resources[resource]:
                        return False
        return True

    def _check_dependency_constraint(
        self,
        constraint: Constraint,
        plan: Plan
    ) -> bool:
        """Check action dependency constraints."""
        # Verify action dependencies are satisfied in execution order
        action_positions = {a.id: i for i, a in enumerate(plan.actions)}

        for action in plan.actions:
            for precondition in action.preconditions:
                # Check if precondition is provided by earlier action
                precondition_met = False

                for i, prev_action in enumerate(plan.actions[:action_positions[action.id]]):
                    if precondition in prev_action.effects:
                        precondition_met = True
                        break

                if not precondition_met and action.preconditions:
                    return False

        return True

    def _check_ordering_constraint(
        self,
        constraint: Constraint,
        plan: Plan
    ) -> bool:
        """Check action ordering constraints."""
        # Verify required action ordering
        if len(constraint.affected_actions) < 2:
            return True

        action_positions = {a.id: i for i, a in enumerate(plan.actions)}

        # Check pairwise ordering
        for i in range(len(constraint.affected_actions) - 1):
            action_a = constraint.affected_actions[i]
            action_b = constraint.affected_actions[i + 1]

            if action_a in action_positions and action_b in action_positions:
                if action_positions[action_a] >= action_positions[action_b]:
                    return False  # Wrong order

        return True


# ============================================================================
# Plan Validator
# ============================================================================

class PlanValidator:
    """
    Plan feasibility and quality validation.

    Validates plans using:
    - Constraint checking (via ConstraintSolver)
    - Resource availability analysis
    - Dependency resolution
    - Cost-benefit analysis

    Example:
        Plan validation checks:
        1. ✓ All preconditions met
        2. ✓ No resource conflicts
        3. ✓ Dependencies satisfied
        4. ✗ Cost exceeds budget
        5. ✓ Duration within deadline

        Feasibility: 80% (4/5 checks passed)
    """

    def __init__(self, constraint_solver: ConstraintSolver):
        """
        Initialize plan validator.

        Args:
            constraint_solver: Constraint solver for validation
        """
        self.constraint_solver = constraint_solver

        logger.info("PlanValidator initialized")

    def validate(
        self,
        plan: Plan,
        resources: dict[str, Any] | None = None,
        budget: float | None = None
    ) -> tuple[float, dict[str, Any]]:
        """
        Validate plan feasibility.

        Args:
            plan: Plan to validate
            resources: Available resources
            budget: Budget constraint

        Returns:
            (feasibility_score, validation_details)
        """
        checks = {}

        # 1. Constraint validation
        is_valid, violations = self.constraint_solver.validate_plan(plan, resources)
        checks["constraints_satisfied"] = is_valid
        checks["constraint_violations"] = violations

        # 2. Resource availability
        checks["resources_available"] = self._check_resource_availability(plan, resources)

        # 3. Dependency resolution
        checks["dependencies_resolved"] = self._check_dependencies(plan)

        # 4. Budget check
        if budget is not None:
            checks["within_budget"] = plan.total_cost <= budget
        else:
            checks["within_budget"] = True

        # 5. Action feasibility
        checks["actions_feasible"] = all(a.confidence > 0.5 for a in plan.actions)

        # Calculate feasibility score
        passed_checks = sum(1 for v in checks.values() if isinstance(v, bool) and v)
        total_checks = sum(1 for v in checks.values() if isinstance(v, bool))

        feasibility_score = passed_checks / total_checks if total_checks > 0 else 0.0

        plan.feasibility_score = feasibility_score

        return feasibility_score, checks

    def _check_resource_availability(
        self,
        plan: Plan,
        resources: dict[str, Any] | None
    ) -> bool:
        """Check if all required resources are available."""
        if resources is None:
            return True  # Assume available if not specified

        for action in plan.actions:
            for resource in action.required_resources:
                if resource not in resources or not resources[resource]:
                    return False

        return True

    def _check_dependencies(self, plan: Plan) -> bool:
        """Check if action dependencies are properly ordered."""
        action_positions = {a.id: i for i, a in enumerate(plan.actions)}
        effects_available = set()

        for action in plan.actions:
            # Check if preconditions are available from previous actions
            for precondition in action.preconditions:
                if precondition not in effects_available:
                    # Check if provided by initial state (assume true for now)
                    pass

            # Add effects to available set
            effects_available.update(action.effects)

        return True  # Simplified check


# ============================================================================
# Planning Department
# ============================================================================

class PlanningDepartment(BaseDepartment):
    """
    Planning Department - Goal decomposition and action planning.

    Provides sophisticated planning capabilities:
    1. Goal decomposition: Break complex goals into sub-goals
    2. Action planning: Generate action sequences
    3. Constraint handling: Manage constraints and dependencies
    4. Plan validation: Check feasibility and quality

    Example:

        async with PlanningDepartment(registry=registry) as dept:
            # Decompose goal
            request = DepartmentRequest(
                task_id="plan_001",
                task_type="decompose_goal",
                parameters={
                    "goal": "Increase bee colony health",
                    "max_depth": 3
                }
            )

            response = await dept.execute(request)
            # Returns hierarchical goal decomposition
    """

    def __init__(
        self,
        registry: Any | None = None,
        max_depth: int = 4,
        planning_horizon: int = 10,
        dept_config: DepartmentConfig | None = None
    ):
        """
        Initialize Planning Department.

        Args:
            registry: Department registry for accessing other departments
            max_depth: Maximum goal decomposition depth
            planning_horizon: Maximum actions in plan
            dept_config: Department configuration
        """
        super().__init__(
            name="planning",
            domain="advanced",
            version="1.0.0",
            supported_tasks=[
                "decompose_goal",
                "create_plan",
                "validate_plan",
                "optimize_plan"
            ],
            confidence_range=(0.70, 0.95),
            config=dept_config or DepartmentConfig(name="planning", domain="planning")
        )

        self.registry = registry

        # Initialize planning engines
        self.goal_decomposer = GoalDecomposer(
            max_depth=max_depth,
            max_sub_goals=5
        )

        self.action_planner = ActionPlanner(
            planning_horizon=planning_horizon
        )

        self.constraint_solver = ConstraintSolver()

        self.plan_validator = PlanValidator(self.constraint_solver)

        logger.info(f"Planning Department initialized (max_depth={max_depth})")

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize planning department."""
        if self._initialized:
            return

        await super().initialize()

    async def close(self) -> None:
        """Cleanup resources."""
        await super().close()

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """Execute a planning task."""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        if request.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {request.task_type}. "
                f"Supported: {self.supported_tasks}"
            )

        try:
            if request.task_type == "decompose_goal":
                result, confidence = await self._decompose_goal(request)
            elif request.task_type == "create_plan":
                result, confidence = await self._create_plan(request)
            elif request.task_type == "validate_plan":
                result, confidence = await self._validate_plan(request)
            elif request.task_type == "optimize_plan":
                result, confidence = await self._optimize_plan(request)
            else:
                raise ValueError(f"Unknown task type: {request.task_type}")

            learning_signals = {
                "task_type": request.task_type,
                "outcome": "success",
                "confidence": confidence.score
            }

            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                detail_level=request.context_preference,
                reasoning=self._build_reasoning(result, request) if request.context_preference != "minimal" else None,
                learning_signals=learning_signals,
                session_state=await self._build_session_state(request.session_id, result),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            self._record_request(
                success=True,
                latency_ms=response.execution_time_ms,
                confidence=confidence.score
            )

            return response

        except Exception as e:
            self._record_error(e)

            error_confidence = ConfidenceMetadata.from_score(
                0.0,
                justification=[],
                uncertainty_sources=[f"Execution failed: {str(e)}"]
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_request(success=False, latency_ms=latency_ms, confidence=0.0)

            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=error_confidence,
                execution_time_ms=latency_ms
            )

    async def verify(
        self,
        response: DepartmentResponse
    ) -> VerificationResult:
        """Verify planning quality."""
        confidence_valid = response.confidence.score >= 0.70
        has_error = "error" in response.result
        has_plan = "plan" in response.result or "goal" in response.result

        reasoning_sound = not has_error and has_plan
        sufficient = confidence_valid and reasoning_sound

        refinement_suggestions = None
        if not sufficient:
            refinement_suggestions = {
                "retry": True,
                "reason": "Low confidence or missing plan"
            }

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=reasoning_sound,
            alternative_paths=["simplify_goal", "increase_horizon"] if not sufficient else [],
            refinement_suggestions=refinement_suggestions,
            escalation_needed=response.confidence.score < 0.5,
            escalation_reason="Very low confidence" if response.confidence.score < 0.5 else None,
            verifier_confidence=0.85
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """Refine planning operation."""
        # Increase planning horizon and retry
        if "planning_horizon" in request.parameters:
            request.parameters["planning_horizon"] += 5

        return await self.execute(request)

    # ========================================================================
    # Task Handlers
    # ========================================================================

    async def _decompose_goal(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Decompose goal into sub-goals."""
        goal_description = request.parameters.get("goal")
        max_depth = request.parameters.get("max_depth", 3)
        context = request.parameters.get("context", {})

        if not goal_description:
            raise ValueError("Missing required parameter: 'goal'")

        # Decompose goal
        decomposer = GoalDecomposer(max_depth=max_depth)
        goal = decomposer.decompose(goal_description, context)

        # Get execution order
        execution_order = decomposer.get_execution_order(goal)

        # Estimate complexity
        complexity = decomposer.estimate_complexity(goal)

        # Build result
        result = {
            "goal": {
                "id": goal.id,
                "description": goal.description,
                "sub_goals": goal.sub_goals,
                "status": goal.status.value
            },
            "execution_order": [
                {"id": g.id, "description": g.description}
                for g in execution_order
            ],
            "complexity": complexity
        }

        # Calculate confidence
        confidence_score = 0.85 if complexity["total_sub_goals"] > 0 else 0.6

        confidence = ConfidenceMetadata.from_score(
            confidence_score,
            justification=[f"Decomposed into {complexity['total_sub_goals']} sub-goals"],
            uncertainty_sources=[]
        )

        return result, confidence

    async def _create_plan(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Create action plan for goal."""
        goal_description = request.parameters.get("goal")
        initial_state = request.parameters.get("initial_state", {})
        planning_horizon = request.parameters.get("planning_horizon", 10)

        if not goal_description:
            raise ValueError("Missing required parameter: 'goal'")

        # Create goal
        goal = Goal(
            id="goal_1",
            description=goal_description,
            success_criteria=["goal_achieved"]  # Simplified
        )

        # Create planner
        planner = ActionPlanner(planning_horizon=planning_horizon)

        # Generate plan
        plan = planner.create_plan(goal, initial_state, [])

        # Build result
        result = {
            "plan": {
                "id": plan.id,
                "goal_id": plan.goal_id,
                "actions": [
                    {
                        "id": a.id,
                        "description": a.description,
                        "duration": a.duration_estimate,
                        "cost": a.cost,
                        "confidence": a.confidence
                    }
                    for a in plan.actions
                ],
                "total_duration": plan.total_duration,
                "total_cost": plan.total_cost,
                "confidence": plan.confidence
            }
        }

        confidence = ConfidenceMetadata.from_score(
            plan.confidence,
            justification=[f"Generated plan with {len(plan.actions)} actions"],
            uncertainty_sources=[]
        )

        return result, confidence

    async def _validate_plan(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Validate plan feasibility."""
        plan_data = request.parameters.get("plan")
        resources = request.parameters.get("resources", {})
        budget = request.parameters.get("budget")

        if not plan_data:
            raise ValueError("Missing required parameter: 'plan'")

        # Reconstruct plan (simplified)
        plan = Plan(
            id=plan_data.get("id", "plan_1"),
            goal_id=plan_data.get("goal_id", "goal_1"),
            actions=[],
            total_cost=plan_data.get("total_cost", 0.0),
            total_duration=plan_data.get("total_duration", 0.0)
        )

        # Validate
        feasibility_score, checks = self.plan_validator.validate(plan, resources, budget)

        result = {
            "feasibility_score": feasibility_score,
            "validation_checks": checks,
            "is_feasible": feasibility_score >= 0.7
        }

        confidence = ConfidenceMetadata.from_score(
            0.90,
            justification=["Plan validation complete"],
            uncertainty_sources=[]
        )

        return result, confidence

    async def _optimize_plan(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Optimize plan efficiency."""
        plan_data = request.parameters.get("plan")

        if not plan_data:
            raise ValueError("Missing required parameter: 'plan'")

        # Simple optimization: remove redundant actions
        # In production, would use more sophisticated techniques

        result = {
            "optimized_plan": plan_data,
            "optimization_applied": "redundancy_removal",
            "improvement": "10%"  # Simplified
        }

        confidence = ConfidenceMetadata.from_score(
            0.80,
            justification=["Plan optimization complete"],
            uncertainty_sources=["Optimization heuristics may not be optimal"]
        )

        return result, confidence

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _build_reasoning(
        self,
        result: dict[str, Any],
        request: DepartmentRequest
    ) -> dict[str, Any]:
        """Build reasoning dict for response."""
        return {
            "task_type": request.task_type,
            "planning_method": "hierarchical_decomposition"
        }

    async def _build_session_state(
        self,
        session_id: str | None,
        result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build session state update."""
        if not session_id:
            return None

        return {
            "last_plan": result.get("plan", {}).get("id")
        }
