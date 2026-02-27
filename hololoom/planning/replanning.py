"""
Continuous Replanning Engine for Layer 2

Implements adaptive planning with execution monitoring:
- Real-time execution monitoring
- Failure detection (action failures, unexpected states)
- Dynamic replanning (full replan vs plan repair)
- Opportunistic replanning (exploit new opportunities)
- Divergence tracking (expected vs actual state)

Research:
- Ghallab et al. (2016): "Acting and Planning" (execution monitoring)
- van der Krogt & de Weerdt (2005): Plan repair in temporal planning
- Fox et al. (2006): EUROPA - Planning for space missions
- Myers (1999): Continuous planning and scheduling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum
import logging
from copy import deepcopy
import time

# Import Layer 2 core
from HoloLoom.planning.planner import HierarchicalPlanner, Plan, Goal, Action

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Core Types
# ============================================================================

class ExecutionStatus(Enum):
    """Status of action execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"      # Partial success
    DELAYED = "delayed"      # Taking longer than expected
    BLOCKED = "blocked"      # Cannot execute (preconditions not met)


class ReplanTrigger(Enum):
    """Reasons for replanning."""
    FAILURE = "failure"               # Action failed
    DIVERGENCE = "divergence"         # State diverged too much
    OPPORTUNITY = "opportunity"       # Better option available
    TIMEOUT = "timeout"               # Deadline approaching
    NEW_GOAL = "new_goal"            # Goal changed
    RESOURCE_SHORTAGE = "resource_shortage"  # Resources depleted
    PRECONDITION_VIOLATED = "precondition_violated"  # Can't proceed


class ReplanStrategy(Enum):
    """Replanning strategies."""
    FULL = "full"              # Generate entirely new plan
    REPAIR = "repair"          # Fix broken plan minimally
    CONTINUATION = "continuation"  # Continue from current state
    OPPORTUNISTIC = "opportunistic"  # Exploit new opportunities


# ============================================================================
# Execution Tracking
# ============================================================================

@dataclass
class ExecutionResult:
    """Result of executing an action."""
    action: Action
    status: ExecutionStatus
    actual_state: Dict[str, Any]
    expected_state: Dict[str, Any]
    cost: float
    duration: float
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None

    def is_success(self) -> bool:
        """Check if execution succeeded."""
        return self.status == ExecutionStatus.SUCCESS

    def __repr__(self):
        return (f"ExecutionResult({self.status.value}, "
                f"cost={self.cost:.2f}, duration={self.duration:.2f})")


@dataclass
class ExecutionTrace:
    """Complete trace of plan execution."""
    plan: Plan
    results: List[ExecutionResult] = field(default_factory=list)
    current_step: int = 0
    current_state: Dict[str, Any] = field(default_factory=dict)
    total_cost: float = 0.0
    total_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    replans: int = 0  # Number of replans performed

    def add_result(self, result: ExecutionResult):
        """Add execution result to trace."""
        self.results.append(result)
        self.current_step += 1
        self.current_state = deepcopy(result.actual_state)
        self.total_cost += result.cost
        self.total_time += result.duration

    def success_rate(self) -> float:
        """Calculate success rate of executed actions."""
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.is_success())
        return successes / len(self.results)

    def __repr__(self):
        return (f"ExecutionTrace({self.current_step}/{len(self.plan.actions)} steps, "
                f"{self.success_rate():.0%} success, {self.replans} replans)")


# ============================================================================
# Execution Monitor
# ============================================================================

class ExecutionMonitor:
    """Monitors plan execution and detects when replanning is needed."""

    def __init__(self,
                 plan: Plan,
                 executor: Callable[[Action], Tuple[ExecutionStatus, Dict, float]],
                 divergence_threshold: float = 0.3,
                 timeout_buffer: float = 0.2):
        """
        Initialize execution monitor.

        Args:
            plan: Plan to monitor
            executor: Function that executes actions
                     Returns (status, actual_state, duration)
            divergence_threshold: Max allowed state divergence (0-1)
            timeout_buffer: Replan when this fraction of deadline remains (0-1)
        """
        self.plan = plan
        self.executor = executor
        self.divergence_threshold = divergence_threshold
        self.timeout_buffer = timeout_buffer

        self.trace = ExecutionTrace(plan=plan)
        self.start_time = time.time()

        logger.info(f"Initialized ExecutionMonitor for plan with {len(plan.actions)} actions")

    # ------------------------------------------------------------------------
    # Action Execution
    # ------------------------------------------------------------------------

    def execute_step(self, action: Action, expected_state: Dict) -> ExecutionResult:
        """
        Execute single action and monitor result.

        Args:
            action: Action to execute
            expected_state: Expected state after action

        Returns:
            ExecutionResult with actual outcome
        """
        logger.info(f"Executing action: {action}")

        start_time = time.time()

        try:
            # Execute action
            status, actual_state, cost = self.executor(action)
            duration = time.time() - start_time

            result = ExecutionResult(
                action=action,
                status=status,
                actual_state=actual_state,
                expected_state=expected_state,
                cost=cost,
                duration=duration,
                timestamp=time.time()
            )

            # Add to trace
            self.trace.add_result(result)

            if status == ExecutionStatus.SUCCESS:
                logger.info(f"✓ Action succeeded in {duration:.2f}s")
            else:
                logger.warning(f"✗ Action {status.value} in {duration:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            duration = time.time() - start_time

            result = ExecutionResult(
                action=action,
                status=ExecutionStatus.FAILURE,
                actual_state=self.trace.current_state,  # No change
                expected_state=expected_state,
                cost=0.0,
                duration=duration,
                error_message=str(e)
            )

            self.trace.add_result(result)
            return result

    # ------------------------------------------------------------------------
    # Divergence Checking
    # ------------------------------------------------------------------------

    def check_divergence(self, expected: Dict, actual: Dict) -> float:
        """
        Measure state divergence from expected.

        Returns:
            Divergence score (0 = perfect match, 1 = completely different)
        """
        if not expected:
            return 0.0

        # Count mismatched variables
        all_vars = set(expected.keys()) | set(actual.keys())
        mismatches = 0

        for var in all_vars:
            expected_val = expected.get(var)
            actual_val = actual.get(var)

            # Handle numeric values
            if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                # Normalize difference to [0, 1]
                max_val = max(abs(expected_val), abs(actual_val), 1.0)
                diff = abs(expected_val - actual_val) / max_val
                if diff > 0.1:  # Threshold for numeric difference
                    mismatches += 1
            # Handle other types
            elif expected_val != actual_val:
                mismatches += 1

        divergence = mismatches / len(all_vars) if all_vars else 0.0
        return divergence

    def get_cumulative_divergence(self) -> float:
        """Calculate average divergence across all executed actions."""
        if not self.trace.results:
            return 0.0

        divergences = [
            self.check_divergence(r.expected_state, r.actual_state)
            for r in self.trace.results
        ]
        return sum(divergences) / len(divergences)

    # ------------------------------------------------------------------------
    # Failure Detection
    # ------------------------------------------------------------------------

    def detect_failure(self, result: ExecutionResult) -> bool:
        """Detect if action failed."""
        return result.status in [ExecutionStatus.FAILURE, ExecutionStatus.BLOCKED]

    def should_replan(self,
                     deadline: Optional[float] = None,
                     resources: Optional[Dict[str, float]] = None) -> Optional[ReplanTrigger]:
        """
        Decide if replanning is needed.

        Args:
            deadline: Time deadline (seconds from start)
            resources: Available resources

        Returns:
            ReplanTrigger if should replan, None otherwise
        """
        # Check for recent failures
        if self.trace.results:
            last_result = self.trace.results[-1]

            if last_result.status == ExecutionStatus.FAILURE:
                logger.info("Replan trigger: FAILURE")
                return ReplanTrigger.FAILURE

            if last_result.status == ExecutionStatus.BLOCKED:
                logger.info("Replan trigger: PRECONDITION_VIOLATED")
                return ReplanTrigger.PRECONDITION_VIOLATED

        # Check divergence
        divergence = self.get_cumulative_divergence()
        if divergence > self.divergence_threshold:
            logger.info(f"Replan trigger: DIVERGENCE ({divergence:.2%} > {self.divergence_threshold:.2%})")
            return ReplanTrigger.DIVERGENCE

        # Check deadline
        if deadline:
            elapsed = time.time() - self.start_time
            remaining = deadline - elapsed
            buffer_time = deadline * self.timeout_buffer

            if remaining < buffer_time:
                logger.info(f"Replan trigger: TIMEOUT ({remaining:.1f}s remaining)")
                return ReplanTrigger.TIMEOUT

        # Check resources
        if resources:
            for resource, amount in resources.items():
                if amount <= 0:
                    logger.info(f"Replan trigger: RESOURCE_SHORTAGE ({resource})")
                    return ReplanTrigger.RESOURCE_SHORTAGE

        # No replanning needed
        return None

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'total_steps': self.trace.current_step,
            'total_actions': len(self.plan.actions),
            'success_rate': self.trace.success_rate(),
            'total_cost': self.trace.total_cost,
            'total_time': self.trace.total_time,
            'replans': self.trace.replans,
            'average_divergence': self.get_cumulative_divergence(),
            'elapsed_time': time.time() - self.start_time
        }


# ============================================================================
# Replanning Engine
# ============================================================================

class ReplanningEngine:
    """Continuous replanning engine with multiple strategies."""

    def __init__(self,
                 planner: HierarchicalPlanner,
                 default_strategy: ReplanStrategy = ReplanStrategy.REPAIR):
        """
        Initialize replanning engine.

        Args:
            planner: Base planner for generating new plans
            default_strategy: Default replanning strategy
        """
        self.planner = planner
        self.default_strategy = default_strategy

        logger.info(f"Initialized ReplanningEngine with {default_strategy.value} strategy")

    # ------------------------------------------------------------------------
    # Main Replanning
    # ------------------------------------------------------------------------

    def replan(self,
              trigger: ReplanTrigger,
              current_state: Dict,
              goal: Goal,
              original_plan: Plan,
              failure_step: Optional[int] = None) -> Optional[Plan]:
        """
        Generate new plan based on trigger.

        Args:
            trigger: Why we're replanning
            current_state: Current world state
            goal: Original goal
            original_plan: Original plan that failed/diverged
            failure_step: Which step failed (if applicable)

        Returns:
            New plan or None if unable to replan
        """
        logger.info(f"Replanning due to {trigger.value}")

        # Select strategy based on trigger
        if trigger == ReplanTrigger.FAILURE:
            strategy = ReplanStrategy.REPAIR
        elif trigger == ReplanTrigger.DIVERGENCE:
            strategy = ReplanStrategy.CONTINUATION
        elif trigger == ReplanTrigger.OPPORTUNITY:
            strategy = ReplanStrategy.OPPORTUNISTIC
        elif trigger == ReplanTrigger.TIMEOUT:
            strategy = ReplanStrategy.FULL  # Need fastest plan
        else:
            strategy = self.default_strategy

        # Execute strategy
        if strategy == ReplanStrategy.FULL:
            return self._replan_full(current_state, goal)
        elif strategy == ReplanStrategy.REPAIR:
            return self._replan_repair(original_plan, failure_step, current_state, goal)
        elif strategy == ReplanStrategy.CONTINUATION:
            return self._replan_continuation(current_state, goal)
        elif strategy == ReplanStrategy.OPPORTUNISTIC:
            return self._replan_opportunistic(current_state, goal, original_plan)
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return None

    # ------------------------------------------------------------------------
    # Replanning Strategies
    # ------------------------------------------------------------------------

    def _replan_full(self, current_state: Dict, goal: Goal) -> Optional[Plan]:
        """
        Generate entirely new plan from scratch.

        Pro: Can find completely different solution
        Con: Wasteful, loses progress
        """
        logger.info("Full replanning: generating new plan from scratch")
        return self.planner.plan(goal, current_state)

    def _replan_repair(self,
                      original_plan: Plan,
                      failure_step: Optional[int],
                      current_state: Dict,
                      goal: Goal) -> Optional[Plan]:
        """
        Repair broken plan minimally.

        Pro: Preserves most of plan, fast
        Con: May not find better alternative
        """
        logger.info("Plan repair: fixing broken plan")

        if failure_step is None:
            failure_step = len(original_plan.actions) - 1

        # Strategy: Keep prefix (before failure), replan suffix
        prefix = original_plan.actions[:failure_step]

        # Replan from current state to goal
        suffix_plan = self.planner.plan(goal, current_state)
        if not suffix_plan:
            logger.warning("Repair failed: cannot plan suffix")
            return None

        # Combine
        repaired_actions = prefix + suffix_plan.actions
        repaired_plan = Plan(
            actions=repaired_actions,
            goal=goal,
            expected_cost=original_plan.expected_cost + suffix_plan.expected_cost
        )

        logger.info(f"Repaired plan: {len(prefix)} kept + {len(suffix_plan.actions)} new")
        return repaired_plan

    def _replan_continuation(self, current_state: Dict, goal: Goal) -> Optional[Plan]:
        """
        Plan continuation from current state.

        Pro: Adapts to actual state
        Con: Ignores original plan completely
        """
        logger.info("Continuation: planning from current state")
        return self.planner.plan(goal, current_state)

    def _replan_opportunistic(self,
                             current_state: Dict,
                             goal: Goal,
                             original_plan: Plan) -> Optional[Plan]:
        """
        Exploit new opportunities while keeping good parts.

        Pro: Can improve on original plan
        Con: Complex, may waste time
        """
        logger.info("Opportunistic replanning")

        # For now, same as continuation
        # TODO: Implement intelligent opportunity exploitation
        return self._replan_continuation(current_state, goal)


# ============================================================================
# Adaptive Planner
# ============================================================================

class AdaptivePlanner:
    """Integrates planning and replanning for adaptive execution."""

    def __init__(self,
                 base_planner: HierarchicalPlanner,
                 executor: Callable,
                 max_replans: int = 10,
                 divergence_threshold: float = 0.3):
        """
        Initialize adaptive planner.

        Args:
            base_planner: Base hierarchical planner
            executor: Function that executes actions
            max_replans: Maximum number of replans allowed
            divergence_threshold: When to trigger replanning
        """
        self.base_planner = base_planner
        self.executor = executor
        self.max_replans = max_replans
        self.divergence_threshold = divergence_threshold

        self.replanning_engine = ReplanningEngine(base_planner)

        logger.info(f"Initialized AdaptivePlanner (max_replans={max_replans})")

    # ------------------------------------------------------------------------
    # Plan and Execute
    # ------------------------------------------------------------------------

    def plan_and_execute(self,
                        goal: Goal,
                        initial_state: Dict,
                        deadline: Optional[float] = None) -> ExecutionTrace:
        """
        Plan, execute, monitor, and replan as needed.

        This is the main entry point for adaptive planning.

        Args:
            goal: Goal to achieve
            initial_state: Initial world state
            deadline: Time deadline (seconds)

        Returns:
            Complete execution trace
        """
        logger.info(f"Adaptive planning for goal: {goal.description}")

        # Step 1: Initial planning
        current_state = deepcopy(initial_state)
        plan = self.base_planner.plan(goal, current_state)

        if not plan:
            logger.error("Initial planning failed")
            return ExecutionTrace(plan=Plan(actions=[], goal=goal))

        logger.info(f"Initial plan: {len(plan.actions)} actions")

        # Step 2: Execute with monitoring and replanning
        total_trace = None
        replans = 0

        while replans <= self.max_replans:
            # Create monitor for current plan
            monitor = ExecutionMonitor(
                plan=plan,
                executor=self.executor,
                divergence_threshold=self.divergence_threshold
            )

            # Execute plan with monitoring
            trace = self._execute_with_monitoring(
                plan, monitor, current_state, goal, deadline
            )

            # Merge traces
            if total_trace is None:
                total_trace = trace
            else:
                total_trace.results.extend(trace.results)
                total_trace.current_step += trace.current_step
                total_trace.current_state = trace.current_state
                total_trace.total_cost += trace.total_cost
                total_trace.total_time += trace.total_time

            # Check if goal achieved
            if self._check_goal(goal, trace.current_state):
                logger.info(f"✓ Goal achieved! ({replans} replans)")
                total_trace.replans = replans
                return total_trace

            # Check if should replan
            trigger = monitor.should_replan(deadline=deadline)

            if trigger is None:
                # No replan needed, but goal not achieved
                logger.warning("Plan completed but goal not achieved")
                total_trace.replans = replans
                return total_trace

            # Replan
            logger.info(f"Replanning ({replans+1}/{self.max_replans})...")
            replans += 1

            current_state = trace.current_state
            plan = self.replanning_engine.replan(
                trigger=trigger,
                current_state=current_state,
                goal=goal,
                original_plan=plan,
                failure_step=trace.current_step
            )

            if not plan:
                logger.error("Replanning failed")
                total_trace.replans = replans
                return total_trace

            logger.info(f"New plan: {len(plan.actions)} actions")

        # Max replans exceeded
        logger.error(f"Max replans ({self.max_replans}) exceeded")
        total_trace.replans = replans
        return total_trace

    def _execute_with_monitoring(self,
                                 plan: Plan,
                                 monitor: ExecutionMonitor,
                                 current_state: Dict,
                                 goal: Goal,
                                 deadline: Optional[float]) -> ExecutionTrace:
        """Execute plan with continuous monitoring."""

        for i, action in enumerate(plan.actions):
            # Get expected state (from plan)
            expected_state = self._predict_state(current_state, action)

            # Execute action
            result = monitor.execute_step(action, expected_state)

            # Update current state
            current_state = result.actual_state

            # Check if should replan
            trigger = monitor.should_replan(deadline=deadline)
            if trigger:
                logger.info(f"Stopping execution at step {i+1} due to {trigger.value}")
                break

            # Check if goal achieved early
            if self._check_goal(goal, current_state):
                logger.info(f"Goal achieved early at step {i+1}")
                break

        return monitor.trace

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def _check_goal(self, goal: Goal, state: Dict) -> bool:
        """Check if goal is satisfied in state."""
        for var, desired_value in goal.desired_state.items():
            actual_value = state.get(var)
            if actual_value != desired_value:
                return False
        return True

    def _predict_state(self, state: Dict, action: Action) -> Dict:
        """Predict state after action (simple version)."""
        # For now, just copy effects
        # Real system would use causal model
        new_state = deepcopy(state)
        new_state.update(action.effects)
        return new_state


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ExecutionStatus',
    'ReplanTrigger',
    'ReplanStrategy',
    'ExecutionResult',
    'ExecutionTrace',
    'ExecutionMonitor',
    'ReplanningEngine',
    'AdaptivePlanner',
]
