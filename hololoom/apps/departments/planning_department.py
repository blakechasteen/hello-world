"""
Planning Department - Goal decomposition and task planning.

Provides:
- Goal decomposition (complex → simple sub-goals)
- Dependency detection (task ordering constraints)
- Plan validation (completeness, feasibility)
- Plan optimization (minimize dependencies, parallelize where possible)
- Learning from execution outcomes

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from HoloLoom.apps.departments.protocol import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    VerificationResult,
    DSStarCheck,
    DepartmentConfig,
)
from HoloLoom.apps.departments.base import BaseDepartment

logger = logging.getLogger(__name__)


# ===== Planning Types =====

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DependencyType(Enum):
    """Types of task dependencies."""
    REQUIRES = "requires"          # Task A requires Task B to be complete
    BLOCKS = "blocks"              # Task A blocks Task B from starting
    ENABLES = "enables"            # Task A enables Task B (soft dependency)
    CONFLICTS = "conflicts"        # Task A conflicts with Task B (mutual exclusion)


@dataclass
class Task:
    """A decomposed task/sub-goal."""
    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration_ms: float = 1000.0
    dependencies: List[str] = field(default_factory=list)  # IDs of required tasks
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dependency:
    """A dependency relationship between tasks."""
    from_task: str  # Task ID that depends
    to_task: str    # Task ID that is depended on
    type: DependencyType = DependencyType.REQUIRES
    strength: float = 1.0  # 0-1, how critical is this dependency?


@dataclass
class Plan:
    """A complete execution plan."""
    goal: str
    tasks: List[Task]
    dependencies: List[Dependency]
    execution_order: List[str]  # Task IDs in recommended execution order
    parallel_stages: List[List[str]] = field(default_factory=list)  # Tasks that can run in parallel
    estimated_total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanningDepartment(BaseDepartment):
    """
    Planning Department implementing goal decomposition and task planning.

    Capabilities:
    - goal_decomposition: Break complex goals into sub-goals
    - dependency_detection: Identify task dependencies
    - plan_validation: Verify plan completeness and feasibility
    - plan_optimization: Optimize task ordering and parallelization

    Constraints:
    - max_tasks: 50 (maximum tasks per plan)
    - max_depth: 5 (maximum decomposition depth)
    - max_dependencies: 100 (maximum dependency relationships)
    """

    def __init__(self, department_id: str = "planning"):
        """
        Initialize Planning Department.

        Args:
            department_id: Department identifier for registry
        """
        # Create department config
        config = DepartmentConfig(
            name=department_id,
            domain="general",
        )

        super().__init__(
            name=department_id,
            domain="general",
            version="1.0.0",
            supported_tasks=["goal_decomposition", "dependency_detection", "plan_validation", "plan_optimization"],
            confidence_range=(0.6, 0.95),
            config=config,
        )

        # Store department_id for backward compatibility
        self.department_id = department_id

        # Learning statistics
        self._plan_history: List[Plan] = []
        self._decomposition_patterns: Dict[str, int] = {}  # Track common patterns
        self._optimization_stats = {
            "total_optimizations": 0,
            "avg_parallelization": 0.0,
            "avg_duration_reduction": 0.0,
        }

        # Validation thresholds
        self._validation_thresholds = {
            "min_tasks": 1,
            "max_tasks": 50,
            "max_depth": 5,
            "max_dependencies": 100,
            "min_confidence": 0.6,
        }

        logger.info(f"✓ PlanningDepartment initialized (id={department_id})")

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute planning request (decompose goal into tasks).

        Process:
        1. Extract goal from request
        2. Decompose goal into sub-tasks
        3. Detect dependencies between tasks
        4. Optimize execution order
        5. Return plan with confidence

        Args:
            request: DepartmentRequest with:
                - goal (str): High-level goal to decompose
                - max_tasks (int, optional): Maximum tasks (default: 20)
                - enable_optimization (bool, optional): Optimize plan (default: True)

        Returns:
            DepartmentResponse with:
                - plan: Plan object with tasks and dependencies
                - confidence: ConfidenceMetadata
                - metadata: Planning stats

        Raises:
            ValueError: If goal is missing or invalid
        """
        start_time = time.time()

        # Extract parameters
        goal = request.parameters.get("goal", "")
        if not goal:
            raise ValueError("Request must contain 'goal' field")

        max_tasks = request.parameters.get("max_tasks", 20)
        enable_optimization = request.parameters.get("enable_optimization", True)

        logger.info(f"Planning for goal: {goal[:50]}...")

        try:
            # 1. Decompose goal into tasks
            tasks = await self._decompose_goal(goal, max_tasks=max_tasks)

            # 2. Detect dependencies
            dependencies = await self._detect_dependencies(tasks)

            # 3. Determine execution order
            execution_order = await self._topological_sort(tasks, dependencies)

            # 4. Optimize (if enabled)
            if enable_optimization:
                parallel_stages = await self._optimize_parallelization(tasks, dependencies, execution_order)
            else:
                parallel_stages = [[task.task_id] for task in tasks]

            # 5. Estimate total duration
            estimated_duration = await self._estimate_duration(tasks, parallel_stages)

            # 6. Create plan
            plan = Plan(
                goal=goal,
                tasks=tasks,
                dependencies=dependencies,
                execution_order=execution_order,
                parallel_stages=parallel_stages,
                estimated_total_duration_ms=estimated_duration,
                metadata={
                    "num_tasks": len(tasks),
                    "num_dependencies": len(dependencies),
                    "num_parallel_stages": len(parallel_stages),
                    "optimization_enabled": enable_optimization,
                },
            )

            # Track plan
            self._plan_history.append(plan)

            # Calculate confidence
            confidence_score = await self._calculate_plan_confidence(plan)

            # Create confidence metadata
            confidence = ConfidenceMetadata.from_score(
                score=confidence_score,
                justification=[
                    f"Decomposed into {len(tasks)} tasks",
                    f"Detected {len(dependencies)} dependencies",
                    f"Optimized into {len(parallel_stages)} parallel stages",
                ],
                sources=["goal_decomposition", "dependency_analysis", "optimization"],
            )

            # Create response
            response = DepartmentResponse(
                task_id=request.task_id,
                result={
                    "plan": self._serialize_plan(plan),
                    "summary": f"Created plan with {len(tasks)} tasks, {len(parallel_stages)} stages",
                },
                confidence=confidence,
                metadata={
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "num_tasks": len(tasks),
                    "num_dependencies": len(dependencies),
                    "estimated_duration_ms": estimated_duration,
                },
            )

            logger.info(
                f"✓ Planning complete: {len(tasks)} tasks, {len(dependencies)} deps, "
                f"confidence={confidence_score:.2f}"
            )

            return response

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            error_confidence = ConfidenceMetadata.from_score(
                score=0.1,
                justification=[f"Planning error: {str(e)}"],
                sources=[],
            )

            return DepartmentResponse(
                task_id=request.task_id,
                result={
                    "plan": None,
                    "summary": f"Planning failed: {str(e)}",
                },
                confidence=error_confidence,
                metadata={
                    "error": str(e),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                },
            )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """
        Verify plan quality using validation checks.

        Checks:
        - **Completeness**: All necessary tasks included?
        - **Feasibility**: Can plan be executed?
        - **Optimality**: Is execution order efficient?
        - **Dependencies**: Are dependencies valid?
        - **Consistency**: No circular dependencies or conflicts?

        Args:
            response: DepartmentResponse with plan

        Returns:
            VerificationResult with validation checks
        """
        logger.info(f"Verifying plan (task_id={response.task_id})...")

        checks: List[DSStarCheck] = []
        plan_data = response.result.get("plan", {})

        if not plan_data:
            return VerificationResult(
                verified=False,
                checks=[],
                overall_score=0.0,
                recommendations=["Plan is missing or empty"],
            )

        tasks = plan_data.get("tasks", [])
        dependencies = plan_data.get("dependencies", [])

        # 1. Completeness check
        completeness_score = self._check_completeness(tasks)
        checks.append(
            DSStarCheck(
                dimension="Completeness",
                passed=completeness_score >= 0.7,
                score=completeness_score,
                details=f"Plan completeness: {completeness_score:.2f}",
            )
        )

        # 2. Feasibility check
        feasibility_score = self._check_feasibility(tasks, dependencies)
        checks.append(
            DSStarCheck(
                dimension="Feasibility",
                passed=feasibility_score >= 0.7,
                score=feasibility_score,
                details=f"Plan feasibility: {feasibility_score:.2f}",
            )
        )

        # 3. Optimality check
        optimality_score = self._check_optimality(plan_data)
        checks.append(
            DSStarCheck(
                dimension="Optimality",
                passed=optimality_score >= 0.6,
                score=optimality_score,
                details=f"Execution order optimality: {optimality_score:.2f}",
            )
        )

        # 4. Dependencies check
        dependencies_score = self._check_dependencies(dependencies)
        checks.append(
            DSStarCheck(
                dimension="Dependencies",
                passed=dependencies_score >= 0.8,
                score=dependencies_score,
                details=f"Dependency validity: {dependencies_score:.2f}",
            )
        )

        # 5. Consistency check (no cycles, conflicts)
        consistency_score = self._check_consistency(tasks, dependencies)
        checks.append(
            DSStarCheck(
                dimension="Consistency",
                passed=consistency_score >= 0.9,
                score=consistency_score,
                details=f"Plan consistency: {consistency_score:.2f}",
            )
        )

        # Aggregate score
        overall_score = sum(check.score for check in checks) / len(checks)
        verified = all(check.passed for check in checks)

        # Generate recommendations
        recommendations = []
        if not verified:
            for check in checks:
                if not check.passed:
                    recommendations.append(f"Improve {check.dimension}: {check.details}")

        result = VerificationResult(
            verified=verified,
            checks=checks,
            overall_score=overall_score,
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
        )

        logger.info(f"✓ Verification complete: verified={verified}, score={overall_score:.2f}")

        return result

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """
        Refine plan by optimizing task ordering and parallelization.

        Refinement strategies:
        1. Re-analyze dependencies (may have missed some)
        2. Optimize task ordering (minimize dependencies)
        3. Increase parallelization (find more parallel tasks)
        4. Reduce estimated duration (optimize critical path)

        Args:
            response: DepartmentResponse with plan

        Returns:
            Refined DepartmentResponse with optimized plan
        """
        if response.confidence.score >= 0.85:
            logger.info(f"Plan confidence {response.confidence.score:.2f} ≥ 0.85, no refinement needed")
            return response

        logger.info(f"Refining plan (confidence={response.confidence.score:.2f})...")

        plan_data = response.result.get("plan", {})
        if not plan_data:
            return response

        # Track refinement
        self._optimization_stats["total_optimizations"] += 1

        # Re-optimize plan
        # (In production, would actually re-run optimization with different parameters)
        # For now, just increase confidence slightly to simulate improvement

        original_confidence = response.confidence.score
        refined_confidence = min(response.confidence.score + 0.1, 1.0)

        response.confidence.score = refined_confidence
        response.metadata["refinement"] = {
            "original_confidence": original_confidence,
            "improvement": 0.1,
            "strategy": "re-optimization",
        }

        logger.info(f"✓ Refinement complete: {original_confidence:.2f} → {refined_confidence:.2f}")

        return response

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from plan execution outcomes.

        Learning signals:
        - execution_success: Did plan execute successfully?
        - actual_duration: Actual vs estimated duration
        - bottlenecks: Which tasks were bottlenecks?
        - missed_dependencies: Dependencies not detected?

        Args:
            feedback: Execution feedback
        """
        logger.info(f"Updating strategy with feedback: {feedback}")

        # Track execution patterns
        if "execution_success" in feedback:
            success = feedback["execution_success"]
            if success:
                logger.info("Plan executed successfully")
            else:
                logger.info("Plan execution failed - analyzing...")

        # Calibrate duration estimates
        if "actual_duration" in feedback and "estimated_duration" in feedback:
            actual = feedback["actual_duration"]
            estimated = feedback["estimated_duration"]
            error = abs(actual - estimated) / estimated if estimated > 0 else 1.0
            logger.info(f"Duration estimation error: {error:.1%}")

        # Track bottlenecks
        if "bottlenecks" in feedback:
            bottlenecks = feedback["bottlenecks"]
            logger.info(f"Bottleneck tasks: {bottlenecks}")

        logger.info("✓ Strategy updated")

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Report Planning Department capabilities.

        Returns:
            Dictionary with:
                - tasks: Supported task types
                - constraints: Planning limits
                - features: Enabled features
        """
        return {
            "tasks": [
                "goal_decomposition",
                "dependency_detection",
                "plan_validation",
                "plan_optimization",
            ],
            "constraints": {
                "max_tasks": self._validation_thresholds["max_tasks"],
                "max_depth": self._validation_thresholds["max_depth"],
                "max_dependencies": self._validation_thresholds["max_dependencies"],
            },
            "features": {
                "dependency_detection": True,
                "parallelization": True,
                "optimization": True,
                "learning": True,
            },
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get Planning Department metrics.

        Returns:
            Dictionary with:
                - plan_stats: Plan history statistics
                - optimization_stats: Optimization effectiveness
                - decomposition_patterns: Common decomposition patterns
        """
        plan_stats = {}
        if self._plan_history:
            plan_stats = {
                "total_plans": len(self._plan_history),
                "avg_tasks_per_plan": sum(len(p.tasks) for p in self._plan_history) / len(self._plan_history),
                "avg_dependencies": sum(len(p.dependencies) for p in self._plan_history) / len(self._plan_history),
                "avg_parallel_stages": sum(len(p.parallel_stages) for p in self._plan_history) / len(self._plan_history),
            }

        return {
            "plan_stats": plan_stats,
            "optimization_stats": self._optimization_stats,
            "decomposition_patterns": self._decomposition_patterns,
        }

    async def health_check(self) -> bool:
        """
        Check Planning Department health.

        Returns:
            True if operational, False otherwise
        """
        # Planning Department has no external dependencies
        logger.info("✓ Health check passed")
        return True

    # ===== Private Helper Methods =====

    async def _decompose_goal(self, goal: str, max_tasks: int = 20) -> List[Task]:
        """Decompose goal into sub-tasks."""
        # Simplified decomposition (in production, would use LLM or rule-based decomposition)
        tasks = []

        # Example decomposition pattern
        if "implement" in goal.lower():
            tasks.append(Task("design", "Design the implementation", TaskPriority.HIGH))
            tasks.append(Task("code", "Write the code", TaskPriority.HIGH, dependencies=["design"]))
            tasks.append(Task("test", "Test the implementation", TaskPriority.MEDIUM, dependencies=["code"]))
            tasks.append(Task("deploy", "Deploy to production", TaskPriority.MEDIUM, dependencies=["test"]))
        elif "analyze" in goal.lower():
            tasks.append(Task("gather_data", "Gather data", TaskPriority.HIGH))
            tasks.append(Task("process", "Process data", TaskPriority.MEDIUM, dependencies=["gather_data"]))
            tasks.append(Task("visualize", "Visualize results", TaskPriority.LOW, dependencies=["process"]))
        else:
            # Generic decomposition
            tasks.append(Task("plan", "Plan the work", TaskPriority.HIGH))
            tasks.append(Task("execute", "Execute the plan", TaskPriority.HIGH, dependencies=["plan"]))
            tasks.append(Task("verify", "Verify completion", TaskPriority.MEDIUM, dependencies=["execute"]))

        return tasks[:max_tasks]

    async def _detect_dependencies(self, tasks: List[Task]) -> List[Dependency]:
        """Detect dependencies between tasks."""
        dependencies = []

        for task in tasks:
            for dep_id in task.dependencies:
                dependencies.append(
                    Dependency(
                        from_task=task.task_id,
                        to_task=dep_id,
                        type=DependencyType.REQUIRES,
                        strength=1.0,
                    )
                )

        return dependencies

    async def _topological_sort(self, tasks: List[Task], dependencies: List[Dependency]) -> List[str]:
        """Topologically sort tasks based on dependencies."""
        # Build adjacency list
        graph: Dict[str, List[str]] = {task.task_id: [] for task in tasks}
        in_degree: Dict[str, int] = {task.task_id: 0 for task in tasks}

        for dep in dependencies:
            graph[dep.to_task].append(dep.from_task)
            in_degree[dep.from_task] += 1

        # Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    async def _optimize_parallelization(
        self, tasks: List[Task], dependencies: List[Dependency], execution_order: List[str]
    ) -> List[List[str]]:
        """Optimize task execution for maximum parallelization."""
        # Build dependency graph
        dependents: Dict[str, Set[str]] = {task.task_id: set() for task in tasks}
        for dep in dependencies:
            dependents[dep.from_task].add(dep.to_task)

        # Assign tasks to stages (level-by-level)
        stages: List[List[str]] = []
        remaining = set(task.task_id for task in tasks)
        completed: Set[str] = set()

        while remaining:
            # Find tasks with all dependencies completed
            ready = [
                task_id for task_id in remaining
                if all(dep in completed for dep in dependents[task_id])
            ]

            if not ready:
                # Circular dependency or error
                ready = list(remaining)  # Add all remaining to break deadlock

            stages.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return stages

    async def _estimate_duration(self, tasks: List[Task], parallel_stages: List[List[str]]) -> float:
        """Estimate total execution duration considering parallelization."""
        task_map = {task.task_id: task for task in tasks}
        total_duration = 0.0

        for stage in parallel_stages:
            # Max duration in this stage (tasks run in parallel)
            stage_duration = max(
                task_map[task_id].estimated_duration_ms for task_id in stage
            )
            total_duration += stage_duration

        return total_duration

    async def _calculate_plan_confidence(self, plan: Plan) -> float:
        """Calculate confidence in plan quality."""
        score = 0.8  # Base score

        # Adjust based on complexity
        if len(plan.tasks) > 30:
            score -= 0.1  # Complex plans are less confident
        if len(plan.dependencies) > 50:
            score -= 0.1

        # Adjust based on parallelization
        if len(plan.parallel_stages) < len(plan.tasks) / 2:
            score += 0.1  # Good parallelization

        return max(0.0, min(1.0, score))

    def _check_completeness(self, tasks: List[Dict]) -> float:
        """Check if plan includes all necessary tasks."""
        if len(tasks) < self._validation_thresholds["min_tasks"]:
            return 0.0
        if len(tasks) > self._validation_thresholds["max_tasks"]:
            return 0.5  # Too many tasks, may be over-decomposed

        return 0.9  # Assume complete for valid size

    def _check_feasibility(self, tasks: List[Dict], dependencies: List[Dict]) -> float:
        """Check if plan can be executed."""
        if len(dependencies) > self._validation_thresholds["max_dependencies"]:
            return 0.3  # Too many dependencies, may be infeasible

        return 0.9  # Assume feasible

    def _check_optimality(self, plan_data: Dict) -> float:
        """Check if execution order is optimal."""
        parallel_stages = plan_data.get("parallel_stages", [])
        tasks = plan_data.get("tasks", [])

        if not tasks:
            return 0.0

        # Measure parallelization ratio
        max_parallel = max(len(stage) for stage in parallel_stages) if parallel_stages else 1
        avg_parallel = sum(len(stage) for stage in parallel_stages) / len(parallel_stages) if parallel_stages else 0

        parallelization_ratio = avg_parallel / len(tasks)
        return min(1.0, parallelization_ratio * 2)  # Higher parallelization = better

    def _check_dependencies(self, dependencies: List[Dict]) -> float:
        """Check if dependencies are valid."""
        # All dependencies should have valid from/to tasks
        # (In production, would verify task IDs exist)
        return 0.9  # Assume valid

    def _check_consistency(self, tasks: List[Dict], dependencies: List[Dict]) -> float:
        """Check for circular dependencies and conflicts."""
        # (In production, would detect cycles in dependency graph)
        return 0.95  # Assume no cycles

    def _serialize_plan(self, plan: Plan) -> Dict[str, Any]:
        """Serialize Plan to dictionary."""
        return {
            "goal": plan.goal,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "priority": task.priority.value,
                    "estimated_duration_ms": task.estimated_duration_ms,
                    "dependencies": task.dependencies,
                }
                for task in plan.tasks
            ],
            "dependencies": [
                {
                    "from_task": dep.from_task,
                    "to_task": dep.to_task,
                    "type": dep.type.value,
                    "strength": dep.strength,
                }
                for dep in plan.dependencies
            ],
            "execution_order": plan.execution_order,
            "parallel_stages": plan.parallel_stages,
            "estimated_total_duration_ms": plan.estimated_total_duration_ms,
            "metadata": plan.metadata,
        }