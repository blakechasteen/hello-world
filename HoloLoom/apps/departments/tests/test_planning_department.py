"""
Unit tests for Planning Department.

Tests all 7 Department protocol methods:
1. execute() - Goal decomposition and task planning
2. verify() - Plan validation with 5 quality checks
3. refine() - Plan optimization
4. update_strategy() - Learning from execution outcomes
5. get_capabilities() - Capability reporting
6. get_metrics() - Performance metrics
7. health_check() - System health verification

Also tests helper methods:
- _decompose_goal() - Goal decomposition patterns
- _detect_dependencies() - Dependency extraction
- _topological_sort() - Task ordering (Kahn's algorithm)
- _optimize_parallelization() - Parallel stage optimization
- _estimate_duration() - Duration estimation
- _calculate_plan_confidence() - Confidence scoring

Test Coverage:
- Happy path: Valid goals decompose into tasks with dependencies
- Error cases: Missing goal, invalid parameters, empty results
- Edge cases: Very simple goals, complex goals, circular dependencies
- Integration: Full planning cycle from goal to optimized plan

Author: HoloLoom Team
Date: November 2025
"""

import pytest
import asyncio
from typing import Dict, List, Any
from datetime import datetime

# Import Planning Department
from HoloLoom.apps.departments.planning_department import (
    PlanningDepartment,
    Task,
    TaskPriority,
    Dependency,
    DependencyType,
    Plan,
)

# Import protocol classes
from HoloLoom.apps.departments.protocol import (
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    ConfidenceLevel,
    VerificationResult,
    DSStarCheck,
)


# ===== Fixtures =====

@pytest.fixture
def planning_department():
    """Create PlanningDepartment instance for testing."""
    return PlanningDepartment(department_id="test_planning")


@pytest.fixture
def simple_goal_request():
    """Simple implementation goal request."""
    return DepartmentRequest(
        task_id="test_001",
        parameters={
            "goal": "Implement a new feature",
            "max_tasks": 10,
            "enable_optimization": True,
        },
        context={},
    )


@pytest.fixture
def complex_goal_request():
    """Complex analysis goal request."""
    return DepartmentRequest(
        task_id="test_002",
        parameters={
            "goal": "Analyze customer data and create predictive model",
            "max_tasks": 20,
            "enable_optimization": True,
        },
        context={},
    )


@pytest.fixture
def sample_plan():
    """Sample plan for testing."""
    tasks = [
        Task("task_1", "Design architecture", TaskPriority.HIGH, 1000.0),
        Task("task_2", "Implement core logic", TaskPriority.HIGH, 2000.0, dependencies=["task_1"]),
        Task("task_3", "Write tests", TaskPriority.MEDIUM, 1500.0, dependencies=["task_2"]),
        Task("task_4", "Deploy", TaskPriority.LOW, 500.0, dependencies=["task_3"]),
    ]

    dependencies = [
        Dependency("task_2", "task_1", DependencyType.REQUIRES, 1.0),
        Dependency("task_3", "task_2", DependencyType.REQUIRES, 1.0),
        Dependency("task_4", "task_3", DependencyType.REQUIRES, 1.0),
    ]

    execution_order = ["task_1", "task_2", "task_3", "task_4"]
    parallel_stages = [["task_1"], ["task_2"], ["task_3"], ["task_4"]]

    return Plan(
        goal="Test goal",
        tasks=tasks,
        dependencies=dependencies,
        execution_order=execution_order,
        parallel_stages=parallel_stages,
        estimated_total_duration_ms=5000.0,
    )


# ===== Initialization Tests =====

class TestInitialization:
    """Test PlanningDepartment initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        dept = PlanningDepartment()

        assert dept.department_id == "planning"
        assert len(dept._plan_history) == 0
        assert dept._validation_thresholds["max_tasks"] == 50
        assert dept._validation_thresholds["max_depth"] == 5

    def test_custom_initialization(self):
        """Test initialization with custom department_id."""
        dept = PlanningDepartment(department_id="custom_planning")

        assert dept.department_id == "custom_planning"
        assert dept._optimization_stats["total_optimizations"] == 0


# ===== Execute Method Tests =====

class TestExecuteMethod:
    """Test execute() method - goal decomposition and planning."""

    @pytest.mark.asyncio
    async def test_execute_simple_goal(self, planning_department, simple_goal_request):
        """Test execution with simple implementation goal."""
        response = await planning_department.execute(simple_goal_request)

        assert response.task_id == "test_001"
        assert "plan" in response.result
        assert response.confidence.score > 0.0

        plan = response.result["plan"]
        assert plan is not None
        assert "tasks" in plan
        assert len(plan["tasks"]) > 0

    @pytest.mark.asyncio
    async def test_execute_complex_goal(self, planning_department, complex_goal_request):
        """Test execution with complex analysis goal."""
        response = await planning_department.execute(complex_goal_request)

        assert response.task_id == "test_002"
        plan = response.result["plan"]

        # Complex goal should decompose into multiple tasks
        assert len(plan["tasks"]) >= 3
        assert "dependencies" in plan
        assert "execution_order" in plan

    @pytest.mark.asyncio
    async def test_execute_missing_goal(self, planning_department):
        """Test execution with missing goal raises ValueError."""
        request = DepartmentRequest(
            task_id="test_003",
            parameters={},  # Missing 'goal'
            context={},
        )

        with pytest.raises(ValueError, match="must contain 'goal' field"):
            await planning_department.execute(request)

    @pytest.mark.asyncio
    async def test_execute_empty_goal(self, planning_department):
        """Test execution with empty goal raises ValueError."""
        request = DepartmentRequest(
            task_id="test_004",
            parameters={"goal": ""},
            context={},
        )

        with pytest.raises(ValueError, match="must contain 'goal' field"):
            await planning_department.execute(request)

    @pytest.mark.asyncio
    async def test_execute_with_optimization_disabled(self, planning_department):
        """Test execution with optimization disabled."""
        request = DepartmentRequest(
            task_id="test_005",
            parameters={
                "goal": "Implement feature",
                "enable_optimization": False,
            },
            context={},
        )

        response = await planning_department.execute(request)
        plan = response.result["plan"]

        # Without optimization, each task in separate stage
        assert len(plan["parallel_stages"]) == len(plan["tasks"])

    @pytest.mark.asyncio
    async def test_execute_with_max_tasks_limit(self, planning_department):
        """Test execution respects max_tasks limit."""
        request = DepartmentRequest(
            task_id="test_006",
            parameters={
                "goal": "Build large system",
                "max_tasks": 3,
            },
            context={},
        )

        response = await planning_department.execute(request)
        plan = response.result["plan"]

        # Should not exceed max_tasks
        assert len(plan["tasks"]) <= 3

    @pytest.mark.asyncio
    async def test_execute_tracks_plan_history(self, planning_department):
        """Test that execution tracks plans in history."""
        initial_count = len(planning_department._plan_history)

        request = DepartmentRequest(
            task_id="test_007",
            parameters={"goal": "Test goal"},
            context={},
        )

        await planning_department.execute(request)

        assert len(planning_department._plan_history) == initial_count + 1

    @pytest.mark.asyncio
    async def test_execute_confidence_metadata(self, planning_department, simple_goal_request):
        """Test that execution returns proper confidence metadata."""
        response = await planning_department.execute(simple_goal_request)

        confidence = response.confidence
        assert isinstance(confidence, ConfidenceMetadata)
        assert 0.0 <= confidence.score <= 1.0
        assert len(confidence.justification) > 0
        assert len(confidence.sources) > 0

    @pytest.mark.asyncio
    async def test_execute_metadata_fields(self, planning_department, simple_goal_request):
        """Test that execution returns complete metadata."""
        response = await planning_department.execute(simple_goal_request)

        assert "execution_time_ms" in response.metadata
        assert "num_tasks" in response.metadata
        assert "num_dependencies" in response.metadata
        assert "estimated_duration_ms" in response.metadata


# ===== Verify Method Tests =====

class TestVerifyMethod:
    """Test verify() method - plan validation."""

    @pytest.mark.asyncio
    async def test_verify_valid_plan(self, planning_department, simple_goal_request):
        """Test verification of valid plan."""
        response = await planning_department.execute(simple_goal_request)
        verification = await planning_department.verify(response)

        assert isinstance(verification, VerificationResult)
        assert len(verification.checks) == 5  # 5 validation checks
        assert verification.overall_score > 0.0

    @pytest.mark.asyncio
    async def test_verify_all_five_checks(self, planning_department, simple_goal_request):
        """Test that verify() performs all 5 quality checks."""
        response = await planning_department.execute(simple_goal_request)
        verification = await planning_department.verify(response)

        check_dimensions = [check.dimension for check in verification.checks]

        assert "Completeness" in check_dimensions
        assert "Feasibility" in check_dimensions
        assert "Optimality" in check_dimensions
        assert "Dependencies" in check_dimensions
        assert "Consistency" in check_dimensions

    @pytest.mark.asyncio
    async def test_verify_completeness_check(self, planning_department, simple_goal_request):
        """Test completeness validation check."""
        response = await planning_department.execute(simple_goal_request)
        verification = await planning_department.verify(response)

        completeness_check = next(
            (check for check in verification.checks if check.dimension == "Completeness"),
            None
        )

        assert completeness_check is not None
        assert isinstance(completeness_check, DSStarCheck)
        assert 0.0 <= completeness_check.score <= 1.0

    @pytest.mark.asyncio
    async def test_verify_empty_plan(self, planning_department):
        """Test verification handles empty plan."""
        response = DepartmentResponse(
            task_id="test_empty",
            result={"plan": None},
            confidence=ConfidenceMetadata.from_score(0.1),
            metadata={},
        )

        verification = await planning_department.verify(response)

        assert not verification.verified
        assert verification.overall_score == 0.0
        assert len(verification.recommendations) > 0

    @pytest.mark.asyncio
    async def test_verify_recommendations(self, planning_department):
        """Test that failed checks generate recommendations."""
        # Create response with low-quality plan
        response = DepartmentResponse(
            task_id="test_low_quality",
            result={
                "plan": {
                    "tasks": [],  # Empty tasks (fails completeness)
                    "dependencies": [],
                    "parallel_stages": [],
                }
            },
            confidence=ConfidenceMetadata.from_score(0.5),
            metadata={},
        )

        verification = await planning_department.verify(response)

        # Should have recommendations for failed checks
        if not verification.verified:
            assert len(verification.recommendations) > 0


# ===== Refine Method Tests =====

class TestRefineMethod:
    """Test refine() method - plan optimization."""

    @pytest.mark.asyncio
    async def test_refine_low_confidence_plan(self, planning_department):
        """Test refinement of low-confidence plan."""
        response = DepartmentResponse(
            task_id="test_refine",
            result={"plan": {"tasks": [{"task_id": "t1", "description": "Task 1"}]}},
            confidence=ConfidenceMetadata.from_score(0.6),  # Low confidence
            metadata={},
        )

        # Save original score before refinement (refine modifies in place)
        original_score = response.confidence.score
        refined = await planning_department.refine(response)

        # Confidence should improve
        assert refined.confidence.score > original_score
        assert "refinement" in refined.metadata

    @pytest.mark.asyncio
    async def test_refine_high_confidence_no_change(self, planning_department):
        """Test that high-confidence plans skip refinement."""
        response = DepartmentResponse(
            task_id="test_no_refine",
            result={"plan": {"tasks": []}},
            confidence=ConfidenceMetadata.from_score(0.9),  # High confidence
            metadata={},
        )

        refined = await planning_department.refine(response)

        # Should return unchanged
        assert refined.confidence.score == response.confidence.score

    @pytest.mark.asyncio
    async def test_refine_tracks_optimization_stats(self, planning_department):
        """Test that refinement updates optimization statistics."""
        initial_count = planning_department._optimization_stats["total_optimizations"]

        response = DepartmentResponse(
            task_id="test_stats",
            result={"plan": {"tasks": []}},
            confidence=ConfidenceMetadata.from_score(0.5),
            metadata={},
        )

        await planning_department.refine(response)

        assert planning_department._optimization_stats["total_optimizations"] == initial_count + 1


# ===== Update Strategy Method Tests =====

class TestUpdateStrategyMethod:
    """Test update_strategy() method - learning from execution."""

    @pytest.mark.asyncio
    async def test_update_strategy_successful_execution(self, planning_department):
        """Test learning from successful execution."""
        feedback = {
            "execution_success": True,
            "actual_duration": 5000,
            "estimated_duration": 4800,
        }

        # Should not raise exception
        await planning_department.update_strategy(feedback)

    @pytest.mark.asyncio
    async def test_update_strategy_failed_execution(self, planning_department):
        """Test learning from failed execution."""
        feedback = {
            "execution_success": False,
            "actual_duration": 10000,
            "estimated_duration": 5000,
            "bottlenecks": ["task_3", "task_5"],
        }

        # Should not raise exception
        await planning_department.update_strategy(feedback)

    @pytest.mark.asyncio
    async def test_update_strategy_duration_calibration(self, planning_department):
        """Test duration estimation calibration."""
        feedback = {
            "actual_duration": 6000,
            "estimated_duration": 5000,
        }

        await planning_department.update_strategy(feedback)
        # Should log duration error


# ===== Get Capabilities Method Tests =====

class TestGetCapabilitiesMethod:
    """Test get_capabilities() method - capability reporting."""

    @pytest.mark.asyncio
    async def test_get_capabilities_structure(self, planning_department):
        """Test capabilities structure is complete."""
        capabilities = await planning_department.get_capabilities()

        assert "tasks" in capabilities
        assert "constraints" in capabilities
        assert "features" in capabilities

    @pytest.mark.asyncio
    async def test_get_capabilities_tasks(self, planning_department):
        """Test supported tasks are listed."""
        capabilities = await planning_department.get_capabilities()

        tasks = capabilities["tasks"]
        assert "goal_decomposition" in tasks
        assert "dependency_detection" in tasks
        assert "plan_validation" in tasks
        assert "plan_optimization" in tasks

    @pytest.mark.asyncio
    async def test_get_capabilities_constraints(self, planning_department):
        """Test constraints are reported."""
        capabilities = await planning_department.get_capabilities()

        constraints = capabilities["constraints"]
        assert "max_tasks" in constraints
        assert "max_depth" in constraints
        assert "max_dependencies" in constraints
        assert constraints["max_tasks"] == 50

    @pytest.mark.asyncio
    async def test_get_capabilities_features(self, planning_department):
        """Test features are reported."""
        capabilities = await planning_department.get_capabilities()

        features = capabilities["features"]
        assert features["dependency_detection"] is True
        assert features["parallelization"] is True
        assert features["optimization"] is True
        assert features["learning"] is True


# ===== Get Metrics Method Tests =====

class TestGetMetricsMethod:
    """Test get_metrics() method - performance metrics."""

    @pytest.mark.asyncio
    async def test_get_metrics_empty_state(self, planning_department):
        """Test metrics with no plans executed."""
        metrics = await planning_department.get_metrics()

        assert "plan_stats" in metrics
        assert "optimization_stats" in metrics
        assert "decomposition_patterns" in metrics

    @pytest.mark.asyncio
    async def test_get_metrics_after_execution(self, planning_department, simple_goal_request):
        """Test metrics after executing plans."""
        await planning_department.execute(simple_goal_request)

        metrics = await planning_department.get_metrics()
        plan_stats = metrics["plan_stats"]

        assert plan_stats["total_plans"] >= 1
        assert "avg_tasks_per_plan" in plan_stats
        assert "avg_dependencies" in plan_stats
        assert "avg_parallel_stages" in plan_stats

    @pytest.mark.asyncio
    async def test_get_metrics_statistics(self, planning_department, simple_goal_request):
        """Test that statistics are calculated correctly."""
        # Execute multiple plans
        await planning_department.execute(simple_goal_request)

        request2 = DepartmentRequest(
            task_id="test_metrics_2",
            parameters={"goal": "Analyze data"},
            context={},
        )
        await planning_department.execute(request2)

        metrics = await planning_department.get_metrics()
        plan_stats = metrics["plan_stats"]

        assert plan_stats["total_plans"] == 2
        assert plan_stats["avg_tasks_per_plan"] > 0


# ===== Health Check Method Tests =====

class TestHealthCheckMethod:
    """Test health_check() method - system health."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self, planning_department):
        """Test health check succeeds (no external dependencies)."""
        health = await planning_department.health_check()

        assert health is True


# ===== Helper Method Tests =====

class TestHelperMethods:
    """Test private helper methods."""

    @pytest.mark.asyncio
    async def test_decompose_goal_implement_pattern(self, planning_department):
        """Test goal decomposition for 'implement' keyword."""
        tasks = await planning_department._decompose_goal("Implement new feature", max_tasks=10)

        assert len(tasks) > 0
        task_ids = [task.task_id for task in tasks]

        # Should follow implement pattern
        assert "design" in task_ids
        assert "code" in task_ids
        assert "test" in task_ids

    @pytest.mark.asyncio
    async def test_decompose_goal_analyze_pattern(self, planning_department):
        """Test goal decomposition for 'analyze' keyword."""
        tasks = await planning_department._decompose_goal("Analyze customer data", max_tasks=10)

        assert len(tasks) > 0
        task_ids = [task.task_id for task in tasks]

        # Should follow analyze pattern
        assert "gather_data" in task_ids
        assert "process" in task_ids

    @pytest.mark.asyncio
    async def test_decompose_goal_generic_pattern(self, planning_department):
        """Test goal decomposition for generic goals."""
        tasks = await planning_department._decompose_goal("Complete project", max_tasks=10)

        assert len(tasks) > 0
        # Should have at least plan, execute, verify
        assert len(tasks) >= 3

    @pytest.mark.asyncio
    async def test_detect_dependencies(self, planning_department):
        """Test dependency detection from tasks."""
        tasks = [
            Task("task_1", "First task", dependencies=[]),
            Task("task_2", "Second task", dependencies=["task_1"]),
            Task("task_3", "Third task", dependencies=["task_1", "task_2"]),
        ]

        dependencies = await planning_department._detect_dependencies(tasks)

        assert len(dependencies) == 3  # task_2→task_1, task_3→task_1, task_3→task_2
        assert all(isinstance(dep, Dependency) for dep in dependencies)

    @pytest.mark.asyncio
    async def test_topological_sort_linear(self, planning_department):
        """Test topological sort with linear dependencies."""
        tasks = [
            Task("task_1", "First"),
            Task("task_2", "Second", dependencies=["task_1"]),
            Task("task_3", "Third", dependencies=["task_2"]),
        ]

        dependencies = [
            Dependency("task_2", "task_1"),
            Dependency("task_3", "task_2"),
        ]

        order = await planning_department._topological_sort(tasks, dependencies)

        assert order == ["task_1", "task_2", "task_3"]

    @pytest.mark.asyncio
    async def test_topological_sort_parallel(self, planning_department):
        """Test topological sort with parallel tasks."""
        tasks = [
            Task("task_1", "First"),
            Task("task_2", "Parallel 1", dependencies=["task_1"]),
            Task("task_3", "Parallel 2", dependencies=["task_1"]),
            Task("task_4", "Final", dependencies=["task_2", "task_3"]),
        ]

        dependencies = [
            Dependency("task_2", "task_1"),
            Dependency("task_3", "task_1"),
            Dependency("task_4", "task_2"),
            Dependency("task_4", "task_3"),
        ]

        order = await planning_department._topological_sort(tasks, dependencies)

        # task_1 must be first, task_4 must be last
        assert order[0] == "task_1"
        assert order[-1] == "task_4"
        # task_2 and task_3 can be in any order (parallel)
        assert "task_2" in order[1:3]
        assert "task_3" in order[1:3]

    @pytest.mark.asyncio
    async def test_optimize_parallelization(self, planning_department):
        """Test parallelization optimization."""
        tasks = [
            Task("task_1", "Root"),
            Task("task_2", "Branch 1", dependencies=["task_1"]),
            Task("task_3", "Branch 2", dependencies=["task_1"]),
        ]

        dependencies = [
            Dependency("task_2", "task_1"),
            Dependency("task_3", "task_1"),
        ]

        execution_order = ["task_1", "task_2", "task_3"]

        stages = await planning_department._optimize_parallelization(tasks, dependencies, execution_order)

        # Should have 2 stages: [task_1] and [task_2, task_3] (parallel)
        assert len(stages) == 2
        assert stages[0] == ["task_1"]
        assert set(stages[1]) == {"task_2", "task_3"}

    @pytest.mark.asyncio
    async def test_estimate_duration(self, planning_department):
        """Test duration estimation with parallelization."""
        tasks = [
            Task("task_1", "Fast", estimated_duration_ms=100.0),
            Task("task_2", "Medium", estimated_duration_ms=500.0),
            Task("task_3", "Slow", estimated_duration_ms=1000.0),
        ]

        # All parallel (max duration = 1000)
        parallel_stages = [["task_1", "task_2", "task_3"]]

        duration = await planning_department._estimate_duration(tasks, parallel_stages)

        assert duration == 1000.0  # Max of parallel tasks

    @pytest.mark.asyncio
    async def test_estimate_duration_sequential(self, planning_department):
        """Test duration estimation with sequential execution."""
        tasks = [
            Task("task_1", "First", estimated_duration_ms=100.0),
            Task("task_2", "Second", estimated_duration_ms=200.0),
            Task("task_3", "Third", estimated_duration_ms=300.0),
        ]

        # All sequential
        sequential_stages = [["task_1"], ["task_2"], ["task_3"]]

        duration = await planning_department._estimate_duration(tasks, sequential_stages)

        assert duration == 600.0  # Sum of all tasks

    @pytest.mark.asyncio
    async def test_calculate_plan_confidence(self, planning_department, sample_plan):
        """Test plan confidence calculation."""
        confidence = await planning_department._calculate_plan_confidence(sample_plan)

        assert 0.0 <= confidence <= 1.0

    def test_serialize_plan(self, planning_department, sample_plan):
        """Test plan serialization to dictionary."""
        serialized = planning_department._serialize_plan(sample_plan)

        assert "goal" in serialized
        assert "tasks" in serialized
        assert "dependencies" in serialized
        assert "execution_order" in serialized
        assert "parallel_stages" in serialized
        assert "estimated_total_duration_ms" in serialized

        # Check task serialization
        assert len(serialized["tasks"]) == len(sample_plan.tasks)
        assert serialized["tasks"][0]["task_id"] == "task_1"


# ===== Integration Tests =====

class TestIntegration:
    """Test full planning cycle integration."""

    @pytest.mark.asyncio
    async def test_full_planning_cycle(self, planning_department):
        """Test complete cycle: execute → verify → refine."""
        # 1. Execute
        request = DepartmentRequest(
            task_id="integration_test",
            parameters={"goal": "Build recommendation system"},
            context={},
        )

        response = await planning_department.execute(request)
        assert "plan" in response.result

        # 2. Verify
        verification = await planning_department.verify(response)
        assert isinstance(verification, VerificationResult)

        # 3. Refine (if needed)
        if response.confidence.score < 0.85:
            refined = await planning_department.refine(response)
            assert refined.confidence.score >= response.confidence.score

    @pytest.mark.asyncio
    async def test_learning_cycle(self, planning_department):
        """Test execute → update_strategy learning cycle."""
        # Execute plan
        request = DepartmentRequest(
            task_id="learning_test",
            parameters={"goal": "Deploy application"},
            context={},
        )

        response = await planning_department.execute(request)
        plan = response.result["plan"]

        # Simulate execution feedback
        feedback = {
            "execution_success": True,
            "actual_duration": plan["estimated_total_duration_ms"] * 1.2,  # 20% over
            "estimated_duration": plan["estimated_total_duration_ms"],
        }

        await planning_department.update_strategy(feedback)

        # Should complete without error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
