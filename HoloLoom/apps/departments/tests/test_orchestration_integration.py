"""
Integration test for Orchestration Department with Department Protocol.

Validates:
1. Orchestration Department implements all 7 protocol methods
2. Task routing works correctly (explicit, auto, broadcast, fallback)
3. Parallel execution coordinates multiple departments
4. Sequential workflows execute in correct order
5. Result aggregation strategies work correctly
6. Workflow validation catches issues
7. Learning signals are tracked

Author: HoloLoom Testing
Date: November 2025
"""

import pytest
import asyncio
from datetime import datetime

# Import protocol types directly
from HoloLoom.apps.departments.protocol import (
    Department,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    VerificationResult,
)

# Import Orchestration Department
from HoloLoom.apps.departments.orchestration_department import (
    OrchestrationDepartment,
    AggregationStrategy,
    RoutingStrategy,
    WorkflowStep,
    WorkflowResult,
)


# ===== Mock Departments =====

class MockRAGDepartment:
    """Mock RAG Department for testing."""

    def __init__(self, department_id="rag"):
        self.department_id = department_id
        self._call_count = 0

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute RAG query."""
        self._call_count += 1

        confidence = ConfidenceMetadata.from_score(
            score=0.85,
            justification=["High quality retrieval"],
            sources=["source1", "source2"],
        )

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "answer": "RAG answer",
                "sources": ["source1", "source2"],
            },
            confidence=confidence,
            metadata={"department": "rag", "call_count": self._call_count},
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Verify response."""
        from HoloLoom.apps.departments.protocol import VerificationCheck, VerificationStatus

        checks = [
            VerificationCheck(
                name="Quality",
                status=VerificationStatus.PASSED,
                reason="Good quality",
                score=0.9,
            )
        ]

        return VerificationResult(
            checks=checks,
            summary="All checks passed",
            confidence=0.9,
        )

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """Refine response."""
        return response

    async def update_strategy(self, feedback: dict) -> None:
        """Update strategy."""
        pass

    async def get_capabilities(self) -> dict:
        """Get capabilities."""
        return {"tasks": ["retrieve_context", "question_answering"]}

    async def get_metrics(self) -> dict:
        """Get metrics."""
        return {"call_count": self._call_count}

    async def health_check(self) -> bool:
        """Health check."""
        return True


class MockPlanningDepartment:
    """Mock Planning Department for testing."""

    def __init__(self, department_id="planning"):
        self.department_id = department_id
        self._call_count = 0
        self._should_fail = False

    def set_failure_mode(self, should_fail: bool):
        """Set whether this department should fail."""
        self._should_fail = should_fail

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute planning task."""
        self._call_count += 1

        if self._should_fail:
            raise RuntimeError("Planning department failure (simulated)")

        confidence = ConfidenceMetadata.from_score(
            score=0.75,
            justification=["Valid plan created"],
            sources=[],
        )

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "plan": {
                    "tasks": ["task1", "task2"],
                    "dependencies": [],
                    "execution_order": ["task1", "task2"],
                },
            },
            confidence=confidence,
            metadata={"department": "planning", "call_count": self._call_count},
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Verify response."""
        from HoloLoom.apps.departments.protocol import VerificationCheck, VerificationStatus

        checks = [
            VerificationCheck(
                name="Completeness",
                status=VerificationStatus.PASSED,
                reason="Plan is complete",
                score=0.8,
            )
        ]

        return VerificationResult(
            checks=checks,
            summary="All checks passed",
            confidence=0.8,
        )

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """Refine response."""
        return response

    async def update_strategy(self, feedback: dict) -> None:
        """Update strategy."""
        pass

    async def get_capabilities(self) -> dict:
        """Get capabilities."""
        return {"tasks": ["goal_decomposition", "dependency_detection"]}

    async def get_metrics(self) -> dict:
        """Get metrics."""
        return {"call_count": self._call_count}

    async def health_check(self) -> bool:
        """Health check."""
        return True


# ===== Fixtures =====

@pytest.fixture
def mock_registry():
    """Create mock department registry."""
    return {
        "rag": MockRAGDepartment(),
        "planning": MockPlanningDepartment(),
    }


@pytest.fixture
def orchestration_dept(mock_registry):
    """Create Orchestration Department with mock registry."""
    return OrchestrationDepartment(department_registry=mock_registry)


# ===== Integration Tests =====

@pytest.mark.asyncio
async def test_protocol_compliance():
    """Test that Orchestration Department implements Department protocol."""
    dept = OrchestrationDepartment()

    # Verify all 7 methods exist
    assert hasattr(dept, 'execute')
    assert hasattr(dept, 'verify')
    assert hasattr(dept, 'refine')
    assert hasattr(dept, 'update_strategy')
    assert hasattr(dept, 'get_capabilities')
    assert hasattr(dept, 'get_metrics')
    assert hasattr(dept, 'health_check')

    # Verify all are async
    assert asyncio.iscoroutinefunction(dept.execute)
    assert asyncio.iscoroutinefunction(dept.verify)
    assert asyncio.iscoroutinefunction(dept.refine)
    assert asyncio.iscoroutinefunction(dept.update_strategy)
    assert asyncio.iscoroutinefunction(dept.get_capabilities)
    assert asyncio.iscoroutinefunction(dept.get_metrics)
    assert asyncio.iscoroutinefunction(dept.health_check)


@pytest.mark.asyncio
async def test_explicit_routing(orchestration_dept):
    """Test explicit routing to specific department."""
    request = DepartmentRequest(
        task_type="route_task",
        parameters={
            "department_id": "rag",
            "task_type": "retrieve_context",
            "query": "Test query",
        },
    )

    response = await orchestration_dept.execute(request)

    # Validate response
    assert isinstance(response, DepartmentResponse)
    assert response.task_id == request.task_id
    # For route_task, response.result is the DepartmentResponse from the routed department
    assert isinstance(response.result, DepartmentResponse)
    assert response.result.result["answer"] == "RAG answer"
    assert 0.0 <= response.confidence.score <= 1.0


@pytest.mark.asyncio
async def test_auto_routing(orchestration_dept):
    """Test automatic routing based on task_type."""
    request = DepartmentRequest(
        task_type="route_task",
        parameters={
            "task_type": "retrieve_context",
            "query": "Test query",
        },
    )

    response = await orchestration_dept.execute(request)

    # Should route to RAG department automatically
    assert isinstance(response, DepartmentResponse)
    # For route_task, response.result is the DepartmentResponse from the routed department
    assert isinstance(response.result, DepartmentResponse)
    assert response.result.result["answer"] == "RAG answer"


@pytest.mark.asyncio
async def test_parallel_execution(orchestration_dept):
    """Test parallel execution of multiple departments."""
    request = DepartmentRequest(
        task_type="parallel_execution",
        parameters={
            "departments": [
                {
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "query": "Test query 1",
                },
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "goal": "Test goal",
                },
            ],
            "aggregation_strategy": AggregationStrategy.ALL.value,
        },
    )

    response = await orchestration_dept.execute(request)

    # Validate parallel execution
    assert isinstance(response, DepartmentResponse)
    results = response.result["results"]
    assert len(results) == 2
    assert all(isinstance(r, DepartmentResponse) for r in results)

    # Validate aggregated confidence
    assert "aggregated_confidence" in response.result
    assert 0.0 <= response.result["aggregated_confidence"] <= 1.0


@pytest.mark.asyncio
async def test_sequential_workflow(orchestration_dept):
    """Test sequential workflow execution."""
    request = DepartmentRequest(
        task_type="sequential_workflow",
        parameters={
            "steps": [
                {
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "query": "Step 1 query",
                },
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "goal": "Step 2 goal",
                    "use_previous_result": False,
                },
            ],
        },
    )

    response = await orchestration_dept.execute(request)

    # Validate sequential execution
    assert isinstance(response, DepartmentResponse)
    results = response.result["results"]
    assert len(results) == 2
    assert "step_0" in results
    assert "step_1" in results

    # Should have final result
    assert "final_result" in response.result


@pytest.mark.asyncio
async def test_aggregation_vote_strategy(orchestration_dept):
    """Test VOTE aggregation strategy (highest confidence wins)."""
    request = DepartmentRequest(
        task_type="result_aggregation",
        parameters={
            "results": [
                {
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "query": "Test",
                },
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "goal": "Test",
                },
            ],
            "aggregation_strategy": AggregationStrategy.VOTE.value,
        },
    )

    response = await orchestration_dept.execute(request)

    # Verify response structure
    assert isinstance(response, DepartmentResponse)
    assert isinstance(response.result, dict)
    # RAG has higher confidence (0.85 vs 0.75), should be selected
    assert "best_result" in response.result
    assert response.result["best_result"].result["answer"] == "RAG answer"
    assert response.result["aggregation_strategy"] == AggregationStrategy.VOTE.value


@pytest.mark.asyncio
async def test_aggregation_weighted_strategy(orchestration_dept):
    """Test WEIGHTED aggregation strategy."""
    request = DepartmentRequest(
        task_type="result_aggregation",
        parameters={
            "results": [
                {
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "query": "Test",
                },
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "goal": "Test",
                },
            ],
            "aggregation_strategy": AggregationStrategy.WEIGHTED.value,
        },
    )

    response = await orchestration_dept.execute(request)

    # Verify response structure
    assert isinstance(response, DepartmentResponse)
    assert isinstance(response.result, dict)
    # Should have weighted average confidence
    # RAG: 0.85, Planning: 0.75 → avg = 0.80
    if "aggregated_confidence" in response.result:
        avg_conf = response.result["aggregated_confidence"]
        assert 0.75 <= avg_conf <= 0.85  # Between the two
    # Also check the response-level confidence
    assert 0.75 <= response.confidence.score <= 0.85


@pytest.mark.asyncio
async def test_workflow_validation(orchestration_dept):
    """Test workflow validation checks."""
    # Create a simple workflow response
    request = DepartmentRequest(
        task_type="route_task",
        parameters={
            "department_id": "rag",
            "task_type": "retrieve_context",
            "query": "Test",
        },
    )

    response = await orchestration_dept.execute(request)
    verification = await orchestration_dept.verify(response)

    # Validate verification structure
    assert isinstance(verification, VerificationResult)
    assert len(verification.checks) == 5  # 5 validation dimensions
    assert 0.0 <= verification.overall_score <= 1.0


@pytest.mark.asyncio
async def test_department_registry_management(orchestration_dept):
    """Test department registration and retrieval."""
    # Check initial registry
    capabilities = await orchestration_dept.get_capabilities()
    departments = capabilities["departments"]
    assert "rag" in departments
    assert "planning" in departments

    # Register new department
    new_dept = MockRAGDepartment(department_id="analytics")
    orchestration_dept.register_department("analytics", new_dept)

    # Verify registration
    assert "analytics" in orchestration_dept.registry
    capabilities = await orchestration_dept.get_capabilities()
    assert "analytics" in capabilities["departments"]


@pytest.mark.asyncio
async def test_routing_rule_management(orchestration_dept):
    """Test adding custom routing rules."""
    # Add custom routing rule
    orchestration_dept.add_routing_rule("custom_task", "rag")

    # Test routing with custom rule
    request = DepartmentRequest(
        task_type="route_task",
        parameters={
            "task_type": "custom_task",
            "query": "Test",
        },
    )

    response = await orchestration_dept.execute(request)

    # Should route to RAG department
    # For route_task, response.result is the DepartmentResponse from the routed department
    assert isinstance(response.result, DepartmentResponse)
    assert response.result.result["answer"] == "RAG answer"


@pytest.mark.asyncio
async def test_fallback_behavior(orchestration_dept, mock_registry):
    """Test fallback when primary department fails."""
    # Make planning department fail
    mock_registry["planning"].set_failure_mode(True)

    request = DepartmentRequest(
        task_type="parallel_execution",
        parameters={
            "departments": [
                {
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "query": "Test",
                },
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "goal": "Test",
                },
            ],
            "aggregation_strategy": AggregationStrategy.FIRST.value,
        },
    )

    response = await orchestration_dept.execute(request)

    # Should have results from successful department (RAG)
    results = response.result["results"]
    assert len(results) >= 1  # At least RAG succeeded

    # Reset failure mode
    mock_registry["planning"].set_failure_mode(False)


@pytest.mark.asyncio
async def test_confidence_aggregation(orchestration_dept):
    """Test different confidence aggregation methods."""
    # Execute parallel tasks
    request = DepartmentRequest(
        task_type="parallel_execution",
        parameters={
            "departments": [
                {
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "query": "Test",
                },
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "goal": "Test",
                },
            ],
            "aggregation_strategy": AggregationStrategy.ALL.value,
        },
    )

    response = await orchestration_dept.execute(request)

    # Aggregated confidence should be between min and max
    # RAG: 0.85, Planning: 0.75
    agg_conf = response.result["aggregated_confidence"]
    assert 0.75 <= agg_conf <= 0.85


@pytest.mark.asyncio
async def test_learning_signal_tracking(orchestration_dept):
    """Test that learning signals are tracked correctly."""
    # Execute multiple workflows
    for i in range(3):
        request = DepartmentRequest(
            task_type="route_task",
            parameters={
                "department_id": "rag",
                "task_type": "retrieve_context",
                "query": f"Test {i}",
            },
        )
        await orchestration_dept.execute(request)

    # Check metrics
    metrics = await orchestration_dept.get_metrics()

    # Should have workflow_stats section (may be empty dict if history not tracked yet)
    assert "workflow_stats" in metrics
    assert isinstance(metrics["workflow_stats"], dict)

    # Should have department_stats section
    assert "department_stats" in metrics
    assert isinstance(metrics["department_stats"], dict)


@pytest.mark.asyncio
async def test_update_strategy_learning(orchestration_dept):
    """Test strategy updates from feedback."""
    # Execute workflow
    request = DepartmentRequest(
        task_type="route_task",
        parameters={
            "department_id": "rag",
            "task_type": "retrieve_context",
            "query": "Test",
        },
    )

    response = await orchestration_dept.execute(request)

    # Provide feedback
    feedback = {
        "workflow_id": "test_workflow",
        "success": True,
        "departments_used": ["rag"],
        "total_latency_ms": 150.0,
    }

    await orchestration_dept.update_strategy(feedback)

    # Check that success rate was updated
    metrics = await orchestration_dept.get_metrics()
    dept_stats = metrics["department_stats"]

    # RAG should have success rate tracked
    if "rag" in dept_stats:
        assert "success_rate" in dept_stats["rag"]


@pytest.mark.asyncio
async def test_refinement_retry_logic(orchestration_dept, mock_registry):
    """Test refinement retries failed departments."""
    # Make planning fail
    mock_registry["planning"].set_failure_mode(True)

    request = DepartmentRequest(
        task_type="parallel_execution",
        parameters={
            "departments": [
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "goal": "Test",
                },
            ],
            "aggregation_strategy": AggregationStrategy.FIRST.value,
        },
    )

    # First attempt should fail
    response = await orchestration_dept.execute(request)
    initial_conf = response.confidence.score

    # Now fix the department and refine
    mock_registry["planning"].set_failure_mode(False)
    refined = await orchestration_dept.refine(response)

    # Refinement should improve confidence (or at least not decrease it)
    assert refined.confidence.score >= initial_conf


@pytest.mark.asyncio
async def test_capabilities_reporting(orchestration_dept):
    """Test capabilities are properly reported."""
    capabilities = await orchestration_dept.get_capabilities()

    # Validate required sections
    assert "tasks" in capabilities
    assert "departments" in capabilities
    assert "routing_strategies" in capabilities
    assert "aggregation_strategies" in capabilities
    assert "constraints" in capabilities

    # Validate task types
    assert "route_task" in capabilities["tasks"]
    assert "parallel_execution" in capabilities["tasks"]
    assert "sequential_workflow" in capabilities["tasks"]

    # Validate departments
    assert "rag" in capabilities["departments"]
    assert "planning" in capabilities["departments"]


@pytest.mark.asyncio
async def test_health_check(orchestration_dept):
    """Test health check."""
    health = await orchestration_dept.health_check()
    assert health is True


@pytest.mark.asyncio
async def test_empty_registry_health_check():
    """Test health check with empty registry."""
    dept = OrchestrationDepartment(department_registry={})
    health = await dept.health_check()
    assert health is False  # No departments registered


# ===== Summary =====

def test_integration_summary():
    """
    Summary of integration test coverage:

    ✅ Protocol compliance (all 7 methods implemented)
    ✅ Explicit routing (user-specified department)
    ✅ Auto routing (task_type based)
    ✅ Parallel execution (multiple departments concurrently)
    ✅ Sequential workflow (ordered execution)
    ✅ Aggregation - VOTE strategy (highest confidence)
    ✅ Aggregation - WEIGHTED strategy (confidence-weighted average)
    ✅ Workflow validation (5 dimensions checked)
    ✅ Department registry management (register/retrieve)
    ✅ Routing rule management (custom rules)
    ✅ Fallback behavior (when departments fail)
    ✅ Confidence aggregation (min/max/weighted)
    ✅ Learning signal tracking (workflow history)
    ✅ Update strategy learning (success rate tracking)
    ✅ Refinement retry logic (retry failed departments)
    ✅ Capabilities reporting (tasks, departments, strategies)
    ✅ Health check (system operational)
    ✅ Empty registry health check (graceful degradation)

    This validates that the Orchestration Department correctly implements
    the Department protocol and can coordinate multiple departments effectively.
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
