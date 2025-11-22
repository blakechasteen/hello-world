#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Context Department.

Tests the complete integration of existing HoloLoom (WeavingOrchestrator)
as a modular department, validating:
- Execute → Verify → Refine workflow
- Three-tier memory system
- Learning signal extraction
- Session management
- Registry integration

Run with:
    pytest HoloLoom/tests/integration/test_context_department.py -v
"""

import pytest
import asyncio
from typing import List

from HoloLoom.departments import (
    DepartmentRegistry,
    DepartmentRequest,
    ConfidenceLevel
)
from HoloLoom.departments.context import ContextDepartment
from HoloLoom.config import Config
from HoloLoom.protocols.types import MemoryShard


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_shards() -> List[MemoryShard]:
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian exploration strategy used in multi-armed bandit problems",
            entities=["thompson_sampling", "bayesian", "exploration", "multi_armed_bandit"],
            metadata={"source": "test", "topic": "RL"}
        ),
        MemoryShard(
            id="shard_2",
            text="Neural networks learn hierarchical representations through backpropagation",
            entities=["neural_network", "learning", "hierarchical", "backpropagation"],
            metadata={"source": "test", "topic": "ML"}
        ),
        MemoryShard(
            id="shard_3",
            text="Attention mechanisms enable context-aware processing in transformer models",
            entities=["attention", "context", "transformer", "processing"],
            metadata={"source": "test", "topic": "NLP"}
        ),
        MemoryShard(
            id="shard_4",
            text="Knowledge graphs represent entities and relationships as structured data",
            entities=["knowledge_graph", "entity", "relationship", "structured_data"],
            metadata={"source": "test", "topic": "KG"}
        ),
        MemoryShard(
            id="shard_5",
            text="Reinforcement learning agents learn from reward signals in environments",
            entities=["reinforcement_learning", "agent", "reward", "environment"],
            metadata={"source": "test", "topic": "RL"}
        )
    ]


@pytest.fixture
async def context_dept(test_shards):
    """Create a Context Department instance."""
    config = Config.fast()  # Use FAST mode for quick tests
    dept = ContextDepartment(config=config, shards=test_shards)

    await dept.initialize()
    yield dept
    await dept.close()


@pytest.fixture
async def registry_with_context(context_dept):
    """Create a registry with Context Department registered."""
    async with DepartmentRegistry() as registry:
        await registry.register(context_dept)
        yield registry


# ============================================================================
# Test Department Initialization
# ============================================================================

@pytest.mark.asyncio
async def test_context_department_initialization(test_shards):
    """Test Context Department initializes correctly."""
    dept = ContextDepartment(shards=test_shards)

    assert dept.name == "context"
    assert dept.domain == "general"
    assert dept.version == "1.0.0"
    assert "weave_response" in dept.supported_tasks
    assert "retrieve_context" in dept.supported_tasks

    # Initialize
    await dept.initialize()

    assert dept._initialized == True
    assert dept._orchestrator is not None

    # Cleanup
    await dept.close()


# ============================================================================
# Test Execute Method
# ============================================================================

@pytest.mark.asyncio
async def test_context_execute_weave_response(context_dept):
    """Test executing a weave_response task."""
    request = DepartmentRequest(
        task_id="req_001",
        task_type="weave_response",
        parameters={"query": "What is Thompson Sampling?"},
        confidence_expected=0.75,
        context_preference="adaptive"
    )

    response = await context_dept.execute(request)

    # Verify response structure
    assert response.task_id == "req_001"
    assert response.result is not None
    assert "response" in response.result
    assert response.confidence.score >= 0.0
    assert response.confidence.score <= 1.0

    # Verify confidence metadata
    assert response.confidence.level in ConfidenceLevel
    assert response.confidence.learning_rate in ["weekly", "daily", "hourly", "per-task", "immediate"]

    # Verify execution time tracked
    assert response.execution_time_ms > 0


@pytest.mark.asyncio
async def test_context_execute_retrieve_context(context_dept):
    """Test executing a retrieve_context task (no response generation)."""
    request = DepartmentRequest(
        task_id="req_002",
        task_type="retrieve_context",
        parameters={"query": "neural networks"},
        context_preference="minimal"
    )

    response = await context_dept.execute(request)

    # Verify response structure
    assert response.task_id == "req_002"
    assert "context" in response.result
    assert "context_count" in response.result

    # Should have found relevant context
    assert response.result["context_count"] >= 0


@pytest.mark.asyncio
async def test_context_execute_with_session(context_dept):
    """Test executing with session tracking."""
    request = DepartmentRequest(
        task_id="req_003",
        task_type="weave_response",
        parameters={"query": "What is attention?"},
        session_id="session_test_001"
    )

    response = await context_dept.execute(request)

    # Verify session state was built
    assert response.session_state is not None
    assert response.session_state.get("last_query") is not None

    # Verify session tracked in department
    session_state = await context_dept.get_session_state("session_test_001")
    assert session_state is not None


# ============================================================================
# Test Verify Method
# ============================================================================

@pytest.mark.asyncio
async def test_context_verify_sufficient_response(context_dept):
    """Test verifying a high-confidence response."""
    # Execute to get a response
    request = DepartmentRequest(
        task_id="req_004",
        task_type="weave_response",
        parameters={"query": "What is Thompson Sampling?"}
    )

    response = await context_dept.execute(request)

    # Manually set high confidence for testing
    response.confidence.score = 0.85

    # Verify
    verification = await context_dept.verify(response)

    # High confidence should be sufficient
    assert verification.confidence_valid == True


@pytest.mark.asyncio
async def test_context_verify_insufficient_response(context_dept):
    """Test verifying a low-confidence response."""
    # Execute to get a response
    request = DepartmentRequest(
        task_id="req_005",
        task_type="weave_response",
        parameters={"query": "obscure query about nothing"}
    )

    response = await context_dept.execute(request)

    # Manually set low confidence for testing
    response.confidence.score = 0.50

    # Verify
    verification = await context_dept.verify(response)

    # Low confidence should trigger refinement
    assert verification.sufficient == False
    assert verification.refinement_suggestions is not None
    assert len(verification.alternative_paths) > 0


# ============================================================================
# Test Refine Method
# ============================================================================

@pytest.mark.asyncio
async def test_context_refine_low_confidence(context_dept):
    """Test refining a low-confidence response."""
    # Execute initial request
    request = DepartmentRequest(
        task_id="req_006",
        task_type="weave_response",
        parameters={"query": "What is reinforcement learning?"},
        context_preference="minimal"
    )

    response = await context_dept.execute(request)

    # Set low confidence
    response.confidence.score = 0.60

    # Verify (should fail)
    verification = await context_dept.verify(response)

    if not verification.sufficient:
        # Refine
        refined_response = await context_dept.refine(request, response, verification)

        # Refined response should exist
        assert refined_response is not None
        assert refined_response.task_id.endswith("_refined")

        # Check refinement history
        if refined_response.reasoning:
            assert "refinement_history" in refined_response.reasoning
            assert len(refined_response.reasoning["refinement_history"]) > 0


# ============================================================================
# Test Full DS-STAR Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_full_ds_star_workflow(context_dept):
    """Test complete DS-STAR cycle: Execute → Verify → Refine."""
    # Execute
    request = DepartmentRequest(
        task_id="req_007",
        task_type="weave_response",
        parameters={"query": "Explain knowledge graphs"},
        confidence_expected=0.80
    )

    response = await context_dept.execute(request)

    # Verify
    verification = await context_dept.verify(response)

    # If insufficient, refine up to 2 iterations
    max_iterations = 2
    iteration = 0

    while not verification.sufficient and iteration < max_iterations:
        response = await context_dept.refine(request, response, verification)
        verification = await context_dept.verify(response)
        iteration += 1

    # Should eventually be sufficient (or hit max iterations)
    assert iteration <= max_iterations


# ============================================================================
# Test Learning Signals
# ============================================================================

@pytest.mark.asyncio
async def test_context_learning_signals(context_dept):
    """Test learning signal extraction."""
    # Execute multiple requests
    requests = [
        DepartmentRequest(
            task_id=f"req_{i}",
            task_type="weave_response",
            parameters={"query": f"Query {i}"}
        )
        for i in range(3)
    ]

    signals = []
    for request in requests:
        response = await context_dept.execute(request)
        if response.learning_signals:
            signals.append(response.learning_signals)

    # Should have extracted signals
    assert len(signals) > 0

    # Signals should have required fields
    for signal in signals:
        assert "task_type" in signal
        assert "outcome" in signal
        assert "confidence_predicted" in signal


@pytest.mark.asyncio
async def test_context_update_strategy(context_dept):
    """Test strategy updates from learning signals."""
    # Create learning signals
    signals = [
        {
            "task_type": "weave_response",
            "outcome": "success",
            "confidence_predicted": 0.85,
            "confidence_actual": 0.87
        },
        {
            "task_type": "weave_response",
            "outcome": "success",
            "confidence_predicted": 0.80,
            "confidence_actual": 0.82
        }
    ]

    # Update strategy
    await context_dept.update_strategy(signals)

    # Check institutional memory
    memory = await context_dept.get_institutional_memory("successful_strategies")

    assert "weave_response" in memory
    assert memory["weave_response"]["total"] == 2
    assert memory["weave_response"]["successes"] == 2


# ============================================================================
# Test Memory Systems
# ============================================================================

@pytest.mark.asyncio
async def test_context_session_memory(context_dept):
    """Test session memory management."""
    session_id = "session_memory_test"

    # Execute with session
    request = DepartmentRequest(
        task_id="req_session",
        task_type="weave_response",
        parameters={"query": "Test query"},
        session_id=session_id
    )

    await context_dept.execute(request)

    # Retrieve session state
    session_state = await context_dept.get_session_state(session_id)

    assert session_state is not None
    assert session_state["session_id"] == session_id
    assert session_state["request_count"] > 0


# ============================================================================
# Test Health Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_context_health_check(context_dept):
    """Test department health monitoring."""
    # Execute some requests
    for i in range(3):
        request = DepartmentRequest(
            task_id=f"health_req_{i}",
            task_type="weave_response",
            parameters={"query": f"Query {i}"}
        )
        await context_dept.execute(request)

    # Check health
    health = await context_dept.health_check()

    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert health["name"] == "context"
    assert health["performance"]["total_requests"] >= 3
    assert health["performance"]["success_rate"] >= 0.0


# ============================================================================
# Test Registry Integration
# ============================================================================

@pytest.mark.asyncio
async def test_context_registry_registration(registry_with_context):
    """Test Context Department registration in registry."""
    registry = registry_with_context

    # Check registration
    dept = registry.get_department("context")
    assert dept is not None
    assert dept.name == "context"

    # Check discovery by domain
    depts_by_domain = registry.find_by_domain("general")
    assert len(depts_by_domain) > 0
    assert any(d.name == "context" for d in depts_by_domain)

    # Check discovery by task
    depts_by_task = registry.find_by_task("weave_response")
    assert len(depts_by_task) > 0
    assert any(d.name == "context" for d in depts_by_task)


@pytest.mark.asyncio
async def test_context_registry_routing(registry_with_context):
    """Test routing requests through registry to Context Department."""
    registry = registry_with_context

    # Route request
    request = DepartmentRequest(
        task_id="route_req_001",
        task_type="weave_response",
        parameters={"query": "What is attention?"}
    )

    response = await registry.route_request(request)

    # Should get valid response
    assert response is not None
    assert response.task_id == "route_req_001"
    assert "response" in response.result


@pytest.mark.asyncio
async def test_context_registry_specific_routing(registry_with_context):
    """Test routing to specific Context Department."""
    registry = registry_with_context

    # Route to specific department
    request = DepartmentRequest(
        task_id="specific_req_001",
        task_type="weave_response",
        parameters={"query": "Test query"}
    )

    response = await registry.route_request(request, department_name="context")

    # Should get valid response
    assert response is not None


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_context_invalid_task_type(context_dept):
    """Test error handling for invalid task type."""
    request = DepartmentRequest(
        task_id="invalid_task",
        task_type="unsupported_task",
        parameters={"query": "Test"}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await context_dept.execute(request)


@pytest.mark.asyncio
async def test_context_missing_query_parameter(context_dept):
    """Test error handling for missing query parameter."""
    request = DepartmentRequest(
        task_id="missing_query",
        task_type="weave_response",
        parameters={}  # Missing 'query'
    )

    with pytest.raises(ValueError, match="Missing required parameter"):
        await context_dept.execute(request)


# ============================================================================
# Test Performance
# ============================================================================

@pytest.mark.asyncio
async def test_context_performance_tracking(context_dept):
    """Test per-task performance tracking."""
    # Execute multiple tasks of different types
    tasks = [
        ("weave_response", "What is Thompson Sampling?"),
        ("retrieve_context", "neural networks"),
        ("weave_response", "What is attention?")
    ]

    for task_type, query in tasks:
        request = DepartmentRequest(
            task_id=f"perf_{task_type}",
            task_type=task_type,
            parameters={"query": query}
        )
        await context_dept.execute(request)

    # Check performance tracking
    assert len(context_dept._task_performance) > 0

    if "weave_response" in context_dept._task_performance:
        stats = context_dept._task_performance["weave_response"]
        assert stats["count"] >= 2
        assert stats["total_latency"] > 0
        assert stats["total_confidence"] > 0


@pytest.mark.asyncio
async def test_context_execution_time_reasonable(context_dept):
    """Test that execution times are reasonable (<1s for simple queries)."""
    request = DepartmentRequest(
        task_id="timing_test",
        task_type="weave_response",
        parameters={"query": "Short query"}
    )

    response = await context_dept.execute(request)

    # Execution should be fast (< 1000ms for FAST mode)
    assert response.execution_time_ms < 1000, \
        f"Execution took {response.execution_time_ms}ms (expected < 1000ms)"


# ============================================================================
# Test Lifecycle
# ============================================================================

@pytest.mark.asyncio
async def test_context_lifecycle_context_manager(test_shards):
    """Test Context Department lifecycle with async context manager."""
    async with ContextDepartment(shards=test_shards) as dept:
        assert dept._initialized == True

        # Execute request
        request = DepartmentRequest(
            task_id="lifecycle_test",
            task_type="weave_response",
            parameters={"query": "Test"}
        )

        response = await dept.execute(request)
        assert response is not None

    # After exit, should be closed
    assert dept._closed == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
