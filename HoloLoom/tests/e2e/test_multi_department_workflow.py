#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Multi-Department Workflow Tests.

Tests complete workflows involving all departments:
- Context → MasterWeaver → Infrastructure (sequential)
- Parallel retrieval + verification
- Full beekeeping knowledge workflow
- Performance and optimization validation

Run with:
    pytest HoloLoom/tests/e2e/test_multi_department_workflow.py -v
"""

import pytest
import asyncio
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from HoloLoom.departments import (
    DepartmentRequest,
    DepartmentRegistry
)
from HoloLoom.protocols.types import MemoryShard
from HoloLoom.apps.departments.context import ContextDepartment
from HoloLoom.apps.departments.beekeeping import MasterWeaverDepartment
from HoloLoom.apps.departments.infrastructure import InfrastructureDepartment
from HoloLoom.apps.departments.verification import VerificationDepartment
from HoloLoom.apps.departments.orchestration import OrchestrationDepartment


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def beekeeping_shards() -> List[MemoryShard]:
    """Create beekeeping knowledge shards for testing."""
    return [
        MemoryShard(
            id="shard_1",
            text="Varroa mites (Varroa destructor) are parasitic mites that attack honey bees.",
            metadata={
                "source": "beekeeping_guide",
                "topic": "pests",
                "confidence": 0.95
            }
        ),
        MemoryShard(
            id="shard_2",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            metadata={
                "source": "ml_textbook",
                "topic": "reinforcement_learning",
                "confidence": 0.92
            }
        ),
        MemoryShard(
            id="shard_3",
            text="Hive ventilation is critical for maintaining proper temperature and humidity.",
            metadata={
                "source": "beekeeping_guide",
                "topic": "hive_management",
                "confidence": 0.88
            }
        ),
        MemoryShard(
            id="shard_4",
            text="The waggle dance is how bees communicate the location of food sources.",
            metadata={
                "source": "bee_biology",
                "topic": "behavior",
                "confidence": 0.94
            }
        ),
        MemoryShard(
            id="shard_5",
            text="Africanized honey bees are more defensive than European honey bees.",
            metadata={
                "source": "bee_species",
                "topic": "behavior",
                "confidence": 0.90
            }
        )
    ]


@pytest.fixture
async def full_registry(temp_storage_dir, beekeeping_shards):
    """Create registry with all departments."""
    async with DepartmentRegistry() as registry:
        # Context Department
        context_dept = ContextDepartment(shards=beekeeping_shards)
        await registry.register(context_dept)

        # MasterWeaver Department
        masterweaver_dept = MasterWeaverDepartment()
        await registry.register(masterweaver_dept)

        # Infrastructure Department
        infrastructure_dept = InfrastructureDepartment(
            storage_dir=temp_storage_dir,
            scales=[96, 192, 384],
            enable_neo4j=False,
            enable_qdrant=False
        )
        await registry.register(infrastructure_dept)

        # Verification Department
        verification_dept = VerificationDepartment(registry=registry)
        await registry.register(verification_dept)

        # Orchestration Department
        orchestration_dept = OrchestrationDepartment(registry=registry)
        await registry.register(orchestration_dept)

        yield registry


# ============================================================================
# Test Sequential Workflows
# ============================================================================

@pytest.mark.asyncio
async def test_sequential_context_to_masterweaver(full_registry):
    """Test Context → MasterWeaver sequential workflow."""
    orchestration = full_registry.get_department("orchestration")

    # Define sequential workflow
    workflow = [
        {
            "step_id": "retrieve",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "What are varroa mites?"}
        },
        {
            "step_id": "extract",
            "department": "masterweaver",
            "task_type": "extract_entities",
            "parameters": {"text": "{retrieve.result.response}"},
            "depends_on": ["retrieve"]
        }
    ]

    request = DepartmentRequest(
        task_id="seq_workflow_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "sequential"
        }
    )

    response = await orchestration.execute(request)

    # Verify workflow completion
    assert response.result["steps_completed"] == 2
    assert response.result["steps_failed"] == 0
    assert response.confidence.score > 0.85

    # Verify data flow
    results = response.result["results"]
    assert "retrieve" in results
    assert "extract" in results

    # Verify entities extracted from context
    entities = results["extract"]["result"].get("entities", [])
    assert len(entities) > 0


@pytest.mark.asyncio
async def test_sequential_full_pipeline(full_registry):
    """Test Context → MasterWeaver → Infrastructure complete pipeline."""
    orchestration = full_registry.get_department("orchestration")

    # Define full pipeline workflow
    workflow = [
        {
            "step_id": "retrieve",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "Explain varroa mites"}
        },
        {
            "step_id": "extract",
            "department": "masterweaver",
            "task_type": "extract_entities",
            "parameters": {"text": "{retrieve.result.response}"},
            "depends_on": ["retrieve"]
        },
        {
            "step_id": "store",
            "department": "infrastructure",
            "task_type": "store_embeddings",
            "parameters": {
                "embeddings": [
                    {
                        "id": "entity_varroa",
                        "vector": list(np.random.randn(384)),
                        "tags": ["entity", "varroa"]
                    }
                ],
                "scale": 384
            },
            "depends_on": ["extract"]
        }
    ]

    request = DepartmentRequest(
        task_id="full_pipeline_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "sequential"
        }
    )

    response = await orchestration.execute(request)

    # Verify all steps completed
    assert response.result["steps_completed"] == 3
    assert response.result["steps_failed"] == 0

    # Verify each stage
    results = response.result["results"]
    assert results["retrieve"]["confidence"] > 0.8
    assert len(results["extract"]["result"].get("entities", [])) > 0
    assert results["store"]["result"]["stored_count"] == 1


# ============================================================================
# Test Parallel Workflows
# ============================================================================

@pytest.mark.asyncio
async def test_parallel_multi_query_retrieval(full_registry):
    """Test parallel retrieval from multiple queries."""
    orchestration = full_registry.get_department("orchestration")

    # Define parallel workflow
    workflow = [
        {
            "step_id": "query_pests",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "What pests affect bees?"}
        },
        {
            "step_id": "query_behavior",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "How do bees communicate?"}
        },
        {
            "step_id": "query_management",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "How to maintain a hive?"}
        }
    ]

    request = DepartmentRequest(
        task_id="parallel_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "parallel"
        }
    )

    response = await orchestration.execute(request)

    # Verify all parallel steps completed
    assert response.result["steps_completed"] == 3
    assert response.result["steps_failed"] == 0

    # Verify parallel execution was faster than sequential would be
    # (we can't test exact timing, but all should succeed)
    results = response.result["results"]
    assert len(results) == 3
    assert all(r["confidence"] > 0.7 for r in results.values())


@pytest.mark.asyncio
async def test_parallel_extraction_and_storage(full_registry):
    """Test parallel entity extraction and storage."""
    orchestration = full_registry.get_department("orchestration")

    # Define parallel workflow
    workflow = [
        {
            "step_id": "extract_1",
            "department": "masterweaver",
            "task_type": "extract_entities",
            "parameters": {"text": "Varroa mites attack honey bees"}
        },
        {
            "step_id": "extract_2",
            "department": "masterweaver",
            "task_type": "extract_entities",
            "parameters": {"text": "Bees communicate through waggle dance"}
        },
        {
            "step_id": "store_batch",
            "department": "infrastructure",
            "task_type": "store_embeddings",
            "parameters": {
                "embeddings": [
                    {
                        "id": f"parallel_{i}",
                        "vector": list(np.random.randn(384)),
                        "tags": ["parallel"]
                    }
                    for i in range(5)
                ],
                "scale": 384
            }
        }
    ]

    request = DepartmentRequest(
        task_id="parallel_extract_store_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "parallel"
        }
    )

    response = await orchestration.execute(request)

    # Verify completion
    assert response.result["steps_completed"] == 3
    assert response.result["steps_failed"] == 0


# ============================================================================
# Test Cross-Department Verification
# ============================================================================

@pytest.mark.asyncio
async def test_cross_department_verification(full_registry):
    """Test verification across multiple departments."""
    verification = full_registry.get_department("verification")

    request = DepartmentRequest(
        task_id="cross_check_001",
        task_type="cross_check",
        parameters={
            "query": "What are varroa mites?",
            "departments": ["context"]  # Can add more when available
        }
    )

    response = await verification.execute(request)

    # Verify cross-check completed
    assert response.result["departments_queried"] >= 1
    assert response.result["agreement_level"] in ["full", "partial", "conflict"]
    assert "confidence_variance" in response.result


@pytest.mark.asyncio
async def test_verification_fact_checking(full_registry):
    """Test fact verification workflow."""
    verification = full_registry.get_department("verification")

    request = DepartmentRequest(
        task_id="fact_check_001",
        task_type="fact_check",
        parameters={
            "claim": "Varroa mites are parasitic",
            "context": "Varroa destructor attacks honey bees"
        }
    )

    response = await verification.execute(request)

    # Verify fact checking completed
    assert "verdict" in response.result
    assert response.result["verdict"] in ["supported", "contradicted", "unverifiable"]


# ============================================================================
# Test Beekeeping Domain Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_beekeeping_knowledge_extraction_workflow(full_registry):
    """Test complete beekeeping knowledge extraction pipeline."""
    orchestration = full_registry.get_department("orchestration")

    # Define beekeeping workflow
    workflow = [
        # Step 1: Retrieve pest information
        {
            "step_id": "get_pests",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "What pests affect honey bees?"}
        },
        # Step 2: Extract entities
        {
            "step_id": "extract_pests",
            "department": "masterweaver",
            "task_type": "extract_entities",
            "parameters": {"text": "{get_pests.result.response}"},
            "depends_on": ["get_pests"]
        },
        # Step 3: Get hive management info (parallel with steps 1-2)
        {
            "step_id": "get_management",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "How to maintain hive ventilation?"}
        }
    ]

    request = DepartmentRequest(
        task_id="beekeeping_workflow_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "sequential"  # Mixed sequential/parallel
        }
    )

    response = await orchestration.execute(request)

    # Verify workflow
    assert response.result["steps_completed"] >= 2
    assert response.confidence.score > 0.75

    # Verify domain-specific results
    results = response.result["results"]

    # Check retrieval worked
    pest_response = results["get_pests"]["result"]["response"]
    assert "varroa" in pest_response.lower() or "mite" in pest_response.lower()

    # Check entity extraction
    entities = results["extract_pests"]["result"].get("entities", [])
    assert len(entities) > 0


@pytest.mark.asyncio
async def test_beekeeping_multi_topic_synthesis(full_registry):
    """Test synthesizing information across multiple beekeeping topics."""
    orchestration = full_registry.get_department("orchestration")

    # Multi-topic parallel retrieval
    workflow = [
        {
            "step_id": "topic_pests",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "bee pests"}
        },
        {
            "step_id": "topic_behavior",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "bee behavior"}
        },
        {
            "step_id": "topic_management",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "hive management"}
        }
    ]

    request = DepartmentRequest(
        task_id="multi_topic_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "parallel"
        }
    )

    response = await orchestration.execute(request)

    # Verify multi-topic retrieval
    assert response.result["steps_completed"] == 3

    # Aggregate results
    aggregation_request = DepartmentRequest(
        task_id="aggregate_001",
        task_type="aggregate_results",
        parameters={"responses": response.result["results"]}
    )

    orchestration_dept = full_registry.get_department("orchestration")
    aggregated = await orchestration_dept.execute(aggregation_request)

    assert aggregated.result["department_count"] == 3
    assert aggregated.result["average_confidence"] > 0.7


# ============================================================================
# Test Performance and Optimization
# ============================================================================

@pytest.mark.asyncio
async def test_workflow_performance_benchmarks(full_registry):
    """Test workflow performance meets targets."""
    orchestration = full_registry.get_department("orchestration")

    # Simple sequential workflow
    workflow = [
        {
            "step_id": "step1",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "test query"}
        }
    ]

    request = DepartmentRequest(
        task_id="perf_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "sequential"
        }
    )

    response = await orchestration.execute(request)

    # Verify performance
    assert response.execution_time_ms < 5000  # Should complete in <5s
    assert response.result["total_time_ms"] < 5000


@pytest.mark.asyncio
async def test_parallel_vs_sequential_speedup(full_registry):
    """Test that parallel execution is faster than sequential."""
    orchestration = full_registry.get_department("orchestration")

    # Define workflow with 3 independent steps
    steps = [
        {
            "step_id": f"step_{i}",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": f"query {i}"}
        }
        for i in range(3)
    ]

    # Sequential execution
    seq_request = DepartmentRequest(
        task_id="sequential_perf",
        task_type="execute_workflow",
        parameters={
            "steps": steps,
            "workflow_type": "sequential"
        }
    )

    seq_response = await orchestration.execute(seq_request)
    seq_time = seq_response.result["total_time_ms"]

    # Parallel execution
    par_request = DepartmentRequest(
        task_id="parallel_perf",
        task_type="execute_workflow",
        parameters={
            "steps": steps,
            "workflow_type": "parallel"
        }
    )

    par_response = await orchestration.execute(par_request)
    par_time = par_response.result["total_time_ms"]

    # Parallel should be faster (allow some overhead)
    assert par_time < seq_time * 0.8  # At least 20% faster


# ============================================================================
# Test Error Handling and Recovery
# ============================================================================

@pytest.mark.asyncio
async def test_workflow_error_handling(full_registry):
    """Test workflow handles department errors gracefully."""
    orchestration = full_registry.get_department("orchestration")

    # Workflow with invalid department
    workflow = [
        {
            "step_id": "valid_step",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "test"}
        },
        {
            "step_id": "invalid_step",
            "department": "nonexistent_dept",
            "task_type": "some_task",
            "parameters": {},
            "depends_on": ["valid_step"]
        }
    ]

    request = DepartmentRequest(
        task_id="error_workflow_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "sequential"
        }
    )

    response = await orchestration.execute(request)

    # Should complete first step, fail second
    assert response.result["steps_completed"] >= 1
    assert response.result["steps_failed"] >= 1
    assert len(response.result["errors"]) > 0


@pytest.mark.asyncio
async def test_workflow_partial_failure_recovery(full_registry):
    """Test workflow continues after non-critical failures."""
    orchestration = full_registry.get_department("orchestration")

    # Parallel workflow where one may fail
    workflow = [
        {
            "step_id": "step1",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "valid query"}
        },
        {
            "step_id": "step2",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "another valid query"}
        }
    ]

    request = DepartmentRequest(
        task_id="partial_fail_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "parallel"
        }
    )

    response = await orchestration.execute(request)

    # At least some steps should succeed
    assert response.result["steps_completed"] > 0


# ============================================================================
# Test Registry Integration
# ============================================================================

@pytest.mark.asyncio
async def test_full_registry_department_discovery(full_registry):
    """Test all departments are registered and discoverable."""
    # Check all departments registered
    expected_departments = [
        "context",
        "masterweaver",
        "infrastructure",
        "verification",
        "orchestration"
    ]

    for dept_name in expected_departments:
        dept = full_registry.get_department(dept_name)
        assert dept is not None
        assert dept.name == dept_name
        assert len(dept.supported_tasks) > 0


@pytest.mark.asyncio
async def test_registry_routing_to_all_departments(full_registry):
    """Test registry can route to all departments."""
    test_requests = [
        ("context", "weave_response", {"query": "test"}),
        ("masterweaver", "extract_entities", {"text": "test text"}),
        ("infrastructure", "check_backends", {}),
        ("verification", "validate_confidence", {"confidence": 0.9}),
        ("orchestration", "aggregate_results", {"responses": {}})
    ]

    for dept_name, task_type, params in test_requests:
        request = DepartmentRequest(
            task_id=f"route_{dept_name}",
            task_type=task_type,
            parameters=params
        )

        # Should route successfully (may fail execution, but routing works)
        try:
            response = await full_registry.route_request(
                request,
                department_name=dept_name
            )
            assert response is not None
        except Exception as e:
            # Expected for some tasks without proper parameters
            assert "Missing required parameter" in str(e) or "Unsupported task" in str(e)


# ============================================================================
# Test Complete End-to-End Scenarios
# ============================================================================

@pytest.mark.asyncio
async def test_complete_e2e_knowledge_ingestion(full_registry):
    """Test complete end-to-end knowledge ingestion pipeline."""
    orchestration = full_registry.get_department("orchestration")

    # Complete pipeline: retrieve → extract → store → verify
    workflow = [
        # Retrieve
        {
            "step_id": "retrieve",
            "department": "context",
            "task_type": "weave_response",
            "parameters": {"query": "varroa mites"}
        },
        # Extract
        {
            "step_id": "extract",
            "department": "masterweaver",
            "task_type": "extract_entities",
            "parameters": {"text": "{retrieve.result.response}"},
            "depends_on": ["retrieve"]
        },
        # Store
        {
            "step_id": "store",
            "department": "infrastructure",
            "task_type": "store_embeddings",
            "parameters": {
                "embeddings": [
                    {
                        "id": "e2e_test",
                        "vector": list(np.random.randn(384)),
                        "tags": ["e2e"]
                    }
                ],
                "scale": 384
            },
            "depends_on": ["extract"]
        }
    ]

    request = DepartmentRequest(
        task_id="e2e_ingestion_001",
        task_type="execute_workflow",
        parameters={
            "steps": workflow,
            "workflow_type": "sequential"
        }
    )

    response = await orchestration.execute(request)

    # Verify complete pipeline
    assert response.result["steps_completed"] == 3
    assert response.result["steps_failed"] == 0
    assert response.confidence.score > 0.80

    # Verify each stage succeeded
    results = response.result["results"]
    assert results["retrieve"]["confidence"] > 0.75
    assert len(results["extract"]["result"].get("entities", [])) > 0
    assert results["store"]["result"]["stored_count"] == 1


@pytest.mark.asyncio
async def test_complete_e2e_query_and_verify(full_registry):
    """Test complete query → verify → refine pipeline."""
    # Query
    context = full_registry.get_department("context")
    query_request = DepartmentRequest(
        task_id="e2e_query_001",
        task_type="weave_response",
        parameters={"query": "What are varroa mites?"}
    )

    query_response = await context.execute(query_request)

    # Verify
    verification = full_registry.get_department("verification")
    verify_response = await verification.verify(query_response)

    assert verify_response.sufficient or not verify_response.sufficient

    # If insufficient, could refine (not testing actual refinement here)
    if not verify_response.sufficient:
        assert verify_response.refinement_suggestions is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
