#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Infrastructure Department.

Tests the complete integration of infrastructure services:
- Zero-copy embedding storage
- Store/retrieve/search operations
- Performance diagnostics
- Backend health checks
- DS-STAR verification workflow

Run with:
    pytest hololoom/tests/integration/test_infrastructure_department.py -v
"""

import pytest
import asyncio
import numpy as np
import tempfile
from pathlib import Path

from hololoom.departments import DepartmentRequest, DepartmentRegistry
from hololoom.apps.departments.infrastructure import (
    InfrastructureDepartment,
    ZeroCopyEmbeddingStore,
    EmbeddingView
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def infrastructure_dept(temp_storage_dir):
    """Create Infrastructure Department instance."""
    dept = InfrastructureDepartment(
        storage_dir=temp_storage_dir,
        scales=[96, 192, 384],
        max_embeddings=1000,
        enable_neo4j=False,  # Disable for tests (no Docker)
        enable_qdrant=False  # Disable for tests (no Docker)
    )

    await dept.initialize()
    yield dept
    await dept.close()


@pytest.fixture
async def zero_copy_store(temp_storage_dir):
    """Create zero-copy embedding store."""
    store = ZeroCopyEmbeddingStore(
        storage_dir=temp_storage_dir / "embeddings",
        dimension=384,
        max_embeddings=1000
    )

    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def registry_with_infrastructure(infrastructure_dept):
    """Create registry with Infrastructure Department."""
    async with DepartmentRegistry() as registry:
        await registry.register(infrastructure_dept)
        yield registry


# ============================================================================
# Sample Data
# ============================================================================

@pytest.fixture
def sample_embeddings_384():
    """Sample 384-dimensional embeddings."""
    return [
        {
            "id": "doc_1",
            "vector": np.random.randn(384).tolist(),
            "tags": ["context", "test"]
        },
        {
            "id": "doc_2",
            "vector": np.random.randn(384).tolist(),
            "tags": ["beekeeping", "test"]
        },
        {
            "id": "doc_3",
            "vector": np.random.randn(384).tolist(),
            "tags": ["context"]
        }
    ]


# ============================================================================
# Test Zero-Copy Embedding Store
# ============================================================================

@pytest.mark.asyncio
async def test_zero_copy_store_initialization(temp_storage_dir):
    """Test zero-copy store initializes correctly."""
    store = ZeroCopyEmbeddingStore(
        storage_dir=temp_storage_dir / "embeddings",
        dimension=384,
        max_embeddings=100
    )

    await store.initialize()

    assert store.embeddings_array is not None
    assert store.embeddings_array.shape == (100, 384)
    assert store.next_position == 0

    await store.close()


@pytest.mark.asyncio
async def test_zero_copy_add_embedding(zero_copy_store):
    """Test adding embeddings to zero-copy store."""
    vector = np.random.randn(384)

    success = await zero_copy_store.add_embedding(
        "test_doc",
        vector,
        tags=["test"]
    )

    assert success == True
    assert zero_copy_store.next_position == 1
    assert "test_doc" in zero_copy_store.index


@pytest.mark.asyncio
async def test_zero_copy_get_embedding(zero_copy_store):
    """Test retrieving embeddings (zero-copy)."""
    # Add embedding
    vector = np.random.randn(384)
    await zero_copy_store.add_embedding("test_doc", vector)

    # Get embedding (zero-copy)
    view = await zero_copy_store.get_embedding("test_doc")

    assert view is not None
    assert view.id == "test_doc"
    assert view.vector.shape == (384,)
    assert isinstance(view.vector, np.ndarray)

    # Verify zero-copy (vector is a view, not a copy)
    assert view.vector.base is not None  # Has underlying buffer


@pytest.mark.asyncio
async def test_zero_copy_search_similar(zero_copy_store):
    """Test similarity search."""
    # Add multiple embeddings
    vectors = [np.random.randn(384) for _ in range(5)]
    for i, vector in enumerate(vectors):
        await zero_copy_store.add_embedding(f"doc_{i}", vector)

    # Search similar to first vector
    results = await zero_copy_store.search_similar(vectors[0], k=3)

    assert len(results) == 3
    assert results[0][0] == "doc_0"  # Should find itself first
    assert results[0][1] > 0.99  # High similarity to itself


# ============================================================================
# Test Infrastructure Department Initialization
# ============================================================================

@pytest.mark.asyncio
async def test_infrastructure_department_initialization(temp_storage_dir):
    """Test Infrastructure Department initializes correctly."""
    dept = InfrastructureDepartment(storage_dir=temp_storage_dir)

    assert dept.name == "infrastructure"
    assert dept.domain == "system"
    assert dept.version == "1.0.0"
    assert "store_embeddings" in dept.supported_tasks
    assert "retrieve_embeddings" in dept.supported_tasks
    assert "search_embeddings" in dept.supported_tasks

    # Initialize
    await dept.initialize()

    assert dept._initialized == True
    assert dept._embedding_store is not None

    # Cleanup
    await dept.close()


# ============================================================================
# Test Store Embeddings
# ============================================================================

@pytest.mark.asyncio
async def test_store_embeddings(infrastructure_dept, sample_embeddings_384):
    """Test storing embeddings."""
    request = DepartmentRequest(
        task_id="store_001",
        task_type="store_embeddings",
        parameters={
            "embeddings": sample_embeddings_384,
            "scale": 384
        },
        confidence_expected=0.90
    )

    response = await infrastructure_dept.execute(request)

    # Verify response
    assert response.task_id == "store_001"
    assert response.result is not None
    assert response.result["stored_count"] == 3
    assert response.result["failed_count"] == 0
    assert response.result["success_rate"] == 1.0

    # Verify confidence
    assert response.confidence.score >= 0.90


# ============================================================================
# Test Retrieve Embeddings
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_embeddings(infrastructure_dept, sample_embeddings_384):
    """Test retrieving embeddings."""
    # First, store embeddings
    store_request = DepartmentRequest(
        task_id="store_002",
        task_type="store_embeddings",
        parameters={
            "embeddings": sample_embeddings_384,
            "scale": 384
        }
    )
    await infrastructure_dept.execute(store_request)

    # Now retrieve
    retrieve_request = DepartmentRequest(
        task_id="retrieve_001",
        task_type="retrieve_embeddings",
        parameters={
            "ids": ["doc_1", "doc_2", "doc_3"],
            "scale": 384
        }
    )

    response = await infrastructure_dept.execute(retrieve_request)

    # Verify response
    assert response.result["found_count"] == 3
    assert response.result["missing_count"] == 0
    assert response.result["found_rate"] == 1.0
    assert len(response.result["embeddings"]) == 3


@pytest.mark.asyncio
async def test_retrieve_missing_embeddings(infrastructure_dept):
    """Test retrieving non-existent embeddings."""
    request = DepartmentRequest(
        task_id="retrieve_002",
        task_type="retrieve_embeddings",
        parameters={
            "ids": ["missing_1", "missing_2"],
            "scale": 384
        }
    )

    response = await infrastructure_dept.execute(request)

    assert response.result["found_count"] == 0
    assert response.result["missing_count"] == 2
    assert response.result["found_rate"] == 0.0


# ============================================================================
# Test Search Embeddings
# ============================================================================

@pytest.mark.asyncio
async def test_search_embeddings(infrastructure_dept, sample_embeddings_384):
    """Test similarity search."""
    # Store embeddings
    store_request = DepartmentRequest(
        task_id="store_003",
        task_type="store_embeddings",
        parameters={
            "embeddings": sample_embeddings_384,
            "scale": 384
        }
    )
    await infrastructure_dept.execute(store_request)

    # Search similar
    query_vector = sample_embeddings_384[0]["vector"]

    search_request = DepartmentRequest(
        task_id="search_001",
        task_type="search_embeddings",
        parameters={
            "query_vector": query_vector,
            "k": 2,
            "scale": 384
        }
    )

    response = await infrastructure_dept.execute(search_request)

    # Verify response
    assert response.result["result_count"] >= 1
    assert len(response.result["results"]) >= 1

    # Top result should be doc_1 (query itself)
    top_result = response.result["results"][0]
    assert top_result["id"] == "doc_1"
    assert top_result["similarity"] > 0.99


# ============================================================================
# Test Performance Diagnostics
# ============================================================================

@pytest.mark.asyncio
async def test_diagnose_performance(infrastructure_dept):
    """Test performance diagnostics."""
    request = DepartmentRequest(
        task_id="diagnose_001",
        task_type="diagnose_performance",
        parameters={}
    )

    response = await infrastructure_dept.execute(request)

    # Verify response structure
    assert "store_stats" in response.result
    assert "latency_stats" in response.result
    assert "backend_health" in response.result
    assert "health_score" in response.result

    # Check backend health
    assert response.result["backend_health"]["embedding_store"] == True


# ============================================================================
# Test Backend Health Checks
# ============================================================================

@pytest.mark.asyncio
async def test_check_backends(infrastructure_dept):
    """Test backend health checks."""
    request = DepartmentRequest(
        task_id="check_001",
        task_type="check_backends",
        parameters={}
    )

    response = await infrastructure_dept.execute(request)

    # Verify response
    assert "backend_health" in response.result
    assert "healthy_count" in response.result
    assert "total_count" in response.result

    # Embedding store should be healthy
    assert response.result["backend_health"]["embedding_store"] == True


# ============================================================================
# Test Verification (DS-STAR)
# ============================================================================

@pytest.mark.asyncio
async def test_verify_sufficient_operation(infrastructure_dept, sample_embeddings_384):
    """Test verifying a successful operation."""
    # Execute store operation
    request = DepartmentRequest(
        task_id="verify_001",
        task_type="store_embeddings",
        parameters={
            "embeddings": sample_embeddings_384,
            "scale": 384
        }
    )

    response = await infrastructure_dept.execute(request)

    # Verify
    verification = await infrastructure_dept.verify(response)

    # Should be sufficient
    assert verification.confidence_valid == True
    assert verification.reasoning_sound == True
    assert verification.sufficient == True


# ============================================================================
# Test Full DS-STAR Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_full_ds_star_workflow(infrastructure_dept, sample_embeddings_384):
    """Test complete DS-STAR cycle."""
    # Execute
    request = DepartmentRequest(
        task_id="workflow_001",
        task_type="store_embeddings",
        parameters={
            "embeddings": sample_embeddings_384,
            "scale": 384
        }
    )

    response = await infrastructure_dept.execute(request)

    # Verify
    verification = await infrastructure_dept.verify(response)

    # If insufficient, refine
    if not verification.sufficient:
        refined_response = await infrastructure_dept.refine(request, response, verification)
        verification = await infrastructure_dept.verify(refined_response)

    # Should eventually be sufficient
    assert verification.sufficient == True


# ============================================================================
# Test Registry Integration
# ============================================================================

@pytest.mark.asyncio
async def test_infrastructure_registry_registration(registry_with_infrastructure):
    """Test Infrastructure Department registration."""
    registry = registry_with_infrastructure

    # Check registration
    dept = registry.get_department("infrastructure")
    assert dept is not None
    assert dept.name == "infrastructure"

    # Check discovery by domain
    depts_by_domain = registry.find_by_domain("system")
    assert len(depts_by_domain) > 0
    assert any(d.name == "infrastructure" for d in depts_by_domain)


@pytest.mark.asyncio
async def test_infrastructure_registry_routing(registry_with_infrastructure, sample_embeddings_384):
    """Test routing through registry."""
    registry = registry_with_infrastructure

    # Route request
    request = DepartmentRequest(
        task_id="route_001",
        task_type="store_embeddings",
        parameters={
            "embeddings": sample_embeddings_384[:1],  # Just one embedding
            "scale": 384
        }
    )

    response = await registry.route_request(request)

    # Should get valid response
    assert response is not None
    assert response.task_id == "route_001"
    assert response.result["stored_count"] >= 0


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_infrastructure_invalid_task_type(infrastructure_dept):
    """Test error handling for invalid task type."""
    request = DepartmentRequest(
        task_id="invalid_task",
        task_type="unsupported_task",
        parameters={}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await infrastructure_dept.execute(request)


@pytest.mark.asyncio
async def test_infrastructure_missing_parameters(infrastructure_dept):
    """Test error handling for missing parameters."""
    request = DepartmentRequest(
        task_id="missing_params",
        task_type="store_embeddings",
        parameters={"scale": 384}  # Missing 'embeddings'
    )

    with pytest.raises(ValueError, match="Missing required parameter"):
        await infrastructure_dept.execute(request)


# ============================================================================
# Test Health Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_infrastructure_health_check(infrastructure_dept):
    """Test department health monitoring."""
    # Execute some operations
    for i in range(3):
        request = DepartmentRequest(
            task_id=f"health_req_{i}",
            task_type="check_backends",
            parameters={}
        )
        await infrastructure_dept.execute(request)

    # Check health
    health = await infrastructure_dept.health_check()

    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert health["name"] == "infrastructure"
    assert health["performance"]["total_requests"] >= 3


# ============================================================================
# Test Lifecycle
# ============================================================================

@pytest.mark.asyncio
async def test_infrastructure_lifecycle_context_manager(temp_storage_dir):
    """Test Infrastructure Department lifecycle."""
    async with InfrastructureDepartment(storage_dir=temp_storage_dir) as dept:
        assert dept._initialized == True

        # Execute request
        request = DepartmentRequest(
            task_id="lifecycle_test",
            task_type="check_backends",
            parameters={}
        )

        response = await dept.execute(request)
        assert response is not None

    # After exit, should be closed
    assert dept._closed == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
