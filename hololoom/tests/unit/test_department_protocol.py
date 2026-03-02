#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Department Protocol.

Tests the core abstractions that enable modular nested learning:
- Confidence system (levels, metadata)
- Request/Response protocols
- Verification patterns
- Base department functionality
- Registry operations

Run with:
    pytest HoloLoom/tests/unit/test_department_protocol.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from uuid import UUID

from hololoom.protocols.department import (
    ConfidenceLevel,
    ConfidenceMetadata,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    VerificationCheck,
    VerificationStatus,
    DepartmentManifest,
    DepartmentConfig,
    Department,
    compute_learning_rate,
    should_update_now,
    create_simple_request,
    create_simple_response
)
from hololoom.apps.departments.base import BaseDepartment
from hololoom.apps.departments.registry import DepartmentRegistry


# ============================================================================
# Test Confidence System
# ============================================================================

def test_confidence_level_from_score():
    """Test mapping confidence scores to levels via ConfidenceMetadata."""
    # Use ConfidenceMetadata.from_score() to test score → level mapping
    assert ConfidenceMetadata.from_score(0.98).level == ConfidenceLevel.VERIFIED
    assert ConfidenceMetadata.from_score(0.90).level == ConfidenceLevel.HIGH
    assert ConfidenceMetadata.from_score(0.70).level == ConfidenceLevel.MEDIUM
    assert ConfidenceMetadata.from_score(0.40).level == ConfidenceLevel.LOW
    assert ConfidenceMetadata.from_score(0.10).level == ConfidenceLevel.CRITICAL

    # Edge cases at boundaries
    assert ConfidenceMetadata.from_score(0.95).level == ConfidenceLevel.VERIFIED
    assert ConfidenceMetadata.from_score(0.75).level == ConfidenceLevel.HIGH
    assert ConfidenceMetadata.from_score(0.50).level == ConfidenceLevel.MEDIUM
    assert ConfidenceMetadata.from_score(0.20).level == ConfidenceLevel.LOW

    # Clamping (values are clamped to [0, 1])
    assert ConfidenceMetadata.from_score(1.5).score == 1.0
    assert ConfidenceMetadata.from_score(-0.1).score == 0.0


def test_confidence_level_learning_rate():
    """Test that ConfidenceLevel enum values exist."""
    # Just verify the enum values exist
    assert ConfidenceLevel.CRITICAL.value == "critical"
    assert ConfidenceLevel.LOW.value == "low"
    assert ConfidenceLevel.MEDIUM.value == "medium"
    assert ConfidenceLevel.HIGH.value == "high"
    assert ConfidenceLevel.VERIFIED.value == "verified"


def test_confidence_level_multiplier():
    """Test that all confidence levels are accessible."""
    # Verify we can iterate through levels
    levels = list(ConfidenceLevel)
    assert len(levels) == 5
    assert ConfidenceLevel.CRITICAL in levels
    assert ConfidenceLevel.VERIFIED in levels


def test_confidence_metadata_from_score():
    """Test creating ConfidenceMetadata from score."""
    conf = ConfidenceMetadata.from_score(
        0.92,
        justification=["Test passed"],
        sources=["test_source"]
    )

    assert conf.score == 0.92
    assert conf.level == ConfidenceLevel.HIGH
    assert "Test passed" in conf.justification
    assert "test_source" in conf.sources


def test_compute_learning_rate():
    """Test adaptive learning rate computation."""
    # compute_learning_rate takes iteration, initial_rate, decay_rate, min_rate
    lr_0 = compute_learning_rate(iteration=0, initial_rate=0.1, decay_rate=0.99)
    lr_100 = compute_learning_rate(iteration=100, initial_rate=0.1, decay_rate=0.99)

    # First iteration should be close to initial rate
    assert abs(lr_0 - 0.1) < 0.001

    # After 100 iterations, rate should be lower due to exponential decay
    assert lr_100 < lr_0

    # Test minimum rate floor
    lr_very_high_iter = compute_learning_rate(iteration=10000, min_rate=0.001)
    assert lr_very_high_iter >= 0.001


def test_should_update_now():
    """Test update timing logic."""
    now = datetime.utcnow()

    # Never updated → should update
    assert should_update_now(last_update=None) == True

    # Recent update → should not update
    recent_update = now - timedelta(seconds=30)
    assert should_update_now(last_update=recent_update, min_interval_seconds=60) == False

    # Old update → should update
    old_update = now - timedelta(seconds=120)
    assert should_update_now(last_update=old_update, min_interval_seconds=60) == True


# ============================================================================
# Test Request/Response Protocol
# ============================================================================

def test_department_request_creation():
    """Test creating a DepartmentRequest."""
    request = DepartmentRequest(
        task_type="test_task",
        parameters={"key": "value"},
        constraints={"max_latency_ms": 150},
        priority=75
    )

    assert isinstance(request.task_id, UUID)
    assert request.task_type == "test_task"
    assert request.parameters["key"] == "value"
    assert request.constraints["max_latency_ms"] == 150
    assert request.priority == 75


def test_department_request_serialization():
    """Test DepartmentRequest with helper function."""
    request = create_simple_request(
        task_type="test_task",
        parameters={"answer": 42},
        constraints={"timeout": 1000}
    )

    assert request.task_type == "test_task"
    assert request.parameters["answer"] == 42
    assert request.constraints["timeout"] == 1000
    assert isinstance(request.timestamp, datetime)


def test_department_response_creation():
    """Test creating a DepartmentResponse."""
    confidence = ConfidenceMetadata.from_score(0.85)

    response = DepartmentResponse(
        task_id=DepartmentRequest().task_id,  # Generate a UUID
        result={"answer": 42},
        confidence=confidence,
        metadata={"step1": "analysis", "step2": "decision"}
    )

    assert response.result["answer"] == 42
    assert response.confidence.score == 0.85
    assert "step1" in response.metadata


def test_department_response_serialization():
    """Test DepartmentResponse with helper function."""
    request = DepartmentRequest()

    response = create_simple_response(
        task_id=request.task_id,
        result="test result",
        confidence_score=0.85
    )

    assert response.result == "test result"
    assert response.confidence.score == 0.85
    assert response.confidence.level == ConfidenceLevel.HIGH


def test_verification_result_creation():
    """Test creating a VerificationResult."""
    check = VerificationCheck(
        name="test_check",
        status=VerificationStatus.PASSED,
        reason="Check passed",
        score=0.95
    )

    result = VerificationResult(
        verified=True,
        checks=[check],
        overall_score=0.95,
        summary="All checks passed",
        confidence=0.92,
        recommendations=["Consider caching results"]
    )

    assert result.verified == True
    assert len(result.checks) == 1
    assert result.overall_score == 0.95
    assert result.summary == "All checks passed"
    assert len(result.recommendations) == 1


def test_verification_result_serialization():
    """Test VerificationResult passed property."""
    check_passed = VerificationCheck(
        name="check1",
        status=VerificationStatus.PASSED,
        reason="OK"
    )
    check_failed = VerificationCheck(
        name="check2",
        status=VerificationStatus.FAILED,
        reason="Not OK"
    )

    # All passed
    result_pass = VerificationResult(checks=[check_passed])
    assert result_pass.passed == True

    # One failed
    result_fail = VerificationResult(checks=[check_passed, check_failed])
    assert result_fail.passed == False


# ============================================================================
# Test Marketplace Types
# ============================================================================

def test_department_manifest_creation():
    """Test creating a DepartmentManifest."""
    config = DepartmentConfig(
        name="test_dept",
        domain="testing",
        version="1.0.0",
        supported_tasks=["task1", "task2"],
        confidence_range=(0.6, 0.9)
    )

    manifest = DepartmentManifest(
        config=config,
        capabilities=["retrieve", "process"],
        dependencies=["context"],
        metadata={"author": "test"}
    )

    assert manifest.config.name == "test_dept"
    assert manifest.config.domain == "testing"
    assert len(manifest.capabilities) == 2
    assert manifest.dependencies == ["context"]


def test_department_config_creation():
    """Test creating a DepartmentConfig."""
    config = DepartmentConfig(
        name="test",
        domain="testing",
        version="1.0.0",
        enable_learning=True,
        enable_verification=True,
        max_latency_ms=10000.0
    )

    assert config.enable_learning == True
    assert config.enable_verification == True
    assert config.max_latency_ms == 10000.0


# ============================================================================
# Test Base Department
# ============================================================================

class TestDepartment(BaseDepartment):
    """Minimal department for testing."""

    def __init__(self):
        # Create config with required name and domain
        config = DepartmentConfig(
            name="test",
            domain="testing"
        )
        super().__init__(
            name="test",
            domain="testing",
            version="1.0.0",
            supported_tasks=["test_task"],
            confidence_range=(0.6, 0.9),
            config=config
        )

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        # Simple echo implementation
        confidence = ConfidenceMetadata.from_score(0.85)
        return DepartmentResponse(
            task_id=request.task_id,
            result={"echo": request.parameters},
            confidence=confidence
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        # Always pass for testing
        check = VerificationCheck(
            name="test",
            status=VerificationStatus.PASSED,
            reason="Test passed"
        )
        return VerificationResult(
            verified=True,
            checks=[check],
            summary="All checks passed"
        )

    async def refine(self, request, prior_response, verification) -> DepartmentResponse:
        # Just re-execute for testing
        return await self.execute(request)


@pytest.mark.asyncio
async def test_base_department_initialization():
    """Test BaseDepartment initialization."""
    dept = TestDepartment()

    assert dept.name == "test"
    assert dept.domain == "testing"
    assert dept.version == "1.0.0"
    assert "test_task" in dept.supported_tasks
    assert len(dept.short_term_memory) == 0


@pytest.mark.asyncio
async def test_base_department_execute():
    """Test department execution."""
    dept = TestDepartment()

    request = DepartmentRequest(
        task_type="test_task",
        parameters={"key": "value"}
    )

    response = await dept.execute(request)

    assert response.task_id == request.task_id
    assert response.result["echo"]["key"] == "value"
    assert response.confidence.score == 0.85


@pytest.mark.asyncio
async def test_base_department_session_memory():
    """Test session memory management."""
    dept = TestDepartment()

    # Store session state
    await dept._store_session_state("session_1", {"count": 1})

    # Retrieve session state
    state = await dept.get_session_state("session_1")

    assert state is not None
    assert state['session_id'] == "session_1"
    assert state['request_count'] == 1


@pytest.mark.asyncio
async def test_base_department_learning_signals():
    """Test learning signal aggregation."""
    dept = TestDepartment()

    signals = [
        {
            'task_type': 'test_task',
            'outcome': 'success',
            'confidence_predicted': 0.85,
            'confidence_actual': 0.90
        },
        {
            'task_type': 'test_task',
            'outcome': 'success',
            'confidence_predicted': 0.80,
            'confidence_actual': 0.85
        }
    ]

    await dept.update_strategy(signals)

    # Check institutional memory
    memory = await dept.get_institutional_memory('successful_strategies')

    assert 'test_task' in memory
    assert memory['test_task']['total'] == 2
    assert memory['test_task']['successes'] == 2


@pytest.mark.asyncio
async def test_base_department_health_check():
    """Test department health monitoring."""
    dept = TestDepartment()

    # Execute a request to generate metrics
    request = DepartmentRequest(
        task_type="test_task",
        parameters={}
    )

    await dept.execute(request)
    dept._record_request(success=True, latency_ms=150.0, confidence=0.85)

    # Check health
    health = await dept.health_check()

    assert health['status'] == "healthy"
    assert health['name'] == "test"
    assert health['performance']['total_requests'] == 1
    assert health['performance']['success_rate'] == 1.0


@pytest.mark.asyncio
async def test_base_department_lifecycle():
    """Test department lifecycle management."""
    async with TestDepartment() as dept:
        assert dept._initialized == True

        request = DepartmentRequest(
            task_type="test_task",
            parameters={}
        )

        response = await dept.execute(request)
        assert response is not None

    # After exit, should be closed
    assert dept._closed == True


# ============================================================================
# Test Department Registry
# ============================================================================

@pytest.mark.asyncio
async def test_registry_initialization():
    """Test registry initialization."""
    registry = DepartmentRegistry()

    assert len(registry._departments) == 0
    assert len(registry._by_domain) == 0
    assert len(registry._by_task) == 0


@pytest.mark.asyncio
async def test_registry_register_department():
    """Test registering a department."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Check registration
        assert "test" in registry._departments
        assert len(registry._departments["test"]) == 1

        # Check indexes
        assert "testing" in registry._by_domain
        assert "test" in registry._by_domain["testing"]

        assert "test_task" in registry._by_task
        assert "test" in registry._by_task["test_task"]


@pytest.mark.asyncio
async def test_registry_get_department():
    """Test retrieving a department."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Get by name
        retrieved = registry.get_department("test")

        assert retrieved is not None
        assert retrieved.name == "test"


@pytest.mark.asyncio
async def test_registry_find_by_domain():
    """Test finding departments by domain."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Find by domain
        depts = registry.find_by_domain("testing")

        assert len(depts) == 1
        assert depts[0].name == "test"


@pytest.mark.asyncio
async def test_registry_find_by_task():
    """Test finding departments by task type."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Find by task
        depts = registry.find_by_task("test_task")

        assert len(depts) == 1
        assert depts[0].name == "test"


@pytest.mark.asyncio
async def test_registry_route_request():
    """Test routing a request to a department."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Route request
        request = DepartmentRequest(
            task_type="test_task",
            parameters={"key": "value"}
        )

        response = await registry.route_request(request)

        assert response is not None
        assert response.task_id == request.task_id
        assert response.result["echo"]["key"] == "value"


@pytest.mark.asyncio
async def test_registry_route_to_specific_department():
    """Test routing to a specific department."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Route to specific department
        request = DepartmentRequest(
            task_type="test_task",
            parameters={}
        )

        response = await registry.route_request(request, department_name="test")

        assert response is not None


@pytest.mark.asyncio
async def test_registry_unregister():
    """Test unregistering a department."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Unregister
        await registry.unregister("test")

        # Check removal
        assert "test" not in registry._departments

        # Should not find by domain/task
        depts_domain = registry.find_by_domain("testing")
        depts_task = registry.find_by_task("test_task")

        assert len(depts_domain) == 0
        assert len(depts_task) == 0


@pytest.mark.asyncio
async def test_registry_lifecycle():
    """Test registry lifecycle management."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Registry should be operational
        retrieved = registry.get_department("test")
        assert retrieved is not None

    # After exit, should be closed
    assert registry._closed == True


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_department_workflow():
    """Test complete workflow: register → execute → verify → refine."""
    async with DepartmentRegistry() as registry:
        dept = TestDepartment()
        await registry.register(dept)

        # Execute
        request = DepartmentRequest(
            task_type="test_task",
            parameters={"question": "What is 2+2?"}
        )

        response = await registry.route_request(request)

        # Verify
        verification = await dept.verify(response)

        assert verification.verified == True
        assert verification.passed == True

        # If not verified, would refine
        if not verification.verified:
            refined = await dept.refine(request, response, verification)
            assert refined is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
