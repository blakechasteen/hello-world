#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Department Protocol.

Tests the core abstractions that enable modular nested learning:
- Confidence system (levels, metadata, learning rates)
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

from HoloLoom.departments import (
    ConfidenceLevel,
    ConfidenceMetadata,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    DepartmentManifest,
    DepartmentConfig,
    Department,
    BaseDepartment,
    DepartmentRegistry,
    compute_learning_rate,
    should_update_now
)


# ============================================================================
# Test Confidence System
# ============================================================================

def test_confidence_level_from_score():
    """Test mapping confidence scores to levels."""
    assert ConfidenceLevel.from_score(0.98) == ConfidenceLevel.CRITICAL
    assert ConfidenceLevel.from_score(0.90) == ConfidenceLevel.HIGH
    assert ConfidenceLevel.from_score(0.75) == ConfidenceLevel.MEDIUM
    assert ConfidenceLevel.from_score(0.50) == ConfidenceLevel.LOW
    assert ConfidenceLevel.from_score(0.20) == ConfidenceLevel.UNCERTAIN

    # Edge cases
    assert ConfidenceLevel.from_score(0.95) == ConfidenceLevel.CRITICAL
    assert ConfidenceLevel.from_score(0.85) == ConfidenceLevel.HIGH
    assert ConfidenceLevel.from_score(0.65) == ConfidenceLevel.MEDIUM
    assert ConfidenceLevel.from_score(0.40) == ConfidenceLevel.LOW

    # Clamping
    assert ConfidenceLevel.from_score(1.5) == ConfidenceLevel.CRITICAL
    assert ConfidenceLevel.from_score(-0.1) == ConfidenceLevel.UNCERTAIN


def test_confidence_level_learning_rate():
    """Test learning rate strings for each confidence level."""
    assert ConfidenceLevel.CRITICAL.learning_rate_str() == "weekly"
    assert ConfidenceLevel.HIGH.learning_rate_str() == "daily"
    assert ConfidenceLevel.MEDIUM.learning_rate_str() == "hourly"
    assert ConfidenceLevel.LOW.learning_rate_str() == "per-task"
    assert ConfidenceLevel.UNCERTAIN.learning_rate_str() == "immediate"


def test_confidence_level_multiplier():
    """Test learning rate multipliers."""
    assert ConfidenceLevel.CRITICAL.learning_rate_multiplier() == 0.1
    assert ConfidenceLevel.HIGH.learning_rate_multiplier() == 0.5
    assert ConfidenceLevel.MEDIUM.learning_rate_multiplier() == 1.0
    assert ConfidenceLevel.LOW.learning_rate_multiplier() == 2.0
    assert ConfidenceLevel.UNCERTAIN.learning_rate_multiplier() == 5.0


def test_confidence_metadata_from_score():
    """Test creating ConfidenceMetadata from score."""
    conf = ConfidenceMetadata.from_score(
        0.92,
        justification=["Test passed"],
        uncertainty_sources=["Limited data"]
    )

    assert conf.score == 0.92
    assert conf.level == ConfidenceLevel.HIGH
    assert conf.learning_rate == "daily"
    assert "Test passed" in conf.justification
    assert "Limited data" in conf.uncertainty_sources


def test_compute_learning_rate():
    """Test adaptive learning rate computation."""
    conf_critical = ConfidenceMetadata.from_score(0.98)
    conf_uncertain = ConfidenceMetadata.from_score(0.20)

    base_lr = 1e-4

    lr_critical = compute_learning_rate(conf_critical, base_lr)
    lr_uncertain = compute_learning_rate(conf_uncertain, base_lr)

    assert lr_critical == 1e-5  # 0.1× base
    assert lr_uncertain == 5e-4  # 5.0× base

    # Uncertain should learn 50× faster than critical
    assert lr_uncertain / lr_critical == 50.0


def test_should_update_now():
    """Test update timing logic."""
    now = datetime.now()

    # CRITICAL: weekly updates
    conf_critical = ConfidenceMetadata.from_score(0.98)
    last_update_7d = now - timedelta(days=7, hours=1)
    last_update_6d = now - timedelta(days=6)

    assert should_update_now(conf_critical, last_update_7d, now) == True
    assert should_update_now(conf_critical, last_update_6d, now) == False

    # HIGH: daily updates
    conf_high = ConfidenceMetadata.from_score(0.90)
    last_update_25h = now - timedelta(hours=25)
    last_update_23h = now - timedelta(hours=23)

    assert should_update_now(conf_high, last_update_25h, now) == True
    assert should_update_now(conf_high, last_update_23h, now) == False

    # LOW/UNCERTAIN: always update
    conf_low = ConfidenceMetadata.from_score(0.50)
    last_update_1m = now - timedelta(minutes=1)

    assert should_update_now(conf_low, last_update_1m, now) == True


# ============================================================================
# Test Request/Response Protocol
# ============================================================================

def test_department_request_creation():
    """Test creating a DepartmentRequest."""
    request = DepartmentRequest(
        task_id="req_001",
        task_type="test_task",
        parameters={"key": "value"},
        confidence_expected=0.75,
        context_preference="adaptive",
        privacy_level="standard",
        session_id="session_123",
        timeout_ms=5000.0
    )

    assert request.task_id == "req_001"
    assert request.task_type == "test_task"
    assert request.parameters["key"] == "value"
    assert request.confidence_expected == 0.75
    assert request.context_preference == "adaptive"
    assert request.session_id == "session_123"


def test_department_request_serialization():
    """Test DepartmentRequest serialization."""
    request = DepartmentRequest(
        task_id="req_001",
        task_type="test_task",
        parameters={"answer": 42}
    )

    data = request.to_dict()

    assert data['task_id'] == "req_001"
    assert data['task_type'] == "test_task"
    assert data['parameters']['answer'] == 42
    assert 'timestamp' in data


def test_department_response_creation():
    """Test creating a DepartmentResponse."""
    confidence = ConfidenceMetadata.from_score(0.85)

    response = DepartmentResponse(
        task_id="req_001",
        result={"answer": 42},
        confidence=confidence,
        reasoning={"step1": "analysis", "step2": "decision"},
        learning_signals={"outcome": "success"}
    )

    assert response.task_id == "req_001"
    assert response.result["answer"] == 42
    assert response.confidence.score == 0.85
    assert "step1" in response.reasoning


def test_department_response_serialization():
    """Test DepartmentResponse serialization."""
    confidence = ConfidenceMetadata.from_score(0.85)

    response = DepartmentResponse(
        task_id="req_001",
        result="test result",
        confidence=confidence
    )

    data = response.to_dict()

    assert data['task_id'] == "req_001"
    assert data['result'] == "test result"
    assert data['confidence']['score'] == 0.85
    assert data['confidence']['level'] == "HIGH"


def test_verification_result_creation():
    """Test creating a VerificationResult."""
    result = VerificationResult(
        sufficient=True,
        confidence_valid=True,
        reasoning_sound=True,
        alternative_paths=["path1", "path2"],
        refinement_suggestions={"expand_context": True}
    )

    assert result.sufficient == True
    assert result.confidence_valid == True
    assert len(result.alternative_paths) == 2
    assert result.refinement_suggestions["expand_context"] == True


def test_verification_result_serialization():
    """Test VerificationResult serialization."""
    result = VerificationResult(
        sufficient=False,
        confidence_valid=False,
        reasoning_sound=True,
        escalation_needed=True,
        escalation_reason="Low confidence"
    )

    data = result.to_dict()

    assert data['sufficient'] == False
    assert data['escalation_needed'] == True
    assert data['escalation_reason'] == "Low confidence"


# ============================================================================
# Test Marketplace Types
# ============================================================================

def test_department_manifest_creation():
    """Test creating a DepartmentManifest."""
    manifest = DepartmentManifest(
        name="test_dept",
        domain="testing",
        version="1.0.0",
        supported_tasks=["task1", "task2"],
        confidence_range=(0.6, 0.9),
        requires=["context"],
        avg_latency_ms=150.0
    )

    assert manifest.name == "test_dept"
    assert manifest.domain == "testing"
    assert len(manifest.supported_tasks) == 2
    assert manifest.requires == ["context"]


def test_department_config_creation():
    """Test creating a DepartmentConfig."""
    config = DepartmentConfig(
        enable_learning=True,
        enable_verification=True,
        max_concurrent_tasks=20,
        default_timeout_ms=10000.0
    )

    assert config.enable_learning == True
    assert config.enable_verification == True
    assert config.max_concurrent_tasks == 20


# ============================================================================
# Test Base Department
# ============================================================================

class TestDepartment(BaseDepartment):
    """Minimal department for testing."""

    def __init__(self):
        super().__init__(
            name="test",
            domain="testing",
            version="1.0.0",
            supported_tasks=["test_task"],
            confidence_range=(0.6, 0.9)
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
        return VerificationResult(
            sufficient=True,
            confidence_valid=True,
            reasoning_sound=True
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
        task_id="req_001",
        task_type="test_task",
        parameters={"key": "value"}
    )

    response = await dept.execute(request)

    assert response.task_id == "req_001"
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

    # Execute some requests
    request = DepartmentRequest(
        task_id="req_001",
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
            task_id="req_001",
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
    registry = DepartmentRegistry()
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
    registry = DepartmentRegistry()
    dept = TestDepartment()

    await registry.register(dept)

    # Get by name
    retrieved = registry.get_department("test")

    assert retrieved is not None
    assert retrieved.name == "test"


@pytest.mark.asyncio
async def test_registry_find_by_domain():
    """Test finding departments by domain."""
    registry = DepartmentRegistry()
    dept = TestDepartment()

    await registry.register(dept)

    # Find by domain
    depts = registry.find_by_domain("testing")

    assert len(depts) == 1
    assert depts[0].name == "test"


@pytest.mark.asyncio
async def test_registry_find_by_task():
    """Test finding departments by task type."""
    registry = DepartmentRegistry()
    dept = TestDepartment()

    await registry.register(dept)

    # Find by task
    depts = registry.find_by_task("test_task")

    assert len(depts) == 1
    assert depts[0].name == "test"


@pytest.mark.asyncio
async def test_registry_route_request():
    """Test routing a request to a department."""
    registry = DepartmentRegistry()
    dept = TestDepartment()

    await registry.register(dept)

    # Route request
    request = DepartmentRequest(
        task_id="req_001",
        task_type="test_task",
        parameters={"key": "value"}
    )

    response = await registry.route_request(request)

    assert response is not None
    assert response.task_id == "req_001"
    assert response.result["echo"]["key"] == "value"


@pytest.mark.asyncio
async def test_registry_route_to_specific_department():
    """Test routing to a specific department."""
    registry = DepartmentRegistry()
    dept = TestDepartment()

    await registry.register(dept)

    # Route to specific department
    request = DepartmentRequest(
        task_id="req_001",
        task_type="test_task",
        parameters={}
    )

    response = await registry.route_request(request, department_name="test")

    assert response is not None


@pytest.mark.asyncio
async def test_registry_unregister():
    """Test unregistering a department."""
    registry = DepartmentRegistry()
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
            task_id="req_001",
            task_type="test_task",
            parameters={"question": "What is 2+2?"}
        )

        response = await registry.route_request(request)

        # Verify
        verification = await dept.verify(response)

        assert verification.sufficient == True
        assert verification.confidence_valid == True

        # If not sufficient, would refine
        if not verification.sufficient:
            refined = await dept.refine(request, response, verification)
            assert refined is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
