"""
Integration Framework Tests

Tests for the core integration framework functionality including:
- Basic pipeline execution
- Parallel stage execution
- Graceful degradation
- Confidence aggregation
- Timeout handling
- Error recovery

Author: HoloLoom Team
Date: 2025-01-21
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

from HoloLoom.integration import (
    IntegrationFramework,
    PipelineDefinition,
    PipelineResult,
    StageDefinition,
    StageResult,
    StageType,
    DependencyGraph,
    create_integration_framework,
    get_pipeline,
    list_pipelines,
)
from HoloLoom.departments.registry import DepartmentRegistry
from HoloLoom.departments.base import BaseDepartment
from HoloLoom.departments.protocol import (
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    ConfidenceMetadata,
)
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query


# Mock Departments for Testing

class MockRetrievalDepartment(BaseDepartment):
    """Mock retrieval department for testing."""

    def __init__(self, latency_ms: int = 50, confidence: float = 0.9):
        super().__init__(
            name="mock_retrieval",
            domain="retrieval",
            version="1.0.0",
            supported_tasks=["retrieve_context"]
        )
        self.latency_ms = latency_ms
        self.confidence = confidence

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute mock retrieval."""
        await asyncio.sleep(self.latency_ms / 1000.0)
        return DepartmentResponse(
            task_id=request.task_id,
            result={"memories": [f"memory_{i}" for i in range(10)]},
            confidence=ConfidenceMetadata.from_score(self.confidence)
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, confidence=self.confidence)


class MockReasoningDepartment(BaseDepartment):
    """Mock reasoning department for testing."""

    def __init__(self, latency_ms: int = 100, confidence: float = 0.85):
        super().__init__(
            name="mock_reasoning",
            domain="reasoning",
            version="1.0.0",
            supported_tasks=["reason"]
        )
        self.latency_ms = latency_ms
        self.confidence = confidence

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute mock reasoning."""
        await asyncio.sleep(self.latency_ms / 1000.0)
        return DepartmentResponse(
            task_id=request.task_id,
            result={"response": "Mock reasoning response"},
            confidence=ConfidenceMetadata.from_score(self.confidence)
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, confidence=self.confidence)


class MockQADepartment(BaseDepartment):
    """Mock QA department for testing."""

    def __init__(self, latency_ms: int = 80, should_fail: bool = False):
        super().__init__(
            name="mock_qa",
            domain="qa",
            version="1.0.0",
            supported_tasks=["analyze"]
        )
        self.latency_ms = latency_ms
        self.should_fail = should_fail

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute mock QA."""
        await asyncio.sleep(self.latency_ms / 1000.0)

        if self.should_fail:
            raise RuntimeError("Mock QA failure")

        return DepartmentResponse(
            task_id=request.task_id,
            result={"issues": [], "total_issues": 0},
            confidence=ConfidenceMetadata.from_score(0.95)
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, confidence=0.95)


class MockVerificationDepartment(BaseDepartment):
    """Mock verification department for testing."""

    def __init__(self, latency_ms: int = 60):
        super().__init__(
            name="mock_verification",
            domain="verification",
            version="1.0.0",
            supported_tasks=["verify"]
        )
        self.latency_ms = latency_ms

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute mock verification."""
        await asyncio.sleep(self.latency_ms / 1000.0)
        return DepartmentResponse(
            task_id=request.task_id,
            result={"verified": True},
            confidence=ConfidenceMetadata.from_score(0.9)
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, confidence=0.9)


# Test Fixtures

@pytest.fixture
async def mock_registry():
    """Create a mock department registry."""
    registry = DepartmentRegistry()

    # Register mock departments
    await registry.register(
        MockRetrievalDepartment(),
        name="mock_retrieval",
        domain="retrieval"
    )
    await registry.register(
        MockReasoningDepartment(),
        name="mock_reasoning",
        domain="reasoning"
    )
    await registry.register(
        MockQADepartment(),
        name="mock_qa",
        domain="qa"
    )
    await registry.register(
        MockVerificationDepartment(),
        name="mock_verification",
        domain="verification"
    )

    return registry


@pytest.fixture
def simple_pipeline():
    """Create a simple pipeline for testing."""
    return PipelineDefinition(
        name="test_simple",
        description="Simple test pipeline",
        stages=[
            StageDefinition(
                stage_type=StageType.RETRIEVAL,
                department="mock_retrieval",
                timeout_ms=5000,
                required=True
            ),
            StageDefinition(
                stage_type=StageType.REASONING,
                department="mock_reasoning",
                timeout_ms=10000,
                required=True
            ),
        ],
        expected_latency_ms=200
    )


@pytest.fixture
def parallel_pipeline():
    """Create a pipeline with parallel stages."""
    return PipelineDefinition(
        name="test_parallel",
        description="Pipeline with parallel execution",
        stages=[
            StageDefinition(
                stage_type=StageType.RETRIEVAL,
                department="mock_retrieval",
                timeout_ms=5000,
                required=True
            ),
            StageDefinition(
                stage_type=StageType.REASONING,
                department="mock_reasoning",
                timeout_ms=10000,
                required=True
            ),
            # Parallel stages
            StageDefinition(
                stage_type=StageType.QUALITY_ASSURANCE,
                department="mock_qa",
                timeout_ms=10000,
                required=False,
                parallel_with=["mock_verification"]
            ),
            StageDefinition(
                stage_type=StageType.VERIFICATION,
                department="mock_verification",
                timeout_ms=5000,
                required=False,
                parallel_with=["mock_qa"]
            ),
        ],
        expected_latency_ms=300
    )


# Core Framework Tests

@pytest.mark.asyncio
async def test_basic_pipeline_execution(mock_registry, simple_pipeline):
    """Test basic pipeline execution."""
    config = Config.fast()
    framework = create_integration_framework(mock_registry, config)

    query = Query(text="Test query")
    result = await framework.execute_pipeline(query, simple_pipeline)

    # Check result
    assert result.success
    assert result.pipeline_name == "test_simple"
    assert len(result.stage_results) == 2
    assert "mock_retrieval" in result.stage_results
    assert "mock_reasoning" in result.stage_results

    # Check retrieval stage
    retrieval_result = result.stage_results["mock_retrieval"]
    assert retrieval_result.success
    assert retrieval_result.result["memories"]

    # Check reasoning stage
    reasoning_result = result.stage_results["mock_reasoning"]
    assert reasoning_result.success
    assert reasoning_result.result["response"]

    # Check overall confidence
    assert result.overall_confidence > 0.0
    assert result.total_duration_ms > 0


@pytest.mark.asyncio
async def test_parallel_stage_execution(mock_registry, parallel_pipeline):
    """Test parallel stage execution."""
    config = Config.fast()
    framework = create_integration_framework(mock_registry, config)

    query = Query(text="Test parallel query")
    result = await framework.execute_pipeline(query, parallel_pipeline)

    # Check result
    assert result.success
    assert len(result.stage_results) == 4

    # Check QA and Verification both executed
    assert "mock_qa" in result.stage_results
    assert "mock_verification" in result.stage_results

    qa_result = result.stage_results["mock_qa"]
    verify_result = result.stage_results["mock_verification"]

    assert qa_result.success
    assert verify_result.success

    # Check that parallel stages saved time
    # (80ms QA + 60ms Verification = 140ms if sequential, but ~80-90ms if parallel)
    # Total should be less than sequential execution would be
    sequential_time = 50 + 100 + 80 + 60  # retrieval + reasoning + qa + verification
    assert result.total_duration_ms < sequential_time


@pytest.mark.asyncio
async def test_graceful_degradation(mock_registry):
    """Test graceful degradation when optional stage fails."""
    config = Config.fast()
    framework = create_integration_framework(mock_registry, config)

    # Create pipeline with failing optional stage
    failing_qa = MockQADepartment(should_fail=True)
    await mock_registry.register(failing_qa, name="failing_qa", domain="qa")

    pipeline = PipelineDefinition(
        name="test_degradation",
        description="Pipeline with failing optional stage",
        stages=[
            StageDefinition(
                stage_type=StageType.RETRIEVAL,
                department="mock_retrieval",
                timeout_ms=5000,
                required=True
            ),
            StageDefinition(
                stage_type=StageType.REASONING,
                department="mock_reasoning",
                timeout_ms=10000,
                required=True
            ),
            StageDefinition(
                stage_type=StageType.QUALITY_ASSURANCE,
                department="failing_qa",
                timeout_ms=10000,
                required=False  # Optional stage
            ),
        ],
        expected_latency_ms=200
    )

    query = Query(text="Test degradation")
    result = await framework.execute_pipeline(query, pipeline)

    # Pipeline should still succeed
    assert result.success

    # Required stages should succeed
    assert result.stage_results["mock_retrieval"].success
    assert result.stage_results["mock_reasoning"].success

    # Optional QA stage should fail gracefully
    assert not result.stage_results["failing_qa"].success
    assert result.stage_results["failing_qa"].error_message


@pytest.mark.asyncio
async def test_confidence_aggregation(mock_registry, parallel_pipeline):
    """Test confidence aggregation across stages."""
    config = Config.fast()
    framework = create_integration_framework(mock_registry, config)

    query = Query(text="Test confidence")
    result = await framework.execute_pipeline(query, parallel_pipeline)

    # Check individual confidences
    retrieval_conf = result.stage_results["mock_retrieval"].confidence
    reasoning_conf = result.stage_results["mock_reasoning"].confidence
    qa_conf = result.stage_results["mock_qa"].confidence
    verify_conf = result.stage_results["mock_verification"].confidence

    assert retrieval_conf == 0.9
    assert reasoning_conf == 0.85
    assert qa_conf == 0.95
    assert verify_conf == 0.9

    # Overall confidence should be weighted average
    # (retrieval: 0.3, reasoning: 0.5, qa: 0.1, verify: 0.1)
    expected = 0.3 * 0.9 + 0.5 * 0.85 + 0.1 * 0.95 + 0.1 * 0.9
    assert abs(result.overall_confidence - expected) < 0.01


@pytest.mark.asyncio
async def test_timeout_handling(mock_registry):
    """Test timeout handling for slow stages."""
    config = Config.fast()
    framework = create_integration_framework(mock_registry, config)

    # Create slow department
    slow_dept = MockReasoningDepartment(latency_ms=2000)  # 2 seconds
    await mock_registry.register(slow_dept, name="slow_reasoning", domain="reasoning")

    pipeline = PipelineDefinition(
        name="test_timeout",
        description="Pipeline with timeout",
        stages=[
            StageDefinition(
                stage_type=StageType.RETRIEVAL,
                department="mock_retrieval",
                timeout_ms=5000,
                required=True
            ),
            StageDefinition(
                stage_type=StageType.REASONING,
                department="slow_reasoning",
                timeout_ms=500,  # Very short timeout
                required=False
            ),
        ],
        expected_latency_ms=200
    )

    query = Query(text="Test timeout")
    result = await framework.execute_pipeline(query, pipeline)

    # Pipeline should still succeed (reasoning is optional)
    assert result.success

    # Retrieval should succeed
    assert result.stage_results["mock_retrieval"].success

    # Reasoning should timeout
    reasoning_result = result.stage_results["slow_reasoning"]
    assert not reasoning_result.success
    assert "timeout" in reasoning_result.error_message.lower()


@pytest.mark.asyncio
async def test_required_stage_failure(mock_registry):
    """Test pipeline failure when required stage fails."""
    config = Config.fast()
    framework = create_integration_framework(mock_registry, config)

    # Create failing required department
    failing_retrieval = MockRetrievalDepartment()
    failing_retrieval.execute = AsyncMock(side_effect=RuntimeError("Retrieval failed"))
    await mock_registry.register(failing_retrieval, name="failing_retrieval", domain="retrieval")

    pipeline = PipelineDefinition(
        name="test_required_failure",
        description="Pipeline with failing required stage",
        stages=[
            StageDefinition(
                stage_type=StageType.RETRIEVAL,
                department="failing_retrieval",
                timeout_ms=5000,
                required=True  # Required stage
            ),
            StageDefinition(
                stage_type=StageType.REASONING,
                department="mock_reasoning",
                timeout_ms=10000,
                required=True
            ),
        ],
        expected_latency_ms=200
    )

    query = Query(text="Test required failure")
    result = await framework.execute_pipeline(query, pipeline)

    # Pipeline should fail
    assert not result.success

    # Retrieval should have error
    assert not result.stage_results["failing_retrieval"].success

    # Reasoning should not execute (critical path broken)
    assert "mock_reasoning" not in result.stage_results or \
           not result.stage_results["mock_reasoning"].success


# DependencyGraph Tests

@pytest.mark.asyncio
async def test_dependency_graph_parallel_groups():
    """Test dependency graph identifies parallel execution groups."""
    pipeline = PipelineDefinition(
        name="test_deps",
        description="Test dependency graph",
        stages=[
            StageDefinition(StageType.RETRIEVAL, "stage1", required=True),
            StageDefinition(StageType.REASONING, "stage2", required=True),
            # Parallel group
            StageDefinition(StageType.QUALITY_ASSURANCE, "stage3",
                          parallel_with=["stage4"]),
            StageDefinition(StageType.VERIFICATION, "stage4",
                          parallel_with=["stage3"]),
            StageDefinition(StageType.OUTPUT, "stage5", required=True),
        ]
    )

    dep_graph = DependencyGraph(pipeline)
    groups = dep_graph.get_parallel_groups()

    # Should have 4 groups: [stage1], [stage2], [stage3, stage4], [stage5]
    assert len(groups) == 4
    assert len(groups[0]) == 1  # stage1
    assert len(groups[1]) == 1  # stage2
    assert len(groups[2]) == 2  # stage3 + stage4 (parallel)
    assert len(groups[3]) == 1  # stage5


@pytest.mark.asyncio
async def test_dependency_graph_no_cycles():
    """Test dependency graph detects cycles."""
    pipeline = PipelineDefinition(
        name="test_cycle",
        description="Test cycle detection",
        stages=[
            StageDefinition(StageType.RETRIEVAL, "stage1",
                          depends_on=["stage3"]),  # Creates cycle
            StageDefinition(StageType.REASONING, "stage2",
                          depends_on=["stage1"]),
            StageDefinition(StageType.OUTPUT, "stage3",
                          depends_on=["stage2"]),
        ]
    )

    # Should raise error due to cycle
    with pytest.raises(ValueError, match="cycle"):
        dep_graph = DependencyGraph(pipeline)
        groups = dep_graph.get_parallel_groups()


# Pipeline Registry Tests

@pytest.mark.asyncio
async def test_pipeline_registry():
    """Test pipeline registry functions."""
    # List all pipelines
    pipelines = list_pipelines()
    assert len(pipelines) >= 6
    assert "simple" in pipelines
    assert "verified" in pipelines
    assert "quality_assured" in pipelines

    # Get specific pipeline
    simple = get_pipeline("simple")
    assert simple.name == "simple"
    assert len(simple.stages) >= 2

    # Get pipeline info
    from HoloLoom.integration.pipelines import get_pipeline_info
    info = get_pipeline_info("quality_assured")
    assert info["name"] == "quality_assured"
    assert "stages" in info
    assert "expected_latency_ms" in info


# Integration Test: Full Quality Assured Pipeline

@pytest.mark.asyncio
async def test_quality_assured_pipeline_integration(mock_registry):
    """Test complete quality assured pipeline."""
    config = Config.fused()
    framework = create_integration_framework(mock_registry, config)

    pipeline = get_pipeline("quality_assured")

    # Replace department names with mock versions
    for stage in pipeline.stages:
        if stage.department == "context":
            stage.department = "mock_retrieval"
        elif stage.department == "orchestrator":
            stage.department = "mock_reasoning"
        elif stage.department == "quality_assurance":
            stage.department = "mock_qa"
        elif stage.department == "verification":
            stage.department = "mock_verification"

    query = Query(text="Integration test query")
    result = await framework.execute_pipeline(query, pipeline)

    # Check pipeline executed successfully
    assert result.success
    assert result.overall_confidence > 0.7

    # Check all stages executed
    assert "mock_retrieval" in result.stage_results
    assert "mock_reasoning" in result.stage_results
    assert "mock_qa" in result.stage_results
    assert "mock_verification" in result.stage_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])