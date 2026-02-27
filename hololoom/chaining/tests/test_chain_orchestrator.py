"""
Tests for Chain Orchestration System

Tests cover:
- Chain definition and validation
- Step execution (all types)
- Context passing between steps
- Conditional branching
- Loop constructs
- Error handling
- Pre-built patterns
- Integration with RAG Department

Author: HoloLoom Testing
Date: November 2025
"""

import pytest
import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from hololoom.chaining import (
    Chain,
    ChainStep,
    StepType,
    ChainOrchestrator,
    ChainPatterns,
    Conditions,
    CommonConditions,
)
from hololoom.protocols.department import DepartmentRequest, DepartmentResponse, ConfidenceMetadata


# ============================================================================
# Mock Department for Testing
# ============================================================================

class MockDepartment:
    """Mock department for testing."""

    def __init__(self):
        self.department_id = "mock_rag"
        self.executed = False
        self.verified = False

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Mock execute."""
        self.executed = True
        return DepartmentResponse(
            task_id=request.task_id,
            result="This is a test answer",
            confidence=ConfidenceMetadata.from_score(0.85),
            metadata={
                "response_data": {
                    "answer": "This is a test answer",
                    "sources": ["source1", "source2"],
                    "reasoning_mode": request.parameters.get("mode", "verify"),
                }
            },
        )

    async def verify(self, response: DepartmentResponse):
        """Mock verify."""
        self.verified = True
        return MagicMock(
            verified=True,
            overall_score=0.90,
            checks=[
                MagicMock(dimension="Domain", passed=True, score=0.95),
                MagicMock(dimension="Sensibility", passed=True, score=0.90),
            ],
            recommendations=[],
        )

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """Mock refine."""
        response.confidence = ConfidenceMetadata.from_score(
            min(1.0, response.confidence.score + 0.07)
        )
        return response

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """Mock update strategy."""
        pass


# ============================================================================
# Chain Definition Tests
# ============================================================================

class TestChainDefinition:
    """Test chain definition and validation."""

    def test_chain_creation(self):
        """Test creating a new chain."""
        chain = Chain(name="test_chain", entry_point="start")
        assert chain.name == "test_chain"
        assert chain.entry_point == "start"
        assert len(chain.steps) == 0

    def test_add_step(self):
        """Test adding a step to a chain."""
        chain = Chain(name="test_chain", entry_point="start")
        step = ChainStep(step_type=StepType.EXECUTE, params={"mode": "verify"})
        chain.add_step("start", step)

        assert "start" in chain.steps
        assert chain.steps["start"].step_type == StepType.EXECUTE

    def test_add_sequential_steps(self):
        """Test adding multiple sequential steps."""
        chain = Chain(name="test_chain", entry_point="start")
        chain.add_sequential_steps(
            ["start", "verify", "end"],
            StepType.EXECUTE,
        )

        assert len(chain.steps) == 3
        assert chain.steps["start"].next_step == "verify"
        assert chain.steps["verify"].next_step == "end"
        assert chain.steps["end"].next_step is None

    def test_validate_missing_entry_point(self):
        """Test validation fails for missing entry point."""
        chain = Chain(name="test_chain", entry_point="start")
        step = ChainStep(step_type=StepType.EXECUTE)
        chain.add_step("other", step)

        errors = chain.validate()
        assert any("Entry point" in e for e in errors)

    def test_validate_condition_without_function(self):
        """Test validation fails for condition step without condition function."""
        chain = Chain(name="test_chain", entry_point="cond")
        step = ChainStep(step_type=StepType.CONDITION)  # No condition function
        chain.add_step("cond", step)

        errors = chain.validate()
        assert any("condition()" in e for e in errors)

    def test_validate_valid_chain(self):
        """Test validation passes for valid chain."""
        chain = Chain(name="test_chain", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            next_step="verify",
        ))
        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
        ))

        errors = chain.validate()
        assert len(errors) == 0

    def test_chain_visualization(self):
        """Test chain visualization."""
        chain = Chain(name="test_chain", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            next_step="verify",
        ))
        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
        ))

        viz = chain.visualize()
        assert "test_chain" in viz
        assert "execute" in viz
        assert "verify" in viz

    def test_chain_to_json(self):
        """Test chain serialization to JSON."""
        chain = Chain(name="test_chain", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={"mode": "verify"},
        ))

        json_str = chain.to_json()
        assert "test_chain" in json_str
        assert "execute" in json_str
        assert "verify" in json_str


# ============================================================================
# Condition Tests
# ============================================================================

class TestConditions:
    """Test condition functions."""

    def test_confidence_above(self):
        """Test confidence_above condition."""
        cond = Conditions.confidence_above(0.75)

        assert cond({"confidence": 0.90}) is True
        assert cond({"confidence": 0.75}) is True
        assert cond({"confidence": 0.70}) is False
        assert cond({}) is False  # Default 0.0

    def test_confidence_below(self):
        """Test confidence_below condition."""
        cond = Conditions.confidence_below(0.75)

        assert cond({"confidence": 0.70}) is True
        assert cond({"confidence": 0.75}) is False
        assert cond({"confidence": 0.90}) is False

    def test_has_sources(self):
        """Test has_sources condition."""
        cond = Conditions.has_sources(min_count=2)

        assert cond({"sources": ["s1", "s2", "s3"]}) is True
        assert cond({"sources": ["s1"]}) is False
        assert cond({}) is False

    def test_all_checks_passed(self):
        """Test all_checks_passed condition."""
        cond = Conditions.all_checks_passed()

        assert cond({
            "verification_checks": [
                {"passed": True},
                {"passed": True},
            ]
        }) is True

        assert cond({
            "verification_checks": [
                {"passed": True},
                {"passed": False},
            ]
        }) is False

    def test_combine_and(self):
        """Test combining conditions with AND."""
        cond = Conditions.combine_and(
            Conditions.confidence_above(0.75),
            Conditions.has_sources(min_count=1),
        )

        assert cond({"confidence": 0.90, "sources": ["s1"]}) is True
        assert cond({"confidence": 0.70, "sources": ["s1"]}) is False
        assert cond({"confidence": 0.90, "sources": []}) is False

    def test_combine_or(self):
        """Test combining conditions with OR."""
        cond = Conditions.combine_or(
            Conditions.confidence_above(0.75),
            Conditions.has_sources(min_count=5),
        )

        assert cond({"confidence": 0.90, "sources": ["s1"]}) is True
        assert cond({"confidence": 0.70, "sources": ["s1", "s2", "s3", "s4", "s5"]}) is True
        assert cond({"confidence": 0.70, "sources": ["s1"]}) is False

    def test_common_conditions(self):
        """Test pre-built common conditions."""
        high_conf = CommonConditions.high_confidence()
        low_conf = CommonConditions.low_confidence()

        ctx_high = {"confidence": 0.90, "sources": ["s1"]}
        ctx_low = {"confidence": 0.60, "response": {"answer": "test"}}

        assert high_conf(ctx_high) is True
        assert low_conf(ctx_low) is True


# ============================================================================
# Chain Orchestrator Tests
# ============================================================================

class TestChainOrchestrator:
    """Test chain orchestration and execution."""

    @pytest.mark.asyncio
    async def test_simple_execute(self):
        """Test simple execution chain."""
        dept = MockDepartment()
        orchestrator = ChainOrchestrator(dept)

        chain = ChainPatterns.simple_query()
        result = await orchestrator.execute_chain(chain, "What is AI?")

        assert result.success is True
        assert result.confidence is not None
        assert dept.executed is True

    @pytest.mark.asyncio
    async def test_verified_query(self):
        """Test verified query chain."""
        dept = MockDepartment()
        orchestrator = ChainOrchestrator(dept)

        chain = ChainPatterns.verified_query()
        result = await orchestrator.execute_chain(chain, "What is AI?")

        assert result.success is True
        assert dept.executed is True
        assert dept.verified is True

    @pytest.mark.asyncio
    async def test_auto_refine_high_confidence(self):
        """Test auto-refine chain with high confidence (no refinement)."""
        dept = MockDepartment()
        orchestrator = ChainOrchestrator(dept)

        # Mock execute to return high confidence
        dept.execute = AsyncMock(return_value=DepartmentResponse(
            task_id="test",
            result="Test answer",
            confidence=ConfidenceMetadata.from_score(0.90),
            metadata={"response_data": {"answer": "Test answer", "sources": ["s1"]}},
        ))

        chain = ChainPatterns.auto_refine()
        result = await orchestrator.execute_chain(chain, "Test question")

        assert result.success is True
        # refine should not be called (confidence is high)
        assert dept.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_refine_low_confidence(self):
        """Test auto-refine chain with low confidence (triggers refinement)."""
        dept = MockDepartment()

        # Mock execute to return low confidence
        execute_response = DepartmentResponse(
            task_id="test",
            result="Test answer",
            confidence=ConfidenceMetadata.from_score(0.60),
            metadata={
                "response_data": {"answer": "Test answer", "sources": ["s1"]},
                "original_query": "Test question"
            },
        )

        dept.execute = AsyncMock(return_value=execute_response)
        dept.verify = AsyncMock(return_value=MagicMock(
            verified=False,
            overall_score=0.60,
            checks=[],
            recommendations=[],
        ))
        dept.refine = AsyncMock(return_value=execute_response)

        orchestrator = ChainOrchestrator(dept)
        chain = ChainPatterns.auto_refine()
        result = await orchestrator.execute_chain(chain, "Test question")

        assert result.success is True
        # refine should be called (confidence is low)
        assert dept.refine.called

    @pytest.mark.asyncio
    async def test_execution_trace(self):
        """Test execution trace generation."""
        dept = MockDepartment()
        orchestrator = ChainOrchestrator(dept, enable_tracing=True)

        chain = ChainPatterns.verified_query()
        result = await orchestrator.execute_chain(chain, "Test question")

        assert result.trace is not None
        assert len(result.trace.step_results) > 0
        assert result.trace.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_context_passing(self):
        """Test context passing between steps."""
        dept = MockDepartment()
        orchestrator = ChainOrchestrator(dept)

        # Custom step that uses context
        def verify_handler(ctx):
            # Should have output from execute step
            assert "response" in ctx
            return {"verified": True}

        chain = Chain(name="context_test", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={"mode": "verify"},
            next_step="verify",
        ))
        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
        ))

        result = await orchestrator.execute_chain(chain, "Test")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_conditional_branching_true(self):
        """Test conditional branching (true branch)."""
        dept = MockDepartment()

        # Mock high confidence response
        dept.execute = AsyncMock(return_value=DepartmentResponse(
            task_id="test",
            result="High confidence answer",
            confidence=ConfidenceMetadata.from_score(0.90),
            metadata={
                "response_data": {"answer": "High confidence answer", "sources": ["s1"]},
                "original_query": "Test"
            },
        ))

        orchestrator = ChainOrchestrator(dept)

        chain = Chain(name="conditional_test", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={"mode": "verify"},
            next_step="check",
        ))

        # Condition: if confidence >= 0.75, branch to success
        chain.add_step("check", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.75),
            on_success=None,  # Success path (end)
            on_failure="refine",
        ))

        # Mock refine to track calls
        dept.refine = AsyncMock(return_value=DepartmentResponse(
            task_id="test",
            result="Refined answer",
            confidence=ConfidenceMetadata.from_score(0.95),
            metadata={"response_data": {"answer": "Refined"}},
        ))

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
        ))

        result = await orchestrator.execute_chain(chain, "Test")
        assert result.success is True
        # refine should not be called (condition was true)
        assert not dept.refine.called

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in chain execution."""
        dept = MockDepartment()

        # Mock execute to raise an error
        async def error_execute(*args, **kwargs):
            raise ValueError("Test error")

        dept.execute = error_execute

        orchestrator = ChainOrchestrator(dept)
        chain = ChainPatterns.simple_query()

        result = await orchestrator.execute_chain(chain, "Test")
        # Chain execution succeeds but has failed steps
        assert result.success is True
        # But should have a failed step
        assert result.stats.failed_steps > 0
        # And error should be captured in trace
        assert len(result.trace.errors) > 0
        assert "Test error" in result.trace.errors[0]

    @pytest.mark.asyncio
    async def test_step_timeout(self):
        """Test step timeout handling."""
        dept = MockDepartment()

        # Mock execute to take too long
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(1.0)
            return DepartmentResponse(
                task_id="test",
                result="Should timeout",
                confidence=ConfidenceMetadata.from_score(0.8),
                metadata={"response_data": {"answer": "Should timeout"}},
            )

        dept.execute = slow_execute

        orchestrator = ChainOrchestrator(dept)

        chain = Chain(name="timeout_test", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={"mode": "verify"},
            timeout_seconds=0.1,  # 100ms timeout
        ))

        result = await orchestrator.execute_chain(chain, "Test")
        # Chain execution completes, but step fails due to timeout
        assert result.success is True
        assert result.stats.failed_steps > 0
        # Error should be in step results
        assert any("timeout" in str(sr.error).lower() for sr in result.trace.step_results)

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """Test maximum steps safety limit."""
        dept = MockDepartment()
        orchestrator = ChainOrchestrator(dept)

        # Create chain that would loop infinitely
        chain = Chain(name="infinite_loop", entry_point="loop")
        chain.add_step("loop", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=lambda ctx: True,  # Always true = infinite loop
            on_success="loop",  # Loop back to itself
            on_failure=None,
        ))

        result = await orchestrator.execute_chain(
            chain,
            "Test",
            max_total_steps=5,  # Safety limit
        )

        assert result.success is False or result.stats.completed_steps <= 5


# ============================================================================
# Pattern Tests
# ============================================================================

class TestChainPatterns:
    """Test pre-built chain patterns."""

    def test_simple_query_pattern(self):
        """Test simple_query pattern structure."""
        chain = ChainPatterns.simple_query()
        assert chain.name == "simple_query"
        assert "execute" in chain.steps
        errors = chain.validate()
        assert len(errors) == 0

    def test_verified_query_pattern(self):
        """Test verified_query pattern structure."""
        chain = ChainPatterns.verified_query()
        assert chain.name == "verified_query"
        assert "execute" in chain.steps
        assert "verify" in chain.steps
        errors = chain.validate()
        assert len(errors) == 0

    def test_auto_refine_pattern(self):
        """Test auto_refine pattern structure."""
        chain = ChainPatterns.auto_refine()
        assert chain.name == "auto_refine"
        assert "execute" in chain.steps
        assert "verify" in chain.steps
        assert "check_confidence" in chain.steps
        assert "refine" in chain.steps
        errors = chain.validate()
        assert len(errors) == 0

    def test_iterative_improve_pattern(self):
        """Test iterative_improve pattern structure."""
        chain = ChainPatterns.iterative_improve()
        assert "execute" in chain.steps
        assert "verify" in chain.steps
        assert "refine" in chain.steps
        errors = chain.validate()
        assert len(errors) == 0

    def test_multi_strategy_pattern(self):
        """Test multi_strategy pattern structure."""
        chain = ChainPatterns.multi_strategy()
        assert "execute_direct" in chain.steps
        assert "execute_research" in chain.steps
        errors = chain.validate()
        assert len(errors) == 0

    def test_research_pipeline_pattern(self):
        """Test research_pipeline pattern structure."""
        chain = ChainPatterns.research_pipeline()
        assert "execute" in chain.steps
        assert "verify" in chain.steps
        assert "learn" in chain.steps
        errors = chain.validate()
        assert len(errors) == 0

    def test_balanced_pattern(self):
        """Test balanced pattern structure."""
        chain = ChainPatterns.balanced()
        assert "execute" in chain.steps
        assert "verify" in chain.steps
        errors = chain.validate()
        assert len(errors) == 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
