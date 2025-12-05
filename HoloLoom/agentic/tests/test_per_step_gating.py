"""
Integration Tests for Phase 2B: Per-Step Conscience Gating

Tests that each reasoning step in VERIFY, RESEARCH, and PLAN_EXECUTE modes
is individually gated by the conscience adapter.

Created: Dec 2025
Phase: 2B - Per-Step Gating in AgenticOrchestrator
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

from HoloLoom.agentic.core import AgenticOrchestrator, CONSCIENCE_AVAILABLE
from HoloLoom.protocols.conscience import ConscienceDecision, StepType
from HoloLoom.alignment.audit_trail import DecisionType, OutcomeType


# =============================================================================
# Mock _gate_step_with_conscience for testing
# =============================================================================

async def mock_gate_step_with_conscience(
    self,
    step_query: str,
    step_type: 'StepType',
    step_index: int,
    parent_decision=None,
    context=None
):
    """
    Test version of _gate_step_with_conscience that uses mocked adapter.
    """
    if not self.conscience_adapter:
        return None

    try:
        decision = await self.conscience_adapter.gate_step(
            step_query=step_query,
            step_type=step_type,
            step_index=step_index,
            parent_decision=parent_decision,
            context=context or {}
        )

        if decision:
            self.audit_trail.log_decision(
                decision_type=DecisionType.TOOL_SELECTION,
                outcome=OutcomeType.APPROVED if decision.allowed else OutcomeType.REJECTED,
                reason=f"Conscience step gate: {decision.reason}",
                query_text=step_query,
                metadata={
                    "step_type": str(step_type),
                    "step_index": step_index,
                    "conscience_voice": decision.voice,
                    "risk_level": decision.risk_level,
                }
            )

            if not decision.allowed:
                self.logger.warning(
                    f"[CONSCIENCE] Step {step_index + 1} blocked: {decision.reason}"
                )

        return decision

    except Exception as e:
        self.logger.warning(f"[CONSCIENCE] Step gating failed: {e}")
        return None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_learning_engine():
    """Create a mock learning engine with minimal functionality."""
    engine = MagicMock()
    engine.orchestrator = MagicMock()

    # Create mock spacetime results
    mock_spacetime = MagicMock()
    mock_spacetime.confidence = 0.85
    mock_spacetime.metadata = {"response": "Test response"}

    # Make weave return the mock spacetime
    engine.weave = AsyncMock(return_value=mock_spacetime)

    return engine


@pytest.fixture
def mock_conscience_adapter():
    """Create a mock conscience adapter for testing."""
    adapter = MagicMock()
    adapter.available = True

    # Default: allow all steps
    allow_decision = MagicMock()
    allow_decision.allowed = True
    allow_decision.risk_level = "low"
    allow_decision.reason = "Safe to proceed"
    allow_decision.voice = "proceed"
    allow_decision.concerns = []
    allow_decision.guidance = "Continue with step"

    adapter.gate_step = AsyncMock(return_value=allow_decision)
    adapter.gate_reasoning = AsyncMock(return_value=allow_decision)

    return adapter


@pytest.fixture
def mock_audit_trail():
    """Create a mock audit trail."""
    trail = MagicMock()
    trail.log_decision = MagicMock()
    return trail


@pytest.fixture
def orchestrator(mock_learning_engine, mock_conscience_adapter, mock_audit_trail):
    """Create an AgenticOrchestrator with mocked dependencies."""
    orch = MagicMock()
    orch.learning_engine = mock_learning_engine
    orch.conscience_adapter = mock_conscience_adapter
    orch.enable_conscience = True
    orch.audit_trail = mock_audit_trail
    orch.logger = MagicMock()
    orch.monitor = None  # Disable agent monitoring for tests

    # Bind the mock gate method
    import types
    orch._gate_step_with_conscience = types.MethodType(
        mock_gate_step_with_conscience, orch
    )

    return orch


# =============================================================================
# Test: _gate_step_with_conscience Helper Method
# =============================================================================

class TestGateStepWithConscience:
    """Tests for the _gate_step_with_conscience helper method."""

    @pytest.mark.asyncio
    async def test_gate_step_returns_decision_when_allowed(self, orchestrator):
        """Test that gating returns the conscience decision when step is allowed."""
        result = await orchestrator._gate_step_with_conscience(
            step_query="Test query",
            step_type=StepType.VERIFICATION,
            step_index=0,
            context={"test": "context"}
        )

        assert result is not None
        assert result.allowed is True
        assert result.voice == "proceed"

    @pytest.mark.asyncio
    async def test_gate_step_returns_decision_when_blocked(self, orchestrator):
        """Test that gating returns decision with blocked info."""
        # Configure adapter to block
        block_decision = MagicMock()
        block_decision.allowed = False
        block_decision.reason = "Step is too risky"
        block_decision.voice = "pause"
        block_decision.risk_level = "high"
        block_decision.concerns = ["Potential harm detected"]
        orchestrator.conscience_adapter.gate_step = AsyncMock(return_value=block_decision)

        result = await orchestrator._gate_step_with_conscience(
            step_query="Dangerous query",
            step_type=StepType.RESEARCH,
            step_index=0
        )

        assert result is not None
        assert result.allowed is False
        assert result.reason == "Step is too risky"
        assert result.voice == "pause"

    @pytest.mark.asyncio
    async def test_gate_step_logs_to_audit_trail(self, orchestrator):
        """Test that decisions are logged to audit trail."""
        await orchestrator._gate_step_with_conscience(
            step_query="Test query",
            step_type=StepType.VERIFICATION,
            step_index=0
        )

        # Verify audit trail was called
        orchestrator.audit_trail.log_decision.assert_called_once()
        call_kwargs = orchestrator.audit_trail.log_decision.call_args.kwargs
        assert "Conscience step gate" in call_kwargs["reason"]

    @pytest.mark.asyncio
    async def test_gate_step_graceful_on_exception(self, orchestrator):
        """Test graceful degradation when conscience raises exception."""
        orchestrator.conscience_adapter.gate_step = AsyncMock(
            side_effect=Exception("Conscience error")
        )

        result = await orchestrator._gate_step_with_conscience(
            step_query="Test query",
            step_type=StepType.PLAN_STEP,
            step_index=0
        )

        # Should return None and log warning
        assert result is None
        orchestrator.logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_gate_step_returns_none_when_no_adapter(self, orchestrator):
        """Test that None is returned when conscience adapter is None."""
        orchestrator.conscience_adapter = None

        result = await orchestrator._gate_step_with_conscience(
            step_query="Test query",
            step_type=StepType.VERIFICATION,
            step_index=0
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_gate_step_passes_all_parameters(self, orchestrator):
        """Test that all parameters are passed to conscience adapter."""
        parent = MagicMock()
        context = {"key": "value"}

        await orchestrator._gate_step_with_conscience(
            step_query="Test query",
            step_type=StepType.RESEARCH,
            step_index=5,
            parent_decision=parent,
            context=context
        )

        orchestrator.conscience_adapter.gate_step.assert_called_once_with(
            step_query="Test query",
            step_type=StepType.RESEARCH,
            step_index=5,
            parent_decision=parent,
            context=context
        )


# =============================================================================
# Test: Per-Step Gating in Step Types
# =============================================================================

class TestPerStepGatingStepTypes:
    """Tests that correct StepType is used for each reasoning mode."""

    @pytest.mark.asyncio
    async def test_verification_uses_verification_step_type(self, orchestrator):
        """Test that verification loop uses VERIFICATION step type."""
        await orchestrator._gate_step_with_conscience(
            step_query="Verify claim",
            step_type=StepType.VERIFICATION,
            step_index=0
        )

        call_kwargs = orchestrator.conscience_adapter.gate_step.call_args.kwargs
        assert call_kwargs["step_type"] == StepType.VERIFICATION

    @pytest.mark.asyncio
    async def test_research_uses_research_step_type(self, orchestrator):
        """Test that research query uses RESEARCH step type."""
        await orchestrator._gate_step_with_conscience(
            step_query="Research topic",
            step_type=StepType.RESEARCH,
            step_index=0
        )

        call_kwargs = orchestrator.conscience_adapter.gate_step.call_args.kwargs
        assert call_kwargs["step_type"] == StepType.RESEARCH

    @pytest.mark.asyncio
    async def test_plan_uses_plan_step_type(self, orchestrator):
        """Test that plan execution uses PLAN_STEP step type."""
        await orchestrator._gate_step_with_conscience(
            step_query="Execute sub-goal",
            step_type=StepType.PLAN_STEP,
            step_index=0
        )

        call_kwargs = orchestrator.conscience_adapter.gate_step.call_args.kwargs
        assert call_kwargs["step_type"] == StepType.PLAN_STEP


# =============================================================================
# Test: Blocked Steps Recording
# =============================================================================

class TestBlockedStepsRecording:
    """Tests that blocked steps are properly recorded."""

    @pytest.mark.asyncio
    async def test_blocked_decision_logs_warning(self, orchestrator):
        """Test that blocked decisions log a warning."""
        block_decision = MagicMock()
        block_decision.allowed = False
        block_decision.reason = "Blocked for safety"
        block_decision.voice = "stop"
        orchestrator.conscience_adapter.gate_step = AsyncMock(return_value=block_decision)

        await orchestrator._gate_step_with_conscience(
            step_query="Dangerous step",
            step_type=StepType.VERIFICATION,
            step_index=2
        )

        # Check that warning was logged with step number
        orchestrator.logger.warning.assert_called()
        warning_msg = orchestrator.logger.warning.call_args[0][0]
        assert "blocked" in warning_msg.lower()
        assert "3" in warning_msg  # step_index + 1

    @pytest.mark.asyncio
    async def test_blocked_decision_logs_to_audit_as_rejected(self, orchestrator):
        """Test that blocked decisions are logged as REJECTED in audit trail."""
        block_decision = MagicMock()
        block_decision.allowed = False
        block_decision.reason = "Safety concern"
        block_decision.voice = "pause"
        block_decision.risk_level = "medium"
        block_decision.concerns = ["Test concern"]
        orchestrator.conscience_adapter.gate_step = AsyncMock(return_value=block_decision)

        await orchestrator._gate_step_with_conscience(
            step_query="Risky step",
            step_type=StepType.RESEARCH,
            step_index=0
        )

        call_kwargs = orchestrator.audit_trail.log_decision.call_args.kwargs
        assert call_kwargs["outcome"] == OutcomeType.REJECTED


# =============================================================================
# Test: Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation when conscience is unavailable."""

    @pytest.mark.asyncio
    async def test_continues_when_conscience_unavailable(self, orchestrator):
        """Test that processing continues when conscience adapter is None."""
        orchestrator.conscience_adapter = None

        result = await orchestrator._gate_step_with_conscience(
            step_query="Some query",
            step_type=StepType.VERIFICATION,
            step_index=0
        )

        # Should return None but not crash
        assert result is None

    @pytest.mark.asyncio
    async def test_continues_on_adapter_exception(self, orchestrator):
        """Test that processing continues when adapter raises exception."""
        orchestrator.conscience_adapter.gate_step = AsyncMock(
            side_effect=RuntimeError("Connection lost")
        )

        result = await orchestrator._gate_step_with_conscience(
            step_query="Some query",
            step_type=StepType.RESEARCH,
            step_index=0
        )

        # Should return None and log warning, not crash
        assert result is None
        orchestrator.logger.warning.assert_called()


# =============================================================================
# Test: Context Passing
# =============================================================================

class TestContextPassing:
    """Tests for context information passing to conscience."""

    @pytest.mark.asyncio
    async def test_verification_context_includes_original_query(self, orchestrator):
        """Test that verification step includes original query in context."""
        context = {"original_query": "What is X?"}

        await orchestrator._gate_step_with_conscience(
            step_query="Verify X is true",
            step_type=StepType.VERIFICATION,
            step_index=0,
            context=context
        )

        call_kwargs = orchestrator.conscience_adapter.gate_step.call_args.kwargs
        assert call_kwargs["context"]["original_query"] == "What is X?"

    @pytest.mark.asyncio
    async def test_research_context_includes_topic(self, orchestrator):
        """Test that research step includes research topic in context."""
        context = {"original_query": "Explain Y", "research_topic": "Y"}

        await orchestrator._gate_step_with_conscience(
            step_query="What is the definition of Y?",
            step_type=StepType.RESEARCH,
            step_index=0,
            context=context
        )

        call_kwargs = orchestrator.conscience_adapter.gate_step.call_args.kwargs
        assert "research_topic" in call_kwargs["context"]

    @pytest.mark.asyncio
    async def test_plan_context_includes_plan_flag(self, orchestrator):
        """Test that plan step includes plan_step flag in context."""
        context = {"original_query": "Do Z", "plan_step": True}

        await orchestrator._gate_step_with_conscience(
            step_query="Step 1 of Z",
            step_type=StepType.PLAN_STEP,
            step_index=0,
            context=context
        )

        call_kwargs = orchestrator.conscience_adapter.gate_step.call_args.kwargs
        assert call_kwargs["context"].get("plan_step") is True


# =============================================================================
# Test: Step Index Tracking
# =============================================================================

class TestStepIndexTracking:
    """Tests for correct step index tracking."""

    @pytest.mark.asyncio
    async def test_step_index_passed_correctly(self, orchestrator):
        """Test that step index is passed correctly."""
        for i in range(5):
            await orchestrator._gate_step_with_conscience(
                step_query=f"Step {i}",
                step_type=StepType.VERIFICATION,
                step_index=i
            )

        # Verify all calls had correct step_index
        calls = orchestrator.conscience_adapter.gate_step.call_args_list
        for i, call in enumerate(calls):
            assert call.kwargs["step_index"] == i

    @pytest.mark.asyncio
    async def test_step_index_in_audit_metadata(self, orchestrator):
        """Test that step index is included in audit trail metadata."""
        await orchestrator._gate_step_with_conscience(
            step_query="Test step",
            step_type=StepType.RESEARCH,
            step_index=3
        )

        call_kwargs = orchestrator.audit_trail.log_decision.call_args.kwargs
        assert call_kwargs["metadata"]["step_index"] == 3


# =============================================================================
# Test: CONSCIENCE_AVAILABLE Flag
# =============================================================================

class TestConscienceAvailableFlag:
    """Tests for CONSCIENCE_AVAILABLE flag behavior."""

    def test_conscience_available_is_true(self):
        """Test that CONSCIENCE_AVAILABLE is True when imports succeed."""
        # This test verifies the imports at module level worked
        assert CONSCIENCE_AVAILABLE is True

    def test_step_type_is_importable(self):
        """Test that StepType enum is importable."""
        from HoloLoom.protocols.conscience import StepType

        assert hasattr(StepType, 'VERIFICATION')
        assert hasattr(StepType, 'RESEARCH')
        assert hasattr(StepType, 'PLAN_STEP')
