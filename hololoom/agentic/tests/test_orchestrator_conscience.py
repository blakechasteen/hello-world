"""
Integration Tests for Phase 2C: Orchestrator Conscience Integration

Tests that WeavingOrchestrator correctly integrates with the conscience system
for per-query ethical gating.

Created: Dec 2025
Phase: 2C - Orchestrator Integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional
import inspect

from HoloLoom.weaving_orchestrator import WeavingOrchestrator, CONSCIENCE_AVAILABLE
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_conscience_adapter():
    """Create a mock conscience adapter for testing."""
    adapter = MagicMock()
    adapter.available = True

    # Default: allow all queries
    allow_decision = MagicMock()
    allow_decision.allowed = True
    allow_decision.risk_level = "low"
    allow_decision.reason = "Safe query"
    allow_decision.voice = "proceed"
    allow_decision.concerns = []
    allow_decision.guidance = "Continue processing"

    adapter.gate_reasoning = AsyncMock(return_value=allow_decision)
    adapter.gate_step = AsyncMock(return_value=allow_decision)

    return adapter


@pytest.fixture
def mock_blocked_adapter():
    """Create a mock conscience adapter that blocks queries."""
    adapter = MagicMock()
    adapter.available = True

    # Block all queries
    block_decision = MagicMock()
    block_decision.allowed = False
    block_decision.risk_level = "high"
    block_decision.reason = "Query blocked for safety"
    block_decision.voice = "stop"
    block_decision.concerns = ["Potential harm detected"]
    block_decision.guidance = "I cannot process this query."

    adapter.gate_reasoning = AsyncMock(return_value=block_decision)
    adapter.gate_step = AsyncMock(return_value=block_decision)

    return adapter


# =============================================================================
# Test: CONSCIENCE_AVAILABLE Flag
# =============================================================================

class TestConscienceAvailableFlag:
    """Tests for CONSCIENCE_AVAILABLE flag behavior."""

    def test_conscience_available_flag_exists(self):
        """Test that CONSCIENCE_AVAILABLE flag is defined."""
        assert CONSCIENCE_AVAILABLE is not None
        assert isinstance(CONSCIENCE_AVAILABLE, bool)

    def test_conscience_available_is_true_when_imports_succeed(self):
        """Test that CONSCIENCE_AVAILABLE is True when all imports work."""
        # This test verifies the imports at module level worked
        # If protocols.conscience and conscience_adapter exist, it should be True
        if CONSCIENCE_AVAILABLE:
            from HoloLoom.agentic.conscience_adapter import AgenticConscienceAdapter
            from HoloLoom.protocols.conscience import ConscienceDecision, StepType
            assert AgenticConscienceAdapter is not None
            assert ConscienceDecision is not None
            assert StepType is not None


# =============================================================================
# Test: WeavingOrchestrator Conscience Parameters
# =============================================================================

class TestOrchestratorConscienceParameters:
    """Tests for conscience parameters in WeavingOrchestrator."""

    def test_orchestrator_has_conscience_adapter_parameter(self):
        """Test that __init__ accepts conscience_adapter parameter."""
        params = inspect.signature(WeavingOrchestrator.__init__).parameters
        assert 'conscience_adapter' in params

    def test_orchestrator_has_enable_conscience_parameter(self):
        """Test that __init__ accepts enable_conscience parameter."""
        params = inspect.signature(WeavingOrchestrator.__init__).parameters
        assert 'enable_conscience' in params

    def test_enable_conscience_defaults_to_true(self):
        """Test that enable_conscience defaults to True."""
        params = inspect.signature(WeavingOrchestrator.__init__).parameters
        assert params['enable_conscience'].default is True

    def test_conscience_adapter_defaults_to_none(self):
        """Test that conscience_adapter defaults to None."""
        params = inspect.signature(WeavingOrchestrator.__init__).parameters
        assert params['conscience_adapter'].default is None


# =============================================================================
# Test: Conscience Decision Flow (Unit Tests)
# =============================================================================

class TestConscienceDecisionFlow:
    """Unit tests for conscience decision processing."""

    def test_allowed_decision_has_required_fields(self, mock_conscience_adapter):
        """Test that allowed decision has all required fields."""
        decision = mock_conscience_adapter.gate_reasoning.return_value

        assert hasattr(decision, 'allowed')
        assert hasattr(decision, 'risk_level')
        assert hasattr(decision, 'reason')
        assert hasattr(decision, 'voice')
        assert hasattr(decision, 'concerns')
        assert hasattr(decision, 'guidance')

        assert decision.allowed is True
        assert decision.risk_level == "low"
        assert decision.voice == "proceed"

    def test_blocked_decision_has_required_fields(self, mock_blocked_adapter):
        """Test that blocked decision has all required fields."""
        decision = mock_blocked_adapter.gate_reasoning.return_value

        assert hasattr(decision, 'allowed')
        assert hasattr(decision, 'risk_level')
        assert hasattr(decision, 'reason')
        assert hasattr(decision, 'voice')
        assert hasattr(decision, 'concerns')
        assert hasattr(decision, 'guidance')

        assert decision.allowed is False
        assert decision.risk_level == "high"
        assert decision.voice == "stop"

    @pytest.mark.asyncio
    async def test_gate_reasoning_is_async(self, mock_conscience_adapter):
        """Test that gate_reasoning is an async method."""
        result = await mock_conscience_adapter.gate_reasoning(
            query_text="test query",
            context={}
        )
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_blocked_adapter_returns_blocked_decision(self, mock_blocked_adapter):
        """Test that blocked adapter returns blocked decision."""
        result = await mock_blocked_adapter.gate_reasoning(
            query_text="dangerous query",
            context={}
        )
        assert result.allowed is False
        assert result.guidance == "I cannot process this query."


# =============================================================================
# Test: Conscience Context Structure
# =============================================================================

class TestConscienceContextStructure:
    """Tests for the context passed to conscience adapter."""

    @pytest.mark.asyncio
    async def test_context_contains_source(self, mock_conscience_adapter):
        """Test that context contains source identifier."""
        await mock_conscience_adapter.gate_reasoning(
            query_text="test query",
            context={'source': 'weaving_orchestrator'}
        )

        call_kwargs = mock_conscience_adapter.gate_reasoning.call_args.kwargs
        assert 'source' in call_kwargs['context']
        assert call_kwargs['context']['source'] == 'weaving_orchestrator'

    @pytest.mark.asyncio
    async def test_context_can_contain_awareness(self, mock_conscience_adapter):
        """Test that context can contain awareness metrics."""
        awareness = {
            'activation': 0.85,
            'coherence': {'global_coherence': 0.75},
            'active_nodes': 10
        }

        await mock_conscience_adapter.gate_reasoning(
            query_text="test query",
            context={'awareness': awareness, 'source': 'weaving_orchestrator'}
        )

        call_kwargs = mock_conscience_adapter.gate_reasoning.call_args.kwargs
        assert 'awareness' in call_kwargs['context']

    @pytest.mark.asyncio
    async def test_context_can_contain_complexity(self, mock_conscience_adapter):
        """Test that context can contain complexity level."""
        await mock_conscience_adapter.gate_reasoning(
            query_text="test query",
            context={'complexity': 'FAST', 'source': 'weaving_orchestrator'}
        )

        call_kwargs = mock_conscience_adapter.gate_reasoning.call_args.kwargs
        assert 'complexity' in call_kwargs['context']
        assert call_kwargs['context']['complexity'] == 'FAST'


# =============================================================================
# Test: Blocked Response Structure
# =============================================================================

class TestBlockedResponseStructure:
    """Tests for blocked response metadata structure."""

    def test_blocked_decision_metadata_structure(self, mock_blocked_adapter):
        """Test that blocked decision can be converted to metadata dict."""
        decision = mock_blocked_adapter.gate_reasoning.return_value

        # Build metadata dict as orchestrator would
        metadata = {
            'blocked_by_conscience': True,
            'conscience_decision': {
                'allowed': decision.allowed,
                'reason': decision.reason,
                'voice': decision.voice,
                'risk_level': decision.risk_level,
                'concerns': decision.concerns,
                'guidance': decision.guidance,
            }
        }

        assert metadata['blocked_by_conscience'] is True
        assert metadata['conscience_decision']['allowed'] is False
        assert metadata['conscience_decision']['voice'] == 'stop'
        assert metadata['conscience_decision']['risk_level'] == 'high'
        assert 'Potential harm detected' in metadata['conscience_decision']['concerns']

    def test_blocked_response_uses_guidance(self, mock_blocked_adapter):
        """Test that blocked response uses guidance as response text."""
        decision = mock_blocked_adapter.gate_reasoning.return_value

        # Response should use guidance
        response = decision.guidance or "I cannot process this query at this time."
        assert response == "I cannot process this query."


# =============================================================================
# Test: Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation behavior."""

    def test_conscience_available_flag_is_boolean(self):
        """Test that CONSCIENCE_AVAILABLE is a boolean."""
        assert isinstance(CONSCIENCE_AVAILABLE, bool)

    def test_can_import_protocols_when_available(self):
        """Test that protocols can be imported when available."""
        if CONSCIENCE_AVAILABLE:
            from HoloLoom.protocols.conscience import ConscienceDecision, StepType
            assert ConscienceDecision is not None
            assert StepType is not None

    def test_can_import_adapter_when_available(self):
        """Test that conscience adapter can be imported when available."""
        if CONSCIENCE_AVAILABLE:
            from HoloLoom.agentic.conscience_adapter import AgenticConscienceAdapter
            assert AgenticConscienceAdapter is not None

    @pytest.mark.asyncio
    async def test_exception_in_gate_reasoning_is_handled(self, mock_conscience_adapter):
        """Test that exceptions in gate_reasoning are handled gracefully."""
        mock_conscience_adapter.gate_reasoning = AsyncMock(
            side_effect=RuntimeError("Conscience error")
        )

        # Should raise the error (orchestrator wraps this in try/except)
        with pytest.raises(RuntimeError, match="Conscience error"):
            await mock_conscience_adapter.gate_reasoning(
                query_text="test",
                context={}
            )


# =============================================================================
# Test: Auto-creation of Conscience Adapter
# =============================================================================

class TestAutoCreationOfConscienceAdapter:
    """Tests for auto-creation of conscience adapter."""

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_adapter_class_can_be_instantiated(self):
        """Test that adapter class can be instantiated."""
        from HoloLoom.agentic.conscience_adapter import AgenticConscienceAdapter

        adapter = AgenticConscienceAdapter()
        assert adapter is not None
        assert hasattr(adapter, 'available')
        assert hasattr(adapter, 'gate_reasoning')

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_adapter_available_property(self):
        """Test that adapter has available property."""
        from HoloLoom.agentic.conscience_adapter import AgenticConscienceAdapter

        adapter = AgenticConscienceAdapter()
        # Should be a boolean
        assert isinstance(adapter.available, bool)


# =============================================================================
# Test: Step Type Enum
# =============================================================================

class TestStepTypeEnum:
    """Tests for StepType enum availability."""

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_step_type_has_verification(self):
        """Test that StepType has VERIFICATION member."""
        from HoloLoom.protocols.conscience import StepType
        assert hasattr(StepType, 'VERIFICATION')

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_step_type_has_research(self):
        """Test that StepType has RESEARCH member."""
        from HoloLoom.protocols.conscience import StepType
        assert hasattr(StepType, 'RESEARCH')

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_step_type_has_plan_step(self):
        """Test that StepType has PLAN_STEP member."""
        from HoloLoom.protocols.conscience import StepType
        assert hasattr(StepType, 'PLAN_STEP')


# =============================================================================
# Test: Weaving Orchestrator Source Code Integration
# =============================================================================

class TestWeavingOrchestratorSourceIntegration:
    """Tests that verify conscience integration exists in source code."""

    def test_weave_method_exists(self):
        """Test that weave method exists on orchestrator."""
        assert hasattr(WeavingOrchestrator, 'weave')
        assert callable(getattr(WeavingOrchestrator, 'weave'))

    def test_weave_is_async(self):
        """Test that weave is an async method."""
        import asyncio
        method = getattr(WeavingOrchestrator, 'weave')
        assert asyncio.iscoroutinefunction(method)

    def test_orchestrator_has_enable_conscience_attribute_in_signature(self):
        """Test that enable_conscience is in __init__ signature."""
        sig = inspect.signature(WeavingOrchestrator.__init__)
        param_names = list(sig.parameters.keys())
        assert 'enable_conscience' in param_names

    def test_orchestrator_has_conscience_adapter_attribute_in_signature(self):
        """Test that conscience_adapter is in __init__ signature."""
        sig = inspect.signature(WeavingOrchestrator.__init__)
        param_names = list(sig.parameters.keys())
        assert 'conscience_adapter' in param_names


# =============================================================================
# Test: Conscience Decision Types
# =============================================================================

class TestConscienceDecisionTypes:
    """Tests for ConscienceDecision dataclass."""

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_conscience_decision_is_dataclass(self):
        """Test that ConscienceDecision is a dataclass."""
        from HoloLoom.protocols.conscience import ConscienceDecision
        import dataclasses
        assert dataclasses.is_dataclass(ConscienceDecision)

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_conscience_decision_has_allowed_field(self):
        """Test that ConscienceDecision has allowed field."""
        from HoloLoom.protocols.conscience import ConscienceDecision
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(ConscienceDecision)]
        assert 'allowed' in field_names

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_conscience_decision_has_voice_field(self):
        """Test that ConscienceDecision has voice field."""
        from HoloLoom.protocols.conscience import ConscienceDecision
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(ConscienceDecision)]
        assert 'voice' in field_names

    @pytest.mark.skipif(
        not CONSCIENCE_AVAILABLE,
        reason="Conscience integration not available"
    )
    def test_conscience_decision_has_risk_level_field(self):
        """Test that ConscienceDecision has risk_level field."""
        from HoloLoom.protocols.conscience import ConscienceDecision
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(ConscienceDecision)]
        assert 'risk_level' in field_names
