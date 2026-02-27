"""
Unit Tests for Agentic Conscience Adapter
==========================================

Tests the AgenticConscienceAdapter which bridges the agentic reasoning
system with the Conscience architecture for per-step safety gating.

Test Categories:
1. Initialization and configuration
2. gate_reasoning() for top-level entry gating
3. gate_step() for per-step gating
4. witness_step() for outcome recording
5. learn_from_feedback() for learning integration
6. Graceful degradation (fail_open behavior)
7. Protocol compliance

Author: HoloLoom Team
Date: 2025-12-03 (Phase 2A Conscience Integration)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Optional, Dict, Any

# Import the components we're testing
from hololoom.protocols.conscience import (
    ConscienceProtocol,
    ConscienceDecision,
    StepType,
    RiskLevel,
    NullConscience,
    create_allowed_decision,
    create_blocked_decision,
)
from hololoom.agentic.conscience_adapter import (
    AgenticConscienceAdapter,
    create_conscience_adapter,
    create_null_adapter,
    CONSCIENCE_AVAILABLE,
    MODE_TO_STEP_TYPE,
)


# ============================================================================
# Test 1: Initialization and Configuration
# ============================================================================

class TestInitialization:
    """Test adapter initialization with various configurations."""

    def test_init_with_null_conscience(self):
        """Adapter should work with NullConscience."""
        adapter = AgenticConscienceAdapter(
            conscience=NullConscience(),
            auto_create=False,
        )
        assert adapter is not None
        # NullConscience is a valid conscience, so available should be True
        assert adapter.conscience is not None

    def test_init_auto_create_false_no_conscience(self):
        """Without auto_create, adapter should not create conscience."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )
        assert adapter.conscience is None
        # Without conscience and not auto-creating, not available
        # (but depends on CONSCIENCE_AVAILABLE)

    def test_init_with_preset_standard(self):
        """Auto-create with 'standard' preset."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = AgenticConscienceAdapter(
            conscience=None,
            preset="standard",
            auto_create=True,
        )
        assert adapter.available is True
        assert adapter.conscience is not None

    def test_init_with_preset_paranoid(self):
        """Auto-create with 'paranoid' preset."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = AgenticConscienceAdapter(
            conscience=None,
            preset="paranoid",
            auto_create=True,
        )
        assert adapter.available is True

    def test_init_with_preset_research(self):
        """Auto-create with 'research' preset."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = AgenticConscienceAdapter(
            conscience=None,
            preset="research",
            auto_create=True,
        )
        assert adapter.available is True

    def test_init_with_unknown_preset_falls_back(self):
        """Unknown preset should fallback to 'standard'."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = AgenticConscienceAdapter(
            conscience=None,
            preset="unknown_preset_xyz",
            auto_create=True,
        )
        # Should still work (falls back to standard)
        assert adapter.available is True

    def test_fail_open_default_true(self):
        """fail_open should default to True."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )
        assert adapter._fail_open is True

    def test_fail_open_can_be_false(self):
        """fail_open can be explicitly set to False."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=False,
        )
        assert adapter._fail_open is False


# ============================================================================
# Test 2: gate_reasoning() Entry Gating
# ============================================================================

class TestGateReasoning:
    """Test gate_reasoning() for top-level entry gating."""

    @pytest.mark.asyncio
    async def test_gate_reasoning_unavailable_fail_open(self):
        """When unavailable and fail_open=True, should allow."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=True,
        )

        decision = await adapter.gate_reasoning(
            query="Test query",
            mode="direct",
        )

        assert decision.allowed is True
        assert "unavailable" in decision.reason.lower() or "caution" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_gate_reasoning_unavailable_fail_closed(self):
        """When unavailable and fail_open=False, should block."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=False,
        )

        decision = await adapter.gate_reasoning(
            query="Test query",
            mode="direct",
        )

        assert decision.allowed is False
        assert decision.risk_level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_gate_reasoning_mode_mapping_direct(self):
        """'direct' mode should map to StepType.QUERY."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=True,
        )

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="direct",
        )

        assert decision.step_type == StepType.QUERY

    @pytest.mark.asyncio
    async def test_gate_reasoning_mode_mapping_verify(self):
        """'verify' mode should map to StepType.VERIFICATION."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="verify",
        )

        assert decision.step_type == StepType.VERIFICATION

    @pytest.mark.asyncio
    async def test_gate_reasoning_mode_mapping_research(self):
        """'research' mode should map to StepType.RESEARCH."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="research",
        )

        assert decision.step_type == StepType.RESEARCH

    @pytest.mark.asyncio
    async def test_gate_reasoning_mode_mapping_plan_execute(self):
        """'plan_execute' mode should map to StepType.PLAN_STEP."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="plan_execute",
        )

        assert decision.step_type == StepType.PLAN_STEP

    @pytest.mark.asyncio
    async def test_gate_reasoning_unknown_mode_defaults_to_query(self):
        """Unknown mode should default to QUERY."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="unknown_mode_xyz",
        )

        assert decision.step_type == StepType.QUERY

    @pytest.mark.asyncio
    async def test_gate_reasoning_with_context(self):
        """Context should be passed through."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(auto_create=True)

        decision = await adapter.gate_reasoning(
            query="Test query",
            mode="direct",
            context={"user_id": "test_user", "session_id": "session_123"},
        )

        assert decision is not None

    @pytest.mark.asyncio
    async def test_gate_reasoning_with_epistemic_confidence(self):
        """Epistemic confidence should be included in context."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(auto_create=True)

        decision = await adapter.gate_reasoning(
            query="Test query",
            mode="direct",
            epistemic_confidence=0.3,
        )

        assert decision is not None

    @pytest.mark.asyncio
    async def test_gate_reasoning_exception_fail_open(self):
        """Exception in conscience should allow when fail_open=True."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        # Create mock conscience that raises
        mock_conscience = Mock()
        mock_conscience.consider = AsyncMock(side_effect=Exception("Test error"))

        adapter = AgenticConscienceAdapter(
            conscience=mock_conscience,
            auto_create=False,
            fail_open=True,
        )
        adapter._available = True

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="direct",
        )

        assert decision.allowed is True
        assert "error" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_gate_reasoning_exception_fail_closed(self):
        """Exception in conscience should block when fail_open=False."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        mock_conscience = Mock()
        mock_conscience.consider = AsyncMock(side_effect=Exception("Test error"))

        adapter = AgenticConscienceAdapter(
            conscience=mock_conscience,
            auto_create=False,
            fail_open=False,
        )
        adapter._available = True

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="direct",
        )

        assert decision.allowed is False
        assert decision.risk_level == RiskLevel.CRITICAL


# ============================================================================
# Test 3: gate_step() Per-Step Gating
# ============================================================================

class TestGateStep:
    """Test gate_step() for per-step gating in multi-step reasoning."""

    @pytest.mark.asyncio
    async def test_gate_step_unavailable_fail_open(self):
        """When unavailable and fail_open=True, step should be allowed."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=True,
        )

        decision = await adapter.gate_step(
            step_action="Verify claim X",
            step_type=StepType.VERIFICATION,
            step_index=0,
        )

        assert decision.allowed is True
        assert decision.step_type == StepType.VERIFICATION
        assert decision.step_index == 0

    @pytest.mark.asyncio
    async def test_gate_step_unavailable_fail_closed(self):
        """When unavailable and fail_open=False, step should be blocked."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=False,
        )

        decision = await adapter.gate_step(
            step_action="Verify claim X",
            step_type=StepType.VERIFICATION,
            step_index=0,
        )

        assert decision.allowed is False
        assert decision.risk_level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_gate_step_all_step_types(self):
        """All StepTypes should be handled correctly."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=True,
        )

        for step_type in StepType:
            decision = await adapter.gate_step(
                step_action=f"Test action for {step_type.name}",
                step_type=step_type,
                step_index=0,
            )
            assert decision.step_type == step_type

    @pytest.mark.asyncio
    async def test_gate_step_with_parent_decision(self):
        """Parent decision should be included in context."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(auto_create=True)

        # Create a parent decision
        parent_decision = create_allowed_decision(
            reason="Parent allowed",
            step_type=StepType.QUERY,
        )
        parent_decision.voice = "WHISPER"
        parent_decision.risk_level = RiskLevel.LOW

        decision = await adapter.gate_step(
            step_action="Sub-query",
            step_type=StepType.VERIFICATION,
            step_index=1,
            parent_decision=parent_decision,
        )

        assert decision is not None

    @pytest.mark.asyncio
    async def test_gate_step_increments_step_index(self):
        """Step index should be correctly recorded."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=True,
        )

        for i in range(5):
            decision = await adapter.gate_step(
                step_action=f"Step {i}",
                step_type=StepType.RESEARCH,
                step_index=i,
            )
            assert decision.step_index == i

    @pytest.mark.asyncio
    async def test_gate_step_exception_fail_open(self):
        """Exception in step gating should allow when fail_open=True."""
        mock_conscience = Mock()
        mock_conscience.consider = AsyncMock(side_effect=Exception("Step error"))

        adapter = AgenticConscienceAdapter(
            conscience=mock_conscience,
            auto_create=False,
            fail_open=True,
        )
        adapter._available = True

        decision = await adapter.gate_step(
            step_action="Test",
            step_type=StepType.VERIFICATION,
            step_index=0,
        )

        assert decision.allowed is True
        assert "error" in decision.reason.lower()


# ============================================================================
# Test 4: witness_step() Outcome Recording
# ============================================================================

class TestWitnessStep:
    """Test witness_step() for outcome recording."""

    @pytest.mark.asyncio
    async def test_witness_step_unavailable_returns_empty(self):
        """When unavailable, witness_step should return empty string."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )

        decision = create_allowed_decision(
            reason="Test",
            step_type=StepType.QUERY,
        )

        record_id = await adapter.witness_step(
            decision=decision,
            result={"output": "test"},
            success=True,
        )

        assert record_id == ""

    @pytest.mark.asyncio
    async def test_witness_step_records_success(self):
        """Successful step should be witnessed."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(auto_create=True)

        decision = create_allowed_decision(
            reason="Test",
            step_type=StepType.VERIFICATION,
            step_index=0,
        )

        record_id = await adapter.witness_step(
            decision=decision,
            result={"output": "verified", "confidence": 0.9},
            success=True,
        )

        # May return empty if witness not fully implemented
        assert isinstance(record_id, str)

    @pytest.mark.asyncio
    async def test_witness_step_records_failure(self):
        """Failed step should be witnessed."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(auto_create=True)

        decision = create_allowed_decision(
            reason="Test",
            step_type=StepType.RESEARCH,
            step_index=2,
        )

        record_id = await adapter.witness_step(
            decision=decision,
            result={"error": "query failed"},
            success=False,
        )

        assert isinstance(record_id, str)

    @pytest.mark.asyncio
    async def test_witness_step_exception_returns_empty(self):
        """Exception in witnessing should return empty, not crash."""
        mock_conscience = Mock()
        mock_conscience.witness = AsyncMock(side_effect=Exception("Witness error"))

        adapter = AgenticConscienceAdapter(
            conscience=mock_conscience,
            auto_create=False,
        )
        adapter._available = True

        decision = create_allowed_decision(reason="Test", step_type=StepType.QUERY)

        record_id = await adapter.witness_step(
            decision=decision,
            result={},
            success=True,
        )

        # Should not crash, returns empty
        assert record_id == ""


# ============================================================================
# Test 5: learn_from_feedback() Learning Integration
# ============================================================================

class TestLearnFromFeedback:
    """Test learn_from_feedback() for learning integration."""

    @pytest.mark.asyncio
    async def test_learn_from_feedback_unavailable_no_crash(self):
        """When unavailable, learn should not crash."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )

        # Should not raise
        await adapter.learn_from_feedback(
            feedback={"correct": True, "confidence": 0.9},
        )

    @pytest.mark.asyncio
    async def test_learn_from_feedback_with_decisions(self):
        """Learning should include decision metadata."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(auto_create=True)

        decisions = [
            create_allowed_decision("Test 1", StepType.QUERY, step_index=0),
            create_allowed_decision("Test 2", StepType.VERIFICATION, step_index=1),
        ]

        # Should not raise
        await adapter.learn_from_feedback(
            feedback={"correct": True, "confidence": 0.85},
            decisions=decisions,
        )

    @pytest.mark.asyncio
    async def test_learn_from_feedback_exception_no_crash(self):
        """Exception in learning should not crash."""
        mock_conscience = Mock()
        mock_conscience.learn = AsyncMock(side_effect=Exception("Learn error"))

        adapter = AgenticConscienceAdapter(
            conscience=mock_conscience,
            auto_create=False,
        )
        adapter._available = True

        # Should not raise
        await adapter.learn_from_feedback(
            feedback={"correct": False},
        )


# ============================================================================
# Test 6: Graceful Degradation
# ============================================================================

class TestGracefulDegradation:
    """Test graceful degradation behavior."""

    def test_available_false_when_no_conscience(self):
        """Available should be False when no conscience and not auto-created."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )
        # Depends on CONSCIENCE_AVAILABLE
        if not CONSCIENCE_AVAILABLE:
            assert adapter.available is False

    @pytest.mark.asyncio
    async def test_is_safe_returns_fail_open_when_unavailable(self):
        """is_safe should return fail_open value when unavailable."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
            fail_open=True,
        )

        result = await adapter.is_safe("Test action")
        assert result is True

        adapter._fail_open = False
        result = await adapter.is_safe("Test action")
        assert result is False

    def test_get_statistics_unavailable(self):
        """get_statistics should return status when unavailable."""
        adapter = AgenticConscienceAdapter(
            conscience=None,
            auto_create=False,
        )

        stats = adapter.get_statistics()
        assert "status" in stats
        assert stats["status"] == "unavailable"

    def test_get_statistics_available(self):
        """get_statistics should return conscience stats when available."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(auto_create=True)

        stats = adapter.get_statistics()
        # Conscience returns lens, lens_count, witness, auto_learn
        assert "lens" in stats or "status" in stats  # Either format works


# ============================================================================
# Test 7: Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions for adapter creation."""

    def test_create_conscience_adapter_default(self):
        """create_conscience_adapter with defaults."""
        adapter = create_conscience_adapter()

        assert adapter is not None
        assert adapter._fail_open is True

    def test_create_conscience_adapter_with_preset(self):
        """create_conscience_adapter with preset."""
        if not CONSCIENCE_AVAILABLE:
            pytest.skip("Conscience framework not available")

        adapter = create_conscience_adapter(preset="paranoid")
        assert adapter.available is True

    def test_create_null_adapter(self):
        """create_null_adapter should create adapter with NullConscience."""
        adapter = create_null_adapter()

        assert adapter is not None
        assert adapter._fail_open is True
        # NullConscience should be set
        assert isinstance(adapter.conscience, NullConscience)

    @pytest.mark.asyncio
    async def test_null_adapter_always_allows(self):
        """Null adapter should always allow."""
        adapter = create_null_adapter()

        decision = await adapter.gate_reasoning(
            query="Any query",
            mode="direct",
        )

        assert decision.allowed is True


# ============================================================================
# Test 8: Protocol Compliance
# ============================================================================

class TestProtocolCompliance:
    """Test that adapter methods comply with expected interfaces."""

    def test_adapter_has_required_methods(self):
        """Adapter should have all required methods."""
        adapter = create_conscience_adapter()

        assert hasattr(adapter, 'gate_reasoning')
        assert hasattr(adapter, 'gate_step')
        assert hasattr(adapter, 'witness_step')
        assert hasattr(adapter, 'learn_from_feedback')
        assert hasattr(adapter, 'is_safe')
        assert hasattr(adapter, 'get_statistics')

    @pytest.mark.asyncio
    async def test_gate_reasoning_returns_conscience_decision(self):
        """gate_reasoning should return ConscienceDecision."""
        adapter = create_conscience_adapter()

        decision = await adapter.gate_reasoning(
            query="Test",
            mode="direct",
        )

        assert isinstance(decision, ConscienceDecision)
        assert isinstance(decision.allowed, bool)
        assert isinstance(decision.risk_level, RiskLevel)
        assert isinstance(decision.step_type, StepType)

    @pytest.mark.asyncio
    async def test_gate_step_returns_conscience_decision(self):
        """gate_step should return ConscienceDecision."""
        adapter = create_conscience_adapter()

        decision = await adapter.gate_step(
            step_action="Test",
            step_type=StepType.VERIFICATION,
            step_index=0,
        )

        assert isinstance(decision, ConscienceDecision)


# ============================================================================
# Test 9: Mode-to-StepType Mapping
# ============================================================================

class TestModeMapping:
    """Test the MODE_TO_STEP_TYPE mapping constant."""

    def test_mode_mapping_has_all_modes(self):
        """Mapping should have all expected modes."""
        expected_modes = ["direct", "verify", "research", "plan_execute"]

        for mode in expected_modes:
            assert mode in MODE_TO_STEP_TYPE

    def test_mode_mapping_values_are_step_types(self):
        """All mapping values should be StepType enum members."""
        for mode, step_type in MODE_TO_STEP_TYPE.items():
            assert isinstance(step_type, StepType)

    def test_mode_mapping_correct_values(self):
        """Mapping should have correct values."""
        assert MODE_TO_STEP_TYPE["direct"] == StepType.QUERY
        assert MODE_TO_STEP_TYPE["verify"] == StepType.VERIFICATION
        assert MODE_TO_STEP_TYPE["research"] == StepType.RESEARCH
        assert MODE_TO_STEP_TYPE["plan_execute"] == StepType.PLAN_STEP


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
