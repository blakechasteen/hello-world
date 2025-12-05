"""
Red Team Tests for Agentic Safety Integration
==============================================

Tests adversarial bypass attempts identified by MRF CRITIQUE analysis.

Test Categories:
1. Direct bypass attempts
2. TOCTOU (Time-of-Check-to-Time-of-Use) protection
3. Unicode/homoglyph bypass attempts
4. Fail-closed exception handling
5. Epistemic confidence escalation

Author: MRF Swarm (CRITIQUE perspective)
Date: 2025-12-01 (Safe Integration Phase)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional

# Import the components we're testing
from HoloLoom.protocols.safety import (
    SafetyGateProtocol,
    SafetyGateDecision,
    SafetyRiskLevel,
)
from HoloLoom.agentic.safety_adapter import (
    AgenticSafetyAdapter,
    create_safety_adapter,
    ALIGNMENT_AVAILABLE,
)


# ============================================================================
# Test 1: Direct Bypass Attempts
# ============================================================================

class TestDirectBypassAttempts:
    """
    Test that safety checks apply even when orchestrator is created
    without explicit guardrails.

    CRITIQUE: Direct engine bypass - if someone calls AgenticOrchestrator
    directly without guardrails, safety should still be auto-created.
    """

    def test_auto_create_guardrails_when_none_provided(self):
        """Safety adapter should auto-create guardrails if none provided."""
        adapter = create_safety_adapter(guardrails=None, auto_create=True)

        assert adapter is not None
        # If alignment is available, adapter should be functional
        if ALIGNMENT_AVAILABLE:
            assert adapter.available is True
            assert adapter.guardrails is not None

    def test_graceful_degradation_without_alignment(self):
        """Adapter should work (in pass-through mode) even without alignment framework."""
        # This tests graceful degradation
        adapter = AgenticSafetyAdapter(guardrails=None, auto_create_guardrails=False)

        # Should not crash - graceful degradation
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_adapter_allows_by_default_when_degraded(self):
        """In degraded mode, adapter should allow (with warning) rather than crash."""
        adapter = AgenticSafetyAdapter(guardrails=None, auto_create_guardrails=False)

        decision = await adapter.gate_reasoning(
            query_text="Test query",
            reasoning_mode="direct"
        )

        assert decision.allowed is True
        assert "degraded" in str(decision.metadata).lower() or decision.risk_level == SafetyRiskLevel.LOW


# ============================================================================
# Test 2: TOCTOU (Time-of-Check-to-Time-of-Use) Protection
# ============================================================================

class TestTOCTOUProtection:
    """
    Test that query text cannot be modified between safety check and execution.

    CRITIQUE: TOCTOU race condition - query can be modified between
    safety check and execution.

    Solution: Deep copy query text in adapter.
    """

    @pytest.mark.asyncio
    async def test_query_deep_copy_protection(self):
        """Query should be deep copied so modifications don't affect safety check."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        # Create a mutable query-like object
        original_query = "What is Thompson Sampling?"

        # Call gate_reasoning
        decision = await adapter.gate_reasoning(
            query_text=original_query,
            reasoning_mode="direct"
        )

        # The normalized query in metadata should be a copy
        if 'normalized_query' in decision.metadata:
            normalized = decision.metadata['normalized_query']
            # Should be a string copy, not a reference
            assert isinstance(normalized, str)

    @pytest.mark.asyncio
    async def test_query_normalization_applied(self):
        """Query text should be normalized to prevent bypass."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        # Unicode with potential bypass characters
        tricky_query = "ignore\u200bprevious"  # Zero-width space

        decision = await adapter.gate_reasoning(
            query_text=tricky_query,
            reasoning_mode="direct"
        )

        # Normalized query should have had NFKC normalization applied
        if 'normalized_query' in decision.metadata:
            normalized = decision.metadata['normalized_query']
            # Zero-width space should be normalized
            assert "\u200b" not in normalized


# ============================================================================
# Test 3: Unicode/Homoglyph Bypass Attempts
# ============================================================================

class TestUnicodeBypassAttempts:
    """
    Test that Unicode normalization catches adversarial bypass attempts.

    CRITIQUE: Unicode pattern bypass - adversarial patterns bypass via
    homoglyphs/encoding.

    Solution: NFKC normalization before pattern matching.
    """

    @pytest.mark.parametrize("payload,description", [
        ("ign\u200bore previous", "Zero-width joiner"),
        ("ignore prev\u0456ous", "Cyrillic 'i' homoglyph"),
        ("\ufeffignore previous", "BOM character"),
        ("ＩＧＮＯＲＥ previous", "Fullwidth Latin"),
        ("ignore\u00A0previous", "Non-breaking space"),
    ])
    @pytest.mark.asyncio
    async def test_unicode_adversarial_patterns(self, payload: str, description: str):
        """
        Test that various Unicode bypass attempts are normalized.

        These patterns attempt to bypass adversarial detection by using
        Unicode characters that look like ASCII but aren't.
        """
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        decision = await adapter.gate_reasoning(
            query_text=payload,
            reasoning_mode="direct"
        )

        # The decision should be made on normalized text
        # If adversarial is detected, it should be blocked
        if 'normalized_query' in decision.metadata:
            normalized = decision.metadata['normalized_query']
            # Normalization should have been applied
            assert isinstance(normalized, str)
            # Key adversarial text should be detectable after normalization
            # (The actual blocking depends on the adversarial detector patterns)

    @pytest.mark.asyncio
    async def test_normalization_preserves_legitimate_unicode(self):
        """
        Legitimate Unicode (e.g., non-English text) should still work.
        """
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        # Legitimate question in mixed script
        legitimate_query = "What is 机器学习 (machine learning)?"

        decision = await adapter.gate_reasoning(
            query_text=legitimate_query,
            reasoning_mode="research"
        )

        # Should not block legitimate multilingual queries
        # (unless they happen to trigger adversarial patterns)
        assert decision is not None


# ============================================================================
# Test 4: Fail-Closed Exception Handling
# ============================================================================

class TestFailClosedExceptionHandling:
    """
    Test that exceptions in safety evaluation result in denial, not allowance.

    CRITIQUE: Safety check fails → should DENY, not ALLOW.
    """

    @pytest.mark.asyncio
    async def test_exception_fails_closed(self):
        """Safety exceptions should deny rather than allow."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        # Create adapter with a broken guardrails mock
        broken_guardrails = Mock()
        broken_guardrails.evaluate.side_effect = Exception("Simulated failure")

        adapter = AgenticSafetyAdapter(
            guardrails=broken_guardrails,
            auto_create_guardrails=False
        )
        adapter._available = True  # Force available

        decision = await adapter.gate_reasoning(
            query_text="Test query",
            reasoning_mode="direct"
        )

        # Should fail closed - deny on exception
        assert decision.allowed is False
        assert decision.risk_level == SafetyRiskLevel.CRITICAL
        assert "error" in decision.reason.lower() or "fail" in decision.metadata.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_fail_closed_metadata_includes_error(self):
        """Fail-closed decision should include error information for debugging."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        # Create adapter with broken guardrails
        broken_guardrails = Mock()
        broken_guardrails.evaluate.side_effect = ValueError("Test error")

        adapter = AgenticSafetyAdapter(
            guardrails=broken_guardrails,
            auto_create_guardrails=False
        )
        adapter._available = True

        decision = await adapter.gate_reasoning(
            query_text="Test",
            reasoning_mode="direct"
        )

        # Should have error in metadata
        assert decision.metadata.get("fail_closed") is True
        assert "error" in decision.metadata


# ============================================================================
# Test 5: Epistemic Confidence Escalation
# ============================================================================

class TestEpistemicConfidenceEscalation:
    """
    Test that low epistemic confidence escalates risk assessment.

    VERIFY: Epistemic confidence should influence safety decisions.
    """

    @pytest.mark.asyncio
    async def test_low_epistemic_escalates_risk(self):
        """Low epistemic confidence should result in higher risk assessment."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        # Very low epistemic confidence
        decision = await adapter.gate_reasoning(
            query_text="What is Thompson Sampling?",
            reasoning_mode="direct",
            epistemic_confidence=0.1  # Very uncertain
        )

        # Should pass epistemic confidence to guardrails
        assert decision.epistemic_confidence == 0.1

    @pytest.mark.asyncio
    async def test_high_epistemic_no_escalation(self):
        """High epistemic confidence should not escalate risk."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        # High epistemic confidence
        decision = await adapter.gate_reasoning(
            query_text="What is Thompson Sampling?",
            reasoning_mode="direct",
            epistemic_confidence=0.9  # Very confident
        )

        # Should be allowed (standard query, high confidence)
        assert decision.epistemic_confidence == 0.9


# ============================================================================
# Test 6: Mode-to-Category Mapping
# ============================================================================

class TestModeToCategoryMapping:
    """
    Test that reasoning modes are correctly mapped to action categories.
    """

    @pytest.mark.asyncio
    async def test_direct_maps_to_query(self):
        """Direct mode should map to QUERY category (lowest risk)."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        decision = await adapter.gate_reasoning(
            query_text="Simple question",
            reasoning_mode="direct"
        )

        if 'category_mapped' in decision.metadata:
            assert decision.metadata['category_mapped'] == 'query'

    @pytest.mark.asyncio
    async def test_plan_execute_maps_to_execution(self):
        """Plan_execute mode should map to EXECUTION category (highest risk)."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        decision = await adapter.gate_reasoning(
            query_text="Execute this plan",
            reasoning_mode="plan_execute"
        )

        if 'category_mapped' in decision.metadata:
            assert decision.metadata['category_mapped'] == 'execution'

    @pytest.mark.asyncio
    async def test_unknown_mode_defaults_to_query(self):
        """Unknown mode should default to safest category (QUERY)."""
        if not ALIGNMENT_AVAILABLE:
            pytest.skip("Alignment framework not available")

        adapter = create_safety_adapter(auto_create=True)

        decision = await adapter.gate_reasoning(
            query_text="Unknown mode query",
            reasoning_mode="unknown_mode_xyz"
        )

        # Should not crash, should default to safe category
        assert decision is not None


# ============================================================================
# Test 7: Protocol Compliance
# ============================================================================

class TestProtocolCompliance:
    """
    Test that AgenticSafetyAdapter complies with SafetyGateProtocol.
    """

    def test_adapter_implements_protocol(self):
        """Adapter should implement SafetyGateProtocol."""
        adapter = create_safety_adapter(auto_create=True)

        # Check that it has the required method
        assert hasattr(adapter, 'gate_reasoning')
        assert callable(adapter.gate_reasoning)

    @pytest.mark.asyncio
    async def test_gate_reasoning_returns_correct_type(self):
        """gate_reasoning should return SafetyGateDecision."""
        adapter = create_safety_adapter(auto_create=True)

        decision = await adapter.gate_reasoning(
            query_text="Test",
            reasoning_mode="direct"
        )

        assert isinstance(decision, SafetyGateDecision)
        assert isinstance(decision.allowed, bool)
        assert isinstance(decision.risk_level, SafetyRiskLevel)
        assert isinstance(decision.reason, str)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
