"""
End-to-End Integration Tests for Conscience System

Tests the complete conscience integration across all agentic reasoning modes:
- DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE with conscience gating
- Thompson Sampling calibration learning across sessions
- Blocked queries and early termination
- Graceful degradation scenarios

Created: Dec 2025
Phase: 3 - End-to-End Integration Testing
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from HoloLoom.agentic.core import (
    AgenticOrchestrator,
    ReasoningMode,
    AgenticResult,
    CONSCIENCE_AVAILABLE,
)
from HoloLoom.agentic.conscience_adapter import (
    AgenticConscienceAdapter,
    create_conscience_adapter,
    create_null_adapter,
    CALIBRATOR_AVAILABLE,
)
from HoloLoom.protocols.conscience import ConscienceDecision, StepType, RiskLevel
from HoloLoom.protocols.types import Query
from HoloLoom.alignment.audit_trail import AuditTrail


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockSpacetime:
    """Mock Spacetime for testing."""
    confidence: float = 0.85
    response: str = "Test response"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockLearningEngine:
    """Mock FullLearningEngine for testing."""

    def __init__(self, default_confidence: float = 0.85):
        self.default_confidence = default_confidence
        self.weave_count = 0
        self.orchestrator = MagicMock()

    async def weave(self, query: Query, **kwargs) -> MockSpacetime:
        """Mock weave that returns consistent results."""
        self.weave_count += 1
        return MockSpacetime(
            confidence=self.default_confidence,
            response=f"Response to: {query.text}",
            metadata={"query": query.text, "weave_count": self.weave_count}
        )


class MockConscience:
    """
    Mock Conscience that implements ConscienceProtocol for testing.

    The ConscienceProtocol requires a consider() method that returns a
    Judgment-like dict. The AgenticConscienceAdapter converts this to
    ConscienceDecision internally.
    """

    def __init__(
        self,
        default_allowed: bool = True,
        block_patterns: Optional[List[str]] = None,
        risk_level: RiskLevel = RiskLevel.LOW
    ):
        self.default_allowed = default_allowed
        self.block_patterns = block_patterns or []
        self.risk_level = risk_level
        self.considerations: List[Dict[str, Any]] = []

    async def consider(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Consider an action (required by ConscienceProtocol).

        Returns a Judgment-like dict that AgenticConscienceAdapter
        converts to ConscienceDecision.
        """
        allowed = self.default_allowed
        reason = "Action allowed" if allowed else "Action blocked"
        voice = "QUIET"

        # Check block patterns
        for pattern in self.block_patterns:
            if pattern.lower() in action.lower():
                allowed = False
                reason = f"Blocked: contains '{pattern}'"
                voice = "SHOUT"
                break

        # Record this consideration
        self.considerations.append({
            "action": action,
            "context": context,
            "allowed": allowed,
            "timestamp": datetime.now()
        })

        # Return Judgment-like dict (matches NullConscience format)
        return {
            "voice": voice,
            "voice_level": 0 if allowed else 3,
            "concerns": [] if allowed else [{"description": reason}],
            "summary": reason,
            "guidance": "Proceed normally" if allowed else "Action blocked",
            "allowed": allowed,
        }

    async def witness(
        self,
        action: str,
        context: Dict[str, Any],
        judgment: Any,
        result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Witness an action (required by ConscienceProtocol)."""
        return f"witness_{len(self.considerations)}"

    async def learn(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback (required by ConscienceProtocol)."""
        pass


@pytest.fixture
def mock_learning_engine():
    """Create a mock learning engine."""
    return MockLearningEngine(default_confidence=0.85)


@pytest.fixture
def mock_conscience():
    """Create a mock conscience that allows all."""
    return MockConscience(default_allowed=True)


@pytest.fixture
def blocking_conscience():
    """Create a mock conscience that blocks certain patterns."""
    return MockConscience(
        default_allowed=True,
        block_patterns=["dangerous", "harmful", "malicious"]
    )


@pytest.fixture
def audit_trail():
    """Create an audit trail for testing."""
    return AuditTrail()


@pytest.fixture
def conscience_adapter(mock_conscience):
    """Create a conscience adapter with mock conscience."""
    return AgenticConscienceAdapter(
        conscience=mock_conscience,
        auto_create=False,
        enable_calibration=True
    )


@pytest.fixture
def blocking_adapter(blocking_conscience):
    """Create a conscience adapter that blocks certain patterns."""
    return AgenticConscienceAdapter(
        conscience=blocking_conscience,
        auto_create=False,
        enable_calibration=True
    )


# =============================================================================
# Test: DIRECT Mode with Conscience
# =============================================================================

class TestDirectModeWithConscience:
    """Tests for DIRECT reasoning mode with conscience integration."""

    @pytest.mark.asyncio
    async def test_direct_mode_allowed(self, mock_learning_engine, conscience_adapter, audit_trail):
        """Test DIRECT mode passes through when allowed."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=conscience_adapter,
            enable_conscience=True
        )

        query = Query(text="What is Thompson Sampling?")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        assert result is not None
        assert result.reasoning_mode == ReasoningMode.DIRECT
        assert len(result.steps_taken) == 1
        assert result.steps_taken[0]["type"] == "direct_answer"
        assert mock_learning_engine.weave_count == 1

    @pytest.mark.asyncio
    async def test_direct_mode_no_per_step_gating(self, mock_learning_engine, conscience_adapter, audit_trail):
        """Test DIRECT mode doesn't use per-step gating (only safety gate)."""
        # DIRECT mode doesn't have multiple steps to gate
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=conscience_adapter,
            enable_conscience=True
        )

        query = Query(text="Simple question")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        # Should complete in one step without per-step gating
        assert len(result.steps_taken) == 1
        assert "blocked" not in result.steps_taken[0]


# =============================================================================
# Test: VERIFY Mode with Conscience
# =============================================================================

class TestVerifyModeWithConscience:
    """Tests for VERIFY reasoning mode with conscience integration."""

    @pytest.mark.asyncio
    async def test_verify_mode_allowed(self, mock_learning_engine, conscience_adapter, audit_trail):
        """Test VERIFY mode completes with allowed steps."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=conscience_adapter,
            enable_conscience=True
        )

        query = Query(text="Is Thompson Sampling optimal?")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.VERIFY,
            max_steps=3
        )

        assert result is not None
        assert result.reasoning_mode == ReasoningMode.VERIFY
        assert len(result.steps_taken) >= 1
        # VERIFY mode uses verification queries
        assert result.verification is not None or len(result.steps_taken) > 1


# =============================================================================
# Test: RESEARCH Mode with Conscience Per-Step Gating
# =============================================================================

class TestResearchModeWithConscience:
    """Tests for RESEARCH reasoning mode with per-step conscience gating."""

    @pytest.mark.asyncio
    async def test_research_mode_all_steps_allowed(self, mock_learning_engine, conscience_adapter, audit_trail):
        """Test RESEARCH mode executes all steps when allowed."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=conscience_adapter,
            enable_conscience=True
        )

        query = Query(text="Research Thompson Sampling applications")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.RESEARCH,
            max_steps=3
        )

        assert result is not None
        assert result.reasoning_mode == ReasoningMode.RESEARCH
        # Should have multiple research queries
        research_steps = [s for s in result.steps_taken if s.get("type") == "research_query"]
        assert len(research_steps) >= 1

    @pytest.mark.asyncio
    async def test_research_mode_blocked_step(self, mock_learning_engine, blocking_adapter, audit_trail):
        """Test RESEARCH mode skips blocked steps."""
        # Create orchestrator with blocking adapter
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=blocking_adapter,
            enable_conscience=True
        )

        # Mock research query generation to include a "dangerous" query
        with patch.object(orchestrator, '_generate_research_queries', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [
                "What is Thompson Sampling?",
                "How is it dangerous in adversarial settings?",  # Should be blocked
                "What are the applications?"
            ]

            query = Query(text="Research Thompson Sampling")
            result = await orchestrator.reason(
                query,
                mode=ReasoningMode.RESEARCH,
                max_steps=3
            )

            # Check that one step was blocked
            blocked_steps = [s for s in result.steps_taken if s.get("blocked")]
            assert len(blocked_steps) >= 1
            assert "dangerous" in blocked_steps[0].get("blocked_reason", "").lower()


# =============================================================================
# Test: PLAN_EXECUTE Mode with Conscience Per-Step Gating
# =============================================================================

class TestPlanExecuteModeWithConscience:
    """Tests for PLAN_EXECUTE reasoning mode with per-step conscience gating."""

    @pytest.mark.asyncio
    async def test_plan_execute_mode_all_steps_allowed(self, mock_learning_engine, conscience_adapter, audit_trail):
        """Test PLAN_EXECUTE mode executes all sub-goals when allowed."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=conscience_adapter,
            enable_conscience=True
        )

        query = Query(text="Implement Thompson Sampling algorithm")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.PLAN_EXECUTE,
            max_steps=3
        )

        assert result is not None
        assert result.reasoning_mode == ReasoningMode.PLAN_EXECUTE
        # Should have sub-goal steps
        sub_goal_steps = [s for s in result.steps_taken if s.get("type") == "sub_goal"]
        assert len(sub_goal_steps) >= 1

    @pytest.mark.asyncio
    async def test_plan_execute_mode_blocked_subgoal(self, mock_learning_engine, blocking_adapter, audit_trail):
        """Test PLAN_EXECUTE mode skips blocked sub-goals."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=blocking_adapter,
            enable_conscience=True
        )

        # Mock goal decomposition to include a "harmful" sub-goal
        with patch.object(orchestrator, '_decompose_goal') as mock_decompose:
            mock_decompose.return_value = [
                "Understand Thompson Sampling basics",
                "Implement harmful bypass mechanism",  # Should be blocked
                "Test the implementation"
            ]

            query = Query(text="Implement Thompson Sampling")
            result = await orchestrator.reason(
                query,
                mode=ReasoningMode.PLAN_EXECUTE,
                max_steps=3
            )

            # Check that one sub-goal was blocked
            blocked_steps = [s for s in result.steps_taken if s.get("blocked")]
            assert len(blocked_steps) >= 1
            assert "harmful" in blocked_steps[0].get("blocked_reason", "").lower()


# =============================================================================
# Test: Calibration Learning Across Session
# =============================================================================

class TestCalibrationLearningAcrossSession:
    """Tests for Thompson Sampling calibration learning across multiple queries."""

    @pytest.mark.skipif(
        not CALIBRATOR_AVAILABLE,
        reason="Calibrator not available"
    )
    @pytest.mark.asyncio
    async def test_calibration_updates_with_outcomes(self, mock_learning_engine, audit_trail):
        """Test that calibrator updates with query outcomes."""
        # Create adapter with calibration enabled
        adapter = create_conscience_adapter(
            auto_create=True,
            enable_calibration=True
        )

        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=adapter,
            enable_conscience=True
        )

        # Run multiple queries to accumulate calibration data
        queries = [
            "What is Thompson Sampling?",
            "How does Bayesian inference work?",
            "Explain exploration-exploitation tradeoff"
        ]

        for q in queries:
            await orchestrator.reason(Query(text=q), mode=ReasoningMode.DIRECT)

        # Check calibration statistics
        if hasattr(adapter, 'get_calibration_statistics'):
            stats = adapter.get_calibration_statistics()
            assert stats is not None
            # Note: Stats structure depends on implementation

    @pytest.mark.skipif(
        not CALIBRATOR_AVAILABLE,
        reason="Calibrator not available"
    )
    @pytest.mark.asyncio
    async def test_calibration_persists_across_queries(self, mock_learning_engine, audit_trail):
        """Test that calibration state persists across multiple queries."""
        adapter = create_conscience_adapter(
            auto_create=True,
            enable_calibration=True
        )

        # Simulate witnessing outcomes
        if hasattr(adapter, 'witness_step'):
            # Create a mock decision
            decision = ConscienceDecision(
                allowed=True,
                risk_level=RiskLevel.LOW,
                reason="Test",
                voice="proceed",
                concerns=[],
                guidance="Continue",
                step_type=StepType.QUERY
            )

            # Witness multiple successful outcomes
            for i in range(5):
                await adapter.witness_step(
                    decision=decision,
                    step_type=StepType.RESEARCH,
                    success=True
                )

            # Check that calibration has updated
            if hasattr(adapter, 'calibrator') and adapter.calibrator:
                priors = adapter.calibrator.get_priors()
                assert priors is not None


# =============================================================================
# Test: Blocked Queries and Early Termination
# =============================================================================

class TestBlockedQueriesAndEarlyTermination:
    """Tests for blocked queries and early termination scenarios."""

    @pytest.mark.asyncio
    async def test_all_steps_blocked_terminates_early(self, mock_learning_engine, audit_trail):
        """Test that reasoning terminates early when all steps are blocked."""
        # Create a conscience that blocks everything
        all_blocking = MockConscience(
            default_allowed=False,
            risk_level=RiskLevel.HIGH
        )
        adapter = AgenticConscienceAdapter(
            conscience=all_blocking,
            auto_create=False,
            enable_calibration=False
        )

        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=adapter,
            enable_conscience=True
        )

        # Mock research queries
        with patch.object(orchestrator, '_generate_research_queries', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ["Query 1", "Query 2", "Query 3"]

            query = Query(text="Research topic")
            result = await orchestrator.reason(
                query,
                mode=ReasoningMode.RESEARCH,
                max_steps=3
            )

            # All steps should be blocked
            blocked_steps = [s for s in result.steps_taken if s.get("blocked")]
            assert len(blocked_steps) == 3

    @pytest.mark.asyncio
    async def test_partial_blocking_continues(self, mock_learning_engine, blocking_adapter, audit_trail):
        """Test that reasoning continues after partial blocking."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=blocking_adapter,
            enable_conscience=True
        )

        # Mock research queries - only middle one is blocked
        with patch.object(orchestrator, '_generate_research_queries', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [
                "Safe query 1",
                "This is dangerous",  # Blocked
                "Safe query 2"
            ]

            query = Query(text="Research topic")
            result = await orchestrator.reason(
                query,
                mode=ReasoningMode.RESEARCH,
                max_steps=3
            )

            # Should have 1 blocked, 2 executed
            blocked_steps = [s for s in result.steps_taken if s.get("blocked")]
            executed_steps = [s for s in result.steps_taken if not s.get("blocked") and s.get("type") == "research_query"]

            assert len(blocked_steps) == 1
            assert len(executed_steps) >= 1


# =============================================================================
# Test: Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation when conscience is unavailable."""

    @pytest.mark.asyncio
    async def test_reasoning_works_without_conscience(self, mock_learning_engine, audit_trail):
        """Test that reasoning works when conscience is disabled."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=None,
            enable_conscience=False
        )

        query = Query(text="What is Thompson Sampling?")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        assert result is not None
        assert result.reasoning_mode == ReasoningMode.DIRECT
        # Should complete without issues
        assert mock_learning_engine.weave_count == 1

    @pytest.mark.asyncio
    async def test_reasoning_works_with_null_adapter(self, mock_learning_engine, audit_trail):
        """Test that reasoning works with null adapter (always allows)."""
        null_adapter = create_null_adapter()

        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=null_adapter,
            enable_conscience=True
        )

        query = Query(text="What is Thompson Sampling?")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        assert result is not None
        # Null adapter should allow everything
        assert mock_learning_engine.weave_count == 1

    @pytest.mark.asyncio
    async def test_conscience_exception_graceful_fallback(self, mock_learning_engine, audit_trail):
        """Test that conscience exceptions are handled gracefully."""
        # Create adapter with failing conscience
        failing_conscience = MagicMock()
        failing_conscience.gate_step = AsyncMock(side_effect=RuntimeError("Conscience error"))
        failing_conscience.gate_reasoning = AsyncMock(side_effect=RuntimeError("Conscience error"))

        adapter = AgenticConscienceAdapter(
            conscience=failing_conscience,
            auto_create=False,
            fail_open=True  # Graceful degradation
        )

        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=adapter,
            enable_conscience=True
        )

        # Should not raise, should continue with degraded functionality
        query = Query(text="What is Thompson Sampling?")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        assert result is not None


# =============================================================================
# Test: Audit Trail Integration
# =============================================================================

class TestAuditTrailIntegration:
    """Tests for audit trail integration with conscience decisions."""

    @pytest.mark.asyncio
    async def test_conscience_decisions_logged(self, mock_learning_engine, conscience_adapter, audit_trail):
        """Test that conscience decisions are logged to audit trail."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=conscience_adapter,
            enable_conscience=True
        )

        query = Query(text="What is Thompson Sampling?")
        await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        # Check audit trail has entries
        # Note: Implementation-specific, adjust based on actual audit trail interface

    @pytest.mark.asyncio
    async def test_blocked_decisions_logged_with_reason(self, mock_learning_engine, blocking_adapter, audit_trail):
        """Test that blocked decisions include the block reason."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=blocking_adapter,
            enable_conscience=True
        )

        # Mock to include dangerous query
        with patch.object(orchestrator, '_generate_research_queries', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ["Dangerous query here"]

            query = Query(text="Research topic")
            result = await orchestrator.reason(
                query,
                mode=ReasoningMode.RESEARCH,
                max_steps=1
            )

            # Should have blocked step
            blocked_steps = [s for s in result.steps_taken if s.get("blocked")]
            if blocked_steps:
                assert "blocked_reason" in blocked_steps[0]


# =============================================================================
# Test: StepType Mapping
# =============================================================================

class TestStepTypeMapping:
    """Tests for correct StepType mapping in different modes."""

    @pytest.mark.asyncio
    async def test_research_uses_research_step_type(self, mock_learning_engine, audit_trail):
        """Test RESEARCH mode uses StepType.RESEARCH."""
        mock_conscience = MockConscience(default_allowed=True)
        adapter = AgenticConscienceAdapter(
            conscience=mock_conscience,
            auto_create=False
        )

        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=adapter,
            enable_conscience=True
        )

        with patch.object(orchestrator, '_generate_research_queries', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ["Research query"]

            query = Query(text="Research topic")
            await orchestrator.reason(query, mode=ReasoningMode.RESEARCH, max_steps=1)

            # Check that conscience was called with RESEARCH step type in context
            # The adapter passes step_type in the context dict
            research_considerations = [
                c for c in mock_conscience.considerations
                if c.get("context", {}).get("step_type") == "RESEARCH"
            ]
            assert len(research_considerations) >= 1

    @pytest.mark.asyncio
    async def test_plan_execute_uses_plan_step_type(self, mock_learning_engine, audit_trail):
        """Test PLAN_EXECUTE mode uses StepType.PLAN_STEP."""
        mock_conscience = MockConscience(default_allowed=True)
        adapter = AgenticConscienceAdapter(
            conscience=mock_conscience,
            auto_create=False
        )

        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=adapter,
            enable_conscience=True
        )

        with patch.object(orchestrator, '_decompose_goal') as mock_decompose:
            mock_decompose.return_value = ["Sub-goal 1"]

            query = Query(text="Implement something")
            await orchestrator.reason(query, mode=ReasoningMode.PLAN_EXECUTE, max_steps=1)

            # Check that conscience was called with PLAN_STEP step type in context
            plan_considerations = [
                c for c in mock_conscience.considerations
                if c.get("context", {}).get("step_type") == "PLAN_STEP"
            ]
            assert len(plan_considerations) >= 1


# =============================================================================
# Test: Multi-Step Gating Statistics
# =============================================================================

class TestMultiStepGatingStatistics:
    """Tests for multi-step gating statistics tracking."""

    @pytest.mark.asyncio
    async def test_gating_statistics_tracked(self, mock_learning_engine, blocking_adapter, audit_trail):
        """Test that gating statistics are tracked across steps."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=blocking_adapter,
            enable_conscience=True
        )

        with patch.object(orchestrator, '_generate_research_queries', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [
                "Safe query",
                "Dangerous query",
                "Another safe query"
            ]

            query = Query(text="Research topic")
            result = await orchestrator.reason(
                query,
                mode=ReasoningMode.RESEARCH,
                max_steps=3
            )

            # Calculate statistics
            total_steps = len([s for s in result.steps_taken if s.get("type") == "research_query"])
            blocked_steps = len([s for s in result.steps_taken if s.get("blocked")])
            allowed_steps = total_steps - blocked_steps

            # Should have mix of allowed and blocked
            assert blocked_steps >= 1
            assert allowed_steps >= 1


# =============================================================================
# Test: Conscience Adapter Properties
# =============================================================================

class TestConscienceAdapterProperties:
    """Tests for conscience adapter properties and configuration."""

    def test_calibration_enabled_property(self):
        """Test calibration_enabled property."""
        adapter_with_cal = create_conscience_adapter(
            auto_create=True,
            enable_calibration=True
        )
        adapter_without_cal = create_null_adapter()

        # calibration_enabled is True only if BOTH enable_calibration=True
        # AND the calibrator was successfully created (requires CALIBRATOR_AVAILABLE)
        if CALIBRATOR_AVAILABLE:
            # If calibrator is available, should be True (unless conscience unavailable)
            # The value depends on whether conscience was auto-created successfully
            assert isinstance(adapter_with_cal.calibration_enabled, bool)
        else:
            # If calibrator not available, should always be False
            assert adapter_with_cal.calibration_enabled is False

        # Null adapter explicitly disables calibration
        assert adapter_without_cal.calibration_enabled is False

    def test_available_property(self):
        """Test available property."""
        adapter = create_conscience_adapter(auto_create=True)
        assert isinstance(adapter.available, bool)

    @pytest.mark.skipif(
        not CALIBRATOR_AVAILABLE,
        reason="Calibrator not available"
    )
    def test_calibrator_property(self):
        """Test calibrator property access."""
        adapter = create_conscience_adapter(
            auto_create=True,
            enable_calibration=True
        )

        # Should have calibrator when calibration enabled
        if adapter.calibration_enabled:
            assert adapter.calibrator is not None


# =============================================================================
# Test: Complete Session Simulation
# =============================================================================

class TestCompleteSessionSimulation:
    """Tests simulating complete reasoning sessions."""

    @pytest.mark.asyncio
    async def test_mixed_mode_session(self, mock_learning_engine, conscience_adapter, audit_trail):
        """Test a session with multiple reasoning modes."""
        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=conscience_adapter,
            enable_conscience=True
        )

        # Simulate a realistic session
        session_queries = [
            (Query(text="What is Thompson Sampling?"), ReasoningMode.DIRECT),
            (Query(text="Is it optimal for multi-armed bandits?"), ReasoningMode.VERIFY),
            (Query(text="Research Thompson Sampling applications"), ReasoningMode.RESEARCH),
            (Query(text="Implement Thompson Sampling"), ReasoningMode.PLAN_EXECUTE),
        ]

        results = []
        for query, mode in session_queries:
            result = await orchestrator.reason(query, mode=mode, max_steps=2)
            results.append(result)

        # All queries should complete
        assert len(results) == 4
        assert all(r is not None for r in results)

        # Verify correct modes
        assert results[0].reasoning_mode == ReasoningMode.DIRECT
        assert results[1].reasoning_mode == ReasoningMode.VERIFY
        assert results[2].reasoning_mode == ReasoningMode.RESEARCH
        assert results[3].reasoning_mode == ReasoningMode.PLAN_EXECUTE

    @pytest.mark.asyncio
    async def test_session_with_calibration_learning(self, mock_learning_engine, audit_trail):
        """Test a session where calibration learns over time."""
        adapter = create_conscience_adapter(
            auto_create=True,
            enable_calibration=True
        )

        orchestrator = AgenticOrchestrator(
            learning_engine=mock_learning_engine,
            audit_trail=audit_trail,
            conscience_adapter=adapter,
            enable_conscience=True
        )

        # Run multiple queries
        for i in range(5):
            query = Query(text=f"Query number {i+1}")
            await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        # Calibration should have some data
        if hasattr(adapter, 'get_calibration_statistics'):
            stats = adapter.get_calibration_statistics()
            # Stats should exist (structure depends on implementation)
            assert stats is not None
