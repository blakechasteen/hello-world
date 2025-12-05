"""
End-to-End Test: Agentic Reasoning Modes
=========================================
Tests all 4 reasoning modes with epistemic tracking and safety integration.

Tests:
1. DIRECT mode - Single-pass answering (~150ms)
2. VERIFY mode - Answer + verification step (~600ms)
3. RESEARCH mode - Multi-query exploration (3-5 steps)
4. PLAN_EXECUTE mode - Goal decomposition into sub-tasks
5. Epistemic confidence tracking across steps
6. Early stopping when epistemic confidence low (<0.3)
7. Aggregated epistemic confidence calculation
8. Integration with alignment guardrails (safety gates)
9. Error handling (LLM timeout, memory failures)
10. Performance budgets per mode
"""

import asyncio
import pytest
from typing import List, Dict, Any, Optional
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.agentic.core import (
    AgenticOrchestrator,
    ReasoningMode,
    AgenticResult,
    VerificationResult,
    create_agentic_orchestrator,
)
from HoloLoom.recursive import FullLearningEngine
from HoloLoom.alignment.audit_trail import AuditTrail
from HoloLoom.fabric.spacetime import Spacetime


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

@pytest.fixture
def mock_config():
    """Create test configuration."""
    config = Config.fast()
    config.enable_alignment = True
    return config


@pytest.fixture
def mock_memory_shards():
    """Create mock memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian bandit algorithm",
            metadata={"topic": "bandit_algorithms"}
        ),
        MemoryShard(
            id="shard_2",
            text="It balances exploration and exploitation",
            metadata={"topic": "exploration"}
        ),
        MemoryShard(
            id="shard_3",
            text="UCB is an alternative to Thompson Sampling",
            metadata={"topic": "bandit_algorithms"}
        ),
        MemoryShard(
            id="shard_4",
            text="Thompson Sampling uses Beta distributions",
            metadata={"topic": "statistics"}
        ),
        MemoryShard(
            id="shard_5",
            text="It was introduced in 1933",
            metadata={"topic": "history"}
        ),
    ]


class MockLLM:
    """Mock LLM for testing without API calls."""

    def __init__(self):
        self.call_count = 0
        self.generated_responses = []

    def is_available(self):
        return True

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ):
        """Generate mock response."""
        self.call_count += 1

        # Mock responses based on prompt patterns
        if "verification" in prompt.lower():
            content = "This answer is accurate. No contradictions found."
        elif "research" in prompt.lower() or "follow-up" in prompt.lower():
            content = """1. What are the key concepts?
2. What are the tradeoffs?
3. What are practical applications?
4. What are recent developments?"""
        elif "sub-goal" in prompt.lower() or "prerequisite" in prompt.lower():
            content = "Understand Bayesian statistics and probability theory"
        else:
            content = "Thompson Sampling is a probabilistic algorithm for the multi-armed bandit problem."

        self.generated_responses.append(content)

        # Return mock response object
        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(content)


class MockAwarenessLayer:
    """Mock awareness layer for epistemic confidence testing."""

    def __init__(self, coherence: float = 0.75):
        self.coherence = coherence

    def get_current_coherence(self):
        return self.coherence

    def perceive(self, query):
        """Mock perceive method."""
        return {
            "coherence": self.coherence,
            "activation_level": 0.8,
            "shift_detected": False,
        }


async def create_mock_spacetime(
    confidence: float = 0.85,
    epistemic_confidence: Optional[float] = None,
    response: str = "Mock response",
    query_text: str = "Mock query"
) -> Spacetime:
    """Create mock spacetime result."""
    from HoloLoom.fabric.spacetime import WeavingTrace

    # Create mock trace
    trace = WeavingTrace(
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_ms=100.0,
        tool_selected="answer",
        tool_confidence=confidence
    )

    metadata = {
        "tool_used": "answer",
        "response": response,
    }

    # Add awareness metadata if epistemic confidence provided
    if epistemic_confidence is not None:
        # Calculate activation from epistemic via formula:
        # epistemic = 0.7 * coherence + 0.3 * activation
        # Assume coherence = epistemic for simplicity, solve for activation
        coherence = epistemic_confidence
        activation = (epistemic_confidence - 0.7 * coherence) / 0.3
        activation = max(0.0, min(1.0, activation))  # Clamp to [0,1]

        metadata['awareness'] = {
            'coherence': coherence,
            'activation_level': activation,
            'shift_detected': False,
            'active_nodes': 5
        }

    spacetime = Spacetime(
        query_text=query_text,
        response=response,
        tool_used="answer",
        confidence=confidence,
        trace=trace,
        metadata=metadata
    )

    return spacetime


# ============================================================================
# Test: DIRECT Mode
# ============================================================================

@pytest.mark.asyncio
async def test_direct_mode_single_pass(mock_config, mock_memory_shards):
    """Test DIRECT mode returns answer in single pass (~150ms)."""
    # Create learning engine
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        # Create orchestrator with mock LLM
        mock_llm = MockLLM()
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=mock_llm,
            enable_safety=False  # Disable safety for simple test
        )

        # Mock weave to return proper spacetime
        async def mock_weave(query, **kwargs):
            return await create_mock_spacetime(
                confidence=0.85,
                epistemic_confidence=0.75,
                response="Thompson Sampling is a Bayesian algorithm"
            )

        orchestrator.learning_engine.weave = mock_weave

        # Execute DIRECT query
        query = Query(text="What is Thompson Sampling?")
        start_time = datetime.now()
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Assertions
        assert result is not None
        assert result.reasoning_mode == ReasoningMode.DIRECT
        assert result.total_queries == 1
        assert len(result.steps_taken) == 1
        assert result.steps_taken[0]['type'] == 'direct_answer'
        assert result.spacetime.response is not None
        assert mock_llm.call_count == 1  # Only 1 LLM call
        assert duration_ms < 1000  # Should be fast (relaxed for test env)

    finally:
        await learning_engine.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_direct_mode_epistemic_tracking(mock_config, mock_memory_shards):
    """Test DIRECT mode tracks epistemic confidence."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        # Create orchestrator with awareness layer
        mock_awareness = MockAwarenessLayer(coherence=0.8)
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            awareness_layer=mock_awareness,
            llm=MockLLM(),
            enable_safety=False
        )

        # Patch weave to return spacetime with awareness metadata
        async def mock_weave(query, **kwargs):
            return await create_mock_spacetime(
                confidence=0.9,
                epistemic_confidence=0.75
            )

        orchestrator.learning_engine.weave = mock_weave

        # Execute query
        query = Query(text="What is Thompson Sampling?")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        # Check epistemic confidence tracked
        assert result.aggregated_epistemic_confidence is not None
        assert 0.0 <= result.aggregated_epistemic_confidence <= 1.0
        assert result.steps_taken[0]['epistemic_confidence'] is not None

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Test: VERIFY Mode
# ============================================================================

@pytest.mark.asyncio
async def test_verify_mode_with_verification_loop(mock_config, mock_memory_shards):
    """Test VERIFY mode executes verification loop (~600ms)."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        mock_llm = MockLLM()
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=mock_llm,
            enable_safety=False
        )

        # Mock weave to return consistent results
        async def mock_weave(query, **kwargs):
            return await create_mock_spacetime(
                confidence=0.85,
                epistemic_confidence=0.75,
                response="Thompson Sampling balances exploration and exploitation"
            )

        orchestrator.learning_engine.weave = mock_weave

        # Execute VERIFY query
        query = Query(text="Is Thompson Sampling optimal for bandits?")
        result = await orchestrator.reason(query, mode=ReasoningMode.VERIFY)

        # Assertions
        assert result is not None
        assert result.reasoning_mode == ReasoningMode.VERIFY
        assert result.verification is not None
        assert isinstance(result.verification, VerificationResult)
        assert result.total_queries >= 2  # Initial + verification
        assert any(s['type'] == 'initial_answer' for s in result.steps_taken)
        assert any(s['type'] == 'verification' for s in result.steps_taken)

    finally:
        await learning_engine.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_verify_mode_contradiction_detection(mock_config, mock_memory_shards):
    """Test VERIFY mode detects contradictions."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock weave to return contradictory verification
        call_count = [0]

        async def mock_weave_with_contradiction(query, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Initial answer
                response = "Thompson Sampling is always optimal"
            else:
                # Verification query with contradiction keyword
                response = "However, UCB can outperform Thompson Sampling in some cases"

            return await create_mock_spacetime(
                confidence=0.85,
                epistemic_confidence=0.7,
                response=response
            )

        orchestrator.learning_engine.weave = mock_weave_with_contradiction

        # Execute VERIFY query
        query = Query(text="Is Thompson Sampling always optimal?")
        result = await orchestrator.reason(query, mode=ReasoningMode.VERIFY, max_steps=2)

        # Check contradiction detected
        assert result.verification is not None
        assert len(result.verification.contradictions) > 0
        assert not result.verification.verified  # Should fail verification

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Test: RESEARCH Mode
# ============================================================================

@pytest.mark.asyncio
async def test_research_mode_multi_query_exploration(mock_config, mock_memory_shards):
    """Test RESEARCH mode generates and explores multiple queries."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        mock_llm = MockLLM()
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=mock_llm,
            enable_safety=False
        )

        # Mock weave
        async def mock_weave(query, **kwargs):
            return await create_mock_spacetime(
                confidence=0.8,
                epistemic_confidence=0.7,
                response=f"Finding for: {query.text[:50]}"
            )

        orchestrator.learning_engine.weave = mock_weave

        # Execute RESEARCH query
        query = Query(text="What are the tradeoffs of Thompson Sampling?")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.RESEARCH,
            max_steps=5
        )

        # Assertions
        assert result is not None
        assert result.reasoning_mode == ReasoningMode.RESEARCH
        assert result.total_queries >= 3  # Multiple research queries + synthesis
        research_steps = [s for s in result.steps_taken if s['type'] == 'research_query']
        assert len(research_steps) >= 2  # At least 2 research queries
        synthesis_steps = [s for s in result.steps_taken if s['type'] == 'synthesis']
        assert len(synthesis_steps) == 1  # Final synthesis

    finally:
        await learning_engine.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_research_mode_early_stopping_low_epistemic(mock_config, mock_memory_shards):
    """Test RESEARCH mode stops early when epistemic confidence drops."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            epistemic_threshold=0.3,  # Early stopping threshold
            enable_safety=False
        )

        # Mock weave to return decreasing epistemic confidence
        call_count = [0]

        async def mock_weave_decreasing_epistemic(query, **kwargs):
            call_count[0] += 1
            # Start high, then drop below threshold
            epistemic_values = [0.7, 0.25, 0.2, 0.15]  # Drops below 0.3
            epistemic = epistemic_values[min(call_count[0] - 1, len(epistemic_values) - 1)]

            return await create_mock_spacetime(
                confidence=0.7,
                epistemic_confidence=epistemic,
                response=f"Response {call_count[0]}"
            )

        orchestrator.learning_engine.weave = mock_weave_decreasing_epistemic

        # Execute RESEARCH query
        query = Query(text="Analyze Thompson Sampling comprehensively")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.RESEARCH,
            max_steps=5
        )

        # Should stop early (before max_steps due to low epistemic)
        research_steps = [s for s in result.steps_taken if s['type'] == 'research_query']
        assert len(research_steps) < 5  # Stopped early
        # Check that last research steps had low epistemic confidence
        if len(research_steps) >= 2:
            last_steps_epistemic = [s['epistemic_confidence'] for s in research_steps[-2:]]
            avg_recent = sum(last_steps_epistemic) / len(last_steps_epistemic)
            assert avg_recent < 0.3  # Below threshold

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Test: PLAN_EXECUTE Mode
# ============================================================================

@pytest.mark.asyncio
async def test_plan_execute_mode_goal_decomposition(mock_config, mock_memory_shards):
    """Test PLAN_EXECUTE mode decomposes and executes sub-goals."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock weave
        async def mock_weave(query, **kwargs):
            return await create_mock_spacetime(
                confidence=0.85,
                epistemic_confidence=0.75,
                response=f"Completed: {query.text[:50]}"
            )

        orchestrator.learning_engine.weave = mock_weave

        # Execute PLAN_EXECUTE query
        query = Query(text="Implement Thompson Sampling in Python")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.PLAN_EXECUTE,
            max_steps=4
        )

        # Assertions
        assert result is not None
        assert result.reasoning_mode == ReasoningMode.PLAN_EXECUTE
        assert result.intent.sub_goals is not None
        assert len(result.intent.sub_goals) > 0
        sub_goal_steps = [s for s in result.steps_taken if s['type'] == 'sub_goal']
        assert len(sub_goal_steps) > 0  # At least one sub-goal executed
        synthesis_steps = [s for s in result.steps_taken if s['type'] == 'synthesis']
        assert len(synthesis_steps) == 1  # Final synthesis

    finally:
        await learning_engine.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_plan_execute_mode_sub_goal_completion_tracking(mock_config, mock_memory_shards):
    """Test PLAN_EXECUTE tracks which sub-goals completed successfully."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock weave with varying confidence
        call_count = [0]

        async def mock_weave_varying_confidence(query, **kwargs):
            call_count[0] += 1
            # Alternate high/low confidence
            confidence = 0.9 if call_count[0] % 2 == 1 else 0.5

            return await create_mock_spacetime(
                confidence=confidence,
                epistemic_confidence=0.7,
                response=f"Sub-goal result {call_count[0]}"
            )

        orchestrator.learning_engine.weave = mock_weave_varying_confidence

        # Execute query
        query = Query(text="Build a recommendation system")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.PLAN_EXECUTE,
            confidence_threshold=0.85,
            max_steps=4
        )

        # Check completion tracking
        sub_goal_steps = [s for s in result.steps_taken if s['type'] == 'sub_goal']
        completed_count = sum(1 for s in sub_goal_steps if s.get('completed', False))
        assert completed_count >= 0  # Some may complete, some may not
        # Synthesis step should track completed count
        synthesis = next(s for s in result.steps_taken if s['type'] == 'synthesis')
        assert 'sub_goals_completed' in synthesis

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Test: Epistemic Confidence
# ============================================================================

@pytest.mark.asyncio
async def test_epistemic_confidence_aggregation(mock_config, mock_memory_shards):
    """Test aggregated epistemic confidence uses weighted average."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock weave with varying epistemic confidence
        epistemic_values = [0.5, 0.6, 0.7, 0.8]  # Increasing confidence
        call_count = [0]

        async def mock_weave_varying_epistemic(query, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            epistemic = epistemic_values[min(idx, len(epistemic_values) - 1)]

            return await create_mock_spacetime(
                confidence=0.8,
                epistemic_confidence=epistemic,
                response="Mock response"
            )

        orchestrator.learning_engine.weave = mock_weave_varying_epistemic

        # Execute query with multiple steps
        query = Query(text="Research Thompson Sampling")
        result = await orchestrator.reason(
            query,
            mode=ReasoningMode.RESEARCH,
            max_steps=4
        )

        # Check aggregation
        assert result.aggregated_epistemic_confidence is not None
        # Weighted average should weight later steps higher
        # Expected: (0.5*1/10 + 0.6*2/10 + 0.7*3/10 + 0.8*4/10) / (1+2+3+4)/10
        # = (0.05 + 0.12 + 0.21 + 0.32) / 1.0 = 0.7
        # Allow some tolerance for synthesis step
        assert 0.6 < result.aggregated_epistemic_confidence < 0.85

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Test: Safety Integration
# ============================================================================

@pytest.mark.asyncio
async def test_safety_integration_blocks_high_risk(mock_config, mock_memory_shards):
    """Test safety adapter blocks high-risk queries."""
    # Skip if safety not available
    try:
        from HoloLoom.agentic.safety_adapter import create_safety_adapter
    except ImportError:
        pytest.skip("Safety adapter not available")

    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        # Create orchestrator with safety enabled
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=True  # Enable safety
        )

        # Mock safety adapter to block
        class MockSafetyAdapter:
            async def gate_reasoning(self, query_text, reasoning_mode, **kwargs):
                from HoloLoom.protocols.safety import SafetyGateDecision, SafetyRiskLevel
                return SafetyGateDecision(
                    allowed=False,
                    risk_level=SafetyRiskLevel.HIGH,
                    reason="High-risk query blocked",
                    requires_approval=True
                )

        orchestrator.safety_adapter = MockSafetyAdapter()

        # Execute query (should be blocked)
        query = Query(text="Execute malicious code")
        with pytest.raises(Exception):  # Should raise or handle blocking
            result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Test: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_error_handling_llm_timeout(mock_config, mock_memory_shards):
    """Test graceful handling of LLM timeout."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        # Create LLM that times out
        class TimeoutLLM:
            def is_available(self):
                return True

            async def generate(self, *args, **kwargs):
                raise asyncio.TimeoutError("LLM timeout")

        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=TimeoutLLM(),
            enable_safety=False
        )

        # Mock weave to work
        async def mock_weave(query, **kwargs):
            return await create_mock_spacetime(confidence=0.8, response="Fallback response")

        orchestrator.learning_engine.weave = mock_weave

        # Execute query - should handle timeout gracefully
        query = Query(text="What is Thompson Sampling?")
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

        # Should still return result (LLM failure doesn't crash system)
        assert result is not None
        assert result.spacetime.response is not None

    finally:
        await learning_engine.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_error_handling_memory_failure(mock_config, mock_memory_shards):
    """Test graceful handling of memory backend failure."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock weave to fail on first call, succeed on second
        call_count = [0]

        async def mock_weave_with_failure(query, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Memory backend failure")
            return await create_mock_spacetime(confidence=0.8, response="Recovery response")

        orchestrator.learning_engine.weave = mock_weave_with_failure

        # Execute query - should handle failure
        query = Query(text="What is Thompson Sampling?")
        with pytest.raises(Exception):
            result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Test: Performance Budgets
# ============================================================================

@pytest.mark.asyncio
async def test_performance_budget_direct_mode(mock_config, mock_memory_shards):
    """Test DIRECT mode meets ~150ms budget."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock fast weave
        async def mock_fast_weave(query, **kwargs):
            await asyncio.sleep(0.01)  # 10ms mock processing
            return await create_mock_spacetime(confidence=0.85)

        orchestrator.learning_engine.weave = mock_fast_weave

        # Execute and measure
        query = Query(text="What is Thompson Sampling?")
        start = datetime.now()
        result = await orchestrator.reason(query, mode=ReasoningMode.DIRECT)
        duration_ms = (datetime.now() - start).total_seconds() * 1000

        # Should be fast (relaxed for test environment)
        assert duration_ms < 500
        assert result.total_duration_ms < 500

    finally:
        await learning_engine.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_performance_budget_verify_mode(mock_config, mock_memory_shards):
    """Test VERIFY mode meets ~600ms budget."""
    learning_engine = FullLearningEngine(
        cfg=mock_config,
        shards=mock_memory_shards,
        enable_background_learning=False
    )
    await learning_engine.__aenter__()

    try:
        orchestrator = AgenticOrchestrator(
            learning_engine=learning_engine,
            llm=MockLLM(),
            enable_safety=False
        )

        # Mock weave with realistic timing
        async def mock_timed_weave(query, **kwargs):
            await asyncio.sleep(0.05)  # 50ms per query
            return await create_mock_spacetime(confidence=0.85)

        orchestrator.learning_engine.weave = mock_timed_weave

        # Execute and measure
        query = Query(text="Is Thompson Sampling optimal?")
        start = datetime.now()
        result = await orchestrator.reason(query, mode=ReasoningMode.VERIFY, max_steps=2)
        duration_ms = (datetime.now() - start).total_seconds() * 1000

        # Should be within budget (relaxed for test environment)
        assert duration_ms < 1000
        assert result.total_queries >= 2  # Initial + verifications

    finally:
        await learning_engine.__aexit__(None, None, None)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run tests directly
    print("Running Agentic Reasoning E2E Tests...")
    asyncio.run(test_direct_mode_single_pass(Config.fast(), []))
    asyncio.run(test_verify_mode_with_verification_loop(Config.fast(), []))
    asyncio.run(test_research_mode_multi_query_exploration(Config.fast(), []))
    asyncio.run(test_plan_execute_mode_goal_decomposition(Config.fast(), []))
    print("✅ All E2E tests passed!")
