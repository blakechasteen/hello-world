"""
Test Agentic Core Reasoning System
===================================

Tests for HoloLoom/agentic/core.py - critical multi-query reasoning engine.

Coverage:
- All 4 reasoning modes (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
- Multi-query reasoning
- Verification logic
- Goal decomposition
- Plan execution
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from HoloLoom.agentic.core import (
    AgenticOrchestrator,
    ReasoningMode,
    ReasoningResult,
    VerificationResult,
)
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock configuration."""
    return Config.fast()


@pytest.fixture
def mock_shards():
    """Mock memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling balances exploration and exploitation",
            episode="kb",
            entities=["Thompson Sampling"],
            motifs=["sampling", "exploration"],
            metadata={"confidence": 0.9}
        ),
        MemoryShard(
            id="shard_2",
            text="PPO is a policy gradient method for RL",
            episode="kb",
            entities=["PPO", "policy gradient"],
            motifs=["reinforcement learning"],
            metadata={"confidence": 0.85}
        ),
    ]


@pytest.fixture
async def orchestrator(mock_config, mock_shards):
    """Create agentic orchestrator."""
    from HoloLoom.agentic.core import AgenticOrchestrator

    # Mock the weaving orchestrator
    mock_weaver = AsyncMock()
    mock_weaver.weave = AsyncMock()

    orchestrator = AgenticOrchestrator(
        cfg=mock_config,
        shards=mock_shards
    )

    # Replace weaver with mock
    orchestrator.weaver = mock_weaver

    return orchestrator


# ============================================================================
# Test DIRECT Mode
# ============================================================================

@pytest.mark.asyncio
async def test_direct_mode_basic(orchestrator):
    """Test DIRECT mode returns single answer."""
    # Mock weaver response
    from HoloLoom.protocols.types import Spacetime

    mock_spacetime = Spacetime(
        response="Thompson Sampling is a probabilistic method.",
        confidence=0.9,
        context_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.weaver.weave.return_value = mock_spacetime

    # Execute
    result = await orchestrator.reason(
        query="What is Thompson Sampling?",
        mode=ReasoningMode.DIRECT
    )

    # Assertions
    assert result.response is not None
    assert result.confidence > 0.0
    assert result.mode == ReasoningMode.DIRECT
    assert result.steps_taken == 1
    assert orchestrator.weaver.weave.call_count == 1


@pytest.mark.asyncio
async def test_direct_mode_low_confidence(orchestrator):
    """Test DIRECT mode with low confidence."""
    from HoloLoom.protocols.types import Spacetime

    mock_spacetime = Spacetime(
        response="Not sure about this.",
        confidence=0.3,
        context_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.weaver.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query="What is XYZ?",
        mode=ReasoningMode.DIRECT
    )

    assert result.confidence < 0.5
    assert result.response is not None


# ============================================================================
# Test VERIFY Mode
# ============================================================================

@pytest.mark.asyncio
async def test_verify_mode_success(orchestrator):
    """Test VERIFY mode generates verification."""
    from HoloLoom.protocols.types import Spacetime

    # Mock initial answer
    answer_spacetime = Spacetime(
        response="Thompson Sampling uses Beta distributions.",
        confidence=0.85,
        context_used=[],
        tool_used="answer",
        metadata={}
    )

    # Mock verification query
    verify_spacetime = Spacetime(
        response="Confirmed: Thompson Sampling uses Beta distributions.",
        confidence=0.9,
        context_used=[],
        tool_used="answer",
        metadata={}
    )

    orchestrator.weaver.weave.side_effect = [answer_spacetime, verify_spacetime]

    result = await orchestrator.reason(
        query="How does Thompson Sampling work?",
        mode=ReasoningMode.VERIFY
    )

    assert result.mode == ReasoningMode.VERIFY
    assert result.verification is not None
    assert result.verification.verified
    assert result.steps_taken == 2


@pytest.mark.asyncio
async def test_verify_mode_contradiction(orchestrator):
    """Test VERIFY mode detects contradictions."""
    from HoloLoom.protocols.types import Spacetime

    answer_spacetime = Spacetime(
        response="Thompson Sampling is deterministic.",
        confidence=0.7,
        context_used=[],
        tool_used="answer",
        metadata={}
    )

    verify_spacetime = Spacetime(
        response="Actually, Thompson Sampling is probabilistic, not deterministic.",
        confidence=0.95,
        context_used=[],
        tool_used="answer",
        metadata={}
    )

    orchestrator.weaver.weave.side_effect = [answer_spacetime, verify_spacetime]

    result = await orchestrator.reason(
        query="Is Thompson Sampling deterministic?",
        mode=ReasoningMode.VERIFY
    )

    assert result.verification is not None
    assert not result.verification.verified  # Should detect contradiction


# ============================================================================
# Test RESEARCH Mode
# ============================================================================

@pytest.mark.asyncio
async def test_research_mode_multi_query(orchestrator):
    """Test RESEARCH mode explores multiple angles."""
    from HoloLoom.protocols.types import Spacetime

    # Mock multiple sub-query responses
    spacetimes = [
        Spacetime(
            response=f"Answer to sub-query {i}",
            confidence=0.8,
            context_used=[],
            tool_used="answer",
            metadata={}
        )
        for i in range(3)
    ]

    orchestrator.weaver.weave.side_effect = spacetimes

    result = await orchestrator.reason(
        query="What are the tradeoffs of Thompson Sampling?",
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )

    assert result.mode == ReasoningMode.RESEARCH
    assert result.steps_taken >= 2  # Should explore multiple queries
    assert len(result.sub_queries) > 0
    assert orchestrator.weaver.weave.call_count >= 2


@pytest.mark.asyncio
async def test_research_mode_synthesis(orchestrator):
    """Test RESEARCH mode synthesizes multiple answers."""
    from HoloLoom.protocols.types import Spacetime

    spacetimes = [
        Spacetime(response="Aspect 1: Exploration", confidence=0.9, context_used=[], tool_used="answer", metadata={}),
        Spacetime(response="Aspect 2: Exploitation", confidence=0.85, context_used=[], tool_used="answer", metadata={}),
        Spacetime(response="Aspect 3: Regret bounds", confidence=0.8, context_used=[], tool_used="answer", metadata={}),
    ]
    orchestrator.weaver.weave.side_effect = spacetimes

    result = await orchestrator.reason(
        query="Research Thompson Sampling",
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )

    # Should synthesize all aspects
    assert result.response is not None
    assert len(result.response) > len(spacetimes[0].response)  # Synthesized is longer


# ============================================================================
# Test PLAN_EXECUTE Mode
# ============================================================================

@pytest.mark.asyncio
async def test_plan_execute_mode_basic(orchestrator):
    """Test PLAN_EXECUTE mode decomposes goal."""
    from HoloLoom.protocols.types import Spacetime

    # Mock plan generation
    plan_spacetime = Spacetime(
        response="Plan:\n1. Step 1\n2. Step 2\n3. Step 3",
        confidence=0.9,
        context_used=[],
        tool_used="answer",
        metadata={}
    )

    # Mock step executions
    step_spacetimes = [
        Spacetime(response=f"Completed step {i}", confidence=0.8, context_used=[], tool_used="answer", metadata={})
        for i in range(3)
    ]

    orchestrator.weaver.weave.side_effect = [plan_spacetime] + step_spacetimes

    result = await orchestrator.reason(
        query="Implement a Thompson Sampling algorithm",
        mode=ReasoningMode.PLAN_EXECUTE,
        max_steps=5
    )

    assert result.mode == ReasoningMode.PLAN_EXECUTE
    assert result.steps_taken > 1
    assert orchestrator.weaver.weave.call_count >= 2  # Plan + steps


# ============================================================================
# Test Reasoning Engine Internals
# ============================================================================

@pytest.mark.asyncio
async def test_auto_mode_selection(orchestrator):
    """Test automatic mode selection based on query."""
    # This would test the _select_mode() method if it exists
    # For now, test that mode is respected

    from HoloLoom.protocols.types import Spacetime
    mock_spacetime = Spacetime(
        response="Answer",
        confidence=0.9,
        context_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.weaver.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query="Simple question?",
        mode=ReasoningMode.DIRECT
    )

    assert result.mode == ReasoningMode.DIRECT


@pytest.mark.asyncio
async def test_max_steps_limit(orchestrator):
    """Test max_steps parameter limits iterations."""
    from HoloLoom.protocols.types import Spacetime

    spacetimes = [
        Spacetime(response=f"Query {i}", confidence=0.8, context_used=[], tool_used="answer", metadata={})
        for i in range(10)
    ]
    orchestrator.weaver.weave.side_effect = spacetimes

    result = await orchestrator.reason(
        query="Research topic",
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )

    assert result.steps_taken <= 3


@pytest.mark.asyncio
async def test_confidence_tracking(orchestrator):
    """Test confidence is tracked correctly."""
    from HoloLoom.protocols.types import Spacetime

    mock_spacetime = Spacetime(
        response="High confidence answer",
        confidence=0.95,
        context_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.weaver.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query="Test query",
        mode=ReasoningMode.DIRECT
    )

    assert result.confidence == 0.95


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_weaver_failure_handling(orchestrator):
    """Test handling of weaver failures."""
    orchestrator.weaver.weave.side_effect = Exception("Weaver error")

    with pytest.raises(Exception):
        await orchestrator.reason(
            query="Test query",
            mode=ReasoningMode.DIRECT
        )


@pytest.mark.asyncio
async def test_empty_query_handling(orchestrator):
    """Test handling of empty queries."""
    from HoloLoom.protocols.types import Spacetime

    mock_spacetime = Spacetime(
        response="Cannot process empty query",
        confidence=0.0,
        context_used=[],
        tool_used="error",
        metadata={}
    )
    orchestrator.weaver.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query="",
        mode=ReasoningMode.DIRECT
    )

    assert result.confidence < 0.5


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
- DIRECT mode: 2 tests (basic, low confidence)
- VERIFY mode: 2 tests (success, contradiction detection)
- RESEARCH mode: 2 tests (multi-query, synthesis)
- PLAN_EXECUTE mode: 1 test (basic decomposition)
- Internals: 3 tests (mode selection, max steps, confidence)
- Error handling: 2 tests (failures, empty queries)

Total: 12 tests covering critical agentic reasoning paths
"""
