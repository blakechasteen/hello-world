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

from hololoom.agentic.core import (
    AgenticOrchestrator,
    ReasoningMode,
    AgenticResult,
    VerificationResult,
)
from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional

@dataclass
class MockSpacetime:
    response: str = ""
    confidence: float = 0.0
    tool_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources_used: List[str] = field(default_factory=list)
    query_text: str = "mock query"
    trace: Any = None
    context_used: List[Any] = field(default_factory=list) # For legacy support if needed

    def __init__(self, **kwargs):
        self.response = kwargs.get('response', '')
        self.confidence = kwargs.get('confidence', 0.0)
        self.tool_used = kwargs.get('tool_used', '')
        self.metadata = kwargs.get('metadata', {})
        self.sources_used = kwargs.get('sources_used', [])
        self.query_text = kwargs.get('query_text', 'mock query')
        self.trace = kwargs.get('trace')
        self.context_used = kwargs.get('context_used', [])
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)


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
    from hololoom.agentic.core import AgenticOrchestrator

    # Mock the learning engine
    mock_learning_engine = AsyncMock()
    mock_learning_engine.weave = AsyncMock()
    
    # Mock internal orchestrator structure for LLM discovery
    mock_inner_orchestrator = Mock()
    mock_tool_executor = Mock()
    mock_tool_executor.llm = AsyncMock()
    mock_inner_orchestrator.tool_executor = mock_tool_executor
    mock_learning_engine.orchestrator = mock_inner_orchestrator

    orchestrator = AgenticOrchestrator(
        learning_engine=mock_learning_engine,
        enable_safety=False,
        enable_conscience=False
    )

    return orchestrator


# ============================================================================
# Test DIRECT Mode
# ============================================================================

@pytest.mark.asyncio
async def test_direct_mode_basic(orchestrator):
    """Test DIRECT mode returns single answer."""
    # Mock weaver response
    Spacetime = MockSpacetime

    mock_spacetime = Spacetime(
        response="Thompson Sampling is a probabilistic method.",
        confidence=0.9,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.learning_engine.weave.return_value = mock_spacetime

    # Execute
    result = await orchestrator.reason(
        query=Query(text="What is Thompson Sampling?"),
        mode=ReasoningMode.DIRECT
    )

    # Assertions
    assert result.spacetime.response is not None
    assert result.spacetime.confidence > 0.0
    assert result.reasoning_mode == ReasoningMode.DIRECT
    assert len(result.steps_taken) > 0
    assert orchestrator.learning_engine.weave.call_count == 1


@pytest.mark.asyncio
async def test_direct_mode_low_confidence(orchestrator):
    """Test DIRECT mode with low confidence."""
    Spacetime = MockSpacetime

    mock_spacetime = Spacetime(
        response="Not sure about this.",
        confidence=0.3,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.learning_engine.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query=Query(text="What is XYZ?"),
        mode=ReasoningMode.DIRECT
    )

    assert result.spacetime.confidence < 0.5
    assert result.spacetime.response is not None


# ============================================================================
# Test VERIFY Mode
# ============================================================================

@pytest.mark.asyncio
async def test_verify_mode_success(orchestrator):
    """Test VERIFY mode generates verification."""
    Spacetime = MockSpacetime

    # Mock initial answer
    answer_spacetime = Spacetime(
        response="Thompson Sampling uses Beta distributions.",
        confidence=0.85,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )

    # Mock verification query
    verify_spacetime = Spacetime(
        response="Confirmed: Thompson Sampling uses Beta distributions.",
        confidence=0.9,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )

    orchestrator.learning_engine.weave.side_effect = [answer_spacetime, verify_spacetime] * 5

    result = await orchestrator.reason(
        query=Query(text="How does Thompson Sampling work?"),
        mode=ReasoningMode.VERIFY
    )

    assert result.reasoning_mode == ReasoningMode.VERIFY
    assert result.verification is not None
    assert result.verification.verified
    assert len(result.steps_taken) >= 2


@pytest.mark.asyncio
async def test_verify_mode_contradiction(orchestrator):
    """Test VERIFY mode detects contradictions."""
    Spacetime = MockSpacetime

    answer_spacetime = Spacetime(
        response="Thompson Sampling is deterministic.",
        confidence=0.7,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )

    verify_spacetime = Spacetime(
        response="Actually, Thompson Sampling is probabilistic, not deterministic.",
        confidence=0.95,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )

    orchestrator.learning_engine.weave.side_effect = [answer_spacetime, verify_spacetime] * 5

    result = await orchestrator.reason(
        query=Query(text="Is Thompson Sampling deterministic?"),
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
    Spacetime = MockSpacetime

    # Mock multiple sub-query responses
    spacetimes = [
        Spacetime(
            response=f"Answer to sub-query {i}",
            confidence=0.8,
            sources_used=[],
            tool_used="answer",
            metadata={}
        )
        for i in range(3)
    ]

    orchestrator.learning_engine.weave.side_effect = spacetimes * 3

    result = await orchestrator.reason(
        query=Query(text="What are the tradeoffs of Thompson Sampling?"),
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )

    assert result.reasoning_mode == ReasoningMode.RESEARCH
    assert len(result.steps_taken) >= 2  # Should explore multiple queries

    assert orchestrator.learning_engine.weave.call_count >= 2


@pytest.mark.asyncio
async def test_research_mode_synthesis(orchestrator):
    """Test RESEARCH mode synthesizes multiple answers."""
    Spacetime = MockSpacetime

    spacetimes = [
        Spacetime(response="Aspect 1: Exploration", confidence=0.9, sources_used=[], tool_used="answer", metadata={}),
        Spacetime(response="Aspect 2: Exploitation", confidence=0.85, sources_used=[], tool_used="answer", metadata={}),
        Spacetime(response="Aspect 3: Regret bounds", confidence=0.8, sources_used=[], tool_used="answer", metadata={}),
    ]
    synthesis_spacetime = Spacetime(
        response="Comprehensive synthesis of all research findings...",
        confidence=0.95,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.learning_engine.weave.side_effect = spacetimes + [synthesis_spacetime]

    result = await orchestrator.reason(
        query=Query(text="Research Thompson Sampling"),
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )

    # Should synthesize all aspects
    assert result.spacetime.response is not None
    assert len(result.spacetime.response) > len(spacetimes[0].response)  # Synthesized is longer


# ============================================================================
# Test PLAN_EXECUTE Mode
# ============================================================================

@pytest.mark.asyncio
async def test_plan_execute_mode_basic(orchestrator):
    """Test PLAN_EXECUTE mode decomposes goal."""
    Spacetime = MockSpacetime

    # Mock plan generation
    plan_spacetime = Spacetime(
        response="Plan:\n1. Step 1\n2. Step 2\n3. Step 3",
        confidence=0.9,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )

    # Mock step executions
    step_spacetimes = [
        Spacetime(response=f"Completed step {i}", confidence=0.8, sources_used=[], tool_used="answer", metadata={})
        for i in range(3)
    ]

    orchestrator.learning_engine.weave.side_effect = ([plan_spacetime] + step_spacetimes) * 5

    result = await orchestrator.reason(
        query=Query(text="Implement a Thompson Sampling algorithm"),
        mode=ReasoningMode.PLAN_EXECUTE,
        max_steps=5
    )

    assert result.reasoning_mode == ReasoningMode.PLAN_EXECUTE
    assert len(result.steps_taken) > 1
    assert orchestrator.learning_engine.weave.call_count >= 2  # Plan + steps


# ============================================================================
# Test Reasoning Engine Internals
# ============================================================================

@pytest.mark.asyncio
async def test_auto_mode_selection(orchestrator):
    """Test automatic mode selection based on query."""
    # This would test the _select_mode() method if it exists
    # For now, test that mode is respected

    Spacetime = MockSpacetime
    mock_spacetime = Spacetime(
        response="Answer",
        confidence=0.9,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.learning_engine.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query=Query(text="Simple question?"),
        mode=ReasoningMode.DIRECT
    )

    assert result.reasoning_mode == ReasoningMode.DIRECT


@pytest.mark.asyncio
async def test_max_steps_limit(orchestrator):
    """Test max_steps parameter limits iterations."""
    Spacetime = MockSpacetime

    spacetimes = [
        Spacetime(response=f"Query {i}", confidence=0.8, sources_used=[], tool_used="answer", metadata={})
        for i in range(10)
    ]
    # Create enough mocks for research steps + synthesis + potential retries
    orchestrator.learning_engine.weave.side_effect = spacetimes * 5
    
    result = await orchestrator.reason(
        query=Query(text="Research topic"),
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )

    assert len(result.steps_taken) <= 4  # 3 research + 1 synthesis (approx)


@pytest.mark.asyncio
async def test_confidence_tracking(orchestrator):
    """Test confidence is tracked correctly."""
    Spacetime = MockSpacetime

    mock_spacetime = Spacetime(
        response="High confidence answer",
        confidence=0.95,
        sources_used=[],
        tool_used="answer",
        metadata={}
    )
    orchestrator.learning_engine.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query=Query(text="Test query"),
        mode=ReasoningMode.DIRECT
    )

    assert result.spacetime.confidence == 0.95


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_weaver_failure_handling(orchestrator):
    """Test handling of weaver failures."""
    orchestrator.learning_engine.weave.side_effect = Exception("Weaver error")

    with pytest.raises(Exception):
        await orchestrator.reason(
            query=Query(text="Test query"),
            mode=ReasoningMode.DIRECT
        )


@pytest.mark.asyncio
async def test_empty_query_handling(orchestrator):
    """Test handling of empty queries."""
    Spacetime = MockSpacetime

    mock_spacetime = Spacetime(
        response="Cannot process empty query",
        confidence=0.0,
        sources_used=[],
        tool_used="error",
        metadata={}
    )
    orchestrator.learning_engine.weave.return_value = mock_spacetime

    result = await orchestrator.reason(
        query=Query(text=""),
        mode=ReasoningMode.DIRECT
    )

    assert result.spacetime.confidence < 0.5


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
