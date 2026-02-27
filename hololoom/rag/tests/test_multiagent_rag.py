"""
Tests for Multi-Agent RAG System

Comprehensive test suite for multi-agent consensus-based query processing.

Test Coverage:
- Parallel execution (all agents run concurrently)
- Consensus mechanisms (majority vote, confidence weighted, LLM judge, ensemble)
- Diversity strategies (different parameters across agents)
- Timeout handling (slow agents killed)
- Partial failures (some agents fail, consensus still works)
- Agreement scoring (detect disagreement)
- Source deduplication
- Performance (parallelization speedup)

Author: Agent J (Claude Code)
Date: November 13, 2025
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from hololoom.config import Config
from hololoom.rag.simple_rag import SimpleRAG, RAGResult
from hololoom.rag.multiagent_rag import (
    MultiAgentRAG,
    AgentResponse,
    MultiAgentRAGResult,
    ConsensusMethod,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    return Config.fast()


@pytest.fixture
async def multiagent_rag(mock_config):
    """Create MultiAgentRAG instance for testing."""
    rag = MultiAgentRAG(
        config=mock_config,
        n_agents=5,
        consensus_method="confidence_weighted",
        agent_timeout=5.0,
    )
    async with rag:
        yield rag


@pytest.fixture
def sample_agent_responses():
    """Create sample agent responses for testing consensus."""
    return [
        AgentResponse(
            agent_id="agent_0",
            strategy="direct_k3",
            response="Thompson Sampling is a Bayesian exploration strategy.",
            confidence=0.9,
            sources=["source1", "source2"],
            latency_ms=100.0,
        ),
        AgentResponse(
            agent_id="agent_1",
            strategy="verify_k5",
            response="Thompson Sampling uses Bayesian statistics for exploration.",
            confidence=0.85,
            sources=["source1", "source3"],
            latency_ms=150.0,
        ),
        AgentResponse(
            agent_id="agent_2",
            strategy="research_k10",
            response="Thompson Sampling is a Bayesian approach to the explore-exploit dilemma.",
            confidence=0.92,
            sources=["source2", "source4"],
            latency_ms=200.0,
        ),
        AgentResponse(
            agent_id="agent_3",
            strategy="verify_k5_hf",
            response="Thompson Sampling balances exploration and exploitation using Bayes.",
            confidence=0.88,
            sources=["source3", "source5"],
            latency_ms=175.0,
        ),
        AgentResponse(
            agent_id="agent_4",
            strategy="plan_k7",
            response="Thompson Sampling is a Bayesian bandit algorithm.",
            confidence=0.87,
            sources=["source1", "source4"],
            latency_ms=180.0,
        ),
    ]


# ============================================================================
# Initialization Tests
# ============================================================================

@pytest.mark.asyncio
async def test_initialization(mock_config):
    """Test MultiAgentRAG initialization."""
    async with MultiAgentRAG(
        config=mock_config,
        n_agents=3,
        consensus_method="majority_vote",
        agent_timeout=10.0,
    ) as rag:
        assert len(rag.agents) == 3
        assert rag.consensus_method == ConsensusMethod.MAJORITY_VOTE
        assert rag.agent_timeout == 10.0
        assert rag._stats['total_agents_spawned'] == 3


@pytest.mark.asyncio
async def test_agent_diversity_strategies(mock_config):
    """Test that agents have diverse strategies."""
    async with MultiAgentRAG(
        config=mock_config,
        n_agents=5,
    ) as rag:
        strategies = rag._agent_strategies

        # Check diversity
        assert len(strategies) == 5
        assert len(set(s['description'] for s in strategies)) >= 3  # At least 3 unique strategies

        # Check variety in parameters
        max_sources_values = [s.get('max_sources', 0) for s in strategies]
        assert len(set(max_sources_values)) >= 2  # At least 2 different k values

        reasoning_modes = [s.get('reasoning_mode', '') for s in strategies]
        assert len(set(reasoning_modes)) >= 2  # At least 2 different modes


# ============================================================================
# Parallel Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_parallel_execution(multiagent_rag, monkeypatch):
    """Test that agents run in parallel (not sequential)."""
    # Mock agent.query to simulate work
    async def mock_query(self, question, mode, max_sources, use_cache):
        await asyncio.sleep(0.1)  # Simulate 100ms work
        return RAGResult(
            response=f"Answer from {id(self)}",
            sources=["source1"],
            confidence=0.9,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Measure time for 5 agents
    start_time = time.time()
    result = await multiagent_rag.query_multiagent("What is X?")
    elapsed_time = time.time() - start_time

    # Should take ~100ms (parallel), not 500ms (sequential)
    assert elapsed_time < 0.3, f"Parallel execution too slow: {elapsed_time:.2f}s"
    assert len(result.agent_responses) == 5


@pytest.mark.asyncio
async def test_timeout_handling(multiagent_rag, monkeypatch):
    """Test that slow agents are killed after timeout."""
    call_count = 0

    async def mock_query_slow(self, question, mode, max_sources, use_cache):
        nonlocal call_count
        call_count += 1

        # First 2 agents timeout, others succeed
        if call_count <= 2:
            await asyncio.sleep(10)  # Simulate slow agent
        else:
            await asyncio.sleep(0.1)

        return RAGResult(
            response="Answer",
            sources=["source1"],
            confidence=0.9,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query_slow)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Query with short timeout
    multiagent_rag.agent_timeout = 1.0
    result = await multiagent_rag.query_multiagent("What is X?")

    # Check that 2 agents timed out
    failed_agents = [r for r in result.agent_responses if r.error is not None]
    successful_agents = [r for r in result.agent_responses if r.error is None]

    assert len(failed_agents) == 2
    assert all("Timeout" in r.error for r in failed_agents)
    assert len(successful_agents) == 3


@pytest.mark.asyncio
async def test_partial_failure_handling(multiagent_rag, monkeypatch):
    """Test that consensus works with partial agent failures."""
    call_count = 0

    async def mock_query_partial_fail(self, question, mode, max_sources, use_cache):
        nonlocal call_count
        call_count += 1

        # First 2 agents fail, others succeed
        if call_count <= 2:
            raise ValueError("Agent failed")

        return RAGResult(
            response="Thompson Sampling is a Bayesian strategy",
            sources=["source1"],
            confidence=0.9,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query_partial_fail)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Query
    result = await multiagent_rag.query_multiagent("What is Thompson Sampling?")

    # Check that consensus was reached despite failures
    assert result.response  # Got a response
    assert result.confidence > 0.0
    assert result.consensus_metadata['n_failed'] == 2
    assert result.consensus_metadata['n_successful'] == 3


@pytest.mark.asyncio
async def test_all_agents_fail(multiagent_rag, monkeypatch):
    """Test handling when all agents fail."""
    async def mock_query_fail(self, question, mode, max_sources, use_cache):
        raise ValueError("All agents fail")

    monkeypatch.setattr(SimpleRAG, "query", mock_query_fail)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Query
    result = await multiagent_rag.query_multiagent("What is X?")

    # Check error handling
    assert "Error" in result.response
    assert result.confidence == 0.0
    assert result.agreement_score == 0.0
    assert result.consensus_metadata['n_successful'] == 0


# ============================================================================
# Consensus Mechanism Tests
# ============================================================================

def test_consensus_majority_vote(multiagent_rag, sample_agent_responses):
    """Test majority vote consensus."""
    # Make 3 agents agree on same response
    sample_agent_responses[0].response = "Answer A"
    sample_agent_responses[1].response = "Answer A"
    sample_agent_responses[2].response = "Answer A"
    sample_agent_responses[3].response = "Answer B"
    sample_agent_responses[4].response = "Answer C"

    response, confidence = multiagent_rag._consensus_majority_vote(sample_agent_responses)

    assert response == "Answer A"
    assert confidence > 0.8  # Average confidence of matching agents


def test_consensus_confidence_weighted(multiagent_rag, sample_agent_responses):
    """Test confidence-weighted consensus."""
    # Agent 2 has highest confidence (0.92)
    response, confidence = multiagent_rag._consensus_confidence_weighted(sample_agent_responses)

    assert response == sample_agent_responses[2].response
    assert confidence > 0.85  # Weighted average should be high


def test_consensus_ensemble(multiagent_rag, sample_agent_responses):
    """Test ensemble consensus (combines top responses)."""
    response, confidence = multiagent_rag._consensus_ensemble(sample_agent_responses)

    # Should combine top 3 responses
    assert "agent_0" in response or "agent_1" in response or "agent_2" in response
    assert confidence > 0.85


def test_consensus_single_agent(multiagent_rag):
    """Test consensus with only one agent."""
    single_response = [
        AgentResponse(
            agent_id="agent_0",
            strategy="direct",
            response="Single agent answer",
            confidence=0.9,
            sources=["source1"],
            latency_ms=100.0,
        )
    ]

    # All consensus methods should return the single response
    for method in [ConsensusMethod.MAJORITY_VOTE, ConsensusMethod.CONFIDENCE_WEIGHTED, ConsensusMethod.ENSEMBLE]:
        response, confidence = multiagent_rag._compute_consensus(single_response, method)
        assert response == "Single agent answer"
        assert confidence == 0.9


# ============================================================================
# Agreement Scoring Tests
# ============================================================================

def test_agreement_score_high(multiagent_rag, sample_agent_responses):
    """Test agreement score with high agreement."""
    # Make all agents give similar responses
    for i, resp in enumerate(sample_agent_responses):
        resp.response = "Thompson Sampling is a Bayesian strategy"
        resp.confidence = 0.9
        resp.sources = ["source1", "source2"]

    score = multiagent_rag._compute_agreement_score(sample_agent_responses)
    assert score > 0.8, f"Expected high agreement, got {score:.2f}"


def test_agreement_score_low(multiagent_rag, sample_agent_responses):
    """Test agreement score with low agreement."""
    # Make agents give very different responses
    sample_agent_responses[0].response = "Answer about machine learning"
    sample_agent_responses[1].response = "Answer about quantum physics"
    sample_agent_responses[2].response = "Answer about biology"
    sample_agent_responses[3].response = "Answer about economics"
    sample_agent_responses[4].response = "Answer about philosophy"

    score = multiagent_rag._compute_agreement_score(sample_agent_responses)
    assert score < 0.5, f"Expected low agreement, got {score:.2f}"


def test_agreement_score_single_agent(multiagent_rag):
    """Test agreement score with single agent (perfect agreement)."""
    single_response = [
        AgentResponse(
            agent_id="agent_0",
            strategy="direct",
            response="Single agent answer",
            confidence=0.9,
            sources=["source1"],
            latency_ms=100.0,
        )
    ]

    score = multiagent_rag._compute_agreement_score(single_response)
    assert score == 1.0, "Single agent should have perfect agreement"


# ============================================================================
# Disagreement Detection Tests
# ============================================================================

def test_disagreement_detection_low_agreement(multiagent_rag, sample_agent_responses):
    """Test disagreement detection with low agreement."""
    # Make agents disagree
    sample_agent_responses[0].response = "Answer A"
    sample_agent_responses[1].response = "Answer B"
    sample_agent_responses[2].response = "Answer C"
    sample_agent_responses[3].response = "Answer D"
    sample_agent_responses[4].response = "Answer E"

    agreement_score = 0.3  # Low agreement
    disagreements = multiagent_rag._detect_disagreements(sample_agent_responses, agreement_score)

    assert disagreements is not None
    assert any("Low agreement" in d for d in disagreements)


def test_disagreement_detection_high_variance(multiagent_rag, sample_agent_responses):
    """Test disagreement detection with high confidence variance."""
    # Make confidences vary widely
    sample_agent_responses[0].confidence = 0.9
    sample_agent_responses[1].confidence = 0.5
    sample_agent_responses[2].confidence = 0.8
    sample_agent_responses[3].confidence = 0.3
    sample_agent_responses[4].confidence = 0.7

    agreement_score = 0.7
    disagreements = multiagent_rag._detect_disagreements(sample_agent_responses, agreement_score)

    assert disagreements is not None
    assert any("variance" in d for d in disagreements)


def test_disagreement_detection_outlier(multiagent_rag, sample_agent_responses):
    """Test disagreement detection with outlier agent."""
    # Make one agent an outlier
    sample_agent_responses[0].confidence = 0.9
    sample_agent_responses[1].confidence = 0.85
    sample_agent_responses[2].confidence = 0.88
    sample_agent_responses[3].confidence = 0.2  # Outlier
    sample_agent_responses[4].confidence = 0.87

    agreement_score = 0.7
    disagreements = multiagent_rag._detect_disagreements(sample_agent_responses, agreement_score)

    assert disagreements is not None
    assert any("outlier" in d for d in disagreements)


# ============================================================================
# Source Deduplication Tests
# ============================================================================

@pytest.mark.asyncio
async def test_source_deduplication(multiagent_rag, monkeypatch):
    """Test that sources are deduplicated across agents."""
    async def mock_query(self, question, mode, max_sources, use_cache):
        # All agents return overlapping sources
        return RAGResult(
            response="Answer",
            sources=["source1", "source2", "source3"],
            confidence=0.9,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Query
    result = await multiagent_rag.query_multiagent("What is X?")

    # Check that sources are deduplicated
    assert len(result.sources) == 3  # Not 15 (5 agents × 3 sources)
    assert len(set(result.sources)) == 3  # All unique


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_parallelization_speedup(multiagent_rag, monkeypatch):
    """Test that parallelization provides speedup over sequential."""
    async def mock_query(self, question, mode, max_sources, use_cache):
        await asyncio.sleep(0.1)  # 100ms per agent
        return RAGResult(
            response="Answer",
            sources=["source1"],
            confidence=0.9,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Measure parallel time (5 agents)
    start_time = time.time()
    result = await multiagent_rag.query_multiagent("What is X?")
    parallel_time = time.time() - start_time

    # Parallel should be ~100ms (max latency), not 500ms (sum)
    # Allow 3× overhead for framework
    assert parallel_time < 0.4, f"Parallel too slow: {parallel_time:.2f}s"

    # Sequential time would be ~500ms (5 × 100ms)
    sequential_time = 0.5
    speedup = sequential_time / parallel_time
    assert speedup > 1.5, f"Insufficient speedup: {speedup:.1f}×"


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_multiagent_workflow(mock_config, monkeypatch):
    """Test complete multi-agent workflow end-to-end."""
    # Mock query to return realistic results
    async def mock_query(self, question, mode, max_sources, use_cache):
        await asyncio.sleep(0.05)  # Simulate work
        return RAGResult(
            response=f"Thompson Sampling is a Bayesian exploration strategy (mode={mode})",
            sources=[f"source_{mode}_1", f"source_{mode}_2"],
            confidence=0.85 + (hash(mode) % 10) / 100,  # Vary slightly by mode
            reasoning_mode=mode,
            metadata={'mode': mode},
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    # Create multi-agent RAG
    async with MultiAgentRAG(
        config=mock_config,
        n_agents=5,
        consensus_method="confidence_weighted",
    ) as rag:
        # Ingest
        await rag.ingest("Thompson Sampling uses Bayesian statistics")

        # Query
        result = await rag.query_multiagent(
            "What is Thompson Sampling?",
            explain_disagreement=True
        )

        # Validate result
        assert result.response
        assert result.confidence > 0.0
        assert len(result.sources) > 0
        assert len(result.agent_responses) == 5
        assert result.agreement_score >= 0.0
        assert result.consensus_metadata['n_successful'] >= 3

        # Check metrics
        metrics = rag.get_metrics()
        assert metrics['n_agents'] == 5
        assert metrics['total_queries'] == 1
        assert metrics['total_agents_succeeded'] >= 3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_consensus_method_comparison(mock_config, monkeypatch):
    """Test different consensus methods produce valid results."""
    async def mock_query(self, question, mode, max_sources, use_cache):
        return RAGResult(
            response=f"Answer from mode {mode}",
            sources=["source1", "source2"],
            confidence=0.85,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    # Test each consensus method
    for method in ["majority_vote", "confidence_weighted", "ensemble"]:
        async with MultiAgentRAG(
            config=mock_config,
            n_agents=5,
            consensus_method=method,
        ) as rag:
            await rag.ingest("Test content")

            result = await rag.query_multiagent("What is X?")

            assert result.response
            assert result.confidence > 0.0
            assert result.consensus_method == method
            assert len(result.agent_responses) == 5


# ============================================================================
# Metrics Tests
# ============================================================================

@pytest.mark.asyncio
async def test_metrics_tracking(multiagent_rag, monkeypatch):
    """Test that metrics are tracked correctly."""
    async def mock_query(self, question, mode, max_sources, use_cache):
        return RAGResult(
            response="Answer",
            sources=["source1"],
            confidence=0.9,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Run multiple queries
    for i in range(3):
        await multiagent_rag.query_multiagent(f"Question {i}")

    # Check metrics
    metrics = multiagent_rag.get_metrics()
    assert metrics['total_queries'] == 3
    assert metrics['total_agents_succeeded'] == 15  # 3 queries × 5 agents
    assert metrics['avg_agreement_score'] > 0.0


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_response_handling(multiagent_rag, monkeypatch):
    """Test handling of empty responses from agents."""
    async def mock_query(self, question, mode, max_sources, use_cache):
        return RAGResult(
            response="",  # Empty response
            sources=[],
            confidence=0.1,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    # Ingest data
    await multiagent_rag.ingest("Test content")

    # Query
    result = await multiagent_rag.query_multiagent("What is X?")

    # Should still produce a result
    assert result.response  # Consensus should handle empty responses
    assert result.confidence >= 0.0


@pytest.mark.asyncio
async def test_single_agent_mode(mock_config, monkeypatch):
    """Test MultiAgentRAG with only 1 agent (edge case)."""
    async def mock_query(self, question, mode, max_sources, use_cache):
        return RAGResult(
            response="Single agent answer",
            sources=["source1"],
            confidence=0.9,
            reasoning_mode=mode,
        )

    monkeypatch.setattr(SimpleRAG, "query", mock_query)

    async with MultiAgentRAG(
        config=mock_config,
        n_agents=1,
    ) as rag:
        await rag.ingest("Test content")

        result = await rag.query_multiagent("What is X?")

        assert result.response == "Single agent answer"
        assert result.confidence == 0.9
        assert result.agreement_score == 1.0  # Perfect agreement with self


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
