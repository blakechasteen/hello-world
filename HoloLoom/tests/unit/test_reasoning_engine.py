#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Reasoning Engine
================================
Tests for Layer 6 reasoning components.

Author: Claude Code
Date: 2025-11-15
Phase: 1 (Foundation)
"""

import pytest
import asyncio
from typing import List

from HoloLoom.documentation.types import Query, Features, Context, MemoryShard
from HoloLoom.reasoning import (
    ReasoningEngine,
    ReasoningMode,
    StepType,
    QueryType,
    QueryPlanner,
    ChainOfThought,
    SelfVerifier,
    reason_with_mode,
    auto_reason,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_query():
    """Simple factual query."""
    return Query(text="What is Thompson Sampling?")


@pytest.fixture
def complex_query():
    """Complex analytical query."""
    return Query(
        text="Explain the relationship between Thompson Sampling and "
             "the exploration-exploitation tradeoff in multi-armed bandits"
    )


@pytest.fixture
def comparative_query():
    """Comparative query."""
    return Query(text="Compare Thompson Sampling and UCB algorithms")


@pytest.fixture
def simple_features():
    """Simple feature set."""
    return Features(
        motifs=["thompson", "sampling"],
        embeddings=[[0.1] * 96],  # Single scale
        spectral=None
    )


@pytest.fixture
def rich_features():
    """Rich feature set."""
    return Features(
        motifs=["thompson", "sampling", "exploration", "exploitation", "bandit"],
        embeddings=[[0.1] * 96, [0.2] * 192, [0.3] * 384],  # Multi-scale
        spectral={"eigenvalues": [0.5, 0.3, 0.2]}
    )


@pytest.fixture
def empty_context():
    """Empty context (no shards)."""
    return Context(shards=[], retrieved_at=None)


@pytest.fixture
def good_context():
    """Good quality context with relevant shards."""
    shards = [
        MemoryShard(
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                 "It uses probability matching to balance exploration and exploitation.",
            entities=["thompson_sampling", "bayesian", "multi_armed_bandit"],
            timestamp=1.0
        ),
        MemoryShard(
            text="The algorithm samples from the posterior distribution of each arm's reward. "
                 "Arms with higher expected rewards are selected more often.",
            entities=["posterior", "reward", "algorithm"],
            timestamp=1.0
        ),
        MemoryShard(
            text="Thompson Sampling naturally balances exploration of uncertain arms "
                 "with exploitation of arms known to be good.",
            entities=["exploration", "exploitation", "uncertainty"],
            timestamp=1.0
        ),
    ]
    return Context(shards=shards, retrieved_at=None)


# ============================================================================
# QueryPlanner Tests
# ============================================================================

def test_query_planner_factual():
    """Test query planner on factual query."""
    planner = QueryPlanner()
    query = Query(text="What is Thompson Sampling?")
    features = Features(motifs=["thompson", "sampling"], embeddings=[[0.1] * 96], spectral=None)

    intent = planner.analyze_intent(query, features)

    assert intent.type == QueryType.FACTUAL
    assert "definition" in intent.requirements or "basic facts" in intent.requirements
    assert 0.0 <= intent.complexity <= 1.0
    assert 0.0 <= intent.confidence <= 1.0
    assert "thompson" in [k.lower() for k in intent.key_concepts]


def test_query_planner_comparative():
    """Test query planner on comparative query."""
    planner = QueryPlanner()
    query = Query(text="Compare Thompson Sampling and UCB")
    features = Features(motifs=["thompson", "ucb", "compare"], embeddings=[[0.1] * 96], spectral=None)

    intent = planner.analyze_intent(query, features)

    assert intent.type == QueryType.COMPARATIVE
    assert "multi-source evidence" in intent.requirements
    assert intent.complexity > 0.4  # Comparative queries are more complex


def test_query_planner_analytical():
    """Test query planner on analytical query."""
    planner = QueryPlanner()
    query = Query(text="Why does Thompson Sampling work well?")
    features = Features(motifs=["why", "thompson", "sampling"], embeddings=[[0.1] * 96], spectral=None)

    intent = planner.analyze_intent(query, features)

    assert intent.type == QueryType.ANALYTICAL
    assert "causal reasoning" in intent.requirements or "supporting evidence" in intent.requirements


def test_query_planner_complexity(simple_features, rich_features):
    """Test complexity estimation."""
    planner = QueryPlanner()

    # Simple query, simple features
    simple_intent = planner.analyze_intent(
        Query(text="What is X?"),
        simple_features
    )

    # Complex query, rich features
    complex_intent = planner.analyze_intent(
        Query(text="Explain the intricate relationship between multiple complex systems"),
        rich_features
    )

    assert complex_intent.complexity > simple_intent.complexity


# ============================================================================
# ChainOfThought Tests
# ============================================================================

def test_cot_evidence_extraction(good_context):
    """Test evidence extraction from context."""
    cot = ChainOfThought()
    features = Features(motifs=["thompson", "sampling"], embeddings=[[0.1] * 96], spectral=None)

    planner = QueryPlanner()
    intent = planner.analyze_intent(Query(text="What is Thompson Sampling?"), features)

    evidence = cot.extract_evidence(good_context, intent)

    assert len(evidence) > 0
    assert all(0.0 <= e.relevance <= 1.0 for e in evidence)
    assert all(e.text for e in evidence)


def test_cot_synthesis():
    """Test synthesis from evidence."""
    cot = ChainOfThought()

    from HoloLoom.reasoning.types import EvidencePiece, QueryIntent
    evidence = [
        EvidencePiece(text="Thompson Sampling uses Bayesian methods", source="s1", relevance=0.9),
        EvidencePiece(text="It balances exploration and exploitation", source="s2", relevance=0.8),
    ]

    intent = QueryIntent(
        type=QueryType.FACTUAL,
        requirements=["definition"],
        complexity=0.3,
        confidence=0.8,
        key_concepts=["thompson", "sampling"]
    )

    synthesis = cot.synthesize(intent, evidence)

    assert synthesis.reasoning
    assert len(synthesis.key_points) > 0
    assert 0.0 <= synthesis.confidence <= 1.0
    assert synthesis.evidence_used == len(evidence)


def test_cot_generate_standard_chain(simple_query, rich_features, good_context):
    """Test standard chain generation."""
    cot = ChainOfThought()
    planner = QueryPlanner()

    intent = planner.analyze_intent(simple_query, rich_features)
    chain = cot.generate_standard_chain(simple_query, intent, good_context)

    assert len(chain) >= 3  # STANDARD mode: 3-5 steps
    assert any(s.step_type == StepType.UNDERSTANDING for s in chain)
    assert any(s.step_type == StepType.SYNTHESIS for s in chain)
    assert all(0.0 <= s.confidence <= 1.0 for s in chain)


def test_cot_confidence_estimation(good_context, empty_context, simple_features):
    """Test confidence estimation."""
    cot = ChainOfThought()

    # Good context should have higher confidence
    good_conf = cot.estimate_confidence(simple_features, good_context)

    # Empty context should have lower confidence
    empty_conf = cot.estimate_confidence(simple_features, empty_context)

    assert 0.0 <= good_conf <= 1.0
    assert 0.0 <= empty_conf <= 1.0
    assert good_conf > empty_conf


# ============================================================================
# SelfVerifier Tests
# ============================================================================

@pytest.mark.asyncio
async def test_verifier_empty_chain(good_context):
    """Test verifier on empty chain."""
    verifier = SelfVerifier()
    chain = []

    result = await verifier.verify(chain, good_context)

    assert not result.passed
    assert "empty" in result.issue.lower()


@pytest.mark.asyncio
async def test_verifier_good_chain(good_context):
    """Test verifier on good quality chain."""
    from HoloLoom.reasoning.types import ReasoningStep

    verifier = SelfVerifier()
    chain = [
        ReasoningStep(
            thought="Understanding query",
            evidence="Detected Thompson Sampling topic",
            confidence=0.9,
            step_type=StepType.UNDERSTANDING
        ),
        ReasoningStep(
            thought="Evidence found",
            evidence="3 relevant sources",
            confidence=0.85,
            step_type=StepType.EVIDENCE
        ),
        ReasoningStep(
            thought="Synthesis complete",
            evidence="Combined evidence into answer",
            confidence=0.88,
            step_type=StepType.SYNTHESIS
        ),
    ]

    result = await verifier.verify(chain, good_context)

    assert result.passed
    assert result.confidence > 0.7


@pytest.mark.asyncio
async def test_verifier_confidence_degradation(good_context):
    """Test verifier detects confidence degradation."""
    from HoloLoom.reasoning.types import ReasoningStep

    verifier = SelfVerifier()
    chain = [
        ReasoningStep(
            thought="High confidence start",
            evidence="Good evidence",
            confidence=0.95,
            step_type=StepType.UNDERSTANDING
        ),
        ReasoningStep(
            thought="Sudden drop",
            evidence="Lost confidence",
            confidence=0.40,  # Big drop!
            step_type=StepType.SYNTHESIS
        ),
    ]

    result = await verifier.verify(chain, good_context)

    assert not result.passed
    assert "confidence drop" in result.issue.lower()


@pytest.mark.asyncio
async def test_verifier_multipass(good_context):
    """Test multi-pass verification."""
    from HoloLoom.reasoning.types import ReasoningStep

    verifier = SelfVerifier()
    chain = [
        ReasoningStep("Understanding", "evidence", 0.9, StepType.UNDERSTANDING),
        ReasoningStep("Evidence", "sources", 0.85, StepType.EVIDENCE),
        ReasoningStep("Synthesis", "combined", 0.88, StepType.SYNTHESIS),
    ]

    result = await verifier.verify_multipass(chain, good_context, num_passes=3)

    assert len(result.passes) == 3
    assert 0.0 <= result.overall_confidence <= 1.0


# ============================================================================
# ReasoningEngine Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_engine_fast_mode(simple_query, simple_features, good_context):
    """Test FAST mode reasoning."""
    engine = ReasoningEngine(mode=ReasoningMode.FAST)

    result = await engine.reason(simple_query, simple_features, good_context)

    assert result.mode == ReasoningMode.FAST
    assert len(result.chain) >= 1
    assert result.duration_ms > 0
    assert 0.0 <= result.total_confidence <= 1.0


@pytest.mark.asyncio
async def test_reasoning_engine_standard_mode(simple_query, rich_features, good_context):
    """Test STANDARD mode reasoning."""
    engine = ReasoningEngine(mode=ReasoningMode.STANDARD)

    result = await engine.reason(simple_query, rich_features, good_context)

    assert result.mode == ReasoningMode.STANDARD
    assert len(result.chain) >= 3  # Multi-step chain
    assert len(result.chain) <= 5  # Max steps
    assert result.duration_ms > 0
    assert "intent" in result.metadata


@pytest.mark.asyncio
async def test_reasoning_engine_deep_mode_fallback(simple_query, rich_features, good_context):
    """Test DEEP mode falls back to STANDARD (Phase 1)."""
    engine = ReasoningEngine(mode=ReasoningMode.DEEP)

    result = await engine.reason(simple_query, rich_features, good_context)

    # Should fall back to STANDARD in Phase 1
    assert result.mode == ReasoningMode.STANDARD


@pytest.mark.asyncio
async def test_reasoning_engine_mode_selection(simple_features, rich_features, good_context, empty_context):
    """Test automatic mode selection."""
    engine = ReasoningEngine()

    # Simple query, good context → FAST
    simple_mode = engine.select_mode(
        Query(text="What is X?"),
        simple_features,
        good_context
    )

    # Complex query, empty context → DEEP
    complex_mode = engine.select_mode(
        Query(text="Explain the intricate philosophical implications of X"),
        rich_features,
        empty_context
    )

    # Fast should be faster than deep
    assert simple_mode == ReasoningMode.FAST or simple_mode == ReasoningMode.STANDARD
    assert complex_mode == ReasoningMode.DEEP or complex_mode == ReasoningMode.STANDARD


@pytest.mark.asyncio
async def test_reasoning_engine_escalation(simple_query, simple_features, empty_context):
    """Test escalation from FAST to STANDARD on low confidence."""
    engine = ReasoningEngine(mode=ReasoningMode.FAST)

    result = await engine.reason(simple_query, simple_features, empty_context)

    # Should detect low confidence
    if result.total_confidence < 0.7:
        assert result.metadata.get("escalation_recommended", False)


@pytest.mark.asyncio
async def test_convenience_reason_with_mode(simple_query, simple_features, good_context):
    """Test convenience function for one-off reasoning."""
    result = await reason_with_mode(
        simple_query,
        simple_features,
        good_context,
        mode=ReasoningMode.STANDARD
    )

    assert result.mode == ReasoningMode.STANDARD
    assert len(result.chain) >= 3


@pytest.mark.asyncio
async def test_convenience_auto_reason(simple_query, rich_features, good_context):
    """Test auto-reasoning with mode selection."""
    result = await auto_reason(
        simple_query,
        rich_features,
        good_context
    )

    assert result.mode in [ReasoningMode.FAST, ReasoningMode.STANDARD, ReasoningMode.DEEP]
    assert len(result.chain) > 0


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_fast_mode_performance(simple_query, simple_features, good_context):
    """FAST mode should complete in <50ms."""
    engine = ReasoningEngine(mode=ReasoningMode.FAST)

    result = await engine.reason(simple_query, simple_features, good_context)

    assert result.duration_ms < 50, f"FAST mode took {result.duration_ms}ms (expected <50ms)"


@pytest.mark.asyncio
async def test_standard_mode_performance(simple_query, rich_features, good_context):
    """STANDARD mode should complete in <200ms."""
    engine = ReasoningEngine(mode=ReasoningMode.STANDARD)

    result = await engine.reason(simple_query, rich_features, good_context)

    assert result.duration_ms < 200, f"STANDARD mode took {result.duration_ms}ms (expected <200ms)"


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_fast(simple_query, simple_features, good_context):
    """Test full pipeline with FAST mode."""
    engine = ReasoningEngine(mode=ReasoningMode.FAST)

    result = await engine.reason(simple_query, simple_features, good_context)

    # Validate result structure
    assert isinstance(result.chain, list)
    assert all(hasattr(s, 'thought') for s in result.chain)
    assert all(hasattr(s, 'confidence') for s in result.chain)
    assert result.mode == ReasoningMode.FAST
    assert result.total_confidence > 0


@pytest.mark.asyncio
async def test_full_pipeline_standard(complex_query, rich_features, good_context):
    """Test full pipeline with STANDARD mode."""
    engine = ReasoningEngine(mode=ReasoningMode.STANDARD)

    result = await engine.reason(complex_query, rich_features, good_context)

    # Validate result structure
    assert len(result.chain) >= 3
    assert any(s.step_type == StepType.UNDERSTANDING for s in result.chain)
    assert any(s.step_type == StepType.SYNTHESIS for s in result.chain)
    assert result.mode == ReasoningMode.STANDARD
    assert "verification_passed" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
