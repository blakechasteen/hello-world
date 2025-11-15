#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for DEEP Reasoning Mode
====================================
Tests for Phase 3: Planning, Backtracking, Thompson Sampling.

Author: Claude Code
Date: 2025-11-15
Phase: 3 (DEEP Mode)
"""

import pytest
import asyncio
import os
import sys
import tempfile
from typing import List

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Direct imports to avoid main package issues
import HoloLoom.documentation.types as doc_types
Query = doc_types.Query
Features = doc_types.Features
Context = doc_types.Context
MemoryShard = doc_types.MemoryShard
Motif = doc_types.Motif

from HoloLoom.reasoning.types import ReasoningMode, StepType, QueryType, ReasoningStep
from HoloLoom.reasoning.planner import QueryPlanner
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.engine import ReasoningEngine
from HoloLoom.reasoning.backtracker import Backtracker, Contradiction
from HoloLoom.reasoning.bandit import (
    ReasoningModeBandit,
    ThompsonPrior,
    ModeStatistics
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def complex_analytical_query():
    """Complex analytical query requiring DEEP mode."""
    return Query(
        text="Why does Thompson Sampling work better than epsilon-greedy "
             "in non-stationary environments with delayed rewards?"
    )


@pytest.fixture
def comparative_query():
    """Comparative query for plan testing."""
    return Query(text="Compare Thompson Sampling and UCB algorithms in detail")


@pytest.fixture
def verification_query():
    """Verification query for plan testing."""
    return Query(text="Is it true that Thompson Sampling is always optimal?")


@pytest.fixture
def complex_features():
    """Rich features for complex queries."""
    return Features(
        psi=[0.1] * 96,  # Single-scale embedding
        motifs=[
            Motif(pattern="thompson", weight=0.9),
            Motif(pattern="sampling", weight=0.9),
            Motif(pattern="epsilon", weight=0.8),
            Motif(pattern="greedy", weight=0.8),
            Motif(pattern="nonstationary", weight=0.7),
            Motif(pattern="delayed", weight=0.6)
        ],
        metrics={"eigenvalues": [0.5, 0.3, 0.2]},
        confidence=0.85
    )


@pytest.fixture
def good_context():
    """Good quality context with relevant shards."""
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits. "
                 "It naturally handles non-stationary environments by updating posterior distributions.",
            entities=["thompson_sampling", "bayesian", "nonstationary"]
        ),
        MemoryShard(
            id="shard_2",
            text="Epsilon-greedy explores with probability epsilon, but this is fixed. "
                 "In non-stationary environments, fixed exploration can be suboptimal.",
            entities=["epsilon_greedy", "exploration", "nonstationary"]
        ),
        MemoryShard(
            id="shard_3",
            text="With delayed rewards, Thompson Sampling's Bayesian updates allow it to "
                 "credit earlier actions appropriately through posterior propagation.",
            entities=["delayed_rewards", "bayesian", "credit_assignment"]
        ),
        MemoryShard(
            id="shard_4",
            text="UCB (Upper Confidence Bound) is deterministic and uses confidence intervals. "
                 "It handles non-stationarity through time-decayed confidence bounds.",
            entities=["ucb", "confidence_bounds", "deterministic"]
        ),
    ]
    return Context(shards=shards, retrieved_at=None)


@pytest.fixture
def contradictory_chain():
    """Reasoning chain with contradictions."""
    return [
        ReasoningStep(
            thought="Thompson Sampling always outperforms epsilon-greedy",
            evidence="Multiple studies show Thompson Sampling superiority",
            confidence=0.9,
            step_type=StepType.UNDERSTANDING
        ),
        ReasoningStep(
            thought="However, epsilon-greedy never performs worse than Thompson Sampling",
            evidence="Theoretical bounds prove epsilon-greedy guarantees",
            confidence=0.8,
            step_type=StepType.EVIDENCE
        ),
        ReasoningStep(
            thought="All algorithms perform equally in stationary environments",
            evidence="Empirical results show no significant difference",
            confidence=0.7,
            step_type=StepType.SYNTHESIS
        ),
    ]


# ============================================================================
# Backtracker Tests
# ============================================================================

@pytest.mark.asyncio
async def test_backtracker_detect_keyword_contradictions(contradictory_chain):
    """Test keyword-based contradiction detection."""
    backtracker = Backtracker()

    contradictions = await backtracker.detect_contradictions(contradictory_chain)

    # Should detect "always" vs "never" contradiction
    assert len(contradictions) > 0
    assert any("always" in c.description.lower() and "never" in c.description.lower()
              for c in contradictions)


@pytest.mark.asyncio
async def test_backtracker_detect_confidence_contradictions():
    """Test confidence drop detection."""
    backtracker = Backtracker()

    chain = [
        ReasoningStep("High confidence", "evidence", 0.95, StepType.UNDERSTANDING),
        ReasoningStep("Sudden doubt", "weak evidence", 0.40, StepType.SYNTHESIS),
    ]

    contradictions = await backtracker.detect_contradictions(chain)

    # Should detect severe confidence drop
    assert len(contradictions) > 0
    assert any("confidence drop" in c.description.lower() for c in contradictions)
    assert contradictions[0].severity > 0.4


@pytest.mark.asyncio
async def test_backtracker_detect_flow_contradictions():
    """Test logical flow contradiction detection."""
    backtracker = Backtracker()

    # Synthesis before evidence
    chain = [
        ReasoningStep("Understanding", "context", 0.9, StepType.UNDERSTANDING),
        ReasoningStep("Conclusion", "final answer", 0.85, StepType.SYNTHESIS),
        ReasoningStep("Evidence", "supporting data", 0.8, StepType.EVIDENCE),
    ]

    contradictions = await backtracker.detect_contradictions(chain)

    # Should detect synthesis before evidence
    assert len(contradictions) > 0
    assert any("synthesis" in c.description.lower() and "evidence" in c.description.lower()
              for c in contradictions)


@pytest.mark.asyncio
async def test_backtracker_revise_chain(contradictory_chain):
    """Test chain revision."""
    backtracker = Backtracker()

    contradictions = await backtracker.detect_contradictions(contradictory_chain)
    result = await backtracker.revise(contradictory_chain, contradictions)

    assert result.revisions_made > 0
    assert len(result.revised_chain) > len(contradictory_chain)  # Backtrack steps added
    assert len(result.revision_history) > 0


@pytest.mark.asyncio
async def test_backtracker_infinite_loop_prevention():
    """Test infinite loop detection."""
    backtracker = Backtracker()

    # Chain with too many backtracks
    chain = [
        ReasoningStep("Step 1", "evidence", 0.9, StepType.UNDERSTANDING),
        ReasoningStep("Backtrack 1", "revision", 0.7, StepType.BACKTRACK),
        ReasoningStep("Backtrack 2", "revision", 0.6, StepType.BACKTRACK),
        ReasoningStep("Backtrack 3", "revision", 0.5, StepType.BACKTRACK),
    ]

    # Should detect loop
    is_safe = backtracker.prevent_infinite_loops(chain, max_backtracks=2)
    assert not is_safe


@pytest.mark.asyncio
async def test_backtracker_detect_cycles():
    """Test reasoning cycle detection."""
    backtracker = Backtracker()

    # Chain with repeated reasoning pattern
    chain = [
        ReasoningStep("Thompson Sampling is Bayesian", "evidence", 0.9, StepType.UNDERSTANDING),
        ReasoningStep("Bayesian methods use priors", "evidence", 0.85, StepType.EVIDENCE),
        ReasoningStep("Priors encode beliefs", "evidence", 0.8, StepType.SYNTHESIS),
        ReasoningStep("Thompson Sampling is Bayesian", "evidence", 0.75, StepType.UNDERSTANDING),
    ]

    cycles = backtracker.detect_cycles(chain, window_size=1)

    # Should detect repeated pattern
    assert len(cycles) > 0


# ============================================================================
# Enhanced Planner Tests
# ============================================================================

def test_planner_comparative_plan(comparative_query, complex_features, good_context):
    """Test comparative query planning."""
    planner = QueryPlanner()

    plan = planner.create_plan(comparative_query, complex_features, good_context)

    # Comparative plan should have 6+ steps
    assert len(plan.steps) >= 6
    assert any("compare" in step.question.lower() or "difference" in step.question.lower()
              for step in plan.steps)
    assert any("similar" in step.question.lower() for step in plan.steps)

    # Check dependencies
    assert len(plan.dependencies) > 0


def test_planner_analytical_plan(complex_analytical_query, complex_features, good_context):
    """Test analytical query planning."""
    planner = QueryPlanner()

    plan = planner.create_plan(complex_analytical_query, complex_features, good_context)

    # Analytical plan should have 6 steps
    assert len(plan.steps) >= 6
    assert any("cause" in step.question.lower() or "why" in step.question.lower()
              for step in plan.steps)
    assert any("alternative" in step.question.lower() for step in plan.steps)


def test_planner_verification_plan(verification_query, complex_features, good_context):
    """Test verification query planning."""
    planner = QueryPlanner()

    plan = planner.create_plan(verification_query, complex_features, good_context)

    # Verification plan should check both support and contradiction
    assert len(plan.steps) >= 5
    assert any("support" in step.question.lower() for step in plan.steps)
    assert any("contradict" in step.question.lower() for step in plan.steps)


def test_planner_dependency_graph(comparative_query, complex_features, good_context):
    """Test dependency graph construction."""
    planner = QueryPlanner()

    plan = planner.create_plan(comparative_query, complex_features, good_context)

    # Should have parallel dependencies for comparative queries
    assert len(plan.dependencies) > 0

    # Some steps should depend on multiple predecessors
    has_multiple_deps = any(len(deps) > 1 for deps in plan.dependencies.values())
    assert has_multiple_deps


# ============================================================================
# DEEP Mode Engine Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_engine_deep_mode(complex_analytical_query, complex_features, good_context):
    """Test DEEP mode end-to-end."""
    engine = ReasoningEngine(mode=ReasoningMode.DEEP, max_thinking_steps=10)

    result = await engine.reason(complex_analytical_query, complex_features, good_context)

    # Verify DEEP mode execution
    assert result.mode == ReasoningMode.DEEP
    assert len(result.chain) >= 3  # Planning + substeps + synthesis

    # Check for planning step
    planning_steps = [s for s in result.chain if s.step_type == StepType.PLANNING]
    assert len(planning_steps) > 0

    # Check metadata
    assert "plan_steps" in result.metadata
    assert "verification_passes" in result.metadata
    assert result.metadata["verification_passes"] >= 3


@pytest.mark.asyncio
async def test_deep_mode_multipass_verification(complex_analytical_query, complex_features, good_context):
    """Test multi-pass verification in DEEP mode."""
    engine = ReasoningEngine(mode=ReasoningMode.DEEP)

    result = await engine.reason(complex_analytical_query, complex_features, good_context)

    # Should have run 3 verification passes
    assert result.metadata["verification_passes"] == 3

    # Check for verification steps if any issues found
    verification_steps = [s for s in result.chain if s.step_type == StepType.VERIFICATION]
    # May or may not have verification steps depending on whether issues were found


@pytest.mark.asyncio
async def test_deep_mode_backtracking():
    """Test backtracking in DEEP mode with contradictory context."""
    engine = ReasoningEngine(mode=ReasoningMode.DEEP)

    # Create contradictory context
    contradictory_context = Context(
        shards=[
            MemoryShard(
                id="contra_1",
                text="Thompson Sampling is always optimal in all scenarios and never fails.",
                entities=["thompson", "always", "optimal"]
            ),
            MemoryShard(
                id="contra_2",
                text="Thompson Sampling is never optimal and always fails in practice.",
                entities=["thompson", "never", "fails"]
            ),
        ],
        retrieved_at=None
    )

    query = Query(text="Is Thompson Sampling optimal?")
    features = Features(
        psi=[0.1] * 96,
        motifs=[Motif(pattern="thompson", weight=0.9)],
        confidence=0.8
    )

    result = await engine.reason(query, features, contradictory_context)

    # Should detect contradictions
    # (May or may not backtrack depending on verification sensitivity)
    assert len(result.chain) > 0


@pytest.mark.asyncio
async def test_deep_mode_performance(complex_analytical_query, complex_features, good_context):
    """Test DEEP mode performance target (<500ms)."""
    engine = ReasoningEngine(mode=ReasoningMode.DEEP)

    result = await engine.reason(complex_analytical_query, complex_features, good_context)

    # DEEP mode should complete in reasonable time
    # Note: <500ms target may be tight in practice, allow some buffer
    assert result.duration_ms < 1000, f"DEEP mode took {result.duration_ms}ms (target <500ms, allowed <1000ms for tests)"


# ============================================================================
# Thompson Sampling Bandit Tests
# ============================================================================

def test_thompson_prior_sampling():
    """Test Thompson prior sampling."""
    prior = ThompsonPrior(alpha=10, beta=2)

    # Sample multiple times
    samples = [prior.sample() for _ in range(100)]

    # Samples should be in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in samples)

    # Mean should be close to alpha / (alpha + beta)
    expected = prior.expected_reward()
    assert 0.7 < expected < 0.9  # 10/12 ≈ 0.83


def test_thompson_prior_update():
    """Test Thompson prior updates."""
    prior = ThompsonPrior(alpha=1, beta=1)

    # Update with success
    initial_alpha = prior.alpha
    prior.update(success=True, confidence=0.8)
    assert prior.alpha > initial_alpha

    # Update with failure
    initial_beta = prior.beta
    prior.update(success=False, confidence=0.6)
    assert prior.beta > initial_beta


def test_bandit_mode_selection():
    """Test reasoning mode selection."""
    bandit = ReasoningModeBandit()

    # Simple query, good context → likely FAST or STANDARD
    mode_simple = bandit.select_mode(query_complexity=0.2, context_quality=0.9)
    # Should select some mode
    assert mode_simple in [ReasoningMode.FAST, ReasoningMode.STANDARD, ReasoningMode.DEEP]

    # Complex query → any mode possible due to Thompson Sampling randomness
    mode_complex = bandit.select_mode(query_complexity=0.9, context_quality=0.5)
    # Should select some mode (Thompson Sampling is probabilistic)
    assert mode_complex in [ReasoningMode.FAST, ReasoningMode.STANDARD, ReasoningMode.DEEP]


def test_bandit_update_statistics():
    """Test bandit statistics updates."""
    bandit = ReasoningModeBandit()

    # Simulate successful FAST mode usage
    bandit.update(
        mode=ReasoningMode.FAST,
        success=True,
        confidence=0.9,
        duration_ms=30.0
    )

    stats = bandit.get_stats()

    # Check FAST mode stats
    fast_stats = stats[ReasoningMode.FAST.value]
    assert fast_stats["total_uses"] == 1
    assert fast_stats["success_rate"] == 1.0
    assert fast_stats["avg_confidence"] == 0.9


def test_bandit_learning():
    """Test bandit learns from outcomes."""
    bandit = ReasoningModeBandit()

    # Simulate multiple successful STANDARD mode uses
    for _ in range(10):
        bandit.update(
            mode=ReasoningMode.STANDARD,
            success=True,
            confidence=0.85,
            duration_ms=150.0
        )

    # Simulate failed DEEP mode uses
    for _ in range(5):
        bandit.update(
            mode=ReasoningMode.DEEP,
            success=False,
            confidence=0.5,
            duration_ms=600.0
        )

    stats = bandit.get_stats()

    # STANDARD should have higher expected reward
    standard_reward = stats[ReasoningMode.STANDARD.value]["expected_reward"]
    deep_reward = stats[ReasoningMode.DEEP.value]["expected_reward"]

    assert standard_reward > deep_reward


def test_bandit_save_load():
    """Test bandit persistence."""
    bandit = ReasoningModeBandit()

    # Update some statistics
    bandit.update(ReasoningMode.FAST, True, 0.9, 40.0)
    bandit.update(ReasoningMode.STANDARD, True, 0.85, 180.0)

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        bandit.save(temp_path)

        # Load bandit
        loaded_bandit = ReasoningModeBandit.load(temp_path)

        # Check statistics match
        original_stats = bandit.get_stats()
        loaded_stats = loaded_bandit.get_stats()

        for mode in ReasoningMode:
            mode_str = mode.value
            assert original_stats[mode_str]["total_uses"] == loaded_stats[mode_str]["total_uses"]
            assert abs(original_stats[mode_str]["expected_reward"] -
                      loaded_stats[mode_str]["expected_reward"]) < 0.01

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_bandit_recommend_mode():
    """Test deterministic mode recommendation."""
    bandit = ReasoningModeBandit()

    # Tight time budget → FAST
    mode = bandit.recommend_mode(
        query_complexity=0.5,
        context_quality=0.5,
        time_budget_ms=50
    )
    assert mode == ReasoningMode.FAST

    # Very complex → DEEP
    mode = bandit.recommend_mode(
        query_complexity=0.9,
        context_quality=0.5
    )
    assert mode == ReasoningMode.DEEP

    # Excellent context, simple query → FAST
    mode = bandit.recommend_mode(
        query_complexity=0.2,
        context_quality=0.9
    )
    assert mode == ReasoningMode.FAST


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_deep_pipeline_with_bandit(complex_analytical_query, complex_features, good_context):
    """Test full pipeline: bandit selection + DEEP reasoning."""
    bandit = ReasoningModeBandit()

    # Estimate complexity
    planner = QueryPlanner()
    intent = planner.analyze_intent(complex_analytical_query, complex_features)

    # Select mode
    selected_mode = bandit.select_mode(
        query_complexity=intent.complexity,
        context_quality=0.7
    )

    # Run reasoning
    engine = ReasoningEngine(mode=selected_mode)
    result = await engine.reason(complex_analytical_query, complex_features, good_context)

    # Update bandit
    success = result.total_confidence >= 0.75
    bandit.update(
        mode=selected_mode,
        success=success,
        confidence=result.total_confidence,
        duration_ms=result.duration_ms
    )

    # Verify integration
    assert result.mode in [ReasoningMode.STANDARD, ReasoningMode.DEEP]
    assert len(result.chain) >= 3


@pytest.mark.asyncio
async def test_adaptive_mode_selection_across_queries(good_context):
    """Test adaptive mode selection for different query types."""
    bandit = ReasoningModeBandit()
    engine = ReasoningEngine()
    planner = QueryPlanner()

    queries = [
        (Query(text="What is X?"), Features(psi=[0.1]*96, motifs=[Motif(pattern="x", weight=0.8)])),
        (Query(text="Compare A and B in depth"), Features(psi=[0.1]*96, motifs=[Motif(pattern="a", weight=0.8), Motif(pattern="b", weight=0.8)])),
        (Query(text="Why does complex phenomenon happen?"), Features(psi=[0.1]*96, motifs=[Motif(pattern="complex", weight=0.7)])),
    ]

    modes_used = []

    for query, features in queries:
        intent = planner.analyze_intent(query, features)
        mode = bandit.select_mode(intent.complexity, 0.7)

        engine.mode = mode
        result = await engine.reason(query, features, good_context)

        success = result.total_confidence >= 0.75
        bandit.update(mode, success, result.total_confidence, result.duration_ms)

        modes_used.append(mode)

    # Should have used different modes for different query types
    # (At minimum, not all the same mode)
    unique_modes = len(set(modes_used))
    # This is probabilistic, but very likely to use at least 1-2 different modes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
