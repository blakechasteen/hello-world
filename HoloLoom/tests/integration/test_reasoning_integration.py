#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests - Reasoning Engine
=====================================
Tests full integration of Layer 6 Reasoning Engine with WeavingOrchestrator.

Tests:
1. Basic reasoning integration
2. Scratchpad provenance tracking
3. Metadata attachment to Spacetime
4. Mode selection (FAST/STANDARD/DEEP)
5. Adaptive reasoning
6. Error handling and fallback

Author: Claude Code
Date: 2025-11-15
Phase: 2 (Integration Testing)
"""

import pytest
import asyncio
from typing import List

# Core types
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config, ReasoningMode

# Orchestrator
from HoloLoom.weaving_orchestrator_reasoning import (
    ReasoningOrchestrator,
    weave_with_reasoning,
    weave_with_auto_reasoning
)

# Provenance
try:
    from HoloLoom.recursive.reasoning_provenance import (
        ReasoningProvenanceTracker,
        ReasoningAwareScratchpadOrchestrator
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration with reasoning enabled."""
    config = Config.fast()
    config.enable_reasoning = True
    config.reasoning_mode = ReasoningMode.STANDARD
    config.max_reasoning_steps = 5
    config.reasoning_verification_threshold = 0.75
    return config


@pytest.fixture
def test_shards():
    """Create test memory shards."""
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian bandit algorithm that balances exploration and exploitation.",
            source="test_source",
            timestamp=0.0,
            metadata={"topic": "reinforcement_learning"}
        ),
        MemoryShard(
            id="shard_2",
            text="The algorithm samples from Beta distributions to select actions probabilistically.",
            source="test_source",
            timestamp=1.0,
            metadata={"topic": "thompson_sampling"}
        ),
        MemoryShard(
            id="shard_3",
            text="Alpha and beta parameters are updated based on rewards (successes and failures).",
            source="test_source",
            timestamp=2.0,
            metadata={"topic": "bayesian_inference"}
        ),
    ]
    return shards


# ============================================================================
# Basic Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_orchestrator_basic(test_config, test_shards):
    """Test basic reasoning orchestrator integration."""
    query = Query(text="What is Thompson Sampling?")

    async with ReasoningOrchestrator(
        cfg=test_config,
        shards=test_shards,
        enable_reasoning=True
    ) as orchestrator:
        # Weave with reasoning
        spacetime = await orchestrator.weave(query)

        # Verify spacetime was created
        assert spacetime is not None
        assert hasattr(spacetime, 'metadata')

        # Verify reasoning chain exists
        assert 'reasoning_chain' in spacetime.metadata
        assert 'reasoning_mode' in spacetime.metadata
        assert 'reasoning_confidence' in spacetime.metadata
        assert 'reasoning_duration_ms' in spacetime.metadata

        # Verify chain structure
        chain = spacetime.metadata['reasoning_chain']
        assert len(chain) >= 1  # At least one reasoning step

        for step in chain:
            assert 'thought' in step
            assert 'evidence' in step
            assert 'confidence' in step
            assert 'step_type' in step
            assert 0.0 <= step['confidence'] <= 1.0

        print(f"\n✓ Basic integration test passed")
        print(f"  Reasoning mode: {spacetime.metadata['reasoning_mode']}")
        print(f"  Steps: {len(chain)}")
        print(f"  Confidence: {spacetime.metadata['reasoning_confidence']:.2f}")
        print(f"  Duration: {spacetime.metadata['reasoning_duration_ms']:.1f}ms")


@pytest.mark.asyncio
async def test_reasoning_disabled_fallback(test_config, test_shards):
    """Test that reasoning can be disabled and falls back to base orchestrator."""
    query = Query(text="What is Thompson Sampling?")

    async with ReasoningOrchestrator(
        cfg=test_config,
        shards=test_shards,
        enable_reasoning=False  # Disabled
    ) as orchestrator:
        spacetime = await orchestrator.weave(query)

        # Should still work, but without reasoning metadata
        assert spacetime is not None

        # Reasoning metadata should not be present
        if hasattr(spacetime, 'metadata') and spacetime.metadata:
            assert 'reasoning_chain' not in spacetime.metadata

        print(f"\n✓ Reasoning disabled fallback test passed")


# ============================================================================
# Mode Selection Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_mode_fast(test_config, test_shards):
    """Test FAST reasoning mode."""
    test_config.reasoning_mode = ReasoningMode.FAST
    query = Query(text="What is Thompson Sampling?")

    async with ReasoningOrchestrator(
        cfg=test_config,
        shards=test_shards,
        enable_reasoning=True
    ) as orchestrator:
        spacetime = await orchestrator.weave(query)

        # Verify FAST mode was used
        assert spacetime.metadata['reasoning_mode'] == 'fast'

        # FAST mode should have 1 step (or escalate to STANDARD)
        chain = spacetime.metadata['reasoning_chain']
        # Allow 1 step OR 3-5 steps (if escalated)
        assert len(chain) >= 1

        # FAST mode should be quick
        assert spacetime.metadata['reasoning_duration_ms'] < 100.0  # <100ms

        print(f"\n✓ FAST mode test passed")
        print(f"  Steps: {len(chain)}")
        print(f"  Duration: {spacetime.metadata['reasoning_duration_ms']:.1f}ms")


@pytest.mark.asyncio
async def test_reasoning_mode_standard(test_config, test_shards):
    """Test STANDARD reasoning mode."""
    test_config.reasoning_mode = ReasoningMode.STANDARD
    query = Query(text="How does Thompson Sampling balance exploration and exploitation?")

    async with ReasoningOrchestrator(
        cfg=test_config,
        shards=test_shards,
        enable_reasoning=True
    ) as orchestrator:
        spacetime = await orchestrator.weave(query)

        # Verify STANDARD mode was used
        assert spacetime.metadata['reasoning_mode'] == 'standard'

        # STANDARD mode should have 3-5 steps
        chain = spacetime.metadata['reasoning_chain']
        assert 3 <= len(chain) <= 5

        # Check step types are varied
        step_types = [step['step_type'] for step in chain]
        assert 'understanding' in step_types or 'evidence' in step_types

        print(f"\n✓ STANDARD mode test passed")
        print(f"  Steps: {len(chain)}")
        print(f"  Step types: {step_types}")
        print(f"  Duration: {spacetime.metadata['reasoning_duration_ms']:.1f}ms")


@pytest.mark.asyncio
async def test_reasoning_mode_override(test_config, test_shards):
    """Test reasoning mode override per query."""
    test_config.reasoning_mode = ReasoningMode.STANDARD
    query = Query(text="What is Thompson Sampling?")

    async with ReasoningOrchestrator(
        cfg=test_config,
        shards=test_shards,
        enable_reasoning=True
    ) as orchestrator:
        # Override to FAST for this query
        spacetime = await orchestrator.weave(
            query,
            reasoning_mode_override=ReasoningMode.FAST
        )

        # Should use FAST, not STANDARD
        assert spacetime.metadata['reasoning_mode'] == 'fast'

        print(f"\n✓ Mode override test passed")


# ============================================================================
# Provenance Integration Tests
# ============================================================================

@pytest.mark.skipif(not PROVENANCE_AVAILABLE, reason="Provenance tracker not available")
@pytest.mark.asyncio
async def test_provenance_extraction(test_config, test_shards):
    """Test reasoning provenance extraction to scratchpad format."""
    from HoloLoom.reasoning.types import ReasoningResult, ReasoningStep, StepType
    from datetime import datetime

    # Create mock reasoning result
    chain = [
        ReasoningStep(
            thought="Understanding the query intent",
            evidence="Query asks about Thompson Sampling",
            confidence=0.9,
            step_type=StepType.UNDERSTANDING
        ),
        ReasoningStep(
            thought="Gathering evidence from context",
            evidence="Found 3 relevant shards about bandit algorithms",
            confidence=0.85,
            step_type=StepType.EVIDENCE
        ),
        ReasoningStep(
            thought="Synthesizing answer",
            evidence="Thompson Sampling is a Bayesian algorithm...",
            confidence=0.88,
            step_type=StepType.SYNTHESIS
        ),
    ]

    reasoning_result = ReasoningResult(
        chain=chain,
        mode=ReasoningMode.STANDARD,
        total_confidence=0.88,
        duration_ms=150.0
    )

    # Extract provenance
    tracker = ReasoningProvenanceTracker()
    entries = tracker.extract_reasoning_provenance(reasoning_result)

    # Verify entries
    assert len(entries) == 3

    for i, entry in enumerate(entries):
        assert entry.thought.startswith('[')  # Has step type prefix
        assert 'reasoning_step' in entry.action
        assert entry.score == chain[i].confidence
        assert entry.iteration == i + 1

    # Verify metadata
    assert all('reasoning_mode' in e.metadata for e in entries)
    assert all('step_type' in e.metadata for e in entries)

    print(f"\n✓ Provenance extraction test passed")
    print(f"  Entries: {len(entries)}")
    for entry in entries:
        print(f"  - {entry.action}: {entry.thought[:50]}... (score: {entry.score:.2f})")


@pytest.mark.skipif(not PROVENANCE_AVAILABLE, reason="Provenance tracker not available")
@pytest.mark.asyncio
async def test_scratchpad_orchestrator_integration(test_config, test_shards):
    """Test ReasoningAwareScratchpadOrchestrator."""
    query = Query(text="What is Thompson Sampling?")

    async with ReasoningAwareScratchpadOrchestrator(
        cfg=test_config,
        shards=test_shards,
        enable_reasoning=True
    ) as orchestrator:
        # Weave
        spacetime = await orchestrator.weave(query)

        # Verify scratchpad was populated
        if orchestrator.scratchpad:
            history = orchestrator.scratchpad.get_history()
            assert len(history) > 0

            # Get reasoning entries
            reasoning_history = orchestrator.get_reasoning_history()
            assert len(reasoning_history) > 0

            print(f"\n✓ Scratchpad orchestrator test passed")
            print(f"  Total entries: {len(history)}")
            print(f"  Reasoning entries: {len(reasoning_history)}")


# ============================================================================
# Convenience Function Tests
# ============================================================================

@pytest.mark.asyncio
async def test_weave_with_reasoning_convenience(test_config, test_shards):
    """Test weave_with_reasoning convenience function."""
    query = Query(text="What is Thompson Sampling?")

    spacetime = await weave_with_reasoning(
        query=query,
        config=test_config,
        shards=test_shards,
        reasoning_mode=ReasoningMode.STANDARD
    )

    # Verify reasoning was applied
    assert 'reasoning_chain' in spacetime.metadata
    assert spacetime.metadata['reasoning_mode'] == 'standard'

    print(f"\n✓ Convenience function test passed")


@pytest.mark.asyncio
async def test_weave_with_auto_reasoning(test_config, test_shards):
    """Test weave_with_auto_reasoning convenience function."""
    query = Query(text="What is Thompson Sampling?")

    spacetime = await weave_with_auto_reasoning(
        query=query,
        config=test_config,
        shards=test_shards
    )

    # Verify reasoning was applied
    assert 'reasoning_chain' in spacetime.metadata

    # Mode should be auto-selected
    mode = spacetime.metadata['reasoning_mode']
    assert mode in ['fast', 'standard', 'deep']

    print(f"\n✓ Auto-reasoning test passed")
    print(f"  Auto-selected mode: {mode}")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_performance_overhead(test_config, test_shards):
    """Test that reasoning overhead is within acceptable limits."""
    query = Query(text="What is Thompson Sampling?")

    # Test each mode
    modes = [
        (ReasoningMode.FAST, 100.0),    # <100ms
        (ReasoningMode.STANDARD, 300.0),  # <300ms
    ]

    for mode, max_duration in modes:
        test_config.reasoning_mode = mode

        async with ReasoningOrchestrator(
            cfg=test_config,
            shards=test_shards,
            enable_reasoning=True
        ) as orchestrator:
            spacetime = await orchestrator.weave(query)

            duration = spacetime.metadata['reasoning_duration_ms']

            # Check overhead is acceptable
            # Note: In testing environment, may be slower
            # In production, these limits should hold
            print(f"  {mode.value} mode: {duration:.1f}ms (limit: {max_duration}ms)")

            # Relaxed check for testing environment
            assert duration < max_duration * 2, \
                f"{mode.value} mode took {duration:.1f}ms (expected <{max_duration}ms)"

    print(f"\n✓ Performance overhead test passed")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
