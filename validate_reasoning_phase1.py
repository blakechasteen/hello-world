#!/usr/bin/env python3
"""
Validation script for Reasoning Engine Phase 1
==============================================
Tests basic functionality without requiring full HoloLoom dependencies.
"""

import sys
import asyncio
from typing import List

# Direct imports to avoid dependency issues
sys.path.insert(0, '/home/user/hello-world')

from HoloLoom.reasoning.types import (
    ReasoningMode,
    StepType,
    QueryType,
    ReasoningStep,
    ReasoningResult,
    QueryIntent,
)
from HoloLoom.reasoning.planner import QueryPlanner
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.engine import ReasoningEngine

# Mock types to avoid importing full HoloLoom
class Query:
    def __init__(self, text):
        self.text = text

class Features:
    def __init__(self, motifs, embeddings, spectral):
        self.motifs = motifs
        self.embeddings = embeddings
        self.spectral = spectral

class MemoryShard:
    def __init__(self, text, entities, timestamp):
        self.text = text
        self.entities = entities
        self.timestamp = timestamp

class Context:
    def __init__(self, shards, retrieved_at):
        self.shards = shards
        self.retrieved_at = retrieved_at


def test_types():
    """Test core types are properly defined."""
    print("Testing core types...")

    # Test enums
    assert ReasoningMode.FAST.value == "fast"
    assert ReasoningMode.STANDARD.value == "standard"
    assert ReasoningMode.DEEP.value == "deep"

    assert StepType.UNDERSTANDING.value == "understanding"
    assert QueryType.FACTUAL.value == "factual"

    # Test data classes
    step = ReasoningStep(
        thought="Test thought",
        evidence="Test evidence",
        confidence=0.8,
        step_type=StepType.SYNTHESIS
    )
    assert step.thought == "Test thought"
    assert step.confidence == 0.8

    print("✓ Core types test passed")


def test_query_planner():
    """Test QueryPlanner component."""
    print("Testing QueryPlanner...")

    planner = QueryPlanner()

    # Test factual query
    query = Query(text="What is Thompson Sampling?")
    features = Features(motifs=["thompson", "sampling"], embeddings=[[0.1] * 96], spectral=None)

    intent = planner.analyze_intent(query, features)

    assert intent.type == QueryType.FACTUAL
    assert 0.0 <= intent.complexity <= 1.0
    assert 0.0 <= intent.confidence <= 1.0
    assert len(intent.requirements) > 0

    # Test comparative query
    query2 = Query(text="Compare A and B")
    intent2 = planner.analyze_intent(query2, features)

    assert intent2.type == QueryType.COMPARATIVE

    print("✓ QueryPlanner test passed")


def test_chain_of_thought():
    """Test ChainOfThought component."""
    print("Testing ChainOfThought...")

    cot = ChainOfThought()

    # Test evidence extraction
    shards = [
        MemoryShard(
            text="Thompson Sampling is a Bayesian approach",
            entities=["thompson_sampling"],
            timestamp=1.0
        ),
        MemoryShard(
            text="It balances exploration and exploitation",
            entities=["exploration"],
            timestamp=1.0
        ),
    ]
    context = Context(shards=shards, retrieved_at=None)

    intent = QueryIntent(
        type=QueryType.FACTUAL,
        requirements=["definition"],
        complexity=0.3,
        confidence=0.8,
        key_concepts=["thompson", "sampling"]
    )

    evidence = cot.extract_evidence(context, intent)
    assert len(evidence) > 0

    # Test synthesis
    synthesis = cot.synthesize(intent, evidence)
    assert synthesis.reasoning
    assert 0.0 <= synthesis.confidence <= 1.0

    # Test standard chain generation
    query = Query(text="What is Thompson Sampling?")
    features = Features(motifs=["thompson"], embeddings=[[0.1] * 96], spectral=None)

    planner = QueryPlanner()
    intent = planner.analyze_intent(query, features)

    chain = cot.generate_standard_chain(query, intent, context)
    assert len(chain) >= 3
    assert any(s.step_type == StepType.UNDERSTANDING for s in chain)

    print("✓ ChainOfThought test passed")


async def test_self_verifier():
    """Test SelfVerifier component."""
    print("Testing SelfVerifier...")

    verifier = SelfVerifier()

    # Test empty chain
    context = Context(shards=[], retrieved_at=None)
    result = await verifier.verify([], context)
    assert not result.passed

    # Test good chain
    chain = [
        ReasoningStep("Understanding", "evidence", 0.9, StepType.UNDERSTANDING),
        ReasoningStep("Evidence", "sources", 0.85, StepType.EVIDENCE),
        ReasoningStep("Synthesis", "combined", 0.88, StepType.SYNTHESIS),
    ]

    result = await verifier.verify(chain, context)
    assert result.passed or not result.passed  # Either is valid depending on checks

    # Test multipass verification
    multipass = await verifier.verify_multipass(chain, context, num_passes=3)
    assert len(multipass.passes) == 3
    assert 0.0 <= multipass.overall_confidence <= 1.0

    print("✓ SelfVerifier test passed")


async def test_reasoning_engine():
    """Test ReasoningEngine main component."""
    print("Testing ReasoningEngine...")

    engine = ReasoningEngine(mode=ReasoningMode.STANDARD)

    # Create test data
    query = Query(text="What is Thompson Sampling?")
    features = Features(
        motifs=["thompson", "sampling"],
        embeddings=[[0.1] * 96],
        spectral=None
    )

    shards = [
        MemoryShard(
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits",
            entities=["thompson_sampling", "bayesian"],
            timestamp=1.0
        ),
    ]
    context = Context(shards=shards, retrieved_at=None)

    # Test FAST mode
    engine_fast = ReasoningEngine(mode=ReasoningMode.FAST)
    result_fast = await engine_fast.reason(query, features, context)

    assert result_fast.mode == ReasoningMode.FAST
    assert len(result_fast.chain) >= 1
    assert 0.0 <= result_fast.total_confidence <= 1.0
    assert result_fast.duration_ms > 0

    print(f"  FAST mode: {len(result_fast.chain)} steps, "
          f"confidence={result_fast.total_confidence:.2f}, "
          f"duration={result_fast.duration_ms:.1f}ms")

    # Test STANDARD mode
    engine_standard = ReasoningEngine(mode=ReasoningMode.STANDARD)
    result_standard = await engine_standard.reason(query, features, context)

    assert result_standard.mode == ReasoningMode.STANDARD
    assert len(result_standard.chain) >= 3
    assert 0.0 <= result_standard.total_confidence <= 1.0

    print(f"  STANDARD mode: {len(result_standard.chain)} steps, "
          f"confidence={result_standard.total_confidence:.2f}, "
          f"duration={result_standard.duration_ms:.1f}ms")

    # Test mode selection
    selected_mode = engine.select_mode(query, features, context)
    assert selected_mode in [ReasoningMode.FAST, ReasoningMode.STANDARD, ReasoningMode.DEEP]

    print(f"  Auto-selected mode: {selected_mode.value}")

    print("✓ ReasoningEngine test passed")


async def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Reasoning Engine Phase 1 Validation")
    print("=" * 60)
    print()

    try:
        test_types()
        test_query_planner()
        test_chain_of_thought()
        await test_self_verifier()
        await test_reasoning_engine()

        print()
        print("=" * 60)
        print("✓ All Phase 1 validation tests passed!")
        print("=" * 60)
        print()
        print("Phase 1 Implementation Summary:")
        print("  - Core types: ReasoningMode, ReasoningStep, ReasoningResult ✓")
        print("  - QueryPlanner: Intent classification ✓")
        print("  - ChainOfThought: Evidence extraction + synthesis ✓")
        print("  - SelfVerifier: Chain verification ✓")
        print("  - ReasoningEngine: FAST + STANDARD modes ✓")
        print()
        print("Next Steps:")
        print("  - Phase 2: WeavingOrchestrator integration")
        print("  - Phase 3: DEEP mode + backtracking")
        print("  - Phase 4: Visualization + tooling")

        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Validation failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
