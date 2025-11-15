#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Reasoning Engine Integration
====================================
Demonstrates Layer 6 Reasoning Engine integrated with WeavingOrchestrator.

This demo shows:
1. Basic reasoning-enhanced weaving
2. Different reasoning modes (FAST, STANDARD)
3. Reasoning chain visualization
4. Scratchpad provenance tracking
5. Performance comparison

Author: Claude Code
Date: 2025-11-15
Phase: 2 (Integration Demo)
"""

import asyncio
import sys
import time
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config, ReasoningMode
from HoloLoom.weaving_orchestrator_reasoning import (
    ReasoningOrchestrator,
    weave_with_reasoning
)

# Try to import provenance tracker
try:
    from HoloLoom.recursive.reasoning_provenance import (
        ReasoningProvenanceTracker,
        ReasoningAwareScratchpadOrchestrator
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False
    print("⚠️  Provenance tracker not available (Promptly not installed)")


# ============================================================================
# Demo Data
# ============================================================================

def create_demo_shards():
    """Create demonstration memory shards about Thompson Sampling."""
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a probabilistic algorithm for the multi-armed bandit problem. "
                 "It uses Bayesian inference to balance exploration and exploitation.",
            source="demo_knowledge_base",
            timestamp=0.0,
            metadata={"topic": "reinforcement_learning", "confidence": 0.95}
        ),
        MemoryShard(
            id="shard_2",
            text="The algorithm maintains Beta distributions for each action, with alpha representing "
                 "successes and beta representing failures. Actions are selected by sampling from these "
                 "distributions.",
            source="demo_knowledge_base",
            timestamp=1.0,
            metadata={"topic": "thompson_sampling", "confidence": 0.92}
        ),
        MemoryShard(
            id="shard_3",
            text="When an action is taken, the parameters are updated: alpha increases on success, "
                 "beta increases on failure. This naturally implements exploration (high uncertainty) "
                 "and exploitation (high expected value).",
            source="demo_knowledge_base",
            timestamp=2.0,
            metadata={"topic": "bayesian_updates", "confidence": 0.88}
        ),
        MemoryShard(
            id="shard_4",
            text="Thompson Sampling has theoretical guarantees: it achieves logarithmic regret bounds, "
                 "meaning the algorithm learns efficiently over time.",
            source="demo_knowledge_base",
            timestamp=3.0,
            metadata={"topic": "regret_bounds", "confidence": 0.85}
        ),
        MemoryShard(
            id="shard_5",
            text="Compared to epsilon-greedy (which explores randomly), Thompson Sampling explores "
                 "more intelligently by favoring uncertain actions that might be optimal.",
            source="demo_knowledge_base",
            timestamp=4.0,
            metadata={"topic": "comparison", "confidence": 0.90}
        ),
    ]
    return shards


# ============================================================================
# Visualization Helpers
# ============================================================================

def print_header(title: str, char: str = "="):
    """Print formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_reasoning_chain(spacetime, verbose: bool = True):
    """Pretty-print reasoning chain from spacetime."""
    if 'reasoning_chain' not in spacetime.metadata:
        print("❌ No reasoning chain found in spacetime metadata")
        return

    chain = spacetime.metadata['reasoning_chain']
    mode = spacetime.metadata['reasoning_mode']
    confidence = spacetime.metadata['reasoning_confidence']
    duration = spacetime.metadata['reasoning_duration_ms']

    # Summary
    print(f"📊 Reasoning Summary:")
    print(f"   Mode: {mode.upper()}")
    print(f"   Steps: {len(chain)}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Duration: {duration:.1f}ms")
    print()

    # Chain visualization
    if verbose:
        print(f"🧠 Reasoning Chain:")
        print(f"{'─' * 70}")

        step_icons = {
            'understanding': '🎯',
            'evidence': '🔍',
            'synthesis': '🔗',
            'verification': '✓',
            'correction': '🔧',
            'planning': '📋',
            'backtrack': '↩'
        }

        for i, step in enumerate(chain):
            icon = step_icons.get(step['step_type'], '•')
            step_type = step['step_type'].upper()

            print(f"\n{i+1}. {icon} [{step_type}] Confidence: {step['confidence']:.2%}")
            print(f"   Thought: {step['thought']}")

            if step['evidence']:
                # Truncate long evidence
                evidence = step['evidence']
                if len(evidence) > 100:
                    evidence = evidence[:97] + "..."
                print(f"   Evidence: {evidence}")

        print(f"\n{'─' * 70}")


def print_comparison_table(results: list):
    """Print comparison table of different modes."""
    print(f"\n📈 Performance Comparison:")
    print(f"{'─' * 70}")
    print(f"{'Mode':<12} {'Steps':<8} {'Confidence':<12} {'Duration':<12}")
    print(f"{'─' * 70}")

    for result in results:
        mode = result['mode']
        steps = result['steps']
        confidence = result['confidence']
        duration = result['duration']

        print(f"{mode:<12} {steps:<8} {confidence:<12.2%} {duration:<12.1f}ms")

    print(f"{'─' * 70}")


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_basic_reasoning(shards):
    """Demo 1: Basic reasoning-enhanced weaving."""
    print_header("Demo 1: Basic Reasoning Integration")

    config = Config.fast()
    config.enable_reasoning = True
    config.reasoning_mode = ReasoningMode.STANDARD

    query = Query(text="What is Thompson Sampling and how does it work?")

    print(f"Query: {query.text}\n")

    async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
        start = time.time()
        spacetime = await orchestrator.weave(query)
        total_duration = (time.time() - start) * 1000

        print(f"✅ Weaving completed in {total_duration:.1f}ms\n")
        print_reasoning_chain(spacetime, verbose=True)


async def demo_mode_comparison(shards):
    """Demo 2: Compare different reasoning modes."""
    print_header("Demo 2: Reasoning Mode Comparison")

    query = Query(text="How does Thompson Sampling balance exploration and exploitation?")
    print(f"Query: {query.text}\n")

    modes = [
        (ReasoningMode.FAST, "FAST"),
        (ReasoningMode.STANDARD, "STANDARD"),
    ]

    results = []

    for mode, name in modes:
        print(f"🔄 Testing {name} mode...")

        config = Config.fast()
        config.enable_reasoning = True
        config.reasoning_mode = mode

        async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
            spacetime = await orchestrator.weave(query)

            chain = spacetime.metadata['reasoning_chain']
            confidence = spacetime.metadata['reasoning_confidence']
            duration = spacetime.metadata['reasoning_duration_ms']

            results.append({
                'mode': name,
                'steps': len(chain),
                'confidence': confidence,
                'duration': duration
            })

            print(f"   Steps: {len(chain)}, Confidence: {confidence:.2%}, Duration: {duration:.1f}ms")

    print_comparison_table(results)


async def demo_reasoning_escalation(shards):
    """Demo 3: Show reasoning escalation (FAST → STANDARD)."""
    print_header("Demo 3: Reasoning Escalation")

    # Simple query (should use FAST)
    simple_query = Query(text="What is Thompson Sampling?")

    # Complex query (might escalate to STANDARD)
    complex_query = Query(
        text="Compare Thompson Sampling with epsilon-greedy and UCB algorithms, "
             "explaining when each is most appropriate and their theoretical guarantees."
    )

    config = Config.fast()
    config.enable_reasoning = True
    config.reasoning_mode = ReasoningMode.FAST  # Start with FAST

    print("Testing simple query (should stay FAST):")
    print(f"  '{simple_query.text}'\n")

    async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
        spacetime = await orchestrator.weave(simple_query)

        mode = spacetime.metadata['reasoning_mode']
        steps = len(spacetime.metadata['reasoning_chain'])
        duration = spacetime.metadata['reasoning_duration_ms']

        print(f"  Result: {mode.upper()} mode, {steps} steps, {duration:.1f}ms")

        if mode == 'fast' and steps == 1:
            print(f"  ✅ Query stayed in FAST mode (high confidence)")
        elif steps > 1:
            print(f"  🔄 Query escalated (low confidence, needed more reasoning)")


async def demo_provenance_tracking(shards):
    """Demo 4: Scratchpad provenance tracking."""
    print_header("Demo 4: Provenance Tracking")

    if not PROVENANCE_AVAILABLE:
        print("⚠️  Skipping (provenance tracker not available)")
        return

    config = Config.fast()
    config.enable_reasoning = True
    config.reasoning_mode = ReasoningMode.STANDARD

    query = Query(text="What is Thompson Sampling?")
    print(f"Query: {query.text}\n")

    async with ReasoningAwareScratchpadOrchestrator(
        cfg=config,
        shards=shards,
        enable_reasoning=True
    ) as orchestrator:
        spacetime = await orchestrator.weave(query)

        if orchestrator.scratchpad:
            # Get reasoning history
            reasoning_history = orchestrator.get_reasoning_history()

            print(f"📝 Scratchpad Provenance:")
            print(f"   Total entries: {len(orchestrator.scratchpad.get_history())}")
            print(f"   Reasoning entries: {len(reasoning_history)}\n")

            print(f"🔍 Reasoning Entry Details:")
            for i, entry in enumerate(reasoning_history[:3]):  # Show first 3
                print(f"\n   Entry {i+1}:")
                print(f"   Thought: {entry.thought}")
                print(f"   Action: {entry.action}")
                print(f"   Score: {entry.score:.2%}")


async def demo_chain_visualization(shards):
    """Demo 5: Detailed reasoning chain visualization."""
    print_header("Demo 5: Detailed Chain Visualization")

    config = Config.fast()
    config.enable_reasoning = True
    config.reasoning_mode = ReasoningMode.STANDARD

    query = Query(text="Explain how Thompson Sampling achieves logarithmic regret bounds.")
    print(f"Query: {query.text}\n")

    async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
        spacetime = await orchestrator.weave(query)

        print_reasoning_chain(spacetime, verbose=True)

        # Show confidence trajectory
        chain = spacetime.metadata['reasoning_chain']
        confidences = [step['confidence'] for step in chain]

        print(f"\n📊 Confidence Trajectory:")
        print(f"   Steps: {' → '.join(f'{c:.0%}' for c in confidences)}")

        # Show average and variance
        avg_conf = sum(confidences) / len(confidences)
        variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)

        print(f"   Average: {avg_conf:.2%}")
        print(f"   Variance: {variance:.4f}")

        if variance < 0.01:
            print(f"   ✅ Stable reasoning (low variance)")
        else:
            print(f"   ⚠️  Variable reasoning (higher uncertainty)")


# ============================================================================
# Main Demo Runner
# ============================================================================

async def run_all_demos():
    """Run all demonstration scenarios."""
    print_header("🧠 Reasoning Engine Integration Demo", char="═")

    print("""
    This demo showcases the Layer 6 Reasoning Engine integrated with
    HoloLoom's WeavingOrchestrator, demonstrating:

    • Multi-step chain-of-thought reasoning
    • Different reasoning modes (FAST, STANDARD)
    • Automatic mode selection and escalation
    • Scratchpad provenance tracking
    • Performance characteristics

    Let's begin!
    """)

    # Create demo data
    shards = create_demo_shards()
    print(f"📚 Loaded {len(shards)} knowledge shards about Thompson Sampling\n")

    # Run demos
    demos = [
        ("Basic Integration", demo_basic_reasoning),
        ("Mode Comparison", demo_mode_comparison),
        ("Escalation", demo_reasoning_escalation),
        ("Provenance Tracking", demo_provenance_tracking),
        ("Chain Visualization", demo_chain_visualization),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            await demo_func(shards)
            print(f"\n✅ Demo {i}/{len(demos)} completed\n")
            await asyncio.sleep(0.5)  # Brief pause between demos
        except Exception as e:
            print(f"\n❌ Demo {i}/{len(demos)} failed: {e}\n")
            import traceback
            traceback.print_exc()

    print_header("🎉 All Demos Complete!", char="═")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    asyncio.run(run_all_demos())
