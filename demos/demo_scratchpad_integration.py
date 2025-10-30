#!/usr/bin/env python3
"""
Demo: HoloLoom + Scratchpad Integration (Phase 1)
==================================================
Demonstrates full provenance tracking with automatic refinement.

Shows:
1. Scratchpad accumulation across queries
2. Provenance extraction from Spacetime traces
3. Automatic refinement on low confidence
4. Complete reasoning history visualization
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.recursive import ScratchpadOrchestrator, ScratchpadConfig
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard


# ============================================================================
# Test Data
# ============================================================================

def create_test_shards() -> list:
    """Create test memory shards for demo"""
    return [
        MemoryShard(
            id="thompson_sampling",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                 "It balances exploration and exploitation by sampling from posterior distributions. "
                 "Each arm has a prior (e.g., Beta distribution), and the algorithm selects the arm "
                 "with the highest sampled value. After observing the reward, it updates the posterior. "
                 "This naturally explores uncertain arms while exploiting known good ones.",
            metadata={
                "tags": ["reinforcement_learning", "bandit", "bayesian"],
                "source": "ml_notes"
            }
        ),
        MemoryShard(
            id="hololoom_arch",
            text="HoloLoom orchestrator implements a 9-step weaving cycle: "
                 "1) Loom Command selects pattern card "
                 "2) Chrono Trigger creates temporal window "
                 "3) Yarn Graph threads selected "
                 "4) Resonance Shed extracts features "
                 "5) Warp Space tensions manifold "
                 "6) Convergence Engine collapses to tool "
                 "7) Tool executes "
                 "8) Spacetime fabric woven "
                 "9) Reflection Buffer learns. "
                 "The system uses multi-scale Matryoshka embeddings and spectral graph features.",
            metadata={
                "tags": ["hololoom", "architecture", "weaving"],
                "source": "docs"
            }
        ),
        MemoryShard(
            id="matryoshka",
            text="Matryoshka embeddings use nested representation learning. "
                 "A single forward pass produces embeddings at multiple scales (96D, 192D, 384D). "
                 "Smaller dimensions are prefixes of larger ones, enabling efficient multi-scale retrieval. "
                 "This allows trading off between speed (96D) and quality (384D) dynamically.",
            metadata={
                "tags": ["embedding", "matryoshka", "multi_scale"],
                "source": "ml_notes"
            }
        ),
        MemoryShard(
            id="scratchpad",
            text="Scratchpad reasoning tracks thought -> action -> observation across iterations. "
                 "Each entry records: what was thought, what action was taken, what was observed, "
                 "and a quality score. This enables full provenance tracking and iterative refinement. "
                 "Inspired by Hofstadter's Strange Loops and chain-of-thought reasoning.",
            metadata={
                "tags": ["reasoning", "scratchpad", "provenance"],
                "source": "promptly_docs"
            }
        )
    ]


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_basic_provenance():
    """Demo 1: Basic scratchpad provenance tracking"""
    print("=" * 80)
    print("DEMO 1: Basic Provenance Tracking")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_test_shards()

    scratchpad_config = ScratchpadConfig(
        enable_scratchpad=True,
        enable_refinement=False  # Disabled for this demo
    )

    async with ScratchpadOrchestrator(
        cfg=config,
        shards=shards,
        scratchpad_config=scratchpad_config
    ) as orchestrator:
        # Query 1: High confidence
        print("Query 1: High-confidence query about Thompson Sampling")
        print("-" * 80)

        spacetime1, scratchpad1 = await orchestrator.weave_with_provenance(
            Query(text="What is Thompson Sampling and how does it work?")
        )

        print(f"Response: {spacetime1.response[:150]}...")
        print(f"Confidence: {spacetime1.trace.tool_confidence:.2f}")
        print(f"Tool: {spacetime1.trace.tool_selected}")
        print(f"Threads activated: {len(spacetime1.trace.threads_activated)}")
        print()

        # Query 2: Different topic
        print("Query 2: Query about HoloLoom architecture")
        print("-" * 80)

        spacetime2, scratchpad2 = await orchestrator.weave_with_provenance(
            Query(text="How does the HoloLoom weaving cycle work?")
        )

        print(f"Response: {spacetime2.response[:150]}...")
        print(f"Confidence: {spacetime2.trace.tool_confidence:.2f}")
        print(f"Tool: {spacetime2.trace.tool_selected}")
        print(f"Threads activated: {len(spacetime2.trace.threads_activated)}")
        print()

        # Show scratchpad history
        print("=" * 80)
        print("SCRATCHPAD HISTORY (2 queries)")
        print("=" * 80)
        print(scratchpad2.get_history())
        print()

        # Show statistics
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80)
        stats = orchestrator.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()


async def demo_recursive_refinement():
    """Demo 2: Automatic refinement on low confidence"""
    print("=" * 80)
    print("DEMO 2: Recursive Refinement (Low Confidence)")
    print("=" * 80)
    print()

    config = Config.bare()  # Use BARE mode for faster, lower-quality results
    shards = create_test_shards()

    scratchpad_config = ScratchpadConfig(
        enable_scratchpad=True,
        enable_refinement=True,
        refinement_threshold=0.85,  # High threshold to trigger refinement
        max_refinement_iterations=2
    )

    async with ScratchpadOrchestrator(
        cfg=config,
        shards=shards,
        scratchpad_config=scratchpad_config
    ) as orchestrator:
        print("Query: Ambiguous query (likely low confidence)")
        print("-" * 80)

        spacetime, scratchpad = await orchestrator.weave_with_provenance(
            Query(text="Tell me about embeddings")  # Vague query
        )

        print(f"Final response: {spacetime.response[:150]}...")
        print(f"Final confidence: {spacetime.trace.tool_confidence:.2f}")
        print()

        # Show refinement history
        print("=" * 80)
        print("REFINEMENT HISTORY")
        print("=" * 80)
        print(scratchpad.get_history())
        print()

        # Show statistics
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80)
        stats = orchestrator.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

        print(f"Refinements triggered: {orchestrator.refinements_triggered}")
        print()


async def demo_provenance_details():
    """Demo 3: Detailed provenance inspection"""
    print("=" * 80)
    print("DEMO 3: Detailed Provenance Inspection")
    print("=" * 80)
    print()

    config = Config.fused()  # Use FUSED mode for full features
    shards = create_test_shards()

    scratchpad_config = ScratchpadConfig(enable_scratchpad=True)

    async with ScratchpadOrchestrator(
        cfg=config,
        shards=shards,
        scratchpad_config=scratchpad_config
    ) as orchestrator:
        print("Query: Rich query to extract detailed provenance")
        print("-" * 80)

        spacetime, scratchpad = await orchestrator.weave_with_provenance(
            Query(text="Explain Matryoshka embeddings and their use in HoloLoom")
        )

        print(f"Response: {spacetime.response[:200]}...")
        print()

        # Show last scratchpad entry details
        if scratchpad and scratchpad.entries:
            entry = scratchpad.entries[-1]

            print("=" * 80)
            print("PROVENANCE ENTRY DETAILS")
            print("=" * 80)
            print(f"Iteration: {entry.iteration}")
            print(f"Score: {entry.score:.3f}")
            print()
            print(f"THOUGHT (What was detected):")
            print(f"  {entry.thought}")
            print()
            print(f"ACTION (What was decided):")
            print(f"  {entry.action}")
            print()
            print(f"OBSERVATION (What happened):")
            print(f"  {entry.observation}")
            print()
            print(f"METADATA:")
            for key, value in entry.metadata.items():
                if not isinstance(value, dict):  # Skip nested dicts
                    print(f"  {key}: {value}")
            print()


async def demo_scratchpad_persistence():
    """Demo 4: Scratchpad persistence"""
    print("=" * 80)
    print("DEMO 4: Scratchpad Persistence")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_test_shards()

    persist_path = Path(__file__).parent.parent / "logs" / "scratchpad_demo.json"

    scratchpad_config = ScratchpadConfig(
        enable_scratchpad=True,
        persist_scratchpad=True,
        persist_path=str(persist_path)
    )

    async with ScratchpadOrchestrator(
        cfg=config,
        shards=shards,
        scratchpad_config=scratchpad_config
    ) as orchestrator:
        # Process several queries
        queries = [
            "What is Thompson Sampling?",
            "How does HoloLoom work?",
            "Explain Matryoshka embeddings"
        ]

        for i, q in enumerate(queries, 1):
            print(f"Query {i}: {q}")
            spacetime, _ = await orchestrator.weave_with_provenance(Query(text=q))
            print(f"  Confidence: {spacetime.trace.tool_confidence:.2f}")

        print()
        print(f"Scratchpad will be persisted to: {persist_path}")
        print()

    # Check if file was created
    if persist_path.exists():
        print(f"SUCCESS: Scratchpad persisted ({persist_path.stat().st_size} bytes)")
        print()

        # Show preview
        import json
        with open(persist_path) as f:
            data = json.load(f)

        print("Persisted data preview:")
        print(f"  Queries processed: {data['queries_processed']}")
        print(f"  Entries: {len(data['entries'])}")
        print(f"  First entry query: {data['entries'][0]['metadata']['query'][:50]}...")
        print()
    else:
        print("WARNING: Scratchpad file not created")
        print()


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos"""
    print()
    print("*" * 80)
    print("HoloLoom + Scratchpad Integration Demo")
    print("Phase 1: Full Provenance Tracking")
    print("*" * 80)
    print()

    try:
        await demo_basic_provenance()
        print()

        await demo_recursive_refinement()
        print()

        await demo_provenance_details()
        print()

        await demo_scratchpad_persistence()
        print()

        print("*" * 80)
        print("ALL DEMOS COMPLETE")
        print("*" * 80)
        print()
        print("Key Achievements:")
        print("  1. Full provenance tracking working")
        print("  2. Scratchpad accumulation across queries")
        print("  3. Automatic refinement on low confidence")
        print("  4. Detailed trace -> scratchpad mapping")
        print("  5. Scratchpad persistence to disk")
        print()
        print("Phase 1 Implementation: COMPLETE")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
