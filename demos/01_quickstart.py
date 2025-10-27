#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Quickstart Demo
========================
The simplest possible usage of HoloLoom: text in, questions answered.

This demonstrates:
1. Loading text knowledge into memory
2. Querying with natural language
3. Getting contextual responses

No external dependencies required (uses 'simple' memory backend).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.autospin import AutoSpinOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query


async def main():
    """Run the quickstart demo."""

    print("=" * 80)
    print("HOLOLOOM QUICKSTART DEMO")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load knowledge base
    # ========================================================================
    print("\n[STEP 1] Loading knowledge base...")
    print("-" * 80)

    knowledge = """
    HoloLoom is a neural decision-making system that combines:
    - Multi-scale embeddings (Matryoshka representations)
    - Knowledge graph memory with spectral features
    - Unified policy engine with Thompson Sampling exploration
    - PPO reinforcement learning for agent training

    The system uses a weaving metaphor:
    - Warp threads: Independent modules (motif, embedding, memory, policy)
    - Shuttle: The orchestrator that weaves threads together
    - Fabric: The final woven response with full computational trace

    SpinningWheel is the input adapter system that converts:
    - Web pages → MemoryShards
    - Audio transcripts → MemoryShards
    - YouTube videos → MemoryShards
    - Raw text → MemoryShards

    The weaving architecture includes:
    - LoomCommand: Selects pattern card (BARE/FAST/FUSED)
    - ChronoTrigger: Manages temporal control
    - ResonanceShed: Creates feature interference
    - WarpSpace: Tensioned manifold for computation
    - ConvergenceEngine: Collapses decisions
    - Spacetime: Woven output with trace
    """

    # Use FAST mode (balanced speed/quality)
    print("Creating AutoSpinOrchestrator with FAST mode...")
    orchestrator = await AutoSpinOrchestrator.from_text(
        text=knowledge,
        config=Config.fast()
    )

    print("✓ Knowledge base loaded and indexed")
    print(f"  Memory shards created from text")
    print(f"  Ready for queries")

    # ========================================================================
    # STEP 2: Ask questions
    # ========================================================================
    print("\n[STEP 2] Querying knowledge base...")
    print("-" * 80)

    queries = [
        "What is HoloLoom?",
        "What is the weaving metaphor?",
        "What is SpinningWheel used for?",
        "Tell me about the weaving architecture components"
    ]

    for idx, question in enumerate(queries, 1):
        print(f"\n[Query {idx}] {question}")
        print()

        query = Query(text=question)
        result = await orchestrator.process(query)

        print(f"Response:")
        print(f"  Tool: {result.tool}")
        print(f"  Output: {result.output[:200]}...")

        if hasattr(result, 'context') and result.context:
            print(f"  Context shards used: {len(result.context.shards)}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("QUICKSTART COMPLETE")
    print("=" * 80)
    print("✓ Loaded text into memory")
    print("✓ Processed queries with context")
    print("✓ Retrieved relevant information")
    print()
    print("Next steps:")
    print("  - Try demos/02_web_to_memory.py for web scraping")
    print("  - Try demos/03_conversational.py for chat interface")
    print("  - See INTEGRATION_SPRINT.md for full architecture")
    print("=" * 80)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
