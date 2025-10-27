#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Simple Demo - Windows Compatible
==========================================
Demonstrates the mythRL/HoloLoom system without Unicode issues.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.autospin import AutoSpinOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query


async def main():
    """Run a simple demo of the mythRL system."""

    print("=" * 80)
    print("MYTHRL / HOLOLOOM DEMO")
    print("=" * 80)

    # ========================================================================
    # Load knowledge base
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
    - Web pages into MemoryShards
    - Audio transcripts into MemoryShards
    - YouTube videos into MemoryShards
    - Raw text into MemoryShards

    The weaving architecture includes:
    - LoomCommand: Selects pattern card (BARE/FAST/FUSED)
    - ChronoTrigger: Manages temporal control
    - ResonanceShed: Creates feature interference
    - WarpSpace: Tensioned manifold for computation
    - ConvergenceEngine: Collapses decisions
    - Spacetime: Woven output with trace

    mythRL is the software development system that uses HoloLoom for:
    - Intelligent code analysis and pattern detection
    - Context-aware decision making in development workflows
    - Multi-modal knowledge integration (code, docs, conversations)
    - Reinforcement learning for optimizing development processes
    """

    print("Creating AutoSpinOrchestrator with FAST mode...")
    orchestrator = await AutoSpinOrchestrator.from_text(
        text=knowledge,
        config=Config.fast()
    )

    print("[OK] Knowledge base loaded and indexed")
    print("  Memory shards created from text")
    print("  Ready for queries")

    # ========================================================================
    # Ask questions
    # ========================================================================
    print("\n[STEP 2] Querying knowledge base...")
    print("-" * 80)

    queries = [
        "What is HoloLoom?",
        "What is the weaving metaphor?",
        "What is SpinningWheel used for?",
        "What is mythRL and how does it use HoloLoom?"
    ]

    for idx, question in enumerate(queries, 1):
        print(f"\n[Query {idx}] {question}")
        print()

        query = Query(text=question)
        result = await orchestrator.process(query)

        print(f"Response:")

        # Handle both dict and object results
        if isinstance(result, dict):
            tool = result.get('tool', 'unknown')
            output = result.get('output', '')
            context = result.get('context')
            print(f"  Tool selected: {tool}")
            if output:
                output_preview = output[:200] if len(output) > 200 else output
                print(f"  Output: {output_preview}...")
            if context and hasattr(context, 'shards'):
                print(f"  Context shards used: {len(context.shards)}")
        else:
            # Object-style access
            print(f"  Tool: {result.tool}")
            output_preview = result.output[:200] if hasattr(result, 'output') and result.output else "No output"
            print(f"  Output: {output_preview}...")
            if hasattr(result, 'context') and result.context:
                print(f"  Context shards used: {len(result.context.shards)}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("[OK] Loaded text into memory")
    print("[OK] Processed queries with context")
    print("[OK] Retrieved relevant information")
    print()
    print("Next steps:")
    print("  - Try demos/complete_weaving_demo.py for full MCTS + Thompson Sampling")
    print("  - Try the Terminal UI for interactive experience")
    print("  - See CLAUDE.md for full architecture documentation")
    print("=" * 80)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()