#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeavingOrchestrator Demo
========================

Example usage of the WeavingOrchestrator showing all three execution modes:
- BARE: Minimal processing (fastest)
- FAST: Balanced processing
- FUSED: Full processing (highest quality)

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass Phase 4)
Original location: lines 2547-2625 (~79 lines)

Usage:
    python demos/orchestrator_demo.py

Author: Claude Code (Elegance Pass Refactoring - Phase 4)
Date: 2025-11-22
"""

import asyncio
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.protocols.types import Query, MemoryShard


async def main():
    """Example usage of WeavingOrchestrator."""
    print("\n" + "="*80)
    print("HoloLoom Weaving Shuttle - Full Architecture Demo")
    print("="*80 + "\n")

    # Create sample memory shards
    shards = [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="docs",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["ALGORITHM", "OPTIMIZATION"]
        ),
        MemoryShard(
            id="shard_002",
            text="The algorithm balances exploration and exploitation by sampling from posterior distributions.",
            episode="docs",
            entities=["exploration", "exploitation", "posterior"],
            motifs=["ALGORITHM", "PROBABILITY"]
        ),
        MemoryShard(
            id="shard_003",
            text="Hive Jodi has 8 frames of brood and is very active with goldenrod flow.",
            episode="inspection_2025_10_13",
            entities=["Hive Jodi", "brood", "goldenrod"],
            motifs=["HIVE_INSPECTION", "SEASONAL"]
        )
    ]

    # Test all three patterns
    for mode in [ExecutionMode.BARE, ExecutionMode.FAST, ExecutionMode.FUSED]:
        print(f"\n{'='*80}")
        print(f"Testing {mode.value.upper()} Mode")
        print(f"{'='*80}\n")

        # Create config
        if mode == ExecutionMode.BARE:
            config = Config.bare()
        elif mode == ExecutionMode.FAST:
            config = Config.fast()
        else:
            config = Config.fused()

        # Create shuttle
        print("Initializing WeavingOrchestrator...")
        shuttle = WeavingOrchestrator(cfg=config, shards=shards)
        print("Shuttle ready!\n")

        # Process a query
        query = Query(text="What is Thompson Sampling?")
        print(f"Processing query: '{query.text}'")
        print("-" * 80)

        spacetime = await shuttle.weave(query)

        # Print spacetime
        print("\n" + "="*80)
        print("SPACETIME FABRIC")
        print("="*80)
        print(f"Query: {spacetime.query_text}")
        print(f"Tool Used: {spacetime.tool_used}")
        print(f"Confidence: {spacetime.confidence:.2f}")
        print(f"Response: {spacetime.response}")
        print(f"\nTrace:")
        print(f"  Duration: {spacetime.trace.duration_ms:.1f}ms")
        print(f"  Motifs: {len(spacetime.trace.motifs_detected)}")
        print(f"  Scales: {spacetime.trace.embedding_scales_used}")
        print(f"  Threads: {len(spacetime.trace.threads_activated)}")
        print(f"  Context Shards: {spacetime.trace.context_shards_count}")
        print(f"\nStage Timings:")
        for stage, duration in spacetime.trace.stage_durations.items():
            print(f"  {stage:25s}: {duration:6.1f}ms")
        print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
