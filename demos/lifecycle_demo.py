#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Lifecycle Management Demo
===================================
Demonstrates proper resource management with async context managers.

This demo shows:
1. Using WeavingShuttle with async context manager
2. Automatic cleanup on exit
3. Background task tracking
4. Reflection buffer persistence
5. Manual cleanup methods

Author: Claude Code
Date: 2025-10-26
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.Documentation.types import Query, MemoryShard


# ============================================================================
# Test Data
# ============================================================================

def create_test_shards():
    """Create sample memory shards for testing."""
    return [
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
            text="Lifecycle management ensures proper cleanup of resources.",
            episode="docs",
            entities=["lifecycle", "cleanup", "resources"],
            motifs=["SYSTEM_DESIGN", "BEST_PRACTICES"]
        )
    ]


# ============================================================================
# Demo 1: Using Async Context Manager (Recommended)
# ============================================================================

async def demo_context_manager():
    """
    Demo 1: Using async context manager for automatic cleanup.

    This is the recommended pattern for production code.
    """
    print("\n" + "="*80)
    print("DEMO 1: Async Context Manager (Automatic Cleanup)")
    print("="*80 + "\n")

    config = Config.fast()
    shards = create_test_shards()

    # Using async context manager - resources automatically cleaned up
    async with WeavingShuttle(cfg=config, shards=shards, enable_reflection=True) as shuttle:
        print("[OK] WeavingShuttle created (inside context manager)")

        # Process a query
        query = Query(text="What is lifecycle management?")
        print(f"\n[QUERY] Processing: '{query.text}'")

        spacetime = await shuttle.weave(query)

        print(f"\n[OK] Weaving complete!")
        print(f"   Tool: {spacetime.tool_used}")
        print(f"   Confidence: {spacetime.confidence:.2f}")
        print(f"   Duration: {spacetime.trace.duration_ms:.1f}ms")

        # Reflect on the result
        await shuttle.reflect(spacetime, feedback={"helpful": True, "rating": 5})
        print(f"\n[OK] Reflection stored")

        # Get metrics
        metrics = shuttle.get_reflection_metrics()
        if metrics:
            print(f"\n[METRICS] Reflection Metrics:")
            print(f"   Total cycles: {metrics['total_cycles']}")
            print(f"   Success rate: {metrics['success_rate']:.1%}")

    # Context manager automatically closes resources here!
    print("\n[OK] Context manager exited - resources automatically cleaned up!")


# ============================================================================
# Demo 2: Manual Cleanup
# ============================================================================

async def demo_manual_cleanup():
    """
    Demo 2: Manual cleanup without context manager.

    Shows explicit close() call for cases where context manager isn't suitable.
    """
    print("\n" + "="*80)
    print("DEMO 2: Manual Cleanup (Explicit close())")
    print("="*80 + "\n")

    config = Config.bare()
    shards = create_test_shards()

    # Create shuttle without context manager
    shuttle = WeavingShuttle(cfg=config, shards=shards, enable_reflection=True)
    print("[OK] WeavingShuttle created (no context manager)")

    try:
        # Process query
        query = Query(text="What is Thompson Sampling?")
        print(f"\n[QUERY] Processing: '{query.text}'")

        spacetime = await shuttle.weave(query)

        print(f"\n[OK] Weaving complete!")
        print(f"   Tool: {spacetime.tool_used}")
        print(f"   Confidence: {spacetime.confidence:.2f}")

    finally:
        # IMPORTANT: Manually close resources
        print("\n[CLEANUP] Manually closing resources...")
        await shuttle.close()
        print("[OK] Resources closed!")


# ============================================================================
# Demo 3: Background Task Tracking
# ============================================================================

async def demo_background_tasks():
    """
    Demo 3: Background task tracking and cleanup.

    Shows how background tasks are automatically cancelled on shutdown.
    """
    print("\n" + "="*80)
    print("DEMO 3: Background Task Tracking")
    print("="*80 + "\n")

    config = Config.fast()
    shards = create_test_shards()

    async def mock_background_work(name: str, duration: float):
        """Simulated background task."""
        print(f"   [START] Background task '{name}' started")
        try:
            await asyncio.sleep(duration)
            print(f"   [OK] Background task '{name}' completed")
        except asyncio.CancelledError:
            print(f"   [CANCEL] Background task '{name}' cancelled")
            raise

    async with WeavingShuttle(cfg=config, shards=shards, enable_reflection=False) as shuttle:
        print("[OK] WeavingShuttle created")

        # Spawn some background tasks
        print("\n[SPAWN] Spawning background tasks...")
        task1 = shuttle.spawn_background_task(mock_background_work("task-1", 10.0))
        task2 = shuttle.spawn_background_task(mock_background_work("task-2", 15.0))
        task3 = shuttle.spawn_background_task(mock_background_work("task-3", 20.0))

        print(f"   Created {len(shuttle._background_tasks)} background tasks")

        # Do some weaving
        query = Query(text="Test query")
        print(f"\n[QUERY] Processing query while tasks run in background...")
        spacetime = await shuttle.weave(query)
        print(f"   [OK] Weaving done (tool: {spacetime.tool_used})")

        # Exit context manager - tasks will be cancelled
        print("\n[EXIT] Exiting context manager (will cancel background tasks)...")

    # All background tasks are cancelled automatically
    print("[OK] All background tasks cancelled on exit!")


# ============================================================================
# Demo 4: Multiple Operations with Reflection
# ============================================================================

async def demo_multiple_operations():
    """
    Demo 4: Multiple weaving operations with reflection persistence.

    Shows how reflection buffer persists data across multiple cycles.
    """
    print("\n" + "="*80)
    print("DEMO 4: Multiple Operations with Reflection")
    print("="*80 + "\n")

    config = Config.fused()
    shards = create_test_shards()

    queries = [
        ("What is Thompson Sampling?", {"helpful": True, "rating": 5}),
        ("Explain lifecycle management", {"helpful": True, "rating": 4}),
        ("How does exploration work?", {"helpful": False, "rating": 2}),
    ]

    async with WeavingShuttle(cfg=config, shards=shards, enable_reflection=True) as shuttle:
        print(f"[OK] WeavingShuttle created")
        print(f"   Reflection enabled: {shuttle.enable_reflection}")
        print(f"   Reflection path: {shuttle.reflection_buffer.persist_path}")

        for i, (query_text, feedback) in enumerate(queries, 1):
            print(f"\n{'-'*80}")
            print(f"Query {i}/{len(queries)}: '{query_text}'")

            query = Query(text=query_text)
            spacetime = await shuttle.weave_and_reflect(query, feedback=feedback)

            print(f"   Tool: {spacetime.tool_used}")
            print(f"   Confidence: {spacetime.confidence:.2f}")
            print(f"   Duration: {spacetime.trace.duration_ms:.1f}ms")
            print(f"   Feedback: {feedback}")

        # Check reflection metrics
        metrics = shuttle.get_reflection_metrics()
        if metrics:
            print(f"\n{'='*80}")
            print("[METRICS] Final Reflection Metrics")
            print(f"{'='*80}")
            print(f"Total cycles: {metrics['total_cycles']}")
            print(f"Success rate: {metrics['success_rate']:.1%}")
            print(f"\nTool success rates:")
            for tool, rate in metrics['tool_success_rates'].items():
                print(f"  {tool:20s}: {rate:.1%}")

    # Reflection buffer data persisted to disk
    print("\n[OK] Reflection data persisted to disk!")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all lifecycle demos."""
    print("\n" + "="*80)
    print("HoloLoom Lifecycle Management Demo")
    print("="*80)

    demos = [
        ("Context Manager", demo_context_manager),
        ("Manual Cleanup", demo_manual_cleanup),
        ("Background Tasks", demo_background_tasks),
        ("Multiple Operations", demo_multiple_operations),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n[ERROR] Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("[OK] All demos complete!")
    print("="*80 + "\n")

    print("Key Takeaways:")
    print("1. Use 'async with' for automatic cleanup (recommended)")
    print("2. Call close() manually if context manager not suitable")
    print("3. Background tasks are tracked and cancelled automatically")
    print("4. Reflection buffer persists data to disk on exit")
    print("5. Multiple close() calls are safe (idempotent)")


if __name__ == "__main__":
    asyncio.run(main())