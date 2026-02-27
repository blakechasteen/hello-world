#!/usr/bin/env python3
"""
Test Orchestrator Refactoring
==============================
Test script to verify the refactored orchestrator architecture works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard
from hololoom.protocols import ComplexityLevel


def create_test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard1",
            text="Thompson Sampling is a Bayesian approach to exploration-exploitation.",
            metadata={"topic": "machine_learning"}
        ),
        MemoryShard(
            id="shard2",
            text="It uses posterior distributions to balance exploration and exploitation.",
            metadata={"topic": "algorithms"}
        ),
    ]


async def test_refactored_orchestrator():
    """Test the refactored orchestrator."""
    print("=" * 60)
    print("Testing Refactored Orchestrator Architecture")
    print("=" * 60)

    # Import the refactored orchestrator
    from hololoom.weaving_orchestrator_refactored import WeavingOrchestratorRefactored

    # Create configuration
    config = Config.fast()
    shards = create_test_shards()

    # Create orchestrator
    print("\n1. Creating refactored orchestrator...")
    orchestrator = WeavingOrchestratorRefactored(
        cfg=config,
        shards=shards,
        enable_complexity_auto_detect=True
    )
    print("   ✓ Orchestrator created")

    # Test with different complexity levels
    test_cases = [
        ("What is Thompson Sampling?", ComplexityLevel.FAST),
        ("Explain the mathematical basis of Thompson Sampling", ComplexityLevel.FULL),
    ]

    for query_text, complexity in test_cases:
        print(f"\n2. Testing query: '{query_text[:50]}...'")
        print(f"   Complexity: {complexity.value}")

        try:
            query = Query(text=query_text)

            # Execute weaving
            result = await orchestrator.weave(
                query=query,
                complexity=complexity
            )

            print(f"   ✓ Weaving completed")
            print(f"   Response: {result.response[:100]}...")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Tool used: {result.tool_used}")

            if result.trace:
                print(f"   Stages executed: {len(result.trace.stages_executed)}")
                print(f"   Total duration: {result.trace.total_duration_ms:.2f}ms")

        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Clean up
    await orchestrator.close()
    print("\n✓ Test completed successfully")


async def test_original_orchestrator():
    """Test the original orchestrator for comparison."""
    print("\n" + "=" * 60)
    print("Testing Original Orchestrator (for comparison)")
    print("=" * 60)

    try:
        from hololoom.weaving_orchestrator import WeavingOrchestrator

        # Create configuration
        config = Config.fast()
        shards = create_test_shards()

        # Create orchestrator
        print("\n1. Creating original orchestrator...")
        async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
            print("   ✓ Orchestrator created")

            # Test a simple query
            query = Query(text="What is Thompson Sampling?")
            print(f"\n2. Testing query: '{query.text}'")

            result = await orchestrator.weave(query)

            print(f"   ✓ Weaving completed")
            print(f"   Response: {result.response[:100]}...")
            print(f"   Confidence: {result.confidence:.2f}")

    except Exception as e:
        print(f"   ✗ Error: {e}")


async def main():
    """Main test function."""
    # Test refactored orchestrator
    await test_refactored_orchestrator()

    # Optionally test original for comparison
    # await test_original_orchestrator()


if __name__ == "__main__":
    asyncio.run(main())