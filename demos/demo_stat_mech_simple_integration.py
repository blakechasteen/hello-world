"""
Simple Statistical Mechanics Integration Test

Quick validation of Phase 5.2 integration without full orchestrator complexity.
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.config import Config
from hololoom.documentation.types import MemoryShard
from hololoom.weaving_orchestrator import WeavingOrchestrator


async def test_basic_integration():
    """Test basic statistical mechanics integration."""
    print("=" * 70)
    print("Phase 5.2 Integration Test")
    print("=" * 70)
    print()

    # Create minimal shards
    shards = [
        MemoryShard(
            id="s1",
            text="Thompson Sampling",
            metadata={"embedding": [1.0, 0.8, 0.2]}
        ),
        MemoryShard(
            id="s2",
            text="Bayesian bandits",
            metadata={"embedding": [0.9, 0.9, 0.1]}
        ),
        MemoryShard(
            id="s3",
            text="Gradient descent",
            metadata={"embedding": [0.1, 0.2, 1.0]}
        ),
    ]

    print(f"Created {len(shards)} test shards")

    # Create config
    config = Config.fast()
    config.enable_semantic_cache = False  # Disable for faster testing

    # Test orchestrator with statistical mechanics
    print("\nInitializing orchestrator with statistical mechanics...")
    async with WeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_statistical_mechanics=True,
        consolidation_interval=60.0  # 1 minute
    ) as orchestrator:
        print("OK Orchestrator initialized")

        # Check stat mech engine
        if orchestrator.stat_mech_engine:
            print("OK Statistical mechanics engine available")
        else:
            print("✗ Statistical mechanics engine NOT initialized")
            return

        # Test conversion
        print("\n Testing shard -> microstate conversion...")
        microstates = orchestrator._shards_to_microstates(shards)
        print(f"OK Converted {len(shards)} shards -> {len(microstates)} microstates")

        # Test consolidation
        print("\nTesting consolidation...")
        macrostates = orchestrator.stat_mech_engine.consolidate_memories(
            microstates,
            num_clusters=2
        )
        print(f"OK Consolidated {len(microstates)} -> {len(macrostates)} clusters")

        for i, macro in enumerate(macrostates):
            print(f"  Cluster {i}: {len(macro.microstates)} members, F={macro.free_energy:.3f}")

        # Test back-conversion
        print("\nTesting macrostate -> shard conversion...")
        consolidated_shards = orchestrator._macrostates_to_shards(macrostates)
        print(f"OK Converted {len(macrostates)} macrostates -> {len(consolidated_shards)} shards")

        print()
        print("=" * 70)
        print("[SUCCESS] Phase 5.2 integration complete!")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_basic_integration())
