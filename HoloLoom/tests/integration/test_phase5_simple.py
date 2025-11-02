#!/usr/bin/env python3
"""Simple Phase 5 integration test."""

import sys
sys.path.insert(0, '.')

import asyncio
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

async def test():
    # Create minimal config
    config = Config.bare()  # Use BARE pattern (96d) for simpler debugging
    config.enable_linguistic_gate = True
    config.linguistic_mode = "disabled"
    config.use_compositional_cache = True

    # Create minimal shards
    shards = [
        MemoryShard(
            id="test_shard",
            text="Thompson Sampling is an algorithm",
            episode="test",
            entities=["Thompson Sampling"],
            motifs=["algorithm"],
            metadata={"category": "AI"}
        )
    ]

    # Test weaving
    async with WeavingOrchestrator(cfg=config, shards=shards) as wo:
        result = await wo.weave(Query(text="test query"))
        print(f"✅ SUCCESS: Got spacetime result")
        print(f"   Context shards: {result.trace.context_shards_count}")
        return result

if __name__ == "__main__":
    asyncio.run(test())
