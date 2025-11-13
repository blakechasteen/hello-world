#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic test for orchestrator initialization timeout issue.
Tests each initialization step to identify bottleneck.
"""
import sys
import os
import asyncio
from datetime import datetime

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, '.')

from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard

async def test_orchestrator_initialization():
    """Test orchestrator initialization step by step."""
    print("=" * 70)
    print("ORCHESTRATOR INITIALIZATION DIAGNOSTIC")
    print("=" * 70)

    # Step 1: Create config
    print("\n[1/5] Creating config...")
    start = datetime.now()
    config = Config.fast()
    elapsed = (datetime.now() - start).total_seconds()
    print(f"  [OK] Config created in {elapsed:.3f}s")

    # Step 2: Create test shards
    print("\n[2/5] Creating test shards...")
    start = datetime.now()
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian exploration strategy",
            entities=["thompson_sampling", "bayesian"],
            metadata={"source": "test"}
        ),
        MemoryShard(
            id="shard_2",
            text="Neural networks learn patterns",
            entities=["neural_network"],
            metadata={"source": "test"}
        )
    ]
    elapsed = (datetime.now() - start).total_seconds()
    print(f"  [OK] Shards created in {elapsed:.3f}s")

    # Step 3: Import orchestrator
    print("\n[3/5] Importing WeavingOrchestrator...")
    start = datetime.now()
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    elapsed = (datetime.now() - start).total_seconds()
    print(f"  [OK] Imported in {elapsed:.3f}s")

    # Step 4: Create orchestrator instance
    print("\n[4/5] Creating orchestrator instance...")
    start = datetime.now()
    try:
        orchestrator = WeavingOrchestrator(cfg=config, shards=shards)
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  [OK] Orchestrator created in {elapsed:.3f}s")
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  [FAIL] FAILED after {elapsed:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Test simple query (with timeout)
    print("\n[5/5] Testing simple weave (timeout: 10s)...")
    start = datetime.now()
    try:
        query = Query(text="What is Thompson Sampling?")

        # Use asyncio.wait_for for timeout
        spacetime = await asyncio.wait_for(
            orchestrator.weave(query),
            timeout=10.0
        )

        elapsed = (datetime.now() - start).total_seconds()
        print(f"  [OK] Weave completed in {elapsed:.3f}s")
        print(f"    Response: {spacetime.response[:100] if spacetime.response else 'None'}...")

        # Clean up
        if hasattr(orchestrator, 'close'):
            await orchestrator.close()

        return True

    except asyncio.TimeoutError:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  [TIMEOUT] TIMEOUT after {elapsed:.3f}s")
        print("  Bottleneck: weave() operation hangs")
        return False
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  [FAIL] FAILED after {elapsed:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_orchestrator_initialization())

    print("\n" + "=" * 70)
    if result:
        print("[SUCCESS] DIAGNOSTIC PASSED - Orchestrator working correctly")
        print("=" * 70)
        sys.exit(0)
    else:
        print("[FAIL] DIAGNOSTIC FAILED - Issue identified above")
        print("=" * 70)
        sys.exit(1)
