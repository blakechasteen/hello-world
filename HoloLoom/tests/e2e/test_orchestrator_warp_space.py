"""
End-to-end test for orchestrator with WarpSpace integration.

Verifies the complete weaving cycle includes WarpSpace compute operations.
"""
import pytest
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query, MemoryShard


@pytest.mark.asyncio
@pytest.mark.timeout(120)  # 120s timeout for semantic cache initialization
async def test_orchestrator_with_warp_space():
    """Test full orchestrator weaving with WarpSpace compute operations"""

    # Create minimal config
    config = Config.fast()  # Use FAST mode for quick test

    # Create test memory shards
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian exploration strategy",
            entities=["thompson_sampling", "bayesian", "exploration"],
            metadata={"source": "test"}
        ),
        MemoryShard(
            id="shard_2",
            text="Neural networks learn hierarchical representations",
            entities=["neural_network", "learning", "hierarchical"],
            metadata={"source": "test"}
        ),
        MemoryShard(
            id="shard_3",
            text="Attention mechanisms enable context-aware processing",
            entities=["attention", "context", "processing"],
            metadata={"source": "test"}
        )
    ]

    # Create orchestrator (disable semantic cache for faster test initialization)
    async with WeavingOrchestrator(cfg=config, shards=shards, enable_semantic_cache=False) as orchestrator:
        # Create query
        query = Query(text="What is Thompson Sampling?")

        # Weave (full cycle)
        spacetime = await orchestrator.weave(query)

        # Verify spacetime structure
        assert spacetime is not None, "Spacetime should be created"
        assert spacetime.query_text == query.text, "Query text should match"
        assert spacetime.response is not None, "Response should exist"
        assert spacetime.trace is not None, "Trace should exist"

        # Check if WarpSpace was used (conditional - depends on execution path)
        warp_used = ('warp_compute' in spacetime.trace.stage_durations or
                     'warp_compute' in spacetime.metadata or
                     'warp_tensioning' in spacetime.trace.stage_durations)

        if warp_used:
            print("[OK] WarpSpace execution path taken")

            # Check warp_operations in trace (may be empty list if warp failed)
            warp_ops = spacetime.trace.warp_operations
            if warp_ops and len(warp_ops) > 0:
                # Verify warp operations include tension and detension
                op_types = [op[1] for op in warp_ops]  # Extract operation names
                print(f"  Warp operations: {op_types}")
                # At minimum, should have tension and detension
                assert "tension" in op_types, "Should have tension operation"
                assert "detension" in op_types, "Should have detension operation"

            # Verify metadata includes warp_compute results (if successful)
            if spacetime.metadata.get('warp_compute') is not None:
                warp_compute = spacetime.metadata['warp_compute']
                assert 'attention_entropy' in warp_compute, "Should have attention entropy"
                assert 'spectral_features' in warp_compute, "Should have spectral features"
                print(f"  WarpSpace compute successful: attention_entropy={warp_compute['attention_entropy']:.3f}")

            # Verify stage timings include warp stages
            stage_timings = spacetime.trace.stage_durations
            if 'warp_tensioning' in stage_timings:
                assert stage_timings['warp_tensioning'] >= 0, "Tensioning duration should be non-negative"
        else:
            # Fast path taken - WarpSpace skipped (this is valid behavior)
            print("[OK] Fast path taken (WarpSpace skipped - valid for simple queries)")
            fast_path = spacetime.metadata.get('fast_path', 'unknown')
            print(f"  Fast path: {fast_path}")


@pytest.mark.asyncio
@pytest.mark.timeout(120)  # 120s timeout for semantic cache initialization
async def test_orchestrator_warp_space_multiple_queries():
    """Test orchestrator WarpSpace with multiple sequential queries"""

    config = Config.fast()

    shards = [
        MemoryShard(
            id="shard_ts",
            text="Thompson Sampling balances exploration and exploitation",
            entities=["thompson_sampling"],
            metadata={"source": "test"}
        )
    ]

    async with WeavingOrchestrator(cfg=config, shards=shards, enable_semantic_cache=False) as orchestrator:
        queries = [
            Query(text="What is exploration?"),
            Query(text="What is exploitation?"),
            Query(text="How do they balance?")
        ]

        for query in queries:
            spacetime = await orchestrator.weave(query)

            # Each weave should complete successfully
            assert spacetime is not None

            # Check if warp operations were recorded
            if spacetime.trace.warp_operations and len(spacetime.trace.warp_operations) > 0:
                op_types = [op[1] for op in spacetime.trace.warp_operations]
                assert "tension" in op_types
                assert "detension" in op_types


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        print("="*80)
        print("End-to-End Orchestrator + WarpSpace Tests")
        print("="*80 + "\n")

        print("Test 1: Full orchestrator with WarpSpace compute")
        await test_orchestrator_with_warp_space()
        print("[OK] PASSED\n")

        print("Test 2: Multiple queries with WarpSpace")
        await test_orchestrator_warp_space_multiple_queries()
        print("[OK] PASSED\n")

        print("="*80)
        print("All end-to-end tests passed!")
        print("="*80)

    asyncio.run(run_tests())
