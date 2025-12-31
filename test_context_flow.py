#!/usr/bin/env python3
"""
Live Test: Sense → Recall → Pack Flow
Tests what's actually working vs documented.
"""
import asyncio
import sys

# Test results
results = {}

async def test_1_basic_memory():
    """Test 1: Basic experience/recall (should work)"""
    print("\n" + "="*60)
    print("TEST 1: Basic Memory (experience/recall)")
    print("="*60)

    try:
        from HoloLoom import HoloLoom

        loom = HoloLoom()

        # Experience
        mem = await loom.experience("Thompson Sampling balances exploration and exploitation")
        print(f"✅ Experience worked: {mem.id}")

        # Recall
        memories = await loom.recall("What is Thompson Sampling?", limit=5)
        print(f"✅ Recall worked: {len(memories)} memories found")

        for m in memories[:3]:
            print(f"   - {m.text[:60]}...")

        results['basic_memory'] = True
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        results['basic_memory'] = False
        return False


async def test_2_context_packing_standalone():
    """Test 2: Context packing standalone (should work)"""
    print("\n" + "="*60)
    print("TEST 2: Context Packing Standalone")
    print("="*60)

    try:
        from HoloLoom.context_packing import ContextPacker, ContextPackerConfig
        import networkx as nx

        # Create simple test graph
        graph = nx.MultiDiGraph()
        graph.add_node("thompson", text="Thompson Sampling is a Bayesian method")
        graph.add_node("bayesian", text="Bayesian methods use priors")
        graph.add_node("bandit", text="Multi-armed bandits are exploration problems")
        graph.add_edge("thompson", "bayesian", type="USES")
        graph.add_edge("thompson", "bandit", type="PART_OF")

        config = ContextPackerConfig.balanced()
        packer = ContextPacker(config)

        result = packer.pack(
            query="What is Thompson Sampling?",
            candidate_nodes=["thompson", "bayesian", "bandit"],
            graph=graph,
            target_tokens=1000
        )

        print(f"✅ Packing worked:")
        print(f"   Original: {result.original_count} nodes")
        print(f"   Compressed: {result.compressed_count} nodes")
        print(f"   Ratio: {result.compression_ratio:.1%}")

        results['context_packing_standalone'] = True
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        results['context_packing_standalone'] = False
        return False


async def test_3_weaving_orchestrator():
    """Test 3: Full weaving orchestrator with beta wave packing"""
    print("\n" + "="*60)
    print("TEST 3: Weaving Orchestrator (full pipeline)")
    print("="*60)

    try:
        from HoloLoom.config import Config
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.protocols.types import Query

        config = Config.fast()
        config.enable_beta_wave_packing = True  # Enable packing

        print(f"   Beta wave packing enabled: {config.enable_beta_wave_packing}")

        # Create test shards
        from HoloLoom.protocols.types import MemoryShard
        shards = [
            MemoryShard(
                id="shard_1",
                content="Thompson Sampling balances exploration and exploitation",
                timestamp=1000.0,
                source="test"
            ),
            MemoryShard(
                id="shard_2",
                content="It uses Bayesian priors for decision making",
                timestamp=1001.0,
                source="test"
            )
        ]

        async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
            query = Query(text="What is Thompson Sampling?")
            spacetime = await orchestrator.weave(query)

            print(f"✅ Weaving worked:")
            print(f"   Response: {spacetime.response[:100]}...")
            print(f"   Confidence: {spacetime.confidence:.2f}")

            # Check if packing was used
            packing_stats = spacetime.metadata.get('packing_stats')
            if packing_stats:
                print(f"   ✅ Packing applied: {packing_stats}")
            else:
                print(f"   ⚠️  Packing NOT applied (check spring_engine)")
                print(f"   Metadata keys: {list(spacetime.metadata.keys())}")

        results['weaving_orchestrator'] = True
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        results['weaving_orchestrator'] = False
        return False


async def test_4_beta_wave_packer():
    """Test 4: BetaWaveContextPacker directly"""
    print("\n" + "="*60)
    print("TEST 4: BetaWaveContextPacker Direct")
    print("="*60)

    try:
        from HoloLoom.memory.awareness.beta_wave_packer import (
            BetaWaveContextPacker, TokenBudget
        )
        print("✅ BetaWaveContextPacker imported successfully")

        # Try to create it (needs spring_engine)
        from HoloLoom.memory.spring_dynamics import SpringDynamics, SpringConfig
        from HoloLoom.memory.graph import KG, KGEdge

        kg = KG()
        kg.add_edge(KGEdge("thompson", "bayesian", "USES", 1.0))

        spring_config = SpringConfig()
        spring_engine = SpringDynamics(kg, spring_config)

        budget = TokenBudget(total=4000)
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=budget
        )

        print(f"✅ BetaWaveContextPacker created with spring_engine")
        results['beta_wave_packer'] = True
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        results['beta_wave_packer'] = False
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("LIVE TEST: Sense → Recall → Pack Flow")
    print("="*60)

    await test_1_basic_memory()
    await test_2_context_packing_standalone()
    await test_3_weaving_orchestrator()
    await test_4_beta_wave_packer()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - Flow is working!")
    else:
        print("SOME TESTS FAILED - Need wiring")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
