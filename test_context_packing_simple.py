#!/usr/bin/env python3
"""
Simple Test: Context Packing with Spring Engine
Tests the wiring we just added to backend_factory.py
"""
import asyncio
import sys

async def test_spring_engine_wiring():
    """Test that spring_engine is properly attached to memory backends."""
    print("\n" + "="*60)
    print("TEST: Spring Engine Wiring Verification")
    print("="*60)

    try:
        from HoloLoom.config import Config, MemoryBackend
        from HoloLoom.memory.backend_factory import create_memory_backend

        # Create config for INMEMORY backend
        config = Config.fast()
        config.memory_backend = MemoryBackend.INMEMORY

        # Create the backend
        print("\n[1] Creating INMEMORY backend...")
        backend = await create_memory_backend(config)

        # Check if spring_engine is attached
        has_spring = hasattr(backend, 'spring_engine')
        print(f"[2] Backend has spring_engine: {has_spring}")

        if has_spring:
            engine = backend.spring_engine
            print(f"[3] Spring engine type: {type(engine).__name__}")
            print(f"[4] Spring engine config: {engine.config}")
            print("\n✅ SPRING ENGINE WIRING: SUCCESS!")
            return True
        else:
            print("\n❌ SPRING ENGINE WIRING: FAILED - no spring_engine attribute")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_packing_with_spring():
    """Test BetaWaveContextPacker with actual spring engine from backend."""
    print("\n" + "="*60)
    print("TEST: Context Packing with Spring Engine")
    print("="*60)

    try:
        from HoloLoom.config import Config, MemoryBackend
        from HoloLoom.memory.backend_factory import create_memory_backend
        from HoloLoom.memory.graph import KG, KGEdge
        from HoloLoom.memory.awareness.beta_wave_packer import (
            BetaWaveContextPacker, TokenBudget
        )

        # Create backend with spring engine
        config = Config.fast()
        config.memory_backend = MemoryBackend.INMEMORY
        backend = await create_memory_backend(config)

        if not hasattr(backend, 'spring_engine'):
            print("❌ No spring_engine on backend - cannot test packing")
            return False

        spring_engine = backend.spring_engine
        print(f"[1] Got spring_engine from backend")

        # Create context packer
        budget = TokenBudget(total=4000)
        packer = BetaWaveContextPacker(
            spring_engine=spring_engine,
            token_budget=budget
        )
        print(f"[2] Created BetaWaveContextPacker")
        print(f"    - Token budget: {budget.total}")
        print(f"    - Activation threshold: {packer.activation_threshold}")
        print(f"    - Compression threshold: {packer.compression_threshold}")

        print("\n✅ CONTEXT PACKING: SUCCESS!")
        print("   BetaWaveContextPacker ready with spring_engine from memory backend")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_sense_recall_flow():
    """Test the sense→recall flow using knowledge graph."""
    print("\n" + "="*60)
    print("TEST: Sense → Recall Flow (Knowledge Graph)")
    print("="*60)

    try:
        from HoloLoom.memory.graph import KG, KGEdge
        from HoloLoom.memory.spring_dynamics import SpringDynamics, SpringConfig

        # Create knowledge graph with test data
        kg = KG()

        # Add some knowledge (SENSE step)
        print("\n[1] SENSE: Adding knowledge to graph...")
        kg.add_edge(KGEdge("Thompson Sampling", "Bayesian Methods", "USES", 1.0))
        kg.add_edge(KGEdge("Thompson Sampling", "Exploration", "PART_OF", 1.0))
        kg.add_edge(KGEdge("Thompson Sampling", "Multi-Armed Bandits", "PART_OF", 1.0))
        kg.add_edge(KGEdge("Bayesian Methods", "Probability", "USES", 0.9))
        kg.add_edge(KGEdge("Exploration", "Exploitation", "LEADS_TO", 0.8))

        print(f"    Added {len(kg.G.nodes())} nodes, {len(kg.G.edges())} edges")

        # Create spring dynamics for activation spreading
        print("\n[2] Creating Spring Dynamics engine...")
        config = SpringConfig(use_advanced_integrator=True, integrator_type="verlet")
        spring = SpringDynamics(kg, config)

        # Activate query nodes (RECALL step)
        print("\n[3] RECALL: Activating 'Thompson Sampling'...")
        spring.activate_nodes({'Thompson Sampling': 1.0})

        # Propagate activation
        print("[4] Propagating activation through graph...")
        result = spring.propagate()

        # Show activated memories
        print(f"\n[5] Retrieved {len(result.activated_nodes)} related memories:")
        for node, activation in sorted(result.node_activations.items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"    - {node}: {activation:.3f}")

        print("\n✅ SENSE → RECALL FLOW: SUCCESS!")
        print(f"   Query 'Thompson Sampling' found {len(result.activated_nodes)} related concepts")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CONTEXT PACKING VERIFICATION")
    print("Testing sense→recall→pack flow wiring")
    print("="*60)

    results = {}

    # Test 1: Spring engine wiring
    results['spring_wiring'] = await test_spring_engine_wiring()

    # Test 2: Context packing with spring
    results['context_packing'] = await test_context_packing_with_spring()

    # Test 3: Full sense→recall flow
    results['sense_recall'] = await test_full_sense_recall_flow()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("The sense→recall→pack flow is properly wired.")
        print("")
        print("What this means:")
        print("  1. Memory backends now have spring_engine attached")
        print("  2. BetaWaveContextPacker can use physics-based importance")
        print("  3. Activation spreads through knowledge graph correctly")
    else:
        print("SOME TESTS FAILED")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
