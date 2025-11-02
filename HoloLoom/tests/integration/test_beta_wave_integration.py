"""
Integration Test: Beta Wave Context Packing in WeavingOrchestrator

Tests the end-to-end integration of physics-based context optimization.

Test Scenarios:
1. Beta wave packing ENABLED with MultiWaveMemoryEngine
2. Beta wave packing DISABLED (fallback to raw shards)
3. Beta wave packing with legacy memory backend (graceful fallback)

Expected Results:
- When enabled: packed_context in metadata, token reduction
- When disabled: packed_context is None, uses raw shards
- Error handling: graceful fallback on failures
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard


def create_test_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Create consistent embedding for test (hash-based)"""
    np.random.seed(hash(text) % (2**32))
    emb = np.random.randn(dim)
    np.random.seed(None)
    return emb / np.linalg.norm(emb)


def create_test_shards() -> list:
    """Create test memory shards"""
    test_data = [
        ('mem1', 'Quantum entanglement enables quantum teleportation between particles.', 2.0),
        ('mem2', 'Bell state measurements are crucial for quantum error correction codes.', 1.8),
        ('mem3', 'Quantum key distribution uses entangled pairs for secure communication.', 1.5),
        ('mem4', 'Machine learning uses gradient descent to optimize parameters.', 0.8),
        ('mem5', 'DNA sequencing has advanced with nanopore technology.', 0.5),
    ]

    shards = []
    for mem_id, text, k in test_data:
        shard = MemoryShard(
            id=mem_id,
            text=text,
            entities=[],
            episode=None,  # Fixed: use correct parameters
            motifs=[]      # Fixed: use correct parameters
        )
        shards.append(shard)

    return shards


async def test_1_beta_wave_packing_enabled():
    """Test 1: Beta wave packing ENABLED (requires MultiWaveMemoryEngine)"""
    print("\n" + "="*80)
    print("TEST 1: Beta Wave Packing ENABLED")
    print("="*80)

    try:
        from HoloLoom.memory.multi_wave_engine import MultiWaveMemoryEngine, MultiWaveConfig
        from HoloLoom.memory.spring_dynamics_engine import SpringEngineConfig
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create config with beta wave packing ENABLED
        config = Config.fast()
        config.enable_beta_wave_packing = True
        config.packing_token_budget = 2000
        config.packing_activation_threshold = 0.3
        config.packing_compression_threshold = 0.7

        # Create MultiWaveMemoryEngine (has spring_engine)
        spring_config = SpringEngineConfig()
        wave_config = MultiWaveConfig()
        memory = MultiWaveMemoryEngine(spring_config, wave_config)

        # Add test memories
        test_data = [
            ('mem1', 'Quantum entanglement enables quantum teleportation.', 2.0),
            ('mem2', 'Bell state measurements for quantum error correction.', 1.8),
            ('mem3', 'Quantum key distribution uses entangled pairs.', 1.5),
        ]

        for mem_id, text, k in test_data:
            embedding = create_test_embedding(text)
            memory.encode_new_memory(mem_id, text, embedding, initial_k=k)

        print(f"✓ Memory engine initialized with {len(test_data)} memories")

        # Create orchestrator with MultiWaveMemoryEngine
        shards = create_test_shards()
        orchestrator = WeavingOrchestrator(cfg=config, shards=shards, memory=memory)

        print(f"✓ Orchestrator created with beta wave packing ENABLED")

        # Test query
        query = Query(text="What are the applications of quantum entanglement?")

        # Weave (should use beta wave packing)
        print(f"\n🌊 Weaving query...")
        spacetime = await orchestrator.weave(query)

        # Verify beta wave packing was used
        packed_ctx = spacetime.metadata.get('packed_context')
        packing_stats = spacetime.metadata.get('packing_stats')

        print(f"\n📊 Results:")
        print(f"  Query: {query.text}")
        print(f"  Tool used: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.2f}")

        if packed_ctx:
            print(f"\n✅ Beta Wave Packing ACTIVE:")
            print(f"  Elements included: {packing_stats['elements_included']}")
            print(f"  Elements compressed: {packing_stats['elements_compressed']}")
            print(f"  Elements excluded: {packing_stats['elements_excluded']}")
            print(f"  Total tokens: {packing_stats['total_tokens']}/{packing_stats['budget_available']}")
            print(f"  Avg activation: {packing_stats['avg_activation']:.3f}")
            print(f"  Activation distribution: {packing_stats.get('activation_distribution', {})}")

            # Validate packing worked
            assert packed_ctx is not None, "Packed context should exist"
            assert packing_stats['total_tokens'] <= 2000, "Should respect token budget"
            print(f"\n[PASS] Test 1: Beta wave packing working correctly!")
            return True
        else:
            print(f"\n⚠️  Beta wave packing NOT used (unexpected)")
            error = spacetime.metadata.get('packing_error')
            if error:
                print(f"  Error: {error}")
            print(f"\n[PARTIAL] Test 1: Packing available but not used")
            return False

    except ImportError as e:
        print(f"\n⚠️  MultiWaveMemoryEngine not available: {e}")
        print(f"[SKIP] Test 1: Dependencies missing")
        return None


async def test_2_beta_wave_packing_disabled():
    """Test 2: Beta wave packing DISABLED (fallback to raw shards)"""
    print("\n" + "="*80)
    print("TEST 2: Beta Wave Packing DISABLED")
    print("="*80)

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create config with beta wave packing DISABLED
        config = Config.fast()
        config.enable_beta_wave_packing = False  # Explicitly disabled

        # Create orchestrator with static shards only (no MultiWaveMemoryEngine)
        shards = create_test_shards()
        orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

        print(f"✓ Orchestrator created with beta wave packing DISABLED")

        # Test query
        query = Query(text="What are the applications of quantum entanglement?")

        # Weave (should NOT use beta wave packing)
        print(f"\n🌊 Weaving query...")
        spacetime = await orchestrator.weave(query)

        # Verify beta wave packing was NOT used
        packed_ctx = spacetime.metadata.get('packed_context')

        print(f"\n📊 Results:")
        print(f"  Query: {query.text}")
        print(f"  Tool used: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.2f}")

        if packed_ctx is None:
            print(f"\n✅ Beta Wave Packing DISABLED (as expected):")
            print(f"  Using raw shards: YES")
            print(f"  packed_context: None")
            print(f"\n[PASS] Test 2: Graceful fallback to raw shards!")
            return True
        else:
            print(f"\n❌ Beta wave packing was used (unexpected)")
            print(f"\n[FAIL] Test 2: Should have fallen back to raw shards")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\n[FAIL] Test 2: Exception occurred")
        import traceback
        traceback.print_exc()
        return False


async def test_3_beta_wave_packing_no_spring_engine():
    """Test 3: Beta wave packing with legacy backend (no spring_engine)"""
    print("\n" + "="*80)
    print("TEST 3: Beta Wave Packing with Legacy Backend")
    print("="*80)

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create config with beta wave packing ENABLED
        config = Config.fast()
        config.enable_beta_wave_packing = True  # Enabled, but no spring_engine available

        # Create orchestrator with static shards only (no MultiWaveMemoryEngine)
        shards = create_test_shards()
        orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

        print(f"✓ Orchestrator created with packing ENABLED but no spring_engine")

        # Test query
        query = Query(text="What are the applications of quantum entanglement?")

        # Weave (should fall back gracefully)
        print(f"\n🌊 Weaving query...")
        spacetime = await orchestrator.weave(query)

        # Verify graceful fallback
        packed_ctx = spacetime.metadata.get('packed_context')

        print(f"\n📊 Results:")
        print(f"  Query: {query.text}")
        print(f"  Tool used: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.2f}")

        if packed_ctx is None:
            print(f"\n✅ Graceful Fallback (as expected):")
            print(f"  Packing enabled: YES")
            print(f"  Spring engine available: NO")
            print(f"  Fell back to raw shards: YES")
            print(f"\n[PASS] Test 3: Graceful fallback when spring_engine unavailable!")
            return True
        else:
            print(f"\n❌ Beta wave packing was used (unexpected - no spring_engine)")
            print(f"\n[FAIL] Test 3: Should have fallen back")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\n[FAIL] Test 3: Exception occurred")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests"""
    print("="*80)
    print("BETA WAVE CONTEXT PACKING - INTEGRATION TESTS")
    print("="*80)
    print("\nTesting end-to-end integration with WeavingOrchestrator...")

    results = []

    # Test 1: Beta wave packing enabled
    result1 = await test_1_beta_wave_packing_enabled()
    results.append(('Test 1: Packing Enabled', result1))

    # Test 2: Beta wave packing disabled
    result2 = await test_2_beta_wave_packing_disabled()
    results.append(('Test 2: Packing Disabled', result2))

    # Test 3: Beta wave packing with legacy backend
    result3 = await test_3_beta_wave_packing_no_spring_engine()
    results.append(('Test 3: Legacy Backend', result3))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    total = len(results)

    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{status} {name}")

    print(f"\n📊 Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print(f"\n🎯 ALL TESTS PASSED! Beta wave packing integration is working!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
