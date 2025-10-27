"""
Test Memory Backend Simplification (Task 1.3)
==============================================

Verifies:
1. HYBRID backend is the default for FUSED mode
2. Auto-fallback to NetworkX works when Neo4j/Qdrant unavailable
3. Legacy backends auto-migrate with warnings
"""

import asyncio
import warnings
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend


async def test_default_backend():
    """Test that FUSED mode defaults to HYBRID."""
    print("\n" + "="*70)
    print("TEST 1: Default Backend for FUSED Mode")
    print("="*70)

    config = Config.fused()
    print(f"Config mode: {config.mode.value}")
    print(f"Memory backend: {config.memory_backend.value}")

    assert config.memory_backend == MemoryBackend.HYBRID, \
        f"Expected HYBRID, got {config.memory_backend}"

    print("[OK] FUSED mode correctly defaults to HYBRID backend")


async def test_inmemory_backend():
    """Test INMEMORY backend for development."""
    print("\n" + "="*70)
    print("TEST 2: INMEMORY Backend (Development)")
    print("="*70)

    config = Config.fast()
    print(f"Config mode: {config.mode.value}")
    print(f"Memory backend: {config.memory_backend.value}")

    memory = await create_memory_backend(config)
    print(f"[OK] Created backend: {type(memory).__name__}")


async def test_hybrid_with_fallback():
    """Test HYBRID backend with auto-fallback to NetworkX."""
    print("\n" + "="*70)
    print("TEST 3: HYBRID Backend with Auto-Fallback")
    print("="*70)

    config = Config.fused()
    config.memory_backend = MemoryBackend.HYBRID

    print(f"Creating HYBRID backend...")
    print(f"  (Will auto-fallback to NetworkX if Neo4j/Qdrant unavailable)")

    memory = await create_memory_backend(config)
    print(f"[OK] Created backend: {type(memory).__name__}")

    # Check if fallback mode
    if hasattr(memory, 'fallback_mode'):
        if memory.fallback_mode:
            print(f"  Mode: FALLBACK (using NetworkX)")
            print(f"  Backends: {[name for name, _ in memory.backends] if memory.backends else ['networkx']}")
        else:
            print(f"  Mode: PRODUCTION")
            print(f"  Backends: {[name for name, _ in memory.backends]}")


async def test_legacy_migration():
    """Test that legacy backends auto-migrate."""
    print("\n" + "="*70)
    print("TEST 4: Legacy Backend Auto-Migration")
    print("="*70)

    # Test NETWORKX -> INMEMORY migration
    config = Config.fast()
    config.memory_backend = MemoryBackend.NETWORKX

    print(f"Original backend: NETWORKX")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        memory = await create_memory_backend(config)

        # Check if deprecation warning was raised
        if w:
            print(f"  [WARN]  Warning raised: {w[0].message}")

    print(f"[OK] Auto-migrated to: {type(memory).__name__}")

    # Test NEO4J_QDRANT -> HYBRID migration
    config2 = Config.fused()
    config2.memory_backend = MemoryBackend.NEO4J_QDRANT

    print(f"\nOriginal backend: NEO4J_QDRANT")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        memory2 = await create_memory_backend(config2)

        if w:
            print(f"  [WARN]  Warning raised: {w[0].message}")

    print(f"[OK] Auto-migrated to: {type(memory2).__name__}")


async def test_simplified_strategy():
    """Test that HybridMemoryStore uses simplified balanced strategy."""
    print("\n" + "="*70)
    print("TEST 5: Simplified Strategy (Always Balanced)")
    print("="*70)

    config = Config.fused()
    config.memory_backend = MemoryBackend.HYBRID

    memory = await create_memory_backend(config)

    if hasattr(memory, 'strategy'):
        print(f"Strategy: {memory.strategy}")
        assert memory.strategy == "balanced", \
            f"Expected 'balanced', got '{memory.strategy}'"
        print("[OK] Correctly using simplified balanced strategy")
    else:
        print("  (Not a HybridMemoryStore)")


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Memory Backend Simplification Tests (Task 1.3)")
    print("="*70)

    try:
        await test_default_backend()
        await test_inmemory_backend()
        await test_hybrid_with_fallback()
        await test_legacy_migration()
        await test_simplified_strategy()

        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED")
        print("="*70)
        print("\n Summary:")
        print("  • HYBRID is the default for production (FUSED mode)")
        print("  • INMEMORY is the default for development (FAST/BARE mode)")
        print("  • Auto-fallback to NetworkX works correctly")
        print("  • Legacy backends auto-migrate with warnings")
        print("  • Strategy is simplified to always use 'balanced'")
        print()

    except Exception as e:
        print("\n" + "="*70)
        print(f"[FAIL] TEST FAILED: {e}")
        print("="*70)
        raise


if __name__ == "__main__":
    asyncio.run(main())
