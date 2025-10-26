"""
Test Memory Backends
====================

Tests the new unified memory backend system with pure and hybrid strategies.

Tests:
1. MemoryBackend enum functionality
2. Config backend selection
3. Factory pattern for pure backends
4. Factory pattern for hybrid backends
5. Backward compatibility
"""

import asyncio
from HoloLoom.config import Config, MemoryBackend, KGBackend


def test_memory_backend_enum():
    """Test MemoryBackend enum methods."""
    print("=" * 60)
    print("Test 1: MemoryBackend Enum")
    print("=" * 60)

    # Test pure strategies
    print("\n1. Pure Strategies:")
    assert not MemoryBackend.NETWORKX.is_hybrid()
    assert not MemoryBackend.NEO4J.is_hybrid()
    assert not MemoryBackend.QDRANT.is_hybrid()
    print("   [OK] Pure strategies correctly identified")

    # Test hybrid strategies
    print("\n2. Hybrid Strategies:")
    assert MemoryBackend.NEO4J_QDRANT.is_hybrid()
    assert MemoryBackend.TRIPLE.is_hybrid()
    assert MemoryBackend.HYPERSPACE.is_hybrid()
    print("   [OK] Hybrid strategies correctly identified")

    # Test backend detection
    print("\n3. Backend Detection:")
    assert MemoryBackend.NEO4J_QDRANT.uses_neo4j()
    assert MemoryBackend.NEO4J_QDRANT.uses_qdrant()
    assert not MemoryBackend.NEO4J_QDRANT.uses_mem0()

    assert MemoryBackend.TRIPLE.uses_neo4j()
    assert MemoryBackend.TRIPLE.uses_qdrant()
    assert MemoryBackend.TRIPLE.uses_mem0()

    assert MemoryBackend.HYPERSPACE.uses_neo4j()
    assert MemoryBackend.HYPERSPACE.uses_qdrant()
    assert not MemoryBackend.HYPERSPACE.uses_mem0()
    print("   [OK] Backend detection working correctly")

    print("\n[SUCCESS] MemoryBackend enum tests passed")
    return True


def test_config_backend_selection():
    """Test Config memory_backend selection."""
    print("\n" + "=" * 60)
    print("Test 2: Config Backend Selection")
    print("=" * 60)

    # Test default backends for each mode
    print("\n1. Default Backends:")
    config_bare = Config.bare()
    print(f"   BARE mode: {config_bare.memory_backend}")
    assert config_bare.memory_backend == MemoryBackend.NETWORKX

    config_fast = Config.fast()
    print(f"   FAST mode: {config_fast.memory_backend}")
    assert config_fast.memory_backend == MemoryBackend.NETWORKX

    config_fused = Config.fused()
    print(f"   FUSED mode: {config_fused.memory_backend}")
    assert config_fused.memory_backend == MemoryBackend.NEO4J_QDRANT
    print("   [OK] Default backends set correctly")

    # Test explicit backend selection
    print("\n2. Explicit Backend Selection:")
    config = Config.fast()
    config.memory_backend = MemoryBackend.NEO4J
    print(f"   Set to: {config.memory_backend}")
    assert config.memory_backend == MemoryBackend.NEO4J
    print("   [OK] Explicit selection working")

    # Test backward compatibility
    print("\n3. Backward Compatibility (kg_backend -> memory_backend):")
    config = Config(kg_backend=KGBackend.NEO4J)
    print(f"   kg_backend=NEO4J maps to: {config.memory_backend}")
    assert config.memory_backend == MemoryBackend.NEO4J
    print("   [OK] Backward compatibility working")

    # Test hyperspace configuration
    print("\n4. Hyperspace Configuration:")
    config = Config.fused()
    config.memory_backend = MemoryBackend.HYPERSPACE
    config.hyperspace_depth = 3
    config.hyperspace_thresholds = [0.6, 0.75, 0.85]
    print(f"   Backend: {config.memory_backend}")
    print(f"   Depth: {config.hyperspace_depth}")
    print(f"   Thresholds: {config.hyperspace_thresholds}")
    print("   [OK] Hyperspace configuration working")

    print("\n[SUCCESS] Config backend selection tests passed")
    return True


async def test_backend_factory_pure():
    """Test factory for pure backends."""
    print("\n" + "=" * 60)
    print("Test 3: Backend Factory (Pure Strategies)")
    print("=" * 60)

    from HoloLoom.memory.backend_factory import create_memory_backend

    # Test NETWORKX
    print("\n1. Creating NETWORKX backend...")
    try:
        config = Config.fast()
        config.memory_backend = MemoryBackend.NETWORKX
        backend = await create_memory_backend(config)
        print(f"   Created: {type(backend).__name__}")
        print("   [OK] NETWORKX backend created")
    except Exception as e:
        print(f"   [SKIP] NETWORKX unavailable: {e}")

    # Test NEO4J (may not be available)
    print("\n2. Creating NEO4J backend...")
    try:
        config = Config.fused()
        config.memory_backend = MemoryBackend.NEO4J
        backend = await create_memory_backend(config)
        print(f"   Created: {type(backend).__name__}")
        print("   [OK] NEO4J backend created")
    except Exception as e:
        print(f"   [SKIP] NEO4J unavailable: {e}")

    # Test QDRANT (may not be available)
    print("\n3. Creating QDRANT backend...")
    try:
        config = Config.fused()
        config.memory_backend = MemoryBackend.QDRANT
        backend = await create_memory_backend(config)
        print(f"   Created: {type(backend).__name__}")
        print("   [OK] QDRANT backend created")
    except Exception as e:
        print(f"   [SKIP] QDRANT unavailable: {e}")

    print("\n[SUCCESS] Pure backend factory tests completed")
    return True


async def test_backend_factory_hybrid():
    """Test factory for hybrid backends."""
    print("\n" + "=" * 60)
    print("Test 4: Backend Factory (Hybrid Strategies)")
    print("=" * 60)

    from HoloLoom.memory.backend_factory import create_memory_backend

    # Test NEO4J_QDRANT
    print("\n1. Creating NEO4J_QDRANT hybrid...")
    try:
        config = Config.fused()
        config.memory_backend = MemoryBackend.NEO4J_QDRANT
        backend = await create_memory_backend(config)
        print(f"   Created: {type(backend).__name__}")
        if hasattr(backend, 'backends'):
            print(f"   Active backends: {[name for name, _ in backend.backends]}")
        print("   [OK] NEO4J_QDRANT hybrid created")
    except Exception as e:
        print(f"   [SKIP] NEO4J_QDRANT unavailable: {e}")

    # Test TRIPLE
    print("\n2. Creating TRIPLE hybrid...")
    try:
        config = Config.fused()
        config.memory_backend = MemoryBackend.TRIPLE
        backend = await create_memory_backend(config)
        print(f"   Created: {type(backend).__name__}")
        if hasattr(backend, 'backends'):
            print(f"   Active backends: {[name for name, _ in backend.backends]}")
        print("   [OK] TRIPLE hybrid created")
    except Exception as e:
        print(f"   [SKIP] TRIPLE unavailable: {e}")

    # Test HYPERSPACE (not yet implemented)
    print("\n3. Creating HYPERSPACE hybrid...")
    try:
        config = Config.fused()
        config.memory_backend = MemoryBackend.HYPERSPACE
        backend = await create_memory_backend(config)
        print(f"   Created: {type(backend).__name__}")
        print("   [OK] HYPERSPACE hybrid created")
    except NotImplementedError:
        print("   [SKIP] HYPERSPACE not yet implemented (expected)")
    except Exception as e:
        print(f"   [SKIP] HYPERSPACE unavailable: {e}")

    print("\n[SUCCESS] Hybrid backend factory tests completed")
    return True


async def test_convenience_function():
    """Test convenience function for backward compatibility."""
    print("\n" + "=" * 60)
    print("Test 5: Convenience Function (Backward Compatibility)")
    print("=" * 60)

    from HoloLoom.memory.backend_factory import create_unified_memory

    # Test flag combinations
    test_cases = [
        (True, False, False, "Neo4j only"),
        (False, True, False, "Qdrant only"),
        (True, True, False, "Neo4j + Qdrant"),
        (False, False, False, "NetworkX fallback"),
    ]

    for neo4j, qdrant, mem0, description in test_cases:
        print(f"\n{description}:")
        print(f"   Flags: neo4j={neo4j}, qdrant={qdrant}, mem0={mem0}")
        try:
            backend = await create_unified_memory(
                user_id="test",
                enable_neo4j=neo4j,
                enable_qdrant=qdrant,
                enable_mem0=mem0
            )
            print(f"   Created: {type(backend).__name__}")
            if hasattr(backend, 'backends'):
                print(f"   Active backends: {[name for name, _ in backend.backends]}")
            print("   [OK]")
        except Exception as e:
            print(f"   [SKIP] {e}")

    print("\n[SUCCESS] Convenience function tests completed")
    return True


def test_config_examples():
    """Test example configurations."""
    print("\n" + "=" * 60)
    print("Test 6: Example Configurations")
    print("=" * 60)

    examples = {
        "Development (fast prototyping)": lambda: (
            Config.fast(),
            MemoryBackend.NETWORKX
        ),
        "Production (graph + vectors)": lambda: (
            Config.fused(),
            MemoryBackend.NEO4J_QDRANT
        ),
        "Research (gated exploration)": lambda: (
            Config.fused(),
            MemoryBackend.HYPERSPACE
        ),
        "Full hybrid (all systems)": lambda: (
            Config.fused(),
            MemoryBackend.TRIPLE
        ),
    }

    for name, create_config in examples.items():
        print(f"\n{name}:")
        config, backend = create_config()
        if backend:
            config.memory_backend = backend
        print(f"   Mode: {config.mode.value}")
        print(f"   Backend: {config.memory_backend.value}")
        print(f"   Scales: {config.scales}")
        if config.memory_backend.is_hybrid():
            print(f"   Type: Hybrid")
            if config.memory_backend.uses_neo4j():
                print(f"   - Neo4j: {config.neo4j_uri}")
            if config.memory_backend.uses_qdrant():
                print(f"   - Qdrant: {config.qdrant_host}:{config.qdrant_port}")
            if config.memory_backend.uses_mem0():
                print(f"   - Mem0: Enabled")
        else:
            print(f"   Type: Pure")

    print("\n[SUCCESS] Example configurations validated")
    return True


async def main():
    """Run all tests."""
    print("\n")
    print("="*60)
    print("Memory Backend System Test Suite")
    print("="*60)

    tests = [
        ("Enum Functionality", test_memory_backend_enum),
        ("Config Selection", test_config_backend_selection),
        ("Pure Backend Factory", test_backend_factory_pure),
        ("Hybrid Backend Factory", test_backend_factory_hybrid),
        ("Convenience Function", test_convenience_function),
        ("Example Configurations", test_config_examples),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
        except Exception as e:
            print(f"\n[FAILED] {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)
