"""
Test Backend Factory After 6-Step Refinement

Validates that all refinements maintain functionality:
- INMEMORY backend creation
- HYBRID backend creation with auto-fallback
- Validation and error handling
- Health check improvements
"""

import asyncio
from hololoom.config import Config, MemoryBackend
from hololoom.memory.backend_factory import create_memory_backend, check_backend_health
from hololoom.memory.protocol import Memory, MemoryQuery
from datetime import datetime


async def test_inmemory_backend():
    """Test INMEMORY backend (NetworkX)."""
    print("\n" + "="*70)
    print("TEST 1: INMEMORY Backend (NetworkX)")
    print("="*70)

    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY

    try:
        backend = await create_memory_backend(config)
        print(f"✓ Created INMEMORY backend: {type(backend).__name__}")

        # Health check
        health = await check_backend_health(backend)
        print(f"✓ Health check: {health.get('status', 'unknown')}")

        # Store a memory
        memory = Memory(
            id="test-inmemory-001",
            text="Test memory for INMEMORY backend",
            timestamp=datetime.now(),
            context={'test': True},
            metadata={'source': 'test'}
        )
        memory_id = await backend.store(memory, user_id="test-user")
        print(f"✓ Stored memory: {memory_id[:16]}...")

        # Recall
        query = MemoryQuery(text="Test memory")
        result = await backend.recall(query, limit=5)
        print(f"✓ Recalled {len(result.memories)} memories")

        return True
    except Exception as e:
        print(f"✗ INMEMORY test failed: {e}")
        return False


async def test_hybrid_backend():
    """Test HYBRID backend with auto-fallback."""
    print("\n" + "="*70)
    print("TEST 2: HYBRID Backend (Neo4j + Qdrant + Auto-Fallback)")
    print("="*70)

    config = Config.fused()
    config.memory_backend = MemoryBackend.HYBRID

    try:
        backend = await create_memory_backend(config)
        print(f"✓ Created HYBRID backend: {type(backend).__name__}")

        # Health check
        health = await check_backend_health(backend)
        print(f"✓ Health check: status={health.get('status')}, mode={health.get('mode')}")

        if 'backends' in health:
            for name, status in health['backends'].items():
                status_str = status.get('status', 'unknown') if isinstance(status, dict) else status
                print(f"  • {name}: {status_str}")

        # Store a memory
        memory = Memory(
            id="test-hybrid-001",
            text="Test memory for HYBRID backend with semantic search capability",
            timestamp=datetime.now(),
            context={'test': True, 'backend': 'hybrid'},
            metadata={'source': 'test', 'has_embedding': False}
        )
        memory_id = await backend.store(memory, user_id="test-user")
        print(f"✓ Stored memory: {memory_id[:16]}...")

        # Recall
        query = MemoryQuery(text="semantic search")
        result = await backend.recall(query, limit=5)
        print(f"✓ Recalled {len(result.memories)} memories")
        print(f"  Strategy: {result.strategy_used}")
        if 'backends' in result.metadata:
            print(f"  Backends queried: {result.metadata['backends']}")

        return True
    except Exception as e:
        print(f"✗ HYBRID test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_validation():
    """Test validation and error handling."""
    print("\n" + "="*70)
    print("TEST 3: Validation and Error Handling")
    print("="*70)

    # Test 1: None config
    try:
        await create_memory_backend(None)
        print("✗ Should have raised ValueError for None config")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected None config: {str(e)[:50]}")

    # Test 2: None backend health check
    health = await check_backend_health(None)
    if health['status'] == 'unhealthy':
        print(f"✓ Correctly handled None backend health check")
    else:
        print(f"✗ Failed to detect None backend")
        return False

    # Test 3: Store with invalid memory
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY
    backend = await create_memory_backend(config)

    try:
        await backend.store(None, user_id="test")
        print("✗ Should have raised error for None memory")
        return False
    except Exception as e:
        print(f"✓ Correctly rejected None memory: {str(e)[:50]}")

    return True


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Backend Factory Refinement - Test Suite")
    print("="*70)
    print("\nValidating 6-step refinement improvements:")
    print("  • Enhanced documentation (ELEGANCE Step 1)")
    print("  • Extracted helper methods (ELEGANCE Step 2)")
    print("  • Emoji logging & structure (ELEGANCE Step 3)")
    print("  • Validation checks (VERIFY Step 4)")
    print("  • Error handling (VERIFY Step 5)")
    print("  • Consistency (VERIFY Step 6)")

    results = []

    # Run tests
    results.append(("INMEMORY Backend", await test_inmemory_backend()))
    results.append(("HYBRID Backend", await test_hybrid_backend()))
    results.append(("Validation", await test_validation()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:30} {status}")

    all_passed = all(passed for _, passed in results)

    print("="*70)
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nBackend Factory refinement is complete and functional!")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review failures above.")
    print()

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
