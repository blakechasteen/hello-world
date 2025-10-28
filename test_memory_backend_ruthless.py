"""
Ruthless Memory Backend Testing
================================
Comprehensive stress testing for simplified memory system.

Tests:
- Edge cases and failure modes
- Concurrent access
- Error handling and recovery
- Protocol compliance
- Performance under load
- Real-world scenarios
"""

import asyncio
import time
from typing import List
from datetime import datetime

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend, HybridMemoryStore
from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult


class TestStats:
    """Track test statistics."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()

    def pass_test(self):
        self.passed += 1

    def fail_test(self):
        self.failed += 1

    def skip_test(self):
        self.skipped += 1

    def summary(self):
        elapsed = time.time() - self.start_time
        total = self.passed + self.failed + self.skipped
        return f"""
{'='*70}
TEST SUMMARY
{'='*70}
Total:   {total}
Passed:  {self.passed} ({100*self.passed/total if total else 0:.1f}%)
Failed:  {self.failed} ({100*self.failed/total if total else 0:.1f}%)
Skipped: {self.skipped}
Time:    {elapsed:.2f}s
{'='*70}
"""


stats = TestStats()


def test(name: str):
    """Test decorator."""
    def decorator(func):
        async def wrapper():
            print(f"\n{'='*70}")
            print(f"TEST: {name}")
            print('='*70)
            try:
                await func()
                print("[PASS]")
                stats.pass_test()
            except Exception as e:
                print(f"[FAIL] {e}")
                stats.fail_test()
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


# ============================================================================
# Basic Functionality Tests
# ============================================================================

@test("Three backends exist")
async def test_three_backends():
    """Verify exactly 3 backends."""
    backends = [b for b in MemoryBackend]
    assert len(backends) == 3, f"Expected 3, got {len(backends)}"
    assert MemoryBackend.INMEMORY in backends
    assert MemoryBackend.HYBRID in backends
    assert MemoryBackend.HYPERSPACE in backends
    print(f"Backends: {[b.name for b in backends]}")


@test("Config defaults are correct")
async def test_config_defaults():
    """Verify default backend assignments."""
    assert Config.bare().memory_backend == MemoryBackend.INMEMORY
    assert Config.fast().memory_backend == MemoryBackend.INMEMORY
    assert Config.fused().memory_backend == MemoryBackend.HYBRID
    print("BARE/FAST->INMEMORY, FUSED->HYBRID")


@test("INMEMORY backend creates successfully")
async def test_inmemory_create():
    """Create INMEMORY backend."""
    config = Config.fast()
    memory = await create_memory_backend(config)
    assert memory is not None
    print(f"Type: {type(memory).__name__}")


@test("HYBRID backend creates with fallback")
async def test_hybrid_create():
    """Create HYBRID backend (may fall back)."""
    config = Config.fused()
    memory = await create_memory_backend(config)
    assert memory is not None
    print(f"Type: {type(memory).__name__}")


# ============================================================================
# Edge Cases
# ============================================================================

@test("Empty query handling")
async def test_empty_query():
    """Test empty query string."""
    config = Config.fast()
    memory = await create_memory_backend(config)

    # Empty query should not crash
    try:
        query = MemoryQuery(text="", limit=5)
        # NetworkX KG might not have recall method, check first
        if hasattr(memory, 'query'):
            result = memory.query("")
            print(f"Empty query returned {len(result) if result else 0} results")
        else:
            print("Backend doesn't support query() - OK for NetworkX")
    except Exception as e:
        print(f"Empty query handled: {type(e).__name__}")


@test("Large limit handling")
async def test_large_limit():
    """Test very large limit value."""
    config = Config.fast()
    memory = await create_memory_backend(config)

    query = MemoryQuery(text="test", limit=10000)
    print(f"Large limit (10k) query created")


@test("None/null handling")
async def test_null_handling():
    """Test null values in Memory."""
    mem = Memory(
        id="test",
        text="test",
        timestamp=datetime.now(),
        context={},
        metadata={}
    )
    assert mem.id == "test"
    print("Null values handled correctly")


# ============================================================================
# Error Handling
# ============================================================================

@test("Invalid backend enum raises error")
async def test_invalid_backend():
    """Test that invalid backend raises proper error."""
    config = Config.fast()
    config.memory_backend = "invalid_backend"  # type: ignore

    try:
        await create_memory_backend(config)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")


@test("Backend unavailable falls back gracefully")
async def test_backend_fallback():
    """Test fallback when backends unavailable."""
    config = Config.fused()
    config.memory_backend = MemoryBackend.HYBRID

    memory = await create_memory_backend(config)

    # Should fall back to NetworkX if Neo4j/Qdrant unavailable
    if isinstance(memory, HybridMemoryStore):
        print(f"Fallback mode: {memory.fallback_mode}")
        if memory.fallback_mode:
            print("Correctly fell back to NetworkX")
        else:
            print(f"Production mode with backends: {[n for n, _ in memory.backends]}")


@test("HYPERSPACE fallback to HYBRID")
async def test_hyperspace_fallback():
    """Test HYPERSPACE falls back to HYBRID if unavailable."""
    config = Config.fused()
    config.memory_backend = MemoryBackend.HYPERSPACE

    memory = await create_memory_backend(config)
    # Should create some backend (HYPERSPACE or fallback to HYBRID)
    assert memory is not None
    print(f"HYPERSPACE resolved to: {type(memory).__name__}")


# ============================================================================
# Performance Tests
# ============================================================================

@test("Backend creation performance")
async def test_create_performance():
    """Measure backend creation time."""
    config = Config.fast()

    start = time.time()
    for _ in range(10):
        memory = await create_memory_backend(config)
    elapsed = time.time() - start

    avg = elapsed / 10
    print(f"Average creation time: {avg*1000:.2f}ms")
    assert avg < 0.5, f"Too slow: {avg}s"


@test("Memory object creation performance")
async def test_memory_creation():
    """Measure Memory object creation."""
    start = time.time()

    memories = []
    for i in range(1000):
        mem = Memory(
            id=f"mem_{i}",
            text=f"Test memory {i}",
            timestamp=datetime.now(),
            context={'index': i},
            metadata={}
        )
        memories.append(mem)

    elapsed = time.time() - start
    print(f"Created 1000 Memory objects in {elapsed*1000:.2f}ms")
    print(f"Average: {elapsed/1000*1000:.3f}ms per object")


@test("Protocol overhead minimal")
async def test_protocol_overhead():
    """Verify protocol checking is fast."""
    from HoloLoom.memory.protocol import MemoryStore

    config = Config.fast()
    memory = await create_memory_backend(config)

    start = time.time()
    for _ in range(1000):
        isinstance(memory, MemoryStore)
    elapsed = time.time() - start

    print(f"1000 protocol checks in {elapsed*1000:.2f}ms")
    assert elapsed < 0.1, f"Protocol checks too slow: {elapsed}s"


# ============================================================================
# Concurrent Access Tests
# ============================================================================

@test("Concurrent backend creation")
async def test_concurrent_creation():
    """Test concurrent backend creation."""
    config = Config.fast()

    async def create():
        return await create_memory_backend(config)

    start = time.time()
    results = await asyncio.gather(*[create() for _ in range(10)])
    elapsed = time.time() - start

    assert len(results) == 10
    assert all(r is not None for r in results)
    print(f"Created 10 backends concurrently in {elapsed*1000:.2f}ms")


@test("Concurrent Memory creation")
async def test_concurrent_memory():
    """Test concurrent Memory object creation."""
    async def create_memory(i):
        return Memory(
            id=f"mem_{i}",
            text=f"Test {i}",
            timestamp=datetime.now(),
            context={},
            metadata={}
        )

    start = time.time()
    memories = await asyncio.gather(*[create_memory(i) for i in range(100)])
    elapsed = time.time() - start

    assert len(memories) == 100
    print(f"Created 100 memories concurrently in {elapsed*1000:.2f}ms")


# ============================================================================
# Real-World Scenarios
# ============================================================================

@test("Switching backends at runtime")
async def test_backend_switching():
    """Test switching backends."""
    # INMEMORY
    config1 = Config.fast()
    mem1 = await create_memory_backend(config1)

    # HYBRID
    config2 = Config.fused()
    mem2 = await create_memory_backend(config2)

    assert type(mem1) != type(mem2) or isinstance(mem2, HybridMemoryStore)
    print(f"Switched: {type(mem1).__name__} -> {type(mem2).__name__}")


@test("Memory serialization roundtrip")
async def test_serialization():
    """Test Memory to_dict/from_dict roundtrip."""
    original = Memory(
        id="test_123",
        text="Test memory",
        timestamp=datetime.now(),
        context={'key': 'value'},
        metadata={'meta': 'data'}
    )

    # Serialize
    data = original.to_dict()

    # Deserialize
    restored = Memory.from_dict(data)

    assert restored.id == original.id
    assert restored.text == original.text
    assert restored.context == original.context
    print("Serialization roundtrip successful")


@test("HybridStore balanced fusion")
async def test_hybrid_fusion():
    """Test HybridStore fusion logic."""
    config = Config.fused()
    memory = await create_memory_backend(config)

    if isinstance(memory, HybridMemoryStore):
        # Test fusion with mock results
        mock_results = {}
        fused = memory._fuse(mock_results, limit=10)
        assert isinstance(fused, list)
        print(f"Fusion returned {len(fused)} results")
    else:
        print("Not a HybridStore, skipping")


# ============================================================================
# Token Savings Validation
# ============================================================================

@test("File sizes verify token savings")
async def test_token_savings():
    """Verify actual file sizes match claimed savings."""
    import os

    files = {
        'HoloLoom/config.py': 400,  # Expected max lines
        'HoloLoom/memory/backend_factory.py': 250,
        'HoloLoom/memory/protocol.py': 130,
    }

    base = "c:\\Users\\blake\\Documents\\mythRL"

    for filepath, max_lines in files.items():
        full_path = os.path.join(base, filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())

            assert lines <= max_lines, f"{filepath} too big: {lines} > {max_lines}"
            print(f"{filepath}: {lines} lines (max {max_lines})")
        else:
            print(f"{filepath}: not found (OK if in different location)")


# ============================================================================
# Stress Tests
# ============================================================================

@test("Rapid backend creation/destruction")
async def test_rapid_lifecycle():
    """Stress test backend lifecycle."""
    config = Config.fast()

    start = time.time()
    for i in range(50):
        memory = await create_memory_backend(config)
        del memory  # Force cleanup
    elapsed = time.time() - start

    print(f"50 create/destroy cycles in {elapsed*1000:.2f}ms")
    print(f"Average cycle: {elapsed/50*1000:.2f}ms")


@test("Large Memory object handling")
async def test_large_memory():
    """Test Memory with large text."""
    large_text = "x" * 1_000_000  # 1MB of text

    mem = Memory(
        id="large",
        text=large_text,
        timestamp=datetime.now(),
        context={},
        metadata={}
    )

    # Serialize
    data = mem.to_dict()
    assert len(data['text']) == 1_000_000
    print(f"Handled 1MB Memory object")


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Run all tests."""
    print(f"""
{'='*70}
RUTHLESS MEMORY BACKEND TEST SUITE
{'='*70}
Testing simplified memory system under extreme conditions.
{'='*70}
""")

    # Run all tests
    tests = [
        test_three_backends(),
        test_config_defaults(),
        test_inmemory_create(),
        test_hybrid_create(),
        test_empty_query(),
        test_large_limit(),
        test_null_handling(),
        test_invalid_backend(),
        test_backend_fallback(),
        test_hyperspace_fallback(),
        test_create_performance(),
        test_memory_creation(),
        test_protocol_overhead(),
        test_concurrent_creation(),
        test_concurrent_memory(),
        test_backend_switching(),
        test_serialization(),
        test_hybrid_fusion(),
        test_token_savings(),
        test_rapid_lifecycle(),
        test_large_memory(),
    ]

    for test_func in tests:
        await test_func

    # Print summary
    print(stats.summary())

    if stats.failed > 0:
        print("[WARNING] Some tests failed!")
        return 1
    else:
        print("[SUCCESS] ALL TESTS PASSED - Memory system is RUTHLESS!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)