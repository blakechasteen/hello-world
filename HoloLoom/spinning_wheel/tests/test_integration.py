#!/usr/bin/env python3
"""
Integration Tests & Performance Benchmarks
===========================================

Tests the complete pipeline:
1. Spinner → MemoryShards
2. Shards → Memory Backend
3. Memory → Orchestrator (if available)

Also includes performance benchmarks for each spinner.

Run with:
    python test_integration.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.spinning_wheel import (
    AudioSpinner, TextSpinner, CodeSpinner, WebsiteSpinner,
    spin_text, spin_webpage, spin_code_file
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.benchmarks = {}

    def record_pass(self, test_name):
        self.passed += 1
        print(f"  [PASS] {test_name}")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        # Avoid Unicode issues with arrows
        print(f"  [FAIL] {test_name}: {error}")

    def record_benchmark(self, name, duration, items=1):
        """Record performance benchmark."""
        self.benchmarks[name] = {
            'duration': duration,
            'items': items,
            'rate': items / duration if duration > 0 else 0
        }

    def print_benchmarks(self):
        """Print performance summary."""
        print("\n" + "=" * 70)
        print("Performance Benchmarks")
        print("=" * 70)
        for name, stats in sorted(self.benchmarks.items()):
            print(f"\n{name}:")
            print(f"  Duration: {stats['duration']:.3f}s")
            print(f"  Items: {stats['items']}")
            print(f"  Rate: {stats['rate']:.1f} items/sec")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Integration Tests: {total} total, {self.passed} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*70}")
        return self.failed == 0


results = TestResults()


# ============================================================================
# Integration Tests
# ============================================================================

async def test_text_to_memory_pipeline():
    """Test: TextSpinner → MemoryShard conversion."""
    try:
        # Spin text
        text = "This is a test document about beekeeping. Hives need regular inspection."
        shards = await spin_text(text=text, source='test.txt')

        assert len(shards) >= 1, f"Expected shards, got {len(shards)}"
        assert shards[0].text == text
        assert shards[0].metadata['source'] == 'test.txt'

        # Verify we can convert to memories
        try:
            from HoloLoom.memory.protocol import shards_to_memories
            memories = shards_to_memories(shards)
            assert len(memories) == len(shards)
            results.record_pass("Text to Shard Conversion")
        except ImportError:
            results.record_pass("Text to Shard Conversion (memory module not available)")

    except Exception as e:
        results.record_fail("Text to Shard Conversion", str(e))


async def test_website_to_memory_pipeline():
    """Test: WebsiteSpinner → MemoryShard conversion."""
    try:
        # Spin webpage (with pre-fetched content)
        # Content must be long enough to pass min_content_length filter
        content = """Winter beekeeping guide for preparing your hives.
        When preparing hives for cold weather, ensure adequate honey stores are available.
        Queens need protection during winter months. Cluster formation is essential for survival.
        Ventilation must be balanced to prevent moisture buildup while retaining heat."""

        shards = await spin_webpage(
            url='https://example.com/winter-guide',
            title='Winter Beekeeping',
            content=content
        )

        assert len(shards) >= 1, f"Expected at least 1 shard, got {len(shards)}"
        assert shards[0].metadata.get('url') == 'https://example.com/winter-guide'
        assert shards[0].metadata.get('domain') == 'example.com'

        results.record_pass("Website to Shard Conversion")

    except Exception as e:
        results.record_fail("Website to Shard Conversion", str(e))


async def test_code_to_memory_pipeline():
    """Test: CodeSpinner → MemoryShard conversion."""
    try:
        code = '''
def hello_world():
    """Print greeting."""
    print("Hello, world!")

def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b
'''

        shards = await spin_code_file(path='test.py', content=code)

        assert len(shards) >= 1
        assert shards[0].metadata.get('language') == 'python'
        assert shards[0].metadata.get('path') == 'test.py'

        results.record_pass("Code to Shard Conversion")

    except Exception as e:
        results.record_fail("Code to Shard Conversion", str(e))


async def test_orchestrator_integration():
    """Test: WeavingOrchestrator can be initialized."""
    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config

        # Initialize orchestrator
        weaver = WeavingOrchestrator(config=Config.bare())

        # Verify it has necessary components
        assert hasattr(weaver, 'loom')
        assert hasattr(weaver, 'chrono')
        assert hasattr(weaver, 'shed')

        results.record_pass("Orchestrator Integration")

    except ImportError:
        results.record_pass("Orchestrator Integration (not available - optional)")
    except Exception as e:
        results.record_fail("Orchestrator Integration", str(e))


# ============================================================================
# Performance Benchmarks
# ============================================================================

async def benchmark_text_spinner():
    """Benchmark TextSpinner performance."""
    iterations = 100
    text = "This is a test document. " * 50  # ~1000 chars

    start = time.time()
    for _ in range(iterations):
        shards = await spin_text(text=text, source='bench.txt')
    duration = time.time() - start

    results.record_benchmark("TextSpinner", duration, iterations)
    results.record_pass(f"Benchmark: TextSpinner ({iterations} iterations)")


async def benchmark_code_spinner():
    """Benchmark CodeSpinner performance."""
    iterations = 100
    code = '''
def function_one():
    pass

def function_two():
    pass

class MyClass:
    pass
''' * 10

    start = time.time()
    for _ in range(iterations):
        shards = await spin_code_file(path='bench.py', content=code)
    duration = time.time() - start

    results.record_benchmark("CodeSpinner", duration, iterations)
    results.record_pass(f"Benchmark: CodeSpinner ({iterations} iterations)")


async def benchmark_website_spinner():
    """Benchmark WebsiteSpinner performance."""
    iterations = 50
    content = "Article content here. " * 100

    start = time.time()
    for _ in range(iterations):
        shards = await spin_webpage(
            url='https://example.com/article',
            content=content
        )
    duration = time.time() - start

    results.record_benchmark("WebsiteSpinner", duration, iterations)
    results.record_pass(f"Benchmark: WebsiteSpinner ({iterations} iterations)")


async def benchmark_memory_storage():
    """Benchmark shard generation performance."""
    try:
        # Generate shards
        texts = [f"Test document {i} about beekeeping." for i in range(100)]
        all_shards = []

        start = time.time()
        for text in texts:
            shards = await spin_text(text=text, source=f'test_{len(all_shards)}.txt')
            all_shards.extend(shards)
        duration = time.time() - start

        # Verify shard conversion works
        try:
            from HoloLoom.memory.protocol import shards_to_memories
            memories = shards_to_memories(all_shards)
            assert len(memories) == len(all_shards)
        except ImportError:
            pass  # Memory module not available - skip conversion test

        results.record_benchmark("Shard Generation", duration, len(all_shards))
        results.record_pass(f"Benchmark: Shard Generation ({len(all_shards)} shards)")

    except Exception as e:
        results.record_fail("Benchmark: Shard Generation", str(e))


async def benchmark_memory_recall():
    """Benchmark shard-to-memory conversion performance."""
    try:
        from HoloLoom.memory.protocol import shards_to_memories

        # Generate shards
        texts = [f"Document about hive inspection {i}." for i in range(50)]
        all_shards = []
        for text in texts:
            shards = await spin_text(text=text, source=f'doc_{len(all_shards)}.txt')
            all_shards.extend(shards)

        # Benchmark conversion
        iterations = 50
        start = time.time()
        for i in range(iterations):
            memories = shards_to_memories(all_shards)
            assert len(memories) == len(all_shards)
        duration = time.time() - start

        results.record_benchmark("Shard Conversion", duration, iterations)
        results.record_pass(f"Benchmark: Shard Conversion ({iterations} iterations)")

    except ImportError:
        results.record_pass("Benchmark: Shard Conversion (memory module not available)")
    except Exception as e:
        results.record_fail("Benchmark: Shard Conversion", str(e))


# ============================================================================
# Test Runner
# ============================================================================

async def run_all_tests():
    """Run all integration tests and benchmarks."""

    print("\n" + "=" * 70)
    print("SpinningWheel Integration Tests & Performance Benchmarks")
    print("=" * 70 + "\n")

    print("Integration Tests:")
    print("-" * 70)

    # Integration tests
    await test_text_to_memory_pipeline()
    await test_website_to_memory_pipeline()
    await test_code_to_memory_pipeline()
    await test_orchestrator_integration()

    print("\nPerformance Benchmarks:")
    print("-" * 70)

    # Benchmarks
    await benchmark_text_spinner()
    await benchmark_code_spinner()
    await benchmark_website_spinner()
    await benchmark_memory_storage()
    await benchmark_memory_recall()

    # Print benchmark summary
    results.print_benchmarks()

    # Print final summary
    return results.summary()


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
