"""
Tracing Performance Benchmark
==============================

Measures overhead of distributed tracing to verify <5ms target.

Usage:
    PYTHONPATH=. python demos/benchmark_tracing_overhead.py

Date: November 16, 2025
"""

import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from HoloLoom.voice.tracing import TracingManager, TracingConfig


async def benchmark_decorator_overhead():
    """Benchmark overhead of tracing decorators."""

    print("=" * 60)
    print("Benchmark: Tracing Decorator Overhead")
    print("=" * 60)

    # Create tracing manager (disabled)
    disabled_config = TracingConfig(enable_tracing=False)
    disabled_tracing = TracingManager(disabled_config)

    # Create tracing manager (enabled)
    enabled_config = TracingConfig(
        enable_tracing=True,
        console_export=False,
        verbose_spans=True
    )
    enabled_tracing = TracingManager(enabled_config)

    # Test function
    @disabled_tracing.trace_voice_command()
    async def operation_disabled(self, transcript: str):
        await asyncio.sleep(0.001)  # Simulate 1ms work
        return "result"

    @enabled_tracing.trace_voice_command()
    async def operation_enabled(self, transcript: str):
        await asyncio.sleep(0.001)  # Simulate 1ms work
        return "result"

    # Warm up
    for _ in range(10):
        await operation_disabled(None, "warmup")
        await operation_enabled(None, "warmup")

    # Benchmark disabled
    iterations = 100
    start = time.time()
    for i in range(iterations):
        await operation_disabled(None, f"test {i}")
    disabled_time = (time.time() - start) / iterations * 1000  # ms

    # Benchmark enabled
    start = time.time()
    for i in range(iterations):
        await operation_enabled(None, f"test {i}")
    enabled_time = (time.time() - start) / iterations * 1000  # ms

    overhead = enabled_time - disabled_time

    print(f"\nIterations: {iterations}")
    print(f"Disabled tracing: {disabled_time:.3f} ms/op")
    print(f"Enabled tracing:  {enabled_time:.3f} ms/op")
    print(f"Overhead:         {overhead:.3f} ms/op")
    print(f"Overhead %:       {(overhead/disabled_time)*100:.1f}%")

    if overhead < 5.0:
        print(f"\n✓ PASS: Overhead {overhead:.3f}ms < 5ms target")
    else:
        print(f"\n✗ FAIL: Overhead {overhead:.3f}ms >= 5ms target")

    # Cleanup
    enabled_tracing.shutdown()

    return overhead


async def benchmark_span_creation():
    """Benchmark manual span creation overhead."""

    print("\n" + "=" * 60)
    print("Benchmark: Manual Span Creation")
    print("=" * 60)

    config = TracingConfig(enable_tracing=True, console_export=False)
    tracing = TracingManager(config)

    # Warm up
    for _ in range(10):
        async with tracing.span("warmup"):
            pass

    # Benchmark
    iterations = 1000
    start = time.time()

    for i in range(iterations):
        async with tracing.span(f"test_span_{i}"):
            pass

    elapsed = (time.time() - start) / iterations * 1000  # ms

    print(f"\nIterations: {iterations}")
    print(f"Time per span: {elapsed:.4f} ms")

    if elapsed < 0.1:
        print(f"\n✓ PASS: Span creation {elapsed:.4f}ms < 0.1ms target")
    else:
        print(f"\n✗ FAIL: Span creation {elapsed:.4f}ms >= 0.1ms target")

    tracing.shutdown()

    return elapsed


async def benchmark_nested_spans():
    """Benchmark nested span overhead."""

    print("\n" + "=" * 60)
    print("Benchmark: Nested Spans")
    print("=" * 60)

    config = TracingConfig(enable_tracing=True, console_export=False)
    tracing = TracingManager(config)

    async def nested_operation(depth: int):
        """Create nested spans."""
        if depth == 0:
            return

        async with tracing.span(f"level_{depth}"):
            await nested_operation(depth - 1)

    # Warm up
    for _ in range(5):
        await nested_operation(3)

    # Benchmark (depth=5)
    iterations = 50
    depth = 5

    start = time.time()
    for _ in range(iterations):
        await nested_operation(depth)
    elapsed = (time.time() - start) / iterations * 1000  # ms

    print(f"\nIterations: {iterations}")
    print(f"Nesting depth: {depth}")
    print(f"Time per operation: {elapsed:.3f} ms")
    print(f"Time per span: {elapsed/depth:.4f} ms")

    if elapsed < 1.0:
        print(f"\n✓ PASS: Nested spans {elapsed:.3f}ms < 1ms target")
    else:
        print(f"\n✗ FAIL: Nested spans {elapsed:.3f}ms >= 1ms target")

    tracing.shutdown()

    return elapsed


async def benchmark_concurrent_traces():
    """Benchmark concurrent trace creation."""

    print("\n" + "=" * 60)
    print("Benchmark: Concurrent Traces")
    print("=" * 60)

    config = TracingConfig(enable_tracing=True, console_export=False)
    tracing = TracingManager(config)

    @tracing.trace_voice_command()
    async def concurrent_operation(self, id: int):
        async with tracing.span("work"):
            await asyncio.sleep(0.001)
        return id

    # Benchmark
    num_concurrent = 50

    start = time.time()
    results = await asyncio.gather(*[
        concurrent_operation(None, i) for i in range(num_concurrent)
    ])
    elapsed = time.time() - start

    print(f"\nConcurrent operations: {num_concurrent}")
    print(f"Total time: {elapsed*1000:.1f} ms")
    print(f"Time per operation: {(elapsed/num_concurrent)*1000:.3f} ms")

    if elapsed < 0.5:
        print(f"\n✓ PASS: Concurrent traces {elapsed*1000:.1f}ms < 500ms")
    else:
        print(f"\n✗ FAIL: Concurrent traces {elapsed*1000:.1f}ms >= 500ms")

    tracing.shutdown()

    return elapsed


async def main():
    """Run all benchmarks."""

    print("\n" + "=" * 60)
    print("HoloLoom Distributed Tracing - Performance Benchmarks")
    print("=" * 60)
    print("\nTarget: <5ms overhead per traced operation")
    print("Target: <0.1ms per span creation")
    print()

    results = {}

    try:
        results['decorator'] = await benchmark_decorator_overhead()
        results['span_creation'] = await benchmark_span_creation()
        results['nested'] = await benchmark_nested_spans()
        results['concurrent'] = await benchmark_concurrent_traces()

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        print(f"\nDecorator overhead: {results['decorator']:.3f} ms")
        print(f"Span creation:      {results['span_creation']:.4f} ms")
        print(f"Nested spans (5x):  {results['nested']:.3f} ms")
        print(f"Concurrent (50x):   {results['concurrent']*1000:.1f} ms")

        # Overall pass/fail
        all_pass = (
            results['decorator'] < 5.0 and
            results['span_creation'] < 0.1 and
            results['nested'] < 1.0 and
            results['concurrent'] < 0.5
        )

        if all_pass:
            print("\n✓ ALL BENCHMARKS PASSED")
        else:
            print("\n✗ SOME BENCHMARKS FAILED")

    except Exception as e:
        print(f"\n✗ Benchmark error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
