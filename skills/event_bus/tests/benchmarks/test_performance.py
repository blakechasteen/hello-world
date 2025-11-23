"""
EventBus Performance Benchmarks
================================

**Created**: November 22, 2025
**Purpose**: Validate <10ms event propagation target

Benchmarks:
- Event emission latency (target: <10ms)
- Topic routing performance (exact vs wildcard)
- Handler execution overhead
- Concurrent delivery throughput
- Pattern cache effectiveness
"""

import asyncio
import time
import statistics
from typing import List

import pytest

# Add parent to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from skills.event_bus import EventBroker, SkillEvent, EventType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def broker():
    """Create event broker for benchmarking."""
    import asyncio
    broker_instance = EventBroker(enable_async_delivery=False)  # Sync for accurate timing
    yield broker_instance
    asyncio.get_event_loop().run_until_complete(broker_instance.close())


@pytest.fixture
def async_broker():
    """Create async event broker for throughput tests."""
    import asyncio
    broker_instance = EventBroker(enable_async_delivery=True)
    yield broker_instance
    asyncio.get_event_loop().run_until_complete(broker_instance.close())


# ============================================================================
# Utility Functions
# ============================================================================

def measure_latency(func, iterations: int = 100) -> dict:
    """
    Measure latency statistics for async function.

    Returns:
        Dict with min, max, mean, median, p95, p99 latencies
    """
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        asyncio.run(func())
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        'min': min(latencies),
        'max': max(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'p95': statistics.quantiles(latencies, n=20)[18],  # 95th percentile
        'p99': statistics.quantiles(latencies, n=100)[98],  # 99th percentile
        'samples': iterations
    }


async def noop_handler(event: SkillEvent):
    """No-op handler for overhead measurement."""
    pass


async def slow_handler(event: SkillEvent):
    """Simulates slow handler (10ms processing)."""
    await asyncio.sleep(0.01)


# ============================================================================
# Benchmark: Event Emission Latency
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_emission_no_subscribers(broker):
    """Benchmark: Event emission with no subscribers."""

    async def emit_event():
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="test.topic",
            skill_name="benchmark",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    stats = measure_latency(lambda: emit_event(), iterations=100)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Event Emission (No Subscribers)")
    print(f"{'='*60}")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"  Min:    {stats['min']:.3f} ms")
    print(f"  Max:    {stats['max']:.3f} ms")
    print(f"{'='*60}")

    # Validate: Should be well under 10ms even with no subscribers
    assert stats['p95'] < 5.0, f"P95 latency {stats['p95']:.3f}ms exceeds 5ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_emission_single_subscriber(broker):
    """Benchmark: Event emission with single subscriber."""

    # Subscribe handler
    await broker.subscribe(topic="test.topic", handler=noop_handler)

    async def emit_event():
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="test.topic",
            skill_name="benchmark",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    stats = measure_latency(lambda: emit_event(), iterations=100)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Event Emission (Single Subscriber)")
    print(f"{'='*60}")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"  Min:    {stats['min']:.3f} ms")
    print(f"  Max:    {stats['max']:.3f} ms")
    print(f"{'='*60}")

    # Validate: <10ms target
    assert stats['p95'] < 10.0, f"P95 latency {stats['p95']:.3f}ms exceeds 10ms target"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_emission_multiple_subscribers(broker):
    """Benchmark: Event emission with 10 subscribers."""

    # Subscribe 10 handlers
    for i in range(10):
        await broker.subscribe(topic="test.topic", handler=noop_handler)

    async def emit_event():
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="test.topic",
            skill_name="benchmark",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    stats = measure_latency(lambda: emit_event(), iterations=100)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Event Emission (10 Subscribers)")
    print(f"{'='*60}")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"  Min:    {stats['min']:.3f} ms")
    print(f"  Max:    {stats['max']:.3f} ms")
    print(f"{'='*60}")

    # Validate: Should still be under 10ms with 10 subscribers
    assert stats['p95'] < 10.0, f"P95 latency {stats['p95']:.3f}ms exceeds 10ms target"


# ============================================================================
# Benchmark: Topic Routing Performance
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_exact_match_routing(broker):
    """Benchmark: Exact topic match (fast path)."""

    # Create 100 exact subscriptions
    for i in range(100):
        await broker.subscribe(topic=f"skill.{i}.exact", handler=noop_handler)

    async def emit_event():
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="skill.50.exact",  # Exact match
            skill_name="benchmark",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    stats = measure_latency(lambda: emit_event(), iterations=100)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Exact Match Routing (100 subscriptions)")
    print(f"{'='*60}")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"{'='*60}")

    # Exact match should be very fast (O(1) lookup)
    assert stats['p95'] < 10.0, f"Exact match P95 {stats['p95']:.3f}ms exceeds 10ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_wildcard_routing(broker):
    """Benchmark: Wildcard topic matching (slow path)."""

    # Create 50 wildcard subscriptions
    for i in range(50):
        await broker.subscribe(topic=f"skill.*.domain.{i}", handler=noop_handler)

    async def emit_event():
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="skill.started.domain.25",  # Matches wildcard
            skill_name="benchmark",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    stats = measure_latency(lambda: emit_event(), iterations=100)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Wildcard Routing (50 patterns)")
    print(f"{'='*60}")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"{'='*60}")

    # Wildcard should still be under 10ms with pattern caching
    assert stats['p95'] < 10.0, f"Wildcard P95 {stats['p95']:.3f}ms exceeds 10ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_pattern_cache_effectiveness(broker):
    """Benchmark: Pattern cache hit rate and speedup."""

    # Create wildcard subscriptions
    for i in range(20):
        await broker.subscribe(topic=f"skill.*.meta.{i}", handler=noop_handler)

    # First emission (cold cache)
    event = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="skill.started.meta.10",
        skill_name="benchmark",
        timestamp="",
        payload={}
    )

    start = time.perf_counter()
    await broker.emit(event)
    cold_latency = (time.perf_counter() - start) * 1000

    # Second emission (warm cache - same topic)
    start = time.perf_counter()
    await broker.emit(event)
    warm_latency = (time.perf_counter() - start) * 1000

    speedup = cold_latency / warm_latency if warm_latency > 0 else 0

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Pattern Cache Effectiveness")
    print(f"{'='*60}")
    print(f"  Cold latency: {cold_latency:.3f} ms")
    print(f"  Warm latency: {warm_latency:.3f} ms")
    print(f"  Speedup:      {speedup:.2f}x")
    print(f"{'='*60}")

    # Cache should provide speedup
    assert speedup >= 1.0, "Cache should not slow down requests"


# ============================================================================
# Benchmark: Handler Execution Overhead
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_handler_overhead(broker):
    """Benchmark: Handler execution overhead."""

    execution_count = [0]

    async def counting_handler(event: SkillEvent):
        execution_count[0] += 1

    await broker.subscribe(topic="test.topic", handler=counting_handler)

    async def emit_event():
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="test.topic",
            skill_name="benchmark",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    stats = measure_latency(lambda: emit_event(), iterations=100)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Handler Execution Overhead")
    print(f"{'='*60}")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms")
    print(f"  Executions: {execution_count[0]}")
    print(f"{'='*60}")

    # Overhead should be minimal
    assert stats['p95'] < 10.0, f"Handler overhead P95 {stats['p95']:.3f}ms exceeds 10ms"
    assert execution_count[0] == 100, "All handlers should execute"


# ============================================================================
# Benchmark: Concurrent Delivery Throughput
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_throughput(async_broker):
    """Benchmark: Concurrent event throughput."""

    received_count = [0]

    async def counting_handler(event: SkillEvent):
        received_count[0] += 1

    # Subscribe 5 handlers
    for i in range(5):
        await async_broker.subscribe(topic="throughput.**", handler=counting_handler)

    # Emit 1000 events concurrently
    num_events = 1000

    start = time.perf_counter()

    tasks = []
    for i in range(num_events):
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic=f"throughput.batch.{i}",
            skill_name="benchmark",
            timestamp="",
            payload={"index": i}
        )
        # Emit without await (fire-and-forget)
        task = asyncio.create_task(async_broker.emit(event))
        tasks.append(task)

    # Wait for all emissions
    await asyncio.gather(*tasks)

    # Wait for delivery
    await async_broker.wait_for_delivery()

    end = time.perf_counter()
    duration = (end - start) * 1000  # ms

    throughput = num_events / (duration / 1000)  # events/sec
    avg_latency = duration / num_events  # ms per event

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Concurrent Throughput")
    print(f"{'='*60}")
    print(f"  Events:     {num_events}")
    print(f"  Handlers:   5 subscribers")
    print(f"  Duration:   {duration:.1f} ms")
    print(f"  Throughput: {throughput:.0f} events/sec")
    print(f"  Avg latency: {avg_latency:.3f} ms/event")
    print(f"  Received:   {received_count[0]} (expected {num_events * 5})")
    print(f"{'='*60}")

    # Validate: Should handle >10,000 events/sec target
    assert throughput > 10000, f"Throughput {throughput:.0f} events/sec below 10,000 target"
    assert avg_latency < 10.0, f"Avg latency {avg_latency:.3f}ms exceeds 10ms"


# ============================================================================
# Benchmark: End-to-End Latency
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_benchmark_end_to_end_latency(broker):
    """Benchmark: End-to-end event propagation latency."""

    received_times: List[float] = []

    async def timestamp_handler(event: SkillEvent):
        received_times.append(time.perf_counter())

    await broker.subscribe(topic="e2e.test", handler=timestamp_handler)

    emit_times: List[float] = []

    # Emit 100 events and measure end-to-end latency
    for i in range(100):
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="e2e.test",
            skill_name="benchmark",
            timestamp="",
            payload={"index": i}
        )

        emit_start = time.perf_counter()
        await broker.emit(event)
        emit_times.append(emit_start)

    # Calculate end-to-end latencies
    latencies = [(received_times[i] - emit_times[i]) * 1000
                 for i in range(len(received_times))]

    stats = {
        'min': min(latencies),
        'max': max(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'p95': statistics.quantiles(latencies, n=20)[18],
        'p99': statistics.quantiles(latencies, n=100)[98]
    }

    print(f"\n{'='*60}")
    print(f"BENCHMARK: End-to-End Latency (Emit → Handler)")
    print(f"{'='*60}")
    print(f"  Mean:   {stats['mean']:.3f} ms")
    print(f"  Median: {stats['median']:.3f} ms")
    print(f"  P95:    {stats['p95']:.3f} ms ⭐ TARGET")
    print(f"  P99:    {stats['p99']:.3f} ms")
    print(f"  Min:    {stats['min']:.3f} ms")
    print(f"  Max:    {stats['max']:.3f} ms")
    print(f"{'='*60}")

    # CRITICAL VALIDATION: <10ms P95 latency
    assert stats['p95'] < 10.0, f"❌ P95 latency {stats['p95']:.3f}ms EXCEEDS 10ms target"

    print(f"\n✅ SUCCESS: P95 latency {stats['p95']:.3f}ms is under 10ms target!")


# ============================================================================
# Summary Report
# ============================================================================

@pytest.mark.performance
def test_benchmark_summary():
    """Print benchmark summary and validation."""

    print(f"\n{'='*60}")
    print(f"PERFORMANCE BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"")
    print(f"Target: <10ms event propagation (P95)")
    print(f"")
    print(f"Run benchmarks with:")
    print(f"  pytest skills/event_bus/tests/benchmarks/ -v -m benchmark")
    print(f"")
    print(f"Key Metrics:")
    print(f"  - Event emission latency")
    print(f"  - Topic routing performance (exact vs wildcard)")
    print(f"  - Handler execution overhead")
    print(f"  - Concurrent throughput (>10,000 events/sec target)")
    print(f"  - End-to-end latency (emit → handler)")
    print(f"")
    print(f"Optimizations:")
    print(f"  - Exact match fast path (O(1) dict lookup)")
    print(f"  - Pattern caching (LRU eviction, 1000 entries)")
    print(f"  - Pre-compiled regex patterns")
    print(f"  - Semaphore-based concurrency control")
    print(f"  - Async delivery for non-blocking propagation")
    print(f"{'='*60}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'benchmark'])
