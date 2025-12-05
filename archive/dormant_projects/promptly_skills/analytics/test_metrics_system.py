#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Phase 5 metrics collection system.

Run this to verify that all components work together:
- MetricsCollector
- TimeSeriesDB
- MetricsAggregator

Usage:
    python -m analytics.test_metrics_system
"""

import sys
import asyncio
import time
import random
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from metrics_collector import MetricsCollector, MetricType
from time_series_db import TimeSeriesDB, Query, AggregationFunction
from aggregator import MetricsAggregator, AggregationPeriod


async def generate_sample_data(collector: MetricsCollector, num_queries: int = 100):
    """Generate sample query data."""
    print(f"\n📊 Generating {num_queries} sample queries...")

    strategies = ['deep', 'scaffold', 'teach', 'verify', 'optimize']
    base_time = time.time() - 3600  # Start 1 hour ago

    for i in range(num_queries):
        strategy = random.choice(strategies)

        # Simulate different performance characteristics
        if strategy == 'deep':
            latency = random.gauss(150, 20)
            confidence = random.gauss(0.92, 0.05)
        elif strategy == 'scaffold':
            latency = random.gauss(120, 15)
            confidence = random.gauss(0.88, 0.06)
        elif strategy == 'teach':
            latency = random.gauss(80, 10)
            confidence = random.gauss(0.85, 0.07)
        elif strategy == 'verify':
            latency = random.gauss(60, 8)
            confidence = random.gauss(0.91, 0.04)
        else:  # optimize
            latency = random.gauss(200, 30)
            confidence = random.gauss(0.94, 0.03)

        # Clamp values
        latency = max(10, min(500, latency))
        confidence = max(0.5, min(1.0, confidence))

        # Random cache hit (30% hit rate)
        cache_hit = random.random() < 0.3

        # Record query (with timestamp in past for trend analysis)
        timestamp_offset = (i / num_queries) * 3600
        await collector.record_query(
            query=f"sample query {i}",
            strategy=strategy,
            latency_ms=latency,
            confidence=confidence,
            cache_hit=cache_hit
        )

        # Manually set timestamp for testing (normally automatic)
        if collector.buffer:
            collector.buffer[-1].timestamp = base_time + timestamp_offset

    print(f"✓ Generated {num_queries} queries")
    print(f"  Strategies: {strategies}")
    print(f"  Time range: last 1 hour")


async def test_metrics_collection():
    """Test basic metrics collection."""
    print("\n" + "="*60)
    print("Test 1: Metrics Collection")
    print("="*60)

    # Create components
    db_path = Path('test_metrics.db')
    if db_path.exists():
        db_path.unlink()

    db = TimeSeriesDB(str(db_path))
    await db.initialize()

    collector = MetricsCollector(db=db, flush_interval=1.0)

    # Generate sample data
    await generate_sample_data(collector, num_queries=100)

    # Flush to database
    await collector._flush()

    # Query data
    query = Query(
        metric_type=MetricType.QUERY,
        start_time=time.time() - 7200,  # Last 2 hours
        limit=1000
    )

    events = await db.query(query)

    print(f"\n✓ Collected {len(events)} events")
    print(f"  First event: {events[-1]['timestamp']:.2f}")
    print(f"  Last event: {events[0]['timestamp']:.2f}")

    # Get top strategies
    top_strategies = await db.get_top_strategies(
        metric='confidence',
        limit=5,
        start_time=time.time() - 7200
    )

    print(f"\n📊 Top strategies by confidence:")
    for i, result in enumerate(top_strategies, 1):
        print(f"  {i}. {result['strategy']}: {result['avg_metric']:.3f} avg confidence ({result['count']} uses)")

    await db.close()

    print("\n✅ Test 1 passed!")
    return True


async def test_aggregation():
    """Test metrics aggregation."""
    print("\n" + "="*60)
    print("Test 2: Metrics Aggregation")
    print("="*60)

    # Open existing database
    db = TimeSeriesDB('test_metrics.db')
    await db.initialize()

    # Create aggregator
    aggregator = MetricsAggregator(db)

    # Compute aggregations
    print("\n📊 Computing aggregations...")
    stats = await aggregator.compute(AggregationPeriod.HOUR)

    print(f"\n✓ Aggregated statistics (last hour):")
    print(f"  Total queries: {stats.total_queries}")
    print(f"  Avg latency: {stats.avg_latency_ms:.1f}ms")
    print(f"  P95 latency: {stats.p95_latency_ms:.1f}ms")
    print(f"  Avg confidence: {stats.avg_confidence:.3f}")
    print(f"  Median confidence: {stats.median_confidence:.3f}")
    print(f"  Cache hit rate: {stats.cache_hit_rate:.1%}")

    print(f"\n📊 Strategy distribution:")
    for strategy, count in sorted(stats.strategy_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats.total_queries * 100
        print(f"  {strategy}: {count} ({percentage:.1f}%)")

    print(f"\n📊 Strategy performance:")
    for strategy, perf in stats.strategy_performance.items():
        print(f"  {strategy}:")
        print(f"    Avg latency: {perf['avg_latency_ms']:.1f}ms")
        print(f"    Avg confidence: {perf['avg_confidence']:.3f}")
        print(f"    Success rate: {perf['success_rate']:.1%}")

    # Get top strategies
    top = await aggregator.get_top_strategies(
        period=AggregationPeriod.HOUR,
        metric='avg_confidence',
        limit=3
    )

    print(f"\n🏆 Top 3 strategies by confidence:")
    for i, result in enumerate(top, 1):
        perf = result['performance']
        print(f"  {i}. {result['strategy']}: {perf['avg_confidence']:.3f}")

    await db.close()

    print("\n✅ Test 2 passed!")
    return True


async def test_time_series():
    """Test time-series queries."""
    print("\n" + "="*60)
    print("Test 3: Time-Series Queries")
    print("="*60)

    db = TimeSeriesDB('test_metrics.db')
    await db.initialize()

    # Get latency trend (5-minute buckets)
    print("\n📈 Latency trend (5-minute buckets):")
    trend = await db.get_time_series(
        metric_type=MetricType.QUERY,
        field='latency_ms',
        interval_seconds=300,  # 5 minutes
        duration_seconds=3600  # Last hour
    )

    for point in trend:
        timestamp_str = time.strftime('%H:%M', time.localtime(point['timestamp']))
        print(f"  {timestamp_str}: {point['value']:.1f}ms")

    # Get confidence trend
    print("\n📈 Confidence trend (5-minute buckets):")
    trend = await db.get_time_series(
        metric_type=MetricType.QUERY,
        field='confidence',
        interval_seconds=300,
        duration_seconds=3600
    )

    for point in trend:
        timestamp_str = time.strftime('%H:%M', time.localtime(point['timestamp']))
        print(f"  {timestamp_str}: {point['value']:.3f}")

    # Cache statistics
    cache_stats = await db.get_cache_stats(duration_seconds=3600)

    print(f"\n💾 Cache performance:")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Total hits: {cache_stats['total_hits']}")
    print(f"  Total misses: {cache_stats['total_misses']}")
    print(f"  Speedup: {cache_stats['speedup']:.1f}×")

    await db.close()

    print("\n✅ Test 3 passed!")
    return True


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Phase 5 Metrics System - Integration Test")
    print("="*60)

    try:
        # Run tests
        test1 = await test_metrics_collection()
        test2 = await test_aggregation()
        test3 = await test_time_series()

        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"✅ Test 1 (Metrics Collection): {'PASSED' if test1 else 'FAILED'}")
        print(f"✅ Test 2 (Aggregation): {'PASSED' if test2 else 'FAILED'}")
        print(f"✅ Test 3 (Time-Series): {'PASSED' if test3 else 'FAILED'}")

        if test1 and test2 and test3:
            print("\n🎉 All tests passed!")
            print("\n✨ Phase 5 metrics collection system is working!")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
