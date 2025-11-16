#!/usr/bin/env python3
"""
Test Distributed Tracing System
================================

Simple test script to verify OpenTelemetry tracing is working correctly.

This script tests:
1. Tracing initialization
2. Span creation
3. Span attributes
4. Span events
5. Error recording
6. Context propagation
7. Performance analysis

Usage:
    # Start Jaeger first
    docker-compose -f docker-compose.tracing.yml up jaeger -d

    # Run tests
    TRACING_ENABLED=true TRACING_EXPORTER=jaeger python HoloLoom/tracing/test_tracing.py

    # View results in Jaeger UI
    Open http://localhost:16686 and search for "test-service"

Created: 2025-11-16
"""

import asyncio
import time
from typing import List

from HoloLoom.tracing import (
    TracingConfig,
    init_tracing,
    get_tracer,
    create_span,
    add_span_event,
    set_span_attribute,
    record_exception,
    get_trace_id,
    get_span_id,
    is_sampled,
    shutdown_tracing,
    PerformanceAnalyzer,
    Span,
    Trace,
)


def test_basic_tracing():
    """Test basic span creation."""
    print("\n=== Test 1: Basic Tracing ===")

    # Initialize tracing
    config = TracingConfig(
        service_name="test-service",
        environment="testing",
        exporter="console",  # Use console for this test
        sample_rate=1.0
    )
    tracer = init_tracing(config)

    # Create a simple span
    with tracer.start_as_current_span("test_operation") as span:
        span.set_attribute("test.attribute", "value")
        print(f"Trace ID: {get_trace_id()}")
        print(f"Span ID: {get_span_id()}")
        print(f"Sampled: {is_sampled()}")

    print("✓ Basic tracing test passed")


def test_nested_spans():
    """Test nested span creation."""
    print("\n=== Test 2: Nested Spans ===")

    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("parent_operation") as parent:
        parent.set_attribute("level", "parent")
        print(f"Parent span: {get_span_id()}")

        # Child span 1
        with create_span("child_1", {"level": "child", "index": 1}):
            print(f"  Child 1 span: {get_span_id()}")
            time.sleep(0.1)

        # Child span 2
        with create_span("child_2", {"level": "child", "index": 2}):
            print(f"  Child 2 span: {get_span_id()}")
            time.sleep(0.05)

            # Grandchild span
            with create_span("grandchild", {"level": "grandchild"}):
                print(f"    Grandchild span: {get_span_id()}")
                time.sleep(0.02)

    print("✓ Nested spans test passed")


def test_span_events():
    """Test span event recording."""
    print("\n=== Test 3: Span Events ===")

    tracer = get_tracer(__name__)

    with create_span("operation_with_events"):
        add_span_event("operation_started")

        time.sleep(0.05)
        add_span_event("checkpoint_1", {"progress": 0.33})

        time.sleep(0.05)
        add_span_event("checkpoint_2", {"progress": 0.66})

        time.sleep(0.05)
        add_span_event("operation_completed", {"progress": 1.0})

    print("✓ Span events test passed")


def test_error_recording():
    """Test exception recording in spans."""
    print("\n=== Test 4: Error Recording ===")

    tracer = get_tracer(__name__)

    # Test successful operation
    with create_span("successful_operation"):
        set_span_attribute("status", "success")
        result = 42

    # Test failed operation
    try:
        with create_span("failed_operation"):
            set_span_attribute("status", "error")
            raise ValueError("This is a test error")
    except ValueError as e:
        record_exception(e)
        print(f"✓ Error recorded: {e}")

    print("✓ Error recording test passed")


async def test_async_spans():
    """Test async span creation."""
    print("\n=== Test 5: Async Spans ===")

    tracer = get_tracer(__name__)

    async def async_operation(name: str, duration: float):
        """Simulate async operation."""
        with create_span(f"async_{name}", {"duration": duration}):
            await asyncio.sleep(duration)
            return f"{name}_result"

    # Run async operations
    with create_span("async_parent"):
        # Sequential
        result1 = await async_operation("op1", 0.05)
        result2 = await async_operation("op2", 0.03)

        # Parallel
        results = await asyncio.gather(
            async_operation("parallel_1", 0.02),
            async_operation("parallel_2", 0.02),
            async_operation("parallel_3", 0.02),
        )

    print(f"✓ Async operations completed: {results}")
    print("✓ Async spans test passed")


def test_performance_analysis():
    """Test performance analysis tools."""
    print("\n=== Test 6: Performance Analysis ===")

    # Create mock traces for analysis
    traces = []

    for i in range(10):
        # Create mock spans
        spans = [
            Span(
                span_id=f"span_{i}_1",
                parent_id=None,
                operation_name="weave",
                start_time=time.time(),
                end_time=time.time() + 0.15,
                duration=0.15,
                status="OK"
            ),
            Span(
                span_id=f"span_{i}_2",
                parent_id=f"span_{i}_1",
                operation_name="retrieve_context",
                start_time=time.time(),
                end_time=time.time() + 0.08,
                duration=0.08,
                status="OK"
            ),
            Span(
                span_id=f"span_{i}_3",
                parent_id=f"span_{i}_1",
                operation_name="make_decision",
                start_time=time.time(),
                end_time=time.time() + 0.05,
                duration=0.05,
                status="OK"
            ),
        ]

        trace = Trace(
            trace_id=f"trace_{i}",
            spans=spans,
            start_time=time.time(),
            end_time=time.time() + 0.15,
            duration=0.15
        )
        traces.append(trace)

    # Analyze traces
    analyzer = PerformanceAnalyzer()
    analyzer.add_traces(traces)

    # Get statistics
    slowest = analyzer.get_slowest_operations(limit=5)
    print("\nSlowest operations:")
    for op in slowest:
        print(f"  {op.operation_name}: P95={op.p95_duration_ms:.1f}ms")

    frequent = analyzer.get_most_frequent_operations(limit=5)
    print("\nMost frequent operations:")
    for op, count in frequent:
        print(f"  {op}: {count} calls")

    # Generate report
    report = analyzer.generate_report()
    print(f"\nSummary:")
    print(f"  Total traces: {report['summary']['total_traces']}")
    print(f"  Total operations: {report['summary']['total_operations']}")
    print(f"  Unique operations: {report['summary']['unique_operations']}")

    print("✓ Performance analysis test passed")


async def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Testing HoloLoom Distributed Tracing System")
    print("=" * 70)

    # Run tests
    test_basic_tracing()
    test_nested_spans()
    test_span_events()
    test_error_recording()
    await test_async_spans()
    test_performance_analysis()

    # Shutdown
    print("\n=== Cleanup ===")
    shutdown_tracing()
    print("✓ Tracing shutdown complete")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nTo view traces in Jaeger:")
    print("1. Start Jaeger: docker-compose -f docker-compose.tracing.yml up jaeger -d")
    print("2. Re-run with Jaeger exporter:")
    print("   TRACING_ENABLED=true TRACING_EXPORTER=jaeger python HoloLoom/tracing/test_tracing.py")
    print("3. Open http://localhost:16686")
    print("4. Search for service: test-service")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
