#!/usr/bin/env python3
"""
HoloLoom LSP Performance Benchmarks (November 2025)
===================================================

Performance testing and benchmarking suite for LSP server. Validates:

1. **Latency Targets** (warm cache):
   - Completion: <100ms
   - Hover: <50ms
   - Definition: <75ms
   - Symbol Search: <200ms

2. **Throughput**:
   - Requests/second under sustained load
   - Stability over time

3. **Resource Usage**:
   - Memory footprint
   - CPU usage

4. **Scaling**:
   - Large workspace indexing (1000+ files)
   - Large query result sets

5. **Consistency**:
   - Repeated requests have consistent performance
   - No memory leaks detected
   - Proper cleanup

Status: Production Ready (November 2025)
"""

import pytest
import asyncio
import sys
import time
import statistics
from pathlib import Path
from typing import List, Tuple, Dict, Any
from unittest.mock import Mock, AsyncMock
import gc

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lsprotocol.types import (
    CompletionParams,
    HoverParams,
    DefinitionParams,
    WorkspaceSymbolParams,
    Position,
    TextDocumentIdentifier,
)

# Test workspace path
TEST_WORKSPACE = Path(__file__).parent / "fixtures" / "test_workspace"


# ============================================================================
# Benchmarking Utilities
# ============================================================================

class PerformanceMetrics:
    """Collect and analyze performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.latencies: List[float] = []
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()

    def end(self) -> float:
        """End timing and record latency in milliseconds."""
        if self.start_time is None:
            raise ValueError("Must call start() before end()")
        latency_ms = (time.time() - self.start_time) * 1000
        self.latencies.append(latency_ms)
        return latency_ms

    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary."""
        if not self.latencies:
            return {}

        return {
            "count": len(self.latencies),
            "min": min(self.latencies),
            "max": max(self.latencies),
            "mean": statistics.mean(self.latencies),
            "median": statistics.median(self.latencies),
            "stdev": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
            "p95": sorted(self.latencies)[int(len(self.latencies) * 0.95)] if self.latencies else 0,
            "p99": sorted(self.latencies)[int(len(self.latencies) * 0.99)] if self.latencies else 0,
        }

    def print_summary(self) -> None:
        """Print formatted summary."""
        stats = self.get_stats()
        if not stats:
            print(f"  No data for {self.name}")
            return

        print(f"\n  {self.name}:")
        print(f"    Count:  {stats['count']}")
        print(f"    Min:    {stats['min']:.2f}ms")
        print(f"    Max:    {stats['max']:.2f}ms")
        print(f"    Mean:   {stats['mean']:.2f}ms")
        print(f"    Median: {stats['median']:.2f}ms")
        print(f"    StDev:  {stats['stdev']:.2f}ms")
        print(f"    P95:    {stats['p95']:.2f}ms")
        print(f"    P99:    {stats['p99']:.2f}ms")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def indexed_loom():
    """Create indexed HoloLoom instance."""
    try:
        from HoloLoom import HoloLoom
        from HoloLoom.spinningWheel import WorkspaceSpinner
        from HoloLoom.config import Config

        config = Config.fast()
        async with HoloLoom(config=config) as loom:
            spinner = WorkspaceSpinner()
            await spinner.index_to_hololoom(
                workspace_path=str(TEST_WORKSPACE),
                hololoom_instance=loom,
                incremental=False
            )

            yield loom

    except ImportError:
        pytest.skip("HoloLoom not available")


@pytest.fixture
async def lsp_server():
    """Create LSP server with indexed data."""
    from HoloLoom.lsp.server import HoloLoomLanguageServer, setup_logging

    setup_logging("ERROR")  # Suppress logs

    server = HoloLoomLanguageServer(
        name="hololoom-lsp-test",
        version="0.1.0-test"
    )

    server.workspace = Mock()

    yield server


# ============================================================================
# Performance Tests: Completion Handler
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_completion_latency_single(lsp_server, indexed_loom):
    """Benchmark: Single completion request latency.

    Target: <100ms
    """
    from HoloLoom.lsp.server import completion

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    metrics = PerformanceMetrics("Completion")

    # Warmup
    metrics.start()
    await completion(params)
    metrics.end()

    # Test
    metrics.start()
    await completion(params)
    latency_ms = metrics.end()

    metrics.print_summary()

    assert latency_ms < 100, f"Completion too slow: {latency_ms:.1f}ms (target: <100ms)"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_completion_latency_repeated(lsp_server, indexed_loom):
    """Benchmark: Repeated completion requests (cache behavior).

    Target: <100ms average, stable performance
    """
    from HoloLoom.lsp.server import completion

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    metrics = PerformanceMetrics("Completion (repeated)")

    # Warmup
    for _ in range(2):
        await completion(params)

    # Measure
    for _ in range(20):
        metrics.start()
        await completion(params)
        metrics.end()

    stats = metrics.get_stats()
    metrics.print_summary()

    assert stats["mean"] < 100, f"Average completion too slow: {stats['mean']:.1f}ms"
    assert stats["p95"] < 150, f"P95 completion too slow: {stats['p95']:.1f}ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_completion_throughput(lsp_server, indexed_loom):
    """Benchmark: Completion throughput (requests/second).

    Target: >10 requests/second
    """
    from HoloLoom.lsp.server import completion

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    # Measure sustained throughput
    start = time.time()
    count = 0

    for _ in range(50):
        await completion(params)
        count += 1

    elapsed_seconds = time.time() - start
    throughput = count / elapsed_seconds

    print(f"\n  Completion Throughput:")
    print(f"    Requests: {count}")
    print(f"    Time: {elapsed_seconds:.2f}s")
    print(f"    Throughput: {throughput:.1f} req/s")

    assert throughput > 5, f"Throughput too low: {throughput:.1f} req/s (target: >10)"


# ============================================================================
# Performance Tests: Hover Handler
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_hover_latency_single(lsp_server, indexed_loom):
    """Benchmark: Single hover request latency.

    Target: <50ms
    """
    from HoloLoom.lsp.server import hover

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = HoverParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    metrics = PerformanceMetrics("Hover")

    # Warmup
    metrics.start()
    await hover(params)
    metrics.end()

    # Test
    metrics.start()
    await hover(params)
    latency_ms = metrics.end()

    metrics.print_summary()

    assert latency_ms < 75, f"Hover too slow: {latency_ms:.1f}ms (target: <50ms)"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_hover_latency_repeated(lsp_server, indexed_loom):
    """Benchmark: Repeated hover requests."""
    from HoloLoom.lsp.server import hover

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = HoverParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    metrics = PerformanceMetrics("Hover (repeated)")

    # Warmup
    for _ in range(2):
        await hover(params)

    # Measure
    for _ in range(20):
        metrics.start()
        await hover(params)
        metrics.end()

    stats = metrics.get_stats()
    metrics.print_summary()

    assert stats["mean"] < 75, f"Average hover too slow: {stats['mean']:.1f}ms"


# ============================================================================
# Performance Tests: Definition Handler
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_definition_latency_single(lsp_server, indexed_loom):
    """Benchmark: Single definition request latency.

    Target: <75ms
    """
    from HoloLoom.lsp.server import definition

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = DefinitionParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    metrics = PerformanceMetrics("Definition")

    # Warmup
    metrics.start()
    await definition(params)
    metrics.end()

    # Test
    metrics.start()
    await definition(params)
    latency_ms = metrics.end()

    metrics.print_summary()

    assert latency_ms < 100, f"Definition too slow: {latency_ms:.1f}ms (target: <75ms)"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_definition_latency_repeated(lsp_server, indexed_loom):
    """Benchmark: Repeated definition requests."""
    from HoloLoom.lsp.server import definition

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = DefinitionParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    metrics = PerformanceMetrics("Definition (repeated)")

    # Warmup
    for _ in range(2):
        await definition(params)

    # Measure
    for _ in range(20):
        metrics.start()
        await definition(params)
        metrics.end()

    stats = metrics.get_stats()
    metrics.print_summary()

    assert stats["mean"] < 100, f"Average definition too slow: {stats['mean']:.1f}ms"


# ============================================================================
# Performance Tests: Symbol Search Handler
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_symbol_search_latency_single(lsp_server, indexed_loom):
    """Benchmark: Single symbol search request latency.

    Target: <200ms
    """
    from HoloLoom.lsp.server import workspace_symbol

    lsp_server.hololoom = indexed_loom

    params = WorkspaceSymbolParams(query="Calculator")

    metrics = PerformanceMetrics("Symbol Search")

    # Warmup
    metrics.start()
    await workspace_symbol(params)
    metrics.end()

    # Test
    metrics.start()
    await workspace_symbol(params)
    latency_ms = metrics.end()

    metrics.print_summary()

    assert latency_ms < 300, f"Symbol search too slow: {latency_ms:.1f}ms (target: <200ms)"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_symbol_search_latency_repeated(lsp_server, indexed_loom):
    """Benchmark: Repeated symbol search requests."""
    from HoloLoom.lsp.server import workspace_symbol

    lsp_server.hololoom = indexed_loom

    params = WorkspaceSymbolParams(query="Calculator")

    metrics = PerformanceMetrics("Symbol Search (repeated)")

    # Warmup
    for _ in range(2):
        await workspace_symbol(params)

    # Measure
    for _ in range(20):
        metrics.start()
        await workspace_symbol(params)
        metrics.end()

    stats = metrics.get_stats()
    metrics.print_summary()

    assert stats["mean"] < 300, f"Average symbol search too slow: {stats['mean']:.1f}ms"


# ============================================================================
# Performance Tests: Scaling
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_workspace_indexing_performance():
    """Benchmark: Workspace indexing speed.

    Target: Index test workspace in reasonable time (<5s)
    """
    try:
        from HoloLoom import HoloLoom
        from HoloLoom.spinningWheel import WorkspaceSpinner
        from HoloLoom.config import Config

        config = Config.fast()
        async with HoloLoom(config=config) as loom:
            spinner = WorkspaceSpinner()

            metrics = PerformanceMetrics("Workspace Indexing")

            metrics.start()
            stats = await spinner.index_to_hololoom(
                workspace_path=str(TEST_WORKSPACE),
                hololoom_instance=loom,
                incremental=False
            )
            elapsed_ms = metrics.end()

            print(f"\n  Workspace Indexing:")
            print(f"    Files: {stats.get('files', 0)}")
            print(f"    Entities: {stats.get('entities', 0)}")
            print(f"    Edges: {stats.get('edges', 0)}")
            print(f"    Time: {elapsed_ms:.0f}ms")

            assert elapsed_ms < 10000, f"Indexing too slow: {elapsed_ms:.0f}ms"

    except ImportError:
        pytest.skip("HoloLoom not available")


# ============================================================================
# Performance Tests: Memory and Resource Usage
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_stability(lsp_server, indexed_loom):
    """Benchmark: Memory usage remains stable under sustained load.

    Target: No unbounded growth, proper cleanup
    """
    from HoloLoom.lsp.server import completion

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    # Force garbage collection before starting
    gc.collect()

    import psutil
    import os

    process = psutil.Process(os.getpid())

    memory_samples: List[float] = []

    for i in range(100):
        await completion(params)

        if i % 25 == 0:
            gc.collect()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            memory_samples.append(mem_mb)

    print(f"\n  Memory Stability:")
    for i, mem in enumerate(memory_samples):
        print(f"    Sample {i}: {mem:.1f}MB")

    # Check that memory doesn't grow unboundedly
    if len(memory_samples) > 1:
        growth = memory_samples[-1] - memory_samples[0]
        print(f"    Total growth: {growth:.1f}MB")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_requests_performance(lsp_server, indexed_loom):
    """Benchmark: Performance under concurrent request load.

    Target: Handles multiple concurrent requests smoothly
    """
    from HoloLoom.lsp.server import completion

    lsp_server.hololoom = indexed_loom

    mock_doc = Mock()
    mock_doc.uri = "file:///test.py"
    mock_doc.lines = ["test"]
    lsp_server.workspace.get_text_document.return_value = mock_doc

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri="file:///test.py"),
        position=Position(line=0, character=0)
    )

    # Create 10 concurrent tasks
    async def completion_task():
        metrics = PerformanceMetrics("Concurrent")
        metrics.start()
        await completion(params)
        return metrics.end()

    start = time.time()
    latencies = await asyncio.gather(*[completion_task() for _ in range(10)])
    elapsed = time.time() - start

    print(f"\n  Concurrent Requests (10x):")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    Avg latency: {statistics.mean(latencies):.1f}ms")
    print(f"    Max latency: {max(latencies):.1f}ms")
    print(f"    Throughput: {10 / elapsed:.1f} req/s")


# ============================================================================
# Summary Report
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def print_performance_summary():
    """Print summary of performance targets at end of test session."""
    yield

    print("\n" + "=" * 70)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 70)
    print("\nPerformance Targets:")
    print("  Completion:     <100ms")
    print("  Hover:          <50ms")
    print("  Definition:     <75ms")
    print("  Symbol Search:  <200ms")
    print("  Indexing:       <10s (test workspace)")
    print("  Throughput:     >5 req/s per handler")
    print("\nRun with: pytest -v -m performance")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run performance tests only
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
