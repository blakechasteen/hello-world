#!/usr/bin/env python3
"""
HoloLoom LSP Server Performance Benchmarks
===========================================

Performance benchmarking suite for measuring LSP handler latencies.

Measured Metrics:
- Completion latency (target: <100ms)
- Hover latency (target: <50ms)
- Definition latency (target: <75ms)
- Symbol search latency (target: <200ms)
- Server startup (target: <500ms)
- Server shutdown (target: <100ms)

Status: Production Ready (November 2025)
"""

import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lsprotocol.types import (
    InitializeParams,
    CompletionParams,
    Position,
    TextDocumentIdentifier,
    HoverParams,
    DefinitionParams,
    WorkspaceSymbolParams,
    ClientInfo,
)


# ============================================================================
# Benchmark Configuration
# ============================================================================

class BenchmarkTarget:
    """Target latencies for LSP handlers (in milliseconds)."""
    COMPLETION = 100.0
    HOVER = 50.0
    DEFINITION = 75.0
    SYMBOL_SEARCH = 200.0
    SERVER_STARTUP = 500.0
    SERVER_SHUTDOWN = 100.0


# ============================================================================
# Mock Setup
# ============================================================================

def create_mock_hololoom():
    """Create mock HoloLoom with configurable latency."""
    hololoom = AsyncMock()

    class MockMemory:
        def __init__(self, text, rank=0):
            self.text = text
            self.metadata = {"rank": rank, "confidence": 0.95 - rank * 0.05}
            self.timestamp = "2025-11-16T12:00:00"

    async def mock_recall(query, limit=10):
        # Simulate HoloLoom latency (~10-20ms)
        await asyncio.sleep(0.010)
        memories = [
            MockMemory(f"Memory {i} for '{query}'", i)
            for i in range(min(5, limit))
        ]
        return memories

    hololoom.recall = mock_recall
    hololoom.__aenter__ = AsyncMock(return_value=hololoom)
    hololoom.__aexit__ = AsyncMock(return_value=None)

    return hololoom


def create_mock_server(hololoom):
    """Create mock LSP server."""
    from HoloLoom.lsp.server import HoloLoomLanguageServer, setup_logging

    setup_logging("CRITICAL")  # Suppress logs during benchmarks

    server = HoloLoomLanguageServer(
        name="hololoom-lsp-benchmark",
        version="0.1.0",
    )

    server.hololoom = hololoom
    server.workspace = Mock()

    # Mock document
    doc = Mock()
    doc.uri = "file:///benchmark/test.py"
    doc.lines = [
        "from HoloLoom import HoloLoom",
        "loom = HoloLoom()",
        "result = loom.recall('query')",
        "print(result)",
    ]
    server.workspace.get_text_document.return_value = doc

    return server


# ============================================================================
# Benchmark Tests
# ============================================================================

class BenchmarkSuite:
    """Run performance benchmarks on LSP handlers."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results: Dict[str, List[float]] = {}
        self.hololoom = create_mock_hololoom()

    async def benchmark_completion(self, iterations: int = 10) -> Tuple[float, float]:
        """Benchmark completion handler.

        Args:
            iterations: Number of times to run the handler

        Returns:
            (avg_latency_ms, min_latency_ms)
        """
        from HoloLoom.lsp.server import completion

        server = create_mock_server(self.hololoom)
        latencies = []

        for i in range(iterations):
            params = CompletionParams(
                text_document=TextDocumentIdentifier(uri="file:///test.py"),
                position=Position(line=2, character=30)
            )

            start = time.perf_counter()
            await completion(params)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

            latencies.append(elapsed)

        self.results["completion"] = latencies
        avg = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        return avg, min_latency, max_latency

    async def benchmark_hover(self, iterations: int = 10) -> Tuple[float, float]:
        """Benchmark hover handler.

        Args:
            iterations: Number of times to run the handler

        Returns:
            (avg_latency_ms, min_latency_ms)
        """
        from HoloLoom.lsp.server import hover

        server = create_mock_server(self.hololoom)
        latencies = []

        for i in range(iterations):
            params = HoverParams(
                text_document=TextDocumentIdentifier(uri="file:///test.py"),
                position=Position(line=1, character=7)  # On "HoloLoom"
            )

            start = time.perf_counter()
            await hover(params)
            elapsed = (time.perf_counter() - start) * 1000

            latencies.append(elapsed)

        self.results["hover"] = latencies
        avg = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        return avg, min_latency, max_latency

    async def benchmark_definition(self, iterations: int = 10) -> Tuple[float, float]:
        """Benchmark definition handler.

        Args:
            iterations: Number of times to run the handler

        Returns:
            (avg_latency_ms, min_latency_ms)
        """
        from HoloLoom.lsp.server import definition

        server = create_mock_server(self.hololoom)
        latencies = []

        for i in range(iterations):
            params = DefinitionParams(
                text_document=TextDocumentIdentifier(uri="file:///test.py"),
                position=Position(line=1, character=7)
            )

            start = time.perf_counter()
            await definition(params)
            elapsed = (time.perf_counter() - start) * 1000

            latencies.append(elapsed)

        self.results["definition"] = latencies
        avg = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        return avg, min_latency, max_latency

    async def benchmark_workspace_symbol(self, iterations: int = 10) -> Tuple[float, float]:
        """Benchmark workspace symbol search.

        Args:
            iterations: Number of times to run the handler

        Returns:
            (avg_latency_ms, min_latency_ms)
        """
        from HoloLoom.lsp.server import workspace_symbol

        server = create_mock_server(self.hololoom)
        latencies = []

        for i in range(iterations):
            params = WorkspaceSymbolParams(query="Thompson")

            start = time.perf_counter()
            await workspace_symbol(params)
            elapsed = (time.perf_counter() - start) * 1000

            latencies.append(elapsed)

        self.results["workspace_symbol"] = latencies
        avg = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        return avg, min_latency, max_latency

    async def benchmark_server_initialization(self) -> float:
        """Benchmark server initialization.

        Returns:
            Latency in milliseconds
        """
        from HoloLoom.lsp.server import initialize

        start = time.perf_counter()

        params = InitializeParams(
            process_id=12345,
            root_uri="file:///workspace",
            client_info=ClientInfo(name="test", version="1.0"),
            capabilities={}
        )

        await initialize(params)

        elapsed = (time.perf_counter() - start) * 1000
        self.results["server_init"] = [elapsed]
        return elapsed

    async def benchmark_server_shutdown(self) -> float:
        """Benchmark server shutdown.

        Returns:
            Latency in milliseconds
        """
        from HoloLoom.lsp.server import shutdown

        server = create_mock_server(self.hololoom)

        start = time.perf_counter()
        await shutdown(None)
        elapsed = (time.perf_counter() - start) * 1000

        self.results["server_shutdown"] = [elapsed]
        return elapsed


# ============================================================================
# Results Reporting
# ============================================================================

class BenchmarkReport:
    """Generate benchmark reports."""

    def __init__(self, suite: BenchmarkSuite):
        """Initialize report generator.

        Args:
            suite: BenchmarkSuite with results
        """
        self.suite = suite

    def print_summary(self):
        """Print human-readable benchmark summary."""
        print("\n" + "=" * 80)
        print("HoloLoom LSP Server Performance Benchmark Report")
        print("=" * 80 + "\n")

        benchmarks = [
            ("Initialization", "server_init", BenchmarkTarget.SERVER_STARTUP),
            ("Completion", "completion", BenchmarkTarget.COMPLETION),
            ("Hover", "hover", BenchmarkTarget.HOVER),
            ("Definition", "definition", BenchmarkTarget.DEFINITION),
            ("Symbol Search", "workspace_symbol", BenchmarkTarget.SYMBOL_SEARCH),
            ("Shutdown", "server_shutdown", BenchmarkTarget.SERVER_SHUTDOWN),
        ]

        results_summary = []

        for name, key, target in benchmarks:
            if key not in self.suite.results:
                continue

            latencies = self.suite.results[key]
            avg = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)

            # Status indicator
            if avg <= target:
                status = "✅ PASS"
            else:
                status = "⚠️  WARN"

            print(f"{name:20} {status:10} {avg:8.2f}ms  (min: {min_lat:6.2f}ms, "
                  f"max: {max_lat:6.2f}ms) [target: {target:.0f}ms]")

            results_summary.append({
                "handler": name,
                "avg_ms": round(avg, 2),
                "min_ms": round(min_lat, 2),
                "max_ms": round(max_lat, 2),
                "target_ms": target,
                "passed": avg <= target,
                "iterations": len(latencies)
            })

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        passed = sum(1 for r in results_summary if r["passed"])
        total = len(results_summary)

        print(f"\n✅ Passed: {passed}/{total}")

        if passed == total:
            print("🎉 All benchmarks within target latencies!")
        else:
            print("⚠️  Some benchmarks exceeded targets. See details above.")

        print()

        return results_summary

    def save_json(self, filepath: str):
        """Save results to JSON file.

        Args:
            filepath: Path to save JSON report
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.suite.results,
            "summaries": []
        }

        benchmarks = [
            ("server_init", "Initialization", BenchmarkTarget.SERVER_STARTUP),
            ("completion", "Completion", BenchmarkTarget.COMPLETION),
            ("hover", "Hover", BenchmarkTarget.HOVER),
            ("definition", "Definition", BenchmarkTarget.DEFINITION),
            ("workspace_symbol", "Symbol Search", BenchmarkTarget.SYMBOL_SEARCH),
            ("server_shutdown", "Shutdown", BenchmarkTarget.SERVER_SHUTDOWN),
        ]

        for key, name, target in benchmarks:
            if key in self.suite.results:
                latencies = self.suite.results[key]
                avg = sum(latencies) / len(latencies)
                report["summaries"].append({
                    "handler": name,
                    "avg_ms": round(avg, 2),
                    "target_ms": target,
                    "passed": avg <= target
                })

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Results saved to {filepath}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run all benchmarks."""
    print("Starting HoloLoom LSP Server Performance Benchmarks...\n")

    suite = BenchmarkSuite()

    try:
        # Run benchmarks (iterations per test)
        print("🔍 Benchmarking server initialization...")
        init_latency = await suite.benchmark_server_initialization()

        print("🔍 Benchmarking completion handler (10 iterations)...")
        comp_avg, comp_min, comp_max = await suite.benchmark_completion(iterations=10)

        print("🔍 Benchmarking hover handler (10 iterations)...")
        hover_avg, hover_min, hover_max = await suite.benchmark_hover(iterations=10)

        print("🔍 Benchmarking definition handler (10 iterations)...")
        def_avg, def_min, def_max = await suite.benchmark_definition(iterations=10)

        print("🔍 Benchmarking workspace symbol search (10 iterations)...")
        sym_avg, sym_min, sym_max = await suite.benchmark_workspace_symbol(iterations=10)

        print("🔍 Benchmarking server shutdown...")
        shutdown_latency = await suite.benchmark_server_shutdown()

        # Generate report
        report = BenchmarkReport(suite)
        results = report.print_summary()

        # Save JSON report
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        report_file = output_dir / "benchmark_report.json"
        report.save_json(str(report_file))

        # Return success/failure
        all_passed = all(r["passed"] for r in results)
        return 0 if all_passed else 1

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
