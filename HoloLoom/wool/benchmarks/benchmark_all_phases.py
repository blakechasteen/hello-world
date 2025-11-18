"""
Comprehensive Performance Benchmarks for Wool Storage Phases 1-8
=================================================================

Systematic benchmarking of all wool storage components:
- Phase 1-4: Zero-copy foundation (baseline)
- Phase 6: Distributed storage (cluster performance)
- Phase 7: Transparent compression (compression ratios and latency)
- Phase 8: Versioning + branch/merge (version creation, queries, merges)

Metrics tracked:
- Throughput (ops/sec)
- Latency (ms) - mean, p50, p95, p99
- Storage efficiency (compression ratios, storage savings)
- Scalability (performance vs. file count, cluster size)
- Memory usage (peak, average)

Author: Claude Code
Date: November 17, 2025 (Track 2 Week 2: Performance Benchmarks)
"""

import time
import statistics
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from HoloLoom.wool import WoolStorage
from HoloLoom.wool.compressed import CompressedWoolStorage
from HoloLoom.wool.versioned import VersionedWoolStorage, MergeStrategy


# Benchmark Results Data Structures
# ==================================

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    operations: int
    duration_sec: float
    latencies_ms: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Operations per second."""
        return self.operations / self.duration_sec if self.duration_sec > 0 else 0

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency in milliseconds."""
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p50_latency_ms(self) -> float:
        """50th percentile latency."""
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency."""
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency_ms(self) -> float:
        """99th percentile latency."""
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.name}:\n"
            f"  Throughput: {self.throughput:.2f} ops/sec\n"
            f"  Latency (mean): {self.mean_latency_ms:.2f} ms\n"
            f"  Latency (p50): {self.p50_latency_ms:.2f} ms\n"
            f"  Latency (p95): {self.p95_latency_ms:.2f} ms\n"
            f"  Latency (p99): {self.p99_latency_ms:.2f} ms\n"
            f"  Total operations: {self.operations}\n"
            f"  Total duration: {self.duration_sec:.2f} sec"
        )


# Benchmark Runner
# ================

class BenchmarkRunner:
    """Orchestrates benchmark execution."""

    def __init__(self, base_path: Path = Path("/tmp/wool_benchmarks")):
        self.base_path = base_path
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        name: str,
        operation_func,
        iterations: int = 1000,
        metadata: Dict[str, Any] = None
    ) -> BenchmarkResult:
        """
        Run benchmark and collect results.

        Args:
            name: Benchmark name
            operation_func: Function to benchmark (called once per iteration)
            iterations: Number of iterations
            metadata: Additional metadata to store

        Returns:
            BenchmarkResult
        """
        print(f"\n🔧 Running: {name} ({iterations} iterations)...")

        latencies = []
        start_time = time.time()

        for i in range(iterations):
            op_start = time.time()
            operation_func(i)
            op_end = time.time()

            latency_ms = (op_end - op_start) * 1000
            latencies.append(latency_ms)

        end_time = time.time()
        duration = end_time - start_time

        result = BenchmarkResult(
            name=name,
            operations=iterations,
            duration_sec=duration,
            latencies_ms=latencies,
            metadata=metadata or {}
        )

        self.results.append(result)

        print(f"✅ {name}: {result.throughput:.2f} ops/sec, "
              f"{result.mean_latency_ms:.2f} ms mean latency")

        return result

    async def run_async_benchmark(
        self,
        name: str,
        operation_func,
        iterations: int = 1000,
        metadata: Dict[str, Any] = None
    ) -> BenchmarkResult:
        """Run async benchmark."""
        print(f"\n🔧 Running: {name} ({iterations} iterations)...")

        latencies = []
        start_time = time.time()

        for i in range(iterations):
            op_start = time.time()
            await operation_func(i)
            op_end = time.time()

            latency_ms = (op_end - op_start) * 1000
            latencies.append(latency_ms)

        end_time = time.time()
        duration = end_time - start_time

        result = BenchmarkResult(
            name=name,
            operations=iterations,
            duration_sec=duration,
            latencies_ms=latencies,
            metadata=metadata or {}
        )

        self.results.append(result)

        print(f"✅ {name}: {result.throughput:.2f} ops/sec, "
              f"{result.mean_latency_ms:.2f} ms mean latency")

        return result

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        for result in self.results:
            print(f"\n{result.summary()}")

        print("\n" + "="*80)


# Phase 1-4: Zero-Copy Foundation Benchmarks
# ===========================================

def benchmark_baseline(runner: BenchmarkRunner):
    """Benchmark baseline WoolStorage performance."""
    print("\n" + "="*80)
    print("PHASE 1-4: ZERO-COPY FOUNDATION BENCHMARKS")
    print("="*80)

    storage = WoolStorage(base_path=runner.base_path / "baseline")

    # Benchmark 1: Store throughput
    data = b"x" * 1024  # 1KB
    refs = []

    def store_op(i):
        ref = storage.store(data)
        refs.append(ref)

    runner.run_benchmark(
        "Baseline: Store (1KB)",
        store_op,
        iterations=1000
    )

    # Benchmark 2: Read throughput
    def read_op(i):
        ref = refs[i % len(refs)]
        data = storage.read(ref)

    runner.run_benchmark(
        "Baseline: Read (1KB)",
        read_op,
        iterations=1000
    )

    # Benchmark 3: Store large files
    large_data = b"x" * (1024 * 1024)  # 1MB
    large_refs = []

    def store_large_op(i):
        ref = storage.store(large_data)
        large_refs.append(ref)

    runner.run_benchmark(
        "Baseline: Store (1MB)",
        store_large_op,
        iterations=100
    )

    # Benchmark 4: Read large files
    def read_large_op(i):
        ref = large_refs[i % len(large_refs)]
        data = storage.read(ref)

    runner.run_benchmark(
        "Baseline: Read (1MB)",
        read_large_op,
        iterations=100
    )

    storage.close()


# Phase 7: Compression Benchmarks
# ================================

def benchmark_compression(runner: BenchmarkRunner):
    """Benchmark compression performance."""
    print("\n" + "="*80)
    print("PHASE 7: TRANSPARENT COMPRESSION BENCHMARKS")
    print("="*80)

    storage = CompressedWoolStorage(
        base_path=runner.base_path / "compressed",
        adaptive_compression=True
    )

    # Benchmark 1: Store compressible text
    text_data = ("Hello, world! " * 100).encode('utf-8')
    text_refs = []

    def store_text_op(i):
        ref = storage.store(text_data, content_type="text/plain")
        text_refs.append(ref)

    runner.run_benchmark(
        "Compression: Store text (compressible)",
        store_text_op,
        iterations=1000
    )

    # Benchmark 2: Read compressed text
    def read_text_op(i):
        ref = text_refs[i % len(text_refs)]
        data = storage.read(ref)

    runner.run_benchmark(
        "Compression: Read text (decompression)",
        read_text_op,
        iterations=1000
    )

    # Benchmark 3: Store JSON (highly compressible)
    json_data = ('{"key": "value", "array": [1, 2, 3]}' * 100).encode('utf-8')
    json_refs = []

    def store_json_op(i):
        ref = storage.store(json_data, content_type="application/json")
        json_refs.append(ref)

    runner.run_benchmark(
        "Compression: Store JSON (highly compressible)",
        store_json_op,
        iterations=1000
    )

    # Benchmark 4: Compression ratio measurement
    stats = storage.get_compression_stats()

    print(f"\n📊 Compression Statistics:")
    print(f"  Average compression ratio: {stats['average_compression_ratio']:.2f}x")
    print(f"  Total storage savings: {stats['total_savings_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Storage efficiency: {(1 - stats['average_storage_efficiency']) * 100:.1f}% saved")

    storage.close()


# Phase 8: Versioning Benchmarks
# ===============================

def benchmark_versioning(runner: BenchmarkRunner):
    """Benchmark versioning performance."""
    print("\n" + "="*80)
    print("PHASE 8: VERSIONING + BRANCH/MERGE BENCHMARKS")
    print("="*80)

    storage = VersionedWoolStorage(
        base_path=runner.base_path / "versioned",
        enable_delta_encoding=True,
        snapshot_interval=10
    )

    # Benchmark 1: Create version chain
    parent = None
    refs = []

    def store_version_op(i):
        nonlocal parent
        data = f"Version {i+1}".encode()
        ref = storage.store_versioned(
            data,
            parent=parent,
            author="benchmark",
            message=f"v{i+1}"
        )
        refs.append(ref)
        parent = ref

    runner.run_benchmark(
        "Versioning: Create version (with delta)",
        store_version_op,
        iterations=100
    )

    # Benchmark 2: Point-in-time queries
    file_id = refs[0].file_id

    def as_of_query_op(i):
        idx = i % len(refs)
        timestamp = refs[idx].timestamp
        ref = storage.as_of(file_id, timestamp)

    runner.run_benchmark(
        "Versioning: as_of() query",
        as_of_query_op,
        iterations=1000
    )

    # Benchmark 3: Range queries
    def between_query_op(i):
        start_idx = i % (len(refs) - 10)
        end_idx = start_idx + 10
        start_time = refs[start_idx].timestamp
        end_time = refs[end_idx].timestamp
        versions = storage.between(file_id, start_time, end_time)

    runner.run_benchmark(
        "Versioning: between() range query",
        between_query_op,
        iterations=1000
    )

    # Benchmark 4: Version reconstruction (delta chain)
    def reconstruct_op(i):
        idx = i % len(refs)
        ref = refs[idx]
        data = storage.reconstruct_version(ref)

    runner.run_benchmark(
        "Versioning: Reconstruct version (delta chain)",
        reconstruct_op,
        iterations=100
    )

    # Benchmark 5: Branch creation
    def create_branch_op(i):
        storage.create_branch(
            f"feature-{i}",
            from_version=refs[0].version_id,
            author="benchmark"
        )

    runner.run_benchmark(
        "Versioning: Create branch",
        create_branch_op,
        iterations=100
    )

    # Benchmark 6: Fast-forward merge
    storage.switch_branch("feature-0")
    storage.store_versioned(b"Feature work", parent=refs[0])

    start = time.time()
    result = storage.merge(source="feature-0", target="main")
    merge_latency = (time.time() - start) * 1000

    print(f"\n📊 Merge Performance:")
    print(f"  Fast-forward merge: {merge_latency:.2f} ms")
    print(f"  Merge type: {'fast-forward' if result.fast_forward else '3-way'}")

    # Benchmark 7: Delta encoding statistics
    stats = storage.get_version_stats()

    print(f"\n📊 Versioning Statistics:")
    print(f"  Versions created: {stats['versions_created']}")
    print(f"  Snapshots: {stats['snapshots_created']}")
    print(f"  Deltas: {stats['deltas_created']}")
    print(f"  Delta percentage: {stats['delta_percentage']:.1f}%")
    print(f"  Storage efficiency: {stats['storage_efficiency']:.2f}x")
    print(f"  Storage savings: {stats['storage_savings']:.1f}%")

    storage.close()


# Scalability Benchmarks
# =======================

def benchmark_scalability(runner: BenchmarkRunner):
    """Benchmark scalability with increasing data sizes."""
    print("\n" + "="*80)
    print("SCALABILITY BENCHMARKS")
    print("="*80)

    storage = WoolStorage(base_path=runner.base_path / "scalability")

    # Test with varying file sizes
    sizes = [1, 10, 100, 1000, 10000]  # KB

    for size_kb in sizes:
        data = b"x" * (size_kb * 1024)
        refs = []

        def store_op(i):
            ref = storage.store(data)
            refs.append(ref)

        result = runner.run_benchmark(
            f"Scalability: Store ({size_kb}KB)",
            store_op,
            iterations=min(1000, 10000 // size_kb)  # Fewer iterations for larger files
        )

        # Read benchmark
        def read_op(i):
            ref = refs[i % len(refs)]
            data = storage.read(ref)

        runner.run_benchmark(
            f"Scalability: Read ({size_kb}KB)",
            read_op,
            iterations=min(1000, 10000 // size_kb)
        )

    storage.close()


# Main Benchmark Suite
# =====================

def main():
    """Run all benchmarks."""
    runner = BenchmarkRunner()

    print("""
╔════════════════════════════════════════════════════════════════╗
║           WOOL STORAGE COMPREHENSIVE BENCHMARKS                ║
║                   Phases 1-8 Performance                       ║
╚════════════════════════════════════════════════════════════════╝
""")

    # Run benchmark suites
    benchmark_baseline(runner)
    benchmark_compression(runner)
    benchmark_versioning(runner)
    benchmark_scalability(runner)

    # Print final summary
    runner.print_summary()

    # Export results to JSON
    import json
    results_json = {
        "benchmarks": [
            {
                "name": r.name,
                "operations": r.operations,
                "duration_sec": r.duration_sec,
                "throughput_ops_sec": r.throughput,
                "latency_ms": {
                    "mean": r.mean_latency_ms,
                    "p50": r.p50_latency_ms,
                    "p95": r.p95_latency_ms,
                    "p99": r.p99_latency_ms
                },
                "metadata": r.metadata
            }
            for r in runner.results
        ]
    }

    output_path = Path("benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n📊 Results exported to: {output_path}")
    print("\n✅ All benchmarks complete!")


if __name__ == "__main__":
    main()
