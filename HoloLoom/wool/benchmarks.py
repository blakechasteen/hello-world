"""
Wool Storage Benchmarks
=======================

Comprehensive performance benchmarks for zero-copy architecture.

Tests:
- WoolStorage performance (store, read, deduplication)
- Memory usage (legacy vs zero-copy)
- Graph query performance (with/without text resolution)
- Migration performance (legacy → zero-copy)
- Real-world scenarios (large documents, batch ingestion)

Author: Claude Code
Date: November 17, 2025
"""

import time
import tracemalloc
import asyncio
from pathlib import Path
from typing import List, Dict
import tempfile
import shutil

from HoloLoom.Documentation.types import MemoryShard
from .storage import WoolStorage, WoolReference
from .text_reference import TextReference
from .zerocopy_shard import ZeroCopyMemoryShard, convert_memory_shard_to_zerocopy
from .hybrid_kg import HybridKG, HybridKGStats


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_test_text(size_kb: int) -> str:
    """Generate test text of specified size."""
    base = "The quick brown fox jumps over the lazy dog. " * 100
    return base * (size_kb * 1024 // len(base))


def generate_memory_shard(text: str, idx: int) -> MemoryShard:
    """Generate test MemoryShard."""
    return MemoryShard(
        id=f"shard_{idx}",
        episode=f"episode_{idx // 10}",
        text=text,
        entities=["entity_a", "entity_b"],
        motifs=["motif_x", "motif_y"],
        metadata={"source": "benchmark", "index": idx}
    )


# ============================================================================
# Benchmark 1: WoolStorage Performance
# ============================================================================

class WoolStorageBenchmark:
    """Benchmark WoolStorage operations."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wool = WoolStorage(base_path=self.temp_dir / "wool")

    def cleanup(self):
        """Cleanup temporary files."""
        self.wool.close()
        shutil.rmtree(self.temp_dir)

    def benchmark_store(self, n_files: int = 1000, file_size_kb: int = 10):
        """Benchmark file storage."""
        print(f"\n📦 Benchmark: Store {n_files} files ({file_size_kb}KB each)")

        text = generate_test_text(file_size_kb)
        data = text.encode('utf-8')

        start = time.perf_counter()
        refs = []
        for i in range(n_files):
            ref = self.wool.store(data, content_type='text/plain')
            refs.append(ref)
        elapsed = time.perf_counter() - start

        stats = self.wool.get_stats()

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per file: {elapsed / n_files * 1000:.2f}ms")
        print(f"  Files stored: {stats['files_stored']}")
        print(f"  Deduplications: {stats['deduplications']}")
        print(f"  Dedup rate: {stats['dedup_rate']:.1%}")

        return {
            'total_time': elapsed,
            'avg_time_ms': elapsed / n_files * 1000,
            'files_stored': stats['files_stored'],
            'deduplications': stats['deduplications']
        }

    def benchmark_read_cold(self, n_reads: int = 100):
        """Benchmark cold reads (no cache)."""
        print(f"\n📖 Benchmark: Cold reads ({n_reads} reads)")

        # Create test file
        text = generate_test_text(10)
        data = text.encode('utf-8')
        ref = self.wool.store(data)

        # Clear cache
        self.wool.clear_cache()

        start = time.perf_counter()
        for _ in range(n_reads):
            view = self.wool.read(ref)
            _ = view.tobytes()  # Force read
        elapsed = time.perf_counter() - start

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per read: {elapsed / n_reads * 1000:.2f}ms")

        return {
            'total_time': elapsed,
            'avg_time_ms': elapsed / n_reads * 1000
        }

    def benchmark_read_warm(self, n_reads: int = 1000):
        """Benchmark warm reads (cached mmap)."""
        print(f"\n📖 Benchmark: Warm reads ({n_reads} reads)")

        # Create test file
        text = generate_test_text(10)
        data = text.encode('utf-8')
        ref = self.wool.store(data)

        # Prime cache
        _ = self.wool.read(ref)

        start = time.perf_counter()
        for _ in range(n_reads):
            view = self.wool.read(ref)
            _ = view.tobytes()  # Force read
        elapsed = time.perf_counter() - start

        stats = self.wool.get_stats()

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per read: {elapsed / n_reads * 1000:.2f}ms")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Speedup vs cold: {10 / (elapsed / n_reads * 1000):.1f}x")

        return {
            'total_time': elapsed,
            'avg_time_ms': elapsed / n_reads * 1000,
            'cache_hit_rate': stats['cache_hit_rate']
        }


# ============================================================================
# Benchmark 2: Memory Usage Comparison
# ============================================================================

class MemoryBenchmark:
    """Benchmark memory usage (legacy vs zero-copy)."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wool = WoolStorage(base_path=self.temp_dir / "wool")

    def cleanup(self):
        """Cleanup temporary files."""
        self.wool.close()
        shutil.rmtree(self.temp_dir)

    def benchmark_legacy_shards(self, n_shards: int = 1000, text_size_kb: int = 1):
        """Benchmark legacy MemoryShard memory usage."""
        print(f"\n💾 Benchmark: Legacy shards ({n_shards} shards, {text_size_kb}KB each)")

        text = generate_test_text(text_size_kb)

        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        shards = []
        for i in range(n_shards):
            shard = generate_memory_shard(text, i)
            shards.append(shard)

        end_mem = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        mem_used_mb = (end_mem - start_mem) / (1024 * 1024)
        mem_per_shard_kb = (end_mem - start_mem) / n_shards / 1024

        print(f"  Total memory: {mem_used_mb:.2f} MB")
        print(f"  Per shard: {mem_per_shard_kb:.2f} KB")

        return {
            'total_memory_mb': mem_used_mb,
            'per_shard_kb': mem_per_shard_kb
        }

    def benchmark_zerocopy_shards(self, n_shards: int = 1000, text_size_kb: int = 1):
        """Benchmark zero-copy MemoryShard memory usage."""
        print(f"\n💾 Benchmark: Zero-copy shards ({n_shards} shards, {text_size_kb}KB each)")

        text = generate_test_text(text_size_kb)
        text_bytes = text.encode('utf-8')

        # Store text in wool
        wool_ref = self.wool.store(text_bytes)

        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        shards = []
        for i in range(n_shards):
            text_ref = TextReference(
                file_id=wool_ref.file_id,
                offset=0,
                length=len(text_bytes)
            )
            shard = ZeroCopyMemoryShard(
                id=f"shard_{i}",
                episode=f"episode_{i // 10}",
                text_ref=text_ref,
                entities=["entity_a", "entity_b"],
                motifs=["motif_x", "motif_y"],
                metadata={"source": "benchmark", "index": i}
            )
            shards.append(shard)

        end_mem = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        mem_used_mb = (end_mem - start_mem) / (1024 * 1024)
        mem_per_shard_kb = (end_mem - start_mem) / n_shards / 1024

        print(f"  Total memory: {mem_used_mb:.2f} MB")
        print(f"  Per shard: {mem_per_shard_kb:.2f} KB")

        return {
            'total_memory_mb': mem_used_mb,
            'per_shard_kb': mem_per_shard_kb
        }


# ============================================================================
# Benchmark 3: Graph Query Performance
# ============================================================================

class GraphQueryBenchmark:
    """Benchmark graph query performance."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wool = WoolStorage(base_path=self.temp_dir / "wool")
        self.kg = HybridKG(wool_storage=self.wool)

    def cleanup(self):
        """Cleanup temporary files."""
        self.wool.close()
        shutil.rmtree(self.temp_dir)

    def benchmark_legacy_query(self, n_nodes: int = 1000, n_queries: int = 100):
        """Benchmark legacy node queries."""
        print(f"\n🔍 Benchmark: Legacy node queries ({n_nodes} nodes, {n_queries} queries)")

        # Create legacy nodes
        text = generate_test_text(1)
        for i in range(n_nodes):
            shard = generate_memory_shard(text, i)
            self.kg.add_shard(shard)

        # Query nodes
        start = time.perf_counter()
        for i in range(n_queries):
            node_id = f"shard_{i % n_nodes}"
            text = self.kg.get_text(node_id)
        elapsed = time.perf_counter() - start

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per query: {elapsed / n_queries * 1000:.2f}ms")

        return {
            'total_time': elapsed,
            'avg_time_ms': elapsed / n_queries * 1000
        }

    def benchmark_zerocopy_query(self, n_nodes: int = 1000, n_queries: int = 100):
        """Benchmark zero-copy node queries."""
        print(f"\n🔍 Benchmark: Zero-copy node queries ({n_nodes} nodes, {n_queries} queries)")

        # Create zero-copy nodes
        text = generate_test_text(1)
        text_bytes = text.encode('utf-8')
        wool_ref = self.wool.store(text_bytes)

        for i in range(n_nodes):
            text_ref = TextReference(
                file_id=wool_ref.file_id,
                offset=0,
                length=len(text_bytes)
            )
            shard = ZeroCopyMemoryShard(
                id=f"zerocopy_{i}",
                episode=f"episode_{i // 10}",
                text_ref=text_ref,
                entities=["entity_a", "entity_b"],
                motifs=["motif_x", "motif_y"]
            )
            self.kg.add_shard(shard)

        # Query nodes
        start = time.perf_counter()
        for i in range(n_queries):
            node_id = f"zerocopy_{i % n_nodes}"
            text = self.kg.get_text(node_id)
        elapsed = time.perf_counter() - start

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per query: {elapsed / n_queries * 1000:.2f}ms")

        return {
            'total_time': elapsed,
            'avg_time_ms': elapsed / n_queries * 1000
        }


# ============================================================================
# Benchmark 4: Migration Performance
# ============================================================================

class MigrationBenchmark:
    """Benchmark migration performance."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wool = WoolStorage(base_path=self.temp_dir / "wool")
        self.kg = HybridKG(wool_storage=self.wool)

    def cleanup(self):
        """Cleanup temporary files."""
        self.wool.close()
        shutil.rmtree(self.temp_dir)

    def benchmark_migration(self, n_nodes: int = 1000):
        """Benchmark legacy → zero-copy migration."""
        print(f"\n🔄 Benchmark: Migration ({n_nodes} nodes)")

        # Create legacy nodes
        text = generate_test_text(1)
        for i in range(n_nodes):
            shard = generate_memory_shard(text, i)
            self.kg.add_shard(shard)

        stats_before = self.kg.get_stats()

        # Migrate
        start = time.perf_counter()
        migrated = self.kg.migrate_all(batch_size=100)
        elapsed = time.perf_counter() - start

        stats_after = self.kg.get_stats()

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per node: {elapsed / n_nodes * 1000:.2f}ms")
        print(f"  Migrated: {migrated}/{n_nodes}")
        print(f"  Memory savings: {stats_after.memory_savings_mb:.2f} MB")

        return {
            'total_time': elapsed,
            'avg_time_ms': elapsed / n_nodes * 1000,
            'migrated': migrated,
            'memory_savings_mb': stats_after.memory_savings_mb
        }


# ============================================================================
# Benchmark 5: Real-World Scenarios
# ============================================================================

class RealWorldBenchmark:
    """Benchmark real-world scenarios."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.wool = WoolStorage(base_path=self.temp_dir / "wool")

    def cleanup(self):
        """Cleanup temporary files."""
        self.wool.close()
        shutil.rmtree(self.temp_dir)

    def benchmark_large_document(self, doc_size_mb: int = 100):
        """Benchmark large document ingestion (e.g., 100MB PDF)."""
        print(f"\n📄 Benchmark: Large document ingestion ({doc_size_mb}MB)")

        # Generate large text
        text = generate_test_text(doc_size_mb * 1024)
        data = text.encode('utf-8')

        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        # Store
        start = time.perf_counter()
        ref = self.wool.store(data, content_type='application/pdf')
        elapsed_store = time.perf_counter() - start

        peak_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # Read
        start = time.perf_counter()
        view = self.wool.read(ref)
        _ = view.tobytes()
        elapsed_read = time.perf_counter() - start

        mem_overhead = (peak_mem - start_mem) / (1024 * 1024)

        print(f"  Store time: {elapsed_store:.3f}s")
        print(f"  Read time: {elapsed_read:.3f}s")
        print(f"  Peak memory overhead: {mem_overhead:.2f} MB")
        print(f"  Memory ratio: {mem_overhead / doc_size_mb:.2f}x")

        return {
            'store_time': elapsed_store,
            'read_time': elapsed_read,
            'memory_overhead_mb': mem_overhead
        }

    def benchmark_batch_ingestion(self, n_documents: int = 1000, doc_size_kb: int = 10):
        """Benchmark batch document ingestion."""
        print(f"\n📚 Benchmark: Batch ingestion ({n_documents} docs, {doc_size_kb}KB each)")

        # Generate documents
        texts = [generate_test_text(doc_size_kb) for _ in range(n_documents)]

        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        # Ingest
        start = time.perf_counter()
        refs = []
        for text in texts:
            data = text.encode('utf-8')
            ref = self.wool.store(data)
            refs.append(ref)
        elapsed = time.perf_counter() - start

        peak_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        stats = self.wool.get_stats()
        mem_overhead = (peak_mem - start_mem) / (1024 * 1024)

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per doc: {elapsed / n_documents * 1000:.2f}ms")
        print(f"  Throughput: {n_documents / elapsed:.1f} docs/sec")
        print(f"  Dedup rate: {stats['dedup_rate']:.1%}")
        print(f"  Peak memory: {mem_overhead:.2f} MB")

        return {
            'total_time': elapsed,
            'avg_time_ms': elapsed / n_documents * 1000,
            'throughput': n_documents / elapsed,
            'dedup_rate': stats['dedup_rate']
        }


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_all_benchmarks():
    """Run all benchmarks and generate report."""
    print("=" * 80)
    print("🧪 WOOL STORAGE BENCHMARKS")
    print("=" * 80)

    results = {}

    # Benchmark 1: WoolStorage Performance
    print("\n" + "=" * 80)
    print("1️⃣  WoolStorage Performance")
    print("=" * 80)
    bench1 = WoolStorageBenchmark()
    try:
        results['store'] = bench1.benchmark_store(n_files=1000, file_size_kb=10)
        results['read_cold'] = bench1.benchmark_read_cold(n_reads=100)
        results['read_warm'] = bench1.benchmark_read_warm(n_reads=1000)
    finally:
        bench1.cleanup()

    # Benchmark 2: Memory Usage
    print("\n" + "=" * 80)
    print("2️⃣  Memory Usage Comparison")
    print("=" * 80)
    bench2 = MemoryBenchmark()
    try:
        results['memory_legacy'] = bench2.benchmark_legacy_shards(n_shards=1000, text_size_kb=1)
        results['memory_zerocopy'] = bench2.benchmark_zerocopy_shards(n_shards=1000, text_size_kb=1)

        savings = results['memory_legacy']['per_shard_kb'] - results['memory_zerocopy']['per_shard_kb']
        savings_pct = savings / results['memory_legacy']['per_shard_kb'] * 100
        print(f"\n  💡 Memory Savings: {savings:.2f} KB per shard ({savings_pct:.1f}%)")
    finally:
        bench2.cleanup()

    # Benchmark 3: Graph Query Performance
    print("\n" + "=" * 80)
    print("3️⃣  Graph Query Performance")
    print("=" * 80)
    bench3 = GraphQueryBenchmark()
    try:
        results['query_legacy'] = bench3.benchmark_legacy_query(n_nodes=1000, n_queries=100)
        results['query_zerocopy'] = bench3.benchmark_zerocopy_query(n_nodes=1000, n_queries=100)
    finally:
        bench3.cleanup()

    # Benchmark 4: Migration
    print("\n" + "=" * 80)
    print("4️⃣  Migration Performance")
    print("=" * 80)
    bench4 = MigrationBenchmark()
    try:
        results['migration'] = bench4.benchmark_migration(n_nodes=1000)
    finally:
        bench4.cleanup()

    # Benchmark 5: Real-World Scenarios
    print("\n" + "=" * 80)
    print("5️⃣  Real-World Scenarios")
    print("=" * 80)
    bench5 = RealWorldBenchmark()
    try:
        results['large_doc'] = bench5.benchmark_large_document(doc_size_mb=100)
        results['batch'] = bench5.benchmark_batch_ingestion(n_documents=1000, doc_size_kb=10)
    finally:
        bench5.cleanup()

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"\n✅ WoolStorage Performance:")
    print(f"   - Store: {results['store']['avg_time_ms']:.2f}ms per file")
    print(f"   - Read (cold): {results['read_cold']['avg_time_ms']:.2f}ms")
    print(f"   - Read (warm): {results['read_warm']['avg_time_ms']:.2f}ms")
    print(f"   - Cache speedup: {results['read_cold']['avg_time_ms'] / results['read_warm']['avg_time_ms']:.1f}x")

    print(f"\n✅ Memory Savings:")
    savings = results['memory_legacy']['per_shard_kb'] - results['memory_zerocopy']['per_shard_kb']
    savings_pct = savings / results['memory_legacy']['per_shard_kb'] * 100
    print(f"   - Legacy: {results['memory_legacy']['per_shard_kb']:.2f} KB/shard")
    print(f"   - Zero-copy: {results['memory_zerocopy']['per_shard_kb']:.2f} KB/shard")
    print(f"   - Savings: {savings:.2f} KB ({savings_pct:.1f}%)")

    print(f"\n✅ Migration:")
    print(f"   - Speed: {results['migration']['avg_time_ms']:.2f}ms per node")
    print(f"   - Memory saved: {results['migration']['memory_savings_mb']:.2f} MB")

    print(f"\n✅ Real-World Performance:")
    print(f"   - Large doc (100MB): {results['large_doc']['store_time']:.3f}s store, {results['large_doc']['read_time']:.3f}s read")
    print(f"   - Batch (1000 docs): {results['batch']['throughput']:.1f} docs/sec")
    print(f"   - Deduplication: {results['batch']['dedup_rate']:.1%}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
