"""
TTS Cache Benchmark Demo
=========================

Demonstrates TTS cache performance characteristics:
- Cold cache (first request): ~500ms
- Warm cache (cache hit): ~50ms (10x speedup)
- Cache hit rate after warmup: 60-80%
- Memory usage
- Redis response times

Usage:
    python demos/demo_tts_cache_benchmark.py

Requirements:
    - Redis running on localhost:6379
    - pip install redis pyyaml

Date: November 16, 2025
"""

import asyncio
import time
import random
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.voice.tts_cache import TTSCache, CacheConfig, REDIS_AVAILABLE


# ============================================================================
# Mock TTS Provider (for benchmarking)
# ============================================================================

class MockTTSProvider:
    """Mock TTS provider that simulates synthesis latency"""

    def __init__(self, latency_ms: float = 500.0):
        """
        Initialize mock provider.

        Args:
            latency_ms: Simulated synthesis latency
        """
        self.latency_ms = latency_ms
        self.synthesis_count = 0

    async def synthesize(self, text: str, voice: str = "nova") -> bytes:
        """
        Simulate TTS synthesis.

        Args:
            text: Text to synthesize
            voice: Voice ID

        Returns:
            Mock audio bytes
        """
        # Simulate network/API latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        self.synthesis_count += 1

        # Return mock audio (repeating text)
        audio = f"AUDIO:{text}:{voice}".encode('utf-8')
        return audio


# ============================================================================
# Benchmark Queries
# ============================================================================

# Realistic beekeeping queries
BENCHMARK_QUERIES = [
    # High-frequency queries (should hit cache often)
    "Navigate to the next hive",
    "Show me the hive status",
    "What is the health of this hive",
    "Check for varroa mites",
    "Where is the queen",
    "Display the honey stores",
    "Start the inspection",
    "End the inspection",
    "Record this observation",
    "Show the overlay",

    # Medium-frequency queries
    "Go to hive 1",
    "Go to hive 2",
    "Go to hive 3",
    "Is this hive healthy",
    "How much honey is in this hive",
    "Show me the brood frames",
    "Take a photo",
    "What is the temperature",
    "Display metrics overlay",
    "Thank you",

    # Low-frequency queries (more varied)
    "Show me the inspection history for this hive",
    "Are there any signs of disease in this colony",
    "When was the last time this hive was inspected",
    "Display the environmental data for today",
    "How many bees are estimated in this hive",
]


# ============================================================================
# Benchmark Functions
# ============================================================================

async def benchmark_cold_cache(cache: TTSCache, provider: MockTTSProvider) -> Dict[str, Any]:
    """
    Benchmark cold cache performance.

    Args:
        cache: TTS cache instance
        provider: Mock TTS provider

    Returns:
        Benchmark results
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Cold Cache (First Request)")
    print("=" * 60)

    query = "Navigate to the next hive"
    voice = "nova"

    # Clear cache first
    await cache.clear()

    # Measure cache miss + synthesis
    start_time = time.time()

    # Try cache (should miss)
    cached = await cache.get(query, voice)
    if not cached:
        # Synthesize
        audio = await provider.synthesize(query, voice)
        # Store in cache
        await cache.set(query, voice, "en", audio)

    cold_latency = (time.time() - start_time) * 1000

    print(f"Query: {query}")
    print(f"Latency: {cold_latency:.1f}ms")
    print(f"Cache Hit: No (cold)")
    print(f"Synthesis Count: {provider.synthesis_count}")

    return {
        'query': query,
        'latency_ms': cold_latency,
        'cache_hit': False
    }


async def benchmark_warm_cache(cache: TTSCache, provider: MockTTSProvider) -> Dict[str, Any]:
    """
    Benchmark warm cache performance.

    Args:
        cache: TTS cache instance
        provider: Mock TTS provider

    Returns:
        Benchmark results
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Warm Cache (Repeated Request)")
    print("=" * 60)

    query = "Navigate to the next hive"
    voice = "nova"

    # Reset synthesis count
    synthesis_count_before = provider.synthesis_count

    # Measure cache hit
    start_time = time.time()

    # Try cache (should hit)
    cached = await cache.get(query, voice)
    if not cached:
        # Should not reach here
        audio = await provider.synthesize(query, voice)
        await cache.set(query, voice, "en", audio)

    warm_latency = (time.time() - start_time) * 1000

    cache_hit = (provider.synthesis_count == synthesis_count_before)

    print(f"Query: {query}")
    print(f"Latency: {warm_latency:.1f}ms")
    print(f"Cache Hit: {'Yes' if cache_hit else 'No'}")
    print(f"Synthesis Count: {provider.synthesis_count}")

    if cache_hit:
        speedup = 500.0 / warm_latency  # vs typical synthesis
        print(f"Speedup: {speedup:.1f}x")

    return {
        'query': query,
        'latency_ms': warm_latency,
        'cache_hit': cache_hit
    }


async def benchmark_realistic_workload(
    cache: TTSCache,
    provider: MockTTSProvider,
    num_queries: int = 100
) -> Dict[str, Any]:
    """
    Benchmark realistic workload with query distribution.

    Args:
        cache: TTS cache instance
        provider: Mock TTS provider
        num_queries: Number of queries to execute

    Returns:
        Benchmark results
    """
    print("\n" + "=" * 60)
    print(f"BENCHMARK 3: Realistic Workload ({num_queries} queries)")
    print("=" * 60)

    # Clear cache
    await cache.clear()
    provider.synthesis_count = 0

    # Generate query sequence with realistic distribution
    # - 40% high-frequency (queries 0-9)
    # - 30% medium-frequency (queries 10-19)
    # - 30% low-frequency (queries 20-24)
    queries = []
    for _ in range(num_queries):
        r = random.random()
        if r < 0.4:  # High-frequency
            queries.append(random.choice(BENCHMARK_QUERIES[:10]))
        elif r < 0.7:  # Medium-frequency
            queries.append(random.choice(BENCHMARK_QUERIES[10:20]))
        else:  # Low-frequency
            queries.append(random.choice(BENCHMARK_QUERIES[20:]))

    voice = "nova"
    latencies = []
    cache_hits = 0
    cache_misses = 0

    start_time = time.time()

    # Execute queries
    for i, query in enumerate(queries):
        query_start = time.time()

        # Try cache
        cached = await cache.get(query, voice)
        if cached:
            cache_hits += 1
        else:
            cache_misses += 1
            # Synthesize
            audio = await provider.synthesize(query, voice)
            # Store in cache
            await cache.set(query, voice, "en", audio)

        query_latency = (time.time() - query_start) * 1000
        latencies.append(query_latency)

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_queries} queries...")

    total_time = (time.time() - start_time) * 1000

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    hit_rate = cache_hits / num_queries

    print(f"\nResults:")
    print(f"  Total Queries: {num_queries}")
    print(f"  Cache Hits: {cache_hits}")
    print(f"  Cache Misses: {cache_misses}")
    print(f"  Hit Rate: {hit_rate:.1%}")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    print(f"  Min Latency: {min_latency:.1f}ms")
    print(f"  Max Latency: {max_latency:.1f}ms")
    print(f"  Total Time: {total_time:.1f}ms")
    print(f"  Synthesis Count: {provider.synthesis_count}")

    # Calculate speedup
    time_without_cache = num_queries * 500  # All queries at 500ms
    speedup = time_without_cache / total_time

    print(f"\nPerformance:")
    print(f"  Time without cache: {time_without_cache:.0f}ms")
    print(f"  Time with cache: {total_time:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")

    return {
        'num_queries': num_queries,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'hit_rate': hit_rate,
        'avg_latency_ms': avg_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'total_time_ms': total_time,
        'speedup': speedup
    }


async def benchmark_cache_size(cache: TTSCache) -> Dict[str, Any]:
    """
    Benchmark cache memory usage.

    Args:
        cache: TTS cache instance

    Returns:
        Memory usage statistics
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 4: Cache Memory Usage")
    print("=" * 60)

    size_bytes = await cache.get_cache_size()
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024

    print(f"Cache Size: {size_bytes} bytes ({size_kb:.1f} KB, {size_mb:.2f} MB)")

    # Get cache stats
    stats = cache.get_stats()
    print(f"Cache Hit Rate: {stats['hit_rate']:.1%}")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Speedup Factor: {stats['speedup_factor']:.1f}x")

    return {
        'size_bytes': size_bytes,
        'size_kb': size_kb,
        'size_mb': size_mb,
        'stats': stats
    }


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all benchmarks"""
    print("=" * 60)
    print("TTS Cache Benchmark")
    print("=" * 60)

    # Check Redis availability
    if not REDIS_AVAILABLE:
        print("❌ ERROR: redis package not installed")
        print("Install with: pip install redis")
        return

    # Create cache
    config = CacheConfig(
        redis_host="localhost",
        redis_port=6379,
        enable_cache=True,
        common_phrase_ttl=86400,
        dynamic_phrase_ttl=3600
    )

    # Create mock TTS provider
    provider = MockTTSProvider(latency_ms=500.0)

    try:
        async with TTSCache(config) as cache:
            # Check if Redis is connected
            if not cache._redis_client:
                print("❌ ERROR: Could not connect to Redis")
                print("Make sure Redis is running: docker-compose up -d redis")
                return

            print("✅ Connected to Redis")

            # Run benchmarks
            results = {}

            results['cold_cache'] = await benchmark_cold_cache(cache, provider)
            results['warm_cache'] = await benchmark_warm_cache(cache, provider)
            results['realistic_workload'] = await benchmark_realistic_workload(
                cache, provider, num_queries=100
            )
            results['cache_size'] = await benchmark_cache_size(cache)

            # Summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Cold Cache Latency: {results['cold_cache']['latency_ms']:.1f}ms")
            print(f"Warm Cache Latency: {results['warm_cache']['latency_ms']:.1f}ms")
            print(f"Realistic Workload Hit Rate: {results['realistic_workload']['hit_rate']:.1%}")
            print(f"Realistic Workload Speedup: {results['realistic_workload']['speedup']:.1f}x")
            print(f"Cache Size: {results['cache_size']['size_mb']:.2f} MB")

            # Performance assessment
            print("\n" + "=" * 60)
            print("PERFORMANCE ASSESSMENT")
            print("=" * 60)

            hit_rate = results['realistic_workload']['hit_rate']
            speedup = results['realistic_workload']['speedup']

            if hit_rate >= 0.6:
                print("✅ Cache hit rate target ACHIEVED (>60%)")
            else:
                print("⚠️  Cache hit rate below target (<60%)")

            if speedup >= 3.0:
                print("✅ Speedup target ACHIEVED (>3x)")
            else:
                print("⚠️  Speedup below target (<3x)")

            if results['warm_cache']['latency_ms'] < 50:
                print("✅ Warm cache latency target ACHIEVED (<50ms)")
            else:
                print("⚠️  Warm cache latency above target (>50ms)")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
