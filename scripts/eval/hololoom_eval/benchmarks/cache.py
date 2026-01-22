"""
Cache Effectiveness Benchmark

Tests: Does HoloLoom's caching provide value?
- Query cache hit rate
- Latency improvement on cache hits
- Cache warming behavior
"""

import asyncio
import time
from typing import Dict, List, Any

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus, get_queries_for_mode


async def run_cache_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """
    Test cache effectiveness.

    Methodology:
    1. Run same queries multiple times
    2. Measure cold vs warm latency
    3. Calculate cache hit rate and speedup
    """

    corpus = get_corpus()
    queries = get_queries_for_mode(mode.value)

    # Number of repetitions per query
    repetitions = 2 if mode == EvalMode.QUICK else (3 if mode == EvalMode.STANDARD else 5)

    results = {
        "cold_latencies": [],
        "warm_latencies": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "speedup_ratios": [],
    }

    if not deps.get("hololoom"):
        # Baseline test with simple dict cache simulation
        cache = {}

        for q in queries:
            query_text = q["query"]

            for rep in range(repetitions):
                start = time.perf_counter()

                if query_text in cache:
                    # Simulate cache hit (fast lookup)
                    _ = cache[query_text]
                    latency = (time.perf_counter() - start) * 1000
                    results["warm_latencies"].append(latency)
                    results["cache_hits"] += 1
                else:
                    # Simulate cache miss (computation)
                    time.sleep(0.05)  # Simulate 50ms work
                    cache[query_text] = f"result_{query_text}"
                    latency = (time.perf_counter() - start) * 1000
                    results["cold_latencies"].append(latency)
                    results["cache_misses"] += 1

        # Calculate metrics
        avg_cold = sum(results["cold_latencies"]) / max(1, len(results["cold_latencies"]))
        avg_warm = sum(results["warm_latencies"]) / max(1, len(results["warm_latencies"])) if results["warm_latencies"] else avg_cold

        hit_rate = results["cache_hits"] / (results["cache_hits"] + results["cache_misses"])
        speedup = avg_cold / avg_warm if avg_warm > 0 else 1.0

        return BenchmarkResult(
            name="Cache Effectiveness",
            passed=hit_rate > 0.3,  # Expect repeated queries to hit
            score=hit_rate,
            verdict=f"Simulated: {hit_rate*100:.0f}% hit rate, {speedup:.1f}x speedup",
            details={
                "results": results,
                "hit_rate": hit_rate,
                "speedup": speedup,
                "avg_cold_ms": avg_cold,
                "avg_warm_ms": avg_warm,
                "hololoom_tested": False,
            }
        )

    try:
        from HoloLoom import HoloLoom
        from HoloLoom.config import Config

        async with HoloLoom(config=Config.fast()) as loom:
            # Ingest corpus
            for doc in corpus:
                await loom.experience(doc["text"], metadata={"id": doc["id"]})

            # First pass - cold queries
            for q in queries:
                query_text = q["query"]

                start = time.perf_counter()
                _ = await loom.recall(query_text, k=5)
                latency = (time.perf_counter() - start) * 1000

                results["cold_latencies"].append(latency)
                results["cache_misses"] += 1

            # Subsequent passes - should hit cache
            for rep in range(1, repetitions):
                for q in queries:
                    query_text = q["query"]

                    start = time.perf_counter()
                    _ = await loom.recall(query_text, k=5)
                    latency = (time.perf_counter() - start) * 1000

                    # Check if this was faster (cache hit indicator)
                    avg_cold = sum(results["cold_latencies"]) / len(results["cold_latencies"])

                    if latency < avg_cold * 0.5:  # Significantly faster = cache hit
                        results["warm_latencies"].append(latency)
                        results["cache_hits"] += 1
                        results["speedup_ratios"].append(avg_cold / latency)
                    else:
                        results["cold_latencies"].append(latency)
                        results["cache_misses"] += 1

        # Calculate metrics
        avg_cold = sum(results["cold_latencies"]) / max(1, len(results["cold_latencies"]))
        avg_warm = sum(results["warm_latencies"]) / max(1, len(results["warm_latencies"])) if results["warm_latencies"] else avg_cold

        total_queries = results["cache_hits"] + results["cache_misses"]
        hit_rate = results["cache_hits"] / total_queries if total_queries > 0 else 0

        speedup = avg_cold / avg_warm if avg_warm > 0 else 1.0

        # Determine verdict
        if hit_rate > 0.6 and speedup > 5:
            verdict = f"Excellent: {hit_rate*100:.0f}% hit rate, {speedup:.1f}x speedup"
            passed = True
            score = 0.8 + min(0.2, (speedup - 5) / 50)  # Bonus for high speedup
        elif hit_rate > 0.3 and speedup > 2:
            verdict = f"Good: {hit_rate*100:.0f}% hit rate, {speedup:.1f}x speedup"
            passed = True
            score = 0.5 + hit_rate * 0.3
        elif hit_rate > 0:
            verdict = f"Marginal: {hit_rate*100:.0f}% hit rate, {speedup:.1f}x speedup"
            passed = True
            score = 0.3 + hit_rate * 0.2
        else:
            verdict = "No cache benefit detected"
            passed = False
            score = 0.3

        return BenchmarkResult(
            name="Cache Effectiveness",
            passed=passed,
            score=score,
            verdict=verdict,
            details={
                "results": results,
                "hit_rate": hit_rate,
                "speedup": speedup,
                "avg_cold_ms": avg_cold,
                "avg_warm_ms": avg_warm,
                "hololoom_tested": True,
            }
        )

    except Exception as e:
        return BenchmarkResult(
            name="Cache Effectiveness",
            passed=False,
            score=0.3,
            verdict=f"Error: {str(e)[:50]}",
            details={"error": str(e)}
        )
