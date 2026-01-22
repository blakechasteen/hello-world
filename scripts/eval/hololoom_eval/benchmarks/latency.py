"""
Latency Benchmark

Tests: Does HoloLoom meet performance claims?
- FAST mode: <150ms
- FUSED mode: <300ms
- Cache hits: <10ms
"""

import asyncio
import time
from typing import Dict, List, Any
import statistics

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus, get_queries_for_mode


async def run_latency_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """
    Test latency characteristics.

    Targets:
    - FAST mode p95: <150ms
    - FUSED mode p95: <300ms
    - Warm cache p95: <10ms
    """

    corpus = get_corpus()
    queries = get_queries_for_mode(mode.value)

    # Latency targets (ms)
    targets = {
        "fast_p95": 150,
        "fused_p95": 300,
        "cache_p95": 10,
    }

    results = {
        "fast": {"latencies": [], "p50": 0, "p95": 0, "target_met": False},
        "fused": {"latencies": [], "p50": 0, "p95": 0, "target_met": False},
        "cache": {"latencies": [], "p50": 0, "p95": 0, "target_met": False},
    }

    if not deps.get("hololoom"):
        # Test with baseline retrievers
        from .retrieval import HybridRetriever
        hybrid = HybridRetriever(corpus)

        latencies = []
        for q in queries:
            start = time.perf_counter()
            _ = hybrid.retrieve(q["query"], k=5)
            latencies.append((time.perf_counter() - start) * 1000)

        if latencies:
            results["fast"]["latencies"] = latencies
            results["fast"]["p50"] = statistics.median(latencies)
            results["fast"]["p95"] = sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) > 1 else latencies[0]
            results["fast"]["target_met"] = results["fast"]["p95"] < targets["fast_p95"]

        verdict = f"Baseline only: p95={results['fast']['p95']:.1f}ms"
        passed = results["fast"]["target_met"]
        score = 1.0 - min(1.0, results["fast"]["p95"] / targets["fast_p95"]) if results["fast"]["p95"] > 0 else 0.5

        return BenchmarkResult(
            name="Latency",
            passed=passed,
            score=max(0.3, score),
            verdict=verdict,
            details={"results": results, "targets": targets, "hololoom_tested": False}
        )

    try:
        from HoloLoom import HoloLoom
        from HoloLoom.config import Config

        # Test FAST mode
        async with HoloLoom(config=Config.fast()) as loom:
            for doc in corpus:
                await loom.experience(doc["text"], metadata={"id": doc["id"]})

            # Cold queries
            for q in queries:
                start = time.perf_counter()
                _ = await loom.recall(q["query"], k=5)
                results["fast"]["latencies"].append((time.perf_counter() - start) * 1000)

            # Warm cache queries (repeat same queries)
            for q in queries[:3]:  # Just a few for cache test
                start = time.perf_counter()
                _ = await loom.recall(q["query"], k=5)
                results["cache"]["latencies"].append((time.perf_counter() - start) * 1000)

        # Test FUSED mode if not quick mode
        if mode != EvalMode.QUICK:
            async with HoloLoom(config=Config.fused()) as loom:
                for doc in corpus:
                    await loom.experience(doc["text"], metadata={"id": doc["id"]})

                for q in queries:
                    start = time.perf_counter()
                    _ = await loom.recall(q["query"], k=5)
                    results["fused"]["latencies"].append((time.perf_counter() - start) * 1000)

        # Calculate statistics
        for mode_name in ["fast", "fused", "cache"]:
            latencies = results[mode_name]["latencies"]
            if latencies:
                results[mode_name]["p50"] = statistics.median(latencies)
                idx = int(0.95 * len(latencies))
                results[mode_name]["p95"] = sorted(latencies)[idx] if len(latencies) > 1 else latencies[0]

        # Check targets
        results["fast"]["target_met"] = results["fast"]["p95"] < targets["fast_p95"] if results["fast"]["latencies"] else False
        results["fused"]["target_met"] = results["fused"]["p95"] < targets["fused_p95"] if results["fused"]["latencies"] else True
        results["cache"]["target_met"] = results["cache"]["p95"] < targets["cache_p95"] if results["cache"]["latencies"] else False

        # Calculate score
        targets_met = sum([
            results["fast"]["target_met"],
            results["fused"]["target_met"],
            results["cache"]["target_met"],
        ])

        score = targets_met / 3

        # Build verdict
        verdicts = []
        if results["fast"]["latencies"]:
            status = "✓" if results["fast"]["target_met"] else "✗"
            verdicts.append(f"FAST p95={results['fast']['p95']:.0f}ms {status}")
        if results["fused"]["latencies"]:
            status = "✓" if results["fused"]["target_met"] else "✗"
            verdicts.append(f"FUSED p95={results['fused']['p95']:.0f}ms {status}")
        if results["cache"]["latencies"]:
            status = "✓" if results["cache"]["target_met"] else "✗"
            verdicts.append(f"Cache p95={results['cache']['p95']:.1f}ms {status}")

        verdict = " | ".join(verdicts) if verdicts else "No latency data"
        passed = targets_met >= 2  # At least 2 out of 3 targets met

        return BenchmarkResult(
            name="Latency",
            passed=passed,
            score=max(0.3, score),
            verdict=verdict,
            details={"results": results, "targets": targets, "hololoom_tested": True}
        )

    except Exception as e:
        return BenchmarkResult(
            name="Latency",
            passed=False,
            score=0.3,
            verdict=f"Error: {str(e)[:50]}",
            details={"error": str(e)}
        )
