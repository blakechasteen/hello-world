"""
Learning Effectiveness Benchmark

Tests: Does HoloLoom improve over time?
- Thompson Sampling adaptation
- Hot pattern feedback
- Policy weight updates
"""

import asyncio
import time
from typing import Dict, List, Any

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus, EVALUATION_QUERIES


async def run_learning_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """
    Test if HoloLoom's learning mechanisms actually improve performance over time.

    Methodology:
    1. Run same queries multiple times (simulating repeated usage)
    2. Track confidence/quality metrics across iterations
    3. Check if metrics trend upward (learning is working)
    """

    if not deps.get("hololoom") or not deps.get("numpy"):
        return BenchmarkResult(
            name="Learning Effectiveness",
            passed=False,
            score=0.0,
            verdict="Missing dependencies (hololoom, numpy)",
            details={"skipped": True, "reason": "dependencies"}
        )

    try:
        from HoloLoom import HoloLoom
        from HoloLoom.config import Config
    except ImportError as e:
        return BenchmarkResult(
            name="Learning Effectiveness",
            passed=False,
            score=0.0,
            verdict=f"Import error: {e}",
            details={"skipped": True, "reason": str(e)}
        )

    corpus = get_corpus()

    # Subset of queries for learning test
    test_queries = EVALUATION_QUERIES[:5]  # Use first 5 queries

    # Number of iterations
    iterations = 3 if mode == EvalMode.QUICK else (5 if mode == EvalMode.STANDARD else 10)

    try:
        async with HoloLoom(config=Config.fast()) as loom:
            # Ingest corpus
            for doc in corpus:
                await loom.experience(doc["text"], metadata={"id": doc["id"]})

            # Track metrics across iterations
            iteration_metrics = []

            for iteration in range(iterations):
                iter_confidences = []
                iter_latencies = []

                for q in test_queries:
                    start = time.perf_counter()
                    memories = await loom.recall(q["query"], k=5)
                    latency = (time.perf_counter() - start) * 1000

                    # Get confidence from memories
                    if memories:
                        avg_conf = sum(
                            getattr(m, "confidence", 0.5)
                            for m in memories
                        ) / len(memories)
                    else:
                        avg_conf = 0.0

                    iter_confidences.append(avg_conf)
                    iter_latencies.append(latency)

                iteration_metrics.append({
                    "iteration": iteration,
                    "avg_confidence": sum(iter_confidences) / len(iter_confidences),
                    "avg_latency_ms": sum(iter_latencies) / len(iter_latencies),
                })

            # Analyze learning trend
            confidences = [m["avg_confidence"] for m in iteration_metrics]
            latencies = [m["avg_latency_ms"] for m in iteration_metrics]

            # Check for improvement (later iterations should be better)
            first_half_conf = sum(confidences[:len(confidences)//2]) / max(1, len(confidences)//2)
            second_half_conf = sum(confidences[len(confidences)//2:]) / max(1, len(confidences) - len(confidences)//2)

            first_half_lat = sum(latencies[:len(latencies)//2]) / max(1, len(latencies)//2)
            second_half_lat = sum(latencies[len(latencies)//2:]) / max(1, len(latencies) - len(latencies)//2)

            conf_improved = second_half_conf >= first_half_conf
            latency_improved = second_half_lat <= first_half_lat

            # Calculate improvement percentages
            conf_delta = (second_half_conf - first_half_conf) / first_half_conf if first_half_conf > 0 else 0
            latency_delta = (first_half_lat - second_half_lat) / first_half_lat if first_half_lat > 0 else 0

            # Determine score
            if conf_improved and latency_improved:
                score = 0.7 + min(0.3, abs(conf_delta) + abs(latency_delta))
                verdict = f"Learning active: conf +{conf_delta*100:.1f}%, latency -{latency_delta*100:.1f}%"
                passed = True
            elif conf_improved or latency_improved:
                score = 0.5 + min(0.2, max(abs(conf_delta), abs(latency_delta)))
                verdict = f"Partial learning: conf {conf_delta*100:+.1f}%, latency {-latency_delta*100:+.1f}%"
                passed = True
            else:
                score = 0.3
                verdict = "No learning detected (may need more iterations or queries)"
                passed = False

            return BenchmarkResult(
                name="Learning Effectiveness",
                passed=passed,
                score=score,
                verdict=verdict,
                details={
                    "iterations": iterations,
                    "iteration_metrics": iteration_metrics,
                    "confidence_improved": conf_improved,
                    "latency_improved": latency_improved,
                    "confidence_delta_pct": conf_delta * 100,
                    "latency_delta_pct": latency_delta * 100,
                }
            )

    except Exception as e:
        return BenchmarkResult(
            name="Learning Effectiveness",
            passed=False,
            score=0.0,
            verdict=f"Error: {str(e)[:50]}",
            details={"error": str(e)}
        )
