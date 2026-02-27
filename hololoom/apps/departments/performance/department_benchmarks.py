"""
Department Performance Benchmarks

Benchmarks individual department performance across various metrics.

Features:
- Latency profiling (p50, p95, p99)
- Throughput measurement (QPS)
- Memory usage tracking
- Confidence quality metrics
- Cache effectiveness

Author: HoloLoom B2B Framework
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import asyncio
from statistics import mean, median, stdev


@dataclass
class BenchmarkResult:
    """Results from department benchmark"""
    department_id: str
    test_name: str

    # Latency metrics (ms)
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    latency_std: float

    # Throughput
    queries_per_second: float
    total_queries: int
    duration_seconds: float

    # Quality metrics
    avg_confidence: float
    success_rate: float

    # Resource metrics
    peak_memory_mb: float
    avg_cpu_percent: float

    # Cache metrics
    cache_hit_rate: float = 0.0

    # Raw latencies for analysis
    latencies: List[float] = field(default_factory=list)

    @property
    def passes_sla(self) -> bool:
        """Check if results meet SLA targets"""
        return (
            self.latency_p95 < 500  # <500ms p95
            and self.success_rate > 0.95  # >95% success
            and self.queries_per_second > 10  # >10 QPS
        )


class DepartmentBenchmark:
    """
    Benchmark individual department performance.

    Measures latency, throughput, quality, and resource usage.

    Example:
        >>> from hololoom.departments import get_department
        >>> benchmark = DepartmentBenchmark(department_id="rag")
        >>> result = await benchmark.run(
        ...     test_queries=["What is X?", "Explain Y"],
        ...     iterations=100
        ... )
        >>> print(f"p95 latency: {result.latency_p95:.1f}ms")
        >>> print(f"QPS: {result.queries_per_second:.1f}")
    """

    def __init__(
        self,
        department_id: str,
        warmup_iterations: int = 10
    ):
        """
        Initialize department benchmark.

        Args:
            department_id: Department to benchmark
            warmup_iterations: Number of warmup queries before measurement
        """
        self.department_id = department_id
        self.warmup_iterations = warmup_iterations

    async def run(
        self,
        test_queries: List[str],
        iterations: int = 100,
        concurrency: int = 1
    ) -> BenchmarkResult:
        """
        Run benchmark on department.

        Args:
            test_queries: List of test queries to execute
            iterations: Number of iterations per query
            concurrency: Concurrent requests (1 = sequential)

        Returns:
            BenchmarkResult with all metrics
        """
        from hololoom.departments import get_department

        dept = get_department(self.department_id)

        # Warmup
        print(f"Warming up {self.department_id} department...")
        for _ in range(self.warmup_iterations):
            await dept.process({
                "task_type": "question_answering",
                "parameters": {"query": test_queries[0]}
            })

        # Benchmark
        print(f"Benchmarking {self.department_id} ({iterations} iterations, concurrency={concurrency})...")

        latencies = []
        confidences = []
        successes = 0
        start_time = time.time()

        if concurrency == 1:
            # Sequential execution
            for i in range(iterations):
                query = test_queries[i % len(test_queries)]

                query_start = time.perf_counter()
                try:
                    response = await dept.process({
                        "task_type": "question_answering",
                        "parameters": {"query": query}
                    })
                    query_end = time.perf_counter()

                    latency_ms = (query_end - query_start) * 1000
                    latencies.append(latency_ms)
                    confidences.append(response.get("confidence", 0.0))

                    if response.get("status") == "success":
                        successes += 1

                except Exception as e:
                    print(f"Query failed: {e}")
                    latencies.append(float('inf'))

        else:
            # Concurrent execution
            async def execute_query(query: str):
                query_start = time.perf_counter()
                try:
                    response = await dept.process({
                        "task_type": "question_answering",
                        "parameters": {"query": query}
                    })
                    query_end = time.perf_counter()

                    latency_ms = (query_end - query_start) * 1000
                    return {
                        "latency": latency_ms,
                        "confidence": response.get("confidence", 0.0),
                        "success": response.get("status") == "success"
                    }
                except Exception:
                    return {"latency": float('inf'), "confidence": 0.0, "success": False}

            # Create batches
            queries = [test_queries[i % len(test_queries)] for i in range(iterations)]
            batches = [queries[i:i+concurrency] for i in range(0, len(queries), concurrency)]

            for batch in batches:
                results = await asyncio.gather(*[execute_query(q) for q in batch])
                for result in results:
                    latencies.append(result["latency"])
                    confidences.append(result["confidence"])
                    if result["success"]:
                        successes += 1

        end_time = time.time()
        duration = end_time - start_time

        # Calculate metrics
        valid_latencies = [l for l in latencies if l != float('inf')]

        if not valid_latencies:
            raise ValueError("All queries failed!")

        latencies_sorted = sorted(valid_latencies)

        return BenchmarkResult(
            department_id=self.department_id,
            test_name=f"benchmark_{iterations}x{concurrency}",
            latency_p50=latencies_sorted[int(len(latencies_sorted) * 0.50)],
            latency_p95=latencies_sorted[int(len(latencies_sorted) * 0.95)],
            latency_p99=latencies_sorted[int(len(latencies_sorted) * 0.99)],
            latency_mean=mean(valid_latencies),
            latency_std=stdev(valid_latencies) if len(valid_latencies) > 1 else 0.0,
            queries_per_second=iterations / duration,
            total_queries=iterations,
            duration_seconds=duration,
            avg_confidence=mean(confidences) if confidences else 0.0,
            success_rate=successes / iterations,
            peak_memory_mb=0.0,  # TODO: Implement memory tracking
            avg_cpu_percent=0.0,  # TODO: Implement CPU tracking
            cache_hit_rate=0.0,   # TODO: Implement cache tracking
            latencies=valid_latencies
        )

    async def run_all_departments(
        self,
        test_queries: List[str],
        iterations: int = 50
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark all departments.

        Args:
            test_queries: Test queries
            iterations: Iterations per department

        Returns:
            Dict of department_id → BenchmarkResult
        """
        from hololoom.departments import get_all_departments

        departments = get_all_departments()
        results = {}

        for dept_id in departments.keys():
            print(f"\n=== Benchmarking {dept_id} ===")
            benchmark = DepartmentBenchmark(dept_id)
            results[dept_id] = await benchmark.run(test_queries, iterations)

        return results

    def compare_results(
        self,
        results: Dict[str, BenchmarkResult]
    ) -> str:
        """
        Compare benchmark results across departments.

        Args:
            results: Dict of department_id → BenchmarkResult

        Returns:
            Formatted comparison report
        """
        report = []
        report.append("=== Department Performance Comparison ===\n")
        report.append(f"{'Department':<20} {'p50 (ms)':>10} {'p95 (ms)':>10} {'QPS':>10} {'Success %':>12}\n")
        report.append("-" * 64 + "\n")

        for dept_id, result in sorted(results.items()):
            report.append(
                f"{dept_id:<20} "
                f"{result.latency_p50:>10.1f} "
                f"{result.latency_p95:>10.1f} "
                f"{result.queries_per_second:>10.1f} "
                f"{result.success_rate * 100:>11.1f}%\n"
            )

        # SLA compliance
        report.append("\n=== SLA Compliance ===\n")
        for dept_id, result in sorted(results.items()):
            status = "[PASS]" if result.passes_sla else "[FAIL]"
            report.append(f"{status} {dept_id}\n")

        return "".join(report)
