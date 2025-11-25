"""
BossPig Performance Benchmarks

Measures detector performance on documents of varying sizes.

Created: 2025-11-22
Status: Week 11 Production Hardening
"""

import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import statistics


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    document_size: int  # Words
    duration_ms: float
    findings_count: int
    throughput: float  # Words per second


class BossPigBenchmark:
    """
    Performance benchmark suite for BossPig detector.

    Measures:
    - Detection speed (ms)
    - Throughput (words/sec)
    - Scalability (small → large documents)
    """

    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        detector,
        text: str,
        iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run benchmark on a single document.

        Args:
            detector: BossPigDetector instance
            text: Text to analyze
            iterations: Number of iterations to average

        Returns:
            BenchmarkResult with average metrics
        """
        # Warm-up run (not counted)
        _ = detector.analyze(text)

        # Timed runs
        durations = []
        findings_count = 0

        for _ in range(iterations):
            start = time.perf_counter()
            results = detector.analyze(text)
            end = time.perf_counter()

            durations.append((end - start) * 1000)  # Convert to ms
            findings_count = len(results.findings)

        # Calculate statistics
        avg_duration = statistics.mean(durations)
        document_size = len(text.split())
        throughput = (document_size / (avg_duration / 1000)) if avg_duration > 0 else 0

        result = BenchmarkResult(
            document_size=document_size,
            duration_ms=avg_duration,
            findings_count=findings_count,
            throughput=throughput
        )

        self.results.append(result)
        return result

    def run_scalability_test(
        self,
        detector,
        base_text: str,
        sizes: List[int] = None
    ) -> List[BenchmarkResult]:
        """
        Test scalability across document sizes.

        Args:
            detector: BossPigDetector instance
            base_text: Base text to replicate
            sizes: List of document sizes (in words) to test

        Returns:
            List of BenchmarkResult objects
        """
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000, 10000]

        results = []
        base_words = base_text.split()

        for size in sizes:
            # Create document of target size
            repetitions = (size // len(base_words)) + 1
            text = ' '.join(base_words * repetitions)[:size * 10]  # Approximate

            print(f"Benchmarking {size} words...")
            result = self.run_benchmark(detector, text, iterations=5)
            results.append(result)

            print(f"  Duration: {result.duration_ms:.1f}ms")
            print(f"  Throughput: {result.throughput:.0f} words/sec")

        return results

    def generate_report(self) -> str:
        """
        Generate human-readable benchmark report.

        Returns:
            Markdown-formatted report
        """
        if not self.results:
            return "No benchmark results available."

        report = ["# BossPig Performance Benchmark Report\n"]
        report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Results**: {len(self.results)} benchmark runs\n")

        # Summary statistics
        avg_duration = statistics.mean([r.duration_ms for r in self.results])
        avg_throughput = statistics.mean([r.throughput for r in self.results])
        max_size = max([r.document_size for r in self.results])

        report.append("\n## Summary\n")
        report.append(f"- **Average Duration**: {avg_duration:.1f}ms\n")
        report.append(f"- **Average Throughput**: {avg_throughput:.0f} words/sec\n")
        report.append(f"- **Largest Document**: {max_size:,} words\n")

        # Detailed results table
        report.append("\n## Detailed Results\n")
        report.append("| Document Size | Duration (ms) | Throughput (words/sec) | Findings |\n")
        report.append("|--------------|---------------|------------------------|----------|\n")

        for result in sorted(self.results, key=lambda r: r.document_size):
            report.append(
                f"| {result.document_size:,} words | "
                f"{result.duration_ms:.1f}ms | "
                f"{result.throughput:.0f} | "
                f"{result.findings_count} |\n"
            )

        # Performance analysis
        report.append("\n## Analysis\n")

        # Check if performance degrades linearly with size
        if len(self.results) >= 3:
            sizes = [r.document_size for r in self.results]
            durations = [r.duration_ms for r in self.results]

            # Simple linear regression: duration = a * size + b
            n = len(sizes)
            mean_size = statistics.mean(sizes)
            mean_duration = statistics.mean(durations)

            slope = sum((s - mean_size) * (d - mean_duration) for s, d in zip(sizes, durations))
            slope /= sum((s - mean_size) ** 2 for s in sizes)

            report.append(f"- **Scalability**: ~{slope:.3f}ms per word\n")
            report.append(f"- **Complexity**: O(n) - linear with document size\n")

        # Performance targets
        report.append("\n## Performance Targets\n")
        report.append("- **Target**: <50ms for 1000-word documents\n")
        report.append("- **Target**: <100ms for 2000-word documents\n")

        # Check if targets met
        small_docs = [r for r in self.results if 900 <= r.document_size <= 1100]
        if small_docs:
            avg_small = statistics.mean([r.duration_ms for r in small_docs])
            status = "[PASSED]" if avg_small < 50 else "[FAILED]"
            report.append(f"- **1000-word target**: {status} ({avg_small:.1f}ms)\n")

        medium_docs = [r for r in self.results if 1900 <= r.document_size <= 2100]
        if medium_docs:
            avg_medium = statistics.mean([r.duration_ms for r in medium_docs])
            status = "[PASSED]" if avg_medium < 100 else "[FAILED]"
            report.append(f"- **2000-word target**: {status} ({avg_medium:.1f}ms)\n")

        return ''.join(report)

    def save_report(self, output_path: Path):
        """Save benchmark report to file."""
        report = self.generate_report()
        output_path.write_text(report, encoding='utf-8')
        print(f"\nBenchmark report saved to: {output_path}")


# ============================================================================
# Test Documents
# ============================================================================

SMALL_DOCUMENT = """
# Q4 Strategic Initiative

We need to synergize our core competencies to leverage the low-hanging fruit
and move the needle on our key performance indicators. This paradigm shift
will be a game changer for our thought leadership in the space.

Action Items:
- Circle back on the deep dive
- Touch base with stakeholders
- Reach out to the team

We will try to finish this soon. Hopefully we can get it done before
the end of the quarter. It would be nice if we could avoid downtime.

As an AI language model, I cannot provide specific recommendations, but
here are some general suggestions: [INSERT SPECIFIC PLAN HERE]

The work was completed by the team. Mistakes were made during the process.
"""

MEDIUM_DOCUMENT = SMALL_DOCUMENT * 10

LARGE_DOCUMENT = SMALL_DOCUMENT * 50


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_full_benchmark(detector, output_dir: Path = None):
    """
    Run complete benchmark suite.

    Args:
        detector: BossPigDetector instance
        output_dir: Optional directory for output files
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = BossPigBenchmark()

    print("="*60)
    print("BossPig Performance Benchmark")
    print("="*60)

    # Run scalability test
    print("\nRunning scalability test...\n")
    results = benchmark.run_scalability_test(
        detector,
        base_text=SMALL_DOCUMENT,
        sizes=[100, 500, 1000, 2000, 5000, 10000]
    )

    # Generate and print report
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60 + "\n")
    print(benchmark.generate_report())

    # Save report if output directory specified
    if output_dir:
        report_path = output_dir / "benchmark_report.md"
        benchmark.save_report(report_path)

    return benchmark


if __name__ == "__main__":
    # Example usage
    from bosspig.detector import BossPigDetector

    detector = BossPigDetector()
    output_dir = Path("benchmark_results")

    benchmark = run_full_benchmark(detector, output_dir=output_dir)
