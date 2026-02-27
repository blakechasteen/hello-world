#!/usr/bin/env python3
"""
Jenny MRF/LLM Integration Performance Benchmarks
=================================================

Measures latency and accuracy of the Jenny panel compilation system.

Date: December 2025
Status: Phase A.8 Performance Benchmarks

Benchmarks:
    1. benchmark_mrf_compiler_latency() - JennyMRFCompiler.compile() (<5ms target)
    2. benchmark_thompson_sampling() - PanelTypeLearner.select() (<1ms target)
    3. benchmark_learning_updates() - PanelTypeLearner.update() (<1ms target)
    4. benchmark_fallback_chain() - LLM fail → MRF → Heuristic (<5ms target)
    5. benchmark_accuracy_comparison() - Learned vs heuristic selection (+35% target)
    6. benchmark_llm_compiler_latency() - LLMJennyCompiler with Ollama (<500ms target)

Run:
    PYTHONPATH=. python benchmarks/benchmark_jenny_llm.py
"""

import asyncio
import time
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


# Import Jenny components
from hololoom.visualization.jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    LifecycleStage,
)
from hololoom.visualization.jenny_mrf import (
    JennyMRFCompiler,
    PanelTypeLearner,
    PanelTypePrior,
    ACTION_LEARNING_MAP,
    create_mrf_compiler,
)
from hololoom.visualization.jenny_llm_compiler import (
    LLMJennyCompiler,
    SemanticProfile,
    analyze_semantic_axes,
    create_llm_compiler,
    create_compiler_with_fallback,
)

# Try to import Spacetime for realistic benchmarks
try:
    from hololoom.fabric.spacetime import Spacetime, WeavingTrace
    SPACETIME_AVAILABLE = True
except ImportError:
    SPACETIME_AVAILABLE = False
    Spacetime = None


# ============================================================================
# Configuration
# ============================================================================

BENCHMARK_CONFIG = {
    "iterations": 100,           # Number of iterations per benchmark
    "warmup_iterations": 10,     # Warmup iterations (excluded from stats)
    "query_types": ["factual", "procedural", "analytical", "creative"],
    "panel_types": [PanelTypeJenny.TEXT, PanelTypeJenny.CODE, PanelTypeJenny.GRAPH,
                    PanelTypeJenny.CONFIDENCE, PanelTypeJenny.TIMELINE],

    # Target latencies (milliseconds)
    "targets": {
        "mrf_compiler": 5.0,         # <5ms for MRF compilation
        "thompson_sampling": 1.0,    # <1ms per selection
        "learning_update": 1.0,      # <1ms per update
        "fallback_chain": 5.0,       # <5ms for fallback
        "llm_compiler": 500.0,       # <500ms for LLM
        "semantic_axes": 10.0,       # <10ms for semantic analysis
    }
}

# Sample queries for benchmarking
SAMPLE_QUERIES = [
    "What is Thompson Sampling?",
    "How do I implement a priority queue in Python?",
    "Compare React and Vue.js performance",
    "URGENT: Fix this production bug now!",
    "Explain the theoretical foundations of neural networks",
    "Show me the code for a binary search tree",
    "What are the tradeoffs of different caching strategies?",
    "Define machine learning",
]


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    target_ms: float
    passed: bool
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_ms:.3f}ms (target: {self.target_ms:.1f}ms) {status}\n"
            f"  Median: {self.median_ms:.3f}ms\n"
            f"  Std: {self.std_ms:.3f}ms\n"
            f"  Range: [{self.min_ms:.3f}ms, {self.max_ms:.3f}ms]\n"
            f"  Iterations: {self.iterations}"
        )


def compute_stats(times_ms: List[float], name: str, target_ms: float,
                  metadata: Optional[Dict] = None) -> BenchmarkResult:
    """Compute statistics from timing measurements."""
    return BenchmarkResult(
        name=name,
        iterations=len(times_ms),
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        target_ms=target_ms,
        passed=statistics.mean(times_ms) < target_ms,
        metadata=metadata or {},
    )


# ============================================================================
# Mock Spacetime for Benchmarking
# ============================================================================

class MockSpacetime:
    """Mock Spacetime for benchmarking when real Spacetime unavailable."""

    def __init__(self, query_text: str = "What is Thompson Sampling?"):
        self.query_text = query_text
        self.confidence = 0.85
        self.response = "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem..."
        self.text = self.response
        self.trace = MockTrace()
        self.metadata = {}

    @property
    def query(self):
        return self.query_text


class MockTrace:
    """Mock WeavingTrace for benchmarking."""

    def __init__(self):
        self.threads = []
        self.total_duration_ms = 150.0
        self.threads_activated = ["thread_1", "thread_2", "thread_3"]
        self.stage_durations = {"retrieval": 50.0, "decision": 30.0, "synthesis": 70.0}
        self.stages = ["retrieval", "decision", "synthesis"]


def create_mock_spacetime(query: str = "What is Thompson Sampling?") -> Any:
    """Create a mock Spacetime for benchmarking."""
    return MockSpacetime(query)


# ============================================================================
# Benchmark Functions
# ============================================================================

async def benchmark_mrf_compiler_latency() -> BenchmarkResult:
    """
    Benchmark 1: JennyMRFCompiler.compile() latency

    Target: <5ms per compilation (fallback path)
    """
    print("\n" + "="*60)
    print("Benchmark 1: MRF Compiler Latency")
    print("="*60)

    compiler = create_mrf_compiler(enable_learning=True)
    times_ms = []

    # Warmup
    for _ in range(BENCHMARK_CONFIG["warmup_iterations"]):
        spacetime = create_mock_spacetime()
        await compiler.compile(spacetime)

    # Benchmark
    for i, query in enumerate(SAMPLE_QUERIES * (BENCHMARK_CONFIG["iterations"] // len(SAMPLE_QUERIES) + 1)):
        if i >= BENCHMARK_CONFIG["iterations"]:
            break

        spacetime = create_mock_spacetime(query)

        start = time.perf_counter()
        specs = await compiler.compile(spacetime)
        elapsed_ms = (time.perf_counter() - start) * 1000

        times_ms.append(elapsed_ms)

    result = compute_stats(
        times_ms,
        "MRF Compiler Latency",
        BENCHMARK_CONFIG["targets"]["mrf_compiler"],
        metadata={"compiler_version": compiler.VERSION}
    )

    print(result)
    return result


def benchmark_thompson_sampling() -> BenchmarkResult:
    """
    Benchmark 2: PanelTypeLearner.select() latency

    Target: <1ms per selection
    """
    print("\n" + "="*60)
    print("Benchmark 2: Thompson Sampling Selection")
    print("="*60)

    learner = PanelTypeLearner()
    candidates = BENCHMARK_CONFIG["panel_types"]
    query_types = BENCHMARK_CONFIG["query_types"]
    times_ms = []

    # Pre-populate with some learning
    for qt in query_types:
        for pt in candidates:
            learner.update(qt, pt, success=True, confidence=0.8)

    # Warmup
    for _ in range(BENCHMARK_CONFIG["warmup_iterations"]):
        qt = query_types[0]
        learner.select(qt, candidates)

    # Benchmark
    for i in range(BENCHMARK_CONFIG["iterations"]):
        qt = query_types[i % len(query_types)]

        start = time.perf_counter()
        selected = learner.select(qt, candidates)
        elapsed_ms = (time.perf_counter() - start) * 1000

        times_ms.append(elapsed_ms)

    result = compute_stats(
        times_ms,
        "Thompson Sampling Selection",
        BENCHMARK_CONFIG["targets"]["thompson_sampling"],
        metadata={"num_priors": len(learner.priors)}
    )

    print(result)
    return result


def benchmark_learning_updates() -> BenchmarkResult:
    """
    Benchmark 3: PanelTypeLearner.update() latency

    Target: <1ms per update
    """
    print("\n" + "="*60)
    print("Benchmark 3: Learning Updates")
    print("="*60)

    learner = PanelTypeLearner()
    candidates = BENCHMARK_CONFIG["panel_types"]
    query_types = BENCHMARK_CONFIG["query_types"]
    times_ms = []

    # Warmup
    for _ in range(BENCHMARK_CONFIG["warmup_iterations"]):
        learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

    # Benchmark
    for i in range(BENCHMARK_CONFIG["iterations"]):
        qt = query_types[i % len(query_types)]
        pt = candidates[i % len(candidates)]
        success = i % 3 != 0  # ~67% success rate
        confidence = 0.5 + (i % 5) * 0.1

        start = time.perf_counter()
        learner.update(qt, pt, success=success, confidence=confidence)
        elapsed_ms = (time.perf_counter() - start) * 1000

        times_ms.append(elapsed_ms)

    result = compute_stats(
        times_ms,
        "Learning Update",
        BENCHMARK_CONFIG["targets"]["learning_update"],
        metadata={"final_priors": len(learner.priors)}
    )

    print(result)
    return result


async def benchmark_fallback_chain() -> BenchmarkResult:
    """
    Benchmark 4: Fallback chain latency (LLM fail → MRF → Heuristic)

    Target: <5ms for complete fallback

    Simulates LLM unavailable scenario to measure fallback performance.
    """
    print("\n" + "="*60)
    print("Benchmark 4: Fallback Chain Latency")
    print("="*60)

    # Create compiler with LLM that will fail (no ollama running)
    compiler = create_compiler_with_fallback(
        prefer_llm=True,
        llm_provider="ollama",
        llm_model="nonexistent-model-xyz",
        enable_learning=True,
    )

    times_ms = []

    # Warmup
    for _ in range(BENCHMARK_CONFIG["warmup_iterations"]):
        spacetime = create_mock_spacetime()
        await compiler.compile(spacetime)

    # Benchmark
    for i, query in enumerate(SAMPLE_QUERIES * (BENCHMARK_CONFIG["iterations"] // len(SAMPLE_QUERIES) + 1)):
        if i >= BENCHMARK_CONFIG["iterations"]:
            break

        spacetime = create_mock_spacetime(query)

        start = time.perf_counter()
        specs = await compiler.compile(spacetime)
        elapsed_ms = (time.perf_counter() - start) * 1000

        times_ms.append(elapsed_ms)

    result = compute_stats(
        times_ms,
        "Fallback Chain",
        BENCHMARK_CONFIG["targets"]["fallback_chain"],
        metadata={
            "compiler_version": compiler.VERSION,
            "llm_available": getattr(compiler, '_llm_available', False),
        }
    )

    print(result)
    return result


def benchmark_accuracy_comparison() -> BenchmarkResult:
    """
    Benchmark 5: Learned vs heuristic selection accuracy

    Target: +35% improvement with learning

    Simulates user preferences and measures how well learning adapts.
    """
    print("\n" + "="*60)
    print("Benchmark 5: Accuracy Comparison (Learned vs Heuristic)")
    print("="*60)

    # Define "ground truth" preferences (simulated user preferences)
    ground_truth = {
        "factual": PanelTypeJenny.TEXT,
        "procedural": PanelTypeJenny.CODE,
        "analytical": PanelTypeJenny.GRAPH,
        "creative": PanelTypeJenny.TEXT,
    }

    candidates = BENCHMARK_CONFIG["panel_types"]
    times_ms = []

    # Heuristic accuracy (random baseline)
    heuristic_correct = 0
    total = BENCHMARK_CONFIG["iterations"]

    for i in range(total):
        qt = list(ground_truth.keys())[i % len(ground_truth)]
        expected = ground_truth[qt]

        # Heuristic: always pick first candidate (TEXT)
        heuristic_choice = candidates[0]
        if heuristic_choice == expected:
            heuristic_correct += 1

    heuristic_accuracy = heuristic_correct / total

    # Learned accuracy
    learner = PanelTypeLearner()

    # Training phase: simulate user feedback
    training_rounds = 50
    for _ in range(training_rounds):
        for qt, preferred_pt in ground_truth.items():
            # User pins preferred panel type
            learner.update(qt, preferred_pt, success=True, confidence=0.9)

            # User dismisses other panel types
            for pt in candidates:
                if pt != preferred_pt:
                    learner.update(qt, pt, success=False, confidence=0.3)

    # Evaluation phase
    learned_correct = 0

    for i in range(total):
        qt = list(ground_truth.keys())[i % len(ground_truth)]
        expected = ground_truth[qt]

        start = time.perf_counter()
        selected = learner.select(qt, candidates)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

        if selected == expected:
            learned_correct += 1

    learned_accuracy = learned_correct / total
    improvement = ((learned_accuracy - heuristic_accuracy) / max(heuristic_accuracy, 0.01)) * 100

    # Check if we hit +35% improvement target
    passed = improvement >= 35.0

    result = BenchmarkResult(
        name="Accuracy Comparison",
        iterations=total,
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        target_ms=35.0,  # Target improvement percentage
        passed=passed,
        metadata={
            "heuristic_accuracy": f"{heuristic_accuracy:.1%}",
            "learned_accuracy": f"{learned_accuracy:.1%}",
            "improvement": f"{improvement:.1f}%",
            "target_improvement": "+35%",
            "training_rounds": training_rounds,
        }
    )

    print(f"\nAccuracy Comparison:")
    print(f"  Heuristic Accuracy: {heuristic_accuracy:.1%}")
    print(f"  Learned Accuracy: {learned_accuracy:.1%}")
    print(f"  Improvement: {improvement:+.1f}% (target: +35%)")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"  Selection Time (mean): {statistics.mean(times_ms):.3f}ms")

    return result


def benchmark_semantic_axes() -> BenchmarkResult:
    """
    Benchmark 6: Semantic axes analysis latency

    Target: <10ms per analysis
    """
    print("\n" + "="*60)
    print("Benchmark 6: Semantic Axes Analysis")
    print("="*60)

    times_ms = []

    # Warmup
    for _ in range(BENCHMARK_CONFIG["warmup_iterations"]):
        analyze_semantic_axes("What is Thompson Sampling?")

    # Benchmark
    for i, query in enumerate(SAMPLE_QUERIES * (BENCHMARK_CONFIG["iterations"] // len(SAMPLE_QUERIES) + 1)):
        if i >= BENCHMARK_CONFIG["iterations"]:
            break

        start = time.perf_counter()
        profile = analyze_semantic_axes(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        times_ms.append(elapsed_ms)

    result = compute_stats(
        times_ms,
        "Semantic Axes Analysis",
        BENCHMARK_CONFIG["targets"]["semantic_axes"],
        metadata={"sample_profile": analyze_semantic_axes(SAMPLE_QUERIES[0]).to_dict()}
    )

    print(result)
    return result


async def benchmark_llm_compiler_latency() -> Optional[BenchmarkResult]:
    """
    Benchmark 7: LLMJennyCompiler with Ollama (optional)

    Target: <500ms per compilation

    Only runs if Ollama is available.
    """
    print("\n" + "="*60)
    print("Benchmark 7: LLM Compiler Latency (Optional)")
    print("="*60)

    # Try to create LLM compiler
    compiler = create_llm_compiler(
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        enable_learning=True,
    )

    if not getattr(compiler, '_llm_available', False):
        print("⚠️  Skipped: Ollama not available")
        print("   To run this benchmark, start Ollama with: ollama serve")
        return None

    times_ms = []

    # Reduced iterations for LLM (slower)
    llm_iterations = min(20, BENCHMARK_CONFIG["iterations"])

    # Warmup (fewer for LLM)
    for _ in range(min(3, BENCHMARK_CONFIG["warmup_iterations"])):
        spacetime = create_mock_spacetime()
        await compiler.compile(spacetime)

    # Benchmark
    for i, query in enumerate(SAMPLE_QUERIES * 3):
        if i >= llm_iterations:
            break

        spacetime = create_mock_spacetime(query)

        start = time.perf_counter()
        specs = await compiler.compile(spacetime)
        elapsed_ms = (time.perf_counter() - start) * 1000

        times_ms.append(elapsed_ms)
        print(f"  Iteration {i+1}/{llm_iterations}: {elapsed_ms:.1f}ms")

    result = compute_stats(
        times_ms,
        "LLM Compiler Latency",
        BENCHMARK_CONFIG["targets"]["llm_compiler"],
        metadata={
            "compiler_version": compiler.VERSION,
            "llm_provider": compiler.llm_provider,
            "llm_model": compiler.llm_model,
        }
    )

    print(result)
    return result


# ============================================================================
# Main Runner
# ============================================================================

async def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    print("\n" + "="*60)
    print("  JENNY MRF/LLM INTEGRATION BENCHMARKS")
    print("  " + "="*56)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Iterations: {BENCHMARK_CONFIG['iterations']}")
    print("="*60)

    results = []

    # Core benchmarks (always run)
    results.append(await benchmark_mrf_compiler_latency())
    results.append(benchmark_thompson_sampling())
    results.append(benchmark_learning_updates())
    results.append(await benchmark_fallback_chain())
    results.append(benchmark_accuracy_comparison())
    results.append(benchmark_semantic_axes())

    # Optional LLM benchmark
    llm_result = await benchmark_llm_compiler_latency()
    if llm_result:
        results.append(llm_result)

    # Summary
    print("\n" + "="*60)
    print("  BENCHMARK SUMMARY")
    print("="*60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}: {result.mean_ms:.3f}ms (target: {result.target_ms:.1f})")

    print(f"\n  Total: {passed}/{total} benchmarks passed")

    # Detailed metadata for accuracy comparison
    accuracy_result = next((r for r in results if r.name == "Accuracy Comparison"), None)
    if accuracy_result:
        print(f"\n  Accuracy Details:")
        for key, value in accuracy_result.metadata.items():
            print(f"    {key}: {value}")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
