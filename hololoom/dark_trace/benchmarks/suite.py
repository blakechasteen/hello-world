"""
Benchmark Suite for Dark Trace Operations

Comprehensive benchmarking of all interpretability operations
with configurable test scenarios.

Author: HoloLoom Team
Created: December 2025
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .profiler import CombinedProfiler


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    ACTIVATION_EXTRACTION = "activation_extraction"
    FINGERPRINTING = "fingerprinting"
    STEERING = "steering"
    CIRCUIT_TRACING = "circuit_tracing"
    SAE_FORWARD = "sae_forward"
    SAE_TRAINING = "sae_training"
    CAUSAL_VALIDATION = "causal_validation"
    FULL_ANALYSIS = "full_analysis"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Iterations
    warmup_iterations: int = 3
    benchmark_iterations: int = 10

    # Model configuration
    hidden_dim: int = 768
    n_layers: int = 12
    sequence_length: int = 128
    batch_size: int = 1

    # SAE configuration
    sae_expansion_factor: int = 4
    sae_n_features: int = 3072  # hidden_dim * expansion_factor

    # Fingerprinting
    fingerprint_n_features: int = 50
    fingerprint_n_samples: int = 100

    # Circuit tracing
    circuit_max_depth: int = 3
    circuit_max_nodes: int = 100

    # Categories to run
    categories: list[BenchmarkCategory] = field(default_factory=lambda: list(BenchmarkCategory))

    # Output
    verbose: bool = True
    save_results: bool = True
    output_path: str = "benchmark_results.json"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    category: BenchmarkCategory
    iterations: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_delta_mb: float
    memory_peak_mb: float
    throughput: float  # operations per second
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Latency: {self.mean_latency_ms:.3f}ms ± {self.std_latency_ms:.3f}ms "
            f"(p95={self.p95_latency_ms:.3f}ms, p99={self.p99_latency_ms:.3f}ms)\n"
            f"  Memory: delta={self.memory_delta_mb:.2f}MB, peak={self.memory_peak_mb:.2f}MB\n"
            f"  Throughput: {self.throughput:.1f} ops/s"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "iterations": self.iterations,
            "latency": {
                "mean_ms": self.mean_latency_ms,
                "std_ms": self.std_latency_ms,
                "min_ms": self.min_latency_ms,
                "max_ms": self.max_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
            },
            "memory": {
                "delta_mb": self.memory_delta_mb,
                "peak_mb": self.memory_peak_mb,
            },
            "throughput_ops_per_sec": self.throughput,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuiteResult:
    """Results from running a full benchmark suite."""
    config: BenchmarkConfig
    results: list[BenchmarkResult]
    total_time_seconds: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "warmup_iterations": self.config.warmup_iterations,
                "benchmark_iterations": self.config.benchmark_iterations,
                "hidden_dim": self.config.hidden_dim,
                "n_layers": self.config.n_layers,
                "sequence_length": self.config.sequence_length,
                "batch_size": self.config.batch_size,
            },
            "results": [r.to_dict() for r in self.results],
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for Dark Trace operations.

    Runs benchmarks across all categories with configurable
    iterations and reports detailed performance metrics.
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.profiler = CombinedProfiler()
        self.results: list[BenchmarkResult] = []

    def _run_benchmark(
        self,
        name: str,
        category: BenchmarkCategory,
        operation: Callable[[], Any],
        setup: Callable[[], None] | None = None,
        teardown: Callable[[], None] | None = None,
        **metadata
    ) -> BenchmarkResult:
        """Run a single benchmark with profiling."""
        latencies_ns: list[int] = []

        # Warmup
        for _ in range(self.config.warmup_iterations):
            if setup:
                setup()
            operation()
            if teardown:
                teardown()

        # Reset profiler
        self.profiler.memory.clear()

        # Benchmark iterations
        self.profiler.memory.start()
        initial_snapshot = self.profiler.memory.snapshot("benchmark_start")

        for _ in range(self.config.benchmark_iterations):
            if setup:
                setup()

            start = time.perf_counter_ns()
            operation()
            end = time.perf_counter_ns()

            latencies_ns.append(end - start)

            if teardown:
                teardown()

        final_snapshot = self.profiler.memory.snapshot("benchmark_end")

        # Calculate statistics
        latencies_ms = [ns / 1_000_000 for ns in latencies_ns]

        mean_latency = np.mean(latencies_ms)
        total_time = sum(latencies_ms) / 1000  # seconds

        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=self.config.benchmark_iterations,
            mean_latency_ms=float(np.mean(latencies_ms)),
            std_latency_ms=float(np.std(latencies_ms)),
            min_latency_ms=float(np.min(latencies_ms)),
            max_latency_ms=float(np.max(latencies_ms)),
            p50_latency_ms=float(np.percentile(latencies_ms, 50)),
            p95_latency_ms=float(np.percentile(latencies_ms, 95)),
            p99_latency_ms=float(np.percentile(latencies_ms, 99)),
            memory_delta_mb=(final_snapshot.current_bytes - initial_snapshot.current_bytes) / (1024 * 1024),
            memory_peak_mb=final_snapshot.peak_mb,
            throughput=self.config.benchmark_iterations / total_time if total_time > 0 else 0,
            metadata=metadata
        )

        self.results.append(result)

        if self.config.verbose:
            print(f"\n{result}")

        return result

    def benchmark_activation_extraction(self, adapter, inputs, layers: list[str]) -> list[BenchmarkResult]:
        """Benchmark activation extraction operations."""
        results = []

        for layer in layers[:3]:  # Limit to 3 layers
            result = self._run_benchmark(
                name=f"activation_extraction_{layer}",
                category=BenchmarkCategory.ACTIVATION_EXTRACTION,
                operation=lambda l=layer: adapter.get_activations(inputs, l),
                layer=layer
            )
            results.append(result)

        # Multi-layer extraction
        if len(layers) >= 3:
            result = self._run_benchmark(
                name="activation_extraction_multi_layer",
                category=BenchmarkCategory.ACTIVATION_EXTRACTION,
                operation=lambda: adapter.get_multi_layer_activations(inputs, layers[:3]),
                n_layers=3
            )
            results.append(result)

        return results

    def benchmark_fingerprinting(self, adapter, inputs) -> list[BenchmarkResult]:
        """Benchmark fingerprinting operations."""
        from hololoom.dark_trace.models import ModelFingerprinter

        fingerprinter = ModelFingerprinter(adapter, model_id="benchmark")
        layers = adapter.get_layer_names()
        test_layer = layers[len(layers) // 2]

        results = []

        # Small fingerprint
        result = self._run_benchmark(
            name="fingerprint_small",
            category=BenchmarkCategory.FINGERPRINTING,
            operation=lambda: fingerprinter.fingerprint_layer(
                test_layer, inputs, n_features=10
            ),
            n_features=10
        )
        results.append(result)

        # Medium fingerprint
        result = self._run_benchmark(
            name="fingerprint_medium",
            category=BenchmarkCategory.FINGERPRINTING,
            operation=lambda: fingerprinter.fingerprint_layer(
                test_layer, inputs, n_features=50
            ),
            n_features=50
        )
        results.append(result)

        return results

    def benchmark_steering(self, adapter, layer: str) -> list[BenchmarkResult]:
        """Benchmark steering operations."""
        steering_vector = np.random.randn(self.config.hidden_dim).astype(np.float32)
        results = []

        # Steering injection only
        config_ids = []

        def inject_op():
            config_id = adapter.inject_steering(steering_vector, layer, scale=0.1)
            config_ids.append(config_id)

        def cleanup():
            if config_ids:
                adapter.remove_steering(config_ids.pop())

        result = self._run_benchmark(
            name="steering_injection",
            category=BenchmarkCategory.STEERING,
            operation=inject_op,
            teardown=cleanup,
            hidden_dim=self.config.hidden_dim
        )
        results.append(result)

        # Injection + removal cycle
        def cycle_op():
            config_id = adapter.inject_steering(steering_vector, layer, scale=0.1)
            adapter.remove_steering(config_id)

        result = self._run_benchmark(
            name="steering_cycle",
            category=BenchmarkCategory.STEERING,
            operation=cycle_op,
            hidden_dim=self.config.hidden_dim
        )
        results.append(result)

        return results

    def benchmark_sae_forward(self, sae, activations: np.ndarray) -> list[BenchmarkResult]:
        """Benchmark SAE forward pass."""
        results = []

        # Encode
        result = self._run_benchmark(
            name="sae_encode",
            category=BenchmarkCategory.SAE_FORWARD,
            operation=lambda: sae.encode(activations),
            input_shape=activations.shape
        )
        results.append(result)

        # Decode
        encoded = sae.encode(activations)
        result = self._run_benchmark(
            name="sae_decode",
            category=BenchmarkCategory.SAE_FORWARD,
            operation=lambda: sae.decode(encoded),
            encoded_shape=encoded.shape
        )
        results.append(result)

        # Full forward (encode + decode)
        result = self._run_benchmark(
            name="sae_forward",
            category=BenchmarkCategory.SAE_FORWARD,
            operation=lambda: sae.forward(activations),
            input_shape=activations.shape
        )
        results.append(result)

        return results

    def run_all(self, adapter=None, sae=None) -> BenchmarkSuiteResult:
        """Run all applicable benchmarks."""
        from datetime import datetime

        start_time = time.time()
        self.results.clear()

        if self.config.verbose:
            print("="*60)
            print("DARK TRACE BENCHMARK SUITE")
            print("="*60)
            print(f"Config: {self.config.benchmark_iterations} iterations, "
                  f"hidden_dim={self.config.hidden_dim}")

        # Create dummy inputs if no adapter provided
        if adapter is None:
            from hololoom.dark_trace.models import DummyAdapter
            adapter = DummyAdapter(
                n_layers=self.config.n_layers,
                hidden_dim=self.config.hidden_dim
            )

        # Create test inputs
        inputs = np.random.randn(
            self.config.batch_size,
            self.config.sequence_length,
            self.config.hidden_dim
        ).astype(np.float32)

        layers = adapter.get_layer_names()

        # Run benchmarks by category
        if BenchmarkCategory.ACTIVATION_EXTRACTION in self.config.categories:
            if self.config.verbose:
                print("\n--- ACTIVATION EXTRACTION ---")
            self.benchmark_activation_extraction(adapter, inputs, layers)

        if BenchmarkCategory.FINGERPRINTING in self.config.categories:
            if self.config.verbose:
                print("\n--- FINGERPRINTING ---")
            self.benchmark_fingerprinting(adapter, inputs)

        if BenchmarkCategory.STEERING in self.config.categories:
            if self.config.verbose:
                print("\n--- STEERING ---")
            test_layer = layers[len(layers) // 2]
            self.benchmark_steering(adapter, test_layer)

        if sae is not None and BenchmarkCategory.SAE_FORWARD in self.config.categories:
            if self.config.verbose:
                print("\n--- SAE FORWARD ---")
            activations = np.random.randn(
                self.config.batch_size * self.config.sequence_length,
                self.config.hidden_dim
            ).astype(np.float32)
            self.benchmark_sae_forward(sae, activations)

        total_time = time.time() - start_time

        suite_result = BenchmarkSuiteResult(
            config=self.config,
            results=self.results,
            total_time_seconds=total_time,
            timestamp=datetime.now().isoformat()
        )

        if self.config.save_results:
            suite_result.save(self.config.output_path)
            if self.config.verbose:
                print(f"\nResults saved to: {self.config.output_path}")

        if self.config.verbose:
            print("\n" + "="*60)
            print(f"COMPLETE: {len(self.results)} benchmarks in {total_time:.2f}s")
            print("="*60)

        return suite_result


def run_benchmark_suite(
    adapter=None,
    sae=None,
    config: BenchmarkConfig | None = None
) -> BenchmarkSuiteResult:
    """Convenience function to run benchmark suite."""
    suite = BenchmarkSuite(config)
    return suite.run_all(adapter=adapter, sae=sae)
