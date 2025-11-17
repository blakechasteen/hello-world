"""
HoloLoom Benchmark Base Classes
================================

Standard interface for all HoloLoom benchmarks.

Ensures consistency:
- Standardized output format (JSON)
- Reproducibility (seeded random)
- Error handling
- Progress reporting
- Result validation

Author: HoloLoom Team
Date: 2025-11-17
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import time
import random
import sys


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BenchmarkMetrics:
    """Standard metrics for all benchmarks."""
    # Accuracy metrics
    precision_at_1: Optional[float] = None
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    mrr: Optional[float] = None  # Mean Reciprocal Rank
    ndcg: Optional[float] = None  # Normalized Discounted Cumulative Gain

    # Performance metrics
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_qps: Optional[float] = None

    # Resource metrics
    memory_mb: Optional[float] = None
    disk_mb: Optional[float] = None
    index_build_time_sec: Optional[float] = None

    # Learning metrics
    accuracy_improvement: Optional[float] = None
    convergence_queries: Optional[int] = None
    adaptation_rate: Optional[float] = None

    # Custom metrics
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    dataset_name: str
    num_memories: int
    num_queries: int
    seed: int = 42
    config_mode: str = "fast"  # BARE, FAST, FUSED
    enable_cache: bool = True
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    benchmark_name: str
    version: str
    timestamp: str
    metrics: BenchmarkMetrics
    config: BenchmarkConfig
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'benchmark_name': self.benchmark_name,
            'version': self.version,
            'timestamp': self.timestamp,
            'metrics': asdict(self.metrics),
            'config': asdict(self.config),
            'dataset_info': self.dataset_info,
            'errors': self.errors,
            'warnings': self.warnings
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, output_path: Path):
        """Save to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(self.to_json())


# ============================================================================
# Base Benchmark Class
# ============================================================================

class BaseBenchmark:
    """
    Base class for all HoloLoom benchmarks.

    Subclasses must implement:
    - load_dataset()
    - run_benchmark()
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0",
        seed: int = 42
    ):
        """
        Initialize benchmark.

        Args:
            name: Benchmark name (e.g., "recall_accuracy_wikipedia")
            version: Benchmark version (for reproducibility)
            seed: Random seed (for determinism)
        """
        self.name = name
        self.version = version
        self.seed = seed

        # Set random seed
        random.seed(seed)

        # Results
        self.result: Optional[BenchmarkResult] = None

    def load_dataset(self) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Load benchmark dataset.

        Returns:
            (data, info) tuple where:
            - data: List of dataset items
            - info: Dict with dataset metadata

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement load_dataset()")

    def run_benchmark(
        self,
        data: List[Any],
        config: BenchmarkConfig
    ) -> BenchmarkMetrics:
        """
        Run benchmark on dataset.

        Args:
            data: Dataset items
            config: Benchmark configuration

        Returns:
            BenchmarkMetrics with results

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement run_benchmark()")

    def run(
        self,
        num_memories: int = 1000,
        num_queries: int = 100,
        config_mode: str = "fast"
    ) -> BenchmarkResult:
        """
        Run complete benchmark.

        Args:
            num_memories: Number of memories to test
            num_queries: Number of queries to run
            config_mode: HoloLoom config mode (BARE/FAST/FUSED)

        Returns:
            BenchmarkResult with complete results
        """
        print(f"\n{'='*70}")
        print(f"Running benchmark: {self.name}")
        print(f"Version: {self.version}")
        print(f"Seed: {self.seed}")
        print(f"{'='*70}\n")

        # Load dataset
        print("Loading dataset...")
        start_time = time.time()
        try:
            data, dataset_info = self.load_dataset()
            load_time = time.time() - start_time
            print(f"✓ Loaded {len(data)} items in {load_time:.2f}s\n")
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            return BenchmarkResult(
                benchmark_name=self.name,
                version=self.version,
                timestamp=datetime.now().isoformat(),
                metrics=BenchmarkMetrics(),
                config=BenchmarkConfig(
                    dataset_name="unknown",
                    num_memories=num_memories,
                    num_queries=num_queries,
                    seed=self.seed,
                    config_mode=config_mode
                ),
                errors=[f"Dataset load failed: {e}"]
            )

        # Create config
        config = BenchmarkConfig(
            dataset_name=dataset_info.get('name', 'unknown'),
            num_memories=min(num_memories, len(data)),
            num_queries=num_queries,
            seed=self.seed,
            config_mode=config_mode
        )

        # Run benchmark
        print(f"Running benchmark (mode={config_mode})...")
        start_time = time.time()
        errors = []
        warnings = []

        try:
            metrics = self.run_benchmark(data[:config.num_memories], config)
            run_time = time.time() - start_time
            print(f"✓ Completed in {run_time:.2f}s\n")
        except Exception as e:
            print(f"✗ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            errors.append(f"Benchmark run failed: {e}")
            metrics = BenchmarkMetrics()

        # Create result
        self.result = BenchmarkResult(
            benchmark_name=self.name,
            version=self.version,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config=config,
            dataset_info=dataset_info,
            errors=errors,
            warnings=warnings
        )

        # Print summary
        self._print_summary()

        return self.result

    def _print_summary(self):
        """Print benchmark summary."""
        if not self.result:
            return

        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}\n")

        metrics = self.result.metrics

        # Accuracy metrics
        if metrics.precision_at_5 is not None:
            print(f"Precision@5:  {metrics.precision_at_5:.3f}")
        if metrics.recall_at_5 is not None:
            print(f"Recall@5:     {metrics.recall_at_5:.3f}")
        if metrics.mrr is not None:
            print(f"MRR:          {metrics.mrr:.3f}")

        # Performance metrics
        if metrics.latency_p95_ms is not None:
            print(f"\nLatency p95:  {metrics.latency_p95_ms:.1f}ms")
        if metrics.throughput_qps is not None:
            print(f"Throughput:   {metrics.throughput_qps:.1f} queries/sec")

        # Errors/warnings
        if self.result.errors:
            print(f"\n⚠ Errors: {len(self.result.errors)}")
            for error in self.result.errors[:3]:
                print(f"  - {error}")

        if self.result.warnings:
            print(f"\n⚠ Warnings: {len(self.result.warnings)}")
            for warning in self.result.warnings[:3]:
                print(f"  - {warning}")

        print()


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Calculate Precision@K.

    Args:
        retrieved: List of retrieved items (ordered by rank)
        relevant: List of relevant items (ground truth)
        k: Cutoff rank

    Returns:
        Precision@K score (0.0-1.0)
    """
    if k <= 0 or not retrieved:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def calculate_recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Calculate Recall@K.

    Args:
        retrieved: List of retrieved items (ordered by rank)
        relevant: List of relevant items (ground truth)
        k: Cutoff rank

    Returns:
        Recall@K score (0.0-1.0)
    """
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def calculate_mrr(
    retrieved_lists: List[List[str]],
    relevant_lists: List[List[str]]
) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.

    Args:
        retrieved_lists: List of retrieved item lists (one per query)
        relevant_lists: List of relevant item lists (one per query)

    Returns:
        MRR score (0.0-1.0)
    """
    if not retrieved_lists:
        return 0.0

    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        relevant_set = set(relevant)
        for rank, item in enumerate(retrieved, 1):
            if item in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate percentile of values.

    Args:
        values: List of numeric values
        percentile: Percentile (0-100)

    Returns:
        Percentile value
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(len(sorted_values) * (percentile / 100.0))
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]
