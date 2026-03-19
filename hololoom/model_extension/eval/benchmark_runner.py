"""
Benchmark Runner for External Benchmarks (Phase 5)
===================================================

Connects HoloLoom's model_extension module to external benchmarks
for standardized comparison and evaluation.

Supported Benchmarks:
- MTEB (Massive Text Embedding Benchmark) - Retrieval subset
- RAFT (Real-world Annotated Few-shot Tasks) - Domain adaptation
- Custom HoloLoom Benchmark - Memory-augmented specific tasks

Author: HoloLoom Architecture Team
Date: December 2025
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class BenchmarkType(Enum):
    """Supported benchmark types."""
    MTEB_RETRIEVAL = "mteb_retrieval"
    RAFT_DOMAIN = "raft_domain"
    HOLOLOOM_MEMORY = "hololoom_memory"
    CUSTOM = "custom"


class BenchmarkStatus(Enum):
    """Status of benchmark execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BenchmarkTask:
    """A single benchmark task/query."""
    task_id: str
    query: str
    expected_docs: list[str]  # Document IDs that should be retrieved
    relevance_scores: dict[str, float]  # doc_id -> relevance (0-3 scale)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """A benchmark dataset containing multiple tasks."""
    name: str
    benchmark_type: BenchmarkType
    tasks: list[BenchmarkTask]
    corpus: dict[str, str]  # doc_id -> document text
    description: str = ""
    version: str = "1.0"
    source_url: str | None = None


@dataclass
class TaskResult:
    """Result for a single benchmark task."""
    task_id: str
    retrieved_docs: list[str]  # Retrieved document IDs in order
    retrieval_scores: list[float]  # Corresponding scores
    latency_ms: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Overall benchmark result."""
    benchmark_name: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    task_results: list[TaskResult]
    aggregate_metrics: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_type": self.benchmark_type.value,
            "status": self.status.value,
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "retrieved_docs": tr.retrieved_docs,
                    "latency_ms": tr.latency_ms,
                    "metrics": tr.metrics,
                }
                for tr in self.task_results
            ],
            "aggregate_metrics": self.aggregate_metrics,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "config": self.config,
            "errors": self.errors,
        }


class RetrievalSystem(Protocol):
    """Protocol for systems that can be benchmarked."""

    async def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> tuple[list[str], list[float]]:
        """
        Retrieve documents for a query.

        Returns:
            Tuple of (doc_ids, scores)
        """
        ...

    async def index_corpus(
        self,
        corpus: dict[str, str],
    ) -> None:
        """Index a corpus of documents."""
        ...


class BenchmarkRunner:
    """
    Runner for external benchmarks.

    Supports MTEB retrieval, RAFT domain adaptation, and custom benchmarks.
    """

    def __init__(
        self,
        retrieval_system: RetrievalSystem | None = None,
        k_values: list[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize benchmark runner.

        Args:
            retrieval_system: System to benchmark (must implement RetrievalSystem protocol)
            k_values: Values of k for retrieval metrics (default: [1, 3, 5, 10])
            verbose: Print progress during benchmarking
        """
        self.retrieval_system = retrieval_system
        self.k_values = k_values or [1, 3, 5, 10]
        self.verbose = verbose
        self._results_history: list[BenchmarkResult] = []

    async def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        k: int = 10,
    ) -> BenchmarkResult:
        """
        Run a benchmark on the retrieval system.

        Args:
            dataset: Benchmark dataset to run
            k: Number of documents to retrieve per query

        Returns:
            BenchmarkResult with all metrics
        """
        if self.retrieval_system is None:
            return BenchmarkResult(
                benchmark_name=dataset.name,
                benchmark_type=dataset.benchmark_type,
                status=BenchmarkStatus.FAILED,
                task_results=[],
                aggregate_metrics={},
                errors=["No retrieval system provided"],
            )

        start_time = time.time()

        # Index corpus
        if self.verbose:
            print(f"Indexing corpus with {len(dataset.corpus)} documents...")

        try:
            await self.retrieval_system.index_corpus(dataset.corpus)
        except Exception as e:
            return BenchmarkResult(
                benchmark_name=dataset.name,
                benchmark_type=dataset.benchmark_type,
                status=BenchmarkStatus.FAILED,
                task_results=[],
                aggregate_metrics={},
                errors=[f"Failed to index corpus: {str(e)}"],
            )

        # Run tasks
        task_results = []
        errors = []

        for i, task in enumerate(dataset.tasks):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Running task {i + 1}/{len(dataset.tasks)}...")

            try:
                task_start = time.time()
                doc_ids, scores = await self.retrieval_system.retrieve(task.query, k=k)
                task_latency = (time.time() - task_start) * 1000

                # Calculate per-task metrics
                task_metrics = self._calculate_task_metrics(
                    retrieved=doc_ids,
                    expected=task.expected_docs,
                    relevance_scores=task.relevance_scores,
                )

                task_results.append(TaskResult(
                    task_id=task.task_id,
                    retrieved_docs=doc_ids,
                    retrieval_scores=scores,
                    latency_ms=task_latency,
                    metrics=task_metrics,
                ))
            except Exception as e:
                errors.append(f"Task {task.task_id} failed: {str(e)}")
                task_results.append(TaskResult(
                    task_id=task.task_id,
                    retrieved_docs=[],
                    retrieval_scores=[],
                    latency_ms=0,
                    metrics={},
                ))

        # Calculate aggregate metrics
        aggregate_metrics = self._aggregate_metrics(task_results)

        duration = time.time() - start_time
        status = BenchmarkStatus.COMPLETED if not errors else BenchmarkStatus.COMPLETED

        result = BenchmarkResult(
            benchmark_name=dataset.name,
            benchmark_type=dataset.benchmark_type,
            status=status,
            task_results=task_results,
            aggregate_metrics=aggregate_metrics,
            duration_seconds=duration,
            config={"k": k, "k_values": self.k_values},
            errors=errors,
        )

        self._results_history.append(result)
        return result

    def _calculate_task_metrics(
        self,
        retrieved: list[str],
        expected: list[str],
        relevance_scores: dict[str, float],
    ) -> dict[str, float]:
        """Calculate retrieval metrics for a single task."""
        metrics = {}

        expected_set = set(expected)

        # Precision@k and Recall@k for different k values
        for k in self.k_values:
            top_k = set(retrieved[:k])

            # Precision@k
            precision = len(top_k & expected_set) / k if k > 0 else 0
            metrics[f"precision@{k}"] = precision

            # Recall@k
            recall = len(top_k & expected_set) / len(expected_set) if expected_set else 0
            metrics[f"recall@{k}"] = recall

        # Hit Rate (at least one relevant doc in top-k)
        for k in self.k_values:
            top_k = set(retrieved[:k])
            hit = 1.0 if (top_k & expected_set) else 0.0
            metrics[f"hit_rate@{k}"] = hit

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in expected_set:
                mrr = 1.0 / rank
                break
        metrics["mrr"] = mrr

        # NDCG@k
        for k in self.k_values:
            ndcg = self._calculate_ndcg(retrieved[:k], relevance_scores)
            metrics[f"ndcg@{k}"] = ndcg

        # MAP (Mean Average Precision)
        ap = self._calculate_average_precision(retrieved, expected_set)
        metrics["map"] = ap

        return metrics

    def _calculate_ndcg(
        self,
        retrieved: list[str],
        relevance_scores: dict[str, float],
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not retrieved:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2 ** rel - 1) / (i + 2)  # log2(i+2) simplified

        # Ideal DCG
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:len(retrieved)]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2 ** rel - 1) / (i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_average_precision(
        self,
        retrieved: list[str],
        expected_set: set,
    ) -> float:
        """Calculate Average Precision."""
        if not expected_set:
            return 0.0

        hits = 0
        sum_precisions = 0.0

        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in expected_set:
                hits += 1
                precision_at_rank = hits / rank
                sum_precisions += precision_at_rank

        return sum_precisions / len(expected_set)

    def _aggregate_metrics(
        self,
        task_results: list[TaskResult],
    ) -> dict[str, float]:
        """Aggregate metrics across all tasks."""
        if not task_results:
            return {}

        # Collect all metric values
        all_metrics: dict[str, list[float]] = {}
        latencies = []

        for result in task_results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
            latencies.append(result.latency_ms)

        # Calculate means
        aggregate = {}
        for metric_name, values in all_metrics.items():
            aggregate[f"mean_{metric_name}"] = sum(values) / len(values)

        # Latency statistics
        if latencies:
            aggregate["mean_latency_ms"] = sum(latencies) / len(latencies)
            aggregate["p50_latency_ms"] = sorted(latencies)[len(latencies) // 2]
            aggregate["p95_latency_ms"] = sorted(latencies)[int(len(latencies) * 0.95)]

        aggregate["total_tasks"] = len(task_results)
        aggregate["successful_tasks"] = sum(1 for r in task_results if r.metrics)

        return aggregate

    def get_results_history(self) -> list[BenchmarkResult]:
        """Get history of all benchmark results."""
        return self._results_history.copy()

    def export_results(self, filepath: str) -> None:
        """Export all results to JSON file."""
        results = [r.to_dict() for r in self._results_history]
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)


# =============================================================================
# MTEB Benchmark Integration
# =============================================================================

class MTEBBenchmark:
    """
    Integration with MTEB (Massive Text Embedding Benchmark).

    Focuses on retrieval subset for HoloLoom evaluation.
    """

    RETRIEVAL_DATASETS = [
        "msmarco",
        "nfcorpus",
        "scifact",
        "arguana",
        "fiqa",
    ]

    @staticmethod
    def create_sample_dataset(name: str = "mteb_sample") -> BenchmarkDataset:
        """
        Create a sample MTEB-style dataset for testing.

        In production, this would load actual MTEB datasets.
        """
        # Sample corpus
        corpus = {
            "doc1": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            "doc2": "Multi-armed bandits balance exploration and exploitation in sequential decision making.",
            "doc3": "UCB (Upper Confidence Bound) is an alternative to Thompson Sampling.",
            "doc4": "Bayesian methods use prior distributions to incorporate uncertainty.",
            "doc5": "Reinforcement learning deals with sequential decision problems.",
            "doc6": "The explore-exploit dilemma is fundamental in online learning.",
            "doc7": "Posterior sampling selects actions proportionally to their probability of being optimal.",
            "doc8": "Regret measures the difference between optimal and actual performance.",
            "doc9": "Confidence intervals quantify uncertainty in parameter estimates.",
            "doc10": "Online learning algorithms update incrementally with new data.",
        }

        # Sample tasks
        tasks = [
            BenchmarkTask(
                task_id="q1",
                query="What is Thompson Sampling?",
                expected_docs=["doc1", "doc7"],
                relevance_scores={"doc1": 3, "doc7": 2, "doc2": 1, "doc4": 1},
            ),
            BenchmarkTask(
                task_id="q2",
                query="exploration exploitation tradeoff",
                expected_docs=["doc2", "doc6"],
                relevance_scores={"doc2": 3, "doc6": 3, "doc5": 1},
            ),
            BenchmarkTask(
                task_id="q3",
                query="Bayesian methods uncertainty",
                expected_docs=["doc4", "doc1", "doc9"],
                relevance_scores={"doc4": 3, "doc1": 2, "doc9": 2, "doc7": 1},
            ),
            BenchmarkTask(
                task_id="q4",
                query="bandit algorithms comparison",
                expected_docs=["doc3", "doc1", "doc2"],
                relevance_scores={"doc3": 3, "doc1": 2, "doc2": 2},
            ),
            BenchmarkTask(
                task_id="q5",
                query="online learning sequential",
                expected_docs=["doc10", "doc5", "doc2"],
                relevance_scores={"doc10": 3, "doc5": 2, "doc2": 2},
            ),
        ]

        return BenchmarkDataset(
            name=name,
            benchmark_type=BenchmarkType.MTEB_RETRIEVAL,
            tasks=tasks,
            corpus=corpus,
            description="Sample MTEB-style retrieval benchmark",
            source_url="https://huggingface.co/datasets/mteb",
        )


# =============================================================================
# RAFT Benchmark Integration
# =============================================================================

class RAFTBenchmark:
    """
    Integration with RAFT (Real-world Annotated Few-shot Tasks).

    Focuses on domain adaptation evaluation.
    """

    DOMAINS = [
        "banking",
        "legal",
        "medical",
        "scientific",
    ]

    @staticmethod
    def create_domain_dataset(
        domain: str = "scientific",
    ) -> BenchmarkDataset:
        """
        Create a RAFT-style domain adaptation dataset.

        In production, this would load actual RAFT datasets.
        """
        if domain == "scientific":
            corpus = {
                "sci1": "Neural networks learn hierarchical representations through backpropagation.",
                "sci2": "Attention mechanisms allow models to focus on relevant input parts.",
                "sci3": "Transformers use self-attention for parallel sequence processing.",
                "sci4": "Gradient descent optimizes parameters by following the loss surface.",
                "sci5": "Dropout prevents overfitting by randomly deactivating neurons.",
                "sci6": "Batch normalization stabilizes training by normalizing activations.",
                "sci7": "Convolutional layers detect local patterns in spatial data.",
                "sci8": "Recurrent networks process sequential data with internal memory.",
            }

            tasks = [
                BenchmarkTask(
                    task_id="raft_sci_1",
                    query="How do neural networks learn?",
                    expected_docs=["sci1", "sci4"],
                    relevance_scores={"sci1": 3, "sci4": 2, "sci5": 1},
                ),
                BenchmarkTask(
                    task_id="raft_sci_2",
                    query="attention mechanism transformer",
                    expected_docs=["sci2", "sci3"],
                    relevance_scores={"sci2": 3, "sci3": 3},
                ),
                BenchmarkTask(
                    task_id="raft_sci_3",
                    query="prevent overfitting techniques",
                    expected_docs=["sci5", "sci6"],
                    relevance_scores={"sci5": 3, "sci6": 2},
                ),
            ]
        else:
            # Default/generic domain
            corpus = {
                "gen1": f"{domain} domain document 1",
                "gen2": f"{domain} domain document 2",
            }
            tasks = [
                BenchmarkTask(
                    task_id=f"raft_{domain}_1",
                    query=f"{domain} query",
                    expected_docs=["gen1"],
                    relevance_scores={"gen1": 3},
                ),
            ]

        return BenchmarkDataset(
            name=f"raft_{domain}",
            benchmark_type=BenchmarkType.RAFT_DOMAIN,
            tasks=tasks,
            corpus=corpus,
            description=f"RAFT-style {domain} domain benchmark",
            source_url="https://huggingface.co/datasets/ought/raft",
        )


# =============================================================================
# HoloLoom Custom Benchmark
# =============================================================================

class HoloLoomBenchmark:
    """
    Custom benchmark designed for HoloLoom's memory-augmented capabilities.

    Tests unique features:
    - Multi-hop reasoning retrieval
    - Temporal knowledge queries
    - Cross-reference resolution
    """

    @staticmethod
    def create_memory_augmented_dataset() -> BenchmarkDataset:
        """
        Create a benchmark testing memory-augmented retrieval features.
        """
        corpus = {
            # Foundation knowledge
            "mem1": "Thompson Sampling was introduced by Thompson in 1933.",
            "mem2": "The multi-armed bandit problem models exploration vs exploitation.",
            "mem3": "Bayesian inference updates beliefs based on evidence.",

            # Connected knowledge (multi-hop)
            "mem4": "Thompson Sampling uses Bayesian posterior sampling for action selection.",
            "mem5": "Beta distributions are commonly used as priors for Thompson Sampling.",

            # Temporal knowledge
            "mem6": "UCB was developed after Thompson Sampling but gained popularity first.",
            "mem7": "Recent work shows Thompson Sampling is often optimal asymptotically.",

            # Cross-references
            "mem8": "Both Thompson Sampling and UCB solve the same bandit problem.",
            "mem9": "Posterior sampling is another name for Thompson Sampling.",

            # Domain-specific
            "mem10": "In recommendation systems, Thompson Sampling personalizes content.",
            "mem11": "A/B testing can be optimized using bandit algorithms.",
            "mem12": "Clinical trials use bandit methods for adaptive allocation.",
        }

        tasks = [
            # Multi-hop reasoning
            BenchmarkTask(
                task_id="holo_multihop_1",
                query="How does Thompson Sampling use Bayesian methods?",
                expected_docs=["mem4", "mem3", "mem5"],
                relevance_scores={"mem4": 3, "mem3": 2, "mem5": 2, "mem1": 1},
                metadata={"type": "multi_hop"},
            ),

            # Temporal query
            BenchmarkTask(
                task_id="holo_temporal_1",
                query="What is the historical development of bandit algorithms?",
                expected_docs=["mem1", "mem6", "mem7"],
                relevance_scores={"mem1": 3, "mem6": 3, "mem7": 2},
                metadata={"type": "temporal"},
            ),

            # Cross-reference resolution
            BenchmarkTask(
                task_id="holo_crossref_1",
                query="What is posterior sampling?",
                expected_docs=["mem9", "mem4", "mem1"],
                relevance_scores={"mem9": 3, "mem4": 2, "mem1": 1},
                metadata={"type": "cross_reference"},
            ),

            # Domain application
            BenchmarkTask(
                task_id="holo_domain_1",
                query="How are bandits used in real applications?",
                expected_docs=["mem10", "mem11", "mem12"],
                relevance_scores={"mem10": 3, "mem11": 3, "mem12": 3},
                metadata={"type": "domain_application"},
            ),

            # Comparison query
            BenchmarkTask(
                task_id="holo_compare_1",
                query="Compare Thompson Sampling and UCB",
                expected_docs=["mem8", "mem6", "mem2"],
                relevance_scores={"mem8": 3, "mem6": 2, "mem2": 2},
                metadata={"type": "comparison"},
            ),
        ]

        return BenchmarkDataset(
            name="hololoom_memory_augmented",
            benchmark_type=BenchmarkType.HOLOLOOM_MEMORY,
            tasks=tasks,
            corpus=corpus,
            description="Custom benchmark for HoloLoom memory-augmented retrieval",
            version="1.0",
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_benchmark_runner(
    retrieval_system: RetrievalSystem | None = None,
    verbose: bool = False,
) -> BenchmarkRunner:
    """Create a benchmark runner with default settings."""
    return BenchmarkRunner(
        retrieval_system=retrieval_system,
        k_values=[1, 3, 5, 10],
        verbose=verbose,
    )


def get_all_sample_datasets() -> list[BenchmarkDataset]:
    """Get all sample benchmark datasets for testing."""
    return [
        MTEBBenchmark.create_sample_dataset(),
        RAFTBenchmark.create_domain_dataset("scientific"),
        HoloLoomBenchmark.create_memory_augmented_dataset(),
    ]


async def run_all_benchmarks(
    retrieval_system: RetrievalSystem,
    verbose: bool = True,
) -> dict[str, BenchmarkResult]:
    """
    Run all available benchmarks on a retrieval system.

    Args:
        retrieval_system: System to benchmark
        verbose: Print progress

    Returns:
        Dictionary of benchmark_name -> result
    """
    runner = create_benchmark_runner(retrieval_system, verbose=verbose)
    datasets = get_all_sample_datasets()

    results = {}
    for dataset in datasets:
        if verbose:
            print(f"\nRunning benchmark: {dataset.name}")
        result = await runner.run_benchmark(dataset)
        results[dataset.name] = result

        if verbose:
            print(f"  Status: {result.status.value}")
            print(f"  Duration: {result.duration_seconds:.2f}s")
            if result.aggregate_metrics:
                ndcg = result.aggregate_metrics.get("mean_ndcg@10", 0)
                mrr = result.aggregate_metrics.get("mean_mrr", 0)
                print(f"  NDCG@10: {ndcg:.3f}")
                print(f"  MRR: {mrr:.3f}")

    return results
