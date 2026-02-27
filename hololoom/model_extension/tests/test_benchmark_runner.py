"""
Test Suite for Benchmark Runner (Phase 5)
==========================================

Comprehensive tests for external benchmark integration:
- MTEB retrieval benchmarks
- RAFT domain adaptation benchmarks
- Custom HoloLoom memory-augmented benchmarks

Author: HoloLoom Architecture Team
Date: December 2025
"""

import pytest
import asyncio
from typing import List, Dict, Tuple

from hololoom.model_extension.eval.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkDataset,
    BenchmarkTask,
    BenchmarkType,
    BenchmarkStatus,
    TaskResult,
    MTEBBenchmark,
    RAFTBenchmark,
    HoloLoomBenchmark,
    create_benchmark_runner,
    get_all_sample_datasets,
    run_all_benchmarks,
)


# =============================================================================
# Mock Retrieval System
# =============================================================================

class MockRetrievalSystem:
    """Mock retrieval system for testing."""

    def __init__(self, perfect: bool = False):
        """
        Args:
            perfect: If True, always returns expected docs in order
        """
        self.perfect = perfect
        self.corpus: Dict[str, str] = {}
        self.indexed = False

    async def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Mock retrieval."""
        if not self.indexed or not self.corpus:
            return [], []

        # Return doc IDs in order, with decreasing scores
        doc_ids = list(self.corpus.keys())[:k]
        scores = [1.0 - i * 0.1 for i in range(len(doc_ids))]

        return doc_ids, scores

    async def index_corpus(
        self,
        corpus: Dict[str, str],
    ) -> None:
        """Mock indexing."""
        self.corpus = corpus.copy()
        self.indexed = True


class PerfectRetrievalSystem:
    """A retrieval system that returns expected docs first."""

    def __init__(self):
        self.corpus: Dict[str, str] = {}
        self.expected_docs: Dict[str, List[str]] = {}

    def set_expected(self, query: str, docs: List[str]) -> None:
        """Set expected docs for a query."""
        self.expected_docs[query] = docs

    async def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return expected docs first, then others."""
        expected = self.expected_docs.get(query, [])
        other_docs = [d for d in self.corpus.keys() if d not in expected]

        # Expected docs first, then others
        result_docs = (expected + other_docs)[:k]
        scores = [1.0 - i * 0.05 for i in range(len(result_docs))]

        return result_docs, scores

    async def index_corpus(
        self,
        corpus: Dict[str, str],
    ) -> None:
        """Index corpus."""
        self.corpus = corpus.copy()


class FailingRetrievalSystem:
    """A retrieval system that fails indexing."""

    async def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> Tuple[List[str], List[float]]:
        return [], []

    async def index_corpus(
        self,
        corpus: Dict[str, str],
    ) -> None:
        raise RuntimeError("Indexing failed!")


# =============================================================================
# BenchmarkRunner Tests
# =============================================================================

class TestBenchmarkRunner:
    """Tests for the main BenchmarkRunner class."""

    @pytest.mark.asyncio
    async def test_runner_no_system(self):
        """Runner without system should fail gracefully."""
        runner = BenchmarkRunner(retrieval_system=None)
        dataset = MTEBBenchmark.create_sample_dataset()

        result = await runner.run_benchmark(dataset)

        assert result.status == BenchmarkStatus.FAILED
        assert "No retrieval system" in result.errors[0]

    @pytest.mark.asyncio
    async def test_runner_basic_execution(self):
        """Test basic benchmark execution."""
        system = MockRetrievalSystem()
        runner = BenchmarkRunner(retrieval_system=system)
        dataset = MTEBBenchmark.create_sample_dataset()

        result = await runner.run_benchmark(dataset)

        assert result.status == BenchmarkStatus.COMPLETED
        assert result.benchmark_name == dataset.name
        assert result.benchmark_type == BenchmarkType.MTEB_RETRIEVAL
        assert len(result.task_results) == len(dataset.tasks)

    @pytest.mark.asyncio
    async def test_runner_aggregate_metrics(self):
        """Test aggregate metrics calculation."""
        system = MockRetrievalSystem()
        runner = BenchmarkRunner(retrieval_system=system)
        dataset = MTEBBenchmark.create_sample_dataset()

        result = await runner.run_benchmark(dataset)

        # Check aggregate metrics exist
        assert "mean_mrr" in result.aggregate_metrics
        assert "mean_ndcg@10" in result.aggregate_metrics
        assert "mean_latency_ms" in result.aggregate_metrics
        assert "total_tasks" in result.aggregate_metrics

    @pytest.mark.asyncio
    async def test_runner_k_values(self):
        """Test different k values for metrics."""
        system = MockRetrievalSystem()
        runner = BenchmarkRunner(
            retrieval_system=system,
            k_values=[1, 5, 10, 20],
        )
        dataset = MTEBBenchmark.create_sample_dataset()

        result = await runner.run_benchmark(dataset)

        # Check metrics for all k values
        for k in [1, 5, 10, 20]:
            assert f"mean_precision@{k}" in result.aggregate_metrics
            assert f"mean_recall@{k}" in result.aggregate_metrics
            assert f"mean_ndcg@{k}" in result.aggregate_metrics

    @pytest.mark.asyncio
    async def test_runner_with_perfect_system(self):
        """Test with a perfect retrieval system."""
        system = PerfectRetrievalSystem()
        dataset = MTEBBenchmark.create_sample_dataset()

        # Set expected docs for each task
        for task in dataset.tasks:
            system.set_expected(task.query, task.expected_docs)

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset)

        # Should have high metrics
        assert result.status == BenchmarkStatus.COMPLETED
        assert result.aggregate_metrics["mean_mrr"] > 0.5

    @pytest.mark.asyncio
    async def test_runner_failing_system(self):
        """Test with failing retrieval system."""
        system = FailingRetrievalSystem()
        runner = BenchmarkRunner(retrieval_system=system)
        dataset = MTEBBenchmark.create_sample_dataset()

        result = await runner.run_benchmark(dataset)

        assert result.status == BenchmarkStatus.FAILED
        assert len(result.errors) > 0
        assert "Failed to index corpus" in result.errors[0]

    @pytest.mark.asyncio
    async def test_runner_results_history(self):
        """Test results history tracking."""
        system = MockRetrievalSystem()
        runner = BenchmarkRunner(retrieval_system=system)

        # Run multiple benchmarks
        datasets = get_all_sample_datasets()
        for dataset in datasets:
            await runner.run_benchmark(dataset)

        history = runner.get_results_history()
        assert len(history) == len(datasets)

    def test_runner_to_dict(self):
        """Test result serialization."""
        result = BenchmarkResult(
            benchmark_name="test",
            benchmark_type=BenchmarkType.MTEB_RETRIEVAL,
            status=BenchmarkStatus.COMPLETED,
            task_results=[
                TaskResult(
                    task_id="q1",
                    retrieved_docs=["doc1", "doc2"],
                    retrieval_scores=[1.0, 0.9],
                    latency_ms=50.0,
                    metrics={"mrr": 1.0},
                )
            ],
            aggregate_metrics={"mean_mrr": 1.0},
            duration_seconds=1.5,
        )

        data = result.to_dict()

        assert data["benchmark_name"] == "test"
        assert data["status"] == "completed"
        assert len(data["task_results"]) == 1


# =============================================================================
# Metrics Calculation Tests
# =============================================================================

class TestMetricsCalculation:
    """Tests for metric calculations."""

    @pytest.mark.asyncio
    async def test_ndcg_calculation(self):
        """Test NDCG calculation."""
        system = PerfectRetrievalSystem()
        dataset = BenchmarkDataset(
            name="ndcg_test",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test query",
                    expected_docs=["doc1", "doc2"],
                    relevance_scores={"doc1": 3, "doc2": 2, "doc3": 0},
                ),
            ],
            corpus={"doc1": "a", "doc2": "b", "doc3": "c"},
        )

        # Set perfect retrieval
        system.set_expected("test query", ["doc1", "doc2"])

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset)

        # Check NDCG is calculated
        assert result.task_results[0].metrics["ndcg@3"] > 0

    @pytest.mark.asyncio
    async def test_mrr_calculation(self):
        """Test MRR calculation."""
        system = PerfectRetrievalSystem()
        dataset = BenchmarkDataset(
            name="mrr_test",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test query",
                    expected_docs=["doc1"],
                    relevance_scores={"doc1": 3},
                ),
            ],
            corpus={"doc1": "a", "doc2": "b", "doc3": "c"},
        )

        # Set doc1 first
        system.set_expected("test query", ["doc1"])

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset)

        # MRR should be 1.0 (doc1 at rank 1)
        assert result.task_results[0].metrics["mrr"] == 1.0

    @pytest.mark.asyncio
    async def test_precision_recall_calculation(self):
        """Test precision and recall calculation."""
        system = PerfectRetrievalSystem()
        dataset = BenchmarkDataset(
            name="pr_test",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test query",
                    expected_docs=["doc1", "doc2"],
                    relevance_scores={"doc1": 3, "doc2": 2},
                ),
            ],
            corpus={
                "doc1": "a", "doc2": "b", "doc3": "c",
                "doc4": "d", "doc5": "e",
            },
        )

        system.set_expected("test query", ["doc1", "doc2"])

        runner = BenchmarkRunner(
            retrieval_system=system,
            k_values=[1, 2, 5],
        )
        result = await runner.run_benchmark(dataset, k=5)

        metrics = result.task_results[0].metrics

        # Precision@1 should be 1.0 (1/1 correct)
        assert metrics["precision@1"] == 1.0

        # Precision@2 should be 1.0 (2/2 correct)
        assert metrics["precision@2"] == 1.0

        # Recall@2 should be 1.0 (2/2 expected found)
        assert metrics["recall@2"] == 1.0

    @pytest.mark.asyncio
    async def test_map_calculation(self):
        """Test MAP calculation."""
        system = PerfectRetrievalSystem()
        dataset = BenchmarkDataset(
            name="map_test",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test query",
                    expected_docs=["doc1", "doc3"],
                    relevance_scores={"doc1": 3, "doc3": 2},
                ),
            ],
            corpus={"doc1": "a", "doc2": "b", "doc3": "c", "doc4": "d"},
        )

        # Return doc1, doc2, doc3, doc4 (expected at positions 1 and 3)
        system.set_expected("test query", ["doc1", "doc3"])

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset)

        # MAP should be calculated
        assert "map" in result.task_results[0].metrics
        assert result.task_results[0].metrics["map"] > 0

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        system = PerfectRetrievalSystem()
        dataset = BenchmarkDataset(
            name="hit_test",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test query",
                    expected_docs=["doc1"],
                    relevance_scores={"doc1": 3},
                ),
            ],
            corpus={"doc1": "a", "doc2": "b", "doc3": "c"},
        )

        system.set_expected("test query", ["doc1"])

        runner = BenchmarkRunner(
            retrieval_system=system,
            k_values=[1, 3],
        )
        result = await runner.run_benchmark(dataset)

        # Hit rate@1 should be 1.0
        assert result.task_results[0].metrics["hit_rate@1"] == 1.0


# =============================================================================
# MTEB Benchmark Tests
# =============================================================================

class TestMTEBBenchmark:
    """Tests for MTEB benchmark integration."""

    def test_create_sample_dataset(self):
        """Test MTEB sample dataset creation."""
        dataset = MTEBBenchmark.create_sample_dataset()

        assert dataset.name == "mteb_sample"
        assert dataset.benchmark_type == BenchmarkType.MTEB_RETRIEVAL
        assert len(dataset.tasks) > 0
        assert len(dataset.corpus) > 0

    def test_mteb_dataset_structure(self):
        """Test MTEB dataset structure."""
        dataset = MTEBBenchmark.create_sample_dataset()

        for task in dataset.tasks:
            # Each task should have expected docs
            assert len(task.expected_docs) > 0
            # Relevance scores should exist for expected docs
            for doc_id in task.expected_docs:
                assert doc_id in task.relevance_scores

    @pytest.mark.asyncio
    async def test_mteb_benchmark_execution(self):
        """Test MTEB benchmark can be executed."""
        system = MockRetrievalSystem()
        runner = BenchmarkRunner(retrieval_system=system)
        dataset = MTEBBenchmark.create_sample_dataset()

        result = await runner.run_benchmark(dataset)

        assert result.status == BenchmarkStatus.COMPLETED
        assert result.benchmark_type == BenchmarkType.MTEB_RETRIEVAL


# =============================================================================
# RAFT Benchmark Tests
# =============================================================================

class TestRAFTBenchmark:
    """Tests for RAFT benchmark integration."""

    def test_create_scientific_dataset(self):
        """Test RAFT scientific dataset creation."""
        dataset = RAFTBenchmark.create_domain_dataset("scientific")

        assert "raft" in dataset.name
        assert dataset.benchmark_type == BenchmarkType.RAFT_DOMAIN
        assert len(dataset.tasks) > 0

    def test_create_generic_dataset(self):
        """Test RAFT generic domain dataset creation."""
        dataset = RAFTBenchmark.create_domain_dataset("banking")

        assert "banking" in dataset.name
        assert dataset.benchmark_type == BenchmarkType.RAFT_DOMAIN

    @pytest.mark.asyncio
    async def test_raft_benchmark_execution(self):
        """Test RAFT benchmark can be executed."""
        system = MockRetrievalSystem()
        runner = BenchmarkRunner(retrieval_system=system)
        dataset = RAFTBenchmark.create_domain_dataset("scientific")

        result = await runner.run_benchmark(dataset)

        assert result.status == BenchmarkStatus.COMPLETED
        assert result.benchmark_type == BenchmarkType.RAFT_DOMAIN


# =============================================================================
# HoloLoom Benchmark Tests
# =============================================================================

class TestHoloLoomBenchmark:
    """Tests for HoloLoom custom benchmark."""

    def test_create_memory_dataset(self):
        """Test HoloLoom memory-augmented dataset creation."""
        dataset = HoloLoomBenchmark.create_memory_augmented_dataset()

        assert "hololoom" in dataset.name
        assert dataset.benchmark_type == BenchmarkType.HOLOLOOM_MEMORY
        assert len(dataset.tasks) > 0

    def test_hololoom_task_types(self):
        """Test HoloLoom benchmark has different task types."""
        dataset = HoloLoomBenchmark.create_memory_augmented_dataset()

        task_types = set()
        for task in dataset.tasks:
            if "type" in task.metadata:
                task_types.add(task.metadata["type"])

        # Should have multiple task types
        assert "multi_hop" in task_types
        assert "temporal" in task_types
        assert "cross_reference" in task_types

    @pytest.mark.asyncio
    async def test_hololoom_benchmark_execution(self):
        """Test HoloLoom benchmark can be executed."""
        system = MockRetrievalSystem()
        runner = BenchmarkRunner(retrieval_system=system)
        dataset = HoloLoomBenchmark.create_memory_augmented_dataset()

        result = await runner.run_benchmark(dataset)

        assert result.status == BenchmarkStatus.COMPLETED
        assert result.benchmark_type == BenchmarkType.HOLOLOOM_MEMORY


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_benchmark_runner(self):
        """Test runner factory function."""
        runner = create_benchmark_runner(verbose=True)

        assert runner is not None
        assert runner.verbose is True
        assert runner.k_values == [1, 3, 5, 10]

    def test_get_all_sample_datasets(self):
        """Test getting all sample datasets."""
        datasets = get_all_sample_datasets()

        assert len(datasets) == 3
        types = {d.benchmark_type for d in datasets}
        assert BenchmarkType.MTEB_RETRIEVAL in types
        assert BenchmarkType.RAFT_DOMAIN in types
        assert BenchmarkType.HOLOLOOM_MEMORY in types

    @pytest.mark.asyncio
    async def test_run_all_benchmarks(self):
        """Test running all benchmarks."""
        system = MockRetrievalSystem()
        results = await run_all_benchmarks(system, verbose=False)

        assert len(results) == 3
        for name, result in results.items():
            assert result.status == BenchmarkStatus.COMPLETED


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_corpus(self):
        """Test with empty corpus."""
        system = MockRetrievalSystem()
        dataset = BenchmarkDataset(
            name="empty",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test",
                    expected_docs=[],
                    relevance_scores={},
                ),
            ],
            corpus={},
        )

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset)

        # Should complete but with no results
        assert result.status == BenchmarkStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_expected_docs(self):
        """Test task with no expected documents."""
        system = MockRetrievalSystem()
        dataset = BenchmarkDataset(
            name="no_expected",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test",
                    expected_docs=[],
                    relevance_scores={},
                ),
            ],
            corpus={"doc1": "content"},
        )

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset)

        # Should handle gracefully
        assert result.status == BenchmarkStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_single_task(self):
        """Test with single task."""
        system = MockRetrievalSystem()
        dataset = BenchmarkDataset(
            name="single",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test",
                    expected_docs=["doc1"],
                    relevance_scores={"doc1": 3},
                ),
            ],
            corpus={"doc1": "content"},
        )

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset)

        assert result.status == BenchmarkStatus.COMPLETED
        assert len(result.task_results) == 1

    @pytest.mark.asyncio
    async def test_large_k_value(self):
        """Test with k larger than corpus."""
        system = MockRetrievalSystem()
        dataset = BenchmarkDataset(
            name="large_k",
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=[
                BenchmarkTask(
                    task_id="q1",
                    query="test",
                    expected_docs=["doc1"],
                    relevance_scores={"doc1": 3},
                ),
            ],
            corpus={"doc1": "a", "doc2": "b"},
        )

        runner = BenchmarkRunner(retrieval_system=system)
        result = await runner.run_benchmark(dataset, k=100)

        # Should handle without error
        assert result.status == BenchmarkStatus.COMPLETED
