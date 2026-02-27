"""
Model Extension Evaluation Framework
=====================================

Comprehensive evaluation framework for HoloLoom's model_extension module.
Implements standard IR metrics, confidence calibration, and learning effectiveness.

Phases:
- Phase 1: Retrieval Quality (NDCG, MRR, MAP, Precision@k, Recall@k)
- Phase 2: Confidence Calibration (ECE, MCE, Brier Score)
- Phase 3: Learning Effectiveness (Thompson Sampling, pattern discovery)
- Phase 4: End-to-End Integration Tests
- Phase 5: Benchmark Suite Integration (MTEB, RAFT)

Usage:
    from hololoom.model_extension.eval import (
        RetrievalEvaluator,
        CalibrationEvaluator,
        LearningEvaluator,
        RetrievalMetrics,
        CalibrationMetrics,
        LearningMetrics
    )

    # Evaluate retrieval quality
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(predictions, ground_truth, k_values=[1, 3, 5, 10])
    print(f"NDCG@10: {metrics.ndcg_at_k[10]:.4f}")

Author: HoloLoom Architecture Team
Date: December 2025
"""

from .retrieval_metrics import (
    RetrievalMetrics,
    RetrievalEvaluator,
    ndcg_at_k,
    mrr,
    map_score,
    precision_at_k,
    recall_at_k,
    hit_rate,
)

from .calibration_metrics import (
    CalibrationMetrics,
    CalibrationEvaluator,
    expected_calibration_error,
    maximum_calibration_error,
    brier_score,
    compute_calibration_curve,
)

from .learning_metrics import (
    LearningMetrics,
    LearningEvaluator,
    improvement_rate,
    thompson_sampling_regret,
    pattern_discovery_rate,
    convergence_speed,
    retention_score,
)

from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkResult,
    BenchmarkDataset,
    BenchmarkTask,
    BenchmarkType,
    BenchmarkStatus,
    TaskResult,
    RetrievalSystem,
    MTEBBenchmark,
    RAFTBenchmark,
    HoloLoomBenchmark,
    create_benchmark_runner,
    get_all_sample_datasets,
    run_all_benchmarks,
)

__all__ = [
    # Retrieval
    "RetrievalMetrics",
    "RetrievalEvaluator",
    "ndcg_at_k",
    "mrr",
    "map_score",
    "precision_at_k",
    "recall_at_k",
    "hit_rate",
    # Calibration
    "CalibrationMetrics",
    "CalibrationEvaluator",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "compute_calibration_curve",
    # Learning
    "LearningMetrics",
    "LearningEvaluator",
    "improvement_rate",
    "thompson_sampling_regret",
    "pattern_discovery_rate",
    "convergence_speed",
    "retention_score",
    # Benchmark Runner (Phase 5)
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkDataset",
    "BenchmarkTask",
    "BenchmarkType",
    "BenchmarkStatus",
    "TaskResult",
    "RetrievalSystem",
    "MTEBBenchmark",
    "RAFTBenchmark",
    "HoloLoomBenchmark",
    "create_benchmark_runner",
    "get_all_sample_datasets",
    "run_all_benchmarks",
]
