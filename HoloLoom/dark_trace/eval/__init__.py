"""
Dark Trace Evaluation: Rigorous Interpretability Validation

This module provides comprehensive evaluation infrastructure for validating
interpretability claims with rigorous metrics, benchmarks, and baselines.

Key Capabilities:
- Reconstruction quality metrics (MSE, cosine, explained variance)
- Sparsity metrics (L0, L1, entropy, dead feature %)
- Interpretability scoring (monosemanticity, polysemanticity)
- Steering effectiveness validation
- Causal validity testing
- Synthetic benchmarks with ground truth
- Baseline comparisons (random, PCA, supervised probes)
- Publication-ready evaluation reports

Research Basis:
- Scaling Monosemanticity (Anthropic 2024): SAE evaluation methodology
- Towards Monosemanticity (Anthropic 2023): Feature quality metrics
- ACDC (Conmy et al., 2023): Circuit evaluation
- Interpretability benchmarking best practices

Usage:
    from HoloLoom.dark_trace.eval import (
        # Metrics
        SAEMetrics,
        ReconstructionMetrics,
        SparsityMetrics,
        InterpretabilityMetrics,
        SteeringMetrics,
        CausalMetrics,
        compute_all_metrics,
        # Benchmarks
        SyntheticBenchmark,
        FeatureRecoveryBenchmark,
        InterventionBenchmark,
        BenchmarkSuite,
        BenchmarkResult,
        # Baselines
        RandomBaseline,
        PCABaseline,
        SupervisedProbeBaseline,
        BaselineComparison,
        # Reporting
        EvaluationReporter,
        EvaluationReport,
        MetricsDashboard,
        RegressionDetector,
    )

    # Compute SAE metrics
    metrics = SAEMetrics(sae, test_data)
    report = metrics.compute_all()
    print(f"Reconstruction MSE: {report.reconstruction.mse:.4f}")
    print(f"Sparsity L0: {report.sparsity.l0:.2f}")
    print(f"Dead features: {report.sparsity.dead_feature_ratio:.2%}")

    # Run benchmarks
    benchmark = SyntheticBenchmark(n_features=100, n_samples=1000)
    result = benchmark.evaluate(sae)
    print(f"Feature recovery: {result.recovery_rate:.2%}")

    # Compare to baselines
    comparison = BaselineComparison(sae, test_data)
    comparison.add_baseline(PCABaseline(n_components=100))
    comparison.add_baseline(RandomBaseline(n_features=100))
    results = comparison.evaluate()
    print(f"SAE vs PCA: {results['sae_vs_pca']}")

    # Generate report
    reporter = EvaluationReporter()
    report = reporter.generate(metrics, benchmark_results, comparison)
    report.save("evaluation_report.html")
"""

# Core metrics
from HoloLoom.dark_trace.eval.metrics import (
    SAEMetrics,
    ReconstructionMetrics,
    SparsityMetrics,
    InterpretabilityMetrics,
    SteeringMetrics,
    CausalMetrics,
    MetricsConfig,
    MetricsReport,
    compute_all_metrics,
)

# Benchmarks
from HoloLoom.dark_trace.eval.benchmarks import (
    SyntheticBenchmark,
    FeatureRecoveryBenchmark,
    InterventionBenchmark,
    ConsistencyBenchmark,
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkConfig,
)

# Baselines
from HoloLoom.dark_trace.eval.baselines import (
    BaselineMethod,
    RandomBaseline,
    PCABaseline,
    SupervisedProbeBaseline,
    MeanBaseline,
    BaselineComparison,
    BaselineResult,
)

# Reporting
from HoloLoom.dark_trace.eval.reporter import (
    EvaluationReporter,
    EvaluationReport,
    MetricsDashboard,
    RegressionDetector,
    ReportConfig,
    ReportFormat,
)

__all__ = [
    # Metrics
    "SAEMetrics",
    "ReconstructionMetrics",
    "SparsityMetrics",
    "InterpretabilityMetrics",
    "SteeringMetrics",
    "CausalMetrics",
    "MetricsConfig",
    "MetricsReport",
    "compute_all_metrics",
    # Benchmarks
    "SyntheticBenchmark",
    "FeatureRecoveryBenchmark",
    "InterventionBenchmark",
    "ConsistencyBenchmark",
    "BenchmarkSuite",
    "BenchmarkResult",
    "BenchmarkConfig",
    # Baselines
    "BaselineMethod",
    "RandomBaseline",
    "PCABaseline",
    "SupervisedProbeBaseline",
    "MeanBaseline",
    "BaselineComparison",
    "BaselineResult",
    # Reporting
    "EvaluationReporter",
    "EvaluationReport",
    "MetricsDashboard",
    "RegressionDetector",
    "ReportConfig",
    "ReportFormat",
]
