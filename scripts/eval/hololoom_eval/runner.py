"""
Main evaluation runner - orchestrates all benchmarks.

v2.0.0 (January 2026): Added stress/semantic benchmarks, statistical rigor,
reproducibility infrastructure, and HTML dashboard generation.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum


class EvalMode(Enum):
    QUICK = "quick"      # ~1 minute, basic sanity checks
    STANDARD = "standard"  # ~5 minutes, thorough testing
    FULL = "full"        # ~10+ minutes, comprehensive


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    passed: bool
    score: float  # 0.0 - 1.0, higher is better
    verdict: str  # Human-readable verdict
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    mode: str
    duration_seconds: float
    overall_verdict: str
    overall_score: float
    overall_passed: bool = True
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    dependencies: Dict[str, bool] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    statistical_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "duration_seconds": self.duration_seconds,
            "overall_verdict": self.overall_verdict,
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "benchmarks": [asdict(b) for b in self.benchmarks],
            "dependencies": self.dependencies,
            "system_info": self.system_info,
            "recommendations": self.recommendations,
        }
        if self.statistical_summary:
            d["statistical_summary"] = self.statistical_summary
        return d


class EvaluationRunner:
    """Main evaluation orchestrator."""

    # Benchmark weights for overall score (v2.0 - 8 benchmarks)
    BENCHMARK_WEIGHTS = {
        "Retrieval Quality": 0.25,
        "Semantic Quality": 0.12,
        "Stress Robustness": 0.08,
        "Learning Effectiveness": 0.12,
        "Feature Ablation": 0.12,
        "Latency": 0.12,
        "Knowledge Graph Value": 0.10,
        "Cache Effectiveness": 0.09,
    }

    def __init__(
        self,
        mode: EvalMode = EvalMode.STANDARD,
        output_dir: str = "scripts/eval/results",
        seed: Optional[int] = None,
        generate_dashboard: bool = False,
        compare_previous: bool = False,
    ):
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.seed = seed
        self.generate_dashboard = generate_dashboard
        self.compare_previous = compare_previous

        # Set global seed for reproducibility
        if seed is not None:
            from .reproducibility import set_global_seed
            set_global_seed(seed)

    def check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are available."""
        deps = {}

        # Core Python
        try:
            import numpy
            deps["numpy"] = True
        except ImportError:
            deps["numpy"] = False

        # Embeddings
        try:
            from sentence_transformers import SentenceTransformer
            deps["sentence_transformers"] = True
        except ImportError:
            deps["sentence_transformers"] = False

        # PyTorch
        try:
            import torch
            deps["torch"] = True
        except ImportError:
            deps["torch"] = False

        # HoloLoom
        try:
            from HoloLoom import HoloLoom
            deps["hololoom"] = True
        except ImportError:
            deps["hololoom"] = False

        # NetworkX (for graphs)
        try:
            import networkx
            deps["networkx"] = True
        except ImportError:
            deps["networkx"] = False

        return deps

    async def run_benchmark(self, name: str, benchmark_fn) -> BenchmarkResult:
        """Run a single benchmark with error handling."""
        start = time.perf_counter()
        try:
            result = await benchmark_fn()
            result.duration_seconds = time.perf_counter() - start
            return result
        except Exception as e:
            return BenchmarkResult(
                name=name,
                passed=False,
                score=0.0,
                verdict=f"Error: {str(e)}",
                error=str(e),
                duration_seconds=time.perf_counter() - start
            )

    async def run_all(self) -> EvaluationReport:
        """Run all benchmarks and generate report."""
        start_time = time.perf_counter()
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("HOLOLOOM COMPREHENSIVE EVALUATION v2.0")
        print("=" * 70)
        print(f"Mode: {self.mode.value}")
        if self.seed is not None:
            print(f"Seed: {self.seed}")
        print(f"Started: {timestamp}")
        print()

        # Check dependencies
        print("📦 Checking dependencies...")
        deps = self.check_dependencies()
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            print(f"   {status} {dep}")
        print()

        missing_critical = []
        if not deps.get("numpy"):
            missing_critical.append("numpy")
        if not deps.get("sentence_transformers"):
            missing_critical.append("sentence-transformers")

        if missing_critical:
            print(f"⚠️  Missing critical dependencies: {', '.join(missing_critical)}")
            print("   Install with: pip install " + " ".join(missing_critical))
            print()

        # Import benchmarks
        from .benchmarks import (
            run_retrieval_benchmark,
            run_learning_benchmark,
            run_ablation_benchmark,
            run_latency_benchmark,
            run_graph_benchmark,
            run_cache_benchmark,
            run_stress_benchmark,
            run_semantic_benchmark,
        )

        # Define benchmark suite based on mode
        # QUICK: core benchmarks only
        benchmarks = [
            ("Retrieval Quality", lambda: run_retrieval_benchmark(self.mode, deps)),
            ("Semantic Quality", lambda: run_semantic_benchmark(self.mode, deps)),
            ("Cache Effectiveness", lambda: run_cache_benchmark(self.mode, deps)),
            ("Latency", lambda: run_latency_benchmark(self.mode, deps)),
        ]

        if self.mode in [EvalMode.STANDARD, EvalMode.FULL]:
            benchmarks.extend([
                ("Knowledge Graph Value", lambda: run_graph_benchmark(self.mode, deps)),
                ("Feature Ablation", lambda: run_ablation_benchmark(self.mode, deps)),
                ("Stress Robustness", lambda: run_stress_benchmark(self.mode, deps)),
            ])

        if self.mode == EvalMode.FULL:
            benchmarks.extend([
                ("Learning Effectiveness", lambda: run_learning_benchmark(self.mode, deps)),
            ])

        # Run benchmarks
        print("🔬 Running benchmarks...")
        print("-" * 70)

        for name, benchmark_fn in benchmarks:
            print(f"\n   [{name}]")
            result = await self.run_benchmark(name, benchmark_fn)
            self.results.append(result)

            if result.error:
                print(f"   ❌ Error: {result.error}")
            else:
                status = "✅" if result.passed else "❌"
                print(f"   {status} Score: {result.score:.2f} | {result.verdict}")
                print(f"      Duration: {result.duration_seconds:.1f}s")

        # Generate report
        total_duration = time.perf_counter() - start_time
        report = self._generate_report(timestamp, total_duration, deps)

        # Add statistical summary
        report.statistical_summary = self._compute_statistical_summary()

        # Print summary
        self._print_summary(report)

        # Save results
        self._save_results(report)

        # Reproducibility: save with metadata
        if self.seed is not None:
            self._save_with_reproducibility(report, deps)

        # HTML dashboard
        if self.generate_dashboard:
            self._generate_html_dashboard(report)

        # Compare with previous run
        if self.compare_previous:
            self._compare_with_previous(report)

        return report

    def _generate_report(self, timestamp: str, duration: float, deps: Dict[str, bool]) -> EvaluationReport:
        """Generate comprehensive report."""
        # Calculate overall score (weighted average)
        total_weight = 0
        weighted_score = 0
        for result in self.results:
            weight = self.BENCHMARK_WEIGHTS.get(result.name, 0.1)
            if not result.error:
                weighted_score += result.score * weight
                total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0

        # Determine verdict
        passed_count = sum(1 for r in self.results if r.passed and not r.error)
        total_count = len(self.results)

        if overall_score >= 0.8 and passed_count == total_count:
            overall_verdict = "✅ EXCELLENT - HoloLoom provides significant value"
        elif overall_score >= 0.6 and passed_count >= total_count * 0.7:
            overall_verdict = "🟡 GOOD - HoloLoom provides value with some caveats"
        elif overall_score >= 0.4:
            overall_verdict = "🟠 MARGINAL - HoloLoom provides limited value over baselines"
        else:
            overall_verdict = "❌ POOR - HoloLoom may not justify its complexity"

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Determine overall pass/fail
        overall_passed = self._determine_overall_passed(overall_score, self.results)

        return EvaluationReport(
            timestamp=timestamp,
            mode=self.mode.value,
            duration_seconds=duration,
            overall_verdict=overall_verdict,
            overall_score=overall_score,
            overall_passed=overall_passed,
            benchmarks=self.results,
            dependencies=deps,
            system_info={"dependencies": deps},
            recommendations=recommendations,
        )

    def _compute_statistical_summary(self) -> Dict[str, Any]:
        """Compute statistical summary with confidence intervals."""
        from .stats import compute_metric_summary, bootstrap_ci

        scores = [r.score for r in self.results if not r.error]
        if not scores:
            return {"note": "No valid benchmark scores for statistical analysis"}

        summary = compute_metric_summary(
            "overall_scores",
            scores,
            ci_confidence=0.95,
            ci_bootstrap=1000,
            seed=self.seed,
        )

        result = {
            "n_benchmarks": len(scores),
            "mean_score": round(summary.mean, 4),
            "median_score": round(summary.median, 4),
            "std_score": round(summary.std, 4),
            "min_score": round(summary.min_val, 4),
            "max_score": round(summary.max_val, 4),
        }

        if summary.ci:
            result["confidence_interval_95"] = {
                "lower": round(summary.ci.lower, 4),
                "upper": round(summary.ci.upper, 4),
                "width": round(summary.ci.width, 4),
            }

        return result

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        for result in self.results:
            if result.error:
                if "numpy" in result.error.lower():
                    recommendations.append("Install numpy: pip install numpy")
                elif "sentence" in result.error.lower():
                    recommendations.append("Install embeddings: pip install sentence-transformers")
                continue

            if result.name == "Retrieval Quality" and result.score < 0.6:
                recommendations.append(
                    "Retrieval quality is low. Consider: "
                    "(1) Adding more relevant documents, "
                    "(2) Tuning similarity thresholds, "
                    "(3) Using domain-specific embeddings"
                )

            if result.name == "Cache Effectiveness" and result.score < 0.5:
                recommendations.append(
                    "Cache is underutilized. Check if queries are being cached properly."
                )

            if result.name == "Latency" and result.score < 0.6:
                recommendations.append(
                    "Latency is high. Consider: "
                    "(1) Using FAST mode instead of FUSED, "
                    "(2) Reducing retrieval depth, "
                    "(3) Enabling query caching"
                )

            if result.name == "Knowledge Graph Value" and result.score < 0.5:
                recommendations.append(
                    "Knowledge graph provides limited value. Your data may not have "
                    "meaningful entity relationships. Consider simpler hybrid retrieval."
                )

            if result.name == "Learning Effectiveness" and result.score < 0.5:
                recommendations.append(
                    "Learning loops show limited improvement. This is expected for "
                    "small query volumes. Value emerges at scale (1000+ queries)."
                )

            if result.name == "Stress Robustness" and result.score < 0.6:
                recommendations.append(
                    "Stress robustness is low. The system may not handle edge cases "
                    "(empty queries, adversarial inputs, unicode) gracefully."
                )

            if result.name == "Semantic Quality" and result.score < 0.5:
                recommendations.append(
                    "Semantic quality is weak. Consider: "
                    "(1) Installing sentence-transformers for vector search, "
                    "(2) Using hybrid retrieval instead of BM25-only"
                )

        if not recommendations:
            recommendations.append("All benchmarks passed. HoloLoom is working well for your use case.")

        return recommendations

    def _determine_overall_passed(self, report_score: float, results: List[BenchmarkResult]) -> bool:
        """Determine if overall evaluation passed."""
        passed_count = sum(1 for r in results if r.passed and not r.error)
        total_count = len(results)

        # Pass if score >= 0.5 and majority of benchmarks passed
        return report_score >= 0.5 and passed_count >= total_count * 0.5

    def _print_summary(self, report: EvaluationReport):
        """Print summary to console."""
        print()
        print("=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print()
        print(f"Overall Score: {report.overall_score:.2f}/1.00")
        print(f"Verdict: {report.overall_verdict}")
        print(f"Duration: {report.duration_seconds:.1f}s")

        if report.statistical_summary and "confidence_interval_95" in report.statistical_summary:
            ci = report.statistical_summary["confidence_interval_95"]
            print(f"95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

        print()

        print("Benchmark Results:")
        print("-" * 70)
        for result in report.benchmarks:
            status = "✅" if result.passed else "❌"
            if result.error:
                status = "⚠️"
            print(f"  {status} {result.name:<30} Score: {result.score:.2f}")

        print()
        print("Recommendations:")
        print("-" * 70)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print()

    def _save_results(self, report: EvaluationReport):
        """Save results to files."""
        # JSON results
        json_path = self.output_dir / f"eval_report_{report.mode}.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"📄 JSON report: {json_path}")

        # Latest symlink
        latest_path = self.output_dir / "eval_report_latest.json"
        with open(latest_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    def _save_with_reproducibility(self, report: EvaluationReport, deps: Dict[str, bool]):
        """Save run with full reproducibility metadata."""
        try:
            from .reproducibility import capture_run_metadata, save_run_with_metadata
            from .benchmarks.datasets import get_corpus, get_queries_for_mode

            corpus = get_corpus()
            queries = get_queries_for_mode(self.mode.value)

            metadata = capture_run_metadata(
                mode=self.mode.value,
                seed=self.seed if self.seed is not None else 0,
                corpus=corpus,
                queries=queries,
            )

            run_path = save_run_with_metadata(
                report_dict=report.to_dict(),
                metadata=metadata,
                results_dir=str(self.output_dir),
            )
            print(f"📋 Reproducibility metadata saved: {run_path}")
        except Exception as e:
            print(f"⚠️  Could not save reproducibility metadata: {e}")

    def _generate_html_dashboard(self, report: EvaluationReport):
        """Generate HTML dashboard report."""
        try:
            from .dashboard import generate_dashboard

            dashboard_path = str(self.output_dir / f"eval_dashboard_{report.mode}.html")
            generate_dashboard(report.to_dict(), dashboard_path)
            print(f"📊 HTML dashboard: {dashboard_path}")
        except Exception as e:
            print(f"⚠️  Could not generate dashboard: {e}")

    def _compare_with_previous(self, report: EvaluationReport):
        """Compare current run with most recent previous run."""
        try:
            from .reproducibility import load_previous_run, compare_runs

            previous = load_previous_run(str(self.output_dir), self.mode.value)
            if previous is None:
                print("ℹ️  No previous run found for comparison.")
                return

            comparison = compare_runs(previous, report.to_dict())

            print()
            print("=" * 70)
            print("RUN COMPARISON")
            print("=" * 70)
            print(f"Score delta: {comparison.score_delta:+.3f}")
            print(f"Summary: {comparison.summary}")

            if comparison.improved:
                print(f"  Improved: {', '.join(comparison.improved)}")
            if comparison.regressed:
                print(f"  Regressed: {', '.join(comparison.regressed)}")
            if comparison.unchanged:
                print(f"  Unchanged: {', '.join(comparison.unchanged)}")
            if comparison.config_diffs:
                print(f"  Config changes: {', '.join(comparison.config_diffs)}")
            print()
        except Exception as e:
            print(f"⚠️  Could not compare with previous run: {e}")


    async def run_single_benchmark(self, benchmark_name: str) -> EvaluationReport:
        """Run a single specific benchmark."""
        start_time = time.perf_counter()
        timestamp = datetime.now().isoformat()

        # Check dependencies
        deps = self.check_dependencies()

        # Import benchmarks
        from .benchmarks import (
            run_retrieval_benchmark,
            run_learning_benchmark,
            run_ablation_benchmark,
            run_latency_benchmark,
            run_graph_benchmark,
            run_cache_benchmark,
            run_stress_benchmark,
            run_semantic_benchmark,
        )

        benchmark_map = {
            "retrieval": ("Retrieval Quality", lambda: run_retrieval_benchmark(self.mode, deps)),
            "learning": ("Learning Effectiveness", lambda: run_learning_benchmark(self.mode, deps)),
            "ablation": ("Feature Ablation", lambda: run_ablation_benchmark(self.mode, deps)),
            "latency": ("Latency", lambda: run_latency_benchmark(self.mode, deps)),
            "graph": ("Knowledge Graph Value", lambda: run_graph_benchmark(self.mode, deps)),
            "cache": ("Cache Effectiveness", lambda: run_cache_benchmark(self.mode, deps)),
            "stress": ("Stress Robustness", lambda: run_stress_benchmark(self.mode, deps)),
            "semantic": ("Semantic Quality", lambda: run_semantic_benchmark(self.mode, deps)),
        }

        if benchmark_name not in benchmark_map:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {', '.join(benchmark_map.keys())}")

        name, benchmark_fn = benchmark_map[benchmark_name]
        result = await self.run_benchmark(name, benchmark_fn)
        self.results.append(result)

        total_duration = time.perf_counter() - start_time
        return self._generate_report(timestamp, total_duration, deps)


async def run_evaluation(
    mode: str = "standard",
    output_dir: str = "scripts/eval/results",
    seed: Optional[int] = None,
    dashboard: bool = False,
    compare: bool = False,
) -> EvaluationReport:
    """Main entry point for running evaluation."""
    eval_mode = EvalMode(mode)
    runner = EvaluationRunner(
        mode=eval_mode,
        output_dir=output_dir,
        seed=seed,
        generate_dashboard=dashboard,
        compare_previous=compare,
    )
    return await runner.run_all()
