"""
Performance Benchmarks for Shuttle + WeavingOrchestrator Integration

Compares performance before/after Shuttle integration across all execution modes:
- BARE mode: Minimal processing
- FAST mode: Balanced processing
- FUSED mode: Full processing

Generates comparison report with:
- Latency (before/after)
- Overhead (absolute + percentage)
- Quality metrics (confidence)
- Thread selection metrics

Created: 2025-01-21
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime

# HoloLoom types
from hololoom.protocols.types import Query, MemoryShard
from hololoom.config import Config
from hololoom.memory.graph import KG, KGEdge


# ============================================================================
# Test Data
# ============================================================================

def create_test_shards() -> List[MemoryShard]:
    """Create test memory shards."""
    return [
        MemoryShard(
            id=f"shard_{i}",
            text=f"Thompson Sampling shard {i}: Content about Bayesian bandits and exploration.",
            metadata={'topic': 'thompson_sampling', 'confidence': 0.9 - i * 0.05}
        )
        for i in range(20)
    ]


def create_test_kg() -> KG:
    """Create test knowledge graph."""
    kg = KG()

    # Add comprehensive graph structure
    edges = [
        ("thompson_sampling", "bandit", "USES", 1.0),
        ("thompson_sampling", "bayesian", "IS_A", 1.0),
        ("bandit", "exploration", "LEADS_TO", 0.8),
        ("bandit", "exploitation", "LEADS_TO", 0.8),
        ("bayesian", "beta_distribution", "USES", 0.9),
        ("bayesian", "prior", "HAS", 0.85),
        ("exploration", "uncertainty", "LEADS_TO", 0.7),
        ("exploitation", "greedy", "IS_A", 0.75),
    ]

    for source, target, edge_type, weight in edges:
        kg.add_edges([KGEdge(source, target, edge_type, weight)])

    # Add descriptions
    for node in kg.G.nodes():
        kg.G.nodes[node]['description'] = f"Description of {node} with detailed content."

    return kg


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Runs performance benchmarks for Shuttle integration."""

    def __init__(self, n_runs: int = 10):
        """
        Args:
            n_runs: Number of runs per benchmark (for statistical significance)
        """
        self.n_runs = n_runs
        self.shards = create_test_shards()
        self.kg = create_test_kg()
        self.queries = [
            "What is Thompson Sampling?",
            "How does exploration work?",
            "Explain Bayesian inference",
            "What is a beta distribution?",
            "How do bandits balance exploration and exploitation?",
        ]

    async def benchmark_mode(
        self,
        config: Config,
        mode_name: str,
        enable_shuttle: bool
    ) -> Dict[str, Any]:
        """
        Benchmark a single execution mode.

        Args:
            config: HoloLoom config
            mode_name: Mode name (BARE/FAST/FUSED)
            enable_shuttle: Enable Shuttle integration

        Returns:
            Benchmark results dict
        """
        from hololoom.weaving_orchestrator import WeavingOrchestrator

        config.enable_shuttle = enable_shuttle

        latencies = []
        confidences = []
        thread_counts = []

        for _ in range(self.n_runs):
            for query_text in self.queries:
                async with WeavingOrchestrator(
                    cfg=config,
                    shards=self.shards,
                    yarn_graph=self.kg
                ) as orchestrator:

                    start = time.time()
                    query = Query(text=query_text)
                    spacetime = await orchestrator.weave(query)
                    latency = (time.time() - start) * 1000  # ms

                    latencies.append(latency)
                    confidences.append(spacetime.confidence)

                    # Extract thread count from metadata
                    metadata = spacetime.metadata or {}
                    # Try to get thread count from various sources
                    thread_count = len(spacetime.sources) if hasattr(spacetime, 'sources') else 0
                    thread_counts.append(thread_count)

        return {
            'mode': mode_name,
            'shuttle_enabled': enable_shuttle,
            'latency_mean': statistics.mean(latencies),
            'latency_median': statistics.median(latencies),
            'latency_p95': sorted(latencies)[int(len(latencies) * 0.95)],
            'latency_std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'confidence_mean': statistics.mean(confidences),
            'thread_count_mean': statistics.mean(thread_counts) if thread_counts else 0,
            'n_runs': self.n_runs * len(self.queries),
        }

    async def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run benchmarks for all modes (before/after)."""
        results = []

        modes = [
            (Config.bare(), "BARE"),
            (Config.fast(), "FAST"),
            (Config.fused(), "FUSED"),
        ]

        print(f"\n{'='*80}")
        print(f"Shuttle Integration Performance Benchmarks")
        print(f"{'='*80}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runs per mode: {self.n_runs} × {len(self.queries)} queries = {self.n_runs * len(self.queries)} total")
        print(f"{'='*80}\n")

        for config, mode_name in modes:
            print(f"Benchmarking {mode_name} mode...")

            # Before (Shuttle disabled)
            print(f"  Running {mode_name} WITHOUT Shuttle...")
            before = await self.benchmark_mode(config, mode_name, enable_shuttle=False)
            results.append(before)

            # After (Shuttle enabled)
            print(f"  Running {mode_name} WITH Shuttle...")
            after = await self.benchmark_mode(config, mode_name, enable_shuttle=True)
            results.append(after)

            # Calculate overhead
            overhead_ms = after['latency_mean'] - before['latency_mean']
            overhead_pct = (overhead_ms / before['latency_mean']) * 100

            print(f"  ✓ {mode_name} complete")
            print(f"    Before: {before['latency_mean']:.1f}ms")
            print(f"    After:  {after['latency_mean']:.1f}ms")
            print(f"    Overhead: +{overhead_ms:.1f}ms ({overhead_pct:.1f}%)")
            print()

        return results

    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate markdown comparison report."""
        report = []

        report.append("# Shuttle Integration Performance Report\n")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Runs per mode**: {self.n_runs} × {len(self.queries)} queries = {self.n_runs * len(self.queries)} total\n\n")

        # Summary Table
        report.append("## Summary\n\n")
        report.append("| Mode | Before (ms) | After (ms) | Overhead | Change |\n")
        report.append("|------|-------------|------------|----------|--------|\n")

        # Group results by mode
        for i in range(0, len(results), 2):
            before = results[i]
            after = results[i + 1]

            overhead_ms = after['latency_mean'] - before['latency_mean']
            overhead_pct = (overhead_ms / before['latency_mean']) * 100

            report.append(
                f"| **{before['mode']}** | "
                f"{before['latency_mean']:.1f} | "
                f"{after['latency_mean']:.1f} | "
                f"+{overhead_ms:.1f}ms | "
                f"+{overhead_pct:.1f}% |\n"
            )

        # Detailed Results
        report.append("\n## Detailed Results\n\n")

        for i in range(0, len(results), 2):
            before = results[i]
            after = results[i + 1]
            mode = before['mode']

            report.append(f"### {mode} Mode\n\n")

            # Latency comparison
            report.append("**Latency (ms)**:\n\n")
            report.append("| Metric | Before | After | Change |\n")
            report.append("|--------|--------|-------|--------|\n")

            for metric in ['mean', 'median', 'p95', 'std']:
                key = f'latency_{metric}'
                before_val = before[key]
                after_val = after[key]
                change = after_val - before_val

                report.append(
                    f"| {metric.upper()} | "
                    f"{before_val:.1f} | "
                    f"{after_val:.1f} | "
                    f"{change:+.1f} |\n"
                )

            # Quality metrics
            report.append("\n**Quality Metrics**:\n\n")
            report.append("| Metric | Before | After |\n")
            report.append("|--------|--------|-------|\n")
            report.append(
                f"| Confidence | "
                f"{before['confidence_mean']:.3f} | "
                f"{after['confidence_mean']:.3f} |\n"
            )
            report.append(
                f"| Threads | "
                f"{before['thread_count_mean']:.1f} | "
                f"{after['thread_count_mean']:.1f} |\n"
            )

            report.append("\n")

        # Conclusions
        report.append("## Conclusions\n\n")
        report.append("**Expected Performance Impact**:\n\n")
        report.append("- **BARE**: Lightweight MCTS (8 sims, 1 depth) adds ~15-30ms\n")
        report.append("- **FAST**: Balanced MCTS (16 sims, 1 depth) adds ~35-50ms\n")
        report.append("- **FUSED**: Full MCTS (32 sims, 2 depth) adds ~75-100ms\n\n")

        report.append("**Quality Trade-off**:\n\n")
        report.append("- Shuttle combines semantic (Warp) + structural (Yarn) context\n")
        report.append("- MCTS explores non-obvious but relevant connections\n")
        report.append("- Thompson Sampling learns optimal trajectories over time\n")
        report.append("- Expected: Slightly higher latency, significantly better context quality\n\n")

        return "".join(report)


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all benchmarks and generate report."""
    runner = BenchmarkRunner(n_runs=5)  # 5 runs × 5 queries = 25 samples per mode

    results = await runner.run_all_benchmarks()
    report = runner.generate_report(results)

    # Print report
    print("\n" + "="*80 + "\n")
    print(report)

    # Save report
    output_path = "HoloLoom/shuttle/benchmarks/shuttle_performance_report.md"
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
