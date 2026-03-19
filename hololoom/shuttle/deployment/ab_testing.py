"""
A/B Testing Script for Shuttle Integration

Implements 50/50 traffic split between Shuttle and legacy thread selection
to validate performance and quality improvements in development/staging.

Usage:
    python -m HoloLoom.shuttle.deployment.ab_testing --duration 3600 --output ab_results.json

Features:
- 50% traffic to Shuttle (test group)
- 50% traffic to legacy Yarn Graph (control group)
- Real-time metrics collection
- Statistical analysis (t-test for latency, confidence)
- Automatic report generation

Created: 2025-01-21
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# HoloLoom
from hololoom.protocols.types import MemoryShard, Query

from hololoom.config import Config

# ============================================================================
# A/B Test Configuration
# ============================================================================

class ABTestConfig:
    """Configuration for A/B test."""

    def __init__(
        self,
        duration_seconds: int = 3600,
        traffic_split: float = 0.5,
        sample_size: int = 100,
        output_path: str = "ab_test_results.json"
    ):
        """
        Args:
            duration_seconds: Test duration (default: 1 hour)
            traffic_split: Percentage to Shuttle (default: 0.5 = 50%)
            sample_size: Minimum samples per group
            output_path: Results output path
        """
        self.duration_seconds = duration_seconds
        self.traffic_split = traffic_split
        self.sample_size = sample_size
        self.output_path = output_path


# ============================================================================
# A/B Test Runner
# ============================================================================

class ABTestRunner:
    """Runs A/B test comparing Shuttle vs legacy."""

    def __init__(self, config: ABTestConfig):
        """
        Args:
            config: A/B test configuration
        """
        self.config = config

        # Metrics storage
        self.control_metrics: list[dict[str, Any]] = []  # Legacy
        self.test_metrics: list[dict[str, Any]] = []     # Shuttle

        # Test queries
        self.queries = [
            "What is Thompson Sampling?",
            "How does MCTS work?",
            "Explain Bayesian inference",
            "What is a beta distribution?",
            "How do bandits balance exploration?",
            "What is UCB1?",
            "Explain knowledge graphs",
            "What is semantic search?",
            "How does vector similarity work?",
            "What is Qdrant?",
        ]

    async def run_single_query(
        self,
        hololoom_config: Config,
        enable_shuttle: bool,
        query_text: str
    ) -> dict[str, Any]:
        """
        Run a single query and collect metrics.

        Args:
            hololoom_config: HoloLoom configuration
            enable_shuttle: Enable Shuttle or use legacy
            query_text: Query text

        Returns:
            Metrics dict
        """
        from hololoom.weaving_orchestrator import WeavingOrchestrator

        # Configure Shuttle
        hololoom_config.enable_shuttle = enable_shuttle

        # Create test data (simplified for demo)
        shards = [
            MemoryShard(
                id=f"shard_{i}",
                text=f"Content about {query_text} - shard {i}",
                metadata={'score': 0.9 - i * 0.05}
            )
            for i in range(10)
        ]

        # Run weaving
        start = time.time()

        async with WeavingOrchestrator(cfg=hololoom_config, shards=shards) as orchestrator:
            query = Query(text=query_text)
            spacetime = await orchestrator.weave(query)

        latency = (time.time() - start) * 1000  # ms

        # Extract metrics
        return {
            'query': query_text,
            'latency_ms': latency,
            'confidence': spacetime.confidence,
            'shuttle_enabled': enable_shuttle,
            'timestamp': datetime.now().isoformat(),
            'thread_count': len(getattr(spacetime, 'sources', [])),
        }

    async def run_test(self):
        """Run complete A/B test."""
        print(f"\n{'='*80}")
        print("Shuttle A/B Test")
        print(f"{'='*80}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {self.config.duration_seconds}s")
        print(f"Traffic split: {self.config.traffic_split*100:.0f}% Shuttle, {(1-self.config.traffic_split)*100:.0f}% Legacy")
        print(f"Target samples: {self.config.sample_size} per group")
        print(f"{'='*80}\n")

        start_time = time.time()
        query_count = 0

        # Test both modes
        test_config = Config.fast()
        control_config = Config.fast()

        while time.time() - start_time < self.config.duration_seconds:
            # Randomly assign to test or control
            use_shuttle = random.random() < self.config.traffic_split

            # Select random query
            query_text = random.choice(self.queries)

            # Run query
            try:
                if use_shuttle:
                    metrics = await self.run_single_query(test_config, True, query_text)
                    self.test_metrics.append(metrics)
                    group = "Shuttle"
                else:
                    metrics = await self.run_single_query(control_config, False, query_text)
                    self.control_metrics.append(metrics)
                    group = "Legacy"

                query_count += 1

                # Progress update
                if query_count % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = self.config.duration_seconds - elapsed
                    print(
                        f"[{elapsed:.0f}s] Queries: {query_count} "
                        f"(Shuttle: {len(self.test_metrics)}, Legacy: {len(self.control_metrics)}) "
                        f"| Remaining: {remaining:.0f}s"
                    )

            except Exception as e:
                print(f"Error running query: {e}")

            # Check if we have enough samples
            if (len(self.test_metrics) >= self.config.sample_size and
                len(self.control_metrics) >= self.config.sample_size):
                print(f"\n✓ Reached target sample size ({self.config.sample_size} per group)")
                break

            # Small delay to avoid overwhelming system
            await asyncio.sleep(0.1)

        print(f"\n{'='*80}")
        print("Test Complete!")
        print(f"Total queries: {query_count}")
        print(f"Shuttle queries: {len(self.test_metrics)}")
        print(f"Legacy queries: {len(self.control_metrics)}")
        print(f"{'='*80}\n")

    def analyze_results(self) -> dict[str, Any]:
        """Analyze A/B test results."""

        # Extract metrics
        control_latencies = [m['latency_ms'] for m in self.control_metrics]
        test_latencies = [m['latency_ms'] for m in self.test_metrics]

        control_confidences = [m['confidence'] for m in self.control_metrics]
        test_confidences = [m['confidence'] for m in self.test_metrics]

        # Statistical analysis
        results = {
            'test_info': {
                'start_time': self.control_metrics[0]['timestamp'] if self.control_metrics else None,
                'end_time': self.test_metrics[-1]['timestamp'] if self.test_metrics else None,
                'total_queries': len(self.control_metrics) + len(self.test_metrics),
                'control_samples': len(self.control_metrics),
                'test_samples': len(self.test_metrics),
            },
            'latency': {
                'control_mean': statistics.mean(control_latencies),
                'control_median': statistics.median(control_latencies),
                'control_p95': sorted(control_latencies)[int(len(control_latencies) * 0.95)],
                'control_std': statistics.stdev(control_latencies),
                'test_mean': statistics.mean(test_latencies),
                'test_median': statistics.median(test_latencies),
                'test_p95': sorted(test_latencies)[int(len(test_latencies) * 0.95)],
                'test_std': statistics.stdev(test_latencies),
                'delta_mean': statistics.mean(test_latencies) - statistics.mean(control_latencies),
                'delta_pct': ((statistics.mean(test_latencies) / statistics.mean(control_latencies)) - 1) * 100,
            },
            'confidence': {
                'control_mean': statistics.mean(control_confidences),
                'control_std': statistics.stdev(control_confidences),
                'test_mean': statistics.mean(test_confidences),
                'test_std': statistics.stdev(test_confidences),
                'delta': statistics.mean(test_confidences) - statistics.mean(control_confidences),
            },
        }

        # T-test for statistical significance (simplified)
        # Note: For production, use scipy.stats.ttest_ind
        pooled_std = ((results['latency']['control_std']**2 + results['latency']['test_std']**2) / 2) ** 0.5
        t_stat = results['latency']['delta_mean'] / (pooled_std * ((1/len(control_latencies) + 1/len(test_latencies)) ** 0.5))

        results['significance'] = {
            't_statistic': t_stat,
            'significant': abs(t_stat) > 1.96,  # p < 0.05 (two-tailed)
            'interpretation': 'Statistically significant' if abs(t_stat) > 1.96 else 'Not statistically significant'
        }

        return results

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate markdown report."""
        report = []

        report.append("# Shuttle A/B Test Results\n\n")
        report.append(f"**Start**: {results['test_info']['start_time']}\n")
        report.append(f"**End**: {results['test_info']['end_time']}\n")
        report.append(f"**Total Queries**: {results['test_info']['total_queries']}\n\n")

        # Latency comparison
        report.append("## Latency Comparison\n\n")
        report.append("| Metric | Legacy (Control) | Shuttle (Test) | Delta |\n")
        report.append("|--------|------------------|----------------|-------|\n")

        lat = results['latency']
        report.append(f"| Mean | {lat['control_mean']:.1f}ms | {lat['test_mean']:.1f}ms | +{lat['delta_mean']:.1f}ms ({lat['delta_pct']:+.1f}%) |\n")
        report.append(f"| Median | {lat['control_median']:.1f}ms | {lat['test_median']:.1f}ms | - |\n")
        report.append(f"| P95 | {lat['control_p95']:.1f}ms | {lat['test_p95']:.1f}ms | - |\n")
        report.append(f"| Std Dev | {lat['control_std']:.1f}ms | {lat['test_std']:.1f}ms | - |\n\n")

        # Quality comparison
        report.append("## Quality Comparison\n\n")
        report.append("| Metric | Legacy (Control) | Shuttle (Test) | Delta |\n")
        report.append("|--------|------------------|----------------|-------|\n")

        conf = results['confidence']
        report.append(f"| Confidence | {conf['control_mean']:.3f} | {conf['test_mean']:.3f} | {conf['delta']:+.3f} |\n\n")

        # Statistical significance
        report.append("## Statistical Significance\n\n")
        sig = results['significance']
        report.append(f"- **T-statistic**: {sig['t_statistic']:.2f}\n")
        report.append(f"- **Significant**: {sig['significant']} (p < 0.05)\n")
        report.append(f"- **Interpretation**: {sig['interpretation']}\n\n")

        # Recommendation
        report.append("## Recommendation\n\n")

        if sig['significant'] and lat['delta_pct'] < 50:  # Less than 50% overhead
            report.append("✅ **PROCEED TO PRODUCTION**\n\n")
            report.append("The Shuttle shows statistically significant differences with acceptable overhead. ")
            report.append("Recommend gradual rollout starting at 10% traffic.\n\n")
        elif not sig['significant']:
            report.append("🟡 **CONTINUE TESTING**\n\n")
            report.append("Differences are not statistically significant. Collect more samples or adjust test parameters.\n\n")
        else:
            report.append("🔴 **DO NOT PROCEED**\n\n")
            report.append(f"Shuttle overhead is too high ({lat['delta_pct']:.1f}%). Investigate performance issues.\n\n")

        return "".join(report)

    def save_results(self, results: dict[str, Any]):
        """Save results to JSON."""
        output = {
            'results': results,
            'control_metrics': self.control_metrics,
            'test_metrics': self.test_metrics,
        }

        with open(self.config.output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Results saved to: {self.config.output_path}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run A/B test."""
    parser = argparse.ArgumentParser(description="Shuttle A/B Test")
    parser.add_argument('--duration', type=int, default=300, help='Test duration (seconds)')
    parser.add_argument('--split', type=float, default=0.5, help='Traffic to Shuttle (0-1)')
    parser.add_argument('--samples', type=int, default=50, help='Minimum samples per group')
    parser.add_argument('--output', type=str, default='ab_test_results.json', help='Output file')

    args = parser.parse_args()

    # Create config
    config = ABTestConfig(
        duration_seconds=args.duration,
        traffic_split=args.split,
        sample_size=args.samples,
        output_path=args.output
    )

    # Run test
    runner = ABTestRunner(config)
    await runner.run_test()

    # Analyze results
    results = runner.analyze_results()

    # Generate report
    report = runner.generate_report(results)
    print(report)

    # Save results
    runner.save_results(results)

    # Save report
    report_path = Path(args.output).with_suffix('.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
