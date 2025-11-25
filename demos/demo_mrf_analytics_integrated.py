"""
MRF Analytics Integration Demo
==============================
Demonstrates the complete integrated analytics system:
- Real-time dashboard
- Thompson Sampling learning
- A/B testing framework
- Prometheus metrics export

**Phase 2 - November 2025**: Complete analytics pipeline demonstration.

Usage:
    PYTHONPATH=. python demos/demo_mrf_analytics_integrated.py
"""

import asyncio
from pathlib import Path
from HoloLoom.prompting.analytics import (
    create_dashboard,
    LEARNING_AVAILABLE,
    AB_TESTING_AVAILABLE
)

def demo_basic_dashboard():
    """Demo 1: Basic dashboard without learning/A/B testing."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Dashboard (No Learning)")
    print("=" * 70)

    # Create basic dashboard
    dashboard = create_dashboard()

    # Simulate MRF enhancements
    test_data = [
        ("agentic", "verify", 0.75, 0.92, 450.0),
        ("rag", "refine", 0.68, 0.85, 380.0),
        ("agentic", "critique", 0.70, 0.88, 420.0),
        ("alignment", "verify", 0.72, 0.89, 410.0),
        ("rag", "verify", 0.65, 0.82, 350.0),
    ]

    for system, strategy, before, after, time_ms in test_data:
        dashboard.log_enhancement(
            system=system,
            query=f"Test query for {system}",
            strategy=strategy,
            quality_before=before,
            quality_after=after,
            execution_time_ms=time_ms
        )

    # Get statistics
    stats = dashboard.get_statistics()
    print(f"\n✅ Total enhancements: {stats['total_enhancements']}")
    print(f"✅ Systems active: {stats['systems_active']}")
    print(f"✅ Avg quality improvement: +{stats['quality_improvement_percent']:.1f}%")
    print(f"✅ Avg execution time: {stats['avg_execution_time_ms']:.1f}ms")

    print("\nPer-system breakdown:")
    for system, data in stats["per_system"].items():
        print(f"  {system}: {data['total']} enhancements, +{data['avg_improvement']*100:.1f}% improvement")

    # Generate report
    dashboard.save_report("mrf_dashboard_basic.html", format="html")
    print(f"\n📊 Report saved: mrf_dashboard_basic.html")


def demo_learning_integration():
    """Demo 2: Dashboard with Thompson Sampling learning."""
    print("\n" + "=" * 70)
    print("DEMO 2: Dashboard with Thompson Sampling Learning")
    print("=" * 70)

    if not LEARNING_AVAILABLE:
        print("⚠️  Thompson Sampling learner not available")
        return

    # Create dashboard with learning enabled
    dashboard = create_dashboard(
        enable_learning=True
    )

    # Simulate enhancements with query_type metadata for learner
    test_data = [
        # Factual queries - verify works best
        ("agentic", "verify", 0.75, 0.92, 450.0, "factual"),
        ("agentic", "verify", 0.73, 0.90, 440.0, "factual"),
        ("agentic", "refine", 0.75, 0.82, 380.0, "factual"),

        # Analytical queries - critique works best
        ("agentic", "critique", 0.70, 0.91, 520.0, "analytical"),
        ("agentic", "critique", 0.72, 0.89, 510.0, "analytical"),
        ("agentic", "verify", 0.70, 0.78, 450.0, "analytical"),

        # Creative queries - elegance works best
        ("agentic", "elegance", 0.68, 0.88, 550.0, "creative"),
        ("agentic", "elegance", 0.70, 0.90, 540.0, "creative"),
        ("agentic", "refine", 0.68, 0.75, 400.0, "creative"),
    ]

    for system, strategy, before, after, time_ms, query_type in test_data:
        dashboard.log_enhancement(
            system=system,
            query=f"Test {query_type} query",
            strategy=strategy,
            quality_before=before,
            quality_after=after,
            execution_time_ms=time_ms,
            metadata={"query_type": query_type}
        )

    # Get strategy recommendations
    print("\n🧠 Strategy Recommendations:")
    for query_type in ["factual", "analytical", "creative"]:
        rec = dashboard.get_strategy_recommendation(
            query_type=query_type,
            system="agentic"
        )
        if rec:
            print(f"\n  {query_type.upper()} queries:")
            print(f"    Recommended: {rec['recommended_strategy']}")
            print(f"    Confidence: {rec['confidence']:.2f}")
            print(f"    Expected reward: {rec['expected_reward']:.2f}")
            print(f"    Success rate: {rec['success_rate']:.2f}")
            print(f"    Total uses: {rec['total_uses']}")

    # Get learning statistics
    learning_stats = dashboard.get_learning_statistics()
    if learning_stats:
        print(f"\n📈 Learning Statistics:")
        print(f"  Total queries: {learning_stats['total_queries']}")
        print(f"  Profiles tracked: {learning_stats['profiles_count']}")

        if learning_stats['strategy_effectiveness_ranking']:
            print("\n  Strategy Effectiveness Ranking:")
            for i, s in enumerate(learning_stats['strategy_effectiveness_ranking'][:3], 1):
                print(f"    {i}. {s['strategy']}: {s['success_rate']:.1%} success, {s['total_uses']} uses")

    dashboard.save_report("mrf_dashboard_learning.html", format="html")
    print(f"\n📊 Report saved: mrf_dashboard_learning.html")


def demo_ab_testing():
    """Demo 3: Dashboard with A/B testing."""
    print("\n" + "=" * 70)
    print("DEMO 3: Dashboard with A/B Testing")
    print("=" * 70)

    if not AB_TESTING_AVAILABLE:
        print("⚠️  A/B testing framework not available")
        return

    # Create dashboard with A/B testing enabled
    dashboard = create_dashboard(
        enable_ab_testing=True
    )

    # Create A/B test
    test_name = dashboard.create_ab_test(
        name="mrf_verify_enhancement",
        control_description="Baseline verify mode",
        treatment_description="MRF-enhanced verify with epistemic confidence",
        traffic_split=0.5  # 50/50 split
    )

    if not test_name:
        print("❌ Failed to create A/B test")
        return

    print(f"✅ Created A/B test: {test_name}")

    # Simulate A/B test results (need 30+ per group for statistical significance)
    import random
    print("\n📊 Simulating A/B test results...")

    # Control group: baseline performance
    for i in range(40):
        user_id = f"user_{i}"
        quality_score = random.gauss(0.75, 0.08)  # Mean 0.75, std 0.08
        execution_time = random.gauss(450, 30)

        dashboard.log_ab_test_result(
            test_name=test_name,
            user_id=user_id,
            quality_score=max(0.0, min(1.0, quality_score)),
            execution_time_ms=max(0, execution_time)
        )

    # Treatment group: MRF-enhanced (better quality, slightly slower)
    for i in range(40, 80):
        user_id = f"user_{i}"
        quality_score = random.gauss(0.89, 0.06)  # Mean 0.89, std 0.06 (IMPROVEMENT!)
        execution_time = random.gauss(480, 35)  # Slightly slower

        dashboard.log_ab_test_result(
            test_name=test_name,
            user_id=user_id,
            quality_score=max(0.0, min(1.0, quality_score)),
            execution_time_ms=max(0, execution_time)
        )

    # Analyze results
    results = dashboard.get_ab_test_results(test_name)

    if results:
        print(f"\n🔬 A/B Test Analysis:")
        print(f"\n  Test: {results['test_name']}")
        print(f"  Statistically significant: {'✅ YES' if results['is_significant'] else '❌ NO'}")
        print(f"  Treatment better: {'✅ YES' if results['treatment_better'] else '❌ NO'}")
        print(f"  Deployment decision: {results['deployment_decision'].upper()}")

        # Statistics
        stats = results['statistics']
        print(f"\n  Control Group:")
        print(f"    Mean quality: {stats['control']['mean_quality']:.3f}")
        print(f"    Sample size: {stats['control']['sample_size']}")

        print(f"\n  Treatment Group:")
        print(f"    Mean quality: {stats['treatment']['mean_quality']:.3f}")
        print(f"    Sample size: {stats['treatment']['sample_size']}")

        print(f"\n  Difference:")
        print(f"    Quality improvement: {stats['difference']['quality_improvement_percent']:+.1f}%")
        print(f"    Cohen's d: {stats['difference']['cohens_d']:.3f}")
        print(f"    Z-score: {stats['difference']['z_score']:.3f}")
        print(f"    Time increase: {stats['difference']['time_increase_percent']:+.1f}%")

        print(f"\n  Interpretation:")
        print(f"    {results['interpretation']}")

    # List all A/B tests
    all_tests = dashboard.list_ab_tests()
    print(f"\n📋 Active A/B Tests: {len(all_tests)}")
    for name, summary in all_tests.items():
        print(f"\n  {name}:")
        print(f"    Control: {summary['control_description']}")
        print(f"    Treatment: {summary['treatment_description']}")
        print(f"    Traffic split: {summary['traffic_split']:.0%}")
        print(f"    Sample sizes: control={summary['sample_sizes']['control']}, treatment={summary['sample_sizes']['treatment']}")
        print(f"    Ready for analysis: {'✅ YES' if summary['ready_for_analysis'] else '❌ NO (need more data)'}")


def demo_prometheus_metrics():
    """Demo 4: Prometheus metrics export."""
    print("\n" + "=" * 70)
    print("DEMO 4: Prometheus Metrics Export")
    print("=" * 70)

    # Create dashboard with all features
    dashboard = create_dashboard(
        enable_learning=LEARNING_AVAILABLE,
        enable_ab_testing=AB_TESTING_AVAILABLE
    )

    # Add some sample data
    test_data = [
        ("agentic", "verify", 0.75, 0.92, 450.0),
        ("rag", "refine", 0.68, 0.85, 380.0),
        ("alignment", "verify", 0.72, 0.89, 410.0),
    ]

    for system, strategy, before, after, time_ms in test_data:
        dashboard.log_enhancement(
            system=system,
            query=f"Test query for {system}",
            strategy=strategy,
            quality_before=before,
            quality_after=after,
            execution_time_ms=time_ms
        )

    # Export Prometheus metrics
    metrics = dashboard.export_prometheus_metrics()

    print("\n📊 Prometheus Metrics (sample):\n")
    # Show first 20 lines
    lines = metrics.split("\n")
    for line in lines[:20]:
        print(f"  {line}")

    if len(lines) > 20:
        print(f"  ... ({len(lines) - 20} more lines)")

    # Save to file
    metrics_file = Path("mrf_prometheus_metrics.txt")
    metrics_file.write_text(metrics)
    print(f"\n💾 Metrics saved: {metrics_file}")


def demo_complete_integration():
    """Demo 5: Complete integrated system."""
    print("\n" + "=" * 70)
    print("DEMO 5: Complete Integrated System")
    print("=" * 70)

    # Create dashboard with all features
    dashboard = create_dashboard(
        enable_learning=LEARNING_AVAILABLE,
        enable_ab_testing=AB_TESTING_AVAILABLE
    )

    print("\n✅ Dashboard Features:")
    print(f"  - Real-time monitoring: Enabled")
    print(f"  - Thompson Sampling learning: {'✅ Enabled' if LEARNING_AVAILABLE else '❌ Disabled'}")
    print(f"  - A/B testing: {'✅ Enabled' if AB_TESTING_AVAILABLE else '❌ Disabled'}")
    print(f"  - Prometheus metrics: Enabled")

    # Simulate realistic production usage
    print("\n📊 Simulating production usage...")

    import random
    systems = ["agentic", "rag", "alignment", "recursive"]
    strategies = ["verify", "refine", "critique", "elegance"]
    query_types = ["factual", "analytical", "procedural", "creative"]

    for i in range(50):
        system = random.choice(systems)
        strategy = random.choice(strategies)
        query_type = random.choice(query_types)

        quality_before = random.uniform(0.60, 0.80)
        quality_after = quality_before + random.uniform(0.05, 0.25)
        execution_time = random.uniform(300, 600)

        dashboard.log_enhancement(
            system=system,
            query=f"Production query {i+1}",
            strategy=strategy,
            quality_before=quality_before,
            quality_after=min(1.0, quality_after),
            execution_time_ms=execution_time,
            metadata={"query_type": query_type}
        )

    # Get comprehensive statistics
    stats = dashboard.get_statistics()
    print(f"\n📈 Production Statistics:")
    print(f"  Total enhancements: {stats['total_enhancements']}")
    print(f"  Systems active: {stats['systems_active']}")
    print(f"  Avg quality improvement: +{stats['quality_improvement_percent']:.1f}%")
    print(f"  Avg execution time: {stats['avg_execution_time_ms']:.1f}ms")

    # Learning recommendations (if available)
    if LEARNING_AVAILABLE and dashboard.learner:
        print(f"\n🧠 Top Strategy Recommendations:")
        learning_stats = dashboard.get_learning_statistics()
        if learning_stats and learning_stats['best_strategies_per_context']:
            for context, data in list(learning_stats['best_strategies_per_context'].items())[:3]:
                print(f"\n  {context}:")
                print(f"    Strategy: {data['strategy']}")
                print(f"    Expected reward: {data['expected_reward']:.2f}")
                print(f"    Success rate: {data['success_rate']:.1%}")

    # Save comprehensive report
    dashboard.save_report("mrf_dashboard_complete.html", format="html")
    dashboard.save_report("mrf_dashboard_complete.md", format="markdown")
    dashboard.save_report("mrf_dashboard_complete.json", format="json")

    print(f"\n💾 Reports saved:")
    print(f"  - mrf_dashboard_complete.html (interactive)")
    print(f"  - mrf_dashboard_complete.md (readable)")
    print(f"  - mrf_dashboard_complete.json (machine-readable)")

    # Export Prometheus metrics
    metrics = dashboard.export_prometheus_metrics()
    Path("mrf_prometheus_metrics_complete.txt").write_text(metrics)
    print(f"  - mrf_prometheus_metrics_complete.txt (Prometheus)")

    print("\n✅ Complete integration demonstration finished!")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("MRF Analytics Integration - Complete Demo Suite")
    print("=" * 70)
    print("\nDemonstrates:")
    print("  1. Basic dashboard (no learning/A/B testing)")
    print("  2. Thompson Sampling learning integration")
    print("  3. A/B testing framework")
    print("  4. Prometheus metrics export")
    print("  5. Complete integrated system")

    # Run demos
    demo_basic_dashboard()
    demo_learning_integration()
    demo_ab_testing()
    demo_prometheus_metrics()
    demo_complete_integration()

    print("\n" + "=" * 70)
    print("All demos completed successfully! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    main()
