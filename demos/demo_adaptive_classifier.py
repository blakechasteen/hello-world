"""
Adaptive Classifier Integration Demo
====================================
Demonstrates complete Phase 3 adaptive learning integration with MoonshotQueryClassifier.

Features demonstrated:
1. Classification with automatic logging
2. Pattern mining from production logs
3. Hourly validation checks
4. Safe pattern deployment
5. Daily/weekly reporting
6. Background learning loop

Author: Claude Code
Date: November 13, 2025
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.routing.query_classifier_adaptive import (
    AdaptiveMoonshotClassifier,
    create_adaptive_classifier
)


async def main():
    print("=" * 80)
    print("ADAPTIVE MOONSHOT CLASSIFIER DEMO")
    print("=" * 80)
    print()

    # Step 1: Create adaptive classifier
    print("Step 1: Creating adaptive classifier...")
    print("-" * 80)

    classifier = AdaptiveMoonshotClassifier(
        enable_semantic_tier=False,  # Faster for demo
        enable_adaptive_learning=True,
        data_dir=Path("./test_adaptive"),
        background_learning=False  # Manual control for demo
    )

    print("[OK] AdaptiveMoonshotClassifier created")
    print(f"  - Data directory: {classifier.data_dir}")
    print(f"  - Classification log: {classifier.classification_log_path}")
    print(f"  - Patterns file: {classifier.patterns_path}")
    print(f"  - Reports directory: {classifier.reports_dir}")
    print()

    # Step 2: Classify sample queries (generate logs)
    print("Step 2: Classifying sample queries...")
    print("-" * 80)

    sample_queries = [
        # TRIVIAL
        "hi",
        "thanks",
        "ok",
        "bye",
        "help",

        # SIMPLE
        "what is Python?",
        "what are transformers?",
        "define machine learning",
        "who is Einstein?",
        "when is Christmas?",

        # COMPLEX
        "explain how neural networks work",
        "describe attention mechanism with examples",
        "why do transformers use self-attention?",
        "how does backpropagation work in detail?",
        "explain gradient descent with examples",

        # RESEARCH
        "analyze tradeoffs between Thompson Sampling and epsilon-greedy",
        "compare and contrast supervised and unsupervised learning",
        "evaluate advantages and disadvantages of neural vs symbolic AI",
        "compare RNNs vs Transformers vs State Space Models",
        "analyze pros and cons of different optimizers",
    ]

    print(f"Classifying {len(sample_queries)} queries...")
    print()

    results = []
    for i, query in enumerate(sample_queries, 1):
        result = classifier.classify(query)
        results.append(result)

        # Show first 5 results
        if i <= 5:
            print(f"{i}. \"{query}\"")
            print(f"   -> {result.complexity.value.upper()} "
                  f"(confidence: {result.confidence:.1%}, "
                  f"tier: {result.tier_used}, "
                  f"latency: {result.latency_ms:.2f}ms)")

    print(f"   ... and {len(sample_queries) - 5} more")
    print()

    # Stats
    complexity_counts = {}
    for result in results:
        complexity = result.complexity.value
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

    print("Classification Summary:")
    for complexity, count in sorted(complexity_counts.items()):
        print(f"  - {complexity.upper():10s}: {count} queries")
    print()

    # Step 3: Run manual learning cycle
    print("Step 3: Running adaptive learning cycle...")
    print("-" * 80)
    print()

    await classifier._run_learning_cycle()

    print()

    # Step 4: View learning statistics
    print("Step 4: Learning statistics...")
    print("-" * 80)

    stats = classifier.get_learning_statistics()

    print()
    print("Adaptive Learning Statistics:")
    print(f"  Queries Logged: {stats['total_queries_logged']}")
    print(f"  Patterns Discovered: {stats['patterns_discovered']}")
    print(f"  Patterns Deployed: {stats['patterns_deployed']}")
    print(f"  Validation Accuracy: {stats['validation_accuracy']:.1%}")
    print(f"  Regression Alerts: {stats['regression_alerts']}")
    print()

    print("Base Classifier Statistics:")
    base_stats = stats['base_classifier']
    print(f"  Total Queries: {base_stats['total_queries']}")
    print(f"  Accuracy: {base_stats['accuracy']:.1%}")
    print(f"  Misclassifications by Tier:")
    for tier, count in base_stats['misclassifications_by_tier'].items():
        print(f"    - {tier}: {count}")
    print()

    # Step 5: Check generated files
    print("Step 5: Generated files...")
    print("-" * 80)

    files_generated = []

    if classifier.classification_log_path.exists():
        with open(classifier.classification_log_path, 'r', encoding='utf-8') as f:
            log_count = sum(1 for _ in f)
        files_generated.append(f"{classifier.classification_log_path} ({log_count} entries)")

    if classifier.patterns_path.exists():
        files_generated.append(str(classifier.patterns_path))

    # Check for reports
    if classifier.reports_dir.exists():
        for report_file in classifier.reports_dir.glob("*"):
            files_generated.append(str(report_file))

    print()
    if files_generated:
        print(f"Files Generated ({len(files_generated)}):")
        for f in files_generated:
            print(f"  - {f}")
    else:
        print("No additional files generated yet (learning cycle complete)")
    print()

    # Step 6: Test background learning (brief demo)
    print("Step 6: Background learning demo...")
    print("-" * 80)
    print()

    print("Starting background learning loop (5 second demo)...")
    classifier.background_learning = True
    classifier.learning_update_interval = 5.0  # 5 seconds for demo

    await classifier.start_background_learning()
    print("[OK] Background learning started")

    # Let it run for 10 seconds
    await asyncio.sleep(10)

    print("[OK] Stopping background learning...")
    await classifier.stop_background_learning()
    print("[OK] Background learning stopped")
    print()

    # Summary
    print("=" * 80)
    print("ADAPTIVE CLASSIFIER DEMO COMPLETE")
    print("=" * 80)
    print()

    print("Key Features Demonstrated:")
    print("  [OK] Adaptive classifier initialization")
    print("  [OK] Classification with automatic logging (JSONL)")
    print("  [OK] Pattern mining from production logs")
    print("  [OK] Hourly validation checks")
    print("  [OK] Pattern deployment (gradual rollout)")
    print("  [OK] Learning statistics and reporting")
    print("  [OK] Background learning loop")
    print()

    print("Architecture:")
    print("  Foreground: Fast multi-tier classification (Tier 1-4)")
    print("    └─ Tier 1: Pattern Cache (0.1ms)")
    print("    └─ Tier 2: Heuristic Scoring (0.5ms)")
    print("    └─ Tier 3: Semantic Embeddings (20ms, optional)")
    print("    └─ Tier 4: Conservative Escalation")
    print()
    print("  Background: Adaptive learning loop (every hour)")
    print("    ├─ PatternMiner: Extract patterns from logs")
    print("    ├─ ContinuousValidator: Validate accuracy hourly")
    print("    ├─ AdaptiveUpdater: Deploy patterns safely")
    print("    └─ PerformanceReporter: Generate daily/weekly reports")
    print()

    print("Production Integration:")
    print("  1. Use AdaptiveMoonshotClassifier for all queries")
    print("  2. Enable background_learning=True for automatic adaptation")
    print("  3. Monitor reports in ./data/reports/")
    print("  4. Configure Slack/email alerts via PerformanceReporter")
    print("  5. Set up Prometheus scraper for metrics")
    print()

    print("Next Steps:")
    print("  1. Test with production traffic")
    print("  2. Monitor validation accuracy trends")
    print("  3. Review and deploy high-quality patterns")
    print("  4. Configure alerting thresholds")
    print("  5. Set up Grafana dashboards")


if __name__ == "__main__":
    asyncio.run(main())
