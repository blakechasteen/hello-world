#!/usr/bin/env python3
"""
Promptly Analytics - Complete Example

Demonstrates all major features of the analytics system.
"""

import time
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptly.promptly import Promptly
from promptly.analytics import (
    enable_analytics,
    ReportGenerator,
    ReportConfig,
    ReportPeriod,
    ReportFormat,
    Visualizer
)


def main():
    """Run analytics example"""
    print("=" * 80)
    print("Promptly Analytics - Complete Example")
    print("=" * 80)

    # 1. Initialize Promptly with analytics
    print("\n1. Initializing Promptly with analytics...")

    promptly = Promptly()

    # Initialize if needed
    try:
        promptly.init()
        print("   ✓ Repository initialized")
    except Exception:
        print("   ✓ Repository already exists")

    # Enable analytics with custom configuration
    config = {
        'retention_days': 30,
        'enable_resource_sampling': True,
        'integrations': {
            'logging': {
                'enabled': True,
                'path': './logs/analytics.jsonl'
            }
        }
    }

    promptly = enable_analytics(promptly, config=config)
    print("   ✓ Analytics enabled")

    # 2. Perform some operations to generate data
    print("\n2. Generating sample data...")

    # Add prompts
    prompts = [
        ('greeting', 'Hello {name}! How are you today?'),
        ('farewell', 'Goodbye {name}! See you soon!'),
        ('question', 'What is your question about {topic}?'),
        ('summary', 'Please summarize: {text}'),
        ('translation', 'Translate to {language}: {text}')
    ]

    for name, content in prompts:
        promptly.add(name, content, user='alice')
        time.sleep(0.1)  # Small delay to generate varied timestamps

    print(f"   ✓ Added {len(prompts)} prompts")

    # Access prompts multiple times
    for _ in range(3):
        for name, _ in prompts:
            promptly.get(name, user='bob')
            time.sleep(0.05)

    print(f"   ✓ Accessed prompts {3 * len(prompts)} times")

    # Run evaluations
    test_cases = [
        {'inputs': {'name': 'Alice'}, 'expected': 'greeting'},
        {'inputs': {'name': 'Bob'}, 'expected': 'greeting'},
    ]

    for name, _ in prompts[:2]:
        # Mock evaluator function
        def mock_evaluator(actual, expected):
            import random
            return random.uniform(0.7, 0.99)

        # Add evaluator to test cases
        for tc in test_cases:
            tc['evaluator'] = mock_evaluator

        promptly.eval_prompt(name, test_cases, user='alice')
        time.sleep(0.1)

    print(f"   ✓ Ran {2 * len(test_cases)} evaluations")

    # Create and execute a chain
    promptly.create_chain('greet_and_ask', ['greeting', 'question'], user='alice')

    try:
        promptly.execute_chain('greet_and_ask', {'name': 'Alice', 'topic': 'AI'}, user='bob')
        print("   ✓ Executed chain")
    except Exception as e:
        print(f"   ! Chain execution failed (expected): {e}")

    # 3. View performance statistics
    print("\n3. Performance Statistics:")
    print("-" * 80)

    perf_stats = promptly.get_performance_stats()

    print(f"\n{'Operation':<20} {'Count':>8} {'Avg (ms)':>10} {'Max (ms)':>10}")
    print("-" * 52)

    for op, stats in sorted(perf_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"{op:<20} {stats['count']:>8} {stats['avg_ms']:>10.2f} {stats['max_ms']:>10.2f}")

    # 4. View usage statistics
    print("\n4. Usage Statistics (Last 7 Days):")
    print("-" * 80)

    usage_stats = promptly.get_usage_stats(days=7)

    print("\nMost Used Prompts:")
    for p in usage_stats['most_used'][:5]:
        print(f"  - {p['prompt_name']}: {p['access_count']} accesses")

    print("\nEvaluation Stats:")
    eval_stats = usage_stats['evaluations']
    print(f"  Total Evaluations: {eval_stats['total_evaluations']}")
    if eval_stats.get('average_score'):
        print(f"  Average Score: {eval_stats['average_score']:.3f}")

    # 5. View quality statistics
    print("\n5. Quality Statistics (Last 30 Days):")
    print("-" * 80)

    quality_stats = promptly.get_quality_stats(days=30)

    if quality_stats['top_performing']:
        print("\nTop Performing Prompts:")
        for p in quality_stats['top_performing'][:5]:
            print(f"  - {p['prompt_name']}: {p['avg_score']:.3f} (n={p['measurements']})")

    if quality_stats['alerts']:
        print("\nActive Alerts:")
        for a in quality_stats['alerts']:
            print(f"  ⚠️  {a['prompt_name']}: {a['message']}")
    else:
        print("\nNo active quality alerts")

    # 6. Generate visualizations (if plotext available)
    print("\n6. Generating Visualizations:")
    print("-" * 80)

    try:
        import plotext
        PLOTEXT_AVAILABLE = True
    except ImportError:
        PLOTEXT_AVAILABLE = False

    viz = Visualizer(
        performance_monitor=promptly.performance,
        usage_analytics=promptly.usage,
        quality_metrics=promptly.quality
    )

    if PLOTEXT_AVAILABLE:
        print("\nOperation Times:")
        viz.plot_operation_times(days=1, show=True)
    else:
        print("  ⚠️  plotext not installed - skipping terminal charts")

    # Generate HTML dashboard
    dashboard_path = './reports/dashboard.html'
    Path('./reports').mkdir(exist_ok=True)

    dashboard = viz.generate_dashboard(dashboard_path, days=7)
    print(f"\n  ✓ HTML Dashboard: {dashboard}")

    # 7. Generate reports
    print("\n7. Generating Reports:")
    print("-" * 80)

    generator = ReportGenerator(
        performance_monitor=promptly.performance,
        usage_analytics=promptly.usage,
        quality_metrics=promptly.quality,
        visualizer=viz
    )

    # Daily report (Markdown)
    daily_md = generator.generate_daily_summary('./reports/daily_report.md')
    print(f"  ✓ Daily Report (MD): {daily_md}")

    # Weekly report (HTML)
    config = ReportConfig(
        period=ReportPeriod.WEEKLY,
        format=ReportFormat.HTML
    )
    weekly_html = generator.generate_report(config, './reports/weekly_report.html')
    print(f"  ✓ Weekly Report (HTML): {weekly_html}")

    # Performance report
    perf_report = generator.generate_performance_report('./reports/performance.md', days=7)
    print(f"  ✓ Performance Report: {perf_report}")

    # Quality report
    quality_report = generator.generate_quality_report('./reports/quality.md', days=30)
    print(f"  ✓ Quality Report: {quality_report}")

    # 8. Export data
    print("\n8. Exporting Data:")
    print("-" * 80)

    # JSON export
    json_path = viz.export_to_json('./exports/analytics.json', days=7)
    print(f"  ✓ JSON Export: {json_path}")

    # CSV exports
    csv_perf = viz.export_to_csv('performance', './exports/performance.csv', days=7)
    print(f"  ✓ Performance CSV: {csv_perf}")

    csv_usage = viz.export_to_csv('usage', './exports/usage.csv', days=7)
    print(f"  ✓ Usage CSV: {csv_usage}")

    csv_quality = viz.export_to_csv('quality', './exports/quality.csv', days=30)
    print(f"  ✓ Quality CSV: {csv_quality}")

    # 9. Cleanup
    print("\n9. Cleanup Old Data:")
    print("-" * 80)

    # Note: This won't delete much since we just created the data
    deleted = promptly.cleanup_old_data()
    print(f"  Performance: {deleted['performance']} records deleted")
    print(f"  Usage: {deleted['usage']} records deleted")
    print(f"  Quality: {deleted['quality']} records deleted")

    # 10. Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\n✓ Analytics example completed successfully!")
    print("\nGenerated Files:")
    print("  - Reports: ./reports/")
    print("  - Exports: ./exports/")
    print("  - Logs: ./logs/analytics.jsonl")
    print("  - Analytics DBs: .promptly/analytics/")

    print("\nNext Steps:")
    print("  1. View the HTML dashboard: open reports/dashboard.html")
    print("  2. Read the reports: cat reports/daily_report.md")
    print("  3. Explore exported data: cat exports/analytics.json")
    print("  4. Try the CLI: python -m promptly.analytics.cli stats performance")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
