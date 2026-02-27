"""
Performance Reporter Demo
==========================
Demonstrates daily/weekly report generation, Prometheus metrics, and alerting.

Author: Claude Code
Date: November 13, 2025
"""

import asyncio
from pathlib import Path
from hololoom.routing.learning import (
    PerformanceReporter,
    ContinuousValidator,
    AdaptiveUpdater,
    RegressionAlert,
    create_validation_set
)


class MockClassifier:
    """Mock classifier for demonstration."""
    def classify(self, query_text: str):
        from dataclasses import dataclass

        @dataclass
        class MockClassification:
            complexity: str
            confidence: float

        class ComplexityValue:
            def __init__(self, value):
                self.value = value

        if "?" in query_text:
            complexity = "simple"
        else:
            complexity = "trivial"

        return MockClassification(
            complexity=ComplexityValue(complexity),
            confidence=0.95
        )


async def main():
    print("=" * 80)
    print("PERFORMANCE REPORTER DEMO")
    print("=" * 80)
    print()

    # Step 1: Setup components
    print("Step 1: Setting up components...")

    classifier = MockClassifier()

    # Create validation set
    from demos.demo_continuous_validator import create_sample_validation_set
    validation_dir = Path("./test_reporting")
    validation_file = validation_dir / "validation_set.json"

    queries = create_sample_validation_set()
    create_validation_set(queries, str(validation_file))

    # Create validator
    validator = ContinuousValidator(
        classifier=classifier,
        validation_set_path=str(validation_file),
        regression_threshold=0.02,
        baseline_accuracy=1.0
    )

    # Create updater
    updater = AdaptiveUpdater(
        classifier=classifier,
        validator=validator
    )

    # Create reporter
    reporter = PerformanceReporter(
        validator=validator,
        updater=updater,
        output_dir=str(validation_dir / "reports")
    )

    print("[OK] MockClassifier initialized")
    print("[OK] ContinuousValidator initialized")
    print("[OK] AdaptiveUpdater initialized")
    print("[OK] PerformanceReporter initialized")
    print()

    # Step 2: Generate some validation data
    print("Step 2: Generating validation data...")
    print("-" * 80)

    # Run a few validations to populate history
    for i in range(3):
        result = await validator.validate_hourly(sample_size=15)
        print(f"Validation {i+1}: accuracy={result.overall_accuracy:.1%}")

    print()
    print(f"Generated {len(validator.validation_history)} validation records")
    print()

    # Step 3: Generate daily report
    print("Step 3: Generating daily report...")
    print("-" * 80)

    daily_report = reporter.generate_daily_report()

    print()
    print("Daily Report Summary:")
    print(f"  Date: {daily_report.date}")
    print(f"  Queries Classified: {daily_report.queries_classified}")
    print(f"  Overall Accuracy: {daily_report.overall_accuracy:.1%}")
    print(f"  Avg Latency: {daily_report.avg_latency_ms:.3f}ms")
    print(f"  Patterns Deployed: {daily_report.patterns_deployed}")
    print(f"  Regressions Detected: {daily_report.regressions_detected}")
    print(f"  Trend vs Yesterday: {'+' if daily_report.trend_vs_yesterday > 0 else ''}{daily_report.trend_vs_yesterday:.1%}")
    print()
    print("  By Complexity:")
    for complexity, accuracy in daily_report.by_complexity.items():
        print(f"    {complexity.upper():10s}: {accuracy:.1%}")
    print()

    # Step 4: Generate weekly report
    print("Step 4: Generating weekly report...")
    print("-" * 80)

    weekly_report = reporter.generate_weekly_report()

    print()
    print("Weekly Report Summary:")
    print(f"  Week: {weekly_report.week_start} to {weekly_report.week_end}")
    print(f"  Total Queries: {weekly_report.total_queries}")
    print(f"  Overall Accuracy: {weekly_report.overall_accuracy:.1%}")
    print(f"  Avg Latency: {weekly_report.avg_latency_ms:.3f}ms")
    print(f"  Patterns Deployed: {weekly_report.patterns_deployed}")
    print(f"  Regressions Detected: {weekly_report.regressions_detected}")
    print(f"  7-Day Trend: {'+' if weekly_report.trend_7day > 0 else ''}{weekly_report.trend_7day:.1%}")
    print()
    print("  By Complexity:")
    for complexity, accuracy in weekly_report.by_complexity.items():
        print(f"    {complexity.upper():10s}: {accuracy:.1%}")
    print()
    print("  Recommendations:")
    for i, rec in enumerate(weekly_report.recommendations, 1):
        print(f"    {i}. {rec[:80]}...")
    print()

    # Step 5: Export Prometheus metrics
    print("Step 5: Exporting Prometheus metrics...")
    print("-" * 80)

    metrics = reporter.export_prometheus_metrics()

    print()
    print("Prometheus Metrics (sample):")
    print(metrics.split('\n')[:15])  # Show first 15 lines
    print("  ...")
    print(f"Total metrics: {len(metrics.split(chr(10)))} lines")
    print()

    # Step 6: Format alerts
    print("Step 6: Formatting alerts...")
    print("-" * 80)

    # Create a sample regression alert
    from datetime import datetime
    sample_alert = RegressionAlert(
        timestamp=datetime.now().timestamp(),
        current_accuracy=0.92,
        baseline_accuracy=1.0,
        drop_percentage=0.08,
        affected_complexity=["trivial(90%)", "complex(85%)"],
        severity="critical",
        message="Test regression alert"
    )

    # Format for Slack
    slack_message = reporter.format_slack_alert(sample_alert)
    print()
    print("Slack Alert:")
    print(slack_message)

    # Format for email
    email_alert = reporter.format_email_alert(sample_alert)
    print()
    print("Email Alert:")
    print(f"  Subject: {email_alert['subject']}")
    print(f"  Body:")
    for line in email_alert['body'].split('\n')[:10]:
        print(f"    {line}")
    print("    ...")
    print()

    # Step 7: Generate markdown reports
    print("Step 7: Generating markdown reports...")
    print("-" * 80)

    # Daily report markdown
    daily_md = reporter.generate_markdown_report(daily_report)
    daily_md_file = validation_dir / "reports" / "daily_report.md"
    with open(daily_md_file, 'w', encoding='utf-8') as f:
        f.write(daily_md)
    print(f"Saved daily report to {daily_md_file}")

    # Weekly report markdown
    weekly_md = reporter.generate_markdown_report(weekly_report)
    weekly_md_file = validation_dir / "reports" / "weekly_report.md"
    with open(weekly_md_file, 'w', encoding='utf-8') as f:
        f.write(weekly_md)
    print(f"Saved weekly report to {weekly_md_file}")
    print()

    print("Daily Report (Markdown preview):")
    print("-" * 40)
    print(daily_md.split('\n\n')[0])  # Show first section
    print("...")
    print()

    # Summary
    print("=" * 80)
    print("PERFORMANCE REPORTER DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  [OK] Daily report generation")
    print("  [OK] Weekly report generation")
    print("  [OK] Prometheus metrics export")
    print("  [OK] Slack alert formatting")
    print("  [OK] Email alert formatting")
    print("  [OK] Markdown report generation")
    print("  [OK] Recommendations engine")
    print()
    print("Generated Files:")
    print(f"  - {validation_dir / 'reports' / f'daily_report_{daily_report.date}.json'}")
    print(f"  - {validation_dir / 'reports' / f'weekly_report_{weekly_report.week_start}_to_{weekly_report.week_end}.json'}")
    print(f"  - {validation_dir / 'reports' / 'prometheus_metrics.txt'}")
    print(f"  - {validation_dir / 'reports' / 'daily_report.md'}")
    print(f"  - {validation_dir / 'reports' / 'weekly_report.md'}")
    print()
    print("Report Schedule:")
    print("  • Daily reports: Generated at 9am UTC")
    print("  • Weekly reports: Generated Sunday 9am UTC")
    print("  • Prometheus metrics: Exported continuously")
    print("  • Alerts: Sent immediately on regression")
    print()
    print("Integration Points:")
    print("  • Grafana: Use Prometheus metrics for dashboards")
    print("  • Slack: POST slack_message to webhook URL")
    print("  • Email: Send via SMTP using email_alert")
    print("  • Monitoring: Track trends and recommendations")
    print()
    print("Next Steps:")
    print("  1. Set up Prometheus scraper to collect metrics")
    print("  2. Create Grafana dashboards using exported metrics")
    print("  3. Configure Slack webhook for regression alerts")
    print("  4. Set up email SMTP for alert distribution")
    print("  5. Schedule daily/weekly report generation")


if __name__ == "__main__":
    asyncio.run(main())
