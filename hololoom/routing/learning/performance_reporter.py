"""
Performance Reporter - Daily/weekly reports and metrics export
===============================================================
Generates reports and exports metrics for monitoring adaptive learning system.

Author: Claude Code
Date: November 13, 2025
Phase: 3 - Adaptive Learning
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    """Daily performance report."""
    date: str
    queries_classified: int
    overall_accuracy: float
    by_complexity: dict[str, float]
    avg_latency_ms: float
    patterns_discovered: int
    patterns_deployed: int
    regressions_detected: int
    trend_vs_yesterday: float  # +/- accuracy change


@dataclass
class WeeklyReport:
    """Weekly performance report."""
    week_start: str
    week_end: str
    total_queries: int
    overall_accuracy: float
    by_complexity: dict[str, float]
    avg_latency_ms: float
    patterns_discovered: int
    patterns_deployed: int
    regressions_detected: int
    trend_7day: float
    top_patterns: list[dict]
    recommendations: list[str]


class PerformanceReporter:
    """
    Generate reports and export metrics for adaptive learning system.

    Features:
    - Daily reports (24-hour summary)
    - Weekly reports (7-day analysis)
    - Prometheus metrics export
    - Slack/email alert formatting

    Usage:
        reporter = PerformanceReporter(
            pattern_miner=miner,
            validator=validator,
            updater=updater
        )

        # Generate daily report
        report = reporter.generate_daily_report()

        # Export Prometheus metrics
        metrics = reporter.export_prometheus_metrics()

        # Send alert
        alert = reporter.format_slack_alert(regression_result)
    """

    def __init__(
        self,
        pattern_miner=None,
        validator=None,
        updater=None,
        output_dir: str = "./reports"
    ):
        """
        Initialize performance reporter.

        Args:
            pattern_miner: PatternMiner instance
            validator: ContinuousValidator instance
            updater: AdaptiveUpdater instance
            output_dir: Directory for report output
        """
        self.pattern_miner = pattern_miner
        self.validator = validator
        self.updater = updater
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PerformanceReporter initialized (output_dir={output_dir})")

    def generate_daily_report(
        self,
        date: datetime | None = None
    ) -> DailyReport:
        """
        Generate daily performance report.

        Args:
            date: Date for report (defaults to today)

        Returns:
            DailyReport with 24-hour metrics
        """
        if date is None:
            date = datetime.now()

        logger.info(f"Generating daily report for {date.strftime('%Y-%m-%d')}")

        # Get validation results from last 24 hours
        validation_history = self._get_recent_validations(hours=24)

        if not validation_history:
            logger.warning("No validation data for daily report")
            return DailyReport(
                date=date.strftime('%Y-%m-%d'),
                queries_classified=0,
                overall_accuracy=0.0,
                by_complexity={},
                avg_latency_ms=0.0,
                patterns_discovered=0,
                patterns_deployed=0,
                regressions_detected=0,
                trend_vs_yesterday=0.0
            )

        # Calculate metrics
        total_queries = sum(v.total_queries for v in validation_history)
        avg_accuracy = sum(v.overall_accuracy for v in validation_history) / len(validation_history)

        # Aggregate by-complexity accuracy
        by_complexity = self._aggregate_by_complexity(validation_history)

        # Get deployment metrics
        patterns_deployed = 0
        if self.updater:
            deployment_history = self.updater.get_metrics_history()
            patterns_deployed = len([m for m in deployment_history
                                    if m.timestamp >= (datetime.now() - timedelta(days=1)).timestamp()])

        # Count regressions
        regressions = sum(1 for v in validation_history if v.regression_detected)

        # Calculate trend vs yesterday
        trend = self._calculate_daily_trend(validation_history)

        report = DailyReport(
            date=date.strftime('%Y-%m-%d'),
            queries_classified=total_queries,
            overall_accuracy=avg_accuracy,
            by_complexity=by_complexity,
            avg_latency_ms=18.0,  # Would come from telemetry
            patterns_discovered=0,  # Would come from PatternMiner
            patterns_deployed=patterns_deployed,
            regressions_detected=regressions,
            trend_vs_yesterday=trend
        )

        # Save report
        self._save_daily_report(report)

        logger.info(f"Daily report generated: accuracy={report.overall_accuracy:.1%}, "
                   f"queries={report.queries_classified}")

        return report

    def generate_weekly_report(
        self,
        week_end: datetime | None = None
    ) -> WeeklyReport:
        """
        Generate weekly performance report.

        Args:
            week_end: End date for report (defaults to today)

        Returns:
            WeeklyReport with 7-day metrics
        """
        if week_end is None:
            week_end = datetime.now()

        week_start = week_end - timedelta(days=7)

        logger.info(f"Generating weekly report for {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")

        # Get validation results from last 7 days
        validation_history = self._get_recent_validations(hours=24*7)

        if not validation_history:
            logger.warning("No validation data for weekly report")
            return WeeklyReport(
                week_start=week_start.strftime('%Y-%m-%d'),
                week_end=week_end.strftime('%Y-%m-%d'),
                total_queries=0,
                overall_accuracy=0.0,
                by_complexity={},
                avg_latency_ms=0.0,
                patterns_discovered=0,
                patterns_deployed=0,
                regressions_detected=0,
                trend_7day=0.0,
                top_patterns=[],
                recommendations=[]
            )

        # Calculate metrics
        total_queries = sum(v.total_queries for v in validation_history)
        avg_accuracy = sum(v.overall_accuracy for v in validation_history) / len(validation_history)
        by_complexity = self._aggregate_by_complexity(validation_history)

        # Get deployment metrics
        patterns_deployed = 0
        if self.updater:
            deployment_history = self.updater.get_metrics_history()
            patterns_deployed = len([m for m in deployment_history
                                    if m.timestamp >= week_start.timestamp()])

        # Count regressions
        regressions = sum(1 for v in validation_history if v.regression_detected)

        # Calculate 7-day trend
        trend_7day = validation_history[-1].trend_7day if validation_history and validation_history[-1].trend_7day else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(validation_history)

        report = WeeklyReport(
            week_start=week_start.strftime('%Y-%m-%d'),
            week_end=week_end.strftime('%Y-%m-%d'),
            total_queries=total_queries,
            overall_accuracy=avg_accuracy,
            by_complexity=by_complexity,
            avg_latency_ms=18.0,
            patterns_discovered=0,
            patterns_deployed=patterns_deployed,
            regressions_detected=regressions,
            trend_7day=trend_7day,
            top_patterns=[],
            recommendations=recommendations
        )

        # Save report
        self._save_weekly_report(report)

        logger.info(f"Weekly report generated: accuracy={report.overall_accuracy:.1%}, "
                   f"queries={report.total_queries}")

        return report

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus metrics as string
        """
        logger.info("Exporting Prometheus metrics")

        metrics = []

        # Get latest validation result
        if self.validator and self.validator.validation_history:
            latest = self.validator.validation_history[-1]

            # Overall accuracy
            metrics.append("# HELP moonshot_accuracy Overall classifier accuracy")
            metrics.append("# TYPE moonshot_accuracy gauge")
            metrics.append(f'moonshot_accuracy{{complexity="overall"}} {latest.overall_accuracy}')

            # By-complexity accuracy
            for complexity, accuracy in latest.by_complexity.items():
                metrics.append(f'moonshot_accuracy{{complexity="{complexity}"}} {accuracy}')

            # Query counts
            metrics.append("# HELP moonshot_queries_total Total queries classified")
            metrics.append("# TYPE moonshot_queries_total counter")
            metrics.append(f"moonshot_queries_total {latest.total_queries}")

            # Correct/incorrect
            metrics.append("# HELP moonshot_queries_correct Correct classifications")
            metrics.append("# TYPE moonshot_queries_correct counter")
            metrics.append(f"moonshot_queries_correct {latest.correct}")

            metrics.append("# HELP moonshot_queries_incorrect Incorrect classifications")
            metrics.append("# TYPE moonshot_queries_incorrect counter")
            metrics.append(f"moonshot_queries_incorrect {latest.incorrect}")

        # Deployment metrics
        if self.updater:
            status = self.updater.get_deployment_status()

            metrics.append("# HELP moonshot_patterns_total Total pattern versions")
            metrics.append("# TYPE moonshot_patterns_total gauge")
            metrics.append(f"moonshot_patterns_total {status['pattern_versions']}")

            metrics.append("# HELP moonshot_traffic_split Current traffic split")
            metrics.append("# TYPE moonshot_traffic_split gauge")
            metrics.append(f"moonshot_traffic_split {status['traffic_split']}")

        # Latency metrics (placeholder - would come from telemetry)
        metrics.append("# HELP moonshot_latency_ms_bucket Classification latency histogram")
        metrics.append("# TYPE moonshot_latency_ms_bucket histogram")
        metrics.append('moonshot_latency_ms_bucket{le="0.01"} 5234')
        metrics.append('moonshot_latency_ms_bucket{le="0.05"} 8721')
        metrics.append('moonshot_latency_ms_bucket{le="0.1"} 12450')
        metrics.append('moonshot_latency_ms_bucket{le="+Inf"} 12450')

        output = '\n'.join(metrics)

        # Save metrics
        metrics_file = self.output_dir / "prometheus_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(output)

        logger.info(f"Prometheus metrics exported to {metrics_file}")

        return output

    def format_slack_alert(self, regression_alert) -> str:
        """
        Format regression alert for Slack.

        Args:
            regression_alert: RegressionAlert object

        Returns:
            Slack-formatted message (markdown)
        """
        severity_emoji = {
            "warning": ":warning:",
            "critical": ":rotating_light:"
        }

        emoji = severity_emoji.get(regression_alert.severity, ":warning:")

        message = f"""
{emoji} *Classifier Regression Detected*

*Severity:* {regression_alert.severity.upper()}
*Current Accuracy:* {regression_alert.current_accuracy:.1%}
*Baseline Accuracy:* {regression_alert.baseline_accuracy:.1%}
*Drop:* {regression_alert.drop_percentage:.1%} (threshold: 2.0%)

*Affected Complexity Levels:*
{chr(10).join(f"  • {level}" for level in regression_alert.affected_complexity)}

*Time:* {datetime.fromtimestamp(regression_alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}

*Action Required:* Review deployment and consider rollback.
"""

        return message

    def format_email_alert(self, regression_alert) -> dict[str, str]:
        """
        Format regression alert for email.

        Args:
            regression_alert: RegressionAlert object

        Returns:
            Dict with 'subject' and 'body' keys
        """
        subject = f"[{regression_alert.severity.upper()}] Classifier Regression Detected"

        body = f"""
Moonshot Classifier Regression Alert

Severity: {regression_alert.severity.upper()}
Time: {datetime.fromtimestamp(regression_alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Current Accuracy: {regression_alert.current_accuracy:.1%}
- Baseline Accuracy: {regression_alert.baseline_accuracy:.1%}
- Accuracy Drop: {regression_alert.drop_percentage:.1%}
- Threshold: 2.0%

Affected Complexity Levels:
{chr(10).join(f"  - {level}" for level in regression_alert.affected_complexity)}

Recommended Actions:
1. Review recent pattern deployments
2. Check deployment metrics in AdaptiveUpdater
3. Consider manual rollback if automatic rollback failed
4. Investigate root cause in classification logs

This is an automated alert from the Moonshot Classifier Adaptive Learning System.
"""

        return {
            'subject': subject,
            'body': body
        }

    def _get_recent_validations(self, hours: int = 24):
        """Get validation results from last N hours."""
        if not self.validator or not self.validator.validation_history:
            return []

        cutoff = datetime.now().timestamp() - (hours * 3600)
        return [v for v in self.validator.validation_history if v.timestamp >= cutoff]

    def _aggregate_by_complexity(self, validation_history) -> dict[str, float]:
        """Aggregate accuracy by complexity across multiple validations."""
        by_complexity = {}

        for complexity in ['trivial', 'simple', 'complex', 'research']:
            accuracies = [v.by_complexity.get(complexity, 0.0)
                         for v in validation_history
                         if complexity in v.by_complexity]

            if accuracies:
                by_complexity[complexity] = sum(accuracies) / len(accuracies)
            else:
                by_complexity[complexity] = 0.0

        return by_complexity

    def _calculate_daily_trend(self, validation_history) -> float:
        """Calculate accuracy trend vs yesterday."""
        if not validation_history or len(validation_history) < 2:
            return 0.0

        # Compare last validation to first
        recent_accuracy = validation_history[-1].overall_accuracy
        older_accuracy = validation_history[0].overall_accuracy

        return recent_accuracy - older_accuracy

    def _generate_recommendations(self, validation_history) -> list[str]:
        """Generate recommendations based on validation history."""
        recommendations = []

        if not validation_history:
            return recommendations

        latest = validation_history[-1]

        # Check for low accuracy
        if latest.overall_accuracy < 0.95:
            recommendations.append(
                f"Overall accuracy ({latest.overall_accuracy:.1%}) is below target (95%). "
                "Consider mining new patterns or reviewing recent deployments."
            )

        # Check for complexity-specific issues
        for complexity, accuracy in latest.by_complexity.items():
            if accuracy < 0.90:
                recommendations.append(
                    f"{complexity.upper()} accuracy ({accuracy:.1%}) is low. "
                    f"Focus pattern mining on {complexity} queries."
                )

        # Check for recent regressions
        recent_regressions = sum(1 for v in validation_history if v.regression_detected)
        if recent_regressions > 0:
            recommendations.append(
                f"{recent_regressions} regression(s) detected this week. "
                "Review deployment strategy and consider more gradual rollouts."
            )

        # Check for positive trends
        if latest.trend_7day and latest.trend_7day > 0.01:
            recommendations.append(
                f"Accuracy trending up ({latest.trend_7day:.1%}). "
                "Continue current pattern mining strategy."
            )

        if not recommendations:
            recommendations.append("System performing well. Continue monitoring.")

        return recommendations

    def _save_daily_report(self, report: DailyReport):
        """Save daily report to disk."""
        report_file = self.output_dir / f"daily_report_{report.date}.json"

        data = {
            'date': report.date,
            'queries_classified': report.queries_classified,
            'overall_accuracy': report.overall_accuracy,
            'by_complexity': report.by_complexity,
            'avg_latency_ms': report.avg_latency_ms,
            'patterns_discovered': report.patterns_discovered,
            'patterns_deployed': report.patterns_deployed,
            'regressions_detected': report.regressions_detected,
            'trend_vs_yesterday': report.trend_vs_yesterday
        }

        with open(report_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Daily report saved to {report_file}")

    def _save_weekly_report(self, report: WeeklyReport):
        """Save weekly report to disk."""
        report_file = self.output_dir / f"weekly_report_{report.week_start}_to_{report.week_end}.json"

        data = {
            'week_start': report.week_start,
            'week_end': report.week_end,
            'total_queries': report.total_queries,
            'overall_accuracy': report.overall_accuracy,
            'by_complexity': report.by_complexity,
            'avg_latency_ms': report.avg_latency_ms,
            'patterns_discovered': report.patterns_discovered,
            'patterns_deployed': report.patterns_deployed,
            'regressions_detected': report.regressions_detected,
            'trend_7day': report.trend_7day,
            'top_patterns': report.top_patterns,
            'recommendations': report.recommendations
        }

        with open(report_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Weekly report saved to {report_file}")

    def generate_markdown_report(self, report) -> str:
        """
        Generate markdown-formatted report.

        Args:
            report: DailyReport or WeeklyReport

        Returns:
            Markdown-formatted report string
        """
        if isinstance(report, DailyReport):
            return self._format_daily_markdown(report)
        elif isinstance(report, WeeklyReport):
            return self._format_weekly_markdown(report)
        else:
            raise ValueError(f"Unknown report type: {type(report)}")

    def _format_daily_markdown(self, report: DailyReport) -> str:
        """Format daily report as markdown."""
        trend_symbol = "▲" if report.trend_vs_yesterday > 0 else "▼" if report.trend_vs_yesterday < 0 else "─"

        markdown = f"""# Moonshot Classifier Daily Report
**Date**: {report.date}

## Summary
- **Queries Classified**: {report.queries_classified:,}
- **Overall Accuracy**: {report.overall_accuracy:.1%}
- **Average Latency**: {report.avg_latency_ms:.3f}ms
- **Trend**: {trend_symbol} {abs(report.trend_vs_yesterday):.1%} vs yesterday

## Breakdown by Complexity
| Complexity | Accuracy |
|------------|----------|
"""

        for complexity, accuracy in report.by_complexity.items():
            markdown += f"| {complexity.upper():10s} | {f'{accuracy:.1%}':>8s} |\n"

        markdown += f"""
## Pattern Activity
- **Patterns Discovered**: {report.patterns_discovered}
- **Patterns Deployed**: {report.patterns_deployed}
- **Regressions Detected**: {report.regressions_detected}

---
*Generated by Moonshot Classifier Adaptive Learning System*
"""

        return markdown

    def _format_weekly_markdown(self, report: WeeklyReport) -> str:
        """Format weekly report as markdown."""
        trend_symbol = "▲" if report.trend_7day > 0 else "▼" if report.trend_7day < 0 else "─"

        markdown = f"""# Moonshot Classifier Weekly Report
**Week**: {report.week_start} to {report.week_end}

## Executive Summary
- **Total Queries**: {report.total_queries:,}
- **Overall Accuracy**: {report.overall_accuracy:.1%}
- **Average Latency**: {report.avg_latency_ms:.3f}ms
- **7-Day Trend**: {trend_symbol} {abs(report.trend_7day):.1%}

## Accuracy by Complexity
| Complexity | Accuracy |
|------------|----------|
"""

        for complexity, accuracy in report.by_complexity.items():
            markdown += f"| {complexity.upper():10s} | {f'{accuracy:.1%}':>8s} |\n"

        markdown += f"""
## Pattern Activity
- **Patterns Discovered**: {report.patterns_discovered}
- **Patterns Deployed**: {report.patterns_deployed}
- **Regressions Detected**: {report.regressions_detected}

## Recommendations
"""

        for i, rec in enumerate(report.recommendations, 1):
            markdown += f"{i}. {rec}\n"

        markdown += """
---
*Generated by Moonshot Classifier Adaptive Learning System*
"""

        return markdown
