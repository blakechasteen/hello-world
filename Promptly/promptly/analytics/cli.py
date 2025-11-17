"""
Analytics CLI - Command-line interface for Promptly analytics

Provides commands to view stats, generate reports, and manage analytics data.
"""

import click
import json
from pathlib import Path
from datetime import datetime, timedelta

from .performance import PerformanceMonitor
from .usage import UsageAnalytics
from .quality import QualityMetrics
from .visualize import Visualizer
from .reports import ReportGenerator, ReportConfig, ReportPeriod, ReportFormat
from .integrations import GrafanaExporter


@click.group()
def analytics_cli():
    """Promptly Analytics - View statistics and generate reports"""
    pass


@analytics_cli.group()
def stats():
    """View analytics statistics"""
    pass


@stats.command(name='performance')
@click.option('--db', default='.promptly/analytics/performance.db', help='Performance database path')
@click.option('--operation', help='Filter by operation name')
def stats_performance(db, operation):
    """View performance statistics"""
    perf = PerformanceMonitor(db)

    click.echo(click.style("\n=== Performance Statistics ===\n", fg='cyan', bold=True))

    # Operation stats
    op_stats = perf.get_operation_stats(operation)

    if operation:
        if not op_stats:
            click.echo(click.style(f"No data for operation: {operation}", fg='yellow'))
            return

        stats = op_stats
        click.echo(f"Operation: {click.style(operation, fg='green', bold=True)}")
        click.echo(f"  Count:    {stats['count']}")
        click.echo(f"  Avg:      {stats['avg_ms']:.2f} ms")
        click.echo(f"  Min:      {stats['min_ms']:.2f} ms")
        click.echo(f"  Max:      {stats['max_ms']:.2f} ms")

        # Percentiles
        percentiles = perf.get_operation_percentiles(operation)
        if percentiles:
            click.echo(f"\nPercentiles:")
            for p, val in percentiles.items():
                click.echo(f"  {p}: {val:.2f} ms")
    else:
        if not op_stats:
            click.echo(click.style("No performance data available", fg='yellow'))
            return

        click.echo(click.style("Operation Statistics:", fg='green'))
        click.echo(f"\n{'Operation':<20} {'Count':>8} {'Avg (ms)':>10} {'Max (ms)':>10}")
        click.echo("-" * 52)

        for op, stats in sorted(op_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            click.echo(f"{op:<20} {stats['count']:>8} {stats['avg_ms']:>10.2f} {stats['max_ms']:>10.2f}")

    # Resource stats
    click.echo(click.style("\nResource Usage (24h):", fg='green'))
    res_stats = perf.get_resource_stats(hours=24)
    if res_stats:
        click.echo(f"  Avg CPU:    {res_stats.get('avg_cpu_percent', 0):.2f}%")
        click.echo(f"  Max CPU:    {res_stats.get('max_cpu_percent', 0):.2f}%")
        click.echo(f"  Avg Memory: {res_stats.get('avg_memory_mb', 0):.2f} MB")
        click.echo(f"  Max Memory: {res_stats.get('max_memory_mb', 0):.2f} MB")
    else:
        click.echo("  No resource data available")

    # Throughput
    click.echo(click.style("\nThroughput (24h):", fg='green'))
    throughput = perf.get_throughput(hours=24)
    click.echo(f"  Total Ops:  {throughput['total_operations']}")
    click.echo(f"  Ops/sec:    {throughput['operations_per_second']:.3f}")
    click.echo(f"  Ops/min:    {throughput['operations_per_minute']:.2f}")
    click.echo(f"  Ops/hour:   {throughput['operations_per_hour']:.2f}")

    click.echo("")


@stats.command(name='usage')
@click.option('--db', default='.promptly/analytics/usage.db', help='Usage database path')
@click.option('--days', default=7, help='Number of days to analyze')
def stats_usage(db, days):
    """View usage statistics"""
    usage = UsageAnalytics(db)

    click.echo(click.style(f"\n=== Usage Statistics ({days} days) ===\n", fg='cyan', bold=True))

    # Most used prompts
    most_used = usage.get_most_used_prompts(days=days)
    if most_used:
        click.echo(click.style("Most Used Prompts:", fg='green'))
        click.echo(f"\n{'Prompt':<30} {'Branch':<15} {'Accesses':>10}")
        click.echo("-" * 57)
        for p in most_used[:10]:
            click.echo(f"{p['prompt_name']:<30} {p['branch']:<15} {p['access_count']:>10}")
    else:
        click.echo(click.style("No usage data available", fg='yellow'))
        return

    # Branch activity
    click.echo(click.style("\nBranch Activity:", fg='green'))
    branch_activity = usage.get_branch_activity(days=days)
    if branch_activity:
        click.echo(f"\n{'Branch':<20} {'Events':>10} {'Prompts':>10} {'Days':>8}")
        click.echo("-" * 50)
        for b in branch_activity:
            click.echo(f"{b['branch']:<20} {b['total_events']:>10} {b['unique_prompts']:>10} {b['active_days']:>8}")

    # Evaluations
    click.echo(click.style("\nEvaluations:", fg='green'))
    eval_stats = usage.get_evaluation_stats(days=days)
    click.echo(f"  Total:      {eval_stats['total_evaluations']}")
    click.echo(f"  Avg Score:  {eval_stats.get('average_score', 0):.3f}")
    click.echo(f"  Per Day:    {eval_stats['evaluations_per_day']:.2f}")

    # Chains
    click.echo(click.style("\nChain Executions:", fg='green'))
    chain_stats = usage.get_chain_stats(days=days)
    click.echo(f"  Total:       {chain_stats['total_executions']}")
    click.echo(f"  Success Rate: {chain_stats['success_rate']:.2f}%")
    click.echo(f"  Per Day:     {chain_stats['executions_per_day']:.2f}")

    click.echo("")


@stats.command(name='quality')
@click.option('--db', default='.promptly/analytics/quality.db', help='Quality database path')
@click.option('--days', default=30, help='Number of days to analyze')
@click.option('--prompt', help='Specific prompt to analyze')
def stats_quality(db, days, prompt):
    """View quality statistics"""
    quality = QualityMetrics(db)

    click.echo(click.style(f"\n=== Quality Statistics ({days} days) ===\n", fg='cyan', bold=True))

    if prompt:
        # Show detailed stats for specific prompt
        stats = quality.get_prompt_quality_stats(prompt, days=days)
        if not stats:
            click.echo(click.style(f"No quality data for: {prompt}", fg='yellow'))
            return

        click.echo(f"Prompt: {click.style(prompt, fg='green', bold=True)}")
        click.echo(f"  Measurements: {stats['measurements']}")
        click.echo(f"  Avg Score:    {stats['avg_score']:.3f}")
        click.echo(f"  Min Score:    {stats['min_score']:.3f}")
        click.echo(f"  Max Score:    {stats['max_score']:.3f}")
        click.echo(f"  Std Dev:      {stats['std_dev']:.3f}")

        # Trend
        trend = quality.get_quality_trend(prompt, days=days)
        if trend:
            click.echo(f"\nTrend:")
            trend_color = 'green' if trend.trend_direction == 'improving' else 'red' if trend.trend_direction == 'declining' else 'yellow'
            click.echo(f"  Direction: {click.style(trend.trend_direction.upper(), fg=trend_color)}")
            click.echo(f"  Slope:     {trend.trend_slope:.4f}")
    else:
        # Show overall quality stats
        top = quality.get_top_performing_prompts(days=days)
        if top:
            click.echo(click.style("Top Performing Prompts:", fg='green'))
            click.echo(f"\n{'Prompt':<30} {'Branch':<15} {'Avg Score':>10} {'Samples':>8}")
            click.echo("-" * 65)
            for p in top[:10]:
                click.echo(f"{p['prompt_name']:<30} {p['branch']:<15} {p['avg_score']:>10.3f} {p['measurements']:>8}")

        # Declining prompts
        declining = quality.get_declining_prompts(days=days)
        if declining:
            click.echo(click.style("\nDeclining Quality (⚠️  Warning):", fg='yellow'))
            click.echo(f"\n{'Prompt':<30} {'Branch':<15} {'Trend':>10} {'Avg':>10}")
            click.echo("-" * 67)
            for p in declining[:5]:
                trend_value = p['trend_slope']
                trend_colored = click.style(f'{trend_value:.4f}', fg='red')
                click.echo(f"{p['prompt_name']:<30} {p['branch']:<15} {trend_colored:>10} {p['avg_score']:>10.3f}")

        # Active alerts
        alerts = quality.get_active_alerts()
        if alerts:
            click.echo(click.style("\nActive Alerts (🚨):", fg='red'))
            for a in alerts[:5]:
                severity_color = 'red' if a['severity'] == 'error' else 'yellow'
                click.echo(f"  [{click.style(a['severity'].upper(), fg=severity_color)}] {a['prompt_name']}: {a['message']}")

    click.echo("")


@analytics_cli.group()
def report():
    """Generate analytics reports"""
    pass


@report.command(name='daily')
@click.option('--output', '-o', default='./reports/daily.md', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['markdown', 'html', 'json', 'text']), default='markdown')
def report_daily(output, format):
    """Generate daily summary report"""
    click.echo(click.style("Generating daily report...", fg='cyan'))

    perf = PerformanceMonitor('.promptly/analytics/performance.db')
    usage = UsageAnalytics('.promptly/analytics/usage.db')
    quality = QualityMetrics('.promptly/analytics/quality.db')
    viz = Visualizer(perf, usage, quality)

    generator = ReportGenerator(perf, usage, quality, viz)

    config = ReportConfig(
        period=ReportPeriod.DAILY,
        format=ReportFormat(format)
    )

    output_path = generator.generate_report(config, output)

    click.echo(click.style(f"✓ Report generated: {output_path}", fg='green'))


@report.command(name='weekly')
@click.option('--output', '-o', default='./reports/weekly.md', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['markdown', 'html', 'json', 'text']), default='markdown')
def report_weekly(output, format):
    """Generate weekly summary report"""
    click.echo(click.style("Generating weekly report...", fg='cyan'))

    perf = PerformanceMonitor('.promptly/analytics/performance.db')
    usage = UsageAnalytics('.promptly/analytics/usage.db')
    quality = QualityMetrics('.promptly/analytics/quality.db')
    viz = Visualizer(perf, usage, quality)

    generator = ReportGenerator(perf, usage, quality, viz)

    config = ReportConfig(
        period=ReportPeriod.WEEKLY,
        format=ReportFormat(format)
    )

    output_path = generator.generate_report(config, output)

    click.echo(click.style(f"✓ Report generated: {output_path}", fg='green'))


@report.command(name='dashboard')
@click.option('--output', '-o', default='./reports/dashboard.html', help='Output file path')
@click.option('--days', default=7, help='Number of days to include')
def report_dashboard(output, days):
    """Generate HTML dashboard"""
    click.echo(click.style("Generating dashboard...", fg='cyan'))

    perf = PerformanceMonitor('.promptly/analytics/performance.db')
    usage = UsageAnalytics('.promptly/analytics/usage.db')
    quality = QualityMetrics('.promptly/analytics/quality.db')
    viz = Visualizer(perf, usage, quality)

    output_path = viz.generate_dashboard(output, days=days)

    click.echo(click.style(f"✓ Dashboard generated: {output_path}", fg='green'))


@analytics_cli.group()
def export():
    """Export analytics data"""
    pass


@export.command(name='csv')
@click.argument('data_type', type=click.Choice(['performance', 'usage', 'quality']))
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--days', default=30, help='Number of days to export')
def export_csv(data_type, output, days):
    """Export analytics data to CSV"""
    click.echo(click.style(f"Exporting {data_type} data to CSV...", fg='cyan'))

    perf = PerformanceMonitor('.promptly/analytics/performance.db')
    usage = UsageAnalytics('.promptly/analytics/usage.db')
    quality = QualityMetrics('.promptly/analytics/quality.db')
    viz = Visualizer(perf, usage, quality)

    output_path = viz.export_to_csv(data_type, output, days=days)

    click.echo(click.style(f"✓ Data exported: {output_path}", fg='green'))


@export.command(name='json')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--days', default=30, help='Number of days to export')
def export_json(output, days):
    """Export all analytics data to JSON"""
    click.echo(click.style(f"Exporting analytics data to JSON...", fg='cyan'))

    perf = PerformanceMonitor('.promptly/analytics/performance.db')
    usage = UsageAnalytics('.promptly/analytics/usage.db')
    quality = QualityMetrics('.promptly/analytics/quality.db')
    viz = Visualizer(perf, usage, quality)

    output_path = viz.export_to_json(output, days=days)

    click.echo(click.style(f"✓ Data exported: {output_path}", fg='green'))


@export.command(name='grafana')
@click.option('--output-dir', '-o', default='./grafana', help='Output directory')
def export_grafana(output_dir):
    """Export Grafana dashboard configuration"""
    click.echo(click.style(f"Exporting Grafana dashboard...", fg='cyan'))

    exporter = GrafanaExporter(output_dir)
    dashboard_path = exporter.export_dashboard()
    config_path = exporter.export_prometheus_config()

    click.echo(click.style(f"✓ Dashboard exported: {dashboard_path}", fg='green'))
    click.echo(click.style(f"✓ Prometheus config: {config_path}", fg='green'))


@analytics_cli.command()
@click.option('--performance/--no-performance', default=True)
@click.option('--usage/--no-usage', default=True)
@click.option('--quality/--no-quality', default=True)
def cleanup(performance, usage, quality):
    """Clean up old analytics data"""
    click.echo(click.style("Cleaning up old analytics data...", fg='cyan'))

    total_deleted = 0

    if performance:
        perf = PerformanceMonitor('.promptly/analytics/performance.db')
        deleted = perf.cleanup_old_metrics()
        click.echo(f"  Performance: {deleted} records deleted")
        total_deleted += deleted

    if usage:
        usage_analytics = UsageAnalytics('.promptly/analytics/usage.db')
        deleted = usage_analytics.cleanup_old_events()
        click.echo(f"  Usage:       {deleted} records deleted")
        total_deleted += deleted

    if quality:
        quality_metrics = QualityMetrics('.promptly/analytics/quality.db')
        deleted = quality_metrics.cleanup_old_measurements()
        click.echo(f"  Quality:     {deleted} records deleted")
        total_deleted += deleted

    click.echo(click.style(f"\n✓ Total records deleted: {total_deleted}", fg='green'))


if __name__ == '__main__':
    analytics_cli()
