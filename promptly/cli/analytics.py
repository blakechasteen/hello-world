"""
Promptly Analytics CLI Commands

Commands for analyzing prompt performance, comparing prompts,
and viewing quality metrics.

Usage:
    promptly analytics stats <name> [--days 30] [--format text|json|html]
    promptly analytics compare <prompt_a> <prompt_b> [--task TYPE]
    promptly analytics trend <name> [--days 30] [--granularity daily]
    promptly analytics underperforming [--threshold 0.6]
    promptly analytics export [--prompts NAME...] [--format json|csv]
    promptly analytics dashboard [--prompt NAME] [--output FILE]
    promptly analytics history [name] [--limit 20] [--task TYPE]
"""

import click
import sys
import json
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

from promptly.core.database import PromptlyDatabase


def get_database() -> PromptlyDatabase:
    """Get database instance."""
    return PromptlyDatabase()


def get_analytics_engine(db: PromptlyDatabase):
    """Get analytics engine."""
    from promptly.analytics.engine import PromptAnalyticsEngine
    return PromptAnalyticsEngine(db)


# ==================== Analytics Command Group ====================

@click.group()
def analytics():
    """Prompt analytics and performance tracking.

    View quality metrics, compare prompts, and identify
    underperforming prompts.

    Examples:
        promptly analytics stats my_prompt --days 30
        promptly analytics compare prompt_a prompt_b
        promptly analytics underperforming --threshold 0.6
    """
    pass


# ==================== Stats Command ====================

@analytics.command()
@click.argument("name")
@click.option("--days", "-d", type=int, default=30, help="Analysis window in days")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json", "html"]), default="text", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file (optional)")
def stats(name: str, days: int, output_format: str, output: Optional[str]):
    """Get comprehensive analytics for a prompt.

    Examples:
        promptly analytics stats summarize --days 30
        promptly analytics stats my_prompt -f json -o stats.json
        promptly analytics stats my_prompt -f html -o report.html
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)
        analytics_result = engine.get_prompt_analytics(name, days=days)

        if not analytics_result:
            click.echo(f"No data found for prompt '{name}'", err=True)
            sys.exit(1)

        if output_format == "json":
            result = json.dumps(analytics_result.to_dict(), indent=2, default=str)
        elif output_format == "html":
            result = _render_analytics_html(analytics_result)
        else:
            result = _format_analytics_text(analytics_result)

        if output:
            Path(output).write_text(result)
            click.echo(f"Analytics saved to {output}")
        else:
            click.echo(result)

    finally:
        db.close()


def _format_analytics_text(analytics) -> str:
    """Format analytics as text."""
    lines = []
    lines.append(f"Analytics for: {analytics.prompt_name}")
    lines.append("=" * 50)
    lines.append("")

    # Basic metrics
    lines.append("Performance Metrics")
    lines.append("-" * 30)
    lines.append(f"Total Executions: {analytics.total_executions}")
    lines.append(f"Average Quality:  {analytics.avg_quality:.3f}")
    lines.append(f"Success Rate:     {analytics.success_rate:.1%}")
    lines.append(f"Average Latency:  {analytics.avg_latency_ms:.1f}ms")
    if analytics.avg_tokens:
        lines.append(f"Average Tokens:   {analytics.avg_tokens}")
    lines.append("")

    # Quality info
    lines.append("Quality Analysis")
    lines.append("-" * 30)
    lines.append(f"Trend:     {analytics.quality_trend.value.upper()}")
    lines.append(f"Min:       {analytics.quality_range['min']:.3f}")
    lines.append(f"Max:       {analytics.quality_range['max']:.3f}")
    lines.append(f"Std Dev:   {analytics.quality_range['std']:.3f}")
    if analytics.thompson_expected_quality:
        lines.append(f"Thompson:  {analytics.thompson_expected_quality:.3f} (expected)")
    lines.append("")

    # Version performance
    if analytics.version_performance:
        lines.append("Version Performance")
        lines.append("-" * 30)
        lines.append(f"{'Version':<10} {'Executions':<12} {'Quality':<10} {'Success':<10}")
        for v in analytics.version_performance[-5:]:  # Last 5 versions
            lines.append(f"v{v.version:<9} {v.total_executions:<12} {v.avg_quality:.3f}     {v.success_rate:.1%}")
        lines.append("")

    # Task distribution
    if analytics.task_type_distribution:
        lines.append("Task Distribution")
        lines.append("-" * 30)
        total = sum(analytics.task_type_distribution.values())
        for task_type, count in sorted(analytics.task_type_distribution.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"  {task_type:<20} {count:>5} ({pct:.1f}%)")
        lines.append("")

    # Recommendation
    if analytics.recommendation:
        lines.append("Recommendation")
        lines.append("-" * 30)
        lines.append(analytics.recommendation)

    return "\n".join(lines)


def _render_analytics_html(analytics) -> str:
    """Render analytics as HTML dashboard."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Promptly Analytics: {analytics.prompt_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .section {{ margin: 20px 0; }}
        .section-title {{ font-weight: bold; margin-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .trend-improving {{ color: #22c55e; }}
        .trend-declining {{ color: #ef4444; }}
        .trend-stable {{ color: #64748b; }}
        .recommendation {{ background: #fef3c7; padding: 15px; border-radius: 5px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analytics: {analytics.prompt_name}</h1>
        <p>Analysis period: {analytics.days_analyzed} days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="section">
        <div class="metric">
            <div class="metric-value">{analytics.total_executions}</div>
            <div class="metric-label">Total Executions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{analytics.avg_quality:.2f}</div>
            <div class="metric-label">Avg Quality</div>
        </div>
        <div class="metric">
            <div class="metric-value">{analytics.success_rate:.0%}</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{analytics.avg_latency_ms:.0f}ms</div>
            <div class="metric-label">Avg Latency</div>
        </div>
        <div class="metric">
            <div class="metric-value trend-{analytics.quality_trend.value}">{analytics.quality_trend.value.upper()}</div>
            <div class="metric-label">Quality Trend</div>
        </div>
    </div>
"""

    # Version performance table
    if analytics.version_performance:
        html += """
    <div class="section">
        <div class="section-title">Version Performance</div>
        <table>
            <tr><th>Version</th><th>Executions</th><th>Avg Quality</th><th>Success Rate</th><th>Avg Latency</th></tr>
"""
        for v in analytics.version_performance[-10:]:
            html += f"""            <tr><td>v{v.version}</td><td>{v.total_executions}</td><td>{v.avg_quality:.3f}</td><td>{v.success_rate:.1%}</td><td>{v.avg_latency_ms:.1f}ms</td></tr>
"""
        html += """        </table>
    </div>
"""

    # Daily breakdown sparkline (simple text representation)
    if analytics.daily_breakdown:
        html += """
    <div class="section">
        <div class="section-title">Daily Quality Trend</div>
        <table>
            <tr><th>Date</th><th>Executions</th><th>Avg Quality</th><th>Success Rate</th></tr>
"""
        for d in analytics.daily_breakdown[-14:]:  # Last 2 weeks
            html += f"""            <tr><td>{d.date}</td><td>{d.executions}</td><td>{d.avg_quality:.3f}</td><td>{d.success_rate:.1%}</td></tr>
"""
        html += """        </table>
    </div>
"""

    # Recommendation
    if analytics.recommendation:
        html += f"""
    <div class="recommendation">
        <strong>Recommendation:</strong> {analytics.recommendation}
    </div>
"""

    html += """
</body>
</html>
"""
    return html


# ==================== Compare Command ====================

@analytics.command()
@click.argument("prompt_a")
@click.argument("prompt_b")
@click.option("--task", "-t", help="Filter by task type")
@click.option("--days", "-d", type=int, default=30, help="Analysis window")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def compare(prompt_a: str, prompt_b: str, task: Optional[str], days: int, output_format: str):
    """Compare two prompts statistically.

    Performs A/B comparison with statistical significance testing.

    Examples:
        promptly analytics compare prompt_v1 prompt_v2
        promptly analytics compare old_prompt new_prompt --task summarization
        promptly analytics compare prompt_a prompt_b -f json
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)
        result = engine.compare_prompts(prompt_a, prompt_b, task_type=task, days=days)

        if not result:
            click.echo("Insufficient data for comparison", err=True)
            click.echo("Need at least 5 executions for each prompt.")
            sys.exit(1)

        if output_format == "json":
            click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            click.echo(f"Comparing: {prompt_a} vs {prompt_b}")
            click.echo("=" * 50)
            click.echo("")

            # Summary table
            click.echo(f"{'Metric':<20} {prompt_a:<15} {prompt_b:<15} {'Difference':<15}")
            click.echo("-" * 65)

            stats_a = result.prompt_a_stats
            stats_b = result.prompt_b_stats

            click.echo(f"{'Executions':<20} {stats_a.total_executions:<15} {stats_b.total_executions:<15}")
            click.echo(f"{'Avg Quality':<20} {stats_a.avg_quality:<15.3f} {stats_b.avg_quality:<15.3f} {result.statistics.quality_difference:+.3f}")
            click.echo(f"{'Success Rate':<20} {stats_a.success_rate:<15.1%} {stats_b.success_rate:<15.1%} {result.statistics.success_rate_difference:+.1%}")
            click.echo(f"{'Avg Latency (ms)':<20} {stats_a.avg_latency_ms:<15.1f} {stats_b.avg_latency_ms:<15.1f} {result.statistics.latency_difference_ms:+.1f}")
            click.echo("")

            # Statistical results
            click.echo("Statistical Analysis")
            click.echo("-" * 30)
            click.echo(f"Significant:  {'Yes' if result.is_significant else 'No'}")
            click.echo(f"P-value:      {result.statistics.statistical_significance:.4f}")
            if result.statistics.cohens_d:
                effect_size = "Large" if abs(result.statistics.cohens_d) > 0.8 else "Medium" if abs(result.statistics.cohens_d) > 0.5 else "Small"
                click.echo(f"Cohen's d:    {result.statistics.cohens_d:.3f} ({effect_size})")
            click.echo("")

            # Winner and recommendation
            if result.winner:
                click.echo(f"Winner: {result.winner}")
            else:
                click.echo("Winner: No clear winner")
            click.echo("")
            click.echo(f"Recommendation: {result.recommendation}")
            click.echo(f"Decision: {result.deployment_decision}")

    finally:
        db.close()


# ==================== Trend Command ====================

@analytics.command()
@click.argument("name")
@click.option("--days", "-d", type=int, default=30, help="Analysis window")
@click.option("--granularity", "-g", type=click.Choice(["daily", "hourly"]), default="daily")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def trend(name: str, days: int, granularity: str, output_format: str):
    """Show quality trend over time.

    Examples:
        promptly analytics trend my_prompt --days 7
        promptly analytics trend my_prompt --granularity hourly
        promptly analytics trend my_prompt -f json
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)
        trend_data = engine.get_quality_trend(name, days=days, granularity=granularity)

        if not trend_data:
            click.echo(f"No trend data found for '{name}'", err=True)
            sys.exit(1)

        if output_format == "json":
            click.echo(json.dumps(trend_data, indent=2))
        else:
            click.echo(f"Quality Trend: {name}")
            click.echo(f"Period: {days} days, Granularity: {granularity}")
            click.echo("=" * 60)
            click.echo("")

            time_col = "Date" if granularity == "daily" else "Hour"
            click.echo(f"{time_col:<20} {'Executions':<12} {'Avg Quality':<12} {'Range':<15}")
            click.echo("-" * 60)

            for point in trend_data:
                time_key = point.get("date", point.get("hour", ""))
                count = point["count"]
                avg_q = point["avg_quality"]
                min_q = point["min_quality"]
                max_q = point["max_quality"]
                range_str = f"[{min_q:.2f}-{max_q:.2f}]"
                click.echo(f"{time_key:<20} {count:<12} {avg_q:<12.3f} {range_str:<15}")

            # Simple ASCII sparkline
            if len(trend_data) >= 5:
                click.echo("")
                click.echo("Sparkline: ", nl=False)
                values = [p["avg_quality"] for p in trend_data]
                _print_sparkline(values)

    finally:
        db.close()


def _print_sparkline(values):
    """Print ASCII sparkline."""
    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1

    chars = "▁▂▃▄▅▆▇█"
    sparkline = ""
    for v in values:
        idx = int((v - min_val) / range_val * (len(chars) - 1))
        sparkline += chars[idx]

    click.echo(sparkline)


# ==================== Underperforming Command ====================

@analytics.command()
@click.option("--threshold", "-t", type=float, default=0.6, help="Quality threshold")
@click.option("--min-executions", "-m", type=int, default=10, help="Minimum executions")
@click.option("--days", "-d", type=int, default=30, help="Analysis window")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def underperforming(threshold: float, min_executions: int, days: int, output_format: str):
    """Identify underperforming prompts.

    Flags prompts with quality below threshold or declining trends.

    Examples:
        promptly analytics underperforming
        promptly analytics underperforming --threshold 0.7
        promptly analytics underperforming -f json
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)
        results = engine.identify_underperforming(
            threshold=threshold,
            min_executions=min_executions,
            days=days
        )

        if not results:
            click.echo("No underperforming prompts found!")
            return

        if output_format == "json":
            click.echo(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            click.echo(f"Underperforming Prompts (threshold: {threshold})")
            click.echo("=" * 70)
            click.echo("")

            for prompt in results:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(prompt.severity.value, "⚪")
                click.echo(f"{severity_icon} {prompt.prompt_name}")
                click.echo(f"   Quality: {prompt.avg_quality:.3f} | Success Rate: {prompt.success_rate:.1%} | Trend: {prompt.quality_trend.value}")
                click.echo(f"   Executions: {prompt.total_executions} | Severity: {prompt.severity.value.upper()}")
                if prompt.issues:
                    click.echo(f"   Issues: {', '.join(prompt.issues)}")
                if prompt.suggested_actions:
                    click.echo(f"   Actions: {', '.join(prompt.suggested_actions)}")
                click.echo("")

    finally:
        db.close()


# ==================== Export Command ====================

@analytics.command()
@click.option("--prompts", "-p", multiple=True, help="Specific prompts to export (default: all)")
@click.option("--days", "-d", type=int, default=30, help="Analysis window")
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "csv"]), default="json")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file")
def export(prompts: Tuple[str, ...], days: int, output_format: str, output: str):
    """Export analytics to file.

    Examples:
        promptly analytics export -o analytics.json
        promptly analytics export -p prompt1 -p prompt2 -o report.json
        promptly analytics export -f csv -o metrics.csv
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)
        prompt_list = list(prompts) if prompts else None

        export_data = engine.export_analytics(
            prompts=prompt_list,
            days=days,
            include_anomalies=True,
            include_thompson=True
        )

        if output_format == "csv":
            result = export_data.to_csv()
        else:
            result = export_data.to_json(indent=2)

        Path(output).write_text(result)
        click.echo(f"Exported analytics to {output}")
        click.echo(f"  Prompts: {export_data.total_prompts}")
        click.echo(f"  Executions: {export_data.total_executions}")
        click.echo(f"  Date Range: {export_data.date_range['start']} to {export_data.date_range['end']}")

    finally:
        db.close()


# ==================== History Command ====================

@analytics.command()
@click.argument("name", required=False)
@click.option("--limit", "-n", type=int, default=20, help="Number of records")
@click.option("--task", "-t", help="Filter by task type")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def history(name: Optional[str], limit: int, task: Optional[str], output_format: str):
    """Show execution history.

    Examples:
        promptly analytics history
        promptly analytics history my_prompt --limit 50
        promptly analytics history --task summarization
    """
    db = get_database()
    try:
        executions = db.get_executions(
            prompt_name=name,
            task_type=task,
            limit=limit
        )

        if not executions:
            click.echo("No execution history found")
            return

        if output_format == "json":
            click.echo(json.dumps([
                {
                    "id": e.id,
                    "prompt_name": e.prompt_name,
                    "version": e.version,
                    "task_type": e.task_type,
                    "quality_score": e.quality_score,
                    "latency_ms": e.latency_ms,
                    "created_at": e.created_at.isoformat()
                }
                for e in executions
            ], indent=2))
        else:
            title = f"Execution History: {name}" if name else "Recent Executions"
            click.echo(title)
            click.echo("=" * 80)
            click.echo("")
            click.echo(f"{'Time':<20} {'Prompt':<25} {'Ver':<5} {'Quality':<10} {'Latency':<10} {'Task'}")
            click.echo("-" * 80)

            for e in executions:
                time_str = e.created_at.strftime("%Y-%m-%d %H:%M")
                prompt = e.prompt_name[:23] + ".." if len(e.prompt_name) > 25 else e.prompt_name
                quality = f"{e.quality_score:.3f}" if e.quality_score else "N/A"
                latency = f"{e.latency_ms:.0f}ms" if e.latency_ms else "N/A"
                task_type = (e.task_type or "")[:15]
                click.echo(f"{time_str:<20} {prompt:<25} v{e.version:<4} {quality:<10} {latency:<10} {task_type}")

    finally:
        db.close()


# ==================== Anomalies Command ====================

@analytics.command()
@click.argument("name")
@click.option("--days", "-d", type=int, default=30, help="Analysis window")
@click.option("--sensitivity", "-s", type=float, default=2.0, help="Detection sensitivity (std devs)")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def anomalies(name: str, days: int, sensitivity: float, output_format: str):
    """Detect quality anomalies for a prompt.

    Identifies sudden drops, prolonged low quality, high variance,
    and outliers in execution quality.

    Examples:
        promptly analytics anomalies my_prompt
        promptly analytics anomalies my_prompt --sensitivity 1.5
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)
        detected = engine.detect_quality_anomalies(name, days=days, sensitivity=sensitivity)

        if not detected:
            click.echo(f"No anomalies detected for '{name}'")
            return

        if output_format == "json":
            click.echo(json.dumps([a.to_dict() for a in detected], indent=2, default=str))
        else:
            click.echo(f"Anomalies Detected: {name}")
            click.echo("=" * 60)
            click.echo("")

            for anomaly in detected:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(anomaly.severity.value, "⚪")
                click.echo(f"{severity_icon} [{anomaly.anomaly_type.upper()}] {anomaly.timestamp.strftime('%Y-%m-%d %H:%M')}")
                click.echo(f"   {anomaly.description}")
                click.echo(f"   Affected executions: {len(anomaly.affected_executions)}")
                click.echo(f"   Action: {anomaly.suggested_action}")
                click.echo("")

    finally:
        db.close()


# ==================== Thompson Command ====================

@analytics.command()
@click.argument("task_type")
@click.option("--candidates", "-c", multiple=True, help="Candidate prompts to consider")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def recommend(task_type: str, candidates: Tuple[str, ...], output_format: str):
    """Get Thompson Sampling recommendation for a task type.

    Returns the best prompt for the given task type based on
    historical performance and Bayesian optimization.

    Examples:
        promptly analytics recommend summarization
        promptly analytics recommend code_review -c prompt_a -c prompt_b
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)
        candidate_list = list(candidates) if candidates else None

        rec = engine.get_thompson_recommendation(task_type, candidates=candidate_list)

        if not rec:
            click.echo(f"No recommendation available for task type '{task_type}'")
            click.echo("Run some prompts with this task type first.")
            return

        if output_format == "json":
            click.echo(json.dumps(rec.to_dict(), indent=2))
        else:
            click.echo(f"Thompson Sampling Recommendation")
            click.echo(f"Task Type: {task_type}")
            click.echo("=" * 50)
            click.echo("")
            click.echo(f"Recommended: {rec.recommended_prompt}")
            click.echo(f"Expected Quality: {rec.expected_quality:.3f}")
            click.echo(f"Confidence: {rec.confidence:.1%}")
            click.echo("")
            click.echo(f"Reasoning: {rec.reasoning}")

            if rec.alternatives:
                click.echo("")
                click.echo("Alternatives:")
                for alt in rec.alternatives:
                    click.echo(f"  - {alt['name']}: expected {alt['expected_quality']:.3f}")

    finally:
        db.close()


# ==================== Dashboard Command ====================

@analytics.command()
@click.option("--prompt", "-p", help="Specific prompt (default: overview)")
@click.option("--days", "-d", type=int, default=30, help="Analysis window")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output HTML file")
def dashboard(prompt: Optional[str], days: int, output: str):
    """Generate HTML analytics dashboard.

    Creates a comprehensive HTML dashboard with visualizations.

    Examples:
        promptly analytics dashboard -o dashboard.html
        promptly analytics dashboard -p my_prompt -o report.html
    """
    db = get_database()
    try:
        engine = get_analytics_engine(db)

        if prompt:
            # Single prompt dashboard
            analytics_result = engine.get_prompt_analytics(prompt, days=days)
            if not analytics_result:
                click.echo(f"No data for prompt '{prompt}'", err=True)
                sys.exit(1)
            html = _render_analytics_html(analytics_result)
        else:
            # Overview dashboard
            export_data = engine.export_analytics(days=days)
            html = _render_overview_dashboard(export_data)

        Path(output).write_text(html)
        click.echo(f"Dashboard saved to {output}")

    finally:
        db.close()


def _render_overview_dashboard(export_data) -> str:
    """Render overview dashboard HTML."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Promptly Analytics Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; background: #f8fafc; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric {{ font-size: 32px; font-weight: bold; color: #2563eb; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 10px; text-align: left; }}
        th {{ background: #f9fafb; }}
        .severity-critical {{ color: #dc2626; }}
        .severity-high {{ color: #ea580c; }}
        .severity-medium {{ color: #ca8a04; }}
        .severity-low {{ color: #2563eb; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Promptly Analytics Dashboard</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Period: {export_data.date_range['start']} to {export_data.date_range['end']}</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="metric">{export_data.total_prompts}</div>
            <div class="metric-label">Total Prompts</div>
        </div>
        <div class="card">
            <div class="metric">{export_data.total_executions}</div>
            <div class="metric-label">Total Executions</div>
        </div>
        <div class="card">
            <div class="metric">{len(export_data.underperforming)}</div>
            <div class="metric-label">Underperforming</div>
        </div>
        <div class="card">
            <div class="metric">{len(export_data.anomalies)}</div>
            <div class="metric-label">Anomalies Detected</div>
        </div>
    </div>
"""

    # Prompt summary table
    if export_data.prompts:
        html += """
    <div class="card" style="margin-top: 20px;">
        <h2>Prompt Performance</h2>
        <table>
            <tr><th>Prompt</th><th>Executions</th><th>Avg Quality</th><th>Success Rate</th><th>Trend</th></tr>
"""
        for p in sorted(export_data.prompts, key=lambda x: -x.total_executions)[:20]:
            trend_class = f"severity-{'low' if p.quality_trend.value == 'improving' else 'medium' if p.quality_trend.value == 'stable' else 'high'}"
            html += f"""            <tr><td>{p.prompt_name}</td><td>{p.total_executions}</td><td>{p.avg_quality:.3f}</td><td>{p.success_rate:.1%}</td><td class="{trend_class}">{p.quality_trend.value}</td></tr>
"""
        html += """        </table>
    </div>
"""

    # Underperforming section
    if export_data.underperforming:
        html += """
    <div class="card" style="margin-top: 20px;">
        <h2>Underperforming Prompts</h2>
        <table>
            <tr><th>Prompt</th><th>Quality</th><th>Success Rate</th><th>Severity</th><th>Issues</th></tr>
"""
        for u in export_data.underperforming:
            html += f"""            <tr><td>{u.prompt_name}</td><td>{u.avg_quality:.3f}</td><td>{u.success_rate:.1%}</td><td class="severity-{u.severity.value}">{u.severity.value.upper()}</td><td>{', '.join(u.issues[:2])}</td></tr>
"""
        html += """        </table>
    </div>
"""

    html += """
</body>
</html>
"""
    return html
