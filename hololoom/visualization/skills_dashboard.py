"""
Skills Metrics Dashboard Generator
===================================

Creates interactive HTML dashboards from skill metrics JSON export.

Features:
- Overall performance summary
- Per-skill breakdown tables
- Top performers identification
- Problem area detection (slow/failing/degrading skills)
- Time series visualization
- Zero dependencies (pure HTML/CSS/SVG)

Usage:
    from hololoom.visualization.skills_dashboard import render_dashboard
    from hololoom.skills.metrics import get_tracker

    tracker = get_tracker()
    html = render_dashboard(tracker.export_json())

    with open('dashboard.html', 'w') as f:
        f.write(html)
"""

from typing import Any


def _format_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1:
        return f"{ms:.2f}ms"
    elif ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def _format_rate(rate: float) -> str:
    """Format rate as percentage."""
    return f"{rate*100:.1f}%"


def _render_summary_card(summary: dict[str, Any]) -> str:
    """Render overall summary card."""
    html = f"""
    <div class="summary-card">
        <h2>Overall Performance</h2>
        <div class="summary-grid">
            <div class="metric">
                <div class="metric-label">Total Skills</div>
                <div class="metric-value">{summary.get('total_skills', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Executions</div>
                <div class="metric-value">{summary.get('total_executions', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value {_get_success_class(summary.get('success_rate', 0))}">{_format_rate(summary.get('success_rate', 0))}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Execution Time</div>
                <div class="metric-value">{_format_duration(summary.get('avg_time_ms', 0))}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Cache Hit Rate</div>
                <div class="metric-value {_get_cache_class(summary.get('cache_hit_rate', 0))}">{_format_rate(summary.get('cache_hit_rate', 0))}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Uptime</div>
                <div class="metric-value">{_format_duration(summary.get('uptime_seconds', 0) * 1000)}</div>
            </div>
        </div>
    </div>
    """
    return html


def _get_success_class(rate: float) -> str:
    """Get CSS class for success rate."""
    if rate >= 0.9:
        return "success"
    elif rate >= 0.7:
        return "warning"
    else:
        return "error"


def _get_cache_class(rate: float) -> str:
    """Get CSS class for cache hit rate."""
    if rate >= 0.7:
        return "success"
    elif rate >= 0.4:
        return "warning"
    else:
        return "error"


def _render_skills_table(skills: dict[str, dict[str, Any]]) -> str:
    """Render skills performance table."""
    if not skills:
        return "<p>No skill executions recorded yet.</p>"

    # Sort by total executions (descending)
    sorted_skills = sorted(
        skills.items(),
        key=lambda x: x[1].get('total_executions', 0),
        reverse=True
    )

    rows = []
    for skill_name, metrics in sorted_skills:
        success_rate = metrics.get('success_rate', 0)
        cache_hit_rate = metrics.get('cache_hit_rate', 0)

        row = f"""
        <tr>
            <td class="skill-name">{skill_name}</td>
            <td class="number">{metrics.get('total_executions', 0)}</td>
            <td class="number {_get_success_class(success_rate)}">{_format_rate(success_rate)}</td>
            <td class="number">{_format_duration(metrics.get('avg_time_ms', 0))}</td>
            <td class="number">{_format_duration(metrics.get('min_time_ms', 0) or 0)}</td>
            <td class="number">{_format_duration(metrics.get('max_time_ms', 0) or 0)}</td>
            <td class="number {_get_cache_class(cache_hit_rate)}">{_format_rate(cache_hit_rate)}</td>
            <td class="number">{metrics.get('successful_executions', 0)}/{metrics.get('failed_executions', 0)}/{metrics.get('timeout_executions', 0)}</td>
        </tr>
        """
        rows.append(row)

    html = f"""
    <div class="skills-table">
        <h2>Skill Performance Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Skill Name</th>
                    <th>Executions</th>
                    <th>Success Rate</th>
                    <th>Avg Time</th>
                    <th>Min Time</th>
                    <th>Max Time</th>
                    <th>Cache Hit Rate</th>
                    <th>Success/Fail/Timeout</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """
    return html


def _render_top_skills(skills: dict[str, dict[str, Any]]) -> str:
    """Render top skills by various metrics."""
    if not skills:
        return ""

    # Top by executions
    by_executions = sorted(
        skills.items(),
        key=lambda x: x[1].get('total_executions', 0),
        reverse=True
    )[:5]

    # Top by success rate (min 5 executions)
    by_success = sorted(
        [(name, m) for name, m in skills.items() if m.get('total_executions', 0) >= 5],
        key=lambda x: x[1].get('success_rate', 0),
        reverse=True
    )[:5]

    # Fastest (min 5 executions)
    by_speed = sorted(
        [(name, m) for name, m in skills.items() if m.get('total_executions', 0) >= 5],
        key=lambda x: x[1].get('avg_time_ms', float('inf'))
    )[:5]

    html = f"""
    <div class="top-skills">
        <h2>Top Performers</h2>
        <div class="top-skills-grid">
            <div class="top-list">
                <h3>Most Used</h3>
                <ol>
                    {''.join(f'<li>{name} ({m["total_executions"]} executions)</li>' for name, m in by_executions)}
                </ol>
            </div>
            <div class="top-list">
                <h3>Most Reliable</h3>
                <ol>
                    {''.join(f'<li>{name} ({_format_rate(m["success_rate"])})</li>' for name, m in by_success)}
                </ol>
            </div>
            <div class="top-list">
                <h3>Fastest</h3>
                <ol>
                    {''.join(f'<li>{name} ({_format_duration(m["avg_time_ms"])})</li>' for name, m in by_speed)}
                </ol>
            </div>
        </div>
    </div>
    """
    return html


def _render_problem_areas(skills: dict[str, dict[str, Any]]) -> str:
    """Render problem area detection."""
    if not skills:
        return ""

    # Find slow skills (>1000ms avg, min 3 executions)
    slow = sorted(
        [(name, m) for name, m in skills.items()
         if m.get('avg_time_ms', 0) > 1000 and m.get('total_executions', 0) >= 3],
        key=lambda x: x[1].get('avg_time_ms', 0),
        reverse=True
    )

    # Find failing skills (success rate <70%, min 5 executions)
    failing = sorted(
        [(name, m) for name, m in skills.items()
         if m.get('success_rate', 1.0) < 0.7 and m.get('total_executions', 0) >= 5],
        key=lambda x: x[1].get('success_rate', 1.0)
    )

    # Find low cache efficiency (<40%, min 10 requests)
    low_cache = sorted(
        [(name, m) for name, m in skills.items()
         if m.get('cache_hit_rate', 1.0) < 0.4
         and (m.get('cache_hits', 0) + m.get('cache_misses', 0)) >= 10],
        key=lambda x: x[1].get('cache_hit_rate', 1.0)
    )

    if not (slow or failing or low_cache):
        return """
        <div class="problem-areas">
            <h2>Problem Areas</h2>
            <p class="success">✓ No significant issues detected</p>
        </div>
        """

    html = """
    <div class="problem-areas">
        <h2>Problem Areas</h2>
        <div class="problem-grid">
    """

    if slow:
        html += """
            <div class="problem-list">
                <h3>⚠ Slow Skills (&gt;1s avg)</h3>
                <ul>
        """
        for name, m in slow:
            html += f'<li>{name}: {_format_duration(m["avg_time_ms"])} avg</li>'
        html += """
                </ul>
            </div>
        """

    if failing:
        html += """
            <div class="problem-list">
                <h3>⚠ Failing Skills (&lt;70% success)</h3>
                <ul>
        """
        for name, m in failing:
            html += f'<li>{name}: {_format_rate(m["success_rate"])} success rate</li>'
        html += """
                </ul>
            </div>
        """

    if low_cache:
        html += """
            <div class="problem-list">
                <h3>⚠ Low Cache Efficiency (&lt;40%)</h3>
                <ul>
        """
        for name, m in low_cache:
            html += f'<li>{name}: {_format_rate(m["cache_hit_rate"])} hit rate</li>'
        html += """
                </ul>
            </div>
        """

    html += """
        </div>
    </div>
    """
    return html


def render_dashboard(metrics_json: dict[str, Any], title: str = "Skills Metrics Dashboard") -> str:
    """
    Render complete HTML dashboard from metrics JSON.

    Args:
        metrics_json: Output from SkillMetricsTracker.export_json()
        title: Dashboard title

    Returns:
        Complete HTML document as string
    """
    summary = metrics_json.get('summary', {})
    skills = metrics_json.get('skills', {})
    timestamp = metrics_json.get('timestamp', 'Unknown')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            color: white;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .timestamp {{
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}

        .summary-card, .skills-table, .top-skills, .problem-areas {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}

        h3 {{
            color: #555;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}

        .metric {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}

        .metric-value.success {{
            color: #10b981;
        }}

        .metric-value.warning {{
            color: #f59e0b;
        }}

        .metric-value.error {{
            color: #ef4444;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}

        thead {{
            background: #667eea;
            color: white;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
        }}

        th {{
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}

        tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}

        tbody tr:hover {{
            background: #e5e7eb;
        }}

        .skill-name {{
            font-weight: 600;
            color: #667eea;
        }}

        .number {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .top-skills-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}

        .top-list {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }}

        .top-list ol {{
            list-style-position: inside;
            padding-left: 0;
        }}

        .top-list li {{
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }}

        .problem-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .problem-list {{
            background: #fef2f2;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ef4444;
        }}

        .problem-list h3 {{
            color: #dc2626;
            margin-bottom: 10px;
        }}

        .problem-list ul {{
            list-style: none;
            padding-left: 0;
        }}

        .problem-list li {{
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 4px;
        }}

        p.success {{
            color: #10b981;
            font-weight: 600;
            text-align: center;
            font-size: 1.1em;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="timestamp">Generated: {timestamp}</div>

        {_render_summary_card(summary)}
        {_render_top_skills(skills)}
        {_render_skills_table(skills)}
        {_render_problem_areas(skills)}
    </div>
</body>
</html>"""

    return html


__all__ = ['render_dashboard']
