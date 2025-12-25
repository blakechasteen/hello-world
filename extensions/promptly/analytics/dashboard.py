"""
Promptly Analytics Dashboard

Tufte-style HTML visualization dashboard for prompt analytics.
Implements high data-ink ratio, minimal chartjunk, and sparklines.

Usage:
    from promptly.analytics.dashboard import PromptlyDashboard

    dashboard = PromptlyDashboard(db)
    html = dashboard.render_prompt_report("my_prompt")
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import html as html_escape

from promptly.core.database import PromptlyDatabase
from promptly.analytics.engine import PromptAnalyticsEngine


@dataclass
class DashboardConfig:
    """Dashboard configuration options."""

    # Chart dimensions
    sparkline_width: int = 100
    sparkline_height: int = 20

    # Color scheme (muted, professional)
    color_primary: str = "#2c3e50"
    color_secondary: str = "#7f8c8d"
    color_success: str = "#27ae60"
    color_warning: str = "#f39c12"
    color_danger: str = "#e74c3c"
    color_info: str = "#3498db"

    # Typography
    font_family: str = "system-ui, -apple-system, 'Segoe UI', sans-serif"
    font_mono: str = "'SF Mono', 'Monaco', 'Consolas', monospace"

    # Data density
    max_sparkline_points: int = 30
    trend_days: int = 30


class PromptlyDashboard:
    """
    Tufte-style HTML dashboard for prompt analytics.

    Design principles:
    - High data-ink ratio (~70%)
    - Sparklines for inline trends
    - Small multiples for comparison
    - Minimal chartjunk
    """

    def __init__(
        self,
        db: PromptlyDatabase,
        config: Optional[DashboardConfig] = None
    ):
        """
        Initialize dashboard.

        Args:
            db: Database instance
            config: Optional dashboard configuration
        """
        self.db = db
        self.engine = PromptAnalyticsEngine(db)
        self.config = config or DashboardConfig()

    def render_prompt_report(
        self,
        prompt_name: str,
        days: int = 30
    ) -> str:
        """
        Render comprehensive HTML report for a single prompt.

        Args:
            prompt_name: Name of the prompt to analyze
            days: Analysis window in days

        Returns:
            Complete HTML document as string
        """
        # Get analytics data
        analytics = self.engine.get_prompt_analytics(prompt_name, days)
        trend_data = self.engine.get_quality_trend(prompt_name, "daily", days)

        if not analytics:
            return self._render_empty_report(prompt_name)

        # Build HTML
        html_parts = [
            self._render_html_head(f"Prompt Report: {prompt_name}"),
            '<body>',
            self._render_header(f"Prompt Report: {prompt_name}"),

            # Summary metrics row
            '<div class="metrics-row">',
            self._render_metric_card("Total Executions", str(analytics.total_executions), icon="📊"),
            self._render_metric_card(
                "Avg Quality",
                f"{analytics.avg_quality:.1%}",
                trend=analytics.quality_trend,
                icon="⭐"
            ),
            self._render_metric_card(
                "Success Rate",
                f"{analytics.success_rate:.1%}",
                icon="✓"
            ),
            self._render_metric_card(
                "Avg Latency",
                f"{analytics.avg_latency_ms:.0f}ms",
                icon="⚡"
            ),
            '</div>',

            # Quality trajectory with sparkline
            '<div class="section">',
            '<h2>Quality Trajectory</h2>',
            self._render_quality_trajectory_chart(trend_data, analytics),
            '</div>',

            # Thompson Sampling section
            '<div class="section">',
            '<h2>Thompson Sampling Insights</h2>',
            self._render_thompson_section(analytics),
            '</div>',

            # Task type distribution
            '<div class="section">',
            '<h2>Task Type Distribution</h2>',
            self._render_task_distribution(analytics.task_type_distribution),
            '</div>',

            # Version performance
            '<div class="section">',
            '<h2>Version Performance</h2>',
            self._render_version_performance(analytics.version_performance),
            '</div>',

            # Recommendation
            '<div class="section recommendation">',
            f'<h2>Recommendation</h2>',
            f'<p>{html_escape.escape(analytics.recommendation)}</p>',
            '</div>',

            self._render_footer(),
            '</body></html>'
        ]

        return '\n'.join(html_parts)

    def render_comparison_report(
        self,
        prompt_a: str,
        prompt_b: str,
        task_type: Optional[str] = None,
        days: int = 30
    ) -> str:
        """
        Render comparison report between two prompts.

        Uses small multiples pattern for fair comparison.

        Args:
            prompt_a: First prompt name
            prompt_b: Second prompt name
            task_type: Optional task type filter
            days: Analysis window in days

        Returns:
            Complete HTML document as string
        """
        comparison = self.engine.compare_prompts(prompt_a, prompt_b, task_type, days)

        if not comparison:
            return self._render_empty_comparison(prompt_a, prompt_b)

        html_parts = [
            self._render_html_head(f"Comparison: {prompt_a} vs {prompt_b}"),
            '<body>',
            self._render_header(f"Prompt Comparison"),

            # Winner banner
            '<div class="winner-banner">',
            f'<span class="winner-label">Winner:</span>',
            f'<span class="winner-name">{html_escape.escape(comparison.winner or "Tie")}</span>',
            self._render_significance_badge(comparison.statistical_significance),
            '</div>',

            # Small multiples comparison
            '<div class="comparison-grid">',

            # Prompt A column
            '<div class="comparison-column">',
            f'<h2>{html_escape.escape(prompt_a)}</h2>',
            self._render_comparison_metrics(comparison.prompt_a_stats),
            '</div>',

            # Prompt B column
            '<div class="comparison-column">',
            f'<h2>{html_escape.escape(prompt_b)}</h2>',
            self._render_comparison_metrics(comparison.prompt_b_stats),
            '</div>',

            '</div>',

            # Difference analysis
            '<div class="section">',
            '<h2>Difference Analysis</h2>',
            self._render_difference_table(comparison),
            '</div>',

            # Recommendation
            '<div class="section recommendation">',
            f'<h2>Recommendation</h2>',
            f'<p>{html_escape.escape(comparison.recommendation)}</p>',
            '</div>',

            self._render_footer(),
            '</body></html>'
        ]

        return '\n'.join(html_parts)

    def render_overview_dashboard(
        self,
        days: int = 30,
        limit: int = 10
    ) -> str:
        """
        Render system-wide overview dashboard.

        Shows all prompts with key metrics and health indicators.

        Args:
            days: Analysis window in days
            limit: Maximum prompts to show

        Returns:
            Complete HTML document as string
        """
        # Get all prompts with analytics
        prompts = self.db.list_prompts()
        prompt_analytics = []

        for prompt in prompts[:limit]:
            analytics = self.engine.get_prompt_analytics(prompt.name, days)
            if analytics and analytics.total_executions > 0:
                prompt_analytics.append({
                    'name': prompt.name,
                    'analytics': analytics,
                    'trend': self.engine.get_quality_trend(prompt.name, "daily", min(days, 7))
                })

        # Sort by executions
        prompt_analytics.sort(
            key=lambda x: x['analytics'].total_executions,
            reverse=True
        )

        # Get underperforming prompts
        underperforming = self.engine.identify_underperforming(0.6, days)

        html_parts = [
            self._render_html_head("Promptly Analytics Dashboard"),
            '<body>',
            self._render_header("Analytics Dashboard"),

            # Summary stats
            '<div class="metrics-row">',
            self._render_metric_card(
                "Total Prompts",
                str(len(prompts)),
                icon="📝"
            ),
            self._render_metric_card(
                "Active (30d)",
                str(len(prompt_analytics)),
                icon="✓"
            ),
            self._render_metric_card(
                "Underperforming",
                str(len(underperforming)),
                trend="declining" if len(underperforming) > 0 else "stable",
                icon="⚠"
            ),
            '</div>',

            # Prompts table with sparklines
            '<div class="section">',
            '<h2>Prompt Performance</h2>',
            self._render_prompts_table(prompt_analytics),
            '</div>',

            # Underperforming alerts
            '<div class="section">',
            '<h2>Attention Required</h2>',
            self._render_underperforming_list(underperforming),
            '</div>',

            self._render_footer(),
            '</body></html>'
        ]

        return '\n'.join(html_parts)

    def render_quality_trajectory(
        self,
        prompt_name: str,
        days: int = 30,
        standalone: bool = False
    ) -> str:
        """
        Render quality trajectory sparkline chart.

        Can be embedded or standalone.

        Args:
            prompt_name: Prompt to analyze
            days: Analysis window
            standalone: If True, returns complete HTML document

        Returns:
            HTML string (fragment or document)
        """
        trend_data = self.engine.get_quality_trend(prompt_name, "daily", days)

        if not trend_data:
            return '<p class="no-data">No data available</p>'

        chart_html = self._render_sparkline_svg(
            [d.get('avg_quality', 0) for d in trend_data],
            width=self.config.sparkline_width * 3,
            height=self.config.sparkline_height * 2,
            show_endpoints=True,
            show_baseline=True
        )

        if not standalone:
            return chart_html

        return '\n'.join([
            self._render_html_head(f"Quality Trajectory: {prompt_name}"),
            '<body>',
            self._render_header(f"Quality Trajectory: {prompt_name}"),
            '<div class="section">',
            chart_html,
            '</div>',
            self._render_footer(),
            '</body></html>'
        ])

    # =========================================================================
    # Private rendering methods
    # =========================================================================

    def _render_html_head(self, title: str) -> str:
        """Render HTML head with styles."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_escape.escape(title)}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: {self.config.font_family};
            font-size: 14px;
            line-height: 1.5;
            color: {self.config.color_primary};
            background: #fafafa;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 1.5rem;
            font-weight: 500;
            margin-bottom: 20px;
            color: {self.config.color_primary};
        }}

        h2 {{
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 12px;
            color: {self.config.color_secondary};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .header {{
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 12px;
            margin-bottom: 20px;
        }}

        .metrics-row {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}

        .metric-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 16px;
            flex: 1;
            min-width: 150px;
        }}

        .metric-label {{
            font-size: 0.75rem;
            color: {self.config.color_secondary};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
            font-family: {self.config.font_mono};
            margin: 4px 0;
        }}

        .metric-trend {{
            font-size: 0.75rem;
        }}

        .trend-improving {{ color: {self.config.color_success}; }}
        .trend-declining {{ color: {self.config.color_danger}; }}
        .trend-stable {{ color: {self.config.color_secondary}; }}

        .section {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 16px;
        }}

        .recommendation {{
            background: #f8f9fa;
            border-left: 3px solid {self.config.color_info};
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}

        th {{
            text-align: left;
            font-weight: 500;
            color: {self.config.color_secondary};
            border-bottom: 2px solid #e0e0e0;
            padding: 8px 12px;
        }}

        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #f0f0f0;
            font-family: {self.config.font_mono};
        }}

        tr:hover td {{
            background: #f8f9fa;
        }}

        .sparkline {{
            display: inline-block;
            vertical-align: middle;
        }}

        .no-data {{
            color: {self.config.color_secondary};
            font-style: italic;
        }}

        .winner-banner {{
            background: linear-gradient(135deg, {self.config.color_success}15, {self.config.color_success}05);
            border: 1px solid {self.config.color_success}40;
            border-radius: 4px;
            padding: 16px;
            text-align: center;
            margin-bottom: 20px;
        }}

        .winner-label {{
            color: {self.config.color_secondary};
            margin-right: 8px;
        }}

        .winner-name {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {self.config.color_success};
        }}

        .significance-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            margin-left: 12px;
        }}

        .significance-high {{
            background: {self.config.color_success}20;
            color: {self.config.color_success};
        }}

        .significance-medium {{
            background: {self.config.color_warning}20;
            color: {self.config.color_warning};
        }}

        .significance-low {{
            background: {self.config.color_secondary}20;
            color: {self.config.color_secondary};
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .comparison-column {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 20px;
        }}

        .bar-container {{
            height: 20px;
            background: #f0f0f0;
            border-radius: 2px;
            margin: 4px 0;
        }}

        .bar {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }}

        .bar-quality {{ background: {self.config.color_info}; }}
        .bar-success {{ background: {self.config.color_success}; }}
        .bar-warning {{ background: {self.config.color_warning}; }}

        .alert-item {{
            display: flex;
            align-items: center;
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
        }}

        .alert-icon {{
            font-size: 1.25rem;
            margin-right: 12px;
        }}

        .alert-content {{
            flex: 1;
        }}

        .alert-name {{
            font-weight: 500;
        }}

        .alert-reason {{
            font-size: 0.875rem;
            color: {self.config.color_secondary};
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: {self.config.color_secondary};
            font-size: 0.75rem;
        }}
    </style>
</head>'''

    def _render_header(self, title: str) -> str:
        """Render page header."""
        return f'''<div class="header">
    <h1>{html_escape.escape(title)}</h1>
    <div class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</div>'''

    def _render_footer(self) -> str:
        """Render page footer."""
        return '''<div class="footer">
    <p>Generated by Promptly Analytics Dashboard</p>
</div>'''

    def _render_metric_card(
        self,
        label: str,
        value: str,
        trend: Optional[str] = None,
        icon: str = ""
    ) -> str:
        """Render a metric card with optional trend."""
        trend_html = ""
        if trend:
            trend_class = f"trend-{trend}"
            trend_symbol = "↑" if trend == "improving" else "↓" if trend == "declining" else "→"
            trend_html = f'<div class="metric-trend {trend_class}">{trend_symbol} {trend}</div>'

        return f'''<div class="metric-card">
    <div class="metric-label">{icon} {html_escape.escape(label)}</div>
    <div class="metric-value">{html_escape.escape(value)}</div>
    {trend_html}
</div>'''

    def _render_sparkline_svg(
        self,
        values: List[float],
        width: int = 100,
        height: int = 20,
        show_endpoints: bool = False,
        show_baseline: bool = False
    ) -> str:
        """
        Render SVG sparkline.

        Following Tufte's design: minimal, word-sized graphics.
        """
        if not values:
            return '<span class="no-data">—</span>'

        # Normalize values
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1

        # Calculate points
        points = []
        x_step = width / max(len(values) - 1, 1)

        for i, v in enumerate(values):
            x = i * x_step
            y = height - ((v - min_val) / range_val * (height - 4) + 2)
            points.append(f"{x:.1f},{y:.1f}")

        path = f'M{" L".join(points)}'

        # Build SVG
        svg_parts = [
            f'<svg class="sparkline" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        ]

        # Baseline at 0.5 (middle)
        if show_baseline:
            baseline_y = height / 2
            svg_parts.append(
                f'<line x1="0" y1="{baseline_y}" x2="{width}" y2="{baseline_y}" '
                f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="2,2"/>'
            )

        # Main line
        svg_parts.append(
            f'<path d="{path}" fill="none" stroke="{self.config.color_info}" stroke-width="1.5"/>'
        )

        # Endpoints
        if show_endpoints and len(points) > 0:
            first = points[0].split(',')
            last = points[-1].split(',')

            svg_parts.append(
                f'<circle cx="{first[0]}" cy="{first[1]}" r="2" fill="{self.config.color_secondary}"/>'
            )
            svg_parts.append(
                f'<circle cx="{last[0]}" cy="{last[1]}" r="2" fill="{self.config.color_info}"/>'
            )

        svg_parts.append('</svg>')

        return ''.join(svg_parts)

    def _render_quality_trajectory_chart(
        self,
        trend_data: List[Dict],
        analytics: Any
    ) -> str:
        """Render quality trajectory section."""
        if not trend_data:
            return '<p class="no-data">No trend data available</p>'

        values = [d.get('avg_quality', 0) for d in trend_data]
        dates = [d.get('date', '') for d in trend_data]

        sparkline = self._render_sparkline_svg(
            values,
            width=300,
            height=60,
            show_endpoints=True,
            show_baseline=True
        )

        # Statistics
        current = values[-1] if values else 0
        previous = values[0] if values else 0
        change = current - previous
        change_pct = (change / previous * 100) if previous > 0 else 0

        return f'''<div style="display: flex; gap: 40px; align-items: center;">
    <div>
        {sparkline}
        <div style="font-size: 0.75rem; color: {self.config.color_secondary}; margin-top: 4px;">
            {dates[0] if dates else ''} → {dates[-1] if dates else ''}
        </div>
    </div>
    <div>
        <div style="margin-bottom: 8px;">
            <span style="font-size: 0.75rem; color: {self.config.color_secondary};">Current</span>
            <div style="font-size: 1.25rem; font-weight: 600;">{current:.1%}</div>
        </div>
        <div>
            <span style="font-size: 0.75rem; color: {self.config.color_secondary};">Change</span>
            <div style="color: {'#27ae60' if change >= 0 else '#e74c3c'};">
                {'+' if change >= 0 else ''}{change:.1%} ({change_pct:+.1f}%)
            </div>
        </div>
    </div>
</div>'''

    def _render_thompson_section(self, analytics: Any) -> str:
        """Render Thompson Sampling insights."""
        expected = analytics.thompson_expected_quality

        return f'''<div style="display: flex; gap: 40px;">
    <div>
        <div style="font-size: 0.75rem; color: {self.config.color_secondary};">Expected Quality</div>
        <div style="font-size: 1.5rem; font-weight: 600;">{expected:.1%}</div>
    </div>
    <div>
        <div style="font-size: 0.75rem; color: {self.config.color_secondary};">Confidence Level</div>
        <div class="bar-container" style="width: 150px;">
            <div class="bar bar-quality" style="width: {expected * 100:.0f}%;"></div>
        </div>
    </div>
</div>'''

    def _render_task_distribution(
        self,
        distribution: Dict[str, Dict[str, Any]]
    ) -> str:
        """Render task type distribution table."""
        if not distribution:
            return '<p class="no-data">No task data available</p>'

        rows = []
        for task_type, stats in sorted(
            distribution.items(),
            key=lambda x: x[1].get('count', 0),
            reverse=True
        ):
            count = stats.get('count', 0)
            quality = stats.get('avg_quality', 0)

            rows.append(f'''<tr>
    <td>{html_escape.escape(task_type)}</td>
    <td>{count}</td>
    <td>{quality:.1%}</td>
    <td>
        <div class="bar-container" style="width: 100px;">
            <div class="bar bar-quality" style="width: {quality * 100:.0f}%;"></div>
        </div>
    </td>
</tr>''')

        return f'''<table>
    <thead>
        <tr>
            <th>Task Type</th>
            <th>Count</th>
            <th>Avg Quality</th>
            <th>Performance</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>'''

    def _render_version_performance(
        self,
        version_data: Dict[int, Dict[str, Any]]
    ) -> str:
        """Render version performance comparison."""
        if not version_data:
            return '<p class="no-data">No version data available</p>'

        rows = []
        for version, stats in sorted(version_data.items()):
            count = stats.get('count', 0)
            quality = stats.get('avg_quality', 0)
            latency = stats.get('avg_latency_ms', 0)

            rows.append(f'''<tr>
    <td>v{version}</td>
    <td>{count}</td>
    <td>{quality:.1%}</td>
    <td>{latency:.0f}ms</td>
</tr>''')

        return f'''<table>
    <thead>
        <tr>
            <th>Version</th>
            <th>Executions</th>
            <th>Avg Quality</th>
            <th>Avg Latency</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>'''

    def _render_comparison_metrics(self, stats: Dict[str, Any]) -> str:
        """Render metrics for comparison column."""
        return f'''<div class="comparison-metrics">
    <div style="margin-bottom: 12px;">
        <div style="font-size: 0.75rem; color: {self.config.color_secondary};">Avg Quality</div>
        <div style="font-size: 1.25rem; font-weight: 600;">{stats.get('avg_quality', 0):.1%}</div>
        <div class="bar-container">
            <div class="bar bar-quality" style="width: {stats.get('avg_quality', 0) * 100:.0f}%;"></div>
        </div>
    </div>
    <div style="margin-bottom: 12px;">
        <div style="font-size: 0.75rem; color: {self.config.color_secondary};">Success Rate</div>
        <div style="font-size: 1.25rem; font-weight: 600;">{stats.get('success_rate', 0):.1%}</div>
        <div class="bar-container">
            <div class="bar bar-success" style="width: {stats.get('success_rate', 0) * 100:.0f}%;"></div>
        </div>
    </div>
    <div style="margin-bottom: 12px;">
        <div style="font-size: 0.75rem; color: {self.config.color_secondary};">Executions</div>
        <div style="font-size: 1.25rem; font-weight: 600;">{stats.get('total_executions', 0)}</div>
    </div>
    <div>
        <div style="font-size: 0.75rem; color: {self.config.color_secondary};">Avg Latency</div>
        <div style="font-size: 1.25rem; font-weight: 600;">{stats.get('avg_latency_ms', 0):.0f}ms</div>
    </div>
</div>'''

    def _render_significance_badge(self, significance: float) -> str:
        """Render statistical significance badge."""
        if significance >= 0.95:
            level = "high"
            text = f"p < 0.05 ({significance:.0%})"
        elif significance >= 0.80:
            level = "medium"
            text = f"p < 0.20 ({significance:.0%})"
        else:
            level = "low"
            text = f"Not significant ({significance:.0%})"

        return f'<span class="significance-badge significance-{level}">{text}</span>'

    def _render_difference_table(self, comparison: Any) -> str:
        """Render difference analysis table."""
        a_stats = comparison.prompt_a_stats
        b_stats = comparison.prompt_b_stats

        metrics = [
            ("Avg Quality", "avg_quality", True, "{:.1%}"),
            ("Success Rate", "success_rate", True, "{:.1%}"),
            ("Avg Latency", "avg_latency_ms", False, "{:.0f}ms"),
            ("Total Executions", "total_executions", True, "{}"),
        ]

        rows = []
        for label, key, higher_better, fmt in metrics:
            val_a = a_stats.get(key, 0)
            val_b = b_stats.get(key, 0)
            diff = val_b - val_a

            # Determine winner for this metric
            if higher_better:
                winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
            else:
                winner = "A" if val_a < val_b else "B" if val_b < val_a else "Tie"

            winner_style = f'color: {self.config.color_success};' if winner != "Tie" else ""

            rows.append(f'''<tr>
    <td>{label}</td>
    <td>{fmt.format(val_a)}</td>
    <td>{fmt.format(val_b)}</td>
    <td style="{winner_style}">{winner}</td>
</tr>''')

        return f'''<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>{html_escape.escape(comparison.prompt_a)}</th>
            <th>{html_escape.escape(comparison.prompt_b)}</th>
            <th>Winner</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>'''

    def _render_prompts_table(
        self,
        prompt_analytics: List[Dict[str, Any]]
    ) -> str:
        """Render prompts overview table with sparklines."""
        if not prompt_analytics:
            return '<p class="no-data">No prompts with activity</p>'

        rows = []
        for item in prompt_analytics:
            analytics = item['analytics']
            trend = item['trend']

            trend_values = [d.get('avg_quality', 0) for d in trend] if trend else []
            sparkline = self._render_sparkline_svg(trend_values, show_endpoints=True)

            # Health indicator
            if analytics.avg_quality >= 0.8:
                health = f'<span style="color: {self.config.color_success};">●</span>'
            elif analytics.avg_quality >= 0.6:
                health = f'<span style="color: {self.config.color_warning};">●</span>'
            else:
                health = f'<span style="color: {self.config.color_danger};">●</span>'

            rows.append(f'''<tr>
    <td>{health} {html_escape.escape(item['name'])}</td>
    <td>{analytics.total_executions}</td>
    <td>{analytics.avg_quality:.1%}</td>
    <td>{analytics.success_rate:.1%}</td>
    <td>{analytics.avg_latency_ms:.0f}ms</td>
    <td>{sparkline}</td>
</tr>''')

        return f'''<table>
    <thead>
        <tr>
            <th>Prompt</th>
            <th>Executions</th>
            <th>Quality</th>
            <th>Success</th>
            <th>Latency</th>
            <th>Trend (7d)</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>'''

    def _render_underperforming_list(
        self,
        underperforming: List[Dict[str, Any]]
    ) -> str:
        """Render underperforming prompts list."""
        if not underperforming:
            return '<p style="color: {0};">✓ No underperforming prompts</p>'.format(
                self.config.color_success
            )

        items = []
        for item in underperforming[:5]:
            items.append(f'''<div class="alert-item">
    <span class="alert-icon">⚠</span>
    <div class="alert-content">
        <div class="alert-name">{html_escape.escape(item.get('prompt_name', 'Unknown'))}</div>
        <div class="alert-reason">{html_escape.escape(item.get('reason', 'Quality below threshold'))}</div>
    </div>
    <div style="font-family: {self.config.font_mono}; color: {self.config.color_danger};">
        {item.get('avg_quality', 0):.1%}
    </div>
</div>''')

        return ''.join(items)

    def _render_empty_report(self, prompt_name: str) -> str:
        """Render empty state for missing prompt."""
        return '\n'.join([
            self._render_html_head(f"Prompt Report: {prompt_name}"),
            '<body>',
            self._render_header(f"Prompt Report: {prompt_name}"),
            '<div class="section">',
            f'<p class="no-data">No analytics data found for prompt "{html_escape.escape(prompt_name)}"</p>',
            '</div>',
            self._render_footer(),
            '</body></html>'
        ])

    def _render_empty_comparison(self, prompt_a: str, prompt_b: str) -> str:
        """Render empty state for comparison."""
        return '\n'.join([
            self._render_html_head(f"Comparison: {prompt_a} vs {prompt_b}"),
            '<body>',
            self._render_header("Prompt Comparison"),
            '<div class="section">',
            '<p class="no-data">Unable to compare prompts. Ensure both have execution history.</p>',
            '</div>',
            self._render_footer(),
            '</body></html>'
        ])


# =============================================================================
# Factory functions
# =============================================================================

def create_dashboard(db: PromptlyDatabase) -> PromptlyDashboard:
    """
    Create a dashboard instance.

    Args:
        db: Database instance

    Returns:
        PromptlyDashboard instance
    """
    return PromptlyDashboard(db)


def render_prompt_report(
    db: PromptlyDatabase,
    prompt_name: str,
    days: int = 30
) -> str:
    """
    Convenience function to render a prompt report.

    Args:
        db: Database instance
        prompt_name: Prompt to analyze
        days: Analysis window

    Returns:
        HTML string
    """
    dashboard = create_dashboard(db)
    return dashboard.render_prompt_report(prompt_name, days)


def render_overview_dashboard(
    db: PromptlyDatabase,
    days: int = 30,
    limit: int = 10
) -> str:
    """
    Convenience function to render overview dashboard.

    Args:
        db: Database instance
        days: Analysis window
        limit: Maximum prompts

    Returns:
        HTML string
    """
    dashboard = create_dashboard(db)
    return dashboard.render_overview_dashboard(days, limit)
