"""Tufte-style learning dashboard visualization for CARTS red team analytics.

Renders key learner metrics with trends, targets, and recommendations.
Follows Edward Tufte's data visualization principles:
- Maximize information density
- Minimize decoration (chartjunk)
- Show meaning first via small multiples
- Enable instant comparison via sparklines

Philosophy: "Above all else show the data."

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-05
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MetricStatus(Enum):
    """Status indicator for learner metrics."""

    EXCELLENT = "excellent"  # Green: >90% of target or positive trend
    GOOD = "good"  # Light green: 70-90% of target
    FAIR = "fair"  # Yellow: 50-70% of target
    POOR = "poor"  # Orange: 30-50% of target
    CRITICAL = "critical"  # Red: <30% of target


@dataclass
class LearnerMetricPanel:
    """Single metric panel for learning dashboard.

    Displays a learner metric with:
    - Current value
    - Trend sparkline (last N observations)
    - Target value (if applicable)
    - Status indicator
    - Delta from previous value
    """

    metric_name: str  # Display name ("Attack Success Rate", "Thompson Convergence", etc.)
    current_value: float  # Current measurement (0.0-1.0 or 0-100)
    trend: list[float] = field(default_factory=list)  # Historical values for sparkline
    target_value: float | None = None  # Optional target (for comparison)
    status: MetricStatus = MetricStatus.FAIR  # Current status (EXCELLENT/GOOD/FAIR/POOR/CRITICAL)
    unit: str = "%"  # Display unit ("%", "attacks/sec", "rate", etc.)
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra context

    def __post_init__(self):
        """Compute delta and normalize trend."""
        if self.trend:
            self.previous_value = self.trend[-2] if len(self.trend) > 1 else self.trend[0]
            self.delta = self.current_value - self.previous_value
        else:
            self.previous_value = self.current_value
            self.delta = 0.0


class LearningDashboardRenderer:
    """Render learning metrics with Tufte-style small multiples.

    Produces pure HTML/CSS/SVG output with:
    - Metric panels: Current value + sparkline + delta
    - Status indicators: Color-coded performance (green/yellow/red)
    - Trend visualization: Small multiples for comparison
    - Summary statistics: Key aggregates (mean, trend, target progress)
    - Recommendations: Automatic suggestions for improvement
    - No external dependencies: All CSS/SVG inline

    Philosophy: Maximum data-ink ratio, minimal decoration, meaning first.
    """

    # Color palette (semantic, accessible)
    COLOR_EXCELLENT = "#27ae60"  # Green: >90% of target
    COLOR_GOOD = "#2ecc71"  # Light green: 70-90%
    COLOR_FAIR = "#f39c12"  # Yellow: 50-70%
    COLOR_POOR = "#e67e22"  # Orange: 30-50%
    COLOR_CRITICAL = "#e74c3c"  # Red: <30%

    COLOR_POSITIVE = "#16a085"  # Teal: improving trend
    COLOR_NEUTRAL = "#95a5a6"  # Gray: stable
    COLOR_NEGATIVE = "#c0392b"  # Dark red: degrading trend

    # Layout constants
    DEFAULT_WIDTH = 800
    DEFAULT_HEIGHT = 600
    PANEL_WIDTH = 180
    PANEL_HEIGHT = 140
    SPARKLINE_WIDTH = 120
    SPARKLINE_HEIGHT = 40

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        """Initialize renderer.

        Args:
            width: Dashboard width in pixels (default: 800)
            height: Dashboard height in pixels (default: 600)
        """
        self.width = width
        self.height = height
        self.panels: list[LearnerMetricPanel] = []
        self.rendered_at = datetime.now().isoformat()

    def add_metric(self, panel: LearnerMetricPanel) -> None:
        """Add metric panel to dashboard.

        Args:
            panel: LearnerMetricPanel instance
        """
        self.panels.append(panel)

    def _compute_status(
        self,
        current: float,
        target: float | None,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> MetricStatus:
        """Compute status based on current vs target.

        Args:
            current: Current value
            target: Target value (optional)
            min_val: Minimum possible value
            max_val: Maximum possible value

        Returns:
            MetricStatus indicator
        """
        if target is None:
            # No target: use absolute scale
            progress = (current - min_val) / (max_val - min_val)
        else:
            # With target: use progress toward target
            progress = min(current / target, 1.0) if target > 0 else 0.0

        if progress >= 0.9:
            return MetricStatus.EXCELLENT
        elif progress >= 0.7:
            return MetricStatus.GOOD
        elif progress >= 0.5:
            return MetricStatus.FAIR
        elif progress >= 0.3:
            return MetricStatus.POOR
        else:
            return MetricStatus.CRITICAL

    def _get_status_color(self, status: MetricStatus) -> str:
        """Get color for status indicator.

        Args:
            status: MetricStatus enum

        Returns:
            Hex color string
        """
        color_map = {
            MetricStatus.EXCELLENT: self.COLOR_EXCELLENT,
            MetricStatus.GOOD: self.COLOR_GOOD,
            MetricStatus.FAIR: self.COLOR_FAIR,
            MetricStatus.POOR: self.COLOR_POOR,
            MetricStatus.CRITICAL: self.COLOR_CRITICAL,
        }
        return color_map.get(status, self.COLOR_NEUTRAL)

    def _get_trend_color(self, values: list[float]) -> str:
        """Get color indicating trend direction.

        Args:
            values: List of values

        Returns:
            Hex color string (green if improving, red if degrading, gray if stable)
        """
        if len(values) < 2:
            return self.COLOR_NEUTRAL

        recent = values[-1] if values else 0.0
        previous = values[-2] if len(values) > 1 else recent
        delta = recent - previous

        if delta > 0.05:  # >5% improvement
            return self.COLOR_POSITIVE
        elif delta < -0.05:  # >5% degradation
            return self.COLOR_NEGATIVE
        else:
            return self.COLOR_NEUTRAL

    def _render_sparkline(
        self,
        values: list[float],
        width: int = SPARKLINE_WIDTH,
        height: int = SPARKLINE_HEIGHT,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> str:
        """Render inline sparkline SVG.

        Args:
            values: List of values for sparkline
            width: Sparkline width in pixels
            height: Sparkline height in pixels
            min_val: Minimum value for scaling
            max_val: Maximum value for scaling

        Returns:
            SVG string
        """
        if not values:
            return f'<svg width="{width}" height="{height}"></svg>'

        # Normalize values to [0, height]
        range_val = max_val - min_val or 1.0
        points = []
        for i, v in enumerate(values):
            x = (i / (len(values) - 1)) * width if len(values) > 1 else width / 2
            y = height - ((v - min_val) / range_val) * height
            points.append((x, y))

        # Create path
        path_data = f"M {points[0][0]} {points[0][1]}"
        for x, y in points[1:]:
            path_data += f" L {x} {y}"

        color = self._get_trend_color(values)

        svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path d="{path_data}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{points[-1][0]}" cy="{points[-1][1]}" r="2" fill="{color}"/>
</svg>'''
        return svg

    def _render_metric_panel(self, panel: LearnerMetricPanel) -> str:
        """Render single metric panel.

        Args:
            panel: LearnerMetricPanel instance

        Returns:
            HTML string for panel
        """
        status_color = self._get_status_color(panel.status)
        delta_color = (
            self.COLOR_POSITIVE if panel.delta > 0
            else self.COLOR_NEGATIVE if panel.delta < 0
            else self.COLOR_NEUTRAL
        )
        delta_sign = "+" if panel.delta > 0 else ""
        delta_indicator = "↑" if panel.delta > 0 else "↓" if panel.delta < 0 else "→"

        sparkline = self._render_sparkline(panel.trend) if panel.trend else ""

        # Format current value
        if "rate" in panel.metric_name.lower() or "%" in panel.unit:
            value_str = f"{panel.current_value:.1%}"
        else:
            value_str = f"{panel.current_value:.2f}"

        # Format target value
        target_str = ""
        if panel.target_value is not None:
            target_str = (
                '<div style="font-size: 11px; color: #7f8c8d; margin-top: 4px;">'
                f'Target: <strong>{panel.target_value:.1%}</strong>'
                '</div>'
            )

        # Format delta
        delta_str = f"{delta_sign}{abs(panel.delta):.1%}" if isinstance(panel.delta, float) else str(panel.delta)

        html = f'''
<div style="
    width: {self.PANEL_WIDTH}px;
    height: {self.PANEL_HEIGHT}px;
    border: 1px solid #ecf0f1;
    border-radius: 4px;
    padding: 10px;
    margin: 8px;
    background: #fff;
    display: inline-block;
    vertical-align: top;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
">
    <!-- Status indicator dot -->
    <div style="
        width: 8px;
        height: 8px;
        background: {status_color};
        border-radius: 4px;
        display: inline-block;
        margin-right: 6px;
        vertical-align: middle;
    "></div>

    <!-- Metric name -->
    <div style="
        font-size: 12px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 6px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    ">{panel.metric_name}</div>

    <!-- Current value -->
    <div style="
        font-size: 20px;
        font-weight: bold;
        color: {status_color};
        font-family: 'Monaco', 'Courier New', monospace;
        line-height: 1.2;
    ">{value_str}</div>

    <!-- Delta indicator -->
    <div style="
        font-size: 11px;
        color: {delta_color};
        font-weight: 600;
        margin-top: 4px;
    ">{delta_indicator} {delta_str}</div>

    {target_str}

    <!-- Sparkline -->
    <div style="margin-top: 6px; display: flex; justify-content: center;">
        {sparkline}
    </div>
</div>
'''
        return html

    def render_html(self, title: str = "CARTS Learning Dashboard") -> str:
        """Render complete dashboard as HTML.

        Args:
            title: Dashboard title

        Returns:
            Complete HTML document
        """
        panels_html = "".join(self._render_metric_panel(p) for p in self.panels)

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #fafafa;
            margin: 0;
            padding: 20px;
            color: #2c3e50;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 28px;
            font-weight: 600;
            margin: 0 0 8px 0;
            color: #2c3e50;
        }}
        .metadata {{
            font-size: 12px;
            color: #95a5a6;
            margin-bottom: 20px;
        }}
        .panels-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start;
            gap: 0;
        }}
        .summary-section {{
            margin-top: 30px;
            padding: 15px;
            background: #fff;
            border: 1px solid #ecf0f1;
            border-radius: 4px;
        }}
        .summary-section h2 {{
            font-size: 14px;
            font-weight: 600;
            color: #2c3e50;
            margin: 0 0 10px 0;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat {{
            padding: 10px;
            background: #f8f9fa;
            border-left: 3px solid #ecf0f1;
        }}
        .stat-label {{
            font-size: 11px;
            color: #7f8c8d;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            font-family: 'Monaco', 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>{title}</h1>
        <div class="metadata">
            Generated {self.rendered_at} | {len(self.panels)} metrics
        </div>

        <div class="panels-container">
            {panels_html}
        </div>

        {self._render_summary_stats()}
        {self._render_recommendations()}
    </div>
</body>
</html>'''

    def _render_summary_stats(self) -> str:
        """Render summary statistics section.

        Returns:
            HTML string with summary stats
        """
        if not self.panels:
            return ""

        # Compute aggregates
        current_values = [p.current_value for p in self.panels if p.current_value >= 0]
        avg_current = sum(current_values) / len(current_values) if current_values else 0.0

        target_values = [p.target_value for p in self.panels if p.target_value is not None]
        avg_target = sum(target_values) / len(target_values) if target_values else 0.0

        # Count status distribution
        excellent = sum(1 for p in self.panels if p.status == MetricStatus.EXCELLENT)
        good = sum(1 for p in self.panels if p.status == MetricStatus.GOOD)
        fair = sum(1 for p in self.panels if p.status == MetricStatus.FAIR)
        poor = sum(1 for p in self.panels if p.status == MetricStatus.POOR)
        critical = sum(1 for p in self.panels if p.status == MetricStatus.CRITICAL)

        overall_health = (
            excellent * 5 + good * 4 + fair * 3 + poor * 2 + critical * 1
        ) / (len(self.panels) * 5) if self.panels else 0.0

        return f'''
<div class="summary-section">
    <h2>Summary Statistics</h2>
    <div class="summary-stats">
        <div class="stat">
            <div class="stat-label">Average Performance</div>
            <div class="stat-value">{avg_current:.1%}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Health Score</div>
            <div class="stat-value">{overall_health:.1%}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Excellent/Good</div>
            <div class="stat-value">{excellent + good}/{len(self.panels)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Critical/Poor</div>
            <div class="stat-value">{critical + poor}/{len(self.panels)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Target Progress</div>
            <div class="stat-value">{min(avg_current / max(avg_target, 0.01), 1.0):.1%}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Total Metrics</div>
            <div class="stat-value">{len(self.panels)}</div>
        </div>
    </div>
</div>
'''

    def _render_recommendations(self) -> str:
        """Generate and render recommendations.

        Returns:
            HTML string with recommendations
        """
        recommendations = self.get_recommendations()
        if not recommendations:
            return ""

        rec_items = "".join(
            f'<li style="margin-bottom: 8px; font-size: 13px; color: #34495e;">{rec}</li>'
            for rec in recommendations[:5]
        )

        return f'''
<div class="summary-section">
    <h2>Recommendations</h2>
    <ul style="margin: 0; padding-left: 20px;">
        {rec_items}
    </ul>
</div>
'''

    def render_small_multiples(self) -> str:
        """Render panels as small multiples (grid layout).

        Returns:
            HTML string with small multiples grid
        """
        cols = max(2, min(5, len(self.panels)))
        grid = f'''
<div style="
    display: grid;
    grid-template-columns: repeat({cols}, 1fr);
    gap: 10px;
    margin: 20px 0;
">
'''
        for panel in self.panels:
            grid += self._render_metric_panel(panel)
        grid += "</div>"
        return grid

    def get_recommendations(self) -> list[str]:
        """Generate recommendations based on metric performance.

        Returns:
            List of recommendation strings (sorted by priority)
        """
        recommendations = []

        # Analyze critical metrics
        critical_panels = [p for p in self.panels if p.status == MetricStatus.CRITICAL]
        if critical_panels:
            critical_names = ", ".join(p.metric_name for p in critical_panels)
            recommendations.append(
                f"❌ CRITICAL: {critical_names} below 30% threshold - immediate action required"
            )

        # Analyze degrading trends
        degrading = [
            p for p in self.panels
            if len(p.trend) >= 3 and p.delta < -0.1
        ]
        if degrading:
            names = ", ".join(p.metric_name for p in degrading)
            recommendations.append(
                f"⚠️ DEGRADING TRENDS: {names} declining sharply - review strategies"
            )

        # Analyze improving trends
        improving = [
            p for p in self.panels
            if len(p.trend) >= 3 and p.delta > 0.1 and p.status in [MetricStatus.GOOD, MetricStatus.EXCELLENT]
        ]
        if improving:
            names = ", ".join(p.metric_name for p in improving)
            recommendations.append(
                f"✅ MOMENTUM: {names} improving - consider reinforcing these strategies"
            )

        # Target progress
        with_targets = [p for p in self.panels if p.target_value is not None]
        if with_targets:
            on_track = sum(
                1 for p in with_targets
                if p.current_value >= p.target_value * 0.7
            )
            if on_track < len(with_targets) * 0.5:
                recommendations.append(
                    f"🎯 TARGET TRACKING: Only {on_track}/{len(with_targets)} metrics on track for targets"
                )

        # Learning convergence
        convergence = [
            p for p in self.panels
            if "convergence" in p.metric_name.lower() or "thompson" in p.metric_name.lower()
        ]
        if convergence and convergence[0].current_value > 0.8:
            recommendations.append(
                "🧠 LEARNING CONVERGED: Thompson Sampling strategy selection is stable"
            )

        return recommendations if recommendations else ["✓ All metrics within acceptable ranges"]
