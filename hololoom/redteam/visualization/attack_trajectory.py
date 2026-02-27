"""Tufte-style attack trajectory visualization for CARTS red team analytics.

Renders attack effectiveness over time with anomaly detection, strategy comparison,
and comprehensive metrics. Follows Edward Tufte's data visualization principles:
- Maximize information density
- Minimize chartjunk (decoration)
- Show meaning first
- Enable comparison via small multiples

Philosophy: "Above all else show the data."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import math


class AnomalyType(Enum):
    """Types of anomalies detected in attack trajectory."""

    SUDDEN_BREAKTHROUGH = "sudden_breakthrough"  # Success rate spike >0.2
    SUSTAINED_SUCCESS = "sustained_success"  # High success for 3+ consecutive points
    PLATEAU = "plateau"  # Success rate stable for 5+ points
    DEGRADATION = "degradation"  # Success rate drops >0.15
    STRATEGY_SHIFT = "strategy_shift"  # Strategy changes mid-sequence


@dataclass
class AttackPoint:
    """Single point in attack trajectory timeline.

    Represents the state of attacks at one time period (query, batch, or session).
    Contains both success metrics and temporal metadata.
    """

    index: int  # Sequence position
    strategy: str  # Attack strategy name ("prompt_injection", etc.)
    success_rate: float  # 0.0-1.0 where 1.0 = all bypassed guardrails
    attack_count: int  # Total attacks attempted in this period
    bypass_count: int  # Number that successfully bypassed
    avg_severity: float  # 0.0-1.0 average severity of bypassed attacks
    timestamp: Optional[float] = None  # Unix timestamp (optional)
    metadata: Optional[Dict[str, Any]] = None  # Extra context

    def __post_init__(self):
        """Validate data consistency."""
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError(f"success_rate must be 0.0-1.0, got {self.success_rate}")
        if not (0.0 <= self.avg_severity <= 1.0):
            raise ValueError(f"avg_severity must be 0.0-1.0, got {self.avg_severity}")
        if self.bypass_count > self.attack_count:
            raise ValueError(
                f"bypass_count ({self.bypass_count}) cannot exceed "
                f"attack_count ({self.attack_count})"
            )
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyMetrics:
    """Aggregated metrics for an attack strategy across all points."""

    strategy: str
    total_attacks: int
    total_bypasses: int
    bypass_rate: float  # total_bypasses / total_attacks
    avg_severity: float  # Mean severity across all attacks
    trend: str  # "improving" (rising), "degrading" (falling), "stable"
    sparkline_data: List[float]  # Success rates for inline visualization
    min_success: float = field(default=0.0)
    max_success: float = field(default=0.0)
    points_count: int = field(default=0)

    def __post_init__(self):
        """Compute derived metrics."""
        if self.sparkline_data:
            self.min_success = min(self.sparkline_data)
            self.max_success = max(self.sparkline_data)
            self.points_count = len(self.sparkline_data)


class AttackTrajectoryRenderer:
    """Render attack effectiveness over time using Tufte principles.

    Produces pure HTML/CSS/SVG output with:
    - Main chart: Success rate over time with strategy overlay
    - Anomaly markers: Sudden breakthroughs, plateaus, degradation
    - Strategy sparklines: Inline comparison via small multiples
    - Statistics panel: Key metrics with delta indicators
    - No external dependencies: All CSS/SVG inline

    Philosophy: Maximum data-ink ratio, minimal decoration, meaning first.
    """

    # Color palette (semantic, accessible)
    COLOR_SUCCESS = "#2ecc71"  # Green: attack succeeded
    COLOR_BLOCKED = "#e74c3c"  # Red: attack blocked
    COLOR_NEUTRAL = "#95a5a6"  # Gray: neutral metrics
    COLOR_WARNING = "#f39c12"  # Orange: warning/anomaly
    COLOR_CRITICAL = "#c0392b"  # Dark red: critical
    COLOR_SECONDARY = "#3498db"  # Blue: secondary data

    # Anomaly colors (based on type)
    ANOMALY_COLORS = {
        AnomalyType.SUDDEN_BREAKTHROUGH: "#e74c3c",  # Red: attack success
        AnomalyType.SUSTAINED_SUCCESS: "#c0392b",  # Dark red: sustained threat
        AnomalyType.PLATEAU: "#f39c12",  # Orange: concerning pattern
        AnomalyType.DEGRADATION: "#2ecc71",  # Green: guardrails improving
        AnomalyType.STRATEGY_SHIFT: "#3498db",  # Blue: tactical change
    }

    def __init__(
        self,
        detect_anomalies: bool = True,
        show_strategy_breakdown: bool = True,
        max_width: int = 1200,
        chart_height: int = 300,
    ):
        """Initialize renderer.

        Args:
            detect_anomalies: Enable anomaly detection and markers
            show_strategy_breakdown: Show strategy comparison sparklines
            max_width: Maximum SVG width in pixels
            chart_height: Chart height in pixels
        """
        self.detect_anomalies = detect_anomalies
        self.show_strategy_breakdown = show_strategy_breakdown
        self.max_width = max_width
        self.chart_height = chart_height

    def render(
        self,
        points: List[AttackPoint],
        title: str = "Attack Trajectory",
        subtitle: Optional[str] = None,
    ) -> str:
        """Render complete HTML visualization.

        Args:
            points: Trajectory data points
            title: Main title
            subtitle: Optional subtitle

        Returns:
            Complete HTML string ready for display
        """
        if not points:
            return self._render_empty_state()

        # Compute statistics
        metrics = self._compute_metrics(points)
        anomalies = (
            self._detect_anomalies(points) if self.detect_anomalies else []
        )

        # Group by strategy
        by_strategy = self._group_by_strategy(points)
        strategy_metrics = self._compute_strategy_metrics(by_strategy)

        # Build HTML sections
        html_parts = [
            self._render_html_header(),
            self._render_styles(),
            '<body>',
            self._render_header(title, subtitle, metrics),
            self._render_chart(points, anomalies),
            self._render_statistics(metrics),
        ]

        if self.show_strategy_breakdown and strategy_metrics:
            html_parts.append(
                self._render_strategy_sparklines(strategy_metrics)
            )

        html_parts.extend(['</body>', '</html>'])

        return '\n'.join(html_parts)

    # ========== Header & Title ==========

    def _render_html_header(self) -> str:
        """Render HTML5 document header."""
        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '  <meta charset="UTF-8">\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '  <title>Attack Trajectory</title>\n'
            '</head>'
        )

    def _render_header(
        self,
        title: str,
        subtitle: Optional[str],
        metrics: Dict[str, Any],
    ) -> str:
        """Render title, subtitle, and key metrics.

        Design: Tufte-style header with metric badges
        """
        overall_rate = metrics.get("overall_bypass_rate", 0.0)
        html = f'''<div class="header">
  <h1>{title}</h1>'''

        if subtitle:
            html += f'\n  <p class="subtitle">{subtitle}</p>'

        # Metric badges (high data density)
        html += f'''
  <div class="metrics-badges">
    <span class="badge bypass-rate" title="Overall attack success rate">
      {overall_rate:.1%} Bypass Rate
    </span>
    <span class="badge attack-count" title="Total attacks analyzed">
      {metrics.get('total_attacks', 0)} Attacks
    </span>
    <span class="badge success-count" title="Successful bypasses">
      {metrics.get('total_bypasses', 0)} Bypassed
    </span>
    <span class="badge severity" title="Average severity of successful attacks">
      {metrics.get('avg_severity', 0):.2f} Avg Severity
    </span>
  </div>
</div>'''

        return html

    # ========== Chart Rendering ==========

    def _render_chart(
        self,
        points: List[AttackPoint],
        anomalies: List[Tuple[int, AnomalyType]],
    ) -> str:
        """Render main SVG chart with success rate trajectory.

        Design: Line chart with semantic colors, anomaly markers, legend
        Tufte principle: Show the data, minimize decoration
        """
        if not points:
            return ""

        # Chart dimensions
        margin = 50
        width = self.max_width - 2 * margin
        height = self.chart_height
        chart_area_width = width - margin
        chart_area_height = height - margin

        # Compute scales
        max_rate = max(p.success_rate for p in points) or 1.0
        max_rate = max(max_rate, 0.3)  # Minimum scale
        x_scale = chart_area_width / max(len(points) - 1, 1)
        y_scale = chart_area_height / max_rate

        # Build path data for line chart
        path_data = self._build_line_path(points, x_scale, y_scale, margin)

        # Anomaly markers
        anomaly_marks = self._build_anomaly_marks(
            anomalies, points, x_scale, y_scale, margin
        )

        # SVG
        svg = f'''<svg class="attack-chart" width="{self.max_width}" height="{height + 100}">
  <!-- Background grid -->
  <defs>
    <pattern id="grid" width="50" height="25" patternUnits="userSpaceOnUse">
      <path d="M 50 0 L 0 0 0 25" fill="none" stroke="#ecf0f1" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect width="{self.max_width}" height="{height + 100}" fill="white"/>
  <rect x="{margin}" y="{margin}" width="{chart_area_width}" height="{chart_area_height}" fill="url(#grid)"/>

  <!-- Axis lines -->
  <line x1="{margin}" y1="{margin + chart_area_height}" x2="{margin + chart_area_width}" y2="{margin + chart_area_height}" stroke="#34495e" stroke-width="1.5"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + chart_area_height}" stroke="#34495e" stroke-width="1.5"/>

  <!-- Y-axis labels (success rate %) -->
  <text x="{margin - 30}" y="{margin + 5}" font-size="11" fill="#34495e" text-anchor="end">100%</text>
  <text x="{margin - 30}" y="{margin + chart_area_height/2 + 5}" font-size="11" fill="#34495e" text-anchor="end">50%</text>
  <text x="{margin - 30}" y="{margin + chart_area_height + 5}" font-size="11" fill="#34495e" text-anchor="end">0%</text>

  <!-- Reference lines (50% threshold) -->
  <line x1="{margin}" y1="{margin + chart_area_height/2}" x2="{margin + chart_area_width}" y2="{margin + chart_area_height/2}" stroke="#bdc3c7" stroke-width="0.5" stroke-dasharray="2,2"/>
  <text x="{margin - 5}" y="{margin + chart_area_height/2 - 5}" font-size="10" fill="#7f8c8d" text-anchor="end">50%</text>

  <!-- Main line chart -->
  <path d="{path_data}" fill="none" stroke="{self.COLOR_SUCCESS}" stroke-width="2.5" stroke-linecap="round"/>

  <!-- Data points -->
'''

        # Add point circles
        for i, point in enumerate(points):
            x = margin + i * x_scale
            y = margin + chart_area_height - (point.success_rate * y_scale)
            color = (
                self.COLOR_BLOCKED
                if point.success_rate > 0.5
                else self.COLOR_SUCCESS
            )
            svg += (
                f'  <circle cx="{x}" cy="{y}" r="3" fill="{color}" '
                f'stroke="white" stroke-width="1" '
                f'title="{point.strategy}: {point.success_rate:.1%}"/>\n'
            )

        # Anomaly markers
        svg += anomaly_marks

        # X-axis labels (every 5 points)
        for i in range(0, len(points), max(1, len(points) // 10)):
            x = margin + i * x_scale
            svg += (
                f'  <text x="{x}" y="{margin + chart_area_height + 20}" '
                f'font-size="10" fill="#7f8c8d" text-anchor="middle">{i}</text>\n'
            )

        # Legend
        svg += f'''  <!-- Legend -->
  <g class="legend">
    <rect x="{margin + 10}" y="{margin + 10}" width="150" height="80" fill="white" stroke="#bdc3c7" stroke-width="0.5"/>
    <circle cx="{margin + 25}" cy="{margin + 25}" r="2.5" fill="{self.COLOR_BLOCKED}"/>
    <text x="{margin + 40}" y="{margin + 30}" font-size="11" fill="#34495e">Attack Succeeded</text>

    <circle cx="{margin + 25}" cy="{margin + 45}" r="2.5" fill="{self.COLOR_SUCCESS}"/>
    <text x="{margin + 40}" y="{margin + 50}" font-size="11" fill="#34495e">Attack Blocked</text>

    <line x1="{margin + 20}" y1="{margin + 65}" x2="{margin + 30}" y2="{margin + 65}" stroke="{self.COLOR_WARNING}" stroke-width="2"/>
    <text x="{margin + 40}" y="{margin + 70}" font-size="11" fill="#34495e">Anomaly</text>
  </g>
'''

        svg += '</svg>'

        return svg

    def _build_line_path(
        self,
        points: List[AttackPoint],
        x_scale: float,
        y_scale: float,
        margin: int,
    ) -> str:
        """Build SVG path data for success rate line.

        Uses smooth curve (quadratic Bezier) for visual appeal.
        """
        if not points:
            return ""

        path_parts = []
        for i, point in enumerate(points):
            x = margin + i * x_scale
            y = margin + self.chart_height - margin - (point.success_rate * y_scale)

            if i == 0:
                path_parts.append(f"M {x} {y}")
            else:
                # Quadratic Bezier for smooth curve
                prev_x = margin + (i - 1) * x_scale
                control_x = (prev_x + x) / 2
                path_parts.append(f"Q {control_x} {path_parts[-1].split()[-1]} {x} {y}")

        return " ".join(path_parts)

    def _build_anomaly_marks(
        self,
        anomalies: List[Tuple[int, AnomalyType]],
        points: List[AttackPoint],
        x_scale: float,
        y_scale: float,
        margin: int,
    ) -> str:
        """Build SVG markers for detected anomalies."""
        if not anomalies:
            return ""

        marks = '<g class="anomalies">\n'

        for point_idx, anomaly_type in anomalies:
            if point_idx >= len(points):
                continue

            point = points[point_idx]
            x = margin + point_idx * x_scale
            y = margin + self.chart_height - margin - (point.success_rate * y_scale)
            color = self.ANOMALY_COLORS.get(anomaly_type, self.COLOR_WARNING)

            # Marker: Diamond shape for anomalies
            size = 5
            marks += (
                f'  <polygon points="{x},{y-size} {x+size},{y} '
                f'{x},{y+size} {x-size},{y}" '
                f'fill="{color}" opacity="0.8" '
                f'title="{anomaly_type.value} at point {point_idx}"/>\n'
            )

        marks += '</g>\n'
        return marks

    # ========== Statistics Panel ==========

    def _render_statistics(self, metrics: Dict[str, Any]) -> str:
        """Render statistics panel with key metrics and trends.

        Design: High data density table following Tufte principles
        """
        html = '<div class="statistics">'
        html += '<h2>Key Metrics</h2>'
        html += '<table class="metrics-table">'

        # Rows: metric name, value, delta indicator
        rows = [
            (
                "Overall Bypass Rate",
                f"{metrics.get('overall_bypass_rate', 0):.1%}",
                metrics.get("bypass_rate_trend", "stable"),
            ),
            (
                "Total Attacks",
                f"{metrics.get('total_attacks', 0)}",
                "neutral",
            ),
            (
                "Successful Bypasses",
                f"{metrics.get('total_bypasses', 0)}",
                metrics.get("bypass_trend", "stable"),
            ),
            (
                "Average Severity",
                f"{metrics.get('avg_severity', 0):.2f}",
                metrics.get("severity_trend", "stable"),
            ),
            (
                "Peak Success Rate",
                f"{metrics.get('max_success_rate', 0):.1%}",
                "neutral",
            ),
            (
                "Success Volatility",
                f"{metrics.get('success_volatility', 0):.3f}",
                "neutral",
            ),
        ]

        for name, value, trend in rows:
            trend_indicator = self._get_trend_indicator(trend)
            html += f'''<tr>
    <td class="metric-name">{name}</td>
    <td class="metric-value">{value}</td>
    <td class="trend-indicator">{trend_indicator}</td>
  </tr>'''

        html += '</table>\n</div>'

        return html

    def _get_trend_indicator(self, trend: str) -> str:
        """Get visual trend indicator (↑ ↓ →)."""
        indicators = {
            "improving": '↓ Improving',  # Lower success = better
            "degrading": '↑ Degrading',  # Higher success = worse
            "stable": '→ Stable',
            "neutral": "—",
        }
        return indicators.get(trend, "—")

    # ========== Strategy Breakdown ==========

    def _render_strategy_sparklines(
        self,
        strategy_metrics: Dict[str, StrategyMetrics],
    ) -> str:
        """Render strategy comparison using small multiples (sparklines).

        Design: Each strategy gets a row with:
        - Name and key metrics
        - Inline sparkline (word-sized chart)
        - Trend indicator
        - Bypass rate with color coding

        Following Tufte's small multiples principle.
        """
        html = '<div class="strategy-breakdown">'
        html += '<h2>Strategy Comparison</h2>'
        html += '<table class="strategy-table">'
        html += '<thead><tr>'
        html += '<th>Strategy</th>'
        html += '<th>Bypass Rate</th>'
        html += '<th>Attacks</th>'
        html += '<th class="sparkline-col">Trend</th>'
        html += '<th>Status</th>'
        html += '</tr></thead>'
        html += '<tbody>'

        for strategy, metrics in sorted(
            strategy_metrics.items(), key=lambda x: x[1].bypass_rate, reverse=True
        ):
            m = metrics
            rate_color = (
                self.COLOR_BLOCKED if m.bypass_rate > 0.5 else self.COLOR_SUCCESS
            )

            # Build sparkline SVG
            sparkline_svg = self._build_sparkline(
                m.sparkline_data, 80, 20, "strategy-sparkline"
            )

            html += f'''<tr>
    <td class="strategy-name">{strategy}</td>
    <td class="bypass-rate" style="color: {rate_color}; font-weight: bold;">{m.bypass_rate:.1%}</td>
    <td class="attack-count">{m.total_attacks}</td>
    <td class="sparkline-col">{sparkline_svg}</td>
    <td class="trend-badge">{self._get_trend_indicator(m.trend)}</td>
  </tr>'''

        html += '</tbody>\n</table>'
        html += '</div>'

        return html

    def _build_sparkline(
        self,
        data: List[float],
        width: int,
        height: int,
        css_class: str,
    ) -> str:
        """Build inline SVG sparkline (word-sized chart).

        Tufte principle: Show trend without axis labels or titles.
        """
        if not data or len(data) < 2:
            return '<span class="sparkline-empty">—</span>'

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val or 1.0

        x_step = (width - 4) / (len(data) - 1)
        y_scale = (height - 4) / range_val

        # Build path
        path_points = []
        for i, val in enumerate(data):
            x = 2 + i * x_step
            y = height - 2 - ((val - min_val) * y_scale)
            path_points.append(f"{x},{y}")

        path_data = "M " + " L ".join(path_points)

        # Determine color: green if trending down (less attacks), red if up
        trend_direction = "down" if data[-1] < data[0] else "up"
        line_color = (
            self.COLOR_SUCCESS if trend_direction == "down" else self.COLOR_BLOCKED
        )

        svg = (
            f'<svg class="{css_class}" width="{width}" height="{height}" '
            f'style="vertical-align: middle; margin: 0 2px;">'
            f'<path d="{path_data}" stroke="{line_color}" stroke-width="1.5" '
            f'fill="none" stroke-linecap="round"/>'
            f'<circle cx="2" cy="{height - 2 - ((data[0] - min_val) * y_scale)}" '
            f'r="1" fill="{line_color}"/>'
            f'<circle cx="{2 + (len(data) - 1) * x_step}" '
            f'cy="{height - 2 - ((data[-1] - min_val) * y_scale)}" '
            f'r="1" fill="{line_color}"/>'
            f'</svg>'
        )

        return svg

    # ========== Styles ==========

    def _render_styles(self) -> str:
        """Render inline CSS with Tufte-inspired design.

        Principles:
        - High contrast for readability
        - Monospace for numbers (easier comparison)
        - Minimal decoration
        - Generous whitespace
        """
        return '''<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.6;
  color: #34495e;
  background: #f8f9fa;
  padding: 40px 20px;
}

.header {
  max-width: 1200px;
  margin: 0 auto 40px;
  padding-bottom: 20px;
  border-bottom: 2px solid #34495e;
}

.header h1 {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 5px;
  color: #1a252f;
}

.header .subtitle {
  font-size: 14px;
  color: #7f8c8d;
  font-style: italic;
  margin-bottom: 15px;
}

.metrics-badges {
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
}

.badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: 500;
  background: #ecf0f1;
  border-left: 3px solid #95a5a6;
}

.badge.bypass-rate {
  border-left-color: #e74c3c;
  background: #fadbd8;
}

.badge.attack-count {
  border-left-color: #3498db;
  background: #d6eaf8;
}

.badge.success-count {
  border-left-color: #2ecc71;
  background: #d5f4e6;
}

.badge.severity {
  border-left-color: #f39c12;
  background: #fdebd0;
}

.attack-chart {
  max-width: 1200px;
  margin: 0 auto 40px;
  border: 1px solid #bdc3c7;
  background: white;
}

.chart-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 10px;
}

.statistics,
.strategy-breakdown {
  max-width: 1200px;
  margin: 0 auto 40px;
  background: white;
  padding: 25px;
  border: 1px solid #bdc3c7;
}

.statistics h2,
.strategy-breakdown h2 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid #ecf0f1;
  color: #1a252f;
}

.metrics-table,
.strategy-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.metrics-table td,
.strategy-table td,
.strategy-table th {
  padding: 8px 12px;
  border-bottom: 1px solid #ecf0f1;
  text-align: left;
}

.metrics-table td:nth-child(2),
.strategy-table td:nth-child(2),
.strategy-table td:nth-child(3) {
  text-align: right;
  font-family: "Courier New", monospace;
}

.metric-name,
.strategy-name {
  font-weight: 500;
  color: #34495e;
}

.metric-value {
  font-weight: 600;
  font-family: "Courier New", monospace;
}

.trend-indicator {
  text-align: center;
  font-size: 11px;
  color: #7f8c8d;
}

.trend-badge {
  font-size: 11px;
  text-align: center;
}

.strategy-table th {
  background: #f8f9fa;
  font-weight: 600;
  color: #1a252f;
  border-bottom: 2px solid #34495e;
}

.strategy-table tbody tr:hover {
  background: #f8f9fa;
}

.sparkline-col {
  text-align: center;
}

.sparkline-empty {
  color: #bdc3c7;
  font-size: 11px;
}

/* Legend styling */
.legend {
  font-size: 11px;
}

/* Empty state */
.empty-state {
  max-width: 1200px;
  margin: 40px auto;
  padding: 40px;
  text-align: center;
  background: white;
  border: 1px solid #bdc3c7;
  border-radius: 4px;
  color: #7f8c8d;
}

/* Responsive design */
@media (max-width: 768px) {
  body {
    padding: 20px 10px;
  }

  .header h1 {
    font-size: 24px;
  }

  .metrics-badges {
    flex-direction: column;
  }

  .badge {
    width: 100%;
  }

  .strategy-table {
    font-size: 11px;
  }

  .strategy-table td,
  .strategy-table th {
    padding: 6px 8px;
  }
}
</style>'''

    # ========== Empty State ==========

    def _render_empty_state(self) -> str:
        """Render empty state when no data available."""
        return f'''{self._render_html_header()}
{self._render_styles()}
<body>
<div class="empty-state">
  <h1>No Data Available</h1>
  <p>Provide attack trajectory points to visualize</p>
</div>
</body>
</html>'''

    # ========== Data Analysis ==========

    def _compute_metrics(self, points: List[AttackPoint]) -> Dict[str, Any]:
        """Compute overall metrics from attack trajectory."""
        if not points:
            return {}

        success_rates = [p.success_rate for p in points]
        severities = [p.avg_severity for p in points]

        total_attacks = sum(p.attack_count for p in points)
        total_bypasses = sum(p.bypass_count for p in points)
        overall_rate = total_bypasses / total_attacks if total_attacks > 0 else 0.0

        # Compute volatility (std dev of success rates)
        mean_success = sum(success_rates) / len(success_rates)
        variance = sum((x - mean_success) ** 2 for x in success_rates) / len(
            success_rates
        )
        volatility = math.sqrt(variance)

        # Trend detection
        if len(success_rates) >= 2:
            early_avg = sum(success_rates[: len(success_rates) // 2]) / (
                len(success_rates) // 2
            )
            late_avg = sum(success_rates[len(success_rates) // 2 :]) / (
                len(success_rates) - len(success_rates) // 2
            )
            bypass_trend = "degrading" if late_avg > early_avg else "improving"
        else:
            bypass_trend = "stable"

        return {
            "overall_bypass_rate": overall_rate,
            "total_attacks": total_attacks,
            "total_bypasses": total_bypasses,
            "avg_severity": sum(severities) / len(severities) if severities else 0,
            "max_success_rate": max(success_rates),
            "min_success_rate": min(success_rates),
            "success_volatility": volatility,
            "bypass_rate_trend": bypass_trend,
            "bypass_trend": bypass_trend,
            "severity_trend": "neutral",
        }

    def _detect_anomalies(
        self, points: List[AttackPoint]
    ) -> List[Tuple[int, AnomalyType]]:
        """Detect anomalies in attack trajectory.

        Detects:
        - Sudden breakthrough: Success rate spike >0.2
        - Sustained success: High success (>0.6) for 3+ consecutive points
        - Plateau: Success rate stable ±0.05 for 5+ points
        - Degradation: Success rate drops >0.15
        - Strategy shift: Strategy changes mid-sequence
        """
        anomalies = []

        if len(points) < 2:
            return anomalies

        # Sudden breakthroughs
        for i in range(1, len(points)):
            delta = points[i].success_rate - points[i - 1].success_rate
            if delta > 0.2:  # Success jumped >20%
                anomalies.append((i, AnomalyType.SUDDEN_BREAKTHROUGH))

        # Sustained success
        for i in range(2, len(points)):
            window = points[max(0, i - 2) : i + 1]
            if all(p.success_rate > 0.6 for p in window):
                anomalies.append((i, AnomalyType.SUSTAINED_SUCCESS))

        # Plateaus (stable region)
        for i in range(4, len(points)):
            window = points[i - 4 : i + 1]
            rates = [p.success_rate for p in window]
            if max(rates) - min(rates) < 0.05:  # All within ±2.5%
                anomalies.append((i, AnomalyType.PLATEAU))

        # Degradation (drop in success = improvement for us)
        for i in range(1, len(points)):
            delta = points[i - 1].success_rate - points[i].success_rate
            if delta > 0.15:  # Success dropped >15%
                anomalies.append((i, AnomalyType.DEGRADATION))

        # Strategy shifts
        for i in range(1, len(points)):
            if points[i].strategy != points[i - 1].strategy:
                anomalies.append((i, AnomalyType.STRATEGY_SHIFT))

        return anomalies

    def _group_by_strategy(self, points: List[AttackPoint]) -> Dict[str, List[AttackPoint]]:
        """Group points by attack strategy."""
        by_strategy: Dict[str, List[AttackPoint]] = {}
        for point in points:
            if point.strategy not in by_strategy:
                by_strategy[point.strategy] = []
            by_strategy[point.strategy].append(point)
        return by_strategy

    def _compute_strategy_metrics(
        self, by_strategy: Dict[str, List[AttackPoint]]
    ) -> Dict[str, StrategyMetrics]:
        """Compute aggregated metrics for each strategy."""
        metrics = {}

        for strategy, points in by_strategy.items():
            if not points:
                continue

            total_attacks = sum(p.attack_count for p in points)
            total_bypasses = sum(p.bypass_count for p in points)
            bypass_rate = (
                total_bypasses / total_attacks if total_attacks > 0 else 0.0
            )

            success_rates = [p.success_rate for p in points]

            # Trend detection
            if len(success_rates) >= 2:
                early_avg = sum(success_rates[: len(success_rates) // 2]) / max(
                    1, len(success_rates) // 2
                )
                late_avg = sum(success_rates[len(success_rates) // 2 :]) / max(
                    1, len(success_rates) - len(success_rates) // 2
                )
                trend = "improving" if late_avg < early_avg else (
                    "degrading" if late_avg > early_avg else "stable"
                )
            else:
                trend = "stable"

            metrics[strategy] = StrategyMetrics(
                strategy=strategy,
                total_attacks=total_attacks,
                total_bypasses=total_bypasses,
                bypass_rate=bypass_rate,
                avg_severity=sum(p.avg_severity for p in points) / len(points),
                trend=trend,
                sparkline_data=success_rates,
            )

        return metrics


# ========== Convenience Function ==========


def render_attack_trajectory(
    strategies: List[str],
    success_rates: List[float],
    attack_counts: List[int],
    bypass_counts: List[int],
    title: str = "Attack Trajectory",
    subtitle: Optional[str] = None,
    detect_anomalies: bool = True,
    show_strategy_breakdown: bool = True,
) -> str:
    """Convenience function to render trajectory from simple lists.

    Args:
        strategies: Strategy name for each point
        success_rates: Success rate (0.0-1.0) for each point
        attack_counts: Total attacks for each point
        bypass_counts: Successful bypasses for each point
        title: Visualization title
        subtitle: Optional subtitle
        detect_anomalies: Enable anomaly detection
        show_strategy_breakdown: Show strategy comparison

    Returns:
        Complete HTML string

    Example:
        >>> strategies = ["prompt_injection", "jailbreak", "overflow"]
        >>> success_rates = [0.65, 0.42, 0.28]
        >>> attack_counts = [10, 10, 10]
        >>> bypass_counts = [6, 4, 3]
        >>>
        >>> html = render_attack_trajectory(
        ...     strategies, success_rates, attack_counts, bypass_counts,
        ...     title="Red Team Attack Analysis"
        ... )
    """
    if not all(
        len(x) == len(strategies)
        for x in [success_rates, attack_counts, bypass_counts]
    ):
        raise ValueError("All lists must have same length")

    points = [
        AttackPoint(
            index=i,
            strategy=strategies[i],
            success_rate=success_rates[i],
            attack_count=attack_counts[i],
            bypass_count=bypass_counts[i],
            avg_severity=bypass_counts[i] / attack_counts[i]
            if attack_counts[i] > 0
            else 0.0,
        )
        for i in range(len(strategies))
    ]

    renderer = AttackTrajectoryRenderer(
        detect_anomalies=detect_anomalies,
        show_strategy_breakdown=show_strategy_breakdown,
    )

    return renderer.render(points, title=title, subtitle=subtitle)
