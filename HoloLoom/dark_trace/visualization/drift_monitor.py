"""
Drift Monitor Visualization for Dark Trace

Real-time visualization of concept drift, feature activation drift,
and distribution shifts in model behavior over time.

Author: HoloLoom Team
Created: December 2025
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import math


class DriftType(Enum):
    """Types of drift that can be detected."""
    CONCEPT = "concept"           # Semantic meaning shift
    ACTIVATION = "activation"     # Feature activation distribution shift
    DISTRIBUTION = "distribution" # Input distribution shift
    BEHAVIORAL = "behavioral"     # Output behavior shift
    CONFIDENCE = "confidence"     # Confidence calibration drift
    TEMPORAL = "temporal"         # Time-based pattern drift


class DriftSeverity(Enum):
    """Severity levels for drift events."""
    INFO = "info"           # Minor drift, informational
    WARNING = "warning"     # Moderate drift, needs monitoring
    CRITICAL = "critical"   # Severe drift, action required
    EMERGENCY = "emergency" # Extreme drift, immediate action


@dataclass
class DriftEvent:
    """A single drift event."""
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    drift_magnitude: float  # How much drift occurred (0-1 normalized)
    affected_features: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def drift_percentage(self) -> float:
        """Get drift as percentage change."""
        if self.baseline_value == 0:
            return 100.0 if self.current_value != 0 else 0.0
        return abs((self.current_value - self.baseline_value) / self.baseline_value) * 100

    @property
    def color(self) -> str:
        """Get color based on severity."""
        colors = {
            DriftSeverity.INFO: "#00bcd4",
            DriftSeverity.WARNING: "#ff9800",
            DriftSeverity.CRITICAL: "#e94560",
            DriftSeverity.EMERGENCY: "#d32f2f",
        }
        return colors.get(self.severity, "#888888")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "drift_magnitude": self.drift_magnitude,
            "drift_percentage": self.drift_percentage,
            "affected_features": self.affected_features,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class DriftMonitorConfig:
    """Configuration for drift monitor visualization."""
    width: int = 900
    height: int = 400
    timeline_height: int = 200
    chart_height: int = 150
    warning_threshold: float = 0.3
    critical_threshold: float = 0.6
    emergency_threshold: float = 0.8
    show_legend: bool = True
    show_threshold_lines: bool = True
    auto_refresh_seconds: int = 0  # 0 = no auto-refresh


class DriftMonitor:
    """
    Real-time drift monitoring visualization.

    Displays drift events over time with severity indicators,
    trend lines, and threshold markers.
    """

    def __init__(self, config: Optional[DriftMonitorConfig] = None):
        self.config = config or DriftMonitorConfig()
        self.events: List[DriftEvent] = []
        self.baseline_metrics: Dict[str, float] = {}

    def add_event(self, event: DriftEvent) -> None:
        """Add a drift event."""
        self.events.append(event)

    def add_events(self, events: List[DriftEvent]) -> None:
        """Add multiple drift events."""
        self.events.extend(events)

    def set_baseline(self, metric_name: str, value: float) -> None:
        """Set baseline value for a metric."""
        self.baseline_metrics[metric_name] = value

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()

    def get_events_by_type(self, drift_type: DriftType) -> List[DriftEvent]:
        """Get events filtered by type."""
        return [e for e in self.events if e.drift_type == drift_type]

    def get_events_by_severity(self, severity: DriftSeverity) -> List[DriftEvent]:
        """Get events filtered by severity."""
        return [e for e in self.events if e.severity == severity]

    def render(
        self,
        title: str = "Drift Monitor",
        subtitle: Optional[str] = None,
        show_timeline: bool = True,
        show_magnitude_chart: bool = True,
    ) -> str:
        """Render drift monitor as HTML."""
        # Sort events by timestamp
        sorted_events = sorted(self.events, key=lambda e: e.timestamp)

        # Build components
        timeline_svg = self._render_timeline(sorted_events) if show_timeline else ""
        magnitude_svg = self._render_magnitude_chart(sorted_events) if show_magnitude_chart else ""
        legend_html = self._render_legend() if self.config.show_legend else ""
        events_table = self._render_events_table(sorted_events[-20:])  # Last 20 events
        summary_stats = self._render_summary_stats(sorted_events)

        subtitle_html = f'<p style="color: #888; margin: 0;">{subtitle}</p>' if subtitle else ""

        auto_refresh_script = ""
        if self.config.auto_refresh_seconds > 0:
            auto_refresh_script = f"""
            <script>
                setTimeout(function() {{
                    location.reload();
                }}, {self.config.auto_refresh_seconds * 1000});
            </script>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #e94560;
            --border: #333;
            --success: #00bcd4;
            --warning: #ff9800;
            --critical: #e94560;
            --emergency: #d32f2f;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--accent);
            margin-bottom: 5px;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
        }}
        .card-title {{
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin: 10px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: #888;
            font-weight: normal;
        }}
        .severity-badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            text-transform: uppercase;
        }}
        .severity-info {{ background: rgba(0, 188, 212, 0.2); color: #00bcd4; }}
        .severity-warning {{ background: rgba(255, 152, 0, 0.2); color: #ff9800; }}
        .severity-critical {{ background: rgba(233, 69, 96, 0.2); color: #e94560; }}
        .severity-emergency {{ background: rgba(211, 47, 47, 0.2); color: #d32f2f; }}
        .drift-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }}
        .drift-fill {{
            height: 100%;
            border-radius: 4px;
        }}
        svg {{
            display: block;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{title}</h1>
    {subtitle_html}

    <div class="dashboard">
        {summary_stats}

        <div class="card">
            <div class="card-title">Drift Timeline</div>
            {legend_html}
            {timeline_svg}
        </div>

        <div class="card">
            <div class="card-title">Drift Magnitude Over Time</div>
            {magnitude_svg}
        </div>

        <div class="card">
            <div class="card-title">Recent Events</div>
            {events_table}
        </div>
    </div>
</div>
{auto_refresh_script}
</body>
</html>
"""

    def _render_timeline(self, events: List[DriftEvent]) -> str:
        """Render timeline visualization."""
        if not events:
            return '<p style="color: #888; text-align: center;">No drift events recorded</p>'

        width = self.config.width
        height = self.config.timeline_height
        padding = 50

        # Calculate time range
        times = [e.timestamp.timestamp() for e in events]
        min_time, max_time = min(times), max(times)
        time_range = max_time - min_time if max_time > min_time else 1

        # Group events by type for lanes
        lanes = {}
        for dt in DriftType:
            lane_events = [e for e in events if e.drift_type == dt]
            if lane_events:
                lanes[dt] = lane_events

        lane_height = (height - 2 * padding) / max(len(lanes), 1)

        # Build SVG
        elements = []

        # Draw lanes
        for i, (drift_type, lane_events) in enumerate(lanes.items()):
            y = padding + i * lane_height + lane_height / 2

            # Lane label
            elements.append(
                f'<text x="5" y="{y + 4}" font-size="10" fill="#888">'
                f'{drift_type.value[:8]}</text>'
            )

            # Lane line
            elements.append(
                f'<line x1="{padding}" y1="{y}" x2="{width - 10}" y2="{y}" '
                f'stroke="#333" stroke-width="1" stroke-dasharray="2,2"/>'
            )

            # Event markers
            for event in lane_events:
                x = padding + ((event.timestamp.timestamp() - min_time) / time_range) * (width - padding - 10)
                radius = 5 + event.drift_magnitude * 10

                elements.append(
                    f'<circle cx="{x}" cy="{y}" r="{radius}" '
                    f'fill="{event.color}" fill-opacity="0.7" stroke="{event.color}" '
                    f'stroke-width="2">'
                    f'<title>{event.metric_name}: {event.drift_magnitude:.2f} ({event.severity.value})</title>'
                    f'</circle>'
                )

        # Time axis
        elements.append(
            f'<line x1="{padding}" y1="{height - 20}" x2="{width - 10}" y2="{height - 20}" '
            f'stroke="#666" stroke-width="1"/>'
        )

        # Time labels
        for i in range(5):
            x = padding + i * (width - padding - 10) / 4
            t = min_time + i * time_range / 4
            time_str = datetime.fromtimestamp(t).strftime("%H:%M")
            elements.append(
                f'<text x="{x}" y="{height - 5}" font-size="10" fill="#888" '
                f'text-anchor="middle">{time_str}</text>'
            )

        return f'<svg width="{width}" height="{height}">{"".join(elements)}</svg>'

    def _render_magnitude_chart(self, events: List[DriftEvent]) -> str:
        """Render magnitude over time chart."""
        if not events:
            return ""

        width = self.config.width
        height = self.config.chart_height
        padding = 50

        times = [e.timestamp.timestamp() for e in events]
        min_time, max_time = min(times), max(times)
        time_range = max_time - min_time if max_time > min_time else 1

        elements = []

        # Threshold lines
        if self.config.show_threshold_lines:
            for threshold, color, label in [
                (self.config.warning_threshold, "#ff9800", "Warning"),
                (self.config.critical_threshold, "#e94560", "Critical"),
                (self.config.emergency_threshold, "#d32f2f", "Emergency"),
            ]:
                y = height - padding - threshold * (height - 2 * padding)
                elements.append(
                    f'<line x1="{padding}" y1="{y}" x2="{width - 10}" y2="{y}" '
                    f'stroke="{color}" stroke-width="1" stroke-dasharray="4,4"/>'
                )
                elements.append(
                    f'<text x="{width - 5}" y="{y - 3}" font-size="9" fill="{color}" '
                    f'text-anchor="end">{label}</text>'
                )

        # Build line path
        if len(events) > 1:
            path_points = []
            for event in events:
                x = padding + ((event.timestamp.timestamp() - min_time) / time_range) * (width - padding - 10)
                y = height - padding - event.drift_magnitude * (height - 2 * padding)
                path_points.append(f"{x},{y}")

            path = "M" + " L".join(path_points)
            elements.append(
                f'<path d="{path}" fill="none" stroke="#e94560" stroke-width="2"/>'
            )

        # Data points
        for event in events:
            x = padding + ((event.timestamp.timestamp() - min_time) / time_range) * (width - padding - 10)
            y = height - padding - event.drift_magnitude * (height - 2 * padding)

            elements.append(
                f'<circle cx="{x}" cy="{y}" r="4" fill="{event.color}">'
                f'<title>{event.metric_name}: {event.drift_magnitude:.3f}</title>'
                f'</circle>'
            )

        # Y-axis
        elements.append(
            f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" '
            f'stroke="#666" stroke-width="1"/>'
        )
        for i in range(5):
            y = padding + i * (height - 2 * padding) / 4
            val = 1.0 - i * 0.25
            elements.append(
                f'<text x="{padding - 5}" y="{y + 4}" font-size="10" fill="#888" '
                f'text-anchor="end">{val:.2f}</text>'
            )

        # X-axis
        elements.append(
            f'<line x1="{padding}" y1="{height - padding}" x2="{width - 10}" y2="{height - padding}" '
            f'stroke="#666" stroke-width="1"/>'
        )

        return f'<svg width="{width}" height="{height}">{"".join(elements)}</svg>'

    def _render_legend(self) -> str:
        """Render legend."""
        items = []
        for severity in DriftSeverity:
            event = DriftEvent(
                timestamp=datetime.now(),
                drift_type=DriftType.CONCEPT,
                severity=severity,
                metric_name="",
                baseline_value=0,
                current_value=0,
                drift_magnitude=0,
            )
            items.append(
                f'<div class="legend-item">'
                f'<div class="legend-dot" style="background: {event.color}"></div>'
                f'{severity.value.title()}'
                f'</div>'
            )

        return f'<div class="legend">{"".join(items)}</div>'

    def _render_summary_stats(self, events: List[DriftEvent]) -> str:
        """Render summary statistics."""
        if not events:
            return '<div class="card"><p style="color: #888;">No events to summarize</p></div>'

        total = len(events)
        by_severity = {s: len([e for e in events if e.severity == s]) for s in DriftSeverity}
        avg_magnitude = sum(e.drift_magnitude for e in events) / total if total > 0 else 0

        # Find most affected features
        feature_counts: Dict[str, int] = {}
        for event in events:
            for f in event.affected_features:
                feature_counts[f] = feature_counts.get(f, 0) + 1

        top_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:3]

        return f"""
        <div class="card">
            <div class="card-title">Summary</div>
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #d32f2f">{by_severity.get(DriftSeverity.EMERGENCY, 0)}</div>
                    <div class="stat-label">Emergency</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #e94560">{by_severity.get(DriftSeverity.CRITICAL, 0)}</div>
                    <div class="stat-label">Critical</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #ff9800">{by_severity.get(DriftSeverity.WARNING, 0)}</div>
                    <div class="stat-label">Warning</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_magnitude:.2f}</div>
                    <div class="stat-label">Avg Magnitude</div>
                </div>
            </div>
        </div>
        """

    def _render_events_table(self, events: List[DriftEvent]) -> str:
        """Render events table."""
        if not events:
            return '<p style="color: #888;">No recent events</p>'

        rows = []
        for event in reversed(events):  # Most recent first
            severity_class = f"severity-{event.severity.value}"
            rows.append(f"""
                <tr>
                    <td>{event.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</td>
                    <td>{event.drift_type.value}</td>
                    <td>{event.metric_name}</td>
                    <td>
                        <div class="drift-bar">
                            <div class="drift-fill" style="width: {event.drift_magnitude * 100}%; background: {event.color}"></div>
                        </div>
                        <span style="font-size: 11px;">{event.drift_magnitude:.3f}</span>
                    </td>
                    <td><span class="severity-badge {severity_class}">{event.severity.value}</span></td>
                </tr>
            """)

        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Type</th>
                        <th>Metric</th>
                        <th>Magnitude</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """

    def to_json(self) -> str:
        """Export events as JSON."""
        return json.dumps({
            "events": [e.to_dict() for e in self.events],
            "baselines": self.baseline_metrics,
            "config": {
                "warning_threshold": self.config.warning_threshold,
                "critical_threshold": self.config.critical_threshold,
                "emergency_threshold": self.config.emergency_threshold,
            }
        }, indent=2)


def render_drift_timeline(
    events: List[DriftEvent],
    title: str = "Drift Timeline",
    show_magnitude_chart: bool = True,
) -> str:
    """
    Convenience function to render a drift timeline.

    Args:
        events: List of drift events to visualize
        title: Chart title
        show_magnitude_chart: Whether to show magnitude chart

    Returns:
        HTML string
    """
    monitor = DriftMonitor()
    monitor.add_events(events)
    return monitor.render(
        title=title,
        show_timeline=True,
        show_magnitude_chart=show_magnitude_chart,
    )


def create_drift_event(
    metric_name: str,
    baseline: float,
    current: float,
    drift_type: DriftType = DriftType.ACTIVATION,
    affected_features: Optional[List[str]] = None,
    threshold_warning: float = 0.3,
    threshold_critical: float = 0.6,
    threshold_emergency: float = 0.8,
) -> DriftEvent:
    """
    Create a drift event with automatic severity classification.

    Args:
        metric_name: Name of the metric that drifted
        baseline: Baseline value
        current: Current value
        drift_type: Type of drift
        affected_features: List of affected feature names
        threshold_*: Severity thresholds

    Returns:
        DriftEvent with computed magnitude and severity
    """
    # Calculate drift magnitude (normalized 0-1)
    if baseline == 0:
        magnitude = 1.0 if current != 0 else 0.0
    else:
        magnitude = min(1.0, abs(current - baseline) / abs(baseline))

    # Determine severity
    if magnitude >= threshold_emergency:
        severity = DriftSeverity.EMERGENCY
    elif magnitude >= threshold_critical:
        severity = DriftSeverity.CRITICAL
    elif magnitude >= threshold_warning:
        severity = DriftSeverity.WARNING
    else:
        severity = DriftSeverity.INFO

    return DriftEvent(
        timestamp=datetime.now(),
        drift_type=drift_type,
        severity=severity,
        metric_name=metric_name,
        baseline_value=baseline,
        current_value=current,
        drift_magnitude=magnitude,
        affected_features=affected_features or [],
        description=f"{metric_name} drifted from {baseline:.4f} to {current:.4f}",
    )
