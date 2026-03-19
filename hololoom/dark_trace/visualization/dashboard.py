"""
Dark Trace Dashboard - Comprehensive Interpretability Dashboard

Unified dashboard combining circuit explorer, feature heatmaps,
drift monitoring, and SAE analysis into a single interactive view.

Author: HoloLoom Team
Created: December 2025
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .circuit_explorer import CircuitEdge, CircuitExplorer, CircuitNode
from .drift_monitor import DriftEvent, DriftMonitor, DriftSeverity
from .feature_heatmap import FeatureHeatmap, HeatmapConfig


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""
    title: str = "Dark Trace Dashboard"
    subtitle: str = ""
    width: int = 1400
    refresh_interval_seconds: int = 0  # 0 = no auto-refresh

    # Panel visibility
    show_circuit_explorer: bool = True
    show_feature_heatmap: bool = True
    show_drift_monitor: bool = True
    show_sae_summary: bool = True
    show_metrics_panel: bool = True
    show_alerts_panel: bool = True

    # Layout
    layout: str = "grid"  # grid, tabs, vertical

    # Theme
    accent_color: str = "#e94560"
    background_color: str = "#1a1a2e"
    card_color: str = "#16213e"


@dataclass
class SAESummary:
    """Summary of SAE analysis."""
    n_features: int
    n_active: int
    sparsity: float
    reconstruction_loss: float
    top_features: list[tuple[int, str, float]] = field(default_factory=list)  # (idx, name, activation)
    feature_correlations: np.ndarray | None = None


@dataclass
class MetricsSnapshot:
    """Snapshot of interpretability metrics."""
    timestamp: datetime
    total_queries: int
    avg_confidence: float
    avg_latency_ms: float
    drift_alerts: int
    safety_blocks: int
    cache_hit_rate: float
    memory_mb: float
    custom_metrics: dict[str, float] = field(default_factory=dict)


class DarkTraceDashboard:
    """
    Comprehensive interpretability dashboard.

    Combines multiple visualization components into a unified
    view with real-time updates and interactive exploration.
    """

    def __init__(self, config: DashboardConfig | None = None):
        self.config = config or DashboardConfig()

        # Component data
        self.circuit_explorer: CircuitExplorer | None = None
        self.feature_activations: np.ndarray | None = None
        self.feature_names: list[str] | None = None
        self.token_names: list[str] | None = None
        self.drift_events: list[DriftEvent] = []
        self.sae_summary: SAESummary | None = None
        self.metrics_history: list[MetricsSnapshot] = []
        self.alerts: list[dict[str, Any]] = []

    def set_circuit(
        self,
        nodes: list[CircuitNode],
        edges: list[CircuitEdge],
        highlighted_path: list[str] | None = None,
    ) -> None:
        """Set circuit explorer data."""
        self.circuit_explorer = CircuitExplorer()
        self.circuit_explorer.add_nodes(nodes)
        self.circuit_explorer.add_edges(edges)
        self._highlighted_path = highlighted_path

    def set_feature_activations(
        self,
        activations: np.ndarray,
        feature_names: list[str] | None = None,
        token_names: list[str] | None = None,
    ) -> None:
        """Set feature activation data for heatmap."""
        self.feature_activations = activations
        self.feature_names = feature_names
        self.token_names = token_names

    def add_drift_event(self, event: DriftEvent) -> None:
        """Add a drift event."""
        self.drift_events.append(event)

        # Auto-create alert for critical/emergency events
        if event.severity in [DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY]:
            self.add_alert(
                level="critical" if event.severity == DriftSeverity.CRITICAL else "emergency",
                message=f"Drift detected: {event.metric_name}",
                details=event.description,
            )

    def set_sae_summary(self, summary: SAESummary) -> None:
        """Set SAE analysis summary."""
        self.sae_summary = summary

    def add_metrics_snapshot(self, snapshot: MetricsSnapshot) -> None:
        """Add a metrics snapshot."""
        self.metrics_history.append(snapshot)
        # Keep only last 100 snapshots
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

    def add_alert(
        self,
        level: str,
        message: str,
        details: str = "",
        source: str = "system",
    ) -> None:
        """Add an alert."""
        self.alerts.append({
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "details": details,
            "source": source,
            "acknowledged": False,
        })

    def render(self) -> str:
        """Render the complete dashboard as HTML."""
        panels = []

        if self.config.show_metrics_panel:
            panels.append(self._render_metrics_panel())

        if self.config.show_alerts_panel and self.alerts:
            panels.append(self._render_alerts_panel())

        if self.config.show_circuit_explorer and self.circuit_explorer:
            panels.append(self._render_circuit_panel())

        if self.config.show_feature_heatmap and self.feature_activations is not None:
            panels.append(self._render_heatmap_panel())

        if self.config.show_drift_monitor and self.drift_events:
            panels.append(self._render_drift_panel())

        if self.config.show_sae_summary and self.sae_summary:
            panels.append(self._render_sae_panel())

        refresh_script = ""
        if self.config.refresh_interval_seconds > 0:
            refresh_script = f"""
            <script>
                setTimeout(function() {{ location.reload(); }},
                    {self.config.refresh_interval_seconds * 1000});
            </script>
            """

        subtitle_html = f'<p class="subtitle">{self.config.subtitle}</p>' if self.config.subtitle else ""

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.config.title}</title>
    <style>
        :root {{
            --bg-color: {self.config.background_color};
            --card-bg: {self.config.card_color};
            --text-color: #eee;
            --accent: {self.config.accent_color};
            --border: #333;
            --success: #00bcd4;
            --warning: #ff9800;
            --critical: #e94560;
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        .dashboard-container {{
            max-width: {self.config.width}px;
            margin: 0 auto;
        }}
        .dashboard-header {{
            margin-bottom: 20px;
        }}
        h1 {{
            color: var(--accent);
            margin: 0 0 5px 0;
        }}
        .subtitle {{
            color: #888;
            margin: 0;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .panel {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            overflow: hidden;
        }}
        .panel-full {{
            grid-column: 1 / -1;
        }}
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .panel-title {{
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin: 0;
        }}
        .panel-badge {{
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(233, 69, 96, 0.2);
            color: var(--accent);
        }}
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 11px;
            color: #888;
            margin-top: 5px;
        }}
        .metric-delta {{
            font-size: 11px;
            margin-top: 3px;
        }}
        .delta-positive {{ color: #4caf50; }}
        .delta-negative {{ color: #e94560; }}
        .alert-list {{
            max-height: 200px;
            overflow-y: auto;
        }}
        .alert-item {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
            margin-bottom: 8px;
        }}
        .alert-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-top: 5px;
            flex-shrink: 0;
        }}
        .alert-dot.warning {{ background: #ff9800; }}
        .alert-dot.critical {{ background: #e94560; }}
        .alert-dot.emergency {{ background: #d32f2f; animation: pulse 1s infinite; }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .alert-content {{
            flex: 1;
        }}
        .alert-message {{
            font-size: 13px;
        }}
        .alert-details {{
            font-size: 11px;
            color: #888;
            margin-top: 3px;
        }}
        .alert-time {{
            font-size: 10px;
            color: #666;
        }}
        .sae-features {{
            display: grid;
            gap: 8px;
        }}
        .sae-feature {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .sae-feature-idx {{
            width: 40px;
            font-size: 12px;
            color: #888;
        }}
        .sae-feature-name {{
            flex: 1;
            font-size: 13px;
        }}
        .sae-feature-bar {{
            width: 100px;
            height: 6px;
            background: #333;
            border-radius: 3px;
            overflow: hidden;
        }}
        .sae-feature-fill {{
            height: 100%;
            background: var(--accent);
            border-radius: 3px;
        }}
        .sae-feature-value {{
            width: 50px;
            font-size: 12px;
            text-align: right;
        }}
        .sparkline {{
            display: inline-block;
            vertical-align: middle;
        }}
        svg {{
            display: block;
            max-width: 100%;
        }}
        .panel-content {{
            overflow-x: auto;
        }}
    </style>
</head>
<body>
<div class="dashboard-container">
    <div class="dashboard-header">
        <h1>{self.config.title}</h1>
        {subtitle_html}
    </div>
    <div class="dashboard-grid">
        {"".join(panels)}
    </div>
</div>
{refresh_script}
</body>
</html>
"""

    def _render_metrics_panel(self) -> str:
        """Render metrics overview panel."""
        if not self.metrics_history:
            metrics = MetricsSnapshot(
                timestamp=datetime.now(),
                total_queries=0,
                avg_confidence=0.0,
                avg_latency_ms=0.0,
                drift_alerts=len([e for e in self.drift_events
                                 if e.severity in [DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY]]),
                safety_blocks=0,
                cache_hit_rate=0.0,
                memory_mb=0.0,
            )
        else:
            metrics = self.metrics_history[-1]

        # Calculate deltas if we have history
        delta_html = {}
        if len(self.metrics_history) >= 2:
            prev = self.metrics_history[-2]
            for key in ["avg_confidence", "avg_latency_ms", "cache_hit_rate"]:
                curr_val = getattr(metrics, key)
                prev_val = getattr(prev, key)
                if prev_val != 0:
                    delta = ((curr_val - prev_val) / prev_val) * 100
                    delta_class = "delta-positive" if delta >= 0 else "delta-negative"
                    if key == "avg_latency_ms":
                        delta_class = "delta-negative" if delta >= 0 else "delta-positive"
                    sign = "+" if delta >= 0 else ""
                    delta_html[key] = f'<div class="metric-delta {delta_class}">{sign}{delta:.1f}%</div>'

        return f"""
        <div class="panel panel-full">
            <div class="panel-header">
                <h3 class="panel-title">System Metrics</h3>
                <span class="panel-badge">Live</span>
            </div>
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-value">{metrics.total_queries:,}</div>
                    <div class="metric-label">Total Queries</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.avg_confidence:.2f}</div>
                    <div class="metric-label">Avg Confidence</div>
                    {delta_html.get("avg_confidence", "")}
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.avg_latency_ms:.0f}ms</div>
                    <div class="metric-label">Avg Latency</div>
                    {delta_html.get("avg_latency_ms", "")}
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: {'#e94560' if metrics.drift_alerts > 0 else '#00bcd4'}">{metrics.drift_alerts}</div>
                    <div class="metric-label">Drift Alerts</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: {'#ff9800' if metrics.safety_blocks > 0 else '#00bcd4'}">{metrics.safety_blocks}</div>
                    <div class="metric-label">Safety Blocks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.cache_hit_rate:.0%}</div>
                    <div class="metric-label">Cache Hit Rate</div>
                    {delta_html.get("cache_hit_rate", "")}
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.memory_mb:.0f}</div>
                    <div class="metric-label">Memory (MB)</div>
                </div>
            </div>
        </div>
        """

    def _render_alerts_panel(self) -> str:
        """Render alerts panel."""
        # Show only unacknowledged alerts, most recent first
        active_alerts = [a for a in self.alerts if not a.get("acknowledged")][-10:]
        active_alerts.reverse()

        if not active_alerts:
            return ""

        alert_items = []
        for alert in active_alerts:
            alert_items.append(f"""
                <div class="alert-item">
                    <div class="alert-dot {alert['level']}"></div>
                    <div class="alert-content">
                        <div class="alert-message">{alert['message']}</div>
                        <div class="alert-details">{alert['details']}</div>
                    </div>
                    <div class="alert-time">{alert['timestamp'].strftime('%H:%M:%S')}</div>
                </div>
            """)

        return f"""
        <div class="panel panel-full">
            <div class="panel-header">
                <h3 class="panel-title">Active Alerts</h3>
                <span class="panel-badge" style="background: rgba(233, 69, 96, 0.3);">{len(active_alerts)}</span>
            </div>
            <div class="alert-list">
                {"".join(alert_items)}
            </div>
        </div>
        """

    def _render_circuit_panel(self) -> str:
        """Render circuit explorer panel."""
        if not self.circuit_explorer:
            return ""

        highlighted = getattr(self, '_highlighted_path', None)
        circuit_svg = self.circuit_explorer.render(
            title="",
            highlighted_path=highlighted,
            show_importance=True,
        )

        # Extract just the SVG content (skip full HTML)
        # Simple extraction - find SVG tag
        import re
        svg_match = re.search(r'<svg[^>]*>.*?</svg>', circuit_svg, re.DOTALL)
        svg_content = svg_match.group(0) if svg_match else '<p>Circuit visualization</p>'

        return f"""
        <div class="panel panel-full">
            <div class="panel-header">
                <h3 class="panel-title">Circuit Explorer</h3>
                <span class="panel-badge">{len(self.circuit_explorer.nodes)} nodes</span>
            </div>
            <div class="panel-content">
                {svg_content}
            </div>
        </div>
        """

    def _render_heatmap_panel(self) -> str:
        """Render feature heatmap panel."""
        if self.feature_activations is None:
            return ""

        config = HeatmapConfig(
            width=600,
            height=300,
            cell_width=20,
            cell_height=15,
            colormap="viridis",
        )
        heatmap = FeatureHeatmap(config)
        heatmap_html = heatmap.render(
            data=self.feature_activations,
            row_labels=self.feature_names,
            col_labels=self.token_names,
            title="",
        )

        # Extract just the SVG content
        import re
        svg_match = re.search(r'<svg[^>]*>.*?</svg>', heatmap_html, re.DOTALL)
        svg_content = svg_match.group(0) if svg_match else '<p>Heatmap visualization</p>'

        n_features, n_tokens = self.feature_activations.shape

        return f"""
        <div class="panel">
            <div class="panel-header">
                <h3 class="panel-title">Feature Activations</h3>
                <span class="panel-badge">{n_features}×{n_tokens}</span>
            </div>
            <div class="panel-content">
                {svg_content}
            </div>
        </div>
        """

    def _render_drift_panel(self) -> str:
        """Render drift monitor panel."""
        if not self.drift_events:
            return ""

        monitor = DriftMonitor()
        monitor.add_events(self.drift_events)

        # Render minimal version
        drift_html = monitor.render(title="", show_timeline=True, show_magnitude_chart=False)

        # Extract timeline SVG
        import re
        svg_match = re.search(r'<svg[^>]*>.*?</svg>', drift_html, re.DOTALL)
        svg_content = svg_match.group(0) if svg_match else '<p>Drift timeline</p>'

        critical_count = len([e for e in self.drift_events
                            if e.severity in [DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY]])

        return f"""
        <div class="panel">
            <div class="panel-header">
                <h3 class="panel-title">Drift Monitor</h3>
                <span class="panel-badge" style="background: {'rgba(233, 69, 96, 0.3)' if critical_count > 0 else 'rgba(0, 188, 212, 0.2)'};">
                    {len(self.drift_events)} events
                </span>
            </div>
            <div class="panel-content">
                {svg_content}
            </div>
        </div>
        """

    def _render_sae_panel(self) -> str:
        """Render SAE summary panel."""
        if not self.sae_summary:
            return ""

        summary = self.sae_summary

        # Top features visualization
        feature_rows = []
        for idx, name, activation in summary.top_features[:10]:
            fill_pct = min(100, activation * 100)
            feature_rows.append(f"""
                <div class="sae-feature">
                    <span class="sae-feature-idx">#{idx}</span>
                    <span class="sae-feature-name">{name}</span>
                    <div class="sae-feature-bar">
                        <div class="sae-feature-fill" style="width: {fill_pct}%"></div>
                    </div>
                    <span class="sae-feature-value">{activation:.3f}</span>
                </div>
            """)

        return f"""
        <div class="panel">
            <div class="panel-header">
                <h3 class="panel-title">SAE Analysis</h3>
                <span class="panel-badge">{summary.n_features} features</span>
            </div>
            <div class="metrics-row" style="margin-bottom: 15px;">
                <div class="metric-card">
                    <div class="metric-value">{summary.n_active}</div>
                    <div class="metric-label">Active</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.sparsity:.1%}</div>
                    <div class="metric-label">Sparsity</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.reconstruction_loss:.4f}</div>
                    <div class="metric-label">Recon Loss</div>
                </div>
            </div>
            <div class="sae-features">
                {"".join(feature_rows)}
            </div>
        </div>
        """

    def to_json(self) -> str:
        """Export dashboard data as JSON."""
        data = {
            "config": {
                "title": self.config.title,
                "subtitle": self.config.subtitle,
            },
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "total_queries": m.total_queries,
                    "avg_confidence": m.avg_confidence,
                    "avg_latency_ms": m.avg_latency_ms,
                    "drift_alerts": m.drift_alerts,
                    "safety_blocks": m.safety_blocks,
                    "cache_hit_rate": m.cache_hit_rate,
                    "memory_mb": m.memory_mb,
                }
                for m in self.metrics_history
            ],
            "drift_events": [e.to_dict() for e in self.drift_events],
            "alerts": [
                {
                    "timestamp": a["timestamp"].isoformat(),
                    "level": a["level"],
                    "message": a["message"],
                    "details": a["details"],
                }
                for a in self.alerts
            ],
        }

        if self.sae_summary:
            data["sae_summary"] = {
                "n_features": self.sae_summary.n_features,
                "n_active": self.sae_summary.n_active,
                "sparsity": self.sae_summary.sparsity,
                "reconstruction_loss": self.sae_summary.reconstruction_loss,
                "top_features": [
                    {"idx": idx, "name": name, "activation": act}
                    for idx, name, act in self.sae_summary.top_features
                ],
            }

        return json.dumps(data, indent=2)


def create_dashboard(
    title: str = "Dark Trace Dashboard",
    subtitle: str = "",
    **kwargs
) -> DarkTraceDashboard:
    """
    Create a new dashboard instance.

    Args:
        title: Dashboard title
        subtitle: Dashboard subtitle
        **kwargs: Additional config options

    Returns:
        DarkTraceDashboard instance
    """
    config = DashboardConfig(title=title, subtitle=subtitle, **kwargs)
    return DarkTraceDashboard(config)
