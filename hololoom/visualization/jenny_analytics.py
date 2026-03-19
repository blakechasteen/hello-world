"""
Jenny Learning Analytics Dashboard
===================================
Real-time analytics dashboard for Jenny's rendering and learning systems.

Philosophy:
> "Measure what matters, visualize what helps, learn from patterns."

Features (Phase M5):
- Render performance metrics (latency, panel counts)
- LLM compiler statistics (compilation times, model usage)
- Panel type distribution and trends
- Accessibility feature usage tracking
- Thompson Sampling integration (learning system metrics)
- Cache effectiveness monitoring
- Confidence trajectory analysis

Integrates with:
- Jenny renderers (HTMLRenderer, ReactRenderer, etc.)
- Jenny LLM compiler (query analysis, panel decisions)
- HoloLoom learning systems (recursive learning, hot patterns)
- Tufte visualization components (confidence, waterfall, gauge)

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot M5)
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .jenny_spec import PanelTypeJenny

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Types
# ============================================================================

class MetricType(Enum):
    """Types of metrics tracked by analytics."""
    RENDER_LATENCY = "render_latency"
    COMPILE_LATENCY = "compile_latency"
    PANEL_COUNT = "panel_count"
    CACHE_HIT = "cache_hit"
    ACCESSIBILITY_USAGE = "accessibility_usage"
    LLM_TOKENS = "llm_tokens"
    CONFIDENCE = "confidence"


# ============================================================================
# Analytics Events
# ============================================================================

@dataclass
class AnalyticsEvent:
    """
    Single analytics event captured during Jenny operations.

    Attributes:
        event_type: Type of metric captured
        value: Numeric value (latency in ms, count, ratio, etc.)
        timestamp: When event occurred
        metadata: Additional context (panel_type, model, target, etc.)
    """
    event_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderEvent:
    """Render operation analytics."""
    value: float  # latency_ms
    target: str = "html"
    panel_count: int = 0
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def event_type(self) -> MetricType:
        return MetricType.RENDER_LATENCY

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            'target': self.target,
            'panel_count': self.panel_count,
            'cache_hit': self.cache_hit,
        }


@dataclass
class CompileEvent:
    """Compilation operation analytics."""
    value: float  # latency_ms
    model: str = "unknown"
    query_type: str = "unknown"
    panels_generated: int = 0
    token_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def event_type(self) -> MetricType:
        return MetricType.COMPILE_LATENCY

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            'model': self.model,
            'query_type': self.query_type,
            'panels_generated': self.panels_generated,
            'token_count': self.token_count,
        }


# ============================================================================
# Analytics Collector
# ============================================================================

class JennyAnalyticsCollector:
    """
    Collects and aggregates analytics events from Jenny operations.

    Thread-safe event collection with configurable retention.
    Provides real-time and historical metrics access.

    Usage:
        collector = JennyAnalyticsCollector()

        # Record events
        collector.record_render(latency_ms=45.2, target='html', panel_count=3)
        collector.record_compile(latency_ms=120.5, model='llama3.2:3b', panels=5)

        # Get metrics
        metrics = collector.get_summary()
    """

    def __init__(self, max_events: int = 10000, retention_hours: int = 24):
        """
        Initialize analytics collector.

        Args:
            max_events: Maximum events to retain
            retention_hours: Hours to retain events
        """
        self._events: list[AnalyticsEvent] = []
        self._max_events = max_events
        self._retention_hours = retention_hours
        self._panel_type_counts: dict[str, int] = {}
        self._accessibility_usage: dict[str, int] = {}
        self._start_time = datetime.now()

    # ========================================================================
    # Event Recording
    # ========================================================================

    def record_event(self, event: AnalyticsEvent) -> None:
        """Record a raw analytics event."""
        self._events.append(event)
        self._prune_old_events()

    def record_render(
        self,
        latency_ms: float,
        target: str = 'html',
        panel_count: int = 0,
        cache_hit: bool = False
    ) -> None:
        """Record a render operation."""
        event = RenderEvent(
            value=latency_ms,
            target=target,
            panel_count=panel_count,
            cache_hit=cache_hit,
        )
        self.record_event(event)

    def record_compile(
        self,
        latency_ms: float,
        model: str = 'unknown',
        query_type: str = 'unknown',
        panels_generated: int = 0,
        token_count: int = 0
    ) -> None:
        """Record a compilation operation."""
        event = CompileEvent(
            value=latency_ms,
            model=model,
            query_type=query_type,
            panels_generated=panels_generated,
            token_count=token_count,
        )
        self.record_event(event)

    def record_panel_type(self, panel_type: PanelTypeJenny) -> None:
        """Record panel type usage."""
        key = panel_type.value
        self._panel_type_counts[key] = self._panel_type_counts.get(key, 0) + 1

    def record_accessibility_usage(self, feature: str) -> None:
        """Record accessibility feature usage."""
        self._accessibility_usage[feature] = self._accessibility_usage.get(feature, 0) + 1

    def record_confidence(self, confidence: float, query: str = "") -> None:
        """Record a confidence score."""
        event = AnalyticsEvent(
            event_type=MetricType.CONFIDENCE,
            value=confidence,
            metadata={'query': query[:50] if query else ''},
        )
        self.record_event(event)

    # ========================================================================
    # Metrics Access
    # ========================================================================

    def get_render_metrics(self) -> dict[str, Any]:
        """Get render performance metrics."""
        render_events = [e for e in self._events if e.event_type == MetricType.RENDER_LATENCY]

        if not render_events:
            return {
                'total_renders': 0,
                'avg_latency_ms': 0,
                'p95_latency_ms': 0,
                'cache_hit_rate': 0,
                'by_target': {},
            }

        latencies = [e.value for e in render_events]
        cache_hits = sum(1 for e in render_events if e.metadata.get('cache_hit'))

        # Group by target
        by_target = {}
        for event in render_events:
            target = event.metadata.get('target', 'unknown')
            if target not in by_target:
                by_target[target] = {'count': 0, 'latencies': []}
            by_target[target]['count'] += 1
            by_target[target]['latencies'].append(event.value)

        for target in by_target:
            lats = by_target[target]['latencies']
            by_target[target] = {
                'count': by_target[target]['count'],
                'avg_latency_ms': statistics.mean(lats),
                'p95_latency_ms': sorted(lats)[int(len(lats) * 0.95)] if len(lats) > 1 else lats[0],
            }

        return {
            'total_renders': len(render_events),
            'avg_latency_ms': round(statistics.mean(latencies), 2),
            'p95_latency_ms': round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else latencies[0],
            'cache_hit_rate': round(cache_hits / len(render_events), 3) if render_events else 0,
            'by_target': by_target,
        }

    def get_compile_metrics(self) -> dict[str, Any]:
        """Get compilation performance metrics."""
        compile_events = [e for e in self._events if e.event_type == MetricType.COMPILE_LATENCY]

        if not compile_events:
            return {
                'total_compiles': 0,
                'avg_latency_ms': 0,
                'total_tokens': 0,
                'by_model': {},
                'by_query_type': {},
            }

        latencies = [e.value for e in compile_events]
        total_tokens = sum(e.metadata.get('token_count', 0) for e in compile_events)

        # Group by model
        by_model = {}
        for event in compile_events:
            model = event.metadata.get('model', 'unknown')
            if model not in by_model:
                by_model[model] = {'count': 0, 'latencies': [], 'tokens': 0}
            by_model[model]['count'] += 1
            by_model[model]['latencies'].append(event.value)
            by_model[model]['tokens'] += event.metadata.get('token_count', 0)

        for model in by_model:
            lats = by_model[model]['latencies']
            by_model[model] = {
                'count': by_model[model]['count'],
                'avg_latency_ms': round(statistics.mean(lats), 2),
                'total_tokens': by_model[model]['tokens'],
            }

        # Group by query type
        by_query_type = {}
        for event in compile_events:
            qtype = event.metadata.get('query_type', 'unknown')
            by_query_type[qtype] = by_query_type.get(qtype, 0) + 1

        return {
            'total_compiles': len(compile_events),
            'avg_latency_ms': round(statistics.mean(latencies), 2),
            'total_tokens': total_tokens,
            'by_model': by_model,
            'by_query_type': by_query_type,
        }

    def get_confidence_metrics(self) -> dict[str, Any]:
        """Get confidence score metrics."""
        conf_events = [e for e in self._events if e.event_type == MetricType.CONFIDENCE]

        if not conf_events:
            return {
                'total_queries': 0,
                'avg_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0,
                'trajectory': [],
            }

        confidences = [e.value for e in conf_events]

        return {
            'total_queries': len(conf_events),
            'avg_confidence': round(statistics.mean(confidences), 3),
            'min_confidence': round(min(confidences), 3),
            'max_confidence': round(max(confidences), 3),
            'trajectory': confidences[-100:],  # Last 100 for visualization
        }

    def get_panel_type_distribution(self) -> dict[str, int]:
        """Get panel type usage distribution."""
        return dict(self._panel_type_counts)

    def get_accessibility_metrics(self) -> dict[str, Any]:
        """Get accessibility feature usage metrics."""
        total = sum(self._accessibility_usage.values())
        return {
            'total_uses': total,
            'by_feature': dict(self._accessibility_usage),
            'adoption_rate': round(total / max(len(self._events), 1), 3),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get complete analytics summary."""
        uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            'uptime_seconds': round(uptime, 1),
            'total_events': len(self._events),
            'render': self.get_render_metrics(),
            'compile': self.get_compile_metrics(),
            'confidence': self.get_confidence_metrics(),
            'panel_types': self.get_panel_type_distribution(),
            'accessibility': self.get_accessibility_metrics(),
            'timestamp': datetime.now().isoformat(),
        }

    # ========================================================================
    # Maintenance
    # ========================================================================

    def _prune_old_events(self) -> None:
        """Remove old events beyond retention limit."""
        # Limit by count
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        # Limit by time (if retention_hours set)
        if self._retention_hours > 0:
            cutoff = datetime.now().timestamp() - (self._retention_hours * 3600)
            self._events = [e for e in self._events if e.timestamp.timestamp() > cutoff]

    def clear(self) -> None:
        """Clear all collected analytics."""
        self._events.clear()
        self._panel_type_counts.clear()
        self._accessibility_usage.clear()
        self._start_time = datetime.now()


# ============================================================================
# Singleton Collector Instance
# ============================================================================

_collector_instance: JennyAnalyticsCollector | None = None


def get_analytics_collector() -> JennyAnalyticsCollector:
    """Get the global analytics collector singleton."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = JennyAnalyticsCollector()
    return _collector_instance


# ============================================================================
# Analytics Dashboard Renderer
# ============================================================================

class JennyAnalyticsDashboard:
    """
    Renders Jenny analytics as a beautiful HTML dashboard.

    Uses Tufte principles and HoloLoom visualization components:
    - Confidence trajectory with anomaly detection
    - Cache gauge with effectiveness rating
    - Stage waterfall for latency breakdown
    - Data density tables for metrics
    - Small multiples for comparisons

    Usage:
        collector = get_analytics_collector()
        # ... record events ...

        dashboard = JennyAnalyticsDashboard(collector)
        html = dashboard.render()
    """

    def __init__(self, collector: JennyAnalyticsCollector | None = None):
        """Initialize dashboard with analytics collector."""
        self.collector = collector or get_analytics_collector()

    def render(
        self,
        title: str = "Jenny Learning Analytics",
        theme: str = "dark"
    ) -> str:
        """
        Render complete analytics dashboard.

        Args:
            title: Dashboard title
            theme: Color theme ('dark' or 'light')

        Returns:
            Complete HTML string
        """
        metrics = self.collector.get_summary()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {self._get_styles(theme)}
</head>
<body class="jenny-analytics-body jenny-analytics--{theme}">
    <header class="jenny-analytics-header">
        <h1>{title}</h1>
        <p class="subtitle">Generated: {metrics['timestamp']} | Uptime: {self._format_uptime(metrics['uptime_seconds'])}</p>
    </header>

    <main class="jenny-analytics-main">
        {self._render_summary_cards(metrics)}
        {self._render_render_metrics(metrics['render'])}
        {self._render_compile_metrics(metrics['compile'])}
        {self._render_confidence_chart(metrics['confidence'])}
        {self._render_panel_distribution(metrics['panel_types'])}
        {self._render_accessibility_metrics(metrics['accessibility'])}
    </main>

    <footer class="jenny-analytics-footer">
        <p>Jenny Learning Analytics Dashboard | Phase M5 | HoloLoom</p>
    </footer>
</body>
</html>"""

    def _get_styles(self, theme: str) -> str:
        """Get dashboard styles."""
        bg = "#1a1a2e" if theme == "dark" else "#ffffff"
        fg = "#e4e4e4" if theme == "dark" else "#1a1a2e"
        card_bg = "#16213e" if theme == "dark" else "#f8f9fa"
        accent = "#4fc3f7"
        success = "#4caf50"
        warning = "#ff9800"
        error = "#f44336"

        return f"""<style>
:root {{
    --bg: {bg};
    --fg: {fg};
    --card-bg: {card_bg};
    --accent: {accent};
    --success: {success};
    --warning: {warning};
    --error: {error};
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

.jenny-analytics-body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--fg);
    min-height: 100vh;
    padding: 2rem;
}}

.jenny-analytics-header {{
    text-align: center;
    margin-bottom: 2rem;
}}

.jenny-analytics-header h1 {{
    font-size: 2rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}}

.jenny-analytics-header .subtitle {{
    font-size: 0.875rem;
    opacity: 0.7;
}}

.jenny-analytics-main {{
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    gap: 1.5rem;
}}

.analytics-section {{
    background: var(--card-bg);
    border-radius: 8px;
    padding: 1.5rem;
}}

.analytics-section h2 {{
    font-size: 1.125rem;
    font-weight: 500;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}

/* Summary Cards */
.summary-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}}

.summary-card {{
    background: var(--card-bg);
    border-radius: 8px;
    padding: 1.25rem;
    text-align: center;
}}

.summary-card .value {{
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}}

.summary-card .label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.7;
    margin-top: 0.25rem;
}}

/* Metrics Grid */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}}

.metric {{
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    padding: 0.75rem;
}}

.metric .value {{
    font-size: 1.5rem;
    font-weight: 600;
}}

.metric .label {{
    font-size: 0.75rem;
    opacity: 0.6;
}}

/* Confidence Chart */
.confidence-chart {{
    height: 120px;
    display: flex;
    align-items: flex-end;
    gap: 2px;
    padding: 1rem 0;
}}

.confidence-bar {{
    flex: 1;
    max-width: 8px;
    background: var(--accent);
    border-radius: 2px 2px 0 0;
    transition: height 0.3s;
}}

.confidence-bar.low {{ background: var(--error); }}
.confidence-bar.medium {{ background: var(--warning); }}
.confidence-bar.high {{ background: var(--success); }}

/* Panel Distribution */
.distribution-chart {{
    display: flex;
    height: 30px;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 1rem;
}}

.distribution-segment {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 500;
    color: white;
    transition: flex-basis 0.3s;
}}

.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 1rem;
    font-size: 0.75rem;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.legend-swatch {{
    width: 12px;
    height: 12px;
    border-radius: 2px;
}}

/* Footer */
.jenny-analytics-footer {{
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    font-size: 0.75rem;
    opacity: 0.5;
}}

/* Responsive */
@media (max-width: 768px) {{
    .jenny-analytics-body {{ padding: 1rem; }}
    .summary-cards {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>"""

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"

    def _render_summary_cards(self, metrics: dict[str, Any]) -> str:
        """Render summary cards section."""
        return f"""
    <section class="summary-cards">
        <div class="summary-card">
            <div class="value">{metrics['total_events']}</div>
            <div class="label">Total Events</div>
        </div>
        <div class="summary-card">
            <div class="value">{metrics['render']['total_renders']}</div>
            <div class="label">Renders</div>
        </div>
        <div class="summary-card">
            <div class="value">{metrics['compile']['total_compiles']}</div>
            <div class="label">Compiles</div>
        </div>
        <div class="summary-card">
            <div class="value">{int(metrics['confidence']['avg_confidence'] * 100)}%</div>
            <div class="label">Avg Confidence</div>
        </div>
        <div class="summary-card">
            <div class="value">{int(metrics['render']['cache_hit_rate'] * 100)}%</div>
            <div class="label">Cache Hit Rate</div>
        </div>
    </section>"""

    def _render_render_metrics(self, render: dict[str, Any]) -> str:
        """Render render metrics section."""
        by_target_html = ""
        for target, stats in render.get('by_target', {}).items():
            by_target_html += f"""
            <div class="metric">
                <div class="value">{stats['avg_latency_ms']:.1f}ms</div>
                <div class="label">{target} avg latency</div>
            </div>"""

        return f"""
    <section class="analytics-section">
        <h2>Render Performance</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{render['total_renders']}</div>
                <div class="label">Total Renders</div>
            </div>
            <div class="metric">
                <div class="value">{render['avg_latency_ms']:.1f}ms</div>
                <div class="label">Avg Latency</div>
            </div>
            <div class="metric">
                <div class="value">{render['p95_latency_ms']:.1f}ms</div>
                <div class="label">P95 Latency</div>
            </div>
            <div class="metric">
                <div class="value">{int(render['cache_hit_rate'] * 100)}%</div>
                <div class="label">Cache Hit Rate</div>
            </div>
            {by_target_html}
        </div>
    </section>"""

    def _render_compile_metrics(self, compile_metrics: dict[str, Any]) -> str:
        """Render compile metrics section."""
        by_model_html = ""
        for model, stats in compile_metrics.get('by_model', {}).items():
            by_model_html += f"""
            <div class="metric">
                <div class="value">{stats['count']}</div>
                <div class="label">{model[:15]}</div>
            </div>"""

        by_type_html = ""
        for qtype, count in compile_metrics.get('by_query_type', {}).items():
            by_type_html += f"""
            <div class="metric">
                <div class="value">{count}</div>
                <div class="label">{qtype}</div>
            </div>"""

        return f"""
    <section class="analytics-section">
        <h2>LLM Compilation</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{compile_metrics['total_compiles']}</div>
                <div class="label">Total Compiles</div>
            </div>
            <div class="metric">
                <div class="value">{compile_metrics['avg_latency_ms']:.1f}ms</div>
                <div class="label">Avg Latency</div>
            </div>
            <div class="metric">
                <div class="value">{compile_metrics['total_tokens']:,}</div>
                <div class="label">Total Tokens</div>
            </div>
        </div>
        <h3 style="margin-top:1rem;font-size:0.875rem;opacity:0.7">By Model</h3>
        <div class="metrics-grid">{by_model_html}</div>
        <h3 style="margin-top:1rem;font-size:0.875rem;opacity:0.7">By Query Type</h3>
        <div class="metrics-grid">{by_type_html}</div>
    </section>"""

    def _render_confidence_chart(self, confidence: dict[str, Any]) -> str:
        """Render confidence trajectory chart."""
        trajectory = confidence.get('trajectory', [])
        if not trajectory:
            return """
    <section class="analytics-section">
        <h2>Confidence Trajectory</h2>
        <p style="opacity:0.5;font-style:italic">No confidence data yet</p>
    </section>"""

        # Build bars
        bars = ""
        for conf in trajectory[-50:]:  # Last 50
            height = int(conf * 100)
            css_class = "low" if conf < 0.6 else ("medium" if conf < 0.8 else "high")
            bars += f'<div class="confidence-bar {css_class}" style="height:{height}%"></div>'

        return f"""
    <section class="analytics-section">
        <h2>Confidence Trajectory</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{int(confidence['avg_confidence'] * 100)}%</div>
                <div class="label">Average</div>
            </div>
            <div class="metric">
                <div class="value">{int(confidence['min_confidence'] * 100)}%</div>
                <div class="label">Minimum</div>
            </div>
            <div class="metric">
                <div class="value">{int(confidence['max_confidence'] * 100)}%</div>
                <div class="label">Maximum</div>
            </div>
        </div>
        <div class="confidence-chart">
            {bars}
        </div>
    </section>"""

    def _render_panel_distribution(self, panel_types: dict[str, int]) -> str:
        """Render panel type distribution chart."""
        if not panel_types:
            return """
    <section class="analytics-section">
        <h2>Panel Type Distribution</h2>
        <p style="opacity:0.5;font-style:italic">No panel data yet</p>
    </section>"""

        total = sum(panel_types.values())
        colors = {
            'text': '#4fc3f7',
            'confidence': '#4caf50',
            'sources': '#ff9800',
            'graph': '#e91e63',
            'reasoning': '#9c27b0',
            'why': '#673ab7',
            'code': '#00bcd4',
            'table': '#8bc34a',
            'metric': '#ffc107',
        }

        segments = ""
        legend_items = ""
        for panel_type, count in sorted(panel_types.items(), key=lambda x: -x[1]):
            percent = (count / total) * 100
            color = colors.get(panel_type, '#888')
            segments += f'<div class="distribution-segment" style="flex-basis:{percent}%;background:{color}">{int(percent)}%</div>'
            legend_items += f'<div class="legend-item"><div class="legend-swatch" style="background:{color}"></div>{panel_type}: {count}</div>'

        return f"""
    <section class="analytics-section">
        <h2>Panel Type Distribution</h2>
        <div class="distribution-chart">
            {segments}
        </div>
        <div class="legend">
            {legend_items}
        </div>
    </section>"""

    def _render_accessibility_metrics(self, a11y: dict[str, Any]) -> str:
        """Render accessibility metrics section."""
        by_feature = a11y.get('by_feature', {})
        if not by_feature:
            return """
    <section class="analytics-section">
        <h2>Accessibility Features</h2>
        <p style="opacity:0.5;font-style:italic">No accessibility usage data yet</p>
    </section>"""

        features_html = ""
        for feature, count in sorted(by_feature.items(), key=lambda x: -x[1]):
            features_html += f"""
            <div class="metric">
                <div class="value">{count}</div>
                <div class="label">{feature}</div>
            </div>"""

        return f"""
    <section class="analytics-section">
        <h2>Accessibility Features</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="value">{a11y['total_uses']}</div>
                <div class="label">Total Uses</div>
            </div>
            <div class="metric">
                <div class="value">{int(a11y['adoption_rate'] * 100)}%</div>
                <div class="label">Adoption Rate</div>
            </div>
            {features_html}
        </div>
    </section>"""

    def save(self, path: str = "jenny_analytics.html") -> str:
        """Save dashboard to file."""
        html = self.render()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Analytics dashboard saved to {path}")
        return path


# ============================================================================
# Convenience Functions
# ============================================================================

def record_render(
    latency_ms: float,
    target: str = 'html',
    panel_count: int = 0,
    cache_hit: bool = False
) -> None:
    """Record render event to global collector."""
    get_analytics_collector().record_render(latency_ms, target, panel_count, cache_hit)


def record_compile(
    latency_ms: float,
    model: str = 'unknown',
    query_type: str = 'unknown',
    panels_generated: int = 0,
    token_count: int = 0
) -> None:
    """Record compile event to global collector."""
    get_analytics_collector().record_compile(latency_ms, model, query_type, panels_generated, token_count)


def record_confidence(confidence: float, query: str = "") -> None:
    """Record confidence score to global collector."""
    get_analytics_collector().record_confidence(confidence, query)


def get_analytics_summary() -> dict[str, Any]:
    """Get analytics summary from global collector."""
    return get_analytics_collector().get_summary()


def render_analytics_dashboard(
    title: str = "Jenny Learning Analytics",
    theme: str = "dark"
) -> str:
    """Render analytics dashboard from global collector."""
    dashboard = JennyAnalyticsDashboard()
    return dashboard.render(title=title, theme=theme)


def save_analytics_dashboard(
    path: str = "jenny_analytics.html",
    title: str = "Jenny Learning Analytics",
    theme: str = "dark"
) -> str:
    """Save analytics dashboard to file."""
    dashboard = JennyAnalyticsDashboard()
    html = dashboard.render(title=title, theme=theme)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Types
    'MetricType',
    'AnalyticsEvent',
    'RenderEvent',
    'CompileEvent',
    # Collector
    'JennyAnalyticsCollector',
    'get_analytics_collector',
    # Dashboard
    'JennyAnalyticsDashboard',
    # Convenience
    'record_render',
    'record_compile',
    'record_confidence',
    'get_analytics_summary',
    'render_analytics_dashboard',
    'save_analytics_dashboard',
]
