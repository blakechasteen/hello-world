"""
Dark Trace Evaluation: Reporting

Generates publication-ready evaluation reports with:
- Comprehensive metrics summaries
- Regression detection
- Interactive dashboards
- Multiple output formats (HTML, Markdown, JSON, LaTeX)

Key Components:
- EvaluationReporter: Main report generator
- EvaluationReport: Report data structure
- MetricsDashboard: Interactive visualization
- RegressionDetector: Quality regression detection

Research Basis:
- ML experiment tracking best practices
- Publication-quality figure generation
- Statistical significance testing
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ReportFormat(Enum):
    """Output formats for reports."""

    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    LATEX = "latex"
    TEXT = "text"


class ReportLevel(Enum):
    """Detail level for reports."""

    SUMMARY = "summary"
    STANDARD = "standard"
    DETAILED = "detailed"
    FULL = "full"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        format: Output format
        level: Detail level
        include_figures: Whether to include visualizations
        include_recommendations: Whether to include recommendations
        include_history: Whether to include historical comparison
        regression_threshold: Threshold for regression detection
        output_path: Output file path
        title: Report title
        author: Report author
    """

    format: ReportFormat = ReportFormat.HTML
    level: ReportLevel = ReportLevel.STANDARD
    include_figures: bool = True
    include_recommendations: bool = True
    include_history: bool = True
    regression_threshold: float = 0.05
    output_path: Optional[str] = None
    title: str = "SAE Evaluation Report"
    author: str = "Dark Trace"

    @classmethod
    def publication(cls) -> "ReportConfig":
        """Configuration for publication-ready report."""
        return cls(
            format=ReportFormat.LATEX,
            level=ReportLevel.FULL,
            include_figures=True,
            include_recommendations=True,
            include_history=True,
            title="SAE Interpretability Evaluation",
        )

    @classmethod
    def dashboard(cls) -> "ReportConfig":
        """Configuration for interactive dashboard."""
        return cls(
            format=ReportFormat.HTML,
            level=ReportLevel.DETAILED,
            include_figures=True,
            include_recommendations=True,
        )

    @classmethod
    def quick(cls) -> "ReportConfig":
        """Configuration for quick summary."""
        return cls(
            format=ReportFormat.TEXT,
            level=ReportLevel.SUMMARY,
            include_figures=False,
            include_recommendations=False,
        )


@dataclass
class RegressionAlert:
    """Alert for metric regression.

    Attributes:
        metric_name: Name of regressed metric
        current_value: Current metric value
        baseline_value: Previous/baseline value
        change_percent: Percentage change
        severity: Alert severity
        recommendation: Suggested action
    """

    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    severity: AlertLevel
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "change_percent": self.change_percent,
            "severity": self.severity.value,
            "recommendation": self.recommendation,
        }


@dataclass
class MetricTrend:
    """Trend information for a metric.

    Attributes:
        metric_name: Name of metric
        values: Historical values
        timestamps: Value timestamps
        direction: Trend direction (up, down, stable)
        slope: Linear trend slope
        volatility: Standard deviation of changes
    """

    metric_name: str
    values: List[float]
    timestamps: List[float]
    direction: str  # "up", "down", "stable"
    slope: float
    volatility: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "values": self.values,
            "timestamps": self.timestamps,
            "direction": self.direction,
            "slope": self.slope,
            "volatility": self.volatility,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report.

    Attributes:
        timestamp: Report generation time
        config: Report configuration
        metrics: Main evaluation metrics
        benchmark_results: Benchmark results
        baseline_comparison: Comparison with baselines
        regressions: Detected regressions
        trends: Metric trends
        recommendations: Improvement recommendations
        metadata: Additional metadata
    """

    timestamp: datetime
    config: ReportConfig
    metrics: Dict[str, float]
    benchmark_results: Optional[Dict[str, Any]] = None
    baseline_comparison: Optional[Dict[str, Any]] = None
    regressions: List[RegressionAlert] = field(default_factory=list)
    trends: List[MetricTrend] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            f"  {self.config.title}",
            "=" * 60,
            f"Generated: {self.timestamp.isoformat()}",
            f"Author: {self.config.author}",
            "",
            "-" * 60,
            "KEY METRICS",
            "-" * 60,
        ]

        # Key metrics
        key_metrics = [
            "overall_score", "reconstruction_mse", "sparsity_l0",
            "dead_feature_ratio", "mean_monosemanticity"
        ]
        for metric in key_metrics:
            if metric in self.metrics:
                lines.append(f"  {metric}: {self.metrics[metric]:.4f}")

        # Regressions
        if self.regressions:
            lines.extend([
                "",
                "-" * 60,
                f"REGRESSIONS DETECTED ({len(self.regressions)})",
                "-" * 60,
            ])
            for alert in self.regressions:
                icon = "⚠" if alert.severity == AlertLevel.WARNING else "❌"
                lines.append(
                    f"  {icon} {alert.metric_name}: "
                    f"{alert.change_percent:+.1f}% ({alert.current_value:.4f})"
                )

        # Recommendations
        if self.recommendations:
            lines.extend([
                "",
                "-" * 60,
                "RECOMMENDATIONS",
                "-" * 60,
            ])
            for i, rec in enumerate(self.recommendations[:5], 1):
                lines.append(f"  {i}. {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "config": {
                "format": self.config.format.value,
                "level": self.config.level.value,
                "title": self.config.title,
                "author": self.config.author,
            },
            "metrics": self.metrics,
            "benchmark_results": self.benchmark_results,
            "baseline_comparison": self.baseline_comparison,
            "regressions": [r.to_dict() for r in self.regressions],
            "trends": [t.to_dict() for t in self.trends],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save report to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.format == ReportFormat.JSON:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif self.config.format == ReportFormat.MARKDOWN:
            with open(path, "w") as f:
                f.write(self._to_markdown())
        elif self.config.format == ReportFormat.HTML:
            with open(path, "w") as f:
                f.write(self._to_html())
        elif self.config.format == ReportFormat.LATEX:
            with open(path, "w") as f:
                f.write(self._to_latex())
        else:
            with open(path, "w") as f:
                f.write(self.summary())

    def _to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            f"# {self.config.title}",
            "",
            f"**Generated:** {self.timestamp.isoformat()}",
            f"**Author:** {self.config.author}",
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        for metric, value in sorted(self.metrics.items()):
            lines.append(f"| {metric} | {value:.4f} |")

        if self.regressions:
            lines.extend([
                "",
                "## Regressions",
                "",
            ])
            for alert in self.regressions:
                icon = "⚠️" if alert.severity == AlertLevel.WARNING else "🔴"
                lines.append(
                    f"- {icon} **{alert.metric_name}**: "
                    f"{alert.change_percent:+.1f}% (now {alert.current_value:.4f})"
                )

        if self.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def _to_html(self) -> str:
        """Convert to HTML format."""
        metrics_rows = "\n".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            for k, v in sorted(self.metrics.items())
        )

        regression_items = ""
        if self.regressions:
            items = "\n".join(
                f'<li class="{a.severity.value}">{a.metric_name}: {a.change_percent:+.1f}%</li>'
                for a in self.regressions
            )
            regression_items = f"<ul>{items}</ul>"

        rec_items = ""
        if self.recommendations:
            items = "\n".join(f"<li>{r}</li>" for r in self.recommendations)
            rec_items = f"<ol>{items}</ol>"

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.config.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .warning {{ color: #f0ad4e; }}
        .error {{ color: #d9534f; }}
        .critical {{ color: #c9302c; font-weight: bold; }}
        .info {{ color: #5bc0de; }}
        .meta {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{self.config.title}</h1>
    <p class="meta">Generated: {self.timestamp.isoformat()} | Author: {self.config.author}</p>

    <h2>Key Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        {metrics_rows}
    </table>

    {"<h2>Regressions</h2>" + regression_items if self.regressions else ""}

    {"<h2>Recommendations</h2>" + rec_items if self.recommendations else ""}
</body>
</html>"""

    def _to_latex(self) -> str:
        """Convert to LaTeX format."""
        metrics_rows = "\n".join(
            f"        {k.replace('_', r'\_')} & {v:.4f} \\\\"
            for k, v in sorted(self.metrics.items())
        )

        return f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}

\\title{{{self.config.title}}}
\\author{{{self.config.author}}}
\\date{{{self.timestamp.strftime('%Y-%m-%d')}}}

\\begin{{document}}
\\maketitle

\\section{{Key Metrics}}
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
Metric & Value \\\\
\\midrule
{metrics_rows}
\\bottomrule
\\end{{tabular}}
\\caption{{SAE Evaluation Metrics}}
\\end{{table}}

\\section{{Analysis}}
% Add analysis here

\\end{{document}}"""


class RegressionDetector:
    """Detects metric regressions compared to baseline.

    Compares current metrics against historical baselines and
    generates alerts for significant regressions.

    Example:
        detector = RegressionDetector()
        detector.set_baseline(baseline_metrics)
        alerts = detector.detect(current_metrics)
    """

    # Metrics where higher is better
    HIGHER_BETTER = {
        "overall_score", "cosine_similarity", "explained_variance",
        "mean_monosemanticity", "recovery_rate", "feature_stability",
        "steering_accuracy", "controllability",
    }

    # Metrics where lower is better
    LOWER_BETTER = {
        "mse", "rmse", "reconstruction_mse", "dead_feature_ratio",
        "false_discovery_rate", "sparsity_error", "side_effect",
    }

    # Default thresholds for different severity levels
    THRESHOLDS = {
        AlertLevel.INFO: 0.02,
        AlertLevel.WARNING: 0.05,
        AlertLevel.ERROR: 0.10,
        AlertLevel.CRITICAL: 0.20,
    }

    def __init__(
        self,
        threshold: float = 0.05,
        history_size: int = 10,
    ):
        """Initialize regression detector.

        Args:
            threshold: Default regression threshold
            history_size: Number of historical values to keep
        """
        self.threshold = threshold
        self.history_size = history_size
        self._baseline: Dict[str, float] = {}
        self._history: Dict[str, List[Tuple[float, float]]] = {}  # metric -> [(timestamp, value)]

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """Set baseline metrics for comparison.

        Args:
            metrics: Baseline metric values
        """
        self._baseline = metrics.copy()

    def add_to_history(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[float] = None,
    ) -> None:
        """Add metrics to history.

        Args:
            metrics: Current metric values
            timestamp: Optional timestamp (defaults to now)
        """
        timestamp = timestamp or time.time()

        for name, value in metrics.items():
            if name not in self._history:
                self._history[name] = []
            self._history[name].append((timestamp, value))

            # Keep only recent history
            if len(self._history[name]) > self.history_size:
                self._history[name] = self._history[name][-self.history_size:]

    def detect(
        self,
        current_metrics: Dict[str, float],
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[RegressionAlert]:
        """Detect regressions in current metrics.

        Args:
            current_metrics: Current metric values
            custom_thresholds: Per-metric thresholds

        Returns:
            List of regression alerts
        """
        custom_thresholds = custom_thresholds or {}
        alerts = []

        for name, current in current_metrics.items():
            if name not in self._baseline:
                continue

            baseline = self._baseline[name]
            threshold = custom_thresholds.get(name, self.threshold)

            # Calculate change
            if baseline == 0:
                change_percent = 100.0 if current != 0 else 0.0
            else:
                change_percent = (current - baseline) / abs(baseline) * 100

            # Determine if regression occurred
            is_regression = False
            if name in self.HIGHER_BETTER:
                is_regression = change_percent < -threshold * 100
            elif name in self.LOWER_BETTER:
                is_regression = change_percent > threshold * 100
            else:
                # Unknown metric - assume lower is better for safety
                is_regression = change_percent > threshold * 100

            if is_regression:
                # Determine severity
                abs_change = abs(change_percent) / 100
                severity = AlertLevel.INFO
                for level, thresh in sorted(
                    self.THRESHOLDS.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    if abs_change >= thresh:
                        severity = level
                        break

                # Generate recommendation
                recommendation = self._generate_recommendation(
                    name, current, baseline, change_percent
                )

                alerts.append(RegressionAlert(
                    metric_name=name,
                    current_value=current,
                    baseline_value=baseline,
                    change_percent=change_percent,
                    severity=severity,
                    recommendation=recommendation,
                ))

        # Sort by severity
        severity_order = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.ERROR: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 3,
        }
        alerts.sort(key=lambda a: severity_order[a.severity])

        return alerts

    def _generate_recommendation(
        self,
        metric_name: str,
        current: float,
        baseline: float,
        change_percent: float,
    ) -> str:
        """Generate recommendation for regression.

        Args:
            metric_name: Name of regressed metric
            current: Current value
            baseline: Baseline value
            change_percent: Change percentage

        Returns:
            Recommendation string
        """
        recommendations = {
            "reconstruction_mse": "Increase training iterations or adjust L1 coefficient",
            "dead_feature_ratio": "Enable ghost gradients or adjust resampling threshold",
            "mean_monosemanticity": "Check for feature collapse, try higher sparsity",
            "sparsity_l0": "Adjust L1 coefficient or use L1 annealing",
            "recovery_rate": "Increase SAE hidden size or improve training data",
            "steering_accuracy": "Verify decoder normalization, check feature alignment",
        }

        if metric_name in recommendations:
            return recommendations[metric_name]

        if "mse" in metric_name or "error" in metric_name:
            return "Review training hyperparameters and data quality"

        return f"Investigate {metric_name} degradation"

    def get_trends(self) -> List[MetricTrend]:
        """Analyze trends in historical metrics.

        Returns:
            List of metric trends
        """
        trends = []

        for name, history in self._history.items():
            if len(history) < 2:
                continue

            timestamps = [h[0] for h in history]
            values = [h[1] for h in history]

            # Linear regression for trend
            t_mean = np.mean(timestamps)
            v_mean = np.mean(values)
            t_diff = np.array(timestamps) - t_mean
            v_diff = np.array(values) - v_mean

            slope = np.sum(t_diff * v_diff) / (np.sum(t_diff ** 2) + 1e-10)

            # Determine direction
            if abs(slope) < 1e-6:
                direction = "stable"
            elif slope > 0:
                direction = "up"
            else:
                direction = "down"

            # Volatility
            volatility = np.std(values)

            trends.append(MetricTrend(
                metric_name=name,
                values=values,
                timestamps=timestamps,
                direction=direction,
                slope=slope,
                volatility=volatility,
            ))

        return trends


class MetricsDashboard:
    """Interactive dashboard for metrics visualization.

    Generates HTML dashboard with:
    - Metric cards with trends
    - Alert panels
    - Comparison charts
    - Historical plots

    Example:
        dashboard = MetricsDashboard()
        html = dashboard.render(metrics, alerts, trends)
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize dashboard.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig.dashboard()

    def render(
        self,
        metrics: Dict[str, float],
        alerts: Optional[List[RegressionAlert]] = None,
        trends: Optional[List[MetricTrend]] = None,
        title: Optional[str] = None,
    ) -> str:
        """Render dashboard HTML.

        Args:
            metrics: Current metrics
            alerts: Regression alerts
            trends: Metric trends
            title: Dashboard title

        Returns:
            HTML string
        """
        title = title or self.config.title

        # Metric cards
        cards = self._render_metric_cards(metrics, trends or [])

        # Alert panel
        alert_panel = self._render_alerts(alerts or [])

        # Trend charts (simplified - would use Chart.js in production)
        trend_section = self._render_trends(trends or [])

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        .dashboard {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ margin-bottom: 20px; }}
        .header h1 {{ margin: 0; color: #1a1a2e; }}
        .header .meta {{ color: #666; font-size: 0.9em; }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card-title {{ color: #666; font-size: 0.85em; margin-bottom: 8px; }}
        .card-value {{ font-size: 1.8em; font-weight: 600; color: #1a1a2e; }}
        .card-trend {{ font-size: 0.85em; margin-top: 4px; }}
        .trend-up {{ color: #22c55e; }}
        .trend-down {{ color: #ef4444; }}
        .trend-stable {{ color: #888; }}
        .alerts {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .alert {{
            padding: 12px;
            margin: 8px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }}
        .alert-warning {{ background: #fef3cd; border-color: #f0ad4e; }}
        .alert-error {{ background: #f8d7da; border-color: #dc3545; }}
        .alert-critical {{ background: #f5c6cb; border-color: #721c24; }}
        .alert-info {{ background: #d1ecf1; border-color: #17a2b8; }}
        .trends {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .trend-item {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
        .sparkline {{
            display: inline-block;
            height: 20px;
            width: 60px;
            background: #f0f0f0;
            border-radius: 2px;
            vertical-align: middle;
            margin-left: 8px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{title}</h1>
            <p class="meta">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="cards">
            {cards}
        </div>

        {alert_panel}

        {trend_section}
    </div>
</body>
</html>"""

    def _render_metric_cards(
        self,
        metrics: Dict[str, float],
        trends: List[MetricTrend],
    ) -> str:
        """Render metric cards."""
        trend_map = {t.metric_name: t for t in trends}

        cards = []
        for name, value in sorted(metrics.items()):
            trend_info = trend_map.get(name)
            trend_class = ""
            trend_icon = ""

            if trend_info:
                if trend_info.direction == "up":
                    trend_class = "trend-up"
                    trend_icon = "↑"
                elif trend_info.direction == "down":
                    trend_class = "trend-down"
                    trend_icon = "↓"
                else:
                    trend_class = "trend-stable"
                    trend_icon = "→"

            trend_html = f'<div class="card-trend {trend_class}">{trend_icon}</div>' if trend_info else ""

            cards.append(f"""
            <div class="card">
                <div class="card-title">{name}</div>
                <div class="card-value">{value:.4f}</div>
                {trend_html}
            </div>
            """)

        return "\n".join(cards)

    def _render_alerts(self, alerts: List[RegressionAlert]) -> str:
        """Render alert panel."""
        if not alerts:
            return ""

        items = []
        for alert in alerts:
            items.append(f"""
            <div class="alert alert-{alert.severity.value}">
                <strong>{alert.metric_name}</strong>: {alert.change_percent:+.1f}%
                (was {alert.baseline_value:.4f}, now {alert.current_value:.4f})
                <br><small>{alert.recommendation}</small>
            </div>
            """)

        return f"""
        <div class="alerts">
            <h2>Regression Alerts ({len(alerts)})</h2>
            {"".join(items)}
        </div>
        """

    def _render_trends(self, trends: List[MetricTrend]) -> str:
        """Render trend section."""
        if not trends:
            return ""

        items = []
        for trend in trends[:10]:  # Top 10 trends
            direction_icon = {"up": "📈", "down": "📉", "stable": "➡️"}.get(
                trend.direction, "➡️"
            )
            items.append(f"""
            <div class="trend-item">
                {direction_icon} <strong>{trend.metric_name}</strong>
                <span class="sparkline"></span>
                <span>slope: {trend.slope:.2e}, volatility: {trend.volatility:.4f}</span>
            </div>
            """)

        return f"""
        <div class="trends">
            <h2>Metric Trends</h2>
            {"".join(items)}
        </div>
        """


class EvaluationReporter:
    """Main report generator for SAE evaluation.

    Orchestrates metrics collection, regression detection,
    and report generation.

    Example:
        reporter = EvaluationReporter()
        report = reporter.generate(
            metrics=sae_metrics.compute_all(),
            benchmark_results=benchmark_suite.run(sae),
            baseline_comparison=baselines.evaluate()
        )
        report.save("evaluation_report.html")
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize reporter.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig.standard()
        self._regression_detector = RegressionDetector(
            threshold=self.config.regression_threshold
        )
        self._dashboard = MetricsDashboard(self.config)
        self._history: List[Dict[str, float]] = []

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """Set baseline metrics for regression detection.

        Args:
            metrics: Baseline metric values
        """
        self._regression_detector.set_baseline(metrics)

    def add_to_history(self, metrics: Dict[str, float]) -> None:
        """Add metrics to history.

        Args:
            metrics: Current metric values
        """
        self._history.append(metrics.copy())
        self._regression_detector.add_to_history(metrics)

    def generate(
        self,
        metrics: Dict[str, float],
        benchmark_results: Optional[Dict[str, Any]] = None,
        baseline_comparison: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Generate evaluation report.

        Args:
            metrics: Main evaluation metrics
            benchmark_results: Benchmark suite results
            baseline_comparison: Baseline comparison results

        Returns:
            Complete evaluation report
        """
        # Add to history
        self.add_to_history(metrics)

        # Detect regressions
        regressions = self._regression_detector.detect(metrics)

        # Get trends
        trends = self._regression_detector.get_trends()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, regressions, benchmark_results, baseline_comparison
        )

        return EvaluationReport(
            timestamp=datetime.now(),
            config=self.config,
            metrics=metrics,
            benchmark_results=benchmark_results,
            baseline_comparison=baseline_comparison,
            regressions=regressions,
            trends=trends,
            recommendations=recommendations,
            metadata={
                "history_size": len(self._history),
                "generator_version": "1.0.0",
            },
        )

    def _generate_recommendations(
        self,
        metrics: Dict[str, float],
        regressions: List[RegressionAlert],
        benchmark_results: Optional[Dict[str, Any]],
        baseline_comparison: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on analysis.

        Args:
            metrics: Current metrics
            regressions: Detected regressions
            benchmark_results: Benchmark results
            baseline_comparison: Baseline comparison

        Returns:
            List of recommendations
        """
        recommendations = []

        # Regression-based recommendations
        for alert in regressions:
            if alert.severity in (AlertLevel.ERROR, AlertLevel.CRITICAL):
                recommendations.append(f"[URGENT] {alert.recommendation}")

        # Metric-based recommendations
        dead_ratio = metrics.get("dead_feature_ratio", 0)
        if dead_ratio > 0.3:
            recommendations.append(
                f"High dead feature ratio ({dead_ratio:.1%}). "
                "Enable ghost gradients or decrease L1 coefficient."
            )

        sparsity = metrics.get("sparsity_l0", 0)
        if sparsity > 0.2:
            recommendations.append(
                f"Low sparsity (L0={sparsity:.1%}). "
                "Increase L1 coefficient for better interpretability."
            )
        elif sparsity < 0.01:
            recommendations.append(
                f"Very high sparsity (L0={sparsity:.1%}). "
                "May be over-regularized, decrease L1 coefficient."
            )

        mse = metrics.get("reconstruction_mse", 0)
        if mse > 0.5:
            recommendations.append(
                f"High reconstruction error (MSE={mse:.4f}). "
                "Increase training time or SAE capacity."
            )

        # Benchmark-based recommendations
        if benchmark_results and not benchmark_results.get("passed", True):
            recommendations.append(
                "Some benchmarks failed. Review benchmark report for details."
            )

        # Baseline comparison recommendations
        if baseline_comparison and not baseline_comparison.get("sae_beats_all", True):
            best = baseline_comparison.get("best_baseline", "unknown")
            recommendations.append(
                f"SAE underperforms {best} baseline. "
                "Review training configuration and data quality."
            )

        # General recommendations
        if not recommendations:
            recommendations.append("All metrics within acceptable ranges. Continue monitoring.")

        return recommendations

    def render_dashboard(
        self,
        metrics: Dict[str, float],
        alerts: Optional[List[RegressionAlert]] = None,
    ) -> str:
        """Render interactive dashboard.

        Args:
            metrics: Current metrics
            alerts: Optional alerts

        Returns:
            HTML dashboard
        """
        trends = self._regression_detector.get_trends()
        return self._dashboard.render(metrics, alerts, trends)


# Convenience functions

def create_reporter(
    config: Optional[ReportConfig] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> EvaluationReporter:
    """Create configured evaluation reporter.

    Args:
        config: Report configuration
        baseline_metrics: Optional baseline for regression detection

    Returns:
        Configured reporter
    """
    reporter = EvaluationReporter(config)
    if baseline_metrics:
        reporter.set_baseline(baseline_metrics)
    return reporter


def quick_report(
    metrics: Dict[str, float],
    output_path: Optional[str] = None,
) -> EvaluationReport:
    """Generate quick summary report.

    Args:
        metrics: Evaluation metrics
        output_path: Optional output path

    Returns:
        Generated report
    """
    config = ReportConfig.quick()
    if output_path:
        config.output_path = output_path

    reporter = EvaluationReporter(config)
    report = reporter.generate(metrics)

    if output_path:
        report.save(output_path)

    return report


def generate_publication_report(
    metrics: Dict[str, float],
    benchmark_results: Dict[str, Any],
    baseline_comparison: Dict[str, Any],
    output_path: str,
) -> EvaluationReport:
    """Generate publication-quality report.

    Args:
        metrics: Evaluation metrics
        benchmark_results: Benchmark suite results
        baseline_comparison: Baseline comparison results
        output_path: Output file path

    Returns:
        Generated report
    """
    config = ReportConfig.publication()
    config.output_path = output_path

    reporter = EvaluationReporter(config)
    report = reporter.generate(metrics, benchmark_results, baseline_comparison)
    report.save(output_path)

    return report
