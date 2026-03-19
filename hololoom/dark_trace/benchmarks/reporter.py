"""
Benchmark Reporter for Dark Trace Operations

Generates reports in multiple formats (HTML, JSON, Markdown)
and exports Prometheus metrics for monitoring integration.

Author: HoloLoom Team
Created: December 2025
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .suite import BenchmarkResult, BenchmarkSuiteResult


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Dark Trace Benchmark Report"
    include_charts: bool = True
    include_raw_data: bool = True
    include_recommendations: bool = True
    prometheus_prefix: str = "dark_trace_benchmark"

    # Thresholds for recommendations
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0
    memory_warning_mb: float = 100.0
    memory_critical_mb: float = 500.0


class BenchmarkReporter:
    """
    Generate reports from benchmark results.

    Supports multiple output formats:
    - HTML: Interactive report with charts
    - JSON: Machine-readable format
    - Markdown: Documentation-friendly format
    - Prometheus: Metrics export for monitoring
    """

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()

    def generate_html(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate interactive HTML report."""
        html_parts = [
            self._html_header(),
            self._html_summary(suite_result),
            self._html_results_table(suite_result),
        ]

        if self.config.include_charts:
            html_parts.append(self._html_charts(suite_result))

        if self.config.include_recommendations:
            html_parts.append(self._html_recommendations(suite_result))

        if self.config.include_raw_data:
            html_parts.append(self._html_raw_data(suite_result))

        html_parts.append(self._html_footer())

        return "\n".join(html_parts)

    def generate_json(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate JSON report."""
        report = {
            "title": self.config.title,
            "generated_at": datetime.now().isoformat(),
            "suite_result": suite_result.to_dict(),
            "analysis": self._analyze_results(suite_result),
        }

        if self.config.include_recommendations:
            report["recommendations"] = self._generate_recommendations(suite_result)

        return json.dumps(report, indent=2)

    def generate_markdown(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate Markdown report."""
        lines = [
            f"# {self.config.title}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Time:** {suite_result.total_time_seconds:.2f}s",
            f"**Benchmarks Run:** {len(suite_result.results)}",
            "",
            "---",
            "",
            "## Summary",
            "",
        ]

        # Category breakdown
        by_category = self._group_by_category(suite_result.results)
        for category, results in by_category.items():
            avg_latency = sum(r.mean_latency_ms for r in results) / len(results)
            lines.append(f"- **{category}**: {len(results)} benchmarks, avg {avg_latency:.2f}ms")

        lines.extend(["", "## Results", ""])

        # Results table
        lines.append("| Benchmark | Latency (ms) | p95 (ms) | Memory (MB) | Throughput |")
        lines.append("|-----------|-------------|----------|-------------|------------|")

        for result in suite_result.results:
            lines.append(
                f"| {result.name} | {result.mean_latency_ms:.2f} ± {result.std_latency_ms:.2f} | "
                f"{result.p95_latency_ms:.2f} | {result.memory_peak_mb:.2f} | {result.throughput:.1f}/s |"
            )

        if self.config.include_recommendations:
            recs = self._generate_recommendations(suite_result)
            if recs:
                lines.extend(["", "## Recommendations", ""])
                for rec in recs:
                    icon = "⚠️" if rec["severity"] == "warning" else "🔴"
                    lines.append(f"{icon} **{rec['benchmark']}**: {rec['message']}")

        return "\n".join(lines)

    def export_prometheus_metrics(self, suite_result: BenchmarkSuiteResult) -> str:
        """Export metrics in Prometheus format."""
        prefix = self.config.prometheus_prefix
        lines = []
        timestamp = int(time.time() * 1000)

        # Help and type declarations
        metrics_spec = [
            ("latency_ms", "gauge", "Operation latency in milliseconds"),
            ("latency_p95_ms", "gauge", "95th percentile latency in milliseconds"),
            ("latency_p99_ms", "gauge", "99th percentile latency in milliseconds"),
            ("memory_delta_mb", "gauge", "Memory change in megabytes"),
            ("memory_peak_mb", "gauge", "Peak memory usage in megabytes"),
            ("throughput", "gauge", "Operations per second"),
        ]

        for metric, mtype, help_text in metrics_spec:
            lines.append(f"# HELP {prefix}_{metric} {help_text}")
            lines.append(f"# TYPE {prefix}_{metric} {mtype}")

        lines.append("")

        # Metric values
        for result in suite_result.results:
            labels = f'benchmark="{result.name}",category="{result.category.value}"'

            lines.append(f'{prefix}_latency_ms{{{labels}}} {result.mean_latency_ms:.6f} {timestamp}')
            lines.append(f'{prefix}_latency_p95_ms{{{labels}}} {result.p95_latency_ms:.6f} {timestamp}')
            lines.append(f'{prefix}_latency_p99_ms{{{labels}}} {result.p99_latency_ms:.6f} {timestamp}')
            lines.append(f'{prefix}_memory_delta_mb{{{labels}}} {result.memory_delta_mb:.6f} {timestamp}')
            lines.append(f'{prefix}_memory_peak_mb{{{labels}}} {result.memory_peak_mb:.6f} {timestamp}')
            lines.append(f'{prefix}_throughput{{{labels}}} {result.throughput:.6f} {timestamp}')

        # Suite-level metrics
        lines.append("")
        lines.append(f"# HELP {prefix}_total_time_seconds Total benchmark suite duration")
        lines.append(f"# TYPE {prefix}_total_time_seconds gauge")
        lines.append(f'{prefix}_total_time_seconds {suite_result.total_time_seconds:.6f} {timestamp}')

        lines.append(f"# HELP {prefix}_benchmarks_total Number of benchmarks run")
        lines.append(f"# TYPE {prefix}_benchmarks_total gauge")
        lines.append(f'{prefix}_benchmarks_total {len(suite_result.results)} {timestamp}')

        return "\n".join(lines)

    def save(
        self,
        suite_result: BenchmarkSuiteResult,
        path: str | Path,
        format: str = "html"
    ) -> None:
        """Save report to file."""
        path = Path(path)

        if format == "html":
            content = self.generate_html(suite_result)
        elif format == "json":
            content = self.generate_json(suite_result)
        elif format == "markdown" or format == "md":
            content = self.generate_markdown(suite_result)
        elif format == "prometheus":
            content = self.export_prometheus_metrics(suite_result)
        else:
            raise ValueError(f"Unknown format: {format}")

        path.write_text(content)

    def _html_header(self) -> str:
        """Generate HTML header with styles."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.config.title}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #e94560;
            --success: #0f9b0f;
            --warning: #ff9f1c;
            --border: #333;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: var(--accent);
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: rgba(233, 69, 96, 0.2);
            color: var(--accent);
        }}
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .metric {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            margin: 2px;
        }}
        .metric.good {{ background: var(--success); color: white; }}
        .metric.warning {{ background: var(--warning); color: black; }}
        .metric.critical {{ background: var(--accent); color: white; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
        }}
        .summary-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent);
        }}
        .summary-label {{
            font-size: 0.9em;
            color: #888;
        }}
        .chart-container {{
            height: 300px;
            margin: 20px 0;
        }}
        .recommendation {{
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid var(--warning);
            background: rgba(255, 159, 28, 0.1);
            border-radius: 0 4px 4px 0;
        }}
        .recommendation.critical {{
            border-left-color: var(--accent);
            background: rgba(233, 69, 96, 0.1);
        }}
        pre {{
            background: #0a0a15;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.85em;
        }}
        code {{
            font-family: 'Monaco', 'Consolas', monospace;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{self.config.title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    def _html_summary(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate HTML summary section."""
        total_benchmarks = len(suite_result.results)
        avg_latency = sum(r.mean_latency_ms for r in suite_result.results) / total_benchmarks if total_benchmarks else 0
        max_memory = max((r.memory_peak_mb for r in suite_result.results), default=0)
        avg_throughput = sum(r.throughput for r in suite_result.results) / total_benchmarks if total_benchmarks else 0

        return f"""
    <div class="card">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">{total_benchmarks}</div>
                <div class="summary-label">Benchmarks Run</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{suite_result.total_time_seconds:.1f}s</div>
                <div class="summary-label">Total Duration</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{avg_latency:.1f}ms</div>
                <div class="summary-label">Avg Latency</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{max_memory:.1f}MB</div>
                <div class="summary-label">Peak Memory</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{avg_throughput:.0f}/s</div>
                <div class="summary-label">Avg Throughput</div>
            </div>
        </div>
    </div>
"""

    def _html_results_table(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate HTML results table."""
        rows = []
        for result in suite_result.results:
            latency_class = self._get_metric_class(
                result.mean_latency_ms,
                self.config.latency_warning_ms,
                self.config.latency_critical_ms
            )
            memory_class = self._get_metric_class(
                result.memory_peak_mb,
                self.config.memory_warning_mb,
                self.config.memory_critical_mb
            )

            rows.append(f"""
            <tr>
                <td>{result.name}</td>
                <td>{result.category.value}</td>
                <td><span class="metric {latency_class}">{result.mean_latency_ms:.2f} ± {result.std_latency_ms:.2f}</span></td>
                <td>{result.p95_latency_ms:.2f}</td>
                <td>{result.p99_latency_ms:.2f}</td>
                <td><span class="metric {memory_class}">{result.memory_peak_mb:.2f}</span></td>
                <td>{result.throughput:.1f}</td>
            </tr>
""")

        return f"""
    <div class="card">
        <h2>Benchmark Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Category</th>
                    <th>Latency (ms)</th>
                    <th>p95 (ms)</th>
                    <th>p99 (ms)</th>
                    <th>Peak Memory (MB)</th>
                    <th>Throughput</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
"""

    def _html_charts(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate HTML charts section (SVG-based, no external deps)."""
        # Simple bar chart for latency comparison
        results = suite_result.results[:10]  # Limit to 10 for readability
        if not results:
            return ""

        max_latency = max(r.mean_latency_ms for r in results)
        bar_width = 40
        spacing = 10
        chart_height = 200
        chart_width = len(results) * (bar_width + spacing) + 50

        bars = []
        labels = []
        for i, result in enumerate(results):
            x = 40 + i * (bar_width + spacing)
            height = (result.mean_latency_ms / max_latency) * (chart_height - 40) if max_latency > 0 else 0
            y = chart_height - height - 20

            bars.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{height}" '
                f'fill="#e94560" rx="2"/>'
            )

            # Shortened label
            short_name = result.name[:8] + "..." if len(result.name) > 8 else result.name
            labels.append(
                f'<text x="{x + bar_width/2}" y="{chart_height - 5}" '
                f'text-anchor="middle" fill="#888" font-size="10">{short_name}</text>'
            )

        return f"""
    <div class="card">
        <h2>Latency Comparison</h2>
        <svg width="{chart_width}" height="{chart_height}" style="background: #0a0a15; border-radius: 4px;">
            {"".join(bars)}
            {"".join(labels)}
            <text x="15" y="20" fill="#888" font-size="12">ms</text>
        </svg>
    </div>
"""

    def _html_recommendations(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate HTML recommendations section."""
        recs = self._generate_recommendations(suite_result)
        if not recs:
            return """
    <div class="card">
        <h2>Recommendations</h2>
        <p>✅ All benchmarks within acceptable thresholds.</p>
    </div>
"""

        rec_html = []
        for rec in recs:
            css_class = "critical" if rec["severity"] == "critical" else ""
            icon = "🔴" if rec["severity"] == "critical" else "⚠️"
            rec_html.append(f"""
        <div class="recommendation {css_class}">
            <strong>{icon} {rec['benchmark']}</strong><br>
            {rec['message']}
        </div>
""")

        return f"""
    <div class="card">
        <h2>Recommendations</h2>
        {"".join(rec_html)}
    </div>
"""

    def _html_raw_data(self, suite_result: BenchmarkSuiteResult) -> str:
        """Generate HTML raw data section."""
        raw_json = json.dumps(suite_result.to_dict(), indent=2)
        return f"""
    <div class="card">
        <h2>Raw Data</h2>
        <details>
            <summary>Click to expand JSON</summary>
            <pre><code>{raw_json}</code></pre>
        </details>
    </div>
"""

    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return """
</div>
</body>
</html>
"""

    def _get_metric_class(self, value: float, warning: float, critical: float) -> str:
        """Get CSS class based on metric thresholds."""
        if value >= critical:
            return "critical"
        elif value >= warning:
            return "warning"
        return "good"

    def _group_by_category(
        self,
        results: list[BenchmarkResult]
    ) -> dict[str, list[BenchmarkResult]]:
        """Group results by category."""
        groups: dict[str, list[BenchmarkResult]] = {}
        for result in results:
            category = result.category.value
            if category not in groups:
                groups[category] = []
            groups[category].append(result)
        return groups

    def _analyze_results(self, suite_result: BenchmarkSuiteResult) -> dict[str, Any]:
        """Analyze benchmark results."""
        results = suite_result.results
        if not results:
            return {}

        latencies = [r.mean_latency_ms for r in results]
        memories = [r.memory_peak_mb for r in results]

        return {
            "latency": {
                "min": min(latencies),
                "max": max(latencies),
                "avg": sum(latencies) / len(latencies),
                "slowest": max(results, key=lambda r: r.mean_latency_ms).name,
                "fastest": min(results, key=lambda r: r.mean_latency_ms).name,
            },
            "memory": {
                "min": min(memories),
                "max": max(memories),
                "avg": sum(memories) / len(memories),
                "highest": max(results, key=lambda r: r.memory_peak_mb).name,
                "lowest": min(results, key=lambda r: r.memory_peak_mb).name,
            },
            "by_category": {
                category: {
                    "count": len(cat_results),
                    "avg_latency": sum(r.mean_latency_ms for r in cat_results) / len(cat_results),
                }
                for category, cat_results in self._group_by_category(results).items()
            },
        }

    def _generate_recommendations(
        self,
        suite_result: BenchmarkSuiteResult
    ) -> list[dict[str, str]]:
        """Generate recommendations based on results."""
        recommendations = []

        for result in suite_result.results:
            # Latency recommendations
            if result.mean_latency_ms >= self.config.latency_critical_ms:
                recommendations.append({
                    "benchmark": result.name,
                    "severity": "critical",
                    "message": f"Latency ({result.mean_latency_ms:.1f}ms) exceeds critical threshold "
                              f"({self.config.latency_critical_ms}ms). Consider optimizing or caching.",
                })
            elif result.mean_latency_ms >= self.config.latency_warning_ms:
                recommendations.append({
                    "benchmark": result.name,
                    "severity": "warning",
                    "message": f"Latency ({result.mean_latency_ms:.1f}ms) exceeds warning threshold "
                              f"({self.config.latency_warning_ms}ms). Monitor for degradation.",
                })

            # Memory recommendations
            if result.memory_peak_mb >= self.config.memory_critical_mb:
                recommendations.append({
                    "benchmark": result.name,
                    "severity": "critical",
                    "message": f"Memory ({result.memory_peak_mb:.1f}MB) exceeds critical threshold "
                              f"({self.config.memory_critical_mb}MB). Investigate memory leaks.",
                })
            elif result.memory_peak_mb >= self.config.memory_warning_mb:
                recommendations.append({
                    "benchmark": result.name,
                    "severity": "warning",
                    "message": f"Memory ({result.memory_peak_mb:.1f}MB) exceeds warning threshold "
                              f"({self.config.memory_warning_mb}MB). Consider memory optimization.",
                })

            # High variance recommendation
            if result.std_latency_ms > result.mean_latency_ms * 0.5:
                recommendations.append({
                    "benchmark": result.name,
                    "severity": "warning",
                    "message": f"High latency variance (std={result.std_latency_ms:.1f}ms vs "
                              f"mean={result.mean_latency_ms:.1f}ms). Performance may be inconsistent.",
                })

        return recommendations


def generate_report(
    suite_result: BenchmarkSuiteResult,
    format: str = "html",
    config: ReportConfig | None = None
) -> str:
    """Convenience function to generate a report."""
    reporter = BenchmarkReporter(config)

    if format == "html":
        return reporter.generate_html(suite_result)
    elif format == "json":
        return reporter.generate_json(suite_result)
    elif format == "markdown" or format == "md":
        return reporter.generate_markdown(suite_result)
    elif format == "prometheus":
        return reporter.export_prometheus_metrics(suite_result)
    else:
        raise ValueError(f"Unknown format: {format}")


def export_prometheus_metrics(suite_result: BenchmarkSuiteResult) -> str:
    """Convenience function to export Prometheus metrics."""
    reporter = BenchmarkReporter()
    return reporter.export_prometheus_metrics(suite_result)
