"""
Visualization - Generate charts, dashboards, and data exports

Provides:
- Terminal-based charts (plotext)
- HTML dashboards
- Export to CSV/JSON
- Time-series plots
- Heatmaps
- Distribution plots
"""

import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import statistics

try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    print("Warning: plotext not installed. Terminal charts will be unavailable.")


class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"


class Visualizer:
    """Generate visualizations for analytics data"""

    def __init__(self, performance_monitor=None, usage_analytics=None,
                 quality_metrics=None):
        """
        Initialize visualizer

        Args:
            performance_monitor: PerformanceMonitor instance
            usage_analytics: UsageAnalytics instance
            quality_metrics: QualityMetrics instance
        """
        self.performance = performance_monitor
        self.usage = usage_analytics
        self.quality = quality_metrics

    # ========================================================================
    # Terminal Visualizations (using plotext)
    # ========================================================================

    def plot_operation_times(self, operation: str = None, days: int = 7,
                            show: bool = True) -> str:
        """Plot operation times over time"""
        if not PLOTEXT_AVAILABLE:
            return "plotext not available"

        if not self.performance:
            return "Performance monitor not available"

        # Get data
        metrics = self.performance.get_recent_operations(limit=1000, operation=operation)

        if not metrics:
            return "No data available"

        # Filter by days
        cutoff = datetime.now() - timedelta(days=days)
        metrics = [m for m in metrics
                  if datetime.fromisoformat(m['timestamp']) > cutoff]

        # Prepare data
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in metrics]
        durations = [m['duration_ms'] for m in metrics]

        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, durations))
        timestamps, durations = zip(*sorted_data) if sorted_data else ([], [])

        # Plot
        plt.clear_figure()
        plt.plot_size(100, 30)

        x = list(range(len(timestamps)))
        plt.plot(x, durations, label=operation or "All operations")

        plt.xlabel("Time")
        plt.ylabel("Duration (ms)")
        plt.title(f"Operation Times - Last {days} Days")

        if show:
            plt.show()

        return plt.build()

    def plot_throughput(self, hours: int = 24, show: bool = True) -> str:
        """Plot operations per hour"""
        if not PLOTEXT_AVAILABLE:
            return "plotext not available"

        if not self.performance:
            return "Performance monitor not available"

        # Get hourly data (simplified - would need proper hourly grouping)
        metrics = self.performance.get_recent_operations(limit=1000)

        if not metrics:
            return "No data available"

        # Group by hour
        cutoff = datetime.now() - timedelta(hours=hours)
        hourly_counts = {}

        for m in metrics:
            ts = datetime.fromisoformat(m['timestamp'])
            if ts > cutoff:
                hour = ts.replace(minute=0, second=0, microsecond=0)
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        # Prepare data
        sorted_hours = sorted(hourly_counts.keys())
        counts = [hourly_counts[h] for h in sorted_hours]

        # Plot
        plt.clear_figure()
        plt.plot_size(100, 30)

        x = list(range(len(sorted_hours)))
        plt.bar(x, counts)

        plt.xlabel("Hour")
        plt.ylabel("Operations")
        plt.title(f"Throughput - Last {hours} Hours")

        if show:
            plt.show()

        return plt.build()

    def plot_resource_usage(self, hours: int = 24, show: bool = True) -> str:
        """Plot CPU and memory usage over time"""
        if not PLOTEXT_AVAILABLE:
            return "plotext not available"

        if not self.performance:
            return "Performance monitor not available"

        stats = self.performance.get_resource_stats(hours=hours)

        if not stats:
            return "No resource data available"

        plt.clear_figure()
        plt.plot_size(100, 30)

        # Simple bar chart of average and max
        categories = ['CPU %', 'Memory %']
        avg_values = [stats.get('avg_cpu_percent', 0), stats.get('avg_memory_percent', 0)]
        max_values = [stats.get('max_cpu_percent', 0), stats.get('max_memory_percent', 0)]

        plt.bar(categories, avg_values, label="Average")
        plt.bar(categories, max_values, label="Maximum")

        plt.ylabel("Percentage")
        plt.title(f"Resource Usage - Last {hours} Hours")

        if show:
            plt.show()

        return plt.build()

    def plot_usage_heatmap(self, days: int = 7, show: bool = True) -> str:
        """Plot usage heatmap by day and hour"""
        if not PLOTEXT_AVAILABLE:
            return "plotext not available"

        if not self.usage:
            return "Usage analytics not available"

        # Get hourly distribution
        hourly = self.usage.get_hourly_distribution(days=days)

        if not hourly:
            return "No usage data available"

        plt.clear_figure()
        plt.plot_size(100, 30)

        hours = [h['hour'] for h in hourly]
        counts = [h['event_count'] for h in hourly]

        plt.bar(hours, counts)

        plt.xlabel("Hour of Day")
        plt.ylabel("Events")
        plt.title(f"Usage Heatmap - Last {days} Days")

        if show:
            plt.show()

        return plt.build()

    def plot_quality_trend(self, prompt_name: str, branch: str = None,
                          days: int = 30, show: bool = True) -> str:
        """Plot quality scores over time for a prompt"""
        if not PLOTEXT_AVAILABLE:
            return "plotext not available"

        if not self.quality:
            return "Quality metrics not available"

        trend = self.quality.get_quality_trend(prompt_name, branch, days)

        if not trend:
            return f"No quality data for {prompt_name}"

        plt.clear_figure()
        plt.plot_size(100, 30)

        # Plot trend line
        info = f"{prompt_name} - {trend.trend_direction.upper()}"
        info += f" (avg: {trend.avg_score:.3f}, slope: {trend.trend_slope:.4f})"

        plt.title(info)
        plt.xlabel("Time")
        plt.ylabel("Quality Score")

        # Would need actual time series data for proper plotting
        # This is a simplified version
        plt.text(info, x=0, y=0)

        if show:
            plt.show()

        return plt.build()

    def plot_score_distribution(self, prompt_name: str = None, branch: str = None,
                               days: int = 30, show: bool = True) -> str:
        """Plot histogram of quality scores"""
        if not PLOTEXT_AVAILABLE:
            return "plotext not available"

        if not self.quality:
            return "Quality metrics not available"

        dist = self.quality.get_score_distribution(prompt_name, branch, days)

        if not dist or not dist['counts']:
            return "No quality data available"

        plt.clear_figure()
        plt.plot_size(100, 30)

        plt.hist(dist['counts'])

        plt.xlabel("Score")
        plt.ylabel("Frequency")
        title = f"Score Distribution - {prompt_name or 'All Prompts'}"
        title += f" (avg: {dist['avg_score']:.3f}, median: {dist['median_score']:.3f})"
        plt.title(title)

        if show:
            plt.show()

        return plt.build()

    # ========================================================================
    # HTML Dashboards
    # ========================================================================

    def generate_dashboard(self, output_path: str, days: int = 7) -> str:
        """Generate comprehensive HTML dashboard"""
        html = self._get_dashboard_template()

        # Gather data
        perf_stats = self._gather_performance_stats(days)
        usage_stats = self._gather_usage_stats(days)
        quality_stats = self._gather_quality_stats(days)

        # Replace placeholders
        html = html.replace('{{TITLE}}', f'Promptly Analytics Dashboard - Last {days} Days')
        html = html.replace('{{GENERATED_AT}}', datetime.now().isoformat())
        html = html.replace('{{DAYS}}', str(days))

        html = html.replace('{{PERF_STATS}}', json.dumps(perf_stats))
        html = html.replace('{{USAGE_STATS}}', json.dumps(usage_stats))
        html = html.replace('{{QUALITY_STATS}}', json.dumps(quality_stats))

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(html)

        return str(output_path)

    def _gather_performance_stats(self, days: int) -> Dict:
        """Gather performance statistics"""
        if not self.performance:
            return {}

        stats = self.performance.get_operation_stats()
        resource_stats = self.performance.get_resource_stats(hours=days * 24)
        throughput = self.performance.get_throughput(hours=days * 24)

        return {
            'operations': stats,
            'resources': resource_stats,
            'throughput': throughput
        }

    def _gather_usage_stats(self, days: int) -> Dict:
        """Gather usage statistics"""
        if not self.usage:
            return {}

        most_used = self.usage.get_most_used_prompts(days=days)
        branch_activity = self.usage.get_branch_activity(days=days)
        eval_stats = self.usage.get_evaluation_stats(days=days)
        chain_stats = self.usage.get_chain_stats(days=days)
        daily = self.usage.get_daily_activity(days=days)

        return {
            'most_used_prompts': most_used,
            'branch_activity': branch_activity,
            'evaluations': eval_stats,
            'chains': chain_stats,
            'daily_activity': daily
        }

    def _gather_quality_stats(self, days: int) -> Dict:
        """Gather quality statistics"""
        if not self.quality:
            return {}

        top_prompts = self.quality.get_top_performing_prompts(days=days)
        declining = self.quality.get_declining_prompts(days=days)
        by_evaluator = self.quality.get_quality_by_evaluator(days=days)
        alerts = self.quality.get_active_alerts()

        return {
            'top_performing': top_prompts,
            'declining': declining,
            'by_evaluator': by_evaluator,
            'active_alerts': alerts
        }

    def _get_dashboard_template(self) -> str:
        """Get HTML dashboard template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; margin-bottom: 10px; }
        .meta { color: #666; margin-bottom: 30px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #333;
            font-size: 18px;
            margin-bottom: 15px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .stat:last-child { border-bottom: none; }
        .stat-label { color: #666; }
        .stat-value {
            color: #333;
            font-weight: 600;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        .badge-info { background: #d1ecf1; color: #0c5460; }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }
        .trend-stable { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{TITLE}}</h1>
        <div class="meta">Generated at: {{GENERATED_AT}}</div>

        <h2 style="margin-top: 30px; margin-bottom: 15px;">Performance Overview</h2>
        <div class="grid" id="performance-grid"></div>

        <h2 style="margin-top: 30px; margin-bottom: 15px;">Usage Analytics</h2>
        <div class="grid" id="usage-grid"></div>

        <h2 style="margin-top: 30px; margin-bottom: 15px;">Quality Metrics</h2>
        <div class="grid" id="quality-grid"></div>
    </div>

    <script>
        const perfStats = {{PERF_STATS}};
        const usageStats = {{USAGE_STATS}};
        const qualityStats = {{QUALITY_STATS}};

        // Render performance stats
        function renderPerformance() {
            const grid = document.getElementById('performance-grid');

            // Operation stats
            if (perfStats.operations && Object.keys(perfStats.operations).length > 0) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Operation Performance</h2>';

                const table = document.createElement('table');
                table.innerHTML = '<tr><th>Operation</th><th>Count</th><th>Avg (ms)</th><th>Max (ms)</th></tr>';

                for (const [op, stats] of Object.entries(perfStats.operations)) {
                    const row = table.insertRow();
                    row.innerHTML = `
                        <td>${op}</td>
                        <td>${stats.count}</td>
                        <td>${stats.avg_ms.toFixed(2)}</td>
                        <td>${stats.max_ms.toFixed(2)}</td>
                    `;
                }

                card.appendChild(table);
                grid.appendChild(card);
            }

            // Resource usage
            if (perfStats.resources && Object.keys(perfStats.resources).length > 0) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Resource Usage</h2>';

                const stats = perfStats.resources;
                card.innerHTML += `
                    <div class="stat">
                        <span class="stat-label">Avg CPU</span>
                        <span class="stat-value">${(stats.avg_cpu_percent || 0).toFixed(2)}%</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Max CPU</span>
                        <span class="stat-value">${(stats.max_cpu_percent || 0).toFixed(2)}%</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Avg Memory</span>
                        <span class="stat-value">${(stats.avg_memory_mb || 0).toFixed(2)} MB</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Max Memory</span>
                        <span class="stat-value">${(stats.max_memory_mb || 0).toFixed(2)} MB</span>
                    </div>
                `;

                grid.appendChild(card);
            }

            // Throughput
            if (perfStats.throughput) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Throughput</h2>';

                const t = perfStats.throughput;
                card.innerHTML += `
                    <div class="stat">
                        <span class="stat-label">Total Operations</span>
                        <span class="stat-value">${t.total_operations}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Per Second</span>
                        <span class="stat-value">${t.operations_per_second.toFixed(3)}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Per Minute</span>
                        <span class="stat-value">${t.operations_per_minute.toFixed(2)}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Per Hour</span>
                        <span class="stat-value">${t.operations_per_hour.toFixed(2)}</span>
                    </div>
                `;

                grid.appendChild(card);
            }
        }

        // Render usage stats
        function renderUsage() {
            const grid = document.getElementById('usage-grid');

            // Most used prompts
            if (usageStats.most_used_prompts && usageStats.most_used_prompts.length > 0) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Most Used Prompts</h2>';

                const table = document.createElement('table');
                table.innerHTML = '<tr><th>Prompt</th><th>Branch</th><th>Accesses</th></tr>';

                usageStats.most_used_prompts.forEach(p => {
                    const row = table.insertRow();
                    row.innerHTML = `
                        <td>${p.prompt_name}</td>
                        <td><span class="badge badge-info">${p.branch}</span></td>
                        <td>${p.access_count}</td>
                    `;
                });

                card.appendChild(table);
                grid.appendChild(card);
            }

            // Evaluations
            if (usageStats.evaluations) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Evaluations</h2>';

                const e = usageStats.evaluations;
                card.innerHTML += `
                    <div class="stat">
                        <span class="stat-label">Total Evaluations</span>
                        <span class="stat-value">${e.total_evaluations}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Average Score</span>
                        <span class="stat-value">${(e.average_score || 0).toFixed(3)}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Per Day</span>
                        <span class="stat-value">${e.evaluations_per_day.toFixed(2)}</span>
                    </div>
                `;

                grid.appendChild(card);
            }

            // Chains
            if (usageStats.chains) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Chain Executions</h2>';

                const c = usageStats.chains;
                card.innerHTML += `
                    <div class="stat">
                        <span class="stat-label">Total Executions</span>
                        <span class="stat-value">${c.total_executions}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Success Rate</span>
                        <span class="stat-value">${c.success_rate.toFixed(2)}%</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Per Day</span>
                        <span class="stat-value">${c.executions_per_day.toFixed(2)}</span>
                    </div>
                `;

                grid.appendChild(card);
            }
        }

        // Render quality stats
        function renderQuality() {
            const grid = document.getElementById('quality-grid');

            // Top performing
            if (qualityStats.top_performing && qualityStats.top_performing.length > 0) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Top Performing Prompts</h2>';

                const table = document.createElement('table');
                table.innerHTML = '<tr><th>Prompt</th><th>Branch</th><th>Avg Score</th><th>Samples</th></tr>';

                qualityStats.top_performing.forEach(p => {
                    const row = table.insertRow();
                    row.innerHTML = `
                        <td>${p.prompt_name}</td>
                        <td><span class="badge badge-info">${p.branch}</span></td>
                        <td>${p.avg_score.toFixed(3)}</td>
                        <td>${p.measurements}</td>
                    `;
                });

                card.appendChild(table);
                grid.appendChild(card);
            }

            // Declining prompts
            if (qualityStats.declining && qualityStats.declining.length > 0) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Declining Quality</h2>';

                const table = document.createElement('table');
                table.innerHTML = '<tr><th>Prompt</th><th>Branch</th><th>Trend</th><th>Avg Score</th></tr>';

                qualityStats.declining.forEach(p => {
                    const row = table.insertRow();
                    row.innerHTML = `
                        <td>${p.prompt_name}</td>
                        <td><span class="badge badge-info">${p.branch}</span></td>
                        <td class="trend-down">${p.trend_slope.toFixed(4)}</td>
                        <td>${p.avg_score.toFixed(3)}</td>
                    `;
                });

                card.appendChild(table);
                grid.appendChild(card);
            }

            // Active alerts
            if (qualityStats.active_alerts && qualityStats.active_alerts.length > 0) {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = '<h2>Active Alerts</h2>';

                const table = document.createElement('table');
                table.innerHTML = '<tr><th>Prompt</th><th>Type</th><th>Message</th></tr>';

                qualityStats.active_alerts.forEach(a => {
                    const row = table.insertRow();
                    const badgeClass = a.severity === 'error' ? 'badge-danger' : 'badge-warning';
                    row.innerHTML = `
                        <td>${a.prompt_name}</td>
                        <td><span class="badge ${badgeClass}">${a.alert_type}</span></td>
                        <td>${a.message}</td>
                    `;
                });

                card.appendChild(table);
                grid.appendChild(card);
            }
        }

        // Render all
        renderPerformance();
        renderUsage();
        renderQuality();
    </script>
</body>
</html>
"""

    # ========================================================================
    # Export Functions
    # ========================================================================

    def export_to_csv(self, data_type: str, output_path: str, days: int = 30) -> str:
        """Export analytics data to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if data_type == 'performance' and self.performance:
            metrics = self.performance.get_recent_operations(limit=10000)
            cutoff = datetime.now() - timedelta(days=days)
            metrics = [m for m in metrics
                      if datetime.fromisoformat(m['timestamp']) > cutoff]

            with open(output_path, 'w', newline='') as f:
                if metrics:
                    writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
                    writer.writeheader()
                    writer.writerows(metrics)

        elif data_type == 'usage' and self.usage:
            daily = self.usage.get_daily_activity(days=days)

            with open(output_path, 'w', newline='') as f:
                if daily:
                    writer = csv.DictWriter(f, fieldnames=daily[0].keys())
                    writer.writeheader()
                    writer.writerows(daily)

        elif data_type == 'quality' and self.quality:
            top = self.quality.get_top_performing_prompts(limit=100, days=days)

            with open(output_path, 'w', newline='') as f:
                if top:
                    writer = csv.DictWriter(f, fieldnames=top[0].keys())
                    writer.writeheader()
                    writer.writerows(top)

        return str(output_path)

    def export_to_json(self, output_path: str, days: int = 30) -> str:
        """Export all analytics data to JSON"""
        data = {
            'export_date': datetime.now().isoformat(),
            'days': days,
            'performance': {},
            'usage': {},
            'quality': {}
        }

        # Gather all data
        if self.performance:
            data['performance'] = self._gather_performance_stats(days)

        if self.usage:
            data['usage'] = self._gather_usage_stats(days)

        if self.quality:
            data['quality'] = self._gather_quality_stats(days)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(output_path)
