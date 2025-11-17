"""
Reporting - Generate comprehensive reports in multiple formats

Provides:
- Daily/weekly/monthly summaries
- Quality reports
- Performance reports
- Usage reports
- Export formats (PDF, HTML, Markdown)
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field


class ReportFormat(Enum):
    """Available report formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


class ReportPeriod(Enum):
    """Report time periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    period: ReportPeriod
    format: ReportFormat
    include_performance: bool = True
    include_usage: bool = True
    include_quality: bool = True
    include_charts: bool = False
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class ReportGenerator:
    """Generate comprehensive analytics reports"""

    def __init__(self, performance_monitor=None, usage_analytics=None,
                 quality_metrics=None, visualizer=None):
        """
        Initialize report generator

        Args:
            performance_monitor: PerformanceMonitor instance
            usage_analytics: UsageAnalytics instance
            quality_metrics: QualityMetrics instance
            visualizer: Visualizer instance
        """
        self.performance = performance_monitor
        self.usage = usage_analytics
        self.quality = quality_metrics
        self.visualizer = visualizer

    def generate_report(self, config: ReportConfig, output_path: str) -> str:
        """
        Generate a report based on configuration

        Args:
            config: Report configuration
            output_path: Where to save the report

        Returns:
            Path to generated report
        """
        # Determine date range
        if config.start_date and config.end_date:
            start_date = config.start_date
            end_date = config.end_date
        else:
            end_date = datetime.now()
            if config.period == ReportPeriod.DAILY:
                start_date = end_date - timedelta(days=1)
            elif config.period == ReportPeriod.WEEKLY:
                start_date = end_date - timedelta(days=7)
            elif config.period == ReportPeriod.MONTHLY:
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=7)

        days = (end_date - start_date).days

        # Gather data
        report_data = {
            'title': f'{config.period.value.title()} Analytics Report',
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': days
            },
            'sections': {}
        }

        if config.include_performance and self.performance:
            report_data['sections']['performance'] = self._gather_performance_report(days)

        if config.include_usage and self.usage:
            report_data['sections']['usage'] = self._gather_usage_report(days)

        if config.include_quality and self.quality:
            report_data['sections']['quality'] = self._gather_quality_report(days)

        # Generate report in requested format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.format == ReportFormat.MARKDOWN:
            content = self._generate_markdown_report(report_data)
        elif config.format == ReportFormat.HTML:
            content = self._generate_html_report(report_data)
        elif config.format == ReportFormat.JSON:
            content = json.dumps(report_data, indent=2, default=str)
        else:  # TEXT
            content = self._generate_text_report(report_data)

        with open(output_path, 'w') as f:
            f.write(content)

        return str(output_path)

    def generate_daily_summary(self, output_path: str, date: datetime = None) -> str:
        """Generate daily summary report"""
        if date is None:
            date = datetime.now() - timedelta(days=1)

        config = ReportConfig(
            period=ReportPeriod.DAILY,
            format=ReportFormat.MARKDOWN,
            start_date=date.replace(hour=0, minute=0, second=0, microsecond=0),
            end_date=date.replace(hour=23, minute=59, second=59, microsecond=999999)
        )

        return self.generate_report(config, output_path)

    def generate_weekly_summary(self, output_path: str) -> str:
        """Generate weekly summary report"""
        config = ReportConfig(
            period=ReportPeriod.WEEKLY,
            format=ReportFormat.MARKDOWN
        )

        return self.generate_report(config, output_path)

    def generate_monthly_summary(self, output_path: str) -> str:
        """Generate monthly summary report"""
        config = ReportConfig(
            period=ReportPeriod.MONTHLY,
            format=ReportFormat.MARKDOWN
        )

        return self.generate_report(config, output_path)

    def generate_performance_report(self, output_path: str, days: int = 7) -> str:
        """Generate detailed performance report"""
        config = ReportConfig(
            period=ReportPeriod.CUSTOM,
            format=ReportFormat.MARKDOWN,
            include_performance=True,
            include_usage=False,
            include_quality=False,
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now()
        )

        return self.generate_report(config, output_path)

    def generate_quality_report(self, output_path: str, days: int = 30) -> str:
        """Generate detailed quality report"""
        config = ReportConfig(
            period=ReportPeriod.CUSTOM,
            format=ReportFormat.MARKDOWN,
            include_performance=False,
            include_usage=False,
            include_quality=True,
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now()
        )

        return self.generate_report(config, output_path)

    # ========================================================================
    # Data Gathering
    # ========================================================================

    def _gather_performance_report(self, days: int) -> Dict:
        """Gather performance data for report"""
        data = {
            'summary': {},
            'operations': {},
            'resources': {},
            'throughput': {},
            'slow_operations': []
        }

        # Operation statistics
        op_stats = self.performance.get_operation_stats()
        data['operations'] = op_stats

        # Calculate summary
        total_ops = sum(s['count'] for s in op_stats.values())
        avg_duration = sum(s['avg_ms'] for s in op_stats.values()) / len(op_stats) if op_stats else 0

        data['summary'] = {
            'total_operations': total_ops,
            'unique_operations': len(op_stats),
            'avg_duration_ms': avg_duration
        }

        # Resource statistics
        data['resources'] = self.performance.get_resource_stats(hours=days * 24)

        # Throughput
        data['throughput'] = self.performance.get_throughput(hours=days * 24)

        # Slow operations
        data['slow_operations'] = self.performance.get_slow_operations(threshold_ms=1000, limit=10)

        return data

    def _gather_usage_report(self, days: int) -> Dict:
        """Gather usage data for report"""
        data = {
            'summary': {},
            'most_used_prompts': [],
            'branch_activity': [],
            'user_activity': [],
            'evaluations': {},
            'chains': {},
            'hourly_distribution': [],
            'daily_activity': []
        }

        # Most used prompts
        data['most_used_prompts'] = self.usage.get_most_used_prompts(limit=10, days=days)

        # Branch activity
        data['branch_activity'] = self.usage.get_branch_activity(days=days)

        # User activity
        data['user_activity'] = self.usage.get_user_activity(days=days, limit=10)

        # Evaluations
        data['evaluations'] = self.usage.get_evaluation_stats(days=days)

        # Chains
        data['chains'] = self.usage.get_chain_stats(days=days)

        # Hourly distribution
        data['hourly_distribution'] = self.usage.get_hourly_distribution(days=days)

        # Daily activity
        data['daily_activity'] = self.usage.get_daily_activity(days=days)

        # Summary
        total_events = sum(d['event_count'] for d in data['daily_activity'])
        data['summary'] = {
            'total_events': total_events,
            'avg_events_per_day': total_events / days if days > 0 else 0,
            'active_branches': len(data['branch_activity']),
            'active_users': len(data['user_activity'])
        }

        return data

    def _gather_quality_report(self, days: int) -> Dict:
        """Gather quality data for report"""
        data = {
            'summary': {},
            'top_performing': [],
            'declining': [],
            'by_evaluator': [],
            'alerts': [],
            'score_distribution': {}
        }

        # Top performing prompts
        data['top_performing'] = self.quality.get_top_performing_prompts(limit=10, days=days)

        # Declining prompts
        data['declining'] = self.quality.get_declining_prompts(days=days)

        # By evaluator
        data['by_evaluator'] = self.quality.get_quality_by_evaluator(days=days)

        # Active alerts
        data['alerts'] = self.quality.get_active_alerts()

        # Score distribution
        data['score_distribution'] = self.quality.get_score_distribution(days=days)

        # Summary
        if data['top_performing']:
            avg_top_score = sum(p['avg_score'] for p in data['top_performing']) / len(data['top_performing'])
        else:
            avg_top_score = 0

        data['summary'] = {
            'total_prompts_evaluated': len(data['top_performing']) + len(data['declining']),
            'avg_top_score': avg_top_score,
            'declining_count': len(data['declining']),
            'active_alerts': len(data['alerts'])
        }

        return data

    # ========================================================================
    # Markdown Generation
    # ========================================================================

    def _generate_markdown_report(self, data: Dict) -> str:
        """Generate Markdown format report"""
        md = []

        # Header
        md.append(f"# {data['title']}\n")
        md.append(f"**Generated:** {data['generated_at']}\n")
        md.append(f"**Period:** {data['period']['start']} to {data['period']['end']} ({data['period']['days']} days)\n")
        md.append("---\n")

        # Performance section
        if 'performance' in data['sections']:
            md.append(self._markdown_performance_section(data['sections']['performance']))

        # Usage section
        if 'usage' in data['sections']:
            md.append(self._markdown_usage_section(data['sections']['usage']))

        # Quality section
        if 'quality' in data['sections']:
            md.append(self._markdown_quality_section(data['sections']['quality']))

        return '\n'.join(md)

    def _markdown_performance_section(self, perf: Dict) -> str:
        """Generate performance section in Markdown"""
        md = []
        md.append("## Performance\n")

        # Summary
        summary = perf['summary']
        md.append("### Summary\n")
        md.append(f"- **Total Operations:** {summary['total_operations']}")
        md.append(f"- **Unique Operations:** {summary['unique_operations']}")
        md.append(f"- **Avg Duration:** {summary['avg_duration_ms']:.2f} ms\n")

        # Resources
        if perf['resources']:
            res = perf['resources']
            md.append("### Resource Usage\n")
            md.append(f"- **Avg CPU:** {res.get('avg_cpu_percent', 0):.2f}%")
            md.append(f"- **Max CPU:** {res.get('max_cpu_percent', 0):.2f}%")
            md.append(f"- **Avg Memory:** {res.get('avg_memory_mb', 0):.2f} MB")
            md.append(f"- **Max Memory:** {res.get('max_memory_mb', 0):.2f} MB\n")

        # Throughput
        if perf['throughput']:
            t = perf['throughput']
            md.append("### Throughput\n")
            md.append(f"- **Operations/sec:** {t['operations_per_second']:.3f}")
            md.append(f"- **Operations/min:** {t['operations_per_minute']:.2f}")
            md.append(f"- **Operations/hour:** {t['operations_per_hour']:.2f}\n")

        # Top operations
        if perf['operations']:
            md.append("### Operation Performance\n")
            md.append("| Operation | Count | Avg (ms) | Max (ms) |")
            md.append("|-----------|-------|----------|----------|")
            for op, stats in list(perf['operations'].items())[:10]:
                md.append(f"| {op} | {stats['count']} | {stats['avg_ms']:.2f} | {stats['max_ms']:.2f} |")
            md.append("")

        # Slow operations
        if perf['slow_operations']:
            md.append("### Slow Operations (>1s)\n")
            md.append("| Operation | Duration (ms) | Timestamp |")
            md.append("|-----------|---------------|-----------|")
            for op in perf['slow_operations'][:5]:
                md.append(f"| {op['operation']} | {op['duration_ms']:.2f} | {op['timestamp']} |")
            md.append("")

        return '\n'.join(md)

    def _markdown_usage_section(self, usage: Dict) -> str:
        """Generate usage section in Markdown"""
        md = []
        md.append("## Usage Analytics\n")

        # Summary
        summary = usage['summary']
        md.append("### Summary\n")
        md.append(f"- **Total Events:** {summary['total_events']}")
        md.append(f"- **Avg Events/Day:** {summary['avg_events_per_day']:.2f}")
        md.append(f"- **Active Branches:** {summary['active_branches']}")
        md.append(f"- **Active Users:** {summary['active_users']}\n")

        # Most used prompts
        if usage['most_used_prompts']:
            md.append("### Most Used Prompts\n")
            md.append("| Prompt | Branch | Accesses |")
            md.append("|--------|--------|----------|")
            for p in usage['most_used_prompts']:
                md.append(f"| {p['prompt_name']} | {p['branch']} | {p['access_count']} |")
            md.append("")

        # Branch activity
        if usage['branch_activity']:
            md.append("### Branch Activity\n")
            md.append("| Branch | Events | Unique Prompts | Active Days |")
            md.append("|--------|--------|----------------|-------------|")
            for b in usage['branch_activity']:
                md.append(f"| {b['branch']} | {b['total_events']} | {b['unique_prompts']} | {b['active_days']} |")
            md.append("")

        # Evaluations
        if usage['evaluations']:
            e = usage['evaluations']
            md.append("### Evaluations\n")
            md.append(f"- **Total Evaluations:** {e['total_evaluations']}")
            md.append(f"- **Average Score:** {e.get('average_score', 0):.3f}")
            md.append(f"- **Evaluations/Day:** {e['evaluations_per_day']:.2f}\n")

        # Chains
        if usage['chains']:
            c = usage['chains']
            md.append("### Chain Executions\n")
            md.append(f"- **Total Executions:** {c['total_executions']}")
            md.append(f"- **Success Rate:** {c['success_rate']:.2f}%")
            md.append(f"- **Executions/Day:** {c['executions_per_day']:.2f}\n")

        return '\n'.join(md)

    def _markdown_quality_section(self, quality: Dict) -> str:
        """Generate quality section in Markdown"""
        md = []
        md.append("## Quality Metrics\n")

        # Summary
        summary = quality['summary']
        md.append("### Summary\n")
        md.append(f"- **Prompts Evaluated:** {summary['total_prompts_evaluated']}")
        md.append(f"- **Avg Top Score:** {summary['avg_top_score']:.3f}")
        md.append(f"- **Declining Prompts:** {summary['declining_count']}")
        md.append(f"- **Active Alerts:** {summary['active_alerts']}\n")

        # Top performing
        if quality['top_performing']:
            md.append("### Top Performing Prompts\n")
            md.append("| Prompt | Branch | Avg Score | Samples |")
            md.append("|--------|--------|-----------|---------|")
            for p in quality['top_performing']:
                md.append(f"| {p['prompt_name']} | {p['branch']} | {p['avg_score']:.3f} | {p['measurements']} |")
            md.append("")

        # Declining
        if quality['declining']:
            md.append("### Declining Quality ⚠️\n")
            md.append("| Prompt | Branch | Trend | Avg Score |")
            md.append("|--------|--------|-------|-----------|")
            for p in quality['declining']:
                md.append(f"| {p['prompt_name']} | {p['branch']} | {p['trend_slope']:.4f} | {p['avg_score']:.3f} |")
            md.append("")

        # Active alerts
        if quality['alerts']:
            md.append("### Active Alerts 🚨\n")
            md.append("| Prompt | Type | Message |")
            md.append("|--------|------|---------|")
            for a in quality['alerts'][:10]:
                md.append(f"| {a['prompt_name']} | {a['alert_type']} | {a['message']} |")
            md.append("")

        # By evaluator
        if quality['by_evaluator']:
            md.append("### Quality by Evaluator\n")
            md.append("| Evaluator | Measurements | Avg Score |")
            md.append("|-----------|--------------|-----------|")
            for e in quality['by_evaluator']:
                md.append(f"| {e['evaluator_type']} | {e['measurements']} | {e['avg_score']:.3f} |")
            md.append("")

        return '\n'.join(md)

    # ========================================================================
    # HTML Generation
    # ========================================================================

    def _generate_html_report(self, data: Dict) -> str:
        """Generate HTML format report"""
        # Reuse dashboard template but with report-specific data
        html = self._get_report_html_template()

        html = html.replace('{{TITLE}}', data['title'])
        html = html.replace('{{GENERATED_AT}}', data['generated_at'])
        html = html.replace('{{PERIOD}}', f"{data['period']['start']} to {data['period']['end']}")
        html = html.replace('{{SECTIONS}}', json.dumps(data['sections']))

        return html

    def _get_report_html_template(self) -> str:
        """Get HTML report template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{TITLE}}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        h2 { color: #007bff; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; }
        .meta { color: #666; margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{TITLE}}</h1>
        <div class="meta">
            <p><strong>Generated:</strong> {{GENERATED_AT}}</p>
            <p><strong>Period:</strong> {{PERIOD}}</p>
        </div>
        <div id="content"></div>
    </div>
    <script>
        const sections = {{SECTIONS}};
        // Render sections (simplified)
        document.getElementById('content').innerHTML = '<pre>' + JSON.stringify(sections, null, 2) + '</pre>';
    </script>
</body>
</html>
"""

    # ========================================================================
    # Text Generation
    # ========================================================================

    def _generate_text_report(self, data: Dict) -> str:
        """Generate plain text format report"""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(data['title'].center(80))
        lines.append("=" * 80)
        lines.append(f"Generated: {data['generated_at']}")
        lines.append(f"Period: {data['period']['start']} to {data['period']['end']} ({data['period']['days']} days)")
        lines.append("=" * 80)
        lines.append("")

        # Sections
        if 'performance' in data['sections']:
            lines.append(self._text_performance_section(data['sections']['performance']))

        if 'usage' in data['sections']:
            lines.append(self._text_usage_section(data['sections']['usage']))

        if 'quality' in data['sections']:
            lines.append(self._text_quality_section(data['sections']['quality']))

        return '\n'.join(lines)

    def _text_performance_section(self, perf: Dict) -> str:
        """Generate performance section in plain text"""
        lines = []
        lines.append("\nPERFORMANCE")
        lines.append("-" * 80)

        summary = perf['summary']
        lines.append(f"Total Operations: {summary['total_operations']}")
        lines.append(f"Avg Duration: {summary['avg_duration_ms']:.2f} ms")

        if perf['resources']:
            res = perf['resources']
            lines.append(f"\nResource Usage:")
            lines.append(f"  CPU: {res.get('avg_cpu_percent', 0):.2f}% avg, {res.get('max_cpu_percent', 0):.2f}% max")
            lines.append(f"  Memory: {res.get('avg_memory_mb', 0):.2f} MB avg, {res.get('max_memory_mb', 0):.2f} MB max")

        return '\n'.join(lines)

    def _text_usage_section(self, usage: Dict) -> str:
        """Generate usage section in plain text"""
        lines = []
        lines.append("\nUSAGE ANALYTICS")
        lines.append("-" * 80)

        summary = usage['summary']
        lines.append(f"Total Events: {summary['total_events']}")
        lines.append(f"Active Branches: {summary['active_branches']}")
        lines.append(f"Active Users: {summary['active_users']}")

        if usage['most_used_prompts']:
            lines.append(f"\nMost Used Prompts:")
            for p in usage['most_used_prompts'][:5]:
                lines.append(f"  - {p['prompt_name']} ({p['branch']}): {p['access_count']} accesses")

        return '\n'.join(lines)

    def _text_quality_section(self, quality: Dict) -> str:
        """Generate quality section in plain text"""
        lines = []
        lines.append("\nQUALITY METRICS")
        lines.append("-" * 80)

        summary = quality['summary']
        lines.append(f"Prompts Evaluated: {summary['total_prompts_evaluated']}")
        lines.append(f"Avg Top Score: {summary['avg_top_score']:.3f}")
        lines.append(f"Declining Prompts: {summary['declining_count']}")
        lines.append(f"Active Alerts: {summary['active_alerts']}")

        if quality['top_performing']:
            lines.append(f"\nTop Performing:")
            for p in quality['top_performing'][:5]:
                lines.append(f"  - {p['prompt_name']}: {p['avg_score']:.3f}")

        return '\n'.join(lines)
