# Promptly Analytics - Complete Guide

Comprehensive observability and monitoring system for Promptly, providing performance tracking, usage analytics, quality metrics, visualization, and reporting.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [CLI Reference](#cli-reference)
- [Integrations](#integrations)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Promptly Analytics provides comprehensive monitoring and observability for your prompt management workflow:

- **Performance Monitoring**: Track operation timing, resource utilization, and identify bottlenecks
- **Usage Analytics**: Understand prompt access patterns, popular prompts, and user activity
- **Quality Metrics**: Monitor prompt quality over time, compare versions, and detect regressions
- **Visualization**: Terminal charts, HTML dashboards, and data exports
- **Reporting**: Automated daily/weekly/monthly reports in multiple formats
- **Integrations**: Prometheus, OpenTelemetry, Grafana, and structured logging

## Quick Start

### Basic Usage

```python
from promptly import Promptly
from promptly.analytics import enable_analytics

# Enable analytics for your Promptly instance
promptly = Promptly()
promptly = enable_analytics(promptly)

# Use Promptly normally - all operations are automatically tracked
promptly.add('greeting', 'Hello {name}!')
promptly.get('greeting')
promptly.eval_prompt('greeting', test_cases)

# View statistics
perf_stats = promptly.get_performance_stats()
usage_stats = promptly.get_usage_stats()
quality_stats = promptly.get_quality_stats()
```

### CLI Usage

```bash
# View performance statistics
python -m promptly.analytics.cli stats performance

# View usage analytics
python -m promptly.analytics.cli stats usage --days 30

# Generate daily report
python -m promptly.analytics.cli report daily --output ./report.md

# Generate HTML dashboard
python -m promptly.analytics.cli report dashboard --output ./dashboard.html

# Export data to JSON
python -m promptly.analytics.cli export json --output ./analytics.json
```

## Installation

### Required Dependencies

```bash
pip install promptly click pyyaml psutil
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# Terminal charts
pip install plotext

# Prometheus integration
pip install prometheus_client

# OpenTelemetry integration
pip install opentelemetry-api opentelemetry-sdk

# Report generation
pip install markdown
```

### All Dependencies

```bash
pip install promptly[analytics]
```

## Core Components

### 1. Performance Monitoring

Tracks operation timing, resource usage, and system performance.

```python
from promptly.analytics import PerformanceMonitor

# Initialize monitor
monitor = PerformanceMonitor(
    db_path='.promptly/analytics/performance.db',
    retention_days=30,
    enable_sampling=True  # Sample CPU/memory every 60s
)

# Time an operation
with monitor.time_operation('my_operation', metadata={'key': 'value'}):
    # Do work
    pass

# Get statistics
stats = monitor.get_operation_stats()
resource_stats = monitor.get_resource_stats(hours=24)
throughput = monitor.get_throughput(hours=1)

# Find slow operations
slow_ops = monitor.get_slow_operations(threshold_ms=1000)

# Calculate percentiles
percentiles = monitor.get_operation_percentiles('get_prompt')
```

**Key Features:**
- Operation timing with microsecond precision
- CPU and memory usage tracking
- Throughput calculation (ops/sec, ops/min, ops/hour)
- Percentile analysis (p50, p75, p90, p95, p99)
- Background resource sampling
- Automatic data retention management

### 2. Usage Analytics

Tracks access patterns, popular prompts, and user activity.

```python
from promptly.analytics import UsageAnalytics

# Initialize analytics
usage = UsageAnalytics(
    db_path='.promptly/analytics/usage.db',
    retention_days=90
)

# Track events
usage.tracker.track_prompt_access('greeting', 'main', 'get', user='alice')
usage.tracker.track_evaluation('greeting', 'main', test_count=10, avg_score=0.95)
usage.tracker.track_chain_execution('my_chain', step_count=5, success=True)

# Get insights
most_used = usage.get_most_used_prompts(days=7)
branch_activity = usage.get_branch_activity(days=30)
user_activity = usage.get_user_activity(days=30)
hourly_dist = usage.get_hourly_distribution(days=7)
```

**Key Features:**
- Prompt access tracking (get, add, update, delete)
- Evaluation execution tracking
- Chain execution tracking
- Branch activity monitoring
- User activity tracking
- Hourly/daily activity patterns
- Event type distribution

### 3. Quality Metrics

Monitors prompt quality, tracks trends, and detects regressions.

```python
from promptly.analytics import QualityMetrics

# Initialize metrics
quality = QualityMetrics(
    db_path='.promptly/analytics/quality.db',
    retention_days=180
)

# Record evaluations
quality.tracker.record_evaluation(
    prompt_name='greeting',
    version=1,
    branch='main',
    score=0.95,
    evaluator_type='semantic',
    test_case_id='test_1'
)

# Analyze quality
stats = quality.get_prompt_quality_stats('greeting', days=30)
trend = quality.get_quality_trend('greeting', days=30)
top_prompts = quality.get_top_performing_prompts(days=30)
declining = quality.get_declining_prompts(days=30)

# Compare versions
comparison = quality.compare_versions('greeting', version_a=1, version_b=2)

# Run A/B test
ab_result = quality.run_ab_test('greeting', version_a=1, version_b=2, days=7)

# Get alerts
alerts = quality.get_active_alerts()
```

**Key Features:**
- Quality score tracking over time
- Trend analysis (improving/declining/stable)
- Version comparison
- A/B testing with statistical significance
- Score distribution analysis
- Automatic quality alerts
- Top/bottom performer identification

### 4. Visualization

Generate charts, dashboards, and exports.

```python
from promptly.analytics import Visualizer

# Initialize visualizer
viz = Visualizer(
    performance_monitor=monitor,
    usage_analytics=usage,
    quality_metrics=quality
)

# Terminal charts (requires plotext)
viz.plot_operation_times(operation='get_prompt', days=7)
viz.plot_throughput(hours=24)
viz.plot_resource_usage(hours=24)
viz.plot_quality_trend('greeting', days=30)
viz.plot_score_distribution('greeting', days=30)

# HTML dashboard
dashboard_path = viz.generate_dashboard('./dashboard.html', days=7)

# Export data
viz.export_to_csv('performance', './performance.csv', days=30)
viz.export_to_json('./analytics.json', days=30)
```

**Key Features:**
- Terminal-based charts (plotext)
- Interactive HTML dashboards
- CSV/JSON exports
- Time-series plots
- Distribution histograms
- Heatmaps

### 5. Reporting

Generate comprehensive reports in multiple formats.

```python
from promptly.analytics import ReportGenerator, ReportConfig, ReportPeriod, ReportFormat

# Initialize generator
generator = ReportGenerator(
    performance_monitor=monitor,
    usage_analytics=usage,
    quality_metrics=quality,
    visualizer=viz
)

# Generate daily report
config = ReportConfig(
    period=ReportPeriod.DAILY,
    format=ReportFormat.MARKDOWN,
    include_performance=True,
    include_usage=True,
    include_quality=True
)
report_path = generator.generate_report(config, './daily_report.md')

# Convenience methods
generator.generate_daily_summary('./daily.md')
generator.generate_weekly_summary('./weekly.md')
generator.generate_monthly_summary('./monthly.md')

# Specialized reports
generator.generate_performance_report('./performance.md', days=7)
generator.generate_quality_report('./quality.md', days=30)
```

**Available Formats:**
- **Markdown**: Clean, readable reports with tables
- **HTML**: Rich reports with embedded styles
- **JSON**: Machine-readable data export
- **Text**: Plain text for email/logging

### 6. Integrations

Connect to external monitoring systems.

#### Prometheus

```python
from promptly.analytics import PrometheusExporter

# Initialize exporter
prometheus = PrometheusExporter(port=9090)
prometheus.start_http_server()

# Record metrics
prometheus.record_operation('get_prompt', duration_ms=12.5, status='success')
prometheus.record_quality_score('greeting', 'semantic', score=0.95)
prometheus.record_prompt_access('greeting', 'main')

# Export to file (for node_exporter textfile collector)
prometheus.export_to_file('/var/lib/node_exporter/promptly.prom')
```

**Metrics Exported:**
- `promptly_operations_total` - Counter of operations by type and status
- `promptly_operation_duration_seconds` - Histogram of operation durations
- `promptly_quality_score` - Histogram of quality scores
- `promptly_prompt_accesses_total` - Counter of prompt accesses
- `promptly_evaluations_total` - Counter of evaluations
- `promptly_chain_executions_total` - Counter of chain executions
- `promptly_cpu_percent` - Gauge of CPU usage
- `promptly_memory_mb` - Gauge of memory usage

#### OpenTelemetry

```python
from promptly.analytics import OpenTelemetryIntegration

# Initialize integration
otel = OpenTelemetryIntegration(
    service_name='promptly',
    enable_tracing=True,
    enable_metrics=True
)

# Trace operations
with otel.trace_operation('get_prompt'):
    # Do work
    pass

# Record metrics
otel.record_operation('get_prompt', duration_ms=12.5, attributes={'branch': 'main'})
otel.record_quality_score('greeting', score=0.95)
```

#### Grafana

```python
from promptly.analytics import GrafanaExporter

# Initialize exporter
grafana = GrafanaExporter(output_dir='./grafana')

# Export dashboard
dashboard_path = grafana.export_dashboard('promptly-overview')
config_path = grafana.export_prometheus_config()
```

#### Structured Logging

```python
from promptly.analytics import StructuredLogger

# Initialize logger
logger = StructuredLogger(log_path='./logs/analytics.jsonl')

# Log events
logger.log_operation('get_prompt', duration_ms=12.5, status='success')
logger.log_quality_score('greeting', version=1, score=0.95, evaluator='semantic')
logger.log_usage('greeting', action='access', user='alice')
logger.log_error('validation_error', 'Invalid prompt format', metadata={'prompt': 'bad'})
```

#### Analytics Hub

Unified interface for all integrations:

```python
from promptly.analytics import AnalyticsHub

# Initialize hub with config
config = {
    'prometheus': {'enabled': True, 'port': 9090, 'start_server': True},
    'opentelemetry': {'enabled': True, 'service_name': 'promptly'},
    'grafana': {'enabled': True, 'output_dir': './grafana'},
    'logging': {'enabled': True, 'path': './logs/analytics.jsonl'}
}

hub = AnalyticsHub(config)

# Record events - automatically sent to all enabled integrations
hub.record_operation('get_prompt', duration_ms=12.5, status='success')
hub.record_quality_score('greeting', version=1, score=0.95, evaluator='semantic')
hub.record_usage('greeting', action='access', user='alice')

# Export configurations
hub.export_grafana_dashboard()
hub.export_prometheus_config()
```

## Usage Examples

### Example 1: Complete Analytics Setup

```python
from promptly import Promptly
from promptly.analytics import enable_analytics

# Configure analytics
config = {
    'retention_days': 30,
    'enable_resource_sampling': True,
    'integrations': {
        'prometheus': {'enabled': True, 'port': 9090},
        'logging': {'enabled': True, 'path': './logs/analytics.jsonl'}
    }
}

# Enable analytics
promptly = Promptly()
promptly = enable_analytics(promptly, config=config)

# Use normally
promptly.add('greeting', 'Hello {name}!', user='alice')
promptly.get('greeting', user='bob')
promptly.eval_prompt('greeting', test_cases, user='alice')

# View real-time stats
print(promptly.get_performance_stats())
print(promptly.get_usage_stats(days=7))
print(promptly.get_quality_stats(days=30))

# Cleanup old data periodically
promptly.cleanup_old_data()
```

### Example 2: Custom Operation Tracking

```python
from promptly.analytics import enable_analytics, track_operation

promptly = Promptly()
promptly = enable_analytics(promptly)

# Track custom operations
@track_operation('custom_analysis', promptly)
def analyze_prompts():
    prompts = promptly.list_prompts()
    # Analysis logic
    return len(prompts)

result = analyze_prompts()
```

### Example 3: Automated Daily Reports

```python
from promptly.analytics import ReportGenerator, ReportConfig, ReportPeriod, ReportFormat
from datetime import datetime
import schedule

def generate_daily_report():
    """Generate and email daily report"""
    generator = ReportGenerator(
        performance_monitor=promptly.performance,
        usage_analytics=promptly.usage,
        quality_metrics=promptly.quality
    )

    date_str = datetime.now().strftime('%Y-%m-%d')
    output_path = f'./reports/daily_{date_str}.md'

    config = ReportConfig(
        period=ReportPeriod.DAILY,
        format=ReportFormat.MARKDOWN
    )

    report_path = generator.generate_report(config, output_path)

    # Email report (pseudo-code)
    # send_email(to='team@example.com', attachment=report_path)

    print(f"Daily report generated: {report_path}")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(generate_daily_report)
```

### Example 4: Quality Monitoring and Alerting

```python
from promptly.analytics import QualityMetrics

quality = QualityMetrics('.promptly/analytics/quality.db')

def check_quality_alerts():
    """Check for quality issues and send alerts"""
    # Get active alerts
    alerts = quality.get_active_alerts(severity='error')

    if alerts:
        for alert in alerts:
            print(f"⚠️  ALERT: {alert['prompt_name']}")
            print(f"   Type: {alert['alert_type']}")
            print(f"   Message: {alert['message']}")

    # Check for declining prompts
    declining = quality.get_declining_prompts(threshold_slope=-0.05, days=7)

    if declining:
        print("\n📉 Prompts with declining quality:")
        for p in declining:
            print(f"   - {p['prompt_name']}: {p['trend_slope']:.4f}")

# Run periodically
check_quality_alerts()
```

### Example 5: A/B Testing

```python
from promptly.analytics import QualityMetrics

quality = QualityMetrics('.promptly/analytics/quality.db')

# Run A/B test between two versions
result = quality.run_ab_test(
    prompt_name='greeting',
    version_a=1,
    version_b=2,
    branch='main',
    days=7,
    confidence_level=0.95
)

print(f"A/B Test Results:")
print(f"  Version A: {result.avg_score_a:.3f} ({result.samples_a} samples)")
print(f"  Version B: {result.avg_score_b:.3f} ({result.samples_b} samples)")
print(f"  Difference: {result.score_diff_percent:.2f}%")
print(f"  P-value: {result.statistical_significance:.4f}")
print(f"  Winner: Version {result.winner}" if result.winner else "  No significant difference")
print(f"  Confidence: {result.confidence * 100:.1f}%")
```

## CLI Reference

### Stats Commands

```bash
# Performance statistics
python -m promptly.analytics.cli stats performance
python -m promptly.analytics.cli stats performance --operation get_prompt

# Usage statistics
python -m promptly.analytics.cli stats usage --days 30

# Quality statistics
python -m promptly.analytics.cli stats quality --days 30
python -m promptly.analytics.cli stats quality --prompt greeting
```

### Report Commands

```bash
# Daily report
python -m promptly.analytics.cli report daily --output ./report.md --format markdown

# Weekly report
python -m promptly.analytics.cli report weekly --output ./report.html --format html

# HTML dashboard
python -m promptly.analytics.cli report dashboard --output ./dashboard.html --days 7
```

### Export Commands

```bash
# Export to CSV
python -m promptly.analytics.cli export csv performance --output ./perf.csv --days 30
python -m promptly.analytics.cli export csv usage --output ./usage.csv --days 30
python -m promptly.analytics.cli export csv quality --output ./quality.csv --days 30

# Export to JSON
python -m promptly.analytics.cli export json --output ./analytics.json --days 30

# Export Grafana dashboard
python -m promptly.analytics.cli export grafana --output-dir ./grafana
```

### Cleanup Commands

```bash
# Cleanup old data
python -m promptly.analytics.cli cleanup
python -m promptly.analytics.cli cleanup --no-usage  # Skip usage data
```

## Configuration

### Analytics Configuration

```python
config = {
    # Data retention (days)
    'retention_days': 30,

    # Resource sampling
    'enable_resource_sampling': True,
    'sample_interval_seconds': 60,

    # Integrations
    'integrations': {
        'prometheus': {
            'enabled': True,
            'port': 9090,
            'start_server': True
        },
        'opentelemetry': {
            'enabled': True,
            'service_name': 'promptly',
            'enable_tracing': True,
            'enable_metrics': True
        },
        'grafana': {
            'enabled': True,
            'output_dir': './grafana'
        },
        'logging': {
            'enabled': True,
            'path': './logs/analytics.jsonl'
        }
    }
}
```

### Environment Variables

```bash
# Analytics database paths
export PROMPTLY_ANALYTICS_PERF_DB=".promptly/analytics/performance.db"
export PROMPTLY_ANALYTICS_USAGE_DB=".promptly/analytics/usage.db"
export PROMPTLY_ANALYTICS_QUALITY_DB=".promptly/analytics/quality.db"

# Retention periods
export PROMPTLY_ANALYTICS_RETENTION_DAYS=30

# Prometheus
export PROMPTLY_PROMETHEUS_PORT=9090
export PROMPTLY_PROMETHEUS_ENABLED=true

# Logging
export PROMPTLY_ANALYTICS_LOG_PATH="./logs/analytics.jsonl"
```

## Best Practices

### 1. Data Retention

- **Performance**: 30 days is usually sufficient
- **Usage**: 90 days for trend analysis
- **Quality**: 180 days to track long-term improvements

### 2. Resource Sampling

- Enable for production systems to track resource usage
- Adjust `sample_interval_seconds` based on your needs (default: 60s)
- Disable if running in resource-constrained environments

### 3. Regular Cleanup

```python
# Schedule weekly cleanup
import schedule

def weekly_cleanup():
    promptly.cleanup_old_data()

schedule.every().sunday.at("02:00").do(weekly_cleanup)
```

### 4. Monitoring Quality Trends

```python
# Daily quality check
def daily_quality_check():
    declining = promptly.quality.get_declining_prompts(days=7)
    if declining:
        # Send alert
        alert_team(f"Found {len(declining)} declining prompts")
```

### 5. Performance Optimization

- Use `get_operation_stats()` to identify slow operations
- Monitor percentiles (p95, p99) for latency-sensitive operations
- Set up alerts for operations exceeding thresholds

### 6. Integrations

- Use **Prometheus** for metrics if you have existing Prometheus infrastructure
- Use **OpenTelemetry** for distributed tracing in microservices
- Use **Structured Logging** for debugging and auditing
- Use **Grafana** for rich dashboards and visualization

## Troubleshooting

### Issue: High Memory Usage

**Solution**: Reduce retention periods or disable resource sampling:

```python
config = {
    'retention_days': 7,  # Shorter retention
    'enable_resource_sampling': False  # Disable sampling
}
```

### Issue: Slow Operations

**Solution**: Check database size and run cleanup:

```bash
# Check database sizes
ls -lh .promptly/analytics/*.db

# Run cleanup
python -m promptly.analytics.cli cleanup
```

### Issue: Missing Dependencies

**Solution**: Install optional dependencies:

```bash
pip install plotext prometheus_client opentelemetry-api opentelemetry-sdk
```

### Issue: Prometheus Metrics Not Appearing

**Solution**: Verify server is running and check port:

```python
# Check if server started
prometheus = PrometheusExporter(port=9090)
prometheus.start_http_server()

# Test endpoint
curl http://localhost:9090/metrics
```

### Issue: Dashboard Not Generating

**Solution**: Ensure visualizer has access to all analytics modules:

```python
viz = Visualizer(
    performance_monitor=promptly.performance,
    usage_analytics=promptly.usage,
    quality_metrics=promptly.quality
)
```

## Performance Considerations

### Database Size

Average database sizes (approximate):

- **Performance DB**: ~1 MB per 10,000 operations
- **Usage DB**: ~500 KB per 10,000 events
- **Quality DB**: ~2 MB per 10,000 measurements

### Memory Usage

- **Base overhead**: ~10-20 MB
- **Resource sampling**: ~5 MB additional
- **In-memory buffers**: ~2 MB (stores last 1000 operations)

### CPU Usage

- **Operation tracking**: <1% overhead
- **Resource sampling**: <0.5% overhead (when enabled)
- **Report generation**: Spike during generation, negligible otherwise

## License

Promptly Analytics is part of the Promptly project and follows the same license.

## Support

For issues, questions, or feature requests:

- Open an issue on GitHub
- Check the documentation
- Review examples in this guide

---

**Version**: 0.1.0
**Last Updated**: 2025-01-17
