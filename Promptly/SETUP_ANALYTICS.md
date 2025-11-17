# Promptly Analytics - Setup Guide

Quick setup guide to get started with Promptly Analytics.

## Quick Install

### 1. Install Promptly

```bash
cd Promptly/promptly
pip install -e .
```

### 2. Install Analytics Dependencies

**Required dependencies:**
```bash
pip install click pyyaml psutil
```

**Optional dependencies (recommended):**
```bash
# Terminal charts
pip install plotext

# Prometheus integration
pip install prometheus_client

# OpenTelemetry integration
pip install opentelemetry-api opentelemetry-sdk
```

**All at once:**
```bash
pip install -r promptly/analytics/requirements.txt
```

## Quick Start

### Option 1: Simple Wrapper (Recommended)

```python
from promptly import Promptly
from promptly.analytics import enable_analytics

# Initialize Promptly
promptly = Promptly()
promptly.init()

# Enable analytics (all operations automatically tracked)
promptly = enable_analytics(promptly)

# Use normally - analytics happen automatically
promptly.add('greeting', 'Hello {name}!')
promptly.get('greeting')

# View stats
print(promptly.get_performance_stats())
```

### Option 2: Manual Integration

```python
from promptly import Promptly
from promptly.analytics import PerformanceMonitor, UsageAnalytics, QualityMetrics

# Initialize analytics components
performance = PerformanceMonitor('.promptly/analytics/performance.db')
usage = UsageAnalytics('.promptly/analytics/usage.db')
quality = QualityMetrics('.promptly/analytics/quality.db')

# Use in your code
with performance.time_operation('my_operation'):
    # Do work
    pass

# Track usage
usage.tracker.track_prompt_access('greeting', 'main', 'get')

# Record quality
quality.tracker.record_evaluation('greeting', 1, 'main', 0.95, 'semantic', 'test_1')
```

## Using the CLI

### View Statistics

```bash
# Performance stats
python -m promptly.analytics.cli stats performance

# Usage stats
python -m promptly.analytics.cli stats usage --days 30

# Quality stats
python -m promptly.analytics.cli stats quality --prompt greeting
```

### Generate Reports

```bash
# Daily summary
python -m promptly.analytics.cli report daily --output ./report.md

# HTML dashboard
python -m promptly.analytics.cli report dashboard --output ./dashboard.html

# Weekly report
python -m promptly.analytics.cli report weekly --output ./weekly.md --format html
```

### Export Data

```bash
# Export to JSON
python -m promptly.analytics.cli export json --output ./analytics.json

# Export to CSV
python -m promptly.analytics.cli export csv performance --output ./perf.csv

# Export Grafana dashboard
python -m promptly.analytics.cli export grafana --output-dir ./grafana
```

## Running the Example

```bash
cd Promptly
python examples/analytics_example.py
```

This will:
1. Initialize Promptly with analytics
2. Generate sample data (prompts, evaluations, chains)
3. Display statistics
4. Generate visualizations and reports
5. Export data to various formats

## Directory Structure

After running analytics, you'll see:

```
.promptly/
└── analytics/
    ├── performance.db  # Performance metrics
    ├── usage.db        # Usage analytics
    └── quality.db      # Quality metrics

reports/
├── dashboard.html      # Interactive dashboard
├── daily_report.md     # Daily summary
└── weekly_report.html  # Weekly summary

exports/
├── analytics.json      # Complete data export
├── performance.csv     # Performance data
├── usage.csv          # Usage data
└── quality.csv        # Quality data

logs/
└── analytics.jsonl    # Structured logs

grafana/
├── promptly-overview.json  # Grafana dashboard
└── prometheus.yml         # Prometheus config
```

## Configuration

### Basic Configuration

```python
config = {
    'retention_days': 30,
    'enable_resource_sampling': True,
}

promptly = enable_analytics(promptly, config=config)
```

### Full Configuration with Integrations

```python
config = {
    'retention_days': 30,
    'enable_resource_sampling': True,
    'sample_interval_seconds': 60,
    'integrations': {
        'prometheus': {
            'enabled': True,
            'port': 9090,
            'start_server': True
        },
        'opentelemetry': {
            'enabled': True,
            'service_name': 'promptly'
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

promptly = enable_analytics(promptly, config=config)
```

## Prometheus Setup

### 1. Start Prometheus Exporter

```python
from promptly.analytics import enable_analytics

config = {
    'integrations': {
        'prometheus': {
            'enabled': True,
            'port': 9090,
            'start_server': True
        }
    }
}

promptly = enable_analytics(promptly, config=config)
```

### 2. Configure Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'promptly'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

### 3. View Metrics

Visit: http://localhost:9090/metrics

## Grafana Setup

### 1. Export Dashboard

```bash
python -m promptly.analytics.cli export grafana --output-dir ./grafana
```

### 2. Import to Grafana

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `grafana/promptly-overview.json`
4. Select Prometheus data source
5. Click Import

### 3. View Dashboard

The dashboard includes:
- Operations per second
- Operation duration (p95)
- CPU usage
- Memory usage
- Quality scores

## Common Use Cases

### 1. Monitor Production Performance

```python
from promptly.analytics import enable_analytics

# Enable with minimal overhead
config = {
    'retention_days': 7,
    'enable_resource_sampling': False,  # Disable in production
    'integrations': {
        'prometheus': {'enabled': True, 'port': 9090}
    }
}

promptly = enable_analytics(promptly, config=config)
```

### 2. Track Quality Improvements

```python
from promptly.analytics import QualityMetrics

quality = QualityMetrics('.promptly/analytics/quality.db')

# Check trend
trend = quality.get_quality_trend('my_prompt', days=30)
print(f"Trend: {trend.trend_direction}")
print(f"Slope: {trend.trend_slope}")

# Compare versions
comparison = quality.compare_versions('my_prompt', version_a=1, version_b=2)
print(f"Version 2 is {comparison['comparison']['score_diff_percent']:.1f}% better")
```

### 3. Generate Weekly Reports

```python
from promptly.analytics import ReportGenerator, ReportConfig, ReportPeriod, ReportFormat
import schedule

def weekly_report():
    generator = ReportGenerator(
        performance_monitor=promptly.performance,
        usage_analytics=promptly.usage,
        quality_metrics=promptly.quality
    )

    config = ReportConfig(
        period=ReportPeriod.WEEKLY,
        format=ReportFormat.MARKDOWN
    )

    report_path = generator.generate_report(config, './reports/weekly.md')
    # Email or post to Slack
    print(f"Report: {report_path}")

# Run every Monday at 9 AM
schedule.every().monday.at("09:00").do(weekly_report)
```

## Troubleshooting

### Issue: Import Error

```python
# Error: No module named 'promptly.analytics'
```

**Solution**: Ensure you're in the correct directory and Promptly is installed:

```bash
cd Promptly/promptly
pip install -e .
```

### Issue: Database Locked

```python
# Error: database is locked
```

**Solution**: Close other connections or use timeout:

```python
import sqlite3
conn = sqlite3.connect('database.db', timeout=10.0)
```

### Issue: plotext Not Found

```
Warning: plotext not installed. Terminal charts will be unavailable.
```

**Solution**: Install plotext:

```bash
pip install plotext
```

### Issue: High Disk Usage

**Solution**: Reduce retention or run cleanup:

```bash
python -m promptly.analytics.cli cleanup
```

Or configure shorter retention:

```python
config = {'retention_days': 7}  # Shorter retention
```

## Next Steps

1. **Read the full documentation**: See `ANALYTICS.md`
2. **Run the example**: `python examples/analytics_example.py`
3. **Try the CLI**: `python -m promptly.analytics.cli stats performance`
4. **Set up monitoring**: Configure Prometheus and Grafana
5. **Automate reports**: Schedule daily/weekly reports

## Resources

- **Full Documentation**: `ANALYTICS.md`
- **Example Code**: `examples/analytics_example.py`
- **CLI Reference**: `python -m promptly.analytics.cli --help`
- **API Reference**: See docstrings in source code

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the full documentation in `ANALYTICS.md`
3. Open an issue on GitHub

---

**Last Updated**: 2025-01-17
