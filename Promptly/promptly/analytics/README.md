# Promptly Analytics

Comprehensive observability and monitoring system for Promptly.

## Features

- **Performance Monitoring**: Track operation timing, resource utilization, and identify bottlenecks
- **Usage Analytics**: Understand prompt access patterns, popular prompts, and user activity
- **Quality Metrics**: Monitor prompt quality over time, compare versions, and detect regressions
- **Visualization**: Terminal charts, HTML dashboards, and data exports
- **Reporting**: Automated daily/weekly/monthly reports in multiple formats
- **Integrations**: Prometheus, OpenTelemetry, Grafana, and structured logging

## Quick Start

```python
from promptly import Promptly
from promptly.analytics import enable_analytics

# Enable analytics
promptly = Promptly()
promptly = enable_analytics(promptly)

# Use normally - all operations are tracked
promptly.add('greeting', 'Hello {name}!')

# View statistics
print(promptly.get_performance_stats())
print(promptly.get_usage_stats())
print(promptly.get_quality_stats())
```

## Installation

```bash
# Required
pip install click pyyaml psutil

# Optional (recommended)
pip install plotext prometheus_client opentelemetry-api opentelemetry-sdk

# All dependencies
pip install -r promptly/analytics/requirements.txt
```

## Modules

### Core Analytics
- **performance.py**: Performance monitoring and resource tracking
- **usage.py**: Usage analytics and access patterns
- **quality.py**: Quality metrics and trend analysis

### Visualization & Reporting
- **visualize.py**: Charts, dashboards, and data exports
- **reports.py**: Report generation in multiple formats

### Integration
- **integrations.py**: Prometheus, OpenTelemetry, Grafana, logging
- **instrumentation.py**: Wrapper for automatic tracking

### CLI
- **cli.py**: Command-line interface for analytics

## CLI Usage

```bash
# View statistics
python -m promptly.analytics.cli stats performance
python -m promptly.analytics.cli stats usage --days 30
python -m promptly.analytics.cli stats quality --prompt greeting

# Generate reports
python -m promptly.analytics.cli report daily --output ./report.md
python -m promptly.analytics.cli report dashboard --output ./dashboard.html

# Export data
python -m promptly.analytics.cli export json --output ./analytics.json
python -m promptly.analytics.cli export grafana --output-dir ./grafana

# Cleanup
python -m promptly.analytics.cli cleanup
```

## Documentation

- **Complete Guide**: See `/Promptly/ANALYTICS.md`
- **Setup Guide**: See `/Promptly/SETUP_ANALYTICS.md`
- **Example**: See `/Promptly/examples/analytics_example.py`

## Architecture

```
promptly/analytics/
├── __init__.py           # Package exports
├── performance.py        # Performance monitoring
├── usage.py             # Usage analytics
├── quality.py           # Quality metrics
├── visualize.py         # Visualization
├── reports.py           # Report generation
├── integrations.py      # External integrations
├── instrumentation.py   # Auto-tracking wrapper
├── cli.py              # Command-line interface
└── requirements.txt     # Dependencies
```

## Configuration

```python
config = {
    'retention_days': 30,
    'enable_resource_sampling': True,
    'integrations': {
        'prometheus': {'enabled': True, 'port': 9090},
        'logging': {'enabled': True, 'path': './logs/analytics.jsonl'}
    }
}

promptly = enable_analytics(promptly, config=config)
```

## Example Output

### Performance Stats
```
Operation            Count   Avg (ms)   Max (ms)
--------------------------------------------------
get_prompt             156      12.34      45.67
add_prompt              23      23.45      67.89
eval_prompt             12     123.45     234.56
```

### Usage Stats
```
Most Used Prompts:
  - greeting: 42 accesses
  - farewell: 28 accesses
  - summary: 19 accesses

Evaluation Stats:
  Total Evaluations: 12
  Average Score: 0.923
```

### Quality Stats
```
Top Performing Prompts:
  - greeting: 0.950 (n=25)
  - summary: 0.923 (n=18)
  - farewell: 0.901 (n=22)

Declining Quality:
  ⚠️  old_prompt: -0.0234 slope
```

## Integrations

### Prometheus

```python
from promptly.analytics import PrometheusExporter

prometheus = PrometheusExporter(port=9090)
prometheus.start_http_server()
```

Metrics: http://localhost:9090/metrics

### Grafana

```bash
python -m promptly.analytics.cli export grafana --output-dir ./grafana
```

Import `grafana/promptly-overview.json` to Grafana.

### OpenTelemetry

```python
from promptly.analytics import OpenTelemetryIntegration

otel = OpenTelemetryIntegration(service_name='promptly')

with otel.trace_operation('my_operation'):
    # Do work
    pass
```

## License

Part of the Promptly project. See main LICENSE file.

## Version

0.1.0

---

For detailed documentation, see **ANALYTICS.md** in the repository root.
