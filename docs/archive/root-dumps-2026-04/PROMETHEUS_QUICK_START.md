# Prometheus Metrics - Quick Start Guide

Get HoloLoom metrics to Prometheus and Grafana in 5 minutes.

## 1. Install prometheus_client

```bash
pip install prometheus-client
```

## 2. Start HoloLoom with Metrics

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.performance.prometheus_metrics import start_metrics_server
from HoloLoom.documentation.types import Query

# Create config
config = Config.fused()

# Create orchestrator (metrics auto-enabled)
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Start metrics HTTP server
    start_metrics_server(port=8001)

    # Process queries (metrics auto-collected)
    spacetime = await orchestrator.weave(Query(text="Your question here"))

# Metrics now available at: http://localhost:8001/metrics
```

## 3. View Raw Metrics

```bash
# View all metrics
curl http://localhost:8001/metrics

# View specific metric
curl http://localhost:8001/metrics | grep hololoom_queries_total
```

## 4. Setup Prometheus

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hololoom'
    static_configs:
      - targets: ['localhost:8001']
```

Start Prometheus:

```bash
# Download from https://prometheus.io/download/
prometheus --config.file=prometheus.yml

# Access at http://localhost:9090
```

## 5. Setup Grafana

1. Start Grafana: `docker run -d -p 3000:3000 grafana/grafana`
2. Open http://localhost:3000 (login: admin/admin)
3. Add Prometheus data source:
   - URL: http://localhost:9090
4. Import dashboard JSON from `HoloLoom/performance/grafana_dashboard_template.json`
5. View metrics!

## Key Metrics

### Performance
- `hololoom_weaving_duration_seconds` - Query latency (p50/p95/p99)
- `hololoom_stage_duration_seconds` - Stage breakdown
- `hololoom_parallel_speedup` - Parallelization effectiveness

### Reliability
- `hololoom_cache_hits_total` vs `hololoom_cache_misses_total` - Cache performance
- `hololoom_errors_total` - Error tracking
- `hololoom_backend_status` - Dependency health

### Usage
- `hololoom_queries_total` - Query volume and distribution
- `hololoom_confidence_score` - Tool confidence trends
- `hololoom_active_threads` - Complexity distribution

## Useful Queries

### 95th Percentile Latency

```promql
histogram_quantile(0.95, sum(rate(hololoom_weaving_duration_seconds_bucket[5m])) by (le))
```

### Cache Hit Rate

```promql
rate(hololoom_cache_hits_total[5m]) /
(rate(hololoom_cache_hits_total[5m]) + rate(hololoom_cache_misses_total[5m]))
```

### Queries Per Second

```promql
sum(rate(hololoom_queries_total[1m]))
```

### Error Rate

```promql
rate(hololoom_errors_total[5m]) / rate(hololoom_queries_total[5m])
```

## Alerting (Optional)

Create `alerts.yml`:

```yaml
groups:
  - name: hololoom
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, hololoom_weaving_duration_seconds) > 0.15
        for: 5m
        annotations:
          summary: "Query latency above 150ms"

      - alert: LowCacheHit
        expr: rate(hololoom_cache_hits_total[5m]) / (rate(hololoom_cache_hits_total[5m]) + rate(hololoom_cache_misses_total[5m])) < 0.5
        for: 10m
        annotations:
          summary: "Cache hit rate below 50%"
```

Enable in Prometheus:

```yaml
# prometheus.yml
rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

## Troubleshooting

### Metrics not appearing?

```bash
# 1. Check metrics endpoint
curl http://localhost:8001/metrics | grep hololoom

# 2. Check logs
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from HoloLoom.performance.prometheus_metrics import metrics
print('Metrics enabled:', metrics.enabled)
"

# 3. Verify prometheus_client installed
pip list | grep prometheus-client
```

### High cardinality?

Limit label values or use recording rules:

```yaml
# prometheus.yml
rule_files:
  - 'recording_rules.yml'
```

```yaml
# recording_rules.yml
groups:
  - name: hololoom
    rules:
      - record: hololoom:p95_latency
        expr: histogram_quantile(0.95, hololoom_weaving_duration_seconds)
```

## Full Documentation

See `PROMETHEUS_METRICS.md` for:
- Complete metrics reference
- Grafana dashboard setup
- Advanced Prometheus queries
- Recording rules and alerts
- Performance impact analysis

## Next Steps

1. **Monitor in Real-Time**: Watch dashboard as queries execute
2. **Set Alerts**: Configure alerts for SLO violations
3. **Tune Performance**: Use metrics to identify bottlenecks
4. **Archive Metrics**: Use remote storage for long-term analysis
5. **Integrate**: Add custom dashboards for your use cases
