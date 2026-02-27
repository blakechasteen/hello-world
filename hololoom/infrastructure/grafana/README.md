# HoloLoom Grafana Dashboards

**Part 5: Production Hardening - Day 25**

Grafana dashboard templates for monitoring HoloLoom in production.

## Dashboard Overview

| Dashboard | Description | Panels |
|-----------|-------------|--------|
| **overview-dashboard.json** | High-level system metrics | 11 panels: Status, QPS, latency, errors, cache, pods |

## Quick Start

### Import Dashboards

**Method 1: Grafana UI**
1. Open Grafana (http://localhost:3000)
2. Go to **Dashboards → Import**
3. Click **Upload JSON file**
4. Select `overview-dashboard.json`
5. Select Prometheus data source
6. Click **Import**

**Method 2: ConfigMap (Kubernetes)**
```bash
# Dashboards are already included in kubernetes/servicemonitor.yaml
kubectl apply -f ../kubernetes/servicemonitor.yaml

# Grafana will auto-discover dashboards with label:
# grafana_dashboard: "1"
```

**Method 3: Grafana API**
```bash
# Upload dashboard via API
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @overview-dashboard.json
```

### Configure Data Source

**Add Prometheus data source**:
1. Go to **Configuration → Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. Configure:
   - **Name**: Prometheus
   - **URL**: http://prometheus:9090 (or your Prometheus URL)
   - **Access**: Server (default)
5. Click **Save & Test**

---

## Dashboard Details

### HoloLoom Overview

**Purpose**: High-level system health and performance metrics

**Panels**:

1. **System Status** (Stat)
   - Metric: `up{job="hololoom-api"}`
   - Shows: UP (green) or DOWN (red)
   - Use: Quick health check

2. **Queries Per Second (QPS)** (Stat)
   - Metric: `sum(rate(hololoom_query_total[5m]))`
   - Thresholds: 0-50 (blue), 50-100 (green), 100-200 (yellow), 200+ (red)
   - Use: Monitor request load

3. **P95 Latency** (Stat)
   - Metric: `histogram_quantile(0.95, sum(rate(hololoom_query_latency_seconds_bucket[5m])) by (le)) * 1000`
   - Thresholds: 0-500ms (green), 500-1000ms (yellow), 1000ms+ (red)
   - Use: Monitor query performance

4. **Error Rate** (Stat)
   - Metric: `sum(rate(hololoom_query_total{status="error"}[5m])) / sum(rate(hololoom_query_total[5m]))`
   - Thresholds: 0-5% (green), 5-10% (yellow), 10%+ (red)
   - Use: Monitor reliability

5. **Query Rate Over Time** (Graph)
   - Metrics: Total QPS, Success QPS, Error QPS
   - Use: Visualize request patterns

6. **Query Latency Percentiles** (Graph)
   - Metrics: P50, P95, P99 latencies
   - Alert: P95 > 1000ms for 5m
   - Use: Monitor latency distribution

7. **Cache Hit Rate** (Gauge)
   - Metric: `sum(rate(hololoom_cache_hits_total[5m])) / sum(rate(hololoom_cache_requests_total[5m]))`
   - Thresholds: 0-30% (red), 30-50% (yellow), 50-100% (green)
   - Use: Monitor cache effectiveness

8. **Active Pods** (Stat)
   - Metric: `count(up{job="hololoom-api"} == 1)`
   - Thresholds: 0 (red), 1-2 (yellow), 3+ (green)
   - Use: Monitor pod availability

9. **Total Queries (24h)** (Stat)
   - Metric: `sum(increase(hololoom_query_total[24h]))`
   - Use: Daily volume tracking

10. **Circuit Breaker Status** (Table)
    - Metric: `hololoom_circuit_breaker_state`
    - Shows: Backend name, state (CLOSED/OPEN/HALF_OPEN)
    - Use: Monitor backend health

11. **Memory Usage** (Graph)
    - Metrics: Total memory, Memory limit
    - Use: Monitor resource consumption

---

## Metrics Reference

### Core Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `hololoom_query_total` | Counter | Total queries processed | status={success\|error} |
| `hololoom_query_latency_seconds` | Histogram | Query latency distribution | - |
| `hololoom_cache_hits_total` | Counter | Cache hits | - |
| `hololoom_cache_requests_total` | Counter | Cache requests | - |
| `hololoom_circuit_breaker_state` | Gauge | Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN) | backend, state |
| `hololoom_rate_limit_rejected_total` | Counter | Rate limit rejections | - |
| `up` | Gauge | Service up (1) or down (0) | job, instance, pod |

### Resource Metrics (cAdvisor)

| Metric | Type | Description |
|--------|------|-------------|
| `container_memory_usage_bytes` | Gauge | Memory usage in bytes |
| `container_spec_memory_limit_bytes` | Gauge | Memory limit in bytes |
| `container_cpu_usage_seconds_total` | Counter | CPU usage in seconds |
| `container_spec_cpu_quota` | Gauge | CPU quota |

---

## Useful Queries

### Performance

**Query rate (QPS)**:
```promql
sum(rate(hololoom_query_total[5m]))
```

**Latency percentiles**:
```promql
# P50
histogram_quantile(0.50, sum(rate(hololoom_query_latency_seconds_bucket[5m])) by (le)) * 1000

# P95
histogram_quantile(0.95, sum(rate(hololoom_query_latency_seconds_bucket[5m])) by (le)) * 1000

# P99
histogram_quantile(0.99, sum(rate(hololoom_query_latency_seconds_bucket[5m])) by (le)) * 1000
```

**Error rate**:
```promql
sum(rate(hololoom_query_total{status="error"}[5m])) / sum(rate(hololoom_query_total[5m]))
```

**Cache hit rate**:
```promql
sum(rate(hololoom_cache_hits_total[5m])) / sum(rate(hololoom_cache_requests_total[5m]))
```

### Reliability

**Service availability**:
```promql
avg(up{job="hololoom-api"})
```

**Circuit breaker open count**:
```promql
count(hololoom_circuit_breaker_state{state="open"} == 1)
```

**Rate limit rejections**:
```promql
rate(hololoom_rate_limit_rejected_total[5m])
```

### Resources

**Memory usage percentage**:
```promql
sum(container_memory_usage_bytes{pod=~"hololoom-api-.*"}) /
sum(container_spec_memory_limit_bytes{pod=~"hololoom-api-.*"}) * 100
```

**CPU usage percentage**:
```promql
sum(rate(container_cpu_usage_seconds_total{pod=~"hololoom-api-.*"}[5m])) /
sum(container_spec_cpu_quota{pod=~"hololoom-api-.*"}) * 100
```

**Pod count**:
```promql
count(up{job="hololoom-api"})
```

---

## Alerting

Alerts are configured in `../kubernetes/servicemonitor.yaml` (PrometheusRule).

**View active alerts in Grafana**:
1. Go to **Alerting → Alert Rules**
2. Filter by: `alertname=~"HoloLoom.*"`

**Key alerts**:
- `HoloLoomHighLatency`: P95 > 1s for 5m
- `HoloLoomHighErrorRate`: Error rate > 10% for 5m
- `HoloLoomLowCacheHitRate`: Cache hit < 30% for 10m
- `HoloLoomServiceDown`: Service down for 1m
- `HoloLoomHighMemory`: Memory > 85% for 5m

---

## Customization

### Add Custom Panel

1. Click **Add panel** in dashboard
2. Select **Add a new panel**
3. Configure query:
   ```promql
   your_metric_name
   ```
4. Select visualization (Graph, Stat, Table, etc.)
5. Configure thresholds and colors
6. Click **Apply**

### Modify Existing Panel

1. Click panel title → **Edit**
2. Modify query, visualization, or options
3. Click **Apply**

### Create Custom Dashboard

1. Go to **Dashboards → New Dashboard**
2. Add panels (see above)
3. Configure variables (optional):
   - Go to **Dashboard settings → Variables**
   - Add variable (e.g., `namespace`, `pod`)
   - Use in queries: `{namespace="$namespace"}`
4. Save dashboard

---

## Variables (Advanced)

Add variables for dynamic filtering:

**Namespace variable**:
```
Name: namespace
Type: Query
Query: label_values(up{job="hololoom-api"}, namespace)
```

**Pod variable**:
```
Name: pod
Type: Query
Query: label_values(up{job="hololoom-api", namespace="$namespace"}, pod)
```

**Use in queries**:
```promql
up{job="hololoom-api", namespace="$namespace", pod="$pod"}
```

---

## Troubleshooting

### Dashboard shows "No data"

**Check**:
1. Prometheus data source configured correctly
2. Metrics are being scraped:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```
3. Queries are correct:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=up{job="hololoom-api"}'
   ```

### Queries return empty

**Possible causes**:
1. **No data**: HoloLoom not running or not exporting metrics
2. **Wrong job name**: Check ServiceMonitor label
3. **Time range**: Adjust time picker (top right)
4. **Metric name**: Verify metric name in Prometheus:
   ```bash
   curl http://localhost:9090/api/v1/label/__name__/values | grep hololoom
   ```

### Panels show errors

**Common errors**:
1. **"parse error"**: Invalid PromQL syntax
2. **"bad_data"**: Wrong label names or values
3. **"timeout"**: Query too complex, simplify or increase timeout

**Debug**:
1. Test query in Prometheus UI (http://localhost:9090/graph)
2. Check Grafana logs:
   ```bash
   kubectl logs -n monitoring -l app.kubernetes.io/name=grafana
   ```

---

## Additional Resources

- **Prometheus Query Examples**: https://prometheus.io/docs/prometheus/latest/querying/examples/
- **Grafana Documentation**: https://grafana.com/docs/
- **Kubernetes Monitoring**: `../kubernetes/servicemonitor.yaml`
- **Operations Runbook**: `../context/OPERATIONS_RUNBOOK.md`

---

## Summary

### What's Included

✅ **Overview Dashboard**: 11 panels covering key metrics
✅ **Prometheus Metrics**: 15+ production-ready metrics
✅ **Alert Rules**: 10+ alerts for critical issues
✅ **Kubernetes Integration**: ServiceMonitor + ConfigMap
✅ **Documentation**: This guide + query reference

### Next Steps

1. Import dashboards into Grafana
2. Configure Prometheus data source
3. Verify metrics are being scraped
4. Customize dashboards for your environment
5. Set up alert notifications (Slack, PagerDuty, email)

---

**End of Grafana Dashboard Guide**
