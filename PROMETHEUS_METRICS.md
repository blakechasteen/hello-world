# Prometheus Metrics for HoloLoom

**Status**: Production Ready (November 2025)
**Location**: `HoloLoom/performance/prometheus_metrics.py`
**Integration**: `HoloLoom/weaving_orchestrator.py`

Comprehensive Prometheus metrics for production monitoring of the HoloLoom orchestrator system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Metric Categories](#metric-categories)
3. [Metrics Reference](#metrics-reference)
4. [Prometheus Queries](#prometheus-queries)
5. [Grafana Dashboard](#grafana-dashboard)
6. [Integration Guide](#integration-guide)
7. [Performance Impact](#performance-impact)

---

## Quick Start

### Enable Metrics

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.performance.prometheus_metrics import start_metrics_server

# Create config with metrics enabled (default)
config = Config.fused()
config.enable_prometheus_metrics = True
config.prometheus_metrics_port = 8001

# Create orchestrator
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(query)

# Start metrics HTTP server
start_metrics_server(port=8001)

# Metrics now available at http://localhost:8001/metrics
```

### Disable Metrics

```python
# If prometheus_client not installed, metrics gracefully disabled
# To explicitly disable:
config.enable_prometheus_metrics = False
```

### View Metrics

```bash
# Direct HTTP endpoint
curl http://localhost:8001/metrics

# With Prometheus running
# Add to prometheus.yml:
# scrape_configs:
#   - job_name: 'hololoom'
#     static_configs:
#       - targets: ['localhost:8001']
```

---

## Metric Categories

### 1. Query Counters

Track query volume and patterns.

| Metric | Labels | Description |
|--------|--------|-------------|
| `hololoom_queries_total` | `complexity`, `pattern_card`, `tool_used` | Total queries processed |
| `hololoom_cache_hits_total` | `cache_type` | Cache hits (query_cache, semantic_cache) |
| `hololoom_cache_misses_total` | `cache_type` | Cache misses |
| `hololoom_errors_total` | `error_type`, `stage` | Errors by type and stage |
| `hololoom_motifs_detected_total` | `motif_type` | Motifs detected (regex, spacy, semantic) |
| `hololoom_reflection_updates_total` | `signal_type` | Reflection loop updates |

### 2. Latency Histograms

Track performance with percentile distributions.

| Metric | Labels | Buckets | Description |
|--------|--------|---------|-------------|
| `hololoom_weaving_duration_seconds` | `complexity`, `pattern_card` | 10ms-5s | Total weaving cycle latency |
| `hololoom_stage_duration_seconds` | `stage_name` | 1ms-500ms | Individual stage execution time |
| `hololoom_tool_execution_duration_seconds` | `tool_name` | 10ms-5s | Tool execution latency |
| `hololoom_parallel_execution_duration_seconds` | `stage_group` | 10ms-500ms | Parallel execution wall-clock time |

**SLO Targets**:
- p50: 100ms (weaving)
- p95: 150ms (weaving)
- p99: 300ms (weaving)

### 3. Gauges

Real-time system state.

| Metric | Labels | Description |
|--------|--------|-------------|
| `hololoom_active_threads` | `pattern_card` | Active memory threads by pattern |
| `hololoom_confidence_score` | `tool_used` | Last query confidence by tool |
| `hololoom_memory_shards_count` | — | Total memory shards available |
| `hololoom_parallel_speedup` | `stage_group` | Parallel execution speedup factor |
| `hololoom_retrieval_context_size` | — | Retrieved context size (shards) |
| `hololoom_backend_status` | `backend` | Backend health (1=healthy, 0=down) |

---

## Metrics Reference

### Query Counters

#### hololoom_queries_total

```
hololoom_queries_total{complexity="FAST",pattern_card="FUSED",tool_used="answer"} 1234.0
```

Track number of queries processed by:
- **complexity**: LITE, FAST, FULL, RESEARCH
- **pattern_card**: BARE, FAST, FUSED
- **tool_used**: answer, search, notion_write, calc

**Use Cases**:
- Identify which pattern cards are used most
- Track tool selection distribution
- Monitor query volume by complexity

#### hololoom_cache_hits_total / hololoom_cache_misses_total

```
hololoom_cache_hits_total{cache_type="query_cache"} 5420.0
hololoom_cache_misses_total{cache_type="query_cache"} 1245.0
```

Track cache performance:
- **cache_type**: query_cache, semantic_cache, linguistic_cache

**Use Cases**:
- Calculate hit rate over time window
- Detect cache effectiveness degradation
- Identify cache tuning opportunities

### Latency Histograms

#### hololoom_weaving_duration_seconds

```
# Histogram bucket (query latency)
hololoom_weaving_duration_seconds_bucket{complexity="FAST",pattern_card="FUSED",le="0.1"} 820
hololoom_weaving_duration_seconds_bucket{complexity="FAST",pattern_card="FUSED",le="0.15"} 950
hololoom_weaving_duration_seconds_bucket{complexity="FAST",pattern_card="FUSED",le="+Inf"} 1000
hololoom_weaving_duration_seconds_sum{complexity="FAST",pattern_card="FUSED"} 125.5
hololoom_weaving_duration_seconds_count{complexity="FAST",pattern_card="FUSED"} 1000
```

Total query processing latency by:
- **complexity**: Distinguish latency by query complexity
- **pattern_card**: Compare pattern performance

**Use Cases**:
- SLO monitoring (p95 < 150ms)
- Performance regression detection
- Pattern card effectiveness comparison

#### hololoom_stage_duration_seconds

```
hololoom_stage_duration_seconds_bucket{stage_name="retrieval",le="0.05"} 4200
hololoom_stage_duration_seconds_bucket{stage_name="feature_extraction",le="0.025"} 3900
hololoom_stage_duration_seconds_bucket{stage_name="tool_execution",le="0.1"} 1200
```

Individual stage latencies for:
- pattern_selection, temporal_setup, thread_selection
- feature_extraction, warp_tensioning, retrieval
- convergence, tool_execution, spacetime_assembly

**Use Cases**:
- Identify bottleneck stages
- Track stage-level performance
- Detect anomalies in specific stages

### Gauges

#### hololoom_active_threads

```
hololoom_active_threads{pattern_card="FUSED"} 12.0
hololoom_active_threads{pattern_card="FAST"} 6.0
hololoom_active_threads{pattern_card="BARE"} 3.0
```

Number of active memory threads.

**Use Cases**:
- Monitor query complexity distribution
- Detect queries activating more threads
- Capacity planning

#### hololoom_confidence_score

```
hololoom_confidence_score{tool_used="answer"} 0.92
hololoom_confidence_score{tool_used="search"} 0.78
hololoom_confidence_score{tool_used="notion_write"} 0.85
```

Last query confidence by tool.

**Use Cases**:
- Monitor tool confidence trends
- Detect tools with low confidence
- Track improvement from learning signals

#### hololoom_parallel_speedup

```
hololoom_parallel_speedup{stage_group="steps_4_6"} 2.35
```

Parallel execution speedup (sequential_time / wall_time).

**Use Cases**:
- Monitor parallelization effectiveness
- Detect degradation in parallel execution
- Identify when parallel overhead exceeds benefit

---

## Prometheus Queries

### Latency Queries

#### 95th Percentile Latency by Complexity

```promql
histogram_quantile(0.95,
  sum(rate(hololoom_weaving_duration_seconds_bucket[5m])) by (complexity, le)
)
```

Shows p95 latency for each complexity level over last 5 minutes.

#### 99th Percentile Latency by Pattern

```promql
histogram_quantile(0.99,
  sum(rate(hololoom_weaving_duration_seconds_bucket[5m])) by (pattern_card, le)
)
```

Shows p99 latency for each pattern card.

#### SLO Compliance (p95 < 150ms)

```promql
histogram_quantile(0.95, hololoom_weaving_duration_seconds_bucket) < 0.15
```

Returns 1 if compliant, 0 if violated.

#### Average Stage Latencies

```promql
avg(hololoom_stage_duration_seconds_bucket) by (stage_name)
```

Average latency for each stage.

#### Slowest Stage (Bottleneck)

```promql
topk(1,
  avg(hololoom_stage_duration_seconds) by (stage_name)
)
```

Identifies the slowest processing stage.

### Cache Queries

#### Cache Hit Rate

```promql
rate(hololoom_cache_hits_total{cache_type="query_cache"}[5m]) /
(rate(hololoom_cache_hits_total{cache_type="query_cache"}[5m]) +
 rate(hololoom_cache_misses_total{cache_type="query_cache"}[5m]))
```

Cache hit rate over 5-minute window.

#### Cache Performance by Type

```promql
rate(hololoom_cache_hits_total[5m]) by (cache_type) /
(rate(hololoom_cache_hits_total[5m]) by (cache_type) +
 rate(hololoom_cache_misses_total[5m]) by (cache_type))
```

Hit rate for each cache type.

#### Estimated Cache Speedup

```promql
# Average latency with cache hit vs total
avg(hololoom_stage_duration_seconds{stage_name="retrieval"}) /
(rate(hololoom_cache_hits_total[5m]) / rate(hololoom_queries_total[5m]))
```

### Query Volume Queries

#### Queries Per Second by Tool

```promql
sum(rate(hololoom_queries_total[1m])) by (tool_used)
```

Query volume per second by tool.

#### Queries Per Second by Complexity

```promql
sum(rate(hololoom_queries_total[1m])) by (complexity)
```

Query volume by complexity level.

#### Tool Distribution

```promql
sum(hololoom_queries_total) by (tool_used) / sum(hololoom_queries_total)
```

Percentage of queries using each tool.

### Pattern Card Performance

#### Average Latency by Pattern

```promql
avg(hololoom_weaving_duration_seconds) by (pattern_card)
```

Average latency for each pattern card.

#### Pattern Usage

```promql
sum(hololoom_queries_total) by (pattern_card) / sum(hololoom_queries_total)
```

Percentage of queries using each pattern card.

#### Pattern-Complexity Breakdown

```promql
sum(hololoom_queries_total) by (pattern_card, complexity)
```

Query counts for each pattern-complexity combination.

### Error Queries

#### Error Rate

```promql
rate(hololoom_errors_total[5m]) / rate(hololoom_queries_total[5m])
```

Percentage of queries resulting in errors.

#### Error Types

```promql
topk(5,
  sum(rate(hololoom_errors_total[5m])) by (error_type)
)
```

Top 5 error types.

#### Errors by Stage

```promql
sum(hololoom_errors_total) by (stage)
```

Errors grouped by processing stage.

### Confidence Queries

#### Average Tool Confidence

```promql
avg(hololoom_confidence_score) by (tool_used)
```

Average confidence for each tool.

#### Confidence Trend

```promql
hololoom_confidence_score
```

Confidence over time by tool.

### Parallel Execution Queries

#### Parallel Speedup

```promql
hololoom_parallel_speedup{stage_group="steps_4_6"}
```

Current parallel speedup factor.

#### Parallel Effectiveness

```promql
avg(hololoom_parallel_speedup) over (5m)
```

Average parallel speedup over 5 minutes.

### Backend Queries

#### Backend Health

```promql
hololoom_backend_status
```

Backend status (1=healthy, 0=down).

#### Backend Availability

```promql
avg(hololoom_backend_status{backend="neo4j"})
```

Availability percentage for Neo4j backend.

---

## Grafana Dashboard

### Create Dashboard

1. **Add Data Source**: Prometheus → http://localhost:9090
2. **Import Dashboard**: Use JSON template below or create custom panels
3. **Configure Alerts**: Set thresholds for SLOs

### Dashboard JSON Template

```json
{
  "dashboard": {
    "title": "HoloLoom Orchestrator Metrics",
    "tags": ["hololoom", "ai", "metrics"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Query Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(hololoom_weaving_duration_seconds_bucket[5m])) by (le))"
          }
        ],
        "alert": {
          "conditions": [{"evaluator": {"params": [0.15]}}]
        }
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(hololoom_cache_hits_total[5m]) / (rate(hololoom_cache_hits_total[5m]) + rate(hololoom_cache_misses_total[5m]))"
          }
        ]
      },
      {
        "title": "Queries Per Second",
        "targets": [
          {
            "expr": "sum(rate(hololoom_queries_total[1m])) by (tool_used)"
          }
        ]
      },
      {
        "title": "Tool Confidence",
        "targets": [
          {
            "expr": "avg(hololoom_confidence_score) by (tool_used)"
          }
        ]
      },
      {
        "title": "Stage Latencies",
        "targets": [
          {
            "expr": "avg(hololoom_stage_duration_seconds) by (stage_name)"
          }
        ]
      },
      {
        "title": "Parallel Speedup",
        "targets": [
          {
            "expr": "hololoom_parallel_speedup"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(hololoom_errors_total[5m]) / rate(hololoom_queries_total[5m])"
          }
        ]
      },
      {
        "title": "Backend Status",
        "targets": [
          {
            "expr": "hololoom_backend_status"
          }
        ]
      }
    ]
  }
}
```

### Key Panels

1. **Query Latency**: p50, p95, p99 + trend
2. **Cache Performance**: Hit rate, speedup
3. **Tool Usage**: Pie chart of tools used
4. **Error Rate**: Trend + top error types
5. **Complexity Distribution**: Bar chart by complexity
6. **Stage Timeline**: Heatmap of stage latencies
7. **Confidence Trends**: Line graph by tool
8. **System Health**: Backend status

---

## Integration Guide

### Orchestrator Integration

Metrics are automatically collected in `weaving_orchestrator.weave()`:

```python
if METRICS_ENABLED:
    # Query counters
    metrics.track_query(pattern, complexity, duration, tool_used)

    # Stage timings
    metrics.track_stage_batch(stage_timings)

    # Tool execution
    metrics.track_tool_execution(tool_name, duration)

    # Parallel execution
    metrics.track_parallel_execution(stage_group, wall_time, speedup)

    # Confidence
    metrics.set_confidence(tool_used, confidence)

    # Memory state
    metrics.set_active_threads(pattern, count)
    metrics.set_retrieval_context_size(count)

    # Features
    metrics.track_motifs(count)
```

### Starting Metrics Server

```python
from HoloLoom.performance.prometheus_metrics import start_metrics_server

# Start metrics endpoint
start_metrics_server(port=8001)

# Metrics available at http://localhost:8001/metrics
```

### Reflection Integration

Track learning signals:

```python
if enable_reflection:
    signals = await orchestrator.learn()
    for signal in signals:
        metrics.track_reflection_update(signal.signal_type)
```

### Error Handling

Errors automatically tracked:

```python
if METRICS_ENABLED:
    metrics.track_error(error_type=type(e).__name__, stage='weaving')
```

---

## Performance Impact

### Overhead

Metrics collection has negligible overhead:

| Operation | Duration | Notes |
|-----------|----------|-------|
| Counter increment | <0.001ms | Per metric label |
| Histogram observe | <0.001ms | Per bucket |
| Gauge set | <0.001ms | Per label |
| Full query tracking | <0.05ms | ~50 metrics per query |

**Total Per-Query Overhead**: ~0.05ms (0.05% of typical 100ms query)

### Memory

Metrics memory usage:
- Per-metric: ~100-200 bytes
- Per-label-combination: ~50-100 bytes
- Total for 50 metrics: ~10KB

### Best Practices

1. **Use Aggregation**: Let Prometheus aggregate, not application
2. **Avoid High-Cardinality Labels**: Limit label values (e.g., stage_name not query_text)
3. **Batch Operations**: Track_stage_batch() for multiple stages
4. **Graceful Degradation**: Metrics disabled if prometheus_client unavailable
5. **Selective Tracking**: Only track what you need to monitor

---

## Troubleshooting

### Metrics Not Appearing

```bash
# Check if prometheus_client is installed
pip list | grep prometheus-client

# Install if missing
pip install prometheus-client

# Verify metrics server is running
curl http://localhost:8001/metrics | head

# Check logs for errors
grep -i prometheus <orchestrator.log>
```

### High Cardinality

If too many time series:

```promql
# Count time series
count({__name__=~"hololoom_.*"})

# Investigate problematic metric
count({hololoom_stage_duration_seconds}) by (stage_name)
```

Solution: Reduce label values or use recording rules.

### Memory Issues

If Prometheus consuming too much memory:

1. Reduce scrape frequency (default 15s)
2. Increase scrape timeout for large instance
3. Use recording rules to aggregate
4. Consider remote storage

### Missing Confidence Scores

Confidence only updated if tool executed:

```python
# Ensure tool_used is set
metrics.set_confidence(collapse_result.tool, collapse_result.confidence)
```

---

## Configuration

### Config Options

```python
from HoloLoom.config import Config

config = Config.fused()

# Enable/disable metrics
config.enable_prometheus_metrics = True

# Metrics port
config.prometheus_metrics_port = 8001

# Pass to orchestrator
orchestrator = WeavingOrchestrator(cfg=config, shards=shards)
```

### Environment Variables

```bash
# Enable metrics (default: true)
export HOLOLOOM_PROMETHEUS_ENABLED=true

# Metrics port (default: 8001)
export HOLOLOOM_PROMETHEUS_PORT=8001
```

---

## Advanced Topics

### Custom Metrics

Add custom metrics:

```python
from prometheus_client import Counter

my_counter = Counter('my_metric', 'Description', ['label1', 'label2'])

# In code
my_counter.labels(label1='value1', label2='value2').inc()
```

### Recording Rules

Create recording rules for efficient aggregation:

```yaml
# prometheus.yml
rule_files:
  - 'recording_rules.yml'
```

```yaml
# recording_rules.yml
groups:
  - name: hololoom
    interval: 30s
    rules:
      - record: hololoom:p95_latency:5m
        expr: histogram_quantile(0.95, sum(rate(hololoom_weaving_duration_seconds_bucket[5m])) by (le))

      - record: hololoom:cache_hit_rate:5m
        expr: rate(hololoom_cache_hits_total[5m]) / (rate(hololoom_cache_hits_total[5m]) + rate(hololoom_cache_misses_total[5m]))

      - record: hololoom:qps:1m
        expr: sum(rate(hololoom_queries_total[1m]))
```

Then query pre-computed values:

```promql
hololoom:p95_latency:5m
hololoom:cache_hit_rate:5m
hololoom:qps:1m
```

### Alerting

Define alerts:

```yaml
# prometheus.yml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

rule_files:
  - 'alert_rules.yml'
```

```yaml
# alert_rules.yml
groups:
  - name: hololoom
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, hololoom_weaving_duration_seconds) > 0.15
        for: 5m
        annotations:
          summary: "High query latency detected"

      - alert: LowCacheHitRate
        expr: hololoom:cache_hit_rate:5m < 0.5
        for: 10m
        annotations:
          summary: "Cache hit rate below 50%"

      - alert: HighErrorRate
        expr: rate(hololoom_errors_total[5m]) / rate(hololoom_queries_total[5m]) > 0.01
        for: 5m
        annotations:
          summary: "Error rate above 1%"
```

---

## References

- **Prometheus**: https://prometheus.io/
- **Grafana**: https://grafana.com/
- **prometheus_client**: https://github.com/prometheus/client_python
- **Prometheus Best Practices**: https://prometheus.io/docs/practices/naming/
- **Histogram Buckets**: https://prometheus.io/docs/concepts/metric_types/#histogram

---

## Support

For issues or questions:

1. Check `HoloLoom/performance/prometheus_metrics.py` docstrings
2. Review Prometheus query examples above
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Check metrics at `http://localhost:8001/metrics`
5. Validate PromQL queries at https://prometheus.io/docs/prometheus/latest/querying/
