# Production Monitoring Guide

**Version**: 1.0.0
**Date**: November 1, 2025
**Purpose**: Real-time P99 latency monitoring and alerting

---

## Overview

The Alignment Framework includes comprehensive production monitoring with:
- **Real-time P99 latency tracking** for all components
- **Configurable alerting** with WARNING and CRITICAL levels
- **Live dashboard** with terminal-based visualization
- **Prometheus integration** for existing monitoring infrastructure
- **Persistent metrics** for historical analysis

---

## Quick Start

### 1. Basic Monitoring

```python
from hololoom.alignment.monitoring import AlignmentMonitor

# Create monitor
monitor = AlignmentMonitor()

# Track component latency
with monitor.track("guardrails"):
    # ... guardrails.evaluate() ...

with monitor.track("detector"):
    # ... detector.run_probe() ...

# View metrics
print(monitor.get_summary())
```

### 2. Production Integration

```python
from pathlib import Path
from hololoom.alignment.monitoring import AlignmentMonitor

# Production monitor with persistence
monitor = AlignmentMonitor(
    thresholds={
        "guardrails": 1.0,    # 1ms P99
        "detector": 2.0,      # 2ms P99
        "guard": 0.5,         # 0.5ms P99
        "audit": 10.0,        # 10ms P99 (allows for I/O spikes)
        "pipeline": 20.0,     # 20ms P99 total
    },
    persist_path=Path("./production_metrics.json")
)

# Process queries with monitoring
async def process_query(query_text):
    with monitor.track("pipeline"):
        # Step 1: Safety check
        with monitor.track("guardrails"):
            decision = guardrails.evaluate(request, text_input=query_text)

        # Step 2: Deception check
        with monitor.track("detector"):
            passed, score = detector.run_probe(probe, response)

        # Step 3: Resource check
        with monitor.track("guard"):
            violation = guard.check_resource_usage(ResourceType.MEMORY, 500.0)

        # Step 4: Audit logging
        with monitor.track("audit"):
            log = audit.log_decision(...)

    # Check for alerts
    critical_alerts = monitor.check_alerts(level=AlertLevel.CRITICAL)
    if critical_alerts:
        print(f"🔴 CRITICAL ALERTS: {len(critical_alerts)}")
```

### 3. Live Dashboard

```bash
# Run live monitoring dashboard
python hololoom/alignment/live_monitor.py
```

**Output**:
```
================================================================================
                    ALIGNMENT FRAMEWORK LIVE MONITOR
================================================================================
  Current Time: 2025-11-01 23:15:42
  Session Start: 2025-11-01 23:10:00
  Uptime: 5m 42s

  COMPONENT LATENCIES (milliseconds)
  ----------------------------------------------------------------------------
  Component          Count      P50        P95        P99        Status
  ----------------------------------------------------------------------------
  audit              1000       0.025      0.045      0.389      ✅ OK
  detector           1000       0.032      0.048      0.091      ✅ OK
  guard              1000       0.001      0.001      0.002      ✅ OK
  guardrails         1000       0.037      0.056      0.084      ✅ OK
  pipeline           1000       0.195      0.312      2.145      ✅ OK
  ----------------------------------------------------------------------------

  RECENT ALERTS
  ----------------------------------------------------------------------------
                         ✅ No alerts - all systems nominal
  ----------------------------------------------------------------------------

  SUMMARY
  ----------------------------------------------------------------------------
  Total Queries Processed: 1000
  Total Alerts: 0 (🔴 0 critical, ⚠️  0 warnings)
  Overall P99 Latency: 2.145 ms
  ----------------------------------------------------------------------------

  Press Ctrl+C to stop monitoring
================================================================================
```

---

## Alert Configuration

### Default Thresholds

| Component | P99 Target | WARNING | CRITICAL |
|-----------|------------|---------|----------|
| guardrails | <1ms | 0.5ms | 1.0ms |
| detector | <2ms | 1.0ms | 2.0ms |
| guard | <0.5ms | 0.25ms | 0.5ms |
| audit | <10ms | 5.0ms | 10.0ms |
| pipeline | <20ms | 10.0ms | 20.0ms |

### Custom Thresholds

```python
# Adjust thresholds for your environment
monitor = AlignmentMonitor(
    thresholds={
        "guardrails": 2.0,    # More lenient
        "detector": 5.0,
        "guard": 1.0,
        "audit": 50.0,        # Allow higher latency for file I/O
        "pipeline": 100.0,
    }
)
```

### Alert Levels

**WARNING (⚠️)**: P99 > 50% of threshold
- Logged to console/file
- 5-minute cooldown between duplicate alerts
- Action: Monitor, no immediate action needed

**CRITICAL (🔴)**: P99 > threshold
- Logged to console/file
- Triggers alerting system (if configured)
- 5-minute cooldown between duplicate alerts
- Action: Investigate immediately, consider rollback

---

## Metrics API

### Recording Metrics

```python
# Manual recording
monitor.record("custom_component", latency_ms=5.2)

# Context manager (recommended)
with monitor.track("custom_component"):
    # ... operation ...
```

### Querying Metrics

```python
# Single component
stats = monitor.get_stats("guardrails")
print(f"P99: {stats['p99']:.3f} ms")

# All components
all_stats = monitor.get_all_stats()
for component, stats in all_stats.items():
    print(f"{component}: P99={stats['p99']:.3f} ms")

# Formatted summary
print(monitor.get_summary())
```

### Checking Alerts

```python
# All alerts
all_alerts = monitor.check_alerts()

# Critical only
critical = monitor.check_alerts(level=AlertLevel.CRITICAL)

# Clear alerts
monitor.clear_alerts()
```

### Persistence

```python
# Save metrics to disk
monitor.persist_metrics()

# Load previous session
monitor.load_metrics()
```

---

## Prometheus Integration

### Export Metrics

```python
# Get Prometheus-formatted metrics
prometheus_text = monitor.export_prometheus()

# Serve via HTTP endpoint (Flask example)
from flask import Flask, Response

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(
        monitor.export_prometheus(),
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
```

### Prometheus Metrics Format

```
# Latency percentiles
alignment_latency_p50{component="guardrails"} 0.039
alignment_latency_p95{component="guardrails"} 0.058
alignment_latency_p99{component="guardrails"} 0.084

alignment_latency_p50{component="detector"} 0.034
alignment_latency_p95{component="detector"} 0.049
alignment_latency_p99{component="detector"} 0.091

# Sample counts
alignment_samples_total{component="guardrails"} 1000
alignment_samples_total{component="detector"} 1000

# Alert counts
alignment_alerts_total{level="critical"} 0
alignment_alerts_total{level="warning"} 2
```

### Grafana Dashboard

Example PromQL queries for Grafana:

```promql
# P99 latency over time
alignment_latency_p99{component="pipeline"}

# Total throughput
rate(alignment_samples_total[5m])

# Alert rate
rate(alignment_alerts_total{level="critical"}[1h])

# Component comparison
sum by (component) (alignment_latency_p99)
```

---

## Production Deployment

### Option 1: Embedded Monitoring (Recommended)

```python
from pathlib import Path
from hololoom.alignment.monitoring import AlignmentMonitor, set_global_monitor

# Create and set global monitor
monitor = AlignmentMonitor(persist_path=Path("./production_metrics.json"))
set_global_monitor(monitor)

# Use throughout application
from hololoom.alignment.monitoring import get_global_monitor

monitor = get_global_monitor()

with monitor.track("guardrails"):
    # ... guardrails logic ...
```

### Option 2: Separate Monitoring Service

```python
# monitoring_service.py
from flask import Flask, Response, jsonify
from hololoom.alignment.monitoring import AlignmentMonitor

app = Flask(__name__)
monitor = AlignmentMonitor()

@app.route('/metrics')
def prometheus_metrics():
    return Response(monitor.export_prometheus(), mimetype='text/plain')

@app.route('/summary')
def summary():
    return jsonify(monitor.get_all_stats())

@app.route('/alerts')
def alerts():
    return jsonify([
        {
            "level": a.level.value,
            "component": a.component,
            "message": a.message,
            "timestamp": a.timestamp.isoformat(),
        }
        for a in monitor.check_alerts()
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
```

### Option 3: Background Dashboard

```python
from hololoom.alignment.live_monitor import LiveDashboard
from hololoom.alignment.monitoring import AlignmentMonitor

# Start background dashboard
monitor = AlignmentMonitor()
dashboard = LiveDashboard(monitor, refresh_interval=2.0)
dashboard.start()

# Process queries...
# Dashboard updates in background

# Stop when done
dashboard.stop()
```

---

## Best Practices

### 1. Track Full Pipeline

```python
# ✅ Good - track end-to-end latency
with monitor.track("pipeline"):
    with monitor.track("guardrails"):
        # ...
    with monitor.track("detector"):
        # ...

# ❌ Bad - missing overall timing
with monitor.track("guardrails"):
    # ...
with monitor.track("detector"):
    # ...
```

### 2. Periodic Persistence

```python
# ✅ Good - persist every N queries
query_count = 0
for query in queries:
    process_query(query)
    query_count += 1

    if query_count % 1000 == 0:
        monitor.persist_metrics()

# ❌ Bad - never persist (lose metrics on crash)
for query in queries:
    process_query(query)
```

### 3. Alert Handling

```python
# ✅ Good - check and handle alerts
critical_alerts = monitor.check_alerts(level=AlertLevel.CRITICAL)
if critical_alerts:
    send_page_to_oncall(critical_alerts)
    consider_rollback()

# ❌ Bad - ignore alerts
monitor.track(...)  # Alerts generated but never checked
```

### 4. Threshold Tuning

```python
# ✅ Good - start conservative, tune based on data
monitor = AlignmentMonitor(thresholds={
    "pipeline": 50.0,  # Start high
})

# After observing actual P99:
# - If P99 = 5ms, lower to 10ms (2x headroom)
# - If P99 = 45ms, investigate performance issues
```

---

## Troubleshooting

### Issue: False Positive Alerts

**Symptom**: Constant WARNING alerts during normal operation

**Cause**: Thresholds too aggressive

**Solution**:
```python
# Increase thresholds based on observed P99
stats = monitor.get_stats("audit")
observed_p99 = stats["p99"]

# Set threshold to 2x observed P99 for headroom
monitor.thresholds["audit"] = observed_p99 * 2
```

### Issue: Missing Metrics

**Symptom**: `get_stats()` returns empty dict

**Cause**: Component not tracked yet

**Solution**:
```python
# Ensure all operations are tracked
if component not in monitor.metrics:
    print(f"⚠️  Component '{component}' not being tracked")

# Add tracking:
with monitor.track(component):
    # ... operation ...
```

### Issue: Dashboard Not Updating

**Symptom**: Live dashboard shows stale data

**Cause**: Dashboard not running or crashed

**Solution**:
```python
# Check dashboard status
if not dashboard.running:
    print("⚠️  Dashboard not running")
    dashboard.start()

# Or render manually
dashboard.render_once()
```

### Issue: High Memory Usage

**Symptom**: Monitor consuming increasing memory

**Cause**: Too many samples stored

**Solution**:
```python
# Reduce window size
monitor = AlignmentMonitor(window_size=500)  # Default 1000

# Or periodically clear old metrics
for component, metrics in monitor.metrics.items():
    if len(metrics.samples) > 5000:
        metrics.samples = metrics.samples[-1000:]  # Keep recent 1000
```

---

## Performance Impact

### Monitoring Overhead

**Per-Query Overhead**: <0.01 ms
- `time.perf_counter()` calls: ~2 per tracked operation
- List append: ~0.001 ms
- Percentile calculation: Lazy (only on request)

**Memory Footprint**:
- Per sample: ~8 bytes (float)
- Per component (1000 samples): ~8 KB
- Total (5 components): ~40 KB

**Verdict**: Negligible overhead, safe for production ✅

---

## Examples

### Example 1: Basic Integration

```python
from hololoom.alignment.monitoring import AlignmentMonitor

monitor = AlignmentMonitor()

async def process_query(query_text):
    with monitor.track("pipeline"):
        result = await full_alignment_pipeline(query_text)

    # Check P99 after every 100 queries
    if monitor.metrics["pipeline"].count % 100 == 0:
        stats = monitor.get_stats("pipeline")
        print(f"P99: {stats['p99']:.2f} ms")

    return result
```

### Example 2: Alert Webhook

```python
import requests

def send_alert_webhook(alert):
    """Send alert to Slack/Discord/etc."""
    webhook_url = "https://hooks.slack.com/services/..."

    message = {
        "text": f"🔴 CRITICAL ALERT: {alert.component}",
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": alert.message}
            }
        ]
    }

    requests.post(webhook_url, json=message)

# Check alerts periodically
import threading

def alert_checker():
    while True:
        critical = monitor.check_alerts(level=AlertLevel.CRITICAL)
        for alert in critical:
            send_alert_webhook(alert)

        time.sleep(60)  # Check every minute

threading.Thread(target=alert_checker, daemon=True).start()
```

### Example 3: Custom Metrics

```python
# Track custom operations
with monitor.track("llm_inference"):
    response = await llm.generate(prompt)

with monitor.track("vector_search"):
    results = vector_db.search(query_embedding)

# View all metrics
print(monitor.get_summary())
```

---

## Production Checklist

### Pre-Deployment

- [ ] Thresholds configured for production hardware
- [ ] Alert handling implemented (webhooks, paging, etc.)
- [ ] Metrics persistence enabled (`persist_path` set)
- [ ] Monitoring dashboard accessible (live or Grafana)
- [ ] Prometheus endpoint exposed (if using existing infrastructure)

### Post-Deployment (First 24 Hours)

- [ ] Monitor P99 latencies continuously
- [ ] Tune thresholds based on observed metrics
- [ ] Verify alerts are firing correctly
- [ ] Check memory usage (should be <100 MB)
- [ ] Review alert history for false positives

### Ongoing (Weekly)

- [ ] Export and archive metrics (`persist_metrics()`)
- [ ] Review P99 trends (improving or degrading?)
- [ ] Adjust thresholds if workload changes
- [ ] Clean up old alert history

---

## FAQ

**Q: How many samples are needed for reliable P99?**
A: Minimum 100 samples. The monitor uses a 1000-sample sliding window for stability.

**Q: Can I monitor non-alignment components?**
A: Yes! Use `monitor.track("custom_component")` for any operation.

**Q: How do I export metrics for analysis?**
A: Use `monitor.persist_metrics()` to save JSON, or `monitor.export_prometheus()` for Prometheus.

**Q: What if P99 spikes temporarily?**
A: The monitor uses a 5-minute cooldown to avoid alert spam. Transient spikes are normal (e.g., during GC pauses).

**Q: Can I use this with async/await?**
A: Yes! The context manager works with both sync and async:
```python
async def async_operation():
    with monitor.track("async_op"):
        await some_async_call()
```

---

**Status**: ✅ Production Ready

**Last Updated**: November 1, 2025
**Version**: 1.0.0
**Next Steps**: Deploy, monitor, tune thresholds based on production data
