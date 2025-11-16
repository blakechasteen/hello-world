# HoloLoom Performance Monitoring - Quick Start

**5-minute setup guide**

## Step 1: Install Dependencies (1 minute)

```bash
# Required for full functionality
pip install prometheus-client psutil

# Optional (already in HoloLoom)
pip install fastapi uvicorn
```

## Step 2: Start Dashboard Server (1 minute)

```bash
cd /home/user/hello-world
uvicorn HoloLoom.dashboard_server:app --reload --port 8000
```

**Expected output**:
```
INFO: Enabling Prometheus metrics collection
INFO: Using slowapi for rate limiting
INFO: Initializing HoloLoom Promptly Dashboard...
INFO: Configuration: fast mode
INFO: Analytics database connected
INFO: Loaded 15 professional skills
INFO: Dashboard server ready! API version: v1, Rate limiting: enabled
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 3: View Performance Dashboard (1 minute)

Open browser:
```
http://localhost:8000/performance
```

You should see:
- Real-time metrics updating every 5 seconds
- CPU, memory, request rate, latency stats
- Charts showing performance trends
- Active alerts (if thresholds exceeded)

## Step 4: Test Prometheus Metrics (1 minute)

```bash
curl http://localhost:8000/metrics
```

**Expected output**:
```
# HELP hololoom_requests_total Total number of requests
# TYPE hololoom_requests_total counter
hololoom_requests_total{endpoint="/api/v1/analytics/summary",method="GET",status="200"} 5.0
...
# HELP hololoom_system_cpu_percent CPU usage percentage
# TYPE hololoom_system_cpu_percent gauge
hololoom_system_cpu_percent 25.3
...
```

## Step 5: Query Current Metrics (1 minute)

```bash
curl http://localhost:8000/api/v1/monitoring/current | jq
```

**Expected output**:
```json
{
  "system": {
    "cpu_percent": 25.3,
    "memory_mb": 512.5,
    "memory_percent": 12.8,
    "active_connections": 3
  },
  "requests": {
    "total": 0,
    "rate_per_second": 15.2,
    "error_rate": 0.0
  },
  "performance": {
    "p50_latency_ms": 85.5,
    "p95_latency_ms": 250.0,
    "p99_latency_ms": 500.0
  },
  "cache": {
    "query_cache_hit_rate": 0.75,
    "embedding_cache_hit_rate": 0.82
  },
  "timestamp": "2025-11-16T12:34:56.789Z"
}
```

## Optional: Set Up Prometheus (5 minutes)

### 1. Install Prometheus

**macOS**:
```bash
brew install prometheus
```

**Linux**:
```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*
```

### 2. Configure Prometheus

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hololoom'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'hololoom-dashboard'
```

### 3. Start Prometheus

```bash
prometheus --config.file=prometheus.yml
```

### 4. Open Prometheus UI

```
http://localhost:9090
```

Try queries:
```
hololoom_requests_total
rate(hololoom_request_duration_seconds[5m])
hololoom_cache_hit_rate
```

## Optional: Set Up Grafana (5 minutes)

### 1. Install Grafana

**macOS**:
```bash
brew install grafana
brew services start grafana
```

**Linux**:
```bash
sudo apt-get install -y grafana
sudo systemctl start grafana-server
```

### 2. Open Grafana

```
http://localhost:3000
```

Default credentials: `admin` / `admin`

### 3. Add Prometheus Data Source

1. Click **⚙️ Configuration** → **Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. URL: `http://localhost:9090`
5. Click **Save & Test**

### 4. Import Dashboard

```bash
# Generate dashboard JSON
python -c "from HoloLoom.monitoring.prometheus_exporter import get_grafana_dashboard_json; import json; print(json.dumps(get_grafana_dashboard_json(), indent=2))" > hololoom_dashboard.json
```

In Grafana:
1. Click **+** → **Import**
2. Upload `hololoom_dashboard.json`
3. Select Prometheus data source
4. Click **Import**

## Troubleshooting

### Prometheus client not available

**Error**: "Prometheus client not available - metrics collection disabled"

**Solution**:
```bash
pip install prometheus-client
```

### psutil not available

**Error**: System metrics show 0%

**Solution**:
```bash
pip install psutil
```

**Note**: System will work without psutil, but CPU/memory metrics will be unavailable.

### Dashboard not loading

**Error**: 404 on `/performance`

**Solution**: Check that `HoloLoom/monitoring/dashboard.html` exists:
```bash
ls -la HoloLoom/monitoring/dashboard.html
```

### No metrics data

**Error**: Dashboard shows "Loading..." forever

**Solution**:
1. Check `/api/v1/monitoring/current` returns data:
   ```bash
   curl http://localhost:8000/api/v1/monitoring/current
   ```
2. Check browser console for errors (F12)
3. Verify dashboard server is running

## Example: Integrate with Your Code

```python
from HoloLoom.monitoring.performance_metrics import get_metrics_collector
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
import time

# Get metrics collector
metrics = get_metrics_collector()

# Record a weaving operation
async def monitored_weave(query):
    start = time.time()

    try:
        spacetime = await orchestrator.weave(query)
        duration = time.time() - start

        # Record metrics
        metrics.record_weaving(
            complexity="FAST",
            duration_seconds=duration,
            confidence=spacetime.confidence
        )

        return spacetime
    except Exception as e:
        # Record error
        metrics.record_error(
            endpoint="/weave",
            error_type=type(e).__name__
        )
        raise
```

## Next Steps

1. **Production deployment**: See `README.md` for full guide
2. **Alert configuration**: Customize thresholds in `prometheus_exporter.py`
3. **Grafana dashboards**: Create custom panels for your metrics
4. **Integration**: Add monitoring to your HoloLoom workflows

## Support

- **Documentation**: `HoloLoom/monitoring/README.md`
- **API Reference**: `HoloLoom/monitoring/README.md#api-endpoints`
- **Examples**: `HoloLoom/monitoring/IMPLEMENTATION_SUMMARY.md#integration-examples`
