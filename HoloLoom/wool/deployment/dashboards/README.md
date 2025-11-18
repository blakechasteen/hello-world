# Wool Storage Grafana Dashboards

Complete set of production-ready Grafana dashboards for monitoring Wool Storage clusters.

## 📊 Available Dashboards

### 1. Cluster Overview (`cluster-overview.json`)
**UID**: `wool-cluster-overview`

High-level cluster health and status monitoring.

**Panels**:
- **Healthy Nodes** (stat) - Number of healthy nodes with color thresholds
- **Total Nodes** (stat) - Total cluster size
- **Node Status** (stat) - Per-node UP/DOWN status
- **Replication Factor** (gauge) - Average replication factor (target: 3)
- **Cluster Health Over Time** (timeseries) - Healthy vs total nodes trend
- **Replication Lag** (timeseries) - Lag in seconds per node
- **Disk Usage by Node** (bargauge) - Storage utilization (70% yellow, 85% red)

**Use Cases**:
- Quick health check
- Capacity planning
- Replication monitoring

### 2. Performance (`performance.json`)
**UID**: `wool-performance`

Detailed performance metrics for throughput, latency, and error rates.

**Panels**:
- **Operations per Second** (timeseries) - Store, read, replicate rates
- **Store Latency** (timeseries) - p50, p95, p99 percentiles
- **Read Latency** (timeseries) - p50, p95, p99 percentiles
- **Error Rate by Type** (timeseries) - Timeout, not found, replication, storage errors
- **Total Throughput** (stat) - Aggregate ops/sec
- **Store p95 Latency** (stat) - Current store latency (500ms yellow, 1s red)
- **Read p95 Latency** (stat) - Current read latency (100ms yellow, 500ms red)
- **Success Rate** (gauge) - Percentage of successful operations

**Use Cases**:
- Performance tuning
- SLA monitoring (p95 < 1s)
- Error diagnosis

### 3. Storage & Compression (`storage.json`)
**UID**: `wool-storage`

Storage utilization, compression ratios, and file counts.

**Panels**:
- **Average Compression Ratio** (gauge) - Overall compression effectiveness (1-20x)
- **Total Storage Used** (stat) - Bytes across all nodes
- **Compression Savings** (stat) - Bytes saved through compression
- **Total Files** (stat) - Number of files stored
- **Compression Ratio by Content Type** (timeseries) - Text, JSON, images
- **Compression Algorithm Usage** (state-timeline) - LZ4 vs Zstd vs none
- **Storage Growth Over Time** (timeseries) - Stacked by node
- **File Count Growth** (timeseries) - Stacked by node
- **Storage Utilization by Node** (bargauge) - Percentage full

**Use Cases**:
- Capacity planning
- Compression optimization
- Storage growth trends

### 4. Versioning & Time-Travel (`versioning.json`)
**UID**: `wool-versioning`

Version control metrics: branches, merges, delta encoding, time-travel queries.

**Panels**:
- **Total Versions** (stat) - Aggregate version count
- **Active Branches** (stat) - Current branch count
- **Merges (24h)** (stat) - Recent merge activity
- **Avg Delta Encoding Ratio** (gauge) - Delta compression effectiveness (1-20x)
- **Version Creation Rate** (timeseries) - Versions/sec per node
- **Merge Rate** (timeseries) - Merges/sec per node
- **Time-Travel Query Rate** (timeseries) - Temporal queries/sec
- **Delta Encoding Efficiency** (timeseries) - Ratio over time
- **Merge Conflicts** (timeseries) - Conflicts/sec (bars)
- **Versions per Branch** (piechart) - Distribution across branches

**Use Cases**:
- Version control monitoring
- Merge conflict tracking
- Delta encoding optimization
- Time-travel usage patterns

### 5. Resource Utilization (`resources.json`)
**UID**: `wool-resources`

System resource usage: CPU, memory, network, disk I/O.

**Panels**:
- **CPU Usage** (timeseries) - Percent per node (70% yellow, 90% red)
- **Memory Usage (RSS)** (timeseries) - Bytes per node (3GB yellow, 3.5GB red)
- **Network I/O** (timeseries) - Receive/transmit bytes/sec (transmit negative)
- **Disk I/O** (timeseries) - Read/write bytes/sec (write negative)
- **Goroutines** (timeseries) - Concurrency level (1k yellow, 5k red)
- **Open File Descriptors** (timeseries) - FD count (5k yellow, 10k red)
- **CPU Usage by Node** (bargauge) - Percentage comparison

**Use Cases**:
- Resource optimization
- Bottleneck identification
- Capacity planning
- Kubernetes HPA tuning

---

## 🚀 Quick Start

### Import Dashboards (Grafana UI)

1. **Access Grafana**:
   ```bash
   # Docker Compose
   open http://localhost:3000

   # Kubernetes
   kubectl port-forward svc/grafana 3000:3000 -n wool-storage
   open http://localhost:3000
   ```

2. **Login**:
   - Username: `admin`
   - Password: `admin` (change on first login)

3. **Import Dashboard**:
   - Click **+** (left sidebar) → **Import**
   - Click **Upload JSON file**
   - Select dashboard JSON (e.g., `cluster-overview.json`)
   - Click **Load**
   - Select **Prometheus** as datasource
   - Click **Import**

4. **Repeat for all 5 dashboards**

### Import via Provisioning (Kubernetes)

Add dashboards to Grafana ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: wool-storage
data:
  cluster-overview.json: |
    <paste cluster-overview.json contents>
  performance.json: |
    <paste performance.json contents>
  # ... etc
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-provider
  namespace: wool-storage
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
      - name: 'Wool Storage'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards
```

Update Grafana deployment to mount dashboards:

```yaml
spec:
  template:
    spec:
      containers:
        - name: grafana
          volumeMounts:
            - name: dashboards
              mountPath: /var/lib/grafana/dashboards
            - name: dashboard-provider
              mountPath: /etc/grafana/provisioning/dashboards
      volumes:
        - name: dashboards
          configMap:
            name: grafana-dashboards
        - name: dashboard-provider
          configMap:
            name: grafana-dashboard-provider
```

---

## 📈 Dashboard Usage

### Default Refresh Rate
All dashboards auto-refresh every **10 seconds** for real-time monitoring.

### Time Ranges
- **Cluster Overview**: Last 1 hour (cluster health is short-term)
- **Performance**: Last 1 hour (latency trends)
- **Storage**: Last 6 hours (growth trends)
- **Versioning**: Last 6 hours (version activity)
- **Resources**: Last 1 hour (resource spikes)

### Color Thresholds

**Cluster Health**:
- Green: 3+ healthy nodes
- Yellow: 2 healthy nodes
- Red: <2 healthy nodes (quorum lost)

**Latency**:
- Green: <500ms (store), <100ms (read)
- Yellow: 500ms-1s (store), 100ms-500ms (read)
- Red: >1s (store), >500ms (read)

**Resource Usage**:
- Green: <70%
- Yellow: 70-90%
- Red: >90%

**Compression**:
- Red: <2x (poor compression)
- Yellow: 2-5x (acceptable)
- Green: 5-10x (good)
- Blue: >10x (excellent)

---

## 🔔 Alert Integration

These dashboards expose metrics for Prometheus alerting rules. Example alerts:

```yaml
groups:
  - name: wool_storage_alerts
    interval: 30s
    rules:
      - alert: WoolNodeDown
        expr: up{job=~"wool-node.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Wool node {{ $labels.node_id }} is down"
          description: "Node has been unreachable for >1 minute"

      - alert: WoolHighLatency
        expr: wool_storage_latency_seconds{operation="store", quantile="0.95"} > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High store latency on {{ $labels.node_id }}"
          description: "p95 latency is {{ $value }}s (threshold: 1s)"

      - alert: WoolDiskSpaceLow
        expr: (wool_storage_bytes_total / wool_storage_capacity_bytes) > 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space on {{ $labels.node_id }}"
          description: "Disk is {{ $value | humanizePercentage }} full"

      - alert: WoolReplicationLag
        expr: wool_replication_lag_seconds > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High replication lag on {{ $labels.node_id }}"
          description: "Lag is {{ $value }}s behind primary"
```

---

## 🎯 Monitoring Best Practices

### Daily Checks
1. **Cluster Overview** - Ensure all nodes healthy
2. **Performance** - Check p95 latency <1s
3. **Storage** - Monitor disk usage <85%

### Weekly Reviews
1. **Performance** - Analyze trends, identify bottlenecks
2. **Storage** - Capacity planning based on growth rate
3. **Resources** - Optimize CPU/memory limits

### Monthly Analysis
1. **Versioning** - Review merge conflict rates
2. **Compression** - Optimize algorithms by content type
3. **Resources** - Right-size Kubernetes resources

### SLA Tracking
- **Availability**: Cluster uptime (target: 99.9%)
- **Latency**: p95 store <1s, read <100ms
- **Throughput**: >1000 ops/sec sustained
- **Replication**: 3x factor maintained

---

## 🛠️ Customization

### Add Custom Panels

All dashboards are editable. To add a panel:

1. Open dashboard
2. Click **Add panel** (top right)
3. Select **Add a new panel**
4. Configure query:
   ```promql
   # Example: custom metric
   wool_custom_metric{node_id="wool-1"}
   ```
5. Choose visualization type
6. Click **Apply**
7. **Save dashboard** (floppy disk icon)

### Create Dashboard Snapshots

For sharing:

1. Open dashboard
2. Click **Share** (top right)
3. Select **Snapshot**
4. Choose expiration (1 hour, 1 day, never)
5. Click **Publish to snapshots.raintank.io**
6. Share URL

### Export Modified Dashboards

After customization:

1. Click **Dashboard settings** (gear icon)
2. Select **JSON Model**
3. Click **Copy to Clipboard**
4. Save to file: `custom-dashboard.json`

---

## 📊 Dashboard Variables (Future Enhancement)

**Planned variables** for filtering:

- `$node` - Select specific node
- `$environment` - Dev/staging/prod
- `$cluster` - Multi-cluster support
- `$time_range` - Quick time range selector

**Usage**:
```promql
# Filter by node variable
wool_storage_operations_total{node_id=~"$node"}

# Filter by environment
wool_storage_operations_total{environment="$environment"}
```

---

## 🔗 Related Documentation

- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - Full deployment guide
- [prometheus.yml](../prometheus.yml) - Prometheus configuration
- [kubernetes/monitoring.yaml](../kubernetes/monitoring.yaml) - K8s monitoring stack

---

## 📝 Dashboard Versions

**Version**: 1.0
**Grafana Version**: 9.0+
**Prometheus Version**: 2.40+
**Date**: November 2025

**Compatibility**:
- ✅ Grafana 9.x, 10.x
- ✅ Prometheus 2.x
- ✅ Wool Storage v1.0+

---

## 💡 Tips & Tricks

### Performance Optimization

1. **Reduce query load**:
   - Increase refresh interval for non-critical dashboards (30s → 1m)
   - Use recording rules for expensive queries

2. **Dashboard performance**:
   - Limit time range (1h faster than 24h)
   - Reduce panel count (<20 per dashboard)
   - Use `rate()` instead of `increase()` for counters

3. **Prometheus optimization**:
   - Set appropriate retention (15d default)
   - Use remote write for long-term storage
   - Enable query caching

### Troubleshooting

**Dashboard shows "No data"**:
- Check Prometheus datasource configured correctly
- Verify wool nodes exposing `/metrics` endpoint
- Check Prometheus scraping targets: `http://localhost:9090/targets`

**Panels show "N/A"**:
- Metric name may have changed (check Prometheus)
- Time range may be too narrow
- Node may not be exporting that metric

**High Grafana CPU usage**:
- Too many dashboards open
- Refresh rate too aggressive (10s → 30s)
- Complex queries (simplify PromQL)

---

**Author**: Claude Code
**Date**: November 18, 2025
**Status**: Production Ready
