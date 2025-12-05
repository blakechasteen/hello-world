# HoloLoom Phase 3 - Production Deployment

**Complete production deployment infrastructure for Phase 3 Adaptive Learning System**

This directory contains everything needed to deploy Phase 3 adaptive learning to production with comprehensive monitoring, alerting, and validation.

## 📦 What's Included

- **Docker Compose stack** - Prometheus + Grafana monitoring
- **Prometheus configuration** - Metrics scraping and alerting rules
- **Grafana dashboard** - 11-panel visualization
- **Alert scripts** - Slack and email notifications
- **Validation script** - Pre-deployment verification (10 checks)
- **Production example** - Complete working application

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Required
- Python 3.8+
- Docker + Docker Compose
- Git

# Optional (for alerts)
- Slack workspace with webhook
- SMTP email account
```

### 2. Clone and Setup

```bash
cd deployment

# Create data directories
mkdir -p data/{logs,patterns,validation,reports}

# Set environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Monitoring Stack

```bash
# Start Prometheus + Grafana
docker-compose up -d

# Verify services
docker-compose ps

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### 4. Validate Deployment

```bash
# Run all 10 validation checks
python scripts/validate_production.py --data-dir ./data

# Expected output: 10/10 checks passed ✅
```

### 5. Run Production Example

```bash
# Set environment variables (optional)
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK"
export SMTP_USER="alerts@yourcompany.com"
export SMTP_PASSWORD="your-password"
export SMTP_TO="team@yourcompany.com"

# Start production application
python production_example.py

# Watch logs
tail -f data/logs/production_example.log
```

## 📊 Monitoring

### Prometheus Metrics

**Exposed Metrics** (exported every 30s):
```
moonshot_accuracy{complexity="overall"}       # Overall accuracy (0-1)
moonshot_accuracy{complexity="trivial"}       # Per-complexity accuracy
moonshot_accuracy{complexity="simple"}
moonshot_accuracy{complexity="complex"}
moonshot_accuracy{complexity="research"}

moonshot_queries_total                        # Total queries processed
moonshot_latency_ms                           # Average latency (ms)
moonshot_patterns_deployed                    # Patterns deployed to date
moonshot_patterns_discovered                  # Patterns discovered to date
moonshot_regressions_detected                 # Regressions detected to date
moonshot_validation_last_updated_timestamp    # Last validation time
```

**View Metrics**:
```bash
# Prometheus UI
open http://localhost:9090

# Query examples:
# - moonshot_accuracy{complexity="overall"}
# - rate(moonshot_queries_total[5m])
# - histogram_quantile(0.95, moonshot_latency_ms)
```

### Grafana Dashboard

**Access**: http://localhost:3000/d/hololoom-phase3

**11 Panels**:
1. Overall Accuracy (gauge, 0-100%)
2. Accuracy Over Time (time series)
3. Total Queries (stat)
4. Patterns Deployed (stat with trend)
5. Regressions Detected (stat with thresholds)
6. Avg Latency (stat with thresholds)
7. Query Rate (time series)
8. Accuracy by Complexity (stacked bars)
9. Classification Latency (time series)
10. Pattern Deployment Trends (step graph)
11. Regression Detection (step graph)

**First-Time Setup**:
1. Login: admin/admin (change password)
2. Dashboard auto-loads on first access
3. Set time range: Last 24 hours
4. Enable auto-refresh: 30s interval

## 🔔 Alerting

### Configured Alerts (15 Total)

| Alert | Threshold | Severity | For | Action |
|-------|-----------|----------|-----|--------|
| **ClassifierAccuracyLow** | <90% | Warning | 5m | Review logs |
| **ClassifierAccuracyCritical** | <85% | Critical | 2m | Page on-call |
| **RegressionDetected** | >0 in 1h | Warning | - | Investigate |
| **HighRegressionRate** | >3 in 24h | Critical | - | Page on-call |
| **HighClassificationLatency** | >200ms | Warning | 5m | Check resources |
| **CriticalClassificationLatency** | >500ms | Critical | 2m | Immediate action |
| **NoQueries** | 0 in 5m | Critical | 5m | Check service |
| **ClassifierDown** | up=0 | Critical | 1m | Restart service |

See `prometheus/prometheus_rules.yml` for complete list.

### Slack Integration

**Setup**:
1. Create Slack incoming webhook: https://api.slack.com/messaging/webhooks
2. Copy webhook URL
3. Set environment variable:
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T00/B00/XXX"
   ```

**Alert Format**:
```
🚨 Classifier Regression Detected

Current accuracy: 85.3%
Baseline accuracy: 95.0%
Drop: 9.7% (threshold: 2.0%)

Affected complexities:
  • complex
  • research

[View Dashboard] [View Metrics]
```

**Test Alert**:
```bash
python scripts/send_slack_alert.py '{
  "title": "Test Alert",
  "current_accuracy": 0.85,
  "baseline_accuracy": 0.95,
  "drop_percentage": 0.10,
  "affected_complexity": ["test"],
  "severity": "WARNING",
  "timestamp": "2025-11-21T10:00:00"
}'
```

### Email Alerts

**Setup**:
1. Configure SMTP credentials:
   ```bash
   export SMTP_HOST="smtp.gmail.com"
   export SMTP_PORT=587
   export SMTP_USER="alerts@yourcompany.com"
   export SMTP_PASSWORD="your-app-password"
   export SMTP_TO="team@yourcompany.com,oncall@yourcompany.com"
   ```

2. For Gmail: Use App Passwords (https://support.google.com/accounts/answer/185833)

**Alert Format**:
- Plain text + HTML multipart
- Severity-based colors (red/amber)
- Clickable dashboard/metrics links
- Complete alert details

**Test Alert**:
```bash
python scripts/send_email_alert.py '{
  "title": "Test Email Alert",
  "current_accuracy": 0.85,
  "baseline_accuracy": 0.95,
  "drop_percentage": 0.10,
  "affected_complexity": ["test"],
  "severity": "WARNING",
  "timestamp": "2025-11-21T10:00:00"
}'
```

## ✅ Validation

The validation script performs 10 comprehensive checks before deployment:

```bash
python scripts/validate_production.py --data-dir ./data
```

**Checks**:
1. ✅ Directory structure exists
2. ✅ Classifier can be initialized
3. ✅ Query classification works
4. ✅ JSONL logging works
5. ✅ Pattern miner works
6. ✅ Continuous validator works
7. ✅ Adaptive updater works
8. ✅ Performance reporter works
9. ✅ Background learning can start/stop
10. ✅ Learning statistics can be retrieved

**Exit Codes**:
- `0` - All checks passed
- `1` - One or more checks failed

**CI/CD Integration**:
```yaml
# .github/workflows/deploy.yml
- name: Validate Phase 3 Deployment
  run: |
    python deployment/scripts/validate_production.py
    if [ $? -ne 0 ]; then
      echo "Validation failed!"
      exit 1
    fi
```

## 🏭 Production Application

The `production_example.py` demonstrates a complete production-ready application with:

**Features**:
- Adaptive learning classifier with background learning (1-hour intervals)
- Query classification workload simulation
- Prometheus metrics export (every 5 queries)
- Regression checking with alerts (every 10 queries)
- Learning statistics reporting (every 20 queries)
- Graceful shutdown (SIGINT/SIGTERM handling)
- Comprehensive logging

**Run**:
```bash
python production_example.py
```

**Output**:
```
2025-11-21 10:00:00 - INFO - Initializing HoloLoom Phase 3 Production System
2025-11-21 10:00:01 - INFO - ✅ Initialization complete
2025-11-21 10:00:02 - INFO - [1] Query: "hi"
2025-11-21 10:00:02 - INFO -      Complexity: trivial (expected: trivial)
2025-11-21 10:00:02 - INFO -      Confidence: 98.5%
2025-11-21 10:00:02 - INFO -      Latency: 142.3ms
2025-11-21 10:00:02 - INFO -      Tier: regex
```

**Systemd Service** (optional):
```ini
# /etc/systemd/system/hololoom.service
[Unit]
Description=HoloLoom Phase 3 Production Service
After=network.target docker.service

[Service]
Type=simple
User=hololoom
WorkingDirectory=/opt/hololoom/deployment
Environment="PYTHONPATH=/opt/hololoom"
Environment="SLACK_WEBHOOK_URL=YOUR_WEBHOOK"
ExecStart=/usr/bin/python3 production_example.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable hololoom
sudo systemctl start hololoom
sudo systemctl status hololoom
```

## 📁 Directory Structure

```
deployment/
├── README.md                           # This file
├── docker-compose.yml                  # Prometheus + Grafana stack
├── production_example.py               # Complete production application
│
├── prometheus/
│   ├── prometheus_phase3.yml          # Prometheus config
│   └── prometheus_rules.yml           # Alert rules (15 alerts)
│
├── grafana/
│   ├── provisioning/
│   │   └── dashboards/
│   │       └── dashboards.yml         # Dashboard auto-provisioning
│   └── dashboards/
│       └── hololoom_phase3_dashboard.json  # 11-panel dashboard
│
├── scripts/
│   ├── validate_production.py         # Pre-deployment validation (10 checks)
│   ├── send_slack_alert.py           # Slack webhook integration
│   └── send_email_alert.py           # SMTP email integration
│
└── data/                               # Runtime data (gitignored)
    ├── logs/                          # Classification logs (JSONL)
    ├── patterns/                      # Discovered patterns
    ├── validation/                    # Validation results
    └── reports/                       # Daily/weekly reports
```

## 🔧 Configuration

### Environment Variables

**Required**:
```bash
HOLOLOOM_DATA_DIR=/path/to/data        # Data directory
```

**Optional (Alerting)**:
```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
SLACK_CHANNEL=#alerts                   # Default: #general

# Email
SMTP_HOST=smtp.gmail.com               # SMTP server
SMTP_PORT=587                          # SMTP port
SMTP_USER=alerts@company.com           # SMTP username
SMTP_PASSWORD=your-password            # SMTP password
SMTP_FROM=alerts@company.com           # From address
SMTP_TO=team@company.com,oncall@company.com  # Recipients (comma-separated)

# Monitoring
PROMETHEUS_PORT=9090                   # Prometheus port
GRAFANA_PORT=3000                     # Grafana port
GRAFANA_ADMIN_PASSWORD=secure-password # Grafana admin password
```

### Classifier Configuration

Edit `production_example.py`:

```python
classifier = AdaptiveMoonshotClassifier(
    enable_semantic_tier=True,          # Use semantic tier
    enable_adaptive_learning=True,      # Enable learning
    data_dir=Path("./data"),           # Data directory
    background_learning=True,           # Background learning loop
    learning_update_interval=3600.0    # Update every 1 hour (seconds)
)
```

## 🐛 Troubleshooting

### Docker Services Won't Start

```bash
# Check port conflicts
netstat -tuln | grep -E '(3000|9090)'

# Check Docker logs
docker-compose logs prometheus
docker-compose logs grafana

# Restart services
docker-compose restart
```

### Prometheus Not Scraping Metrics

```bash
# Check Prometheus targets
open http://localhost:9090/targets

# Expected: hololoom-phase3 (1/1 up)

# Check metrics file
cat /tmp/hololoom_phase3_metrics.prom

# Should contain: moonshot_accuracy, moonshot_queries_total, etc.
```

### Grafana Dashboard Not Loading

```bash
# Check provisioning
docker exec -it hololoom-grafana ls /var/lib/grafana/dashboards

# Expected: hololoom_phase3_dashboard.json

# Check Grafana logs
docker-compose logs grafana | grep -i error

# Reimport dashboard manually
# Settings → Dashboards → Import → Upload JSON
```

### Alerts Not Firing

```bash
# Check Prometheus rules
open http://localhost:9090/rules

# Expected: hololoom_phase3_adaptive_learning (15 rules)

# Check alertmanager (if configured)
open http://localhost:9093

# Test alert manually
python scripts/send_slack_alert.py '{"title": "Test", ...}'
```

### Validation Checks Failing

```bash
# Run with verbose output
python scripts/validate_production.py --data-dir ./data 2>&1 | tee validation.log

# Check data directory permissions
ls -la data/

# Check Python environment
python -c "from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier"

# Reinstall dependencies
pip install -e .
```

## 📚 Documentation

- **[PHASE_3_PRODUCTION_DEPLOYMENT.md](PHASE_3_PRODUCTION_DEPLOYMENT.md)** - Comprehensive deployment guide (1000+ lines)
- **[PHASE_3_DOCUMENTATION.md](../PHASE_3_DOCUMENTATION.md)** - Phase 3 architecture and API reference
- **[PHASE_3_PROGRESS.md](../PHASE_3_PROGRESS.md)** - Implementation timeline and progress

## 🔗 Useful Links

**Monitoring**:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Prometheus Targets: http://localhost:9090/targets
- Prometheus Rules: http://localhost:9090/rules

**Documentation**:
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/
- Slack Webhooks: https://api.slack.com/messaging/webhooks

## 🎯 Next Steps

1. **Customize workload** - Replace example queries with your production data
2. **Tune alert thresholds** - Adjust based on your accuracy targets
3. **Add more metrics** - Export custom metrics from your application
4. **Set up Alertmanager** - Route alerts to PagerDuty, Opsgenie, etc.
5. **Enable SSL/TLS** - Secure Prometheus and Grafana endpoints
6. **Scale horizontally** - Deploy multiple classifiers with load balancing
7. **Backup data** - Regularly backup `data/` directory

## ✅ Success Checklist

Before going live, verify:

- [ ] Docker Compose stack running (`docker-compose ps`)
- [ ] Prometheus scraping metrics (check `/targets`)
- [ ] Grafana dashboard loading (visit http://localhost:3000)
- [ ] All 10 validation checks passing
- [ ] Slack alerts working (test with script)
- [ ] Email alerts working (test with script)
- [ ] Production example running without errors
- [ ] Logs being written to `data/logs/`
- [ ] Patterns being discovered (`data/patterns/`)
- [ ] Background learning active (check logs)

## 🎉 You're Ready!

Phase 3 Adaptive Learning is now ready for production deployment with:
- ✅ Complete monitoring infrastructure
- ✅ Real-time alerting (Slack + Email)
- ✅ Comprehensive validation
- ✅ Production-ready example
- ✅ Complete documentation

**Questions?** See [PHASE_3_PRODUCTION_DEPLOYMENT.md](PHASE_3_PRODUCTION_DEPLOYMENT.md) for detailed guide.

---

**HoloLoom Phase 3** - November 2025
**Status**: ✅ Production Ready
