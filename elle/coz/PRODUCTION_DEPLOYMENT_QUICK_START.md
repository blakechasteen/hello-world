# COZ Daily Brief - Production Deployment Quick Start

**Created**: 2025-11-22
**Status**: Production Ready
**Time to Deploy**: 10-15 minutes

Complete guide for deploying COZ daily brief automation to production with monitoring.

---

## Prerequisites

- Python 3.11+
- Git repository access
- Production server (Linux/Docker)
- COZ data files (CSV parsers)

**Optional**:
- Prometheus for monitoring
- Slack webhook for alerts
- Email SMTP server

---

## Option 1: systemd Service (Linux Production)

### 1. Install Dependencies

```bash
cd /path/to/mythRL
pip install schedule  # Only required dependency
```

### 2. Create systemd Service

**File**: `/etc/systemd/system/coz-daily-brief.service`

```ini
[Unit]
Description=COZ Daily Brief Automation
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/mythRL
Environment="PYTHONPATH=/path/to/mythRL"
Environment="ELLE_ENABLE_REFINEMENT=false"  # Set true for HoloLoom refinement
ExecStart=/usr/bin/python3 elle/coz/daily_brief_automation.py --schedule "09:00"
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 3. Enable and Start

```bash
# Enable service
sudo systemctl daemon-reload
sudo systemctl enable coz-daily-brief

# Start service
sudo systemctl start coz-daily-brief

# Check status
sudo systemctl status coz-daily-brief

# View logs
sudo journalctl -u coz-daily-brief -f
```

### 4. Verify Health

```bash
# Check automation health
PYTHONPATH=. python elle/coz/health_check.py

# Should see: Overall Status: HEALTHY
```

**Complete!** Daily briefs will generate at 9 AM daily.

---

## Option 2: Docker Deployment

### 1. Create Dockerfile

**File**: `Dockerfile.coz-automation`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir schedule

# Copy application
COPY elle/ /app/elle/
COPY OPTION_2_AUTOMATION_COMPLETE.md /app/

# Create output directory
RUN mkdir -p /app/daily_briefs

# Run automation
CMD ["python", "elle/coz/daily_brief_automation.py", "--schedule", "09:00"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  coz-automation:
    build:
      context: .
      dockerfile: Dockerfile.coz-automation
    container_name: coz-daily-brief
    volumes:
      - ./daily_briefs:/app/daily_briefs
      - ./elle/coz:/app/elle/coz:ro  # Mount COZ data
    environment:
      - ELLE_ENABLE_REFINEMENT=false
      - ELLE_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "elle/coz/health_check.py", "--json"]
      interval: 1h
      timeout: 10s
      retries: 3
      start_period: 10s
```

### 3. Deploy

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f coz-automation

# Check health
docker-compose exec coz-automation python elle/coz/health_check.py
```

**Complete!** Container will auto-restart and generate briefs daily.

---

## Option 3: Kubernetes Deployment

### 1. Create ConfigMap

**File**: `k8s/coz-automation-config.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: coz-automation-config
data:
  ELLE_ENABLE_REFINEMENT: "false"
  ELLE_LOG_LEVEL: "INFO"
```

### 2. Create Deployment

**File**: `k8s/coz-automation-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coz-automation
spec:
  replicas: 1
  selector:
    matchLabels:
      app: coz-automation
  template:
    metadata:
      labels:
        app: coz-automation
    spec:
      containers:
      - name: coz-automation
        image: your-registry/coz-automation:latest
        envFrom:
        - configMapRef:
            name: coz-automation-config
        volumeMounts:
        - name: output
          mountPath: /app/daily_briefs
        - name: data
          mountPath: /app/elle/coz
          readOnly: true
        livenessProbe:
          exec:
            command:
            - python
            - elle/coz/health_check.py
            - --json
          initialDelaySeconds: 60
          periodSeconds: 3600
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: coz-briefs-pvc
      - name: data
        configMap:
          name: coz-data
```

### 3. Deploy

```bash
# Apply configuration
kubectl apply -f k8s/coz-automation-config.yaml
kubectl apply -f k8s/coz-automation-deployment.yaml

# Check status
kubectl get pods -l app=coz-automation
kubectl logs -f deployment/coz-automation
```

---

## Monitoring Integration

### Prometheus Metrics

**1. Expose Metrics Endpoint**

Add to automation script or create separate metrics server:

```python
# metrics_server.py
from flask import Flask, Response
from elle.coz.health_check import HealthCheck

app = Flask(__name__)
health_check = HealthCheck()

@app.route('/metrics')
def metrics():
    return Response(
        health_check.export_prometheus(),
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
```

**2. Configure Prometheus**

**File**: `prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'coz-automation'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5m
```

**3. Create Alerting Rules**

**File**: `prometheus_alerts.yml`

```yaml
groups:
  - name: coz_automation
    interval: 5m
    rules:
      - alert: COZAutomationUnhealthy
        expr: coz_automation_health_status < 1
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "COZ automation unhealthy"
          description: "Health status: {{ $value }}"

      - alert: COZNoRecentBrief
        expr: coz_automation_check_status{check="recent_briefs"} == 0
        for: 24h
        labels:
          severity: warning
        annotations:
          summary: "No recent COZ brief generated"
```

### Grafana Dashboard

**Import dashboard JSON**:

```json
{
  "dashboard": {
    "title": "COZ Daily Brief Automation",
    "panels": [
      {
        "title": "Health Status",
        "targets": [{
          "expr": "coz_automation_health_status"
        }],
        "type": "stat"
      },
      {
        "title": "Failed Checks",
        "targets": [{
          "expr": "coz_automation_checks_failed"
        }],
        "type": "graph"
      }
    ]
  }
}
```

---

## Health Checks

### Manual Health Check

```bash
# Human-readable
PYTHONPATH=. python elle/coz/health_check.py

# JSON format
PYTHONPATH=. python elle/coz/health_check.py --json

# Prometheus format
PYTHONPATH=. python elle/coz/health_check.py --prometheus
```

### Automated Monitoring

**Add to cron** (checks every hour):

```bash
# Health check alert
0 * * * * cd /path/to/mythRL && PYTHONPATH=. python elle/coz/health_check.py || echo "COZ automation unhealthy" | mail -s "ALERT" admin@example.com
```

**systemd timer** (checks every hour):

**File**: `/etc/systemd/system/coz-health-check.timer`

```ini
[Unit]
Description=COZ Health Check Timer
Requires=coz-health-check.service

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

**File**: `/etc/systemd/system/coz-health-check.service`

```ini
[Unit]
Description=COZ Health Check

[Service]
Type=oneshot
WorkingDirectory=/path/to/mythRL
Environment="PYTHONPATH=/path/to/mythRL"
ExecStart=/usr/bin/python3 elle/coz/health_check.py
```

```bash
sudo systemctl enable coz-health-check.timer
sudo systemctl start coz-health-check.timer
```

---

## Troubleshooting

### Issue: Service won't start

**Solution**:
```bash
# Check logs
sudo journalctl -u coz-daily-brief -n 50

# Common issues:
# - PYTHONPATH not set → Add to service file
# - Permissions → Change User= in service file
# - Missing schedule module → pip install schedule
```

### Issue: No briefs generated

**Solution**:
```bash
# Check parser status
PYTHONPATH=. python elle/coz/health_check.py

# Test manual generation
PYTHONPATH=. python elle/coz/daily_brief_automation.py --once

# Check output directory permissions
ls -la ./daily_briefs
```

### Issue: Health check failing

**Solution**:
```bash
# Run health check with details
PYTHONPATH=. python elle/coz/health_check.py --json | jq '.checks'

# Common failures:
# - recent_briefs → Normal if just deployed
# - hololoom → Optional, doesn't affect core functionality
# - parsers → Check CSV files exist
```

---

## Performance Tuning

### Reduce Latency

**Disable refinement** (2000ms → 520ms):
```bash
# Environment variable
export ELLE_ENABLE_REFINEMENT=false

# Or command-line flag
python elle/coz/daily_brief_automation.py --once --no-refinement
```

### Reduce Memory Usage

**Python 3.11 optimizations**:
- Uses ~50MB base memory
- ~150MB during refinement (if enabled)

### Reduce Disk Usage

**Log rotation** (systemd):
```bash
# Limit journal size
sudo journalctl --vacuum-size=100M
sudo journalctl --vacuum-time=7d
```

---

## Security Considerations

### Environment Variables

**Never commit**:
- API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
- SMTP passwords
- Database credentials

**Use**:
- Environment files (.env)
- Secrets management (Kubernetes secrets, AWS Secrets Manager)
- systemd EnvironmentFile=

### File Permissions

```bash
# Restrict service file
sudo chmod 644 /etc/systemd/system/coz-daily-brief.service

# Restrict output directory
chmod 700 ./daily_briefs
```

### Network Security

- Run on internal network only
- Use firewall rules to restrict access
- Enable TLS for any external metrics endpoints

---

## Backup and Recovery

### Backup Strategy

**Daily briefs**:
```bash
# Backup to S3 (example)
aws s3 sync ./daily_briefs s3://your-bucket/coz-briefs/
```

**Database**:
```bash
# If using persistent storage
rsync -av ./daily_briefs /backup/location/
```

### Recovery

**Restore from backup**:
```bash
# Restore briefs
aws s3 sync s3://your-bucket/coz-briefs/ ./daily_briefs

# Restart service
sudo systemctl restart coz-daily-brief
```

---

## Production Checklist

- [ ] Dependencies installed (`pip install schedule`)
- [ ] Service file created and enabled
- [ ] Output directory exists and is writable
- [ ] Health check passing
- [ ] Logs accessible (journalctl or Docker logs)
- [ ] Monitoring configured (Prometheus/Grafana)
- [ ] Alerts configured (email/Slack)
- [ ] Backup strategy implemented
- [ ] Documentation updated with deployment details

---

## Next Steps

1. **Week 1**: Deploy to staging environment
2. **Week 2**: Monitor for 7 days, tune scheduling
3. **Week 3**: Deploy to production
4. **Week 4**: Add email delivery (optional)
5. **Week 5+**: A/B test refinement quality, optimize

---

## Support

**Health Check**: `python elle/coz/health_check.py`
**Documentation**: `OPTION_2_AUTOMATION_COMPLETE.md`
**Logs**: `sudo journalctl -u coz-daily-brief -f`

**Common Commands**:
```bash
# Start service
sudo systemctl start coz-daily-brief

# Stop service
sudo systemctl stop coz-daily-brief

# Restart service
sudo systemctl restart coz-daily-brief

# View status
sudo systemctl status coz-daily-brief

# Run once (manual test)
PYTHONPATH=. python elle/coz/daily_brief_automation.py --once

# Check health
PYTHONPATH=. python elle/coz/health_check.py
```

---

**Deployment Time**: 10-15 minutes
**Maintenance**: ~5 min/week (monitoring logs)
**Uptime Target**: 99.9% (systemd auto-restart)

✅ **Production Ready** - Deploy with confidence!
