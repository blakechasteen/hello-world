# Phase 3: Production Deployment Guide

**Date**: November 13, 2025
**Status**: Production Ready
**Estimated Deployment Time**: 2-4 hours

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Prometheus Configuration](#prometheus-configuration)
4. [Grafana Dashboard Setup](#grafana-dashboard-setup)
5. [Slack Integration](#slack-integration)
6. [Email Alerting](#email-alerting)
7. [Production Integration](#production-integration)
8. [Validation & Testing](#validation--testing)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

### Prerequisites

- ✅ Phase 3 code complete and tested (13/13 tests passing)
- ✅ Python 3.8+ environment
- ✅ Production server access
- ✅ Docker installed (for Prometheus + Grafana)
- ✅ Slack workspace admin access (for webhook)
- ✅ SMTP server credentials (for email alerts)

### Environment Variables

Create a `.env` file in your production environment:

```bash
# HoloLoom Phase 3 Configuration
HOLOLOOM_DATA_DIR=/var/lib/hololoom/data
HOLOLOOM_LOGS_DIR=/var/log/hololoom
HOLOLOOM_REPORTS_DIR=/var/lib/hololoom/reports

# Prometheus
PROMETHEUS_PORT=9090
PROMETHEUS_METRICS_PATH=/var/lib/prometheus/textfile_collector

# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=<secure-password>

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#hololoom-alerts

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@yourcompany.com
SMTP_PASSWORD=<app-specific-password>
SMTP_FROM=alerts@yourcompany.com
SMTP_TO=team@yourcompany.com,ops@yourcompany.com

# Adaptive Learning
ADAPTIVE_LEARNING_ENABLED=true
BACKGROUND_LEARNING_ENABLED=true
LEARNING_UPDATE_INTERVAL=3600  # 1 hour
REGRESSION_THRESHOLD=0.02  # 2%
MIN_PATTERN_PRECISION=0.95  # 95%
MIN_PATTERN_SUPPORT=10
```

### Directory Structure

Create required directories:

```bash
# Data directories
sudo mkdir -p /var/lib/hololoom/data/{logs,patterns,validation,reports}
sudo mkdir -p /var/log/hololoom

# Prometheus
sudo mkdir -p /var/lib/prometheus/textfile_collector

# Permissions
sudo chown -R hololoom:hololoom /var/lib/hololoom
sudo chown -R hololoom:hololoom /var/log/hololoom
sudo chmod 755 /var/lib/prometheus/textfile_collector
```

---

## Infrastructure Setup

### Option 1: Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus_rules.yml:/etc/prometheus/prometheus_rules.yml
      - /var/lib/prometheus/textfile_collector:/var/lib/prometheus/textfile_collector
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

Deploy:

```bash
docker-compose up -d
```

### Option 2: System Services

Install Prometheus and Grafana as system services (see respective documentation).

---

## Prometheus Configuration

### 1. Create Prometheus Config

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'hololoom-production'
    environment: 'prod'

# Load alerting rules
rule_files:
  - 'prometheus_rules.yml'

# Scrape configurations
scrape_configs:
  # HoloLoom Metrics (textfile collector)
  - job_name: 'hololoom'
    static_configs:
      - targets: ['localhost:9090']
    file_sd_configs:
      - files:
          - '/var/lib/prometheus/textfile_collector/*.prom'
        refresh_interval: 60s

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

# Alertmanager configuration (optional)
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

### 2. Create Alert Rules

Create `prometheus_rules.yml`:

```yaml
groups:
  - name: hololoom_adaptive_learning
    interval: 30s
    rules:
      # Accuracy alerts
      - alert: ClassifierAccuracyLow
        expr: moonshot_accuracy{complexity="overall"} < 0.90
        for: 5m
        labels:
          severity: warning
          component: adaptive_learning
        annotations:
          summary: "Classifier accuracy below 90%"
          description: "Overall accuracy is {{ $value | humanizePercentage }}. Expected >90%."

      - alert: ClassifierAccuracyCritical
        expr: moonshot_accuracy{complexity="overall"} < 0.85
        for: 2m
        labels:
          severity: critical
          component: adaptive_learning
        annotations:
          summary: "Classifier accuracy critically low"
          description: "Overall accuracy is {{ $value | humanizePercentage }}. Expected >85%."

      # Per-complexity alerts
      - alert: ComplexityAccuracyLow
        expr: moonshot_accuracy < 0.80
        for: 5m
        labels:
          severity: warning
          component: adaptive_learning
        annotations:
          summary: "Low accuracy for {{ $labels.complexity }} queries"
          description: "Accuracy for {{ $labels.complexity }} is {{ $value | humanizePercentage }}."

      # Regression alerts
      - alert: RegressionDetected
        expr: increase(moonshot_regressions_detected[1h]) > 0
        labels:
          severity: warning
          component: adaptive_learning
        annotations:
          summary: "Regression detected"
          description: "{{ $value }} regressions detected in last hour."

      - alert: HighRegressionRate
        expr: increase(moonshot_regressions_detected[24h]) > 3
        labels:
          severity: critical
          component: adaptive_learning
        annotations:
          summary: "High regression rate"
          description: "{{ $value }} regressions in last 24 hours. Investigation required."

      # Latency alerts
      - alert: HighLatency
        expr: moonshot_latency_ms > 200
        for: 5m
        labels:
          severity: warning
          component: adaptive_learning
        annotations:
          summary: "High classification latency"
          description: "Average latency is {{ $value }}ms. Expected <150ms."

      # Pattern deployment alerts
      - alert: NoPatternDeployment
        expr: increase(moonshot_patterns_deployed[7d]) == 0
        labels:
          severity: info
          component: adaptive_learning
        annotations:
          summary: "No patterns deployed in 7 days"
          description: "No new patterns have been deployed. System may not be learning."

      # Query volume alerts
      - alert: LowQueryVolume
        expr: rate(moonshot_queries_total[5m]) < 1
        for: 10m
        labels:
          severity: warning
          component: adaptive_learning
        annotations:
          summary: "Low query volume"
          description: "Query rate is {{ $value }}/s. Expected >1/s in production."
```

### 3. Metrics Export Script

Create `/usr/local/bin/export_hololoom_metrics.py`:

```python
#!/usr/bin/env python3
"""
Export HoloLoom Phase 3 metrics to Prometheus textfile collector.
Run this script every minute via cron.
"""

import sys
import os
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, '/opt/hololoom')

from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier
from HoloLoom.routing.learning import PerformanceReporter

METRICS_PATH = Path("/var/lib/prometheus/textfile_collector/moonshot.prom")

def export_metrics():
    """Export current metrics to Prometheus."""
    try:
        # Get classifier instance (singleton)
        classifier = AdaptiveMoonshotClassifier.get_instance()

        # Get reporter
        reporter = PerformanceReporter(
            validator=classifier.validator,
            updater=classifier.updater,
            pattern_miner=classifier.pattern_miner
        )

        # Export metrics
        metrics = reporter.export_prometheus_metrics()

        # Write to textfile collector
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, 'w') as f:
            f.write(metrics)

        print(f"Metrics exported successfully to {METRICS_PATH}")

    except Exception as e:
        print(f"Error exporting metrics: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    export_metrics()
```

Add to crontab:

```bash
# Export HoloLoom metrics every minute
* * * * * /usr/local/bin/export_hololoom_metrics.py >> /var/log/hololoom/metrics_export.log 2>&1
```

---

## Grafana Dashboard Setup

### 1. Create Dashboard Provisioning

Create `grafana/provisioning/dashboards/dashboard.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'HoloLoom'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

### 2. Create Dashboard JSON

Create `grafana/dashboards/hololoom_phase3.json`:

```json
{
  "dashboard": {
    "title": "HoloLoom Phase 3 - Adaptive Learning",
    "tags": ["hololoom", "phase3", "adaptive-learning"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Overall Accuracy",
        "type": "graph",
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "moonshot_accuracy{complexity=\"overall\"}",
            "legendFormat": "Overall Accuracy",
            "refId": "A"
          }
        ],
        "yaxes": [
          {"format": "percentunit", "min": 0, "max": 1}
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {"params": [0.90], "type": "lt"},
              "operator": {"type": "and"},
              "query": {"params": ["A", "5m", "now"]},
              "reducer": {"params": [], "type": "avg"},
              "type": "query"
            }
          ],
          "executionErrorState": "alerting",
          "frequency": "1m",
          "handler": 1,
          "name": "Overall Accuracy Alert",
          "noDataState": "no_data",
          "notifications": []
        }
      },
      {
        "id": 2,
        "title": "Accuracy by Complexity",
        "type": "graph",
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "moonshot_accuracy",
            "legendFormat": "{{complexity}}",
            "refId": "A"
          }
        ],
        "yaxes": [
          {"format": "percentunit", "min": 0, "max": 1}
        ]
      },
      {
        "id": 3,
        "title": "Query Rate",
        "type": "graph",
        "gridPos": {"x": 0, "y": 8, "w": 8, "h": 6},
        "targets": [
          {
            "expr": "rate(moonshot_queries_total[5m])",
            "legendFormat": "Queries/sec",
            "refId": "A"
          }
        ],
        "yaxes": [
          {"format": "ops", "min": 0}
        ]
      },
      {
        "id": 4,
        "title": "Classification Latency",
        "type": "graph",
        "gridPos": {"x": 8, "y": 8, "w": 8, "h": 6},
        "targets": [
          {
            "expr": "moonshot_latency_ms",
            "legendFormat": "Latency (ms)",
            "refId": "A"
          }
        ],
        "yaxes": [
          {"format": "ms", "min": 0}
        ]
      },
      {
        "id": 5,
        "title": "Total Queries",
        "type": "stat",
        "gridPos": {"x": 16, "y": 8, "w": 4, "h": 3},
        "targets": [
          {
            "expr": "moonshot_queries_total",
            "refId": "A"
          }
        ],
        "options": {
          "graphMode": "none",
          "colorMode": "value"
        }
      },
      {
        "id": 6,
        "title": "Patterns Deployed",
        "type": "stat",
        "gridPos": {"x": 20, "y": 8, "w": 4, "h": 3},
        "targets": [
          {
            "expr": "moonshot_patterns_deployed",
            "refId": "A"
          }
        ],
        "options": {
          "graphMode": "area",
          "colorMode": "value"
        }
      },
      {
        "id": 7,
        "title": "Regressions Detected",
        "type": "stat",
        "gridPos": {"x": 16, "y": 11, "w": 4, "h": 3},
        "targets": [
          {
            "expr": "moonshot_regressions_detected",
            "refId": "A"
          }
        ],
        "options": {
          "graphMode": "none",
          "colorMode": "value",
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"value": 0, "color": "green"},
              {"value": 1, "color": "yellow"},
              {"value": 3, "color": "red"}
            ]
          }
        }
      },
      {
        "id": 8,
        "title": "Current Accuracy",
        "type": "gauge",
        "gridPos": {"x": 20, "y": 11, "w": 4, "h": 3},
        "targets": [
          {
            "expr": "moonshot_accuracy{complexity=\"overall\"}",
            "refId": "A"
          }
        ],
        "options": {
          "showThresholdLabels": false,
          "showThresholdMarkers": true,
          "min": 0,
          "max": 1,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"value": 0, "color": "red"},
              {"value": 0.85, "color": "yellow"},
              {"value": 0.90, "color": "green"}
            ]
          }
        }
      }
    ],
    "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"}
  }
}
```

### 3. Access Grafana

1. Navigate to `http://localhost:3000`
2. Login (admin / password from `.env`)
3. Dashboard should auto-load
4. Configure alerts and notifications

---

## Slack Integration

### 1. Create Slack Webhook

1. Go to https://api.slack.com/apps
2. Create new app: "HoloLoom Alerts"
3. Enable Incoming Webhooks
4. Create webhook for `#hololoom-alerts` channel
5. Copy webhook URL to `.env`

### 2. Create Alert Handler

Create `/opt/hololoom/scripts/send_slack_alert.py`:

```python
#!/usr/bin/env python3
"""Send Slack alert for HoloLoom regression."""

import os
import sys
import requests
from datetime import datetime

SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL', '#hololoom-alerts')

def send_alert(alert_data):
    """Send formatted alert to Slack."""

    # Format message
    message = {
        "channel": SLACK_CHANNEL,
        "username": "HoloLoom Monitor",
        "icon_emoji": ":robot_face:",
        "text": f"🚨 *{alert_data['title']}*",
        "attachments": [
            {
                "color": "danger" if alert_data['severity'] == "CRITICAL" else "warning",
                "fields": [
                    {
                        "title": "Current Accuracy",
                        "value": f"{alert_data['current_accuracy']:.1%}",
                        "short": True
                    },
                    {
                        "title": "Baseline Accuracy",
                        "value": f"{alert_data['baseline_accuracy']:.1%}",
                        "short": True
                    },
                    {
                        "title": "Drop",
                        "value": f"{alert_data['drop_percentage']:.1%}",
                        "short": True
                    },
                    {
                        "title": "Severity",
                        "value": alert_data['severity'],
                        "short": True
                    },
                    {
                        "title": "Affected Complexities",
                        "value": ", ".join(alert_data['affected_complexity']),
                        "short": False
                    },
                    {
                        "title": "Time",
                        "value": alert_data['timestamp'],
                        "short": False
                    }
                ],
                "footer": "HoloLoom Phase 3 Adaptive Learning",
                "ts": int(datetime.now().timestamp())
            }
        ]
    }

    # Send to Slack
    response = requests.post(SLACK_WEBHOOK_URL, json=message)

    if response.status_code != 200:
        print(f"Error sending Slack alert: {response.text}", file=sys.stderr)
        return False

    return True

if __name__ == "__main__":
    # Example usage
    alert = {
        "title": "Classifier Regression Detected",
        "current_accuracy": 0.853,
        "baseline_accuracy": 0.950,
        "drop_percentage": 0.097,
        "affected_complexity": ["complex", "research"],
        "severity": "CRITICAL",
        "timestamp": datetime.now().isoformat()
    }

    send_alert(alert)
```

### 3. Integration in Learning Loop

Update `AdaptiveMoonshotClassifier._run_learning_cycle()`:

```python
# In _run_learning_cycle method
if validation_result.regression_detected:
    alert = self.validator.alerts[-1]

    # Format for Slack
    alert_data = {
        "title": "Classifier Regression Detected",
        "current_accuracy": alert.current_accuracy,
        "baseline_accuracy": alert.baseline_accuracy,
        "drop_percentage": alert.drop_percentage,
        "affected_complexity": alert.affected_complexity,
        "severity": alert.severity,
        "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat()
    }

    # Send alert
    import subprocess
    subprocess.run([
        "/opt/hololoom/scripts/send_slack_alert.py",
        json.dumps(alert_data)
    ])
```

---

## Email Alerting

### 1. Create Email Handler

Create `/opt/hololoom/scripts/send_email_alert.py`:

```python
#!/usr/bin/env python3
"""Send email alert for HoloLoom regression."""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_FROM = os.getenv('SMTP_FROM', SMTP_USER)
SMTP_TO = os.getenv('SMTP_TO', '').split(',')

def send_email(alert_data):
    """Send formatted email alert."""

    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"[{alert_data['severity']}] {alert_data['title']}"
    msg['From'] = SMTP_FROM
    msg['To'] = ', '.join(SMTP_TO)

    # Plain text body
    text = f"""
{alert_data['title']}
{'=' * len(alert_data['title'])}

Current accuracy: {alert_data['current_accuracy']:.1%}
Baseline accuracy: {alert_data['baseline_accuracy']:.1%}
Drop: {alert_data['drop_percentage']:.1%} (threshold: 2.0%)

Affected complexities:
{chr(10).join(f"  - {c}" for c in alert_data['affected_complexity'])}

Time: {alert_data['timestamp']}
Severity: {alert_data['severity']}

Action Required: Please investigate and consider rollback if necessary.

---
HoloLoom Phase 3 Adaptive Learning System
"""

    # HTML body
    html = f"""
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
.header {{ background-color: #d32f2f; color: white; padding: 20px; }}
.content {{ padding: 20px; }}
.metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-left: 4px solid #2196f3; }}
.footer {{ background-color: #f5f5f5; padding: 10px; margin-top: 20px; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
<h1>🚨 {alert_data['title']}</h1>
</div>
<div class="content">
<div class="metric">
<strong>Current Accuracy:</strong> {alert_data['current_accuracy']:.1%}
</div>
<div class="metric">
<strong>Baseline Accuracy:</strong> {alert_data['baseline_accuracy']:.1%}
</div>
<div class="metric">
<strong>Drop:</strong> {alert_data['drop_percentage']:.1%} (threshold: 2.0%)
</div>
<div class="metric">
<strong>Affected Complexities:</strong><br>
{'<br>'.join(f"• {c}" for c in alert_data['affected_complexity'])}
</div>
<div class="metric">
<strong>Time:</strong> {alert_data['timestamp']}<br>
<strong>Severity:</strong> <span style="color: red;">{alert_data['severity']}</span>
</div>
<p><strong>Action Required:</strong> Please investigate and consider rollback if necessary.</p>
</div>
<div class="footer">
HoloLoom Phase 3 Adaptive Learning System
</div>
</body>
</html>
"""

    # Attach parts
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')
    msg.attach(part1)
    msg.attach(part2)

    # Send email
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"Email alert sent to {', '.join(SMTP_TO)}")
        return True

    except Exception as e:
        print(f"Error sending email: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    # Example usage
    alert = {
        "title": "Classifier Regression Detected",
        "current_accuracy": 0.853,
        "baseline_accuracy": 0.950,
        "drop_percentage": 0.097,
        "affected_complexity": ["complex", "research"],
        "severity": "CRITICAL",
        "timestamp": datetime.now().isoformat()
    }

    send_email(alert)
```

---

## Production Integration

### 1. Create Production Application

Create `/opt/hololoom/production_app.py`:

```python
#!/usr/bin/env python3
"""
HoloLoom Phase 3 Production Application
Integrates adaptive learning with real production workloads.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime

from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier
from HoloLoom.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/hololoom/production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HoloLoomProduction:
    """Production HoloLoom application with Phase 3 integration."""

    def __init__(self):
        self.classifier = None
        self.running = False

    async def initialize(self):
        """Initialize production classifier."""
        logger.info("Initializing HoloLoom Production Application...")

        # Create adaptive classifier
        self.classifier = AdaptiveMoonshotClassifier(
            enable_semantic_tier=True,
            enable_adaptive_learning=True,
            data_dir=Path("/var/lib/hololoom/data"),
            validation_set_path=Path("/var/lib/hololoom/data/validation/validation_set.json"),
            background_learning=True,
            learning_update_interval=3600.0  # 1 hour
        )

        # Start background learning
        await self.classifier.start_background_learning()

        logger.info("Production application initialized successfully")

    async def classify_query(self, query: str):
        """Classify a single query."""
        try:
            result = self.classifier.classify(query)
            return {
                "query": query,
                "complexity": result.complexity.value,
                "confidence": result.confidence,
                "tier_used": result.tier_used,
                "latency_ms": result.metadata.get('latency_ms', 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return None

    async def run(self):
        """Main production loop."""
        self.running = True
        logger.info("Production application running...")

        # Your production query processing loop
        # This is a placeholder - replace with your actual workload
        while self.running:
            # Example: process queries from queue/API/etc.
            # query = await get_next_query()
            # result = await self.classify_query(query)
            # await process_result(result)

            await asyncio.sleep(1)

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down production application...")
        self.running = False

        if self.classifier:
            await self.classifier.stop_background_learning()

        # View final statistics
        if self.classifier:
            stats = self.classifier.get_learning_statistics()
            logger.info(f"Final Statistics:")
            logger.info(f"  Total Queries: {stats['total_queries_logged']}")
            logger.info(f"  Patterns Discovered: {stats['patterns_discovered']}")
            logger.info(f"  Patterns Deployed: {stats['patterns_deployed']}")
            logger.info(f"  Validation Accuracy: {stats['validation_accuracy']:.1%}")
            logger.info(f"  Regression Alerts: {stats['regression_alerts']}")

        logger.info("Shutdown complete")

# Global app instance
app = None

def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {sig}, initiating shutdown...")
    if app:
        asyncio.create_task(app.shutdown())

async def main():
    """Main entry point."""
    global app

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and initialize app
    app = HoloLoomProduction()
    await app.initialize()

    # Run production loop
    try:
        await app.run()
    except Exception as e:
        logger.error(f"Production error: {e}")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Create Systemd Service

Create `/etc/systemd/system/hololoom.service`:

```ini
[Unit]
Description=HoloLoom Phase 3 Production Service
After=network.target

[Service]
Type=simple
User=hololoom
Group=hololoom
WorkingDirectory=/opt/hololoom
EnvironmentFile=/opt/hololoom/.env
ExecStart=/opt/hololoom/.venv/bin/python /opt/hololoom/production_app.py
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=hololoom

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/hololoom /var/log/hololoom

[Install]
WantedBy=multi-user.target
```

Enable and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hololoom
sudo systemctl start hololoom
sudo systemctl status hololoom
```

---

## Validation & Testing

### 1. Create Validation Script

Create `scripts/validate_production.py`:

```python
#!/usr/bin/env python3
"""Validate Phase 3 production deployment."""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, '/opt/hololoom')

from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier

async def validate_deployment():
    """Run production validation checks."""

    print("=" * 60)
    print("HoloLoom Phase 3 - Production Validation")
    print("=" * 60)
    print()

    checks_passed = 0
    checks_total = 0

    # Check 1: Classifier initialization
    checks_total += 1
    print("[1/8] Testing classifier initialization...")
    try:
        classifier = AdaptiveMoonshotClassifier(
            enable_adaptive_learning=True,
            data_dir=Path("/var/lib/hololoom/data")
        )
        print("  ✅ Classifier initialized successfully")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Failed to initialize classifier: {e}")
        return

    # Check 2: Classification
    checks_total += 1
    print("[2/8] Testing query classification...")
    try:
        result = classifier.classify("What is machine learning?")
        assert result.complexity is not None
        assert 0 <= result.confidence <= 1
        print(f"  ✅ Classification working (complexity={result.complexity.value}, confidence={result.confidence:.1%})")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Classification failed: {e}")

    # Check 3: JSONL logging
    checks_total += 1
    print("[3/8] Testing JSONL logging...")
    try:
        log_file = classifier.classification_log_path
        assert log_file.exists()
        assert log_file.stat().st_size > 0
        print(f"  ✅ JSONL log exists: {log_file}")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ JSONL logging failed: {e}")

    # Check 4: Pattern miner
    checks_total += 1
    print("[4/8] Testing pattern miner...")
    try:
        patterns = classifier.pattern_miner.mine_patterns(days_lookback=7)
        print(f"  ✅ Pattern miner working ({len(patterns)} patterns found)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Pattern miner failed: {e}")

    # Check 5: Validator
    checks_total += 1
    print("[5/8] Testing continuous validator...")
    try:
        result = await classifier.validator.validate_hourly(sample_size=5)
        print(f"  ✅ Validator working (accuracy={result.overall_accuracy:.1%})")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Validator failed: {e}")

    # Check 6: Updater
    checks_total += 1
    print("[6/8] Testing adaptive updater...")
    try:
        status = classifier.updater.get_deployment_status()
        print(f"  ✅ Updater working (deployments={status['total_deployments']})")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Updater failed: {e}")

    # Check 7: Reporter
    checks_total += 1
    print("[7/8] Testing performance reporter...")
    try:
        metrics = classifier.reporter.export_prometheus_metrics()
        assert "moonshot_accuracy" in metrics
        print(f"  ✅ Reporter working (metrics exported)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Reporter failed: {e}")

    # Check 8: Background learning
    checks_total += 1
    print("[8/8] Testing background learning...")
    try:
        await classifier.start_background_learning()
        await asyncio.sleep(2)  # Let it start
        await classifier.stop_background_learning()
        print(f"  ✅ Background learning working")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Background learning failed: {e}")

    # Summary
    print()
    print("=" * 60)
    print(f"Validation Complete: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)

    if checks_passed == checks_total:
        print("✅ All checks passed! Production deployment validated.")
        return 0
    else:
        print(f"❌ {checks_total - checks_passed} checks failed. Please investigate.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(validate_deployment())
    sys.exit(exit_code)
```

Run validation:

```bash
python scripts/validate_production.py
```

---

## Monitoring & Maintenance

### Daily Checks

- Review daily reports in `/var/lib/hololoom/reports/daily/`
- Check Grafana dashboard for anomalies
- Verify no critical alerts in Slack
- Check log files for errors: `tail -f /var/log/hololoom/production.log`

### Weekly Tasks

- Review weekly reports
- Analyze pattern quality trends
- Update validation set if needed
- Review and approve high-quality patterns

### Monthly Tasks

- Review and update validation set
- Analyze long-term accuracy trends
- Optimize quality thresholds based on production data
- Backup classification logs and patterns

### Rotation & Archiving

Add to crontab:

```bash
# Rotate classification logs weekly
0 0 * * 0 /opt/hololoom/scripts/rotate_logs.sh

# Archive old reports monthly
0 0 1 * * /opt/hololoom/scripts/archive_reports.sh
```

---

## Troubleshooting

### Common Issues

**Issue: No metrics appearing in Prometheus**
- Check metrics export script is running (crontab)
- Verify file permissions on `/var/lib/prometheus/textfile_collector`
- Check Prometheus logs: `docker logs prometheus`

**Issue: Background learning not running**
- Check systemd service status: `systemctl status hololoom`
- Review logs: `journalctl -u hololoom -f`
- Verify `.env` configuration

**Issue: High memory usage**
- Rotate classification logs
- Reduce validation history size
- Check for pattern accumulation

**Issue: No patterns discovered**
- Verify sufficient classification logs (>100 queries)
- Lower quality thresholds temporarily
- Check pattern miner logs

**Issue: Frequent regression alerts**
- Review validation set quality
- Adjust regression threshold
- Investigate actual accuracy issues

---

## Support & Resources

- **Documentation**: [PHASE_3_DOCUMENTATION.md](PHASE_3_DOCUMENTATION.md:1)
- **Progress**: [PHASE_3_PROGRESS.md](PHASE_3_PROGRESS.md:1)
- **Tests**: `pytest HoloLoom/routing/learning/tests/ -v`
- **Logs**: `/var/log/hololoom/`
- **Metrics**: `http://localhost:9090` (Prometheus)
- **Dashboard**: `http://localhost:3000` (Grafana)

---

**Production Deployment Guide Complete**
**Ready for Live Deployment**: ✅
