# HoloLoom Security Dashboards - Infrastructure

**Version**: 1.0.0 (2025-11-15)
**Status**: Production Ready
**Components**: 5 Dashboards, 2 Datasources, 4 Alert Rules

## Overview

This directory contains complete Grafana configuration for HoloLoom's Phase 3 Security Pipeline, providing real-time monitoring, visualization, and alerting for security events, attacks, anomalies, and incidents.

## Directory Structure

```
infra/grafana/
├── dashboards/                 # Grafana dashboard definitions (JSON)
│   ├── security_overview.json   # Main security dashboard (792 lines)
│   ├── authentication.json      # Auth/authz metrics (912 lines)
│   ├── attacks.json             # Real-time attack monitoring (801 lines)
│   ├── anomalies.json           # ML-based anomaly detection (833 lines)
│   └── incidents.json           # Incident tracking/SLA (753 lines)
│
├── datasources/                # Datasource configurations
│   ├── prometheus.yaml          # Prometheus time-series DB (21 lines)
│   └── elasticsearch.yaml       # Elasticsearch logs/SIEM (26 lines)
│
└── README.md                   # This file
```

## Dashboards Overview

### 1. Security Overview (`security_overview.json`)

**Primary**: Main entry point for security monitoring
**Panels**: 10
**Update Interval**: 30 seconds
**Key Metrics**:
- Attack rate (requests/min) - CRITICAL if >100
- Blocked vs allowed requests
- Active API keys and OAuth sessions
- Rate limit utilization (P99)
- Top 10 IPs by request volume
- Authentication success rate (target >95%)
- Current risk level (0-1 scale)
- Anomaly score distribution

**Visualization Types**:
- Pie charts (attack types, asset distribution)
- Time series (allowed vs blocked over time)
- Stat panels (key metrics)
- Gauges (risk level, rate limits)

### 2. Authentication & Authorization (`authentication.json`)

**Primary**: Identity and access control monitoring
**Panels**: 11
**Update Interval**: 30 seconds
**Key Metrics**:
- Login success rate (target >95%)
- Failed logins per minute (alert if >10)
- OAuth tokens issued per minute
- MFA challenge rate (target 10-25/min)
- Failed login attempts by IP
- RBAC permission denials
- Session duration distribution
- Top users by activity

**Visualization Types**:
- Stat panels (success rates)
- Time series (login attempts)
- Tables (failed IPs, user activity)
- Pie charts (API scope distribution)

### 3. Real-Time Attack Monitoring (`attacks.json`)

**Primary**: Live attack detection and response
**Panels**: 9
**Update Interval**: 10 seconds (real-time)
**Key Metrics**:
- Real-time attack feed (last 100)
- Attack type breakdown (SQLi, XSS, CSRF, RFI)
- Blocked vs allowed requests
- Attack volume time series
- Blocked IPs (temporary bans)
- WAF rule triggers
- Top attack patterns
- Attack success rate (should be 0%)

**Visualization Types**:
- Tables (attack feed, blocked IPs, patterns)
- Pie charts (type breakdown, success rate)
- Time series (volume trends)
- Stat panels (success rate, blocked count)

### 4. Anomaly Detection (`anomalies.json`)

**Primary**: ML-based behavioral anomaly detection
**Panels**: 10
**Update Interval**: 30 seconds
**Key Metrics**:
- Anomaly score time series (target <0.3)
- Anomalies by type (rate, behavior, geo, etc.)
- False positive tracking (target <5%)
- Top anomalous users
- Model performance metrics:
  - Precision (target >95%)
  - Recall (target >90%)
  - F1 score (target >92%)
- Baseline vs current behavior deviation

**Visualization Types**:
- Time series with thresholds (anomaly score)
- Pie charts (anomaly types, FP tracking)
- Tables (anomalous users)
- Stat panels (model metrics)
- Histograms (score distribution)

### 5. Security Incident Timeline (`incidents.json`)

**Primary**: Incident tracking and SLA monitoring
**Panels**: 9
**Update Interval**: 30 seconds
**Time Range**: 7 days (configurable)
**Key Metrics**:
- Security incidents (chronological)
- Incident severity distribution (CRITICAL, HIGH, MEDIUM)
- Incident status tracking (new, investigating, resolved)
- Mean time to detect (MTTD) - target <30 min
- Mean time to respond (MTTR) - target <60 min
- Incident trends by severity (hourly)
- MTTD & MTTR trends
- Remediation actions taken
- Post-mortem reviews

**Visualization Types**:
- Tables (incident timeline, remediation, post-mortems)
- Pie charts (severity, status distribution)
- Stat panels (MTTD, MTTR)
- Time series (incident volume, trends)

## Dashboard Statistics

| Dashboard | Panels | Metrics | Data Sources | Refresh |
|-----------|--------|---------|--------------|---------|
| Security Overview | 10 | 8 | Prometheus, ES | 30s |
| Authentication | 11 | 8 | Prometheus, ES | 30s |
| Attacks | 9 | 8 | Prometheus, ES | 10s |
| Anomalies | 10 | 10 | Prometheus, ES | 30s |
| Incidents | 9 | 8 | Prometheus, ES | 30s |
| **Total** | **49** | **42** | **2** | **Variable** |

## Datasources

### Prometheus (Time-Series Database)

**File**: `datasources/prometheus.yaml`
**URL**: `http://prometheus:9090`
**Type**: Prometheus 2.40+
**Metrics Scraped**: Security metrics with `hololoom_security_*` prefix
**Scrape Interval**: 15 seconds
**Queries Used**: PromQL (Prometheus Query Language)

**Key Metrics**:
- `hololoom_security_requests_blocked_total` - Blocked requests counter
- `hololoom_security_login_attempts_total` - Login attempts counter
- `hololoom_security_attacks_total` - Attack counter by type
- `hololoom_security_anomaly_score` - Anomaly score gauge
- `hololoom_security_incidents_total` - Incident counter
- `hololoom_security_mean_time_to_*_minutes` - SLA metrics

### Elasticsearch (Log Aggregation & SIEM)

**File**: `datasources/elasticsearch.yaml`
**URL**: `http://elasticsearch:9200`
**Type**: Elasticsearch 8.0+
**Authentication**: Basic Auth (user: `elastic`, password: `elastic`)
**Indices Used**: `hololoom-security-*` (time-based)
**Queries Used**: Elasticsearch DSL + KQL

**Key Indices**:
- `hololoom-security-logs` - All security events
- `hololoom-security-attacks` - Attack-specific logs
- `hololoom-security-incidents` - Incident tracking
- `hololoom-security-audit` - Audit trail

## Setting Up Dashboards

### Quick Start (Automated)

```bash
chmod +x scripts/setup_grafana.sh
./scripts/setup_grafana.sh
```

Expected time: 2-3 minutes

### Manual Import

1. **Configure Datasources**:
   - Grafana UI: Settings > Data sources
   - Import `datasources/prometheus.yaml`
   - Import `datasources/elasticsearch.yaml`

2. **Import Dashboards**:
   - Grafana UI: Dashboards > Import
   - Upload each JSON file from `dashboards/`
   - Select datasources
   - Click Import

3. **Create Alert Rules**:
   - See section below

### Using Docker Compose

```yaml
version: '3.8'
services:
  grafana:
    image: grafana/grafana:10.0.0
    container_name: hololoom-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_INSTALL_PLUGINS: grafana-piechart-panel
      GF_ANALYTICS_REPORTING_ENABLED: "false"
    volumes:
      # Mount datasources
      - ./infra/grafana/datasources:/etc/grafana/provisioning/datasources
      # Mount dashboards
      - ./infra/grafana/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - hololoom
    depends_on:
      - prometheus
      - elasticsearch
```

## Alerting Configuration

### Alert Rules (Core 4)

1. **Attack Rate Critical**
   - Threshold: >100 requests/min
   - Duration: 5 minutes
   - Severity: CRITICAL
   - Action: Auto-escalate

2. **Failed Auth Threshold**
   - Threshold: >10 failures per IP per 5min
   - Duration: 5 minutes
   - Severity: WARNING
   - Action: Temporary IP ban

3. **Anomaly Score Critical**
   - Threshold: Score >0.9
   - Duration: 5 minutes
   - Severity: CRITICAL
   - Action: Manual investigation

4. **WAF Rule Trigger Warning**
   - Threshold: >1000 triggers per 5min
   - Duration: 5 minutes
   - Severity: WARNING
   - Action: Configuration review

### Creating Custom Alerts

**In Grafana UI**:
1. Alerting > Alert rules > New alert rule
2. Set query and condition
3. Set duration and frequency
4. Add notification channel
5. Save rule

**Via API**:
```bash
curl -X POST http://grafana:3000/api/ruler/grafana/rules/SecurityAlerts \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{ "rules": [...] }'
```

## Notification Channels

### Slack

```bash
# Webhook URL from Slack
https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Configure in Grafana:
# 1. Alerting > Notification channels
# 2. Type: Slack
# 3. Webhook URL: paste above
# 4. Channel: #security-alerts
```

### Email

```yaml
# docker-compose environment
GF_SMTP_ENABLED: "true"
GF_SMTP_HOST: "smtp.gmail.com:587"
GF_SMTP_USER: "alerts@hololoom.com"
GF_SMTP_PASSWORD: "<app-password>"
GF_SMTP_FROM_ADDRESS: "alerts@hololoom.com"
```

### Custom Webhooks

```json
{
  "type": "webhook",
  "name": "Custom Webhook",
  "url": "https://incident-management.example.com/api/alerts"
}
```

## Metrics & Queries

### Key PromQL Queries

```promql
# Attack rate (requests/min)
sum(rate(hololoom_security_requests_blocked_total[5m]))

# Authentication success rate
hololoom_security_login_success_rate

# Anomaly score with rolling average
hololoom_security_anomaly_score vs hololoom_security_anomaly_score_rolling_mean

# Model precision (ML accuracy)
hololoom_security_model_precision

# Mean time to detect
hololoom_security_mean_time_to_detect_minutes
```

### Key Elasticsearch Queries

```json
// Get top attack types (last 1h)
{
  "size": 0,
  "aggs": {
    "attack_types": {
      "terms": {
        "field": "attack_type",
        "size": 10
      }
    }
  },
  "query": {
    "range": {
      "@timestamp": { "gte": "now-1h" }
    }
  }
}

// Failed login attempts by IP
{
  "aggs": {
    "by_ip": {
      "terms": {
        "field": "source_ip",
        "order": { "_count": "desc" }
      }
    }
  },
  "query": { "match": { "event_type": "login_failure" } }
}
```

## Performance Tuning

### Dashboard Performance

| Component | Optimization | Impact |
|-----------|-------------|--------|
| Query Time | Use 5m intervals instead of 7d | -80% query time |
| Result Set | topk(10) instead of all results | -90% transfer |
| Cache | Query caching (60s) | -70% subsequent |
| Panels | Reduce from 15 to 10 panels | -40% render time |
| Refresh | 30s instead of 5s | -80% API calls |

### Recommended Settings

```yaml
# prometheus.yml
global:
  scrape_interval: 15s        # Balance between freshness and load
  evaluation_interval: 15s

# Retention (adjust based on storage)
--storage.tsdb.retention.time=15d    # 15 days of metrics

# Elasticsearch index lifecycle
hot: 0 days
warm: 3 days
cold: 30 days
delete: 90 days
```

## Troubleshooting

### Dashboards Show "No Data"

1. Check Prometheus is scraping:
   ```bash
   curl http://prometheus:9090/targets
   ```

2. Verify metric exists:
   ```bash
   curl 'http://prometheus:9090/api/v1/query?query=hololoom_security_attacks_total'
   ```

3. Check Elasticsearch indices:
   ```bash
   curl http://elasticsearch:9200/_cat/indices
   ```

### Alerts Not Firing

1. Check alert rule status:
   ```bash
   curl http://prometheus:9090/api/v1/rules
   ```

2. Test notification:
   - Grafana UI: Alerting > Notification channels > Test

3. Check alert logs:
   ```bash
   docker logs grafana | grep -i alert
   docker logs prometheus | grep -i rule
   ```

### High Memory Usage

1. Reduce query time range (5m instead of 7d)
2. Use `topk(n)` to limit results
3. Increase Grafana memory: `mem_limit: 2gb`
4. Increase Prometheus storage: `--storage.tsdb.max-block-duration=2h`

## Version History

### v1.0.0 (2025-11-15)
- Initial release
- 5 production dashboards
- 49 total panels
- 4 core alerting rules
- Prometheus + Elasticsearch integration
- Complete documentation

## Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| security_overview.json | 792 | Main dashboard |
| authentication.json | 912 | Auth metrics |
| attacks.json | 801 | Attack monitoring |
| anomalies.json | 833 | Anomaly detection |
| incidents.json | 753 | Incident tracking |
| prometheus.yaml | 21 | Prometheus config |
| elasticsearch.yaml | 26 | Elasticsearch config |
| **Total** | **4,138** | **All files** |

## Configuration Files

### Prometheus Scrape Config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'hololoom-security'
    metrics_path: '/metrics/security'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
```

### Elasticsearch Mappings

```json
{
  "mappings": {
    "properties": {
      "@timestamp": { "type": "date" },
      "event_type": { "type": "keyword" },
      "severity": { "type": "keyword" },
      "source_ip": { "type": "ip" },
      "attack_type": { "type": "keyword" },
      "anomaly_score": { "type": "float" }
    }
  }
}
```

## Integration Points

- **Slack**: Receive real-time alerts
- **Email**: Daily/weekly digests
- **Webhooks**: Custom incident management
- **PagerDuty**: Escalation policies
- **Splunk**: Extended SIEM integration
- **DataDog**: APM correlation

## Support & Documentation

- **Setup Guide**: `docs/GRAFANA_SETUP.md` (1,182 lines)
- **Main Documentation**: `CLAUDE.md`
- **Architecture**: `docs/ARCHITECTURE_VISUAL_MAP.md`
- **Metrics**: See PromQL queries in dashboards
- **Alerting**: See alert rules in setup script

## Maintenance

### Weekly
- [ ] Review alert accuracy
- [ ] Check for false positives
- [ ] Verify data freshness

### Monthly
- [ ] Optimize slow queries
- [ ] Update thresholds
- [ ] Rotate API tokens

### Quarterly
- [ ] Update security rules
- [ ] Review effectiveness
- [ ] Train team on updates

---

**Last Updated**: 2025-11-15
**Version**: 1.0.0
**Status**: Production Ready
**Maintainers**: HoloLoom Security Team
