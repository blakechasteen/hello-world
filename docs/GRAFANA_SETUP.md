# HoloLoom Security Dashboards - Grafana Setup Guide

**Version:** 1.0.0 (2025-11-15)
**Status:** Production Ready
**Author:** HoloLoom Security Team

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Dashboards](#dashboards)
5. [Metrics Reference](#metrics-reference)
6. [Alerting Rules](#alerting-rules)
7. [Integration](#integration)
8. [Troubleshooting](#troubleshooting)
9. [Security Considerations](#security-considerations)

## Overview

This guide covers the setup and configuration of comprehensive security dashboards for HoloLoom's Phase 3 security pipeline. The dashboards provide real-time monitoring, visualization, and alerting for security events, attacks, anomalies, and incidents.

### Key Features

- **5 Production-Ready Dashboards**: Cover all security aspects
- **Real-Time Monitoring**: 10-30 second refresh intervals
- **Multi-Source Integration**: Prometheus + Elasticsearch
- **Automated Alerts**: 4 core alerting rules with extensibility
- **Complete Lineage**: Full audit trail and incident tracking
- **Performance Analytics**: Attack patterns, success rates, anomaly detection
- **Incident Management**: MTTD/MTTR tracking, remediation workflows

### Dashboard Statistics

| Dashboard | Panels | Data Sources | Update Frequency |
|-----------|--------|--------------|------------------|
| Security Overview | 10 | Prometheus + Elasticsearch | 30s |
| Authentication | 11 | Prometheus + Elasticsearch | 30s |
| Attacks | 9 | Prometheus + Elasticsearch | 10s (real-time) |
| Anomalies | 10 | Prometheus + Elasticsearch | 30s |
| Incidents | 9 | Prometheus + Elasticsearch | 30s |
| **Total** | **49 panels** | **2 datasources** | **Variable** |

## Quick Start

### Prerequisites

- Docker & Docker Compose installed
- Grafana 9.0+ running
- Prometheus 2.40+ running
- Elasticsearch 8.0+ running
- curl command-line tool
- 500MB free disk space for logs/indices

### 1. Automated Setup (Recommended)

```bash
# Make script executable
chmod +x scripts/setup_grafana.sh

# Run setup with default configuration
./scripts/setup_grafana.sh

# Or specify custom Grafana URL
./scripts/setup_grafana.sh http://grafana.example.com:3000 <api_token>
```

This will:
- ✅ Verify prerequisites
- ✅ Create API token
- ✅ Configure Prometheus datasource
- ✅ Configure Elasticsearch datasource
- ✅ Import 5 security dashboards
- ✅ Create alert rules
- ✅ Configure notification channels
- ✅ Generate summary report

**Expected Duration:** 2-3 minutes

### 2. Manual Setup

If automated setup fails, follow these steps:

#### Step 1: Configure Datasources

Navigate to **Configuration > Data sources** in Grafana:

**Prometheus:**
- Name: `Prometheus`
- Type: `Prometheus`
- URL: `http://prometheus:9090`
- Default: Yes
- Save & Test

**Elasticsearch:**
- Name: `Elasticsearch`
- Type: `Elasticsearch`
- URL: `http://elasticsearch:9200`
- Basic Auth: Enabled
- User: `elastic`
- Password: `elastic`
- Version: `8.0.0`
- Save & Test

#### Step 2: Import Dashboards

Navigate to **Dashboards > Import**:

1. Upload JSON file: `infra/grafana/dashboards/security_overview.json`
2. Select datasources (Prometheus, Elasticsearch)
3. Click "Import"
4. Repeat for all 5 dashboards

#### Step 3: Create Alert Rules

Navigate to **Alerting > Alert rules** and create rules:

```
rule: attack_rate_critical
expr: rate(hololoom_security_requests_blocked_total[5m]) > 100
severity: CRITICAL
for: 5m

rule: failed_auth_threshold
expr: sum(rate(hololoom_security_login_failures_total[5m])) by (source_ip) > 10
severity: WARNING
for: 5m

rule: anomaly_score_critical
expr: hololoom_security_anomaly_score_avg > 0.9
severity: CRITICAL
for: 5m

rule: waf_rule_trigger_warning
expr: sum(rate(hololoom_security_waf_rules_triggered_total[5m])) > 1000
severity: WARNING
for: 5m
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                   HoloLoom Security Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Security Event Sources                      │   │
│  │  - Authentication System                            │   │
│  │  - WAF / Request Filtering                          │   │
│  │  - Anomaly Detection Engine                         │   │
│  │  - Incident Management                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Metric Collection                          │   │
│  │  - Prometheus (metric exporter)                     │   │
│  │  - Elasticsearch (log aggregation)                  │   │
│  └──────────────────────────────────────────────────────┘   │
│           ↓                          ↓                       │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │   Prometheus DB     │  │  Elasticsearch      │           │
│  │   (Time Series)     │  │  (Logs + Indexing)  │           │
│  └─────────────────────┘  └─────────────────────┘           │
│           ↓                          ↓                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Grafana (Visualization)                   │   │
│  │  - 5 Security Dashboards                            │   │
│  │  - Real-time Alerting                              │   │
│  │  - Alert Routing (Slack, Email, Webhooks)          │   │
│  └──────────────────────────────────────────────────────┘   │
│           ↓                          ↓                       │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │  Alert Manager      │  │  Notification Mgr   │           │
│  │  (Rule Evaluation)  │  │  (Slack, Email)     │           │
│  └─────────────────────┘  └─────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Security Event
   ↓
2. Prometheus Scrape (15s interval)
3. Elasticsearch Ingest (Real-time)
   ↓
4. Query Aggregation (Grafana)
   ├─ Panel Queries (PromQL, Elasticsearch DSL)
   ├─ Query Caching (60s)
   └─ Visualization Rendering
   ↓
5. Alert Evaluation (every 1-5 minutes)
   ├─ Rule Threshold Check
   ├─ Duration Check
   └─ State Transition
   ↓
6. Notification
   ├─ Slack Channel
   ├─ Email Alert
   └─ Webhook
```

## Dashboards

### 1. Security Overview (Main Dashboard)

**Purpose**: High-level security posture and key metrics
**UID**: `hololoom-security-overview`
**Refresh**: 30 seconds
**Panels**: 10

#### Key Metrics

| Panel | Metric | Threshold | Alert |
|-------|--------|-----------|-------|
| Attack Rate | requests/min | >100 | CRITICAL |
| Blocked vs Allowed | ratio | N/A | INFO |
| API Keys | count | N/A | WATCH |
| OAuth Sessions | count | N/A | WATCH |
| Rate Limit | utilization % | p99 | MONITOR |
| Top IPs | request volume | Top 10 | INFO |
| Auth Success Rate | % | >95% | WARNING |
| Risk Level | score 0-1 | >0.7 | CRITICAL |

#### Example Queries

```promql
# Attack Rate (requests/min)
sum(rate(hololoom_security_requests_blocked_total[5m]))

# Allowed vs Blocked (5min avg)
rate(hololoom_security_requests_allowed_total[5m])
vs
rate(hololoom_security_requests_blocked_total[5m])

# Active API Keys
hololoom_security_active_api_keys

# Current Risk Level
hololoom_security_anomaly_score_avg

# Anomaly Score Distribution
hololoom_security_anomaly_score_distribution{quantile=...}
```

#### What to Watch

- **Attack Rate Spikes**: Sudden increases indicate active attacks
- **Risk Level Trending**: Upward trend suggests degraded security posture
- **Blocked API Keys**: Increase may indicate compromised credentials
- **Rate Limit Utilization**: High P99 suggests DDoS attempts

### 2. Authentication & Authorization Metrics

**Purpose**: User access, identity, and authorization tracking
**UID**: `hololoom-auth-metrics`
**Refresh**: 30 seconds
**Panels**: 11

#### Key Metrics

| Panel | Metric | Threshold | Alert |
|-------|--------|-----------|-------|
| Login Success Rate | % | >95% | WARNING |
| Failed Logins | per minute | >10 | WATCH |
| OAuth Tokens | issued/min | N/A | INFO |
| MFA Challenges | per minute | <25 | WATCH |
| Failed Attempts by IP | count | >5 from same IP | CRITICAL |
| RBAC Denials | per minute | <50 | WATCH |

#### Elasticsearch Queries

```json
// Failed login attempts by IP
{
  "size": 0,
  "aggs": {
    "by_ip": {
      "terms": {
        "field": "source_ip",
        "size": 10,
        "order": { "_count": "desc" }
      }
    }
  },
  "query": { "match": { "event_type": "login_failure" } }
}
```

#### What to Watch

- **Spike in Failed Logins**: Brute force attack attempt
- **Concentrated Failures from IP**: IP should be temporarily banned
- **Low MFA Challenge Rate**: May indicate MFA bypass or misconfiguration
- **RBAC Denial Spike**: Permission escalation attempts

### 3. Real-Time Attack Monitoring

**Purpose**: Live attack detection and WAF effectiveness
**UID**: `hololoom-attack-monitoring`
**Refresh**: 10 seconds (real-time)
**Panels**: 9

#### Key Metrics

| Panel | Metric | Threshold | Alert |
|-------|--------|-----------|-------|
| Attack Feed | last 100 | Real-time | INFO |
| Attack Types | breakdown | SQLi/XSS high | WARNING |
| Attack Rate | per minute | >100 | CRITICAL |
| WAF Triggers | per minute | >1000 | WARNING |
| Attack Success Rate | % | >0% | CRITICAL |
| Blocked IPs | temporary bans | N/A | INFO |

#### Attack Type Breakdown

```
- SQLi (SQL Injection): Malicious SQL queries
- XSS (Cross-Site Scripting): Injected scripts
- CSRF (Cross-Site Request Forgery): Unauthorized requests
- RFI (Remote File Inclusion): File upload exploits
- LFI (Local File Inclusion): Local file access attempts
- Command Injection: OS command execution attempts
- Path Traversal: Directory traversal attempts
```

#### Example Queries

```promql
# Attack type distribution (last 1h)
sum(increase(hololoom_security_attacks_total[1h])) by (attack_type)

# Attack volume time series
rate(hololoom_security_attacks_total{attack_type=~"SQLi|XSS|CSRF|RFI"}[5m])

# Attack success rate (should be 0%)
hololoom_security_attack_success_rate

# WAF rule triggers
rate(hololoom_security_waf_rules_triggered_total[5m])

# Attacks blocked (last hour)
sum(increase(hololoom_security_attacks_blocked_total[1h]))
```

#### What to Watch

- **Attack Success Rate >0%**: Security breach occurred
- **Sudden Attack Type Change**: New attack method detected
- **WAF Bypass**: Attack bypassed WAF protection
- **High False Positive Rate**: Legitimate traffic being blocked

### 4. Anomaly Detection Dashboard

**Purpose**: ML-based behavioral anomaly detection and modeling
**UID**: `hololoom-anomalies`
**Refresh**: 30 seconds
**Panels**: 10

#### Key Metrics

| Panel | Metric | Threshold | Alert |
|-------|--------|-----------|-------|
| Anomaly Score | 0-1 | >0.9 | CRITICAL |
| Anomalies by Type | count | N/A | INFO |
| False Positives | rate | <5% | WATCH |
| Model Precision | % | >95% | WATCH |
| Model Recall | % | >90% | WATCH |
| Model F1 Score | % | >92% | WATCH |
| Baseline Deviation | score | >0.5 | WARNING |

#### Anomaly Types

```
- Rate Anomaly: Unusual request rate/frequency
- Behavior Anomaly: Unusual user behavior pattern
- Geo Anomaly: Access from unusual geographic location
- Feature Anomaly: Unusual feature combination
- Ensemble Anomaly: Multiple weak signals
```

#### Model Performance Interpretation

```
Precision = TP / (TP + FP)
- Measures false alarm rate
- Target: >95% (want few false alarms)
- Low precision = too many false alarms

Recall = TP / (TP + FN)
- Measures detection coverage
- Target: >90% (want to catch most attacks)
- Low recall = missing real attacks

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Target: >92% (balanced performance)
- Sweet spot between accuracy and coverage
```

#### Example Queries

```promql
# Anomaly score time series with rolling mean
hololoom_security_anomaly_score
vs
hololoom_security_anomaly_score_rolling_mean

# Anomalies by type (last 1h)
sum(increase(hololoom_security_anomalies_total[1h])) by (anomaly_type)

# Model performance metrics
hololoom_security_model_precision        # Target: >0.95
hololoom_security_model_recall           # Target: >0.90
hololoom_security_model_f1_score         # Target: >0.92

# Baseline deviation
hololoom_security_baseline_deviation

# Anomaly score distribution
histogram_quantile(0.99, rate(hololoom_security_anomaly_score_distribution_bucket[5m]))
```

#### What to Watch

- **Anomaly Score Spike**: Unusual activity detected
- **Model Degradation**: Precision/Recall decreasing
- **False Positive Spike**: Model miscalibration
- **Baseline Drift**: Normal behavior changing

### 5. Security Incident Timeline

**Purpose**: Incident tracking, SLA monitoring, post-mortems
**UID**: `hololoom-incidents`
**Refresh**: 30 seconds
**Panels**: 9
**Time Range**: 7 days (can be adjusted)

#### Key Metrics

| Panel | Metric | Target | SLA |
|-------|--------|--------|-----|
| Incidents (Chronological) | timeline | All incidents | N/A |
| Severity Distribution | breakdown | Minimize CRITICAL | N/A |
| Status Distribution | pie | Minimize "Investigating" | N/A |
| MTTD | minutes | <30 min | SLA |
| MTTR | minutes | <60 min | SLA |
| Incident Volume | time series | Minimize | N/A |
| Remediation Actions | table | All documented | Audit |
| Post-Mortems | table | All incidents | Compliance |

#### SLA Definitions

```
MTTD (Mean Time to Detect):
- From incident start to first detection/alert
- Target: <30 minutes
- Measures detection effectiveness

MTTR (Mean Time to Respond):
- From detection to remediation start
- Target: <60 minutes
- Measures response speed

MTTR-Full (Mean Time to Resolve):
- From detection to complete resolution
- Target: <4 hours
- Measures resolution completeness
```

#### Incident Status Workflow

```
New → Investigating → Acknowledged → In Progress → Resolved → Closed
               ↑                                      ↓
               └──────────── Escalation ────────────┘

Alert Triggered (New)
      ↓
Alert Reviewed & Confirmed (Investigating)
      ↓
Incident Created & Assigned (Acknowledged)
      ↓
Actions Taken & Documented (In Progress)
      ↓
Malicious Activity Blocked (Resolved)
      ↓
Post-Mortem & Lessons Learned (Closed)
```

#### Example Queries

```promql
# Mean time to detect
hololoom_security_mean_time_to_detect_minutes

# Mean time to respond
hololoom_security_mean_time_to_respond_minutes

# Incident volume by severity (last 24h)
increase(hololoom_security_incidents_total[1h]) by (severity)

# MTTD trend for critical incidents
hololoom_security_mttd_by_severity{severity="CRITICAL"}
```

#### What to Watch

- **MTTD Trending Up**: Detection latency increasing
- **MTTR Trending Up**: Response time degrading
- **Incident Clustering**: Multiple related incidents
- **Unresolved Incidents**: Follow up required
- **Missing Post-Mortems**: Compliance violation

## Metrics Reference

### Metric Naming Convention

All security metrics follow the naming pattern:

```
hololoom_security_<subsystem>_<metric_type>_<unit>
```

Examples:
- `hololoom_security_requests_blocked_total` - Total blocked requests (counter)
- `hololoom_security_login_attempts_total` - Total login attempts (counter)
- `hololoom_security_anomaly_score` - Current anomaly score (gauge)
- `hololoom_security_session_duration_seconds` - Session duration histogram

### Core Metrics by Category

#### Request Metrics

```promql
# Total requests
hololoom_security_requests_allowed_total
hololoom_security_requests_blocked_total

# Request rates
rate(hololoom_security_requests_allowed_total[5m])
rate(hololoom_security_requests_blocked_total[5m])

# Request percentiles
histogram_quantile(0.95, rate(hololoom_security_request_duration_seconds_bucket[5m]))
```

#### Attack Metrics

```promql
# Attack counts
hololoom_security_attacks_total{attack_type=...}
hololoom_security_attacks_blocked_total{attack_type=...}

# Attack rates by type
rate(hololoom_security_attacks_total{attack_type="SQLi"}[5m])
rate(hololoom_security_attacks_total{attack_type="XSS"}[5m])

# Attack success rate
hololoom_security_attack_success_rate

# Blocked IPs
hololoom_security_blocked_ips_count
```

#### Authentication Metrics

```promql
# Login metrics
hololoom_security_login_attempts_total
hololoom_security_login_successes_total
hololoom_security_login_failures_total

# Login rate and success rate
rate(hololoom_security_login_attempts_total[5m])
hololoom_security_login_success_rate

# Session metrics
hololoom_security_session_duration_seconds
hololoom_security_active_sessions

# Token metrics
hololoom_security_oauth_token_issued_total
hololoom_security_api_key_usage_total{scope=...}
```

#### Anomaly Metrics

```promql
# Anomaly scoring
hololoom_security_anomaly_score                      # Current score (0-1)
hololoom_security_anomaly_score_rolling_mean         # 1h rolling average
hololoom_security_anomaly_score_distribution         # Histogram/distribution

# Anomaly detection
hololoom_security_anomalies_total{anomaly_type=...}
hololoom_security_anomaly_false_positives_total

# Model performance
hololoom_security_model_precision
hololoom_security_model_recall
hololoom_security_model_f1_score

# Baseline tracking
hololoom_security_baseline_deviation
```

#### Incident Metrics

```promql
# Incident counts
hololoom_security_incidents_total{severity=...}
hololoom_security_incidents_total{status=...}

# Response time
hololoom_security_mean_time_to_detect_minutes
hololoom_security_mean_time_to_respond_minutes
hololoom_security_incident_duration_minutes

# By severity
hololoom_security_mttd_by_severity{severity=...}
hololoom_security_mttr_by_severity{severity=...}
```

### Sample Prometheus Scrape Config

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hololoom-security'
    static_configs:
      - targets: ['localhost:8000']  # HoloLoom metrics endpoint
    scrape_interval: 15s
    scrape_timeout: 10s
    metrics_path: '/metrics/security'
```

## Alerting Rules

### Alert Rule Structure

```yaml
groups:
  - name: HoloLoomSecurity
    interval: 1m
    rules:
      - alert: AttackRateCritical
        expr: rate(hololoom_security_requests_blocked_total[5m]) > 100
        for: 5m
        labels:
          severity: critical
          team: security
        annotations:
          summary: "Critical attack rate detected"
          description: "Attack rate: {{ $value }} requests/min"
```

### Core Alert Rules

#### 1. Attack Rate Critical

```yaml
alert: AttackRateCritical
expr: rate(hololoom_security_requests_blocked_total[5m]) > 100
for: 5m
severity: CRITICAL
action: Auto-escalate to Security Team
```

**Triggers when:**
- Attack rate exceeds 100 requests/min for 5+ minutes
- Indicates active, sustained attack

**Response:**
- Activate incident response
- Increase monitoring frequency
- Prepare incident report

#### 2. Failed Auth Threshold

```yaml
alert: FailedAuthThreshold
expr: sum(rate(hololoom_security_login_failures_total[5m])) by (source_ip) > 10
for: 5m
severity: WARNING
action: Temporary IP ban (1 hour)
```

**Triggers when:**
- >10 failed login attempts from same IP in 5 minutes
- Indicates brute force attack

**Response:**
- Auto-ban IP for configurable duration
- Send alert to security team
- Log for forensics

#### 3. Anomaly Score Critical

```yaml
alert: AnomalyScoreCritical
expr: hololoom_security_anomaly_score_avg > 0.9
for: 5m
severity: CRITICAL
action: Manual investigation required
```

**Triggers when:**
- Anomaly score exceeds 0.9 for 5+ minutes
- Indicates unusual security activity

**Response:**
- Review anomaly details in dashboard
- Check correlated attacks/incidents
- Investigate user behavior

#### 4. WAF Rule Trigger Warning

```yaml
alert: WAFRuleTriggerWarning
expr: sum(rate(hololoom_security_waf_rules_triggered_total[5m])) > 1000
for: 5m
severity: WARNING
action: Investigate WAF configuration
```

**Triggers when:**
- WAF rules triggered >1000 times in 5 minutes
- Indicates high false positive rate or evasion

**Response:**
- Review WAF logs for patterns
- Check for legitimate traffic miscategorization
- Investigate attack patterns

### Custom Alert Rules

To add custom alert rules:

1. **Edit prometheus.yml:**
```yaml
rule_files:
  - 'infra/prometheus/security_rules.yml'

evaluation_interval: 1m
```

2. **Create security_rules.yml:**
```yaml
groups:
  - name: CustomSecurityAlerts
    interval: 1m
    rules:
      - alert: YourCustomAlert
        expr: your_custom_metric > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Custom alert triggered"
```

3. **Reload Prometheus:**
```bash
curl -X POST http://prometheus:9090/-/reload
```

## Integration

### Slack Integration

**Setup Slack Webhook:**

1. Create Slack App: https://api.slack.com/apps/new
2. Enable Incoming Webhooks
3. Create Webhook URL: `https://hooks.slack.com/services/YOUR/WEBHOOK/URL`

**Configure in Grafana:**

1. Navigate to **Alerting > Notification channels**
2. Click **New channel**
3. Type: `Slack`
4. Name: `Security-Alerts`
5. Webhook URL: paste webhook URL
6. Channel: `#security-alerts`
7. Custom message template:
```
{{ .Title }}
{{ .Message }}
Severity: {{ .Labels.severity }}
Dashboard: {{ .DashboardURL }}
```

### Email Integration

**Configure SMTP:**

1. Edit `docker-compose.yml`:
```yaml
grafana:
  environment:
    GF_SMTP_ENABLED: "true"
    GF_SMTP_HOST: "smtp.gmail.com:587"
    GF_SMTP_USER: "your-email@gmail.com"
    GF_SMTP_PASSWORD: "app-password"
    GF_SMTP_FROM_ADDRESS: "alerts@hololoom.com"
```

2. Restart Grafana:
```bash
docker-compose up -d grafana
```

3. Add notification channel in Grafana UI

### Webhook Integration

For custom integrations (e.g., incident management systems):

```bash
curl -X POST http://your-webhook-endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "alert": "AttackRateCritical",
    "status": "firing",
    "severity": "critical",
    "message": "Attack rate: 125 requests/min",
    "timestamp": "2025-11-15T10:30:00Z"
  }'
```

## Troubleshooting

### Issue: "No Data" in Dashboards

**Symptoms:**
- Panels show "No data returned"
- Empty graphs/tables
- Missing time series

**Root Causes:**
1. Prometheus not scraping metrics
2. Metric names don't match dashboard queries
3. Data retention period expired
4. Security metrics endpoint not running

**Solutions:**

```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Verify metric existence
curl 'http://prometheus:9090/api/v1/query?query=hololoom_security_attacks_total'

# Check metric names
curl 'http://prometheus:9090/api/v1/label/__name__/values' \
  | grep hololoom_security

# Verify scrape config
cat prometheus.yml | grep -A 10 "hololoom-security"
```

### Issue: Alerts Not Firing

**Symptoms:**
- Alert rules exist but never trigger
- No notifications received

**Root Causes:**
1. Alert rule condition never met
2. Alert duration not reached
3. Notification channel misconfigured
4. Prometheus evaluation failed

**Solutions:**

```bash
# Check alert rule status
curl http://prometheus:9090/api/v1/rules

# Check alert state
curl http://prometheus:9090/api/v1/alerts

# View Prometheus logs
docker logs prometheus | tail -100

# Test notification channel
# In Grafana: Alerting > Notification channels > Test
```

### Issue: Elasticsearch Connection Failed

**Symptoms:**
- "Connection refused" or "Unauthorized"
- Elasticsearch panels not loading
- No log data visible

**Root Causes:**
1. Elasticsearch not running
2. Wrong credentials
3. Elasticsearch index not created
4. Network connectivity issue

**Solutions:**

```bash
# Verify Elasticsearch running
curl http://localhost:9200/

# Check authentication
curl -u elastic:elastic http://localhost:9200/

# List indices
curl http://localhost:9200/_cat/indices

# Create index if needed
curl -X PUT http://localhost:9200/hololoom-security-logs

# Check network connectivity
docker exec grafana curl http://elasticsearch:9200/
```

### Issue: High Memory Usage

**Symptoms:**
- Grafana slow/unresponsive
- Memory usage > 1GB
- Dashboard loading slow

**Root Causes:**
1. Too many panels/queries per dashboard
2. Prometheus query cardinality explosion
3. Elasticsearch returning too many results
4. Dashboard refresh rate too aggressive

**Solutions:**

```bash
# Optimize queries - reduce time range:
rate(metric[5m]) instead of rate(metric[7d])

# Add query limits:
topk(10, metric) instead of all results

# Increase Grafana memory limit:
docker-compose.yml: mem_limit: 2gb

# Adjust refresh intervals:
- Real-time: 10s
- Normal: 30-60s
- Historical: 5 minutes
```

### Issue: Dashboard Import Failed

**Symptoms:**
- "Error importing dashboard"
- "Invalid JSON"
- Datasource mismatch

**Solutions:**

```bash
# Validate JSON syntax
jq . infra/grafana/dashboards/security_overview.json

# Check dashboard JSON structure
grep -E '"uid"|"title"|"panels"' infra/grafana/dashboards/security_overview.json

# Manually create dashboard if import fails:
# 1. Create new dashboard
# 2. Add panels one by one
# 3. Copy queries from JSON file
# 4. Save and export
```

## Security Considerations

### 1. Access Control

**Restrict Dashboard Access:**
```bash
# Set Grafana roles
- Admin: Full access
- Editor: Create/edit dashboards
- Viewer: Read-only access

# Use folder-level permissions
Settings > Folders > Manage folder access
```

**Recommended Setup:**
- Admins: 2-3 senior engineers
- Editors: 5-10 operations team members
- Viewers: Entire security team

### 2. Sensitive Data Protection

**Do Not Display in Dashboards:**
- ❌ API keys or tokens
- ❌ User passwords
- ❌ Personally identifiable information (PII)
- ❌ SQL query contents
- ❌ Full attack payloads

**Masking Strategies:**
- Replace sensitive values: `***MASKED***`
- Show only first/last 4 chars: `abcd****wxyz`
- Hash sensitive data: `SHA256(value)`
- Use pattern matching: `IP: XXX.XXX.XXX.XXX`

### 3. Audit Logging

**Track Dashboard Access:**
- Enable Grafana audit logging
- Log all dashboard modifications
- Log alert rule changes
- Correlate with security events

**Configuration:**
```yaml
# docker-compose.yml
grafana:
  environment:
    GF_LOG_MODE: file
    GF_LOG_LEVEL: info
    GF_AUDIT_LOGGING_ENABLED: "true"
```

### 4. Network Security

**Secure Grafana Access:**
```bash
# Use reverse proxy (Nginx)
server {
  listen 443 ssl;
  server_name grafana.example.com;
  ssl_certificate /etc/ssl/certs/cert.pem;
  ssl_certificate_key /etc/ssl/private/key.pem;

  location / {
    proxy_pass http://grafana:3000;
  }
}

# Only allow internal IPs
docker-compose.yml:
  ports:
    - "127.0.0.1:3000:3000"  # Localhost only
```

### 5. Credential Management

**API Tokens:**
- Rotate tokens regularly (quarterly)
- Use service accounts for automation
- Revoke unused tokens
- Document token purpose

**Database Credentials:**
- Use environment variables (not hardcoded)
- Rotate credentials quarterly
- Use separate DB user per environment
- Enable DB audit logging

**Slack/Email Secrets:**
- Store in secure credential management
- Rotate API keys quarterly
- Use service accounts
- Audit integration changes

### 6. Data Retention

**Prometheus Retention:**
```yaml
# prometheus.yml
global:
  retention: 15d  # Keep 15 days of metrics
```

**Elasticsearch Retention:**
```bash
# Set index lifecycle policy
PUT _ilm/policy/security-logs-policy
{
  "policy": "security-logs-policy",
  "phases": {
    "hot": {
      "min_age": "0d",
      "actions": { ... }
    },
    "delete": {
      "min_age": "90d",
      "actions": { "delete": {} }
    }
  }
}
```

### 7. Alerting Security

**Prevent Alert Fatigue:**
- Set thresholds carefully (avoid false alarms)
- Require explicit acknowledgment
- Escalate only critical alerts
- Review alert effectiveness monthly

**Prevent Alert Suppression:**
- Require approval for muting
- Log all alert modifications
- Set maximum mute duration (1-24h)
- Review muted alerts weekly

## Dashboard Maintenance

### Weekly Tasks

- [ ] Review alert rules for accuracy
- [ ] Check false positive rate
- [ ] Verify all datasources connected
- [ ] Review incident metrics (MTTD, MTTR)
- [ ] Check storage usage (Prometheus, Elasticsearch)

### Monthly Tasks

- [ ] Update threshold values based on trends
- [ ] Review and optimize slow queries
- [ ] Rotate API tokens
- [ ] Audit dashboard access logs
- [ ] Test disaster recovery procedures

### Quarterly Tasks

- [ ] Update security rules and patterns
- [ ] Review and update alerting rules
- [ ] Assess dashboard effectiveness
- [ ] Train team on new features
- [ ] Update documentation

## Support & Resources

### Documentation

- **HoloLoom Main Docs**: `CLAUDE.md`
- **Security Architecture**: `docs/ARCHITECTURE_VISUAL_MAP.md`
- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **Elasticsearch Docs**: https://www.elastic.co/guide/en/elasticsearch/reference/

### Contact

- **Security Team**: security@hololoom.com
- **On-Call**: Page from incident management system
- **Issues**: File GitHub issue with `[grafana]` tag

### Performance Benchmarks

Expected dashboard performance:

| Dashboard | Query Time | Render Time | Total |
|-----------|------------|------------|-------|
| Security Overview | 200ms | 800ms | 1000ms |
| Authentication | 250ms | 900ms | 1150ms |
| Attacks | 150ms | 700ms | 850ms |
| Anomalies | 300ms | 1000ms | 1300ms |
| Incidents | 400ms | 1200ms | 1600ms |

---

**Last Updated**: 2025-11-15
**Version**: 1.0.0
**Status**: Production Ready
**Maintainers**: HoloLoom Security Team
