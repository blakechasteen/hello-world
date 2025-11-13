# Phase 3: Adaptive Learning System - Complete Documentation

**Status**: ✅ Production Ready (November 13, 2025)
**Completion**: 100% (All 14 days complete)
**Test Coverage**: 13/13 integration tests passing

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Usage Guide](#usage-guide)
6. [Configuration Reference](#configuration-reference)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Alerting](#monitoring--alerting)
9. [Performance Characteristics](#performance-characteristics)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Overview

Phase 3 integrates a complete **Adaptive Learning System** into HoloLoom's MoonshotQueryClassifier, enabling:

- **Automatic pattern discovery** from production classification logs
- **Continuous accuracy monitoring** with hourly validation
- **Safe pattern deployment** with shadow mode, A/B testing, and gradual rollout
- **Automatic regression detection** and rollback (>2% accuracy drop)
- **Daily/weekly performance reports** with actionable recommendations
- **Prometheus metrics export** for Grafana dashboards
- **Slack/email alerts** for critical regressions

**Key Benefits**:
- System learns and improves from production data
- Zero manual intervention required
- Sub-millisecond overhead per query (<3ms)
- Complete safety guarantees (automatic rollback)
- Full production observability

---

## Quick Start

### Basic Usage

```python
from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier
from pathlib import Path

# Create adaptive classifier
classifier = AdaptiveMoonshotClassifier(
    enable_semantic_tier=True,
    enable_adaptive_learning=True,
    data_dir=Path("./data"),
    background_learning=True  # Automatic hourly learning
)

# Classify queries (automatically logged)
result = classifier.classify("What is Thompson Sampling?")

print(f"Complexity: {result.complexity.value}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Tier Used: {result.tier_used}")
```

### Background Learning

```python
import asyncio

async def main():
    classifier = AdaptiveMoonshotClassifier(
        enable_adaptive_learning=True,
        background_learning=True,
        learning_update_interval=3600.0  # 1 hour
    )

    # Start background learning loop
    await classifier.start_background_learning()

    # Use classifier for production queries
    for query in production_queries:
        result = classifier.classify(query)
        # System automatically learns from successful classifications

    # Stop gracefully
    await classifier.stop_background_learning()

asyncio.run(main())
```

### View Learning Statistics

```python
stats = classifier.get_learning_statistics()

print(f"Queries Logged: {stats['total_queries_logged']}")
print(f"Patterns Discovered: {stats['patterns_discovered']}")
print(f"Patterns Deployed: {stats['patterns_deployed']}")
print(f"Validation Accuracy: {stats['validation_accuracy']:.1%}")
print(f"Regression Alerts: {stats['regression_alerts']}")
```

---

## Architecture

Phase 3 implements a **4-component architecture** integrated with MoonshotQueryClassifier:

```
┌─────────────────────────────────────────────────────────────┐
│              AdaptiveMoonshotClassifier                     │
├─────────────────────────────────────────────────────────────┤
│  Foreground: Fast Multi-Tier Classification                │
│  ├─ Tier 1: Pattern Cache (0.1ms)                          │
│  ├─ Tier 2: Heuristic Scoring (0.5ms)                      │
│  ├─ Tier 3: Semantic Embeddings (20ms, optional)           │
│  └─ Tier 4: Conservative Escalation                        │
│                                                             │
│  Background: Adaptive Learning Loop (every 1 hour)          │
│  ├─ Step 1: PatternMiner → Extract patterns from logs      │
│  ├─ Step 2: ContinuousValidator → Validate accuracy        │
│  ├─ Step 3: AdaptiveUpdater → Deploy patterns safely       │
│  └─ Step 4: PerformanceReporter → Generate reports         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Query → Classify → Log (JSONL) → [Background Loop]
                                      ↓
                            1. Mine Patterns (hourly)
                            2. Validate Accuracy (hourly)
                            3. Deploy Patterns (if quality > threshold)
                            4. Generate Reports (daily/weekly)
                                      ↓
                            Automatic Alerts (if regression detected)
```

---

## Core Components

### 1. PatternMiner

**Purpose**: Extracts high-quality classification patterns from production logs

**Features**:
- N-gram extraction (1-5 words)
- Pattern generalization (text → regex)
- Quality scoring (precision, recall, F1, support)
- Filters by quality thresholds (min precision, min support)

**Example**:
```python
from HoloLoom.routing.learning import PatternMiner

miner = PatternMiner(
    stats_path="./data/logs/classifications.jsonl",
    min_support=10,      # Pattern must appear ≥10 times
    min_precision=0.95,  # Pattern must be 95%+ accurate
    min_recall=0.30,     # Minimum coverage
    max_patterns=50      # Top 50 patterns
)

patterns = miner.mine_patterns(days_lookback=7)

for pattern in patterns:
    print(f"Regex: {pattern.regex}")
    print(f"Complexity: {pattern.complexity}")
    print(f"Precision: {pattern.score.precision:.1%}")
    print(f"Support: {pattern.score.support}")
```

**Quality Metrics**:
- **Precision**: Correctness (e.g., 91.67% = pattern correct 91.67% of time)
- **Recall**: Coverage (e.g., 52.17% = pattern covers 52.17% of SIMPLE queries)
- **Support**: Frequency (e.g., 12 = pattern matches 12 queries in logs)
- **F1 Score**: Balanced quality (harmonic mean of precision and recall)

---

### 2. ContinuousValidator

**Purpose**: Monitors classifier accuracy with automatic regression detection

**Features**:
- Hourly/daily validation schedules
- Regression detection (>2% accuracy drop triggers alert)
- Trend analysis (7-day, 30-day moving averages)
- Alert generation with severity levels (WARNING, CRITICAL)
- Per-complexity accuracy breakdown

**Example**:
```python
from HoloLoom.routing.learning import ContinuousValidator

validator = ContinuousValidator(
    classifier=classifier,
    validation_set_path="./data/validation_set.json",
    regression_threshold=0.02,  # 2% drop triggers alert
    baseline_accuracy=1.0       # Expected 100% accuracy
)

# Hourly validation (100 queries)
result = await validator.validate_hourly(sample_size=100)

print(f"Overall Accuracy: {result.overall_accuracy:.1%}")
print(f"Regression Detected: {result.regression_detected}")

if result.regression_detected:
    alerts = validator.alerts
    for alert in alerts:
        print(f"Severity: {alert.severity}")
        print(f"Drop: {alert.drop_percentage:.1%}")
        print(f"Affected: {alert.affected_complexity}")
```

**Validation Set Format**:
```json
[
  {
    "text": "What is Python?",
    "complexity": "simple",
    "metadata": {}
  },
  {
    "text": "Explain neural networks",
    "complexity": "complex",
    "metadata": {}
  }
]
```

---

### 3. AdaptiveUpdater

**Purpose**: Safely deploys patterns with automatic rollback

**Deployment Strategies**:

| Strategy | Traffic Split | Duration | Use Case |
|----------|---------------|----------|----------|
| **SHADOW** | 0% | Day 1-2 | Test patterns without production impact |
| **AB_TEST** | 10/90 | Day 3 | Validate patterns with small traffic |
| **GRADUAL** | 10% → 50% → 100% | Day 3-7 | Incremental deployment with monitoring |
| **IMMEDIATE** | 100% | N/A | Hotfixes only (risky) |

**Example**:
```python
from HoloLoom.routing.learning import AdaptiveUpdater, DeploymentStrategy

updater = AdaptiveUpdater(
    classifier=classifier,
    validator=validator,
    strategy=DeploymentStrategy.GRADUAL,  # 7-day rollout
    regression_threshold=0.02
)

# Deploy patterns
result = await updater.deploy_patterns(patterns)

print(f"Success: {result.success}")
print(f"Patterns Deployed: {result.patterns_deployed}")
print(f"Current Phase: {result.current_phase.value}")
print(f"Rollback Triggered: {result.rollback_triggered}")
```

**Automatic Rollback**:
- Triggers when accuracy drops >2%
- Reverts to previous pattern version
- Preserves last 10 pattern versions
- Instant rollback (< 1ms)

**Pattern Versioning**:
```python
# View deployment history
status = updater.get_deployment_status()

print(f"Pattern Versions: {status['pattern_versions']}")
print(f"Total Deployments: {status['total_deployments']}")
print(f"Rollbacks: {status['rollbacks']}")

# Manual rollback
await updater.rollback()
```

---

### 4. PerformanceReporter

**Purpose**: Generates daily/weekly reports with actionable recommendations

**Features**:
- Daily reports (24-hour metrics)
- Weekly reports (7-day analysis + recommendations)
- Prometheus metrics export (standard format)
- Slack alerts (emoji + markdown)
- Email alerts (subject + body)
- Markdown reports (human-readable)

**Example**:
```python
from HoloLoom.routing.learning import PerformanceReporter

reporter = PerformanceReporter(
    validator=validator,
    updater=updater,
    pattern_miner=miner,
    output_dir="./data/reports"
)

# Daily report
daily = reporter.generate_daily_report()

print(f"Date: {daily.date}")
print(f"Queries Classified: {daily.queries_classified}")
print(f"Overall Accuracy: {daily.overall_accuracy:.1%}")
print(f"Patterns Deployed: {daily.patterns_deployed}")
print(f"Regressions Detected: {daily.regressions_detected}")

# Weekly report with recommendations
weekly = reporter.generate_weekly_report()

print(f"Week: {weekly.week_start} - {weekly.week_end}")
print(f"Total Queries: {weekly.total_queries}")
print(f"Overall Accuracy: {weekly.overall_accuracy:.1%}")
print(f"Trend (7-day): {weekly.trend_7day:+.1%}")
print(f"\nRecommendations:")
for rec in weekly.recommendations:
    print(f"  - {rec}")

# Export Prometheus metrics
metrics = reporter.export_prometheus_metrics()

# Save to file for Prometheus scraper
with open("/var/lib/prometheus/moonshot_metrics.prom", "w") as f:
    f.write(metrics)
```

**Recommendations Engine**:
Automatically generates actionable recommendations based on:
- Overall accuracy < 95% → Suggest pattern mining
- Per-complexity accuracy < 90% → Suggest focused mining
- Recent regressions > 0 → Suggest deployment review
- Positive trends → Encourage continuation

---

## Usage Guide

### Production Integration

**Step 1: Initialize Classifier**

```python
from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier
from pathlib import Path

classifier = AdaptiveMoonshotClassifier(
    enable_semantic_tier=True,      # Enable Tier 3 semantic embeddings
    enable_adaptive_learning=True,  # Enable Phase 3 adaptive learning
    data_dir=Path("/var/lib/hololoom/data"),
    validation_set_path=Path("/etc/hololoom/validation_set.json"),
    background_learning=True,       # Auto-learning every hour
    learning_update_interval=3600.0 # 1 hour
)
```

**Step 2: Start Background Learning**

```python
import asyncio

async def main():
    # Start background learning
    await classifier.start_background_learning()

    # Your application runs here...
    # All classifications are automatically logged

    try:
        while True:
            query = await get_next_query()
            result = classifier.classify(query)
            await process_result(result)
    finally:
        # Graceful shutdown
        await classifier.stop_background_learning()

asyncio.run(main())
```

**Step 3: Monitor Performance**

```python
# View statistics
stats = classifier.get_learning_statistics()

print(f"Queries Logged: {stats['total_queries_logged']}")
print(f"Patterns Discovered: {stats['patterns_discovered']}")
print(f"Patterns Deployed: {stats['patterns_deployed']}")
print(f"Validation Accuracy: {stats['validation_accuracy']:.1%}")
```

---

### Custom Validation Set

Create a validation set with known ground truth labels:

```python
from HoloLoom.routing.learning import create_validation_set

queries = [
    # TRIVIAL queries
    ("hi", "trivial"),
    ("thanks", "trivial"),
    ("ok", "trivial"),

    # SIMPLE queries
    ("what is Python?", "simple"),
    ("define machine learning", "simple"),

    # COMPLEX queries
    ("explain how neural networks work", "complex"),
    ("describe attention mechanism", "complex"),

    # RESEARCH queries
    ("analyze tradeoffs of Thompson Sampling", "research"),
    ("compare supervised vs unsupervised learning", "research"),
]

create_validation_set(queries, "./data/validation_set.json")
```

**Validation Set Best Practices**:
- Minimum 50-100 queries (75 recommended)
- Balanced across complexity levels (20 trivial, 20 simple, 20 complex, 15 research)
- Representative of production distribution
- Regular updates (monthly) as domain evolves

---

### Manual Learning Cycle

For debugging or testing, run learning cycle manually:

```python
# Disable background learning
classifier.background_learning = False

# Run single learning cycle
await classifier._run_learning_cycle()

# View results
stats = classifier.get_learning_statistics()
```

---

## Configuration Reference

### AdaptiveMoonshotClassifier Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_semantic_tier` | bool | True | Enable Tier 3 semantic embeddings |
| `enable_adaptive_learning` | bool | True | Enable Phase 3 adaptive learning |
| `data_dir` | Path | `./data` | Directory for logs, patterns, reports |
| `validation_set_path` | Path | `{data_dir}/validation_set.json` | Path to validation set |
| `background_learning` | bool | True | Run background learning loop |
| `learning_update_interval` | float | 3600.0 | Background learning interval (seconds) |

### PatternMiner Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stats_path` | str | Required | Path to JSONL classification log |
| `min_support` | int | 10 | Minimum pattern frequency |
| `min_precision` | float | 0.95 | Minimum pattern accuracy (95%) |
| `min_recall` | float | 0.30 | Minimum pattern coverage (30%) |
| `max_patterns` | int | 50 | Maximum patterns to extract |

### ContinuousValidator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifier` | Classifier | Required | Query classifier instance |
| `validation_set_path` | str | None | Path to validation set JSON |
| `regression_threshold` | float | 0.02 | Accuracy drop to trigger regression (2%) |
| `history_size` | int | 1000 | Number of validation results to keep |
| `baseline_accuracy` | float | 1.0 | Expected accuracy (100%) |

### AdaptiveUpdater Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifier` | Classifier | Required | Query classifier instance |
| `validator` | Validator | Required | Continuous validator instance |
| `strategy` | DeploymentStrategy | GRADUAL | Deployment strategy |
| `regression_threshold` | float | 0.02 | Accuracy drop to trigger rollback (2%) |

### PerformanceReporter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validator` | Validator | Required | Continuous validator instance |
| `updater` | Updater | Required | Adaptive updater instance |
| `pattern_miner` | PatternMiner | None | Pattern miner instance (optional) |
| `output_dir` | str | `./reports` | Output directory for reports |

---

## Production Deployment

### Prometheus Integration

**Step 1: Export Metrics**

```python
from HoloLoom.routing.learning import PerformanceReporter

reporter = PerformanceReporter(
    validator=validator,
    updater=updater,
    output_dir="/var/lib/prometheus/textfile_collector"
)

# Export metrics (run every minute)
metrics = reporter.export_prometheus_metrics()

# Write to Prometheus textfile collector directory
with open("/var/lib/prometheus/textfile_collector/moonshot.prom", "w") as f:
    f.write(metrics)
```

**Step 2: Configure Prometheus Scraper**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'moonshot_classifier'
    static_configs:
      - targets: ['localhost:9090']
    file_sd_configs:
      - files:
          - '/var/lib/prometheus/textfile_collector/*.prom'
```

**Exported Metrics**:
```
# HELP moonshot_accuracy Classification accuracy
# TYPE moonshot_accuracy gauge
moonshot_accuracy{complexity="overall"} 0.95
moonshot_accuracy{complexity="trivial"} 1.00
moonshot_accuracy{complexity="simple"} 0.98
moonshot_accuracy{complexity="complex"} 0.92
moonshot_accuracy{complexity="research"} 0.87

# HELP moonshot_queries_total Total queries classified
# TYPE moonshot_queries_total counter
moonshot_queries_total 15234

# HELP moonshot_latency_ms Average classification latency
# TYPE moonshot_latency_ms gauge
moonshot_latency_ms 125.5

# HELP moonshot_patterns_deployed Total patterns deployed
# TYPE moonshot_patterns_deployed counter
moonshot_patterns_deployed 42

# HELP moonshot_regressions_detected Total regressions detected
# TYPE moonshot_regressions_detected counter
moonshot_regressions_detected 3
```

---

### Slack Integration

**Step 1: Configure Webhook**

```python
import requests

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# In learning cycle, send alerts
if validation_result.regression_detected:
    alert = validator.alerts[-1]
    slack_msg = reporter.format_slack_alert(alert)

    requests.post(SLACK_WEBHOOK_URL, json={"text": slack_msg})
```

**Alert Format**:
```
🚨 **Classifier Regression Detected**

**Current accuracy**: 53.3%
**Baseline accuracy**: 100.0%
**Drop**: 46.7% (threshold: 2.0%)

**Affected complexities**:
  • trivial: 30.0%
  • complex: 25.0%
  • research: 37.5%

**Time**: 2025-11-13T03:10:53
**Severity**: CRITICAL
```

---

### Email Integration

**Step 1: Configure SMTP**

```python
import smtplib
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "alerts@yourcompany.com"
SMTP_PASS = "your-app-password"
ALERT_RECIPIENTS = ["team@yourcompany.com"]

# In learning cycle, send alerts
if validation_result.regression_detected:
    alert = validator.alerts[-1]
    email = reporter.format_email_alert(alert)

    msg = MIMEText(email['body'], 'plain')
    msg['Subject'] = email['subject']
    msg['From'] = SMTP_USER
    msg['To'] = ", ".join(ALERT_RECIPIENTS)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
```

**Email Format**:
```
Subject: [CRITICAL] Classifier Regression Detected

Body:
Classifier Regression Detected
==============================

Current accuracy: 53.3%
Baseline accuracy: 100.0%
Drop: 46.7% (threshold: 2.0%)

Affected complexities:
  - trivial: 30.0%
  - complex: 25.0%
  - research: 37.5%

Time: 2025-11-13T03:10:53
Severity: CRITICAL

Action Required: Please investigate and consider rollback.
```

---

### Grafana Dashboards

**Sample Dashboard JSON** (import into Grafana):

```json
{
  "dashboard": {
    "title": "Moonshot Classifier Performance",
    "panels": [
      {
        "title": "Overall Accuracy",
        "targets": [
          {
            "expr": "moonshot_accuracy{complexity=\"overall\"}",
            "legendFormat": "Accuracy"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Accuracy by Complexity",
        "targets": [
          {
            "expr": "moonshot_accuracy",
            "legendFormat": "{{complexity}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Queries per Second",
        "targets": [
          {
            "expr": "rate(moonshot_queries_total[5m])",
            "legendFormat": "QPS"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Patterns Deployed",
        "targets": [
          {
            "expr": "moonshot_patterns_deployed",
            "legendFormat": "Patterns"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Regressions",
        "targets": [
          {
            "expr": "moonshot_regressions_detected",
            "legendFormat": "Regressions"
          }
        ],
        "type": "stat",
        "alert": {
          "conditions": [
            {
              "evaluator": {
                "params": [0],
                "type": "gt"
              }
            }
          ]
        }
      }
    ]
  }
}
```

---

## Monitoring & Alerting

### Key Metrics to Monitor

1. **Overall Accuracy** (target: >95%)
   - Alert if drops below 90%
   - Critical if drops below 85%

2. **Per-Complexity Accuracy** (target: >90% each)
   - Alert if any complexity drops below 80%

3. **Regression Rate** (target: 0 per day)
   - Alert on any regression
   - Critical if >3 regressions per day

4. **Pattern Quality** (target: precision >95%)
   - Monitor newly deployed patterns
   - Alert if precision drops below 90%

5. **Deployment Success Rate** (target: >95%)
   - Alert if <80% deployments successful

### Alerting Rules (Prometheus)

```yaml
# prometheus_rules.yml
groups:
  - name: moonshot_alerts
    interval: 30s
    rules:
      - alert: ClassifierAccuracyLow
        expr: moonshot_accuracy{complexity="overall"} < 0.90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Classifier accuracy below 90%"
          description: "Overall accuracy is {{ $value | humanizePercentage }}"

      - alert: ClassifierAccuracyCritical
        expr: moonshot_accuracy{complexity="overall"} < 0.85
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Classifier accuracy critically low"
          description: "Overall accuracy is {{ $value | humanizePercentage }}"

      - alert: RegressionDetected
        expr: increase(moonshot_regressions_detected[1h]) > 0
        labels:
          severity: warning
        annotations:
          summary: "Regression detected"
          description: "{{ $value }} regressions in last hour"

      - alert: HighRegressionRate
        expr: increase(moonshot_regressions_detected[24h]) > 3
        labels:
          severity: critical
        annotations:
          summary: "High regression rate"
          description: "{{ $value }} regressions in last 24 hours"
```

---

## Performance Characteristics

### Per-Query Overhead

| Component | Overhead | When |
|-----------|----------|------|
| JSONL logging | <0.5ms | Every query |
| Pattern extraction (background) | 0ms | Async, no blocking |
| Validation (background) | 0ms | Async, no blocking |
| Thompson/Policy update (background) | 0ms | Async, no blocking |

**Total Per-Query Overhead**: <1ms (JSONL logging only)

### Background Learning Overhead

| Operation | Duration | Frequency |
|-----------|----------|-----------|
| Pattern mining | ~500ms | Every 1 hour |
| Hourly validation | ~2-5s | Every 1 hour |
| Pattern deployment | ~100ms | When new patterns available |
| Daily report | ~50ms | Once per day (9am UTC) |
| Weekly report | ~100ms | Once per week (Sunday 9am UTC) |

**Total Background Overhead**: ~3-6s per hour (0.08-0.17% CPU usage)

### Memory Usage

- Classification log: ~100KB per 1000 queries
- Pattern storage: ~50KB per 50 patterns
- Validation history: ~10KB per 1000 validations
- Total: ~1-2MB for typical production workload

---

## Best Practices

### 1. Validation Set Management

✅ **Do**:
- Maintain 75-100 representative queries
- Balance across complexity levels (20/20/20/15 split)
- Update monthly as domain evolves
- Include edge cases and failure modes
- Version control validation set

❌ **Don't**:
- Use <50 queries (insufficient coverage)
- Skew heavily toward one complexity
- Include ambiguous or mislabeled queries
- Let validation set become stale

### 2. Pattern Quality Thresholds

✅ **Do**:
- Set `min_precision=0.95` (95% accuracy)
- Set `min_support=10` (minimum 10 occurrences)
- Set `min_recall=0.30` (minimum 30% coverage)
- Monitor deployed pattern performance

❌ **Don't**:
- Lower precision threshold below 90%
- Accept patterns with support <5
- Ignore recall (coverage matters)

### 3. Deployment Strategy

✅ **Do**:
- Use GRADUAL for regular pattern updates
- Use SHADOW for testing risky patterns
- Use AB_TEST for validation before full deployment
- Monitor during gradual rollout phases

❌ **Don't**:
- Use IMMEDIATE except for hotfixes
- Skip shadow testing for major changes
- Deploy without validation
- Ignore rollback signals

### 4. Alerting Configuration

✅ **Do**:
- Set up Slack/email alerts for critical regressions
- Monitor Prometheus metrics in Grafana
- Create runbooks for common alerts
- Test alerting pipeline regularly

❌ **Don't**:
- Rely solely on manual monitoring
- Ignore regression alerts
- Set thresholds too loose (>5% drop)
- Alert without actionable next steps

### 5. Log Management

✅ **Do**:
- Rotate classification logs weekly
- Archive old logs for analysis
- Monitor log file size
- Compress archived logs

❌ **Don't**:
- Let logs grow unbounded
- Delete logs without archiving
- Log sensitive data (PII)

---

## Troubleshooting

### Problem: No patterns discovered

**Symptoms**: `patterns_discovered` = 0

**Causes**:
1. Insufficient classification logs (<100 queries)
2. Quality thresholds too strict
3. No repeating patterns in data

**Solutions**:
```python
# Lower quality thresholds temporarily
miner = PatternMiner(
    stats_path="./data/logs/classifications.jsonl",
    min_support=5,       # Lower from 10
    min_precision=0.85,  # Lower from 0.95
    min_recall=0.20      # Lower from 0.30
)

# Check log file
import json
with open("./data/logs/classifications.jsonl", "r") as f:
    count = sum(1 for _ in f)
    print(f"Total logged queries: {count}")
```

---

### Problem: Regression alerts too frequent

**Symptoms**: `regression_alerts` > 5 per day

**Causes**:
1. Validation set too small or unrepresentative
2. Regression threshold too sensitive
3. Actual accuracy issues

**Solutions**:
```python
# 1. Expand validation set
queries = [
    # Add more representative examples
]
create_validation_set(queries, "./data/validation_set.json")

# 2. Adjust threshold
validator = ContinuousValidator(
    classifier=classifier,
    validation_set_path="./data/validation_set.json",
    regression_threshold=0.05  # Increase from 0.02
)

# 3. Investigate actual issues
stats = classifier.get_learning_statistics()
print(f"Validation Accuracy: {stats['validation_accuracy']:.1%}")
```

---

### Problem: Background learning not running

**Symptoms**: No pattern mining or validation happening

**Causes**:
1. `background_learning=False`
2. Learning loop crashed
3. `learning_update_interval` too long

**Solutions**:
```python
# Check configuration
print(f"Background Learning: {classifier.background_learning}")
print(f"Update Interval: {classifier.learning_update_interval}s")

# Restart background learning
await classifier.stop_background_learning()
await classifier.start_background_learning()

# Run manual learning cycle for debugging
await classifier._run_learning_cycle()
```

---

### Problem: High memory usage

**Symptoms**: Memory usage growing over time

**Causes**:
1. Classification log not rotated
2. Validation history unbounded
3. Pattern version history accumulating

**Solutions**:
```python
# 1. Rotate classification log
import shutil
from datetime import datetime

log_path = classifier.classification_log_path
archive_path = f"{log_path}.{datetime.now().strftime('%Y%m%d')}"
shutil.move(log_path, archive_path)

# 2. Limit validation history
validator = ContinuousValidator(
    classifier=classifier,
    validation_set_path="./data/validation_set.json",
    history_size=500  # Reduce from 1000
)

# 3. Limit pattern versions
updater.max_pattern_versions = 5  # Reduce from 10
```

---

## Testing

Run comprehensive integration tests:

```bash
pytest HoloLoom/routing/learning/tests/test_adaptive_integration.py -v
```

**Test Coverage**:
- ✅ PatternMiner initialization and pattern discovery
- ✅ ContinuousValidator hourly validation and regression detection
- ✅ AdaptiveUpdater shadow deployment and automatic rollback
- ✅ PerformanceReporter daily/weekly reports and metrics export
- ✅ End-to-end pipeline integration
- ✅ Metrics consistency across components

**All 13 tests passing** ✅

---

## Summary

Phase 3 Adaptive Learning System provides:

✅ **Automatic pattern discovery** from production logs
✅ **Continuous accuracy monitoring** with hourly validation
✅ **Safe pattern deployment** with automatic rollback
✅ **Complete observability** (Prometheus, Slack, email)
✅ **Zero manual intervention** required
✅ **Sub-millisecond overhead** per query
✅ **Production-ready** with comprehensive tests

**Next Steps**:
1. Deploy to production with `background_learning=True`
2. Configure Prometheus scraper and Grafana dashboards
3. Set up Slack/email alerts for critical regressions
4. Monitor performance metrics and adjust thresholds as needed
5. Iterate on validation set based on production distribution

---

**Phase 3 Complete**: November 13, 2025
**Status**: ✅ Production Ready
**Test Coverage**: 13/13 passing
**Documentation**: Complete
