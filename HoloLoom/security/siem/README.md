# SIEM Integration for HoloLoom Security Pipeline

**Status**: ✅ Complete (Phase 3 - November 15, 2025)
**Location**: `HoloLoom/security/siem/`
**Total Code**: 2,106 lines (implementation + tests + demos)

## Overview

HoloLoom's SIEM (Security Information and Event Management) integration provides structured security logging with support for multiple enterprise SIEM backends (Splunk, ELK/Elasticsearch, Datadog).

### Key Features

- **Multi-Backend Support**: Splunk, ELK, Datadog with unified interface
- **Structured Logging**: JSON format with security event taxonomy
- **Automatic PII Redaction**: Emails, SSNs, credit cards, phone numbers
- **Circuit Breaker**: Graceful degradation when SIEM backend unavailable
- **Buffering & Batching**: Async log forwarding with retry logic
- **Fallback Storage**: File-based logging when backend down
- **Query Interface**: Search and retrieve events from SIEM backends

## Quick Start

```python
from HoloLoom.security.siem import (
    SIEMIntegration,
    SIEMConfig,
    create_security_event,
    SecurityEventCategory,
    SecurityEventSubcategory,
)

# Configure SIEM
config = SIEMConfig(
    backend="splunk",
    splunk_config={
        "hec_url": "https://splunk.example.com:8088",
        "hec_token": "your-hec-token",
        "index": "hololoom_security",
    },
    buffer_size=1000,
    flush_interval=10.0,
    enable_fallback=True,
)

# Create SIEM integration
async with SIEMIntegration(config, backend) as siem:
    # Log security event
    event = create_security_event(
        category=SecurityEventCategory.ATTACK,
        subcategory=SecurityEventSubcategory.SQL_INJECTION,
        action="query_database",
        blocked=True,
        risk_score=9.0,
        source_ip="192.168.1.100",
        user_id="user@example.com",
        payload="SELECT * FROM users",
    )

    await siem.log_event(event)
```

## Security Event Taxonomy

### Categories (6)

| Category | Description | Subcategories |
|----------|-------------|---------------|
| **AUTH** | Authentication events | 6 (login, logout, failed_auth, mfa_challenge, password_reset, session_expired) |
| **AUTHZ** | Authorization events | 4 (permission_denied, role_change, privilege_escalation, access_granted) |
| **ATTACK** | Attack attempts | 8 (sql_injection, xss, csrf, brute_force, dos, path_traversal, command_injection, malicious_payload) |
| **DATA** | Data access events | 6 (query, export, delete, modify, bulk_access, sensitive_data_access) |
| **SYSTEM** | System events | 5 (startup, shutdown, config_change, service_error, resource_exhaustion) |
| **INCIDENT** | Security incidents | 4 (breach_detected, anomaly_detected, policy_violation, data_leak) |

**Total**: 6 categories, 33 subcategories, 5 severity levels

### Severity Levels

Auto-inferred from event characteristics:

- **CRITICAL**: High-risk attacks (score ≥8.0), incidents, breaches
- **ERROR**: Medium-risk attacks (score ≥6.0), service errors
- **WARNING**: Low-risk attacks (score ≥4.0), failed auth, permission denied
- **INFO**: Normal operations, successful auth
- **DEBUG**: Diagnostic information

## PII Redaction

Automatic PII redaction for compliance:

```python
from HoloLoom.security.siem import PIIRedactor

text = "Contact: alice@example.com, SSN: 123-45-6789"
redacted = PIIRedactor.redact(text)
# Result: "Contact: [EMAIL_REDACTED], SSN: [SSN_REDACTED]"

# User ID hashing
user_id = "alice@example.com"
hashed = PIIRedactor.hash_user_id(user_id)
# Result: "a1b2c3d4e5f6g7h8" (16-char hash)
```

**Patterns Detected**:
- Emails: `[EMAIL_REDACTED]`
- SSNs: `[SSN_REDACTED]`
- Credit Cards: `[CC_REDACTED]`
- Phone Numbers: `[PHONE_REDACTED]`
- IP Addresses: `[IP_REDACTED]` (optional)

## SIEM Backends

### 1. Splunk (HTTP Event Collector)

```python
from HoloLoom.security.siem import create_splunk_backend

backend = create_splunk_backend({
    "hec_url": "https://splunk.example.com:8088",
    "hec_token": "your-hec-token",
    "index": "hololoom_security",
    "source": "hololoom",
    "verify_ssl": True,
})
```

**Query Example (SPL)**:
```spl
index=hololoom_security category=ATTACK risk_score>=7.0
| stats count by subcategory, blocked
| timechart span=1h count by subcategory
```

### 2. ELK/Elasticsearch

```python
from HoloLoom.security.siem import create_elk_backend

backend = create_elk_backend({
    "es_url": "https://elasticsearch.example.com:9200",
    "username": "elastic",
    "password": "your-password",
    "index_pattern": "hololoom-security-{date}",
    "verify_ssl": True,
})
```

**Query Example (Elasticsearch DSL)**:
```json
GET hololoom-security-*/_search
{
  "query": {
    "bool": {
      "must": [
        {"term": {"category": "ATTACK"}},
        {"range": {"risk_score": {"gte": 7.0}}}
      ]
    }
  },
  "aggs": {
    "by_type": {
      "terms": {"field": "subcategory"}
    }
  }
}
```

### 3. Datadog

```python
from HoloLoom.security.siem import create_datadog_backend

backend = create_datadog_backend({
    "api_key": "your-datadog-api-key",
    "site": "datadoghq.com",
    "service": "hololoom",
    "tags": ["env:production", "team:security"],
})
```

**Query Example (Datadog)**:
```
service:hololoom source:security @category:ATTACK @risk_score:>=7.0
| group by @subcategory, @blocked
```

## Architecture

### Components

```
SIEMIntegration (Orchestrator)
├── LogBuffer (Thread-safe buffer)
├── CircuitBreaker (Failure detection)
├── SIEMBackend (Protocol)
│   ├── SplunkBackend
│   ├── ELKBackend
│   └── DatadogBackend
└── FileFallbackBackend (Local storage)
```

### Workflow

```
1. Security Event Created
   ↓
2. PII Auto-Redaction
   ↓
3. Add to LogBuffer
   ↓
4. Async Flush (every 10s or buffer full)
   ↓
5. Circuit Breaker Check
   ↓
6. Send to SIEM Backend (with retry)
   ├─ Success → Stats Updated
   └─ Failure → Fallback to File
```

## Configuration

### SIEMConfig Options

```python
config = SIEMConfig(
    # Backend selection
    backend="splunk",  # "splunk", "elk", "datadog", "file"

    # Buffering
    buffer_size=1000,        # Max events in buffer
    flush_interval=10.0,     # Seconds between flushes
    batch_size=100,          # Max events per batch

    # Retry logic
    max_retries=3,
    retry_delay=1.0,         # Initial delay (seconds)
    retry_backoff=2.0,       # Exponential backoff multiplier

    # Circuit breaker
    failure_threshold=5,     # Failures before opening circuit
    recovery_timeout=60.0,   # Seconds before trying half-open
    success_threshold=2,     # Successes needed to close circuit

    # Retention
    retention_days=90,       # Total retention
    hot_storage_days=7,      # Fast storage duration

    # Fallback
    fallback_dir=Path("./security_logs"),
    enable_fallback=True,

    # Backend-specific configs
    splunk_config={...},
    elk_config={...},
    datadog_config={...},
)
```

## Circuit Breaker

Protects against SIEM backend failures:

```
CLOSED (Normal) → OPEN (Degraded) → HALF_OPEN (Testing) → CLOSED
```

**States**:
- **CLOSED**: Normal operation, all requests sent
- **OPEN**: Backend down, using fallback only
- **HALF_OPEN**: Testing recovery, limited requests

**Triggers**:
- Open circuit: 5 consecutive failures
- Try half-open: After 60s timeout
- Close circuit: 2 consecutive successes

## Integration with HoloLoom Security

### Components

```python
from HoloLoom.security.siem import SIEMIntegration, create_security_event

# Integrate with SafetyGuardrails
async def gate_action(action, context):
    result = await guardrails.gate_action(action, context)

    # Log to SIEM
    event = create_security_event(
        category=SecurityEventCategory.AUTHZ,
        subcategory=SecurityEventSubcategory.PERMISSION_DENIED,
        action=action,
        blocked=result.blocked,
        risk_score=result.risk_score,
        metadata={"reason": result.reason},
    )
    await siem.log_event(event)

    return result
```

**Integration Points**:
- `SafetyGuardrails` → Log blocked/allowed actions
- `DeceptionDetection` → Log deception attempts
- `InstrumentalConvergence` → Log power-seeking behavior
- `AuditTrail` → Forward audit entries
- `RateLimiter` → Log rate limit violations
- `WAF` → Log attack attempts

## Performance

| Operation | Overhead | Notes |
|-----------|----------|-------|
| **Event creation** | <0.1ms | Including PII redaction |
| **Buffer add** | <0.01ms | Thread-safe async |
| **Batch flush** | 10-50ms | Depends on backend latency |
| **Circuit breaker check** | <0.001ms | State check only |
| **Fallback write** | 1-5ms | JSON to disk |

**Per-Query Overhead**: <0.1ms (buffer add only, flush is async)

## Testing

```bash
# Run tests (requires pytest)
pytest HoloLoom/security/tests/test_siem.py -v

# Run demo
python demos/demo_siem_minimal.py
```

**Test Coverage**:
- ✓ Security event creation (8 tests)
- ✓ PII redaction (6 tests)
- ✓ Log buffer (3 tests)
- ✓ Circuit breaker (4 tests)
- ✓ SIEM integration (6 tests)
- ✓ Event serialization (3 tests)
- ✓ Severity inference (6 tests)

**Total**: 36 tests

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `taxonomy.py` | 378 | Event classification and PII redaction |
| `core.py` | 468 | Main SIEM integration orchestrator |
| `splunk_backend.py` | 228 | Splunk HEC integration |
| `elk_backend.py` | 268 | Elasticsearch integration |
| `datadog_backend.py` | 257 | Datadog Logs API integration |
| `test_siem.py` | 476 | Comprehensive test suite |
| **Total** | **2,075** | **Production code + tests** |

## Example: Full Production Setup

```python
import asyncio
from pathlib import Path
from HoloLoom.security.siem import (
    SIEMIntegration,
    SIEMConfig,
    create_splunk_backend,
    create_security_event,
    SecurityEventCategory,
    SecurityEventSubcategory,
)

async def main():
    # Create Splunk backend
    backend = create_splunk_backend({
        "hec_url": "https://splunk.example.com:8088",
        "hec_token": "your-hec-token",
        "index": "hololoom_security",
    })

    # Configure SIEM
    config = SIEMConfig(
        backend="splunk",
        buffer_size=1000,
        flush_interval=10.0,
        batch_size=100,
        fallback_dir=Path("./security_logs"),
        enable_fallback=True,
    )

    # Start SIEM integration
    async with SIEMIntegration(config, backend) as siem:
        # Log attack attempt
        attack = create_security_event(
            category=SecurityEventCategory.ATTACK,
            subcategory=SecurityEventSubcategory.SQL_INJECTION,
            action="query_database",
            blocked=True,
            risk_score=9.0,
            source_ip="10.0.0.5",
            payload="SELECT * FROM users WHERE id='1' OR '1'='1'",
        )
        await siem.log_event(attack)

        # Log data export
        export = create_security_event(
            category=SecurityEventCategory.DATA,
            subcategory=SecurityEventSubcategory.EXPORT,
            action="export_user_data",
            blocked=False,
            risk_score=4.5,
            target="users_table",
            metadata={"record_count": 1500},
        )
        await siem.log_event(export)

        # Check health
        health = await siem.health_check()
        print(f"Backend healthy: {health['backend_healthy']}")
        print(f"Circuit state: {health['circuit_state']}")
        print(f"Stats: {health['stats']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Monitoring & Alerting

### Prometheus Metrics

```python
# Export SIEM stats to Prometheus
from prometheus_client import Counter, Gauge

events_logged = Counter('siem_events_logged_total', 'Total events logged')
events_sent = Counter('siem_events_sent_total', 'Events sent to backend')
events_failed = Counter('siem_events_failed_total', 'Failed events')
circuit_state = Gauge('siem_circuit_state', 'Circuit breaker state')

# Update metrics
events_logged.inc(siem.stats['events_logged'])
events_sent.inc(siem.stats['events_sent'])
circuit_state.set(1 if siem.circuit_breaker.state == CircuitState.CLOSED else 0)
```

### Alert Rules

**Splunk**:
```spl
index=hololoom_security category=ATTACK blocked=false
| stats count by source_ip
| where count > 10
| alert when count > 10 in last 5 minutes
```

**Elasticsearch (Watcher)**:
```json
{
  "trigger": {
    "schedule": {"interval": "5m"}
  },
  "input": {
    "search": {
      "request": {
        "indices": ["hololoom-security-*"],
        "body": {
          "query": {
            "bool": {
              "must": [
                {"term": {"category": "ATTACK"}},
                {"term": {"blocked": false}}
              ]
            }
          }
        }
      }
    }
  },
  "condition": {
    "compare": {"ctx.payload.hits.total": {"gt": 10}}
  }
}
```

## Best Practices

1. **PII Protection**: Always enable `auto_redact=True` (default)
2. **Circuit Breaker**: Set appropriate thresholds for your SLA
3. **Fallback Storage**: Enable for critical production systems
4. **Retention**: Configure based on compliance requirements
5. **Batching**: Tune `batch_size` and `flush_interval` for latency/throughput tradeoff
6. **Monitoring**: Track circuit breaker state and fallback usage

## Troubleshooting

### Backend Connection Failures

```python
# Check health
health = await siem.health_check()
if not health['backend_healthy']:
    print("Backend unavailable, using fallback")
    print(f"Circuit state: {health['circuit_state']}")
```

### High Buffer Usage

```python
# Monitor buffer
if health['buffer_size'] > health['buffer_max_size'] * 0.8:
    print("Warning: Buffer 80% full")
    # Increase flush_interval or batch_size
```

### Fallback File Accumulation

```bash
# Archive old fallback files
find ./security_logs -name "security_events_*.json" -mtime +7 -exec gzip {} \;
```

## Future Enhancements

- **Additional Backends**: Sumo Logic, Azure Sentinel, AWS Security Lake
- **Compression**: Gzip compression for large batches
- **Encryption**: TLS 1.3 for all backend connections
- **Correlation**: Event correlation engine
- **ML Detection**: Anomaly detection on event patterns

## License

Part of HoloLoom security framework - see main LICENSE file.

---

**Created**: November 15, 2025
**Version**: 1.0.0
**Status**: Production Ready
