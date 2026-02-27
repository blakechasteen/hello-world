# HoloLoom Security Implementation Documentation

## Overview

This document describes the comprehensive security monitoring and protection system implemented for HoloLoom's production API servers. The system provides defense-in-depth with multiple layers of security.

**Created**: 2025-11-26

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Request                            │
└─────────────────────────────────┬───────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────┐
│            WAF (Web Application Firewall)                    │
│  • SQL Injection Detection      • XSS Detection              │
│  • Path Traversal Detection     • Command Injection          │
│  • Header Validation            • Request Size Limits        │
└─────────────────────────────────┬───────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────┐
│              Request Signing Validation                       │
│  • HMAC-SHA256 Signature        • Timestamp Validation       │
│  • API Key Management           • Replay Protection          │
└─────────────────────────────────┬───────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────┐
│                   Rate Limiting                              │
│  • Per-IP Limits                • Per-Endpoint Limits        │
│  • Sliding Window               • Auto-Blocking              │
└─────────────────────────────────┬───────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────┐
│                Security Monitoring                           │
│  • Event Logging                • Threshold Alerts           │
│  • Email/Slack Notifications    • Prometheus Metrics         │
│  • Real-time Dashboard          • Anomaly Detection          │
└─────────────────────────────────┬───────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────┐
│                   Application Logic                          │
│                  (HoloLoom API Endpoints)                    │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. WAF (Web Application Firewall)

**File**: `waf_middleware.py`

**Protection Against**:
- SQL Injection
- Cross-Site Scripting (XSS)
- Path Traversal
- Command Injection
- XML External Entity (XXE)
- LDAP Injection
- Header Injection
- Oversized Requests

**Key Features**:
- ModSecurity-style rule engine
- Configurable rule sets
- IP whitelisting/blacklisting
- Automatic violation logging
- Configurable blocking behavior

**Configuration**:
```python
app.add_middleware(
    WAFMiddleware,
    enabled=True,
    whitelist_ips=["trusted_ip"],
    blacklist_ips=["malicious_ip"],
    custom_rules=[CustomRule()],
    log_violations=True,
    block_on_violation=True
)
```

### 2. Request Signing (API Authentication)

**File**: `request_signing.py`

**Features**:
- HMAC-SHA256 request signing
- API key generation and management
- Timestamp validation (5-minute window)
- Nonce-based replay protection
- Per-key rate limiting
- IP restrictions per key
- Permission-based access control

**Client-Side Usage**:
```python
from hololoom.server.request_signing import RequestSigningClient

client = RequestSigningClient(key_id="your_key_id", secret="your_secret")
headers = client.sign_request(
    method="POST",
    url="https://api.hololoom.io/ar/query",
    body=json.dumps({"query": "test"}).encode()
)
# Include headers in your request
```

**Server-Side Validation**:
```python
@app.post("/protected-endpoint")
async def protected_endpoint(
    request: Request,
    api_key: APIKey = Depends(validate_api_request)
):
    # api_key contains validated key info
    return {"client": api_key.client_name}
```

### 3. Security Monitoring

**File**: `security_monitor.py`

**Capabilities**:
- Real-time event tracking
- Threshold-based alerting
- Email notifications (via SMTP)
- Slack notifications (via webhook)
- Prometheus metrics export
- WebSocket dashboard updates
- Attacker profiling
- Automatic IP blocking

**Event Types Tracked**:
- Authentication failures
- Rate limit violations
- WAF violations
- Invalid signatures
- Suspicious patterns
- Brute force attacks
- DoS attacks
- Data exfiltration attempts

**Alert Configuration**:
```python
monitor = SecurityMonitor(
    alert_threshold_auth_failures=5,    # Alert after 5 auth failures
    alert_threshold_rate_limit=10,      # Alert after 10 rate limits
    alert_threshold_waf_violations=3,   # Alert after 3 WAF violations
    time_window_seconds=60,             # Within 60-second window
    enable_email_alerts=True,
    enable_slack_alerts=True,
    smtp_config={...},
    slack_webhook_url="https://hooks.slack.com/..."
)
```

### 4. Rate Limiting

**Implementation**: `EnhancedRateLimiter` class in `ar_api_secured.py`

**Features**:
- Sliding window algorithm
- Per-IP tracking
- Per-endpoint limits
- Auto-blocking after excessive violations
- Configurable windows and thresholds

**Configuration**:
```python
rate_limiter = EnhancedRateLimiter(
    requests=100,  # 100 requests
    window=60      # per 60 seconds
)
```

## Security Policies

### API Key Management

1. **Generation**:
   - Secure random generation using `secrets` module
   - Configurable expiration (default: 30 days)
   - Client-specific rate limits
   - IP restrictions support

2. **Validation**:
   - Constant-time signature comparison
   - Timestamp validation (5-minute window)
   - Nonce tracking for replay protection
   - Automatic expiration checking

3. **Revocation**:
   - Immediate effect
   - Logged for audit trail
   - Notification to affected client

### Rate Limiting Strategy

1. **Global Limits**:
   - 100 requests per minute per IP (configurable)
   - 1000 requests per minute per endpoint

2. **Endpoint-Specific**:
   - Vision endpoints: 10 requests per minute
   - Query endpoints: 100 requests per minute
   - Admin endpoints: 10 requests per minute

3. **Auto-Blocking**:
   - Block IP after 2x rate limit violations
   - Block duration: 15 minutes (configurable)
   - Manual unblock available via admin API

### WAF Rules

1. **SQL Injection**:
   - Pattern matching for SQL keywords
   - Detection of comment indicators
   - Hex encoding detection
   - Stacked query detection

2. **XSS**:
   - Script tag detection
   - Event handler detection
   - JavaScript protocol detection
   - Data URL detection

3. **Command Injection**:
   - Shell command detection
   - Command chaining detection
   - Backtick detection
   - Path to shell detection

## Monitoring & Alerting

### Prometheus Metrics

Available at `/security/metrics` endpoint:

```
security_events_total{type="auth_failure"} 42
security_events_total{type="waf_violation"} 15
active_alerts_count 3
blocked_ips_count 2
auth_failures_per_minute 5
waf_violations_per_minute 2
rate_limits_per_minute 8
high_risk_ips 1
```

### Email Alerts

Sent for HIGH and CRITICAL severity events:

```html
Subject: [CRITICAL] Multiple WAF Violations

Alert ID: ALERT-00042
Severity: CRITICAL
Time: 2025-11-26T10:30:00Z
Description: IP 192.168.1.100 triggered 5 WAF violations

Event Details:
- 2025-11-26T10:29:45Z - SQL Injection attempt
- 2025-11-26T10:29:50Z - XSS attempt
- ...

ACTION REQUIRED
```

### Slack Notifications

Real-time notifications to Slack channel:

```json
{
  "attachments": [{
    "color": "#ff0000",
    "title": "⚠️ Security Alert",
    "text": "Multiple authentication failures detected",
    "fields": [
      {"title": "Alert ID", "value": "ALERT-00042"},
      {"title": "Severity", "value": "HIGH"},
      {"title": "IP Address", "value": "192.168.1.100"}
    ]
  }]
}
```

### Security Dashboard

Real-time dashboard available at `/security/dashboard`:

```json
{
  "summary": {
    "total_events_hour": 150,
    "total_events_day": 2500,
    "active_alerts": 3,
    "blocked_ips": 5,
    "high_risk_ips": 2
  },
  "top_attackers": [...],
  "event_distribution": {...},
  "recent_alerts": [...],
  "timeline": [...]
}
```

## Deployment

### Environment Variables

```bash
# Security Features
ENABLE_WAF=true
ENABLE_SIGNING=true
ENABLE_MONITORING=true
ENABLE_RATE_LIMITING=true
WAF_BLOCK=true

# API Key Management
ADMIN_KEY=your_admin_secret
GENERATE_TEST_KEYS=false

# Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=security@hololoom.io
SMTP_PASSWORD=your_password
SMTP_FROM=security@hololoom.io
SMTP_TO=admin@hololoom.io

# Slack Alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY HoloLoom /app/HoloLoom

# Set environment variables
ENV ENABLE_WAF=true
ENV ENABLE_SIGNING=true
ENV ENABLE_MONITORING=true

# Run server
CMD ["uvicorn", "hololoom.server.ar_api_secured:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hololoom-api-secured
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: hololoom:secured
        env:
        - name: ENABLE_WAF
          value: "true"
        - name: ENABLE_SIGNING
          valueFrom:
            secretKeyRef:
              name: security-config
              key: enable_signing
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
```

## Testing

### WAF Testing

Test WAF rules with OWASP ZAP or manual tests:

```bash
# Test SQL injection detection
curl -X POST https://api.hololoom.io/ar/query \
  -d '{"query": "test OR 1=1"}' \
  -H "Content-Type: application/json"
# Expected: 400 Bad Request

# Test XSS detection
curl -X POST https://api.hololoom.io/ar/query \
  -d '{"query": "<script>alert(1)</script>"}' \
  -H "Content-Type: application/json"
# Expected: 400 Bad Request
```

### Request Signing Testing

```python
# Generate test key
response = requests.post(
    "https://api.hololoom.io/admin/api-keys/generate",
    params={
        "client_name": "Test Client",
        "admin_key": "admin_secret"
    }
)
key_data = response.json()

# Use key to sign requests
client = RequestSigningClient(
    key_id=key_data["key_id"],
    secret=key_data["secret"]
)

headers = client.sign_request(
    method="POST",
    url="https://api.hololoom.io/ar/query",
    body=json.dumps({"query": "test"}).encode()
)

response = requests.post(
    "https://api.hololoom.io/ar/query",
    headers=headers,
    json={"query": "test"}
)
```

### Rate Limiting Testing

```python
import asyncio
import aiohttp

async def test_rate_limiting():
    async with aiohttp.ClientSession() as session:
        # Send 150 requests (exceeds 100/minute limit)
        for i in range(150):
            async with session.post(
                "https://api.hololoom.io/ar/query",
                json={"query": f"test {i}"}
            ) as response:
                if response.status == 429:
                    print(f"Rate limited at request {i}")
                    break
```

### Security Monitoring Testing

```python
# Trigger security events
for i in range(10):
    # Trigger auth failures
    requests.post(
        "https://api.hololoom.io/login",
        json={"username": "admin", "password": "wrong"}
    )

# Check alerts were generated
response = requests.get(
    "https://api.hololoom.io/security/dashboard",
    headers=admin_headers
)
dashboard = response.json()
assert dashboard["summary"]["active_alerts"] > 0
```

## Performance Impact

| Component | Overhead | Notes |
|-----------|----------|-------|
| WAF | ~2-5ms | Pattern matching overhead |
| Request Signing | ~1-2ms | HMAC calculation |
| Rate Limiting | <1ms | In-memory lookups |
| Security Monitoring | <1ms | Async logging |
| **Total** | **~5-10ms** | Acceptable for most applications |

## Best Practices

1. **Defense in Depth**: Use all security layers, don't rely on just one
2. **Regular Updates**: Keep WAF rules updated with latest attack patterns
3. **Key Rotation**: Rotate API keys every 30-90 days
4. **Monitoring**: Set up alerts for critical security events
5. **Testing**: Regularly test security controls with penetration testing
6. **Logging**: Keep audit logs for at least 90 days
7. **Incident Response**: Have a plan for security incidents

## Troubleshooting

### Common Issues

1. **"Invalid signature" errors**:
   - Check timestamp synchronization between client and server
   - Verify API key and secret are correct
   - Ensure request body matches exactly

2. **Rate limiting too aggressive**:
   - Adjust `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`
   - Consider per-endpoint limits instead of global

3. **False positive WAF blocks**:
   - Review WAF logs for patterns
   - Add exceptions for legitimate patterns
   - Consider reducing sensitivity

4. **Missing alerts**:
   - Verify SMTP/Slack configuration
   - Check alert thresholds
   - Review security monitor logs

## Support

For security issues or questions:
- Email: security@hololoom.io
- Slack: #security channel
- Documentation: This file

## Security Disclosure

If you discover a security vulnerability:
1. DO NOT create a public GitHub issue
2. Email security@hololoom.io with details
3. Allow 48 hours for initial response
4. Work with us on responsible disclosure