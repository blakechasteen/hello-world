# HoloLoom Web Application Firewall (WAF) Setup Guide

**Version**: 1.0.0
**Date**: 2025-11-15
**Status**: Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Deployment](#deployment)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The HoloLoom WAF is a comprehensive security solution combining:
- **ModSecurity 3.x** - Core WAF engine
- **OWASP Core Rule Set (CRS)** - OWASP Top 10 protection
- **Custom HoloLoom Rules** - Application-specific threat detection
- **Nginx reverse proxy** - HTTP/2, SSL/TLS termination
- **Geo-blocking** - Geographic IP filtering
- **Rate limiting** - DDoS and brute-force protection
- **Request/response filtering** - Sensitive data protection

### Threat Coverage

| Threat | Detection | Blocking |
|--------|-----------|----------|
| SQL Injection | ✓ | ✓ |
| Cross-Site Scripting (XSS) | ✓ | ✓ |
| Path Traversal | ✓ | ✓ |
| Command Injection | ✓ | ✓ |
| File Upload Attacks | ✓ | ✓ |
| CSRF Attacks | ✓ | ✓ |
| Brute Force | ✓ | ✓ |
| Denial of Service (DoS) | ✓ | ✓ |
| Malware Signatures | ✓ | ✓ |
| Prompt Injection | ✓ | ✓ |

---

## Architecture

```
Internet
    ↓
[Nginx + ModSecurity WAF]
    ├─ Phase 1: Request Headers
    ├─ Phase 2: Request Body
    ├─ Phase 3: Response Headers
    ├─ Phase 4: Response Body
    └─ Phase 5: Logging
    ↓
[HoloLoom Backend]
    └─ /api/*
    └─ /auth/*
    └─ /query/*
```

### Configuration Files

```
infra/
├── nginx/
│   ├── nginx.conf           # Main Nginx configuration
│   └── proxy_params.conf    # Proxy headers and settings
├── waf/
│   ├── modsecurity.conf     # ModSecurity core config
│   ├── owasp-crs.conf       # OWASP Core Rule Set rules
│   ├── custom-rules.conf    # HoloLoom-specific rules
│   ├── whitelist.conf       # Whitelist exceptions
│   ├── geoip.conf           # Geo-blocking config
│   └── whitelist-ips.conf   # Trusted IP list
└── docker/
    └── docker-compose.yml   # Docker deployment
```

---

## Installation

### Prerequisites

- Docker & Docker Compose
- Linux kernel 4.4+ (for GeoIP module)
- 2GB RAM minimum
- 100MB disk space

### Option 1: Docker Deployment (Recommended)

#### Step 1: Build Docker Image

Create `infra/docker/Dockerfile.nginx`:

```dockerfile
FROM nginx:alpine-slim

# Install ModSecurity module
RUN apk add --no-cache \
    gcc \
    g++ \
    make \
    linux-headers \
    pcre-dev \
    libxml2-dev \
    lua-dev \
    yajl-dev

# Install ModSecurity
RUN cd /tmp && \
    git clone https://github.com/SpiderLabs/ModSecurity.git && \
    cd ModSecurity && \
    ./build.sh && \
    ./configure --with-apxs=/usr/bin/apxs2 && \
    make && \
    make install

# Copy configurations
COPY infra/nginx/*.conf /etc/nginx/
COPY infra/waf/*.conf /etc/nginx/modsec/
COPY infra/waf/whitelist-ips.conf /etc/nginx/

# Create necessary directories
RUN mkdir -p /var/log/modsec /var/lib/modsec/uploads /var/tmp/modsec && \
    chmod 755 /var/log/modsec /var/lib/modsec/uploads /var/tmp/modsec

EXPOSE 80 443 8080

CMD ["nginx", "-g", "daemon off;"]
```

#### Step 2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  nginx-waf:
    build:
      context: .
      dockerfile: infra/docker/Dockerfile.nginx
    container_name: hololoom-waf
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - ./infra/nginx:/etc/nginx:ro
      - ./infra/waf:/etc/nginx/modsec:ro
      - ./certs:/etc/nginx/ssl:ro
      - waf-logs:/var/log/modsec
      - waf-uploads:/var/lib/modsec/uploads
    environment:
      - ENVIRONMENT=production
      - MODSEC_MODE=On
    networks:
      - hololoom
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "https://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  hololoom-backend:
    image: hololoom:latest
    container_name: hololoom-api
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    networks:
      - hololoom
    restart: unless-stopped

volumes:
  waf-logs:
  waf-uploads:

networks:
  hololoom:
    driver: bridge
```

#### Step 3: Generate SSL Certificates

```bash
# Self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -nodes -out infra/certs/hololoom.crt \
    -keyout infra/certs/hololoom.key -days 365 -subj "/CN=hololoom.local"

# Production: Use Let's Encrypt with Certbot
certbot certonly --standalone -d hololoom.example.com
```

#### Step 4: Deploy

```bash
cd infra/docker
docker-compose up -d
docker-compose logs -f nginx-waf
```

### Option 2: Manual Installation

#### Step 1: Install Nginx with ModSecurity

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nginx

# Install ModSecurity module
sudo apt-get install -y libnginx-mod-http-modsecurity

# Install GeoIP module
sudo apt-get install -y libnginx-mod-geoip2

# Create directories
sudo mkdir -p /etc/nginx/modsec /var/log/modsec /var/lib/modsec/uploads
sudo chown -R www-data:www-data /var/lib/modsec /var/log/modsec
```

#### Step 2: Copy Configuration Files

```bash
sudo cp infra/nginx/nginx.conf /etc/nginx/nginx.conf
sudo cp infra/nginx/proxy_params.conf /etc/nginx/
sudo cp infra/waf/modsecurity.conf /etc/nginx/modsec/
sudo cp infra/waf/owasp-crs.conf /etc/nginx/modsec/
sudo cp infra/waf/custom-rules.conf /etc/nginx/modsec/
sudo cp infra/waf/whitelist.conf /etc/nginx/modsec/
sudo cp infra/waf/whitelist-ips.conf /etc/nginx/
```

#### Step 3: Download GeoIP Database

```bash
# Free: MaxMind GeoLite2
sudo apt-get install geoip2-database

# Or download manually
curl -L -o /usr/share/GeoIP/GeoLite2-Country.mmdb \
    https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key=YOUR_KEY
```

#### Step 4: Validate Configuration

```bash
sudo nginx -t
# Should output: "configuration file test is successful"
```

#### Step 5: Start Nginx

```bash
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl status nginx
```

---

## Configuration

### WAF Modes

#### Detection Only (Development)

ModSecurity logs violations without blocking:

```bash
# In modsecurity.conf
SecAuditEngine On
SecDefaultAction "phase:1,log,pass"
SecDefaultAction "phase:2,log,pass"
```

**Use Case**: Development, testing, tuning

#### Block Mode (Production)

ModSecurity blocks malicious requests:

```bash
# In modsecurity.conf
SecAuditEngine On
SecDefaultAction "phase:1,log,auditlog,deny"
SecDefaultAction "phase:2,log,auditlog,deny"
```

**Use Case**: Production deployment

### Paranoia Levels

Adjust sensitivity with paranoia level (1-4):

```bash
# In owasp-crs.conf
SecRule &PARANOIA_LEVEL "@eq 0" \
    "id:900000,phase:1,pass,nolog,setvar:tx.paranoia_level=1"
```

| Level | False Positives | Coverage | Use Case |
|-------|-----------------|----------|----------|
| 1 | Minimal | 90% | Production |
| 2 | Low | 95% | Higher security |
| 3 | Medium | 98% | Sensitive apps |
| 4 | High | 99% | Maximum protection |

### Adding Custom Whitelists

1. **Whitelist by IP**:

```bash
# In whitelist-ips.conf
203.0.113.50 1;  # Trust this IP
```

2. **Whitelist by Endpoint**:

```bash
# In whitelist.conf
SecRule REQUEST_URI "@eq /api/internal" \
    "id:80001,phase:1,pass,nolog,ctl:ruleEngine=Off"
```

3. **Whitelist by Parameter**:

```bash
# In whitelist.conf
SecRule REQUEST_URI "@eq /api/document" \
    "id:80010,phase:2,pass,nolog,ctl:ruleRemoveByID=941100"
```

### Rate Limiting Configuration

Adjust limits in `nginx.conf`:

```nginx
# API endpoints: 100 requests/minute
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;

# Login endpoint: 5 requests/minute
limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

# Upload endpoint: 1 request/second
limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;
```

### Geo-Blocking Configuration

Edit `geoip.conf`:

```bash
# Block high-risk countries
KP 1;  # North Korea
IR 1;  # Iran
SY 1;  # Syria

# Restrict countries (require authentication)
CN 2;  # China
RU 2;  # Russia
```

---

## Testing

### Run Full Test Suite

```bash
# Test all attacks
python scripts/test_waf.py --url https://localhost --skip-ssl-verify

# Test specific category
python scripts/test_waf.py --category SQL_INJECTION --skip-ssl-verify

# Test critical severity only
python scripts/test_waf.py --severity CRITICAL --skip-ssl-verify

# Test false positives (legitimate requests)
python scripts/test_waf.py --false-positives --skip-ssl-verify

# Export results
python scripts/test_waf.py --export results.json --skip-ssl-verify
```

### Expected Test Results

```
============================================================
HoloLoom WAF Test Suite
Target: https://localhost
Timestamp: 2025-11-15T12:34:56.789012
Total Payloads: 42
============================================================

[1/42] SQL Injection - Basic UNION... ✓ (403 - 15.2ms)
[2/42] SQL Injection - Comment Bypass... ✓ (403 - 12.8ms)
[3/42] XSS - Basic Script Tag... ✓ (403 - 10.5ms)
...

============================================================
TEST SUMMARY
============================================================
Total Tests: 42
Passed: 42 (100.0%)
Failed: 0 (0.0%)

By Category:
  SQL_INJECTION         4/4 (100.0%)
  XSS                   5/5 (100.0%)
  PATH_TRAVERSAL        4/4 (100.0%)
  COMMAND_INJECTION     4/4 (100.0%)
  FILE_UPLOAD           3/3 (100.0%)
  CSRF                  2/2 (100.0%)
  DOS                   3/3 (100.0%)
  MALWARE               2/2 (100.0%)
  PROMPT_INJECTION      3/3 (100.0%)
  BRUTE_FORCE           2/2 (100.0%)
```

### Manual Testing

#### Test SQL Injection Block

```bash
curl -X GET "https://localhost/api/search?query=1' UNION SELECT * FROM users--" \
  -k -v

# Expected: 403 Forbidden
```

#### Test XSS Block

```bash
curl -X GET "https://localhost/api/search?query=<script>alert('XSS')</script>" \
  -k -v

# Expected: 403 Forbidden
```

#### Test Legitimate Request

```bash
curl -X GET "https://localhost/api/search?query=python" \
  -k -v

# Expected: 200 OK
```

#### Test Rate Limiting

```bash
for i in {1..150}; do
  curl -s "https://localhost/api" -k -o /dev/null -w "%{http_code}\n"
done

# Expected: 200 for first 100, then 429 (Too Many Requests)
```

---

## Deployment

### Pre-Production Checklist

- [ ] SSL/TLS certificates installed
- [ ] GeoIP database updated
- [ ] Custom rules tested and tuned
- [ ] Whitelists configured for internal services
- [ ] Rate limits adjusted for expected traffic
- [ ] Logging configured for SIEM integration
- [ ] Monitoring and alerting configured
- [ ] Backup of configuration files
- [ ] Incident response plan documented
- [ ] Team training completed

### Blue-Green Deployment

```bash
# Blue: Current WAF
docker-compose -f docker-compose.blue.yml up -d

# Test green deployment
docker-compose -f docker-compose.green.yml up -d

# Switch traffic to green
nginx -s reload

# Monitor for 24 hours
watch -n 5 'tail -20 /var/log/modsec/audit.log'

# Remove blue if stable
docker-compose -f docker-compose.blue.yml down
```

### Canary Deployment

```bash
# Phase 1: 10% traffic (detective mode)
# Phase 2: 25% traffic (detective mode)
# Phase 3: 50% traffic (block mode)
# Phase 4: 100% traffic (block mode)
```

### Rollback Procedure

```bash
# If issues detected:
docker-compose down
git checkout HEAD~1  # Revert configuration
docker-compose up -d

# Verify
curl https://localhost/health
```

---

## Monitoring

### Log Files

```bash
# WAF audit log
/var/log/nginx/waf.log

# ModSecurity audit log
/var/log/modsec/audit.log

# Nginx error log
/var/log/nginx/error.log

# Security-specific log
/var/log/nginx/security.log
```

### Real-Time Monitoring

```bash
# Watch WAF blocks
tail -f /var/log/modsec/audit.log | grep "SecAction"

# Watch rate limiting
tail -f /var/log/nginx/access.log | grep "429"

# Watch security events
tail -f /var/log/nginx/security.log | grep "CRITICAL"
```

### Key Metrics to Track

1. **Block Rate**: % of requests blocked
2. **False Positive Rate**: Legitimate requests blocked
3. **Latency Impact**: Average request latency added by WAF
4. **Rule Violations**: Top triggered rules (find tuning opportunities)
5. **Geo-Blocks**: IPs blocked by geo-blocking
6. **Rate Limit Hits**: Brute force detection
7. **Upload Violations**: Malicious file uploads blocked

### Prometheus Metrics

Export metrics for Prometheus monitoring:

```bash
# Enable in nginx.conf
location /metrics {
    stub_status on;
    access_log off;
    allow 127.0.0.1;
    deny all;
}

# Scrape config
scrape_configs:
  - job_name: 'nginx-waf'
    static_configs:
      - targets: ['localhost:8080']
```

### Alert Rules

```yaml
# Prometheus alert rules
groups:
  - name: WAF
    rules:
      - alert: HighBlockRate
        expr: (rate(waf_blocks[5m]) / rate(waf_requests[5m])) > 0.1
        for: 5m
        annotations:
          summary: "WAF blocking >10% of requests"

      - alert: SuspiciousActivity
        expr: rate(waf_blocks_critical[1m]) > 5
        for: 1m
        annotations:
          summary: "Multiple critical attacks detected"

      - alert: RateLimitExceeded
        expr: rate(http_429_responses[5m]) > 10
        for: 5m
        annotations:
          summary: "High rate limit violations"
```

### SIEM Integration

#### Send to Splunk

```bash
# In docker-compose.yml
splunk-agent:
  image: splunk/universalforwarder:latest
  volumes:
    - /var/log/modsec:/var/log/modsec
  environment:
    - SPLUNK_START_ARGS=--accept-license
    - SPLUNK_PASSWORD=your_password
```

#### Send to ELK Stack

```yaml
# Filebeat configuration
- type: log
  enabled: true
  paths:
    - /var/log/modsec/audit.log
  fields:
    service: waf
    type: modsecurity
  processors:
    - add_kubernetes_metadata:

output.elasticsearch:
  hosts: ["elk:9200"]
```

---

## Troubleshooting

### Issue: High False Positive Rate

**Solution**: Lower paranoia level

```bash
# In owasp-crs.conf
setvar:tx.paranoia_level=1  # From 2 or 3
```

### Issue: Legitimate File Upload Blocked

**Solution**: Add whitelist exception

```bash
# In whitelist.conf
SecRule REQUEST_URI "@eq /api/upload" \
    "id:80020,phase:2,pass,nolog,ctl:ruleRemoveByID=933100"
```

### Issue: API Integration Failing

**Solution**: Check request headers

```bash
# Log request details
curl -X POST https://localhost/api \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}' \
  -k -v
```

### Issue: Performance Degradation

**Solutions**:
1. Increase ModSecurity buffer sizes
2. Reduce paranoia level
3. Whitelist known good sources
4. Enable caching

### Debug Mode

Enable detailed logging:

```bash
# In modsecurity.conf
SecDebugLog /var/log/modsec/debug.log
SecDebugLogLevel 5
```

```bash
# Monitor debug output
tail -f /var/log/modsec/debug.log | grep "rule_id"
```

### Test Configuration

```bash
# Validate nginx.conf
nginx -t

# Check ModSecurity syntax
nginx -t -c /etc/nginx/nginx.conf

# Verify modules loaded
nginx -V 2>&1 | grep modsecurity
```

---

## Security Best Practices

1. **Keep Rules Updated**
   ```bash
   # Monthly update of OWASP CRS
   crontab: 0 2 1 * * cd /etc/nginx/modsec && git pull origin main
   ```

2. **Monitor Whitelist Bloat**
   - Review whitelist quarterly
   - Remove unnecessary exceptions
   - Document business reason for each exception

3. **Incident Response**
   - Archive logs after attacks
   - Root cause analysis
   - Tune rules based on incidents
   - Update threat intelligence

4. **Access Control**
   - Restrict WAF config access
   - Log all configuration changes
   - Use version control (Git)
   - Require approvals for production changes

5. **Regular Testing**
   - Monthly WAF penetration testing
   - Quarterly security audits
   - Annual comprehensive review

---

## Performance Characteristics

| Operation | Latency | Overhead |
|-----------|---------|----------|
| Request inspection | <1ms | <5% |
| Rule matching | <2ms | <10% |
| Logging | <1ms | <5% |
| **Total per request** | **<5ms** | **<15%** |

**Note**: Actual overhead depends on:
- Request size (larger = more processing)
- Rule complexity
- Number of rules
- Logging verbosity
- Server resources

---

## Support and Maintenance

### Regular Maintenance Tasks

- **Daily**: Monitor logs for anomalies
- **Weekly**: Review top blocked rules
- **Monthly**: Update GeoIP database
- **Quarterly**: Review and tune whitelist
- **Annually**: Security audit, penetration testing

### Getting Help

- **ModSecurity**: https://modsecurity.org/
- **OWASP CRS**: https://coreruleset.org/
- **Nginx**: https://nginx.org/en/support.html
- **HoloLoom**: Check CLAUDE.md for project contact

---

## References

- [ModSecurity Documentation](https://modsecurity.org/doc/)
- [OWASP Core Rule Set](https://coreruleset.org/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [OWASP Top 10](https://owasp.org/Top10/)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)

---

**Last Updated**: 2025-11-15
**Maintained By**: HoloLoom Security Team
