# HoloLoom WAF Quick Start Guide

**Date**: 2025-11-15
**Status**: Ready to Deploy

## 5-Minute Setup

### Prerequisites
- Docker & Docker Compose installed
- 4GB RAM, 2 CPU cores available
- Port 80, 443 available (or modify docker-compose.yml)

### Step 1: Generate SSL Certificates

```bash
# Create certs directory
mkdir -p ../../certs

# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -nodes \
  -out ../../certs/hololoom.crt \
  -keyout ../../certs/hololoom.key \
  -days 365 \
  -subj "/CN=hololoom.local"

# For production: use Let's Encrypt
# certbot certonly --standalone -d your-domain.com
# Then copy certs to ../../certs/
```

### Step 2: Start Services

```bash
# Navigate to docker directory
cd infra/docker

# Start all services (WAF + Backend + Monitoring)
docker-compose up -d

# Watch startup
docker-compose logs -f nginx-waf

# Wait for "nginx: master process" message (30-40 seconds)
```

### Step 3: Verify Deployment

```bash
# Check service health
docker-compose ps

# Expected output:
# CONTAINER          STATUS
# nginx-waf          Up (healthy)
# hololoom-backend   Up (healthy)
# prometheus         Up
# grafana            Up
# elasticsearch      Up
# kibana             Up
# filebeat           Up

# Test endpoint
curl -k https://localhost/health
# Expected: "OK" or {"status": "healthy"}
```

### Step 4: Run WAF Tests

```bash
# From repository root
python scripts/test_waf.py --skip-ssl-verify

# Expected output:
# Total Tests: 42
# Passed: 42 (100.0%)
# Failed: 0 (0.0%)
```

### Step 5: Access Services

Open in browser:
- **HoloLoom**: https://localhost (may show security warning, click "proceed")
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Kibana**: http://localhost:5601

## Common Tasks

### View WAF Logs

```bash
# Real-time WAF blocks
docker-compose exec nginx-waf tail -f /var/log/modsec/audit.log

# Or from host
tail -f waf-modsec-logs/_default@*/audit.log

# Search for specific attack
docker-compose logs nginx-waf | grep "SQL_INJECTION"
```

### Test Specific Attack

```bash
# Test SQL injection only
python scripts/test_waf.py --category SQL_INJECTION --skip-ssl-verify

# Test critical severity
python scripts/test_waf.py --severity CRITICAL --skip-ssl-verify

# Test and export results
python scripts/test_waf.py --export results.json --skip-ssl-verify
```

### Modify Rate Limits

Edit `infra/nginx/nginx.conf`:

```nginx
# Change login rate limit (default: 5 req/min)
limit_req_zone $binary_remote_addr zone=login:10m rate=10r/m;
                                                    ^^^^ change here

# Then reload
docker-compose exec nginx-waf nginx -s reload
```

### Add Whitelist Exception

Edit `infra/waf/whitelist.conf`:

```bash
# Add your IP as trusted
SecRule REMOTE_ADDR "@ipMatch YOUR_IP" \
    "id:70100,phase:1,pass,nolog,setvar:trusted_ip=1"

# Then reload
docker-compose restart nginx-waf
```

### Enable Development Mode (No Blocking)

Edit `infra/waf/modsecurity.conf`:

```bash
# Change from "On" to "DetectionOnly"
SecEngine DetectionOnly
# or
SecAuditEngine DetectionOnly

# Then reload
docker-compose restart nginx-waf
```

### Monitor Performance

```bash
# CPU and memory usage
docker stats nginx-waf

# Response times
docker-compose exec nginx-waf \
  tail -f /var/log/nginx/access.log | grep response_time

# Grafana dashboards
# http://localhost:3000 (select nginx or WAF dashboard)
```

## Troubleshooting

### Nginx Won't Start
```bash
# Check logs
docker-compose logs nginx-waf

# Validate nginx config
docker-compose exec nginx-waf nginx -t

# Restart
docker-compose restart nginx-waf
```

### High CPU Usage
```bash
# Reduce paranoia level
# In owasp-crs.conf, set: paranoia_level=1 (from 2)

# Or disable specific rules
# In whitelist.conf, add:
# SecRule ... "ctl:ruleRemoveByID=942100"

# Then restart
docker-compose restart nginx-waf
```

### False Positives (Legitimate Requests Blocked)
```bash
# Enable detection-only mode
# Edit modsecurity.conf: SecEngine DetectionOnly

# Or add whitelist:
# In whitelist.conf:
# SecRule REQUEST_URI "@eq /api/my-endpoint" \
#     "id:80020,phase:2,pass,nolog,ctl:ruleRemoveByID=942100"

docker-compose restart nginx-waf
```

### Certificate Errors
```bash
# Chrome/Firefox shows security warning:
# This is normal for self-signed certs
# Click "Advanced" → "Proceed to localhost"

# Or ignore in curl:
curl -k https://localhost

# For production: use Let's Encrypt
certbot certonly --standalone -d your-domain.com
# Then copy to certs/
```

## Performance Baseline

After deployment, expect:

| Metric | Value |
|--------|-------|
| Healthy services | 7/7 |
| Request latency | <20ms |
| WAF overhead | <5ms |
| Block rate | 0% (legitimate traffic) |
| Memory usage | ~500MB (all services) |
| CPU usage | <10% idle |

## Next Steps

1. **Review logs** (10 min)
   - Check real-time WAF activity
   - Verify no false positives

2. **Tune configuration** (30 min)
   - Adjust rate limits for your traffic
   - Update whitelist with internal IPs
   - Customize GeoIP blocking if needed

3. **Run full test suite** (5 min)
   - All 42 attack payloads
   - False positive detection
   - Performance metrics

4. **Monitor for 24 hours** (continuous)
   - Watch real requests
   - Check false positive rate
   - Monitor latency impact

5. **Deploy to production** (after verification)
   - Enable blocking mode
   - Update monitoring/alerting
   - Document runbooks

## Configuration Files

All configuration is modular and hot-reloadable:

```
infra/
├── nginx/
│   ├── nginx.conf              ← Main config (reload after edit)
│   └── proxy_params.conf       ← Shared proxy settings
├── waf/
│   ├── modsecurity.conf        ← Core engine (restart after edit)
│   ├── owasp-crs.conf          ← OWASP rules (restart after edit)
│   ├── custom-rules.conf       ← HoloLoom rules (restart after edit)
│   ├── whitelist.conf          ← Exceptions (restart after edit)
│   ├── geoip.conf             ← Geo-blocking (reload after edit)
│   └── whitelist-ips.conf     ← IP whitelist (reload after edit)
└── docker/
    └── docker-compose.yml      ← Service orchestration
```

### Reloading Configuration

```bash
# Reload nginx (no-downtime config reload)
docker-compose exec nginx-waf nginx -s reload

# Restart nginx (brief downtime)
docker-compose restart nginx-waf

# Restart all services
docker-compose restart
```

## Security Checklist

- [ ] SSL certificates installed (self-signed or Let's Encrypt)
- [ ] Rate limits appropriate for your traffic
- [ ] Whitelist configured with internal IPs
- [ ] GeoIP blocking configured (if needed)
- [ ] Monitoring/alerting set up
- [ ] Test suite passes (42/42)
- [ ] No false positives observed
- [ ] Latency impact acceptable (<10ms)
- [ ] Logs being collected and archived
- [ ] Incident response procedures documented

## Support

For detailed documentation:
- Setup & Operation: `/home/user/hello-world/docs/WAF_SETUP.md`
- Implementation Details: `/home/user/hello-world/docs/WAF_IMPLEMENTATION_SUMMARY.md`
- Testing Guide: `/home/user/hello-world/docs/WAF_SETUP.md#Testing`
- Troubleshooting: `/home/user/hello-world/docs/WAF_SETUP.md#Troubleshooting`

For external help:
- ModSecurity: https://modsecurity.org/
- OWASP CRS: https://coreruleset.org/
- Nginx: https://nginx.org/

## Quick Reference

```bash
# Show all services status
docker-compose ps

# View logs (all services)
docker-compose logs

# View WAF logs only
docker-compose logs nginx-waf -f

# Stop all services
docker-compose down

# Stop and remove volumes (cleanup)
docker-compose down -v

# Rebuild images after config changes
docker-compose up -d --build

# Test single attack category
python scripts/test_waf.py --category SQL_INJECTION --skip-ssl-verify

# Export test results
python scripts/test_waf.py --export results.json --skip-ssl-verify
```

---

**Status**: ✅ Ready to Deploy
**Last Updated**: 2025-11-15
