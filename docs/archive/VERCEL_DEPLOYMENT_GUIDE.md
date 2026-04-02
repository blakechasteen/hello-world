# HoloLoom Lite Vercel Deployment - Complete Guide

**Last Updated:** December 2024
**Status:** Production Ready
**Version:** 1.0.0

## Table of Contents

1. [Quick Deploy (5 minutes)](#quick-deploy)
2. [Manual Deployment](#manual-deployment)
3. [Configuration](#configuration)
4. [Testing](#testing)
5. [Production Checklist](#production-checklist)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Setup](#advanced-setup)

---

## Quick Deploy

### Option A: One-Click Deploy Button

1. Click button in README
2. Authorize GitHub
3. Vercel creates project automatically
4. Deployment completes in ~2 minutes
5. Get live URL

**✅ Recommended for first deployment**

### Option B: CLI Deploy (Fastest)

```bash
# Install Vercel CLI
npm install -g vercel

# Clone repository
git clone https://github.com/youruser/mythrl
cd mythrl

# Deploy to production
vercel deploy --prod

# Output:
# ✓ Deployed to https://hololoom-lite.vercel.app [in 45s]
```

### Option C: Git Push to Deploy

1. Push to GitHub
2. Vercel auto-detects push
3. Auto-builds and deploys
4. Deployment complete

**Note:** Requires Vercel GitHub app installed

---

## Manual Deployment

### Step 1: Setup Vercel Account

```bash
# Login to Vercel
vercel login

# Or create account at https://vercel.com
```

### Step 2: Link Repository

```bash
# In project directory
vercel link

# Follow prompts to select/create project
```

### Step 3: Configure Environment

```bash
# Set environment variables
vercel env add HOLOLOOM_MEMORY_BACKEND
vercel env add HOLOLOOM_MAX_MEMORIES
vercel env add RATE_LIMIT_MAX_REQUESTS

# Or edit in Vercel Dashboard:
# Project → Settings → Environment Variables
```

### Step 4: Deploy

```bash
# Development (preview)
vercel deploy

# Production
vercel deploy --prod

# View logs
vercel logs --tail
```

---

## Configuration

### Environment Variables Reference

Set in Vercel Dashboard or `.env.local` (local development only):

#### Storage Configuration

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `HOLOLOOM_MEMORY_BACKEND` | `INMEMORY` | Storage backend type | `INMEMORY`, `HYBRID` |
| `HOLOLOOM_MAX_MEMORIES` | `1000` | Max in-memory entries | `500`, `2000` |
| `HOLOLOOM_CACHE_TTL` | `3600` | Memory entry lifetime (seconds) | `1800`, `7200` |

#### Rate Limiting Configuration

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `RATE_LIMIT_MAX_REQUESTS` | `100` | Max requests per window | `50`, `500` |
| `RATE_LIMIT_WINDOW_MS` | `60000` | Rate limit window (milliseconds) | `30000`, `120000` |

#### Optional: Vercel KV Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `KV_REST_API_URL` | No | Vercel KV REST API endpoint |
| `KV_REST_API_TOKEN` | No | Vercel KV authentication token |

**Note:** KV variables are optional. System works with just in-memory storage.

### Recommended Configurations

#### 1. Development (Small Memory)

```bash
HOLOLOOM_MEMORY_BACKEND=INMEMORY
HOLOLOOM_MAX_MEMORIES=100
HOLOLOOM_CACHE_TTL=600
RATE_LIMIT_MAX_REQUESTS=10
RATE_LIMIT_WINDOW_MS=60000
```

#### 2. Production (No KV)

```bash
HOLOLOOM_MEMORY_BACKEND=INMEMORY
HOLOLOOM_MAX_MEMORIES=1000
HOLOLOOM_CACHE_TTL=3600
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_MS=60000
```

#### 3. Production (With Vercel KV - Persistent)

```bash
HOLOLOOM_MEMORY_BACKEND=INMEMORY
HOLOLOOM_MAX_MEMORIES=10000
HOLOLOOM_CACHE_TTL=86400
RATE_LIMIT_MAX_REQUESTS=1000
RATE_LIMIT_WINDOW_MS=60000
KV_REST_API_URL=https://your-kv.kv.vercel.sh
KV_REST_API_TOKEN=your-token-here
```

#### 4. High-Traffic (Auto-Scaling)

```bash
HOLOLOOM_MEMORY_BACKEND=INMEMORY
HOLOLOOM_MAX_MEMORIES=500
HOLOLOOM_CACHE_TTL=1800
RATE_LIMIT_MAX_REQUESTS=500
RATE_LIMIT_WINDOW_MS=30000
```

### Local Development Setup

```bash
# Create .env.local
cat > .env.local << 'EOF'
HOLOLOOM_MEMORY_BACKEND=INMEMORY
HOLOLOOM_MAX_MEMORIES=1000
HOLOLOOM_CACHE_TTL=3600
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_MS=60000
EOF

# Install dependencies
npm install

# Start development server
npm run dev

# Server runs on http://localhost:3000

# Test locally
curl http://localhost:3000/api/health
```

---

## Testing

### 1. Health Check

```bash
curl https://your-app.vercel.app/api/health | jq
```

**Expected:**
```json
{
  "status": "healthy",
  "uptime_seconds": 123
}
```

### 2. Store Memory

```bash
curl -X POST https://your-app.vercel.app/api/experience \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Thompson Sampling is a Bayesian algorithm",
    "confidence": 0.95
  }' | jq
```

**Expected:**
```json
{
  "id": "mem_1704067200000_abc123",
  "cached": true,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 3. Recall Memory

```bash
curl "https://your-app.vercel.app/api/recall?q=thompson&limit=5" | jq
```

**Expected:**
```json
{
  "query": "thompson",
  "results": [
    {
      "id": "mem_1704067200000_abc123",
      "content": "Thompson Sampling is a Bayesian algorithm",
      "relevance": 0.95
    }
  ],
  "count": 1
}
```

### 4. Full Query Cycle

```bash
curl -X POST https://your-app.vercel.app/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "bandit algorithms",
    "experience": {
      "content": "Thompson Sampling uses Beta distribution",
      "confidence": 0.9
    },
    "recall_limit": 5
  }' | jq
```

### 5. Rate Limiting Test

```bash
# Get current rate limit headers
curl -i https://your-app.vercel.app/api/health | grep "X-RateLimit"

# Output:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 99
# X-RateLimit-Reset: 1704067260
```

### 6. Stress Test (Load Testing)

```bash
# Install Apache Bench
# macOS: brew install httpd
# Ubuntu: apt-get install apache2-utils

# Simple load test
ab -n 100 -c 10 https://your-app.vercel.app/api/health

# With POST requests
ab -n 50 -c 5 -p post.json -T application/json \
  https://your-app.vercel.app/api/experience
```

### 7. Automated Testing Script

```bash
#!/bin/bash
# test-deployment.sh

BASE_URL="${1:-https://your-app.vercel.app}"

echo "🧪 Testing HoloLoom API at $BASE_URL"

# Test 1: Health check
echo -n "✓ Health check... "
HEALTH=$(curl -s "$BASE_URL/api/health" | jq -r '.status')
[ "$HEALTH" = "healthy" ] && echo "OK" || echo "FAILED"

# Test 2: Store memory
echo -n "✓ Store memory... "
STORE=$(curl -s -X POST "$BASE_URL/api/experience" \
  -H "Content-Type: application/json" \
  -d '{"content":"Test memory","confidence":0.9}' \
  | jq -r '.id')
[ ! -z "$STORE" ] && echo "OK ($STORE)" || echo "FAILED"

# Test 3: Recall memory
echo -n "✓ Recall memory... "
COUNT=$(curl -s "$BASE_URL/api/recall?q=test&limit=5" | jq -r '.count')
[ "$COUNT" -ge 0 ] && echo "OK ($COUNT results)" || echo "FAILED"

# Test 4: Rate limiting
echo -n "✓ Rate limiting... "
LIMIT=$(curl -s -i "$BASE_URL/api/health" | grep "X-RateLimit-Limit" | cut -d' ' -f2)
[ ! -z "$LIMIT" ] && echo "OK (limit: $LIMIT)" || echo "FAILED"

# Test 5: Stats
echo -n "✓ Statistics... "
STATS=$(curl -s "$BASE_URL/api/stats" | jq -r '.storage.total_memories')
[ ! -z "$STATS" ] && echo "OK ($STATS memories)" || echo "FAILED"

echo "✅ All tests passed!"
```

**Run tests:**
```bash
chmod +x test-deployment.sh
./test-deployment.sh https://your-app.vercel.app
```

---

## Production Checklist

- [ ] **Deployment**
  - [ ] Vercel project created
  - [ ] GitHub repository connected
  - [ ] Environment variables set
  - [ ] Initial deployment successful

- [ ] **Testing**
  - [ ] Health check passes
  - [ ] Store memory works
  - [ ] Recall memory works
  - [ ] Rate limiting works
  - [ ] No console errors

- [ ] **Monitoring**
  - [ ] Vercel Analytics enabled
  - [ ] Error tracking set up
  - [ ] Log monitoring configured
  - [ ] Alerts configured for:
    - High memory usage (>80%)
    - High error rate (>1%)
    - Deployment failures

- [ ] **Configuration**
  - [ ] Environment variables finalized
  - [ ] Rate limits appropriate for traffic
  - [ ] Cache TTL set correctly
  - [ ] Memory limits adjusted for workload

- [ ] **Security**
  - [ ] No sensitive data in memories
  - [ ] HTTPS enforced
  - [ ] Rate limiting prevents abuse
  - [ ] API not publicly indexing sensitive data

- [ ] **Documentation**
  - [ ] Team knows API endpoints
  - [ ] Rate limits documented
  - [ ] Error handling documented
  - [ ] Incident response procedure documented

---

## Monitoring

### Real-Time Monitoring

```bash
# Watch deployment logs
vercel logs --tail

# Watch stats every 10 seconds
watch -n 10 'curl -s https://your-app.vercel.app/api/stats | jq'

# Monitor specific metric
watch -n 5 'curl -s https://your-app.vercel.app/api/stats | \
  jq ".storage | {total_memories, memory_usage_mb}"'
```

### Vercel Dashboard

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select project
3. View:
   - Deployments
   - Analytics
   - Logs
   - Function Metrics
   - Performance

### Key Metrics to Monitor

**Memory Usage:**
```bash
curl -s https://your-app.vercel.app/api/stats | jq '.memory'
```

**Storage Stats:**
```bash
curl -s https://your-app.vercel.app/api/stats | jq '.storage'
```

**Rate Limit Tracking:**
```bash
curl -s https://your-app.vercel.app/api/stats | jq '.rate_limiting'
```

### Alerts Setup

In Vercel Dashboard → Project Settings → Alerts:

1. **High Memory Usage**
   - Trigger: Heap used > 80%
   - Action: Email notification

2. **High Error Rate**
   - Trigger: Error rate > 1%
   - Action: Email + Slack

3. **Deployment Failure**
   - Trigger: Deploy status = failed
   - Action: Email notification

---

## Troubleshooting

### Issue: 429 Too Many Requests

**Symptoms:**
```
HTTP/1.1 429 Too Many Requests
{
  "error": "Rate limit exceeded",
  "remaining": 0,
  "retry_after_seconds": 45
}
```

**Solutions:**
1. **Wait and retry:**
   ```bash
   sleep 60  # Wait for window reset
   curl https://your-app.vercel.app/api/health
   ```

2. **Increase rate limit:**
   - Vercel Dashboard → Settings → Environment Variables
   - Increase `RATE_LIMIT_MAX_REQUESTS`
   - Redeploy

3. **Spread requests over time:**
   - Add delays between requests
   - Batch operations

### Issue: Memory Growing Too Large

**Symptoms:**
```json
{
  "storage": {
    "total_memories": 1000,
    "memory_usage_mb": 8.5
  }
}
```

**Solutions:**
1. **Reduce max memories:**
   ```bash
   # Set in Vercel Dashboard
   HOLOLOOM_MAX_MEMORIES=500
   vercel deploy --prod
   ```

2. **Reduce TTL:**
   ```bash
   HOLOLOOM_CACHE_TTL=1800  # 30 minutes instead of 1 hour
   ```

3. **Enable Vercel KV:**
   - Create KV store in Vercel
   - Add `KV_REST_API_URL` and `KV_REST_API_TOKEN`
   - Redeploy

### Issue: Memories Not Found in Recall

**Symptoms:**
```json
{
  "query": "my search",
  "results": [],
  "count": 0
}
```

**Solutions:**
1. **Check if memories exist:**
   ```bash
   curl https://your-app.vercel.app/api/stats | jq '.storage.total_memories'
   ```

2. **Try exact substring:**
   ```bash
   # If memory contains "Thompson Sampling"
   curl "https://your-app.vercel.app/api/recall?q=Thompson+Sampling"
   ```

3. **Check memory age:**
   ```bash
   curl https://your-app.vercel.app/api/stats | \
     jq '.storage.oldest_memory_seconds'
   ```

4. **Increase TTL:**
   ```bash
   HOLOLOOM_CACHE_TTL=7200  # 2 hours
   ```

### Issue: Cold Starts/Slow Response

**Symptoms:**
```
First request: 2-3 seconds
Second request: 50ms
```

**Causes:**
- Normal for serverless (function startup)
- Memory initialization
- First JIT compilation

**Solutions:**
1. **Accept cold starts** - Vercel caches containers
2. **Keep function warm** - Periodic ping endpoint
3. **Optimize code** - Remove unnecessary imports
4. **Use regional pinning** - Vercel Pro feature

**Keep warm script:**
```bash
# Keep function alive with periodic health checks
*/5 * * * * curl -s https://your-app.vercel.app/api/health > /dev/null
```

### Issue: KV Not Persisting Data

**Symptoms:**
```json
{
  "experience_stored": {
    "id": "mem_123",
    "cached": false
  }
}
```

**Solutions:**
1. **Verify KV credentials:**
   ```bash
   vercel env list
   # Check KV_REST_API_URL and KV_REST_API_TOKEN
   ```

2. **Check KV status:**
   - Vercel Dashboard → Storage → KV Store
   - Ensure store is active

3. **Test KV connection:**
   ```bash
   curl "$KV_REST_API_URL/get/test" \
     -H "Authorization: Bearer $KV_REST_API_TOKEN"
   ```

4. **Fallback to in-memory:**
   - Remove KV env variables
   - System automatically uses in-memory storage

---

## Advanced Setup

### Custom Domain

1. **Add domain in Vercel:**
   - Project → Settings → Domains
   - Add custom domain
   - Configure DNS

2. **Update API calls:**
   ```bash
   # Instead of
   https://hololoom-lite.vercel.app/api/...

   # Use
   https://api.yourdomain.com/api/...
   ```

### CORS Setup

```typescript
// In api/lib/cors.ts (create new file)
export function applyCORSHeaders(req: any, res: any) {
  const origin = req.headers.origin || '*';

  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
}

// In each API route
import { applyCORSHeaders } from './lib/cors';

export default function handler(req: VercelRequest, res: VercelResponse) {
  applyCORSHeaders(req, res);
  // ... rest of handler
}
```

### Custom Analytics

```typescript
// Track usage
const analytics = {
  experience_calls: 0,
  recall_calls: 0,
  query_calls: 0
};

// Log to Vercel Analytics
vercel analytics {
  "event": "api_call",
  "endpoint": "/api/experience",
  "latency_ms": 45
}
```

### Webhook Integration

```typescript
// Send notifications on high memory
if (stats.memory_usage_mb > 80) {
  await fetch('https://hooks.slack.com/...', {
    method: 'POST',
    body: JSON.stringify({
      text: `⚠️ HoloLoom memory high: ${stats.memory_usage_mb}MB`
    })
  });
}
```

---

## Performance Optimization Tips

1. **Enable caching:**
   ```bash
   HOLOLOOM_CACHE_TTL=7200  # 2 hours
   ```

2. **Right-size max memories:**
   ```bash
   # For 1000 memories: ~5-10MB
   # For 500 memories: ~3-5MB
   HOLOLOOM_MAX_MEMORIES=750
   ```

3. **Optimize rate limits:**
   ```bash
   # High traffic: higher limit with shorter window
   RATE_LIMIT_MAX_REQUESTS=500
   RATE_LIMIT_WINDOW_MS=30000  # 30 seconds
   ```

4. **Use Vercel KV for persistence:**
   - Frees up function memory
   - Survives redeployments
   - Shared across instances

---

## Getting Help

- **Documentation:** [VERCEL_README.md](VERCEL_README.md)
- **Issues:** [GitHub Issues](https://github.com/youruser/mythrl/issues)
- **Discussions:** [GitHub Discussions](https://github.com/youruser/mythrl/discussions)
- **Vercel Support:** [Vercel Docs](https://vercel.com/docs)

---

**Happy deploying! 🚀**
