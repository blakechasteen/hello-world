# Redis Cache Loss Recovery Playbook

**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**RTO Target**: 10 minutes
**RPO Target**: Acceptable (cache is ephemeral)

## Overview

This playbook covers recovery from Redis TTS cache loss, corruption, or performance degradation. Redis cache failure impacts TTS performance but does not prevent system operation (graceful degradation).

## Symptoms

- TTS queries are slow (regenerating audio instead of cache hits)
- Redis container not running
- Redis connection errors in logs
- Cache hit rate drops to 0%
- Memory pressure warnings
- Eviction rate spike

## Impact Assessment

**Severity**: MEDIUM (system degrades but remains operational)

- **User Impact**: Slower TTS responses (150ms → 2-5s)
- **System Impact**: Increased CPU/GPU load for TTS generation
- **Data Loss**: None (cache is ephemeral, can be rebuilt)

## Prerequisites

- Docker access
- Redis backup (if available)
- Knowledge of Redis configuration

## Recovery Steps

### Step 1: Diagnose Redis Status

**Time**: 1-2 minutes

```bash
# Check if Redis is running
docker ps | grep hololoom-tts-cache

# If running, check health
docker exec hololoom-tts-cache redis-cli PING
# Expected: PONG

# Check memory usage
docker exec hololoom-tts-cache redis-cli INFO memory

# Check hit rate
docker exec hololoom-tts-cache redis-cli INFO stats | grep keyspace
```

**Common Issues**:
- Container not running → Restart
- Out of memory → Clear old keys or increase memory
- Connection errors → Network issue
- High eviction rate → Increase maxmemory

### Step 2A: Quick Restart (Container Down)

**Time**: 2 minutes

```bash
# Restart Redis container
docker-compose -f docker-compose.voice.yml restart redis

# Verify startup
docker logs hololoom-tts-cache --tail 20

# Test connectivity
docker exec hololoom-tts-cache redis-cli PING
```

**If successful**, skip to Step 5 (Verify System).

### Step 2B: Clear and Rebuild (Memory Issues)

**Time**: 1 minute

```bash
# If Redis is out of memory, clear cache
docker exec hololoom-tts-cache redis-cli FLUSHALL

# Restart Redis
docker-compose -f docker-compose.voice.yml restart redis
```

**Note**: Cache will rebuild automatically as TTS requests come in. Performance will be degraded temporarily.

### Step 2C: Restore from Backup (Data Corruption)

**Time**: 5-7 minutes

```bash
# Stop Redis
docker-compose -f docker-compose.voice.yml down redis

# Find latest backup
ls -lt /var/backups/hololoom/hololoom_backup_*/redis.rdb

# Or download from S3
LATEST_BACKUP=$(aws s3 ls s3://hololoom-backups/ | grep hololoom_backup | tail -1 | awk '{print $4}')
aws s3 cp "s3://hololoom-backups/$LATEST_BACKUP" /tmp/
tar -xzf "/tmp/$LATEST_BACKUP" -C /tmp/

# Restore RDB file
docker volume rm redis_data || true
docker volume create redis_data
docker run --rm \
  -v redis_data:/data \
  -v /tmp/$(tar -tzf "/tmp/$LATEST_BACKUP" | head -1)/redis.rdb:/backup/dump.rdb \
  alpine sh -c "cp /backup/dump.rdb /data/dump.rdb && chmod 644 /data/dump.rdb"

# Start Redis
docker-compose -f docker-compose.voice.yml up -d redis

# Wait for loading
sleep 10

# Verify data loaded
docker exec hololoom-tts-cache redis-cli DBSIZE
# Should show non-zero number of keys
```

### Step 3: Verify Redis Functionality

**Time**: 2 minutes

```bash
# Test basic operations
docker exec hololoom-tts-cache redis-cli SET test_key "hello"
docker exec hololoom-tts-cache redis-cli GET test_key
docker exec hololoom-tts-cache redis-cli DEL test_key

# Check cache size
docker exec hololoom-tts-cache redis-cli DBSIZE

# Check memory usage
docker exec hololoom-tts-cache redis-cli INFO memory | grep used_memory_human
```

### Step 4: Test TTS Caching

**Time**: 2 minutes

```bash
# Make TTS request (should cache)
curl -X POST http://localhost:8000/voice/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Testing Redis cache recovery",
    "language": "en",
    "voice": "en-US-Neural2-A"
  }' \
  -o /tmp/test1.wav

# Same request (should hit cache)
curl -X POST http://localhost:8000/voice/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Testing Redis cache recovery",
    "language": "en",
    "voice": "en-US-Neural2-A"
  }' \
  -o /tmp/test2.wav

# Check if cached (should be identical and fast)
diff /tmp/test1.wav /tmp/test2.wav
# Should show: Files are identical
```

### Step 5: Monitor Cache Performance

**Time**: 3 minutes

```bash
# Watch cache hit rate
watch -n 1 'docker exec hololoom-tts-cache redis-cli INFO stats | grep keyspace'

# Check Grafana TTS Cache dashboard
open http://localhost:3000/d/tts-cache

# Monitor for 3-5 minutes, verify:
# - Hit rate increasing (as cache warms up)
# - No evictions (unless by design)
# - Memory stable
```

### Step 6: Optimize Configuration (If Needed)

**Time**: 2 minutes

If memory issues persist:

```yaml
# In docker-compose.voice.yml, adjust Redis config:
services:
  redis:
    environment:
      - REDIS_MAXMEMORY=2gb  # Increase from 1gb
      - REDIS_MAXMEMORY_POLICY=allkeys-lru  # LRU eviction
```

Restart Redis:
```bash
docker-compose -f docker-compose.voice.yml restart redis
```

## Post-Recovery Checklist

- [ ] Redis container running
- [ ] Redis responding to PING
- [ ] Cache hit rate > 0%
- [ ] Memory usage normal (<80% of maxmemory)
- [ ] TTS requests completing successfully
- [ ] Cache hit rate trending upward
- [ ] No errors in Redis logs
- [ ] Grafana dashboard showing metrics

## Cache Performance Expectations

**Warm Cache** (after recovery):
- Hit rate: 70-90%
- TTS latency: 50-150ms
- Memory usage: 500MB - 1.5GB

**Cold Cache** (after full flush):
- Hit rate: 0% → 70% over 1-2 hours
- TTS latency: 2-5s (generating audio)
- Memory usage: 0MB → 500MB over 1-2 hours

**Healthy Metrics**:
```
# Example healthy output
used_memory_human:800MB
keyspace_hits:15234
keyspace_misses:2456
evicted_keys:0
hit_rate:86.1%
```

## Common Issues and Solutions

### Issue: Redis OOM (Out of Memory)

**Symptoms**:
```
OOM command not allowed when used memory > 'maxmemory'
```

**Solution**:
```bash
# Option 1: Increase memory
# Edit docker-compose.voice.yml → REDIS_MAXMEMORY=2gb

# Option 2: Enable eviction
docker exec hololoom-tts-cache redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Option 3: Clear old keys
docker exec hololoom-tts-cache redis-cli --scan --pattern "tts:*" | \
  xargs -L 100 docker exec hololoom-tts-cache redis-cli DEL
```

### Issue: Slow Cache Performance

**Symptoms**:
- High latency on cache hits (>50ms)

**Solution**:
```bash
# Check for slow queries
docker exec hololoom-tts-cache redis-cli SLOWLOG GET 10

# Check CPU/memory on host
docker stats hololoom-tts-cache

# Consider: Move Redis to dedicated server
```

### Issue: Cache Persistence Failed

**Symptoms**:
```
Background saving error
```

**Solution**:
```bash
# Check disk space
df -h

# Disable persistence (if not needed)
docker exec hololoom-tts-cache redis-cli CONFIG SET save ""

# Or fix permissions
docker run --rm -v redis_data:/data alpine chmod 777 /data
```

## Prevention Strategies

### 1. Memory Monitoring

Set up alerts:
```yaml
# Prometheus alert
- alert: RedisCacheMemoryHigh
  expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
  for: 5m
  annotations:
    summary: "Redis cache memory usage high"
```

### 2. Regular Backups

```bash
# Daily Redis persistence
# Crontab: 0 3 * * *
docker exec hololoom-tts-cache redis-cli BGSAVE
```

### 3. Cache Warming

```bash
# Pre-populate cache with common phrases
python3 scripts/warm_tts_cache.py --phrases common_phrases.yaml
```

### 4. Eviction Policy

Configure appropriate eviction:
```yaml
# For TTS cache, use LRU (least recently used)
REDIS_MAXMEMORY_POLICY: allkeys-lru
```

### 5. Monitoring Dashboard

Grafana panels to monitor:
- Cache hit rate
- Memory usage
- Eviction rate
- Key count
- Commands/sec
- Network I/O

## Cache Rebuilding Strategy

After complete cache loss, expect gradual performance improvement:

```
Time     Hit Rate    Avg Latency
0 min    0%          3000ms
30 min   40%         1500ms
1 hour   60%         800ms
2 hours  75%         400ms
4 hours  85%         200ms
1 day    90%         150ms
```

**Accelerate warming**:
```bash
# Replay recent queries from logs
python3 scripts/replay_queries.py --last-hours 24 --limit 1000
```

## Escalation

Cache issues are typically self-healing. Escalate if:

- Redis repeatedly crashes (>3 times/hour)
- Memory leaks detected
- Performance doesn't improve after 2 hours
- Underlying infrastructure issues

**Contact**:
- **Slack**: #platform-redis
- **Email**: platform-oncall@hololoom.ai

## Impact on SLAs

**TTS Response Time SLA**: < 200ms (p99)

With Redis down:
- **Cold cache**: 2-5s (SLA violated)
- **Warm cache**: 150-200ms (SLA met)

Acceptable degradation period: **1-2 hours** (cache warming)

## References

- [Redis Documentation](https://redis.io/documentation)
- [TTS Cache Implementation](../../HoloLoom/voice/tts_cache.py)
- [TTS Cache README](../../HoloLoom/voice/TTS_CACHE_README.md)
- [Grafana TTS Dashboard](../grafana/dashboards/tts-cache.json)

---

**Last Reviewed**: 2025-11-16
**Reviewer**: Agent H - Wave 3 Production Hardening
**Next Review**: 2025-12-16
