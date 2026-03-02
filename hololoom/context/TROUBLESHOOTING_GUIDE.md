# HoloLoom Production Troubleshooting Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-13
**Part 5: Production Hardening - Day 25**

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Log Analysis](#log-analysis)
3. [Performance Issues](#performance-issues)
4. [Circuit Breaker Problems](#circuit-breaker-problems)
5. [Rate Limiting Issues](#rate-limiting-issues)
6. [Memory and Resource Problems](#memory-and-resource-problems)
7. [Backend Connectivity](#backend-connectivity)
8. [Health Check Failures](#health-check-failures)
9. [Error Messages Reference](#error-messages-reference)
10. [Advanced Debugging](#advanced-debugging)

---

## Quick Diagnostic Checklist

When HoloLoom is experiencing issues, run through this checklist first:

### 1. Basic Health Check

```bash
# Check health endpoint
curl http://localhost:8080/health

# Expected healthy response:
# {"healthy": true, "status": "healthy", "checks": {...}}

# If unhealthy, check which component is degraded:
# - overall: High error rate or latency
# - backends: Circuit breakers open
# - learning: Poor calibration
# - resources: High memory/CPU
```

### 2. Metrics Overview

```bash
# Get current metrics
curl http://localhost:8080/metrics | python -m json.tool

# Key metrics to check:
# - performance.query_count: Total queries processed
# - performance.error_rate: Should be < 0.10 (10%)
# - performance.latency_p95: Should be < 1000ms
# - resources.memory_mb: Should be < 1600MB (80% of 2GB)
# - resources.cpu_percent: Should be < 80%
```

### 3. Circuit Breaker Status

```bash
# Check circuit breaker states
curl http://localhost:8080/circuit-breakers | python -m json.tool

# Look for:
# - Any breakers in "open" state (backend is down)
# - Any breakers in "half_open" state (recovering)
# - High failure_count (backend is flaky)
```

### 4. Recent Logs

```bash
# Tail application logs
tail -100 /var/log/hololoom/app.log | grep -E "(ERROR|WARNING)"

# Check for patterns:
# - Repeated connection errors → Backend connectivity
# - Rate limit exceeded → Traffic spike
# - Memory warnings → Resource exhaustion
# - Circuit breaker opened → Backend failure
```

### 5. Docker/Pod Status

```bash
# Docker
docker ps -a | grep hololoom
docker logs hololoom-api --tail=50

# Kubernetes
kubectl get pods -n hololoom
kubectl logs -n hololoom hololoom-api-<pod-id> --tail=50
kubectl describe pod -n hololoom hololoom-api-<pod-id>
```

---

## Log Analysis

### Log Levels and What They Mean

HoloLoom uses standard Python logging levels:

| Level | When Used | Action Required |
|-------|-----------|-----------------|
| **DEBUG** | Detailed diagnostic info | Review for deep debugging |
| **INFO** | Normal operations | Review for trends |
| **WARNING** | Recoverable issues | Investigate if frequent |
| **ERROR** | Operation failures | Immediate investigation |
| **CRITICAL** | System-level failures | Immediate escalation |

### Common Log Patterns

#### Pattern 1: Backend Connection Errors

```
ERROR [2025-11-13 10:23:45] [PRODUCTION] Backend error: Neo4j connection failed
WARNING [2025-11-13 10:23:45] [PRODUCTION] Circuit breaker opened: neo4j (failures: 5)
INFO [2025-11-13 10:23:45] [PRODUCTION] Falling back to INMEMORY backend
```

**Diagnosis**: Neo4j backend is down or unreachable.

**Resolution**:
1. Check Neo4j container status: `docker ps | grep neo4j`
2. Check Neo4j logs: `docker logs neo4j`
3. Verify network connectivity: `telnet neo4j 7687`
4. If Neo4j is down, restart: `docker restart neo4j`
5. System will auto-recover when Neo4j is back (half-open → closed)

#### Pattern 2: Rate Limiting

```
WARNING [2025-11-13 10:25:30] [PRODUCTION] Rate limit exceeded for query: "What is..."
INFO [2025-11-13 10:25:30] Rate limiter stats: total=1050, rejected=50, current_concurrent=52
```

**Diagnosis**: Traffic spike exceeded rate limits.

**Resolution**:
1. Check if legitimate traffic or attack: Review query sources
2. If legitimate, increase rate limits:
   ```python
   config.rate_limit.global_qps = 2000.0  # from 1000.0
   config.rate_limit.max_concurrent = 150  # from 100
   ```
3. If attack, enable IP blocking at load balancer level
4. Monitor rejected rate: Should be < 5% of total

#### Pattern 3: Memory Warnings

```
WARNING [2025-11-13 10:30:15] [PRODUCTION] High memory usage: 1750.5 MB (85.5% of limit)
WARNING [2025-11-13 10:30:16] [PRODUCTION] Resource health degraded: memory threshold exceeded
INFO [2025-11-13 10:30:16] [PRODUCTION] Cache eviction triggered (LRU)
```

**Diagnosis**: Memory approaching limits, cache eviction triggered.

**Resolution**:
1. Check if temporary spike: Monitor for 5 minutes
2. If sustained, increase memory limit:
   ```python
   config.resource.max_memory_mb = 4096  # from 2048
   ```
3. Or reduce cache size:
   ```python
   config.resource.max_cache_size = 5000  # from 10000
   ```
4. Check for memory leaks: Profile with `memory_profiler`

#### Pattern 4: High Error Rate

```
ERROR [2025-11-13 10:35:20] [PRODUCTION] Query failed: LowConfidence (confidence: 0.32)
WARNING [2025-11-13 10:35:21] [PRODUCTION] Error rate: 15.2% (threshold: 10%)
WARNING [2025-11-13 10:35:21] [PRODUCTION] Overall health: UNHEALTHY
```

**Diagnosis**: High percentage of queries failing (>10% error rate).

**Resolution**:
1. Check query patterns: Are queries malformed or out-of-scope?
2. Check knowledge base completeness: May need more training data
3. Check backend connectivity: Multiple backend failures?
4. Enable refinement for low-confidence queries:
   ```python
   config.enable_refinement = True
   config.refinement_threshold = 0.75
   ```

#### Pattern 5: Circuit Breaker Stuck Open

```
WARNING [2025-11-13 10:40:00] [PRODUCTION] Circuit breaker stuck in OPEN state: qdrant (60s)
INFO [2025-11-13 10:40:01] [PRODUCTION] Manual intervention required: Reset breaker
```

**Diagnosis**: Circuit breaker has been open for extended period, not recovering.

**Resolution**:
1. Verify backend is actually healthy: `curl http://qdrant:6333/health`
2. If healthy, manually reset breaker:
   ```python
   orchestrator.breaker_registry.get_or_create("qdrant").force_close()
   ```
3. If still failing, check for:
   - Network connectivity issues
   - Authentication problems
   - Resource exhaustion on backend

### Log Aggregation Queries

If using centralized logging (ELK, Splunk, Loki):

**Find all errors in last hour**:
```
level:ERROR AND timestamp:[now-1h TO now]
```

**Find rate limit rejections**:
```
message:"Rate limit exceeded" AND timestamp:[now-1h TO now]
```

**Find circuit breaker events**:
```
message:"Circuit breaker" AND (state:open OR state:half_open)
```

**Find slow queries (>1s)**:
```
message:"latency_ms" AND latency_ms:>1000
```

---

## Performance Issues

### Symptom 1: High Query Latency (p95 > 1000ms)

**Diagnosis Steps**:

1. **Check metrics breakdown**:
   ```bash
   curl http://localhost:8080/metrics | jq '.performance'
   ```
   Look at:
   - `latency_p50`: Should be <200ms
   - `latency_p95`: Should be <1000ms
   - `latency_p99`: Indicates tail latency

2. **Identify bottleneck**:
   ```python
   # Enable detailed stage timing
   config.enable_stage_timing = True

   # Check Spacetime trace
   spacetime = await orchestrator.weave(query)
   print(spacetime.trace.stage_durations)

   # Example output:
   # {'retrieval': 450.2, 'decision': 120.5, 'execution': 380.1}
   ```

3. **Common bottlenecks**:

   **a) Retrieval slow (>500ms)**:
   ```python
   # Solution: Reduce retrieval limits
   config.retrieval.max_memories = 50  # from 100
   config.retrieval.max_depth = 2  # from 3

   # Or enable caching
   config.enable_compositional_cache = True
   config.parse_cache_size = 10000
   ```

   **b) Decision slow (>200ms)**:
   ```python
   # Solution: Simplify policy network
   config.policy_mode = "FAST"  # from "FULL"

   # Or reduce context size
   config.context_window = 512  # from 1024
   ```

   **c) Backend queries slow**:
   ```python
   # Solution: Add indexes to Neo4j
   CREATE INDEX entity_name FOR (n:Entity) ON (n.name)
   CREATE INDEX edge_type FOR ()-[r:RELATIONSHIP]-() ON (r.type)

   # Or switch to HYBRID mode with Qdrant
   config.memory_backend = MemoryBackend.HYBRID
   ```

4. **Check for cache effectiveness**:
   ```bash
   curl http://localhost:8080/metrics | jq '.performance.cache_hit_rate'

   # Should be >0.5 (50%) for typical workloads
   # If <0.3, investigate:
   # - Are queries too diverse?
   # - Is cache size too small?
   # - Is cache being evicted too quickly?
   ```

### Symptom 2: High CPU Usage (>80%)

**Diagnosis Steps**:

1. **Profile CPU usage**:
   ```bash
   # Docker
   docker stats hololoom-api

   # Kubernetes
   kubectl top pod -n hololoom hololoom-api-<pod-id>
   ```

2. **Identify hot code paths**:
   ```python
   # Use cProfile
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   await orchestrator.weave(query)

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions
   ```

3. **Common causes**:

   **a) Too many concurrent queries**:
   ```python
   # Solution: Reduce concurrent limit
   config.rate_limit.max_concurrent = 50  # from 100
   ```

   **b) Expensive embedding computation**:
   ```python
   # Solution: Use smaller embedding model
   config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # 384d
   # Instead of: "sentence-transformers/all-mpnet-base-v2"  # 768d

   # Or reduce scales
   config.matryoshka_scales = [96, 192]  # from [96, 192, 384]
   ```

   **c) Graph traversal expensive**:
   ```python
   # Solution: Limit traversal depth
   config.graph_max_depth = 2  # from 3
   config.graph_max_nodes = 100  # from 200
   ```

### Symptom 3: Memory Leak

**Diagnosis Steps**:

1. **Monitor memory over time**:
   ```bash
   # Watch memory usage
   watch -n 5 'curl -s http://localhost:8080/metrics | jq ".resources.memory_mb"'
   ```

2. **Profile memory allocation**:
   ```python
   from memory_profiler import profile

   @profile
   async def test_weave():
       for i in range(100):
           await orchestrator.weave(Query(text=f"Query {i}"))

   asyncio.run(test_weave())
   ```

3. **Common causes**:

   **a) Cache not evicting**:
   ```python
   # Solution: Enable LRU eviction
   config.cache_eviction_policy = "LRU"
   config.max_cache_size = 5000  # Hard limit
   ```

   **b) Background tasks not cleaning up**:
   ```python
   # Solution: Use context manager for proper cleanup
   async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
       await orchestrator.weave(query)
       # Automatic cleanup on exit
   ```

   **c) Large graph in memory**:
   ```python
   # Solution: Use persistent backend instead of INMEMORY
   config.memory_backend = MemoryBackend.HYBRID
   # Graph stored in Neo4j, not in-process memory
   ```

---

## Circuit Breaker Problems

### Symptom 1: Breaker Opens Immediately

**Error Message**:
```
WARNING [2025-11-13 10:45:00] Circuit breaker opened: neo4j (failures: 3)
```

**Diagnosis**:
1. Backend is actually down or unreachable
2. Failure threshold is too low
3. Backend is slow, causing timeouts

**Resolution**:

1. **Verify backend health**:
   ```bash
   # Neo4j
   curl http://neo4j:7474/

   # Qdrant
   curl http://qdrant:6333/health
   ```

2. **Check network connectivity**:
   ```bash
   # From HoloLoom container
   docker exec -it hololoom-api bash
   ping neo4j
   telnet neo4j 7687
   ```

3. **Increase failure threshold** (if backend is flaky but recovering):
   ```python
   config.circuit_breaker.failure_threshold = 10  # from 3
   ```

4. **Increase timeout** (if backend is slow):
   ```python
   config.circuit_breaker.timeout = 10.0  # from 5.0 seconds
   ```

### Symptom 2: Breaker Stuck in Half-Open

**Error Message**:
```
INFO [2025-11-13 10:50:00] Circuit breaker in HALF_OPEN: neo4j (testing recovery)
WARNING [2025-11-13 10:50:05] Circuit breaker reopened: neo4j (recovery failed)
```

**Diagnosis**:
Backend is intermittently failing during recovery tests.

**Resolution**:

1. **Increase recovery timeout** (give backend more time to stabilize):
   ```python
   config.circuit_breaker.recovery_timeout = 300.0  # from 120.0 seconds
   ```

2. **Check backend logs** for errors during recovery period:
   ```bash
   docker logs neo4j --since 5m
   ```

3. **Manual recovery** (if backend is actually healthy):
   ```python
   # Force breaker closed
   orchestrator.breaker_registry.get_or_create("neo4j").force_close()
   ```

### Symptom 3: Too Many Breakers Open

**Error Message**:
```
ERROR [2025-11-13 10:55:00] Backend health: DEGRADED (2 breakers open)
```

**Diagnosis**:
Multiple backends are failing simultaneously.

**Resolution**:

1. **Check for infrastructure-wide issues**:
   - Network outage?
   - DNS resolution failing?
   - Load balancer issues?
   - Kubernetes node failure?

2. **Verify Docker networking**:
   ```bash
   docker network inspect hololoom_default
   docker network ls
   ```

3. **Check service dependencies**:
   ```bash
   # Are backend services actually running?
   docker ps | grep -E "(neo4j|qdrant)"

   # Check resource usage
   docker stats
   ```

4. **Graceful degradation**:
   HoloLoom should automatically fall back to INMEMORY backend:
   ```
   INFO [2025-11-13 10:55:01] Falling back to INMEMORY backend
   ```

---

## Rate Limiting Issues

### Symptom 1: Legitimate Traffic Being Rejected

**Error Message**:
```
WARNING [2025-11-13 11:00:00] Rate limit exceeded for query: "What is..."
```

**Diagnosis**:
Rate limits are too restrictive for current traffic patterns.

**Resolution**:

1. **Check current rate limiter stats**:
   ```bash
   curl http://localhost:8080/metrics | jq '.rate_limiter'
   ```
   Output:
   ```json
   {
     "total_requests": 10500,
     "total_rejected": 500,
     "rejection_rate": 0.048,  # 4.8%
     "current_concurrent": 95
   }
   ```

2. **Increase limits** if rejection rate >5%:
   ```python
   # Global QPS limit
   config.rate_limit.global_qps = 2000.0  # from 1000.0

   # Per-session limit
   config.rate_limit.session_qps = 100.0  # from 50.0

   # Concurrent limit
   config.rate_limit.max_concurrent = 150  # from 100
   ```

3. **Increase burst capacity**:
   ```python
   # Allow larger bursts
   rate_limiter = create_rate_limiter(
       rate=1000.0,
       capacity=300,  # from 200 (30% burst)
       max_concurrent=100
   )
   ```

4. **Monitor over time** to find optimal limits:
   ```bash
   # Track rejection rate
   watch -n 10 'curl -s http://localhost:8080/metrics | jq ".rate_limiter.rejection_rate"'
   ```

### Symptom 2: Rate Limiter Not Rejecting Attacks

**Diagnosis**:
Rate limits are too permissive, allowing abusive traffic through.

**Resolution**:

1. **Analyze request patterns**:
   ```bash
   # Group queries by source IP
   grep "Rate limit" /var/log/hololoom/app.log | \
     awk '{print $5}' | sort | uniq -c | sort -rn | head -20
   ```

2. **Implement stricter per-IP limits**:
   ```python
   # Add IP-based rate limiting
   from HoloLoom.context import IPRateLimiter

   ip_limiter = IPRateLimiter(
       rate_per_ip=10.0,  # 10 QPS per IP
       capacity_per_ip=20,  # 20 burst per IP
       block_duration=300  # Block for 5 minutes
   )
   ```

3. **Enable request authentication**:
   ```python
   # Require API keys
   config.require_authentication = True
   config.api_key_rate_limit = 50.0  # Per key
   ```

### Symptom 3: Burst Traffic Causing Rejections

**Error Message**:
```
WARNING [2025-11-13 11:05:00] Rate limit exceeded (burst): 250 requests in 5s
```

**Diagnosis**:
Burst capacity too small for legitimate traffic spikes.

**Resolution**:

1. **Increase burst capacity**:
   ```python
   # Allow 50% burst (from 20%)
   rate_limiter = create_rate_limiter(
       rate=1000.0,
       capacity=500,  # 50% burst capacity
       max_concurrent=100
   )
   ```

2. **Use token bucket wait** instead of immediate rejection:
   ```python
   # Wait for token availability (up to 2 seconds)
   await rate_limiter.token_bucket.acquire_wait(timeout=2.0)
   ```

---

## Memory and Resource Problems

### Symptom 1: Out of Memory (OOM) Crashes

**Error Message**:
```
CRITICAL [2025-11-13 11:10:00] Out of memory: Container killed by OOM killer
```

**Diagnosis**:

1. **Check container memory limits**:
   ```bash
   # Docker
   docker inspect hololoom-api | jq '.[0].HostConfig.Memory'

   # Kubernetes
   kubectl describe pod -n hololoom hololoom-api-<pod-id> | grep -A 5 "Limits"
   ```

2. **Profile memory usage**:
   ```python
   from memory_profiler import memory_usage

   mem_before = memory_usage()[0]
   await orchestrator.weave(query)
   mem_after = memory_usage()[0]

   print(f"Memory used: {mem_after - mem_before:.2f} MB")
   ```

**Resolution**:

1. **Increase container memory**:
   ```yaml
   # docker-compose.yml
   services:
     hololoom-api:
       mem_limit: 4g  # from 2g

   # Kubernetes deployment.yaml
   resources:
     limits:
       memory: 4Gi  # from 2Gi
   ```

2. **Enable memory limits in config**:
   ```python
   config.resource.max_memory_mb = 3072  # 3GB (leave 1GB for OS)
   ```

3. **Reduce memory footprint**:
   ```python
   # Smaller cache
   config.resource.max_cache_size = 3000  # from 10000

   # Smaller embeddings
   config.matryoshka_scales = [96, 192]  # from [96, 192, 384]

   # Use persistent backend (not INMEMORY)
   config.memory_backend = MemoryBackend.HYBRID
   ```

### Symptom 2: High Memory Usage (>80%)

**Warning Message**:
```
WARNING [2025-11-13 11:15:00] High memory usage: 1750 MB (85% of limit)
```

**Diagnosis**:

1. **Check which component is using memory**:
   ```python
   import tracemalloc

   tracemalloc.start()

   # Run workload
   await orchestrator.weave(query)

   # Get top memory users
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')

   for stat in top_stats[:10]:
       print(stat)
   ```

**Resolution**:

1. **Trigger cache eviction**:
   ```python
   # Manual cache clear
   orchestrator.cache.clear()

   # Or configure automatic eviction
   config.cache_eviction_threshold = 0.7  # Evict at 70% memory
   ```

2. **Check for memory leaks**:
   ```python
   # Look for objects not being released
   import gc
   import sys

   gc.collect()

   # Count object types
   from collections import Counter
   type_counts = Counter(type(obj).__name__ for obj in gc.get_objects())
   print(type_counts.most_common(20))
   ```

### Symptom 3: Disk Space Exhaustion

**Error Message**:
```
ERROR [2025-11-13 11:20:00] Failed to write logs: No space left on device
```

**Diagnosis**:

1. **Check disk usage**:
   ```bash
   df -h /var/log/hololoom
   du -sh /var/log/hololoom/*
   ```

**Resolution**:

1. **Rotate logs**:
   ```bash
   # Manual rotation
   logrotate -f /etc/logrotate.d/hololoom

   # Or configure automatic rotation
   cat > /etc/logrotate.d/hololoom <<EOF
   /var/log/hololoom/*.log {
       daily
       rotate 7
       compress
       delaycompress
       missingok
       notifempty
       create 0640 hololoom hololoom
   }
   EOF
   ```

2. **Reduce log retention**:
   ```python
   config.logging.retention_days = 3  # from 7
   ```

3. **Clean up old data**:
   ```bash
   # Remove old checkpoints
   find /var/lib/hololoom/checkpoints -mtime +30 -delete

   # Remove old metrics
   find /var/lib/hololoom/metrics -mtime +14 -delete
   ```

---

## Backend Connectivity

### Symptom 1: Neo4j Connection Failed

**Error Message**:
```
ERROR [2025-11-13 11:25:00] Neo4j connection failed: ServiceUnavailable
```

**Diagnosis Steps**:

1. **Check Neo4j is running**:
   ```bash
   docker ps | grep neo4j
   docker logs neo4j --tail=50
   ```

2. **Verify network connectivity**:
   ```bash
   # From HoloLoom container
   docker exec -it hololoom-api bash
   telnet neo4j 7687
   ping neo4j
   ```

3. **Check authentication**:
   ```bash
   # Test connection manually
   cypher-shell -a bolt://neo4j:7687 -u neo4j -p password
   ```

**Resolution**:

1. **Restart Neo4j**:
   ```bash
   docker restart neo4j
   # Wait 30 seconds for full startup
   ```

2. **Fix authentication**:
   ```python
   # Update credentials in config
   config.neo4j_uri = "bolt://neo4j:7687"
   config.neo4j_user = "neo4j"
   config.neo4j_password = "correct_password"
   ```

3. **Check Neo4j logs** for errors:
   ```bash
   docker logs neo4j | grep -E "(ERROR|WARN)"
   ```

4. **Verify Neo4j health**:
   ```bash
   curl http://neo4j:7474/
   # Should return: {"bolt_routing": "neo4j://neo4j:7687", ...}
   ```

### Symptom 2: Qdrant Connection Failed

**Error Message**:
```
ERROR [2025-11-13 11:30:00] Qdrant connection failed: Connection refused
```

**Diagnosis Steps**:

1. **Check Qdrant is running**:
   ```bash
   docker ps | grep qdrant
   docker logs qdrant --tail=50
   ```

2. **Verify API accessibility**:
   ```bash
   curl http://qdrant:6333/health
   # Expected: {"title":"qdrant - vector search engine","version":"1.7.0"}
   ```

**Resolution**:

1. **Restart Qdrant**:
   ```bash
   docker restart qdrant
   ```

2. **Check Qdrant collection exists**:
   ```bash
   curl http://qdrant:6333/collections

   # If collection missing, create it:
   curl -X PUT http://qdrant:6333/collections/hololoom \
     -H 'Content-Type: application/json' \
     -d '{
       "vectors": {
         "size": 384,
         "distance": "Cosine"
       }
     }'
   ```

3. **Verify Qdrant storage**:
   ```bash
   # Check disk space for Qdrant volume
   docker exec -it qdrant df -h /qdrant/storage
   ```

### Symptom 3: Intermittent Connection Failures

**Error Message**:
```
WARNING [2025-11-13 11:35:00] Neo4j connection timeout (attempt 1/3)
INFO [2025-11-13 11:35:01] Retry successful
```

**Diagnosis**:
Network instability or backend overload.

**Resolution**:

1. **Increase retry attempts**:
   ```python
   config.error_handling.max_retries = 5  # from 3
   config.error_handling.retry_backoff = 2.0  # Exponential backoff
   ```

2. **Increase connection timeout**:
   ```python
   config.neo4j_timeout = 30.0  # from 10.0 seconds
   config.qdrant_timeout = 30.0  # from 10.0 seconds
   ```

3. **Enable connection pooling**:
   ```python
   config.neo4j_max_pool_size = 50  # from 10
   config.neo4j_connection_timeout = 60.0
   ```

---

## Health Check Failures

### Symptom 1: Overall Health Degraded

**Health Check Response**:
```json
{
  "healthy": false,
  "status": "unhealthy",
  "checks": {
    "overall": {
      "healthy": false,
      "status": "unhealthy",
      "message": "Error rate 15.2% exceeds threshold 10%"
    }
  }
}
```

**Diagnosis**:
High error rate across the system.

**Resolution**:

1. **Check error distribution**:
   ```bash
   # What types of errors?
   grep ERROR /var/log/hololoom/app.log | \
     awk '{print $NF}' | sort | uniq -c | sort -rn
   ```

2. **Identify failing queries**:
   ```python
   # Enable query logging
   config.logging.log_queries = True
   config.logging.log_failures = True

   # Review failed queries
   grep "Query failed" /var/log/hololoom/app.log
   ```

3. **Adjust error rate threshold** (if acceptable):
   ```python
   config.health_check.error_rate_threshold = 0.15  # from 0.10 (10%)
   ```

4. **Enable refinement** to reduce failures:
   ```python
   config.enable_refinement = True
   config.refinement_threshold = 0.75
   ```

### Symptom 2: Backend Health Degraded

**Health Check Response**:
```json
{
  "checks": {
    "backends": {
      "healthy": false,
      "status": "degraded",
      "message": "1 circuit breaker open",
      "details": {
        "open_breakers": ["neo4j"]
      }
    }
  }
}
```

**Diagnosis**:
Neo4j circuit breaker is open.

**Resolution**:
See [Circuit Breaker Problems](#circuit-breaker-problems) section above.

### Symptom 3: Learning Health Degraded

**Health Check Response**:
```json
{
  "checks": {
    "learning": {
      "healthy": false,
      "status": "degraded",
      "message": "Poor calibration: ECE 0.22 exceeds threshold 0.15"
    }
  }
}
```

**Diagnosis**:
Model calibration is poor (predictions not matching actual confidence).

**Resolution**:

1. **Review recent queries**:
   ```python
   # Check confidence vs actual accuracy
   stats = orchestrator.get_metrics()
   print(stats['learning'])
   ```

2. **Retrain Thompson Sampling priors**:
   ```python
   # Reset bandit statistics
   policy.bandit.reset_stats()

   # Or adjust exploration
   policy.bandit.epsilon = 0.15  # More exploration
   ```

3. **Adjust calibration threshold** (if acceptable):
   ```python
   config.health_check.calibration_threshold = 0.25  # from 0.15
   ```

---

## Error Messages Reference

### Production Hardening Errors

| Error | Meaning | Resolution |
|-------|---------|------------|
| `RateLimitExceededError` | Too many requests | Increase rate limits or reject traffic |
| `BackendError` | Backend operation failed | Check backend connectivity |
| `CircuitBreakerOpenError` | Circuit breaker protecting backend | Wait for recovery or fix backend |
| `HealthCheckFailedError` | Health check returned unhealthy | Investigate failing component |
| `ResourceExhaustedError` | Memory/CPU limit reached | Increase limits or reduce load |

### Context Module Errors

| Error | Meaning | Resolution |
|-------|---------|------------|
| `ContextError` | Base context error | Check logs for specific error |
| `RoutingError` | Query routing failed | Check knowledge graph connectivity |
| `Neo4jConnectionError` | Neo4j unreachable | Restart Neo4j or check network |
| `QdrantConnectionError` | Qdrant unreachable | Restart Qdrant or check network |
| `EmbeddingError` | Embedding computation failed | Check sentence-transformers installed |

### Configuration Errors

| Error | Meaning | Resolution |
|-------|---------|------------|
| `ConfigurationError` | Invalid config | Run config.validate() |
| `ValidationError` | Config validation failed | Check specific validation message |
| `EnvironmentDetectionError` | Can't detect environment | Set CONTEXT_ENV explicitly |

---

## Advanced Debugging

### Enable Debug Logging

```python
import logging

# Set root logger to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Or specific modules
logging.getLogger('HoloLoom.weaving_orchestrator').setLevel(logging.DEBUG)
logging.getLogger('HoloLoom.context').setLevel(logging.DEBUG)
```

### Capture Full Stack Traces

```python
import sys
import traceback

try:
    await orchestrator.weave(query)
except Exception as e:
    # Print full stack trace
    traceback.print_exc(file=sys.stdout)

    # Or capture to variable
    tb = traceback.format_exc()
    logger.error(f"Full traceback:\n{tb}")
```

### Profile Performance

```python
import cProfile
import pstats
from io import StringIO

profiler = cProfile.Profile()
profiler.enable()

# Run code to profile
await orchestrator.weave(query)

profiler.disable()

# Print stats
s = StringIO()
stats = pstats.Stats(profiler, stream=s)
stats.sort_stats('cumulative')
stats.print_stats(30)  # Top 30 functions
print(s.getvalue())
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
async def test_memory():
    for i in range(100):
        spacetime = await orchestrator.weave(Query(text=f"Query {i}"))

asyncio.run(test_memory())
```

### Network Debugging

```bash
# Capture packets between HoloLoom and Neo4j
tcpdump -i any -nn port 7687 -w neo4j_traffic.pcap

# Analyze with Wireshark or tcpdump
tcpdump -r neo4j_traffic.pcap -A | less
```

### Distributed Tracing (Future)

When distributed tracing is implemented (Part 6):

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("weave_query"):
    spacetime = await orchestrator.weave(query)

# View traces in Jaeger UI: http://localhost:16686
```

---

## When to Escalate

### P0 (Critical - Immediate Escalation)
- System completely down (all health checks failing)
- Data loss or corruption
- Security breach detected
- Multiple backends failing simultaneously

### P1 (High - Escalate within 1 hour)
- High error rate (>20%)
- All circuit breakers open
- Memory exhaustion causing OOM kills
- Performance degradation (p95 >2000ms)

### P2 (Medium - Escalate within 4 hours)
- Single backend failure with auto-fallback working
- Moderate error rate (10-20%)
- Single circuit breaker stuck open
- Disk space >80% full

### P3 (Low - Escalate within 24 hours)
- Warning-level health checks
- Poor calibration (ECE >0.15 but <0.25)
- Cache effectiveness <50%
- Non-critical configuration issues

---

## Useful Commands Reference

### Quick Health Check
```bash
curl http://localhost:8080/health | jq '.'
```

### Get Metrics
```bash
curl http://localhost:8080/metrics | jq '.performance'
```

### Check Circuit Breakers
```bash
curl http://localhost:8080/circuit-breakers | jq '.breakers'
```

### View Recent Logs
```bash
tail -100 /var/log/hololoom/app.log | grep ERROR
```

### Docker Container Stats
```bash
docker stats hololoom-api --no-stream
```

### Kubernetes Pod Logs
```bash
kubectl logs -n hololoom -l app=hololoom-api --tail=100 -f
```

### Test Backend Connectivity
```bash
# Neo4j
curl http://neo4j:7474/

# Qdrant
curl http://qdrant:6333/health
```

### Manual Cache Clear
```python
# Python
orchestrator.cache.clear()
```

### Force Circuit Breaker State
```python
# Close breaker
orchestrator.breaker_registry.get_or_create("neo4j").force_close()

# Open breaker
orchestrator.breaker_registry.get_or_create("neo4j").force_open()
```

---

## Additional Resources

- **Operations Runbook**: `OPERATIONS_RUNBOOK.md` (deployment, monitoring, incident response)
- **Production Integration Guide**: `PRODUCTION_INTEGRATION_COMPLETE.md` (usage examples)
- **Performance Tuning Guide**: `PERFORMANCE_TUNING_GUIDE.md` (optimization strategies)
- **API Reference**: `HoloLoom/context/README.md` (API documentation)
- **Test Suite**: `HoloLoom/context/test_integration_e2e.py` (end-to-end tests)

---

**End of Troubleshooting Guide**
