# HoloLoom VoiceAgent + Elle AR Load Testing Guide

**Status**: Production Ready (2025-11-16)
**Version**: 1.0.0
**Author**: Wave 3 Production Hardening

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Running Tests](#running-tests)
4. [Interpreting Results](#interpreting-results)
5. [Performance Baselines](#performance-baselines)
6. [Capacity Planning](#capacity-planning)
7. [Auto-Scaling Configuration](#auto-scaling-configuration)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)

---

## Quick Start

### For the Impatient

```bash
# Install Locust
pip install locust

# Start your HoloLoom VoiceAgent server
# (In separate terminal) python -m hololoom.server.agentic_api

# Run baseline test (10 users, 5 minutes)
cd /home/user/hello-world
PYTHONPATH=. locust -f tests/load/locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 5m --headless
```

Expected output:
```
=== Load Test Summary ===
Total Requests: ~1500
Failures: 0
Median Response Time: 150ms
95th Percentile: 300ms
Requests/sec: 5.0
✅ All endpoints within SLA
```

---

## Installation

### Prerequisites

- Python 3.9+
- HoloLoom VoiceAgent server running
- Locust 2.0+

### Step 1: Install Locust

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Locust
pip install locust==2.15.1

# Install optional dependencies
pip install requests pillow pandas matplotlib
```

### Step 2: Verify Installation

```bash
# Check Locust version
locust --version
# Expected: locust 2.15.1

# Run Locust help
locust --help
```

### Step 3: Start HoloLoom Server

```bash
# In terminal 1
cd /home/user/hello-world
PYTHONPATH=. python -m hololoom.server.agentic_api
# Expected: INFO: Uvicorn running on http://0.0.0.0:8000
```

Verify server is running:
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", ...}
```

---

## Running Tests

### Test Scenarios

HoloLoom provides four standard load test scenarios:

| Scenario | Users | Duration | Purpose | Commands |
|----------|-------|----------|---------|----------|
| **Baseline** | 10 | 5 min | Validate normal load | [See below](#baseline) |
| **Stress** | 100 | 10 min | Test ramping load | [See below](#stress) |
| **Spike** | 200 | 2 min | Test sudden surge | [See below](#spike) |
| **Endurance** | 50 | 1 hour | Test stability | [See below](#endurance) |

### Baseline Test

**Purpose**: Validate system under normal load (minimum test before deployment)

**Command**:
```bash
cd /home/user/hello-world
PYTHONPATH=. locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 10 \
    --spawn-rate 2 \
    --run-time 5m \
    --headless \
    --csv=tests/load/results/baseline
```

**Expected Results**:
- Total Requests: ~1,500
- Error Rate: <0.5%
- p50 Latency: ~150ms
- p95 Latency: <300ms
- Cache Hit Rate: 50-70%

**Success Criteria**: ✅ All endpoints within SLA

### Stress Test

**Purpose**: Validate performance under increasing load

**Command**:
```bash
cd /home/user/hello-world
PYTHONPATH=. locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m \
    --headless \
    --csv=tests/load/results/stress
```

**Expected Results**:
- Total Requests: ~10,000
- Error Rate: <2%
- p95 Latency: 300-800ms
- Cache Hit Rate: 40-60%
- Throughput: 50-150 RPS

**Watch For**:
- When does p95 exceed 500ms?
- When does CPU reach 70% (HPA trigger)?
- Any memory growth patterns?

### Spike Test

**Purpose**: Validate response to sudden traffic surge

**Command**:
```bash
cd /home/user/hello-world
PYTHONPATH=. locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 200 \
    --spawn-rate 50 \
    --run-time 2m \
    --headless \
    --csv=tests/load/results/spike
```

**Expected Results**:
- Initial surge: p95 could reach 1-2 seconds
- Auto-scaling triggered: <2 minutes
- Error Rate: <5% (queuing acceptable)
- Recovery Time: <5 minutes

**Critical Metrics**:
- Time to reach 1000ms p95
- Auto-scaling response lag
- Error rate during spike

### Endurance Test

**Purpose**: Validate long-running stability (1+ hour test)

**Command**:
```bash
cd /home/user/hello-world
PYTHONPATH=. locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 1h \
    --headless \
    --csv=tests/load/results/endurance
```

**Expected Results**:
- Stable p95 Latency: 200-400ms (no growth)
- Stable Error Rate: <0.5%
- Memory Growth: <10% after warmup
- No exceptions or crashes

**Monitor With** (in separate terminal):
```bash
# Watch CPU and memory
watch -n 1 'ps aux | grep hololoom'

# Watch memory
watch -n 1 'free -m'

# Watch network connections
watch -n 1 'netstat -an | grep ESTABLISHED | wc -l'
```

### Using Web UI

For interactive testing with Locust's web UI:

```bash
cd /home/user/hello-world
PYTHONPATH=. locust -f tests/load/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser:

1. Enter "Number of users": 10
2. Enter "Spawn rate": 2
3. Click "Start swarming"

The UI shows:
- Real-time request graphs
- Response time distribution
- Error rates by endpoint
- User count progression
- RPS (requests per second)

---

## Interpreting Results

### Key Metrics Explained

#### Latency Percentiles

- **p50 (Median)**: 50% of users experience this latency or better
  - Good baseline for typical user experience
  - Target: 100-200ms for voice commands

- **p95**: 95% of users experience this latency or better
  - **Primary SLA threshold** - most important
  - Most users notice if p95 > 500ms
  - Target: <300ms for voice queries

- **p99**: 99% of users experience this latency or better
  - Extreme case, acceptable if rare
  - Target: <500ms

- **max**: Absolute worst-case latency
  - Should be <2 seconds
  - Indicates queue depth problems if much higher

#### Example Interpretation

```
p50: 150ms ← Good, typical user experience
p95: 280ms ← Good, SLA target is <300ms
p99: 450ms ← Acceptable, majority of users unaffected
max: 1200ms ← One slow request, acceptable
```

**vs. Bad Results**:
```
p50: 450ms ← Users noticing slowness
p95: 1200ms ← SLA violation, users frustrated
p99: 2500ms ← System overloaded
max: 5000ms ← Critical slowness, possible timeout
```

#### Error Rate

- **<0.5%**: Excellent (expected for baseline)
- **0.5-2%**: Acceptable (some transient failures OK)
- **2-5%**: Concerning (investigate cause)
- **>5%**: Critical (system overloaded or broken)

#### Cache Hit Rate

- **>75%**: Excellent (caching working well)
- **50-75%**: Good (cold caches warming up)
- **25-50%**: Fair (may need cache tuning)
- **<25%**: Poor (investigate cache strategy)

### Example Report Interpretation

```
=== BASELINE TEST RESULTS ===

Total Requests: 1,487
Total Failures: 1
Overall Error Rate: 0.07%

Cache Performance:
  Cache Hits: 892
  Cache Misses: 595
  Cache Hit Rate: 60.0%

Endpoint Performance:
Endpoint                    Count    P50    P95    P99    Max  Errors
/query [voice_command]      600      145    280    420    1200    0
/tts/synthesize [cached]    150       15     28     45     120     0
/tts/synthesize [uncached]  150      520    780    980    1300    0
/health                     280        5     10     15      45     0
/stats                      120       50     95    150     280     0
/annotate-photo             100      190    420    650    1100    1
/context/update             87        18     65    110     180     0
/batch/query                0          -      -      -       -     -

✅ All endpoints within SLA
```

**Analysis**:
- ✅ Error rate 0.07% (excellent, <0.5% target)
- ✅ All p95 latencies under target
- ⚠️ Cache hit rate 60% (good but could improve)
- ⚠️ One annotation request failed (investigate)
- ✅ Overall system health: PASS

---

## Performance Baselines

### Current Production Targets (2025-11-16)

| Endpoint | p50 | p95 | p99 | Max | Error% |
|----------|-----|-----|-----|-----|--------|
| `/query [voice_command]` | 150ms | 300ms | 500ms | 2000ms | <0.5% |
| `/tts/synthesize [cached]` | 15ms | 30ms | 50ms | 100ms | <0.1% |
| `/tts/synthesize [uncached]` | 500ms | 800ms | 1000ms | 1500ms | <1.0% |
| `/health` | 5ms | 10ms | 20ms | 50ms | 0% |
| `/stats` | 50ms | 100ms | 200ms | 500ms | <0.5% |
| `/annotate-photo` | 200ms | 500ms | 800ms | 1500ms | <1.0% |
| `/context/update` | 20ms | 100ms | 150ms | 300ms | <0.2% |
| `/batch/query` | 500ms | 1000ms | 1500ms | 3000ms | <1.0% |

### Baseline Rationale

**Voice Commands (p95: 300ms)**
- Users expect responsive speech interaction
- >500ms is noticeably slow
- Includes LLM reasoning time (~150ms typical)

**TTS Cached (p95: 30ms)**
- Cached speech should be near-instant
- Shows caching effectiveness
- <100ms is industry standard

**Health Checks (p95: 10ms)**
- Monitoring priority: must be fast
- Load balancers check every 5-10 seconds
- >50ms could cause false alarms

**Context Updates (p95: 100ms)**
- AR tracking requires low latency
- Typical VR/AR standard: <100ms
- Higher latency = apparent jitter in AR

### Interpreting Baseline Violations

**Violation**: p95=500ms (target: 300ms)
```
Ratio: 500/300 = 1.67x (67% over target)

Possible Causes:
1. High load overwhelming system
2. Memory pressure (GC pauses)
3. Disk I/O blocking
4. Network latency
5. Downstream service slowness

Investigation Steps:
1. Check CPU utilization
2. Check memory utilization and GC logs
3. Check database query times
4. Check network latency
5. Check downstream service health
```

---

## Capacity Planning

### Estimating Required Capacity

#### From Load Test Results

If baseline test shows:
- 10 users, 1,500 requests in 5 min
- p95 latency: 280ms

**Calculate RPS**:
```
RPS = 1,500 requests / 300 seconds = 5 RPS
```

**Estimate Max Concurrent Users**:
```
Users = (RPS × Response Time ms) / 1000
Users = (5 × 280) / 1000 = 1.4 ≈ 2 users per instance (conservative)
```

**For 100 Concurrent Users**:
```
Required instances = 100 users / 2 users per instance = 50 instances
```

#### With Headroom (Recommended)

Add 30% headroom for spikes and GC pauses:

```
Required instances = 50 × 1.3 = 65 instances
```

### Capacity Table

| Target RPS | Instances (0% headroom) | Instances (30% headroom) | Notes |
|------------|------------------------|--------------------------|-------|
| 10 RPS | 2 | 3 | Dev/test |
| 50 RPS | 10 | 13 | Small production |
| 100 RPS | 20 | 26 | Medium production |
| 500 RPS | 100 | 130 | Large production |
| 1000 RPS | 200 | 260 | Enterprise |

### Cost Estimation

**Assumptions**:
- AWS EC2 t3.medium (2 CPU, 4GB RAM): $0.041/hour
- Load per instance: ~5 RPS
- Cost: $0.041 × (1000 RPS / 5 RPS) = $8.20/hour

**Annual Cost for 100 RPS**:
```
100 RPS / 5 RPS per instance = 20 instances
20 instances × $0.041/hour × 730 hours/month × 12 months
= ~$7,177/year
```

### Scaling Recommendations

Based on load test:

```
User Load     Auto-Scaling Action          Expected Result
0-10 users    Minimum 3 instances          Baseline SLA met
10-50 users   Linear scaling (1 per 5 RPS) p95 < 400ms
50-100 users  Aggressive scaling           p95 < 500ms
>100 users    Evaluate architecture        Consider caching, CDN, etc.
```

---

## Auto-Scaling Configuration

### Kubernetes HPA (Recommended)

**File**: `/deployment/kubernetes/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voice-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-agent

  minReplicas: 2
  maxReplicas: 20

  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70  # Scale up at 70% CPU

    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80  # Scale up at 80% memory

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100  # Double pods if needed
          periodSeconds: 30
        - type: Pods
          value: 4  # Or add 4 pods max
          periodSeconds: 30

    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50  # Remove 50% of pods
          periodSeconds: 60
```

### Deploy HPA

```bash
# Apply HPA configuration
kubectl apply -f deployment/kubernetes/hpa.yaml

# Verify HPA is created
kubectl get hpa -n hololoom-voice
# NAME                REFERENCE                        TARGETS          MINPODS  MAXPODS  REPLICAS  AGE
# voice-agent-hpa     Deployment/voice-agent           42%/70%, 75%/80%  2        20       3         2m

# Watch HPA scaling decisions
kubectl describe hpa voice-agent-hpa -n hololoom-voice

# Watch pod scaling
kubectl get pods -n hololoom-voice --watch
```

### Monitoring HPA

**View scaling history**:
```bash
# Check HPA events
kubectl describe hpa voice-agent-hpa -n hololoom-voice | grep -A 20 Events

# Example output:
# Events:
# Type     Reason                   Age   From                       Message
# ----     ------                   ---   ----                       -------
# Normal   SuccessfulRescale        2m    horizontal-pod-autoscaler  New size: 4; reason: cpu resource utilization is above target
# Normal   SuccessfulRescale        1m    horizontal-pod-autoscaler  New size: 8; reason: cpu resource utilization is above target
```

**Prometheus metrics** (for dashboarding):
```
kube_hpa_status_current_replicas{hpa="voice-agent-hpa"}
kube_hpa_status_desired_replicas{hpa="voice-agent-hpa"}
hpa_metrics_cpu_utilization{deployment="voice-agent"}
hpa_metrics_memory_utilization{deployment="voice-agent"}
```

---

## Troubleshooting

### Issue: "Connection refused" error

**Symptom**:
```
Error: Failed to connect to http://localhost:8000
```

**Solution**:
```bash
# Verify server is running
curl http://localhost:8000/health

# If failing, start server
cd /home/user/hello-world
PYTHONPATH=. python -m hololoom.server.agentic_api

# If port 8000 in use, use different port
PYTHONPATH=. python -m hololoom.server.agentic_api --port 8001
# Then in Locust: --host=http://localhost:8001
```

### Issue: "p95 latency much higher than baseline"

**Symptom**:
```
Baseline: p95 = 280ms
Now: p95 = 1200ms (4x worse!)
```

**Investigation Steps**:
```bash
# 1. Check server CPU
ps aux | grep hololoom
# Look for %CPU - should be <70% during baseline test

# 2. Check memory
free -m
# Should be stable, not growing continuously

# 3. Check active connections
netstat -an | grep ESTABLISHED | wc -l
# Should be ~10-20 for 10 user test

# 4. Check slow logs
tail -f logs/app.log | grep "latency\|duration"

# 5. Check for GC pauses (if Python)
python -m cProfile -o stats.prof -c "import hololoom.server"

# 6. Check network latency
ping google.com
# Latency should be <10ms for localhost
```

**Common Causes**:
1. Server overloaded (CPU >90%)
   - Solution: Reduce user count or scale up
2. Memory pressure (free memory <20%)
   - Solution: Increase heap/memory limits
3. Disk I/O bottleneck
   - Solution: Check disk usage and switch to SSD if possible
4. Network timeout on dependencies
   - Solution: Check Neo4j, Qdrant, or LLM service connectivity

### Issue: "Cache hit rate too low"

**Symptom**:
```
Cache Hit Rate: 25% (target: 60%+)
```

**Investigation**:
```bash
# 1. Verify cache is enabled
curl http://localhost:8000/stats | grep cache

# 2. Check cache size
# Default should be 1000 entries - may need increase

# 3. Run baseline test longer (10 min instead of 5 min)
PYTHONPATH=. locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 5 \
    --spawn-rate 1 \
    --run-time 10m
# Cache should warm up over time

# 4. Check for cache eviction
tail logs/app.log | grep "cache evict"
```

**Solutions**:
1. Increase cache size in config
2. Run longer tests (cache takes time to warm)
3. Verify repeated queries are truly identical
4. Check cache TTL settings

### Issue: "Out of memory" or "memory keeps growing"

**Symptom**:
```
Memory Usage:
0 min: 200MB
10 min: 350MB
30 min: 600MB
(keeps growing)
```

**Indicates**: Memory leak

**Investigation**:
```bash
# 1. Check for Python memory leaks
python -m tracemalloc hololoom.server.agentic_api

# 2. Monitor with 'top' command
top -p $(pgrep -f hololoom)
# VIRT (virtual) should stabilize
# RES (resident) should not keep growing

# 3. Check for unclosed resources
grep -n "open\|socket\|connection" hololoom/server/*.py

# 4. Monitor by endpoint
# Which endpoints cause memory growth?
PYTHONPATH=. locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 5 \
    --run-time 10m \
    --loglevel INFO
```

**Common Causes**:
1. Unclosed database connections
2. Unreleased model/embeddings
3. Growing list/cache without bounds
4. Memory leaks in dependencies

**Solution**:
```bash
# Restart server periodically
docker restart hololoom-voice-agent

# Or enable memory limits
docker run --memory 512m hololoom-voice-agent
```

### Issue: "Error rate > 2% suddenly"

**Symptom**:
```
First 5 min: 0.1% errors
Next 5 min: 5.2% errors (spike!)
```

**Indicates**: Cascading failure or resource exhaustion

**Immediate Steps**:
```bash
# 1. Check server logs for errors
tail -f logs/app.log | grep ERROR

# 2. Check downstream services
curl http://neo4j:7687/health  # Graph DB
curl http://qdrant:6333/health  # Vector DB
curl http://ollama:11434/health  # LLM

# 3. Check queue depth
curl http://localhost:8000/stats | grep queue

# 4. Stop load test and restart server
# Kill Locust: Ctrl+C
# Restart server: just restart the container
```

**Root Causes**:
1. Downstream service down (Neo4j, Qdrant, Ollama)
2. Queue overflow (too many pending requests)
3. Connection pool exhausted
4. Timeout settings too strict

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Baseline test passing (p95 <300ms, errors <0.5%)
- [ ] Stress test completed (validates load handling)
- [ ] Spike test completed (validates auto-scaling)
- [ ] Endurance test completed (no memory leaks)
- [ ] HPA configured and tested
- [ ] Alerting configured (CloudWatch, Datadog, etc.)
- [ ] Logging configured (ELK, Splunk, etc.)
- [ ] Database backups enabled
- [ ] Rollback plan documented
- [ ] On-call team trained

### Deployment Steps

```bash
# 1. Run final baseline test against staging
PYTHONPATH=. locust -f tests/load/locustfile.py \
    --host=http://staging-server:8000 \
    --users 10 \
    --run-time 5m \
    --headless

# 2. Deploy to production
kubectl apply -f deployment/kubernetes/

# 3. Wait for HPA to initialize
kubectl get hpa -n hololoom-voice --watch

# 4. Run smoke test
for i in {1..10}; do
    curl http://production:8000/health
    sleep 1
done

# 5. Monitor for 1 hour
kubectl logs -f deployment/voice-agent -n hololoom-voice

# 6. Set up continuous monitoring
# See monitoring section below
```

### Continuous Monitoring

**Recommended Alerts**:

| Alert | Condition | Action |
|-------|-----------|--------|
| **High Latency** | p95 > 500ms for 5 min | Page on-call engineer |
| **High Error Rate** | >2% for 5 min | Page on-call engineer |
| **Cache Degradation** | Hit rate <40% | Investigate cache config |
| **Memory Growth** | Growing >50MB/min | Investigate leaks |
| **CPU Spike** | >90% for 10 min | Trigger HPA or scale manually |
| **Service Down** | 5 consecutive failures | Page on-call + executive |

**Prometheus Rules**:
```yaml
groups:
  - name: hololoom-voice-agent
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, voice_agent_request_duration_ms) > 500
        for: 5m
        annotations:
          summary: "Voice Agent p95 latency > 500ms"

      - alert: HighErrorRate
        expr: rate(voice_agent_requests_failed[5m]) > 0.02
        for: 5m
        annotations:
          summary: "Voice Agent error rate > 2%"

      - alert: MemoryLeak
        expr: rate(container_memory_rss[5m]) > 50 * 1024 * 1024
        for: 10m
        annotations:
          summary: "Possible memory leak detected"
```

### Rollback Plan

```bash
# If production deployment fails:

# Option 1: Rollback to previous version
kubectl rollout undo deployment/voice-agent -n hololoom-voice

# Option 2: Scale down to previous capacity
kubectl scale deployment/voice-agent -n hololoom-voice --replicas=3

# Option 3: Kill new version, restart old
kubectl delete deployment/voice-agent -n hololoom-voice
kubectl apply -f deployment/kubernetes/voice-agent-previous.yaml

# Verify rollback
kubectl get deployment voice-agent -n hololoom-voice
kubectl logs deployment/voice-agent -n hololoom-voice
```

---

## Contact & Support

For issues or questions:

1. **Documentation**: See [LOAD_TESTING_README.md](LOAD_TESTING_README.md) (this file)
2. **Performance Baselines**: [tests/load/benchmarks.py](benchmarks.py)
3. **Locust Documentation**: https://docs.locust.io/
4. **HoloLoom Docs**: [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](../../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)

---

**Document Last Updated**: 2025-11-16
**Locust Version Tested**: 2.15.1
**HoloLoom Version**: Phase 5+ (Universal Grammar + Compositional Cache)
