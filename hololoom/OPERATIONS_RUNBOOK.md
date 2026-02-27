# HoloLoom Operations Runbook

**Version**: 1.0
**Date**: 2025-11-13
**Audience**: Production Operations, SRE, DevOps

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Incident Response](#incident-response)
6. [Common Issues](#common-issues)
7. [Maintenance](#maintenance)
8. [Scaling](#scaling)
9. [Security](#security)
10. [Disaster Recovery](#disaster-recovery)

---

## System Overview

### What is HoloLoom?

HoloLoom is a neural decision-making system that combines:
- Multi-scale embeddings (Matryoshka representations)
- Knowledge graph memory with spectral features
- Unified policy engine with Thompson Sampling
- Production hardening (monitoring, circuit breakers, rate limiting)

### Key Components

- **WeavingOrchestrator**: Main orchestration engine (9-step weaving cycle)
- **Context Module**: Production hardening (monitoring, circuit breakers, rate limiting, health checks)
- **Memory Backends**: NetworkX (dev), Neo4j + Qdrant (prod)
- **Policy Engine**: Neural decision-making with Thompson Sampling

### Performance Characteristics

- **Latency**: 50-300ms per query (mode-dependent)
- **Throughput**: 100-1000 QPS (configurable)
- **Memory**: 2GB typical, 4GB max
- **CPU**: 1-2 cores typical, 4 cores max

---

## Architecture

### Production Stack

```
┌────────────────────────────────────────────────────────┐
│               Load Balancer (Nginx/AWS ALB)           │
│                Health Check: /health                   │
└────────────────────┬───────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│  HoloLoom API  │       │  HoloLoom API  │
│    (Pod 1)     │  ...  │    (Pod N)     │
│  Port: 8080    │       │  Port: 8080    │
└───────┬────────┘       └───────┬────────┘
        │                         │
        └─────────┬───────────────┘
                  │
     ┌────────────┴──────────────────────┐
     │                                   │
┌────▼──────┐  ┌──────────┐  ┌─────────▼─────┐
│   Neo4j   │  │ Qdrant   │  │ Prometheus    │
│  (Graph)  │  │ (Vector) │  │  (Metrics)    │
└───────────┘  └──────────┘  └───────────────┘
```

### Component Dependencies

| Component | Required | Optional | Graceful Fallback |
|-----------|----------|----------|-------------------|
| NetworkX | ✅ Yes | - | No fallback (core) |
| Neo4j | ❌ No | ✅ Yes | Falls back to NetworkX |
| Qdrant | ❌ No | ✅ Yes | Falls back to NetworkX |
| psutil | ❌ No | ✅ Yes | Returns 0 for resource metrics |
| spaCy | ❌ No | ✅ Yes | Falls back to regex motifs |

**Key Principle**: HoloLoom degrades gracefully. Missing optional dependencies reduce functionality but don't crash the system.

---

## Deployment

### Environment Variables

```bash
# Required
CONTEXT_ENV=production  # or "staging", "development"

# Optional (override defaults)
CONTEXT_LOG_LEVEL=WARNING
CONTEXT_MAX_QPS=1000
CONTEXT_MAX_MEMORY_MB=2048
CONTEXT_METRICS_EXPORT=prometheus

# Backend configuration (if using Neo4j/Qdrant)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<secret>
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  hololoom:
    build: .
    environment:
      - CONTEXT_ENV=production
    ports:
      - "8080:8080"  # API
      - "9090:9090"  # Prometheus metrics (if enabled)
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2048M
        reservations:
          cpus: '1'
          memory: 1024M
      replicas: 3
    restart: unless-stopped

  neo4j:
    image: neo4j:5.12
    environment:
      - NEO4J_AUTH=neo4j/production-password
    ports:
      - "7687:7687"
      - "7474:7474"
    volumes:
      - neo4j-data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

volumes:
  neo4j-data:
  qdrant-data:
  prometheus-data:
```

### Kubernetes Deployment

See `KUBERNETES_DEPLOYMENT.md` for complete manifests.

**Quick Start**:
```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n hololoom
kubectl logs -n hololoom -l app=hololoom --tail=50

# Test health
kubectl port-forward -n hololoom svc/hololoom 8080:8080
curl http://localhost:8080/health
```

---

## Monitoring

### Health Check Endpoint

**URL**: `GET /health`

**Response (Healthy)**:
```json
{
  "healthy": true,
  "status": "healthy",
  "checks": {
    "overall": {
      "name": "overall",
      "healthy": true,
      "status": "healthy",
      "message": "System healthy",
      "details": {
        "error_rate": 0.025,
        "latency_p95": 180.0,
        "qps": 16.67
      }
    },
    "backends": {"healthy": true, "status": "healthy"},
    "learning": {"healthy": true, "status": "healthy"},
    "resources": {"healthy": true, "status": "healthy"}
  },
  "timestamp": 1698595200.0
}
```

**Response (Unhealthy)** - HTTP 503:
```json
{
  "healthy": false,
  "status": "degraded",
  "checks": {
    "overall": {
      "healthy": false,
      "status": "unhealthy",
      "message": "High error rate: 25.0%"
    },
    "backends": {
      "healthy": false,
      "status": "degraded",
      "message": "Circuit breakers open: neo4j"
    }
  }
}
```

### Metrics Endpoint

**URL**: `GET /metrics`

**Response** (Prometheus text format):
```
# HELP hololoom_queries_total Total number of queries processed
# TYPE hololoom_queries_total counter
hololoom_queries_total 1523

# HELP hololoom_qps Queries per second
# TYPE hololoom_qps gauge
hololoom_qps 16.7

# HELP hololoom_latency_p95 95th percentile latency (ms)
# TYPE hololoom_latency_p95 gauge
hololoom_latency_p95 180.5

# HELP hololoom_error_rate Error rate (0.0-1.0)
# TYPE hololoom_error_rate gauge
hololoom_error_rate 0.025

# HELP hololoom_cache_hit_rate Cache hit rate (0.0-1.0)
# TYPE hololoom_cache_hit_rate gauge
hololoom_cache_hit_rate 0.75
```

### Key Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| `hololoom_error_rate` | >10% | **CRITICAL**: Check logs, backends |
| `hololoom_latency_p95` | >1000ms | **WARNING**: Investigate performance |
| `hololoom_qps` | >900 (of 1000 limit) | **WARNING**: Scale up or reject traffic |
| `hololoom_cache_hit_rate` | <50% | **INFO**: Review query patterns |
| `hololoom_memory_mb` | >1600MB (of 2048) | **WARNING**: Restart or scale |
| `hololoom_circuit_breakers_open` | >0 | **CRITICAL**: Backend failure |

### Alerting Rules (Prometheus)

```yaml
groups:
  - name: hololoom_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: hololoom_error_rate > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "HoloLoom high error rate"
          description: "Error rate is {{ $value }} (>10%)"

      - alert: HighLatency
        expr: hololoom_latency_p95 > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "HoloLoom high latency"
          description: "P95 latency is {{ $value }}ms (>1000ms)"

      - alert: CircuitBreakerOpen
        expr: hololoom_circuit_breakers_open > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker open"
          description: "{{ $value }} circuit breakers are open"

      - alert: HighMemoryUsage
        expr: hololoom_memory_mb > 1600
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}MB (>1600MB)"
```

### Grafana Dashboard

Import dashboard JSON: `GRAFANA_DASHBOARD.json`

**Key Panels**:
1. QPS (queries per second) - line graph
2. Error rate - line graph with alert threshold
3. Latency percentiles (p50, p90, p95, p99) - line graph
4. Circuit breaker states - status panel
5. Cache hit rate - gauge
6. Memory/CPU usage - line graph
7. Top errors - table
8. Query distribution by tool - pie chart

---

## Incident Response

### Severity Levels

| Level | Definition | Response Time | Example |
|-------|------------|---------------|---------|
| **P0** | Service down | 15 minutes | All requests failing |
| **P1** | Degraded service | 1 hour | High error rate (>10%) |
| **P2** | Performance issue | 4 hours | High latency (>1s) |
| **P3** | Minor issue | 1 business day | Cache hit rate low |

### P0: Service Down

**Symptoms**:
- Health check returning 503
- All requests failing
- No responses from pods

**Immediate Actions**:
1. Check pod status: `kubectl get pods -n hololoom`
2. Check recent logs: `kubectl logs -n hololoom -l app=hololoom --tail=100`
3. Check backend connectivity (Neo4j, Qdrant)
4. Check resource limits (CPU, memory)

**Resolution Steps**:
1. **If pods are CrashLooping**:
   ```bash
   # Check crash reason
   kubectl describe pod -n hololoom <pod-name>

   # Check resource limits
   kubectl top pods -n hololoom

   # If OOMKilled, increase memory limit
   kubectl edit deployment -n hololoom hololoom
   # Increase resources.limits.memory
   ```

2. **If backends are down**:
   ```bash
   # Check Neo4j
   kubectl logs -n hololoom neo4j-0

   # Check Qdrant
   kubectl logs -n hololoom qdrant-0

   # Restart if needed
   kubectl rollout restart statefulset -n hololoom neo4j
   ```

3. **If configuration issue**:
   ```bash
   # Check config
   kubectl get configmap -n hololoom hololoom-config -o yaml

   # Fix and restart
   kubectl edit configmap -n hololoom hololoom-config
   kubectl rollout restart deployment -n hololoom hololoom
   ```

### P1: High Error Rate

**Symptoms**:
- Error rate >10%
- Increased 5xx responses
- Circuit breakers opening

**Immediate Actions**:
1. Check health endpoint: `curl http://hololoom/health`
2. Check circuit breaker status
3. Review recent error logs

**Resolution Steps**:
1. **Identify error type**:
   ```bash
   # View recent errors
   kubectl logs -n hololoom -l app=hololoom --tail=500 | grep ERROR

   # Common error types:
   # - BackendError: Neo4j/Qdrant down
   # - TimeoutError: Slow backends
   # - RateLimitExceededError: Too much traffic
   ```

2. **If BackendError (circuit breakers open)**:
   - Check backend health
   - Restart backends if needed
   - Circuit breakers will auto-recover after 120s (production)

3. **If TimeoutError**:
   - Check backend latency
   - Scale up backends if needed
   - Increase timeout if appropriate

### P2: High Latency

**Symptoms**:
- P95 latency >1000ms
- Slow response times
- User complaints

**Resolution Steps**:
1. **Check resource usage**:
   ```bash
   kubectl top pods -n hololoom
   # If CPU >80% or Memory >80%, scale up
   ```

2. **Check cache hit rate**:
   - If <50%, queries are not being cached
   - Check query patterns (are they unique?)

3. **Check backend latency**:
   - Neo4j query time
   - Qdrant search time
   - If high, scale backends

4. **Scale horizontally**:
   ```bash
   kubectl scale deployment -n hololoom hololoom --replicas=6
   ```

---

## Common Issues

### Issue: Circuit Breaker Stuck Open

**Symptoms**: Backend shows as "degraded" in health check, circuit breaker won't close.

**Cause**: Backend is down or consistently failing.

**Resolution**:
1. Check backend health: `curl http://neo4j:7474` or `curl http://qdrant:6333`
2. Restart backend if needed
3. Wait for recovery timeout (120s in production)
4. Circuit breaker will automatically try HALF_OPEN
5. If successful, circuit closes automatically

**Manual Reset** (last resort):
```python
# Connect to pod
kubectl exec -it -n hololoom hololoom-pod-xyz -- python

# Reset circuit breaker
from hololoom.weaving_orchestrator import orchestrator
orchestrator.breaker_registry.reset_all()
```

### Issue: Rate Limit Rejections

**Symptoms**: HTTP 429 responses, "Rate limit exceeded" errors.

**Cause**: Too many requests exceeding configured QPS limit.

**Resolution**:
1. **Temporary**: Increase rate limit
   ```python
   # Edit deployment config
   CONTEXT_MAX_QPS=2000  # Increase from 1000
   ```

2. **Permanent**: Scale horizontally (more pods = more capacity)
   ```bash
   kubectl scale deployment -n hololoom hololoom --replicas=5
   ```

3. **Client-side**: Implement backoff/retry on 429 responses

### Issue: High Memory Usage

**Symptoms**: Memory >1600MB, OOMKilled pods, slow performance.

**Cause**: Large cache, memory leak, or high query volume.

**Resolution**:
1. **Check cache size**:
   ```python
   metrics = orchestrator.get_metrics()
   print(f"Cache size: {metrics['performance'].get('cache_size', 0)}")
   ```

2. **Clear cache** (if too large):
   ```python
   orchestrator.query_cache.clear()
   ```

3. **Increase memory limit**:
   ```yaml
   # k8s/deployment.yaml
   resources:
     limits:
       memory: 4096Mi  # Increase from 2048Mi
   ```

4. **Restart pod** (clears memory):
   ```bash
   kubectl delete pod -n hololoom <pod-name>
   ```

### Issue: Low Cache Hit Rate

**Symptoms**: Cache hit rate <50%, high latency.

**Cause**: Unique queries, cache not warming up, cache too small.

**Resolution**:
1. **Analyze query patterns**: Are queries truly unique?
2. **Increase cache size**:
   ```python
   # In config
   config.resource.max_cache_size = 20000  # Increase from 10000
   ```
3. **Warm up cache**: Send common queries on startup

---

## Maintenance

### Routine Maintenance

| Task | Frequency | Owner |
|------|-----------|-------|
| Review error logs | Daily | Ops |
| Check resource usage | Daily | Ops |
| Review alerting rules | Weekly | SRE |
| Capacity planning | Monthly | SRE |
| Security patches | As needed | DevOps |
| Dependency updates | Quarterly | Dev |

### Planned Downtime

**Steps for Rolling Update**:
```bash
# 1. Update container image
kubectl set image deployment/hololoom -n hololoom \
  hololoom=hololoom:v1.2.0

# 2. Monitor rollout
kubectl rollout status deployment/hololoom -n hololoom

# 3. If issues, rollback
kubectl rollout undo deployment/hololoom -n hololoom
```

**Zero-Downtime Deployment**:
- Use rolling update strategy (default)
- Ensure minReadySeconds=30
- Ensure readinessProbe is configured
- Load balancer automatically routes to healthy pods

### Log Rotation

**Docker/Kubernetes**: Logs are automatically rotated by container runtime.

**Manual Log Management**:
```bash
# View recent logs
kubectl logs -n hololoom -l app=hololoom --tail=1000

# Save logs for analysis
kubectl logs -n hololoom -l app=hololoom > hololoom-logs.txt

# Follow logs in real-time
kubectl logs -n hololoom -l app=hololoom -f
```

---

## Scaling

### Horizontal Scaling

**When to Scale Up**:
- QPS >90% of limit (e.g., >900 of 1000 QPS)
- CPU >80%
- Memory >80%
- P95 latency >500ms

**How to Scale**:
```bash
# Scale to 6 replicas
kubectl scale deployment -n hololoom hololoom --replicas=6

# Or enable HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment -n hololoom hololoom \
  --min=3 --max=10 --cpu-percent=70
```

### Vertical Scaling

**When to Scale Up**:
- Pods OOMKilled
- CPU throttling
- Memory consistently >80%

**How to Scale**:
```yaml
# Edit deployment
resources:
  limits:
    cpus: '4'      # Increase from 2
    memory: 4096Mi # Increase from 2048Mi
  requests:
    cpus: '2'      # Increase from 1
    memory: 2048Mi # Increase from 1024Mi
```

### Backend Scaling

**Neo4j**:
- Scale vertically (more CPU/memory)
- Consider Neo4j Causal Cluster for HA

**Qdrant**:
- Scale horizontally (sharding)
- Scale vertically (more memory for vectors)

---

## Security

### Access Control

- **Network Policies**: Restrict pod-to-pod communication
- **RBAC**: Limit kubectl access by team
- **Secrets**: Use Kubernetes Secrets for passwords

**Example Network Policy**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hololoom-network-policy
  namespace: hololoom
spec:
  podSelector:
    matchLabels:
      app: hololoom
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: neo4j
    ports:
    - protocol: TCP
      port: 7687
```

### Secrets Management

**Store sensitive data in Kubernetes Secrets**:
```bash
kubectl create secret generic hololoom-secrets -n hololoom \
  --from-literal=neo4j-password=<password> \
  --from-literal=qdrant-api-key=<key>
```

**Reference in deployment**:
```yaml
env:
- name: NEO4J_PASSWORD
  valueFrom:
    secretKeyRef:
      name: hololoom-secrets
      key: neo4j-password
```

### TLS/HTTPS

**Enable TLS at ingress level**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hololoom-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - hololoom.example.com
    secretName: hololoom-tls
  rules:
  - host: hololoom.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hololoom
            port:
              number: 8080
```

---

## Disaster Recovery

### Backup Strategy

**What to Backup**:
1. Neo4j database (graph data)
2. Qdrant vectors
3. Configuration files
4. Kubernetes manifests

**Backup Schedule**:
- **Daily**: Incremental backups
- **Weekly**: Full backups
- **Retention**: 30 days

**Neo4j Backup**:
```bash
# Backup Neo4j database
kubectl exec -n hololoom neo4j-0 -- \
  neo4j-admin dump --database=neo4j --to=/backups/neo4j-$(date +%Y%m%d).dump

# Copy to persistent storage
kubectl cp hololoom/neo4j-0:/backups/neo4j-$(date +%Y%m%d).dump \
  ./backups/
```

**Qdrant Backup**:
```bash
# Qdrant snapshots
curl -X POST http://qdrant:6333/collections/hololoom/snapshots
```

### Restore Procedure

**Neo4j Restore**:
```bash
# Stop Neo4j
kubectl scale statefulset -n hololoom neo4j --replicas=0

# Restore dump
kubectl cp ./backups/neo4j-20251113.dump hololoom/neo4j-0:/backups/
kubectl exec -n hololoom neo4j-0 -- \
  neo4j-admin load --from=/backups/neo4j-20251113.dump --database=neo4j --force

# Start Neo4j
kubectl scale statefulset -n hololoom neo4j --replicas=1
```

### Recovery Time Objective (RTO)

- **RTO**: 1 hour (time to restore service)
- **RPO**: 24 hours (data loss tolerance)

---

## Contact Information

**On-Call Rotation**: See PagerDuty schedule

**Escalation Path**:
1. On-call engineer (PagerDuty)
2. SRE team lead
3. Engineering manager
4. CTO

**Support Channels**:
- **Slack**: #hololoom-ops
- **Email**: ops@example.com
- **PagerDuty**: hololoom-oncall

---

## Appendix

### Useful Commands

```bash
# Quick health check
curl http://hololoom:8080/health | jq '.healthy'

# Get metrics
curl http://hololoom:8080/metrics | grep hololoom_

# View logs
kubectl logs -n hololoom -l app=hololoom --tail=100

# Restart deployment
kubectl rollout restart deployment -n hololoom hololoom

# Scale up
kubectl scale deployment -n hololoom hololoom --replicas=5

# Check resource usage
kubectl top pods -n hololoom

# Port forward for local testing
kubectl port-forward -n hololoom svc/hololoom 8080:8080
```

### Configuration Reference

See [PRODUCTION_QUICK_START.md](PRODUCTION_QUICK_START.md) for complete configuration reference.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
**Next Review**: 2025-12-13
