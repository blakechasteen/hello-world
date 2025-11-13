# HoloLoom Kubernetes Deployment

**Part 5: Production Hardening - Day 25**

Production-ready Kubernetes manifests for deploying HoloLoom with full production hardening features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Manifest Overview](#manifest-overview)
4. [Deployment Steps](#deployment-steps)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

Deploy HoloLoom to Kubernetes in 5 minutes:

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Configure secrets (IMPORTANT: Change default passwords!)
kubectl create secret generic hololoom-secrets \
  --from-literal=neo4j.user=neo4j \
  --from-literal=neo4j.password=YOUR_STRONG_PASSWORD \
  --namespace=hololoom

# 3. Deploy all manifests
kubectl apply -k .

# 4. Verify deployment
kubectl get pods -n hololoom
kubectl get svc -n hololoom
kubectl get ingress -n hololoom

# 5. Check health
kubectl port-forward -n hololoom svc/hololoom-api 8080:8080
curl http://localhost:8080/health
```

---

## Prerequisites

### Required

- **Kubernetes cluster**: v1.24+ (tested on v1.28)
- **kubectl**: v1.24+
- **kustomize**: v4.0+ (included in kubectl 1.14+)

### Optional (Recommended for Production)

- **cert-manager**: For TLS certificate management
  ```bash
  kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
  ```

- **Prometheus Operator**: For monitoring and alerting
  ```bash
  helm install prometheus-operator prometheus-community/kube-prometheus-stack \
    --namespace monitoring --create-namespace
  ```

- **Nginx Ingress Controller**: For ingress
  ```bash
  helm install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx --create-namespace
  ```

- **Metrics Server**: For HPA (Horizontal Pod Autoscaler)
  ```bash
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  ```

---

## Manifest Overview

### Core Manifests

| File | Description | Purpose |
|------|-------------|---------|
| **namespace.yaml** | Namespace definition | Isolate HoloLoom resources |
| **deployment.yaml** | Deployment + RBAC | Main application deployment |
| **service.yaml** | Service (ClusterIP) | Internal service discovery |
| **ingress.yaml** | Ingress + TLS | External access with SSL |
| **configmap.yaml** | Configuration | Non-sensitive config |
| **secret.yaml** | Secrets | Sensitive credentials |
| **hpa.yaml** | HorizontalPodAutoscaler | Auto-scaling based on metrics |
| **networkpolicy.yaml** | NetworkPolicy | Network security isolation |
| **servicemonitor.yaml** | ServiceMonitor + PrometheusRule | Monitoring and alerting |
| **kustomization.yaml** | Kustomize config | Orchestrate deployment |

### Features Enabled

- ✅ **Production hardening**: Rate limiting, circuit breakers, health checks
- ✅ **Auto-scaling**: HPA based on CPU, memory, QPS, latency
- ✅ **High availability**: 3 replicas, PodDisruptionBudget, anti-affinity
- ✅ **Security**: NetworkPolicy, RBAC, non-root user, read-only filesystem
- ✅ **Monitoring**: Prometheus metrics, Grafana dashboards, alerting
- ✅ **Zero-downtime deployments**: Rolling updates with health checks
- ✅ **TLS/SSL**: Automatic certificate management with cert-manager
- ✅ **Resource limits**: CPU/memory requests and limits

---

## Deployment Steps

### 1. Prepare Cluster

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Verify namespace
kubectl get namespace hololoom
```

### 2. Configure Secrets

**⚠️ IMPORTANT**: Change default passwords before deploying!

```bash
# Option 1: Create from command line
kubectl create secret generic hololoom-secrets \
  --from-literal=neo4j.user=neo4j \
  --from-literal=neo4j.password=YOUR_STRONG_PASSWORD \
  --from-literal=api.key=YOUR_API_KEY \
  --from-literal=jwt.secret=YOUR_JWT_SECRET \
  --namespace=hololoom

# Option 2: Create from file
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: hololoom-secrets
  namespace: hololoom
type: Opaque
stringData:
  neo4j.user: neo4j
  neo4j.password: YOUR_STRONG_PASSWORD
  api.key: YOUR_API_KEY
  jwt.secret: YOUR_JWT_SECRET
EOF

# Option 3: Use Sealed Secrets (recommended for GitOps)
kubeseal --format yaml <secret.yaml >sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

### 3. Deploy Dependencies (Neo4j, Qdrant)

**Neo4j**:
```bash
helm install neo4j neo4j/neo4j \
  --namespace hololoom \
  --set neo4j.password=YOUR_NEO4J_PASSWORD \
  --set volumes.data.mode=defaultStorageClass \
  --set volumes.data.defaultStorageClass.requests.storage=10Gi
```

**Qdrant**:
```bash
helm install qdrant qdrant/qdrant \
  --namespace hololoom \
  --set persistence.size=10Gi
```

### 4. Configure Ingress Domain

**Edit ingress.yaml**:
```yaml
spec:
  tls:
    - hosts:
        - api.yourdomain.com  # Change this
      secretName: hololoom-api-tls
  rules:
    - host: api.yourdomain.com  # Change this
```

### 5. Deploy HoloLoom

```bash
# Option 1: Deploy with kustomize (recommended)
kubectl apply -k .

# Option 2: Deploy individual manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
kubectl apply -f networkpolicy.yaml
kubectl apply -f servicemonitor.yaml

# Wait for deployment
kubectl rollout status deployment/hololoom-api -n hololoom --timeout=5m
```

### 6. Verify Deployment

```bash
# Check pods
kubectl get pods -n hololoom

# Expected output:
# NAME                           READY   STATUS    RESTARTS   AGE
# hololoom-api-xxxxxxxxxx-xxxxx  1/1     Running   0          2m
# hololoom-api-xxxxxxxxxx-xxxxx  1/1     Running   0          2m
# hololoom-api-xxxxxxxxxx-xxxxx  1/1     Running   0          2m

# Check services
kubectl get svc -n hololoom

# Check ingress
kubectl get ingress -n hololoom

# Check logs
kubectl logs -n hololoom -l app=hololoom-api --tail=50 -f
```

### 7. Test Endpoints

```bash
# Port-forward for testing
kubectl port-forward -n hololoom svc/hololoom-api 8080:8080

# Health check
curl http://localhost:8080/health

# Expected response:
# {"healthy": true, "status": "healthy", "checks": {...}}

# Query endpoint
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Thompson Sampling?", "context": {}}'

# Metrics endpoint
curl http://localhost:8080/metrics
```

---

## Configuration

### Environment-Specific Configurations

Create overlays for different environments:

```bash
# Directory structure
kubernetes/
  base/
    - *.yaml
  overlays/
    production/
      - kustomization.yaml
      - patches/
    staging/
      - kustomization.yaml
      - patches/
    development/
      - kustomization.yaml
      - patches/
```

**Production overlay** (overlays/production/kustomization.yaml):
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: hololoom-prod

resources:
  - ../../base

replicas:
  - name: hololoom-api
    count: 10  # 10 replicas for production

images:
  - name: hololoom/api
    newTag: "1.0.0"

patchesStrategicMerge:
  - patches/resource-limits.yaml
  - patches/rate-limits.yaml
```

**Staging overlay** (overlays/staging/kustomization.yaml):
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: hololoom-staging

resources:
  - ../../base

replicas:
  - name: hololoom-api
    count: 3  # 3 replicas for staging

images:
  - name: hololoom/api
    newTag: "staging-latest"
```

**Deploy specific environment**:
```bash
# Production
kubectl apply -k overlays/production

# Staging
kubectl apply -k overlays/staging

# Development
kubectl apply -k overlays/development
```

### Resource Tuning

**Low-resource environment** (small cluster):
```yaml
# deployment.yaml
resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 2Gi

# hpa.yaml
minReplicas: 1
maxReplicas: 3
```

**High-resource environment** (production):
```yaml
# deployment.yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi

# hpa.yaml
minReplicas: 5
maxReplicas: 20
```

---

## Monitoring

### Prometheus Metrics

HoloLoom exports Prometheus metrics on port 9090:

**Key Metrics**:
- `hololoom_query_total`: Total queries processed (counter)
- `hololoom_query_latency_seconds`: Query latency histogram
- `hololoom_cache_hit_rate`: Cache hit rate (gauge)
- `hololoom_circuit_breaker_state`: Circuit breaker states (gauge)
- `hololoom_rate_limit_rejected_total`: Rate limit rejections (counter)
- `hololoom_memory_usage_bytes`: Memory usage (gauge)

**Query Examples**:
```promql
# QPS (queries per second)
rate(hololoom_query_total[5m])

# P95 latency
histogram_quantile(0.95, rate(hololoom_query_latency_seconds_bucket[5m]))

# Error rate
rate(hololoom_query_total{status="error"}[5m]) / rate(hololoom_query_total[5m])

# Cache hit rate
rate(hololoom_cache_hits_total[5m]) / rate(hololoom_cache_requests_total[5m])
```

### Grafana Dashboard

Import the provided Grafana dashboard:

```bash
# Get dashboard JSON
kubectl get configmap hololoom-dashboard -n hololoom -o jsonpath='{.data.hololoom-overview\.json}'

# Or import via Grafana UI:
# 1. Open Grafana
# 2. Go to Dashboards → Import
# 3. Paste dashboard JSON
# 4. Select Prometheus data source
```

**Dashboard Panels**:
1. Query Rate (QPS)
2. Query Latency (p50, p95, p99)
3. Error Rate
4. Cache Hit Rate
5. Memory Usage
6. CPU Usage
7. Circuit Breaker States
8. Rate Limit Rejections

### Alerting

Alerts are defined in **servicemonitor.yaml** (PrometheusRule).

**Active Alerts**:
- `HoloLoomHighLatency`: P95 latency > 1s for 5m
- `HoloLoomHighErrorRate`: Error rate > 10% for 5m
- `HoloLoomLowCacheHitRate`: Cache hit rate < 30% for 10m
- `HoloLoomServiceDown`: Service unreachable for 1m
- `HoloLoomHighMemory`: Memory usage > 85% for 5m

**Configure Alertmanager** (alerts/alertmanager-config.yaml):
```yaml
route:
  receiver: hololoom-team
  routes:
    - match:
        severity: critical
      receiver: pagerduty
    - match:
        severity: warning
      receiver: slack

receivers:
  - name: hololoom-team
    email_configs:
      - to: team@hololoom.ai
  - name: pagerduty
    pagerduty_configs:
      - service_key: YOUR_PAGERDUTY_KEY
  - name: slack
    slack_configs:
      - api_url: YOUR_SLACK_WEBHOOK
        channel: '#alerts'
```

---

## Scaling

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment hololoom-api --replicas=5 -n hololoom

# Verify
kubectl get pods -n hololoom
```

### Horizontal Pod Autoscaling (HPA)

HPA automatically scales based on metrics:

```bash
# Check HPA status
kubectl get hpa -n hololoom

# Example output:
# NAME           REFERENCE                 TARGETS         MINPODS   MAXPODS   REPLICAS
# hololoom-api   Deployment/hololoom-api   50%/70%,60%/80% 3         10        4

# Describe HPA (detailed view)
kubectl describe hpa hololoom-api -n hololoom
```

**Scaling Triggers**:
- **CPU >70%**: Scale up
- **Memory >80%**: Scale up
- **QPS >80 per pod**: Scale up
- **Latency p95 >800ms**: Scale up

**Scaling Behavior**:
- **Scale up**: Aggressive (double capacity in 30s)
- **Scale down**: Conservative (50% reduction after 5min stabilization)

### Vertical Pod Autoscaling (VPA)

VPA automatically adjusts resource requests/limits:

```bash
# Check VPA status
kubectl get vpa -n hololoom

# VPA recommendations
kubectl describe vpa hololoom-api -n hololoom
```

**⚠️ Warning**: Do NOT use VPA and HPA together on the same metrics (CPU/memory).

---

## Troubleshooting

### Pods Not Starting

**Check pod status**:
```bash
kubectl get pods -n hololoom
kubectl describe pod <pod-name> -n hololoom
```

**Common issues**:
1. **ImagePullBackOff**: Image doesn't exist or auth failed
   ```bash
   # Check image
   kubectl describe pod <pod-name> -n hololoom | grep -A 5 "Events:"

   # Fix: Update image tag in deployment.yaml or create imagePullSecret
   ```

2. **CrashLoopBackOff**: Container crashing on startup
   ```bash
   # Check logs
   kubectl logs <pod-name> -n hololoom --previous

   # Common causes:
   # - Missing secrets
   # - Neo4j/Qdrant not available
   # - Invalid configuration
   ```

3. **Pending**: Not enough resources
   ```bash
   # Check events
   kubectl describe pod <pod-name> -n hololoom

   # Fix: Reduce resource requests or add nodes
   ```

### Service Not Reachable

**Check service and endpoints**:
```bash
# Service
kubectl get svc hololoom-api -n hololoom

# Endpoints (should show pod IPs)
kubectl get endpoints hololoom-api -n hololoom

# If no endpoints, check pod labels
kubectl get pods -n hololoom --show-labels
```

### Ingress Not Working

**Check ingress**:
```bash
# Ingress resource
kubectl get ingress -n hololoom
kubectl describe ingress hololoom-api -n hololoom

# Ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

**Common issues**:
1. **Certificate not ready**: Wait for cert-manager
   ```bash
   kubectl get certificate -n hololoom
   kubectl describe certificate hololoom-api-tls -n hololoom
   ```

2. **DNS not resolving**: Update DNS records
   ```bash
   # Check DNS
   nslookup api.yourdomain.com

   # Should point to ingress controller's external IP
   kubectl get svc -n ingress-nginx
   ```

### High Latency

**Check metrics**:
```bash
# Port-forward metrics
kubectl port-forward -n hololoom svc/hololoom-api 9090:9090

# View metrics
curl http://localhost:9090/metrics | grep hololoom_query_latency
```

**Debugging steps**:
1. Check if backends (Neo4j, Qdrant) are slow
2. Check circuit breaker states
3. Enable debug logging
4. Profile with `py-spy` or `cProfile`

### Memory Issues

**Check memory usage**:
```bash
# Top pods by memory
kubectl top pods -n hololoom --sort-by=memory

# Describe pod
kubectl describe pod <pod-name> -n hololoom | grep -A 5 "Limits"
```

**Solutions**:
1. Increase memory limits in deployment.yaml
2. Reduce cache size in configmap.yaml
3. Enable memory profiling and find leaks

### Network Policy Issues

**Test connectivity**:
```bash
# From HoloLoom pod to Neo4j
kubectl exec -it <hololoom-pod> -n hololoom -- curl http://neo4j:7474

# From Ingress to HoloLoom
kubectl run -it --rm debug --image=curlimages/curl --restart=Never \
  -n ingress-nginx -- curl http://hololoom-api.hololoom:8080/health
```

**Temporarily disable NetworkPolicy** (for debugging):
```bash
# Delete NetworkPolicy
kubectl delete networkpolicy hololoom-api-ingress -n hololoom

# Test again
# Re-apply when done
kubectl apply -f networkpolicy.yaml
```

---

## Additional Resources

- **Operations Runbook**: `../context/OPERATIONS_RUNBOOK.md`
- **Troubleshooting Guide**: `../context/TROUBLESHOOTING_GUIDE.md`
- **Performance Tuning**: `../context/PERFORMANCE_TUNING_GUIDE.md`
- **Production Integration**: `../context/PRODUCTION_INTEGRATION_COMPLETE.md`

---

## Support

For issues or questions:
- **Documentation**: See guides in `HoloLoom/context/`
- **Issues**: https://github.com/yourusername/hololoom/issues
- **Email**: support@hololoom.ai

---

**End of Kubernetes Deployment Guide**
