># HoloLoom Kubernetes Deployment Guide

This document provides comprehensive instructions for deploying HoloLoom to a Kubernetes cluster with high availability, auto-scaling, and monitoring.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Scaling](#scaling)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)

## Overview

HoloLoom Kubernetes deployment provides:

- **High Availability**: 3+ replicas for API and workers with pod anti-affinity
- **Auto-Scaling**: HorizontalPodAutoscaler based on CPU/memory/custom metrics
- **Persistent Storage**: StatefulSets for Neo4j, Qdrant, Redis, and RabbitMQ
- **Monitoring**: Prometheus + Grafana with pre-configured dashboards and alerts
- **Zero-Downtime Deployments**: Rolling updates with health checks
- **Distributed Architecture**: Celery workers with RabbitMQ queue and Redis cache

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Ingress (NGINX)                       │
│                    SSL/TLS Termination                        │
└────────────────────────┬─────────────────────────────────────┘
                         │
            ┌────────────┴─────────────┐
            │                          │
    ┌───────▼────────┐        ┌────────▼────────┐
    │  API Gateway   │        │    Grafana      │
    │  (3 replicas)  │        │  (monitoring)   │
    └───────┬────────┘        └─────────────────┘
            │
            ├─────────────┬─────────────┬──────────────┐
            │             │             │              │
    ┌───────▼────┐  ┌────▼─────┐  ┌────▼─────┐  ┌────▼────────┐
    │  Workers   │  │  Neo4j   │  │  Qdrant  │  │  RabbitMQ   │
    │ (2-10 HPA) │  │(3 nodes) │  │(3 nodes) │  │  (3 nodes)  │
    └────────────┘  └──────────┘  └──────────┘  └─────────────┘
                         │             │              │
                    ┌────▼─────────────▼──────────────▼────┐
                    │        Redis (Master + Replicas)     │
                    └──────────────────────────────────────┘
```

### Components

1. **API Gateway**: FastAPI-based REST API (3 replicas)
2. **Workers**: Celery worker pool (HPA 2-10 replicas)
3. **Neo4j**: Knowledge graph database (3-node cluster)
4. **Qdrant**: Vector database (3-node cluster)
5. **Redis**: Cache and Celery result backend (master + 2 replicas)
6. **RabbitMQ**: Message queue (3-node cluster)
7. **Prometheus**: Metrics collection and alerting
8. **Grafana**: Visualization and dashboards

## Prerequisites

### Required Tools

- **kubectl** >= 1.25
- **Helm** >= 3.10
- **Docker** >= 20.10 (for building images)
- **Kubernetes cluster** >= 1.25

### Optional Tools

- **minikube** or **kind** for local testing
- **k9s** for cluster management
- **kubectx** for context switching

### Cluster Requirements

**Minimum (Development/Testing):**
- 4 CPU cores
- 16 GB RAM
- 100 GB storage

**Recommended (Production):**
- 16+ CPU cores
- 64+ GB RAM
- 500+ GB storage with SSD
- Dedicated storage class for databases

## Quick Start

### 1. Local Testing with Minikube

```bash
# Start Minikube with sufficient resources
minikube start --cpus=4 --memory=16384 --disk-size=50g

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server

# Build Docker images (in minikube environment)
eval $(minikube docker-env)
cd kubernetes/docker
docker build -f Dockerfile.api -t hololoom/api:latest ../..
docker build -f Dockerfile.worker -t hololoom/worker:latest ../..

# Deploy HoloLoom
cd kubernetes/scripts
./deploy.sh

# Access the application
kubectl port-forward -n hololoom svc/hololoom-api 8000:8000

# Access Grafana
kubectl port-forward -n hololoom svc/hololoom-grafana 3000:3000
```

### 2. Production Deployment

```bash
# Configure kubectl for your cluster
kubectl config use-context <your-cluster>

# Update values.yaml with production settings
cd kubernetes/helm/hololoom
vim values.yaml
# - Change all passwords
# - Configure ingress domains
# - Set resource limits
# - Configure storage classes

# Build and push images to registry
export DOCKER_REGISTRY=your-registry.io
export IMAGE_TAG=v1.0.0
cd kubernetes/docker

docker build -f Dockerfile.api -t ${DOCKER_REGISTRY}/hololoom/api:${IMAGE_TAG} ../..
docker build -f Dockerfile.worker -t ${DOCKER_REGISTRY}/hololoom/worker:${IMAGE_TAG} ../..

docker push ${DOCKER_REGISTRY}/hololoom/api:${IMAGE_TAG}
docker push ${DOCKER_REGISTRY}/hololoom/worker:${IMAGE_TAG}

# Deploy
cd kubernetes/scripts
BUILD_IMAGES=false ENVIRONMENT=production ./deploy.sh
```

## Configuration

### values.yaml Overview

The Helm chart is configured through `kubernetes/helm/hololoom/values.yaml`. Key sections:

#### Global Settings

```yaml
global:
  namespace: hololoom
  imageRegistry: docker.io
  storageClass: standard
  environment: production
```

#### API Gateway

```yaml
api:
  enabled: true
  replicaCount: 3
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
  autoscaling:
    enabled: false  # Set true for production
```

#### Workers

```yaml
workers:
  enabled: true
  replicaCount: 2
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 75
```

#### Databases

```yaml
neo4j:
  enabled: true
  replicaCount: 3
  persistence:
    enabled: true
    size: 50Gi

qdrant:
  enabled: true
  replicaCount: 3
  persistence:
    enabled: true
    size: 30Gi
```

### Security Configuration

**IMPORTANT**: Before production deployment:

1. **Change all default passwords** in `values.yaml`:
   - `secrets.neo4j.password`
   - `secrets.redis.password`
   - `secrets.rabbitmq.password`
   - `secrets.grafana.adminPassword`
   - `secrets.api.secretKey`
   - `secrets.api.jwtSecret`

2. **Use Sealed Secrets** or external secret management:
   ```bash
   # Install sealed-secrets controller
   helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
   helm install sealed-secrets sealed-secrets/sealed-secrets -n kube-system

   # Create sealed secret
   kubeseal --format=yaml < secret.yaml > sealed-secret.yaml
   ```

3. **Configure TLS certificates**:
   ```bash
   # Install cert-manager
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

   # Configure ClusterIssuer in values.yaml
   ingress:
     annotations:
       cert-manager.io/cluster-issuer: "letsencrypt-prod"
   ```

## Deployment

### Using Helm Directly

```bash
# Install
helm install hololoom kubernetes/helm/hololoom \
  --namespace hololoom \
  --create-namespace \
  --values kubernetes/helm/hololoom/values.yaml

# Upgrade
helm upgrade hololoom kubernetes/helm/hololoom \
  --namespace hololoom \
  --values kubernetes/helm/hololoom/values.yaml

# Uninstall
helm uninstall hololoom --namespace hololoom
```

### Using Deployment Scripts

```bash
# Deploy (with optional dry-run)
DRY_RUN=true ./kubernetes/scripts/deploy.sh

# Deploy for real
./kubernetes/scripts/deploy.sh

# Rollback to previous version
./kubernetes/scripts/rollback.sh

# Rollback to specific revision
./kubernetes/scripts/rollback.sh --revision 5

# Emergency rollback (no confirmations)
./kubernetes/scripts/rollback.sh --emergency
```

### Deployment Verification

```bash
# Check deployment status
kubectl get pods -n hololoom
kubectl get svc -n hololoom
kubectl get pvc -n hololoom

# Run health checks
./kubernetes/scripts/health-check.sh

# View logs
kubectl logs -f -n hololoom -l app.kubernetes.io/component=api
kubectl logs -f -n hololoom -l app.kubernetes.io/component=worker
```

## Scaling

### Manual Scaling

```bash
# Scale workers to 5 replicas
./kubernetes/scripts/scale.sh scale worker 5

# Scale API to 5 replicas
./kubernetes/scripts/scale.sh scale api 5

# Scale up by 1
./kubernetes/scripts/scale.sh up worker

# Scale down by 1
./kubernetes/scripts/scale.sh down worker

# View current status
./kubernetes/scripts/scale.sh status
```

### Auto-Scaling with HPA

Enable HPA in `values.yaml`:

```yaml
workers:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 75
    targetMemoryUtilizationPercentage: 80
    customMetrics:
      - type: Pods
        pods:
          metric:
            name: queue_depth
          target:
            type: AverageValue
            averageValue: "30"
```

Monitor HPA:

```bash
# Watch HPA status
kubectl get hpa -n hololoom -w

# Describe HPA
kubectl describe hpa hololoom-worker -n hololoom
```

### Load Testing

```bash
# Scale for load test
./kubernetes/scripts/scale.sh load-test

# Returns system to normal
./kubernetes/scripts/scale.sh scale-down
```

## Monitoring

### Accessing Dashboards

#### Grafana

```bash
# Port forward
kubectl port-forward -n hololoom svc/hololoom-grafana 3000:3000

# Get admin password
kubectl get secret hololoom-secrets -n hololoom -o jsonpath='{.data.grafana-admin-password}' | base64 -d

# Visit http://localhost:3000
# Username: admin
# Password: <from above>
```

#### Prometheus

```bash
# Port forward
kubectl port-forward -n hololoom svc/hololoom-prometheus 9090:9090

# Visit http://localhost:9090
```

### Pre-configured Dashboards

1. **HoloLoom Overview**: System-wide metrics and health
2. **API Gateway**: Request rates, latency, errors
3. **Workers**: Queue depth, task processing, CPU/memory
4. **Databases**: Neo4j, Qdrant, Redis, RabbitMQ metrics

### Alert Rules

Configured alerts (see `kubernetes/helm/hololoom/templates/prometheus-rules.yaml`):

- **Critical**:
  - API down
  - Worker down
  - Database down
  - High error rate (>5%)

- **Warning**:
  - High latency (>1s)
  - High queue depth (>1000)
  - High memory usage (>90%)
  - Slow database queries

### Custom Metrics

Export custom metrics from your application:

```python
from prometheus_client import Counter, Histogram

# Request counter
requests_total = Counter('hololoom_requests_total', 'Total requests', ['method', 'endpoint'])

# Latency histogram
request_duration = Histogram('hololoom_request_duration_seconds', 'Request duration')

# Use in your code
requests_total.labels(method='POST', endpoint='/query').inc()
with request_duration.time():
    # Process request
    pass
```

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n hololoom

# Describe pod
kubectl describe pod <pod-name> -n hololoom

# Check logs
kubectl logs <pod-name> -n hololoom

# Check events
kubectl get events -n hololoom --sort-by='.lastTimestamp'
```

#### Persistent Volumes Not Binding

```bash
# Check PVC status
kubectl get pvc -n hololoom

# Describe PVC
kubectl describe pvc <pvc-name> -n hololoom

# Check storage class
kubectl get storageclass

# Manually create PV if needed
kubectl apply -f pv.yaml
```

#### Database Connection Issues

```bash
# Test Neo4j connection
kubectl exec -it hololoom-neo4j-0 -n hololoom -- cypher-shell -u neo4j -p <password>

# Test Qdrant connection
kubectl exec -it hololoom-qdrant-0 -n hololoom -- wget -O- http://localhost:6333/

# Test Redis connection
kubectl exec -it hololoom-redis-master-0 -n hololoom -- redis-cli ping

# Test RabbitMQ connection
kubectl exec -it hololoom-rabbitmq-0 -n hololoom -- rabbitmq-diagnostics ping
```

#### High Memory Usage

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n hololoom

# Adjust resource limits in values.yaml
workers:
  resources:
    limits:
      memory: "8Gi"  # Increase limit
```

#### Network Policies Blocking Traffic

```bash
# Check network policies
kubectl get networkpolicy -n hololoom

# Temporarily disable
kubectl delete networkpolicy <policy-name> -n hololoom
```

### Debug Mode

Enable verbose logging:

```bash
# Update configmap
kubectl edit configmap hololoom-config -n hololoom

# Set LOG_LEVEL: "DEBUG"

# Restart pods
kubectl rollout restart deployment/hololoom-api -n hololoom
kubectl rollout restart deployment/hololoom-worker -n hololoom
```

### Health Check Script

```bash
# Run comprehensive health check
./kubernetes/scripts/health-check.sh --verbose

# Check specific namespace
NAMESPACE=hololoom ./kubernetes/scripts/health-check.sh
```

## Production Considerations

### High Availability

1. **Multi-Zone Deployment**:
   ```yaml
   affinity:
     podAntiAffinity:
       requiredDuringSchedulingIgnoredDuringExecution:
         - labelSelector:
             matchExpressions:
               - key: app
                 operator: In
                 values:
                   - hololoom-api
           topologyKey: topology.kubernetes.io/zone
   ```

2. **Database Replication**:
   - Neo4j: 3+ core nodes in cluster mode
   - Qdrant: 3+ nodes with replication
   - Redis: Master + 2 replicas
   - RabbitMQ: 3+ nodes with mirrored queues

3. **Pod Disruption Budgets**:
   ```yaml
   podDisruptionBudget:
     enabled: true
     api:
       minAvailable: 2
     workers:
       minAvailable: 1
   ```

### Performance Optimization

1. **Resource Requests/Limits**: Set appropriate values based on load testing
2. **Connection Pooling**: Configure in application settings
3. **Cache Strategy**: Use Redis cache for frequently accessed data
4. **Database Indexing**: Ensure proper indexes in Neo4j
5. **Worker Concurrency**: Tune based on workload

### Security Hardening

1. **Network Policies**: Enable and configure
   ```bash
   kubectl apply -f kubernetes/network-policies/
   ```

2. **Pod Security Standards**: Enforce restricted mode
   ```yaml
   securityContext:
     runAsNonRoot: true
     runAsUser: 1000
     fsGroup: 1000
     readOnlyRootFilesystem: true
   ```

3. **RBAC**: Minimize permissions
4. **Secrets Management**: Use external secrets operator
5. **Image Scanning**: Scan images for vulnerabilities
6. **Audit Logging**: Enable Kubernetes audit logs

### Backup and Recovery

1. **Database Backups**:
   ```bash
   # Neo4j backup
   kubectl exec hololoom-neo4j-0 -n hololoom -- \
     neo4j-admin backup --backup-dir=/backup --name=backup-$(date +%Y%m%d)

   # Qdrant snapshot
   kubectl exec hololoom-qdrant-0 -n hololoom -- \
     curl -X POST http://localhost:6333/collections/hololoom/snapshots
   ```

2. **Volume Snapshots**: Use VolumeSnapshot resources
3. **Disaster Recovery**: Document and test recovery procedures

### Cost Optimization

1. **Right-Sizing**: Monitor and adjust resource requests/limits
2. **Spot Instances**: Use for non-critical workers
3. **Storage Tiers**: Use appropriate storage classes
4. **Auto-Scaling**: Configure aggressive scale-down policies

### Monitoring and Alerting

1. **SLO/SLI Definition**: Define service level objectives
2. **Alert Routing**: Configure PagerDuty/Slack integration
3. **Log Aggregation**: Use ELK or Loki stack
4. **Distributed Tracing**: Consider Jaeger or Zipkin

## Additional Resources

- [Helm Chart Reference](kubernetes/helm/hololoom/README.md)
- [Docker Build Guide](kubernetes/docker/README.md)
- [API Documentation](docs/API.md)
- [Architecture Overview](CLAUDE.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/hololoom/hololoom/issues
- Slack Community: https://hololoom.slack.com
- Email: support@hololoom.io
