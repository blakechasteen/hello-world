# Wool Storage Production Deployment Guide

**Author**: Claude Code
**Date**: November 17, 2025
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Local Development (Docker Compose)](#local-development)
3. [Production Deployment (Kubernetes)](#production-deployment)
4. [Monitoring Setup](#monitoring-setup)
5. [Operations Guide](#operations-guide)
6. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers deploying Wool Storage (Phases 6-8) in both development and production environments:
- **Phase 6**: Distributed storage cluster (3+ nodes)
- **Phase 7**: Transparent compression (LZ4/Zstd)
- **Phase 8**: Versioning + time-travel + branch/merge

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              Production Architecture                 │
│                                                       │
│  Load Balancer (NodePort/Ingress)                   │
│         ↓                                             │
│  ┌───────────┬───────────┬───────────┐              │
│  │  Wool-1   │  Wool-2   │  Wool-3   │  (3-10 pods) │
│  │ Primary   │ Replica   │ Replica   │              │
│  └─────┬─────┴─────┬─────┴─────┬─────┘              │
│        │           │           │                     │
│        └───────────┴───────────┘                     │
│              Gossip + Replication                     │
│                                                       │
│  Persistent Volumes: 100Gi per node                  │
│  Monitoring: Prometheus + Grafana                    │
└─────────────────────────────────────────────────────┘
```

### Requirements

**Local Development**:
- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 50GB disk space

**Production (Kubernetes)**:
- Kubernetes 1.25+
- kubectl configured
- 3+ worker nodes (recommended)
- StorageClass with dynamic provisioning
- 400GB+ total storage (for 3 nodes + Prometheus)

---

## Local Development

### Quick Start

```bash
# Navigate to deployment directory
cd HoloLoom/wool/deployment

# Start cluster
docker-compose up -d

# View logs
docker-compose logs -f

# Check cluster health
docker-compose ps

# Access services
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Wool Node 1: http://localhost:9000
# Wool Node 2: http://localhost:9001
# Wool Node 3: http://localhost:9002

# Stop cluster
docker-compose down

# Stop and remove volumes (DESTROYS DATA)
docker-compose down -v
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| wool-1 | 9000 | Primary storage node |
| wool-2 | 9001 | Replica storage node |
| wool-3 | 9002 | Replica storage node |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Dashboards |
| neo4j | 7474, 7687 | Graph database (future) |
| qdrant | 6333 | Vector database (future) |

### Testing Local Cluster

```bash
# Store a file on node 1
curl -X POST http://localhost:9000/store \
  -H "Content-Type: application/octet-stream" \
  --data-binary "Hello, distributed world!"

# Response: {"file_id": "abc123...", "replicas": 3}

# Read from node 2 (should have replica)
curl http://localhost:9001/read/abc123...

# Check replication status
curl http://localhost:9000/stats | jq '.replication'

# Check compression stats
curl http://localhost:9000/stats | jq '.compression'

# Check versioning stats
curl http://localhost:9000/stats | jq '.versioning'
```

### Configuration

Edit `docker-compose.yml` to customize:

```yaml
# Change cluster size
# Add more nodes:
wool-4:
  image: python:3.11-slim
  # ... same config as wool-1, but different port
  ports:
    - "9003:9003"
  environment:
    - NODE_ID=wool-4
    - PEERS=wool-1,wool-2,wool-3,wool-4

# Adjust resource limits
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G

# Change replication factor
environment:
  - REPLICATION_FACTOR=3

# Enable/disable features
environment:
  - ENABLE_COMPRESSION=true
  - ENABLE_VERSIONING=true
  - ENABLE_DELTA_ENCODING=true
```

---

## Production Deployment

### Prerequisites

```bash
# Verify kubectl access
kubectl cluster-info

# Create namespace
kubectl create namespace wool-storage

# Verify storage class
kubectl get storageclass

# If no default storage class, create one
# (Example for local-path provisioner)
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
```

### Deploy Wool Storage Cluster

```bash
cd HoloLoom/wool/deployment/kubernetes

# Deploy wool storage
kubectl apply -f wool-statefulset.yaml

# Verify deployment
kubectl get pods -n wool-storage -w

# Expected output:
# wool-storage-0   1/1     Running   0          2m
# wool-storage-1   1/1     Running   0          1m30s
# wool-storage-2   1/1     Running   0          1m

# Check PVCs
kubectl get pvc -n wool-storage

# Check services
kubectl get svc -n wool-storage
```

### Deploy Monitoring

```bash
# Deploy Prometheus + Grafana
kubectl apply -f monitoring.yaml

# Wait for pods
kubectl get pods -n wool-storage -w

# Access Grafana (port-forward)
kubectl port-forward svc/grafana 3000:3000 -n wool-storage

# Open http://localhost:3000
# Login: admin/admin (change immediately!)
```

### Verify Cluster

```bash
# Check pod logs
kubectl logs wool-storage-0 -n wool-storage

# Should see:
# INFO: Initialized versioned wool storage
# INFO: Delta encoding: enabled
# INFO: Joined cluster with 3 nodes
# INFO: Gossip: All nodes HEALTHY

# Check metrics endpoint
kubectl exec wool-storage-0 -n wool-storage -- curl localhost:9001/metrics

# Test replication
kubectl exec wool-storage-0 -n wool-storage -- \
  curl -X POST localhost:9000/store \
  -H "Content-Type: text/plain" \
  -d "Production test data"

# Verify on replica
kubectl exec wool-storage-1 -n wool-storage -- \
  curl localhost:9000/stats | grep replication
```

### Scaling

```bash
# Manual scaling
kubectl scale statefulset wool-storage --replicas=5 -n wool-storage

# HPA will auto-scale between 3-10 based on CPU
kubectl get hpa -n wool-storage

# Monitor scaling
kubectl get pods -n wool-storage -w

# Rebalancing happens automatically
# Check logs for rebalancing progress
kubectl logs wool-storage-0 -n wool-storage | grep rebalance
```

### External Access

**Option 1: NodePort** (already configured)
```bash
# Get node IP
kubectl get nodes -o wide

# Access via NodePort (30900)
curl http://<NODE_IP>:30900/health
```

**Option 2: Ingress** (recommended for production)
```yaml
# Create ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: wool-storage-ingress
  namespace: wool-storage
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
    - host: wool.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: wool-storage-external
                port:
                  number: 9000

# Apply
kubectl apply -f ingress.yaml

# Access via domain
curl http://wool.example.com/health
```

---

## Monitoring Setup

### Prometheus Targets

```bash
# Access Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n wool-storage

# Open http://localhost:9090
# Navigate to Status > Targets

# Should see:
# - prometheus (up)
# - wool-node-1 (up)
# - wool-node-2 (up)
# - wool-node-3 (up)
```

### Key Metrics

**Wool Storage Metrics** (exposed on port 9001/metrics):

```
# Operations
wool_storage_operations_total{operation="store"}
wool_storage_operations_total{operation="read"}
wool_storage_operations_total{operation="replicate"}

# Performance
wool_storage_latency_seconds{operation="store",quantile="0.95"}
wool_storage_latency_seconds{operation="read",quantile="0.95"}

# Compression
wool_compression_ratio{content_type="text/plain"}
wool_compression_savings_bytes_total

# Versioning
wool_versions_created_total
wool_versions_merged_total
wool_delta_encoding_ratio

# Cluster
wool_cluster_nodes_total
wool_cluster_nodes_healthy
wool_replication_lag_seconds
```

### Grafana Dashboards

**Create Wool Storage Dashboard**:

```bash
# Port-forward Grafana
kubectl port-forward svc/grafana 3000:3000 -n wool-storage

# Login: http://localhost:3000 (admin/admin)

# Create dashboard with panels:
```

**Panel 1: Cluster Health**
```promql
# Query
up{job=~"wool-node.*"}

# Visualization: Stat
# Title: Cluster Health
```

**Panel 2: Throughput**
```promql
# Query
rate(wool_storage_operations_total[5m])

# Visualization: Graph
# Title: Operations/sec
```

**Panel 3: Latency (p95)**
```promql
# Query
wool_storage_latency_seconds{quantile="0.95"}

# Visualization: Graph
# Title: 95th Percentile Latency
```

**Panel 4: Compression Ratio**
```promql
# Query
wool_compression_ratio

# Visualization: Gauge
# Title: Compression Ratio
```

**Panel 5: Storage Usage**
```promql
# Query
sum(wool_storage_bytes_total) by (node_id)

# Visualization: Bar gauge
# Title: Storage per Node
```

### Alerting Rules

Create `alerts.yml`:

```yaml
groups:
  - name: wool_storage_alerts
    interval: 30s
    rules:
      - alert: WoolNodeDown
        expr: up{job=~"wool-node.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Wool storage node is down"
          description: "Node {{ $labels.node_id }} has been down for >1 minute"

      - alert: HighLatency
        expr: wool_storage_latency_seconds{quantile="0.95"} > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High storage latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 1s)"

      - alert: LowReplicationFactor
        expr: wool_replication_factor < 3
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Replication factor below target"
          description: "Current replication: {{ $value }}, target: 3"

      - alert: DiskSpaceWarning
        expr: (wool_storage_bytes_total / wool_storage_capacity_bytes) > 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Disk space running low"
          description: "Node {{ $labels.node_id }} is {{ $value | humanizePercentage }} full"
```

---

## Operations Guide

### Backup & Restore

**Backup (Kubernetes)**:
```bash
# Backup PVCs using Velero (recommended)
velero backup create wool-backup --include-namespaces wool-storage

# Or manual backup
for pod in $(kubectl get pods -n wool-storage -o name | grep wool-storage); do
  kubectl exec $pod -n wool-storage -- tar czf /tmp/backup.tar.gz /data
  kubectl cp wool-storage/$pod:/tmp/backup.tar.gz ./backup-$pod.tar.gz
done
```

**Restore**:
```bash
# Using Velero
velero restore create --from-backup wool-backup

# Or manual
kubectl cp ./backup-wool-storage-0.tar.gz wool-storage/wool-storage-0:/tmp/backup.tar.gz
kubectl exec wool-storage-0 -n wool-storage -- tar xzf /tmp/backup.tar.gz -C /
```

### Upgrading

```bash
# Update image in statefulset
kubectl set image statefulset/wool-storage \
  wool-storage=wool-storage:v2.0 \
  -n wool-storage

# Rolling update (one pod at a time)
kubectl rollout status statefulset/wool-storage -n wool-storage

# Rollback if needed
kubectl rollout undo statefulset/wool-storage -n wool-storage
```

### Node Maintenance

```bash
# Drain node for maintenance
NODE_NAME="wool-storage-1"
kubectl drain $NODE_NAME --ignore-daemonsets --delete-emptydir-data

# Node will be rescheduled automatically
# Cluster continues operating (PodDisruptionBudget ensures ≥2 pods)

# After maintenance
kubectl uncordon $NODE_NAME
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod wool-storage-0 -n wool-storage

# Common issues:
# - PVC pending: Check storage class
kubectl get pvc -n wool-storage

# - Image pull error: Verify image exists
kubectl get events -n wool-storage

# - Resource limits: Check node capacity
kubectl top nodes
```

### Cluster Not Forming

```bash
# Check logs for gossip messages
kubectl logs wool-storage-0 -n wool-storage | grep gossip

# Verify network connectivity between pods
kubectl exec wool-storage-0 -n wool-storage -- \
  ping wool-storage-1.wool-storage-headless.wool-storage.svc.cluster.local

# Check peer list
kubectl logs wool-storage-0 -n wool-storage | grep PEERS
```

### High Latency

```bash
# Check Prometheus metrics
# Query: wool_storage_latency_seconds{quantile="0.95"}

# Common causes:
# 1. Disk I/O bottleneck
kubectl top pods -n wool-storage

# 2. Network congestion
kubectl exec wool-storage-0 -n wool-storage -- iperf3 -c wool-storage-1

# 3. Resource limits
kubectl describe pod wool-storage-0 -n wool-storage | grep -A 5 Limits
```

### Replication Lag

```bash
# Check replication metrics
curl http://<NODE_IP>:30900/stats | jq '.replication'

# Force replication
kubectl exec wool-storage-0 -n wool-storage -- \
  curl -X POST localhost:9000/admin/force-replication
```

---

## Security Considerations

### Production Hardening

1. **Change default passwords**:
```bash
# Grafana
kubectl create secret generic grafana-secret \
  --from-literal=admin-password=<STRONG_PASSWORD> \
  -n wool-storage --dry-run=client -o yaml | kubectl apply -f -

# Neo4j
kubectl set env statefulset/neo4j NEO4J_AUTH=neo4j/<STRONG_PASSWORD> -n wool-storage
```

2. **Enable TLS**:
```yaml
# Add to wool-storage container env
- name: ENABLE_TLS
  value: "true"
- name: TLS_CERT_PATH
  value: "/certs/tls.crt"
- name: TLS_KEY_PATH
  value: "/certs/tls.key"
```

3. **Network policies**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: wool-storage-netpol
  namespace: wool-storage
spec:
  podSelector:
    matchLabels:
      app: wool-storage
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: wool-storage
      ports:
        - protocol: TCP
          port: 9000
        - protocol: TCP
          port: 9001
```

4. **Pod Security Policies**:
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: wool-storage-psp
spec:
  privileged: false
  runAsUser:
    rule: MustRunAsNonRoot
  fsGroup:
    rule: RunAsAny
  volumes:
    - 'persistentVolumeClaim'
```

---

## Performance Tuning

### Resource Optimization

**High Throughput** (increase CPU/memory):
```yaml
resources:
  requests:
    cpu: "2000m"
    memory: "8Gi"
  limits:
    cpu: "4000m"
    memory: "16Gi"
```

**Cost Optimization** (reduce resources):
```yaml
resources:
  requests:
    cpu: "250m"
    memory: "512Mi"
  limits:
    cpu: "1000m"
    memory: "2Gi"
```

### Storage Tuning

**SSD Storage Class** (recommended):
```yaml
volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"  # Change to SSD class
      resources:
        requests:
          storage: 100Gi
```

### Compression Tuning

```yaml
env:
  - name: COMPRESSION_ALGORITHM
    value: "zstd:3"  # or "lz4" for speed
  - name: COMPRESSION_THRESHOLD
    value: "1024"    # Compress files >1KB
```

---

**End of Deployment Guide**

For additional support, see:
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Migration Guide](MIGRATION.md)
- [Performance Tuning](PERFORMANCE.md)
