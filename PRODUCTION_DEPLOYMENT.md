# HoloLoom Production Deployment Guide

**Version**: 1.0.0
**Date**: November 17, 2025
**Target SLA**: 99.9% uptime (8.76 hours downtime/year)
**Capacity**: 10,000 concurrent users
**Expected Latency**: p50=150ms, p95=500ms, p99=1000ms

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Requirements](#infrastructure-requirements)
4. [Deployment Options](#deployment-options)
5. [Quick Start (Docker Compose)](#quick-start-docker-compose)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Monitoring Setup](#monitoring-setup)
8. [Database Management](#database-management)
9. [Scaling Guide](#scaling-guide)
10. [Troubleshooting](#troubleshooting)
11. [Cost Estimates](#cost-estimates)

---

## Overview

This guide covers deploying HoloLoom to production with:
- **High Availability**: 3+ replicas with auto-scaling
- **Zero-Downtime Deployments**: Blue-green strategy
- **Comprehensive Monitoring**: Prometheus + Grafana
- **Automated Backups**: Daily database backups with 30-day retention
- **Security**: Non-root containers, network policies, secrets management

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
   ┌────▼─────┐            ┌─────▼────┐
   │ HoloLoom │  (3-20     │ HoloLoom │
   │   API    │  replicas) │   API    │
   └────┬─────┘            └─────┬────┘
        │                         │
        └────────────┬────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
    ┌───▼───┐   ┌───▼───┐   ┌───▼───┐
    │ Neo4j │   │Qdrant │   │Postgres│
    │(Graph)│   │(Vector│   │  (SQL) │
    └───────┘   └───────┘   └────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
  ┌─────▼─────┐ ┌───▼────────┐  │
  │Prometheus │ │  Grafana   │  │
  │ (Metrics) │ │(Dashboards)│  │
  └───────────┘ └────────────┘  │
```

---

## Prerequisites

### Required Software

- **Docker**: 20.10+ (for local deployment)
- **Docker Compose**: 2.0+ (for local deployment)
- **Kubernetes**: 1.25+ (for production)
- **kubectl**: Latest version
- **Helm**: 3.0+ (optional, for easier deployments)
- **Python**: 3.11+ (for local development)

### Required Accounts & Access

- **Container Registry**: GitHub Container Registry, Docker Hub, or AWS ECR
- **Kubernetes Cluster**: AWS EKS, Google GKE, or Azure AKS
- **Domain Name**: For production deployment (e.g., `hololoom.example.com`)
- **SSL Certificate**: Let's Encrypt (cert-manager) or custom
- **Alerting**: Slack webhook or email SMTP (for alerts)

### Required Secrets

Create these secrets before deployment:

```bash
# Neo4j credentials
NEO4J_PASSWORD=<strong-password>

# PostgreSQL credentials
POSTGRES_PASSWORD=<strong-password>

# Grafana admin credentials
GRAFANA_ADMIN_PASSWORD=<strong-password>

# GitHub Container Registry (if using GHCR)
GITHUB_TOKEN=<personal-access-token>
```

---

## Infrastructure Requirements

### Minimum Requirements (Staging/Development)

| Component | CPU | Memory | Storage | Count |
|-----------|-----|--------|---------|-------|
| **HoloLoom API** | 500m | 1Gi | 10Gi | 1 |
| **Neo4j** | 1 | 2Gi | 20Gi | 1 |
| **Qdrant** | 500m | 1Gi | 20Gi | 1 |
| **PostgreSQL** | 500m | 1Gi | 10Gi | 1 |
| **Redis** | 250m | 512Mi | 5Gi | 1 |
| **Prometheus** | 500m | 2Gi | 50Gi | 1 |
| **Grafana** | 250m | 512Mi | 5Gi | 1 |
| **Total** | **3.5 CPU** | **8.5Gi RAM** | **120Gi** | - |

### Production Requirements (10,000 concurrent users)

| Component | CPU | Memory | Storage | Count | Notes |
|-----------|-----|--------|---------|-------|-------|
| **HoloLoom API** | 2 | 4Gi | 10Gi | 3-20 | Auto-scales |
| **Neo4j** | 4 | 8Gi | 100Gi | 1-3 | HA cluster recommended |
| **Qdrant** | 2 | 4Gi | 100Gi | 1-3 | HA cluster recommended |
| **PostgreSQL** | 2 | 4Gi | 50Gi | 1-3 | HA cluster recommended |
| **Redis** | 1 | 2Gi | 10Gi | 1-3 | Optional HA |
| **Prometheus** | 2 | 4Gi | 200Gi | 1 | 30-day retention |
| **Grafana** | 500m | 1Gi | 10Gi | 1 | - |
| **Nginx LB** | 500m | 512Mi | 1Gi | 2 | HA pair |
| **Total** | **32-140 CPU** | **72-320Gi** | **681Gi** | - | Scales with load |

**Recommended Instance Types**:
- **AWS**: 3x m5.4xlarge (16 vCPU, 64 GB RAM) + EBS volumes
- **GCP**: 3x n2-standard-16 (16 vCPU, 64 GB RAM) + persistent disks
- **Azure**: 3x Standard_D16s_v3 (16 vCPU, 64 GB RAM) + managed disks

---

## Deployment Options

### Option 1: Docker Compose (Local/Staging)

**Best for**: Development, staging, small deployments (<100 users)

**Pros**:
- Quick setup (5 minutes)
- Single server deployment
- Easy debugging
- No Kubernetes knowledge required

**Cons**:
- Limited scalability
- No auto-scaling
- Manual failover

### Option 2: Kubernetes (Production)

**Best for**: Production, high availability, auto-scaling

**Pros**:
- Auto-scaling (3-20 replicas)
- Zero-downtime deployments
- Self-healing
- Industry standard

**Cons**:
- Complex setup
- Requires Kubernetes knowledge
- Higher infrastructure costs

### Option 3: Managed Services

**Best for**: Enterprises, teams without DevOps expertise

**Use**:
- AWS ECS Fargate (for containers)
- AWS RDS (for PostgreSQL)
- AWS ElastiCache (for Redis)
- Managed Neo4j (Neo4j Aura)
- Managed Qdrant (Qdrant Cloud)

**Pros**:
- Fully managed
- Automatic backups
- Built-in HA

**Cons**:
- Higher costs (2-3x)
- Less control
- Vendor lock-in

---

## Quick Start (Docker Compose)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/hololoom.git
cd hololoom
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp deployment/docker/.env.example deployment/docker/.env

# Edit environment variables
nano deployment/docker/.env
```

**Required `.env` variables**:
```env
# Database passwords
NEO4J_PASSWORD=your_strong_password_here
POSTGRES_PASSWORD=your_strong_password_here

# Grafana admin
GRAFANA_USER=admin
GRAFANA_PASSWORD=your_strong_password_here

# Optional: OpenAI/Anthropic API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Step 3: Start Services

```bash
cd deployment/docker
docker-compose -f docker-compose.production.yml up -d
```

**Expected output**:
```
✅ Creating network "hololoom-network"
✅ Creating volume "neo4j-data"
✅ Creating volume "qdrant-storage"
✅ Creating volume "postgres-data"
✅ Creating volume "prometheus-data"
✅ Creating volume "grafana-data"
✅ Creating hololoom-neo4j
✅ Creating hololoom-qdrant
✅ Creating hololoom-postgres
✅ Creating hololoom-redis
✅ Creating hololoom-api
✅ Creating hololoom-prometheus
✅ Creating hololoom-grafana
✅ Creating hololoom-nginx
```

### Step 4: Verify Deployment

```bash
# Check all services are running
docker-compose ps

# Test API health
curl http://localhost/health

# Expected response:
# {"status":"online","uptime":123.45,"version":"1.0.0"}

# Test query endpoint
curl -X POST http://localhost/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is Thompson Sampling?",
    "mode": "direct",
    "max_steps": 1
  }'
```

### Step 5: Access Services

- **HoloLoom API**: http://localhost (port 80)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)

### Step 6: Initialize Database

```bash
# Run initial schema migration
docker exec -i hololoom-postgres psql -U hololoom -d hololoom \
  < deployment/database/schemas/001_initial_schema.sql

# Verify schema
docker exec hololoom-postgres psql -U hololoom -d hololoom -c "\dt"
```

---

## Kubernetes Deployment

### Step 1: Prepare Cluster

**Create Kubernetes cluster** (choose one):

**AWS EKS**:
```bash
eksctl create cluster \
  --name hololoom-production \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type m5.4xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed
```

**GCP GKE**:
```bash
gcloud container clusters create hololoom-production \
  --region us-central1 \
  --machine-type n2-standard-16 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10
```

**Azure AKS**:
```bash
az aks create \
  --resource-group hololoom-rg \
  --name hololoom-production \
  --node-count 3 \
  --node-vm-size Standard_D16s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10
```

### Step 2: Install Prerequisites

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.0/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager (for TLS certificates)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install Metrics Server (for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Step 3: Create Namespace & Secrets

```bash
# Create namespace
kubectl apply -f deployment/k8s/namespace.yaml

# Create secrets
kubectl create secret generic hololoom-secrets \
  --namespace=hololoom \
  --from-literal=NEO4J_USER=neo4j \
  --from-literal=NEO4J_PASSWORD=<your-password> \
  --from-literal=POSTGRES_USER=hololoom \
  --from-literal=POSTGRES_PASSWORD=<your-password> \
  --from-literal=POSTGRES_DB=hololoom \
  --from-literal=GRAFANA_ADMIN_USER=admin \
  --from-literal=GRAFANA_ADMIN_PASSWORD=<your-password>

# Verify secrets
kubectl get secrets -n hololoom
```

### Step 4: Deploy Databases

```bash
# Create Persistent Volume Claims
kubectl apply -f deployment/k8s/pvc.yaml

# Deploy Neo4j
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: hololoom
spec:
  serviceName: neo4j-service
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.14.0
        ports:
        - containerPort: 7474
        - containerPort: 7687
        - containerPort: 2004
        env:
        - name: NEO4J_AUTH
          value: neo4j/\$(NEO4J_PASSWORD)
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: hololoom-secrets
              key: NEO4J_PASSWORD
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
EOF

# Deploy Qdrant, PostgreSQL, Redis (similar patterns)
# Or use Helm charts for easier deployment
```

### Step 5: Deploy HoloLoom API

```bash
# Apply ConfigMap
kubectl apply -f deployment/k8s/configmap.yaml

# Deploy application
kubectl apply -f deployment/k8s/deployment.yaml

# Create service
kubectl apply -f deployment/k8s/service.yaml

# Create ingress
kubectl apply -f deployment/k8s/ingress.yaml

# Enable auto-scaling
kubectl apply -f deployment/k8s/hpa.yaml
```

### Step 6: Verify Deployment

```bash
# Check pod status
kubectl get pods -n hololoom

# Expected output:
# NAME                           READY   STATUS    RESTARTS   AGE
# hololoom-api-xxxxx-yyyyy       1/1     Running   0          2m
# hololoom-api-xxxxx-zzzzz       1/1     Running   0          2m
# hololoom-api-xxxxx-wwwww       1/1     Running   0          2m
# neo4j-0                        1/1     Running   0          5m
# qdrant-0                       1/1     Running   0          5m
# postgres-0                     1/1     Running   0          5m

# Check service endpoints
kubectl get svc -n hololoom

# Check ingress
kubectl get ingress -n hololoom

# Test health endpoint
INGRESS_IP=$(kubectl get ingress hololoom-ingress -n hololoom \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

curl http://$INGRESS_IP/health
```

### Step 7: Initialize Database Schema

```bash
# Port-forward to PostgreSQL
kubectl port-forward -n hololoom svc/postgres-service 5432:5432 &

# Run schema migration
PGPASSWORD=<your-password> psql -h localhost -U hololoom -d hololoom \
  < deployment/database/schemas/001_initial_schema.sql

# Verify
PGPASSWORD=<your-password> psql -h localhost -U hololoom -d hololoom -c "\dt"
```

---

## Monitoring Setup

### Prometheus Configuration

Prometheus is configured to scrape metrics from:
- HoloLoom API (port 8000, path `/metrics`)
- Neo4j (port 2004)
- Qdrant (port 6333, path `/metrics`)
- Kubernetes API server
- Node metrics

**Configure alerts**:

```bash
# Alerting rules are already configured in:
# deployment/prometheus/alerts/hololoom_alerts.yml

# Verify alerts are loaded
kubectl exec -n hololoom prometheus-0 -- \
  promtool check rules /etc/prometheus/alerts/*.yml
```

### Grafana Dashboards

**Access Grafana**:
```
URL: http://<ingress-ip>/grafana
User: admin
Password: <your-grafana-password>
```

**Import Dashboards**:

1. Navigate to Dashboards → Import
2. Upload dashboard JSON files from `deployment/grafana/dashboards/`
3. Select Prometheus as data source

**Available Dashboards**:
- **Query Latency** - p50, p95, p99 latency over time
- **Cache Performance** - Hit rates, latencies, effectiveness
- **Workflow Execution** - Workflow status, durations, success rates
- **System Health** - CPU, memory, disk, network

### Alert Configuration

**Slack Integration**:

```bash
# Create Alertmanager config
kubectl create configmap alertmanager-config \
  --namespace=hololoom \
  --from-literal=config.yml='
global:
  slack_api_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

route:
  receiver: "slack"
  group_by: ["alertname", "severity"]
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h

receivers:
  - name: "slack"
    slack_configs:
      - channel: "#hololoom-alerts"
        title: "HoloLoom Alert"
        text: "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}"
'
```

---

## Database Management

### Backup Strategy

**Automated Daily Backups**:

```bash
# Schedule daily backups (using CronJob)
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: hololoom-backup
  namespace: hololoom
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: hololoom/backup-tool:latest
            env:
            - name: BACKUP_DIR
              value: /backups
            - name: RETENTION_DAYS
              value: "30"
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          restartPolicy: OnFailure
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
EOF
```

**Manual Backup**:

```bash
# Backup all databases
./deployment/database/backup_restore.sh backup-all

# Backup specific database
./deployment/database/backup_restore.sh backup-postgres
./deployment/database/backup_restore.sh backup-neo4j
./deployment/database/backup_restore.sh backup-qdrant
```

### Restore Procedures

**Full Restore**:

```bash
# List available backups
ls -lh /backups/

# Restore PostgreSQL
./deployment/database/backup_restore.sh restore-postgres \
  /backups/postgres/hololoom_20251117_120000.dump

# Restore Neo4j
./deployment/database/backup_restore.sh restore-neo4j \
  /backups/neo4j/neo4j_20251117_120000.dump

# Restore Qdrant
./deployment/database/backup_restore.sh restore-qdrant \
  /backups/qdrant/qdrant_20251117_120000.tar.gz
```

### Database Migrations

```bash
# Run new migration
kubectl exec -i postgres-0 -n hololoom -- \
  psql -U hololoom -d hololoom < deployment/database/schemas/002_new_migration.sql

# Verify migration
kubectl exec postgres-0 -n hololoom -- \
  psql -U hololoom -d hololoom -c "SELECT version FROM schema_migrations;"
```

---

## Scaling Guide

### Horizontal Pod Autoscaling (HPA)

HPA is configured to scale based on CPU and memory:

```bash
# Check HPA status
kubectl get hpa -n hololoom

# Expected output:
# NAME              REFERENCE                TARGETS         MINPODS   MAXPODS   REPLICAS
# hololoom-api-hpa  Deployment/hololoom-api  45%/70%         3         20        3

# Manual scale (override HPA temporarily)
kubectl scale deployment hololoom-api -n hololoom --replicas=10

# View scaling events
kubectl describe hpa hololoom-api-hpa -n hololoom
```

### Vertical Pod Autoscaling (VPA)

VPA automatically adjusts resource requests/limits:

```bash
# Check VPA recommendations
kubectl get vpa -n hololoom

# View recommendations
kubectl describe vpa hololoom-api-vpa -n hololoom
```

### Cluster Autoscaling

**AWS EKS**:
```bash
# Cluster autoscaler is enabled during cluster creation
# Monitor autoscaling events
kubectl logs -f -n kube-system \
  -l app=cluster-autoscaler
```

**Manual node scaling**:
```bash
# AWS
eksctl scale nodegroup \
  --cluster=hololoom-production \
  --name=standard-workers \
  --nodes=5

# GCP
gcloud container clusters resize hololoom-production \
  --node-pool=default-pool \
  --num-nodes=5 \
  --region=us-central1
```

### Database Scaling

**Neo4j HA Cluster**:
```bash
# Scale Neo4j to 3 replicas (1 leader + 2 followers)
kubectl scale statefulset neo4j -n hololoom --replicas=3
```

**Qdrant Cluster**:
```bash
# Scale Qdrant to 3 nodes
kubectl scale statefulset qdrant -n hololoom --replicas=3
```

**PostgreSQL HA** (requires operator):
```bash
# Use Patroni or Postgres Operator for HA
# https://github.com/zalando/postgres-operator
```

---

## Troubleshooting

### Common Issues

#### 1. Pods Stuck in `Pending`

**Symptom**: Pods show `Pending` status for >5 minutes

**Check**:
```bash
kubectl describe pod <pod-name> -n hololoom
```

**Common Causes**:
- Insufficient cluster resources (CPU/memory)
- PVC not bound (storage class issue)
- Image pull failure (authentication issue)

**Solutions**:
```bash
# Scale up cluster
eksctl scale nodegroup --cluster=hololoom-production --name=standard-workers --nodes=5

# Check PVC status
kubectl get pvc -n hololoom

# Check image pull secrets
kubectl get secrets -n hololoom
```

#### 2. High Latency (p95 > 500ms)

**Check Metrics**:
```bash
# Query Prometheus
curl -X POST http://prometheus:9090/api/v1/query \
  -d 'query=histogram_quantile(0.95, rate(hololoom_query_duration_seconds_bucket[5m]))'
```

**Common Causes**:
- Database connection pool exhaustion
- Memory pressure causing GC pauses
- Network latency to databases

**Solutions**:
```bash
# Scale up API pods
kubectl scale deployment hololoom-api -n hololoom --replicas=10

# Check database connections
kubectl exec neo4j-0 -n hololoom -- \
  cypher-shell -u neo4j -p <password> \
  "CALL dbms.listConnections();"

# Check pod memory usage
kubectl top pods -n hololoom
```

#### 3. Database Connection Failures

**Check Neo4j**:
```bash
# Test connection
kubectl exec -it hololoom-api-xxxxx -n hololoom -- \
  python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://neo4j-service:7687', auth=('neo4j', 'password'))
driver.verify_connectivity()
print('✅ Neo4j connection successful')
"
```

**Check PostgreSQL**:
```bash
# Test connection
kubectl exec -it hololoom-api-xxxxx -n hololoom -- \
  python -c "
import psycopg2
conn = psycopg2.connect('postgresql://hololoom:password@postgres-service:5432/hololoom')
print('✅ PostgreSQL connection successful')
"
```

#### 4. Out of Memory (OOM) Kills

**Check OOM events**:
```bash
kubectl get events -n hololoom | grep OOM

# Check pod memory usage
kubectl top pods -n hololoom --sort-by=memory
```

**Solutions**:
```bash
# Increase memory limits
kubectl patch deployment hololoom-api -n hololoom -p '
spec:
  template:
    spec:
      containers:
      - name: hololoom-api
        resources:
          limits:
            memory: 8Gi
'

# Enable VPA for automatic recommendations
kubectl apply -f deployment/k8s/hpa.yaml
```

### Debug Logs

```bash
# View API logs
kubectl logs -f hololoom-api-xxxxx -n hololoom

# View logs from all replicas
kubectl logs -f -l app=hololoom,component=api -n hololoom

# View Neo4j logs
kubectl logs -f neo4j-0 -n hololoom

# View Prometheus logs
kubectl logs -f prometheus-0 -n hololoom
```

### Performance Profiling

```bash
# Enable profiling endpoint
kubectl port-forward -n hololoom svc/hololoom-api 8000:8000 &

# Capture CPU profile (30 seconds)
curl http://localhost:8000/debug/pprof/profile?seconds=30 > cpu.prof

# Analyze with pprof
go tool pprof -http=:8080 cpu.prof
```

---

## Cost Estimates

### Docker Compose (Single Server)

**Infrastructure**:
- **AWS EC2**: t3.2xlarge (8 vCPU, 32 GB RAM) = $0.33/hour
- **EBS Storage**: 200 GB GP3 = $16/month
- **Data Transfer**: ~100 GB/month = $9/month

**Total**: ~$265/month

**Suitable for**: <100 concurrent users, staging environments

### Kubernetes (Production - 10,000 users)

**Infrastructure** (AWS EKS example):
- **EKS Cluster**: $0.10/hour = $73/month
- **EC2 Instances**: 3x m5.4xlarge (16 vCPU, 64 GB each) = $1.54/hour × 3 = $3,348/month
- **EBS Storage**: 700 GB GP3 = $56/month
- **Load Balancer**: Network Load Balancer = $16/month + $0.006/GB = ~$100/month
- **Data Transfer**: ~1 TB/month = $90/month
- **Backups**: S3 (500 GB) = $11/month

**Subtotal Infrastructure**: ~$3,678/month

**LLM API Costs** (if using external LLMs):
- **Anthropic Claude 3.5 Sonnet**: $3/M input tokens, $15/M output tokens
- **OpenAI GPT-4**: $30/M input tokens, $60/M output tokens
- **Estimated** (10,000 users, 10 queries/day, 500 tokens avg):
  - Total tokens/month: 10,000 × 10 × 30 × 500 = 1.5B tokens
  - Anthropic: $4,500 - $22,500/month
  - OpenAI: $45,000 - $90,000/month

**Total (with Anthropic)**: ~$8,000 - $26,000/month
**Total (local Ollama)**: ~$3,678/month

**Cost Optimization**:
- Use local Ollama models (free, but slower)
- Implement aggressive caching (100x speedup)
- Use Spot Instances (70% cheaper)
- Reduce instance sizes during off-hours

### Managed Services (Enterprise)

**AWS Managed**:
- **ECS Fargate**: 10 tasks (4 vCPU, 8 GB each) = $1,800/month
- **RDS PostgreSQL**: db.r6g.2xlarge = $800/month
- **ElastiCache Redis**: cache.r6g.xlarge = $350/month
- **Neo4j Aura**: Professional (64 GB) = $1,500/month
- **Qdrant Cloud**: Business (100 GB) = $500/month

**Total**: ~$4,950/month (excludes LLM costs)

**Pros**: Fully managed, automatic backups, HA
**Cons**: 35% more expensive than self-managed

---

## Performance Benchmarks

### Expected Throughput

| Metric | Value | Notes |
|--------|-------|-------|
| **Queries/Second** | 200-500 | With 10 API replicas |
| **P50 Latency** | 150ms | Typical query |
| **P95 Latency** | 500ms | Complex query |
| **P99 Latency** | 1000ms | Very complex query |
| **Cache Hit Rate** | 60-80% | With warm cache |
| **Concurrent Users** | 10,000+ | With auto-scaling |
| **Uptime SLA** | 99.9% | 8.76 hours downtime/year |

### Load Testing

```bash
# Install k6
brew install k6  # macOS
# or
sudo apt-get install k6  # Ubuntu

# Run load test
k6 run - <<EOF
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 1000 },  // Ramp up to 1000 users
    { duration: '5m', target: 1000 },  // Stay at 1000 users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests < 500ms
  },
};

export default function () {
  const payload = JSON.stringify({
    text: 'What is Thompson Sampling?',
    mode: 'direct',
    max_steps: 1,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  let res = http.post('http://hololoom.example.com/query', payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
EOF
```

---

## Security Checklist

Before going to production, verify:

- [ ] **Change default passwords** (Neo4j, PostgreSQL, Grafana)
- [ ] **Enable TLS/SSL** (cert-manager + Let's Encrypt)
- [ ] **Configure network policies** (restrict pod-to-pod communication)
- [ ] **Use secrets management** (Sealed Secrets or Vault)
- [ ] **Enable RBAC** (Role-Based Access Control in Kubernetes)
- [ ] **Scan Docker images** (Trivy, Snyk, or Clair)
- [ ] **Enable audit logging** (Kubernetes audit logs)
- [ ] **Configure rate limiting** (NGINX Ingress + Cloudflare)
- [ ] **Enable DDoS protection** (Cloudflare, AWS Shield)
- [ ] **Set up WAF** (Web Application Firewall)
- [ ] **Implement backup encryption** (encrypt backups at rest)
- [ ] **Configure alerting** (Slack, PagerDuty, email)

---

## Next Steps

1. **Deploy to Staging**: Follow [Quick Start](#quick-start-docker-compose) or [Kubernetes Deployment](#kubernetes-deployment)
2. **Load Test**: Run k6 load tests to validate performance
3. **Monitor**: Set up Grafana dashboards and Prometheus alerts
4. **Backup**: Configure automated daily backups
5. **Scale**: Enable HPA and VPA for auto-scaling
6. **Optimize**: Review metrics and optimize based on real usage
7. **Document**: Update runbooks with your specific procedures

---

## Support & Contact

- **Documentation**: https://docs.hololoom.ai
- **GitHub Issues**: https://github.com/your-org/hololoom/issues
- **Slack Community**: https://hololoom.slack.com
- **Email**: support@hololoom.ai

---

**Last Updated**: November 17, 2025
**Version**: 1.0.0
**Maintainers**: HoloLoom DevOps Team
