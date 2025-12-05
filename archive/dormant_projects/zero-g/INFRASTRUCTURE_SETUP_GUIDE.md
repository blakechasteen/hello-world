# Zero-G Infrastructure Setup Guide

**Version**: 1.0.0
**Date**: 2025-11-22
**Author**: Agent A - Infrastructure Architect

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Environment (Docker Compose)](#development-environment-docker-compose)
3. [Production Deployment (Kubernetes)](#production-deployment-kubernetes)
4. [Health Check Verification](#health-check-verification)
5. [Monitoring & Observability](#monitoring--observability)
6. [Troubleshooting](#troubleshooting)
7. [Security Best Practices](#security-best-practices)

---

## Prerequisites

### Required Tools

#### Docker Compose (Development)
- **Docker**: v24.0+ ([Install Guide](https://docs.docker.com/get-docker/))
- **Docker Compose**: v2.20+ (included with Docker Desktop)

#### Kubernetes (Production)
- **kubectl**: v1.28+ ([Install Guide](https://kubernetes.io/docs/tasks/tools/))
- **kustomize**: v5.0+ ([Install Guide](https://kubectl.docs.kubernetes.io/installation/kustomize/))
- **Helm**: v3.12+ (optional, for advanced deployments)

#### Access Requirements
- **Docker Hub** or private registry access
- **Kubernetes cluster** (AWS EKS, GKE, AKS, or local k3s/minikube)
- **kubectl** configured with cluster credentials

### System Requirements

**Development (Docker Compose)**:
- CPU: 4 cores minimum
- RAM: 8GB minimum, 16GB recommended
- Disk: 50GB available space
- OS: Linux, macOS, or Windows with WSL2

**Production (Kubernetes)**:
- Worker nodes: 3+ nodes recommended
- Node specs: 4 vCPU, 16GB RAM per node
- Storage: Persistent volume support (AWS EBS, GCE PD, Azure Disk)

---

## Development Environment (Docker Compose)

### Quick Start

```bash
# Navigate to zero-g directory
cd zero-g/

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes (data loss!)
docker-compose down -v
```

### Services Overview

| Service | Port | URL | Credentials |
|---------|------|-----|-------------|
| **Zero-G Backend** | 8000 | http://localhost:8000 | N/A |
| **Neo4j Browser** | 7474 | http://localhost:7474 | neo4j / zerog_password_change_me |
| **Neo4j Bolt** | 7687 | bolt://localhost:7687 | neo4j / zerog_password_change_me |
| **Qdrant HTTP** | 6333 | http://localhost:6333 | N/A |
| **Qdrant gRPC** | 6334 | localhost:6334 | N/A |
| **Prometheus** | 9090 | http://localhost:9090 | N/A |
| **Grafana** | 3000 | http://localhost:3000 | admin / zerog_admin_change_me |

### Step-by-Step Setup

#### 1. Configure Environment Variables

Create `.env` file (optional, overrides docker-compose.yml):

```bash
# Database credentials
NEO4J_PASSWORD=your-secure-password
GRAFANA_ADMIN_PASSWORD=your-secure-password

# Backend configuration
SECRET_KEY=your-random-secret-key-here
CORS_ORIGINS=http://localhost:3001,http://localhost:3000
```

#### 2. Build Backend Image

```bash
cd backend/
docker build -t zerog-backend:latest .
cd ..
```

#### 3. Start Infrastructure Services First

```bash
# Start databases only
docker-compose up -d neo4j qdrant

# Wait for health checks (60 seconds)
docker-compose ps

# Verify Neo4j is ready
curl http://localhost:7474

# Verify Qdrant is ready
curl http://localhost:6333/health
```

#### 4. Start Monitoring Stack

```bash
# Start Prometheus and Grafana
docker-compose up -d prometheus grafana

# Wait 30 seconds for Grafana to initialize
sleep 30

# Access Grafana
open http://localhost:3000
```

#### 5. Start Zero-G Backend

```bash
# Start backend (depends on databases)
docker-compose up -d zerog-backend

# View backend logs
docker-compose logs -f zerog-backend

# Test backend health
curl http://localhost:8000/health
```

### Development Workflow

#### Hot Reload

Enable hot reload for development:

```yaml
# Add to docker-compose.yml under zerog-backend
volumes:
  - ./backend:/app
command: uvicorn mission_control.main:app --reload --host 0.0.0.0 --port 8000
```

#### Run Tests

```bash
# Inside backend container
docker-compose exec zerog-backend pytest tests/

# Or from host
docker-compose run --rm zerog-backend pytest tests/
```

#### Database Migrations

```bash
# Neo4j console (Cypher queries)
docker-compose exec neo4j cypher-shell -u neo4j -p zerog_password_change_me

# Qdrant collections
curl -X PUT http://localhost:6333/collections/zerog \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'
```

---

## Production Deployment (Kubernetes)

### Prerequisites

#### 1. Kubernetes Cluster

Ensure you have a running cluster:

```bash
# Verify cluster access
kubectl cluster-info

# Check nodes
kubectl get nodes
```

#### 2. Configure kubectl Context

```bash
# List contexts
kubectl config get-contexts

# Switch to production context
kubectl config use-context your-prod-cluster
```

#### 3. Create Container Registry Secret (if using private registry)

```bash
kubectl create secret docker-registry zerog-registry-secret \
  --docker-server=your-registry.io \
  --docker-username=your-username \
  --docker-password=your-password \
  --docker-email=your-email@example.com \
  --namespace=zerog-prod
```

### Deployment Steps

#### 1. Create Production Secrets

**CRITICAL**: Never commit secrets to Git!

```bash
# Create secrets from file (recommended)
kubectl create secret generic zerog-secrets \
  --from-literal=neo4j-auth='neo4j/your-secure-password' \
  --from-literal=neo4j-password='your-secure-password' \
  --from-literal=secret-key='generate-random-64-char-string' \
  --from-literal=grafana-admin-user='admin' \
  --from-literal=grafana-admin-password='your-secure-password' \
  --namespace=zerog-prod

# Verify secrets
kubectl get secrets -n zerog-prod
```

**Alternative**: Use external secrets management:
- AWS Secrets Manager
- HashiCorp Vault
- Sealed Secrets (https://github.com/bitnami-labs/sealed-secrets)

#### 2. Deploy Base Resources

```bash
cd k8s/

# Deploy namespaces
kubectl apply -f base/namespace.yaml

# Deploy base resources (development mode)
kubectl apply -k base/

# Verify deployment
kubectl get all -n zerog-prod
```

#### 3. Deploy Production Overlay

```bash
# Build and deploy production configuration
kubectl apply -k overlays/production/

# Verify production deployment
kubectl get all -n zerog-prod

# Check pod status
kubectl get pods -n zerog-prod -w
```

#### 4. Deploy Staging (Optional)

```bash
# Deploy to staging namespace
kubectl apply -k overlays/staging/

# Verify staging
kubectl get all -n zerog-staging
```

### Verify Deployment

```bash
# Check all resources
kubectl get all -n zerog-prod

# Check persistent volumes
kubectl get pvc -n zerog-prod

# Check secrets and configmaps
kubectl get secrets,configmaps -n zerog-prod

# View deployment status
kubectl rollout status deployment/zerog-backend -n zerog-prod
kubectl rollout status deployment/neo4j -n zerog-prod
kubectl rollout status deployment/qdrant -n zerog-prod
```

### Access Services

#### Port Forwarding (Development Access)

```bash
# Forward Zero-G Backend
kubectl port-forward -n zerog-prod svc/zerog-backend 8000:8000

# Forward Grafana
kubectl port-forward -n zerog-prod svc/grafana 3000:3000

# Forward Prometheus
kubectl port-forward -n zerog-prod svc/prometheus 9090:9090

# Forward Neo4j Browser
kubectl port-forward -n zerog-prod svc/neo4j 7474:7474
```

#### LoadBalancer (Production Access)

```bash
# Get external IPs
kubectl get svc -n zerog-prod

# Access services via LoadBalancer external IP
# Example: http://<EXTERNAL-IP>:8000
```

---

## Health Check Verification

### Docker Compose Health Checks

```bash
# Check all service health
docker-compose ps

# Expected output:
# NAME                STATUS
# zerog-backend       Up (healthy)
# neo4j              Up (healthy)
# qdrant             Up (healthy)
# prometheus         Up (healthy)
# grafana            Up (healthy)
```

### Kubernetes Health Checks

```bash
# Check pod readiness
kubectl get pods -n zerog-prod

# All pods should show READY 1/1 and STATUS Running

# Detailed pod health
kubectl describe pod <pod-name> -n zerog-prod

# View events
kubectl get events -n zerog-prod --sort-by='.lastTimestamp'
```

### Manual Endpoint Tests

#### Zero-G Backend

```bash
# Health endpoint
curl http://localhost:8000/health

# Expected: {"status": "healthy", "version": "1.0.0"}

# Metrics endpoint
curl http://localhost:8000/metrics

# Expected: Prometheus metrics output
```

#### Neo4j

```bash
# HTTP health
curl http://localhost:7474

# Cypher query (requires auth)
curl -u neo4j:your-password http://localhost:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements": [{"statement": "RETURN 1"}]}'
```

#### Qdrant

```bash
# Health endpoint
curl http://localhost:6333/health

# Expected: {"title": "qdrant - vector search engine", "version": "1.7.0"}

# Collections list
curl http://localhost:6333/collections

# Expected: {"result": {"collections": []}}
```

#### Prometheus

```bash
# Health endpoint
curl http://localhost:9090/-/healthy

# Expected: Prometheus is Healthy.

# Check targets
curl http://localhost:9090/api/v1/targets
```

#### Grafana

```bash
# Health endpoint
curl http://localhost:3000/api/health

# Expected: {"database": "ok", "version": "10.2.2"}
```

---

## Monitoring & Observability

### Accessing Grafana

1. **Navigate**: http://localhost:3000
2. **Login**: admin / zerog_admin_change_me
3. **Dashboard**: Zero-G → Zero-G Mission Control

### Dashboard Panels

| Panel | Metric | Threshold |
|-------|--------|-----------|
| **Launch Status** | Backend uptime | Green = 1 (up) |
| **System Health** | All services up | All green |
| **Request Latency** | p50/p95/p99 | p95 < 100ms |
| **Error Rate** | 5xx errors | < 1% |
| **Active Astronauts** | Active users | > 0 |
| **Data Sources** | Connected sources | > 0 |

### Prometheus Queries

Access Prometheus at http://localhost:9090 and run queries:

```promql
# Backend uptime
up{job="zerog-backend"}

# Request rate
rate(zerog_requests_total[5m])

# Error rate
sum(rate(zerog_requests_total{status=~"5.."}[5m])) / sum(rate(zerog_requests_total[5m]))

# Request latency (p95)
histogram_quantile(0.95, sum(rate(zerog_request_duration_seconds_bucket[5m])) by (le))

# Memory usage
process_resident_memory_bytes{job="zerog-backend"} / 1024 / 1024

# Graph database stats
neo4j_node_count
neo4j_relationship_count
qdrant_collection_vectors_count
```

### Alerts

Alerts are defined in `monitoring/alerts.yml`:

- **Critical**: Backend down, Neo4j down, Qdrant down
- **Warning**: High error rate (>1%), high latency (>100ms), high memory (>1.5GB)
- **Info**: Low astronaut activity, no data sources

Configure Alertmanager to receive alerts via:
- Email
- Slack
- PagerDuty
- Webhook

---

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

**Symptom**: Docker containers exit immediately

```bash
# Check logs
docker-compose logs zerog-backend

# Common causes:
# - Database not ready (increase healthcheck wait time)
# - Environment variables missing
# - Port conflicts
```

**Fix**:
```bash
# Ensure databases are healthy first
docker-compose up -d neo4j qdrant
docker-compose ps

# Wait for healthy status, then start backend
docker-compose up -d zerog-backend
```

#### 2. Connection Refused to Neo4j/Qdrant

**Symptom**: Backend logs show connection errors

```bash
# Logs show:
# neo4j.exceptions.ServiceUnavailable: Could not connect to bolt://neo4j:7687
```

**Fix**:
```bash
# Check if databases are running
docker-compose ps neo4j qdrant

# Restart databases
docker-compose restart neo4j qdrant

# Wait for healthy status
docker-compose ps
```

#### 3. Prometheus Not Scraping Metrics

**Symptom**: Grafana shows "No data"

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq
```

**Fix**:
```yaml
# Ensure backend exposes metrics endpoint
# Add to backend code:
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

#### 4. Grafana Dashboard Not Loading

**Symptom**: Dashboard shows empty panels

```bash
# Check Grafana logs
docker-compose logs grafana

# Check Prometheus datasource
curl http://localhost:3000/api/datasources
```

**Fix**:
```bash
# Restart Grafana
docker-compose restart grafana

# Re-provision datasources
docker-compose exec grafana grafana-cli admin reset-admin-password admin
```

#### 5. Kubernetes Pods Not Starting

**Symptom**: Pods stuck in Pending or CrashLoopBackOff

```bash
# Check pod events
kubectl describe pod <pod-name> -n zerog-prod

# Check logs
kubectl logs <pod-name> -n zerog-prod

# Common causes:
# - Insufficient resources (CPU/memory)
# - Secrets missing
# - PVC not bound
```

**Fix**:
```bash
# Check PVC status
kubectl get pvc -n zerog-prod

# Check secrets exist
kubectl get secrets -n zerog-prod

# Check resource quotas
kubectl describe resourcequota -n zerog-prod
```

#### 6. HPA Not Scaling

**Symptom**: Pods not autoscaling under load

```bash
# Check HPA status
kubectl get hpa -n zerog-prod

# Check metrics-server
kubectl top nodes
kubectl top pods -n zerog-prod
```

**Fix**:
```bash
# Install metrics-server if missing
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify metrics-server
kubectl get deployment metrics-server -n kube-system
```

---

## Security Best Practices

### 1. Secret Management

**Never commit secrets to Git!**

- Use environment variables
- Use external secrets management (Vault, AWS Secrets Manager)
- Rotate secrets regularly (every 90 days)
- Use strong random passwords (32+ characters)

### 2. Network Security

**Docker Compose**:
- Use custom bridge network (already configured)
- Don't expose ports unnecessarily
- Enable TLS for production

**Kubernetes**:
- Use NetworkPolicies to restrict pod-to-pod traffic
- Enable Pod Security Standards (restricted)
- Use Ingress with TLS (cert-manager + Let's Encrypt)

### 3. Image Security

- Scan images for vulnerabilities (Trivy, Snyk)
- Use minimal base images (python:3.11-slim)
- Run as non-root user (already configured)
- Update dependencies regularly

```bash
# Scan backend image
docker scan zerog-backend:latest

# Or use Trivy
trivy image zerog-backend:latest
```

### 4. Access Control

**Kubernetes RBAC**:
- Create service accounts for applications
- Use RBAC to limit permissions
- Enable audit logging

```bash
# Create service account
kubectl create serviceaccount zerog-sa -n zerog-prod

# Bind minimal permissions
kubectl create rolebinding zerog-sa-binding \
  --clusterrole=view \
  --serviceaccount=zerog-prod:zerog-sa \
  --namespace=zerog-prod
```

### 5. Database Security

**Neo4j**:
- Change default password immediately
- Enable encryption (SSL/TLS)
- Restrict network access (firewall rules)
- Regular backups

**Qdrant**:
- Use API keys in production
- Enable HTTPS
- Restrict network access

### 6. Monitoring Security

- Secure Prometheus (authentication, HTTPS)
- Secure Grafana (strong admin password, HTTPS)
- Don't expose Prometheus/Grafana publicly
- Use VPN or bastion host for access

---

## Next Steps

### Production Checklist

- [ ] Change all default passwords
- [ ] Configure TLS/SSL for all services
- [ ] Set up external secrets management
- [ ] Configure backup strategy
- [ ] Set up CI/CD pipeline
- [ ] Configure monitoring alerts
- [ ] Perform security scan
- [ ] Load testing
- [ ] Disaster recovery plan
- [ ] Documentation for team

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy Zero-G

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build and push image
        run: |
          docker build -t your-registry.io/zerog-backend:${{ github.sha }} backend/
          docker push your-registry.io/zerog-backend:${{ github.sha }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/zerog-backend \
            zerog-backend=your-registry.io/zerog-backend:${{ github.sha }} \
            -n zerog-prod
```

### Backup Strategy

**Neo4j**:
```bash
# Backup
docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j-$(date +%Y%m%d).dump

# Restore
docker-compose exec neo4j neo4j-admin load --from=/backups/neo4j-20251122.dump --database=neo4j --force
```

**Qdrant**:
```bash
# Snapshot
curl -X POST http://localhost:6333/collections/zerog/snapshots

# Download snapshot
curl http://localhost:6333/collections/zerog/snapshots/snapshot-2025-11-22 > qdrant-backup.snapshot
```

---

## Support

For issues or questions:

1. Check logs: `docker-compose logs -f` or `kubectl logs <pod> -n zerog-prod`
2. Review [Troubleshooting](#troubleshooting) section
3. Consult documentation:
   - Docker: https://docs.docker.com
   - Kubernetes: https://kubernetes.io/docs
   - Neo4j: https://neo4j.com/docs
   - Qdrant: https://qdrant.tech/documentation

---

**Version History**:
- v1.0.0 (2025-11-22): Initial infrastructure setup
