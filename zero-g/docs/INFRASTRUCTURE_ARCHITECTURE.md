# Zero-G Infrastructure Architecture

**Version**: 1.0.0
**Date**: 2025-11-22
**Phase**: 2 - Infrastructure Setup

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Zero-G Data Onboarding System                     │
│                              (Phase 2 Complete)                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          Development Environment                         │
│                            (Docker Compose)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Zero-G      │  │   Neo4j     │  │   Qdrant    │  │ Prometheus  │   │
│  │  Backend    │◄─┤   Graph     │◄─┤   Vector    │◄─┤  Metrics    │   │
│  │             │  │  Database   │  │   Store     │  │ Collection  │   │
│  │ :8000       │  │ :7474/:7687 │  │ :6333/:6334 │  │    :9090    │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         ▲                                                    │           │
│         │                                                    ▼           │
│         │                                           ┌─────────────┐     │
│         │                                           │   Grafana   │     │
│         │                                           │  Dashboard  │     │
│         │                                           │    :3000    │     │
│         │                                           └─────────────┘     │
│         │                                                                │
│         └────────────────────────────────────────────────────────────── │
│                        Metrics Endpoint (/metrics)                       │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         Production Environment                           │
│                              (Kubernetes)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      zerog-prod namespace                         │   │
│  │                                                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐ │   │
│  │  │  Zero-G Backend (HPA: 2-10 pods)                            │ │   │
│  │  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐               │ │   │
│  │  │  │ Pod 1 │  │ Pod 2 │  │ Pod 3 │  │  ...  │  LoadBalancer │ │   │
│  │  │  └───┬───┘  └───┬───┘  └───┬───┘  └───────┘       :8000    │ │   │
│  │  └──────┼──────────┼──────────┼──────────────────────────────┘ │   │
│  │         │          │          │                                  │   │
│  │         └──────────┴──────────┘                                  │   │
│  │                    │                                              │   │
│  │         ┌──────────┴──────────┐                                  │   │
│  │         │                     │                                  │   │
│  │  ┌──────▼─────┐        ┌─────▼──────┐                           │   │
│  │  │   Neo4j    │        │   Qdrant   │                           │   │
│  │  │ StatefulSet│        │ Deployment │                           │   │
│  │  │            │        │            │                           │   │
│  │  │ ClusterIP  │        │ ClusterIP  │                           │   │
│  │  └────────────┘        └────────────┘                           │   │
│  │         │                     │                                  │   │
│  │         ▼                     ▼                                  │   │
│  │  ┌──────────────────────────────┐                               │   │
│  │  │  Persistent Volume Claims    │                               │   │
│  │  │  - neo4j-data-pvc (10Gi)     │                               │   │
│  │  │  - neo4j-logs-pvc (5Gi)      │                               │   │
│  │  │  - qdrant-storage-pvc (20Gi) │                               │   │
│  │  └──────────────────────────────┘                               │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Monitoring Stack                               │   │
│  │                                                                    │   │
│  │  ┌──────────────┐           ┌──────────────┐                     │   │
│  │  │  Prometheus  │◄──────────┤   Grafana    │   LoadBalancer      │   │
│  │  │  Deployment  │  Scrapes  │  Deployment  │      :3000          │   │
│  │  │              │           │              │                     │   │
│  │  │  ClusterIP   │           │              │                     │   │
│  │  └──────┬───────┘           └──────────────┘                     │   │
│  │         │                                                          │   │
│  │         ▼                                                          │   │
│  │  ┌──────────────┐                                                 │   │
│  │  │ prometheus-  │                                                 │   │
│  │  │ data-pvc     │                                                 │   │
│  │  │ (10Gi)       │                                                 │   │
│  │  └──────────────┘                                                 │   │
│  └────────────────────────────────────────────────────────────────────┘│
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### Core Services

#### 1. Zero-G Backend (FastAPI)
**Purpose**: Main application server handling data ingestion and API requests

**Development**:
- Port: 8000
- Workers: 4
- Container: zerog-backend:latest
- Network: zerog-network

**Production**:
- Replicas: 2-10 (HPA controlled)
- Service Type: LoadBalancer
- Resources:
  - Request: 500m CPU, 1Gi RAM
  - Limit: 2000m CPU, 2Gi RAM
- Health Checks: /health endpoint

**Environment Variables**:
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
- QDRANT_HOST, QDRANT_PORT, QDRANT_GRPC_PORT
- ENVIRONMENT, LOG_LEVEL, WORKERS
- SECRET_KEY, CORS_ORIGINS
- PROMETHEUS_ENABLED

#### 2. Neo4j (Graph Database)
**Purpose**: Yarn Graph storage - discrete thread structure for symbolic memory

**Development**:
- Port: 7474 (HTTP), 7687 (Bolt)
- Container: neo4j:5.13.0
- Volume: neo4j_data

**Production**:
- Deployment: StatefulSet (single instance)
- Service Type: ClusterIP
- Resources:
  - Request: 500m CPU, 1Gi RAM
  - Limit: 1000m CPU, 2Gi RAM
- Storage:
  - Data: 10Gi PVC
  - Logs: 5Gi PVC

**Configuration**:
- Page cache: 512M
- Heap: 512M initial, 1G max
- Plugins: APOC

#### 3. Qdrant (Vector Store)
**Purpose**: Warp Space storage - continuous semantic embeddings

**Development**:
- Port: 6333 (HTTP), 6334 (gRPC)
- Container: qdrant/qdrant:v1.7.0
- Volume: qdrant_storage

**Production**:
- Deployment: Deployment (single instance)
- Service Type: ClusterIP
- Resources:
  - Request: 250m CPU, 512Mi RAM
  - Limit: 500m CPU, 1Gi RAM
- Storage: 20Gi PVC

**Features**:
- Vector similarity search
- gRPC for high-performance queries
- HTTP for management

#### 4. Prometheus (Metrics)
**Purpose**: Time-series metrics collection and alerting

**Development**:
- Port: 9090
- Container: prom/prometheus:v2.48.0
- Volume: prometheus_data

**Production**:
- Deployment: Deployment (single instance)
- Service Type: ClusterIP
- Resources:
  - Request: 250m CPU, 512Mi RAM
  - Limit: 500m CPU, 1Gi RAM
- Storage: 10Gi PVC
- Retention: 30 days

**Scrape Targets**:
- zerog-backend:8000/metrics (10s interval)
- neo4j:7474/metrics (15s interval)
- qdrant:6333/metrics (15s interval)

**Alert Rules**:
- Backend down (critical)
- Database down (critical)
- High error rate >1% (warning)
- High latency >100ms (warning)
- Low astronaut activity (info)

#### 5. Grafana (Visualization)
**Purpose**: Metrics visualization and dashboards

**Development**:
- Port: 3000
- Container: grafana/grafana:10.2.2
- Volume: grafana_data

**Production**:
- Deployment: Deployment (single instance)
- Service Type: LoadBalancer
- Resources:
  - Request: 100m CPU, 256Mi RAM
  - Limit: 250m CPU, 512Mi RAM
- Storage: 5Gi PVC

**Dashboards**:
- Zero-G Mission Control (auto-provisioned)
- 9 panels: Launch Status, System Health, Latency, Error Rate, etc.

---

## Data Flow

### Request Flow (Development)

```
User/Client
    │
    ▼
Zero-G Backend :8000
    │
    ├──► Neo4j :7687 (Bolt)
    │     └──► Yarn Graph queries/writes
    │
    ├──► Qdrant :6333 (HTTP)
    │     └──► Vector similarity search
    │
    └──► Prometheus :8000/metrics
          └──► Metrics collection (pull)
                │
                ▼
         Grafana :3000
              └──► Dashboard visualization
```

### Request Flow (Production)

```
External Load Balancer
    │
    ▼
Zero-G Backend Service (LoadBalancer)
    │
    ├──► Pod 1 (HPA controlled)
    ├──► Pod 2
    └──► Pod N (up to 10)
          │
          ├──► Neo4j Service (ClusterIP)
          │     └──► Neo4j Pod
          │           └──► neo4j-data PVC (10Gi)
          │
          └──► Qdrant Service (ClusterIP)
                └──► Qdrant Pod
                      └──► qdrant-storage PVC (20Gi)

Prometheus Service (ClusterIP)
    │
    ├──► Scrapes Zero-G Backend pods
    ├──► Scrapes Neo4j pod
    ├──► Scrapes Qdrant pod
    └──► Stores to prometheus-data PVC (10Gi)

Grafana Service (LoadBalancer)
    └──► Queries Prometheus
          └──► Displays Zero-G Mission Control dashboard
```

---

## Network Architecture

### Development (Docker Compose)

**Network**: zerog-network (bridge)
**Subnet**: 172.28.0.0/16

**Service Discovery**: By container name
- neo4j → neo4j:7687
- qdrant → qdrant:6333
- prometheus → prometheus:9090

**Exposed Ports**:
- 8000 (Backend)
- 7474, 7687 (Neo4j)
- 6333, 6334 (Qdrant)
- 9090 (Prometheus)
- 3000 (Grafana)

### Production (Kubernetes)

**Namespace**: zerog-prod (or zerog-staging)

**Service Types**:
- ClusterIP: Neo4j, Qdrant, Prometheus (internal only)
- LoadBalancer: Zero-G Backend, Grafana (external access)

**Service Discovery**: Kubernetes DNS
- neo4j.zerog-prod.svc.cluster.local:7687
- qdrant.zerog-prod.svc.cluster.local:6333
- prometheus.zerog-prod.svc.cluster.local:9090

**Ingress** (optional, not configured):
- Use Ingress controller for TLS termination
- Route traffic to services via hostname

---

## Storage Architecture

### Development Volumes

All volumes are Docker named volumes with local driver:

| Volume | Service | Size | Purpose |
|--------|---------|------|---------|
| neo4j_data | Neo4j | ~10GB | Graph database data |
| neo4j_logs | Neo4j | ~5GB | Database logs |
| neo4j_import | Neo4j | ~1GB | Import staging |
| neo4j_plugins | Neo4j | ~500MB | APOC plugins |
| qdrant_storage | Qdrant | ~20GB | Vector embeddings |
| prometheus_data | Prometheus | ~10GB | Time-series metrics |
| grafana_data | Grafana | ~5GB | Dashboards & config |
| zerog_logs | Backend | ~1GB | Application logs |

**Total Storage**: ~52GB

### Production Persistent Volumes

All volumes use Kubernetes PersistentVolumeClaims:

| PVC | Service | Size | Access Mode | Storage Class |
|-----|---------|------|-------------|---------------|
| neo4j-data-pvc | Neo4j | 10Gi | ReadWriteOnce | standard |
| neo4j-logs-pvc | Neo4j | 5Gi | ReadWriteOnce | standard |
| qdrant-storage-pvc | Qdrant | 20Gi | ReadWriteOnce | standard |
| prometheus-data-pvc | Prometheus | 10Gi | ReadWriteOnce | standard |
| grafana-data-pvc | Grafana | 5Gi | ReadWriteOnce | standard |

**Total Storage**: 50Gi

**Storage Providers**:
- AWS: EBS (gp3)
- GCP: Persistent Disk (pd-standard)
- Azure: Managed Disk (Standard_LRS)

---

## Scaling Strategy

### Horizontal Pod Autoscaler (HPA)

**Target**: Zero-G Backend only

**Configuration**:
```yaml
minReplicas: 2
maxReplicas: 10

metrics:
  - CPU: 70% utilization
  - Memory: 80% utilization

behavior:
  scaleUp:
    stabilization: 60s
    policies:
      - Percent: 100% (double)
      - Pods: 4
    selectPolicy: Max

  scaleDown:
    stabilization: 300s (5 min)
    policies:
      - Percent: 50% (half)
      - Pods: 2
    selectPolicy: Min
```

**Scaling Trigger**:
- High CPU: Scale up by 100% or 4 pods (whichever is more)
- High Memory: Same as CPU
- Low CPU/Memory: Scale down by 50% or 2 pods after 5 min stabilization

**Database Scaling**:
- Neo4j: Single instance (manual scaling)
- Qdrant: Single instance (future: clustering)

---

## Monitoring & Observability

### Metrics Collected

**Zero-G Backend**:
- `zerog_requests_total{status}` - Request count by status code
- `zerog_request_duration_seconds{quantile}` - Request latency histogram
- `zerog_active_astronauts` - Current active users
- `zerog_connected_data_sources` - Connected data sources
- `process_resident_memory_bytes` - Memory usage
- `process_cpu_seconds_total` - CPU usage

**Neo4j**:
- `neo4j_node_count` - Total nodes
- `neo4j_relationship_count` - Total relationships
- `neo4j_memory_heap_used_bytes` - Heap memory
- `neo4j_disk_free_bytes` - Disk space

**Qdrant**:
- `qdrant_collection_vectors_count` - Total vectors
- `qdrant_collection_size_bytes` - Storage size

### Grafana Dashboard Panels

1. **Launch Status** (Gauge) - Backend uptime (0/1)
2. **System Health** (Stat) - All services up/down
3. **Request Latency** (Time Series) - p50, p95, p99
4. **Error Rate** (Time Series) - 5xx errors percentage
5. **Active Astronauts** (Stat) - Current users
6. **Connected Data Sources** (Stat) - Data source count
7. **Requests per Second** (Time Series) - RPS breakdown
8. **Memory Usage** (Time Series) - Backend & Neo4j
9. **Graph Database Stats** (Stat) - Nodes, relationships, vectors

### Alert Severity Levels

| Level | Response Time | Examples |
|-------|---------------|----------|
| **Critical** | Immediate | Backend down, database down |
| **Warning** | <1 hour | High error rate, high latency |
| **Info** | <24 hours | Low activity, no data sources |

---

## Security Architecture

### Development Security

**Limitations**:
- Default passwords (must be changed)
- HTTP only (no TLS)
- No authentication on databases
- Exposed ports on localhost

**Acceptable for**: Local development only

### Production Security

**Implemented**:
- ✅ Secrets stored in Kubernetes Secrets
- ✅ Non-root containers (user: zerog)
- ✅ Network isolation (ClusterIP for internal)
- ✅ Resource limits (prevent DoS)
- ✅ Health checks (detect failures)
- ✅ CORS configured per environment

**Required for Production**:
- 🔲 TLS/SSL for all services
- 🔲 Authentication enabled on databases
- 🔲 Network policies (restrict pod-to-pod)
- 🔲 RBAC (service accounts with minimal permissions)
- 🔲 Secret rotation (every 90 days)
- 🔲 Image scanning (Trivy/Snyk)
- 🔲 Pod Security Standards (restricted)

---

## Deployment Strategies

### Development

**Strategy**: Direct deployment
```bash
docker-compose up -d
```

**Rollback**:
```bash
docker-compose down
docker-compose up -d
```

**Zero-downtime**: Not applicable (single instance)

### Production

**Strategy**: Rolling update (default)
```bash
kubectl apply -k k8s/overlays/production/
```

**Parameters**:
- Max unavailable: 25%
- Max surge: 25%

**Rollback**:
```bash
kubectl rollout undo deployment/zerog-backend -n zerog-prod
```

**Zero-downtime**: Yes (with min 2 replicas)

**Blue-Green** (future):
```bash
# Switch service selector to new deployment
kubectl patch service zerog-backend -p '{"spec":{"selector":{"version":"v2"}}}'
```

**Canary** (future):
- Use Argo Rollouts or Flagger
- Progressive traffic shifting: 10% → 50% → 100%

---

## Backup & Recovery

### Backup Targets

| Service | What | Frequency | Retention |
|---------|------|-----------|-----------|
| Neo4j | Database dump | Daily | 30 days |
| Qdrant | Snapshots | Daily | 30 days |
| Prometheus | Metrics | Weekly | 90 days |
| Grafana | Dashboards | Weekly | 90 days |

### Backup Commands

**Neo4j** (Development):
```bash
docker-compose exec neo4j neo4j-admin dump \
  --database=neo4j \
  --to=/backups/neo4j-$(date +%Y%m%d).dump
```

**Qdrant** (Development):
```bash
curl -X POST http://localhost:6333/collections/zerog/snapshots
curl http://localhost:6333/collections/zerog/snapshots/latest \
  > qdrant-backup-$(date +%Y%m%d).snapshot
```

**Neo4j** (Production):
```bash
kubectl exec -n zerog-prod neo4j-0 -- \
  neo4j-admin dump --database=neo4j --to=/backups/neo4j-$(date +%Y%m%d).dump
```

### Recovery Commands

**Neo4j**:
```bash
docker-compose exec neo4j neo4j-admin load \
  --from=/backups/neo4j-20251122.dump \
  --database=neo4j --force
```

**Qdrant**:
```bash
curl -X PUT http://localhost:6333/collections/zerog/snapshots/upload \
  --data-binary @qdrant-backup-20251122.snapshot
```

---

## Cost Estimation

### Development (Local)

**Cost**: $0 (runs on local machine)

**Resources Required**:
- 4 CPU cores
- 8GB RAM
- 50GB disk

### Production (AWS)

**Infrastructure** (3 worker nodes):
- EC2 instances: 3× m5.xlarge (4 vCPU, 16GB RAM)
  - $0.192/hr × 3 = $0.576/hr
  - Monthly: ~$414

**Storage** (EBS gp3):
- 50Gi total: $0.08/GB-month
  - Monthly: ~$4

**Load Balancer** (2× ALB):
- $0.0225/hr × 2 = $0.045/hr
  - Monthly: ~$32

**Data Transfer** (1TB/month):
- $0.09/GB × 1000GB
  - Monthly: ~$90

**Total Monthly Cost**: ~$540/month

**Optimizations**:
- Use spot instances: -70% compute cost → $270/month
- Reserved instances (1 year): -40% compute cost → $350/month
- Reduce data transfer: -50% → $495/month

---

## Performance Expectations

### Latency Targets

| Endpoint | Target (p95) | Target (p99) |
|----------|--------------|--------------|
| /health | 5ms | 10ms |
| /query | 100ms | 200ms |
| /ingest | 150ms | 300ms |
| Neo4j query | 50ms | 100ms |
| Qdrant search | 20ms | 40ms |

### Throughput Targets

| Operation | Target (RPS) | Max (RPS) |
|-----------|--------------|-----------|
| Read queries | 100 | 1000 |
| Write queries | 50 | 500 |
| Bulk ingest | 10 | 100 |

### Resource Utilization Targets

| Service | CPU (avg) | Memory (avg) | Disk I/O |
|---------|-----------|--------------|----------|
| Backend | 40% | 60% | Low |
| Neo4j | 50% | 70% | Medium |
| Qdrant | 30% | 60% | High |
| Prometheus | 20% | 50% | Medium |
| Grafana | 10% | 30% | Low |

---

## Future Enhancements

### Phase 3: High Availability

- [ ] Multi-zone deployment
- [ ] Neo4j clustering (causal cluster)
- [ ] Qdrant distributed mode
- [ ] Redis cache layer
- [ ] Database replication

### Phase 4: Advanced Monitoring

- [ ] Distributed tracing (Jaeger)
- [ ] Log aggregation (ELK/Loki)
- [ ] APM (Application Performance Monitoring)
- [ ] Synthetic monitoring
- [ ] Alerting via PagerDuty/OpsGenie

### Phase 5: Security Hardening

- [ ] mTLS between services
- [ ] Certificate rotation automation
- [ ] Secret encryption at rest
- [ ] Vulnerability scanning pipeline
- [ ] Compliance reporting (SOC2, HIPAA)

### Phase 6: Cost Optimization

- [ ] Resource right-sizing
- [ ] Auto-scaling optimization
- [ ] Spot instance integration
- [ ] Storage tiering
- [ ] Cost allocation tags

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-22
**Author**: Agent A - Infrastructure Architect
