# HoloLoom Production Deployment Guide

Complete guide for deploying HoloLoom in production with Docker.

## Quick Start

### 1. Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available
- At least 10GB disk space

### 2. Configuration

Copy and customize the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your production settings:

```bash
# Database Configuration
NEO4J_PASSWORD=your_secure_password_here
QDRANT_COLLECTION=hololoom_production

# HoloLoom Configuration
EXECUTION_MODE=fast  # or 'fused' for maximum quality
LOG_LEVEL=INFO
ENABLE_PROFILING=true
ENABLE_METRICS=true

# Security
JWT_SECRET=your_jwt_secret_here
API_KEY=your_api_key_here
```

### 3. Deploy

Start the complete stack:

```bash
# Basic deployment (HoloLoom + Neo4j + Qdrant)
docker-compose -f docker-compose.production.yml up -d

# With monitoring (adds Prometheus + Grafana)
docker-compose -f docker-compose.production.yml --profile monitoring up -d
```

### 4. Verify

Check all services are healthy:

```bash
docker-compose -f docker-compose.production.yml ps
```

Expected output:
```
NAME                 COMMAND                  STATUS              PORTS
hololoom-app         "python -m HoloLoom.…"   Up (healthy)        0.0.0.0:8000->8000/tcp
hololoom-neo4j       "docker-entrypoint.s…"   Up (healthy)        0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
hololoom-qdrant      "./qdrant"               Up (healthy)        0.0.0.0:6333->6333/tcp
```

Access the services:
- **HoloLoom API**: http://localhost:8000
- **Neo4j Browser**: http://localhost:7474 (credentials: neo4j/your_password)
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Grafana** (if monitoring enabled): http://localhost:3000 (admin/admin)

## Architecture

### Services

```
┌─────────────────────────────────────────────────────────────┐
│                     HoloLoom Stack                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Client     │─────▶│  HoloLoom    │                    │
│  │  (User/API)  │      │     App      │                    │
│  └──────────────┘      └───────┬──────┘                    │
│                                │                            │
│                     ┌──────────┴──────────┐                │
│                     │                     │                │
│              ┌──────▼──────┐      ┌──────▼──────┐          │
│              │   Neo4j     │      │   Qdrant    │          │
│              │   (Graph)   │      │  (Vector)   │          │
│              └─────────────┘      └─────────────┘          │
│                                                             │
│  Optional Monitoring:                                       │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ Prometheus   │─────▶│   Grafana    │                    │
│  │  (Metrics)   │      │ (Dashboard)  │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Volumes

Persistent data storage:

| Volume | Purpose | Path |
|--------|---------|------|
| `neo4j_data` | Graph database storage | `/data` |
| `neo4j_logs` | Database logs | `/logs` |
| `qdrant_data` | Vector database storage | `/qdrant/storage` |
| `prometheus_data` | Metrics storage | `/prometheus` |
| `grafana_data` | Dashboard configs | `/var/lib/grafana` |
| `./logs` | Application logs | `/app/logs` |
| `./checkpoints` | Model checkpoints | `/app/checkpoints` |

## Configuration

### Execution Modes

Choose based on your requirements:

| Mode | Speed | Quality | Memory | Use Case |
|------|-------|---------|--------|----------|
| `bare` | Fast | Basic | Low | Development, testing |
| `fast` | Medium | Good | Medium | Production (recommended) |
| `fused` | Slow | Best | High | High-quality inference |

Set via environment variable:
```bash
EXECUTION_MODE=fast
```

### Performance Tuning

#### Neo4j Memory Configuration

Edit `docker-compose.production.yml`:

```yaml
neo4j:
  environment:
    # For 8GB+ RAM systems
    - NEO4J_dbms_memory_pagecache_size=2G
    - NEO4J_dbms_memory_heap_max__size=4G
```

#### Qdrant Performance

```yaml
qdrant:
  environment:
    # Enable mmap for better performance
    - QDRANT__STORAGE__MMAP=true
```

#### HoloLoom Scaling

```yaml
hololoom:
  deploy:
    replicas: 3  # Horizontal scaling
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
```

## Monitoring

### Health Checks

Check service health:

```bash
# All services
docker-compose -f docker-compose.production.yml ps

# Individual service
docker-compose -f docker-compose.production.yml exec hololoom python -c "from HoloLoom import config; print('healthy')"
```

### Logs

View logs:

```bash
# All services
docker-compose -f docker-compose.production.yml logs -f

# HoloLoom only
docker-compose -f docker-compose.production.yml logs -f hololoom

# Neo4j only
docker-compose -f docker-compose.production.yml logs -f neo4j
```

### Performance Metrics

#### Built-in Dashboard

HoloLoom includes a terminal dashboard:

```bash
docker-compose -f docker-compose.production.yml exec hololoom python -m HoloLoom.performance.dashboard
```

#### Prometheus Metrics

If monitoring is enabled, metrics are available at:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

Example queries:
```promql
# Query latency P95
hololoom_query_processing_latency_seconds{quantile="0.95"}

# Queries per second
rate(hololoom_queries_processed_total[5m])

# Memory usage
hololoom_memory_mb
```

## Backup & Recovery

### Backup

#### Neo4j Backup

```bash
# Stop HoloLoom (keep Neo4j running)
docker-compose -f docker-compose.production.yml stop hololoom

# Dump database
docker-compose -f docker-compose.production.yml exec neo4j neo4j-admin dump \
  --database=neo4j --to=/backups/neo4j-backup-$(date +%Y%m%d).dump

# Copy backup to host
docker cp hololoom-neo4j:/backups/neo4j-backup-$(date +%Y%m%d).dump ./backups/
```

#### Qdrant Backup

```bash
# Create snapshot
curl -X POST "http://localhost:6333/collections/hololoom/snapshots"

# Download snapshot
curl -O "http://localhost:6333/collections/hololoom/snapshots/{snapshot_name}"
```

### Restore

#### Neo4j Restore

```bash
# Stop services
docker-compose -f docker-compose.production.yml down

# Load backup
docker-compose -f docker-compose.production.yml run neo4j neo4j-admin load \
  --from=/backups/neo4j-backup-YYYYMMDD.dump --database=neo4j --force

# Restart
docker-compose -f docker-compose.production.yml up -d
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Symptoms:**
- Container restarts
- "Killed" messages in logs

**Solution:**
```bash
# Reduce memory usage
docker-compose -f docker-compose.production.yml exec hololoom \
  sh -c "export EXECUTION_MODE=bare && python app.py"
```

#### 2. Neo4j Connection Refused

**Symptoms:**
- "Connection refused" errors
- HoloLoom can't connect to Neo4j

**Solution:**
```bash
# Check Neo4j health
docker-compose -f docker-compose.production.yml logs neo4j

# Verify connectivity
docker-compose -f docker-compose.production.yml exec hololoom \
  python -c "from neo4j import GraphDatabase; print('OK')"
```

#### 3. Slow Queries

**Symptoms:**
- High latency (>2s per query)
- Low QPS (<1)

**Solution:**

1. Check execution mode (should be `fast` for production)
2. Review Neo4j indexes:
```cypher
// In Neo4j Browser
SHOW INDEXES;

// Create missing indexes
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);
```

3. Enable query cache:
```python
# In config
config.enable_cache = True
config.cache_ttl_seconds = 300
```

### Debug Mode

Enable verbose logging:

```yaml
hololoom:
  environment:
    - LOG_LEVEL=DEBUG
    - ENABLE_PROFILING=true
```

## Security

### Production Checklist

- [ ] Change default passwords (Neo4j, Grafana)
- [ ] Use secrets management (Docker secrets or Vault)
- [ ] Enable TLS/SSL for external access
- [ ] Restrict network access (firewall rules)
- [ ] Enable authentication for APIs
- [ ] Regular security updates
- [ ] Backup verification
- [ ] Log rotation

### Secrets Management

Use Docker secrets for sensitive data:

```yaml
secrets:
  neo4j_password:
    file: ./secrets/neo4j_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt

services:
  hololoom:
    secrets:
      - neo4j_password
      - jwt_secret
    environment:
      - NEO4J_PASSWORD_FILE=/run/secrets/neo4j_password
```

## Scaling

### Horizontal Scaling

Run multiple HoloLoom instances:

```yaml
hololoom:
  deploy:
    replicas: 3
```

Add load balancer (Nginx):

```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
  depends_on:
    - hololoom
```

### Vertical Scaling

Increase resources per container:

```yaml
hololoom:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
```

## Maintenance

### Updates

Update HoloLoom:

```bash
# Pull latest code
git pull origin main

# Rebuild image
docker-compose -f docker-compose.production.yml build hololoom

# Rolling update (zero downtime)
docker-compose -f docker-compose.production.yml up -d --no-deps --build hololoom
```

### Database Maintenance

#### Neo4j

```cypher
// Check database size
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Store sizes")

// Compact database
CALL db.checkpoint()
```

#### Qdrant

```bash
# Optimize collection
curl -X POST "http://localhost:6333/collections/hololoom/optimizer"
```

## Support

- **Documentation**: `/docs`
- **Issues**: GitHub Issues
- **Community**: Discord/Slack

---

**Last Updated**: October 27, 2025
