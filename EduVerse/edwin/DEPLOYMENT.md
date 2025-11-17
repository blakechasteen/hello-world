# EdWIN AI Tutor - Deployment Guide

**Complete production deployment guide for EdWIN AI tutoring platform**

**Implementation Date**: November 15, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Configuration](#configuration)
7. [Database Setup](#database-setup)
8. [Monitoring](#monitoring)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## Overview

EdWIN AI Tutor is a K-12 adaptive learning platform that consists of:

- **EdWIN API** (port 8000): Main API server
- **Teacher Dashboard** (port 8001): Web interface for teachers
- **Mobile API** (port 8002): Mobile-optimized API
- **Neo4j**: Knowledge graph database
- **Qdrant**: Vector database for RAG
- **Redis**: Cache layer

### Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐
│   Client    │───▶│  Kubernetes  │───▶│  Services  │
│ (Browser/   │    │   Ingress    │    │            │
│  Mobile)    │    └──────────────┘    └────────────┘
└─────────────┘                              │
                                             ▼
                        ┌────────────────────┴────────────────────┐
                        │                                          │
              ┌─────────┴────────┐  ┌──────────┐  ┌──────────────┐
              │  EdWIN Services  │  │ Database │  │  Monitoring  │
              │  - API           │  │ - Neo4j  │  │ - Prometheus │
              │  - Dashboard     │  │ - Qdrant │  │ - Grafana    │
              │  - Mobile API    │  │ - Redis  │  │              │
              └──────────────────┘  └──────────┘  └──────────────┘
```

---

## Prerequisites

### Required

- **Docker** 20.10+ (for Docker deployment)
- **Docker Compose** 1.29+ (for Docker deployment)
- **Kubernetes** 1.24+ (for production deployment)
- **kubectl** configured with cluster access
- **Minimum Resources**:
  - 4 CPU cores
  - 8 GB RAM
  - 50 GB disk space

### Recommended

- **Nginx Ingress Controller** (for Kubernetes)
- **cert-manager** (for automatic TLS certificates)
- **Helm** 3.0+ (for advanced deployments)

---

## Quick Start

### Docker (Development)

```bash
# Clone repository
git clone https://github.com/yourorg/edwin.git
cd edwin

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose -f docker-compose.edwin.yml up -d

# Check status
docker-compose -f docker-compose.edwin.yml ps

# View logs
docker-compose -f docker-compose.edwin.yml logs -f api

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8001
# Mobile API: http://localhost:8002
# Neo4j Browser: http://localhost:7474
```

### Kubernetes (Production)

```bash
# Configure secrets
cp k8s/secrets.yaml.template k8s/secrets.yaml
# Edit k8s/secrets.yaml with actual values (base64 encoded)

# Deploy
./scripts/deploy/deploy.sh production

# Check health
./scripts/deploy/health-check.sh

# Access via Ingress
# https://api.edwin.edu
# https://dashboard.edwin.edu
# https://mobile.edwin.edu
```

---

## Docker Deployment

### 1. Environment Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```env
EDWIN_ENV=production
NEO4J_PASSWORD=your_secure_password_here
JWT_SECRET_KEY=your_jwt_secret_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
```

### 2. Build Images

```bash
docker build -t edwin-ai-tutor:latest .
```

### 3. Start Services

```bash
docker-compose -f docker-compose.edwin.yml up -d
```

### 4. Initialize Database

```bash
# Run migrations
docker-compose -f docker-compose.edwin.yml exec edwin-api python -c "
from EduVerse.edwin.database import DatabaseManager
import asyncio

async def migrate():
    db = DatabaseManager()
    await db.initialize()
    await db.run_migrations()
    await db.close()

asyncio.run(migrate())
"
```

### 5. Create Admin User

```bash
docker-compose -f docker-compose.edwin.yml exec edwin-api python -c "
from EduVerse.edwin.auth import create_default_admin
import asyncio

asyncio.run(create_default_admin())
"
```

### 6. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "cache": "connected"
}
```

---

## Kubernetes Deployment

See [KUBERNETES.md](./KUBERNETES.md) for complete Kubernetes deployment guide.

### Quick Deploy

```bash
# 1. Create secrets
cp k8s/secrets.yaml.template k8s/secrets.yaml
# Edit secrets.yaml

# 2. Run deployment script
./scripts/deploy/deploy.sh production

# 3. Check status
kubectl get pods -n edwin

# 4. View logs
kubectl logs -n edwin -l component=api
```

---

## Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Application
EDWIN_ENV=production
LOG_LEVEL=INFO

# Database
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=your_password
QDRANT_HOST=qdrant
REDIS_HOST=redis

# Authentication
JWT_SECRET_KEY=your_secret_key
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# LLM (Optional)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key

# Features
ENABLE_ANALYTICS=true
ENABLE_CACHING=true
ENABLE_MULTIMODAL=true
```

### YAML Configuration

Production settings in `config/production.yaml`:

```yaml
environment: production

database:
  use_persistent_storage: true

rate_limit:
  enabled: true
  per_minute: 60
  per_hour: 1000

monitoring:
  prometheus:
    enabled: true
  sentry:
    enabled: true
```

---

## Database Setup

### Neo4j

**Initial Setup**:
```bash
# Access Neo4j Browser
http://localhost:7474

# Default credentials
Username: neo4j
Password: (set in NEO4J_PASSWORD)
```

**Create Indexes**:
```cypher
CREATE INDEX user_email IF NOT EXISTS FOR (u:User) ON (u.email);
CREATE INDEX student_id IF NOT EXISTS FOR (s:Student) ON (s.student_id);
CREATE INDEX objective_id IF NOT EXISTS FOR (o:LearningObjective) ON (o.objective_id);
```

### Qdrant

**Collections**:
- `edwin_curriculum`: Curriculum embeddings
- `edwin_questions`: Student question embeddings
- `edwin_explanations`: Explanation embeddings

**Verify**:
```bash
curl http://localhost:6333/collections
```

### Redis

**Verify Connection**:
```bash
redis-cli ping
# Response: PONG
```

---

## Monitoring

### Prometheus

**Access**: http://localhost:9090

**Key Metrics**:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `db_query_duration_seconds`: Database query time
- `cache_hit_rate`: Cache effectiveness

### Grafana

**Access**: http://localhost:3000
**Default Login**: admin / admin

**Pre-built Dashboards**:
1. **EdWIN API Metrics**: Request rate, latency, errors
2. **Database Performance**: Query time, connection pool
3. **User Activity**: Active users, query patterns

### Health Checks

```bash
# Run health check script
./scripts/deploy/health-check.sh

# Manual checks
curl http://localhost:8000/health        # API
curl http://localhost:8001/health        # Dashboard
curl http://localhost:8002/health        # Mobile
```

---

## Backup & Recovery

### Automated Backups

**Schedule** (via cron):
```bash
# Daily backup at 2 AM
0 2 * * * /path/to/scripts/deploy/backup.sh production /backups
```

### Manual Backup

```bash
./scripts/deploy/backup.sh edwin ./backups
```

**Backup includes**:
- Neo4j graph export
- Qdrant snapshots
- Configuration files

### Restore

```bash
# Extract backup
tar -xzf backups/edwin_backup_20251115_020000.tar.gz

# Restore Neo4j
kubectl exec -n edwin neo4j-0 -- cypher-shell < backups/neo4j_backup.cypher

# Restore Qdrant
# (Restore from snapshot files)
```

---

## Troubleshooting

### Services Won't Start

**Check logs**:
```bash
# Docker
docker-compose -f docker-compose.edwin.yml logs api

# Kubernetes
kubectl logs -n edwin -l component=api
```

**Common issues**:
1. **Database connection failed**: Check Neo4j/Qdrant/Redis are running
2. **Port already in use**: Change port in docker-compose or config
3. **Permission denied**: Check file permissions on volumes

### Database Connection Issues

**Test Neo4j**:
```bash
docker exec edwin-neo4j cypher-shell -u neo4j -p password "RETURN 1"
```

**Test Qdrant**:
```bash
curl http://localhost:6333/health
```

**Test Redis**:
```bash
docker exec edwin-redis redis-cli ping
```

### High Memory Usage

**Check resource usage**:
```bash
docker stats
```

**Adjust limits** in `docker-compose.edwin.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 2G
```

### Performance Issues

**Check metrics**:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

**Common fixes**:
1. Increase Neo4j heap size (`NEO4J_dbms_memory_heap_max__size`)
2. Enable Redis persistence (`--appendonly yes`)
3. Scale API replicas in Kubernetes

---

## Scaling

### Docker Scaling

```bash
docker-compose -f docker-compose.edwin.yml up -d --scale edwin-api=3
```

### Kubernetes Scaling

```bash
# Manual scaling
kubectl scale deployment edwin-api -n edwin --replicas=5

# Auto-scaling (HPA already configured)
kubectl get hpa -n edwin
```

---

## Security Checklist

See [SECURITY.md](./SECURITY.md) for complete security guide.

- [ ] Change default passwords
- [ ] Generate secure JWT secret
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Enable rate limiting
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Regular security updates

---

## Support

- **Documentation**: https://docs.edwin.edu
- **Issues**: https://github.com/yourorg/edwin/issues
- **Email**: support@edwin.edu

---

**Last Updated**: November 15, 2025
