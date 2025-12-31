# Self-Hosting HoloLoom

Deploy HoloLoom on your own infrastructure. From 5-minute quick start to production-grade deployment.

## Quick Start (5 Minutes)

**Requirements**: Docker and Docker Compose

```bash
# Clone the repository
git clone https://github.com/blake/mythRL.git
cd mythRL

# Start HoloLoom (Lite deployment)
docker-compose -f docker-compose.lite.yml up -d

# Verify it's running
curl http://localhost:8000/health

# Try a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Thompson Sampling?"}'
```

**Access Points**:
- HoloLoom API: http://localhost:8000
- Neo4j Browser: http://localhost:7474 (neo4j/hololoom)
- Qdrant Dashboard: http://localhost:6333/dashboard

**Stop**:
```bash
docker-compose -f docker-compose.lite.yml down
```

---

## Deployment Options

| Option | Services | RAM | Use Case | Complexity |
|--------|----------|-----|----------|------------|
| **[Lite](#lite-deployment)** | Neo4j + Qdrant + API | 4GB | Development, Testing | Simple |
| **[Full](#full-production-deployment)** | + PostgreSQL + Redis | 8GB | Production | Medium |
| **[Monitored](#monitored-deployment)** | + Prometheus + Grafana | 12GB | Enterprise | Medium |
| **[Kubernetes](#kubernetes-deployment)** | All services + scaling | 16GB+ | Scale/HA | Advanced |

---

## Lite Deployment

Best for development, testing, and small-scale usage.

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

### Start Services

```bash
docker-compose -f docker-compose.lite.yml up -d
```

### Verify Deployment

```bash
# Check all containers are running
docker-compose -f docker-compose.lite.yml ps

# Expected output:
# NAME                    STATUS
# hololoom-lite-neo4j     healthy
# hololoom-lite-qdrant    healthy
# hololoom-lite-api       healthy

# Test the API
curl http://localhost:8000/health
# {"status": "healthy", "version": "1.0.0"}
```

### Configure LLM (Optional)

For AI-powered features, set environment variables:

```bash
# Option 1: Use local Ollama (free)
# Install Ollama: https://ollama.ai
ollama pull llama3.2:3b

# Option 2: Use cloud providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Restart with environment
docker-compose -f docker-compose.lite.yml up -d
```

### Stop Services

```bash
# Stop (preserves data)
docker-compose -f docker-compose.lite.yml down

# Stop and delete data
docker-compose -f docker-compose.lite.yml down -v
```

---

## Full Production Deployment

For production with additional services (PostgreSQL, Redis).

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 50GB disk space
- Linux recommended (Ubuntu 22.04 LTS)

### Configuration

1. **Copy environment template**:

```bash
cp .env.example .env
```

2. **Edit `.env` with secure passwords**:

```bash
# REQUIRED - Set secure passwords
NEO4J_PASSWORD=your-secure-neo4j-password
REDIS_PASSWORD=your-secure-redis-password

# Optional - LLM API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

3. **Start services**:

```bash
docker-compose -f HoloLoom/docker-compose.yml up -d
```

### Service Endpoints

| Service | Port | Purpose |
|---------|------|---------|
| HoloLoom API | 8000 | Main API endpoint |
| Neo4j HTTP | 7474 | Graph database admin |
| Neo4j Bolt | 7687 | Graph database protocol |
| Qdrant HTTP | 6333 | Vector database API |
| Qdrant gRPC | 6334 | Vector database (fast) |
| PostgreSQL | 5432 | Relational database |
| Redis | 6379 | Cache and sessions |

### Health Checks

```bash
# All services
docker-compose -f HoloLoom/docker-compose.yml ps

# Individual service logs
docker logs hololoom-neo4j
docker logs hololoom-qdrant
docker logs hololoom-api
```

---

## Monitored Deployment

Production deployment with Prometheus metrics and Grafana dashboards.

### Start with Monitoring

```bash
# Set required environment variables
export NEO4J_PASSWORD=secure-password
export REDIS_PASSWORD=secure-password
export GRAFANA_PASSWORD=admin-password

# Start full stack with monitoring
docker-compose -f infra/docker/docker-compose.yml up -d
```

### Monitoring Endpoints

| Service | Port | Purpose |
|---------|------|---------|
| HoloLoom Metrics | 9090 | Prometheus metrics |
| Prometheus | 9092 | Metrics aggregation |
| Grafana | 3001 | Dashboards |
| Emotion Metrics | 9091 | Voice UX metrics |

### Access Grafana

1. Open http://localhost:3001
2. Login: admin / (your GRAFANA_PASSWORD)
3. Pre-configured dashboards:
   - HoloLoom Overview
   - Memory System Metrics
   - Query Performance

### Prometheus Queries

Example queries for monitoring:

```promql
# Request rate
rate(hololoom_requests_total[5m])

# Query latency P95
histogram_quantile(0.95, rate(hololoom_query_duration_seconds_bucket[5m]))

# Memory usage
hololoom_memory_usage_bytes
```

---

## Kubernetes Deployment

For high availability and horizontal scaling.

### Prerequisites

- Kubernetes 1.25+
- kubectl configured
- Helm 3.0+ (optional)
- PersistentVolume provisioner

### Quick Deploy

```bash
# Create namespace
kubectl create namespace hololoom

# Create secrets
kubectl create secret generic hololoom-secrets \
  --from-literal=neo4j-password=secure-password \
  --from-literal=redis-password=secure-password \
  -n hololoom

# Apply manifests
kubectl apply -f k8s/ -n hololoom

# Verify
kubectl get pods -n hololoom
```

### Scaling

```bash
# Scale API replicas
kubectl scale deployment hololoom-api --replicas=3 -n hololoom

# Enable HPA
kubectl apply -f k8s/hpa.yaml -n hololoom
```

### Ingress (Optional)

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hololoom-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.yourdomain.com
      secretName: hololoom-tls
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: hololoom-api
                port:
                  number: 8000
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_PASSWORD` | Production | hololoom | Neo4j password |
| `REDIS_PASSWORD` | Production | - | Redis password |
| `ANTHROPIC_API_KEY` | No | - | Claude API key |
| `OPENAI_API_KEY` | No | - | OpenAI API key |
| `OLLAMA_HOST` | No | localhost | Ollama server host |
| `HOLOLOOM_ENV` | No | development | Environment mode |
| `PROMETHEUS_ENABLED` | No | false | Enable metrics |

### LLM Configuration

HoloLoom supports multiple LLM providers:

```bash
# Local (free, private)
OLLAMA_HOST=localhost
# Pull a model: ollama pull llama3.2:3b

# Anthropic Claude (recommended)
ANTHROPIC_API_KEY=sk-ant-api03-...

# OpenAI
OPENAI_API_KEY=sk-...
```

**Priority**: Anthropic > OpenAI > Ollama (auto-fallback)

### Memory Settings

For production workloads, tune Neo4j memory:

```yaml
# docker-compose.yml
environment:
  - NEO4J_dbms_memory_heap_initial__size=1G   # Min heap
  - NEO4J_dbms_memory_heap_max__size=4G       # Max heap
  - NEO4J_dbms_memory_pagecache_size=2G       # Page cache
```

**Recommendations**:
- Development: heap=512M-1G, pagecache=512M
- Production: heap=2-4G, pagecache=2-4G
- Enterprise: heap=4-8G, pagecache=4-8G

---

## Reverse Proxy & TLS

### Traefik (Recommended)

```yaml
# docker-compose.override.yml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
      - "--certificatesresolvers.letsencrypt.acme.email=you@example.com"
      - "--certificatesresolvers.letsencrypt.acme.storage=/acme.json"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./acme.json:/acme.json

  hololoom-api:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.hololoom.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.hololoom.tls.certresolver=letsencrypt"
```

### nginx

```nginx
# /etc/nginx/sites-available/hololoom
upstream hololoom {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://hololoom;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Backup & Recovery

### Neo4j Backup

```bash
# Stop writes (optional but recommended)
docker exec hololoom-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "CALL dbms.setConfigValue('dbms.read_only', 'true')"

# Create backup
docker exec hololoom-neo4j neo4j-admin database dump neo4j --to-path=/backup
docker cp hololoom-neo4j:/backup/neo4j.dump ./backups/

# Resume writes
docker exec hololoom-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD \
  "CALL dbms.setConfigValue('dbms.read_only', 'false')"
```

### Qdrant Backup

```bash
# Create snapshot
curl -X POST "http://localhost:6333/collections/hololoom/snapshots"

# List snapshots
curl "http://localhost:6333/collections/hololoom/snapshots"

# Download snapshot
curl -o backup.snapshot \
  "http://localhost:6333/collections/hololoom/snapshots/{snapshot_name}"
```

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup.sh
set -e

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Backing up Neo4j..."
docker exec hololoom-neo4j neo4j-admin database dump neo4j --to-path=/backup
docker cp hololoom-neo4j:/backup/neo4j.dump "$BACKUP_DIR/"

echo "Backing up Qdrant..."
SNAPSHOT=$(curl -s -X POST "http://localhost:6333/collections/hololoom/snapshots" | jq -r '.result.name')
curl -o "$BACKUP_DIR/qdrant-$SNAPSHOT.snapshot" \
  "http://localhost:6333/collections/hololoom/snapshots/$SNAPSHOT"

echo "Backup complete: $BACKUP_DIR"
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Container won't start | Port conflict | Check `docker ps`, change ports in compose |
| Neo4j unhealthy | Memory limits | Increase Docker memory to 4GB+ |
| API can't connect | Network isolation | Ensure same Docker network |
| Slow queries | Insufficient cache | Increase Neo4j pagecache |
| Out of memory | Too many services | Use Lite deployment or add RAM |

### Debug Commands

```bash
# View all logs
docker-compose -f docker-compose.lite.yml logs -f

# View specific service
docker logs -f hololoom-lite-api

# Check container resources
docker stats

# Enter container shell
docker exec -it hololoom-lite-api bash

# Check network connectivity
docker exec hololoom-lite-api ping neo4j
```

### Reset Everything

```bash
# Stop all services
docker-compose -f docker-compose.lite.yml down -v

# Remove all HoloLoom images
docker images | grep hololoom | awk '{print $3}' | xargs docker rmi

# Start fresh
docker-compose -f docker-compose.lite.yml up -d --build
```

---

## Security Checklist

Before production deployment:

- [ ] **Change default passwords** - Set `NEO4J_PASSWORD`, `REDIS_PASSWORD`, `GRAFANA_PASSWORD`
- [ ] **Enable TLS** - Use reverse proxy with Let's Encrypt
- [ ] **Restrict network access** - Bind internal services to localhost
- [ ] **API authentication** - Enable API keys for production
- [ ] **Firewall rules** - Only expose port 443 (HTTPS)
- [ ] **Regular backups** - Automate daily backups
- [ ] **Monitor logs** - Set up alerting for errors
- [ ] **Update regularly** - Keep Docker images up to date

### Minimal Security Config

```bash
# .env for production
NEO4J_PASSWORD=<32-char-random-string>
REDIS_PASSWORD=<32-char-random-string>
GRAFANA_PASSWORD=<32-char-random-string>
HOLOLOOM_ENV=production
TESTING_MODE=false
```

---

## Resource Requirements

### Minimum (Lite)
- 2 CPU cores
- 4GB RAM
- 10GB disk
- Docker 20.10+

### Recommended (Production)
- 4 CPU cores
- 8GB RAM
- 50GB SSD
- Linux (Ubuntu 22.04 LTS)

### Enterprise (Monitored + Scale)
- 8+ CPU cores
- 16GB+ RAM
- 100GB+ SSD
- Kubernetes cluster

---

## Getting Help

- **Issues**: https://github.com/blake/mythRL/issues
- **Discussions**: https://github.com/blake/mythRL/discussions
- **Documentation**: See `docs/` directory

---

*Last updated: December 2025*
