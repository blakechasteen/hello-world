# HoloLoom Self-Hosting Guide

**Self-host your HoloLoom instance with full control over your data and infrastructure.**

> **Philosophy**: HoloLoom is designed to be self-hosted first. No phone-home, no mandatory cloud dependencies, complete data sovereignty.

## Quick Start (5 Minutes)

### Option 1: SQLite (Zero Dependencies)

The fastest way to get started - no Docker, no external databases:

```bash
# Clone and setup
git clone https://github.com/your-org/hololoom.git
cd hololoom
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run with SQLite backend (everything in-memory/local)
PYTHONPATH=. python -c "
from hololoom import hololoom
import asyncio

async def main():
    async with HoloLoom() as loom:
        await loom.experience('Hello, HoloLoom!')
        memories = await loom.recall('Hello')
        print(f'Found {len(memories)} memories')

asyncio.run(main())
"
```

### Option 2: Docker Compose (Recommended)

One command to start everything:

```bash
# Start Neo4j + Qdrant + HoloLoom API
docker-compose -f docker-compose.lite.yml up -d

# Verify services
curl http://localhost:8000/health
# {"status": "healthy"}

# Access services:
# - HoloLoom API: http://localhost:8000
# - Neo4j Browser: http://localhost:7474 (neo4j/hololoom)
# - Qdrant Dashboard: http://localhost:6333/dashboard
```

## Deployment Options

### Development (SQLite)

Best for: Local development, testing, demos

```python
from hololoom.saas import SaaSConfig, create_saas_backend

# Zero dependencies - just works
config = SaaSConfig.auth_only(
    sqlite_path="./data/my_app.db"
)
backend = create_saas_backend(config)
```

**Pros**: No setup, instant start, portable
**Cons**: Single-user, no persistence across restarts (unless you specify sqlite_path)

### Docker Compose (Recommended for Production)

Best for: Small to medium deployments, single-server setups

**Services included**:
- **Neo4j 5.14**: Graph database for knowledge storage
- **Qdrant**: Vector database for semantic search
- **PostgreSQL** (optional): For SaaS backend with user management
- **Redis** (optional): Caching layer

```bash
# Start all services
docker-compose -f hololoom/docker-compose.yml up -d

# Check health
docker-compose ps
docker-compose logs -f hololoom-neo4j
```

### Kubernetes (Enterprise Scale)

Best for: Large deployments, multi-region, high availability

See [k8s/README.md](../../k8s/README.md) for Kubernetes manifests.

```bash
# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Deploy databases
kubectl apply -f k8s/neo4j-deployment.yaml
kubectl apply -f k8s/qdrant-deployment.yaml

# Deploy HoloLoom API
kubectl apply -f k8s/hololoom-api-deployment.yaml

# Check status
kubectl get pods -n hololoom
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection string | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `hololoom123` |
| `QDRANT_HOST` | Qdrant hostname | `localhost` |
| `QDRANT_PORT` | Qdrant HTTP port | `6333` |
| `POSTGRES_HOST` | PostgreSQL hostname | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | PostgreSQL database | `hololoom_saas` |
| `POSTGRES_USER` | PostgreSQL username | `hololoom` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `hololoom123` |
| `OLLAMA_HOST` | Ollama LLM host | `localhost:11434` |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) | - |
| `OPENAI_API_KEY` | OpenAI API key (optional) | - |

### SaaS Backend Configuration

```python
from hololoom.saas import SaaSConfig, create_saas_backend

# Development (SQLite, auth only)
config = SaaSConfig.auth_only(
    sqlite_path="./data/dev.db",
    key_prefix="dev"
)

# Staging (SQLite, with usage tracking)
config = SaaSConfig.with_usage(
    sqlite_path="./data/staging.db",
    key_prefix="stg"
)

# Production (PostgreSQL, with billing)
config = SaaSConfig.with_billing(
    host="db.example.com",
    port=5432,
    database="hololoom_prod",
    user="hololoom",
    password="secure_password",
    stripe_api_key="sk_live_...",
    stripe_webhook_secret="whsec_..."
)

backend = create_saas_backend(config)
```

### Memory Backend Configuration

```python
from hololoom.config import Config, MemoryBackend

# In-memory (development)
config = Config.fast()
config.memory_backend = MemoryBackend.INMEMORY

# Hybrid (production) - Neo4j + Qdrant
config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID
```

## Database Setup

### Neo4j

```bash
# Docker
docker run -d \
  --name hololoom-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/hololoom123 \
  -v neo4j_data:/data \
  neo4j:5.14.0

# Verify
curl http://localhost:7474
```

**First-time setup**:
1. Open http://localhost:7474
2. Login with neo4j/neo4j
3. Change password to your chosen password
4. Update `NEO4J_PASSWORD` environment variable

### Qdrant

```bash
# Docker
docker run -d \
  --name hololoom-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Verify
curl http://localhost:6333/health
```

### PostgreSQL (for SaaS backend)

```bash
# Docker
docker run -d \
  --name hololoom-postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=hololoom \
  -e POSTGRES_PASSWORD=hololoom123 \
  -e POSTGRES_DB=hololoom_saas \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:16-alpine

# Verify
psql -h localhost -U hololoom -d hololoom_saas -c "SELECT 1"
```

## Health Checks

### API Health Endpoint

```bash
curl http://localhost:8000/health
# {"status": "healthy", "backend": "hybrid"}
```

### Service-Level Checks

```bash
# Neo4j
curl http://localhost:7474
# Returns HTML if healthy

# Qdrant
curl http://localhost:6333/health
# {"title":"qdrant - vector search engine","version":"..."}

# PostgreSQL
docker exec hololoom-postgres pg_isready -U hololoom
# /var/run/postgresql:5432 - accepting connections
```

### Docker Compose Health

```bash
docker-compose ps
# All services should show "healthy"

docker-compose logs --tail=50 hololoom-api
```

## Security Best Practices

### 1. Change Default Passwords

```bash
# Neo4j
NEO4J_PASSWORD=your_secure_password_here

# PostgreSQL
POSTGRES_PASSWORD=another_secure_password

# API Keys
API_KEY_SECRET=generate_a_long_random_string
```

### 2. Network Isolation

```yaml
# docker-compose.yml
services:
  hololoom-api:
    networks:
      - frontend  # Exposed to internet
      - backend   # Internal only

  neo4j:
    networks:
      - backend   # Internal only, not exposed

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access
```

### 3. TLS/HTTPS

For production, always use TLS:

```bash
# Use reverse proxy (nginx, traefik, caddy)
# Example with Caddy
caddy reverse-proxy --from https://api.example.com --to localhost:8000
```

### 4. Firewall Rules

```bash
# Only expose what's needed
ufw allow 443/tcp   # HTTPS
ufw deny 7474/tcp   # Block Neo4j browser externally
ufw deny 6333/tcp   # Block Qdrant externally
```

## Backup & Recovery

### Neo4j Backup

```bash
# Stop Neo4j first for consistent backup
docker stop hololoom-neo4j

# Backup data volume
docker run --rm \
  -v neo4j_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/neo4j-$(date +%Y%m%d).tar.gz /data

# Restart
docker start hololoom-neo4j
```

### Qdrant Backup

```bash
# Qdrant supports snapshots via API
curl -X POST http://localhost:6333/collections/memories/snapshots

# Download snapshot
curl http://localhost:6333/collections/memories/snapshots/{snapshot_name} \
  --output snapshot.tar
```

### PostgreSQL Backup

```bash
# pg_dump
docker exec hololoom-postgres pg_dump -U hololoom hololoom_saas > backup.sql

# Restore
docker exec -i hololoom-postgres psql -U hololoom hololoom_saas < backup.sql
```

## Monitoring

### Health Endpoints

HoloLoom provides comprehensive health check endpoints:

| Endpoint | Purpose | Use Case |
|----------|---------|----------|
| `GET /health` | Basic health check | Load balancers, uptime monitors |
| `GET /health/detailed` | Component-level status | Debugging, ops dashboards |
| `GET /health/features` | Feature flags status | Deployment verification |
| `GET /metrics` | Prometheus metrics | Monitoring stack |
| `GET /ready` | Kubernetes readiness | K8s readiness probe |
| `GET /live` | Kubernetes liveness | K8s liveness probe |

### Prometheus Metrics

HoloLoom exports Prometheus-compatible metrics on `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**SaaS Toolkit Metrics:**
- `hololoom_saas_up` - Service health (1=healthy, 0=unhealthy)
- `hololoom_saas_uptime_seconds` - Service uptime in seconds
- `hololoom_saas_requests_total` - Total requests processed
- `hololoom_saas_errors_total` - Total errors
- `hololoom_saas_feature_enabled{feature="..."}` - Feature flag status
- `hololoom_saas_info{backend="..."}` - Service info labels

**Core HoloLoom Metrics:**
- `hololoom_queries_total` - Total queries processed
- `hololoom_query_latency_ms` - Query latency histogram
- `hololoom_memory_operations_total` - Memory operations
- `hololoom_backend_health` - Backend health status

### Grafana Dashboard

Import the SaaS health dashboard:

```bash
# SaaS-specific dashboard
hololoom/saas/dashboards/saas_health.json

# ChatOps job monitoring
hololoom/chatops/dashboards/hololoom_jobs.json
```

**Dashboard Panels:**
- Service Health (stat) - Current health status
- Uptime (stat) - Hours since last restart
- Backend Type (stat) - postgresql or sqlite
- Error Rate (gauge) - Percentage of failed requests
- Request Rate (timeseries) - Requests/min over time
- Feature Flags (table) - Enabled/disabled features
- Uptime History (timeseries) - Health over time

### Prometheus Scrape Config

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'hololoom-saas'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Log Aggregation

```bash
# Docker logs to stdout
docker-compose logs -f hololoom-api

# Or configure JSON logging for aggregation
# (ELK, Loki, CloudWatch, etc.)
```

## Troubleshooting

### "Connection refused" to Neo4j

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Check logs
docker logs hololoom-neo4j

# Common fix: Wait for startup
# Neo4j can take 30-60 seconds to fully start
```

### "No healthy upstream" from Qdrant

```bash
# Check Qdrant health
curl http://localhost:6333/health

# Restart Qdrant
docker restart hololoom-qdrant
```

### "Module not found" errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/path/to/hololoom

# Or run with explicit path
PYTHONPATH=. python -m hololoom.server.agentic_api
```

### Memory issues

```bash
# Increase Neo4j heap
-e NEO4J_dbms_memory_heap_max__size=4G

# Increase Qdrant limits
-e QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=64
```

### Port conflicts

```bash
# Check what's using a port
lsof -i :7474
netstat -tulpn | grep 7474

# Use different ports
docker run -p 17474:7474 -p 17687:7687 neo4j:5.14.0
```

## Upgrading

### Minor Version Upgrades

```bash
# Pull new images
docker-compose pull

# Restart with new versions
docker-compose up -d
```

### Major Version Upgrades

1. Backup all data (see Backup section)
2. Stop services
3. Update docker-compose.yml with new versions
4. Start services
5. Run any migration scripts
6. Verify functionality

```bash
# Example upgrade path
docker-compose down
# Edit docker-compose.yml to update versions
docker-compose up -d
# Check logs for migration messages
docker-compose logs -f
```

## Support

- **Documentation**: https://github.com/your-org/hololoom/docs
- **Issues**: https://github.com/your-org/hololoom/issues
- **Discussions**: https://github.com/your-org/hololoom/discussions

## License

MIT License - Use freely in your own deployments.
