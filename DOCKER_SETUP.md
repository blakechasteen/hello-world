# Docker Development Setup

Complete Docker Compose setup for local LMS development with all required services.

## Services Included

### Core Services
- **PostgreSQL** (port 5432) - Relational database
- **Neo4j** (ports 7474, 7687) - Knowledge graph database
- **Qdrant** (port 6333) - Vector database for embeddings
- **Redis** (port 6379) - Cache and task queue

### Application Services (when implemented)
- **API** (port 8000) - FastAPI backend
- **Frontend** (port 3000) - React frontend
- **Celery Worker** - Background tasks
- **Celery Beat** - Scheduled tasks

### Optional Services (use profiles)
- **Ollama** (port 11434) - Local LLM (profile: llm)
- **Prometheus** (port 9090) - Metrics (profile: monitoring)
- **Grafana** (port 3001) - Dashboards (profile: monitoring)
- **Adminer** (port 8080) - DB admin (profile: dev-tools)
- **PgAdmin** (port 5050) - PostgreSQL admin (profile: dev-tools)
- **MailHog** (ports 1025, 8025) - Email testing (profile: dev-tools)

## Quick Start

### 1. Prerequisites

```bash
# Install Docker and Docker Compose
# macOS: Docker Desktop
# Linux: docker.io and docker-compose
# Windows: Docker Desktop

# Verify installation
docker --version
docker-compose --version
```

### 2. Start Core Services

```bash
# Start all core services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Initialize Databases

```bash
# Run database migrations (when API is implemented)
docker-compose exec api alembic upgrade head

# Seed initial data
docker-compose exec api python scripts/seed_data.py

# Verify Neo4j
docker-compose exec neo4j cypher-shell -u neo4j -p lms_dev_password "RETURN 1"

# Verify Qdrant
curl http://localhost:6333/collections
```

### 4. Access Services

#### Neo4j Browser
- URL: http://localhost:7474
- Username: `neo4j`
- Password: `lms_dev_password`

#### Qdrant Dashboard
- URL: http://localhost:6333/dashboard

#### Redis
```bash
redis-cli -h localhost -p 6379
```

#### PostgreSQL
```bash
psql -h localhost -p 5432 -U lms_user -d lms_dev
# Password: lms_dev_password
```

## Profiles

Start services with specific profiles:

### Development Tools

```bash
docker-compose --profile dev-tools up -d

# Access:
# - Adminer: http://localhost:8080
# - PgAdmin: http://localhost:5050
# - MailHog: http://localhost:8025
```

### Monitoring Stack

```bash
docker-compose --profile monitoring up -d

# Access:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001 (admin/admin)
```

### LLM Support

```bash
docker-compose --profile llm up -d

# Pull a model
docker-compose exec ollama ollama pull llama3.2:3b

# Test
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Hello!"
}'
```

## Environment Variables

Create `.env` file for custom configuration:

```bash
# Database
POSTGRES_DB=lms_dev
POSTGRES_USER=lms_user
POSTGRES_PASSWORD=lms_dev_password

# Neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=lms_dev_password

# Redis
REDIS_PASSWORD=lms_dev_password

# Application
ENV=development
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=dev_secret_key_change_in_production
```

## Volume Management

### Backup Volumes

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U lms_user lms_dev > backup.sql

# Backup Neo4j
docker-compose exec neo4j neo4j-admin dump --to=/tmp/backup.dump
docker cp lms-neo4j:/tmp/backup.dump ./neo4j_backup.dump

# Backup Redis
docker-compose exec redis redis-cli --rdb /data/backup.rdb
docker cp lms-redis:/data/backup.rdb ./redis_backup.rdb
```

### Restore Volumes

```bash
# Restore PostgreSQL
docker cp backup.sql lms-postgres:/tmp/
docker-compose exec postgres psql -U lms_user -d lms_dev -f /tmp/backup.sql

# Restore Neo4j
docker cp neo4j_backup.dump lms-neo4j:/tmp/
docker-compose exec neo4j neo4j-admin load --from=/tmp/neo4j_backup.dump

# Restore Redis
docker cp redis_backup.rdb lms-redis:/data/dump.rdb
docker-compose restart redis
```

### Reset Everything

```bash
# Stop and remove all containers, networks, volumes
docker-compose down -v

# Start fresh
docker-compose up -d
```

## Performance Tuning

### PostgreSQL

Edit `docker/postgres/postgresql.conf`:

```conf
# Memory
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB

# Query Planning
random_page_cost = 1.1
effective_io_concurrency = 200

# Connections
max_connections = 100
```

### Neo4j

Edit docker-compose.yml:

```yaml
neo4j:
  environment:
    NEO4J_dbms_memory_heap_initial__size: 1G
    NEO4J_dbms_memory_heap_max__size: 4G
    NEO4J_dbms_memory_pagecache_size: 1G
```

### Redis

Edit docker-compose.yml:

```yaml
redis:
  command: >
    redis-server
    --maxmemory 256mb
    --maxmemory-policy allkeys-lru
    --appendonly yes
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose logs <service_name>

# Restart service
docker-compose restart <service_name>

# Rebuild service
docker-compose up -d --build <service_name>
```

### Port Conflicts

If ports are already in use, modify `docker-compose.yml`:

```yaml
ports:
  - "5433:5432"  # Use different host port
```

### Volume Permissions

If you get permission errors:

```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./docker/data
```

### Out of Memory

```bash
# Check Docker resources
docker stats

# Increase Docker memory
# Docker Desktop → Settings → Resources → Memory: 8GB
```

### Network Issues

```bash
# Recreate network
docker-compose down
docker network prune
docker-compose up -d
```

## Health Checks

```bash
# Check all services
docker-compose ps

# Individual health checks
docker-compose exec postgres pg_isready
docker-compose exec neo4j cypher-shell -u neo4j -p lms_dev_password "RETURN 1"
docker-compose exec redis redis-cli ping
curl http://localhost:6333/health
```

## Development Workflow

### 1. Start Services

```bash
docker-compose up -d
```

### 2. Make Changes

```bash
# Edit code (hot reload enabled)
# API: Changes auto-reload via uvicorn --reload
# Frontend: Changes auto-reload via vite
```

### 3. Run Tests

```bash
# Unit tests
docker-compose exec api pytest tests/unit/ -v

# Integration tests
docker-compose exec api pytest tests/integration/ -v

# E2E tests
docker-compose exec api pytest tests/e2e/ -v
```

### 4. View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Tail last 100 lines
docker-compose logs --tail=100 api
```

### 5. Stop Services

```bash
# Stop (keeps data)
docker-compose stop

# Down (removes containers)
docker-compose down

# Down with volumes (removes all data)
docker-compose down -v
```

## Production Deployment

For production, use separate `docker-compose.prod.yml`:

```bash
# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# With nginx reverse proxy
docker-compose --profile production up -d
```

Key production changes:
- Remove volume mounts for code
- Use environment files for secrets
- Enable HTTPS with SSL certificates
- Configure log shipping
- Set up monitoring and alerting

## CI/CD Integration

### GitHub Actions

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start services
        run: docker-compose up -d
      - name: Wait for services
        run: sleep 30
      - name: Run tests
        run: docker-compose exec -T api pytest
      - name: Cleanup
        run: docker-compose down -v
```

## Resource Requirements

### Minimum

- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB

### Recommended

- **CPU**: 8 cores
- **RAM**: 16GB
- **Disk**: 50GB SSD

### Per Service

| Service | RAM | CPU | Disk |
|---------|-----|-----|------|
| PostgreSQL | 512MB | 0.5 | 5GB |
| Neo4j | 2GB | 1.0 | 5GB |
| Qdrant | 1GB | 0.5 | 10GB |
| Redis | 256MB | 0.25 | 1GB |
| API | 512MB | 1.0 | 1GB |
| Frontend | 256MB | 0.5 | 1GB |

## Security Considerations

### Development

- Default passwords are for development only
- Services are exposed on all interfaces
- Debug mode is enabled

### Production

- [ ] Change all default passwords
- [ ] Use Docker secrets for credentials
- [ ] Enable TLS for all services
- [ ] Restrict network access
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Use non-root users

## Support

- **Documentation**: https://docs.lms.edu/docker
- **Issues**: https://github.com/lms/lms-orchestration/issues
- **Forum**: https://community.lms.edu/docker

## License

MIT License - See [LICENSE](LICENSE) for details.
