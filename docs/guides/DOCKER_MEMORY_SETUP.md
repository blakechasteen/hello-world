# HoloLoom Memory Backend Setup

Quick guide to running Neo4j + Qdrant backends for HoloLoom unified memory.

## Prerequisites

- Docker Desktop installed
- Docker Compose V2

## Quick Start

```bash
# Start both backends
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Services

### Neo4j (Graph Database)

- **HTTP UI**: http://localhost:7474
- **Bolt**: bolt://localhost:7687
- **Username**: `neo4j`
- **Password**: `hololoom123`

**Test Connection:**
```bash
# Using cypher-shell
docker exec -it hololoom-neo4j cypher-shell -u neo4j -p hololoom123

# Or open browser to http://localhost:7474
```

### Qdrant (Vector Database)

- **HTTP API**: http://localhost:6333
- **gRPC API**: localhost:6334
- **Dashboard**: http://localhost:6333/dashboard

**Test Connection:**
```bash
# Check health
curl http://localhost:6333/health

# List collections
curl http://localhost:6333/collections
```

## HoloLoom Configuration

Once backends are running, use them in HoloLoom:

```python
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.weaving_shuttle import WeavingShuttle

# Configure for Neo4j + Qdrant hybrid
config = Config.fused()
config.memory_backend = MemoryBackend.NEO4J_QDRANT

# Connections use defaults from config:
# - Neo4j: bolt://localhost:7687 (neo4j/hololoom123)
# - Qdrant: http://localhost:6333

# Create memory backend
memory = await create_memory_backend(config)

# Use with WeavingShuttle
async with WeavingShuttle(cfg=config, memory=memory) as shuttle:
    spacetime = await shuttle.weave(query)
```

## Management Commands

```bash
# Stop services (keeps data)
docker-compose stop

# Start stopped services
docker-compose start

# Stop and remove containers (keeps volumes)
docker-compose down

# Remove everything including data
docker-compose down -v

# View resource usage
docker-compose stats

# Backup Neo4j data
docker exec hololoom-neo4j neo4j-admin database dump neo4j --to-path=/tmp/backup
docker cp hololoom-neo4j:/tmp/backup ./backup

# Backup Qdrant data
docker cp hololoom-qdrant:/qdrant/storage ./qdrant_backup
```

## Troubleshooting

### Neo4j won't start

```bash
# Check logs
docker-compose logs neo4j

# Common issues:
# - Port 7687 already in use
# - Insufficient memory (needs 512MB minimum)
```

### Qdrant won't start

```bash
# Check logs
docker-compose logs qdrant

# Common issues:
# - Port 6333 already in use
# - Storage permission issues
```

### Connection refused from HoloLoom

```bash
# Verify containers are running
docker-compose ps

# Check health
docker-compose ps | grep healthy

# Test connectivity
curl http://localhost:6333/health
docker exec hololoom-neo4j cypher-shell -u neo4j -p hololoom123 "RETURN 1"
```

## Data Persistence

Data is stored in named Docker volumes:
- `neo4j_data`: Graph database storage
- `neo4j_logs`: Neo4j logs
- `qdrant_data`: Vector storage

Volumes persist even when containers are removed (unless you use `docker-compose down -v`).

## Performance Tuning

### Neo4j

Edit `docker-compose.yml` to adjust memory:

```yaml
environment:
  - NEO4J_dbms_memory_heap_max__size=4G  # Increase for larger graphs
  - NEO4J_dbms_memory_pagecache_size=2G  # Increase for better caching
```

### Qdrant

Qdrant automatically adapts to available resources. For better performance:

```yaml
environment:
  - QDRANT__STORAGE__OPTIMIZERS_MEMMAP_THRESHOLD=1000000  # Larger datasets
```

## Development vs Production

### Development (Current Setup)

- Default passwords
- No authentication on Qdrant
- Exposed ports
- Local volumes

### Production Recommendations

1. **Change passwords** in environment variables
2. **Enable Qdrant API keys**:
   ```yaml
   environment:
     - QDRANT__SERVICE__API_KEY=your-secret-key
   ```
3. **Use secrets management** (Docker Swarm secrets or Kubernetes)
4. **Reverse proxy** (Nginx/Traefik) for HTTPS
5. **Network isolation** (don't expose ports publicly)
6. **Monitoring** (Prometheus + Grafana)
7. **Regular backups** (automated)

## Next Steps

1. **Start backends**: `docker-compose up -d`
2. **Run demo**: `python demos/unified_memory_demo.py`
3. **Explore data**:
   - Neo4j browser: http://localhost:7474
   - Qdrant dashboard: http://localhost:6333/dashboard

## Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)