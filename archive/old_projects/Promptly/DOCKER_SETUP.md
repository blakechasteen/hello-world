# Promptly Docker Setup

Quick guide to running Promptly with persistent backends (Neo4j + Qdrant + Redis).

---

## Quick Start

### Start All Services
```bash
# Start Neo4j, Qdrant, and Redis
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Stop Services
```bash
# Stop all
docker-compose down

# Stop and remove volumes (DELETES DATA!)
docker-compose down -v
```

---

## Services

### Neo4j (Knowledge Graph)
- **URL**: http://localhost:7474
- **Bolt**: bolt://localhost:7687
- **User**: `neo4j`
- **Password**: `promptly123`

**Browser Access**:
1. Open http://localhost:7474
2. Login with credentials above
3. Run Cypher queries

### Qdrant (Vector Database)
- **API**: http://localhost:6333
- **gRPC**: localhost:6334
- **Dashboard**: http://localhost:6333/dashboard

**Health Check**:
```bash
curl http://localhost:6333/health
```

### Redis (Caching)
- **Port**: 6379
- **Connection**: `redis://localhost:6379`

**Test Connection**:
```bash
redis-cli ping
# Should return: PONG
```

---

## Using with Promptly

### Enable HoloLoom Backends

```python
from hololoom_unified import create_unified_bridge

# Use persistent backends
bridge = create_unified_bridge(
    enable_neo4j=True,
    enable_qdrant=True
)

# Store prompts
bridge.store_prompt(my_prompt)

# Search (now persistent!)
results = bridge.search_prompts("database optimization")
```

### Environment Variables

Create `.env` file:
```bash
HOLOLOOM_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=promptly123
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_URL=redis://localhost:6379
```

---

## Data Persistence

### Volumes
- `neo4j-data`: Knowledge graph data
- `qdrant-data`: Vector embeddings
- `redis-data`: Cache data

### Backup
```bash
# Backup Neo4j
docker-compose exec neo4j neo4j-admin dump --to=/data/backup.dump

# Backup Qdrant
docker-compose exec qdrant tar -czf /qdrant/storage/backup.tar.gz /qdrant/storage
```

### Restore
```bash
# Stop services
docker-compose down

# Remove old data
docker volume rm promptly_neo4j-data

# Start and restore
docker-compose up -d
docker-compose exec neo4j neo4j-admin load --from=/data/backup.dump
```

---

## Troubleshooting

### Neo4j Won't Start
```bash
# Check logs
docker-compose logs neo4j

# Common issues:
# 1. Port 7687 already in use
# 2. Not enough memory (increase heap size in docker-compose.yml)
# 3. Permission issues with volumes
```

### Qdrant Connection Failed
```bash
# Test health
curl http://localhost:6333/health

# Check logs
docker-compose logs qdrant

# Common issues:
# 1. Port 6333 already in use
# 2. Volume mount permissions
```

### Redis Not Responding
```bash
# Test connection
redis-cli ping

# Check logs
docker-compose logs redis

# Restart
docker-compose restart redis
```

---

## Performance Tuning

### Neo4j Memory
Edit `docker-compose.yml`:
```yaml
environment:
  - NEO4J_dbms_memory_heap_initial__size=1g  # Default: 512m
  - NEO4J_dbms_memory_heap_max__size=4g      # Default: 2g
```

### Qdrant Collection Settings
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Create optimized collection
client.create_collection(
    collection_name="prompts",
    vectors_config={
        "size": 384,  # Embedding dimension
        "distance": "Cosine"
    }
)
```

---

## Production Deployment

### Use Secrets
```bash
# Generate secure password
docker run --rm alpine/openssl rand -base64 32

# Update docker-compose.yml with generated password
```

### Enable SSL/TLS
```yaml
# Neo4j with SSL
environment:
  - NEO4J_dbms_connector_bolt_tls__level=REQUIRED
  - NEO4J_dbms_ssl_policy_bolt_enabled=true
```

### Resource Limits
```yaml
services:
  neo4j:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

---

## Quick Commands Reference

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f neo4j
docker-compose logs -f qdrant

# Check status
docker-compose ps

# Remove all data
docker-compose down -v

# Update images
docker-compose pull
docker-compose up -d
```

---

## Next Steps

1. **Start services**: `docker-compose up -d`
2. **Verify**: Open http://localhost:7474 (Neo4j) and http://localhost:6333/dashboard (Qdrant)
3. **Test integration**: Run `python demo_hololoom_integration.py`
4. **Check data**: Browse Neo4j graph and Qdrant collections

**Full stack now running!** 🚀
