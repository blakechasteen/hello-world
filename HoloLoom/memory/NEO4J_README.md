# Neo4j Knowledge Graph Integration

This guide explains how to use Neo4j as a backend for HoloLoom's knowledge graph, replacing the in-memory NetworkX implementation with a production-grade graph database.

## Why Neo4j?

**NetworkX (default):**
- ✅ Fast for small graphs (<10k nodes)
- ✅ No external dependencies
- ✅ Easy to debug and visualize
- ❌ In-memory only (no persistence)
- ❌ Doesn't scale to large graphs
- ❌ Single-threaded access

**Neo4j:**
- ✅ Persistent storage with ACID guarantees
- ✅ Scales to millions of nodes
- ✅ Advanced graph algorithms (PageRank, community detection, etc.)
- ✅ Concurrent multi-user access
- ✅ Production-grade with enterprise support
- ✅ Cypher query language for complex patterns
- ❌ Requires external database setup
- ❌ Slightly slower for very small graphs

## Quick Start

### 1. Start Neo4j with Docker

```bash
# Start Neo4j container
docker-compose up -d

# Check logs
docker-compose logs -f neo4j

# Wait for "Started." message
```

### 2. Access Neo4j Browser

Open [http://localhost:7474](http://localhost:7474) in your browser.

**Login credentials:**
- Username: `neo4j`
- Password: `hololoom123`

### 3. Use Neo4j in Python

```python
from holoLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
from holoLoom.memory.graph import KGEdge

# Create Neo4j knowledge graph
config = Neo4jConfig()
kg = Neo4jKG(config)

# Add edges
kg.add_edge(KGEdge("attention", "transformer", "USES", 1.0))
kg.add_edge(KGEdge("transformer", "neural_network", "IS_A", 1.0))

# Query
neighbors = kg.get_neighbors("attention", direction="both")
print(f"Neighbors of attention: {neighbors}")

# Get subgraph (returns NetworkX graph for compatibility)
subgraph = kg.subgraph_for_entities(["attention", "transformer"], expand=True)

# Close connection
kg.close()
```

### 4. Configure HoloLoom to Use Neo4j

```python
from holoLoom.config import Config, KGBackend

cfg = Config(
    kg_backend=KGBackend.NEO4J,
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="hololoom123"
)

# Use config with orchestrator
# orchestrator = Orchestrator(config=cfg)
```

## Migration Guide

### Migrate Existing NetworkX Graph to Neo4j

```bash
# Export NetworkX graph to JSONL (if not already saved)
python -c "
from holoLoom.memory.graph import KG
kg = KG()
# ... populate graph ...
kg.save('my_graph.jsonl')
"

# Migrate to Neo4j
python -m holoLoom.memory.migrate_to_neo4j \
    --from-networkx my_graph.jsonl \
    --clear

# Verify migration
python -m holoLoom.memory.migrate_to_neo4j \
    --verify my_graph.jsonl
```

### Export Neo4j Graph to NetworkX

```bash
# Export from Neo4j to JSONL
python -m holoLoom.memory.migrate_to_neo4j \
    --from-neo4j \
    --output exported_graph.jsonl
```

### Merge Multiple Graphs

```bash
# Merge NetworkX graphs
python -m holoLoom.memory.migrate_to_neo4j \
    --merge graph1.jsonl graph2.jsonl graph3.jsonl \
    --output merged_graph.jsonl

# Merge directly into Neo4j
python -m holoLoom.memory.migrate_to_neo4j \
    --merge graph1.jsonl graph2.jsonl \
    --to-neo4j \
    --clear
```

## Advanced Features

### Custom Neo4j Configuration

```python
from holoLoom.memory.neo4j_graph import Neo4jConfig

config = Neo4jConfig(
    uri="bolt://production-server:7687",
    username="hololoom",
    password="secure_password_123",
    database="knowledge_graph",
    max_connection_pool_size=100,
    connection_timeout=60.0
)

kg = Neo4jKG(config)
```

### Environment Variables

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="hololoom123"
export NEO4J_DATABASE="neo4j"
```

```python
from holoLoom.memory.neo4j_graph import Neo4jConfig

# Load from environment
config = Neo4jConfig.from_env()
kg = Neo4jKG(config)
```

### Raw Cypher Queries

```python
# Execute custom Cypher
results = kg.run_cypher("""
    MATCH (e:Entity)-[r:RELATES]->(other:Entity)
    WHERE r.type = 'IS_A'
    RETURN e.name AS entity, other.name AS parent, r.weight AS confidence
    ORDER BY r.weight DESC
    LIMIT 10
""")

for record in results:
    print(f"{record['entity']} IS_A {record['parent']} (confidence: {record['confidence']})")
```

### Graph Algorithms (requires Graph Data Science plugin)

```python
# Compute PageRank centrality
rankings = kg.pagerank(limit=10)
for entity, score in rankings:
    print(f"{entity}: {score:.4f}")
```

### Time-Based Queries

```python
from datetime import datetime

# Connect entity to time bucket
thread_id = kg.connect_entity_to_time(
    entity="user_logged_in",
    timestamp=datetime.now(),
    edge_type="OCCURRED_AT"
)

# Query by time bucket
results = kg.run_cypher("""
    MATCH (e:Entity)-[:RELATES {type: 'OCCURRED_AT'}]->(t:TimeThread)
    WHERE t.bucket CONTAINS '2024-01'
    RETURN e.name AS event, t.bucket AS time_bucket
""")
```

## Performance Tips

### 1. Bulk Imports

```python
# Use add_edges() for bulk operations (much faster)
edges = [
    KGEdge("entity1", "entity2", "RELATES", 1.0),
    KGEdge("entity2", "entity3", "RELATES", 0.8),
    # ... thousands more ...
]

kg.add_edges(edges)  # Batched transaction
```

### 2. Indexes and Constraints

Indexes are created automatically on first connection:
- Unique constraint on `Entity.name`
- Index on `TimeThread.bucket`
- Full-text index on entity names (if available)

### 3. Connection Pooling

```python
config = Neo4jConfig(
    max_connection_pool_size=50,  # Increase for concurrent access
    max_connection_lifetime=3600   # Recycle connections after 1 hour
)
```

## Troubleshooting

### Connection Refused

```
Failed to connect to Neo4j at bolt://localhost:7687
```

**Solution:** Ensure Neo4j is running:
```bash
docker-compose ps
docker-compose up -d  # If not running
```

### Authentication Failed

```
AuthError: The client is unauthorized due to authentication failure.
```

**Solution:** Check credentials in `docker-compose.yml` and your config:
```yaml
environment:
  - NEO4J_AUTH=neo4j/hololoom123
```

### Database Not Found

```
Unable to get a routing table for database 'wrong_db'
```

**Solution:** Use default database `neo4j` or create a new one:
```cypher
CREATE DATABASE my_database;
```

### Performance Issues

If queries are slow:

1. **Check indexes:**
   ```cypher
   SHOW INDEXES
   ```

2. **Analyze query plan:**
   ```cypher
   EXPLAIN MATCH (e:Entity {name: 'attention'})-[:RELATES*1..2]->(other)
   RETURN other.name
   ```

3. **Use batching for bulk operations:**
   ```python
   kg.add_edges(edges)  # Instead of multiple add_edge() calls
   ```

## Docker Compose Configuration

The provided `docker-compose.yml` includes:

- **Image:** `neo4j:5.15-community`
- **Ports:**
  - `7474`: HTTP (Browser interface)
  - `7687`: Bolt protocol (Python driver)
- **Volumes:** Persistent storage for data, logs, imports, plugins
- **APOC Plugin:** Advanced graph procedures enabled
- **Authentication:** `neo4j/hololoom123`

### Production Deployment

For production, consider:

1. **Use strong passwords:**
   ```yaml
   environment:
     - NEO4J_AUTH=neo4j/$(openssl rand -base64 32)
   ```

2. **Enable SSL/TLS:**
   ```yaml
   environment:
     - NEO4J_dbms_ssl_policy_bolt_enabled=true
   ```

3. **Resource limits:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
         cpus: '2'
   ```

4. **Backup volumes:**
   ```bash
   docker run --rm \
     -v hololoom_neo4j_data:/data \
     -v $(pwd)/backups:/backups \
     ubuntu tar czf /backups/neo4j-backup-$(date +%Y%m%d).tar.gz /data
   ```

## KGStore Protocol Compatibility

`Neo4jKG` implements the same `KGStore` protocol as `KG` (NetworkX), so it's a drop-in replacement:

```python
from holoLoom.memory.graph import KG
from holoLoom.memory.neo4j_graph import Neo4jKG

# Both implement the same interface
kg_networkx = KG()
kg_neo4j = Neo4jKG()

# All methods work the same
kg_networkx.add_edge(edge)
kg_neo4j.add_edge(edge)

kg_networkx.get_neighbors("entity")
kg_neo4j.get_neighbors("entity")

kg_networkx.subgraph_for_entities(["a", "b"])
kg_neo4j.subgraph_for_entities(["a", "b"])
```

## Next Steps

1. **Explore the Browser:** Use Neo4j Browser at [http://localhost:7474](http://localhost:7474) to visualize your graph
2. **Learn Cypher:** Neo4j's query language - [Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
3. **Install GDS Plugin:** For advanced graph algorithms - [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)
4. **Monitor Performance:** Use `PROFILE` and `EXPLAIN` to optimize queries
5. **Scale Up:** Consider Neo4j Enterprise for clustering and advanced features

## Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [APOC Procedures](https://neo4j.com/labs/apoc/)
- [Graph Data Science Library](https://neo4j.com/docs/graph-data-science/current/)
- [Cypher Cheat Sheet](https://neo4j.com/docs/cypher-cheat-sheet/)

---

**Happy Graph Weaving!** 🕸️
