# Docker Memory Setup

Start Neo4j and Qdrant backends for persistent memory.

## Quick Start

```bash
cd config
docker-compose up -d
```

## Services

### Neo4j (Graph Database)

- Browser: http://localhost:7474
- Bolt: `bolt://localhost:7687`
- Credentials: `neo4j` / `hololoom123`
- APOC plugin enabled

```python
from hololoom.core.memory.stores import Neo4jStore

store = Neo4jStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="hololoom123",
)
```

### Qdrant (Vector Database)

- REST: http://localhost:6333
- gRPC: `localhost:6334`

```python
from hololoom.core.memory.stores import QdrantStore

store = QdrantStore(host="localhost", port=6333)
```

### Hybrid (Best Quality)

```python
from hololoom.core.memory.stores import HybridNeo4jQdrant

store = HybridNeo4jQdrant(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="hololoom123",
    qdrant_host="localhost",
    qdrant_port=6333,
)
```

## Management

```bash
docker-compose ps          # Status
docker-compose logs -f     # Logs
docker-compose down        # Stop
docker-compose down -v     # Stop + delete data
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Connection refused | `docker-compose up -d` then wait ~10s |
| Neo4j unauthorized | Check credentials in `docker-compose.yml` |
| Port conflict | Change ports in `docker-compose.yml` |
| Volume permissions | `sudo chown -R $(whoami):$(whoami) ~/.docker` |

## Performance Tuning

Neo4j: Set `NEO4J_dbms_memory_heap_max__size=2G` in docker-compose for large graphs.

Qdrant: Default config handles millions of vectors. For 10M+, increase `--max-threads`.
