# Memory Backend System

HoloLoom supports multiple memory backends through a unified `MemoryStore` protocol. Choose the right backend for your use case, or combine them with hybrid strategies.

## Available Backends

| Backend | Best For | Latency | Persistence |
|---------|----------|---------|-------------|
| InMemory | Dev/testing, session state | ~0ms | No |
| Neo4j | Relationships, temporal patterns | ~5ms | Yes |
| Qdrant | Semantic similarity search | ~3ms | Yes |
| Hybrid | Production (best of all) | ~8ms | Yes |

## Usage

### InMemory (Default)

```python
from hololoom.core.memory.stores import InMemoryStore

store = InMemoryStore()
# Zero setup, zero latency, lost on restart
```

### Neo4j (Graph Database)

Best for "who did what where when" — relationship reasoning.

```python
from hololoom.core.memory.stores import Neo4jStore

store = Neo4jStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="hololoom123",
)
```

Stores memories as graph nodes with THREAD edges (PLACE, ACTOR, TIME, THEME). Enables graph traversal queries and temporal patterns.

### Qdrant (Vector Database)

Best for "find similar content" — semantic similarity.

```python
from hololoom.core.memory.stores import QdrantStore

store = QdrantStore(host="localhost", port=6333, scales=[96, 192, 384])
```

Multi-scale Matryoshka embeddings enable coarse-to-fine retrieval.

### Hybrid (Neo4j + Qdrant)

Best overall quality — combines graph relationships with vector similarity.

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

## Retrieval Strategies

```python
from hololoom.core.memory.awareness_types import ActivationStrategy

# Semantic similarity (default)
results = await store.retrieve(query, strategy=ActivationStrategy.SEMANTIC)

# Graph traversal (Neo4j)
results = await store.retrieve(query, strategy=ActivationStrategy.GRAPH)

# Time-based
results = await store.retrieve(query, strategy=ActivationStrategy.TEMPORAL)

# Combined (best coverage)
results = await store.retrieve(query, strategy=ActivationStrategy.FUSED)
```

## Smart Routing

Route queries to the optimal backend automatically:

```python
async def smart_retrieve(query: str) -> list:
    query_lower = query.lower()

    if any(w in query_lower for w in ['who', 'when', 'where', 'related']):
        return await neo4j.retrieve(query, strategy="graph")
    elif any(w in query_lower for w in ['similar', 'like', 'find']):
        return await qdrant.retrieve(query, strategy="semantic")
    elif any(w in query_lower for w in ['recent', 'latest', 'today']):
        return await cache.retrieve(query, strategy="temporal")
    else:
        return await hybrid.retrieve(query, strategy="fused")
```

## Auto-Fallback

The system automatically degrades if a backend is unavailable:

```
HYBRID → NEO4J_ONLY → QDRANT_ONLY → INMEMORY
```

No single point of failure. Archive instead of delete. Never lose user data.

## Docker Setup

See [DOCKER_MEMORY_SETUP.md](DOCKER_MEMORY_SETUP.md) for starting Neo4j and Qdrant via Docker Compose.
