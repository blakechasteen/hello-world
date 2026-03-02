# Backend Usage Patterns

Strategic patterns for leveraging each memory backend's strengths.

## Backend Specialization

| Backend | Strength | Query Types |
|---------|----------|-------------|
| **Neo4j** | Relationship reasoning | "Who did what where when" |
| **Qdrant** | Content similarity | "Find similar content" |
| **Mem0** | User personalization | "My preferences" |
| **InMemory** | Speed | "What happened just now" |

## Multi-Backend Storage

Route memories to the right backend based on content:

```python
async def intelligent_store(memory: Memory) -> dict:
    results = {}

    # Always cache for speed
    results['cache'] = await cache.store(memory)

    # Neo4j if rich relationships (people, places, time)
    if memory.context.get('people') or memory.context.get('place'):
        results['neo4j'] = await neo4j.store(memory)

    # Qdrant if content-heavy
    if len(memory.text) > 50:
        results['qdrant'] = await qdrant.store(memory)

    return results
```

## Query-Optimized Retrieval

```python
async def smart_retrieve(query: str) -> list:
    query_lower = query.lower()

    if any(w in query_lower for w in ['who', 'where', 'when', 'related']):
        return await neo4j.retrieve(query, strategy="graph")

    elif any(w in query_lower for w in ['similar', 'like', 'find']):
        return await qdrant.retrieve(query, strategy="semantic")

    elif any(w in query_lower for w in ['recent', 'latest', 'today']):
        return await cache.retrieve(query, strategy="temporal")

    else:
        return await hybrid.retrieve(query, strategy="fused")
```

## Production Architectures

### Speed-First

```python
# Cache-first with vector fallback
cache_results = await cache.retrieve(query)
if not cache_results:
    return await qdrant.retrieve(query)
```

### Intelligence-First

```python
# LLM-powered extraction + relationship enrichment
mem0_results = await mem0.retrieve(query)
graph_results = await neo4j.retrieve(query)
return fuse_results(mem0_results, graph_results)
```

### Relationship-First

```python
# Graph traversal + content expansion
graph_results = await neo4j.retrieve(query, strategy="graph")
if len(graph_results) < 5:
    vector_results = await qdrant.retrieve(query)
    return fuse_results(graph_results, vector_results)
return graph_results
```

## Graceful Degradation

The protocol-based design means any backend can fail without crashing the system. Auto-fallback chain: `HYBRID -> NEO4J -> QDRANT -> INMEMORY`.
