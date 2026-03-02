# ADR-003: Three-Tier Memory Backend Architecture

**Status**: Accepted
**Date**: 2025-10-15
**Author**: HoloLoom Core Team
**Updated**: 2025-11-22 (Documentation Consolidation)

---

## Context

HoloLoom needs a flexible memory architecture that supports:
- **Development**: Fast iteration, zero setup
- **Production**: Persistent storage, scalability
- **Research**: Advanced features (gated multipass, spring dynamics)

**Requirements**:
1. Zero setup for development (no Docker/databases required)
2. Production-grade persistence (survive restarts)
3. Graceful degradation (fallback if services unavailable)
4. Performance: <200ms for hybrid retrieval (BM25 + semantic)
5. Scalability: Handle 1M+ memories

**Previous Architecture** (pre-Oct 2025):
- 10+ backend options (SQL, Neo4j, Hyperspace, etc.)
- Complex configuration
- Difficult to reason about which backend to use

---

## Decision

We will implement a **three-tier memory backend architecture** with automatic fallback:

### Three Tiers

| Tier | Backend | Setup | Persistence | Use Case |
|------|---------|-------|-------------|----------|
| **Tier 1** | INMEMORY | Zero | ❌ | Development, testing |
| **Tier 2** | HYBRID | Docker | ✅ | **Production (recommended)** |
| **Tier 3** | HYPERSPACE | Docker + config | ✅ | Research, experiments |

### INMEMORY (Tier 1)

**Technology**: NetworkX MultiDiGraph (in-memory)

**Characteristics**:
- Zero setup (always available)
- Fast (50ms queries)
- No persistence (data lost on restart)
- Limited scalability (~100K memories)

**Configuration**:
```python
from hololoom.config import Config, MemoryBackend

config = Config.fast()
config.memory_backend = MemoryBackend.INMEMORY
```

**Use Cases**:
- Development
- Unit testing
- Demos
- Prototyping

### HYBRID (Tier 2) - Production Recommended

**Technology**: Neo4j (graph) + Qdrant (vector)

**Characteristics**:
- Requires Docker (docker-compose up -d)
- Persistent storage (survives restarts)
- Good performance (150ms queries)
- Excellent scalability (1M+ memories)
- **Auto-fallback** to INMEMORY if Docker unavailable

**Configuration**:
```python
config = Config.fast()
config.memory_backend = MemoryBackend.HYBRID

# Auto-falls back to INMEMORY if Neo4j/Qdrant unavailable
```

**Docker Setup**:
```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5.9.0
    ports:
      - "7474:7474"  # Web interface
      - "7687:7687"  # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/hololoom123

  qdrant:
    image: qdrant/qdrant:v1.5.0
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC
```

**Use Cases**:
- Production deployments
- Multi-user systems
- Enterprise applications
- Long-running services

### HYPERSPACE (Tier 3) - Research Only

**Technology**: HYBRID + gated multipass + spring dynamics

**Characteristics**:
- All HYBRID features +
- Gated multipass retrieval (intelligent multi-hop)
- Spring dynamics (physics-based memory connectivity)
- Experimental features (subject to change)

**Configuration**:
```python
config = Config.fused()
config.memory_backend = MemoryBackend.HYPERSPACE
config.enable_gated_multipass = True
config.enable_spring_dynamics = True
```

**Use Cases**:
- Research experiments
- Advanced features testing
- Not recommended for production (experimental)

---

## Automatic Fallback

**Key Feature**: HYBRID automatically falls back to INMEMORY if Docker services unavailable.

```python
from hololoom.memory.backend_factory import create_memory_backend

async def main():
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID

    # Attempt HYBRID, fallback to INMEMORY if needed
    memory = await create_memory_backend(config)

    # Check actual backend
    if isinstance(memory, InMemoryBackend):
        print("⚠ Fallback to INMEMORY (Docker services unavailable)")
        print("  Start services: docker-compose up -d")
    else:
        print("✓ HYBRID backend connected")
```

**Fallback Behavior**:
1. Attempt to connect to Neo4j (7687)
2. Attempt to connect to Qdrant (6333)
3. If either fails, fall back to INMEMORY
4. Log warning with remediation instructions

**Why Fallback?**
- Developers can run HoloLoom without Docker (zero setup)
- Production deployments fail gracefully if services restart
- Tests run without external dependencies

---

## Comparison of Backends

### Performance

| Operation | INMEMORY | HYBRID | HYPERSPACE |
|-----------|----------|--------|------------|
| **Add Memory** | 5ms | 15ms | 20ms |
| **Retrieve (k=10)** | 50ms | 150ms | 200ms |
| **Graph Traversal** | 30ms | 80ms | 120ms |
| **Persistence** | ❌ | ✅ | ✅ |
| **Scalability** | 100K | 10M+ | 10M+ |

### Storage

| Backend | Graph | Vector | Metadata |
|---------|-------|--------|----------|
| **INMEMORY** | NetworkX (RAM) | NumPy (RAM) | Dict (RAM) |
| **HYBRID** | Neo4j (disk) | Qdrant (disk) | Neo4j (disk) |
| **HYPERSPACE** | Neo4j (disk) | Qdrant (disk) | Neo4j (disk) |

### Cost

| Backend | Infrastructure | Complexity | Total Cost |
|---------|----------------|------------|------------|
| **INMEMORY** | $0 | Low | **Lowest** |
| **HYBRID** | ~$50/mo (VPS) | Medium | Medium |
| **HYPERSPACE** | ~$50/mo (VPS) | High | **Highest** |

---

## Migration from Legacy Backends

**Pre-Oct 2025**: 10+ backends (SQL, Neo4j, CHROMADB, PINECONE, etc.)

**Simplified Oct 2025**: 3 backends (INMEMORY, HYBRID, HYPERSPACE)

### Migration Guide

```python
# BEFORE (legacy)
config.memory_backend = MemoryBackend.NEO4J_GRAPH  # Removed
config.memory_backend = MemoryBackend.SQL_BACKEND  # Removed
config.memory_backend = MemoryBackend.CHROMADB     # Removed

# AFTER (simplified)
config.memory_backend = MemoryBackend.INMEMORY     # Development
config.memory_backend = MemoryBackend.HYBRID       # Production
config.memory_backend = MemoryBackend.HYPERSPACE   # Research
```

**Removed Backends**:
- SQL_BACKEND → Use HYBRID
- NEO4J_GRAPH → Use HYBRID
- CHROMADB → Use HYBRID (Qdrant is better)
- PINECONE → Use HYBRID (avoid vendor lock-in)
- FAISS → Use INMEMORY (for in-memory) or HYBRID (for persistence)

**Why Simplify?**
- Reduced cognitive load (3 options instead of 10+)
- Clear decision tree: Development → INMEMORY, Production → HYBRID, Research → HYPERSPACE
- Fewer code paths to test and maintain

---

## Implementation Details

### Backend Factory

```python
from hololoom.memory.backend_factory import create_memory_backend
from hololoom.config import Config, MemoryBackend

async def create_memory_backend(config: Config):
    """Create memory backend based on configuration"""

    if config.memory_backend == MemoryBackend.INMEMORY:
        return InMemoryBackend()

    elif config.memory_backend == MemoryBackend.HYBRID:
        try:
            # Attempt Neo4j + Qdrant
            neo4j = await connect_neo4j(config)
            qdrant = await connect_qdrant(config)
            return HybridBackend(neo4j, qdrant)
        except Exception as e:
            logger.warning(f"HYBRID backend unavailable: {e}")
            logger.warning("Falling back to INMEMORY backend")
            return InMemoryBackend()

    elif config.memory_backend == MemoryBackend.HYPERSPACE:
        hybrid = await create_hybrid_backend(config)
        return HyperspaceBackend(hybrid, config)

    else:
        raise ValueError(f"Unknown backend: {config.memory_backend}")
```

### Interface Protocol

All backends implement `KGStore` protocol:

```python
class KGStore(Protocol):
    async def add_edges(self, edges: List[KGEdge]) -> None:
        """Add edges to graph"""
        ...

    async def get_neighbors(self, node: str, edge_type: Optional[str] = None) -> List[str]:
        """Get neighbors of node"""
        ...

    async def subgraph(self, nodes: List[str], hops: int = 1) -> KGStore:
        """Extract subgraph around nodes"""
        ...

    async def search(self, query: str, k: int = 10) -> List[KGEdge]:
        """Hybrid search (BM25 + semantic)"""
        ...

    async def close(self) -> None:
        """Close connections"""
        ...
```

**Why Protocol?** Allows swapping backends without changing orchestrator code.

---

## Consequences

### Positive

**✓ Zero Setup for Development**
- INMEMORY backend always available
- No Docker/databases required for simple use cases

**✓ Production-Ready Persistence**
- HYBRID backend with Neo4j + Qdrant
- Data survives restarts
- Scalable to 1M+ memories

**✓ Graceful Degradation**
- HYBRID auto-falls back to INMEMORY
- System never crashes due to missing services

**✓ Clear Decision Tree**
- 3 options instead of 10+
- Easy to choose: Development → INMEMORY, Production → HYBRID, Research → HYPERSPACE

**✓ Performance**
- INMEMORY: <50ms (development)
- HYBRID: <150ms (production)
- Both acceptable for real-time queries

### Negative

**✗ Docker Dependency for Production**
- HYBRID requires Docker (docker-compose up -d)
- Mitigated by auto-fallback to INMEMORY

**✗ Configuration Complexity**
- Neo4j + Qdrant each have ~20 config options
- Mitigated by sane defaults in docker-compose.yml

**✗ HYPERSPACE Experimental**
- Advanced features may change
- Not recommended for production

---

## Metrics

**Performance Benchmarks** (1000 queries, k=10 retrieval):

| Backend | Add Memory | Retrieve | Graph Traversal | Total |
|---------|------------|----------|-----------------|-------|
| INMEMORY | 4.2ms | 48.5ms | 28.3ms | **81.0ms** |
| HYBRID | 14.7ms | 148.2ms | 79.1ms | **242.0ms** |
| HYPERSPACE | 18.3ms | 195.6ms | 115.4ms | **329.3ms** |

**Scalability** (memory count):

| Backend | 10K | 100K | 1M | 10M |
|---------|-----|------|-----|-----|
| INMEMORY | ✓ | ✓ | ✗ (OOM) | ✗ |
| HYBRID | ✓ | ✓ | ✓ | ✓ |
| HYPERSPACE | ✓ | ✓ | ✓ | ✓ |

**Storage** (1M memories, 384D embeddings):

| Backend | Graph | Vector | Total |
|---------|-------|--------|-------|
| INMEMORY | 500MB (RAM) | 1.5GB (RAM) | **2GB (RAM)** |
| HYBRID | 300MB (disk) | 800MB (disk) | **1.1GB (disk)** |

---

## Related ADRs

- [ADR-001: Multi-Department Architecture](ADR-001-multi-department.md) - Departments use memory backend
- [ADR-002: Thompson Sampling for Routing](ADR-002-thompson-sampling.md) - Routing uses memory features

---

## References

- **Implementation**: `hololoom/memory/backend_factory.py` (231 lines, simplified from 550)
- **Backends**: `hololoom/memory/graph.py` (INMEMORY), `hololoom/memory/neo4j_graph.py` (HYBRID)
- **Tests**: `hololoom/tests/integration/test_backends_quick.py`
- **Docker**: `docker-compose.yml` (Neo4j + Qdrant)
- **Simplification Review**: `MEMORY_SIMPLIFICATION_REVIEW.md` (Task 1.3 - Oct 2025)

---

**Last Updated**: 2025-11-22 | **Status**: Production Ready | **Version**: 1.1.0 (3-Tier Architecture)
