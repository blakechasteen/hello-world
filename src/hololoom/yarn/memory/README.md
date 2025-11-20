# HoloLoom Memory System

**Purpose**: Unified memory architecture with multiple backend options
**Key Concept**: Knowledge graph + vector store with auto-fallback
**Status**: Production-ready (9.2/10)

## Overview

The memory system provides persistent storage of knowledge with three backend options:
- **INMEMORY**: NetworkX in-memory graph (development, always works)
- **HYBRID**: Neo4j + Qdrant with auto-fallback (production, recommended)
- **HYPERSPACE**: Advanced gated multipass (research only)

## Architecture

```
Memory System
├── protocol.py          # Memory protocols (120 lines)
├── backend_factory.py   # Backend creation (231 lines)
├── graph.py            # NetworkX KG (default, always works)
├── neo4j_graph.py      # Production backend
├── hyperspace_backend.py  # Research backend
├── cache.py            # BM25 + semantic retrieval
└── unified.py          # Unified interface
```

## Key Files

### protocol.py (120 lines)
**Purpose**: Protocol definitions for memory backends
**Key Classes**: `KGStore`, `Retriever`, `MemoryBackend`

### backend_factory.py (231 lines)
**Purpose**: Create memory backends with auto-fallback
**Key Function**: `create_memory_backend(config)`

### graph.py
**Purpose**: Default NetworkX knowledge graph (always works)
**Key Classes**: `KG` (alias for `YarnGraph`)

### cache.py
**Purpose**: BM25 + semantic retrieval with caching
**Key Classes**: `RetrieverMS`, `MemoryManager`

## Usage

### Basic Usage (INMEMORY)
```python
from HoloLoom.config import Config
from HoloLoom.memory.backend_factory import create_memory_backend

config = Config.bare()
config.memory_backend = MemoryBackend.INMEMORY

memory = await create_memory_backend(config)
# Use with WeavingOrchestrator
```

### Production Usage (HYBRID with auto-fallback)
```python
config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID

# Automatically falls back to INMEMORY if Neo4j/Qdrant unavailable
memory = await create_memory_backend(config)
```

### Docker Setup
```bash
docker-compose up -d  # Start Neo4j + Qdrant
```

See [DOCKER_MEMORY_SETUP.md](../DOCKER_MEMORY_SETUP.md) for details.

## Backend Comparison

| Backend | Storage | Speed | Persistence | Use Case |
|---------|---------|-------|-------------|----------|
| **INMEMORY** | NetworkX | Fast | No | Development, testing |
| **HYBRID** | Neo4j + Qdrant | Medium | Yes | Production (recommended) |
| **HYPERSPACE** | Advanced | Slow | Yes | Research only |

## Auto-Fallback

HYBRID automatically falls back to INMEMORY if:
- Neo4j not available (Docker not running)
- Qdrant not available
- Connection errors

```python
# Will use HYBRID if available, otherwise INMEMORY
memory = await create_memory_backend(config)
# ↓
# Neo4j unavailable -> falls back to INMEMORY
# ✅ System continues working
```

## Key Features

### Knowledge Graph
- Typed edges (IS_A, USES, MENTIONS, etc.)
- Subgraph extraction for context expansion
- Path finding between entities
- Spectral graph features for policy input

### Vector Retrieval
- BM25 text search
- Semantic similarity (Matryoshka embeddings)
- Multi-scale fusion
- Caching for performance

## Testing

### Unit Tests
```bash
pytest HoloLoom/tests/unit/test_memory_graph.py -v  # 80+ assertions
pytest HoloLoom/tests/unit/test_memory_cache.py -v  # 70+ assertions
```

### Integration Tests
```bash
pytest HoloLoom/tests/integration/test_backends.py -v
```

## Simplification (Oct 2025)

**Before**: 10+ backend enums (NETWORKX, NEO4J, QDRANT, MEM0, etc.)
**After**: 3 core backends (INMEMORY, HYBRID, HYPERSPACE)

**Impact**: -58% code in backend_factory.py (550 → 231 lines)

See [MEMORY_SIMPLIFICATION_REVIEW.md](../MEMORY_SIMPLIFICATION_REVIEW.md) for details.

## Performance

| Operation | INMEMORY | HYBRID | HYPERSPACE |
|-----------|----------|--------|------------|
| **Add edge** | <1ms | ~5ms | ~10ms |
| **Search** | ~10ms | ~20ms | ~30ms |
| **Subgraph** | ~5ms | ~15ms | ~25ms |

## Future Enhancements

- [ ] Memory compression for large graphs
- [ ] Distributed memory across multiple Neo4j instances
- [ ] Real-time memory streaming
- [ ] Memory versioning and rollback

## Related Documentation

- [UNIFIED_MEMORY_INTEGRATION.md](../UNIFIED_MEMORY_INTEGRATION.md)
- [DOCKER_MEMORY_SETUP.md](../DOCKER_MEMORY_SETUP.md)
- [MEMORY_SIMPLIFICATION_REVIEW.md](../MEMORY_SIMPLIFICATION_REVIEW.md)

---

**Status**: Production-ready
**Last Updated**: November 2, 2025
**Maintainer**: HoloLoom team
