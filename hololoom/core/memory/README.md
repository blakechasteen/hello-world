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

## LiteMemoryBus

Pure-Python 4-level cascade memory bus. No external databases required.
Drop-in for development, testing, or single-session use.

```
lite_bus.py              # 4-level cascade: Exact → Structured → Graph → Semantic
├── bus.py               # MemoryQuery, MemoryItem, MemoryResult (shared types)
├── bus_config.py         # MemoryBusConfig, PressureTier, ResolutionPath
└── hybrid_retrieval.py  # BM25 + Semantic + Graph + RRF fusion (HybridRetriever)
```

### Basic Usage

```python
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.bus import MemoryQuery, MemoryItem

async with LiteMemoryBus() as bus:
    # Store
    item_id = await bus.store(MemoryItem(
        content="Garlic is aromatic and essential in French cooking",
        memory_type="factual",
        importance=0.7,
    ))

    # Query (cascades through L1-L4 automatically)
    result = await bus.query(MemoryQuery(intent="garlic recipes"))
    for item in result.items:
        print(item["content"])

    # Persist
    bus.save_snapshot("memory.json")
```

## Borrowed Features (March 2026)

Six opt-in augmentations borrowed from open-source memory projects.
All are composable — pass them as constructor parameters to `LiteMemoryBus`.

```
Borrowed Features
├── retrieval_planner.py   # Intent-aware query planning (from SimpleMem)
├── version_log.py         # Append-only delta log with rollback (from Letta/GCC)
├── synthesis.py           # Write-time dedup/merge (from SimpleMem)
├── memfs.py               # Virtual markdown filesystem view (from Letta MemFS)
├── sleep_consolidator.py  # Background consolidation + promotion (from Letta/GCC)
└── security_screen.py     # Sensitive data blocking (from Repomix)
```

### Retrieval Planner

Classifies query complexity to skip expensive cascade levels.

```python
from hololoom.core.memory.retrieval_planner import RetrievalPlanner

bus = LiteMemoryBus(retrieval_planner=RetrievalPlanner())

# entity_ids query → POINT (L1-2 only, skips graph + semantic)
# intent + entity_type → NEIGHBORHOOD (L1-3, skips semantic)
# bare intent → EXPLORATORY (all 4 levels)
```

| Query Shape | Complexity | Cascade Levels |
|------------|-----------|---------------|
| `entity_ids` present | POINT | L1-2 |
| Structured filters only | POINT | L1-2 |
| Intent + structured | NEIGHBORHOOD | L1-3 |
| Free text only | EXPLORATORY | L1-4 |

### Version Log

Append-only delta log for every mutation. Supports replay, rollback, diff, milestones.

```python
from hololoom.core.memory.version_log import VersionLog

vlog = VersionLog()
bus = LiteMemoryBus(version_log=vlog)

# Every store/forget is recorded as a MemoryDelta
await bus.store(MemoryItem(content="fact", memory_type="factual"))

# Replay all deltas
deltas = vlog.replay()

# Rollback to any point
snapshot = vlog.rollback(deltas[0].delta_id)

# Diff between two points
changes = vlog.diff(delta_a.delta_id, delta_b.delta_id)

# Milestones (GCC COMMIT equivalent)
vlog.record_milestone("Pre-consolidation checkpoint")

# Persists alongside snapshots
bus.save_snapshot("memory.json")  # Also saves memory.deltas.json
```

### Write-Time Synthesis

Proactive dedup/merge at store time. Checks for similar existing items before storing.

```python
from hololoom.core.memory.synthesis import default_synthesis_fn

bus = LiteMemoryBus(
    semantic_fn=my_embedding_search,  # Required for finding similar items
    synthesis_fn=default_synthesis_fn(),
)

# Storing duplicate content → merges into existing item (returns existing ID)
# Storing related content → stores new + creates SIMILAR_TO edge
# Storing unrelated content → stores normally
```

**Actions**: `store_new` | `merge_into` | `link_to` | `supersedes`

The default heuristic uses substring matching. Pass a custom `synthesis_fn` for
embedding-based similarity.

### MemFS

Read-only virtual filesystem view for human inspection and LLM prompt injection.

```python
from hololoom.core.memory.memfs import MemFS

fs = MemFS(bus)

# File tree overview
print(fs.tree())
# memory/
#   entity/ (5 items, ~200 tokens)
#     ingredient/ (3 items, ~120 tokens)
#   episodic/ (12 items, ~800 tokens)

# Render single item with YAML frontmatter
mf = fs.render(item_id)
print(mf.render())
# ---
# importance: 0.8
# memory_type: entity
# token_estimate: 45
# ---
# Garlic is aromatic...

# Progressive disclosure for LLM system prompts
prompt = fs.render_for_prompt("garlic recipes", token_budget=2000)
# 1. Tree header (always)
# 2. Pinned high-importance items (up to 40% budget)
# 3. Query-relevant items (remaining budget)
```

### Sleep Consolidator

Background memory maintenance. Composes existing `bus.consolidate()` with
milestone promotion and version logging.

```python
from hololoom.core.memory.sleep_consolidator import SleepConsolidator

sc = SleepConsolidator(
    bus,
    version_log=vlog,
    milestone_access_threshold=5,     # Access count to qualify for promotion
    milestone_importance_threshold=0.7, # Importance to qualify
    consolidate_max_items=10,          # Per memory_type before consolidation
)

# Run one cycle
stats = await sc.run_cycle()
# stats["consolidated"]  → which types were consolidated
# stats["promoted"]      → item IDs promoted to PERMANENT (ttl=None)

# Or run as background task (every 30 minutes)
await sc.start_background(interval_seconds=1800)
await sc.stop_background()
```

### Security Screen

Blocks sensitive data (API keys, tokens, SSNs, private keys) from entering the store.

```python
from hololoom.core.memory.security_screen import (
    default_security_screen,
    create_security_screen,
    SecurityScreenError,
)

bus = LiteMemoryBus(security_screen=default_security_screen)

# This raises SecurityScreenError:
await bus.store(MemoryItem(content="sk-abc123...", memory_type="factual"))

# Custom patterns
screen = create_security_screen(
    extra_patterns=[("internal_id", r"INT-\d{8}", "Internal ID")],
    disabled_patterns=["ssn"],
)
bus = LiteMemoryBus(security_screen=screen)
```

**Default patterns**: AWS keys, OpenAI/Anthropic keys, GitHub tokens, generic
secrets (`password = "..."`), US SSNs, private keys.

### All Features Together

```python
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.retrieval_planner import RetrievalPlanner
from hololoom.core.memory.version_log import VersionLog
from hololoom.core.memory.synthesis import default_synthesis_fn
from hololoom.core.memory.security_screen import default_security_screen
from hololoom.core.memory.memfs import MemFS
from hololoom.core.memory.sleep_consolidator import SleepConsolidator

vlog = VersionLog()
bus = LiteMemoryBus(
    retrieval_planner=RetrievalPlanner(),
    version_log=vlog,
    synthesis_fn=default_synthesis_fn(),
    semantic_fn=my_embedding_search,
    security_screen=default_security_screen,
)
await bus.initialize()

# Store, query, consolidate...
fs = MemFS(bus)
sc = SleepConsolidator(bus, version_log=vlog)
```

### StoredItem.token_estimate

Every item gets a pre-computed `token_estimate` at store time (~`len(content) // 4`
plus properties). Used by token budgeting in `_build_result()` and MemFS rendering.
Persisted in v2 snapshots; v1 snapshots are backfilled on load.

## Testing

### Unit Tests
```bash
pytest tests/unit/test_retrieval_planner.py -v   # 21 tests
pytest tests/unit/test_version_log.py -v          # 22 tests
pytest tests/unit/test_synthesis.py -v            # 14 tests
pytest tests/unit/test_memfs.py -v                # 13 tests
pytest tests/unit/test_sleep_consolidator.py -v   # 11 tests
pytest tests/unit/test_security_screen.py -v      # 13 tests
```

### Integration Tests
```bash
pytest tests/integration/test_borrowed_features.py -v  # 2 tests (full round-trip)
pytest tests/integration/test_backends.py -v
```

## Future Enhancements

- [ ] Memory compression for large graphs
- [ ] Distributed memory across multiple Neo4j instances
- [ ] Real-time memory streaming
- [x] Memory versioning and rollback (version_log.py, March 2026)
- [ ] Git backend for VersionLog (deferred — `backends/git_version_backend.py`)
- [ ] LLM-powered synthesis_fn (cosine similarity instead of substring matching)

## Related Documentation

- [UNIFIED_MEMORY_INTEGRATION.md](../UNIFIED_MEMORY_INTEGRATION.md)
- [DOCKER_MEMORY_SETUP.md](../DOCKER_MEMORY_SETUP.md)
- [MEMORY_SIMPLIFICATION_REVIEW.md](../MEMORY_SIMPLIFICATION_REVIEW.md)

---

**Status**: Production-ready
**Last Updated**: March 5, 2026
**Maintainer**: HoloLoom team
