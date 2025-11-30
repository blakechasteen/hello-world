# HoloLoom Memory Module - Comprehensive Analysis

**Purpose**: Unified memory architecture with knowledge graph + vector retrieval
**Status**: Production-ready (Nov 2025)
**Key Concept**: Multi-tier memory (working → episodic → persistent) with auto-fallback

---

## 1. Module Structure: 51 Python Files

### Core Memory Layer (7 files)
- **`__init__.py`** - Package exports, weaving aliases (YarnGraph, ReflectionBuffer)
- **`protocol.py`** - Memory protocols & canonical type imports (120 lines)
- **`backend_factory.py`** - Backend creation with intelligent auto-fallback (856 lines)
- **`cache.py`** - Multi-scale retrieval + memory manager (600+ lines)
- **`graph.py`** - NetworkX knowledge graph (YarnGraph/KG) - always works
- **`unified.py`** - Unified memory interface (elegant user-facing API)
- **`base.py`** - Compatibility shim for backward compatibility

### Backend Implementations (4 files)
- **`neo4j_graph.py`** - Production Neo4j backend
- **`neo4j_memory_store.py`** - Neo4j memory store wrapper
- **`hyperspace_backend.py`** - Research-only gated multipass backend
- **`hybrid_retrieval.py`** - Advanced hybrid fusion strategies

### Vector Storage (7 files in `stores/`)
- **`qdrant_store.py`** - Qdrant vector search backend
- **`neo4j_vector_store.py`** - Neo4j vector capabilities
- **`in_memory_store.py`** - In-memory vector store
- **`hybrid_store.py`** - Hybrid backend wrapper
- **`hybrid_neo4j_qdrant.py`** - Optimized Neo4j+Qdrant fusion
- **`file_store.py`** - File-based persistence
- **`mem0_store.py`** - Mem0 integration layer

### Advanced Retrieval (6 files)
- **`retrieval_strategies.py`** - Multiple retrieval algorithms
- **`spring_dynamics_engine.py`** - Spring-based memory activation
- **`spring_graph_retriever.py`** - Graph traversal with physics simulation
- **`spring_memory_scoring.py`** - Activation-based scoring
- **`hybrid_neo4j_qdrant.py`** - Optimized fusion
- **`beta_wave_retrieval.py`** - Oscillatory retrieval patterns

### Multimodal & Integration (7 files)
- **`multimodal_memory.py`** - Text + image memory support
- **`multimodal_encoder.py`** - Encoding for multiple modalities
- **`photo_tokens.py`** - CLIP embeddings for visual retrieval
- **`visual_compression.py`** - Knowledge graph → image compression
- **`integrators.py`** - Multi-backend integrators
- **`mcp_rag_server.py`** - RAG via Model Context Protocol
- **`mcp_server.py`** - MCP server implementation

### Learning & Adaptation (8 files)
- **`activation_field.py`** - Neural activation tracking
- **`awareness_graph.py`** - Memory awareness & introspection
- **`awareness_types.py`** - Type definitions for awareness
- **`consolidation.py`** - Memory consolidation (sleep-like)
- **`llm_consolidator.py`** - LLM-powered consolidation
- **`multi_wave_engine.py`** - Multi-frequency memory retrieval
- **`repository_context.py`** - Repository-aware memory
- **`weaving_adapter.py`** - Weaving orchestrator integration

### Utilities & Demos (5 files)
- **`lifecycle_manager.py`** - Resource lifecycle management
- **`demo_awareness.py`** - Awareness system demonstration
- **`demo_beta_wave_retrieval.py`** - Beta wave demo
- **`demo_streaming_memory.py`** - Streaming memory demo
- **`mem0_adapter.py`** - Mem0 intelligence extraction

### Additional (3 files)
- **`integrated_memory_system.py`** - End-to-end integration
- **`spring_dynamics.py`** & **`spring_dynamics_advanced.py`** - Physics-based activation

---

## 2. Key Classes & Interfaces

### Yarn Graph (Knowledge Graph)

```python
from HoloLoom.memory import KG, YarnGraph, KGEdge

# These are equivalent:
kg = KG()  # Short form
kg = YarnGraph()  # Weaving metaphor alias

# Add edges (relationships between entities)
kg.add_edges([
    KGEdge("Python", "programming_language", "IS_A", weight=1.0),
    KGEdge("attention", "transformer", "USES", weight=0.95),
    KGEdge("GPT", "language_model", "IS_A", weight=1.0),
])

# Retrieve subgraph for entity
subgraph = kg.get_subgraph("transformer", depth=2)

# Find paths between entities
path = kg.find_path("attention", "neural_network")

# Get spectral features (graph Laplacian eigenvalues)
features = kg.get_spectral_features("attention")
```

**Architecture**: NetworkX MultiDiGraph with bi-temporal support
- **Event time**: When relationship occurred
- **Ingestion time**: When we learned about it
- **Valid from/to**: Temporal validity for point-in-time queries

---

### Memory Retrieval System

```python
from HoloLoom.memory import MemoryShard, RetrieverMS, MemoryManager, Retriever

# Memory unit - atomic chunk
shard = MemoryShard(
    id="doc_42",
    text="Thompson Sampling balances exploration/exploitation...",
    episode="learning_session_1",
    entities=["Thompson Sampling", "exploration", "exploitation"],
    motifs=["algorithm", "explanation"],
    scales={  # Pre-computed embeddings at multiple scales
        "96": [0.1, 0.2, ...],   # Small (96d) - fast
        "192": [0.15, 0.25, ...],  # Medium (192d)
        "384": [0.18, 0.28, ...]   # Large (384d) - accurate
    }
)

# Multi-Scale Retriever with BM25 fusion
retriever = RetrieverMS(
    shards=shards,
    emb=embeddings,  # MatryoshkaEmbeddings instance
    fusion_weights={96: 0.2, 192: 0.3, 384: 0.5},  # Larger = more weight
    bm25_weight=0.15  # 15% lexical matching
)

# Search (returns ranked shards with scores)
results = await retriever.search(
    query="What is Thompson Sampling?",
    k=5,  # Top-5
    fast=False  # Use all scales (True = fastest scale only)
)
# Returns: [(MemoryShard, float), ...] sorted by relevance
```

**Retrieval Strategy**:
- **Fast mode**: Use smallest scale (96d) only - ultra fast
- **Full mode**: Fuse all scales (96d + 192d + 384d) + BM25 - high quality
- **Scale fusion**: Larger embeddings get higher weight (0.5 vs 0.2)

---

### Memory Manager (Multi-Tier Memory)

```python
from HoloLoom.memory import MemoryManager, create_memory_manager

manager = await create_memory_manager(
    shards=shards,
    emb=embeddings,
    root="data"  # Persistence directory
)

# 4-tier architecture:
# 1. Working Memory (Tier 1) - O(1) hash cache, hot queries
# 2. Episodic Buffer (Tier 2) - Recent interactions (bounded deque)
# 3. PDV (Tier 3) - Persistent raw storage (append-only JSONL)
# 4. MemoAI (Tier 4) - Persistent vector index (disk)

# Retrieve with automatic caching
context = await manager.retrieve(
    query=Query(text="What is Thompson Sampling?"),
    kg_sub=kg_subgraph,
    fast=False
)
# Returns Context with:
# - hits: retrieved shards
# - kg_sub: knowledge graph context
# - shard_texts: extracted texts
# - relevance: aggregated score

# Persist (non-blocking, queued background task)
await manager.persist(
    query=query,
    results={"tool": "search", "count": 5},
    features=features
)

# Graceful shutdown (waits for persistence queue to drain)
await manager.shutdown()
```

**Tier Characteristics**:
```
Tier 1: Working Memory
├─ Storage: Dict (in-memory hash)
├─ Speed: <1ms
├─ Persistence: No
└─ Capacity: ~100 queries (configurable)

Tier 2: Episodic Buffer
├─ Storage: Bounded deque
├─ Speed: ~10ms
├─ Persistence: No
└─ Capacity: ~100 interactions (configurable)

Tier 3: PDV (Personal Data Vault)
├─ Storage: Append-only JSONL on disk
├─ Speed: ~50-100ms
├─ Persistence: Yes, durable
└─ Capacity: Unlimited

Tier 4: MemoAI (Vector Index)
├─ Storage: Indexed embeddings on disk
├─ Speed: ~20-50ms
├─ Persistence: Yes, queryable
└─ Capacity: Unlimited
```

---

### Persistence Clients

```python
# PDV: Personal Data Vault (raw storage)
from HoloLoom.memory import PDVClient

pdv = PDVClient(root="data")
await pdv.store_shard(shard)  # Append to pdv_shards.jsonl
shards = await pdv.load_all_shards()  # Load all

# MemoAI: Vector index (semantic search)
from HoloLoom.memory import MemoAIClient

memo = MemoAIClient(root="data")
await memo.upsert_vectors(
    shard_id="doc_42",
    scale_vectors={
        "96": [0.1, 0.2, ...],
        "192": [0.15, 0.25, ...],
        "384": [0.18, 0.28, ...]
    }
)
```

---

## 3. Backend Options

### INMEMORY: NetworkX (Development, Testing)

```python
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend

config = Config.bare()
config.memory_backend = MemoryBackend.INMEMORY

memory = await create_memory_backend(config)
```

**Characteristics**:
- Storage: In-memory NetworkX MultiDiGraph
- Persistence: None (session only)
- Speed: <10ms
- Use: Development, unit tests, prototyping
- Fallback: Built-in (always available)

---

### HYBRID: Neo4j + Qdrant (Production, Recommended)

```python
config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID

# Set connection details (or use Docker defaults)
config.neo4j_uri = "bolt://localhost:7687"
config.neo4j_username = "neo4j"
config.neo4j_password = "password"

config.qdrant_host = "localhost"
config.qdrant_port = 6333
config.qdrant_collection = "hololoom"

memory = await create_memory_backend(config)
```

**Characteristics**:
- Storage: Neo4j (graph) + Qdrant (vectors)
- Persistence: Durable (Docker containers)
- Speed: ~50ms (balanced)
- Use: Production workloads
- Auto-fallback chain:
  1. Neo4j + Qdrant (both)
  2. Neo4j only (graph reasoning)
  3. Qdrant only (vector search)
  4. NetworkX (INMEMORY fallback)

**Query Strategy**:
```
Store (parallel):
├─ Neo4j: Entity relationships, hierarchies
└─ Qdrant: Vector embeddings, similarity

Recall (fusion):
├─ Neo4j: Subgraph extraction + path finding
├─ Qdrant: Multi-scale vector similarity
└─ Fusion: Balanced weighting of both
```

---

### HYPERSPACE: Research Backend

```python
config = Config.fused()
config.memory_backend = MemoryBackend.HYPERSPACE

memory = await create_memory_backend(config)
```

**Characteristics**:
- Storage: Advanced gated multipass (research)
- Persistence: Yes
- Speed: ~150ms
- Use: Research, advanced experiments
- Falls back to HYBRID if unavailable

---

## 4. Usage Patterns

### Pattern 1: Basic Memory Retrieval

```python
from HoloLoom.config import Config
from HoloLoom.memory.backend_factory import create_memory_backend

async def main():
    # Create backend
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY  # For demo
    memory = await create_memory_backend(config)
    
    # Build context (details depend on backend implementation)
    # This would be integrated with orchestrator
```

---

### Pattern 2: Multi-Tier Retrieval with Caching

```python
from HoloLoom.memory import create_memory_manager
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Query

async def retrieve_with_caching():
    # Initialize
    embeddings = MatryoshkaEmbeddings()
    manager = await create_memory_manager(shards, embeddings)
    
    # Query 1 (cold cache)
    ctx1 = await manager.retrieve(Query(text="What is X?"), kg_sub, fast=False)
    # ~150ms latency
    
    # Query 2 (warm cache - same query)
    ctx2 = await manager.retrieve(Query(text="What is X?"), kg_sub, fast=False)
    # <1ms latency (100x faster!)
    
    await manager.shutdown()
```

---

### Pattern 3: Knowledge Graph Reasoning

```python
from HoloLoom.memory import KG, KGEdge

async def build_knowledge():
    kg = KG()
    
    # Add domain knowledge
    edges = [
        KGEdge("Python", "language", "IS_A"),
        KGEdge("Python", "interpreted", "PROPERTY"),
        KGEdge("PyTorch", "Python", "USES"),
        KGEdge("deep_learning", "PyTorch", "USES"),
    ]
    
    kg.add_edges(edges)
    
    # Reason: What uses Python?
    subgraph = kg.get_subgraph("Python", depth=2)
    
    # Find path: deep_learning → Python
    path = kg.find_path("deep_learning", "Python")
    # Returns: ["deep_learning", "PyTorch", "Python"]
```

---

### Pattern 4: Graceful Degradation

```python
async def production_setup():
    # Try HYBRID first, fall back automatically
    config = Config.fused()
    config.memory_backend = MemoryBackend.HYBRID
    config.neo4j_uri = "bolt://neo4j.company.com:7687"
    config.qdrant_host = "qdrant.company.com"
    
    memory = await create_memory_backend(config)
    # If Neo4j unavailable → uses Qdrant only
    # If both unavailable → uses NetworkX (INMEMORY)
    # ✅ System always works
    
    # Check what backend is actually active
    health = await memory.health_check()
    print(health['backends'])  # Shows which backends are working
```

---

## 5. Integration Examples

### Integration 1: With Orchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.memory.backend_factory import create_memory_backend

async def main():
    config = Config.fused()
    memory = await create_memory_backend(config)
    
    async with WeavingOrchestrator(cfg=config, memory=memory) as orchestrator:
        # Memory is automatically used for context retrieval
        spacetime = await orchestrator.weave(query)
```

---

### Integration 2: With RAG System

```python
from HoloLoom.rag import SimpleRAG
from HoloLoom.memory.backend_factory import create_memory_backend

async def rag_with_persistent_memory():
    # Create persistent backend
    config = Config.fused()
    memory = await create_memory_backend(config)
    
    # RAG leverages memory system
    async with SimpleRAG(config=config, memory=memory) as rag:
        await rag.ingest("Thompson Sampling balances...")
        result = await rag.query("What is Thompson Sampling?")
        print(result.response)
```

---

### Integration 3: Hybrid Memory Operations

```python
from HoloLoom.memory import create_memory_manager, KG

async def hybrid_memory():
    # Vector retrieval
    manager = await create_memory_manager(shards, embeddings)
    context = await manager.retrieve(query, None)
    
    # Graph reasoning
    kg = KG()
    kg.add_edges([...])
    subgraph = kg.get_subgraph("entity", depth=2)
    
    # Combine both
    combined_context = {
        "vector_hits": context.hits,
        "graph_context": subgraph,
        "relevance": context.relevance
    }
```

---

## 6. API Quick Reference

### Factory Functions

```python
# Create retriever
from HoloLoom.memory import create_retriever
retriever = create_retriever(shards, embeddings, fusion_weights)

# Create memory manager
from HoloLoom.memory import create_memory_manager
manager = await create_memory_manager(shards, embeddings, root="data")

# Create backend
from HoloLoom.memory.backend_factory import create_memory_backend
memory = await create_memory_backend(config)
```

---

### Key Exports from `__init__.py`

```python
from HoloLoom.memory import (
    # Cache & Retrieval
    Retriever,           # Protocol
    MemoryShard,         # Atomic memory unit
    RetrieverMS,         # Multi-scale retriever
    MemoryManager,       # Multi-tier memory
    create_retriever,    # Factory
    create_memory_manager,  # Factory
    
    # Graph
    KG,                  # Knowledge graph (NetworkX)
    YarnGraph,           # Weaving alias for KG
    KGEdge,              # Graph edge
    KGStore,             # Protocol
    
    # Aliases (Weaving metaphor)
    ReflectionBuffer,    # = MemoryManager
)
```

---

## 7. Configuration

### Memory Backend Selection

```python
from HoloLoom.config import Config, MemoryBackend

# Configure in Config class
config = Config.fused()

# Set backend
config.memory_backend = MemoryBackend.INMEMORY  # or HYBRID, HYPERSPACE

# Set connection details (for HYBRID)
config.neo4j_uri = "bolt://localhost:7687"
config.neo4j_username = "neo4j"
config.neo4j_password = "password"

config.qdrant_host = "localhost"
config.qdrant_port = 6333
config.qdrant_collection = "hololoom"
```

---

## 8. Performance Characteristics

| Operation | Latency | When | Notes |
|-----------|---------|------|-------|
| **Vector search (small shard set)** | ~50ms | Cold cache | INMEMORY/HYBRID |
| **Vector search (cache hit)** | <1ms | Working memory | 100x speedup |
| **Graph traversal** | ~20ms | Small subgraph | Depth ≤ 3 |
| **Multi-scale fusion** | ~80ms | FUSED mode | 96d+192d+384d |
| **Store (single)** | ~10ms | INMEMORY | Sync operation |
| **Store (parallel)** | ~50ms | HYBRID | Neo4j + Qdrant |
| **Persistence (async)** | ~100ms | Background | Non-blocking |

---

## 9. Testing & Validation

```bash
# Basic tests
pytest HoloLoom/tests/unit/test_*.py -v

# Memory integration tests
pytest HoloLoom/tests/integration/ -k memory -v

# Backend health check
python -c "
import asyncio
from HoloLoom.config import Config
from HoloLoom.memory.backend_factory import create_memory_backend

async def check():
    config = Config.fast()
    memory = await create_memory_backend(config)
    health = await memory.health_check()
    print(health)

asyncio.run(check())
"
```

---

## 10. Common Issues & Solutions

### Issue: Neo4j/Qdrant connection timeout
**Solution**: Check Docker is running
```bash
docker-compose ps
docker-compose up -d  # Start containers
```

### Issue: NetworkX unavailable
**Solution**: Install networkx
```bash
pip install networkx numpy scipy
```

### Issue: Slow retrieval on large shard sets
**Solution**: Use HYBRID backend + fast mode
```python
results = await retriever.search(query, k=5, fast=True)
```

### Issue: Working memory growing too large
**Solution**: Configure manager capacity
```python
manager = MemoryManager(
    retriever=retriever,
    pdv=pdv,
    memo=memo,
    working_memory_size=50,      # Reduce from 100
    episodic_buffer_size=50      # Reduce from 100
)
```

---

## Summary

The HoloLoom memory module provides:

✅ **3 backend options**: INMEMORY (dev) → HYBRID (prod) → HYPERSPACE (research)
✅ **Multi-tier retrieval**: Working memory → episodic → persistent (4 tiers)
✅ **Knowledge graph**: YarnGraph with bi-temporal relationships
✅ **Multi-scale embeddings**: Matryoshka fusion (96d + 192d + 384d)
✅ **Automatic fallback**: Always works, degrades gracefully
✅ **Production-ready**: Async, durable, zero data loss

**Key files to know**:
- `backend_factory.py` - Backend selection
- `cache.py` - Multi-tier retrieval
- `graph.py` - Knowledge graph
- `protocol.py` - Type definitions

