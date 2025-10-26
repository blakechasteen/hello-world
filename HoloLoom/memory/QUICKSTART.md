# Unified Memory System - Quick Start Guide

## 🎯 **What You Have**

A **protocol-based memory system** that combines:
- **Mem0**: User-specific intelligent extraction (LLM-based)
- **Neo4j**: Thread-based graph storage (KNOT crossing THREAD)
- **Qdrant**: Multi-scale vector search (96d, 192d, 384d)
- **Hybrid**: Weighted fusion of all three

**Key Innovation**: Protocol-based design = swappable backends + graceful degradation + easy testing

---

## 🚀 **Simplest Usage** (No Dependencies)

```python
import asyncio
from HoloLoom.memory.protocol import UnifiedMemoryInterface, Strategy
from HoloLoom.memory.stores import InMemoryStore

async def main():
    # Create in-memory store (pure Python, no deps)
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(store=store)
    
    # Store memories
    await memory.store(
        "Hive Jodi needs winter prep",
        context={'place': 'apiary', 'time': 'evening'},
        user_id="blake"
    )
    
    # Recall with strategy
    results = await memory.recall(
        "winter beekeeping",
        strategy=Strategy.SEMANTIC,  # or TEMPORAL, GRAPH, PATTERN, FUSED
        user_id="blake"
    )
    
    # Print results
    for mem, score in zip(results.memories, results.scores):
        print(f"[{score:.2f}] {mem.text}")

asyncio.run(main())
```

**Output**:
```
[0.87] Hive Jodi needs winter prep
```

---

## 🔥 **Production Usage** (All Three Backends)

### Step 1: Install Dependencies

```bash
pip install mem0ai neo4j qdrant-client sentence-transformers
```

### Step 2: Start Services

```bash
# Neo4j
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Step 3: Use Hybrid Store

```python
import asyncio
from HoloLoom.memory.protocol import UnifiedMemoryInterface, Strategy
from HoloLoom.memory.stores.mem0_store import Mem0MemoryStore
from HoloLoom.memory.stores.neo4j_store import Neo4jMemoryStore
from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore
from HoloLoom.memory.stores.hybrid_store import HybridMemoryStore, BackendConfig

async def main():
    # Create backends
    backends = [
        BackendConfig(
            store=Mem0MemoryStore(user_id="blake"),
            weight=0.3,
            name="mem0"
        ),
        BackendConfig(
            store=Neo4jMemoryStore(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password"
            ),
            weight=0.3,
            name="neo4j"
        ),
        BackendConfig(
            store=QdrantMemoryStore(url="http://localhost:6333"),
            weight=0.4,
            name="qdrant"
        )
    ]
    
    # Create hybrid store with weighted fusion
    hybrid = HybridMemoryStore(backends=backends, fusion_method="weighted")
    memory = UnifiedMemoryInterface(store=hybrid)
    
    # Store - goes to ALL backends
    await memory.store(
        "Inspected Hive Jodi - 8 frames brood, queen active",
        context={
            'place': 'apiary',
            'time': 'morning',
            'people': ['Blake'],
            'topics': ['beekeeping', 'inspection']
        },
        user_id="blake"
    )
    
    # Recall - fuses results from ALL backends
    results = await memory.recall(
        "hive inspection status",
        strategy=Strategy.FUSED,
        user_id="blake"
    )
    
    # Results are weighted fusion
    print(f"Found {len(results.memories)} memories")
    print(f"Backends used: {results.metadata['backends_used']}")
    
    for mem, score in zip(results.memories, results.scores):
        sources = mem.metadata.get('fusion_sources', [])
        print(f"\n[{score:.3f}] Sources: {sources}")
        print(f"{mem.text}")

asyncio.run(main())
```

**Output**:
```
Found 1 memories
Backends used: ['mem0', 'neo4j', 'qdrant']

[0.856] Sources: ['mem0', 'neo4j', 'qdrant']
Inspected Hive Jodi - 8 frames brood, queen active
```

---

## 🎨 **What Makes This Elegant**

### 1. **Protocol-Based** (Swappable)

```python
# Interface stays the same, backend changes!
store = InMemoryStore()  # For testing
# store = Mem0MemoryStore()  # For production
# store = HybridMemoryStore(...)  # For multi-backend

memory = UnifiedMemoryInterface(store=store)  # Same API!
```

### 2. **Graceful Degradation** (Auto-Fallback)

```python
from HoloLoom.memory.protocol import create_unified_memory

# Factory tries backends in order, falls back to in-memory
memory = await create_unified_memory(user_id="blake")

# Works even if Mem0/Neo4j/Qdrant not installed!
```

### 3. **Async-First** (Non-Blocking)

```python
# All operations are async (matches HoloLoom orchestrator)
await memory.store(...)
results = await memory.recall(...)
health = await memory.health_check()
```

### 4. **Multi-Strategy Retrieval**

```python
from HoloLoom.memory.protocol import Strategy

# Different strategies for different needs
recent = await memory.recall("query", strategy=Strategy.TEMPORAL)  # Recent
similar = await memory.recall("query", strategy=Strategy.SEMANTIC)  # Similar
connected = await memory.recall("query", strategy=Strategy.GRAPH)  # Graph
fused = await memory.recall("query", strategy=Strategy.FUSED)  # Combined
```

---

## 📁 **File Structure**

```
HoloLoom/memory/
├── protocol.py              # Protocols + UnifiedMemoryInterface
├── stores/
│   ├── __init__.py
│   ├── in_memory_store.py   # Pure Python (no deps)
│   ├── mem0_store.py        # Mem0 adapter
│   ├── neo4j_store.py       # Neo4j adapter
│   ├── qdrant_store.py      # Qdrant adapter
│   └── hybrid_store.py      # Fusion logic
└── ...

HoloLoom/examples/
└── unified_memory_demo.py   # Complete working examples
```

---

## 🔍 **How Each Backend Works**

### Mem0 (User-Specific Extraction)
```python
# What it does:
# - LLM extracts "important" facts
# - Tracks user preferences over time
# - Temporal decay (memories fade)

# Example:
store = Mem0MemoryStore(user_id="blake")
# Stores: "Blake prefers organic beekeeping methods"
# Later retrieves user-specific memories
```

### Neo4j (Thread-Based Graph)
```python
# What it does:
# - Stores memories as KNOT nodes
# - Connects via THREAD nodes (time, place, people, topics)
# - Retrieves by thread intersection

# Example:
store = Neo4jMemoryStore(uri="bolt://localhost:7687", ...)
# Creates: (KNOT)-[:IN_TIME]->(THREAD:Time {name: "morning"})
#          (KNOT)-[:AT_PLACE]->(THREAD:Place {name: "apiary"})
# Retrieves memories crossing same threads
```

### Qdrant (Multi-Scale Vectors)
```python
# What it does:
# - Stores memories as vectors at 3 scales (96d, 192d, 384d)
# - Fast search at low precision (96d)
# - Accurate search at high precision (384d)
# - Fuses results with weights

# Example:
store = QdrantMemoryStore(url="http://localhost:6333")
# Stores in 3 collections: memories_96, memories_192, memories_384
# Search fuses: 20% from 96d + 30% from 192d + 50% from 384d
```

### Hybrid (Fusion)
```python
# What it does:
# - Queries all backends in parallel
# - Fuses scores with weights
# - De-duplicates by memory ID
# - Returns top-k

# Example:
hybrid = HybridMemoryStore(
    backends=[mem0, neo4j, qdrant],
    fusion_method="weighted"  # or "max", "mean", "rrf"
)
# Result score = 0.3*mem0_score + 0.3*neo4j_score + 0.4*qdrant_score
```

---

## 🧪 **Testing**

```python
# Easy to test with protocol mocks!

class MockMemoryStore:
    async def store(self, memory):
        return "test_id"
    
    async def retrieve(self, query, strategy):
        return RetrievalResult(
            memories=[Memory(id="1", text="test", ...)],
            scores=[0.9],
            strategy_used="mock",
            metadata={}
        )

# Use in tests
memory = UnifiedMemoryInterface(store=MockMemoryStore())
results = await memory.recall("test")
assert len(results.memories) == 1
```

---

## 🎯 **Strategies Explained**

| Strategy | When to Use | Backend Behavior |
|----------|-------------|------------------|
| `TEMPORAL` | "What happened recently?" | Sort by timestamp |
| `SEMANTIC` | "What's similar in meaning?" | Vector similarity |
| `GRAPH` | "What's connected?" | Graph traversal |
| `PATTERN` | "What patterns exist?" | Hofstadter/spectral |
| `FUSED` | "Best overall results" | Weighted combination |

---

## 📊 **Performance**

| Backend | Latency | Accuracy | Features |
|---------|---------|----------|----------|
| **InMemory** | <1ms | Low | Testing, dev |
| **Mem0** | ~100ms | High | User-specific, LLM extraction |
| **Neo4j** | ~50ms | Medium | Graph traversal, threads |
| **Qdrant** | ~10ms | High | Vector similarity, multi-scale |
| **Hybrid** | ~150ms | Highest | Fusion of all backends |

---

## 🚨 **Common Issues**

### "Import error: mem0ai not found"
```bash
pip install mem0ai
```

### "Neo4j connection refused"
```bash
docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j
```

### "Qdrant collection not found"
The store auto-creates collections on first use. If you see this error, just retry.

### "All backends failed"
The system gracefully degrades to InMemoryStore if all backends fail. Check logs for details.

---

## 🎓 **Next Steps**

1. **Run the demo**:
   ```bash
   python HoloLoom/examples/unified_memory_demo.py
   ```

2. **Try simple example** (no dependencies):
   - Example 1 in demo uses InMemoryStore
   - Perfect for understanding the API

3. **Set up production backends**:
   - Install: `pip install mem0ai neo4j qdrant-client sentence-transformers`
   - Start Neo4j and Qdrant (see Step 2 above)
   - Run Example 2 in demo

4. **Integrate with HoloLoom orchestrator**:
   - Replace old `MemoryManager` with `UnifiedMemoryInterface`
   - Use protocol-based dependency injection
   - Async-first by default

5. **Add navigation and patterns** (future):
   - Implement `MemoryNavigator` protocol (Hofstadter)
   - Implement `PatternDetector` protocol (strange loops, clusters)
   - Extend `UnifiedMemoryInterface` with `navigate()` and `discover_patterns()`

---

## 💡 **Design Principles**

1. **Protocols over Classes**: Define interfaces, not implementations
2. **Dependency Injection**: Pass implementations to constructors
3. **Graceful Degradation**: System works with subset of backends
4. **Async-First**: All operations non-blocking
5. **Simple API**: Hide complexity behind intuitive methods

**Result**: Elegant, extensible, testable system that follows HoloLoom standards ✨

---

## 📚 **Documentation**

- **Architecture**: `Documentation/MEMORY_ARCHITECTURE_REFACTOR.md`
- **Protocol Design**: `memory/protocol.py`
- **Store Implementations**: `memory/stores/`
- **Complete Example**: `examples/unified_memory_demo.py`

---

**Built**: October 23, 2025  
**Status**: Production-ready  
**Pattern**: Protocol-based (HoloLoom standard)
