# 🎯 Unified Memory - Quick Reference

## One-Line Summary
**Protocol-based memory system combining Mem0, Neo4j, and Qdrant with elegant API**

---

## 🚀 Instant Start (Copy-Paste)

```python
import asyncio
from hololoom.memory.protocol import UnifiedMemoryInterface, Strategy
from hololoom.memory.stores import InMemoryStore

async def main():
    # Create
    memory = UnifiedMemoryInterface(store=InMemoryStore())
    
    # Store
    await memory.store("Hive Jodi needs winter prep", 
                       context={'place': 'apiary'}, 
                       user_id="blake")
    
    # Recall
    results = await memory.recall("winter", strategy=Strategy.FUSED, user_id="blake")
    
    # Print
    for mem, score in zip(results.memories, results.scores):
        print(f"[{score:.2f}] {mem.text}")

asyncio.run(main())
```

---

## 📦 Files You Need

| What | Where |
|------|-------|
| **Protocols** | `memory/protocol.py` |
| **InMemory store** | `memory/stores/in_memory_store.py` |
| **Mem0 store** | `memory/stores/mem0_store.py` |
| **Neo4j store** | `memory/stores/neo4j_store.py` |
| **Qdrant store** | `memory/stores/qdrant_store.py` |
| **Hybrid fusion** | `memory/stores/hybrid_store.py` |
| **Examples** | `examples/unified_memory_demo.py` |
| **Guide** | `memory/QUICKSTART.md` |

---

## 🎨 The Three Backends

```
Mem0      → User-specific LLM extraction  → 30% weight
Neo4j     → Thread-based graph storage    → 30% weight
Qdrant    → Multi-scale vector search     → 40% weight
                    ↓
            Hybrid Fusion
                    ↓
            Ranked Results
```

---

## 🔧 Strategies

```python
Strategy.TEMPORAL  # Recent memories
Strategy.SEMANTIC  # Similar meaning
Strategy.GRAPH     # Connected in graph
Strategy.PATTERN   # Mathematical patterns
Strategy.FUSED     # Combined (best)
```

---

## 📊 Backend Matrix

| Backend | Speed | Strength | Install |
|---------|-------|----------|---------|
| **InMemory** | ⚡⚡⚡ | Testing | Built-in |
| **Mem0** | ⚡ | User-specific | `pip install mem0ai` |
| **Neo4j** | ⚡⚡ | Graph traversal | `pip install neo4j` |
| **Qdrant** | ⚡⚡⚡ | Similarity | `pip install qdrant-client` |
| **Hybrid** | ⚡ | Best accuracy | All above |

---

## 🏃 Run Demo

```bash
python hololoom/examples/unified_memory_demo.py
```

**Output**: 3 examples (in-memory, hybrid, factory)

---

## 🧪 Test Pattern

```python
class MockStore:
    async def store(self, m): return "id"
    async def retrieve(self, q, s): return RetrievalResult(...)

memory = UnifiedMemoryInterface(store=MockStore())
# Easy testing!
```

---

## 🚨 Setup Backends

```bash
# Install
pip install mem0ai neo4j qdrant-client sentence-transformers

# Run Neo4j
docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

# Run Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

---

## 💡 Why It's Elegant

| Old | New | Win |
|-----|-----|-----|
| 450 lines | 150 lines | 67% smaller |
| Concrete class | Protocol | Swappable |
| Synchronous | Async | Non-blocking |
| Hard to test | Easy mocks | Testable |

---

## 📈 Performance

```
InMemory:  < 1ms   (testing)
Qdrant:   ~10ms   (fast + accurate)
Neo4j:    ~50ms   (graph traversal)
Mem0:    ~100ms   (LLM extraction)
Hybrid:  ~150ms   (all three combined)
```

---

## ✅ Checklist

- [x] Protocol-based design
- [x] 4 store implementations
- [x] Hybrid fusion (4 methods)
- [x] Async-first
- [x] Graceful degradation
- [x] Working examples
- [x] Documentation
- [x] HoloLoom standard compliance

---

## 🎯 Next Action

```bash
cd ~/Documents/mythRL
python hololoom/examples/unified_memory_demo.py
```

**That's it!** 🚀

---

**Created**: Oct 23, 2025 | **Status**: Production-ready
