# Memory + Query Foundation Architecture
**Status**: Foundation review for MVP scoping
**Date**: 2025-10-24

## 🎯 Core Philosophy

The memory system is the **foundation** of HoloLoom. The loom can't weave without yarn.

**Key Insight**: We need solid memory storage + query before we can do fancy loom operations.

---

## 📦 What We Have (Implemented)

### 1. **Protocol Layer** (`HoloLoom/memory/protocol.py`)
✅ **COMPLETE** - Clean protocol-based architecture

```python
# Core protocols define WHAT, not HOW
- MemoryStore: store(), retrieve(), delete()
- MemoryNavigator: navigate_forward(), navigate_backward(), find_neighbors()
- PatternDetector: detect_patterns()
- UnifiedMemoryInterface: Facade pattern over all protocols
```

**Key Types:**
- `Memory` - single memory object (compatible with MemoryShard!)
- `MemoryQuery` - query specification
- `RetrievalResult` - search results with scores
- `Strategy` - TEMPORAL, SEMANTIC, GRAPH, PATTERN, FUSED

**Bridge Functions:**
- `Memory.from_shard(shard)` - converts SpinningWheel output → Memory
- `shards_to_memories(shards)` - batch conversion
- `pipe_text_to_memory(text, memory)` - one-liner pipeline

### 2. **Store Implementations**

#### ✅ InMemoryStore (`stores/in_memory_store.py`)
- Pure Python dict-based
- No dependencies
- Good for testing/development
- **WORKING**

#### ⚠️ Mem0Store (`stores/mem0_store.py` + `stores/mem0_memory_store.py`)
- Two implementations (need to consolidate)
- LLM-powered memory extraction
- User-specific memory
- **PARTIAL** - needs testing

#### ⚠️ Neo4jStore (`stores/neo4j_store.py` + `stores/neo4j_memory_store.py`)
- Two implementations (need to consolidate)
- Graph-based "thread" model
- Relationship traversal
- **PARTIAL** - needs Neo4j running

#### ⚠️ QdrantStore (`stores/qdrant_store.py`)
- Vector similarity search
- Embedding-based retrieval
- **PARTIAL** - needs Qdrant running

#### ✅ HybridStore (`stores/hybrid_store.py`)
- Combines multiple stores
- Fusion of retrieval strategies
- **IMPLEMENTED** - needs testing

### 3. **SpinningWheel Integration**

✅ **COMPLETE** - TextSpinner with gated multipass
- `TextSpinner` - converts text → MemoryShards
- Gated multipass processing (overview → conditional deep passes)
- Multiple chunking strategies
- Entity extraction
- **WORKING** (we just tested this!)

---

## 🚧 What's Missing (For MVP)

### 1. **No Working End-to-End Test**
We have pieces but haven't validated:
```python
# This should work but hasn't been tested:
text → TextSpinner → shards → Memory.from_shard() → store() → retrieve()
```

### 2. **Store Implementations Need Validation**
- InMemoryStore: ✅ Should work
- Mem0Store: ❓ Untested, two versions
- Neo4jStore: ❓ Untested, needs setup
- QdrantStore: ❓ Untested, needs setup
- HybridStore: ❓ Untested fusion logic

### 3. **Query Quality Unknown**
We can store, but how good is retrieval?
- Does semantic search actually work?
- Are scores meaningful?
- Does fusion make sense?

### 4. **Navigator Not Implemented**
Protocols exist, but no implementation:
- `MemoryNavigator` - protocol only
- `HofstadterNavigator` - referenced but doesn't exist
- No spatial traversal

### 5. **Pattern Detector Not Implemented**
Protocols exist, but no implementation:
- `PatternDetector` - protocol only
- `MultiPatternDetector` - referenced but doesn't exist
- No pattern discovery

### 6. **Unified Memory Facade Untested**
- `UnifiedMemoryInterface` looks good
- `create_unified_memory()` factory with graceful degradation
- **BUT**: Never been run end-to-end

---

## 🎯 MVP Scope (What We Actually Need)

Let's get laser-focused on the minimum viable foundation:

### Phase 1: Core Loop (MUST HAVE)
```python
# This must work reliably:
1. Text → TextSpinner → Shards
2. Shards → Memories (conversion)
3. Memories → Store (persistence)
4. Query → Retrieve (search)
5. Results → User
```

**Requirements:**
- ✅ TextSpinner (working)
- ✅ Memory data types (working)
- ✅ MemoryStore protocol (working)
- ⚠️ At least ONE working store (InMemoryStore probably works)
- ❌ End-to-end test (MISSING)

### Phase 2: Quality Retrieval (SHOULD HAVE)
```python
# Retrieval should be useful:
- TEMPORAL: Recent memories work
- SEMANTIC: Text similarity works
- FUSED: Combination makes sense
```

**Requirements:**
- ⚠️ Scoring functions validated
- ❌ Comparison of strategies
- ❌ Benchmarks/examples

### Phase 3: Persistence (COULD HAVE)
```python
# Memories persist across sessions:
- Save to disk/database
- Load on startup
- Handle crashes
```

**Requirements:**
- ⚠️ One persistent store (Neo4j OR Qdrant OR Mem0)
- ❌ Serialization/deserialization
- ❌ Migration tools

### Phase 4: Advanced (WON'T HAVE FOR MVP)
```python
# Nice to have later:
- Navigation (forward/backward)
- Pattern detection
- Hybrid fusion
- Multi-scale embeddings
```

---

## 🔧 Immediate Next Steps

### Option A: Validate What We Have
1. Write end-to-end test with InMemoryStore
2. Test TextSpinner → Memory conversion
3. Test store + retrieve cycle
4. Validate scoring makes sense
5. **THEN** worry about other stores

### Option B: Build One Solid Store
1. Pick ONE store (Neo4j? Mem0? Qdrant?)
2. Make sure it works perfectly
3. Test with real data (your beekeeping notes?)
4. Get retrieval quality high
5. **THEN** add others

### Option C: Simplify and Consolidate
1. Remove duplicate implementations (2x mem0, 2x neo4j)
2. Pick best-of-breed for each
3. Test each in isolation
4. Build hybrid on solid foundation

---

## 💭 Blake's Call

You said: "i feel like we're not done with the memory + query componentry of the MVP"

**You're right.**

We have:
- ✅ Great architecture (protocols are clean)
- ✅ Working TextSpinner (we just proved it)
- ✅ Clean data piping (Memory.from_shard)
- ⚠️ Multiple store implementations (untested)
- ❌ No working end-to-end flow
- ❌ Unknown query quality

**What's the next move?**
1. Test end-to-end with InMemoryStore?
2. Pick a real store and get it working?
3. Build integration tests?
4. Something else?

The loom can wait. Let's nail the foundation first.
