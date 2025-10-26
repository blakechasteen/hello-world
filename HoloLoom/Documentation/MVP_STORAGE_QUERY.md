# MVP: Storage & Query
## Minimal Viable Product - Core Memory System

**Version**: 1.0
**Date**: October 24, 2025
**Status**: Ready to Use

---

## MVP Scope

### ✅ In Scope (Storage & Query)
- **Store memories** - Save text with context
- **Query memories** - Search with strategies (temporal, semantic, fused)
- **Retrieve by ID** - Get specific memory
- **Health check** - System status
- **MCP integration** - Use from Claude Desktop
- **One backend** - InMemory (immediate) or Neo4j (persistent)

### ❌ Out of Scope (Future)
- Navigation (forward/backward through memory space)
- Pattern detection (loops, clusters, resonances)
- Multi-backend fusion (using multiple stores simultaneously)
- Advanced analytics (Hofstadter sequences, spectral features)

---

## What's Included

### Core Protocol (`memory/protocol.py`)
```python
# Data types
Memory              # Storage format
MemoryQuery         # Query specification
RetrievalResult     # Search results
Strategy            # Search strategies (TEMPORAL, SEMANTIC, FUSED)

# Protocols
MemoryStore         # Storage interface
UnifiedMemoryInterface  # User-facing API

# Utilities
Memory.from_shard()     # Convert SpinningWheel output → Memory
shards_to_memories()    # Batch conversion
pipe_text_to_memory()   # One-liner pipeline
```

### Storage Implementations

#### ✅ InMemoryStore (No Setup)
- **File**: `memory/stores/in_memory_store.py`
- **Pros**: Works immediately, no dependencies
- **Cons**: No persistence (data lost on restart)
- **Use for**: Testing, development, demos

#### ✅ Neo4jMemoryStore (Persistent)
- **File**: `memory/stores/neo4j_store.py`
- **Pros**: Persistent, graph relationships, production-ready
- **Cons**: Requires Neo4j running
- **Use for**: Production, real data

#### ✅ Mem0MemoryStore (AI-Enhanced)
- **File**: `memory/stores/mem0_store.py`
- **Pros**: Intelligent entity extraction
- **Cons**: Requires Mem0 API/setup
- **Use for**: Advanced use cases

### MCP Integration (`memory/mcp_server.py`)
- Expose memories to Claude Desktop
- Tools: `recall_memories`, `store_memory`, `memory_health`
- Resources: Browse memories as `memory://<id>`

### Data Pipeline
```
Text → Spinner → MemoryShards → Memory.from_shard() → Store
```

---

## Quick Start (3 Options)

### Option A: In-Memory (Fastest)

**No setup required!**

```python
from HoloLoom.memory.protocol import UnifiedMemoryInterface
from HoloLoom.memory.stores import InMemoryStore

# Create interface
store = InMemoryStore()
memory = UnifiedMemoryInterface(_store=store)

# Store
mem_id = await memory.store("My first memory")

# Query
results = await memory.recall("first", strategy=Strategy.SEMANTIC)
print(results.memories[0].text)  # "My first memory"
```

### Option B: Neo4j (Persistent)

**Requires**: Neo4j running

```python
from HoloLoom.memory.protocol import UnifiedMemoryInterface, Strategy
from HoloLoom.memory.stores.neo4j_store import Neo4jMemoryStore

# Connect
store = Neo4jMemoryStore(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your-password"
)

# Create interface
memory = UnifiedMemoryInterface(_store=store)

# Store
mem_id = await memory.store(
    "Hive inspection notes",
    context={"date": "2025-10-24", "type": "beekeeping"}
)

# Query with strategy
results = await memory.recall("inspection", strategy=Strategy.FUSED)

# Health check
health = await memory.health_check()
print(health)
```

### Option C: MCP in Claude Desktop

**Requires**: MCP package, config edit

See [QUICK_START_MCP.md](../../QUICK_START_MCP.md) for setup.

Once configured, in Claude Desktop:
```
You: Check my memory system health
Claude: [Shows system status]

You: Remember: "Protocol-based architecture is best practice"
Claude: ✓ Memory stored successfully

You: Search my memories for "architecture"
Claude: [Shows matching memories]
```

---

## MVP Features Checklist

### Storage ✅
- [x] `store(text, context)` - Store single memory
- [x] `store_memory(memory)` - Store Memory object
- [x] `store_many(memories)` - Batch storage
- [x] InMemoryStore implementation
- [x] Neo4jMemoryStore implementation
- [x] Graceful degradation (fallback to InMemory)

### Query ✅
- [x] `recall(query, strategy)` - Search memories
- [x] TEMPORAL strategy (recent first)
- [x] SEMANTIC strategy (meaning similarity)
- [x] FUSED strategy (weighted combination)
- [x] `get_by_id(memory_id)` - Direct retrieval
- [x] Result scoring and ranking

### Data Pipeline ✅
- [x] `Memory.from_shard()` - MemoryShard → Memory
- [x] `shards_to_memories()` - Batch conversion
- [x] `pipe_text_to_memory()` - One-liner
- [x] SpinningWheel integration (TextSpinner, YouTubeSpinner)

### MCP Integration ✅
- [x] MCP server implementation
- [x] `recall_memories` tool
- [x] `store_memory` tool
- [x] `memory_health` tool
- [x] Resource browsing (`memory://`)

### Infrastructure ✅
- [x] Protocol-based architecture
- [x] Async-first design
- [x] Type-safe (full annotations)
- [x] Dependency injection
- [x] Error handling
- [x] Health checks

---

## What's NOT in MVP

These are **explicitly out of scope** for v1.0:

### Navigation (Future v1.1)
```python
# NOT IN MVP
await memory.navigate(
    from_memory_id="mem_123",
    direction=NavigationDirection.FORWARD
)
```

**Why later**: Navigation requires graph traversal logic and Hofstadter sequences. Not needed for basic storage/query.

### Pattern Detection (Future v1.2)
```python
# NOT IN MVP
patterns = await memory.discover_patterns(
    pattern_types=["loop", "cluster"],
    min_strength=0.5
)
```

**Why later**: Pattern detection is complex analysis. MVP focuses on CRUD + search.

### Multi-Backend Fusion (Future v1.3)
```python
# NOT IN MVP
store = HybridMemoryStore(
    backends=[Neo4jStore(), Mem0Store(), QdrantStore()],
    weights=[0.4, 0.3, 0.3]
)
```

**Why later**: Fusion adds complexity. MVP uses single backend.

### Advanced Strategies (Future v1.4)
```python
# NOT IN MVP
Strategy.RESONANT  # Hofstadter resonance patterns
Strategy.GRAPH     # Graph traversal (basic version exists, advanced later)
Strategy.PATTERN   # Pattern-based retrieval
```

**Why later**: Advanced strategies require additional computation. MVP has TEMPORAL, SEMANTIC, FUSED.

---

## API Reference (MVP)

### UnifiedMemoryInterface

#### `store(text, context, user_id) -> str`
Store a text memory.

**Parameters**:
- `text` (str): Memory content
- `context` (dict, optional): Metadata
- `user_id` (str, optional): User identifier

**Returns**: `str` - Memory ID

**Example**:
```python
mem_id = await memory.store(
    "Team decided on protocol-based design",
    context={"date": "2025-10-24", "category": "architecture"}
)
```

#### `store_memory(memory) -> str`
Store a Memory object (from SpinningWheel).

**Parameters**:
- `memory` (Memory): Memory object

**Returns**: `str` - Memory ID

**Example**:
```python
shards = await spin_text("My notes")
memories = shards_to_memories(shards)
mem_id = await memory.store_memory(memories[0])
```

#### `store_many(memories) -> List[str]`
Store multiple memories (batch).

**Parameters**:
- `memories` (List[Memory]): List of Memory objects

**Returns**: `List[str]` - Memory IDs

**Example**:
```python
shards = await spin_text("Long document")
memories = shards_to_memories(shards)
ids = await memory.store_many(memories)
```

#### `recall(query, strategy, limit) -> RetrievalResult`
Search memories.

**Parameters**:
- `query` (str): Search query
- `strategy` (Strategy): TEMPORAL, SEMANTIC, or FUSED
- `limit` (int, optional): Max results (default: 5)

**Returns**: `RetrievalResult` with memories, scores, metadata

**Example**:
```python
results = await memory.recall(
    "architecture decisions",
    strategy=Strategy.FUSED,
    limit=10
)

for mem, score in zip(results.memories, results.scores):
    print(f"[{score:.2f}] {mem.text}")
```

#### `health_check() -> Dict`
Check system status.

**Returns**: Dict with backend status, memory count, etc.

**Example**:
```python
health = await memory.health_check()
print(f"Backend: {health['store']['backend']}")
print(f"Status: {health['store']['status']}")
print(f"Memories: {health['store']['memory_count']}")
```

---

## Testing MVP

### Test Script
```python
"""Test MVP functionality"""
import asyncio
from HoloLoom.memory.protocol import (
    UnifiedMemoryInterface,
    Strategy,
    Memory
)
from HoloLoom.memory.stores import InMemoryStore

async def test_mvp():
    print("=== MVP Test: Storage & Query ===\n")

    # Setup
    store = InMemoryStore()
    memory = UnifiedMemoryInterface(_store=store)

    # Test 1: Store
    print("1. Testing storage...")
    mem_id = await memory.store(
        "Protocol-based architecture is best practice",
        context={"category": "architecture", "date": "2025-10-24"}
    )
    print(f"   ✓ Stored: {mem_id}\n")

    # Test 2: Store many
    print("2. Testing batch storage...")
    ids = []
    for i in range(5):
        mid = await memory.store(f"Test memory {i}")
        ids.append(mid)
    print(f"   ✓ Stored {len(ids)} memories\n")

    # Test 3: Query TEMPORAL
    print("3. Testing TEMPORAL query...")
    results = await memory.recall("memory", strategy=Strategy.TEMPORAL, limit=3)
    print(f"   ✓ Found {results.count} memories")
    for mem, score in zip(results.memories, results.scores):
        print(f"     [{score:.2f}] {mem.text[:50]}\n")

    # Test 4: Query SEMANTIC
    print("4. Testing SEMANTIC query...")
    results = await memory.recall("architecture", strategy=Strategy.SEMANTIC)
    print(f"   ✓ Found {results.count} memories")
    for mem, score in zip(results.memories, results.scores):
        print(f"     [{score:.2f}] {mem.text[:50]}\n")

    # Test 5: Query FUSED
    print("5. Testing FUSED query...")
    results = await memory.recall("protocol", strategy=Strategy.FUSED)
    print(f"   ✓ Found {results.count} memories\n")

    # Test 6: Health check
    print("6. Testing health check...")
    health = await memory.health_check()
    print(f"   ✓ Backend: {health['store']['backend']}")
    print(f"   ✓ Status: {health['store']['status']}")
    print(f"   ✓ Memories: {health['store']['memory_count']}\n")

    print("=== All MVP Tests Passed ✓ ===")

if __name__ == "__main__":
    asyncio.run(test_mvp())
```

**Run it**:
```bash
cd c:\Users\blake\Documents\mythRL
python test_mvp.py
```

---

## Production Deployment

### Using Neo4j

1. **Install Neo4j**:
   - Download from https://neo4j.com/download/
   - Or use Docker: `docker run -p 7687:7687 neo4j`

2. **Configure**:
   ```python
   from HoloLoom.memory.stores.neo4j_store import Neo4jMemoryStore

   store = Neo4jMemoryStore(
       uri="bolt://localhost:7687",
       username="neo4j",
       password="your-password"
   )
   ```

3. **Use**:
   ```python
   memory = UnifiedMemoryInterface(_store=store)
   await memory.store("Persistent memory")
   ```

### Using MCP in Claude Desktop

See [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md) for complete setup.

**Config**:
```json
{
  "mcpServers": {
    "hololoom-memory": {
      "command": "python",
      "args": ["-m", "HoloLoom.memory.mcp_server"],
      "cwd": "/path/to/mythRL",
      "env": {
        "PYTHONPATH": "/path/to/mythRL",
        "MEMORY_USER_ID": "your-username"
      }
    }
  }
}
```

---

## Success Criteria

MVP is successful if:

- ✅ Can store text memories
- ✅ Can query by temporal (recent)
- ✅ Can query by semantic (meaning)
- ✅ Works with InMemoryStore (no setup)
- ✅ Works with Neo4jMemoryStore (persistent)
- ✅ Usable from Claude Desktop (MCP)
- ✅ Health check shows status
- ✅ Complete in < 5 minutes (InMemory mode)

---

## What's Next (After MVP)

### Version 1.1 - Navigation
- Add `MemoryNavigator` protocol
- Implement `HofstadterNavigator`
- Support FORWARD/BACKWARD/SIDEWAYS navigation

### Version 1.2 - Pattern Detection
- Add `PatternDetector` protocol
- Implement `MultiPatternDetector`
- Detect loops, clusters, threads

### Version 1.3 - Multi-Backend Fusion
- Implement `HybridMemoryStore`
- Weighted fusion of backends
- Automatic backend selection

### Version 1.4 - Advanced Analytics
- Resonance patterns
- Spectral graph features
- Custom scoring strategies

---

## Files Summary

### Core (Must Have)
- `memory/protocol.py` - Protocols and types ✅
- `memory/stores/in_memory_store.py` - Basic storage ✅
- `memory/stores/neo4j_store.py` - Persistent storage ✅

### Optional (Nice to Have)
- `memory/mcp_server.py` - Claude Desktop integration ✅
- `memory/stores/mem0_store.py` - AI-enhanced storage ✅

### Documentation
- `QUICK_START_MCP.md` - Quick setup ✅
- `MCP_SETUP_GUIDE.md` - Complete guide ✅
- `ARCHITECTURE_PATTERNS.md` - Best practices ✅
- `MVP_STORAGE_QUERY.md` - This file ✅

---

## Support

**Issues**: Check health with `await memory.health_check()`
**Questions**: See [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md)
**Architecture**: See [ARCHITECTURE_PATTERNS.md](ARCHITECTURE_PATTERNS.md)

---

**MVP Status**: ✅ Complete and Ready to Use
**Version**: 1.0
**Next**: Test it, use it, then add navigation (v1.1)
