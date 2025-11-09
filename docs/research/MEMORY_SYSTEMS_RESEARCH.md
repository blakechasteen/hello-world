# Memory Systems Research: LangMem, Graphiti, Mem0, SuperMemory

**Date**: November 7, 2025
**Purpose**: Understand state-of-the-art memory systems before implementing HoloLoom memory roadmap

---

## Executive Summary

Four leading memory systems studied:
1. **LangMem** (LangChain) - Agent-driven memory with background consolidation
2. **Graphiti** (Zep) - Temporal knowledge graphs for dynamic environments
3. **Mem0** - Multi-level memory layer with 26% accuracy improvement
4. **SuperMemory** - Knowledge graph with evolving relationships

**Key Insight**: All four systems solve different slices of the memory problem. HoloLoom should integrate best ideas from each rather than choosing one.

---

## System Comparison Matrix

| Feature | LangMem | Graphiti | Mem0 | SuperMemory | HoloLoom (Current) |
|---------|---------|----------|------|-------------|-------------------|
| **Memory Types** | Semantic, Episodic, Procedural | Episodic, Semantic, Community | User, Session, Agent | Documents → Memories | Preferences, Facts, Knowledge, Episodic, Procedural |
| **Storage** | Postgres, InMemory | Neo4j, FalkorDB, Kuzu, Neptune | Vector stores | Knowledge graph | NetworkX, Qdrant, Config |
| **Temporal Awareness** | ❌ Not mentioned | ✅ Bi-temporal (event + ingestion) | ❌ Not mentioned | ✅ Updates tracked via `isLatest` | ⚠️ Timestamps only (no bi-temporal) |
| **Graph Structure** | ❌ Vector-based | ✅ Entity-relationship triplets | ❌ Vector-based | ✅ Memories with 3 relationship types | ✅ NetworkX MultiDiGraph |
| **Agent Involvement** | ✅ Active (agent decides what to store) | ⚠️ Passive (automatic extraction) | ⚠️ Passive (add/search) | ⚠️ Passive (document → memories) | ⚠️ Passive (reflection buffer) |
| **Lifecycle Management** | ❌ Not detailed | ✅ Temporal edge invalidation | ⚠️ User/session/agent levels | ✅ Updates/extends/derives | ❌ No explicit lifecycles |
| **Retrieval Strategy** | Semantic search | Hybrid (semantic + BM25 + graph) | Semantic search | Semantic + relationships | Semantic + graph traversal + spectral |
| **Performance** | Not specified | <200ms target | 91% faster than full context | 1-2 min for 100pg PDF | <500ms for queries |
| **Portability** | LangChain ecosystem | Graph database export | Python/TypeScript SDKs | Not mentioned | ❌ Locked in HoloLoom |
| **Background Processing** | ✅ Knowledge consolidation | ✅ Incremental updates | ❌ Not mentioned | ✅ Async processing pipeline | ❌ Manual reflection only |

---

## Deep Dive: LangMem

### Architecture Philosophy
**"Agents learn and adapt from their interactions over time"**

### Key Innovation: Two-Path Design
1. **Hot Path** (in-conversation):
   - Agent actively manages memory using tools
   - `create_manage_memory_tool` - agent decides what/when to store
   - `create_search_memory_tool` - retrieval during conversation

2. **Background Path** (asynchronous):
   - Automatic knowledge consolidation
   - Converts episodic memories → semantic facts
   - No user/agent interruption

### Memory Types
- **Semantic**: Important facts extracted from conversations
- **Episodic**: Individual interaction records
- **Procedural**: Tools and behaviors agents optimize over time

### Storage
- **Development**: `InMemoryStore` (non-persistent)
- **Production**: `AsyncPostgresStore` (persistent across sessions)
- **Embeddings**: OpenAI text-embedding-3-small for semantic search

### Key Differentiator from Traditional RAG
**Agent participation**: Agent decides what to remember, not passive document indexing.

### What HoloLoom Can Learn
1. ✅ **Adopt**: Background consolidation thread (automatic semantic → episodic conversion)
2. ✅ **Adopt**: Agent-controlled memory tools (let agent decide what to store)
3. ⚠️ **Consider**: Postgres backend for production (in addition to NetworkX/Qdrant)

---

## Deep Dive: Graphiti

### Architecture Philosophy
**"Temporally-aware knowledge graphs for AI agents in dynamic environments"**

### Key Innovation: Bi-Temporal Data Model
- **Event time**: When the event occurred
- **Ingestion time**: When we learned about it
- **Use case**: Point-in-time queries ("What did we know on Oct 12?")

### Graph Structure
**Triplets**: Entity1 → Relationship → Entity2

Example: "Kendra loves Adidas shoes"
- Entity1: Kendra
- Relationship: loves
- Entity2: Adidas shoes

### Temporal Edge Invalidation
**Problem**: What happens when info changes?
**Solution**: Invalidate old edge, add new edge with timestamp

Example:
1. 2024-01-01: "Kendra loves Adidas shoes" (valid)
2. 2024-06-01: "Kendra loves Nike shoes" (new info)
3. Action: Invalidate Adidas edge (mark invalid_from=2024-06-01), add Nike edge

### Storage Backends
- Neo4j 5.26+
- FalkorDB 1.1.2+
- Kuzu 0.11.2+
- Amazon Neptune + OpenSearch

### Retrieval: Hybrid Search
1. **Semantic embeddings** (vector similarity)
2. **BM25** (keyword search)
3. **Graph traversal** (relationship paths)

**Performance**: Sub-200ms at scale (no LLM summarization needed)

### Memory Consolidation
**NOT** LLM-driven summarization (expensive, slow).
**INSTEAD**: Graph structure + temporal invalidation.

**Incremental updates**: New episodes integrate without batch recomputation.

### API Design
```python
graphiti = Graphiti(uri="neo4j://...", credentials=...)

# Add episode (text or structured)
await graphiti.add_episode(
    name="Kendra's preference",
    episode_body="Kendra loves Nike shoes",
    source_description="Conversation on 2024-06-01"
)

# Search (hybrid: semantic + BM25 + graph)
results = await graphiti.search(
    query="What shoes does Kendra like?",
    num_results=10
)
```

### What HoloLoom Can Learn
1. ✅ **CRITICAL**: Implement bi-temporal model (event_time + ingestion_time)
2. ✅ **CRITICAL**: Temporal edge invalidation (for facts that change)
3. ✅ **Adopt**: Hybrid retrieval (semantic + BM25 + graph)
4. ⚠️ **Consider**: Neo4j backend for production (in addition to NetworkX)
5. ✅ **Adopt**: Incremental updates (no batch recomputation)

---

## Deep Dive: Mem0

### Architecture Philosophy
**"Universal memory layer for personalized AI agents"**

### Key Innovation: Multi-Level Memory
1. **User-level**: Long-term preferences (across all sessions)
2. **Session-level**: Conversation-specific (ephemeral)
3. **Agent-level**: System state and behaviors

**This directly maps to transcript "Lifecycle Separation" principle!**

### Performance Benchmarks
- **+26% accuracy** vs OpenAI native memory (LOCOMO benchmark)
- **91% faster** than full-context approaches
- **90% token reduction** vs exhaustive context

### API Design
```python
memory = Memory()

# Add memory (with user/session scoping)
memory.add(messages, user_id="blake", session_id="session_123")

# Search (scoped retrieval)
memories = memory.search(
    query="What are my preferences?",
    user_id="blake",
    limit=3
)

# Context injection
context = "\n".join([m["text"] for m in memories])
response = llm.chat(f"Context: {context}\n\nUser: {message}")
```

### Storage
- Multiple vector store backends supported
- Improved vector store support in v1.0.0

### Deployment Models
- **Hosted**: app.mem0.ai (managed infrastructure)
- **Self-hosted**: Open-source deployment

### What HoloLoom Can Learn
1. ✅ **CRITICAL**: Multi-level memory (user/session/agent) - EXACTLY what transcript recommends
2. ✅ **Adopt**: Scoped retrieval (retrieve from specific levels only)
3. ✅ **Adopt**: Context injection pattern (selective, not exhaustive)
4. ✅ **Validate**: HoloLoom already has better performance (<500ms vs 91% improvement claim)

---

## Deep Dive: SuperMemory

### Architecture Philosophy
**"Knowledge graph that mimics human cognitive processes"**

### Key Innovation: Documents → Memories Transformation
**Documents** = Raw input (PDFs, web pages, text, images, videos)
**Memories** = Intelligent semantic units with:
- Semantic chunks with meaning
- Embeddings for similarity
- Interconnected relationships that evolve

### Relationship Types
1. **Updates**: New info replaces old (tracked via `isLatest` flag)
2. **Extends**: New info enriches without replacing
3. **Derives**: Inferred connections based on patterns

### Processing Pipeline
Queued → Extracting → Chunking → Embedding → Indexing → Done

**Performance**:
- 100-page PDF: 1-2 minutes
- 1-hour video: 5-10 minutes

### What HoloLoom Can Learn
1. ✅ **Adopt**: Updates/Extends/Derives relationship types (better than generic RELATES_TO)
2. ✅ **Adopt**: `isLatest` flag for versioning
3. ⚠️ **Consider**: Background processing pipeline for large documents
4. ✅ **Validate**: HoloLoom already has similar chunking (SpinningWheel spinners)

---

## Synthesis: Best Ideas for HoloLoom

### 🔴 Priority 0: Multi-Level Memory (Mem0 approach)
**Why**: Directly implements transcript Principle 2 ("Separate by lifecycle")

```python
# HoloLoom should adopt Mem0's three-level model
class MemoryScope(Enum):
    USER = "user"        # Personal preferences (PERMANENT)
    SESSION = "session"  # Conversation state (EPHEMERAL)
    AGENT = "agent"      # System behaviors (TEMPORARY)

# Scoped retrieval
memories = memory_manager.search(
    query="What are my style preferences?",
    scope=MemoryScope.USER,  # Only search user-level
    limit=10
)
```

**Implementation**: Use Mem0's API design pattern in `HoloLoom/memory/lifecycle_manager.py`

---

### 🔴 Priority 0b: Bi-Temporal Model (Graphiti approach)
**Why**: Critical for "What did we know on Oct 12?" queries

```python
# Every edge needs TWO timestamps
class TemporalEdge:
    event_time: datetime      # When event occurred
    ingestion_time: datetime  # When we learned about it
    valid_from: datetime      # When edge became valid
    valid_to: Optional[datetime]  # When edge was invalidated (None = still valid)

# Example: Preference change
# 2024-01-01: "Blake prefers Python" (event_time=2024-01-01, ingestion_time=2024-01-01, valid_from=2024-01-01, valid_to=None)
# 2024-06-01: "Blake prefers Rust" (event_time=2024-06-01, ingestion_time=2024-06-01, valid_from=2024-06-01, valid_to=None)
# Action: Invalidate Python edge (valid_to=2024-06-01), keep for historical queries
```

**Implementation**: Extend `KGEdge` in `HoloLoom/memory/graph.py`

---

### 🟠 Priority 1: Agent-Controlled Memory (LangMem approach)
**Why**: Agent decides what to remember (not passive accumulation)

```python
# LangMem pattern: Tools for memory management
class MemoryTools:
    @tool
    def store_memory(self, content: str, type: MemoryType, scope: MemoryScope):
        """Agent calls this to explicitly store important info."""
        pass

    @tool
    def search_memory(self, query: str, scope: MemoryScope):
        """Agent calls this to retrieve relevant context."""
        pass

# Agent reasoning:
# "User mentioned they prefer concise responses. This is a USER-level PREFERENCE.
#  I should store this explicitly."
agent.call_tool("store_memory", {
    "content": "User prefers concise responses",
    "type": MemoryType.PREFERENCE,
    "scope": MemoryScope.USER
})
```

**Implementation**: Add memory tools to `HoloLoom/agentic/core.py`

---

### 🟠 Priority 1b: Background Consolidation (LangMem approach)
**Why**: Automatic episodic → semantic conversion without blocking queries

```python
# Background thread (like Full Learning Engine, but for memory consolidation)
class MemoryConsolidator:
    """Background task: Convert episodic → semantic."""

    async def consolidation_loop(self):
        """Run every 60 minutes."""
        while True:
            await asyncio.sleep(3600)  # 1 hour

            # Get recent episodic memories
            recent_episodes = self.memory.get_episodic(hours=24)

            # Extract semantic facts using LLM
            facts = await self._extract_facts(recent_episodes)

            # Store as semantic memories
            for fact in facts:
                self.memory.add_semantic(fact)

            # Archive episodic (mark as consolidated)
            for episode in recent_episodes:
                episode.metadata["consolidated"] = True
```

**Implementation**: Add to `HoloLoom/memory/consolidation.py` (NEW FILE)

---

### 🟡 Priority 2: Hybrid Retrieval (Graphiti approach)
**Why**: Semantic + BM25 + Graph traversal = better recall

```python
# Graphiti's hybrid approach
class HybridRetriever:
    async def retrieve(self, query: str, k: int = 10):
        # 1. Semantic search (embeddings)
        semantic_results = await self.vector_search(query, k=k*2)

        # 2. BM25 keyword search
        bm25_results = await self.keyword_search(query, k=k*2)

        # 3. Graph traversal (multi-hop)
        graph_results = await self.graph_search(query, k=k*2)

        # 4. Combine and rerank
        all_results = semantic_results + bm25_results + graph_results
        reranked = self._rerank(query, all_results)

        return reranked[:k]
```

**Implementation**: Extend `ModeAwareRetriever` in `HoloLoom/memory/task_aware_retrieval.py`

---

### 🟡 Priority 2b: Relationship Types (SuperMemory approach)
**Why**: Better than generic RELATES_TO

```python
# SuperMemory's three relationship types
class RelationshipType(Enum):
    UPDATES = "updates"    # New info replaces old
    EXTENDS = "extends"    # New info enriches old
    DERIVES = "derives"    # Inferred connection

    # Existing HoloLoom types (keep)
    IS_A = "IS_A"
    USES = "USES"
    MENTIONS = "MENTIONS"
    LEADS_TO = "LEADS_TO"

# Example: Version tracking
# Original: "Blake uses Python for backend"
# Update: "Blake uses Rust for backend"
# Relationship: new_edge.type = RelationshipType.UPDATES, old_edge.metadata["isLatest"] = False
```

**Implementation**: Extend `EdgeType` in `HoloLoom/memory/graph.py`

---

### 🟢 Priority 3: Incremental Updates (Graphiti approach)
**Why**: No batch recomputation (expensive, slow)

**Current HoloLoom Issue**: Adding new memories requires full graph traversal.

**Graphiti Solution**: Incremental edge addition with temporal invalidation.

```python
# Instead of rebuilding graph
kg.clear()
kg.rebuild_from_scratch(all_memories)  # SLOW

# Do incremental updates
kg.add_edge(entity1, entity2, relationship_type)  # FAST
if contradicts_existing:
    kg.invalidate_edge(old_edge, invalid_from=now)  # Mark old edge invalid
```

**Implementation**: Already partially implemented in `HoloLoom/memory/graph.py`, optimize further

---

## Comparative Advantages Table

| Feature | LangMem | Graphiti | Mem0 | SuperMemory | **HoloLoom (Proposed)** |
|---------|---------|----------|------|-------------|------------------------|
| **Multi-level memory** | ❌ | ❌ | ✅ (user/session/agent) | ❌ | ✅ **ADOPT from Mem0** |
| **Bi-temporal model** | ❌ | ✅ (event + ingestion) | ❌ | ⚠️ (isLatest flag) | ✅ **ADOPT from Graphiti** |
| **Agent-controlled** | ✅ (tools) | ❌ | ❌ | ❌ | ✅ **ADOPT from LangMem** |
| **Background consolidation** | ✅ | ✅ (incremental) | ❌ | ✅ (pipeline) | ✅ **ADOPT from LangMem** |
| **Hybrid retrieval** | ❌ (semantic only) | ✅ (semantic+BM25+graph) | ❌ (semantic only) | ⚠️ (semantic+graph) | ✅ **ADOPT from Graphiti** |
| **Relationship types** | ❌ | ⚠️ (generic triplets) | ❌ | ✅ (updates/extends/derives) | ✅ **ADOPT from SuperMemory** |
| **Temporal invalidation** | ❌ | ✅ | ❌ | ⚠️ (isLatest) | ✅ **ADOPT from Graphiti** |
| **Performance** | Not specified | <200ms | 91% faster | 1-2min for 100pg | **<500ms (already better)** ✅ |
| **Graph structure** | ❌ | ✅ (Neo4j/FalkorDB) | ❌ | ✅ (custom) | ✅ (NetworkX, can add Neo4j) |
| **Portability** | ⚠️ (LangChain only) | ✅ (graph export) | ⚠️ (SDK-based) | ❌ | **PLANNED (Priority 1)** |

---

## Implementation Roadmap (Updated with Research)

### Week 1: Multi-Level Memory + Bi-Temporal Model
**From**: Mem0 (multi-level) + Graphiti (bi-temporal)

**Files**:
- `HoloLoom/memory/lifecycle_manager.py` - Multi-level memory (USER/SESSION/AGENT)
- `HoloLoom/memory/graph.py` - Add bi-temporal edges (event_time + ingestion_time + valid_from + valid_to)

**Tests**:
- `test_multi_level_memory.py` - Test user/session/agent scoping
- `test_bitemporal_model.py` - Test temporal edge invalidation

---

### Week 2: Agent-Controlled Memory + Background Consolidation
**From**: LangMem (both features)

**Files**:
- `HoloLoom/agentic/memory_tools.py` - Memory management tools for agents
- `HoloLoom/memory/consolidation.py` - Background episodic → semantic conversion

**Tests**:
- `test_agent_memory_control.py` - Test agent-driven storage
- `test_background_consolidation.py` - Test async consolidation loop

---

### Week 3: Hybrid Retrieval + Relationship Types
**From**: Graphiti (hybrid retrieval) + SuperMemory (relationship types)

**Files**:
- `HoloLoom/memory/hybrid_retrieval.py` - Semantic + BM25 + Graph
- `HoloLoom/memory/graph.py` - Add UPDATES/EXTENDS/DERIVES relationships

**Tests**:
- `test_hybrid_retrieval.py` - Test combined retrieval strategies
- `test_relationship_types.py` - Test updates/extends/derives

---

### Week 4: Portability + Production Backends
**From**: Graphiti (Neo4j) + Research goals

**Files**:
- `HoloLoom/memory/portability.py` - Export to Neo4j, Mem0, LangMem formats
- `HoloLoom/memory/neo4j_backend.py` - Production Neo4j backend (optional)

**Tests**:
- `test_portability.py` - Test export/import across systems
- `test_neo4j_backend.py` - Test Neo4j integration

---

## Key Takeaways

### What HoloLoom Already Does Better
1. ✅ **Multi-problem separation**: Already separates preferences/facts/knowledge/episodic/procedural
2. ✅ **Performance**: <500ms vs Graphiti <200ms (close), Mem0 "91% faster" (vague)
3. ✅ **Matryoshka embeddings**: More sophisticated than single-scale embeddings in other systems
4. ✅ **Thompson Sampling**: Exploration/exploitation balance (none of the other systems have this)

### What HoloLoom Must Adopt
1. 🔴 **Multi-level memory** (Mem0) - USER/SESSION/AGENT scoping
2. 🔴 **Bi-temporal model** (Graphiti) - Event time + ingestion time
3. 🟠 **Agent-controlled memory** (LangMem) - Let agent decide what to store
4. 🟠 **Background consolidation** (LangMem) - Episodic → semantic without blocking
5. 🟡 **Hybrid retrieval** (Graphiti) - Semantic + BM25 + graph
6. 🟡 **Relationship types** (SuperMemory) - UPDATES/EXTENDS/DERIVES

### What HoloLoom Can Skip
1. ❌ **Postgres backend** (LangMem) - NetworkX + Qdrant sufficient for now
2. ❌ **Video processing** (SuperMemory) - Not core to memory problem
3. ❌ **Managed hosting** (Mem0) - Local-first is better for privacy

---

## Next Steps

1. **Implement** Multi-level memory (Week 1) - CRITICAL for lifecycle separation
2. **Implement** Bi-temporal model (Week 1) - CRITICAL for temporal queries
3. **Prototype** Agent-controlled memory tools (Week 2)
4. **Test** Background consolidation (Week 2)
5. **Evaluate** Neo4j backend for production (Week 4+)

**Recommendation**: Start with Week 1 (multi-level + bi-temporal) as these are foundational changes that affect all other features.

---

**Research Date**: November 7, 2025
**Next Review**: After Week 1 implementation
**Status**: Ready to implement
