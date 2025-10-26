# Loom Memory Integration: Token-Efficient Context Retrieval
**Date**: 2025-10-24
**Status**: Architecture design

## 🎯 Core Insight

**Memory retrieval directly impacts the loom's token budget.**

The loom operates in cycles with different execution modes (BARE/FAST/FUSED). Each mode has a different token budget. Memory context consumed during retrieval affects how many tokens are left for reasoning and response.

**Goal**: Adaptive memory retrieval that optimizes for token efficiency per loom cycle.

---

## 🧵 How Memory Plugs Into The Loom

### The Weaving Cycle (from CLAUDE.md)

```
1. Loom Command selects Pattern Card (BARE/FAST/FUSED)
2. Chrono Trigger fires, creates TemporalWindow
3. Yarn Graph threads selected based on temporal window  <-- MEMORY RETRIEVAL
4. Resonance Shed lifts feature threads, creates DotPlasma
5. Warp Space tensions threads into continuous manifold
6. Convergence Engine collapses to discrete tool selection
7. Tool executes, results woven into Spacetime fabric
8. Reflection Buffer learns from outcome
9. Chrono Trigger detensions, cycle completes
```

**Step 3 is critical**: Yarn Graph (memory) determines what context enters the cycle.

### Token Flow

```
User Query (input tokens)
    ↓
Memory Retrieval → Context (context tokens)
    ↓
Features Extraction → DotPlasma (processing)
    ↓
Policy Decision → Tool Selection (reasoning tokens)
    ↓
Response Generation (output tokens)

TOTAL = input + context + reasoning + output
```

**Context tokens are controllable via memory retrieval strategy!**

---

## 🎨 Pattern Card → Memory Strategy Mapping

### Mode Definitions (from config.py)

**BARE Mode:**
- Minimal processing
- Fastest execution
- Lowest token budget
- Use case: Quick lookups, simple queries

**FAST Mode:**
- Balanced processing
- Medium token budget
- Use case: Normal queries, moderate complexity

**FUSED Mode:**
- Full processing
- Maximum token budget
- Use case: Complex reasoning, deep analysis

### Memory Strategy Per Mode

```python
class PatternCard:
    """
    Pattern card that determines execution strategy.
    Now includes memory retrieval optimization.
    """
    mode: Mode  # BARE, FAST, FUSED

    def memory_config(self) -> Dict[str, Any]:
        """
        Memory retrieval configuration based on cycle mode.
        Optimizes for token efficiency.
        """
        if self.mode == Mode.BARE:
            return {
                "store": "neo4j",           # Graph traversal only
                "strategy": Strategy.GRAPH,  # Symbolic relationships
                "limit": 3,                  # Minimal context
                "max_tokens": 500,           # Hard cap on context
                "prefer_recent": True        # Temporal bias
            }

        elif self.mode == Mode.FAST:
            return {
                "store": "qdrant",           # Vector similarity only
                "strategy": Strategy.SEMANTIC, # Meaning-based
                "limit": 5,                  # Moderate context
                "max_tokens": 1000,          # Medium cap
                "prefer_recent": False       # Semantic relevance
            }

        else:  # FUSED
            return {
                "store": "hybrid",           # Neo4j + Qdrant fusion
                "strategy": Strategy.FUSED,  # Best of both
                "limit": 7,                  # Rich context
                "max_tokens": 2000,          # High quality
                "prefer_recent": False       # Pure relevance
            }
```

---

## 🔀 Hybrid Fusion Strategy

### Why Neo4j + Qdrant?

**Neo4j (Symbolic)**:
- Graph relationships: "Hive Jodi" → "Winter Prep" → "Insulation"
- Explicit connections
- Guaranteed relevance (if connected, it's relevant)
- Can store embeddings as properties (symbolic vectors!)

**Qdrant (Semantic)**:
- Vector similarity: "winter preparation" ≈ "cold weather care"
- Implicit connections
- Catches semantic matches missed by graph
- Optimized for speed (ANN search)

**Fusion (Symbolic + Semantic)**:
```python
# Example query: "winter prep for Hive Jodi"

# Neo4j returns (via graph traversal):
# - "Hive Jodi needs insulation" (connected via PREPARES_FOR)
# - "Last inspection showed weak colony" (connected via HAS_INSPECTION)
# Scores: [0.92, 0.85] (relationship strength)

# Qdrant returns (via vector similarity):
# - "Winter feeding strategy for weak hives" (semantic match)
# - "Cold weather preparation checklist" (semantic match)
# Scores: [0.88, 0.81] (cosine similarity)

# Hybrid fusion:
def fuse_scores(neo_result, qdrant_result, mode):
    if mode == Mode.FUSED:
        # High quality: balanced fusion
        alpha = 0.6  # Neo4j weight
        beta = 0.4   # Qdrant weight
    elif mode == Mode.FAST:
        # Semantic bias
        alpha = 0.3
        beta = 0.7
    else:  # BARE
        # Symbolic only (shouldn't hit this path)
        alpha = 1.0
        beta = 0.0

    # Combine and re-rank
    combined = []
    for mem, score in neo_result:
        combined.append((mem, alpha * score, "neo4j"))
    for mem, score in qdrant_result:
        combined.append((mem, beta * score, "qdrant"))

    # Sort by fused score
    combined.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate (same memory from both sources)
    seen = set()
    deduped = []
    for mem, score, source in combined:
        if mem.id not in seen:
            deduped.append((mem, score))
            seen.add(mem.id)

    return deduped[:limit]  # Return top K
```

**Result**: Most relevant memories with minimal noise = token-efficient context!

---

## 📊 Token Budget Examples

### Scenario 1: BARE Mode Query
```
Query: "Check Hive Jodi"
Mode: BARE
Memory: Neo4j only, limit=3

Retrieved context (3 memories):
1. "Hive Jodi last inspection: 8 frames brood" (120 tokens)
2. "Hive Jodi needs winter prep" (80 tokens)
3. "Hive Jodi located in north apiary" (90 tokens)

Total context: ~290 tokens
Reasoning budget: ~200 tokens
Response budget: ~100 tokens
TOTAL CYCLE: ~590 tokens

✅ Fast, cheap, good enough for simple query
```

### Scenario 2: FAST Mode Query
```
Query: "What winter prep does Hive Jodi need?"
Mode: FAST
Memory: Qdrant semantic, limit=5

Retrieved context (5 memories):
1. "Winter preparation checklist" (150 tokens)
2. "Hive Jodi needs insulation" (100 tokens)
3. "Weak colonies require sugar fondant" (120 tokens)
4. "Mouse guards installation before November" (110 tokens)
5. "Ventilation important in winter" (100 tokens)

Total context: ~580 tokens
Reasoning budget: ~400 tokens
Response budget: ~200 tokens
TOTAL CYCLE: ~1180 tokens

✅ Good quality, semantic understanding, moderate cost
```

### Scenario 3: FUSED Mode Query
```
Query: "Comprehensive winter strategy for Hive Jodi given last inspection showed weakness"
Mode: FUSED
Memory: Hybrid Neo4j+Qdrant, limit=7

Retrieved context (7 memories, fused):
1. "Hive Jodi inspection Oct 15: weak colony, 8 frames" (140 tokens) [Neo4j: 0.95]
2. "Winter feeding for weak colonies: sugar fondant" (130 tokens) [Hybrid: 0.92]
3. "Insulation techniques for overwintering" (150 tokens) [Qdrant: 0.88]
4. "Mouse guard installation procedure" (100 tokens) [Qdrant: 0.85]
5. "Hive Jodi located north apiary (cold exposure)" (90 tokens) [Neo4j: 0.83]
6. "Weak colony survival strategies" (140 tokens) [Qdrant: 0.82]
7. "Ventilation prevents moisture buildup" (120 tokens) [Qdrant: 0.79]

Total context: ~870 tokens
Reasoning budget: ~600 tokens
Response budget: ~300 tokens
TOTAL CYCLE: ~1770 tokens

✅ Highest quality, complete picture, worth the cost for complex query
```

---

## 🚀 Implementation in Loom Command

### Modified LoomCommand

```python
from HoloLoom.loom.command import LoomCommand, PatternCard, Mode
from HoloLoom.memory.protocol import UnifiedMemoryInterface, Strategy

class AdaptiveMemoryLoom:
    """
    Loom with adaptive memory retrieval.
    Memory strategy tied to Pattern Card mode.
    """

    def __init__(self, memory: UnifiedMemoryInterface):
        self.memory = memory

    async def execute_cycle(self, query: str, pattern: PatternCard):
        """Execute loom cycle with adaptive memory."""

        # 1. Get memory config for this mode
        mem_config = pattern.memory_config()

        # 2. Retrieve context (token-optimized)
        context_result = await self.memory.recall(
            query=query,
            strategy=mem_config["strategy"],
            limit=mem_config["limit"]
        )

        # 3. Enforce token budget
        context = self._truncate_context(
            context_result.memories,
            max_tokens=mem_config["max_tokens"]
        )

        # 4. Continue with rest of cycle
        # - Feature extraction (DotPlasma)
        # - Policy decision
        # - Tool execution
        # - Response generation

        return response

    def _truncate_context(self, memories, max_tokens):
        """Ensure context fits within token budget."""
        total = 0
        truncated = []
        for mem in memories:
            # Rough estimate: 1 token ≈ 4 chars
            est_tokens = len(mem.text) // 4
            if total + est_tokens <= max_tokens:
                truncated.append(mem)
                total += est_tokens
            else:
                break
        return truncated
```

### Chrono Trigger Integration

```python
class ChronoTrigger:
    """
    Temporal control with memory-aware cycle budgets.
    """

    def create_temporal_window(self, pattern: PatternCard):
        """
        Create execution window with token budget.
        Memory retrieval budget included.
        """
        mem_config = pattern.memory_config()

        return TemporalWindow(
            mode=pattern.mode,
            memory_budget=mem_config["max_tokens"],  # Tokens for context
            reasoning_budget=self._reasoning_budget(pattern.mode),
            response_budget=self._response_budget(pattern.mode),
            timeout=self._timeout(pattern.mode)
        )
```

---

## 🎯 Benefits of This Architecture

### 1. Self-Optimizing Token Efficiency
- BARE mode: minimal context, fast
- FAST mode: semantic relevance, balanced
- FUSED mode: comprehensive, high quality
- **System automatically adapts to query complexity**

### 2. Pluggable Memory Backends
- Can swap Neo4j ↔ Qdrant ↔ Hybrid
- Graceful degradation (falls back to InMemory)
- Test with simple store, deploy with production

### 3. Symbolic + Semantic Fusion
- Neo4j: "What's connected?" (symbolic)
- Qdrant: "What's similar?" (semantic)
- Hybrid: "What's relevant?" (both)
- **Best of both worlds**

### 4. Clear Resource Budgets
- Memory retrieval budget
- Reasoning budget
- Response budget
- **No token waste**

### 5. Hyperspace Navigation
- Neo4j symbolic vectors = embeddings as graph properties
- Qdrant semantic vectors = pure vector space
- Navigate in both symbolic AND semantic space
- **True hyperspace traversal**

---

## 🔧 Next Steps

1. ✅ **Build InMemoryStore integration test** (validates architecture)
2. 🎯 **Implement Neo4j store with symbolic vectors**
3. 🎯 **Implement Qdrant vector store**
4. 🎯 **Build hybrid fusion strategy**
5. 🎯 **Integrate with LoomCommand pattern cards**
6. 🎯 **Test with beekeeping data**
7. 🎯 **Benchmark token efficiency per mode**

---

## 💎 The Vision

```
User: "What winter prep does Hive Jodi need?"

Loom:
  - Analyzes query complexity → selects FAST mode
  - Pattern card → qdrant semantic search, limit=5
  - Retrieves relevant memories (~600 tokens)
  - Features + Policy + Response (~600 tokens)
  - TOTAL: ~1200 tokens, <1s latency
  - Result: Accurate, efficient, fast

User: "Comprehensive winter strategy for all weak hives considering historical patterns"

Loom:
  - Analyzes query complexity → selects FUSED mode
  - Pattern card → hybrid neo4j+qdrant, limit=7
  - Graph: historical connections
  - Vectors: semantic patterns
  - Fusion: complete picture (~1500 tokens)
  - Deep reasoning (~800 tokens)
  - TOTAL: ~2300 tokens, ~3s latency
  - Result: Comprehensive, insightful, worth the cost
```

**The loom adapts its memory strategy to the task at hand.**
**Token efficiency through intelligent retrieval.**
**Welcome to hyperspace.**

---

**Status**: Ready to implement
**Blocker**: Need InMemoryStore test first
**Timeline**: ~5 hours to full hybrid stack
