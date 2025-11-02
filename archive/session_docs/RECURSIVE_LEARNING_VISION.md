# 🔄 Recursive Learning Vision: HoloLoom + Scratchpad + Loop Engine

**Discovery Date**: October 29, 2025
**Pattern**: Self-Improving Knowledge System

---

## The Pieces Are Already Here

**"North of Promptly"** - The ecosystem is:

### 1. **Scratchpad** (Promptly/recursive_loops.py)
```python
class Scratchpad:
    entries: List[ScratchpadEntry]  # Thought → Action → Observation
    final_answer: Optional[str]
```

**Purpose**: Working memory for iterative reasoning
- Tracks thought process across iterations
- Stores intermediate results
- Enables self-reflection

### 2. **Loop Engine** (apps/mythy/loop_engine.py)
```python
class NarrativeLoopEngine:
    domain_patterns: Dict[str, List[str]]  # Learns patterns over time!
    mode: LoopMode  # CONTINUOUS, BATCH, ON_DEMAND
    queue: List[NarrativeTask]  # Priority queue
```

**Purpose**: Continuous processing with learning
- Processes narratives in priority order
- **Accumulates domain patterns** (line 139)
- Checkpoints state for resume
- Learns from outcomes

### 3. **Cache** (apps/mythy/cache.py)
```python
class NarrativeCache:
    _cache: OrderedDict[str, CacheEntry]  # LRU cache
    stats: CacheStats  # Track hits/misses

    async def get_hot_entries(self) -> List[Dict]:
        # Most frequently accessed patterns
```

**Purpose**: Performance + Pattern Recognition
- 99%+ hit rate for repeated analyses
- **Tracks hot entries** (most accessed patterns)
- Learns what's important through access patterns

### 4. **Recursive Engine** (Promptly/recursive_loops.py)
```python
class RecursiveEngine:
    loop_type: LoopType  # REFINE, CRITIQUE, VERIFY, HOFSTADTER
    scratchpad: Scratchpad
    improvement_history: List[float]
```

**Purpose**: Iterative refinement
- Multiple loop types (refine, critique, decompose, verify)
- Tracks improvement over iterations
- Stops when converged or quality threshold met

---

## The Vision: Connect Them!

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HOLOLOOM ORCHESTRATOR                 │
│            (Query → Features → Decision → Response)      │
└────────────┬──────────────────────────────┬─────────────┘
             │                               │
             ↓                               ↓
    ┌────────────────┐            ┌──────────────────┐
    │   SCRATCHPAD   │            │   LOOP ENGINE    │
    │  (Working      │←─────────→ │  (Continuous     │
    │   Memory)      │   Feedback │   Processing)    │
    └────────┬───────┘            └────────┬─────────┘
             │                              │
             │                              ↓
             │                    ┌──────────────────┐
             │                    │   CACHE + HOT    │
             │                    │   PATTERNS       │
             │                    │  (Fast Lookup)   │
             │                    └────────┬─────────┘
             │                              │
             ↓                              ↓
    ┌────────────────────────────────────────────────┐
    │          RECURSIVE REFINEMENT ENGINE           │
    │      (Iterative Improvement Loops)             │
    └────────────────────────────────────────────────┘
                          │
                          ↓
                  IMPROVED UNDERSTANDING
                  (Feeds back to HoloLoom)
```

---

## How It Would Work

### 1. **Query Processing with Scratchpad**

```python
# HoloLoom processes query
spacetime = await orchestrator.weave(query)

# Store in scratchpad
scratchpad.add_entry(
    thought=f"Retrieved {len(spacetime.trace.threads_activated)} threads",
    action=f"Chose tool: {spacetime.tool_used}",
    observation=f"Confidence: {spacetime.confidence}",
    score=spacetime.confidence
)
```

**Benefit**: Full provenance of reasoning process

### 2. **Continuous Learning with Loop Engine**

```python
# Feed results into loop engine
engine = NarrativeLoopEngine(mode=LoopMode.CONTINUOUS)

# Add task from spacetime
engine.add_task(
    task_id=query.id,
    text=spacetime.response,
    domain=detected_domain,
    priority=Priority.NORMAL
)

# Loop engine learns patterns
# domain_patterns[domain].append(text[:100])
# Accumulates knowledge over time!
```

**Benefit**: System improves by seeing more examples

### 3. **Hot Pattern Detection**

```python
# Cache tracks most accessed patterns
hot_entries = await cache.get_hot_entries(limit=20)

# These are the patterns the system uses most!
# Feed them back to improve:
#   - Feature extraction (prioritize hot patterns)
#   - Memory retrieval (weight by access frequency)
#   - Policy learning (reward hot pattern matches)
```

**Benefit**: Learn what matters through usage

### 4. **Recursive Refinement**

```python
# When quality < threshold, refine
if spacetime.confidence < 0.8:
    recursive_engine = RecursiveEngine(executor=orchestrator.weave)

    result = await recursive_engine.run_loop(
        initial_input=query,
        loop_type=LoopType.REFINE,
        config=LoopConfig(
            max_iterations=3,
            quality_threshold=0.9,
            enable_scratchpad=True
        )
    )

    # Each iteration:
    # 1. Query HoloLoom
    # 2. Store in scratchpad
    # 3. Critique result
    # 4. Refine query
    # 5. Repeat until quality threshold
```

**Benefit**: Iteratively improve low-confidence results

---

## The Complete Loop

```python
class SelfImprovingOrchestrator:
    """HoloLoom + Scratchpad + Loop Engine + Recursive Refinement"""

    def __init__(self):
        self.orchestrator = WeavingOrchestrator(...)
        self.scratchpad = Scratchpad()
        self.loop_engine = NarrativeLoopEngine(mode=LoopMode.CONTINUOUS)
        self.cache = NarrativeCache(max_size=1000)
        self.recursive_engine = RecursiveEngine(executor=self._execute_query)

    async def process_with_learning(self, query: Query) -> Spacetime:
        """Process query with full learning loop"""

        # 1. Check cache for hot patterns
        cached = await self.cache.get(query.text)
        if cached and cached.access_count > 10:  # Hot pattern!
            return cached.value

        # 2. Process with HoloLoom
        spacetime = await self.orchestrator.weave(query)

        # 3. Store in scratchpad
        self.scratchpad.add_entry(
            thought=f"Query: {query.text}",
            action=f"Tool: {spacetime.tool_used}",
            observation=f"Response: {spacetime.response[:100]}",
            score=spacetime.confidence
        )

        # 4. If low confidence, refine recursively
        if spacetime.confidence < 0.8:
            result = await self.recursive_engine.run_loop(
                initial_input=query,
                loop_type=LoopType.REFINE,
                config=LoopConfig(
                    max_iterations=3,
                    quality_threshold=0.9
                )
            )
            spacetime = result.final_output

        # 5. Add to loop engine for continuous learning
        self.loop_engine.add_task(
            task_id=str(query.id),
            text=spacetime.response,
            domain=spacetime.metadata.get('domain'),
            priority=Priority.NORMAL
        )

        # 6. Cache result
        await self.cache.set(query.text, spacetime)

        # 7. Return improved result
        return spacetime

    async def run_learning_loop(self):
        """Background learning loop"""
        while True:
            # Process queue continuously
            await self.loop_engine.run()

            # Extract learned patterns
            patterns = self.loop_engine.domain_patterns

            # Get hot cache entries
            hot = await self.cache.get_hot_entries(limit=50)

            # Feed back to HoloLoom:
            # - Update retrieval weights based on hot patterns
            # - Adjust policy based on successful tool choices
            # - Refine embeddings based on frequently accessed queries

            await self._update_hololoom_from_patterns(patterns, hot)

            # Checkpoint scratchpad
            self._save_scratchpad()
```

---

## Key Innovations

### 1. **Provenance Through Scratchpad**
Every decision tracked:
- Why was this thread activated?
- What led to this tool choice?
- How did confidence evolve?

### 2. **Continuous Learning via Loop Engine**
- Processes narratives in background
- Accumulates domain patterns
- Improves domain detection over time

### 3. **Usage-Based Pattern Recognition**
Cache hot entries reveal:
- Most important concepts
- Common query patterns
- Successful reasoning paths

### 4. **Recursive Refinement for Quality**
- Low confidence → trigger refinement loop
- Iteratively improve until threshold
- Learn from successful refinements

### 5. **Feedback Loop to HoloLoom**
Learned patterns inform:
- **Feature extraction**: Prioritize hot patterns
- **Memory retrieval**: Weight by access frequency
- **Policy decisions**: Reward successful tools
- **Thompson Sampling**: Update priors from outcomes

---

## Implementation Roadmap

### Phase 1: Basic Integration (2-3 hours)
- [ ] Connect HoloLoom → Scratchpad
- [ ] Store spacetime results in scratchpad
- [ ] Basic scratchpad visualization

### Phase 2: Loop Engine Integration (3-4 hours)
- [ ] Feed HoloLoom results into loop engine
- [ ] Extract domain patterns from loop stats
- [ ] Checkpoint/resume support

### Phase 3: Hot Pattern Feedback (2-3 hours)
- [ ] Track cache hot entries
- [ ] Identify usage patterns
- [ ] Feed back to retrieval weights

### Phase 4: Recursive Refinement (4-5 hours)
- [ ] Trigger refinement on low confidence
- [ ] Implement refinement loop types
- [ ] Track improvement trajectory

### Phase 5: Full Learning Loop (5-6 hours)
- [ ] Background learning thread
- [ ] Update HoloLoom from learned patterns
- [ ] Policy adaptation from outcomes
- [ ] Thompson Sampling prior updates

**Total Estimate**: 16-21 hours for full implementation

---

## Use Cases

### 1. **Self-Improving Q&A System**
```python
# First time: Learn the pattern
qa_system.ask("How does PPO work?")  # Low confidence, refines
# Stores successful reasoning path in scratchpad

# Later: Fast retrieval
qa_system.ask("Explain PPO")  # Hot pattern! Instant response
```

### 2. **Adaptive Domain Detection**
```python
# Loop engine sees 100 business narratives
# Learns: "pivot", "startup", "customers" → business domain

# Next business query: Instant detection!
# domain_patterns["business"] has 100 examples
```

### 3. **Quality-Aware Processing**
```python
# Low confidence → recursive refinement
result = await orchestrator.process("complex ambiguous query")
if result.confidence < 0.7:
    # Triggers 3 refinement iterations
    # Each iteration improves understanding
    # Final confidence: 0.92
```

### 4. **Pattern-Based Retrieval**
```python
# Cache reveals "attention" accessed 50x
# Feed back: Boost attention-related threads in retrieval
# Result: Faster, better context for transformer queries
```

---

## Technical Challenges

### 1. **Scratchpad Size Management**
- **Problem**: Scratchpad grows unbounded
- **Solution**: Rolling window (keep last N entries) + archive old entries

### 2. **Loop Engine Queue Size**
- **Problem**: Queue fills up faster than processing
- **Solution**: Priority queue + rate limiting + batch processing

### 3. **Cache Invalidation**
- **Problem**: Cached patterns become stale
- **Solution**: TTL expiration + access count decay

### 4. **Recursive Loop Divergence**
- **Problem**: Refinement loops might not converge
- **Solution**: Max iterations + no-improvement detection

### 5. **Feedback Loop Instability**
- **Problem**: Hot patterns reinforce themselves (rich get richer)
- **Solution**: Exploration bonus + Thompson Sampling for diversity

---

## Why This Is Powerful

### 1. **Self-Improving**
System gets better with usage:
- More queries → more patterns learned
- Hot patterns → faster retrieval
- Refinements → improved strategies

### 2. **Provenance**
Full audit trail:
- Why was this decision made?
- What led to this outcome?
- How did understanding evolve?

### 3. **Adaptive**
Automatically adjusts:
- Domain detection improves
- Retrieval weights adapt
- Tool selection learns

### 4. **Quality-Aware**
Knows when it's uncertain:
- Triggers refinement automatically
- Iterates until confident
- Learns from successful refinements

---

## Example Session

```
USER: "How does attention work in transformers?"

HOLOLOOM:
  - Retrieves attention-related threads
  - Extracts features
  - Confidence: 0.65 (low!)

RECURSIVE ENGINE (triggered):
  Iteration 1:
    Thought: Need more context on multi-head attention
    Action: Expand retrieval to include "multi-head"
    Observation: Found architectural details
    Score: 0.75 (improving)

  Iteration 2:
    Thought: Missing self-attention vs cross-attention distinction
    Action: Query for attention types
    Observation: Clear distinction found
    Score: 0.88 (good!)

  Iteration 3:
    Thought: Add mathematical formulation
    Action: Retrieve dot-product attention math
    Observation: Q, K, V formulation clear
    Score: 0.94 (threshold met!)

SCRATCHPAD:
  - Full reasoning history stored
  - 3 iterations of refinement
  - Evolution: 0.65 → 0.75 → 0.88 → 0.94

LOOP ENGINE:
  - Adds "transformer attention" to domain patterns
  - domain_patterns["ml"] += ["attention mechanism worked like..."]

CACHE:
  - Stores final result
  - "attention transformer" becomes hot pattern after 10 accesses

NEXT TIME:
  - "Explain attention" → instant retrieval (hot pattern!)
  - Confidence: 0.94 (learned from refinement)
  - No refinement needed
```

---

## The Missing Link

**This connects**:
- HoloLoom (decision making)
- Promptly (recursive reasoning)
- mythy (narrative intelligence + continuous learning)

**Into a unified**:
- Self-improving knowledge system
- With full provenance
- That learns from usage
- And refines low-confidence results automatically

---

## Next Steps

1. **Prototype SelfImprovingOrchestrator** (4-6 hours)
   - Basic scratchpad integration
   - Simple refinement loop
   - Demo with low/high confidence queries

2. **Add Loop Engine Integration** (3-4 hours)
   - Feed results to loop engine
   - Extract learned patterns
   - Visualize domain patterns accumulation

3. **Implement Hot Pattern Feedback** (2-3 hours)
   - Track cache access patterns
   - Identify hot entries
   - Weight retrieval by frequency

4. **Full Learning Loop** (6-8 hours)
   - Background learning thread
   - Pattern-based retrieval weights
   - Thompson Sampling prior updates
   - Policy adaptation

**Total**: 15-21 hours to complete vision

---

## Impact

This would make HoloLoom:
1. **Self-improving** - Gets better with every query
2. **Explainable** - Full provenance in scratchpad
3. **Adaptive** - Learns what matters from usage
4. **Quality-aware** - Automatically refines low-confidence results
5. **Continuous** - Background learning from all interactions

**It's a cognitive architecture with working memory, continuous learning, and recursive refinement.**

---

**Status**: Vision documented, ready to prototype
**Next**: Implement Phase 1 (Scratchpad integration)
**Timeline**: 15-21 hours for full implementation

---

_"The pieces are all here. We just need to connect them." - October 29, 2025_
