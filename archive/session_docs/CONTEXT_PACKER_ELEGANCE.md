# Context Packer Elegance - Beta Wave Integration

**Date**: October 30, 2025
**Vision**: Replace ad-hoc importance scoring with beta wave activation levels

---

## The Problem: Knapsack Optimization Complexity

Current context packer (`HoloLoom/awareness/context_packer.py`) solves a **multi-dimensional knapsack problem**:

- **506 lines** of greedy optimization code
- **3 separate passes** (critical → high → medium/low)
- **Ad-hoc importance scoring** (boost by 1.2x, 1.15x, 1.1x based on heuristics)
- **Manual compression strategies** (full → detailed → summary → minimal)
- **Rough token estimation** (len/4)

### Complexity Sources

1. **Importance Calculation** (lines 366-399):
   ```python
   # Boost awareness elements when uncertain
   if element.source == "awareness" and conf.uncertainty_level > 0.7:
       element.importance = min(1.0, element.importance * 1.2)

   # Boost pattern elements when familiar
   if patterns.seen_count > 10:
       element.importance = min(1.0, element.importance * 1.1)

   # Boost memory elements from same domain
   if patterns.domain.lower() in content_lower:
       element.importance = min(1.0, element.importance * 1.15)
   ```

   **Problem**: Magic numbers (1.2, 1.15, 1.1) and brittle heuristics.

2. **3-Pass Greedy Packing** (lines 401-478):
   ```python
   # Pass 1: Critical elements (always full)
   for element in elements:
       if element.importance >= ContextImportance.CRITICAL.value:
           # ... pack ...

   # Pass 2: High-importance (compress if needed)
   for element in elements:
       if ContextImportance.HIGH.value <= element.importance < ContextImportance.CRITICAL.value:
           # Try full → detailed → summary

   # Pass 3: Medium/low (summary only)
   for element in elements:
       if element.importance < ContextImportance.HIGH.value:
           # Only summary
   ```

   **Problem**: Arbitrary thresholds (1.0, 0.8, 0.5, 0.2) and redundant loops.

3. **Manual Compression** (lines 48-64):
   ```python
   def compress(self, level: CompressionLevel) -> str:
       if level == CompressionLevel.FULL:
           return self.content
       elif level == CompressionLevel.DETAILED and self.detailed:
           return self.detailed
       elif level == CompressionLevel.SUMMARY and self.summary:
           return self.summary
       # ...
   ```

   **Problem**: Requires manually writing summary/detailed versions for every element.

---

## The Elegant Solution: Beta Wave Activation Levels

**Key Insight**: The multi-wave memory system we just built **already solves importance scoring**!

### How Beta Waves Solve Importance

From `HoloLoom/memory/spring_dynamics_engine.py`:

```python
def retrieve_memories(query_embedding, top_k=10) -> BetaWaveRecallResult:
    """
    Beta wave activation spreading returns:
    - recalled_memories: [(node_id, activation), ...] sorted by activation
    - all_activations: {node_id: activation_level} complete map
    - creative_insights: Distant but activated (cross-domain)
    - seed_nodes: Direct semantic matches
    """
```

**Activation level = natural importance metric!**

- **High activation** = strong relevance (high importance)
- **Medium activation** = indirect association (medium importance)
- **Low activation** = distant connection (low importance)
- **No activation** = irrelevant (exclude)

### Beta Wave Provides

1. **Unified Importance Metric**:
   - No more ad-hoc boosts (1.2x, 1.15x, 1.1x)
   - Activation level IS importance (0.0-1.0 naturally)
   - Physics-based, not heuristic-based

2. **Natural Ranking**:
   - Beta wave spreading already ranks by relevance
   - No need for manual importance scoring
   - Handles cross-domain associations via creative insights

3. **Recency/Freshness Built-In**:
   - Spring constant k encodes how recently accessed
   - Strong springs (high k) → better activation conductivity
   - Weak springs (low k) → faded memories, lower activation

4. **Compression Threshold**:
   - High activation (>0.7) → include full content
   - Medium activation (0.3-0.7) → use summary
   - Low activation (<0.3) → exclude
   - Single threshold, not 3 passes!

---

## Proposed Elegant Architecture

### New Design: Activation-Weighted Context Packer

```python
class BetaWaveContextPacker:
    """
    Elegant context packing using beta wave activation levels.

    Replaces:
    - Ad-hoc importance scoring → activation levels
    - 3-pass greedy packing → single activation-sorted pass
    - Manual compression → activation-threshold compression
    - Magic numbers → physics-based activation spreading
    """

    def __init__(
        self,
        spring_engine: SpringDynamicsEngine,
        token_budget: TokenBudget,
        activation_threshold: float = 0.3
    ):
        self.engine = spring_engine
        self.budget = token_budget
        self.activation_threshold = activation_threshold

    async def pack_context(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        awareness_context
    ) -> PackedContext:
        """
        Pack context using beta wave activation spreading.

        Elegance:
        1. Single retrieval: Beta wave spreading ranks everything
        2. Single pass: Pack by activation level until budget exhausted
        3. Natural compression: Activation threshold determines compression
        4. No heuristics: Physics handles importance
        """

        # 1. Beta wave retrieval (single unified ranking)
        result = self.engine.retrieve_memories(
            query_embedding=query_embedding,
            top_k=50,  # Get more than needed, filter by budget
            activation_threshold=self.activation_threshold
        )

        # 2. Convert to context elements with activation as importance
        elements = []

        # Query (always critical)
        elements.append(ContextElement(
            content=query_text,
            importance=1.0,  # Always critical
            token_count=self._estimate_tokens(query_text),
            source="query"
        ))

        # Awareness signals (activation-boosted)
        elements.extend(self._create_awareness_elements(
            awareness_context,
            base_activation=0.8  # High but not critical
        ))

        # Memories (use activation from beta wave spreading)
        for node_id, activation in result.recalled_memories:
            node = self.engine.nodes[node_id]
            elements.append(ContextElement(
                content=node.content,
                importance=activation,  # Direct from beta waves!
                token_count=self._estimate_tokens(node.content),
                source="memory",
                metadata={"node_id": node_id, "spring_k": node.spring_constant}
            ))

        # Creative insights (marked as cross-domain)
        for node_id, activation, insight_type in result.creative_insights:
            node = self.engine.nodes[node_id]
            elements.append(ContextElement(
                content=node.content,
                importance=activation * 0.9,  # Slightly lower (not direct match)
                token_count=self._estimate_tokens(node.content),
                source="creative_insight",
                metadata={
                    "node_id": node_id,
                    "insight_type": insight_type,
                    "cross_domain": True
                }
            ))

        # 3. Single-pass packing (already sorted by activation)
        elements.sort(key=lambda e: e.importance, reverse=True)

        packed = []
        remaining_budget = self.budget.available_for_context

        for element in elements:
            # Activation-based compression
            if element.importance >= 0.7:
                # High activation → full content
                if element.token_count <= remaining_budget:
                    packed.append(element)
                    remaining_budget -= element.token_count

            elif element.importance >= 0.3:
                # Medium activation → compressed (50% reduction)
                compressed_tokens = element.token_count // 2
                if compressed_tokens <= remaining_budget:
                    element.content = self._compress_content(element.content, ratio=0.5)
                    element.token_count = compressed_tokens
                    packed.append(element)
                    remaining_budget -= compressed_tokens

            # Low activation (<0.3) → exclude automatically

        # 4. Format for LLM
        return self._format_packed_context(packed, result)

    def _compress_content(self, content: str, ratio: float) -> str:
        """
        Simple compression: Take first N% of content.

        Future: Use extractive summarization or LLM-based compression.
        """
        target_length = int(len(content) * ratio)
        return content[:target_length] + "..."

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4
```

---

## Key Advantages

### 1. Unified Importance Metric

**Before** (506 lines, ad-hoc):
```python
# Manual importance assignment
if element.source == "awareness":
    importance = 0.8
elif element.source == "memory":
    importance = 0.8 - (position * 0.05)
elif conf.uncertainty_level > 0.7:
    importance *= 1.2
```

**After** (physics-based):
```python
# Direct from beta wave spreading
importance = activation_level  # Already computed by spring dynamics
```

### 2. Single-Pass Packing

**Before** (3 separate loops):
```python
# Pass 1: Critical (always full)
for element in elements:
    if element.importance >= 1.0: ...

# Pass 2: High (compress if needed)
for element in elements:
    if 0.8 <= element.importance < 1.0: ...

# Pass 3: Medium/low (summary only)
for element in elements:
    if element.importance < 0.8: ...
```

**After** (single loop):
```python
for element in sorted_by_activation:
    compress_if_needed(element.activation)
    if fits_budget: pack()
```

### 3. Natural Compression Threshold

**Before** (manual compression levels):
```python
class CompressionLevel(Enum):
    FULL = "full"
    DETAILED = "detailed"
    SUMMARY = "summary"
    MINIMAL = "minimal"

# Requires manually writing summary/detailed versions
```

**After** (activation-threshold):
```python
if activation >= 0.7: use_full_content
elif activation >= 0.3: compress_to_50_percent
else: exclude
```

### 4. Cross-Domain Insights

**Before** (not handled):
```python
# Context packer doesn't know about creative associations
```

**After** (built-in):
```python
# Beta wave creative_insights automatically included
for node_id, activation, insight_type in result.creative_insights:
    # Mark as cross-domain, slightly lower priority
    pack_with_metadata(insight_type="bridge_node")
```

---

## Integration with Multi-Wave System

The context packer becomes **part of the beta wave cycle**:

```
Query arrives
  ↓
Beta Wave Activation Spreading (SpringDynamicsEngine)
  ↓
Activation Map: {node_id: activation_level}
  ↓
BetaWaveContextPacker (uses activation as importance)
  ↓
PackedContext (optimal, physics-based selection)
  ↓
LLM Generation
```

### Full Pipeline

```python
# 1. Create multi-wave engine with memories
engine = MultiWaveMemoryEngine(config)
await engine.start()

# 2. Ingest data from spinners
async for shard in spinner.spin_stream():
    engine.encode_new_memory(shard.id, shard.content, embedding)

# 3. Query triggers beta wave retrieval
query_embedding = embed_query(query_text)
result = engine.retrieve_memories(query_embedding)

# 4. Context packer uses activation levels
packer = BetaWaveContextPacker(engine, budget)
packed = await packer.pack_context(query_embedding, query_text, awareness_ctx)

# 5. LLM generates response
response = await llm.generate(packed.format_for_llm())
```

---

## Implementation Plan

### Phase 1: Core Integration (2-3 hours)

1. **Create BetaWaveContextPacker** (new file)
   - Use activation levels from SpringDynamicsEngine
   - Single-pass packing with activation threshold
   - Simple compression (first N%)

2. **Wire to WeavingOrchestrator**
   - Replace SmartContextPacker with BetaWaveContextPacker
   - Pass SpringDynamicsEngine instance
   - Use beta wave retrieval result

3. **Test with demo_streaming_memory.py**
   - Query → beta wave retrieval → context packing
   - Verify activation levels used as importance
   - Compare with old packer

### Phase 2: Advanced Features (3-4 hours)

4. **Intelligent Compression**
   - Use extractive summarization (TextRank)
   - LLM-based summarization for high-value content
   - Preserve semantic density

5. **Temporal Weighting**
   - Use spring constant k for recency boost
   - Recent memories (high k) → slight activation boost
   - Old memories (low k) → slight activation penalty

6. **Adaptive Thresholds**
   - High confidence queries → stricter threshold (0.5)
   - Low confidence queries → looser threshold (0.2)
   - Use awareness signals to adjust

### Phase 3: Optimization (2-3 hours)

7. **Token Estimation**
   - Replace len/4 with actual tokenizer
   - Cache token counts in MemoryNode
   - Fast lookup during packing

8. **Budget Optimization**
   - Dynamic programming for exact knapsack solution
   - Fallback to greedy if DP too slow
   - Benchmark both approaches

9. **Visualization**
   - Show activation map overlaid on context
   - Highlight compressed vs full elements
   - Display creative insights separately

---

## Expected Outcomes

### Code Reduction

**Before**: 506 lines (context_packer.py)
**After**: ~250 lines (beta_wave_context_packer.py)

**Reduction**: ~50% fewer lines, much simpler logic

### Performance

**Before**: 3 passes over elements = O(3n)
**After**: Single pass = O(n)

**Speedup**: ~3x faster packing

### Quality

**Before**: Ad-hoc importance scoring, magic numbers
**After**: Physics-based activation spreading, no heuristics

**Result**: More consistent, explainable importance

### Integration

**Before**: Context packer isolated from memory system
**After**: Context packer IS part of beta wave cycle

**Result**: Unified architecture, shared activation metric

---

## The Elegant Core Principle

**"Activation IS Importance"**

Instead of:
- Calculating importance separately (heuristics)
- Scoring elements independently (no context)
- Using magic numbers (brittle)

We use:
- **Beta wave activation spreading** (physics)
- **Graph-based relevance** (context-aware)
- **Spring dynamics** (adaptive, self-organizing)

The context packer becomes **trivial**: sort by activation, pack until budget exhausted, compress based on activation threshold.

---

## Why This Is Elegant

1. **Single Unified Metric**: Activation level replaces all ad-hoc importance scoring
2. **Physics-Based**: Spring dynamics handles complexity, not hand-crafted heuristics
3. **Self-Organizing**: Spring constants encode recency, activation encodes relevance
4. **Integrated**: Context packer is now part of the multi-wave memory system
5. **Simple**: ~50% code reduction, single-pass algorithm
6. **Explainable**: "This memory has activation 0.85" is clear, "boost by 1.15x" is not

---

**Status**: Ready to implement - connect context packer to beta wave retrieval system.

**Next**: Implement `BetaWaveContextPacker` and wire to `WeavingOrchestrator`.
