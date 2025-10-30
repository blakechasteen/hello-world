# Semantic Cache Hybrid Design - Verification & Architecture

**Date:** October 28, 2025
**Purpose:** Three-tier caching system for 244D semantic projections
**Status:** Design Complete, Awaiting Implementation

---

## Problem Statement

### The Pipeline Bottleneck

Current semantic projection pipeline:

```python
# Full pipeline (1.53ms total)
text = "hero's journey"

# Step 1: Text → 384D embedding (1.26ms - 82% of time)
vec_384d = embedder.encode([text])  # Neural network forward pass

# Step 2: 384D → 244D semantic scores (0.27ms - 18% of time)
semantic_scores = {}
for dim in EXTENDED_244_DIMENSIONS:  # 244 iterations
    semantic_scores[dim.name] = np.dot(vec_384d, dim.axis)  # 187,392 FLOPs

# Step 3: Cache lookup (0.00008ms - baseline)
cached = cache[text]  # Just dictionary lookup
```

### Performance Measurements

**Actual benchmarks** (from `measure_projection_cost.py`):

```
Cache lookup:     0.00008ms  (instant)
Projection only:  0.27ms     (3,418× slower than cache)
Embedding only:   1.26ms     (15,716× slower than cache)
Full pipeline:    1.53ms     (19,134× slower than cache)

Cost breakdown:
  Embedding:  82.1% of total time  ← Neural network bottleneck
  Projection: 17.9% of total time  ← 187,392 floating-point ops
```

### The Opportunity

**19,134× speedup** for cached queries by skipping BOTH expensive operations.

---

## The Hybrid Design

### Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TIER 1: HOT CACHE                      │
│  Pre-loaded high-value patterns (1,000 entries)             │
│  Never evicted, loaded at startup                           │
│  Example: "hero", "journey", "hero's journey"               │
│  Lookup: O(1) dict access                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ miss
┌─────────────────────────────────────────────────────────────┐
│                     TIER 2: WARM CACHE                      │
│  LRU cache for recently accessed (5,000 entries)            │
│  Adapts to usage patterns                                   │
│  Auto-evicts least recently used                            │
│  Lookup: O(1) dict access + LRU update                      │
└─────────────────────────────────────────────────────────────┘
                            ↓ miss
┌─────────────────────────────────────────────────────────────┐
│                     TIER 3: COLD PATH                       │
│  Full computation: embedding + projection                   │
│  Fully compositional (handles ANY text)                     │
│  Result automatically added to warm cache                   │
│  Cost: 1.53ms (but only on first access)                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: Compositionality Preserved

**The hybrid design does NOT lose compositionality** because:

1. ✓ **Cache hits**: Instant (hot/warm tier)
2. ✓ **Cache misses**: Full computation via embeddings (cold path)
3. ✓ **Novel phrases**: "quantum trickster rebellion" → cold path → works!
4. ✓ **Learning**: Frequently accessed patterns migrate to warm tier
5. ✓ **No limitations**: Can handle ANY text, not just cached words

**This is compositional** = Can understand combinations never seen before.

---

## Implementation Design

### Complete Implementation

```python
from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np

class AdaptiveSemanticCache:
    """
    Three-tier cache for 244D semantic projections.

    Performance profile:
    - Hot tier hit:  0.00008ms (19,134× faster than full pipeline)
    - Warm tier hit: 0.00008ms (same as hot, just needs LRU update)
    - Cold path:     1.53ms (embedding + projection + cache insertion)

    Memory usage:
    - Hot: 1,000 entries × 244 floats × 4 bytes = ~1 MB
    - Warm: 5,000 entries × 244 floats × 4 bytes = ~5 MB
    - Total: ~6 MB (tiny compared to model size)
    """

    def __init__(self,
                 semantic_spectrum,  # SemanticSpectrum with 244 dimensions
                 embedder,           # MatryoshkaEmbeddings
                 hot_size: int = 1000,
                 warm_size: int = 5000):
        self.spectrum = semantic_spectrum
        self.emb = embedder

        # Tier 1: Hot cache (never evicted)
        self.hot: Dict[str, Dict[str, float]] = {}
        self.preload_hot_tier()

        # Tier 2: Warm cache (LRU)
        self.warm: OrderedDict[str, Dict[str, float]] = OrderedDict()
        self.warm_size = warm_size

        # Statistics
        self.hits = {"hot": 0, "warm": 0, "cold": 0}

    def preload_hot_tier(self):
        """
        Pre-compute 244D scores for high-value patterns.

        Patterns chosen based on:
        - Narrative analysis (hero, journey, shadow, etc.)
        - Common phrases in mythological texts
        - Query patterns from user logs
        """
        hot_patterns = [
            # Core narrative words (single tokens)
            "hero", "journey", "quest", "transformation", "sacrifice",
            "wisdom", "courage", "death", "rebirth", "love", "hate",
            "mentor", "trickster", "shadow", "threshold", "guardian",

            # Common narrative phrases (multi-word!)
            "hero's journey", "call to adventure", "dark night of the soul",
            "shadow self", "divine intervention", "tragic flaw",
            "moment of truth", "return with elixir", "crossing the threshold",

            # Archetypal patterns
            "mentor figure", "threshold guardian", "shapeshifter",
            "trickster energy", "mother archetype", "father wound",
            "inner child", "wise old man", "great mother",

            # Philosophical concepts
            "existential dread", "authentic being", "bad faith",
            "death awareness", "freedom and responsibility",

            # Emotional depth
            "unconditional love", "righteous anger", "profound grief",
            "existential loneliness", "transcendent joy",

            # Character traits
            "noble sacrifice", "hubris and fall", "redemptive arc",
            "moral ambiguity", "tragic inevitability",

            # Extend to 1,000 based on:
            # - Most frequent words in corpus
            # - High semantic importance (via TF-IDF)
            # - User query logs
        ]

        print(f"Preloading {len(hot_patterns)} patterns into hot tier...")
        for pattern in hot_patterns:
            vec = self.emb.encode([pattern])[0]
            self.hot[pattern] = self.spectrum.project_vector(vec)
        print(f"  Hot tier loaded: {len(self.hot)} patterns")

    def get_scores(self, text: str) -> Dict[str, float]:
        """
        Get 244D semantic scores with three-tier lookup.

        Returns:
            Dict mapping dimension name → score
            Example: {"Heroism": 0.92, "Courage": 0.87, ...}
        """
        # Tier 1: Hot cache
        if text in self.hot:
            self.hits["hot"] += 1
            return self.hot[text]

        # Tier 2: Warm cache (LRU)
        if text in self.warm:
            self.hits["warm"] += 1
            self.warm.move_to_end(text)  # Mark as recently used
            return self.warm[text]

        # Tier 3: Cold path (full computation)
        self.hits["cold"] += 1

        # Full pipeline: embedding + projection
        vec = self.emb.encode([text])[0]  # 1.26ms - neural network
        scores = self.spectrum.project_vector(vec)  # 0.27ms - dot products

        # Add to warm cache
        self.warm[text] = scores

        # Evict LRU if over capacity
        if len(self.warm) > self.warm_size:
            self.warm.popitem(last=False)  # Remove least recently used

        return scores

    def get_batch_scores(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Batch-optimized version for multiple texts.

        Optimization: Separates cache hits from misses,
        computes all misses in single batch (faster embedding).
        """
        results = [None] * len(texts)
        misses = []
        miss_indices = []

        # Phase 1: Check caches
        for i, text in enumerate(texts):
            if text in self.hot:
                results[i] = self.hot[text]
                self.hits["hot"] += 1
            elif text in self.warm:
                results[i] = self.warm[text]
                self.hits["warm"] += 1
                self.warm.move_to_end(text)
            else:
                misses.append(text)
                miss_indices.append(i)

        # Phase 2: Batch compute misses (MUCH faster than individual)
        if misses:
            self.hits["cold"] += len(misses)

            # Batch embedding (amortizes model overhead)
            vecs = self.emb.encode(misses)  # Single forward pass

            for idx, text, vec in zip(miss_indices, misses, vecs):
                scores = self.spectrum.project_vector(vec)
                results[idx] = scores

                # Add to warm cache
                self.warm[text] = scores
                if len(self.warm) > self.warm_size:
                    self.warm.popitem(last=False)

        return results

    def print_stats(self):
        """Display cache performance metrics."""
        total = sum(self.hits.values())
        if total == 0:
            print("No queries yet")
            return

        print("\n=== Cache Performance ===")
        print(f"Hot tier:  {self.hits['hot']:>6} hits ({100*self.hits['hot']/total:>5.1f}%)")
        print(f"Warm tier: {self.hits['warm']:>6} hits ({100*self.hits['warm']/total:>5.1f}%)")
        print(f"Cold path: {self.hits['cold']:>6} hits ({100*self.hits['cold']/total:>5.1f}%)")
        print(f"Total:     {total:>6} queries")

        cache_hit_rate = (self.hits['hot'] + self.hits['warm']) / total
        print(f"\nOverall cache hit rate: {cache_hit_rate:.1%}")
        print(f"Cache sizes: hot={len(self.hot)}, warm={len(self.warm)}")

        # Calculate speedup
        avg_time_without_cache = 1.53  # ms
        avg_time_with_cache = (
            (self.hits['hot'] + self.hits['warm']) * 0.00008 +
            self.hits['cold'] * 1.53
        ) / total
        speedup = avg_time_without_cache / avg_time_with_cache

        print(f"\nEstimated speedup: {speedup:.1f}× faster than no cache")
```

---

## Verification Against Existing Code

### ✅ Compatible with Current Architecture

**1. MatryoshkaEmbeddings (HoloLoom/embedding/spectral.py:84-289)**

Current implementation already has embedding cache (lines 118-120):
```python
# Embedding cache (text -> embedding)
from HoloLoom.performance.cache import QueryCache
self._embedding_cache = QueryCache(max_size=500, ttl_seconds=3600)
```

**Status:** ✅ Already caches at 384D level. Our design adds caching at 244D level.

**2. SemanticSpectrum (HoloLoom/semantic_calculus/dimensions.py:1332-1470)**

Current implementation (lines 1361-1371):
```python
def project_vector(self, vector: np.ndarray) -> Dict[str, float]:
    """Project a single vector onto all dimensions."""
    if not self._axes_learned:
        raise ValueError("Axes not learned yet. Call learn_axes() first.")

    return {dim.name: dim.project(vector) for dim in self.dimensions}
```

**Status:** ✅ Pure function, perfect for caching. No side effects.

**3. QueryCache (HoloLoom/performance/cache.py:26-100)**

Current LRU cache implementation:
```python
class QueryCache:
    """Simple LRU cache with TTL for query results."""
    def __init__(self, max_size: int = 50, ttl_seconds: float = 300):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
```

**Status:** ✅ Can be used as-is for warm tier, or extend for three-tier design.

---

## Design Validation

### Question 1: Does this lose compositionality?

**Answer: NO**

Evidence:
1. ✅ Cold path uses full embedding model (handles ANY text)
2. ✅ Cache is just an optimization layer (transparent to caller)
3. ✅ Novel combinations: "cybernetic trickster hero" → cold path → works
4. ✅ Phrases work: "hero's journey into darkness" → compositional

**Compositionality = Can understand never-seen-before combinations**

The cache doesn't break this because:
- Cache miss → Falls back to full computation
- Full computation → Uses sentence-transformers (compositional by design)
- Result → Same as if cache didn't exist

### Question 2: What about memory usage?

**Memory Breakdown:**

```
Hot tier:  1,000 entries × 244 dimensions × 4 bytes/float = ~1 MB
Warm tier: 5,000 entries × 244 dimensions × 4 bytes/float = ~5 MB
Total cache: ~6 MB

For comparison:
- Sentence-transformers model: ~90 MB (all-MiniLM-L6-v2)
- Python interpreter: ~50 MB base memory
- Cache overhead: 6% of model size
```

**Verdict:** Memory cost is negligible.

### Question 3: What's the expected hit rate?

**Expected Performance:**

Based on typical query patterns:
- **Hot tier:** 40-60% hit rate (common narrative words/phrases)
- **Warm tier:** 20-30% hit rate (recently accessed, user-specific patterns)
- **Cold path:** 10-40% miss rate (novel queries)

**Overall cache hit rate: 60-90%**

**Speedup calculation:**
```python
# Assume 70% cache hit rate
avg_time = 0.7 * 0.00008ms + 0.3 * 1.53ms
         = 0.000056ms + 0.459ms
         = 0.459ms

# Without cache: 1.53ms
# With cache: 0.459ms
# Speedup: 3.3× faster

# At 90% hit rate:
avg_time = 0.9 * 0.00008ms + 0.1 * 1.53ms = 0.153ms
speedup = 1.53 / 0.153 = 10× faster
```

### Question 4: Can we cache phrases?

**Answer: YES!**

Evidence from design:
```python
hot_patterns = [
    # Single words
    "hero", "journey", "shadow",

    # Multi-word phrases (fully supported!)
    "hero's journey",
    "call to adventure",
    "dark night of the soul",
    "crossing the threshold"
]
```

**How it works:**
1. Sentence-transformers creates single embedding for entire phrase
2. We cache the final 244D projection
3. Lookup: `cache["hero's journey"]` → instant retrieval

**Limitation:** We cache exact string matches.
- ✅ "hero's journey" (cached) → instant
- ❌ "journey of the hero" (different string) → cold path

But this is fine! The cold path handles variations compositionally.

---

## Implementation Checklist

### Phase 1: Basic Three-Tier Cache (2-3 hours)

- [ ] Create `AdaptiveSemanticCache` class
- [ ] Implement hot tier preloading
- [ ] Implement warm tier (LRU)
- [ ] Implement cold path fallback
- [ ] Add statistics tracking

### Phase 2: Hot Tier Optimization (2-4 hours)

- [ ] Analyze corpus for high-frequency patterns
- [ ] Extract 1,000 most important words/phrases
- [ ] Pre-compute 244D projections at startup
- [ ] Validate hot tier coverage

### Phase 3: Integration (1-2 hours)

- [ ] Integrate with `SemanticSpectrum`
- [ ] Add to `MatryoshkaEmbeddings` pipeline
- [ ] Update `WeavingOrchestrator` to use cache
- [ ] Add configuration options

### Phase 4: Benchmarking (1 hour)

- [ ] Run performance tests
- [ ] Measure hit rates (hot/warm/cold)
- [ ] Calculate actual speedup
- [ ] Validate memory usage

### Phase 5: Tuning (1-2 hours)

- [ ] Adjust cache sizes based on measurements
- [ ] Refine hot tier patterns
- [ ] Optimize batch processing
- [ ] Document performance characteristics

**Total estimated time: 7-12 hours**

---

## Comparison to Standard Embeddings

### Standard Embedding Cache (What EXISTS)

```python
# Current: Cache at 384D embedding level
class MatryoshkaEmbeddings:
    def __init__(self):
        self._embedding_cache = QueryCache(max_size=500)

    def encode_base(self, texts):
        cached = self._embedding_cache.get(text)
        if cached:
            return cached  # ⚡ Skip neural network (1.26ms saved)

        vec = self._model.encode(texts)  # 1.26ms
        self._embedding_cache.put(text, vec)
        return vec
```

**What it saves:** Neural network forward pass (1.26ms, 82% of cost)
**What it doesn't save:** Projection to 244D (0.27ms, 18% of cost)

### Semantic Cache (What We're ADDING)

```python
# Proposed: Cache at 244D semantic level
class AdaptiveSemanticCache:
    def get_scores(self, text):
        if text in self.cache:
            return self.cache[text]  # ⚡ Skip BOTH operations (1.53ms saved)

        vec = self.emb.encode([text])[0]  # 1.26ms
        scores = self.spectrum.project_vector(vec)  # 0.27ms
        self.cache[text] = scores
        return scores
```

**What it saves:** Neural network + projection (1.53ms, 100% of cost)
**Advantage:** Higher-level cache, more semantic

### Why Both?

**Layered caching strategy:**
```
Query: "hero's journey"
    ↓
Semantic cache (244D) → HIT ⚡ (0.00008ms) RETURN
    ↓ (if miss)
Embedding cache (384D) → HIT ⚡ (0.27ms for projection only)
    ↓ (if miss)
Neural network (compute) → (1.53ms full pipeline)
```

**This is optimal!** We cache at multiple levels:
1. **Semantic cache**: Skip everything (best case)
2. **Embedding cache**: Skip neural network (good case)
3. **Neural network**: Full computation (worst case)

---

## Theoretical Foundations

### Why This Design is Sound

**1. Locality of Reference**
- Users repeat queries
- Narrative analysis uses common vocabulary
- Zipf's law: Few patterns account for most usage

**2. Compositionality at the Right Level**
- Embeddings provide compositionality
- Cache provides speed
- Cold path ensures correctness

**3. Graceful Degradation**
- Cache miss → Full computation
- No correctness loss
- Only performance impact

**4. Adaptive Learning**
- Warm tier learns user patterns
- Hot tier covers common cases
- System improves with use

---

## Potential Issues & Solutions

### Issue 1: Cache Invalidation

**Problem:** What if semantic axes change?

**Solution:**
```python
def learn_axes(self, embed_fn):
    """Re-learn semantic axes and invalidate cache."""
    for dim in self.dimensions:
        dim.learn_axis(embed_fn)

    # Invalidate semantic cache (axes changed!)
    if hasattr(self, '_semantic_cache'):
        self._semantic_cache.clear()
```

### Issue 2: Memory Pressure

**Problem:** 6MB cache might be too large for mobile devices.

**Solution:** Configurable cache sizes
```python
# Server: Large cache
cache = AdaptiveSemanticCache(hot_size=1000, warm_size=5000)  # 6 MB

# Mobile: Tiny cache
cache = AdaptiveSemanticCache(hot_size=100, warm_size=500)  # 0.6 MB
```

### Issue 3: Cold Start

**Problem:** First query always misses cache.

**Solution:** Pre-warm cache during initialization
```python
# Load from previous session
cache.load_from_disk("~/.hololoom/semantic_cache.json")
```

---

## Conclusion

### Verdict: ✅ DESIGN IS SOUND

**Strengths:**
1. ✅ Preserves compositionality (cold path handles any text)
2. ✅ Achieves massive speedup (up to 19,134× for cache hits)
3. ✅ Memory efficient (6MB cache, 6% of model size)
4. ✅ Adapts to usage patterns (LRU warm tier)
5. ✅ Compatible with existing code (minimal changes needed)
6. ✅ Handles phrases naturally (multi-word caching works)

**Weaknesses:**
1. ⚠️ Requires startup time to pre-load hot tier (~30s)
2. ⚠️ Cache invalidation needed if axes change
3. ⚠️ String-based matching (doesn't handle paraphrases)

**Overall Assessment:**

This hybrid design is a **textbook example** of multi-level caching done right:
- Fast path for common cases (hot tier)
- Adaptive path for user patterns (warm tier)
- Fallback path for correctness (cold path)
- Compositional guarantees preserved
- Performance gains are massive (3-10× typical, up to 19,000× best case)

**Recommendation:** IMPLEMENT THIS

**Estimated ROI:**
- Development time: 7-12 hours
- Performance gain: 3-10× average speedup
- User experience: Sub-millisecond responses for cached queries
- Cost: Negligible (6MB memory, no accuracy loss)

**This is a no-brainer optimization.** ✅

---

## Next Steps

1. **Implement `AdaptiveSemanticCache`** (3 hours)
2. **Pre-compute hot tier patterns** (2 hours)
3. **Integrate with `SemanticSpectrum`** (1 hour)
4. **Benchmark and tune** (2 hours)
5. **Document performance characteristics** (1 hour)

**Total: ~9 hours to production-ready implementation**

---

**Files to Create:**
- `HoloLoom/performance/semantic_cache.py` (implementation)
- `HoloLoom/semantic_calculus/cache_integration.py` (integration)
- `measure_semantic_cache_performance.py` (benchmarks)
- `SEMANTIC_CACHE_PERFORMANCE.md` (results)

**Files to Modify:**
- `HoloLoom/semantic_calculus/dimensions.py` (add cache option)
- `HoloLoom/embedding/spectral.py` (integrate semantic cache)
- `HoloLoom/weaving_orchestrator.py` (use semantic cache)