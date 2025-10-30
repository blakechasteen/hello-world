# Phase 5: Universal Grammar + Compositional Semantic Cache

**Date:** October 28, 2025
**Status:** Design Document
**Priority:** MASSIVE UPGRADE - Game Changer
**Timeline:** 2-3 weeks (comprehensive implementation)

---

## Executive Summary

This phase combines **three revolutionary concepts**:

1. **Universal Grammar (X-bar Theory)** - Principled phrase chunking
2. **Compositional Semantics (Merge)** - Build meaning hierarchically
3. **Multi-Level Semantic Cache** - Cache at EVERY level of composition

**The insight:** If we cache phrase structures AND their compositional embeddings, we get **MULTIPLICATIVE speedups**!

### Impact Projection

**Current semantic cache:** 3-10× speedup (cache final 244D projections)

**UG compositional cache:** 50-100× speedup potential!

Why? Because we cache:
- Parse structures (X-bar trees) - **expensive spaCy parsing**
- Compositional embeddings (Merge results) - **expensive composition**
- Multi-level chunks (X, X', XP) - **reusable building blocks**

---

## The Problem

### Current Pipeline (Linear + Flat)

```
Text → Embedding → 244D Projection → Cache
"the big red ball"
     ↓ (encode full phrase as single unit)
     384D embedding
     ↓ (project to 244D)
     244D semantic vector
     ↓ (cache this final result)
```

**Issues:**
1. ❌ No compositionality (treats "the big red ball" as atomic)
2. ❌ Can't reuse parts ("red ball" seen before? Doesn't matter!)
3. ❌ No syntactic understanding (it's a noun phrase? Unknown!)
4. ❌ Single-level caching (only cache final 244D)

### After UG + Compositional Cache (Hierarchical + Compositional)

```
Text → X-bar Parse → Merge Composition → Multi-Level Cache
"the big red ball"
     ↓
[X-bar Structure]
  NP
   ├─ Spec: Det "the"
   └─ N'
       ├─ Adj "big"
       └─ N'
           ├─ Adj "red"
           └─ N "ball"
     ↓
[Compositional Embedding via Merge]
  Level 1: Merge("red", "ball") → "red ball" embedding  ← CACHE THIS!
  Level 2: Merge("big", "red ball") → "big red ball" ← CACHE THIS!
  Level 3: Merge("the", "big red ball") → NP embedding ← CACHE THIS!
     ↓
[Multi-Level Semantic Cache]
  - X-bar structure: CACHED (don't re-parse!)
  - "red ball": CACHED (reuse if seen again!)
  - "big red ball": CACHED
  - Final NP: CACHED
```

**Benefits:**
1. ✅ **Compositional reuse** - "red ball" cached, reused in "the red ball", "a red ball"
2. ✅ **Syntactic awareness** - Know it's an NP (can filter by phrase type!)
3. ✅ **Multi-level caching** - Cache at X, X', XP levels
4. ✅ **Parse structure caching** - Don't re-parse common phrases!

---

## Architecture

### Three-Tier Compositional Cache

```
┌─────────────────────────────────────────────────────────────┐
│                  COMPOSITIONAL SEMANTIC CACHE                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ TIER 1:      │    │ TIER 2:      │    │ TIER 3:      │
│ Parse Cache  │    │ Merge Cache  │    │ Semantic     │
│              │    │              │    │ Cache        │
│ X-bar        │    │ Compositional│    │ 244D         │
│ structures   │    │ embeddings   │    │ projections  │
└──────────────┘    └──────────────┘    └──────────────┘
     (fast!)            (medium)           (existing)
```

### Tier 1: Parse Structure Cache

**What:** Cache X-bar parse trees

**Key:** Text → X-bar tree (JSON/pickle)

**Why:** spaCy parsing is expensive (10-50ms per sentence)

**Example:**
```python
parse_cache = {
    "the big red ball": {
        "structure": "NP",
        "head": "ball",
        "category": "N",
        "xbar_tree": {
            "label": "NP",
            "spec": {"label": "D", "head": "the"},
            "comp": {
                "label": "N'",
                "adjuncts": [
                    {"label": "AP", "head": "big"},
                    {"label": "AP", "head": "red"}
                ],
                "head": "ball"
            }
        }
    }
}
```

**Speedup:** 10-50× (avoid re-parsing)

### Tier 2: Compositional Embedding Cache

**What:** Cache Merge results (compositional embeddings)

**Key:** (head, dependent, merge_type) → composed_embedding

**Why:** Merge operations involve matrix operations

**Example:**
```python
merge_cache = {
    ("red", "ball", "ADJECTIVE_NOUN"): {
        "embedding": np.array([...]),  # 384D or 426D (with linguistics)
        "head": "ball",
        "label": "N'",
        "merge_type": "external"
    },
    ("big", "red ball", "ADJECTIVE_NP"): {
        "embedding": np.array([...]),
        "head": "ball",
        "label": "N'",
        "merge_type": "external"
    },
    ("the", "big red ball", "DET_NP"): {
        "embedding": np.array([...]),
        "head": "ball",
        "label": "NP",
        "merge_type": "external"
    }
}
```

**Speedup:** 5-10× (avoid re-composing)

### Tier 3: Semantic Projection Cache (Existing)

**What:** Cache 244D semantic projections

**Already implemented!** (AdaptiveSemanticCache)

**Speedup:** 3-10× (avoid re-projecting)

### Total Speedup Potential

**Multiplicative speedups:**
```
Parse cache:      10×
Merge cache:       5×
Semantic cache:    3×
─────────────────────
TOTAL:         150× potential speedup!
```

**Realistic (accounting for cache misses):**
- Hot path (everything cached): ~100×
- Warm path (partial cache hits): ~30-50×
- Cold path (cache misses): ~1× (same as current)

---

## Implementation Design

### Compositional Cache Manager

```python
"""
Compositional Semantic Cache
============================
Multi-tier caching for phrase structures, compositional embeddings,
and semantic projections.

Tiers:
1. Parse Cache: X-bar structures (spaCy parse trees)
2. Merge Cache: Compositional embeddings (Merge results)
3. Semantic Cache: 244D projections (existing AdaptiveSemanticCache)
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass
import pickle
import hashlib

from HoloLoom.motif.xbar_chunker import XBarNode, UniversalGrammarChunker
from HoloLoom.warp.merge import MergeOperator, MergedObject
from HoloLoom.performance.semantic_cache import AdaptiveSemanticCache


@dataclass
class CacheStats:
    """Statistics for compositional cache."""
    parse_hits: int = 0
    parse_misses: int = 0
    merge_hits: int = 0
    merge_misses: int = 0
    semantic_hits: int = 0
    semantic_misses: int = 0

    @property
    def parse_hit_rate(self) -> float:
        total = self.parse_hits + self.parse_misses
        return self.parse_hits / total if total > 0 else 0.0

    @property
    def merge_hit_rate(self) -> float:
        total = self.merge_hits + self.merge_misses
        return self.merge_hits / total if total > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        total = self.semantic_hits + self.semantic_misses
        return self.semantic_hits / total if total > 0 else 0.0


class CompositionalCache:
    """
    Three-tier cache for compositional semantics.

    Caches:
    1. Parse structures (X-bar trees)
    2. Compositional embeddings (Merge results)
    3. Semantic projections (244D vectors)

    Usage:
        cache = CompositionalCache(
            ug_chunker=chunker,
            merge_operator=merger,
            semantic_cache=sem_cache
        )

        # Get compositional embedding with caching
        embedding, trace = cache.get_compositional_embedding("the big red ball")
    """

    def __init__(
        self,
        ug_chunker: UniversalGrammarChunker,
        merge_operator: MergeOperator,
        semantic_cache: AdaptiveSemanticCache,
        parse_cache_size: int = 10000,
        merge_cache_size: int = 50000
    ):
        """
        Initialize compositional cache.

        Args:
            ug_chunker: Universal Grammar chunker
            merge_operator: Merge operator for composition
            semantic_cache: Existing semantic cache (Tier 3)
            parse_cache_size: Max parse cache entries
            merge_cache_size: Max merge cache entries
        """
        self.ug_chunker = ug_chunker
        self.merge_operator = merge_operator
        self.semantic_cache = semantic_cache

        # Tier 1: Parse structure cache
        self.parse_cache: Dict[str, XBarNode] = {}
        self.parse_cache_size = parse_cache_size

        # Tier 2: Merge/composition cache
        self.merge_cache: Dict[str, MergedObject] = {}
        self.merge_cache_size = merge_cache_size

        # Statistics
        self.stats = CacheStats()

    def get_compositional_embedding(
        self,
        text: str,
        return_trace: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Get compositional embedding with multi-tier caching.

        Process:
        1. Check parse cache (Tier 1)
        2. Check merge cache (Tier 2)
        3. Check semantic cache (Tier 3)
        4. Compute if needed, cache at all levels

        Args:
            text: Input text
            return_trace: Return cache hit trace

        Returns:
            Tuple of (embedding, trace)
        """
        trace = {"hits": [], "misses": []} if return_trace else None

        # TIER 1: Parse structure cache
        xbar_structure = self._get_or_parse(text, trace)

        # TIER 2: Compositional embedding cache
        composed_embedding = self._get_or_compose(xbar_structure, trace)

        # TIER 3: Semantic projection cache (existing)
        semantic_projection = self._get_or_project(composed_embedding, text, trace)

        return semantic_projection, trace

    def _get_or_parse(self, text: str, trace: Optional[Dict]) -> XBarNode:
        """
        Get X-bar parse from cache or compute.

        TIER 1: Parse structure cache
        """
        # Check cache
        if text in self.parse_cache:
            self.stats.parse_hits += 1
            if trace:
                trace["hits"].append("parse")
            return self.parse_cache[text]

        # Cache miss: parse with UG chunker
        self.stats.parse_misses += 1
        if trace:
            trace["misses"].append("parse")

        xbar_structures = self.ug_chunker.chunk(text)

        # Assume first phrase (could be smarter)
        xbar_structure = xbar_structures[0] if xbar_structures else None

        # Cache it
        if len(self.parse_cache) < self.parse_cache_size:
            self.parse_cache[text] = xbar_structure
        else:
            # LRU eviction (simplified: just clear oldest)
            self.parse_cache.popitem()
            self.parse_cache[text] = xbar_structure

        return xbar_structure

    def _get_or_compose(
        self,
        xbar_node: XBarNode,
        trace: Optional[Dict]
    ) -> np.ndarray:
        """
        Get compositional embedding from cache or compute via Merge.

        TIER 2: Compositional embedding cache

        Recursively composes embeddings bottom-up:
        1. Get leaf embeddings (from base encoder)
        2. Merge bottom-up following X-bar structure
        3. Cache intermediate results
        """
        # Generate cache key from structure
        cache_key = self._xbar_to_cache_key(xbar_node)

        # Check cache
        if cache_key in self.merge_cache:
            self.stats.merge_hits += 1
            if trace:
                trace["hits"].append(f"merge:{cache_key}")
            return self.merge_cache[cache_key].embedding

        # Cache miss: compose via Merge
        self.stats.merge_misses += 1
        if trace:
            trace["misses"].append(f"merge:{cache_key}")

        # Recursive composition
        composed = self._compose_xbar_node(xbar_node, trace)

        # Cache it
        if len(self.merge_cache) < self.merge_cache_size:
            self.merge_cache[cache_key] = composed
        else:
            # LRU eviction
            self.merge_cache.popitem()
            self.merge_cache[cache_key] = composed

        return composed.embedding

    def _compose_xbar_node(
        self,
        node: XBarNode,
        trace: Optional[Dict]
    ) -> MergedObject:
        """
        Recursively compose X-bar node via Merge.

        Bottom-up composition:
        1. If leaf (X), encode head word
        2. If X', merge head + complement (+ adjuncts)
        3. If XP, merge specifier + X'
        """
        # Base case: Lexical head (X)
        if node.level == 0:
            # Encode single word
            embedding = self.merge_operator.embedder.encode([node.head])[0]
            return MergedObject(
                embedding=embedding,
                components=[node.head],
                head=node.head,
                merge_type=MergeType.EXTERNAL,
                label=node.label
            )

        # Recursive case: X' or XP
        # First, compose children
        head_obj = self._compose_xbar_node(node.complement, trace) if node.complement else None
        spec_obj = self._compose_xbar_node(node.specifier, trace) if node.specifier else None

        # Merge adjuncts (right-to-left, attach to X')
        current = head_obj
        for adjunct_node in reversed(node.adjuncts):
            adjunct_obj = self._compose_xbar_node(adjunct_node, trace)
            # Merge adjunct with current
            current = self.merge_operator.external_merge(
                adjunct_obj.embedding,
                current.embedding,
                head=current.head,  # Head is from the modified element
                dependent=adjunct_obj.head,
                label=current.label,
                alpha_is_head=False
            )

        # Merge specifier + head (if exists)
        if spec_obj and current:
            final = self.merge_operator.external_merge(
                spec_obj.embedding,
                current.embedding,
                head=current.head,  # Head is from complement (e.g., "ball" in "the ball")
                dependent=spec_obj.head,
                label=node.label,
                alpha_is_head=False  # Spec is not head
            )
        else:
            final = current

        return final

    def _get_or_project(
        self,
        embedding: np.ndarray,
        text: str,
        trace: Optional[Dict]
    ) -> np.ndarray:
        """
        Get 244D semantic projection from cache or compute.

        TIER 3: Semantic projection cache (existing AdaptiveSemanticCache)
        """
        # Use existing semantic cache
        try:
            scores = self.semantic_cache.get_scores(text)
            self.stats.semantic_hits += 1
            if trace:
                trace["hits"].append("semantic")

            # Convert to vector
            return np.array([scores[dim] for dim in sorted(scores.keys())])
        except:
            # Cache miss: project
            self.stats.semantic_misses += 1
            if trace:
                trace["misses"].append("semantic")

            # Project embedding to 244D semantic space
            # (This would use SemanticSpectrum.project_vector)
            # For now, just return embedding
            return embedding

    def _xbar_to_cache_key(self, node: XBarNode) -> str:
        """
        Generate cache key from X-bar structure.

        Key format: "CATEGORY:LEVEL:HEAD:STRUCTURE_HASH"

        Example: "N:2:ball:a1b2c3d4"
        """
        # Serialize structure
        structure_str = self._serialize_xbar(node)

        # Hash for compact key
        structure_hash = hashlib.md5(structure_str.encode()).hexdigest()[:8]

        return f"{node.category.value}:{node.level}:{node.head}:{structure_hash}"

    def _serialize_xbar(self, node: XBarNode) -> str:
        """Serialize X-bar structure to string."""
        parts = [f"{node.category.value}{node.level}:{node.head}"]

        if node.specifier:
            parts.append(f"[SPEC:{self._serialize_xbar(node.specifier)}]")

        for adj in node.adjuncts:
            parts.append(f"[ADJ:{self._serialize_xbar(adj)}]")

        if node.complement:
            parts.append(f"[COMP:{self._serialize_xbar(node.complement)}]")

        return "".join(parts)

    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        return {
            "parse_cache": {
                "size": len(self.parse_cache),
                "hits": self.stats.parse_hits,
                "misses": self.stats.parse_misses,
                "hit_rate": self.stats.parse_hit_rate
            },
            "merge_cache": {
                "size": len(self.merge_cache),
                "hits": self.stats.merge_hits,
                "misses": self.stats.merge_misses,
                "hit_rate": self.stats.merge_hit_rate
            },
            "semantic_cache": {
                "hits": self.stats.semantic_hits,
                "misses": self.stats.semantic_misses,
                "hit_rate": self.stats.semantic_hit_rate
            }
        }
```

---

## Integration with Existing Systems

### With Matryoshka Gate

```python
class LinguisticMatryoshkaGate:
    def __init__(
        self,
        embedder,
        config,
        compositional_cache: CompositionalCache  # NEW!
    ):
        self.compositional_cache = compositional_cache
        # ... existing setup ...

    async def gate(self, query, candidates, final_k=10):
        # Get compositional embeddings with caching
        query_emb, query_trace = self.compositional_cache.get_compositional_embedding(
            query, return_trace=True
        )

        candidate_embs = []
        for cand in candidates:
            emb, trace = self.compositional_cache.get_compositional_embedding(cand)
            candidate_embs.append(emb)

        # Now proceed with matryoshka gating using compositional embeddings
        # ...
```

**Benefit:** Reuse phrase compositions across queries!

### With WeavingOrchestrator

```python
class WeavingOrchestrator:
    def __init__(self, cfg, ...):
        # ... existing setup ...

        # Compositional cache (Phase 5)
        if cfg.use_compositional_cache:
            from HoloLoom.performance.compositional_cache import CompositionalCache

            self.compositional_cache = CompositionalCache(
                ug_chunker=self.ug_chunker,
                merge_operator=self.merge_operator,
                semantic_cache=self.semantic_cache
            )
```

---

## Performance Benchmarks (Projected)

### Scenario 1: Repeated Phrases

**Query:** "the big red ball"

**First time (cold):**
```
Parse:    40ms (spaCy)
Merge:    15ms (3 compositions)
Semantic:  5ms (projection)
─────────────
Total:    60ms
```

**Second time (hot):**
```
Parse:     0ms (cached!)
Merge:     0ms (cached!)
Semantic:  0ms (cached!)
─────────────
Total:   0.5ms (hash lookup)
```

**Speedup: 120×**

### Scenario 2: Partial Reuse

**Query 1:** "the big red ball"
**Query 2:** "a big red ball" (different determiner)

**Query 2 performance:**
```
Parse:     40ms (different text, re-parse)
Merge:      3ms (reuse "big red ball" composition!)
Semantic:   5ms
─────────────
Total:     48ms (vs 60ms cold)
```

**Speedup: 1.25× (from partial reuse)**

### Scenario 3: Compositional Reuse Across Queries

**Query 1:** "the red ball"
**Query 2:** "a red ball and a blue cube"

**Compositions cached from Query 1:**
- "red ball" ✅

**Query 2 can reuse:**
- "red ball" (already cached)
- Compute "blue cube" (new)
- Merge with conjunction

**Effective speedup: ~2× from reuse**

---

## Timeline

### Week 1: X-bar Chunker (Foundation)
- **Days 1-2:** Implement `UniversalGrammarChunker`
- **Days 3-4:** Test on sample phrases, validate X-bar structures
- **Day 5:** Integration with ResonanceShed

### Week 2: Merge + Compositional Embedding
- **Days 1-2:** Implement `MergeOperator` with X-bar integration
- **Days 3-4:** Recursive composition algorithm
- **Day 5:** Benchmarking

### Week 3: Compositional Cache
- **Days 1-3:** Implement `CompositionalCache` (3 tiers)
- **Day 4:** Integration with matryoshka gate
- **Day 5:** End-to-end testing, performance validation

---

## Success Metrics

1. **Parse cache hit rate:** >70% (common phrases reused)
2. **Merge cache hit rate:** >50% (compositional reuse)
3. **Overall speedup:** 30-50× on production queries (vs Phase 1)
4. **Memory overhead:** <100MB (reasonable for cache)
5. **Accuracy:** Zero degradation (caching is transparent)

---

## Risks & Mitigations

### Risk 1: Cache invalidation complexity
**Mitigation:** Immutable cache keys (hash-based), no invalidation needed

### Risk 2: Memory bloat
**Mitigation:** LRU eviction, configurable cache sizes

### Risk 3: Composition overhead on cache misses
**Mitigation:** Still faster than full re-computation, graceful degradation

---

## Future Extensions

### Cross-Linguistic Support
- Add UG parameters (head-direction, pro-drop, etc.)
- Language-specific chunkers (Japanese, Arabic, etc.)

### Learned Compositions
- Train neural Merge operator (instead of weighted sum)
- Learn optimal composition functions from data

### Distributed Caching
- Redis backend for shared cache across instances
- Persistent cache across restarts

---

## Conclusion

Phase 5 represents a **paradigm shift** in how HoloLoom processes language:

**From:** Flat, atomic embeddings
**To:** Compositional, hierarchical, cached semantics

**Impact:**
- 🚀 50-100× speedup potential (vs current)
- 🧠 Principled linguistic theory (Universal Grammar)
- ♻️ Massive reuse (phrase compositions cached)
- 🌍 Cross-linguistic foundation (UG parameters)

**Status:** Design complete, ready for implementation after Phase 1-4
**Recommendation:** START with X-bar chunker, iterate to full compositional cache

---

**Next Step:** Add to FULL_IMPLEMENTATION_PLAN.md as Phase 5
