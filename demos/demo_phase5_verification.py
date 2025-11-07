#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5 Compositional Cache Verification Demo
==============================================

Verifies that Phase 5 is integrated and working correctly with the WeavingOrchestrator.

Expected Results:
- First query: COLD (cache misses, ~100-150ms)
- Second identical query: HOT (cache hits, ~0.5-3ms) → 50-300× speedup!
- Related query: WARM (partial cache hits, ~30-50ms) → compositional reuse!

The magic of Phase 5 is compositional reuse:
- Query 1: "the red ball" → caches "red ball" composition
- Query 2: "a red ball" → REUSES "red ball"! (different determiner)
"""

import asyncio
import time
import sys
import io

# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

def create_test_shards():
    """Create simple test shards about balls."""
    return [
        MemoryShard(
            id="shard_1",
            text="A ball is a round object used in many sports.",
            episode="knowledge",
            entities=["ball", "sports"],
            motifs=["sports", "objects"],
            metadata={"topic": "sports"}
        ),
        MemoryShard(
            id="shard_2",
            text="Red balls are often used in cricket and dodgeball.",
            episode="knowledge",
            entities=["red ball", "cricket", "dodgeball"],
            motifs=["sports", "color"],
            metadata={"topic": "sports", "color": "red"}
        ),
        MemoryShard(
            id="shard_3",
            text="The passive voice uses a form of 'to be' plus a past participle.",
            episode="knowledge",
            entities=["passive voice", "past participle"],
            motifs=["grammar", "linguistics"],
            metadata={"topic": "grammar"}
        ),
    ]

async def main():
    """Run Phase 5 verification demo."""

    print("=" * 80)
    print("Phase 5 Compositional Cache Verification Demo")
    print("=" * 80)
    print()

    # Create config with Phase 5 enabled
    config = Config.fused()

    # Verify Phase 5 is enabled
    print(f"✓ Phase 5 enabled: {config.enable_linguistic_gate}")
    print(f"✓ Linguistic mode: {config.linguistic_mode}")
    print(f"✓ Compositional cache: {config.use_compositional_cache}")
    print(f"✓ Parse cache size: {config.parse_cache_size}")
    print(f"✓ Merge cache size: {config.merge_cache_size}")
    print()

    # Create orchestrator
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        print("🚀 WeavingOrchestrator initialized with Phase 5")

        # Check if linguistic gate was created
        if hasattr(orchestrator, 'linguistic_gate') and orchestrator.linguistic_gate:
            print("✓ Linguistic Matryoshka Gate active")
            if hasattr(orchestrator.linguistic_gate, 'compositional_cache'):
                print("✓ Compositional Cache active")
            else:
                print("⚠ Warning: Compositional Cache not found in linguistic gate")
        else:
            print("⚠ Warning: Linguistic gate not initialized (spaCy might not be installed)")
            print("   Install spaCy: pip install spacy && python -m spacy download en_core_web_sm")

        print()
        print("-" * 80)
        print("Test 1: COLD PATH (first query)")
        print("-" * 80)

        query1 = Query(text="What is a red ball?")

        start = time.perf_counter()
        result1 = await orchestrator.weave(query1)
        elapsed1 = (time.perf_counter() - start) * 1000

        print(f"Query: '{query1.text}'")
        print(f"Latency: {elapsed1:.2f}ms (COLD PATH)")
        print(f"Confidence: {result1.confidence:.3f}")
        print()

        # Check cache stats
        if hasattr(orchestrator, 'linguistic_gate') and orchestrator.linguistic_gate:
            if hasattr(orchestrator.linguistic_gate, 'compositional_cache'):
                cache = orchestrator.linguistic_gate.compositional_cache
                print(f"Cache Stats:")
                print(f"  Parse cache: {cache.stats.parse_hits}/{cache.stats.parse_hits + cache.stats.parse_misses} ({cache.stats.parse_hit_rate:.1%})")
                print(f"  Merge cache: {cache.stats.merge_hits}/{cache.stats.merge_hits + cache.stats.merge_misses} ({cache.stats.merge_hit_rate:.1%})")
                print(f"  Overall: {cache.stats.overall_hit_rate:.1%}")

        print()
        print("-" * 80)
        print("Test 2: HOT PATH (identical query - should hit cache)")
        print("-" * 80)

        start = time.perf_counter()
        result2 = await orchestrator.weave(query1)  # Same query
        elapsed2 = (time.perf_counter() - start) * 1000

        print(f"Query: '{query1.text}'")
        print(f"Latency: {elapsed2:.2f}ms (HOT PATH)")
        print(f"Speedup: {elapsed1/elapsed2:.1f}× faster")
        print(f"Confidence: {result2.confidence:.3f}")
        print()

        # Check cache stats
        if hasattr(orchestrator, 'linguistic_gate') and orchestrator.linguistic_gate:
            if hasattr(orchestrator.linguistic_gate, 'compositional_cache'):
                cache = orchestrator.linguistic_gate.compositional_cache
                print(f"Cache Stats (after hot query):")
                print(f"  Parse cache: {cache.stats.parse_hits}/{cache.stats.parse_hits + cache.stats.parse_misses} ({cache.stats.parse_hit_rate:.1%})")
                print(f"  Merge cache: {cache.stats.merge_hits}/{cache.stats.merge_hits + cache.stats.merge_misses} ({cache.stats.merge_hit_rate:.1%})")
                print(f"  Overall: {cache.stats.overall_hit_rate:.1%}")

        print()
        print("-" * 80)
        print("Test 3: WARM PATH (related query - compositional reuse)")
        print("-" * 80)

        query3 = Query(text="Is a red ball round?")  # Different syntax, same "red ball"

        start = time.perf_counter()
        result3 = await orchestrator.weave(query3)
        elapsed3 = (time.perf_counter() - start) * 1000

        print(f"Query: '{query3.text}'")
        print(f"Latency: {elapsed3:.2f}ms (WARM PATH - partial reuse)")
        print(f"Speedup vs cold: {elapsed1/elapsed3:.1f}× faster")
        print(f"Confidence: {result3.confidence:.3f}")
        print()

        # Check cache stats
        if hasattr(orchestrator, 'linguistic_gate') and orchestrator.linguistic_gate:
            if hasattr(orchestrator.linguistic_gate, 'compositional_cache'):
                cache = orchestrator.linguistic_gate.compositional_cache
                print(f"Cache Stats (after compositional reuse):")
                print(f"  Parse cache: {cache.stats.parse_hits}/{cache.stats.parse_hits + cache.stats.parse_misses} ({cache.stats.parse_hit_rate:.1%})")
                print(f"  Merge cache: {cache.stats.merge_hits}/{cache.stats.merge_hits + cache.stats.merge_misses} ({cache.stats.merge_hit_rate:.1%})")
                print(f"  Overall: {cache.stats.overall_hit_rate:.1%}")
                print()
                print(f"📊 Total speedup from caching: {elapsed1/elapsed2:.1f}× (hot) to {elapsed1/elapsed3:.1f}× (warm)")

        print()
        print("=" * 80)
        print("Phase 5 Verification Complete!")
        print("=" * 80)

        if hasattr(orchestrator, 'linguistic_gate') and orchestrator.linguistic_gate:
            if hasattr(orchestrator.linguistic_gate, 'compositional_cache'):
                cache = orchestrator.linguistic_gate.compositional_cache
                if cache.stats.overall_hit_rate > 0.3:
                    print("✅ SUCCESS: Compositional cache is working!")
                    print(f"   Cache hit rate: {cache.stats.overall_hit_rate:.1%}")
                    print(f"   Peak speedup: {elapsed1/elapsed2:.1f}×")
                else:
                    print("⚠️  Cache hit rate lower than expected.")
                    print("   This might be normal for very different queries.")
            else:
                print("⚠️  Compositional cache not available.")
                print("   Install spaCy to enable Phase 5:")
                print("   pip install spacy && python -m spacy download en_core_web_sm")
        else:
            print("⚠️  Linguistic gate not initialized.")
            print("   Install spaCy to enable Phase 5:")
            print("   pip install spacy && python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    asyncio.run(main())
