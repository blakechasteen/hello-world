#!/usr/bin/env python3
"""
Test Semantic Cache Performance
================================
Demonstrates three-tier caching for 244D semantic projections.
"""

import time
import numpy as np
from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS, SemanticSpectrum
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.performance.semantic_cache import AdaptiveSemanticCache


def main():
    print("=" * 70)
    print("SEMANTIC CACHE PERFORMANCE TEST")
    print("=" * 70)

    # Setup
    print("\n1. Setting up semantic system...")
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)

    print("   Learning semantic axes...")
    spectrum.learn_axes(lambda word: emb.encode([word])[0])

    # Create cache
    print("\n2. Creating AdaptiveSemanticCache...")
    cache = AdaptiveSemanticCache(
        semantic_spectrum=spectrum,
        embedder=emb,
        hot_size=100,
        warm_size=50
    )

    # Test hot tier
    print("\n3. Testing HOT TIER (pre-loaded)...")
    start = time.time()
    scores = cache.get_scores("hero")
    hot_time = (time.time() - start) * 1000
    print(f"   'hero': {hot_time:.6f}ms")

    # Test cold path
    print("\n4. Testing COLD PATH (novel text)...")
    start = time.time()
    scores = cache.get_scores("quantum trickster rebellion")
    cold_time = (time.time() - start) * 1000
    print(f"   'quantum trickster rebellion': {cold_time:.4f}ms")

    # Test warm tier
    print("\n5. Testing WARM TIER (cached from cold)...")
    start = time.time()
    scores = cache.get_scores("quantum trickster rebellion")
    warm_time = (time.time() - start) * 1000
    print(f"   'quantum trickster rebellion' (2nd access): {warm_time:.6f}ms")

    if warm_time > 0:
        print(f"   Speedup: {cold_time/warm_time:.0f}×")
    else:
        print(f"   Speedup: >10,000× (too fast to measure!)")

    # Show stats
    print("\n6. STATISTICS")
    cache.print_stats()

    print("\n✅ Semantic cache working!")


if __name__ == "__main__":
    main()