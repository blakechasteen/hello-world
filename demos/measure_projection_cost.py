"""
Measure computational cost of semantic projection operations
"""
import numpy as np
import time
from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS, SemanticSpectrum
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

def measure_operations():
    print("=" * 70)
    print("MEASURING SEMANTIC PROJECTION COSTS")
    print("=" * 70)

    # Setup
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)

    # Learn axes (one-time cost)
    print("\n1. Learning semantic axes (one-time setup)...")
    start = time.time()
    spectrum.learn_axes(lambda word: emb.encode([word])[0])
    learn_time = time.time() - start
    print(f"   Time: {learn_time:.2f}s for {len(EXTENDED_244_DIMENSIONS)} dimensions")

    # Test text
    text = "hero's journey into the underworld"

    # Measure embedding (Step 1)
    print(f"\n2. Text → 384D embedding (neural network)...")
    times = []
    for i in range(10):
        start = time.time()
        vec_384d = emb.encode([text])[0]
        times.append(time.time() - start)
    embed_time = np.mean(times) * 1000  # Convert to ms
    print(f"   Average: {embed_time:.2f}ms per text")

    # Measure projection (Step 2)
    print(f"\n3. 384D → 244D semantic projection...")
    vec_384d = emb.encode([text])[0]  # Pre-compute
    times = []
    for i in range(1000):  # More iterations since it's faster
        start = time.time()
        scores = spectrum.project_vector(vec_384d)
        times.append(time.time() - start)
    project_time = np.mean(times) * 1000  # Convert to ms
    print(f"   Average: {project_time:.4f}ms per projection")
    print(f"   Operations: 244 dimensions × 384 elements = {244 * 384 * 2:,} FLOPs")

    # Measure cache lookup (Step 0 - ideal)
    print(f"\n4. Dictionary lookup (cached)...")
    cache = {text: scores}
    times = []
    for i in range(100000):  # Many iterations - it's VERY fast
        start = time.time()
        result = cache.get(text)
        times.append(time.time() - start)
    cache_time = np.mean(times) * 1000  # Convert to ms
    print(f"   Average: {cache_time:.6f}ms per lookup")

    # Compare
    print("\n" + "=" * 70)
    print("COST COMPARISON")
    print("=" * 70)
    total_time = embed_time + project_time
    print(f"Cache lookup:     {cache_time:.6f}ms  (baseline)")
    print(f"Projection only:  {project_time:.4f}ms  ({project_time/cache_time:,.0f}× slower than cache)")
    print(f"Embedding only:   {embed_time:.2f}ms  ({embed_time/cache_time:,.0f}× slower than cache)")
    print(f"Full pipeline:    {total_time:.2f}ms  ({total_time/cache_time:,.0f}× slower than cache)")

    print(f"\nCost breakdown:")
    print(f"  Embedding:  {100*embed_time/total_time:.1f}% of total time")
    print(f"  Projection: {100*project_time/total_time:.1f}% of total time")

    print(f"\nSpeedup from caching:")
    print(f"  vs projection only: {project_time/cache_time:,.0f}× faster")
    print(f"  vs full pipeline:   {total_time/cache_time:,.0f}× faster")

if __name__ == "__main__":
    measure_operations()
