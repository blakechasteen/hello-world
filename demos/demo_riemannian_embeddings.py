"""
Demo: Riemannian Embeddings vs Euclidean Distance
==================================================
Demonstrates the difference between Euclidean L2 distance and geodesic distance
on a product manifold H×S×E for hierarchical concepts.

This demo shows WHY Riemannian geometry matters for semantic embeddings:
- Hierarchical concepts (e.g., "mammal" → "dog" → "beagle") live in hyperbolic space
- Euclidean distance underestimates semantic similarity in hierarchies
- Geodesic distance captures true semantic distance along the manifold

Author: HoloLoom Mathematical Moonshot Team
Date: 2025-11-03
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hololoom.embedding.riemannian_matryoshka import RiemannianMatryoshka, create_riemannian_embedder
from hololoom.embedding.spectral import MatryoshkaEmbeddings


def demo_hierarchical_concepts():
    """
    Compare Euclidean vs Riemannian distances on hierarchical concepts.

    Example: Taxonomy hierarchy
    - Animal > Mammal > Dog > Beagle

    Hyperbolic space naturally represents hierarchies because:
    - Distance grows exponentially with depth
    - More "room" at deeper levels (tree structure)
    """
    print("=" * 80)
    print("Riemannian Embeddings Demo: Hierarchical Concepts")
    print("=" * 80)
    print()

    # Hierarchical concepts (taxonomy)
    concepts = [
        "animal",
        "mammal",
        "dog",
        "beagle",
        "labrador",
        "bird",
        "robin",
        "eagle"
    ]

    print(f"Testing {len(concepts)} hierarchical concepts:")
    for i, concept in enumerate(concepts, 1):
        print(f"  {i}. {concept}")
    print()

    # Create embedders
    print("Creating embedders...")
    print("  - Euclidean: Standard L2 distance")
    print("  - Riemannian: H×S×E product manifold (128×128×128 = 384D)")
    print()

    euclidean_embedder = RiemannianMatryoshka(
        base_embedder=MatryoshkaEmbeddings(sizes=[384]),
        use_riemannian=False  # Standard Euclidean
    )

    riemannian_embedder = create_riemannian_embedder(
        use_riemannian=True,
        sizes=[384],
        hyperbolic_dim=128,  # Hierarchical structure
        spherical_dim=128,   # Clustering structure
        euclidean_dim=128    # Linear features
    )

    # Encode concepts
    print("Encoding concepts...")
    euclidean_embs = euclidean_embedder.encode(concepts)
    riemannian_embs = riemannian_embedder.encode(concepts)
    print(f"  Embedding shape: {euclidean_embs.shape}")
    print()

    # Compare distances for hierarchical pairs
    print("=" * 80)
    print("Distance Comparison: Hierarchical Pairs")
    print("=" * 80)
    print()

    pairs = [
        ("mammal", "dog"),      # Parent → child
        ("dog", "beagle"),      # Parent → child
        ("mammal", "beagle"),   # Grandparent → grandchild
        ("beagle", "labrador"), # Siblings (same parent)
        ("dog", "bird"),        # Different branches
        ("mammal", "robin"),    # Cross-hierarchy
    ]

    print(f"{'Pair':<30} {'Euclidean':<15} {'Riemannian':<15} {'Ratio (R/E)':<15}")
    print("-" * 80)

    for concept1, concept2 in pairs:
        idx1 = concepts.index(concept1)
        idx2 = concepts.index(concept2)

        # Euclidean distance
        euclidean_dist = euclidean_embedder.distance(
            euclidean_embs[idx1],
            euclidean_embs[idx2]
        )

        # Riemannian geodesic distance
        riemannian_dist = riemannian_embedder.distance(
            riemannian_embs[idx1],
            riemannian_embs[idx2]
        )

        ratio = riemannian_dist / (euclidean_dist + 1e-9)

        pair_str = f"{concept1} ↔ {concept2}"
        print(f"{pair_str:<30} {euclidean_dist:<15.4f} {riemannian_dist:<15.4f} {ratio:<15.4f}")

    print()
    print("INTERPRETATION:")
    print("  - Ratio > 1: Riemannian distance captures greater semantic separation")
    print("  - Ratio < 1: Riemannian distance reveals hidden proximity")
    print("  - Hierarchical pairs should show consistent geodesic structure")
    print()


def demo_pairwise_distances():
    """
    Compute full pairwise distance matrix for visualization.
    """
    print("=" * 80)
    print("Pairwise Distance Matrix")
    print("=" * 80)
    print()

    concepts = ["mammal", "dog", "beagle", "cat", "bird", "robin"]

    print(f"Testing {len(concepts)} concepts:")
    for i, concept in enumerate(concepts, 1):
        print(f"  {i}. {concept}")
    print()

    # Create Riemannian embedder
    embedder = create_riemannian_embedder(
        use_riemannian=True,
        sizes=[384],
        hyperbolic_dim=128,
        spherical_dim=128,
        euclidean_dim=128
    )

    # Encode
    embs = embedder.encode(concepts)

    # Compute pairwise distances
    print("Computing pairwise geodesic distances...")
    distances = embedder.pairwise_distances(embs)

    # Print matrix
    print()
    print("Geodesic Distance Matrix:")
    print()

    # Header
    print(f"{'':>12}", end="")
    for concept in concepts:
        print(f"{concept:>12}", end="")
    print()

    # Rows
    for i, concept1 in enumerate(concepts):
        print(f"{concept1:>12}", end="")
        for j, concept2 in enumerate(concepts):
            print(f"{distances[i, j]:>12.4f}", end="")
        print()

    print()
    print("INTERPRETATION:")
    print("  - Diagonal: 0 (self-distance)")
    print("  - Hierarchical pairs (e.g., mammal-dog) have smaller distances")
    print("  - Cross-branch pairs (e.g., dog-bird) have larger distances")
    print()


def demo_geodesic_interpolation():
    """
    Demonstrate geodesic interpolation between concepts.

    Shows how to move along the manifold from one concept to another.
    """
    print("=" * 80)
    print("Geodesic Interpolation")
    print("=" * 80)
    print()

    print("Moving along geodesic: 'mammal' → 'beagle'")
    print()

    # Create embedder
    embedder = create_riemannian_embedder(
        use_riemannian=True,
        sizes=[384],
        hyperbolic_dim=128,
        spherical_dim=128,
        euclidean_dim=128
    )

    # Encode concepts
    concepts = ["mammal", "beagle"]
    embs = embedder.encode(concepts)

    start_point = embs[0]
    end_point = embs[1]

    # Compute tangent vector (log map)
    print("1. Computing tangent vector from 'mammal' to 'beagle'...")
    tangent_vector = embedder.log_map(start_point, end_point)
    tangent_norm = np.linalg.norm(tangent_vector)
    print(f"   Tangent vector norm: {tangent_norm:.4f}")
    print()

    # Interpolate along geodesic
    print("2. Interpolating along geodesic (5 steps)...")
    print()

    steps = 5
    for i in range(steps + 1):
        t = i / steps
        scaled_tangent = t * tangent_vector

        # Move along geodesic using exponential map
        point = embedder.exp_map(start_point, scaled_tangent)

        # Compute distance from start
        dist_from_start = embedder.distance(start_point, point)

        print(f"   t={t:.2f}  Distance from start: {dist_from_start:.4f}")

    print()
    print("INTERPRETATION:")
    print("  - Exponential map: Move along geodesic from base point")
    print("  - Logarithmic map: Get tangent vector between two points")
    print("  - Geodesic path: True 'shortest path' on the manifold")
    print()


def demo_performance_comparison():
    """
    Compare performance: Euclidean vs Riemannian distance.
    """
    print("=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print()

    import time

    n_samples = 100
    n_trials = 10

    print(f"Benchmarking distance computation:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Trials: {n_trials}")
    print()

    # Create embedders
    euclidean_embedder = RiemannianMatryoshka(use_riemannian=False)
    riemannian_embedder = create_riemannian_embedder(use_riemannian=True)

    # Generate random embeddings
    np.random.seed(42)
    embs = np.random.randn(n_samples, 384)

    # Benchmark Euclidean
    print("Benchmarking Euclidean L2 distance...")
    euclidean_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for i in range(n_samples - 1):
            _ = euclidean_embedder.distance(embs[i], embs[i + 1])
        euclidean_times.append(time.perf_counter() - start)

    euclidean_avg = np.mean(euclidean_times) * 1000  # ms
    euclidean_std = np.std(euclidean_times) * 1000

    # Benchmark Riemannian
    print("Benchmarking Riemannian geodesic distance...")
    riemannian_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for i in range(n_samples - 1):
            _ = riemannian_embedder.distance(embs[i], embs[i + 1])
        riemannian_times.append(time.perf_counter() - start)

    riemannian_avg = np.mean(riemannian_times) * 1000  # ms
    riemannian_std = np.std(riemannian_times) * 1000

    # Results
    print()
    print(f"{'Method':<20} {'Time (ms)':<15} {'Overhead':<15}")
    print("-" * 50)
    print(f"{'Euclidean L2':<20} {euclidean_avg:<15.2f} {'1.0x (baseline)':<15}")
    print(f"{'Riemannian Geodesic':<20} {riemannian_avg:<15.2f} {f'{riemannian_avg/euclidean_avg:.1f}x':<15}")
    print()

    print("INTERPRETATION:")
    print("  - Riemannian distance is more expensive (geodesic computation)")
    print("  - Trade-off: Accuracy vs performance")
    print("  - Use Riemannian for hierarchical/semantic tasks, Euclidean for speed")
    print()


def main():
    """Run all demos."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "RIEMANNIAN EMBEDDINGS DEMONSTRATION" + " " * 28 + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Comparing Euclidean L2 vs Geodesic Distance on Semantic Manifolds" + " " * 9 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    try:
        # Demo 1: Hierarchical concepts
        demo_hierarchical_concepts()

        # Demo 2: Pairwise distances
        demo_pairwise_distances()

        # Demo 3: Geodesic interpolation
        demo_geodesic_interpolation()

        # Demo 4: Performance comparison
        demo_performance_comparison()

        print("=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        print()
        print("KEY TAKEAWAYS:")
        print("  1. Riemannian geometry provides TRUE semantic distance")
        print("  2. Hierarchies naturally live in hyperbolic space")
        print("  3. Geodesic distance captures manifold structure")
        print("  4. Trade-off: Accuracy vs computational cost")
        print()
        print("NEXT STEPS:")
        print("  - Enable use_riemannian=True in config.py")
        print("  - Adjust dimension splits (hyperbolic/spherical/euclidean)")
        print("  - Test on your domain-specific hierarchies")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
