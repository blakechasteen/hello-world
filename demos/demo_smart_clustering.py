"""
Smart Clustering Demo
=====================

Demonstrates HoloLoom's one-line clustering API.

WASM + EXO + Magic Sauce = Easy Clustering

Run:
    PYTHONPATH=. python demos/demo_smart_clustering.py
"""

import sys
import time

# Sample data for clustering
SAMPLE_TEXTS = [
    # Technology / AI cluster
    "Machine learning models require training data",
    "Neural networks use backpropagation for learning",
    "Deep learning has revolutionized computer vision",
    "GPT models generate human-like text",
    "Transformers use attention mechanisms",

    # Technology / Web cluster
    "React is a popular JavaScript framework",
    "CSS styles web pages and layouts",
    "HTML provides structure for websites",
    "Node.js runs JavaScript on servers",
    "REST APIs enable web service communication",

    # Science cluster
    "Photosynthesis converts sunlight to energy",
    "DNA contains genetic information",
    "Cells are the basic unit of life",
    "Evolution explains species diversity",
    "Mitochondria produce cellular energy",

    # Business cluster
    "Marketing strategies drive customer acquisition",
    "Revenue growth requires sales optimization",
    "Customer retention improves profitability",
    "Brand awareness builds market presence",
    "Product launches need market research",

    # Food / Cooking cluster
    "Pasta should be cooked al dente",
    "Fresh ingredients improve dish quality",
    "Baking requires precise measurements",
    "Spices add flavor to recipes",
    "Sauces enhance meal presentation",
]


def main():
    print("=" * 60)
    print("HoloLoom Smart Clustering Demo")
    print("=" * 60)
    print()

    # Check dependencies
    try:
        from HoloLoom.clustering import cluster, generate_cluster_summary
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure to run with PYTHONPATH=.")
        sys.exit(1)

    # Demo 1: Basic one-line clustering
    print("Demo 1: One-Line Clustering")
    print("-" * 40)

    start = time.time()
    results = cluster(SAMPLE_TEXTS)
    elapsed = time.time() - start

    print(f"Clustered {len(SAMPLE_TEXTS)} texts into {results.n_clusters} clusters")
    print(f"Time: {elapsed:.2f}s")
    print(f"Silhouette score: {results.silhouette:.3f}")
    print()

    for c in results:
        print(f"  {c.label}")
        print(f"    Size: {c.size} items")
        print(f"    Confidence: {c.confidence:.1%}")
        print(f"    Example: {c.items[0][:50]}...")
        print()

    # Demo 2: Fixed k
    print("\nDemo 2: Fixed k=3 Clusters")
    print("-" * 40)

    results = cluster(SAMPLE_TEXTS, k=3)
    print(f"Forced {results.n_clusters} clusters")
    for c in results:
        print(f"  {c.label}: {c.size} items")

    # Demo 3: Using different methods
    print("\n\nDemo 3: Clustering Methods")
    print("-" * 40)

    for method in ["kmeans", "hierarchical"]:
        start = time.time()
        results = cluster(SAMPLE_TEXTS, method=method, use_thompson=False)
        elapsed = time.time() - start
        print(f"  {method}: {results.n_clusters} clusters, {elapsed:.2f}s")

    # Demo 4: Thompson Sampling vs Grid Search
    print("\n\nDemo 4: Thompson Sampling vs Grid Search")
    print("-" * 40)

    # Grid search (try all k values)
    start = time.time()
    results_grid = cluster(SAMPLE_TEXTS, use_thompson=False)
    grid_time = time.time() - start

    # Thompson Sampling (intelligent exploration)
    start = time.time()
    results_thompson = cluster(SAMPLE_TEXTS, use_thompson=True, thompson_budget=10)
    thompson_time = time.time() - start

    print(f"  Grid Search:      k={results_grid.n_clusters}, "
          f"silhouette={results_grid.silhouette:.3f}, time={grid_time:.2f}s")
    print(f"  Thompson Sampling: k={results_thompson.n_clusters}, "
          f"silhouette={results_thompson.silhouette:.3f}, time={thompson_time:.2f}s")

    # Demo 5: Cluster summary
    print("\n\nDemo 5: Human-Readable Summary")
    print("-" * 40)

    results = cluster(SAMPLE_TEXTS)
    summary = generate_cluster_summary(results.clusters, include_examples=2)
    print(summary)

    # Demo 6: With custom data
    print("\nDemo 6: Clustering Custom Data")
    print("-" * 40)

    custom_data = [
        {"text": "Python is great for data science"},
        {"text": "R is used for statistical analysis"},
        {"text": "JavaScript powers the web"},
        {"text": "TypeScript adds types to JavaScript"},
        {"text": "Pandas handles data manipulation"},
        {"text": "NumPy enables numerical computing"},
    ]

    results = cluster(custom_data)
    print(f"Clustered {len(custom_data)} items into {results.n_clusters} clusters:")
    for c in results:
        print(f"  {c.label}: {[item['text'][:30] + '...' for item in custom_data if item['text'] in c.items]}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
