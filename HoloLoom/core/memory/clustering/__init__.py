"""
HoloLoom Smart Clustering
=========================

One-line API for intelligent clustering with automatic optimization.

WASM + EXO + HoloLoom Magic = Easy Clustering

Quick Start:
    from HoloLoom.clustering import cluster

    results = cluster(["AI is cool", "Python rocks", "Neural nets learn"])

    for c in results:
        print(f"{c.label}: {c.items}")

Features:
- Auto-detects optimal number of clusters (no guessing k)
- Semantic labels (not "Cluster 0, 1, 2")
- Matryoshka multi-scale embeddings (fast → accurate)
- Thompson Sampling for intelligent k exploration
- Works with texts, dicts, or pre-computed embeddings
"""

from .core import (
    cluster,
    smart_cluster,
    auto_cluster,
    ClusterResult,
    ClusteringOutput,
)

from .labeler import (
    label_clusters,
    generate_cluster_summary,
    LabelCandidate,
    DOMAIN_CATEGORIES,
)

from .thompson import (
    ThompsonClusterSampler,
    find_optimal_k_thompson,
    adaptive_k_search,
    BetaPrior,
)

__all__ = [
    # Main API
    "cluster",
    "smart_cluster",
    "auto_cluster",

    # Result types
    "ClusterResult",
    "ClusteringOutput",

    # Labeling
    "label_clusters",
    "generate_cluster_summary",
    "LabelCandidate",
    "DOMAIN_CATEGORIES",

    # Thompson Sampling
    "ThompsonClusterSampler",
    "find_optimal_k_thompson",
    "adaptive_k_search",
    "BetaPrior",
]


# Version
__version__ = "1.0.0"
