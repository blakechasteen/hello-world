"""
Demo: Advanced Spectral Methods for HoloLoom
===========================================
Demonstrates graph wavelets, diffusion maps, and multi-scale spectral analysis.

This demo shows:
1. Wavelet detection of local communities
2. Diffusion maps for dimensionality reduction
3. Spectral clustering for community detection
4. Multi-scale spectral analysis (coarse-to-fine)
5. Integration with HoloLoom's spectral feature extraction

Run: python demos/demo_spectral_methods.py
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.memory.graph import KG, KGEdge
from hololoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings
from hololoom.embedding.spectral_multiscale import (
    create_multiscale_analyzer,
    create_hierarchical_clusterer
)
from hololoom.config import Config


def create_test_knowledge_graph() -> KG:
    """Create a test knowledge graph with clear community structure."""
    kg = KG()

    # Community 1: Transformer Architecture
    transformer_edges = [
        KGEdge("attention", "transformer", "USES", 1.0),
        KGEdge("transformer", "neural_network", "IS_A", 1.0),
        KGEdge("attention", "neural_network", "PART_OF", 0.8),
        KGEdge("multi-head attention", "attention", "IS_A", 1.0),
        KGEdge("self-attention", "attention", "IS_A", 1.0),
        KGEdge("cross-attention", "attention", "IS_A", 1.0),
    ]

    # Community 2: Transformer Models
    model_edges = [
        KGEdge("BERT", "transformer", "IS_A", 1.0),
        KGEdge("GPT", "transformer", "IS_A", 1.0),
        KGEdge("T5", "transformer", "IS_A", 1.0),
        KGEdge("BERT", "masked_language_model", "USES", 1.0),
        KGEdge("GPT", "autoregressive_model", "USES", 1.0),
        KGEdge("T5", "seq2seq_model", "USES", 1.0),
    ]

    # Community 3: NLP Applications
    app_edges = [
        KGEdge("BERT", "NLP", "USED_IN", 1.0),
        KGEdge("GPT", "NLP", "USED_IN", 1.0),
        KGEdge("T5", "NLP", "USED_IN", 1.0),
        KGEdge("NLP", "text_classification", "INCLUDES", 1.0),
        KGEdge("NLP", "question_answering", "INCLUDES", 1.0),
        KGEdge("NLP", "text_generation", "INCLUDES", 1.0),
    ]

    # Community 4: Training Methods
    training_edges = [
        KGEdge("transformer", "backpropagation", "TRAINED_WITH", 1.0),
        KGEdge("backpropagation", "optimization", "IS_A", 1.0),
        KGEdge("optimization", "Adam", "USES", 1.0),
        KGEdge("optimization", "SGD", "USES", 1.0),
        KGEdge("Adam", "gradient_descent", "IS_A", 1.0),
    ]

    # Bridge edges (connecting communities)
    bridge_edges = [
        KGEdge("transformer", "NLP", "USED_IN", 0.5),
        KGEdge("neural_network", "optimization", "REQUIRES", 0.5),
    ]

    all_edges = transformer_edges + model_edges + app_edges + training_edges + bridge_edges
    kg.add_edges(all_edges)

    return kg


async def demo_wavelets(kg: KG):
    """Demonstrate wavelet features for local community detection."""
    print("=" * 70)
    print("DEMO 1: Graph Wavelets - Local Community Detection")
    print("=" * 70)

    from hololoom.warp.spectral_methods import GraphLaplacian, SpectralWavelet, LaplacianType

    # Create Laplacian
    laplacian = GraphLaplacian(kg, laplacian_type=LaplacianType.NORMALIZED)

    # Create multi-scale wavelet analyzer
    wavelet = SpectralWavelet(
        laplacian=laplacian,
        n_scales=3,
        scale_min=0.1,
        scale_max=10.0
    )

    print(f"\nGraph: {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")
    print(f"Wavelet scales: {wavelet.scales}")

    # Create uniform signal
    n_nodes = kg.G.number_of_nodes()
    signal = np.ones(n_nodes) / n_nodes

    # Compute wavelets
    heat_coeffs = wavelet.transform(signal, kernel='heat')
    mhat_coeffs = wavelet.transform(signal, kernel='mexican_hat')

    print("\nWavelet Energies (mean absolute coefficient):")
    print("-" * 50)
    print(f"{'Scale':<10} {'Heat Kernel':<20} {'Mexican Hat':<20}")
    print("-" * 50)

    for scale in wavelet.scales:
        heat_energy = np.mean(np.abs(heat_coeffs[scale]))
        mhat_energy = np.mean(np.abs(mhat_coeffs[scale]))
        print(f"{scale:<10.2f} {heat_energy:<20.4f} {mhat_energy:<20.4f}")

    print("\nInterpretation:")
    print("- Heat kernel: Smooth diffusion (detects broad communities)")
    print("- Mexican hat: Edge detection (detects sharp boundaries)")
    print("- Large scales: Global structure")
    print("- Small scales: Local patterns")


async def demo_diffusion_maps(kg: KG):
    """Demonstrate diffusion maps for dimensionality reduction."""
    print("\n" + "=" * 70)
    print("DEMO 2: Diffusion Maps - Nonlinear Dimensionality Reduction")
    print("=" * 70)

    # Compute diffusion map (uses built-in caching)
    embedding = kg.compute_diffusion_map(n_dims=8, t=1.0)

    print(f"\nDiffusion Map Embedding:")
    print(f"  Original graph: {kg.G.number_of_nodes()} nodes")
    print(f"  Embedding dimension: {embedding.shape[1]}")
    print(f"  Embedding shape: {embedding.shape}")

    # Show coordinates for sample entities
    sample_entities = ["transformer", "BERT", "NLP", "attention", "optimization"]
    print(f"\nSample Diffusion Coordinates (first 4 dims):")
    print("-" * 50)

    for entity in sample_entities:
        coords = kg.get_diffusion_coordinates(entity, n_dims=8)
        if coords is not None:
            print(f"{entity:<20s}: [{coords[0]:6.3f}, {coords[1]:6.3f}, {coords[2]:6.3f}, {coords[3]:6.3f}, ...]")

    # Compute pairwise distances in diffusion space
    print(f"\nDiffusion Distances (random walk similarity):")
    print("-" * 50)

    entities = list(kg.G.nodes())
    entity_indices = {e: i for i, e in enumerate(entities)}

    pairs = [
        ("transformer", "BERT"),
        ("transformer", "optimization"),
        ("BERT", "GPT"),
        ("attention", "NLP"),
    ]

    for e1, e2 in pairs:
        if e1 in entity_indices and e2 in entity_indices:
            idx1 = entity_indices[e1]
            idx2 = entity_indices[e2]
            dist = np.linalg.norm(embedding[idx1] - embedding[idx2])
            print(f"{e1:<15s} <-> {e2:<15s}: {dist:.4f}")

    print("\nInterpretation:")
    print("- Small distance: Entities are semantically close (many paths between them)")
    print("- Large distance: Entities are distant in graph structure")


async def demo_spectral_clustering(kg: KG):
    """Demonstrate spectral clustering for community detection."""
    print("\n" + "=" * 70)
    print("DEMO 3: Spectral Clustering - Community Detection")
    print("=" * 70)

    # Cluster with different numbers
    for n_clusters in [2, 3, 4]:
        print(f"\nClustering into {n_clusters} communities:")
        print("-" * 50)

        clusters = kg.spectral_cluster(n_clusters=n_clusters)

        # Group entities by cluster
        cluster_groups = {}
        for entity, cluster_id in clusters.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(entity)

        # Print clusters
        for cluster_id in sorted(cluster_groups.keys()):
            entities = cluster_groups[cluster_id]
            print(f"  Cluster {cluster_id}: {', '.join(sorted(entities)[:5])}")
            if len(entities) > 5:
                print(f"             ... and {len(entities) - 5} more")

    print("\nInterpretation:")
    print("- Spectral clustering uses graph Laplacian eigenvectors")
    print("- Entities in same cluster are structurally similar")
    print("- Useful for: knowledge organization, retrieval optimization")


async def demo_multiscale_analysis(kg: KG):
    """Demonstrate multi-scale spectral analysis."""
    print("\n" + "=" * 70)
    print("DEMO 4: Multi-Scale Spectral Analysis (Coarse-to-Fine)")
    print("=" * 70)

    # Create multi-scale analyzer
    analyzer = create_multiscale_analyzer(
        scales=[96, 192, 384],
        wavelet_scales=[0.1, 1.0, 10.0]
    )

    # Query entities
    query_entities = ["transformer", "attention"]

    print(f"\nQuery entities: {query_entities}")
    print(f"Scales: {analyzer.scales} (matching Matryoshka embeddings)")

    # Analyze at multiple scales
    multiscale_results = analyzer.analyze_multiscale(kg, query_entities)

    print("\nMulti-Scale Analysis Results:")
    print("-" * 70)

    for scale, result in multiscale_results.items():
        print(f"\nScale: {scale}d")

        # Eigenvalues
        eigs = result['eigenvalues']
        if len(eigs) > 0:
            print(f"  Eigenvalues (top 4): {eigs[:4]}")
            fiedler = eigs[1] if len(eigs) > 1 else 0.0
            print(f"  Fiedler value: {fiedler:.4f} (connectivity)")

        # Wavelet energy
        wavelets = result['wavelets']
        if len(wavelets) > 0:
            wavelet_energy = np.mean(np.abs(wavelets))
            print(f"  Wavelet energy: {wavelet_energy:.4f}")

        # Clusters
        clusters = result['clusters']
        n_communities = len(set(clusters.values())) if clusters else 0
        print(f"  Communities detected: {n_communities}")

        # Diffusion
        diffusion = result['diffusion']
        if len(diffusion) > 0:
            print(f"  Diffusion embedding: {diffusion.shape}")

    # Fuse features
    print("\n" + "-" * 70)
    fused_features = analyzer.fuse_multiscale_features(multiscale_results)
    print(f"\nFused multi-scale feature vector:")
    print(f"  Dimension: {len(fused_features)}")
    print(f"  Vector: {fused_features}")

    print("\nInterpretation:")
    print("- Coarse scale (96d): Global structure, major communities")
    print("- Medium scale (192d): Regional patterns, subgraphs")
    print("- Fine scale (384d): Local neighborhoods, micro-clusters")
    print("- Fused features: Complete multi-resolution representation")


async def demo_hierarchical_clustering(kg: KG):
    """Demonstrate hierarchical spectral clustering."""
    print("\n" + "=" * 70)
    print("DEMO 5: Hierarchical Spectral Clustering")
    print("=" * 70)

    # Create hierarchical clusterer
    clusterer = create_hierarchical_clusterer(max_depth=3, min_cluster_size=2)

    # Cluster hierarchically
    cluster_paths = clusterer.cluster_hierarchical(kg)

    print(f"\nHierarchical Clustering (max_depth=3):")
    print("-" * 70)
    print(f"{'Entity':<30s} | {'Cluster Path':<20s}")
    print("-" * 70)

    for entity, path in sorted(cluster_paths.items()):
        path_str = ' -> '.join(map(str, path))
        print(f"{entity:<30s} | {path_str}")

    # Analyze cluster tree structure
    depth_counts = {}
    for path in cluster_paths.values():
        depth = len(path)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print(f"\nCluster Tree Structure:")
    print("-" * 50)
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]} entities")

    print("\nInterpretation:")
    print("- Hierarchical clustering creates a tree structure")
    print("- Level 0: Coarsest partition")
    print("- Level N: Finest partition")
    print("- Useful for: multi-resolution retrieval, knowledge organization")


async def demo_integration_with_spectral_fusion(kg: KG):
    """Demonstrate integration with HoloLoom's SpectralFusion."""
    print("\n" + "=" * 70)
    print("DEMO 6: Integration with HoloLoom's Spectral Feature Extraction")
    print("=" * 70)

    # Create embeddings
    emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Test texts
    texts = [
        "Transformer architecture uses multi-head attention",
        "BERT is a bidirectional transformer model",
        "GPT uses autoregressive language modeling"
    ]

    # Create SpectralFusion WITHOUT wavelets/diffusion (baseline)
    print("\nBaseline (without wavelets/diffusion):")
    print("-" * 50)

    spectral_baseline = SpectralFusion(
        k_eigen=4,
        svd_components=2,
        use_wavelets=False,
        use_diffusion_maps=False
    )

    subgraph = kg.subgraph_for_entities(["transformer", "BERT", "attention"], expand=True)
    psi_baseline, metrics_baseline = await spectral_baseline.features(subgraph, texts, emb)

    print(f"Feature dimension: {len(psi_baseline)}")
    print(f"Metrics: {metrics_baseline}")

    # Create SpectralFusion WITH wavelets/diffusion
    print("\nEnhanced (with wavelets + diffusion maps):")
    print("-" * 50)

    spectral_enhanced = SpectralFusion(
        k_eigen=4,
        svd_components=2,
        use_wavelets=True,
        wavelet_scales=[0.1, 1.0, 10.0],
        use_diffusion_maps=True,
        diffusion_map_dims=32
    )

    psi_enhanced, metrics_enhanced = await spectral_enhanced.features(subgraph, texts, emb)

    print(f"Feature dimension: {len(psi_enhanced)}")
    print(f"Metrics: {metrics_enhanced}")

    print(f"\nFeature Comparison:")
    print("-" * 50)
    print(f"Baseline dimension:  {len(psi_baseline)}")
    print(f"Enhanced dimension:  {len(psi_enhanced)}")
    print(f"Additional features: {len(psi_enhanced) - len(psi_baseline)}")
    print(f"  -> {len(psi_enhanced) - len(psi_baseline)} new spectral features from wavelets + diffusion")

    print("\nPerformance Characteristics:")
    print("-" * 50)
    print("Baseline:")
    print("  - Computation: O(n²) for eigenvalues")
    print("  - Time: ~1-5ms for typical graphs")
    print("  - Features: Graph connectivity + topic coherence")
    print("\nEnhanced:")
    print("  - Computation: O(n³) for wavelets (cached)")
    print("  - Time: ~10-50ms for typical graphs")
    print("  - Features: + Multi-scale local structure + Intrinsic geometry")
    print("  - Benefit: Detects local communities wavelets can't see")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("ADVANCED SPECTRAL METHODS FOR HOLOLOOM - COMPREHENSIVE DEMO")
    print("=" * 70)

    # Create test knowledge graph
    kg = create_test_knowledge_graph()

    print(f"\nTest Knowledge Graph:")
    print(f"  Nodes: {kg.G.number_of_nodes()}")
    print(f"  Edges: {kg.G.number_of_edges()}")
    print(f"  Communities: Transformer architecture, Models, NLP, Training")

    # Run demos
    await demo_wavelets(kg)
    await demo_diffusion_maps(kg)
    await demo_spectral_clustering(kg)
    await demo_multiscale_analysis(kg)
    await demo_hierarchical_clustering(kg)
    await demo_integration_with_spectral_fusion(kg)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)

    print("\nKey Takeaways:")
    print("-" * 70)
    print("1. Wavelets detect local communities at multiple scales")
    print("2. Diffusion maps reveal intrinsic graph geometry")
    print("3. Spectral clustering finds structural communities")
    print("4. Multi-scale analysis enables coarse-to-fine reasoning")
    print("5. Hierarchical clustering creates knowledge organization trees")
    print("6. Integration with HoloLoom adds rich structural features")

    print("\nConfiguration:")
    print("-" * 70)
    print("To enable in HoloLoom:")
    print("  config.use_wavelets = True")
    print("  config.wavelet_scales = [0.1, 1.0, 10.0]")
    print("  config.use_diffusion_maps = True")
    print("  config.diffusion_map_dims = 32")
    print("\nPerformance:")
    print("  - Baseline spectral features: ~1-5ms")
    print("  - With wavelets: ~10-50ms (O(n³), cached)")
    print("  - With diffusion maps: ~20-100ms (O(n³), cached)")
    print("  - Combined: ~30-150ms (first computation, then cached)")

    print("\n[OK] All demos completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(main())
