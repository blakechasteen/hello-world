"""
Multi-Scale Spectral Analysis for Knowledge Graphs
==================================================
Hierarchical spectral analysis combining wavelets with Matryoshka embeddings.

This module provides:
1. Coarse-to-fine spectral analysis
2. Hierarchical spectral clustering
3. Multi-scale wavelet decomposition
4. Integration with Matryoshka multi-scale embeddings

Philosophy:
-----------
Just as Matryoshka embeddings provide multi-scale semantic representations,
multi-scale spectral analysis provides multi-scale structural representations.

- Coarse scales (large wavelets): Capture global graph structure
- Fine scales (small wavelets): Capture local communities and patterns
- Together: Enable adaptive complexity matching query requirements

Author: HoloLoom Spectral Theory Team
Date: 2025-11-03
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
import warnings

from HoloLoom.documentation.types import Vector

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    warnings.warn("scipy not available - multi-scale spectral features will use dense fallback")


# ============================================================================
# Multi-Scale Spectral Analyzer
# ============================================================================

@dataclass
class MultiScaleSpectralAnalyzer:
    """
    Multi-scale spectral analysis combining wavelets with Matryoshka embeddings.

    Provides hierarchical spectral features at multiple resolutions:
    - Coarse (global): Graph-level connectivity, major communities
    - Medium (regional): Subgraph structure, local patterns
    - Fine (local): Individual node neighborhoods, micro-clusters

    This mirrors the Matryoshka embedding hierarchy:
    - Large embeddings (384d) ↔ Fine spectral features
    - Medium embeddings (192d) ↔ Medium spectral features
    - Small embeddings (96d) ↔ Coarse spectral features
    """

    scales: List[int] = field(default_factory=lambda: [96, 192, 384])
    wavelet_scales: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])

    def __post_init__(self):
        """Validate scales are in ascending order."""
        assert sorted(self.scales) == self.scales, "Scales must be in ascending order"

    def analyze_multiscale(
        self,
        kg: 'KG',
        query_entities: List[str]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Perform multi-scale spectral analysis.

        Args:
            kg: Knowledge graph
            query_entities: Entities relevant to current query

        Returns:
            Dict mapping scale → {
                'eigenvalues': Laplacian spectrum,
                'wavelets': Wavelet coefficients,
                'diffusion': Diffusion coordinates,
                'clusters': Cluster assignments
            }
        """
        results = {}

        for i, scale in enumerate(self.scales):
            # Determine analysis depth based on scale
            if i == 0:  # Coarse (96d equivalent)
                max_hops = 1
                n_clusters = 2
                wavelet_scale = self.wavelet_scales[0]  # Large scale
            elif i == 1:  # Medium (192d equivalent)
                max_hops = 2
                n_clusters = 4
                wavelet_scale = self.wavelet_scales[1]  # Medium scale
            else:  # Fine (384d equivalent)
                max_hops = 3
                n_clusters = 8
                wavelet_scale = self.wavelet_scales[2]  # Fine scale

            # Extract subgraph at this scale
            subgraph = kg.subgraph_for_entities(
                query_entities,
                expand=True,
                max_hops=max_hops
            )

            # Compute spectral features
            scale_features = self._analyze_subgraph(
                subgraph,
                n_clusters=n_clusters,
                wavelet_scale=wavelet_scale
            )

            results[scale] = scale_features

        return results

    def _analyze_subgraph(
        self,
        subgraph: nx.MultiDiGraph,
        n_clusters: int,
        wavelet_scale: float
    ) -> Dict[str, np.ndarray]:
        """
        Analyze a single subgraph at one scale.

        Args:
            subgraph: NetworkX subgraph
            n_clusters: Number of clusters for spectral clustering
            wavelet_scale: Wavelet scale parameter

        Returns:
            Dict with spectral features
        """
        if subgraph.number_of_nodes() == 0:
            return {
                'eigenvalues': np.array([]),
                'wavelets': np.array([]),
                'diffusion': np.array([]),
                'clusters': {}
            }

        features = {}

        try:
            from HoloLoom.warp.spectral_methods import (
                GraphLaplacian,
                SpectralWavelet,
                DiffusionMap,
                spectral_clustering,
                LaplacianType
            )

            # Wrap subgraph for spectral methods
            class _TempKG:
                def __init__(self, G):
                    self.G = G

            temp_kg = _TempKG(subgraph)

            # 1. Laplacian eigenvalues
            laplacian = GraphLaplacian(temp_kg, laplacian_type=LaplacianType.NORMALIZED)
            eigenvalues, _ = laplacian.compute_spectrum(k=min(8, subgraph.number_of_nodes() - 1))
            features['eigenvalues'] = eigenvalues

            # 2. Wavelet coefficients
            wavelet = SpectralWavelet(
                laplacian=laplacian,
                n_scales=1,
                scale_min=wavelet_scale,
                scale_max=wavelet_scale
            )

            # Uniform signal
            n_nodes = subgraph.number_of_nodes()
            signal = np.ones(n_nodes) / n_nodes

            # Heat kernel wavelet
            heat_coeffs = wavelet.transform(signal, kernel='heat')
            features['wavelets'] = heat_coeffs[wavelet_scale]

            # 3. Diffusion map (if graph large enough)
            if subgraph.number_of_nodes() > 10:
                diffusion = DiffusionMap(
                    laplacian=laplacian,
                    t=1.0,
                    n_components=min(8, subgraph.number_of_nodes() - 1)
                )
                features['diffusion'] = diffusion.compute_embedding()
            else:
                features['diffusion'] = np.array([])

            # 4. Spectral clustering (if enough nodes)
            if subgraph.number_of_nodes() >= n_clusters:
                cluster_labels = spectral_clustering(laplacian, n_clusters=n_clusters)
                nodes = list(subgraph.nodes())
                features['clusters'] = {
                    node: int(label) for node, label in zip(nodes, cluster_labels)
                }
            else:
                features['clusters'] = {}

        except Exception as e:
            warnings.warn(f"Multi-scale spectral analysis failed: {e}")
            features = {
                'eigenvalues': np.array([]),
                'wavelets': np.array([]),
                'diffusion': np.array([]),
                'clusters': {}
            }

        return features

    def fuse_multiscale_features(
        self,
        multiscale_results: Dict[int, Dict[str, np.ndarray]],
        fusion_weights: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """
        Fuse multi-scale spectral features into a single vector.

        Args:
            multiscale_results: Results from analyze_multiscale()
            fusion_weights: Optional weights for each scale (default: equal)

        Returns:
            Fused feature vector
        """
        if fusion_weights is None:
            fusion_weights = {scale: 1.0 / len(self.scales) for scale in self.scales}

        # Normalize weights
        total_weight = sum(fusion_weights.values())
        fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}

        fused_parts = []

        for scale in self.scales:
            if scale not in multiscale_results:
                continue

            result = multiscale_results[scale]
            weight = fusion_weights.get(scale, 0.0)

            # Extract eigenvalues (first 4)
            eigenvalues = result.get('eigenvalues', np.array([]))
            if len(eigenvalues) > 0:
                eig_features = eigenvalues[:4]
                # Pad if necessary
                if len(eig_features) < 4:
                    eig_features = np.pad(eig_features, (0, 4 - len(eig_features)))
                fused_parts.append(weight * eig_features)

            # Extract wavelet energy
            wavelets = result.get('wavelets', np.array([]))
            if len(wavelets) > 0:
                wavelet_energy = np.array([np.mean(np.abs(wavelets))])
                fused_parts.append(weight * wavelet_energy)

            # Extract diffusion variance
            diffusion = result.get('diffusion', np.array([]))
            if len(diffusion) > 0:
                diffusion_variance = np.array([np.var(diffusion)])
                fused_parts.append(weight * diffusion_variance)

        if not fused_parts:
            return np.zeros(6)  # Default: 6-dim zero vector

        return np.concatenate(fused_parts)


# ============================================================================
# Hierarchical Spectral Clustering
# ============================================================================

@dataclass
class HierarchicalSpectralClusterer:
    """
    Hierarchical spectral clustering for multi-level graph partitioning.

    Recursively partitions graph using Fiedler vector bisection:
    1. Compute Fiedler vector (2nd smallest Laplacian eigenvector)
    2. Bisect graph at median
    3. Recurse on each partition

    This creates a hierarchical clustering tree, useful for:
    - Coarse-to-fine retrieval
    - Hierarchical knowledge organization
    - Multi-resolution reasoning
    """

    max_depth: int = 3
    min_cluster_size: int = 3

    def cluster_hierarchical(
        self,
        kg: 'KG'
    ) -> Dict[str, List[int]]:
        """
        Perform hierarchical spectral clustering.

        Args:
            kg: Knowledge graph

        Returns:
            Dict mapping entity → cluster_path (list of cluster IDs at each level)
            Example: {'entity1': [0, 1, 3], 'entity2': [0, 1, 2]}
                     Level 0: cluster 0, Level 1: cluster 1, Level 2: cluster 3
        """
        if kg.G.number_of_nodes() < self.min_cluster_size:
            # Too small to cluster
            return {node: [0] for node in kg.G.nodes()}

        # Initialize cluster paths
        cluster_paths = {node: [] for node in kg.G.nodes()}

        # Recursive bisection
        self._recursive_bisect(
            kg,
            list(kg.G.nodes()),
            cluster_paths,
            depth=0,
            parent_cluster_id=0
        )

        return cluster_paths

    def _recursive_bisect(
        self,
        kg: 'KG',
        nodes: List[str],
        cluster_paths: Dict[str, List[int]],
        depth: int,
        parent_cluster_id: int
    ):
        """
        Recursively bisect graph using Fiedler vector.

        Args:
            kg: Knowledge graph
            nodes: Nodes in current partition
            cluster_paths: Accumulator for cluster paths
            depth: Current recursion depth
            parent_cluster_id: Cluster ID at parent level
        """
        # Base case: max depth or too small to split
        if depth >= self.max_depth or len(nodes) < self.min_cluster_size * 2:
            # Assign all nodes to same cluster
            for node in nodes:
                cluster_paths[node].append(parent_cluster_id)
            return

        # Extract subgraph
        subgraph = kg.G.subgraph(nodes).copy()

        try:
            from HoloLoom.warp.spectral_methods import GraphLaplacian, LaplacianType

            # Wrap subgraph
            class _TempKG:
                def __init__(self, G):
                    self.G = G

            temp_kg = _TempKG(subgraph)

            # Compute Fiedler vector
            laplacian = GraphLaplacian(temp_kg, laplacian_type=LaplacianType.NORMALIZED)
            _, eigenvectors = laplacian.compute_spectrum(k=2)
            fiedler_vector = eigenvectors[:, 1]

            # Bisect at median
            median = np.median(fiedler_vector)
            partition_0 = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] < median]
            partition_1 = [nodes[i] for i in range(len(nodes)) if fiedler_vector[i] >= median]

            # Ensure both partitions have minimum size
            if len(partition_0) < self.min_cluster_size or len(partition_1) < self.min_cluster_size:
                # Can't split - assign all to same cluster
                for node in nodes:
                    cluster_paths[node].append(parent_cluster_id)
                return

            # Assign cluster IDs
            cluster_id_0 = parent_cluster_id * 2
            cluster_id_1 = parent_cluster_id * 2 + 1

            for node in partition_0:
                cluster_paths[node].append(cluster_id_0)

            for node in partition_1:
                cluster_paths[node].append(cluster_id_1)

            # Recurse on partitions
            self._recursive_bisect(kg, partition_0, cluster_paths, depth + 1, cluster_id_0)
            self._recursive_bisect(kg, partition_1, cluster_paths, depth + 1, cluster_id_1)

        except Exception as e:
            warnings.warn(f"Hierarchical clustering failed at depth {depth}: {e}")
            # Fallback: assign all to same cluster
            for node in nodes:
                cluster_paths[node].append(parent_cluster_id)


# ============================================================================
# Utility Functions
# ============================================================================

def create_multiscale_analyzer(
    scales: List[int] = [96, 192, 384],
    wavelet_scales: List[float] = [0.1, 1.0, 10.0]
) -> MultiScaleSpectralAnalyzer:
    """
    Factory function to create multi-scale spectral analyzer.

    Args:
        scales: Matryoshka embedding scales
        wavelet_scales: Wavelet scale parameters

    Returns:
        MultiScaleSpectralAnalyzer instance
    """
    return MultiScaleSpectralAnalyzer(
        scales=scales,
        wavelet_scales=wavelet_scales
    )


def create_hierarchical_clusterer(
    max_depth: int = 3,
    min_cluster_size: int = 3
) -> HierarchicalSpectralClusterer:
    """
    Factory function to create hierarchical spectral clusterer.

    Args:
        max_depth: Maximum clustering depth
        min_cluster_size: Minimum nodes per cluster

    Returns:
        HierarchicalSpectralClusterer instance
    """
    return HierarchicalSpectralClusterer(
        max_depth=max_depth,
        min_cluster_size=min_cluster_size
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from HoloLoom.memory.graph import KG, KGEdge

    async def demo():
        print("=== Multi-Scale Spectral Analysis Demo ===\n")

        # Create knowledge graph
        kg = KG()
        edges = [
            # Core concepts
            KGEdge("attention", "transformer", "USES", 1.0),
            KGEdge("transformer", "neural_network", "IS_A", 1.0),
            KGEdge("attention", "neural_network", "PART_OF", 0.8),

            # Variants
            KGEdge("BERT", "transformer", "IS_A", 1.0),
            KGEdge("GPT", "transformer", "IS_A", 1.0),
            KGEdge("T5", "transformer", "IS_A", 1.0),

            # Components
            KGEdge("multi-head attention", "attention", "IS_A", 1.0),
            KGEdge("self-attention", "attention", "IS_A", 1.0),
            KGEdge("cross-attention", "attention", "IS_A", 1.0),

            # Applications
            KGEdge("BERT", "NLP", "USED_IN", 1.0),
            KGEdge("GPT", "NLP", "USED_IN", 1.0),
            KGEdge("T5", "NLP", "USED_IN", 1.0),
        ]
        kg.add_edges(edges)

        print(f"Graph: {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges\n")

        # 1. Multi-scale analysis
        print("1. Multi-Scale Spectral Analysis")
        print("-" * 40)

        analyzer = create_multiscale_analyzer()
        query_entities = ["attention", "transformer"]

        multiscale_results = analyzer.analyze_multiscale(kg, query_entities)

        for scale, result in multiscale_results.items():
            print(f"\nScale: {scale}d")
            print(f"  Eigenvalues: {result['eigenvalues'][:4]}")
            print(f"  Wavelet energy: {np.mean(np.abs(result['wavelets'])) if len(result['wavelets']) > 0 else 0.0:.4f}")
            print(f"  Clusters: {len(set(result['clusters'].values()))} communities")

        # 2. Fused features
        print("\n\n2. Fused Multi-Scale Features")
        print("-" * 40)

        fused = analyzer.fuse_multiscale_features(multiscale_results)
        print(f"Fused feature vector: {fused}")
        print(f"Dimension: {len(fused)}")

        # 3. Hierarchical clustering
        print("\n\n3. Hierarchical Spectral Clustering")
        print("-" * 40)

        clusterer = create_hierarchical_clusterer(max_depth=2, min_cluster_size=2)
        cluster_paths = clusterer.cluster_hierarchical(kg)

        # Print cluster tree
        for entity, path in sorted(cluster_paths.items()):
            print(f"{entity:25s} → {path}")

        print("\n✓ Demo complete!")

    asyncio.run(demo())
