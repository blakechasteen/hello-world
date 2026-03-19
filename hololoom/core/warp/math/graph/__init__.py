"""Graph Theory — Spectral Methods, Coloring, and Combinatorics."""
from .graph_coloring import (
    ColoringResult,
    ConflictGraph,
    GraphColoring,
    chromatic_polynomial,
    color_graph,
    independent_sets,
    maximum_independent_set,
    schedule_tasks,
)
from .spectral_clustering import (
    GraphLaplacian,
    KnowledgeGraphClusterer,
    LaplacianType,
    SpectralClustering,
    algebraic_connectivity,
    cluster_graph,
    compute_laplacian,
    fiedler_vector,
    spectral_embed,
)

__all__ = [
    'GraphLaplacian', 'LaplacianType', 'SpectralClustering',
    'compute_laplacian', 'spectral_embed', 'cluster_graph',
    'fiedler_vector', 'algebraic_connectivity',
    'KnowledgeGraphClusterer',
    # Graph Coloring
    'ColoringResult',
    'GraphColoring',
    'ConflictGraph',
    'chromatic_polynomial',
    'independent_sets',
    'maximum_independent_set',
    'color_graph',
    'schedule_tasks',
]
