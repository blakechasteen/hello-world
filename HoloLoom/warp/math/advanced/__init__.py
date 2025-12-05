"""Advanced Mathematical Methods module.

Includes:
- Differential Privacy: Privacy-preserving computations
- Topological Data Analysis: Persistent homology for embeddings
"""
from .differential_privacy import (
    DifferentialPrivacy, PrivacyBudget, PrivacyMechanism,
    laplace_mechanism, gaussian_mechanism, exponential_mechanism,
    compose_privacy, compute_sensitivity,
    PrivateAggregator, PrivateVectorMean
)
from .topological_analysis import (
    PersistentHomology, PersistenceDiagram, BettiNumbers,
    compute_persistence, bottleneck_distance, wasserstein_distance,
    EmbeddingTopologyAnalyzer
)

__all__ = [
    # Differential Privacy
    'DifferentialPrivacy', 'PrivacyBudget', 'PrivacyMechanism',
    'laplace_mechanism', 'gaussian_mechanism', 'exponential_mechanism',
    'compose_privacy', 'compute_sensitivity',
    'PrivateAggregator', 'PrivateVectorMean',
    # Topological Data Analysis
    'PersistentHomology', 'PersistenceDiagram', 'BettiNumbers',
    'compute_persistence', 'bottleneck_distance', 'wasserstein_distance',
    'EmbeddingTopologyAnalyzer'
]
