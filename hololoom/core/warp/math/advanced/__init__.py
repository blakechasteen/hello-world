"""Advanced Mathematical Methods module.

Includes:
- Differential Privacy: Privacy-preserving computations
- Topological Data Analysis: Persistent homology for embeddings
- Covering Spaces: Fundamental group, path lifting, higher homotopy, CW complexes
- Extended Persistence: Extended & zigzag persistent homology
- Alpha & Cech Complexes: Geometric simplicial complexes, TDA pipeline
- Sheaf Extensions: Sheaf morphisms, derived cohomology
- Quotient & Product Spaces: Topological space constructions
"""
from .alpha_cech_complexes import (
    AlphaComplex,
    CechComplex,
    PersistencePair,
    SimplexTree,
    TDAPipeline,
    TDAResult,
)
from .covering_spaces import (
    CoveringSpace,
    CWComplex,
    FundamentalGroup,
    HigherHomotopy,
    SimplicialComplex,
)
from .differential_privacy import (
    DifferentialPrivacy,
    PrivacyBudget,
    PrivacyMechanism,
    PrivateAggregator,
    PrivateVectorMean,
    compose_privacy,
    compute_sensitivity,
    exponential_mechanism,
    gaussian_mechanism,
    laplace_mechanism,
)
from .extended_persistence import (
    ExtendedDiagram,
    ExtendedPersistence,
    PersistenceInterval,
    ZigzagDiagram,
    ZigzagPersistence,
)
from .quotient_product_spaces import (
    CoproductTopology,
    ProductTopology,
    QuotientSpace,
    TopologicalSpace,
)
from .sheaf_extensions import (
    DerivedSheafCohomology,
    Sheaf,
    SheafMorphism,
)
from .topological_analysis import (
    BettiNumbers,
    EmbeddingTopologyAnalyzer,
    PersistenceDiagram,
    PersistentHomology,
    bottleneck_distance,
    compute_persistence,
    wasserstein_distance,
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
    'EmbeddingTopologyAnalyzer',
    # Covering Spaces & Homotopy
    'SimplicialComplex', 'FundamentalGroup', 'CoveringSpace',
    'HigherHomotopy', 'CWComplex',
    # Extended Persistence
    'PersistenceInterval', 'ExtendedDiagram', 'ZigzagDiagram',
    'ExtendedPersistence', 'ZigzagPersistence',
    # Alpha & Cech Complexes
    'SimplexTree', 'PersistencePair', 'TDAResult',
    'AlphaComplex', 'CechComplex', 'TDAPipeline',
    # Sheaf Extensions
    'Sheaf', 'SheafMorphism', 'DerivedSheafCohomology',
    # Quotient & Product Spaces
    'TopologicalSpace', 'QuotientSpace', 'ProductTopology', 'CoproductTopology',
]
