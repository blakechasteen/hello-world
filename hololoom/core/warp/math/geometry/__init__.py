"""
Geometry Module - Differential, Riemannian, Symplectic, Information Geometry
=============================================================================

Complete geometric framework for curved spaces, manifolds, and physics.

Modules:
    differential_geometry: Manifolds, tangent bundles, differential forms
    riemannian_geometry: Metrics, curvature, geodesics, Ricci flow
    mathematical_physics: Lagrangian/Hamiltonian mechanics, symplectic geometry
    information_geometry: Fisher metric, α-connections, natural gradient, Bregman divergences
    symplectic_completion: Moment maps, symplectic reduction, cotangent bundles
"""

# Differential Geometry
# Algebraic Geometry
from .algebraic_geometry import (
    AffineVariety,
    IntersectionTheory,
    MultiPoly,
    ProjectiveVariety,
    TropicalGeometry,
)
from .algebraic_geometry import (
    Monomial as AGMonomial,
)

# Computational Geometry
from .computational_geometry import (
    ConvexHull,
    DelaunayTriangulation,
    KDTree,
    Triangle,
    VoronoiDiagram,
    VoronoiResult,
)
from .differential_geometry import (
    Chart,
    DifferentialForm,
    ExteriorCalculus,
    LieDerivative,
    SmoothManifold,
    TangentBundle,
    TangentSpace,
    TangentVector,
    VectorField,
)

# Information Geometry
from .information_geometry import (
    AlphaConnection,
    BregmanDivergence,
    FisherMetric,
    NaturalGradient,
    StatisticalGeodesic,
    StatisticalManifold,
)

# Mathematical Physics
from .mathematical_physics import (
    CanonicalTransformation,
    GaugeTheory,
    HamiltonianMechanics,
    LagrangianMechanics,
    NoetherTheorem,
    PoissonBracket,
    SymplecticManifold,
)

# Riemannian Geometry
from .riemannian_geometry import (
    Christoffel,
    CurvatureAnalysis,
    Geodesic,
    ParallelTransport,
    RicciFlow,
    RiemannCurvature,
    RiemannianMetric,
)

# Symplectic Completion
from .symplectic_completion import (
    CotangentBundle,
    MomentMap,
    SymplecticForm,
    SymplecticReduction,
)

__all__ = [
    # Differential Geometry
    "Chart",
    "SmoothManifold",
    "TangentSpace",
    "TangentVector",
    "TangentBundle",
    "VectorField",
    "DifferentialForm",
    "ExteriorCalculus",
    "LieDerivative",

    # Riemannian Geometry
    "RiemannianMetric",
    "Christoffel",
    "Geodesic",
    "RiemannCurvature",
    "CurvatureAnalysis",
    "RicciFlow",
    "ParallelTransport",

    # Mathematical Physics
    "LagrangianMechanics",
    "HamiltonianMechanics",
    "SymplecticManifold",
    "PoissonBracket",
    "CanonicalTransformation",
    "NoetherTheorem",
    "GaugeTheory",

    # Information Geometry
    "FisherMetric",
    "AlphaConnection",
    "BregmanDivergence",
    "NaturalGradient",
    "StatisticalManifold",
    "StatisticalGeodesic",

    # Symplectic Completion
    "SymplecticForm",
    "MomentMap",
    "SymplecticReduction",
    "CotangentBundle",

    # Algebraic Geometry
    "AffineVariety", "ProjectiveVariety",
    "TropicalGeometry", "IntersectionTheory",
    "MultiPoly", "AGMonomial",

    # Computational Geometry
    "VoronoiDiagram", "VoronoiResult",
    "DelaunayTriangulation", "Triangle",
    "ConvexHull", "KDTree",
]
