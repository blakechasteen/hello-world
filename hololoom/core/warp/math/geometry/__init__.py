"""
Geometry Module - Differential, Riemannian, Symplectic, Information Geometry
=============================================================================

Complete geometric framework for curved spaces, manifolds, and physics.

Modules:
    differential_geometry: Manifolds, tangent bundles, differential forms
    riemannian_geometry: Metrics, curvature, geodesics, Ricci flow
    mathematical_physics: Lagrangian/Hamiltonian mechanics, symplectic geometry
    information_geometry: Fisher metric, α-connections, natural gradient, Bregman divergences
"""

# Differential Geometry
from .differential_geometry import (
    Chart,
    SmoothManifold,
    TangentSpace,
    TangentVector,
    TangentBundle,
    VectorField,
    DifferentialForm,
    ExteriorCalculus,
    LieDerivative,
)

# Riemannian Geometry
from .riemannian_geometry import (
    RiemannianMetric,
    Christoffel,
    Geodesic,
    RiemannCurvature,
    CurvatureAnalysis,
    RicciFlow,
    ParallelTransport,
)

# Mathematical Physics
from .mathematical_physics import (
    LagrangianMechanics,
    HamiltonianMechanics,
    SymplecticManifold,
    PoissonBracket,
    CanonicalTransformation,
    NoetherTheorem,
    GaugeTheory,
)

# Information Geometry
from .information_geometry import (
    FisherMetric,
    AlphaConnection,
    BregmanDivergence,
    NaturalGradient,
    StatisticalManifold,
    StatisticalGeodesic,
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
]
