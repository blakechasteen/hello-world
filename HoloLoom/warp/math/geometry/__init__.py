"""
Geometry Module - Differential Geometry, Riemannian Geometry, Mathematical Physics
==================================================================================

Complete geometric framework for curved spaces, manifolds, and physics.

Modules:
    differential_geometry: Manifolds, tangent bundles, differential forms
    riemannian_geometry: Metrics, curvature, geodesics, Ricci flow
    mathematical_physics: Lagrangian/Hamiltonian mechanics, symplectic geometry

Sprint 4: Geometry & Physics
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
]
