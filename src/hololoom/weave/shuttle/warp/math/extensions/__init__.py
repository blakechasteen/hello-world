"""
Specialized Extensions Module - Advanced Topics Beyond Core Mathematics
======================================================================

Deep specialized topics building on the core mathematical foundation.

Modules:
    advanced_combinatorics: Generating functions, partitions, q-analogs
    multivariable_calculus: Vector calculus, Stokes' theorem, integral theorems
    advanced_curvature: Sectional curvature, Ricci flow, Perelman's work
    hyperbolic_geometry: Poincaré ball, half-space, hyperboloid models

Sprint 7: Specialized Extensions
"""

# Advanced Combinatorics
from .advanced_combinatorics import (
    GeneratingFunction,
    IntegerPartition,
    QAnalogs,
    CatalanNumbers,
    AsymptoticEnumeration,
    SymmetricFunctions,
)

# Multivariable Calculus
from .multivariable_calculus import (
    ScalarField,
    VectorField,
    LineIntegral,
    SurfaceIntegral,
    IntegralTheorems,
    GradientCurlDiv,
)

# Advanced Curvature
from .advanced_curvature import (
    SectionalCurvature,
    ComparisonTheorems,
    RicciFlowAdvanced,
    PerelmanFunctionals,
    GeometricInvariants,
    SpectralGeometry,
)

# Hyperbolic Geometry
from .hyperbolic_geometry import (
    PoincareBall,
    PoincareDisc,
    HalfSpace,
    Hyperboloid,
    HyperbolicGeodesics,
    HyperbolicNeuralNetworks,
)

__all__ = [
    # Advanced Combinatorics
    "GeneratingFunction",
    "IntegerPartition",
    "QAnalogs",
    "CatalanNumbers",
    "AsymptoticEnumeration",
    "SymmetricFunctions",

    # Multivariable Calculus
    "ScalarField",
    "VectorField",
    "LineIntegral",
    "SurfaceIntegral",
    "IntegralTheorems",
    "GradientCurlDiv",

    # Advanced Curvature
    "SectionalCurvature",
    "ComparisonTheorems",
    "RicciFlowAdvanced",
    "PerelmanFunctionals",
    "GeometricInvariants",
    "SpectralGeometry",

    # Hyperbolic Geometry
    "PoincareBall",
    "PoincareDisc",
    "HalfSpace",
    "Hyperboloid",
    "HyperbolicGeodesics",
    "HyperbolicNeuralNetworks",
]
