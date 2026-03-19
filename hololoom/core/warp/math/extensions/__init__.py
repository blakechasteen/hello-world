"""
Specialized Extensions Module - Advanced Topics Beyond Core Mathematics
======================================================================

Deep specialized topics building on the core mathematical foundation.

Modules:
    advanced_combinatorics: Generating functions, partitions, q-analogs
    multivariable_calculus: Vector calculus, Stokes' theorem, integral theorems
    advanced_curvature: Sectional curvature, Ricci flow, Perelman's work
    hyperbolic_geometry: Poincaré ball, half-space, hyperboloid models

    cellular_automata: Elementary CA, Game of Life, Langton parameter
    reservoir_computing: Echo State Networks, Conceptors

Sprint 7: Specialized Extensions
"""

# Advanced Combinatorics
from .advanced_combinatorics import (
    AsymptoticEnumeration,
    CatalanNumbers,
    GeneratingFunction,
    IntegerPartition,
    QAnalogs,
    SymmetricFunctions,
)

# Advanced Curvature
from .advanced_curvature import (
    ComparisonTheorems,
    GeometricInvariants,
    PerelmanFunctionals,
    RicciFlowAdvanced,
    SectionalCurvature,
    SpectralGeometry,
)

# Algebraic Combinatorics
from .algebraic_combinatorics import (
    DiscreteMorseInequalities,
    HallLittlewood,
    RSKCorrespondence,
    SchurFunction,
    YoungTableauxOps,
)

# Autopoiesis
from .autopoiesis import (
    AutopoiesisAnalyzer,
    AutopoieticSystem,
    Component,
    StructuralCoupling,
)

# Cellular Automata
from .cellular_automata import (
    ElementaryCA,
    GameOfLife,
    LangtonParameter,
)

# Combinatorial Numbers
from .combinatorial_numbers import (
    BellNumbers,
    ExponentialFormula,
    Species,
    StirlingNumbers,
    TransferMatrix,
)

# Complexity & Cybernetics
from .complexity_cybernetics import (
    ApproximationAlgorithm,
    IntuitionisticLogic,
    KripkeFrame,
    ParameterizedComplexity,
    TypeTheory,
    ViabilityTheory,
)

# Extremal Combinatorics
from .extremal_combinatorics import (
    LovaszLocalLemma,
    SpernerTheorem,
    SzemerediRegularity,
    TuranTheorem,
)

# Advanced Graph Theory
from .graph_theory_advanced import (
    MatroidTheory,
    RamseyTheory,
    TuttePolynomial,
)

# Hyperbolic Geometry
from .hyperbolic_geometry import (
    HalfSpace,
    HyperbolicGeodesics,
    HyperbolicNeuralNetworks,
    Hyperboloid,
    PoincareBall,
    PoincareDisc,
)

# Manifold Learning
from .manifold_learning import (
    TSNE,
    UMAP,
    Isomap,
    LocallyLinearEmbedding,
    ManifoldEmbedding,
    trustworthiness,
)

# Multivariable Calculus
from .multivariable_calculus import (
    GradientCurlDiv,
    IntegralTheorems,
    LineIntegral,
    ScalarField,
    SurfaceIntegral,
    VectorField,
)

# Reservoir Computing
from .reservoir_computing import (
    Conceptor,
    EchoStateNetwork,
)

# Second-Order Cybernetics
from .second_order_cybernetics import (
    CircularCausality,
    CyberneticState,
    Observer,
    RequisiteVariety,
    SecondOrderObserver,
    analyze_cybernetic_system,
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

    # Manifold Learning
    "ManifoldEmbedding",
    "TSNE",
    "UMAP",
    "Isomap",
    "LocallyLinearEmbedding",
    "trustworthiness",

    # Second-Order Cybernetics
    "Observer",
    "CyberneticState",
    "SecondOrderObserver",
    "RequisiteVariety",
    "CircularCausality",
    "analyze_cybernetic_system",

    # Autopoiesis
    "Component",
    "AutopoieticSystem",
    "AutopoiesisAnalyzer",
    "StructuralCoupling",

    # Cellular Automata
    "ElementaryCA",
    "GameOfLife",
    "LangtonParameter",

    # Reservoir Computing
    "EchoStateNetwork",
    "Conceptor",

    # Combinatorial Numbers
    "BellNumbers",
    "StirlingNumbers",
    "Species",
    "TransferMatrix",
    "ExponentialFormula",

    # Algebraic Combinatorics
    "YoungTableauxOps",
    "RSKCorrespondence",
    "SchurFunction",
    "HallLittlewood",
    "DiscreteMorseInequalities",

    # Extremal Combinatorics
    "TuranTheorem",
    "SzemerediRegularity",
    "SpernerTheorem",
    "LovaszLocalLemma",

    # Advanced Graph Theory
    "RamseyTheory",
    "MatroidTheory",
    "TuttePolynomial",

    # Complexity & Cybernetics
    "ApproximationAlgorithm",
    "ParameterizedComplexity",
    "KripkeFrame",
    "IntuitionisticLogic",
    "TypeTheory",
    "ViabilityTheory",
]
