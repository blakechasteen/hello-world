"""
HoloLoom Mathematical Modules
==============================

Advanced mathematics for rigorous analysis and computation.

Modules:
- analysis: Real, complex, measure, functional analysis + statistics
- algebra: (coming soon) Abstract algebra, Lie algebras
- geometry: (coming soon) Differential geometry, algebraic geometry
- decision: Game theory, operations research, RL, optimal control
- self_assembly: Genetic algorithms, swarm optimization
"""

# Analysis modules
try:
    from .analysis.real_analysis import (
        ContinuityChecker,
        Differentiator,
        MetricSpace,
        RiemannIntegrator,
        SequenceAnalyzer,
    )
    HAS_REAL_ANALYSIS = True
except ImportError:
    HAS_REAL_ANALYSIS = False

try:
    from .analysis.complex_analysis import (
        AnalyticContinuation,
        ComplexFunction,
        ConformalMapper,
        ContourIntegrator,
        ResidueCalculator,
        SeriesExpansion,
    )
    HAS_COMPLEX_ANALYSIS = True
except ImportError:
    HAS_COMPLEX_ANALYSIS = False

__all__ = []

if HAS_REAL_ANALYSIS:
    __all__.extend([
        "MetricSpace",
        "SequenceAnalyzer",
        "ContinuityChecker",
        "Differentiator",
        "RiemannIntegrator"
    ])

if HAS_COMPLEX_ANALYSIS:
    __all__.extend([
        "ComplexFunction",
        "ContourIntegrator",
        "ResidueCalculator",
        "ConformalMapper",
        "SeriesExpansion",
        "AnalyticContinuation"
    ])

# Self-Assembly modules
try:
    from .self_assembly.genetic_algorithms import (
        DifferentialEvolution,
        GeneticAlgorithm,
        Individual,
    )
    from .self_assembly.swarm import (
        AntColonyOptimization,
        ParticleSwarmOptimization,
        stigmergy_update,
    )
    HAS_SELF_ASSEMBLY = True
except ImportError:
    HAS_SELF_ASSEMBLY = False

if HAS_SELF_ASSEMBLY:
    __all__.extend([
        "Individual",
        "GeneticAlgorithm",
        "DifferentialEvolution",
        "ParticleSwarmOptimization",
        "AntColonyOptimization",
        "stigmergy_update",
    ])

# Algebra modules
try:
    from .algebra import (
        AlgebraicStructure,
        AlgebraicVariety,
        BurnsideLemma,
        CharacterTable,
        ClassGroup,
        DynkinDiagram,
        Field,
        FreeGroup,
        GrobnerBasis,
        GrothendieckGroup,
        Group,
        GroupHomomorphism,
        GroupPresentation,
        GroupRepresentation,
        Ideal,
        Lattice,
        LieAlgebra,
        NumberField,
        Polynomial,
        QuotientGroup,
        QuotientRing,
        Ring,
        RootSystem,
        SchurLemma,
        VectorBundle,
        YoungTableaux,
    )
    HAS_ALGEBRA = True
except ImportError:
    HAS_ALGEBRA = False

if HAS_ALGEBRA:
    __all__.extend([
        "Group", "GroupHomomorphism", "Ring", "Ideal", "Field", "Polynomial",
        "QuotientGroup", "QuotientRing", "FreeGroup", "GroupPresentation",
        "LieAlgebra", "RootSystem", "DynkinDiagram",
        "GroupRepresentation", "CharacterTable", "SchurLemma", "BurnsideLemma",
        "YoungTableaux", "GrobnerBasis", "AlgebraicVariety",
        "NumberField", "ClassGroup", "AlgebraicStructure", "Lattice",
        "GrothendieckGroup", "VectorBundle",
    ])

# Computational Mathematics modules
try:
    from .computational import (
        FPTAS,
        GMRES,
        JVP,
        VJP,
        ArnoldiIteration,
        BiCGSTAB,
        CholeskyDecomposition,
        DiscreteLogarithm,
        DualNumber,
        EllipticCurve,
        ForwardModeAD,
        GrobnerComputation,
        HermiteNormalForm,
        IntegerFactorization,
        LanczosIteration,
        LLLReduction,
        LUDecomposition,
        MinHash,
        Multigrid,
        PatternMatcher,
        PolynomialGCD,
        PrimalityTesting,
        RandomizedRounding,
        ReverseModeAD,
        RewriteSystem,
        Sketching,
        SmithNormalForm,
        SymbolicExpr,
    )
    HAS_COMPUTATIONAL = True
except ImportError:
    HAS_COMPUTATIONAL = False

if HAS_COMPUTATIONAL:
    __all__.extend([
        "LUDecomposition", "CholeskyDecomposition", "LanczosIteration",
        "ArnoldiIteration", "GrobnerComputation", "PolynomialGCD",
        "SmithNormalForm", "HermiteNormalForm",
        "PrimalityTesting", "IntegerFactorization", "DiscreteLogarithm",
        "LLLReduction", "EllipticCurve",
        "SymbolicExpr", "RewriteSystem", "PatternMatcher",
        "DualNumber", "ForwardModeAD", "ReverseModeAD", "JVP", "VJP",
        "RandomizedRounding", "FPTAS", "Sketching", "MinHash",
        "GMRES", "BiCGSTAB", "Multigrid",
    ])
