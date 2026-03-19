"""
Computational Mathematics for HoloLoom Warp Drive
===================================================

Matrix factorizations, polynomial algorithms, number theory,
symbolic computation, automatic differentiation, approximation
algorithms, and Krylov methods.

Pure numpy implementations of fundamental computational mathematics.
"""

from .approximation_algorithms import (
    FPTAS,
    MinHash,
    RandomizedRounding,
    Sketching,
)
from .automatic_differentiation import (
    JVP,
    VJP,
    DualNumber,
    ForwardModeAD,
    ReverseModeAD,
)
from .krylov_methods import (
    GMRES,
    BiCGSTAB,
    Multigrid,
)
from .matrix_factorizations import (
    ArnoldiIteration,
    CholeskyDecomposition,
    LanczosIteration,
    LUDecomposition,
)
from .number_theory_algorithms import (
    DiscreteLogarithm,
    EllipticCurve,
    IntegerFactorization,
    LLLReduction,
    PrimalityTesting,
)
from .polynomial_algorithms import (
    FiniteFieldFactorization,
    FiniteFieldPoly,
    GrobnerComputation,
    HermiteNormalForm,
    PolynomialGCD,
    SmithNormalForm,
)
from .symbolic_computation import (
    PatternMatcher,
    RewriteSystem,
    SymbolicExpr,
)

__all__ = [
    # Matrix Factorizations
    "LUDecomposition", "CholeskyDecomposition", "LanczosIteration", "ArnoldiIteration",
    # Polynomial Algorithms
    "GrobnerComputation", "PolynomialGCD", "SmithNormalForm", "HermiteNormalForm",
    "FiniteFieldPoly", "FiniteFieldFactorization",
    # Number Theory
    "PrimalityTesting", "IntegerFactorization", "DiscreteLogarithm",
    "LLLReduction", "EllipticCurve",
    # Symbolic Computation
    "SymbolicExpr", "RewriteSystem", "PatternMatcher",
    # Automatic Differentiation
    "DualNumber", "ForwardModeAD", "ReverseModeAD", "JVP", "VJP",
    # Approximation Algorithms
    "RandomizedRounding", "FPTAS", "Sketching", "MinHash",
    # Krylov Methods
    "GMRES", "BiCGSTAB", "Multigrid",
]
