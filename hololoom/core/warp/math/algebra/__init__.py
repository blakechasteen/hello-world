"""
Algebra - Symmetry and Algebraic Structures
============================================

Abstract Algebra: Groups, rings, fields, morphisms
Galois Theory: Field extensions, Galois groups, solvability
Module Theory: Modules over rings, tensor products, exact sequences
Homological Algebra: Chain complexes, derived functors, Ext/Tor
Adjoint Functors: Category theory, adjunctions, Kan extensions
Quotient Structures: Quotient groups/rings, free groups, presentations
Lie Algebras: Bracket, Killing form, root systems, Dynkin diagrams
Representation Theory: Characters, Schur, Maschke, Young tableaux
Commutative Algebra: Noetherian rings, Groebner bases, varieties
Universal Algebra: Algebraic structures, lattices, Birkhoff HSP
Algebraic Number Theory: Number fields, class groups
K-Theory: Grothendieck group, Whitehead group, vector bundles

Complete algebraic foundation for symmetry analysis in AI.
"""

from .abstract_algebra import Field, Group, GroupHomomorphism, Ideal, Polynomial, Ring
from .adjoint_functors import (
    AdjointPair,
    Category,
    Functor,
    KanExtension,
    Morphism,
    NaturalTransformation,
)
from .algebraic_number_theory import (
    AlgebraicInteger,
    ClassGroup,
    NumberField,
    RingOfIntegers,
)
from .commutative_algebra import (
    AlgebraicVariety,
    Fraction,
    GrobnerBasis,
    Localization,
    Monomial,
    MVPolynomial,
    NoetherianRing,
    PrimaryDecomposition,
)
from .enriched_categories import (
    EnrichedCategory,
    MonoidalCategory,
    TwoCategory,
)
from .galois_theory import (
    ClassicalImpossibilities,
    FieldExtension,
    FiniteFieldTheory,
    FundamentalTheoremGalois,
    GaloisGroup,
    MinimalPolynomial,
    SolvabilityByRadicals,
)
from .homological_algebra import (
    ChainComplex,
    CochainComplex,
    DerivedFunctors,
    HomologicalDimension,
    LongExactSequence,
    QuotientModule,
    SpectralSequence,
)
from .k_theory import (
    GrothendieckGroup,
    VectorBundle,
    WhiteheadGroup,
)
from .lie_algebras import (
    DynkinDiagram,
    LieAlgebra,
    RootSystem,
    UniversalEnveloping,
)
from .matrix_functions import (
    SparseMatrix,
    generalized_eigenvalue,
    matrix_exponential,
    matrix_logarithm,
)
from .module_theory import (
    ExactSequence,
    Module,
    ModuleHomomorphism,
    ProjectiveModule,
    TensorProduct,
)
from .quotient_structures import (
    Coset,
    FreeGroup,
    GroupPresentation,
    Letter,
    QuotientGroup,
    QuotientRing,
)
from .representation_theory import (
    BurnsideLemma,
    CharacterTable,
    GroupRepresentation,
    MaschkeTheorem,
    SchurLemma,
    YoungTableaux,
)
from .tensor_decomposition import (
    TensorNetwork,
    TensorTrain,
    TensorTrainResult,
    TuckerDecomposition,
    TuckerResult,
)
from .toposes import (
    Box,
    CategoricalDatabase,
    CategoricalSchema,
    MarkovCategory,
    Operad,
    StringDiagram,
    SubobjectClassifier,
    Topos,
)
from .toposes import (
    Operation as OperadicOperation,
)
from .universal_algebra import (
    AlgebraicStructure,
    BirkhoffHSP,
    Lattice,
    Operation,
)

__all__ = [
    # Abstract Algebra
    "Group", "GroupHomomorphism", "Ring", "Ideal", "Field", "Polynomial",
    # Galois Theory
    "FieldExtension", "MinimalPolynomial", "GaloisGroup", "FundamentalTheoremGalois",
    "SolvabilityByRadicals", "ClassicalImpossibilities", "FiniteFieldTheory",
    # Module Theory
    "Module", "ModuleHomomorphism", "TensorProduct", "ExactSequence", "ProjectiveModule",
    # Homological Algebra
    "ChainComplex", "QuotientModule", "CochainComplex", "DerivedFunctors",
    "LongExactSequence", "SpectralSequence", "HomologicalDimension",
    # Adjoint Functors & Category Theory
    "Morphism", "Category", "Functor", "NaturalTransformation",
    "AdjointPair", "KanExtension",
    # Quotient Structures
    "Coset", "QuotientGroup", "QuotientRing", "FreeGroup", "Letter",
    "GroupPresentation",
    # Lie Algebras
    "LieAlgebra", "RootSystem", "DynkinDiagram", "UniversalEnveloping",
    # Representation Theory
    "GroupRepresentation", "CharacterTable", "SchurLemma", "MaschkeTheorem",
    "BurnsideLemma", "YoungTableaux",
    # Commutative Algebra
    "Monomial", "MVPolynomial", "NoetherianRing", "Localization", "Fraction",
    "PrimaryDecomposition", "GrobnerBasis", "AlgebraicVariety",
    # Universal Algebra
    "Operation", "AlgebraicStructure", "Lattice", "BirkhoffHSP",
    # Algebraic Number Theory
    "NumberField", "AlgebraicInteger", "RingOfIntegers", "ClassGroup",
    # K-Theory
    "GrothendieckGroup", "WhiteheadGroup", "VectorBundle",
    # Enriched Categories & 2-Categories
    "MonoidalCategory", "EnrichedCategory", "TwoCategory",
    # Toposes, Operads, String Diagrams, Markov Categories
    "SubobjectClassifier", "Topos", "OperadicOperation", "Operad",
    "Box", "StringDiagram", "MarkovCategory",
    "CategoricalSchema", "CategoricalDatabase",
    # Matrix Functions
    "matrix_exponential", "matrix_logarithm", "SparseMatrix",
    "generalized_eigenvalue",
    # Tensor Decomposition
    "TuckerResult", "TuckerDecomposition",
    "TensorTrainResult", "TensorTrain", "TensorNetwork",
]
