"""
Algebra - Symmetry and Algebraic Structures
===========================================

Abstract Algebra: Groups, rings, fields, morphisms
Galois Theory: Field extensions, Galois groups, solvability
Module Theory: Modules over rings, tensor products, exact sequences
Homological Algebra: Chain complexes, derived functors, Ext/Tor

Complete algebraic foundation for symmetry analysis in AI.
"""

from .abstract_algebra import (
    Group,
    GroupHomomorphism,
    Ring,
    Ideal,
    Field,
    Polynomial
)

from .galois_theory import (
    FieldExtension,
    MinimalPolynomial,
    GaloisGroup,
    FundamentalTheoremGalois,
    SolvabilityByRadicals,
    ClassicalImpossibilities,
    FiniteFieldTheory
)

from .module_theory import (
    Module,
    ModuleHomomorphism,
    TensorProduct,
    ExactSequence,
    ProjectiveModule
)

from .homological_algebra import (
    ChainComplex,
    QuotientModule,
    CochainComplex,
    DerivedFunctors,
    LongExactSequence,
    SpectralSequence,
    HomologicalDimension
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
    "LongExactSequence", "SpectralSequence", "HomologicalDimension"
]
