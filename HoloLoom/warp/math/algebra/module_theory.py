"""
Module Theory for HoloLoom Warp Drive
=====================================

Modules over rings - generalized vector spaces.

Core Concepts:
- Modules: Like vector spaces but over rings (not just fields)
- Module Homomorphisms: R-linear maps
- Tensor Products: M ⊗_R N - universal bilinear construction
- Exact Sequences: 0 → A → B → C → 0
- Free Modules: R^n - modules with basis
- Projective/Injective Modules: Important for homological algebra

Mathematical Foundation:
R-module M: Abelian group with R-action r·m satisfying:
- (r + s)·m = r·m + s·m
- r·(m + n) = r·m + r·n
- (rs)·m = r·(s·m)
- 1·m = m

Tensor product M ⊗_R N satisfies universal property:
For bilinear β: M × N → P, ∃! φ: M ⊗_R N → P with β = φ ∘ ⊗

Applications to Warp Space:
- Knowledge graph embeddings as modules
- Tensor operations on semantic spaces
- Exact sequences for data transformations
- Projective resolutions for computing derived functors

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Any
from dataclasses import dataclass
from .abstract_algebra import Ring
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MODULES
# ============================================================================

class Module:
    """
    R-module: Abelian group with ring action.

    Generalizes vector spaces (modules over fields).
    """

    def __init__(
        self,
        ring: Ring,
        elements: List[Any],
        addition: Callable[[Any, Any], Any],
        scalar_mult: Callable[[Any, Any], Any],
        zero: Any,
        name: str = "M"
    ):
        """Initialize R-module."""
        self.ring = ring
        self.elements = elements
        self.add = addition
        self.smul = scalar_mult  # Scalar multiplication
        self.zero = zero
        self.name = name

        logger.info(f"Module {name} over {ring.name} with {len(elements)} elements")

    def is_submodule(self, subset: List[Any]) -> bool:
        """Check if subset is a submodule."""
        # Closed under addition and scalar multiplication
        for a in subset:
            for b in subset:
                if self.add(a, b) not in subset:
                    return False

            for r in self.ring.elements:
                if self.smul(r, a) not in subset:
                    return False

        return True

    @staticmethod
    def free_module(ring: Ring, rank: int) -> 'Module':
        """
        Free R-module R^n of rank n.

        Elements: n-tuples (r₁, ..., rₙ) with rᵢ ∈ R
        """
        # Simplified for finite rings
        if len(ring.elements) > 100:
            logger.warning("Large ring - using sample")
            elements = []
        else:
            # All n-tuples
            from itertools import product
            elements = list(product(ring.elements, repeat=rank))

        def add_tuples(a, b):
            return tuple(ring.add(ai, bi) for ai, bi in zip(a, b))

        def scalar_mult_tuple(r, a):
            return tuple(ring.mul(r, ai) for ai in a)

        zero = tuple(ring.zero for _ in range(rank))

        return Module(ring, elements, add_tuples, scalar_mult_tuple, zero, f"{ring.name}^{rank}")


@dataclass
class ModuleHomomorphism:
    """
    R-module homomorphism φ: M → N.

    R-linear: φ(r·m) = r·φ(m), φ(m + n) = φ(m) + φ(n)
    """
    source: Module
    target: Module
    mapping: Callable[[Any], Any]
    name: str = "φ"

    def __call__(self, element: Any) -> Any:
        """Apply homomorphism."""
        return self.mapping(element)

    def kernel(self) -> List[Any]:
        """ker(φ) = {m ∈ M : φ(m) = 0}"""
        return [m for m in self.source.elements if self.mapping(m) == self.target.zero]

    def image(self) -> List[Any]:
        """im(φ) = {φ(m) : m ∈ M}"""
        return list(set(self.mapping(m) for m in self.source.elements))

    def is_injective(self) -> bool:
        """Injective ⟺ ker(φ) = {0}"""
        return len(self.kernel()) == 1  # Only zero

    def is_surjective(self) -> bool:
        """Surjective ⟺ im(φ) = N"""
        return len(self.image()) == len(self.target.elements)


# ============================================================================
# TENSOR PRODUCTS
# ============================================================================

class TensorProduct:
    """
    Tensor product M ⊗_R N.

    Universal bilinear construction.
    """

    @staticmethod
    def construct(M: Module, N: Module) -> Module:
        """
        Construct M ⊗_R N.

        Elements: Formal sums Σ (mᵢ ⊗ nᵢ) modulo bilinearity relations.
        """
        R = M.ring

        # For free modules R^m ⊗ R^n ≅ R^{mn}
        # Simplified implementation

        logger.info(f"Constructing {M.name} ⊗ {N.name}")

        # Placeholder: would need quotient by bilinearity relations
        return Module.free_module(R, 1)

    @staticmethod
    def universal_property(
        M: Module,
        N: Module,
        bilinear_map: Callable[[Any, Any], Any],
        target: Module
    ) -> ModuleHomomorphism:
        """
        Universal property: bilinear β: M × N → P factors through M ⊗ N.

        ∃! φ: M ⊗ N → P with β(m,n) = φ(m ⊗ n)
        """
        tensor_prod = TensorProduct.construct(M, N)

        # Define induced map
        def induced(tensor_elem):
            # Would decompose tensor_elem = Σ mᵢ ⊗ nᵢ
            # and return Σ β(mᵢ, nᵢ)
            return target.zero  # Placeholder

        return ModuleHomomorphism(tensor_prod, target, induced, "β̃")


# ============================================================================
# EXACT SEQUENCES
# ============================================================================

class ExactSequence:
    """
    Exact sequence of modules.

    ... → Mᵢ₋₁ →^{fᵢ₋₁} Mᵢ →^{fᵢ} Mᵢ₊₁ → ...

    Exact at Mᵢ: im(fᵢ₋₁) = ker(fᵢ)
    """

    def __init__(self, modules: List[Module], homomorphisms: List[ModuleHomomorphism]):
        """Initialize exact sequence."""
        self.modules = modules
        self.homomorphisms = homomorphisms

    def is_exact_at(self, index: int) -> bool:
        """
        Check exactness at Mᵢ.

        Requires im(fᵢ₋₁) = ker(fᵢ)
        """
        if index == 0 or index >= len(self.modules):
            return True

        f_prev = self.homomorphisms[index - 1]
        f_curr = self.homomorphisms[index]

        image_prev = set(f_prev.image())
        kernel_curr = set(f_curr.kernel())

        return image_prev == kernel_curr

    def is_exact(self) -> bool:
        """Check if sequence is exact at all modules."""
        for i in range(len(self.modules)):
            if not self.is_exact_at(i):
                return False
        return True

    def is_short_exact(self) -> bool:
        """
        Check if sequence is short exact: 0 → A → B → C → 0

        Requires: f injective, g surjective, im(f) = ker(g)
        """
        if len(self.modules) != 3:
            return False

        f = self.homomorphisms[0]
        g = self.homomorphisms[1]

        return f.is_injective() and g.is_surjective() and self.is_exact_at(1)


# ============================================================================
# PROJECTIVE & INJECTIVE MODULES
# ============================================================================

class ProjectiveModule:
    """
    Projective R-module.

    P is projective if every surjection M → P splits.
    Equivalently: P is direct summand of free module.
    """

    @staticmethod
    def is_projective(module: Module) -> bool:
        """
        Check if module is projective.

        Over fields: all modules are projective (= free).
        Over general rings: complicated to check.
        """
        # Free modules are projective
        if module.name.endswith("^"):
            return True

        # General test would require checking lifting property
        return False  # Conservative

    @staticmethod
    def projective_resolution(module: Module, length: int = 3) -> ExactSequence:
        """
        Projective resolution of module.

        ... → P₂ → P₁ → P₀ → M → 0

        with each Pᵢ projective.
        """
        R = module.ring

        # Construct projective modules (free modules)
        projectives = [Module.free_module(R, i+1) for i in range(length)]
        projectives.append(module)

        # Define homomorphisms (would need actual construction)
        homs = []
        for i in range(length):
            hom = ModuleHomomorphism(
                projectives[i],
                projectives[i+1],
                lambda x: module.zero,  # Placeholder
                f"d_{i}"
            )
            homs.append(hom)

        logger.info(f"Constructed projective resolution of length {length}")

        return ExactSequence(projectives, homs)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Module',
    'ModuleHomomorphism',
    'TensorProduct',
    'ExactSequence',
    'ProjectiveModule'
]
