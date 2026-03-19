"""
Enriched Categories & 2-Categories — Higher Category Theory
=============================================================

Categories where hom-sets carry additional algebraic structure
(enrichment) and 2-categories where morphisms between morphisms
(2-morphisms) exist.

Core Concepts:
- Enriched Category: Hom(A,B) is an object in a monoidal category V, not just a set
- 2-Category: Has objects, 1-morphisms, and 2-morphisms with two compositions
- V-enriched: Composition is a V-morphism Hom(B,C) ⊗ Hom(A,B) → Hom(A,C)

Mathematical Foundation:
Enrichment over V = (V, ⊗, I):
  - For each pair (A,B): Hom(A,B) ∈ Ob(V)
  - Composition: c_{A,B,C}: Hom(B,C) ⊗ Hom(A,B) → Hom(A,C)
  - Identity: j_A: I → Hom(A,A)
  - Associativity and unit coherence diagrams commute

2-Category:
  - Horizontal composition: α ∗ β (along objects)
  - Vertical composition: α · β (along 1-morphisms)
  - Interchange law: (α₁ · α₂) ∗ (β₁ · β₂) = (α₁ ∗ β₁) · (α₂ ∗ β₂)

Applications to Warp Space:
- Enriched categories for weighted/scored morphism composition
- 2-categories for meta-operations on weaving strategies
- Metric-enriched categories for quantitative type theory

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MonoidalCategory:
    """A monoidal category (V, ⊗, I) for enrichment.

    Finite representation: objects are indices, tensor product and
    unit are specified.

    Attributes:
        objects: Set of object indices.
        tensor: (a, b) -> a ⊗ b mapping.
        unit: Unit object index.
    """
    objects: set[int]
    tensor: dict[tuple[int, int], int]
    unit: int

    def tensor_product(self, a: int, b: int) -> int:
        """Compute a ⊗ b."""
        return self.tensor.get((a, b), a)


# ============================================================================
# ENRICHED CATEGORY
# ============================================================================

class EnrichedCategory:
    """Category enriched over a monoidal category V.

    Instead of hom-sets being mere sets, each Hom(A,B) is an object
    in V. Composition is a V-morphism, and identities are V-morphisms
    from the unit.

    This enables categories where morphism spaces have structure:
    - V = ([0,∞], ≥, +, 0): Lawvere metric spaces
    - V = (Bool, ∧, True): Preorders
    - V = (Set, ×, {*}): Ordinary categories
    - V = (Ab, ⊗, Z): Preadditive categories

    Example:
        >>> # Metric-enriched category (generalized metric space)
        >>> V = MonoidalCategory(objects={0}, tensor={(0,0): 0}, unit=0)
        >>> cat = EnrichedCategory(V)
        >>> cat.add_object('A')
        >>> cat.add_object('B')
        >>> cat.set_hom('A', 'B', distance=2.5)
    """

    def __init__(self, enrichment_category: MonoidalCategory | None = None):
        """Initialize enriched category.

        Args:
            enrichment_category: The monoidal category V for enrichment.
                               If None, uses a default Set-like enrichment.
        """
        self.V = enrichment_category
        self._objects: list[str] = []
        self._hom: dict[tuple[str, str], Any] = {}
        self._composition: dict[tuple[str, str, str], Callable] = {}
        self._identities: dict[str, Any] = {}

    def add_object(self, name: str) -> None:
        """Add an object to the category."""
        if name not in self._objects:
            self._objects.append(name)
            # Default identity
            self._hom[(name, name)] = 0.0  # Identity distance
            self._identities[name] = np.eye(1)

    def objects(self) -> list[str]:
        """Return all objects."""
        return list(self._objects)

    def set_hom(self, a: str, b: str, distance: float = 0.0, morphism_data: Any = None) -> None:
        """Set the hom-object Hom(a, b).

        For metric enrichment, this is the distance d(a,b).
        For general enrichment, this is any V-object.

        Args:
            a, b: Source and target objects.
            distance: Metric distance (for metric enrichment).
            morphism_data: Additional morphism data.
        """
        self._hom[(a, b)] = distance if morphism_data is None else morphism_data

    def hom(self, a: str, b: str) -> Any:
        """Get the hom-object Hom(a, b).

        Returns:
            V-object Hom(a,b), or None if not set.
        """
        return self._hom.get((a, b))

    def compose(self, f_data: Any, g_data: Any) -> Any:
        """Compose two morphism data values using V-composition.

        For metric enrichment: d(a,c) ≤ d(a,b) + d(b,c) (triangle inequality).

        Args:
            f_data: Hom(a, b) data.
            g_data: Hom(b, c) data.

        Returns:
            Composed Hom(a, c) data.
        """
        if isinstance(f_data, (int, float)) and isinstance(g_data, (int, float)):
            return f_data + g_data  # Metric composition
        if isinstance(f_data, np.ndarray) and isinstance(g_data, np.ndarray):
            return g_data @ f_data  # Matrix composition
        return (f_data, g_data)

    def identity(self, a: str) -> Any:
        """Get identity morphism j_a: I → Hom(a, a).

        For metric enrichment: d(a,a) = 0.
        """
        return self._identities.get(a, 0.0)

    def verify_triangle_inequality(self) -> list[tuple[str, str, str, float]]:
        """For metric enrichment: check triangle inequality.

        d(a,c) ≤ d(a,b) + d(b,c) for all triples.

        Returns:
            List of (a, b, c, violation) where violation > 0 means failure.
        """
        violations = []
        for a in self._objects:
            for b in self._objects:
                for c in self._objects:
                    d_ac = self._hom.get((a, c))
                    d_ab = self._hom.get((a, b))
                    d_bc = self._hom.get((b, c))
                    if all(isinstance(d, (int, float)) for d in [d_ac, d_ab, d_bc]
                           if d is not None):
                        if d_ac is not None and d_ab is not None and d_bc is not None:
                            violation = d_ac - (d_ab + d_bc)
                            if violation > 1e-10:
                                violations.append((a, b, c, violation))
        return violations

    def underlying_category(self) -> dict[str, Any]:
        """Extract the underlying ordinary category.

        Forgets the enrichment, keeping only whether hom is nonempty.

        Returns:
            Dict with 'objects' and 'morphisms' keys.
        """
        morphisms = []
        for (a, b), data in self._hom.items():
            if data is not None:
                morphisms.append({'source': a, 'target': b})

        return {
            'objects': list(self._objects),
            'morphisms': morphisms,
        }


# ============================================================================
# 2-CATEGORY
# ============================================================================

class TwoCategory:
    """Strict 2-category with objects, 1-morphisms, and 2-morphisms.

    Has two composition operations:
    - Vertical: 2-morphisms α: f→g and β: g→h compose to β·α: f→h
    - Horizontal: 2-morphisms α: f→f' (A→B) and β: g→g' (B→C) compose to β∗α: g∘f→g'∘f'
    The interchange law ensures these are compatible.

    Example:
        >>> C = TwoCategory()
        >>> C.add_object('A')
        >>> C.add_object('B')
        >>> C.add_one_morphism('f', 'A', 'B', data=np.array([1,0]))
        >>> C.add_one_morphism('g', 'A', 'B', data=np.array([0,1]))
        >>> C.add_two_morphism('alpha', 'f', 'g', data=np.array([[0,1],[1,0]]))
    """

    def __init__(self):
        self._objects: set[str] = set()
        self._one_morphisms: dict[str, dict[str, Any]] = {}  # name -> {source, target, data}
        self._two_morphisms: dict[str, dict[str, Any]] = {}  # name -> {source, target, data}

    def add_object(self, name: str) -> None:
        """Add a 0-cell (object)."""
        self._objects.add(name)

    def objects(self) -> list[str]:
        """Return all objects."""
        return sorted(self._objects)

    def add_one_morphism(self, name: str, source: str, target: str, data: Any = None) -> None:
        """Add a 1-cell (1-morphism) between objects.

        Args:
            name: Morphism name.
            source: Source object.
            target: Target object.
            data: Associated data (e.g., matrix).
        """
        self._one_morphisms[name] = {
            'source': source,
            'target': target,
            'data': data,
        }

    def one_morphisms(self, a: str, b: str) -> list[str]:
        """Get all 1-morphisms from a to b."""
        return [
            name for name, info in self._one_morphisms.items()
            if info['source'] == a and info['target'] == b
        ]

    def add_two_morphism(self, name: str, source: str, target: str, data: Any = None) -> None:
        """Add a 2-cell (2-morphism) between 1-morphisms.

        Args:
            name: 2-morphism name.
            source: Source 1-morphism name.
            target: Target 1-morphism name.
            data: Associated data (e.g., natural transformation matrix).
        """
        self._two_morphisms[name] = {
            'source': source,
            'target': target,
            'data': data,
        }

    def two_morphisms(self, f: str, g: str) -> list[str]:
        """Get all 2-morphisms from f to g."""
        return [
            name for name, info in self._two_morphisms.items()
            if info['source'] == f and info['target'] == g
        ]

    def horizontal_compose(self, alpha_name: str, beta_name: str) -> np.ndarray | None:
        """Horizontal composition of 2-morphisms: β ∗ α.

        Given α: f → f' (A→B) and β: g → g' (B→C),
        produces β∗α: g∘f → g'∘f' (A→C).

        Uses Kronecker product for matrix representations.

        Args:
            alpha_name: Name of first 2-morphism (applied first).
            beta_name: Name of second 2-morphism (applied second).

        Returns:
            Composed 2-morphism data, or None if incompatible.
        """
        alpha = self._two_morphisms.get(alpha_name)
        beta = self._two_morphisms.get(beta_name)
        if alpha is None or beta is None:
            return None

        # Check composability: target of alpha's 1-morphisms = source of beta's
        f = self._one_morphisms.get(alpha['source'])
        g = self._one_morphisms.get(beta['source'])
        if f is None or g is None:
            return None
        if f['target'] != g['source']:
            return None

        alpha_data = alpha['data']
        beta_data = beta['data']

        if isinstance(alpha_data, np.ndarray) and isinstance(beta_data, np.ndarray):
            return np.kron(beta_data, alpha_data)
        return None

    def vertical_compose(self, alpha_name: str, beta_name: str) -> np.ndarray | None:
        """Vertical composition of 2-morphisms: β · α.

        Given α: f → g and β: g → h, produces β·α: f → h.

        Uses matrix multiplication for matrix representations.

        Args:
            alpha_name: Name of first 2-morphism (f → g).
            beta_name: Name of second 2-morphism (g → h).

        Returns:
            Composed 2-morphism data, or None if incompatible.
        """
        alpha = self._two_morphisms.get(alpha_name)
        beta = self._two_morphisms.get(beta_name)
        if alpha is None or beta is None:
            return None

        # Check composability: target of alpha = source of beta
        if alpha['target'] != beta['source']:
            return None

        alpha_data = alpha['data']
        beta_data = beta['data']

        if isinstance(alpha_data, np.ndarray) and isinstance(beta_data, np.ndarray):
            return beta_data @ alpha_data
        return None

    def check_interchange_law(
        self,
        alpha1: str, alpha2: str,
        beta1: str, beta2: str,
        tol: float = 1e-10,
    ) -> bool:
        """Verify interchange law: (α₁ · α₂) ∗ (β₁ · β₂) = (α₁ ∗ β₁) · (α₂ ∗ β₂).

        Args:
            alpha1, alpha2: Vertically composable 2-morphisms.
            beta1, beta2: Vertically composable 2-morphisms.
            tol: Numerical tolerance.

        Returns:
            True if interchange law holds.
        """
        # LHS: (α₁ · α₂) ∗ (β₁ · β₂)
        vert_alpha = self.vertical_compose(alpha1, alpha2)
        vert_beta = self.vertical_compose(beta1, beta2)
        if vert_alpha is None or vert_beta is None:
            return False

        lhs = np.kron(vert_beta, vert_alpha)

        # RHS: (α₁ ∗ β₁) · (α₂ ∗ β₂)
        horiz_1 = self.horizontal_compose(alpha1, beta1)
        horiz_2 = self.horizontal_compose(alpha2, beta2)
        if horiz_1 is None or horiz_2 is None:
            return False

        rhs = horiz_2 @ horiz_1

        return bool(np.allclose(lhs, rhs, atol=tol))

    def is_strict(self) -> bool:
        """Check if this is a strict 2-category (composition is strictly associative).

        Returns True by construction since we use matrix arithmetic.
        In a weak 2-category (bicategory), associators would be needed.
        """
        return True

    def nerve(self) -> dict[str, Any]:
        """Compute the simplicial nerve of the underlying 1-category.

        Returns:
            Dict with 0-simplices (objects), 1-simplices (1-morphisms),
            2-simplices (composable pairs).
        """
        zero = sorted(self._objects)
        one = [
            (name, info['source'], info['target'])
            for name, info in self._one_morphisms.items()
        ]
        two = []
        for name_f, info_f in self._one_morphisms.items():
            for name_g, info_g in self._one_morphisms.items():
                if info_f['target'] == info_g['source']:
                    two.append((name_f, name_g, info_f['source'], info_f['target'], info_g['target']))

        return {'0-simplices': zero, '1-simplices': one, '2-simplices': two}
