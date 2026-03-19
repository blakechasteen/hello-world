"""
Universal Algebra for HoloLoom Warp Drive
==========================================

Algebraic structures, lattices, and Birkhoff's HSP theorem.

Universal algebra studies algebraic structures (algebras) in full generality:
a carrier set with operations of specified arities. This abstraction unifies
groups, rings, lattices, Boolean algebras, etc.

Key concepts:
- Algebraic structure: carrier set + operations satisfying identities
- Subalgebra: closed under all operations
- Homomorphic image: preserve operations via surjective map
- Product algebra: componentwise operations
- Variety (HSP class): closed under H, S, P (Birkhoff's theorem)
- Lattice: partial order with meet (∧) and join (∨) for every pair

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product as cartesian_product
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ALGEBRAIC STRUCTURE
# ============================================================================

@dataclass
class Operation:
    """An operation on an algebra: name, arity, and function."""
    name: str
    arity: int
    func: Callable

    def __call__(self, *args):
        if len(args) != self.arity:
            raise ValueError(f"{self.name} expects {self.arity} args, got {len(args)}")
        return self.func(*args)


class AlgebraicStructure:
    """
    Universal algebra (A, F) where A is the carrier set
    and F is a set of operations on A.

    An algebra of type τ = (n₁, n₂, ..., nₖ) has k operations
    with arities n₁, n₂, ..., nₖ.
    """

    def __init__(self, carrier: list, operations: list[Operation],
                 name: str = "A"):
        self.carrier = list(carrier)
        self.operations = {op.name: op for op in operations}
        self.name = name
        self._carrier_set = set(self._freeze(x) for x in carrier)

        logger.info(
            f"AlgebraicStructure {name}: |A|={len(carrier)}, "
            f"ops={[f'{op.name}/{op.arity}' for op in operations]}"
        )

    def _freeze(self, x):
        if isinstance(x, (list, np.ndarray)):
            return tuple(x) if not isinstance(x, np.ndarray) else tuple(x.flat)
        return x

    def is_closed(self) -> bool:
        """Check closure: all operations map back into the carrier."""
        for op in self.operations.values():
            if op.arity == 0:
                result = op.func()
                if self._freeze(result) not in self._carrier_set:
                    return False
            elif op.arity == 1:
                for a in self.carrier:
                    result = op.func(a)
                    if self._freeze(result) not in self._carrier_set:
                        return False
            elif op.arity == 2:
                for a in self.carrier:
                    for b in self.carrier:
                        result = op.func(a, b)
                        if self._freeze(result) not in self._carrier_set:
                            return False
        return True

    def is_subalgebra(self, subset: list) -> bool:
        """
        Check if subset is a subalgebra (closed under all operations).
        """
        sub_set = set(self._freeze(x) for x in subset)

        for op in self.operations.values():
            if op.arity == 0:
                if self._freeze(op.func()) not in sub_set:
                    return False
            elif op.arity == 1:
                for a in subset:
                    if self._freeze(op.func(a)) not in sub_set:
                        return False
            elif op.arity == 2:
                for a in subset:
                    for b in subset:
                        if self._freeze(op.func(a, b)) not in sub_set:
                            return False
        return True

    def subalgebra_generated_by(self, generators: list) -> list:
        """
        Generate the smallest subalgebra containing the generators.

        Repeatedly apply all operations until closure.
        """
        sub = set(self._freeze(g) for g in generators)
        # Map frozen -> original for operation application
        frozen_to_orig = {self._freeze(x): x for x in self.carrier}

        changed = True
        max_iter = 1000
        iteration = 0

        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            new_elements = set(sub)

            elements = [frozen_to_orig[f] for f in sub if f in frozen_to_orig]

            for op in self.operations.values():
                if op.arity == 0:
                    r = self._freeze(op.func())
                    if r not in sub:
                        new_elements.add(r)
                        changed = True
                elif op.arity == 1:
                    for a in elements:
                        r = self._freeze(op.func(a))
                        if r not in sub and r in self._carrier_set:
                            new_elements.add(r)
                            changed = True
                elif op.arity == 2:
                    for a in elements:
                        for b in elements:
                            r = self._freeze(op.func(a, b))
                            if r not in sub and r in self._carrier_set:
                                new_elements.add(r)
                                changed = True

            sub = new_elements

        return [frozen_to_orig[f] for f in sub if f in frozen_to_orig]

    def homomorphic_image(self, h: Callable) -> 'AlgebraicStructure':
        """
        Compute the homomorphic image h(A).

        h must preserve all operations: h(f(a₁,...,aₙ)) = f(h(a₁),...,h(aₙ)).
        """
        image_carrier = list(set(self._freeze(h(a)) for a in self.carrier))
        # Reconstruct from frozen
        image_elements = []
        seen = set()
        for a in self.carrier:
            ha = h(a)
            key = self._freeze(ha)
            if key not in seen:
                seen.add(key)
                image_elements.append(ha)

        new_ops = []
        for op in self.operations.values():
            if op.arity == 2:
                def make_op(original_op):
                    def new_func(a, b):
                        # Find preimages and apply
                        return original_op.func(a, b)
                    return new_func
                new_ops.append(Operation(op.name, op.arity, make_op(op)))
            else:
                new_ops.append(op)

        return AlgebraicStructure(image_elements, new_ops, f"h({self.name})")

    def congruence_classes(self, congruence: Callable[[Any, Any], bool]) -> list[list]:
        """
        Partition carrier into equivalence classes of a congruence relation.

        A congruence is an equivalence relation compatible with all operations.
        """
        classes = []
        assigned = set()

        for a in self.carrier:
            a_key = self._freeze(a)
            if a_key in assigned:
                continue

            cls = [a]
            assigned.add(a_key)

            for b in self.carrier:
                b_key = self._freeze(b)
                if b_key not in assigned and congruence(a, b):
                    cls.append(b)
                    assigned.add(b_key)

            classes.append(cls)

        return classes


# ============================================================================
# LATTICE
# ============================================================================

class Lattice:
    """
    Lattice (L, ∧, ∨): partially ordered set where every pair has
    a meet (greatest lower bound) and join (least upper bound).

    Properties to check:
    - Modular: a ≤ c ⟹ a ∨ (b ∧ c) = (a ∨ b) ∧ c
    - Distributive: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
    - Complemented: every element has a complement (a ∧ a' = 0, a ∨ a' = 1)
    - Boolean: complemented + distributive

    Note: distributive ⟹ modular. Complemented + distributive = Boolean algebra.
    """

    def __init__(self, elements: list, meet: Callable, join: Callable,
                 name: str = "L"):
        """
        Args:
            elements: Elements of the lattice.
            meet: Binary operation ∧ (greatest lower bound).
            join: Binary operation ∨ (least upper bound).
        """
        self.elements = elements
        self.meet = meet
        self.join = join
        self.name = name

        self._bottom = None
        self._top = None
        self._find_bounds()

        logger.info(f"Lattice {name}: |L|={len(elements)}")

    def _freeze(self, x):
        if isinstance(x, (list, np.ndarray)):
            return tuple(x) if not isinstance(x, np.ndarray) else tuple(x.flat)
        return x

    def _find_bounds(self):
        """Find bottom (0) and top (1) if they exist."""
        for a in self.elements:
            is_bottom = all(self.meet(a, b) == a or self._freeze(self.meet(a, b)) == self._freeze(a)
                            for b in self.elements)
            if is_bottom:
                self._bottom = a

            is_top = all(self.join(a, b) == a or self._freeze(self.join(a, b)) == self._freeze(a)
                         for b in self.elements)
            if is_top:
                self._top = a

    def leq(self, a, b) -> bool:
        """Partial order: a ≤ b ⟺ a ∧ b = a."""
        return self._freeze(self.meet(a, b)) == self._freeze(a)

    def is_modular(self) -> bool:
        """
        Check modularity: for all a ≤ c, a ∨ (b ∧ c) = (a ∨ b) ∧ c.

        Equivalent to: no sublattice isomorphic to N₅ (pentagon).
        """
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if not self.leq(a, c):
                        continue
                    lhs = self.join(a, self.meet(b, c))
                    rhs = self.meet(self.join(a, b), c)
                    if self._freeze(lhs) != self._freeze(rhs):
                        return False
        return True

    def is_distributive(self) -> bool:
        """
        Check distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c) for all a, b, c.

        Equivalent to: no sublattice isomorphic to M₃ (diamond) or N₅ (pentagon).
        """
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    lhs = self.meet(a, self.join(b, c))
                    rhs = self.join(self.meet(a, b), self.meet(a, c))
                    if self._freeze(lhs) != self._freeze(rhs):
                        return False
        return True

    def is_complemented(self) -> bool:
        """
        Check if every element has a complement.

        a' is a complement of a if a ∧ a' = 0 and a ∨ a' = 1.
        Requires bounded lattice (0 and 1 exist).
        """
        if self._bottom is None or self._top is None:
            return False

        bottom = self._freeze(self._bottom)
        top = self._freeze(self._top)

        for a in self.elements:
            has_complement = False
            for b in self.elements:
                m = self._freeze(self.meet(a, b))
                j = self._freeze(self.join(a, b))
                if m == bottom and j == top:
                    has_complement = True
                    break
            if not has_complement:
                return False
        return True

    def is_boolean(self) -> bool:
        """Boolean algebra = complemented distributive lattice."""
        return self.is_complemented() and self.is_distributive()

    def complement(self, a) -> Any | None:
        """Find complement of a, if it exists."""
        if self._bottom is None or self._top is None:
            return None

        bottom = self._freeze(self._bottom)
        top = self._freeze(self._top)

        for b in self.elements:
            if (self._freeze(self.meet(a, b)) == bottom and
                    self._freeze(self.join(a, b)) == top):
                return b
        return None

    def atoms(self) -> list:
        """
        Atoms: elements that cover the bottom element.

        a is an atom if 0 < a and there is no b with 0 < b < a.
        """
        if self._bottom is None:
            return []

        result = []
        bottom = self._freeze(self._bottom)

        for a in self.elements:
            a_key = self._freeze(a)
            if a_key == bottom:
                continue
            if not self.leq(self._bottom, a):
                continue

            is_atom = True
            for b in self.elements:
                b_key = self._freeze(b)
                if b_key == bottom or b_key == a_key:
                    continue
                if self.leq(self._bottom, b) and self.leq(b, a):
                    is_atom = False
                    break

            if is_atom:
                result.append(a)

        return result

    def interval(self, a, b) -> list:
        """Interval [a, b] = {x : a ≤ x ≤ b}."""
        return [x for x in self.elements if self.leq(a, x) and self.leq(x, b)]

    @staticmethod
    def power_set(n: int) -> 'Lattice':
        """
        Power set lattice P({0, 1, ..., n-1}).

        Meet = intersection, Join = union. Boolean algebra of 2ⁿ elements.
        """
        elements = []
        for i in range(2 ** n):
            elements.append(frozenset(j for j in range(n) if i & (1 << j)))

        return Lattice(
            elements,
            meet=lambda a, b: a & b,
            join=lambda a, b: a | b,
            name=f"P({n})"
        )

    @staticmethod
    def divisor_lattice(n: int) -> 'Lattice':
        """
        Divisor lattice of n: elements are divisors of n.

        Meet = gcd, Join = lcm.
        """
        from math import gcd

        divisors = [d for d in range(1, n + 1) if n % d == 0]

        def lcm(a, b):
            return a * b // gcd(a, b)

        return Lattice(divisors, meet=gcd, join=lcm, name=f"Div({n})")


# ============================================================================
# BIRKHOFF HSP THEOREM
# ============================================================================

class BirkhoffHSP:
    """
    Birkhoff's HSP Theorem: A class of algebras is a variety
    (definable by identities) iff it is closed under H, S, P.

    H = Homomorphic images
    S = Subalgebras
    P = Products

    Given an algebra A, the variety generated by A is HSP({A}).
    """

    @staticmethod
    def H(algebra: AlgebraicStructure,
          congruence: Callable[[Any, Any], bool]) -> AlgebraicStructure:
        """
        Homomorphic image via a congruence relation.

        The quotient algebra A/θ where θ is a congruence.
        """
        classes = algebra.congruence_classes(congruence)
        representatives = [cls[0] for cls in classes]

        # Map each element to its class representative
        elem_to_rep = {}
        for cls in classes:
            rep = algebra._freeze(cls[0])
            for x in cls:
                elem_to_rep[algebra._freeze(x)] = rep

        new_ops = []
        for op in algebra.operations.values():
            if op.arity == 2:
                def make_op(original_op, e2r):
                    def new_func(a, b):
                        result = original_op.func(a, b)
                        # Map to representative
                        for cls_list in classes:
                            for x in cls_list:
                                if algebra._freeze(x) == algebra._freeze(result):
                                    return cls_list[0]
                        return result
                    return new_func
                new_ops.append(Operation(op.name, op.arity, make_op(op, elem_to_rep)))
            elif op.arity == 1:
                def make_unary(original_op):
                    def new_func(a):
                        result = original_op.func(a)
                        for cls_list in classes:
                            for x in cls_list:
                                if algebra._freeze(x) == algebra._freeze(result):
                                    return cls_list[0]
                        return result
                    return new_func
                new_ops.append(Operation(op.name, op.arity, make_unary(op)))
            else:
                new_ops.append(op)

        return AlgebraicStructure(representatives, new_ops, f"{algebra.name}/θ")

    @staticmethod
    def S(algebra: AlgebraicStructure, subset: list) -> AlgebraicStructure:
        """
        Subalgebra on a given subset (must be closed under all operations).
        """
        if not algebra.is_subalgebra(subset):
            raise ValueError("Subset is not closed under operations")

        sub_ops = []
        for op in algebra.operations.values():
            sub_ops.append(Operation(op.name, op.arity, op.func))

        return AlgebraicStructure(subset, sub_ops, f"Sub({algebra.name})")

    @staticmethod
    def P(algebras: list[AlgebraicStructure]) -> AlgebraicStructure:
        """
        Direct product of algebras.

        Carrier = A₁ × A₂ × ... × Aₙ (Cartesian product).
        Operations applied componentwise.
        """
        if not algebras:
            raise ValueError("Need at least one algebra")

        # Carrier = Cartesian product
        carriers = [a.carrier for a in algebras]
        product_carrier = list(cartesian_product(*carriers))

        # Operations applied componentwise
        op_names = list(algebras[0].operations.keys())
        product_ops = []

        for op_name in op_names:
            arity = algebras[0].operations[op_name].arity

            if arity == 2:
                def make_prod_op(name, algs):
                    def prod_func(a_tuple, b_tuple):
                        return tuple(
                            algs[i].operations[name].func(a_tuple[i], b_tuple[i])
                            for i in range(len(algs))
                        )
                    return prod_func
                product_ops.append(Operation(op_name, arity, make_prod_op(op_name, algebras)))

            elif arity == 1:
                def make_prod_unary(name, algs):
                    def prod_func(a_tuple):
                        return tuple(
                            algs[i].operations[name].func(a_tuple[i])
                            for i in range(len(algs))
                        )
                    return prod_func
                product_ops.append(Operation(op_name, arity, make_prod_unary(op_name, algebras)))

            elif arity == 0:
                def make_prod_nullary(name, algs):
                    def prod_func():
                        return tuple(algs[i].operations[name].func() for i in range(len(algs)))
                    return prod_func
                product_ops.append(Operation(op_name, arity, make_prod_nullary(op_name, algebras)))

        names = " × ".join(a.name for a in algebras)
        return AlgebraicStructure(product_carrier, product_ops, names)


# ============================================================================
# CLONE THEORY (AL8.4)
# ============================================================================

class Clone:
    """
    Clone: a set of finitary operations on a set A, closed under
    composition and containing all projections.

    A clone on A is determined by its operations f: A^n -> A for various n.
    Clones form a lattice under inclusion. Post's lattice classifies all
    clones on {0, 1} (countably many, forming a complex lattice).

    Key properties:
    - Contains all projections: π_i^n(x_1,...,x_n) = x_i
    - Closed under composition: if f ∈ Clone and g_1,...,g_n ∈ Clone,
      then f(g_1,...,g_n) ∈ Clone
    """

    def __init__(self, base_set: list, operations: list[Callable] = None):
        """
        Args:
            base_set: The underlying set A
            operations: Initial generating operations
        """
        self.base_set = list(base_set)
        self.n = len(self.base_set)
        self.operations: list[tuple[int, Callable]] = []  # (arity, function)

        if operations:
            for op in operations:
                self.add_operation(op)

    def add_operation(self, op: Callable, arity: int = None) -> None:
        """Add an operation to the clone."""
        if arity is None:
            # Try to infer arity
            import inspect
            try:
                sig = inspect.signature(op)
                arity = len(sig.parameters)
            except (ValueError, TypeError):
                arity = 1
        self.operations.append((arity, op))

    def projection(self, n: int, i: int) -> Callable:
        """Return the i-th projection of arity n: π_i^n(x_1,...,x_n) = x_i."""
        def proj(*args):
            return args[i]
        return proj

    def has_projections(self) -> bool:
        """Check if the clone contains all projections (up to tested arity)."""
        max_arity = max((a for a, _ in self.operations), default=1)
        for n in range(1, max_arity + 1):
            for i in range(n):
                proj = self.projection(n, i)
                # Check if an equivalent operation exists
                found = False
                for arity, op in self.operations:
                    if arity == n:
                        # Test on all tuples
                        all_match = True
                        import itertools
                        for args in itertools.product(self.base_set, repeat=n):
                            if op(*args) != args[i]:
                                all_match = False
                                break
                        if all_match:
                            found = True
                            break
                if not found:
                    return False
        return True

    def compose(self, f: Callable, f_arity: int,
                gs: list[Callable], g_arity: int) -> Callable:
        """Compose f with g_1,...,g_n: h(x) = f(g_1(x),...,g_n(x))."""
        def composed(*args):
            g_results = tuple(g(*args) for g in gs)
            return f(*g_results)
        return composed

    def is_closed_under_composition(self, max_test_arity: int = 2) -> bool:
        """Check if the clone is closed under composition (sampled)."""
        import itertools
        for f_arity, f in self.operations:
            for g_arity, g in self.operations:
                if f_arity == 1:
                    # f(g(x_1,...,x_k))
                    composed = lambda *args, _f=f, _g=g: _f(_g(*args))
                    # Check if result is in clone (by testing on all inputs)
                    for args in itertools.product(self.base_set, repeat=g_arity):
                        result = composed(*args)
                        if result not in self.base_set:
                            return False
        return True

    def generate(self, generators: list[Callable], arities: list[int],
                 max_depth: int = 3) -> 'Clone':
        """Generate a clone from a set of operations by closure."""
        clone = Clone(self.base_set)
        # Add generators
        for op, arity in zip(generators, arities):
            clone.add_operation(op, arity)
        # Add all projections
        max_arity = max(arities) if arities else 1
        for n in range(1, max_arity + 1):
            for i in range(n):
                clone.add_operation(clone.projection(n, i), n)
        return clone

    @staticmethod
    def posts_lattice_info() -> dict:
        """Information about Post's lattice (clones on {0, 1}).

        Post (1941) classified all clones on a 2-element set.
        There are countably infinitely many, forming a lattice.
        """
        return {
            "description": "Post's lattice classifies all clones on {0, 1}",
            "n_clones": "countably infinite",
            "key_clones": {
                "T_0": "operations preserving 0",
                "T_1": "operations preserving 1",
                "S": "self-dual operations: f(not x) = not f(x)",
                "M": "monotone operations",
                "L": "affine/linear operations (XOR + constants)",
            },
            "top": "all operations on {0, 1}",
            "bottom": "projections only",
            "sheffer_functions": "NAND and NOR generate the top clone",
        }
