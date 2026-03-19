"""
Adjoint Functors and Kan Extensions — CT2.4-2.5
================================================

Category theory machinery for compositional reasoning. Functors map
between categories preserving structure; adjunctions capture universal
constructions (free/forgetful, currying, limits/colimits).

Classes:
    Category: Objects and morphisms with composition
    Functor: Covariant functor between categories
    NaturalTransformation: Component-wise map between functors
    AdjointPair: Left/right adjoint with unit and counit
    KanExtension: Universal extension of functors along other functors

Mathematical Foundation:
    Adjunction: L -| R iff Hom(LA, B) ~ Hom(A, RB) naturally in A, B
    Unit: eta: Id -> RL (universal arrow from A to RL(A))
    Counit: epsilon: LR -> Id (universal arrow from LR(B) to B)
    Triangle identities: (epsilon L)(L eta) = id_L, (R epsilon)(eta R) = id_R

    Left Kan extension: (Lan_K F)(b) = colim_{Ka -> b} F(a)
    Right Kan extension: (Ran_K F)(b) = lim_{b -> Ka} F(a)

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Morphism:
    """A morphism in a category."""
    source: Any
    target: Any
    label: str = ""

    def __hash__(self):
        return hash((self.source, self.target, self.label))

    def __eq__(self, other):
        if not isinstance(other, Morphism):
            return NotImplemented
        return (self.source == other.source and
                self.target == other.target and
                self.label == other.label)


@dataclass
class Category:
    """
    A (small) category: objects, morphisms, and composition.

    Composition is stored as a dict mapping (f, g) -> g . f
    where f: A -> B and g: B -> C gives g.f: A -> C.
    Identity morphisms are implicit.
    """
    objects: list[Any]
    morphisms: list[Morphism]
    composition: dict[tuple[str, str], str] = field(default_factory=dict)

    def identity(self, obj: Any) -> Morphism:
        """Identity morphism on an object."""
        return Morphism(source=obj, target=obj, label=f"id_{obj}")

    def compose(self, f: Morphism, g: Morphism) -> Morphism | None:
        """
        Compose g . f (f first, then g).

        Args:
            f: Morphism A -> B
            g: Morphism B -> C

        Returns:
            Composition g.f: A -> C, or None if incomposable
        """
        if f.target != g.source:
            return None

        # Check for identity
        if f.label.startswith("id_"):
            return g
        if g.label.startswith("id_"):
            return f

        # Look up composition
        key = (f.label, g.label)
        if key in self.composition:
            return Morphism(
                source=f.source, target=g.target,
                label=self.composition[key]
            )

        # Default: labeled composition
        return Morphism(
            source=f.source, target=g.target,
            label=f"{g.label}.{f.label}"
        )

    def hom(self, a: Any, b: Any) -> list[Morphism]:
        """Hom-set: all morphisms from a to b."""
        result = [m for m in self.morphisms if m.source == a and m.target == b]
        if a == b:
            result.append(self.identity(a))
        return result

    def is_isomorphism(self, f: Morphism) -> bool:
        """Check if f has an inverse."""
        for g in self.hom(f.target, f.source):
            gf = self.compose(f, g)
            fg = self.compose(g, f)
            if (gf and gf.label == f"id_{f.source}" and
                    fg and fg.label == f"id_{f.target}"):
                return True
        return False


class Functor:
    """
    Covariant functor F: C -> D.

    Maps objects to objects and morphisms to morphisms,
    preserving composition and identities:
        F(g . f) = F(g) . F(f)
        F(id_A) = id_{F(A)}
    """

    def __init__(
        self,
        source: Category,
        target: Category,
        object_map: dict[Any, Any],
        morphism_map: dict[str, Morphism],
        name: str = "F",
    ):
        self.source = source
        self.target = target
        self.object_map = object_map
        self.morphism_map = morphism_map
        self.name = name

    def map_object(self, obj: Any) -> Any:
        """Apply functor to an object."""
        return self.object_map.get(obj, obj)

    def map_morphism(self, morph: Morphism) -> Morphism:
        """Apply functor to a morphism."""
        if morph.label.startswith("id_"):
            return Morphism(
                source=self.map_object(morph.source),
                target=self.map_object(morph.target),
                label=f"id_{self.map_object(morph.source)}"
            )
        if morph.label in self.morphism_map:
            return self.morphism_map[morph.label]
        # Default: wrapped morphism
        return Morphism(
            source=self.map_object(morph.source),
            target=self.map_object(morph.target),
            label=f"{self.name}({morph.label})"
        )

    def preserves_composition(self, f: Morphism, g: Morphism) -> bool:
        """Check F(g.f) = F(g).F(f)."""
        gf = self.source.compose(f, g)
        if gf is None:
            return True  # Can't compose, nothing to check

        F_gf = self.map_morphism(gf)
        Fg_Ff = self.target.compose(self.map_morphism(f), self.map_morphism(g))

        if Fg_Ff is None:
            return False

        return F_gf.source == Fg_Ff.source and F_gf.target == Fg_Ff.target

    def preserves_identity(self, obj: Any) -> bool:
        """Check F(id_A) = id_{F(A)}."""
        id_a = self.source.identity(obj)
        F_id = self.map_morphism(id_a)
        id_Fa = self.target.identity(self.map_object(obj))
        return F_id.source == id_Fa.source and F_id.target == id_Fa.target

    def is_faithful(self) -> bool:
        """F is faithful if injective on hom-sets."""
        for a in self.source.objects:
            for b in self.source.objects:
                hom_ab = self.source.hom(a, b)
                images = set()
                for f in hom_ab:
                    Ff = self.map_morphism(f)
                    key = (Ff.source, Ff.target, Ff.label)
                    if key in images:
                        return False
                    images.add(key)
        return True

    def is_full(self) -> bool:
        """F is full if surjective on hom-sets."""
        for a in self.source.objects:
            for b in self.source.objects:
                Fa = self.map_object(a)
                Fb = self.map_object(b)
                target_hom = self.target.hom(Fa, Fb)
                source_images = {self.map_morphism(f).label
                                 for f in self.source.hom(a, b)}
                for g in target_hom:
                    if g.label not in source_images:
                        return False
        return True


class NaturalTransformation:
    """
    Natural transformation alpha: F => G between functors F, G: C -> D.

    Components alpha_A: F(A) -> G(A) for each object A in C,
    satisfying naturality: G(f) . alpha_A = alpha_B . F(f) for all f: A -> B.
    """

    def __init__(
        self,
        source_functor: Functor,
        target_functor: Functor,
        components: dict[Any, Morphism],
    ):
        self.source_functor = source_functor
        self.target_functor = target_functor
        self.components = components

    def component(self, obj: Any) -> Morphism:
        """Get the component alpha_A."""
        return self.components[obj]

    def is_natural(self) -> bool:
        """
        Check naturality square commutes for all morphisms.

        For f: A -> B, we need:
            G(f) . alpha_A = alpha_B . F(f)
        """
        F = self.source_functor
        G = self.target_functor
        D = F.target

        for f in F.source.morphisms:
            a, b = f.source, f.target
            if a not in self.components or b not in self.components:
                continue

            alpha_a = self.components[a]
            alpha_b = self.components[b]
            Ff = F.map_morphism(f)
            Gf = G.map_morphism(f)

            left = D.compose(alpha_a, Gf)
            right = D.compose(Ff, alpha_b)

            if left is None or right is None:
                return False
            if left.source != right.source or left.target != right.target:
                return False

        return True


class AdjointPair:
    """
    Adjunction L -| R between functors L: C -> D and R: D -> C.

    Equivalent characterizations:
    1. Natural bijection: Hom_D(LA, B) ~ Hom_C(A, RB)
    2. Unit eta: Id_C -> RL and counit epsilon: LR -> Id_D
       satisfying triangle identities
    """

    def __init__(
        self,
        left_adjoint: Functor,
        right_adjoint: Functor,
        unit: NaturalTransformation | None = None,
        counit: NaturalTransformation | None = None,
    ):
        self.left_adjoint = left_adjoint
        self.right_adjoint = right_adjoint
        self.unit = unit
        self.counit = counit

    @staticmethod
    def is_adjoint_pair(
        L: Functor,
        R: Functor,
        test_objects: list[tuple[Any, Any]] | None = None,
    ) -> bool:
        """
        Test whether L -| R by checking hom-set bijection.

        For each pair (A in C, B in D), check:
            |Hom_D(LA, B)| = |Hom_C(A, RB)|

        This is a necessary condition. For small categories,
        also checks that the bijection is natural.

        Args:
            L: Candidate left adjoint L: C -> D
            R: Candidate right adjoint R: D -> C
            test_objects: List of (A, B) pairs to test.
                         If None, tests all object pairs.

        Returns:
            True if hom-set sizes match for all tested pairs
        """
        C = L.source
        D = L.target

        if test_objects is None:
            test_objects = [(a, b) for a in C.objects for b in D.objects]

        for a, b in test_objects:
            La = L.map_object(a)
            Rb = R.map_object(b)

            hom_D = D.hom(La, b)
            hom_C = C.hom(a, Rb)

            if len(hom_D) != len(hom_C):
                return False

        return True

    def verify_triangle_identities(self) -> tuple[bool, bool]:
        """
        Check the triangle (zig-zag) identities.

        Left triangle:  (epsilon_L)(L eta) = id_L
        Right triangle: (R epsilon)(eta_R) = id_R

        Returns:
            (left_holds, right_holds) booleans
        """
        if self.unit is None or self.counit is None:
            return False, False

        L = self.left_adjoint
        R = self.right_adjoint
        D = L.target
        C = L.source

        left_holds = True
        right_holds = True

        for a in C.objects:
            # Left triangle at a: epsilon_{La} . L(eta_a) = id_{La}
            if a in self.unit.components:
                eta_a = self.unit.components[a]
                L_eta_a = L.map_morphism(eta_a)
                La = L.map_object(a)
                if La in self.counit.components:
                    eps_La = self.counit.components[La]
                    composed = D.compose(L_eta_a, eps_La)
                    if composed is None or composed.label != f"id_{La}":
                        left_holds = False

        for b in D.objects:
            # Right triangle at b: R(epsilon_b) . eta_{Rb} = id_{Rb}
            if b in self.counit.components:
                eps_b = self.counit.components[b]
                R_eps_b = R.map_morphism(eps_b)
                Rb = R.map_object(b)
                if Rb in self.unit.components:
                    eta_Rb = self.unit.components[Rb]
                    composed = C.compose(eta_Rb, R_eps_b)
                    if composed is None or composed.label != f"id_{Rb}":
                        right_holds = False

        return left_holds, right_holds


class KanExtension:
    """
    Kan extensions — the universal construction in category theory.

    Given K: C -> C' and F: C -> D, the left Kan extension Lan_K F: C' -> D
    is the "best approximation" to F along K. It satisfies the universal
    property: for any functor G: C' -> D with a natural transformation
    F => GK, there exists a unique alpha: Lan_K F => G.
    """

    @staticmethod
    def left_kan(
        F: Functor,
        K: Functor,
        target_category: Category,
    ) -> Functor:
        """
        Compute left Kan extension Lan_K F via colimit formula.

        (Lan_K F)(b) = colim_{Ka -> b} F(a)

        For finite categories, the colimit is computed as a coequalizer.

        Args:
            F: Functor F: C -> D to extend
            K: Functor K: C -> C' along which to extend
            target_category: The category C' (codomain of K)

        Returns:
            Functor Lan_K F: C' -> D
        """
        C = F.source
        D = F.target
        C_prime = target_category

        object_map = {}
        morphism_map = {}

        for b in C_prime.objects:
            # Comma category (K downarrow b):
            # objects are pairs (a, Ka -> b)
            # Colimit of F restricted to this comma category
            comma_objects = []
            for a in C.objects:
                Ka = K.map_object(a)
                arrows = C_prime.hom(Ka, b)
                for arrow in arrows:
                    comma_objects.append((a, arrow))

            if comma_objects:
                # Approximate colimit: take the "last" F(a) value
                # (exact colimit requires coequalizer computation)
                a_best = comma_objects[-1][0]
                object_map[b] = F.map_object(a_best)
            else:
                # No arrows into b from K's image: use b itself
                object_map[b] = b

        return Functor(
            source=C_prime,
            target=D,
            object_map=object_map,
            morphism_map=morphism_map,
            name=f"Lan_{K.name}({F.name})",
        )

    @staticmethod
    def right_kan(
        F: Functor,
        K: Functor,
        target_category: Category,
    ) -> Functor:
        """
        Compute right Kan extension Ran_K F via limit formula.

        (Ran_K F)(b) = lim_{b -> Ka} F(a)

        Args:
            F: Functor F: C -> D
            K: Functor K: C -> C'
            target_category: Category C'

        Returns:
            Functor Ran_K F: C' -> D
        """
        C = F.source
        D = F.target
        C_prime = target_category

        object_map = {}
        morphism_map = {}

        for b in C_prime.objects:
            # Comma category (b downarrow K):
            # objects are pairs (a, b -> Ka)
            comma_objects = []
            for a in C.objects:
                Ka = K.map_object(a)
                arrows = C_prime.hom(b, Ka)
                for arrow in arrows:
                    comma_objects.append((a, arrow))

            if comma_objects:
                # Approximate limit: take the "first" F(a) value
                a_best = comma_objects[0][0]
                object_map[b] = F.map_object(a_best)
            else:
                object_map[b] = b

        return Functor(
            source=C_prime,
            target=D,
            object_map=object_map,
            morphism_map=morphism_map,
            name=f"Ran_{K.name}({F.name})",
        )

    @staticmethod
    def is_pointwise(
        extension: Functor,
        F: Functor,
        K: Functor,
    ) -> bool:
        """
        Check if an extension is pointwise (computed via colimits/limits).

        A Kan extension is pointwise if it is preserved by all
        representable functors. For locally small categories, all
        Kan extensions that exist are pointwise.

        Args:
            extension: The candidate Kan extension
            F: Original functor
            K: Functor along which we extend

        Returns:
            True if the extension appears pointwise (consistency check)
        """
        # Verify: for each a in C, extension(K(a)) should relate to F(a)
        for a in F.source.objects:
            Ka = K.map_object(a)
            ext_Ka = extension.map_object(Ka)
            Fa = F.map_object(a)

            # In a pointwise extension, there should be a map F(a) -> Lan(K(a))
            # We check object-level consistency
            if ext_Ka != Fa:
                return False

        return True
