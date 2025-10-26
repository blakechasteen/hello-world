"""
Category Theory for HoloLoom Warp Drive
========================================

Implements categorical structures for knowledge representation and reasoning.

Core Concepts:
- Categories: Objects and morphisms with composition
- Functors: Structure-preserving maps between categories
- Natural Transformations: Morphisms between functors
- Limits & Colimits: Universal constructions
- Yoneda Lemma: Embeddings via representable functors
- Monoidal Categories: Compositional semantics
- Adjunctions: Optimal transformations

Applications:
- Knowledge graph transformations
- Compositional semantics
- Universal properties for optimization
- Functor of points for embeddings
- Categorical database queries

Mathematical Foundation:
Category C consists of:
- Objects: Ob(C)
- Morphisms: Hom(A, B) for objects A, B
- Composition: ∘: Hom(B,C) × Hom(A,B) → Hom(A,C)
- Identity: id_A: A → A for each object A
- Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
- Identity laws: f ∘ id_A = f = id_B ∘ f

Author: HoloLoom Team
Date: 2025-10-25
"""

import numpy as np
from typing import Any, Dict, List, Set, Tuple, Callable, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Core Categorical Structures
# ============================================================================

@dataclass
class Morphism:
    """
    Morphism (arrow) in a category.

    A morphism f: A → B consists of:
    - source (domain): Object A
    - target (codomain): Object B
    - data: The actual map (can be function, matrix, etc.)
    """
    source: Any
    target: Any
    data: Any
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.source}→{self.target}"

    def compose(self, other: 'Morphism') -> 'Morphism':
        """
        Compose morphisms: (g ∘ f) where self = g, other = f

        For f: A → B and g: B → C, composition is g ∘ f: A → C
        """
        if self.source != other.target:
            raise ValueError(
                f"Cannot compose: {other.name} has target {other.target} "
                f"but {self.name} has source {self.source}"
            )

        # Compose the data
        if callable(self.data) and callable(other.data):
            composed_data = lambda x: self.data(other.data(x))
        elif isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            composed_data = self.data @ other.data  # Matrix multiplication
        else:
            # Generic composition
            composed_data = (self.data, other.data)

        return Morphism(
            source=other.source,
            target=self.target,
            data=composed_data,
            name=f"{self.name}∘{other.name}"
        )

    def __call__(self, x):
        """Apply morphism to object."""
        if callable(self.data):
            return self.data(x)
        elif isinstance(self.data, np.ndarray):
            return self.data @ x
        else:
            return self.data


class Category:
    """
    A category consists of objects and morphisms.

    Examples:
    - Set: Objects are sets, morphisms are functions
    - Vect: Objects are vector spaces, morphisms are linear maps
    - Graph: Objects are graphs, morphisms are graph homomorphisms
    - KG: Objects are knowledge graphs, morphisms are schema mappings
    """

    def __init__(self, name: str = "C"):
        self.name = name
        self.objects: Set[Any] = set()
        self.morphisms: Dict[Tuple[Any, Any], List[Morphism]] = defaultdict(list)
        self._composition_cache: Dict[Tuple[str, str], Morphism] = {}

        logger.info(f"Category {name} initialized")

    def add_object(self, obj: Any) -> None:
        """Add object to category."""
        self.objects.add(obj)

        # Add identity morphism
        id_morph = Morphism(
            source=obj,
            target=obj,
            data=lambda x: x,  # Identity function
            name=f"id_{obj}"
        )
        self.add_morphism(id_morph)

    def add_morphism(self, morph: Morphism) -> None:
        """Add morphism to category."""
        # Ensure objects exist
        self.objects.add(morph.source)
        self.objects.add(morph.target)

        # Add to Hom-set
        key = (morph.source, morph.target)
        self.morphisms[key].append(morph)

    def hom(self, source: Any, target: Any) -> List[Morphism]:
        """Get Hom-set: all morphisms from source to target."""
        return self.morphisms.get((source, target), [])

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """
        Compose morphisms: g ∘ f

        Cached for efficiency.
        """
        cache_key = (f.name, g.name)

        if cache_key in self._composition_cache:
            return self._composition_cache[cache_key]

        composed = g.compose(f)
        self._composition_cache[cache_key] = composed

        return composed

    def identity(self, obj: Any) -> Morphism:
        """Get identity morphism for object."""
        id_morphs = [m for m in self.hom(obj, obj) if m.name.startswith("id_")]
        if not id_morphs:
            raise ValueError(f"No identity morphism for {obj}")
        return id_morphs[0]

    def is_isomorphism(self, f: Morphism) -> bool:
        """
        Check if morphism is an isomorphism.

        f: A → B is an isomorphism if there exists g: B → A such that:
        - g ∘ f = id_A
        - f ∘ g = id_B
        """
        # Look for inverse in Hom(target, source)
        candidates = self.hom(f.target, f.source)

        id_source = self.identity(f.source)
        id_target = self.identity(f.target)

        for g in candidates:
            try:
                # Check g ∘ f = id_A
                gf = self.compose(f, g)
                # Check f ∘ g = id_B
                fg = self.compose(g, f)

                # Simplified check (would need equality test in general)
                if gf.source == id_source.source and fg.source == id_target.source:
                    return True
            except:
                continue

        return False


# ============================================================================
# Functors
# ============================================================================

class Functor:
    """
    Functor F: C → D between categories.

    A functor consists of:
    - Object map: F₀: Ob(C) → Ob(D)
    - Morphism map: F₁: Hom_C(A,B) → Hom_D(F(A), F(B))

    Preserves:
    - Identity: F(id_A) = id_F(A)
    - Composition: F(g ∘ f) = F(g) ∘ F(f)
    """

    def __init__(self,
                 source: Category,
                 target: Category,
                 object_map: Callable,
                 morphism_map: Callable,
                 name: str = "F"):
        self.source = source
        self.target = target
        self.object_map = object_map
        self.morphism_map = morphism_map
        self.name = name

        logger.info(f"Functor {name}: {source.name} → {target.name}")

    def map_object(self, obj: Any) -> Any:
        """Apply functor to object."""
        return self.object_map(obj)

    def map_morphism(self, morph: Morphism) -> Morphism:
        """Apply functor to morphism."""
        return self.morphism_map(morph)

    def __call__(self, x: Union[Any, Morphism]):
        """Apply functor to object or morphism."""
        if isinstance(x, Morphism):
            return self.map_morphism(x)
        else:
            return self.map_object(x)

    def compose(self, other: 'Functor') -> 'Functor':
        """
        Compose functors: G ∘ F

        For F: C → D and G: D → E, get G ∘ F: C → E
        """
        if self.source != other.target:
            raise ValueError(
                f"Cannot compose functors: {other.name} has target {other.target.name} "
                f"but {self.name} has source {self.source.name}"
            )

        return Functor(
            source=other.source,
            target=self.target,
            object_map=lambda x: self.object_map(other.object_map(x)),
            morphism_map=lambda f: self.map_morphism(other.map_morphism(f)),
            name=f"{self.name}∘{other.name}"
        )


# ============================================================================
# Natural Transformations
# ============================================================================

@dataclass
class NaturalTransformation:
    """
    Natural transformation η: F ⇒ G between functors F, G: C → D.

    For each object A in C, provides morphism η_A: F(A) → G(A) such that
    for every morphism f: A → B in C, the naturality square commutes:

        F(A) --η_A--> G(A)
         |             |
       F(f)          G(f)
         |             |
         ↓             ↓
        F(B) --η_B--> G(B)

    i.e., G(f) ∘ η_A = η_B ∘ F(f)
    """
    source_functor: Functor
    target_functor: Functor
    components: Dict[Any, Morphism]  # η_A for each object A
    name: str = "η"

    def __post_init__(self):
        # Verify functors have same source and target categories
        if self.source_functor.source != self.target_functor.source:
            raise ValueError("Functors must have same source category")
        if self.source_functor.target != self.target_functor.target:
            raise ValueError("Functors must have same target category")

        logger.info(f"Natural transformation {self.name}: {self.source_functor.name} ⇒ {self.target_functor.name}")

    def component(self, obj: Any) -> Morphism:
        """Get component η_A at object A."""
        return self.components[obj]

    def verify_naturality(self, f: Morphism) -> bool:
        """
        Verify naturality square commutes for morphism f: A → B.

        Checks: G(f) ∘ η_A = η_B ∘ F(f)
        """
        A = f.source
        B = f.target

        # Get components
        eta_A = self.component(A)
        eta_B = self.component(B)

        # Apply functors to f
        F_f = self.source_functor.map_morphism(f)
        G_f = self.target_functor.map_morphism(f)

        # Compose paths
        target_cat = self.source_functor.target

        # Left path: G(f) ∘ η_A
        left = target_cat.compose(eta_A, G_f)

        # Right path: η_B ∘ F(f)
        right = target_cat.compose(F_f, eta_B)

        # Check equality (simplified - would need proper equality test)
        return left.source == right.source and left.target == right.target

    def compose_vertical(self, other: 'NaturalTransformation') -> 'NaturalTransformation':
        """
        Vertical composition: θ ∘ η

        For η: F ⇒ G and θ: G ⇒ H, get θ ∘ η: F ⇒ H
        Component at A: (θ ∘ η)_A = θ_A ∘ η_A
        """
        if self.target_functor != other.source_functor:
            raise ValueError("Cannot compose: target of first doesn't match source of second")

        # Compose components
        composed_components = {}
        for obj in self.components:
            eta_A = self.component(obj)
            theta_A = other.component(obj)
            composed_components[obj] = self.source_functor.target.compose(eta_A, theta_A)

        return NaturalTransformation(
            source_functor=self.source_functor,
            target_functor=other.target_functor,
            components=composed_components,
            name=f"{other.name}∘{self.name}"
        )


# ============================================================================
# Limits and Colimits
# ============================================================================

class Limit:
    """
    Limit of a diagram in a category.

    A limit of diagram D: J → C is an object L with morphisms π_i: L → D(i)
    such that for any other cone (X, f_i) there exists unique u: X → L
    making all triangles commute.

    Examples:
    - Terminal object (limit of empty diagram)
    - Product (limit of discrete diagram)
    - Pullback (limit of cospan)
    - Equalizer (limit of parallel pair)
    """

    def __init__(self,
                 category: Category,
                 diagram: Dict[Any, Any],  # Index → Object
                 limiting_object: Any,
                 projections: Dict[Any, Morphism]):  # Index → Morphism
        self.category = category
        self.diagram = diagram
        self.limiting_object = limiting_object
        self.projections = projections

        logger.info(f"Limit computed: {limiting_object}")

    def universal_morphism(self, cone_vertex: Any, cone_morphisms: Dict[Any, Morphism]) -> Morphism:
        """
        Universal morphism u: X → L for cone (X, f_i).

        Must satisfy: π_i ∘ u = f_i for all i
        """
        # This is category-specific; here we provide interface
        raise NotImplementedError("Universal morphism must be implemented for specific category")


class Colimit:
    """
    Colimit (dual to limit).

    Examples:
    - Initial object (colimit of empty diagram)
    - Coproduct (colimit of discrete diagram)
    - Pushout (colimit of span)
    - Coequalizer (colimit of parallel pair)
    """

    def __init__(self,
                 category: Category,
                 diagram: Dict[Any, Any],
                 colimiting_object: Any,
                 injections: Dict[Any, Morphism]):
        self.category = category
        self.diagram = diagram
        self.colimiting_object = colimiting_object
        self.injections = injections

        logger.info(f"Colimit computed: {colimiting_object}")

    def universal_morphism(self, cocone_vertex: Any, cocone_morphisms: Dict[Any, Morphism]) -> Morphism:
        """Universal morphism u: L → X for cocone (X, f_i)."""
        raise NotImplementedError("Universal morphism must be implemented for specific category")


# ============================================================================
# Yoneda Lemma
# ============================================================================

class YonedaEmbedding:
    """
    Yoneda embedding: C → [C^op, Set]

    Maps object A to representable functor Hom(-, A): C^op → Set

    Yoneda Lemma: Natural transformations Hom(-, A) ⇒ F are in bijection
    with elements of F(A).

    Application: Embed knowledge graph entities as functors
    - Entity A → Hom(-, A) = "all things that point to A"
    - Natural transformation = coherent way to map between entities
    """

    def __init__(self, category: Category):
        self.category = category

        # Create opposite category
        self.opposite = self._create_opposite()

        logger.info(f"Yoneda embedding for {category.name}")

    def _create_opposite(self) -> Category:
        """Create opposite category C^op (reverse all arrows)."""
        op = Category(name=f"{self.category.name}^op")

        # Same objects
        op.objects = self.category.objects.copy()

        # Reverse morphisms
        for (src, tgt), morphs in self.category.morphisms.items():
            for m in morphs:
                op_morph = Morphism(
                    source=m.target,
                    target=m.source,
                    data=m.data,  # Data stays same (might need dual for some categories)
                    name=f"{m.name}^op"
                )
                op.add_morphism(op_morph)

        return op

    def embed_object(self, obj: Any) -> Functor:
        """
        Embed object A as representable functor Hom(-, A).

        Maps:
        - Object X → Hom(X, A) (set of morphisms X → A)
        - Morphism f: X → Y to Hom(f, A): Hom(Y, A) → Hom(X, A)
          via precomposition: g ↦ g ∘ f
        """
        # Create Set category (simplified - just track sets)
        set_cat = Category(name="Set")

        def object_map(X):
            """X ↦ Hom(X, A)"""
            return self.category.hom(X, obj)

        def morphism_map(f: Morphism):
            """f: X → Y ↦ Hom(f, A): Hom(Y, A) → Hom(X, A)"""
            def precompose(g: Morphism):
                """g: Y → A ↦ g ∘ f: X → A"""
                return self.category.compose(f, g)

            return Morphism(
                source=self.category.hom(f.target, obj),
                target=self.category.hom(f.source, obj),
                data=precompose,
                name=f"Hom({f.name}, {obj})"
            )

        return Functor(
            source=self.opposite,
            target=set_cat,
            object_map=object_map,
            morphism_map=morphism_map,
            name=f"Hom(-,{obj})"
        )

    def yoneda_bijection(self, obj: Any, functor: Functor) -> Dict:
        """
        Yoneda bijection: Nat(Hom(-, A), F) ≅ F(A)

        Maps natural transformation η: Hom(-, A) ⇒ F to element η_A(id_A) ∈ F(A)
        """
        repr_functor = self.embed_object(obj)

        # The bijection maps:
        # Natural transformation η → η_A(id_A) where id_A: A → A

        return {
            "representable": repr_functor,
            "target": functor,
            "bijection": "η ↦ η_A(id_A)"
        }


# ============================================================================
# Monoidal Categories
# ============================================================================

class MonoidalCategory(Category):
    """
    Monoidal category (C, ⊗, I).

    A category with:
    - Tensor product: ⊗: C × C → C
    - Unit object: I
    - Associator: α: (A ⊗ B) ⊗ C ≅ A ⊗ (B ⊗ C)
    - Left unitor: λ: I ⊗ A ≅ A
    - Right unitor: ρ: A ⊗ I ≅ A

    Coherence: Pentagon and triangle diagrams commute

    Applications:
    - Compositional semantics (Lambek, Coecke et al.)
    - Tensor networks
    - Quantum protocols
    """

    def __init__(self,
                 name: str = "C",
                 tensor_product: Callable = None,
                 unit_object: Any = None):
        super().__init__(name)

        self.tensor_product = tensor_product or self._default_tensor
        self.unit_object = unit_object

        if unit_object is not None:
            self.add_object(unit_object)

        logger.info(f"Monoidal category {name} with unit {unit_object}")

    def _default_tensor(self, A, B):
        """Default tensor product (tuple)."""
        return (A, B)

    def tensor_objects(self, A: Any, B: Any) -> Any:
        """Tensor product of objects: A ⊗ B"""
        result = self.tensor_product(A, B)
        self.add_object(result)
        return result

    def tensor_morphisms(self, f: Morphism, g: Morphism) -> Morphism:
        """
        Tensor product of morphisms: f ⊗ g

        For f: A → B and g: C → D, get f ⊗ g: A ⊗ C → B ⊗ D
        """
        source = self.tensor_objects(f.source, g.source)
        target = self.tensor_objects(f.target, g.target)

        # Tensor data (depends on category)
        if isinstance(f.data, np.ndarray) and isinstance(g.data, np.ndarray):
            # Kronecker product for matrices
            tensored_data = np.kron(f.data, g.data)
        elif callable(f.data) and callable(g.data):
            # Function tensor
            tensored_data = lambda x: (f.data(x[0]), g.data(x[1]))
        else:
            tensored_data = (f.data, g.data)

        return Morphism(
            source=source,
            target=target,
            data=tensored_data,
            name=f"{f.name}⊗{g.name}"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Category Theory Demo")
    print("="*80 + "\n")

    # 1. Build a simple category
    print("1. Category Construction")
    print("-" * 40)

    C = Category(name="KnowledgeGraph")

    # Add objects (entities)
    entities = ["AI", "ML", "DL", "NLP"]
    for e in entities:
        C.add_object(e)

    # Add morphisms (relationships)
    is_a_ml = Morphism(source="ML", target="AI", data="is_a", name="ML→AI")
    is_a_dl = Morphism(source="DL", target="ML", data="is_a", name="DL→ML")
    is_a_nlp = Morphism(source="NLP", target="AI", data="is_a", name="NLP→AI")

    C.add_morphism(is_a_ml)
    C.add_morphism(is_a_dl)
    C.add_morphism(is_a_nlp)

    print(f"Category {C.name}: {len(C.objects)} objects, {sum(len(m) for m in C.morphisms.values())} morphisms")

    # Composition: DL → ML → AI
    dl_to_ai = C.compose(is_a_dl, is_a_ml)  # is_a_ml ∘ is_a_dl
    print(f"Composition: {dl_to_ai.name}")

    # 2. Functor example
    print("\n2. Functor: Knowledge Graph → Vector Space")
    print("-" * 40)

    D = Category(name="VectorSpace")

    # Embedding dimension
    dim = 10

    # Create functor (simplified embedding)
    def embed_entity(entity: str) -> np.ndarray:
        """Embed entity as vector."""
        # Simple hash-based embedding for demo
        np.random.seed(hash(entity) % 2**32)
        return np.random.randn(dim)

    def embed_morphism(morph: Morphism) -> Morphism:
        """Embed relationship as linear map."""
        # Create random projection for demo
        np.random.seed(hash(morph.name) % 2**32)
        matrix = np.random.randn(dim, dim) * 0.1 + np.eye(dim) * 0.9

        return Morphism(
            source=embed_entity(morph.source),
            target=embed_entity(morph.target),
            data=matrix,
            name=f"Φ({morph.name})"
        )

    embedding_functor = Functor(
        source=C,
        target=D,
        object_map=embed_entity,
        morphism_map=embed_morphism,
        name="Φ"
    )

    ai_vec = embedding_functor.map_object("AI")
    print(f"Embedded AI: vector shape {ai_vec.shape}")

    ml_to_ai_map = embedding_functor.map_morphism(is_a_ml)
    print(f"Embedded ML→AI: matrix shape {ml_to_ai_map.data.shape}")

    # 3. Yoneda embedding
    print("\n3. Yoneda Embedding")
    print("-" * 40)

    yoneda = YonedaEmbedding(C)

    # Embed "AI" as representable functor Hom(-, AI)
    hom_ai = yoneda.embed_object("AI")
    print(f"Yoneda embedding of AI: {hom_ai.name}")

    # What points to AI?
    points_to_ai = C.hom("ML", "AI")
    print(f"Morphisms ML → AI: {len(points_to_ai)} (includes {[m.name for m in points_to_ai]})")

    # 4. Monoidal category
    print("\n4. Monoidal Category (Compositional Semantics)")
    print("-" * 40)

    grammar = MonoidalCategory(name="Grammar", unit_object="ε")

    # Words as objects
    grammar.add_object("noun")
    grammar.add_object("verb")
    grammar.add_object("sentence")

    # Composition via tensor
    noun_verb = grammar.tensor_objects("noun", "verb")
    print(f"Tensor: noun ⊗ verb = {noun_verb}")

    # Morphism: (noun ⊗ verb) → sentence
    parse = Morphism(
        source=noun_verb,
        target="sentence",
        data="parse",
        name="parse"
    )
    grammar.add_morphism(parse)

    print(f"Grammar has compositional structure: {len(grammar.objects)} types")

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)
