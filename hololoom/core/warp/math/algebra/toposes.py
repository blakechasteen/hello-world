"""
Toposes, Operads, String Diagrams & Markov Categories
======================================================

Higher-order categorical structures for logic, composition, and
probabilistic reasoning.

Core Concepts:
- Topos: Category with subobject classifier, power objects, pullbacks
- Operad: Colored operations with substitution composition
- String Diagram: Graphical calculus for monoidal categories
- Markov Category: Categorical framework for probability (copy + delete)

Mathematical Foundation:
Topos: (C, Ω, {·}*, Ω^A, pullbacks) where Ω classifies subobjects
Operad: O(c₁,...,cₙ; c) with γ: O(d;c) × ΠO(cᵢ;dⱼ) → O(cᵢ;c)
String diagram: Boxes (morphisms) + wires (objects), read top-to-bottom
Markov: Symmetric monoidal + copy Δ: A→A⊗A + delete ε: A→I

Applications to Warp Space:
- Topos-theoretic logic for sheaf-valued reasoning
- Operadic composition for multi-input weaving operations
- String diagrams for visual verification of tensor computations
- Markov categories for probabilistic weaving cycle semantics

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
# TOPOS
# ============================================================================

@dataclass
class SubobjectClassifier:
    """The subobject classifier Ω in a topos.

    Ω has a distinguished morphism true: 1 → Ω such that for every
    mono m: S ↪ A, there is a unique χ_S: A → Ω making a pullback.
    """
    values: set[str]  # Elements of Ω
    true_element: str = 'true'
    false_element: str = 'false'

    def characteristic(self, total: set[Any], subset: set[Any]) -> dict[Any, str]:
        """Compute characteristic function χ_S: A → Ω.

        χ_S(a) = true if a ∈ S, false otherwise.
        """
        return {a: self.true_element if a in subset else self.false_element for a in total}


class Topos:
    """Elementary topos — a category with finite limits and subobject classifier.

    Implements a finite Set-like topos where objects are finite sets
    and morphisms are functions. Provides the topos-theoretic
    operations: subobject classifier, power objects, exponentials, pullbacks.

    Example:
        >>> T = Topos()
        >>> A = T.make_object({1, 2, 3})
        >>> Omega = T.subobject_classifier()
        >>> PA = T.power_object(A)  # Set of subsets of A
    """

    def __init__(self):
        self._objects: dict[str, set[Any]] = {}
        self._morphisms: dict[str, dict[str, Any]] = {}
        self._omega = SubobjectClassifier(values={'true', 'false'})

    def make_object(self, elements: set[Any], name: str | None = None) -> str:
        """Create a topos object (finite set).

        Args:
            elements: Elements of the set.
            name: Optional name (auto-generated if not provided).

        Returns:
            Object name/identifier.
        """
        if name is None:
            name = f"Obj_{len(self._objects)}"
        self._objects[name] = set(elements)
        return name

    def subobject_classifier(self) -> SubobjectClassifier:
        """Return the subobject classifier Ω = {true, false}.

        In Set, Ω = {0, 1} and the characteristic function of
        S ⊆ A is χ_S: A → {0,1}.
        """
        return self._omega

    def power_object(self, A: str) -> set[frozenset]:
        """Compute power object P(A) = Ω^A.

        In Set, P(A) = set of all subsets of A.

        Args:
            A: Object name.

        Returns:
            Set of frozensets (subsets of A).
        """
        elements = self._objects.get(A, set())
        elem_list = sorted(elements, key=str)
        n = len(elem_list)
        power = set()
        for mask in range(2 ** n):
            subset = frozenset(elem_list[i] for i in range(n) if mask & (1 << i))
            power.add(subset)
        return power

    def exponential(self, A: str, B: str) -> set[tuple]:
        """Compute exponential object B^A (set of functions A → B).

        In Set, B^A is the set of all functions from A to B.

        Args:
            A: Domain object.
            B: Codomain object.

        Returns:
            Set of function representations (as tuples of (input, output) pairs).
        """
        a_elems = sorted(self._objects.get(A, set()), key=str)
        b_elems = sorted(self._objects.get(B, set()), key=str)

        if not a_elems:
            return {()}

        functions: set[tuple] = set()
        n_a = len(a_elems)
        n_b = len(b_elems)

        if n_b == 0:
            return set()

        # Generate all functions A → B
        def generate(idx: int, current: tuple) -> None:
            if idx == n_a:
                functions.add(current)
                return
            for b in b_elems:
                generate(idx + 1, current + ((a_elems[idx], b),))

        generate(0, ())
        return functions

    def pullback(self, f: dict[Any, Any], g: dict[Any, Any]) -> tuple[set[tuple], dict[tuple, Any], dict[tuple, Any]]:
        """Compute pullback of f: A → C and g: B → C.

        P = {(a, b) ∈ A × B : f(a) = g(b)}

        Args:
            f: Function A → C as dict.
            g: Function B → C as dict.

        Returns:
            (pullback_set, proj_A, proj_B) where proj_A/B are projections.
        """
        pullback = set()
        proj_a: dict[tuple, Any] = {}
        proj_b: dict[tuple, Any] = {}

        for a, fc in f.items():
            for b, gc in g.items():
                if fc == gc:
                    pair = (a, b)
                    pullback.add(pair)
                    proj_a[pair] = a
                    proj_b[pair] = b

        return pullback, proj_a, proj_b

    def internal_hom(self, A: str, B: str) -> int:
        """Compute size of internal hom [A, B].

        In Set, |[A,B]| = |B|^|A|.

        Args:
            A, B: Object names.

        Returns:
            Number of morphisms A → B.
        """
        n_a = len(self._objects.get(A, set()))
        n_b = len(self._objects.get(B, set()))
        return n_b ** n_a if n_a > 0 else 1

    def classify_subobject(self, A: str, S: set[Any]) -> dict[Any, str]:
        """Classify a subobject S ⊆ A via characteristic morphism.

        Args:
            A: Ambient object.
            S: Subobject (subset of A's elements).

        Returns:
            Characteristic function χ_S: A → Ω.
        """
        elements = self._objects.get(A, set())
        return self._omega.characteristic(elements, S)


# ============================================================================
# OPERAD
# ============================================================================

@dataclass
class Operation:
    """An operation in an operad.

    Attributes:
        name: Operation name.
        inputs: Input color sequence.
        output: Output color.
        data: Associated data (function, matrix, etc.).
    """
    name: str
    inputs: tuple[str, ...]
    output: str
    data: Any = None

    @property
    def arity(self) -> int:
        return len(self.inputs)


class Operad:
    """Colored operad — operations with typed inputs and outputs.

    An operad O has:
    - Colors (types): C
    - Operations: O(c₁,...,cₙ; c) for each input/output color sequence
    - Composition: plug output of one operation into input of another
    - Identity: id_c ∈ O(c; c) for each color

    The key axiom is associativity of composition (operadic substitution).

    Example:
        >>> op = Operad(['int', 'float', 'string'])
        >>> op.add_operation('add', ('int', 'int'), 'int')
        >>> op.add_operation('to_string', ('int',), 'string')
        >>> result = op.compose('to_string', 'add', 0)  # to_string(add(_, _))
    """

    def __init__(self, colors: list[str]):
        """Initialize operad with given colors.

        Args:
            colors: Set of colors (types).
        """
        self.colors = set(colors)
        self._operations: dict[str, Operation] = {}

        # Add identity operations
        for c in colors:
            self._operations[f"id_{c}"] = Operation(
                name=f"id_{c}",
                inputs=(c,),
                output=c,
            )

    def add_operation(self, name: str, inputs: tuple[str, ...], output: str, data: Any = None) -> None:
        """Add an operation to the operad.

        Args:
            name: Operation name.
            inputs: Input color sequence.
            output: Output color.
            data: Associated data.
        """
        for c in inputs:
            if c not in self.colors:
                raise ValueError(f"Unknown input color: {c}")
        if output not in self.colors:
            raise ValueError(f"Unknown output color: {output}")

        self._operations[name] = Operation(
            name=name, inputs=inputs, output=output, data=data
        )

    def compose(self, outer: str, inner: str, slot: int) -> Operation | None:
        """Compose operations: plug inner's output into outer's slot.

        Given outer ∈ O(c₁,...,cₙ; c) and inner ∈ O(d₁,...,dₘ; cᵢ),
        produces composed ∈ O(c₁,...,d₁,...,dₘ,...,cₙ; c).

        Args:
            outer: Name of outer operation.
            inner: Name of inner operation.
            slot: Input slot of outer to plug inner into (0-indexed).

        Returns:
            Composed operation, or None if type mismatch.
        """
        op_outer = self._operations.get(outer)
        op_inner = self._operations.get(inner)
        if op_outer is None or op_inner is None:
            return None

        if slot >= op_outer.arity:
            return None

        # Type check: inner's output must match outer's slot color
        if op_inner.output != op_outer.inputs[slot]:
            logger.warning(
                f"Type mismatch: {op_inner.output} != {op_outer.inputs[slot]}"
            )
            return None

        # Build new input sequence
        new_inputs = (
            op_outer.inputs[:slot]
            + op_inner.inputs
            + op_outer.inputs[slot + 1:]
        )

        composed_name = f"{outer}∘{inner}@{slot}"
        composed = Operation(
            name=composed_name,
            inputs=new_inputs,
            output=op_outer.output,
        )

        self._operations[composed_name] = composed
        return composed

    def is_associative(self, op1: str, op2: str, op3: str, slot_outer: int = 0, slot_inner: int = 0) -> bool:
        """Check associativity of composition for specific triple.

        (op1 ∘_i op2) ∘_j op3 = op1 ∘_i (op2 ∘_k op3) when applicable.

        Returns:
            True if both sides produce operations with same type signature.
        """
        # Left association: (op1 ∘ op2) ∘ op3
        left_mid = self.compose(op1, op2, slot_outer)
        if left_mid is None:
            return False
        left = self.compose(left_mid.name, op3, slot_inner)

        # Right association: op1 ∘ (op2 ∘ op3)
        right_mid = self.compose(op2, op3, slot_inner)
        if right_mid is None:
            return False
        right = self.compose(op1, right_mid.name, slot_outer)

        if left is None or right is None:
            return left is None and right is None

        return left.inputs == right.inputs and left.output == right.output

    def operations_of_type(self, inputs: tuple[str, ...], output: str) -> list[str]:
        """Find all operations with given type signature."""
        return [
            name for name, op in self._operations.items()
            if op.inputs == inputs and op.output == output
        ]


# ============================================================================
# STRING DIAGRAM
# ============================================================================

@dataclass
class Box:
    """A box (morphism) in a string diagram.

    Attributes:
        name: Box label.
        inputs: Number/labels of input wires.
        outputs: Number/labels of output wires.
        data: Associated morphism data.
    """
    name: str
    inputs: list[str]
    outputs: list[str]
    data: Any = None


class StringDiagram:
    """String diagram — graphical calculus for monoidal categories.

    Represents morphisms as boxes connected by wires. Reading
    top-to-bottom gives composition, side-by-side gives tensor product.

    Example:
        >>> d1 = StringDiagram()
        >>> d1.add_box(Box('f', ['A'], ['B', 'C']))
        >>> d2 = StringDiagram()
        >>> d2.add_box(Box('g', ['B'], ['D']))
        >>> composed = StringDiagram.compose_vertical(d1, d2)
    """

    def __init__(self):
        self._boxes: list[Box] = []
        self._wires: list[tuple[tuple[int, int], tuple[int, int]]] = []
        # Wire: ((source_box, output_idx), (target_box, input_idx))

    def add_box(self, box: Box) -> int:
        """Add a box and return its index."""
        self._boxes.append(box)
        return len(self._boxes) - 1

    @property
    def boxes(self) -> list[Box]:
        return list(self._boxes)

    @property
    def input_wires(self) -> list[str]:
        """Free input wires (not connected to any box output)."""
        if not self._boxes:
            return []
        return list(self._boxes[0].inputs)

    @property
    def output_wires(self) -> list[str]:
        """Free output wires (not connected to any box input)."""
        if not self._boxes:
            return []
        return list(self._boxes[-1].outputs)

    @staticmethod
    def compose_vertical(d1: 'StringDiagram', d2: 'StringDiagram') -> 'StringDiagram':
        """Vertical composition (sequential): d2 after d1.

        Output wires of d1 connect to input wires of d2.
        Requires output types of d1 to match input types of d2.

        Args:
            d1: First diagram (applied first).
            d2: Second diagram (applied second).

        Returns:
            Composed diagram.
        """
        result = StringDiagram()

        # Add all boxes from d1
        offset_1 = 0
        for box in d1._boxes:
            result.add_box(Box(box.name, list(box.inputs), list(box.outputs), box.data))

        # Add all boxes from d2 with offset
        offset_2 = len(d1._boxes)
        for box in d2._boxes:
            result.add_box(Box(box.name, list(box.inputs), list(box.outputs), box.data))

        # Connect d1 outputs to d2 inputs
        if d1._boxes and d2._boxes:
            last_d1 = len(d1._boxes) - 1
            first_d2 = offset_2
            n_connect = min(len(d1._boxes[-1].outputs), len(d2._boxes[0].inputs))
            for i in range(n_connect):
                result._wires.append(((last_d1, i), (first_d2, i)))

        return result

    @staticmethod
    def compose_horizontal(d1: 'StringDiagram', d2: 'StringDiagram') -> 'StringDiagram':
        """Horizontal composition (tensor product): d1 ⊗ d2.

        Places diagrams side by side with no wire connections between them.

        Args:
            d1: Left diagram.
            d2: Right diagram.

        Returns:
            Tensor product diagram.
        """
        result = StringDiagram()

        for box in d1._boxes:
            result.add_box(Box(box.name, list(box.inputs), list(box.outputs), box.data))

        offset = len(d1._boxes)
        for box in d2._boxes:
            result.add_box(Box(box.name, list(box.inputs), list(box.outputs), box.data))

        # Copy wires with offset for d2
        for (s, t) in d1._wires:
            result._wires.append((s, t))
        for (s, t) in d2._wires:
            result._wires.append(
                ((s[0] + offset, s[1]), (t[0] + offset, t[1]))
            )

        return result

    def to_morphism(self) -> np.ndarray | None:
        """Convert to matrix representation (if boxes have matrix data).

        Composes all box matrices vertically (matrix multiplication)
        and horizontally (Kronecker product).

        Returns:
            Composed matrix, or None if boxes lack matrix data.
        """
        if not self._boxes:
            return None

        matrices = []
        for box in self._boxes:
            if isinstance(box.data, np.ndarray):
                matrices.append(box.data)
            elif box.data is None:
                return None
            else:
                return None

        if not matrices:
            return None

        # Simple sequential composition
        result = matrices[0]
        for m in matrices[1:]:
            if result.shape[1] == m.shape[0]:
                result = m @ result
            else:
                result = np.kron(result, m)

        return result

    def diagram_info(self) -> dict[str, Any]:
        """Get diagram structure info."""
        return {
            'n_boxes': len(self._boxes),
            'n_wires': len(self._wires),
            'inputs': self.input_wires,
            'outputs': self.output_wires,
            'boxes': [(b.name, b.inputs, b.outputs) for b in self._boxes],
        }


# ============================================================================
# MARKOV CATEGORY
# ============================================================================

class MarkovCategory:
    """Markov category — categorical probability theory.

    A Markov category is a symmetric monoidal category with:
    - Copy morphism Δ_A: A → A ⊗ A (duplication)
    - Delete morphism ε_A: A → I (marginalization)
    such that (A, Δ, ε) forms a commutative comonoid.

    Morphisms f: A → B are "stochastic maps" (Markov kernels).
    Deterministic morphisms satisfy: Δ_B ∘ f = (f ⊗ f) ∘ Δ_A.

    Example:
        >>> mc = MarkovCategory()
        >>> mc.add_object('X', dim=3)
        >>> mc.add_morphism('f', 'X', 'X', stochastic_matrix)
        >>> mc.is_stochastic('f')
    """

    def __init__(self):
        self._objects: dict[str, int] = {}  # name -> dimension
        self._morphisms: dict[str, dict[str, Any]] = {}

    def add_object(self, name: str, dim: int = 1) -> None:
        """Add an object with given dimension.

        Args:
            name: Object name.
            dim: Dimension (size of state space).
        """
        self._objects[name] = dim

    def add_morphism(self, name: str, source: str, target: str, kernel: np.ndarray) -> None:
        """Add a morphism (Markov kernel).

        Args:
            name: Morphism name.
            source: Source object.
            target: Target object.
            kernel: Stochastic matrix (target_dim, source_dim) with columns summing to 1.
        """
        self._morphisms[name] = {
            'source': source,
            'target': target,
            'kernel': kernel,
        }

    def copy(self, x: str) -> np.ndarray:
        """Copy morphism Δ_X: X → X ⊗ X.

        For finite sets, Δ(i) = (i, i). As a matrix in the
        product basis: Δ_{(i,j), k} = δ_{ik} δ_{jk}.

        Args:
            x: Object name.

        Returns:
            Copy matrix (dim², dim).
        """
        dim = self._objects.get(x, 1)
        delta = np.zeros((dim * dim, dim))
        for i in range(dim):
            delta[i * dim + i, i] = 1.0
        return delta

    def delete(self, x: str) -> np.ndarray:
        """Delete morphism ε_X: X → I.

        Marginalization: maps every state to the single element of I.

        Args:
            x: Object name.

        Returns:
            Delete matrix (1, dim).
        """
        dim = self._objects.get(x, 1)
        return np.ones((1, dim))

    def is_stochastic(self, morphism_name: str) -> bool:
        """Check if a morphism is a valid stochastic map (Markov kernel).

        Requirements:
        - All entries non-negative
        - Columns sum to 1

        Args:
            morphism_name: Morphism to check.

        Returns:
            True if valid stochastic map.
        """
        info = self._morphisms.get(morphism_name)
        if info is None:
            return False

        kernel = info['kernel']
        if not isinstance(kernel, np.ndarray):
            return False

        # Non-negative
        if np.any(kernel < -1e-10):
            return False

        # Columns sum to 1
        col_sums = kernel.sum(axis=0)
        return bool(np.allclose(col_sums, 1.0, atol=1e-8))

    def is_deterministic(self, morphism_name: str) -> bool:
        """Check if morphism is deterministic.

        A morphism f is deterministic iff Δ_B ∘ f = (f ⊗ f) ∘ Δ_A,
        i.e., copying after f equals applying f to each copy.
        Equivalently, each column has exactly one nonzero entry.

        Args:
            morphism_name: Morphism to check.

        Returns:
            True if deterministic.
        """
        info = self._morphisms.get(morphism_name)
        if info is None:
            return False

        kernel = info['kernel']
        # Each column should have exactly one 1 and rest 0
        for col in range(kernel.shape[1]):
            nonzero = np.sum(kernel[:, col] > 1e-10)
            if nonzero != 1:
                return False
        return True

    def compose(self, f_name: str, g_name: str) -> np.ndarray | None:
        """Compose morphisms: g ∘ f.

        For Markov kernels, composition is matrix multiplication.

        Args:
            f_name: First morphism.
            g_name: Second morphism.

        Returns:
            Composed kernel matrix, or None if incompatible.
        """
        f_info = self._morphisms.get(f_name)
        g_info = self._morphisms.get(g_name)
        if f_info is None or g_info is None:
            return None

        if f_info['target'] != g_info['source']:
            return None

        return g_info['kernel'] @ f_info['kernel']

    def tensor_product_morphism(self, f_name: str, g_name: str) -> np.ndarray | None:
        """Tensor product of morphisms: f ⊗ g.

        Uses Kronecker product of kernel matrices.

        Args:
            f_name, g_name: Morphism names.

        Returns:
            Tensor product kernel, or None.
        """
        f_info = self._morphisms.get(f_name)
        g_info = self._morphisms.get(g_name)
        if f_info is None or g_info is None:
            return None

        return np.kron(f_info['kernel'], g_info['kernel'])

    def bayesian_inversion(self, morphism_name: str, prior: np.ndarray) -> np.ndarray | None:
        """Compute Bayesian inverse of a Markov kernel.

        Given f: A → B and prior π on A, the Bayesian inverse
        f†: B → A satisfies f†(b)(a) = f(a)(b) π(a) / Σ_a' f(a')(b) π(a').

        Args:
            morphism_name: Forward kernel f.
            prior: Prior distribution on source (dim_A,).

        Returns:
            Inverse kernel (dim_A, dim_B), or None.
        """
        info = self._morphisms.get(morphism_name)
        if info is None:
            return None

        kernel = info['kernel']  # (dim_B, dim_A)
        prior = np.asarray(prior, dtype=float)

        # Joint: P(a, b) = kernel(b|a) * prior(a)
        joint = kernel * prior[np.newaxis, :]  # (dim_B, dim_A)

        # Marginal: P(b) = Σ_a P(a, b)
        marginal = joint.sum(axis=1)  # (dim_B,)

        # Bayes: P(a|b) = P(a,b) / P(b)
        inverse = np.zeros((prior.shape[0], kernel.shape[0]))
        for b in range(kernel.shape[0]):
            if marginal[b] > 1e-12:
                inverse[:, b] = joint[b, :] / marginal[b]

        return inverse


# ============================================================================
# CATEGORICAL DATABASE (CT3.4)
# ============================================================================

@dataclass
class CategoricalSchema:
    """A database schema as a category.

    Objects are tables, morphisms are foreign key relationships.
    This is Spivak's functorial data model: a schema is a category C,
    and an instance is a functor C → Set.

    Attributes:
        tables: Set of table names (objects).
        foreign_keys: Dict mapping (source_table, key_name) to target_table.
    """
    tables: set[str]
    foreign_keys: dict[tuple[str, str], str]


class CategoricalDatabase:
    """Applied category theory for relational data.

    Models a database as:
    - Schema = category C (tables as objects, foreign keys as morphisms)
    - Instance = functor I: C → Set (rows as set elements)
    - Query = functor Q: C → C' (data migration)

    The functorial approach guarantees:
    - Schema migrations preserve data integrity
    - Queries compose (functors compose)
    - Constraints are natural transformations

    Example:
        >>> db = CategoricalDatabase()
        >>> db.add_table('Employee', [
        ...     {'id': 1, 'name': 'Alice', 'dept': 'eng'},
        ...     {'id': 2, 'name': 'Bob', 'dept': 'sales'},
        ... ])
        >>> db.add_table('Department', [
        ...     {'id': 'eng', 'name': 'Engineering'},
        ...     {'id': 'sales', 'name': 'Sales'},
        ... ])
        >>> db.add_foreign_key('Employee', 'dept', 'Department', 'id')
        >>> schema = db.schema()
    """

    def __init__(self):
        self._tables: dict[str, list[dict]] = {}
        self._foreign_keys: dict[tuple[str, str], tuple[str, str]] = {}

    def add_table(self, name: str, rows: list[dict]) -> None:
        """Add a table (object in schema category).

        Args:
            name: Table name.
            rows: List of row dictionaries.
        """
        self._tables[name] = list(rows)

    def add_foreign_key(self, source: str, source_col: str, target: str, target_col: str) -> None:
        """Add a foreign key (morphism in schema category).

        Args:
            source: Source table.
            source_col: Column in source referencing target.
            target: Target table.
            target_col: Column in target being referenced.
        """
        self._foreign_keys[(source, source_col)] = (target, target_col)

    def schema(self) -> CategoricalSchema:
        """Extract the schema as a category.

        Returns:
            CategoricalSchema with tables and foreign keys.
        """
        fk_map = {(src, col): tgt for (src, col), (tgt, _) in self._foreign_keys.items()}
        return CategoricalSchema(tables=set(self._tables.keys()), foreign_keys=fk_map)

    def instance(self) -> dict[str, list[dict]]:
        """Extract the current instance (functor C → Set).

        Each table maps to its set of rows.

        Returns:
            Dict mapping table name to list of rows.
        """
        return {name: list(rows) for name, rows in self._tables.items()}

    def query(self, source_table: str, projection: list[str] | None = None,
              predicate: Callable[[dict], bool] | None = None) -> list[dict]:
        """Query via functorial projection and selection.

        Projection = postcomposition with a functor that forgets columns.
        Selection = subfunctor (subobject in the functor category).

        Args:
            source_table: Table to query.
            projection: Columns to keep (None = all).
            predicate: Row filter function.

        Returns:
            List of matching rows.
        """
        rows = self._tables.get(source_table, [])

        if predicate is not None:
            rows = [r for r in rows if predicate(r)]

        if projection is not None:
            rows = [{k: r[k] for k in projection if k in r} for r in rows]

        return rows

    def join(self, table1: str, col1: str, table2: str, col2: str) -> list[dict]:
        """Join via pullback in the functor category.

        The categorical join is a pullback: find all pairs of rows
        that agree on the join columns.

        Args:
            table1, col1: Left table and join column.
            table2, col2: Right table and join column.

        Returns:
            List of joined row dicts (prefixed with table name).
        """
        rows1 = self._tables.get(table1, [])
        rows2 = self._tables.get(table2, [])

        result = []
        for r1 in rows1:
            for r2 in rows2:
                if r1.get(col1) == r2.get(col2):
                    joined = {}
                    for k, v in r1.items():
                        joined[f"{table1}.{k}"] = v
                    for k, v in r2.items():
                        joined[f"{table2}.{k}"] = v
                    result.append(joined)

        return result

    def verify_foreign_keys(self) -> list[str]:
        """Check referential integrity (naturality condition).

        Returns:
            List of violation descriptions (empty if valid).
        """
        violations = []

        for (src, src_col), (tgt, tgt_col) in self._foreign_keys.items():
            src_rows = self._tables.get(src, [])
            tgt_rows = self._tables.get(tgt, [])

            tgt_values = {r.get(tgt_col) for r in tgt_rows}

            for row in src_rows:
                val = row.get(src_col)
                if val is not None and val not in tgt_values:
                    violations.append(
                        f"{src}.{src_col} = {val} has no match in {tgt}.{tgt_col}"
                    )

        return violations
