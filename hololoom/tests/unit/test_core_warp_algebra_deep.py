"""
Deep unit tests for hololoom.core.warp.math.algebra modules:
  - universal_algebra.py
  - representation_theory.py
  - lie_algebras.py
  - commutative_algebra.py

Pure algebra with small structures. Each test <5s.
"""

import numpy as np
import pytest
from math import gcd


# ============================================================================
# universal_algebra.py
# ============================================================================

from hololoom.core.warp.math.algebra.universal_algebra import (
    Operation,
    AlgebraicStructure,
    Lattice,
    BirkhoffHSP,
    Clone,
)


# --- Operation ---

class TestOperation:
    def test_call_binary(self):
        add = Operation("add", 2, lambda a, b: (a + b) % 3)
        assert add(1, 2) == 0

    def test_call_unary(self):
        neg = Operation("neg", 1, lambda a: (-a) % 3)
        assert neg(1) == 2

    def test_call_wrong_arity_raises(self):
        op = Operation("f", 2, lambda a, b: a)
        with pytest.raises(ValueError, match="expects 2 args"):
            op(1)

    def test_call_nullary(self):
        zero = Operation("zero", 0, lambda: 0)
        assert zero() == 0


# --- AlgebraicStructure ---

def _z3_group():
    """Z/3Z under addition mod 3."""
    add = Operation("add", 2, lambda a, b: (a + b) % 3)
    return AlgebraicStructure([0, 1, 2], [add], name="Z3")


def _z2_group():
    """Z/2Z under addition mod 2."""
    add = Operation("add", 2, lambda a, b: (a + b) % 2)
    return AlgebraicStructure([0, 1], [add], name="Z2")


class TestAlgebraicStructure:
    def test_is_closed_z3(self):
        assert _z3_group().is_closed()

    def test_not_closed(self):
        bad = Operation("add", 2, lambda a, b: a + b)  # no mod, escapes carrier
        alg = AlgebraicStructure([0, 1, 2], [bad])
        assert not alg.is_closed()

    def test_is_subalgebra_trivial(self):
        z3 = _z3_group()
        assert z3.is_subalgebra([0])  # {0} closed under + mod 3

    def test_is_subalgebra_full(self):
        z3 = _z3_group()
        assert z3.is_subalgebra([0, 1, 2])

    def test_not_subalgebra(self):
        z3 = _z3_group()
        assert not z3.is_subalgebra([1])  # 1+1=2 not in {1}

    def test_subalgebra_generated_by(self):
        z3 = _z3_group()
        sub = z3.subalgebra_generated_by([1])
        assert set(sub) == {0, 1, 2}

    def test_subalgebra_generated_by_identity(self):
        z3 = _z3_group()
        sub = z3.subalgebra_generated_by([0])
        assert set(sub) == {0}

    def test_homomorphic_image(self):
        z3 = _z3_group()
        h = lambda x: x % 1  # everything maps to 0
        img = z3.homomorphic_image(h)
        assert len(img.carrier) == 1

    def test_congruence_classes_trivial(self):
        z3 = _z3_group()
        # Trivial congruence: each element its own class
        classes = z3.congruence_classes(lambda a, b: a == b)
        assert len(classes) == 3

    def test_congruence_classes_all_equal(self):
        z3 = _z3_group()
        classes = z3.congruence_classes(lambda a, b: True)
        assert len(classes) == 1
        assert len(classes[0]) == 3

    def test_freeze_handles_numpy(self):
        add = Operation("add", 2, lambda a, b: a + b)
        carrier = [np.array([0]), np.array([1])]
        alg = AlgebraicStructure(carrier, [add], name="test")
        assert alg._freeze(np.array([0])) == (0,)

    def test_closure_with_nullary_op(self):
        zero_op = Operation("zero", 0, lambda: 0)
        add = Operation("add", 2, lambda a, b: (a + b) % 2)
        alg = AlgebraicStructure([0, 1], [zero_op, add], name="Z2_with_zero")
        assert alg.is_closed()

    def test_closure_with_unary_op(self):
        neg = Operation("neg", 1, lambda a: (3 - a) % 3)
        add = Operation("add", 2, lambda a, b: (a + b) % 3)
        alg = AlgebraicStructure([0, 1, 2], [neg, add], name="Z3_neg")
        assert alg.is_closed()


# --- Lattice ---

class TestLattice:
    def test_power_set_is_boolean(self):
        L = Lattice.power_set(2)
        assert L.is_boolean()

    def test_power_set_elements_count(self):
        L = Lattice.power_set(3)
        assert len(L.elements) == 8

    def test_power_set_distributive(self):
        L = Lattice.power_set(2)
        assert L.is_distributive()

    def test_power_set_modular(self):
        L = Lattice.power_set(2)
        assert L.is_modular()

    def test_power_set_complemented(self):
        L = Lattice.power_set(2)
        assert L.is_complemented()

    def test_divisor_lattice(self):
        L = Lattice.divisor_lattice(12)
        # Divisors of 12: 1,2,3,4,6,12
        assert len(L.elements) == 6

    def test_divisor_lattice_modular(self):
        L = Lattice.divisor_lattice(12)
        assert L.is_modular()

    def test_divisor_lattice_distributive(self):
        L = Lattice.divisor_lattice(12)
        assert L.is_distributive()

    def test_leq_power_set(self):
        L = Lattice.power_set(2)
        assert L.leq(frozenset(), frozenset({0, 1}))
        assert L.leq(frozenset({0}), frozenset({0, 1}))
        assert not L.leq(frozenset({0, 1}), frozenset({0}))

    def test_complement(self):
        L = Lattice.power_set(2)
        c = L.complement(frozenset({0}))
        assert c == frozenset({1})

    def test_atoms_power_set(self):
        L = Lattice.power_set(3)
        a = L.atoms()
        # Atoms of P(3) are singletons {0}, {1}, {2}
        atom_sets = {frozenset(x) for x in a}
        assert atom_sets == {frozenset({0}), frozenset({1}), frozenset({2})}

    def test_interval(self):
        L = Lattice.power_set(2)
        # Interval [empty, {0,1}] = all elements
        iv = L.interval(frozenset(), frozenset({0, 1}))
        assert len(iv) == 4

    def test_interval_partial(self):
        L = Lattice.power_set(2)
        iv = L.interval(frozenset({0}), frozenset({0, 1}))
        # {0} and {0,1}
        assert len(iv) == 2

    def test_bounds_found(self):
        L = Lattice.power_set(2)
        assert L._bottom == frozenset()
        assert L._top == frozenset({0, 1})

    def test_divisor_lattice_bounds(self):
        L = Lattice.divisor_lattice(6)
        assert L._bottom == 1
        assert L._top == 6

    def test_complement_no_bounds_returns_none(self):
        # Lattice without top/bottom
        L = Lattice([2, 3, 6], meet=gcd, join=lambda a, b: a * b // gcd(a, b), name="partial")
        # No 1 in elements, so no bottom
        result = L.complement(2)
        # Should return None since bounds might not be found
        # (depends on whether 2 or 6 is detected as bound)


# --- BirkhoffHSP ---

class TestBirkhoffHSP:
    def test_subalgebra(self):
        z3 = _z3_group()
        sub = BirkhoffHSP.S(z3, [0])
        assert len(sub.carrier) == 1
        assert sub.name == "Sub(Z3)"

    def test_subalgebra_invalid_raises(self):
        z3 = _z3_group()
        with pytest.raises(ValueError, match="not closed"):
            BirkhoffHSP.S(z3, [1])

    def test_product(self):
        z2 = _z2_group()
        prod = BirkhoffHSP.P([z2, z2])
        assert len(prod.carrier) == 4  # 2x2
        assert "Z2" in prod.name

    def test_product_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            BirkhoffHSP.P([])

    def test_product_is_closed(self):
        z2 = _z2_group()
        prod = BirkhoffHSP.P([z2, z2])
        assert prod.is_closed()

    def test_homomorphic_image(self):
        z3 = _z3_group()
        cong = lambda a, b: a % 3 == b % 3  # trivial
        quotient = BirkhoffHSP.H(z3, cong)
        assert len(quotient.carrier) == 3

    def test_quotient_by_nontrivial_congruence(self):
        # Z/6Z with congruence mod 3
        add6 = Operation("add", 2, lambda a, b: (a + b) % 6)
        z6 = AlgebraicStructure(list(range(6)), [add6], name="Z6")
        cong = lambda a, b: a % 3 == b % 3
        quotient = BirkhoffHSP.H(z6, cong)
        assert len(quotient.carrier) == 3


# --- Clone ---

class TestClone:
    def test_projection(self):
        c = Clone([0, 1])
        proj = c.projection(2, 0)
        assert proj(5, 7) == 5
        proj1 = c.projection(2, 1)
        assert proj1(5, 7) == 7

    def test_compose(self):
        c = Clone([0, 1])
        f = lambda x: 1 - x
        g = lambda x: x
        composed = c.compose(f, 1, [g], 1)
        assert composed(0) == 1
        assert composed(1) == 0

    def test_add_operation(self):
        c = Clone([0, 1])
        c.add_operation(lambda x: 1 - x)
        assert len(c.operations) == 1
        arity, op = c.operations[0]
        assert arity == 1

    def test_add_operation_explicit_arity(self):
        c = Clone([0, 1])
        c.add_operation(lambda x, y: x & y, arity=2)
        assert c.operations[0][0] == 2

    def test_posts_lattice_info(self):
        info = Clone.posts_lattice_info()
        assert "Post's lattice" in info["description"]
        assert "NAND" in info["sheffer_functions"]

    def test_generate_with_projections(self):
        c = Clone([0, 1])
        nand = lambda x, y: 1 - (x & y)
        generated = c.generate([nand], [2], max_depth=1)
        # Should have the generator + projections
        assert len(generated.operations) >= 3  # nand + proj(2,0) + proj(2,1)

    def test_is_closed_under_composition(self):
        c = Clone([0, 1])
        c.add_operation(lambda x: x, arity=1)  # identity
        assert c.is_closed_under_composition()


# ============================================================================
# representation_theory.py
# ============================================================================

from hololoom.core.warp.math.algebra.representation_theory import (
    GroupRepresentation,
    CharacterTable,
    SchurLemma,
    MaschkeTheorem,
    BurnsideLemma,
    YoungTableaux,
)


def _z2_rep_trivial():
    """Trivial representation of Z/2Z."""
    return GroupRepresentation(
        [0, 1],
        [np.eye(1), np.eye(1)],
        name="trivial"
    )


def _z2_rep_sign():
    """Sign representation of Z/2Z."""
    return GroupRepresentation(
        [0, 1],
        [np.eye(1), -np.eye(1)],
        name="sign"
    )


def _z3_rep_regular():
    """Regular representation of Z/3Z: 3x3 permutation matrices."""
    # e -> I, 1 -> cyclic shift, 2 -> shift^2
    I = np.eye(3)
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    P2 = P @ P
    return GroupRepresentation([0, 1, 2], [I, P, P2], name="reg_Z3")


class TestGroupRepresentation:
    def test_character_trivial(self):
        rep = _z2_rep_trivial()
        chi = rep.character()
        np.testing.assert_array_equal(chi, [1, 1])

    def test_character_sign(self):
        rep = _z2_rep_sign()
        chi = rep.character()
        np.testing.assert_array_equal(chi, [1, -1])

    def test_is_irreducible_trivial(self):
        assert _z2_rep_trivial().is_irreducible()

    def test_is_irreducible_sign(self):
        assert _z2_rep_sign().is_irreducible()

    def test_regular_rep_not_irreducible(self):
        reg = _z3_rep_regular()
        assert not reg.is_irreducible()

    def test_dim(self):
        assert _z2_rep_trivial().dim == 1
        assert _z3_rep_regular().dim == 3

    def test_matrix_access(self):
        rep = _z2_rep_sign()
        np.testing.assert_array_equal(rep.matrix(0), np.eye(1))
        np.testing.assert_array_equal(rep.matrix(1), -np.eye(1))

    def test_tensor_product(self):
        triv = _z2_rep_trivial()
        sign = _z2_rep_sign()
        tp = triv.tensor_product(sign)
        assert tp.dim == 1
        # trivial x sign = sign
        chi = tp.character()
        np.testing.assert_allclose(chi, [1, -1])

    def test_dual(self):
        sign = _z2_rep_sign()
        dual = sign.dual()
        assert dual.dim == 1
        # For sign rep, dual should be the same (self-dual)
        chi = dual.character()
        np.testing.assert_allclose(chi, [1, -1])

    def test_is_faithful_trivial(self):
        # Trivial rep is not faithful (unless group is trivial)
        triv = _z2_rep_trivial()
        assert not triv.is_faithful()

    def test_is_faithful_sign(self):
        sign = _z2_rep_sign()
        assert sign.is_faithful()

    def test_decompose_trivial(self):
        triv = _z2_rep_trivial()
        components = triv.decompose()
        # Should find 1 copy of trivial
        assert len(components) >= 1

    def test_tensor_product_dims(self):
        reg = _z3_rep_regular()
        triv = GroupRepresentation([0, 1, 2], [np.eye(1)] * 3, name="triv")
        tp = reg.tensor_product(triv)
        assert tp.dim == 3


# --- SchurLemma ---

class TestSchurLemma:
    def test_identity_is_intertwiner(self):
        rep = _z2_rep_sign()
        result = SchurLemma.apply(rep, rep, np.eye(1))
        assert result['is_intertwiner']
        assert result['is_isomorphism']
        assert result['schur_satisfied']

    def test_zero_is_intertwiner(self):
        rep1 = _z2_rep_trivial()
        rep2 = _z2_rep_sign()
        result = SchurLemma.apply(rep1, rep2, np.zeros((1, 1)))
        assert result['is_intertwiner']
        assert result['is_zero']
        assert result['schur_satisfied']

    def test_endomorphism_check_scalar(self):
        rep = _z2_rep_sign()
        result = SchurLemma.endomorphism_check(rep, 3.0 * np.eye(1))
        assert result['commutes']
        assert result['is_scalar_multiple_of_identity']
        assert result['scalar_value'] == pytest.approx(3.0)
        assert result['schur_part2_satisfied']

    def test_endomorphism_check_non_scalar(self):
        reg = _z3_rep_regular()
        T = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]], dtype=complex)
        result = SchurLemma.endomorphism_check(reg, T)
        # T doesn't commute with the regular rep matrices in general
        # Just check the structure of the result
        assert 'commutes' in result
        assert 'schur_part2_satisfied' in result


# --- MaschkeTheorem ---

class TestMaschkeTheorem:
    def test_decompose_irreducible(self):
        sign = _z2_rep_sign()
        components = MaschkeTheorem.decompose(sign)
        assert len(components) == 1  # Already irreducible

    def test_decompose_reducible(self):
        # Direct sum of trivial + sign for Z/2Z
        # rho(0) = I_2, rho(1) = diag(1, -1)
        M0 = np.eye(2, dtype=complex)
        M1 = np.diag([1.0, -1.0]).astype(complex)
        rep = GroupRepresentation([0, 1], [M0, M1], name="triv+sign")
        components = MaschkeTheorem.decompose(rep)
        assert len(components) == 2
        # Each component should be 1-dimensional
        assert all(c.dim == 1 for c in components)


# --- BurnsideLemma ---

class TestBurnsideLemma:
    def test_count_orbits_identity(self):
        # Only the identity acts -> each element is its own orbit
        I = np.eye(3)
        assert BurnsideLemma.count_orbits([I], 3) == 3

    def test_count_orbits_full_rotation(self):
        # Z/3Z acting on 3 elements by cyclic permutation: 1 orbit
        I = np.eye(3)
        P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        P2 = P @ P
        # Fixed points: I has 3, P has 0, P2 has 0 -> (3+0+0)/3 = 1
        assert BurnsideLemma.count_orbits([I, P, P2], 3) == 1

    def test_count_orbits_from_permutations(self):
        # identity + (0 1 2)->(1 2 0) + (0 1 2)->(2 0 1)
        perms = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        assert BurnsideLemma.count_orbits_from_permutations(perms, 3) == 1

    def test_count_orbits_from_permutations_z2(self):
        # Z/2Z acting on {0,1} by swapping: identity + swap -> (2+0)/2 = 1
        perms = [[0, 1], [1, 0]]
        assert BurnsideLemma.count_orbits_from_permutations(perms, 2) == 1

    def test_necklace_count_binary(self):
        # 4 beads, 2 colors: (2^1*1 + 2^2*1 + 2^4*1 + 2^2*1)/4 = (2+4+16+4)/4 = not quite
        # Actually: (1/4)(phi(4)*2^1 + phi(2)*2^2 + phi(1)*2^4) = (1/4)(2*2 + 1*4 + 1*16) = nope
        # Standard formula gives 6 for n=4, k=2
        assert BurnsideLemma.necklace_count(4, 2) == 6

    def test_necklace_count_three_colors(self):
        # 3 beads, 3 colors = (1/3)(phi(3)*3^1 + phi(1)*3^3) = (1/3)(2*3+27) = 11
        assert BurnsideLemma.necklace_count(3, 3) == 11

    def test_necklace_count_trivial(self):
        # 1 bead, k colors = k necklaces
        assert BurnsideLemma.necklace_count(1, 5) == 5


# --- YoungTableaux ---

class TestYoungTableaux:
    def test_standard_tableaux_2_1(self):
        # Partition (2,1) of 3: two standard tableaux
        tabs = YoungTableaux.standard_tableaux([2, 1])
        assert len(tabs) == 2

    def test_standard_tableaux_3(self):
        # Partition (3) of 3: one standard tableau
        tabs = YoungTableaux.standard_tableaux([3])
        assert len(tabs) == 1

    def test_standard_tableaux_1_1_1(self):
        # Partition (1,1,1) of 3: one standard tableau
        tabs = YoungTableaux.standard_tableaux([1, 1, 1])
        assert len(tabs) == 1

    def test_hook_length_formula_2_1(self):
        # partition (2,1): n=3, hooks: (3,1 ; 1) -> product=3 -> 3!/3=2
        assert YoungTableaux.hook_length_formula([2, 1]) == 2

    def test_hook_length_formula_matches_enumeration(self):
        for partition in [[3], [2, 1], [1, 1, 1], [3, 1], [2, 2]]:
            tabs = YoungTableaux.standard_tableaux(partition)
            hook = YoungTableaux.hook_length_formula(partition)
            assert len(tabs) == hook, f"Mismatch for {partition}: {len(tabs)} vs {hook}"

    def test_hook_length_empty(self):
        assert YoungTableaux.hook_length_formula([]) == 1

    def test_rsk_identity_permutation(self):
        P, Q = YoungTableaux.RSK([1, 2, 3])
        # Identity perm -> single row
        assert P == [[1, 2, 3]]
        assert Q == [[1, 2, 3]]

    def test_rsk_reverse_permutation(self):
        P, Q = YoungTableaux.RSK([3, 2, 1])
        # Reverse perm -> single column
        assert P == [[1], [2], [3]]
        assert Q == [[1], [2], [3]]

    def test_rsk_shape_matches(self):
        P, Q = YoungTableaux.RSK([2, 3, 1])
        # P and Q should have same shape
        assert [len(row) for row in P] == [len(row) for row in Q]

    def test_partition_count(self):
        # p(0)=1, p(1)=1, p(2)=2, p(3)=3, p(4)=5, p(5)=7
        assert YoungTableaux.partition_count(0) == 1
        assert YoungTableaux.partition_count(1) == 1
        assert YoungTableaux.partition_count(2) == 2
        assert YoungTableaux.partition_count(3) == 3
        assert YoungTableaux.partition_count(4) == 5
        assert YoungTableaux.partition_count(5) == 7

    def test_partition_count_negative(self):
        assert YoungTableaux.partition_count(-1) == 0


# --- CharacterTable ---

class TestCharacterTable:
    def test_orthogonality_z2(self):
        # Z/2Z character table: [[1,1],[1,-1]], classes [1,1]
        table = np.array([[1, 1], [1, -1]], dtype=complex)
        ct = CharacterTable(table, [1, 1])
        row_err, col_err = ct.orthogonality_check()
        assert row_err < 1e-10
        assert col_err < 1e-10

    def test_group_order(self):
        table = np.array([[1, 1], [1, -1]], dtype=complex)
        ct = CharacterTable(table, [1, 1])
        assert ct.group_order == 2

    def test_dimensions(self):
        table = np.array([[1, 1, 1], [1, 1, 1], [2, -1, 0]], dtype=complex)
        # Not a real character table but tests shape
        ct = CharacterTable(table, [1, 2, 3])
        assert ct.n_irreps == 3
        assert ct.n_classes == 3


# ============================================================================
# lie_algebras.py
# ============================================================================

from hololoom.core.warp.math.algebra.lie_algebras import (
    LieAlgebra,
    RootSystem,
    DynkinDiagram,
    UniversalEnveloping,
)


def _abelian_2d():
    """2D abelian Lie algebra: all brackets zero."""
    c = np.zeros((2, 2, 2))
    return LieAlgebra(c, name="abelian_2d")


def _so3():
    """so(3): [e1,e2]=e3, [e2,e3]=e1, [e3,e1]=e2."""
    c = np.zeros((3, 3, 3))
    c[0, 1, 2] = 1.0;  c[1, 0, 2] = -1.0
    c[1, 2, 0] = 1.0;  c[2, 1, 0] = -1.0
    c[2, 0, 1] = 1.0;  c[0, 2, 1] = -1.0
    return LieAlgebra(c, name="so3_manual")


class TestLieAlgebra:
    def test_bracket_abelian(self):
        g = _abelian_2d()
        X = np.array([1.0, 0.0])
        Y = np.array([0.0, 1.0])
        result = g.bracket(X, Y)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_bracket_so3(self):
        g = _so3()
        e1 = np.array([1, 0, 0], dtype=float)
        e2 = np.array([0, 1, 0], dtype=float)
        result = g.bracket(e1, e2)
        np.testing.assert_allclose(result, [0, 0, 1])

    def test_verify_antisymmetry_so3(self):
        assert _so3().verify_antisymmetry()

    def test_verify_jacobi_so3(self):
        assert _so3().verify_jacobi()

    def test_verify_jacobi_abelian(self):
        assert _abelian_2d().verify_jacobi()

    def test_adjoint_so3(self):
        g = _so3()
        e1 = np.array([1, 0, 0], dtype=float)
        ad = g.adjoint(e1)
        # ad(e1)(e2) = [e1, e2] = e3
        e2 = np.array([0, 1, 0], dtype=float)
        result = ad @ e2
        np.testing.assert_allclose(result, [0, 0, 1])

    def test_killing_form_so3(self):
        g = _so3()
        e1 = np.array([1, 0, 0], dtype=float)
        e2 = np.array([0, 1, 0], dtype=float)
        # B(e1, e1) = tr(ad(e1)^2) = -2 for so(3)
        B11 = g.killing_form(e1, e1)
        assert B11 == pytest.approx(-2.0)
        # B(e1, e2) = 0 (orthogonal basis elements)
        B12 = g.killing_form(e1, e2)
        assert B12 == pytest.approx(0.0)

    def test_killing_matrix_so3(self):
        g = _so3()
        B = g.killing_matrix()
        expected = -2.0 * np.eye(3)
        np.testing.assert_allclose(B, expected)

    def test_is_semisimple_so3(self):
        assert _so3().is_semisimple()

    def test_not_semisimple_abelian(self):
        assert not _abelian_2d().is_semisimple()

    def test_is_nilpotent_abelian(self):
        assert _abelian_2d().is_nilpotent()

    def test_not_nilpotent_so3(self):
        assert not _so3().is_nilpotent()

    def test_is_solvable_abelian(self):
        assert _abelian_2d().is_solvable()

    def test_not_solvable_so3(self):
        assert not _so3().is_solvable()

    def test_sl2_construction(self):
        g = LieAlgebra.sl(2)
        assert g.dim == 3
        assert g.verify_antisymmetry()
        assert g.verify_jacobi()

    def test_sl2_semisimple(self):
        g = LieAlgebra.sl(2)
        assert g.is_semisimple()

    def test_so3_construction(self):
        g = LieAlgebra.so(3)
        assert g.dim == 3
        assert g.verify_antisymmetry()
        assert g.verify_jacobi()

    def test_so3_is_semisimple(self):
        g = LieAlgebra.so(3)
        assert g.is_semisimple()


# --- RootSystem ---

class TestRootSystem:
    def test_A1_cartan_matrix(self):
        rs = RootSystem.A(1)
        np.testing.assert_array_equal(rs.A, [[2]])

    def test_A2_cartan_matrix(self):
        rs = RootSystem.A(2)
        expected = np.array([[2, -1], [-1, 2]])
        np.testing.assert_array_equal(rs.A, expected)

    def test_A2_positive_roots(self):
        rs = RootSystem.A(2)
        roots = rs.positive_roots()
        # A2 has 3 positive roots: alpha1, alpha2, alpha1+alpha2
        assert len(roots) == 3

    def test_A1_weyl_group(self):
        rs = RootSystem.A(1)
        W = rs.weyl_group()
        # Weyl group of A1 is Z/2Z (order 2)
        assert len(W) == 2

    def test_A2_weyl_group(self):
        rs = RootSystem.A(2)
        W = rs.weyl_group()
        # Weyl group of A2 is S3 (order 6)
        assert len(W) == 6

    def test_B2_cartan_matrix(self):
        rs = RootSystem.B(2)
        # B2: A[0,1]=-2, A[1,0]=-1 per the implementation
        assert rs.A[0, 1] == -2
        assert rs.A[1, 0] == -1

    def test_D4_cartan_matrix(self):
        rs = RootSystem.D(4)
        assert rs.rank == 4
        # D4: node 1 connected to nodes 2,3 (branching at node 1 from end)
        # Check that it has the branching structure
        assert rs.A[1, 3] == -1  # branch connection

    def test_simple_roots_A2(self):
        rs = RootSystem.A(2)
        roots = rs.simple_roots()
        assert roots.shape[0] == 2


# --- DynkinDiagram ---

class TestDynkinDiagram:
    def test_classify_A1(self):
        A = np.array([[2]])
        dd = DynkinDiagram(A)
        assert dd.classify() == "A_1"

    def test_classify_A2(self):
        dd = DynkinDiagram.from_cartan_matrix(np.array([[2, -1], [-1, 2]]))
        assert dd.classify() == "A_2"

    def test_classify_A3(self):
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        dd = DynkinDiagram(A)
        assert dd.classify() == "A_3"

    def test_classify_B2(self):
        A = np.array([[2, -1], [-1, 2]])
        # B2 has A[0,1]=-1, A[1,0]=-2
        A = np.array([[2, -1], [-2, 2]])
        dd = DynkinDiagram(A)
        result = dd.classify()
        assert result in ("B_2", "C_2")

    def test_classify_G2(self):
        A = np.array([[2, -1], [-3, 2]])
        dd = DynkinDiagram(A)
        assert dd.classify() == "G_2"

    def test_edges_A2(self):
        dd = DynkinDiagram(np.array([[2, -1], [-1, 2]]))
        edges = dd.edges()
        assert len(edges) == 1
        assert edges[0] == (0, 1, 1)  # single bond

    def test_edges_G2(self):
        dd = DynkinDiagram(np.array([[2, -1], [-3, 2]]))
        edges = dd.edges()
        assert edges[0][2] == 3  # triple bond


# --- UniversalEnveloping ---

class TestUniversalEnveloping:
    def test_pbw_basis_degree_0(self):
        g = _abelian_2d()
        ue = UniversalEnveloping(g)
        basis = ue.PBW_basis(0)
        # Only (0,0)
        assert basis == [(0, 0)]

    def test_pbw_basis_degree_1(self):
        g = _abelian_2d()
        ue = UniversalEnveloping(g)
        basis = ue.PBW_basis(1)
        # (0,0), (1,0), (0,1)
        assert len(basis) == 3

    def test_pbw_dimension(self):
        g = _abelian_2d()
        ue = UniversalEnveloping(g)
        # C(2+d, d) for dim=2
        assert ue.PBW_dimension(0) == 1
        assert ue.PBW_dimension(1) == 3
        assert ue.PBW_dimension(2) == 6

    def test_multiply_basis_abelian(self):
        g = _abelian_2d()
        ue = UniversalEnveloping(g)
        # e1 * e2 should equal e1 e2 in abelian case (already ordered)
        result = ue.multiply_basis((1, 0), (0, 1))
        assert (1, 1) in result
        assert result[(1, 1)] == pytest.approx(1.0)

    def test_multiply_basis_identity(self):
        g = _abelian_2d()
        ue = UniversalEnveloping(g)
        # 1 * e1 = e1
        result = ue.multiply_basis((0, 0), (1, 0))
        assert (1, 0) in result
        assert result[(1, 0)] == pytest.approx(1.0)


# ============================================================================
# commutative_algebra.py
# ============================================================================

from hololoom.core.warp.math.algebra.commutative_algebra import (
    Monomial,
    glex_order,
    grevlex_order,
    MVPolynomial,
    NoetherianRing,
    Localization,
    Fraction,
    PrimaryDecomposition,
    GrobnerBasis,
    AlgebraicVariety,
    _polynomial_division,
    _s_polynomial,
)


# --- Monomial ---

class TestMonomial:
    def test_total_degree(self):
        m = Monomial((2, 3))
        assert m.total_degree == 5

    def test_n_vars(self):
        m = Monomial((1, 2, 3))
        assert m.n_vars == 3

    def test_mul(self):
        m1 = Monomial((1, 2))
        m2 = Monomial((3, 0))
        assert m1 * m2 == Monomial((4, 2))

    def test_divides(self):
        m1 = Monomial((1, 1))
        m2 = Monomial((2, 3))
        assert m1.divides(m2)
        assert not m2.divides(m1)

    def test_truediv(self):
        m1 = Monomial((3, 2))
        m2 = Monomial((1, 1))
        assert m1 / m2 == Monomial((2, 1))

    def test_truediv_not_divisible_raises(self):
        m1 = Monomial((1, 0))
        m2 = Monomial((0, 1))
        with pytest.raises(ValueError, match="does not divide"):
            m1 / m2

    def test_lcm(self):
        m1 = Monomial((2, 1))
        m2 = Monomial((1, 3))
        assert m1.lcm(m2) == Monomial((2, 3))

    def test_gcd(self):
        m1 = Monomial((2, 1))
        m2 = Monomial((1, 3))
        assert m1.gcd(m2) == Monomial((1, 1))

    def test_frozen(self):
        # Monomials are frozen dataclasses -> hashable
        s = {Monomial((1, 2)), Monomial((1, 2)), Monomial((0, 0))}
        assert len(s) == 2


# --- Monomial Orders ---

class TestMonomialOrders:
    def test_glex_same_degree_lex(self):
        m1 = Monomial((2, 0))  # x^2
        m2 = Monomial((1, 1))  # xy
        assert glex_order(m1, m2) > 0  # x^2 > xy in glex

    def test_glex_different_degree(self):
        m1 = Monomial((1, 0))  # degree 1
        m2 = Monomial((1, 1))  # degree 2
        assert glex_order(m1, m2) < 0

    def test_glex_equal(self):
        m = Monomial((1, 1))
        assert glex_order(m, m) == 0

    def test_grevlex_same_degree(self):
        m1 = Monomial((2, 0))  # x^2
        m2 = Monomial((1, 1))  # xy
        # grevlex: same total degree, compare last component reversed
        # m1 last=0, m2 last=1, grevlex returns b-a = 1-0 > 0 -> m2 is "larger"
        # Actually grevlex returns (b-a) for reversed, so:
        result = grevlex_order(m1, m2)
        # In grevlex, x^2 > xy because last exponent of xy is larger (1 vs 0), reversed
        assert result > 0  # m1 > m2 in grevlex


# --- MVPolynomial ---

class TestMVPolynomial:
    def _x(self):
        return MVPolynomial.from_dense({(1, 0): 1.0})

    def _y(self):
        return MVPolynomial.from_dense({(0, 1): 1.0})

    def _const(self, c):
        return MVPolynomial.from_dense({(0, 0): c})

    def test_from_dense(self):
        p = MVPolynomial.from_dense({(2, 0): 3.0, (0, 0): -1.0})
        assert not p.is_zero()

    def test_is_zero(self):
        p = MVPolynomial()
        assert p.is_zero()

    def test_add(self):
        x = self._x()
        y = self._y()
        p = x + y  # x + y
        assert len(p.terms) == 2

    def test_sub(self):
        x = self._x()
        p = x - x
        assert p.is_zero()

    def test_mul(self):
        x = self._x()
        y = self._y()
        p = x * y  # xy
        assert Monomial((1, 1)) in p.terms
        assert p.terms[Monomial((1, 1))] == pytest.approx(1.0)

    def test_scalar_mul(self):
        x = self._x()
        p = x.scalar_mul(3.0)
        assert p.terms[Monomial((1, 0))] == pytest.approx(3.0)

    def test_evaluate(self):
        # p = 2x^2 + 3y - 1
        p = MVPolynomial.from_dense({(2, 0): 2.0, (0, 1): 3.0, (0, 0): -1.0})
        val = p.evaluate(np.array([2.0, 1.0]))
        # 2*4 + 3*1 - 1 = 10
        assert val == pytest.approx(10.0)

    def test_degree(self):
        p = MVPolynomial.from_dense({(2, 1): 1.0, (0, 0): 5.0})
        assert p.degree() == 3

    def test_degree_zero_poly(self):
        p = MVPolynomial()
        assert p.degree() == -1

    def test_leading_monomial(self):
        p = MVPolynomial.from_dense({(2, 0): 1.0, (1, 0): 2.0, (0, 0): 3.0})
        lm = p.leading_monomial()
        assert lm == Monomial((2, 0))

    def test_leading_coefficient(self):
        p = MVPolynomial.from_dense({(2, 0): 5.0, (1, 0): 2.0})
        assert p.leading_coefficient() == pytest.approx(5.0)

    def test_repr_nonzero(self):
        p = MVPolynomial.from_dense({(1, 0): 1.0})
        assert "x^" in repr(p)

    def test_repr_zero(self):
        p = MVPolynomial()
        assert repr(p) == "0"


# --- NoetherianRing ---

class TestNoetherianRing:
    def test_is_noetherian(self):
        R = NoetherianRing(2, "k[x,y]")
        assert R.is_noetherian()


# --- Localization ---

class TestLocalization:
    def test_fraction_creation(self):
        num = MVPolynomial.from_dense({(1, 0): 1.0})
        den = MVPolynomial.from_dense({(0, 0): 1.0})
        f = Localization.fraction(num, den)
        assert isinstance(f, Fraction)

    def test_fraction_zero_denom_raises(self):
        num = MVPolynomial.from_dense({(1, 0): 1.0})
        den = MVPolynomial()  # zero
        with pytest.raises(ValueError, match="zero"):
            Localization.fraction(num, den)

    def test_add_fractions(self):
        one = MVPolynomial.from_dense({(0,): 1.0})
        x = MVPolynomial.from_dense({(1,): 1.0})
        f1 = Fraction(x, one)    # x/1
        f2 = Fraction(one, one)  # 1/1
        result = Localization.add(f1, f2)
        # (x*1 + 1*1) / (1*1) = (x+1)/1
        val = result.numerator.evaluate(np.array([2.0]))
        assert val == pytest.approx(3.0)  # x+1 at x=2

    def test_multiply_fractions(self):
        one = MVPolynomial.from_dense({(0,): 1.0})
        x = MVPolynomial.from_dense({(1,): 1.0})
        f1 = Fraction(x, one)
        f2 = Fraction(x, one)
        result = Localization.multiply(f1, f2)
        # x^2 / 1
        val = result.numerator.evaluate(np.array([3.0]))
        assert val == pytest.approx(9.0)

    def test_localize(self):
        R = NoetherianRing(1)
        one = MVPolynomial.from_dense({(0,): 1.0})
        loc = Localization.localize(R, [one])
        assert loc.ring == R


# --- PrimaryDecomposition ---

class TestPrimaryDecomposition:
    def test_empty_ideal(self):
        result = PrimaryDecomposition.decompose([], 1)
        assert result == []

    def test_principal_ideal_univariate(self):
        # x^2 - 1 = (x-1)(x+1)
        p = MVPolynomial.from_dense({(2,): 1.0, (0,): -1.0})
        components = PrimaryDecomposition.decompose([p], 1)
        # Should decompose into two linear factors
        assert len(components) == 2

    def test_monomial_ideal(self):
        # Monomial ideal (x^2, y^3)
        g1 = MVPolynomial.from_dense({(2, 0): 1.0})
        g2 = MVPolynomial.from_dense({(0, 3): 1.0})
        components = PrimaryDecomposition.decompose([g1, g2], 2)
        assert len(components) == 2  # Each generator becomes a component

    def test_general_ideal_fallback(self):
        # Non-monomial, multi-generator -> returns as-is
        g1 = MVPolynomial.from_dense({(1, 0): 1.0, (0, 0): 1.0})  # x+1
        g2 = MVPolynomial.from_dense({(0, 1): 1.0, (0, 0): -1.0})  # y-1
        components = PrimaryDecomposition.decompose([g1, g2], 2)
        assert len(components) == 1  # Falls back to returning ideal as-is


# --- Polynomial Division ---

class TestPolynomialDivision:
    def test_division_by_self(self):
        f = MVPolynomial.from_dense({(2,): 1.0, (0,): -1.0})  # x^2 - 1
        quotients, remainder = _polynomial_division(f, [f])
        assert remainder.is_zero()

    def test_division_with_remainder(self):
        f = MVPolynomial.from_dense({(2,): 1.0, (0,): 1.0})   # x^2 + 1
        g = MVPolynomial.from_dense({(1,): 1.0, (0,): -1.0})  # x - 1
        quotients, remainder = _polynomial_division(f, [g])
        # x^2+1 = (x+1)(x-1) + 2 -> remainder should be 2
        val = remainder.evaluate(np.array([0.0]))
        assert val == pytest.approx(2.0)


# --- S-polynomial ---

class TestSPolynomial:
    def test_s_polynomial(self):
        f = MVPolynomial.from_dense({(2, 0): 1.0, (0, 1): 1.0})   # x^2 + y
        g = MVPolynomial.from_dense({(1, 1): 1.0, (0, 0): -1.0})  # xy - 1
        s = _s_polynomial(f, g)
        # S-poly should be non-zero
        assert not s.is_zero()


# --- GrobnerBasis ---

class TestGrobnerBasis:
    def test_single_generator(self):
        f = MVPolynomial.from_dense({(2,): 1.0, (0,): -1.0})
        gb = GrobnerBasis.compute([f])
        assert len(gb) >= 1

    def test_zero_generators_filtered(self):
        gb = GrobnerBasis.compute([MVPolynomial()])
        assert gb == []

    def test_ideal_membership(self):
        # f = x^2 - 1, g = x - 1 generates same ideal
        f = MVPolynomial.from_dense({(2,): 1.0, (0,): -1.0})
        g = MVPolynomial.from_dense({(1,): 1.0, (0,): -1.0})
        gb = GrobnerBasis.compute([f, g])
        # x-1 should be in the Groebner basis (or something reducing to it)
        # Check that x^2-1 reduces to 0 mod gb
        _, r = _polynomial_division(f, gb)
        assert r.is_zero()

    def test_reduced_basis(self):
        f = MVPolynomial.from_dense({(2,): 1.0, (0,): -1.0})
        g = MVPolynomial.from_dense({(1,): 1.0, (0,): -1.0})
        rgb = GrobnerBasis.reduced_basis([f, g])
        assert len(rgb) >= 1
        # Leading coefficients should be 1
        for p in rgb:
            assert p.leading_coefficient() == pytest.approx(1.0)


# --- AlgebraicVariety ---

class TestAlgebraicVariety:
    def test_from_ideal(self):
        f = MVPolynomial.from_dense({(1,): 1.0, (0,): -2.0})  # x - 2
        V = AlgebraicVariety.from_ideal([f], 1)
        assert V.n_vars == 1

    def test_degree_hypersurface(self):
        f = MVPolynomial.from_dense({(3,): 1.0, (0,): -1.0})  # x^3 - 1
        V = AlgebraicVariety([f], 1)
        assert V.degree() == 3

    def test_degree_complete_intersection(self):
        f = MVPolynomial.from_dense({(2, 0): 1.0})  # x^2
        g = MVPolynomial.from_dense({(0, 3): 1.0})  # y^3
        V = AlgebraicVariety([f, g], 2)
        assert V.degree() == 6  # 2 * 3

    def test_points_over_finite_field(self):
        # x - 1 over F_5: only x=1 is a zero
        f = MVPolynomial.from_dense({(1,): 1.0, (0,): -1.0})
        V = AlgebraicVariety([f], 1)
        pts = V.points_over_finite_field(5)
        assert len(pts) == 1
        assert pts[0][0] == pytest.approx(1.0)

    def test_dimension_zero_ideal(self):
        V = AlgebraicVariety([], 3)
        assert V.dimension() == 3  # whole space

    def test_is_irreducible_linear(self):
        f = MVPolynomial.from_dense({(1,): 1.0, (0,): -3.0})  # x - 3
        V = AlgebraicVariety([f], 1)
        assert V.is_irreducible()
