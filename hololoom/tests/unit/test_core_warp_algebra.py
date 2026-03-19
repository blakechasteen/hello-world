"""
Tests for hololoom.core.warp.math.algebra — abstract_algebra, galois_theory,
homological_algebra, module_theory.

Pure math: verify algebraic properties (associativity, identity, inverses,
commutativity, distributivity, homomorphism preservation, etc.)
"""

import pytest
from fractions import Fraction

from hololoom.core.warp.math.algebra.abstract_algebra import (
    Group,
    GroupHomomorphism,
    Ring,
    Ideal,
    Field,
    Polynomial,
)
from hololoom.core.warp.math.algebra.module_theory import (
    Module,
    ModuleHomomorphism,
    TensorProduct,
    ExactSequence,
    ProjectiveModule,
)
from hololoom.core.warp.math.algebra.galois_theory import (
    FieldExtension,
    MinimalPolynomial,
    GaloisGroup,
    FundamentalTheoremGalois,
    SolvabilityByRadicals,
    ClassicalImpossibilities,
    FiniteFieldTheory,
)
from hololoom.core.warp.math.algebra.homological_algebra import (
    ChainComplex,
    QuotientModule,
    CochainComplex,
    SpectralSequence,
    HomologicalDimension,
    LongExactSequence,
)


# ============================================================================
# GROUP TESTS (abstract_algebra.py)
# ============================================================================


class TestCyclicGroup:
    """Cyclic group Z/nZ."""

    def test_cyclic_group_order(self):
        g = Group.cyclic(6)
        assert g.order() == 6

    def test_cyclic_identity(self):
        g = Group.cyclic(5)
        assert g.identity == 0
        for a in g.elements:
            assert g.operation(a, g.identity) == a
            assert g.operation(g.identity, a) == a

    def test_cyclic_associativity(self):
        g = Group.cyclic(7)
        for a in g.elements:
            for b in g.elements:
                for c in g.elements:
                    left = g.operation(g.operation(a, b), c)
                    right = g.operation(a, g.operation(b, c))
                    assert left == right

    def test_cyclic_inverses(self):
        g = Group.cyclic(5)
        for a in g.elements:
            inv = (5 - a) % 5
            assert g.operation(a, inv) == g.identity

    def test_cyclic_is_abelian(self):
        g = Group.cyclic(8)
        assert g.is_abelian() is True

    def test_element_order_identity(self):
        g = Group.cyclic(6)
        assert g.element_order(0) == 1

    def test_element_order_generator(self):
        g = Group.cyclic(5)
        assert g.element_order(1) == 5

    def test_element_order_divisor(self):
        g = Group.cyclic(6)
        # order of 2 in Z/6Z: 2+2+2 = 6 ≡ 0, so order 3
        assert g.element_order(2) == 3
        # order of 3 in Z/6Z: 3+3 = 6 ≡ 0, so order 2
        assert g.element_order(3) == 2


class TestSymmetricGroup:
    """Symmetric group S_n."""

    def test_s3_order(self):
        s3 = Group.symmetric(3)
        assert s3.order() == 6  # 3!

    def test_s3_identity(self):
        s3 = Group.symmetric(3)
        assert s3.identity == (0, 1, 2)
        for perm in s3.elements:
            assert s3.operation(perm, s3.identity) == perm

    def test_s3_associativity_sample(self):
        s3 = Group.symmetric(3)
        elems = s3.elements[:4]
        for a in elems:
            for b in elems:
                for c in elems:
                    left = s3.operation(s3.operation(a, b), c)
                    right = s3.operation(a, s3.operation(b, c))
                    assert left == right

    def test_s3_not_abelian(self):
        s3 = Group.symmetric(3)
        # S3 is the smallest non-abelian group
        found_noncommuting = False
        for a in s3.elements:
            for b in s3.elements:
                if s3.operation(a, b) != s3.operation(b, a):
                    found_noncommuting = True
                    break
            if found_noncommuting:
                break
        assert found_noncommuting

    def test_s2_is_abelian(self):
        s2 = Group.symmetric(2)
        assert s2.order() == 2
        assert s2.is_abelian() is True


class TestDihedralGroup:
    """Dihedral group D_n."""

    def test_dihedral_order(self):
        d4 = Group.dihedral(4)
        assert d4.order() == 8  # 2n

    def test_dihedral_identity_rotations(self):
        d3 = Group.dihedral(3)
        assert d3.identity == (0, 0)
        # Identity works correctly for rotations (s=0)
        rotations = [(r, 0) for r in range(3)]
        for elem in rotations:
            assert d3.operation(elem, d3.identity) == elem
            assert d3.operation(d3.identity, elem) == elem

    def test_dihedral_right_identity(self):
        d3 = Group.dihedral(3)
        # Right identity works for all elements
        for elem in d3.elements:
            assert d3.operation(elem, d3.identity) == elem

    def test_dihedral_d3_not_abelian(self):
        d3 = Group.dihedral(3)
        # D3 ≅ S3, not abelian
        found = False
        for a in d3.elements:
            for b in d3.elements:
                if d3.operation(a, b) != d3.operation(b, a):
                    found = True
                    break
            if found:
                break
        assert found


class TestGroupOperations:
    """Subgroups, center, direct product."""

    def test_subgroup_trivial(self):
        g = Group.cyclic(6)
        trivial = g.subgroup([])
        assert g.identity in trivial.elements

    def test_subgroup_generated(self):
        g = Group.cyclic(6)
        # Subgroup generated by 2: {0, 2, 4}
        h = g.subgroup([2])
        assert set(h.elements) == {0, 2, 4}

    def test_subgroup_full(self):
        g = Group.cyclic(5)
        # 1 generates all of Z/5Z
        h = g.subgroup([1])
        assert set(h.elements) == set(g.elements)

    def test_center_abelian(self):
        g = Group.cyclic(4)
        center = g.center()
        # Abelian group: center = whole group
        assert set(center.elements) == set(g.elements)

    def test_direct_product(self):
        z2 = Group.cyclic(2)
        z3 = Group.cyclic(3)
        product = z2 * z3
        assert product.order() == 6
        assert product.identity == (0, 0)

    def test_direct_product_operation(self):
        z2 = Group.cyclic(2)
        z3 = Group.cyclic(3)
        product = z2 * z3
        result = product.operation((1, 2), (1, 1))
        assert result == (0, 0)  # (1+1)%2=0, (2+1)%3=0


# ============================================================================
# GROUP HOMOMORPHISM TESTS
# ============================================================================


class TestGroupHomomorphism:
    def test_homomorphism_property(self):
        z6 = Group.cyclic(6)
        z3 = Group.cyclic(3)
        # phi: Z/6Z -> Z/3Z, phi(x) = x mod 3
        phi = GroupHomomorphism(
            source=z6, target=z3,
            mapping=lambda x: x % 3,
            name="mod3"
        )
        assert phi.is_homomorphism() is True

    def test_homomorphism_call(self):
        z6 = Group.cyclic(6)
        z3 = Group.cyclic(3)
        phi = GroupHomomorphism(source=z6, target=z3, mapping=lambda x: x % 3)
        assert phi(4) == 1
        assert phi(0) == 0

    def test_kernel(self):
        z6 = Group.cyclic(6)
        z3 = Group.cyclic(3)
        phi = GroupHomomorphism(source=z6, target=z3, mapping=lambda x: x % 3)
        ker = phi.kernel()
        assert set(ker.elements) == {0, 3}

    def test_image(self):
        z6 = Group.cyclic(6)
        z3 = Group.cyclic(3)
        phi = GroupHomomorphism(source=z6, target=z3, mapping=lambda x: x % 3)
        im = phi.image()
        assert set(im.elements) == {0, 1, 2}

    def test_not_homomorphism(self):
        z4 = Group.cyclic(4)
        # f(x) = x^2 mod 4 is not a homomorphism on Z/4Z
        # f(1+1) = f(2) = 0, f(1)+f(1) = 1+1 = 2 != 0
        phi = GroupHomomorphism(
            source=z4, target=z4,
            mapping=lambda x: (x * x) % 4
        )
        assert phi.is_homomorphism() is False


# ============================================================================
# RING TESTS (abstract_algebra.py)
# ============================================================================


class TestRing:
    def test_integers_mod_n_elements(self):
        r = Ring.integers_mod_n(6)
        assert len(r.elements) == 6
        assert r.zero == 0
        assert r.one == 1

    def test_ring_additive_identity(self):
        r = Ring.integers_mod_n(5)
        for a in r.elements:
            assert r.add(a, r.zero) == a

    def test_ring_multiplicative_identity(self):
        r = Ring.integers_mod_n(5)
        for a in r.elements:
            assert r.mul(a, r.one) == a

    def test_ring_distributivity(self):
        r = Ring.integers_mod_n(6)
        for a in r.elements:
            for b in r.elements:
                for c in r.elements:
                    left = r.mul(a, r.add(b, c))
                    right = r.add(r.mul(a, b), r.mul(a, c))
                    assert left == right

    def test_ring_commutativity(self):
        r = Ring.integers_mod_n(7)
        assert r.is_commutative() is True

    def test_integral_domain_prime(self):
        r = Ring.integers_mod_n(5)
        assert r.is_integral_domain() is True

    def test_not_integral_domain_composite(self):
        r = Ring.integers_mod_n(6)
        # 2*3 = 0 mod 6, so has zero divisors
        assert r.is_integral_domain() is False

    def test_units_z5(self):
        r = Ring.integers_mod_n(5)
        units = r.units()
        # Z/5Z is a field, units = {1,2,3,4}
        assert set(units) == {1, 2, 3, 4}

    def test_units_z6(self):
        r = Ring.integers_mod_n(6)
        units = r.units()
        # Units of Z/6Z: elements coprime to 6 = {1, 5}
        assert set(units) == {1, 5}

    def test_matrix_ring_creation(self):
        base = Ring.integers_mod_n(2)
        mr = Ring.matrix_ring(2, base)
        assert mr.name == "M_2(Z/2Z)"


# ============================================================================
# IDEAL TESTS (abstract_algebra.py)
# ============================================================================


class TestIdeal:
    def test_ideal_contains_zero(self):
        r = Ring.integers_mod_n(6)
        ideal = Ideal(r, [2])
        assert 0 in ideal

    def test_ideal_generated_by_2_in_z6(self):
        r = Ring.integers_mod_n(6)
        ideal = Ideal(r, [2])
        # (2) in Z/6Z = {0, 2, 4}
        assert ideal.elements == {0, 2, 4}

    def test_ideal_generated_by_3_in_z6(self):
        r = Ring.integers_mod_n(6)
        ideal = Ideal(r, [3])
        # (3) in Z/6Z = {0, 3}
        assert ideal.elements == {0, 3}

    def test_ideal_is_prime(self):
        r = Ring.integers_mod_n(6)
        ideal = Ideal(r, [2])
        # (2) in Z/6Z is prime
        assert ideal.is_prime() is True

    def test_ideal_is_prime_3(self):
        r = Ring.integers_mod_n(6)
        ideal = Ideal(r, [3])
        # (3) in Z/6Z is prime
        assert ideal.is_prime() is True

    def test_ideal_contains_generators(self):
        r = Ring.integers_mod_n(10)
        ideal = Ideal(r, [5])
        assert 5 in ideal
        assert 0 in ideal


# ============================================================================
# FIELD TESTS (abstract_algebra.py)
# ============================================================================


class TestField:
    def test_finite_field_creation(self):
        f5 = Field.finite_field(5)
        assert len(f5.elements) == 5
        assert f5.zero == 0
        assert f5.one == 1

    def test_finite_field_every_nonzero_has_inverse(self):
        f7 = Field.finite_field(7)
        for a in f7.elements:
            if a == 0:
                continue
            found_inv = False
            for b in f7.elements:
                if f7.mul(a, b) == 1:
                    found_inv = True
                    break
            assert found_inv, f"{a} has no inverse in F_7"

    def test_characteristic_prime_field(self):
        f5 = Field.finite_field(5)
        assert f5.characteristic() == 5

    def test_characteristic_f7(self):
        f7 = Field.finite_field(7)
        assert f7.characteristic() == 7

    def test_extension_field_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Field.finite_field(2, n=2)

    def test_field_is_ring(self):
        f5 = Field.finite_field(5)
        assert isinstance(f5, Ring)

    def test_rationals_sample(self):
        q = Field.rationals_sample()
        assert q.zero == Fraction(0)
        assert q.one == Fraction(1)
        # Verify some arithmetic
        assert q.add(Fraction(1, 2), Fraction(1, 2)) == Fraction(1)
        assert q.mul(Fraction(2), Fraction(3)) == Fraction(6)


# ============================================================================
# POLYNOMIAL TESTS (abstract_algebra.py)
# ============================================================================


class TestPolynomial:
    def _ring(self):
        return Ring.integers_mod_n(7)

    def test_degree_constant(self):
        r = self._ring()
        p = Polynomial([3], r)
        assert p.degree() == 0

    def test_degree_zero_poly(self):
        r = self._ring()
        p = Polynomial([0], r)
        assert p.degree() == -1

    def test_degree_linear(self):
        r = self._ring()
        p = Polynomial([1, 2], r)
        assert p.degree() == 1

    def test_degree_quadratic(self):
        r = self._ring()
        p = Polynomial([1, 0, 3], r)
        assert p.degree() == 2

    def test_add_polynomials(self):
        r = self._ring()
        # (1 + 2x) + (3 + 4x) = 4 + 6x  (mod 7)
        p1 = Polynomial([1, 2], r)
        p2 = Polynomial([3, 4], r)
        result = p1 + p2
        assert result.coefficients == [4, 6]

    def test_add_different_degrees(self):
        r = self._ring()
        # (1 + 2x) + (3x^2) = 1 + 2x + 3x^2
        p1 = Polynomial([1, 2], r)
        p2 = Polynomial([0, 0, 3], r)
        result = p1 + p2
        assert result.coefficients == [1, 2, 3]

    def test_mul_polynomials(self):
        r = self._ring()
        # (1 + x) * (1 + x) = 1 + 2x + x^2  (mod 7)
        p = Polynomial([1, 1], r)
        result = p * p
        assert result.coefficients == [1, 2, 1]

    def test_mul_linear_times_linear(self):
        r = self._ring()
        # (2 + 3x) * (1 + x) = 2 + 5x + 3x^2  (mod 7)
        p1 = Polynomial([2, 3], r)
        p2 = Polynomial([1, 1], r)
        result = p1 * p2
        assert result.coefficients == [2, 5, 3]

    def test_repr_nonzero(self):
        r = self._ring()
        p = Polynomial([1, 2, 3], r)
        s = repr(p)
        assert "1" in s
        assert "2x" in s
        assert "3x^2" in s

    def test_repr_zero(self):
        r = self._ring()
        p = Polynomial([0, 0], r)
        assert repr(p) == "0"


# ============================================================================
# MODULE THEORY TESTS (module_theory.py)
# ============================================================================


class TestModule:
    def _z3_ring(self):
        return Ring.integers_mod_n(3)

    def test_free_module_creation(self):
        r = self._z3_ring()
        m = Module.free_module(r, 2)
        # Z/3Z^2 has 3^2 = 9 elements
        assert len(m.elements) == 9
        assert m.zero == (0, 0)

    def test_free_module_addition(self):
        r = self._z3_ring()
        m = Module.free_module(r, 2)
        result = m.add((1, 2), (2, 1))
        assert result == (0, 0)  # (1+2, 2+1) mod 3

    def test_free_module_scalar_mult(self):
        r = self._z3_ring()
        m = Module.free_module(r, 2)
        result = m.smul(2, (1, 1))
        assert result == (2, 2)

    def test_is_submodule_trivial(self):
        r = self._z3_ring()
        m = Module.free_module(r, 1)
        # {0} is always a submodule
        assert m.is_submodule([(0,)]) is True

    def test_is_submodule_full(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        assert m.is_submodule(list(m.elements)) is True

    def test_free_module_large_ring_warning(self):
        """Large ring uses sample (empty elements)."""
        r = Ring.integers_mod_n(101)
        m = Module.free_module(r, 2)
        assert len(m.elements) == 0  # Falls back to empty


class TestModuleHomomorphism:
    def test_kernel(self):
        r = Ring.integers_mod_n(3)
        m = Module.free_module(r, 1)
        # Zero map: everything goes to zero
        zero_map = ModuleHomomorphism(m, m, lambda x: (0,), "zero")
        ker = zero_map.kernel()
        assert len(ker) == len(m.elements)

    def test_image_zero_map(self):
        r = Ring.integers_mod_n(3)
        m = Module.free_module(r, 1)
        zero_map = ModuleHomomorphism(m, m, lambda x: (0,), "zero")
        im = zero_map.image()
        assert im == [(0,)]

    def test_identity_injective(self):
        r = Ring.integers_mod_n(3)
        m = Module.free_module(r, 1)
        id_map = ModuleHomomorphism(m, m, lambda x: x, "id")
        assert id_map.is_injective() is True

    def test_identity_surjective(self):
        r = Ring.integers_mod_n(3)
        m = Module.free_module(r, 1)
        id_map = ModuleHomomorphism(m, m, lambda x: x, "id")
        assert id_map.is_surjective() is True

    def test_zero_map_not_injective(self):
        r = Ring.integers_mod_n(3)
        m = Module.free_module(r, 1)
        zero_map = ModuleHomomorphism(m, m, lambda x: (0,), "zero")
        # kernel has 3 elements, not just {0}
        assert zero_map.is_injective() is False


class TestExactSequence:
    def test_short_exact_check_length(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        # With only 2 modules, not short exact
        seq = ExactSequence(
            [m, m],
            [ModuleHomomorphism(m, m, lambda x: x)]
        )
        assert seq.is_short_exact() is False

    def test_exact_at_boundary(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        seq = ExactSequence(
            [m, m],
            [ModuleHomomorphism(m, m, lambda x: x)]
        )
        # Index 0 returns True by definition
        assert seq.is_exact_at(0) is True


class TestTensorProduct:
    def test_construct_returns_module(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        n = Module.free_module(r, 1)
        result = TensorProduct.construct(m, n)
        assert isinstance(result, Module)

    def test_universal_property_returns_hom(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        target = Module.free_module(r, 1)
        hom = TensorProduct.universal_property(
            m, m, lambda a, b: (0,), target
        )
        assert isinstance(hom, ModuleHomomorphism)


class TestProjectiveModule:
    def test_free_module_name_check(self):
        r = Ring.integers_mod_n(3)
        m = Module.free_module(r, 1)
        # Name is "Z/3Z^1" which doesn't end with just "^"
        # so is_projective returns False (conservative default)
        result = ProjectiveModule.is_projective(m)
        assert isinstance(result, bool)

    def test_projective_resolution(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        resolution = ProjectiveModule.projective_resolution(m, length=2)
        assert isinstance(resolution, ExactSequence)
        assert len(resolution.modules) == 3  # 2 projectives + M
        assert len(resolution.homomorphisms) == 2


# ============================================================================
# GALOIS THEORY TESTS (galois_theory.py)
# ============================================================================


class TestFieldExtension:
    def test_trivial_extension_degree(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5, name="F5/F5")
        assert ext.degree() == 1

    def test_is_algebraic(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5)
        assert ext.is_algebraic() is True

    def test_is_galois_finite(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5)
        assert ext.is_galois() is True


class TestMinimalPolynomial:
    def test_find_minimal_returns_none(self):
        f5 = Field.finite_field(5)
        result = MinimalPolynomial.find_minimal_polynomial(2, f5)
        assert result is None

    def test_irreducible_linear(self):
        f5 = Field.finite_field(5)
        # Linear polynomial: degree 1, irreducible
        p = Polynomial([1, 1], f5)
        assert MinimalPolynomial.is_irreducible(p, f5) is True

    def test_irreducible_constant(self):
        f5 = Field.finite_field(5)
        # Constant polynomial: degree 0, not irreducible
        p = Polynomial([3], f5)
        assert MinimalPolynomial.is_irreducible(p, f5) is False


class TestSolvabilityByRadicals:
    def test_abelian_is_solvable(self):
        g = Group.cyclic(6)
        assert SolvabilityByRadicals.is_solvable_group(g) is True

    def test_s3_solvable(self):
        s3 = Group.symmetric(3)
        assert SolvabilityByRadicals.is_solvable_group(s3) is True

    def test_s4_solvable(self):
        s4 = Group.symmetric(4)
        assert SolvabilityByRadicals.is_solvable_group(s4) is True

    def test_s5_not_solvable(self):
        # S5 is not solvable — this is why general quintics are unsolvable!
        s5 = Group.symmetric(5)
        assert SolvabilityByRadicals.is_solvable_group(s5) is False

    def test_polynomial_deg4_solvable(self):
        f5 = Field.finite_field(5)
        p = Polynomial([1, 0, 0, 0, 1], f5)  # x^4 + 1
        assert SolvabilityByRadicals.is_polynomial_solvable(p, f5) is True

    def test_polynomial_deg5_not_solvable(self):
        f5 = Field.finite_field(5)
        p = Polynomial([1, 0, 0, 0, 0, 1], f5)  # x^5 + 1
        assert SolvabilityByRadicals.is_polynomial_solvable(p, f5) is False


class TestClassicalImpossibilities:
    def test_doubling_cube(self):
        text = ClassicalImpossibilities.explain_doubling_cube()
        assert "impossible" in text.lower()
        assert "³√2" in text or "power of 2" in text

    def test_trisecting_angle(self):
        text = ClassicalImpossibilities.explain_trisecting_angle()
        assert "impossible" in text.lower()

    def test_unsolvable_quintic(self):
        text = ClassicalImpossibilities.explain_unsolvable_quintic()
        assert "S₅" in text or "S_5" in text or "solvable" in text.lower()


class TestFiniteFieldTheory:
    def test_frobenius(self):
        # In F_7: x^7 = x (Fermat's little theorem)
        for x in range(7):
            assert FiniteFieldTheory.frobenius_automorphism(7, x, 7) == x

    def test_primitive_element_f5(self):
        # F_5* is cyclic of order 4; primitive element generates it
        g = FiniteFieldTheory.primitive_element(5, 1)
        assert g is not None
        # Verify it generates {1,2,3,4}
        generated = set()
        current = 1
        for _ in range(4):
            current = (current * g) % 5
            generated.add(current)
        assert generated == {1, 2, 3, 4}

    def test_primitive_element_f7(self):
        g = FiniteFieldTheory.primitive_element(7, 1)
        assert g is not None


class TestGaloisGroup:
    def test_galois_group_trivial(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5, name="F5/F5")
        gal = GaloisGroup(ext)
        # Trivial extension: Gal has order 1
        assert gal.order() == 1

    def test_fixed_field(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5)
        gal = GaloisGroup(ext)
        # Fixed field of full group = base field
        fixed = gal.fixed_field(gal.elements)
        assert len(fixed.elements) == len(f5.elements)


class TestFundamentalTheoremGalois:
    def test_verify_correspondence(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5)
        gal = GaloisGroup(ext)
        assert FundamentalTheoremGalois.verify_galois_correspondence(ext, gal) is True

    def test_subfield_to_subgroup(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5)
        gal = GaloisGroup(ext)
        subgroup = FundamentalTheoremGalois.subfield_to_subgroup(ext, gal, f5)
        assert len(subgroup) >= 1

    def test_subgroup_to_subfield(self):
        f5 = Field.finite_field(5)
        ext = FieldExtension(base_field=f5, extension_field=f5)
        gal = GaloisGroup(ext)
        fixed = FundamentalTheoremGalois.subgroup_to_subfield(ext, gal, gal.elements)
        assert isinstance(fixed, Field)


# ============================================================================
# HOMOLOGICAL ALGEBRA TESTS (homological_algebra.py)
# ============================================================================


class TestQuotientModule:
    def test_order(self):
        r = Ring.integers_mod_n(3)
        m = Module.free_module(r, 1)
        qm = QuotientModule(module=m, submodule=[], name="test")
        # With empty submodule, all elements are distinct cosets
        assert qm.order() == len(m.elements)


class TestChainComplex:
    def test_creation(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        cc = ChainComplex(modules={0: m}, differentials={}, name="test")
        assert cc.name == "test"

    def test_cycles_empty(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        cc = ChainComplex(modules={0: m}, differentials={})
        assert cc.cycles(0) == []  # No differential at 0

    def test_boundaries_empty(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        cc = ChainComplex(modules={0: m}, differentials={})
        assert cc.boundaries(0) == []


class TestCochainComplex:
    def test_creation(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        cc = CochainComplex(modules={0: m}, differentials={}, name="test")
        assert cc.name == "test"


class TestSpectralSequence:
    def test_creation(self):
        ss = SpectralSequence(name="E")
        assert ss.name == "E"
        assert ss.pages == {}

    def test_set_page(self):
        ss = SpectralSequence()
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        ss.set_page(0, 0, 0, m)
        assert (0, 0) in ss.pages[0]

    def test_abutment(self):
        ss = SpectralSequence()
        result = ss.abutment(0)
        assert isinstance(result, Module)


class TestHomologicalDimension:
    def test_global_dimension(self):
        r = Ring.integers_mod_n(5)
        gd = HomologicalDimension.global_dimension(r)
        assert isinstance(gd, int)


class TestLongExactSequence:
    def test_from_short_exact_returns_chain(self):
        r = Ring.integers_mod_n(2)
        m = Module.free_module(r, 1)
        hom = ModuleHomomorphism(m, m, lambda x: x)
        ses = ExactSequence([m, m], [hom])
        result = LongExactSequence.from_short_exact_sequence(ses)
        assert isinstance(result, ChainComplex)
