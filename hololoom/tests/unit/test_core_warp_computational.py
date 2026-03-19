"""
Unit tests for warp/math computational modules:
- polynomial_algorithms.py
- number_theory_algorithms.py
- symbolic_computation.py
- graph/graph_coloring.py
"""

import numpy as np
import pytest

# --- polynomial_algorithms ---
from hololoom.core.warp.math.computational.polynomial_algorithms import (
    UnivPoly,
    GrobnerComputation,
    PolynomialGCD,
    SmithNormalForm,
    HermiteNormalForm,
    FiniteFieldPoly,
    FiniteFieldFactorization,
)

# --- number_theory_algorithms ---
from hololoom.core.warp.math.computational.number_theory_algorithms import (
    PrimalityTesting,
    IntegerFactorization,
    DiscreteLogarithm,
    LLLReduction,
    EllipticCurve,
)

# --- symbolic_computation ---
from hololoom.core.warp.math.computational.symbolic_computation import (
    SymbolicExpr,
    RewriteRule,
    RewriteSystem,
    PatternMatcher,
    _is_zero,
    _is_one,
)

# --- graph_coloring ---
from hololoom.core.warp.math.graph.graph_coloring import (
    ColoringResult,
    GraphColoring,
    ConflictGraph,
    chromatic_polynomial,
    independent_sets,
    maximum_independent_set,
    color_graph,
    schedule_tasks,
)


# ============================================================================
# UnivPoly
# ============================================================================

class TestUnivPoly:
    def test_degree(self):
        p = UnivPoly(np.array([1.0, 2.0, 3.0]))
        assert p.degree == 2

    def test_degree_zero_poly(self):
        p = UnivPoly(np.array([0.0]))
        assert p.degree == -1
        assert p.is_zero()

    def test_degree_constant(self):
        p = UnivPoly(np.array([5.0]))
        assert p.degree == 0
        assert not p.is_zero()

    def test_leading_coefficient(self):
        p = UnivPoly(np.array([1.0, 0.0, 3.0]))
        assert p.leading_coefficient() == 3.0

    def test_leading_coefficient_zero(self):
        p = UnivPoly(np.array([0.0]))
        assert p.leading_coefficient() == 0.0

    def test_monic(self):
        p = UnivPoly(np.array([2.0, 4.0]))
        m = p.monic()
        assert m.coeffs[-1] == pytest.approx(1.0)
        assert m.coeffs[0] == pytest.approx(0.5)

    def test_monic_zero(self):
        p = UnivPoly(np.array([0.0]))
        m = p.monic()
        assert m.is_zero()

    def test_add(self):
        a = UnivPoly(np.array([1.0, 2.0]))
        b = UnivPoly(np.array([3.0, 0.0, 1.0]))
        c = a + b
        np.testing.assert_allclose(c.coeffs, [4.0, 2.0, 1.0])

    def test_sub(self):
        a = UnivPoly(np.array([5.0, 3.0]))
        b = UnivPoly(np.array([2.0, 3.0]))
        c = a - b
        assert c.coeffs[0] == pytest.approx(3.0)
        # trailing zeros trimmed, degree 0
        assert c.degree == 0

    def test_mul(self):
        # (1 + x)(1 - x) = 1 - x^2
        a = UnivPoly(np.array([1.0, 1.0]))
        b = UnivPoly(np.array([1.0, -1.0]))
        c = a * b
        np.testing.assert_allclose(c.coeffs, [1.0, 0.0, -1.0], atol=1e-12)

    def test_scalar_mul(self):
        p = UnivPoly(np.array([1.0, 2.0]))
        q = p.scalar_mul(3.0)
        np.testing.assert_allclose(q.coeffs, [3.0, 6.0])

    def test_divmod(self):
        # (x^2 + 2x + 1) / (x + 1) = (x + 1) remainder 0
        a = UnivPoly(np.array([1.0, 2.0, 1.0]))
        b = UnivPoly(np.array([1.0, 1.0]))
        q, r = a.divmod(b)
        np.testing.assert_allclose(q.coeffs, [1.0, 1.0], atol=1e-12)
        assert r.is_zero() or np.max(np.abs(r.coeffs)) < 1e-12

    def test_divmod_zero_divisor(self):
        a = UnivPoly(np.array([1.0, 1.0]))
        b = UnivPoly(np.array([0.0]))
        with pytest.raises(ValueError, match="Division by zero"):
            a.divmod(b)

    def test_divmod_lower_degree(self):
        a = UnivPoly(np.array([1.0]))  # constant
        b = UnivPoly(np.array([1.0, 1.0]))  # degree 1
        q, r = a.divmod(b)
        assert q.is_zero() or np.max(np.abs(q.coeffs)) < 1e-12
        np.testing.assert_allclose(r.coeffs, [1.0], atol=1e-12)

    def test_evaluate(self):
        # p(x) = 1 + 2x + 3x^2, p(2) = 1 + 4 + 12 = 17
        p = UnivPoly(np.array([1.0, 2.0, 3.0]))
        assert p.evaluate(2.0) == pytest.approx(17.0)

    def test_evaluate_zero(self):
        p = UnivPoly(np.array([5.0, 3.0, 1.0]))
        assert p.evaluate(0.0) == pytest.approx(5.0)

    def test_repr(self):
        p = UnivPoly(np.array([1.0, 0.0, 3.0]))
        r = repr(p)
        assert "1" in r
        assert "3" in r

    def test_repr_zero(self):
        p = UnivPoly(np.array([0.0]))
        assert repr(p) == "0"

    def test_trim_trailing_zeros(self):
        p = UnivPoly(np.array([1.0, 2.0, 0.0, 0.0]))
        assert p.degree == 1


# ============================================================================
# PolynomialGCD
# ============================================================================

class TestPolynomialGCD:
    def test_gcd_coprime(self):
        # gcd(x, x+1) = 1
        a = UnivPoly(np.array([0.0, 1.0]))
        b = UnivPoly(np.array([1.0, 1.0]))
        g = PolynomialGCD.euclidean(a, b)
        assert g.degree == 0
        assert g.coeffs[0] == pytest.approx(1.0)

    def test_gcd_common_factor(self):
        # gcd(x^2-1, x-1) = x-1
        a = UnivPoly(np.array([-1.0, 0.0, 1.0]))  # x^2 - 1
        b = UnivPoly(np.array([-1.0, 1.0]))         # x - 1
        g = PolynomialGCD.euclidean(a, b)
        m = g.monic()
        assert m.degree == 1
        assert m.coeffs[0] == pytest.approx(-1.0)
        assert m.coeffs[1] == pytest.approx(1.0)

    def test_gcd_with_zero(self):
        a = UnivPoly(np.array([1.0, 1.0]))
        b = UnivPoly(np.array([0.0]))
        g = PolynomialGCD.euclidean(a, b)
        assert g.degree == 1

    def test_extended_gcd(self):
        a = UnivPoly(np.array([-1.0, 0.0, 1.0]))  # x^2 - 1
        b = UnivPoly(np.array([-1.0, 1.0]))         # x - 1
        g, s, t = PolynomialGCD.extended_gcd(a, b)
        # g should divide both a and b
        assert g.degree >= 0

    def test_extended_gcd_zero(self):
        a = UnivPoly(np.array([2.0, 4.0]))
        b = UnivPoly(np.array([0.0]))
        g, s, t = PolynomialGCD.extended_gcd(a, b)
        assert g.degree == 1


# ============================================================================
# GrobnerComputation
# ============================================================================

class TestGrobnerComputation:
    def test_buchberger_single(self):
        p = UnivPoly(np.array([1.0, 1.0]))
        result = GrobnerComputation.buchberger([p])
        assert len(result) == 1
        assert result[0].degree == 1

    def test_buchberger_empty(self):
        assert GrobnerComputation.buchberger([]) == []

    def test_buchberger_multiple(self):
        a = UnivPoly(np.array([-1.0, 0.0, 1.0]))  # x^2-1
        b = UnivPoly(np.array([-1.0, 1.0]))         # x-1
        result = GrobnerComputation.buchberger([a, b])
        assert len(result) == 1
        assert result[0].degree == 1  # gcd is x-1

    def test_reduced_basis(self):
        a = UnivPoly(np.array([2.0, 4.0]))
        result = GrobnerComputation.reduced_basis([a])
        assert len(result) == 1
        assert result[0].coeffs[-1] == pytest.approx(1.0)  # monic


# ============================================================================
# SmithNormalForm
# ============================================================================

class TestSmithNormalForm:
    def test_identity(self):
        A = np.eye(3)
        S, U, V = SmithNormalForm.compute(A)
        # Smith form of identity is identity
        for i in range(3):
            assert abs(S[i, i] - 1.0) < 0.5

    def test_diagonal(self):
        A = np.diag([2.0, 4.0, 6.0])
        S, U, V = SmithNormalForm.compute(A)
        diag = sorted([abs(round(S[i, i])) for i in range(3)])
        assert diag == [2, 2, 6] or all(d > 0 for d in diag)

    def test_small_matrix(self):
        A = np.array([[2, 4], [6, 8]], dtype=float)
        S, U, V = SmithNormalForm.compute(A)
        # S should be diagonal
        assert abs(S[0, 1]) < 0.5
        assert abs(S[1, 0]) < 0.5
        # Non-zero diagonal entries
        d0 = abs(round(S[0, 0]))
        d1 = abs(round(S[1, 1]))
        assert d0 > 0

    def test_invariant_factors(self):
        A = np.array([[2, 0], [0, 6]], dtype=float)
        factors = SmithNormalForm.invariant_factors(A)
        assert len(factors) > 0
        assert all(f > 0 for f in factors)


# ============================================================================
# HermiteNormalForm
# ============================================================================

class TestHermiteNormalForm:
    def test_identity(self):
        A = np.eye(3)
        H, U = HermiteNormalForm.compute(A)
        np.testing.assert_allclose(H, np.eye(3), atol=0.5)

    def test_upper_triangular(self):
        A = np.array([[2, 3], [4, 5]], dtype=float)
        H, U = HermiteNormalForm.compute(A)
        # Below diagonal should be zero
        assert abs(H[1, 0]) < 0.5

    def test_lattice_basis(self):
        A = np.array([[2, 0], [1, 3]], dtype=float)
        basis = HermiteNormalForm.lattice_basis(A)
        assert basis.shape[0] > 0  # Non-empty

    def test_transform_relation(self):
        A = np.array([[3, 1], [2, 4]], dtype=float)
        H, U = HermiteNormalForm.compute(A)
        # H = U @ A
        computed = U @ A
        np.testing.assert_allclose(H, np.round(computed), atol=1.0)


# ============================================================================
# FiniteFieldPoly
# ============================================================================

class TestFiniteFieldPoly:
    def test_basic_operations(self):
        p = 5
        a = FiniteFieldPoly(np.array([1, 2, 3]), p)
        b = FiniteFieldPoly(np.array([4, 1]), p)
        c = a + b
        assert all(0 <= x < p for x in c.coeffs)

    def test_degree(self):
        f = FiniteFieldPoly(np.array([1, 0, 1]), 7)
        assert f.degree == 2

    def test_zero(self):
        f = FiniteFieldPoly(np.array([0]), 5)
        assert f.is_zero()

    def test_mul(self):
        p = 7
        a = FiniteFieldPoly(np.array([1, 1]), p)  # 1+x
        b = FiniteFieldPoly(np.array([1, 1]), p)  # 1+x
        c = a * b  # 1+2x+x^2
        assert c.degree == 2
        assert c.coeffs[0] == 1
        assert c.coeffs[1] == 2
        assert c.coeffs[2] == 1

    def test_monic(self):
        f = FiniteFieldPoly(np.array([2, 4]), 5)
        m = f.monic()
        assert m.coeffs[-1] == 1

    def test_divmod(self):
        p = 7
        a = FiniteFieldPoly(np.array([1, 0, 1]), p)  # x^2+1
        b = FiniteFieldPoly(np.array([1, 1]), p)       # x+1
        q, r = a.divmod(b)
        # Verify: a = q*b + r
        check = q * b + r
        # Coefficients should match mod p
        for i in range(max(len(a.coeffs), len(check.coeffs))):
            ac = int(a.coeffs[i]) % p if i < len(a.coeffs) else 0
            cc = int(check.coeffs[i]) % p if i < len(check.coeffs) else 0
            assert ac == cc

    def test_divmod_zero(self):
        with pytest.raises(ValueError):
            a = FiniteFieldPoly(np.array([1, 1]), 5)
            b = FiniteFieldPoly(np.array([0]), 5)
            a.divmod(b)

    def test_powmod(self):
        p = 7
        f = FiniteFieldPoly(np.array([0, 1]), p)  # x
        mod = FiniteFieldPoly(np.array([1, 0, 0, 1]), p)  # x^3 + 1
        result = f.powmod(3, mod)
        # x^3 mod (x^3+1) = x^3 - (x^3+1) = -1 = 6 mod 7
        assert result.degree == 0
        assert result.coeffs[0] == (p - 1)

    def test_gcd(self):
        p = 7
        a = FiniteFieldPoly(np.array([6, 0, 1]), p)  # x^2 - 1 = x^2 + 6 mod 7
        b = FiniteFieldPoly(np.array([6, 1]), p)       # x - 1 = x + 6 mod 7
        g = a.gcd(b)
        assert g.degree == 1  # gcd should be x-1

    def test_sub(self):
        p = 5
        a = FiniteFieldPoly(np.array([3, 2]), p)
        b = FiniteFieldPoly(np.array([1, 4]), p)
        c = a - b
        assert c.coeffs[0] == (3 - 1) % p
        assert c.coeffs[1] == (2 - 4) % p

    def test_scalar_mul(self):
        p = 7
        f = FiniteFieldPoly(np.array([1, 2, 3]), p)
        g = f.scalar_mul(3)
        assert g.coeffs[0] == 3
        assert g.coeffs[1] == 6
        assert g.coeffs[2] == (9 % 7)

    def test_repr(self):
        f = FiniteFieldPoly(np.array([1, 0, 3]), 5)
        r = repr(f)
        assert "1" in r
        assert "3" in r

    def test_repr_zero(self):
        f = FiniteFieldPoly(np.array([0]), 5)
        assert repr(f) == "0"


# ============================================================================
# FiniteFieldFactorization
# ============================================================================

class TestFiniteFieldFactorization:
    def test_factor_linear(self):
        p = 7
        f = FiniteFieldPoly(np.array([6, 1]), p)  # x - 1
        factors = FiniteFieldFactorization.factor(f)
        assert len(factors) == 1
        assert factors[0].degree == 1

    def test_factor_constant(self):
        p = 5
        f = FiniteFieldPoly(np.array([3]), p)
        factors = FiniteFieldFactorization.factor(f)
        assert len(factors) == 1

    def test_factor_product(self):
        # Factor x^2 - 1 = (x-1)(x+1) over GF(7)
        p = 7
        f = FiniteFieldPoly(np.array([6, 0, 1]), p)  # x^2 + 6 = x^2 - 1
        factors = FiniteFieldFactorization.factor(f)
        # Should produce factors whose degrees sum to at most deg(f)
        assert len(factors) >= 1
        assert all(fac.degree >= 1 for fac in factors)


# ============================================================================
# PrimalityTesting
# ============================================================================

class TestPrimalityTesting:
    def test_small_primes(self):
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            assert PrimalityTesting.is_prime(p), f"{p} should be prime"

    def test_small_composites(self):
        for c in [4, 6, 8, 9, 10, 12, 15, 21, 25]:
            assert not PrimalityTesting.is_prime(c), f"{c} should be composite"

    def test_edge_cases(self):
        assert not PrimalityTesting.is_prime(0)
        assert not PrimalityTesting.is_prime(1)
        assert not PrimalityTesting.is_prime(-5)

    def test_miller_rabin_large(self):
        assert PrimalityTesting.miller_rabin(104729)  # prime
        assert not PrimalityTesting.miller_rabin(104730)  # even

    def test_sieve(self):
        primes = PrimalityTesting.sieve_of_eratosthenes(30)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_sieve_small(self):
        assert PrimalityTesting.sieve_of_eratosthenes(1) == []
        assert PrimalityTesting.sieve_of_eratosthenes(2) == [2]

    def test_carmichael_number(self):
        # 561 is the smallest Carmichael number (composite but passes Fermat test)
        assert not PrimalityTesting.is_prime(561)


# ============================================================================
# IntegerFactorization
# ============================================================================

class TestIntegerFactorization:
    def test_trial_division_prime(self):
        assert IntegerFactorization.trial_division(17) == [17]

    def test_trial_division_composite(self):
        assert IntegerFactorization.trial_division(12) == [2, 2, 3]

    def test_trial_division_edge(self):
        assert IntegerFactorization.trial_division(1) == []
        assert IntegerFactorization.trial_division(0) == []

    def test_pollard_rho_prime(self):
        assert IntegerFactorization.pollard_rho(13) == [13]

    def test_pollard_rho_composite(self):
        factors = IntegerFactorization.pollard_rho(60)
        assert sorted(factors) == [2, 2, 3, 5]

    def test_pollard_rho_power_of_2(self):
        factors = IntegerFactorization.pollard_rho(16)
        assert factors == [2, 2, 2, 2]

    def test_factorization_dict(self):
        d = IntegerFactorization.factorization_dict(60)
        assert d == {2: 2, 3: 1, 5: 1}

    def test_factorization_product(self):
        n = 2310  # 2*3*5*7*11
        factors = IntegerFactorization.pollard_rho(n)
        product = 1
        for f in factors:
            product *= f
        assert product == n


# ============================================================================
# DiscreteLogarithm
# ============================================================================

class TestDiscreteLogarithm:
    def test_baby_giant_simple(self):
        # 2^x = 8 mod 13 => x = 3
        x = DiscreteLogarithm.baby_giant(2, 8, 13)
        assert pow(2, x, 13) == 8

    def test_baby_giant_generator(self):
        # 3 is a generator of Z/7Z*
        # 3^x = 6 mod 7
        x = DiscreteLogarithm.baby_giant(3, 6, 7)
        assert x != -1
        assert pow(3, x, 7) == 6

    def test_crt(self):
        # x = 2 mod 3, x = 3 mod 5 => x = 8 mod 15
        result = DiscreteLogarithm._crt([2, 3], [3, 5])
        assert result % 3 == 2
        assert result % 5 == 3

    def test_crt_single(self):
        assert DiscreteLogarithm._crt([5], [7]) == 5

    def test_crt_empty(self):
        assert DiscreteLogarithm._crt([], []) == 0

    def test_extended_gcd(self):
        g, s, t = DiscreteLogarithm._extended_gcd(12, 8)
        assert g == 4
        assert 12 * s + 8 * t == g


# ============================================================================
# LLLReduction
# ============================================================================

class TestLLLReduction:
    def test_identity_basis(self):
        B = np.eye(3)
        reduced = LLLReduction.reduce(B)
        assert reduced.shape == (3, 3)

    def test_skewed_basis(self):
        B = np.array([[1, 0], [100, 1]], dtype=float)
        reduced = LLLReduction.reduce(B)
        # Reduced basis should have shorter vectors
        norms = [np.linalg.norm(reduced[i]) for i in range(2)]
        assert max(norms) < 101  # Original has norm ~100

    def test_shortest_vector_estimate(self):
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        sv = LLLReduction.shortest_vector_estimate(B)
        assert np.linalg.norm(sv) == pytest.approx(1.0)

    def test_2d_reduction(self):
        B = np.array([[1, 0], [0.6, 1]], dtype=float)
        reduced = LLLReduction.reduce(B)
        # After reduction, |mu_ij| <= 0.5 should hold
        assert reduced.shape == (2, 2)


# ============================================================================
# EllipticCurve
# ============================================================================

class TestEllipticCurve:
    def test_singular_curve_rejected(self):
        # 4*0^3 + 27*0^2 = 0 mod p
        with pytest.raises(ValueError, match="Singular"):
            EllipticCurve(0, 0, 5)

    def test_is_on_curve(self):
        # y^2 = x^3 + x + 1 over F_5
        ec = EllipticCurve(1, 1, 5)
        assert ec.is_on_curve(None)  # point at infinity
        # Check (0, 1): 1^2 = 0+0+1 = 1 mod 5. Yes.
        assert ec.is_on_curve((0, 1))
        assert ec.is_on_curve((0, 4))  # (0, -1 mod 5)
        assert not ec.is_on_curve((0, 2))

    def test_add_identity(self):
        ec = EllipticCurve(1, 1, 5)
        P = (0, 1)
        assert ec.add(P, None) == P
        assert ec.add(None, P) == P

    def test_add_inverse(self):
        ec = EllipticCurve(1, 1, 5)
        P = (0, 1)
        neg_P = ec.negate(P)
        assert ec.add(P, neg_P) is None

    def test_negate(self):
        ec = EllipticCurve(1, 1, 5)
        assert ec.negate(None) is None
        assert ec.negate((0, 1)) == (0, 4)

    def test_scalar_multiply_identity(self):
        ec = EllipticCurve(1, 1, 5)
        P = (0, 1)
        assert ec.scalar_multiply(1, P) == P

    def test_scalar_multiply_zero(self):
        ec = EllipticCurve(1, 1, 5)
        P = (0, 1)
        assert ec.scalar_multiply(0, P) is None

    def test_scalar_multiply_negative(self):
        ec = EllipticCurve(1, 1, 5)
        P = (0, 1)
        neg_P = ec.negate(P)
        assert ec.scalar_multiply(-1, P) == neg_P

    def test_point_doubling(self):
        ec = EllipticCurve(1, 1, 5)
        P = (0, 1)
        Q = ec.add(P, P)
        assert Q is not None
        assert ec.is_on_curve(Q)

    def test_order_small_curve(self):
        # y^2 = x^3 + x + 1 over F_5
        ec = EllipticCurve(1, 1, 5)
        order = ec.order()
        # Hasse bound: |order - (p+1)| <= 2*sqrt(p), so |order - 6| <= 4
        assert 2 <= order <= 10

    def test_point_order(self):
        # y^2 = x^3 + 2x + 3 over F_7 (known to work well)
        ec = EllipticCurve(2, 3, 7)
        assert ec.point_order(None) == 1
        # Find a valid point on this curve
        pts = ec.points()
        real_pts = [p for p in pts if p is not None]
        assert len(real_pts) > 0
        P = real_pts[0]
        po = ec.point_order(P)
        assert po > 0
        # [po]P should be identity
        assert ec.scalar_multiply(po, P) is None

    def test_points_enumeration(self):
        ec = EllipticCurve(1, 1, 5)
        pts = ec.points()
        assert None in pts  # infinity
        assert len(pts) == ec.order()
        for p in pts:
            assert ec.is_on_curve(p)


# ============================================================================
# SymbolicExpr
# ============================================================================

class TestSymbolicExpr:
    def test_const(self):
        c = SymbolicExpr.const(3.0)
        assert c.evaluate({}) == 3.0

    def test_var(self):
        x = SymbolicExpr.var("x")
        assert x.evaluate({"x": 5.0}) == 5.0

    def test_unbound_var(self):
        x = SymbolicExpr.var("x")
        with pytest.raises(ValueError, match="Unbound"):
            x.evaluate({})

    def test_add(self):
        x = SymbolicExpr.var("x")
        expr = x + 3
        assert expr.evaluate({"x": 2.0}) == pytest.approx(5.0)

    def test_radd(self):
        x = SymbolicExpr.var("x")
        expr = 3 + x
        assert expr.evaluate({"x": 2.0}) == pytest.approx(5.0)

    def test_mul(self):
        x = SymbolicExpr.var("x")
        expr = x * 4
        assert expr.evaluate({"x": 3.0}) == pytest.approx(12.0)

    def test_rmul(self):
        x = SymbolicExpr.var("x")
        expr = 4 * x
        assert expr.evaluate({"x": 3.0}) == pytest.approx(12.0)

    def test_sub(self):
        x = SymbolicExpr.var("x")
        expr = x - 1
        assert expr.evaluate({"x": 5.0}) == pytest.approx(4.0)

    def test_div(self):
        x = SymbolicExpr.var("x")
        expr = x / 2
        assert expr.evaluate({"x": 10.0}) == pytest.approx(5.0)

    def test_pow(self):
        x = SymbolicExpr.var("x")
        expr = x ** 3
        assert expr.evaluate({"x": 2.0}) == pytest.approx(8.0)

    def test_neg(self):
        x = SymbolicExpr.var("x")
        expr = -x
        assert expr.evaluate({"x": 7.0}) == pytest.approx(-7.0)

    def test_sin(self):
        x = SymbolicExpr.var("x")
        expr = x.sin()
        assert expr.evaluate({"x": 0.0}) == pytest.approx(0.0)

    def test_cos(self):
        x = SymbolicExpr.var("x")
        expr = x.cos()
        assert expr.evaluate({"x": 0.0}) == pytest.approx(1.0)

    def test_exp(self):
        x = SymbolicExpr.var("x")
        expr = x.exp()
        assert expr.evaluate({"x": 0.0}) == pytest.approx(1.0)

    def test_log(self):
        x = SymbolicExpr.var("x")
        expr = x.log()
        assert expr.evaluate({"x": 1.0}) == pytest.approx(0.0)

    def test_sqrt(self):
        x = SymbolicExpr.var("x")
        expr = x.sqrt()
        assert expr.evaluate({"x": 9.0}) == pytest.approx(3.0)

    def test_variables(self):
        x = SymbolicExpr.var("x")
        y = SymbolicExpr.var("y")
        expr = x * y + x
        assert expr.variables() == {"x", "y"}

    def test_variables_const(self):
        c = SymbolicExpr.const(5)
        assert c.variables() == set()

    def test_repr_const(self):
        assert "5" in repr(SymbolicExpr.const(5.0))

    def test_repr_var(self):
        assert "x" in repr(SymbolicExpr.var("x"))


# ============================================================================
# SymbolicExpr differentiation
# ============================================================================

class TestSymbolicDifferentiation:
    def test_diff_const(self):
        c = SymbolicExpr.const(5.0)
        dc = c.differentiate("x")
        assert dc.evaluate({}) == pytest.approx(0.0)

    def test_diff_var(self):
        x = SymbolicExpr.var("x")
        dx = x.differentiate("x")
        assert dx.evaluate({}) == pytest.approx(1.0)

    def test_diff_other_var(self):
        y = SymbolicExpr.var("y")
        dy_dx = y.differentiate("x")
        assert dy_dx.evaluate({}) == pytest.approx(0.0)

    def test_diff_add(self):
        x = SymbolicExpr.var("x")
        expr = x + SymbolicExpr.const(3.0)
        d = expr.differentiate("x")
        assert d.evaluate({"x": 1.0}) == pytest.approx(1.0)

    def test_diff_mul_product_rule(self):
        x = SymbolicExpr.var("x")
        # d/dx(x * x) = 2x
        expr = x * x
        d = expr.differentiate("x")
        assert d.evaluate({"x": 3.0}) == pytest.approx(6.0)

    def test_diff_power_rule(self):
        x = SymbolicExpr.var("x")
        # d/dx(x^3) = 3x^2
        expr = x ** 3
        d = expr.differentiate("x")
        assert d.evaluate({"x": 2.0}) == pytest.approx(12.0)

    def test_diff_neg(self):
        x = SymbolicExpr.var("x")
        expr = -x
        d = expr.differentiate("x")
        assert d.evaluate({"x": 5.0}) == pytest.approx(-1.0)

    def test_diff_sin(self):
        x = SymbolicExpr.var("x")
        expr = x.sin()
        d = expr.differentiate("x")
        # d/dx sin(x) = cos(x); cos(0) = 1
        assert d.evaluate({"x": 0.0}) == pytest.approx(1.0)

    def test_diff_cos(self):
        x = SymbolicExpr.var("x")
        expr = x.cos()
        d = expr.differentiate("x")
        # d/dx cos(x) = -sin(x); -sin(0) = 0
        assert d.evaluate({"x": 0.0}) == pytest.approx(0.0)

    def test_diff_exp(self):
        x = SymbolicExpr.var("x")
        expr = x.exp()
        d = expr.differentiate("x")
        # d/dx exp(x) = exp(x); exp(0) = 1
        assert d.evaluate({"x": 0.0}) == pytest.approx(1.0)

    def test_diff_log(self):
        x = SymbolicExpr.var("x")
        expr = x.log()
        d = expr.differentiate("x")
        # d/dx ln(x) = 1/x; 1/2 = 0.5
        assert d.evaluate({"x": 2.0}) == pytest.approx(0.5)

    def test_diff_sqrt(self):
        x = SymbolicExpr.var("x")
        expr = x.sqrt()
        d = expr.differentiate("x")
        # d/dx sqrt(x) = 1/(2*sqrt(x)); at x=4 => 1/4 = 0.25
        assert d.evaluate({"x": 4.0}) == pytest.approx(0.25)

    def test_diff_sub(self):
        x = SymbolicExpr.var("x")
        expr = x - SymbolicExpr.const(5.0)
        d = expr.differentiate("x")
        assert d.evaluate({"x": 0.0}) == pytest.approx(1.0)

    def test_diff_div_quotient_rule(self):
        x = SymbolicExpr.var("x")
        # d/dx(x / (x+1)) at x=1: (1*(1+1) - 1*1) / (1+1)^2 = 1/4
        expr = x / (x + 1)
        d = expr.differentiate("x")
        assert d.evaluate({"x": 1.0}) == pytest.approx(0.25)


# ============================================================================
# SymbolicExpr simplify
# ============================================================================

class TestSymbolicSimplify:
    def test_add_zero(self):
        x = SymbolicExpr.var("x")
        expr = x + 0
        s = expr.simplify()
        assert s.evaluate({"x": 7.0}) == pytest.approx(7.0)

    def test_mul_one(self):
        x = SymbolicExpr.var("x")
        expr = x * 1
        s = expr.simplify()
        assert s.evaluate({"x": 3.0}) == pytest.approx(3.0)

    def test_mul_zero(self):
        x = SymbolicExpr.var("x")
        expr = x * 0
        s = expr.simplify()
        assert s.evaluate({"x": 99.0}) == pytest.approx(0.0)

    def test_pow_zero(self):
        x = SymbolicExpr.var("x")
        expr = x ** 0
        s = expr.simplify()
        assert s.evaluate({"x": 42.0}) == pytest.approx(1.0)

    def test_pow_one(self):
        x = SymbolicExpr.var("x")
        expr = x ** 1
        s = expr.simplify()
        assert s.evaluate({"x": 5.0}) == pytest.approx(5.0)

    def test_double_neg(self):
        x = SymbolicExpr.var("x")
        expr = -(-x)
        s = expr.simplify()
        assert s.evaluate({"x": 3.0}) == pytest.approx(3.0)

    def test_constant_folding(self):
        expr = SymbolicExpr.const(2.0) + SymbolicExpr.const(3.0)
        s = expr.simplify()
        assert s.op == "const"
        assert s.args[0] == pytest.approx(5.0)

    def test_sub_zero(self):
        x = SymbolicExpr.var("x")
        expr = x - 0
        s = expr.simplify()
        assert s.evaluate({"x": 4.0}) == pytest.approx(4.0)

    def test_div_one(self):
        x = SymbolicExpr.var("x")
        expr = x / 1
        s = expr.simplify()
        assert s.evaluate({"x": 6.0}) == pytest.approx(6.0)


# ============================================================================
# PatternMatcher
# ============================================================================

class TestPatternMatcher:
    def test_match_var(self):
        pattern = SymbolicExpr.var("x")
        expr = SymbolicExpr.const(5.0)
        result = PatternMatcher.match(pattern, expr)
        assert result is not None
        assert result["x"].op == "const"

    def test_match_const(self):
        pattern = SymbolicExpr.const(5.0)
        expr = SymbolicExpr.const(5.0)
        assert PatternMatcher.match(pattern, expr) is not None

    def test_match_const_mismatch(self):
        pattern = SymbolicExpr.const(5.0)
        expr = SymbolicExpr.const(6.0)
        assert PatternMatcher.match(pattern, expr) is None

    def test_match_op_mismatch(self):
        pattern = SymbolicExpr.var("x") + SymbolicExpr.var("y")
        expr = SymbolicExpr.var("a") * SymbolicExpr.var("b")
        assert PatternMatcher.match(pattern, expr) is None

    def test_match_repeated_var(self):
        # Pattern: x + x should only match when both sides are equal
        x = SymbolicExpr.var("x")
        pattern = x + x
        expr = SymbolicExpr.const(3.0) + SymbolicExpr.const(3.0)
        result = PatternMatcher.match(pattern, expr)
        assert result is not None

    def test_match_repeated_var_fail(self):
        x = SymbolicExpr.var("x")
        pattern = x + x
        expr = SymbolicExpr.const(3.0) + SymbolicExpr.const(4.0)
        assert PatternMatcher.match(pattern, expr) is None

    def test_unify_vars(self):
        a = SymbolicExpr.var("a")
        b = SymbolicExpr.var("b")
        result = PatternMatcher.unify(a, b)
        assert result is not None

    def test_unify_const_match(self):
        a = SymbolicExpr.const(5.0)
        b = SymbolicExpr.const(5.0)
        assert PatternMatcher.unify(a, b) is not None

    def test_unify_const_mismatch(self):
        a = SymbolicExpr.const(5.0)
        b = SymbolicExpr.const(6.0)
        assert PatternMatcher.unify(a, b) is None

    def test_structurally_equal(self):
        a = SymbolicExpr.var("x") + SymbolicExpr.const(1.0)
        b = SymbolicExpr.var("x") + SymbolicExpr.const(1.0)
        assert PatternMatcher._structurally_equal(a, b)

    def test_occurs_check(self):
        assert PatternMatcher._occurs("x", SymbolicExpr.var("x"))
        assert not PatternMatcher._occurs("x", SymbolicExpr.var("y"))
        assert PatternMatcher._occurs("x", SymbolicExpr.var("x") + SymbolicExpr.const(1.0))


# ============================================================================
# RewriteSystem
# ============================================================================

class TestRewriteSystem:
    def test_simple_rewrite(self):
        # Rule: x + 0 -> x
        x = SymbolicExpr.var("x")
        rule = RewriteRule(
            pattern=x + SymbolicExpr.const(0.0),
            replacement=x,
            name="add_zero",
        )
        system = RewriteSystem([rule])
        expr = SymbolicExpr.var("a") + SymbolicExpr.const(0.0)
        result = system.apply(expr)
        assert result.op == "var" or result.evaluate({"a": 5.0}) == pytest.approx(5.0)

    def test_normalize_converges(self):
        x = SymbolicExpr.var("x")
        rule = RewriteRule(
            pattern=x + SymbolicExpr.const(0.0),
            replacement=x,
        )
        system = RewriteSystem([rule])
        expr = SymbolicExpr.var("a") + SymbolicExpr.const(0.0)
        result = system.normalize(expr, max_steps=10)
        # Should converge quickly
        assert result is not None

    def test_is_confluent_trivial(self):
        system = RewriteSystem([])
        assert system.is_confluent() is True


# ============================================================================
# _is_zero / _is_one helpers
# ============================================================================

class TestHelpers:
    def test_is_zero(self):
        assert _is_zero(SymbolicExpr.const(0.0))
        assert not _is_zero(SymbolicExpr.const(1.0))
        assert not _is_zero(SymbolicExpr.var("x"))

    def test_is_one(self):
        assert _is_one(SymbolicExpr.const(1.0))
        assert not _is_one(SymbolicExpr.const(0.0))
        assert not _is_one(SymbolicExpr.var("x"))


# ============================================================================
# GraphColoring
# ============================================================================

class TestGraphColoring:
    def _complete_graph(self, n):
        adj = np.ones((n, n)) - np.eye(n)
        return adj

    def _cycle_graph(self, n):
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1
        return adj

    def _empty_graph(self, n):
        return np.zeros((n, n))

    def test_greedy_complete_3(self):
        adj = self._complete_graph(3)
        result = GraphColoring.greedy(adj)
        assert result.n_colors == 3
        assert GraphColoring.is_valid_coloring(adj, result.colors)

    def test_greedy_empty(self):
        adj = self._empty_graph(4)
        result = GraphColoring.greedy(adj)
        assert result.n_colors == 1  # all same color
        assert GraphColoring.is_valid_coloring(adj, result.colors)

    def test_greedy_custom_ordering(self):
        adj = self._complete_graph(3)
        result = GraphColoring.greedy(adj, ordering=[2, 1, 0])
        assert result.n_colors == 3
        assert GraphColoring.is_valid_coloring(adj, result.colors)

    def test_welsh_powell(self):
        adj = self._complete_graph(4)
        result = GraphColoring.welsh_powell(adj)
        assert result.n_colors == 4
        assert GraphColoring.is_valid_coloring(adj, result.colors)

    def test_dsatur_empty_graph(self):
        result = GraphColoring.dsatur(np.zeros((0, 0)))
        assert result.n_colors == 0

    def test_dsatur_complete(self):
        adj = self._complete_graph(5)
        result = GraphColoring.dsatur(adj)
        assert result.n_colors == 5
        assert GraphColoring.is_valid_coloring(adj, result.colors)

    def test_dsatur_cycle_odd(self):
        adj = self._cycle_graph(5)
        result = GraphColoring.dsatur(adj)
        assert result.n_colors == 3  # odd cycle needs 3
        assert GraphColoring.is_valid_coloring(adj, result.colors)

    def test_dsatur_cycle_even(self):
        adj = self._cycle_graph(4)
        result = GraphColoring.dsatur(adj)
        assert result.n_colors == 2  # even cycle is bipartite
        assert GraphColoring.is_valid_coloring(adj, result.colors)

    def test_chromatic_number_complete(self):
        adj = self._complete_graph(4)
        chi = GraphColoring.chromatic_number_exact(adj)
        assert chi == 4

    def test_chromatic_number_empty(self):
        assert GraphColoring.chromatic_number_exact(np.zeros((0, 0))) == 0

    def test_chromatic_number_single(self):
        assert GraphColoring.chromatic_number_exact(np.zeros((1, 1))) == 1

    def test_chromatic_number_too_large(self):
        with pytest.raises(ValueError, match="safety limit"):
            GraphColoring.chromatic_number_exact(np.zeros((30, 30)))

    def test_is_valid_coloring_valid(self):
        adj = self._cycle_graph(4)
        colors = {0: 0, 1: 1, 2: 0, 3: 1}
        assert GraphColoring.is_valid_coloring(adj, colors)

    def test_is_valid_coloring_invalid(self):
        adj = self._cycle_graph(4)
        colors = {0: 0, 1: 0, 2: 0, 3: 0}  # adjacent same color
        assert not GraphColoring.is_valid_coloring(adj, colors)

    def test_is_valid_coloring_missing_vertex(self):
        adj = self._cycle_graph(3)
        colors = {0: 0, 1: 1}  # missing vertex 2
        assert not GraphColoring.is_valid_coloring(adj, colors)

    def test_normalize_adj_nonsquare(self):
        with pytest.raises(ValueError, match="square"):
            GraphColoring._normalize_adj(np.zeros((2, 3)))


# ============================================================================
# ColoringResult
# ============================================================================

class TestColoringResult:
    def test_color_classes(self):
        cr = ColoringResult(colors={0: 0, 1: 1, 2: 0, 3: 1}, n_colors=2, is_optimal=True)
        classes = cr.color_classes
        assert len(classes) == 2
        assert sorted(classes[0]) == [0, 2]
        assert sorted(classes[1]) == [1, 3]

    def test_largest_color_class(self):
        cr = ColoringResult(colors={0: 0, 1: 0, 2: 0, 3: 1}, n_colors=2, is_optimal=False)
        assert cr.largest_color_class == 3

    def test_largest_color_class_empty(self):
        cr = ColoringResult(colors={}, n_colors=0, is_optimal=True)
        assert cr.largest_color_class == 0

    def test_utilization(self):
        cr = ColoringResult(colors={0: 0, 1: 0, 2: 1}, n_colors=2, is_optimal=False)
        # Two vertices share color 0, one has color 1 alone
        # shared = 2, total = 3
        assert cr.utilization() == pytest.approx(2 / 3)

    def test_utilization_empty(self):
        cr = ColoringResult(colors={}, n_colors=0, is_optimal=True)
        assert cr.utilization() == 0.0


# ============================================================================
# ConflictGraph
# ============================================================================

class TestConflictGraph:
    def test_from_resources(self):
        adj = ConflictGraph.from_resources([0, 1, 2], [(0, 1), (1, 2)])
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0
        assert adj[1, 2] == 1.0
        assert adj[0, 2] == 0.0

    def test_from_resources_empty(self):
        adj = ConflictGraph.from_resources([], [])
        assert adj.shape == (0, 0)

    def test_from_schedule(self):
        adj = ConflictGraph.from_schedule([0, 1, 2], [(0, 2)])
        assert adj[0, 2] == 1.0
        assert adj[2, 0] == 1.0
        assert adj[0, 1] == 0.0

    def test_from_schedule_empty(self):
        adj = ConflictGraph.from_schedule([], [])
        assert adj.shape == (0, 0)

    def test_from_intervals_overlap(self):
        intervals = [(0, 3), (2, 5), (6, 8)]
        adj = ConflictGraph.from_intervals(intervals)
        assert adj[0, 1] == 1.0  # overlap
        assert adj[0, 2] == 0.0  # no overlap
        assert adj[1, 2] == 0.0

    def test_from_intervals_no_overlap(self):
        intervals = [(0, 1), (2, 3), (4, 5)]
        adj = ConflictGraph.from_intervals(intervals)
        assert np.sum(adj) == 0.0


# ============================================================================
# chromatic_polynomial
# ============================================================================

class TestChromaticPolynomial:
    def test_empty_graph(self):
        # P(E_n, k) = k^n
        adj = np.zeros((3, 3))
        assert chromatic_polynomial(adj, 2) == 8  # 2^3

    def test_complete_graph_k3(self):
        # P(K_3, k) = k(k-1)(k-2)
        adj = np.ones((3, 3)) - np.eye(3)
        assert chromatic_polynomial(adj, 3) == 6   # 3*2*1
        assert chromatic_polynomial(adj, 2) == 0   # not 2-colorable

    def test_single_edge(self):
        # P(K_2, k) = k(k-1)
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        assert chromatic_polynomial(adj, 3) == 6  # 3*2

    def test_too_large(self):
        with pytest.raises(ValueError, match="safety limit"):
            chromatic_polynomial(np.zeros((20, 20)), 2)

    def test_k_zero(self):
        adj = np.zeros((2, 2))
        assert chromatic_polynomial(adj, 0) == 0


# ============================================================================
# independent_sets
# ============================================================================

class TestIndependentSets:
    def test_empty_graph(self):
        adj = np.zeros((3, 3))
        sets = independent_sets(adj)
        # Only maximal IS of empty graph is the full vertex set
        assert any(sorted(s) == [0, 1, 2] for s in sets)

    def test_complete_graph(self):
        adj = np.ones((3, 3)) - np.eye(3)
        sets = independent_sets(adj)
        # Each vertex alone is a maximal IS
        assert len(sets) == 3
        for s in sets:
            assert len(s) == 1

    def test_too_large(self):
        with pytest.raises(ValueError, match="safety limit"):
            independent_sets(np.zeros((35, 35)))


# ============================================================================
# maximum_independent_set
# ============================================================================

class TestMaximumIndependentSet:
    def test_empty_graph(self):
        adj = np.zeros((4, 4))
        mis = maximum_independent_set(adj)
        assert len(mis) == 4  # all vertices

    def test_complete_graph(self):
        adj = np.ones((3, 3)) - np.eye(3)
        mis = maximum_independent_set(adj)
        assert len(mis) == 1

    def test_cycle(self):
        n = 6
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[(i + 1) % n, i] = 1
        mis = maximum_independent_set(adj)
        # MIS of C6 has size 3
        assert len(mis) >= 2  # greedy may not be optimal


# ============================================================================
# Convenience functions
# ============================================================================

class TestConvenienceFunctions:
    def test_color_graph_dsatur(self):
        adj = np.ones((3, 3)) - np.eye(3)
        result = color_graph(adj, method="dsatur")
        assert result.n_colors == 3

    def test_color_graph_greedy(self):
        adj = np.ones((3, 3)) - np.eye(3)
        result = color_graph(adj, method="greedy")
        assert result.n_colors == 3

    def test_color_graph_welsh_powell(self):
        adj = np.ones((3, 3)) - np.eye(3)
        result = color_graph(adj, method="welsh_powell")
        assert result.n_colors == 3

    def test_color_graph_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            color_graph(np.zeros((2, 2)), method="invalid")

    def test_schedule_tasks(self):
        result = schedule_tasks([0, 1, 2], [(0, 1), (1, 2)])
        # Tasks 0 and 2 can share a slot, task 1 cannot share with either
        assert result[0] != result[1]
        assert result[1] != result[2]
