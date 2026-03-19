"""
Classical Analysis Theorems
============================

Scope slots: AN1.5 (Stone-Weierstrass), AN1.6 (Arzela-Ascoli),
AN3.7 (weak topology / Hahn-Banach), AN3.8 (Hahn-Banach extension),
AN4.6 (Haar measure).

Pure numpy implementations of fundamental analysis theorems:
- Stone-Weierstrass: uniform approximation by polynomial/basis functions
- Arzela-Ascoli: compactness criterion for function families
- Weak topology: weak convergence of sequences in dual spaces
- Hahn-Banach: extension of bounded linear functionals
- Haar measure: left-invariant measure on finite groups

Author: Claude Code
Date: March 2026
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# STONE-WEIERSTRASS APPROXIMATION (AN1.5)
# ============================================================================

class StoneWeierstrass:
    """Uniform approximation of continuous functions.

    The Stone-Weierstrass theorem: if A is a subalgebra of C(X, R) that
    separates points and vanishes nowhere, then A is dense in C(X, R).

    Constructively: any continuous function on [a,b] can be uniformly
    approximated by polynomials (Weierstrass) or by any separating
    subalgebra (Stone).

    We implement:
    1. Bernstein polynomial approximation (constructive Weierstrass)
    2. General basis function approximation via least-squares

    Example:
        >>> sw = StoneWeierstrass()
        >>> f = lambda x: np.abs(x - 0.5)  # Not smooth, but continuous
        >>> coeffs = sw.approximate(f, degree=20, domain=(0, 1))
    """

    def approximate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        basis_fns: list[Callable] | None = None,
        domain: tuple[float, float] = (0.0, 1.0),
        degree: int = 10,
        n_samples: int = 200,
    ) -> np.ndarray:
        """Approximate f uniformly using basis functions.

        If basis_fns is None, uses monomials {1, x, x², ..., x^degree}
        (polynomial approximation).

        Args:
            f: Target function.
            basis_fns: List of basis functions. If None, uses polynomials.
            domain: Approximation domain [a, b].
            degree: Polynomial degree (used if basis_fns is None).
            n_samples: Number of sample points for least-squares fit.

        Returns:
            Coefficient vector.
        """
        a, b = domain
        x = np.linspace(a, b, n_samples)
        y = f(x)

        if basis_fns is None:
            # Monomial basis
            basis_fns = [self._monomial(k) for k in range(degree + 1)]

        # Build Vandermonde-like matrix
        n_basis = len(basis_fns)
        V = np.zeros((n_samples, n_basis))
        for j, phi in enumerate(basis_fns):
            V[:, j] = phi(x)

        # Least-squares fit
        coeffs, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        return coeffs

    def bernstein_approximate(
        self,
        f: Callable[[float], float],
        degree: int,
        domain: tuple[float, float] = (0.0, 1.0),
    ) -> Callable[[float], float]:
        """Bernstein polynomial approximation (constructive Weierstrass proof).

        B_n(f)(x) = Σ_{k=0}^{n} f(k/n) C(n,k) x^k (1-x)^{n-k}

        Converges uniformly to f on [0,1] as n → ∞.

        Args:
            f: Target function.
            degree: Polynomial degree n.
            domain: Domain [a, b] (mapped to [0, 1] internally).

        Returns:
            Bernstein polynomial approximation function.
        """
        a, b = domain
        n = degree

        # Precompute f values at Bernstein nodes
        f_values = np.array([f(a + k / n * (b - a)) for k in range(n + 1)])

        # Binomial coefficients
        from math import comb
        binom = np.array([comb(n, k) for k in range(n + 1)])

        def bernstein(x_raw):
            # Map to [0, 1]
            x = (np.asarray(x_raw, dtype=float) - a) / (b - a)
            x = np.clip(x, 0, 1)
            result = np.zeros_like(x, dtype=float)
            for k in range(n + 1):
                result += f_values[k] * binom[k] * x**k * (1 - x)**(n - k)
            return result

        return bernstein

    def approximation_error(
        self,
        f: Callable,
        coeffs: np.ndarray,
        basis_fns: list[Callable] | None = None,
        domain: tuple[float, float] = (0.0, 1.0),
        degree: int | None = None,
        n_test: int = 500,
    ) -> float:
        """Compute max approximation error (sup norm).

        Args:
            f: Original function.
            coeffs: Approximation coefficients.
            basis_fns: Basis functions (monomials if None).
            domain: Domain.
            degree: Polynomial degree (if basis_fns is None).
            n_test: Number of test points.

        Returns:
            Maximum absolute error.
        """
        a, b = domain
        x = np.linspace(a, b, n_test)
        y_true = f(x)

        if basis_fns is None:
            if degree is None:
                degree = len(coeffs) - 1
            basis_fns = [self._monomial(k) for k in range(degree + 1)]

        y_approx = sum(c * phi(x) for c, phi in zip(coeffs, basis_fns))
        return float(np.max(np.abs(y_true - y_approx)))

    @staticmethod
    def _monomial(k: int) -> Callable:
        return lambda x, k=k: np.asarray(x, dtype=float) ** k


# ============================================================================
# ARZELA-ASCOLI COMPACTNESS (AN1.6)
# ============================================================================

class ArzelaAscoli:
    """Arzela-Ascoli compactness criterion for function families.

    A family F ⊂ C(X) is relatively compact iff:
    1. F is pointwise bounded: sup_{f ∈ F} |f(x)| < ∞ for each x
    2. F is equicontinuous: ∀ε > 0, ∃δ > 0 such that
       |x - y| < δ ⟹ |f(x) - f(y)| < ε for ALL f ∈ F

    We verify these conditions numerically on discretized functions.

    Example:
        >>> aa = ArzelaAscoli()
        >>> fns = [lambda x, n=n: np.sin(n*x)/n for n in range(1, 20)]
        >>> aa.is_compact(fns, domain=(0, np.pi), epsilon=0.1)
        True
    """

    def is_compact(
        self,
        function_family: list[Callable],
        domain: tuple[float, float],
        epsilon: float = 0.1,
        n_points: int = 100,
    ) -> bool:
        """Check Arzela-Ascoli conditions numerically.

        Args:
            function_family: List of functions (callables).
            domain: Domain [a, b].
            epsilon: Equicontinuity tolerance.
            n_points: Number of test points.

        Returns:
            True if both conditions appear satisfied.
        """
        a, b = domain
        x = np.linspace(a, b, n_points)

        # Condition 1: Pointwise boundedness
        if not self._is_pointwise_bounded(function_family, x):
            return False

        # Condition 2: Equicontinuity
        if not self._is_equicontinuous(function_family, x, epsilon):
            return False

        return True

    def _is_pointwise_bounded(
        self,
        family: list[Callable],
        x: np.ndarray,
        bound: float = 1e6,
    ) -> bool:
        """Check pointwise boundedness."""
        for f in family:
            vals = f(x)
            if np.any(np.abs(vals) > bound):
                return False
        return True

    def _is_equicontinuous(
        self,
        family: list[Callable],
        x: np.ndarray,
        epsilon: float,
    ) -> bool:
        """Check equicontinuity via finite differences.

        For each pair of adjacent grid points, check that ALL functions
        have |f(x_i) - f(x_{i+1})| < epsilon when dx is small enough.
        """
        n = len(x)
        dx = x[1] - x[0]

        max_variation = 0.0
        for f in family:
            vals = f(x)
            diffs = np.abs(np.diff(vals))
            max_variation = max(max_variation, np.max(diffs))

        # Equicontinuous if the modulus of continuity is uniformly bounded
        return max_variation < epsilon

    @staticmethod
    def modulus_of_continuity(
        f: Callable,
        domain: tuple[float, float],
        delta: float,
        n_points: int = 200,
    ) -> float:
        """Compute modulus of continuity ω(f, δ) = sup{|f(x)-f(y)| : |x-y| ≤ δ}.

        Args:
            f: Function.
            domain: Domain [a, b].
            delta: Distance threshold.
            n_points: Grid resolution.

        Returns:
            Approximate modulus of continuity.
        """
        a, b = domain
        x = np.linspace(a, b, n_points)
        vals = f(x)

        max_diff = 0.0
        k = max(1, int(delta / ((b - a) / n_points)))
        for offset in range(1, k + 1):
            diffs = np.abs(vals[offset:] - vals[:-offset])
            if len(diffs) > 0:
                max_diff = max(max_diff, np.max(diffs))

        return max_diff


# ============================================================================
# WEAK TOPOLOGY & CONVERGENCE (AN3.7)
# ============================================================================

class WeakTopology:
    """Weak convergence in normed spaces.

    A sequence (x_n) in a normed space X converges weakly to x iff
    φ(x_n) → φ(x) for every φ ∈ X* (continuous linear functional).

    Weak convergence is strictly weaker than norm convergence:
    e_n → 0 weakly in l² but ||e_n|| = 1 (no norm convergence).

    Example:
        >>> wt = WeakTopology()
        >>> # Standard basis in R^10, check weak convergence to 0
        >>> seq = [np.eye(10)[i] for i in range(10)]
        >>> functionals = [lambda x, j=j: x[j] for j in range(10)]
        >>> # Each functional f_j(e_n) = δ_{jn}, doesn't converge
    """

    def weak_convergence(
        self,
        sequence: list[np.ndarray],
        dual_elements: list[Callable],
        target: np.ndarray | None = None,
        tol: float = 1e-6,
    ) -> bool:
        """Test weak convergence of a sequence.

        Args:
            sequence: List of vectors x_1, x_2, ...
            dual_elements: List of linear functionals φ_1, φ_2, ...
            target: Candidate weak limit (auto-detected if None).
            tol: Convergence tolerance.

        Returns:
            True if φ(x_n) → φ(target) for all given φ.
        """
        if not sequence:
            return True

        if target is None:
            # Estimate weak limit as Cesaro mean
            target = np.mean(sequence[-max(1, len(sequence)//4):], axis=0)

        for phi in dual_elements:
            target_val = phi(target)
            vals = [phi(x) for x in sequence]

            # Check if vals converges to target_val
            n = len(vals)
            tail_start = max(1, n // 2)
            tail = vals[tail_start:]
            if not all(abs(v - target_val) < tol for v in tail):
                return False

        return True

    @staticmethod
    def weak_star_convergence(
        functionals: list[Callable],
        target_functional: Callable,
        test_vectors: list[np.ndarray],
        tol: float = 1e-6,
    ) -> bool:
        """Test weak-* convergence of a sequence of functionals.

        (φ_n) → φ weak-* iff φ_n(x) → φ(x) for all x ∈ X.

        Args:
            functionals: Sequence of linear functionals.
            target_functional: Candidate weak-* limit.
            test_vectors: Test vectors from X.
            tol: Tolerance.

        Returns:
            True if converges weak-* on all test vectors.
        """
        for x in test_vectors:
            target_val = target_functional(x)
            vals = [phi(x) for phi in functionals]
            n = len(vals)
            tail = vals[max(1, n // 2):]
            if not all(abs(v - target_val) < tol for v in tail):
                return False
        return True


# ============================================================================
# HAHN-BANACH EXTENSION (AN3.8)
# ============================================================================

class HahnBanach:
    """Hahn-Banach theorem: extension of bounded linear functionals.

    If M is a subspace of X and f: M → R is bounded linear with ||f|| ≤ C,
    then there exists F: X → R extending f with ||F|| = ||f||.

    We implement the constructive version for finite-dimensional spaces:
    extend by orthogonal projection + scaling.

    Example:
        >>> hb = HahnBanach()
        >>> # Functional on span{e_1} in R³
        >>> subspace_basis = np.array([[1, 0, 0]])
        >>> f = lambda x: 2 * x[0]  # Defined on subspace
        >>> F = hb.extend(f, subspace_basis, n=3)
        >>> F(np.array([1, 2, 3]))  # Should be 2 (extends to full space)
    """

    def extend(
        self,
        functional: Callable[[np.ndarray], float],
        subspace_basis: np.ndarray,
        n: int,
    ) -> Callable[[np.ndarray], float]:
        """Extend a bounded linear functional from a subspace to the full space.

        Uses the Riesz representation: on a Hilbert space, every bounded
        linear functional is an inner product with some fixed vector.
        Find that vector on the subspace, then use it on the full space.

        Args:
            functional: Linear functional on the subspace.
            subspace_basis: Basis vectors of the subspace, shape (m, n).
            n: Dimension of the full space.

        Returns:
            Extended functional on R^n.
        """
        subspace_basis = np.asarray(subspace_basis, dtype=float)
        m = subspace_basis.shape[0]

        # Orthogonalize the basis
        Q, _ = np.linalg.qr(subspace_basis.T)
        Q = Q[:, :m]  # Orthonormal basis, shape (n, m)

        # Find Riesz representative: the vector v in span such that
        # f(x) = <v, x> for x in subspace
        # v = Σ f(q_i) q_i
        v = np.zeros(n)
        for i in range(m):
            v += functional(Q[:, i]) * Q[:, i]

        def extended(x: np.ndarray) -> float:
            return float(np.dot(v, x))

        return extended

    @staticmethod
    def norm_preserving_check(
        original: Callable,
        extended: Callable,
        subspace_basis: np.ndarray,
        n_test: int = 100,
    ) -> float:
        """Verify that ||F|| = ||f|| (norm preservation).

        Returns the ratio ||F||/||f|| which should be ≈ 1.

        Args:
            original: Original functional on subspace.
            extended: Extended functional on full space.
            subspace_basis: Subspace basis.
            n_test: Number of random test points.

        Returns:
            Ratio of operator norms.
        """
        n = subspace_basis.shape[1]
        m = subspace_basis.shape[0]

        # Estimate ||f|| on subspace
        max_f = 0.0
        for _ in range(n_test):
            coeffs = np.random.randn(m)
            x = subspace_basis.T @ coeffs
            norm_x = np.linalg.norm(x)
            if norm_x > 1e-10:
                max_f = max(max_f, abs(original(x / norm_x)))

        # Estimate ||F|| on full space
        max_F = 0.0
        for _ in range(n_test):
            x = np.random.randn(n)
            x /= np.linalg.norm(x)
            max_F = max(max_F, abs(extended(x)))

        if max_f < 1e-15:
            return 1.0
        return max_F / max_f


# ============================================================================
# HAAR MEASURE (AN4.6)
# ============================================================================

class HaarMeasure:
    """Left-invariant (Haar) measure on finite groups.

    For a finite group G, the Haar measure is the counting measure
    normalized by |G|: μ(A) = |A|/|G|.

    For matrix groups (represented by finite generating sets), we
    approximate the Haar measure via uniform sampling.

    The Haar measure is the unique (up to scalar) left-invariant
    measure: μ(gA) = μ(A) for all g ∈ G.

    Example:
        >>> hm = HaarMeasure()
        >>> # Cyclic group Z/4Z
        >>> elements = [0, 1, 2, 3]  # group elements
        >>> measure = hm.compute(elements)
        >>> measure  # [0.25, 0.25, 0.25, 0.25]
    """

    def compute(self, group_elements: list) -> np.ndarray:
        """Compute normalized Haar measure on a finite group.

        For finite groups, the Haar measure is simply the uniform
        probability measure.

        Args:
            group_elements: List of group elements.

        Returns:
            Measure values (uniform), shape (|G|,).
        """
        n = len(group_elements)
        if n == 0:
            return np.array([])
        return np.full(n, 1.0 / n)

    def verify_left_invariance(
        self,
        group_elements: list,
        operation: Callable,
        measure: np.ndarray,
        tol: float = 1e-10,
    ) -> bool:
        """Verify left-invariance: μ(gA) = μ(A) for all g.

        Args:
            group_elements: List of group elements.
            operation: Group operation (a, b) -> a*b.
            measure: Measure values.
            tol: Tolerance.

        Returns:
            True if left-invariant.
        """
        n = len(group_elements)
        elem_to_idx = {e: i for i, e in enumerate(group_elements)}

        for g in group_elements:
            # Left-translate: gA maps element a to g*a
            translated_measure = np.zeros(n)
            for i, a in enumerate(group_elements):
                ga = operation(g, a)
                if ga in elem_to_idx:
                    translated_measure[elem_to_idx[ga]] = measure[i]

            if not np.allclose(translated_measure, measure, atol=tol):
                return False

        return True

    def integrate(
        self,
        f: Callable,
        group_elements: list,
        measure: np.ndarray | None = None,
    ) -> float:
        """Integrate a function over the group w.r.t. Haar measure.

        ∫_G f dμ = (1/|G|) Σ_{g ∈ G} f(g)

        Args:
            f: Function on the group.
            group_elements: Group elements.
            measure: Haar measure (computed if None).

        Returns:
            Integral value.
        """
        if measure is None:
            measure = self.compute(group_elements)

        return sum(measure[i] * f(g) for i, g in enumerate(group_elements))
