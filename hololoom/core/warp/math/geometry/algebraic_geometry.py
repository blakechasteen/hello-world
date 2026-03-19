"""
Algebraic & Tropical Geometry
===============================

Scope slots: GE6.1 (affine varieties), GE6.2 (projective varieties),
GE6.3 (tropical geometry), GE6.4-6.6 (intersection theory, extensions).

Pure numpy implementations:
- Affine varieties: zero sets of polynomial systems
- Projective varieties: homogeneous polynomials over projective space
- Tropical geometry: min-plus algebra, tropical curves
- Intersection theory: Bezout's theorem, intersection numbers

Author: Claude Code
Date: March 2026
"""

import logging
from dataclasses import dataclass
from itertools import product as iterproduct

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# POLYNOMIAL REPRESENTATION
# ============================================================================

@dataclass
class Monomial:
    """A monomial c · x₁^{a₁} x₂^{a₂} ... xₙ^{aₙ}.

    Attributes:
        coeff: Coefficient.
        exponents: Tuple of exponents (a₁, ..., aₙ).
    """
    coeff: float
    exponents: tuple[int, ...]

    @property
    def degree(self) -> int:
        return sum(self.exponents)

    def evaluate(self, point: np.ndarray) -> float:
        result = self.coeff
        for x, e in zip(point, self.exponents):
            result *= x ** e
        return result


@dataclass
class MultiPoly:
    """Multivariate polynomial as a sum of monomials.

    Attributes:
        terms: List of monomials.
        n_vars: Number of variables.
    """
    terms: list[Monomial]
    n_vars: int

    @property
    def degree(self) -> int:
        if not self.terms:
            return 0
        return max(t.degree for t in self.terms)

    def evaluate(self, point: np.ndarray) -> float:
        return sum(t.evaluate(point) for t in self.terms)

    def is_homogeneous(self) -> bool:
        """Check if all terms have the same total degree."""
        if not self.terms:
            return True
        d = self.terms[0].degree
        return all(t.degree == d for t in self.terms)

    @staticmethod
    def from_coeffs(coeffs: dict[tuple[int, ...], float], n_vars: int) -> 'MultiPoly':
        """Create polynomial from coefficient dictionary.

        Args:
            coeffs: Map from exponent tuple to coefficient.
            n_vars: Number of variables.

        Returns:
            MultiPoly.

        Example:
            >>> # x² + y² - 1
            >>> p = MultiPoly.from_coeffs({(2,0): 1, (0,2): 1, (0,0): -1}, n_vars=2)
        """
        terms = [Monomial(coeff=c, exponents=e) for e, c in coeffs.items() if abs(c) > 1e-15]
        return MultiPoly(terms=terms, n_vars=n_vars)


# ============================================================================
# AFFINE VARIETY (GE6.1)
# ============================================================================

class AffineVariety:
    """Affine algebraic variety V(I) = {x ∈ k^n : f(x) = 0 ∀f ∈ I}.

    An affine variety is the common zero set of a collection of polynomials.
    Over finite fields, we can enumerate all points. Over R, we use
    numerical root finding.

    Example:
        >>> # Circle: x² + y² = 1
        >>> circle = AffineVariety.from_polynomials([
        ...     MultiPoly.from_coeffs({(2,0): 1, (0,2): 1, (0,0): -1}, 2)
        ... ])
        >>> circle.degree()
        2
    """

    def __init__(self, polynomials: list[MultiPoly], n_vars: int):
        self.polynomials = polynomials
        self.n_vars = n_vars

    @staticmethod
    def from_polynomials(polys: list[MultiPoly]) -> 'AffineVariety':
        """Create variety from defining polynomials."""
        if not polys:
            raise ValueError("Need at least one polynomial")
        n_vars = polys[0].n_vars
        return AffineVariety(polynomials=polys, n_vars=n_vars)

    def dimension(self) -> int:
        """Estimate dimension of the variety.

        For a variety defined by m independent equations in n variables,
        dim V = n - m (generically).

        Returns:
            Estimated dimension.
        """
        m = len(self.polynomials)
        return max(0, self.n_vars - m)

    def degree(self) -> int:
        """Degree of the variety (product of polynomial degrees by Bezout).

        For a complete intersection, degree = product of defining
        polynomial degrees.

        Returns:
            Degree (upper bound by Bezout's theorem).
        """
        if not self.polynomials:
            return 0
        d = 1
        for p in self.polynomials:
            d *= p.degree
        return d

    def points_over_field(self, field_size: int) -> list[np.ndarray]:
        """Enumerate all points of the variety over F_p.

        Brute-force evaluation over the finite field Z/pZ.

        Args:
            field_size: Prime p (field characteristic).

        Returns:
            List of points [x₁, ..., xₙ] ∈ F_p^n on the variety.
        """
        p = field_size
        points = []

        for pt in iterproduct(range(p), repeat=self.n_vars):
            x = np.array(pt, dtype=float)
            if all(abs(poly.evaluate(x) % p) < 1e-10 for poly in self.polynomials):
                points.append(x)

        return points

    def is_on_variety(self, point: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a point lies on the variety."""
        return all(abs(poly.evaluate(point)) < tol for poly in self.polynomials)

    def sample_points(self, n_samples: int = 1000, bounds: float = 5.0, tol: float = 0.01) -> list[np.ndarray]:
        """Sample approximate points on the variety via rejection.

        Args:
            n_samples: Number of candidate samples.
            bounds: Sample from [-bounds, bounds]^n.
            tol: Tolerance for being "on" the variety.

        Returns:
            List of approximate points.
        """
        points = []
        for _ in range(n_samples):
            x = np.random.uniform(-bounds, bounds, self.n_vars)
            vals = [abs(poly.evaluate(x)) for poly in self.polynomials]
            if max(vals) < tol:
                points.append(x)
        return points


# ============================================================================
# PROJECTIVE VARIETY (GE6.2)
# ============================================================================

class ProjectiveVariety:
    """Projective algebraic variety in P^n.

    Points of P^n are equivalence classes [x₀ : x₁ : ... : xₙ]
    where (x₀, ..., xₙ) ≠ 0 and [λx₀ : ... : λxₙ] = [x₀ : ... : xₙ].

    A projective variety is defined by homogeneous polynomials.

    Example:
        >>> # Projective circle: X² + Y² - Z² = 0
        >>> pv = ProjectiveVariety.from_homogeneous([
        ...     MultiPoly.from_coeffs({(2,0,0): 1, (0,2,0): 1, (0,0,2): -1}, 3)
        ... ])
    """

    def __init__(self, polynomials: list[MultiPoly], ambient_dim: int):
        """Initialize projective variety.

        Args:
            polynomials: List of homogeneous polynomials in n+1 variables.
            ambient_dim: Dimension n of projective space P^n.
        """
        self.polynomials = polynomials
        self.ambient_dim = ambient_dim  # n in P^n

        for p in polynomials:
            if not p.is_homogeneous():
                logger.warning(f"Polynomial of degree {p.degree} is not homogeneous")

    @staticmethod
    def from_homogeneous(polys: list[MultiPoly]) -> 'ProjectiveVariety':
        """Create projective variety from homogeneous polynomials."""
        if not polys:
            raise ValueError("Need at least one polynomial")
        n = polys[0].n_vars - 1  # P^n has n+1 homogeneous coordinates
        return ProjectiveVariety(polynomials=polys, ambient_dim=n)

    def dimension(self) -> int:
        """Estimated dimension in projective space."""
        m = len(self.polynomials)
        return max(0, self.ambient_dim - m)

    def degree(self) -> int:
        """Degree by Bezout's theorem."""
        d = 1
        for p in self.polynomials:
            d *= p.degree
        return d

    def points_over_field(self, field_size: int) -> list[np.ndarray]:
        """Enumerate points in P^n(F_p).

        Points are normalized representatives: the first nonzero
        coordinate is 1.

        Args:
            field_size: Prime p.

        Returns:
            List of normalized homogeneous coordinates.
        """
        p = field_size
        n = self.ambient_dim + 1  # Number of homogeneous coordinates
        points = []
        seen = set()

        for pt in iterproduct(range(p), repeat=n):
            if all(x == 0 for x in pt):
                continue

            # Normalize: divide by first nonzero coordinate
            x = np.array(pt, dtype=float)
            first_nonzero = next(i for i, v in enumerate(pt) if v != 0)
            x = (x * pow(int(pt[first_nonzero]), p - 2, p)) % p  # Modular inverse

            key = tuple(int(v) % p for v in x)
            if key in seen:
                continue
            seen.add(key)

            if all(abs(poly.evaluate(x) % p) < 1e-10 for poly in self.polynomials):
                points.append(x)

        return points

    def affine_chart(self, index: int = 0) -> AffineVariety:
        """Dehomogenize to get affine variety in the i-th chart.

        Sets x_i = 1 in all polynomials.

        Args:
            index: Which coordinate to set to 1.

        Returns:
            Affine variety in the i-th chart.
        """
        affine_polys = []
        for poly in self.polynomials:
            new_terms = []
            for term in poly.terms:
                exps = list(term.exponents)
                # Remove the index-th coordinate (set to 1)
                new_exp = tuple(exps[:index] + exps[index + 1:])
                new_terms.append(Monomial(coeff=term.coeff, exponents=new_exp))
            affine_polys.append(MultiPoly(terms=new_terms, n_vars=self.ambient_dim))

        return AffineVariety(polynomials=affine_polys, n_vars=self.ambient_dim)


# ============================================================================
# TROPICAL GEOMETRY (GE6.3)
# ============================================================================

class TropicalGeometry:
    """Tropical geometry: algebraic geometry over the tropical semiring.

    The tropical semiring (R ∪ {∞}, ⊕, ⊙):
    - Tropical addition: a ⊕ b = min(a, b)
    - Tropical multiplication: a ⊙ b = a + b

    A tropical polynomial p(x) = min_i(a_i + Σ_j c_{ij} x_j) is a
    piecewise-linear function. Its "zero set" (tropical variety) is
    the set of points where the minimum is achieved by at least two terms.

    Tropical geometry connects to:
    - Amoebas of complex varieties
    - Newton polytopes and polyhedral geometry
    - Enumerative geometry (Mikhalkin's theorem)

    Example:
        >>> tg = TropicalGeometry()
        >>> tg.tropical_add(3, 5)  # min(3, 5) = 3
        3
        >>> tg.tropical_mul(3, 5)  # 3 + 5 = 8
        8
    """

    @staticmethod
    def tropical_add(a: float, b: float) -> float:
        """Tropical addition: a ⊕ b = min(a, b)."""
        return min(a, b)

    @staticmethod
    def tropical_mul(a: float, b: float) -> float:
        """Tropical multiplication: a ⊙ b = a + b."""
        return a + b

    @staticmethod
    def tropical_zero() -> float:
        """Tropical additive identity (∞)."""
        return float('inf')

    @staticmethod
    def tropical_one() -> float:
        """Tropical multiplicative identity (0)."""
        return 0.0

    def tropical_polynomial_eval(
        self,
        coeffs: dict[tuple[int, ...], float],
        point: np.ndarray,
    ) -> float:
        """Evaluate tropical polynomial at a point.

        trop(p)(x) = min_i(a_i + Σ_j e_{ij} · x_j)

        Args:
            coeffs: Map from exponent tuple to tropical coefficient.
            point: Evaluation point.

        Returns:
            Tropical polynomial value.
        """
        val = float('inf')
        for exponents, coeff in coeffs.items():
            term = coeff + sum(e * x for e, x in zip(exponents, point))
            val = min(val, term)
        return val

    def tropical_curve(
        self,
        coeffs: dict[tuple[int, ...], float],
        x_range: tuple[float, float] = (-5, 5),
        y_range: tuple[float, float] = (-5, 5),
        resolution: int = 200,
    ) -> np.ndarray:
        """Compute tropical curve (2D) as the non-smooth locus.

        The tropical curve is where the minimum is achieved by at least
        two monomials — the "corners" of the piecewise-linear function.

        Args:
            coeffs: Tropical polynomial coefficients (2 variables).
            x_range, y_range: Domain.
            resolution: Grid resolution.

        Returns:
            Boolean mask of shape (resolution, resolution) indicating
            points on/near the tropical curve.
        """
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)

        curve = np.zeros((resolution, resolution), dtype=bool)

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                point = np.array([xi, yj])

                # Evaluate each monomial
                term_values = []
                for exponents, coeff in coeffs.items():
                    term = coeff + exponents[0] * xi + exponents[1] * yj
                    term_values.append(term)

                if len(term_values) < 2:
                    continue

                term_values.sort()
                # On the curve: two smallest values are close
                if abs(term_values[0] - term_values[1]) < 0.1 * (x_range[1] - x_range[0]) / resolution:
                    curve[j, i] = True

        return curve

    @staticmethod
    def newton_polygon(coeffs: dict[tuple[int, ...], float]) -> list[tuple[int, ...]]:
        """Compute Newton polygon (convex hull of exponent vectors).

        The Newton polygon of a polynomial Σ a_I x^I is the convex
        hull of {I : a_I ≠ 0} in R^n.

        Args:
            coeffs: Polynomial coefficient dictionary.

        Returns:
            Vertices of the Newton polygon.
        """
        points = [np.array(e, dtype=float) for e in coeffs.keys()
                  if abs(coeffs[e]) > 1e-15]

        if len(points) <= 1:
            return [tuple(int(x) for x in p) for p in points]

        if len(points[0]) == 2:
            # 2D convex hull via gift wrapping
            return TropicalGeometry._convex_hull_2d(points)

        # Higher dimensions: return all points (hull computation complex)
        return [tuple(int(x) for x in p) for p in points]

    @staticmethod
    def _convex_hull_2d(points: list[np.ndarray]) -> list[tuple[int, ...]]:
        """Simple 2D convex hull via gift wrapping."""
        pts = sorted(points, key=lambda p: (p[0], p[1]))
        if len(pts) <= 2:
            return [tuple(int(x) for x in p) for p in pts]

        # Build lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2:
                o, a, b = lower[-2], lower[-1], p
                if (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) <= 0:
                    lower.pop()
                else:
                    break
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2:
                o, a, b = upper[-2], upper[-1], p
                if (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) <= 0:
                    upper.pop()
                else:
                    break
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return [tuple(int(x) for x in p) for p in hull]


# ============================================================================
# INTERSECTION THEORY (GE6.4-6.6)
# ============================================================================

class IntersectionTheory:
    """Intersection theory for algebraic varieties.

    Computes intersection numbers and applies Bezout's theorem.

    Bezout's theorem: in P^n, varieties V₁, ..., Vₙ of degrees d₁, ..., dₙ
    intersect in exactly d₁ · d₂ · ... · dₙ points (counted with multiplicity),
    assuming the intersection is zero-dimensional and transverse.

    Example:
        >>> it = IntersectionTheory()
        >>> it.bezout_bound([2, 3])  # Line meets conic in 6 points (P²)
        6
    """

    @staticmethod
    def intersection_number(V1: AffineVariety, V2: AffineVariety,
                           field_size: int = 97) -> int:
        """Estimate intersection number by counting points over F_p.

        For varieties V₁, V₂ ⊂ A^n, count |V₁ ∩ V₂| over F_p.
        This gives a lower bound on the geometric intersection number.

        Args:
            V1, V2: Affine varieties.
            field_size: Prime for finite field computation.

        Returns:
            Number of intersection points over F_p.
        """
        combined_polys = V1.polynomials + V2.polynomials
        intersection = AffineVariety(
            polynomials=combined_polys,
            n_vars=V1.n_vars,
        )
        points = intersection.points_over_field(field_size)
        return len(points)

    @staticmethod
    def bezout_bound(degrees: list[int]) -> int:
        """Compute Bezout bound: product of degrees.

        By Bezout's theorem, n hypersurfaces of degrees d₁,...,dₙ in P^n
        intersect in at most d₁·d₂·...·dₙ points (with multiplicity).

        Args:
            degrees: Degrees of the hypersurfaces.

        Returns:
            Bezout bound.
        """
        bound = 1
        for d in degrees:
            bound *= d
        return bound

    @staticmethod
    def self_intersection(poly: MultiPoly, field_size: int = 97) -> int:
        """Compute self-intersection number via partial derivatives.

        For a curve f(x,y) = 0, the self-intersection relates to
        the number of singular points (where f = f_x = f_y = 0).

        Args:
            poly: Defining polynomial (2 variables).
            field_size: Prime for computation.

        Returns:
            Number of singular points over F_p.
        """
        if poly.n_vars != 2:
            raise ValueError("Self-intersection requires 2-variable polynomial")

        p = field_size
        singular = 0

        for pt in iterproduct(range(p), repeat=2):
            x = np.array(pt, dtype=float)
            val = poly.evaluate(x) % p

            if abs(val) > 1e-10:
                continue

            # Numerical partial derivatives
            dx = np.array([1.0, 0.0])
            dy = np.array([0.0, 1.0])
            fx = (poly.evaluate(x + dx) - poly.evaluate(x - dx)) / 2
            fy = (poly.evaluate(x + dy) - poly.evaluate(x - dy)) / 2

            if abs(fx % p) < 1e-10 and abs(fy % p) < 1e-10:
                singular += 1

        return singular

    @staticmethod
    def genus_formula(degree: int) -> int:
        """Genus of a smooth projective plane curve of degree d.

        By the genus-degree formula: g = (d-1)(d-2)/2.

        Args:
            degree: Degree of the curve.

        Returns:
            Genus.
        """
        return (degree - 1) * (degree - 2) // 2
