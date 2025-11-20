"""
Differential Geometry - Smooth Manifolds, Tangent Bundles, Differential Forms
=============================================================================

Fundamental geometric structures for modern differential geometry.

Classes:
    SmoothManifold: Chart-based smooth manifold representation
    TangentSpace: Tangent vectors at a point
    TangentBundle: Collection of all tangent spaces
    DifferentialForm: k-forms for integration on manifolds
    VectorField: Smooth assignment of tangent vectors
    LieDerivative: Lie derivative along vector fields
    ExteriorCalculus: Wedge product, exterior derivative, Stokes' theorem

Applications:
    - General relativity (spacetime manifolds)
    - Classical mechanics (phase space)
    - Geometric deep learning
    - Control theory on manifolds
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Chart:
    """Local coordinate chart on a manifold."""
    domain: str  # Description of domain
    phi: Callable  # Coordinate map: M -> R^n
    phi_inv: Callable  # Inverse map: R^n -> M
    dimension: int

    def transition_map(self, other: 'Chart', point: np.ndarray) -> np.ndarray:
        """
        Transition map between charts: phi_2 ∘ phi_1^{-1}.
        Must be smooth for smooth manifold.
        """
        manifold_point = self.phi_inv(point)
        return other.phi(manifold_point)


class SmoothManifold:
    """
    Smooth manifold with atlas of charts.

    A manifold locally looks like R^n, globally may be curved.
    Examples: Spheres, tori, Lie groups, configuration spaces.
    """

    def __init__(self, dimension: int, name: str = "M"):
        self.dimension = dimension
        self.name = name
        self.atlas: List[Chart] = []

    def add_chart(self, chart: Chart):
        """Add coordinate chart to atlas."""
        if chart.dimension != self.dimension:
            raise ValueError(f"Chart dimension {chart.dimension} != manifold dimension {self.dimension}")
        self.atlas.append(chart)

    def verify_smoothness(self, chart1: Chart, chart2: Chart, test_points: np.ndarray) -> bool:
        """
        Verify transition maps are smooth (C^∞).
        In practice, check differentiability numerically.
        """
        # Simplified: check transition map is continuous
        try:
            for point in test_points:
                transition = chart1.transition_map(chart2, point)
                if not np.all(np.isfinite(transition)):
                    return False
            return True
        except:
            return False

    @staticmethod
    def circle(radius: float = 1.0) -> 'SmoothManifold':
        """S^1 (circle) as 1D manifold."""
        S1 = SmoothManifold(dimension=1, name="S^1")

        # Chart 1: Upper semicircle (θ ∈ (0, π))
        def phi1(theta):
            return np.array([theta])
        def phi1_inv(x):
            return x[0]
        chart1 = Chart("upper", phi1, phi1_inv, dimension=1)

        # Chart 2: Lower semicircle (θ ∈ (π, 2π))
        def phi2(theta):
            return np.array([theta - np.pi])
        def phi2_inv(x):
            return x[0] + np.pi
        chart2 = Chart("lower", phi2, phi2_inv, dimension=1)

        S1.add_chart(chart1)
        S1.add_chart(chart2)
        return S1

    @staticmethod
    def sphere(radius: float = 1.0) -> 'SmoothManifold':
        """S^2 (sphere) as 2D manifold with stereographic projection."""
        S2 = SmoothManifold(dimension=2, name="S^2")

        # North pole chart: stereographic projection
        def phi_N(p):
            x, y, z = p
            if abs(z - radius) < 1e-10:
                raise ValueError("North pole not in chart domain")
            u = x / (radius - z)
            v = y / (radius - z)
            return np.array([u, v])

        def phi_N_inv(coords):
            u, v = coords
            denom = 1 + u**2 + v**2
            x = 2 * radius * u / denom
            y = 2 * radius * v / denom
            z = radius * (u**2 + v**2 - 1) / denom
            return np.array([x, y, z])

        chart_N = Chart("north", phi_N, phi_N_inv, dimension=2)
        S2.add_chart(chart_N)
        return S2

    @staticmethod
    def torus() -> 'SmoothManifold':
        """T^2 = S^1 × S^1 (torus) as 2D manifold."""
        T2 = SmoothManifold(dimension=2, name="T^2")

        def phi(theta_phi):
            theta, phi = theta_phi
            return np.array([theta, phi])

        def phi_inv(coords):
            return coords

        chart = Chart("standard", phi, phi_inv, dimension=2)
        T2.add_chart(chart)
        return T2


class TangentSpace:
    """
    Tangent space T_p M at point p on manifold M.

    Vector space of tangent vectors (derivatives of curves through p).
    Dimension = dimension of manifold.
    """

    def __init__(self, manifold: SmoothManifold, point: np.ndarray):
        self.manifold = manifold
        self.point = point
        self.dimension = manifold.dimension

    def tangent_vector(self, components: np.ndarray) -> 'TangentVector':
        """Create tangent vector at this point."""
        if len(components) != self.dimension:
            raise ValueError(f"Tangent vector must have {self.dimension} components")
        return TangentVector(self, components)

    def differential(self, f: Callable, chart: Chart) -> Callable:
        """
        Differential of function f: M -> R.
        df: T_p M -> R (linear map on tangent space).
        """
        def df(v: 'TangentVector') -> float:
            # df(v) = directional derivative of f along v
            h = 1e-7
            point_coords = chart.phi(self.point)
            perturbed = point_coords + h * v.components
            f_perturbed = f(chart.phi_inv(perturbed))
            f_current = f(self.point)
            return (f_perturbed - f_current) / h
        return df


@dataclass
class TangentVector:
    """Tangent vector v ∈ T_p M."""
    tangent_space: TangentSpace
    components: np.ndarray  # In local coordinates

    def __add__(self, other: 'TangentVector') -> 'TangentVector':
        if self.tangent_space.point is not other.tangent_space.point:
            raise ValueError("Cannot add tangent vectors at different points")
        return TangentVector(self.tangent_space, self.components + other.components)

    def __mul__(self, scalar: float) -> 'TangentVector':
        return TangentVector(self.tangent_space, scalar * self.components)

    def norm(self, metric: Optional[np.ndarray] = None) -> float:
        """Norm of tangent vector (requires Riemannian metric)."""
        if metric is None:
            metric = np.eye(len(self.components))  # Euclidean metric
        return np.sqrt(self.components @ metric @ self.components)


class TangentBundle:
    """
    Tangent bundle TM = union of all tangent spaces.

    TM = {(p, v) : p ∈ M, v ∈ T_p M}
    Dimension = 2 × dim(M)
    """

    def __init__(self, manifold: SmoothManifold):
        self.manifold = manifold
        self.dimension = 2 * manifold.dimension

    def section(self, vector_field: Callable) -> 'VectorField':
        """
        Section of tangent bundle (vector field).
        Smooth assignment p |-> v(p) ∈ T_p M.
        """
        return VectorField(self.manifold, vector_field)

    def zero_section(self) -> 'VectorField':
        """Zero vector field: p |-> 0 ∈ T_p M."""
        def zero_field(p):
            return np.zeros(self.manifold.dimension)
        return VectorField(self.manifold, zero_field)


class VectorField:
    """
    Vector field X: M -> TM.

    Smooth assignment of tangent vector to each point.
    Acts as derivation on smooth functions.
    """

    def __init__(self, manifold: SmoothManifold, field: Callable):
        self.manifold = manifold
        self.field = field  # p |-> X(p) components

    def __call__(self, point: np.ndarray) -> np.ndarray:
        """Evaluate vector field at point."""
        return self.field(point)

    def apply_to_function(self, f: Callable, point: np.ndarray) -> float:
        """
        Apply vector field as derivation: X(f)(p).
        Directional derivative of f along X at p.
        """
        h = 1e-7
        X_p = self.field(point)
        f_perturbed = f(point + h * X_p)
        f_current = f(point)
        return (f_perturbed - f_current) / h

    def commutator(self, other: 'VectorField') -> 'VectorField':
        """
        Lie bracket [X, Y] of vector fields.
        [X, Y](f) = X(Y(f)) - Y(X(f))
        """
        def bracket_field(p):
            # Approximate [X, Y] via finite differences
            h = 1e-5
            X_p = self.field(p)
            Y_p = other.field(p)

            # Y at p + h*X
            Y_forward = other.field(p + h * X_p)
            # X at p + h*Y
            X_forward = self.field(p + h * Y_p)

            # [X,Y] ≈ (dY/dt along X) - (dX/dt along Y)
            bracket = (Y_forward - Y_p) / h - (X_forward - X_p) / h
            return bracket

        return VectorField(self.manifold, bracket_field)

    def flow(self, initial_point: np.ndarray, t: float, dt: float = 0.01) -> np.ndarray:
        """
        Integral curve (flow) of vector field.
        Solves dx/dt = X(x), x(0) = initial_point.
        """
        point = initial_point.copy()
        num_steps = int(abs(t) / dt)
        sign = np.sign(t)

        for _ in range(num_steps):
            X_p = self.field(point)
            point = point + sign * dt * X_p

        return point


class DifferentialForm:
    """
    Differential k-form on manifold.

    Alternating multilinear map: (T_p M)^k -> R
    Examples:
        0-form: smooth function f: M -> R
        1-form: covector field (like df)
        2-form: area element, symplectic form
        n-form: volume form
    """

    def __init__(self, manifold: SmoothManifold, degree: int, form_function: Callable):
        self.manifold = manifold
        self.degree = degree  # k in k-form
        self.form_function = form_function  # (p, v1, ..., vk) -> R

    def __call__(self, point: np.ndarray, *vectors: np.ndarray) -> float:
        """Evaluate k-form on k tangent vectors at point."""
        if len(vectors) != self.degree:
            raise ValueError(f"{self.degree}-form requires {self.degree} vectors")
        return self.form_function(point, *vectors)

    def wedge(self, other: 'DifferentialForm') -> 'DifferentialForm':
        """
        Wedge product ω ∧ η of differential forms.
        If ω is k-form and η is l-form, result is (k+l)-form.
        """
        k = self.degree
        l = other.degree

        def wedge_function(point, *vectors):
            if len(vectors) != k + l:
                raise ValueError(f"Wedge product requires {k+l} vectors")

            # Alternating sum over permutations (simplified)
            # For production: use proper antisymmetrization
            from itertools import permutations, combinations

            result = 0.0
            for indices_omega in combinations(range(k+l), k):
                indices_eta = [i for i in range(k+l) if i not in indices_omega]
                vecs_omega = [vectors[i] for i in indices_omega]
                vecs_eta = [vectors[i] for i in indices_eta]

                # Sign from permutation
                perm = list(indices_omega) + list(indices_eta)
                sign = 1  # Simplified: should compute sign of permutation

                omega_val = self.form_function(point, *vecs_omega)
                eta_val = other.form_function(point, *vecs_eta)
                result += sign * omega_val * eta_val

            return result / np.math.factorial(k) / np.math.factorial(l)

        return DifferentialForm(self.manifold, k + l, wedge_function)

    def exterior_derivative(self, chart: Chart) -> 'DifferentialForm':
        """
        Exterior derivative d: Ω^k -> Ω^{k+1}.

        Properties:
            d(d(ω)) = 0 (Poincaré lemma)
            d(ω ∧ η) = dω ∧ η + (-1)^k ω ∧ dη
        """
        k = self.degree

        def d_form_function(point, *vectors):
            # Numerical approximation of exterior derivative
            # For k-form ω, dω is (k+1)-form
            h = 1e-6
            result = 0.0

            # Simplified: alternating sum of directional derivatives
            for i, v in enumerate(vectors):
                # Derivative along v_i
                point_shifted = point + h * v
                omega_shifted = self.form_function(point_shifted, *[vectors[j] for j in range(len(vectors)) if j != i])
                omega_current = self.form_function(point, *[vectors[j] for j in range(len(vectors)) if j != i])
                deriv = (omega_shifted - omega_current) / h
                result += (-1)**i * deriv

            return result

        return DifferentialForm(self.manifold, k + 1, d_form_function)

    @staticmethod
    def dx(manifold: SmoothManifold, i: int) -> 'DifferentialForm':
        """Coordinate 1-form dx^i."""
        def dx_i(point, v):
            return v[i]
        return DifferentialForm(manifold, degree=1, form_function=dx_i)


class ExteriorCalculus:
    """
    Exterior calculus operations.

    Foundation of differential geometry and integration on manifolds.
    """

    @staticmethod
    def stokes_theorem(omega: DifferentialForm, manifold: SmoothManifold,
                      boundary: Optional[SmoothManifold] = None) -> str:
        """
        Stokes' theorem: ∫_M dω = ∫_{∂M} ω

        Generalizes:
            - Fundamental theorem of calculus (0D boundary)
            - Green's theorem (2D)
            - Divergence theorem (3D)
            - Classical Stokes (surface in R^3)
        """
        return (
            f"Stokes' Theorem:\n"
            f"∫_M dω = ∫_{{∂M}} ω\n"
            f"Manifold: {manifold.name} (dim {manifold.dimension})\n"
            f"Form: {omega.degree}-form\n"
            f"Exterior derivative: d({omega.degree}-form) = {omega.degree+1}-form\n"
            f"Integration domain: {omega.degree+1}-dimensional manifold\n"
            f"Boundary integral: {omega.degree}-form on {omega.degree}-dim boundary"
        )

    @staticmethod
    def integrate_form(omega: DifferentialForm, manifold: SmoothManifold,
                      region: Callable, chart: Chart) -> float:
        """
        Integrate n-form over n-dimensional manifold.
        ∫_M ω = ∫_U ω(∂/∂x^1, ..., ∂/∂x^n) dx^1...dx^n
        """
        if omega.degree != manifold.dimension:
            raise ValueError(f"Cannot integrate {omega.degree}-form over {manifold.dimension}-manifold")

        # Monte Carlo integration (simplified)
        n_samples = 10000
        samples = region(n_samples)  # Sample points in region

        integral = 0.0
        for point in samples:
            # Coordinate basis vectors
            basis = [np.eye(manifold.dimension)[i] for i in range(manifold.dimension)]
            form_value = omega(point, *basis)
            integral += form_value

        volume = 1.0  # Should compute volume of region
        return integral * volume / n_samples

    @staticmethod
    def closed_vs_exact(omega: DifferentialForm) -> Dict[str, str]:
        """
        Classification of differential forms.

        Closed: dω = 0
        Exact: ω = dη for some η

        Fact: Exact => Closed (d² = 0)
        Converse (Poincaré lemma): Closed => Exact locally
        """
        return {
            "closed": "dω = 0 (kernel of exterior derivative)",
            "exact": "ω = dη (image of exterior derivative)",
            "de_Rham_cohomology": "H^k(M) = {closed k-forms} / {exact k-forms}",
            "Poincaré_lemma": "On contractible manifold, closed => exact"
        }


class LieDerivative:
    """
    Lie derivative along vector field.

    Measures change of tensor field along flow of vector field.
    """

    @staticmethod
    def lie_derivative_function(X: VectorField, f: Callable, point: np.ndarray) -> float:
        """
        Lie derivative of function: L_X f = X(f).
        (Just directional derivative)
        """
        return X.apply_to_function(f, point)

    @staticmethod
    def lie_derivative_vector_field(X: VectorField, Y: VectorField) -> VectorField:
        """
        Lie derivative of vector field: L_X Y = [X, Y].
        (Lie bracket)
        """
        return X.commutator(Y)

    @staticmethod
    def lie_derivative_form(X: VectorField, omega: DifferentialForm) -> DifferentialForm:
        """
        Lie derivative of differential form: L_X ω.

        Cartan's formula: L_X = d ∘ i_X + i_X ∘ d
        where i_X is interior product (contraction with X).
        """
        # Simplified implementation
        def lie_omega(point, *vectors):
            h = 1e-6
            # Flow along X for time h
            point_flowed = X.flow(point, h)
            omega_flowed = omega(point_flowed, *vectors)
            omega_current = omega(point, *vectors)
            return (omega_flowed - omega_current) / h

        return DifferentialForm(omega.manifold, omega.degree, lie_omega)


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_sphere_tangent_space():
    """Example: Tangent space to S^2 at north pole."""
    S2 = SmoothManifold.sphere()
    north_pole = np.array([0, 0, 1])
    T_p_S2 = TangentSpace(S2, north_pole)

    # Tangent vectors (e.g., pointing east and north)
    v_east = T_p_S2.tangent_vector(np.array([1.0, 0.0]))
    v_north = T_p_S2.tangent_vector(np.array([0.0, 1.0]))

    return v_east, v_north


def example_vector_field_on_circle():
    """Example: Rotation vector field on S^1."""
    S1 = SmoothManifold.circle()

    # Vector field: d/dθ (tangent to circle, constant rotation)
    def rotation_field(theta):
        return np.array([1.0])  # Constant speed rotation

    X = VectorField(S1, rotation_field)

    # Flow: integrate from θ=0 for time t
    theta_0 = np.array([0.0])
    theta_t = X.flow(theta_0, t=np.pi/2)  # Quarter rotation

    return X, theta_t


def example_differential_forms():
    """Example: 1-form and 2-form on R^2."""
    R2 = SmoothManifold(dimension=2, name="R^2")

    # 1-form: ω = x dy
    def omega_1(point, v):
        x, y = point
        return x * v[1]
    omega = DifferentialForm(R2, degree=1, form_function=omega_1)

    # 2-form: volume form dx ∧ dy
    def volume_2(point, v1, v2):
        # Determinant of [v1, v2]
        return v1[0] * v2[1] - v1[1] * v2[0]
    vol = DifferentialForm(R2, degree=2, form_function=volume_2)

    return omega, vol


if __name__ == "__main__":
    print("Differential Geometry Module")
    print("=" * 60)

    # Test 1: Manifolds
    print("\n[Test 1] Smooth manifolds")
    S1 = SmoothManifold.circle()
    S2 = SmoothManifold.sphere()
    T2 = SmoothManifold.torus()
    print(f"S^1: {S1.name}, dim={S1.dimension}, charts={len(S1.atlas)}")
    print(f"S^2: {S2.name}, dim={S2.dimension}, charts={len(S2.atlas)}")
    print(f"T^2: {T2.name}, dim={T2.dimension}, charts={len(T2.atlas)}")

    # Test 2: Tangent spaces
    print("\n[Test 2] Tangent space to sphere")
    v_east, v_north = example_sphere_tangent_space()
    print(f"Tangent vector (east): {v_east.components}")
    print(f"Tangent vector (north): {v_north.components}")
    v_sum = v_east + v_north
    print(f"Sum: {v_sum.components}")

    # Test 3: Vector fields
    print("\n[Test 3] Vector field on circle")
    X, theta_t = example_vector_field_on_circle()
    print(f"Initial: θ=0")
    print(f"After flow t=π/2: θ={theta_t[0]:.4f} (expect ~1.571)")

    # Test 4: Differential forms
    print("\n[Test 4] Differential forms")
    omega, vol = example_differential_forms()
    point = np.array([1.0, 2.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    omega_val = omega(point, v1)
    vol_val = vol(point, v1, v2)
    print(f"1-form ω at (1,2) on v1: {omega_val} (expect 0)")
    print(f"2-form (volume) at (1,2) on (v1,v2): {vol_val} (expect 1)")

    # Test 5: Stokes' theorem
    print("\n[Test 5] Stokes' theorem")
    R2 = SmoothManifold(dimension=2, name="R^2")
    stokes = ExteriorCalculus.stokes_theorem(omega, R2)
    print(stokes)

    print("\n" + "=" * 60)
    print("All differential geometry tests complete!")
    print("Manifolds, tangent bundles, forms ready for Riemannian geometry.")
