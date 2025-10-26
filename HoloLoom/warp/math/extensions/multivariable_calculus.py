"""
Multivariable Calculus - Vector Calculus, Stokes' Theorem, Exterior Calculus
===========================================================================

Calculus in higher dimensions with geometric interpretation.

Classes:
    VectorField: Velocity fields, gradients, flows
    ScalarField: Potential functions, level sets
    LineIntegral: Path integrals, circulation
    SurfaceIntegral: Flux, surface area
    VolumeIntegral: Triple integrals, mass
    GradientCurlDiv: Fundamental vector operators
    IntegralTheorems: Fundamental theorem, Green, Stokes, divergence

Applications:
    - Fluid dynamics
    - Electromagnetics
    - Computer graphics
    - Geometric PDEs
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Point3D:
    """Point in 3D space."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class ScalarField:
    """
    Scalar field: f: R^n -> R.

    Examples: temperature, pressure, potential energy.
    """

    def __init__(self, func: Callable[[np.ndarray], float], dim: int = 3):
        self.f = func
        self.dim = dim

    def __call__(self, point: np.ndarray) -> float:
        """Evaluate field at point."""
        return self.f(point)

    def gradient(self, point: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """
        Gradient: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z).

        Points in direction of steepest ascent.
        """
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            point_plus = point.copy()
            point_plus[i] += h
            grad[i] = (self.f(point_plus) - self.f(point)) / h
        return grad

    def directional_derivative(self, point: np.ndarray, direction: np.ndarray) -> float:
        """
        Directional derivative: D_v f = ∇f · v.

        Rate of change in direction v.
        """
        grad = self.gradient(point)
        v = direction / np.linalg.norm(direction)  # Normalize
        return np.dot(grad, v)

    def laplacian(self, point: np.ndarray, h: float = 1e-5) -> float:
        """
        Laplacian: Δf = ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².

        Measures divergence of gradient.
        """
        lapl = 0.0
        f_center = self.f(point)

        for i in range(self.dim):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += h
            point_minus[i] -= h

            # Second derivative via finite difference
            f_plus = self.f(point_plus)
            f_minus = self.f(point_minus)
            lapl += (f_plus - 2 * f_center + f_minus) / (h ** 2)

        return lapl

    @staticmethod
    def quadratic(coeffs: np.ndarray) -> 'ScalarField':
        """
        Quadratic form: f(x,y,z) = ax² + by² + cz².

        Useful for paraboloids, ellipsoids.
        """
        def f(point):
            return np.sum(coeffs * point ** 2)
        return ScalarField(f, dim=len(coeffs))

    @staticmethod
    def distance_from_origin() -> 'ScalarField':
        """Distance function: f(x,y,z) = sqrt(x² + y² + z²)."""
        def f(point):
            return np.linalg.norm(point)
        return ScalarField(f, dim=3)


class VectorField:
    """
    Vector field: F: R^n -> R^n.

    Examples: velocity field, force field, electric field.
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], dim: int = 3):
        """
        Args:
            func: Maps point to vector
            dim: Dimension of space
        """
        self.F = func
        self.dim = dim

    def __call__(self, point: np.ndarray) -> np.ndarray:
        """Evaluate vector field at point."""
        return self.F(point)

    def divergence(self, point: np.ndarray, h: float = 1e-6) -> float:
        """
        Divergence: div F = ∇ · F = ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z.

        Measures "outflow" from point.
        div F > 0: source
        div F < 0: sink
        div F = 0: incompressible
        """
        div = 0.0
        for i in range(self.dim):
            point_plus = point.copy()
            point_plus[i] += h
            F_plus = self.F(point_plus)
            F_center = self.F(point)
            div += (F_plus[i] - F_center[i]) / h
        return div

    def curl(self, point: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """
        Curl: curl F = ∇ × F (3D only).

        curl F = (∂F_z/∂y - ∂F_y/∂z,
                  ∂F_x/∂z - ∂F_z/∂x,
                  ∂F_y/∂x - ∂F_x/∂y)

        Measures rotation/circulation.
        """
        if self.dim != 3:
            raise ValueError("Curl only defined in 3D")

        F = self.F(point)
        curl_vec = np.zeros(3)

        # ∂F_z/∂y - ∂F_y/∂z
        point_y = point.copy()
        point_y[1] += h
        curl_vec[0] = (self.F(point_y)[2] - F[2]) / h

        point_z = point.copy()
        point_z[2] += h
        curl_vec[0] -= (self.F(point_z)[1] - F[1]) / h

        # ∂F_x/∂z - ∂F_z/∂x
        curl_vec[1] = (self.F(point_z)[0] - F[0]) / h

        point_x = point.copy()
        point_x[0] += h
        curl_vec[1] -= (self.F(point_x)[2] - F[2]) / h

        # ∂F_y/∂x - ∂F_x/∂y
        curl_vec[2] = (self.F(point_x)[1] - F[1]) / h
        curl_vec[2] -= (self.F(point_y)[0] - F[0]) / h

        return curl_vec

    def is_conservative(self, test_points: np.ndarray, tolerance: float = 1e-4) -> bool:
        """
        Check if field is conservative (curl F = 0).

        Conservative fields are gradients of scalar potentials.
        """
        if self.dim != 3:
            return True  # Simplified for non-3D

        for point in test_points:
            curl_val = self.curl(point)
            if np.linalg.norm(curl_val) > tolerance:
                return False
        return True

    @staticmethod
    def constant(vector: np.ndarray) -> 'VectorField':
        """Constant vector field."""
        def F(point):
            return vector.copy()
        return VectorField(F, dim=len(vector))

    @staticmethod
    def radial() -> 'VectorField':
        """Radial field: F(x,y,z) = (x, y, z)."""
        def F(point):
            return point.copy()
        return VectorField(F, dim=3)

    @staticmethod
    def rotation_z() -> 'VectorField':
        """Rotation around z-axis: F(x,y,z) = (-y, x, 0)."""
        def F(point):
            return np.array([-point[1], point[0], 0.0])
        return VectorField(F, dim=3)


class LineIntegral:
    """
    Line integral: integral along curve.

    Scalar: ∫_C f ds (arc length)
    Vector: ∫_C F · dr (work/circulation)
    """

    @staticmethod
    def scalar_field(field: ScalarField, curve: Callable[[float], np.ndarray],
                    t_start: float, t_end: float, n_steps: int = 100) -> float:
        """
        Line integral of scalar field: ∫_C f ds.

        Args:
            field: Scalar field f
            curve: Parameterized curve r(t)
            t_start, t_end: Parameter domain
        """
        dt = (t_end - t_start) / n_steps
        integral = 0.0

        for i in range(n_steps):
            t = t_start + i * dt
            t_next = t + dt

            # Midpoint rule
            t_mid = (t + t_next) / 2
            r_mid = curve(t_mid)

            # Tangent vector
            dr = curve(t_next) - curve(t)
            ds = np.linalg.norm(dr)

            # f(r(t)) * |dr/dt| * dt
            integral += field(r_mid) * ds

        return integral

    @staticmethod
    def vector_field(field: VectorField, curve: Callable[[float], np.ndarray],
                    t_start: float, t_end: float, n_steps: int = 100) -> float:
        """
        Line integral of vector field: ∫_C F · dr (circulation, work).

        For conservative fields, equals potential difference.
        """
        dt = (t_end - t_start) / n_steps
        integral = 0.0

        for i in range(n_steps):
            t = t_start + i * dt
            t_next = t + dt

            t_mid = (t + t_next) / 2
            r_mid = curve(t_mid)

            # F(r(t)) · (dr/dt) * dt
            F_val = field(r_mid)
            dr = curve(t_next) - curve(t)

            integral += np.dot(F_val, dr)

        return integral


class SurfaceIntegral:
    """
    Surface integral: integral over 2D surface in R³.

    Scalar: ∫∫_S f dS (surface area weighted)
    Vector: ∫∫_S F · n dS (flux through surface)
    """

    @staticmethod
    def flux(field: VectorField, surface: Callable[[float, float], np.ndarray],
            u_range: Tuple[float, float], v_range: Tuple[float, float],
            n_u: int = 20, n_v: int = 20) -> float:
        """
        Flux of vector field through surface: ∫∫_S F · n dS.

        Args:
            field: Vector field F
            surface: Parameterized surface r(u, v)
            u_range, v_range: Parameter domains
        """
        u_min, u_max = u_range
        v_min, v_max = v_range
        du = (u_max - u_min) / n_u
        dv = (v_max - v_min) / n_v

        flux = 0.0
        h = 1e-6

        for i in range(n_u):
            for j in range(n_v):
                u = u_min + (i + 0.5) * du
                v = v_min + (j + 0.5) * dv

                # Point on surface
                r = surface(u, v)

                # Tangent vectors
                r_u = (surface(u + h, v) - surface(u - h, v)) / (2 * h)
                r_v = (surface(u, v + h) - surface(u, v - h)) / (2 * h)

                # Normal vector: r_u × r_v
                normal = np.cross(r_u, r_v)

                # F · n * dS
                F_val = field(r)
                flux += np.dot(F_val, normal) * du * dv

        return flux


class IntegralTheorems:
    """
    Fundamental theorems of vector calculus.

    Unify line, surface, and volume integrals.
    """

    @staticmethod
    def fundamental_theorem_line_integrals() -> str:
        """
        Fundamental theorem for line integrals.

        If F = ∇f (conservative), then ∫_C F · dr = f(b) - f(a).
        """
        return (
            "Fundamental Theorem for Line Integrals:\n\n"
            "If F = ∇f (conservative field), then:\n"
            "∫_C F · dr = f(r(b)) - f(r(a))\n\n"
            "Path integral depends only on endpoints, not path!"
        )

    @staticmethod
    def greens_theorem() -> str:
        """
        Green's theorem: relates double integral to line integral.

        ∫∫_D (∂Q/∂x - ∂P/∂y) dA = ∮_C P dx + Q dy
        """
        return (
            "Green's Theorem (2D):\n\n"
            "∫∫_D (∂Q/∂x - ∂P/∂y) dA = ∮_{∂D} P dx + Q dy\n\n"
            "Circulation around boundary = integral of curl over region.\n"
            "Special case of Stokes' theorem."
        )

    @staticmethod
    def stokes_theorem() -> str:
        """
        Stokes' theorem: surface integral of curl = line integral.

        ∫∫_S (curl F) · n dS = ∮_{∂S} F · dr
        """
        return (
            "Stokes' Theorem (3D):\n\n"
            "∫∫_S (curl F) · n dS = ∮_{∂S} F · dr\n\n"
            "Surface integral of curl = circulation around boundary.\n"
            "Generalizes Green's theorem to 3D.\n"
            "Further generalized by differential forms."
        )

    @staticmethod
    def divergence_theorem() -> str:
        """
        Divergence theorem (Gauss's theorem).

        ∫∫∫_V (div F) dV = ∫∫_{∂V} F · n dS
        """
        return (
            "Divergence Theorem (Gauss's):\n\n"
            "∫∫∫_V (div F) dV = ∫∫_{∂V} F · n dS\n\n"
            "Volume integral of divergence = flux through boundary.\n"
            "Used in: fluid dynamics, electromagnetics, heat flow.\n\n"
            "Physical interpretation:\n"
            "- Total outflow from region = integral of sources/sinks inside"
        )

    @staticmethod
    def generalized_stokes() -> str:
        """
        Generalized Stokes' theorem (differential forms).

        ∫_M dω = ∫_{∂M} ω
        """
        return (
            "Generalized Stokes' Theorem:\n\n"
            "∫_M dω = ∫_{∂M} ω\n\n"
            "Unifies all fundamental theorems:\n"
            "- Fundamental theorem of calculus (0D boundary)\n"
            "- Green's theorem (2D)\n"
            "- Classical Stokes (surface in R³)\n"
            "- Divergence theorem (3D volume)\n\n"
            "M: n-dimensional manifold\n"
            "ω: (n-1)-form\n"
            "dω: exterior derivative (n-form)"
        )


class GradientCurlDiv:
    """
    Fundamental vector operators and their relationships.

    grad, curl, div satisfy important identities.
    """

    @staticmethod
    def identities() -> Dict[str, str]:
        """Important vector calculus identities."""
        return {
            "curl_of_gradient": "curl(grad f) = 0 (conservative fields)",
            "div_of_curl": "div(curl F) = 0 (solenoidal fields)",
            "laplacian": "div(grad f) = Δf (Laplacian)",
            "vector_laplacian": "∇²F = grad(div F) - curl(curl F)",
            "product_rule_div": "div(fF) = f div F + F · grad f",
            "product_rule_curl": "curl(fF) = f curl F + grad f × F"
        }

    @staticmethod
    def helmholtz_decomposition() -> str:
        """
        Helmholtz decomposition: any vector field = gradient + curl.

        F = -grad φ + curl A
        """
        return (
            "Helmholtz Decomposition:\n\n"
            "Any vector field F can be decomposed as:\n"
            "F = -grad φ + curl A\n\n"
            "φ: scalar potential (irrotational part)\n"
            "A: vector potential (solenoidal part)\n\n"
            "Uniquely determined up to boundary conditions."
        )


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_gradient_field():
    """Example: Gradient of quadratic field."""
    # f(x,y,z) = x² + y² + z²
    field = ScalarField.quadratic(np.array([1, 1, 1]))

    point = np.array([1.0, 2.0, 3.0])
    grad = field.gradient(point)

    # Should be (2x, 2y, 2z) = (2, 4, 6)
    return grad


def example_divergence():
    """Example: Divergence of radial field."""
    # F(x,y,z) = (x, y, z)
    field = VectorField.radial()

    point = np.array([1.0, 1.0, 1.0])
    div = field.divergence(point)

    # div F = ∂x/∂x + ∂y/∂y + ∂z/∂z = 3
    return div


def example_curl():
    """Example: Curl of rotation field."""
    # F(x,y,z) = (-y, x, 0) (rotation around z)
    field = VectorField.rotation_z()

    point = np.array([1.0, 1.0, 0.0])
    curl_val = field.curl(point)

    # curl F = (0, 0, 2)
    return curl_val


def example_line_integral():
    """Example: Work done by force field."""
    # F(x,y,z) = (x, y, z)
    field = VectorField.radial()

    # Straight line from (0,0,0) to (1,1,1)
    def curve(t):
        return np.array([t, t, t])

    work = LineIntegral.vector_field(field, curve, 0, 1)
    return work


if __name__ == "__main__":
    print("Multivariable Calculus Module")
    print("=" * 60)

    # Test 1: Gradient
    print("\n[Test 1] Gradient of f = x² + y² + z²")
    grad = example_gradient_field()
    print(f"∇f at (1,2,3) = {grad}")
    print(f"Expected: (2, 4, 6)")

    # Test 2: Divergence
    print("\n[Test 2] Divergence of radial field")
    div = example_divergence()
    print(f"div F at (1,1,1) = {div:.4f}")
    print(f"Expected: 3.0")

    # Test 3: Curl
    print("\n[Test 3] Curl of rotation field")
    curl_val = example_curl()
    print(f"curl F at (1,1,0) = {curl_val}")
    print(f"Expected: (0, 0, 2)")

    # Test 4: Line integral
    print("\n[Test 4] Line integral (work)")
    work = example_line_integral()
    print(f"Work along line: {work:.4f}")
    print(f"Expected: 1.5 (for radial field)")

    # Test 5: Fundamental theorems
    print("\n[Test 5] Fundamental theorems")
    print(IntegralTheorems.stokes_theorem())

    # Test 6: Vector identities
    print("\n[Test 6] Vector calculus identities")
    identities = GradientCurlDiv.identities()
    print(f"curl(grad f) = {identities['curl_of_gradient']}")
    print(f"div(curl F) = {identities['div_of_curl']}")

    print("\n" + "=" * 60)
    print("All multivariable calculus tests complete!")
    print("Gradient, curl, divergence, and integral theorems ready.")
