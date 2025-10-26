"""
Riemannian Geometry - Metrics, Curvature, Geodesics
===================================================

Geometry on smooth manifolds with inner product structure.

Classes:
    RiemannianMetric: Inner product on tangent spaces
    Christoffel: Connection coefficients (Levi-Civita connection)
    RiemannCurvature: Curvature tensor and derived quantities
    Geodesic: Shortest paths on manifolds
    ParallelTransport: Moving vectors along curves
    CurvatureAnalysis: Sectional, Ricci, scalar curvature
    RicciFlow: Evolution of metric (geometric PDE)

Applications:
    - General relativity (curved spacetime)
    - Shape analysis in computer vision
    - Geometric deep learning
    - Optimal transport on manifolds
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass


class RiemannianMetric:
    """
    Riemannian metric g on manifold.

    Inner product g_p: T_p M × T_p M -> R at each point p.
    In coordinates: g = g_{ij} dx^i ⊗ dx^j
    """

    def __init__(self, manifold_dim: int, metric_tensor: Callable):
        """
        Args:
            manifold_dim: Dimension of manifold
            metric_tensor: Function point -> g_{ij} (metric matrix)
        """
        self.dim = manifold_dim
        self.g = metric_tensor  # point -> matrix

    def __call__(self, point: np.ndarray) -> np.ndarray:
        """Metric tensor g_{ij} at point."""
        return self.g(point)

    def inner_product(self, point: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Inner product <v, w>_g = g_{ij} v^i w^j.
        """
        g_p = self.g(point)
        return float(v @ g_p @ w)

    def norm(self, point: np.ndarray, v: np.ndarray) -> float:
        """Norm ||v||_g = sqrt(<v, v>_g)."""
        return np.sqrt(max(0, self.inner_product(point, v, v)))

    def inverse(self, point: np.ndarray) -> np.ndarray:
        """Inverse metric g^{ij}."""
        g_p = self.g(point)
        return np.linalg.inv(g_p)

    def volume_form(self, point: np.ndarray) -> float:
        """Volume element sqrt(det(g))."""
        g_p = self.g(point)
        det_g = np.linalg.det(g_p)
        return np.sqrt(abs(det_g))

    @staticmethod
    def euclidean(dim: int) -> 'RiemannianMetric':
        """Euclidean metric: g_{ij} = δ_{ij}."""
        def g(point):
            return np.eye(dim)
        return RiemannianMetric(dim, g)

    @staticmethod
    def sphere(radius: float = 1.0) -> 'RiemannianMetric':
        """
        Metric on S^2 (sphere) in spherical coordinates (θ, φ).
        g = R² (dθ² + sin²θ dφ²)
        """
        def g(point):
            theta, phi = point
            return radius**2 * np.array([
                [1.0, 0.0],
                [0.0, np.sin(theta)**2]
            ])
        return RiemannianMetric(2, g)

    @staticmethod
    def hyperbolic(dim: int = 2) -> 'RiemannianMetric':
        """
        Hyperbolic metric (Poincaré disk model).
        g = (4 / (1 - r²)²) δ_{ij} where r² = x² + y²
        """
        def g(point):
            r_squared = np.sum(point**2)
            if r_squared >= 1.0:
                raise ValueError("Point outside Poincaré disk")
            scale = 4.0 / (1 - r_squared)**2
            return scale * np.eye(dim)
        return RiemannianMetric(dim, g)


class Christoffel:
    """
    Christoffel symbols (connection coefficients).

    Γ^k_{ij} defines Levi-Civita connection (torsion-free, metric-compatible).
    ∇_i ∂_j = Γ^k_{ij} ∂_k
    """

    def __init__(self, metric: RiemannianMetric):
        self.metric = metric
        self.dim = metric.dim

    def symbols(self, point: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """
        Christoffel symbols: Γ^k_{ij} = 1/2 g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})

        Returns: shape (dim, dim, dim) array Γ[k,i,j]
        """
        g = self.metric(point)
        g_inv = np.linalg.inv(g)
        Gamma = np.zeros((self.dim, self.dim, self.dim))

        # Numerical derivatives of metric
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    # ∂_i g_{jl}
                    point_i_plus = point.copy()
                    point_i_plus[i] += h
                    dg_i_jk = (self.metric(point_i_plus)[j, k] - g[j, k]) / h

                    # ∂_j g_{il}
                    point_j_plus = point.copy()
                    point_j_plus[j] += h
                    dg_j_ik = (self.metric(point_j_plus)[i, k] - g[i, k]) / h

                    # ∂_k g_{ij}
                    point_k_plus = point.copy()
                    point_k_plus[k] += h
                    dg_k_ij = (self.metric(point_k_plus)[i, j] - g[i, j]) / h

                    # Sum contributions
                    for l in range(self.dim):
                        point_l_plus = point.copy()
                        point_l_plus[l] += h
                        dg_i_jl = (self.metric(point_l_plus)[j, l] - g[j, l]) / h
                        dg_j_il = (self.metric(point_l_plus)[i, l] - g[i, l]) / h
                        dg_l_ij = (self.metric(point_l_plus)[i, j] - g[i, j]) / h

                        Gamma[k, i, j] += 0.5 * g_inv[k, l] * (dg_i_jl + dg_j_il - dg_l_ij)

        return Gamma

    def geodesic_equation(self, point: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Geodesic equation: d²x^k/dt² + Γ^k_{ij} dx^i/dt dx^j/dt = 0

        Returns: acceleration = -Γ^k_{ij} v^i v^j
        """
        Gamma = self.symbols(point)
        acceleration = np.zeros(self.dim)

        for k in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    acceleration[k] -= Gamma[k, i, j] * velocity[i] * velocity[j]

        return acceleration


class Geodesic:
    """
    Geodesics: shortest paths on Riemannian manifolds.

    Curves γ(t) satisfying ∇_{γ'} γ' = 0 (parallel transport of velocity).
    """

    def __init__(self, metric: RiemannianMetric):
        self.metric = metric
        self.christoffel = Christoffel(metric)

    def integrate(self, initial_point: np.ndarray, initial_velocity: np.ndarray,
                 t_final: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate geodesic equation from initial conditions.

        Returns: (points, velocities) arrays of shape (n_steps, dim)
        """
        n_steps = int(t_final / dt)
        points = np.zeros((n_steps, self.metric.dim))
        velocities = np.zeros((n_steps, self.metric.dim))

        points[0] = initial_point
        velocities[0] = initial_velocity

        for i in range(n_steps - 1):
            # Euler step: x_{n+1} = x_n + v_n dt
            points[i+1] = points[i] + velocities[i] * dt

            # Velocity update: v_{n+1} = v_n + a_n dt
            acceleration = self.christoffel.geodesic_equation(points[i], velocities[i])
            velocities[i+1] = velocities[i] + acceleration * dt

        return points, velocities

    def distance(self, point_a: np.ndarray, point_b: np.ndarray,
                n_samples: int = 100) -> float:
        """
        Approximate geodesic distance between two points.
        (Simplified: integrate along straight line in coordinates)
        """
        path = np.linspace(point_a, point_b, n_samples)
        distance = 0.0

        for i in range(n_samples - 1):
            p = path[i]
            dp = path[i+1] - path[i]
            length_element = self.metric.norm(p, dp)
            distance += length_element

        return distance


class RiemannCurvature:
    """
    Riemann curvature tensor and derived quantities.

    R^l_{ijk} measures non-commutativity of covariant derivatives.
    [∇_i, ∇_j] = R^l_{ijk} ∂_l (on vector fields)
    """

    def __init__(self, metric: RiemannianMetric):
        self.metric = metric
        self.christoffel = Christoffel(metric)
        self.dim = metric.dim

    def riemann_tensor(self, point: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Riemann curvature tensor:
        R^l_{ijk} = ∂_i Γ^l_{jk} - ∂_j Γ^l_{ik} + Γ^l_{im} Γ^m_{jk} - Γ^l_{jm} Γ^m_{ik}

        Returns: shape (dim, dim, dim, dim) array R[l,i,j,k]
        """
        Gamma = self.christoffel.symbols(point, h)
        R = np.zeros((self.dim, self.dim, self.dim, self.dim))

        for l in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        # Partial derivatives of Christoffel symbols
                        point_i = point.copy()
                        point_i[i] += h
                        Gamma_i = self.christoffel.symbols(point_i, h)
                        dGamma_i_ljk = (Gamma_i[l, j, k] - Gamma[l, j, k]) / h

                        point_j = point.copy()
                        point_j[j] += h
                        Gamma_j = self.christoffel.symbols(point_j, h)
                        dGamma_j_lik = (Gamma_j[l, i, k] - Gamma[l, i, k]) / h

                        # Christoffel products
                        product1 = sum(Gamma[l, i, m] * Gamma[m, j, k] for m in range(self.dim))
                        product2 = sum(Gamma[l, j, m] * Gamma[m, i, k] for m in range(self.dim))

                        R[l, i, j, k] = dGamma_i_ljk - dGamma_j_lik + product1 - product2

        return R

    def ricci_tensor(self, point: np.ndarray) -> np.ndarray:
        """
        Ricci curvature: Ric_{ij} = R^k_{ikj} (contraction of Riemann).

        Returns: shape (dim, dim) symmetric matrix
        """
        R = self.riemann_tensor(point)
        Ric = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                Ric[i, j] = sum(R[k, i, k, j] for k in range(self.dim))

        return Ric

    def scalar_curvature(self, point: np.ndarray) -> float:
        """
        Scalar curvature: S = g^{ij} Ric_{ij} (trace of Ricci).

        Single number characterizing curvature at point.
        """
        Ric = self.ricci_tensor(point)
        g_inv = self.metric.inverse(point)
        S = np.trace(g_inv @ Ric)
        return float(S)

    def sectional_curvature(self, point: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Sectional curvature of plane spanned by v, w:
        K(v,w) = <R(v,w)w, v> / (||v||² ||w||² - <v,w>²)

        Constant curvature:
            K > 0: sphere (positive curvature)
            K = 0: Euclidean space (flat)
            K < 0: hyperbolic space (negative curvature)
        """
        R = self.riemann_tensor(point)

        # R(v,w)w in coordinates
        Rvww = np.zeros(self.dim)
        for l in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        Rvww[l] += R[l, i, j, k] * v[i] * w[j] * w[k]

        numerator = self.metric.inner_product(point, Rvww, v)
        norm_v_sq = self.metric.inner_product(point, v, v)
        norm_w_sq = self.metric.inner_product(point, w, w)
        inner_vw = self.metric.inner_product(point, v, w)
        denominator = norm_v_sq * norm_w_sq - inner_vw**2

        if abs(denominator) < 1e-10:
            return 0.0  # Degenerate plane

        return numerator / denominator


class CurvatureAnalysis:
    """
    Advanced curvature analysis.

    Classification of manifolds by curvature properties.
    """

    @staticmethod
    def classify_curvature(scalar_curvature: float, tolerance: float = 1e-6) -> str:
        """Classify manifold by scalar curvature."""
        if scalar_curvature > tolerance:
            return "Positive curvature (sphere-like)"
        elif scalar_curvature < -tolerance:
            return "Negative curvature (hyperbolic-like)"
        else:
            return "Zero curvature (flat)"

    @staticmethod
    def einstein_manifold(ricci_tensor: np.ndarray, metric_tensor: np.ndarray,
                         tolerance: float = 1e-6) -> bool:
        """
        Check if manifold is Einstein: Ric = λ g for some constant λ.

        Examples: Spheres, hyperbolic spaces, many Calabi-Yau manifolds.
        """
        # Compute λ = Ric_{00} / g_{00}
        if abs(metric_tensor[0, 0]) < tolerance:
            return False

        lambda_val = ricci_tensor[0, 0] / metric_tensor[0, 0]

        # Check Ric_{ij} = λ g_{ij} for all i,j
        expected_ricci = lambda_val * metric_tensor
        diff = np.linalg.norm(ricci_tensor - expected_ricci)
        return diff < tolerance

    @staticmethod
    def ricci_flat(ricci_tensor: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if manifold is Ricci-flat: Ric = 0.

        Examples: Calabi-Yau manifolds (important in string theory).
        """
        return np.linalg.norm(ricci_tensor) < tolerance


class RicciFlow:
    """
    Ricci flow: geometric PDE evolving metric.

    ∂g/∂t = -2 Ric(g)

    Introduced by Hamilton, used by Perelman to prove Poincaré conjecture.
    """

    def __init__(self, initial_metric: RiemannianMetric):
        self.metric = initial_metric
        self.dim = initial_metric.dim

    def flow_step(self, point: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Single Ricci flow step: g(t + dt) = g(t) - 2 Ric(g(t)) dt

        Returns: updated metric tensor at point
        """
        curvature = RiemannCurvature(self.metric)
        Ric = curvature.ricci_tensor(point)
        g = self.metric(point)

        g_new = g - 2 * Ric * dt
        return g_new

    def normalize_flow_step(self, point: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Normalized Ricci flow: ∂g/∂t = -2(Ric - (S/n)g)
        where S = scalar curvature, n = dimension.

        Preserves volume.
        """
        curvature = RiemannCurvature(self.metric)
        Ric = curvature.ricci_tensor(point)
        S = curvature.scalar_curvature(point)
        g = self.metric(point)

        g_new = g - 2 * (Ric - (S / self.dim) * g) * dt
        return g_new

    @staticmethod
    def convergence_criteria(ricci_tensor: np.ndarray, tolerance: float = 1e-4) -> bool:
        """
        Check if Ricci flow has converged to constant curvature.

        Convergence => Einstein metric (Ric ∝ g).
        """
        norm_ricci = np.linalg.norm(ricci_tensor)
        return norm_ricci < tolerance


class ParallelTransport:
    """
    Parallel transport of vectors along curves.

    ∇_{γ'} V = 0 along curve γ.
    Preserves inner products (isometry of tangent spaces).
    """

    def __init__(self, metric: RiemannianMetric):
        self.metric = metric
        self.christoffel = Christoffel(metric)
        self.dim = metric.dim

    def transport(self, curve: np.ndarray, curve_velocity: np.ndarray,
                 initial_vector: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Parallel transport vector along curve.

        Equation: dV^k/dt + Γ^k_{ij} V^i (dγ^j/dt) = 0

        Args:
            curve: points along curve (n_steps, dim)
            curve_velocity: velocities along curve (n_steps, dim)
            initial_vector: vector at curve[0]

        Returns: transported vector at curve[-1]
        """
        V = initial_vector.copy()
        n_steps = len(curve)

        for i in range(n_steps - 1):
            point = curve[i]
            velocity = curve_velocity[i]
            Gamma = self.christoffel.symbols(point)

            # Update: V_{n+1} = V_n - Γ V velocity dt
            dV = np.zeros(self.dim)
            for k in range(self.dim):
                for i_idx in range(self.dim):
                    for j in range(self.dim):
                        dV[k] -= Gamma[k, i_idx, j] * V[i_idx] * velocity[j]

            V = V + dV * dt

        return V


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_sphere_curvature():
    """Example: Curvature of unit sphere S^2."""
    metric = RiemannianMetric.sphere(radius=1.0)
    curvature = RiemannCurvature(metric)

    # Point on sphere (θ=π/4, φ=π/4)
    point = np.array([np.pi/4, np.pi/4])

    # Scalar curvature (should be 2/R² = 2 for unit sphere)
    S = curvature.scalar_curvature(point)
    return S


def example_geodesic_on_sphere():
    """Example: Great circle (geodesic) on sphere."""
    metric = RiemannianMetric.sphere(radius=1.0)
    geodesic = Geodesic(metric)

    # Start at north pole, move eastward
    initial_point = np.array([0.1, 0.0])  # Near north pole
    initial_velocity = np.array([0.0, 1.0])  # Moving in φ direction

    points, velocities = geodesic.integrate(initial_point, initial_velocity, t_final=1.0)
    return points


def example_hyperbolic_curvature():
    """Example: Negative curvature of hyperbolic space."""
    metric = RiemannianMetric.hyperbolic(dim=2)
    curvature = RiemannCurvature(metric)

    # Point in Poincaré disk
    point = np.array([0.5, 0.0])

    # Scalar curvature (hyperbolic space has constant negative curvature)
    S = curvature.scalar_curvature(point)
    return S


if __name__ == "__main__":
    print("Riemannian Geometry Module")
    print("=" * 60)

    # Test 1: Metrics
    print("\n[Test 1] Riemannian metrics")
    euclidean = RiemannianMetric.euclidean(3)
    sphere = RiemannianMetric.sphere(radius=1.0)
    hyperbolic = RiemannianMetric.hyperbolic(dim=2)
    print(f"Euclidean metric (3D): {euclidean.dim}D")
    print(f"Sphere metric: {sphere.dim}D")
    print(f"Hyperbolic metric: {hyperbolic.dim}D")

    # Test 2: Christoffel symbols
    print("\n[Test 2] Christoffel symbols on sphere")
    point = np.array([np.pi/4, np.pi/4])
    christ = Christoffel(sphere)
    Gamma = christ.symbols(point)
    print(f"Γ at (π/4, π/4): shape {Gamma.shape}")
    print(f"Sample Γ^0_{01} = {Gamma[0, 0, 1]:.6f}")

    # Test 3: Geodesics
    print("\n[Test 3] Geodesic on sphere")
    points = example_geodesic_on_sphere()
    print(f"Geodesic integrated: {len(points)} points")
    print(f"Initial: θ={points[0][0]:.4f}, φ={points[0][1]:.4f}")
    print(f"Final: θ={points[-1][0]:.4f}, φ={points[-1][1]:.4f}")

    # Test 4: Curvature
    print("\n[Test 4] Curvature of sphere")
    S_sphere = example_sphere_curvature()
    print(f"Scalar curvature of unit sphere: {S_sphere:.4f} (expect ~2.0)")
    classification = CurvatureAnalysis.classify_curvature(S_sphere)
    print(f"Classification: {classification}")

    # Test 5: Hyperbolic curvature
    print("\n[Test 5] Hyperbolic space curvature")
    S_hyp = example_hyperbolic_curvature()
    print(f"Scalar curvature of hyperbolic space: {S_hyp:.4f} (expect negative)")
    classification = CurvatureAnalysis.classify_curvature(S_hyp)
    print(f"Classification: {classification}")

    # Test 6: Ricci flow
    print("\n[Test 6] Ricci flow initialization")
    flow = RicciFlow(sphere)
    point = np.array([np.pi/4, np.pi/4])
    g_new = flow.flow_step(point, dt=0.01)
    print(f"Metric after one Ricci flow step:\n{g_new}")

    print("\n" + "=" * 60)
    print("All Riemannian geometry tests complete!")
    print("Curvature, geodesics, and Ricci flow ready for physics.")
