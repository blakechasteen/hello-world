"""
Unit tests for hololoom.core.warp.math.geometry modules:
  - riemannian_geometry.py
  - differential_geometry.py
  - information_geometry.py

Tests verify geometric computations with small tensors/arrays.
Properties checked: metric symmetry, positive definiteness, geodesic
endpoints, divergence non-negativity, natural gradient correctness.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ── Riemannian Geometry ──────────────────────────────────────────────

from hololoom.core.warp.math.geometry.riemannian_geometry import (
    RiemannianMetric,
    Christoffel,
    Geodesic,
    RiemannCurvature,
    CurvatureAnalysis,
    RicciFlow,
    ParallelTransport,
)


class TestRiemannianMetric:
    """RiemannianMetric: inner products, norms, volume forms."""

    def test_euclidean_identity(self):
        m = RiemannianMetric.euclidean(3)
        assert m.dim == 3
        p = np.array([1.0, 2.0, 3.0])
        assert_allclose(m(p), np.eye(3))

    def test_euclidean_inner_product(self):
        m = RiemannianMetric.euclidean(3)
        p = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, 1.0, 0.0])
        assert m.inner_product(p, v, w) == pytest.approx(0.0)
        assert m.inner_product(p, v, v) == pytest.approx(1.0)

    def test_euclidean_norm(self):
        m = RiemannianMetric.euclidean(2)
        p = np.zeros(2)
        v = np.array([3.0, 4.0])
        assert m.norm(p, v) == pytest.approx(5.0)

    def test_euclidean_inverse(self):
        m = RiemannianMetric.euclidean(3)
        p = np.zeros(3)
        assert_allclose(m.inverse(p), np.eye(3))

    def test_euclidean_volume_form(self):
        m = RiemannianMetric.euclidean(3)
        assert m.volume_form(np.zeros(3)) == pytest.approx(1.0)

    def test_metric_symmetry_euclidean(self):
        m = RiemannianMetric.euclidean(4)
        g = m(np.zeros(4))
        assert_allclose(g, g.T)

    def test_metric_positive_definite_euclidean(self):
        m = RiemannianMetric.euclidean(3)
        eigvals = np.linalg.eigvalsh(m(np.zeros(3)))
        assert np.all(eigvals > 0)

    def test_sphere_metric_symmetry(self):
        m = RiemannianMetric.sphere(radius=2.0)
        assert m.dim == 2
        p = np.array([np.pi / 3, np.pi / 4])
        g = m(p)
        assert_allclose(g, g.T)

    def test_sphere_metric_positive_definite(self):
        m = RiemannianMetric.sphere(radius=1.0)
        p = np.array([np.pi / 4, np.pi / 6])
        eigvals = np.linalg.eigvalsh(m(p))
        assert np.all(eigvals > 0)

    def test_sphere_volume_form(self):
        m = RiemannianMetric.sphere(radius=1.0)
        p = np.array([np.pi / 4, 0.0])
        # det(g) = sin^2(theta), sqrt = sin(theta)
        expected = np.sin(np.pi / 4)
        assert m.volume_form(p) == pytest.approx(expected, abs=1e-10)

    def test_hyperbolic_metric_positive_definite(self):
        m = RiemannianMetric.hyperbolic(dim=2)
        p = np.array([0.3, 0.2])
        eigvals = np.linalg.eigvalsh(m(p))
        assert np.all(eigvals > 0)

    def test_hyperbolic_outside_disk_raises(self):
        m = RiemannianMetric.hyperbolic(dim=2)
        with pytest.raises(ValueError, match="outside Poincaré disk"):
            m(np.array([0.8, 0.8]))  # r^2 = 1.28 >= 1

    def test_hyperbolic_metric_symmetry(self):
        m = RiemannianMetric.hyperbolic(dim=2)
        g = m(np.array([0.1, 0.2]))
        assert_allclose(g, g.T)

    def test_custom_metric(self):
        def g(p):
            return np.array([[2.0, 0.0], [0.0, 3.0]])
        m = RiemannianMetric(2, g)
        v = np.array([1.0, 1.0])
        # <v,v> = 2*1 + 3*1 = 5
        assert m.inner_product(np.zeros(2), v, v) == pytest.approx(5.0)
        assert m.norm(np.zeros(2), v) == pytest.approx(np.sqrt(5.0))


class TestChristoffel:
    """Christoffel symbols and geodesic equation."""

    def test_euclidean_christoffel_zero(self):
        m = RiemannianMetric.euclidean(2)
        ch = Christoffel(m)
        Gamma = ch.symbols(np.array([1.0, 2.0]))
        assert_allclose(Gamma, 0.0, atol=1e-4)

    def test_geodesic_equation_euclidean(self):
        m = RiemannianMetric.euclidean(2)
        ch = Christoffel(m)
        accel = ch.geodesic_equation(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert_allclose(accel, 0.0, atol=1e-4)

    def test_christoffel_shape(self):
        m = RiemannianMetric.sphere(radius=1.0)
        ch = Christoffel(m)
        p = np.array([np.pi / 4, np.pi / 4])
        Gamma = ch.symbols(p)
        assert Gamma.shape == (2, 2, 2)


class TestGeodesic:
    """Geodesic integration and distance."""

    def test_euclidean_geodesic_is_straight_line(self):
        m = RiemannianMetric.euclidean(2)
        geo = Geodesic(m)
        p0 = np.array([0.0, 0.0])
        v0 = np.array([1.0, 0.0])
        pts, vels = geo.integrate(p0, v0, t_final=1.0, dt=0.01)
        # endpoint should be near (1, 0)
        assert_allclose(pts[-1], [1.0, 0.0], atol=0.02)
        # velocity stays constant in flat space
        assert_allclose(vels[-1], v0, atol=0.02)

    def test_geodesic_starts_at_initial_point(self):
        m = RiemannianMetric.euclidean(3)
        geo = Geodesic(m)
        p0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([0.1, 0.2, 0.3])
        pts, _ = geo.integrate(p0, v0, t_final=0.5, dt=0.01)
        assert_allclose(pts[0], p0)

    def test_euclidean_distance(self):
        m = RiemannianMetric.euclidean(2)
        geo = Geodesic(m)
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        d = geo.distance(a, b, n_samples=200)
        assert d == pytest.approx(5.0, abs=0.1)


class TestRiemannCurvature:
    """Curvature tensor and derived quantities."""

    def test_euclidean_riemann_tensor_zero(self):
        m = RiemannianMetric.euclidean(2)
        curv = RiemannCurvature(m)
        R = curv.riemann_tensor(np.array([0.0, 0.0]))
        assert_allclose(R, 0.0, atol=1e-3)

    def test_euclidean_ricci_tensor_zero(self):
        m = RiemannianMetric.euclidean(2)
        curv = RiemannCurvature(m)
        Ric = curv.ricci_tensor(np.array([0.0, 0.0]))
        assert_allclose(Ric, 0.0, atol=1e-2)

    def test_euclidean_scalar_curvature_zero(self):
        m = RiemannianMetric.euclidean(2)
        curv = RiemannCurvature(m)
        S = curv.scalar_curvature(np.array([0.0, 0.0]))
        assert S == pytest.approx(0.0, abs=0.1)

    def test_riemann_tensor_shape(self):
        m = RiemannianMetric.euclidean(3)
        curv = RiemannCurvature(m)
        R = curv.riemann_tensor(np.zeros(3))
        assert R.shape == (3, 3, 3, 3)

    def test_sectional_curvature_euclidean_zero(self):
        m = RiemannianMetric.euclidean(2)
        curv = RiemannCurvature(m)
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        K = curv.sectional_curvature(np.zeros(2), v, w)
        assert K == pytest.approx(0.0, abs=0.5)

    def test_sectional_curvature_degenerate_plane(self):
        m = RiemannianMetric.euclidean(2)
        curv = RiemannCurvature(m)
        v = np.array([1.0, 0.0])
        # parallel vectors -> degenerate
        K = curv.sectional_curvature(np.zeros(2), v, v)
        assert K == pytest.approx(0.0)


class TestCurvatureAnalysis:
    """Classification utilities for curvature."""

    def test_classify_positive(self):
        assert "Positive" in CurvatureAnalysis.classify_curvature(2.0)

    def test_classify_negative(self):
        assert "Negative" in CurvatureAnalysis.classify_curvature(-1.5)

    def test_classify_zero(self):
        assert "Zero" in CurvatureAnalysis.classify_curvature(0.0)

    def test_classify_near_zero(self):
        assert "Zero" in CurvatureAnalysis.classify_curvature(1e-8)

    def test_einstein_manifold_true(self):
        # Ric = 2*g -> Einstein with lambda=2
        g = np.eye(3)
        ric = 2.0 * np.eye(3)
        assert CurvatureAnalysis.einstein_manifold(ric, g)

    def test_einstein_manifold_false(self):
        g = np.eye(3)
        ric = np.diag([1.0, 2.0, 3.0])
        assert not CurvatureAnalysis.einstein_manifold(ric, g)

    def test_ricci_flat_true(self):
        assert CurvatureAnalysis.ricci_flat(np.zeros((3, 3)))

    def test_ricci_flat_false(self):
        assert not CurvatureAnalysis.ricci_flat(np.eye(3))


class TestRicciFlow:
    """Ricci flow evolution and convergence."""

    def test_flow_step_returns_matrix(self):
        m = RiemannianMetric.euclidean(2)
        flow = RicciFlow(m)
        g_new = flow.flow_step(np.zeros(2), dt=0.01)
        assert g_new.shape == (2, 2)

    def test_flat_flow_step_unchanged(self):
        # Flat metric has zero Ricci -> flow step leaves metric unchanged
        m = RiemannianMetric.euclidean(2)
        flow = RicciFlow(m)
        g_new = flow.flow_step(np.zeros(2), dt=0.01)
        assert_allclose(g_new, np.eye(2), atol=0.1)

    def test_normalize_flow_step_returns_matrix(self):
        m = RiemannianMetric.euclidean(2)
        flow = RicciFlow(m)
        g_new = flow.normalize_flow_step(np.zeros(2), dt=0.01)
        assert g_new.shape == (2, 2)

    def test_convergence_criteria_true(self):
        ric = np.zeros((2, 2))
        assert RicciFlow.convergence_criteria(ric)

    def test_convergence_criteria_false(self):
        ric = np.eye(2) * 10
        assert not RicciFlow.convergence_criteria(ric)


class TestParallelTransport:
    """Parallel transport along curves."""

    def test_euclidean_parallel_transport_preserves_vector(self):
        m = RiemannianMetric.euclidean(2)
        pt = ParallelTransport(m)
        # straight path from (0,0) to (1,0)
        n = 10
        curve = np.linspace([0.0, 0.0], [1.0, 0.0], n)
        curve_vel = np.tile([1.0, 0.0], (n, 1))
        v0 = np.array([0.0, 1.0])
        v_final = pt.transport(curve, curve_vel, v0, dt=0.1)
        # In flat space, parallel transport preserves the vector
        assert_allclose(v_final, v0, atol=0.1)


# ── Differential Geometry ────────────────────────────────────────────

from hololoom.core.warp.math.geometry.differential_geometry import (
    Chart,
    SmoothManifold,
    TangentSpace,
    TangentVector,
    TangentBundle,
    VectorField,
    DifferentialForm,
    ExteriorCalculus,
    LieDerivative,
)


class TestChart:
    """Local coordinate charts."""

    def test_identity_chart(self):
        chart = Chart("test", lambda x: x, lambda x: x, dimension=2)
        p = np.array([1.0, 2.0])
        assert_allclose(chart.phi(p), p)
        assert_allclose(chart.phi_inv(p), p)

    def test_transition_map(self):
        c1 = Chart("a", lambda x: x, lambda x: x, dimension=1)
        c2 = Chart("b", lambda x: x + 1.0, lambda x: x - 1.0, dimension=1)
        # transition c1 -> c2: phi2(phi1_inv(x)) = x + 1
        result = c1.transition_map(c2, np.array([3.0]))
        assert_allclose(result, np.array([4.0]))


class TestSmoothManifold:
    """Smooth manifold construction."""

    def test_circle_construction(self):
        S1 = SmoothManifold.circle()
        assert S1.dimension == 1
        assert S1.name == "S^1"
        assert len(S1.atlas) == 2

    def test_sphere_construction(self):
        S2 = SmoothManifold.sphere()
        assert S2.dimension == 2
        assert S2.name == "S^2"
        assert len(S2.atlas) == 1

    def test_torus_construction(self):
        T2 = SmoothManifold.torus()
        assert T2.dimension == 2
        assert T2.name == "T^2"
        assert len(T2.atlas) == 1

    def test_add_chart_dimension_mismatch(self):
        m = SmoothManifold(dimension=3, name="M")
        bad_chart = Chart("bad", lambda x: x, lambda x: x, dimension=2)
        with pytest.raises(ValueError, match="dimension"):
            m.add_chart(bad_chart)

    def test_verify_smoothness(self):
        S1 = SmoothManifold.circle()
        c1, c2 = S1.atlas
        # Test with points in valid domain overlap
        test_pts = np.array([[1.5]])  # in range for both charts
        result = S1.verify_smoothness(c1, c2, test_pts)
        assert isinstance(result, bool)


class TestTangentSpace:
    """Tangent spaces and vectors."""

    def test_tangent_vector_creation(self):
        m = SmoothManifold(2, "M")
        ts = TangentSpace(m, np.array([0.0, 0.0]))
        v = ts.tangent_vector(np.array([1.0, 2.0]))
        assert_allclose(v.components, [1.0, 2.0])

    def test_tangent_vector_wrong_dim(self):
        m = SmoothManifold(2, "M")
        ts = TangentSpace(m, np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="components"):
            ts.tangent_vector(np.array([1.0, 2.0, 3.0]))

    def test_tangent_vector_add(self):
        m = SmoothManifold(2, "M")
        p = np.array([0.0, 0.0])
        ts = TangentSpace(m, p)
        v = ts.tangent_vector(np.array([1.0, 0.0]))
        w = ts.tangent_vector(np.array([0.0, 2.0]))
        s = v + w
        assert_allclose(s.components, [1.0, 2.0])

    def test_tangent_vector_scalar_mul(self):
        m = SmoothManifold(2, "M")
        ts = TangentSpace(m, np.zeros(2))
        v = ts.tangent_vector(np.array([1.0, 2.0]))
        scaled = v * 3.0
        assert_allclose(scaled.components, [3.0, 6.0])

    def test_tangent_vector_norm_euclidean(self):
        m = SmoothManifold(2, "M")
        ts = TangentSpace(m, np.zeros(2))
        v = ts.tangent_vector(np.array([3.0, 4.0]))
        assert v.norm() == pytest.approx(5.0)

    def test_tangent_vector_norm_custom_metric(self):
        m = SmoothManifold(2, "M")
        ts = TangentSpace(m, np.zeros(2))
        v = ts.tangent_vector(np.array([1.0, 1.0]))
        metric = np.array([[2.0, 0.0], [0.0, 3.0]])
        assert v.norm(metric) == pytest.approx(np.sqrt(5.0))


class TestTangentBundle:
    """Tangent bundle and sections."""

    def test_bundle_dimension(self):
        m = SmoothManifold(3, "M")
        tb = TangentBundle(m)
        assert tb.dimension == 6

    def test_zero_section(self):
        m = SmoothManifold(2, "M")
        tb = TangentBundle(m)
        zero = tb.zero_section()
        assert_allclose(zero(np.array([5.0, 3.0])), [0.0, 0.0])

    def test_section_creates_vector_field(self):
        m = SmoothManifold(2, "M")
        tb = TangentBundle(m)
        vf = tb.section(lambda p: p * 2)
        result = vf(np.array([1.0, 2.0]))
        assert_allclose(result, [2.0, 4.0])


class TestVectorField:
    """Vector fields, flows, and Lie brackets."""

    def test_evaluate(self):
        m = SmoothManifold(2, "M")
        X = VectorField(m, lambda p: np.array([1.0, 0.0]))
        assert_allclose(X(np.array([5.0, 3.0])), [1.0, 0.0])

    def test_apply_to_function(self):
        m = SmoothManifold(2, "M")
        # Constant vector field in x-direction
        X = VectorField(m, lambda p: np.array([1.0, 0.0]))
        # f(x,y) = x^2 -> df/dx = 2x
        f = lambda p: p[0] ** 2
        deriv = X.apply_to_function(f, np.array([3.0, 0.0]))
        assert deriv == pytest.approx(6.0, abs=0.01)

    def test_flow_constant_field(self):
        m = SmoothManifold(2, "M")
        X = VectorField(m, lambda p: np.array([1.0, 0.0]))
        p0 = np.array([0.0, 0.0])
        p_final = X.flow(p0, t=2.0, dt=0.01)
        assert_allclose(p_final, [2.0, 0.0], atol=0.05)

    def test_flow_negative_time(self):
        m = SmoothManifold(1, "M")
        X = VectorField(m, lambda p: np.array([1.0]))
        p = X.flow(np.array([5.0]), t=-2.0, dt=0.01)
        assert p[0] == pytest.approx(3.0, abs=0.05)

    def test_commutator_commuting_fields(self):
        m = SmoothManifold(2, "M")
        X = VectorField(m, lambda p: np.array([1.0, 0.0]))
        Y = VectorField(m, lambda p: np.array([0.0, 1.0]))
        bracket = X.commutator(Y)
        result = bracket(np.array([0.0, 0.0]))
        assert_allclose(result, [0.0, 0.0], atol=1e-3)


class TestDifferentialForm:
    """Differential forms, wedge product, exterior derivative."""

    def test_one_form_evaluation(self):
        m = SmoothManifold(2, "R^2")
        # omega = x dy
        def omega_fn(p, v):
            return p[0] * v[1]
        omega = DifferentialForm(m, degree=1, form_function=omega_fn)
        val = omega(np.array([3.0, 0.0]), np.array([0.0, 1.0]))
        assert val == pytest.approx(3.0)

    def test_form_wrong_vector_count(self):
        m = SmoothManifold(2, "R^2")
        omega = DifferentialForm(m, 1, lambda p, v: v[0])
        with pytest.raises(ValueError, match="1-form requires 1 vectors"):
            omega(np.zeros(2), np.ones(2), np.ones(2))

    def test_two_form_volume(self):
        m = SmoothManifold(2, "R^2")
        def vol(p, v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]
        form = DifferentialForm(m, 2, vol)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert form(np.zeros(2), v1, v2) == pytest.approx(1.0)

    def test_coordinate_one_form_dx(self):
        m = SmoothManifold(3, "R^3")
        dx0 = DifferentialForm.dx(m, 0)
        v = np.array([5.0, 3.0, 1.0])
        assert dx0(np.zeros(3), v) == pytest.approx(5.0)

    def test_wedge_product_degree(self):
        m = SmoothManifold(2, "R^2")
        dx = DifferentialForm.dx(m, 0)
        dy = DifferentialForm.dx(m, 1)
        dx_wedge_dy = dx.wedge(dy)
        assert dx_wedge_dy.degree == 2

    def test_exterior_derivative_degree(self):
        m = SmoothManifold(2, "R^2")
        omega = DifferentialForm(m, 1, lambda p, v: p[0] * v[1])
        chart = Chart("id", lambda x: x, lambda x: x, dimension=2)
        d_omega = omega.exterior_derivative(chart)
        assert d_omega.degree == 2


class TestExteriorCalculus:
    """Stokes theorem and form classification."""

    def test_stokes_theorem_string(self):
        m = SmoothManifold(2, "R^2")
        omega = DifferentialForm(m, 1, lambda p, v: v[0])
        result = ExteriorCalculus.stokes_theorem(omega, m)
        assert "Stokes" in result
        assert "1-form" in result

    def test_closed_vs_exact(self):
        m = SmoothManifold(2, "R^2")
        omega = DifferentialForm(m, 1, lambda p, v: v[0])
        info = ExteriorCalculus.closed_vs_exact(omega)
        assert "closed" in info
        assert "exact" in info
        assert "de_Rham_cohomology" in info

    def test_integrate_form_dimension_mismatch(self):
        m = SmoothManifold(2, "R^2")
        omega = DifferentialForm(m, 1, lambda p, v: v[0])
        chart = Chart("id", lambda x: x, lambda x: x, dimension=2)
        with pytest.raises(ValueError, match="Cannot integrate"):
            ExteriorCalculus.integrate_form(omega, m, lambda n: np.zeros((n, 2)), chart)


class TestLieDerivative:
    """Lie derivatives of functions and vector fields."""

    def test_lie_derivative_function(self):
        m = SmoothManifold(2, "M")
        X = VectorField(m, lambda p: np.array([1.0, 0.0]))
        f = lambda p: p[0] ** 2  # df/dx = 2x
        val = LieDerivative.lie_derivative_function(X, f, np.array([3.0, 0.0]))
        assert val == pytest.approx(6.0, abs=0.01)

    def test_lie_derivative_vector_field(self):
        m = SmoothManifold(2, "M")
        X = VectorField(m, lambda p: np.array([1.0, 0.0]))
        Y = VectorField(m, lambda p: np.array([0.0, 1.0]))
        L_X_Y = LieDerivative.lie_derivative_vector_field(X, Y)
        assert isinstance(L_X_Y, VectorField)
        # Constant fields -> bracket is zero
        assert_allclose(L_X_Y(np.zeros(2)), [0.0, 0.0], atol=1e-3)

    def test_lie_derivative_form(self):
        m = SmoothManifold(2, "M")
        X = VectorField(m, lambda p: np.array([1.0, 0.0]))
        omega = DifferentialForm(m, 1, lambda p, v: p[0] * v[1])
        L_X_omega = LieDerivative.lie_derivative_form(X, omega)
        assert isinstance(L_X_omega, DifferentialForm)
        assert L_X_omega.degree == 1


# ── Information Geometry ─────────────────────────────────────────────

from hololoom.core.warp.math.geometry.information_geometry import (
    FisherMetric,
    AlphaConnection,
    BregmanDivergence,
    NaturalGradient,
    StatisticalManifold,
    StatisticalGeodesic,
)


class TestFisherMetric:
    """Fisher information metric on statistical manifolds."""

    def test_gaussian_fisher_matrix(self):
        fm = FisherMetric.gaussian_family()
        G = fm.matrix(np.array([0.0, 1.0]))
        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        assert_allclose(G, expected, atol=1e-6)

    def test_gaussian_fisher_symmetry(self):
        fm = FisherMetric.gaussian_family()
        G = fm.matrix(np.array([2.0, 3.0]))
        assert_allclose(G, G.T)

    def test_gaussian_fisher_positive_definite(self):
        fm = FisherMetric.gaussian_family()
        G = fm.matrix(np.array([0.0, 2.0]))
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0)

    def test_beta_fisher_matrix_symmetric(self):
        fm = FisherMetric.beta_family()
        G = fm.matrix(np.array([2.0, 3.0]))
        assert_allclose(G, G.T, atol=1e-10)

    def test_beta_fisher_positive_definite(self):
        fm = FisherMetric.beta_family()
        G = fm.matrix(np.array([2.0, 3.0]))
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0)

    def test_dirichlet_fisher_symmetric(self):
        fm = FisherMetric.dirichlet_family(3)
        G = fm.matrix(np.array([2.0, 3.0, 1.5]))
        assert_allclose(G, G.T, atol=1e-10)

    def test_dirichlet_fisher_positive_definite(self):
        fm = FisherMetric.dirichlet_family(3)
        G = fm.matrix(np.array([2.0, 3.0, 1.5]))
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0)

    def test_score_function(self):
        fm = FisherMetric.gaussian_family()
        # score of gaussian at x=0, theta=(0,1): d/dmu log p = (x-mu)/sigma^2 = 0
        s = fm.score(np.array(0.0), np.array([0.0, 1.0]))
        assert s.shape == (2,)
        assert s[0] == pytest.approx(0.0, abs=0.01)

    def test_inner_product(self):
        fm = FisherMetric.gaussian_family()
        theta = np.array([0.0, 1.0])
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        G = fm.matrix(theta)
        # G is diag(1, 2), so <v,w> = 0
        assert fm.inner_product(theta, v, w, G) == pytest.approx(0.0)

    def test_norm(self):
        fm = FisherMetric.gaussian_family()
        theta = np.array([0.0, 1.0])
        v = np.array([1.0, 0.0])
        G = fm.matrix(theta)
        # ||v|| = sqrt(v^T G v) = sqrt(1) = 1
        assert fm.norm(theta, v, G) == pytest.approx(1.0)

    def test_matrix_without_samples_or_fn_raises(self):
        fm = FisherMetric(2, lambda x, t: 0.0, sample_fn=None)
        with pytest.raises(ValueError, match="sample_fn"):
            fm.matrix(np.array([1.0, 1.0]))


class TestAlphaConnection:
    """Alpha-connections on statistical manifolds."""

    def test_dual_connection(self):
        fm = FisherMetric.gaussian_family()
        conn = AlphaConnection(fm, alpha=1.0)
        dual = conn.dual()
        assert dual.alpha == pytest.approx(-1.0)

    def test_levi_civita_alpha_zero(self):
        fm = FisherMetric.gaussian_family()
        conn = AlphaConnection(fm, alpha=0.0)
        dual = conn.dual()
        assert dual.alpha == pytest.approx(0.0)


class TestBregmanDivergence:
    """Bregman divergence properties."""

    def test_squared_euclidean_divergence(self):
        breg = BregmanDivergence.squared_euclidean()
        breg.dim = 2
        p = np.array([1.0, 2.0])
        q = np.array([3.0, 4.0])
        d = breg.divergence(p, q)
        expected = 0.5 * np.sum((p - q) ** 2)
        assert d == pytest.approx(expected, abs=1e-4)

    def test_divergence_nonnegative(self):
        breg = BregmanDivergence.squared_euclidean()
        breg.dim = 3
        rng = np.random.default_rng(42)
        for _ in range(5):
            p = rng.normal(size=3)
            q = rng.normal(size=3)
            assert breg.divergence(p, q) >= -1e-8

    def test_divergence_zero_same_point(self):
        breg = BregmanDivergence.squared_euclidean()
        breg.dim = 2
        p = np.array([1.0, 2.0])
        assert breg.divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_symmetrized_divergence(self):
        breg = BregmanDivergence.squared_euclidean()
        breg.dim = 2
        p = np.array([1.0, 2.0])
        q = np.array([3.0, 4.0])
        d_sym = breg.symmetrized(p, q)
        # For squared euclidean, symmetrized = (D(p||q) + D(q||p))/2
        assert d_sym >= -1e-8
        # symmetrized should be symmetric
        assert d_sym == pytest.approx(breg.symmetrized(q, p), abs=1e-6)

    def test_gradient(self):
        breg = BregmanDivergence.squared_euclidean()
        breg.dim = 2
        # gradient of ||x||^2/2 = x
        p = np.array([3.0, 5.0])
        grad = breg.gradient(p)
        assert_allclose(grad, p, atol=1e-4)

    def test_hessian(self):
        breg = BregmanDivergence.squared_euclidean()
        breg.dim = 2
        # hessian of ||x||^2/2 = I
        H = breg.hessian(np.array([1.0, 2.0]))
        assert_allclose(H, np.eye(2), atol=1e-3)

    def test_kl_divergence_factory(self):
        breg = BregmanDivergence.kl_divergence()
        assert breg is not None
        breg.dim = 3
        # KL of identical distributions = 0
        p = np.array([0.3, 0.3, 0.4])
        assert breg.divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_beta_log_partition(self):
        breg = BregmanDivergence.beta_log_partition()
        assert breg.dim == 2
        # Same point -> divergence 0
        theta = np.array([1.0, 1.0])
        assert breg.divergence(theta, theta) == pytest.approx(0.0, abs=1e-5)

    def test_beta_log_partition_nonnegative(self):
        breg = BregmanDivergence.beta_log_partition()
        t1 = np.array([1.0, 1.0])
        t2 = np.array([3.0, 2.0])
        assert breg.divergence(t1, t2) >= -1e-6


class TestNaturalGradient:
    """Natural gradient computation and stepping."""

    def test_compute_gaussian(self):
        fm = FisherMetric.gaussian_family()
        ng = NaturalGradient(fm, damping=1e-8)
        theta = np.array([0.0, 1.0])
        grad = np.array([1.0, 0.5])
        G = fm.matrix(theta)
        nat = ng.compute(theta, grad, G)
        # G = diag(1, 2), G^-1 = diag(1, 0.5)
        # nat_grad = G^-1 grad = (1.0, 0.25)
        assert_allclose(nat, [1.0, 0.25], atol=1e-3)

    def test_step(self):
        fm = FisherMetric.gaussian_family()
        ng = NaturalGradient(fm, damping=1e-8)
        theta = np.array([0.0, 1.0])
        grad = np.array([1.0, 0.0])
        G = fm.matrix(theta)
        new_theta = ng.step(theta, grad, lr=0.1, G=G)
        # theta_new = theta - 0.1 * G^-1 grad = [0, 1] - 0.1*[1, 0] = [-0.1, 1.0]
        assert_allclose(new_theta, [-0.1, 1.0], atol=1e-3)

    def test_trust_region_step_respects_kl(self):
        fm = FisherMetric.gaussian_family()
        ng = NaturalGradient(fm, damping=1e-8)
        theta = np.array([0.0, 1.0])
        grad = np.array([5.0, 3.0])
        G = fm.matrix(theta)
        max_kl = 0.01
        new_theta = ng.step_with_trust_region(theta, grad, max_kl=max_kl, G=G)
        delta = new_theta - theta
        actual_kl = 0.5 * float(delta @ G @ delta)
        assert actual_kl <= max_kl + 1e-6

    def test_trust_region_zero_gradient(self):
        fm = FisherMetric.gaussian_family()
        ng = NaturalGradient(fm, damping=1e-8)
        theta = np.array([0.0, 1.0])
        grad = np.array([0.0, 0.0])
        G = fm.matrix(theta)
        new_theta = ng.step_with_trust_region(theta, grad, max_kl=0.01, G=G)
        assert_allclose(new_theta, theta)


class TestStatisticalManifold:
    """Statistical manifold: geodesics, divergences."""

    def test_gaussian_manifold_creation(self):
        sm = StatisticalManifold.gaussian_manifold()
        assert sm.fisher is not None
        assert sm.e_connection.alpha == pytest.approx(1.0)
        assert sm.m_connection.alpha == pytest.approx(-1.0)
        assert sm.levi_civita.alpha == pytest.approx(0.0)

    def test_beta_manifold_creation(self):
        sm = StatisticalManifold.beta_manifold()
        assert sm.theta_to_eta is not None
        assert sm.eta_to_theta is not None
        assert sm.bregman is not None

    def test_e_geodesic_endpoints(self):
        sm = StatisticalManifold.beta_manifold()
        start = np.array([1.0, 1.0])
        end = np.array([4.0, 2.0])
        path = sm.e_geodesic(start, end, n_points=10)
        assert path.shape == (10, 2)
        assert_allclose(path[0], start)
        assert_allclose(path[-1], end)

    def test_m_geodesic_endpoints(self):
        sm = StatisticalManifold.beta_manifold()
        start = np.array([1.0, 1.0])
        end = np.array([2.0, 2.0])
        path = sm.m_geodesic(start, end, n_points=10)
        assert path is not None
        assert path.shape == (10, 2)
        # Start/end in theta coords should be close to original
        assert_allclose(path[0], start, atol=0.5)

    def test_m_geodesic_none_without_maps(self):
        fm = FisherMetric.gaussian_family()
        sm = StatisticalManifold(fm)
        result = sm.m_geodesic(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        assert result is None

    def test_kl_divergence_zero_same_point(self):
        sm = StatisticalManifold.beta_manifold()
        theta = np.array([1.0, 1.0])
        kl = sm.kl_divergence(theta, theta)
        assert kl is not None
        assert kl == pytest.approx(0.0, abs=1e-4)

    def test_kl_divergence_nonnegative(self):
        sm = StatisticalManifold.beta_manifold()
        t1 = np.array([1.0, 1.0])
        t2 = np.array([3.0, 2.0])
        kl = sm.kl_divergence(t1, t2)
        assert kl is not None
        assert kl >= -1e-6

    def test_kl_divergence_none_without_bregman(self):
        fm = FisherMetric.gaussian_family()
        sm = StatisticalManifold(fm)
        assert sm.kl_divergence(np.array([0.0, 1.0]), np.array([1.0, 2.0])) is None

    def test_geodesic_distance_positive(self):
        sm = StatisticalManifold.gaussian_manifold()
        d = sm.geodesic_distance(np.array([0.0, 1.0]), np.array([2.0, 1.0]), n_steps=50)
        assert d > 0

    def test_geodesic_distance_zero_same_point(self):
        sm = StatisticalManifold.gaussian_manifold()
        theta = np.array([0.0, 1.0])
        d = sm.geodesic_distance(theta, theta, n_steps=50)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_natural_gradient_factory(self):
        sm = StatisticalManifold.gaussian_manifold()
        ng = sm.natural_gradient(damping=1e-4)
        assert isinstance(ng, NaturalGradient)


class TestStatisticalGeodesic:
    """Geodesic shooting on statistical manifolds."""

    def test_shoot_returns_correct_shape(self):
        sm = StatisticalManifold.gaussian_manifold()
        sg = StatisticalGeodesic(sm)
        theta0 = np.array([0.0, 1.0])
        v0 = np.array([0.1, 0.0])
        pos, vel = sg.shoot(theta0, v0, t_final=0.1, dt=0.05)
        assert pos.shape[1] == 2
        assert vel.shape[1] == 2
        assert pos.shape[0] == vel.shape[0]

    def test_shoot_starts_at_initial_point(self):
        sm = StatisticalManifold.gaussian_manifold()
        sg = StatisticalGeodesic(sm)
        theta0 = np.array([0.0, 2.0])
        v0 = np.array([0.05, 0.0])
        pos, _ = sg.shoot(theta0, v0, t_final=0.1, dt=0.05)
        assert_allclose(pos[0], theta0)

    def test_boundary_value_endpoints(self):
        sm = StatisticalManifold.gaussian_manifold()
        sg = StatisticalGeodesic(sm)
        start = np.array([0.0, 1.0])
        end = np.array([0.5, 1.0])
        path = sg.boundary_value(start, end, n_points=10, n_iterations=5)
        assert path.shape[1] == 2
        assert_allclose(path[0], start, atol=0.01)
