"""
Comprehensive Tests for Multivariable Calculus Module.

Validates:
- curl() using central differences (Phase 1.3 fix)
- Known curl analytical solutions
- Vector calculus identities: curl(grad(f)) = 0, div(curl(F)) = 0
- Gradient, divergence, Laplacian operators
- Line integrals and conservative fields

Author: Claude Code
Date: December 2025 (Math Module Upgrade)
"""

import numpy as np
import pytest
from HoloLoom.warp.math.extensions.multivariable_calculus import (
    ScalarField, VectorField, LineIntegral, Point3D,
    SurfaceIntegral, IntegralTheorems, GradientCurlDiv
)


class TestCurlCentralDifferences:
    """Test curl() implementation with central differences (Phase 1.3)."""

    def test_rotation_field_curl(self):
        """F = (-y, x, 0) should have curl = (0, 0, 2)."""
        field = VectorField.rotation_z()
        point = np.array([1.0, 1.0, 0.0])

        curl = field.curl(point)

        # Analytical: curl(-y, x, 0) = (0 - 0, 0 - 0, 1 - (-1)) = (0, 0, 2)
        np.testing.assert_array_almost_equal(curl, [0, 0, 2], decimal=4)

    def test_radial_field_curl_is_zero(self):
        """F = (x, y, z) should have curl = 0 (conservative)."""
        field = VectorField.radial()
        point = np.array([1.0, 2.0, 3.0])

        curl = field.curl(point)

        # Radial field is conservative, curl = 0
        np.testing.assert_array_almost_equal(curl, [0, 0, 0], decimal=4)

    def test_constant_field_curl_is_zero(self):
        """Constant field should have curl = 0."""
        field = VectorField.constant(np.array([1.0, 2.0, 3.0]))
        point = np.array([5.0, 5.0, 5.0])

        curl = field.curl(point)

        np.testing.assert_array_almost_equal(curl, [0, 0, 0], decimal=6)

    def test_custom_curl_analytical(self):
        """Test custom field F = (y, -x, 0) has curl = (0, 0, -2)."""
        def F(p):
            return np.array([p[1], -p[0], 0.0])

        field = VectorField(F, dim=3)
        point = np.array([1.0, 1.0, 0.0])

        curl = field.curl(point)

        # Analytical: curl(y, -x, 0) = (0 - 0, 0 - 0, -1 - 1) = (0, 0, -2)
        np.testing.assert_array_almost_equal(curl, [0, 0, -2], decimal=4)

    def test_curl_xy_field(self):
        """Test F = (0, 0, xy) has curl = (x, -y, 0)."""
        def F(p):
            return np.array([0.0, 0.0, p[0] * p[1]])

        field = VectorField(F, dim=3)
        point = np.array([2.0, 3.0, 1.0])

        curl = field.curl(point)

        # Analytical: curl(0, 0, xy) = (∂(xy)/∂y, -∂(xy)/∂x, 0) = (x, -y, 0) = (2, -3, 0)
        np.testing.assert_array_almost_equal(curl, [2, -3, 0], decimal=3)


class TestVectorCalculusIdentities:
    """Test fundamental vector calculus identities."""

    def test_curl_of_gradient_is_zero(self):
        """curl(grad(f)) = 0 for any scalar field f."""
        # f(x,y,z) = x^2 + y^2 + z^2
        scalar_field = ScalarField.quadratic(np.array([1.0, 1.0, 1.0]))

        # Create vector field from gradient
        def grad_F(p):
            return scalar_field.gradient(p)

        grad_field = VectorField(grad_F, dim=3)

        # Test at multiple points
        test_points = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 3.0, 4.0])
        ]

        for point in test_points:
            curl = grad_field.curl(point)
            assert np.linalg.norm(curl) < 0.01, f"curl(grad(f)) != 0 at {point}"

    def test_divergence_of_curl_is_zero(self):
        """div(curl(F)) = 0 for any vector field F."""
        # Use rotation field (has non-zero curl)
        base_field = VectorField.rotation_z()

        # Create field of curl values
        def curl_F(p):
            return base_field.curl(p)

        curl_field = VectorField(curl_F, dim=3)

        test_points = [
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 3.0, 4.0])
        ]

        for point in test_points:
            div = curl_field.divergence(point)
            assert abs(div) < 0.1, f"div(curl(F)) != 0 at {point}"

    def test_laplacian_equals_div_grad(self):
        """Laplacian: div(grad(f)) = Δf."""
        # f(x,y,z) = x^2 + y^2 + z^2
        scalar_field = ScalarField.quadratic(np.array([1.0, 1.0, 1.0]))

        point = np.array([1.0, 2.0, 3.0])

        # Analytical Laplacian of x^2 + y^2 + z^2 = 2 + 2 + 2 = 6
        laplacian = scalar_field.laplacian(point)

        assert abs(laplacian - 6.0) < 0.1


class TestScalarFieldOperations:
    """Test scalar field operations."""

    def test_gradient_quadratic(self):
        """Gradient of f = x^2 + y^2 + z^2 is (2x, 2y, 2z)."""
        field = ScalarField.quadratic(np.array([1.0, 1.0, 1.0]))
        point = np.array([1.0, 2.0, 3.0])

        grad = field.gradient(point)

        np.testing.assert_array_almost_equal(grad, [2, 4, 6], decimal=4)

    def test_gradient_mixed_coefficients(self):
        """Gradient with mixed coefficients."""
        # f = 2x^2 + 3y^2 + 4z^2
        field = ScalarField.quadratic(np.array([2.0, 3.0, 4.0]))
        point = np.array([1.0, 1.0, 1.0])

        grad = field.gradient(point)

        # grad = (4x, 6y, 8z) = (4, 6, 8)
        np.testing.assert_array_almost_equal(grad, [4, 6, 8], decimal=4)

    def test_directional_derivative(self):
        """Directional derivative in gradient direction equals |grad|."""
        field = ScalarField.quadratic(np.array([1.0, 1.0, 1.0]))
        point = np.array([1.0, 1.0, 1.0])

        grad = field.gradient(point)
        direction = grad  # Direction of steepest ascent

        dir_deriv = field.directional_derivative(point, direction)

        # D_grad f = |grad|
        expected = np.linalg.norm(grad)
        assert abs(dir_deriv - expected) < 0.01

    def test_laplacian_quadratic(self):
        """Laplacian of f = ax^2 + by^2 + cz^2 is 2(a+b+c)."""
        # f = x^2 + 2y^2 + 3z^2
        field = ScalarField.quadratic(np.array([1.0, 2.0, 3.0]))
        point = np.array([1.0, 1.0, 1.0])

        laplacian = field.laplacian(point)

        # Δf = 2 + 4 + 6 = 12
        assert abs(laplacian - 12.0) < 0.1

    def test_distance_field(self):
        """Distance from origin field."""
        field = ScalarField.distance_from_origin()

        point = np.array([3.0, 4.0, 0.0])
        value = field(point)

        assert abs(value - 5.0) < 1e-10


class TestVectorFieldOperations:
    """Test vector field operations."""

    def test_divergence_radial(self):
        """Divergence of radial field F = (x, y, z) is 3."""
        field = VectorField.radial()
        point = np.array([1.0, 2.0, 3.0])

        div = field.divergence(point)

        # div(x, y, z) = 1 + 1 + 1 = 3
        assert abs(div - 3.0) < 0.01

    def test_divergence_scaled_radial(self):
        """Divergence of scaled radial field."""
        # F = (2x, 3y, 4z)
        def F(p):
            return np.array([2*p[0], 3*p[1], 4*p[2]])

        field = VectorField(F, dim=3)
        point = np.array([1.0, 1.0, 1.0])

        div = field.divergence(point)

        # div = 2 + 3 + 4 = 9
        assert abs(div - 9.0) < 0.1

    def test_divergence_constant_is_zero(self):
        """Constant field has zero divergence."""
        field = VectorField.constant(np.array([5.0, 10.0, 15.0]))
        point = np.array([1.0, 1.0, 1.0])

        div = field.divergence(point)

        assert abs(div) < 1e-6

    def test_is_conservative_radial(self):
        """Radial field is conservative (curl = 0)."""
        field = VectorField.radial()
        test_points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])

        assert field.is_conservative(test_points)

    def test_is_not_conservative_rotation(self):
        """Rotation field is NOT conservative."""
        field = VectorField.rotation_z()
        test_points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])

        assert not field.is_conservative(test_points)


class TestLineIntegrals:
    """Test line integral calculations."""

    def test_work_radial_field_straight_line(self):
        """Work done by radial field along straight line."""
        field = VectorField.radial()

        # Line from (0,0,0) to (1,1,1)
        def curve(t):
            return np.array([t, t, t])

        work = LineIntegral.vector_field(field, curve, 0, 1)

        # Analytical: ∫ (t, t, t) · (1, 1, 1) dt = ∫ 3t dt = 3/2 = 1.5
        assert abs(work - 1.5) < 0.1

    def test_circulation_rotation_field_circle(self):
        """Circulation of rotation field around unit circle."""
        field = VectorField.rotation_z()

        # Circle in xy-plane: (cos t, sin t, 0)
        def curve(t):
            return np.array([np.cos(t), np.sin(t), 0.0])

        circulation = LineIntegral.vector_field(field, curve, 0, 2*np.pi, n_steps=200)

        # F = (-y, x, 0), dr = (-sin t, cos t, 0) dt
        # F · dr = (-sin t)(-sin t) + (cos t)(cos t) = 1
        # ∮ = 2π
        assert abs(circulation - 2*np.pi) < 0.1

    def test_conservative_field_path_independent(self):
        """Conservative field: integral is path-independent."""
        # Gradient of f = x^2 + y^2 + z^2
        def grad_F(p):
            return 2 * p

        field = VectorField(grad_F, dim=3)

        # Path 1: straight line (0,0,0) to (1,1,1)
        def path1(t):
            return np.array([t, t, t])

        # Path 2: L-shaped path
        def path2(t):
            if t < 0.5:
                return np.array([2*t, 0, 0])  # x from 0 to 1
            else:
                s = 2 * (t - 0.5)
                return np.array([1, s, s])  # y,z from 0 to 1

        work1 = LineIntegral.vector_field(field, path1, 0, 1)
        work2 = LineIntegral.vector_field(field, path2, 0, 1)

        # Both should equal f(1,1,1) - f(0,0,0) = 3 - 0 = 3
        assert abs(work1 - 3.0) < 0.2
        assert abs(work2 - 3.0) < 0.2

    def test_scalar_line_integral_arc_length(self):
        """Line integral of f=1 gives arc length."""
        # f = 1 (constant)
        def one(p):
            return 1.0

        field = ScalarField(one, dim=3)

        # Quarter circle: (cos t, sin t, 0), t in [0, π/2]
        def curve(t):
            return np.array([np.cos(t), np.sin(t), 0.0])

        arc_length = LineIntegral.scalar_field(field, curve, 0, np.pi/2)

        # Should be π/2
        assert abs(arc_length - np.pi/2) < 0.1


class TestPoint3D:
    """Test Point3D dataclass."""

    def test_to_array(self):
        """Point3D should convert to numpy array."""
        p = Point3D(1.0, 2.0, 3.0)
        arr = p.to_array()

        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


class TestIntegralTheorems:
    """Test integral theorem documentation strings."""

    def test_fundamental_theorem(self):
        """Fundamental theorem string should exist."""
        text = IntegralTheorems.fundamental_theorem_line_integrals()
        assert "conservative" in text.lower()
        # Check for endpoint difference (may be f(b)-f(a) or f(r(b))-f(r(a)))
        assert "f(r(b))" in text or "f(b)" in text

    def test_greens_theorem(self):
        """Green's theorem string should exist."""
        text = IntegralTheorems.greens_theorem()
        assert "Green" in text
        assert "2D" in text

    def test_stokes_theorem(self):
        """Stokes' theorem string should exist."""
        text = IntegralTheorems.stokes_theorem()
        assert "Stokes" in text
        assert "curl" in text.lower()

    def test_divergence_theorem(self):
        """Divergence theorem string should exist."""
        text = IntegralTheorems.divergence_theorem()
        assert "Gauss" in text or "divergence" in text.lower()

    def test_generalized_stokes(self):
        """Generalized Stokes' theorem string should exist."""
        text = IntegralTheorems.generalized_stokes()
        assert "differential forms" in text.lower() or "manifold" in text.lower()


class TestGradientCurlDiv:
    """Test vector calculus identity documentation."""

    def test_identities_dict(self):
        """Identities dictionary should contain key identities."""
        identities = GradientCurlDiv.identities()

        assert "curl_of_gradient" in identities
        assert "div_of_curl" in identities
        assert "laplacian" in identities

    def test_helmholtz_decomposition(self):
        """Helmholtz decomposition string should exist."""
        text = GradientCurlDiv.helmholtz_decomposition()
        assert "Helmholtz" in text
        assert "grad" in text.lower()
        assert "curl" in text.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_curl_at_origin(self):
        """Curl should be defined at origin."""
        field = VectorField.rotation_z()
        point = np.array([0.0, 0.0, 0.0])

        curl = field.curl(point)

        # Should be (0, 0, 2) even at origin
        np.testing.assert_array_almost_equal(curl, [0, 0, 2], decimal=4)

    def test_gradient_at_origin(self):
        """Gradient should be zero at origin for f = r^2."""
        field = ScalarField.quadratic(np.array([1.0, 1.0, 1.0]))
        point = np.array([0.0, 0.0, 0.0])

        grad = field.gradient(point)

        np.testing.assert_array_almost_equal(grad, [0, 0, 0], decimal=4)

    def test_curl_2d_raises(self):
        """Curl should raise error for non-3D field."""
        def F(p):
            return np.array([p[0], p[1]])

        field = VectorField(F, dim=2)
        point = np.array([1.0, 1.0])

        with pytest.raises(ValueError) as excinfo:
            field.curl(point)

        assert "3D" in str(excinfo.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
