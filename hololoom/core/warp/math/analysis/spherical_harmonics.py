"""
Spherical Harmonics — AN5.7
============================

Complete orthonormal basis for functions on the sphere S^2.
Y_l^m(theta, phi) = normalization * P_l^m(cos(theta)) * exp(i*m*phi)

Used for angular decomposition, rotational analysis, and spherical
convolutions in 3D feature spaces.

Classes:
    AssociatedLegendre: P_l^m(x) via recurrence relations
    SphericalHarmonics: Y_l^m evaluation, decomposition, reconstruction

Mathematical Foundation:
    Y_l^m(theta, phi) = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos(theta)) * e^{im*phi}
    Orthogonality: integral Y_l^m * conj(Y_l'^m') dOmega = delta_{ll'} delta_{mm'}
    Completeness: f(theta, phi) = sum_{l,m} c_l^m Y_l^m(theta, phi)

Applications:
    - Spherical feature decomposition for embedding analysis
    - Rotational invariant representations
    - Angular power spectrum estimation
    - Gravitational/electromagnetic multipole expansions

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AssociatedLegendre:
    """
    Associated Legendre polynomials P_l^m(x) via stable recurrence.

    Uses the three-term recurrence:
        (l-m) P_l^m = x(2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m

    with sectoral seed:
        P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    """

    @staticmethod
    def evaluate(l: int, m: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate P_l^m(x) for |m| <= l.

        Uses Condon-Shortley phase convention ((-1)^m factor included).

        Args:
            l: Degree (non-negative integer)
            m: Order (integer, |m| <= l)
            x: Evaluation points in [-1, 1]

        Returns:
            P_l^m(x) array with same shape as x
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))

        if abs(m) > l:
            return np.zeros_like(x)

        # Handle negative m via relation:
        # P_l^{-m} = (-1)^m (l-m)!/(l+m)! P_l^m
        if m < 0:
            m_abs = -m
            result = AssociatedLegendre.evaluate(l, m_abs, x)
            sign = (-1) ** m_abs
            ratio = 1.0
            for k in range(l - m_abs + 1, l + m_abs + 1):
                ratio *= k
            return sign * result / ratio

        # Sectoral: P_m^m
        pmm = np.ones_like(x)
        if m > 0:
            somx2 = np.sqrt(1.0 - x * x)
            fact = 1.0
            for i in range(1, m + 1):
                pmm *= (-fact) * somx2
                fact += 2.0

        if l == m:
            return pmm

        # P_{m+1}^m = x(2m+1) P_m^m
        pmm1 = x * (2 * m + 1) * pmm
        if l == m + 1:
            return pmm1

        # General recurrence
        pll = np.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = (x * (2 * ll - 1) * pmm1 - (ll + m - 1) * pmm) / (ll - m)
            pmm = pmm1
            pmm1 = pll

        return pll

    @staticmethod
    def evaluate_all(l_max: int, x: np.ndarray) -> dict:
        """
        Evaluate all P_l^m(x) for l = 0..l_max, m = 0..l.

        More efficient than calling evaluate() individually due to
        shared recurrence computation.

        Args:
            l_max: Maximum degree
            x: Evaluation points

        Returns:
            Dict mapping (l, m) -> P_l^m(x) arrays
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = {}

        for m in range(l_max + 1):
            # Sectoral seed
            pmm = np.ones_like(x)
            if m > 0:
                somx2 = np.sqrt(1.0 - x * x)
                fact = 1.0
                for i in range(1, m + 1):
                    pmm *= (-fact) * somx2
                    fact += 2.0

            result[(m, m)] = pmm.copy()

            if m < l_max:
                pmm1 = x * (2 * m + 1) * pmm
                result[(m + 1, m)] = pmm1.copy()

                for ll in range(m + 2, l_max + 1):
                    pll = (x * (2 * ll - 1) * pmm1 - (ll + m - 1) * pmm) / (ll - m)
                    result[(ll, m)] = pll.copy()
                    pmm = pmm1
                    pmm1 = pll

        return result


class SphericalHarmonics:
    """
    Spherical harmonics Y_l^m(theta, phi) on the unit sphere.

    Convention: physics convention with Condon-Shortley phase.
    theta in [0, pi] (polar), phi in [0, 2*pi) (azimuthal).
    """

    @staticmethod
    def _normalization(l: int, m: int) -> float:
        """Normalization factor for Y_l^m."""
        # sqrt((2l+1)/(4pi) * (l-|m|)! / (l+|m|)!)
        am = abs(m)
        num = (2 * l + 1)
        # Compute factorial ratio (l-|m|)!/(l+|m|)! carefully
        ratio = 1.0
        for k in range(l - am + 1, l + am + 1):
            ratio *= k
        return np.sqrt(num / (4.0 * np.pi * ratio))

    @staticmethod
    def evaluate(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Evaluate Y_l^m(theta, phi).

        Args:
            l: Degree (non-negative integer)
            m: Order (integer, -l <= m <= l)
            theta: Polar angle(s) in [0, pi]
            phi: Azimuthal angle(s) in [0, 2*pi)

        Returns:
            Complex array Y_l^m(theta, phi)
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))

        norm = SphericalHarmonics._normalization(l, m)
        plm = AssociatedLegendre.evaluate(l, abs(m), np.cos(theta))

        if m >= 0:
            return norm * plm * np.exp(1j * m * phi)
        else:
            return ((-1) ** m) * norm * plm * np.exp(1j * m * phi)

    @staticmethod
    def evaluate_real(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Evaluate real spherical harmonics.

        Real version used in many applications:
            m > 0: sqrt(2) * Re(Y_l^m) = sqrt(2) * N * P_l^m * cos(m*phi)
            m = 0: Y_l^0 (already real)
            m < 0: sqrt(2) * Im(Y_l^{|m|}) = sqrt(2) * N * P_l^{|m|} * sin(|m|*phi)
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))

        norm = SphericalHarmonics._normalization(l, abs(m))
        plm = AssociatedLegendre.evaluate(l, abs(m), np.cos(theta))

        if m > 0:
            return np.sqrt(2.0) * norm * plm * np.cos(m * phi)
        elif m == 0:
            return norm * plm
        else:
            return np.sqrt(2.0) * norm * plm * np.sin(abs(m) * phi)

    @staticmethod
    def decompose(
        f_values: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        l_max: int,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Decompose a function on the sphere into spherical harmonic coefficients.

        f(theta, phi) = sum_{l=0}^{l_max} sum_{m=-l}^{l} c_l^m Y_l^m(theta, phi)

        Uses numerical quadrature: c_l^m = integral conj(Y_l^m) f dOmega.

        Args:
            f_values: Function values at sample points, shape (n_points,)
            theta: Polar angles of sample points, shape (n_points,)
            phi: Azimuthal angles of sample points, shape (n_points,)
            l_max: Maximum degree for expansion
            weights: Quadrature weights, shape (n_points,). If None,
                     uses sin(theta) * uniform solid angle.

        Returns:
            Coefficient array of shape ((l_max+1)^2,), indexed as
            c[l*(l+1) + m] for each (l, m) pair.
        """
        n_points = len(f_values)
        n_coeffs = (l_max + 1) ** 2

        if weights is None:
            # Approximate uniform weights on sphere
            weights = np.sin(theta) * (4.0 * np.pi / n_points)

        coeffs = np.zeros(n_coeffs, dtype=complex)

        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                idx = l * (l + 1) + m
                ylm = SphericalHarmonics.evaluate(l, m, theta, phi)
                coeffs[idx] = np.sum(weights * np.conj(ylm) * f_values)

        return coeffs

    @staticmethod
    def reconstruct(
        coefficients: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        l_max: int | None = None,
    ) -> np.ndarray:
        """
        Reconstruct a function from spherical harmonic coefficients.

        Args:
            coefficients: Coefficient array, shape ((l_max+1)^2,)
            theta: Polar angles for reconstruction
            phi: Azimuthal angles for reconstruction
            l_max: Maximum degree. Inferred from len(coefficients) if None.

        Returns:
            Reconstructed function values at (theta, phi)
        """
        if l_max is None:
            l_max = int(np.sqrt(len(coefficients))) - 1

        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        phi = np.atleast_1d(np.asarray(phi, dtype=float))

        result = np.zeros(len(theta), dtype=complex)

        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                idx = l * (l + 1) + m
                if idx < len(coefficients):
                    ylm = SphericalHarmonics.evaluate(l, m, theta, phi)
                    result += coefficients[idx] * ylm

        return result

    @staticmethod
    def power_spectrum(coefficients: np.ndarray, l_max: int | None = None) -> np.ndarray:
        """
        Angular power spectrum C_l = (1/(2l+1)) sum_m |c_l^m|^2.

        Args:
            coefficients: Spherical harmonic coefficients
            l_max: Maximum degree

        Returns:
            Power spectrum array, shape (l_max+1,)
        """
        if l_max is None:
            l_max = int(np.sqrt(len(coefficients))) - 1

        spectrum = np.zeros(l_max + 1)
        for l in range(l_max + 1):
            power = 0.0
            for m in range(-l, l + 1):
                idx = l * (l + 1) + m
                if idx < len(coefficients):
                    power += np.abs(coefficients[idx]) ** 2
            spectrum[l] = power / (2 * l + 1)

        return spectrum

    @staticmethod
    def rotation_matrix(alpha: float, beta: float, gamma: float, l: int) -> np.ndarray:
        """
        Wigner D-matrix for rotating spherical harmonics of degree l.

        Euler angles (alpha, beta, gamma) in ZYZ convention.
        D^l_{m'm}(alpha, beta, gamma) gives the rotation matrix
        such that rotated coefficients c'_{l,m'} = sum_m D^l_{m'm} c_{l,m}.

        Args:
            alpha, beta, gamma: Euler angles (radians)
            l: Degree of the representation

        Returns:
            Wigner D-matrix, shape (2l+1, 2l+1)
        """
        size = 2 * l + 1
        D = np.zeros((size, size), dtype=complex)

        for mp_idx, mp in enumerate(range(-l, l + 1)):
            for m_idx, m in enumerate(range(-l, l + 1)):
                # Small Wigner d-matrix via Jacobi polynomials
                d_val = SphericalHarmonics._wigner_d_element(l, mp, m, beta)
                D[mp_idx, m_idx] = np.exp(-1j * mp * alpha) * d_val * np.exp(-1j * m * gamma)

        return D

    @staticmethod
    def _wigner_d_element(l: int, mp: int, m: int, beta: float) -> float:
        """Small Wigner d-matrix element d^l_{m'm}(beta)."""
        # Use explicit sum formula
        cos_half = np.cos(beta / 2.0)
        sin_half = np.sin(beta / 2.0)

        s_min = max(0, m - mp)
        s_max = min(l + m, l - mp)

        result = 0.0
        for s in range(s_min, s_max + 1):
            num = _factorial(l + m) * _factorial(l - m) * _factorial(l + mp) * _factorial(l - mp)
            den = (_factorial(l + m - s) * _factorial(s) *
                   _factorial(mp - m + s) * _factorial(l - mp - s))

            sign = (-1) ** (mp - m + s)
            power_cos = 2 * l + m - mp - 2 * s
            power_sin = mp - m + 2 * s

            term = sign * np.sqrt(num) / den
            if power_cos >= 0 and power_sin >= 0:
                term *= cos_half ** power_cos * sin_half ** power_sin

            result += term

        return result


def _factorial(n: int) -> float:
    """Factorial with float output for large values."""
    if n < 0:
        return 0.0
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result
