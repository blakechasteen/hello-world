"""
Harmonic Analysis on Groups & Riemann Surfaces
================================================

Scope slots: AN2.7 (Riemann surfaces), AN5.8 (harmonic analysis on groups).

Pure numpy implementations:
- Riemann surfaces: algebraic curves, genus, Euler characteristic
- Group harmonic analysis: Fourier transform on finite groups, Peter-Weyl

The Peter-Weyl theorem says: L²(G) decomposes into matrix coefficients
of irreducible representations. This is the non-abelian generalization
of Fourier series.

Author: Claude Code
Date: March 2026
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# RIEMANN SURFACES (AN2.7)
# ============================================================================

@dataclass
class RiemannSurfaceData:
    """Data describing a Riemann surface.

    Attributes:
        genus: Topological genus g.
        euler_char: Euler characteristic χ = 2 - 2g.
        degree: Degree of the defining curve.
        singular_points: List of singular points (if any).
    """
    genus: int
    euler_char: int
    degree: int
    singular_points: list[np.ndarray]


class RiemannSurface:
    """Compact Riemann surface defined by an algebraic curve.

    A Riemann surface is a 1-dimensional complex manifold. Every compact
    Riemann surface arises as the completion of an algebraic curve
    f(z, w) = 0, where z and w are complex variables.

    The genus g determines the topology:
    - g = 0: sphere (Riemann sphere)
    - g = 1: torus (elliptic curve)
    - g ≥ 2: higher genus (hyperbolic)

    For a smooth projective plane curve of degree d:
        g = (d-1)(d-2)/2  (genus-degree formula)

    Example:
        >>> rs = RiemannSurface.from_algebraic_curve(degree=3)
        >>> rs.genus()  # Elliptic curve: g = 1
        1
        >>> rs.euler_characteristic()  # χ = 2 - 2g = 0
        0
    """

    def __init__(self, coefficients: dict[tuple[int, int], complex] | None = None, degree: int = 2):
        """Initialize from polynomial f(z, w) = Σ a_{ij} z^i w^j.

        Args:
            coefficients: Map (i, j) -> a_{ij}. If None, uses default curve.
            degree: Degree of the curve (used if coefficients is None).
        """
        if coefficients is None:
            # Default: w^d - z(z-1)(z-2)...(z-d+1)
            self._degree = degree
            self._coefficients = self._default_curve(degree)
        else:
            self._coefficients = coefficients
            self._degree = max(i + j for i, j in coefficients.keys())

    @staticmethod
    def from_algebraic_curve(degree: int = 2) -> 'RiemannSurface':
        """Create Riemann surface from a smooth curve of given degree.

        Args:
            degree: Degree d. Creates a generic smooth curve.

        Returns:
            RiemannSurface.
        """
        return RiemannSurface(degree=degree)

    def genus(self) -> int:
        """Compute genus via the genus-degree formula.

        For a smooth projective plane curve of degree d:
            g = (d-1)(d-2)/2

        For curves with singularities, the genus is reduced by the
        delta invariant of each singularity.

        Returns:
            Genus of the surface.
        """
        d = self._degree
        g = (d - 1) * (d - 2) // 2

        # Correct for singularities
        singularities = self._find_singularities()
        for _ in singularities:
            g -= 1  # Simple node reduces genus by 1

        return max(0, g)

    def euler_characteristic(self) -> int:
        """Compute Euler characteristic χ = 2 - 2g.

        Returns:
            Euler characteristic.
        """
        return 2 - 2 * self.genus()

    def ramification_points(self, n_samples: int = 100) -> list[complex]:
        """Find ramification points of the projection π: S → P¹.

        Ramification points are where dw/dz is undefined (the curve
        has a vertical tangent). These are the branch points.

        Args:
            n_samples: Number of sample points for numerical search.

        Returns:
            List of approximate ramification z-values.
        """
        branch_points = []

        for z_re in np.linspace(-3, 3, n_samples):
            for z_im in np.linspace(-3, 3, n_samples // 4):
                z = complex(z_re, z_im)
                # Check if the discriminant vanishes
                disc = self._discriminant_at(z)
                if abs(disc) < 0.1:
                    branch_points.append(z)

        # Cluster nearby points
        return self._cluster_points(branch_points)

    def riemann_hurwitz(self, degree_of_map: int) -> int:
        """Apply Riemann-Hurwitz formula.

        For a branched covering f: S → T of degree n:
            2g(S) - 2 = n(2g(T) - 2) + Σ (e_p - 1)

        where e_p is the ramification index at point p.

        Args:
            degree_of_map: Degree n of the covering map.

        Returns:
            Genus of the covering surface.
        """
        g_target = 0  # Assume target is P¹ (genus 0)
        n = degree_of_map

        # Count ramification: for generic curve of degree d projecting to P¹
        # Total ramification = 2(d + g - 1) by Riemann-Hurwitz
        g = self.genus()
        total_ram = 2 * (self._degree + g - 1)

        # Genus of covering
        g_source = 1 + n * (g_target - 1) + total_ram // 2
        return max(0, g_source)

    def _find_singularities(self, grid_size: int = 50) -> list[np.ndarray]:
        """Find singular points where f = ∂f/∂z = ∂f/∂w = 0."""
        singularities = []
        # Simplified: check on a grid
        for z_re in np.linspace(-3, 3, grid_size):
            for w_re in np.linspace(-3, 3, grid_size):
                z, w = complex(z_re), complex(w_re)
                f = self._eval(z, w)
                fz = self._eval_dz(z, w)
                fw = self._eval_dw(z, w)
                if abs(f) < 0.01 and abs(fz) < 0.01 and abs(fw) < 0.01:
                    singularities.append(np.array([z_re, w_re]))
        return singularities

    def _eval(self, z: complex, w: complex) -> complex:
        """Evaluate f(z, w)."""
        result = 0.0
        for (i, j), coeff in self._coefficients.items():
            result += coeff * z**i * w**j
        return result

    def _eval_dz(self, z: complex, w: complex, dz: float = 1e-6) -> complex:
        return (self._eval(z + dz, w) - self._eval(z - dz, w)) / (2 * dz)

    def _eval_dw(self, z: complex, w: complex, dw: float = 1e-6) -> complex:
        return (self._eval(z, w + dw) - self._eval(z, w - dw)) / (2 * dw)

    def _discriminant_at(self, z: complex) -> float:
        """Approximate discriminant of f(z, w) as polynomial in w."""
        # Evaluate f at several w values
        ws = np.linspace(-3, 3, 20)
        vals = [abs(self._eval(z, complex(w))) for w in ws]
        # Simple check: are there near-zeros?
        min_val = min(vals)
        return min_val

    @staticmethod
    def _cluster_points(points: list[complex], tol: float = 0.5) -> list[complex]:
        """Cluster nearby points."""
        if not points:
            return []
        clusters = [points[0]]
        for p in points[1:]:
            if all(abs(p - c) > tol for c in clusters):
                clusters.append(p)
        return clusters

    @staticmethod
    def _default_curve(d: int) -> dict[tuple[int, int], complex]:
        """Default curve: w^d = z(z-1)(z-2)...(z-d+1)."""
        coeffs = {(0, d): 1.0}
        # Expand product z(z-1)...(z-d+1) using recursion
        poly = np.array([1.0])  # Start with 1
        for k in range(d):
            # Multiply by (z - k)
            poly = np.convolve(poly, [1, -k])

        for i, c in enumerate(reversed(poly)):
            if abs(c) > 1e-15:
                coeffs[(i, 0)] = -c

        return coeffs


# ============================================================================
# HARMONIC ANALYSIS ON GROUPS (AN5.8)
# ============================================================================

class GroupHarmonicAnalysis:
    """Harmonic analysis on finite groups.

    The Peter-Weyl theorem decomposes L²(G) into irreducible representations:
        L²(G) = ⊕_π dim(π) · V_π

    The Fourier transform on a finite group:
        f̂(π) = Σ_{g ∈ G} f(g) π(g)    (matrix-valued)

    Inverse transform:
        f(g) = (1/|G|) Σ_π dim(π) Tr[f̂(π) π(g)⁻¹]

    For abelian groups, this reduces to the discrete Fourier transform.

    Example:
        >>> gha = GroupHarmonicAnalysis()
        >>> # Cyclic group Z/4Z with character representations
        >>> elements = [0, 1, 2, 3]
        >>> # Representations: π_k(g) = exp(2πi k g / 4)
        >>> reps = {k: lambda g, k=k: np.exp(2j * np.pi * k * g / 4)
        ...         for k in range(4)}
        >>> f = lambda g: g**2  # Function on the group
        >>> ft = gha.fourier_transform(f, elements, reps)
    """

    def fourier_transform(
        self,
        f: Callable,
        group_elements: list,
        representations: dict[str, Callable],
    ) -> dict[str, np.ndarray]:
        """Compute Fourier transform on a finite group.

        f̂(π) = Σ_{g ∈ G} f(g) · π(g)

        For 1D representations (characters), f̂(π) is a scalar.
        For higher-dim representations, f̂(π) is a matrix.

        Args:
            f: Function on the group (g → scalar).
            group_elements: List of group elements.
            representations: Dict mapping rep name to rep function.
                Each rep function: g → scalar or matrix.

        Returns:
            Dict mapping rep name to Fourier coefficient (scalar or matrix).
        """
        result = {}

        for name, pi in representations.items():
            coeff = None
            for g in group_elements:
                val = f(g)
                rep_val = pi(g)

                if np.isscalar(rep_val):
                    rep_val = np.array(rep_val)

                term = val * rep_val

                if coeff is None:
                    coeff = np.zeros_like(term, dtype=complex)
                coeff += term

            result[name] = coeff

        return result

    def inverse_fourier_transform(
        self,
        fourier_coeffs: dict[str, np.ndarray],
        group_elements: list,
        representations: dict[str, Callable],
        dimensions: dict[str, int] | None = None,
    ) -> dict:
        """Inverse Fourier transform: recover f from f̂.

        f(g) = (1/|G|) Σ_π dim(π) Tr[f̂(π) · π(g)^{-1}]

        Args:
            fourier_coeffs: Fourier coefficients f̂(π).
            group_elements: Group elements.
            representations: Representation functions.
            dimensions: Rep dimensions (auto-detected if None).

        Returns:
            Dict mapping group element to function value.
        """
        n = len(group_elements)
        result = {}

        for g in group_elements:
            val = 0.0
            for name, coeff in fourier_coeffs.items():
                rep_val = representations[name](g)

                if np.isscalar(rep_val):
                    # 1D rep
                    dim = 1
                    contribution = dim * np.real(coeff * np.conj(rep_val))
                else:
                    # Matrix rep
                    rep_mat = np.atleast_2d(rep_val)
                    dim = rep_mat.shape[0]
                    rep_inv = np.conj(rep_mat.T)  # For unitary reps
                    contribution = dim * np.real(np.trace(np.atleast_2d(coeff) @ rep_inv))

                val += contribution

            result[g] = val / n

        return result

    def peter_weyl_decomposition(
        self,
        f: Callable,
        group_elements: list,
        representations: dict[str, Callable],
    ) -> dict[str, float]:
        """Decompose f into irreducible components (Peter-Weyl).

        Computes the energy ||f̂(π)||² / |G| in each irreducible
        representation, showing how f distributes across the
        representation spectrum.

        Args:
            f: Function on the group.
            group_elements: Group elements.
            representations: Irreducible representations.

        Returns:
            Dict mapping rep name to energy fraction.
        """
        ft = self.fourier_transform(f, group_elements, representations)
        n = len(group_elements)

        # Total energy: ||f||² = (1/|G|) Σ |f(g)|²
        total_energy = sum(abs(f(g))**2 for g in group_elements) / n

        energy_by_rep = {}
        for name, coeff in ft.items():
            if np.isscalar(coeff):
                energy = abs(coeff)**2 / n
            else:
                energy = np.sum(np.abs(coeff)**2) / n
            energy_by_rep[name] = energy

        # Normalize to fractions
        if total_energy > 1e-15:
            for name in energy_by_rep:
                energy_by_rep[name] /= total_energy

        return energy_by_rep

    @staticmethod
    def cyclic_group_representations(n: int) -> dict[str, Callable]:
        """Generate all irreducible representations of Z/nZ.

        The irreps of Z/nZ are the characters χ_k(g) = exp(2πi k g / n).

        Args:
            n: Order of the cyclic group.

        Returns:
            Dict of n character functions.
        """
        reps = {}
        for k in range(n):
            reps[f"chi_{k}"] = lambda g, k=k, n=n: np.exp(2j * np.pi * k * g / n)
        return reps

    @staticmethod
    def plancherel_formula(
        f: Callable,
        group_elements: list,
        fourier_coeffs: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        """Verify Plancherel formula: ||f||² = (1/|G|) Σ dim(π) ||f̂(π)||².

        Args:
            f: Function on the group.
            group_elements: Group elements.
            fourier_coeffs: Fourier coefficients.

        Returns:
            (lhs, rhs) — should be equal.
        """
        n = len(group_elements)
        lhs = sum(abs(f(g))**2 for g in group_elements) / n

        rhs = 0.0
        for name, coeff in fourier_coeffs.items():
            if np.isscalar(coeff):
                rhs += abs(coeff)**2 / n
            else:
                dim = np.atleast_2d(coeff).shape[0]
                rhs += dim * np.sum(np.abs(coeff)**2) / n

        return lhs, rhs
