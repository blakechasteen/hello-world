"""
Sheaf Extensions — Sheaf Morphisms and Derived Cohomology
==========================================================

Sheaves generalize vector bundles by assigning algebraic data to
open sets with consistency conditions. Sheaf morphisms capture
structure-preserving maps, and derived cohomology measures
obstruction to global sections.

Core Concepts:
- Sheaf Morphism: Structure-preserving map between sheaves
- Kernel/Image: Sub-sheaf constructions from morphisms
- Derived Cohomology: H^n(X, F) via injective/Cech resolution

Mathematical Foundation:
Sheaf: F(U) for each open U, with restriction maps ρ_{U,V}: F(U) → F(V)
Morphism: φ: F → G with commuting squares: φ_V ∘ ρ^F = ρ^G ∘ φ_U
Cohomology: 0 → F(X) → ΠF(Uᵢ) → ΠF(Uᵢ∩Uⱼ) → ... (Cech complex)
Exactness: 0 → ker φ → F → G → coker φ → 0

Applications to Warp Space:
- Consistent local-to-global information aggregation
- Detecting when local embedding patches fail to extend globally
- Cohomological obstruction analysis for memory consistency
- Sheaf-theoretic data fusion across overlapping contexts

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Stalk:
    """Data at a single point/set in the sheaf.

    Attributes:
        data: Vector space element (numpy array).
        dimension: Dimension of the stalk.
    """
    data: np.ndarray

    @property
    def dimension(self) -> int:
        return len(self.data) if self.data.ndim == 1 else self.data.shape[0]


@dataclass
class Sheaf:
    """Cellular sheaf over a cell complex.

    A sheaf assigns a vector space (stalk) to each cell and a
    linear map (restriction) to each incidence relation.

    Attributes:
        stalks: Map from cell index to stalk dimension.
        restrictions: Map from (face, coface) to restriction matrix.
    """
    stalks: dict[int, int]
    restrictions: dict[tuple[int, int], np.ndarray]

    def stalk_dimension(self, cell: int) -> int:
        """Get dimension of stalk at cell."""
        return self.stalks.get(cell, 0)

    def restriction(self, face: int, coface: int) -> np.ndarray | None:
        """Get restriction map from coface to face."""
        return self.restrictions.get((face, coface))

    def total_dimension(self) -> int:
        """Sum of all stalk dimensions."""
        return sum(self.stalks.values())

    def section(self, assignments: dict[int, np.ndarray]) -> bool:
        """Check if assignments form a global section.

        A global section must be consistent: for each restriction
        (face, coface), F(face→coface) @ section(coface) = section(face).
        """
        for (face, coface), R in self.restrictions.items():
            if face in assignments and coface in assignments:
                expected = R @ assignments[coface]
                actual = assignments[face]
                if not np.allclose(expected, actual, atol=1e-10):
                    return False
        return True


# ============================================================================
# SHEAF MORPHISM
# ============================================================================

class SheafMorphism:
    """Morphism between cellular sheaves.

    A sheaf morphism φ: F → G assigns to each cell σ a linear map
    φ_σ: F(σ) → G(σ) such that the restriction maps commute:
    φ_face ∘ ρ^F_{face,coface} = ρ^G_{face,coface} ∘ φ_coface

    Example:
        >>> F = Sheaf(stalks={0: 2, 1: 2}, restrictions={(0, 1): np.eye(2)})
        >>> G = Sheaf(stalks={0: 3, 1: 3}, restrictions={(0, 1): np.eye(3)})
        >>> maps = {0: np.random.randn(3, 2), 1: np.random.randn(3, 2)}
        >>> phi = SheafMorphism(F, G, maps)
        >>> phi.is_valid()
    """

    def __init__(self, source: Sheaf, target: Sheaf, maps: dict[int, np.ndarray]):
        """Initialize sheaf morphism.

        Args:
            source: Source sheaf F.
            target: Target sheaf G.
            maps: Map from cell index to matrix φ_σ: F(σ) → G(σ).
        """
        self.source = source
        self.target = target
        self.maps = maps

    def is_valid(self, tol: float = 1e-10) -> bool:
        """Check if this is a valid sheaf morphism.

        Verifies commutativity: φ_face ∘ ρ^F = ρ^G ∘ φ_coface
        for all incidence relations.
        """
        for (face, coface) in self.source.restrictions:
            if face not in self.maps or coface not in self.maps:
                continue

            rho_F = self.source.restrictions[(face, coface)]
            rho_G = self.target.restrictions.get((face, coface))
            if rho_G is None:
                continue

            phi_face = self.maps[face]
            phi_coface = self.maps[coface]

            lhs = phi_face @ rho_F
            rhs = rho_G @ phi_coface

            if not np.allclose(lhs, rhs, atol=tol):
                return False

        return True

    def kernel(self) -> Sheaf:
        """Compute kernel sheaf ker(φ).

        ker(φ)(σ) = ker(φ_σ) for each cell σ.
        Restriction maps are induced from F.

        Returns:
            Kernel sheaf.
        """
        stalks: dict[int, int] = {}
        restrictions: dict[tuple[int, int], np.ndarray] = {}
        bases: dict[int, np.ndarray] = {}  # Basis for each kernel

        for cell, phi in self.maps.items():
            # Compute kernel of φ_cell
            U, S, Vt = np.linalg.svd(phi, full_matrices=True)
            rank = np.sum(S > 1e-10)
            null_dim = Vt.shape[0] - rank
            stalks[cell] = null_dim
            if null_dim > 0:
                bases[cell] = Vt[rank:].T  # Columns form kernel basis
            else:
                bases[cell] = np.zeros((phi.shape[1], 0))

        # Induced restriction maps on kernel
        for (face, coface), rho_F in self.source.restrictions.items():
            if face in bases and coface in bases:
                B_face = bases[face]
                B_coface = bases[coface]
                if B_face.shape[1] > 0 and B_coface.shape[1] > 0:
                    # Project rho_F restricted to ker onto ker basis
                    projected = B_face.T @ rho_F @ B_coface
                    restrictions[(face, coface)] = projected

        return Sheaf(stalks=stalks, restrictions=restrictions)

    def image(self) -> Sheaf:
        """Compute image sheaf im(φ).

        im(φ)(σ) = im(φ_σ) for each cell σ.
        Restriction maps are induced from G.

        Returns:
            Image sheaf.
        """
        stalks: dict[int, int] = {}
        restrictions: dict[tuple[int, int], np.ndarray] = {}
        bases: dict[int, np.ndarray] = {}

        for cell, phi in self.maps.items():
            U, S, Vt = np.linalg.svd(phi, full_matrices=False)
            rank = np.sum(S > 1e-10)
            stalks[cell] = rank
            if rank > 0:
                bases[cell] = U[:, :rank]  # Columns form image basis
            else:
                bases[cell] = np.zeros((phi.shape[0], 0))

        # Induced restriction maps
        for (face, coface), rho_G in self.target.restrictions.items():
            if face in bases and coface in bases:
                B_face = bases[face]
                B_coface = bases[coface]
                if B_face.shape[1] > 0 and B_coface.shape[1] > 0:
                    projected = B_face.T @ rho_G @ B_coface
                    restrictions[(face, coface)] = projected

        return Sheaf(stalks=stalks, restrictions=restrictions)

    def is_injective(self) -> bool:
        """Check if morphism is injective (ker φ = 0 at each stalk)."""
        for cell, phi in self.maps.items():
            S = np.linalg.svd(phi, compute_uv=False)
            rank = np.sum(S > 1e-10)
            if rank < phi.shape[1]:
                return False
        return True

    def is_surjective(self) -> bool:
        """Check if morphism is surjective (im φ = G at each stalk)."""
        for cell, phi in self.maps.items():
            S = np.linalg.svd(phi, compute_uv=False)
            rank = np.sum(S > 1e-10)
            target_dim = self.target.stalk_dimension(cell)
            if rank < target_dim:
                return False
        return True

    def cokernel_dimensions(self) -> dict[int, int]:
        """Compute cokernel dimensions at each cell.

        coker(φ)_σ = G(σ) / im(φ_σ)
        """
        dims: dict[int, int] = {}
        for cell, phi in self.maps.items():
            S = np.linalg.svd(phi, compute_uv=False)
            rank = np.sum(S > 1e-10)
            target_dim = self.target.stalk_dimension(cell)
            dims[cell] = target_dim - rank
        return dims


# ============================================================================
# DERIVED SHEAF COHOMOLOGY
# ============================================================================

class DerivedSheafCohomology:
    """Sheaf cohomology via Cech-like coboundary operators.

    Computes H^n(X, F) — the n-th cohomology of sheaf F over space X.

    H^0 = global sections (consistent assignments)
    H^1 = obstruction to extending local sections
    H^n = higher obstructions (gluing conditions)

    For cellular sheaves, the coboundary operator δ is built from
    the restriction maps and the incidence structure.

    Example:
        >>> dsc = DerivedSheafCohomology()
        >>> F = Sheaf(stalks={0: 2, 1: 2, 2: 1},
        ...           restrictions={(2, 0): np.array([[1, 0]]),
        ...                         (2, 1): np.array([[0, 1]])})
        >>> dims = dsc.compute(F, cell_dimensions={0: 0, 1: 0, 2: 1})
    """

    def compute(
        self,
        sheaf: Sheaf,
        cell_dimensions: dict[int, int],
        max_degree: int = 3,
    ) -> list[int]:
        """Compute sheaf cohomology dimensions.

        Args:
            sheaf: Cellular sheaf.
            cell_dimensions: Dimension of each cell in the base complex.
            max_degree: Maximum cohomological degree to compute.

        Returns:
            List of Betti numbers [dim H^0, dim H^1, ..., dim H^{max_degree}].
        """
        # Group cells by dimension
        cells_by_dim: dict[int, list[int]] = {}
        for cell, dim in cell_dimensions.items():
            if dim not in cells_by_dim:
                cells_by_dim[dim] = []
            cells_by_dim[dim].append(cell)

        # Sort cells within each dimension for consistent ordering
        for dim in cells_by_dim:
            cells_by_dim[dim].sort()

        # Build coboundary matrices
        coboundary_matrices = []
        max_cell_dim = max(cell_dimensions.values()) if cell_dimensions else 0

        for p in range(max_cell_dim + 1):
            if p not in cells_by_dim or p + 1 not in cells_by_dim:
                coboundary_matrices.append(None)
                continue

            p_cells = cells_by_dim[p]
            q_cells = cells_by_dim[p + 1]

            # Domain dimension: sum of stalk dims over p-cells
            domain_dims = [(c, sheaf.stalk_dimension(c)) for c in p_cells]
            # Codomain dimension: sum of stalk dims over (p+1)-cells
            codomain_dims = [(c, sheaf.stalk_dimension(c)) for c in q_cells]

            total_domain = sum(d for _, d in domain_dims)
            total_codomain = sum(d for _, d in codomain_dims)

            if total_domain == 0 or total_codomain == 0:
                coboundary_matrices.append(None)
                continue

            delta = np.zeros((total_codomain, total_domain))

            # Fill in blocks from restriction maps
            row_offset = 0
            for q_cell, q_dim in codomain_dims:
                col_offset = 0
                for p_cell, p_dim in domain_dims:
                    R = sheaf.restriction(p_cell, q_cell)
                    if R is not None:
                        r, c = R.shape
                        delta[row_offset: row_offset + r, col_offset: col_offset + c] = R
                    col_offset += p_dim
                row_offset += q_dim

            coboundary_matrices.append(delta)

        # Compute cohomology: H^p = ker δ_p / im δ_{p-1}
        betti = []
        for p in range(min(max_degree + 1, max_cell_dim + 2)):
            # Kernel of δ_p
            if p < len(coboundary_matrices) and coboundary_matrices[p] is not None:
                delta_p = coboundary_matrices[p]
                U, S, Vt = np.linalg.svd(delta_p, full_matrices=True)
                rank_p = np.sum(S > 1e-10)
                ker_dim = delta_p.shape[1] - rank_p
            else:
                # No outgoing coboundary — entire space is kernel
                if p in cells_by_dim:
                    ker_dim = sum(sheaf.stalk_dimension(c) for c in cells_by_dim[p])
                else:
                    ker_dim = 0

            # Image of δ_{p-1}
            if p > 0 and p - 1 < len(coboundary_matrices) and coboundary_matrices[p - 1] is not None:
                delta_pm1 = coboundary_matrices[p - 1]
                S_pm1 = np.linalg.svd(delta_pm1, compute_uv=False)
                im_dim = int(np.sum(S_pm1 > 1e-10))
            else:
                im_dim = 0

            betti.append(max(0, ker_dim - im_dim))

        return betti

    def global_sections(self, sheaf: Sheaf) -> np.ndarray:
        """Compute space of global sections H^0(X, F).

        A global section assigns a vector to each cell such that
        all restriction maps are satisfied.

        Args:
            sheaf: Cellular sheaf.

        Returns:
            Matrix whose columns form a basis for global sections.
            Shape (total_stalk_dim, dim_H0).
        """
        cells = sorted(sheaf.stalks.keys())
        dims = [sheaf.stalk_dimension(c) for c in cells]
        total = sum(dims)

        if total == 0:
            return np.zeros((0, 0))

        # Build system of equations from restrictions
        equations = []
        offsets = {}
        offset = 0
        for c, d in zip(cells, dims):
            offsets[c] = offset
            offset += d

        for (face, coface), R in sheaf.restrictions.items():
            if face not in offsets or coface not in offsets:
                continue
            r, c = R.shape
            # Equation: R @ x_coface - x_face = 0
            row = np.zeros((r, total))
            row[:, offsets[coface]: offsets[coface] + c] = R
            row[:, offsets[face]: offsets[face] + r] = -np.eye(r)
            equations.append(row)

        if not equations:
            return np.eye(total)

        A = np.vstack(equations)
        U, S, Vt = np.linalg.svd(A, full_matrices=True)
        rank = np.sum(S > 1e-10)
        null_dim = Vt.shape[0] - rank

        if null_dim == 0:
            return np.zeros((total, 0))

        return Vt[rank:].T  # Columns are basis vectors for global sections

    def euler_characteristic(self, betti: list[int]) -> int:
        """Compute Euler characteristic from Betti numbers.

        χ = Σ (-1)^n dim H^n

        Args:
            betti: List of Betti numbers.

        Returns:
            Euler characteristic.
        """
        return sum((-1) ** n * b for n, b in enumerate(betti))
