"""
Exterior Calculus
==================

Scope slots: CA4.4 (wedge product), CA4.5 (Hodge star),
CA4.6 (de Rham cohomology), CA4.7 (exterior derivative).

Pure numpy implementations of differential forms calculus:
- Wedge product: antisymmetric tensor product of forms
- Hodge star: metric-dependent duality operator
- De Rham cohomology: Betti numbers via rank computation
- Exterior derivative: d: Ω^k → Ω^{k+1}

Forms are represented as antisymmetric tensors. A k-form on an
n-dimensional space is stored as a (n choose k)-component vector
in the standard basis {dx^{i1} ∧ ... ∧ dx^{ik} : i1 < ... < ik}.

Author: Claude Code
Date: March 2026
"""

import logging
from itertools import combinations
from math import comb

import numpy as np

logger = logging.getLogger(__name__)


def _multi_index_to_flat(indices: tuple[int, ...], n: int) -> int:
    """Map sorted multi-index to flat position in lexicographic order."""
    k = len(indices)
    pos = 0
    for i, idx in enumerate(indices):
        start = indices[i - 1] + 1 if i > 0 else 0
        for j in range(start, idx):
            pos += comb(n - j - 1, k - i - 1)
    return pos


def _flat_to_multi_index(flat: int, k: int, n: int) -> tuple[int, ...]:
    """Map flat position back to sorted multi-index."""
    indices = []
    remaining = flat
    start = 0
    for i in range(k):
        for j in range(start, n):
            c = comb(n - j - 1, k - i - 1)
            if remaining < c:
                indices.append(j)
                start = j + 1
                break
            remaining -= c
    return tuple(indices)


def _sign_of_permutation(perm: tuple[int, ...]) -> int:
    """Compute sign of a permutation (+1 even, -1 odd)."""
    n = len(perm)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len % 2 == 0:
                sign *= -1
    return sign


# ============================================================================
# WEDGE PRODUCT (CA4.4)
# ============================================================================

class WedgeProduct:
    """Wedge (exterior) product of differential forms.

    The wedge product ∧: Ω^p × Ω^q → Ω^{p+q} is the antisymmetric
    tensor product:

        (α ∧ β)(v₁,...,v_{p+q}) = (1/(p!q!)) Σ sgn(σ) α(v_{σ(1)},...) β(v_{σ(p+1)},...)

    Properties:
    - Anticommutativity: α ∧ β = (-1)^{pq} β ∧ α
    - Associativity: (α ∧ β) ∧ γ = α ∧ (β ∧ γ)
    - α ∧ α = 0 for odd-degree forms

    Example:
        >>> wp = WedgeProduct(n=3)
        >>> # dx ∧ dy in R³
        >>> dx = np.array([1, 0, 0])  # 1-form
        >>> dy = np.array([0, 1, 0])  # 1-form
        >>> dxdy = wp.compute(dx, dy, p=1, q=1)
        >>> dxdy  # Should be [1, 0, 0] representing dx∧dy (basis: dx∧dy, dx∧dz, dy∧dz)
    """

    def __init__(self, n: int):
        """Initialize for n-dimensional ambient space.

        Args:
            n: Dimension of the underlying vector space.
        """
        self.n = n

    def compute(self, form1: np.ndarray, form2: np.ndarray, p: int, q: int) -> np.ndarray:
        """Compute wedge product α ∧ β.

        Args:
            form1: p-form coefficients, shape (C(n,p),).
            form2: q-form coefficients, shape (C(n,q),).
            p: Degree of form1.
            q: Degree of form2.

        Returns:
            (p+q)-form coefficients, shape (C(n,p+q),).
        """
        n = self.n
        r = p + q
        if r > n:
            return np.zeros(0)

        n_out = comb(n, r)
        result = np.zeros(n_out)

        # Enumerate all basis multi-indices for the output
        basis_p = list(combinations(range(n), p))
        basis_q = list(combinations(range(n), q))
        basis_r = list(combinations(range(n), r))

        for I, idx_I in enumerate(basis_p):
            if abs(form1[I]) < 1e-15:
                continue
            for J, idx_J in enumerate(basis_q):
                if abs(form2[J]) < 1e-15:
                    continue

                # Check for repeated indices (result is zero)
                merged = idx_I + idx_J
                if len(set(merged)) < r:
                    continue

                # Sort to canonical order and compute sign
                sorted_indices = tuple(sorted(merged))
                # Find the permutation that sorts merged
                temp = list(merged)
                sign = 1
                for i in range(len(temp)):
                    for j in range(i + 1, len(temp)):
                        if temp[i] > temp[j]:
                            temp[i], temp[j] = temp[j], temp[i]
                            sign *= -1

                # Find position of sorted_indices in basis_r
                K = basis_r.index(sorted_indices)
                result[K] += sign * form1[I] * form2[J]

        return result

    def is_anticommutative(self, form1: np.ndarray, form2: np.ndarray, p: int, q: int) -> bool:
        """Verify α ∧ β = (-1)^{pq} β ∧ α."""
        ab = self.compute(form1, form2, p, q)
        ba = self.compute(form2, form1, q, p)
        sign = (-1) ** (p * q)
        return np.allclose(ab, sign * ba, atol=1e-12)


# ============================================================================
# HODGE STAR (CA4.5)
# ============================================================================

class HodgeStar:
    """Hodge star operator ★: Ω^k → Ω^{n-k}.

    The Hodge dual maps a k-form to an (n-k)-form using the metric:
        ★(dx^{i1} ∧ ... ∧ dx^{ik}) = √|det g| · g^{-1}_{...} ε_{...} dx^{j1} ∧ ... ∧ dx^{jn-k}

    For the Euclidean metric (g = I), this simplifies to:
        ★(dx^{i1} ∧ ... ∧ dx^{ik}) = sgn(i1,...,ik,j1,...,jn-k) dx^{j1} ∧ ... ∧ dx^{jn-k}

    Properties:
    - ★★α = (-1)^{k(n-k)} α  (in Euclidean signature)
    - ⟨α, β⟩ vol = α ∧ ★β  (inner product on forms)

    Example:
        >>> hs = HodgeStar(n=3)
        >>> # ★dx = dy ∧ dz in R³
        >>> dx = np.array([1.0, 0.0, 0.0])
        >>> star_dx = hs.compute(dx, metric=np.eye(3))
    """

    def __init__(self, n: int):
        self.n = n

    def compute(self, form: np.ndarray, metric: np.ndarray, k: int | None = None) -> np.ndarray:
        """Compute Hodge dual ★α.

        Args:
            form: k-form coefficients.
            metric: Metric tensor g of shape (n, n).
            k: Degree of the form (inferred from size if None).

        Returns:
            (n-k)-form coefficients.
        """
        n = self.n
        if k is None:
            # Infer k from form size
            for kk in range(n + 1):
                if comb(n, kk) == len(form):
                    k = kk
                    break
            if k is None:
                raise ValueError(f"Cannot infer form degree from size {len(form)} in dimension {n}")

        nk = n - k
        n_out = comb(n, nk)
        result = np.zeros(n_out)

        det_g = np.linalg.det(metric)
        sqrt_det_g = np.sqrt(abs(det_g))
        g_inv = np.linalg.inv(metric)

        basis_k = list(combinations(range(n), k))
        basis_nk = list(combinations(range(n), nk))

        for I, idx_I in enumerate(basis_k):
            if abs(form[I]) < 1e-15:
                continue

            # Complement indices
            complement = tuple(j for j in range(n) if j not in idx_I)
            # This complement is already sorted

            # Sign: sgn of permutation (idx_I + complement) relative to (0,1,...,n-1)
            perm = idx_I + complement
            sign = 1
            temp = list(perm)
            for i in range(len(temp)):
                for j in range(i + 1, len(temp)):
                    if temp[i] > temp[j]:
                        temp[i], temp[j] = temp[j], temp[i]
                        sign *= -1

            # For non-Euclidean metric, raise indices using g^{-1}
            # Simplified: for diagonal-dominant metrics
            metric_factor = 1.0
            for idx in idx_I:
                metric_factor *= g_inv[idx, idx]

            # Find complement in basis
            J = basis_nk.index(complement)
            result[J] += sign * sqrt_det_g * metric_factor * form[I]

        return result

    def inner_product(self, alpha: np.ndarray, beta: np.ndarray, metric: np.ndarray, k: int) -> float:
        """Compute inner product ⟨α, β⟩ = ∫ α ∧ ★β (pointwise).

        For constant forms, this is the algebraic inner product.

        Args:
            alpha, beta: k-forms.
            metric: Metric tensor.
            k: Form degree.

        Returns:
            Inner product value.
        """
        star_beta = self.compute(beta, metric, k)
        wp = WedgeProduct(self.n)
        vol_form = wp.compute(alpha, star_beta, k, self.n - k)
        # vol_form should be an n-form (scalar times volume element)
        return float(vol_form[0]) if len(vol_form) > 0 else 0.0


# ============================================================================
# EXTERIOR DERIVATIVE (CA4.7)
# ============================================================================

class ExteriorDerivative:
    """Exterior derivative d: Ω^k → Ω^{k+1}.

    The exterior derivative generalizes grad, curl, and div:
    - d on 0-forms (functions) = gradient
    - d on 1-forms in R³ = curl
    - d on 2-forms in R³ = divergence

    Key property: d ∘ d = 0 (exactness).

    For a k-form α = Σ α_I dx^I, the exterior derivative is:
        dα = Σ_I Σ_j (∂α_I/∂x^j) dx^j ∧ dx^I

    Example:
        >>> ed = ExteriorDerivative(n=3)
        >>> # d(f) where f = x² + y² (0-form, gradient)
        >>> # ∂f/∂x = 2x, ∂f/∂y = 2y, ∂f/∂z = 0
        >>> # At point (1,1,0): df = 2dx + 2dy
        >>> grad_f = np.array([2.0, 2.0, 0.0])  # This IS the exterior derivative
    """

    def __init__(self, n: int):
        self.n = n

    def compute(
        self,
        form_field: np.ndarray,
        k: int,
        dx: float = 1.0,
    ) -> np.ndarray:
        """Compute exterior derivative of a discretized form field.

        For a k-form field defined on a grid, computes d(form) which
        is a (k+1)-form field.

        Args:
            form_field: Array of shape (grid_shape..., C(n,k)) where
                grid_shape covers the spatial grid.
            k: Degree of the input form.
            dx: Grid spacing (uniform).

        Returns:
            (k+1)-form field of shape (grid_shape..., C(n,k+1)).
        """
        n = self.n
        n_in = comb(n, k)
        n_out = comb(n, k + 1)

        if k + 1 > n:
            # d of an n-form is zero
            return np.zeros(form_field.shape[:-1] + (0,))

        # Determine grid dimensions
        grid_shape = form_field.shape[:-1]
        grid_ndim = len(grid_shape)

        if grid_ndim == 0:
            raise ValueError("Form field must have at least 1 grid dimension")

        basis_k = list(combinations(range(n), k))
        basis_k1 = list(combinations(range(n), k + 1))

        result = np.zeros(grid_shape + (n_out,))

        # For each output basis element dx^J (J is a (k+1)-tuple)
        for J_idx, J in enumerate(basis_k1):
            for pos, j in enumerate(J):
                # Remaining indices after removing j from J
                I = J[:pos] + J[pos + 1:]
                sign = (-1) ** pos

                # Find I in basis_k
                try:
                    I_idx = basis_k.index(I)
                except ValueError:
                    continue

                # Partial derivative ∂/∂x^j of form component I
                if j < grid_ndim:
                    # Compute numerical derivative along axis j
                    deriv = np.gradient(form_field[..., I_idx], dx, axis=j)
                    result[..., J_idx] += sign * deriv

        return result

    @staticmethod
    def verify_dd_zero(form: np.ndarray, n: int, k: int, dx: float = 1.0) -> float:
        """Verify d(d(form)) = 0 (fundamental property).

        Args:
            form: k-form field.
            n: Ambient dimension.
            k: Form degree.
            dx: Grid spacing.

        Returns:
            Max absolute value of d(d(form)) — should be ~0.
        """
        ed = ExteriorDerivative(n)
        d_form = ed.compute(form, k, dx)
        if d_form.shape[-1] == 0:
            return 0.0
        dd_form = ed.compute(d_form, k + 1, dx)
        if dd_form.shape[-1] == 0:
            return 0.0
        return float(np.max(np.abs(dd_form)))


# ============================================================================
# DE RHAM COHOMOLOGY (CA4.6)
# ============================================================================

class DeRhamCohomology:
    """De Rham cohomology: compute Betti numbers from a simplicial complex.

    The de Rham theorem connects topology and calculus:
        H^k_dR(M) ≅ H^k(M; R)

    The k-th Betti number β_k = dim H^k counts "k-dimensional holes":
    - β₀ = number of connected components
    - β₁ = number of independent loops
    - β₂ = number of enclosed cavities

    We compute cohomology via the chain complex:
        ... → C^{k-1} →^{d_{k-1}} C^k →^{d_k} C^{k+1} → ...
        H^k = ker(d_k) / im(d_{k-1})
        β_k = dim ker(d_k) - dim im(d_{k-1})

    Example:
        >>> dr = DeRhamCohomology()
        >>> # Simplicial complex for a triangle (1-sphere boundary)
        >>> # Vertices: 0, 1, 2
        >>> # Edges: (0,1), (1,2), (0,2)
        >>> complex = {
        ...     0: [(0,), (1,), (2,)],           # vertices
        ...     1: [(0,1), (1,2), (0,2)],         # edges
        ... }
        >>> betti = dr.compute(complex)
        >>> betti  # [1, 1] — one component, one loop
    """

    def compute(self, complex: dict) -> list[int]:
        """Compute Betti numbers of a simplicial complex.

        Args:
            complex: Dict mapping dimension k to list of k-simplices.
                Each simplex is a tuple of vertex indices (sorted).

        Returns:
            List of Betti numbers [β₀, β₁, ..., β_d].
        """
        dims = sorted(complex.keys())
        if not dims:
            return []

        max_dim = max(dims)

        # Build boundary matrices
        boundary_matrices = {}
        for k in range(1, max_dim + 1):
            if k in complex and (k - 1) in complex:
                boundary_matrices[k] = self._boundary_matrix(
                    complex[k], complex[k - 1]
                )

        # Compute Betti numbers: β_k = dim ker(∂_k) - dim im(∂_{k+1})
        betti = []
        for k in range(max_dim + 1):
            n_k = len(complex.get(k, []))
            if n_k == 0:
                betti.append(0)
                continue

            # dim ker(∂_k): nullity of boundary map from k-chains
            if k in boundary_matrices:
                B_k = boundary_matrices[k]
                rank_k = np.linalg.matrix_rank(B_k, tol=1e-10)
                ker_k = B_k.shape[1] - rank_k  # nullity
            else:
                ker_k = n_k  # No boundary map means everything is a cycle

            # dim im(∂_{k+1}): rank of boundary map from (k+1)-chains
            if (k + 1) in boundary_matrices:
                B_k1 = boundary_matrices[k + 1]
                im_k = np.linalg.matrix_rank(B_k1, tol=1e-10)
            else:
                im_k = 0

            betti.append(max(0, ker_k - im_k))

        return betti

    @staticmethod
    def _boundary_matrix(simplices_k: list[tuple[int, ...]], simplices_km1: list[tuple[int, ...]]) -> np.ndarray:
        """Build the boundary matrix ∂_k.

        ∂_k maps k-chains to (k-1)-chains:
            ∂[v₀, ..., v_k] = Σ (-1)^i [v₀, ..., v̂ᵢ, ..., v_k]

        Args:
            simplices_k: List of k-simplices.
            simplices_km1: List of (k-1)-simplices.

        Returns:
            Boundary matrix of shape (n_{k-1}, n_k).
        """
        n_k = len(simplices_k)
        n_km1 = len(simplices_km1)

        # Index lookup for (k-1)-simplices
        idx_map = {s: i for i, s in enumerate(simplices_km1)}

        B = np.zeros((n_km1, n_k))

        for j, simplex in enumerate(simplices_k):
            for i in range(len(simplex)):
                # Remove i-th vertex
                face = simplex[:i] + simplex[i + 1:]
                if face in idx_map:
                    sign = (-1) ** i
                    B[idx_map[face], j] = sign

        return B

    @staticmethod
    def euler_characteristic(betti_numbers: list[int]) -> int:
        """Compute Euler characteristic χ = Σ (-1)^k β_k.

        Args:
            betti_numbers: List of Betti numbers.

        Returns:
            Euler characteristic.
        """
        return sum((-1) ** k * b for k, b in enumerate(betti_numbers))
