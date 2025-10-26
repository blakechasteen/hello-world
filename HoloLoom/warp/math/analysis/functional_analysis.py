"""
Functional Analysis for HoloLoom Warp Drive
============================================

Infinite-dimensional spaces and operator theory for advanced ML and signal processing.

Core Concepts:
- Normed Spaces & Banach Spaces: Complete normed vector spaces
- Hilbert Spaces: Complete inner product spaces (L², function spaces)
- Bounded Linear Operators: Continuous linear maps
- Spectral Theory: Eigenvalues, eigenfunctions, spectral decomposition
- Sobolev Spaces: Function spaces with weak derivatives (PDEs, neural operators)
- Compact Operators: Generalized finite rank
- Dual Spaces: Linear functionals

Mathematical Foundation:
Hilbert space H with inner product ⟨·,·⟩:
- Complete: Cauchy sequences converge
- Inner product induces norm: ||x|| = √⟨x,x⟩
- Orthonormal basis: {eₙ}, x = Σ ⟨x, eₙ⟩eₙ
- Riesz representation: Every bounded linear functional is inner product

Applications:
- Neural operators (infinite-dimensional networks)
- Quantum mechanics formulation
- Signal processing (L² spaces)
- PDE-constrained optimization
- Kernel methods (RKHS theory)

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Normed Spaces and Banach Spaces
# ============================================================================

class NormedSpace:
    """
    Normed vector space (X, ||·||).

    Banach space if complete (all Cauchy sequences converge).
    """

    def __init__(self, elements: Optional[List] = None, norm: Optional[Callable] = None, name: str = "X"):
        self.elements = elements or []
        self.norm = norm or self._l2_norm
        self.name = name

        logger.info(f"Normed space {name} initialized")

    @staticmethod
    def _l2_norm(x: np.ndarray) -> float:
        """Default L² norm."""
        return np.linalg.norm(x)

    def distance(self, x, y) -> float:
        """Induced metric: d(x,y) = ||x - y||"""
        return self.norm(x - y)

    def is_complete(self) -> bool:
        """
        Check if space is Banach (complete).

        Finite-dimensional normed spaces are always complete.
        """
        # Finite-dimensional spaces are complete
        if self.elements and len(self.elements) < float('inf'):
            return True

        # Check if first element suggests finite dimension
        if self.elements and hasattr(self.elements[0], 'shape'):
            return True  # ℝⁿ is complete

        return None  # Cannot determine in general

    def unit_ball(self) -> List:
        """Return elements in closed unit ball B̄ = {x : ||x|| ≤ 1}"""
        if not self.elements:
            return []
        return [x for x in self.elements if self.norm(x) <= 1.0]

    def unit_sphere(self) -> List:
        """Return elements on unit sphere S = {x : ||x|| = 1}"""
        if not self.elements:
            return []
        return [x for x in self.elements if abs(self.norm(x) - 1.0) < 1e-6]


# ============================================================================
# Hilbert Spaces
# ============================================================================

class HilbertSpace:
    """
    Hilbert space: Complete inner product space.

    Foundation for quantum mechanics, signal processing, kernel methods.
    """

    def __init__(self,
                 elements: Optional[List] = None,
                 inner_product: Optional[Callable] = None,
                 name: str = "H"):
        self.elements = elements or []
        self.inner_product = inner_product or self._standard_inner_product
        self.name = name

        logger.info(f"Hilbert space {name} initialized")

    @staticmethod
    def _standard_inner_product(x: np.ndarray, y: np.ndarray) -> float:
        """Standard inner product: ⟨x,y⟩ = x·y"""
        return np.dot(x, y)

    def norm(self, x) -> float:
        """Induced norm: ||x|| = √⟨x,x⟩"""
        return np.sqrt(abs(self.inner_product(x, x)))

    def distance(self, x, y) -> float:
        """Induced metric: d(x,y) = ||x - y||"""
        return self.norm(x - y)

    def angle(self, x, y) -> float:
        """
        Angle between vectors: cos(θ) = ⟨x,y⟩ / (||x|| ||y||)
        """
        inner = self.inner_product(x, y)
        norm_product = self.norm(x) * self.norm(y)

        if norm_product < 1e-10:
            return 0.0

        cos_theta = inner / norm_product
        # Clamp to [-1, 1] for numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.arccos(cos_theta)

    def are_orthogonal(self, x, y, tolerance: float = 1e-6) -> bool:
        """Check if ⟨x,y⟩ = 0"""
        return abs(self.inner_product(x, y)) < tolerance

    def gram_schmidt(self, vectors: List) -> List:
        """
        Gram-Schmidt orthogonalization.

        Returns orthonormal basis from linearly independent vectors.
        """
        orthonormal = []

        for v in vectors:
            # Subtract projections onto previous vectors
            u = v.copy()
            for e in orthonormal:
                projection = self.inner_product(v, e) * e
                u = u - projection

            # Normalize
            norm_u = self.norm(u)
            if norm_u > 1e-10:  # Check linear independence
                orthonormal.append(u / norm_u)

        return orthonormal

    def projection_onto_subspace(self, x, basis: List) -> np.ndarray:
        """
        Project x onto subspace spanned by orthonormal basis.

        proj(x) = Σᵢ ⟨x, eᵢ⟩eᵢ
        """
        projection = np.zeros_like(x)

        for e in basis:
            coef = self.inner_product(x, e)
            projection = projection + coef * e

        return projection

    def riesz_representor(self, functional: Callable) -> np.ndarray:
        """
        Riesz representation theorem: Every bounded linear functional
        f: H → ℝ has unique representation f(x) = ⟨x, y⟩ for some y ∈ H.

        Approximates y given f (requires basis).
        """
        # This is a simplified version - full version needs complete basis
        if not self.elements:
            raise ValueError("Need elements to compute Riesz representor")

        # Use first few elements as approximate basis
        basis_candidates = self.elements[:min(100, len(self.elements))]
        basis = self.gram_schmidt(basis_candidates)

        # Compute coefficients: f(eᵢ) = ⟨eᵢ, y⟩
        coefficients = [functional(e) for e in basis]

        # Reconstruct: y = Σ f(eᵢ)eᵢ
        y = sum(c * e for c, e in zip(coefficients, basis))

        return y


# ============================================================================
# Bounded Linear Operators
# ============================================================================

class BoundedOperator:
    """
    Bounded linear operator T: H₁ → H₂.

    ||T|| = sup{||Tx|| : ||x|| = 1} < ∞
    """

    def __init__(self,
                 operator: Callable,
                 domain: HilbertSpace,
                 codomain: HilbertSpace,
                 name: str = "T"):
        self.operator = operator
        self.domain = domain
        self.codomain = codomain
        self.name = name

        logger.info(f"Bounded operator {name}: {domain.name} → {codomain.name}")

    def __call__(self, x):
        """Apply operator: Tx"""
        return self.operator(x)

    def operator_norm(self, sample_size: int = 100) -> float:
        """
        Compute operator norm: ||T|| = sup{||Tx|| / ||x|| : x ≠ 0}

        Uses sampling for infinite-dimensional spaces.
        """
        if not self.domain.elements:
            logger.warning("No domain elements to compute norm")
            return None

        max_ratio = 0.0

        for x in self.domain.elements[:sample_size]:
            norm_x = self.domain.norm(x)
            if norm_x < 1e-10:
                continue

            Tx = self.operator(x)
            norm_Tx = self.codomain.norm(Tx)

            ratio = norm_Tx / norm_x
            max_ratio = max(max_ratio, ratio)

        return max_ratio

    def is_bounded(self, sample_size: int = 100) -> bool:
        """Check if operator norm is finite."""
        norm = self.operator_norm(sample_size)
        return norm is not None and np.isfinite(norm)

    def adjoint(self) -> 'BoundedOperator':
        """
        Compute adjoint operator T*: H₂ → H₁.

        Defined by: ⟨Tx, y⟩ = ⟨x, T*y⟩
        """
        # For finite-dimensional case with matrix representation
        # This is a simplified version

        def adjoint_operator(y):
            # Use Riesz representation to find x such that T*y = x
            # This requires solving ⟨Tx', y⟩ = ⟨x', x⟩ for all x'

            # Simplified: if T is matrix A, then T* is Aᵀ
            # For general case, need more structure

            logger.warning("Adjoint computation is simplified")
            return y  # Placeholder

        return BoundedOperator(
            operator=adjoint_operator,
            domain=self.codomain,
            codomain=self.domain,
            name=f"{self.name}*"
        )

    def is_self_adjoint(self, tolerance: float = 1e-6) -> bool:
        """
        Check if T = T* (self-adjoint/Hermitian).

        Important for spectral theorem.
        """
        # Sample test: ⟨Tx, y⟩ = ⟨x, Ty⟩ for all x, y

        if not self.domain.elements or len(self.domain.elements) < 2:
            return None

        for x in self.domain.elements[:10]:
            for y in self.domain.elements[:10]:
                Tx = self.operator(x)
                Ty = self.operator(y)

                lhs = self.codomain.inner_product(Tx, y)
                rhs = self.domain.inner_product(x, Ty)

                if abs(lhs - rhs) > tolerance:
                    return False

        return True


# ============================================================================
# Spectral Theory
# ============================================================================

class SpectralAnalyzer:
    """
    Spectral theory for operators.

    For self-adjoint operators: spectral decomposition T = Σ λᵢ|eᵢ⟩⟨eᵢ|
    """

    @staticmethod
    def eigendecomposition(operator_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition for finite-dimensional operator.

        Returns (eigenvalues, eigenvectors)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(operator_matrix)

        # Sort by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    @staticmethod
    def spectral_decomposition(operator_matrix: np.ndarray) -> Tuple[List[float], List[np.ndarray]]:
        """
        Spectral theorem: T = Σ λᵢ Pᵢ where Pᵢ are projection operators.

        Returns (eigenvalues, projection_matrices)
        """
        eigenvalues, eigenvectors = SpectralAnalyzer.eigendecomposition(operator_matrix)

        projections = []
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i:i+1]
            P = v @ v.T  # Projection onto eigenspace
            projections.append(P)

        return eigenvalues.tolist(), projections

    @staticmethod
    def resolvent(operator_matrix: np.ndarray, z: complex) -> np.ndarray:
        """
        Resolvent operator: R(z) = (z·I - T)⁻¹

        Exists for z not in spectrum.
        """
        I = np.eye(operator_matrix.shape[0])
        try:
            return np.linalg.inv(z * I - operator_matrix)
        except np.linalg.LinAlgError:
            logger.warning(f"Resolvent does not exist at z={z} (in spectrum)")
            return None

    @staticmethod
    def spectrum(operator_matrix: np.ndarray) -> List[complex]:
        """
        Compute spectrum (set of eigenvalues).
        """
        eigenvalues, _ = np.linalg.eig(operator_matrix)
        return eigenvalues.tolist()

    @staticmethod
    def spectral_radius(operator_matrix: np.ndarray) -> float:
        """
        Spectral radius: ρ(T) = max{|λ| : λ ∈ σ(T)}
        """
        eigenvalues = SpectralAnalyzer.spectrum(operator_matrix)
        return max(abs(λ) for λ in eigenvalues)


# ============================================================================
# Sobolev Spaces
# ============================================================================

class SobolevSpace:
    """
    Sobolev spaces Wᵏ'ᵖ: Functions with weak derivatives in Lᵖ.

    Essential for PDEs and neural operators.
    W¹'²(Ω) = H¹(Ω) is most common (Hilbert space).
    """

    def __init__(self, order: int = 1, p: float = 2.0, domain_dim: int = 1):
        """
        Initialize Sobolev space.

        Args:
            order: Derivative order k
            p: Lᵖ exponent (p=2 gives Hilbert space)
            domain_dim: Spatial dimension
        """
        self.order = order
        self.p = p
        self.domain_dim = domain_dim

        logger.info(f"Sobolev space W^{order},{p}(R^{domain_dim})")

    def sobolev_norm(self, function_values: np.ndarray,
                     derivatives: List[np.ndarray],
                     grid_spacing: float = 1.0) -> float:
        """
        Compute Sobolev norm ||u||_{W^{k,p}}.

        ||u||²_{W^{k,2}} = Σ_{|α|≤k} ∫|D^α u|² dx
        """
        # L² norm of function
        norm_squared = np.sum(np.abs(function_values) ** self.p) * (grid_spacing ** self.domain_dim)

        # Add norms of derivatives
        for deriv in derivatives[:self.order]:
            norm_squared += np.sum(np.abs(deriv) ** self.p) * (grid_spacing ** self.domain_dim)

        return norm_squared ** (1 / self.p)

    def weak_derivative(self,
                       function_values: np.ndarray,
                       test_functions: List[np.ndarray],
                       grid_spacing: float = 1.0) -> np.ndarray:
        """
        Compute weak derivative via integration by parts.

        ⟨D^α u, φ⟩ = (-1)^|α| ⟨u, D^α φ⟩
        """
        # Simplified 1D case
        # Weak derivative: ∫ u·φ' dx = -∫ u'·φ dx

        # Numerical derivative of function
        weak_deriv = -np.gradient(function_values, grid_spacing)

        return weak_deriv

    def embedding_inequality(self, u_sobolev_norm: float) -> float:
        """
        Sobolev embedding: W^{k,p} ↪ C^m for appropriate k,p,m.

        Returns bound on supremum norm.
        """
        # Simplified: W^{1,2}(Ω) ↪ L^{2*}(Ω) where 2* = 2n/(n-2)
        # For 1D: W^{1,2} ↪ C^0 (continuous functions)

        # Sobolev inequality constant depends on domain
        C_sobolev = 1.0  # Simplified

        return C_sobolev * u_sobolev_norm


# ============================================================================
# Compact Operators
# ============================================================================

class CompactOperator:
    """
    Compact operator: Maps bounded sets to relatively compact sets.

    Generalization of finite rank operators.
    Spectral theory particularly nice for compact self-adjoint operators.
    """

    @staticmethod
    def is_compact(operator_matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if operator is compact (finite rank).

        For finite-dimensional spaces, every operator is compact.
        Check numerical rank.
        """
        singular_values = np.linalg.svd(operator_matrix, compute_uv=False)

        # Count non-zero singular values
        rank = np.sum(singular_values > tolerance)

        # Finite rank ⟹ compact
        return rank < min(operator_matrix.shape)

    @staticmethod
    def singular_value_decomposition(operator_matrix: np.ndarray) -> Tuple:
        """
        SVD: T = UΣV* for compact operators.

        Returns (U, singular_values, Vᵀ)
        """
        return np.linalg.svd(operator_matrix)

    @staticmethod
    def nuclear_norm(operator_matrix: np.ndarray) -> float:
        """
        Nuclear norm (trace norm): ||T||₁ = Σᵢ σᵢ.

        Sum of singular values.
        """
        singular_values = np.linalg.svd(operator_matrix, compute_uv=False)
        return np.sum(singular_values)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Functional Analysis Demo")
    print("="*80 + "\n")

    # 1. Hilbert Space
    print("1. Hilbert Space Operations")
    print("-" * 40)

    # Create vectors in R^3
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0])
    ]

    H = HilbertSpace(elements=vectors, name="R^3")

    # Gram-Schmidt
    orthonormal = H.gram_schmidt(vectors)
    print(f"Orthonormal basis computed: {len(orthonormal)} vectors")

    # Check orthonormality
    for i, e_i in enumerate(orthonormal):
        for j, e_j in enumerate(orthonormal):
            dot = H.inner_product(e_i, e_j)
            expected = 1.0 if i == j else 0.0
            print(f"  <e{i}, e{j}> = {dot:.6f} (expected: {expected})")

    # 2. Bounded Operators
    print("\n2. Bounded Linear Operators")
    print("-" * 40)

    # Matrix operator
    A = np.array([[2.0, 1.0], [1.0, 2.0]])

    def matrix_operator(x):
        return A @ x

    vectors_2d = [np.random.randn(2) for _ in range(50)]
    H_2d = HilbertSpace(elements=vectors_2d, name="R^2")

    T = BoundedOperator(matrix_operator, H_2d, H_2d, name="A")

    norm_T = T.operator_norm()
    print(f"Operator norm ||A||: {norm_T:.6f}")

    is_self_adjoint = T.is_self_adjoint()
    print(f"Self-adjoint: {is_self_adjoint}")

    # 3. Spectral Theory
    print("\n3. Spectral Decomposition")
    print("-" * 40)

    eigenvalues, eigenvectors = SpectralAnalyzer.eigendecomposition(A)
    print(f"Eigenvalues: {eigenvalues}")

    spectral_radius = SpectralAnalyzer.spectral_radius(A)
    print(f"Spectral radius: {spectral_radius:.6f}")

    # Verify spectral decomposition
    reconstructed = sum(λ * np.outer(v, v)
                       for λ, v in zip(eigenvalues, eigenvectors.T))
    print(f"Reconstruction error: {np.linalg.norm(A - reconstructed):.2e}")

    # 4. Sobolev Spaces
    print("\n4. Sobolev Space")
    print("-" * 40)

    W = SobolevSpace(order=1, p=2.0, domain_dim=1)

    # Sample function on grid
    x = np.linspace(0, 1, 100)
    u = np.sin(2 * np.pi * x)
    u_prime = np.gradient(u, x[1] - x[0])

    sobolev_norm = W.sobolev_norm(u, [u_prime], grid_spacing=x[1]-x[0])
    print(f"Sobolev norm ||u||_W^1,2: {sobolev_norm:.6f}")

    # 5. Compact Operators
    print("\n5. Compact Operators")
    print("-" * 40)

    # Low-rank operator
    B = np.array([[1.0, 0.0], [0.0, 0.0]])  # Rank 1

    is_compact = CompactOperator.is_compact(B)
    print(f"Operator is compact: {is_compact}")

    nuclear_norm = CompactOperator.nuclear_norm(B)
    print(f"Nuclear norm: {nuclear_norm:.6f}")

    U, s, Vt = CompactOperator.singular_value_decomposition(B)
    print(f"Singular values: {s}")

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)
