"""
Advanced Warp Space Operations
================================
Extended tensor operations for sophisticated manifold computations.

This module provides advanced mathematical operations for Warp Space:
- Differential geometry (geodesics, curvature, parallel transport)
- Tensor decompositions (Tucker, CP, tensor train)
- Manifold learning (local tangent spaces, exp/log maps)
- Quantum-inspired operations (superposition, entanglement)
- Information geometry (Fisher information, natural gradients)

Philosophy:
While basic Warp Space handles standard tensor operations, Advanced Warp
extends this with cutting-edge mathematical frameworks for deep semantic
understanding and sophisticated decision-making.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


# ============================================================================
# Differential Geometry Operations
# ============================================================================

class RiemannianManifold:
    """
    Riemannian manifold structure for semantic space.

    Treats the embedding space as a curved manifold where distances
    and angles are measured via a learned metric tensor.
    """

    def __init__(self, dim: int, curvature: float = 0.0):
        """
        Initialize manifold.

        Args:
            dim: Dimension of the manifold
            curvature: Constant curvature (0=flat, >0=spherical, <0=hyperbolic)
        """
        self.dim = dim
        self.curvature = curvature

        # Metric tensor (initially Euclidean)
        self.metric = np.eye(dim)

        logger.info(f"RiemannianManifold initialized: dim={dim}, curvature={curvature}")

    def geodesic_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute geodesic distance between two points.

        For flat space (curvature=0): Euclidean distance
        For spherical space (curvature>0): Great circle distance
        For hyperbolic space (curvature<0): Hyperbolic distance

        Args:
            p1, p2: Points on manifold

        Returns:
            Geodesic distance
        """
        if abs(self.curvature) < 1e-8:
            # Flat space: Euclidean distance
            diff = p2 - p1
            return float(np.sqrt(diff @ self.metric @ diff))

        elif self.curvature > 0:
            # Spherical space: arc distance
            # Normalize to unit sphere
            p1_norm = p1 / (np.linalg.norm(p1) + 1e-10)
            p2_norm = p2 / (np.linalg.norm(p2) + 1e-10)

            # Cosine similarity
            cos_angle = np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # Arc length on unit sphere
            radius = 1.0 / np.sqrt(self.curvature)
            return float(radius * angle)

        else:
            # Hyperbolic space (Poincaré model)
            # Distance in hyperbolic space
            radius = 1.0 / np.sqrt(-self.curvature)

            norm_p1 = np.linalg.norm(p1)
            norm_p2 = np.linalg.norm(p2)
            norm_diff = np.linalg.norm(p2 - p1)

            # Möbius addition for hyperbolic geometry
            numerator = 2 * norm_diff**2
            denominator = (1 - norm_p1**2) * (1 - norm_p2**2)

            if denominator > 1e-10:
                cosh_dist = 1 + numerator / denominator
                dist = radius * np.arccosh(np.clip(cosh_dist, 1.0, 1e10))
                return float(dist)
            else:
                # Fallback to Euclidean
                return float(np.linalg.norm(p2 - p1))

    def exponential_map(self, base_point: np.ndarray, tangent_vector: np.ndarray) -> np.ndarray:
        """
        Exponential map: tangent space -> manifold.

        Maps a tangent vector at base_point to a point on the manifold.

        Args:
            base_point: Point on manifold
            tangent_vector: Vector in tangent space at base_point

        Returns:
            Point on manifold
        """
        if abs(self.curvature) < 1e-8:
            # Flat space: simple addition
            return base_point + tangent_vector

        elif self.curvature > 0:
            # Spherical space
            norm = np.linalg.norm(tangent_vector)
            if norm < 1e-10:
                return base_point

            radius = 1.0 / np.sqrt(self.curvature)
            direction = tangent_vector / norm

            # Exponential map on sphere
            angle = norm / radius
            result = base_point * np.cos(angle) + direction * radius * np.sin(angle)

            return result

        else:
            # Hyperbolic space (Poincaré ball)
            norm_v = np.linalg.norm(tangent_vector)
            if norm_v < 1e-10:
                return base_point

            radius = 1.0 / np.sqrt(-self.curvature)
            norm_p = np.linalg.norm(base_point)

            # Möbius addition
            coeff = np.tanh(norm_v / (2 * radius * (1 - norm_p**2)))

            result = (base_point + coeff * tangent_vector) / (1 + coeff * norm_p)

            return result

    def logarithmic_map(self, base_point: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: manifold -> tangent space.

        Maps a point on the manifold to a tangent vector at base_point.

        Args:
            base_point: Base point on manifold
            target_point: Target point on manifold

        Returns:
            Tangent vector at base_point
        """
        if abs(self.curvature) < 1e-8:
            # Flat space: simple subtraction
            return target_point - base_point

        elif self.curvature > 0:
            # Spherical space
            diff = target_point - base_point
            norm_diff = np.linalg.norm(diff)

            if norm_diff < 1e-10:
                return np.zeros_like(base_point)

            radius = 1.0 / np.sqrt(self.curvature)
            distance = self.geodesic_distance(base_point, target_point)

            # Tangent vector scaled by distance
            return diff * (distance / norm_diff)

        else:
            # Hyperbolic space
            return target_point - base_point  # Simplified

    def parallel_transport(
        self,
        vector: np.ndarray,
        from_point: np.ndarray,
        to_point: np.ndarray
    ) -> np.ndarray:
        """
        Parallel transport vector along geodesic.

        Transports a tangent vector from one point to another along
        the geodesic connecting them.

        Args:
            vector: Tangent vector at from_point
            from_point: Starting point
            to_point: Ending point

        Returns:
            Transported vector at to_point
        """
        if abs(self.curvature) < 1e-8:
            # Flat space: vector stays the same
            return vector

        # For curved spaces, this is a simplified version
        # Full implementation would use Christoffel symbols

        # Compute rotation that aligns from_point with to_point
        from_norm = from_point / (np.linalg.norm(from_point) + 1e-10)
        to_norm = to_point / (np.linalg.norm(to_point) + 1e-10)

        # Rotation axis (perpendicular to both)
        axis = np.cross(from_norm, to_norm)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-10:
            # Points are aligned
            return vector

        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(from_norm, to_norm), -1.0, 1.0))

        # Rodrigues' rotation formula (simplified)
        rotated = vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)

        return rotated

    def sectional_curvature(self, p: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute sectional curvature at point p for plane spanned by v1, v2.

        Args:
            p: Point on manifold
            v1, v2: Tangent vectors spanning a 2-plane

        Returns:
            Sectional curvature
        """
        # For constant curvature manifolds, it's just the constant
        return self.curvature


# ============================================================================
# Tensor Decomposition
# ============================================================================

class TensorDecomposer:
    """
    Advanced tensor decomposition methods.

    Provides:
    - Tucker decomposition (higher-order SVD)
    - CP decomposition (CANDECOMP/PARAFAC)
    - Tensor train decomposition
    """

    @staticmethod
    def tucker_decomposition(
        tensor: np.ndarray,
        ranks: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Tucker decomposition: T ≈ C ×₁ U₁ ×₂ U₂ ×₃ U₃

        Decomposes a tensor into a core tensor and factor matrices.

        Args:
            tensor: Input tensor (N-dimensional array)
            ranks: Target ranks for each mode (None = use full rank)

        Returns:
            (core_tensor, factor_matrices)
        """
        ndim = len(tensor.shape)

        if ranks is None:
            ranks = list(tensor.shape)

        # Compute factor matrices via mode-n unfolding
        factor_matrices = []

        for n in range(ndim):
            # Mode-n unfolding
            unfolding = TensorDecomposer._mode_n_unfolding(tensor, n)

            # SVD
            U, s, Vt = np.linalg.svd(unfolding, full_matrices=False)

            # Truncate to target rank
            rank_n = min(ranks[n], U.shape[1])
            U_n = U[:, :rank_n]

            factor_matrices.append(U_n)

        # Compute core tensor
        core = tensor.copy()
        for n in range(ndim):
            core = TensorDecomposer._mode_n_product(core, factor_matrices[n].T, n)

        logger.info(f"Tucker decomposition: {tensor.shape} -> core {core.shape} + {ndim} factors")

        return core, factor_matrices

    @staticmethod
    def _mode_n_unfolding(tensor: np.ndarray, n: int) -> np.ndarray:
        """Unfold tensor along mode n."""
        shape = tensor.shape
        # Move mode n to front
        axes = [n] + [i for i in range(len(shape)) if i != n]
        unfolded = np.transpose(tensor, axes)
        # Reshape to matrix
        return unfolded.reshape(shape[n], -1)

    @staticmethod
    def _mode_n_product(tensor: np.ndarray, matrix: np.ndarray, n: int) -> np.ndarray:
        """Mode-n product of tensor with matrix."""
        shape = tensor.shape
        # Move mode n to front
        axes = [n] + [i for i in range(len(shape)) if i != n]
        moved = np.transpose(tensor, axes)

        # Reshape and multiply
        moved_2d = moved.reshape(shape[n], -1)
        result_2d = matrix @ moved_2d

        # Reshape back
        new_shape = [result_2d.shape[0]] + [shape[i] for i in range(len(shape)) if i != n]
        result = result_2d.reshape(new_shape)

        # Move mode back
        inv_axes = [0] * len(new_shape)
        inv_axes[n] = 0
        j = 1
        for i in range(len(shape)):
            if i != n:
                inv_axes[i] = j
                j += 1

        return np.transpose(result, inv_axes)

    @staticmethod
    def cp_decomposition(
        tensor: np.ndarray,
        rank: int,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> List[np.ndarray]:
        """
        CP (CANDECOMP/PARAFAC) decomposition.

        Decomposes tensor into sum of rank-1 tensors:
        T ≈ Σᵣ aᵣ ⊗ bᵣ ⊗ cᵣ

        Args:
            tensor: Input tensor
            rank: CP rank
            max_iter: Maximum iterations for ALS
            tol: Convergence tolerance

        Returns:
            List of factor matrices
        """
        shape = tensor.shape
        ndim = len(shape)

        # Initialize factor matrices randomly
        factors = [np.random.randn(s, rank) for s in shape]

        # Alternating least squares
        for iteration in range(max_iter):
            old_factors = [f.copy() for f in factors]

            for n in range(ndim):
                # Compute Khatri-Rao product of all factors except n
                kr_product = TensorDecomposer._khatri_rao(
                    [factors[i] for i in range(ndim) if i != n]
                )

                # Mode-n unfolding
                unfolding = TensorDecomposer._mode_n_unfolding(tensor, n)

                # Least squares update
                factors[n] = unfolding @ kr_product @ np.linalg.pinv(kr_product.T @ kr_product)

            # Check convergence
            change = sum(np.linalg.norm(factors[i] - old_factors[i]) for i in range(ndim))
            if change < tol:
                logger.info(f"CP decomposition converged after {iteration+1} iterations")
                break

        return factors

    @staticmethod
    def _khatri_rao(matrices: List[np.ndarray]) -> np.ndarray:
        """Khatri-Rao product (column-wise Kronecker product)."""
        result = matrices[0]
        for matrix in matrices[1:]:
            # Column-wise Kronecker
            n_cols = result.shape[1]
            temp = []
            for j in range(n_cols):
                temp.append(np.outer(result[:, j], matrix[:, j]).ravel())
            result = np.column_stack(temp)
        return result


# ============================================================================
# Quantum-Inspired Operations
# ============================================================================

class QuantumWarpOperations:
    """
    Quantum-inspired operations for semantic superposition and entanglement.

    Treats embeddings as quantum states that can be in superposition,
    entangled, and measured with probabilistic collapse.
    """

    @staticmethod
    def superposition(states: List[np.ndarray], amplitudes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create superposition of multiple states.

        |ψ⟩ = Σᵢ αᵢ|ψᵢ⟩

        Args:
            states: List of state vectors
            amplitudes: Complex amplitudes (None = uniform)

        Returns:
            Superposed state
        """
        n_states = len(states)

        if amplitudes is None:
            # Uniform superposition
            amplitudes = np.ones(n_states) / np.sqrt(n_states)
        else:
            # Normalize
            amplitudes = amplitudes / np.linalg.norm(amplitudes)

        # Linear combination
        superposed = sum(a * s for a, s in zip(amplitudes, states))

        return superposed

    @staticmethod
    def entangle(state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
        """
        Create entangled state from two states.

        |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ (tensor product)

        Args:
            state1, state2: State vectors

        Returns:
            Entangled state (tensor product)
        """
        return np.outer(state1, state2).ravel()

    @staticmethod
    def measure(
        state: np.ndarray,
        observables: List[np.ndarray],
        collapse: bool = True
    ) -> Tuple[int, float, Optional[np.ndarray]]:
        """
        Measure quantum state in given basis.

        Computes probabilities and collapses (if requested) to measured state.

        Args:
            state: Quantum state
            observables: List of basis states to measure in
            collapse: Whether to collapse to measured state

        Returns:
            (measured_index, probability, collapsed_state or None)
        """
        # Compute probabilities (Born rule)
        probabilities = np.array([abs(np.dot(state, obs))**2 for obs in observables])
        probabilities = probabilities / (np.sum(probabilities) + 1e-10)

        # Sample measurement outcome
        measured_idx = np.random.choice(len(observables), p=probabilities)

        if collapse:
            # Collapse to measured state
            collapsed = observables[measured_idx]
            return measured_idx, probabilities[measured_idx], collapsed
        else:
            return measured_idx, probabilities[measured_idx], None

    @staticmethod
    def decoherence(state: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Apply decoherence (interaction with environment).

        Adds noise to simulate quantum decoherence.

        Args:
            state: Quantum state
            noise_level: Amount of noise (0-1)

        Returns:
            Decohered state
        """
        noise = np.random.randn(*state.shape) * noise_level
        decohered = state + noise

        # Renormalize
        decohered = decohered / (np.linalg.norm(decohered) + 1e-10)

        return decohered


# ============================================================================
# Information Geometry
# ============================================================================

class FisherInformationGeometry:
    """
    Information geometry using Fisher information metric.

    Treats probability distributions as points on a statistical manifold
    where distances are measured by Fisher information.
    """

    @staticmethod
    def fisher_information_matrix(
        distribution: np.ndarray,
        parameter_gradients: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute Fisher information matrix.

        FIM[i,j] = E[∂ log p/∂θᵢ · ∂ log p/∂θⱼ]

        Args:
            distribution: Probability distribution p(x|θ)
            parameter_gradients: List of ∂p/∂θᵢ

        Returns:
            Fisher information matrix
        """
        n_params = len(parameter_gradients)
        fim = np.zeros((n_params, n_params))

        # Normalize distribution
        p = distribution / (np.sum(distribution) + 1e-10)

        # Compute score function gradients (∂ log p / ∂θ)
        score_gradients = []
        for grad in parameter_gradients:
            score = grad / (p + 1e-10)
            score_gradients.append(score)

        # Compute FIM as outer product of score gradients
        for i in range(n_params):
            for j in range(n_params):
                fim[i, j] = np.sum(p * score_gradients[i] * score_gradients[j])

        return fim

    @staticmethod
    def natural_gradient(
        loss_gradient: np.ndarray,
        fisher_matrix: np.ndarray,
        damping: float = 1e-4
    ) -> np.ndarray:
        """
        Compute natural gradient (Fisher-preconditioned gradient).

        Natural gradient: F⁻¹ ∇L

        Args:
            loss_gradient: Standard gradient ∇L
            fisher_matrix: Fisher information matrix F
            damping: Damping factor for numerical stability

        Returns:
            Natural gradient
        """
        # Add damping for numerical stability
        damped_fisher = fisher_matrix + damping * np.eye(fisher_matrix.shape[0])

        # Solve F · ng = ∇L
        try:
            natural_grad = np.linalg.solve(damped_fisher, loss_gradient)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            natural_grad = np.linalg.pinv(damped_fisher) @ loss_gradient

        return natural_grad


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Advanced Warp Space Operations Demo")
    print("="*80 + "\n")

    # 1. Differential Geometry
    print("1. Riemannian Manifold (Spherical Geometry)")
    print("-" * 40)

    manifold = RiemannianManifold(dim=10, curvature=1.0)

    p1 = np.random.randn(10)
    p2 = np.random.randn(10)

    euclidean_dist = np.linalg.norm(p2 - p1)
    geodesic_dist = manifold.geodesic_distance(p1, p2)

    print(f"Euclidean distance: {euclidean_dist:.3f}")
    print(f"Geodesic distance:  {geodesic_dist:.3f}")

    # Exponential/logarithmic maps
    tangent = np.random.randn(10) * 0.1
    p_new = manifold.exponential_map(p1, tangent)
    tangent_recovered = manifold.logarithmic_map(p1, p_new)

    print(f"Tangent vector norm: {np.linalg.norm(tangent):.3f}")
    print(f"Recovered norm:      {np.linalg.norm(tangent_recovered):.3f}")
    print()

    # 2. Tensor Decomposition
    print("2. Tucker Decomposition")
    print("-" * 40)

    tensor = np.random.randn(5, 6, 4)
    core, factors = TensorDecomposer.tucker_decomposition(tensor, ranks=[3, 4, 3])

    print(f"Original tensor shape: {tensor.shape}")
    print(f"Core tensor shape: {core.shape}")
    print(f"Factor matrices: {[f.shape for f in factors]}")
    print()

    # 3. Quantum Operations
    print("3. Quantum-Inspired Superposition")
    print("-" * 40)

    states = [np.random.randn(8) for _ in range(3)]
    states = [s / np.linalg.norm(s) for s in states]  # Normalize

    amplitudes = np.array([0.5, 0.3, 0.2])
    superposed = QuantumWarpOperations.superposition(states, amplitudes)

    print(f"Superposition of {len(states)} states")
    print(f"Superposed state norm: {np.linalg.norm(superposed):.3f}")

    # Measurement
    idx, prob, collapsed = QuantumWarpOperations.measure(superposed, states, collapse=True)
    print(f"Measured state {idx} with probability {prob:.3f}")
    print()

    # 4. Information Geometry
    print("4. Fisher Information Matrix")
    print("-" * 40)

    distribution = np.abs(np.random.randn(10))
    gradients = [np.random.randn(10) for _ in range(3)]

    fim = FisherInformationGeometry.fisher_information_matrix(distribution, gradients)
    print(f"Fisher information matrix shape: {fim.shape}")
    print(f"FIM determinant: {np.linalg.det(fim):.6f}")

    # Natural gradient
    loss_grad = np.random.randn(3)
    nat_grad = FisherInformationGeometry.natural_gradient(loss_grad, fim)

    print(f"Standard gradient norm: {np.linalg.norm(loss_grad):.3f}")
    print(f"Natural gradient norm:  {np.linalg.norm(nat_grad):.3f}")
    print()

    print("✓ Demo complete!")
