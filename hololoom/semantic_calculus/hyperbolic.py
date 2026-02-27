"""
Hyperbolic Semantic Space with Complex Embeddings

Combines three powerful mathematical structures:
1. Hyperbolic Geometry (Poincaré ball) - for hierarchies
2. Complex Embeddings (C^n) - for phase/orientation
3. Representation Theory - for symmetries

Key insights:
- Semantic space is naturally HIERARCHICAL → hyperbolic!
- Meanings have ORIENTATION (framing) → complex coordinates!
- Symmetries reveal INVARIANT structure → representation theory!

Mathematical Framework:
- Poincaré ball: B^n = {z ∈ C^n : |z| < 1}
- Hyperbolic metric: ds² = 4/(1-|z|²)² |dz|²
- Group actions: G acts on B^n via Möbius transformations
- Invariants: quantities preserved under symmetry

This is the deepest level of semantic geometry.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class HyperbolicPoint:
    """
    Point in Poincaré ball with complex coordinates

    z: Complex coordinates (n_dims,)
    norm: |z| < 1 (must be in unit ball)
    """
    z: np.ndarray  # complex array

    def __post_init__(self):
        """Ensure point is in Poincaré ball"""
        self.norm = np.linalg.norm(self.z)
        if self.norm >= 1.0:
            # Project to boundary
            self.z = 0.99 * self.z / self.norm
            self.norm = 0.99

    @property
    def magnitude(self) -> np.ndarray:
        """Real magnitude of each component"""
        return np.abs(self.z)

    @property
    def phase(self) -> np.ndarray:
        """Phase angle of each component"""
        return np.angle(self.z)

    def to_euclidean(self) -> np.ndarray:
        """
        Convert to Euclidean coordinates (for visualization)

        Flatten complex → real: [real parts, imaginary parts]
        """
        return np.concatenate([self.z.real, self.z.imag])


class PoincareGeometry:
    """
    Hyperbolic geometry in Poincaré ball model

    Key operations:
    - Hyperbolic distance
    - Geodesics (hyperbolic lines)
    - Parallel transport
    - Exponential/logarithmic maps
    """

    @staticmethod
    def hyperbolic_distance(z1: np.ndarray, z2: np.ndarray) -> float:
        """
        Hyperbolic distance in Poincaré ball

        d(z1, z2) = arcosh(1 + 2*|z1-z2|² / ((1-|z1|²)(1-|z2|²)))

        This grows exponentially as points approach boundary!
        """
        diff = z1 - z2
        norm_diff_sq = np.sum(np.abs(diff)**2)

        norm_z1_sq = np.sum(np.abs(z1)**2)
        norm_z2_sq = np.sum(np.abs(z2)**2)

        # Avoid division by zero
        denom = (1 - norm_z1_sq) * (1 - norm_z2_sq)
        if denom < 1e-10:
            return np.inf

        # Hyperbolic distance formula
        cosh_d = 1 + 2 * norm_diff_sq / denom

        # Numerical stability
        if cosh_d < 1:
            cosh_d = 1

        d_hyp = np.arccosh(cosh_d)
        return d_hyp

    @staticmethod
    def mobius_add(z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Möbius addition (hyperbolic translation)

        z ⊕ w = (z + w) / (1 + z̄w)

        This is the group operation in hyperbolic space!
        """
        numerator = z + w
        denominator = 1 + np.sum(np.conj(z) * w)

        # Clip to stay in ball
        result = numerator / denominator
        norm = np.linalg.norm(result)
        if norm >= 1:
            result = 0.99 * result / norm

        return result

    @staticmethod
    def exponential_map(z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: tangent vector → point on manifold

        Exp_z(v) = z ⊕ (tanh(λ_z |v|/2) * v/|v|)
        where λ_z = 2/(1-|z|²) is conformal factor

        This moves along geodesic starting at z in direction v
        """
        lambda_z = 2 / (1 - np.sum(np.abs(z)**2))
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-10:
            return z

        # Geodesic step
        v_normalized = v / v_norm
        step_size = np.tanh(lambda_z * v_norm / 2)
        step = step_size * v_normalized

        return PoincareGeometry.mobius_add(z, step)

    @staticmethod
    def logarithmic_map(z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: point → tangent vector

        Log_z(w) = inverse of Exp_z

        Returns tangent vector at z that points toward w
        """
        # Möbius subtraction
        diff = PoincareGeometry.mobius_add(w, -z)

        lambda_z = 2 / (1 - np.sum(np.abs(z)**2))
        diff_norm = np.linalg.norm(diff)

        if diff_norm < 1e-10:
            return np.zeros_like(z)

        # Inverse of tanh
        v_norm = 2 / lambda_z * np.arctanh(diff_norm)
        v = v_norm * diff / diff_norm

        return v


class HyperbolicSemanticSpace:
    """
    Semantic space with hyperbolic geometry

    Hierarchy naturally encoded:
    - General concepts near center (r ≈ 0)
    - Specific concepts near boundary (r → 1)
    - Distance grows exponentially with specificity
    """

    def __init__(self, n_dims: int = 8):
        """
        Args:
            n_dims: Number of complex dimensions (16 real dims total)
        """
        self.n_dims = n_dims
        self.geometry = PoincareGeometry()

    def embed_hierarchy(self, concepts: List[str],
                       parent_child_pairs: List[Tuple[str, str]],
                       embed_fn: Callable) -> dict:
        """
        Embed concepts in hyperbolic space respecting hierarchy

        More general concepts → center
        More specific concepts → boundary

        Args:
            concepts: List of concept names
            parent_child_pairs: (parent, child) tuples defining hierarchy
            embed_fn: Euclidean embedding function

        Returns:
            {concept_name: HyperbolicPoint}
        """
        # Build hierarchy tree
        children = {c: [] for c in concepts}
        parents = {}

        for parent, child in parent_child_pairs:
            children[parent].append(child)
            parents[child] = parent

        # Find root (no parent)
        roots = [c for c in concepts if c not in parents]

        # Compute depth for each concept
        depths = {}

        def compute_depth(concept, current_depth=0):
            depths[concept] = current_depth
            for child in children[concept]:
                compute_depth(child, current_depth + 1)

        for root in roots:
            compute_depth(root)

        max_depth = max(depths.values()) if depths else 1

        # Embed in hyperbolic space
        hyperbolic_embeddings = {}

        for concept in concepts:
            # Get Euclidean embedding
            vec = embed_fn(concept)

            # Convert to complex (first n_dims components)
            z_euclidean = vec[:self.n_dims] + 1j * vec[self.n_dims:2*self.n_dims]

            # Scale by depth (deeper = closer to boundary)
            depth = depths.get(concept, 0)
            radius = 0.05 + 0.90 * (depth / max_depth)  # 0.05 to 0.95

            # Normalize and scale
            z_norm = np.linalg.norm(z_euclidean)
            if z_norm > 1e-10:
                z_hyperbolic = radius * z_euclidean / z_norm
            else:
                z_hyperbolic = radius * np.ones(self.n_dims, dtype=complex) / np.sqrt(self.n_dims)

            hyperbolic_embeddings[concept] = HyperbolicPoint(z_hyperbolic)

        return hyperbolic_embeddings

    def hierarchical_distance(self, concept1: HyperbolicPoint,
                             concept2: HyperbolicPoint) -> dict:
        """
        Compute distances accounting for hierarchy

        Returns both Euclidean and hyperbolic distances
        """
        # Euclidean distance
        euclidean_dist = np.linalg.norm(concept1.z - concept2.z)

        # Hyperbolic distance (respects hierarchy!)
        hyperbolic_dist = self.geometry.hyperbolic_distance(concept1.z, concept2.z)

        # Specificity difference (depth in hierarchy)
        specificity_diff = abs(concept1.norm - concept2.norm)

        return {
            'euclidean': euclidean_dist,
            'hyperbolic': hyperbolic_dist,
            'specificity_diff': specificity_diff,
            'ratio': hyperbolic_dist / (euclidean_dist + 1e-10)
        }


class ComplexSemanticFlow:
    """
    Semantic flow with complex coordinates

    dz/dt = v_real + i*v_imag

    Real part: change in magnitude (intensity)
    Imaginary part: rotational flow (reframing)
    """

    def __init__(self, n_dims: int = 8):
        self.n_dims = n_dims

    def compute_complex_velocity(self, trajectory_complex: np.ndarray,
                                 dt: float = 1.0) -> np.ndarray:
        """
        Compute complex velocity from trajectory

        Args:
            trajectory_complex: (n_steps, n_dims) complex array

        Returns:
            Complex velocity array (n_steps, n_dims)
        """
        velocity = np.gradient(trajectory_complex, dt, axis=0)
        return velocity

    def decompose_flow(self, velocity_complex: np.ndarray) -> dict:
        """
        Decompose complex velocity into radial and rotational components

        v = v_r (radial) + i*v_θ (rotational)

        Args:
            velocity_complex: Complex velocity

        Returns:
            {
                'radial': change in magnitude,
                'rotational': change in phase (reframing),
                'total_magnitude': |v|,
                'flow_angle': arg(v)
            }
        """
        # Radial component (real part of v in polar form)
        radial = velocity_complex.real

        # Rotational component (imaginary part = phase change)
        rotational = velocity_complex.imag

        # Total flow magnitude
        magnitude = np.abs(velocity_complex)

        # Flow angle (direction in complex plane)
        angle = np.angle(velocity_complex)

        return {
            'radial': radial,
            'rotational': rotational,
            'magnitude': magnitude,
            'angle': angle
        }


class SemanticSymmetryGroup:
    """
    Representation theory: symmetries of semantic space

    Group operations:
    - Translation (topic shift)
    - Rotation (perspective change)
    - Scaling (intensity change)
    - Reflection (negation)
    """

    @staticmethod
    def translate(z: np.ndarray, direction: np.ndarray, distance: float) -> np.ndarray:
        """
        Translate in semantic space (hyperbolic translation)
        """
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-10)
        step = distance * direction_normalized
        return PoincareGeometry.mobius_add(z, step)

    @staticmethod
    def rotate(z: np.ndarray, angle: float, axis_idx: int = 0) -> np.ndarray:
        """
        Rotate around an axis (complex phase rotation)

        Multiply by e^(iθ) in specified dimension
        """
        z_rotated = z.copy()
        z_rotated[axis_idx] *= np.exp(1j * angle)
        return z_rotated

    @staticmethod
    def scale(z: np.ndarray, factor: float) -> np.ndarray:
        """
        Scale magnitude (intensity change)

        Preserves phase, changes magnitude
        """
        phases = np.angle(z)
        magnitudes = np.abs(z) * factor

        # Clip to stay in Poincaré ball
        magnitudes = np.minimum(magnitudes, 0.99)

        z_scaled = magnitudes * np.exp(1j * phases)
        return z_scaled

    @staticmethod
    def reflect(z: np.ndarray, axis_idx: int = 0) -> np.ndarray:
        """
        Reflect across hyperplane (semantic negation)
        """
        z_reflected = z.copy()
        z_reflected[axis_idx] = -z_reflected[axis_idx].conj()
        return z_reflected

    @staticmethod
    def compute_invariant(z: np.ndarray) -> float:
        """
        Compute group invariant (objective meaning)

        Invariant under rotation and scaling → pure structural property
        """
        # Use hyperbolic norm (conformal invariant)
        norm = np.linalg.norm(z)
        invariant = np.arctanh(norm)  # hyperbolic radius
        return invariant


def visualize_hyperbolic_hierarchy(embeddings: dict,
                                   parent_child_pairs: List[Tuple[str, str]],
                                   save_path: Optional[str] = None):
    """
    Visualize semantic hierarchy in 2D Poincaré disk
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw Poincaré disk boundary
    boundary = Circle((0, 0), 1.0, fill=False, edgecolor='black',
                     linewidth=2, linestyle='--')
    ax.add_patch(boundary)

    # Plot concepts (use first 2 dimensions)
    for concept, point in embeddings.items():
        z = point.z[:2]  # first 2 complex dims
        x = z.real
        y = z.imag

        # Color by radius (depth in hierarchy)
        radius = np.sqrt(np.sum(x**2 + y**2))
        color = plt.cm.viridis(radius)

        ax.scatter(x, y, c=[color], s=200, alpha=0.8, edgecolors='white', linewidths=2)
        ax.text(x[0], y[0], f'  {concept}', fontsize=10, fontweight='bold')

    # Draw hierarchy edges
    for parent, child in parent_child_pairs:
        if parent in embeddings and child in embeddings:
            z_parent = embeddings[parent].z[:2]
            z_child = embeddings[child].z[:2]

            ax.plot([z_parent[0].real, z_child[0].real],
                   [z_parent[1].imag, z_child[1].imag],
                   'gray', linewidth=1.5, alpha=0.5)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('Hyperbolic Semantic Hierarchy (Poincaré Disk)',
                fontsize=14, fontweight='bold')

    # Add legend explaining radius
    ax.text(0.02, 0.98, 'Radius = Specificity\nCenter = General\nBoundary = Specific',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved hyperbolic hierarchy: {save_path}")

    return fig, ax