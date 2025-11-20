"""
Integral Geometry for Semantic Space

Measures geometric quantities by integrating over families of subspaces.
This is the third pillar alongside differential geometry and geometric integration:

1. DIFFERENTIAL GEOMETRY: Local properties (curvature at a point)
2. GEOMETRIC INTEGRATION: Following flows (trajectories)
3. INTEGRAL GEOMETRY: Global properties (averaging over all views)

Key concepts:
- Radon Transform: Project semantic fields onto lower dimensions
- Inverse Radon: Reconstruct from projections (tomography!)
- Crofton Formulas: Measure perimeter by counting line intersections
- Kinematic Formulas: Relate curvature to motion-averaged intersections

Applications:
- Reconstruct semantic potential V(q) from trajectories in many contexts
- Define coordinate-free measures of complexity
- Average meaning across all interpretations
- Tomographic reconstruction of ethical landscape

This is how we recover FULL semantic structure from partial observations.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


class RadonTransform:
    """
    Radon transform for semantic fields

    Projects a function V: R^n → R onto lower-dimensional subspaces
    by integrating along lines/planes.

    This is the mathematical foundation of CT scans, applied to semantic space!
    """

    def __init__(self, n_dims: int):
        """
        Args:
            n_dims: Dimensionality of semantic space (typically 16)
        """
        self.n_dims = n_dims

    def project_along_line(self, field: Callable[[np.ndarray], float],
                          direction: np.ndarray,
                          offset: float,
                          bounds: Tuple[float, float] = (-2.0, 2.0),
                          n_samples: int = 100) -> float:
        """
        Integrate field along a line

        Line parameterized as: q(t) = offset * direction_perp + t * direction

        Args:
            field: Function V(q) to integrate
            direction: Line direction (normalized)
            offset: Perpendicular offset from origin
            bounds: Integration bounds along line
            n_samples: Number of quadrature points

        Returns:
            Integral of V along the line
        """
        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Find perpendicular direction (simplified for 2D projection)
        if self.n_dims == 2:
            direction_perp = np.array([-direction[1], direction[0]])
        else:
            # For higher dimensions, use Gram-Schmidt
            direction_perp = self._get_perpendicular(direction)

        # Sample points along line
        t_values = np.linspace(bounds[0], bounds[1], n_samples)
        dt = (bounds[1] - bounds[0]) / n_samples

        # Integrate using trapezoidal rule
        integral = 0.0
        for t in t_values:
            q = offset * direction_perp + t * direction
            integral += field(q) * dt

        return integral

    def compute_sinogram(self, field: Callable[[np.ndarray], float],
                        n_angles: int = 180,
                        n_offsets: int = 100,
                        max_offset: float = 2.0) -> np.ndarray:
        """
        Compute full Radon transform (sinogram)

        For each angle θ, integrate along parallel lines at various offsets.
        This creates the "sinogram" - input for tomographic reconstruction.

        Args:
            field: Semantic potential V(q)
            n_angles: Number of projection angles
            n_offsets: Number of parallel lines per angle
            max_offset: Maximum perpendicular offset

        Returns:
            Sinogram array (n_angles, n_offsets)
        """
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        offsets = np.linspace(-max_offset, max_offset, n_offsets)

        sinogram = np.zeros((n_angles, n_offsets))

        for i, theta in enumerate(angles):
            # Direction for this angle
            if self.n_dims == 2:
                direction = np.array([np.cos(theta), np.sin(theta)])
            else:
                # For higher dims, rotate in first 2D plane
                direction = np.zeros(self.n_dims)
                direction[0] = np.cos(theta)
                direction[1] = np.sin(theta)

            for j, offset in enumerate(offsets):
                sinogram[i, j] = self.project_along_line(field, direction, offset)

        return sinogram

    def _get_perpendicular(self, v: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to v using Gram-Schmidt"""
        # Start with standard basis vector least aligned with v
        basis = np.eye(len(v))
        dots = np.abs(v @ basis)
        least_aligned_idx = np.argmin(dots)

        # Gram-Schmidt orthogonalization
        u = basis[least_aligned_idx]
        perp = u - np.dot(u, v) * v
        perp = perp / (np.linalg.norm(perp) + 1e-10)

        return perp


class InverseRadonTransform:
    """
    Inverse Radon transform - reconstruct field from projections

    This is filtered backprojection: the standard algorithm for CT reconstruction.
    We use it to reconstruct semantic potential V(q) from observed trajectories!
    """

    def __init__(self, n_dims: int = 2):
        """
        Args:
            n_dims: Dimensionality (2D for visualization, 16D for full semantic)
        """
        self.n_dims = n_dims

    def reconstruct_2d(self, sinogram: np.ndarray,
                      output_size: int = 100) -> np.ndarray:
        """
        Reconstruct 2D field from sinogram using filtered backprojection

        Args:
            sinogram: Radon transform (n_angles, n_offsets)
            output_size: Size of output image

        Returns:
            Reconstructed field (output_size, output_size)
        """
        n_angles, n_offsets = sinogram.shape

        # Create output grid
        x = np.linspace(-2, 2, output_size)
        y = np.linspace(-2, 2, output_size)
        X, Y = np.meshgrid(x, y)

        # Initialize reconstruction
        reconstruction = np.zeros((output_size, output_size))

        # Angles
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)

        # Offsets
        offsets = np.linspace(-2, 2, n_offsets)

        # Filter sinogram (ramp filter in Fourier space)
        filtered_sinogram = self._ramp_filter(sinogram)

        # Backprojection
        for i, theta in enumerate(angles):
            # Compute offset for each point in output grid
            point_offsets = X * np.cos(theta) + Y * np.sin(theta)

            # Interpolate filtered projection at these offsets
            projection_values = np.interp(
                point_offsets.ravel(),
                offsets,
                filtered_sinogram[i, :],
                left=0, right=0
            ).reshape(output_size, output_size)

            # Accumulate backprojection
            reconstruction += projection_values

        # Normalize
        reconstruction *= np.pi / n_angles

        return reconstruction

    def _ramp_filter(self, sinogram: np.ndarray) -> np.ndarray:
        """
        Apply ramp filter in Fourier domain

        This is the "filtered" part of filtered backprojection
        """
        n_angles, n_offsets = sinogram.shape

        # Fourier transform along offset axis
        sinogram_fft = np.fft.fft(sinogram, axis=1)

        # Create ramp filter
        freq = np.fft.fftfreq(n_offsets)
        ramp = np.abs(freq)

        # Apply filter
        filtered_fft = sinogram_fft * ramp[np.newaxis, :]

        # Inverse FFT
        filtered_sinogram = np.fft.ifft(filtered_fft, axis=1).real

        return filtered_sinogram


class CroftonFormula:
    """
    Crofton-type formulas for measuring geometric quantities

    Key idea: Measure perimeter/area/curvature by counting intersections
    with random lines/curves.

    Applications:
    - Semantic complexity = average intersections with test curves
    - Boundary detection = where intersection count spikes
    - Intrinsic curvature = intersection counts with moving frames
    """

    @staticmethod
    def estimate_perimeter(region_indicator: Callable[[np.ndarray], bool],
                          n_lines: int = 1000,
                          domain_radius: float = 2.0) -> float:
        """
        Estimate perimeter of a region using Crofton formula

        Crofton Formula: Perimeter = (1/2) * E[number of intersections]
        where expectation is over uniformly random lines.

        Args:
            region_indicator: Function returns True if point in region
            n_lines: Number of random lines to sample
            domain_radius: Size of domain

        Returns:
            Estimated perimeter
        """
        intersection_counts = []

        for _ in range(n_lines):
            # Random line: point + direction
            point = np.random.uniform(-domain_radius, domain_radius, size=2)
            direction = np.random.randn(2)
            direction = direction / np.linalg.norm(direction)

            # Count crossings along this line
            t_values = np.linspace(-domain_radius * 2, domain_radius * 2, 500)
            in_region = [region_indicator(point + t * direction) for t in t_values]

            # Count transitions
            crossings = np.sum(np.diff(np.array(in_region, dtype=int)) != 0)
            intersection_counts.append(crossings)

        # Crofton formula
        avg_intersections = np.mean(intersection_counts)
        perimeter = 0.5 * avg_intersections * (4 * domain_radius) / 500

        return perimeter

    @staticmethod
    def semantic_complexity(trajectory: np.ndarray,
                           n_test_curves: int = 100) -> float:
        """
        Measure complexity of semantic trajectory using intersection counts

        Higher complexity = more intersections with random test curves

        Args:
            trajectory: Semantic path (n_steps, n_dims)
            n_test_curves: Number of random curves to test

        Returns:
            Complexity score
        """
        n_steps, n_dims = trajectory.shape

        total_intersections = 0

        for _ in range(n_test_curves):
            # Generate random test curve (smooth random walk)
            test_curve = np.cumsum(np.random.randn(n_steps, n_dims) * 0.1, axis=0)

            # Count approximate intersections (nearby points)
            for i in range(n_steps):
                distances = np.linalg.norm(test_curve - trajectory[i], axis=1)
                nearby = np.sum(distances < 0.1)
                total_intersections += nearby

        complexity = total_intersections / n_test_curves
        return complexity


class SemanticTomography:
    """
    Tomographic reconstruction of semantic potential from multiple contexts

    Key insight: Observing text in MANY contexts = multiple "projections"
    Use inverse Radon transform to reconstruct full V(q)!
    """

    def __init__(self, n_dims: int = 16):
        self.n_dims = n_dims
        self.radon = RadonTransform(n_dims)
        self.inverse_radon = InverseRadonTransform(n_dims=2)  # 2D for visualization

    def collect_projections(self, word: str,
                           contexts: List[str],
                           embed_fn: Callable,
                           spectrum) -> List[np.ndarray]:
        """
        Observe word in multiple contexts to get different "views"

        Each context gives us a projection of the semantic potential

        Args:
            word: Target word to analyze
            contexts: Different contexts to observe word in
            embed_fn: Embedding function
            spectrum: SemanticSpectrum for projection

        Returns:
            List of semantic projections (one per context)
        """
        projections = []

        for context in contexts:
            # Embed word in this context
            full_phrase = f"{context} {word}"
            vec = embed_fn(full_phrase)

            # Project to semantic space
            q_sem = spectrum.project_vector(vec)
            q_array = np.array([q_sem[d.name] for d in spectrum.dimensions])

            projections.append(q_array)

        return projections

    def reconstruct_semantic_field(self, trajectories: List[np.ndarray],
                                  grid_size: int = 50) -> np.ndarray:
        """
        Reconstruct 2D semantic potential field from multiple trajectories

        Each trajectory is like a "scan line" through semantic space.
        By combining many trajectories from different angles, we can
        reconstruct the full potential landscape.

        Args:
            trajectories: List of semantic trajectories from different contexts
            grid_size: Resolution of output field

        Returns:
            Reconstructed potential field (grid_size, grid_size)
        """
        # For simplicity, work in 2D (project 16D to 2D first)
        # In practice, would use higher-dimensional reconstruction

        # Project all trajectories to 2D using PCA
        from sklearn.decomposition import PCA

        all_points = np.vstack(trajectories)
        pca = PCA(n_components=2)
        all_points_2d = pca.fit_transform(all_points)

        # Split back into trajectories
        trajectories_2d = []
        idx = 0
        for traj in trajectories:
            n_points = len(traj)
            trajectories_2d.append(all_points_2d[idx:idx+n_points])
            idx += n_points

        # Estimate potential from trajectories (assume gradient descent)
        # V(q) ~ -∫ (velocity · direction) along path

        # Create grid
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        X, Y = np.meshgrid(x, y)

        # Initialize field
        V_field = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))

        # Accumulate contributions from each trajectory
        for traj_2d in trajectories_2d:
            velocities = np.gradient(traj_2d, axis=0)

            for i, (pos, vel) in enumerate(zip(traj_2d, velocities)):
                # Find nearest grid point
                ix = np.argmin(np.abs(x - pos[0]))
                iy = np.argmin(np.abs(y - pos[1]))

                # Accumulate potential (inversely related to velocity magnitude)
                speed = np.linalg.norm(vel)
                if speed > 1e-6:
                    V_field[iy, ix] += 1.0 / (speed + 0.1)
                    counts[iy, ix] += 1

        # Average
        V_field = np.where(counts > 0, V_field / (counts + 1e-10), 0)

        # Smooth
        V_field = gaussian_filter(V_field, sigma=2.0)

        return V_field


def visualize_tomographic_reconstruction(V_field: np.ndarray,
                                        trajectories_2d: List[np.ndarray],
                                        save_path: Optional[str] = None):
    """
    Visualize reconstructed semantic potential with overlaid trajectories
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot potential field
    extent = [-2, 2, -2, 2]
    im = ax.imshow(V_field, extent=extent, origin='lower',
                   cmap='viridis', aspect='auto', alpha=0.8)

    # Overlay trajectories
    for traj in trajectories_2d:
        ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2, alpha=0.6)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100,
                  marker='o', zorder=5, edgecolors='white', linewidths=2)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100,
                  marker='X', zorder=5, edgecolors='white', linewidths=2)

    plt.colorbar(im, ax=ax, label='Semantic Potential V(q)')
    ax.set_xlabel('Semantic Dimension 1', fontsize=12)
    ax.set_ylabel('Semantic Dimension 2', fontsize=12)
    ax.set_title('Tomographic Reconstruction of Semantic Potential',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved tomographic reconstruction: {save_path}")

    return fig, ax
