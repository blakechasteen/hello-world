"""
Geometric Integration in Semantic Dimension Space

Implements structure-preserving integrators for semantic flows.
Uses symplectic methods to preserve:
- Hamiltonian energy
- Phase space volume
- Long-term stability

The key innovation: Project to 16D interpretable semantic space FIRST,
then integrate there. This gives us:
1. Speed: 16D vs 384D operations
2. Interpretability: "Warmth increasing, Formality decreasing"
3. Conservation: Energy and momentum preserved

This is the mathematical foundation for tracking semantic forces.
"""

import numpy as np
from typing import Callable, Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SemanticState:
    """
    State in semantic dimension space

    q: Position in semantic coordinates (16D)
    p: Momentum in semantic coordinates (16D)
    t: Time
    """
    q: np.ndarray  # semantic position
    p: np.ndarray  # semantic momentum
    t: float = 0.0

    @property
    def kinetic_energy(self, mass: float = 1.0) -> float:
        """T = (1/2m)||p||²"""
        return 0.5 * np.dot(self.p, self.p) / mass

    def hamiltonian(self, potential_fn: Callable, mass: float = 1.0) -> float:
        """H = T + V"""
        return self.kinetic_energy(mass) + potential_fn(self.q)


class GeometricIntegrator:
    """
    Symplectic integrator for semantic flows

    Uses Störmer-Verlet (leapfrog) method which preserves:
    - Symplectic structure: dq ∧ dp
    - Energy: H = T + V (approximately)
    - Time-reversibility
    """

    def __init__(self, projection_matrix: np.ndarray, mass: float = 1.0):
        """
        Args:
            projection_matrix: P matrix (n_dims, embedding_dim) that projects
                             full embeddings to semantic coordinates
            mass: Semantic "mass" for momentum
        """
        self.P = projection_matrix  # (16, 384)
        self.P_T = projection_matrix.T  # (384, 16) for lifting back
        self.mass = mass
        self.n_dims = projection_matrix.shape[0]

    def project_to_semantic(self, q_full: np.ndarray) -> np.ndarray:
        """
        Project full embedding to semantic coordinates
        q_semantic = P @ q_full
        """
        return self.P @ q_full

    def lift_to_full(self, q_semantic: np.ndarray) -> np.ndarray:
        """
        Lift semantic coordinates back to full space (approximate)
        q_full ≈ P.T @ q_semantic
        """
        return self.P_T @ q_semantic

    def project_gradient(self, grad_full: np.ndarray) -> np.ndarray:
        """
        Project gradient from full space to semantic space
        ∇V_semantic = P @ ∇V_full
        """
        return self.P @ grad_full

    def stormer_verlet_step(self, state: SemanticState,
                           gradient_fn: Callable[[np.ndarray], np.ndarray],
                           dt: float) -> SemanticState:
        """
        Störmer-Verlet integrator (symplectic, order 2)

        Also known as leapfrog or Verlet integration.
        Preserves symplectic structure exactly.

        Args:
            state: Current state (q, p, t)
            gradient_fn: Function that computes ∇V(q) in semantic space
            dt: Time step

        Returns:
            New state at t + dt
        """
        q, p, t = state.q, state.p, state.t

        # Half-step momentum: p(t+dt/2) = p(t) - (dt/2) * ∇V(q(t))
        grad_q = gradient_fn(q)
        p_half = p - 0.5 * dt * grad_q

        # Full-step position: q(t+dt) = q(t) + dt * p(t+dt/2) / m
        q_new = q + dt * p_half / self.mass

        # Half-step momentum (complete): p(t+dt) = p(t+dt/2) - (dt/2) * ∇V(q(t+dt))
        grad_q_new = gradient_fn(q_new)
        p_new = p_half - 0.5 * dt * grad_q_new

        return SemanticState(q=q_new, p=p_new, t=t + dt)

    def integrate_trajectory(self, q0_full: np.ndarray, p0_full: np.ndarray,
                            gradient_fn_full: Callable[[np.ndarray], np.ndarray],
                            dt: float, n_steps: int) -> List[SemanticState]:
        """
        Integrate semantic trajectory using geometric integrator

        Args:
            q0_full: Initial position in full embedding space (384D)
            p0_full: Initial momentum in full embedding space (384D)
            gradient_fn_full: Gradient function in full space
            dt: Time step
            n_steps: Number of integration steps

        Returns:
            List of states in semantic space
        """
        # Project initial conditions to semantic space
        q0_sem = self.project_to_semantic(q0_full)
        p0_sem = self.project_to_semantic(p0_full)

        # Create gradient function in semantic space
        def gradient_fn_semantic(q_sem):
            # Lift to full space, compute gradient, project back
            q_full = self.lift_to_full(q_sem)
            grad_full = gradient_fn_full(q_full)
            return self.project_gradient(grad_full)

        # Integrate
        states = []
        state = SemanticState(q=q0_sem, p=p0_sem, t=0.0)
        states.append(state)

        for _ in range(n_steps):
            state = self.stormer_verlet_step(state, gradient_fn_semantic, dt)
            states.append(state)

        return states

    def compute_energy_drift(self, states: List[SemanticState],
                            potential_fn: Callable) -> Dict:
        """
        Measure how well energy is conserved

        For a good symplectic integrator, energy should be nearly constant
        """
        energies = [state.hamiltonian(potential_fn, self.mass) for state in states]
        energies = np.array(energies)

        return {
            'energies': energies,
            'mean': np.mean(energies),
            'std': np.std(energies),
            'drift': (energies[-1] - energies[0]) / energies[0],  # relative drift
            'conservation_quality': 1.0 - np.std(energies) / np.mean(energies)
        }


class MultiScaleGeometricFlow:
    """
    Geometric flow at multiple Matryoshka scales with resonance

    Each scale has its own frequency → creates interference patterns
    """

    def __init__(self, projection_matrices: Dict[int, np.ndarray],
                 masses: Optional[Dict[int, float]] = None):
        """
        Args:
            projection_matrices: {scale: P_matrix} for each Matryoshka scale
            masses: {scale: mass} semantic masses (default: 1.0 for all)
        """
        self.scales = sorted(projection_matrices.keys())
        self.integrators = {
            scale: GeometricIntegrator(P, mass=masses.get(scale, 1.0) if masses else 1.0)
            for scale, P in projection_matrices.items()
        }

    def compute_resonance(self, states_by_scale: Dict[int, List[SemanticState]]) -> np.ndarray:
        """
        Compute resonance between scales

        Resonance occurs when different scales have coherent phase
        """
        # Extract positions from each scale
        positions = {
            scale: np.array([s.q for s in states])
            for scale, states in states_by_scale.items()
        }

        # Compute phase coherence between scales
        # (simplified: use correlation)
        resonance_over_time = []

        for t in range(len(next(iter(positions.values())))):
            # Get positions at time t across all scales
            pos_t = {scale: positions[scale][t] for scale in self.scales}

            # Compute pairwise correlations
            correlations = []
            for i, s1 in enumerate(self.scales):
                for s2 in self.scales[i+1:]:
                    # Normalize and compute correlation
                    p1 = pos_t[s1] / (np.linalg.norm(pos_t[s1]) + 1e-10)
                    p2 = pos_t[s2] / (np.linalg.norm(pos_t[s2]) + 1e-10)

                    # Pad to same length
                    min_len = min(len(p1), len(p2))
                    corr = np.dot(p1[:min_len], p2[:min_len])
                    correlations.append(abs(corr))

            resonance_over_time.append(np.mean(correlations))

        return np.array(resonance_over_time)

    def integrate_all_scales(self, q0_full_by_scale: Dict[int, np.ndarray],
                            p0_full_by_scale: Dict[int, np.ndarray],
                            gradient_fns: Dict[int, Callable],
                            dt: float, n_steps: int) -> Dict:
        """
        Integrate all scales simultaneously and compute interference
        """
        states_by_scale = {}

        for scale in self.scales:
            integrator = self.integrators[scale]
            states = integrator.integrate_trajectory(
                q0_full_by_scale[scale],
                p0_full_by_scale[scale],
                gradient_fns[scale],
                dt, n_steps
            )
            states_by_scale[scale] = states

        # Compute resonance
        resonance = self.compute_resonance(states_by_scale)

        return {
            'states_by_scale': states_by_scale,
            'resonance': resonance
        }


def visualize_geometric_flow(states: List[SemanticState],
                            dimension_names: List[str],
                            save_path: Optional[str] = None):
    """
    Visualize geometric flow in semantic space
    """
    import matplotlib.pyplot as plt

    # Extract trajectories for each dimension
    n_dims = len(dimension_names)
    positions = np.array([s.q for s in states])  # (n_steps, n_dims)
    momenta = np.array([s.p for s in states])
    times = np.array([s.t for s in states])

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Positions (select top 6 most active dimensions)
    ax1 = axes[0]
    variances = np.var(positions, axis=0)
    top_dims = np.argsort(variances)[-6:][::-1]

    for dim_idx in top_dims:
        ax1.plot(times, positions[:, dim_idx],
                label=dimension_names[dim_idx], linewidth=2, alpha=0.7)

    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title('Geometric Flow: Semantic Positions', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Momenta (same dimensions)
    ax2 = axes[1]
    for dim_idx in top_dims:
        ax2.plot(times, momenta[:, dim_idx],
                label=dimension_names[dim_idx], linewidth=2, alpha=0.7)

    ax2.set_ylabel('Momentum', fontsize=12)
    ax2.set_title('Geometric Flow: Semantic Momenta', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Phase space for dominant dimension
    ax3 = axes[2]
    dominant_dim = top_dims[0]
    ax3.plot(positions[:, dominant_dim], momenta[:, dominant_dim],
            'b-', linewidth=2, alpha=0.7)
    ax3.scatter(positions[0, dominant_dim], momenta[0, dominant_dim],
               c='green', s=200, marker='o', label='Start', zorder=5)
    ax3.scatter(positions[-1, dominant_dim], momenta[-1, dominant_dim],
               c='red', s=200, marker='X', label='End', zorder=5)

    ax3.set_xlabel(f'{dimension_names[dominant_dim]} Position', fontsize=12)
    ax3.set_ylabel(f'{dimension_names[dominant_dim]} Momentum', fontsize=12)
    ax3.set_title(f'Phase Space: {dimension_names[dominant_dim]}',
                 fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved geometric flow visualization: {save_path}")

    return fig, axes


def compute_semantic_force_field(projection_matrix: np.ndarray,
                                 gradient_fn_full: Callable,
                                 sample_points: np.ndarray) -> np.ndarray:
    """
    Compute semantic force field (gradient field in semantic space)

    Args:
        projection_matrix: P matrix for projection
        gradient_fn_full: Gradient function in full embedding space
        sample_points: Points in semantic space to evaluate at (n_points, n_dims)

    Returns:
        Forces at each sample point (n_points, n_dims)
    """
    P = projection_matrix
    P_T = P.T

    forces = []
    for q_sem in sample_points:
        # Lift to full space
        q_full = P_T @ q_sem

        # Compute gradient in full space
        grad_full = gradient_fn_full(q_full)

        # Project force back to semantic space
        force_sem = -P @ grad_full  # F = -∇V

        forces.append(force_sem)

    return np.array(forces)