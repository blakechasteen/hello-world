"""
Semantic Flow: PDE-Based Temporal Dynamics for Semantic Space
==============================================================

Integrates PDE solvers from HoloLoom.warp.semantic_pde with 244D semantic space
to enable temporal evolution of meaning.

PHILOSOPHY:
----------
Semantic meaning is not static - it flows and evolves over time.
Different PDEs model different types of semantic dynamics:

1. **Heat Equation (Diffusion)**:
   - Smooths activation patterns
   - Information spreads from high-activation to low-activation regions
   - Use for: Exploration, activation spreading, concept blending

2. **Wave Equation (Oscillation)**:
   - Creates resonance patterns
   - Oscillatory back-and-forth reasoning
   - Use for: Dialectical reasoning, debate, refinement

3. **Reaction-Diffusion (Competition)**:
   - Competitive activation patterns
   - Tool competition with mutual inhibition
   - Use for: Disambiguation, competitive selection

4. **Hamilton-Jacobi (Optimal Paths)**:
   - Optimal semantic trajectories
   - Gradient-based semantic flow
   - Use for: Goal-directed reasoning

INTEGRATION STRATEGY:
--------------------
1. Project 244D semantic space onto graph Laplacian (discrete → continuous)
2. Evolve activation using PDE solvers
3. Track semantic trajectory over time
4. Visualize how meaning evolves during a session

PERFORMANCE:
-----------
- PDEs are expensive (implicit solvers require matrix inversion)
- Use opt-in for research mode only
- Implicit methods provide stability for large timesteps
- Consider using smaller semantic subspaces (16D) for speed

Author: HoloLoom Mathematical Moonshot Team
Date: 2025-11-03
Priority: 6
"""

from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings

# Import PDE solvers
try:
    from HoloLoom.warp.semantic_pde import (
        PDEType,
        HeatEquationSolver,
        WaveEquationSolver,
        ReactionDiffusionSolver,
        HamiltonJacobiSolver,
        create_heat_solver,
        create_wave_solver,
        create_reaction_diffusion_solver,
        logistic_reaction,
        competitive_reaction,
    )
    _PDE_AVAILABLE = True
except ImportError:
    _PDE_AVAILABLE = False
    warnings.warn("semantic_pde not available. Install or check HoloLoom.warp.semantic_pde")

# Import semantic dimensions
try:
    from HoloLoom.semantic_calculus.dimensions import (
        SemanticSpectrum,
        STANDARD_DIMENSIONS,
        EXTENDED_244_DIMENSIONS,
    )
    _DIMENSIONS_AVAILABLE = True
except ImportError:
    _DIMENSIONS_AVAILABLE = False
    warnings.warn("semantic_calculus.dimensions not available")


# ============================================================================
# Semantic Flow State
# ============================================================================

@dataclass
class SemanticFlowState:
    """
    State of semantic flow at a particular time.

    Attributes:
        activation: Activation vector in semantic space (n_dims,)
        time: Current time
        metadata: Additional metadata (energy, entropy, etc.)
    """
    activation: np.ndarray
    time: float
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate state."""
        if not isinstance(self.activation, np.ndarray):
            self.activation = np.array(self.activation)

    @property
    def energy(self) -> float:
        """Total activation energy: E = ||activation||²"""
        return float(np.sum(self.activation ** 2))

    @property
    def entropy(self) -> float:
        """Shannon entropy of activation distribution."""
        # Normalize to probability distribution
        p = np.abs(self.activation)
        p = p / (np.sum(p) + 1e-10)
        # Remove zeros
        p = p[p > 1e-10]
        # Compute entropy
        return float(-np.sum(p * np.log(p + 1e-10)))


# ============================================================================
# Semantic Flow Orchestrator
# ============================================================================

class SemanticFlow:
    """
    Orchestrates PDE-based temporal dynamics in semantic space.

    Integrates PDE solvers with 244D semantic dimensions to enable
    temporal evolution of meaning.

    Example:
        >>> flow = SemanticFlow(
        ...     dimensions=244,
        ...     pde_type="heat",
        ...     dt=0.01
        ... )
        >>>
        >>> # Evolve semantic state over time
        >>> initial_state = flow.create_state(semantic_projection)
        >>> evolved_state = flow.evolve(initial_state, steps=10)
        >>>
        >>> # Track trajectory
        >>> trajectory = flow.get_trajectory()
        >>> flow.visualize_trajectory()
    """

    def __init__(
        self,
        dimensions: int = 16,  # Use 16D for speed (can use 244 for research)
        pde_type: str = "heat",
        dt: float = 0.01,
        laplacian: Optional[np.ndarray] = None,
        adjacency: Optional[np.ndarray] = None,
        **pde_kwargs
    ):
        """
        Initialize semantic flow orchestrator.

        Args:
            dimensions: Number of semantic dimensions (16 or 244)
            pde_type: Type of PDE ("heat", "wave", "reaction_diffusion", "hamilton_jacobi")
            dt: Time step for integration
            laplacian: Graph Laplacian matrix (n_dims × n_dims)
                      If None, creates default nearest-neighbor Laplacian
            adjacency: Adjacency matrix (for Hamilton-Jacobi)
            **pde_kwargs: Additional arguments for PDE solver
        """
        if not _PDE_AVAILABLE:
            raise ImportError(
                "semantic_pde not available. "
                "Ensure HoloLoom.warp.semantic_pde is installed."
            )

        self.dimensions = dimensions
        self.pde_type = pde_type
        self.dt = dt
        self.pde_kwargs = pde_kwargs

        # Create or validate Laplacian
        if laplacian is None:
            self.laplacian = self._create_default_laplacian(dimensions)
        else:
            if laplacian.shape != (dimensions, dimensions):
                raise ValueError(
                    f"Laplacian shape {laplacian.shape} doesn't match "
                    f"dimensions {dimensions}"
                )
            self.laplacian = laplacian

        # Create adjacency if needed
        if adjacency is None and pde_type == "hamilton_jacobi":
            self.adjacency = self._laplacian_to_adjacency(self.laplacian)
        else:
            self.adjacency = adjacency

        # Create PDE solver
        self.solver = self._create_solver()

        # Trajectory tracking
        self.trajectory: List[SemanticFlowState] = []
        self._trajectory_metadata: Dict = {}

    def _create_default_laplacian(self, n: int) -> np.ndarray:
        """
        Create default nearest-neighbor Laplacian.

        L = D - A where:
        - A is adjacency matrix (nearest neighbors on circle)
        - D is degree matrix
        """
        # Create circular nearest-neighbor adjacency
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = 1.0  # Right neighbor
            A[i, (i - 1) % n] = 1.0  # Left neighbor

        # Degree matrix
        D = np.diag(np.sum(A, axis=1))

        # Laplacian: L = D - A
        L = D - A

        return L

    def _laplacian_to_adjacency(self, L: np.ndarray) -> np.ndarray:
        """Convert Laplacian to adjacency matrix: A = D - L"""
        D = np.diag(np.diag(L))
        return D - L

    def _create_solver(self):
        """Create PDE solver based on pde_type."""
        if self.pde_type == "heat":
            return create_heat_solver(
                self.laplacian,
                dt=self.dt,
                **self.pde_kwargs
            )

        elif self.pde_type == "wave":
            return create_wave_solver(
                self.laplacian,
                dt=self.dt,
                **self.pde_kwargs
            )

        elif self.pde_type == "reaction_diffusion":
            # Default to competitive reaction
            reaction_type = self.pde_kwargs.pop('reaction_type', 'competitive')
            return create_reaction_diffusion_solver(
                self.laplacian,
                reaction_type=reaction_type,
                dt=self.dt,
                **self.pde_kwargs
            )

        elif self.pde_type == "hamilton_jacobi":
            if self.adjacency is None:
                raise ValueError("Hamilton-Jacobi requires adjacency matrix")

            # Default Hamiltonian: H(p) = (1/2)||p||²
            hamiltonian = self.pde_kwargs.pop(
                'hamiltonian',
                lambda p: 0.5 * np.sum(p ** 2)
            )

            return HamiltonJacobiSolver(
                adjacency=self.adjacency,
                hamiltonian=hamiltonian,
                dt=self.dt,
                **self.pde_kwargs
            )

        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

    def create_state(
        self,
        activation: np.ndarray,
        time: float = 0.0,
        **metadata
    ) -> SemanticFlowState:
        """
        Create a semantic flow state.

        Args:
            activation: Activation vector in semantic space
            time: Current time
            **metadata: Additional metadata

        Returns:
            SemanticFlowState
        """
        if len(activation) != self.dimensions:
            raise ValueError(
                f"Activation dimension {len(activation)} doesn't match "
                f"expected {self.dimensions}"
            )

        return SemanticFlowState(
            activation=np.array(activation),
            time=time,
            metadata=metadata
        )

    def evolve(
        self,
        initial_state: SemanticFlowState,
        steps: int = 10,
        track_trajectory: bool = True
    ) -> SemanticFlowState:
        """
        Evolve semantic state using PDE.

        Args:
            initial_state: Initial semantic flow state
            steps: Number of integration steps
            track_trajectory: Whether to track full trajectory

        Returns:
            Final semantic flow state
        """
        # Clear previous trajectory if tracking
        if track_trajectory:
            self.trajectory = [initial_state]

        # Evolve based on PDE type
        if self.pde_type == "heat":
            u0 = initial_state.activation
            times, solutions = self.solver.solve(
                u0=u0,
                t_final=steps * self.dt,
                n_snapshots=steps
            )

            # Create states from solutions
            states = [
                SemanticFlowState(
                    activation=sol,
                    time=t,
                    metadata={'pde_type': 'heat'}
                )
                for t, sol in zip(times, solutions)
            ]

        elif self.pde_type == "wave":
            u0 = initial_state.activation
            v0 = initial_state.metadata.get('velocity', np.zeros_like(u0))

            times, solutions = self.solver.solve(
                u0=u0,
                v0=v0,
                t_final=steps * self.dt,
                n_snapshots=steps
            )

            states = [
                SemanticFlowState(
                    activation=sol,
                    time=t,
                    metadata={'pde_type': 'wave'}
                )
                for t, sol in zip(times, solutions)
            ]

        elif self.pde_type == "reaction_diffusion":
            u0 = initial_state.activation
            times, solutions = self.solver.solve(
                u0=u0,
                t_final=steps * self.dt,
                n_snapshots=steps
            )

            states = [
                SemanticFlowState(
                    activation=sol,
                    time=t,
                    metadata={'pde_type': 'reaction_diffusion'}
                )
                for t, sol in zip(times, solutions)
            ]

        elif self.pde_type == "hamilton_jacobi":
            u0 = initial_state.activation
            times, solutions = self.solver.solve(
                u0=u0,
                t_final=steps * self.dt,
                n_snapshots=steps
            )

            states = [
                SemanticFlowState(
                    activation=sol,
                    time=t,
                    metadata={'pde_type': 'hamilton_jacobi'}
                )
                for t, sol in zip(times, solutions)
            ]

        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

        # Update trajectory
        if track_trajectory:
            self.trajectory = states

        return states[-1]

    def get_trajectory(self) -> List[SemanticFlowState]:
        """Get the full semantic trajectory."""
        return self.trajectory

    def get_trajectory_array(self) -> np.ndarray:
        """Get trajectory as numpy array (n_steps × n_dims)."""
        return np.array([state.activation for state in self.trajectory])

    def compute_energy_evolution(self) -> np.ndarray:
        """Compute energy over trajectory."""
        return np.array([state.energy for state in self.trajectory])

    def compute_entropy_evolution(self) -> np.ndarray:
        """Compute entropy over trajectory."""
        return np.array([state.entropy for state in self.trajectory])

    def visualize_trajectory(
        self,
        dimension_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize semantic trajectory.

        Args:
            dimension_names: Names of semantic dimensions
            save_path: Path to save figure
        """
        if not self.trajectory:
            raise ValueError("No trajectory to visualize. Call evolve() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available. Cannot visualize.")
            return

        # Extract trajectory data
        times = np.array([state.time for state in self.trajectory])
        activations = self.get_trajectory_array()  # (n_steps, n_dims)
        energies = self.compute_energy_evolution()
        entropies = self.compute_entropy_evolution()

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Activation evolution (top 6 most active dimensions)
        ax1 = axes[0]
        variances = np.var(activations, axis=0)
        top_dims = np.argsort(variances)[-6:][::-1]

        for dim_idx in top_dims:
            label = (
                dimension_names[dim_idx] if dimension_names
                else f"Dim {dim_idx}"
            )
            ax1.plot(times, activations[:, dim_idx], label=label, linewidth=2, alpha=0.7)

        ax1.set_ylabel('Activation', fontsize=12)
        ax1.set_title(
            f'Semantic Flow: {self.pde_type.upper()} PDE',
            fontsize=14,
            fontweight='bold'
        )
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Energy evolution
        ax2 = axes[1]
        ax2.plot(times, energies, 'b-', linewidth=2, label='Total Energy')
        ax2.set_ylabel('Energy', fontsize=12)
        ax2.set_title('Semantic Energy Evolution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Entropy evolution
        ax3 = axes[2]
        ax3.plot(times, entropies, 'r-', linewidth=2, label='Entropy')
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Entropy (nats)', fontsize=12)
        ax3.set_title('Semantic Entropy Evolution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved trajectory visualization: {save_path}")

        return fig, axes

    def compute_semantic_distance(
        self,
        state1: SemanticFlowState,
        state2: SemanticFlowState
    ) -> float:
        """
        Compute semantic distance between two states.

        Uses L2 norm in semantic space.
        """
        return float(np.linalg.norm(state1.activation - state2.activation))

    def reset(self):
        """Reset trajectory."""
        self.trajectory = []
        self._trajectory_metadata = {}


# ============================================================================
# Factory Functions
# ============================================================================

def create_semantic_flow(
    dimensions: int = 16,
    pde_type: str = "heat",
    dt: float = 0.01,
    **kwargs
) -> SemanticFlow:
    """
    Factory function to create semantic flow orchestrator.

    Args:
        dimensions: Number of semantic dimensions
        pde_type: Type of PDE ("heat", "wave", "reaction_diffusion", "hamilton_jacobi")
        dt: Time step
        **kwargs: Additional arguments for SemanticFlow

    Returns:
        SemanticFlow instance
    """
    return SemanticFlow(
        dimensions=dimensions,
        pde_type=pde_type,
        dt=dt,
        **kwargs
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SemanticFlowState',
    'SemanticFlow',
    'create_semantic_flow',
]
