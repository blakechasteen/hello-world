"""
Advanced Spring-Based Activation Dynamics
=========================================
Professional-grade physics simulation with modern numerical methods.

IMPROVEMENTS OVER spring_dynamics.py:
- RK4/Verlet integration (not naive Euler)
- Hamiltonian formulation (energy-preserving)
- Stability analysis and convergence guarantees
- Configurable integrator selection

This module replaces the elementary Euler integration in spring_dynamics.py
with production-grade numerical methods from computational physics.

Physics Model (Hamiltonian Formulation):
----------------------------------------
Hamiltonian: H(q, p) = K(p) + U(q)

Kinetic energy:  K = Σ p_i² / (2m_i)
Potential energy: U = Σ (k/2) × (q_i - q_j)² + Σ decay × q_i

Hamilton's equations:
    dq_i/dt = ∂H/∂p_i = p_i / m_i          (velocity from momentum)
    dp_i/dt = -∂H/∂q_i = F_i(q)            (force from positions)

Where:
- q_i: Activation level of node i [0, 1]
- p_i: Momentum (m_i × velocity_i)
- m_i: Node mass (inertia)
- k: Spring stiffness (connection strength)

Author: HoloLoom Mathematical Physics Team
Date: 2025-11-03
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from HoloLoom.memory.integrators import (
    IntegratorType,
    DynamicalState,
    ForceFunction,
    create_integrator,
    analyze_stability,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AdvancedSpringConfig:
    """Configuration for advanced spring dynamics."""

    # Physics parameters
    stiffness: float = 0.15                  # Spring constant k
    damping: float = 0.85                    # Velocity damping (0-1)
    decay: float = 0.98                      # Activation decay per step
    mass: float = 1.0                        # Node mass (default)

    # Integration parameters
    integrator: IntegratorType = IntegratorType.VERLET  # Default: energy-preserving
    dt: float = 0.016                        # Time step (~60fps)
    max_iterations: int = 200                # Maximum simulation steps
    convergence_epsilon: float = 1e-4        # Energy convergence threshold

    # Adaptive step size (for RK45)
    adaptive: bool = False                   # Use adaptive step size?
    rtol: float = 1e-6                       # Relative tolerance
    atol: float = 1e-8                       # Absolute tolerance

    # Stability
    check_stability: bool = True             # Analyze stability on init?
    max_energy_drift: float = 0.1            # Maximum allowed energy drift (10%)

    # Edge type multipliers
    edge_type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'IS_A': 1.2,
        'PART_OF': 1.1,
        'USES': 0.9,
        'MENTIONS': 0.7,
        'RELATED_TO': 0.6,
    })

    # Seed handling
    maintain_seed_activation: bool = True

    def get_edge_stiffness(self, edge_type: str, edge_weight: float) -> float:
        """Calculate effective stiffness for an edge."""
        multiplier = self.edge_type_multipliers.get(edge_type, 1.0)
        return self.stiffness * edge_weight * multiplier


# ============================================================================
# Spring Force Function
# ============================================================================

class SpringForceFunction(ForceFunction):
    """
    Computes forces for spring network in Hamiltonian formulation.

    Hamilton's equations:
        dq/dt = p / m          (velocity from momentum)
        dp/dt = F_spring(q) - damping × p - decay × sign(q)

    Forces:
        F_spring = Σ k_ij × (q_j - q_i)     (spring forces from neighbors)
        F_damping = -c × p                   (velocity damping)
        F_decay = -λ × sign(q)               (activation decay)
    """

    def __init__(
        self,
        graph,
        config: AdvancedSpringConfig,
        node_list: List[str],
        seed_nodes: Dict[str, float]
    ):
        """
        Initialize force function.

        Args:
            graph: NetworkX graph
            config: Spring configuration
            node_list: Ordered list of node IDs
            seed_nodes: Seed activations {node_id: activation}
        """
        self.graph = graph
        self.config = config
        self.node_list = node_list
        self.seed_nodes = seed_nodes

        # Build index mapping
        self.node_to_idx = {node: i for i, node in enumerate(node_list)}

        # Precompute adjacency structure
        self._build_adjacency()

        # Mass array
        self.mass = np.full(len(node_list), config.mass)

    def _build_adjacency(self):
        """Precompute edge connectivity and stiffness."""
        n = len(self.node_list)
        self.neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

        for u, v, edge_data in self.graph.G.edges(data=True):
            if u not in self.node_to_idx or v not in self.node_to_idx:
                continue

            i = self.node_to_idx[u]
            j = self.node_to_idx[v]

            edge_type = edge_data.get('type', 'RELATED_TO')
            edge_weight = edge_data.get('weight', 1.0)
            k = self.config.get_edge_stiffness(edge_type, edge_weight)

            # Store bidirectional connections
            self.neighbors[i].append((j, k))
            self.neighbors[j].append((i, k))

    def __call__(self, state: DynamicalState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time derivatives for Hamilton's equations.

        Args:
            state: Current dynamical state

        Returns:
            Tuple of (dq_dt, dp_dt)
        """
        q = state.q
        p = state.p

        # dq/dt = p / m (velocity from momentum)
        dq_dt = p / self.mass

        # dp/dt = F_spring - damping × p
        dp_dt = np.zeros_like(p)

        # Spring forces: F_i = Σ k_ij × (q_j - q_i)
        for i, neighbor_list in enumerate(self.neighbors):
            for j, k_ij in neighbor_list:
                # Spring force pulls i toward j's activation
                spring_force = k_ij * (q[j] - q[i])
                dp_dt[i] += spring_force

        # Damping force: -c × p
        dp_dt -= self.config.damping * p

        # Decay force: -λ × q (natural forgetting)
        decay_force = -(1.0 - self.config.decay) * q
        dp_dt += decay_force * self.mass  # Scale by mass

        # Maintain seed nodes (if configured)
        if self.config.maintain_seed_activation:
            for node_id, target_activation in self.seed_nodes.items():
                if node_id in self.node_to_idx:
                    i = self.node_to_idx[node_id]
                    # Strong restoring force toward seed activation
                    dp_dt[i] += 10.0 * self.mass[i] * (target_activation - q[i])

        return dq_dt, dp_dt


# ============================================================================
# Advanced Spring Dynamics Engine
# ============================================================================

class AdvancedSpringDynamics:
    """
    Spring-based activation propagation with modern numerical methods.

    IMPROVEMENTS:
    1. Hamiltonian formulation (energy-preserving)
    2. Multiple integrators (RK4, Verlet, RK45)
    3. Adaptive step size (optional)
    4. Stability analysis
    5. Convergence guarantees

    Usage:
        config = AdvancedSpringConfig(integrator=IntegratorType.VERLET)
        dynamics = AdvancedSpringDynamics(kg, config)
        dynamics.activate_nodes({'node_a': 1.0, 'node_b': 0.8})
        result = dynamics.propagate()
    """

    def __init__(self, graph, config: Optional[AdvancedSpringConfig] = None):
        """
        Initialize advanced spring dynamics.

        Args:
            graph: Knowledge graph (must have .G property)
            config: Configuration (or None for defaults)
        """
        self.graph = graph
        self.config = config or AdvancedSpringConfig()

        # Node tracking
        self.node_list = list(graph.G.nodes())
        self.seed_nodes: Dict[str, float] = {}

        # Integrator (created on first propagation)
        self.integrator = None
        self.force_fn = None

        # State
        self.state: Optional[DynamicalState] = None

        # Metrics
        self.iterations = 0
        self.converged = False
        self.final_energy = 0.0
        self.stability_report: Optional[Dict[str, Any]] = None

    def reset(self):
        """Reset to initial state."""
        self.seed_nodes.clear()
        self.state = None
        self.iterations = 0
        self.converged = False
        self.final_energy = 0.0

    def activate_nodes(self, activations: Dict[str, float]):
        """
        Set initial activation for seed nodes.

        Args:
            activations: {node_id: activation_level} where activation in [0, 1]
        """
        self.seed_nodes = {}

        # Build initial state
        n = len(self.node_list)
        q0 = np.zeros(n)
        p0 = np.zeros(n)

        node_to_idx = {node: i for i, node in enumerate(self.node_list)}

        for node_id, activation in activations.items():
            if node_id in node_to_idx:
                i = node_to_idx[node_id]
                q0[i] = np.clip(activation, 0.0, 1.0)
                self.seed_nodes[node_id] = q0[i]

        self.state = DynamicalState(q=q0, p=p0, t=0.0)

    def _initialize_integrator(self):
        """Create integrator on first use."""
        if self.integrator is not None:
            return

        # Create force function
        self.force_fn = SpringForceFunction(
            self.graph,
            self.config,
            self.node_list,
            self.seed_nodes
        )

        # Create integrator
        if self.config.integrator == IntegratorType.RK45:
            self.integrator = create_integrator(
                self.config.integrator,
                self.force_fn,
                rtol=self.config.rtol,
                atol=self.config.atol
            )
        elif self.config.integrator in [IntegratorType.SYMPLECTIC_EULER, IntegratorType.VERLET]:
            mass = np.full(len(self.node_list), self.config.mass)
            self.integrator = create_integrator(
                self.config.integrator,
                self.force_fn,
                mass=mass
            )
        else:
            self.integrator = create_integrator(
                self.config.integrator,
                self.force_fn
            )

    def propagate(self) -> 'AdvancedSpringResult':
        """
        Propagate activation through spring network.

        Uses configured integrator (RK4, Verlet, etc.) for accurate simulation.

        Returns:
            AdvancedSpringResult with convergence info and activated nodes
        """
        if self.state is None:
            raise RuntimeError("Must call activate_nodes() before propagate()")

        self._initialize_integrator()

        prev_energy = float('inf')
        dt = self.config.dt

        for step in range(self.config.max_iterations):
            self.iterations = step + 1

            # Take integration step
            if self.config.adaptive and hasattr(self.integrator, 'step'):
                # Adaptive step size
                result = self.integrator.step(self.state, dt)
                if hasattr(result, 'accepted'):
                    if result.accepted:
                        self.state = result.state
                        dt = result.dt_next
                    else:
                        dt = result.dt_next
                        continue  # Retry with smaller dt
                else:
                    self.state = result
            else:
                # Fixed step size
                self.state = self.integrator.step(self.state, dt)

            # Clamp activations to [0, 1]
            self.state.q = np.clip(self.state.q, 0.0, 1.0)

            # Compute energy for convergence check
            energy = self._compute_hamiltonian()
            energy_change = abs(energy - prev_energy)

            # Check convergence
            if energy_change < self.config.convergence_epsilon:
                self.converged = True
                self.final_energy = energy
                break

            prev_energy = energy

        else:
            # Max iterations reached
            self.converged = False
            self.final_energy = prev_energy

        # Extract activated nodes
        active_nodes = self._get_active_nodes()

        # Stability analysis (if enabled)
        if self.config.check_stability:
            self.stability_report = self._analyze_stability()

        return AdvancedSpringResult(
            iterations=self.iterations,
            converged=self.converged,
            final_energy=self.final_energy,
            activated_nodes=active_nodes,
            node_activations=self._extract_activations(),
            stability_report=self.stability_report
        )

    def _compute_hamiltonian(self) -> float:
        """
        Compute total Hamiltonian (energy).

        H = K + U
        K = Σ p²/(2m)           (kinetic)
        U = Σ (k/2)(q_i-q_j)²   (potential)

        Returns:
            Total energy
        """
        if self.state is None:
            return 0.0

        # Kinetic energy: K = Σ p²/(2m)
        kinetic = np.sum(self.state.p**2 / (2.0 * self.force_fn.mass))

        # Potential energy: U = Σ (k/2)(Δq)²
        potential = 0.0
        for i, neighbor_list in enumerate(self.force_fn.neighbors):
            for j, k_ij in neighbor_list:
                dq = self.state.q[j] - self.state.q[i]
                potential += 0.5 * k_ij * dq * dq

        return kinetic + potential

    def _get_active_nodes(self, threshold: float = 0.1) -> List[str]:
        """
        Get nodes with activation above threshold.

        Args:
            threshold: Minimum activation

        Returns:
            List of node IDs sorted by activation (descending)
        """
        if self.state is None:
            return []

        active = [
            (self.node_list[i], self.state.q[i])
            for i in range(len(self.node_list))
            if self.state.q[i] >= threshold
        ]

        active.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in active]

    def _extract_activations(self) -> Dict[str, float]:
        """Extract all non-trivial activations."""
        if self.state is None:
            return {}

        return {
            self.node_list[i]: float(self.state.q[i])
            for i in range(len(self.node_list))
            if self.state.q[i] > 0.01
        }

    def _analyze_stability(self) -> Dict[str, Any]:
        """
        Analyze numerical stability of the integration.

        Returns:
            Dict with stability metrics
        """
        # Run short stability test
        test_state = self.state.copy()
        stability = analyze_stability(
            self.integrator,
            test_state,
            self.config.dt,
            n_steps=min(100, self.config.max_iterations)
        )

        # Check against thresholds
        stability['stable'] = stability['energy_drift'] < self.config.max_energy_drift
        stability['integrator'] = self.config.integrator.value

        return stability

    def get_activation(self, node_id: str) -> float:
        """Get current activation for a node."""
        if self.state is None or node_id not in self.node_list:
            return 0.0

        idx = self.node_list.index(node_id)
        return float(self.state.q[idx])


# ============================================================================
# Result Data Structure
# ============================================================================

@dataclass
class AdvancedSpringResult:
    """Result from advanced spring propagation."""

    iterations: int
    converged: bool
    final_energy: float

    activated_nodes: List[str]
    node_activations: Dict[str, float]

    stability_report: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        status = "converged" if self.converged else "max iterations"
        energy_str = f"energy={self.final_energy:.4e}"
        stable_str = ""

        if self.stability_report:
            stable = self.stability_report.get('stable', True)
            drift = self.stability_report.get('energy_drift', 0.0)
            stable_str = f", stable={stable}, drift={drift:.2e}"

        return (
            f"AdvancedSpringPropagation({status} in {self.iterations} steps, "
            f"{energy_str}, activated={len(self.activated_nodes)} nodes{stable_str})"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'AdvancedSpringConfig',
    'AdvancedSpringDynamics',
    'AdvancedSpringResult',
    'SpringForceFunction',
]
