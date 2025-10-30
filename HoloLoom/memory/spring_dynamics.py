"""
Spring-Based Activation Dynamics for Knowledge Graphs
=====================================================
Physics-driven spreading activation using Hooke's Law.

This module extends YarnGraph with dynamic energy propagation.
Instead of static similarity measures, activation spreads through
elastic connections (springs) until the system reaches equilibrium.

Physics Model:
--------------
F = -k × (a_i - a_j) - c × v_i

Where:
- k (stiffness): Connection strength (from edge weight)
- Δa: Activation difference between connected nodes
- c (damping): Prevents oscillation, models forgetting
- v (velocity): Rate of activation change

Energy Landscape:
-----------------
E = Σ (1/2 × k × (a_i - a_j)²)  [spring potential]
  + Σ (1/2 × m × v_i²)           [kinetic energy]
  + Σ decay × a_i                [dissipation]

Query activation creates high-energy state. System relaxes toward
equilibrium, revealing semantically related memories.

Author: HoloLoom Physics Team
Date: 2025-10-29
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import math


# ============================================================================
# Spring Dynamics Configuration
# ============================================================================

@dataclass
class SpringConfig:
    """Configuration for spring-based activation dynamics."""

    # Physics parameters
    stiffness: float = 0.15          # k: Spring stiffness (0.05-0.5 typical)
    damping: float = 0.85            # c: Damping coefficient (0.5-0.95 typical)
    decay: float = 0.98              # Activation decay per step (0.90-0.99)

    # Simulation parameters
    max_iterations: int = 200        # Maximum propagation steps
    convergence_epsilon: float = 1e-4  # Energy change threshold for convergence
    dt: float = 0.016                # Time step (~60fps for visualization)

    # Activation parameters
    activation_threshold: float = 0.1  # Minimum activation to be considered "active"
    mass: float = 1.0                # Node mass (for future extensions)

    # Edge type multipliers (different relationship types have different stiffness)
    edge_type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'IS_A': 1.2,          # Taxonomic relationships are strong
        'PART_OF': 1.1,       # Compositional relationships
        'USES': 0.9,          # Usage relationships
        'MENTIONS': 0.7,      # Weaker associative links
        'RELATED_TO': 0.6,    # Generic relationships
    })

    def get_edge_stiffness(self, edge_type: str, edge_weight: float) -> float:
        """Calculate effective stiffness for an edge."""
        multiplier = self.edge_type_multipliers.get(edge_type, 1.0)
        return self.stiffness * edge_weight * multiplier


# ============================================================================
# Node Activation State
# ============================================================================

@dataclass
class NodeState:
    """Dynamic state for a single node in the spring network."""

    node_id: str
    activation: float = 0.0      # Current activation level [0, 1]
    velocity: float = 0.0        # Activation velocity (rate of change)
    mass: float = 1.0            # Mass for physics calculations

    def reset(self):
        """Reset to inactive state."""
        self.activation = 0.0
        self.velocity = 0.0

    def apply_force(self, force: float, dt: float, damping: float, decay: float):
        """
        Apply force and update state.

        Uses simple Euler integration:
        1. acceleration = F / m
        2. velocity += acceleration × dt
        3. velocity *= damping (energy dissipation)
        4. activation += velocity × dt
        5. activation *= decay (natural forgetting)
        """
        # F = ma → a = F/m
        acceleration = force / self.mass

        # Update velocity
        self.velocity += acceleration * dt
        self.velocity *= damping  # Damping reduces velocity

        # Update activation
        self.activation += self.velocity * dt
        self.activation *= decay  # Natural decay (forgetting)

        # Clamp activation to [0, 1]
        self.activation = max(0.0, min(1.0, self.activation))


# ============================================================================
# Spring Dynamics Engine
# ============================================================================

class SpringDynamics:
    """
    Spring-based activation propagation for knowledge graphs.

    This class extends a knowledge graph (KG) with physics-based
    spreading activation. It maintains node states and propagates
    activation energy through graph edges (modeled as springs).

    Usage:
        # Create dynamics engine for a knowledge graph
        dynamics = SpringDynamics(kg, config)

        # Activate seed nodes (e.g., from query embedding)
        dynamics.activate_nodes({'Thompson Sampling': 1.0, 'Bandits': 0.8})

        # Propagate activation
        result = dynamics.propagate()

        # Get activated nodes above threshold
        active_nodes = dynamics.get_active_nodes()
    """

    def __init__(self, graph, config: Optional[SpringConfig] = None):
        """
        Initialize spring dynamics engine.

        Args:
            graph: Knowledge graph (must have .G property that's a NetworkX graph)
            config: SpringConfig (or None for defaults)
        """
        self.graph = graph
        self.config = config or SpringConfig()

        # Node states: {node_id: NodeState}
        self.node_states: Dict[str, NodeState] = {}

        # Initialize states for all graph nodes
        self._initialize_states()

        # Metrics
        self.iterations = 0
        self.converged = False
        self.final_energy = 0.0

    def _initialize_states(self):
        """Create NodeState for every node in graph."""
        for node in self.graph.G.nodes():
            self.node_states[node] = NodeState(
                node_id=node,
                mass=self.config.mass
            )

    def reset(self):
        """Reset all node states to inactive."""
        for state in self.node_states.values():
            state.reset()
        self.iterations = 0
        self.converged = False
        self.final_energy = 0.0

    def activate_nodes(self, activations: Dict[str, float]):
        """
        Set initial activation for seed nodes.

        Args:
            activations: {node_id: activation_level}
                        where activation_level is in [0, 1]
        """
        for node_id, activation in activations.items():
            if node_id in self.node_states:
                self.node_states[node_id].activation = max(0.0, min(1.0, activation))
                self.node_states[node_id].velocity = 0.0

    def propagate(self) -> 'SpringPropagationResult':
        """
        Propagate activation through springs until convergence.

        Returns:
            SpringPropagationResult with convergence info and activated nodes
        """
        prev_energy = float('inf')

        for step in range(self.config.max_iterations):
            self.iterations = step + 1

            # Compute forces for all nodes
            forces = self._compute_forces()

            # Apply forces to update states
            for node_id, force in forces.items():
                state = self.node_states[node_id]
                state.apply_force(
                    force=force,
                    dt=self.config.dt,
                    damping=self.config.damping,
                    decay=self.config.decay
                )

            # Check convergence (energy change < epsilon)
            energy = self._compute_energy()
            energy_change = abs(energy - prev_energy)

            if energy_change < self.config.convergence_epsilon:
                self.converged = True
                self.final_energy = energy
                break

            prev_energy = energy

        else:
            # Max iterations reached without convergence
            self.converged = False
            self.final_energy = prev_energy

        # Build result
        active_nodes = self.get_active_nodes()
        return SpringPropagationResult(
            iterations=self.iterations,
            converged=self.converged,
            final_energy=self.final_energy,
            activated_nodes=active_nodes,
            node_activations={nid: state.activation for nid, state in self.node_states.items()
                            if state.activation > 0.01}  # Only non-trivial activations
        )

    def _compute_forces(self) -> Dict[str, float]:
        """
        Compute spring forces for all nodes.

        For each node:
        F = Σ (over neighbors) [-k × (a_i - a_j)]

        Returns:
            {node_id: total_force}
        """
        forces: Dict[str, float] = {nid: 0.0 for nid in self.node_states.keys()}

        # Iterate over all edges
        for u, v, edge_data in self.graph.G.edges(data=True):
            # Get node states
            state_u = self.node_states.get(u)
            state_v = self.node_states.get(v)

            if not state_u or not state_v:
                continue  # Skip if either node not in states

            # Calculate spring force: F = -k × Δa
            edge_type = edge_data.get('type', 'RELATED_TO')
            edge_weight = edge_data.get('weight', 1.0)
            stiffness = self.config.get_edge_stiffness(edge_type, edge_weight)

            # Activation difference
            activation_diff = state_v.activation - state_u.activation

            # Spring force (pulls u toward v's activation level)
            force = stiffness * activation_diff

            # Apply to both nodes (Newton's third law)
            forces[u] += force
            forces[v] -= force  # Equal and opposite

        return forces

    def _compute_energy(self) -> float:
        """
        Compute total system energy.

        E = Σ (spring potential) + Σ (kinetic energy)

        Returns:
            Total energy (float)
        """
        energy = 0.0

        # Spring potential energy: (1/2) × k × (Δa)²
        for u, v, edge_data in self.graph.G.edges(data=True):
            state_u = self.node_states.get(u)
            state_v = self.node_states.get(v)

            if not state_u or not state_v:
                continue

            edge_type = edge_data.get('type', 'RELATED_TO')
            edge_weight = edge_data.get('weight', 1.0)
            stiffness = self.config.get_edge_stiffness(edge_type, edge_weight)

            activation_diff = state_v.activation - state_u.activation
            spring_energy = 0.5 * stiffness * (activation_diff ** 2)
            energy += spring_energy

        # Kinetic energy: (1/2) × m × v²
        for state in self.node_states.values():
            kinetic_energy = 0.5 * state.mass * (state.velocity ** 2)
            energy += kinetic_energy

        return energy

    def get_active_nodes(self, threshold: Optional[float] = None) -> List[str]:
        """
        Get nodes with activation above threshold.

        Args:
            threshold: Minimum activation (or None for config default)

        Returns:
            List of node IDs sorted by activation (highest first)
        """
        if threshold is None:
            threshold = self.config.activation_threshold

        active = [
            (node_id, state.activation)
            for node_id, state in self.node_states.items()
            if state.activation >= threshold
        ]

        # Sort by activation (descending)
        active.sort(key=lambda x: x[1], reverse=True)

        return [node_id for node_id, _ in active]

    def get_activation(self, node_id: str) -> float:
        """Get current activation level for a node."""
        state = self.node_states.get(node_id)
        return state.activation if state else 0.0


# ============================================================================
# Result Data Structure
# ============================================================================

@dataclass
class SpringPropagationResult:
    """Result of spring activation propagation."""

    iterations: int                      # Steps taken
    converged: bool                     # Did it reach equilibrium?
    final_energy: float                 # Final energy state

    activated_nodes: List[str]          # Node IDs above threshold (sorted by activation)
    node_activations: Dict[str, float]  # {node_id: activation_level}

    def __str__(self) -> str:
        status = "converged" if self.converged else "max iterations"
        return (
            f"SpringPropagation({status} in {self.iterations} steps, "
            f"energy={self.final_energy:.4f}, "
            f"activated={len(self.activated_nodes)} nodes)"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SpringConfig',
    'SpringDynamics',
    'SpringPropagationResult',
    'NodeState',
]
