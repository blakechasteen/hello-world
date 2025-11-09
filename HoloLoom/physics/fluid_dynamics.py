"""
Context Flow Dynamics - Navier-Stokes Solver for Context Packing

Implements fluid dynamics for adaptive context window packing.
Context flows like water through the knowledge graph, naturally filling
available space with maximum relevance.

Physics Model:
    Navier-Stokes Equation:
        ∂v/∂t + (v·∇)v = -∇p + ν∇²v + f

    Where:
        v = velocity field (information flow)
        p = pressure field (importance density)
        ν = viscosity (prevents abrupt changes)
        f = external forces (user queries)

Core Insight:
    High-pressure (important) context flows to low-pressure (sparse) regions,
    naturally optimizing context window packing without manual tuning.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio

from .pressure_field import PressureField, PressureSource
from .velocity_field import VelocityField

try:
    from ..memory.graph import KG
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


@dataclass
class FlowState:
    """Current state of the flow system."""
    timestep: int
    time: float
    pressure_stats: Dict[str, float]
    velocity_stats: Dict[str, float]
    tokens_used: int
    sparse_regions: List[str]


class ContextFlowEngine:
    """
    Navier-Stokes solver for adaptive context packing.

    Context flows naturally through the knowledge graph:
        - High-pressure (important) → Low-pressure (sparse)
        - Viscosity prevents abrupt changes
        - Reaches equilibrium automatically

    Usage:
        engine = ContextFlowEngine(graph, viscosity=0.01)

        # Inject high-pressure context (user query)
        engine.inject_context("ThompsonSampling", importance=0.95, tokens=50)

        # Propagate via Navier-Stokes
        for _ in range(10):
            engine.step(dt=0.01)

        # Extract optimally packed context
        packed = engine.extract_context(max_tokens=8000)
    """

    def __init__(
        self,
        graph: Optional["KG"] = None,
        viscosity: float = 0.01,
        timestep: float = 0.01,
        sparse_threshold: float = 0.3
    ):
        """
        Initialize context flow engine.

        Args:
            graph: Knowledge graph (optional, can use simple dict of neighbors)
            viscosity: Damping coefficient (0.01 = low damping, fast propagation)
            timestep: Integration timestep
            sparse_threshold: Pressure below this is considered sparse
        """
        self.graph = graph
        self.viscosity = viscosity
        self.default_timestep = timestep
        self.sparse_threshold = sparse_threshold

        # Physics fields
        self.pressure = PressureField(default_pressure=0.0)
        self.velocity = VelocityField(viscosity=viscosity)

        # Simulation state
        self.current_timestep = 0
        self.current_time = 0.0
        self.tokens_used = 0

        # Graph structure (if no KG provided)
        self.neighbors: Dict[str, List[str]] = {}

    def add_edge(self, source: str, target: str):
        """Add an edge to the graph structure."""
        if source not in self.neighbors:
            self.neighbors[source] = []
        if target not in self.neighbors[source]:
            self.neighbors[source].append(target)

        # Bidirectional
        if target not in self.neighbors:
            self.neighbors[target] = []
        if source not in self.neighbors[target]:
            self.neighbors[target].append(source)

    def get_neighbors(self, node: str) -> List[str]:
        """Get neighbors of a node."""
        if self.graph and HAS_GRAPH:
            # Use KG if available
            neighbors = []
            for edge in self.graph.get_edges():
                if edge.source == node:
                    neighbors.append(edge.target)
                elif edge.target == node:
                    neighbors.append(edge.source)
            return list(set(neighbors))
        else:
            # Use simple neighbors dict
            return self.neighbors.get(node, [])

    def inject_context(
        self,
        node: str,
        importance: float,
        tokens: int = 0,
        text: Optional[str] = None
    ):
        """
        Inject context at a node (creates high-pressure source).

        Like injecting water into a pipe system - creates pressure
        that drives flow to other regions.

        Args:
            node: Target node
            importance: Importance value (0.0 - 1.0)
            tokens: Token count for this context
            text: Optional context text
        """
        self.pressure.inject_pressure(node, importance, tokens, text)
        self.tokens_used += tokens

    def step(self, dt: Optional[float] = None):
        """
        Single Navier-Stokes timestep.

        Implements:
            ∂v/∂t = -∇p + ν∇²v + advection

        Args:
            dt: Timestep (defaults to self.default_timestep)
        """
        if dt is None:
            dt = self.default_timestep

        # Get all nodes with pressure
        nodes = set(self.pressure.pressures.keys())

        # Update velocities based on pressure gradients
        for node in nodes:
            neighbors = self.get_neighbors(node)
            if not neighbors:
                continue

            # Compute pressure gradient
            gradients = self.pressure.compute_gradient(node, neighbors)

            # Update velocity for each neighbor connection
            for neighbor, gradient in gradients.items():
                self.velocity.update_from_pressure_gradient(
                    source=node,
                    target=neighbor,
                    gradient=gradient,
                    dt=dt
                )

        # Advection: velocity transports itself
        for node in nodes:
            neighbors = self.get_neighbors(node)
            if neighbors:
                self.velocity.advect(node, neighbors, dt)

        # Propagate pressure based on velocity
        self._propagate_pressure(dt)

        # Apply viscosity damping
        self.velocity.decay(decay_rate=1.0 - self.viscosity * dt)

        # Update state
        self.current_timestep += 1
        self.current_time += dt

    def _propagate_pressure(self, dt: float):
        """
        Propagate pressure based on velocity field.

        Pressure flows from high to low, driven by velocity.
        """
        nodes = set(self.pressure.pressures.keys())

        # Collect pressure changes
        pressure_deltas = {}

        for node in nodes:
            neighbors = self.get_neighbors(node)
            if not neighbors:
                continue

            # Net flow determines pressure change
            net_flow = self.velocity.get_net_flow(node)

            # If net_flow > 0: pressure increases (accumulation)
            # If net_flow < 0: pressure decreases (depletion)
            pressure_deltas[node] = net_flow * dt

        # Apply pressure changes
        for node, delta in pressure_deltas.items():
            current_p = self.pressure.get_pressure(node)
            new_p = current_p + delta
            self.pressure.set_pressure(node, new_p)

    def detect_sparse_regions(self) -> List[str]:
        """
        Find low-pressure (sparse context) regions.

        These are candidates for reverse prompting.
        """
        nodes = set(self.pressure.pressures.keys())
        return self.pressure.detect_sparse_regions(
            threshold=self.sparse_threshold,
            nodes=nodes
        )

    def generate_reverse_prompt(self, node: str) -> Optional[str]:
        """
        Generate query to fill sparse region.

        Low-pressure regions "pull" context toward them.

        Args:
            node: Sparse node

        Returns:
            Query string or None if no neighbors
        """
        neighbors = self.get_neighbors(node)
        if not neighbors:
            return f"What is {node}?"

        # Find high-pressure neighbors
        high_pressure_neighbors = [
            n for n in neighbors
            if self.pressure.get_pressure(n) > 0.7
        ]

        if high_pressure_neighbors:
            neighbor = high_pressure_neighbors[0]
            return f"How does {node} relate to {neighbor}?"
        else:
            return f"What is {node}?"

    def get_state(self) -> FlowState:
        """Get current flow state."""
        return FlowState(
            timestep=self.current_timestep,
            time=self.current_time,
            pressure_stats=self.pressure.get_statistics(),
            velocity_stats=self.velocity.get_statistics(),
            tokens_used=self.tokens_used,
            sparse_regions=self.detect_sparse_regions()
        )

    def extract_context(
        self,
        max_tokens: int,
        sort_by: str = "pressure"
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        Extract optimally packed context.

        Fills token budget with highest-pressure (most relevant) nodes first.

        Args:
            max_tokens: Maximum token budget
            sort_by: Sort criterion ("pressure" or "flow")

        Returns:
            List of (node, importance, text) tuples
        """
        # Get all nodes with context
        nodes = [
            (node, self.pressure.get_pressure(node))
            for node in self.pressure.pressures.keys()
        ]

        # Sort by pressure (highest first)
        if sort_by == "pressure":
            nodes.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == "flow":
            # Sort by total flow (inflow + outflow)
            nodes.sort(
                key=lambda x: abs(self.velocity.get_net_flow(x[0])),
                reverse=True
            )

        # Pack context up to token limit
        packed = []
        tokens_remaining = max_tokens

        for node, importance in nodes:
            # Get source metadata if available
            source = self.pressure.sources.get(node)
            if source:
                tokens = source.tokens
                text = source.text
            else:
                tokens = 50  # Default estimate
                text = None

            if tokens <= tokens_remaining:
                packed.append((node, importance, text))
                tokens_remaining -= tokens

            if tokens_remaining <= 0:
                break

        return packed

    def __repr__(self) -> str:
        state = self.get_state()
        return (
            f"ContextFlowEngine(t={state.timestep}, "
            f"nodes={len(self.pressure.pressures)}, "
            f"tokens={state.tokens_used}, "
            f"sparse={len(state.sparse_regions)})"
        )
