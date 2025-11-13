"""
HoloLoom Physics - Unified Physics Engine

Integrates all physics phases into a single coordinated system:
    - Phase 1: Gradient Flow (Routing)
    - Phase 2: Fluid Dynamics (Context Packing)
    - Phase 3: Thermodynamics (Exploration/Exploitation)
    - Phase 4: Wave Mechanics (Pattern Detection)

Architecture:
    UnifiedPhysicsEngine orchestrates all physics systems:
    1. Gradient flow routes queries to optimal tools
    2. Fluid dynamics packs context optimally
    3. Thermodynamics balances exploration/exploitation
    4. Wave mechanics detects patterns via interference

Usage:
    from HoloLoom.physics import UnifiedPhysicsEngine

    # Create unified engine
    physics = UnifiedPhysicsEngine()

    # Process query through all physics layers
    result = await physics.process(query, context, available_actions)

    # All physics systems work together!
    #   - Routing: Gradient flow
    #   - Packing: Fluid dynamics
    #   - Selection: Thermodynamics
    #   - Patterns: Wave mechanics

Author: Claude Code (with HoloLoom architecture by Blake)
Date: November 9, 2025
Phase: Unified Physics (Integration of Phases 1-4)
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Import all physics engines
from .gradient_flow import GradientFlowEngine, RouteDecision, Target
from .multi_physics_packer import MultiPhysicsPacker, MultiPhysicsResult
from .thermodynamics import ThermodynamicOptimizer, EnergyCalculator
from .wave_mechanics import WaveMechanicsEngine, InterferencePattern, ResonancePattern


# ============================================================================
# Unified Physics Result
# ============================================================================

@dataclass
class UnifiedPhysicsResult:
    """Result from unified physics processing."""

    # Phase 1: Gradient Flow Routing
    routing_decision: Optional[RouteDecision] = None
    routing_loss: float = 0.0

    # Phase 2: Fluid Dynamics Context Packing
    packing_result: Optional[MultiPhysicsResult] = None
    context_efficiency: float = 0.0

    # Phase 3: Thermodynamics Action Selection
    selected_action: Optional[str] = None
    exploration_temperature: float = 1.0
    free_energy: float = 0.0

    # Phase 4: Wave Mechanics Pattern Detection
    constructive_patterns: List[InterferencePattern] = field(default_factory=list)
    destructive_patterns: List[InterferencePattern] = field(default_factory=list)
    resonances: List[ResonancePattern] = field(default_factory=list)

    # Unified metrics
    total_energy: float = 0.0      # Total system energy
    total_entropy: float = 0.0     # Total system entropy
    total_free_energy: float = 0.0 # F = E - T*S

    # Execution time
    duration_ms: float = 0.0

    # Component timings
    routing_ms: float = 0.0
    packing_ms: float = 0.0
    thermodynamics_ms: float = 0.0
    wave_mechanics_ms: float = 0.0


# ============================================================================
# Physics Integration Strategy
# ============================================================================

class PhysicsMode(Enum):
    """Physics integration mode."""
    SEQUENTIAL = "sequential"      # Run phases sequentially
    PARALLEL = "parallel"          # Run independent phases in parallel
    ADAPTIVE = "adaptive"          # Adapt based on query complexity


# ============================================================================
# Unified Physics Engine
# ============================================================================

class UnifiedPhysicsEngine:
    """
    Unified physics engine orchestrating all physics systems.

    Integrates:
    - Phase 1: Gradient Flow (routing queries to optimal tools)
    - Phase 2: Fluid Dynamics (packing context optimally)
    - Phase 3: Thermodynamics (balancing exploration/exploitation)
    - Phase 4: Wave Mechanics (detecting patterns)

    All physics systems work together to provide intelligent,
    self-optimizing behavior with complete provenance.
    """

    def __init__(
        self,
        enable_routing: bool = True,
        enable_packing: bool = True,
        enable_thermodynamics: bool = True,
        enable_wave_mechanics: bool = True,
        mode: str = "adaptive"
    ):
        """
        Initialize unified physics engine.

        Args:
            enable_routing: Enable gradient flow routing
            enable_packing: Enable fluid dynamics packing
            enable_thermodynamics: Enable thermodynamic optimization
            enable_wave_mechanics: Enable wave pattern detection
            mode: Integration mode (sequential/parallel/adaptive)
        """
        self.enable_routing = enable_routing
        self.enable_packing = enable_packing
        self.enable_thermodynamics = enable_thermodynamics
        self.enable_wave_mechanics = enable_wave_mechanics
        self.mode = PhysicsMode(mode)

        # Initialize physics engines
        if enable_routing:
            from .gradient_flow import combined_loss
            self.gradient_engine = GradientFlowEngine(
                loss_fn=combined_loss(load_weight=0.3, latency_weight=0.3, cost_weight=0.4)
            )
        else:
            self.gradient_engine = None

        if enable_packing:
            self.packing_engine = MultiPhysicsPacker(max_tokens=8000)
        else:
            self.packing_engine = None

        if enable_thermodynamics:
            self.thermo_optimizer = ThermodynamicOptimizer(
                initial_temperature=5.0,
                cooling_schedule="exponential",
                cooling_rate=0.95,
                auto_anneal=True
            )
        else:
            self.thermo_optimizer = None

        if enable_wave_mechanics:
            self.wave_engine = WaveMechanicsEngine(
                wave_speed=1.0,
                damping=0.01,
                history_length=50
            )
        else:
            self.wave_engine = None

        # Statistics
        self.total_queries = 0
        self.total_energy_history = []

    async def process(
        self,
        query: str,
        actions: List[str],
        action_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        components: Optional[Dict[str, Any]] = None,
        graph_structure: Optional[List[Tuple[str, str]]] = None
    ) -> UnifiedPhysicsResult:
        """
        Process query through all physics layers.

        Args:
            query: Input query
            actions: Available actions
            action_metrics: Metrics per action (cost, quality, latency, etc.)
            components: Components for context packing (optional)
            graph_structure: Graph edges for wave mechanics (optional)

        Returns:
            Unified physics result
        """
        import time
        start_time = time.time()

        result = UnifiedPhysicsResult()

        # ================================================================
        # Phase 1: Gradient Flow Routing
        # ================================================================
        if self.enable_routing and action_metrics:
            routing_start = time.time()

            # Convert actions to targets
            targets = [
                Target(name=action, metrics=action_metrics.get(action, {}))
                for action in actions
            ]

            # Route via gradient flow
            routing_decision = self.gradient_engine.route(
                targets=targets,
                max_steps=20
            )

            result.routing_decision = routing_decision
            result.routing_loss = routing_decision.loss
            result.routing_ms = (time.time() - routing_start) * 1000

        # ================================================================
        # Phase 2: Fluid Dynamics Context Packing
        # ================================================================
        if self.enable_packing and components:
            packing_start = time.time()

            # Pack context via multi-physics
            packing_result = await self.packing_engine.pack(components)

            result.packing_result = packing_result
            result.context_efficiency = (
                packing_result.total_tokens_used /
                packing_result.total_tokens_available
                if packing_result.total_tokens_available > 0 else 0.0
            )
            result.packing_ms = (time.time() - packing_start) * 1000

        # ================================================================
        # Phase 3: Thermodynamics Action Selection
        # ================================================================
        if self.enable_thermodynamics and action_metrics:
            thermo_start = time.time()

            # Compute energies for each action
            energies = {}
            for action in actions:
                metrics = action_metrics.get(action, {})
                energy = EnergyCalculator.compute_energy(
                    cost=metrics.get("cost", 0.0),
                    error=metrics.get("error", 0.0),
                    latency=metrics.get("latency", 0.0)
                )
                energies[action] = energy

            # Select action via Boltzmann distribution
            selected_action = self.thermo_optimizer.select_action(
                actions=actions,
                energies=energies
            )

            # Update thermodynamic state
            self.thermo_optimizer.update(
                action=selected_action,
                energy=energies[selected_action],
                actions=actions
            )

            stats = self.thermo_optimizer.get_statistics()
            result.selected_action = selected_action
            result.exploration_temperature = stats["temperature"]
            result.free_energy = stats["free_energy"]
            result.thermodynamics_ms = (time.time() - thermo_start) * 1000

        # ================================================================
        # Phase 4: Wave Mechanics Pattern Detection
        # ================================================================
        if self.enable_wave_mechanics:
            wave_start = time.time()

            # Build graph structure if provided
            if graph_structure:
                for source, target in graph_structure:
                    self.wave_engine.add_edge(source, target)

            # Inject wave for query patterns
            # (In real usage, would extract entities from query)
            query_nodes = query.split()[:5]  # Simple: first 5 words
            for node in query_nodes:
                self.wave_engine.inject_wave(node, amplitude=1.0)

            # Propagate waves
            for _ in range(5):  # Few steps for real-time
                self.wave_engine.step(dt=0.1)

            # Detect patterns
            constructive, destructive = self.wave_engine.get_interference_patterns()
            resonances = self.wave_engine.get_resonances()

            result.constructive_patterns = constructive
            result.destructive_patterns = destructive
            result.resonances = resonances
            result.wave_mechanics_ms = (time.time() - wave_start) * 1000

        # ================================================================
        # Unified Metrics
        # ================================================================

        # Compute total energy (sum of all subsystems)
        total_energy = 0.0
        if result.routing_loss:
            total_energy += result.routing_loss
        if result.free_energy:
            total_energy += abs(result.free_energy)

        # Compute total entropy (thermodynamics)
        total_entropy = 0.0
        if self.enable_thermodynamics:
            stats = self.thermo_optimizer.get_statistics()
            total_entropy = stats["entropy"]

        # Total free energy
        if result.exploration_temperature:
            total_free_energy = total_energy - result.exploration_temperature * total_entropy
        else:
            total_free_energy = total_energy

        result.total_energy = total_energy
        result.total_entropy = total_entropy
        result.total_free_energy = total_free_energy

        # Duration
        result.duration_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.total_queries += 1
        self.total_energy_history.append(total_energy)

        return result

    def get_statistics(self) -> Dict:
        """Get unified physics statistics."""
        stats = {
            "total_queries": self.total_queries,
            "average_energy": (
                sum(self.total_energy_history) / len(self.total_energy_history)
                if self.total_energy_history else 0.0
            ),
            "enabled_systems": {
                "routing": self.enable_routing,
                "packing": self.enable_packing,
                "thermodynamics": self.enable_thermodynamics,
                "wave_mechanics": self.enable_wave_mechanics
            }
        }

        # Add component statistics
        if self.enable_thermodynamics:
            stats["thermodynamics"] = self.thermo_optimizer.get_statistics()

        if self.enable_wave_mechanics:
            stats["wave_mechanics"] = self.wave_engine.get_statistics()

        return stats


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "UnifiedPhysicsEngine",
    "UnifiedPhysicsResult",
    "PhysicsMode",
]
