"""
Physics Coupling Protocol — shared state and phase contract.

Every physics engine implements PhysicsPhase: step(state) -> state.
State flows through phases in a ring. The ring closes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class PhysicsState:
    """Shared state vector flowing between coupled phases.

    Scalar fields describe the system as a whole.
    Sparse dicts describe per-node state (keyed by node ID).
    """

    # Thermodynamic scalars
    temperature: float = 1.0        # T — current system temperature
    entropy: float = 0.0            # S — system entropy
    free_energy: float = 0.0        # F = E - TS (Helmholtz)
    energy: float = 0.0             # E — total internal energy
    order_parameter: float = 0.0    # [0, 1] — system organization

    # Mechanical scalars
    tension: float = 0.0            # Mean spring tension
    wave_energy: float = 0.0        # Total wave field energy
    pressure: float = 0.0           # Mean fluid pressure
    gradient_loss: float = 0.0      # Current routing loss

    # Per-node state (sparse — only non-zero entries)
    activations: dict[str, float] = field(default_factory=dict)
    velocities: dict[str, float] = field(default_factory=dict)
    amplitudes: dict[str, float] = field(default_factory=dict)
    pressures: dict[str, float] = field(default_factory=dict)

    # Time
    timestep: int = 0
    dt: float = 0.01

    # Routing (from gradient phase)
    routing_target: str = ""
    routing_confidence: float = 0.0

    # Phase transition info (from statmech)
    phase_type: str = "disordered"  # disordered, critical, ordered

    def copy(self) -> PhysicsState:
        """Shallow copy with independent dicts."""
        return PhysicsState(
            temperature=self.temperature,
            entropy=self.entropy,
            free_energy=self.free_energy,
            energy=self.energy,
            order_parameter=self.order_parameter,
            tension=self.tension,
            wave_energy=self.wave_energy,
            pressure=self.pressure,
            gradient_loss=self.gradient_loss,
            activations=dict(self.activations),
            velocities=dict(self.velocities),
            amplitudes=dict(self.amplitudes),
            pressures=dict(self.pressures),
            timestep=self.timestep,
            dt=self.dt,
            routing_target=self.routing_target,
            routing_confidence=self.routing_confidence,
            phase_type=self.phase_type,
        )

    def to_dict(self) -> dict:
        """Serialize scalars for logging/viz."""
        return {
            "timestep": self.timestep,
            "temperature": round(self.temperature, 4),
            "entropy": round(self.entropy, 4),
            "free_energy": round(self.free_energy, 4),
            "energy": round(self.energy, 4),
            "order_parameter": round(self.order_parameter, 4),
            "tension": round(self.tension, 4),
            "wave_energy": round(self.wave_energy, 4),
            "pressure": round(self.pressure, 4),
            "gradient_loss": round(self.gradient_loss, 4),
            "phase_type": self.phase_type,
            "n_active_nodes": len(self.activations),
        }


@runtime_checkable
class PhysicsPhase(Protocol):
    """A single phase in the coupled physics loop.

    Each engine adapter implements this. The coupler calls step()
    on each phase in ring order. State flows through.
    """

    name: str

    def step(self, state: PhysicsState) -> PhysicsState:
        """Advance this phase one timestep. Return updated state."""
        ...

    def available(self) -> bool:
        """Whether this phase's engine is importable and initialized."""
        ...
