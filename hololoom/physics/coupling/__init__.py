"""
Physics Coupling — 6 engines in a feedback ring.

Spring -> Gradient -> Fluid -> Thermo -> Wave -> StatMech -> Spring.
The loop closes. Meta-thermodynamics: the system learns its own temperature.
Waves propagate at every scale.
"""
from .protocol import PhysicsPhase, PhysicsState
from .coupler import PhysicsCoupler, get_coupler
from .meta_thermo import MetaThermodynamics
from .wave_hierarchy import WaveHierarchy, BoundaryEvent

__all__ = [
    "PhysicsPhase",
    "PhysicsState",
    "PhysicsCoupler",
    "get_coupler",
    "MetaThermodynamics",
    "WaveHierarchy",
    "BoundaryEvent",
]
