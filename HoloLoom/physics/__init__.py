"""
HoloLoom Physics - Multi-Physics Optimization Systems

Implements physics-based optimization for intelligent decision-making:
    - Phase 0: Spring Physics (COMPLETE) - Graph retrieval via Hooke's Law
    - Phase 1: Gradient Flow (COMPLETE) - Natural load balancing
    - Phase 2: Fluid Dynamics (COMPLETE) - Adaptive context packing via Navier-Stokes

Future Phases:
    - Phase 3: Thermodynamics - Free energy minimization
    - Phase 4: Wave Mechanics - Pattern detection
    - Phase 5: Statistical Mechanics - Emergence
    - Phase 6: Unified Physics Engine

See PHYSICS_INTEGRATION_ROADMAP.md for complete details.
"""

# Gradient Flow (Phase 1)
from .gradient_flow import (
    GradientFlowEngine,
    RouteDecision,
    Target,
    combined_loss,
    create_tool_selection_loss,
    load_loss,
    latency_loss,
    cost_loss,
    quality_loss
)

# Fluid Dynamics (Phase 2)
from .pressure_field import PressureField, PressureSource
from .velocity_field import VelocityField, FlowVector
from .fluid_dynamics import ContextFlowEngine, FlowState
from .adaptive_packer import AdaptivePacker, PackedContext

# Multi-Physics Integration (Phases 1+2)
from .multi_physics_packer import (
    MultiPhysicsPacker,
    MultiPhysicsResult,
    ComponentAllocation,
    pack_context_multiphysics
)

# Thermodynamics (Phase 3)
from .thermodynamics import (
    ThermodynamicOptimizer,
    TemperatureScheduler,
    EnergyCalculator,
    EntropyCalculator,
    CoolingSchedule,
    ThermodynamicState,
    TemperatureState
)

# Wave Mechanics (Phase 4)
from .wave_mechanics import (
    WaveMechanicsEngine,
    WaveField,
    InterferenceCalculator,
    ResonanceDetector,
    WaveState,
    InterferencePattern,
    ResonancePattern
)

# Unified Physics (All Phases Integrated)
from .unified_physics import (
    UnifiedPhysicsEngine,
    UnifiedPhysicsResult,
    PhysicsMode
)

# Statistical Mechanics (Phase 5)
from .statistical_mechanics import (
    StatisticalMechanicsEngine,
    CanonicalEnsemble,
    EntropyCalculator,
    PhaseTransitionDetector,
    Microstate,
    Macrostate,
    PhaseTransition,
    PhaseType,
    BOLTZMANN_CONSTANT
)

__all__ = [
    # Gradient Flow
    "GradientFlowEngine",
    "RouteDecision",
    "Target",
    "combined_loss",
    "create_tool_selection_loss",
    "load_loss",
    "latency_loss",
    "cost_loss",
    "quality_loss",

    # Pressure
    "PressureField",
    "PressureSource",

    # Velocity
    "VelocityField",
    "FlowVector",

    # Flow Engine
    "ContextFlowEngine",
    "FlowState",

    # Packer
    "AdaptivePacker",
    "PackedContext",

    # Multi-Physics
    "MultiPhysicsPacker",
    "MultiPhysicsResult",
    "ComponentAllocation",
    "pack_context_multiphysics",

    # Thermodynamics
    "ThermodynamicOptimizer",
    "TemperatureScheduler",
    "EnergyCalculator",
    "EntropyCalculator",
    "CoolingSchedule",
    "ThermodynamicState",
    "TemperatureState",

    # Wave Mechanics
    "WaveMechanicsEngine",
    "WaveField",
    "InterferenceCalculator",
    "ResonanceDetector",
    "WaveState",
    "InterferencePattern",
    "ResonancePattern",

    # Unified Physics
    "UnifiedPhysicsEngine",
    "UnifiedPhysicsResult",
    "PhysicsMode",

    # Statistical Mechanics
    "StatisticalMechanicsEngine",
    "CanonicalEnsemble",
    "EntropyCalculator",
    "PhaseTransitionDetector",
    "Microstate",
    "Macrostate",
    "PhaseTransition",
    "PhaseType",
    "BOLTZMANN_CONSTANT",
]
