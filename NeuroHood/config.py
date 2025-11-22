"""
NeuroHood Configuration

Configuration for consciousness-level neighborhood simulation.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SimulationMode(Enum):
    """Simulation complexity modes."""
    PROTOTYPE = "prototype"     # 3 residents, simplified physics
    NEIGHBORHOOD = "neighborhood"  # 20 residents, full features
    COMMUNITY = "community"     # 100+ residents, emergent phenomena


@dataclass
class NeuroHoodConfig:
    """Configuration for NeuroHood simulator."""

    # World settings
    world_id: str = "maple_street"
    save_path: Optional[str] = "./neurohood_saves"
    auto_save: bool = True
    auto_save_interval: int = 300  # 5 minutes

    # Simulation settings
    simulation_mode: SimulationMode = SimulationMode.PROTOTYPE
    num_residents: int = 3  # Overridden by simulation_mode
    num_houses: int = 20
    time_scale: float = 1.0  # 1.0 = real-time, 60.0 = 1 minute = 1 hour

    # Consciousness settings
    enable_internal_dialogue: bool = True
    enable_strange_loops: bool = True
    enable_meta_awareness: bool = True
    internal_dialogue_depth: int = 5  # Max recursion depth
    strange_loop_threshold: float = 0.7  # Strength threshold for detection

    # Social physics
    enable_spring_dynamics: bool = True
    relationship_stiffness: float = 0.8  # Strong connections
    relationship_damping: float = 0.85  # Gradual forgetting
    relationship_decay: float = 0.99  # Very slow natural decay
    activation_spread_enabled: bool = True

    # Learning settings
    enable_background_learning: bool = True
    learning_update_interval: float = 60.0  # 1 minute
    enable_memory_consolidation: bool = True
    consolidation_interval: float = 3600.0  # 1 hour (simulated "sleep")

    # Agent settings
    agent_loop_interval: float = 60.0  # Agents "think" every 60 seconds
    max_conversation_budget: int = 20  # Max messages per conversation
    enable_policy_governance: bool = True  # Social hierarchy/permissions

    # Causal reasoning
    enable_causal_scm: bool = True  # Experimental
    causal_window_size: int = 20  # Look back N events

    # Performance
    max_concurrent_agents: int = 100
    enable_agent_pooling: bool = True
    memory_consolidation_threshold: int = 1000  # Consolidate when >1000 memories

    # Visualization
    default_consciousness_mode: str = "normal"  # "normal", "thoughts", "strange_loops", "full_dmt"
    show_personality_vectors: bool = False
    show_activation_spreading: bool = False
    show_causal_chains: bool = False

    # HoloLoom integration
    hololoom_mode: str = "fused"  # "bare", "fast", "fused"
    enable_thompson_sampling: bool = True
    exploration_epsilon: float = 0.15

    # Safety
    enable_content_filtering: bool = True
    max_simulation_time: Optional[float] = None  # Max in-world time (None = unlimited)
    enable_rollback: bool = True  # Can rewind time

    @classmethod
    def prototype(cls) -> "NeuroHoodConfig":
        """Quick prototype config (3 residents, 1 street)."""
        return cls(
            world_id="prototype_street",
            simulation_mode=SimulationMode.PROTOTYPE,
            num_residents=3,
            num_houses=5,
            enable_causal_scm=False,  # Simplify for prototype
        )

    @classmethod
    def neighborhood(cls) -> "NeuroHoodConfig":
        """Full neighborhood (20 residents, complete features)."""
        return cls(
            world_id="maple_street",
            simulation_mode=SimulationMode.NEIGHBORHOOD,
            num_residents=20,
            num_houses=20,
            enable_causal_scm=True,
        )

    @classmethod
    def community(cls) -> "NeuroHoodConfig":
        """Large-scale community (100+ residents, emergent phenomena)."""
        return cls(
            world_id="downtown_district",
            simulation_mode=SimulationMode.COMMUNITY,
            num_residents=100,
            num_houses=50,
            enable_agent_pooling=True,
            max_concurrent_agents=100,
        )
