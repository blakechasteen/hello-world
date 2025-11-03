"""
HoloLoom Self-Tuning System
============================

Adaptive parameter optimization using Thompson Sampling bandits.

Philosophy: "The system configures itself based on what actually works."

Architecture:
- 7 specialized tuning agents (TimeoutTuner, CacheTuner, etc.)
- Master coordinator using meta-bandit for agent selection
- Thompson Sampling for parameter exploration
- Safe tuning with rollback guarantees
- Persistence across sessions

Usage:
    from HoloLoom.tuning import MasterTuningCoordinator

    coordinator = MasterTuningCoordinator()
    await coordinator.run_tuning_cycle()
"""

from HoloLoom.tuning.base import TuningAgent, ThompsonBandit
from HoloLoom.tuning.coordinator import MasterTuningCoordinator
from HoloLoom.tuning.timeout_tuner import TimeoutTuner
from HoloLoom.tuning.cache_tuner import CacheTuner
from HoloLoom.tuning.threshold_tuner import ThresholdTuner
from HoloLoom.tuning.memory_tuner import MemoryTuner
from HoloLoom.tuning.complexity_tuner import ComplexityTuner
from HoloLoom.tuning.policy_tuner import PolicyTuner
from HoloLoom.tuning.physics_tuner import PhysicsTuner
from HoloLoom.tuning.persistence import TuningStateManager

__all__ = [
    'TuningAgent',
    'ThompsonBandit',
    'MasterTuningCoordinator',
    'TimeoutTuner',
    'CacheTuner',
    'ThresholdTuner',
    'MemoryTuner',
    'ComplexityTuner',
    'PolicyTuner',
    'PhysicsTuner',
    'TuningStateManager',
]
