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
    from hololoom.tuning import MasterTuningCoordinator

    coordinator = MasterTuningCoordinator()
    await coordinator.run_tuning_cycle()
"""

from hololoom.tuning.base import ThompsonBandit, TuningAgent
from hololoom.tuning.cache_tuner import CacheTuner
from hololoom.tuning.complexity_tuner import ComplexityTuner
from hololoom.tuning.coordinator import MasterTuningCoordinator
from hololoom.tuning.memory_tuner import MemoryTuner
from hololoom.tuning.persistence import TuningStateManager
from hololoom.tuning.physics_tuner import PhysicsTuner
from hololoom.tuning.policy_tuner import PolicyTuner
from hololoom.tuning.threshold_tuner import ThresholdTuner
from hololoom.tuning.timeout_tuner import TimeoutTuner

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
