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

    coordinator = MasterTuningCoordinator()
    await coordinator.run_tuning_cycle()
"""

from HoloLoom.tuning.coordinator import MasterTuningCoordinator

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
