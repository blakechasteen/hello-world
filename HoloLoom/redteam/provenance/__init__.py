"""Attack provenance tracking for CARTS (red team system)."""

from .attack_scratchpad import (
    AttackScratchpad,
    AttackScratchpadEntry,
    AttackChain,
    AttackStrategy,
    DefenseLayer,
)

__all__ = [
    'AttackScratchpad',
    'AttackScratchpadEntry',
    'AttackChain',
    'AttackStrategy',
    'DefenseLayer',
]
