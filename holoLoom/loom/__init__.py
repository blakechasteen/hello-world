"""
Loom - Pattern Card Selector
=============================
The control system that selects execution templates.

Exports:
- LoomCommand: Main pattern selector
- PatternCard: Enum of pattern cards
- PatternSpec: Pattern specification dataclass
- BARE_PATTERN, FAST_PATTERN, FUSED_PATTERN: Pre-configured patterns
- create_loom_command: Factory function
"""

from .command import (
    LoomCommand,
    PatternCard,
    PatternSpec,
    BARE_PATTERN,
    FAST_PATTERN,
    FUSED_PATTERN,
    create_loom_command
)

__all__ = [
    "LoomCommand",
    "PatternCard",
    "PatternSpec",
    "BARE_PATTERN",
    "FAST_PATTERN",
    "FUSED_PATTERN",
    "create_loom_command"
]
