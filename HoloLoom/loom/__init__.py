"""
Loom - Multi-Loom Architecture
===============================

The Loom package provides two complementary capabilities:

1. **Pattern Card Selector** (command.py)
   - LoomCommand: Main pattern selector
   - PatternCard: Enum of pattern cards (BARE/FAST/FUSED)
   - PatternSpec: Pattern specification dataclass

2. **Multi-Loom Weave House** (protocol.py, base_loom.py) - December 2025
   - Loom Protocol: A coherent way of seeing (extends Department)
   - DreamInsight: Insights shared during collective dreaming
   - BaseLoom: Base implementation with DreamingMixin

The Five Core Looms:
- RECALL (The Librarian): Dense retrieval, memory-grounded
- REASON (The Logician): Causal inference, logic-grounded
- REACH (The Artist): Lateral association, creativity-grounded
- REFLECT (The Auditor): Consistency checking, verification-grounded
- REFUSE (The Skeptic): Adversarial probing, breaking-grounded

Philosophy:
"Each loom is a coherent way of seeing. Different looms see different truths.
Together, they weave a fabric richer than any single perspective."
"""

# Pattern Card Selector (existing)
from .command import (
    LoomCommand,
    PatternCard,
    PatternSpec,
    BARE_PATTERN,
    FAST_PATTERN,
    FUSED_PATTERN,
    create_loom_command
)

# Multi-Loom Protocol (December 2025)
from .protocol import (
    # Core Protocol
    Loom,

    # Insight Types
    InsightType,
    DreamInsight,
    MathInsight,
    PatternInsight,
    CorrectionInsight,
    DiscoveryInsight,

    # Perspective Constants
    RECALL,
    REASON,
    REACH,
    REFLECT,
    REFUSE,
)

# Base Implementation (December 2025)
from .base_loom import (
    DreamingMixin,
    BaseLoom,
)

# Consensus Engine (December 2025)
from .consensus import (
    LoomConsensus,
    Claim,
    Agreement,
    create_loom_consensus,
)

__all__ = [
    # ===== Pattern Card Selector =====
    "LoomCommand",
    "PatternCard",
    "PatternSpec",
    "BARE_PATTERN",
    "FAST_PATTERN",
    "FUSED_PATTERN",
    "create_loom_command",

    # ===== Multi-Loom Protocol =====
    "Loom",
    "InsightType",
    "DreamInsight",
    "MathInsight",
    "PatternInsight",
    "CorrectionInsight",
    "DiscoveryInsight",

    # ===== Perspective Constants =====
    "RECALL",
    "REASON",
    "REACH",
    "REFLECT",
    "REFUSE",

    # ===== Base Implementation =====
    "DreamingMixin",
    "BaseLoom",

    # ===== Consensus Engine =====
    "LoomConsensus",
    "Claim",
    "Agreement",
    "create_loom_consensus",
]
