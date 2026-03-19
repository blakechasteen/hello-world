"""
Loom - Multi-Loom Architecture
===============================

The Loom package provides the complete Multi-Loom Weave House:

1. **Pattern Card Selector** (command.py)
   - LoomCommand: Main pattern selector
   - PatternCard: Enum of pattern cards (BARE/FAST/FUSED)
   - PatternSpec: Pattern specification dataclass

2. **Multi-Loom Protocol** (protocol.py, base_loom.py) - December 2025
   - Loom Protocol: A coherent way of seeing (extends Department)
   - DreamInsight: Insights shared during collective dreaming
   - BaseLoom: Base implementation with DreamingMixin

3. **WeaveHouse** (weave_house.py) - December 2025
   - WeaveHouse: Composite loom with parallel weaving and auto-exploration
   - WeaveResult: Complete result of a weaving cycle
   - ExplorationResult: Result of exploring tension zones

4. **Dreaming** (dreaming.py) - December 2025
   - DreamOrchestrator: Manages collective dreaming with bleed rates
   - DreamCycleStats: Statistics from a dream cycle

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
# Base Implementation (December 2025)
from .base_loom import (
    BaseLoom,
    DreamingMixin,
)
from .command import (
    BARE_PATTERN,
    FAST_PATTERN,
    FUSED_PATTERN,
    LoomCommand,
    PatternCard,
    PatternSpec,
    create_loom_command,
)

# Consensus Engine (December 2025)
from .consensus import (
    Agreement,
    Claim,
    LoomConsensus,
    create_loom_consensus,
)

# Dreaming - Collective Consolidation (December 2025)
from .dreaming import (
    DreamCycleStats,
    DreamOrchestrator,
    DreamOrchestratorMetrics,
    create_dream_orchestrator,
)

# Multi-Loom Protocol (December 2025)
from .protocol import (
    REACH,
    REASON,
    # Perspective Constants
    RECALL,
    REFLECT,
    REFUSE,
    CorrectionInsight,
    DiscoveryInsight,
    DreamInsight,
    # Insight Types
    InsightType,
    # Core Protocol
    Loom,
    MathInsight,
    PatternInsight,
)

# WeaveHouse - Composite Loom (December 2025)
from .weave_house import (
    ExplorationResult,
    WeaveHouse,
    WeaveResult,
    create_weave_house,
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

    # ===== WeaveHouse =====
    "WeaveHouse",
    "WeaveResult",
    "ExplorationResult",
    "create_weave_house",

    # ===== Dreaming =====
    "DreamCycleStats",
    "DreamOrchestratorMetrics",
    "DreamOrchestrator",
    "create_dream_orchestrator",
]
