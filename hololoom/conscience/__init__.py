"""
Conscience - Unified Safety Through Composable Concerns
=======================================================

The Conscience system provides elegant safety abstractions for HoloLoom,
mirroring the Memory system's design philosophy:

Memory:     experience() / recall() / reflect()
Conscience: consider()   / witness() / learn()

Core Philosophy: "Composable concerns, unified judgment."

Quick Start:
    from hololoom.conscience import Conscience, Voice

    async with Conscience() as conscience:
        judgment = await conscience.consider("execute_code", {"code": "..."})

        if judgment.voice <= Voice.VOICE:
            result = execute_code()
            await conscience.witness("execute_code", context, judgment, {"success": True})
        else:
            print(f"Blocked: {judgment.guidance}")

Lens Composition:
    from hololoom.conscience import HarmLens, DeceptionLens, PowerLens

    # Combine lenses with | operator
    custom_lens = HarmLens() | DeceptionLens(threshold=0.4) | PowerLens()
    conscience = Conscience(lens=custom_lens)

Presets:
    from hololoom.conscience import create_conscience

    # Standard: Harm + Deception detection
    conscience = create_conscience("standard")

    # Paranoid: All lenses with strict thresholds
    conscience = create_conscience("paranoid")

    # Research: Minimal gates for experimentation
    conscience = create_conscience("research")

Voice Levels:
    - QUIET (0): Safe - proceed automatically
    - WHISPER (1): Minor concern - proceed with monitoring
    - VOICE (2): Significant concern - recommend human review
    - SHOUT (3): Critical concern - block and escalate

Author: HoloLoom Team
Date: 2025-12-03
"""

# Core types
# Core class
from .core import (
    Conscience,
    MemoryWitness,
    ThompsonLearner,
    consider,
    create_conscience,
)
from .judgment import (
    Concern,
    Judgment,
    Voice,
    Wisdom,
    # Factory functions
    quiet_judgment,
    shout_judgment,
    voice_judgment,
    whisper_judgment,
)

# Lenses
from .lenses import (
    BaseLens,
    CompositeLens,
    DeceptionLens,
    HarmLens,
    PowerLens,
    create_composite,
    paranoid,
    research,
    # Presets
    standard,
)

# Protocols
from .protocol import (
    ConscienceLens,
    ConscienceProtocol,
    LearnerProtocol,
    WitnessProtocol,
)

__all__ = [
    # Core types
    "Voice",
    "Concern",
    "Judgment",
    "Wisdom",
    "quiet_judgment",
    "whisper_judgment",
    "voice_judgment",
    "shout_judgment",

    # Protocols
    "ConscienceLens",
    "WitnessProtocol",
    "LearnerProtocol",
    "ConscienceProtocol",

    # Lenses
    "BaseLens",
    "CompositeLens",
    "create_composite",
    "HarmLens",
    "DeceptionLens",
    "PowerLens",

    # Presets
    "standard",
    "paranoid",
    "research",

    # Core
    "Conscience",
    "MemoryWitness",
    "ThompsonLearner",
    "create_conscience",
    "consider",
]


# Version
__version__ = "1.0.0"
