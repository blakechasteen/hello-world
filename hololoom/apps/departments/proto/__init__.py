"""
Proto - General Purpose Code Agent
===================================
A relaxed, context-aware code assistant that integrates
deeply with HoloLoom's infrastructure.

Quick Start:
    from hololoom.apps.departments.proto import ProtoEngine, ProtoConfig

    async with ProtoEngine(ProtoConfig.default()) as proto:
        response = await proto.process("explain this code")
        print(response.content)

CLI Usage:
    proto ask "explain this function"
    proto repl
    proto review path/to/file.py

Architecture:
    Proto follows the thin waist pattern - all requests flow
    through ProtoEngine.process() which handles:
    - Intent parsing (what does the user want?)
    - Action selection (which ability to use?)
    - Execution (run via ability or agentic reasoning)
    - Response generation (format with personality)

Abilities:
    Proto supports three tiers of abilities:
    - Tier 1: Skill Mapping (wraps HoloLoom's skills)
    - Tier 2: Plugin Protocol (typed interface with permissions)
    - Tier 3: Full Sandbox (container/process isolation)

Integration:
    - AgenticBridge: Wrapper around HoloLoom agentic reasoning
    - ProtoDepartment: First-class HoloLoom department
    - Department Registry: Auto-discovery and orchestration

Author: HoloLoom Team
Version: 0.1.0
Status: Production Ready (2025-12-02)
"""

__version__ = "0.1.0"

# Core
from hololoom.apps.departments.proto.core import ProtoEngine, ProtoConfig, ProtoMode

# Domain
from hololoom.apps.departments.proto.domain import (
    ProtoIntent,
    ProtoAction,
    ProtoResponse,
    CodeContext,
    IntentType,
    ProtoSession,
    ProtoEventType,
    ProtoEvent,
)

# Abilities
from hololoom.apps.departments.proto.abilities import (
    Ability,
    BaseAbility,
    AbilityManifest,
    AbilityRegistry,
    AbilityTrustLevel,
    AbilityTier,
    AbilityContext,
    AbilityResult,
    PreflightResult,
    VerificationResult,
)

# Integration
from hololoom.apps.departments.proto.integration import (
    AgenticBridge,
    ProtoReasoningMode,
    AgenticBridgeResult,
    ProtoDepartment,
)

__all__ = [
    # Version
    "__version__",

    # Core
    "ProtoEngine",
    "ProtoConfig",
    "ProtoMode",

    # Domain
    "ProtoIntent",
    "ProtoAction",
    "ProtoResponse",
    "CodeContext",
    "IntentType",
    "ProtoSession",
    "ProtoEventType",
    "ProtoEvent",

    # Abilities
    "Ability",
    "BaseAbility",
    "AbilityManifest",
    "AbilityRegistry",
    "AbilityTrustLevel",
    "AbilityTier",
    "AbilityContext",
    "AbilityResult",
    "PreflightResult",
    "VerificationResult",

    # Integration
    "AgenticBridge",
    "ProtoReasoningMode",
    "AgenticBridgeResult",
    "ProtoDepartment",
]
