#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Engine - Layer 6
===========================
Multi-step reasoning layer for HoloLoom 1.1.

Phase 1: FAST and STANDARD modes (Foundation)
Phase 2: WeavingOrchestrator integration
Phase 3: DEEP mode with planning and backtracking

Author: Claude Code
Date: 2025-11-15
"""

from HoloLoom.reasoning.types import (
    # Enums
    ReasoningMode,
    StepType,
    QueryType,
    VerificationSeverity,

    # Data classes
    QueryIntent,
    ReasoningStep,
    ReasoningResult,
    VerificationResult,
    MultiPassVerification,
    EvidencePiece,
    Synthesis,
    QueryPlan,
    PlanStep,

    # Constants
    DEFAULT_CONFIDENCE_THRESHOLDS,
    STEP_TYPE_ICONS,
    QUERY_TYPE_COMPLEXITY,
)

from HoloLoom.reasoning.planner import QueryPlanner
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.engine import (
    ReasoningEngine,
    reason_with_mode,
    auto_reason,
)


__all__ = [
    # Enums
    "ReasoningMode",
    "StepType",
    "QueryType",
    "VerificationSeverity",

    # Data classes
    "QueryIntent",
    "ReasoningStep",
    "ReasoningResult",
    "VerificationResult",
    "MultiPassVerification",
    "EvidencePiece",
    "Synthesis",
    "QueryPlan",
    "PlanStep",

    # Components
    "QueryPlanner",
    "ChainOfThought",
    "SelfVerifier",
    "ReasoningEngine",

    # Convenience functions
    "reason_with_mode",
    "auto_reason",

    # Constants
    "DEFAULT_CONFIDENCE_THRESHOLDS",
    "STEP_TYPE_ICONS",
    "QUERY_TYPE_COMPLEXITY",
]


# Version info
__version__ = "1.1.0-phase1"
__phase__ = "Phase 1: Foundation (FAST + STANDARD modes)"
