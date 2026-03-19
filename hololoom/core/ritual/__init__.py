"""
Ritual Grammar & Hook System
=============================

First-class, versioned, budget-aware behavioral units with identity,
failure policy, and memory footprint. The hook system binds rituals
to lifecycle events.

Core distinction:
  - Ritual  — what the behavior IS (identity, logic, budget, failure policy)
  - HookBinding — WHEN it fires (trigger, condition, priority)

These are separate classes. A ritual can be rebound to different triggers
without changing its identity.
"""

from .confidence import ConfidenceModel, ConfidenceSignal
from .hooks import HookEvent
from .policies import FailurePolicy, LiturgyFailurePolicy
from .registry import RitualDAGError, RitualRegistry
from .types import (
    HookBinding,
    Priority,
    Ritual,
    RitualBudget,
    RitualContext,
    RitualOutcome,
    RitualRecord,
    RitualResult,
    TokenBudget,
)

__all__ = [
    "Ritual",
    "HookBinding",
    "RitualContext",
    "RitualResult",
    "RitualOutcome",
    "RitualRecord",
    "RitualBudget",
    "TokenBudget",
    "Priority",
    "HookEvent",
    "FailurePolicy",
    "LiturgyFailurePolicy",
    "ConfidenceModel",
    "ConfidenceSignal",
    "RitualRegistry",
    "RitualDAGError",
]
