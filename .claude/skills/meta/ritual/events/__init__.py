"""
Ritual Events - Re-exports from EventBus ritual_events module.

Created: December 30, 2025
Purpose: Provides convenient access to ritual event helpers for phase handlers.

Usage:
    from events import RitualEventType, create_ritual_event, phase_completed

The event_bus module provides:
- RitualEventType enum: Event types (RITUAL_STARTED, PHASE_COMPLETED, etc.)
- RitualContext: Tracks workflow state with correlation_id
- Event factory functions: ritual_started(), phase_completed(), etc.

All events in a ritual workflow share the same correlation_id, enabling
workflow tracking and replay through the EventBus.
"""

import sys
from pathlib import Path

# Add parent paths to find event_bus module
_skills_root = Path(__file__).parent.parent.parent.parent
if str(_skills_root) not in sys.path:
    sys.path.insert(0, str(_skills_root))

# Try to import from event_bus, provide fallback if not available
try:
    from event_bus.ritual_events import (
        # Enums
        RitualEventType,
        RitualPhase,
        DecisionType,

        # Data structures
        RitualContext,
        RitualDecision,

        # Decision constants
        PHASE_DECISION_OPTIONS,

        # Event creation
        create_ritual_event,

        # Event factories - Ritual lifecycle
        ritual_started,
        ritual_completed,
        ritual_cancelled,
        ritual_paused,
        ritual_resumed,

        # Event factories - Phase lifecycle
        phase_started,
        phase_completed,
        phase_skipped,
        phase_failed,

        # Event factories - Decisions
        decision_required,
        decision_made,

        # Event factories - Memory
        memory_stored,
        context_restored,
        context_handoff,

        # Event factories - Learning
        learning_update,
    )

    EVENT_BUS_AVAILABLE = True

except ImportError:
    # Provide stub implementations if event_bus not available
    # This ensures the ritual skill works standalone
    EVENT_BUS_AVAILABLE = False

    from enum import Enum
    from dataclasses import dataclass, field
    from typing import Dict, List, Any, Optional
    from datetime import datetime
    import uuid

    class RitualPhase(Enum):
        OPEN = "open"
        DESIGN = "design"
        BUILD = "build"
        CHECK = "check"
        CLOSE = "close"

    class RitualEventType(Enum):
        RITUAL_STARTED = "ritual.started"
        RITUAL_COMPLETED = "ritual.completed"
        RITUAL_CANCELLED = "ritual.cancelled"
        RITUAL_PAUSED = "ritual.paused"
        RITUAL_RESUMED = "ritual.resumed"
        PHASE_STARTED = "ritual.phase.started"
        PHASE_COMPLETED = "ritual.phase.completed"
        PHASE_SKIPPED = "ritual.phase.skipped"
        PHASE_FAILED = "ritual.phase.failed"
        DECISION_REQUIRED = "ritual.decision.required"
        DECISION_MADE = "ritual.decision.made"
        MEMORY_STORED = "ritual.memory.stored"
        CONTEXT_RESTORED = "ritual.context.restored"
        CONTEXT_HANDOFF = "ritual.context.handoff"
        LEARNING_UPDATE = "ritual.learning.update"

    class DecisionType(Enum):
        PROCEED = "proceed"
        REFINE = "refine"
        SKIP = "skip"
        EXPLORE = "explore"
        APPROVE = "approve"
        CONFIRM = "confirm"

    @dataclass
    class RitualDecision:
        decision_type: DecisionType
        decision_text: str
        phase: RitualPhase
        timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
        confidence: float = 0.5
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class RitualContext:
        ritual_id: str = field(default_factory=lambda: f"ritual-{uuid.uuid4().hex[:8]}")
        feature_name: str = ""
        started_at: str = field(default_factory=lambda: datetime.now().isoformat())
        current_phase: RitualPhase = RitualPhase.OPEN
        phase_results: Dict[str, Any] = field(default_factory=dict)
        decisions_made: List[Dict[str, Any]] = field(default_factory=list)
        memories_stored: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    PHASE_DECISION_OPTIONS: Dict[RitualPhase, List[str]] = {}

    # Stub event functions that do nothing (no-op when EventBus unavailable)
    def create_ritual_event(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def ritual_started(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def ritual_completed(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def ritual_cancelled(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def ritual_paused(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def ritual_resumed(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def phase_started(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def phase_completed(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def phase_skipped(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def phase_failed(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def decision_required(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def decision_made(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def memory_stored(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def context_restored(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def context_handoff(*args, **kwargs) -> Dict[str, Any]:
        return {}

    def learning_update(*args, **kwargs) -> Dict[str, Any]:
        return {}


__all__ = [
    # Availability flag
    'EVENT_BUS_AVAILABLE',

    # Enums
    'RitualEventType',
    'RitualPhase',
    'DecisionType',

    # Data structures
    'RitualContext',
    'RitualDecision',

    # Decision constants
    'PHASE_DECISION_OPTIONS',

    # Event creation
    'create_ritual_event',

    # Event factories - Ritual lifecycle
    'ritual_started',
    'ritual_completed',
    'ritual_cancelled',
    'ritual_paused',
    'ritual_resumed',

    # Event factories - Phase lifecycle
    'phase_started',
    'phase_completed',
    'phase_skipped',
    'phase_failed',

    # Event factories - Decisions
    'decision_required',
    'decision_made',

    # Event factories - Memory
    'memory_stored',
    'context_restored',
    'context_handoff',

    # Event factories - Learning
    'learning_update',
]