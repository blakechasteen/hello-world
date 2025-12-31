"""
Ritual Events - EventBus integration for coding ritual workflow.

Created: December 30, 2025
Purpose: Event types, ritual context, and event factory functions

Provides the event infrastructure for the 5-phase coding ritual:
AWAKEN -> PLAN -> IMPLEMENT -> REVIEW -> REFLECT
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


# ============================================================================
# Ritual Phases
# ============================================================================

class RitualPhase(Enum):
    """The 5 phases of the coding ritual."""
    AWAKEN = "awaken"
    PLAN = "plan"
    IMPLEMENT = "implement"
    REVIEW = "review"
    REFLECT = "reflect"


# ============================================================================
# Event Types
# ============================================================================

class RitualEventType(Enum):
    """Types of events emitted during ritual execution."""
    # Ritual lifecycle
    RITUAL_STARTED = "ritual.started"
    RITUAL_COMPLETED = "ritual.completed"
    RITUAL_CANCELLED = "ritual.cancelled"
    RITUAL_PAUSED = "ritual.paused"
    RITUAL_RESUMED = "ritual.resumed"

    # Phase lifecycle
    PHASE_STARTED = "ritual.phase.started"
    PHASE_COMPLETED = "ritual.phase.completed"
    PHASE_SKIPPED = "ritual.phase.skipped"
    PHASE_FAILED = "ritual.phase.failed"

    # Decision points
    DECISION_REQUIRED = "ritual.decision.required"
    DECISION_MADE = "ritual.decision.made"
    DECISION_TIMEOUT = "ritual.decision.timeout"

    # Memory operations
    MEMORY_STORED = "ritual.memory.stored"
    MEMORY_RECALLED = "ritual.memory.recalled"
    CONTEXT_RESTORED = "ritual.context.restored"
    CONTEXT_HANDOFF = "ritual.context.handoff"

    # Learning events
    LEARNING_UPDATE = "ritual.learning.update"
    PATTERN_DETECTED = "ritual.pattern.detected"


# ============================================================================
# Ritual Context
# ============================================================================

@dataclass
class RitualContext:
    """
    Tracks state across the entire ritual workflow.

    Contains:
    - Unique ritual identifier
    - Feature being worked on
    - Current phase
    - Phase results from each completed phase
    - Decisions made during the ritual
    - Memories stored during the ritual
    """
    ritual_id: str = field(default_factory=lambda: f"ritual-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    feature_name: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_phase: RitualPhase = RitualPhase.AWAKEN
    phase_results: Dict[str, Any] = field(default_factory=dict)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    memories_stored: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_decision(
        self,
        phase: str,
        decision: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a decision made during the ritual.

        Args:
            phase: Phase where decision was made
            decision: The decision text
            details: Optional additional details
        """
        self.decisions_made.append({
            "phase": phase,
            "decision": decision,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })

    def record_phase_result(
        self,
        phase: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Record the result of a completed phase.

        Args:
            phase: Phase that completed
            result: Phase execution result
        """
        self.phase_results[phase] = {
            **result,
            "completed_at": datetime.now().isoformat()
        }

    def record_memory(self, memory_id: str) -> None:
        """
        Record a memory stored during the ritual.

        Args:
            memory_id: ID of the stored memory
        """
        self.memories_stored.append(memory_id)

    def get_duration_seconds(self) -> float:
        """Get the duration of the ritual in seconds."""
        started = datetime.fromisoformat(self.started_at)
        return (datetime.now() - started).total_seconds()

    def to_summary(self) -> str:
        """Generate a human-readable summary of the ritual."""
        duration = self.get_duration_seconds()
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        return f"""
Ritual: {self.ritual_id}
Feature: {self.feature_name}
Current Phase: {self.current_phase.value}
Duration: {minutes}m {seconds}s
Phases Completed: {len(self.phase_results)}
Decisions Made: {len(self.decisions_made)}
Memories Stored: {len(self.memories_stored)}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "ritual_id": self.ritual_id,
            "feature_name": self.feature_name,
            "started_at": self.started_at,
            "current_phase": self.current_phase.value,
            "phase_results": self.phase_results,
            "decisions_made": self.decisions_made,
            "memories_stored": self.memories_stored,
            "metadata": self.metadata,
            "duration_seconds": self.get_duration_seconds(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RitualContext':
        """Deserialize context from dictionary."""
        ctx = cls(
            ritual_id=data.get("ritual_id", ""),
            feature_name=data.get("feature_name", ""),
            started_at=data.get("started_at", datetime.now().isoformat()),
            current_phase=RitualPhase(data.get("current_phase", "awaken")),
            phase_results=data.get("phase_results", {}),
            decisions_made=data.get("decisions_made", []),
            memories_stored=data.get("memories_stored", []),
            metadata=data.get("metadata", {}),
        )
        return ctx


# ============================================================================
# Event Creation
# ============================================================================

def create_ritual_event(
    event_type: RitualEventType,
    context: RitualContext,
    payload: Optional[Dict[str, Any]] = None,
    causation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a ritual event for EventBus emission.

    Args:
        event_type: Type of event
        context: Current ritual context
        payload: Event-specific payload
        causation_id: ID of event that caused this one

    Returns:
        Event dictionary ready for emission
    """
    return {
        "event_type": event_type.value,
        "skill_name": "ritual",
        "timestamp": datetime.now().isoformat(),
        "event_id": str(uuid.uuid4()),
        "correlation_id": context.ritual_id,  # Links all events in workflow
        "causation_id": causation_id,
        "payload": {
            "ritual_id": context.ritual_id,
            "feature_name": context.feature_name,
            "phase": context.current_phase.value,
            **(payload or {})
        }
    }


# ============================================================================
# Event Factory Functions
# ============================================================================

def ritual_started(context: RitualContext) -> Dict[str, Any]:
    """Create a RITUAL_STARTED event."""
    return create_ritual_event(
        RitualEventType.RITUAL_STARTED,
        context,
        {"started_at": context.started_at}
    )


def ritual_completed(context: RitualContext) -> Dict[str, Any]:
    """Create a RITUAL_COMPLETED event."""
    return create_ritual_event(
        RitualEventType.RITUAL_COMPLETED,
        context,
        {
            "summary": context.to_summary(),
            "duration_seconds": context.get_duration_seconds(),
            "phases_completed": list(context.phase_results.keys()),
            "decisions_count": len(context.decisions_made),
            "memories_count": len(context.memories_stored),
        }
    )


def ritual_cancelled(
    context: RitualContext,
    reason: str = ""
) -> Dict[str, Any]:
    """Create a RITUAL_CANCELLED event."""
    return create_ritual_event(
        RitualEventType.RITUAL_CANCELLED,
        context,
        {
            "reason": reason,
            "partial_progress": context.to_dict(),
        }
    )


def ritual_paused(context: RitualContext) -> Dict[str, Any]:
    """Create a RITUAL_PAUSED event."""
    return create_ritual_event(
        RitualEventType.RITUAL_PAUSED,
        context,
        {"state": context.to_dict()}
    )


def ritual_resumed(context: RitualContext) -> Dict[str, Any]:
    """Create a RITUAL_RESUMED event."""
    return create_ritual_event(
        RitualEventType.RITUAL_RESUMED,
        context,
        {"state": context.to_dict()}
    )


def phase_started(
    context: RitualContext,
    phase: RitualPhase
) -> Dict[str, Any]:
    """Create a PHASE_STARTED event."""
    context.current_phase = phase
    return create_ritual_event(
        RitualEventType.PHASE_STARTED,
        context,
        {"phase": phase.value}
    )


def phase_completed(
    context: RitualContext,
    phase: RitualPhase,
    result: Optional[Dict[str, Any]] = None,
    confidence: float = 0.5
) -> Dict[str, Any]:
    """Create a PHASE_COMPLETED event."""
    if result:
        context.record_phase_result(phase.value, result)

    return create_ritual_event(
        RitualEventType.PHASE_COMPLETED,
        context,
        {
            "phase": phase.value,
            "success": True,
            "confidence": confidence,
            "result": result,
        }
    )


def phase_skipped(
    context: RitualContext,
    phase: RitualPhase,
    reason: str = ""
) -> Dict[str, Any]:
    """Create a PHASE_SKIPPED event."""
    return create_ritual_event(
        RitualEventType.PHASE_SKIPPED,
        context,
        {
            "phase": phase.value,
            "reason": reason,
        }
    )


def phase_failed(
    context: RitualContext,
    phase: RitualPhase,
    error: str = ""
) -> Dict[str, Any]:
    """Create a PHASE_FAILED event."""
    return create_ritual_event(
        RitualEventType.PHASE_FAILED,
        context,
        {
            "phase": phase.value,
            "error": error,
        }
    )


def decision_required(
    context: RitualContext,
    options: List[str],
    suggested: Optional[str] = None
) -> Dict[str, Any]:
    """Create a DECISION_REQUIRED event."""
    return create_ritual_event(
        RitualEventType.DECISION_REQUIRED,
        context,
        {
            "options": options,
            "phase": context.current_phase.value,
            "suggested": suggested,
        }
    )


def decision_made(
    context: RitualContext,
    decision: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a DECISION_MADE event."""
    context.record_decision(context.current_phase.value, decision, details)
    return create_ritual_event(
        RitualEventType.DECISION_MADE,
        context,
        {
            "decision": decision,
            "details": details,
        }
    )


def memory_stored(
    context: RitualContext,
    memory_id: str,
    content_preview: str = ""
) -> Dict[str, Any]:
    """Create a MEMORY_STORED event."""
    context.record_memory(memory_id)
    return create_ritual_event(
        RitualEventType.MEMORY_STORED,
        context,
        {
            "memory_id": memory_id,
            "content_preview": content_preview[:100] if content_preview else "",
        }
    )


def context_restored(
    context: RitualContext,
    memories_count: int,
    context_summary: str = ""
) -> Dict[str, Any]:
    """Create a CONTEXT_RESTORED event."""
    return create_ritual_event(
        RitualEventType.CONTEXT_RESTORED,
        context,
        {
            "memories_count": memories_count,
            "context_summary": context_summary,
        }
    )


def context_handoff(
    context: RitualContext,
    from_phase: str,
    to_phase: str,
    items_kept: int,
    items_total: int
) -> Dict[str, Any]:
    """Create a CONTEXT_HANDOFF event."""
    return create_ritual_event(
        RitualEventType.CONTEXT_HANDOFF,
        context,
        {
            "from_phase": from_phase,
            "to_phase": to_phase,
            "items_kept": items_kept,
            "items_total": items_total,
            "keep_ratio": items_kept / items_total if items_total > 0 else 0.0,
        }
    )


def learning_update(
    context: RitualContext,
    agent_id: str,
    success: bool,
    confidence: float
) -> Dict[str, Any]:
    """Create a LEARNING_UPDATE event."""
    return create_ritual_event(
        RitualEventType.LEARNING_UPDATE,
        context,
        {
            "agent_id": agent_id,
            "success": success,
            "confidence": confidence,
        }
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'RitualPhase',
    'RitualEventType',

    # Data structures
    'RitualContext',

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
