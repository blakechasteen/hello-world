"""
Ritual Events - EventBus integration for coding ritual workflow.

Created: December 30, 2025
Purpose: Event definitions and context tracking for the 5-phase coding ritual.

The ritual system orchestrates: AWAKEN → PLAN → IMPLEMENT → REVIEW → REFLECT
Each phase emits events through the EventBus for:
- Inter-phase communication
- Workflow tracking via correlation IDs
- Decision point coordination
- Learning and improvement

This module provides the foundation for both human-invoked and agent-invoked
ritual execution, following the "software for agents" paradigm.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


# ============================================================================
# Enums
# ============================================================================

class RitualPhase(Enum):
    """
    The 5 phases of the coding ritual.

    Each phase maps to an AgentCapability for agent-based orchestration:
    - AWAKEN → CONTEXT_RESTORATION
    - PLAN → PLANNING
    - IMPLEMENT → CODE_ASSISTANCE
    - REVIEW → QUALITY_ASSURANCE
    - REFLECT → KNOWLEDGE_CONSOLIDATION
    """
    AWAKEN = "awaken"       # Restore context from memory
    PLAN = "plan"           # Capture requirements, create plan
    IMPLEMENT = "implement" # Manual coding (assisted)
    REVIEW = "review"       # Quality check, capture learnings
    REFLECT = "reflect"     # End-of-session consolidation


class RitualEventType(Enum):
    """
    Event types for ritual workflow communication.

    Events follow the pattern: ritual.{category}.{action}
    This enables wildcard subscriptions like `ritual.**` or `ritual.phase.*`
    """
    # Lifecycle events
    RITUAL_STARTED = "ritual.started"
    RITUAL_COMPLETED = "ritual.completed"
    RITUAL_CANCELLED = "ritual.cancelled"
    RITUAL_PAUSED = "ritual.paused"
    RITUAL_RESUMED = "ritual.resumed"

    # Phase events
    PHASE_STARTED = "ritual.phase.started"
    PHASE_COMPLETED = "ritual.phase.completed"
    PHASE_SKIPPED = "ritual.phase.skipped"
    PHASE_FAILED = "ritual.phase.failed"

    # Decision events (guided decisioning at phase transitions)
    DECISION_REQUIRED = "ritual.decision.required"
    DECISION_MADE = "ritual.decision.made"
    DECISION_TIMEOUT = "ritual.decision.timeout"

    # Memory events
    MEMORY_STORED = "ritual.memory.stored"
    MEMORY_RECALLED = "ritual.memory.recalled"
    CONTEXT_RESTORED = "ritual.context.restored"

    # Learning events
    PATTERN_DETECTED = "ritual.learning.pattern_detected"
    IMPROVEMENT_PROPOSED = "ritual.learning.improvement_proposed"
    PRIOR_UPDATED = "ritual.learning.prior_updated"


class DecisionType(Enum):
    """Types of decisions at phase transitions."""
    PHASE_TRANSITION = "phase_transition"   # Proceed to next phase?
    PLAN_APPROVAL = "plan_approval"         # Approve implementation plan?
    REVIEW_OUTCOME = "review_outcome"       # Approve reviewed code?
    REFLECTION_CONFIRMATION = "reflection"  # Confirm session summary?
    CUSTOM = "custom"                       # User-defined decision


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RitualDecision:
    """
    A decision made at a phase transition.

    Decisions are logged for learning - Thompson Sampling priors are
    updated based on decision outcomes.
    """
    phase: str
    decision_type: DecisionType
    decision: str
    options_presented: List[str]
    suggested_option: str
    confidence: float = 0.5
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def was_suggested_accepted(self) -> bool:
        """Check if user accepted the suggested option."""
        return self.decision == self.suggested_option


@dataclass
class RitualContext:
    """
    Tracks state across the entire ritual workflow.

    This is the "correlation context" - all events in a ritual share
    the same ritual_id as correlation_id, enabling workflow tracking.

    Attributes:
        ritual_id: Unique identifier for this ritual instance
        feature_name: What feature/task is being worked on
        started_at: ISO timestamp of ritual start
        current_phase: Current phase in the workflow
        phase_results: Results from each completed phase
        decisions_made: All decisions at phase transitions
        memories_stored: IDs of memories stored during ritual
        skipped_phases: Phases that were skipped
        metadata: Additional context (user preferences, etc.)
    """
    ritual_id: str = field(default_factory=lambda: f"ritual-{uuid.uuid4().hex[:8]}")
    feature_name: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_phase: RitualPhase = RitualPhase.AWAKEN
    phase_results: Dict[str, Any] = field(default_factory=dict)
    decisions_made: List[RitualDecision] = field(default_factory=list)
    memories_stored: List[str] = field(default_factory=list)
    skipped_phases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_decision(
        self,
        decision: str,
        options: List[str],
        suggested: str,
        confidence: float = 0.5,
        decision_type: DecisionType = DecisionType.PHASE_TRANSITION,
        details: Optional[Dict] = None
    ) -> RitualDecision:
        """Record a decision made at current phase."""
        ritual_decision = RitualDecision(
            phase=self.current_phase.value,
            decision_type=decision_type,
            decision=decision,
            options_presented=options,
            suggested_option=suggested,
            confidence=confidence,
            details=details or {}
        )
        self.decisions_made.append(ritual_decision)
        return ritual_decision

    def record_memory(self, memory_id: str) -> None:
        """Record a memory stored during this ritual."""
        self.memories_stored.append(memory_id)

    def skip_phase(self, phase: RitualPhase) -> None:
        """Mark a phase as skipped."""
        self.skipped_phases.append(phase.value)

    def advance_phase(self, next_phase: RitualPhase) -> None:
        """Advance to the next phase."""
        self.current_phase = next_phase

    def set_phase_result(self, phase: RitualPhase, result: Any) -> None:
        """Store result from a completed phase."""
        self.phase_results[phase.value] = result

    def get_phase_result(self, phase: RitualPhase) -> Optional[Any]:
        """Get result from a completed phase."""
        return self.phase_results.get(phase.value)

    def get_duration_seconds(self) -> float:
        """Get duration of ritual so far in seconds."""
        started = datetime.fromisoformat(self.started_at)
        return (datetime.now() - started).total_seconds()

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        duration = self.get_duration_seconds()
        duration_str = f"{duration/60:.1f} minutes" if duration > 60 else f"{duration:.0f} seconds"

        return f"""
Ritual: {self.ritual_id}
Feature: {self.feature_name or '(unnamed)'}
Current Phase: {self.current_phase.value}
Duration: {duration_str}
Phases Completed: {list(self.phase_results.keys())}
Phases Skipped: {self.skipped_phases}
Decisions Made: {len(self.decisions_made)}
Memories Stored: {len(self.memories_stored)}
""".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "ritual_id": self.ritual_id,
            "feature_name": self.feature_name,
            "started_at": self.started_at,
            "current_phase": self.current_phase.value,
            "phase_results": self.phase_results,
            "decisions_made": [
                {
                    "phase": d.phase,
                    "decision_type": d.decision_type.value,
                    "decision": d.decision,
                    "options_presented": d.options_presented,
                    "suggested_option": d.suggested_option,
                    "confidence": d.confidence,
                    "details": d.details,
                    "timestamp": d.timestamp
                }
                for d in self.decisions_made
            ],
            "memories_stored": self.memories_stored,
            "skipped_phases": self.skipped_phases,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RitualContext':
        """Deserialize from dictionary."""
        context = cls(
            ritual_id=data.get("ritual_id", ""),
            feature_name=data.get("feature_name", ""),
            started_at=data.get("started_at", ""),
            current_phase=RitualPhase(data.get("current_phase", "awaken")),
            phase_results=data.get("phase_results", {}),
            memories_stored=data.get("memories_stored", []),
            skipped_phases=data.get("skipped_phases", []),
            metadata=data.get("metadata", {})
        )

        # Reconstruct decisions
        for d in data.get("decisions_made", []):
            context.decisions_made.append(RitualDecision(
                phase=d["phase"],
                decision_type=DecisionType(d["decision_type"]),
                decision=d["decision"],
                options_presented=d["options_presented"],
                suggested_option=d["suggested_option"],
                confidence=d.get("confidence", 0.5),
                details=d.get("details", {}),
                timestamp=d.get("timestamp", "")
            ))

        return context


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

    All events in a ritual workflow share the same correlation_id
    (the ritual_id), enabling workflow tracking and replay.

    Args:
        event_type: Type of event
        context: Current ritual context
        payload: Additional event data
        causation_id: ID of event that caused this one

    Returns:
        Event dict ready for EventBus emission
    """
    return {
        "event_type": event_type.value,
        "skill_name": "ritual",
        "timestamp": datetime.now().isoformat(),
        "event_id": str(uuid.uuid4()),
        "topic": event_type.value,  # Topic for routing
        "correlation_id": context.ritual_id,  # Links all events in workflow
        "causation_id": causation_id,
        "payload": {
            "ritual_id": context.ritual_id,
            "feature_name": context.feature_name,
            "phase": context.current_phase.value,
            **(payload or {})
        },
        "metadata": {
            "duration_seconds": context.get_duration_seconds(),
            "decisions_count": len(context.decisions_made),
            "memories_count": len(context.memories_stored)
        }
    }


# ============================================================================
# Event Factory Functions
# ============================================================================

def ritual_started(context: RitualContext) -> Dict[str, Any]:
    """Create RITUAL_STARTED event."""
    return create_ritual_event(
        RitualEventType.RITUAL_STARTED,
        context,
        {"started_at": context.started_at}
    )


def ritual_completed(context: RitualContext) -> Dict[str, Any]:
    """Create RITUAL_COMPLETED event with summary."""
    return create_ritual_event(
        RitualEventType.RITUAL_COMPLETED,
        context,
        {
            "summary": context.to_summary(),
            "phases_completed": list(context.phase_results.keys()),
            "phases_skipped": context.skipped_phases,
            "total_decisions": len(context.decisions_made),
            "total_memories": len(context.memories_stored),
            "duration_seconds": context.get_duration_seconds()
        }
    )


def ritual_cancelled(context: RitualContext, reason: str = "") -> Dict[str, Any]:
    """Create RITUAL_CANCELLED event."""
    return create_ritual_event(
        RitualEventType.RITUAL_CANCELLED,
        context,
        {"reason": reason, "cancelled_at_phase": context.current_phase.value}
    )


def phase_started(context: RitualContext, phase: RitualPhase) -> Dict[str, Any]:
    """Create PHASE_STARTED event and update context."""
    context.advance_phase(phase)
    return create_ritual_event(
        RitualEventType.PHASE_STARTED,
        context,
        {"phase": phase.value}
    )


def phase_completed(
    context: RitualContext,
    phase: RitualPhase,
    result: Any,
    success: bool = True
) -> Dict[str, Any]:
    """Create PHASE_COMPLETED event and store result."""
    context.set_phase_result(phase, result)
    return create_ritual_event(
        RitualEventType.PHASE_COMPLETED,
        context,
        {
            "phase": phase.value,
            "success": success,
            "result_summary": str(result)[:500] if result else None
        }
    )


def phase_skipped(context: RitualContext, phase: RitualPhase, reason: str = "") -> Dict[str, Any]:
    """Create PHASE_SKIPPED event."""
    context.skip_phase(phase)
    return create_ritual_event(
        RitualEventType.PHASE_SKIPPED,
        context,
        {"phase": phase.value, "reason": reason}
    )


def decision_required(
    context: RitualContext,
    options: List[str],
    suggested: str,
    prompt: str = "",
    timeout_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create DECISION_REQUIRED event for guided decisioning.

    This event signals that user input is needed before proceeding.
    Agents can auto-accept suggested options based on confidence.
    """
    return create_ritual_event(
        RitualEventType.DECISION_REQUIRED,
        context,
        {
            "options": options,
            "suggested": suggested,
            "prompt": prompt,
            "timeout_seconds": timeout_seconds,
            "phase": context.current_phase.value
        }
    )


def decision_made(
    context: RitualContext,
    decision: str,
    options: List[str],
    suggested: str,
    confidence: float = 0.5,
    decision_type: DecisionType = DecisionType.PHASE_TRANSITION,
    details: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create DECISION_MADE event and record in context."""
    ritual_decision = context.record_decision(
        decision=decision,
        options=options,
        suggested=suggested,
        confidence=confidence,
        decision_type=decision_type,
        details=details
    )

    return create_ritual_event(
        RitualEventType.DECISION_MADE,
        context,
        {
            "decision": decision,
            "suggested": suggested,
            "accepted_suggestion": ritual_decision.was_suggested_accepted(),
            "confidence": confidence,
            "decision_type": decision_type.value
        }
    )


def memory_stored(
    context: RitualContext,
    memory_id: str,
    content_summary: str = ""
) -> Dict[str, Any]:
    """Create MEMORY_STORED event."""
    context.record_memory(memory_id)
    return create_ritual_event(
        RitualEventType.MEMORY_STORED,
        context,
        {"memory_id": memory_id, "content_summary": content_summary}
    )


def context_restored(
    context: RitualContext,
    memories_recalled: int,
    summary: str = ""
) -> Dict[str, Any]:
    """Create CONTEXT_RESTORED event (used in AWAKEN phase)."""
    return create_ritual_event(
        RitualEventType.CONTEXT_RESTORED,
        context,
        {"memories_recalled": memories_recalled, "summary": summary}
    )


def prior_updated(
    context: RitualContext,
    agent_id: str,
    alpha: float,
    beta: float,
    expected_reward: float
) -> Dict[str, Any]:
    """Create PRIOR_UPDATED event for Thompson Sampling learning."""
    return create_ritual_event(
        RitualEventType.PRIOR_UPDATED,
        context,
        {
            "agent_id": agent_id,
            "alpha": alpha,
            "beta": beta,
            "expected_reward": expected_reward
        }
    )


# ============================================================================
# Phase Transition Helpers
# ============================================================================

PHASE_ORDER = [
    RitualPhase.AWAKEN,
    RitualPhase.PLAN,
    RitualPhase.IMPLEMENT,
    RitualPhase.REVIEW,
    RitualPhase.REFLECT
]

PHASE_DECISION_OPTIONS = {
    RitualPhase.AWAKEN: {
        "default": ["Proceed to PLAN", "Explore more context", "Skip to IMPLEMENT"],
        "suggested": "Proceed to PLAN"
    },
    RitualPhase.PLAN: {
        "default": ["Approve plan", "Refine plan", "Research more", "Skip to IMPLEMENT"],
        "suggested": "Approve plan"
    },
    RitualPhase.IMPLEMENT: {
        "default": ["Continue implementing", "Request assistance", "Move to REVIEW"],
        "suggested": "Move to REVIEW"
    },
    RitualPhase.REVIEW: {
        "default": ["Approve", "Address issues first", "Skip review"],
        "suggested": "Approve"
    },
    RitualPhase.REFLECT: {
        "default": ["Confirm summary", "Add more notes", "Quick close"],
        "suggested": "Confirm summary"
    }
}


def get_next_phase(current: RitualPhase) -> Optional[RitualPhase]:
    """Get the next phase in the ritual workflow."""
    try:
        idx = PHASE_ORDER.index(current)
        if idx < len(PHASE_ORDER) - 1:
            return PHASE_ORDER[idx + 1]
        return None  # REFLECT is the last phase
    except ValueError:
        return None


def get_phase_decision_options(phase: RitualPhase) -> Dict[str, Any]:
    """Get default decision options for a phase transition."""
    return PHASE_DECISION_OPTIONS.get(phase, {
        "default": ["Continue", "Skip"],
        "suggested": "Continue"
    })


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'RitualPhase',
    'RitualEventType',
    'DecisionType',

    # Data Structures
    'RitualContext',
    'RitualDecision',

    # Event Creation
    'create_ritual_event',

    # Event Factories
    'ritual_started',
    'ritual_completed',
    'ritual_cancelled',
    'phase_started',
    'phase_completed',
    'phase_skipped',
    'decision_required',
    'decision_made',
    'memory_stored',
    'context_restored',
    'prior_updated',

    # Helpers
    'PHASE_ORDER',
    'PHASE_DECISION_OPTIONS',
    'get_next_phase',
    'get_phase_decision_options',
]