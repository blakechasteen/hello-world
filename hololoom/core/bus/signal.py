"""
Unified Signal Schema
=====================

Single signal type that subsumes all three existing bus message types:
- RitualMessageBus.PhaseMessage (priority queue, TTL, correlation chains)
- Multi-Agent MessageBus.Message (agent-addressed, content-based)
- LiteMemoryBus (store/query — becomes a delegate, not a signal source)

Every action in HoloLoom — API calls, heartbeats, chain steps, MCTS decisions,
ritual triggers, memory operations — is a Signal flowing through the UnifiedBus.

Design: Compose from existing patterns, don't reinvent.
- Priority ordering: from RitualMessageBus (heapq __lt__)
- Correlation tracking: from RitualMessageBus (correlation_id + causation_id)
- TTL expiration: from RitualMessageBus (is_expired)
- Serialization: to_dict/from_dict for SQLite journal + Neo4j persistence
"""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# ============================================================================
# Enums
# ============================================================================

class SignalKind(str, Enum):
    """
    Unified signal types across all HoloLoom subsystems.

    Merges:
    - RitualMessageBus.MessageType (REQUEST, RESPONSE, HANDOFF, BROADCAST, CONTROL)
    - Multi-Agent MessageType (QUESTION, ANSWER, INSIGHT, REQUEST_HELP, etc.)
    - New execution lifecycle signals
    """
    # --- Inter-agent (from multi_agent_communication.py) ---
    QUESTION = "question"
    ANSWER = "answer"
    INSIGHT = "insight"
    REQUEST_HELP = "request_help"
    OFFER_HELP = "offer_help"
    ACKNOWLEDGE = "acknowledge"

    # --- Ritual/phase (from ritual message_bus.py) ---
    REQUEST = "request"
    RESPONSE = "response"
    HANDOFF = "handoff"
    BROADCAST = "broadcast"
    CONTROL = "control"

    # --- Execution lifecycle (new) ---
    QUERY = "query"
    EXECUTION_START = "execution_start"
    STEP_COMPLETE = "step_complete"
    STEP_FAILED = "step_failed"
    EXECUTION_COMPLETE = "execution_complete"
    EXECUTION_FAILED = "execution_failed"

    # --- MCTS (new) ---
    MCTS_DECISION = "mcts_decision"
    MCTS_BACKPROP = "mcts_backprop"
    BREAKTHROUGH = "breakthrough"

    # --- Convergence (new) ---
    CONVERGENCE = "convergence"
    CONVERGENCE_CHECK = "convergence_check"

    # --- Memory (new — emitted by bus wrapper around LiteMemoryBus) ---
    MEMORY_STORED = "memory_stored"
    MEMORY_QUERIED = "memory_queried"
    MEMORY_DECAYED = "memory_decayed"

    # --- Triggers (new) ---
    HEARTBEAT = "heartbeat"
    RITUAL = "ritual"
    CRON = "cron"

    # --- Refinement (multi-pass LLM improvement) ---
    REFINEMENT_START = "refinement_start"
    REFINEMENT_COMPLETE = "refinement_complete"

    # --- System ---
    ERROR = "error"
    WARNING = "warning"


class SignalPriority(int, Enum):
    """
    Priority levels (lower value = higher priority).
    Matches RitualMessageBus.MessagePriority exactly.
    """
    CRITICAL = 0    # Safety, errors — processed first
    HIGH = 1        # User decisions, handoffs
    NORMAL = 2      # Standard execution signals
    LOW = 3         # Background, learning, heartbeats


class TriggerMode(str, Enum):
    """How a signal was created."""
    ACTIVE = "active"           # Direct API call / explicit invocation
    HEARTBEAT = "heartbeat"     # ChronoTrigger pulse (60s)
    CRON = "cron"               # RitualScheduler / cron expression
    REACTIVE = "reactive"       # Triggered by another signal (causation chain)


# ============================================================================
# Signal Data Structure
# ============================================================================

@dataclass
class Signal:
    """
    Universal signal for the unified bus.

    Inherits design from RitualMessageBus.PhaseMessage:
    - heapq-compatible ordering via __lt__
    - TTL-based expiration
    - Correlation/causation chains for workflow tracking
    - Serialization for journal persistence

    Extended with:
    - SignalKind enum covering all subsystem message types
    - TriggerMode for distinguishing how signals originate
    - Source/target for routing (generalizes from_phase/to_phase)
    """
    # Identity
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: SignalKind = SignalKind.REQUEST
    priority: SignalPriority = SignalPriority.NORMAL

    # Routing
    source: str = ""       # Emitter identity (endpoint, agent, scheduler, etc.)
    target: str = ""       # "" = broadcast, else specific handler/topic

    # Content
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Workflow tracking (from RitualMessageBus)
    correlation_id: str | None = None
    causation_id: str | None = None

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: int | None = None

    # Origin
    trigger_mode: TriggerMode = TriggerMode.ACTIVE

    def is_expired(self) -> bool:
        """Check if signal has expired based on TTL (from PhaseMessage pattern)."""
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(self.created_at)
        return datetime.now() > created + timedelta(seconds=self.ttl_seconds)

    def __lt__(self, other: 'Signal') -> bool:
        """Priority ordering for heapq (from PhaseMessage.__lt__)."""
        if not isinstance(other, Signal):
            return NotImplemented
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def derive(
        self,
        kind: SignalKind,
        payload: dict[str, Any] | None = None,
        priority: SignalPriority | None = None,
        source: str = "",
        target: str = "",
    ) -> 'Signal':
        """
        Create a child signal inheriting correlation chain.

        The new signal's causation_id points to this signal.
        The correlation_id propagates (or starts from this signal's id).
        """
        return Signal(
            kind=kind,
            priority=priority or self.priority,
            source=source or self.source,
            target=target,
            payload=payload or {},
            correlation_id=self.correlation_id or self.id,
            causation_id=self.id,
            trigger_mode=TriggerMode.REACTIVE,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for SQLite journal / Neo4j / wire format."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "priority": self.priority.value,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "trigger_mode": self.trigger_mode.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Signal':
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            kind=SignalKind(data.get("kind", "request")),
            priority=SignalPriority(data.get("priority", 2)),
            source=data.get("source", ""),
            target=data.get("target", ""),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            ttl_seconds=data.get("ttl_seconds"),
            trigger_mode=TriggerMode(data.get("trigger_mode", "active")),
        )


# ============================================================================
# Handler Type
# ============================================================================

SignalHandler = Callable[[Signal], Awaitable[Signal | None]]
"""
Async handler that receives a Signal and optionally returns a response Signal.

If a response Signal is returned, the bus automatically:
- Sets correlation_id from the original signal
- Sets causation_id to the original signal's id
- Re-emits it through the bus

This matches the RitualMessageBus handler pattern exactly.
"""


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SignalKind",
    "SignalPriority",
    "TriggerMode",
    "Signal",
    "SignalHandler",
]
