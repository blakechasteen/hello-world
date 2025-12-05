"""Domain events for Proto code agent.

Events:
    - ProtoEventType: Enumeration of domain event types
    - ProtoEvent: Domain event representation

Implements event sourcing pattern for Proto session lifecycle.

Implemented: 2025-12-01
Status: Production ready
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class ProtoEventType(Enum):
    """Domain event types for Proto.

    Event types:
        SESSION_STARTED: Session initialization
        SESSION_ENDED: Session termination
        INTENT_RECEIVED: User intent received
        ACTION_STARTED: Action execution started
        ACTION_COMPLETED: Action execution completed
        ACTION_FAILED: Action execution failed
        RESPONSE_GENERATED: Response generated and ready
        CONTEXT_UPDATED: Code context updated
    """
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    INTENT_RECEIVED = "intent_received"
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"
    RESPONSE_GENERATED = "response_generated"
    CONTEXT_UPDATED = "context_updated"


@dataclass
class ProtoEvent:
    """Domain event for Proto.

    Represents a significant event in Proto's lifecycle, used for
    event sourcing, auditing, and debugging.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        session_id: Associated session ID
        data: Event-specific data payload
        timestamp: Timestamp when event occurred
    """
    event_id: UUID = field(default_factory=uuid4)
    event_type: ProtoEventType = ProtoEventType.SESSION_STARTED
    session_id: Optional[UUID] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_event_name(self) -> str:
        """Get human-readable event name.

        Returns:
            Event type value
        """
        return self.event_type.value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Safely get event data by key.

        Args:
            key: Data key
            default: Default value if key not found

        Returns:
            Value from data dict or default
        """
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation.

        Returns:
            Dictionary representation of event
        """
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "session_id": str(self.session_id) if self.session_id else None,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def session_started(cls, session_id: UUID) -> "ProtoEvent":
        """Create SESSION_STARTED event.

        Args:
            session_id: Session that started

        Returns:
            ProtoEvent for session start
        """
        return cls(
            event_type=ProtoEventType.SESSION_STARTED,
            session_id=session_id,
            data={"session_id": str(session_id)},
        )

    @classmethod
    def session_ended(cls, session_id: UUID, duration_seconds: float) -> "ProtoEvent":
        """Create SESSION_ENDED event.

        Args:
            session_id: Session that ended
            duration_seconds: Session duration

        Returns:
            ProtoEvent for session end
        """
        return cls(
            event_type=ProtoEventType.SESSION_ENDED,
            session_id=session_id,
            data={
                "session_id": str(session_id),
                "duration_seconds": duration_seconds,
            },
        )

    @classmethod
    def intent_received(
        cls,
        session_id: UUID,
        intent_id: UUID,
        intent_type: str,
        query: str,
    ) -> "ProtoEvent":
        """Create INTENT_RECEIVED event.

        Args:
            session_id: Session ID
            intent_id: Intent ID
            intent_type: Type of intent
            query: User query

        Returns:
            ProtoEvent for intent received
        """
        return cls(
            event_type=ProtoEventType.INTENT_RECEIVED,
            session_id=session_id,
            data={
                "intent_id": str(intent_id),
                "intent_type": intent_type,
                "query": query,
            },
        )

    @classmethod
    def action_started(
        cls,
        session_id: UUID,
        action_id: UUID,
        action_type: str,
    ) -> "ProtoEvent":
        """Create ACTION_STARTED event.

        Args:
            session_id: Session ID
            action_id: Action ID
            action_type: Type of action

        Returns:
            ProtoEvent for action started
        """
        return cls(
            event_type=ProtoEventType.ACTION_STARTED,
            session_id=session_id,
            data={
                "action_id": str(action_id),
                "action_type": action_type,
            },
        )

    @classmethod
    def action_completed(
        cls,
        session_id: UUID,
        action_id: UUID,
        duration_ms: float,
        confidence: float,
    ) -> "ProtoEvent":
        """Create ACTION_COMPLETED event.

        Args:
            session_id: Session ID
            action_id: Action ID
            duration_ms: Execution duration
            confidence: Action confidence

        Returns:
            ProtoEvent for action completed
        """
        return cls(
            event_type=ProtoEventType.ACTION_COMPLETED,
            session_id=session_id,
            data={
                "action_id": str(action_id),
                "duration_ms": duration_ms,
                "confidence": confidence,
            },
        )

    @classmethod
    def action_failed(
        cls,
        session_id: UUID,
        action_id: UUID,
        error: str,
    ) -> "ProtoEvent":
        """Create ACTION_FAILED event.

        Args:
            session_id: Session ID
            action_id: Action ID
            error: Error message

        Returns:
            ProtoEvent for action failed
        """
        return cls(
            event_type=ProtoEventType.ACTION_FAILED,
            session_id=session_id,
            data={
                "action_id": str(action_id),
                "error": error,
            },
        )

    @classmethod
    def response_generated(
        cls,
        session_id: UUID,
        response_id: UUID,
        confidence: float,
    ) -> "ProtoEvent":
        """Create RESPONSE_GENERATED event.

        Args:
            session_id: Session ID
            response_id: Response ID
            confidence: Response confidence

        Returns:
            ProtoEvent for response generated
        """
        return cls(
            event_type=ProtoEventType.RESPONSE_GENERATED,
            session_id=session_id,
            data={
                "response_id": str(response_id),
                "confidence": confidence,
            },
        )

    @classmethod
    def context_updated(
        cls,
        session_id: UUID,
        file_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> "ProtoEvent":
        """Create CONTEXT_UPDATED event.

        Args:
            session_id: Session ID
            file_path: Updated file path
            language: Programming language

        Returns:
            ProtoEvent for context updated
        """
        return cls(
            event_type=ProtoEventType.CONTEXT_UPDATED,
            session_id=session_id,
            data={
                "file_path": file_path,
                "language": language,
            },
        )
