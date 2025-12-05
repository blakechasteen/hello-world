"""Proto domain layer.

Core domain entities, sessions, and events for the Proto code agent department.

Exports:
    - IntentType: User intent categories
    - CodeContext: Code execution context
    - ProtoIntent: User intent with context
    - ProtoAction: Action to be executed
    - ProtoResponse: Response from Proto
    - ConversationTurn: Single conversation turn
    - ProtoSession: Session state
    - ProtoEventType: Domain event types
    - ProtoEvent: Domain event
"""

from HoloLoom.departments.proto.domain.entities import (
    IntentType,
    CodeContext,
    ProtoIntent,
    ProtoAction,
    ProtoResponse,
)

from HoloLoom.departments.proto.domain.session import (
    ConversationTurn,
    ProtoSession,
)

from HoloLoom.departments.proto.domain.events import (
    ProtoEventType,
    ProtoEvent,
)

__all__ = [
    "IntentType",
    "CodeContext",
    "ProtoIntent",
    "ProtoAction",
    "ProtoResponse",
    "ConversationTurn",
    "ProtoSession",
    "ProtoEventType",
    "ProtoEvent",
]
