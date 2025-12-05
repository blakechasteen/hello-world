"""Session management for Proto code agent.

Entities:
    - ConversationTurn: Single turn in a conversation
    - ProtoSession: Session state and conversation history

Implemented: 2025-12-01
Status: Production ready
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4

from HoloLoom.departments.proto.domain.entities import (
    CodeContext,
    ProtoIntent,
    ProtoResponse,
)


@dataclass
class ConversationTurn:
    """Single turn in a conversation.

    Represents an exchange of messages between user and Proto, including
    optional intent and response objects.

    Attributes:
        turn_id: Unique turn identifier
        role: Role of speaker ('user' or 'proto')
        content: Message content
        intent: Associated intent (if role='user')
        response: Associated response (if role='proto')
        timestamp: Timestamp of turn
    """
    turn_id: UUID = field(default_factory=uuid4)
    role: str = "user"
    content: str = ""
    intent: Optional[ProtoIntent] = None
    response: Optional[ProtoResponse] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def is_user_turn(self) -> bool:
        """Check if this is a user turn."""
        return self.role == "user"

    def is_proto_turn(self) -> bool:
        """Check if this is a Proto response turn."""
        return self.role == "proto"

    def has_intent(self) -> bool:
        """Check if turn has associated intent."""
        return self.intent is not None

    def has_response(self) -> bool:
        """Check if turn has associated response."""
        return self.response is not None


@dataclass
class ProtoSession:
    """Session state for Proto code agent.

    Maintains conversation history, code context, and user preferences
    throughout an interactive session.

    Attributes:
        session_id: Unique session identifier
        turns: List of conversation turns
        code_context: Current code context
        preferences: User preferences and settings
        created_at: Timestamp when session started
        last_activity: Timestamp of last activity
    """
    session_id: UUID = field(default_factory=uuid4)
    turns: List[ConversationTurn] = field(default_factory=list)
    code_context: Optional[CodeContext] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def add_turn(
        self,
        role: str,
        content: str,
        intent: Optional[ProtoIntent] = None,
        response: Optional[ProtoResponse] = None,
    ) -> ConversationTurn:
        """Add a turn to the conversation.

        Args:
            role: Role of speaker ('user' or 'proto')
            content: Message content
            intent: Associated intent (optional)
            response: Associated response (optional)

        Returns:
            Created ConversationTurn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            response=response,
        )
        self.turns.append(turn)
        self.last_activity = datetime.now()
        return turn

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get N most recent turns.

        Args:
            n: Number of recent turns to return

        Returns:
            List of most recent turns (up to N items)
        """
        return self.turns[-n:] if len(self.turns) > 0 else []

    def get_user_turns(self) -> List[ConversationTurn]:
        """Get all user turns in session.

        Returns:
            List of user turns
        """
        return [t for t in self.turns if t.is_user_turn()]

    def get_proto_turns(self) -> List[ConversationTurn]:
        """Get all Proto response turns in session.

        Returns:
            List of Proto response turns
        """
        return [t for t in self.turns if t.is_proto_turn()]

    def update_context(self, context: CodeContext) -> None:
        """Update current code context.

        Args:
            context: New code context
        """
        self.code_context = context
        self.last_activity = datetime.now()

    def get_turn_count(self) -> int:
        """Get total number of turns.

        Returns:
            Total turn count
        """
        return len(self.turns)

    def get_duration_seconds(self) -> float:
        """Get session duration in seconds.

        Returns:
            Duration since session creation in seconds
        """
        return (self.last_activity - self.created_at).total_seconds()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the session.

        Returns:
            Dictionary with session statistics
        """
        user_turns = self.get_user_turns()
        proto_turns = self.get_proto_turns()

        return {
            "session_id": str(self.session_id),
            "total_turns": self.get_turn_count(),
            "user_turns": len(user_turns),
            "proto_turns": len(proto_turns),
            "duration_seconds": self.get_duration_seconds(),
            "has_code_context": self.code_context is not None,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }
