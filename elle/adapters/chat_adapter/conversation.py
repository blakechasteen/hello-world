"""Conversation state management for multi-turn chat.

Maintains history of user/Elle exchanges and provides
context for the prompt builder.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum
import uuid


class TurnRole(Enum):
    """Who said what."""
    USER = "user"
    ELLE = "elle"
    SYSTEM = "system"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: TurnRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """
    Multi-turn conversation state.

    Tracks history and provides context for prompts.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 20  # Keep last N turns for context
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def add_user_message(self, content: str) -> ConversationTurn:
        """Add a user message to the conversation."""
        turn = ConversationTurn(
            role=TurnRole.USER,
            content=content,
        )
        self._add_turn(turn)
        return turn

    def add_elle_response(
        self,
        content: str,
        reasoning: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> ConversationTurn:
        """Add Elle's response to the conversation."""
        metadata = {}
        if reasoning:
            metadata["reasoning"] = reasoning
        if mode:
            metadata["mode"] = mode

        turn = ConversationTurn(
            role=TurnRole.ELLE,
            content=content,
            metadata=metadata,
        )
        self._add_turn(turn)
        return turn

    def add_system_note(self, content: str) -> ConversationTurn:
        """Add a system note (e.g., scene change, error)."""
        turn = ConversationTurn(
            role=TurnRole.SYSTEM,
            content=content,
        )
        self._add_turn(turn)
        return turn

    def _add_turn(self, turn: ConversationTurn) -> None:
        """Add turn and enforce max_turns limit."""
        self.turns.append(turn)
        # Trim old turns, but keep at least some context
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context_for_prompt(self, last_n: int = 10) -> str:
        """
        Format recent conversation for inclusion in prompt.

        Returns formatted string showing recent exchanges.
        """
        recent = self.turns[-last_n:] if len(self.turns) > last_n else self.turns

        if not recent:
            return "(No conversation history yet)"

        lines = []
        for turn in recent:
            if turn.role == TurnRole.USER:
                lines.append(f"User: {turn.content}")
            elif turn.role == TurnRole.ELLE:
                lines.append(f"Elle: {turn.content}")
            elif turn.role == TurnRole.SYSTEM:
                lines.append(f"[System: {turn.content}]")

        return "\n".join(lines)

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for turn in reversed(self.turns):
            if turn.role == TurnRole.USER:
                return turn.content
        return None

    def get_last_elle_response(self) -> Optional[str]:
        """Get Elle's most recent response."""
        for turn in reversed(self.turns):
            if turn.role == TurnRole.ELLE:
                return turn.content
        return None

    @property
    def turn_count(self) -> int:
        """Total number of turns."""
        return len(self.turns)

    @property
    def user_message_count(self) -> int:
        """Number of user messages."""
        return sum(1 for t in self.turns if t.role == TurnRole.USER)

    def to_thread_context(self) -> List[str]:
        """
        Convert to thread context format for voice_interface.py compatibility.

        Returns list of messages for thread-aware processing.
        """
        return [
            f"{t.role.value}: {t.content}"
            for t in self.turns
        ]

    def clear(self) -> None:
        """Clear conversation history."""
        self.turns = []

    def summary(self) -> str:
        """Get a brief summary of the conversation."""
        if not self.turns:
            return "Empty conversation"

        duration = (datetime.now() - self.created_at).seconds
        return (
            f"Session {self.session_id}: "
            f"{self.turn_count} turns, "
            f"{self.user_message_count} user messages, "
            f"{duration}s duration"
        )


class ConversationManager:
    """
    Manages multiple conversations across sessions.

    For Phase 2: Persist to HoloLoom memory.
    """

    def __init__(self):
        self.active_conversation: Optional[Conversation] = None
        self.past_conversations: List[Conversation] = []

    def start_conversation(self) -> Conversation:
        """Start a new conversation."""
        if self.active_conversation:
            self.past_conversations.append(self.active_conversation)

        self.active_conversation = Conversation()
        return self.active_conversation

    def get_or_create(self) -> Conversation:
        """Get active conversation or create new one."""
        if not self.active_conversation:
            return self.start_conversation()
        return self.active_conversation

    def end_conversation(self) -> Optional[Conversation]:
        """End and archive the current conversation."""
        if self.active_conversation:
            self.past_conversations.append(self.active_conversation)
            ended = self.active_conversation
            self.active_conversation = None
            return ended
        return None

    @property
    def total_conversations(self) -> int:
        """Total number of conversations (past + active)."""
        count = len(self.past_conversations)
        if self.active_conversation:
            count += 1
        return count
