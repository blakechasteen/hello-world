"""
Session management for Elle Game Engine.

Provides persistent state tracking across multiple requests:
- Conversation history (last N exchanges)
- World flags (persistent world state)
- NPC relationships (reputation, mood changes)

Design: Protocol-based with multiple storage backends for flexibility.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol
from abc import ABC, abstractmethod


# ============================================================================
# Session Models
# ============================================================================

@dataclass
class ConversationExchange:
    """A single exchange in the conversation history."""

    player_query: str
    elle_response: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    npc_id: Optional[str] = None  # Which NPC was involved (if any)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "player_query": self.player_query,
            "elle_response": self.elle_response,
            "timestamp": self.timestamp,
            "npc_id": self.npc_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationExchange':
        """Create from dictionary."""
        return cls(
            player_query=data["player_query"],
            elle_response=data["elle_response"],
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            npc_id=data.get("npc_id"),
        )


@dataclass
class NPCRelationship:
    """Relationship state with a specific NPC."""

    npc_id: str
    reputation: int = 0  # -100 to 100
    interactions: int = 0
    last_mood: Optional[str] = None
    custom_flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "npc_id": self.npc_id,
            "reputation": self.reputation,
            "interactions": self.interactions,
            "last_mood": self.last_mood,
            "custom_flags": self.custom_flags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NPCRelationship':
        """Create from dictionary."""
        return cls(
            npc_id=data["npc_id"],
            reputation=data.get("reputation", 0),
            interactions=data.get("interactions", 0),
            last_mood=data.get("last_mood"),
            custom_flags=data.get("custom_flags", {}),
        )


@dataclass
class GameSession:
    """
    Game session state that persists across requests.

    Tracks conversation history, world flags, and NPC relationships
    for continuous narrative experiences.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    player_id: Optional[str] = None
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    last_accessed: float = field(default_factory=lambda: datetime.now().timestamp())

    # Conversation history (circular buffer, keeps last N)
    conversation_history: List[ConversationExchange] = field(default_factory=list)
    max_history_size: int = 10

    # Persistent world state
    world_flags: Dict[str, bool] = field(default_factory=dict)

    # NPC relationship tracking
    npc_relationships: Dict[str, NPCRelationship] = field(default_factory=dict)

    def add_exchange(
        self,
        player_query: str,
        elle_response: str,
        npc_id: Optional[str] = None,
    ) -> None:
        """
        Add a conversation exchange to history.

        Maintains circular buffer of last N exchanges.
        """
        exchange = ConversationExchange(
            player_query=player_query,
            elle_response=elle_response,
            npc_id=npc_id,
        )

        self.conversation_history.append(exchange)

        # Trim to max size (circular buffer)
        if len(self.conversation_history) > self.max_history_size:
            self.conversation_history = self.conversation_history[-self.max_history_size:]

        # Update last accessed
        self.last_accessed = datetime.now().timestamp()

    def update_world_flags(self, flag_changes: Dict[str, bool]) -> None:
        """Update persistent world flags."""
        self.world_flags.update(flag_changes)
        self.last_accessed = datetime.now().timestamp()

    def update_npc_relationship(
        self,
        npc_id: str,
        reputation_delta: int = 0,
        mood: Optional[str] = None,
        custom_flags: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Update relationship with an NPC."""
        if npc_id not in self.npc_relationships:
            self.npc_relationships[npc_id] = NPCRelationship(npc_id=npc_id)

        rel = self.npc_relationships[npc_id]
        rel.reputation += reputation_delta
        rel.reputation = max(-100, min(100, rel.reputation))  # Clamp to [-100, 100]
        rel.interactions += 1

        if mood is not None:
            rel.last_mood = mood

        if custom_flags is not None:
            rel.custom_flags.update(custom_flags)

        self.last_accessed = datetime.now().timestamp()

    def get_conversation_context(self, max_exchanges: Optional[int] = None) -> str:
        """
        Get conversation history as formatted string for prompt injection.

        Args:
            max_exchanges: Maximum number of recent exchanges to include (default: all)

        Returns:
            Formatted conversation history
        """
        if not self.conversation_history:
            return "(No previous conversation)"

        exchanges = self.conversation_history
        if max_exchanges is not None:
            exchanges = exchanges[-max_exchanges:]

        lines = []
        for exchange in exchanges:
            npc_note = f" (with NPC: {exchange.npc_id})" if exchange.npc_id else ""
            lines.append(f"Player{npc_note}: {exchange.player_query}")
            lines.append(f"Elle: {exchange.elle_response}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "player_id": self.player_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "conversation_history": [ex.to_dict() for ex in self.conversation_history],
            "max_history_size": self.max_history_size,
            "world_flags": self.world_flags,
            "npc_relationships": {
                npc_id: rel.to_dict()
                for npc_id, rel in self.npc_relationships.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GameSession':
        """Create from dictionary."""
        session = cls(
            session_id=data["session_id"],
            player_id=data.get("player_id"),
            created_at=data.get("created_at", datetime.now().timestamp()),
            last_accessed=data.get("last_accessed", datetime.now().timestamp()),
            max_history_size=data.get("max_history_size", 10),
            world_flags=data.get("world_flags", {}),
        )

        # Restore conversation history
        for ex_data in data.get("conversation_history", []):
            session.conversation_history.append(ConversationExchange.from_dict(ex_data))

        # Restore NPC relationships
        for npc_id, rel_data in data.get("npc_relationships", {}).items():
            session.npc_relationships[npc_id] = NPCRelationship.from_dict(rel_data)

        return session


# ============================================================================
# Session Store Protocol
# ============================================================================

class SessionStore(Protocol):
    """
    Protocol for session storage backends.

    Implementations can use in-memory dicts, files, databases, etc.
    """

    def create_session(
        self,
        player_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GameSession:
        """
        Create a new game session.

        Args:
            player_id: Optional player identifier
            session_id: Optional session ID (generated if not provided)

        Returns:
            New GameSession
        """
        ...

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            GameSession if found, None otherwise
        """
        ...

    def update_session(self, session: GameSession) -> None:
        """
        Update an existing session.

        Args:
            session: Session to update
        """
        ...

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    def save(self) -> None:
        """
        Persist all sessions to storage.

        For in-memory stores, this is a no-op.
        For file/database stores, this writes to disk.
        """
        ...


# ============================================================================
# In-Memory Session Store
# ============================================================================

class InMemorySessionStore:
    """
    In-memory session storage for development/testing.

    Fast but non-persistent - sessions lost when service restarts.
    """

    def __init__(self):
        self._sessions: Dict[str, GameSession] = {}

    def create_session(
        self,
        player_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GameSession:
        """Create new session."""
        session = GameSession(
            session_id=session_id or str(uuid.uuid4()),
            player_id=player_id,
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def update_session(self, session: GameSession) -> None:
        """Update session."""
        self._sessions[session.session_id] = session

    def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def save(self) -> None:
        """No-op for in-memory store."""
        pass

    def __len__(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)


# ============================================================================
# JSON File Session Store
# ============================================================================

class JSONSessionStore:
    """
    File-based session storage using JSON.

    Persistent across service restarts. Good for small-medium deployments.
    """

    def __init__(self, storage_path: str = "sessions"):
        """
        Initialize JSON session store.

        Args:
            storage_path: Directory path for session files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache for faster access
        self._cache: Dict[str, GameSession] = {}

        # Load existing sessions on startup
        self._load_all()

    def _get_session_file(self, session_id: str) -> Path:
        """Get file path for session."""
        return self.storage_path / f"{session_id}.json"

    def _load_all(self) -> None:
        """Load all existing sessions from disk."""
        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    session = GameSession.from_dict(data)
                    self._cache[session.session_id] = session
            except Exception as e:
                print(f"Warning: Failed to load session {session_file}: {e}")

    def create_session(
        self,
        player_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> GameSession:
        """Create new session and save to disk."""
        session = GameSession(
            session_id=session_id or str(uuid.uuid4()),
            player_id=player_id,
        )
        self._cache[session.session_id] = session
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """Get session from cache or disk."""
        # Try cache first
        if session_id in self._cache:
            return self._cache[session_id]

        # Try loading from disk
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    session = GameSession.from_dict(data)
                    self._cache[session_id] = session
                    return session
            except Exception as e:
                print(f"Error loading session {session_id}: {e}")

        return None

    def update_session(self, session: GameSession) -> None:
        """Update session in cache and save to disk."""
        self._cache[session.session_id] = session
        self._save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete session from cache and disk."""
        # Remove from cache
        if session_id in self._cache:
            del self._cache[session_id]

        # Remove from disk
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            session_file.unlink()
            return True

        return False

    def save(self) -> None:
        """Save all cached sessions to disk."""
        for session in self._cache.values():
            self._save_session(session)

    def _save_session(self, session: GameSession) -> None:
        """Save individual session to disk."""
        session_file = self._get_session_file(session.session_id)
        with open(session_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

    def __len__(self) -> int:
        """Number of active sessions."""
        return len(self._cache)
