"""
Collaboration Session Manager for Multi-User HoloLoom Sessions.

Phase 3: Multi-User Collaboration - Core session infrastructure.

Created: November 2025
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle states."""
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    CLOSING = auto()
    CLOSED = auto()


class ParticipantRole(Enum):
    """Roles within a collaboration session."""
    OWNER = "owner"           # Full control, can delete session
    ADMIN = "admin"           # Manage participants, edit all
    EDITOR = "editor"         # Can edit shared content
    COMMENTER = "commenter"   # Can comment, not edit
    VIEWER = "viewer"         # Read-only access


class SessionType(Enum):
    """Types of collaboration sessions."""
    KNOWLEDGE_BASE = "knowledge_base"   # Shared knowledge exploration
    WHITEBOARD = "whiteboard"           # Collaborative drawing
    RESEARCH = "research"               # Multi-query research session
    REVIEW = "review"                   # Content review workflow
    PRESENTATION = "presentation"       # Follow-the-leader mode


@dataclass
class Participant:
    """A participant in a collaboration session."""
    user_id: str
    display_name: str
    role: ParticipantRole
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    avatar_id: str | None = None
    connection_id: str | None = None
    is_connected: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "role": self.role.value,
            "joined_at": self.joined_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "avatar_id": self.avatar_id,
            "is_connected": self.is_connected,
            "metadata": self.metadata
        }

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_active = datetime.now()


@dataclass
class SessionSettings:
    """Configuration for a collaboration session."""
    max_participants: int = 50
    allow_anonymous: bool = False
    require_approval: bool = False
    auto_pause_timeout: int = 300  # seconds
    enable_voice: bool = True
    enable_video: bool = False
    enable_screen_share: bool = True
    allow_recording: bool = False
    sync_mode: str = "real_time"  # real_time, periodic, manual
    persistence: str = "session"  # session, persistent, temporary

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_participants": self.max_participants,
            "allow_anonymous": self.allow_anonymous,
            "require_approval": self.require_approval,
            "auto_pause_timeout": self.auto_pause_timeout,
            "enable_voice": self.enable_voice,
            "enable_video": self.enable_video,
            "enable_screen_share": self.enable_screen_share,
            "allow_recording": self.allow_recording,
            "sync_mode": self.sync_mode,
            "persistence": self.persistence
        }


@dataclass
class JoinRequest:
    """Pending request to join a session."""
    user_id: str
    display_name: str
    requested_role: ParticipantRole
    requested_at: datetime = field(default_factory=datetime.now)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "requested_role": self.requested_role.value,
            "requested_at": self.requested_at.isoformat(),
            "message": self.message
        }


@dataclass
class Session:
    """A collaboration session for multi-user interaction."""
    session_id: str
    name: str
    session_type: SessionType
    owner_id: str
    state: SessionState = SessionState.INITIALIZING
    settings: SessionSettings = field(default_factory=SessionSettings)
    created_at: datetime = field(default_factory=datetime.now)

    participants: dict[str, Participant] = field(default_factory=dict)
    join_requests: dict[str, JoinRequest] = field(default_factory=dict)

    # Shared state references
    knowledge_base_id: str | None = None
    whiteboard_ids: list[str] = field(default_factory=list)

    # Session metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Event callbacks
    _event_handlers: dict[str, list[Callable]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._event_handlers = {}

    def add_participant(self, participant: Participant) -> bool:
        """Add a participant to the session."""
        if len(self.participants) >= self.settings.max_participants:
            logger.warning(f"Session {self.session_id} is full")
            return False

        self.participants[participant.user_id] = participant
        self._emit_event("participant_joined", {"participant": participant.to_dict()})
        logger.info(f"Participant {participant.user_id} joined session {self.session_id}")
        return True

    def remove_participant(self, user_id: str) -> bool:
        """Remove a participant from the session."""
        if user_id not in self.participants:
            return False

        participant = self.participants.pop(user_id)
        self._emit_event("participant_left", {"user_id": user_id})
        logger.info(f"Participant {user_id} left session {self.session_id}")

        # If owner leaves, transfer ownership
        if user_id == self.owner_id and self.participants:
            new_owner = next(iter(self.participants.values()))
            self.owner_id = new_owner.user_id
            new_owner.role = ParticipantRole.OWNER
            self._emit_event("ownership_transferred", {"new_owner": new_owner.user_id})

        return True

    def update_participant_role(self, user_id: str, new_role: ParticipantRole) -> bool:
        """Update a participant's role."""
        if user_id not in self.participants:
            return False

        old_role = self.participants[user_id].role
        self.participants[user_id].role = new_role
        self._emit_event("role_changed", {
            "user_id": user_id,
            "old_role": old_role.value,
            "new_role": new_role.value
        })
        return True

    def add_join_request(self, request: JoinRequest) -> bool:
        """Add a pending join request."""
        if not self.settings.require_approval:
            # Auto-approve if not requiring approval
            participant = Participant(
                user_id=request.user_id,
                display_name=request.display_name,
                role=request.requested_role
            )
            return self.add_participant(participant)

        self.join_requests[request.user_id] = request
        self._emit_event("join_requested", {"request": request.to_dict()})
        return True

    def approve_join_request(self, user_id: str, assigned_role: ParticipantRole | None = None) -> bool:
        """Approve a pending join request."""
        if user_id not in self.join_requests:
            return False

        request = self.join_requests.pop(user_id)
        role = assigned_role or request.requested_role

        participant = Participant(
            user_id=request.user_id,
            display_name=request.display_name,
            role=role
        )
        return self.add_participant(participant)

    def deny_join_request(self, user_id: str, reason: str | None = None) -> bool:
        """Deny a pending join request."""
        if user_id not in self.join_requests:
            return False

        request = self.join_requests.pop(user_id)
        self._emit_event("join_denied", {
            "user_id": user_id,
            "reason": reason
        })
        return True

    def get_active_participants(self) -> list[Participant]:
        """Get list of currently connected participants."""
        return [p for p in self.participants.values() if p.is_connected]

    def has_permission(self, user_id: str, required_role: ParticipantRole) -> bool:
        """Check if user has at least the required role level."""
        if user_id not in self.participants:
            return False

        user_role = self.participants[user_id].role
        role_hierarchy = [
            ParticipantRole.VIEWER,
            ParticipantRole.COMMENTER,
            ParticipantRole.EDITOR,
            ParticipantRole.ADMIN,
            ParticipantRole.OWNER
        ]

        user_level = role_hierarchy.index(user_role)
        required_level = role_hierarchy.index(required_role)
        return user_level >= required_level

    def on(self, event: str, handler: Callable):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: dict[str, Any]):
        """Emit an event to all handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(event, data, self)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "session_type": self.session_type.value,
            "owner_id": self.owner_id,
            "state": self.state.name,
            "settings": self.settings.to_dict(),
            "created_at": self.created_at.isoformat(),
            "participants": {k: v.to_dict() for k, v in self.participants.items()},
            "pending_requests": len(self.join_requests),
            "knowledge_base_id": self.knowledge_base_id,
            "whiteboard_ids": self.whiteboard_ids,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata
        }


class SessionManager:
    """
    Manages multiple collaboration sessions.

    Provides session lifecycle management, discovery, and coordination.
    """

    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.user_sessions: dict[str, set[str]] = {}  # user_id -> session_ids
        self._lock = asyncio.Lock()
        self._event_handlers: dict[str, list[Callable]] = {}

    async def create_session(
        self,
        name: str,
        owner_id: str,
        owner_name: str,
        session_type: SessionType = SessionType.KNOWLEDGE_BASE,
        settings: SessionSettings | None = None,
        knowledge_base_id: str | None = None,
        description: str = "",
        tags: list[str] | None = None
    ) -> Session:
        """Create a new collaboration session."""
        async with self._lock:
            session_id = f"session_{uuid4().hex[:12]}"

            session = Session(
                session_id=session_id,
                name=name,
                session_type=session_type,
                owner_id=owner_id,
                settings=settings or SessionSettings(),
                knowledge_base_id=knowledge_base_id,
                description=description,
                tags=tags or []
            )

            # Add owner as first participant
            owner = Participant(
                user_id=owner_id,
                display_name=owner_name,
                role=ParticipantRole.OWNER
            )
            session.add_participant(owner)

            session.state = SessionState.ACTIVE
            self.sessions[session_id] = session

            # Track user's sessions
            if owner_id not in self.user_sessions:
                self.user_sessions[owner_id] = set()
            self.user_sessions[owner_id].add(session_id)

            self._emit_event("session_created", {"session": session.to_dict()})
            logger.info(f"Created session {session_id} by {owner_id}")

            return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    async def join_session(
        self,
        session_id: str,
        user_id: str,
        display_name: str,
        requested_role: ParticipantRole = ParticipantRole.EDITOR,
        message: str | None = None
    ) -> Session | None:
        """Request to join a session."""
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None

        if session.state != SessionState.ACTIVE:
            logger.warning(f"Session {session_id} is not active")
            return None

        if user_id in session.participants:
            # Already a participant
            return session

        async with self._lock:
            if session.settings.require_approval:
                request = JoinRequest(
                    user_id=user_id,
                    display_name=display_name,
                    requested_role=requested_role,
                    message=message
                )
                session.add_join_request(request)
            else:
                participant = Participant(
                    user_id=user_id,
                    display_name=display_name,
                    role=requested_role
                )
                if not session.add_participant(participant):
                    return None

            # Track user's sessions
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)

            return session

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a collaboration session."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        async with self._lock:
            result = session.remove_participant(user_id)

            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)

            # Close session if no participants
            if not session.participants:
                await self._close_session(session_id)

            return result

    async def close_session(self, session_id: str, closer_id: str) -> bool:
        """Close a session (must be owner or admin)."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        if not session.has_permission(closer_id, ParticipantRole.ADMIN):
            logger.warning(f"User {closer_id} cannot close session {session_id}")
            return False

        return await self._close_session(session_id)

    async def _close_session(self, session_id: str) -> bool:
        """Internal session close."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        async with self._lock:
            session.state = SessionState.CLOSING
            self._emit_event("session_closing", {"session_id": session_id})

            # Notify all participants
            for user_id in list(session.participants.keys()):
                if user_id in self.user_sessions:
                    self.user_sessions[user_id].discard(session_id)

            session.state = SessionState.CLOSED
            del self.sessions[session_id]

            self._emit_event("session_closed", {"session_id": session_id})
            logger.info(f"Closed session {session_id}")

            return True

    async def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all sessions a user is participating in."""
        session_ids = self.user_sessions.get(user_id, set())
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]

    async def list_public_sessions(
        self,
        session_type: SessionType | None = None,
        tags: list[str] | None = None,
        limit: int = 50
    ) -> list[Session]:
        """List available public sessions."""
        sessions = []

        for session in self.sessions.values():
            if session.state != SessionState.ACTIVE:
                continue

            if session_type and session.session_type != session_type:
                continue

            if tags:
                if not any(tag in session.tags for tag in tags):
                    continue

            sessions.append(session)

            if len(sessions) >= limit:
                break

        return sessions

    async def broadcast_to_session(
        self,
        session_id: str,
        event: str,
        data: dict[str, Any],
        exclude_user: str | None = None
    ):
        """Broadcast an event to all session participants."""
        session = self.sessions.get(session_id)
        if not session:
            return

        for user_id, participant in session.participants.items():
            if exclude_user and user_id == exclude_user:
                continue

            if participant.is_connected:
                # This would integrate with WebSocket delivery
                self._emit_event(f"broadcast:{user_id}", {
                    "session_id": session_id,
                    "event": event,
                    "data": data
                })

    def on(self, event: str, handler: Callable):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: dict[str, Any]):
        """Emit an event to handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def to_state(self) -> dict[str, Any]:
        """Export manager state for persistence."""
        return {
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "user_sessions": {k: list(v) for k, v in self.user_sessions.items()},
            "total_sessions": len(self.sessions),
            "total_users": len(self.user_sessions)
        }


# Convenience factory function
def create_session_manager() -> SessionManager:
    """Create a new session manager instance."""
    return SessionManager()
