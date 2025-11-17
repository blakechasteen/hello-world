"""
Session Management
==================
Manages multi-user collaborative sessions.

A session is like a shared loom where multiple users weave together.
Each session has:
- Unique ID
- List of participants (users)
- Shared state (conversation history, memory)
- Ownership and permissions
- Persistence (save/load)
"""

import logging
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SessionUser:
    """User in a collaborative session."""
    user_id: str
    username: str
    role: str = "editor"  # viewer, editor, admin, owner
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "joined_at": self.joined_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SessionUser':
        """Create from dictionary."""
        return SessionUser(
            user_id=data["user_id"],
            username=data["username"],
            role=data.get("role", "editor"),
            joined_at=datetime.fromisoformat(data["joined_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"])
        )


@dataclass
class CollaborativeSession:
    """
    Collaborative session with multiple users.

    Attributes:
        session_id: Unique session identifier
        name: Human-readable session name
        owner_id: User ID of session owner
        users: Dictionary of active users
        state: Shared session state (conversation, memory, etc.)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional session metadata
    """
    session_id: str
    name: str
    owner_id: str
    users: Dict[str, SessionUser] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_user(self, user_id: str, username: str, role: str = "editor") -> SessionUser:
        """Add user to session."""
        if user_id in self.users:
            logger.warning(f"User {user_id} already in session {self.session_id}")
            return self.users[user_id]

        user = SessionUser(user_id=user_id, username=username, role=role)
        self.users[user_id] = user
        self.updated_at = datetime.now(timezone.utc)

        logger.info(f"User {username} ({user_id}) joined session {self.session_id} as {role}")
        return user

    def remove_user(self, user_id: str) -> bool:
        """Remove user from session."""
        if user_id not in self.users:
            logger.warning(f"User {user_id} not in session {self.session_id}")
            return False

        username = self.users[user_id].username
        del self.users[user_id]
        self.updated_at = datetime.now(timezone.utc)

        logger.info(f"User {username} ({user_id}) left session {self.session_id}")
        return True

    def get_user(self, user_id: str) -> Optional[SessionUser]:
        """Get user by ID."""
        return self.users.get(user_id)

    def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp."""
        if user_id in self.users:
            self.users[user_id].last_activity = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)

    def transfer_ownership(self, new_owner_id: str) -> bool:
        """Transfer session ownership to another user."""
        if new_owner_id not in self.users:
            logger.error(f"Cannot transfer ownership: user {new_owner_id} not in session")
            return False

        # Update old owner to admin
        if self.owner_id in self.users:
            self.users[self.owner_id].role = "admin"

        # Update new owner
        self.owner_id = new_owner_id
        self.users[new_owner_id].role = "owner"
        self.updated_at = datetime.now(timezone.utc)

        logger.info(f"Session {self.session_id} ownership transferred to {new_owner_id}")
        return True

    def get_active_users(self, inactive_threshold_seconds: int = 300) -> List[SessionUser]:
        """Get users active within threshold."""
        now = datetime.now(timezone.utc)
        active = []

        for user in self.users.values():
            inactive_time = (now - user.last_activity).total_seconds()
            if inactive_time < inactive_threshold_seconds:
                active.append(user)

        return active

    def update_state(self, key: str, value: Any):
        """Update session state."""
        self.state[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "owner_id": self.owner_id,
            "users": {uid: user.to_dict() for uid, user in self.users.items()},
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CollaborativeSession':
        """Create from dictionary."""
        users = {
            uid: SessionUser.from_dict(user_data)
            for uid, user_data in data.get("users", {}).items()
        }

        return CollaborativeSession(
            session_id=data["session_id"],
            name=data["name"],
            owner_id=data["owner_id"],
            users=users,
            state=data.get("state", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {})
        )

    def save(self, path: Path):
        """Save session to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved session {self.session_id} to {path}")

    @staticmethod
    def load(path: Path) -> 'CollaborativeSession':
        """Load session from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        session = CollaborativeSession.from_dict(data)
        logger.info(f"Loaded session {session.session_id} from {path}")
        return session


class SessionManager:
    """
    Manages multiple collaborative sessions.

    Central registry for all active sessions.
    Provides session lifecycle management:
    - Create/delete sessions
    - User join/leave
    - Session persistence
    - Session discovery
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            storage_dir: Directory for session persistence
        """
        self.sessions: Dict[str, CollaborativeSession] = {}
        self.storage_dir = storage_dir or Path("./data/sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SessionManager initialized with storage: {self.storage_dir}")

    def create_session(
        self,
        name: str,
        owner_id: str,
        owner_username: str,
        session_id: Optional[str] = None
    ) -> CollaborativeSession:
        """
        Create new collaborative session.

        Args:
            name: Session name
            owner_id: Owner user ID
            owner_username: Owner username
            session_id: Optional custom session ID

        Returns:
            Created session
        """
        session_id = session_id or str(uuid.uuid4())

        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists")
            return self.sessions[session_id]

        # Create session
        session = CollaborativeSession(
            session_id=session_id,
            name=name,
            owner_id=owner_id
        )

        # Add owner as first user
        session.add_user(owner_id, owner_username, role="owner")

        # Store
        self.sessions[session_id] = session

        logger.info(f"Created session {session_id}: {name} (owner: {owner_username})")
        return session

    def get_session(self, session_id: str) -> Optional[CollaborativeSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return False

        session = self.sessions[session_id]
        del self.sessions[session_id]

        # Delete persisted file if exists
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

        logger.info(f"Deleted session {session_id}: {session.name}")
        return True

    def join_session(
        self,
        session_id: str,
        user_id: str,
        username: str,
        role: str = "editor"
    ) -> Optional[SessionUser]:
        """
        Add user to session.

        Args:
            session_id: Session to join
            user_id: User ID
            username: Username
            role: User role

        Returns:
            SessionUser if successful
        """
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Cannot join: session {session_id} not found")
            return None

        return session.add_user(user_id, username, role)

    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Remove user from session."""
        session = self.get_session(session_id)
        if not session:
            return False

        return session.remove_user(user_id)

    def list_sessions(self, user_id: Optional[str] = None) -> List[CollaborativeSession]:
        """
        List all sessions (optionally filtered by user).

        Args:
            user_id: Filter by user membership

        Returns:
            List of sessions
        """
        if user_id:
            return [
                session for session in self.sessions.values()
                if user_id in session.users
            ]
        else:
            return list(self.sessions.values())

    def save_session(self, session_id: str):
        """Persist session to disk."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot save: session {session_id} not found")
            return

        session_file = self.storage_dir / f"{session_id}.json"
        session.save(session_file)

    def load_session(self, session_id: str) -> Optional[CollaborativeSession]:
        """Load session from disk."""
        session_file = self.storage_dir / f"{session_id}.json"

        if not session_file.exists():
            logger.warning(f"Session file not found: {session_file}")
            return None

        session = CollaborativeSession.load(session_file)
        self.sessions[session_id] = session
        return session

    def load_all_sessions(self):
        """Load all persisted sessions."""
        loaded = 0

        for session_file in self.storage_dir.glob("*.json"):
            try:
                session = CollaborativeSession.load(session_file)
                self.sessions[session.session_id] = session
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load session {session_file}: {e}")

        logger.info(f"Loaded {loaded} sessions from {self.storage_dir}")

    def save_all_sessions(self):
        """Persist all sessions to disk."""
        saved = 0

        for session in self.sessions.values():
            try:
                session_file = self.storage_dir / f"{session.session_id}.json"
                session.save(session_file)
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save session {session.session_id}: {e}")

        logger.info(f"Saved {saved} sessions to {self.storage_dir}")

    def cleanup_inactive_sessions(self, inactive_days: int = 7):
        """Remove sessions with no recent activity."""
        now = datetime.now(timezone.utc)
        to_delete = []

        for session_id, session in self.sessions.items():
            age = (now - session.updated_at).days
            if age > inactive_days and len(session.users) == 0:
                to_delete.append(session_id)

        for session_id in to_delete:
            self.delete_session(session_id)

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} inactive sessions")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("="*80)
    print("Collaborative Session Management")
    print("="*80 + "\n")

    # Create manager
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_dir=Path(tmpdir))

        # Create session
        print("1. Creating session...")
        session = manager.create_session(
            name="AI Research Discussion",
            owner_id="user1",
            owner_username="Alice"
        )
        print(f"   ✓ Session: {session.name} ({session.session_id})")

        # Add users
        print("\n2. Adding users...")
        manager.join_session(session.session_id, "user2", "Bob", role="editor")
        manager.join_session(session.session_id, "user3", "Charlie", role="viewer")
        print(f"   ✓ Users: {len(session.users)}")

        # Update state
        print("\n3. Updating state...")
        session.update_state("conversation", [
            {"user": "Alice", "message": "Welcome to the discussion!"},
            {"user": "Bob", "message": "Thanks for having me!"}
        ])
        print(f"   ✓ State updated")

        # Save session
        print("\n4. Persisting session...")
        manager.save_session(session.session_id)
        print(f"   ✓ Saved to disk")

        # Transfer ownership
        print("\n5. Transferring ownership...")
        session.transfer_ownership("user2")
        print(f"   ✓ New owner: Bob")

        # List sessions
        print("\n6. User sessions:")
        for user_id in ["user1", "user2", "user3"]:
            user_sessions = manager.list_sessions(user_id=user_id)
            print(f"   • User {user_id}: {len(user_sessions)} sessions")

        # Activity
        print("\n7. Active users:")
        active = session.get_active_users()
        print(f"   ✓ {len(active)} users active")

        print("\n✓ Session management ready!")
