"""
HoloLoom Phase 3: User Management
November 2025

User identification, authentication, and profile management
for multi-user collaboration.
"""

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class UserRole(Enum):
    """User roles with hierarchical permissions."""
    VIEWER = "viewer"        # Can read shared knowledge
    CONTRIBUTOR = "contributor"  # Can add knowledge
    EDITOR = "editor"        # Can modify knowledge
    ADMIN = "admin"          # Full control
    OWNER = "owner"          # Team owner


# Role hierarchy (higher index = more permissions)
ROLE_HIERARCHY = [UserRole.VIEWER, UserRole.CONTRIBUTOR, UserRole.EDITOR, UserRole.ADMIN, UserRole.OWNER]


@dataclass
class User:
    """User profile."""
    user_id: str
    username: str
    email: str | None = None
    display_name: str | None = None
    role: UserRole = UserRole.CONTRIBUTOR
    teams: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    # Stats
    contributions_count: int = 0
    queries_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "display_name": self.display_name or self.username,
            "role": self.role.value,
            "teams": list(self.teams),
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "contributions_count": self.contributions_count,
            "queries_count": self.queries_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Deserialize from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data.get("email"),
            display_name=data.get("display_name"),
            role=UserRole(data.get("role", "contributor")),
            teams=set(data.get("teams", [])),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            last_active=datetime.fromisoformat(data["last_active"]) if "last_active" in data else datetime.now(),
            contributions_count=data.get("contributions_count", 0),
            queries_count=data.get("queries_count", 0),
            metadata=data.get("metadata", {})
        )

    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has required permission level."""
        user_level = ROLE_HIERARCHY.index(self.role)
        required_level = ROLE_HIERARCHY.index(required_role)
        return user_level >= required_level


@dataclass
class Team:
    """Team for collaborative work."""
    team_id: str
    name: str
    owner_id: str
    members: dict[str, UserRole] = field(default_factory=dict)  # user_id -> role
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def add_member(self, user_id: str, role: UserRole = UserRole.CONTRIBUTOR):
        """Add member to team."""
        self.members[user_id] = role

    def remove_member(self, user_id: str):
        """Remove member from team."""
        if user_id in self.members and user_id != self.owner_id:
            del self.members[user_id]

    def get_member_role(self, user_id: str) -> UserRole | None:
        """Get member's role in team."""
        return self.members.get(user_id)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "team_id": self.team_id,
            "name": self.name,
            "owner_id": self.owner_id,
            "members": {uid: role.value for uid, role in self.members.items()},
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "metadata": self.metadata
        }


class UserManager:
    """
    Manages users and teams for collaborative HoloLoom.

    Features:
    - User registration and authentication
    - Team management
    - Session handling
    - Activity tracking
    """

    def __init__(self, storage_path: str | None = None):
        """
        Initialize user manager.

        Args:
            storage_path: Path to persist user data (None for in-memory)
        """
        self.storage_path = storage_path
        self.users: dict[str, User] = {}
        self.teams: dict[str, Team] = {}
        self.sessions: dict[str, str] = {}  # session_token -> user_id
        self._username_to_id: dict[str, str] = {}

        # Load persisted data
        if storage_path:
            self._load()

    def _generate_id(self, prefix: str = "usr") -> str:
        """Generate unique ID."""
        return f"{prefix}_{secrets.token_hex(8)}"

    def _generate_session_token(self) -> str:
        """Generate session token."""
        return secrets.token_urlsafe(32)

    def _hash_password(self, password: str, salt: str = "") -> str:
        """Hash password (simple implementation)."""
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

    # === User Operations ===

    def create_user(self, username: str, email: str | None = None,
                    display_name: str | None = None,
                    role: UserRole = UserRole.CONTRIBUTOR) -> User:
        """Create a new user."""
        if username in self._username_to_id:
            raise ValueError(f"Username '{username}' already exists")

        user_id = self._generate_id("usr")
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            display_name=display_name,
            role=role
        )

        self.users[user_id] = user
        self._username_to_id[username] = user_id

        self._save()
        return user

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        user_id = self._username_to_id.get(username)
        if user_id:
            return self.users.get(user_id)
        return None

    def update_user(self, user_id: str, **updates) -> User | None:
        """Update user profile."""
        user = self.users.get(user_id)
        if not user:
            return None

        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)

        self._save()
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        user = self.users.get(user_id)
        if not user:
            return False

        # Remove from teams
        for team in self.teams.values():
            if user_id in team.members:
                team.remove_member(user_id)

        del self._username_to_id[user.username]
        del self.users[user_id]

        self._save()
        return True

    def list_users(self, team_id: str | None = None) -> list[User]:
        """List users, optionally filtered by team."""
        if team_id:
            team = self.teams.get(team_id)
            if not team:
                return []
            return [self.users[uid] for uid in team.members if uid in self.users]
        return list(self.users.values())

    # === Session Operations ===

    def login(self, username: str) -> str | None:
        """
        Login user and return session token.

        Note: Simplified authentication for demo. Production should
        use proper password hashing and verification.
        """
        user = self.get_user_by_username(username)
        if not user:
            return None

        token = self._generate_session_token()
        self.sessions[token] = user.user_id

        # Update activity
        user.last_active = datetime.now()
        self._save()

        return token

    def logout(self, session_token: str) -> bool:
        """Logout and invalidate session."""
        if session_token in self.sessions:
            del self.sessions[session_token]
            return True
        return False

    def get_current_user(self, session_token: str) -> User | None:
        """Get user from session token."""
        user_id = self.sessions.get(session_token)
        if user_id:
            user = self.users.get(user_id)
            if user:
                user.last_active = datetime.now()
            return user
        return None

    def validate_session(self, session_token: str) -> bool:
        """Check if session is valid."""
        return session_token in self.sessions

    # === Team Operations ===

    def create_team(self, name: str, owner_id: str, description: str = "") -> Team:
        """Create a new team."""
        if owner_id not in self.users:
            raise ValueError(f"Owner user '{owner_id}' not found")

        team_id = self._generate_id("team")
        team = Team(
            team_id=team_id,
            name=name,
            owner_id=owner_id,
            description=description
        )
        team.add_member(owner_id, UserRole.OWNER)

        self.teams[team_id] = team
        self.users[owner_id].teams.add(team_id)

        self._save()
        return team

    def get_team(self, team_id: str) -> Team | None:
        """Get team by ID."""
        return self.teams.get(team_id)

    def add_team_member(self, team_id: str, user_id: str,
                        role: UserRole = UserRole.CONTRIBUTOR) -> bool:
        """Add user to team."""
        team = self.teams.get(team_id)
        user = self.users.get(user_id)

        if not team or not user:
            return False

        team.add_member(user_id, role)
        user.teams.add(team_id)

        self._save()
        return True

    def remove_team_member(self, team_id: str, user_id: str) -> bool:
        """Remove user from team."""
        team = self.teams.get(team_id)
        user = self.users.get(user_id)

        if not team or not user:
            return False

        if user_id == team.owner_id:
            return False  # Cannot remove owner

        team.remove_member(user_id)
        user.teams.discard(team_id)

        self._save()
        return True

    def list_teams(self, user_id: str | None = None) -> list[Team]:
        """List teams, optionally filtered by user membership."""
        if user_id:
            user = self.users.get(user_id)
            if not user:
                return []
            return [self.teams[tid] for tid in user.teams if tid in self.teams]
        return list(self.teams.values())

    # === Activity Tracking ===

    def record_query(self, user_id: str):
        """Record a query by user."""
        user = self.users.get(user_id)
        if user:
            user.queries_count += 1
            user.last_active = datetime.now()

    def record_contribution(self, user_id: str):
        """Record a contribution by user."""
        user = self.users.get(user_id)
        if user:
            user.contributions_count += 1
            user.last_active = datetime.now()

    # === Persistence ===

    def _save(self):
        """Save to storage."""
        if not self.storage_path:
            return

        data = {
            "users": {uid: u.to_dict() for uid, u in self.users.items()},
            "teams": {tid: t.to_dict() for tid, t in self.teams.items()}
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for uid, udata in data.get("users", {}).items():
                user = User.from_dict(udata)
                self.users[uid] = user
                self._username_to_id[user.username] = uid

            for tid, tdata in data.get("teams", {}).items():
                team = Team(
                    team_id=tdata["team_id"],
                    name=tdata["name"],
                    owner_id=tdata["owner_id"],
                    description=tdata.get("description", ""),
                    created_at=datetime.fromisoformat(tdata["created_at"]) if "created_at" in tdata else datetime.now(),
                    metadata=tdata.get("metadata", {})
                )
                for mid, role in tdata.get("members", {}).items():
                    team.members[mid] = UserRole(role)
                self.teams[tid] = team

        except FileNotFoundError:
            pass


# Quick test
if __name__ == "__main__":
    print("Testing User Manager...")

    manager = UserManager()

    # Create users
    alice = manager.create_user("alice", email="alice@example.com", display_name="Alice")
    bob = manager.create_user("bob", email="bob@example.com", role=UserRole.EDITOR)
    charlie = manager.create_user("charlie", role=UserRole.VIEWER)

    print(f"Created users: {[u.username for u in manager.list_users()]}")

    # Create team
    team = manager.create_team("Research Team", alice.user_id, "Collaborative research")
    manager.add_team_member(team.team_id, bob.user_id, UserRole.CONTRIBUTOR)
    manager.add_team_member(team.team_id, charlie.user_id, UserRole.VIEWER)

    print(f"Team members: {list(team.members.keys())}")

    # Login
    token = manager.login("alice")
    print(f"Alice logged in with token: {token[:20]}...")

    current = manager.get_current_user(token)
    print(f"Current user: {current.username if current else 'None'}")

    # Check permissions
    print(f"Alice is admin? {alice.has_permission(UserRole.ADMIN)}")
    print(f"Charlie is contributor? {charlie.has_permission(UserRole.CONTRIBUTOR)}")
