#!/usr/bin/env python3
"""
Team Collaboration Module
==========================
User authentication, shared prompts/skills, team analytics
"""

import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import hashlib
import secrets
import json

@dataclass
class User:
    """User account"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: str = "member"  # admin, member, viewer
    team_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: Optional[str] = None

@dataclass
class Team:
    """Team/Organization"""
    team_id: str
    name: str
    description: str
    owner_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    settings: Dict = field(default_factory=dict)

@dataclass
class SharedPrompt:
    """Shared prompt with permissions"""
    prompt_id: str
    prompt_name: str
    content: str
    owner_id: str
    team_id: Optional[str]
    is_public: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

@dataclass
class TeamActivity:
    """Team activity log"""
    activity_id: str
    team_id: str
    user_id: str
    action: str  # created_prompt, shared_skill, executed_prompt, etc.
    resource_type: str  # prompt, skill, etc.
    resource_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict = field(default_factory=dict)

class TeamCollaboration:
    """Team collaboration manager"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".promptly" / "team.db")

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'member',
                team_id TEXT,
                created_at TEXT,
                last_login TEXT
            )
        """)

        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id TEXT NOT NULL,
                created_at TEXT,
                settings TEXT
            )
        """)

        # Shared prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shared_prompts (
                prompt_id TEXT PRIMARY KEY,
                prompt_name TEXT NOT NULL,
                content TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                team_id TEXT,
                is_public INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)

        # Team activities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_activities (
                activity_id TEXT PRIMARY KEY,
                team_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                timestamp TEXT,
                details TEXT
            )
        """)

        # Permissions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS permissions (
                permission_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                permission_level TEXT NOT NULL,
                granted_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def create_user(self, username: str, email: str, password: str, role: str = "member") -> User:
        """Create new user account"""
        user_id = secrets.token_urlsafe(16)
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO users (user_id, username, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user.user_id, user.username, user.email, user.password_hash, user.role, user.created_at))

        conn.commit()
        conn.close()

        return user

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user and return user object"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, username, email, password_hash, role, team_id, created_at, last_login
            FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))

        row = cursor.fetchone()

        if row:
            # Update last login
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE user_id = ?
            """, (datetime.now().isoformat(), row[0]))
            conn.commit()

            user = User(
                user_id=row[0],
                username=row[1],
                email=row[2],
                password_hash=row[3],
                role=row[4],
                team_id=row[5],
                created_at=row[6],
                last_login=datetime.now().isoformat()
            )

            conn.close()
            return user

        conn.close()
        return None

    def create_team(self, name: str, description: str, owner_id: str) -> Team:
        """Create new team"""
        team_id = secrets.token_urlsafe(16)

        team = Team(
            team_id=team_id,
            name=name,
            description=description,
            owner_id=owner_id,
            settings={}
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO teams (team_id, name, description, owner_id, created_at, settings)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (team.team_id, team.name, team.description, team.owner_id, team.created_at, json.dumps(team.settings)))

        # Add owner to team
        cursor.execute("""
            UPDATE users SET team_id = ?, role = 'admin' WHERE user_id = ?
        """, (team.team_id, owner_id))

        conn.commit()
        conn.close()

        return team

    def add_user_to_team(self, user_id: str, team_id: str, role: str = "member"):
        """Add user to team"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE users SET team_id = ?, role = ? WHERE user_id = ?
        """, (team_id, role, user_id))

        conn.commit()
        conn.close()

    def share_prompt(self, prompt_name: str, content: str, owner_id: str,
                    team_id: Optional[str] = None, is_public: bool = False,
                    metadata: Optional[Dict] = None) -> SharedPrompt:
        """Share a prompt with team or publicly"""
        prompt_id = secrets.token_urlsafe(16)

        shared_prompt = SharedPrompt(
            prompt_id=prompt_id,
            prompt_name=prompt_name,
            content=content,
            owner_id=owner_id,
            team_id=team_id,
            is_public=is_public,
            metadata=metadata or {}
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO shared_prompts (prompt_id, prompt_name, content, owner_id, team_id, is_public, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (shared_prompt.prompt_id, shared_prompt.prompt_name, shared_prompt.content,
              shared_prompt.owner_id, shared_prompt.team_id, 1 if shared_prompt.is_public else 0,
              shared_prompt.created_at, shared_prompt.updated_at, json.dumps(shared_prompt.metadata)))

        conn.commit()
        conn.close()

        # Log activity
        self.log_activity(team_id or "public", owner_id, "shared_prompt", "prompt", prompt_id)

        return shared_prompt

    def get_team_prompts(self, team_id: str) -> List[SharedPrompt]:
        """Get all prompts shared with team"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT prompt_id, prompt_name, content, owner_id, team_id, is_public, created_at, updated_at, metadata
            FROM shared_prompts
            WHERE team_id = ? OR is_public = 1
            ORDER BY updated_at DESC
        """, (team_id,))

        prompts = []
        for row in cursor.fetchall():
            prompts.append(SharedPrompt(
                prompt_id=row[0],
                prompt_name=row[1],
                content=row[2],
                owner_id=row[3],
                team_id=row[4],
                is_public=bool(row[5]),
                created_at=row[6],
                updated_at=row[7],
                metadata=json.loads(row[8]) if row[8] else {}
            ))

        conn.close()
        return prompts

    def get_team_analytics(self, team_id: str) -> Dict:
        """Get analytics for entire team"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get team member count
        cursor.execute("SELECT COUNT(*) FROM users WHERE team_id = ?", (team_id,))
        member_count = cursor.fetchone()[0]

        # Get shared prompts count
        cursor.execute("SELECT COUNT(*) FROM shared_prompts WHERE team_id = ?", (team_id,))
        prompt_count = cursor.fetchone()[0]

        # Get recent activities (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("""
            SELECT COUNT(*) FROM team_activities
            WHERE team_id = ? AND timestamp > ?
        """, (team_id, week_ago))
        recent_activities = cursor.fetchone()[0]

        # Get most active users
        cursor.execute("""
            SELECT u.username, COUNT(*) as activity_count
            FROM team_activities ta
            JOIN users u ON ta.user_id = u.user_id
            WHERE ta.team_id = ?
            GROUP BY ta.user_id
            ORDER BY activity_count DESC
            LIMIT 5
        """, (team_id,))

        most_active = [{'username': row[0], 'activities': row[1]} for row in cursor.fetchall()]

        conn.close()

        return {
            'team_id': team_id,
            'member_count': member_count,
            'prompt_count': prompt_count,
            'recent_activities': recent_activities,
            'most_active_users': most_active
        }

    def log_activity(self, team_id: str, user_id: str, action: str,
                    resource_type: str, resource_id: str, details: Optional[Dict] = None):
        """Log team activity"""
        activity_id = secrets.token_urlsafe(16)

        activity = TeamActivity(
            activity_id=activity_id,
            team_id=team_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {}
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO team_activities (activity_id, team_id, user_id, action, resource_type, resource_id, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (activity.activity_id, activity.team_id, activity.user_id, activity.action,
              activity.resource_type, activity.resource_id, activity.timestamp, json.dumps(activity.details)))

        conn.commit()
        conn.close()

    def get_team_activity_feed(self, team_id: str, limit: int = 50) -> List[Dict]:
        """Get recent team activity feed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ta.activity_id, ta.action, ta.resource_type, ta.resource_id, ta.timestamp, ta.details, u.username
            FROM team_activities ta
            JOIN users u ON ta.user_id = u.user_id
            WHERE ta.team_id = ?
            ORDER BY ta.timestamp DESC
            LIMIT ?
        """, (team_id, limit))

        activities = []
        for row in cursor.fetchall():
            activities.append({
                'activity_id': row[0],
                'action': row[1],
                'resource_type': row[2],
                'resource_id': row[3],
                'timestamp': row[4],
                'details': json.loads(row[5]) if row[5] else {},
                'username': row[6]
            })

        conn.close()
        return activities

# Demo/Testing
if __name__ == "__main__":
    print("Team Collaboration Demo")
    print("=" * 60)

    collab = TeamCollaboration()

    # Create users
    try:
        user1 = collab.create_user("alice", "alice@example.com", "password123", "admin")
        user2 = collab.create_user("bob", "bob@example.com", "password456", "member")
        print(f"[OK] Created users: {user1.username}, {user2.username}")
    except:
        print("[OK] Users already exist")
        user1 = collab.authenticate_user("alice", "password123")
        user2 = collab.authenticate_user("bob", "password456")

    # Create team
    try:
        team = collab.create_team("Data Science Team", "Our awesome DS team", user1.user_id)
        print(f"[OK] Created team: {team.name}")
    except:
        print("[OK] Team already exists")

    # Add user to team
    # collab.add_user_to_team(user2.user_id, team.team_id, "member")

    # Share prompt
    prompt = collab.share_prompt(
        "SQL Optimizer",
        "Optimize this SQL query: {query}",
        user1.user_id,
        team_id=team.team_id if 'team' in locals() else None,
        metadata={"tags": ["sql", "optimization"]}
    )
    print(f"[OK] Shared prompt: {prompt.prompt_name}")

    # Get analytics
    if 'team' in locals():
        analytics = collab.get_team_analytics(team.team_id)
        print(f"\n[OK] Team Analytics:")
        print(f"   - Members: {analytics['member_count']}")
        print(f"   - Shared Prompts: {analytics['prompt_count']}")
        print(f"   - Recent Activities: {analytics['recent_activities']}")

    print("\n[OK] Team collaboration system ready!")
