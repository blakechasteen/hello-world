"""
Promptly Database Layer

Handles SQLite database operations with dual storage support:
- Global database: ~/.promptly/prompts.db
- Local database: .promptly/prompts.db (project-specific)

Local prompts override global prompts with the same name.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class PromptVersion:
    """A single version of a prompt."""
    id: int
    prompt_id: int
    version: int
    content: str
    commit_hash: str
    metadata: Dict[str, Any]
    created_at: datetime

    @property
    def short_hash(self) -> str:
        """Return first 12 characters of commit hash."""
        return self.commit_hash[:12]


@dataclass
class Prompt:
    """A prompt with its current version."""
    id: int
    name: str
    current_version: int
    branch: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None
    commit_hash: Optional[str] = None

    def render(self, **kwargs) -> str:
        """Render prompt with variable substitution."""
        if self.content is None:
            return ""
        result = self.content
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result


@dataclass
class Chain:
    """A sequence of prompts to execute."""
    id: int
    name: str
    steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class Skill:
    """A reusable skill with attached files."""
    id: int
    name: str
    description: str
    prompt_template: str
    metadata: Dict[str, Any]
    created_at: datetime
    files: List[Dict[str, Any]] = field(default_factory=list)


class PromptlyDB:
    """
    SQLite database manager for Promptly.

    Supports dual storage:
    - Global: ~/.promptly/prompts.db
    - Local: .promptly/prompts.db

    Local prompts override global prompts with the same name.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path, is_global: bool = False):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            is_global: Whether this is the global database
        """
        self.db_path = Path(db_path)
        self.is_global = is_global
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open database connection and ensure tables exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PromptlyDB":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _init_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                current_version INTEGER DEFAULT 1,
                branch TEXT DEFAULT 'main',
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Prompt versions table (git-like history)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                content TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id),
                UNIQUE(prompt_id, version)
            )
        """)

        # Branches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                parent_branch TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure 'main' branch exists
        cursor.execute("""
            INSERT OR IGNORE INTO branches (name) VALUES ('main')
        """)

        # Evaluations table (LLM Judge results)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                score REAL,
                criteria TEXT,
                feedback TEXT,
                model TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        """)

        # Chains table (prompt sequences)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                steps TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Skills table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                prompt_template TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Skill files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                file_type TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (skill_id) REFERENCES skills(id)
            )
        """)

        # Config table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Schema version
        cursor.execute("""
            INSERT OR REPLACE INTO config (key, value)
            VALUES ('schema_version', ?)
        """, (str(self.SCHEMA_VERSION),))

        self.conn.commit()

    @staticmethod
    def compute_hash(content: str, metadata: Dict = None) -> str:
        """Compute SHA-256 hash for content + metadata."""
        data = content + json.dumps(metadata or {}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    # ==================== Prompt Operations ====================

    def add_prompt(
        self,
        name: str,
        content: str,
        metadata: Dict = None,
        branch: str = "main"
    ) -> Tuple[int, str]:
        """
        Add or update a prompt.

        Args:
            name: Prompt name (unique identifier)
            content: Prompt content
            metadata: Optional metadata dict
            branch: Branch name (default: main)

        Returns:
            Tuple of (version, commit_hash)
        """
        cursor = self.conn.cursor()
        metadata = metadata or {}
        commit_hash = self.compute_hash(content, metadata)

        # Check if prompt exists
        cursor.execute("SELECT id, current_version FROM prompts WHERE name = ?", (name,))
        row = cursor.fetchone()

        if row:
            # Update existing prompt
            prompt_id = row["id"]
            new_version = row["current_version"] + 1

            cursor.execute("""
                UPDATE prompts
                SET current_version = ?, updated_at = CURRENT_TIMESTAMP, branch = ?
                WHERE id = ?
            """, (new_version, branch, prompt_id))
        else:
            # Create new prompt
            cursor.execute("""
                INSERT INTO prompts (name, current_version, branch, metadata)
                VALUES (?, 1, ?, ?)
            """, (name, branch, json.dumps(metadata)))
            prompt_id = cursor.lastrowid
            new_version = 1

        # Add version record
        cursor.execute("""
            INSERT INTO prompt_versions (prompt_id, version, content, commit_hash, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (prompt_id, new_version, content, commit_hash, json.dumps(metadata)))

        self.conn.commit()
        return new_version, commit_hash

    def get_prompt(
        self,
        name: str,
        version: int = None,
        commit_hash: str = None
    ) -> Optional[Prompt]:
        """
        Get a prompt by name, optionally at specific version.

        Args:
            name: Prompt name
            version: Specific version number (optional)
            commit_hash: Specific commit hash (optional)

        Returns:
            Prompt object or None if not found
        """
        cursor = self.conn.cursor()

        # Get prompt metadata
        cursor.execute("SELECT * FROM prompts WHERE name = ?", (name,))
        prompt_row = cursor.fetchone()

        if not prompt_row:
            return None

        prompt_id = prompt_row["id"]

        # Get specific version
        if commit_hash:
            cursor.execute("""
                SELECT * FROM prompt_versions
                WHERE prompt_id = ? AND commit_hash LIKE ?
                ORDER BY version DESC LIMIT 1
            """, (prompt_id, f"{commit_hash}%"))
        elif version:
            cursor.execute("""
                SELECT * FROM prompt_versions
                WHERE prompt_id = ? AND version = ?
            """, (prompt_id, version))
        else:
            # Get current version
            cursor.execute("""
                SELECT * FROM prompt_versions
                WHERE prompt_id = ? AND version = ?
            """, (prompt_id, prompt_row["current_version"]))

        version_row = cursor.fetchone()

        if not version_row:
            return None

        return Prompt(
            id=prompt_row["id"],
            name=prompt_row["name"],
            current_version=prompt_row["current_version"],
            branch=prompt_row["branch"],
            created_at=datetime.fromisoformat(prompt_row["created_at"]),
            updated_at=datetime.fromisoformat(prompt_row["updated_at"]),
            metadata=json.loads(prompt_row["metadata"]),
            content=version_row["content"],
            commit_hash=version_row["commit_hash"]
        )

    def list_prompts(self, branch: str = None) -> List[Prompt]:
        """
        List all prompts, optionally filtered by branch.

        Args:
            branch: Filter by branch name (optional)

        Returns:
            List of Prompt objects
        """
        cursor = self.conn.cursor()

        if branch:
            cursor.execute("SELECT * FROM prompts WHERE branch = ?", (branch,))
        else:
            cursor.execute("SELECT * FROM prompts")

        prompts = []
        for row in cursor.fetchall():
            prompts.append(Prompt(
                id=row["id"],
                name=row["name"],
                current_version=row["current_version"],
                branch=row["branch"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"])
            ))
        return prompts

    def get_prompt_history(self, name: str) -> List[PromptVersion]:
        """
        Get version history for a prompt.

        Args:
            name: Prompt name

        Returns:
            List of PromptVersion objects (newest first)
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM prompts WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return []

        prompt_id = row["id"]

        cursor.execute("""
            SELECT * FROM prompt_versions
            WHERE prompt_id = ?
            ORDER BY version DESC
        """, (prompt_id,))

        versions = []
        for row in cursor.fetchall():
            versions.append(PromptVersion(
                id=row["id"],
                prompt_id=row["prompt_id"],
                version=row["version"],
                content=row["content"],
                commit_hash=row["commit_hash"],
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"])
            ))
        return versions

    def delete_prompt(self, name: str) -> bool:
        """
        Delete a prompt and all its versions.

        Args:
            name: Prompt name

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM prompts WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return False

        prompt_id = row["id"]

        cursor.execute("DELETE FROM prompt_versions WHERE prompt_id = ?", (prompt_id,))
        cursor.execute("DELETE FROM evaluations WHERE prompt_id = ?", (prompt_id,))
        cursor.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))

        self.conn.commit()
        return True

    # ==================== Branch Operations ====================

    def create_branch(self, name: str, from_branch: str = "main") -> bool:
        """
        Create a new branch.

        Args:
            name: Branch name
            from_branch: Parent branch name

        Returns:
            True if created, False if already exists
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO branches (name, parent_branch)
                VALUES (?, ?)
            """, (name, from_branch))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def list_branches(self) -> List[str]:
        """Get list of all branch names."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM branches ORDER BY name")
        return [row["name"] for row in cursor.fetchall()]

    def get_current_branch(self) -> str:
        """Get currently checked out branch."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = 'current_branch'")
        row = cursor.fetchone()
        return row["value"] if row else "main"

    def checkout_branch(self, name: str) -> bool:
        """
        Switch to a branch.

        Args:
            name: Branch name

        Returns:
            True if switched, False if branch doesn't exist
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM branches WHERE name = ?", (name,))
        if not cursor.fetchone():
            return False

        cursor.execute("""
            INSERT OR REPLACE INTO config (key, value)
            VALUES ('current_branch', ?)
        """, (name,))
        self.conn.commit()
        return True

    # ==================== Evaluation Operations ====================

    def add_evaluation(
        self,
        prompt_name: str,
        version: int,
        score: float,
        criteria: str,
        feedback: str,
        model: str,
        metadata: Dict = None
    ) -> int:
        """
        Add an LLM Judge evaluation result.

        Returns:
            Evaluation ID
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM prompts WHERE name = ?", (prompt_name,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        prompt_id = row["id"]

        cursor.execute("""
            INSERT INTO evaluations (prompt_id, version, score, criteria, feedback, model, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (prompt_id, version, score, criteria, feedback, model, json.dumps(metadata or {})))

        self.conn.commit()
        return cursor.lastrowid

    def get_evaluations(self, prompt_name: str) -> List[Dict]:
        """Get all evaluations for a prompt."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM prompts WHERE name = ?", (prompt_name,))
        row = cursor.fetchone()
        if not row:
            return []

        prompt_id = row["id"]

        cursor.execute("""
            SELECT * FROM evaluations
            WHERE prompt_id = ?
            ORDER BY created_at DESC
        """, (prompt_id,))

        return [dict(row) for row in cursor.fetchall()]

    # ==================== Chain Operations ====================

    def create_chain(self, name: str, steps: List[Dict], metadata: Dict = None) -> int:
        """
        Create a prompt chain.

        Args:
            name: Chain name
            steps: List of step definitions
            metadata: Optional metadata

        Returns:
            Chain ID
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO chains (name, steps, metadata)
            VALUES (?, ?, ?)
        """, (name, json.dumps(steps), json.dumps(metadata or {})))

        self.conn.commit()
        return cursor.lastrowid

    def get_chain(self, name: str) -> Optional[Chain]:
        """Get a chain by name."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM chains WHERE name = ?", (name,))
        row = cursor.fetchone()

        if not row:
            return None

        return Chain(
            id=row["id"],
            name=row["name"],
            steps=json.loads(row["steps"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"])
        )

    def list_chains(self) -> List[Chain]:
        """List all chains."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM chains")

        return [
            Chain(
                id=row["id"],
                name=row["name"],
                steps=json.loads(row["steps"]),
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"])
            )
            for row in cursor.fetchall()
        ]

    # ==================== Skill Operations ====================

    def create_skill(
        self,
        name: str,
        description: str,
        prompt_template: str,
        files: List[Dict] = None,
        metadata: Dict = None
    ) -> int:
        """
        Create a skill with optional attached files.

        Args:
            name: Skill name
            description: Skill description
            prompt_template: The prompt template
            files: List of {filename, content, file_type} dicts
            metadata: Optional metadata

        Returns:
            Skill ID
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO skills (name, description, prompt_template, metadata)
            VALUES (?, ?, ?, ?)
        """, (name, description, prompt_template, json.dumps(metadata or {})))

        skill_id = cursor.lastrowid

        # Add files
        if files:
            for file in files:
                cursor.execute("""
                    INSERT INTO skill_files (skill_id, filename, content, file_type, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    skill_id,
                    file.get("filename", "unnamed"),
                    file.get("content", ""),
                    file.get("file_type", "text"),
                    json.dumps(file.get("metadata", {}))
                ))

        self.conn.commit()
        return skill_id

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill with its files."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM skills WHERE name = ?", (name,))
        row = cursor.fetchone()

        if not row:
            return None

        skill_id = row["id"]

        cursor.execute("SELECT * FROM skill_files WHERE skill_id = ?", (skill_id,))
        files = [
            {
                "filename": f["filename"],
                "content": f["content"],
                "file_type": f["file_type"],
                "metadata": json.loads(f["metadata"])
            }
            for f in cursor.fetchall()
        ]

        return Skill(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            prompt_template=row["prompt_template"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            files=files
        )

    def list_skills(self) -> List[Skill]:
        """List all skills (without file contents for efficiency)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM skills")

        return [
            Skill(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                prompt_template=row["prompt_template"],
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"])
            )
            for row in cursor.fetchall()
        ]

    # ==================== Config Operations ====================

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]
        return default

    def set_config(self, key: str, value: Any) -> None:
        """Set a config value."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO config (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, json.dumps(value) if not isinstance(value, str) else value))
        self.conn.commit()
