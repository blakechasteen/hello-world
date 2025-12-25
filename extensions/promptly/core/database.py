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


@dataclass
class PromptExecutionRecord:
    """Record of a prompt execution for analytics."""
    id: int
    prompt_id: int
    prompt_name: str
    version: int
    task_type: Optional[str]
    input_data: Dict[str, Any]
    output: str
    quality_score: float
    latency_ms: float
    llm_provider: Optional[str]
    llm_model: Optional[str]
    token_count: Optional[int]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class ThompsonPrior:
    """Thompson Sampling prior for a prompt-task combination."""
    id: int
    task_type: str
    prompt_id: int
    prompt_name: str
    alpha: float
    beta: float
    updated_at: datetime

    @property
    def expected_quality(self) -> float:
        """E[X] = alpha / (alpha + beta)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def total_samples(self) -> float:
        """Total number of samples (alpha + beta - 2 for prior)."""
        return self.alpha + self.beta - 2.0


@dataclass
class MRFRefinementDBRecord:
    """Record of an MRF refinement execution."""
    id: int
    prompt_name: str
    strategy: str
    quality_before: float
    quality_after: float
    improvement: float
    latency_ms: float
    model_provider: str
    components_applied: List[str]
    metadata: Dict[str, Any]
    created_at: datetime

    @property
    def improvement_percent(self) -> float:
        """Calculate improvement as percentage."""
        if self.quality_before > 0:
            return ((self.quality_after - self.quality_before) / self.quality_before) * 100
        return 0.0

    @property
    def is_success(self) -> bool:
        """Whether refinement improved quality (threshold: 0.7)."""
        return self.quality_after >= 0.7


@dataclass
class MRFStrategyPrior:
    """Thompson Sampling prior for an MRF strategy."""
    id: int
    strategy: str
    alpha: float
    beta: float
    updated_at: datetime

    @property
    def expected_quality(self) -> float:
        """E[X] = alpha / (alpha + beta)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def total_samples(self) -> float:
        """Total number of samples (alpha + beta - 2 for prior)."""
        return self.alpha + self.beta - 2.0


class PromptlyDB:
    """
    SQLite database manager for Promptly.

    Supports dual storage:
    - Global: ~/.promptly/prompts.db
    - Local: .promptly/prompts.db

    Local prompts override global prompts with the same name.
    """

    SCHEMA_VERSION = 2

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

        # Prompt executions table (detailed history for analytics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                task_type TEXT,
                input_data TEXT,
                output TEXT,
                quality_score REAL,
                latency_ms REAL,
                llm_provider TEXT,
                llm_model TEXT,
                token_count INTEGER,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        """)

        # Create index for execution queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_executions_prompt_id
            ON prompt_executions(prompt_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_executions_task_type
            ON prompt_executions(task_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_executions_created_at
            ON prompt_executions(created_at)
        """)

        # Thompson sampling priors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thompson_priors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                prompt_id INTEGER NOT NULL,
                alpha REAL DEFAULT 1.0,
                beta REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id),
                UNIQUE(task_type, prompt_id)
            )
        """)

        # Create index for Thompson prior queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_thompson_task_type
            ON thompson_priors(task_type)
        """)

        # MRF refinements table (Metaprompt Refinement Framework analytics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mrf_refinements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_name TEXT NOT NULL,
                strategy TEXT NOT NULL,
                quality_before REAL NOT NULL,
                quality_after REAL NOT NULL,
                improvement REAL NOT NULL,
                latency_ms REAL,
                model_provider TEXT,
                components_applied TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for MRF refinement queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mrf_refinements_prompt_name
            ON mrf_refinements(prompt_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mrf_refinements_strategy
            ON mrf_refinements(strategy)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mrf_refinements_created_at
            ON mrf_refinements(created_at)
        """)

        # MRF Thompson Sampling priors table (per-strategy learning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mrf_thompson_priors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT UNIQUE NOT NULL,
                alpha REAL DEFAULT 1.0,
                beta REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for MRF Thompson prior queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mrf_thompson_strategy
            ON mrf_thompson_priors(strategy)
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

    # ==================== Execution Analytics Operations ====================

    def add_execution(
        self,
        prompt_name: str,
        version: int,
        output: str,
        quality_score: float,
        latency_ms: float,
        task_type: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        token_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record a prompt execution for analytics.

        Args:
            prompt_name: Name of the executed prompt
            version: Prompt version used
            output: Generated output
            quality_score: Quality score (0.0-1.0)
            latency_ms: Execution time in milliseconds
            task_type: Type of task (e.g., 'summarization', 'code_review')
            input_data: Input variables used
            llm_provider: LLM provider (e.g., 'anthropic', 'openai')
            llm_model: Model name
            token_count: Token count for the execution
            metadata: Additional metadata

        Returns:
            Execution ID
        """
        cursor = self.conn.cursor()

        # Get prompt_id
        cursor.execute("SELECT id FROM prompts WHERE name = ?", (prompt_name,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        prompt_id = row["id"]

        cursor.execute("""
            INSERT INTO prompt_executions
            (prompt_id, version, task_type, input_data, output, quality_score,
             latency_ms, llm_provider, llm_model, token_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_id,
            version,
            task_type,
            json.dumps(input_data or {}),
            output,
            quality_score,
            latency_ms,
            llm_provider,
            llm_model,
            token_count,
            json.dumps(metadata or {})
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_executions(
        self,
        prompt_name: Optional[str] = None,
        task_type: Optional[str] = None,
        days: int = 30,
        limit: int = 100
    ) -> List[PromptExecutionRecord]:
        """
        Query execution history.

        Args:
            prompt_name: Filter by prompt name (optional)
            task_type: Filter by task type (optional)
            days: Number of days to look back (default: 30)
            limit: Maximum records to return (default: 100)

        Returns:
            List of PromptExecutionRecord objects
        """
        cursor = self.conn.cursor()

        query = """
            SELECT pe.*, p.name as prompt_name
            FROM prompt_executions pe
            JOIN prompts p ON pe.prompt_id = p.id
            WHERE pe.created_at >= datetime('now', ?)
        """
        params = [f'-{days} days']

        if prompt_name:
            query += " AND p.name = ?"
            params.append(prompt_name)

        if task_type:
            query += " AND pe.task_type = ?"
            params.append(task_type)

        query += " ORDER BY pe.created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        records = []
        for row in cursor.fetchall():
            records.append(PromptExecutionRecord(
                id=row["id"],
                prompt_id=row["prompt_id"],
                prompt_name=row["prompt_name"],
                version=row["version"],
                task_type=row["task_type"],
                input_data=json.loads(row["input_data"]),
                output=row["output"],
                quality_score=row["quality_score"],
                latency_ms=row["latency_ms"],
                llm_provider=row["llm_provider"],
                llm_model=row["llm_model"],
                token_count=row["token_count"],
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"])
            ))
        return records

    def get_prompt_analytics(
        self,
        prompt_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get aggregate analytics for a prompt.

        Args:
            prompt_name: Prompt name
            days: Analysis window in days

        Returns:
            Analytics dict with stats, trends, and recommendations
        """
        cursor = self.conn.cursor()

        # Get prompt_id
        cursor.execute("SELECT id FROM prompts WHERE name = ?", (prompt_name,))
        row = cursor.fetchone()
        if not row:
            return {"error": f"Prompt '{prompt_name}' not found"}

        prompt_id = row["id"]

        # Get aggregate statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_executions,
                AVG(quality_score) as avg_quality,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN quality_score >= 0.7 THEN 1 ELSE 0 END) as successes,
                MIN(quality_score) as min_quality,
                MAX(quality_score) as max_quality,
                AVG(token_count) as avg_tokens
            FROM prompt_executions
            WHERE prompt_id = ? AND created_at >= datetime('now', ?)
        """, (prompt_id, f'-{days} days'))

        stats_row = cursor.fetchone()

        total = stats_row["total_executions"] or 0
        if total == 0:
            return {
                "prompt_name": prompt_name,
                "total_executions": 0,
                "message": "No executions found in the specified window"
            }

        # Get daily breakdown for trend analysis
        cursor.execute("""
            SELECT
                DATE(created_at) as date,
                COUNT(*) as executions,
                AVG(quality_score) as avg_quality
            FROM prompt_executions
            WHERE prompt_id = ? AND created_at >= datetime('now', ?)
            GROUP BY DATE(created_at)
            ORDER BY date
        """, (prompt_id, f'-{days} days'))

        daily_data = [dict(row) for row in cursor.fetchall()]

        # Calculate trend (simple linear regression on quality)
        quality_trend = "stable"
        if len(daily_data) >= 3:
            qualities = [d["avg_quality"] for d in daily_data if d["avg_quality"]]
            if len(qualities) >= 3:
                first_half = sum(qualities[:len(qualities)//2]) / (len(qualities)//2)
                second_half = sum(qualities[len(qualities)//2:]) / (len(qualities) - len(qualities)//2)
                if second_half > first_half + 0.05:
                    quality_trend = "improving"
                elif second_half < first_half - 0.05:
                    quality_trend = "declining"

        # Get task type distribution
        cursor.execute("""
            SELECT task_type, COUNT(*) as count
            FROM prompt_executions
            WHERE prompt_id = ? AND created_at >= datetime('now', ?) AND task_type IS NOT NULL
            GROUP BY task_type
        """, (prompt_id, f'-{days} days'))

        task_distribution = {row["task_type"]: row["count"] for row in cursor.fetchall()}

        # Get Thompson Sampling expected quality if available
        thompson_quality = None
        cursor.execute("""
            SELECT AVG(alpha / (alpha + beta)) as expected_quality
            FROM thompson_priors
            WHERE prompt_id = ?
        """, (prompt_id,))
        thompson_row = cursor.fetchone()
        if thompson_row and thompson_row["expected_quality"]:
            thompson_quality = thompson_row["expected_quality"]

        return {
            "prompt_name": prompt_name,
            "total_executions": total,
            "avg_quality": round(stats_row["avg_quality"], 3) if stats_row["avg_quality"] else None,
            "avg_latency_ms": round(stats_row["avg_latency_ms"], 1) if stats_row["avg_latency_ms"] else None,
            "success_rate": round(stats_row["successes"] / total, 3) if total else 0,
            "quality_range": {
                "min": round(stats_row["min_quality"], 3) if stats_row["min_quality"] else None,
                "max": round(stats_row["max_quality"], 3) if stats_row["max_quality"] else None
            },
            "avg_tokens": round(stats_row["avg_tokens"]) if stats_row["avg_tokens"] else None,
            "quality_trend": quality_trend,
            "task_type_distribution": task_distribution,
            "thompson_expected_quality": round(thompson_quality, 3) if thompson_quality else None,
            "daily_breakdown": daily_data,
            "days_analyzed": days
        }

    # ==================== Thompson Sampling Operations ====================

    def update_thompson_prior(
        self,
        prompt_name: str,
        task_type: str,
        success: bool,
        quality: float = 0.5
    ) -> Tuple[float, float]:
        """
        Update Thompson Sampling prior based on execution outcome.

        Args:
            prompt_name: Prompt name
            task_type: Task type for this prior
            success: Whether the execution was successful (quality >= threshold)
            quality: Quality score to weight the update

        Returns:
            Tuple of (alpha, beta) after update
        """
        cursor = self.conn.cursor()

        # Get prompt_id
        cursor.execute("SELECT id FROM prompts WHERE name = ?", (prompt_name,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        prompt_id = row["id"]

        # Get or create prior
        cursor.execute("""
            SELECT id, alpha, beta FROM thompson_priors
            WHERE task_type = ? AND prompt_id = ?
        """, (task_type, prompt_id))

        row = cursor.fetchone()

        if row:
            # Update existing prior
            alpha = row["alpha"]
            beta = row["beta"]

            if success:
                alpha += quality
            else:
                beta += (1.0 - quality)

            cursor.execute("""
                UPDATE thompson_priors
                SET alpha = ?, beta = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (alpha, beta, row["id"]))
        else:
            # Create new prior with initial update
            alpha = 1.0 + (quality if success else 0.0)
            beta = 1.0 + (0.0 if success else (1.0 - quality))

            cursor.execute("""
                INSERT INTO thompson_priors (task_type, prompt_id, alpha, beta)
                VALUES (?, ?, ?, ?)
            """, (task_type, prompt_id, alpha, beta))

        self.conn.commit()
        return (alpha, beta)

    def get_thompson_recommendation(
        self,
        task_type: str,
        candidates: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get Thompson Sampling recommendation for best prompt.

        Args:
            task_type: Task type to get recommendation for
            candidates: Optional list of prompt names to consider

        Returns:
            Dict with recommended prompt and expected quality, or None
        """
        import random

        cursor = self.conn.cursor()

        if candidates:
            # Get priors for specific candidates
            placeholders = ",".join(["?" for _ in candidates])
            cursor.execute(f"""
                SELECT tp.*, p.name as prompt_name
                FROM thompson_priors tp
                JOIN prompts p ON tp.prompt_id = p.id
                WHERE tp.task_type = ? AND p.name IN ({placeholders})
            """, [task_type] + candidates)
        else:
            # Get all priors for this task type
            cursor.execute("""
                SELECT tp.*, p.name as prompt_name
                FROM thompson_priors tp
                JOIN prompts p ON tp.prompt_id = p.id
                WHERE tp.task_type = ?
            """, (task_type,))

        rows = cursor.fetchall()

        if not rows:
            return None

        # Thompson Sampling: sample from Beta distribution for each prompt
        best_prompt = None
        best_sample = -1.0
        all_priors = []

        for row in rows:
            alpha = row["alpha"]
            beta = row["beta"]
            expected = alpha / (alpha + beta)

            # Sample from Beta distribution
            sample = random.betavariate(alpha, beta)

            all_priors.append({
                "prompt_name": row["prompt_name"],
                "expected_quality": round(expected, 3),
                "alpha": alpha,
                "beta": beta,
                "sample": round(sample, 3)
            })

            if sample > best_sample:
                best_sample = sample
                best_prompt = row["prompt_name"]

        # Sort by expected quality for alternatives
        all_priors.sort(key=lambda x: x["expected_quality"], reverse=True)

        return {
            "recommended_prompt": best_prompt,
            "expected_quality": round(all_priors[0]["expected_quality"], 3) if all_priors else None,
            "confidence": round(best_sample, 3),
            "task_type": task_type,
            "alternatives": all_priors[:5]  # Top 5 alternatives
        }

    def get_thompson_priors(
        self,
        task_type: Optional[str] = None,
        prompt_name: Optional[str] = None
    ) -> List[ThompsonPrior]:
        """
        Get Thompson Sampling priors.

        Args:
            task_type: Filter by task type (optional)
            prompt_name: Filter by prompt name (optional)

        Returns:
            List of ThompsonPrior objects
        """
        cursor = self.conn.cursor()

        query = """
            SELECT tp.*, p.name as prompt_name
            FROM thompson_priors tp
            JOIN prompts p ON tp.prompt_id = p.id
            WHERE 1=1
        """
        params = []

        if task_type:
            query += " AND tp.task_type = ?"
            params.append(task_type)

        if prompt_name:
            query += " AND p.name = ?"
            params.append(prompt_name)

        query += " ORDER BY tp.updated_at DESC"

        cursor.execute(query, params)

        priors = []
        for row in cursor.fetchall():
            priors.append(ThompsonPrior(
                id=row["id"],
                task_type=row["task_type"],
                prompt_id=row["prompt_id"],
                prompt_name=row["prompt_name"],
                alpha=row["alpha"],
                beta=row["beta"],
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        return priors

    # ==================== MRF (Metaprompt Refinement Framework) Operations ====================

    def add_mrf_refinement(
        self,
        prompt_name: str,
        strategy: str,
        quality_before: float,
        quality_after: float,
        latency_ms: float = 0.0,
        model_provider: str = "unknown",
        components_applied: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record an MRF refinement execution.

        Args:
            prompt_name: Name of the prompt that was refined
            strategy: MRF strategy used (REFINE, CRITIQUE, VERIFY, ELEGANCE, HOFSTADTER, AUTO)
            quality_before: Quality score before refinement (0.0-1.0)
            quality_after: Quality score after refinement (0.0-1.0)
            latency_ms: Time taken for refinement in milliseconds
            model_provider: LLM provider used (claude, gemini, gpt, ollama)
            components_applied: List of 7-component sections applied
            metadata: Additional metadata

        Returns:
            Refinement record ID
        """
        cursor = self.conn.cursor()

        improvement = quality_after - quality_before
        components = components_applied or []

        cursor.execute("""
            INSERT INTO mrf_refinements
            (prompt_name, strategy, quality_before, quality_after, improvement,
             latency_ms, model_provider, components_applied, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_name,
            strategy,
            quality_before,
            quality_after,
            improvement,
            latency_ms,
            model_provider,
            json.dumps(components),
            json.dumps(metadata or {})
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_mrf_refinements(
        self,
        prompt_name: Optional[str] = None,
        strategy: Optional[str] = None,
        days: int = 30,
        limit: int = 100
    ) -> List[MRFRefinementDBRecord]:
        """
        Query MRF refinement history.

        Args:
            prompt_name: Filter by prompt name (optional)
            strategy: Filter by MRF strategy (optional)
            days: Number of days to look back (default: 30)
            limit: Maximum records to return (default: 100)

        Returns:
            List of MRFRefinementDBRecord objects
        """
        cursor = self.conn.cursor()

        query = """
            SELECT * FROM mrf_refinements
            WHERE created_at >= datetime('now', ?)
        """
        params = [f'-{days} days']

        if prompt_name:
            query += " AND prompt_name = ?"
            params.append(prompt_name)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        records = []
        for row in cursor.fetchall():
            records.append(MRFRefinementDBRecord(
                id=row["id"],
                prompt_name=row["prompt_name"],
                strategy=row["strategy"],
                quality_before=row["quality_before"],
                quality_after=row["quality_after"],
                improvement=row["improvement"],
                latency_ms=row["latency_ms"] or 0.0,
                model_provider=row["model_provider"] or "unknown",
                components_applied=json.loads(row["components_applied"]),
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"])
            ))
        return records

    def get_mrf_analytics(
        self,
        strategy: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get aggregate MRF analytics.

        Args:
            strategy: Filter by strategy (optional)
            days: Analysis window in days

        Returns:
            Analytics dict with stats by strategy
        """
        cursor = self.conn.cursor()

        # Base query for aggregate stats
        base_where = "WHERE created_at >= datetime('now', ?)"
        params = [f'-{days} days']

        if strategy:
            base_where += " AND strategy = ?"
            params.append(strategy)

        # Get overall stats
        cursor.execute(f"""
            SELECT
                COUNT(*) as total_refinements,
                AVG(quality_before) as avg_quality_before,
                AVG(quality_after) as avg_quality_after,
                AVG(improvement) as avg_improvement,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN quality_after >= 0.7 THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN improvement > 0 THEN 1 ELSE 0 END) as improvements
            FROM mrf_refinements
            {base_where}
        """, params)

        stats_row = cursor.fetchone()
        total = stats_row["total_refinements"] or 0

        if total == 0:
            return {
                "total_refinements": 0,
                "message": "No MRF refinements found in the specified window",
                "days_analyzed": days
            }

        # Get per-strategy breakdown
        cursor.execute(f"""
            SELECT
                strategy,
                COUNT(*) as count,
                AVG(quality_before) as avg_quality_before,
                AVG(quality_after) as avg_quality_after,
                AVG(improvement) as avg_improvement,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN quality_after >= 0.7 THEN 1 ELSE 0 END) as successes
            FROM mrf_refinements
            {base_where}
            GROUP BY strategy
            ORDER BY avg_improvement DESC
        """, params)

        strategy_breakdown = {}
        for row in cursor.fetchall():
            count = row["count"]
            strategy_breakdown[row["strategy"]] = {
                "count": count,
                "avg_quality_before": round(row["avg_quality_before"], 3) if row["avg_quality_before"] else None,
                "avg_quality_after": round(row["avg_quality_after"], 3) if row["avg_quality_after"] else None,
                "avg_improvement": round(row["avg_improvement"], 3) if row["avg_improvement"] else None,
                "avg_latency_ms": round(row["avg_latency_ms"], 1) if row["avg_latency_ms"] else None,
                "success_rate": round(row["successes"] / count, 3) if count else 0
            }

        return {
            "total_refinements": total,
            "avg_quality_before": round(stats_row["avg_quality_before"], 3) if stats_row["avg_quality_before"] else None,
            "avg_quality_after": round(stats_row["avg_quality_after"], 3) if stats_row["avg_quality_after"] else None,
            "avg_improvement": round(stats_row["avg_improvement"], 3) if stats_row["avg_improvement"] else None,
            "avg_improvement_percent": round((stats_row["avg_improvement"] / stats_row["avg_quality_before"]) * 100, 1) if stats_row["avg_quality_before"] and stats_row["avg_improvement"] else None,
            "avg_latency_ms": round(stats_row["avg_latency_ms"], 1) if stats_row["avg_latency_ms"] else None,
            "success_rate": round(stats_row["successes"] / total, 3) if total else 0,
            "improvement_rate": round(stats_row["improvements"] / total, 3) if total else 0,
            "strategy_breakdown": strategy_breakdown,
            "days_analyzed": days
        }

    def update_mrf_thompson_prior(
        self,
        strategy: str,
        success: bool,
        quality: float = 0.5
    ) -> Tuple[float, float]:
        """
        Update Thompson Sampling prior for an MRF strategy.

        Args:
            strategy: MRF strategy name (REFINE, CRITIQUE, VERIFY, ELEGANCE, HOFSTADTER)
            success: Whether the refinement was successful (quality >= 0.7 threshold)
            quality: Quality score to weight the update (0.0-1.0)

        Returns:
            Tuple of (alpha, beta) after update
        """
        cursor = self.conn.cursor()

        # Get or create prior for this strategy
        cursor.execute("""
            SELECT id, alpha, beta FROM mrf_thompson_priors
            WHERE strategy = ?
        """, (strategy,))

        row = cursor.fetchone()

        if row:
            # Update existing prior
            alpha = row["alpha"]
            beta = row["beta"]

            if success:
                alpha += quality
            else:
                beta += (1.0 - quality)

            cursor.execute("""
                UPDATE mrf_thompson_priors
                SET alpha = ?, beta = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (alpha, beta, row["id"]))
        else:
            # Create new prior with initial update
            alpha = 1.0 + (quality if success else 0.0)
            beta = 1.0 + (0.0 if success else (1.0 - quality))

            cursor.execute("""
                INSERT INTO mrf_thompson_priors (strategy, alpha, beta)
                VALUES (?, ?, ?)
            """, (strategy, alpha, beta))

        self.conn.commit()
        return (alpha, beta)

    def get_mrf_thompson_priors(
        self,
        strategy: Optional[str] = None
    ) -> List[MRFStrategyPrior]:
        """
        Get Thompson Sampling priors for MRF strategies.

        Args:
            strategy: Filter by strategy name (optional)

        Returns:
            List of MRFStrategyPrior objects
        """
        cursor = self.conn.cursor()

        if strategy:
            cursor.execute("""
                SELECT * FROM mrf_thompson_priors
                WHERE strategy = ?
                ORDER BY updated_at DESC
            """, (strategy,))
        else:
            cursor.execute("""
                SELECT * FROM mrf_thompson_priors
                ORDER BY updated_at DESC
            """)

        priors = []
        for row in cursor.fetchall():
            priors.append(MRFStrategyPrior(
                id=row["id"],
                strategy=row["strategy"],
                alpha=row["alpha"],
                beta=row["beta"],
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        return priors

    def get_mrf_strategy_recommendation(
        self,
        candidates: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get Thompson Sampling recommendation for best MRF strategy.

        Uses Beta distribution sampling to balance exploration/exploitation
        when selecting the most effective refinement strategy.

        Args:
            candidates: Optional list of strategy names to consider.
                       If None, considers all strategies with priors.

        Returns:
            Dict with recommended strategy and expected quality, or None
        """
        import random

        cursor = self.conn.cursor()

        if candidates:
            placeholders = ",".join(["?" for _ in candidates])
            cursor.execute(f"""
                SELECT * FROM mrf_thompson_priors
                WHERE strategy IN ({placeholders})
            """, candidates)
        else:
            cursor.execute("SELECT * FROM mrf_thompson_priors")

        rows = cursor.fetchall()

        if not rows:
            # No priors - return default strategy
            return {
                "recommended_strategy": "AUTO",
                "expected_quality": 0.5,
                "confidence": 0.5,
                "message": "No prior data available, using AUTO strategy",
                "alternatives": []
            }

        # Thompson Sampling: sample from Beta distribution for each strategy
        best_strategy = None
        best_sample = -1.0
        all_priors = []

        for row in rows:
            alpha = row["alpha"]
            beta = row["beta"]
            expected = alpha / (alpha + beta)

            # Sample from Beta distribution
            sample = random.betavariate(alpha, beta)

            all_priors.append({
                "strategy": row["strategy"],
                "expected_quality": round(expected, 3),
                "alpha": alpha,
                "beta": beta,
                "sample": round(sample, 3),
                "total_samples": alpha + beta - 2.0
            })

            if sample > best_sample:
                best_sample = sample
                best_strategy = row["strategy"]

        # Sort by expected quality for alternatives
        all_priors.sort(key=lambda x: x["expected_quality"], reverse=True)

        return {
            "recommended_strategy": best_strategy,
            "expected_quality": round(all_priors[0]["expected_quality"], 3) if all_priors else 0.5,
            "confidence": round(best_sample, 3),
            "alternatives": all_priors[:5]  # Top 5 alternatives
        }
