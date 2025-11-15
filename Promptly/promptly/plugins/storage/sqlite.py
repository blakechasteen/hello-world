#!/usr/bin/env python3
"""
SQLite Storage Backend Plugin

Refactored from the original PromptlyDB implementation.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..base import BaseStorageBackend


class SQLiteStorage(BaseStorageBackend):
    """SQLite-based storage backend for Promptly"""

    def __init__(self):
        super().__init__(
            name="sqlite",
            description="SQLite database storage backend (default)"
        )
        self.db_path = None
        self.conn = None

    def init_storage(self, storage_path: str) -> None:
        """Initialize SQLite database"""
        self.db_path = storage_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                branch TEXT NOT NULL DEFAULT 'main',
                version INTEGER NOT NULL DEFAULT 1,
                parent_id INTEGER,
                commit_hash TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (parent_id) REFERENCES prompts(id)
            )
        """)

        # Branches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                head_commit TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (head_commit) REFERENCES prompts(commit_hash)
            )
        """)

        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_name TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                test_case TEXT NOT NULL,
                expected TEXT,
                actual TEXT,
                score REAL,
                metrics TEXT,
                evaluator_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (commit_hash) REFERENCES prompts(commit_hash)
            )
        """)

        # Chains table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                steps TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Config table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Initialize main branch
        cursor.execute("INSERT OR IGNORE INTO branches (name, head_commit) VALUES ('main', 'init')")
        cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('current_branch', 'main')")

        self.conn.commit()

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Save a prompt to the database"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata')

        timestamp = datetime.now().isoformat()
        commit_hash = self._generate_commit_hash(name, content, timestamp)

        cursor = self.conn.cursor()

        # Check if prompt exists on this branch
        cursor.execute("""
            SELECT id, version, commit_hash FROM prompts
            WHERE name = ? AND branch = ?
            ORDER BY version DESC LIMIT 1
        """, (name, branch))

        existing = cursor.fetchone()
        parent_id = existing[0] if existing else None
        version = existing[1] + 1 if existing else 1

        # Insert new version
        cursor.execute("""
            INSERT INTO prompts (name, content, branch, version, parent_id, commit_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, content, branch, version, parent_id, commit_hash,
              json.dumps(metadata) if metadata else None))

        # Update branch head
        cursor.execute("""
            UPDATE branches SET head_commit = ? WHERE name = ?
        """, (commit_hash, branch))

        self.conn.commit()

        return commit_hash

    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt from the database"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()

        if commit_hash:
            cursor.execute("""
                SELECT name, content, branch, version, commit_hash, created_at, metadata
                FROM prompts WHERE name = ? AND commit_hash = ?
            """, (name, commit_hash))
        elif version:
            cursor.execute("""
                SELECT name, content, branch, version, commit_hash, created_at, metadata
                FROM prompts WHERE name = ? AND branch = ? AND version = ?
            """, (name, branch, version))
        else:
            cursor.execute("""
                SELECT name, content, branch, version, commit_hash, created_at, metadata
                FROM prompts WHERE name = ? AND branch = ?
                ORDER BY version DESC LIMIT 1
            """, (name, branch))

        result = cursor.fetchone()

        if not result:
            return None

        return {
            'name': result[0],
            'content': result[1],
            'branch': result[2],
            'version': result[3],
            'commit_hash': result[4],
            'created_at': result[5],
            'metadata': json.loads(result[6]) if result[6] else {}
        }

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on a branch"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT name, MAX(version) as version, commit_hash, created_at
            FROM prompts
            WHERE branch = ?
            GROUP BY name
            ORDER BY name
        """, (branch,))

        results = cursor.fetchall()
        return [dict(row) for row in results]

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()

        # Get head commit of source branch
        cursor.execute("SELECT head_commit FROM branches WHERE name = ?", (from_branch,))
        result = cursor.fetchone()

        if not result:
            raise Exception(f"Branch '{from_branch}' does not exist")

        head_commit = result[0]

        # Create new branch
        try:
            cursor.execute("""
                INSERT INTO branches (name, head_commit)
                VALUES (?, ?)
            """, (branch_name, head_commit))

            # Copy prompts from source branch
            cursor.execute("""
                INSERT INTO prompts (name, content, branch, version, parent_id, commit_hash, metadata)
                SELECT name, content, ?, version, parent_id,
                       substr(hex(randomblob(6)), 1, 12), metadata
                FROM prompts
                WHERE branch = ? AND version IN (
                    SELECT MAX(version) FROM prompts WHERE branch = ? GROUP BY name
                )
            """, (branch_name, from_branch, from_branch))

            self.conn.commit()

        except sqlite3.IntegrityError:
            raise Exception(f"Branch '{branch_name}' already exists")

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = 'current_branch'")
        result = cursor.fetchone()
        return result[0] if result else 'main'

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()

        # Check if branch exists
        cursor.execute("SELECT name FROM branches WHERE name = ?", (branch_name,))
        if not cursor.fetchone():
            raise Exception(f"Branch '{branch_name}' does not exist")

        # Update current branch
        cursor.execute("UPDATE config SET value = ? WHERE key = 'current_branch'", (branch_name,))
        self.conn.commit()

    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get commit history"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()

        if name:
            cursor.execute("""
                SELECT commit_hash, name, version, branch, created_at
                FROM prompts
                WHERE name = ? AND branch = ?
                ORDER BY version DESC
                LIMIT ?
            """, (name, branch, limit))
        else:
            cursor.execute("""
                SELECT commit_hash, name, version, branch, created_at
                FROM prompts
                WHERE branch = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (branch, limit))

        results = cursor.fetchall()
        return [dict(row) for row in results]

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO evaluations (prompt_name, commit_hash, test_case, expected,
                                    actual, score, metrics, evaluator_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_data['prompt_name'],
            eval_data['commit_hash'],
            json.dumps(eval_data.get('test_case', {})),
            eval_data.get('expected'),
            eval_data.get('actual'),
            eval_data.get('score'),
            json.dumps(eval_data.get('metrics', {})),
            eval_data.get('evaluator_name')
        ))

        self.conn.commit()

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO chains (name, steps, description)
                VALUES (?, ?, ?)
            """, (
                chain_data['name'],
                json.dumps(chain_data['steps']),
                chain_data.get('description')
            ))

            self.conn.commit()

        except sqlite3.IntegrityError:
            raise Exception(f"Chain '{chain_data['name']}' already exists")

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        if not self.conn:
            raise RuntimeError("Storage not initialized")

        cursor = self.conn.cursor()
        cursor.execute("SELECT name, steps, description FROM chains WHERE name = ?", (name,))
        result = cursor.fetchone()

        if not result:
            return None

        return {
            'name': result[0],
            'steps': json.loads(result[1]),
            'description': result[2]
        }

    def close(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
