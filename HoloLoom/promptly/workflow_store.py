"""
WorkflowStore Integration Layer

Extends PromptlyDB to provide workflow-specific versioning, branching, and
diffing capabilities for prompt workflows and execution pipelines.
"""

import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


class WorkflowStore:
    """Extends PromptlyDB with workflow-specific storage and retrieval."""

    def __init__(self, db_path: str):
        """
        Initialize WorkflowStore with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._init_tables()

    def connect(self) -> sqlite3.Connection:
        """
        Establish database connection.

        Returns:
            SQLite connection object
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _init_tables(self):
        """Initialize workflow tables if they don't exist."""
        conn = self.connect()
        cursor = conn.cursor()

        # Workflows table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                workflow_json TEXT NOT NULL,
                branch TEXT NOT NULL DEFAULT 'main',
                version INTEGER NOT NULL DEFAULT 1,
                parent_id INTEGER,
                commit_hash TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message TEXT,
                metadata TEXT,
                FOREIGN KEY (parent_id) REFERENCES workflows(id),
                UNIQUE(name, branch, version)
            )
        """)

        # Branches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_branches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                head_commit TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize main branch
        cursor.execute(
            "INSERT OR IGNORE INTO workflow_branches (name, head_commit) VALUES ('main', 'init')"
        )

        conn.commit()

    def _generate_commit_hash(self, workflow_dict: Dict) -> str:
        """
        Generate commit hash from workflow dictionary.

        Args:
            workflow_dict: Workflow configuration dictionary

        Returns:
            12-character SHA256 hash
        """
        workflow_json = json.dumps(workflow_dict, sort_keys=True)
        timestamp = datetime.now().isoformat()
        data = f"{workflow_json}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def save_workflow(
        self,
        name: str,
        workflow_dict: Dict,
        branch: str = "main",
        message: str = "",
    ) -> str:
        """
        Save a workflow with automatic version incrementing.

        Args:
            name: Workflow name (unique within branch)
            workflow_dict: Workflow configuration as dictionary
            branch: Git-like branch name (default: 'main')
            message: Commit message describing changes

        Returns:
            Commit hash of saved workflow

        Raises:
            ValueError: If workflow_dict is not a valid dictionary
        """
        if not isinstance(workflow_dict, dict):
            raise ValueError("workflow_dict must be a dictionary")

        conn = self.connect()
        cursor = conn.cursor()

        # Generate commit hash
        commit_hash = self._generate_commit_hash(workflow_dict)

        # Get next version number
        cursor.execute(
            """
            SELECT MAX(version) as max_version FROM workflows
            WHERE name = ? AND branch = ?
            """,
            (name, branch),
        )
        result = cursor.fetchone()
        version = (result["max_version"] or 0) + 1

        # Get parent ID
        cursor.execute(
            """
            SELECT id FROM workflows
            WHERE name = ? AND branch = ?
            ORDER BY version DESC LIMIT 1
            """,
            (name, branch),
        )
        parent_result = cursor.fetchone()
        parent_id = parent_result["id"] if parent_result else None

        # Insert workflow
        workflow_json = json.dumps(workflow_dict, indent=2)
        cursor.execute(
            """
            INSERT INTO workflows (name, workflow_json, branch, version, parent_id,
                                  commit_hash, message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                workflow_json,
                branch,
                version,
                parent_id,
                commit_hash,
                message,
                json.dumps({"size": len(workflow_json), "keys": list(workflow_dict.keys())}),
            ),
        )

        # Update branch head
        cursor.execute(
            "UPDATE workflow_branches SET head_commit = ? WHERE name = ?",
            (commit_hash, branch),
        )

        conn.commit()
        return commit_hash

    def get_workflow(
        self,
        name: str,
        version: Optional[int] = None,
        branch: str = "main",
    ) -> Optional[Dict]:
        """
        Retrieve a workflow by name and optional version.

        Args:
            name: Workflow name
            version: Specific version number (default: latest)
            branch: Branch name (default: 'main')

        Returns:
            Workflow dictionary or None if not found
        """
        conn = self.connect()
        cursor = conn.cursor()

        if version is not None:
            cursor.execute(
                """
                SELECT workflow_json, version, commit_hash, branch, created_at, message
                FROM workflows
                WHERE name = ? AND branch = ? AND version = ?
                """,
                (name, branch, version),
            )
        else:
            cursor.execute(
                """
                SELECT workflow_json, version, commit_hash, branch, created_at, message
                FROM workflows
                WHERE name = ? AND branch = ?
                ORDER BY version DESC LIMIT 1
                """,
                (name, branch),
            )

        result = cursor.fetchone()
        if not result:
            return None

        workflow_dict = json.loads(result["workflow_json"])
        workflow_dict["_metadata"] = {
            "version": result["version"],
            "commit_hash": result["commit_hash"],
            "branch": result["branch"],
            "created_at": result["created_at"],
            "message": result["message"],
        }

        return workflow_dict

    def list_workflows(self, branch: str = "main") -> List[Dict]:
        """
        List all workflows on a specific branch.

        Args:
            branch: Branch name (default: 'main')

        Returns:
            List of workflow metadata dictionaries
        """
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name, MAX(version) as version, commit_hash, created_at, message
            FROM workflows
            WHERE branch = ?
            GROUP BY name
            ORDER BY name
            """,
            (branch,),
        )

        results = cursor.fetchall()
        return [dict(row) for row in results]

    def branch_workflow(
        self, name: str, from_branch: str, to_branch: str
    ) -> str:
        """
        Create a new branch from an existing workflow.

        Args:
            name: Workflow name
            from_branch: Source branch (default: 'main')
            to_branch: Target branch name

        Returns:
            Commit hash of new branch head

        Raises:
            ValueError: If source workflow not found
        """
        # Get source workflow
        source = self.get_workflow(name, branch=from_branch)
        if not source is None and "_metadata" in source:
            del source["_metadata"]

        if source is None:
            raise ValueError(f"Workflow '{name}' not found on branch '{from_branch}'")

        # Save to new branch
        commit_hash = self._generate_commit_hash(source)
        conn = self.connect()
        cursor = conn.cursor()

        # Create new branch if needed
        cursor.execute(
            "INSERT OR IGNORE INTO workflow_branches (name, head_commit) VALUES (?, ?)",
            (to_branch, commit_hash),
        )

        # Insert workflow on new branch (version 1)
        workflow_json = json.dumps(source, indent=2)
        cursor.execute(
            """
            INSERT INTO workflows (name, workflow_json, branch, version, parent_id,
                                  commit_hash, message, metadata)
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                name,
                workflow_json,
                to_branch,
                1,
                commit_hash,
                f"Branched from {from_branch}",
                json.dumps({"branched_from": from_branch}),
            ),
        )

        conn.commit()
        return commit_hash

    def diff_workflows(
        self, name: str, version1: int, version2: int, branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Compare two workflow versions and return differences.

        Args:
            name: Workflow name
            version1: First version number
            version2: Second version number
            branch: Branch name (default: 'main')

        Returns:
            Dictionary with 'added', 'removed', 'modified', 'unchanged' keys
        """
        w1 = self.get_workflow(name, version=version1, branch=branch)
        w2 = self.get_workflow(name, version=version2, branch=branch)

        if w1 is None or w2 is None:
            raise ValueError(f"One or both versions not found for workflow '{name}'")

        # Clean metadata for comparison
        if "_metadata" in w1:
            del w1["_metadata"]
        if "_metadata" in w2:
            del w2["_metadata"]

        changes = {
            "added": {},
            "removed": {},
            "modified": {},
            "unchanged": {},
            "version1": version1,
            "version2": version2,
        }

        # Compare keys
        keys1 = set(w1.keys())
        keys2 = set(w2.keys())

        # Added keys
        for key in keys2 - keys1:
            changes["added"][key] = w2[key]

        # Removed keys
        for key in keys1 - keys2:
            changes["removed"][key] = w1[key]

        # Modified and unchanged keys
        for key in keys1 & keys2:
            if w1[key] == w2[key]:
                changes["unchanged"][key] = w1[key]
            else:
                changes["modified"][key] = {"before": w1[key], "after": w2[key]}

        return changes

    def delete_workflow(self, name: str, version: int, branch: str = "main") -> bool:
        """
        Delete a specific workflow version (soft delete via marking).

        Args:
            name: Workflow name
            version: Version to delete
            branch: Branch name (default: 'main')

        Returns:
            True if deleted, False if not found
        """
        conn = self.connect()
        cursor = conn.cursor()

        # Mark as deleted by setting empty JSON
        cursor.execute(
            """
            UPDATE workflows
            SET workflow_json = NULL
            WHERE name = ? AND version = ? AND branch = ?
            """,
            (name, version, branch),
        )

        if cursor.rowcount == 0:
            return False

        conn.commit()
        return True
