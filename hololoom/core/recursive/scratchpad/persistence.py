#!/usr/bin/env python3
"""
Thought Persistence
===================

SQLite backend for persistent scratchpad storage.

Created: 2025-01-20

Schema:
- sessions: Session metadata
- thoughts: Individual thoughts
- dialogues: Dialogue tree structure

Features:
- Save/load complete dialogue trees
- Session management
- Thought history
- Search capabilities
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from hololoom.recursive.scratchpad.recursive_scratchpad import (
    DialogueTree,
    Thought,
    ThoughtType,
)


class ThoughtPersistence:
    """SQLite persistence for scratchpad thoughts."""

    def __init__(self, db_path: Path):
        """
        Initialize persistence.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database schema."""
        self.db = await aiosqlite.connect(str(self.db_path))

        # Enable foreign keys
        await self.db.execute("PRAGMA foreign_keys = ON")

        # Create tables
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS thoughts (
                thought_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT NOT NULL,
                type TEXT NOT NULL,
                level INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                parent_id TEXT,
                confidence REAL DEFAULT 0.5,
                metadata TEXT,
                verification TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES thoughts(thought_id)
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS dialogues (
                dialogue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                root_thought_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (root_thought_id) REFERENCES thoughts(thought_id)
            )
        """)

        # Create indices
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_thoughts_session
            ON thoughts(session_id)
        """)

        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_thoughts_parent
            ON thoughts(parent_id)
        """)

        await self.db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self.db:
            await self.db.close()
            self.db = None

    async def save_session(
        self,
        session_name: str,
        tree: DialogueTree
    ) -> str:
        """
        Save dialogue tree as session.

        Args:
            session_name: Unique session name
            tree: Dialogue tree to save

        Returns:
            Session ID
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        session_id = session_name
        now = datetime.now().isoformat()

        # Upsert session
        await self.db.execute("""
            INSERT INTO sessions (session_id, session_name, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_name)
            DO UPDATE SET updated_at = ?, metadata = ?
        """, (session_id, session_name, now, now, "{}", now, "{}"))

        # Delete existing thoughts for this session (if any)
        await self.db.execute("""
            DELETE FROM thoughts WHERE session_id = ?
        """, (session_id,))

        # Save all thoughts
        for thought in tree.thoughts.values():
            await self.db.execute("""
                INSERT INTO thoughts (
                    thought_id, session_id, text, type, level, timestamp,
                    parent_id, confidence, metadata, verification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                thought.id,
                session_id,
                thought.text,
                thought.type.value,
                thought.level,
                thought.timestamp.isoformat(),
                thought.parent_id,
                thought.confidence,
                json.dumps(thought.metadata),
                json.dumps(thought.verification) if thought.verification else None
            ))

        # Save dialogue entry
        await self.db.execute("""
            INSERT INTO dialogues (session_id, root_thought_id, created_at)
            VALUES (?, ?, ?)
        """, (session_id, tree.root.id, now))

        await self.db.commit()
        return session_id

    async def load_session(self, session_name: str) -> DialogueTree | None:
        """
        Load dialogue tree from session.

        Args:
            session_name: Session name

        Returns:
            Dialogue tree or None
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        # Get session
        async with self.db.execute("""
            SELECT session_id FROM sessions WHERE session_name = ?
        """, (session_name,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            session_id = row[0]

        # Load thoughts
        thoughts = {}
        async with self.db.execute("""
            SELECT
                thought_id, text, type, level, timestamp,
                parent_id, confidence, metadata, verification
            FROM thoughts
            WHERE session_id = ?
            ORDER BY level ASC, timestamp ASC
        """, (session_id,)) as cursor:
            async for row in cursor:
                thought = Thought(
                    id=row[0],
                    text=row[1],
                    type=ThoughtType(row[2]),
                    level=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    parent_id=row[5],
                    confidence=row[6],
                    metadata=json.loads(row[7]) if row[7] else {},
                    verification=json.loads(row[8]) if row[8] else None
                )
                thoughts[thought.id] = thought

        if not thoughts:
            return None

        # Find root
        root = None
        for thought in thoughts.values():
            if thought.parent_id is None or thought.parent_id not in thoughts:
                root = thought
                break

        if not root:
            # Fallback: use first thought
            root = list(thoughts.values())[0]

        # Construct tree
        tree = DialogueTree(root=root, thoughts=thoughts)
        return tree

    async def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session metadata
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        sessions = []
        async with self.db.execute("""
            SELECT
                s.session_id,
                s.session_name,
                s.created_at,
                s.updated_at,
                COUNT(t.thought_id) as thought_count
            FROM sessions s
            LEFT JOIN thoughts t ON s.session_id = t.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
        """) as cursor:
            async for row in cursor:
                sessions.append({
                    'session_id': row[0],
                    'session_name': row[1],
                    'created_at': row[2],
                    'updated_at': row[3],
                    'thought_count': row[4]
                })

        return sessions

    async def delete_session(self, session_name: str) -> bool:
        """
        Delete session.

        Args:
            session_name: Session name

        Returns:
            True if deleted
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        result = await self.db.execute("""
            DELETE FROM sessions WHERE session_name = ?
        """, (session_name,))

        await self.db.commit()
        return result.rowcount > 0

    async def search_thoughts(
        self,
        query: str,
        session_name: str | None = None,
        limit: int = 10
    ) -> list[Thought]:
        """
        Search thoughts by text.

        Args:
            query: Search query
            session_name: Optional session filter
            limit: Max results

        Returns:
            List of matching thoughts
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        sql = """
            SELECT
                t.thought_id, t.text, t.type, t.level, t.timestamp,
                t.parent_id, t.confidence, t.metadata, t.verification
            FROM thoughts t
        """

        params = []

        if session_name:
            sql += """
                JOIN sessions s ON t.session_id = s.session_id
                WHERE s.session_name = ? AND t.text LIKE ?
            """
            params = [session_name, f"%{query}%"]
        else:
            sql += " WHERE t.text LIKE ?"
            params = [f"%{query}%"]

        sql += f" LIMIT {limit}"

        thoughts = []
        async with self.db.execute(sql, params) as cursor:
            async for row in cursor:
                thought = Thought(
                    id=row[0],
                    text=row[1],
                    type=ThoughtType(row[2]),
                    level=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    parent_id=row[5],
                    confidence=row[6],
                    metadata=json.loads(row[7]) if row[7] else {},
                    verification=json.loads(row[8]) if row[8] else None
                )
                thoughts.append(thought)

        return thoughts


class SessionManager:
    """
    High-level session management.

    Provides convenience methods for common operations.
    """

    def __init__(self, persistence: ThoughtPersistence):
        """
        Initialize session manager.

        Args:
            persistence: Thought persistence backend
        """
        self.persistence = persistence

    async def create_session(self, name: str) -> str:
        """Create new session."""
        # Session creation is implicit in save
        return name

    async def get_session(self, name: str) -> DialogueTree | None:
        """Get session by name."""
        return await self.persistence.load_session(name)

    async def list_all_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        return await self.persistence.list_sessions()

    async def delete_session(self, name: str) -> bool:
        """Delete session."""
        return await self.persistence.delete_session(name)

    async def search_across_sessions(
        self,
        query: str,
        limit: int = 20
    ) -> list[Thought]:
        """Search thoughts across all sessions."""
        return await self.persistence.search_thoughts(query, limit=limit)

    async def get_session_stats(self, name: str) -> dict[str, Any] | None:
        """
        Get session statistics.

        Returns:
            Stats dict or None
        """
        tree = await self.get_session(name)
        if not tree:
            return None

        # Compute stats
        thoughts = list(tree.thoughts.values())

        stats = {
            'name': name,
            'total_thoughts': len(thoughts),
            'max_depth': tree.get_depth(),
            'thought_types': {},
            'avg_confidence': sum(t.confidence for t in thoughts) / len(thoughts),
            'questions': len([t for t in thoughts if t.type == ThoughtType.QUESTION]),
            'answers': len([t for t in thoughts if t.type == ThoughtType.ANSWER]),
            'reflections': len([t for t in thoughts if t.type == ThoughtType.REFLECTION]),
        }

        # Count by type
        for thought in thoughts:
            type_name = thought.type.value
            stats['thought_types'][type_name] = stats['thought_types'].get(type_name, 0) + 1

        return stats
