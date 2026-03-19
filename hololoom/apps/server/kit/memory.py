"""
ConversationMemory — SQLite-backed rolling conversation history per room.

Extracted from promptly/memory.py and elle_chat.py. Parameterized by label,
db path, and turn count so any agent can use the same implementation.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DATA_DIR = _REPO_ROOT / "data"


# ============================================================================
# Protocol — the contract any memory implementation must satisfy
# ============================================================================

@runtime_checkable
class Memory(Protocol):
    """Memory protocol for agent conversation history.

    Implementations: ConversationMemory (SQLite), NullMemory (ephemeral).
    Swap freely — agent() and deploy() accept any Memory.
    """
    max_turns: int

    def get_messages(self, room_id: str) -> list[dict[str, str]]: ...
    def add_turn(self, room_id: str, user_msg: str, assistant_msg: str) -> None: ...
    def room_count(self) -> int: ...
    def turn_count(self, room_id: str) -> int: ...
    def total_turns(self) -> int: ...


class ConversationMemory:
    """SQLite-backed rolling conversation history per room."""

    def __init__(
        self,
        db_path: str,
        max_turns: int = 20,
        label: str = "agent",
    ) -> None:
        self.max_turns = max_turns
        self.db_path = db_path
        self._label = label
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL,
                user_msg TEXT NOT NULL,
                assistant_msg TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{label}_turns_room ON turns(room_id, id)"
        )
        self._conn.commit()
        logger.info(
            "%s memory: %s (%d rooms, %d total turns)",
            label.capitalize(), db_path, self.room_count(), self.total_turns(),
        )

    def add_turn(self, room_id: str, user_msg: str, assistant_msg: str) -> None:
        self._conn.execute(
            "INSERT INTO turns (room_id, user_msg, assistant_msg, created_at) "
            "VALUES (?, ?, ?, ?)",
            (room_id, user_msg, assistant_msg, datetime.now().isoformat()),
        )
        self._conn.execute("""
            DELETE FROM turns WHERE id IN (
                SELECT id FROM turns WHERE room_id = ?
                ORDER BY id DESC LIMIT -1 OFFSET ?
            )
        """, (room_id, self.max_turns))
        self._conn.commit()

    def get_messages(self, room_id: str) -> list[dict[str, str]]:
        """Build Ollama messages array from history."""
        rows = self._conn.execute(
            "SELECT user_msg, assistant_msg FROM turns "
            "WHERE room_id = ? ORDER BY id ASC",
            (room_id,),
        ).fetchall()
        messages = []
        for user_msg, assistant_msg in rows:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def room_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT room_id) FROM turns"
        ).fetchone()
        return row[0] if row else 0

    def turn_count(self, room_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM turns WHERE room_id = ?", (room_id,),
        ).fetchone()
        return row[0] if row else 0

    def total_turns(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()
        return row[0] if row else 0


def conversation(
    turns: int = 20,
    db: str = "memory.db",
    label: str = "agent",
) -> ConversationMemory:
    """Factory: create a ConversationMemory with resolved db path in data/."""
    db_env = os.environ.get(f"{label.upper()}_MEMORY_DB", "")
    if db_env:
        db_path = db_env
    else:
        db_path = str(_DATA_DIR / db)
    return ConversationMemory(db_path=db_path, max_turns=turns, label=label)
