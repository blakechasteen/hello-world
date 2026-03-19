"""
Ritual Journal — event-sourced execution history.
==================================================

Append-only event log for ritual dispatch. Every checkpoint in ritual
execution (condition check, admission, body start/complete, policy invocation)
is recorded as a journal event. Enables replay and debugging.

Follows the Temporal pattern: log steps as events, replay to reconstruct.
The journal is optional — ritual execution never blocks on journal writes.

Storage: SQLite via the same pattern as SignalJournal. Can share the same
database or use a separate one.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RITUAL_JOURNAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS ritual_journal (
    id TEXT PRIMARY KEY,
    ritual_id TEXT NOT NULL,
    ritual_version TEXT NOT NULL,
    session_id TEXT NOT NULL,
    correlation_id TEXT,
    event_type TEXT NOT NULL,
    payload TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rj_ritual_id ON ritual_journal(ritual_id);
CREATE INDEX IF NOT EXISTS idx_rj_session_id ON ritual_journal(session_id);
CREATE INDEX IF NOT EXISTS idx_rj_correlation_id ON ritual_journal(correlation_id);
CREATE INDEX IF NOT EXISTS idx_rj_event_type ON ritual_journal(event_type);
CREATE INDEX IF NOT EXISTS idx_rj_created_at ON ritual_journal(created_at);
"""


@dataclass
class RitualJournalEvent:
    """A single event in the ritual journal."""
    id: str
    ritual_id: str
    ritual_version: str
    session_id: str
    correlation_id: str | None
    event_type: str
    payload: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Event types — the vocabulary of the journal
DISPATCH = "DISPATCH"
CONDITION_PASS = "CONDITION_PASS"
CONDITION_FAIL = "CONDITION_FAIL"
ADMISSION_PASS = "ADMISSION_PASS"
ADMISSION_FAIL = "ADMISSION_FAIL"
SKIP = "SKIP"
BODY_START = "BODY_START"
BODY_COMPLETE = "BODY_COMPLETE"
BODY_FAILED = "BODY_FAILED"
BODY_HEARTBEAT = "BODY_HEARTBEAT"
POLICY_INVOKED = "POLICY_INVOKED"
OUTCOME_WRITTEN = "OUTCOME_WRITTEN"


class RitualJournal:
    """
    SQLite-backed ritual execution journal.

    Append-only. Every ritual dispatch produces a sequence of events.
    Replay reconstructs the full execution trace.

    Usage:
        journal = RitualJournal(":memory:")
        journal.record("pre_code_health_check", "1.0.0", "sess-1", None, "DISPATCH")
        events = journal.replay_ritual("pre_code_health_check", "sess-1")
    """

    def __init__(self, db_path: str = "data/ritual_journal.db"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_RITUAL_JOURNAL_SCHEMA)
            self._conn.commit()
        return self._conn

    def record(
        self,
        ritual_id: str,
        ritual_version: str,
        session_id: str,
        correlation_id: str | None,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """
        Record a ritual journal event. Returns the event ID.
        Never raises — logs and continues on failure.
        """
        event_id = uuid.uuid4().hex[:12]
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO ritual_journal
                   (id, ritual_id, ritual_version, session_id,
                    correlation_id, event_type, payload, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    ritual_id,
                    ritual_version,
                    session_id,
                    correlation_id,
                    event_type,
                    json.dumps(payload or {}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.warning("Ritual journal write failed: %s", e)
        return event_id

    def replay_ritual(
        self,
        ritual_id: str,
        session_id: str,
        limit: int = 500,
    ) -> list[RitualJournalEvent]:
        """Replay all events for a ritual in a session, ordered by time."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM ritual_journal
               WHERE ritual_id = ? AND session_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (ritual_id, session_id, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def replay_session(
        self,
        session_id: str,
        limit: int = 1000,
    ) -> list[RitualJournalEvent]:
        """Replay all ritual events in a session, ordered by time."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM ritual_journal
               WHERE session_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def replay_from_correlation(
        self,
        correlation_id: str,
        limit: int = 500,
    ) -> list[RitualJournalEvent]:
        """Replay all ritual events for a correlation chain."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM ritual_journal
               WHERE correlation_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (correlation_id, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count_events(
        self,
        ritual_id: str | None = None,
        session_id: str | None = None,
        event_type: str | None = None,
    ) -> int:
        """Count journal events matching filters."""
        conn = self._get_conn()
        conditions = []
        params: list[str] = []
        if ritual_id:
            conditions.append("ritual_id = ?")
            params.append(ritual_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM ritual_journal {where}", params,
        ).fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> RitualJournalEvent:
        return RitualJournalEvent(
            id=row["id"],
            ritual_id=row["ritual_id"],
            ritual_version=row["ritual_version"],
            session_id=row["session_id"],
            correlation_id=row["correlation_id"],
            event_type=row["event_type"],
            payload=json.loads(row["payload"]),
            created_at=row["created_at"],
        )
