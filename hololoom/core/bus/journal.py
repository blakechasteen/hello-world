"""
Signal Journal — SQLite-backed execution history.
==================================================

Append-only event log for all signals flowing through the UnifiedBus.
Answers "what happened?" for debugging, MCTS backpropagation, and observability.

Tables:
  signal_journal    — every signal emitted (append-heavy)
  execution_trace   — one row per chain/workflow execution
  step_result       — one row per step within an execution
  mcts_decision     — MCTS search decisions for strategy learning

Design: aiosqlite for async, WAL mode for concurrent reads during writes,
indexes on correlation_id and kind for fast correlation queries.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .signal import Signal

logger = logging.getLogger(__name__)

# Use synchronous sqlite3 — simpler, reliable, and the journal writes
# are fast enough that async overhead isn't justified. If aiosqlite is
# available, a future version can swap in.

SCHEMA_VERSION = 1

_SCHEMA = """
-- Signal journal: every signal emitted through the bus
CREATE TABLE IF NOT EXISTS signal_journal (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    priority INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    correlation_id TEXT,
    causation_id TEXT,
    trigger_mode TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    ttl_seconds INTEGER
);

CREATE INDEX IF NOT EXISTS idx_signal_correlation ON signal_journal(correlation_id);
CREATE INDEX IF NOT EXISTS idx_signal_kind ON signal_journal(kind);
CREATE INDEX IF NOT EXISTS idx_signal_created ON signal_journal(created_at);
CREATE INDEX IF NOT EXISTS idx_signal_source ON signal_journal(source);

-- Execution traces: one row per chain/workflow execution
CREATE TABLE IF NOT EXISTS execution_trace (
    id TEXT PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    execution_type TEXT NOT NULL,
    name TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    success INTEGER,
    total_steps INTEGER,
    duration_ms REAL,
    final_confidence REAL,
    error TEXT,
    context_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trace_correlation ON execution_trace(correlation_id);
CREATE INDEX IF NOT EXISTS idx_trace_type ON execution_trace(execution_type);
CREATE INDEX IF NOT EXISTS idx_trace_started ON execution_trace(started_at);

-- Step results: one row per step within an execution
CREATE TABLE IF NOT EXISTS step_result (
    id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_type TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL,
    output_json TEXT,
    error TEXT,
    model_used TEXT,
    confidence REAL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (execution_id) REFERENCES execution_trace(id)
);

CREATE INDEX IF NOT EXISTS idx_step_execution ON step_result(execution_id);
CREATE INDEX IF NOT EXISTS idx_step_type ON step_result(step_type);

-- MCTS decisions: strategic + tactical tree search results
CREATE TABLE IF NOT EXISTS mcts_decision (
    id TEXT PRIMARY KEY,
    correlation_id TEXT,
    tree_level TEXT NOT NULL,
    domain TEXT,
    simulations_run INTEGER,
    time_budget_ms REAL,
    selected_action TEXT,
    root_visits INTEGER,
    best_value REAL,
    alternatives_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mcts_domain ON mcts_decision(domain);
CREATE INDEX IF NOT EXISTS idx_mcts_created ON mcts_decision(created_at);
CREATE INDEX IF NOT EXISTS idx_mcts_correlation ON mcts_decision(correlation_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS journal_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SignalJournal:
    """
    SQLite-backed signal journal.

    Provides:
    - append(signal): Record a signal
    - record_execution(): Record a chain/workflow execution
    - record_step(): Record a step within an execution
    - record_mcts_decision(): Record an MCTS search result
    - query_signals(): Query signals by filters
    - query_traces(): Query execution traces
    - get_workflow_history(): Full history for a correlation chain
    """

    def __init__(self, db_path: str = "data/signal_journal.db"):
        """
        Initialize journal.

        Args:
            db_path: Path to SQLite database file.
                     Use ":memory:" for in-memory (testing).
        """
        self.db_path = db_path

        # Ensure parent directory exists
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create connection with WAL mode."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)

            # Set schema version
            self._conn.execute(
                "INSERT OR REPLACE INTO journal_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self._conn.commit()
        return self._conn

    # ========================================================================
    # Signal Journal
    # ========================================================================

    async def append(self, signal: Signal) -> None:
        """Append a signal to the journal."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO signal_journal
                   (id, kind, priority, source, target, payload, metadata,
                    correlation_id, causation_id, trigger_mode, created_at, ttl_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.id,
                    signal.kind.value,
                    signal.priority.value,
                    signal.source,
                    signal.target,
                    json.dumps(signal.payload),
                    json.dumps(signal.metadata),
                    signal.correlation_id,
                    signal.causation_id,
                    signal.trigger_mode.value,
                    signal.created_at,
                    signal.ttl_seconds,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Journal append failed: {e}")

    async def query_signals(
        self,
        correlation_id: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query signals with filters."""
        conn = self._get_conn()
        conditions = []
        params = []

        if correlation_id:
            conditions.append("correlation_id = ?")
            params.append(correlation_id)
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM signal_journal {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ========================================================================
    # Execution Traces
    # ========================================================================

    async def record_execution(
        self,
        trace_id: str,
        correlation_id: str,
        execution_type: str,
        name: str | None = None,
        started_at: str = "",
        completed_at: str | None = None,
        success: bool | None = None,
        total_steps: int | None = None,
        duration_ms: float | None = None,
        final_confidence: float | None = None,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record or update an execution trace."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO execution_trace
               (id, correlation_id, execution_type, name, started_at,
                completed_at, success, total_steps, duration_ms,
                final_confidence, error, context_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                correlation_id,
                execution_type,
                name,
                started_at,
                completed_at,
                1 if success else (0 if success is not None else None),
                total_steps,
                duration_ms,
                final_confidence,
                error,
                json.dumps(context) if context else None,
            ),
        )
        conn.commit()

    async def query_traces(
        self,
        execution_type: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query execution traces."""
        conn = self._get_conn()
        conditions = []
        params = []

        if execution_type:
            conditions.append("execution_type = ?")
            params.append(execution_type)
        if success is not None:
            conditions.append("success = ?")
            params.append(1 if success else 0)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM execution_trace {where} ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ========================================================================
    # Step Results
    # ========================================================================

    async def record_step(
        self,
        step_record_id: str,
        execution_id: str,
        step_id: str,
        step_type: str,
        status: str,
        timestamp: str,
        duration_ms: float | None = None,
        output: Any | None = None,
        error: str | None = None,
        model_used: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Record a step result."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO step_result
               (id, execution_id, step_id, step_type, status, duration_ms,
                output_json, error, model_used, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                step_record_id,
                execution_id,
                step_id,
                step_type,
                status,
                duration_ms,
                json.dumps(output) if output is not None else None,
                error,
                model_used,
                confidence,
                timestamp,
            ),
        )
        conn.commit()

    async def query_steps(self, execution_id: str) -> list[dict[str, Any]]:
        """Get all steps for an execution."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM step_result WHERE execution_id = ? ORDER BY timestamp",
            (execution_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ========================================================================
    # MCTS Decisions
    # ========================================================================

    async def record_mcts_decision(
        self,
        decision_id: str,
        tree_level: str,
        created_at: str,
        correlation_id: str | None = None,
        domain: str | None = None,
        simulations_run: int | None = None,
        time_budget_ms: float | None = None,
        selected_action: str | None = None,
        root_visits: int | None = None,
        best_value: float | None = None,
        alternatives: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record an MCTS decision."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO mcts_decision
               (id, correlation_id, tree_level, domain, simulations_run,
                time_budget_ms, selected_action, root_visits, best_value,
                alternatives_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id,
                correlation_id,
                tree_level,
                domain,
                simulations_run,
                time_budget_ms,
                selected_action,
                root_visits,
                best_value,
                json.dumps(alternatives) if alternatives else None,
                created_at,
            ),
        )
        conn.commit()

    async def query_mcts_decisions(
        self,
        domain: str | None = None,
        tree_level: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query MCTS decisions."""
        conn = self._get_conn()
        conditions = []
        params = []

        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if tree_level:
            conditions.append("tree_level = ?")
            params.append(tree_level)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM mcts_decision {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ========================================================================
    # Workflow History (full correlation chain)
    # ========================================================================

    async def get_workflow_history(self, correlation_id: str) -> dict[str, Any]:
        """
        Get complete history for a workflow (correlation chain).

        Returns all signals, traces, steps, and MCTS decisions linked
        by correlation_id.
        """
        signals = await self.query_signals(correlation_id=correlation_id, limit=1000)

        conn = self._get_conn()
        traces = [
            dict(r) for r in conn.execute(
                "SELECT * FROM execution_trace WHERE correlation_id = ? ORDER BY started_at",
                (correlation_id,),
            ).fetchall()
        ]

        steps = []
        for trace in traces:
            trace_steps = await self.query_steps(trace["id"])
            steps.extend(trace_steps)

        mcts = [
            dict(r) for r in conn.execute(
                "SELECT * FROM mcts_decision WHERE correlation_id = ? ORDER BY created_at",
                (correlation_id,),
            ).fetchall()
        ]

        return {
            "correlation_id": correlation_id,
            "signals": signals,
            "traces": traces,
            "steps": steps,
            "mcts_decisions": mcts,
        }

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()


# ============================================================================
# Exports
# ============================================================================

__all__ = ["SignalJournal"]
