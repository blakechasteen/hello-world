"""
Causation Tracker — trace from outcomes back to decisions.

Credit assignment is the hard problem: when a federation note gets promoted
three days later, which tool selection, which model routing decision, which
rubric scoring made that outcome possible? The CausationTracker records every
decision point and its correlation ID, then walks the chain backward when
outcomes arrive.

Temporal discount: 0.9^depth per hop in the chain. Decisions closer to the
outcome get more credit; upstream decisions get geometrically less.

Storage: SQLite if persist_path is given, otherwise in-memory dict.
Both backends expose the same interface.

Author: Claude Code (with HoloLoom by Blake)
Date: 2026-03-19
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

try:
    from hololoom.core.protocols.feedback import (
        FeedbackSignal,
        Timescale,
    )
except ImportError:
    from hololoom.protocols.feedback import (
        FeedbackSignal,
        Timescale,
    )

logger = logging.getLogger(__name__)

# Credit decays by this factor per hop in the causation chain
TEMPORAL_DISCOUNT = 0.9


# ============================================================================
# Decision Record
# ============================================================================

@dataclass
class DecisionRecord:
    """
    A recorded decision point — what was chosen, what the alternatives were,
    and enough context to understand why.
    """
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    correlation_id: str = ""
    decision_type: str = ""  # "tool_selection", "model_routing", "rubric_scoring", "math_operation"
    chosen: str = ""
    alternatives: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "correlation_id": self.correlation_id,
            "decision_type": self.decision_type,
            "chosen": self.chosen,
            "alternatives": self.alternatives,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


# ============================================================================
# In-Memory Backend
# ============================================================================

class _InMemoryBackend:
    """Dict-based storage for decisions. No persistence."""

    def __init__(self) -> None:
        # signal_id -> DecisionRecord
        self._decisions: dict[str, DecisionRecord] = {}
        # correlation_id -> list of signal_ids (ordered by time)
        self._by_correlation: dict[str, list[str]] = {}

    def store(self, record: DecisionRecord) -> None:
        self._decisions[record.signal_id] = record
        if record.correlation_id:
            self._by_correlation.setdefault(record.correlation_id, []).append(
                record.signal_id
            )

    def get(self, signal_id: str) -> DecisionRecord | None:
        return self._decisions.get(signal_id)

    def get_chain(self, correlation_id: str) -> list[DecisionRecord]:
        ids = self._by_correlation.get(correlation_id, [])
        records = [self._decisions[sid] for sid in ids if sid in self._decisions]
        records.sort(key=lambda r: r.timestamp)
        return records

    def prune(self, cutoff: datetime) -> int:
        to_remove = [
            sid for sid, rec in self._decisions.items() if rec.timestamp < cutoff
        ]
        for sid in to_remove:
            rec = self._decisions.pop(sid)
            if rec.correlation_id in self._by_correlation:
                chain = self._by_correlation[rec.correlation_id]
                if sid in chain:
                    chain.remove(sid)
                if not chain:
                    del self._by_correlation[rec.correlation_id]
        return len(to_remove)

    def count(self) -> int:
        return len(self._decisions)


# ============================================================================
# SQLite Backend
# ============================================================================

class _SqliteBackend:
    """SQLite-backed storage for decisions."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                signal_id TEXT PRIMARY KEY,
                correlation_id TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                chosen TEXT NOT NULL,
                alternatives TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_corr
            ON decisions(correlation_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_decisions_ts
            ON decisions(timestamp)
        """)
        self._conn.commit()

    def store(self, record: DecisionRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.signal_id,
                record.correlation_id,
                record.decision_type,
                record.chosen,
                json.dumps(record.alternatives),
                record.timestamp.isoformat(),
                json.dumps(record.context),
            ),
        )
        self._conn.commit()

    def get(self, signal_id: str) -> DecisionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_chain(self, correlation_id: str) -> list[DecisionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM decisions WHERE correlation_id = ? ORDER BY timestamp",
            (correlation_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def prune(self, cutoff: datetime) -> int:
        cursor = self._conn.execute(
            "DELETE FROM decisions WHERE timestamp < ?", (cutoff.isoformat(),)
        )
        self._conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return row[0] if row else 0

    @staticmethod
    def _row_to_record(row: tuple) -> DecisionRecord:
        return DecisionRecord(
            signal_id=row[0],
            correlation_id=row[1],
            decision_type=row[2],
            chosen=row[3],
            alternatives=json.loads(row[4]),
            timestamp=datetime.fromisoformat(row[5]),
            context=json.loads(row[6]),
        )

    def close(self) -> None:
        self._conn.close()


# ============================================================================
# CausationTracker
# ============================================================================

class CausationTracker:
    """
    Records decisions and assigns credit when outcomes arrive.

    Decision recording:
        tracker.record_decision("sig_abc", "corr_123", "tool_selection", "answer", ["search", "calc"], {...})

    Credit assignment:
        When an outcome signal arrives with a causation_chain, walk the chain
        and generate credit-assignment FeedbackSignals with temporal discount.

    Pruning:
        tracker.prune(max_age_days=7)  # Remove old decisions
    """

    def __init__(self, persist_path: str | None = None):
        """
        Args:
            persist_path: Path to SQLite database. None = in-memory only.
        """
        if persist_path:
            self._backend = _SqliteBackend(persist_path)
            logger.info("CausationTracker: SQLite backend at %s", persist_path)
        else:
            self._backend = _InMemoryBackend()
            logger.debug("CausationTracker: in-memory backend")

    def record_decision(
        self,
        signal_id: str,
        correlation_id: str,
        decision_type: str,
        chosen: str,
        alternatives: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        """Record a decision point for later credit assignment."""
        record = DecisionRecord(
            signal_id=signal_id,
            correlation_id=correlation_id,
            decision_type=decision_type,
            chosen=chosen,
            alternatives=alternatives or [],
            context=context or {},
        )
        self._backend.store(record)
        logger.debug(
            "Recorded decision: %s chose=%s (type=%s, corr=%s)",
            signal_id, chosen, decision_type, correlation_id,
        )
        return record

    def record(self, signal: FeedbackSignal) -> None:
        """
        Record a FeedbackSignal as a decision if it has a causation chain.

        This is used for signals that themselves represent decisions (e.g.,
        tool selections emitted as FeedbackSignals).
        """
        if not signal.causation_chain:
            return
        corr_id = signal.causation_chain[0] if signal.causation_chain else signal.signal_id
        self.record_decision(
            signal_id=signal.signal_id,
            correlation_id=corr_id,
            decision_type=signal.dimension,
            chosen=signal.subject,
            context=signal.metadata,
        )

    def assign_credit(self, outcome_signal: FeedbackSignal) -> list[FeedbackSignal]:
        """
        Walk causation chain, generate credit-assignment FeedbackSignals.

        For each decision in the chain, a credit signal is generated with:
            value = outcome_value * discount^depth
            dimension = "outcome_credit"
            subject = the decision's chosen option

        Temporal discount: TEMPORAL_DISCOUNT^depth (0.9^depth by default).
        Depth 0 = the most recent decision, depth N = the earliest.

        Args:
            outcome_signal: The outcome that triggers credit assignment.

        Returns:
            List of credit-assignment FeedbackSignals.
        """
        if not outcome_signal.causation_chain:
            return []

        credit_signals: list[FeedbackSignal] = []

        for chain_id in outcome_signal.causation_chain:
            # Try as correlation_id first (gets all decisions in that workflow)
            chain_decisions = self._backend.get_chain(chain_id)

            # If nothing found, try as a direct signal_id
            if not chain_decisions:
                direct = self._backend.get(chain_id)
                if direct:
                    chain_decisions = [direct]

            if not chain_decisions:
                continue

            # Assign credit in reverse order (most recent = depth 0)
            for depth, decision in enumerate(reversed(chain_decisions)):
                discount = TEMPORAL_DISCOUNT ** depth
                credit_value = outcome_signal.value * discount
                credit_value = max(-1.0, min(1.0, credit_value))

                credit_signal = FeedbackSignal(
                    source="causation_tracker",
                    timescale=outcome_signal.timescale,
                    subject=decision.chosen,
                    dimension="outcome_credit",
                    value=credit_value,
                    confidence=outcome_signal.confidence * discount,
                    causation_chain=[chain_id, decision.signal_id],
                    metadata={
                        "decision_type": decision.decision_type,
                        "depth": depth,
                        "discount": discount,
                        "outcome_source": outcome_signal.source,
                        "alternatives": decision.alternatives,
                    },
                )
                credit_signals.append(credit_signal)

        logger.debug(
            "Credit assignment: %d chain IDs -> %d credit signals",
            len(outcome_signal.causation_chain),
            len(credit_signals),
        )
        return credit_signals

    def get_decision(self, signal_id: str) -> DecisionRecord | None:
        """Retrieve a specific decision record."""
        return self._backend.get(signal_id)

    def get_workflow_decisions(self, correlation_id: str) -> list[DecisionRecord]:
        """Retrieve all decisions in a workflow by correlation ID."""
        return self._backend.get_chain(correlation_id)

    def prune(self, max_age_days: int = 7) -> int:
        """
        Remove decisions older than max_age_days.

        Returns the number of pruned records.
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        count = self._backend.prune(cutoff)
        if count > 0:
            logger.info("Pruned %d decisions older than %d days", count, max_age_days)
        return count

    @property
    def decision_count(self) -> int:
        """Number of stored decisions."""
        return self._backend.count()

    def close(self) -> None:
        """Clean up backend resources."""
        if isinstance(self._backend, _SqliteBackend):
            self._backend.close()
