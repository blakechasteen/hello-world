"""
SpecLedger - Audit Trail for Jenny UI Specifications
=====================================================
Complete provenance tracking and replay capability for generated UI panels.

Philosophy:
> "Disposable pixels, durable decisions."
>
> Every UI panel dissolves, but the decision to show it is permanent.
> SpecLedger captures the complete history of UI generation.

Features:
- JSONL persistence (append-only, efficient)
- Query by spacetime_id, time range, panel_type
- Full replay capability (recreate any past session)
- Statistics for understanding UI generation patterns
- Session-based organization

This module implements SpecLedgerProtocol and extends the patterns
from HoloLoom's alignment/audit_trail.py.

References:
- jenny_spec.py (JennySpec dataclass)
- protocols/jenny.py (SpecLedgerProtocol)
- alignment/audit_trail.py (AuditTrail pattern)

Author: HoloLoom Team
Date: 2025-12-01 (Jenny MVP Week 1)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import json

from .jenny_spec import (
    JennySpec,
    LifecycleStage,
    DissolutionTrigger,
    PanelTypeJenny,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Ledger Entry Dataclass
# ============================================================================

@dataclass
class SpecLedgerEntry:
    """
    Single entry in the SpecLedger.

    Wraps a JennySpec with additional audit metadata.
    """
    entry_id: str
    spec: JennySpec
    spacetime_id: str
    session_id: str
    logged_at: datetime = field(default_factory=datetime.now)

    # Lifecycle tracking
    lifecycle_history: List[Dict[str, Any]] = field(default_factory=list)

    # Additional audit metadata
    compiler_context: Dict[str, Any] = field(default_factory=dict)
    render_count: int = 0
    pin_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "entry_id": self.entry_id,
            "spec": self.spec.to_dict(),
            "spacetime_id": self.spacetime_id,
            "session_id": self.session_id,
            "logged_at": self.logged_at.isoformat(),
            "lifecycle_history": self.lifecycle_history,
            "compiler_context": self.compiler_context,
            "render_count": self.render_count,
            "pin_count": self.pin_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecLedgerEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            spec=JennySpec.from_dict(data["spec"]),
            spacetime_id=data["spacetime_id"],
            session_id=data["session_id"],
            logged_at=datetime.fromisoformat(data["logged_at"]),
            lifecycle_history=data.get("lifecycle_history", []),
            compiler_context=data.get("compiler_context", {}),
            render_count=data.get("render_count", 0),
            pin_count=data.get("pin_count", 0),
        )


# ============================================================================
# SpecLedger - Main Audit Trail
# ============================================================================

class SpecLedger:
    """
    Audit trail for Jenny UI specifications.

    Provides complete provenance tracking and replay capability.
    All generated specs are logged for debugging, analysis, and replay.

    Features:
    - JSONL persistence (append-only, streaming-friendly)
    - Multiple query methods (by spacetime, time, panel type)
    - Session replay for debugging
    - Statistics for pattern analysis
    - Index structures for fast queries

    Philosophy:
    - Every spec is logged (no exceptions)
    - Logs are append-only (immutable history)
    - Queries are fast via indexes
    - Memory-efficient streaming for large histories

    Usage:
        ledger = SpecLedger(persist_path="./spec_ledger")

        # Log a spec
        entry_id = await ledger.log_spec(
            spec=my_spec,
            spacetime_id="st_123",
            session_id="session_456"
        )

        # Query
        specs = await ledger.query_by_spacetime("st_123")
        recent = await ledger.query_by_time_range(start, end)

        # Replay session
        session_specs = await ledger.replay_session("session_456")

        # Statistics
        stats = await ledger.get_statistics()
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        auto_flush: bool = True,
        max_memory_entries: int = 10000,
    ):
        """
        Initialize SpecLedger.

        Args:
            persist_path: Directory for persistent storage (None = memory only)
            auto_flush: Automatically flush to disk after each log
            max_memory_entries: Maximum entries to keep in memory (LRU eviction)
        """
        # In-memory storage
        self.entries: Dict[str, SpecLedgerEntry] = {}

        # Indexes for fast queries
        self._by_spacetime: Dict[str, List[str]] = defaultdict(list)
        self._by_session: Dict[str, List[str]] = defaultdict(list)
        self._by_panel_type: Dict[str, List[str]] = defaultdict(list)
        self._by_spec_id: Dict[str, str] = {}  # spec_id → entry_id

        # Configuration
        self.persist_path = Path(persist_path) if persist_path else None
        self.auto_flush = auto_flush
        self.max_memory_entries = max_memory_entries

        # Statistics counters (in-memory, fast)
        self._total_logged = 0
        self._panel_type_counts: Dict[str, int] = defaultdict(int)
        self._dissolution_trigger_counts: Dict[str, int] = defaultdict(int)

        # Initialize persistence
        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.info(
            f"SpecLedger initialized (persist_path={persist_path}, "
            f"auto_flush={auto_flush}, max_memory={max_memory_entries})"
        )

    # ========================================================================
    # Core Logging Methods
    # ========================================================================

    async def log_spec(
        self,
        spec: JennySpec,
        spacetime_id: str,
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a spec to the ledger.

        Args:
            spec: JennySpec to log
            spacetime_id: ID of Spacetime that generated this spec
            session_id: Session identifier for grouping
            metadata: Optional additional metadata

        Returns:
            Entry ID for this log entry
        """
        entry_id = f"entry_{spec.spec_id}_{datetime.now().timestamp()}"

        entry = SpecLedgerEntry(
            entry_id=entry_id,
            spec=spec,
            spacetime_id=spacetime_id,
            session_id=session_id,
            logged_at=datetime.now(),
            lifecycle_history=[{
                "stage": spec.lifecycle.value,
                "timestamp": datetime.now().isoformat(),
                "trigger": None
            }],
            compiler_context=metadata or {},
        )

        # Store in memory
        self.entries[entry_id] = entry

        # Update indexes
        self._by_spacetime[spacetime_id].append(entry_id)
        self._by_session[session_id].append(entry_id)
        self._by_panel_type[spec.panel_type.value].append(entry_id)
        self._by_spec_id[spec.spec_id] = entry_id

        # Update statistics
        self._total_logged += 1
        self._panel_type_counts[spec.panel_type.value] += 1

        # Memory management (LRU eviction if needed)
        if len(self.entries) > self.max_memory_entries:
            self._evict_oldest()

        # Persist
        if self.auto_flush and self.persist_path:
            self._flush_entry(entry)

        logger.debug(f"Logged spec: {spec.spec_id} (entry: {entry_id})")
        return entry_id

    async def log_lifecycle_transition(
        self,
        spec_id: str,
        new_stage: LifecycleStage,
        trigger: Optional[DissolutionTrigger] = None
    ) -> None:
        """
        Log a lifecycle transition for a spec.

        Args:
            spec_id: Spec ID that transitioned
            new_stage: New lifecycle stage
            trigger: What triggered the transition (for DISSOLVING)
        """
        if spec_id not in self._by_spec_id:
            logger.warning(f"Spec {spec_id} not found in ledger")
            return

        entry_id = self._by_spec_id[spec_id]
        entry = self.entries.get(entry_id)

        if entry:
            entry.lifecycle_history.append({
                "stage": new_stage.value,
                "timestamp": datetime.now().isoformat(),
                "trigger": trigger.value if trigger else None
            })

            # Update dissolution stats
            if trigger:
                self._dissolution_trigger_counts[trigger.value] += 1

            # Update pin count
            if new_stage == LifecycleStage.STABLE:
                entry.pin_count += 1

            logger.debug(f"Logged transition for {spec_id}: → {new_stage.value}")

    async def log_render(self, spec_id: str) -> None:
        """
        Log that a spec was rendered.

        Args:
            spec_id: Spec ID that was rendered
        """
        if spec_id not in self._by_spec_id:
            return

        entry_id = self._by_spec_id[spec_id]
        entry = self.entries.get(entry_id)

        if entry:
            entry.render_count += 1

    # ========================================================================
    # Query Methods
    # ========================================================================

    async def get_spec(self, spec_id: str) -> Optional[JennySpec]:
        """
        Retrieve a spec by ID.

        Args:
            spec_id: Spec ID to retrieve

        Returns:
            Spec if found, None otherwise
        """
        if spec_id not in self._by_spec_id:
            return None

        entry_id = self._by_spec_id[spec_id]
        entry = self.entries.get(entry_id)
        return entry.spec if entry else None

    async def get_entry(self, entry_id: str) -> Optional[SpecLedgerEntry]:
        """
        Retrieve a full ledger entry by entry ID.

        Args:
            entry_id: Ledger entry ID

        Returns:
            Entry if found, None otherwise
        """
        return self.entries.get(entry_id)

    async def query_by_spacetime(self, spacetime_id: str) -> List[JennySpec]:
        """
        Query all specs generated from a Spacetime.

        Args:
            spacetime_id: Spacetime ID to query

        Returns:
            List of specs generated from that Spacetime
        """
        entry_ids = self._by_spacetime.get(spacetime_id, [])
        return [
            self.entries[eid].spec
            for eid in entry_ids
            if eid in self.entries
        ]

    async def query_by_time_range(
        self,
        start_time: str,
        end_time: str,
        limit: int = 100
    ) -> List[JennySpec]:
        """
        Query specs within a time range.

        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            limit: Maximum results

        Returns:
            List of specs within range
        """
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)

        results = []
        for entry in self.entries.values():
            if start_dt <= entry.logged_at <= end_dt:
                results.append(entry.spec)
                if len(results) >= limit:
                    break

        return results

    async def query_by_panel_type(
        self,
        panel_type: str,
        limit: int = 100
    ) -> List[JennySpec]:
        """
        Query specs by panel type.

        Args:
            panel_type: Panel type to query (e.g., "graph", "text")
            limit: Maximum results

        Returns:
            List of specs of that type
        """
        entry_ids = self._by_panel_type.get(panel_type, [])[:limit]
        return [
            self.entries[eid].spec
            for eid in entry_ids
            if eid in self.entries
        ]

    async def query_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[JennySpec]:
        """
        Query specs from a session.

        Args:
            session_id: Session ID to query
            limit: Maximum results

        Returns:
            List of specs from that session
        """
        entry_ids = self._by_session.get(session_id, [])[:limit]
        return [
            self.entries[eid].spec
            for eid in entry_ids
            if eid in self.entries
        ]

    async def replay_session(self, session_id: str) -> List[JennySpec]:
        """
        Replay all specs from a session in order.

        Enables full UI reconstruction for debugging.

        Args:
            session_id: Session ID to replay

        Returns:
            Ordered list of specs from session
        """
        entry_ids = self._by_session.get(session_id, [])

        # Get entries and sort by logged_at
        entries = [
            self.entries[eid]
            for eid in entry_ids
            if eid in self.entries
        ]
        entries.sort(key=lambda e: e.logged_at)

        return [e.spec for e in entries]

    # ========================================================================
    # Statistics Methods
    # ========================================================================

    async def get_statistics(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get ledger statistics.

        Args:
            start_time: Optional start filter
            end_time: Optional end filter

        Returns:
            Statistics dict
        """
        # Filter entries if time range specified
        if start_time or end_time:
            start_dt = datetime.fromisoformat(start_time) if start_time else datetime.min
            end_dt = datetime.fromisoformat(end_time) if end_time else datetime.max

            entries = [
                e for e in self.entries.values()
                if start_dt <= e.logged_at <= end_dt
            ]
        else:
            entries = list(self.entries.values())

        if not entries:
            return {
                "total_specs": 0,
                "panel_type_counts": {},
                "avg_panels_per_spacetime": 0.0,
                "dissolution_trigger_counts": {},
                "unique_sessions": 0,
                "unique_spacetimes": 0,
            }

        # Calculate statistics
        panel_type_counts = defaultdict(int)
        spacetime_panel_counts = defaultdict(int)
        dissolution_counts = defaultdict(int)
        sessions = set()
        spacetimes = set()
        total_pins = 0
        total_renders = 0

        for entry in entries:
            panel_type_counts[entry.spec.panel_type.value] += 1
            spacetime_panel_counts[entry.spacetime_id] += 1
            sessions.add(entry.session_id)
            spacetimes.add(entry.spacetime_id)
            total_pins += entry.pin_count
            total_renders += entry.render_count

            # Count dissolution triggers from lifecycle history
            for hist in entry.lifecycle_history:
                if hist.get("trigger"):
                    dissolution_counts[hist["trigger"]] += 1

        avg_per_spacetime = len(entries) / len(spacetimes) if spacetimes else 0.0

        return {
            "total_specs": len(entries),
            "panel_type_counts": dict(panel_type_counts),
            "avg_panels_per_spacetime": round(avg_per_spacetime, 2),
            "dissolution_trigger_counts": dict(dissolution_counts),
            "unique_sessions": len(sessions),
            "unique_spacetimes": len(spacetimes),
            "total_pins": total_pins,
            "total_renders": total_renders,
            "avg_pins_per_spec": round(total_pins / len(entries), 2) if entries else 0.0,
            "avg_renders_per_spec": round(total_renders / len(entries), 2) if entries else 0.0,
        }

    def get_quick_stats(self) -> Dict[str, Any]:
        """
        Get quick statistics (from in-memory counters).

        Faster than get_statistics() for dashboards.
        """
        return {
            "total_logged": self._total_logged,
            "in_memory_entries": len(self.entries),
            "panel_type_counts": dict(self._panel_type_counts),
            "dissolution_trigger_counts": dict(self._dissolution_trigger_counts),
            "unique_sessions": len(self._by_session),
            "unique_spacetimes": len(self._by_spacetime),
        }

    # ========================================================================
    # Persistence Methods
    # ========================================================================

    def _flush_entry(self, entry: SpecLedgerEntry) -> None:
        """Flush a single entry to disk (JSONL format)."""
        if not self.persist_path:
            return

        log_file = self.persist_path / "specs.jsonl"

        with log_file.open("a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def flush_all(self) -> None:
        """Flush all in-memory entries to disk."""
        if not self.persist_path:
            return

        for entry in self.entries.values():
            self._flush_entry(entry)

        logger.info(f"Flushed {len(self.entries)} entries to disk")

    def _load_from_disk(self) -> None:
        """Load entries from disk on initialization."""
        if not self.persist_path:
            return

        log_file = self.persist_path / "specs.jsonl"
        if not log_file.exists():
            return

        loaded = 0
        with log_file.open() as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    entry = SpecLedgerEntry.from_dict(data)

                    # Store in memory (skip if at capacity)
                    if len(self.entries) < self.max_memory_entries:
                        self.entries[entry.entry_id] = entry

                        # Update indexes
                        self._by_spacetime[entry.spacetime_id].append(entry.entry_id)
                        self._by_session[entry.session_id].append(entry.entry_id)
                        self._by_panel_type[entry.spec.panel_type.value].append(entry.entry_id)
                        self._by_spec_id[entry.spec.spec_id] = entry.entry_id

                    # Update statistics
                    self._total_logged += 1
                    self._panel_type_counts[entry.spec.panel_type.value] += 1

                    loaded += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse ledger line: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load entry: {e}")

        logger.info(f"Loaded {loaded} entries from disk")

    def _evict_oldest(self) -> None:
        """Evict oldest entries when at capacity."""
        if not self.entries:
            return

        # Find oldest entry
        oldest = min(self.entries.values(), key=lambda e: e.logged_at)

        # Remove from indexes
        if oldest.spacetime_id in self._by_spacetime:
            self._by_spacetime[oldest.spacetime_id].remove(oldest.entry_id)
        if oldest.session_id in self._by_session:
            self._by_session[oldest.session_id].remove(oldest.entry_id)
        if oldest.spec.panel_type.value in self._by_panel_type:
            self._by_panel_type[oldest.spec.panel_type.value].remove(oldest.entry_id)
        if oldest.spec.spec_id in self._by_spec_id:
            del self._by_spec_id[oldest.spec.spec_id]

        # Remove entry
        del self.entries[oldest.entry_id]

        logger.debug(f"Evicted oldest entry: {oldest.entry_id}")


# ============================================================================
# Factory Function
# ============================================================================

def create_spec_ledger(
    persist_path: Optional[Path] = None,
    auto_flush: bool = True,
    max_memory_entries: int = 10000,
) -> SpecLedger:
    """
    Create a SpecLedger instance.

    Args:
        persist_path: Directory for persistent storage (None = memory only)
        auto_flush: Automatically flush to disk after each log
        max_memory_entries: Maximum entries to keep in memory

    Returns:
        Configured SpecLedger instance
    """
    return SpecLedger(
        persist_path=persist_path,
        auto_flush=auto_flush,
        max_memory_entries=max_memory_entries,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SpecLedgerEntry',
    'SpecLedger',
    'create_spec_ledger',
]
