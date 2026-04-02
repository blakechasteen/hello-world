"""
Agent Calendar - Scheduled tasks with completion tracking for any agent.

The agent calendar replaces static cron with living, memory-aware scheduling.
Tasks recur on human-readable patterns. When an agent completes a task, the
outcome is recorded as a CompletionRecord and (optionally) persisted to the
AwarenessGraph so the agent can recall what happened last time.

Usage:

    from hololoom.core.chrono.calendar import AgentCalendar, CalendarTask

    cal = AgentCalendar(agent_id="farm-agent")

    cal.add(CalendarTask(
        id="soil-ph-check",
        name="Check soil pH in raised beds",
        recurrence="every:monday:9am",
        tags=["maintenance", "garden"],
    ))

    # What's due?
    due = cal.due_tasks()

    # Agent does the work, then records it:
    cal.complete("soil-ph-check", outcome="pH 6.4, slightly acidic", notes="Added lime")

    # Later, the agent can ask:
    history = cal.history("soil-ph-check", limit=5)
    streaks = cal.streak("soil-ph-check")

    # Persist completions to awareness graph:
    await cal.sync_to_awareness(loom)

Created: 2026-04-02
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from .recurrence import Recurrence, parse_recurrence

logger = logging.getLogger(__name__)


# ============================================================================
# Task priority (domain-agnostic)
# ============================================================================

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Completion outcomes
# ============================================================================

class Outcome(Enum):
    """What happened when the agent attempted the task."""
    COMPLETED = "completed"       # Done successfully
    PARTIAL = "partial"           # Partially done
    SKIPPED = "skipped"           # Intentionally skipped
    FAILED = "failed"             # Attempted but failed
    DEFERRED = "deferred"         # Pushed to next occurrence


# ============================================================================
# Core data structures
# ============================================================================

@dataclass
class CalendarTask:
    """A recurring (or one-time) task on the agent calendar."""

    id: str
    name: str
    recurrence: str  # human-readable, parsed on access
    tags: list[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    description: str = ""
    reminder_before: Optional[timedelta] = timedelta(days=1)
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed on load
    _parsed: Optional[Recurrence] = field(default=None, repr=False, compare=False)

    @property
    def schedule(self) -> Recurrence:
        if self._parsed is None:
            self._parsed = parse_recurrence(self.recurrence)
        return self._parsed

    def next_occurrence(self, after: Optional[datetime] = None) -> datetime:
        return self.schedule.next_occurrence(after)

    def is_due(self, at: Optional[datetime] = None, tolerance_minutes: int = 30) -> bool:
        return self.schedule.is_due(at, tolerance_minutes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "recurrence": self.recurrence,
            "tags": self.tags,
            "priority": self.priority.value,
            "description": self.description,
            "reminder_before_seconds": (
                self.reminder_before.total_seconds() if self.reminder_before else None
            ),
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalendarTask:
        reminder = None
        if data.get("reminder_before_seconds") is not None:
            reminder = timedelta(seconds=data["reminder_before_seconds"])
        return cls(
            id=data["id"],
            name=data["name"],
            recurrence=data["recurrence"],
            tags=data.get("tags", []),
            priority=Priority(data.get("priority", "medium")),
            description=data.get("description", ""),
            reminder_before=reminder,
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CompletionRecord:
    """What happened when a task was (or wasn't) completed."""

    record_id: str
    task_id: str
    completed_at: datetime
    outcome: Outcome
    agent_id: str = ""
    notes: str = ""
    duration_seconds: Optional[float] = None
    data: dict[str, Any] = field(default_factory=dict)
    synced_memory_id: Optional[str] = None  # set after awareness sync

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_id": self.task_id,
            "completed_at": self.completed_at.isoformat(),
            "outcome": self.outcome.value,
            "agent_id": self.agent_id,
            "notes": self.notes,
            "duration_seconds": self.duration_seconds,
            "data": self.data,
            "synced_memory_id": self.synced_memory_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionRecord:
        return cls(
            record_id=data["record_id"],
            task_id=data["task_id"],
            completed_at=datetime.fromisoformat(data["completed_at"]),
            outcome=Outcome(data["outcome"]),
            agent_id=data.get("agent_id", ""),
            notes=data.get("notes", ""),
            duration_seconds=data.get("duration_seconds"),
            data=data.get("data", {}),
            synced_memory_id=data.get("synced_memory_id"),
        )


# ============================================================================
# Agent Calendar
# ============================================================================

class AgentCalendar:
    """
    A calendar of scheduled tasks for any agent, with completion history.

    The calendar is domain-agnostic. Use tags and metadata to add domain
    meaning (farm events, maintenance windows, report deadlines, etc.).
    Presets provide ready-made task sets for specific domains.

    Completions accumulate over time, giving the agent a longitudinal
    record it can query: "what happened last spring planting?" or
    "how often do we miss the Monday pH check?"
    """

    def __init__(
        self,
        agent_id: str = "default",
        persistence_path: Optional[Path] = None,
    ):
        self.agent_id = agent_id
        self.persistence_path = persistence_path

        self._tasks: dict[str, CalendarTask] = {}
        self._completions: list[CompletionRecord] = []

        if persistence_path and persistence_path.exists():
            self._load(persistence_path)

    # ── Task management ──────────────────────────────────────────────

    def add(self, task: CalendarTask) -> None:
        """Add a task to the calendar."""
        self._tasks[task.id] = task
        logger.debug("Calendar task added: %s (%s)", task.id, task.recurrence)

    def remove(self, task_id: str) -> Optional[CalendarTask]:
        """Remove a task. Returns the removed task or None."""
        return self._tasks.pop(task_id, None)

    def get(self, task_id: str) -> Optional[CalendarTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    @property
    def tasks(self) -> list[CalendarTask]:
        """All tasks, sorted by next occurrence."""
        now = datetime.now()
        return sorted(
            self._tasks.values(),
            key=lambda t: t.next_occurrence(now) if t.enabled else datetime.max,
        )

    def tasks_by_tag(self, tag: str) -> list[CalendarTask]:
        """Get all tasks with a given tag."""
        return [t for t in self._tasks.values() if tag in t.tags]

    # ── Due / upcoming queries ───────────────────────────────────────

    def due_tasks(
        self,
        at: Optional[datetime] = None,
        tolerance_minutes: int = 30,
    ) -> list[CalendarTask]:
        """Tasks that are due now (within tolerance)."""
        return [
            t for t in self._tasks.values()
            if t.enabled and t.is_due(at, tolerance_minutes)
        ]

    def upcoming(
        self,
        within: timedelta = timedelta(days=7),
        at: Optional[datetime] = None,
    ) -> list[CalendarTask]:
        """Tasks coming up within a time window."""
        now = at or datetime.now()
        cutoff = now + within
        result = []
        for task in self._tasks.values():
            if not task.enabled:
                continue
            nxt = task.next_occurrence(now)
            if nxt <= cutoff:
                result.append(task)
        return sorted(result, key=lambda t: t.next_occurrence(now))

    def overdue(self, at: Optional[datetime] = None) -> list[CalendarTask]:
        """Tasks that were due but have no recent completion."""
        now = at or datetime.now()
        result = []
        for task in self._tasks.values():
            if not task.enabled:
                continue
            last = self.last_completion(task.id)
            if last is None:
                # Never completed - if next occurrence is past, it's overdue
                # (for recurring tasks, check if we missed the window)
                result.append(task)
                continue
            # Check if there's been a due date since last completion
            nxt_after_last = task.next_occurrence(last.completed_at)
            if nxt_after_last < now:
                result.append(task)
        return result

    def needs_reminder(self, at: Optional[datetime] = None) -> list[CalendarTask]:
        """Tasks within their reminder window."""
        now = at or datetime.now()
        result = []
        for task in self._tasks.values():
            if not task.enabled or task.reminder_before is None:
                continue
            nxt = task.next_occurrence(now)
            if timedelta() < (nxt - now) <= task.reminder_before:
                result.append(task)
        return result

    # ── Completion tracking ──────────────────────────────────────────

    def complete(
        self,
        task_id: str,
        outcome: Outcome | str = Outcome.COMPLETED,
        notes: str = "",
        duration_seconds: Optional[float] = None,
        data: Optional[dict[str, Any]] = None,
        at: Optional[datetime] = None,
    ) -> CompletionRecord:
        """Record a task completion."""
        if isinstance(outcome, str):
            outcome = Outcome(outcome)

        record = CompletionRecord(
            record_id=str(uuid.uuid4()),
            task_id=task_id,
            completed_at=at or datetime.now(),
            outcome=outcome,
            agent_id=self.agent_id,
            notes=notes,
            duration_seconds=duration_seconds,
            data=data or {},
        )
        self._completions.append(record)

        task = self._tasks.get(task_id)
        task_name = task.name if task else task_id
        logger.info(
            "Task completed: %s [%s] by %s",
            task_name, outcome.value, self.agent_id,
        )
        return record

    def history(
        self,
        task_id: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> list[CompletionRecord]:
        """Get completion history for a task, newest first."""
        records = [
            r for r in self._completions
            if r.task_id == task_id
            and (since is None or r.completed_at >= since)
        ]
        records.sort(key=lambda r: r.completed_at, reverse=True)
        if limit:
            records = records[:limit]
        return records

    def last_completion(self, task_id: str) -> Optional[CompletionRecord]:
        """Most recent completion for a task."""
        h = self.history(task_id, limit=1)
        return h[0] if h else None

    def all_completions(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[CompletionRecord]:
        """All completions across all tasks, newest first."""
        records = self._completions
        if since:
            records = [r for r in records if r.completed_at >= since]
        records = sorted(records, key=lambda r: r.completed_at, reverse=True)
        if limit:
            records = records[:limit]
        return records

    # ── Streak & reliability ─────────────────────────────────────────

    def streak(self, task_id: str) -> dict[str, Any]:
        """
        Calculate completion streak for a task.

        Returns:
            current_streak: consecutive successful completions
            longest_streak: all-time longest streak
            completion_rate: successful / total attempts
            total_completions: total records
        """
        records = self.history(task_id)  # newest first
        if not records:
            return {
                "current_streak": 0,
                "longest_streak": 0,
                "completion_rate": 0.0,
                "total_completions": 0,
            }

        successful = [r for r in records if r.outcome == Outcome.COMPLETED]

        # Current streak (consecutive from most recent)
        current = 0
        for r in records:
            if r.outcome == Outcome.COMPLETED:
                current += 1
            else:
                break

        # Longest streak
        longest = 0
        run = 0
        for r in reversed(records):  # oldest first for longest
            if r.outcome == Outcome.COMPLETED:
                run += 1
                longest = max(longest, run)
            else:
                run = 0

        return {
            "current_streak": current,
            "longest_streak": longest,
            "completion_rate": len(successful) / len(records),
            "total_completions": len(records),
        }

    # ── Awareness graph sync ─────────────────────────────────────────

    async def sync_to_awareness(self, loom: Any, unsynced_only: bool = True) -> int:
        """
        Persist completion records to the AwarenessGraph via loom.experience().

        Each completion becomes a memory node the agent can recall later.
        Returns the number of records synced.

        Args:
            loom: A HoloLoom instance (or anything with an experience() method).
            unsynced_only: If True, only sync records not yet persisted.
        """
        count = 0
        for record in self._completions:
            if unsynced_only and record.synced_memory_id is not None:
                continue

            task = self._tasks.get(record.task_id)
            task_name = task.name if task else record.task_id

            content = (
                f"Calendar task {record.outcome.value}: {task_name}. "
                f"{record.notes}" if record.notes else
                f"Calendar task {record.outcome.value}: {task_name}."
            )

            context = {
                "source": "agent_calendar",
                "task_id": record.task_id,
                "record_id": record.record_id,
                "outcome": record.outcome.value,
                "agent_id": record.agent_id,
                "tags": task.tags if task else [],
                "completed_at": record.completed_at.isoformat(),
            }
            if record.duration_seconds is not None:
                context["duration_seconds"] = record.duration_seconds
            if record.data:
                context["task_data"] = record.data

            memory = await loom.experience(content, context=context)
            record.synced_memory_id = memory.id
            count += 1

        if count:
            logger.info("Synced %d completion records to awareness graph", count)
        return count

    # ── Persistence (JSON) ───────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Save calendar state to JSON."""
        target = path or self.persistence_path
        if target is None:
            raise ValueError("No persistence path configured")

        data = {
            "agent_id": self.agent_id,
            "saved_at": datetime.now().isoformat(),
            "tasks": [t.to_dict() for t in self._tasks.values()],
            "completions": [c.to_dict() for c in self._completions],
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug("Calendar saved to %s", target)

    def _load(self, path: Path) -> None:
        """Load calendar state from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.agent_id = data.get("agent_id", self.agent_id)

        for task_data in data.get("tasks", []):
            task = CalendarTask.from_dict(task_data)
            self._tasks[task.id] = task

        for comp_data in data.get("completions", []):
            self._completions.append(CompletionRecord.from_dict(comp_data))

        logger.debug(
            "Calendar loaded: %d tasks, %d completions",
            len(self._tasks), len(self._completions),
        )

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self, at: Optional[datetime] = None) -> dict[str, Any]:
        """Calendar summary for agent context injection."""
        now = at or datetime.now()
        return {
            "agent_id": self.agent_id,
            "timestamp": now.isoformat(),
            "total_tasks": len(self._tasks),
            "due_now": [t.to_dict() for t in self.due_tasks(now)],
            "upcoming_7d": [t.to_dict() for t in self.upcoming(timedelta(days=7), now)],
            "overdue": [t.to_dict() for t in self.overdue(now)],
            "reminders": [t.to_dict() for t in self.needs_reminder(now)],
            "recent_completions": [
                c.to_dict() for c in self.all_completions(limit=5)
            ],
        }

    def format_text(self, at: Optional[datetime] = None) -> str:
        """Human-readable calendar summary."""
        now = at or datetime.now()
        lines = [
            f"=== Agent Calendar ({self.agent_id}) ===",
            f"    {now.strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        due = self.due_tasks(now)
        if due:
            lines.append("DUE NOW:")
            for t in due:
                last = self.last_completion(t.id)
                last_str = (
                    f" (last: {last.completed_at.strftime('%Y-%m-%d')})"
                    if last else " (never completed)"
                )
                lines.append(f"  [{t.priority.value.upper():8s}] {t.name}{last_str}")
            lines.append("")

        overdue = self.overdue(now)
        if overdue:
            lines.append("OVERDUE:")
            for t in overdue:
                lines.append(f"  [{t.priority.value.upper():8s}] {t.name}")
            lines.append("")

        upcoming = self.upcoming(timedelta(days=7), now)
        if upcoming:
            lines.append("NEXT 7 DAYS:")
            for t in upcoming:
                nxt = t.next_occurrence(now)
                lines.append(
                    f"  {nxt.strftime('%a %b %d %H:%M')} - {t.name}"
                )
            lines.append("")

        recent = self.all_completions(limit=5)
        if recent:
            lines.append("RECENT COMPLETIONS:")
            for c in recent:
                task = self._tasks.get(c.task_id)
                name = task.name if task else c.task_id
                lines.append(
                    f"  {c.completed_at.strftime('%Y-%m-%d')} "
                    f"[{c.outcome.value}] {name}"
                )
                if c.notes:
                    lines.append(f"    {c.notes}")
            lines.append("")

        return "\n".join(lines)
