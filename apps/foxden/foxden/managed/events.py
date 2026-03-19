"""Event bus — the nervous system of a managed focus group.

Every session emits events. Management agents subscribe to patterns they
care about. The bus is the only coupling between participants and managers.

Design: async, in-process, zero-dependency. Events are typed dataclasses
with a kind discriminator. Handlers are async callables filtered by kind.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine


class EventKind(str, Enum):
    # Session lifecycle.
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    SESSION_STEP = "session.step"

    # Findings.
    ISSUE_FOUND = "finding.issue"
    CONFUSION_FOUND = "finding.confusion"
    GOAL_COMPLETED = "finding.goal_completed"
    GOAL_FAILED = "finding.goal_failed"

    # Behavioral.
    FRUSTRATION_HIGH = "behavior.frustration_high"
    PATIENCE_EXHAUSTED = "behavior.patience_exhausted"
    ACTION_FAILED = "behavior.action_failed"

    # Management directives (manager → session).
    INJECT_PROMPT = "directive.inject_prompt"
    REDIRECT_GOAL = "directive.redirect_goal"
    TERMINATE_SESSION = "directive.terminate"

    # Orchestration.
    SPAWN_AGENT = "orchestration.spawn"
    SCALE_UP = "orchestration.scale_up"
    GROUP_COMPLETE = "orchestration.complete"
    BUDGET_WARNING = "orchestration.budget_warning"


@dataclass(frozen=True)
class Event:
    """A single event on the bus."""

    kind: EventKind
    session_id: str = ""
    persona: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __repr__(self) -> str:
        return f"Event({self.kind.value}, {self.persona}, {self.session_id[:8]})"


# Handler signature: async (Event) -> None
Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async pub/sub bus with kind-filtered subscriptions.

    Usage:
        bus = EventBus()
        bus.on(EventKind.ISSUE_FOUND, my_handler)
        bus.on_any(my_catch_all)
        await bus.emit(Event(kind=EventKind.ISSUE_FOUND, ...))
    """

    def __init__(self):
        self._handlers: dict[EventKind, list[Handler]] = {}
        self._any_handlers: list[Handler] = []
        self._history: list[Event] = []
        self._max_history: int = 1000

    def on(self, kind: EventKind, handler: Handler) -> None:
        """Subscribe to a specific event kind."""
        self._handlers.setdefault(kind, []).append(handler)

    def on_many(self, kinds: list[EventKind], handler: Handler) -> None:
        """Subscribe to multiple event kinds."""
        for kind in kinds:
            self.on(kind, handler)

    def on_any(self, handler: Handler) -> None:
        """Subscribe to all events."""
        self._any_handlers.append(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all matching handlers."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        handlers = list(self._handlers.get(event.kind, []))
        handlers.extend(self._any_handlers)

        if handlers:
            await asyncio.gather(
                *(h(event) for h in handlers),
                return_exceptions=True,
            )

    def query(
        self,
        kind: EventKind | None = None,
        persona: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[Event]:
        """Query recent event history."""
        results = self._history
        if kind:
            results = [e for e in results if e.kind == kind]
        if persona:
            results = [e for e in results if e.persona == persona]
        if session_id:
            results = [e for e in results if e.session_id == session_id]
        return results[-limit:]

    def count(self, kind: EventKind, **filters) -> int:
        """Count events matching criteria."""
        return len(self.query(kind=kind, **filters))

    @property
    def stats(self) -> dict[str, int]:
        """Event counts by kind."""
        counts: dict[str, int] = {}
        for e in self._history:
            counts[e.kind.value] = counts.get(e.kind.value, 0) + 1
        return counts
