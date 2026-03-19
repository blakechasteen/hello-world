"""Base protocol for management agents.

Every management agent follows the same lifecycle:
  1. attach(bus) — subscribe to events it cares about
  2. react to events via handlers
  3. emit directives back onto the bus
  4. detach() — clean up

Management agents are peers, not a hierarchy. They compose freely.
The facilitator is just the one that starts and stops things.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..llm.backend import LLMBackend
from .events import Event, EventBus, EventKind


@dataclass
class ManagerState:
    """Shared state container for management agents."""

    active_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_issues: int = 0
    total_confusion: int = 0
    total_goals_completed: int = 0
    total_goals_failed: int = 0
    token_budget_used: int = 0
    token_budget_max: int = 0  # 0 = unlimited


class Manager(ABC):
    """Base class for focus group management agents."""

    def __init__(self, bus: EventBus, backend: LLMBackend | None = None):
        self.bus = bus
        self.backend = backend
        self._attached = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @abstractmethod
    def attach(self) -> None:
        """Subscribe to events. Called once at startup."""
        ...

    def detach(self) -> None:
        """Clean up. Default is no-op."""
        pass

    async def _emit(self, kind: EventKind, **kwargs) -> None:
        """Convenience: emit an event from this manager."""
        await self.bus.emit(Event(kind=kind, **kwargs))
