"""Interface protocol — how agents interact with software under test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Observation:
    """What the agent sees after taking an action."""

    content: str  # text representation of current state
    raw: dict | None = None  # interface-specific raw data
    screenshot: bytes | None = None  # optional screenshot


@dataclass
class Action:
    """An action the agent wants to take."""

    kind: str  # "command", "type", "click", "request", etc.
    value: str  # the actual command/input/url
    metadata: dict | None = None


@runtime_checkable
class TestInterface(Protocol):
    """Protocol for interacting with software under test."""

    async def start(self) -> Observation:
        """Initialize the interface and return the first observation."""
        ...

    async def act(self, action: Action) -> Observation:
        """Perform an action and return what happened."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
