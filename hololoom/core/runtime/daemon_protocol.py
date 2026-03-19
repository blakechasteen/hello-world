"""
Daemon Protocol — wire format for the HoloLoom daemon.
=======================================================

Immutable message types. This file defines the contract between
clients and the daemon. The daemon.py (mutable) implements it.

Three primitives (adapted from Codex App Server):
  - Thread: correlation chain (maps to correlation_id)
  - Turn: request-response within a thread
  - Item: signal or outcome within a turn

Wire format: JSON over WebSocket on ws://127.0.0.1:4242.
Localhost only, no auth for v1.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DaemonRequest:
    """Client → Daemon message."""
    type: str                      # "query" | "fire_ritual" | "status" |
                                   # "replay" | "skill_query" | "subscribe"
    thread_id: str | None = None   # existing thread, or None to create new
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "DaemonRequest":
        d = json.loads(data)
        return cls(
            type=d["type"],
            thread_id=d.get("thread_id"),
            payload=d.get("payload", {}),
            request_id=d.get("request_id", uuid.uuid4().hex[:12]),
        )

    @classmethod
    def validate(cls, data: str) -> tuple[bool, str]:
        """Validate a JSON string as a DaemonRequest. Returns (valid, error)."""
        try:
            d = json.loads(data)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        if not isinstance(d, dict):
            return False, "Expected JSON object"
        if "type" not in d:
            return False, "Missing 'type' field"
        valid_types = {"query", "fire_ritual", "status", "replay", "skill_query", "subscribe"}
        if d["type"] not in valid_types:
            return False, f"Unknown type '{d['type']}'. Valid: {valid_types}"
        return True, ""


@dataclass
class DaemonResponse:
    """Daemon → Client response to a request."""
    type: str = "result"           # "result" | "error"
    thread_id: str = ""
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    request_id: str = ""           # echoes the request
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "DaemonResponse":
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def error(cls, message: str, request_id: str = "") -> "DaemonResponse":
        return cls(type="error", payload={"error": message}, request_id=request_id)


@dataclass
class DaemonEvent:
    """Daemon → Client streaming event (bus signal forwarding)."""
    type: str = "event"
    thread_id: str = ""
    signal: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "DaemonEvent":
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
