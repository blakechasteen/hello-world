"""
Fire-and-forget observability — bus + memory bus emission.

Never blocks. Never throws. Gracefully degrades if bus/memory_bus unavailable.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def emit(
    agent_name: str,
    query: str,
    response: str,
    room_id: str,
    tokens: int,
    duration_ms: float,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """
    Write observation to memory bus + emit signal to UnifiedBus.

    Both operations are fire-and-forget. Neither blocks the caller.
    Neither throws — all exceptions are swallowed with a debug log.
    """
    _write_to_memory_bus(agent_name, query, response, room_id)
    _emit_to_bus(agent_name, query, response, room_id, tokens, duration_ms, extra_payload)
    _propose_to_vault(agent_name, query, response)


def _write_to_memory_bus(
    agent_name: str, query: str, response: str, room_id: str,
) -> None:
    """Write observation to memory bus (cross-agent searchable). Never throws."""
    try:
        from hololoom.apps.server.memory_bus import get_store
        store = get_store()
        store.write(
            agent=agent_name,
            content=f"Q: {query[:200]}\nA: {response[:500]}",
            tags=["chat", agent_name],
            room_id=room_id,
        )
    except Exception:
        pass  # Memory bus unavailable — agent works fine without it


def _emit_to_bus(
    agent_name: str,
    query: str,
    response: str,
    room_id: str,
    tokens: int,
    duration_ms: float,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget signal emission for observability. Never throws."""
    try:
        from hololoom.apps.server.bus_setup import get_bus

        bus = get_bus()
        if bus is None:
            return

        from hololoom.core.bus import Signal, SignalKind, SignalPriority

        payload = {
            "text": query[:200],
            "room_id": room_id,
            "tokens": tokens,
            "duration_ms": round(duration_ms, 1),
            "response_length": len(response),
            "strategy": "direct_pass",
            "confidence": 1.0,
            "success": True,
        }
        if extra_payload:
            payload.update(extra_payload)

        signal = Signal(
            kind=SignalKind.EXECUTION_COMPLETE,
            priority=SignalPriority.LOW,
            source=agent_name,
            payload=payload,
        )
        asyncio.ensure_future(bus.emit(signal))
    except Exception:
        pass  # Bus failure never crashes an agent


# --------------------------------------------------------------------------- #
# Vault federation — propose substantive responses to PARA inbox
# --------------------------------------------------------------------------- #

# Minimum response length to consider proposing (short replies aren't notes)
_VAULT_MIN_LENGTH = 200

# Skip proposals for generic/trivial queries
_TRIVIAL_PREFIXES = ("hi", "hello", "hey", "thanks", "ok", "sure", "yes", "no")


def _propose_to_vault(agent_name: str, query: str, response: str) -> None:
    """Fire-and-forget vault proposal. Never throws, never blocks."""
    try:
        if len(response) < _VAULT_MIN_LENGTH:
            return
        if query.strip().lower().rstrip("!., ") in _TRIVIAL_PREFIXES:
            return

        from hololoom.apps.server import vault_bridge

        title = f"{agent_name.title()}: {query[:60].strip()}"
        vault_bridge.propose_note(
            title=title,
            content=response,
            agent=agent_name,
        )
        logger.debug("Vault proposal from %s: %s", agent_name, title)
    except Exception:
        pass  # Vault unavailable — agent works fine without it
