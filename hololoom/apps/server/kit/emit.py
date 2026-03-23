"""
Post-response observers — memory bus, signal bus, vault federation.

Each observer is independent. Each degrades gracefully. None block the caller.

The pattern: Agent.__call__() finishes a response → emit() runs all observers.
Adding a new observer means adding one function here and one line in emit().
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
    Run all post-response observers. Each is fire-and-forget.

    Observers never block the caller. Failures are logged, never raised.
    Add new observers as functions below and wire them here.
    """
    _to_memory_bus(agent_name, query, response, room_id)
    _to_signal_bus(agent_name, query, response, room_id, tokens, duration_ms, extra_payload)
    _to_vault(agent_name, query, response)


# --------------------------------------------------------------------------- #
# Observer 1: Memory bus (cross-agent searchable store)
# --------------------------------------------------------------------------- #

def _to_memory_bus(
    agent_name: str, query: str, response: str, room_id: str,
) -> None:
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
        pass


# --------------------------------------------------------------------------- #
# Observer 2: Signal bus (observability / monitoring)
# --------------------------------------------------------------------------- #

def _to_signal_bus(
    agent_name: str,
    query: str,
    response: str,
    room_id: str,
    tokens: int,
    duration_ms: float,
    extra_payload: dict[str, Any] | None = None,
) -> None:
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
        pass


# --------------------------------------------------------------------------- #
# Observer 3: Vault federation (knowledge capture → PARA inbox)
# --------------------------------------------------------------------------- #
#
# propose_note() already handles quality scoring, rate limiting, content
# validation, and deduplication. This observer just decides whether to
# bother calling it — a lightweight gate so we don't invoke the full
# federation machinery on every "hello" / "thanks".
#

# Responses shorter than this aren't notes — they're replies.
_MIN_RESPONSE_LENGTH = 200

# Queries that never produce note-worthy responses.
_TRIVIAL = frozenset({
    "hi", "hello", "hey", "thanks", "thank you",
    "ok", "sure", "yes", "no", "bye", "goodbye",
})


def _is_worth_proposing(query: str, response: str) -> bool:
    """Lightweight pre-gate. The real quality filter is in propose_note()."""
    if len(response) < _MIN_RESPONSE_LENGTH:
        return False
    normalized = query.strip().lower().rstrip("!.,? ")
    return normalized not in _TRIVIAL


def _to_vault(agent_name: str, query: str, response: str) -> None:
    """Propose substantive responses to the PARA vault. Tracks outcomes."""
    try:
        if not _is_worth_proposing(query, response):
            return

        from hololoom.apps.server.vault_bridge import propose_note, get_tracker

        title = f"{agent_name.title()}: {query[:60].strip()}"
        result = propose_note(
            title=title,
            content=response,
            agent=agent_name,
        )

        status = result.get("status")
        code = result.get("code")
        score = result.get("quality_score", 0)

        if status == "proposed":
            logger.debug(
                "vault/%s proposed (%.2f): %s", agent_name, score, title,
            )
        else:
            # Track the rejection so federation stats reflect reality.
            # FederationTracker.federation_stats.total_rejected exists
            # but was never incremented — until now.
            logger.debug(
                "vault/%s rejected (%s, %.2f): %s",
                agent_name, code, score, title,
            )
            try:
                get_tracker().record_rejection(agent_name)
            except Exception:
                pass  # Tracking is best-effort
    except Exception:
        pass  # Vault unavailable — agent works fine without it
