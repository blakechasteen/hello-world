"""
WebSocket Transport (Stub)
===========================

Future: real-time interrupt support via WebSocket connection to daemon relay.
MVP: not implemented.
"""

from __future__ import annotations


class WeaverletWSTransport:
    """Placeholder for future WebSocket-based interrupt channel."""

    async def connect(self) -> None:
        raise NotImplementedError("WebSocket transport not available in MVP")

    async def disconnect(self) -> None:
        pass

    async def listen_for_interrupt(self) -> None:
        raise NotImplementedError("WebSocket transport not available in MVP")
