"""
BigPlay Engine - Multiplayer Module

Provides Redis-backed shared world state and WebSocket real-time synchronization
for multiplayer game sessions.

Components:
- SharedWorldState: Redis-backed world flags and NPC state synchronization
- WebSocketManager: Real-time bidirectional communication
- SessionCoordinator: NPC locking and race condition prevention
"""

from .world_state import SharedWorldState
from .websocket_server import MultiplayerWebSocketManager
from .session_coordinator import SessionCoordinator

__all__ = [
    "SharedWorldState",
    "MultiplayerWebSocketManager",
    "SessionCoordinator",
]
