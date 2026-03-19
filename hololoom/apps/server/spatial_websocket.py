"""
Spatial WebSocket Manager
==========================
Real-time WebSocket streaming of spatial scene state.

Clients connect to /ws/spatial/{room_id} and receive scene_update events
whenever the conversation graph changes significantly. Uses the same
pattern-based subscription model as SafetyWebSocketManager.

Protocol:
  Server → Client: connected, scene_update, heartbeat
  Client → Server: heartbeat, request_scene

Author: HoloLoom Team
Date: 2026-03-11
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SpatialClient:
    """A connected WebSocket client subscribed to a room's spatial scene."""
    client_id: str
    room_id: str
    websocket: Any
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)


class SpatialWebSocketManager:
    """
    Manages WebSocket connections for spatial scene broadcasts.

    Room-scoped: each client subscribes to a specific room's scene.
    Broadcasts full scene state on each significant graph change.
    """

    def __init__(self, heartbeat_interval: float = 30.0):
        self.clients: dict[str, SpatialClient] = {}
        self.websocket_to_id: dict[Any, str] = {}
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: asyncio.Task | None = None
        self._client_counter = 0
        self._lock = asyncio.Lock()

        # Callback for scene requests (set by promptly_chat)
        self._scene_provider = None

    async def start(self):
        """Start heartbeat background task."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("SpatialWebSocketManager started")

    async def stop(self):
        """Stop background tasks and close all connections."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        for client in list(self.clients.values()):
            try:
                await client.websocket.close()
            except Exception:
                pass

        self.clients.clear()
        self.websocket_to_id.clear()
        logger.info("SpatialWebSocketManager stopped")

    async def connect(self, websocket: Any, room_id: str) -> str:
        """Handle new WebSocket connection for a room."""
        async with self._lock:
            self._client_counter += 1
            client_id = f"spatial_{self._client_counter}"

            client = SpatialClient(
                client_id=client_id,
                room_id=room_id,
                websocket=websocket,
            )

            self.clients[client_id] = client
            self.websocket_to_id[websocket] = client_id

            logger.info("Spatial client connected: %s (room=%s)", client_id, room_id)

            await self._send(client, {
                "type": "connected",
                "client_id": client_id,
                "room_id": room_id,
                "timestamp": time.time(),
            })

            return client_id

    def disconnect(self, websocket: Any):
        """Handle WebSocket disconnection."""
        client_id = self.websocket_to_id.pop(websocket, None)
        if client_id:
            client = self.clients.pop(client_id, None)
            if client:
                logger.info(
                    "Spatial client disconnected: %s (room=%s)",
                    client_id, client.room_id,
                )

    async def handle_message(self, websocket: Any, data: dict[str, Any]):
        """Handle incoming client message."""
        client_id = self.websocket_to_id.get(websocket)
        if not client_id:
            return

        client = self.clients.get(client_id)
        if not client:
            return

        msg_type = data.get("type", "")

        if msg_type == "heartbeat":
            client.last_heartbeat = datetime.now()
            await self._send(client, {
                "type": "heartbeat_ack",
                "timestamp": time.time(),
            })

        elif msg_type == "request_scene":
            # Client requests current scene state (e.g., on reconnect)
            if self._scene_provider:
                scene = self._scene_provider(client.room_id)
                if scene:
                    await self._send(client, {
                        "type": "scene_update",
                        **scene,
                    })

    async def broadcast_scene_update(self, room_id: str, state: dict[str, Any]):
        """Broadcast scene state to all clients subscribed to this room."""
        event = {
            "type": "scene_update",
            **state,
        }

        sent = 0
        for client in list(self.clients.values()):
            if client.room_id == room_id:
                await self._send(client, event)
                sent += 1

        if sent > 0:
            logger.debug(
                "Spatial scene broadcast: room=%s sent_to=%d clients",
                room_id, sent,
            )

    def set_scene_provider(self, provider):
        """Set callback for request_scene: fn(room_id) -> Optional[Dict]."""
        self._scene_provider = provider

    @property
    def client_count(self) -> int:
        return len(self.clients)

    def clients_for_room(self, room_id: str) -> int:
        return sum(1 for c in self.clients.values() if c.room_id == room_id)

    async def _send(self, client: SpatialClient, data: dict[str, Any]):
        """Send message to a client."""
        try:
            await client.websocket.send_json(data)
        except Exception as e:
            logger.warning("Failed to send to spatial client %s: %s", client.client_id, e)

    async def _heartbeat_loop(self):
        """Background heartbeat check."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                now = datetime.now()
                for client in list(self.clients.values()):
                    elapsed = (now - client.last_heartbeat).total_seconds()
                    if elapsed > self.heartbeat_interval * 2:
                        logger.warning("Spatial client %s stale, closing", client.client_id)
                        try:
                            await client.websocket.close()
                        except Exception:
                            pass
                        self.disconnect(client.websocket)
                    else:
                        await self._send(client, {
                            "type": "heartbeat",
                            "timestamp": time.time(),
                        })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Spatial heartbeat error: %s", e)


# ============================================================================
# Module-level singleton + FastAPI setup
# ============================================================================

_spatial_ws_manager: SpatialWebSocketManager | None = None


def get_spatial_ws_manager() -> SpatialWebSocketManager | None:
    """Get the spatial WebSocket manager singleton."""
    return _spatial_ws_manager


def setup_spatial_routes(app):
    """
    Mount the spatial WebSocket endpoint on a FastAPI app.

    Call this during app setup when PROMPTLY_JENNY_SPATIAL=true.
    """
    global _spatial_ws_manager

    from fastapi import WebSocket, WebSocketDisconnect

    _spatial_ws_manager = SpatialWebSocketManager()

    @app.on_event("startup")
    async def _start_spatial_ws():
        await _spatial_ws_manager.start()

    @app.on_event("shutdown")
    async def _stop_spatial_ws():
        await _spatial_ws_manager.stop()

    @app.websocket("/ws/spatial/{room_id}")
    async def spatial_ws_endpoint(websocket: WebSocket, room_id: str):
        await websocket.accept()
        await _spatial_ws_manager.connect(websocket, room_id)
        try:
            while True:
                data = await websocket.receive_json()
                await _spatial_ws_manager.handle_message(websocket, data)
        except WebSocketDisconnect:
            _spatial_ws_manager.disconnect(websocket)
        except Exception:
            _spatial_ws_manager.disconnect(websocket)

    logger.info("Spatial WebSocket routes mounted at /ws/spatial/{room_id}")
