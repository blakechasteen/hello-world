"""
MultiplayerWebSocketManager - Real-time bidirectional communication

Manages WebSocket connections for multiplayer games, subscribes to Redis pub/sub,
and routes updates to connected clients.

Author: BigPlay Team
Created: 2025-11-17
"""

import json
import asyncio
from typing import Dict, Optional, Set, Callable
from dataclasses import dataclass
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from redis import asyncio as aioredis


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    player_id: str
    websocket: WebSocket
    connected_at: datetime
    last_ping: datetime


class MultiplayerWebSocketManager:
    """
    Manages WebSocket connections for multiplayer games.

    Features:
    - Connection lifecycle management
    - Redis pub/sub subscription
    - Message broadcasting
    - Player presence tracking
    - Heartbeat/ping system

    Usage:
        manager = MultiplayerWebSocketManager(redis_url="redis://localhost:6379")
        await manager.connect_player(websocket, player_id="player_123")
        await manager.start_redis_listener()
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize WebSocket manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.connections: Dict[str, ConnectionInfo] = {}
        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )

    async def disconnect(self):
        """Disconnect from Redis and close all WebSockets"""
        # Stop background tasks
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all WebSocket connections
        for player_id in list(self.connections.keys()):
            await self.disconnect_player(player_id)

        # Close Redis
        if self.redis:
            await self.redis.close()

    # ===== Connection Management =====

    async def connect_player(self, websocket: WebSocket, player_id: str) -> bool:
        """
        Accept and register new player WebSocket connection.

        Args:
            websocket: FastAPI WebSocket object
            player_id: Player identifier

        Returns:
            True if connected successfully
        """
        try:
            # Accept WebSocket connection
            await websocket.accept()

            # Store connection info
            self.connections[player_id] = ConnectionInfo(
                player_id=player_id,
                websocket=websocket,
                connected_at=datetime.now(),
                last_ping=datetime.now()
            )

            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "player_id": player_id,
                "message": "Connected to BigPlay multiplayer",
                "timestamp": datetime.now().isoformat()
            })

            # Broadcast player joined
            await self.broadcast({
                "type": "player_joined",
                "player_id": player_id,
                "total_players": len(self.connections),
                "timestamp": datetime.now().isoformat()
            }, exclude=player_id)

            return True

        except Exception as e:
            print(f"Error connecting player {player_id}: {e}")
            return False

    async def disconnect_player(self, player_id: str):
        """
        Disconnect player and clean up.

        Args:
            player_id: Player identifier
        """
        if player_id not in self.connections:
            return

        conn = self.connections[player_id]

        try:
            await conn.websocket.close()
        except:
            pass  # Already closed

        # Remove from connections
        del self.connections[player_id]

        # Broadcast player left
        await self.broadcast({
            "type": "player_left",
            "player_id": player_id,
            "total_players": len(self.connections),
            "timestamp": datetime.now().isoformat()
        })

    async def get_active_players(self) -> Set[str]:
        """
        Get set of active player IDs.

        Returns:
            Set of player IDs
        """
        return set(self.connections.keys())

    # ===== Message Sending =====

    async def send_to_player(self, player_id: str, message: dict):
        """
        Send message to specific player.

        Args:
            player_id: Target player ID
            message: Message dictionary
        """
        if player_id not in self.connections:
            return

        conn = self.connections[player_id]

        try:
            await conn.websocket.send_json(message)
        except WebSocketDisconnect:
            await self.disconnect_player(player_id)
        except Exception as e:
            print(f"Error sending to player {player_id}: {e}")
            await self.disconnect_player(player_id)

    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        """
        Broadcast message to all connected players.

        Args:
            message: Message dictionary
            exclude: Optional player ID to exclude from broadcast
        """
        disconnected = []

        for player_id, conn in self.connections.items():
            if player_id == exclude:
                continue

            try:
                await conn.websocket.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(player_id)
            except Exception as e:
                print(f"Error broadcasting to player {player_id}: {e}")
                disconnected.append(player_id)

        # Clean up disconnected players
        for player_id in disconnected:
            await self.disconnect_player(player_id)

    async def broadcast_to_players(self, player_ids: Set[str], message: dict):
        """
        Broadcast message to specific set of players.

        Args:
            player_ids: Set of player IDs
            message: Message dictionary
        """
        for player_id in player_ids:
            if player_id in self.connections:
                await self.send_to_player(player_id, message)

    # ===== Redis Pub/Sub Listener =====

    async def start_redis_listener(self):
        """
        Start listening to Redis pub/sub channels and forward to WebSocket clients.
        Runs as background task.
        """
        self._listener_task = asyncio.create_task(self._redis_listener_loop())

    async def _redis_listener_loop(self):
        """Background task: Listen to Redis and forward messages"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("world:updates", "npc:updates", "world:events")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])

                        # Check if message should exclude a player
                        exclude_player = data.get("exclude_player")

                        # Forward to all connected clients (except excluded)
                        await self.broadcast({
                            "channel": message["channel"],
                            "data": data,
                            "timestamp": datetime.now().isoformat()
                        }, exclude=exclude_player)

                    except json.JSONDecodeError:
                        print(f"Invalid JSON in Redis message: {message['data']}")
                    except Exception as e:
                        print(f"Error processing Redis message: {e}")

        except asyncio.CancelledError:
            await pubsub.close()
            raise
        except Exception as e:
            print(f"Redis listener error: {e}")
            await pubsub.close()

    # ===== Heartbeat System =====

    async def start_heartbeat(self, interval: int = 30):
        """
        Start heartbeat ping system to detect stale connections.

        Args:
            interval: Ping interval in seconds (default: 30)
        """
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval))

    async def _heartbeat_loop(self, interval: int):
        """Background task: Send periodic pings"""
        try:
            while True:
                await asyncio.sleep(interval)
                await self._send_pings()
        except asyncio.CancelledError:
            raise

    async def _send_pings(self):
        """Send ping to all connected players"""
        disconnected = []

        for player_id, conn in self.connections.items():
            try:
                await conn.websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
            except:
                disconnected.append(player_id)

        # Clean up disconnected
        for player_id in disconnected:
            await self.disconnect_player(player_id)

    async def handle_pong(self, player_id: str):
        """
        Handle pong response from player.

        Args:
            player_id: Player who sent pong
        """
        if player_id in self.connections:
            self.connections[player_id].last_ping = datetime.now()

    # ===== Player Message Handler =====

    async def handle_player_message(
        self,
        player_id: str,
        websocket: WebSocket,
        message_handler: Callable
    ):
        """
        Listen for messages from specific player WebSocket.

        Args:
            player_id: Player identifier
            websocket: Player's WebSocket connection
            message_handler: Async function to handle messages
        """
        try:
            while True:
                # Receive message
                data = await websocket.receive_json()

                # Handle pong
                if data.get("type") == "pong":
                    await self.handle_pong(player_id)
                    continue

                # Call custom message handler
                await message_handler(player_id, data)

        except WebSocketDisconnect:
            await self.disconnect_player(player_id)
        except Exception as e:
            print(f"Error handling message from player {player_id}: {e}")
            await self.disconnect_player(player_id)

    # ===== Statistics =====

    async def get_statistics(self) -> dict:
        """
        Get WebSocket manager statistics.

        Returns:
            Dictionary with connection stats
        """
        return {
            "total_connections": len(self.connections),
            "active_players": list(self.connections.keys()),
            "uptime_seconds": [
                {
                    "player_id": player_id,
                    "uptime": (datetime.now() - conn.connected_at).total_seconds()
                }
                for player_id, conn in self.connections.items()
            ],
            "timestamp": datetime.now().isoformat()
        }
