"""
WebRTC Signaling Server

FastAPI-based WebSocket server for WebRTC signaling.
Facilitates peer-to-peer connections between avatar clients.

Features:
- WebSocket connections for real-time signaling
- Peer discovery and session management
- Offer/Answer/ICE candidate relay
- Automatic peer list broadcasting
- Connection monitoring

Usage:
    python signaling_server.py

    Or with uvicorn:
    uvicorn signaling_server:app --host 0.0.0.0 --port 8080

Created: 2025-11-22 (Phase 6.2)
"""

import asyncio
import json
import logging
from typing import Dict, Set
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="WebRTC Signaling Server", version="1.0.0")

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Connection Management
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections and peer discovery"""

    def __init__(self):
        # Active connections: user_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

        # Connection metadata
        self.connection_times: Dict[str, datetime] = {}

        # Sessions (for future multi-room support)
        self.sessions: Dict[str, Set[str]] = {}  # session_id -> set of user_ids

    async def connect(self, user_id: str, websocket: WebSocket):
        """Register new connection"""
        await websocket.accept()

        self.active_connections[user_id] = websocket
        self.connection_times[user_id] = datetime.now()

        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")

        # Send current peer list to new user
        await self.send_peer_list(user_id)

        # Notify all other users about new peer
        await self.broadcast_peer_list(exclude=user_id)

    def disconnect(self, user_id: str):
        """Unregister connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]

        if user_id in self.connection_times:
            del self.connection_times[user_id]

        logger.info(f"User {user_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {user_id}: {e}")

    async def relay_message(self, message: dict, from_user: str, to_user: str):
        """Relay message from one user to another"""
        if to_user in self.active_connections:
            # Add sender info
            message['from'] = from_user

            await self.send_personal_message(message, to_user)
        else:
            logger.warning(f"Cannot relay to {to_user}: not connected")

    async def send_peer_list(self, user_id: str):
        """Send current peer list to user"""
        # Get all peers except the requesting user
        peers = [uid for uid in self.active_connections.keys() if uid != user_id]

        message = {
            'type': 'peer-list',
            'data': {
                'peers': peers,
                'count': len(peers),
            }
        }

        await self.send_personal_message(message, user_id)

    async def broadcast_peer_list(self, exclude: str = None):
        """Broadcast updated peer list to all users"""
        for user_id in self.active_connections.keys():
            if user_id != exclude:
                await self.send_peer_list(user_id)

    async def broadcast_leave(self, user_id: str):
        """Notify all users that a peer has left"""
        message = {
            'type': 'leave',
            'from': user_id,
        }

        for uid in self.active_connections.keys():
            if uid != user_id:
                await self.send_personal_message(message, uid)

    def get_stats(self) -> dict:
        """Get server statistics"""
        return {
            'total_connections': len(self.active_connections),
            'users': list(self.active_connections.keys()),
            'uptime_seconds': {
                user_id: (datetime.now() - conn_time).total_seconds()
                for user_id, conn_time in self.connection_times.items()
            }
        }


# Global connection manager
manager = ConnectionManager()


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for WebRTC signaling

    Message format:
    {
        "type": "join" | "offer" | "answer" | "ice-candidate" | "leave",
        "from": "user-id",
        "to": "user-id",  // Optional, for targeted messages
        "data": {}  // Type-specific data
    }
    """
    user_id = None

    try:
        # Wait for join message with user ID
        await websocket.accept()

        # First message should be JOIN
        raw_message = await websocket.receive_text()
        join_message = json.loads(raw_message)

        if join_message.get('type') != 'join':
            logger.warning("First message must be JOIN")
            await websocket.close(code=1002, reason="First message must be JOIN")
            return

        user_id = join_message.get('from')
        if not user_id:
            logger.warning("JOIN message missing user ID")
            await websocket.close(code=1002, reason="Missing user ID")
            return

        # Register connection
        await manager.connect(user_id, websocket)

        # Message handling loop
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)

            message_type = message.get('type')
            from_user = message.get('from', user_id)
            to_user = message.get('to')

            logger.debug(f"Message from {from_user}: {message_type} -> {to_user}")

            # Handle different message types
            if message_type in ['offer', 'answer', 'ice-candidate']:
                # Relay WebRTC signaling messages to target peer
                if to_user:
                    await manager.relay_message(message, from_user, to_user)
                else:
                    logger.warning(f"Signaling message missing 'to' field: {message_type}")

            elif message_type == 'leave':
                # User leaving
                break

            else:
                logger.warning(f"Unknown message type: {message_type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {user_id}")

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)

    finally:
        # Cleanup on disconnect
        if user_id:
            manager.disconnect(user_id)
            await manager.broadcast_leave(user_id)


# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "WebRTC Signaling Server",
        "version": "1.0.0",
    }


@app.get("/stats")
async def stats():
    """Get server statistics"""
    return manager.get_stats()


@app.get("/health")
async def health():
    """Detailed health check"""
    stats = manager.get_stats()

    return {
        "status": "healthy",
        "connections": stats['total_connections'],
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Server startup"""
    logger.info("=" * 60)
    logger.info("WebRTC Signaling Server Starting")
    logger.info("=" * 60)
    logger.info("Endpoints:")
    logger.info("  WebSocket: ws://localhost:8080/ws")
    logger.info("  HTTP:      http://localhost:8080")
    logger.info("  Stats:     http://localhost:8080/stats")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown"""
    logger.info("WebRTC Signaling Server Shutting Down")

    # Close all connections
    for user_id in list(manager.active_connections.keys()):
        websocket = manager.active_connections[user_id]
        await websocket.close(code=1001, reason="Server shutdown")
        manager.disconnect(user_id)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
    )
