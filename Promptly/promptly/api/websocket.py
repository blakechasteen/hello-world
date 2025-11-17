"""
WebSocket Support for Real-time Features
"""

from fastapi import WebSocket, WebSocketDisconnect, status
from typing import Dict, Set, Optional
import asyncio
import json
from datetime import datetime
import logging

from .config import get_settings

settings = get_settings()
logger = logging.getLogger("promptly.websocket")


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, dict] = {}
        self.max_connections = settings.ws_max_connections

    async def connect(self, websocket: WebSocket, client_id: str, metadata: dict = None):
        """Accept and register a WebSocket connection"""
        await websocket.accept()

        # Check max connections
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        if total_connections >= self.max_connections:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise Exception("Maximum connections exceeded")

        # Register connection
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()

        self.active_connections[client_id].add(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.utcnow(),
            "metadata": metadata or {}
        }

        logger.info(f"Client {client_id} connected. Total connections: {total_connections + 1}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.connection_metadata:
            client_id = self.connection_metadata[websocket]["client_id"]

            if client_id in self.active_connections:
                self.active_connections[client_id].discard(websocket)

                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]

            del self.connection_metadata[websocket]
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, client_id: str = None):
        """Broadcast message to all connections or specific client"""
        if client_id:
            # Send to specific client's connections
            connections = self.active_connections.get(client_id, set())
        else:
            # Send to all connections
            connections = [ws for conns in self.active_connections.values() for ws in conns]

        disconnected = []

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_heartbeat(self):
        """Send heartbeat to all connections"""
        while True:
            await asyncio.sleep(settings.ws_heartbeat_interval)

            heartbeat = {
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.broadcast(heartbeat)

    def get_stats(self) -> dict:
        """Get connection statistics"""
        total = sum(len(conns) for conns in self.active_connections.values())

        return {
            "total_connections": total,
            "unique_clients": len(self.active_connections),
            "max_connections": self.max_connections
        }


# Global connection manager
manager = ConnectionManager()


class StreamingEvaluationHandler:
    """Handles streaming evaluation results via WebSocket"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_progress(self, step: str, progress: float, data: dict = None):
        """Send progress update"""
        message = {
            "type": "evaluation_progress",
            "step": step,
            "progress": progress,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)

    async def send_result(self, test_index: int, result: dict):
        """Send individual test result"""
        message = {
            "type": "evaluation_result",
            "test_index": test_index,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)

    async def send_complete(self, summary: dict):
        """Send completion message"""
        message = {
            "type": "evaluation_complete",
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)

    async def send_error(self, error: str):
        """Send error message"""
        message = {
            "type": "error",
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)


class StreamingChainHandler:
    """Handles streaming chain execution via WebSocket"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_step_start(self, step_index: int, step_name: str):
        """Send step start notification"""
        message = {
            "type": "chain_step_start",
            "step_index": step_index,
            "step_name": step_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)

    async def send_step_result(self, step_index: int, result: dict):
        """Send step result"""
        message = {
            "type": "chain_step_result",
            "step_index": step_index,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)

    async def send_complete(self, summary: dict):
        """Send completion message"""
        message = {
            "type": "chain_complete",
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)

    async def send_error(self, error: str, step_index: int = None):
        """Send error message"""
        message = {
            "type": "error",
            "error": error,
            "step_index": step_index,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(message, self.websocket)


class BranchUpdateHandler:
    """Handles real-time branch update notifications"""

    @staticmethod
    async def notify_branch_created(branch_name: str, from_branch: str):
        """Notify all clients of new branch"""
        message = {
            "type": "branch_created",
            "branch_name": branch_name,
            "from_branch": from_branch,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.broadcast(message)

    @staticmethod
    async def notify_branch_checkout(branch_name: str, client_id: str):
        """Notify client of branch checkout"""
        message = {
            "type": "branch_checkout",
            "branch_name": branch_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.broadcast(message, client_id=client_id)

    @staticmethod
    async def notify_prompt_updated(prompt_name: str, branch: str, version: int):
        """Notify all clients of prompt update"""
        message = {
            "type": "prompt_updated",
            "prompt_name": prompt_name,
            "branch": branch,
            "version": version,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.broadcast(message)
