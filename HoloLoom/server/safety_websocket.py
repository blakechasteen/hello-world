"""
Safety WebSocket Handler
========================
Real-time WebSocket streaming for safety dashboard.

E2.2 Implementation - December 2025

Features:
- WebSocket connections for live safety events
- Pattern-based subscriptions (safety:*, deception:*, etc.)
- Integration with AlignmentMonitor
- Heartbeat and connection management
- Event broadcasting for safety decisions, deception flags, etc.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Set, List, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger("HoloLoom.server.safety_websocket")


@dataclass
class SafetyClient:
    """Represents a connected WebSocket client."""
    client_id: str
    websocket: Any  # WebSocket connection
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)


class SafetyWebSocketManager:
    """
    Manages WebSocket connections for safety dashboard.

    Usage:
        manager = SafetyWebSocketManager()

        # In FastAPI:
        @app.websocket("/ws/safety")
        async def safety_ws(websocket: WebSocket):
            await manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_json()
                    await manager.handle_message(websocket, data)
            except WebSocketDisconnect:
                manager.disconnect(websocket)

        # Broadcast events:
        await manager.broadcast_safety_decision(risk_level="HIGH", outcome="allowed")
        await manager.broadcast_deception_flag(flag_type="HIDDEN_GOAL", description="...")
    """

    def __init__(self, heartbeat_interval: float = 30.0):
        """
        Initialize WebSocket manager.

        Args:
            heartbeat_interval: Seconds between heartbeat checks
        """
        self.clients: Dict[str, SafetyClient] = {}
        self.websocket_to_id: Dict[Any, str] = {}
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._client_counter = 0
        self._lock = asyncio.Lock()

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

    async def start(self):
        """Start background tasks."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("SafetyWebSocketManager started")

    async def stop(self):
        """Stop background tasks and close all connections."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Close all connections
        for client in list(self.clients.values()):
            try:
                await client.websocket.close()
            except Exception:
                pass

        self.clients.clear()
        self.websocket_to_id.clear()
        logger.info("SafetyWebSocketManager stopped")

    async def connect(self, websocket: Any) -> str:
        """
        Handle new WebSocket connection.

        Args:
            websocket: WebSocket connection

        Returns:
            Client ID
        """
        async with self._lock:
            self._client_counter += 1
            client_id = f"safety_client_{self._client_counter}"

            client = SafetyClient(
                client_id=client_id,
                websocket=websocket,
                subscriptions={"safety:*"}  # Default subscription
            )

            self.clients[client_id] = client
            self.websocket_to_id[websocket] = client_id

            logger.info(f"Safety client connected: {client_id}")

            # Send welcome message
            await self._send_to_client(client, {
                "type": "connected",
                "client_id": client_id,
                "timestamp": time.time()
            })

            return client_id

    def disconnect(self, websocket: Any):
        """
        Handle WebSocket disconnection.

        Args:
            websocket: WebSocket connection
        """
        client_id = self.websocket_to_id.get(websocket)
        if client_id:
            del self.clients[client_id]
            del self.websocket_to_id[websocket]
            logger.info(f"Safety client disconnected: {client_id}")

    async def handle_message(self, websocket: Any, data: Dict[str, Any]):
        """
        Handle incoming WebSocket message.

        Args:
            websocket: WebSocket connection
            data: Message data
        """
        client_id = self.websocket_to_id.get(websocket)
        if not client_id:
            return

        client = self.clients.get(client_id)
        if not client:
            return

        msg_type = data.get("type", "")

        if msg_type == "subscribe":
            # Subscribe to pattern
            pattern = data.get("pattern", "")
            if pattern:
                client.subscriptions.add(pattern)
                logger.debug(f"Client {client_id} subscribed to: {pattern}")
                await self._send_to_client(client, {
                    "type": "subscribed",
                    "pattern": pattern,
                    "timestamp": time.time()
                })

        elif msg_type == "unsubscribe":
            # Unsubscribe from pattern
            pattern = data.get("pattern", "")
            client.subscriptions.discard(pattern)
            logger.debug(f"Client {client_id} unsubscribed from: {pattern}")

        elif msg_type == "heartbeat":
            # Update last heartbeat
            client.last_heartbeat = datetime.now()
            await self._send_to_client(client, {
                "type": "heartbeat_ack",
                "timestamp": time.time()
            })

        elif msg_type == "get_metrics":
            # Request current metrics
            metrics = await self._get_current_metrics()
            await self._send_to_client(client, {
                "type": "metrics_update",
                "alignment_metrics": metrics,
                "timestamp": time.time()
            })

    async def broadcast_safety_decision(
        self,
        risk_level: str,
        outcome: str,
        action: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast safety decision event.

        Args:
            risk_level: Risk level (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
            outcome: Decision outcome (allowed, blocked)
            action: Action being gated
            details: Additional details
        """
        event = {
            "type": "safety_decision",
            "risk_level": risk_level,
            "outcome": outcome,
            "action": action,
            "details": details or {},
            "timestamp": time.time()
        }
        await self._broadcast("safety:decision", event)

    async def broadcast_deception_flag(
        self,
        flag_type: str,
        description: str,
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast deception detection flag.

        Args:
            flag_type: Type of deception (HIDDEN_GOAL, GOAL_DRIFT, etc.)
            description: Description of the detected behavior
            confidence: Detection confidence (0.0-1.0)
            details: Additional details
        """
        event = {
            "type": "deception_flag",
            "flag_type": flag_type,
            "description": description,
            "confidence": confidence,
            "details": details or {},
            "timestamp": time.time()
        }
        await self._broadcast("safety:deception", event)

        # Also broadcast as alert
        await self.broadcast_alert(
            level="CRITICAL",
            title=f"Deception Detected: {flag_type}",
            message=description
        )

    async def broadcast_convergence_violation(
        self,
        violation_type: str,
        description: str,
        severity: str = "WARNING",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast convergence violation event.

        Args:
            violation_type: Type (POWER_SEEKING, RESOURCE_ACQUISITION, etc.)
            description: Description of the violation
            severity: Severity level
            details: Additional details
        """
        event = {
            "type": "convergence_violation",
            "violation_type": violation_type,
            "description": description,
            "severity": severity,
            "details": details or {},
            "timestamp": time.time()
        }
        await self._broadcast("safety:convergence", event)

        # Also broadcast as alert
        await self.broadcast_alert(
            level=severity,
            title=f"Convergence Violation: {violation_type}",
            message=description
        )

    async def broadcast_autonomy_action(
        self,
        step_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast autonomy action event.

        Args:
            step_type: Type of action (research, verification, plan_execute, direct)
            description: Description of the action
            details: Additional details
        """
        event = {
            "type": "autonomy_action",
            "step_type": step_type,
            "description": description,
            "details": details or {},
            "timestamp": time.time()
        }
        await self._broadcast("safety:autonomy", event)

    async def broadcast_resource_update(
        self,
        resource_type: str,
        ratio: float,
        current: float = 0.0,
        limit: float = 0.0
    ):
        """
        Broadcast resource utilization update.

        Args:
            resource_type: Type (API_CALLS, COMPUTE, MEMORY, etc.)
            ratio: Utilization ratio (0.0-1.0)
            current: Current usage
            limit: Maximum limit
        """
        event = {
            "type": "resource_update",
            "resource_type": resource_type,
            "ratio": ratio,
            "current": current,
            "limit": limit,
            "timestamp": time.time()
        }
        await self._broadcast("safety:resource", event)

    async def broadcast_alert(
        self,
        level: str,
        title: str,
        message: str
    ):
        """
        Broadcast alert event.

        Args:
            level: Alert level (WARNING, CRITICAL)
            title: Alert title
            message: Alert message
        """
        event = {
            "type": "alert",
            "level": level,
            "title": title,
            "message": message,
            "timestamp": time.time()
        }
        await self._broadcast("safety:alert", event)

    async def broadcast_metrics_update(self, metrics: Dict[str, Any]):
        """
        Broadcast alignment metrics update.

        Args:
            metrics: Alignment metrics dictionary
        """
        event = {
            "type": "metrics_update",
            "alignment_metrics": metrics,
            "timestamp": time.time()
        }
        await self._broadcast("safety:metrics", event)

    async def _broadcast(self, channel: str, event: Dict[str, Any]):
        """
        Broadcast event to subscribed clients.

        Args:
            channel: Event channel (e.g., "safety:decision")
            event: Event data
        """
        for client in list(self.clients.values()):
            if self._matches_subscription(channel, client.subscriptions):
                await self._send_to_client(client, event)

    def _matches_subscription(self, channel: str, subscriptions: Set[str]) -> bool:
        """
        Check if channel matches any subscription pattern.

        Args:
            channel: Event channel
            subscriptions: Set of subscription patterns

        Returns:
            True if matches
        """
        for pattern in subscriptions:
            if pattern == "*":
                return True
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if channel.startswith(prefix):
                    return True
            if pattern == channel:
                return True
        return False

    async def _send_to_client(self, client: SafetyClient, data: Dict[str, Any]):
        """
        Send message to specific client.

        Args:
            client: Target client
            data: Message data
        """
        try:
            await client.websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Failed to send to client {client.client_id}: {e}")
            # Don't disconnect here - let the receive loop handle it

    async def _heartbeat_loop(self):
        """Background task for heartbeat checks."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Send heartbeat to all clients
                now = datetime.now()
                for client in list(self.clients.values()):
                    # Check if client is stale (no heartbeat for 2x interval)
                    elapsed = (now - client.last_heartbeat).total_seconds()
                    if elapsed > self.heartbeat_interval * 2:
                        logger.warning(f"Client {client.client_id} stale, closing")
                        try:
                            await client.websocket.close()
                        except Exception:
                            pass
                        self.disconnect(client.websocket)
                    else:
                        # Send heartbeat ping
                        await self._send_to_client(client, {
                            "type": "heartbeat",
                            "timestamp": time.time()
                        })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def _get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current alignment metrics.

        Returns:
            Metrics dictionary
        """
        try:
            from HoloLoom.alignment import get_global_monitor
            monitor = get_global_monitor()
            if monitor:
                return monitor.get_alignment_summary()
        except Exception as e:
            logger.error(f"Failed to get alignment metrics: {e}")

        return {}

    @property
    def connected_clients(self) -> int:
        """Number of connected clients."""
        return len(self.clients)


# Global instance for easy access
_safety_ws_manager: Optional[SafetyWebSocketManager] = None


def get_safety_ws_manager() -> SafetyWebSocketManager:
    """Get global SafetyWebSocketManager instance."""
    global _safety_ws_manager
    if _safety_ws_manager is None:
        _safety_ws_manager = SafetyWebSocketManager()
    return _safety_ws_manager


async def setup_safety_websocket(app):
    """
    Setup safety WebSocket endpoint on FastAPI app.

    Args:
        app: FastAPI application

    Usage:
        from fastapi import FastAPI
        from HoloLoom.server.safety_websocket import setup_safety_websocket

        app = FastAPI()
        await setup_safety_websocket(app)
    """
    from fastapi import WebSocket, WebSocketDisconnect

    manager = get_safety_ws_manager()
    await manager.start()

    @app.websocket("/ws/safety")
    async def safety_websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        client_id = await manager.connect(websocket)

        try:
            while True:
                data = await websocket.receive_json()
                await manager.handle_message(websocket, data)
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
            manager.disconnect(websocket)

    @app.on_event("shutdown")
    async def shutdown_ws_manager():
        await manager.stop()

    logger.info("Safety WebSocket endpoint registered at /ws/safety")
