#!/usr/bin/env python3
"""
WebSocket Progress Streaming for HoloLoom Jobs
================================================

Provides real-time progress updates for async weave jobs via WebSocket,
replacing polling with push-based updates.

Architecture:
- JobProgressManager: Manages WebSocket connections by job_id
- JobProgressMessage: Standardized progress message format
- Integration with HoloLoomMatrixHandlers._pending_jobs

Usage:
    # With FastAPI
    from fastapi import FastAPI, WebSocket
    from HoloLoom.chatops.handlers.websocket_progress import (
        JobProgressManager, create_progress_router
    )

    app = FastAPI()
    progress_manager = JobProgressManager()
    app.include_router(create_progress_router(progress_manager))

    # In hololoom_handlers.py
    handlers = HoloLoomMatrixHandlers(bot)
    handlers.set_progress_broadcaster(progress_manager)

Message Types:
- JOB_QUEUED: Job submitted, waiting to start
- JOB_STARTED: Job execution beginning
- JOB_PROGRESS: Step-by-step progress update
- JOB_COMPLETED: Job finished successfully
- JOB_FAILED: Job failed with error
- JOB_CANCELLED: Job was cancelled
- HEARTBEAT: Connection keepalive
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Set, Any, Callable, List

logger = logging.getLogger(__name__)


# ============================================================================
# Message Types
# ============================================================================

class JobMessageType(Enum):
    """Types of job progress messages."""
    JOB_QUEUED = "job_queued"
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


@dataclass
class JobProgressMessage:
    """Standardized job progress message for WebSocket streaming."""
    type: JobMessageType
    job_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Progress fields (for JOB_PROGRESS)
    current_step: int = 0
    total_steps: int = 9
    step_name: str = ""
    percentage: float = 0.0
    message: str = ""

    # Status fields
    status: str = ""
    query: str = ""
    user_id: str = ""
    room_id: str = ""

    # Result fields (for JOB_COMPLETED)
    tool_used: Optional[str] = None
    confidence: Optional[float] = None
    duration_ms: Optional[float] = None
    response_preview: Optional[str] = None

    # Error fields (for JOB_FAILED)
    error: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['type'] = self.type.value
        # Remove None values for cleaner JSON
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# ============================================================================
# Connection Manager
# ============================================================================

class JobProgressManager:
    """
    Manages WebSocket connections for job progress streaming.

    Supports:
    - Subscribe to specific job_id
    - Subscribe to all jobs in a room_id
    - Subscribe to all jobs from a user_id
    - Broadcast progress updates to subscribers
    - Connection lifecycle management
    - Heartbeat for connection keepalive
    """

    def __init__(self, heartbeat_interval: float = 30.0):
        """
        Initialize the progress manager.

        Args:
            heartbeat_interval: Seconds between heartbeat messages
        """
        # WebSocket connections by connection_id
        self._connections: Dict[str, Any] = {}  # connection_id -> WebSocket

        # Subscriptions: connection_id -> set of patterns
        # Patterns: "job:{job_id}", "room:{room_id}", "user:{user_id}", "*"
        self._subscriptions: Dict[str, Set[str]] = {}

        # Reverse index: pattern -> set of connection_ids
        self._pattern_subscribers: Dict[str, Set[str]] = {}

        # Heartbeat task
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        # Message queue for buffering during disconnects
        self._message_buffer: Dict[str, List[JobProgressMessage]] = {}
        self._buffer_size = 100  # Max buffered messages per pattern

        logger.info("JobProgressManager initialized")

    async def start(self):
        """Start the heartbeat loop."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("JobProgressManager heartbeat started")

    async def stop(self):
        """Stop the heartbeat loop and close all connections."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn_id in list(self._connections.keys()):
            await self.disconnect(conn_id)

        logger.info("JobProgressManager stopped")

    async def _heartbeat_loop(self):
        """Send heartbeat messages to all connections."""
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                if not self._connections:
                    continue

                heartbeat = JobProgressMessage(
                    type=JobMessageType.HEARTBEAT,
                    job_id="*",
                    message="keepalive",
                    metadata={"active_connections": len(self._connections)}
                )

                # Send to all connections
                for conn_id in list(self._connections.keys()):
                    try:
                        await self._send_to_connection(conn_id, heartbeat)
                    except Exception as e:
                        logger.warning(f"Heartbeat failed for {conn_id}: {e}")
                        await self.disconnect(conn_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    async def connect(self, websocket: Any, connection_id: str) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection (FastAPI WebSocket or compatible)
            connection_id: Unique identifier for this connection
        """
        try:
            await websocket.accept()
            self._connections[connection_id] = websocket
            self._subscriptions[connection_id] = set()

            logger.info(f"WebSocket connected: {connection_id}")

            # Send welcome message
            welcome = JobProgressMessage(
                type=JobMessageType.SUBSCRIBED,
                job_id="*",
                message="Connected to job progress stream",
                metadata={"connection_id": connection_id}
            )
            await self._send_to_connection(connection_id, welcome)

        except Exception as e:
            logger.error(f"Connection failed for {connection_id}: {e}")
            raise

    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect and cleanup a WebSocket connection.

        Args:
            connection_id: Connection to disconnect
        """
        # Remove from subscriptions
        if connection_id in self._subscriptions:
            patterns = self._subscriptions.pop(connection_id)
            for pattern in patterns:
                if pattern in self._pattern_subscribers:
                    self._pattern_subscribers[pattern].discard(connection_id)
                    if not self._pattern_subscribers[pattern]:
                        del self._pattern_subscribers[pattern]

        # Close WebSocket
        if connection_id in self._connections:
            websocket = self._connections.pop(connection_id)
            try:
                await websocket.close()
            except Exception:
                pass

        logger.info(f"WebSocket disconnected: {connection_id}")

    def subscribe(self, connection_id: str, pattern: str) -> bool:
        """
        Subscribe a connection to a pattern.

        Args:
            connection_id: Connection to subscribe
            pattern: Pattern to subscribe to (job:{id}, room:{id}, user:{id}, *)

        Returns:
            True if subscribed successfully
        """
        if connection_id not in self._connections:
            return False

        # Add to subscription set
        self._subscriptions[connection_id].add(pattern)

        # Add to reverse index
        if pattern not in self._pattern_subscribers:
            self._pattern_subscribers[pattern] = set()
        self._pattern_subscribers[pattern].add(connection_id)

        logger.debug(f"Connection {connection_id} subscribed to {pattern}")

        # Deliver any buffered messages
        if pattern in self._message_buffer:
            asyncio.create_task(self._deliver_buffered(connection_id, pattern))

        return True

    def unsubscribe(self, connection_id: str, pattern: str) -> bool:
        """
        Unsubscribe a connection from a pattern.

        Args:
            connection_id: Connection to unsubscribe
            pattern: Pattern to unsubscribe from

        Returns:
            True if unsubscribed successfully
        """
        if connection_id not in self._subscriptions:
            return False

        self._subscriptions[connection_id].discard(pattern)

        if pattern in self._pattern_subscribers:
            self._pattern_subscribers[pattern].discard(connection_id)
            if not self._pattern_subscribers[pattern]:
                del self._pattern_subscribers[pattern]

        logger.debug(f"Connection {connection_id} unsubscribed from {pattern}")
        return True

    async def _deliver_buffered(self, connection_id: str, pattern: str) -> None:
        """Deliver buffered messages to a newly subscribed connection."""
        if pattern not in self._message_buffer:
            return

        for msg in self._message_buffer[pattern]:
            try:
                await self._send_to_connection(connection_id, msg)
            except Exception as e:
                logger.warning(f"Failed to deliver buffered message: {e}")
                break

    async def _send_to_connection(self, connection_id: str, message: JobProgressMessage) -> None:
        """Send a message to a specific connection."""
        if connection_id not in self._connections:
            return

        websocket = self._connections[connection_id]
        try:
            await websocket.send_json(message.to_dict())
        except Exception as e:
            logger.warning(f"Send failed for {connection_id}: {e}")
            await self.disconnect(connection_id)
            raise

    def _get_subscribers_for_job(self, job_id: str, room_id: str = "", user_id: str = "") -> Set[str]:
        """Get all connection_ids that should receive updates for a job."""
        subscribers = set()

        # Check exact job subscription
        job_pattern = f"job:{job_id}"
        if job_pattern in self._pattern_subscribers:
            subscribers.update(self._pattern_subscribers[job_pattern])

        # Check room subscription
        if room_id:
            room_pattern = f"room:{room_id}"
            if room_pattern in self._pattern_subscribers:
                subscribers.update(self._pattern_subscribers[room_pattern])

        # Check user subscription
        if user_id:
            user_pattern = f"user:{user_id}"
            if user_pattern in self._pattern_subscribers:
                subscribers.update(self._pattern_subscribers[user_pattern])

        # Check wildcard subscription
        if "*" in self._pattern_subscribers:
            subscribers.update(self._pattern_subscribers["*"])

        return subscribers

    async def broadcast_progress(
        self,
        job_id: str,
        message_type: JobMessageType,
        room_id: str = "",
        user_id: str = "",
        current_step: int = 0,
        total_steps: int = 9,
        step_name: str = "",
        message: str = "",
        query: str = "",
        tool_used: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_ms: Optional[float] = None,
        response_preview: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast a progress update to all subscribers.

        Args:
            job_id: Job identifier
            message_type: Type of progress message
            room_id: Optional room for room-based subscriptions
            user_id: Optional user for user-based subscriptions
            current_step: Current step number
            total_steps: Total steps in pipeline
            step_name: Name of current step
            message: Human-readable progress message
            query: Original query text
            tool_used: Tool selected (for completed jobs)
            confidence: Confidence score (for completed jobs)
            duration_ms: Duration in ms (for completed jobs)
            response_preview: Preview of response (for completed jobs)
            error: Error message (for failed jobs)
            metadata: Additional metadata

        Returns:
            Number of connections that received the message
        """
        # Create message
        percentage = (current_step / total_steps * 100) if total_steps > 0 else 0.0

        msg = JobProgressMessage(
            type=message_type,
            job_id=job_id,
            current_step=current_step,
            total_steps=total_steps,
            step_name=step_name,
            percentage=percentage,
            message=message,
            status=message_type.value.replace("job_", ""),
            query=query,
            user_id=user_id,
            room_id=room_id,
            tool_used=tool_used,
            confidence=confidence,
            duration_ms=duration_ms,
            response_preview=response_preview,
            error=error,
            metadata=metadata or {}
        )

        # Get subscribers
        subscribers = self._get_subscribers_for_job(job_id, room_id, user_id)

        # Buffer message for late subscribers
        job_pattern = f"job:{job_id}"
        if job_pattern not in self._message_buffer:
            self._message_buffer[job_pattern] = []
        self._message_buffer[job_pattern].append(msg)
        if len(self._message_buffer[job_pattern]) > self._buffer_size:
            self._message_buffer[job_pattern] = self._message_buffer[job_pattern][-self._buffer_size:]

        # Send to subscribers
        sent_count = 0
        for conn_id in subscribers:
            try:
                await self._send_to_connection(conn_id, msg)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Broadcast failed for {conn_id}: {e}")

        logger.debug(f"Broadcast {message_type.value} for job {job_id} to {sent_count} subscribers")
        return sent_count

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "active_connections": len(self._connections),
            "total_subscriptions": sum(len(s) for s in self._subscriptions.values()),
            "patterns": list(self._pattern_subscribers.keys()),
            "buffered_patterns": list(self._message_buffer.keys()),
            "running": self._running
        }

    def clear_job_buffer(self, job_id: str) -> None:
        """Clear buffered messages for a completed/cancelled job."""
        pattern = f"job:{job_id}"
        if pattern in self._message_buffer:
            del self._message_buffer[pattern]


# ============================================================================
# FastAPI Integration
# ============================================================================

def create_progress_router(manager: JobProgressManager):
    """
    Create a FastAPI router for WebSocket job progress streaming.

    Args:
        manager: JobProgressManager instance

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
    except ImportError:
        logger.warning("FastAPI not available, WebSocket router not created")
        return None

    router = APIRouter(prefix="/jobs", tags=["Job Progress"])

    @router.websocket("/ws/progress")
    async def websocket_progress(
        websocket: WebSocket,
        job_id: str = Query(None, description="Subscribe to specific job"),
        room_id: str = Query(None, description="Subscribe to room's jobs"),
        user_id: str = Query(None, description="Subscribe to user's jobs"),
        all_jobs: bool = Query(False, description="Subscribe to all jobs")
    ):
        """
        WebSocket endpoint for job progress streaming.

        Query parameters determine subscription:
        - job_id: Subscribe to a specific job
        - room_id: Subscribe to all jobs in a room
        - user_id: Subscribe to all jobs from a user
        - all_jobs: Subscribe to all jobs (admin only)

        Messages:
        - Receive: {"action": "subscribe"/"unsubscribe", "pattern": "job:xxx"}
        - Send: JobProgressMessage
        """
        import uuid
        connection_id = str(uuid.uuid4())[:8]

        try:
            await manager.connect(websocket, connection_id)

            # Auto-subscribe based on query params
            if job_id:
                manager.subscribe(connection_id, f"job:{job_id}")
            if room_id:
                manager.subscribe(connection_id, f"room:{room_id}")
            if user_id:
                manager.subscribe(connection_id, f"user:{user_id}")
            if all_jobs:
                manager.subscribe(connection_id, "*")

            # If no subscriptions, subscribe to all
            if not manager._subscriptions.get(connection_id):
                manager.subscribe(connection_id, "*")

            # Handle incoming messages
            while True:
                try:
                    data = await websocket.receive_json()
                    action = data.get("action")
                    pattern = data.get("pattern")

                    if action == "subscribe" and pattern:
                        manager.subscribe(connection_id, pattern)
                        await manager._send_to_connection(
                            connection_id,
                            JobProgressMessage(
                                type=JobMessageType.SUBSCRIBED,
                                job_id="*",
                                message=f"Subscribed to {pattern}"
                            )
                        )

                    elif action == "unsubscribe" and pattern:
                        manager.unsubscribe(connection_id, pattern)
                        await manager._send_to_connection(
                            connection_id,
                            JobProgressMessage(
                                type=JobMessageType.UNSUBSCRIBED,
                                job_id="*",
                                message=f"Unsubscribed from {pattern}"
                            )
                        )

                    elif action == "ping":
                        await manager._send_to_connection(
                            connection_id,
                            JobProgressMessage(
                                type=JobMessageType.HEARTBEAT,
                                job_id="*",
                                message="pong"
                            )
                        )

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(f"WebSocket message error: {e}")
                    await manager._send_to_connection(
                        connection_id,
                        JobProgressMessage(
                            type=JobMessageType.ERROR,
                            job_id="*",
                            error=str(e)
                        )
                    )

        finally:
            await manager.disconnect(connection_id)

    @router.get("/progress/stats")
    async def get_progress_stats():
        """Get job progress manager statistics."""
        return manager.get_stats()

    return router


# ============================================================================
# Integration Helper
# ============================================================================

class JobProgressBroadcaster:
    """
    Helper class to integrate JobProgressManager with HoloLoomMatrixHandlers.

    This wraps the manager and provides convenient methods that match
    the job lifecycle in hololoom_handlers.py.
    """

    def __init__(self, manager: JobProgressManager):
        """
        Initialize broadcaster.

        Args:
            manager: JobProgressManager instance
        """
        self.manager = manager

    async def job_queued(self, job_id: str, query: str, room_id: str, user_id: str) -> None:
        """Broadcast that a job has been queued."""
        await self.manager.broadcast_progress(
            job_id=job_id,
            message_type=JobMessageType.JOB_QUEUED,
            room_id=room_id,
            user_id=user_id,
            current_step=0,
            total_steps=9,
            step_name="Queued",
            message="Job submitted, waiting in queue",
            query=query
        )

    async def job_started(self, job_id: str, query: str, room_id: str, user_id: str) -> None:
        """Broadcast that a job has started."""
        await self.manager.broadcast_progress(
            job_id=job_id,
            message_type=JobMessageType.JOB_STARTED,
            room_id=room_id,
            user_id=user_id,
            current_step=1,
            total_steps=9,
            step_name="Initializing",
            message="Starting weaving cycle",
            query=query
        )

    async def job_progress(
        self,
        job_id: str,
        current_step: int,
        step_name: str,
        message: str,
        room_id: str = "",
        user_id: str = "",
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast job progress update."""
        await self.manager.broadcast_progress(
            job_id=job_id,
            message_type=JobMessageType.JOB_PROGRESS,
            room_id=room_id,
            user_id=user_id,
            current_step=current_step,
            total_steps=9,
            step_name=step_name,
            message=message,
            query=query,
            metadata=metadata
        )

    async def job_completed(
        self,
        job_id: str,
        tool_used: str,
        confidence: float,
        duration_ms: float,
        response_preview: str = "",
        room_id: str = "",
        user_id: str = "",
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast that a job has completed."""
        await self.manager.broadcast_progress(
            job_id=job_id,
            message_type=JobMessageType.JOB_COMPLETED,
            room_id=room_id,
            user_id=user_id,
            current_step=9,
            total_steps=9,
            step_name="Complete",
            message="Weaving cycle complete",
            query=query,
            tool_used=tool_used,
            confidence=confidence,
            duration_ms=duration_ms,
            response_preview=response_preview[:500] if response_preview else "",
            metadata=metadata
        )

        # Clear buffer after completion
        self.manager.clear_job_buffer(job_id)

    async def job_failed(
        self,
        job_id: str,
        error: str,
        room_id: str = "",
        user_id: str = "",
        query: str = ""
    ) -> None:
        """Broadcast that a job has failed."""
        await self.manager.broadcast_progress(
            job_id=job_id,
            message_type=JobMessageType.JOB_FAILED,
            room_id=room_id,
            user_id=user_id,
            step_name="Failed",
            message=f"Job failed: {error[:100]}",
            query=query,
            error=error
        )

        # Clear buffer after failure
        self.manager.clear_job_buffer(job_id)

    async def job_cancelled(
        self,
        job_id: str,
        room_id: str = "",
        user_id: str = ""
    ) -> None:
        """Broadcast that a job was cancelled."""
        await self.manager.broadcast_progress(
            job_id=job_id,
            message_type=JobMessageType.JOB_CANCELLED,
            room_id=room_id,
            user_id=user_id,
            step_name="Cancelled",
            message="Job cancelled by user"
        )

        # Clear buffer after cancellation
        self.manager.clear_job_buffer(job_id)


# ============================================================================
# Singleton for Easy Access
# ============================================================================

_global_manager: Optional[JobProgressManager] = None


def get_global_manager() -> JobProgressManager:
    """Get or create the global JobProgressManager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = JobProgressManager()
    return _global_manager


def get_global_broadcaster() -> JobProgressBroadcaster:
    """Get a broadcaster using the global manager."""
    return JobProgressBroadcaster(get_global_manager())


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
WebSocket Progress Streaming for HoloLoom Jobs
================================================

This module provides real-time progress updates for async weave jobs.

Components:
- JobProgressManager: Manages WebSocket connections
- JobProgressMessage: Standardized message format
- JobProgressBroadcaster: Integration helper for handlers

Message Types:
- JOB_QUEUED: Job submitted, waiting
- JOB_STARTED: Execution beginning
- JOB_PROGRESS: Step-by-step updates
- JOB_COMPLETED: Finished successfully
- JOB_FAILED: Failed with error
- JOB_CANCELLED: Cancelled by user

Subscription Patterns:
- job:{job_id} - Specific job updates
- room:{room_id} - All jobs in room
- user:{user_id} - All jobs from user
- * - All jobs (admin)

FastAPI Integration:
    from fastapi import FastAPI
    from HoloLoom.chatops.handlers.websocket_progress import (
        JobProgressManager, create_progress_router
    )

    app = FastAPI()
    manager = JobProgressManager()
    router = create_progress_router(manager)
    if router:
        app.include_router(router)

    # Start manager
    @app.on_event("startup")
    async def startup():
        await manager.start()

    @app.on_event("shutdown")
    async def shutdown():
        await manager.stop()

WebSocket Client (JavaScript):
    const ws = new WebSocket('ws://localhost:8000/jobs/ws/progress?job_id=weave-abc123');

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        console.log(`${msg.type}: ${msg.message} (${msg.percentage}%)`);

        if (msg.type === 'job_completed') {
            console.log(`Result: ${msg.tool_used} (${msg.confidence}%)`);
        }
    };

    // Subscribe to more patterns
    ws.send(JSON.stringify({action: 'subscribe', pattern: 'room:!abc123:matrix.org'}));
""")
