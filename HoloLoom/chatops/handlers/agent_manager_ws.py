#!/usr/bin/env python3
"""
WebSocket Message Handlers for Agent Manager
=============================================

Provides real-time updates for multi-threaded agent execution via WebSocket.

Architecture:
- AgentManagerWSManager: Manages WebSocket connections for agent threads
- AgentManagerMessage: Standardized message format
- Integration with AgentManagerHub

Message Types (from plan):
- Thread lifecycle: THREAD_CREATED, THREAD_STARTED, THREAD_PAUSED, etc.
- Progress: STEP_STARTED, STEP_PROGRESS, STEP_COMPLETED
- Priority: PRIORITY_CHANGED
- Swarm: SWARM_SPAWNED, SWARM_CHILD_COMPLETED, SWARM_MERGED
- Injection: MRF_INJECTED, MCTS_INJECTED
- Files: FILE_ACTIVITY
- Temporal: SNAPSHOT_CREATED, ROLLBACK_COMPLETED, BRANCH_CREATED, etc.

Date: December 2025
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Set, Any, List, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Message Types
# ============================================================================

class AgentManagerMessageType(Enum):
    """Types of agent manager messages."""

    # Thread lifecycle
    THREAD_CREATED = "thread_created"
    THREAD_STARTED = "thread_started"
    THREAD_PAUSED = "thread_paused"
    THREAD_RESUMED = "thread_resumed"
    THREAD_COMPLETED = "thread_completed"
    THREAD_CANCELLED = "thread_cancelled"
    THREAD_ERROR = "thread_error"

    # Progress
    STEP_STARTED = "step_started"
    STEP_PROGRESS = "step_progress"
    STEP_COMPLETED = "step_completed"

    # Priority
    PRIORITY_CHANGED = "priority_changed"

    # Swarm
    SWARM_CREATED = "swarm_created"
    SWARM_SPAWNED = "swarm_spawned"
    SWARM_CHILD_COMPLETED = "swarm_child_completed"
    SWARM_MERGED = "swarm_merged"

    # Injection
    MRF_INJECTED = "mrf_injected"
    MCTS_INJECTED = "mcts_injected"

    # Files (for file tree viewer)
    FILE_ACTIVITY = "file_activity"

    # Temporal (git-style rollback)
    COMMIT_CREATED = "commit_created"
    BRANCH_CREATED = "branch_created"
    BRANCH_DELETED = "branch_deleted"
    CHECKOUT_COMPLETED = "checkout_completed"
    RESET_COMPLETED = "reset_completed"
    MERGE_COMPLETED = "merge_completed"
    CHERRY_PICK_COMPLETED = "cherry_pick_completed"

    # Cost tracking
    COST_UPDATED = "cost_updated"

    # System
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


@dataclass
class AgentManagerMessage:
    """Standardized agent manager message for WebSocket streaming."""
    type: AgentManagerMessageType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Thread fields
    thread_id: Optional[str] = None
    thread_name: Optional[str] = None
    thread_status: Optional[str] = None

    # Progress fields
    current_step: int = 0
    total_steps: int = 0
    step_name: str = ""
    step_type: str = ""
    percentage: float = 0.0
    message: str = ""

    # Confidence
    confidence: float = 0.0
    epistemic_confidence: float = 0.0

    # Priority
    priority: int = 0
    old_priority: Optional[int] = None

    # Swarm fields
    swarm_id: Optional[str] = None
    child_thread_ids: Optional[List[str]] = None
    merge_strategy: Optional[str] = None

    # Injection fields
    strategy_name: Optional[str] = None
    mcts_budget: Optional[int] = None
    step_id: Optional[str] = None

    # Files
    files: Optional[List[str]] = None
    file_action: Optional[str] = None  # "read", "write", "delete"

    # Temporal (git) fields
    commit_sha: Optional[str] = None
    commit_message: Optional[str] = None
    parent_sha: Optional[str] = None
    branch_name: Optional[str] = None
    from_branch: Optional[str] = None
    to_branch: Optional[str] = None

    # Cost fields
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    provider: Optional[str] = None
    model: Optional[str] = None
    is_local: bool = False

    # Timing
    duration_ms: float = 0.0
    elapsed_ms: float = 0.0
    eta_ms: Optional[float] = None

    # Error
    error: Optional[str] = None
    error_type: Optional[str] = None

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

class AgentManagerWSManager:
    """
    Manages WebSocket connections for agent manager streaming.

    Supports:
    - Subscribe to specific thread_id
    - Subscribe to swarm updates
    - Subscribe to all updates from a project
    - Broadcast progress updates to subscribers
    - Connection lifecycle management
    - Heartbeat for connection keepalive
    """

    def __init__(self, heartbeat_interval: float = 30.0):
        """
        Initialize the WebSocket manager.

        Args:
            heartbeat_interval: Seconds between heartbeat messages
        """
        # WebSocket connections by connection_id
        self._connections: Dict[str, Any] = {}

        # Subscriptions: connection_id -> set of patterns
        # Patterns: "thread:{id}", "swarm:{id}", "project:{id}", "*"
        self._subscriptions: Dict[str, Set[str]] = {}

        # Reverse index: pattern -> set of connection_ids
        self._pattern_subscribers: Dict[str, Set[str]] = {}

        # Heartbeat task
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        # Message queue for buffering during disconnects
        self._message_buffer: Dict[str, List[AgentManagerMessage]] = {}
        self._buffer_size = 100

        # Event handlers (for hub integration)
        self._event_handlers: Dict[str, List[Callable]] = {}

        logger.info("AgentManagerWSManager initialized")

    async def start(self):
        """Start the heartbeat loop."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("AgentManagerWSManager heartbeat started")

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

        logger.info("AgentManagerWSManager stopped")

    async def _heartbeat_loop(self):
        """Send heartbeat messages to all connections."""
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                if not self._connections:
                    continue

                heartbeat = AgentManagerMessage(
                    type=AgentManagerMessageType.HEARTBEAT,
                    message="keepalive",
                    metadata={"active_connections": len(self._connections)}
                )

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
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()
            self._connections[connection_id] = websocket
            self._subscriptions[connection_id] = set()

            logger.info(f"Agent Manager WebSocket connected: {connection_id}")

            welcome = AgentManagerMessage(
                type=AgentManagerMessageType.SUBSCRIBED,
                message="Connected to agent manager stream",
                metadata={"connection_id": connection_id}
            )
            await self._send_to_connection(connection_id, welcome)

        except Exception as e:
            logger.error(f"Connection failed for {connection_id}: {e}")
            raise

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect and cleanup a WebSocket connection."""
        if connection_id in self._subscriptions:
            patterns = self._subscriptions.pop(connection_id)
            for pattern in patterns:
                if pattern in self._pattern_subscribers:
                    self._pattern_subscribers[pattern].discard(connection_id)
                    if not self._pattern_subscribers[pattern]:
                        del self._pattern_subscribers[pattern]

        if connection_id in self._connections:
            websocket = self._connections.pop(connection_id)
            try:
                await websocket.close()
            except Exception:
                pass

        logger.info(f"Agent Manager WebSocket disconnected: {connection_id}")

    def subscribe(self, connection_id: str, pattern: str) -> bool:
        """Subscribe a connection to a pattern."""
        if connection_id not in self._connections:
            return False

        self._subscriptions[connection_id].add(pattern)

        if pattern not in self._pattern_subscribers:
            self._pattern_subscribers[pattern] = set()
        self._pattern_subscribers[pattern].add(connection_id)

        logger.debug(f"Connection {connection_id} subscribed to {pattern}")

        if pattern in self._message_buffer:
            asyncio.create_task(self._deliver_buffered(connection_id, pattern))

        return True

    def unsubscribe(self, connection_id: str, pattern: str) -> bool:
        """Unsubscribe a connection from a pattern."""
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

    async def _send_to_connection(
        self,
        connection_id: str,
        message: AgentManagerMessage
    ) -> None:
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

    def _get_subscribers(
        self,
        thread_id: Optional[str] = None,
        swarm_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Set[str]:
        """Get all connection_ids that should receive updates."""
        subscribers = set()

        if thread_id:
            pattern = f"thread:{thread_id}"
            if pattern in self._pattern_subscribers:
                subscribers.update(self._pattern_subscribers[pattern])

        if swarm_id:
            pattern = f"swarm:{swarm_id}"
            if pattern in self._pattern_subscribers:
                subscribers.update(self._pattern_subscribers[pattern])

        if project_id:
            pattern = f"project:{project_id}"
            if pattern in self._pattern_subscribers:
                subscribers.update(self._pattern_subscribers[pattern])

        # Wildcard subscribers
        if "*" in self._pattern_subscribers:
            subscribers.update(self._pattern_subscribers["*"])

        return subscribers

    async def broadcast(
        self,
        message: AgentManagerMessage,
        thread_id: Optional[str] = None,
        swarm_id: Optional[str] = None,
        project_id: Optional[str] = None,
        buffer: bool = True
    ) -> int:
        """
        Broadcast a message to all relevant subscribers.

        Args:
            message: Message to broadcast
            thread_id: Thread ID for filtering
            swarm_id: Swarm ID for filtering
            project_id: Project ID for filtering
            buffer: Whether to buffer for late subscribers

        Returns:
            Number of connections that received the message
        """
        # Get subscribers
        subscribers = self._get_subscribers(thread_id, swarm_id, project_id)

        # Buffer message
        if buffer and thread_id:
            pattern = f"thread:{thread_id}"
            if pattern not in self._message_buffer:
                self._message_buffer[pattern] = []
            self._message_buffer[pattern].append(message)
            if len(self._message_buffer[pattern]) > self._buffer_size:
                self._message_buffer[pattern] = \
                    self._message_buffer[pattern][-self._buffer_size:]

        # Send to subscribers
        sent_count = 0
        for conn_id in subscribers:
            try:
                await self._send_to_connection(conn_id, message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Broadcast failed for {conn_id}: {e}")

        logger.debug(
            f"Broadcast {message.type.value} to {sent_count} subscribers"
        )

        # Trigger event handlers
        event_type = message.type.value
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

        return sent_count

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[AgentManagerMessage], Any]
    ) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def unregister_handler(
        self,
        event_type: str,
        handler: Callable[[AgentManagerMessage], Any]
    ) -> None:
        """Unregister an event handler."""
        if event_type in self._event_handlers:
            if handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)

    def clear_thread_buffer(self, thread_id: str) -> None:
        """Clear buffered messages for a completed thread."""
        pattern = f"thread:{thread_id}"
        if pattern in self._message_buffer:
            del self._message_buffer[pattern]

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "active_connections": len(self._connections),
            "total_subscriptions": sum(
                len(s) for s in self._subscriptions.values()
            ),
            "patterns": list(self._pattern_subscribers.keys()),
            "buffered_patterns": list(self._message_buffer.keys()),
            "running": self._running,
            "event_handlers": {
                k: len(v) for k, v in self._event_handlers.items()
            }
        }


# ============================================================================
# Broadcaster Helper
# ============================================================================

class AgentManagerBroadcaster:
    """
    Helper class to broadcast agent manager events.

    Provides convenient methods that match the thread lifecycle.
    """

    def __init__(self, manager: AgentManagerWSManager):
        """Initialize broadcaster."""
        self.manager = manager

    # -------------------------------------------------------------------------
    # Thread Lifecycle
    # -------------------------------------------------------------------------

    async def thread_created(
        self,
        thread_id: str,
        thread_name: str,
        query: str,
        mode: str,
        priority: int = 0,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast thread creation."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_CREATED,
            thread_id=thread_id,
            thread_name=thread_name,
            thread_status="created",
            priority=priority,
            message=f"Thread created: {thread_name}",
            metadata={
                "query": query,
                "mode": mode,
                "project_id": project_id
            }
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def thread_started(
        self,
        thread_id: str,
        thread_name: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast thread start."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_STARTED,
            thread_id=thread_id,
            thread_name=thread_name,
            thread_status="running",
            message=f"Thread started: {thread_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def thread_paused(
        self,
        thread_id: str,
        thread_name: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast thread pause."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_PAUSED,
            thread_id=thread_id,
            thread_name=thread_name,
            thread_status="paused",
            message=f"Thread paused: {thread_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def thread_resumed(
        self,
        thread_id: str,
        thread_name: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast thread resume."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_RESUMED,
            thread_id=thread_id,
            thread_name=thread_name,
            thread_status="running",
            message=f"Thread resumed: {thread_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def thread_completed(
        self,
        thread_id: str,
        thread_name: str,
        confidence: float,
        duration_ms: float,
        response_preview: str = "",
        project_id: Optional[str] = None,
        cost_usd: float = 0.0
    ) -> None:
        """Broadcast thread completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_COMPLETED,
            thread_id=thread_id,
            thread_name=thread_name,
            thread_status="completed",
            confidence=confidence,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            message=f"Thread completed: {thread_name}",
            metadata={"response_preview": response_preview[:500]}
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )
        self.manager.clear_thread_buffer(thread_id)

    async def thread_cancelled(
        self,
        thread_id: str,
        thread_name: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast thread cancellation."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_CANCELLED,
            thread_id=thread_id,
            thread_name=thread_name,
            thread_status="cancelled",
            message=f"Thread cancelled: {thread_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )
        self.manager.clear_thread_buffer(thread_id)

    async def thread_error(
        self,
        thread_id: str,
        thread_name: str,
        error: str,
        error_type: str = "RuntimeError",
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast thread error."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_ERROR,
            thread_id=thread_id,
            thread_name=thread_name,
            thread_status="failed",
            error=error,
            error_type=error_type,
            message=f"Thread error: {thread_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )
        self.manager.clear_thread_buffer(thread_id)

    # -------------------------------------------------------------------------
    # Progress
    # -------------------------------------------------------------------------

    async def step_started(
        self,
        thread_id: str,
        step_index: int,
        total_steps: int,
        step_name: str,
        step_type: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast step start."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.STEP_STARTED,
            thread_id=thread_id,
            current_step=step_index,
            total_steps=total_steps,
            step_name=step_name,
            step_type=step_type,
            percentage=(step_index / total_steps * 100) if total_steps > 0 else 0,
            message=f"Step {step_index}/{total_steps}: {step_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def step_progress(
        self,
        thread_id: str,
        step_index: int,
        total_steps: int,
        step_name: str,
        confidence: float = 0.0,
        epistemic_confidence: float = 0.0,
        elapsed_ms: float = 0.0,
        eta_ms: Optional[float] = None,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast step progress."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.STEP_PROGRESS,
            thread_id=thread_id,
            current_step=step_index,
            total_steps=total_steps,
            step_name=step_name,
            percentage=(step_index / total_steps * 100) if total_steps > 0 else 0,
            confidence=confidence,
            epistemic_confidence=epistemic_confidence,
            elapsed_ms=elapsed_ms,
            eta_ms=eta_ms,
            metadata=metadata or {}
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def step_completed(
        self,
        thread_id: str,
        step_index: int,
        total_steps: int,
        step_name: str,
        step_type: str,
        duration_ms: float,
        confidence: float = 0.0,
        project_id: Optional[str] = None,
        finding: Optional[str] = None
    ) -> None:
        """Broadcast step completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.STEP_COMPLETED,
            thread_id=thread_id,
            current_step=step_index,
            total_steps=total_steps,
            step_name=step_name,
            step_type=step_type,
            percentage=(step_index / total_steps * 100) if total_steps > 0 else 0,
            duration_ms=duration_ms,
            confidence=confidence,
            message=f"Step completed: {step_name}",
            metadata={"finding": finding} if finding else {}
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    # -------------------------------------------------------------------------
    # Priority
    # -------------------------------------------------------------------------

    async def priority_changed(
        self,
        thread_id: str,
        thread_name: str,
        new_priority: int,
        old_priority: int,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast priority change."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.PRIORITY_CHANGED,
            thread_id=thread_id,
            thread_name=thread_name,
            priority=new_priority,
            old_priority=old_priority,
            message=f"Priority changed: {old_priority} -> {new_priority}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    # -------------------------------------------------------------------------
    # Swarm
    # -------------------------------------------------------------------------

    async def swarm_created(
        self,
        swarm_id: str,
        parent_thread_id: str,
        fan_out_count: int,
        merge_strategy: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast swarm creation."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.SWARM_CREATED,
            swarm_id=swarm_id,
            thread_id=parent_thread_id,
            merge_strategy=merge_strategy,
            message=f"Swarm created with {fan_out_count} children",
            metadata={"fan_out_count": fan_out_count}
        )
        await self.manager.broadcast(
            msg, swarm_id=swarm_id, project_id=project_id
        )

    async def swarm_spawned(
        self,
        swarm_id: str,
        child_thread_ids: List[str],
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast swarm child spawn."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.SWARM_SPAWNED,
            swarm_id=swarm_id,
            child_thread_ids=child_thread_ids,
            message=f"Spawned {len(child_thread_ids)} child threads"
        )
        await self.manager.broadcast(
            msg, swarm_id=swarm_id, project_id=project_id
        )

    async def swarm_child_completed(
        self,
        swarm_id: str,
        child_thread_id: str,
        confidence: float,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast swarm child completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.SWARM_CHILD_COMPLETED,
            swarm_id=swarm_id,
            thread_id=child_thread_id,
            confidence=confidence,
            message=f"Child thread completed: {child_thread_id}"
        )
        await self.manager.broadcast(
            msg, swarm_id=swarm_id, project_id=project_id
        )

    async def swarm_merged(
        self,
        swarm_id: str,
        merged_thread_id: str,
        merge_strategy: str,
        confidence: float,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast swarm merge completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.SWARM_MERGED,
            swarm_id=swarm_id,
            thread_id=merged_thread_id,
            merge_strategy=merge_strategy,
            confidence=confidence,
            message=f"Swarm merged using {merge_strategy} strategy"
        )
        await self.manager.broadcast(
            msg, swarm_id=swarm_id, project_id=project_id
        )

    # -------------------------------------------------------------------------
    # Injection
    # -------------------------------------------------------------------------

    async def mrf_injected(
        self,
        thread_id: str,
        step_id: str,
        strategy_name: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast MRF injection."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.MRF_INJECTED,
            thread_id=thread_id,
            step_id=step_id,
            strategy_name=strategy_name,
            message=f"MRF injected: {strategy_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def mcts_injected(
        self,
        thread_id: str,
        step_id: str,
        mcts_budget: int,
        exploration_constant: float = 1.41,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast MCTS injection."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.MCTS_INJECTED,
            thread_id=thread_id,
            step_id=step_id,
            mcts_budget=mcts_budget,
            message=f"MCTS injected: budget={mcts_budget}",
            metadata={"exploration_constant": exploration_constant}
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------

    async def file_activity(
        self,
        thread_id: str,
        files: List[str],
        action: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast file activity."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.FILE_ACTIVITY,
            thread_id=thread_id,
            files=files,
            file_action=action,
            message=f"File {action}: {len(files)} files"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    # -------------------------------------------------------------------------
    # Temporal (Git-style)
    # -------------------------------------------------------------------------

    async def commit_created(
        self,
        thread_id: str,
        commit_sha: str,
        commit_message: str,
        parent_sha: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast commit creation."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.COMMIT_CREATED,
            thread_id=thread_id,
            commit_sha=commit_sha,
            commit_message=commit_message,
            parent_sha=parent_sha,
            message=f"Commit: {commit_message[:50]}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def branch_created(
        self,
        thread_id: str,
        branch_name: str,
        from_sha: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast branch creation."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.BRANCH_CREATED,
            thread_id=thread_id,
            branch_name=branch_name,
            commit_sha=from_sha,
            message=f"Branch created: {branch_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def branch_deleted(
        self,
        thread_id: str,
        branch_name: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast branch deletion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.BRANCH_DELETED,
            thread_id=thread_id,
            branch_name=branch_name,
            message=f"Branch deleted: {branch_name}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def checkout_completed(
        self,
        thread_id: str,
        target: str,
        is_branch: bool = True,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast checkout completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.CHECKOUT_COMPLETED,
            thread_id=thread_id,
            branch_name=target if is_branch else None,
            commit_sha=target if not is_branch else None,
            message=f"Checked out: {target}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def reset_completed(
        self,
        thread_id: str,
        target_sha: str,
        mode: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast reset completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.RESET_COMPLETED,
            thread_id=thread_id,
            commit_sha=target_sha,
            message=f"Reset ({mode}) to {target_sha[:8]}",
            metadata={"mode": mode}
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def merge_completed(
        self,
        thread_id: str,
        source_branch: str,
        target_branch: str,
        merge_sha: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast merge completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.MERGE_COMPLETED,
            thread_id=thread_id,
            from_branch=source_branch,
            to_branch=target_branch,
            commit_sha=merge_sha,
            message=f"Merged {source_branch} into {target_branch}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    async def cherry_pick_completed(
        self,
        thread_id: str,
        source_sha: str,
        new_sha: str,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast cherry-pick completion."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.CHERRY_PICK_COMPLETED,
            thread_id=thread_id,
            commit_sha=new_sha,
            parent_sha=source_sha,
            message=f"Cherry-picked {source_sha[:8]}"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )

    # -------------------------------------------------------------------------
    # Cost Tracking
    # -------------------------------------------------------------------------

    async def cost_updated(
        self,
        thread_id: str,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float,
        provider: str,
        model: str,
        is_local: bool = False,
        project_id: Optional[str] = None
    ) -> None:
        """Broadcast cost update."""
        msg = AgentManagerMessage(
            type=AgentManagerMessageType.COST_UPDATED,
            thread_id=thread_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            provider=provider,
            model=model,
            is_local=is_local,
            message=f"Cost: ${cost_usd:.4f}" if not is_local else "Cost: $0.00 (local)"
        )
        await self.manager.broadcast(
            msg, thread_id=thread_id, project_id=project_id
        )


# ============================================================================
# FastAPI Integration
# ============================================================================

def create_agent_manager_router(manager: AgentManagerWSManager):
    """
    Create a FastAPI router for WebSocket agent manager streaming.

    Args:
        manager: AgentManagerWSManager instance

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
    except ImportError:
        logger.warning("FastAPI not available, WebSocket router not created")
        return None

    router = APIRouter(prefix="/agent-manager", tags=["Agent Manager"])

    @router.websocket("/ws")
    async def websocket_agent_manager(
        websocket: WebSocket,
        thread_id: str = Query(None, description="Subscribe to specific thread"),
        swarm_id: str = Query(None, description="Subscribe to swarm updates"),
        project_id: str = Query(None, description="Subscribe to project threads"),
        all_updates: bool = Query(False, description="Subscribe to all updates")
    ):
        """
        WebSocket endpoint for agent manager streaming.

        Query parameters determine subscription:
        - thread_id: Subscribe to a specific thread
        - swarm_id: Subscribe to swarm updates
        - project_id: Subscribe to all threads in project
        - all_updates: Subscribe to all updates

        Messages:
        - Receive: {"action": "subscribe"/"unsubscribe", "pattern": "thread:xxx"}
        - Send: AgentManagerMessage
        """
        import uuid
        connection_id = str(uuid.uuid4())[:8]

        try:
            await manager.connect(websocket, connection_id)

            # Auto-subscribe based on query params
            if thread_id:
                manager.subscribe(connection_id, f"thread:{thread_id}")
            if swarm_id:
                manager.subscribe(connection_id, f"swarm:{swarm_id}")
            if project_id:
                manager.subscribe(connection_id, f"project:{project_id}")
            if all_updates:
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
                            AgentManagerMessage(
                                type=AgentManagerMessageType.SUBSCRIBED,
                                message=f"Subscribed to {pattern}"
                            )
                        )

                    elif action == "unsubscribe" and pattern:
                        manager.unsubscribe(connection_id, pattern)
                        await manager._send_to_connection(
                            connection_id,
                            AgentManagerMessage(
                                type=AgentManagerMessageType.UNSUBSCRIBED,
                                message=f"Unsubscribed from {pattern}"
                            )
                        )

                    elif action == "ping":
                        await manager._send_to_connection(
                            connection_id,
                            AgentManagerMessage(
                                type=AgentManagerMessageType.HEARTBEAT,
                                message="pong"
                            )
                        )

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(f"WebSocket message error: {e}")
                    await manager._send_to_connection(
                        connection_id,
                        AgentManagerMessage(
                            type=AgentManagerMessageType.ERROR,
                            error=str(e)
                        )
                    )

        finally:
            await manager.disconnect(connection_id)

    @router.get("/ws/stats")
    async def get_websocket_stats():
        """Get WebSocket manager statistics."""
        return manager.get_stats()

    return router


# ============================================================================
# Singleton for Easy Access
# ============================================================================

_global_manager: Optional[AgentManagerWSManager] = None


def get_global_manager() -> AgentManagerWSManager:
    """Get or create the global AgentManagerWSManager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = AgentManagerWSManager()
    return _global_manager


def get_global_broadcaster() -> AgentManagerBroadcaster:
    """Get a broadcaster using the global manager."""
    return AgentManagerBroadcaster(get_global_manager())


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
Agent Manager WebSocket Handlers
================================

This module provides real-time updates for multi-threaded agent execution.

Components:
- AgentManagerWSManager: Manages WebSocket connections
- AgentManagerMessage: Standardized message format
- AgentManagerBroadcaster: High-level broadcast API

Message Types:
- Thread: CREATED, STARTED, PAUSED, RESUMED, COMPLETED, CANCELLED, ERROR
- Progress: STEP_STARTED, STEP_PROGRESS, STEP_COMPLETED
- Priority: PRIORITY_CHANGED
- Swarm: CREATED, SPAWNED, CHILD_COMPLETED, MERGED
- Injection: MRF_INJECTED, MCTS_INJECTED
- Files: FILE_ACTIVITY
- Temporal: COMMIT_CREATED, BRANCH_*, CHECKOUT_*, RESET_*, MERGE_*
- Cost: COST_UPDATED

Subscription Patterns:
- thread:{thread_id} - Specific thread updates
- swarm:{swarm_id} - Swarm updates
- project:{project_id} - All threads in project
- * - All updates

FastAPI Integration:
    from fastapi import FastAPI
    from HoloLoom.chatops.handlers.agent_manager_ws import (
        AgentManagerWSManager, create_agent_manager_router
    )

    app = FastAPI()
    manager = AgentManagerWSManager()
    router = create_agent_manager_router(manager)
    if router:
        app.include_router(router)

    @app.on_event("startup")
    async def startup():
        await manager.start()

    @app.on_event("shutdown")
    async def shutdown():
        await manager.stop()

WebSocket Client (JavaScript):
    const ws = new WebSocket('ws://localhost:8002/agent-manager/ws?thread_id=abc123');

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        console.log(`${msg.type}: ${msg.message}`);

        if (msg.type === 'thread_completed') {
            console.log(`Confidence: ${msg.confidence}`);
        }
    };

    // Subscribe to more patterns
    ws.send(JSON.stringify({action: 'subscribe', pattern: 'swarm:xyz789'}));
""")
