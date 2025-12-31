"""
Ritual Message Bus
==================

Created: December 30, 2025
Purpose: Priority-queued message bus for inter-phase communication

Mirrors HoloLoom/agentic/multi_agent.py:MessageBus pattern.
Enables ritual phases to communicate via structured messages with:
- Priority ordering (CRITICAL > HIGH > NORMAL > LOW)
- TTL-based message expiration
- Correlation IDs for workflow tracking
- Async handler dispatch
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import heapq
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Message Types and Priorities
# ============================================================================

class MessageType(Enum):
    """Types of inter-phase messages."""
    REQUEST = "request"           # Request phase execution
    RESPONSE = "response"         # Phase execution result
    HANDOFF = "handoff"           # Context handoff between phases
    BROADCAST = "broadcast"       # Notify all phases
    CONTROL = "control"           # Control flow (pause, resume, cancel)


class MessagePriority(Enum):
    """Message priorities (lower value = higher priority)."""
    CRITICAL = 0    # Safety, errors - processed first
    HIGH = 1        # User decisions - high priority
    NORMAL = 2      # Standard phase messages
    LOW = 3         # Background, learning - lowest priority


# ============================================================================
# Message Data Structure
# ============================================================================

@dataclass
class PhaseMessage:
    """
    Message passed between ritual phases.

    Supports correlation tracking, priority ordering, and TTL expiration.
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Workflow tracking (links related messages)
    correlation_id: Optional[str] = None  # Links all messages in a workflow
    causation_id: Optional[str] = None    # What message triggered this one

    # Routing
    from_phase: str = ""       # Source phase (empty = orchestrator)
    to_phase: str = ""         # Target phase (empty = broadcast)

    # Message properties
    type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL

    # Content
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: Optional[int] = None  # Time-to-live (None = infinite)

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(self.created_at)
        return datetime.now() > created + timedelta(seconds=self.ttl_seconds)

    def __lt__(self, other: 'PhaseMessage') -> bool:
        """For heapq priority comparison (lower priority value = higher priority)."""
        if not isinstance(other, PhaseMessage):
            return NotImplemented
        # Primary: priority value (lower = more urgent)
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # Secondary: creation time (older = more urgent)
        return self.created_at < other.created_at

    def __eq__(self, other: object) -> bool:
        """Equality based on message ID."""
        if not isinstance(other, PhaseMessage):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on message ID."""
        return hash(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "type": self.type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhaseMessage':
        """Deserialize message from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            from_phase=data.get("from_phase", ""),
            to_phase=data.get("to_phase", ""),
            type=MessageType(data.get("type", "request")),
            priority=MessagePriority(data.get("priority", 2)),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            ttl_seconds=data.get("ttl_seconds"),
        )


# Type alias for message handlers
MessageHandler = Callable[[PhaseMessage], Awaitable[Optional[PhaseMessage]]]


# ============================================================================
# Message Bus Implementation
# ============================================================================

class RitualMessageBus:
    """
    Priority-queued message bus for inter-phase communication.

    Features:
    - Priority ordering (CRITICAL > HIGH > NORMAL > LOW)
    - TTL-based message expiration
    - Correlation IDs for workflow tracking
    - Async handler dispatch
    - Message history for debugging

    Usage:
        bus = RitualMessageBus()

        # Subscribe to messages
        async def my_handler(msg: PhaseMessage) -> Optional[PhaseMessage]:
            print(f"Received: {msg.payload}")
            return None  # Or return response message

        bus.subscribe("my_phase", my_handler)

        # Send messages
        await bus.send(PhaseMessage(
            to_phase="my_phase",
            type=MessageType.REQUEST,
            payload={"action": "process"}
        ))
    """

    def __init__(self, history_limit: int = 1000):
        """
        Initialize message bus.

        Args:
            history_limit: Maximum messages to keep in history
        """
        self._queue: List[PhaseMessage] = []  # heapq
        self._handlers: Dict[str, List[MessageHandler]] = {}  # phase_id → handlers
        self._broadcast_handlers: List[MessageHandler] = []
        self._message_history: List[PhaseMessage] = []
        self._history_limit = history_limit
        self._lock = asyncio.Lock()
        self._processing = False
        self._processed_ids: Set[str] = set()  # Prevent duplicate processing

    async def send(self, message: PhaseMessage) -> str:
        """
        Send a message to the bus.

        Args:
            message: Message to send

        Returns:
            Message ID
        """
        async with self._lock:
            heapq.heappush(self._queue, message)
            self._add_to_history(message)

        # Dispatch to handlers (non-blocking)
        asyncio.create_task(self._dispatch(message))

        logger.debug(f"Message sent: {message.id} ({message.type.value}) → {message.to_phase or 'broadcast'}")
        return message.id

    async def _dispatch(self, message: PhaseMessage) -> None:
        """Dispatch message to appropriate handlers."""
        # Skip expired messages
        if message.is_expired():
            logger.debug(f"Skipping expired message: {message.id}")
            return

        # Skip already processed messages (prevent duplicates)
        if message.id in self._processed_ids:
            return
        self._processed_ids.add(message.id)

        # Limit processed IDs set size
        if len(self._processed_ids) > self._history_limit * 2:
            # Remove oldest entries (approximately)
            oldest = list(self._processed_ids)[:len(self._processed_ids) // 2]
            for oid in oldest:
                self._processed_ids.discard(oid)

        handlers: List[MessageHandler] = []

        if message.to_phase:
            # Targeted message
            handlers = self._handlers.get(message.to_phase, [])
        else:
            # Broadcast
            handlers = self._broadcast_handlers.copy()

        # Execute handlers
        for handler in handlers:
            try:
                response = await handler(message)
                if response is not None:
                    # Handler returned a response message
                    response.correlation_id = message.correlation_id or message.id
                    response.causation_id = message.id
                    await self.send(response)
            except Exception as e:
                logger.error(f"Handler error for message {message.id}: {e}")

    def _add_to_history(self, message: PhaseMessage) -> None:
        """Add message to history, maintaining limit."""
        self._message_history.append(message)
        if len(self._message_history) > self._history_limit:
            self._message_history = self._message_history[-self._history_limit:]

    def subscribe(self, phase_id: str, handler: MessageHandler) -> str:
        """
        Subscribe a phase to receive messages.

        Args:
            phase_id: Phase identifier
            handler: Async handler function

        Returns:
            Subscription ID
        """
        if phase_id not in self._handlers:
            self._handlers[phase_id] = []
        self._handlers[phase_id].append(handler)
        sub_id = f"{phase_id}:{len(self._handlers[phase_id])-1}"
        logger.debug(f"Subscribed handler: {sub_id}")
        return sub_id

    def subscribe_broadcast(self, handler: MessageHandler) -> str:
        """
        Subscribe to all broadcast messages.

        Args:
            handler: Async handler function

        Returns:
            Subscription ID
        """
        self._broadcast_handlers.append(handler)
        sub_id = f"broadcast:{len(self._broadcast_handlers)-1}"
        logger.debug(f"Subscribed broadcast handler: {sub_id}")
        return sub_id

    def unsubscribe(self, phase_id: str, handler: MessageHandler) -> bool:
        """
        Unsubscribe a handler from a phase.

        Args:
            phase_id: Phase identifier
            handler: Handler to remove

        Returns:
            True if handler was found and removed
        """
        if phase_id in self._handlers:
            try:
                self._handlers[phase_id].remove(handler)
                return True
            except ValueError:
                pass
        return False

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    async def request_phase(
        self,
        from_phase: str,
        to_phase: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Convenience: Send a REQUEST message to a phase.

        Args:
            from_phase: Source phase
            to_phase: Target phase
            payload: Message payload
            priority: Message priority
            correlation_id: Workflow correlation ID
            ttl_seconds: Message TTL

        Returns:
            Message ID
        """
        message = PhaseMessage(
            from_phase=from_phase,
            to_phase=to_phase,
            type=MessageType.REQUEST,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
            ttl_seconds=ttl_seconds,
        )
        return await self.send(message)

    async def respond(
        self,
        original: PhaseMessage,
        payload: Dict[str, Any],
        priority: Optional[MessagePriority] = None
    ) -> str:
        """
        Convenience: Send a RESPONSE to a previous message.

        Args:
            original: Original message being responded to
            payload: Response payload
            priority: Response priority (defaults to original's priority)

        Returns:
            Message ID
        """
        message = PhaseMessage(
            from_phase=original.to_phase,
            to_phase=original.from_phase,
            type=MessageType.RESPONSE,
            priority=priority or original.priority,
            payload=payload,
            correlation_id=original.correlation_id or original.id,
            causation_id=original.id,
        )
        return await self.send(message)

    async def handoff(
        self,
        from_phase: str,
        to_phase: str,
        context: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Convenience: Send context handoff between phases.

        Args:
            from_phase: Source phase
            to_phase: Target phase
            context: Context data to hand off
            correlation_id: Workflow correlation ID

        Returns:
            Message ID
        """
        message = PhaseMessage(
            from_phase=from_phase,
            to_phase=to_phase,
            type=MessageType.HANDOFF,
            priority=MessagePriority.HIGH,
            payload={"context": context},
            correlation_id=correlation_id,
        )
        return await self.send(message)

    async def broadcast(
        self,
        from_phase: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Convenience: Broadcast message to all subscribers.

        Args:
            from_phase: Source phase
            payload: Broadcast payload
            priority: Message priority
            correlation_id: Workflow correlation ID

        Returns:
            Message ID
        """
        message = PhaseMessage(
            from_phase=from_phase,
            to_phase="",  # Empty = broadcast
            type=MessageType.BROADCAST,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
        )
        return await self.send(message)

    async def control(
        self,
        to_phase: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Convenience: Send control message (pause, resume, cancel).

        Args:
            to_phase: Target phase (or empty for broadcast)
            action: Control action (pause, resume, cancel)
            details: Additional details
            correlation_id: Workflow correlation ID

        Returns:
            Message ID
        """
        message = PhaseMessage(
            from_phase="coordinator",
            to_phase=to_phase,
            type=MessageType.CONTROL,
            priority=MessagePriority.CRITICAL,
            payload={"action": action, "details": details or {}},
            correlation_id=correlation_id,
        )
        return await self.send(message)

    # ========================================================================
    # Query Methods
    # ========================================================================

    def get_history(
        self,
        correlation_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        from_phase: Optional[str] = None,
        to_phase: Optional[str] = None,
        limit: int = 100
    ) -> List[PhaseMessage]:
        """
        Get message history with optional filters.

        Args:
            correlation_id: Filter by correlation ID
            message_type: Filter by message type
            from_phase: Filter by source phase
            to_phase: Filter by target phase
            limit: Maximum messages to return

        Returns:
            List of matching messages (newest first)
        """
        result = self._message_history.copy()

        if correlation_id:
            result = [m for m in result if m.correlation_id == correlation_id]

        if message_type:
            result = [m for m in result if m.type == message_type]

        if from_phase:
            result = [m for m in result if m.from_phase == from_phase]

        if to_phase:
            result = [m for m in result if m.to_phase == to_phase]

        # Return newest first, limited
        return list(reversed(result[-limit:]))

    def get_pending_count(self) -> int:
        """Get number of messages pending in queue."""
        return len(self._queue)

    def get_handler_count(self, phase_id: Optional[str] = None) -> int:
        """
        Get number of registered handlers.

        Args:
            phase_id: Specific phase (None = total count)

        Returns:
            Handler count
        """
        if phase_id:
            return len(self._handlers.get(phase_id, []))
        total = sum(len(handlers) for handlers in self._handlers.values())
        total += len(self._broadcast_handlers)
        return total

    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        type_counts = {}
        priority_counts = {}

        for msg in self._message_history:
            type_counts[msg.type.value] = type_counts.get(msg.type.value, 0) + 1
            priority_counts[msg.priority.value] = priority_counts.get(msg.priority.value, 0) + 1

        return {
            "total_messages": len(self._message_history),
            "pending_messages": len(self._queue),
            "handlers_registered": self.get_handler_count(),
            "by_type": type_counts,
            "by_priority": priority_counts,
            "phases_subscribed": list(self._handlers.keys()),
        }

    def clear_history(self) -> int:
        """
        Clear message history.

        Returns:
            Number of messages cleared
        """
        count = len(self._message_history)
        self._message_history = []
        self._processed_ids.clear()
        return count


# ============================================================================
# Factory Function
# ============================================================================

def create_message_bus(history_limit: int = 1000) -> RitualMessageBus:
    """
    Create a new message bus instance.

    Args:
        history_limit: Maximum messages to keep in history

    Returns:
        Configured RitualMessageBus
    """
    return RitualMessageBus(history_limit=history_limit)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'MessageType',
    'MessagePriority',

    # Data structures
    'PhaseMessage',

    # Type aliases
    'MessageHandler',

    # Main class
    'RitualMessageBus',

    # Factory
    'create_message_bus',
]