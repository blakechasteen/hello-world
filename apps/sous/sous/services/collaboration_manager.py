#!/usr/bin/env python3
"""
Collaboration Manager - Real-Time Multi-User Synchronization

Provides:
- Real-time task updates across devices
- Conflict detection and resolution
- Presence tracking (who's online/active)
- Activity feed (recent actions)
- Operational transformation for concurrent edits
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable
from enum import Enum


class EventType(Enum):
    """Types of collaborative events"""
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_UPDATED = "task_updated"
    TASK_REASSIGNED = "task_reassigned"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    PROGRESS_UPDATE = "progress_update"
    CONFLICT_DETECTED = "conflict_detected"
    MESSAGE = "message"


class PresenceStatus(Enum):
    """User presence states"""
    ONLINE = "online"
    ACTIVE = "active"  # Currently cooking
    IDLE = "idle"
    OFFLINE = "offline"


@dataclass
class CollaborativeEvent:
    """Event in the collaboration stream"""
    event_id: str
    event_type: EventType
    user_id: str
    household_id: str
    timestamp: datetime
    data: Dict

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "household_id": self.household_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CollaborativeEvent':
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            user_id=data["user_id"],
            household_id=data["household_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {})
        )


@dataclass
class UserPresence:
    """User presence information"""
    user_id: str
    status: PresenceStatus
    last_seen: datetime
    current_task_id: Optional[str] = None
    device_id: Optional[str] = None

    def is_active(self) -> bool:
        """Check if user is currently active (seen in last 5 minutes)"""
        return (datetime.now() - self.last_seen).total_seconds() < 300

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "status": self.status.value,
            "last_seen": self.last_seen.isoformat(),
            "current_task_id": self.current_task_id,
            "device_id": self.device_id,
            "is_active": self.is_active()
        }


@dataclass
class ConflictResolution:
    """Resolution for a detected conflict"""
    conflict_id: str
    conflict_type: str
    description: str
    resolution_strategy: str  # "last_write_wins", "manual", "merge"
    resolved_by: Optional[str] = None  # user_id
    resolved_at: Optional[datetime] = None


class CollaborationManager:
    """
    Manages real-time collaboration for multi-user kitchen coordination.

    Features:
    - Event broadcasting to all household members
    - Presence tracking (who's online and what they're doing)
    - Conflict detection (concurrent edits)
    - Activity feed (recent actions)
    - WebSocket-ready (can integrate with web/mobile apps)
    """

    def __init__(self, household_id: str):
        self.household_id = household_id

        # Event stream
        self.events: List[CollaborativeEvent] = []
        self.event_counter = 0

        # Presence tracking
        self.presence: Dict[str, UserPresence] = {}

        # Subscribers (for real-time updates)
        self.subscribers: Dict[str, List[Callable]] = {}  # user_id -> [callback functions]

        # Conflict tracking
        self.conflicts: Dict[str, ConflictResolution] = {}

        # Activity feed (recent events)
        self.activity_feed_size = 100

    async def emit_event(
        self,
        event_type: EventType,
        user_id: str,
        data: Dict
    ) -> CollaborativeEvent:
        """
        Emit an event to all subscribers.

        Broadcasts event to all household members in real-time.
        """
        event = CollaborativeEvent(
            event_id=f"event_{self.event_counter}",
            event_type=event_type,
            user_id=user_id,
            household_id=self.household_id,
            timestamp=datetime.now(),
            data=data
        )
        self.event_counter += 1

        # Store event
        self.events.append(event)

        # Trim event history to last N events
        if len(self.events) > self.activity_feed_size:
            self.events = self.events[-self.activity_feed_size:]

        # Update user presence
        if user_id in self.presence:
            self.presence[user_id].last_seen = datetime.now()
            self.presence[user_id].status = PresenceStatus.ACTIVE

        # Broadcast to subscribers
        await self._broadcast_event(event)

        return event

    async def _broadcast_event(self, event: CollaborativeEvent):
        """Broadcast event to all subscribers."""
        # In production, this would use WebSocket or similar real-time protocol
        for user_id, callbacks in self.subscribers.items():
            for callback in callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    print(f"Error broadcasting to {user_id}: {e}")

    def subscribe(self, user_id: str, callback: Callable):
        """Subscribe a user to event stream."""
        if user_id not in self.subscribers:
            self.subscribers[user_id] = []
        self.subscribers[user_id].append(callback)

    def unsubscribe(self, user_id: str, callback: Callable):
        """Unsubscribe from event stream."""
        if user_id in self.subscribers and callback in self.subscribers[user_id]:
            self.subscribers[user_id].remove(callback)

    async def update_presence(
        self,
        user_id: str,
        status: PresenceStatus,
        current_task_id: Optional[str] = None,
        device_id: Optional[str] = None
    ):
        """Update user presence information."""
        if user_id not in self.presence:
            self.presence[user_id] = UserPresence(
                user_id=user_id,
                status=status,
                last_seen=datetime.now(),
                current_task_id=current_task_id,
                device_id=device_id
            )
        else:
            self.presence[user_id].status = status
            self.presence[user_id].last_seen = datetime.now()
            self.presence[user_id].current_task_id = current_task_id
            if device_id:
                self.presence[user_id].device_id = device_id

        # Emit presence update event
        await self.emit_event(
            EventType.USER_JOINED if status == PresenceStatus.ONLINE else EventType.USER_LEFT,
            user_id,
            {"status": status.value, "task_id": current_task_id}
        )

    def get_active_users(self) -> List[UserPresence]:
        """Get list of currently active users."""
        return [
            presence for presence in self.presence.values()
            if presence.is_active()
        ]

    def get_user_presence(self, user_id: str) -> Optional[UserPresence]:
        """Get presence info for specific user."""
        return self.presence.get(user_id)

    async def detect_conflict(
        self,
        task_id: str,
        user1_id: str,
        user2_id: str,
        description: str
    ) -> str:
        """
        Detect and record a conflict.

        Returns conflict_id for resolution tracking.
        """
        conflict_id = f"conflict_{len(self.conflicts)}"

        conflict = ConflictResolution(
            conflict_id=conflict_id,
            conflict_type="concurrent_edit",
            description=description,
            resolution_strategy="last_write_wins"  # Default strategy
        )

        self.conflicts[conflict_id] = conflict

        # Emit conflict event
        await self.emit_event(
            EventType.CONFLICT_DETECTED,
            user1_id,
            {
                "conflict_id": conflict_id,
                "task_id": task_id,
                "other_user": user2_id,
                "description": description
            }
        )

        return conflict_id

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolved_by: str,
        resolution_data: Dict
    ):
        """Resolve a conflict."""
        if conflict_id not in self.conflicts:
            return

        conflict = self.conflicts[conflict_id]
        conflict.resolved_by = resolved_by
        conflict.resolved_at = datetime.now()

        await self.emit_event(
            EventType.TASK_UPDATED,
            resolved_by,
            {
                "conflict_id": conflict_id,
                "resolution": resolution_data
            }
        )

    def get_activity_feed(
        self,
        limit: int = 20,
        user_id: Optional[str] = None
    ) -> List[CollaborativeEvent]:
        """
        Get recent activity feed.

        Args:
            limit: Maximum number of events to return
            user_id: Filter by specific user (optional)
        """
        events = self.events

        if user_id:
            events = [e for e in events if e.user_id == user_id]

        return list(reversed(events[-limit:]))

    def get_events_since(self, since: datetime) -> List[CollaborativeEvent]:
        """Get all events since a specific timestamp."""
        return [e for e in self.events if e.timestamp > since]

    async def send_message(
        self,
        from_user_id: str,
        message: str,
        to_user_id: Optional[str] = None
    ):
        """
        Send a message (chat-like feature for kitchen coordination).

        Args:
            from_user_id: Sender
            message: Text message
            to_user_id: Recipient (None = broadcast to all)
        """
        await self.emit_event(
            EventType.MESSAGE,
            from_user_id,
            {
                "message": message,
                "to_user": to_user_id
            }
        )

    def get_statistics(self) -> Dict:
        """Get collaboration statistics."""
        total_events = len(self.events)
        active_users = len(self.get_active_users())

        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": total_events,
            "active_users": active_users,
            "total_users": len(self.presence),
            "event_breakdown": event_counts,
            "conflicts_detected": len(self.conflicts),
            "conflicts_resolved": sum(1 for c in self.conflicts.values() if c.resolved_at),
            "subscribers": len(self.subscribers)
        }

    def to_dict(self) -> dict:
        """Serialize collaboration state."""
        return {
            "household_id": self.household_id,
            "events": [e.to_dict() for e in self.events[-50:]],  # Last 50 events
            "presence": {uid: p.to_dict() for uid, p in self.presence.items()},
            "statistics": self.get_statistics()
        }


class OperationalTransform:
    """
    Operational Transformation for resolving concurrent edits.

    When two users edit the same task simultaneously, OT ensures
    both edits are applied correctly without conflicts.
    """

    @staticmethod
    def transform_operations(op1: Dict, op2: Dict) -> tuple:
        """
        Transform two concurrent operations.

        Returns (op1', op2') where:
        - op1' is op1 transformed against op2
        - op2' is op2 transformed against op1

        This is a simplified OT implementation. Production systems
        would use more sophisticated algorithms (e.g., Google Docs OT).
        """
        # Example: Both users update task progress
        if op1.get("field") == "progress" and op2.get("field") == "progress":
            # Take maximum progress
            max_progress = max(op1.get("value", 0), op2.get("value", 0))
            op1["value"] = max_progress
            op2["value"] = max_progress

        # Example: Both users add notes
        elif op1.get("field") == "notes" and op2.get("field") == "notes":
            # Merge notes
            merged_notes = f"{op1.get('value', '')} | {op2.get('value', '')}"
            op1["value"] = merged_notes
            op2["value"] = merged_notes

        return (op1, op2)

    @staticmethod
    def apply_operation(task_data: Dict, operation: Dict) -> Dict:
        """Apply an operation to task data."""
        field = operation.get("field")
        value = operation.get("value")

        if field and value is not None:
            task_data[field] = value

        return task_data


class PresenceHeartbeat:
    """
    Heartbeat mechanism for presence tracking.

    Automatically updates user presence and detects disconnections.
    """

    def __init__(
        self,
        collaboration_manager: CollaborationManager,
        heartbeat_interval: int = 30  # seconds
    ):
        self.manager = collaboration_manager
        self.interval = heartbeat_interval
        self.running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start heartbeat monitoring."""
        self.running = True
        self._task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop heartbeat monitoring."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self):
        """Heartbeat loop that checks for stale presence."""
        while self.running:
            await asyncio.sleep(self.interval)

            # Mark inactive users as idle
            for user_id, presence in self.manager.presence.items():
                time_since_seen = (datetime.now() - presence.last_seen).total_seconds()

                if time_since_seen > 300:  # 5 minutes
                    if presence.status != PresenceStatus.OFFLINE:
                        presence.status = PresenceStatus.IDLE

                if time_since_seen > 600:  # 10 minutes
                    presence.status = PresenceStatus.OFFLINE


# Helper functions for common collaboration patterns

async def create_collaboration_session(
    household_id: str,
    enable_heartbeat: bool = True
) -> tuple:
    """
    Create a collaboration session.

    Returns (manager, heartbeat) tuple.
    """
    manager = CollaborationManager(household_id)

    heartbeat = None
    if enable_heartbeat:
        heartbeat = PresenceHeartbeat(manager)
        await heartbeat.start()

    return manager, heartbeat


async def join_collaboration(
    manager: CollaborationManager,
    user_id: str,
    device_id: Optional[str] = None
):
    """User joins collaboration session."""
    await manager.update_presence(
        user_id,
        PresenceStatus.ONLINE,
        device_id=device_id
    )


async def leave_collaboration(
    manager: CollaborationManager,
    user_id: str
):
    """User leaves collaboration session."""
    await manager.update_presence(
        user_id,
        PresenceStatus.OFFLINE
    )
