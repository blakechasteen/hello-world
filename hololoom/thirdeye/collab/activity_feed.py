"""
Activity Feed System
====================

Real-time activity tracking for collaborative ThirdEye scenes.

Features:
- Track all scene operations (add, update, delete)
- Track presence changes (join, leave, focus)
- Track role and style changes
- Filtering by type, user, time range
- Aggregation of similar events
- Notification subscriptions
- Activity persistence and retrieval

Integration:
- Listens to SceneOperation events from scene_sync
- Listens to StyleChangeEvent from shared_styles
- Listens to RoleChangeEvent from scene_roles
- Listens to presence updates from scene_presence

Created: 2025-12-03
Author: HoloLoom Team
"""

import logging
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Activity Types and Priority
# -----------------------------------------------------------------------------

class ActivityType(Enum):
    """Types of activities tracked in the feed."""
    # Scene operations
    ELEMENT_ADDED = "element_added"
    ELEMENT_UPDATED = "element_updated"
    ELEMENT_DELETED = "element_deleted"
    ELEMENT_MOVED = "element_moved"
    CONNECTION_ADDED = "connection_added"
    CONNECTION_DELETED = "connection_deleted"

    # Presence
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    USER_FOCUS_CHANGED = "user_focus_changed"

    # Roles
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"

    # Styles
    STYLE_CREATED = "style_created"
    STYLE_PROPOSED = "style_proposed"
    STYLE_VOTED = "style_voted"
    STYLE_APPROVED = "style_approved"
    STYLE_REJECTED = "style_rejected"
    STYLE_USED = "style_used"

    # Comments and reactions
    COMMENT_ADDED = "comment_added"
    REACTION_ADDED = "reaction_added"

    # Scene-level
    SCENE_CREATED = "scene_created"
    SCENE_SETTINGS_CHANGED = "scene_settings_changed"
    SCENE_EXPORTED = "scene_exported"

    # System
    SYSTEM_MESSAGE = "system_message"


class ActivityPriority(Enum):
    """Priority levels for activities (affects notification behavior)."""
    LOW = "low"           # Background info, no notification
    NORMAL = "normal"     # Standard activity, optional notification
    HIGH = "high"         # Important, should notify
    URGENT = "urgent"     # Critical, always notify


# Activity type to default priority mapping
ACTIVITY_PRIORITIES: dict[ActivityType, ActivityPriority] = {
    # Scene operations
    ActivityType.ELEMENT_ADDED: ActivityPriority.NORMAL,
    ActivityType.ELEMENT_UPDATED: ActivityPriority.LOW,
    ActivityType.ELEMENT_DELETED: ActivityPriority.NORMAL,
    ActivityType.ELEMENT_MOVED: ActivityPriority.LOW,
    ActivityType.CONNECTION_ADDED: ActivityPriority.LOW,
    ActivityType.CONNECTION_DELETED: ActivityPriority.LOW,

    # Presence
    ActivityType.USER_JOINED: ActivityPriority.HIGH,
    ActivityType.USER_LEFT: ActivityPriority.NORMAL,
    ActivityType.USER_FOCUS_CHANGED: ActivityPriority.LOW,

    # Roles
    ActivityType.ROLE_ASSIGNED: ActivityPriority.HIGH,
    ActivityType.ROLE_REVOKED: ActivityPriority.HIGH,

    # Styles
    ActivityType.STYLE_CREATED: ActivityPriority.LOW,
    ActivityType.STYLE_PROPOSED: ActivityPriority.NORMAL,
    ActivityType.STYLE_VOTED: ActivityPriority.LOW,
    ActivityType.STYLE_APPROVED: ActivityPriority.HIGH,
    ActivityType.STYLE_REJECTED: ActivityPriority.NORMAL,
    ActivityType.STYLE_USED: ActivityPriority.LOW,

    # Comments
    ActivityType.COMMENT_ADDED: ActivityPriority.NORMAL,
    ActivityType.REACTION_ADDED: ActivityPriority.LOW,

    # Scene-level
    ActivityType.SCENE_CREATED: ActivityPriority.HIGH,
    ActivityType.SCENE_SETTINGS_CHANGED: ActivityPriority.NORMAL,
    ActivityType.SCENE_EXPORTED: ActivityPriority.LOW,

    # System
    ActivityType.SYSTEM_MESSAGE: ActivityPriority.HIGH,
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class ActivityEntry:
    """
    A single activity entry in the feed.
    """
    id: str
    activity_type: ActivityType
    user_id: str
    timestamp: datetime
    priority: ActivityPriority

    # Details
    summary: str  # Human-readable summary
    details: dict[str, Any] = field(default_factory=dict)

    # Related entities
    element_id: str | None = None
    target_user_id: str | None = None  # For role assignments, etc.
    style_id: str | None = None

    # Aggregation
    aggregation_key: str | None = None  # For grouping similar activities
    aggregation_count: int = 1  # Count of aggregated activities

    # Read tracking
    read_by: set[str] = field(default_factory=set)

    def mark_read(self, user_id: str):
        """Mark this activity as read by a user."""
        self.read_by.add(user_id)

    def is_read_by(self, user_id: str) -> bool:
        """Check if user has read this activity."""
        return user_id in self.read_by

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'activity_type': self.activity_type.value,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'summary': self.summary,
            'details': self.details,
            'element_id': self.element_id,
            'target_user_id': self.target_user_id,
            'style_id': self.style_id,
            'aggregation_key': self.aggregation_key,
            'aggregation_count': self.aggregation_count,
            'read_by': list(self.read_by),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ActivityEntry':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            activity_type=ActivityType(data['activity_type']),
            user_id=data['user_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=ActivityPriority(data['priority']),
            summary=data['summary'],
            details=data.get('details', {}),
            element_id=data.get('element_id'),
            target_user_id=data.get('target_user_id'),
            style_id=data.get('style_id'),
            aggregation_key=data.get('aggregation_key'),
            aggregation_count=data.get('aggregation_count', 1),
            read_by=set(data.get('read_by', [])),
        )


@dataclass
class ActivityFilter:
    """
    Filter criteria for querying activities.
    """
    activity_types: set[ActivityType] | None = None
    user_ids: set[str] | None = None
    priority_min: ActivityPriority | None = None
    element_ids: set[str] | None = None
    style_ids: set[str] | None = None
    since: datetime | None = None
    until: datetime | None = None
    unread_by: str | None = None  # Filter to unread by this user

    def matches(self, entry: ActivityEntry) -> bool:
        """Check if an entry matches this filter."""
        # Type filter
        if self.activity_types and entry.activity_type not in self.activity_types:
            return False

        # User filter
        if self.user_ids and entry.user_id not in self.user_ids:
            return False

        # Priority filter
        if self.priority_min:
            priority_order = [
                ActivityPriority.LOW,
                ActivityPriority.NORMAL,
                ActivityPriority.HIGH,
                ActivityPriority.URGENT,
            ]
            if priority_order.index(entry.priority) < priority_order.index(self.priority_min):
                return False

        # Element filter
        if self.element_ids and entry.element_id not in self.element_ids:
            return False

        # Style filter
        if self.style_ids and entry.style_id not in self.style_ids:
            return False

        # Time filters
        if self.since and entry.timestamp < self.since:
            return False
        if self.until and entry.timestamp > self.until:
            return False

        # Unread filter
        if self.unread_by and self.unread_by in entry.read_by:
            return False

        return True


@dataclass
class NotificationSubscription:
    """A subscription to activity notifications."""
    subscriber_id: str
    activity_types: set[ActivityType]
    priority_min: ActivityPriority = ActivityPriority.NORMAL
    callback: Callable[[ActivityEntry], None] | None = None
    enabled: bool = True


# -----------------------------------------------------------------------------
# Activity Feed Manager
# -----------------------------------------------------------------------------

class ActivityFeed:
    """
    Manages activity feed for a collaborative scene.

    Usage:
        feed = ActivityFeed(scene_id="scene_123")

        # Record activity
        feed.record_activity(
            activity_type=ActivityType.ELEMENT_ADDED,
            user_id="user_1",
            summary="User 1 added a box element",
            element_id="element_123"
        )

        # Query activities
        recent = feed.get_recent(limit=20)

        # Subscribe to notifications
        feed.subscribe(
            subscriber_id="user_2",
            activity_types={ActivityType.ELEMENT_ADDED, ActivityType.USER_JOINED}
        )
    """

    # Priority order for comparison
    PRIORITY_ORDER = [
        ActivityPriority.LOW,
        ActivityPriority.NORMAL,
        ActivityPriority.HIGH,
        ActivityPriority.URGENT,
    ]

    def __init__(
        self,
        scene_id: str,
        max_entries: int = 1000,
        aggregation_window_seconds: float = 60.0,
    ):
        """
        Initialize activity feed.

        Args:
            scene_id: ID of the scene
            max_entries: Maximum entries to keep (older ones are pruned)
            aggregation_window_seconds: Time window for aggregating similar activities
        """
        self.scene_id = scene_id
        self.max_entries = max_entries
        self.aggregation_window = timedelta(seconds=aggregation_window_seconds)

        self._entries: list[ActivityEntry] = []
        self._subscriptions: dict[str, NotificationSubscription] = {}
        self._user_display_names: dict[str, str] = {}  # Cache of user display names

    # -------------------------------------------------------------------------
    # Recording Activities
    # -------------------------------------------------------------------------

    def record_activity(
        self,
        activity_type: ActivityType,
        user_id: str,
        summary: str,
        details: dict[str, Any] | None = None,
        element_id: str | None = None,
        target_user_id: str | None = None,
        style_id: str | None = None,
        priority: ActivityPriority | None = None,
        allow_aggregation: bool = True,
    ) -> ActivityEntry:
        """
        Record a new activity.

        Args:
            activity_type: Type of activity
            user_id: User who performed the action
            summary: Human-readable summary
            details: Additional details
            element_id: Related element ID
            target_user_id: Target user (for role changes, etc.)
            style_id: Related style ID
            priority: Override default priority
            allow_aggregation: Whether to aggregate with recent similar activities

        Returns:
            The created (or updated) activity entry
        """
        # Determine priority
        if priority is None:
            priority = ACTIVITY_PRIORITIES.get(activity_type, ActivityPriority.NORMAL)

        # Generate aggregation key for similar activities
        aggregation_key = None
        if allow_aggregation:
            aggregation_key = f"{user_id}:{activity_type.value}:{element_id or ''}"

        # Check for aggregation with recent activity
        if allow_aggregation and aggregation_key:
            existing = self._find_aggregatable(aggregation_key)
            if existing:
                existing.aggregation_count += 1
                existing.timestamp = datetime.now()  # Update timestamp
                existing.summary = self._make_aggregated_summary(
                    activity_type, user_id, existing.aggregation_count
                )
                if details:
                    existing.details.update(details)

                logger.debug(f"Aggregated activity: {aggregation_key} (count: {existing.aggregation_count})")

                # Still notify for aggregated activities
                self._notify_subscribers(existing)

                return existing

        # Create new entry
        entry = ActivityEntry(
            id=f"activity_{uuid.uuid4().hex[:12]}",
            activity_type=activity_type,
            user_id=user_id,
            timestamp=datetime.now(),
            priority=priority,
            summary=summary,
            details=details or {},
            element_id=element_id,
            target_user_id=target_user_id,
            style_id=style_id,
            aggregation_key=aggregation_key,
        )

        self._entries.append(entry)
        self._prune_old_entries()

        logger.debug(f"Recorded activity: {activity_type.value} by {user_id}")

        # Notify subscribers
        self._notify_subscribers(entry)

        return entry

    def _find_aggregatable(self, aggregation_key: str) -> ActivityEntry | None:
        """Find a recent entry that can be aggregated with."""
        cutoff = datetime.now() - self.aggregation_window

        for entry in reversed(self._entries):
            if entry.timestamp < cutoff:
                break
            if entry.aggregation_key == aggregation_key:
                return entry

        return None

    def _make_aggregated_summary(
        self,
        activity_type: ActivityType,
        user_id: str,
        count: int,
    ) -> str:
        """Generate summary for aggregated activities."""
        user_name = self._get_user_display_name(user_id)

        summaries = {
            ActivityType.ELEMENT_UPDATED: f"{user_name} made {count} edits",
            ActivityType.ELEMENT_MOVED: f"{user_name} moved elements {count} times",
            ActivityType.USER_FOCUS_CHANGED: f"{user_name} changed focus {count} times",
            ActivityType.STYLE_USED: f"{user_name} applied styles {count} times",
        }

        return summaries.get(activity_type, f"{user_name} performed {count} actions")

    def _prune_old_entries(self):
        """Remove oldest entries if over max."""
        if len(self._entries) > self.max_entries:
            prune_count = len(self._entries) - self.max_entries
            self._entries = self._entries[prune_count:]
            logger.debug(f"Pruned {prune_count} old activity entries")

    def _get_user_display_name(self, user_id: str) -> str:
        """Get display name for a user."""
        return self._user_display_names.get(user_id, user_id)

    def set_user_display_name(self, user_id: str, display_name: str):
        """Set display name for a user."""
        self._user_display_names[user_id] = display_name

    # -------------------------------------------------------------------------
    # Convenience Recording Methods
    # -------------------------------------------------------------------------

    def record_element_added(
        self,
        user_id: str,
        element_id: str,
        element_type: str,
    ) -> ActivityEntry:
        """Record element addition."""
        user_name = self._get_user_display_name(user_id)
        return self.record_activity(
            activity_type=ActivityType.ELEMENT_ADDED,
            user_id=user_id,
            summary=f"{user_name} added a {element_type}",
            element_id=element_id,
            details={"element_type": element_type},
        )

    def record_element_updated(
        self,
        user_id: str,
        element_id: str,
        changes: dict[str, Any] | None = None,
    ) -> ActivityEntry:
        """Record element update."""
        user_name = self._get_user_display_name(user_id)
        return self.record_activity(
            activity_type=ActivityType.ELEMENT_UPDATED,
            user_id=user_id,
            summary=f"{user_name} edited an element",
            element_id=element_id,
            details={"changes": changes} if changes else {},
            allow_aggregation=True,
        )

    def record_element_deleted(
        self,
        user_id: str,
        element_id: str,
        element_type: str,
    ) -> ActivityEntry:
        """Record element deletion."""
        user_name = self._get_user_display_name(user_id)
        return self.record_activity(
            activity_type=ActivityType.ELEMENT_DELETED,
            user_id=user_id,
            summary=f"{user_name} deleted a {element_type}",
            element_id=element_id,
            details={"element_type": element_type},
        )

    def record_user_joined(self, user_id: str) -> ActivityEntry:
        """Record user joining the scene."""
        user_name = self._get_user_display_name(user_id)
        return self.record_activity(
            activity_type=ActivityType.USER_JOINED,
            user_id=user_id,
            summary=f"{user_name} joined the scene",
            allow_aggregation=False,
        )

    def record_user_left(self, user_id: str) -> ActivityEntry:
        """Record user leaving the scene."""
        user_name = self._get_user_display_name(user_id)
        return self.record_activity(
            activity_type=ActivityType.USER_LEFT,
            user_id=user_id,
            summary=f"{user_name} left the scene",
            allow_aggregation=False,
        )

    def record_role_change(
        self,
        user_id: str,
        target_user_id: str,
        new_role: str,
        previous_role: str | None = None,
    ) -> ActivityEntry:
        """Record role assignment/change."""
        user_name = self._get_user_display_name(user_id)
        target_name = self._get_user_display_name(target_user_id)

        if previous_role:
            summary = f"{user_name} changed {target_name}'s role from {previous_role} to {new_role}"
        else:
            summary = f"{user_name} assigned {new_role} role to {target_name}"

        return self.record_activity(
            activity_type=ActivityType.ROLE_ASSIGNED,
            user_id=user_id,
            target_user_id=target_user_id,
            summary=summary,
            details={"new_role": new_role, "previous_role": previous_role},
            allow_aggregation=False,
        )

    def record_style_proposed(
        self,
        user_id: str,
        style_id: str,
        style_name: str,
    ) -> ActivityEntry:
        """Record style proposal."""
        user_name = self._get_user_display_name(user_id)
        return self.record_activity(
            activity_type=ActivityType.STYLE_PROPOSED,
            user_id=user_id,
            style_id=style_id,
            summary=f"{user_name} proposed style '{style_name}' for voting",
            details={"style_name": style_name},
        )

    def record_style_vote(
        self,
        user_id: str,
        style_id: str,
        style_name: str,
        vote_type: str,
    ) -> ActivityEntry:
        """Record style vote."""
        user_name = self._get_user_display_name(user_id)
        return self.record_activity(
            activity_type=ActivityType.STYLE_VOTED,
            user_id=user_id,
            style_id=style_id,
            summary=f"{user_name} voted {vote_type} on '{style_name}'",
            details={"style_name": style_name, "vote_type": vote_type},
        )

    def record_style_approved(
        self,
        user_id: str,
        style_id: str,
        style_name: str,
        approval_type: str = "auto",
    ) -> ActivityEntry:
        """Record style approval."""
        if approval_type == "auto":
            summary = f"Style '{style_name}' was approved by community vote"
        else:
            user_name = self._get_user_display_name(user_id)
            summary = f"{user_name} approved style '{style_name}'"

        return self.record_activity(
            activity_type=ActivityType.STYLE_APPROVED,
            user_id=user_id,
            style_id=style_id,
            summary=summary,
            details={"style_name": style_name, "approval_type": approval_type},
        )

    def record_comment(
        self,
        user_id: str,
        comment_text: str,
        element_id: str | None = None,
    ) -> ActivityEntry:
        """Record a comment."""
        user_name = self._get_user_display_name(user_id)
        preview = comment_text[:50] + "..." if len(comment_text) > 50 else comment_text

        if element_id:
            summary = f"{user_name} commented on an element: \"{preview}\""
        else:
            summary = f"{user_name} commented: \"{preview}\""

        return self.record_activity(
            activity_type=ActivityType.COMMENT_ADDED,
            user_id=user_id,
            element_id=element_id,
            summary=summary,
            details={"comment": comment_text},
        )

    def record_system_message(
        self,
        message: str,
        priority: ActivityPriority = ActivityPriority.HIGH,
    ) -> ActivityEntry:
        """Record a system message."""
        return self.record_activity(
            activity_type=ActivityType.SYSTEM_MESSAGE,
            user_id="system",
            summary=message,
            priority=priority,
            allow_aggregation=False,
        )

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def get_recent(
        self,
        limit: int = 50,
        filter: ActivityFilter | None = None,
    ) -> list[ActivityEntry]:
        """
        Get recent activities.

        Args:
            limit: Maximum entries to return
            filter: Optional filter criteria

        Returns:
            List of activities (newest first)
        """
        entries = reversed(self._entries)

        if filter:
            entries = [e for e in entries if filter.matches(e)]
        else:
            entries = list(entries)

        return entries[:limit]

    def get_unread(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[ActivityEntry]:
        """Get unread activities for a user."""
        filter = ActivityFilter(unread_by=user_id)
        return self.get_recent(limit=limit, filter=filter)

    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread activities for a user."""
        return sum(1 for e in self._entries if not e.is_read_by(user_id))

    def get_by_element(
        self,
        element_id: str,
        limit: int = 50,
    ) -> list[ActivityEntry]:
        """Get activities related to a specific element."""
        filter = ActivityFilter(element_ids={element_id})
        return self.get_recent(limit=limit, filter=filter)

    def get_by_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[ActivityEntry]:
        """Get activities by a specific user."""
        filter = ActivityFilter(user_ids={user_id})
        return self.get_recent(limit=limit, filter=filter)

    def get_high_priority(
        self,
        limit: int = 20,
    ) -> list[ActivityEntry]:
        """Get high priority activities."""
        filter = ActivityFilter(priority_min=ActivityPriority.HIGH)
        return self.get_recent(limit=limit, filter=filter)

    def get_activity_summary(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics of activities.

        Args:
            since: Only count activities since this time

        Returns:
            Summary statistics
        """
        if since is None:
            since = datetime.now() - timedelta(hours=24)

        filter = ActivityFilter(since=since)
        entries = [e for e in self._entries if filter.matches(e)]

        # Count by type
        by_type: dict[str, int] = defaultdict(int)
        for entry in entries:
            by_type[entry.activity_type.value] += 1

        # Count by user
        by_user: dict[str, int] = defaultdict(int)
        for entry in entries:
            by_user[entry.user_id] += 1

        # Active users
        active_users = set(e.user_id for e in entries if e.user_id != "system")

        return {
            'total_activities': len(entries),
            'by_type': dict(by_type),
            'by_user': dict(by_user),
            'active_users': len(active_users),
            'since': since.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Read Tracking
    # -------------------------------------------------------------------------

    def mark_read(
        self,
        user_id: str,
        entry_ids: list[str] | None = None,
    ):
        """
        Mark activities as read.

        Args:
            user_id: User marking as read
            entry_ids: Specific entries to mark (or None for all)
        """
        if entry_ids:
            for entry in self._entries:
                if entry.id in entry_ids:
                    entry.mark_read(user_id)
        else:
            for entry in self._entries:
                entry.mark_read(user_id)

    def mark_all_read(self, user_id: str):
        """Mark all activities as read for a user."""
        self.mark_read(user_id, entry_ids=None)

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    def subscribe(
        self,
        subscriber_id: str,
        activity_types: set[ActivityType] | None = None,
        priority_min: ActivityPriority = ActivityPriority.NORMAL,
        callback: Callable[[ActivityEntry], None] | None = None,
    ) -> NotificationSubscription:
        """
        Subscribe to activity notifications.

        Args:
            subscriber_id: User subscribing
            activity_types: Types to subscribe to (None = all)
            priority_min: Minimum priority to notify
            callback: Callback function for notifications

        Returns:
            The subscription
        """
        if activity_types is None:
            activity_types = set(ActivityType)

        subscription = NotificationSubscription(
            subscriber_id=subscriber_id,
            activity_types=activity_types,
            priority_min=priority_min,
            callback=callback,
        )

        self._subscriptions[subscriber_id] = subscription

        logger.info(f"User {subscriber_id} subscribed to {len(activity_types)} activity types")

        return subscription

    def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from notifications."""
        if subscriber_id in self._subscriptions:
            del self._subscriptions[subscriber_id]
            logger.info(f"User {subscriber_id} unsubscribed")

    def update_subscription(
        self,
        subscriber_id: str,
        activity_types: set[ActivityType] | None = None,
        priority_min: ActivityPriority | None = None,
        enabled: bool | None = None,
    ):
        """Update an existing subscription."""
        sub = self._subscriptions.get(subscriber_id)
        if not sub:
            return

        if activity_types is not None:
            sub.activity_types = activity_types
        if priority_min is not None:
            sub.priority_min = priority_min
        if enabled is not None:
            sub.enabled = enabled

    def _notify_subscribers(self, entry: ActivityEntry):
        """Notify relevant subscribers of a new activity."""
        for subscriber_id, sub in self._subscriptions.items():
            if not sub.enabled:
                continue

            # Don't notify user about their own actions
            if subscriber_id == entry.user_id:
                continue

            # Check if activity type matches
            if entry.activity_type not in sub.activity_types:
                continue

            # Check priority threshold
            if self.PRIORITY_ORDER.index(entry.priority) < self.PRIORITY_ORDER.index(sub.priority_min):
                continue

            # Call callback if provided
            if sub.callback:
                try:
                    sub.callback(entry)
                except Exception as e:
                    logger.error(f"Notification callback error for {subscriber_id}: {e}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'scene_id': self.scene_id,
            'max_entries': self.max_entries,
            'aggregation_window_seconds': self.aggregation_window.total_seconds(),
            'entries': [e.to_dict() for e in self._entries],
            'user_display_names': self._user_display_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ActivityFeed':
        """Deserialize from dictionary."""
        feed = cls(
            scene_id=data['scene_id'],
            max_entries=data.get('max_entries', 1000),
            aggregation_window_seconds=data.get('aggregation_window_seconds', 60.0),
        )

        feed._entries = [
            ActivityEntry.from_dict(e) for e in data.get('entries', [])
        ]
        feed._user_display_names = data.get('user_display_names', {})

        return feed


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def create_activity_feed(
    scene_id: str,
    max_entries: int = 1000,
    aggregation_window_seconds: float = 60.0,
) -> ActivityFeed:
    """
    Create an activity feed for a scene.

    Args:
        scene_id: ID of the scene
        max_entries: Maximum entries to keep
        aggregation_window_seconds: Window for aggregating similar activities

    Returns:
        Configured ActivityFeed
    """
    return ActivityFeed(
        scene_id=scene_id,
        max_entries=max_entries,
        aggregation_window_seconds=aggregation_window_seconds,
    )


__all__ = [
    # Enums
    'ActivityType',
    'ActivityPriority',
    # Data classes
    'ActivityEntry',
    'ActivityFilter',
    'NotificationSubscription',
    # Main class
    'ActivityFeed',
    # Factory
    'create_activity_feed',
    # Constants
    'ACTIVITY_PRIORITIES',
]
