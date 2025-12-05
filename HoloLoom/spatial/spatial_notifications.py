"""
Spatial Notifications and Alerts for HoloLoom Spatial Computing.

3D notification system for XR environments:
- Toast notifications
- Floating alerts
- Badge indicators
- Progress displays
- System messages
- Knowledge node highlights
- Attention guides

Created: November 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
import uuid
import math


class NotificationType(Enum):
    """Notification types."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"
    KNOWLEDGE = "knowledge"
    SOCIAL = "social"
    ACHIEVEMENT = "achievement"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = 0
    NORMAL = 1
    MEDIUM = 2      # Between normal and high
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class NotificationPosition(Enum):
    """Where notification appears."""
    HEAD_LOCKED = "head_locked"       # Follows user view
    WORLD_FIXED = "world_fixed"       # Fixed in world
    NODE_ATTACHED = "node_attached"   # Attached to node
    HAND_NEAR = "hand_near"           # Near user's hand
    PERIPHERAL = "peripheral"         # Edge of vision
    CENTER = "center"                 # Center of view


class AnimationType(Enum):
    """Notification animations."""
    FADE = "fade"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SCALE = "scale"
    BOUNCE = "bounce"
    PULSE = "pulse"
    SHAKE = "shake"
    SPIRAL = "spiral"


@dataclass
class NotificationStyle:
    """Visual style for notifications."""
    background_color: Tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.9)
    text_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    accent_color: Tuple[float, float, float, float] = (0.3, 0.6, 1.0, 1.0)
    border_color: Tuple[float, float, float, float] = (0.4, 0.4, 0.5, 0.5)
    border_width: float = 0.002
    border_radius: float = 0.01
    padding: float = 0.015
    font_size: float = 0.02
    icon_size: float = 0.03
    shadow: bool = True
    glow: bool = False
    glow_color: Optional[Tuple[float, float, float, float]] = None


# Preset styles for notification types
NOTIFICATION_STYLES = {
    NotificationType.INFO: NotificationStyle(
        accent_color=(0.3, 0.6, 1.0, 1.0)
    ),
    NotificationType.SUCCESS: NotificationStyle(
        accent_color=(0.2, 0.8, 0.4, 1.0),
        glow=True,
        glow_color=(0.2, 0.8, 0.4, 0.3)
    ),
    NotificationType.WARNING: NotificationStyle(
        accent_color=(1.0, 0.7, 0.0, 1.0),
        background_color=(0.2, 0.15, 0.05, 0.9)
    ),
    NotificationType.ERROR: NotificationStyle(
        accent_color=(1.0, 0.3, 0.3, 1.0),
        background_color=(0.2, 0.05, 0.05, 0.9),
        glow=True,
        glow_color=(1.0, 0.3, 0.3, 0.3)
    ),
    NotificationType.SYSTEM: NotificationStyle(
        accent_color=(0.5, 0.5, 0.6, 1.0),
        background_color=(0.15, 0.15, 0.2, 0.95)
    ),
    NotificationType.KNOWLEDGE: NotificationStyle(
        accent_color=(0.6, 0.4, 1.0, 1.0),
        glow=True,
        glow_color=(0.6, 0.4, 1.0, 0.2)
    ),
    NotificationType.SOCIAL: NotificationStyle(
        accent_color=(0.2, 0.8, 0.9, 1.0)
    ),
    NotificationType.ACHIEVEMENT: NotificationStyle(
        accent_color=(1.0, 0.8, 0.2, 1.0),
        glow=True,
        glow_color=(1.0, 0.8, 0.2, 0.4),
        background_color=(0.15, 0.12, 0.05, 0.95)
    ),
}


@dataclass
class NotificationAction:
    """Action button in notification."""
    action_id: str
    label: str
    icon: Optional[str] = None
    primary: bool = False
    callback_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """A single notification."""
    notification_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    message: str = ""
    notification_type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.NORMAL

    # Position
    position: NotificationPosition = NotificationPosition.HEAD_LOCKED
    world_position: Optional[Tuple[float, float, float]] = None
    attached_node_id: Optional[str] = None
    offset: Tuple[float, float, float] = (0, 0, 0)

    # Timing
    duration: float = 5.0  # Seconds, 0 = persistent
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    expires_at: Optional[float] = None

    # Animation
    enter_animation: AnimationType = AnimationType.FADE
    exit_animation: AnimationType = AnimationType.FADE
    animation_duration: float = 0.3

    # Style
    style: Optional[NotificationStyle] = None
    icon: Optional[str] = None
    image_url: Optional[str] = None

    # Actions
    actions: List[NotificationAction] = field(default_factory=list)
    dismissible: bool = True

    # State
    is_read: bool = False
    is_dismissed: bool = False

    # Audio
    sound: Optional[str] = None
    haptic: Optional[str] = None  # Haptic pattern ID

    # Metadata
    source: str = ""
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.duration > 0 and not self.expires_at:
            self.expires_at = self.created_at + self.duration
        if not self.style:
            self.style = NOTIFICATION_STYLES.get(
                self.notification_type,
                NotificationStyle()
            )

    def is_expired(self) -> bool:
        """Check if notification has expired."""
        if self.expires_at is None:
            return False
        return datetime.now().timestamp() > self.expires_at

    def mark_read(self):
        """Mark notification as read."""
        self.is_read = True

    def dismiss(self):
        """Dismiss notification."""
        self.is_dismissed = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.notification_id,
            "title": self.title,
            "message": self.message,
            "type": self.notification_type.value,
            "priority": self.priority.value,
            "position": self.position.value,
            "worldPosition": self.world_position,
            "attachedNodeId": self.attached_node_id,
            "offset": self.offset,
            "duration": self.duration,
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
            "enterAnimation": self.enter_animation.value,
            "exitAnimation": self.exit_animation.value,
            "style": {
                "backgroundColor": self.style.background_color,
                "textColor": self.style.text_color,
                "accentColor": self.style.accent_color,
                "glow": self.style.glow
            } if self.style else None,
            "icon": self.icon,
            "imageUrl": self.image_url,
            "actions": [
                {"id": a.action_id, "label": a.label, "primary": a.primary}
                for a in self.actions
            ],
            "dismissible": self.dismissible,
            "isRead": self.is_read,
            "isDismissed": self.is_dismissed,
            "sound": self.sound,
            "haptic": self.haptic
        }


@dataclass
class Badge:
    """Badge indicator on an object/node."""
    badge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attached_to: str = ""  # Node or object ID
    value: Any = None  # Number, text, or icon
    badge_type: str = "count"  # count, dot, icon, text
    color: Tuple[float, float, float, float] = (1.0, 0.3, 0.3, 1.0)
    position: str = "top_right"  # top_left, top_right, bottom_left, bottom_right
    offset: Tuple[float, float] = (0, 0)
    pulse: bool = False
    visible: bool = True
    max_value: int = 99

    def to_dict(self) -> Dict[str, Any]:
        display_value = self.value
        if self.badge_type == "count" and isinstance(self.value, int):
            if self.value > self.max_value:
                display_value = f"{self.max_value}+"

        return {
            "id": self.badge_id,
            "attachedTo": self.attached_to,
            "value": display_value,
            "type": self.badge_type,
            "color": self.color,
            "position": self.position,
            "offset": self.offset,
            "pulse": self.pulse,
            "visible": self.visible
        }


@dataclass
class ProgressIndicator:
    """Progress indicator in 3D space."""
    indicator_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    position: Tuple[float, float, float] = (0, 0, 0)
    attached_to: Optional[str] = None
    progress: float = 0.0  # 0-1
    style: str = "ring"  # ring, bar, dots, spinner
    size: float = 0.1  # Meters
    color: Tuple[float, float, float, float] = (0.3, 0.6, 1.0, 1.0)
    background_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.3, 0.5)
    label: str = ""
    show_percentage: bool = True
    indeterminate: bool = False
    visible: bool = True

    def set_progress(self, value: float):
        """Set progress value (0-1)."""
        self.progress = max(0.0, min(1.0, value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.indicator_id,
            "position": self.position,
            "attachedTo": self.attached_to,
            "progress": self.progress,
            "style": self.style,
            "size": self.size,
            "color": self.color,
            "backgroundColor": self.background_color,
            "label": self.label,
            "showPercentage": self.show_percentage,
            "indeterminate": self.indeterminate,
            "visible": self.visible
        }


@dataclass
class AttentionGuide:
    """Visual guide to draw attention to something."""
    guide_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_position: Tuple[float, float, float] = (0, 0, 0)
    target_node_id: Optional[str] = None
    guide_type: str = "arrow"  # arrow, highlight, pulse, beam, orbit
    color: Tuple[float, float, float, float] = (1.0, 0.8, 0.2, 1.0)
    intensity: float = 1.0
    duration: float = 5.0
    label: str = ""
    active: bool = True
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def is_expired(self) -> bool:
        if self.duration <= 0:
            return False
        return datetime.now().timestamp() > self.created_at + self.duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.guide_id,
            "targetPosition": self.target_position,
            "targetNodeId": self.target_node_id,
            "type": self.guide_type,
            "color": self.color,
            "intensity": self.intensity,
            "duration": self.duration,
            "label": self.label,
            "active": self.active
        }


class NotificationQueue:
    """Queue for managing notification display order."""

    def __init__(self, max_visible: int = 3):
        self.max_visible = max_visible
        self.pending: List[Notification] = []
        self.visible: List[Notification] = []
        self.history: List[Notification] = []

    def add(self, notification: Notification):
        """Add notification to queue."""
        # Sort by priority
        insert_index = 0
        for i, n in enumerate(self.pending):
            if notification.priority.value > n.priority.value:
                break
            insert_index = i + 1
        self.pending.insert(insert_index, notification)

    def update(self) -> List[Notification]:
        """Update queue, return newly visible notifications."""
        # Remove expired/dismissed from visible
        self.visible = [
            n for n in self.visible
            if not n.is_expired() and not n.is_dismissed
        ]

        # Move from pending to visible
        newly_visible = []
        while len(self.visible) < self.max_visible and self.pending:
            notification = self.pending.pop(0)
            self.visible.append(notification)
            newly_visible.append(notification)

        return newly_visible

    def dismiss(self, notification_id: str):
        """Dismiss a notification."""
        for n in self.visible:
            if n.notification_id == notification_id:
                n.dismiss()
                self.history.append(n)
                break

    def clear_all(self):
        """Clear all notifications."""
        for n in self.visible:
            n.dismiss()
            self.history.append(n)
        self.visible.clear()
        self.pending.clear()


class SpatialNotificationManager:
    """
    Main notification management system for XR environments.
    """

    def __init__(
        self,
        max_visible_notifications: int = 3,
        notification_spacing: float = 0.12,
        default_duration: float = 5.0
    ):
        self.max_visible = max_visible_notifications
        self.notification_spacing = notification_spacing
        self.default_duration = default_duration

        # Notification queues by position
        self.queues: Dict[NotificationPosition, NotificationQueue] = {
            pos: NotificationQueue(max_visible_notifications)
            for pos in NotificationPosition
        }

        # All notifications by ID
        self.notifications: Dict[str, Notification] = {}

        # Badges
        self.badges: Dict[str, Badge] = {}

        # Progress indicators
        self.progress_indicators: Dict[str, ProgressIndicator] = {}

        # Attention guides
        self.attention_guides: Dict[str, AttentionGuide] = {}

        # Settings
        self.sounds_enabled: bool = True
        self.haptics_enabled: bool = True
        self.do_not_disturb: bool = False
        self.priority_filter: NotificationPriority = NotificationPriority.LOW

        # Callbacks
        self.on_notification_shown: List[Callable[[Notification], None]] = []
        self.on_notification_dismissed: List[Callable[[str], None]] = []
        self.on_action_clicked: List[Callable[[str, str], None]] = []

    def notify(
        self,
        title: str,
        message: str = "",
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        position: NotificationPosition = NotificationPosition.HEAD_LOCKED,
        duration: Optional[float] = None,
        actions: Optional[List[NotificationAction]] = None,
        sound: Optional[str] = None,
        haptic: Optional[str] = None,
        **kwargs
    ) -> Optional[Notification]:
        """Create and queue a notification."""
        # Check do not disturb
        if self.do_not_disturb and priority.value < NotificationPriority.URGENT.value:
            return None

        # Check priority filter
        if priority.value < self.priority_filter.value:
            return None

        notification = Notification(
            title=title,
            message=message,
            notification_type=notification_type,
            priority=priority,
            position=position,
            duration=duration if duration is not None else self.default_duration,
            actions=actions or [],
            sound=sound,
            haptic=haptic,
            **kwargs
        )

        # Add to storage and queue
        self.notifications[notification.notification_id] = notification
        self.queues[position].add(notification)

        return notification

    def info(self, title: str, message: str = "", **kwargs) -> Optional[Notification]:
        """Show info notification."""
        return self.notify(title, message, NotificationType.INFO, **kwargs)

    def success(self, title: str, message: str = "", **kwargs) -> Optional[Notification]:
        """Show success notification."""
        kwargs.setdefault("sound", "success")
        kwargs.setdefault("haptic", "success")
        return self.notify(title, message, NotificationType.SUCCESS, **kwargs)

    def warning(self, title: str, message: str = "", **kwargs) -> Optional[Notification]:
        """Show warning notification."""
        kwargs.setdefault("priority", NotificationPriority.HIGH)
        kwargs.setdefault("sound", "warning")
        return self.notify(title, message, NotificationType.WARNING, **kwargs)

    def error(self, title: str, message: str = "", **kwargs) -> Optional[Notification]:
        """Show error notification."""
        kwargs.setdefault("priority", NotificationPriority.HIGH)
        kwargs.setdefault("duration", 0)  # Persistent
        kwargs.setdefault("sound", "error")
        kwargs.setdefault("haptic", "error")
        return self.notify(title, message, NotificationType.ERROR, **kwargs)

    def knowledge_update(
        self,
        title: str,
        message: str = "",
        node_id: Optional[str] = None,
        **kwargs
    ) -> Optional[Notification]:
        """Show knowledge-related notification."""
        kwargs.setdefault("attached_node_id", node_id)
        if node_id:
            kwargs.setdefault("position", NotificationPosition.NODE_ATTACHED)
        return self.notify(title, message, NotificationType.KNOWLEDGE, **kwargs)

    def social(
        self,
        title: str,
        message: str = "",
        user_name: Optional[str] = None,
        **kwargs
    ) -> Optional[Notification]:
        """Show social notification (user joined, message, etc)."""
        kwargs.setdefault("sound", "social")
        if user_name:
            kwargs.setdefault("metadata", {})["user_name"] = user_name
        return self.notify(title, message, NotificationType.SOCIAL, **kwargs)

    def achievement(
        self,
        title: str,
        message: str = "",
        **kwargs
    ) -> Optional[Notification]:
        """Show achievement notification."""
        kwargs.setdefault("priority", NotificationPriority.HIGH)
        kwargs.setdefault("duration", 8.0)
        kwargs.setdefault("sound", "achievement")
        kwargs.setdefault("haptic", "success")
        kwargs.setdefault("enter_animation", AnimationType.SCALE)
        return self.notify(title, message, NotificationType.ACHIEVEMENT, **kwargs)

    def update(self) -> Dict[str, List[Notification]]:
        """Update all queues, return newly visible notifications by position."""
        newly_visible = {}
        for position, queue in self.queues.items():
            new = queue.update()
            if new:
                newly_visible[position.value] = new
                for n in new:
                    for callback in self.on_notification_shown:
                        callback(n)

        # Update attention guides
        self.attention_guides = {
            gid: g for gid, g in self.attention_guides.items()
            if not g.is_expired() and g.active
        }

        return newly_visible

    def dismiss(self, notification_id: str):
        """Dismiss a notification."""
        notification = self.notifications.get(notification_id)
        if notification:
            notification.dismiss()
            for callback in self.on_notification_dismissed:
                callback(notification_id)

    def dismiss_all(self):
        """Dismiss all notifications."""
        for queue in self.queues.values():
            queue.clear_all()

    def handle_action(self, notification_id: str, action_id: str):
        """Handle action button click."""
        notification = self.notifications.get(notification_id)
        if notification:
            for action in notification.actions:
                if action.action_id == action_id:
                    for callback in self.on_action_clicked:
                        callback(notification_id, action_id)
                    break

    def add_badge(
        self,
        attached_to: str,
        value: Any = None,
        badge_type: str = "count",
        color: Tuple[float, float, float, float] = (1.0, 0.3, 0.3, 1.0),
        **kwargs
    ) -> Badge:
        """Add badge to object/node."""
        badge = Badge(
            attached_to=attached_to,
            value=value,
            badge_type=badge_type,
            color=color,
            **kwargs
        )
        self.badges[badge.badge_id] = badge
        return badge

    def update_badge(self, badge_id: str, value: Any):
        """Update badge value."""
        if badge_id in self.badges:
            self.badges[badge_id].value = value

    def remove_badge(self, badge_id: str):
        """Remove badge."""
        self.badges.pop(badge_id, None)

    def clear_badges_for(self, attached_to: str):
        """Remove all badges for an object."""
        self.badges = {
            bid: b for bid, b in self.badges.items()
            if b.attached_to != attached_to
        }

    def create_progress(
        self,
        position: Tuple[float, float, float] = (0, 0, 0),
        style: str = "ring",
        label: str = "",
        attached_to: Optional[str] = None,
        **kwargs
    ) -> ProgressIndicator:
        """Create progress indicator."""
        indicator = ProgressIndicator(
            position=position,
            style=style,
            label=label,
            attached_to=attached_to,
            **kwargs
        )
        self.progress_indicators[indicator.indicator_id] = indicator
        return indicator

    def update_progress(self, indicator_id: str, progress: float):
        """Update progress value."""
        if indicator_id in self.progress_indicators:
            self.progress_indicators[indicator_id].set_progress(progress)

    def complete_progress(self, indicator_id: str, remove_delay: float = 1.0):
        """Mark progress as complete."""
        if indicator_id in self.progress_indicators:
            self.progress_indicators[indicator_id].progress = 1.0
            # Would schedule removal after delay in real implementation

    def remove_progress(self, indicator_id: str):
        """Remove progress indicator."""
        self.progress_indicators.pop(indicator_id, None)

    def guide_attention(
        self,
        target_position: Optional[Tuple[float, float, float]] = None,
        target_node_id: Optional[str] = None,
        guide_type: str = "arrow",
        duration: float = 5.0,
        label: str = "",
        **kwargs
    ) -> AttentionGuide:
        """Create attention guide."""
        guide = AttentionGuide(
            target_position=target_position or (0, 0, 0),
            target_node_id=target_node_id,
            guide_type=guide_type,
            duration=duration,
            label=label,
            **kwargs
        )
        self.attention_guides[guide.guide_id] = guide
        return guide

    def stop_guide(self, guide_id: str):
        """Stop attention guide."""
        if guide_id in self.attention_guides:
            self.attention_guides[guide_id].active = False

    def set_do_not_disturb(self, enabled: bool):
        """Enable/disable do not disturb mode."""
        self.do_not_disturb = enabled

    def get_unread_count(self) -> int:
        """Get count of unread notifications."""
        return sum(
            1 for n in self.notifications.values()
            if not n.is_read and not n.is_dismissed
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        total = len(self.notifications)
        by_type = {}
        for n in self.notifications.values():
            t = n.notification_type.value
            by_type[t] = by_type.get(t, 0) + 1

        visible_count = sum(len(q.visible) for q in self.queues.values())
        pending_count = sum(len(q.pending) for q in self.queues.values())

        return {
            "total_notifications": total,
            "visible": visible_count,
            "pending": pending_count,
            "unread": self.get_unread_count(),
            "by_type": by_type,
            "badges": len(self.badges),
            "progress_indicators": len(self.progress_indicators),
            "attention_guides": len(self.attention_guides),
            "do_not_disturb": self.do_not_disturb
        }

    def to_state(self) -> Dict[str, Any]:
        """Export full state for WebXR client."""
        visible_notifications = []
        for position, queue in self.queues.items():
            for i, n in enumerate(queue.visible):
                n_dict = n.to_dict()
                n_dict["displayIndex"] = i
                n_dict["displayOffset"] = i * self.notification_spacing
                visible_notifications.append(n_dict)

        return {
            "type": "notification_state",
            "notifications": visible_notifications,
            "badges": [b.to_dict() for b in self.badges.values()],
            "progressIndicators": [p.to_dict() for p in self.progress_indicators.values()],
            "attentionGuides": [g.to_dict() for g in self.attention_guides.values() if g.active],
            "settings": {
                "soundsEnabled": self.sounds_enabled,
                "hapticsEnabled": self.haptics_enabled,
                "doNotDisturb": self.do_not_disturb
            },
            "statistics": self.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }
