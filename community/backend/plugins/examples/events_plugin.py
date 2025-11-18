"""
Events Plugin

Adds event creation and RSVP functionality to communities
"""
from typing import Dict, Any
from plugins.base import PluginBase, PluginType, HookType, PluginContext
import logging

logger = logging.getLogger(__name__)


class EventsPlugin(PluginBase):
    """
    Events plugin - community event management with RSVP tracking

    Configuration:
        - max_events_per_community: Max concurrent events (default: 50)
        - allow_recurring: Allow recurring events (default: True)
        - rsvp_deadline_hours: Hours before event to close RSVP (default: 24)
        - max_attendees: Max attendees per event (default: unlimited)
    """

    @property
    def name(self) -> str:
        return "Events"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Community event management with RSVP tracking, calendar sync, and reminders"

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.CONTENT

    async def initialize(self) -> bool:
        """Initialize the events plugin"""
        logger.info(f"Initializing {self.name} plugin v{self.version}")

        # Register hooks
        self.register_hook(HookType.BEFORE_POST_CREATE, self._validate_event)
        self.register_hook(HookType.AFTER_POST_CREATE, self._create_event)
        self.register_hook(HookType.POST_RENDER, self._render_event)
        self.register_hook(HookType.COMMUNITY_JOIN, self._notify_upcoming_events)

        # Configuration
        self.max_events = self.get_config("max_events_per_community", 50)
        self.allow_recurring = self.get_config("allow_recurring", True)
        self.rsvp_deadline_hours = self.get_config("rsvp_deadline_hours", 24)
        self.max_attendees = self.get_config("max_attendees", None)

        logger.info(f"{self.name} plugin initialized")
        return True

    async def shutdown(self) -> bool:
        """Shutdown the events plugin"""
        logger.info(f"Shutting down {self.name} plugin")
        return True

    async def _validate_event(self, context: PluginContext) -> PluginContext:
        """Validate event data before post creation"""
        post_type = context.get("post_type")

        if post_type != "event":
            return context

        event_data = context.get("event_data")

        if not event_data:
            context.set("error", "Event data is required for event posts")
            return context

        # Validate event structure
        required_fields = ["title", "start_time", "location"]
        for field in required_fields:
            if field not in event_data:
                context.set("error", f"Event {field} is required")
                return context

        # Validate start time is in future
        # TODO: Add actual date validation

        # Check max events limit for community
        community_id = context.get("community_id")
        if community_id:
            # TODO: Query event count from database
            # For now, just log
            logger.debug(f"Checking event limit for community {community_id}")

        # Validation passed
        context.set("event_validated", True)
        logger.debug(f"Event validated: {event_data.get('title')}")

        return context

    async def _create_event(self, context: PluginContext) -> PluginContext:
        """
        Create event data after post creation

        Hook: AFTER_POST_CREATE
        """
        post_type = context.get("post_type")

        if post_type != "event":
            return context

        post_id = context.get("post_id")
        event_data = context.get("event_data")

        if not event_data:
            return context

        # In a real implementation, this would:
        # 1. Create event record in database
        # 2. Set up RSVP tracking
        # 3. Schedule reminder notifications
        # 4. Generate calendar invite (.ics file)

        logger.info(
            f"Created event for post {post_id}: {event_data.get('title')} "
            f"at {event_data.get('start_time')}"
        )

        # Store event metadata in context
        context.set("event_created", True)
        context.set("event_id", f"event_{post_id}")

        return context

    async def _render_event(self, context: PluginContext) -> PluginContext:
        """
        Render event UI in post

        Hook: POST_RENDER
        """
        post_type = context.get("post_type")

        if post_type != "event":
            return context

        event_data = context.get("event_data")
        user_id = context.get("user_id")

        if not event_data:
            return context

        # Check if user has RSVP'd
        has_rsvp = self._check_user_rsvp(user_id, event_data.get("event_id"))

        # Build event HTML
        html = self._build_event_html(
            event_data,
            has_rsvp=has_rsvp,
        )

        context.set("event_html", html)
        logger.debug(f"Rendered event for user {user_id}")

        return context

    async def _notify_upcoming_events(self, context: PluginContext) -> PluginContext:
        """
        Notify user of upcoming events when joining community

        Hook: COMMUNITY_JOIN
        """
        user_id = context.get("user_id")
        community_id = context.get("community_id")

        # TODO: Query upcoming events in community
        # Send notification to user

        logger.debug(f"Notified user {user_id} of upcoming events in community {community_id}")

        return context

    def _check_user_rsvp(self, user_id: str, event_id: str) -> bool:
        """Check if user has RSVP'd to event"""
        # Placeholder - would query database
        return False

    def _build_event_html(self, event_data: Dict[str, Any], has_rsvp: bool) -> str:
        """Build event HTML markup"""
        title = event_data.get("title", "")
        start_time = event_data.get("start_time", "")
        end_time = event_data.get("end_time", "")
        location = event_data.get("location", "")
        description = event_data.get("description", "")
        attendees = event_data.get("attendees", [])

        html = f'<div class="event">'
        html += f'<div class="event-header">'
        html += f'<h2 class="event-title">{title}</h2>'
        html += f'</div>'

        html += f'<div class="event-details">'
        html += f'<div class="event-time">'
        html += f'<span class="icon">📅</span>'
        html += f'<span>{start_time} - {end_time}</span>'
        html += f'</div>'
        html += f'<div class="event-location">'
        html += f'<span class="icon">📍</span>'
        html += f'<span>{location}</span>'
        html += f'</div>'
        html += f'</div>'

        if description:
            html += f'<div class="event-description">{description}</div>'

        html += f'<div class="event-attendees">'
        html += f'<span class="attendee-count">{len(attendees)} attending</span>'
        html += f'</div>'

        if has_rsvp:
            html += f'<button class="event-rsvp event-rsvp-cancel">Cancel RSVP</button>'
        else:
            html += f'<button class="event-rsvp event-rsvp-attend">RSVP</button>'

        html += f'<button class="event-calendar">Add to Calendar</button>'

        html += f'</div>'

        return html


# Export plugin class
__all__ = ["EventsPlugin"]
