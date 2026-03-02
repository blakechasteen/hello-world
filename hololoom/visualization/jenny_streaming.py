"""
Jenny Streaming Manager
========================
Real-time data updates for reactive and streaming panel bindings.

Week 3 MVP - December 2025

Philosophy:
> "Live data, live decisions."

This module handles data binding updates for Jenny panels:
- STATIC: No updates (handled at render time)
- REACTIVE: Poll-based refresh at configurable intervals
- STREAMING: SSE/WebSocket push updates

Data Flow:
    Data Source → StreamingManager.subscribe()
                → Panel receives update
                → Renderer re-renders affected content
                → SpecLedger logs data change event

References:
- jenny_spec.py (BindingMode enum)
- jenny_lifecycle.py (panel state management)
- spec_ledger.py (provenance tracking)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable, Set
from enum import Enum
from datetime import datetime
from uuid import uuid4
import asyncio
import json

from .jenny_spec import JennySpec, BindingMode


# ============================================================================
# Streaming Types
# ============================================================================

class StreamStatus(str, Enum):
    """Status of a streaming subscription."""
    ACTIVE = "active"           # Receiving updates
    PAUSED = "paused"           # Temporarily paused
    DISCONNECTED = "disconnected"  # Connection lost
    CLOSED = "closed"           # Explicitly closed


class UpdateType(str, Enum):
    """Type of data update."""
    FULL = "full"               # Complete data replacement
    PARTIAL = "partial"         # Incremental update (patch)
    HEARTBEAT = "heartbeat"     # Keep-alive signal
    ERROR = "error"             # Error notification


@dataclass(frozen=True)
class StreamUpdate:
    """
    Immutable data update event.

    Sent from StreamingManager to subscribed panels.
    """
    update_id: str = field(default_factory=lambda: str(uuid4()))
    spec_id: str = ""
    data_source: str = ""
    update_type: UpdateType = UpdateType.FULL
    timestamp: datetime = field(default_factory=datetime.now)

    # Update payload
    data: Dict[str, Any] = field(default_factory=dict)

    # Error info (if update_type == ERROR)
    error: Optional[str] = None

    # Sequence number for ordering
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket/SSE transmission."""
        return {
            "update_id": self.update_id,
            "spec_id": self.spec_id,
            "data_source": self.data_source,
            "update_type": self.update_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "error": self.error,
            "sequence": self.sequence,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Subscription:
    """
    Active subscription to a data source.

    Tracks subscription state and configuration.
    """
    subscription_id: str = field(default_factory=lambda: str(uuid4()))
    spec_id: str = ""
    data_source: str = ""
    binding_mode: BindingMode = BindingMode.REACTIVE

    # Configuration
    refresh_interval_ms: int = 5000  # For REACTIVE mode
    max_updates_per_second: int = 10  # Backpressure limit

    # State
    status: StreamStatus = StreamStatus.ACTIVE
    last_update: Optional[datetime] = None
    sequence: int = 0
    error_count: int = 0

    # Callback
    callback: Optional[Callable[[StreamUpdate], Awaitable[None]]] = None


# ============================================================================
# Data Source Protocol
# ============================================================================

class DataSourceProtocol:
    """
    Protocol for data sources that can be subscribed to.

    Implementations provide actual data fetching/streaming logic.
    """

    async def fetch(self, query: str) -> Dict[str, Any]:
        """Fetch current data for a query (REACTIVE mode)."""
        raise NotImplementedError

    async def subscribe(
        self,
        query: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> str:
        """Subscribe to streaming data (STREAMING mode). Returns subscription ID."""
        raise NotImplementedError

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from streaming data."""
        raise NotImplementedError


# ============================================================================
# Mock Data Source (for testing)
# ============================================================================

class MockDataSource(DataSourceProtocol):
    """
    Mock data source for testing streaming functionality.

    Generates predictable data updates.
    """

    def __init__(self):
        self._subscriptions: Dict[str, asyncio.Task] = {}
        self._counter: int = 0

    async def fetch(self, query: str) -> Dict[str, Any]:
        """Return mock data for query."""
        self._counter += 1
        return {
            "query": query,
            "result": f"Mock result #{self._counter}",
            "timestamp": datetime.now().isoformat(),
        }

    async def subscribe(
        self,
        query: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> str:
        """Start mock streaming updates."""
        subscription_id = str(uuid4())

        async def stream_loop():
            counter = 0
            while True:
                counter += 1
                data = {
                    "query": query,
                    "result": f"Stream update #{counter}",
                    "timestamp": datetime.now().isoformat(),
                }
                await callback(data)
                await asyncio.sleep(1.0)  # 1 update per second

        task = asyncio.create_task(stream_loop())
        self._subscriptions[subscription_id] = task
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Stop mock streaming."""
        if subscription_id in self._subscriptions:
            self._subscriptions[subscription_id].cancel()
            del self._subscriptions[subscription_id]


# ============================================================================
# Streaming Manager
# ============================================================================

class StreamingManager:
    """
    Manages data subscriptions for Jenny panels.

    Handles:
    - REACTIVE: Poll-based refresh at intervals
    - STREAMING: Push-based updates via callbacks
    - Backpressure control
    - Reconnection logic
    - Error handling

    Usage:
        manager = StreamingManager(data_source=MyDataSource())

        # Subscribe a panel
        sub_id = await manager.subscribe(
            spec=my_spec,
            callback=handle_update
        )

        # Later: unsubscribe
        await manager.unsubscribe(sub_id)
    """

    def __init__(
        self,
        data_source: Optional[DataSourceProtocol] = None,
        max_subscriptions: int = 100,
        default_refresh_interval_ms: int = 5000,
        max_reconnect_attempts: int = 3,
        reconnect_delay_ms: int = 1000,
    ):
        """
        Initialize streaming manager.

        Args:
            data_source: Data source implementation (defaults to MockDataSource)
            max_subscriptions: Maximum concurrent subscriptions
            default_refresh_interval_ms: Default poll interval for REACTIVE mode
            max_reconnect_attempts: Max reconnection attempts on failure
            reconnect_delay_ms: Delay between reconnection attempts
        """
        self._data_source = data_source or MockDataSource()
        self._max_subscriptions = max_subscriptions
        self._default_refresh_interval_ms = default_refresh_interval_ms
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay_ms = reconnect_delay_ms

        # Active subscriptions
        self._subscriptions: Dict[str, Subscription] = {}

        # Background tasks for reactive polling
        self._poll_tasks: Dict[str, asyncio.Task] = {}

        # Streaming source subscriptions
        self._stream_sub_ids: Dict[str, str] = {}  # our_id → source_id

        # Statistics
        self._total_updates_sent: int = 0
        self._total_errors: int = 0

    async def subscribe(
        self,
        spec: JennySpec,
        callback: Callable[[StreamUpdate], Awaitable[None]],
        refresh_interval_ms: Optional[int] = None,
    ) -> str:
        """
        Subscribe a panel to its data source.

        Args:
            spec: JennySpec with binding_mode and data_source
            callback: Async function to call with updates
            refresh_interval_ms: Override refresh interval (REACTIVE mode)

        Returns:
            Subscription ID

        Raises:
            ValueError: If spec.binding_mode is STATIC (no subscription needed)
            RuntimeError: If max subscriptions exceeded
        """
        if spec.binding_mode == BindingMode.STATIC:
            raise ValueError("STATIC binding mode does not support subscriptions")

        if len(self._subscriptions) >= self._max_subscriptions:
            raise RuntimeError(
                f"Max subscriptions ({self._max_subscriptions}) exceeded"
            )

        if not spec.data_source:
            raise ValueError("Spec must have data_source for subscription")

        # Create subscription record
        subscription = Subscription(
            spec_id=spec.spec_id,
            data_source=spec.data_source,
            binding_mode=spec.binding_mode,
            refresh_interval_ms=refresh_interval_ms or spec.refresh_interval_ms or self._default_refresh_interval_ms,
            callback=callback,
        )

        self._subscriptions[subscription.subscription_id] = subscription

        # Start appropriate update mechanism
        if spec.binding_mode == BindingMode.REACTIVE:
            await self._start_polling(subscription)
        elif spec.binding_mode == BindingMode.STREAMING:
            await self._start_streaming(subscription)

        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from data updates.

        Args:
            subscription_id: ID from subscribe()

        Returns:
            True if unsubscribed, False if not found
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]

        # Stop polling task if exists
        if subscription_id in self._poll_tasks:
            self._poll_tasks[subscription_id].cancel()
            del self._poll_tasks[subscription_id]

        # Unsubscribe from streaming source
        if subscription_id in self._stream_sub_ids:
            source_sub_id = self._stream_sub_ids[subscription_id]
            await self._data_source.unsubscribe(source_sub_id)
            del self._stream_sub_ids[subscription_id]

        # Update status and remove
        subscription.status = StreamStatus.CLOSED
        del self._subscriptions[subscription_id]

        return True

    async def pause(self, subscription_id: str) -> bool:
        """Pause a subscription (stop receiving updates)."""
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]
        subscription.status = StreamStatus.PAUSED

        # Cancel polling task
        if subscription_id in self._poll_tasks:
            self._poll_tasks[subscription_id].cancel()
            del self._poll_tasks[subscription_id]

        return True

    async def resume(self, subscription_id: str) -> bool:
        """Resume a paused subscription."""
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]
        if subscription.status != StreamStatus.PAUSED:
            return False

        subscription.status = StreamStatus.ACTIVE

        # Restart polling if reactive
        if subscription.binding_mode == BindingMode.REACTIVE:
            await self._start_polling(subscription)

        return True

    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription details."""
        return self._subscriptions.get(subscription_id)

    def get_subscriptions_for_spec(self, spec_id: str) -> List[Subscription]:
        """Get all subscriptions for a spec."""
        return [
            sub for sub in self._subscriptions.values()
            if sub.spec_id == spec_id
        ]

    def get_active_subscriptions(self) -> List[Subscription]:
        """Get all active subscriptions."""
        return [
            sub for sub in self._subscriptions.values()
            if sub.status == StreamStatus.ACTIVE
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        active = sum(1 for s in self._subscriptions.values() if s.status == StreamStatus.ACTIVE)
        paused = sum(1 for s in self._subscriptions.values() if s.status == StreamStatus.PAUSED)

        return {
            "total_subscriptions": len(self._subscriptions),
            "active_subscriptions": active,
            "paused_subscriptions": paused,
            "total_updates_sent": self._total_updates_sent,
            "total_errors": self._total_errors,
            "reactive_count": sum(
                1 for s in self._subscriptions.values()
                if s.binding_mode == BindingMode.REACTIVE
            ),
            "streaming_count": sum(
                1 for s in self._subscriptions.values()
                if s.binding_mode == BindingMode.STREAMING
            ),
        }

    async def _start_polling(self, subscription: Subscription) -> None:
        """Start polling loop for REACTIVE subscription."""

        async def poll_loop():
            while subscription.status == StreamStatus.ACTIVE:
                try:
                    # Fetch data
                    data = await self._data_source.fetch(subscription.data_source)

                    # Create update
                    subscription.sequence += 1
                    update = StreamUpdate(
                        spec_id=subscription.spec_id,
                        data_source=subscription.data_source,
                        update_type=UpdateType.FULL,
                        data=data,
                        sequence=subscription.sequence,
                    )

                    # Send to callback
                    if subscription.callback:
                        await subscription.callback(update)

                    subscription.last_update = datetime.now()
                    subscription.error_count = 0
                    self._total_updates_sent += 1

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    subscription.error_count += 1
                    self._total_errors += 1

                    # Send error update
                    error_update = StreamUpdate(
                        spec_id=subscription.spec_id,
                        data_source=subscription.data_source,
                        update_type=UpdateType.ERROR,
                        error=str(e),
                        sequence=subscription.sequence,
                    )
                    if subscription.callback:
                        await subscription.callback(error_update)

                    # Disconnect after too many errors
                    if subscription.error_count >= self._max_reconnect_attempts:
                        subscription.status = StreamStatus.DISCONNECTED
                        break

                # Wait for next poll
                await asyncio.sleep(subscription.refresh_interval_ms / 1000.0)

        task = asyncio.create_task(poll_loop())
        self._poll_tasks[subscription.subscription_id] = task

    async def _start_streaming(self, subscription: Subscription) -> None:
        """Start streaming subscription via data source."""

        async def handle_stream_data(data: Dict[str, Any]):
            subscription.sequence += 1
            update = StreamUpdate(
                spec_id=subscription.spec_id,
                data_source=subscription.data_source,
                update_type=UpdateType.FULL,
                data=data,
                sequence=subscription.sequence,
            )
            if subscription.callback:
                await subscription.callback(update)
            subscription.last_update = datetime.now()
            self._total_updates_sent += 1

        try:
            source_sub_id = await self._data_source.subscribe(
                subscription.data_source,
                handle_stream_data,
            )
            self._stream_sub_ids[subscription.subscription_id] = source_sub_id
        except Exception as e:
            subscription.status = StreamStatus.DISCONNECTED
            subscription.error_count += 1
            self._total_errors += 1

    async def close(self) -> None:
        """Close all subscriptions and clean up."""
        # Unsubscribe all
        sub_ids = list(self._subscriptions.keys())
        for sub_id in sub_ids:
            await self.unsubscribe(sub_id)


# ============================================================================
# Streaming Panel Indicators (Task 3.4 - December 2025)
# ============================================================================
# Visual indicators for streaming panels:
# - Pulse animation for STREAMING binding mode
# - Token count / progress indicator
# - "Live" badge for SSE connections
# - Reconnection status

class StreamingBadge(str, Enum):
    """
    Badge types for streaming panels.

    Each badge type has associated styling and behavior.
    """
    LIVE = "live"           # Green pulsing dot - active SSE/WebSocket
    STREAMING = "streaming"  # Blue animated wave - actively receiving data
    PAUSED = "paused"       # Yellow pause icon - subscription paused
    RECONNECTING = "reconnecting"  # Orange spinning icon - reconnection in progress
    ERROR = "error"         # Red exclamation - connection error
    OFFLINE = "offline"     # Gray icon - not connected


@dataclass(frozen=True)
class StreamingIndicator:
    """
    Visual indicator configuration for streaming panels.

    Immutable specification for rendering streaming status in the UI.

    Fields:
        badge: Type of badge to display
        badge_label: Human-readable label (e.g., "LIVE", "STREAMING")
        badge_tooltip: Detailed tooltip text
        pulse_animation: Whether to show pulse animation
        pulse_color: CSS color for pulse (e.g., "#22c55e" for green)
        pulse_duration_ms: Animation duration in milliseconds
        show_progress: Whether to show progress indicator
        progress_label: Progress text (e.g., "127 tokens", "3.2KB received")
        progress_value: Progress percentage (0.0-1.0) or None for indeterminate
        show_reconnect_status: Whether to show reconnection info
        reconnect_attempts: Number of reconnection attempts so far
        reconnect_max: Maximum reconnection attempts
        reconnect_message: Reconnection status message
        render_hints: Additional CSS/styling hints for renderer
    """
    badge: StreamingBadge = StreamingBadge.STREAMING
    badge_label: str = "STREAMING"
    badge_tooltip: str = "Receiving live updates"

    # Pulse animation
    pulse_animation: bool = True
    pulse_color: str = "#3b82f6"  # Blue
    pulse_duration_ms: int = 2000

    # Progress indicator
    show_progress: bool = False
    progress_label: str = ""
    progress_value: Optional[float] = None  # None = indeterminate

    # Reconnection status
    show_reconnect_status: bool = False
    reconnect_attempts: int = 0
    reconnect_max: int = 3
    reconnect_message: str = ""

    # Render hints
    render_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON transmission."""
        return {
            "badge": self.badge.value,
            "badge_label": self.badge_label,
            "badge_tooltip": self.badge_tooltip,
            "pulse_animation": self.pulse_animation,
            "pulse_color": self.pulse_color,
            "pulse_duration_ms": self.pulse_duration_ms,
            "show_progress": self.show_progress,
            "progress_label": self.progress_label,
            "progress_value": self.progress_value,
            "show_reconnect_status": self.show_reconnect_status,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_max": self.reconnect_max,
            "reconnect_message": self.reconnect_message,
            "render_hints": self.render_hints,
        }


# Badge configurations with styling
BADGE_STYLES: Dict[StreamingBadge, Dict[str, Any]] = {
    StreamingBadge.LIVE: {
        "label": "LIVE",
        "tooltip": "Live SSE/WebSocket connection active",
        "color": "#22c55e",  # Green
        "bg_color": "#dcfce7",  # Light green
        "pulse": True,
        "icon": "●",  # Filled circle
    },
    StreamingBadge.STREAMING: {
        "label": "STREAMING",
        "tooltip": "Receiving streaming updates",
        "color": "#3b82f6",  # Blue
        "bg_color": "#dbeafe",  # Light blue
        "pulse": True,
        "icon": "◐",  # Half-filled circle (animated rotation)
    },
    StreamingBadge.PAUSED: {
        "label": "PAUSED",
        "tooltip": "Streaming temporarily paused",
        "color": "#eab308",  # Yellow
        "bg_color": "#fef9c3",  # Light yellow
        "pulse": False,
        "icon": "⏸",  # Pause icon
    },
    StreamingBadge.RECONNECTING: {
        "label": "RECONNECTING",
        "tooltip": "Attempting to reconnect...",
        "color": "#f97316",  # Orange
        "bg_color": "#ffedd5",  # Light orange
        "pulse": True,
        "icon": "↻",  # Spinning arrows
    },
    StreamingBadge.ERROR: {
        "label": "ERROR",
        "tooltip": "Connection error - click to retry",
        "color": "#ef4444",  # Red
        "bg_color": "#fee2e2",  # Light red
        "pulse": False,
        "icon": "⚠",  # Warning
    },
    StreamingBadge.OFFLINE: {
        "label": "OFFLINE",
        "tooltip": "Not connected",
        "color": "#6b7280",  # Gray
        "bg_color": "#f3f4f6",  # Light gray
        "pulse": False,
        "icon": "○",  # Empty circle
    },
}


def get_streaming_indicator(
    spec: JennySpec,
    subscription: Optional[Subscription] = None,
    token_count: int = 0,
    bytes_received: int = 0,
) -> Optional[StreamingIndicator]:
    """
    Generate a streaming indicator for a panel.

    Creates visual indicator based on binding mode and subscription state.

    Args:
        spec: JennySpec to generate indicator for
        subscription: Active subscription (if any)
        token_count: Number of tokens received (for progress display)
        bytes_received: Number of bytes received (for progress display)

    Returns:
        StreamingIndicator or None (for STATIC binding mode)

    Usage:
        indicator = get_streaming_indicator(spec, subscription)
        if indicator:
            # Add indicator to panel rendering
            panel_html += render_streaming_badge(indicator)
    """
    # STATIC panels don't get indicators
    if spec.binding_mode == BindingMode.STATIC:
        return None

    # Determine badge based on subscription state
    if subscription is None:
        # No subscription yet - show pending state
        badge = StreamingBadge.OFFLINE
    elif subscription.status == StreamStatus.ACTIVE:
        if spec.binding_mode == BindingMode.STREAMING:
            badge = StreamingBadge.LIVE
        else:
            badge = StreamingBadge.STREAMING
    elif subscription.status == StreamStatus.PAUSED:
        badge = StreamingBadge.PAUSED
    elif subscription.status == StreamStatus.DISCONNECTED:
        if subscription.error_count > 0 and subscription.error_count < 3:
            badge = StreamingBadge.RECONNECTING
        else:
            badge = StreamingBadge.ERROR
    else:
        badge = StreamingBadge.OFFLINE

    # Get badge styling
    style = BADGE_STYLES[badge]

    # Build progress label if we have data
    progress_label = ""
    show_progress = False
    if token_count > 0 or bytes_received > 0:
        show_progress = True
        parts = []
        if token_count > 0:
            parts.append(f"{token_count:,} tokens")
        if bytes_received > 0:
            if bytes_received < 1024:
                parts.append(f"{bytes_received}B")
            elif bytes_received < 1024 * 1024:
                parts.append(f"{bytes_received / 1024:.1f}KB")
            else:
                parts.append(f"{bytes_received / (1024 * 1024):.1f}MB")
        progress_label = " • ".join(parts)

    # Build reconnection message
    reconnect_message = ""
    show_reconnect = False
    if subscription and subscription.error_count > 0:
        show_reconnect = True
        remaining = 3 - subscription.error_count
        if remaining > 0:
            reconnect_message = f"Retrying ({subscription.error_count}/3)..."
        else:
            reconnect_message = "Connection failed. Click to retry."

    # Create indicator
    return StreamingIndicator(
        badge=badge,
        badge_label=style["label"],
        badge_tooltip=style["tooltip"],
        pulse_animation=style["pulse"],
        pulse_color=style["color"],
        pulse_duration_ms=2000 if badge == StreamingBadge.LIVE else 1500,
        show_progress=show_progress,
        progress_label=progress_label,
        progress_value=None,  # Indeterminate for streaming
        show_reconnect_status=show_reconnect,
        reconnect_attempts=subscription.error_count if subscription else 0,
        reconnect_max=3,
        reconnect_message=reconnect_message,
        render_hints={
            "bg_color": style["bg_color"],
            "icon": style["icon"],
            "binding_mode": spec.binding_mode.value,
            "refresh_interval_ms": spec.refresh_interval_ms,
        }
    )


def get_streaming_css() -> str:
    """
    Get CSS for streaming indicators.

    Returns CSS string for embedding in HTML renderers.
    Includes animations, badge styles, and progress indicators.

    Usage:
        html = f"<style>{get_streaming_css()}</style>" + panel_html
    """
    return """
/* Jenny Streaming Indicators - Task 3.4 (December 2025) */

/* Base badge style */
.jenny-streaming-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Badge variants */
.jenny-badge-live {
    background-color: #dcfce7;
    color: #22c55e;
}

.jenny-badge-streaming {
    background-color: #dbeafe;
    color: #3b82f6;
}

.jenny-badge-paused {
    background-color: #fef9c3;
    color: #eab308;
}

.jenny-badge-reconnecting {
    background-color: #ffedd5;
    color: #f97316;
}

.jenny-badge-error {
    background-color: #fee2e2;
    color: #ef4444;
}

.jenny-badge-offline {
    background-color: #f3f4f6;
    color: #6b7280;
}

/* Pulse animation for LIVE badge */
.jenny-pulse {
    animation: jenny-pulse-animation 2s ease-in-out infinite;
}

@keyframes jenny-pulse-animation {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

/* Pulse dot indicator */
.jenny-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: jenny-pulse-dot 2s ease-in-out infinite;
}

.jenny-pulse-dot-live {
    background-color: #22c55e;
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
}

@keyframes jenny-pulse-dot {
    0% {
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(34, 197, 94, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0);
    }
}

/* Spinning icon for streaming/reconnecting */
.jenny-spin {
    animation: jenny-spin 1s linear infinite;
}

@keyframes jenny-spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

/* Wave animation for streaming data */
.jenny-wave {
    animation: jenny-wave 1.5s ease-in-out infinite;
}

@keyframes jenny-wave {
    0%, 100% {
        transform: scaleY(0.5);
    }
    50% {
        transform: scaleY(1);
    }
}

/* Progress indicator */
.jenny-streaming-progress {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: #6b7280;
    margin-top: 2px;
}

.jenny-progress-bar {
    width: 60px;
    height: 3px;
    background-color: #e5e7eb;
    border-radius: 2px;
    overflow: hidden;
}

.jenny-progress-fill {
    height: 100%;
    background-color: #3b82f6;
    transition: width 0.3s ease;
}

/* Indeterminate progress animation */
.jenny-progress-indeterminate {
    background: linear-gradient(90deg, #3b82f6 0%, #3b82f6 30%, transparent 30%, transparent 70%, #3b82f6 70%, #3b82f6 100%);
    background-size: 200% 100%;
    animation: jenny-progress-indeterminate 1.5s linear infinite;
}

@keyframes jenny-progress-indeterminate {
    0% {
        background-position: 100% 0;
    }
    100% {
        background-position: -100% 0;
    }
}

/* Reconnection status */
.jenny-reconnect-status {
    font-size: 10px;
    color: #f97316;
    margin-top: 2px;
}

/* Token counter */
.jenny-token-counter {
    font-family: monospace;
    font-size: 10px;
    color: #6b7280;
}

/* Streaming wrapper (positions badge in panel header) */
.jenny-panel-streaming-header {
    display: flex;
    align-items: center;
    gap: 8px;
}

.jenny-streaming-indicator-wrapper {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}
"""


def render_streaming_badge_html(indicator: StreamingIndicator) -> str:
    """
    Render streaming indicator as HTML.

    Args:
        indicator: StreamingIndicator to render

    Returns:
        HTML string for the streaming badge
    """
    badge_class = f"jenny-badge-{indicator.badge.value}"
    pulse_class = "jenny-pulse" if indicator.pulse_animation else ""
    icon = indicator.render_hints.get("icon", "●")

    # Spin icon for reconnecting
    icon_class = ""
    if indicator.badge == StreamingBadge.RECONNECTING:
        icon_class = "jenny-spin"
    elif indicator.badge == StreamingBadge.LIVE:
        icon_class = ""  # Pulse dot handled separately

    html_parts = [
        f'<div class="jenny-streaming-indicator-wrapper" title="{indicator.badge_tooltip}">',
        f'  <span class="jenny-streaming-badge {badge_class} {pulse_class}">',
    ]

    # Add pulse dot for LIVE badge
    if indicator.badge == StreamingBadge.LIVE:
        html_parts.append('    <span class="jenny-pulse-dot jenny-pulse-dot-live"></span>')
    else:
        html_parts.append(f'    <span class="{icon_class}">{icon}</span>')

    html_parts.append(f'    {indicator.badge_label}')
    html_parts.append('  </span>')

    # Progress indicator
    if indicator.show_progress:
        html_parts.append('  <div class="jenny-streaming-progress">')
        html_parts.append('    <div class="jenny-progress-bar">')
        if indicator.progress_value is not None:
            width = int(indicator.progress_value * 100)
            html_parts.append(f'      <div class="jenny-progress-fill" style="width: {width}%"></div>')
        else:
            html_parts.append('      <div class="jenny-progress-fill jenny-progress-indeterminate"></div>')
        html_parts.append('    </div>')
        if indicator.progress_label:
            html_parts.append(f'    <span class="jenny-token-counter">{indicator.progress_label}</span>')
        html_parts.append('  </div>')

    # Reconnection status
    if indicator.show_reconnect_status and indicator.reconnect_message:
        html_parts.append(f'  <div class="jenny-reconnect-status">{indicator.reconnect_message}</div>')

    html_parts.append('</div>')

    return '\n'.join(html_parts)


# ============================================================================
# Factory Functions
# ============================================================================

def create_streaming_manager(
    data_source: Optional[DataSourceProtocol] = None,
    max_subscriptions: int = 100,
) -> StreamingManager:
    """
    Create a streaming manager instance.

    Args:
        data_source: Data source implementation (defaults to mock)
        max_subscriptions: Maximum concurrent subscriptions

    Returns:
        Configured StreamingManager
    """
    return StreamingManager(
        data_source=data_source,
        max_subscriptions=max_subscriptions,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Types
    "StreamStatus",
    "UpdateType",
    "StreamUpdate",
    "Subscription",

    # Protocols
    "DataSourceProtocol",
    "MockDataSource",

    # Main class
    "StreamingManager",

    # Factory
    "create_streaming_manager",

    # Streaming Indicators (Task 3.4 - December 2025)
    "StreamingBadge",
    "StreamingIndicator",
    "BADGE_STYLES",
    "get_streaming_indicator",
    "get_streaming_css",
    "render_streaming_badge_html",
]
