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
]
