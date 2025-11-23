"""
Event Broker - Core Pub/Sub Engine
==================================

**Created**: November 22, 2025
**Purpose**: Core event broker with async dispatch

## Features

- Async event emission and delivery
- Topic-based routing with wildcard support
- Priority-based handler execution
- Circuit breaker pattern for failing subscribers
- Event statistics and monitoring

## Performance

- **Target**: <10ms event propagation
- **Concurrency**: AsyncIO-based parallel delivery
- **Throughput**: 10,000+ events/second
"""

import asyncio
import uuid
from typing import Optional, Callable, Awaitable, Dict, List, Set
from datetime import datetime
import logging

from skills.protocol import SkillEvent, EventType
from skills.event_bus.topic_router import TopicRouter
from skills.event_bus.subscription import (
    SubscriptionManager,
    EventHandler,
    EventFilter,
    SubscriptionPriority
)

logger = logging.getLogger(__name__)


# ============================================================================
# Event Broker
# ============================================================================

class EventBroker:
    """
    Core event broker with async pub/sub.

    Responsibilities:
    - Emit events to subscribers
    - Route events based on topic patterns
    - Manage subscriptions
    - Track event statistics
    """

    def __init__(
        self,
        enable_async_delivery: bool = True,
        max_concurrent_deliveries: int = 100
    ):
        """
        Initialize event broker.

        Args:
            enable_async_delivery: Deliver events asynchronously (non-blocking)
            max_concurrent_deliveries: Max concurrent handler executions
        """
        self.router = TopicRouter()
        self.subscription_manager = SubscriptionManager()

        # Configuration
        self.enable_async_delivery = enable_async_delivery
        self.max_concurrent_deliveries = max_concurrent_deliveries

        # Semaphore for concurrent delivery limit
        self._delivery_semaphore = asyncio.Semaphore(max_concurrent_deliveries)

        # Statistics
        self.stats = {
            'events_emitted': 0,
            'events_delivered': 0,
            'total_subscribers_notified': 0,
            'total_delivery_time_ms': 0.0
        }

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()

        logger.info(
            f"EventBroker initialized (async={enable_async_delivery}, "
            f"max_concurrent={max_concurrent_deliveries})"
        )

    async def emit(
        self,
        event: SkillEvent,
        routing_fn: Optional[Callable[[SkillEvent], str]] = None
    ) -> str:
        """
        Emit event to all subscribers.

        Args:
            event: Event to emit
            routing_fn: Optional function to dynamically compute topic

        Returns:
            Event ID
        """
        import time
        start_time = time.time()

        # Generate event ID if not set
        if not hasattr(event, 'event_id') or not event.event_id:
            event.event_id = str(uuid.uuid4())

        # Set timestamp if not set
        if not event.timestamp:
            event.timestamp = datetime.now().isoformat()

        # Compute topic (use routing function if provided)
        if routing_fn:
            event.topic = routing_fn(event)
        elif not event.topic:
            # Generate topic from event_type and skill_name
            event.topic = f"{event.event_type.value}.{event.skill_name}"

        logger.debug(
            f"Emitting event {event.event_id}: {event.topic} "
            f"(type={event.event_type.value}, skill={event.skill_name})"
        )

        # Find subscribers
        subscriber_ids = self.router.get_subscribers(event.topic)

        if not subscriber_ids:
            logger.debug(f"No subscribers for topic: {event.topic}")
            self.stats['events_emitted'] += 1
            return event.event_id

        logger.debug(
            f"Found {len(subscriber_ids)} subscribers for topic: {event.topic}"
        )

        # Deliver to subscribers
        if self.enable_async_delivery:
            # Async delivery (non-blocking)
            task = asyncio.create_task(
                self._deliver_async(event, subscriber_ids)
            )
            # Track background task
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            # Sync delivery (blocking)
            await self._deliver_sync(event, subscriber_ids)

        # Update stats
        self.stats['events_emitted'] += 1
        delivery_time = (time.time() - start_time) * 1000
        self.stats['total_delivery_time_ms'] += delivery_time

        logger.debug(
            f"Event {event.event_id} dispatched in {delivery_time:.2f}ms "
            f"to {len(subscriber_ids)} subscribers"
        )

        return event.event_id

    async def _deliver_sync(
        self,
        event: SkillEvent,
        subscriber_ids: Set[str]
    ) -> Dict[str, bool]:
        """Deliver event synchronously (blocking)."""
        async with self._delivery_semaphore:
            results = await self.subscription_manager.deliver_event(
                event,
                subscriber_ids
            )

        # Update stats
        self.stats['events_delivered'] += 1
        self.stats['total_subscribers_notified'] += len(subscriber_ids)

        return results

    async def _deliver_async(
        self,
        event: SkillEvent,
        subscriber_ids: Set[str]
    ) -> Dict[str, bool]:
        """Deliver event asynchronously (non-blocking)."""
        try:
            return await self._deliver_sync(event, subscriber_ids)
        except Exception as e:
            logger.error(f"Error in async delivery: {e}", exc_info=True)
            return {}

    async def subscribe(
        self,
        topic: str,
        handler: EventHandler,
        filter_fn: Optional[EventFilter] = None,
        priority: int = SubscriptionPriority.NORMAL.value,
        skill_name: Optional[str] = None
    ) -> str:
        """
        Subscribe to topic pattern.

        Args:
            topic: Topic pattern (supports wildcards: *, **, #)
            handler: Async callback function
            filter_fn: Optional filter predicate
            priority: Handler priority (higher = earlier execution)
            skill_name: Name of subscribing skill

        Returns:
            Subscription ID (for later unsubscribe)
        """
        # Register subscription
        subscription_id = self.subscription_manager.subscribe(
            topic=topic,
            handler=handler,
            filter_fn=filter_fn,
            priority=priority,
            skill_name=skill_name
        )

        # Add to router
        self.router.add_subscription(topic, subscription_id)

        logger.info(
            f"Subscription created: {subscription_id} for topic '{topic}' "
            f"(skill={skill_name}, priority={priority})"
        )

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID from subscribe()

        Returns:
            True if subscription was found and removed
        """
        # Get subscription to find topic
        subscription = self.subscription_manager.subscriptions.get(subscription_id)
        if not subscription:
            logger.warning(f"Subscription not found: {subscription_id}")
            return False

        topic = subscription.topic

        # Remove from subscription manager
        success = self.subscription_manager.unsubscribe(subscription_id)

        # Remove from router
        if success:
            self.router.remove_subscription(topic, subscription_id)

        logger.info(f"Subscription removed: {subscription_id}")

        return success

    async def unsubscribe_all(self, skill_name: str) -> int:
        """
        Unsubscribe all subscriptions for a skill.

        Args:
            skill_name: Name of skill

        Returns:
            Number of subscriptions removed
        """
        subscriptions = self.subscription_manager.get_subscriptions_by_skill(skill_name)
        count = 0

        for subscription in subscriptions:
            success = await self.unsubscribe(subscription.subscription_id)
            if success:
                count += 1

        logger.info(f"Unsubscribed all for skill {skill_name}: {count} subscriptions")

        return count

    def get_stats(self) -> dict:
        """Get broker statistics."""
        # Get router stats
        router_stats = self.router.get_stats()

        # Get subscription manager stats
        subscription_stats = self.subscription_manager.stats

        # Calculate avg delivery time
        avg_delivery_time = (
            self.stats['total_delivery_time_ms'] / self.stats['events_emitted']
            if self.stats['events_emitted'] > 0
            else 0.0
        )

        return {
            'broker': {
                **self.stats,
                'avg_delivery_time_ms': avg_delivery_time,
                'background_tasks': len(self._background_tasks)
            },
            'router': router_stats,
            'subscriptions': subscription_stats
        }

    def get_subscription_stats(self, subscription_id: str) -> Optional[dict]:
        """Get statistics for a subscription."""
        return self.subscription_manager.get_subscription_stats(subscription_id)

    async def wait_for_delivery(self) -> None:
        """Wait for all background deliveries to complete."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            logger.info(f"Waited for {len(self._background_tasks)} background deliveries")

    async def close(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down EventBroker...")

        # Wait for background tasks
        await self.wait_for_delivery()

        # Clear subscriptions
        subscription_ids = list(self.subscription_manager.subscriptions.keys())
        for subscription_id in subscription_ids:
            await self.unsubscribe(subscription_id)

        logger.info("EventBroker shutdown complete")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
