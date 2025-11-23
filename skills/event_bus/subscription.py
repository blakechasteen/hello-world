"""
Subscription Manager
===================

**Created**: November 22, 2025
**Purpose**: Manage event subscriptions with handlers, filters, and priority

## Features

- Handler registration with async callbacks
- Optional filter predicates
- Priority-based execution order
- Subscription lifecycle management
"""

import asyncio
import uuid
from typing import Callable, Awaitable, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import logging

from skills.protocol import SkillEvent

logger = logging.getLogger(__name__)


# ============================================================================
# Types
# ============================================================================

# Handler function type
EventHandler = Callable[[SkillEvent], Awaitable[None]]

# Filter function type (returns True to deliver event)
EventFilter = Callable[[SkillEvent], Awaitable[bool]]


class SubscriptionPriority(Enum):
    """Subscription priority levels."""
    CRITICAL = 100  # Execute first (security, monitoring)
    HIGH = 75       # Important handlers
    NORMAL = 50     # Standard priority
    LOW = 25        # Background processing
    LOWEST = 0      # Cleanup, logging


# ============================================================================
# Subscription
# ============================================================================

@dataclass
class Subscription:
    """Individual event subscription."""
    subscription_id: str
    topic: str
    handler: EventHandler
    filter_fn: Optional[EventFilter] = None
    priority: int = SubscriptionPriority.NORMAL.value
    skill_name: Optional[str] = None  # Subscribing skill

    # Statistics
    events_received: int = 0
    events_filtered: int = 0
    events_processed: int = 0
    events_failed: int = 0
    total_execution_time_ms: float = 0.0

    # Circuit breaker state
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5
    is_open: bool = False  # Circuit breaker open = disabled

    async def should_deliver(self, event: SkillEvent) -> bool:
        """Check if event should be delivered to this subscription."""
        self.events_received += 1

        # Check circuit breaker
        if self.is_open:
            logger.warning(
                f"Circuit breaker open for subscription {self.subscription_id}, "
                f"skipping event"
            )
            return False

        # Apply filter if present
        if self.filter_fn:
            try:
                should_deliver = await self.filter_fn(event)
                if not should_deliver:
                    self.events_filtered += 1
                    return False
            except Exception as e:
                logger.error(
                    f"Filter function error for subscription {self.subscription_id}: {e}"
                )
                self.events_filtered += 1
                return False

        return True

    async def execute(self, event: SkillEvent) -> bool:
        """Execute handler for event. Returns True on success."""
        import time
        start_time = time.time()

        try:
            await self.handler(event)

            # Success - reset circuit breaker
            self.consecutive_failures = 0
            if self.is_open:
                logger.info(f"Circuit breaker closed for subscription {self.subscription_id}")
                self.is_open = False

            self.events_processed += 1

            # Update timing stats
            execution_time = (time.time() - start_time) * 1000
            self.total_execution_time_ms += execution_time

            return True

        except Exception as e:
            logger.error(
                f"Handler error for subscription {self.subscription_id}: {e}",
                exc_info=True
            )

            self.events_failed += 1
            self.consecutive_failures += 1

            # Check circuit breaker threshold
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.error(
                    f"Circuit breaker opened for subscription {self.subscription_id} "
                    f"after {self.consecutive_failures} consecutive failures"
                )
                self.is_open = True

            return False

    def get_stats(self) -> dict:
        """Get subscription statistics."""
        return {
            'subscription_id': self.subscription_id,
            'topic': self.topic,
            'skill_name': self.skill_name,
            'priority': self.priority,
            'events_received': self.events_received,
            'events_filtered': self.events_filtered,
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'avg_execution_time_ms': (
                self.total_execution_time_ms / self.events_processed
                if self.events_processed > 0
                else 0.0
            ),
            'consecutive_failures': self.consecutive_failures,
            'circuit_breaker_open': self.is_open
        }


# ============================================================================
# Subscription Manager
# ============================================================================

class SubscriptionManager:
    """
    Manages event subscriptions.

    Responsibilities:
    - Register/unregister subscriptions
    - Filter events by subscription criteria
    - Execute handlers in priority order
    - Track subscription statistics
    """

    def __init__(self):
        # All subscriptions by ID
        self.subscriptions: Dict[str, Subscription] = {}

        # Subscriptions grouped by topic (for fast lookup)
        self.subscriptions_by_topic: Dict[str, List[str]] = {}

        # Statistics
        self.stats = {
            'total_subscriptions': 0,
            'active_subscriptions': 0,
            'total_events_delivered': 0,
            'total_handler_errors': 0
        }

    def subscribe(
        self,
        topic: str,
        handler: EventHandler,
        filter_fn: Optional[EventFilter] = None,
        priority: int = SubscriptionPriority.NORMAL.value,
        skill_name: Optional[str] = None
    ) -> str:
        """Register a new subscription. Returns subscription ID."""
        subscription_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=subscription_id,
            topic=topic,
            handler=handler,
            filter_fn=filter_fn,
            priority=priority,
            skill_name=skill_name
        )

        self.subscriptions[subscription_id] = subscription

        # Add to topic index
        if topic not in self.subscriptions_by_topic:
            self.subscriptions_by_topic[topic] = []
        self.subscriptions_by_topic[topic].append(subscription_id)

        # Update stats
        self.stats['total_subscriptions'] += 1
        self.stats['active_subscriptions'] = len(self.subscriptions)

        logger.info(
            f"Subscribed: {subscription_id} to topic '{topic}' "
            f"(priority={priority}, skill={skill_name})"
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription. Returns True if found and removed."""
        if subscription_id not in self.subscriptions:
            logger.warning(f"Subscription not found: {subscription_id}")
            return False

        subscription = self.subscriptions[subscription_id]

        # Remove from topic index
        if subscription.topic in self.subscriptions_by_topic:
            self.subscriptions_by_topic[subscription.topic].remove(subscription_id)
            if not self.subscriptions_by_topic[subscription.topic]:
                del self.subscriptions_by_topic[subscription.topic]

        # Remove subscription
        del self.subscriptions[subscription_id]

        # Update stats
        self.stats['active_subscriptions'] = len(self.subscriptions)

        logger.info(f"Unsubscribed: {subscription_id}")
        return True

    def get_subscriptions_for_ids(
        self,
        subscription_ids: set[str]
    ) -> List[Subscription]:
        """Get subscription objects for IDs, sorted by priority."""
        subscriptions = [
            self.subscriptions[sub_id]
            for sub_id in subscription_ids
            if sub_id in self.subscriptions
        ]

        # Sort by priority (highest first)
        subscriptions.sort(key=lambda s: s.priority, reverse=True)

        return subscriptions

    async def deliver_event(
        self,
        event: SkillEvent,
        subscription_ids: set[str]
    ) -> Dict[str, bool]:
        """
        Deliver event to subscriptions.

        Args:
            event: Event to deliver
            subscription_ids: Set of subscription IDs to deliver to

        Returns:
            Dict mapping subscription_id -> success (True/False)
        """
        results = {}

        # Get subscriptions sorted by priority
        subscriptions = self.get_subscriptions_for_ids(subscription_ids)

        # Execute handlers
        for subscription in subscriptions:
            # Check if should deliver
            should_deliver = await subscription.should_deliver(event)

            if should_deliver:
                # Execute handler
                success = await subscription.execute(event)
                results[subscription.subscription_id] = success

                # Update stats
                self.stats['total_events_delivered'] += 1
                if not success:
                    self.stats['total_handler_errors'] += 1
            else:
                results[subscription.subscription_id] = False

        return results

    def get_subscription_stats(self, subscription_id: str) -> Optional[dict]:
        """Get statistics for a subscription."""
        if subscription_id not in self.subscriptions:
            return None

        return self.subscriptions[subscription_id].get_stats()

    def get_all_stats(self) -> dict:
        """Get statistics for all subscriptions."""
        return {
            'manager_stats': self.stats,
            'subscriptions': [
                sub.get_stats()
                for sub in self.subscriptions.values()
            ]
        }

    def get_subscriptions_by_skill(self, skill_name: str) -> List[Subscription]:
        """Get all subscriptions for a skill."""
        return [
            sub for sub in self.subscriptions.values()
            if sub.skill_name == skill_name
        ]

    def close_circuit_breaker(self, subscription_id: str) -> bool:
        """Manually close circuit breaker for a subscription."""
        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]
        subscription.is_open = False
        subscription.consecutive_failures = 0

        logger.info(f"Circuit breaker manually closed for subscription {subscription_id}")
        return True
