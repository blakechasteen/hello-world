"""
EventBroker Tests
=================

**Created**: November 22, 2025
**Purpose**: Comprehensive tests for event broker

Tests cover:
- Basic pub/sub functionality
- Wildcard subscriptions
- Filtered subscriptions
- Priority handling
- Circuit breaker behavior
- Async delivery
"""

import asyncio
import pytest
from typing import List

# Add parent to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from skills.event_bus import EventBroker, SkillEvent, EventType
from skills.event_bus.subscription import SubscriptionPriority


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def broker():
    """Create event broker for testing."""
    import asyncio
    broker_instance = EventBroker(enable_async_delivery=False)  # Sync for predictable tests

    yield broker_instance

    # Cleanup
    asyncio.get_event_loop().run_until_complete(broker_instance.close())


@pytest.fixture
def async_broker():
    """Create async event broker for testing."""
    import asyncio
    broker_instance = EventBroker(enable_async_delivery=True)

    yield broker_instance

    # Cleanup
    asyncio.get_event_loop().run_until_complete(broker_instance.close())


# ============================================================================
# Basic Pub/Sub Tests
# ============================================================================

@pytest.mark.asyncio
async def test_emit_no_subscribers(broker):
    """Test emitting event with no subscribers."""
    event = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="test.topic",
        skill_name="test_skill",
        timestamp="",
        payload={"data": "test"}
    )

    event_id = await broker.emit(event)

    assert event_id
    assert broker.stats['events_emitted'] == 1


@pytest.mark.asyncio
async def test_subscribe_and_emit(broker):
    """Test basic subscription and event delivery."""
    received_events: List[SkillEvent] = []

    async def handler(event: SkillEvent):
        received_events.append(event)

    # Subscribe
    subscription_id = await broker.subscribe(
        topic="test.topic",
        handler=handler
    )

    # Emit
    event = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="test.topic",
        skill_name="test_skill",
        timestamp="",
        payload={"data": "test"}
    )

    await broker.emit(event)

    # Verify
    assert len(received_events) == 1
    assert received_events[0].payload["data"] == "test"
    assert broker.stats['events_emitted'] == 1


@pytest.mark.asyncio
async def test_multiple_subscribers(broker):
    """Test multiple subscribers to same topic."""
    received_events_1: List[SkillEvent] = []
    received_events_2: List[SkillEvent] = []

    async def handler_1(event: SkillEvent):
        received_events_1.append(event)

    async def handler_2(event: SkillEvent):
        received_events_2.append(event)

    # Subscribe both
    await broker.subscribe(topic="test.topic", handler=handler_1)
    await broker.subscribe(topic="test.topic", handler=handler_2)

    # Emit
    event = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="test.topic",
        skill_name="test_skill",
        timestamp="",
        payload={"data": "test"}
    )

    await broker.emit(event)

    # Verify both received
    assert len(received_events_1) == 1
    assert len(received_events_2) == 1


@pytest.mark.asyncio
async def test_unsubscribe(broker):
    """Test unsubscribing from events."""
    received_events: List[SkillEvent] = []

    async def handler(event: SkillEvent):
        received_events.append(event)

    # Subscribe
    subscription_id = await broker.subscribe(topic="test.topic", handler=handler)

    # Emit first event
    event1 = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="test.topic",
        skill_name="test_skill",
        timestamp="",
        payload={"data": "test1"}
    )
    await broker.emit(event1)

    # Unsubscribe
    success = await broker.unsubscribe(subscription_id)
    assert success

    # Emit second event (should not be received)
    event2 = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="test.topic",
        skill_name="test_skill",
        timestamp="",
        payload={"data": "test2"}
    )
    await broker.emit(event2)

    # Verify only first event received
    assert len(received_events) == 1
    assert received_events[0].payload["data"] == "test1"


# ============================================================================
# Wildcard Subscription Tests
# ============================================================================

@pytest.mark.asyncio
async def test_single_level_wildcard(broker):
    """Test single-level wildcard (*)."""
    received_events: List[SkillEvent] = []

    async def handler(event: SkillEvent):
        received_events.append(event)

    # Subscribe to wildcard pattern
    await broker.subscribe(topic="skill.*.meta", handler=handler)

    # Emit matching event
    event1 = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="skill.started.meta",
        skill_name="test_skill",
        timestamp="",
        payload={}
    )
    await broker.emit(event1)

    # Emit non-matching event
    event2 = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="skill.started.domain",
        skill_name="test_skill",
        timestamp="",
        payload={}
    )
    await broker.emit(event2)

    # Verify only matching event received
    assert len(received_events) == 1
    assert received_events[0].topic == "skill.started.meta"


@pytest.mark.asyncio
async def test_multi_level_wildcard(broker):
    """Test multi-level wildcard (**)."""
    received_events: List[SkillEvent] = []

    async def handler(event: SkillEvent):
        received_events.append(event)

    # Subscribe to multi-level wildcard
    await broker.subscribe(topic="skill.**", handler=handler)

    # Emit various skill events
    topics = [
        "skill.started",
        "skill.started.meta",
        "skill.completed.domain.foo.bar"
    ]

    for topic in topics:
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic=topic,
            skill_name="test_skill",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    # Verify all received
    assert len(received_events) == len(topics)


@pytest.mark.asyncio
async def test_remainder_wildcard(broker):
    """Test remainder wildcard (#)."""
    received_events: List[SkillEvent] = []

    async def handler(event: SkillEvent):
        received_events.append(event)

    # Subscribe to remainder wildcard
    await broker.subscribe(topic="pattern.#", handler=handler)

    # Emit various pattern events
    topics = [
        "pattern.detected",
        "pattern.detected.meta",
        "pattern.detected.meta.foo.bar.baz"
    ]

    for topic in topics:
        event = SkillEvent(
            event_id="",
            event_type=EventType.PATTERN_DETECTED,
            topic=topic,
            skill_name="test_skill",
            timestamp="",
            payload={}
        )
        await broker.emit(event)

    # Verify all received
    assert len(received_events) == len(topics)


# ============================================================================
# Filtered Subscription Tests
# ============================================================================

@pytest.mark.asyncio
async def test_filtered_subscription(broker):
    """Test filtered subscription."""
    received_events: List[SkillEvent] = []

    async def handler(event: SkillEvent):
        received_events.append(event)

    # Filter: only high confidence events
    async def high_confidence_filter(event: SkillEvent) -> bool:
        return event.payload.get('confidence', 0.0) >= 0.9

    # Subscribe with filter
    await broker.subscribe(
        topic="skill.completed.**",
        handler=handler,
        filter_fn=high_confidence_filter
    )

    # Emit high confidence event (should be delivered)
    event1 = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_COMPLETED,
        topic="skill.completed.meta.foo",
        skill_name="test_skill",
        timestamp="",
        payload={"confidence": 0.95}
    )
    await broker.emit(event1)

    # Emit low confidence event (should be filtered)
    event2 = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_COMPLETED,
        topic="skill.completed.meta.bar",
        skill_name="test_skill",
        timestamp="",
        payload={"confidence": 0.5}
    )
    await broker.emit(event2)

    # Verify only high confidence event received
    assert len(received_events) == 1
    assert received_events[0].payload["confidence"] == 0.95


# ============================================================================
# Priority Tests
# ============================================================================

@pytest.mark.asyncio
async def test_priority_ordering(broker):
    """Test handler execution order by priority."""
    execution_order: List[str] = []

    async def low_priority_handler(event: SkillEvent):
        execution_order.append("low")

    async def high_priority_handler(event: SkillEvent):
        execution_order.append("high")

    async def critical_priority_handler(event: SkillEvent):
        execution_order.append("critical")

    # Subscribe with different priorities
    await broker.subscribe(
        topic="test.topic",
        handler=low_priority_handler,
        priority=SubscriptionPriority.LOW.value
    )

    await broker.subscribe(
        topic="test.topic",
        handler=high_priority_handler,
        priority=SubscriptionPriority.HIGH.value
    )

    await broker.subscribe(
        topic="test.topic",
        handler=critical_priority_handler,
        priority=SubscriptionPriority.CRITICAL.value
    )

    # Emit event
    event = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="test.topic",
        skill_name="test_skill",
        timestamp="",
        payload={}
    )
    await broker.emit(event)

    # Verify execution order (highest priority first)
    assert execution_order == ["critical", "high", "low"]


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

@pytest.mark.asyncio
async def test_circuit_breaker(broker):
    """Test circuit breaker opens after consecutive failures."""
    execution_count = [0]

    async def failing_handler(event: SkillEvent):
        execution_count[0] += 1
        raise Exception("Handler error")

    # Subscribe
    subscription_id = await broker.subscribe(
        topic="test.topic",
        handler=failing_handler
    )

    # Emit multiple events to trigger circuit breaker
    for i in range(10):
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="test.topic",
            skill_name="test_skill",
            timestamp="",
            payload={"iteration": i}
        )
        await broker.emit(event)

    # Verify circuit breaker opened (handler stopped being called)
    # Default max_consecutive_failures is 5
    assert execution_count[0] == 5

    # Check subscription stats
    stats = broker.get_subscription_stats(subscription_id)
    assert stats['circuit_breaker_open'] is True


# ============================================================================
# Async Delivery Tests
# ============================================================================

@pytest.mark.asyncio
async def test_async_delivery(async_broker):
    """Test async (non-blocking) event delivery."""
    received_events: List[SkillEvent] = []

    async def slow_handler(event: SkillEvent):
        await asyncio.sleep(0.1)  # Simulate slow processing
        received_events.append(event)

    # Subscribe
    await async_broker.subscribe(topic="test.topic", handler=slow_handler)

    # Emit event
    event = SkillEvent(
        event_id="",
        event_type=EventType.SKILL_STARTED,
        topic="test.topic",
        skill_name="test_skill",
        timestamp="",
        payload={}
    )

    await async_broker.emit(event)

    # emit() should return immediately (async delivery)
    # Event may not be received yet
    assert async_broker.stats['events_emitted'] == 1

    # Wait for delivery
    await async_broker.wait_for_delivery()

    # Now event should be received
    assert len(received_events) == 1


# ============================================================================
# Statistics Tests
# ============================================================================

@pytest.mark.asyncio
async def test_broker_stats(broker):
    """Test broker statistics collection."""
    async def handler(event: SkillEvent):
        pass

    # Subscribe
    await broker.subscribe(topic="test.topic", handler=handler)

    # Emit multiple events
    for i in range(5):
        event = SkillEvent(
            event_id="",
            event_type=EventType.SKILL_STARTED,
            topic="test.topic",
            skill_name="test_skill",
            timestamp="",
            payload={"i": i}
        )
        await broker.emit(event)

    # Get stats
    stats = broker.get_stats()

    assert stats['broker']['events_emitted'] == 5
    assert stats['broker']['events_delivered'] == 5
    assert stats['subscriptions']['total_subscriptions'] >= 1
    assert stats['subscriptions']['active_subscriptions'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
